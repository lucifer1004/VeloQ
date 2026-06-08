# Time-window + NVTX attribution semantics

Both topics are VeloQ conventions, not anything an agent can infer
from `--help`. They're the load-bearing concepts behind every
windowed or NVTX-scoped query.

## Trace origins — two anchors

VeloQ tracks **two** notions of "trace start":

- **`primary`** = `MIN(start)` over the execution tables
  (kernel / memcpy / memset / runtime / sync). This is the anchor
  every relative `--from <T>` / `--to <T>` flag uses. It's what an
  agent means when they say "the first 3 seconds of the trace."
- **`full`** = `MIN(start)` over **all** event tables, including
  `OSRT_API` and `NVTX_EVENTS`. NSys sometimes anchors OSRT/NVTX
  bootstrap markers to CUDA driver init — hundreds of seconds before
  any GPU work — so `full` can sit deep in negative territory and
  is wrong as a default anchor. Kept for diagnostics only.

The envelope's `trace_span` exposes `primary` (= `origin_ns` /
`span_ns`); the diagnostic `full` span lives under
`summary.data.auxiliary.full_time_range_ns`. Agents should anchor
windows on `primary` (via `trace_span` or relative `--from`/`--to`
which already resolve against it) unless they explicitly want
OSRT/NVTX bootstrap.

## `--from` / `--to` — relative vs absolute

Both flags take a time literal:

- **Relative** to primary origin: `1.2s`, `100ms`, `42ns`, `1500us`.
- **Absolute ns** with `@` prefix: `@-185s` = ns `-185_000_000_000`.
  Useful for pinning at a specific timestamp from a previous
  `inspect` result.

The pair is required: setting only one is an error. Endpoints can
mix (`@-1s` and `2s` together select absolute -1 s through relative
2 s). The window is **half-open**: `[from, to)`.

## Window-overlap semantics across the five windowed commands

`stats` / `search` / `gaps` / `slices` / `timeline` all accept
`--from`/`--to`. They share **overlap inclusion** — any event /
range / gap whose `[start, end]` intersects the window qualifies.

Where they differ:

- **`stats` and `timeline`** additionally **clip** each event's
  duration to the in-window portion before aggregating. So
  `percentage` / `gbps` / per-bucket `total_ns` reflect in-window
  work, not full-event duration. A 5 ms kernel that overlaps the
  window by 1 ms contributes 1 ms.
- **`search` / `gaps` / `slices`** report the **full** bounds of
  the qualifying event, even when only part overlaps the window.
  Useful for "show me everything that touched 1.2-1.5 s" without
  losing context about events that started before or ended after.

## NVTX attribution — the correlation walk

NSys's `NVTX_EVENTS` table records **CPU-side** range timestamps
only. GPU work happens on other tables (kernel / memcpy / memset),
none of which carry an "I happened inside NVTX range X" pointer.

VeloQ derives GPU attribution by walking the CUPTI correlation
graph:

```
NVTX_EVENTS range  →  CUPTI_ACTIVITY_KIND_RUNTIME calls inside the range
                  →  every kernel/memcpy/memset sharing the runtime call's correlationId
```

Two prerequisites must hold:

1. `NVTX_EVENTS` is populated (capture used `--trace=cuda,nvtx`)
2. `TARGET_INFO_CUDA_CONTEXT_INFO` is populated for the (device,
   context) → processId bridge. Without it, attribution still
   succeeds for runtime-only requests (NVTX → runtime walks on
   thread id alone), but GPU-side kinds (kernel / memcpy / memset
   / sync) cannot be disambiguated from their `correlationId` and
   the verb bails up-front rather than producing partial results.

Gate before issuing:

```bash
veloq summary T | jq '.data.auxiliary.capabilities |
  {has_nvtx, has_runtime, has_cuda_contexts}'
```

## `--nvtx <glob>` scope on stats / search

Applies the correlation walk above as an extra `WHERE rowid IN
(<attributed-via-NVTX>)` filter on GPU-event subqueries. The glob
matches NVTX range `name`/`text`; shell-style `*` and `?` wildcards.

```bash
veloq stats T --nvtx '*forward_step*' --group-by demangled
veloq search T --type kernel --nvtx 'iter_172*'
```

The response echoes `nvtx_scope: "<glob>"` so the output is
self-describing. Attribution applies to **kernel / memcpy / memset /
sync / runtime**; osrt / graph / graph_node / cuda_event / overhead /
cpu_sample are not attributable (no CUPTI correlation chain to walk).
With `--type all` + `--nvtx`, non-attributable kinds drop out of the
result set implicitly; explicit `--type <non-attributable>` + `--nvtx`
errors.

## `slices` — the NVTX primary-axis command

`slices` is "for each NVTX range, what was its CPU bounds and what
GPU work was attributed to it." One row per matching range; each
row carries CPU bounds + a `gpu_attributed` list split per
(device, stream).

```bash
veloq slices T --name '*step*' --limit 10
```

Each slice's `attributed_kernel_ns` is the agent-actionable
regression-hunt signal. Compare two iterations' values to find
which one slowed down.

`slices` doesn't take `--device` / `--stream` filters — its primary
axis is NVTX-range, and the response already splits `gpu_attributed`
per (device, stream).

`slices --aggregate` is the SQL-side aggregate view for "which NVTX
scopes own GPU work." It defaults to leaf-name grouping, and path
mode keeps nested same-name scopes distinct:

```bash
veloq slices T --aggregate --name '*' --group-by path \
  --sort path:asc --limit 200
```

Path-mode rows carry `.path` and keys shaped as
`scope|path:<path>`.

## NVTX nesting depth

NVTX ranges nest within a single `(globalTid, domainId)` stack:

```
outer (depth 0)
  inner (depth 1)
    leaf (depth 2)
sibling (depth 0)
```

`search --type nvtx` and `slices` surface `nesting_depth` per row.
Filter `cpu.nesting_depth == 0` to compare only "outermost"
iteration spans, ignoring inner phases:

```bash
veloq slices T --name '*step*' --limit 100 |
  jq '.data.rows[] | select(.cpu.nesting_depth == 0)'
```

Depth is per `(global_tid, domain_id)` — "depth 0 on the trainer
thread" and "depth 0 on a logging thread" are both root spans for
their respective threads, not a global root.

`nesting_depth` on NVTX rows and `nvtx_context.depth` on
kernel/memcpy/memset/sync rows share the same value space — both
come from the per-trace metadata cache's NVTX nesting map, so
cross-row comparisons (e.g. "is this kernel inside
the outer iteration span") work without a separate calibration.

## Time-and-NVTX in `correlate` and `inspect`

`correlate` doesn't use windows or NVTX at all; it walks
correlationId from a single row_id.

`inspect` is mostly context-free — feed it a row_id, get its
details. It always populates `nvtx_context` on
`kernel` / `memcpy` / `memset` / `sync` rows when the trace has
`NVTX_EVENTS` + `CUPTI_ACTIVITY_KIND_RUNTIME` (default-on, no flag
needed). `nvtx_context` carries `range_id`, `name`, `depth`, and
`iter_index` for the innermost NVTX range that was open on the
launching host thread:

```bash
veloq inspect T kernel:1234 |
  jq '.data.rows[0].nvtx_context'
# {"range_id":"nvtx:42","name":"step","depth":1,"iter_index":7}
```

`iter_index` is the 0-based ordinal among same-`(global_tid,
domain_id, name)` repeats — answers "which step is this kernel
in" directly when iterations share a name. For nested NVTX rows,
`inspect nvtx:N` also carries `path`, `parent_row_id`, and
`parent_name` when the NVTX tree sidecar can be built. NVTX rows
themselves carry `nesting_depth` (same value space as
`nvtx_context.depth`); other rows carry their own position fields
(e.g., kernel `start_ns`/`end_ns`).

For GPU work grouped by full hierarchy path, use the stats axis:

```bash
veloq stats T --type kernel --group-by nvtx-path \
  --sort total:desc --limit 20
```

This preserves the existing correlation-based attribution semantics
and only changes the grouping key from innermost range rowid/name to
the slash-joined NVTX path.

For bulk decoration, `search --with-nvtx` runs the same lookup
batched (one SQL per CUPTI kind in the result), so a result set
of N kernels still costs O(1) SQL roundtrips. Off by default;
opt in when the diff target is "which iteration did each slow
kernel belong to":

```bash
veloq search T --type kernel --duration '>1ms' --with-nvtx |
  jq '.data.rows | map(.nvtx_context.name) | group_by(.) | map({name: .[0], count: length})'
```

`--nvtx` (filter) and `--with-nvtx` (decorate) are independent
levers and can be combined.

## NVTX caveats agents should know

- **NVTX is CPU-side**: the range's `start_ns`/`end_ns` come from
  host clocks. GPU work attributed to a range can extend past the
  range's `end_ns` (queued behind CPU but executing later). This is
  why `gpu_attributed` carries its own `start_ns`/`end_ns` per
  stream.
- **NVTX names can be huge**: demangled templates and dynamic
  per-iter strings (`"step 172: 32 gen reqs"`). For pattern
  matching, prefer `--name '*step*'` over `--name 'step 172: ...'`.
- **NVTX domain != thread**: ranges in different domains on the
  same thread don't share a stack. The default domain is `id=0`.
- **Slices may have empty `gpu_attributed`**: an NVTX range may
  bracket CPU-only work (e.g., preprocessing). The slice still
  appears — agents should not assume non-empty attribution.
