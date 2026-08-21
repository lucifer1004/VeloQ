# Pitfalls + edge cases

Real boundaries that cost wrong answers or wasted CLI calls — read
before concluding anything from a trace.

## Graph coverage: presence ≠ completeness

`has_graph_trace: true` proves the `GRAPH_TRACE` table exists, not
that it covers every graph launch. Measured on a TP2PP2 decode trace:
`graph-replays` returned 753 rows inside a 1.7 s window while the
runtime table logged 4183 `cudaGraphLaunch` calls per rank through
the end of the trace. Kernels inside graph replays never land in
`CUPTI_ACTIVITY_KIND_KERNEL` — their `graphId` stays NULL.

- Before any graph share/hotspot answer, cross-check counts:

```bash
veloq graph-replays T | jq '.data.total_matched'
veloq search T --type runtime --name 'cudaGraphLaunch*' --limit 1 | jq '.data.total_matched'
```

- Diverging counts ⇒ declare partial coverage and restrict every
  conclusion to the covered window.
- `graph_id: null` on a kernel row means "eager OR inside an
  unrecorded replay" — never cite it as eager-launch evidence.
- Node-mode captures (`--cuda-graph-trace=node`) are the opposite:
  graph kernels DO land in the kernel table with `graph_id` /
  `graph_node_id` set — answer with `--group-by graph_node`, not
  `--type graph`.

## Thread identity

- `globalTid` packs `[HW/Host 16b | PID 24b | Source Domain 8b | TID
  16b]`. TID is `globalTid & 0xFFFF` — 16 bits, not 24. tid ≠ pid.
- Never assume a "main thread": worker, logging, and CUPTI threads
  all carry equal weight. Resolve thread names first.
- Thread names live in `ThreadNames`; joining to
  `PROCESSES.globalPid` across source domains needs the `>> 24`
  PID-only mask — the domain byte (`0x00` OSRT, `0x3B` CUDA driver)
  otherwise adds a constant offset.

## When `veloq` is the wrong tool

- Kernel-internal questions (occupancy, warp stalls, memory
  transactions, source lines) → NCU / `ncu-profile-analysis`. NSys
  shows timing and causality, not microarchitecture.
- Re-capture with different flags, canned reports, visual zooming →
  native `nsys` / NSys GUI. VeloQ cannot add events missing at
  capture or fix buffer drops. Cross-trace diff → not shipped
  (`veloq compare` on the roadmap).
- NSys → NCU handoff: `veloq nsys ncu-command T kernel:N --print`
  emits a best-effort rerun recipe, not a replay guarantee —
  non-deterministic control flow, changed environment, and CUDA
  Graph behavior are not reproduced.

## Capture-time prerequisites (missing table = not captured)

| Want                             | Required at capture                                                          |
| -------------------------------- | ---------------------------------------------------------------------------- |
| `metrics --type gpu`             | `--gpu-metrics-devices=<…> --gpu-metrics-set=<chip>`                         |
| `metrics --type cpu-sampling`    | `--sample=process-tree` (callchain join needs the same)                      |
| `metrics --type nic`             | ConnectX-5+ NIC, OFED userspace incl. `libibmad.so.5`; `--nic-metrics=lf/hf` |
| `--nvtx` scoping / `slices`      | workload emits NVTX ranges; `--trace=…,nvtx`                                 |
| `correlate` runtime-side lookups | `TARGET_INFO_CUDA_CONTEXT_INFO` (default on `--trace=cuda`)                  |
| `stats --type graph`             | `--cuda-graph-trace=graph`                                                   |
| `stats --group-by graph_node`    | `--cuda-graph-trace=node`                                                    |

The matching `has_*` bit reads `false`; VeloQ errors actionably
("re-capture with …") rather than returning empty data.

## NSys data quirks

- **CUDA identity is process-local**: `deviceId` / `contextId` /
  `streamId` / `correlationId` can repeat in another rank. The
  correlation identity is the lossless `(process, device, context,
  raw_corr)`; `correlate.synthetic_id` renders all four axes.
- **`--device 0` refuses when ambiguous** across processes: use
  `--process <pid> --device 0`, or `--all-devices` for an aggregate.
- **`NVTX_EVENTS` is optional** — probe `has_nvtx` first.
- **Durations can be degenerate**: runtime `start == end` for
  enqueue-only calls (lower bound); `NVTX_EVENTS.end` NULL for
  `nvtxMarkA`; `cuda_event` rows are instantaneous (`search` projects
  `duration_ns = 0`).

## Time windows

- Two origins: `primary` (MIN(start) over execution tables) anchors
  relative `--from`/`--to`; `full` (incl. OSRT/NVTX bootstrap,
  possibly hundreds of seconds earlier) is diagnostics-only under
  `summary.data.auxiliary.full_time_range_ns`.
- `--from`/`--to` are a required pair, half-open `[from, to)`.
  Relative literals (`1.2s`) resolve against primary; `@`-prefixed
  absolute ns (`@-185s`) pins a raw timestamp from e.g. `inspect`.
- All five windowed commands (`stats`/`search`/`gaps`/`slices`/
  `timeline`) include any event intersecting the window. `stats` and
  `timeline` clip durations to the in-window portion before
  aggregating; `search`/`gaps`/`slices` report full event bounds.
- `--limit 0` is rejected (would suppress `total_matched`); use
  `--limit 1` for one row + totals.

## NVTX attribution

NSys records NVTX ranges CPU-side only; GPU tables carry no
back-pointer. VeloQ attributes via the correlation walk:

```text
NVTX range → runtime calls inside it → kernel/memcpy/memset sharing
the runtime call's correlationId
```

- Prerequisites: `NVTX_EVENTS` + `CUPTI_ACTIVITY_KIND_RUNTIME` +
  `TARGET_INFO_CUDA_CONTEXT_INFO` (`has_nvtx` / `has_runtime` /
  `has_cuda_contexts`). Without the context bridge, GPU-side `--nvtx`
  scopes bail up-front rather than mis-attribute.
- `--nvtx <glob>` (stats/search) filters to attributed rows; matches
  range name with `*`/`?`; echoes `nvtx_scope`. Attributable kinds:
  kernel/memcpy/memset/sync/runtime — `--type <non-attributable>` +
  `--nvtx` errors. `--with-nvtx` (search) is the independent batched
  decoration lever; `inspect` populates `nvtx_context` by default.
- `slices` inverts the axis: one row per NVTX range, CPU bounds +
  `gpu_attributed` split per (device, stream);
  `attributed_kernel_ns` is the regression signal. `--stream`
  requires a single device; `--aggregate --group-by path` keeps
  nested same-name scopes distinct.
- Nesting depth is per `(global_tid, domain_id)` — depth 0 is root
  per thread, not a global root. `iter_index` is per
  `(global_tid, domain_id, name)`; use `range_id` for identity.
- Caveats: range timestamps are host-clock (attributed GPU work can
  run past `end_ns`); names can be huge (prefer `--name '*step*'`
  globs); domain ≠ thread; empty `gpu_attributed` is legal.

## Per-command quirks

- `gaps`: default `--scope device` is cross-stream (idle peer streams
  produce no phantom gaps); GPU work = kernel + memcpy + memset +
  graph-trace, so `prev`/`next` may be `kind: graph`. `--scope
  stream` for per-lane starvation; `--scope trace` for multi-GPU rig
  idle (rejects `--device`).
- `timeline.graph_ns`: in graph-mode captures this is the ONLY record
  of kernels inside captured graphs. Per-bucket GPU busy =
  `kernel_ns + memcpy_ns + memset_ns + graph_ns`.
- `metrics --bucket`: `mean` by default (GPU `sum` only for `[Cycles
  Active]` / `[Requests]`; NIC always `mean` — rows are already
  rates/averages). NIC correlation is sampled-counter evidence of
  pressure, not message-level attribution.
- `metrics --type cpu-sampling --group-by tid|cpu` ignores `--name`
  (errors rather than silently no-op'ing). `inspect cpu_sample:N`
  uses `COMPOSITE_EVENTS.id`, not the implicit row number.
- Raw NVTX `event_type` → style: {59,70} → push_pop, {60,71} →
  start_end, else unknown (`stats --type nvtx` derives `nvtx_style`).

## Wire-format invariants

- All time fields are signed int64 nanoseconds (negative legal).
- `row_id` is `"<kind>:<rowid>"` end-to-end; round-trips through
  `inspect` / `correlate` verbatim. List responses use `data.rows[]`
  with a per-row `key`; the envelope carries `trace_span`.
- Ignore unknown JSON keys — VeloQ evolves additively; versions bump
  only on breaking shape changes.

## Generated files + path constraints

- All products live under one `<trace>.veloq/` root (parquetdir,
  `correlation.bin`, `meta.bin`, `nvtx-*.parquet`, figures). Do not
  open them with DuckDB/PyArrow/pandas — cache behavior, not an API.
- `veloq clean T` removes the artifact root only (never the input or
  a direct `_pqtdir/` input); a generated `parquetdir/` child passed
  back aliases to the owning `.nsys-rep`.
- `.nsys-rep` export needs `nsys >= 2024.6` on PATH for the first
  call; direct `_pqtdir/` inputs must be real directories ending in
  `_pqtdir`; concurrent first calls queue on an advisory flock.
