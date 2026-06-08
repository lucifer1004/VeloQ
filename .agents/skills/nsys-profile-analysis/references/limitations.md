# Limitations + edge cases

Real boundaries an agent should know about — saves wasted CLI
invocations and protects against silent miscounts.

## When `veloq` is the wrong tool

- **Register pressure, SM occupancy, warp stalls, memory bandwidth
  utilisation** — these are kernel-report signals, not Nsight Systems
  timeline signals. Switch to a `.ncu-rep` report and the separate
  `ncu-profile-analysis` skill.
- **Cross-trace diff (`trace_a` vs `trace_b`)** — on roadmap as
  `veloq compare`, not shipped.
- **Flame graphs / Perfetto-compatible output** — VeloQ emits JSON
  envelopes, not Chrome trace format. Convert downstream if needed.
- **Live profiling** — VeloQ reads exported traces only; it doesn't
  run nsys for you.

## Tool handoff decisions

Use VeloQ while the question can be answered by querying exported
timeline rows. Switch tools when the next diagnostic step needs data
or context VeloQ intentionally does not provide:

| Need                                                                                    | Use                                                                                  | Why                                                                                                                                 |
| --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| Re-capture with different traces, metric cadence, duration, graph mode, or CPU sampling | native `nsys profile`                                                                | VeloQ cannot add missing events or fix buffer drops after capture                                                                   |
| Official NSys reports / canned statistics not modeled by VeloQ yet                      | native `nsys stats`                                                                  | Faster than recreating every built-in report through ad hoc queries                                                                 |
| Convert `.nsys-rep` manually or pick export options                                     | native `nsys export -t parquetdir`                                                   | Capture/export maintenance, not the normal VeloQ analysis path                                                                      |
| Visual overlap, lane ordering, zooming, or screenshots                                  | NSys GUI                                                                             | Timeline visual context is often faster than row output                                                                             |
| Selected kernel needs NCU after timeline triage                                         | `veloq nsys ncu-command T kernel:N --print`, NSys GUI → NCU, or native `ncu` capture | VeloQ can generate a best-effort native `ncu` rerun command from NSys metadata; it does not run NCU or guarantee replay equivalence |
| One-off unsupported raw-table investigation explicitly requested by the user            | DuckDB/PyArrow against a pre-exported `_pqtdir` or manually exported parquetdir      | Last resort before adding a VeloQ command; do not start from generated `.veloq/` sidecars                                           |
| Kernel-internal cause: occupancy, stalls, memory transactions, source lines             | NCU / `ncu-profile-analysis`                                                         | NSys shows timing and causality, not detailed kernel microarchitecture                                                              |
| Source-level fix after evidence is clear                                                | compiler/source tools                                                                | Profilers point to symptoms; code/compiler changes need separate validation                                                         |

Practical rule: if VeloQ can identify the event, row id, NVTX scope,
time window, or missing capability, continue with VeloQ verbs/recipes
for the analysis step. If the next step is "collect different data",
"look visually at the timeline", or "change code and validate", switch
tools.

## Capture-time prerequisites VeloQ can't recover from

| Want                             | Required at capture                                                                                    |
| -------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `metrics --type gpu`             | `nsys profile --gpu-metrics-devices=<…> --gpu-metrics-set=<chip>`                                      |
| `metrics --type cpu-sampling`    | `nsys profile --sample=process-tree` (callchain join needs the same)                                   |
| `metrics --type nic`             | ConnectX-5+ NIC, Linux/SBSA support, OFED userspace incl. `libibmad.so.5`; `--nic-metrics=lf` or `=hf` |
| `--nvtx` scoping / `slices`      | Workload must emit NVTX ranges; capture must include `--trace=…,nvtx`                                  |
| `correlate` runtime-side lookups | `TARGET_INFO_CUDA_CONTEXT_INFO` (present by default on `--trace=cuda`)                                 |
| `stats --type graph`             | `--cuda-graph-trace=graph` (default for inference)                                                     |
| `stats --group-by graph_node`    | `--cuda-graph-trace=node`                                                                              |

For each, `summary.data.auxiliary.capabilities` will show the corresponding
`has_*` bit as `false`. VeloQ surfaces actionable errors
("re-capture with `nsys profile --sample=process-tree`") rather
than empty data — but the recapture step is on the agent.

## NSys-side data quirks

- **`globalTid` is 4-field packed**: `[HW/Host 16b | PID 24b |
Source Domain 8b | TID 16b]`. Native TID is 16 bits, not 24.
  Source-domain `0x00` is OSRT, `0x3B` is CUDA driver; joining
  `PROCESSES.globalPid` to `ThreadNames.globalTid` across domains
  needs the `>> 24` PID-only mask, otherwise you get a constant
  offset. VeloQ's `decode_global_tid` helper handles this.
- **Synthetic correlation id**: raw `correlationId` isn't globally
  unique — it resets per (device, context). VeloQ packs
  `(device, context, raw_corr)` into a 64-bit synthetic id;
  `correlate.synthetic_id` is the rendered form.
- **`CUPTI_ACTIVITY_KIND_RUNTIME.start` can equal `end`** for
  enqueue-only API calls. Treat duration as a lower bound.
- **`NVTX_EVENTS.end` can be `NULL`** for instant markers
  (`nvtxMarkA`). VeloQ surfaces `end_ns: null` and
  `duration_ns: null` on these rows.
- **`CUPTI_ACTIVITY_KIND_CUDA_EVENT` rows have no `end`**:
  `cudaEventRecord` placements are instantaneous. `inspect
cuda_event:N` shows `start_ns` only; VeloQ projects `duration_ns
= 0` in `search` for shape consistency.
- **`[Max depth]` sentinel** in `SAMPLING_CALLCHAINS`: when the
  stack walker ran out of slots, NSys writes a frame with this
  literal symbol name as the deepest entry. `metrics --type
cpu-sampling`'s `truncated_stack_share` counts samples whose
  deepest frame matches this string. To get fuller stacks, raise
  capture-side `--samples-per-backtrace`.

## Time / window quirks

- **`@`-prefixed absolute ns** is the escape hatch for windows that
  start before primary origin (OSRT/NVTX bootstrap region). Most
  agent queries should stay relative.
- **Negative-origin samples**: GPU metrics can have negative
  `timestamp` values (anchored to driver init, before primary
  origin). `metrics --type gpu` includes them; coverage is still
  computed against primary span.
- **`--limit 0` is rejected**: would silently suppress
  `total_matched` and scope totals. Use `--limit 1` for "one row
  plus all totals."

## Per-command quirks

- **`stats --group-by graph`**: only populates `graph_id` field
  on rows where the kernel was inside a captured graph. Eager
  kernels get `graph_id: null` and roll into their own row.
- **`gaps` scopes**: default `--scope device` is cross-stream
  (gap = no stream running GPU work on that device), so an idle
  peer stream does not produce phantom gaps. `--scope stream`
  reverts to per-(device, stream) for starvation diagnostics —
  with overlap on the same stream (rare; CUDA Graphs may do this)
  producing non-positive gaps that `--min-duration` drops.
  `--scope trace` collapses across devices for multi-GPU rig idle;
  it rejects `--device` upfront. `--stream` / `--sort stream` are
  rejected outside `--scope stream`; `--sort device` is rejected
  under `--scope trace`.
- **`timeline.graph_ns`**: in `--cuda-graph-trace=graph` captures,
  this is the ONLY record of kernel work that ran inside captured
  graphs — those kernels do not appear in `kernel_ns`. Treat
  `kernel_ns + memcpy_ns + memset_ns + graph_ns` as the per-bucket
  GPU busy total.
- **`slices --stream` needs a device parent**: stream ids are
  device-local. Use `--device D --stream S` for one lane, or keep
  the query all-device and read the per-(device, stream)
  `gpu_attributed` breakdown for comparison.
- **`metrics --type gpu --bucket` aggregator**: `mean` by default;
  `sum` for `[Cycles Active]` / `[Requests]` tally counters only.
  Other tally-shaped units (`[Bytes]`, `[Instructions Issued]`,
  …) currently mean-aggregate — conservative-first. Consumers can
  override post-hoc via the `agg` field on each bucket row.
- **`metrics --type nic --bucket` aggregator**: `mean`. NSys exports
  NIC rows as rates (`bytes/ms`, `packets/ms`, `ticks/ms`) or
  already-averaged sizes (`bytes`), so summing bucket samples would
  overstate traffic.
- **NIC metric correlation is approximate**: NSys network hardware
  profiling is sampled counter data, not direct communication API
  events. Align it with CUDA/NVTX windows as evidence of pressure,
  then use communication traces or application logs when exact
  message-level attribution matters.
- **`metrics --type cpu-sampling --group-by tid|cpu`**: ignores
  `--name` (numeric keys aren't glob-meaningful). Errors out
  rather than silently no-op'ing.
- **`inspect cpu_sample:N`** uses `COMPOSITE_EVENTS.id`, not the
  implicit per-table row number. On typical exports these align, but
  agents should treat the column-id as authoritative.
- **`nvtx_context` prerequisites**: reverse attribution needs
  `NVTX_EVENTS` + `CUPTI_ACTIVITY_KIND_RUNTIME` +
  `TARGET_INFO_CUDA_CONTEXT_INFO`. When any is missing, rows return
  without `nvtx_context` instead of with the wrong one — the
  context-info bridge is what disambiguates `correlationId` across
  processes.
- **`iter_index` is per `(global_tid, domain_id, name)` bucket**:
  same name on two threads/domains gets independent counters. Use
  `nvtx_context.range_id` for absolute range identity, not just
  ordinal.
- **`ncu-command` is a rerun recipe, not a replay guarantee**: it
  recovers argv/cwd/env from `META_DATA_CAPTURE`, selects the kernel
  with `--kernel-name` + `--launch-skip`, and emits `--launch-count 1`.
  It cannot reproduce non-deterministic control flow, missing input
  files, changed environment, CUDA Graph behavior, or multi-process
  launch ordering exactly. Use `--print` for shell piping; without it
  the command returns the normal JSON envelope.

## Wire-format invariants

- All time fields are **nanoseconds**, signed 64-bit (negative
  values legal for trace prologue / OSRT bootstrap).
- `row_id` is `"<kind>:<sqlite-compatible-rowid>"` end-to-end. No
  bit-packing. Under parquetdir ingestion VeloQ derives the row id
  from DuckDB's per-file row number plus one so v0.1-style IDs keep
  round-tripping through `inspect` / `correlate` verbatim.
- **`schema`** bumps on removal or rename of envelope-shape
  fields; additive evolution stays on the same string. Current
  value is `"v1"`. Every list response uses the canonical
  `data.rows[]` shape with a per-row `key`, search and event
  results share the `EventRef` form, and the envelope carries
  `trace_span`.
- **`source.version`** is per-source and bumps independently on
  any breaking shape change to that source's payloads. Currently
  `"v1"` for NSys and `"v1"` for NCU; source versions are
  independent.
- **Agents should ignore unknown JSON keys**. VeloQ adds fields
  forward-compatibly between schema bumps; consumers that
  hard-fail on unknown break on the next minor release.

## Generated files

Generated products live under one `<trace>.veloq/` artifact root:
they document cache behavior and cleanup only. They are not the
agent-facing analysis API. Do not open generated `.veloq/` files with
DuckDB, PyArrow, pandas, or ad hoc SQL during normal profile analysis.

| File                                       | Built by                                                      | Cost                         |
| ------------------------------------------ | ------------------------------------------------------------- | ---------------------------- |
| `<trace>.veloq/parquetdir/<TABLE>.parquet` | First command on a `.nsys-rep`, or `veloq prep`               | hundreds of MB, multi-second |
| `<trace>.veloq/correlation.bin`            | First `correlate` / `correlation-stats` call                  | KB-MB, sub-second            |
| `<trace>.veloq/meta.bin`                   | First `summary` call, or `veloq prep`                         | few KB, ms                   |
| `<trace>.veloq/nvtx-parent.parquet`        | First NVTX-parent grouped stats path that needs it            | KB-MB, sub-second to seconds |
| `<trace>.veloq/nvtx-tree.parquet`          | First NVTX path grouping or `inspect nvtx:N` hierarchy lookup | KB-MB, sub-second to seconds |

Use `veloq clean T` to remove the artifact root and force rebuild.
For `.nsys-rep` inputs, the parquetdir freshness check follows ctime
ordering on exported table files. The generated `parquetdir/` child
aliases back to the owning `.nsys-rep` if passed to VeloQ. For direct
`_pqtdir/` inputs, the input directory is not removed by `veloq clean`;
derived VeloQ caches live under `<pqtdir>.veloq/` and fingerprint child
parquet files so rewriting one table invalidates dependent caches.
VeloQ holds an advisory flock during Parquet conversion so concurrent
calls on a fresh trace queue rather than corrupt the cache.

## Path constraints

- `.nsys-rep` export requires `nsys >= 2024.6` on `PATH` for the first
  call against a fresh report. Converted output is
  `<trace>.veloq/parquetdir/`, not a private SQLite sidecar. VeloQ
  consumes this cache; agents should still call VeloQ verbs.
- Direct `_pqtdir/` inputs must be real directories ending in
  `_pqtdir` and containing NSys table parquet files.

## Official references

- NVIDIA Nsight Systems User Guide:
  https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- Network hardware and NIC metric profiling:
  https://docs.nvidia.com/nsight-systems/UserGuide/index.html#network-hardware-profiling
