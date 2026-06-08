# Capability gate + trust signals

## Why a capability gate

Most VeloQ calls are conditional on what's actually in the trace.
NSys exports different table sets depending on capture flags
(`--gpu-metrics-devices`, `--sample`, `--cuda-graph-trace=…`, etc.)
and on the NSys version itself. An agent that issues `slices
--name '*step*'` against a trace with no `NVTX_EVENTS` table burns
both wall-clock time and patience.

**Always probe `summary.data.auxiliary.capabilities` first.** It's
cheap (every flag is a `SELECT 1 … LIMIT 0` probe, sub-millisecond),
cached to `<trace>.veloq/meta.bin` on first call, and gives you a boolean
per table VeloQ cares about. (Capabilities live in
`data.auxiliary` next to `full_time_range_ns`; `data.rows[]`
carries the per-table summary.)

```bash
veloq summary T | jq '.data.auxiliary.capabilities'
```

For strict-typed access:

```bash
veloq schema summary | jq '.data.schema.$defs.CapabilityFlags'
```

## Capability flags — what each unlocks

| Flag                      | NSys table                                                                                            | Commands gated                                                                                                                                         |
| ------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------- |
| `has_kernels`             | `CUPTI_ACTIVITY_KIND_KERNEL`                                                                          | `stats --type kernel`, `search --type kernel`, `timeline` (kernel column), `inspect kernel:N`                                                          |
| `has_memcpy`              | `CUPTI_ACTIVITY_KIND_MEMCPY`                                                                          | `stats --type memcpy` + bandwidth analysis, `timeline.memcpy_ns`                                                                                       |
| `has_memset`              | `CUPTI_ACTIVITY_KIND_MEMSET`                                                                          | `stats --type memset`, `timeline.memset_ns`                                                                                                            |
| `has_sync`                | `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION`                                                                 | `stats --type sync` — the "is CPU blocked on GPU?" signal                                                                                              |
| `has_runtime`             | `CUPTI_ACTIVITY_KIND_RUNTIME`                                                                         | `correlate` (runtime API row hop), NVTX→runtime attribution in `slices`, `stats --nvtx`, `search --nvtx`                                               |
| `has_osrt`                | `OSRT_API`                                                                                            | `search --type osrt`, `inspect osrt:N`                                                                                                                 |
| `has_nvtx`                | `NVTX_EVENTS`                                                                                         | `slices` (any), `slices --aggregate --group-by path`, `stats --nvtx`, `stats --group-by nvtx-path`, `search --nvtx`, `inspect nvtx:N` hierarchy fields |
| `has_graph_trace`         | `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`                                                                     | `stats --type graph` — `--cuda-graph-trace=graph` captures only                                                                                        |
| `has_graph_nodes`         | `CUDA_GRAPH_NODE_EVENTS`                                                                              | `stats --group-by graph_node`, `inspect graph_node:N` — `--cuda-graph-trace=node` captures only                                                        |
| `has_graph_events`        | `CUDA_GRAPH_EVENTS`                                                                                   | `search --type graph_event` — host-side graph construction log                                                                                         |
| `has_cuda_event_activity` | `CUPTI_ACTIVITY_KIND_CUDA_EVENT`                                                                      | `search --type cuda_event`, sync ↔ event_sync_id pairing                                                                                               |
| `has_overhead`            | `CUPTI_ACTIVITY_KIND_OVERHEAD`                                                                        | `search --type overhead` — CUPTI's own profiling cost; trust signal                                                                                    |
| `has_cuda_contexts`       | `TARGET_INFO_CUDA_CONTEXT_INFO`                                                                       | `correlate` device/context disambiguation (multi-context processes need this)                                                                          |
| `has_sampling`            | `SAMPLING_CALLCHAINS`                                                                                 | `metrics --type cpu-sampling --group-by symbol\|module\|stack`, `inspect cpu_sample:N` callchain                                                       |
| `has_composite_events`    | `COMPOSITE_EVENTS`                                                                                    | `metrics --type cpu-sampling` (any axis)                                                                                                               |
| `has_sched_events`        | `SCHED_EVENTS`                                                                                        | `metrics --type cpu-sched` per-thread / per-CPU / per-state scheduler breakdown                                                                        |
| `has_gpu_metrics`         | `GPU_METRICS` + `TARGET_INFO_GPU_METRICS` (dictionary)                                                | `metrics --type gpu`                                                                                                                                   |
| `has_nic_metrics`         | `NET_NIC_METRIC` + `TARGET_INFO_NETWORK_METRICS` (dictionary) + `NIC_ID_MAP` + `TARGET_INFO_NIC_INFO` | `metrics --type nic`                                                                                                                                   |
| `has_target_info`         | `TARGET_INFO_SYSTEM_ENV`                                                                              | `hardware` (returns empty `rows[]` otherwise)                                                                                                          |

`has_graph_trace` and `has_graph_nodes` are **mutually exclusive** in
practice — NSys's `--cuda-graph-trace` flag picks one. `has_graph_events`
can coexist with either.

## Schema support — strict 3.x

Today VeloQ ships a single adapter (`v3_standard`) for NSys schema
3.x. Pre-3.x traces fail at `Trace::open` with a clear error rather
than being papered over, so a successful `summary` already implies
canonical column positions and reliable numbers. `summary` does not
expose an `adapter` block — the trace either opened on 3.x or it
didn't.

## Trust signals on `metrics`

The `coverage` block is universal across all four `--type` variants
(`gpu`, `nic`, `cpu-sampling`, `cpu-sched`) and lives under
`.data.auxiliary.common` on every response so the gate is one
navigation step away regardless of source:

| Field           | Meaning                                                                                                                             |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `samples_total` | Primary count for this source's row set after filters: GPU sample-sum, cpu-sampling per-leaf sample count, or cpu-sched event count |
| `covered_ns`    | `metrics_span_ns.1 - metrics_span_ns.0` — **first-to-last span, not gap-aware**                                                     |
| `trace_ns`      | Primary span duration                                                                                                               |
| `ratio`         | `covered_ns / trace_ns` clamped to [0,1]                                                                                            |

**Read `coverage.ratio` before trusting metric values.** nsys's GPU
metric buffer, CPU sample buffer, and SCHED_EVENTS buffer can all
silently drop data on long captures. For example, if GPU metrics cover
only a small slice of a longer trace, the reported `mean` for SMs
Active may describe only that slice rather than the full workload.

A ratio < ~0.9 typically means re-capture is warranted. Practical
mitigations on capture side:

- GPU: lower `--gpu-metrics-frequency` (default cadence is high, fewer
  samples = longer coverage)
- CPU: drop `--cpuctxsw=system-wide` if it's enabled (more events =
  faster buffer fill)
- Always: shorter capture windows or `nsys profile --duration` cap

## CPU-sampling-specific trust signals

`metrics --type cpu-sampling` adds three more:

| Field                   | What it means                                                             | High value (>0.5) implies                                                                                                         |
| ----------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `unresolved_leaf_share` | Fraction of leaf frames with `unresolved=1` (no debug info)               | Most "samples" are unresolved kernel addresses — usually CPU sleeping in syscalls (futex_wait, poll, etc.), not user-code burning |
| `kernel_leaf_share`     | Fraction of leaves in kernel mode (`kernelMode=1`)                        | Most samples caught threads inside the kernel — pair with `unresolved` for the "blocked" vs "syscall-in-progress" distinction     |
| `truncated_stack_share` | Fraction of stacks whose deepest frame is `"[Max depth]"` (nsys sentinel) | Stack walks ran out of slots — raise capture-side `--samples-per-backtrace` for fuller stacks                                     |

**Interpretation pattern**: high `unresolved_leaf_share` +
high `kernel_leaf_share` = "CPU is mostly idle / waiting on the
GPU". Low values + concrete leaf symbols (`blas_thread_server`,
`_PyEval_EvalFrameDefault`, etc.) = "CPU is genuinely doing
work in user-space". The two together let an agent classify the
workload's nature without reading any stack.

## CPU-sched-specific trust signals

`metrics --type cpu-sched` adds two:

| Field                    | What it means                                                     | High value implies                                                                                                                                                    |
| ------------------------ | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `unresolved_state_share` | Fraction of `SCHED_EVENTS` rows with `threadState = Unknown`      | Kernel didn't label sched-out reasons; the `state` axis is unreliable for this capture (use `tid` / `cpu` axes instead)                                               |
| `per_cpu_max_gap_ns`     | Max gap (ns) between consecutive `SCHED_EVENTS` on any single CPU | That CPU's stream stopped logging — sched buffer drops or genuine idle. Compare to `bucket_ns` / `coverage.covered_ns` to judge severity. `null` when <2 events match |

These are on `.data.auxiliary` (next to `.common`) because they're
specific to the cpu-sched source — coverage lives on `.common` so
the universal gate stays one navigation step across every
`--type`, while per-source signals stay where they're scoped.

## When the gate matters most

- **Long traces** (>30 s, especially with `--gpu-metrics-devices=all`):
  `coverage.ratio` will likely be low.
- **Cluster traces** without OFED user-space: `TARGET_INFO_NIC_INFO`
  can be present even when `NET_NIC_METRIC` is absent. In that case
  `hardware` lists NICs, but `summary.auxiliary.capabilities.has_nic_metrics`
  is false and `metrics --type nic` will ask for a recapture with
  `--nic-metrics=lf` or `=hf`.
- **Multi-process traces**: `has_cuda_contexts` is required for
  `correlate` and for `--nvtx`-scoped queries on GPU-side kinds
  (kernel / memcpy / memset / sync) to disambiguate which (dev, ctx)
  a runtime call targeted. Without it the verb bails up-front for
  GPU-side scopes rather than silently mis-attributing; runtime-only
  scopes (`--type runtime --nvtx`) still work because they walk on
  thread id alone.
- **Node-mode CUDA graph captures**: kernels-inside-graphs are in
  `CUPTI_ACTIVITY_KIND_KERNEL` (with `graphId` / `graphNodeId`
  populated), not in `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`. `stats
--type graph` returns nothing; use `--group-by graph` or
  `--group-by graph_node` on `--type kernel` instead.
