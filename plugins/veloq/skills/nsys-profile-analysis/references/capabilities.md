# Capability gate + trust signals

Most verbs are conditional on what's actually in the trace — NSys
exports different table sets per capture flags and version. Always
probe first (sub-millisecond probes, cached to `meta.bin`;
field types via `veloq schema summary`):

```bash
veloq summary T | jq '.data.auxiliary.capabilities'
```

## Capability flags — what each unlocks

| Flag                      | NSys table                                                                                            | Commands gated                                                                                                                                         |
| ------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
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

`has_graph_trace` and `has_graph_nodes` are mutually exclusive
(`--cuda-graph-trace` picks one); `has_graph_events` coexists with
either. `has_graph_trace: true` does NOT imply complete graph
coverage — see `pitfalls.md` "Graph coverage".

## Schema support — strict 3.x

Single adapter (`v3_standard`) for NSys schema 3.x; pre-3.x traces
fail at open with a clear error. A successful `summary` already
implies canonical column positions and reliable numbers.

## Trust signals on `metrics`

Universal `coverage` block under `.data.auxiliary.common` on all four
`--type` variants:

| Field           | Meaning                                                                                                                             |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `samples_total` | Primary count for this source's row set after filters: GPU sample-sum, cpu-sampling per-leaf sample count, or cpu-sched event count |
| `covered_ns`    | `metrics_span_ns.1 - metrics_span_ns.0` — **first-to-last span, not gap-aware**                                                     |
| `trace_ns`      | Primary span duration                                                                                                               |
| `ratio`         | `covered_ns / trace_ns` clamped to [0,1]                                                                                            |

**Read `coverage.ratio` before trusting metric values** — nsys
buffers (GPU metrics, CPU sampling, SCHED_EVENTS) silently drop on
long captures, so a reported mean may describe only the covered
slice. Ratio < ~0.9 ⇒ re-capture: lower `--gpu-metrics-frequency`,
drop `--cpuctxsw=system-wide`, or cap with `--duration`.

## CPU-sampling-specific signals

On `.data.auxiliary`:

| Field                   | What it means                                                             | High value (>0.5) implies                                                                             |
| ----------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `unresolved_leaf_share` | Fraction of leaf frames with `unresolved=1` (no debug info)               | Most samples are unresolved kernel addresses — usually CPU sleeping in syscalls, not user-code burn   |
| `kernel_leaf_share`     | Fraction of leaves in kernel mode (`kernelMode=1`)                        | Threads caught inside the kernel — pair with `unresolved` for "blocked" vs "syscall-in-progress"      |
| `truncated_stack_share` | Fraction of stacks whose deepest frame is `"[Max depth]"` (nsys sentinel) | Stack walks ran out of slots — raise capture-side `--samples-per-backtrace`                           |

High `unresolved_leaf_share` + high `kernel_leaf_share` = CPU mostly
idle/waiting on GPU. Low values + concrete leaf symbols = real
user-space work.

## CPU-sched-specific signals

On `.data.auxiliary` (next to `.common`):

| Field                    | What it means                                                     | High value implies                                                                                                                |
| ------------------------ | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `unresolved_state_share` | Fraction of `SCHED_EVENTS` rows with `threadState = Unknown`      | Kernel didn't label sched-out reasons; the `state` axis is unreliable — use `tid` / `cpu` axes instead                            |
| `per_cpu_max_gap_ns`     | Max gap (ns) between consecutive `SCHED_EVENTS` on any single CPU | That CPU's stream stopped logging (buffer drops or genuine idle); compare to `bucket_ns` / `covered_ns`. `null` when <2 events    |

## When the gate matters most

- Long traces (>30 s, especially `--gpu-metrics-devices=all`):
  expect low `coverage.ratio`.
- Cluster traces without OFED user-space: `hardware` may list NICs
  while `has_nic_metrics` is false — `metrics --type nic` asks for
  recapture with `--nic-metrics=lf|hf`.
- Multi-process traces: `has_cuda_contexts` gates `correlate` and
  GPU-side `--nvtx` scopes; without it verbs bail up-front rather
  than mis-attribute (runtime-only scopes still work).
