---
name: pytorch-profile-analysis
description: "Analyze PyTorch/Kineto profiler Chrome traces. Use for `.pt.trace.json` / `.pt.trace.json.gz` files or directories of per-rank traces, CPU op to CUDA runtime/driver/kernel correlation, ProfilerStep and user annotation slicing, memory/shape grouping, and DDP/NCCL collective skew questions. This is a profiling-workflow skill first: decide the investigation path, then use `veloq pytorch` to extract evidence. Do not use for Nsight Systems `.nsys-rep` traces or Nsight Compute `.ncu-rep` reports."
---

# PyTorch Profile Analysis

Use `veloq pytorch` for PyTorch/Kineto Chrome traces:

```bash
veloq pytorch summary T
veloq pytorch search T --type kernel --name-regex 'nccl|gemm' --limit 20
veloq pytorch inspect T kernel:91
veloq pytorch correlate T kernel:91
veloq pytorch slices T --aggregate --group-by step
veloq pytorch collectives TRACE_DIR
```

## Tool Boundary

Use `veloq pytorch` verbs as the analysis interface. Do not query
`<input>.veloq/pytorch/` sidecars, generated Parquet files, or raw
Kineto trace tables directly with DuckDB, PyArrow, pandas, or ad hoc
SQL unless the user explicitly asks for raw-trace exploration or you
are developing VeloQ itself.

`veloq pytorch prep T` only builds/checks sidecars. After prep, continue
with `summary`, `search`, `inspect`, `stats`, `correlate`, `timeline`,
`slices`, or `collectives`.

## Inputs

- Single trace: `.pt.trace.json` or `.pt.trace.json.gz`.
- Multi-rank trace set: a directory containing per-rank PyTorch trace
  files. VeloQ sorts paths to derive stable row ids.

## Row IDs

PyTorch row ids use `<kind>:<stable_index>`, where the stable index is
derived from sorted trace files plus original `traceEvents` order. Do not
use Kineto `Ev Idx` as a stable key.

Common prefixes:

| Type       | Row id prefix  |
| ---------- | -------------- |
| CPU op     | `cpu_op:N`     |
| Annotation | `annotation:N` |
| Step       | `step:N`       |
| Runtime    | `runtime:N`    |
| Driver     | `driver:N`     |
| Kernel     | `kernel:N`     |
| Memcpy     | `memcpy:N`     |
| Memset     | `memset:N`     |
| Memory     | `memory:N`     |
| Python     | `python:N`     |
| Comm       | `comm:N`       |

## Workflow

1. Inventory first:

   ```bash
   veloq pytorch summary T
   ```

   Read `data.auxiliary.capabilities` before choosing a path.

2. Find events:

   ```bash
   veloq pytorch search T --type cpu-op --name '*aten::*' --limit 20
   veloq pytorch search T --type kernel --is-comm --limit 20
   ```

3. Drill into one event:

   ```bash
   veloq pytorch inspect T ROW_ID
   ```

   Inspect returns raw args, typed args, parent/children, enclosing step,
   and link metadata.

4. Answer launch-cause questions:

   ```bash
   veloq pytorch correlate T kernel:91
   ```

   Read `data.rows[0].events[]` for the CPU op, annotation/step,
   runtime/driver, and GPU activity chain.

5. For multi-rank directories, be explicit about rank scope:

   ```bash
   veloq pytorch stats TRACE_DIR --rank 0 --type kernel --group-by name
   veloq pytorch stats TRACE_DIR --all-ranks --type comm --group-by comm-kind,rank
   ```

   `collectives` is explicitly cross-rank:

   ```bash
   veloq pytorch collectives TRACE_DIR
   ```

## Event Types

`--type` accepts `cpu-op`, `annotation`, `step`, `runtime`, `driver`,
`kernel`, `memcpy`, `memset`, `memory`, `python`, `comm`, or `all`.

`comm` is a communication-related set. Use `--type kernel --is-comm` to
focus on NCCL kernels.

## Limits

PyTorch support is experimental (`source.version = "v0"`). Classification
is based on Kineto category/name/arg conventions and may need extension
for profiler variants not yet represented by tests.
