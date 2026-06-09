---
name: pytorch-profile-analysis
description: "Analyze single-file PyTorch/Kineto `.pt.trace.json(.gz)` Chrome traces using the VeloQ CLI. Use for CPU/CUDA/kernel correlation, ProfilerStep/annotation slicing, memory/shape grouping, and single-trace NCCL evidence."
---

# PyTorch Profile Analysis

Use `veloq pytorch` for PyTorch/Kineto Chrome traces:

```bash
veloq pytorch summary T
veloq pytorch search T --type kernel --name-regex 'nccl|gemm' --limit 20
veloq pytorch inspect T kernel:91
veloq pytorch correlate T kernel:91
veloq pytorch slices T --aggregate --group-by step
veloq pytorch collectives T
```

This is not a standalone raw-Kineto runbook. The skill can be installed
separately from the binary, but analysis requires the VeloQ CLI on
`PATH`; if `veloq` is missing, install it before continuing.

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
- Directory inputs are not supported in PyTorch v0. Ask the user to
  choose one trace file if they point at a directory.

## Row IDs

PyTorch row ids use `<kind>:<stable_index>`, where the stable index is
derived from the original `traceEvents` order after non-event flow markers
are skipped. Do not use Kineto `Ev Idx` as a stable key.
Use `veloq pytorch schema <target>` for the authoritative response field
inventory; do not infer the public contract from raw Kineto fields.

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

5. Attribute CPU overhead to Python context when captured:

   Traces exported from `torch.profiler.profile(..., with_stack=True)`
   include `python_function` events. `inspect` returns
   `python_context` / `python_stack`, and `stats` can group CPU work by
   `python-context` or `python-path`:

   ```bash
   veloq pytorch stats T --type cpu-op --group-by python-path,name --limit 20
   veloq pytorch inspect T cpu_op:42
   ```

6. Slice ProfilerStep/user annotation ranges:

   `slices --from/--to` selects ranges that overlap the time window and
   clips attributed GPU/comm time to that window. Slice row `start_ns` and
   `duration_ns` remain the original trace range so the row id still points
   to the inspectable event.

7. For communication questions, stay within one trace file:

   ```bash
   veloq pytorch stats T --type comm --group-by comm-kind,rank
   veloq pytorch search T --type kernel --is-comm --limit 20
   ```

   `collectives` groups single-trace communication evidence and reports
   linked CPU/NCCL row ids. If a trace file contains multiple rank values,
   rank-scoped commands (`search`, `stats`, `timeline`, `slices`, and
   `collectives`) require `--rank <n>` or `--all-ranks`. `inspect` and
   `correlate` operate on explicit row ids and are not rank-scope gated.
   Device ids are rank-local and stream ids are device-local: filter a
   stream with `--rank <n> --device <id> --stream <id>`, or compare lanes
   with `--group-by rank,device,stream`.
   It does not compute cross-rank skew
   in v0:

   ```bash
   veloq pytorch collectives T
   ```

## Event Types

`--type` accepts `cpu-op`, `annotation`, `step`, `runtime`, `driver`,
`kernel`, `memcpy`, `memset`, `memory`, `python`, `comm`, or `all`.

`comm` is a communication-related set. Use `--type kernel --is-comm` to
focus on NCCL kernels.

## Limits

PyTorch support is experimental (`source.version = "v0"`). Classification
is based on Kineto category/name/arg conventions and may need extension
for profiler variants not yet represented by tests. Treat documented
fields, schema targets, row ids/keys, command ids, and output modes as the
versioned source contract even while the source remains `v0`.
