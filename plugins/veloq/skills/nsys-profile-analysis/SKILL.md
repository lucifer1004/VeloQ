---
name: nsys-profile-analysis
description: "Analyze Nsight Systems `.nsys-rep` or `_pqtdir/` timeline traces using the VeloQ CLI. Use for GPU idle gaps, launch causes, CPU/GPU correlation, NVTX, CUDA graphs, metrics, sampling, and overlap/concurrency."
---

# Nsight Systems Profile Analysis

Use `veloq` as the evidence extractor for NSys timeline traces: what ran,
when, how CPU work caused GPU work, where idle gaps are, and whether
captured metric streams are trustworthy. Requires `veloq` on `PATH`.

For Nsight Compute `.ncu-rep` kernel reports use `ncu-profile-analysis`;
for PyTorch/Kineto Chrome traces use `pytorch-profile-analysis`.

## Quickstart

```bash
veloq info trace.nsys-rep   # detect source, trace map, applicable recipes
veloq summary trace.nsys-rep | jq '.data.auxiliary.capabilities'
```

Then query with `stats` / `search` / `inspect` / `correlate` / `slices` /
`gaps` / `timeline` / `concurrency` / `graph-replays` / `metrics` /
`hardware`. Per-command flags and the response schema live in
`veloq <cmd> --help` and `veloq schema <cmd>`; canonical multi-step
workflows live in `veloq recipes` / `veloq recipes <id>`. Those are the
SSOT — do not duplicate them here.

stdout is always one JSON envelope: `data.rows[]` on success (every row
carries a stable `key`), `error` on failure. Parse `.data`/`.error`,
never stderr.

## What veloq gives you over raw tables

Querying the exported sqlite/parquet directly is possible but you
re-implement — and can silently get wrong — things veloq already does:

- **Correlation decode**: runtime↔kernel/memcpy/memset joins through
  process-local `correlationId` bridged by
  `TARGET_INFO_CUDA_CONTEXT_INFO` and the `globalTid` PID mask
  (`correlate`/`inspect` do this for you).
- **NVTX attribution**: forward (range → GPU work) and reverse (kernel →
  enclosing ranges) trees, incl. nesting depth and `--nvtx` scoping.
- **A correlation index + sidecar caches** (`<trace>.veloq/`) reused
  across queries; first `.nsys-rep` access runs `nsys export` for you.
- **Stable row keys** (`kernel:1234`, `gap|pid:..|@..`) — diff two
  captures with `INDEX(.data.rows; .key)` in jq.

If you do read raw tables anyway (user asked, or veloq lacks the query),
you own the invariants in the pitfalls list below — every one of them
has produced a plausible-but-wrong answer in practice.

## Pitfalls that change answers

1. **Capability presence ≠ coverage.** A table can exist yet cover only
   part of the run. Measured case: `GRAPH_TRACE` held 753 replay rows in
   a ~1.7 s window while `cudaGraphLaunch` ran ~4.2k times per rank to
   the end of the trace. Before any CUDA-graph split/hotspot claim,
   cross-check `graph-replays` `total_matched` against the runtime
   `cudaGraphLaunch` count; on mismatch, declare partial coverage and
   conclude only for the covered window.
2. **graph replay kernels may not enter the KERNEL table at all**;
   `graphId` NULL means "eager OR unrecorded replay", never proof of
   eager.
3. **`globalTid` is packed**: PID = `(id >> 24) & 0xFFFFFF`, TID =
   `id & 0xFFFF` (16 bits). TID ≠ PID — do not assume the main thread;
   resolve names via ThreadNames (join on the PID mask).
4. **deviceId/contextId/streamId/correlationId are process-local.** Any
   cross-process join must carry the process dimension, or you collide
   with another rank's rows.
5. **`--from/--to` duration literals are relative to the trace origin**;
   absolute timestamps need `@<ns>`. A wrong window returns
   plausible-looking numbers from the wrong region — or zero rows.
6. **Busy ratios need union, not sum.** Per-stream `sum` double-counts
   overlap and can exceed 100% of wall; use `concurrency`'s
   `union_busy_ns` (and `compute_vs_copy` for copy hiding).
7. **Trust `metrics --type *` only after checking `coverage`**; low
   coverage means recapture, not "the counter was low".

## Evidence discipline

- Cite the command, trace, filters, and time window behind every number
  you report.
- A missing table/capability means "not captured", never "did not
  happen" — probe `summary.data.auxiliary.capabilities` first and say
  INSUFFICIENT_EVIDENCE when the trace cannot answer.
- NSys answers timeline causality and overlap. Kernel-internal questions
  (occupancy, warp stalls, memory transactions, source lines) go to NCU
  (`ncu-profile-analysis`; `veloq nsys ncu-command T kernel:N --print`
  emits the capture command).

## References

- [references/pitfalls.md](references/pitfalls.md) — extended edge
  cases: capture prerequisites, data quirks, window/NVTX semantics
- [references/capabilities.md](references/capabilities.md) — capability
  bit list and metric trust signals
- [references/inspect-shapes.md](references/inspect-shapes.md) —
  machine-generated per-EventKind payload shapes for `inspect`
  (kept in sync with the Rust structs; `veloq schema inspect` is the
  interactive equivalent)
- Official Nsight Systems User Guide →
  https://docs.nvidia.com/nsight-systems/UserGuide/index.html

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/lucifer1004/veloq/main/scripts/install.sh | bash
# or from source: cargo build --release -p veloq
```

`.nsys-rep` inputs need `nsys >= 2024.6` on `PATH` once (parquet
export); `_pqtdir/` inputs are read directly.
