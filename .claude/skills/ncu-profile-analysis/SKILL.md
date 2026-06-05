---
name: ncu-profile-analysis
description: "Analyze Nsight Compute kernel reports and kernel-internal performance bottlenecks. Use for `.ncu-rep` files, Nsight Compute / ncu reports, low occupancy, warp stalls, memory throughput, instruction mix, source/SASS/PTX correlation, rule findings, per-source-line metric attribution, or CSV/table export of report metrics. This is a profiling-workflow skill first: classify the kernel bottleneck, then use `veloq ncu` to extract report evidence, and know when to switch to native NCU, NSys, compiler/source inspection, or recapture. Do not use for Nsight Systems timeline traces; use nsys-profile-analysis for `.nsys-rep` files or `_pqtdir/` parquet directories."
---

# Nsight Compute Profile Analysis

Use this skill for Nsight Compute `.ncu-rep` kernel reports and
kernel-internal bottleneck analysis. NCU explains why a kernel is slow
internally; NSys explains what ran when. Use `veloq ncu` as the
evidence extractor for report data, then decide whether the evidence
points to occupancy, memory, scheduler, instruction, launch, or
source-line work.

## Tool Boundary

Use `veloq ncu` verbs as the analysis interface. Do not parse
`<file>.veloq/ncu-native.json.gz`, disasm cache files, or other
`.veloq/` artifacts directly unless the user explicitly asks for raw
artifact inspection or you are developing VeloQ itself.

## Detection

```bash
veloq info path/to/report.ncu-rep
veloq ncu summary --help
```

If `veloq` is missing, install it:

```bash
# Pre-built binary (Linux x86_64/aarch64, macOS x86_64/arm64)
curl -fsSL https://raw.githubusercontent.com/lucifer1004/veloq/main/scripts/install.sh | bash

# Or build from source — clone the repo first, then:
cargo build --release -p veloq
```

## Envelope contract

JSON output is the agent contract. Success:

```json
{
  "schema": "v1",
  "source": { "kind": "ncu", "version": "v1" },
  "command": "ncu.launches",
  "trace": { "kind": "ncu", "path": "..." },
  "data": {
    "count": 50,
    "total_matched": 1234,
    "rows": [
      /* ... */
    ],
    "auxiliary": {
      /* per-verb */
    }
  }
}
```

`trace_span` is omitted on NCU verbs (`.ncu-rep` doesn't carry the
NSys-style primary-execution window). Every list verb returns
canonical `data.rows[]` with a stable per-row `key`:

- `ncu summary` → one totals row (`key: "totals"`)
- `ncu launches` / `inspect` → `key = "launch:<idx>"`
- `ncu disasm` → `key = "kernel|<function_name>"`; requested
  launch id is echoed under `data.auxiliary.row_id`
- `ncu metrics` (long form) → `key = "launch:<idx>|counter:<name>"`
- `ncu metrics --per-launch` → `key = "launch:<idx>"`
- `ncu source-metrics --by line` → `key = "launch:<idx>|line:<file>:<line>"`
- `ncu source-metrics --by sass` → `key = "launch:<idx>|sass:0x<addr>"`
- `ncu source-metrics --by file` → `key = "launch:<idx>|file:<file>"`
- `ncu warp-stalls --by line` → `key = "launch:<idx>|line:<file>:<line>"`
- `ncu warp-stalls --by sass` → `key = "launch:<idx>|sass:0x<addr>"`
- `ncu warp-stalls --by reason` → `key = "launch:<idx>|reason:<stall_reason>"`
- `ncu sources` → `key = "source:<idx>"`
- `ncu ranges` / `graphs` → `key = "<kind>:<idx>"`

`ncu metrics` always keeps the primary list at `data.rows[]`; branch
on `data.format` (`"long"` vs `"per_launch"`) before reading
long-only fields such as `counter_name` or wide-only fields such as
`counters`.

Failure replaces `data` with `error: { message, chain }`. Parse
stdout, not stderr. `veloq ncu schema <target>` returns the
schemars-derived JSON Schema for any verb's response.

## Data Integrity

Treat profiler numbers as evidence, not decoration:

1. Quote the command or report path behind every metric you cite.
2. Do not invent missing metrics. If a section or metric family is
   absent, say the report lacks that evidence and recapture/export
   with native `ncu` if needed.
3. Keep units with values. VeloQ preserves raw/base values; native NCU
   console output may auto-scale values for display.
4. Diagnose first, optimize second. A rule or high metric identifies a
   hypothesis; source changes still need remeasurement.

## Profiling Workflow

NCU is structured as a verb matrix that mirrors NSys's surface. All
trace-reading verbs share a per-report artifact root, `<file>.veloq/`,
built on first call; later calls deserialise cached data instead of
re-walking the report. `ncu summary` returns a slim totals overview;
the full per-launch metric list + rule findings live under `ncu inspect`;
cross-launch metric projection under `ncu metrics`; per-cubin disasm
under `ncu disasm`.

Start by classifying the kernel question:

| Symptom / question                   | Evidence path                                                                                                                                                                                                                                                                             |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| "What's in this report?"             | `ncu summary` → totals + session metadata; `ncu launches` for the per-launch headline list                                                                                                                                                                                                |
| "Why is this launch slow?"           | `ncu launches --kernel '<glob>'` → pick `launch:<idx>` → `ncu inspect --row-id launch:<idx>` for the full metric list + rule findings                                                                                                                                                     |
| Low theoretical/achieved occupancy   | `ncu inspect --row-id launch:N` (full metric list) or `ncu metrics --counter '*occupancy*'` → registers/thread, shared memory/block, block size                                                                                                                                           |
| Memory throughput or poor coalescing | `ncu metrics --counter 'dram__*'` for a cross-launch comparison, or `ncu inspect --row-id launch:N` for the launch's full memory metrics                                                                                                                                                  |
| Warp stalls / scheduler issue        | `ncu warp-stalls --row-id launch:N --by line` for the per-source-line stall-reason histogram (raw `timed_warp_samples`; `--by reason` for the kernel-wide breakdown, `not_issued_samples` for SM-issue stalls). Also `ncu metrics --counter '*stall*'` or `ncu source-metrics --by line`. |
| Instruction mix or pipe pressure     | `ncu metrics --counter 'sm__inst_*'` (cross-launch) or `ncu inspect --row-id launch:N`                                                                                                                                                                                                    |
| Source line suspected                | `ncu disasm --row-id launch:N` for one launch's correlated SASS / PTX / source-index; pair with `ncu inspect` for the offending metric                                                                                                                                                    |
| Cross-launch regression              | `ncu metrics --counter '<glob>'` long-form (one row per `(launch, counter)`) → jq diff between two reports by `key`                                                                                                                                                                       |
| Kernel is slow only in one iteration | Use `nsys-profile-analysis` first to identify `kernel:N`; `veloq nsys ncu-command T kernel:N --print` can generate the native NCU rerun command                                                                                                                                           |
| Need spreadsheet/terminal totals     | `ncu summary --format csv` for a totals + session projection; use `--format table` for terminal review                                                                                                                                                                                    |

For a broad first pass:

1. Inventory: `veloq ncu summary report.ncu-rep` for totals, then
   `veloq ncu launches report.ncu-rep --limit 100` for the
   per-launch headline list. Filter with `--kernel '<glob>'` /
   `--nvtx-range '<glob>'` / `--grid WxHxD` / `--block WxHxD`
   to narrow the candidate set.
2. Pick a culprit launch from the headline list and drill in:
   `veloq ncu inspect report.ncu-rep --row-id launch:<idx>`. The
   inspect response carries the launch's full metric list (each per-PC
   instance placement-tagged) + rule findings; rule findings are NCU's
   fastest bottleneck triage signal, verify against the named metrics.
3. Cross-launch comparison: `veloq ncu metrics report.ncu-rep --counter '<glob>'`
   returns one row per `(launch, counter)` pair — jq-friendly diff
   shape. Compare two reports by `INDEX(.data.rows; .key)`.
4. Source/disasm: `veloq ncu disasm report.ncu-rep --row-id launch:<idx>`
   for SASS / PTX / source-line correlation on that launch's cubin.
   Cached per-cubin under `<file>.veloq/disasm/<sha>.correlated.json`,
   so second+ calls for the same cubin skip nvdisasm / cuobjdump.
5. Per-source-line / per-SASS counter attribution:
   `veloq ncu source-metrics report.ncu-rep --row-id launch:<idx>
--counter '<glob>' --by line|sass|file`. Joins per-PC `MetricInstance`
   values with the DWARF source-line attribution from disasm so
   "which lines have the most bank conflicts?" is one query. Run
   `veloq recipes source-line-hotspots` (or `source-instruction-walk`
   for the per-SASS drill) for the canonical invocation. Requires
   the report to carry source counters (`ncu --set source` or `--set
full` at capture time).
6. If the report lacks needed sections/metrics or source info, stop
   and use native `ncu` recapture/export workflows; see
   [references/limitations.md](references/limitations.md).

Detailed kernel-analysis recipes:
[references/workflow.md](references/workflow.md)

## CSV/table totals and disassembly

For spreadsheet or terminal review, `ncu summary --format
csv|table` renders a totals + session projection (full flag detail
in `veloq ncu summary --help`); JSON emits the full payload. For
per-launch SASS / PTX / source-line correlation, use the dedicated
`ncu disasm --row-id launch:<idx>` — it is the only disasm entry
point, with a per-cubin cache.

All generated products live under the per-report artifact root,
`<file>.veloq/`. These files document cache behavior and cleanup only;
they are not the normal analysis API. Remove them with
`veloq clean <file>` when you need a fresh rebuild:

- `<file>.veloq/ncu-native.json.gz` — the `ncu_report`-native sidecar
  (v1 wire) and the single ingest path: `summary` / `launches` /
  `inspect` / `metrics` / `ranges` / `graphs` / `sources` all read it.
  Built once via the bundled `ncu_report` helper (NCU required at
  build time only); invalidated on a sha256 content-hash of the
  `.ncu-rep` (checkout-stable, so a committed sidecar serves NCU-free).
- `<file>.veloq/disasm/<sha>.correlated.json` — per-cubin
  SASS / PTX / source-line index from nvdisasm + cuobjdump. The cubin
  is extracted from the report (embedded ELF); SHA-keyed so stale
  entries are simply overwritten.

## Where VeloQ Stops

Use `veloq ncu` to read and reshape existing `.ncu-rep` evidence. Turn
to other tools when the next step requires new data or visual context:

- Native `ncu` CLI: capture missing sections/metrics, replay kernels,
  export official CSV pages, or use custom metric sets. When starting
  from an NSys timeline row, `veloq nsys ncu-command T kernel:N --print`
  can generate the initial native NCU command, but it still needs to be
  reviewed and executed as a capture step.
- NCU GUI: inspect complex source pages, guided analysis UI, and
  visual cross-highlighting.
- NSys / `nsys-profile-analysis`: determine whether the kernel matters
  in the timeline, why it was launched late, or whether CPU/GPU overlap
  is the real issue.
- Compiler/source tools: change launch bounds, inspect PTXAS resource
  usage, compare generated SASS, or redesign the algorithm.

## Routing boundary

Use `ncu-profile-analysis` for `.ncu-rep` and kernel-internal metric
questions. Use `nsys-profile-analysis` for `.nsys-rep` / `_pqtdir/`
timeline questions, NVTX ranges, CPU↔GPU launch correlation, GPU idle
bubbles, and PM-counter time series.

## References

- Signal → command → heuristic-threshold table for the first
  routing decision →
  [references/diagnosis-reference.md](references/diagnosis-reference.md)
- Six analysis dimensions (occupancy, balance, stalls, tensor,
  timeline, memory) with per-dimension queries and value tiers →
  [references/analysis-dimensions.md](references/analysis-dimensions.md)
- Enumerate-before-assuming metric names; cross-architecture
  rename pitfalls →
  [references/metric-name-arch-notes.md](references/metric-name-arch-notes.md)
- Headline / per-dimension / ranked-priorities / confidence
  structure for the diagnosis write-up →
  [references/report-template.md](references/report-template.md)
- Bottleneck triage workflows →
  [references/workflow.md](references/workflow.md)
- Common NCU metric families and section interpretation →
  [references/metrics-and-sections.md](references/metrics-and-sections.md)
- VeloQ limits, recapture decisions, and external-tool handoffs →
  [references/limitations.md](references/limitations.md)
- Official NVIDIA Nsight Compute CLI docs →
  https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
- Official NVIDIA Nsight Compute Profiling Guide →
  https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
