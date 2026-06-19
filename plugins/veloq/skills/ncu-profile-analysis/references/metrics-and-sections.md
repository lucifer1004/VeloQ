# NCU Metrics And Sections

NCU section names and collected metrics depend on the report's section
set. Treat these as families and confirm against the actual
`metric.name` and `metric.label` fields surfaced under `ncu inspect`.
The native model has no section catalog — inspect surfaces the
flat metric + rule arrays, not a `.sections[]` listing — so enumerate
the metric-name prefixes (`__`-delimited) to see what families a
report carries.

## How VeloQ represents metrics

Per-launch metric entries / rules live under `ncu
inspect --row-id launch:<idx>`; cross-launch projection under
`ncu metrics --counter '<glob>'`. The verb matrix is summarised
in [`../SKILL.md`](../SKILL.md).

Each metric entry carries exactly `{name, label, unit, value,
value_type, metric_type, metric_subtype, rollup, instances}` (the
`NativeMetric` shape — confirm with `veloq ncu schema inspect`).
`metric_type` / `metric_subtype` / `rollup` are `ncu_report` enum
_names_ (`"counter"`, `"throughput"`, `"ratio"`, `"pct"`,
`"per_second"`, `"sum"`, …); the raw integer code sits alongside as
`metric_type_code` / `metric_subtype_code` / `rollup_code`.
Values are raw / base values; units come
from the report or are inferred from NCU metric names when the
report omits unit metadata. Do not assume values are auto-scaled.

## Inventory before interpreting

Before using section-specific guidance, list the launches and metric
families present in the actual report:

```bash
# Headlines first — which launches exist, what kernels are they?
veloq ncu launches R | jq '.data.rows[] | {key, kernel: .kernel_demangled, grid: .grid_size, block: .block_size}'

# Then drill into one launch for the metric families it carries.
# The native model has no section catalog; enumerate metric-name
# prefixes (the `__`-delimited head) to see "what's in this report".
veloq ncu inspect R --row-id launch:0 \
  | jq '{kernel: .data.rows[0].kernel_demangled,
         metric_count: (.data.rows[0].metrics | length),
         metric_prefixes: (.data.rows[0].metrics
                            | map(.name | split("__")[0]) | unique)}'
```

If an expected metric family is absent, the correct
diagnosis is usually "report lacks the evidence; recapture/export with
native NCU", not "the issue is absent".

## Section families

| Family                 | Typical question                                                        | Evidence                                                                          |
| ---------------------- | ----------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Speed of Light         | Is the kernel compute-bound, memory-bound, or underutilized?            | Throughput percentages, elapsed duration, compute/memory peak ratios              |
| Roofline               | Is arithmetic intensity aligned with the observed compute/memory limit? | Roofline section metrics, operational intensity, achieved FLOP/s, memory ceilings |
| Launch / Occupancy     | Is resource usage limiting resident work?                               | Block/grid size, registers/thread, shared memory/block, occupancy limit metrics   |
| Memory Workload        | Are memory accesses inefficient or bandwidth-limited?                   | DRAM/L1/L2 bytes, sectors, requests, hit rates, conflicts, replay/excess metrics  |
| Scheduler / Warp State | Are warps eligible and issuing? Why are they stalled?                   | Active/eligible warps, issue active, stall reason metrics                         |
| Instruction Statistics | What instruction classes dominate?                                      | SASS/PTX instruction counts, pipe utilization, branch/load/store/memory op counts |
| Source / PC sampling   | Which code locations carry counters or stalls?                          | Source metrics, SASS addresses, source-index rows, sampled stall metrics          |
| Rules                  | What did NCU's heuristics flag?                                         | Rule messages, focus metrics, estimated speedups                                  |

## Speed of Light / SOL triage

NCU's Speed of Light section is the fastest first classifier. The
official profiling guide describes it as the place to decide whether
the kernel is mainly compute-limited, memory-limited, or
underutilized. In VeloQ, treat it as a routing signal.

SOL extraction: see [`analysis-dimensions.md`](analysis-dimensions.md)
§2 (Balance) for the canonical per-launch and cross-launch jq.

Interpretation pattern:

1. High compute-throughput percentage and lower memory-throughput
   percentage: inspect compute workload, pipe pressure, and
   instruction mix.
2. High memory-throughput percentage and lower compute-throughput
   percentage: inspect DRAM/L2/L1 traffic, bytes moved, coalescing,
   and reuse.
3. Both low: inspect launch/occupancy, eligible warps, stall reasons,
   synchronization, and source/SASS correlation.
4. SOL is a classifier, not proof of a fix. Reprofile after any
   source or launch-configuration change.

Roofline metrics and charts are useful when a visual compute-vs-memory
model is the fastest way to reason about arithmetic intensity. VeloQ
can expose the metrics already present in the report; use native NCU
GUI or native CSV/export when the analysis depends on the chart
itself.

## Metric-name cues

Use metric names for stable filtering with `ncu metrics --counter`:

- `gpu__time_duration.sum`: kernel duration in `ns`.
- `launch__*`: launch configuration, occupancy limits, registers,
  shared memory, block/grid dimensions.
- `sm__*`, `smsp__*`: SM/SMSP compute, instruction, warp, scheduler,
  and stall behavior.
- `dram__*`: device memory traffic and DRAM peak/throughput metrics.
- `lts__*`: L2 traffic, sectors, requests, and hit/miss behavior.
- `l1tex__*`: L1/TEX traffic, sectors, requests, and shared/global
  access behavior.
- `.pct` or `.pct_of_peak_*`: percentages. They are ratios, not time.
- `.per_second`: rate metrics. Read the unit row/field, not just the
  suffix.
- `.per_cycle_*`: per-cycle metrics such as `inst/cycle` or
  `sector/cycle`.

## How to interpret common outcomes

Use these as confirmation rules, not as independent proof. Always
pair an interpretation with the relevant section/rule context and the
kernel's application importance from NSys when available.

### High memory throughput near peak

The kernel is likely memory-bandwidth-bound. VeloQ can show the bytes,
rates, sectors, and percentage-of-peak metrics. Optimization is usually
algorithmic: reduce bytes, improve reuse, use better layout, fuse work,
or change precision.

### Low memory throughput but many sectors/requests

Look for poor coalescing, replay, low hit rate, shared-memory bank
conflicts, or inefficient access patterns. Use `ncu disasm --row-id
launch:<idx>` for source/SASS correlation if the report has enough
source data.

### Low eligible warps / high stall reasons

The next step depends on the dominant stall:

- Memory dependency stalls: pivot to memory workload metrics.
- Barrier/sync stalls: inspect block-level synchronization and
  divergence around barriers.
- Dispatch/math pipe pressure: inspect instruction mix and pipe
  utilization.
- No instruction / instruction fetch: inspect control flow, code
  layout, or generated SASS.

### Occupancy limited by registers/shared memory

VeloQ exposes the limit and resource values. It cannot decide the
tradeoff. Use source/compiler work: launch bounds, register pressure
reduction, shared-memory layout changes, or PTXAS reports. Higher
occupancy is not automatically faster; verify with timing and
throughput metrics.

### Rules disagree with intuition

Do not blindly accept or reject rules. A practical order:

1. Read the rule message and section (from `ncu inspect`).
2. Inspect the focus metric or related metric family.
3. Check whether the estimated speedup is material.
4. Cross-check with kernel duration and application importance from
   NSys if available.
5. Only then propose a source-level optimization.

## Useful jq filters

List rule findings for one launch:

```bash
veloq ncu inspect R --row-id launch:0 \
  | jq '.data.rows[0]
        | {kernel: .kernel_demangled,
           rules: [.rules[]? | {name: .display_name,
                                section: .section_identifier,
                                speedup: .speedup}]}'
```

Rule findings across the whole report (batch-inspect the headline list):

```bash
mapfile -t IDS < <(veloq ncu launches R --limit 1000 | jq -r '.data.rows[].key')
veloq ncu inspect R $(printf -- '--row-id %s ' "${IDS[@]}") \
  | jq '.data.rows[]
        | {kernel: .kernel_demangled,
           rules: [.rules[]? | {name: .display_name, speedup: .speedup}]}'
```

List the metric families (name prefixes) and counts for one launch:

```bash
veloq ncu inspect R --row-id launch:0 \
  | jq '{kernel: .data.rows[0].kernel_demangled,
         metric_count: (.data.rows[0].metrics | length),
         metric_prefixes: (.data.rows[0].metrics
                           | map(.name | split("__")[0]) | unique)}'
```

Cross-launch metric value sweep (e.g. compare DRAM traffic):

```bash
veloq ncu metrics R --counter 'dram__bytes*' \
  | jq '.data.rows[] | {launch: .launch_row_id, counter: .counter_name, value, unit}'
```

The response exposes `data.format`. Default `"long"` rows have
`launch_row_id`, `counter_name`, `value`, and `unit`; `--per-launch`
returns `"per_launch"` rows with `row_id` and a `counters` map.

Wide form (one row per launch, counters as nested map):

```bash
veloq ncu metrics R --counter 'sm__*active*' --per-launch \
  | jq '.data.rows[] | {key, counters}'
```

Diff two reports by `(launch, counter)` key:

```bash
veloq ncu metrics R1 --counter '*' > a.json
veloq ncu metrics R2 --counter '*' > b.json
jq -n --slurpfile a a.json --slurpfile b b.json '
  ($a[0].data.rows | INDEX(.key)) as $A |
  ($b[0].data.rows | INDEX(.key)) as $B |
  ($A + $B | keys)
  | map({key: ., delta: (($B[.].value // 0) - ($A[.].value // 0))})
  | sort_by(-(.delta | fabs))
  | .[0:20]'
```
