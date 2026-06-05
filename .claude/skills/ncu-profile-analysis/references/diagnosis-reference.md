# NCU Diagnosis Reference

A signals → command → heuristic-threshold table for NCU kernel
analysis. Use this when the agent already has an `.ncu-rep` and
wants a fast routing decision before drilling into a single
dimension. Read alongside
[`metrics-and-sections.md`](metrics-and-sections.md) (which
explains what each metric family means) and
[`workflow.md`](workflow.md) (which gives the per-dimension drill
recipes).

**Heuristics, not verdicts.** Every threshold here is a _trigger
to look further_, not a diagnosis. NCU's own rules ship the same
way — they highlight likely problems and motivate an
investigation. Two-report comparisons against your own fixture
are the only way to know whether a number is locally surprising.
Calibrate against your workload before quoting any threshold
from this file.

Assume `R` is a `.ncu-rep` path throughout. The jq recipes are
runnable verbatim against `veloq ncu inspect`, `veloq ncu
metrics`, and `veloq ncu source-metrics` JSON output.

## How to read this document

Each signal section follows the same shape:

1. **What the signal answers** — one-line framing.
2. **VeloQ command** — the jq pipeline that extracts the
   evidence.
3. **Heuristic ranges** — three tiers (_typical_, _noteworthy_,
   _red-flag_) keyed against NCU's percentage-of-peak units and
   stall percentages. The same tier vocabulary is used in
   [`analysis-dimensions.md`](analysis-dimensions.md). VeloQ
   emits raw values; the jq pipelines compute the ratios you'd
   compare against the tiers.
4. **What rules out the signal** — the cheapest check before
   committing to a diagnosis.

The tiers are deliberately broad because per-architecture and
per-workload variation is wide. Treat them as "stop and look"
markers.

## Common signals

### Speed-of-Light classification (sol_class)

Speed-of-Light percentages route the first question: is the
kernel compute-bound, memory-bound, or underutilised?

SOL extraction: see [`analysis-dimensions.md`](analysis-dimensions.md)
§2 (Balance) for the canonical per-launch jq that pulls the
`__throughput.avg.pct_of_peak` percentages.

Classify by the dominant percentage, using the same cutoffs
NCU's shipped SOL rule (`SpeedOfLight.py`) uses to decide
when to fire its own messages:

| sol_class  | Typical pattern                                                           | Trigger to investigate                                             |
| ---------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `compute`  | `sm__throughput` ≥ ~60% AND ≥ memory pct                                  | Inspect instruction mix, pipe utilisation, tensor-core usage       |
| `memory`   | `gpu__compute_memory_throughput` ≥ ~60% AND ≥ compute pct                 | Inspect DRAM/L2/L1 traffic, sectors-per-request, hit-rate          |
| `latency`  | Both `sm__throughput`/memory < ~60% (NCU's `latency_bound_threshold`)     | Inspect occupancy, eligible warps, stall reasons                   |
| `balanced` | Both ≥ ~60% AND within ~10 pts of each other (NCU's `balanced_threshold`) | Treat as compute-bound, but pivot to memory if any stall dominates |

These tiers track NCU's own SOL-rule cutoffs (`latency_bound_threshold = 60`,
`balanced_threshold = 10`, `no_bound_threshold = 80`) so the
classification agrees with the rule message NCU would surface.
They are heuristics, not specification — recalibrate against
your fixture if NCU updates the cutoffs in a later release.

#### What rules it out

- SOL throughput metrics are missing from the report
  (`--set base` capture) → recapture with `--set full` or at
  least `--set roofline`.
- The kernel runs < ~1 µs — most NCU percentage metrics become
  unstable; trust `gpu__time_duration` and the launch headline
  only.

### Memory traffic ratios

Bytes alone don't decide whether memory is the issue; ratios
do. The two most actionable ratios. `ncu metrics --counter`
takes one glob, not a comma-joined list (only `ncu
source-metrics` splits on commas) — fan out one glob at a time
and merge in jq, or use a single broad glob:

```bash
# Sectors per request — coalescing & uniformity probe. One
# broad glob covers both the sector and request halves; the
# jq sums each half separately by counter-name prefix.
veloq ncu metrics R --counter 'l1tex__t_*_pipe_lsu_mem_global_op_*.sum' \
  | jq '
      [.data.rows[] | {launch: .launch_row_id,
                       counter: .counter_name,
                       value: (.value | tonumber? // 0)}]
      | group_by(.launch)
      | map({launch: .[0].launch,
             sectors_per_request:
               (([.[] | select(.counter | test("t_sectors_"))]  | map(.value) | add // 0) as $s
                | ([.[] | select(.counter | test("t_requests_"))] | map(.value) | add // 0) as $r
                | (if $r > 0 then ($s / $r) else null end))})'

# DRAM byte balance — how much DRAM each kernel pulls. Run the
# read and write globs separately, then join in jq.
veloq ncu metrics R --counter 'dram__bytes_read.sum' \
  > dram_read.json
veloq ncu metrics R --counter 'dram__bytes_write.sum' \
  > dram_write.json
jq -n --slurpfile r dram_read.json --slurpfile w dram_write.json '
  ($r[0].data.rows | INDEX(.launch_row_id)) as $R |
  ($w[0].data.rows | INDEX(.launch_row_id)) as $W |
  ($R + $W | keys) as $K |
  $K | map({launch: .,
            dram_read:  ($R[.].value | tonumber? // null),
            dram_write: ($W[.].value | tonumber? // null)})'
```

Heuristic tiers (calibrated against NCU's shipped rules where
they exist; see source notes below each row):

| Ratio                                        | Typical | Noteworthy | Red-flag            |
| -------------------------------------------- | ------- | ---------- | ------------------- |
| Global LSU `sectors / request`               | 1 – ~4  | ~4 – ~10   | ≥ ~10 (uncoalesced) |
| Shared bank conflicts / wavefronts (%)       | < ~10%  | ~10 – ~25% | ≥ ~25%              |
| L1 hit-rate, _only on a memory-bound kernel_ | ≥ ~70%  | ~40 – ~70% | < ~40%              |
| L2 hit-rate, _only on a memory-bound kernel_ | ≥ ~50%  | ~25 – ~50% | < ~25%              |

The bank-conflict row uses the denominator NCU's
`SharedMemoryConflicts.py` rule uses — `conflicts/wavefronts`,
not "% of shared accesses"; that rule fires at ≥ 10%. The L1
/ L2 hit-rate rows are conditional on the kernel being
memory-bound by SOL classification — on a compute-bound kernel
a low L2 hit-rate isn't actionable. The sectors-per-request
tier doesn't correspond to an NCU rule cutoff and was set from
the observation that DRAM-coalesced loads sit at 1–2 sectors
per request on the local fixtures.

Some metric names change across architectures (e.g. shared
bank-conflict counters); see
[`metric-name-arch-notes.md`](metric-name-arch-notes.md) for the
enumeration recipe.

#### What rules it out

- No `l1tex__*` or `dram__*` metrics in the report → recapture
  with the memory section enabled.
- `gpu__time_duration.sum` is dominated by launch overhead
  (kernel < ~5 µs) → DRAM ratios are dominated by warm-up
  noise.

### Stall family dominance

NCU reports per-warp stall reason percentages. The dominant
family decides which dimension to drill.

```bash
veloq ncu inspect R --row-id launch:0 \
  | jq '.data.rows[0]
        | {kernel: .kernel_demangled,
           stalls: (.metrics
                     | map(select(((.name // "") | test("smsp__average_warps_issue_stalled_.*_per_issue_active"; "i"))))
                     | map({name, value: (.value | tonumber? // null), unit})
                     | sort_by(-.value))}'
```

Stall-family interpretations are _consistent with_ the listed
cause, not proof of it. When a family dominates, the next-drill
column is the cheapest place to look for confirming evidence:

| Dominant stall family | Consistent with                    | Next drill                                       |
| --------------------- | ---------------------------------- | ------------------------------------------------ |
| `long_scoreboard`     | Global / L2 memory latency         | Memory traffic ratios + source-line hotspots     |
| `short_scoreboard`    | Shared / texture / surface latency | Shared-mem bank conflicts + source-line hotspots |
| `wait`                | Math pipe latency (FP/INT pipe)    | Instruction mix + pipe utilisation               |
| `mio_throttle`        | Memory I/O queue saturation        | DRAM throughput + sectors/request                |
| `lg_throttle`         | Local/global pipe queue saturation | DRAM/L1 traffic + launch config (occupancy)      |
| `barrier`             | `__syncthreads()` / sync waiting   | Source structure, divergence, async pipelines    |
| `dispatch_stall`      | Scheduler can't issue              | Eligible warps, occupancy, instruction count     |
| `imc_miss`            | Instruction cache miss             | Kernel size, control-flow density, SASS layout   |

Tier guidance for the dominant stall family. NCU's
`CPIStall.py` rule fires when a single stall family ≥ 30% of
issue-active cycles; VeloQ treats the same boundary as
_noteworthy_ and reserves _red-flag_ for the (stricter) ≥ 40%
case where the family is unambiguously dominant:

| `stalled / issue_active`   | Typical | Noteworthy | Red-flag |
| -------------------------- | ------- | ---------- | -------- |
| Any single family          | < ~30%  | ~30 – ~40% | ≥ ~40%   |
| Sum of memory-class stalls | < ~30%  | ~30 – ~50% | ≥ ~50%   |

#### What rules it out

- `Scheduler` / `WarpStateStats` sections are absent → recapture
  with `--set full` (these sections aren't in `--set base`).
- Eligible-warps metric is non-trivial (`smsp__warps_eligible.*`
  consistently ≥ 1) — stall ratios are reported but the
  scheduler is finding work anyway.

### Occupancy gap

Achieved occupancy below theoretical points to runtime barriers,
not just the static launch shape.

```bash
veloq ncu inspect R --row-id launch:0 \
  | jq '.data.rows[0]
        | {kernel: .kernel_demangled,
           theoretical:  (.metrics | map(select(.name == "sm__warps_active.avg.pct_of_peak_sustained_active")) | .[0].value),
           achieved:     (.metrics | map(select(.name == "sm__warps_active.avg.pct_of_peak_sustained_elapsed")) | .[0].value),
           limit_block:  (.metrics | map(select(.name == "launch__occupancy_limit_blocks")) | .[0].value),
           limit_reg:    (.metrics | map(select(.name == "launch__occupancy_limit_registers")) | .[0].value),
           limit_smem:   (.metrics | map(select(.name == "launch__occupancy_limit_shared_mem")) | .[0].value),
           limit_warps:  (.metrics | map(select(.name == "launch__occupancy_limit_warps")) | .[0].value),
           reg_per_thread:  (.metrics | map(select(.name == "launch__registers_per_thread")) | .[0].value),
           smem_per_block:  (.metrics | map(select(.name == "launch__shared_mem_per_block_static")) | .[0].value)}'
```

NCU's `AchievedOccupancy.py` rule fires when the gap exceeds
~10 pts; VeloQ's tier table aligns with that cutoff:

| Gap (`theoretical − achieved`) | Typical  | Noteworthy   | Red-flag  |
| ------------------------------ | -------- | ------------ | --------- |
| Percentage points              | < ~5 pts | ~5 – ~10 pts | ≥ ~10 pts |

A persistent gap with no obvious stall family usually means
synchronisation, divergence, or workload imbalance.

#### What rules it out

- `--set base` capture omits Launch + Occupancy sections; only
  the `launch__*` headline scalars survive. Recapture with at
  least `--set roofline` for occupancy.
- The kernel issues fewer waves than expected (`grid_size /
sm_count < ~1.5`): low achieved occupancy is unavoidable.

### Tensor-core utilisation

When the kernel is _supposed_ to use tensor cores, check that
the tensor pipe carries the load.

```bash
veloq ncu inspect R --row-id launch:0 \
  | jq '.data.rows[0]
        | {kernel: .kernel_demangled,
           tensor: (.metrics
                     | map(select(((.name // "") | test("sm__pipe_tensor.*pct_of_peak"; "i"))))
                     | map({name, value, unit}))}'
```

Tiers track the `HighPipeUtilization` rule's cutoffs
(`low_utilization_threshold = 20`,
`high_utilization_threshold = 60`) — that rule covers the
tensor pipe alongside the other compute pipes, so a tensor
kernel under ~20% peak is the same "underutilised" signal:

| Tensor pipe % of peak | Typical (tensor-core kernel) | Noteworthy | Red-flag                    |
| --------------------- | ---------------------------- | ---------- | --------------------------- |
| `sm__pipe_tensor_*`   | ≥ ~60%                       | ~20 – ~60% | ≤ ~20% (suspected fallback) |

#### What rules it out

- The kernel doesn't use tensor cores by design — confirm via
  `ncu inspect` instruction-mix or by reading the kernel source.
- Mixed-precision kernels often split work; sum the relevant
  tensor pipe metrics before classifying.

### Source-line hotspots

When a stall or memory signal points at one source line, use
`ncu source-metrics` to join per-PC `MetricInstance` data with
DWARF line attribution. The verb requires the report to carry
source counters (`--set source` or `--set full` at capture).

```bash
# Top source lines by bank conflicts (or any source counter).
# Note: ncu source-metrics's --counter glob does accept a
# comma-separated list (unlike ncu metrics).
veloq ncu source-metrics R --row-id launch:0 \
  --counter 'l1tex__data_bank_conflicts*' --by line --limit 5 \
  | jq '.data.rows[] | {file, line, sass_count, counters}'

# Per-SASS drill on a single line for instruction-level evidence.
# jq's tostring is nullary; format the address with a custom
# hex helper.
veloq ncu source-metrics R --row-id launch:0 \
  --counter 'l1tex__data_bank_conflicts*' --by sass --limit 20 \
  | jq '
      def hex:
        if . == 0 then "0"
        else [while(. > 0; . / 16 | floor)
                | . % 16
                | if . < 10 then 48 + . else 87 + . end]
             | reverse | implode
        end;
      .data.rows[] | {addr: ("0x" + (.address | hex)),
                      file: .source.file, line: .source.line,
                      opcode, operands, counters}'
```

Heuristic tier:

| Single source-line share of counter total | Typical | Noteworthy | Red-flag |
| ----------------------------------------- | ------- | ---------- | -------- |
| Top line % of attributed counter sum      | < ~10%  | ~10 – ~25% | ≥ ~25%   |

Concentrated counters on one line are the cheapest source-level
intervention point; spread-out counters often need an
algorithmic change rather than a per-line fix.

#### What rules it out

- `auxiliary.unattributed_sass_counter_totals` carries most of
  the counter sum → the cubin was built without `-lineinfo`;
  recompile or use SASS-axis attribution only.
- `auxiliary.skipped_counters` contains the requested counter
  with reason `not-a-source-counter` → the counter family
  doesn't carry per-PC instances in this report; pick a
  different counter or recapture with the source section
  enabled.

## Two-report compare recipe

The canonical jq pattern for "did this change make things
worse?" comparison. Captures two `.ncu-rep` files A and B with
identical kernels, then diffs every metric by `(launch, counter)`
key.

```bash
veloq ncu metrics A --counter '*' > a.json
veloq ncu metrics B --counter '*' > b.json

jq -n --slurpfile a a.json --slurpfile b b.json '
  ($a[0].data.rows | INDEX(.key)) as $A |
  ($b[0].data.rows | INDEX(.key)) as $B |
  ($A + $B | keys)
  | map(. as $k | {
      key: $k,
      counter: ($A[$k].counter_name // $B[$k].counter_name),
      launch:  ($A[$k].launch_row_id // $B[$k].launch_row_id),
      a:       ($A[$k].value // null),
      b:       ($B[$k].value // null),
      delta:   (((($B[$k].value // 0) | tonumber? // 0)
                  - (($A[$k].value // 0) | tonumber? // 0)))})
  | sort_by(-(.delta | fabs))
  | .[0:30]'
```

Practical notes:

- `--counter "*"` returns long-form (one row per `(launch,
counter)`). The shared `key` field is the join column.
- Numeric `value` fields are raw — the jq `tonumber?` guard
  filters out string-valued metrics (rule messages, label
  strings) without panicking.
- `sort_by(-(.delta | fabs))` ranks biggest absolute movers
  first; use `sort_by(.delta)` to surface the most-negative
  movements (regressions).
- For a relative diff, project `(b - a) / a` instead of `delta`
  — guard against `a == 0` first.

For two reports that _don't_ share kernels (e.g. two different
algorithms), match on `kernel_demangled` first, then sweep
metrics — but be aware that block/grid dimensions, occupancy
limits, and section sets may differ enough that direct counter
comparison is misleading.

## When to leave VeloQ

The diagnosis-reference jq patterns will tell you whether the
report has the evidence you need. They can't:

- Edit the source / launch config / compiler flags. Source
  changes still need a remeasurement.
- Visualise roofline charts, source-pages, or section detail
  panes. NCU GUI is the canonical visual analysis surface.
- Recapture. Use the native `ncu` CLI to grow the section set
  or enable PC sampling — see
  [`limitations.md`](limitations.md).

When a hypothesis can't be confirmed from the present report,
the right move is almost always recapture + reprofile, not a
deeper jq dig.
