# NCU Analysis Dimensions

Six dimensions to evaluate when reading an `.ncu-rep` or `.ncu-repz`. Each
dimension is a structured _question_ with one VeloQ query and
three response tiers — _typical_, _noteworthy_, _red-flag_.
The same tier vocabulary is used in
[`diagnosis-reference.md`](diagnosis-reference.md), which
routes by signal; here we sweep across the kernel.

**Tiers are heuristics, not verdicts.** The numbers are
calibrated against NCU's shipped rule cutoffs
(`SpeedOfLight.py`, `HighPipeUtilization.py`,
`AchievedOccupancy.py`, `SharedMemoryConflicts.py`,
`CPIStall.py`), the NVIDIA profiling guide, and observation on
the fixtures shipped with VeloQ. Confirm against your own
workload before treating any threshold as final.

Assume `R` is an `.ncu-rep` or `.ncu-repz` path and `L = launch:<idx>` a chosen
row id throughout. Substitute your launch when you copy the
queries.

## The six dimensions at a glance

| #   | Dimension               | What it answers                                                  | Primary VeloQ verb                                 |
| --- | ----------------------- | ---------------------------------------------------------------- | -------------------------------------------------- |
| 1   | Occupancy               | Is enough resident work scheduled to hide latency?               | `ncu inspect` (Launch + Occupancy section metrics) |
| 2   | Balance                 | Compute pipe vs memory pipe — which is closer to peak?           | `ncu inspect` (SOL section)                        |
| 3   | Stalls                  | When warps don't issue, why are they stalled?                    | `ncu inspect` (WarpStateStats, Scheduler sections) |
| 4   | Tensor core             | Are tensor-pipe-eligible kernels actually using the tensor pipe? | `ncu inspect` (ComputeWorkloadAnalysis, SOL)       |
| 5   | Timeline (cross-launch) | Which launches dominate; is variance high or stable?             | `ncu metrics --counter '*'` + `gpu__time_duration` |
| 6   | Memory access           | Are memory accesses efficient given the bytes moved?             | `ncu metrics` (`l1tex__*`, `lts__*`, `dram__*`)    |

Always answer dimension 1 (occupancy) before treating dimension
3 (stalls) as a problem — a single warp's stall reasons mean
little when the SM is over-subscribed and the scheduler is
finding eligible work anyway.

## 1. Occupancy

**Question.** Is the launch shape giving the GPU enough work
to keep SMs busy?

```bash
veloq ncu inspect R --row-id L \
  | jq '.data.rows[0]
        | {kernel: .kernel_demangled,
           grid:  .grid_size,
           block: .block_size,
           reg_per_thread:        (.metrics | map(select(.name == "launch__registers_per_thread")) | .[0].value),
           smem_per_block:        (.metrics | map(select(.name == "launch__shared_mem_per_block_static")) | .[0].value),
           theoretical_occupancy: (.metrics | map(select(.name == "sm__warps_active.avg.pct_of_peak_sustained_active")) | .[0].value),
           achieved_occupancy:    (.metrics | map(select(.name == "sm__warps_active.avg.pct_of_peak_sustained_elapsed")) | .[0].value),
           limit_blocks:   (.metrics | map(select(.name == "launch__occupancy_limit_blocks"))   | .[0].value),
           limit_registers:(.metrics | map(select(.name == "launch__occupancy_limit_registers"))| .[0].value),
           limit_smem:     (.metrics | map(select(.name == "launch__occupancy_limit_shared_mem"))| .[0].value),
           limit_warps:    (.metrics | map(select(.name == "launch__occupancy_limit_warps"))    | .[0].value)}'
```

NCU's `AchievedOccupancy.py` fires when the
`(theoretical − achieved)` gap exceeds ~10 pts; VeloQ aligns
with that cutoff. The waves row carries a _persistent-kernel_
caveat: kernels that deliberately run with very few blocks per
SM (persistent loops, GEMM tail tiles) won't match the typical
tier and that's by design.

| Reading                              | Typical | Noteworthy | Red-flag                                              |
| ------------------------------------ | ------- | ---------- | ----------------------------------------------------- |
| `achieved_occupancy` (% of peak)     | ≥ ~50%  | ~20 – ~50% | < ~20%                                                |
| `theoretical − achieved` gap (pts)   | < ~5    | ~5 – ~10   | ≥ ~10                                                 |
| `waves = grid_size_total / sm_count` | ≥ ~5    | ~2 – ~5    | < ~2 (tail-latency risk; persistent kernels excluded) |

What the limit metrics tell you:

- `limit_registers` < `limit_warps` → register pressure caps
  occupancy. Source-level fixes: launch bounds, register
  pressure reduction, or accept the cap.
- `limit_shared_mem` < `limit_warps` → shared memory caps
  occupancy. Reduce per-block allocation or split work.
- `limit_blocks` < `limit_warps` on small-grid kernels →
  occupancy is launch-shape-bound; bigger grid or a different
  decomposition.

Higher occupancy isn't automatically faster. If the kernel is
already memory- or compute-bound, raising occupancy gains
nothing — re-check dimension 2 first.

## 2. Balance (compute pipe vs memory pipe)

**Question.** Which side of the roofline is the kernel close to?

```bash
veloq ncu inspect R --row-id L \
  | jq '.data.rows[0]
        | {kernel: .kernel_demangled,
           compute_pct:     (.metrics | map(select(.name == "sm__throughput.avg.pct_of_peak_sustained_elapsed")) | .[0].value),
           memory_pct:      (.metrics | map(select(.name == "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")) | .[0].value),
           duration_ns:     (.metrics | map(select(.name == "gpu__time_duration.sum")) | .[0].value)}'
```

Pattern table tracks NCU's `SpeedOfLight.py` rule
(`latency_bound_threshold = 60`, `balanced_threshold = 10`,
`no_bound_threshold = 80`) so this dimension classifies the
kernel the same way NCU's own SOL rule would:

| Pattern (`compute_pct`, `memory_pct`) | Class    | What to inspect next                      |
| ------------------------------------- | -------- | ----------------------------------------- |
| compute ≥ ~60% AND compute ≥ memory   | compute  | Instruction mix, pipe utilisation, tensor |
| memory ≥ ~60% AND memory ≥ compute    | memory   | Sectors-per-request, hit-rate, dram bytes |
| both < ~60%                           | latency  | Occupancy, stalls, eligible warps         |
| both ≥ ~60% AND within ~10 pts        | balanced | Treat as compute-bound for first attack   |

Pipe-utilisation tiers follow NCU's `HighPipeUtilization.py`
cutoffs (`low_utilization_threshold = 20`,
`high_utilization_threshold = 60`):

| Reading                            | Typical (well-tuned) | Noteworthy   | Red-flag               |
| ---------------------------------- | -------------------- | ------------ | ---------------------- |
| Dominant pipe % of peak            | ≥ ~60%               | ~20 – ~60%   | < ~20% (underutilised) |
| `\|compute_pct − memory_pct\|` gap | ≥ ~10 pts            | ~5 – ~10 pts | < ~5 pts ("balanced")  |

A "balanced" kernel near peak is often pipe-pressure-limited at
the instruction-mix level — see dimension 3 (stalls) and
dimension 4 (tensor) before claiming the workload is healthy.

## 3. Stalls

**Question.** When warps aren't issuing, what's blocking them?

```bash
veloq ncu inspect R --row-id L \
  | jq '.data.rows[0]
        | {kernel: .kernel_demangled,
           eligible_warps:
             (.metrics | map(select(.name == "smsp__warps_eligible.avg.per_cycle_active")) | .[0].value),
           stalls:
             (.metrics
               | map(select(((.name // "") | test("smsp__average_warps_issue_stalled_.*_per_issue_active"; "i"))))
               | map({reason: (.name | capture("stalled_(?<f>[a-z0-9_]+)_per_issue_active").f),
                      pct:   (.value | tonumber? // null)})
               | sort_by(-.pct))}'
```

NCU's `CPIStall.py` rule fires when a single stall family ≥
~30%. VeloQ treats ≥ 30% as _noteworthy_ and reserves
_red-flag_ for the (stricter) ≥ 40% case where the family is
clearly dominant. `eligible_warps` is the
`smsp__warps_eligible.avg.per_cycle_active` metric — the
average number of eligible warps each cycle the scheduler was
active.

| Reading                                     | Typical | Noteworthy  | Red-flag |
| ------------------------------------------- | ------- | ----------- | -------- |
| Dominant single stall family (%)            | < ~30%  | ~30 – ~40%  | ≥ ~40%   |
| Sum of memory-class stalls (%)              | < ~30%  | ~30 – ~50%  | ≥ ~50%   |
| `smsp__warps_eligible.avg.per_cycle_active` | ≥ ~1.0  | ~0.5 – ~1.0 | < ~0.5   |

Stall-family interpretations are _consistent with_ the listed
cause — when a family dominates, the next-dimension column is
the cheapest place to look for confirming evidence:

| Family                                                  | Pivot to                                    |
| ------------------------------------------------------- | ------------------------------------------- |
| `long_scoreboard` (consistent with global / L2 latency) | Dimension 6 (memory access)                 |
| `short_scoreboard` (shared / texture / surface latency) | Dimension 6 + source-line hotspots          |
| `wait` (math pipe latency)                              | Dimension 4 (tensor) + instruction mix      |
| `mio_throttle` (memory I/O queue saturation)            | Dimension 6 (DRAM throughput)               |
| `lg_throttle` (local/global pipe queue saturation)      | Dimension 6 + dimension 1 (occupancy)       |
| `barrier`                                               | Source: sync, divergence, async pipelines   |
| `dispatch_stall`                                        | Dimension 1 (occupancy) + instruction count |
| `imc_miss` (instruction cache)                          | Kernel size / control flow / SASS layout    |

When `eligible_warps` is healthy (≥ ~1) the scheduler is
finding work despite the reported stalls; treat the dominant
reason as a _cap on peak ILP_, not a primary bottleneck.

## 4. Tensor core

**Question.** For kernels that should use tensor cores — are
they?

```bash
veloq ncu inspect R --row-id L \
  | jq '.data.rows[0]
        | {kernel: .kernel_demangled,
           tensor_pct:    (.metrics | map(select(((.name // "") | test("sm__pipe_tensor.*pct_of_peak_sustained_elapsed"; "i"))))
                            | map({name, value, unit})),
           inst_executed_tensor:
                          (.metrics | map(select(((.name // "") | test("sm__inst_executed_pipe_(hmma|imma|tensor)"; "i"))))
                            | map({name, value, unit}))}'
```

The tensor pipe is one of the pipes covered by NCU's
`HighPipeUtilization.py` rule; the tier cutoffs below borrow
its `low_utilization_threshold = 20` and
`high_utilization_threshold = 60` rather than coming from a
tensor-specific rule:

| Reading                               | Typical (tensor kernel) | Noteworthy | Red-flag |
| ------------------------------------- | ----------------------- | ---------- | -------- |
| Best `sm__pipe_tensor_*` % of peak    | ≥ ~60%                  | ~20 – ~60% | < ~20%   |
| Tensor instruction count / total inst | ≥ ~30%                  | ~5 – ~30%  | < ~5%    |

Common failure modes when the tensor pipe is under-utilised:

- Operand precision / layout doesn't match a tensor-pipe
  variant → cuBLAS / cuDNN falls back to a non-tensor kernel.
  Verify via `kernel_demangled` and the linked library version.
- Kernel was launched with the wrong shape (small `m`, `n`, or
  `k`) for the tensor pipe's natural tile size.
- Mixed-precision kernel where the _expected_ pipe is split
  across multiple counters — sum them before comparing.

If the kernel by design doesn't use tensor cores, skip this
dimension; don't chase a zero.

## 5. Timeline (cross-launch)

**Question.** Which launches dominate this report's wall
time? How much does duration vary between similar launches?

```bash
veloq ncu metrics R --counter 'gpu__time_duration.sum' \
  | jq '
      [.data.rows[] | {launch: .launch_row_id,
                       duration: (.value | tonumber? // null)}]
      | sort_by(-(.duration // 0))
      | {top_by_duration: .[0:10],
         total_duration: (map(.duration // 0) | add),
         count: length}'
```

For variance across launches of the same kernel:

```bash
veloq ncu launches R --limit 1000 \
  | jq '
      .data.rows
      | map({key, kernel: .kernel_demangled})' > launches.json

veloq ncu metrics R --counter 'gpu__time_duration.sum' \
  | jq --slurpfile L launches.json '
      ($L[0] | INDEX(.key)) as $K |
      [.data.rows[] | . + {kernel: ($K[.launch_row_id].kernel // null),
                            duration: (.value | tonumber? // null)}]
      | group_by(.kernel)
      | map({kernel: .[0].kernel,
             n: length,
             min: ([.[].duration] | min),
             p50: ([.[].duration] | sort | .[length / 2 | floor]),
             max: ([.[].duration] | max),
             stdev_over_mean:
               ( ([.[].duration] | (length as $n |
                   if $n > 1 then
                     (add as $s |
                      (map((. - ($s/$n))*((. - ($s/$n)))) | add / ($n - 1)) | sqrt)
                   else 0 end)) /
                 ([.[].duration] | (add / length))) })
      | sort_by(-.max)
      | .[0:20]'
```

Tiers:

| Reading                                 | Typical | Noteworthy  | Red-flag                     |
| --------------------------------------- | ------- | ----------- | ---------------------------- |
| Single kernel's share of total duration | ≤ ~10%  | ~10 – ~30%  | ≥ ~30% (heavy concentration) |
| Per-kernel `stdev/mean` across repeats  | ≤ ~5%   | ~5 – ~20%   | ≥ ~20% (jitter / outliers)   |
| `max / p50` for repeated kernel         | ≤ ~1.1  | ~1.1 – ~1.5 | ≥ ~1.5                       |

A heavily concentrated kernel deserves the next deep drill;
high variance with stable median often points to launch-order
or thermal effects, not the kernel itself — pivot to NSys for
the timeline question.

## 6. Memory access efficiency

**Question.** Given the bytes moved, are accesses well-shaped?

`ncu metrics --counter` takes one glob, not a comma list (only
`ncu source-metrics` splits on commas) — fan out one glob at a
time and merge in jq. The broadest single glob that covers all
four ratios below is `l1tex__t_*_pipe_lsu_mem_global_op_*.sum`
plus separate calls for the L2 / DRAM scalars:

```bash
veloq ncu metrics R --counter 'l1tex__t_*_pipe_lsu_mem_global_op_*.sum' > l1.json
veloq ncu metrics R --counter 'lts__t_sector_hit_rate.pct'              > l2hit.json
veloq ncu metrics R --counter 'dram__bytes_read.sum'                    > dram_r.json
veloq ncu metrics R --counter 'dram__bytes_write.sum'                   > dram_w.json
veloq ncu metrics R --counter 'dram__throughput.avg.pct_of_peak_sustained_elapsed' > dram_pct.json

jq -n \
  --slurpfile l1 l1.json \
  --slurpfile l2 l2hit.json \
  --slurpfile dr dram_r.json \
  --slurpfile dw dram_w.json \
  --slurpfile dp dram_pct.json '
  ($l1[0].data.rows | group_by(.launch_row_id)) as $groups |
  $groups | map(
    .[0].launch_row_id as $launch |
    (([.[] | select(.counter_name | test("t_sectors_"))]  | map(.value | tonumber? // 0) | add) // 0) as $s |
    (([.[] | select(.counter_name | test("t_requests_"))] | map(.value | tonumber? // 0) | add) // 0) as $r |
    (($l2[0].data.rows | map(select(.launch_row_id == $launch)) | .[0].value)             // null) as $l2hit |
    ((($dr[0].data.rows | map(select(.launch_row_id == $launch)) | .[0].value | tonumber? // 0)
      + ($dw[0].data.rows | map(select(.launch_row_id == $launch)) | .[0].value | tonumber? // 0))) as $dram_bytes |
    (($dp[0].data.rows | map(select(.launch_row_id == $launch)) | .[0].value)             // null) as $dram_pct |
    {launch: $launch,
     sectors_per_request: (if $r > 0 then ($s / $r) else null end),
     l2_sector_hit_rate_pct: $l2hit,
     dram_bytes: $dram_bytes,
     dram_pct_of_peak: $dram_pct})'
```

| Reading                             | Typical | Noteworthy | Red-flag                                       |
| ----------------------------------- | ------- | ---------- | ---------------------------------------------- |
| Global LSU sectors per request      | 1 – ~4  | ~4 – ~10   | ≥ ~10                                          |
| L2 sector hit rate, _memory-bound_  | ≥ ~50%  | ~25 – ~50% | < ~25%                                         |
| DRAM throughput % of peak           | ≥ ~70%  | ~30 – ~70% | < ~30% (only on memory-bound; otherwise pivot) |
| Replay metrics share of issued inst | < ~2%   | ~2 – ~5%   | ≥ ~5%                                          |

The L2 hit-rate and DRAM throughput red-flag rows are _only_
meaningful when SOL classification (dimension 2) already
points at the memory pipe. On a compute-bound kernel, a low
DRAM throughput is the natural state — pivot back to dimensions
1, 3, or 4 rather than treating it as a memory red-flag.

Pivot points:

- High sectors/request → uncoalesced or non-power-of-two stride
  access. Inspect with `ncu source-metrics ... --by line` for
  the loop / load that's responsible.
- Low DRAM throughput with high sectors/request → access
  inefficiency, not bandwidth saturation. Source/SASS work.
- Low DRAM throughput with low traffic → likely a _non_-memory
  bottleneck; pivot back to dimensions 1 / 3 / 4.
- Shared-memory bank conflicts → known cross-architecture
  rename. See [`metric-name-arch-notes.md`](metric-name-arch-notes.md)
  for the enumeration.

## Sweeping all six in one pass

For a fast "what does this report look like?" sweep, pipe a
single `ncu inspect` through one jq that pulls the headline
fields from each dimension:

```bash
veloq ncu inspect R --row-id L \
  | jq '.data.rows[0] |
        {kernel:     .kernel_demangled,
         occupancy:  (.metrics | map(select(((.name // "") | test("sm__warps_active.*pct_of_peak_sustained|launch__occupancy_limit_"))))
                       | map({name, value})),
         balance:    (.metrics | map(select(((.name // "") | test("sm__throughput\\.avg\\.pct_of_peak|gpu__compute_memory_throughput.*pct_of_peak"))))
                       | map({name, value})),
         stalls:     (.metrics | map(select(((.name // "") | test("smsp__average_warps_issue_stalled_.*_per_issue_active"))))
                       | map({name, value: (.value | tonumber? // null)})
                       | sort_by(-.value) | .[0:4]),
         tensor:     (.metrics | map(select(((.name // "") | test("sm__pipe_tensor.*pct_of_peak"))))
                       | map({name, value})),
         duration:   (.metrics | map(select(.name == "gpu__time_duration.sum")) | .[0].value),
         rules:      (.rules   | map({name: .display_name, speedup: .speedup}))}'
```

Treat the result as a triage scaffold: route to the dimension
whose tier landed in the "red-flag" column first. If two land
together, default to _memory before compute_ (dimension 6
before dimension 4) — memory fixes are more often
algorithmic and yield bigger wins.

## Limits of this framing

Six dimensions cover most kernel work but miss:

- **Multi-stream / dependency** questions. `ncu` reports a
  single kernel in isolation; cross-kernel ordering is an NSys
  question.
- **Launch overhead vs body**. NCU's measurement window covers
  the kernel body, not the host-side launch path; very-short
  kernels (< ~5 µs) are dominated by launch noise and the
  tiers above don't apply.
- **Driver / runtime** problems (kernel never launched,
  context misconfigured). VeloQ surfaces the launches NCU
  recorded — absent kernels mean an NSys / runtime
  investigation, not a metric inspection.
