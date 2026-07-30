# NCU Metric-Name Architecture Notes

Metric names in `.ncu-rep` and `.ncu-repz` reports drift across GPU
architectures and NCU versions. Counters get renamed, split,
or moved between sections, and sections themselves come and go
between releases. **This file is not a translation table.** It
documents how to _enumerate_ the names a report actually
carries, and lists the renames that historically caused jq
filters to silently match zero rows.

The VeloQ philosophy is read-only: every command reads exactly
what NCU wrote in the report. VeloQ does not normalise metric
names across architectures because there is no canonical
mapping that survives all reports — the best VeloQ can do is
get out of the way and let you enumerate first.

Assume `R` is an `.ncu-rep` or `.ncu-repz` path throughout.

## Enumerate before assuming a name exists

Before quoting any metric in a jq filter, list what the report
carries:

```bash
# Every metric name on one launch:
veloq ncu inspect R --row-id launch:0 \
  | jq -r '.data.rows[0].metrics[].name // empty' \
  | sort -u

# Metric-prefix inventory — the native model has no section
# catalog, so enumerate by the metric-name prefix (the text
# before `__`) instead:
veloq ncu inspect R --row-id launch:0 \
  | jq '.data.rows[0]
        | {kernel: .kernel_demangled,
           metric_count: (.metrics|length),
           metric_prefixes: ([.metrics[]?.name // empty | split("__")[0]] | unique)}'

# Grep the cross-launch sweep — useful when only some launches
# carry the metric (e.g. cuDNN heuristic kernels):
veloq ncu metrics R --counter '*' \
  | jq -r '.data.rows[].counter_name' \
  | sort -u
```

For a quick "does this report have any counter matching `<glob>`"
check:

```bash
veloq ncu metrics R --counter '<glob>' \
  | jq '{matched_counters: (.data.rows | map(.counter_name) | unique),
         total_matched: .data.total_matched}'
```

`total_matched == 0` means no counter in the report's section
set matches; this is the cheapest evidence that the metric
simply isn't captured.

For per-PC counters (source-metrics input), the source-counter
detection + rollup gate requires a per-instance correlation type
of `uint64` plus PCs inside the cubin range. Failed counters land in
`auxiliary.skipped_counters` with a reason:

```bash
veloq ncu source-metrics R --row-id launch:0 --counter '<glob>' --by line \
  | jq '{matched: .data.auxiliary.matched_counters,
         skipped: .data.auxiliary.skipped_counters,
         warnings: .data.auxiliary.warnings}'
```

If your favourite counter is in `skipped` with
`not-a-source-counter`, the report's section set didn't enable
per-PC sampling for it — recapture with the right NCU section.

## Known cross-architecture renames

These pairs caused user-visible bugs in earlier VeloQ users.
The list is **not exhaustive** — enumerate. Inclusion here
just means "we've been bitten by this rename in the wild";
absence doesn't mean a name is stable.

### Section set

| Family           | Older NCU / arch          | Newer NCU / arch                           |
| ---------------- | ------------------------- | ------------------------------------------ |
| Speed of Light   | `SpeedOfLight`            | `SpeedOfLight_RooflineChart` (split)       |
| Memory workload  | `MemoryWorkloadAnalysis`  | `MemoryWorkloadAnalysis_Chart` (split)     |
| Source counters  | `SourceCounters`          | unchanged; _body items_ added in newer NCU |
| Warp state       | `SchedulerStats`          | `WarpStateStats` (renamed)                 |
| Compute workload | `ComputeWorkloadAnalysis` | unchanged                                  |

Rule-of-thumb: identifiers without a trailing suffix (e.g.
`SpeedOfLight`) are older; the split / suffixed variants
(`*_RooflineChart`, `*_Chart`) appear in newer NCU. List
`section.identifier` before trusting a hard-coded name.

### DRAM bytes

Some report sets expose a single `dram__bytes.sum`; others
expose `dram__bytes_read.sum` + `dram__bytes_write.sum` only.
Always sum both halves before quoting "DRAM bytes" and guard
each half against `null` (an empty `map` followed by `add`
yields `null`, which then errors in arithmetic):

```bash
veloq ncu metrics R --counter 'dram__bytes*' \
  | jq '
      [.data.rows[] | {launch: .launch_row_id,
                        counter: .counter_name,
                        value:   (.value | tonumber? // 0)}]
      | group_by(.launch)
      | map({launch: .[0].launch,
             dram_total_bytes:
               ( ([.[] | select(.counter == "dram__bytes.sum")] | map(.value) | add) //
                  ( (([.[] | select(.counter == "dram__bytes_read.sum")]  | map(.value) | add) // 0) +
                    (([.[] | select(.counter == "dram__bytes_write.sum")] | map(.value) | add) // 0) ) )})'
```

### Shared-memory bank conflicts

The counter family has different names depending on which
section is collected:

| Capture path               | Counter family                                              |
| -------------------------- | ----------------------------------------------------------- |
| Memory section             | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_*.sum`   |
| Roofline / lighter capture | `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_*.sum`       |
| PC sampling (`--set full`) | per-PC counter; surfaced via `ncu source-metrics --by sass` |

Enumerate the actual names with two passes (one glob per
call; `ncu metrics --counter` is a single glob, not a comma
list):

```bash
veloq ncu metrics R --counter 'l1tex__*conflict*'
veloq ncu metrics R --counter 'l1tex__*shared*'
```

before quoting one.

### Global memory sectors / requests

`l1tex__t_sectors_pipe_lsu_mem_global_op_*` and
`l1tex__t_requests_pipe_lsu_mem_global_op_*` are the modern
names. Older NCU sets used `smsp__sass_inst_executed_op_global_*`
plus separate sector counters; the modern names are part of the
unified L1TEX hierarchy.

For "sectors per request" — the coalescing probe — use a
single broad glob that covers both halves (`ncu metrics
--counter` is a single glob, not a comma-list; the verb that
splits commas is `ncu source-metrics`):

```bash
veloq ncu metrics R --counter 'l1tex__t_*_pipe_lsu_mem_global_op_*.sum' \
  | jq '.data.rows | group_by(.launch_row_id) | map({
        launch: .[0].launch_row_id,
        s_global: (([.[] | select(.counter_name | test("t_sectors_pipe_lsu_mem_global"))]  | map(.value | tonumber? // 0) | add) // 0),
        r_global: (([.[] | select(.counter_name | test("t_requests_pipe_lsu_mem_global"))] | map(.value | tonumber? // 0) | add) // 0)
      }
      | . + {sectors_per_request: (if .r_global > 0 then (.s_global / .r_global) else null end)})'
```

### Warp stall metric naming

Stall reason families are consistent across recent NCU but the
_set_ of reasons collected depends on the section. The portable
pattern is to filter by the `smsp__average_warps_issue_stalled_*_per_issue_active`
regex rather than naming individual reasons:

```bash
veloq ncu inspect R --row-id launch:0 \
  | jq '.data.rows[0]
        | (.metrics
            | map(select(.name | test("smsp__average_warps_issue_stalled_.*_per_issue_active"; "i")))
            | map({reason: (.name | capture("stalled_(?<f>[a-z0-9_]+)_per_issue_active").f),
                   pct:    (.value | tonumber? // null)})
            | sort_by(-.pct))'
```

This captures whichever reasons NCU actually collected without
hard-coding the list.

### Tensor pipe variants

Mixed-precision kernels split work across counters such as
`sm__pipe_tensor_op_hmma_cycles_active`,
`sm__pipe_tensor_op_imma_cycles_active`, etc. The
unified-percentage roll-up is
`sm__pipe_tensor.avg.pct_of_peak_sustained_elapsed` when the
section set carries it; otherwise sum the variants. Always
prefer the regex
`sm__pipe_tensor.*pct_of_peak_sustained_elapsed` before a
hard-coded counter name.

### PC sampling counter family

Reports captured with PC sampling expose
`smsp__pcsamp_warps_issue_stalled_*` (per-PC) plus
`smsp__pcsamp_warps_issue_stalled_*.sum` (aggregated). The
per-PC variants pass the source-attribution gate; the
`.sum` variants do not — they're launch-level scalars. If the
verb's `auxiliary.skipped_counters` reports a `.sum` pcsamp
counter as `not-a-source-counter`, drop the `.sum` suffix.

NCU 2026 reports also carry a **second, distinct** counter
family with the same suffix but a different origin and
different correlation semantics — the
`warpsampling:smsp__pcsamp_warps_issue_stalled_*` family.
**These are not aliases of each other**; both can co-exist in
the same launch with different metadata:

| Field                     | `smsp__pcsamp_warps_issue_stalled_<reason>` | `warpsampling:smsp__pcsamp_warps_issue_stalled_<reason>` |
| ------------------------- | ------------------------------------------- | -------------------------------------------------------- |
| `name` prefix             | bare `smsp__pcsamp_...`                     | `warpsampling:smsp__pcsamp_...`                          |
| `value_type`              | `uint64`                                    | `double`                                                 |
| `label`                   | populated (e.g. `stall_long_sb`)            | `null`                                                   |
| instance `correlation_id` | SASS VA inside the cubin's load-base range  | non-VA quantity (likely a packed time/warp encoding)     |
| Source-attributable?      | Yes — attribution accepts                   | No — correlation_ids land outside any cubin range        |

The two families are told apart by the metric **name** itself:
the `warpsampling:`-prefixed family is its own sampling pipeline.
The bare `smsp__pcsamp_...` family is the per-PC PC-sampling
family VeloQ's source attribution operates on; the
`warpsampling:` family is something else (a different
hardware/section path that captures stall reasons over time /
per warp, not per SASS PC), so its non-VA correlation_ids land
in `auxiliary.skipped_counters` rather than attributing to source.

**Practical consequence.** A portable
`*pcsamp_warps_issue_stalled_*` glob (leading wildcard) matches
both. VeloQ's positive-evidence gate (`correlation_type=uint64`

- PC inside cubin range) accepts only the source-attributable
  family; the `warpsampling:` family lands in
  `auxiliary.skipped_counters` with reason `not-a-source-counter`.
  The agent doesn't need to know which name is which — the gate
  routes correctly. Check `auxiliary.matched_counters` to see
  which counters contributed to the rows, and
  `auxiliary.skipped_counters` for the rejected ones.

### PM sampling counter family

A separate, **non**-PC family. PM-sampling captures the
time-series of standard performance counters during the
kernel's run; newer reports surface those counters under a
`pmsampling:` origin prefix:

| Surface           | Example                                                    |
| ----------------- | ---------------------------------------------------------- |
| Direct counter    | `dram__bytes.sum`                                          |
| PM-sampled mirror | `pmsampling:dram__bytes.sum.per_second`                    |
| Direct percentage | `dram__throughput.avg.pct_of_peak_sustained_elapsed`       |
| PM-sampled mirror | `pmsampling:dram__bytes.avg.pct_of_peak_sustained_elapsed` |

PM-sampling counters carry instance counts (one per sample
window) but the correlation type is the sample's time index,
**not** a SASS PC. They are not source-attributable and the
gate correctly rejects them for `ncu source-metrics`. Use them
for cross-time analysis via `ncu metrics`, not for source-line
attribution.

## Strategies for arch-portable jq

1. **Enumerate first, filter second.** Walk the metric list
   once with `jq -r '.data.rows[0].metrics[].name'`, grep for
   the family, then write the filter against the names you
   confirmed are present.

2. **Use regex tests, not equality.** `test("__pct_of_peak"; "i")`
   matches every percentage-of-peak counter regardless of which
   section produced it. `select(.name == "...")` breaks
   silently when the counter is renamed.

3. **Sum families instead of quoting one name.** "DRAM bytes",
   "global sectors", "tensor pipe ops" all live across multiple
   counters in some captures. A `map(.value | tonumber? // 0) | add`
   sum is portable; a single named counter is not.

4. **Test for `null` before computing ratios.** Counters can be
   missing entirely; without the guard, `(a / b)` raises a
   division error in some jq versions and silently produces
   `null` in others.

5. **Carry the actual names through your output.** Project the
   counter name alongside the value so a downstream reviewer
   can tell which counter family actually matched.

6. **Two reports, one filter.** When comparing across reports,
   anchor on `key` (built by VeloQ) rather than counter names.
   Different report captures may have different section sets but
   the keys are stable.

## Reference cross-links

- [`diagnosis-reference.md`](diagnosis-reference.md) — signal →
  command → threshold (uses regex-based filters throughout for
  arch-portability).
- [`analysis-dimensions.md`](analysis-dimensions.md) — the six
  dimensions with their canonical (regex-based) queries.
- [`metrics-and-sections.md`](metrics-and-sections.md) — what
  each section family answers; cross-link before quoting any
  family name from this file.
