# NCU Diagnosis Report Template

Use this as the _shape_ of an NCU diagnosis write-up. The
template makes evidence traceable and ranks recommendations by
expected impact and confidence so a reviewer can decide what
to act on first.

The template is a checklist, not a form. Skip sections that
don't apply. Every evidence line should carry the VeloQ
command that produced it — that's the audit trail.

Assume `R` is the `.ncu-rep` or `.ncu-repz` path under analysis. Substitute
your concrete report path, launch row id, kernel name, and
counter names when filling the template.

## Shape

```text
1. Headline
2. Inventory
3. Per-dimension findings
4. Ranked priorities
5. Confidence + open questions
6. Evidence appendix
```

A complete report needs sections 1, 3, and 4. Section 2 is
optional when the report is well-known. Section 5 grows as the
work continues and is the place to capture what would
disconfirm the diagnosis.

---

## 1. Headline

One paragraph. State:

- The report (path, capture command if known).
- The launch under analysis (`launch:<idx>`, kernel name,
  duration).
- The diagnosis class (compute / memory / latency / balanced —
  see [`diagnosis-reference.md`](diagnosis-reference.md)
  §sol_class).
- The single highest-confidence recommendation.

Template:

```markdown
**Report:** `path/to/report.ncu-rep`
**Captured with:** `ncu --set full --target-processes all ...` (if known)
**Launch:** `launch:N` — `<kernel_demangled>` — duration `<ns> ns`
**Class:** <compute | memory | latency | balanced>
**Top recommendation:** <one sentence>
```

Source command for the headline:

```bash
veloq ncu launches R --limit 5 \
  | jq '.data.rows[] | {key, kernel: .kernel_demangled, grid: .grid_size, block: .block_size}'

veloq ncu inspect R --row-id launch:0 \
  | jq '.data.rows[0]
        | {kernel: .kernel_demangled,
           duration_ns: (.metrics | map(select(.name == "gpu__time_duration.sum")) | .[0].value),
           sol: (.metrics | map(select(.name | test("__throughput\\.avg\\.pct_of_peak"; "i"))) | map({name, value}))}'
```

## 2. Inventory (optional)

For reports of unknown provenance, document what the report
actually carries before interpreting anything:

- Section identifiers present.
- Counter families present (metric-name prefixes).
- Whether source counters are populated.
- Whether `auxiliary.unattributed_sass_counter_totals` covers
  a significant fraction of any counter sum (suggests
  `-lineinfo` was missing at compile time).

```bash
veloq ncu inspect R --row-id launch:0 \
  | jq '.data.rows[0]
        | {metric_count: (.metrics|length),
           rule_count: (.rules|length),
           metric_prefixes: ([.metrics[]?.name // empty | split("__")[0]] | unique)}'

veloq ncu source-metrics R --row-id launch:0 --counter '*' --by line --limit 1 \
  | jq '{matched: .data.auxiliary.matched_counters,
         skipped: .data.auxiliary.skipped_counters,
         unattributed_share: (.data.auxiliary.unattributed_sass_counter_totals)}'
```

When inventory shows missing sections / counters, **stop and
recapture** rather than diagnosing around the gap. Note the
recapture decision under §5 (Confidence + open questions).

## 3. Per-dimension findings

One subsection per dimension that produced a noteworthy
signal. Skip dimensions whose readings landed in the _typical_
tier. The six dimensions in
[`analysis-dimensions.md`](analysis-dimensions.md) cover most
kernels; add other dimensions only when the question demands
them.

For each included dimension:

```markdown
### <Dimension name>

**Reading:** <one-line summary with concrete values>
**Tier:** <typical | noteworthy | red-flag>
**Evidence:**

- `veloq ncu inspect R --row-id launch:0 | jq '...'`
- `<output paste, trimmed to the relevant fields>`
  **Interpretation:** <one paragraph; pivot decision>
```

The evidence command must produce the values quoted in the
reading. Don't paraphrase numbers — paste them. If the value
came from a multi-step query, include the full pipeline.

Worked example:

```markdown
### Stalls

**Reading:** Dominant stall is `long_scoreboard` at ~52%; sum
of memory-class stalls ~64%; `eligible_warps` ~0.4/cycle.
**Tier:** red-flag (long_scoreboard ≥ 40%; memory-class sum ≥ 50%).
**Evidence:**

- `veloq ncu inspect R --row-id launch:0 | jq '.data.rows[0]
| (.metrics | map(select(.name | test("smsp__average_warps_issue_stalled_.*_per_issue_active")))
   | map({reason: (.name | capture("stalled_(?<f>[a-z0-9_]+)").f),
          pct:    (.value | tonumber)})
   | sort_by(-.pct))'`
- `[{"reason": "long_scoreboard", "pct": 52.4},
{"reason": "wait", "pct": 12.0}, ...]`
  **Interpretation:** memory-latency-bound. Pivot to dimension 6
  (memory access) — sectors-per-request + L2 hit rate are the
  next evidence to collect.
```

## 4. Ranked priorities

A numbered list of action items, ordered by _expected impact
first, cost second_. Each entry carries:

- **What** — the action (source change, launch-config change,
  recapture).
- **Why** — link to the dimension finding that motivates it.
- **Rule's own estimate** — when NCU itself emitted a speedup
  on the relevant rule, surface its `value_pct` verbatim; this
  is the rule's heuristic, not a verified forecast.
- **Verification** — the metric to remeasure after the change
  to confirm.

Template:

```markdown
1. **What:** Coalesce the global load in `<file>:<line>`.
   **Why:** §Memory access — sectors/request is ~12.
   **Rule's own estimate:** NCU's `UncoalescedGlobalAccesses`
   rule reported `speedup.value_pct ≈ 30` on this launch — its
   own heuristic, not a verified forecast. Treat as a relative
   priority signal, not a duration prediction.
   **Verification:** rerun → confirm `l1tex__t_sectors_pipe_lsu_mem_global_op_*.sum`
   falls; `gpu__time_duration.sum` drops.

2. **What:** Recapture with `--set source` + `--set roofline`
   for missing roofline + per-PC stall evidence.
   **Why:** §Inventory — source counters absent on current capture.
   **Rule's own estimate:** none (no NCU rule attached;
   unblocking action).
   **Verification:** new report carries SourceCounters section
   with non-empty per-PC instances.
```

Rule speedup estimates (`.rules[].speedup.value_pct` on the
inspect response) are NCU's own per-rule heuristics; they're
the best ordering signal VeloQ can surface but they remain
predictions, not measurements. Always verify with a
remeasurement before quoting them as outcomes.

## 5. Confidence + open questions

Honest accounting of what could disconfirm the diagnosis:

- **Confidence:** high / medium / low. _High_ needs at least
  two independent dimensions agreeing; _low_ is one rule
  finding or one cross-launch ratio.
- **Open questions:** specific things the present report
  can't answer; what would be needed.
- **What would change the diagnosis:** the cheapest
  observation that would flip the conclusion. If you can't
  name one, the diagnosis isn't falsifiable and confidence
  should drop.

Template:

```markdown
**Confidence:** medium
**Open questions:**

- Whether the dominant stall on `launch:1` matches `launch:0`
  (only `launch:0` analysed so far).
- Whether the kernel runs differently outside the profiler
  (NCU's serialised replay can mask scheduler effects).
  **Would flip the diagnosis if:**
- Sectors/request fell to ≤ 4 on a remeasurement, OR
- `eligible_warps` ≥ 1 on the next capture (different scheduler
  behaviour).
```

## 6. Evidence appendix

A flat list of every VeloQ command run during the analysis, in
the order they were run. Skip if §3 already inlines everything.
The appendix is for reproducibility — copy-paste-runnable by
the next person.

```bash
# 6.1 Launch inventory
veloq ncu launches R --limit 5 | jq '.data.rows[] | {key, kernel: .kernel_demangled}'

# 6.2 Launch inspection
veloq ncu inspect R --row-id launch:0 | jq '...'

# 6.3 Stall family
veloq ncu inspect R --row-id launch:0 | jq '.data.rows[0]
    | (.metrics | map(select(((.name // "") | test("smsp__average_warps_issue_stalled_.*_per_issue_active")))))'

# 6.4 Memory access (one glob; ncu metrics does not split on commas)
veloq ncu metrics R --counter 'l1tex__t_*_pipe_lsu_mem_global_op_*.sum' | jq '...'

# 6.5 Source-line hotspots (if relevant)
veloq ncu source-metrics R --row-id launch:0 --counter '<glob>' --by line --limit 10
```

If two reports were compared, include the diff command:

```bash
# 6.6 Two-report compare
veloq ncu metrics R_before --counter '*' > a.json
veloq ncu metrics R_after  --counter '*' > b.json
jq -n --slurpfile a a.json --slurpfile b b.json '<diff pipeline>'
```

## Anti-patterns to avoid

- **Quoting a percentage without the metric name.** "Occupancy
  was 35%" is unverifiable; "`sm__warps_active.avg.pct_of_peak_sustained_elapsed`
  was 35%" can be reproduced.
- **Citing a rule speedup without confirming the focus
  metric.** Rules are heuristics; verify against the linked
  metric family before quoting the speedup.
- **Recommending a fix without naming the metric to remeasure.**
  Every recommendation has a falsification path; if you can't
  name it, the recommendation isn't actionable.
- **Mixing units silently.** Counter values are raw / base
  units; carry units through every transformation.
- **Treating absence of a section as absence of the issue.**
  If the SOL section isn't in the report, the kernel isn't
  necessarily latency-bound — the evidence isn't there.
  Recapture before concluding.

## Cross-links

- [`SKILL.md`](../SKILL.md) — verb matrix + envelope shape.
- [`diagnosis-reference.md`](diagnosis-reference.md) — the
  signal → command → threshold table to populate §3.
- [`analysis-dimensions.md`](analysis-dimensions.md) — the six
  dimensions with their queries and tiers.
- [`metric-name-arch-notes.md`](metric-name-arch-notes.md) —
  how to enumerate before quoting a counter name.
- [`workflow.md`](workflow.md) — per-dimension bottleneck drills.
- [`metrics-and-sections.md`](metrics-and-sections.md) —
  section-family interpretation guide.
- [`limitations.md`](limitations.md) — when to leave VeloQ for
  native NCU recapture.
