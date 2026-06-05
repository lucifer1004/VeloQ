# NCU Limits And Tool Handoffs

`veloq ncu` reads existing `.ncu-rep` files and reshapes them for
agents. It does not profile live workloads, replay kernels, invent
missing metrics, or replace every native NCU UI workflow.

## When VeloQ helps

- Slim overview via `ncu summary` (totals + session metadata); full
  per-launch payload via `ncu inspect --row-id launch:<idx>`.
- Find kernels, workload ids, rules, sections, metrics, labels, units,
  and raw values across launches with `ncu launches`, `ncu inspect`,
  and `ncu metrics --counter '<glob>'`.
- List range / graph / source workloads via the dedicated
  `ncu ranges` / `graphs` / `sources` verbs.
- Produce a totals + session projection via
  `ncu summary --format csv|table`.
- Surface embedded sources, binary flags, relocations, SASS, PTX, and
  source-index rows when the report and CUDA tools allow it via
  `ncu disasm --row-id launch:<idx>` for per-launch correlation (the
  only disasm entry point).
- Let agents filter and summarize report data with `jq` on canonical
  `data.rows[]` plus per-row `key`; same recipe works on every verb,
  only the verb changes.

## When to use native `ncu`

Use native `ncu` CLI when the next step requires new profiler data:

- The report lacks the section or metric family needed for the
  question.
- A veloq/jq filter for a metric family returns no rows and the report
  inventory does not show that family.
- You need a different section set, custom metric list, replay mode,
  launch skip/count, cache control, or kernel selection.
- You need official CSV parity for a page VeloQ does not project yet,
  such as NCU's full source page.
- You need to verify a proposed fix with repeated fresh captures.

If the kernel came from an NSys timeline row, start with:

```bash
veloq nsys ncu-command trace.nsys-rep kernel:N --print
```

That produces a best-effort native `ncu` command using NSys-captured
argv/cwd/env and launch skip/count. Review it before execution,
especially for distributed jobs, non-deterministic launch order, CUDA
Graphs, or inputs that changed since the NSys capture.

Typical recapture pattern:

```bash
ncu --section SpeedOfLight \
    --section LaunchStats \
    --section Occupancy \
    --target-processes all \
    --export new-report \
    ./app args...

ncu --import new-report.ncu-rep --page details
veloq ncu summary new-report.ncu-rep        # totals overview
veloq ncu launches new-report.ncu-rep       # per-launch headline list
veloq ncu inspect new-report.ncu-rep --row-id launch:0   # full metrics + rules
```

Use targeted `--section` flags that match the question. Escalate to a
larger set only when the section inventory is genuinely unknown or
the investigation needs a broad sweep. Native NCU's sections are
groups of metrics organized around performance questions, so a
section-first recapture is usually faster and easier to interpret
than collecting everything.

For multi-process or distributed workloads with dependent concurrent
kernels, use native NCU's communicator/lockstep options rather than
assuming a normal single-process replay will make progress:

```bash
mpirun <args> ncu --communicator tcp \
                 --communicator-num-peers <ranks> \
                 --lockstep-kernel-launch \
                 --export report \
                 ./app args...
```

## When to use NCU GUI

Use NCU GUI when visual context is the work:

- Source page cross-highlighting across source/PTX/SASS/metrics.
- Guided analysis trees where the UI's context is faster than raw
  metrics.
- Manual inspection of many per-instruction counters.
- Screenshots or human-facing reports.

VeloQ can expose source/disassembly rows, but it is intentionally a
machine-readable CLI, not a GUI replacement.

## When to use NSys first

Use `nsys-profile-analysis` before NCU when the question is about the
application timeline:

- Is this kernel important enough to optimize?
- Is the GPU idle between kernels?
- Was the kernel launched late because the CPU was busy or blocked?
- Are memcpy/sync/graph events dominating instead of kernel internals?
- Which iteration or NVTX range regressed?

After NSys identifies a material kernel, use NCU to analyze that
kernel's internals.

## When to use compiler/source tools

Use compiler and source-level tools when the evidence already points
to a code change:

- PTXAS resource reports for registers, spill stores/loads, and shared
  memory.
- Source inspection for memory layout, coalescing, synchronization,
  divergence, algorithmic work, and launch configuration.
- SASS/PTX comparison across compiler flags or source variants.
- Unit/perf tests to verify a change outside profiler noise.

VeloQ can justify why to look there; it cannot prove a source change is
correct or profitable without new measurements.

## Data quality caveats

- Metrics are only as complete as the captured report. Missing metrics
  usually mean recapture, not a parser issue.
- Empty filtered metric lists are not evidence of absence until you
  have inventoried the report's sections and metric families.
- Some metric units are inferred from metric names because the report
  omits explicit unit metadata for some metrics.
- Raw values are not auto-scaled. Preserve `unit` / the raw-page unit
  row when reporting numbers.
- Multi-kernel reports need workload identity. Do not mix metrics from
  different kernels without keeping the `launch:<idx>` key plus
  `context_id`, `stream_id`, `device_id`, `grid_size`, `block_size`,
  and kernel name.
- A single NCU report rarely captures run-to-run variance. For
  release-quality claims, compare repeated captures and application
  timing.

## Current veloq gaps to remember

- No baked compare/diff verb for NCU reports yet — but the canonical
  `data.rows[]` + per-row `key` shape lets agents do `INDEX` + jq
  diffs themselves across two reports (especially via
  `ncu metrics --counter '<glob>'` long form).
- No full NCU source-page CSV/table projection yet; use native NCU or
  GUI for that page.
- VeloQ surfaces rule findings and metrics but does not implement a
  separate expert system beyond NCU's own rules.
- VeloQ does not run `ncu` capture commands for you. The NSys
  `ncu-command` helper only generates a rerun recipe.
- The `<file>.veloq/ncu-native.json.gz` sidecar is the single ingest
  path, built on first call and keyed on a sha256 content-hash of the
  `.ncu-rep`. If you replace the report with different content at the
  same path, the hash changes and VeloQ rebuilds. If generated
  products go stale for some other reason, run `veloq clean R` to
  remove the report's `<file>.veloq/` artifact root.

## NCU version coupling

VeloQ ingests `.ncu-rep` through NVIDIA's official `ncu_report` Python
API (required only at prep / first-touch; query-time is NCU-free).
Coupling to the installed `ncu` version is bounded by design:

- **No silent misclassification.** The version-specific metric enums
  (`metric_type` / `metric_subtype` / `rollup`) are resolved to stable
  _names_ from the live `ncu_report` enum at export, not interpreted as
  integer codes — so an enum renumber across `ncu` versions cannot
  silently flip VeloQ's additivity rollups. A name a newer `ncu` adds
  degrades to the documented name-suffix fallback.
- **Loud, not silent, on structural change.** A renamed/relocated enum
  container collapses the export's enum maps and stamps a
  `classification: "degraded"` marker plus a stderr warning. Renamed or
  removed `ncu_report` _methods_ make the export fail outright with the
  helper's error — never wrong data.
- **Drift is detectable on demand.** On a box with `ncu` installed,
  `VELOQ_NCU_LIVE=1 cargo test -p veloq-ncu --test ncu_live_drift`
  re-exports a committed report and flags any enum renumber / rename /
  API drift against the committed fixture.
- **Escape hatches** when discovery or the interpreter guess is wrong:
  `VELOQ_NCU_REPORT_DIR` (directory holding `ncu_report.py`) and
  `VELOQ_PYTHON` (interpreter to run the export helper).

## Official references

- NVIDIA Nsight Compute CLI:
  https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
- NVIDIA Nsight Compute Profiling Guide:
  https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
