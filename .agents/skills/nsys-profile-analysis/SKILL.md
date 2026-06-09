---
name: nsys-profile-analysis
description: "Analyze Nsight Systems `.nsys-rep` or `_pqtdir/` timeline traces using the VeloQ CLI. Use for GPU idle gaps, launch causes, CPU/GPU correlation, NVTX, CUDA graphs, metrics, sampling, and overlap/concurrency."
---

# Nsight Systems Profile Analysis

This skill guides Nsight Systems timeline investigations: what ran,
when it ran, how CPU work caused GPU work, where idle gaps appear, and
whether capture-time metric streams are trustworthy. Use `veloq` as
the evidence extractor for exported NSys traces, not as the starting
point for deciding what question to ask.

This is not a standalone native-NSys runbook. The skill can be installed
separately from the binary, but analysis requires the VeloQ CLI on
`PATH`; if `veloq` is missing, install it before continuing.

Use this skill only for Nsight Systems timeline traces. For Nsight
Compute `.ncu-rep` kernel reports, use the separate
`ncu-profile-analysis` skill.

## Tool Boundary

Use VeloQ verbs as the analysis interface. Do not query generated
`.veloq/` sidecars, generated `parquetdir/` children, `_pqtdir` tables,
or other profiler-cache files directly with DuckDB, PyArrow, pandas, or
ad hoc SQL unless the user explicitly asks for raw-table exploration or
you are developing VeloQ itself.

`veloq prep T` only warms caches. After prep, continue with VeloQ verbs:
`summary`, `stats`, `search`, `inspect`, `correlate`, `timeline`,
`viz timeline`, `slices`, `gaps`, `metrics`, `hardware`, or a
`veloq recipes <id>` workflow.

If no VeloQ verb can answer the question, first report the VeloQ
coverage gap and the closest command or recipe you tried. Use raw-table
exploration only as a last resort with explicit user approval.

Every JSON subcommand emits the same source-qualified envelope shape
(success or error); CSV/table are explicit row-shaped projections.
stderr is irrelevant to the JSON contract. Per-command flag and
response details live in `veloq <cmd> --help`; cross-cutting profiling
concepts live in the reference files below.

## Detection

```bash
veloq --version 2>&1 | head -1   # Is VeloQ installed?
veloq info path/to/trace.nsys-rep # Confirm VeloQ detects the source
veloq info path/to/trace_pqtdir   # Or a pre-exported parquetdir
file path/to/trace.nsys-rep       # Optional: confirm the NSys report
```

On a cached parquetdir, `veloq info` projects a trace map — devices,
processes (with rank-style env labels when present), NVTX top paths,
and an `applicable_recipes` list filtered by trace shape. Read those
recipe ids before composing a query: `veloq recipes` lists every
canonical workflow, `veloq recipes <id>` prints the body.

## Canonical workflows (recipes)

`veloq recipes` is the SSOT for guided investigations. Do not keep a
parallel recipe list in this skill: new and changed workflows live in
the compiled registry and are surfaced through `veloq recipes`,
`veloq info`'s `applicable_recipes`, and per-verb `--help`.

Agent workflow:

1. Run `veloq info <trace>` and read `data.applicable_recipes` when
   the trace is warm enough to expose capability predicates.
2. Run `veloq recipes` only when you need the full catalog.
3. Run `veloq recipes <id>` for the selected workflow and use its
   body as the canonical command sequence.
4. Use [references/cookbook.md](references/cookbook.md) only for
   extended examples, jq post-processing, and interpretation notes.

Per-verb `--help` also surfaces the recipes that touch that verb,
above the response schema reference.

`.nsys-rep` inputs require `nsys >= 2024.6` on `PATH` for the first
call. VeloQ exports them to `<trace>.nsys-rep.veloq/parquetdir/`
with `nsys export -t parquetdir` and reuses that generated parquetdir
on later calls. If you pass the generated `parquetdir/` child itself,
VeloQ maps it back to the owning `.nsys-rep`; it is not a separate
source. Pre-exported `_pqtdir/` directories are accepted too; their
derived VeloQ caches live under `<pqtdir>.veloq/`.

If `veloq` is missing, install it:

```bash
# Pre-built binary (Linux x86_64/aarch64, macOS x86_64/arm64)
curl -fsSL https://raw.githubusercontent.com/lucifer1004/veloq/main/scripts/install.sh | bash

# Or build from source — clone the repo first, then:
cargo build --release -p veloq
# binary lands at target/release/veloq
```

## Envelope contract (universal)

```json
{
  "schema": "v1",
  "source": { "kind": "nsys", "version": "v2" },
  "command": "nsys.stats",
  "trace": { "kind": "nsys", "path": "..." },
  "trace_span": { "origin_ns": 0, "span_ns": 12345000000 },
  "data": {
    "count": 50,
    "total_matched": 1234,
    "rows": [
      /* ... */
    ]
  }
}
```

Failure replaces `data` with `error: { message, chain }`; everything
else is unchanged.

Every list-shaped verb returns canonical `data.rows[]` with a
stable per-row `key` — `"kernel:1234"`, `"bucket|0..1000000"`,
`"gap|dev:0|@123"` (default device-scope; `--scope stream` adds
`|stream:N`, `--scope trace` drops both axes), `"slice|step_42|@123"`,
etc. Diff two captures by `INDEX(.data.rows; .key)` in jq — same
recipe across every verb. Non-primary data lands under
`data.auxiliary`.

**`search` and `correlate` event rows carry a top-level `type`
discriminator** (`"kernel"`, `"memcpy"`, `"sync"`, …) and per-kind
headline fields. `kernel` rows surface `grid` / `block` / `registers_per_thread`
/ `mangled_name` / `demangled_name` directly; `memcpy` and `memset`
carry `bytes`; `nvtx` carries `event_type` / `domain_id`. Reach
through with jq instead of doing a second `inspect`:

```bash
# Top 5 kernels by registers/thread without an inspect round-trip:
veloq search T --type kernel --limit 100 |
  jq '[.data.rows[] | select(.type=="kernel")] |
      sort_by(-.registers_per_thread)[:5] |
      map({demangled_name, grid, block, registers_per_thread})'

# Total bytes per memcpy direction, single query:
veloq search T --type memcpy --limit 5000 |
  jq '[.data.rows[] | select(.type=="memcpy")] |
      group_by(.copy_kind_name) |
      map({copy_kind_name: .[0].copy_kind_name, total_bytes: (map(.bytes) | add)})'
```

Run `veloq schema search` for the full per-kind payload schema.

**Timeline figures for reports**: after you have a bounded window from
`timeline`, `gaps`, `search`, or an NVTX slice, run
`veloq viz timeline T --from <start> --to <end>` to export an SVG under
the trace artifact root. Use the JSON response row's `path` for the
artifact and read `data.auxiliary.resolved_tracks` plus each track's
`role`. Treat `summary` tracks as rollups, `detail` tracks as concrete
lanes such as CUDA streams, `annotation` tracks as context, and idle
gaps as overlays rather than ordinary events. Also read the row counters
(`aggregated`, `omitted_track_count`, `suppressed_label_count`,
`truncated_label_count`) before embedding it. Mention those counters
when they are non-zero; the figure is static evidence, not a GUI-style
interactive timeline.

**Name search on large traces**: prefer `--name-regex 'foo'` over the
`--name '*foo*'` glob. Regex lets VeloQ resolve the matching names once
and prune the scan before name resolution, so it runs several times
faster on tens-of-millions-of-events traces; the glob resolves names
table-wide. Results are identical — it's purely a speed choice.

`trace_span` is the primary-execution window; divide totals by
`span_ns` for per-second rates without a separate `summary` call.
Omitted on verbs that don't read a trace.

Parse `.data` or `.error`, never stderr. The one exception is
`veloq nsys ncu-command --print`, which emits a raw shell script
for piping and keeps failures on stderr.

**For per-command shape detail**: run `veloq <cmd> --help` — the
`Response (.data)` block is projected directly from the response
struct's `JsonSchema` derive, so it can't drift from the actual
wire format. Strict-typed access: `veloq schema <cmd>` returns
the schemars-derived JSON Schema document.

## Evidence discipline

Treat timeline numbers as sourced evidence:

1. Cite the VeloQ command, trace path, and filters/time window behind
   every duration, count, or metric you report.
2. Probe `summary.data.auxiliary.capabilities` before capability-gated queries.
   Missing tables mean missing capture evidence, not absence of a
   behavior.
3. Read `coverage` before trusting GPU, NIC, CPU sampling, or CPU
   scheduler metrics. Low coverage means recapture before quoting
   conclusions.
4. Use NSys for timeline causality and overlap. Switch to NCU when
   the question becomes kernel-internal throughput, occupancy, warp
   stalls, memory transactions, or source-line attribution.

## Capability gate (read before every heavy query)

Most queries are conditional on what's in the trace.
`summary.data.auxiliary.capabilities` is a boolean bitmap of which
tables exist:

```bash
veloq summary T | jq '.data.auxiliary.capabilities'
```

Bail cleanly when a required capability is missing instead of
issuing a doomed query.

→ **Full bit list + when each matters + the three CPU-sampling
trust signals**: see [references/capabilities.md](references/capabilities.md)

## Where to look

| You want…                                                                                         | Read                                                         |
| ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| Per-command flags, JSON response shape, sort keys, examples                                       | `veloq <cmd> --help`                                         |
| Strict JSON Schema (machine-validated wire format)                                                | `veloq schema <cmd>`                                         |
| What questions can this trace answer (capability bits, trust signals)                             | [references/capabilities.md](references/capabilities.md)     |
| Time-window / NVTX attribution semantics                                                          | [references/time-and-nvtx.md](references/time-and-nvtx.md)   |
| Per-EventKind sub-shapes for `inspect`                                                            | [references/inspect-shapes.md](references/inspect-shapes.md) |
| Extended examples, jq snippets, and interpretation notes                                          | [references/cookbook.md](references/cookbook.md)             |
| Edge cases, capture-time prerequisites, "VeloQ can't do X"                                        | [references/limitations.md](references/limitations.md)       |

## Profiling Workflow

Start by classifying the symptom, then gather the minimum evidence:

| Symptom / question                         | Evidence path                                                                                                                                                                                                                                                                                                                             |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| "What dominates wall time?"                | `summary` → `stats --limit 20` → `stats --group-by demangled`                                                                                                                                                                                                                                                                             |
| GPU idle bubbles                           | `gaps --min-duration ...` → `correlate <next.row_id>` → inspect launch/runtime context                                                                                                                                                                                                                                                    |
| Poor overlap / "are my streams concurrent" | `concurrency` → per-device `overlap_ns` / `max_concurrency`, per-stream overlap (same-stream PDL), and `compute_vs_copy` (is copy hidden behind compute). Extraction-only; compute ratios in jq                                                                                                                                           |
| Report-ready visual timeline               | First identify a narrow window with `timeline` / `gaps` / `search` / `slices`, then `viz timeline --from ... --to ...`; cite both the textual evidence and the SVG row metadata                                                                                                                                                         |
| CPU blocked on GPU                         | `stats --type sync` → `search --type sync --sort duration:desc` → `correlate sync:N`                                                                                                                                                                                                                                                      |
| Iteration-to-iteration regression          | Require NVTX. `slices --name ...` → compare root ranges → scoped `stats --nvtx ...`. For nested same-name ranges use `stats --group-by nvtx-path` or `slices --aggregate --group-by path`. For per-kernel attribution: `inspect kernel:N` (default-on `nvtx_context.iter_index`) or `search --type kernel --with-nvtx` for a batched view |
| CUDA Graph behavior                        | Probe graph capability bits → `veloq recipes graph-replay-survey` or `graph-replay-hotspots`; use cookbook only for extra interpretation                                                                                                                                                                                                  |
| GPU utilization / bandwidth over time      | Require metric capture. `metrics --type gpu`; trust only after checking `coverage`                                                                                                                                                                                                                                                        |
| CPU host bottleneck                        | Require sampling/sched capture. `metrics --type cpu-sampling --group-by symbol\|stack` or `--type cpu-sched`, then `inspect <rows[].sample_row_id>` for a representative callchain                                                                                                                                                        |
| Kernel is internally inefficient           | Use NSys to identify `kernel:N`, generate a rerun with `ncu-command`, then switch to `ncu-profile-analysis` / native NCU                                                                                                                                                                                                                  |

For a broad first pass:

1. Run `summary` and read capabilities before issuing expensive or
   capability-gated queries.
2. Use `stats` / `timeline` / `gaps` to decide whether the issue is
   kernel time, idle time, copies, synchronization, or graph behavior.
3. Use `correlate` and `inspect` only after you have a concrete
   `row_id`; they answer why a visible timeline event happened.
4. When writing a report, export `viz timeline` for the selected
   bounded window and cite its resolved tracks/counters alongside the
   textual commands.
5. Use PM counters and CPU sampling only if the trace captured them,
   and only quote them after checking coverage/trust signals.
6. Stop using NSys when the question becomes instruction mix,
   occupancy, memory transaction efficiency, source lines, or warp
   stalls; those are NCU/kernel-analysis questions. If you have the
   selected kernel row, `veloq nsys ncu-command T kernel:N --print`
   can generate the native `ncu` command, but the NCU capture and
   analysis are separate steps.

Canonical command sequences live in `veloq recipes`; extended examples
and jq snippets live in [references/cookbook.md](references/cookbook.md).

## References

- Full bit-by-bit capability list + trust-signal interpretation →
  [references/capabilities.md](references/capabilities.md)
- Time-window + NVTX attribution model →
  [references/time-and-nvtx.md](references/time-and-nvtx.md)
- Per-EventKind shapes for `inspect` (auto-generated from Rust
  structs) →
  [references/inspect-shapes.md](references/inspect-shapes.md)
- Edge cases, capture-time prerequisites, "VeloQ can't do X" →
  [references/limitations.md](references/limitations.md)
- Extended examples and jq snippets →
  [references/cookbook.md](references/cookbook.md)
- Official NVIDIA Nsight Systems User Guide →
  https://docs.nvidia.com/nsight-systems/UserGuide/index.html
