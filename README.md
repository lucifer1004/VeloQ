<p align="center">
  <img src="docs/assets/logo.svg" alt="VeloQ logo" width="104" height="104" />
</p>

<h1 align="center">VeloQ</h1>

<p align="center"><em>Pure CLI in / JSON contract out; no GUI required.</em></p>

**Agent-friendly profile-query CLI family.** JSON by default, with
CSV/table projections where useful. One command answers one question.
VeloQ is designed for coding agents and scripts that need GPU profile
evidence without opening a GUI.

VeloQ covers three profile sources today — **Nsight Systems** (timeline
traces), **Nsight Compute** (kernel reports), and experimental
**PyTorch/Kineto** Chrome traces — through a single binary with a shared
envelope and a pluggable `ProfileSource` trait. The PyTorch/Kineto source
covers the Perfetto-style Chrome trace shape used by PyTorch profiler.

## Status

- 17 NSys verbs, including timeline analysis, static SVG figures,
  kernel overlap, NCU handoff, prep/cache helpers, and schema.
- 11 NCU verbs: `summary`, `launches`, `inspect`, `metrics`, `disasm`,
  `ranges`, `graphs`, `sources`, `source-metrics`, `warp-stalls`,
  and `schema`.
- 10 experimental PyTorch verbs: `summary`, `search`, `inspect`,
  `stats`, `correlate`, `timeline`, `slices`, `collectives`, `prep`,
  and `schema`.
- Six root meta verbs: `info`, `sources`, `clean`, `recipes`, `agent`,
  and `self-update`.

JSON output uses one `v1` envelope on stdout. List responses use
canonical `data.rows[]` with a stable per-row `key`; NSys trace
responses also carry top-level `trace_span` for per-second normalization.
Errors use the same envelope shape and a non-zero exit code.

### NSys ingestion

NSys traces are read through `nsys export -t parquetdir`. **Minimum
required nsys version is 2024.6** (the release that introduced the
`parquetdir` `--type`). All VeloQ-generated products live under one
`<report>.veloq/` artifact root; the NSys parquet cache is its
`parquetdir/` child with ctime invalidation.

## How it compares

For a GPU-profile question an agent usually reaches for one of three
interfaces. VeloQ focuses on the agent-facing axes: a **stable typed
contract**, **token economy**, and **scriptability**. It does not replace
`nsys` or `ncu`; it reads their exported evidence.

|                              | Nsight GUI | Raw `nsys`/`ncu` text in context | Hand-rolled SQLite + jq |             **VeloQ**              |
| ---------------------------- | :--------: | :------------------------------: | :---------------------: | :--------------------------------: |
| Scriptable / one-shot        |     ✗      |             ~ ad hoc             |            ✓            |                 ✓                  |
| Token-efficient for an agent |    n/a     |          ✗ broad dumps           |            ~            | ✓ shaped rows + truncation signals |
| Stable typed contract        |     ✗      |           ✗ free text            |    ✗ schema you own     |     ✓ versioned JSON envelope      |
| Cross-capture diffable       |     ✗      |                ✗                 |            ~            |       ✓ stable per-row `key`       |
| Zero setup per query         |     ✓      |                ✓                 |            ✗            |                 ✓                  |

Use the Nsight GUI for interactive timeline exploration or one-off visual
inspection. Use VeloQ for programmatic, repeatable, agent- or
script-driven querying.

## Install

For Linux and macOS, the install script is the shortest path: it installs
both the `veloq` binary and the bundled Agent Skills.

```bash
# Linux x86_64 / aarch64 and macOS x86_64 / arm64
curl -fsSL https://raw.githubusercontent.com/lucifer1004/veloq/main/scripts/install.sh | bash
```

Installs the `veloq` binary under `~/.local/bin` and the Agent Skills
for profile analysis (`nsys-profile-analysis`, `ncu-profile-analysis`,
`pytorch-profile-analysis`) under
`~/.agents/skills/`. Pass `--no-skills` to install just the
binary, or `--no-binary` to refresh the skills when you manage the
VeloQ CLI separately. The skills are VeloQ-backed: they can be
installed separately, but profile evidence extraction still requires a
`veloq` binary on `PATH`. `--bin-dir <path>` overrides the binary
install location.

For Windows, use `cargo binstall veloq` below or grab
`veloq-x86_64-windows.exe` from the
[Releases](https://github.com/lucifer1004/veloq/releases) page directly.

### Cargo binstall (binary only)

If you use [`cargo-binstall`](https://github.com/cargo-bins/cargo-binstall),
install the prebuilt `veloq` binary from the GitHub release:

```bash
cargo binstall veloq
```

`cargo binstall` installs only the executable. To fetch the bundled
skills from the latest release without replacing the binstall-managed
VeloQ binary, run:

```bash
veloq self-update --no-binary
```

Use `--skills-dir <path>` on that second command to install skills under a
non-default root such as `.claude/skills/`.

```bash
veloq self-update --no-binary --skills-dir .claude
```

### Agent plugin integrations (checkout)

From a VeloQ checkout, the `agent` meta command can install VeloQ-backed
Agent Skills through each runtime's native plugin CLI:

```bash
veloq agent doctor
veloq agent install codex --from-checkout .
veloq agent install claude --from-checkout .
veloq agent update codex --from-checkout .
veloq agent update claude --from-checkout .
veloq agent uninstall codex
veloq agent uninstall claude
```

Local marketplace paths are persistent runtime configuration. Keep the
checkout available after installation and pass it again to `agent update`.
Omitting `--from-checkout` preserves the named Git-marketplace update path for
installations registered from Git.

The plugin installs handle Agent Skills only. Those skills require the
VeloQ CLI for evidence extraction, so install the `veloq` binary separately
via `cargo binstall veloq` or `scripts/install.sh --no-skills`.

### Codex plugin (native CLI equivalent)

VeloQ ships a Codex plugin package under `plugins/veloq/` and a local
marketplace under `.agents/plugins/`. The `veloq agent install codex`
command wraps the native equivalent:

```bash
codex plugin marketplace add .
codex plugin add veloq@veloq
# Repeat both commands to refresh a local-checkout installation.
codex plugin remove veloq@veloq
```

The repo's canonical Agent Skills and plugin manifests live under
`plugins/veloq/`. The `.agents/skills/`, `.claude/skills/`,
`.codex-plugin/plugin.json`, and `.claude-plugin/plugin.json` paths are
compatibility aliases; root marketplace files remain as discoverable
registries. Contributors should edit `plugins/veloq/`, then run
`scripts/check-agent-plugin-package.sh` before committing plugin package
changes.

### Claude Code plugin (native CLI equivalent)

VeloQ ships a one-plugin marketplace listing under
`.claude-plugin/`. The `veloq agent install claude` command wraps the
native equivalent:

```bash
claude plugin marketplace add ./
claude plugin install veloq@veloq
# For a local-checkout refresh:
claude plugin marketplace add ./
claude plugin update veloq@veloq
claude plugin uninstall veloq@veloq
```

This uses the same Agent Skills through the Claude-specific plugin metadata;
the marketplace points at the shared `plugins/veloq/` package root.

### Updating

```bash
veloq self-update                              # binary AND bundled Agent Skills
veloq self-update --check                      # is a newer release out? (JSON)
veloq self-update --no-skills                  # binary only
veloq self-update --no-binary                  # Agent Skills only; keep your binary manager
veloq self-update --skills-dir .claude          # install skills to .claude/skills/
```

`self-update` pulls the latest GitHub release. By default it replaces the
running binary and refreshes the bundled Agent Skills, removing stale
skill files from earlier installs. Skills go to `~/.agents/skills/` by
default; `--skills-dir <path>` or `VELOQ_SKILLS_DIR` selects another
root such as project-local `.agents` or `.claude`. Passing either the
root or the final `skills/` directory works. `--check` reports
`update_available` without changing files. All modes emit the standard
envelope on stdout.

If the binary was installed with `cargo-binstall` and you want
`cargo-binstall` to remain the binary manager, use
`veloq self-update --no-binary` to refresh Agent Skills only, and use
`cargo binstall` again for binary updates.

## Quick start

These examples assume `veloq` is on `PATH` (via one of the install
methods above). For contributors building from source, see
[Build from source](#build-from-source) — the binary lands at
`target/release/veloq`.

```bash
# ── NSys (timeline) — hoisted to the top level and also available as `veloq nsys ...`
# Summarize a trace
veloq summary path/to/trace.nsys-rep
veloq nsys summary path/to/trace.nsys-rep

# Top kernels by total time
veloq stats path/to/trace.nsys-rep --limit 10
# Aggregate attributable kernels by full NVTX hierarchy path
veloq stats path/to/trace.nsys-rep --type kernel --group-by nvtx-path

# Human-friendly comfy-table view
veloq stats path/to/trace.nsys-rep --limit 10 --format table

# Find kernels by name. On large traces, --name-regex prunes the scan
# before name resolution and runs several times faster than the
# equivalent --name '*...*' glob (identical results).
veloq search path/to/trace.nsys-rep --type kernel --name-regex 'gemm' --sort duration:desc --limit 10

# Export a bounded timeline window as a report-ready SVG artifact.
# The JSON row returns the SVG path relative to <trace>.veloq/; resolved
# tracks carry roles such as group, summary, detail, and annotation.
# Dense events are aggregated into per-track density bins by default.
veloq viz timeline path/to/trace.nsys-rep --from @100000000 --to @120000000

# Select one process-private logical GPU exactly in a figure.
veloq viz timeline path/to/trace.nsys-rep --from @100000000 --to @120000000 \
  --track gpu:process=12345,device=0

# Highlight the top kernel names in that window while preserving the
# base event-type legend; metadata lands in data.auxiliary.resolved_highlights.
veloq viz timeline path/to/trace.nsys-rep --from @100000000 --to @120000000 --highlight-kernels top=3,scope=name
```

<p align="center">
  <img src="docs/assets/examples/nsys-timeline.svg" alt="Example VeloQ NSys timeline SVG with multi-GPU streams, NVTX lanes, density bins, and highlighted kernels" width="900" />
</p>

<p align="center"><em>Example <code>viz timeline</code> SVG artifact with per-device NVTX lanes, density bins, and top-kernel highlights.</em></p>

See the [examples index](examples/README.md) and the
[NSys timeline visualization report](examples/reports/nsys-timeline-vis/report.md)
for an agent-written Markdown report with embedded SVG figures and scrubbed
evidence metadata.

```bash
# Discover canonical workflows (nvtx-breakdown, gpu-idle-audit,
# timeline-figure-report, memcpy-asymmetry, cold-kernel-hotspot, ...)
veloq recipes
veloq recipes nvtx-breakdown

# GPU performance-counter samples (needs --gpu-metrics-devices at capture time)
veloq metrics path/to/trace.nsys-rep --type gpu --limit 8 --sort=mean:desc
# Same data as a 50ms time series
veloq metrics path/to/trace.nsys-rep --type gpu --counter '*Throughput*' --bucket 50ms

# NIC performance-counter samples (needs --nic-metrics=lf or =hf at capture time)
veloq metrics path/to/trace.nsys-rep --type nic --counter 'IB: Bytes*' --bucket 50ms

# CPU hotspot (needs --sample=process-tree at capture time)
veloq metrics path/to/trace.nsys-rep --type cpu-sampling --limit 20
# Per-thread breakdown
veloq metrics path/to/trace.nsys-rep --type cpu-sampling --group-by tid
# Drill: full callchain for one sample
veloq inspect path/to/trace.nsys-rep cpu_sample:1234

# Generate an Nsight Compute rerun command for a selected NSys kernel event
veloq nsys ncu-command path/to/trace.nsys-rep kernel:1234
veloq nsys ncu-command path/to/trace.nsys-rep kernel:1234 --print | bash

# ── NCU (`.ncu-rep` / `.ncu-repz` kernel reports) — namespaced under `ncu`
# Slim overview (launch-derived totals + NCU-version session)
veloq ncu summary path/to/report.ncu-rep
veloq ncu summary --format csv path/to/report.ncu-rep
# List launches; drill in for full per-launch metrics / rules
veloq ncu launches path/to/report.ncu-rep --kernel '*gemm*'
veloq ncu inspect path/to/report.ncu-rep --row-id launch:0
# Cross-launch metric projection (long form by default; jq-friendly diff shape)
veloq ncu metrics path/to/report.ncu-rep --counter 'sm__*active*'
# Per-launch SASS / PTX / source-line correlation (cached per cubin)
veloq ncu disasm path/to/report.ncu-rep --row-id launch:0 \
  | jq '.data.rows[0] | {function_name, instruction_count: (.instructions|length)}'
# Per-source-line warp-stall-reason histogram (from timed_warp_samples)
veloq ncu warp-stalls path/to/report.ncu-rep --row-id launch:0
# Other list verbs
veloq ncu sources path/to/report.ncu-rep
veloq ncu ranges path/to/report.ncu-rep
veloq ncu schema launches

# ── PyTorch/Kineto (Chrome traces) — namespaced under `pytorch`
veloq pytorch summary path/to/worker0.pt.trace.json
veloq pytorch search path/to/worker0.pt.trace.json --type kernel --is-comm
veloq pytorch correlate path/to/worker0.pt.trace.json kernel:91
veloq pytorch slices path/to/worker0.pt.trace.json --aggregate --group-by step
veloq pytorch stats path/to/worker0.pt.trace.json --type comm --group-by comm-kind,rank
veloq pytorch collectives path/to/worker0.pt.trace.json
veloq pytorch schema search

# ── Meta verbs
veloq sources
veloq info path/to/file.ncu-rep
veloq schema metrics
```

### Build from source

```bash
# The repo pins Rust 1.89.0 via rust-toolchain.toml.
cargo build --release -p veloq
# Binary lands at target/release/veloq — either invoke it via the
# full path or run `cp target/release/veloq ~/.local/bin/` to put
# it on PATH manually.
./target/release/veloq --help
```

> Heads-up: nsys's GPU/NIC/CPU-sample/SCHED buffers can silently
> drop data on long captures. Every `metrics` response carries
> coverage + per-type trust signals at `data.auxiliary.common`;
> `veloq metrics --type <gpu|nic|cpu-sampling|cpu-sched> --help`
> lists them. Read coverage before quoting numbers.

The first command on a new `.nsys-rep` runs `nsys export -t parquetdir`,
caching `<trace>.nsys-rep.veloq/parquetdir/<TABLE>.parquet` for reuse;
passing that generated `parquetdir/` back resolves to the owning
`.nsys-rep`, so sidecars stay under one artifact root. `veloq prep
<trace>` exports upfront and reports registered sidecar readiness in
`data.rows[]`; `veloq clean <trace>` removes the generated products
for one report.

## Response envelope

Every successful JSON call returns the source-qualified v1 envelope:

```json
{
  "schema": "v1",
  "source": { "kind": "nsys", "version": "v4" },
  "command": "nsys.stats",
  "trace": { "kind": "nsys", "path": "trace.nsys-rep" },
  "trace_span": { "origin_ns": 0, "span_ns": 12345000000 },
  "data": {
    "count": 50,
    "total_matched": 1234,
    "rows": [{ "key": "kernel|...|pid:12345|dev:0|stream:7", "...": "..." }]
  }
}
```

- `schema` — envelope-format version. Bumps on every breaking
  envelope-shape change.
- `source.kind` — which profile backend produced the response
  (`"nsys"`, `"ncu"`, `"pytorch"`, or `"veloq"` for meta verbs).
- `source.version` — per-source wire-format version. Bumps
  independently from the envelope when the source's payload shapes
  change. Currently NSys reports `v4` (`v1` introduced the NVTX domain
  dimension on `stats --group-by nvtx-path` rows; `v2` makes `prep` and
  `prep --status` canonical list responses where `data.rows[]` carries
  registered sidecar readiness keyed as `sidecar|<sidecar-id>`; `v3`
  removes the `viz timeline` label character-cap option and response
  echo; `v4` makes CUDA-local identities, filters, keys, aggregations,
  graph correlation, and trace-map device scopes process-aware) and NCU reports `v1` (the
  `ncu_report`-native wire — `inspect` carries no section catalog and
  `summary.auxiliary.session` keeps only the NCU version; each
  `ncu inspect` metric's
  `metric_type` / `metric_subtype` / `rollup` is the `ncu_report` enum
  _name_ such as `"counter"` rather than the integer `1`, with the raw
  integer kept alongside as `*_code`). PyTorch reports `v0`: it is
  experimental, but documented response fields, schema-target
  inventories, row ids/keys, command ids, and output-mode semantics are
  still part of the versioned source contract.
- `command` — qualified as `<source>.<verb>` for source verbs
  (`nsys.stats`, `ncu.summary`), or just `<verb>` for meta verbs
  (`info`, `sources`, `clean`).
- `trace.kind` — mirrors the producing `source.kind` (or the
  detected source kind for `veloq info`). Omitted entirely for
  trace-less verbs (`sources`, `schema`, `ncu.schema`).
- `trace_span` — primary-execution `(origin_ns, span_ns)` window.
  Agents normalize totals by `span_ns` to get per-second rates
  without a separate `summary` call. Omitted when the source does not
  provide a trace-wide window, and on meta verbs that don't read a
  trace.
- `data.rows[]` — canonical primary list on every list-shaped verb.
  Each row carries a `key: string` composed from its identifying
  axes (e.g. `"kernel:1234"`, `"bucket|0..1000000"`,
  `"slice|step_42|@1234567"`) so agents can
  `INDEX(.data.rows; .key)` across two captures and diff by key.
  Non-primary data lives under `data.auxiliary`.

**Stability.** The JSON envelope and the per-source `version`s are VeloQ's
public contract. Additive fields are non-breaking and keep the version; any
breaking shape change bumps `schema` (`ENVELOPE_VERSION`) or the affected
`source.version` and lands a CHANGELOG entry. The crate's `0.x` Cargo
version is **independent** of the wire version — pin behavior to the
envelope/source versions, not the crate version.

Errors share the same shape, with `data` replaced by `error`:

```json
{
  "schema": "v1",
  "source": { "kind": "nsys", "version": "v4" },
  "command": "nsys.stats",
  "trace": { "kind": "nsys", "path": "trace.nsys-rep" },
  "error": {
    "message": "invalid --from `1s`: must pair with --to",
    "chain": ["resolving --from/--to"]
  }
}
```

CLI-level parse failures (unknown flag, bad subcommand) omit `source`,
`command`, and `trace`. `--help` / `--version` print clap's native
usage text unchanged.

Exception: `veloq nsys ncu-command --print` intentionally writes a
raw shell script on stdout for piping, and writes failures to stderr
without a JSON envelope.

## Subcommands

### NSys verbs (hoisted to top level, also available under `nsys`)

| Command             | Purpose                                                                                                                                                                                 |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `summary`           | Overview: version, capabilities, per-table, primary vs full span                                                                                                                        |
| `stats`             | Aggregation across kernel/memcpy/memset/sync/runtime/osrt/graph/nvtx by name + composable axes                                                                                          |
| `search`            | Filter events → list of `row_id`s plus headline columns                                                                                                                                 |
| `inspect`           | Full per-kind details for one or more `row_id`s                                                                                                                                         |
| `correlate`         | CPU↔GPU causal chain for a `row_id`                                                                                                                                                     |
| `ncu-command`       | Generate a native `ncu` rerun command for one selected kernel event                                                                                                                     |
| `gaps`              | GPU idle bubbles. Default `--scope device` is per process/device and cross-stream; `--scope stream` for per-stream starvation; `--scope trace` for whole-trace idle                    |
| `timeline`          | Time-bucketed GPU activity (busy ns + per-kind breakdown per bucket)                                                                                                                    |
| `viz timeline`      | Export a bounded NSys timeline window as an SVG artifact with resolved track roles, placement provenance, render metadata, and label counters                                           |
| `concurrency`       | Kernel/transfer overlap: per-process/device union vs sum busy time, peak concurrency, per-stream (incl. same-stream PDL) + compute/copy overlap. Extraction-only (ratios in jq)         |
| `graph-replays`     | CUDA Graph replay decomposition: per-replay GPU work keyed by `(process, device, context, correlationId)`, across both `--cuda-graph-trace=graph` and `=node` captures                  |
| `slices`            | Per-NVTX-range CPU bounds + attributed GPU work                                                                                                                                         |
| `hardware`          | CPU / GPU / NIC inventory from the trace's `TARGET_INFO_*` tables                                                                                                                       |
| `metrics`           | GPU/NIC PM counters, CPU IP samples, or CPU scheduler events — hotspot summary, time series, callchain via `inspect`                                                                    |
| `prep`              | Build the Parquet cache + registered sidecars eagerly; `--status` reports sidecar readiness without building                                                                            |
| `correlation-stats` | Build/load the correlation index and report counts                                                                                                                                      |
| `schema <target>`   | Strict JSON Schema for one NSys verb's response                                                                                                                                         |

Every NSys command above can also be invoked as `veloq nsys <command>
...`; the top-level form is kept as the default-source shorthand.

NSys CUDA device ordinals are process-local. If multiple rank processes
each expose logical device 0, `--device 0` alone is ambiguous; use
`--process <native-pid> --device 0` for an exact scope. When an ordinal
matches only one process, the original `--device <ordinal>` form remains
sufficient and VeloQ resolves the PID automatically. `veloq info`
reports `trace_map.devices.physical` separately from
`trace_map.devices.logical_scopes[]` so physical inventory is not confused
with `(process_id, device_id)` query identity. `viz timeline` expresses the
same exact scope inside each device-bearing track spec:
`--track gpu:process=<native-pid>,device=0`.

### NCU verbs (namespaced under `ncu`)

NCU verbs share a `<trace>.veloq/ncu-native.json.gz` sidecar built on
first use; subsequent calls deserialise it instead of re-ingesting the
report.

All NCU detail verbs accept `--format json\|csv\|table`; tabular
output mirrors the JSON `data.rows[]` one row per output line
(nested objects become dotted-key columns, `BTreeMap` fields like
`counters` expand to one column per resolved counter name). `ncu
schema` is JSON-only.

| Command               | Formats            | Purpose                                                                                                                                                                                                                                                       |
| --------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ncu summary`         | json / csv / table | Slim overview: one launch-derived totals row + degraded session (NCU version only). `--format csv\|table` renders the totals + session as a `section,key,value` projection.                                                                                   |
| `ncu launches`        | json / csv / table | List CUDA kernel launches as headline rows (`launch:<idx>`); filters: `--kernel '<glob>'`, `--nvtx-range '<glob>'`, `--grid WxHxD`, `--block WxHxD`, `--limit`                                                                                                |
| `ncu inspect`         | json / csv / table | Full per-launch payload (full metric list with placement-tagged instances + rules + recovered identity scalars) for one or more `--row-id launch:<idx>`; malformed, unsupported-kind, and out-of-range ids return `not_found` rows so partial batches survive |
| `ncu metrics`         | json / csv / table | Cross-launch metric projection. Default long form (one row per `(launch, counter)`); `--per-launch` for wide form (BTreeMap counters expand to one column per name)                                                                                           |
| `ncu disasm`          | json / csv / table | SASS / PTX / source-index correlation for the cubin one launch ran out of (cubin extracted from the report, cached per-cubin under `<report>.veloq/disasm/`); tabular emits one row per SASS instruction with denormalised kernel identity                    |
| `ncu source-metrics`  | json / csv / table | Per-source-line / per-SASS / per-file NCU counter attribution. Joins per-PC metric instances with DWARF source-line attribution; `--by line\|sass\|file`. See `veloq recipes source-line-hotspots` for the canonical invocation.                              |
| `ncu warp-stalls`     | json / csv / table | Per-source-line warp-stall-reason histogram from `timed_warp_samples` (the raw warp-state stream); `--by line\|sass\|reason`, `--file '<glob>'`. Raw sample counts + `not_issued`; jq for percentages.                                                        |
| `ncu ranges`          | json / csv / table | List range workloads (`--replay-mode range`)                                                                                                                                                                                                                  |
| `ncu graphs`          | json / csv / table | List CUDA-graph workloads (`--graph-profiling graph`)                                                                                                                                                                                                         |
| `ncu sources`         | json / csv / table | Per-cubin source metadata (`cuda_sm_name`, `embedded_source_file_count`, `has_disasm`), one row per launch's cubin                                                                                                                                            |
| `ncu schema <target>` | json               | Strict JSON Schema for one NCU response. Targets are the response field inventory: `summary \| launches \| inspect \| metrics \| disasm \| ranges \| graphs \| sources \| source-metrics \| warp-stalls`                                                      |

NCU drill verbs other than `inspect` may return handled diagnostic
errors for malformed, unsupported-kind, or out-of-range launch row ids.

### PyTorch verbs (namespaced under `pytorch`)

PyTorch is an experimental `source.version = "v0"` source for Kineto
Chrome trace files. Explicit `veloq pytorch ...` commands accept any
`.json` / `.json.gz` filename; automatic source detection remains limited
to `.pt.trace.json` / `.pt.trace.json.gz` so VeloQ does not claim unrelated
JSON files. Directory inputs and cross-rank collective skew are planned,
not shipped in v0.
When one trace file contains multiple rank values, rank-scoped commands
(`search`, `stats`, `timeline`, `slices`, and `collectives`) require
`--rank <n>` or `--all-ranks`. `inspect` and `correlate` operate on
explicit row ids and are not rank-scope gated.
CUDA device ids are rank-local in multi-rank traces, and stream ids are
device-local: use `--rank <n> --device <id> --stream <id>` for a fixed
stream, or project parent axes with `--group-by rank,device,stream` for
comparison.
It uses the same general VeloQ verbs instead
of adding parallel `steps`, `memory`, or `comm` commands; communication
questions use `--type comm`, `--is-comm`, grouping axes, `slices`, and the
source-specific `collectives` verb.

| Command                   | Formats            | Purpose                                                                                                        |
| ------------------------- | ------------------ | -------------------------------------------------------------------------------------------------------------- |
| `pytorch summary`         | json / csv / table | Trace inventory, capabilities, active devices, rank/worker inference, versions, capture flags                  |
| `pytorch search`          | json / csv / table | Typed event refs; filters include `--type`, name glob/regex, duration, time, rank, device, stream, step        |
| `pytorch inspect`         | json / csv / table | Raw args, typed args, parent/children, step/Python context, and correlation/flow links for one or more row ids |
| `pytorch stats`           | json / csv / table | Duration/count aggregation by `name,type,step,rank,device,stream,shape,comm-kind,python-context,python-path`   |
| `pytorch correlate`       | json / csv / table | CPU op / annotation / runtime / driver / GPU activity causal chain for one or more row ids                     |
| `pytorch timeline`        | json / csv / table | Time buckets with CPU, GPU, communication, and per-type time                                                   |
| `pytorch slices`          | json / csv / table | ProfilerStep and user annotation range instances or aggregates                                                 |
| `pytorch collectives`     | json / csv / table | Single-trace communication groups with CPU/NCCL evidence row ids and link/ordinal confidence                   |
| `pytorch prep`            | json / csv / table | Build or inspect PyTorch sidecars under `<input>.veloq/pytorch/`                                               |
| `pytorch schema <target>` | json               | Strict JSON Schema for one PyTorch response; schema targets are the response field inventory                   |

### Meta verbs (root, owned by the binary)

| Command          | Purpose                                                                                                                                                                                                                                                                                                                           |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `info <trace>`   | First-touch trace map: source kind, filesystem facts, capability bitmap, plus (on a cached parquetdir) device/process inventory, NVTX domains + top paths, and `applicable_recipes` filtered by trace shape. Sub-100ms on a parquetdir; basics-only on a cold `.nsys-rep` with a `meta.next_steps` hint pointing at `veloq prep`. |
| `recipes [<id>]` | List or show registered workflow recipes (run `veloq recipes` for the catalog, `veloq recipes <id>` for one).                                                                                                                                                                                                                     |
| `sources`        | Registered sources and their wire-format versions                                                                                                                                                                                                                                                                                 |
| `clean <trace>`  | Remove the `<trace>.veloq/` artifact root generated by VeloQ                                                                                                                                                                                                                                                                      |
| `agent`          | Install, update, uninstall, and diagnose VeloQ Agent Skills integrations for supported agent runtimes (`doctor` / `install <agent> --from-checkout <path>` / `update <agent> [--from-checkout <path>]` / `uninstall <agent>`)                                                                                                               |
| `self-update`    | Update the binary and bundled Agent Skills from the latest GitHub release (`--check` / `--no-skills` / `--no-binary` / `--skills-dir`)                                                                                                                                                                                            |

Per-verb flag detail, response shape, sort keys, and examples live
in `veloq <verb> --help` (which is projected from the same
`JsonSchema` derive as the response, so it can't drift).

## NVTX caveat

NSys's `NVTX_EVENTS` table records CPU-side range timestamps only;
GPU work is reached by walking `correlationId` from `NVTX → runtime
API → kernel/memcpy/memset` with `(process, device, context)` disambiguation
from `TARGET_INFO_CUDA_CONTEXT_INFO`. VeloQ does this walk in SQL for
`stats --nvtx`/`search --nvtx`/`slices` and in a pre-built index
(`<trace>.veloq/correlation.bin`) for `correlate`.

The same walk runs in reverse for `inspect` (default-on) and
`search --with-nvtx` (opt-in batched): given a kernel / memcpy /
memset / sync `row_id`, VeloQ surfaces `nvtx_context: { range_id,
name, depth, iter_index }` for the innermost enclosing NVTX range.
`iter_index` is the 0-based ordinal among same-`(global_tid,
domain_id, name)` repeats — answers "which step did this kernel
belong to" without a second jq pass.

For nested NVTX, `veloq stats T --group-by nvtx-path` and
`veloq slices T --aggregate --group-by path` group by the full
slash-joined hierarchy path, so repeated leaf
names under different parents remain distinct. `inspect T nvtx:N`
also includes `path`, `parent_row_id`, and `parent_name` when the
NVTX tree can be built.

## Inputs

| Source  | Extensions                  | Notes                                                                                                         |
| ------- | --------------------------- | ------------------------------------------------------------------------------------------------------------- |
| NSys    | `.nsys-rep`                 | Primary path; exported via `nsys export -t parquetdir` on first use                                           |
| NSys    | `<stem>_pqtdir/`            | Pre-exported parquetdir; opened directly                                                                      |
| NSys    | `<trace>.veloq/parquetdir/` | Generated alias for the owning `.nsys-rep`; not a separate source                                             |
| NCU     | `.ncu-rep`, `.ncu-repz`     | Nsight Compute kernel report, plain or zstd-compressed (ingested via NVIDIA's `ncu_report` API at prep time)   |
| PyTorch | `.pt.trace.json`            | PyTorch/Kineto Chrome trace JSON; eligible for automatic detection                                            |
| PyTorch | `.pt.trace.json.gz`         | Gzipped PyTorch/Kineto Chrome trace JSON; eligible for automatic detection                                    |
| PyTorch | `.json`, `.json.gz`         | Accepted when explicitly routed through `veloq pytorch`; not claimed by automatic source detection            |

`veloq info <trace>` reports which source claims the file based on
the same `detect()` heuristic the dispatcher uses, so an agent can
probe a path without having to maintain its own extension list.

NCU ingestion runs NVIDIA's `ncu_report` Python API at **prep time only**;
query-time is NCU-free and the generated `<report>.veloq/` sidecar is
portable across Linux/macOS/Windows. VeloQ first uses an `ncu_report`
module already importable by the selected interpreter, including NVIDIA's
official `ncu-report` PyPI package. It otherwise auto-discovers a full
Nsight Compute install (`extras/python`, or the macOS app bundle's
`Contents/MacOS/python`). For a non-standard location, set
`VELOQ_NCU_REPORT_DIR` to the directory containing `ncu_report.py`, and/or
`VELOQ_PYTHON` to the interpreter to run the helper with.

## License

MIT
