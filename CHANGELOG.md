# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **self-update for `cargo install` builds** — map glibc Linux build targets
  (`*-unknown-linux-gnu`) to the shipped musl release archives, so
  cargo-installed binaries can self-update instead of failing with
  `meta.self-update.binary-install` (WI-2026-08-21-001).
- **Kernel tables without graph columns** — tolerate
  `CUPTI_ACTIVITY_KIND_KERNEL` exports that lack `graphId`/`graphNodeId`
  (seen on Nsight 2025.3 node-mode captures) in `stats`, `search`,
  `graph-replays`, and `inspect graph_node`, reporting NULL graph
  attribution instead of hard errors (WI-2026-08-21-002).

## [0.6.2] - 2026-08-20

### Added

- **Checkout-free agent plugin install** — `veloq agent install <agent>` now
  defaults to the Git marketplace `lucifer1004/veloq`, so binary-only users
  can install VeloQ Agent Skills without a source checkout;
  `--from-checkout` remains the validated local path, unchanged
  (WI-2026-08-20-001).

### Changed

- **Slimmed `nsys-profile-analysis` skill for strong models** — SKILL.md
  289 → 113 lines, value-proposition-first with answer-changing pitfalls
  promoted to first-class (incl. two bench-discovered traps: GRAPH_TRACE
  presence ≠ coverage, and globalTid TID extraction); command detail defers
  to `--help` / `veloq schema` / `veloq recipes`; references consolidated
  from five files to three (`pitfalls.md`, `capabilities.md`,
  `inspect-shapes.md`).

## [0.6.1] - 2026-08-04

### Fixed

- **Active NCU installation discovery** — resolve a PATH-selected `ncu`
  symlink before locating its bundled `extras/python/ncu_report.py`, so
  package environments use their matching Nsight Compute installation before
  unrelated platform-wide installs.
- **NSys host-only queryability** — accept canonical NVTX-only schema 3.x
  exports and keep implicit single-device resolution from becoming an invalid
  device filter for explicit NVTX, CUDA runtime, and OS runtime queries.

## [0.6.0] - 2026-07-30

### Added

- **Official PyPI NCU reader support** — use an `ncu_report` module already
  importable by the selected Python interpreter, including NVIDIA's
  `ncu-report` package, before falling back to full Nsight Compute
  installation discovery.
- **Explicit PyTorch Chrome trace filenames** — accept `.json` and
  `.json.gz` inputs under `veloq pytorch` while keeping automatic source
  detection restricted to `.pt.trace.json` and `.pt.trace.json.gz`.
- **Compressed NCU reports** — recognize and query zstd-compressed
  `.ncu-repz` reports alongside `.ncu-rep`, including cubin-backed
  disassembly and an explicit diagnostic for report readers that predate
  compressed-report support.
- **Optional local query daemon** — add manual `veloq daemon
  start/status/stop` lifecycle commands and `--daemon auto|off|required`
  routing over current-user-only local IPC. Resident sessions, exact rendered
  responses, admission, cancellation, idle expiry, freshness invalidation, and
  result-first/cost-aware session eviction are bounded by explicit daemon
  resource settings.
- **Daemon-resident NSys interval index** — after a second changing scan miss,
  eligible NSys sessions build one disposable, process-partitioned index over
  the existing fresh `gpu-work-events` sidecar. Start frontiers, compact gap
  references, and activity summaries accelerate varying `timeline`,
  `concurrency`, and `gaps` requests while one-off, stale, missing,
  over-capacity, or ineligible inputs retain the established path.
- **Daemon-resident NSys graph replay reuse** — graph replay sessions
  materialize process-qualified summaries, launchers, busy-time decomposition,
  and ranked node aggregates once. Changing windows, sorting, limits, and NVTX
  scopes reuse session-local evidence without creating persistent artifacts.
- **Daemon benchmark gate** — extend the leak-safe local benchmark to separate
  one-shot execution, resident construction, varying-argument cache misses,
  exact-response hits, and optional session-eviction rebuilds while reporting
  retained-memory and cache counters.

### Changed

- **Source execution boundary** — render source-owned JSON, CSV, table, and
  contextual errors into transport-neutral buffers so one-shot and daemon
  execution share the same typed dispatch and byte-for-byte output contract.
- **Daemon private framing** — replace newline-delimited JSON frames with a
  version-coupled, length-prefixed binary protocol. Bounded stdout and stderr
  chunks retain their native bytes instead of expanding each byte into a JSON
  number.
- **Daemon default resource budgets** — default to one active query, use the
  shared host-aware query-worker cap, and derive the resident-memory ceiling
  from the effective host or cgroup memory capacity. A single active query
  retains its source engine's machine-aware memory default; an explicit query
  memory ceiling remains available and is required when enabling concurrency.
- **Reuse-aware exact response admission** — exact responses that fit unused
  resident capacity remain immediately reusable. A first successful result
  that would require pressure eviction retains only small, accounted key
  evidence; the same exact query must succeed again before it may displace
  colder results or idle sessions under the existing eviction order.

### Fixed

- **Daemon launch lifecycle** — detach the service process from the invoking
  terminal process group so a successful `daemon start` remains live after
  its launcher exits.
- **Daemon execution correctness** — apply bounded admission even without
  reusable session identity, interrupt active NSys DuckDB work on shutdown,
  serialize work within each resident session, and close rather than relabel a
  session when post-query freshness changes.
- **Daemon transport and caching** — stream buffered output through bounded
  private-protocol chunks, preserve known source failures as completed CLI
  outcomes, cache only successful responses, and account retained exact keys
  and payloads.
- **Daemon raw output routing** — enable resident routing for NSys
  `ncu-command`, including byte-identical `--print` stdout and pipe-safe
  handled errors on stderr.

## [0.5.1] - 2026-07-28

### Added

- **Local agent plugin updates** — `veloq agent update` accepts
  `--from-checkout <path>` to re-register a durable local marketplace source
  before refreshing the selected Codex or Claude plugin.

### Fixed

- **Agent plugin lifecycle targeting** — upgrade `agent-plugin-installer` to
  use qualified `veloq@veloq` identities for Claude update and uninstall while
  preserving the existing named Git-marketplace update behavior when no local
  checkout is supplied.

## [0.5.0] - 2026-07-28

### Changed

- **NSys source wire version v4** — make CUDA identity process-aware.
  Process-sensitive rows now carry `process_id`; device, stream, context,
  graph replay, slice, gap, concurrency, and visualization keys include a
  `pid:` axis where required. `--process <PID> --device <ID>` precisely
  selects a rank-private CUDA device when logical ordinals collide.
- **NSys trace-map device inventory** — report physical GPU ids from
  `TARGET_INFO_GPU.id` separately from process-local
  `(process_id, device_id)` CUDA scopes.

### Fixed

- **Cross-process CUDA identity collisions** — stop merging ranks that reuse
  the same private `(device, context, stream, correlationId)` values in CUDA
  graph replay, correlate, NVTX attribution, gaps, concurrency, slices,
  statistics, and static timeline tracks.
- **Exact-scope recovery** — ambiguity diagnostics and scoped follow-up
  commands now preserve both native PID and logical device ordinal instead
  of suggesting another ambiguous bare `--device`.
- **Partial-trace scope discovery** — when CUDA context metadata is absent,
  recover process/device scopes from activity `globalPid` while retaining
  inactive ordinals from the target GPU inventory.

## [0.4.1] - 2026-06-19

### Added

- **Agent integration command** — add `veloq agent doctor/install/update/uninstall`
  for Codex and Claude VeloQ Agent Skills integrations, orchestrated through
  each runtime's native plugin CLI and backed by the reusable
  `agent-plugin-installer` crate.

### Changed

- **Agent plugin package layout** — make `plugins/veloq` the canonical package
  root for VeloQ Agent Skills and plugin manifests. Repo-local skill and
  manifest paths remain compatibility aliases, while CI, release packaging, and
  version bump tooling now validate the package-first layout directly.

### Fixed

- **Plugin checkout installs** — Codex installs now use materialized package
  contents, and Claude checkout installs target the lightweight
  `plugins/veloq` package root with normalized local paths instead of scanning
  the repository root.

## [0.4.0] - 2026-06-11

### Added

- **Timeline placement provenance** — `veloq viz timeline` now reports
  source axes, placement axes, and placement source for resolved tracks so
  agents can explain whether lanes are native device/stream tracks or
  attribution-derived context.
- **Top-k highlight scores** — timeline top-k kernel highlights include the
  ranking score and score total, and SVG legends show each highlighted
  kernel's contribution share.
- **Density rendering for dense timelines** — very small same-track intervals
  can be compacted into density bins instead of turning into misleading
  stretched bars or unreadable ticks. The response reports selected,
  rendered, density, and omitted counts.
- **Visualization examples** — add an example NSys timeline-visualization
  report with generated SVG figures and evidence summaries.

### Changed

- **NSys source wire version v3** — remove the `viz timeline`
  character-count label cap from the command surface and from
  `data.auxiliary.label_policy`. Timeline labels now use available pixel
  space and SVG clipping instead of a fixed character limit.
- **NSys timeline visualization internals** — split the NSys timeline
  exporter and `veloq-vis` renderer into focused modules for events,
  tracks, highlights, layout, painting, text fitting, and artifact writing.
- **Agent-facing NSys docs** — update the NSys Agent Skill, README, website,
  and RFC-0009 materials to describe placement provenance, density bins,
  and timeline figure interpretation.

### Fixed

- **SVG class sanitization** — sanitize interval CSS class tokens and escape
  raw class metadata in SVG output.
- **Release metadata** — bump workspace crates and Agent Skills plugin
  manifests to `0.4.0` so release artifacts do not reuse the already-published
  `0.3.0` version.

## [0.3.0] - 2026-06-10

### Added

- **NSys timeline SVG artifacts** — `veloq viz timeline` exports bounded,
  report-ready SVG figures under the trace artifact root while stdout remains
  the standard JSON envelope.
- **Timeline track roles** — NSys figures resolve GPU group rows, busy summary
  rows, stream detail rows, CUDA API annotations, and idle overlays so static
  reports can explain what each lane means.
- **Top-kernel highlights** — timeline figures can color the top kernel names
  or instances and report highlight metadata in `data.auxiliary`.
- **Visualization crate** — `veloq-vis` now owns source-neutral scene and SVG
  rendering primitives used by the NSys timeline exporter.
- **README and website figure showcase** — documentation now includes a sample
  NSys timeline SVG generated by VeloQ.

### Fixed

- **Viz timeline window flags** — `--from` and `--to` are enforced as paired
  arguments at parse time.
- **NVTX depth track keys** — `--track nvtx:depth=<N>` now routes intervals to
  the requested dynamic depth key instead of a hardcoded depth-1 key.
- **Viz table metadata** — table/CSV projections emit the timeline figure's
  `time_window_ns` metadata once, using the same `start-end` convention as
  other views.

## [0.2.2] - 2026-06-09

### Added

- **Governance artifacts** — RFCs now document VeloQ's agent-friendly,
  high-performance design center plus stable CLI I/O, row identity,
  source-registration, artifact-lifecycle, and per-source wire contracts.
- **NSys gaps sidecar** — repeated full-trace gaps queries can reuse a
  normalized GPU work sidecar, including graph-trace intervals.

### Changed

- **NSys gaps planning** — time-windowed gaps queries push scoped local
  windows into the sweep input and keep exact `total_matched` under
  row limits.
- **NSys GPU busy semantics** — gaps now count CUDA graph-trace intervals
  as GPU work, avoiding phantom idle on graph-only workloads.
- **Schema and wire guards** — PyTorch, NCU, and NSys schema targets now
  use source-of-truth registries with focused smoke coverage for canonical
  list shapes and keyed rows.

### Fixed

- **Timeline windows** — NSys timeline buckets clip events before bucket
  bounds are computed, so empty in-window work returns an empty result
  rather than an error.
- **CI governance install** — the CI workflow installs govctl from the
  release asset with the correct executable filter and a glibc-compatible
  runner image.

## [0.2.1] - 2026-06-07

### Added

- **Codex plugin marketplace** — repo-local Codex plugin metadata and
  marketplace entries now install the VeloQ Agent Skills with
  `codex plugin marketplace add .` followed by `codex plugin add veloq@veloq`.

### Changed

- **Agent Skills layout** — the canonical repo source and default install root
  moved to `.agents/skills`, with `.claude/skills` kept as a compatibility
  alias. Release skill archives include both `.agents/skills` and
  `.claude/skills` so old installers keep working while new installs default
  to the agent-neutral path.
- **Distribution docs** — installer, self-update, website, and plugin metadata
  now use Agent Skills terminology and the official VeloQ branding.
- **Prep stdout coverage** — CLI smoke coverage now guards that prep/export
  progress stays off stdout so JSON output remains parseable.

## [0.2.0] - 2026-06-07

### Added

- **PyTorch/Kineto source** — experimental Chrome-trace analysis for
  `.pt.trace.json` / `.pt.trace.json.gz` profiles, with `summary`, `search`,
  `inspect`, `stats`, `correlate`, `timeline`, `slices`, `collectives`,
  `prep`, and `schema`.
- **PyTorch profile-analysis skill** — `scripts/install.sh`,
  `veloq self-update`, and the plugin metadata now include
  `pytorch-profile-analysis` alongside the NSys and NCU skills.

### Changed

- **Distribution** — installer and self-update Agent Skills installs replace
  each bundled skill directory wholesale, so files removed from a later release do
  not linger from an older install.

## [0.1.0] - 2026-06-04

Initial public release.

### Added

- **Agent-friendly profile-query CLI.** A single `veloq` binary that answers
  one query per invocation and returns a stable JSON envelope on stdout
  (`schema: "v1"`), with CSV/table projections where useful. Designed so a
  coding agent or shell script can reason about GPU profiles without a GUI.
- **Nsight Systems (NSys) source** — timeline-trace analysis read through
  `nsys export -t parquetdir` (minimum nsys 2024.6). Verbs cover timeline and
  kernel statistics, kernel overlap/concurrency, NVTX path-aware attribution
  with domain identity, NCU handoff, prep/cache helpers, and `schema`.
- **Nsight Compute (NCU) source** — kernel-report analysis via NVIDIA's public
  `ncu_report` API into a leak-free native sidecar. Verbs: `summary`,
  `launches`, `inspect`, `metrics`, `disasm`, `ranges`, `graphs`, `sources`,
  `source-metrics` (per source-line counter attribution), `warp-stalls`, and
  `schema`.
- **Shared contract** — a common envelope and a pluggable `ProfileSource`
  trait across the NSys and NCU sources; every list response uses canonical
  `data.rows[]` with a stable per-row `key`, NSys trace responses carry top-level
  `trace_span` for per-second normalization, and errors come back through the
  same envelope shape with a non-zero exit code. The `inspect` not-found row
  uses the discriminator `type: "not_found"` consistently across both the NSys
  and NCU sources.
- **Root meta verbs** — `info`, `sources`, `clean`, `recipes`, and `self-update`. All VeloQ-generated
  products live under one `<report>.veloq/` artifact root with content/mtime
  cache invalidation.
- **Distribution** — `scripts/install.sh` installs the binary plus the
  `nsys-profile-analysis` and `ncu-profile-analysis` Agent Skills; a
  one-plugin marketplace listing ships under `.claude-plugin/`.

[Unreleased]: https://github.com/lucifer1004/veloq/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/lucifer1004/veloq/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/lucifer1004/veloq/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/lucifer1004/veloq/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/lucifer1004/veloq/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/lucifer1004/veloq/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/lucifer1004/veloq/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/lucifer1004/veloq/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/lucifer1004/veloq/releases/tag/v0.1.0
