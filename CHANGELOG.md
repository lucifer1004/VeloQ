# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.2]: https://github.com/lucifer1004/veloq/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/lucifer1004/veloq/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/lucifer1004/veloq/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/lucifer1004/veloq/releases/tag/v0.1.0
