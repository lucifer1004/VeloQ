# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
  trait across both sources; every list response uses canonical `data.rows[]`
  with a stable per-row `key`, NSys trace responses carry top-level
  `trace_span` for per-second normalization, and errors come back through the
  same envelope shape with a non-zero exit code. The `inspect` not-found row
  uses the discriminator `type: "not_found"` consistently across both the NSys
  and NCU sources.
- **Root meta verbs** — `info`, `sources`, `clean`, `recipes`, and `self-update`. All veloq-generated
  products live under one `<report>.veloq/` artifact root with content/mtime
  cache invalidation.
- **Distribution** — `scripts/install.sh` installs the binary plus the
  `nsys-profile-analysis` and `ncu-profile-analysis` Claude Code skills; a
  one-plugin marketplace listing ships under `.claude-plugin/`.

[0.1.0]: https://github.com/lucifer1004/veloq/releases/tag/v0.1.0
