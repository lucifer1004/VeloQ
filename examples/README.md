# VeloQ Examples

This directory contains public, scrubbed examples of outputs that VeloQ can
produce. Examples are meant to show workflow shape and report style, not to
serve as benchmark claims.

## Reports

- [`reports/nsys-timeline-vis/`](reports/nsys-timeline-vis/) - an
  agent-written NSys timeline report with static SVG figures produced by
  `veloq viz timeline`, top-k kernel highlights, density metadata, and a
  scrubbed evidence summary.

## Conventions

- Example traces are not committed.
- Local paths, hostnames, usernames, and private trace names are omitted.
- Commands use placeholders such as `TRACE`, `START_NS`, and `END_NS`.
- Committed figures may use compact render settings so the repository does not
  carry large raw timeline artifacts.
