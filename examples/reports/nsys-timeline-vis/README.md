# NSys Timeline Visualization Example

This directory is a public, scrubbed example of an agent-written report that
uses VeloQ NSys timeline SVG artifacts.

The workflow shown here is:

1. Inspect an NSys trace with VeloQ JSON commands.
2. Select bounded timeline windows.
3. Export static SVG timeline figures with top-k kernel highlights.
4. Write a concise Markdown report for human review.

## Files

- `report.md` - the human-facing analysis report.
- `figures/*.svg` - static `veloq viz timeline` artifacts.
- `evidence/summary.json` - portable, scrubbed metadata extracted from the
  original VeloQ JSON envelopes.

## Scope

The source trace names and local artifact paths are intentionally omitted.
The numbers and SVGs are preserved to demonstrate VeloQ's report workflow, not
as a general benchmark claim for any model, hardware, framework, or serving
stack.

The committed SVGs are size-limited public examples. They preserve resolved
tracks, top-k highlight legends, bounded-window shape, and density metadata.
They may use compact render settings so the repository does not carry
multi-megabyte raw timeline artifacts.
