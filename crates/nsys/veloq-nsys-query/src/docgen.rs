//! Auto-generated reference-doc bodies.
//!
//! veloq's skill (`.claude/skills/nsys-profile-analysis/`) keeps a
//! couple of files whose content is purely a projection of the
//! response structs in this crate — they're not narrative, they're
//! "what does each event-kind variant look like." Generating them
//! from [`veloq_core::wire_format::wire_format_for`] keeps them in sync with the Rust types
//! that `serde_json` actually emits.
//!
//! Two consumers share each generator:
//!
//! - `crates/veloq-nsys-query/examples/gen_inspect_shapes.rs` — runs in
//!   `write` mode to regenerate the file after struct changes.
//! - `crates/veloq-nsys-query/tests/inspect_shapes_freshness.rs` — runs
//!   in `check` mode (the default) so CI catches stale on-disk
//!   content.

/// Markdown body for `references/inspect-shapes.md`. Includes a
/// header explaining provenance and how to regenerate, then the
/// projected `EventDetails` schema inside a fenced code block.
pub fn inspect_shapes_body() -> String {
    use crate::inspect::EventDetails;
    use veloq_core::wire_format::wire_format_for;

    let wf = wire_format_for::<EventDetails>();
    let mut out = String::new();
    out.push_str(
        "<!--\n  \
         AUTO-GENERATED — do not edit by hand.\n  \
         Source of truth: `#[derive(JsonSchema)]` on the structs in\n  \
         `crates/veloq-nsys-query/src/inspect.rs`. Regenerate with:\n  \
           cargo run --release -p veloq-nsys-query --example gen_inspect_shapes -- write\n  \
         CI runs `cargo test -p veloq-nsys-query --test inspect_shapes_freshness`\n  \
         which asserts on-disk content == projected output.\n\
         -->\n\n",
    );
    out.push_str("# `inspect` — per-EventKind sub-shapes\n\n");
    out.push_str(
        "`veloq inspect <TRACE> <ROW_ID> [<ROW_ID>...]` returns \
         `{ count, total_matched, rows: EventDetails[] }`. Row_ids \
         are positional (one or more) in `<kind>:<rowid>` form, \
         e.g. `kernel:1234`. Each row's `type` tag selects which \
         sub-shape applies. The block below is projected from the \
         Rust structs so it stays in sync with the actual wire \
         format — same source as `veloq schema inspect`.\n\n",
    );
    out.push_str("```\n");
    out.push_str(&wf.render());
    out.push_str("\n```\n");
    out
}
