//! Recipe-aware `--help` projector for NCU verbs.
//!
//! NCU subcommands carry their long-about text via clap-derive doc
//! comments on `Cmd` variants. Touchpoint 2, we want
//! the agent to see "which recipes use this verb" *before* the long
//! reference text. This module post-processes the augmented command
//! tree: for every NCU verb that appears in a registry recipe's
//! `related_verbs`, it prepends a "Recipes for this verb" block to the
//! existing `long_about`. Verbs absent from every recipe's
//! `related_verbs` get no extra text, so help output stays unchanged
//! for the rest.
//!
//! Composition pattern mirrors `crates/nsys/veloq-nsys/src/help.rs`
//! but stays much smaller — NCU verbs don't share the
//! `--device`/`--from`/`--nvtx` flag matrix NSys uses, so there's no
//! Common-flags block to render.

use veloq_core::recipes::recipes_for_verb;

/// List of every NCU subcommand name we project recipes onto. Keep in
/// sync with the variants in [`crate::cli::Cmd`]; the projector simply
/// looks up each name in the recipe registry.
const NCU_VERBS: &[&str] = &[
    "summary",
    "launches",
    "inspect",
    "metrics",
    "disasm",
    "ranges",
    "graphs",
    "sources",
    "source-metrics",
    "warp-stalls",
    // `schema` is a meta endpoint — no recipes target it.
];

/// Patch each NCU subcommand's `long_about` with a Recipes-for-this-
/// verb block when at least one registry recipe references it. Returns
/// the augmented command tree.
pub fn inject_long_about(mut cmd: clap::Command) -> clap::Command {
    for verb in NCU_VERBS {
        let recipes: Vec<_> = recipes_for_verb(verb).collect();
        if recipes.is_empty() {
            continue;
        }
        let block = render_recipes_block(&recipes);
        cmd = cmd.mut_subcommand(*verb, |sub| {
            let existing = sub
                .get_long_about()
                .map(|s| s.to_string())
                .or_else(|| sub.get_about().map(|s| s.to_string()))
                .unwrap_or_default();
            let composed = if existing.is_empty() {
                block.clone()
            } else {
                format!("{existing}\n\n{block}")
            };
            sub.long_about(composed)
        });
    }
    cmd
}

fn render_recipes_block(recipes: &[&'static veloq_core::recipes::Recipe]) -> String {
    let mut out = String::from(
        "Recipes for this verb (run `veloq recipes <id>` for the canonical command):\n",
    );
    for r in recipes {
        out.push_str(&format!("  {:<28} {}\n", r.id, r.title));
    }
    if out.ends_with('\n') {
        out.pop();
    }
    out
}
