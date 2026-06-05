//! Build-time codegen for the recipe registry.
//!
//! Reads `src/recipes/registry.toml`, validates the schema, and emits
//! a Rust file at `OUT_DIR/recipes_generated.rs` carrying a
//! `pub static RECIPES: &[Recipe] = &[ ... ];` literal. The runtime
//! `mod recipes` defines the [`Recipe`] struct and `include!()`s the
//! generated file.
//!
//! Per [[feedback_build_rs_bundling]]: compile-time codegen catches
//! malformed entries when the maintainer edits registry.toml (build
//! failure), not on first call to `veloq recipes` (runtime parse
//! error). The parser crate (`toml`) lives under
//! `[build-dependencies]` so it doesn't ship with the binary.

use serde::Deserialize;
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::Write as _;
use std::path::PathBuf;

const REGISTRY_PATH: &str = "src/recipes/registry.toml";

/// Predicates recognised on `trace_shape`. Stay in sync with the
/// runtime `Recipe::matches_trace_shape` matcher in
/// `crates/veloq-core/src/recipes/mod.rs`.
const TRACE_SHAPE_PREDICATES: &[&str] = &[
    "has_kernels",
    "has_memcpy",
    "has_nvtx",
    "has_target_info",
    "multi_device",
    "multi_process",
    "has_graph_trace",
    "has_graph_nodes",
];

#[derive(Deserialize)]
struct Registry {
    #[serde(default)]
    recipe: Vec<Recipe>,
}

#[derive(Deserialize)]
struct Recipe {
    id: String,
    title: String,
    body: String,
    keywords: Vec<String>,
    related_verbs: Vec<String>,
    #[serde(default)]
    trace_shape: Vec<String>,
}

type BuildResult = Result<(), Box<dyn Error>>;

fn main() -> BuildResult {
    println!("cargo:rerun-if-changed={REGISTRY_PATH}");
    println!("cargo:rerun-if-changed=build.rs");

    let src = std::fs::read_to_string(REGISTRY_PATH)
        .map_err(|e| format!("reading {REGISTRY_PATH}: {e}"))?;
    let registry: Registry =
        toml::from_str(&src).map_err(|e| format!("parsing {REGISTRY_PATH}: {e}"))?;

    validate(&registry.recipe)?;

    let out_dir = std::env::var_os("OUT_DIR")
        .ok_or("OUT_DIR not set — cargo must invoke this build script")?;
    let out_path = PathBuf::from(out_dir).join("recipes_generated.rs");
    let code = render(&registry.recipe)?;
    std::fs::write(&out_path, code).map_err(|e| format!("writing {}: {e}", out_path.display()))?;
    Ok(())
}

fn validate(recipes: &[Recipe]) -> BuildResult {
    let mut seen_ids: BTreeSet<&str> = BTreeSet::new();
    for r in recipes {
        if !seen_ids.insert(&r.id) {
            return Err(format!("duplicate recipe id `{}`", r.id).into());
        }
        if r.id.is_empty()
            || !r
                .id
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        {
            return Err(format!(
                "recipe id `{}` must be a kebab-case slug (lowercase ASCII letters / digits / hyphens)",
                r.id
            )
            .into());
        }
        if r.title.trim().is_empty() {
            return Err(format!("recipe `{}` has empty title", r.id).into());
        }
        if r.body.trim().is_empty() {
            return Err(format!("recipe `{}` has empty body", r.id).into());
        }
        if r.keywords.is_empty() {
            return Err(format!("recipe `{}` must declare at least one keyword", r.id).into());
        }
        for kw in &r.keywords {
            if kw.trim().is_empty() || kw.contains(' ') {
                return Err(format!(
                    "recipe `{}` keyword `{kw}` must be a non-empty lowercase token without spaces",
                    r.id
                )
                .into());
            }
        }
        if r.related_verbs.is_empty() {
            return Err(format!("recipe `{}` must declare at least one related_verb", r.id).into());
        }
        for group in &r.trace_shape {
            // Predicate semantics: see `Recipe::matches_trace_shape`.
            // Empty alternants (e.g. leading/trailing `|`) are rejected
            // so the runtime matcher doesn't have to decide what an
            // empty predicate name means.
            if group.is_empty() {
                return Err(format!("recipe `{}` has an empty trace_shape entry", r.id).into());
            }
            for pred in group.split('|') {
                if pred.is_empty() {
                    return Err(format!(
                        "recipe `{}` trace_shape group `{group}` has an empty alternant; \
                         remove the stray `|` or fill in the predicate",
                        r.id,
                    )
                    .into());
                }
                if !TRACE_SHAPE_PREDICATES.contains(&pred) {
                    return Err(format!(
                        "recipe `{}` references unknown trace_shape predicate `{pred}` \
                         (expected one of: {})",
                        r.id,
                        TRACE_SHAPE_PREDICATES.join(", "),
                    )
                    .into());
                }
            }
        }
    }
    Ok(())
}

fn render(recipes: &[Recipe]) -> Result<String, Box<dyn Error>> {
    let mut out = String::new();
    out.push_str(
        "// Generated by build.rs — do not edit.\n\
         // Source: src/recipes/registry.toml\n\n",
    );
    out.push_str("pub static RECIPES: &[Recipe] = &[\n");
    for r in recipes {
        out.push_str("    Recipe {\n");
        writeln!(out, "        id: {},", string_literal(&r.id))?;
        writeln!(out, "        title: {},", string_literal(&r.title))?;
        writeln!(out, "        body: {},", string_literal(&r.body))?;
        writeln!(
            out,
            "        keywords: &[{}],",
            r.keywords
                .iter()
                .map(|s| string_literal(s))
                .collect::<Vec<_>>()
                .join(", ")
        )?;
        writeln!(
            out,
            "        related_verbs: &[{}],",
            r.related_verbs
                .iter()
                .map(|s| string_literal(s))
                .collect::<Vec<_>>()
                .join(", ")
        )?;
        writeln!(
            out,
            "        trace_shape: &[{}],",
            r.trace_shape
                .iter()
                .map(|s| string_literal(s))
                .collect::<Vec<_>>()
                .join(", ")
        )?;
        out.push_str("    },\n");
    }
    out.push_str("];\n");
    Ok(out)
}

/// Emit a Rust raw string literal that round-trips arbitrary text
/// without needing escape rules. Picks the smallest `r#…#` hash count
/// that doesn't collide with a closing quote sequence in `s`.
fn string_literal(s: &str) -> String {
    let mut hashes = 0;
    loop {
        let close = format!("\"{}", "#".repeat(hashes));
        if !s.contains(&close) {
            break;
        }
        hashes += 1;
    }
    let h = "#".repeat(hashes);
    format!("r{h}\"{s}\"{h}")
}
