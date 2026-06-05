//! Canonical workflows surfaced through `veloq recipes`, per-verb
//! `--help`, and `info`'s `applicable_recipes` block.
//!
//! The registry lives at `src/recipes/registry.toml` and is parsed +
//! validated **at build time** by the crate's `build.rs`. The
//! generated table is included below. No runtime parsing, no
//! `OnceLock`, no TOML crate in the shipped binary — malformed entries
//! fail `cargo build` instead of the first `veloq recipes` call.
//!
//! Schema fields on `Recipe` mirror `build.rs::Recipe`; keep the two
//! shapes in sync if a column is added.

/// One canonical workflow. All fields are static string slices — the
/// generator emits raw string literals so multi-line recipe bodies
/// stay readable and don't need escape rules.
#[derive(Debug, Clone, Copy)]
pub struct Recipe {
    /// kebab-case-slug, unique across the registry. Used as the
    /// addressable handle on `veloq recipes <id>` and as the entry
    /// the `--help` recipe block points at.
    pub id: &'static str,
    /// One-line summary surfaced in the recipe list / verb help.
    pub title: &'static str,
    /// The canonical command(s) the recipe represents. May be
    /// multi-line; agents copy-paste verbatim.
    pub body: &'static str,
    /// Lowercase keyword tokens. Reserved for an optional fuzzy
    /// matcher in the future; today they're metadata only.
    pub keywords: &'static [&'static str],
    /// NSys/NCU verb names the recipe involves. Used by per-verb
    /// `--help` to surface recipes that touch this verb.
    pub related_verbs: &'static [&'static str],
    /// Trace-shape predicates. When non-empty, the recipe is
    /// *applicable* only when every predicate evaluates true against
    /// the trace's [`TraceShape`]. `info` filters by this list to
    /// produce `applicable_recipes`.
    pub trace_shape: &'static [&'static str],
}

/// Snapshot of trace properties the recipe-applicability predicates
/// evaluate against. Populated by `info` / `summary` from
/// the source's `CapabilityFlags` + a few extra counts; this struct
/// stays local to the recipes module so the registry doesn't grow a
/// dependency on the source-specific capability bitmap.
///
#[derive(Debug, Clone, Copy, Default)]
pub struct TraceShape {
    pub has_kernels: bool,
    pub has_memcpy: bool,
    pub has_nvtx: bool,
    pub has_target_info: bool,
    pub multi_device: bool,
    pub multi_process: bool,
    /// CUPTI_ACTIVITY_KIND_GRAPH_TRACE present — wall-only graph
    /// replay rows (`--cuda-graph-trace=graph` capture mode).
    pub has_graph_trace: bool,
    /// CUDA_GRAPH_NODE_EVENTS present — graph replays with full
    /// per-kernel decomposition (`--cuda-graph-trace=node` capture
    /// mode). Mutually exclusive with `has_graph_trace` in practice.
    pub has_graph_nodes: bool,
}

impl Recipe {
    /// Does this recipe apply to a trace with the given shape? Empty
    /// `trace_shape` matches anything; otherwise every predicate must
    /// evaluate true.
    pub fn matches_trace_shape(&self, shape: &TraceShape) -> bool {
        // AND-of-groups, OR-within-group: each `trace_shape` entry is
        // a pipe-separated alternation that must contribute at least
        // one true predicate. A bare predicate (no `|`) is just a
        // one-element group, so the simple case is unchanged.
        self.trace_shape
            .iter()
            .all(|group| group.split('|').any(|p| eval_predicate(p, shape)))
    }
}

/// One predicate name → bool against the shape. The pipe-alternation
/// split happens in [`Recipe::matches_trace_shape`].
/// Unknown predicate names return `false` defensively — `build.rs`
/// validates names at compile time so an unknown here means the
/// matcher and the registry have drifted.
fn eval_predicate(name: &str, shape: &TraceShape) -> bool {
    match name {
        "has_kernels" => shape.has_kernels,
        "has_memcpy" => shape.has_memcpy,
        "has_nvtx" => shape.has_nvtx,
        "has_target_info" => shape.has_target_info,
        "multi_device" => shape.multi_device,
        "multi_process" => shape.multi_process,
        "has_graph_trace" => shape.has_graph_trace,
        "has_graph_nodes" => shape.has_graph_nodes,
        _ => false,
    }
}

// Build-time-generated `pub static RECIPES: &[Recipe] = &[...];`. The
// file is rewritten on every cargo build that touches registry.toml
// (per `cargo:rerun-if-changed` in build.rs).
include!(concat!(env!("OUT_DIR"), "/recipes_generated.rs"));

/// Recipes whose `related_verbs` includes `verb`. Order matches the
/// registry's declaration order.
pub fn recipes_for_verb(verb: &str) -> impl Iterator<Item = &'static Recipe> {
    RECIPES
        .iter()
        .filter(move |r| r.related_verbs.contains(&verb))
}

/// Recipes whose `trace_shape` is satisfied by `shape`. `info`
/// projects this into its `applicable_recipes` block.
pub fn recipes_for_trace_shape(shape: &TraceShape) -> impl Iterator<Item = &'static Recipe> + '_ {
    RECIPES.iter().filter(move |r| r.matches_trace_shape(shape))
}

/// Look up a recipe by its `id`. Used by `veloq recipes <id>`.
pub fn recipe_by_id(id: &str) -> Option<&'static Recipe> {
    RECIPES.iter().find(|r| r.id == id)
}

/// Every registered recipe. Used by `veloq recipes` (list mode).
pub fn all_recipes() -> &'static [Recipe] {
    RECIPES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_is_populated() {
        // The lower bound is loose so future additions don't break the
        // test.
        assert!(
            all_recipes().len() >= 8,
            "expected at least 8 recipes; got {}",
            all_recipes().len(),
        );
    }

    #[test]
    fn recipe_by_id_returns_none_for_unknown_slug() {
        assert!(recipe_by_id("nope-not-a-recipe").is_none());
    }

    #[test]
    fn recipes_for_verb_filters_correctly() {
        // `stats` is referenced by multiple recipes; the exact count is
        // a moving target so just assert >0. The unknown-verb path stays
        // empty.
        assert!(recipes_for_verb("stats").count() > 0);
        assert_eq!(recipes_for_verb("does-not-exist").count(), 0);
    }

    #[test]
    fn recipes_for_trace_shape_filters_correctly() {
        // Default shape: only recipes without `trace_shape` predicates
        // match. The all-bits-set shape matches everything.
        let empty = TraceShape::default();
        let full = TraceShape {
            has_kernels: true,
            has_memcpy: true,
            has_nvtx: true,
            has_target_info: true,
            multi_device: true,
            multi_process: true,
            has_graph_trace: true,
            has_graph_nodes: true,
        };
        let unrestricted = recipes_for_trace_shape(&empty).count();
        let all_match = recipes_for_trace_shape(&full).count();
        assert!(
            all_match >= unrestricted,
            "full-shape match count must dominate the unrestricted count",
        );
        assert_eq!(all_match, all_recipes().len());
    }

    /// Trace-shape matcher is exhaustive over the predicate set —
    /// recipes with no shape constraints match anything.
    #[test]
    fn empty_trace_shape_matches_anything() {
        let r = Recipe {
            id: "x",
            title: "x",
            body: "x",
            keywords: &["x"],
            related_verbs: &["stats"],
            trace_shape: &[],
        };
        assert!(r.matches_trace_shape(&TraceShape::default()));
        assert!(r.matches_trace_shape(&TraceShape {
            has_nvtx: true,
            ..Default::default()
        }));
    }

    /// One row in the predicate exhaustiveness table: the
    /// `trace_shape` slice the recipe carries, and a setter that
    /// flips the matching field on a fresh [`TraceShape`].
    type PredicateCase = (&'static [&'static str], fn(&mut TraceShape));

    /// Pipe-separated OR groups: at least one alternant must hold for
    /// the group to pass. Each top-level entry still combines with
    /// AND, so a recipe carrying two entries — one a singleton, the
    /// other an OR-group — must satisfy both.
    #[test]
    fn or_group_matches_when_any_alternant_holds() {
        let r = Recipe {
            id: "or-x",
            title: "x",
            body: "x",
            keywords: &["x"],
            related_verbs: &["graph-replays"],
            trace_shape: &["has_graph_trace|has_graph_nodes"],
        };
        // Neither flag: refuses.
        assert!(!r.matches_trace_shape(&TraceShape::default()));
        // Only the first alternant: matches.
        assert!(r.matches_trace_shape(&TraceShape {
            has_graph_trace: true,
            ..Default::default()
        }));
        // Only the second: also matches.
        assert!(r.matches_trace_shape(&TraceShape {
            has_graph_nodes: true,
            ..Default::default()
        }));
    }

    #[test]
    fn and_of_or_groups_combines_correctly() {
        // `[A|B, C]` means "(A or B) AND C".
        let r = Recipe {
            id: "and-or",
            title: "x",
            body: "x",
            keywords: &["x"],
            related_verbs: &["graph-replays"],
            trace_shape: &["has_graph_trace|has_graph_nodes", "has_nvtx"],
        };
        assert!(!r.matches_trace_shape(&TraceShape::default()));
        // OR group satisfied but second AND term missing.
        assert!(!r.matches_trace_shape(&TraceShape {
            has_graph_trace: true,
            ..Default::default()
        }));
        // Both terms satisfied.
        assert!(r.matches_trace_shape(&TraceShape {
            has_graph_nodes: true,
            has_nvtx: true,
            ..Default::default()
        }));
    }

    /// Reverse direction: the graph-replay recipes match the right
    /// capture modes. graph-replay-survey accepts
    /// either; graph-replay-hotspots requires node decomposition.
    /// Tested via the same `recipes_for_trace_shape` iterator that
    /// `info` projects into `applicable_recipes`, so a regression in
    /// either the registry or the matcher surfaces here.
    #[test]
    fn graph_replay_recipes_match_their_capture_modes() {
        let ids_for = |shape: &TraceShape| {
            recipes_for_trace_shape(shape)
                .map(|r| r.id)
                .collect::<Vec<_>>()
        };
        let only_trace_mode = TraceShape {
            has_graph_trace: true,
            ..Default::default()
        };
        let only_node_mode = TraceShape {
            has_graph_nodes: true,
            ..Default::default()
        };

        let trace_ids = ids_for(&only_trace_mode);
        assert!(
            trace_ids.contains(&"graph-replay-survey"),
            "graph-replay-survey should match has_graph_trace; got {trace_ids:?}",
        );
        assert!(
            !trace_ids.contains(&"graph-replay-hotspots"),
            "graph-replay-hotspots must NOT match has_graph_trace alone; got {trace_ids:?}",
        );

        let node_ids = ids_for(&only_node_mode);
        assert!(
            node_ids.contains(&"graph-replay-survey"),
            "graph-replay-survey should match has_graph_nodes; got {node_ids:?}",
        );
        assert!(
            node_ids.contains(&"graph-replay-hotspots"),
            "graph-replay-hotspots should match has_graph_nodes; got {node_ids:?}",
        );

        let none_ids = ids_for(&TraceShape::default());
        assert!(!none_ids.contains(&"graph-replay-survey"));
        assert!(!none_ids.contains(&"graph-replay-hotspots"));
    }

    #[test]
    fn all_predicates_evaluate() {
        // Static slices keep the 'static-bound `trace_shape` field
        // satisfied without ceremony. One row per predicate; if a
        // predicate name is added to the matcher, add a row here too.
        const ALL: &[PredicateCase] = &[
            (&["has_kernels"], |s| s.has_kernels = true),
            (&["has_memcpy"], |s| s.has_memcpy = true),
            (&["has_nvtx"], |s| s.has_nvtx = true),
            (&["has_target_info"], |s| s.has_target_info = true),
            (&["multi_device"], |s| s.multi_device = true),
            (&["multi_process"], |s| s.multi_process = true),
            (&["has_graph_trace"], |s| s.has_graph_trace = true),
            (&["has_graph_nodes"], |s| s.has_graph_nodes = true),
        ];
        for (trace_shape, set) in ALL {
            let mut shape = TraceShape::default();
            set(&mut shape);
            let r = Recipe {
                id: "x",
                title: "x",
                body: "x",
                keywords: &["x"],
                related_verbs: &["stats"],
                trace_shape,
            };
            assert!(
                r.matches_trace_shape(&shape),
                "predicate `{trace_shape:?}` failed against {shape:?}",
            );
        }
    }
}
