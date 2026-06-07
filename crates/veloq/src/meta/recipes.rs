//! `veloq recipes` — list canonical workflows from the compiled-in
//! registry (`crates/veloq-core/src/recipes/registry.toml`).
//!
//! The verb lists all recipes when called with no positional
//! argument, and shows one recipe's full body when called with a
//! recipe id. JSON-only — the responses are small and the registry is
//! the SSOT for per-verb `--help` recipe injection plus `info`'s
//! `applicable_recipes` field.

use clap::{Arg, ArgMatches, Command};
use serde::Serialize;

use super::{MetaError, MetaResult, emit_meta_error, emit_or_error};
use veloq_core::{OutputFormat, recipes};

const VERB: &str = "recipes";

#[derive(Serialize)]
#[serde(untagged)]
enum RecipesPayload {
    List(RecipesListPayload),
    Show(RecipeDetailPayload),
}

#[derive(Serialize)]
struct RecipesListPayload {
    count: usize,
    rows: Vec<RecipeSummary>,
}

#[derive(Serialize)]
struct RecipeSummary {
    /// cross-trace key — equal to the recipe id so agents can
    /// `INDEX(.rows; .key)` against this list.
    key: &'static str,
    id: &'static str,
    title: &'static str,
    related_verbs: &'static [&'static str],
    /// Trace-shape predicates this recipe gates on. Empty when the
    /// recipe applies to every trace.
    trace_shape: &'static [&'static str],
}

#[derive(Serialize)]
struct RecipeDetailPayload {
    key: &'static str,
    id: &'static str,
    title: &'static str,
    body: &'static str,
    keywords: &'static [&'static str],
    related_verbs: &'static [&'static str],
    trace_shape: &'static [&'static str],
}

pub fn cli() -> Command {
    Command::new(VERB)
        .about("List or show canonical veloq workflows (recipes)")
        .arg(Arg::new("id").value_name("ID").required(false).help(
            "Recipe id (kebab-case slug). When omitted, every registered \
                     recipe is listed as a summary row.",
        ))
}

pub fn run(matches: &ArgMatches, fmt: OutputFormat) -> MetaResult<i32> {
    match matches.get_one::<String>("id") {
        Some(id) => show_recipe(id, fmt),
        None => list_recipes(fmt),
    }
}

fn list_recipes(fmt: OutputFormat) -> MetaResult<i32> {
    let rows: Vec<RecipeSummary> = recipes::all_recipes()
        .iter()
        .map(|r| RecipeSummary {
            key: r.id,
            id: r.id,
            title: r.title,
            related_verbs: r.related_verbs,
            trace_shape: r.trace_shape,
        })
        .collect();
    let payload = RecipesPayload::List(RecipesListPayload {
        count: rows.len(),
        rows,
    });
    Ok(emit_or_error(fmt, VERB, None, None, payload))
}

fn show_recipe(id: &str, fmt: OutputFormat) -> MetaResult<i32> {
    let Some(recipe) = recipes::recipe_by_id(id) else {
        let err = MetaError::UnknownRecipe { id: id.to_string() };
        emit_meta_error(fmt, VERB, None, &err);
        return Ok(1);
    };
    let payload = RecipesPayload::Show(RecipeDetailPayload {
        key: recipe.id,
        id: recipe.id,
        title: recipe.title,
        body: recipe.body,
        keywords: recipe.keywords,
        related_verbs: recipe.related_verbs,
        trace_shape: recipe.trace_shape,
    });
    Ok(emit_or_error(fmt, VERB, None, None, payload))
}
