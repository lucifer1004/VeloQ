//! `veloq sources` — list registered profile sources.
//!
//! Tiny utility verb: no trace input, no flags, just a JSON dump
//! of `{ kind, version }` per registered source plus the binary's
//! own version. Useful for agents probing what verbs they can run
//! and what wire-format version each source emits.
//!
//! list-payload contract — even meta verbs canonicalise list payloads as
//! `data: { count, total_matched, rows: [{ key, ... }], auxiliary }`.
//! The binary's own `veloq_version` lives in `auxiliary` since it
//! isn't a per-row attribute.

use clap::{ArgMatches, Command};
use serde::Serialize;
use veloq_core::{OutputFormat, ProfileSource};

use super::{META_SOURCE, MetaResult, emit_or_error};

const VERB: &str = "sources";

#[derive(Serialize)]
struct SourcesPayload {
    count: usize,
    total_matched: usize,
    rows: Vec<SourceRow>,
    auxiliary: SourcesAuxiliary,
}

#[derive(Serialize)]
struct SourceRow {
    /// cross-trace key. `source:<kind>`.
    key: String,
    kind: &'static str,
    version: &'static str,
}

#[derive(Serialize)]
struct SourcesAuxiliary {
    /// Version of the `veloq` binary itself, independent of any
    /// per-source wire-format version. Agents reading this can pin
    /// to a binary version separately from a payload version.
    veloq_version: &'static str,
}

pub fn cli() -> Command {
    Command::new(VERB).about("List registered profile sources and their wire-format versions")
}

pub fn run(
    _matches: &ArgMatches,
    sources: &[Box<dyn ProfileSource>],
    fmt: OutputFormat,
) -> MetaResult<i32> {
    let rows: Vec<SourceRow> = sources
        .iter()
        .map(|s| SourceRow {
            key: format!("source:{}", s.kind()),
            kind: s.kind(),
            version: s.version(),
        })
        .collect();
    let count = rows.len();
    let payload = SourcesPayload {
        count,
        total_matched: count,
        rows,
        auxiliary: SourcesAuxiliary {
            veloq_version: META_SOURCE.version,
        },
    };
    Ok(emit_or_error(fmt, VERB, None, None, payload))
}
