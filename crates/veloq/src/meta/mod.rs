//! Meta verbs — CLI-owned subcommands that don't belong to any one
//! profile source. They sit at the same level as the source-
//! contributed verbs (`veloq info`, `veloq sources` alongside the
//! hoisted `veloq stats`, `veloq summary`, etc.).
//!
//! `info` and `clean` inspect one path; `sources` lists every
//! registered source's identity. Meta verbs stay JSON-only — the
//! responses are tiny and CSV / table flatteners would buy nothing.

pub mod clean;
pub mod info;
pub mod recipes;
pub mod self_update;
pub mod sources;

use anyhow::Result;
use clap::{ArgMatches, Command};
use veloq_core::{EnvelopeError, EnvelopeTraceRef, ProfileSource, ResponseMeta, SourceRef};

/// Source identity emitted on every meta envelope. The binary
/// surfaces *itself* as the source for meta verbs (no profile
/// backend is involved); agents can distinguish meta responses by
/// `source.kind == "veloq"`.
pub const META_SOURCE: SourceRef = SourceRef {
    kind: "veloq",
    version: env!("CARGO_PKG_VERSION"),
};

/// Is `verb` a meta-verb name? Used by the top-level dispatcher to
/// short-circuit before falling through to per-source dispatch.
pub fn is_meta(verb: &str) -> bool {
    matches!(
        verb,
        "info" | "sources" | "clean" | "recipes" | "self-update"
    )
}

/// Build the meta-verb clap subtrees so the binary can graft them
/// onto its top-level `Command`.
pub fn cli() -> Vec<Command> {
    vec![
        info::cli(),
        sources::cli(),
        clean::cli(),
        recipes::cli(),
        self_update::cli(),
    ]
}

/// Route a meta-verb dispatch. Mirrors `ProfileSource::run`'s
/// contract: `Ok(0)` on success (envelope already on stdout),
/// `Ok(1)` when the verb failed and emitted its own `EnvelopeError`,
/// `Err(_)` on top-level glue failures (dispatcher emits a
/// CLI-level envelope).
pub fn run(verb: &str, matches: &ArgMatches, sources: &[Box<dyn ProfileSource>]) -> Result<i32> {
    match verb {
        "clean" => clean::run(matches, sources),
        "info" => info::run(matches, sources),
        "sources" => sources::run(matches, sources),
        "recipes" => recipes::run(matches),
        "self-update" => self_update::run(matches),
        other => anyhow::bail!("unknown meta verb `{other}`"),
    }
}

/// Wrap `data` in the envelope with `source = veloq` and
/// `command = <verb>` (unqualified — the source-kind already says
/// it's a meta response) and pretty-print to stdout. JSON-only,
/// per the meta-verb contract. `trace_span` is always `None` because
/// meta verbs don't open the trace deeply enough to compute it.
pub(crate) fn emit_meta_envelope<T: serde::Serialize>(
    verb: &'static str,
    trace: Option<EnvelopeTraceRef>,
    meta: Option<ResponseMeta>,
    data: T,
) -> Result<()> {
    veloq_core::emit_envelope(META_SOURCE, verb.to_string(), trace, None, meta, data)?;
    Ok(())
}

/// Emit the success envelope, or — on serialization failure — the error
/// envelope, returning the process exit code (0 ok / 1 failed). Collapses
/// the emit-then-error-fallback tail every meta verb repeated; owns the
/// `trace` clone the error arm needs.
pub(crate) fn emit_or_error<T: serde::Serialize>(
    verb: &'static str,
    trace: Option<EnvelopeTraceRef>,
    meta: Option<ResponseMeta>,
    data: T,
) -> i32 {
    if let Err(err) = emit_meta_envelope(verb, trace.clone(), meta, data) {
        emit_meta_error(verb, trace, &err);
        return 1;
    }
    0
}

/// Mirror of `veloq_nsys::output::emit_error` for meta-verb failures.
/// Dual-channel policy: see `write_error_envelope`.
pub(crate) fn emit_meta_error(
    verb: &'static str,
    trace: Option<EnvelopeTraceRef>,
    err: &anyhow::Error,
) {
    let env =
        EnvelopeError::from_anyhow(Some(META_SOURCE), Some(verb.to_string()), trace, None, err);
    eprintln!("veloq: {err:#}");
    if let Ok(s) = env.to_json_pretty() {
        println!("{s}");
    }
}
