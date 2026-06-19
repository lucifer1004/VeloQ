//! Meta verbs — CLI-owned subcommands that don't belong to any one
//! profile source. They sit at the same level as the source-
//! contributed verbs (`veloq info`, `veloq sources` alongside the
//! hoisted `veloq stats`, `veloq summary`, etc.).
//!
//! `info` and `clean` inspect one path; `sources` lists every
//! registered source's identity. Meta verbs stay JSON-only — the
//! responses are tiny and CSV / table flatteners would buy nothing.

pub mod agent;
pub mod clean;
pub mod error;
pub mod info;
pub mod recipes;
pub mod self_update;
pub mod sources;

use clap::{ArgMatches, Command};
use veloq_core::{
    EnvelopeError, EnvelopeTraceRef, OutputFormat, ProfileSource, ResponseMeta, SourceRef,
};

pub use error::{MetaError, MetaResult};

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
        "agent" | "info" | "sources" | "clean" | "recipes" | "self-update"
    )
}

/// Build the meta-verb clap subtrees so the binary can graft them
/// onto its top-level `Command`.
pub fn cli() -> Vec<Command> {
    vec![
        agent::cli(),
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
pub fn run(
    verb: &str,
    matches: &ArgMatches,
    sources: &[Box<dyn ProfileSource>],
    fmt: OutputFormat,
) -> MetaResult<i32> {
    match verb {
        "agent" => agent::run(matches, fmt),
        "clean" => clean::run(matches, sources, fmt),
        "info" => info::run(matches, sources, fmt),
        "sources" => sources::run(matches, sources, fmt),
        "recipes" => recipes::run(matches, fmt),
        "self-update" => self_update::run(matches, fmt),
        other => Err(MetaError::UnknownVerb {
            verb: other.to_string(),
        }),
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
) -> MetaResult<()> {
    veloq_core::emit_envelope(META_SOURCE, verb.to_string(), trace, None, meta, data)
        .map_err(|source| MetaError::SerializeEnvelope { source })?;
    Ok(())
}

/// Emit the success envelope, or — on serialization failure — the error
/// envelope, returning the process exit code (0 ok / 1 failed). Collapses
/// the emit-then-error-fallback tail every meta verb repeated; owns the
/// `trace` clone the error arm needs.
pub(crate) fn emit_or_error<T: serde::Serialize>(
    fmt: OutputFormat,
    verb: &'static str,
    trace: Option<EnvelopeTraceRef>,
    meta: Option<ResponseMeta>,
    data: T,
) -> i32 {
    if let Err(err) = emit_meta_envelope(verb, trace.clone(), meta, data) {
        emit_meta_error(fmt, verb, trace, &err);
        return 1;
    }
    0
}

/// Mirror of `veloq_nsys::output::emit_error` for meta-verb failures.
/// stdout carries the structured envelope; stderr carries the
/// human-readable mirror only for non-JSON requests.
pub(crate) fn emit_meta_error<E>(
    fmt: OutputFormat,
    verb: &'static str,
    trace: Option<EnvelopeTraceRef>,
    err: &E,
) where
    E: veloq_core::VeloqDiagnostic,
{
    let env =
        EnvelopeError::from_diagnostic(Some(META_SOURCE), Some(verb.to_string()), trace, None, err);
    if !matches!(fmt, OutputFormat::Json) {
        eprintln!("veloq: {err}");
    }
    if let Ok(s) = env.to_json_pretty() {
        println!("{s}");
    }
}
