//! Output dispatch — JSON envelope emit, CSV / table rendering,
//! and the error-envelope contract that lets every veloq failure
//! travel through the same `{schema, source, command, trace, error}`
//! shape agents parse on success.
//!
//! Three classes of caller end up here:
//!
//! - [`render`] — called from per-command dispatch on success; picks
//!   JSON (envelope-wrapped) / CSV / table from the user's `--format`
//!   flag and tucks the per-command [`TabularView`] flattener in via
//!   a closure.
//! - [`emit_error`] — called by the source on command failure or
//!   format-flag failure with the in-flight `anyhow::Error`.
//! - [`emit_parse_error`] — called by the binary on clap parse
//!   failure. Special-cases `--help` / `--version` and otherwise
//!   emits the envelope without `command` / `trace` (parsing didn't
//!   get that far).

use anyhow::Result;
use std::path::Path;
use veloq_core::{EnvelopeError, EnvelopeTraceRef, ResponseMeta, SourceRef, TraceSpan};

use crate::format::{Format, TabularView, emit_csv, emit_table};
use crate::source::NsysSource;

/// Build the `EnvelopeTraceRef` every NSys verb emits. `path` is
/// stringified once here so the JSON path matches what's printed on
/// the CSV / table headers without per-caller drift.
fn nsys_trace_ref(trace: &Path) -> EnvelopeTraceRef {
    EnvelopeTraceRef {
        kind: NsysSource::KIND,
        path: trace.display().to_string(),
    }
}

fn nsys_source_ref() -> SourceRef {
    SourceRef {
        kind: NsysSource::KIND,
        version: NsysSource::VERSION,
    }
}

/// Wrap `data` in the [`veloq_core::Envelope`] (schema, source, qualified
/// command, trace, trace_span) and pretty-print it to stdout.
/// JSON-only — CSV / table render through [`render`].
///
/// `meta = None` is the default for verbs that don't carry envelope
/// metadata. List verbs that accept scope filters should call
/// [`emit_with_meta`] with `Some(ResponseMeta { applied_scope:
/// Some(...), .. })`.
pub fn emit<T: serde::Serialize>(
    trace: &Path,
    trace_span: Option<TraceSpan>,
    verb: &str,
    data: T,
) -> Result<()> {
    emit_with_meta(trace, trace_span, verb, None, data)
}

/// Same as [`emit`] but populates the envelope's `meta` block. Used by
/// list verbs that resolve scope (see
/// `crates/nsys/veloq-nsys-data/src/scope.rs::resolve_scope`).
pub fn emit_with_meta<T: serde::Serialize>(
    trace: &Path,
    trace_span: Option<TraceSpan>,
    verb: &str,
    meta: Option<ResponseMeta>,
    data: T,
) -> Result<()> {
    // `compute_trace_span` ran pre-dispatch and only consulted an
    // existing `<trace>.veloq/meta.bin`. Verbs that build the cache (summary,
    // stats, search, …) populate it during their work; re-read here
    // so cold-trace envelopes still emit a populated `trace_span`.
    // Sub-ms when the sidecar exists; harmless `None` when it
    // doesn't.
    let trace_span = trace_span.or_else(|| veloq_nsys_data::meta_cache::trace_span_for_path(trace));
    veloq_core::emit_envelope(
        nsys_source_ref(),
        format!("{}.{verb}", NsysSource::KIND),
        Some(nsys_trace_ref(trace)),
        trace_span,
        meta,
        data,
    )?;
    Ok(())
}

/// Same as [`emit`] but with no trace — for meta verbs like
/// `schema` that don't read a trace. The envelope's `trace` and
/// `trace_span` fields are omitted entirely (rather than emitted with
/// an empty path / zero span).
pub fn emit_meta<T: serde::Serialize>(verb: &str, data: T) -> Result<()> {
    veloq_core::emit_envelope(
        nsys_source_ref(),
        format!("{}.{verb}", NsysSource::KIND),
        None,
        None,
        None,
        data,
    )?;
    Ok(())
}

/// One-shot output dispatcher: JSON keeps the envelope; CSV / table
/// run the caller-supplied `view_fn` to flatten the response into a
/// `TabularView` and render accordingly. The `view_fn` argument is
/// what lets each command tailor its tabular view without this
/// helper having to know per-command structure.
///
/// Verbs that resolve scope should call [`render_with_meta`] so the
/// envelope's `meta.applied_scope` is populated.
pub fn render<T, F>(
    fmt: Format,
    trace: &Path,
    trace_span: Option<TraceSpan>,
    verb: &str,
    data: T,
    view_fn: F,
) -> Result<()>
where
    T: serde::Serialize,
    F: FnOnce(&T) -> TabularView,
{
    render_with_meta(fmt, trace, trace_span, verb, None, data, view_fn)
}

/// Same as [`render`] but populates the envelope's `meta` block on the
/// JSON path. The CSV / table paths ignore `meta` — they project
/// `data` directly via `view_fn` and the meta block has no tabular
/// equivalent.
pub fn render_with_meta<T, F>(
    fmt: Format,
    trace: &Path,
    trace_span: Option<TraceSpan>,
    verb: &str,
    meta: Option<ResponseMeta>,
    data: T,
    view_fn: F,
) -> Result<()>
where
    T: serde::Serialize,
    F: FnOnce(&T) -> TabularView,
{
    let qualified = format!("{}.{verb}", NsysSource::KIND);
    let trace_str = trace.display().to_string();
    match fmt {
        Format::Json => emit_with_meta(trace, trace_span, verb, meta, data),
        Format::Csv => emit_csv(&view_fn(&data), &qualified, &trace_str),
        Format::Table => emit_table(&view_fn(&data), &qualified, &trace_str),
    }
}

/// Emit a structured ambiguity-refusal error envelope. The error
/// envelope carries `meta.warnings` with the
/// `multi-device-ambiguous` code so the agent reads the refusal in
/// structured form (rather than parsing the human message).
pub fn emit_ambiguity_error(
    verb: &str,
    trace: &Path,
    trace_span: Option<TraceSpan>,
    err: &veloq_nsys_data::scope::AmbiguityError,
    fmt: Format,
) {
    let qualified = format!("{}.{verb}", NsysSource::KIND);
    let trace_ref = Some(nsys_trace_ref(trace));
    let meta = Some(ResponseMeta {
        warnings: vec![err.warning.clone()],
        ..ResponseMeta::default()
    });
    let env = EnvelopeError::new(
        Some(nsys_source_ref()),
        Some(qualified),
        trace_ref,
        trace_span,
        meta,
        err.message.clone(),
        Vec::new(),
    );
    if !matches!(fmt, Format::Json) {
        eprintln!("veloq: {}", err.message);
    }
    if let Ok(s) = env.to_json_pretty() {
        println!("{s}");
    }
}

/// Shim around [`veloq_core::write_error_envelope`] that takes the
/// NSys-typed trace path. Centralizes the
/// `Option<&Path> -> Option<EnvelopeTraceRef>` projection so the
/// `commands::run` arms stay terse.
///
/// `trace` is `None` for trace-less verbs (`Schema`) — the envelope's
/// `trace` field is then omitted entirely rather than fabricated with
/// an empty path. `trace_span` follows the same Some-iff-trace-was-read
/// contract: a verb that failed before opening the trace gets `None`.
pub fn emit_error(
    verb: &str,
    trace: Option<&Path>,
    trace_span: Option<TraceSpan>,
    err: &anyhow::Error,
    fmt: Format,
) {
    veloq_core::write_error_envelope(
        nsys_source_ref(),
        verb,
        trace.map(nsys_trace_ref),
        trace_span,
        err,
        fmt,
    );
}

/// Emit a JSON error envelope for a clap parse failure.
///
/// `--help` / `--version` are special-cased: clap models them as
/// `Err` but the user's intent was clearly informational, so we let
/// clap print its native output and skip the JSON envelope.
/// Everything else — including the missing-subcommand case
/// (`veloq` with no args) — routes through the envelope so agents
/// have one parsing contract. `source` / `command` / `trace` are
/// omitted because parsing didn't get far enough to know which
/// subcommand the user wanted.
pub fn emit_parse_error(err: &clap::Error, fmt: Format) {
    use clap::error::ErrorKind;
    if matches!(
        err.kind(),
        ErrorKind::DisplayHelp | ErrorKind::DisplayVersion
    ) {
        let _ = err.print();
        return;
    }
    let message = parse_error_message(err);
    let env = EnvelopeError::new(
        None,
        None,
        None,
        None,
        None,
        message.clone(),
        vec![format!("clap::ErrorKind::{:?}", err.kind())],
    );
    if !matches!(fmt, Format::Json) {
        eprintln!("veloq: {message}");
    }
    if let Ok(s) = env.to_json_pretty() {
        println!("{s}");
    }
}

/// Pick the human-readable message for a clap parse failure. Carved
/// out of [`emit_parse_error`] so the routing concern (help vs.
/// error) and the message-extraction concern stay separately
/// testable.
fn parse_error_message(err: &clap::Error) -> String {
    use clap::error::ErrorKind;
    match err.kind() {
        // Missing-subcommand renders as the full help text — useful
        // on a TTY but noisy in JSON. Substitute a short message;
        // agents that want the full usage can run `veloq --help`
        // explicitly.
        ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand => {
            "missing subcommand (run `veloq --help` for usage)".to_string()
        }
        _ => strip_ansi(&err.render().to_string()).trim().to_string(),
    }
}

/// Strip ANSI escape sequences so the JSON payload stays plain
/// text. Clap colourises errors when stderr is a TTY; we don't
/// want the bytes in the wire format. Only handles CSI sequences
/// (`ESC [ ... letter`), which is what clap emits.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' && chars.peek() == Some(&'[') {
            chars.next();
            for nc in chars.by_ref() {
                if nc.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}
