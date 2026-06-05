//! Source-qualified JSON envelope every veloq subcommand wraps its
//! response (and every CLI error) in.
//!
//! The envelope-level [`ResponseMeta`] block carries `applied_scope`
//! (uniform scope echo), `next_steps` (per-verb follow-up hints), and
//! `warnings` (silent-failure guardrail notices). The
//! meta block is the structural location every list verb populates
//! with the same shape, so an agent reads one address regardless of
//! source. Multi-device refusal: see the `meta` module.
//!
//! The envelope-level [`TraceSpan`] (`{origin_ns, span_ns}`) is the
//! per-second normalisation denominator for cross-capture diffs.

use crate::meta::ResponseMeta;
use serde::Serialize;

/// Wire-format version emitted on `Envelope.schema`. Bumps with the
/// envelope itself (e.g. adding a top-level field). Source-specific
/// shape changes belong in [`SourceRef::version`] instead.
pub const ENVELOPE_VERSION: &str = "v1";

/// Trace-wide origin + span an agent uses as the normalization
/// denominator. `origin_ns` is the primary-table `MIN(start)`
/// (`summary.primary_time_range_ns.start`); `span_ns` is the same
/// table's `MAX(end) - MIN(start)`. Both are absolute nanoseconds.
///
/// Present when a source can compute a trace-wide normalization
/// window. Absent for sources without that concept, for meta verbs
/// (`schema`, `sources`, `info`), and for errors that fired before the
/// source opened the trace.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct TraceSpan {
    pub origin_ns: i64,
    pub span_ns: i64,
}

/// Source-qualified success envelope. Generic in `T` so each
/// subcommand emits its strongly-typed response struct directly;
/// agents can deserialize into a known shape after reading `command`.
#[derive(Debug, Serialize)]
pub struct Envelope<T: Serialize> {
    pub schema: &'static str,
    pub source: SourceRef,
    /// Qualified verb — `<source>.<verb>` for source-owned commands
    /// (`nsys.stats`, `ncu.summary`), or just `<verb>` for meta verbs
    /// (`info`, `sources`, `clean`).
    pub command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace: Option<EnvelopeTraceRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_span: Option<TraceSpan>,
    /// Envelope metadata block: `applied_scope`,
    /// `next_steps`, `warnings`. Omitted from serialised JSON when
    /// every sub-field is empty. Required (non-None) on success
    /// envelopes from every list verb that accepts scope filters —
    /// `wire_format_smoke` enforces structurally.
    #[serde(skip_serializing_if = "meta_is_absent")]
    pub meta: Option<ResponseMeta>,
    pub data: T,
}

/// Serde filter — omit `meta` from the wire when it is None or has
/// every sub-field empty.
fn meta_is_absent(m: &Option<ResponseMeta>) -> bool {
    match m {
        None => true,
        Some(m) => m.is_empty(),
    }
}

/// Identifies which profile backend produced the response. `version`
/// bumps independently per-source so adding a field to `nsys.stats`
/// rolls `nsys.version` without affecting `ncu.version`.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct SourceRef {
    pub kind: &'static str,
    pub version: &'static str,
}

/// Trace artifact reference. `kind` matches the producing
/// [`SourceRef::kind`] except for meta verbs that accept a trace
/// without specifying a source, where it carries the auto-detected
/// kind.
#[derive(Debug, Clone, Serialize)]
pub struct EnvelopeTraceRef {
    pub kind: &'static str,
    pub path: String,
}

/// Error variant of [`Envelope`]. `source`, `command`, and `trace`
/// are optional because some errors fire before dispatch knows which
/// source/verb the user wanted (clap parse failures, unknown
/// subcommand): those fields are absent and the agent should treat
/// the failure as CLI-level rather than command-level.
///
/// `meta` is present only when the error itself carries
/// scope-resolution context (e.g. an ambiguity refusal
/// populates `meta.applied_scope.device: null` plus a
/// `multi-device-ambiguous` warning so the agent can read the refusal
/// in structured form). For all other error paths, `meta` is omitted.
#[derive(Debug, Serialize)]
pub struct EnvelopeError {
    pub schema: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace: Option<EnvelopeTraceRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_span: Option<TraceSpan>,
    #[serde(skip_serializing_if = "meta_is_absent")]
    pub meta: Option<ResponseMeta>,
    pub error: EnvelopeErrorDetails,
}

#[derive(Debug, Serialize)]
pub struct EnvelopeErrorDetails {
    /// User-facing summary — what an agent should surface first.
    /// Mirrors `anyhow::Error`'s `to_string()`, i.e. the outermost
    /// context line.
    pub message: String,
    /// `anyhow::Error::chain()` from outermost wrapper inwards, with
    /// the head (which equals `message`) elided. Empty when there are
    /// no `context` layers.
    pub chain: Vec<String>,
}

impl<T: Serialize> Envelope<T> {
    /// Construct an envelope. `meta = None` is the default for meta
    /// verbs and other paths that don't carry envelope metadata. List
    /// verbs that accept scope filters MUST pass `Some(ResponseMeta {
    /// applied_scope: Some(...), .. })` on success.
    pub fn new(
        source: SourceRef,
        command: impl Into<String>,
        trace: Option<EnvelopeTraceRef>,
        trace_span: Option<TraceSpan>,
        meta: Option<ResponseMeta>,
        data: T,
    ) -> Self {
        Self {
            schema: ENVELOPE_VERSION,
            source,
            command: command.into(),
            trace,
            trace_span,
            meta,
            data,
        }
    }

    /// Serialize to a single-line JSON string. Agents consume one
    /// document per CLI invocation, so a newline-terminated dump is
    /// the default; pretty-printing is reserved for human inspection.
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string(self)
    }

    /// Pretty (indented) serialization. Use for human inspection only.
    pub fn to_json_pretty(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }
}

/// Build the success envelope and print it to stdout. The single
/// construct-and-print point every source's emit path shares (NSys
/// `output::emit*`, NCU `source`), so the schema/version/qualified-
/// command wrapping + the `to_json_pretty` + `println!` tail live once.
/// Sources keep their own trace-span resolution and
/// pass the result in.
pub fn emit_envelope<T: Serialize>(
    source: SourceRef,
    command: impl Into<String>,
    trace: Option<EnvelopeTraceRef>,
    trace_span: Option<TraceSpan>,
    meta: Option<ResponseMeta>,
    data: T,
) -> serde_json::Result<()> {
    let env = Envelope::new(source, command, trace, trace_span, meta, data);
    println!("{}", env.to_json_pretty()?);
    Ok(())
}

impl EnvelopeError {
    /// Construct an error envelope. `meta = None` is the default; a
    /// scope-ambiguity refusal passes
    /// `Some(ResponseMeta { warnings: vec![...], .. })` so the agent
    /// can read the refusal in structured form.
    pub fn new(
        source: Option<SourceRef>,
        command: Option<String>,
        trace: Option<EnvelopeTraceRef>,
        trace_span: Option<TraceSpan>,
        meta: Option<ResponseMeta>,
        message: impl Into<String>,
        chain: Vec<String>,
    ) -> Self {
        Self {
            schema: ENVELOPE_VERSION,
            source,
            command,
            trace,
            trace_span,
            meta,
            error: EnvelopeErrorDetails {
                message: message.into(),
                chain,
            },
        }
    }

    /// Project an [`anyhow::Error`] into an error envelope. The error's
    /// top-level rendering lands in `error.message`; every context
    /// layer below it (i.e. `chain().skip(1)`) becomes a chain entry.
    pub fn from_anyhow(
        source: Option<SourceRef>,
        command: Option<String>,
        trace: Option<EnvelopeTraceRef>,
        trace_span: Option<TraceSpan>,
        err: &anyhow::Error,
    ) -> Self {
        let message = err.to_string();
        let chain: Vec<String> = err.chain().skip(1).map(|c| c.to_string()).collect();
        Self::new(source, command, trace, trace_span, None, message, chain)
    }

    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string(self)
    }

    pub fn to_json_pretty(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }
}

/// Write a source-level error envelope.
///
/// In `--format=csv` / `--format=table` (human-targeted), a one-line
/// `veloq: <error>` mirror goes to stderr and the structured
/// [`EnvelopeError`] JSON goes to stdout. In `--format=json` (the
/// agent contract, also the default) the stderr line is *suppressed*:
/// the JSON envelope on stdout is the single source of truth and
/// agents shouldn't have to dedupe a "veloq: …" line that says the
/// same thing.
///
/// The qualified command is built as `"<source.kind>.<verb>"`; pass
/// `verb` as the bare verb name (`"stats"`, `"summary"`). Source
/// authors should call this from their [`crate::ProfileSource::run`]
/// impl when a verb fails, then return `Ok(1)` so the caller knows
/// the envelope is already on the wire.
///
/// JSON serialization errors are swallowed deliberately: if we
/// already failed to render the user's actual error, raising a second
/// error from the renderer is strictly less informative.
pub fn write_error_envelope(
    source: SourceRef,
    verb: &str,
    trace: Option<EnvelopeTraceRef>,
    trace_span: Option<TraceSpan>,
    err: &anyhow::Error,
    fmt: crate::OutputFormat,
) {
    let env = EnvelopeError::from_anyhow(
        Some(source),
        Some(format!("{}.{verb}", source.kind)),
        trace,
        trace_span,
        err,
    );
    if !matches!(fmt, crate::OutputFormat::Json) {
        eprintln!("veloq: {err:#}");
    }
    if let Ok(s) = env.to_json_pretty() {
        println!("{s}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn source() -> SourceRef {
        SourceRef {
            kind: "nsys",
            version: "v0",
        }
    }

    fn trace() -> EnvelopeTraceRef {
        EnvelopeTraceRef {
            kind: "nsys",
            path: "/tmp/trace.nsys-rep".into(),
        }
    }

    /// Pull a value by JSON Pointer, erroring out if the path isn't
    /// present. Tests use this rather than `value["foo"]` indexing
    /// because the workspace lints deny panicking index ops.
    fn at<'a>(v: &'a Value, ptr: &str) -> anyhow::Result<&'a Value> {
        v.pointer(ptr)
            .ok_or_else(|| anyhow::anyhow!("missing pointer `{ptr}` in {v}"))
    }

    fn at_str<'a>(v: &'a Value, ptr: &str) -> anyhow::Result<&'a str> {
        at(v, ptr)?
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("not a string at `{ptr}` in {v}"))
    }

    fn span() -> TraceSpan {
        TraceSpan {
            origin_ns: 1_000_000,
            span_ns: 5_000_000_000,
        }
    }

    #[test]
    fn envelope_shape_has_qualified_command_and_source() -> anyhow::Result<()> {
        let env = Envelope::new(
            source(),
            "nsys.stats",
            Some(trace()),
            Some(span()),
            None,
            serde_json::json!({"rows": []}),
        );
        let v: Value = serde_json::from_str(&env.to_json()?)?;
        assert_eq!(at_str(&v, "/schema")?, "v1");
        assert_eq!(at_str(&v, "/source/kind")?, "nsys");
        assert_eq!(at_str(&v, "/source/version")?, "v0");
        assert_eq!(at_str(&v, "/command")?, "nsys.stats");
        assert_eq!(at_str(&v, "/trace/kind")?, "nsys");
        assert_eq!(at_str(&v, "/trace/path")?, "/tmp/trace.nsys-rep");
        assert_eq!(at(&v, "/trace_span/origin_ns")?.as_i64(), Some(1_000_000));
        assert_eq!(at(&v, "/trace_span/span_ns")?.as_i64(), Some(5_000_000_000));
        assert!(
            v.get("meta").is_none(),
            "meta with None payload must be omitted: {v}"
        );
        assert!(at(&v, "/data/rows")?.is_array());
        Ok(())
    }

    #[test]
    fn envelope_trace_and_span_omitted_when_none() -> anyhow::Result<()> {
        let env = Envelope::new(source(), "sources", None, None, None, serde_json::json!({}));
        let v: Value = serde_json::from_str(&env.to_json()?)?;
        assert!(v.get("trace").is_none(), "got: {v}");
        assert!(v.get("trace_span").is_none(), "got: {v}");
        assert!(v.get("meta").is_none(), "got: {v}");
        Ok(())
    }

    #[test]
    fn envelope_meta_serialises_when_applied_scope_set() -> anyhow::Result<()> {
        use crate::meta::{AppliedScope, ResponseMeta};
        let meta = ResponseMeta {
            applied_scope: Some(AppliedScope {
                device: Some(0),
                kind: Some("kernel".into()),
                ..AppliedScope::default()
            }),
            ..ResponseMeta::default()
        };
        let env = Envelope::new(
            source(),
            "nsys.stats",
            Some(trace()),
            Some(span()),
            Some(meta),
            serde_json::json!({"rows": []}),
        );
        let v: Value = serde_json::from_str(&env.to_json()?)?;
        assert_eq!(at(&v, "/meta/applied_scope/device")?.as_i64(), Some(0));
        assert_eq!(at_str(&v, "/meta/applied_scope/kind")?, "kernel");
        // Empty sub-fields stay absent on the wire.
        assert!(v.pointer("/meta/next_steps").is_none(), "got: {v}");
        assert!(v.pointer("/meta/warnings").is_none(), "got: {v}");
        Ok(())
    }

    #[test]
    fn envelope_meta_omitted_when_every_subfield_empty() -> anyhow::Result<()> {
        use crate::meta::ResponseMeta;
        let env = Envelope::new(
            source(),
            "nsys.stats",
            Some(trace()),
            Some(span()),
            Some(ResponseMeta::default()),
            serde_json::json!({"rows": []}),
        );
        let v: Value = serde_json::from_str(&env.to_json()?)?;
        assert!(
            v.get("meta").is_none(),
            "Some(empty ResponseMeta) must serialise as absent: {v}"
        );
        Ok(())
    }

    #[test]
    fn envelope_error_carries_chain_from_anyhow() -> anyhow::Result<()> {
        use anyhow::Context;
        let inner: anyhow::Error = anyhow::anyhow!("disk full");
        let wrapped: anyhow::Error = Err::<(), _>(inner)
            .context("writing parquet sidecar")
            .context("caching trace summary")
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected err"))?;
        let env = EnvelopeError::from_anyhow(
            Some(source()),
            Some("nsys.stats".into()),
            None,
            None,
            &wrapped,
        );
        let v: Value = serde_json::from_str(&env.to_json()?)?;
        assert_eq!(at_str(&v, "/schema")?, "v1");
        assert_eq!(at_str(&v, "/error/message")?, "caching trace summary");
        let chain = at(&v, "/error/chain")?
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("chain not an array"))?;
        let msgs: Vec<&str> = chain.iter().filter_map(|c| c.as_str()).collect();
        assert!(msgs.contains(&"writing parquet sidecar"), "got: {msgs:?}");
        assert!(msgs.contains(&"disk full"), "got: {msgs:?}");
        Ok(())
    }

    #[test]
    fn envelope_error_omits_optional_fields_when_unset() -> anyhow::Result<()> {
        let env = EnvelopeError::new(None, None, None, None, None, "boom", Vec::new());
        let v: Value = serde_json::from_str(&env.to_json()?)?;
        assert!(v.get("source").is_none());
        assert!(v.get("command").is_none());
        assert!(v.get("trace").is_none());
        assert!(v.get("trace_span").is_none());
        assert_eq!(at_str(&v, "/error/message")?, "boom");
        Ok(())
    }

    #[test]
    fn envelope_error_emits_trace_span_when_set() -> anyhow::Result<()> {
        let env = EnvelopeError::new(
            Some(source()),
            Some("nsys.stats".into()),
            Some(trace()),
            Some(span()),
            None,
            "boom",
            Vec::new(),
        );
        let v: Value = serde_json::from_str(&env.to_json()?)?;
        assert_eq!(at(&v, "/trace_span/origin_ns")?.as_i64(), Some(1_000_000));
        assert_eq!(at(&v, "/trace_span/span_ns")?.as_i64(), Some(5_000_000_000));
        Ok(())
    }
}
