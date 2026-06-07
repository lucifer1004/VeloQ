//! `ProfileSource` — the trait every profile backend implements so the
//! `veloq` binary can dispatch to it.
//!
//! A source is more than a parser: it owns its clap subcommand tree,
//! the trace-detection heuristic that lets meta verbs find it without
//! the user typing the source name, and the run glue that produces
//! output in whichever [`OutputFormat`] the user asked for.

use crate::diagnostic::{ErrorCode, VeloqDiagnostic};
use crate::envelope::{SourceRef, TraceSpan};
use std::error::Error;
use std::path::Path;
use thiserror::Error;

pub type SourceRunError = Box<dyn Error + Send + Sync + 'static>;
pub type SourceRunResult<T> = Result<T, SourceRunError>;

#[derive(Debug, Error)]
pub enum OutputFormatError {
    #[error("unknown --format `{value}` (expected: json, csv, table)")]
    Unknown { value: String },
}

impl OutputFormatError {
    pub fn unknown(value: &str) -> Self {
        Self::Unknown {
            value: value.to_string(),
        }
    }
}

impl VeloqDiagnostic for OutputFormatError {
    fn code(&self) -> ErrorCode {
        match self {
            Self::Unknown { .. } => ErrorCode::new("cli.unknown-format"),
        }
    }
}

/// Output format every CLI invocation has to pick. JSON is the agent
/// contract; CSV / table are human-only conveniences. Lives in
/// `veloq-core` so sources don't each redefine the same enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputFormat {
    Json,
    Csv,
    Table,
}

impl OutputFormat {
    /// Parse the `--format` flag. Accepts `json`, `csv`, `table`,
    /// case-insensitive (plus `tbl` as a shorter alias for `table`).
    pub fn parse(s: &str) -> Result<Self, OutputFormatError> {
        let normalized = s.to_ascii_lowercase();
        match normalized.as_str() {
            "json" => Ok(Self::Json),
            "csv" => Ok(Self::Csv),
            "table" | "tbl" => Ok(Self::Table),
            _ => Err(OutputFormatError::unknown(s)),
        }
    }
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Json => "json",
            Self::Csv => "csv",
            Self::Table => "table",
        })
    }
}

/// One pluggable profile backend.
///
/// The `veloq` binary registers a `Vec<Box<dyn ProfileSource>>` at
/// startup; CLI dispatch is the sum of every source's contribution
/// under a top-level `veloq <kind> …` namespace, plus a configured
/// default whose verbs are hoisted to `veloq <verb> …`.
///
/// `Send + Sync` is required only so a registry can be shared across
/// threads if a future server frontend wants to. The trait itself
/// makes no concurrency assumptions; methods take `&self`.
///
/// Each source owns its own emission: `run` writes the response
/// (envelope-wrapped JSON or human-format CSV/table) directly to
/// stdout. This avoids forcing every source's strongly-typed
/// response shape through a `Deserialize` round trip just so CSV /
/// table flatteners can re-claim it.
pub trait ProfileSource: Send + Sync {
    /// Stable short name. Becomes the CLI namespace
    /// (`veloq <kind> …`) and lands in `envelope.source.kind`.
    /// Lowercase ASCII, no spaces.
    fn kind(&self) -> &'static str;

    /// Source-specific semver (`"v0"`, `"v1"`…). Bumps independently
    /// from the envelope schema version; lands in
    /// `envelope.source.version`.
    fn version(&self) -> &'static str;

    /// Combine `kind()` + `version()` into the envelope's source ref.
    fn source_ref(&self) -> SourceRef {
        SourceRef {
            kind: self.kind(),
            version: self.version(),
        }
    }

    /// Heuristic: does this path look like a trace this source can
    /// handle? Used by meta verbs (`veloq info <trace>`) to pick a
    /// source by trace extension or magic when the user didn't name
    /// one. Side-effect-free; should not open the file.
    fn detect(&self, trace: &Path) -> bool;

    /// Best-effort trace-wide `(origin_ns, span_ns)` an agent uses as
    /// the normalization denominator on cross-trace diffs. Called once
    /// per CLI invocation, after argument parsing and before verb
    /// dispatch; the result lands on the envelope (`trace_span`) so
    /// every response carries it.
    ///
    /// Default returns `None` — sources opt in when they can derive
    /// the answer cheaply (e.g. NSys reads from its metadata cache).
    /// Failures should return `None` rather than `Err`: a
    /// missing `trace_span` degrades agent normalization but should
    /// never block the verb itself. Implementations are responsible
    /// for logging any underlying error at warn level.
    fn compute_trace_span(&self, _trace: &Path) -> Option<TraceSpan> {
        None
    }

    /// The clap subcommand tree this source contributes. Returned as
    /// a built [`clap::Command`]; the top-level CLI grafts it under
    /// the source name (or hoists its subcommands when this is the
    /// configured default).
    ///
    /// Sources are free to compose their subtree however they like —
    /// `Command::new(self.kind()).subcommand(...)` is the obvious
    /// idiom but not required.
    fn cli(&self) -> clap::Command;

    /// Run the dispatched verb and write its output to stdout in the
    /// requested format.
    ///
    /// `matches` is the [`ArgMatches`] for this source's subtree
    /// (the result of parsing against [`Self::cli`]); sources need
    /// not handle their own namespace prefix.
    ///
    /// Return contract:
    /// - `Ok(0)` — verb succeeded, success envelope written to stdout.
    /// - `Ok(1)` — verb failed, source already wrote its
    ///   `EnvelopeError` envelope to stdout (with full verb/trace
    ///   context). Caller just propagates the exit code.
    /// - `Err(_)` — top-level / unhandled failure. The caller emits a
    ///   CLI-level error envelope (no verb/trace context) and exits 1.
    ///
    /// Splitting the "handled" case (envelope already on stdout) from
    /// the "unhandled" case keeps verb-level error envelopes
    /// agent-actionable without forcing a panic-style exit through
    /// `process::exit` from inside the source.
    ///
    /// [`ArgMatches`]: clap::ArgMatches
    fn run(&self, matches: &clap::ArgMatches, fmt: OutputFormat) -> SourceRunResult<i32>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Command;

    struct FakeSource;
    impl ProfileSource for FakeSource {
        fn kind(&self) -> &'static str {
            "fake"
        }
        fn version(&self) -> &'static str {
            "v1"
        }
        fn detect(&self, p: &Path) -> bool {
            p.extension().is_some_and(|e| e == "fake")
        }
        fn cli(&self) -> Command {
            Command::new("fake").subcommand(Command::new("ping"))
        }
        fn run(&self, m: &clap::ArgMatches, _fmt: OutputFormat) -> SourceRunResult<i32> {
            // Real impls emit to stdout. The test impl just verifies
            // the dispatch path: assert we got the subcommand we
            // expected.
            let verb = m
                .subcommand_name()
                .ok_or_else(|| std::io::Error::other("no subcommand"))?;
            if verb != "ping" {
                return Err(
                    std::io::Error::other(format!("unexpected subcommand `{verb}`")).into(),
                );
            }
            Ok(0)
        }
    }

    #[test]
    fn source_ref_combines_kind_and_version() {
        let s = FakeSource;
        let r = s.source_ref();
        assert_eq!(r.kind, "fake");
        assert_eq!(r.version, "v1");
    }

    #[test]
    fn detect_matches_by_extension() {
        let s = FakeSource;
        assert!(s.detect(Path::new("/tmp/t.fake")));
        assert!(!s.detect(Path::new("/tmp/t.nsys-rep")));
        assert!(!s.detect(Path::new("/tmp/t")));
    }

    #[test]
    fn run_dispatches_subcommand() -> SourceRunResult<()> {
        let s = FakeSource;
        let m = s.cli().try_get_matches_from(["fake", "ping"])?;
        assert_eq!(s.run(&m, OutputFormat::Json)?, 0);
        Ok(())
    }

    #[test]
    fn output_format_parse_round_trip() {
        for s in ["json", "JSON", "csv", "Csv", "table", "tbl"] {
            assert!(OutputFormat::parse(s).is_ok());
        }
        assert!(OutputFormat::parse("bogus").is_err());
    }
}
