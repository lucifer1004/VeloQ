//! `NsysSource` — the `ProfileSource` impl for Nsight Systems traces.
//!
//! Owns the source-identity constants (`KIND`, `VERSION`), the trace
//! detection heuristic (`.nsys-rep` / `_pqtdir/` / generated
//! `parquetdir/` alias), the clap subcommand tree (delegated to
//! [`crate::cli::Cmd`] + [`crate::help`]), and the run glue that
//! hands off to [`crate::commands::run`].

use crate::cli::Cmd;
use crate::commands;
use crate::error::NsysSourceError;
use crate::help::inject_long_about;
use clap::{Command, FromArgMatches, Subcommand};
use std::path::Path;
use veloq_core::source::{OutputFormat, ProfileSource};
use veloq_core::{SourceRunResult, TraceSpan};

pub struct NsysSource;

impl NsysSource {
    /// Source kind. Matches `envelope.source.kind` and the
    /// `veloq <kind> …` CLI namespace. NSys verbs are also hoisted
    /// to the root because NSys is the default source.
    pub const KIND: &'static str = "nsys";

    /// Source-specific wire-format version. Bumps independently from
    /// the envelope's `ENVELOPE_VERSION`; this is the version of the
    /// NSys data structures (StatsResponse, SummaryResponse, …), not
    /// the envelope wrapping them.
    ///
    /// Every list response is canonical `data.rows[]` with a per-row
    /// `key`, and event rows share the `EventRef` type. `stats
    /// --group-by nvtx-path` rows carry the NVTX domain dimension: the
    /// primary row `key` gains a `domain:<pid>:<domainId>` component and
    /// each real nvtx-path row carries the resolved domain identity
    /// (`domain_id`, `domain_pid`) plus its `domain_name` when
    /// registered. Same-name/same-parent ranges in distinct
    /// `(pid, domainId)` domains stay distinct.
    pub const VERSION: &'static str = "v3";
}

impl ProfileSource for NsysSource {
    fn kind(&self) -> &'static str {
        Self::KIND
    }

    fn version(&self) -> &'static str {
        Self::VERSION
    }

    fn detect(&self, trace: &Path) -> bool {
        // veloq reads `.nsys-rep` (auto-exported to
        // `<trace>.veloq/parquetdir/` on first open), bare
        // `<stem>_pqtdir/` directories (the nsys-emitted parquet
        // output, used as the direct input by tests and pre-export
        // workflows), and its own generated `parquetdir/` as an alias
        // back to the owning report. `.sqlite` is not accepted —
        // veloq reads NSys traces only through the parquetdir export.
        if matches!(trace.extension().and_then(|e| e.to_str()), Some("nsys-rep")) {
            return true;
        }
        let direct_pqtdir = trace
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with("_pqtdir"));
        direct_pqtdir || veloq_nsys_data::nsys_rep::is_valid_generated_parquetdir(trace)
    }

    /// Build the envelope-level `trace_span` from the trace's primary
    /// (kernel/memcpy/memset/runtime/sync) time range. **Sidecar-only**
    /// path: reads `<trace>.veloq/meta.bin` if present and fingerprint-matches,
    /// otherwise returns `None` without triggering a cold metadata
    /// build. The cold cost doesn't belong on the pre-dispatch path —
    /// the verb that ends up needing meta will build it anyway, and
    /// `output::emit` re-reads the sidecar after dispatch so cold-trace
    /// summary still emits a populated `trace_span`. Missing path is
    /// silent (the verb will surface a `trace not found` error
    /// moments later).
    fn compute_trace_span(&self, trace: &Path) -> Option<TraceSpan> {
        if !trace.exists() {
            return None;
        }
        veloq_nsys_data::meta_cache::trace_span_for_path(trace)
    }

    fn cli(&self) -> Command {
        // Parent command for this source. The binary either grafts
        // this subtree at `veloq nsys …` or — when NSys is the
        // configured default — hoists every subcommand to the top
        // level so users keep typing `veloq stats <trace>`.
        let parent = Command::new(Self::KIND)
            .about("Nsight Systems profile-query verbs")
            .subcommand_required(true)
            .arg_required_else_help(true);
        let parent = Cmd::augment_subcommands(parent);
        inject_long_about(parent)
    }

    fn run(&self, matches: &clap::ArgMatches, fmt: OutputFormat) -> SourceRunResult<i32> {
        // Parse the matches back into the typed `Cmd`. The clap
        // dance happens twice (binary builds the same tree, parses
        // once for global `--format`; we parse this subtree again to
        // recover the strongly-typed enum) — cheap, no I/O.
        let cmd = Cmd::from_arg_matches(matches)?;
        let verb = cmd.name();
        // `schema` has no trace input; every other verb does. Plumb
        // the `Option<&Path>` through so trace-less failures don't
        // fabricate an `envelope.trace.path == ""`.
        let raw_stdout = cmd.raw_stdout();
        let trace_path = cmd.trace_path().map(Path::to_path_buf);
        // Compute the envelope's `trace_span` once, before dispatch.
        // The meta-cache lookup is sub-ms on warm sidecar; on a cold
        // trace it pays the metadata-build cost that the upcoming
        // verb (or the next one) would pay anyway. We propagate it
        // into both the success and error paths so a failed verb
        // still carries the normalization denominator.
        let trace_span = trace_path
            .as_deref()
            .and_then(|p| self.compute_trace_span(p));
        match commands::run(cmd, fmt, trace_path.as_deref(), trace_span) {
            Ok(code) => Ok(code),
            Err(err) => {
                if raw_stdout {
                    eprintln!("veloq: {err}");
                    return Ok(1);
                }
                // Emit the verb-context error envelope before
                // returning so agents pipe a single structured
                // failure document on stdout. The non-zero exit
                // code propagates via the `Ok(1)` return.
                emit_err(verb, trace_path.as_deref(), trace_span, &err, fmt);
                Ok(1)
            }
        }
    }
}

fn emit_err(
    verb: &str,
    trace: Option<&Path>,
    trace_span: Option<TraceSpan>,
    err: &NsysSourceError,
    fmt: OutputFormat,
) {
    crate::output::emit_error(verb, trace, trace_span, err, fmt);
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use std::collections::BTreeSet;
    use std::fs;

    #[test]
    fn command_surface_matches_governing_rfcs() {
        let actual: BTreeSet<String> = NsysSource
            .cli()
            .get_subcommands()
            .map(|cmd| cmd.get_name().to_string())
            .collect();
        // RFC-0006 owns the stable NSys command surface. RFC-0009
        // allows the visible draft `viz` subtree while the static
        // timeline artifact contract is still being proven out.
        let expected: BTreeSet<String> = [
            "concurrency",
            "correlate",
            "correlation-stats",
            "gaps",
            "graph-replays",
            "hardware",
            "inspect",
            "metrics",
            "ncu-command",
            "prep",
            "schema",
            "search",
            "slices",
            "stats",
            "summary",
            "timeline",
            "viz",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        assert_eq!(actual, expected);
    }

    #[test]
    fn detect_claims_nsys_wire_inputs() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let source = dir.path().join("trace.nsys-rep");
        fs::write(&source, b"source")?;

        let direct_pqtdir = dir.path().join("trace_pqtdir");
        fs::create_dir_all(&direct_pqtdir)?;

        let generated = veloq_nsys_data::nsys_rep::pqtdir_path_for(&source);
        fs::create_dir_all(&generated)?;

        let sqlite = dir.path().join("trace.sqlite");
        fs::write(&sqlite, b"sqlite")?;

        let orphan_generated = dir.path().join("missing.nsys-rep.veloq/parquetdir");
        fs::create_dir_all(&orphan_generated)?;

        assert!(NsysSource.detect(&source));
        assert!(NsysSource.detect(&direct_pqtdir));
        assert!(NsysSource.detect(&generated));
        assert!(!NsysSource.detect(&sqlite));
        assert!(!NsysSource.detect(&orphan_generated));
        Ok(())
    }
}
