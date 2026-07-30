//! `NsysSource` — the `ProfileSource` impl for Nsight Systems traces.
//!
//! Owns the source-identity constants (`KIND`, `VERSION`), the trace
//! detection heuristic (`.nsys-rep` / `_pqtdir/` / generated
//! `parquetdir/` alias), the clap subcommand tree (delegated to
//! [`crate::cli::Cmd`] + [`crate::help`]), and the run glue that
//! hands off to [`crate::commands::run`].

use crate::cli::{
    CONCURRENCY_COMMAND, CORRELATE_COMMAND, Cmd, GAPS_COMMAND, INSPECT_COMMAND, SEARCH_COMMAND,
    SLICES_COMMAND, STATS_COMMAND, SUMMARY_COMMAND, TIMELINE_COMMAND,
};
use crate::commands;
use crate::error::NsysSourceError;
use crate::help::inject_long_about;
use clap::{Command, FromArgMatches, Subcommand};
use std::path::Path;
use veloq_core::source::{OutputFormat, ProfileSource};
use veloq_core::{
    CancellationToken, ProfileSession, SourceExecution, SourceQueryContext, SourceRunResult,
    SourceSessionConfig, SourceSessionIdentity, TraceSpan,
};

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
    /// Version `v4` makes process identity part of every process-local
    /// CUDA identity and row key. Rank processes that each expose
    /// logical device 0 therefore remain distinct across graph replay,
    /// correlation, NVTX attribution, aggregation, gap, and
    /// visualization responses.
    pub const VERSION: &'static str = "v4";
    pub const DAEMON_COMMANDS: &'static [&'static str] = &[
        SUMMARY_COMMAND,
        SEARCH_COMMAND,
        INSPECT_COMMAND,
        CORRELATE_COMMAND,
        STATS_COMMAND,
        TIMELINE_COMMAND,
        CONCURRENCY_COMMAND,
        GAPS_COMMAND,
        SLICES_COMMAND,
    ];
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

    fn daemon_command_verbs(&self) -> &'static [&'static str] {
        Self::DAEMON_COMMANDS
    }

    fn daemon_session_identity(
        &self,
        trace: &Path,
    ) -> SourceRunResult<Option<SourceSessionIdentity>> {
        let identity = veloq_nsys_data::resident_trace_identity(trace)?;
        Ok(Some(SourceSessionIdentity {
            trace_kind: "nsys".to_string(),
            canonical_trace_path: identity.canonical_source_path,
            configuration_key: String::new(),
            freshness_key: identity.freshness_key,
            resident_memory_estimate_bytes: identity.resident_memory_estimate_bytes,
        }))
    }

    fn open_daemon_session(
        &self,
        resolved_trace: &Path,
        config: SourceSessionConfig,
    ) -> SourceRunResult<Option<Box<dyn ProfileSession>>> {
        let worker_threads = usize::try_from(config.query_workers)?;
        Ok(Some(Box::new(NsysProfileSession {
            trace: veloq_nsys_data::Trace::open_for_daemon(
                resolved_trace,
                worker_threads,
                config.query_memory_bytes,
            )?,
            interval_view_failed: false,
            interval_view_accounted_bytes: 0,
        })))
    }

    fn query_context(&self, matches: &clap::ArgMatches) -> SourceRunResult<SourceQueryContext> {
        let cmd = Cmd::from_arg_matches(matches)?;
        Ok(SourceQueryContext {
            command: format!("{}.{}", Self::KIND, cmd.name()),
            trace_path: cmd.trace_path().map(Path::to_path_buf),
            raw_stdout: cmd.raw_stdout(),
        })
    }

    fn execute(
        &self,
        matches: &clap::ArgMatches,
        fmt: OutputFormat,
    ) -> SourceRunResult<SourceExecution> {
        self.execute_with_trace(matches, fmt, None, None)
    }

    fn execute_daemon(
        &self,
        matches: &clap::ArgMatches,
        fmt: OutputFormat,
        resolved_trace: &Path,
    ) -> SourceRunResult<SourceExecution> {
        self.execute_with_trace(matches, fmt, Some(resolved_trace), None)
    }

    fn execute_daemon_cancellable(
        &self,
        matches: &clap::ArgMatches,
        fmt: OutputFormat,
        resolved_trace: &Path,
        config: SourceSessionConfig,
        cancellation: &CancellationToken,
    ) -> SourceRunResult<SourceExecution> {
        if cancellation.is_cancelled() {
            return Err(
                std::io::Error::new(std::io::ErrorKind::Interrupted, "query cancelled").into(),
            );
        }
        let worker_threads = usize::try_from(config.query_workers)?;
        let trace = match veloq_nsys_data::Trace::open_for_daemon(
            resolved_trace,
            worker_threads,
            config.query_memory_bytes,
        ) {
            Ok(trace) => trace,
            Err(error) => {
                let cmd = Cmd::from_arg_matches(matches)?;
                let trace_path = cmd.trace_path().map(Path::to_path_buf);
                let trace_span = self.compute_trace_span(resolved_trace);
                return Ok(render_command_error(
                    cmd.name(),
                    cmd.raw_stdout(),
                    trace_path.as_deref(),
                    Some(resolved_trace),
                    trace_span,
                    &error.into(),
                    fmt,
                ));
            }
        };
        let mut session = NsysProfileSession {
            trace,
            interval_view_failed: false,
            interval_view_accounted_bytes: 0,
        };
        session.execute(matches, fmt, cancellation)
    }
}

impl NsysSource {
    fn execute_with_trace(
        &self,
        matches: &clap::ArgMatches,
        fmt: OutputFormat,
        resolved_trace: Option<&Path>,
        resident_trace: Option<&veloq_nsys_data::Trace>,
    ) -> SourceRunResult<SourceExecution> {
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
            .map(|path| resolved_trace.unwrap_or(path))
            .and_then(|path| self.compute_trace_span(path));
        let mut output = SourceExecution::new();
        match commands::run(
            cmd,
            fmt,
            trace_path.as_deref(),
            resolved_trace,
            resident_trace,
            trace_span,
            &mut output,
        ) {
            Ok(code) => {
                output.set_exit_code(code);
                Ok(output)
            }
            Err(err) => {
                let evidence_trace = trace_path
                    .as_deref()
                    .map(|path| resolved_trace.unwrap_or(path));
                Ok(render_command_error(
                    verb,
                    raw_stdout,
                    trace_path.as_deref(),
                    evidence_trace,
                    trace_span,
                    &err,
                    fmt,
                ))
            }
        }
    }
}

struct NsysProfileSession {
    trace: veloq_nsys_data::Trace,
    interval_view_failed: bool,
    interval_view_accounted_bytes: u64,
}

impl ProfileSession for NsysProfileSession {
    fn execute(
        &mut self,
        matches: &clap::ArgMatches,
        fmt: OutputFormat,
        cancellation: &CancellationToken,
    ) -> SourceRunResult<SourceExecution> {
        if cancellation.is_cancelled() {
            return Err(
                std::io::Error::new(std::io::ErrorKind::Interrupted, "query cancelled").into(),
            );
        }
        let interrupt = self.trace.conn().interrupt_handle();
        cancellation.register_interrupt(move || interrupt.interrupt());
        let command = Cmd::from_arg_matches(matches)?;
        if self.interval_view_accounted_bytes == 0
            && !self.interval_view_failed
            && matches!(command.name(), "timeline" | "concurrency" | "gaps")
        {
            match veloq_nsys_query::resident_intervals::ensure(&self.trace) {
                Ok(Some(info)) => {
                    self.interval_view_accounted_bytes = info.accounted_bytes;
                }
                Ok(None) => {
                    log::debug!(
                        "resident intervals: no eligible registered view; using established query paths"
                    );
                }
                Err(error) => {
                    self.interval_view_failed = true;
                    log::warn!(
                        "resident intervals: session-local build failed; using established query paths: {error:#}"
                    );
                }
            }
        }
        let execution =
            NsysSource.execute_with_trace(matches, fmt, Some(self.trace.path()), Some(&self.trace));
        if cancellation.is_cancelled() {
            return Err(
                std::io::Error::new(std::io::ErrorKind::Interrupted, "query cancelled").into(),
            );
        }
        execution
    }

    fn additional_resident_memory_estimate_bytes(&self) -> u64 {
        let query_engine_bytes = self
            .trace
            .query_engine_resident_memory_estimate_bytes()
            .unwrap_or_else(|error| {
                log::warn!("duckdb resident memory could not be accounted: {error:#}");
                u64::MAX
            });
        query_engine_bytes
            .saturating_add(self.trace.additional_resident_memory_estimate_bytes())
            .saturating_add(self.interval_view_accounted_bytes)
    }
}

fn emit_err(
    verb: &str,
    trace: Option<&Path>,
    trace_span: Option<TraceSpan>,
    err: &NsysSourceError,
    fmt: OutputFormat,
    output: &mut SourceExecution,
) {
    crate::output::emit_error(verb, trace, trace_span, err, fmt, output);
}

fn render_command_error(
    verb: &str,
    raw_stdout: bool,
    trace: Option<&Path>,
    evidence_trace: Option<&Path>,
    trace_span: Option<TraceSpan>,
    error: &NsysSourceError,
    fmt: OutputFormat,
) -> SourceExecution {
    let mut output = SourceExecution::new();
    if raw_stdout {
        output.write_stderr_line(format!("veloq: {error}"));
    } else {
        let trace_span = trace_span
            .or_else(|| evidence_trace.and_then(veloq_nsys_data::meta_cache::trace_span_for_path));
        emit_err(verb, trace, trace_span, error, fmt, &mut output);
    }
    output.set_exit_code(1);
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context, Result};
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
    fn process_filter_is_exposed_on_every_standard_process_sensitive_list_verb() -> Result<()> {
        let command = NsysSource.cli();
        for name in [
            "stats",
            "search",
            "graph-replays",
            "concurrency",
            "gaps",
            "timeline",
            "slices",
        ] {
            let subcommand = command
                .find_subcommand(name)
                .with_context(|| format!("missing {name} subcommand"))?;
            assert!(
                subcommand
                    .get_arguments()
                    .any(|argument| argument.get_id() == "process"),
                "{name} must expose --process for process-private CUDA ordinals"
            );
        }
        Ok(())
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
