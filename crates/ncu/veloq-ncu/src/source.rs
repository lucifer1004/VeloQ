//! `NcuSource` — the `ProfileSource` impl for `.ncu-rep` kernel reports.
//!
//! Owns the source identity, the trace-detection heuristic
//! (`.ncu-rep`), and the run glue that reads the `ncu_report` native
//! sidecar ([`crate::native::cache`]) and emits a shared [`Envelope`]
//! (or [`EnvelopeError`] on failure).
//!
//! `summary` also supports a native table/CSV console projection; JSON
//! remains the full agent-facing payload.
//!
//! [`Envelope`]: veloq_core::Envelope
//! [`EnvelopeError`]: veloq_core::EnvelopeError

use crate::cli::Cmd;
use crate::disasm;
use crate::inspect;
use crate::launches;
use crate::lists;
use crate::metrics;
use crate::native::{
    self, NativeSessionInfo, NativeSummaryAuxiliary, NativeSummaryResponse, NativeTotalsRow,
};
use crate::schema::{SchemaPayload, schema_value_for};
use crate::views::{
    disasm_view, graphs_view, inspect_view, launches_view, metrics_view, native_summary_view,
    ranges_view, source_metrics_view, sources_view, warp_stalls_view,
};
use anyhow::Result;
use clap::{ArgMatches, Command, FromArgMatches, Subcommand};
use std::path::Path;
use veloq_core::{
    Envelope, EnvelopeTraceRef, OutputFormat, ProfileSource, SourceRef,
    tabular::{TabularView, emit_csv, emit_table},
    write_error_envelope,
};

pub struct NcuSource;

impl NcuSource {
    pub const KIND: &'static str = "ncu";
    /// Wire-format version of the NCU source. The `ncu_report`-native
    /// wire: every response is the canonical `data.rows[]` shape with a
    /// per-row `key`. The detail verbs read the `<report>.veloq/`
    /// sidecar; `summary` carries the launch-derived totals plus the NCU
    /// version. `ncu inspect` reports each metric's `metric_type` /
    /// `metric_subtype` / `rollup` as the `ncu_report` enum *name*
    /// (`"counter"` rather than `1`), with the raw integer preserved
    /// alongside as `*_code`.
    pub const VERSION: &'static str = "v1";

    fn source_ref() -> SourceRef {
        SourceRef {
            kind: Self::KIND,
            version: Self::VERSION,
        }
    }

    fn trace_ref(trace: &Path) -> EnvelopeTraceRef {
        EnvelopeTraceRef {
            kind: Self::KIND,
            path: trace.display().to_string(),
        }
    }
}

impl ProfileSource for NcuSource {
    fn kind(&self) -> &'static str {
        Self::KIND
    }

    fn version(&self) -> &'static str {
        Self::VERSION
    }

    fn detect(&self, trace: &Path) -> bool {
        matches!(trace.extension().and_then(|e| e.to_str()), Some("ncu-rep"))
    }

    fn cli(&self) -> Command {
        let parent = Command::new(Self::KIND)
            .about("Nsight Compute kernel-report query verbs")
            .subcommand_required(true)
            .arg_required_else_help(true);
        crate::help::inject_long_about(Cmd::augment_subcommands(parent))
    }

    fn run(&self, matches: &ArgMatches, fmt: OutputFormat) -> Result<i32> {
        let cmd = Cmd::from_arg_matches(matches)?;
        let verb = cmd.name();
        let qualified = format!("{}.{verb}", Self::KIND);

        if let Cmd::Schema { target } = &cmd {
            if fmt != OutputFormat::Json {
                let err = anyhow::anyhow!(
                    "veloq-ncu schema currently supports only --format json (got `{fmt}`)"
                );
                emit_err(verb, None, &err, fmt);
                return Ok(1);
            }
            match schema_value_for(target) {
                Ok(schema) => {
                    veloq_core::emit_envelope(
                        Self::source_ref(),
                        qualified,
                        None,
                        None,
                        None,
                        SchemaPayload {
                            target: target.clone(),
                            schema,
                        },
                    )?;
                    return Ok(0);
                }
                Err(err) => {
                    emit_err(verb, None, &err, fmt);
                    return Ok(1);
                }
            }
        }

        let trace = cmd
            .trace_path()
            .ok_or_else(|| anyhow::anyhow!("internal: ncu verb missing trace path"))?
            .to_path_buf();

        // Detail verbs (launches / inspect / metrics / disasm / ranges
        // / graphs / sources / source-metrics) ship their own narrow
        // response shapes. Each dispatches through
        // [`emit_response`] which selects JSON or a per-verb tabular
        // projector based on `fmt`.
        match &cmd {
            Cmd::Launches {
                kernel,
                nvtx_range,
                grid,
                block,
                limit,
                ..
            } => {
                let grid = match grid.as_deref().map(launches::parse_dims).transpose() {
                    Ok(v) => v,
                    Err(err) => {
                        emit_err(verb, Some(&trace), &err, fmt);
                        return Ok(1);
                    }
                };
                let block = match block.as_deref().map(launches::parse_dims).transpose() {
                    Ok(v) => v,
                    Err(err) => {
                        emit_err(verb, Some(&trace), &err, fmt);
                        return Ok(1);
                    }
                };
                let req = launches::LaunchesRequest {
                    kernel_glob: kernel.clone(),
                    nvtx_range_glob: nvtx_range.clone(),
                    grid,
                    block,
                    limit: *limit,
                };
                return emit_response(
                    verb,
                    &qualified,
                    &trace,
                    fmt,
                    launches::run(&trace, req),
                    launches_view,
                );
            }
            Cmd::Inspect { row_ids, .. } => {
                return emit_response(
                    verb,
                    &qualified,
                    &trace,
                    fmt,
                    inspect::run(&trace, row_ids),
                    inspect_view,
                );
            }
            Cmd::Metrics {
                counter,
                kernel,
                per_launch,
                limit,
                ..
            } => {
                let req = metrics::MetricsRequest {
                    counter_glob: counter.clone(),
                    kernel_glob: kernel.clone(),
                    per_launch: *per_launch,
                    limit: *limit,
                };
                return emit_response(
                    verb,
                    &qualified,
                    &trace,
                    fmt,
                    metrics::run(&trace, req),
                    metrics_view,
                );
            }
            Cmd::Disasm { row_id, .. } => {
                return emit_response(
                    verb,
                    &qualified,
                    &trace,
                    fmt,
                    disasm::run(&trace, row_id),
                    disasm_view,
                );
            }
            // Auxiliary list verbs — each is a thin sidecar projection.
            Cmd::Ranges { limit, .. } => {
                return emit_response(
                    verb,
                    &qualified,
                    &trace,
                    fmt,
                    lists::ranges(&trace, *limit),
                    ranges_view,
                );
            }
            Cmd::Graphs { limit, .. } => {
                return emit_response(
                    verb,
                    &qualified,
                    &trace,
                    fmt,
                    lists::graphs(&trace, *limit),
                    graphs_view,
                );
            }
            Cmd::Sources { limit, .. } => {
                return emit_response(
                    verb,
                    &qualified,
                    &trace,
                    fmt,
                    lists::sources(&trace, *limit),
                    sources_view,
                );
            }
            Cmd::SourceMetrics {
                row_id,
                counter,
                by,
                file,
                line,
                sort,
                limit,
                ..
            } => {
                let axis = match crate::source_metrics::Axis::parse(by) {
                    Ok(a) => a,
                    Err(err) => {
                        emit_err(verb, Some(&trace), &err, fmt);
                        return Ok(1);
                    }
                };
                let req = crate::source_metrics::SourceMetricsRequest {
                    row_id: row_id.clone(),
                    counter_glob: counter.clone(),
                    by: axis,
                    file_glob: file.clone(),
                    line: *line,
                    sort: sort.clone(),
                    limit: *limit,
                };
                return emit_response(
                    verb,
                    &qualified,
                    &trace,
                    fmt,
                    crate::source_metrics::run(&trace, req),
                    source_metrics_view,
                );
            }
            Cmd::WarpStalls {
                row_id,
                by,
                file,
                limit,
                ..
            } => {
                let axis = match crate::warp_stalls::Axis::parse(by) {
                    Ok(a) => a,
                    Err(err) => {
                        emit_err(verb, Some(&trace), &err, fmt);
                        return Ok(1);
                    }
                };
                let req = crate::warp_stalls::WarpStallsRequest {
                    row_id: row_id.clone(),
                    by: axis,
                    file_glob: file.clone(),
                    limit: *limit,
                };
                return emit_response(
                    verb,
                    &qualified,
                    &trace,
                    fmt,
                    crate::warp_stalls::run(&trace, req),
                    warp_stalls_view,
                );
            }
            _ => {}
        }

        // Only `Cmd::Summary` reaches here — every detail verb returned
        // from the match above. `summary` reads the native sidecar
        // for every format: launch-derived totals + the
        // NCU-version-only session.
        //
        // NCU reports don't carry a contiguous trace-wide wall-clock
        // window the way NSys does, so the envelope's optional
        // `trace_span` stays `None` for NCU.
        let sidecar = match native::cache::build_or_load(&trace) {
            Ok(sidecar) => sidecar,
            Err(err) => {
                emit_err(verb, Some(&trace), &err, fmt);
                return Ok(1);
            }
        };
        let response = NativeSummaryResponse {
            count: 1,
            total_matched: 1,
            rows: vec![NativeTotalsRow {
                key: "totals".to_string(),
                totals: sidecar.totals(),
            }],
            auxiliary: NativeSummaryAuxiliary {
                session: NativeSessionInfo {
                    versions: sidecar.session.versions.clone(),
                },
                ncu_version: sidecar.ncu_version.clone(),
                meta_cache_path: native::cache::path_for(&trace).display().to_string(),
            },
        };
        let trace_str = trace.display().to_string();
        match fmt {
            OutputFormat::Json => {
                veloq_core::emit_envelope(
                    Self::source_ref(),
                    qualified,
                    Some(Self::trace_ref(&trace)),
                    None,
                    None,
                    response,
                )?;
            }
            OutputFormat::Csv => emit_csv(&native_summary_view(&response), &qualified, &trace_str)?,
            OutputFormat::Table => {
                emit_table(&native_summary_view(&response), &qualified, &trace_str)?;
            }
        }
        Ok(0)
    }
}

/// Shim around [`write_error_envelope`] that takes the NCU-typed
/// trace path. Centralizes the `Option<&Path> -> Option<EnvelopeTraceRef>`
/// projection so the `run()` arms stay terse. `trace_span` is always
/// `None` for NCU today (see the trace-span note in `run()`).
fn emit_err(verb: &str, trace: Option<&Path>, err: &anyhow::Error, fmt: OutputFormat) {
    write_error_envelope(
        NcuSource::source_ref(),
        verb,
        trace.map(NcuSource::trace_ref),
        None,
        err,
        fmt,
    );
}

/// Per-verb response dispatch. Builds and prints either a
/// JSON envelope or a CSV/table projection of the response. The
/// `to_view` closure is per verb (`ranges_view`, `launches_view`,
/// …) and only runs in the CSV/table branches, so JSON callers
/// don't pay for tabular column derivation.
fn emit_response<T, F>(
    verb: &str,
    qualified: &str,
    trace: &Path,
    fmt: OutputFormat,
    response: Result<T>,
    to_view: F,
) -> Result<i32>
where
    T: serde::Serialize,
    F: FnOnce(&T) -> TabularView,
{
    let response = match response {
        Ok(v) => v,
        Err(err) => {
            emit_err(verb, Some(trace), &err, fmt);
            return Ok(1);
        }
    };
    let trace_str = trace.display().to_string();
    match fmt {
        OutputFormat::Json => {
            let env = Envelope::new(
                NcuSource::source_ref(),
                qualified.to_string(),
                Some(NcuSource::trace_ref(trace)),
                None,
                None,
                response,
            );
            println!("{}", env.to_json_pretty()?);
        }
        OutputFormat::Csv => emit_csv(&to_view(&response), qualified, &trace_str)?,
        OutputFormat::Table => emit_table(&to_view(&response), qualified, &trace_str)?,
    }
    Ok(0)
}
