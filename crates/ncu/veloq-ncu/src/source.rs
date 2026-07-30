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
use crate::error::{NcuSourceError, NcuSourceResult};
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
use clap::{ArgMatches, Command, FromArgMatches, Subcommand};
use std::path::Path;
use veloq_core::{
    Envelope, EnvelopeError, EnvelopeTraceRef, OutputFormat, ProfileSource, SourceExecution,
    SourceQueryContext, SourceRef, SourceRunResult,
    tabular::{TabularView, render_csv, render_table},
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

    fn query_context(&self, matches: &ArgMatches) -> SourceRunResult<SourceQueryContext> {
        let cmd = Cmd::from_arg_matches(matches)?;
        Ok(SourceQueryContext {
            command: format!("{}.{}", Self::KIND, cmd.name()),
            trace_path: cmd.trace_path().map(Path::to_path_buf),
            raw_stdout: false,
        })
    }

    fn execute(&self, matches: &ArgMatches, fmt: OutputFormat) -> SourceRunResult<SourceExecution> {
        let mut output = SourceExecution::new();
        let code = (|| -> SourceRunResult<i32> {
            let cmd = Cmd::from_arg_matches(matches)?;
            let verb = cmd.name();
            let qualified = format!("{}.{verb}", Self::KIND);

            if let Cmd::Schema { target } = &cmd {
                if fmt != OutputFormat::Json {
                    let err = NcuSourceError::UnsupportedSchemaFormat { fmt };
                    emit_err(verb, None, &err, fmt, &mut output);
                    return Ok(1);
                }
                match schema_value_for(target) {
                    Ok(schema) => {
                        let envelope = Envelope::new(
                            Self::source_ref(),
                            qualified,
                            None,
                            None,
                            None,
                            SchemaPayload {
                                target: target.clone(),
                                schema,
                            },
                        );
                        match envelope
                            .to_json_pretty()
                            .map_err(NcuSourceError::serialize_envelope)
                        {
                            Ok(rendered) => output.write_stdout_line(rendered),
                            Err(err) => {
                                emit_err(verb, None, &err, fmt, &mut output);
                                return Ok(1);
                            }
                        }
                        return Ok(0);
                    }
                    Err(err) => {
                        emit_err(verb, None, &err, fmt, &mut output);
                        return Ok(1);
                    }
                }
            }

            let trace = cmd
                .trace_path()
                .ok_or(NcuSourceError::MissingTracePath)?
                .to_path_buf();
            if emit_missing_trace_error(verb, &trace, fmt, &mut output) {
                return Ok(1);
            }

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
                    let grid = match grid.as_deref() {
                        Some(value) => match launches::parse_dims(value).map(Some) {
                            Ok(value) => value,
                            Err(source) => {
                                let err =
                                    NcuSourceError::invalid_launch_dims("--grid", value, source);
                                emit_err(verb, Some(&trace), &err, fmt, &mut output);
                                return Ok(1);
                            }
                        },
                        None => None,
                    };
                    let block = match block.as_deref() {
                        Some(value) => match launches::parse_dims(value).map(Some) {
                            Ok(value) => value,
                            Err(source) => {
                                let err =
                                    NcuSourceError::invalid_launch_dims("--block", value, source);
                                emit_err(verb, Some(&trace), &err, fmt, &mut output);
                                return Ok(1);
                            }
                        },
                        None => None,
                    };
                    let req = launches::LaunchesRequest {
                        kernel_glob: kernel.clone(),
                        nvtx_range_glob: nvtx_range.clone(),
                        grid,
                        block,
                        limit: *limit,
                    };
                    if emit_limit_error(verb, &trace, *limit, fmt, &mut output) {
                        return Ok(1);
                    }
                    return Ok(emit_typed_response(
                        verb,
                        &qualified,
                        &trace,
                        fmt,
                        launches::run(&trace, req),
                        launches_view,
                        &mut output,
                    )?);
                }
                Cmd::Inspect { row_ids, .. } => {
                    return Ok(emit_typed_response(
                        verb,
                        &qualified,
                        &trace,
                        fmt,
                        inspect::run(&trace, row_ids),
                        inspect_view,
                        &mut output,
                    )?);
                }
                Cmd::Metrics {
                    counter,
                    kernel,
                    per_launch,
                    limit,
                    ..
                } => {
                    if emit_limit_error(verb, &trace, *limit, fmt, &mut output) {
                        return Ok(1);
                    }
                    if emit_counter_glob_error(verb, &trace, counter, fmt, &mut output) {
                        return Ok(1);
                    }
                    let req = metrics::MetricsRequest {
                        counter_glob: counter.clone(),
                        kernel_glob: kernel.clone(),
                        per_launch: *per_launch,
                        limit: *limit,
                    };
                    return Ok(emit_typed_response(
                        verb,
                        &qualified,
                        &trace,
                        fmt,
                        metrics::run(&trace, req),
                        metrics_view,
                        &mut output,
                    )?);
                }
                Cmd::Disasm { row_id, .. } => {
                    if emit_launch_row_id_error(verb, &trace, row_id, fmt, &mut output) {
                        return Ok(1);
                    }
                    return Ok(emit_typed_response(
                        verb,
                        &qualified,
                        &trace,
                        fmt,
                        disasm::run(&trace, row_id),
                        disasm_view,
                        &mut output,
                    )?);
                }
                // Auxiliary list verbs — each is a thin sidecar projection.
                Cmd::Ranges { limit, .. } => {
                    if emit_limit_error(verb, &trace, *limit, fmt, &mut output) {
                        return Ok(1);
                    }
                    return Ok(emit_typed_response(
                        verb,
                        &qualified,
                        &trace,
                        fmt,
                        lists::ranges(&trace, *limit),
                        ranges_view,
                        &mut output,
                    )?);
                }
                Cmd::Graphs { limit, .. } => {
                    if emit_limit_error(verb, &trace, *limit, fmt, &mut output) {
                        return Ok(1);
                    }
                    return Ok(emit_typed_response(
                        verb,
                        &qualified,
                        &trace,
                        fmt,
                        lists::graphs(&trace, *limit),
                        graphs_view,
                        &mut output,
                    )?);
                }
                Cmd::Sources { limit, .. } => {
                    if emit_limit_error(verb, &trace, *limit, fmt, &mut output) {
                        return Ok(1);
                    }
                    return Ok(emit_typed_response(
                        verb,
                        &qualified,
                        &trace,
                        fmt,
                        lists::sources(&trace, *limit),
                        sources_view,
                        &mut output,
                    )?);
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
                            emit_err(verb, Some(&trace), &err, fmt, &mut output);
                            return Ok(1);
                        }
                    };
                    if emit_limit_error(verb, &trace, *limit, fmt, &mut output) {
                        return Ok(1);
                    }
                    if line.is_some() && file.is_none() {
                        let err = NcuSourceError::SourceMetricsLineWithoutFile;
                        emit_err(verb, Some(&trace), &err, fmt, &mut output);
                        return Ok(1);
                    }
                    if emit_counter_glob_error(verb, &trace, counter, fmt, &mut output) {
                        return Ok(1);
                    }
                    if emit_launch_row_id_error(verb, &trace, row_id, fmt, &mut output) {
                        return Ok(1);
                    }
                    let req = crate::source_metrics::SourceMetricsRequest {
                        row_id: row_id.clone(),
                        counter_glob: counter.clone(),
                        by: axis,
                        file_glob: file.clone(),
                        line: *line,
                        sort: sort.clone(),
                        limit: *limit,
                    };
                    return Ok(emit_typed_response(
                        verb,
                        &qualified,
                        &trace,
                        fmt,
                        crate::source_metrics::run(&trace, req),
                        source_metrics_view,
                        &mut output,
                    )?);
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
                            emit_err(verb, Some(&trace), &err, fmt, &mut output);
                            return Ok(1);
                        }
                    };
                    if emit_limit_error(verb, &trace, *limit, fmt, &mut output) {
                        return Ok(1);
                    }
                    if emit_launch_row_id_error(verb, &trace, row_id, fmt, &mut output) {
                        return Ok(1);
                    }
                    let req = crate::warp_stalls::WarpStallsRequest {
                        row_id: row_id.clone(),
                        by: axis,
                        file_glob: file.clone(),
                        limit: *limit,
                    };
                    return Ok(emit_typed_response(
                        verb,
                        &qualified,
                        &trace,
                        fmt,
                        crate::warp_stalls::run(&trace, req),
                        warp_stalls_view,
                        &mut output,
                    )?);
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
                    emit_err(verb, Some(&trace), &err, fmt, &mut output);
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
            match emit_response_value(
                &qualified,
                &trace,
                fmt,
                response,
                native_summary_view,
                &mut output,
            ) {
                Ok(code) => Ok(code),
                Err(err) => {
                    emit_err(verb, Some(&trace), &err, fmt, &mut output);
                    Ok(1)
                }
            }
        })()?;
        output.set_exit_code(code);
        Ok(output)
    }
}

fn emit_limit_error(
    verb: &str,
    trace: &Path,
    limit: usize,
    fmt: OutputFormat,
    output: &mut SourceExecution,
) -> bool {
    if limit > 0 {
        return false;
    }
    let err = NcuSourceError::limit_too_small(limit);
    emit_err(verb, Some(trace), &err, fmt, output);
    true
}

fn emit_counter_glob_error(
    verb: &str,
    trace: &Path,
    counter: &str,
    fmt: OutputFormat,
    output: &mut SourceExecution,
) -> bool {
    if counter.split(',').any(|part| !part.trim().is_empty()) {
        return false;
    }
    let err = NcuSourceError::counter_glob_empty();
    emit_err(verb, Some(trace), &err, fmt, output);
    true
}

fn emit_missing_trace_error(
    verb: &str,
    trace: &Path,
    fmt: OutputFormat,
    output: &mut SourceExecution,
) -> bool {
    if trace.exists() || native::cache::path_for(trace).is_file() {
        return false;
    }
    let err = NcuSourceError::trace_not_found(trace);
    emit_err(verb, Some(trace), &err, fmt, output);
    true
}

fn emit_launch_row_id_error(
    verb: &str,
    trace: &Path,
    row_id: &str,
    fmt: OutputFormat,
    output: &mut SourceExecution,
) -> bool {
    match crate::row_id::parse_launch_idx(row_id) {
        Ok(_) => false,
        Err(err) => {
            emit_err(verb, Some(trace), &err, fmt, output);
            true
        }
    }
}

/// Render a handled NCU diagnostic into the shared execution output.
/// Centralizes the `Option<&Path> -> Option<EnvelopeTraceRef>` projection so
/// the execution arms stay terse. `trace_span` is always `None` for NCU today.
fn emit_err(
    verb: &str,
    trace: Option<&Path>,
    err: &NcuSourceError,
    fmt: OutputFormat,
    output: &mut SourceExecution,
) {
    let envelope = EnvelopeError::from_diagnostic(
        Some(NcuSource::source_ref()),
        Some(format!("{}.{verb}", NcuSource::KIND)),
        trace.map(NcuSource::trace_ref),
        None,
        err,
    );
    if !matches!(fmt, OutputFormat::Json) {
        output.write_stderr_line(format!("veloq: {err}"));
    }
    if let Ok(rendered) = envelope.to_json_pretty() {
        output.write_stdout_line(rendered);
    }
}

fn emit_typed_response<T, F>(
    verb: &str,
    qualified: &str,
    trace: &Path,
    fmt: OutputFormat,
    response: NcuSourceResult<T>,
    to_view: F,
    output: &mut SourceExecution,
) -> NcuSourceResult<i32>
where
    T: serde::Serialize,
    F: FnOnce(&T) -> TabularView,
{
    let response = match response {
        Ok(v) => v,
        Err(err) => {
            emit_err(verb, Some(trace), &err, fmt, output);
            return Ok(1);
        }
    };
    match emit_response_value(qualified, trace, fmt, response, to_view, output) {
        Ok(code) => Ok(code),
        Err(err) => {
            emit_err(verb, Some(trace), &err, fmt, output);
            Ok(1)
        }
    }
}

fn emit_response_value<T, F>(
    qualified: &str,
    trace: &Path,
    fmt: OutputFormat,
    response: T,
    to_view: F,
    output: &mut SourceExecution,
) -> NcuSourceResult<i32>
where
    T: serde::Serialize,
    F: FnOnce(&T) -> TabularView,
{
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
            output.write_stdout_line(
                env.to_json_pretty()
                    .map_err(NcuSourceError::serialize_envelope)?,
            );
        }
        OutputFormat::Csv => {
            output.write_stdout(render_csv(&to_view(&response), qualified, &trace_str)?);
        }
        OutputFormat::Table => {
            output.write_stdout(render_table(&to_view(&response), qualified, &trace_str));
        }
    }
    Ok(0)
}
