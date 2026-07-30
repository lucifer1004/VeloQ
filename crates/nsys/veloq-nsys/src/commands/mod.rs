//! Per-command dispatch: extract clap args, build the relevant
//! `veloq-nsys-query` request, call `run()`, and render the result.
//!
//! Each match arm here is "thin glue" — heavy lifting (SQL,
//! correlation walk, projector, etc.) lives downstream in
//! `veloq-nsys-query`. The pattern is uniform across commands:
//!
//! 1. Pull fields out of the `Cmd::*` variant.
//! 2. Translate CLI strings (`--type kernel,memcpy`, `--from 1s`,
//!    `--sort total:desc`) via the shared `filters::*` helpers and
//!    [`parse_sort_spec`].
//! 3. Build the `veloq-nsys-query::*Request` struct.
//! 4. Call `veloq_nsys_query::<cmd>::run(...)`.
//! 5. Hand the response to [`crate::output::render`] with the
//!    per-command flattener from `crate::views`.
//!
//! `Cmd::Schema` is the lone shape exception — it doesn't read a
//! trace, so it bypasses the `Response` envelope and emits its
//! bespoke `SchemaEnvelope` directly.

mod meta;
mod next_steps;
mod parse;
mod prep_status;
mod scope;

use std::path::Path;
use veloq_core::time::{TimePoint, TimeWindow};
use veloq_core::{OutputFormat, SortKeyDef, SourceExecution, TraceSpan, guards};
use veloq_nsys_query::search::SearchRequest;
use veloq_nsys_query::stats::{ALLOWED_KINDS as STATS_ALLOWED_KINDS, GroupBy, StatsRequest};
use veloq_nsys_query::{EventKind, RowId};
use veloq_vis::{VizAggregation, VizLabelMode};

use crate::cli::{Cmd, VizCmd};
use crate::error::{NsysSourceError, NsysSourceResult};
use crate::output::{RenderContext, emit_meta, emit_with_evidence};
use crate::payloads::{CorrelationStatsPayload, SchemaPayload};
use crate::schema::schema_value_for;
use crate::views;
use meta::{meta_with_scope, projected_scope, run_guards};
use next_steps::{
    gaps_next_steps, search_next_steps, slices_aggregate_next_steps, slices_instance_next_steps,
    stats_next_steps,
};
use parse::{kinds_csv, parse_duration_filter, parse_row_id, parse_sort_spec};
use prep_status::collect_prep_response;
use scope::{
    resolve_or_refuse, scope_request_from, scope_request_from_device,
    scope_request_from_device_with_implicit_all,
};

fn render_context<'a>(
    fmt: OutputFormat,
    trace: &'a Path,
    evidence_trace: &'a Path,
    trace_span: Option<TraceSpan>,
    verb: &'a str,
) -> RenderContext<'a> {
    RenderContext::new(fmt, trace, trace_span, verb).with_evidence_trace(evidence_trace)
}

/// Gate for hidden flags. Returns `Ok(())` only when `VELOQ_UNSTABLE=1`
/// is present in the process environment; otherwise an error with the
/// canonical experimental-feature wording.
fn require_unstable(verb: &str) -> NsysSourceResult<()> {
    if std::env::var("VELOQ_UNSTABLE").as_deref() == Ok("1") {
        return Ok(());
    }
    Err(NsysSourceError::unstable_feature_disabled(verb))
}

fn validate_gaps_scope_args(
    gap_scope: veloq_nsys_query::gaps::GapScope,
    location: &crate::filters::GpuLocationFilters,
    sort: Option<&veloq_core::SortSpec>,
) -> NsysSourceResult<()> {
    use veloq_nsys_query::NsysQueryError;
    use veloq_nsys_query::gaps::{GapScope, SortKey};

    if location.stream.is_some() && gap_scope != GapScope::Stream {
        return Err(NsysQueryError::GapsStreamRequiresStreamScope {
            scope: gap_scope.as_str(),
        }
        .into());
    }

    if let (Some(device), GapScope::Trace) = (location.device, gap_scope) {
        return Err(NsysQueryError::GapsDeviceInTraceScope { device }.into());
    }

    if let Some(spec) = sort {
        for field in spec.fields() {
            let (key, _) = SortKey::from_field(field).map_err(NsysQueryError::gaps_sort_invalid)?;
            match (key, gap_scope) {
                (SortKey::Stream, scope) if scope != GapScope::Stream => {
                    return Err(NsysQueryError::GapsSortStreamRequiresStreamScope {
                        scope: scope.as_str(),
                    }
                    .into());
                }
                (SortKey::Device, GapScope::Trace) => {
                    return Err(NsysQueryError::GapsSortDeviceInTraceScope.into());
                }
                _ => {}
            }
        }
    }

    Ok(())
}

/// Top-level dispatch from the source's `run()` once clap parsing
/// succeeds. `trace` is `None` only for the trace-less `Schema` verb
/// — every other arm requires a path and bails internally if none
/// was supplied. `trace_span` is the pre-computed envelope-level
/// normalization denominator (computed by `NsysSource::compute_trace_span`):
/// `Some` for trace-reading verbs, `None` for `Schema`.
///
/// Each `Cmd::*` arm is small enough to live inline; if a command
/// grows complex (multi-step pipeline, async work, etc.) it can
/// graduate to its own helper without touching this signature.
pub fn run(
    cmd: Cmd,
    fmt: OutputFormat,
    trace: Option<&Path>,
    resolved_trace: Option<&Path>,
    resident_trace: Option<&veloq_nsys_data::Trace>,
    trace_span: Option<TraceSpan>,
    output: &mut SourceExecution,
) -> NsysSourceResult<i32> {
    // Return contract: `Ok(0)` on success, `Ok(1)` when this dispatch
    // already wrote a structured error envelope to stdout (the
    // ambiguity-refuse path does this — the
    // envelope carries `meta.warnings[multi-device-ambiguous]` so the
    // caller skips the generic error-envelope emit), `Err(_)` for
    // every other failure (caller emits the standard error envelope).

    // `Schema` is the lone trace-less verb. Pull it out first so the
    // rest of the body can unwrap `Option<&Path>` once.
    if let Cmd::Schema { target } = cmd {
        // CSV / table N/A for this verb — JSON-only meta endpoint.
        let _ = fmt;
        let schema = schema_value_for(&target)?;
        emit_meta("schema", SchemaPayload { target, schema }, output)?;
        return Ok(0);
    }
    let trace = trace.ok_or(NsysSourceError::MissingTracePath)?;
    let query_trace = resolved_trace.unwrap_or(trace);
    match cmd {
        Cmd::Summary { .. } => {
            let data = match resident_trace {
                Some(trace) => veloq_nsys_query::summary::run_with_trace(trace)?,
                None => veloq_nsys_query::summary::run(query_trace)?,
            };
            let trace_span = trace_span.or_else(|| {
                resident_trace
                    .and_then(|trace| trace.meta_cache().ok())
                    .map(|cache| TraceSpan {
                        origin_ns: cache.origins.primary.start_ns,
                        span_ns: cache.origins.primary.duration_ns(),
                    })
            });
            render_context(fmt, trace, query_trace, trace_span, "summary").render(
                data,
                views::summary_view,
                output,
            )?;
        }

        Cmd::Stats {
            group_by,
            hist,
            collapse_versioned,
            sort,
            by,
            gpu,
            location,
            common,
            ..
        } => {
            let group_by = GroupBy::from_arg(&group_by)?;
            // The --by size branch is hidden behind an env gate.
            // Public default (--by ns) is untouched —
            // the gate is on the dispatch branch only, not at the
            // top of Cmd::Stats.
            match by {
                crate::cli::StatsBy::Ns => {
                    // Resolve scope before opening DuckDB for the
                    // query. On ambiguity, the resolver emits an
                    // EnvelopeError with `meta.warnings[multi-device-ambiguous]`
                    // and we return without running the query.
                    let scope = match resolve_or_refuse(
                        query_trace,
                        resident_trace,
                        fmt,
                        "stats",
                        trace_span,
                        scope_request_from(&location),
                        output,
                    )? {
                        Some(s) => s,
                        None => return Ok(1),
                    };

                    let kinds = gpu.kinds(&STATS_ALLOWED_KINDS)?;
                    let kind_echo = kinds_csv(&kinds);
                    let sort = parse_sort_spec(&sort)?;
                    let time_window = common.time_window()?;
                    let request = StatsRequest {
                        kinds,
                        group_by,
                        time_window,
                        nvtx: gpu.nvtx.clone(),
                        process_id: scope.applied.native_pid,
                        // Use the resolver-locked device (handles
                        // single-device auto-resolve) over the raw
                        // CLI value.
                        device: scope.applied.device,
                        stream: scope.applied.stream,
                        hist,
                        sort,
                        limit: common.limit_or(50)?,
                        collapse_versioned,
                    };
                    let data = match resident_trace {
                        Some(trace) => veloq_nsys_query::stats::run_with_trace(trace, request)?,
                        None => veloq_nsys_query::stats::run(query_trace, request)?,
                    };
                    let next_steps = stats_next_steps(&data.rows, &scope);
                    let projected = projected_scope(
                        &scope,
                        kind_echo.as_deref(),
                        gpu.nvtx.as_deref(),
                        data.time_window_ns,
                    );
                    let warnings = run_guards(data.rows.len(), &projected, query_trace, trace_span);
                    let meta = meta_with_scope(
                        &scope,
                        kind_echo,
                        gpu.nvtx.clone(),
                        data.time_window_ns,
                        next_steps,
                        warnings,
                    );
                    render_context(fmt, trace, query_trace, trace_span, "stats")
                        .with_meta(meta)
                        .render(data, views::stats_view, output)?;
                }
                crate::cli::StatsBy::Size => {
                    require_unstable("--by size")?;
                    if hist {
                        return Err(NsysSourceError::StatsBySizeHistogramUnsupported);
                    }
                    // NVTX attribution isn't wired through the
                    // byte-axis path yet; reject rather than return
                    // unscoped totals labelled with the user's pattern.
                    if gpu.nvtx.is_some() {
                        return Err(NsysSourceError::StatsBySizeNvtxUnsupported);
                    }
                    // Reject group-by axes the byte-axis path doesn't
                    // implement so the response is never silently
                    // missing a grouping the caller asked for.
                    if group_by.graph
                        || group_by.graph_node
                        || group_by.grid_block
                        || group_by.nvtx_parent
                        || group_by.nvtx_path
                    {
                        let unsupported = [
                            ("graph", group_by.graph),
                            ("graph_node", group_by.graph_node),
                            ("grid_block", group_by.grid_block),
                            ("nvtx-parent", group_by.nvtx_parent),
                            ("nvtx-path", group_by.nvtx_path),
                        ]
                        .iter()
                        .filter_map(|(name, on)| if *on { Some(*name) } else { None })
                        .collect::<Vec<_>>()
                        .join(", ");
                        return Err(NsysSourceError::stats_by_size_group_by_unsupported(
                            unsupported,
                        ));
                    }
                    let kinds = gpu.kinds(&veloq_nsys_query::stats_by_size::ALLOWED_KINDS)?;
                    // Reuse the duration-axis sort parser when the
                    // spec is "total" (default) — it always means
                    // "the trace's primary aggregate axis". Other
                    // keys parse against the by-size sort key set so
                    // `--sort gbps` / `--sort p50_ns` reject up
                    // front (the columns don't exist in this mode).
                    let sort = if sort.trim().is_empty() || sort.trim() == "total" {
                        None
                    } else {
                        Some(
                            veloq_core::SortSpec::parse(&sort)
                                .map_err(|source| NsysSourceError::invalid_sort(&sort, source))?,
                        )
                    };
                    let scope = match resolve_or_refuse(
                        query_trace,
                        resident_trace,
                        fmt,
                        "stats-by-size",
                        trace_span,
                        scope_request_from(&location),
                        output,
                    )? {
                        Some(s) => s,
                        None => return Ok(1),
                    };
                    let data = veloq_nsys_query::stats_by_size::run(
                        query_trace,
                        veloq_nsys_query::stats_by_size::StatsBySizeRequest {
                            kinds,
                            group_by,
                            time_window: common.time_window()?,
                            process_id: scope.applied.native_pid,
                            device: scope.applied.device,
                            stream: scope.applied.stream,
                            sort,
                            limit: common.limit_or(50)?,
                        },
                    )?;
                    render_context(fmt, trace, query_trace, trace_span, "stats-by-size").render(
                        data,
                        views::stats_by_size_view,
                        output,
                    )?;
                }
            }
        }

        Cmd::Search {
            name,
            name_regex,
            duration,
            sort,
            with_nvtx,
            gpu,
            location,
            common,
            ..
        } => {
            let scope = match resolve_or_refuse(
                query_trace,
                resident_trace,
                fmt,
                "search",
                trace_span,
                scope_request_from(&location),
                output,
            )? {
                Some(s) => s,
                None => return Ok(1),
            };
            let kinds = gpu.kinds(EventKind::ALL)?;
            let kind_echo = kinds_csv(&kinds);
            let duration = match duration.as_deref() {
                Some(s) => Some(parse_duration_filter(s)?),
                None => None,
            };
            let sort = parse_sort_spec(&sort)?;
            let request = SearchRequest {
                kinds,
                name_glob: name,
                name_regex,
                duration,
                time_window: common.time_window()?,
                nvtx: gpu.nvtx.clone(),
                process_id: scope.applied.native_pid,
                device: scope.applied.device,
                stream: scope.applied.stream,
                sort,
                limit: common.limit_or(100)?,
                with_nvtx,
            };
            let data = match resident_trace {
                Some(trace) => veloq_nsys_query::search::run_with_trace(trace, request)?,
                None => veloq_nsys_query::search::run(query_trace, request)?,
            };
            let next_steps = search_next_steps(&data.rows, &scope);
            let projected = projected_scope(
                &scope,
                kind_echo.as_deref(),
                gpu.nvtx.as_deref(),
                data.time_window_ns,
            );
            let warnings = run_guards(data.rows.len(), &projected, query_trace, trace_span);
            let meta = meta_with_scope(
                &scope,
                kind_echo,
                gpu.nvtx.clone(),
                data.time_window_ns,
                next_steps,
                warnings,
            );
            render_context(fmt, trace, query_trace, trace_span, "search")
                .with_meta(meta)
                .render(data, views::search_view, output)?;
        }

        Cmd::Inspect { row_ids, .. } => {
            let parsed: Vec<RowId> = row_ids
                .iter()
                .map(|s| parse_row_id(s))
                .collect::<NsysSourceResult<_>>()?;
            let data = match resident_trace {
                Some(trace) => veloq_nsys_query::inspect::run_with_trace(trace, &parsed)?,
                None => veloq_nsys_query::inspect::run(query_trace, &parsed)?,
            };
            render_context(fmt, trace, query_trace, trace_span, "inspect").render(
                data,
                views::inspect_view,
                output,
            )?;
        }

        Cmd::Correlate { row_ids, .. } => {
            let parsed: Vec<RowId> = row_ids
                .iter()
                .map(|s| parse_row_id(s))
                .collect::<NsysSourceResult<_>>()?;
            let data = match resident_trace {
                Some(trace) => veloq_nsys_query::correlate::run_with_trace(trace, &parsed)?,
                None => veloq_nsys_query::correlate::run(query_trace, &parsed)?,
            };
            render_context(fmt, trace, query_trace, trace_span, "correlate").render(
                data,
                views::correlate_view,
                output,
            )?;
        }

        Cmd::GraphReplays {
            location,
            nvtx,
            sort,
            top_nodes,
            common,
            ..
        } => {
            let scope = match resolve_or_refuse(
                query_trace,
                resident_trace,
                fmt,
                "graph-replays",
                trace_span,
                scope_request_from_device(&location),
                output,
            )? {
                Some(s) => s,
                None => return Ok(1),
            };
            let sort = parse_sort_spec(&sort)?;
            let data = veloq_nsys_query::graph_replays::run(
                query_trace,
                veloq_nsys_query::graph_replays::GraphReplaysRequest {
                    time_window: common.time_window()?,
                    nvtx: nvtx.clone(),
                    process_id: scope.applied.native_pid,
                    device: scope.applied.device,
                    sort,
                    limit: common.limit_or(20)?,
                    top_nodes_limit: top_nodes,
                },
            )?;
            let projected = projected_scope(&scope, None, nvtx.as_deref(), data.time_window_ns);
            let warnings = run_guards(data.rows.len(), &projected, query_trace, trace_span);
            let meta = meta_with_scope(
                &scope,
                None,
                nvtx.clone(),
                data.time_window_ns,
                Vec::new(),
                warnings,
            );
            render_context(fmt, trace, query_trace, trace_span, "graph-replays")
                .with_meta(meta)
                .render(data, views::graph_replays_view, output)?;
        }

        Cmd::NcuCommand {
            row_id, print, env, ..
        } => {
            let row_id = parse_row_id(&row_id)?;
            let env_policy = veloq_nsys_query::ncu_command::EnvPolicy::parse(&env)?;
            let data = veloq_nsys_query::ncu_command::run(
                query_trace,
                veloq_nsys_query::ncu_command::NcuCommandRequest { row_id, env_policy },
            )?;
            if print {
                output.write_stdout(data.script);
            } else {
                if fmt != OutputFormat::Json {
                    return Err(NsysSourceError::ncu_command_unsupported_format(fmt));
                }
                emit_with_evidence(trace, query_trace, trace_span, "ncu-command", data, output)?;
            }
        }

        Cmd::Concurrency {
            location, common, ..
        } => {
            let resolved = match resolve_or_refuse(
                query_trace,
                resident_trace,
                fmt,
                "concurrency",
                trace_span,
                scope_request_from_device_with_implicit_all(&location),
                output,
            )? {
                Some(s) => s,
                None => return Ok(1),
            };
            let request = veloq_nsys_query::concurrency::ConcurrencyRequest {
                process_id: resolved.applied.native_pid,
                device: resolved.applied.device,
                time_window: common.time_window()?,
                limit: common.limit_or(100)?,
            };
            let data = match resident_trace {
                Some(trace) => veloq_nsys_query::concurrency::run_with_trace(trace, request)?,
                None => veloq_nsys_query::concurrency::run(query_trace, request)?,
            };
            let projected = projected_scope(&resolved, None, None, data.time_window_ns);
            let warnings = run_guards(data.rows.len(), &projected, query_trace, trace_span);
            let meta =
                meta_with_scope(&resolved, None, None, data.time_window_ns, vec![], warnings);
            render_context(fmt, trace, query_trace, trace_span, "concurrency")
                .with_meta(meta)
                .render(data, views::concurrency_view, output)?;
        }

        Cmd::Gaps {
            scope: scope_arg,
            min_duration,
            location,
            sort,
            common,
            ..
        } => {
            let gap_scope = veloq_nsys_query::gaps::GapScope::parse(&scope_arg)?;
            let min_ns = veloq_nsys_query::gaps::GapsRequest::parse_min_duration(&min_duration)?;
            let sort = parse_sort_spec(&sort)?;
            validate_gaps_scope_args(gap_scope, &location, sort.as_ref())?;

            let mut scope_req = scope_request_from(&location);
            if gap_scope == veloq_nsys_query::gaps::GapScope::Trace {
                scope_req.implicit_all_devices = true;
            }
            let resolved = match resolve_or_refuse(
                query_trace,
                resident_trace,
                fmt,
                "gaps",
                trace_span,
                scope_req,
                output,
            )? {
                Some(s) => s,
                None => return Ok(1),
            };
            let request = veloq_nsys_query::gaps::GapsRequest {
                min_ns,
                scope: gap_scope,
                process_id: resolved.applied.native_pid,
                device: resolved.applied.device,
                stream: resolved.applied.stream,
                time_window: common.time_window()?,
                sort,
                limit: common.limit_or(100)?,
            };
            let data = match resident_trace {
                Some(trace) => veloq_nsys_query::gaps::run_with_trace(trace, request)?,
                None => veloq_nsys_query::gaps::run(query_trace, request)?,
            };
            let next_steps = gaps_next_steps(&data.rows);
            let projected = projected_scope(&resolved, None, None, data.time_window_ns);
            let warnings = run_guards(data.rows.len(), &projected, query_trace, trace_span);
            let meta = meta_with_scope(
                &resolved,
                None,
                None,
                data.time_window_ns,
                next_steps,
                warnings,
            );
            render_context(fmt, trace, query_trace, trace_span, "gaps")
                .with_meta(meta)
                .render(data, views::gaps_view, output)?;
        }

        Cmd::Timeline {
            interval,
            gpu,
            location,
            common,
            ..
        } => {
            let scope = match resolve_or_refuse(
                query_trace,
                resident_trace,
                fmt,
                "timeline",
                trace_span,
                scope_request_from(&location),
                output,
            )? {
                Some(s) => s,
                None => return Ok(1),
            };
            let interval_ns =
                veloq_nsys_query::timeline::TimelineRequest::parse_interval(&interval)?;
            let kind_policy =
                veloq_nsys_query::timeline::TimelineKindPolicy::from_gpu_work_definition()?;
            let kinds = gpu.kinds(kind_policy.allowed())?;
            let kind_echo = kinds_csv(&kinds);
            let request = veloq_nsys_query::timeline::TimelineRequest {
                interval_ns,
                kinds,
                time_window: common.time_window()?,
                nvtx: gpu.nvtx.clone(),
                process_id: scope.applied.native_pid,
                device: scope.applied.device,
                stream: scope.applied.stream,
                limit: common.limit_or(1000)?,
            };
            let data = match resident_trace {
                Some(trace) => veloq_nsys_query::timeline::run_with_trace(trace, request)?,
                None => veloq_nsys_query::timeline::run(query_trace, request)?,
            };
            let projected = projected_scope(
                &scope,
                kind_echo.as_deref(),
                gpu.nvtx.as_deref(),
                data.time_window_ns,
            );
            // Timeline rows are buckets, not individual events, so the
            // empty-with-scope guard would be noisy on a low-traffic
            // window. Skip the row-count guard; keep narrow-window. Re-
            // read the trace span from the sidecar in case the pre-
            // dispatch hook returned None.
            let span_ns = trace_span
                .or_else(|| veloq_nsys_data::meta_cache::trace_span_for_path(query_trace))
                .map(|s| s.span_ns);
            let warnings =
                if let Some(w) = guards::check_time_window(projected.time_window_ns, span_ns) {
                    vec![w]
                } else {
                    Vec::new()
                };
            let timeline_meta = meta_with_scope(
                &scope,
                kind_echo,
                gpu.nvtx.clone(),
                data.time_window_ns,
                Vec::new(),
                warnings,
            );
            render_context(fmt, trace, query_trace, trace_span, "timeline")
                .with_meta(timeline_meta)
                .render(data, views::timeline_view, output)?;
        }

        Cmd::Viz {
            command:
                VizCmd::Timeline {
                    from,
                    to,
                    tracks,
                    highlight_kernels,
                    width_px,
                    max_tracks,
                    max_items,
                    min_interval_px,
                    density_bin_px,
                    no_density,
                    min_label_px,
                    ..
                },
        } => {
            if !density_bin_px.is_finite() || density_bin_px <= 0.0 {
                return Err(crate::error::NsysSourceError::InvalidRenderOption {
                    flag: "--density-bin-px",
                    value: density_bin_px.to_string(),
                });
            }
            let aggregation = if no_density {
                VizAggregation::ItemLimit
            } else {
                VizAggregation::DensityBins
            };
            let data = veloq_nsys_query::viz_timeline::run(
                query_trace,
                veloq_nsys_query::viz_timeline::VizTimelineRequest {
                    time_window: viz_time_window(from.as_deref(), to.as_deref())?,
                    tracks,
                    highlight_kernels,
                    render_policy: veloq_vis::VizRenderPolicy {
                        width_px,
                        max_tracks,
                        max_items,
                        min_interval_px,
                        density_bin_px,
                        aggregation,
                    },
                    label_policy: veloq_vis::VizLabelPolicy {
                        mode: VizLabelMode::Auto,
                        min_label_px,
                    },
                },
            )?;
            render_context(fmt, trace, query_trace, trace_span, "viz.timeline").render(
                data,
                views::viz_timeline_view,
                output,
            )?;
        }

        Cmd::Slices {
            name,
            name_regex,
            sort,
            aggregate,
            group_by,
            location,
            common,
            ..
        } => {
            let scope = match resolve_or_refuse(
                query_trace,
                resident_trace,
                fmt,
                "slices",
                trace_span,
                scope_request_from(&location),
                output,
            )? {
                Some(s) => s,
                None => return Ok(1),
            };
            let sort = parse_sort_spec(&sort)?;
            let group_by = veloq_nsys_query::slices::SlicesAggregateGroupBy::parse(&group_by)?;
            let view = if aggregate {
                veloq_nsys_query::slices::SlicesView::Aggregate
            } else {
                if group_by != veloq_nsys_query::slices::SlicesAggregateGroupBy::Name {
                    return Err(NsysSourceError::slices_group_by_requires_aggregate(
                        group_by.as_str(),
                    ));
                }
                veloq_nsys_query::slices::SlicesView::Instance
            };
            let request = veloq_nsys_query::slices::SlicesRequest {
                name,
                name_regex,
                view,
                group_by,
                time_window: common.time_window()?,
                sort,
                limit: common.limit_or(100)?,
                device: scope.applied.device,
                stream: scope.applied.stream,
                native_pid: scope.applied.native_pid,
            };
            let data = match resident_trace {
                Some(trace) => veloq_nsys_query::slices::run_with_trace(trace, request)?,
                None => veloq_nsys_query::slices::run(query_trace, request)?,
            };
            let next_steps = match view {
                veloq_nsys_query::slices::SlicesView::Instance => {
                    slices_instance_next_steps(&data.rows)
                }
                veloq_nsys_query::slices::SlicesView::Aggregate => {
                    slices_aggregate_next_steps(&data.rows, &scope)
                }
            };
            let projected = projected_scope(&scope, None, None, data.time_window_ns);
            let warnings = run_guards(data.rows.len(), &projected, query_trace, trace_span);
            let meta = meta_with_scope(
                &scope,
                None,
                None,
                data.time_window_ns,
                next_steps,
                warnings,
            );
            render_context(fmt, trace, query_trace, trace_span, "slices")
                .with_meta(meta)
                .render(data, views::slices_view, output)?;
        }

        Cmd::Prep { status, .. } => {
            let started = std::time::Instant::now();
            if status {
                let data = collect_prep_response(
                    query_trace,
                    false,
                    started.elapsed().as_millis() as u64,
                )?;
                render_context(fmt, trace, query_trace, trace_span, "prep").render(
                    data,
                    views::prep_view,
                    output,
                )?;
            } else {
                let trace_handle = veloq_nsys_data::Trace::open(query_trace)?;
                veloq_nsys_data::sidecar_registry::ensure_prep_sidecars(&trace_handle)?;
                let elapsed_ms = started.elapsed().as_millis() as u64;
                let data = collect_prep_response(trace_handle.path(), true, elapsed_ms)?;
                render_context(fmt, trace, query_trace, trace_span, "prep").render(
                    data,
                    views::prep_view,
                    output,
                )?;
            }
        }

        Cmd::Hardware { .. } => {
            let data = veloq_nsys_query::hardware::run(query_trace)?;
            render_context(fmt, trace, query_trace, trace_span, "hardware").render(
                data,
                views::hardware_view,
                output,
            )?;
        }

        Cmd::Metrics {
            source,
            counter,
            group_by,
            name,
            cpu,
            tid,
            bucket,
            sort,
            common,
            ..
        } => {
            use veloq_nsys_query::metrics::{
                CpuSamplingRequest, CpuSchedRequest, GpuMetricsRequest, MetricSource,
                MetricsRequest, MetricsRequestCommon, NicMetricsRequest,
            };
            let source = MetricSource::parse(&source)
                .map_err(|_| NsysSourceError::metrics_unknown_source(&source))?;
            let bucket_ns = match bucket.as_deref() {
                Some(s) => Some(
                    veloq_nsys_query::metrics::parse_bucket(s)
                        .map_err(|source| NsysSourceError::metrics_invalid_bucket(s, source))?,
                ),
                None => None,
            };
            // `--sort` is optional at the CLI layer; the library picks
            // the per-source default. Forwarding `None` also lets bucket
            // mode reject sort the same way across sources.
            let sort_spec = match sort.as_deref() {
                Some(s) => parse_sort_spec(s)?,
                None => None,
            };
            let common_req = MetricsRequestCommon {
                bucket_ns,
                time_window: common.time_window()?,
                sort: sort_spec,
                limit: common.limit_or(1000)?,
            };
            // Cross-source flag rejection up front: the request enum
            // already enforces "only fields that belong to this
            // source", but clap forwards every `--counter` / `--name`
            // / `--group-by` regardless of `--type`. Diagnose here
            // rather than silently dropping them.
            let req = match source {
                MetricSource::Gpu => {
                    if group_by.is_some() || name.is_some() || cpu.is_some() || tid.is_some() {
                        return Err(NsysSourceError::MetricsCpuFlagsForCounterSource {
                            metric_source: "gpu",
                        });
                    }
                    MetricsRequest::Gpu(GpuMetricsRequest {
                        counter_glob: counter,
                        common: common_req,
                    })
                }
                MetricSource::Nic => {
                    if group_by.is_some() || name.is_some() || cpu.is_some() || tid.is_some() {
                        return Err(NsysSourceError::MetricsCpuFlagsForCounterSource {
                            metric_source: "nic",
                        });
                    }
                    MetricsRequest::Nic(NicMetricsRequest {
                        counter_glob: counter,
                        common: common_req,
                    })
                }
                MetricSource::CpuSampling => {
                    if counter.is_some() {
                        return Err(NsysSourceError::MetricsCounterFlagForCpuSampling);
                    }
                    MetricsRequest::CpuSampling(CpuSamplingRequest {
                        group_by,
                        name_glob: name,
                        cpu,
                        tid,
                        common: common_req,
                    })
                }
                MetricSource::CpuSched => {
                    if counter.is_some() {
                        return Err(NsysSourceError::MetricsCounterFlagForCpuSched);
                    }
                    if name.is_some() {
                        return Err(NsysSourceError::MetricsNameFlagForCpuSched);
                    }
                    MetricsRequest::CpuSched(CpuSchedRequest {
                        group_by,
                        cpu,
                        tid,
                        common: common_req,
                    })
                }
            };
            let data = veloq_nsys_query::metrics::run(query_trace, req)?;
            render_context(fmt, trace, query_trace, trace_span, "metrics").render(
                data,
                views::metrics_view,
                output,
            )?;
        }

        Cmd::Schema { .. } => {
            // Handled by the early-return above; this arm exists
            // only to satisfy match exhaustiveness.
            return Err(NsysSourceError::SchemaHandledBeforeTraceDispatch);
        }

        Cmd::CorrelationStats { .. } => {
            let started = std::time::Instant::now();
            let trace_handle = veloq_nsys_data::Trace::open(query_trace)?;
            let index = trace_handle.correlation_index()?;
            let elapsed_ms = started.elapsed().as_millis() as u64;
            let stats = index.stats();
            let cache_present =
                veloq_nsys_data::correlation::path_for(trace_handle.path()).exists();
            render_context(fmt, trace, query_trace, trace_span, "correlation-stats").render(
                CorrelationStatsPayload {
                    elapsed_ms,
                    cache_present_after: cache_present,
                    contexts: stats.contexts,
                    processes: stats.processes,
                    unique_groups: stats.unique_groups,
                    kernel_rows: stats.kernel_rows,
                    memcpy_rows: stats.memcpy_rows,
                    memset_rows: stats.memset_rows,
                    runtime_rows: stats.runtime_rows,
                    sync_rows: stats.sync_rows,
                    graph_rows: stats.graph_rows,
                },
                views::key_value_view,
                output,
            )?;
        }
    }

    Ok(0)
}

fn viz_time_window(from: Option<&str>, to: Option<&str>) -> NsysSourceResult<Option<TimeWindow>> {
    match (from, to) {
        (Some(from), Some(to)) => {
            let start = TimePoint::parse(from)
                .map_err(|source| NsysSourceError::invalid_from(from, source))?;
            let end =
                TimePoint::parse(to).map_err(|source| NsysSourceError::invalid_to(to, source))?;
            Ok(Some(TimeWindow { start, end }))
        }
        (None, None) => Ok(None),
        _ => Err(NsysSourceError::MissingTimeBound),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::TraceArg;
    use std::path::{Path, PathBuf};

    #[test]
    fn viz_timeline_rejects_invalid_density_bin_even_when_disabled() {
        let cmd = Cmd::Viz {
            command: VizCmd::Timeline {
                trace_arg: TraceArg {
                    trace: PathBuf::from("missing.nsys-rep"),
                },
                from: Some("@0".to_string()),
                to: Some("@1".to_string()),
                tracks: Vec::new(),
                highlight_kernels: Vec::new(),
                width_px: 1200,
                max_tracks: 64,
                max_items: 5000,
                min_interval_px: 1.0,
                density_bin_px: -1.0,
                no_density: true,
                min_label_px: 48.0,
            },
        };

        let mut output = SourceExecution::new();
        let err = run(
            cmd,
            OutputFormat::Json,
            Some(Path::new("missing.nsys-rep")),
            None,
            None,
            None,
            &mut output,
        );

        assert!(matches!(
            err,
            Err(NsysSourceError::InvalidRenderOption {
                flag: "--density-bin-px",
                ..
            })
        ));
    }
}
