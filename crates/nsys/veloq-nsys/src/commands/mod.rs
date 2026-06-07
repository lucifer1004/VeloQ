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
use veloq_core::{OutputFormat, TraceSpan, guards};
use veloq_nsys_query::search::SearchRequest;
use veloq_nsys_query::stats::{ALLOWED_KINDS as STATS_ALLOWED_KINDS, GroupBy, StatsRequest};
use veloq_nsys_query::{EventKind, RowId};

use crate::cli::Cmd;
use crate::error::{NsysSourceError, NsysSourceResult};
use crate::output::{emit, emit_meta, render, render_with_meta};
use crate::payloads::{CorrelationStatsPayload, PrepPayload, SchemaPayload};
use crate::schema::schema_value_for;
use crate::views;
use meta::{meta_with_scope, projected_scope, run_guards};
use next_steps::{
    gaps_next_steps, search_next_steps, slices_aggregate_next_steps, slices_instance_next_steps,
    stats_next_steps,
};
use parse::{kinds_csv, parse_duration_filter, parse_row_id, parse_sort_spec};
use prep_status::collect_prep_status;
use scope::{resolve_or_refuse, scope_request_from, scope_request_from_device};

/// Gate for hidden flags. Returns `Ok(())` only when `VELOQ_UNSTABLE=1`
/// is present in the process environment; otherwise an error with the
/// canonical experimental-feature wording.
fn require_unstable(verb: &str) -> NsysSourceResult<()> {
    if std::env::var("VELOQ_UNSTABLE").as_deref() == Ok("1") {
        return Ok(());
    }
    Err(NsysSourceError::unstable_feature_disabled(verb))
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
    trace_span: Option<TraceSpan>,
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
        emit_meta("schema", SchemaPayload { target, schema })?;
        return Ok(0);
    }
    let trace = trace.ok_or(NsysSourceError::MissingTracePath)?;
    match cmd {
        Cmd::Summary { .. } => {
            let data = veloq_nsys_query::summary::run(trace)?;
            render(fmt, trace, trace_span, "summary", data, views::summary_view)?;
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
                        trace,
                        fmt,
                        "stats",
                        trace_span,
                        scope_request_from(&location),
                    )? {
                        Some(s) => s,
                        None => return Ok(1),
                    };

                    let kinds = gpu.kinds(&STATS_ALLOWED_KINDS)?;
                    let kind_echo = kinds_csv(&kinds);
                    let sort = parse_sort_spec(&sort)?;
                    let time_window = common.time_window()?;
                    let data = veloq_nsys_query::stats::run(
                        trace,
                        StatsRequest {
                            kinds,
                            group_by,
                            time_window,
                            nvtx: gpu.nvtx.clone(),
                            // Use the resolver-locked device (handles
                            // single-device auto-resolve) over the raw
                            // CLI value.
                            device: scope.applied.device,
                            stream: scope.applied.stream,
                            hist,
                            sort,
                            limit: common.limit_or(50)?,
                            collapse_versioned,
                        },
                    )?;
                    let next_steps = stats_next_steps(&data.rows, &scope);
                    let projected = projected_scope(
                        &scope,
                        kind_echo.as_deref(),
                        gpu.nvtx.as_deref(),
                        data.time_window_ns,
                    );
                    let warnings = run_guards(data.rows.len(), &projected, trace, trace_span);
                    let meta = meta_with_scope(
                        &scope,
                        kind_echo,
                        gpu.nvtx.clone(),
                        data.time_window_ns,
                        next_steps,
                        warnings,
                    );
                    render_with_meta(
                        fmt,
                        trace,
                        trace_span,
                        "stats",
                        meta,
                        data,
                        views::stats_view,
                    )?;
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
                    let data = veloq_nsys_query::stats_by_size::run(
                        trace,
                        veloq_nsys_query::stats_by_size::StatsBySizeRequest {
                            kinds,
                            group_by,
                            time_window: common.time_window()?,
                            device: location.device,
                            stream: location.stream,
                            sort,
                            limit: common.limit_or(50)?,
                        },
                    )?;
                    render(
                        fmt,
                        trace,
                        trace_span,
                        "stats-by-size",
                        data,
                        views::stats_by_size_view,
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
                trace,
                fmt,
                "search",
                trace_span,
                scope_request_from(&location),
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
            let data = veloq_nsys_query::search::run(
                trace,
                SearchRequest {
                    kinds,
                    name_glob: name,
                    name_regex,
                    duration,
                    time_window: common.time_window()?,
                    nvtx: gpu.nvtx.clone(),
                    device: scope.applied.device,
                    stream: scope.applied.stream,
                    sort,
                    limit: common.limit_or(100)?,
                    with_nvtx,
                },
            )?;
            let next_steps = search_next_steps(&data.rows, &scope);
            let projected = projected_scope(
                &scope,
                kind_echo.as_deref(),
                gpu.nvtx.as_deref(),
                data.time_window_ns,
            );
            let warnings = run_guards(data.rows.len(), &projected, trace, trace_span);
            let meta = meta_with_scope(
                &scope,
                kind_echo,
                gpu.nvtx.clone(),
                data.time_window_ns,
                next_steps,
                warnings,
            );
            render_with_meta(
                fmt,
                trace,
                trace_span,
                "search",
                meta,
                data,
                views::search_view,
            )?;
        }

        Cmd::Inspect { row_ids, .. } => {
            let parsed: Vec<RowId> = row_ids
                .iter()
                .map(|s| parse_row_id(s))
                .collect::<NsysSourceResult<_>>()?;
            let data = veloq_nsys_query::inspect::run(trace, &parsed)?;
            render(fmt, trace, trace_span, "inspect", data, views::inspect_view)?;
        }

        Cmd::Correlate { row_ids, .. } => {
            let parsed: Vec<RowId> = row_ids
                .iter()
                .map(|s| parse_row_id(s))
                .collect::<NsysSourceResult<_>>()?;
            let data = veloq_nsys_query::correlate::run(trace, &parsed)?;
            render(
                fmt,
                trace,
                trace_span,
                "correlate",
                data,
                views::correlate_view,
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
                trace,
                fmt,
                "graph-replays",
                trace_span,
                scope_request_from_device(&location),
            )? {
                Some(s) => s,
                None => return Ok(1),
            };
            let sort = parse_sort_spec(&sort)?;
            let data = veloq_nsys_query::graph_replays::run(
                trace,
                veloq_nsys_query::graph_replays::GraphReplaysRequest {
                    time_window: common.time_window()?,
                    nvtx: nvtx.clone(),
                    device: scope.applied.device,
                    sort,
                    limit: common.limit_or(20)?,
                    top_nodes_limit: top_nodes,
                },
            )?;
            let projected = projected_scope(&scope, None, nvtx.as_deref(), data.time_window_ns);
            let warnings = run_guards(data.rows.len(), &projected, trace, trace_span);
            let meta = meta_with_scope(
                &scope,
                None,
                nvtx.clone(),
                data.time_window_ns,
                Vec::new(),
                warnings,
            );
            render_with_meta(
                fmt,
                trace,
                trace_span,
                "graph-replays",
                meta,
                data,
                views::graph_replays_view,
            )?;
        }

        Cmd::NcuCommand {
            row_id, print, env, ..
        } => {
            let row_id = parse_row_id(&row_id)?;
            let env_policy = veloq_nsys_query::ncu_command::EnvPolicy::parse(&env)?;
            let data = veloq_nsys_query::ncu_command::run(
                trace,
                veloq_nsys_query::ncu_command::NcuCommandRequest { row_id, env_policy },
            )?;
            if print {
                print!("{}", data.script);
            } else {
                if fmt != OutputFormat::Json {
                    return Err(NsysSourceError::ncu_command_unsupported_format(fmt));
                }
                emit(trace, trace_span, "ncu-command", data)?;
            }
        }

        Cmd::Concurrency {
            location, common, ..
        } => {
            let resolved = match resolve_or_refuse(
                trace,
                fmt,
                "concurrency",
                trace_span,
                scope_request_from_device(&location),
            )? {
                Some(s) => s,
                None => return Ok(1),
            };
            let data = veloq_nsys_query::concurrency::run(
                trace,
                veloq_nsys_query::concurrency::ConcurrencyRequest {
                    device: resolved.applied.device,
                    time_window: common.time_window()?,
                    limit: common.limit_or(100)?,
                },
            )?;
            let projected = projected_scope(&resolved, None, None, data.time_window_ns);
            let warnings = run_guards(data.rows.len(), &projected, trace, trace_span);
            let meta =
                meta_with_scope(&resolved, None, None, data.time_window_ns, vec![], warnings);
            render_with_meta(
                fmt,
                trace,
                trace_span,
                "concurrency",
                meta,
                data,
                views::concurrency_view,
            )?;
        }

        Cmd::Gaps {
            scope: scope_arg,
            min_duration,
            location,
            sort,
            common,
            ..
        } => {
            let resolved = match resolve_or_refuse(
                trace,
                fmt,
                "gaps",
                trace_span,
                scope_request_from(&location),
            )? {
                Some(s) => s,
                None => return Ok(1),
            };
            let min_ns = veloq_nsys_query::gaps::GapsRequest::parse_min_duration(&min_duration)?;
            let sort = parse_sort_spec(&sort)?;
            let gap_scope = veloq_nsys_query::gaps::GapScope::parse(&scope_arg)?;
            let data = veloq_nsys_query::gaps::run(
                trace,
                veloq_nsys_query::gaps::GapsRequest {
                    min_ns,
                    scope: gap_scope,
                    device: resolved.applied.device,
                    stream: resolved.applied.stream,
                    time_window: common.time_window()?,
                    sort,
                    limit: common.limit_or(100)?,
                },
            )?;
            let next_steps = gaps_next_steps(&data.rows);
            let projected = projected_scope(&resolved, None, None, data.time_window_ns);
            let warnings = run_guards(data.rows.len(), &projected, trace, trace_span);
            let meta = meta_with_scope(
                &resolved,
                None,
                None,
                data.time_window_ns,
                next_steps,
                warnings,
            );
            render_with_meta(fmt, trace, trace_span, "gaps", meta, data, views::gaps_view)?;
        }

        Cmd::Timeline {
            interval,
            gpu,
            location,
            common,
            ..
        } => {
            let scope = match resolve_or_refuse(
                trace,
                fmt,
                "timeline",
                trace_span,
                scope_request_from(&location),
            )? {
                Some(s) => s,
                None => return Ok(1),
            };
            let interval_ns =
                veloq_nsys_query::timeline::TimelineRequest::parse_interval(&interval)?;
            let kinds = gpu.kinds(&veloq_nsys_query::timeline::ALLOWED_KINDS)?;
            let kind_echo = kinds_csv(&kinds);
            let data = veloq_nsys_query::timeline::run(
                trace,
                veloq_nsys_query::timeline::TimelineRequest {
                    interval_ns,
                    kinds,
                    time_window: common.time_window()?,
                    nvtx: gpu.nvtx.clone(),
                    device: scope.applied.device,
                    stream: scope.applied.stream,
                    limit: common.limit_or(1000)?,
                },
            )?;
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
                .or_else(|| veloq_nsys_data::meta_cache::trace_span_for_path(trace))
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
            render_with_meta(
                fmt,
                trace,
                trace_span,
                "timeline",
                timeline_meta,
                data,
                views::timeline_view,
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
                trace,
                fmt,
                "slices",
                trace_span,
                scope_request_from(&location),
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
            let data = veloq_nsys_query::slices::run(
                trace,
                veloq_nsys_query::slices::SlicesRequest {
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
                },
            )?;
            let next_steps = match view {
                veloq_nsys_query::slices::SlicesView::Instance => {
                    slices_instance_next_steps(&data.rows)
                }
                veloq_nsys_query::slices::SlicesView::Aggregate => {
                    slices_aggregate_next_steps(&data.rows, &scope)
                }
            };
            let projected = projected_scope(&scope, None, None, data.time_window_ns);
            let warnings = run_guards(data.rows.len(), &projected, trace, trace_span);
            let meta = meta_with_scope(
                &scope,
                None,
                None,
                data.time_window_ns,
                next_steps,
                warnings,
            );
            render_with_meta(
                fmt,
                trace,
                trace_span,
                "slices",
                meta,
                data,
                views::slices_view,
            )?;
        }

        Cmd::Prep { status, .. } => {
            if status {
                let data = collect_prep_status(trace)?;
                render(fmt, trace, trace_span, "prep", data, views::key_value_view)?;
            } else {
                let started = std::time::Instant::now();
                let trace_handle = veloq_nsys_data::Trace::open(trace)?;
                // Warm the metadata sidecar so the next `summary` (or
                // any command that consults the meta cache) runs
                // zero-SQL. `build_or_load` is idempotent: when the
                // cache exists and matches the trace mtime/size, this
                // is a deserialise plus a few bincode allocations.
                trace_handle.meta_cache()?;
                let meta_cache_path = veloq_nsys_data::meta_cache::path_for(trace_handle.path())
                    .display()
                    .to_string();
                let elapsed_ms = started.elapsed().as_millis() as u64;
                render(
                    fmt,
                    trace,
                    trace_span,
                    "prep",
                    PrepPayload {
                        elapsed_ms,
                        cache_root: veloq_core::artifact_dir_for(trace_handle.path())
                            .display()
                            .to_string(),
                        parquet_tables: trace_handle.tables().to_vec(),
                        meta_cache_path,
                    },
                    views::key_value_view,
                )?;
            }
        }

        Cmd::Hardware { .. } => {
            let data = veloq_nsys_query::hardware::run(trace)?;
            render(
                fmt,
                trace,
                trace_span,
                "hardware",
                data,
                views::hardware_view,
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
            let data = veloq_nsys_query::metrics::run(trace, req)?;
            render(fmt, trace, trace_span, "metrics", data, views::metrics_view)?;
        }

        Cmd::Schema { .. } => {
            // Handled by the early-return above; this arm exists
            // only to satisfy match exhaustiveness.
            return Err(NsysSourceError::SchemaHandledBeforeTraceDispatch);
        }

        Cmd::CorrelationStats { .. } => {
            let started = std::time::Instant::now();
            let trace_handle = veloq_nsys_data::Trace::open(trace)?;
            let index = trace_handle.correlation_index()?;
            let elapsed_ms = started.elapsed().as_millis() as u64;
            let stats = index.stats();
            let cache_present =
                veloq_nsys_data::correlation::path_for(trace_handle.path()).exists();
            render(
                fmt,
                trace,
                trace_span,
                "correlation-stats",
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
            )?;
        }
    }

    Ok(0)
}
