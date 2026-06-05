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

use anyhow::{Context, Result};
use std::path::Path;
use veloq_core::{
    NextStep, ResponseMeta, SortSpec, TraceSpan, Warning, guards, time::DurationFilter,
};
use veloq_nsys_data::scope::{ResolveError, ResolvedScope, ScopeRequest, resolve_scope};
use veloq_nsys_query::search::SearchRequest;
use veloq_nsys_query::stats::{ALLOWED_KINDS as STATS_ALLOWED_KINDS, GroupBy, StatsRequest};
use veloq_nsys_query::{EventKind, RowId};

use crate::cli::Cmd;
use crate::filters::{DeviceLocationFilters, GpuLocationFilters};
use crate::format::Format;
use crate::output::{emit, emit_ambiguity_error, emit_meta, render, render_with_meta};
use crate::payloads::{
    CorrelationStatsPayload, ParquetCacheStatus, PrepPayload, PrepStatusPayload, SchemaPayload,
    SidecarStatus,
};
use crate::schema::schema_value_for;
use crate::views;

/// Gate for hidden flags. Returns `Ok(())` only when `VELOQ_UNSTABLE=1`
/// is present in the process environment; otherwise an error with the
/// canonical experimental-feature wording.
fn require_unstable(verb: &str) -> Result<()> {
    if std::env::var("VELOQ_UNSTABLE").as_deref() == Ok("1") {
        return Ok(());
    }
    anyhow::bail!(
        "`{verb}` is experimental — set VELOQ_UNSTABLE=1 to opt in. \
         The verb's shape and semantics may change before promotion to public."
    )
}

/// Open the trace, resolve scope, and either return the resolved
/// scope to the caller or emit a structured ambiguity-refusal error
/// envelope and signal the caller to stop.
///
/// When the trace has >1 device and the user gave
/// neither `--device` nor `--all-devices`, the resolver refuses and
/// veloq returns `Ok(None)`. The error envelope (including the
/// `multi-device-ambiguous` warning code) has
/// already been written to stdout; the caller's verb arm should
/// `return Ok(())` immediately.
///
/// `Ok(Some(scope))` means the query is unambiguous and the verb
/// should continue, using `scope.applied` for both the envelope's
/// `meta.applied_scope` block and any SQL WHERE clauses it builds.
fn resolve_or_refuse(
    trace_path: &Path,
    fmt: Format,
    verb: &str,
    trace_span: Option<TraceSpan>,
    req: ScopeRequest,
) -> Result<Option<ResolvedScope>> {
    let trace_handle =
        veloq_nsys_data::Trace::open(trace_path).context("opening trace for scope resolution")?;
    match resolve_scope(&trace_handle, req) {
        Ok(scope) => Ok(Some(scope)),
        Err(ResolveError::Ambiguous(amb)) => {
            emit_ambiguity_error(verb, trace_path, trace_span, &amb, fmt);
            Ok(None)
        }
        Err(ResolveError::Probe(e)) => Err(e),
    }
}

/// Convenience: build a `ScopeRequest` from the CLI args a list verb
/// pulled out of its `Cmd::*` variant. Keeps per-verb arms terse.
fn scope_request_from(location: &GpuLocationFilters) -> ScopeRequest {
    ScopeRequest {
        device: location.device,
        stream: location.stream,
        all_devices: location.all_devices,
    }
}

fn scope_request_from_device(location: &DeviceLocationFilters) -> ScopeRequest {
    ScopeRequest {
        device: location.device,
        stream: None,
        all_devices: location.all_devices,
    }
}

/// Wrap a resolved scope into the envelope's optional `meta` block.
/// The verb dispatch fills `kind` / `nvtx_pattern` / `time_window_ns`
/// from its own parsed args (the resolver leaves them None) before
/// constructing the meta block. `next_steps` carries the verb's
/// follow-up hints (computed from the top result row); pass `vec![]`
/// when the verb has nothing to suggest (empty result, or a verb that
/// hasn't been wired yet).
fn meta_with_scope(
    scope: &ResolvedScope,
    kind: Option<String>,
    nvtx_pattern: Option<String>,
    time_window_ns: Option<(i64, i64)>,
    next_steps: Vec<NextStep>,
    warnings: Vec<Warning>,
) -> Option<ResponseMeta> {
    let mut applied = scope.applied.clone();
    applied.kind = kind;
    applied.nvtx_pattern = nvtx_pattern;
    applied.time_window_ns = time_window_ns;
    Some(ResponseMeta {
        applied_scope: Some(applied),
        next_steps,
        warnings,
    })
}

/// Compose every guardrail check into one warning list. Each
/// `check_*` returns `Option<Warning>`; we collect the `Some`s in
/// declaration order so the wire shape is deterministic across runs.
/// `applied` is the resolved scope projected with kind/nvtx/window
/// already populated — same value the envelope ships, so the guards
/// see exactly what the user asked for.
///
/// The trace span is re-read from the meta sidecar when the pre-
/// dispatch hook returned `None` (e.g. cold parquetdir on the first
/// call). The query path above will have built the sidecar by now,
/// so this is the same warm read `emit_with_meta` performs after
/// dispatch.
fn run_guards(
    row_count: usize,
    applied: &veloq_core::AppliedScope,
    trace: &Path,
    trace_span: Option<TraceSpan>,
) -> Vec<Warning> {
    // The narrow-window guard needs the trace span as a denominator.
    // Pre-dispatch we may have None (no warm sidecar); re-read it now,
    // and as a last resort build it eagerly so the guard fires on
    // cold-first-run too. Skip the build when there's no time window
    // to compare against — the guard wouldn't fire anyway and we'd
    // pay the cost for nothing.
    let span = if applied.time_window_ns.is_some() {
        trace_span
            .or_else(|| veloq_nsys_data::meta_cache::trace_span_for_path(trace))
            .or_else(|| resolve_trace_span_eager(trace))
            .map(|s| s.span_ns)
    } else {
        None
    };
    let mut warnings = Vec::new();
    if let Some(w) = guards::check_time_window(applied.time_window_ns, span) {
        warnings.push(w);
    }
    if let Some(w) = guards::check_empty_result(row_count, applied) {
        warnings.push(w);
    }
    warnings
}

/// Build (or warm-load) the meta sidecar so the guards have a non-
/// `None` trace span to compare against. Best-effort — any failure
/// silently leaves the span as `None` (the narrow-window guard then
/// simply doesn't fire, which is the conservative default).
fn resolve_trace_span_eager(trace: &Path) -> Option<TraceSpan> {
    let trace_handle = veloq_nsys_data::Trace::open(trace).ok()?;
    let cache = trace_handle.meta_cache().ok()?;
    Some(TraceSpan {
        origin_ns: cache.origins.primary.start_ns,
        span_ns: cache.origins.primary.duration_ns(),
    })
}

/// Materialise the projected `AppliedScope` that a verb will assign to
/// its envelope — same kind/nvtx/window fields the meta builder fills
/// — so the guards see exactly what the agent will read back.
fn projected_scope(
    scope: &ResolvedScope,
    kind: Option<&str>,
    nvtx_pattern: Option<&str>,
    time_window_ns: Option<(i64, i64)>,
) -> veloq_core::AppliedScope {
    let mut applied = scope.applied.clone();
    applied.kind = kind.map(str::to_string);
    applied.nvtx_pattern = nvtx_pattern.map(str::to_string);
    applied.time_window_ns = time_window_ns;
    applied
}

/// Append `--device N` to a hint command when the resolver locked in a
/// concrete device. On single-device traces the user didn't type the
/// flag; the suggestion includes it anyway so the agent can copy-paste
/// without forgetting the scope on a multi-device follow-up.
fn device_suffix(scope: &ResolvedScope) -> String {
    match scope.applied.device {
        Some(d) => format!(" --device {d}"),
        None => String::new(),
    }
}

/// Top-row follow-up for `slices` in instance view: inspect the NVTX
/// range itself, which expands the GPU-attributed kernels under it.
/// `inspect` is row-id-keyed so the resolved scope doesn't carry into
/// the follow-up command — agents that need to keep scoping just re-run
/// `slices --device N` themselves.
fn slices_instance_next_steps(rows: &[veloq_nsys_query::slices::SlicesRow]) -> Vec<NextStep> {
    use veloq_nsys_query::slices::SlicesRow;
    let Some(SlicesRow::Instance(top)) = rows.first() else {
        return Vec::new();
    };
    vec![NextStep {
        hint: format!(
            "Drill into the top NVTX range `{}` to see every GPU kernel \
             attributed under it.",
            top.name
        ),
        command: format!("veloq inspect {}", top.row_id),
    }]
}

/// Top-row follow-up for `slices` in aggregate view: re-run `slices`
/// scoped to the heaviest aggregate's NVTX name so the agent sees each
/// instance's contribution.
fn slices_aggregate_next_steps(
    rows: &[veloq_nsys_query::slices::SlicesRow],
    scope: &ResolvedScope,
) -> Vec<NextStep> {
    use veloq_nsys_query::slices::SlicesRow;
    let Some(SlicesRow::Aggregate(top)) = rows.first() else {
        return Vec::new();
    };
    let pattern = top.path.as_deref().unwrap_or(&top.name);
    vec![NextStep {
        hint: format!(
            "List per-instance contributions for the heaviest aggregate \
             (`{pattern}`)."
        ),
        command: format!(
            "veloq slices <trace> --name '{}'{}",
            top.name,
            device_suffix(scope)
        ),
    }]
}

/// Top-row follow-up for `stats`: stats rows are aggregates without a
/// row_id, so the natural drill-down is to `search` the same kind for
/// individual events.
fn stats_next_steps(
    rows: &[veloq_nsys_query::stats::StatRow],
    scope: &ResolvedScope,
) -> Vec<NextStep> {
    let Some(top) = rows.first() else {
        return Vec::new();
    };
    let name_clause = match top.name.as_deref() {
        Some(n) if !n.is_empty() => format!(" --name '{n}'"),
        _ => String::new(),
    };
    vec![NextStep {
        hint: format!(
            "List individual events behind the top stats row (`{}`).",
            top.name.as_deref().unwrap_or(top.kind)
        ),
        command: format!(
            "veloq search <trace> --type {}{}{}",
            top.kind,
            name_clause,
            device_suffix(scope)
        ),
    }]
}

/// Top-row follow-up for `search`: drill into the headline row with
/// `inspect`.
fn search_next_steps(
    rows: &[veloq_nsys_query::event_ref::EventRef],
    _scope: &ResolvedScope,
) -> Vec<NextStep> {
    let Some(top) = rows.first() else {
        return Vec::new();
    };
    vec![NextStep {
        hint: format!(
            "Expand the top hit `{}` with the full headline + NVTX \
             context.",
            top.base().name
        ),
        command: format!("veloq inspect {}", top.base().row_id),
    }]
}

/// Top-row follow-up for `gaps`: inspect both the event that ended the
/// previous activity and the event that started the next one.
fn gaps_next_steps(rows: &[veloq_nsys_query::gaps::Gap]) -> Vec<NextStep> {
    let Some(top) = rows.first() else {
        return Vec::new();
    };
    vec![NextStep {
        hint: format!(
            "Inspect the events bracketing the largest gap \
             (`{}` → `{}`).",
            top.prev.name, top.next.name
        ),
        command: format!("veloq inspect {} {}", top.prev.row_id, top.next.row_id),
    }]
}

/// Project a [`KindFilter`] into the comma-joined string form
/// `applied_scope.kind` carries. `All` is reported as `None` (the
/// default, equivalent to "no kind filter"); `Only(...)` joins the
/// kind names with commas.
fn kinds_csv(kf: &veloq_nsys_query::KindFilter) -> Option<String> {
    use veloq_nsys_query::KindFilter;
    match kf {
        KindFilter::All => None,
        KindFilter::Only(v) => {
            if v.is_empty() {
                None
            } else {
                Some(
                    v.iter()
                        .map(|k| k.as_str())
                        .collect::<Vec<&str>>()
                        .join(","),
                )
            }
        }
    }
}

/// Parse the user's `--sort` string into a `SortSpec` for the request.
/// Returns `None` when the input is empty (lets the module pick its own
/// default), and an `anyhow::Error` on syntax problems.
fn parse_sort_spec(s: &str) -> Result<Option<SortSpec>> {
    let t = s.trim();
    if t.is_empty() {
        return Ok(None);
    }
    let spec = SortSpec::parse(t).with_context(|| format!("invalid --sort `{s}`"))?;
    Ok(Some(spec))
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
    fmt: Format,
    trace: Option<&Path>,
    trace_span: Option<TraceSpan>,
) -> Result<i32> {
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
    let trace = trace.ok_or_else(|| anyhow::anyhow!("internal: nsys verb missing trace path"))?;
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
                        anyhow::bail!(
                            "histograms not yet supported in --by size mode \
                             (the duration-bucket schema doesn't map to byte axes); \
                             deferred to a future byte-histogram PR"
                        );
                    }
                    // NVTX attribution isn't wired through the
                    // byte-axis path yet; reject rather than return
                    // unscoped totals labelled with the user's pattern.
                    if gpu.nvtx.is_some() {
                        anyhow::bail!(
                            "--by size + --nvtx is not yet implemented — \
                             NVTX attribution on the byte-axis path is \
                             deferred. For the duration-axis equivalent, \
                             use `--nvtx` without `--by size`."
                        );
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
                        anyhow::bail!(
                            "--by size does not yet support --group-by {unsupported}. \
                             Supported axes today are name (short/demangled/mangled/no-name) \
                             and device/context/stream. Drop the unsupported axes or \
                             unset --by size."
                        );
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
                                .with_context(|| format!("invalid --sort `{sort}`"))?,
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
                Some(s) => Some(
                    DurationFilter::parse(s)
                        .with_context(|| format!("invalid --duration `{s}`"))?,
                ),
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
                .map(|s| {
                    s.parse::<RowId>()
                        .with_context(|| format!("invalid row_id `{s}`"))
                })
                .collect::<Result<_>>()?;
            let data = veloq_nsys_query::inspect::run(trace, &parsed)?;
            render(fmt, trace, trace_span, "inspect", data, views::inspect_view)?;
        }

        Cmd::Correlate { row_ids, .. } => {
            let parsed: Vec<RowId> = row_ids
                .iter()
                .map(|s| {
                    s.parse::<RowId>()
                        .with_context(|| format!("invalid row_id `{s}`"))
                })
                .collect::<Result<_>>()?;
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
            let row_id = row_id
                .parse::<RowId>()
                .with_context(|| format!("invalid row_id `{row_id}`"))?;
            let env_policy = veloq_nsys_query::ncu_command::EnvPolicy::parse(&env)?;
            let data = veloq_nsys_query::ncu_command::run(
                trace,
                veloq_nsys_query::ncu_command::NcuCommandRequest { row_id, env_policy },
            )?;
            if print {
                print!("{}", data.script);
            } else {
                if fmt != Format::Json {
                    anyhow::bail!(
                        "ncu-command only supports JSON output; use --print for a pipe-ready shell script"
                    );
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
                    anyhow::bail!("slices --group-by path requires --aggregate");
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
            let source = MetricSource::parse(&source)?;
            let bucket_ns = match bucket.as_deref() {
                Some(s) => Some(veloq_nsys_query::metrics::parse_bucket(s)?),
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
                        anyhow::bail!(
                            "--group-by / --name / --cpu / --tid are cpu-* flags; \
                             drop them or switch to `--type cpu-sampling` / `cpu-sched`"
                        );
                    }
                    MetricsRequest::Gpu(GpuMetricsRequest {
                        counter_glob: counter,
                        common: common_req,
                    })
                }
                MetricSource::Nic => {
                    if group_by.is_some() || name.is_some() || cpu.is_some() || tid.is_some() {
                        anyhow::bail!(
                            "--group-by / --name / --cpu / --tid are cpu-* flags; \
                             drop them or switch to `--type cpu-sampling` / `cpu-sched`"
                        );
                    }
                    MetricsRequest::Nic(NicMetricsRequest {
                        counter_glob: counter,
                        common: common_req,
                    })
                }
                MetricSource::CpuSampling => {
                    if counter.is_some() {
                        anyhow::bail!(
                            "--counter is a gpu/nic flag (matches PM counter names); \
                             use `--name <glob>` to filter cpu-sampling rows"
                        );
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
                        anyhow::bail!(
                            "--counter is a gpu/nic flag (matches PM counter names); \
                             cpu-sched has no name field to filter on"
                        );
                    }
                    if name.is_some() {
                        anyhow::bail!(
                            "--name is a cpu-sampling flag (matches stack-frame symbols); \
                             cpu-sched has no name field — use --group-by / --tid / --cpu"
                        );
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
            anyhow::bail!("internal: schema handled before this match")
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

/// `veloq prep --status` — assemble the cache-status payload without
/// rebuilding anything. Reads filesystem metadata only. The
/// parquetdir has no manifest; its contents are
/// whatever `nsys export -t parquetdir` last wrote next to the trace.
fn collect_prep_status(trace: &Path) -> Result<PrepStatusPayload> {
    // Where the parquetdir lives. For a `.nsys-rep`, that's
    // `<trace>.veloq/parquetdir/`; for a directly-passed `_pqtdir/`,
    // that's the input itself.
    let source_path = veloq_nsys_data::nsys_rep::sidecar_source_path(trace);
    let parquet_dir = if source_path.extension().and_then(|e| e.to_str()) == Some("nsys-rep") {
        veloq_nsys_data::nsys_rep::pqtdir_path_for(&source_path)
    } else {
        trace.to_path_buf()
    };
    let mut tables: Vec<String> = if parquet_dir.is_dir() {
        std::fs::read_dir(&parquet_dir)
            .with_context(|| format!("reading parquetdir {}", parquet_dir.display()))?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "parquet"))
            .filter_map(|p| p.file_stem().and_then(|s| s.to_str()).map(str::to_owned))
            .collect()
    } else {
        Vec::new()
    };
    tables.sort();
    let parquet_status = ParquetCacheStatus {
        dir: parquet_dir.display().to_string(),
        present: parquet_dir.is_dir(),
        tables,
    };

    let meta_path = veloq_nsys_data::meta_cache::path_for(&source_path);
    let (present, size_bytes, mtime_secs) = match std::fs::metadata(&meta_path) {
        Ok(m) => {
            let mtime = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64);
            (true, Some(m.len()), mtime)
        }
        Err(_) => (false, None, None),
    };
    let meta_version_on_disk = veloq_nsys_data::meta_cache::read_header(&source_path)
        .ok()
        .flatten()
        .map(|h| h.version);
    // `try_load_existing` returns `Some(_)` only when the sidecar's
    // version + trace fingerprint both validate. Errors fold to
    // `false` so a corrupt or unreadable file shows up as
    // "present but not fingerprint-matching."
    let fingerprint_match = veloq_nsys_data::meta_cache::try_load_existing(&source_path)
        .ok()
        .flatten()
        .is_some();
    let meta_status = SidecarStatus {
        path: meta_path.display().to_string(),
        present,
        size_bytes,
        mtime_secs,
        format_version_expected: veloq_nsys_data::META_CACHE_VERSION,
        format_version_on_disk: meta_version_on_disk,
        fingerprint_match,
    };

    Ok(PrepStatusPayload {
        cache_root: veloq_core::artifact_dir_for(&source_path)
            .display()
            .to_string(),
        parquet_cache: parquet_status,
        meta_cache: meta_status,
    })
}
