//! `veloq gaps --min Nms` — GPU idle-bubble detection.
//!
//! Three scopes, picked via `--scope`:
//!
//! - **`device` (default)**: per device, gap = window where *no
//!   stream* was running GPU work. Multi-stream workloads see only
//!   the real device-wide idle bubbles —
//!   streams running concurrently don't produce phantom gaps on
//!   their idle peers.
//! - **`stream`**: per (device, stream), gap = window between
//!   consecutive events on that stream. Useful for "is this
//!   specific stream getting starved" diagnostics; not the right
//!   default because long-idle streams dominate output.
//! - **`trace`**: across all devices, gap = window where no device
//!   was running GPU work. Useful for distributed / multi-GPU jobs
//!   where the goal is "find moments where the whole rig was idle."
//!
//! Each gap carries the bracketing events (`prev` / `next`) so the
//! agent can `inspect` them or anchor follow-up correlate queries.
//! Under unified scopes (device / trace) the bracketing events may
//! live on different streams; the neighbor's `stream_id` is surfaced
//! so the agent can see which streams induced the boundary.
//!
//! ## Concurrency model
//!
//! `--scope device` and `--scope trace` use a sweep-line
//! (`MAX(end_ns) OVER ... ROWS UNBOUNDED PRECEDING ... 1 PRECEDING`)
//! that handles overlapping events correctly: a gap fires only when
//! the next event's `start_ns` exceeds the running max of all
//! previous events' `end_ns`. `--scope stream` keeps the existing
//! per-stream `LEAD()` formulation, where overlap on a single
//! stream (rare in real captures) produces a non-positive gap that
//! the threshold filter drops.

use crate::query_sql::{exec, gpu_work::GpuWorkSet};
use duckdb::types::Value;
use serde::Serialize;
use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use veloq_core::{
    Direction, SortKeyDef, SortKeySpec, SortSpec,
    time::{TimeWindow, parse_duration_ns},
};
use veloq_nsys_data::Trace;

use crate::{EventKind, NsysQueryError, NsysQueryResult, RowId};

/// Aggregation scope for the gap computation. See the module doc
/// for the per-scope semantics; the request's other filters
/// (`device`, `stream`, `time_window`) compose on top.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GapScope {
    /// Per device, unified across streams. The default — answers
    /// "where was *this device's* GPU idle?" without phantom gaps
    /// on long-idle peer streams.
    #[default]
    Device,
    /// Per (device, stream); kept for the rare "is this specific
    /// stream getting starved" question.
    Stream,
    /// Trace-wide, unified across devices. Gap = window where no
    /// device ran GPU work. Useful for multi-GPU jobs that want to
    /// see whole-rig idle bubbles.
    Trace,
}

impl GapScope {
    pub fn parse(s: &str) -> NsysQueryResult<Self> {
        match s.to_ascii_lowercase().as_str() {
            "device" => Ok(Self::Device),
            "stream" => Ok(Self::Stream),
            "trace" => Ok(Self::Trace),
            other => Err(NsysQueryError::gaps_invalid_scope(other)),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Device => "device",
            Self::Stream => "stream",
            Self::Trace => "trace",
        }
    }
}

#[derive(Debug, Clone)]
pub struct GapsRequest {
    /// Minimum gap duration to report, in nanoseconds.
    pub min_ns: i64,
    /// Aggregation scope — see [`GapScope`]. Default `Device`.
    pub scope: GapScope,
    /// Optional `device_id` filter (NSys `deviceId`). Valid under
    /// every scope; rejected under `--scope trace` only when the
    /// filter would semantically conflict with the unified view.
    pub device: Option<i32>,
    /// Optional `stream_id` filter (NSys `streamId`). Only valid
    /// under `--scope stream` — under unified scopes there's no
    /// per-stream output to filter on. Use the row's `prev` / `next`
    /// neighbors and their `stream_id` to drill down post-hoc.
    pub stream: Option<i64>,
    /// Optional window — keeps gaps whose full `[start, end)` interval
    /// overlaps it. Rows still report the full gap bounds for context.
    pub time_window: Option<TimeWindow>,
    /// Sort spec. `None` falls back to `duration` descending (biggest
    /// bubbles first — the default before sort was customisable).
    pub sort: Option<SortSpec>,
    /// Max rows to return.
    pub limit: usize,
}

impl Default for GapsRequest {
    fn default() -> Self {
        Self {
            min_ns: 1_000_000, // 1ms
            scope: GapScope::default(),
            device: None,
            stream: None,
            time_window: None,
            sort: None,
            limit: 100,
        }
    }
}

/// Sort axes `gaps` supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortKey {
    Duration,
    Start,
    Device,
    Stream,
}

impl SortKeyDef for SortKey {
    fn specs() -> &'static [SortKeySpec<Self>] {
        &[
            SortKeySpec {
                variant: SortKey::Duration,
                canonical: "duration",
                aliases: &["dur", "gap", "gap_ns"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SortKey::Start,
                canonical: "start",
                aliases: &["time", "start_ns"],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: SortKey::Device,
                canonical: "device",
                aliases: &["device_id", "dev"],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: SortKey::Stream,
                canonical: "stream",
                aliases: &["stream_id"],
                default_dir: Direction::Asc,
            },
        ]
    }
}

impl SortKey {
    fn column(self) -> &'static str {
        match self {
            Self::Duration => "gap_ns",
            Self::Start => "gap_start_ns",
            Self::Device => "device_id",
            Self::Stream => "stream_id",
        }
    }
}

fn gaps_sort_sql(spec: &SortSpec) -> NsysQueryResult<String> {
    let mut resolved = Vec::new();
    for f in spec.fields() {
        let (k, d) = SortKey::from_field(f).map_err(NsysQueryError::gaps_sort_invalid)?;
        resolved.push((k.column(), d));
    }
    // gap_start_ns as tiebreaker — guarantees deterministic order for
    // ties on the primary key.
    Ok(veloq_core::sort::build_order_by(&resolved, "gap_start_ns"))
}

impl GapsRequest {
    /// Parse the `--min-duration` CLI string ("1ms", "100us", "1.2s",
    /// "42ns") into ns.
    pub fn parse_min_duration(s: &str) -> NsysQueryResult<i64> {
        parse_duration_ns(s).map_err(|source| NsysQueryError::GapsMinDurationInvalid {
            value: s.to_string(),
            source,
        })
    }
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct GapsResponse {
    pub min_ns: i64,
    /// Echoed scope (`device` / `stream` / `trace`) so agents that
    /// see the response in isolation know which gap semantic
    /// produced it — the per-scope `key` format also differs.
    pub scope: &'static str,
    /// Gaps returned (after LIMIT).
    pub count: usize,
    /// Gaps matching `--min` + filters before LIMIT was applied.
    pub total_matched: i64,
    /// Resolved `--time-range`, if any (absolute ns).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_window_ns: Option<(i64, i64)>,
    /// Canonical primary table. Each row is one GPU idle bubble.
    pub rows: Vec<Gap>,
    /// Stream-level summary covering the same `(device, stream)`
    /// scope as `rows`. Surfaced so agents can pre-filter on
    /// `busy_ratio` (e.g. "ignore streams that ran <5% of the
    /// window") before iterating gaps — long-idle streams produce
    /// massive trace-wide gaps that dominate the response otherwise.
    pub auxiliary: GapsAuxiliary,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct GapsAuxiliary {
    /// One entry per `(device_id, stream_id)` that has at least one
    /// event in the gaps scope (after `--device` / `--stream` /
    /// `--time-range` filtering). Sorted by `(device_id, stream_id)`
    /// ascending so two traces with the same topology produce
    /// matching index positions for jq joins.
    pub streams: Vec<StreamActivity>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct StreamActivity {
    /// Cross-trace key. `stream|dev:<device_id>|stream:<stream_id>`.
    pub key: String,
    pub device_id: i32,
    pub stream_id: i64,
    /// Sum of in-scope event durations on this stream. Events
    /// straddling the `--time-range` boundary contribute only their
    /// in-window slice — matches how `stats` clips durations.
    pub busy_ns: i64,
    /// Wall-clock span covered by this scope: `--time-range`
    /// duration if set, otherwise the trace's primary origin span.
    pub span_ns: i64,
    /// `busy_ns / span_ns`. 1.0 = saturated; near 0 = effectively
    /// idle. `f64::NAN` when `span_ns == 0` (degenerate trace);
    /// agents should treat NaN as "unknown" rather than divide.
    pub busy_ratio: f64,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct Gap {
    /// Cross-trace key. Format depends on scope:
    /// - `device` → `gap|dev:<device_id>|@<start_ns>`
    /// - `stream` → `gap|dev:<device_id>|stream:<stream_id>|@<start_ns>`
    /// - `trace`  → `gap|@<start_ns>`
    ///
    /// Two runs of the same workload produce matching keys at
    /// matching axes; agents pre-normalize using envelope
    /// `trace_span.origin_ns`.
    pub key: String,
    /// `None` only under `--scope trace`, where the gap is a
    /// trace-wide bubble with no single device axis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_id: Option<i32>,
    /// `Some` only under `--scope stream`. Under `device` / `trace`
    /// the bracketing events may live on different streams; consult
    /// `prev.stream_id` / `next.stream_id` instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_id: Option<i64>,
    pub start_ns: i64,
    pub end_ns: i64,
    pub duration_ns: i64,
    pub prev: GapNeighbor,
    pub next: GapNeighbor,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct GapNeighbor {
    pub row_id: RowId,
    pub name: String,
    /// For `prev`: the event's `end_ns` (= gap start). For `next`: the
    /// event's `start_ns` (= gap end). Same value as the enclosing
    /// `Gap`'s endpoint — exposed here so the agent doesn't have to
    /// recompute.
    pub timestamp_ns: i64,
    /// Stream the bracketing event ran on. Under unified scopes
    /// (`device` / `trace`) this is the only place the stream
    /// shows up — the enclosing `Gap.stream_id` is `None` there.
    pub stream_id: i64,
}

pub fn run<P: AsRef<Path>>(path: P, req: GapsRequest) -> NsysQueryResult<GapsResponse> {
    crate::check_limit(req.limit)?;
    if req.min_ns <= 0 {
        return Err(NsysQueryError::GapsMinTooSmall { min_ns: req.min_ns });
    }
    // --stream filter only makes sense under per-stream scope.
    // Under unified scopes (device / trace) we'd silently drop
    // gaps bracketed by events on other streams — confusing rather
    // than useful.
    if req.stream.is_some() && req.scope != GapScope::Stream {
        return Err(NsysQueryError::GapsStreamRequiresStreamScope {
            scope: req.scope.as_str(),
        });
    }
    // --device under --scope trace: the row-level device_id is
    // projected NULL (a trace-scope gap spans every device), so a
    // `device_id = ?` filter would silently drop every row. Reject
    // upfront instead. Use `--scope device --device N` for per-device.
    if let (Some(dev), GapScope::Trace) = (req.device, req.scope) {
        return Err(NsysQueryError::GapsDeviceInTraceScope { device: dev });
    }
    // Sort by `stream` requires per-stream scope. Sort by `device`
    // is meaningless under `trace` (single all-device partition).
    if let Some(s) = &req.sort {
        for f in s.fields() {
            let (k, _) = SortKey::from_field(f).map_err(NsysQueryError::gaps_sort_invalid)?;
            match (k, req.scope) {
                (SortKey::Stream, scope) if scope != GapScope::Stream => {
                    return Err(NsysQueryError::GapsSortStreamRequiresStreamScope {
                        scope: scope.as_str(),
                    });
                }
                (SortKey::Device, GapScope::Trace) => {
                    return Err(NsysQueryError::GapsSortDeviceInTraceScope);
                }
                _ => {}
            }
        }
    }

    let trace = Trace::open(path).map_err(NsysQueryError::trace_open)?;
    let abs_window = trace
        .resolve_window(req.time_window)
        .map_err(NsysQueryError::time_window_resolve)?;

    // Kernel / memcpy / memset / graph-trace rows are the GPU work
    // that keeps the device busy. Runtime API calls (cudaLaunchKernel,
    // cudaMemcpyAsync) are CPU-side and don't count; sync events are
    // CPU blocking and intentionally excluded from the "device in
    // flight" definition.
    let Some(event_source) = gpu_event_source(&trace, abs_window)? else {
        return Ok(GapsResponse {
            min_ns: req.min_ns,
            scope: req.scope.as_str(),
            count: 0,
            total_matched: 0,
            time_window_ns: abs_window,
            rows: Vec::new(),
            auxiliary: GapsAuxiliary {
                streams: Vec::new(),
            },
        });
    };
    let union = event_source.sql.as_str();

    let gap_sql = match req.scope {
        GapScope::Stream => build_stream_sql(union, &req, abs_window)?,
        GapScope::Device => {
            build_unified_sql(union, &req, abs_window, /*partition_device=*/ true)?
        }
        GapScope::Trace => {
            build_unified_sql(union, &req, abs_window, /*partition_device=*/ false)?
        }
    };

    let mut gaps = hydrate_gap_rows(
        trace.conn(),
        &gap_sql.rows_sql,
        &gap_sql.rows_params,
        req.scope,
    )?;
    let total_from_rows = truncate_gap_rows_to_limit(&mut gaps, req.limit);
    if event_source.needs_name_hydration {
        hydrate_gap_neighbor_names(&trace, &mut gaps)?;
    }
    let total_matched = match total_from_rows {
        Some(total) => total,
        None => hydrate_gap_total(trace.conn(), &gap_sql.total_sql, &gap_sql.total_params)?,
    };

    let span_ns = match abs_window {
        Some((s, e)) => (e - s).max(0),
        None => {
            let (origins, _) = trace.read_origins().map_err(NsysQueryError::data)?;
            origins.primary.duration_ns().max(0)
        }
    };
    let streams = compute_stream_activity(&trace, union, &req, abs_window, span_ns)?;

    Ok(GapsResponse {
        min_ns: req.min_ns,
        scope: req.scope.as_str(),
        count: gaps.len(),
        total_matched,
        time_window_ns: abs_window,
        rows: gaps,
        auxiliary: GapsAuxiliary { streams },
    })
}

fn hydrate_gap_rows(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
    scope: GapScope,
) -> NsysQueryResult<Vec<Gap>> {
    let rows = exec::query_rows(conn, sql, params, exec::GAPS_GAP, gap_sql_row)?;
    rows.into_iter()
        .map(|row| gap_from_sql_row(scope, row))
        .collect::<NsysQueryResult<Vec<_>>>()
}

fn truncate_gap_rows_to_limit(gaps: &mut Vec<Gap>, limit: usize) -> Option<i64> {
    if gaps.len() > limit {
        gaps.truncate(limit);
        None
    } else {
        Some(usize_to_i64_saturating(gaps.len()))
    }
}

fn limit_probe_value(limit: usize) -> i64 {
    usize_to_i64_saturating(limit.saturating_add(1))
}

fn usize_to_i64_saturating(value: usize) -> i64 {
    match i64::try_from(value) {
        Ok(value) => value,
        Err(_) => i64::MAX,
    }
}

fn hydrate_gap_total(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
) -> NsysQueryResult<i64> {
    Ok(exec::query_optional_row(conn, sql, params, exec::GAPS_GAP, |row| row.get(0))?.unwrap_or(0))
}

struct GapSqlRow {
    device_id: Option<i32>,
    stream_id: Option<i64>,
    start_ns: i64,
    end_ns: i64,
    duration_ns: i64,
    prev_kind: String,
    prev_row_num: i64,
    prev_name: String,
    prev_stream_id: i64,
    next_kind: String,
    next_row_num: i64,
    next_name: String,
    next_stream_id: i64,
}

fn gap_sql_row(row: &duckdb::Row<'_>) -> Result<GapSqlRow, duckdb::Error> {
    // Column layout is stable across scope SQL paths.
    //   0 device_id (i32, NULLABLE under scope=trace)
    //   1 stream_id (i64, NULLABLE under scope=device|trace)
    //   2 start_ns  3 end_ns  4 duration_ns
    //   5 prev_kind 6 prev_row 7 prev_name 8 prev_stream_id
    //   9 next_kind 10 next_row 11 next_name 12 next_stream_id
    Ok(GapSqlRow {
        device_id: row.get(0)?,
        stream_id: row.get(1)?,
        start_ns: row.get(2)?,
        end_ns: row.get(3)?,
        duration_ns: row.get(4)?,
        prev_kind: row.get(5)?,
        prev_row_num: row.get(6)?,
        prev_name: row.get(7)?,
        prev_stream_id: row.get(8)?,
        next_kind: row.get(9)?,
        next_row_num: row.get(10)?,
        next_name: row.get(11)?,
        next_stream_id: row.get(12)?,
    })
}

fn gap_from_sql_row(scope: GapScope, row: GapSqlRow) -> NsysQueryResult<Gap> {
    let key = match (scope, row.device_id, row.stream_id) {
        (GapScope::Stream, Some(d), Some(s)) => {
            format!("gap|dev:{d}|stream:{s}|@{}", row.start_ns)
        }
        (GapScope::Device, Some(d), _) => format!("gap|dev:{d}|@{}", row.start_ns),
        (GapScope::Trace, _, _) => format!("gap|@{}", row.start_ns),
        // Shouldn't happen under correct SQL — scope-stream implies both
        // device + stream are populated, scope-device implies device is
        // populated. Fall back rather than bail so a SQL quirk doesn't kill
        // the whole response.
        _ => format!("gap|@{}", row.start_ns),
    };

    Ok(Gap {
        key,
        device_id: row.device_id,
        stream_id: row.stream_id,
        start_ns: row.start_ns,
        end_ns: row.end_ns,
        duration_ns: row.duration_ns,
        prev: GapNeighbor {
            row_id: RowId::new(parse_kind(&row.prev_kind)?, row.prev_row_num),
            name: row.prev_name,
            timestamp_ns: row.start_ns,
            stream_id: row.prev_stream_id,
        },
        next: GapNeighbor {
            row_id: RowId::new(parse_kind(&row.next_kind)?, row.next_row_num),
            name: row.next_name,
            timestamp_ns: row.end_ns,
            stream_id: row.next_stream_id,
        },
    })
}

fn hydrate_gap_neighbor_names(trace: &Trace, gaps: &mut [Gap]) -> NsysQueryResult<()> {
    if gaps.is_empty() {
        return Ok(());
    }

    let mut ids_by_kind: HashMap<EventKind, BTreeSet<i64>> = HashMap::new();
    for gap in gaps.iter() {
        ids_by_kind
            .entry(gap.prev.row_id.kind)
            .or_default()
            .insert(gap.prev.row_id.rowid);
        ids_by_kind
            .entry(gap.next.row_id.kind)
            .or_default()
            .insert(gap.next.row_id.rowid);
    }

    let mut names: HashMap<RowId, String> = HashMap::new();
    for (kind, ids) in ids_by_kind {
        if ids.is_empty() || !trace.table_exists(kind.table()) {
            continue;
        }
        let placeholders = (0..ids.len()).map(|_| "?").collect::<Vec<_>>().join(", ");
        let table = kind.table();
        let name_expr = crate::kind_sql::display_name_expr(kind);
        let joins = crate::kind_sql::name_joins(kind);
        let sql = format!(
            r#"
            SELECT
                t.rowid AS row_id,
                {name_expr} AS name
            FROM nsight.{table} t {joins}
            WHERE t.rowid IN ({placeholders})
            "#
        );
        let params: Vec<Value> = ids.into_iter().map(Value::BigInt).collect();
        for row in exec::query_rows(
            trace.conn(),
            &sql,
            &params,
            exec::GAPS_NAME_LOOKUP,
            gap_name_sql_row,
        )? {
            names.insert(RowId::new(kind, row.row_id), row.name);
        }
    }

    for gap in gaps {
        if let Some(name) = names.get(&gap.prev.row_id) {
            gap.prev.name.clone_from(name);
        }
        if let Some(name) = names.get(&gap.next.row_id) {
            gap.next.name.clone_from(name);
        }
    }
    Ok(())
}

struct GapNameSqlRow {
    row_id: i64,
    name: String,
}

fn gap_name_sql_row(row: &duckdb::Row<'_>) -> Result<GapNameSqlRow, duckdb::Error> {
    Ok(GapNameSqlRow {
        row_id: row.get(0)?,
        name: row.get(1)?,
    })
}

/// Per-(device, stream) busy time. Same `(device, stream, time-window)`
/// filters as the gap query — keeps the auxiliary scope consistent
/// with `rows` so a `--stream 7` request returns one stream's
/// activity, not all of them.
fn compute_stream_activity(
    trace: &Trace,
    union: &str,
    req: &GapsRequest,
    abs_window: Option<(i64, i64)>,
    span_ns: i64,
) -> NsysQueryResult<Vec<StreamActivity>> {
    // Window-clip the durations the same way `stats` does so events
    // straddling the boundary don't over-count: an event 0..200ms
    // inside a 50..150ms window contributes 100ms, not 200ms. The
    // outer WHERE still uses the overlap predicate so we keep events
    // whose body touches the window.
    let (duration_expr, mut where_parts, mut params): (String, Vec<String>, Vec<Value>) =
        match abs_window {
            Some((s, e)) => {
                let p = vec![
                    Value::BigInt(e),
                    Value::BigInt(s),
                    Value::BigInt(e),
                    Value::BigInt(s),
                ];
                (
                    "LEAST(end_ns, ?) - GREATEST(start_ns, ?)".to_string(),
                    vec!["start_ns < ? AND end_ns > ?".to_string()],
                    p,
                )
            }
            None => ("end_ns - start_ns".to_string(), Vec::new(), Vec::new()),
        };

    crate::kind_policy::LocationFilter {
        device: req.device,
        stream: req.stream,
    }
    .push_where(&mut where_parts, &mut params);
    let where_clause = if where_parts.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", where_parts.join(" AND "))
    };

    let sql = format!(
        r#"
        WITH events AS ({union})
        SELECT
            device_id,
            stream_id,
            CAST(SUM({duration_expr}) AS BIGINT) AS busy_ns
        FROM events
        {where_clause}
        GROUP BY device_id, stream_id
        ORDER BY device_id, stream_id
        "#
    );

    hydrate_stream_activity_rows(trace.conn(), &sql, &params, span_ns)
}

fn hydrate_stream_activity_rows(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
    span_ns: i64,
) -> NsysQueryResult<Vec<StreamActivity>> {
    let rows = exec::query_rows(
        conn,
        sql,
        params,
        exec::GAPS_STREAM_ACTIVITY,
        stream_activity_sql_row,
    )?;

    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let busy_ns = row.busy_ns.unwrap_or(0);
        let busy_ratio = if span_ns > 0 {
            (busy_ns as f64) / (span_ns as f64)
        } else {
            f64::NAN
        };
        out.push(StreamActivity {
            key: format!("stream|dev:{}|stream:{}", row.device_id, row.stream_id),
            device_id: row.device_id,
            stream_id: row.stream_id,
            busy_ns,
            span_ns,
            busy_ratio,
        });
    }
    Ok(out)
}

struct StreamActivitySqlRow {
    device_id: i32,
    stream_id: i64,
    busy_ns: Option<i64>,
}

fn stream_activity_sql_row(row: &duckdb::Row<'_>) -> Result<StreamActivitySqlRow, duckdb::Error> {
    Ok(StreamActivitySqlRow {
        device_id: row.get(0)?,
        stream_id: row.get(1)?,
        busy_ns: row.get(2)?,
    })
}

struct GpuEventSource {
    sql: String,
    needs_name_hydration: bool,
}

fn gpu_event_source(
    trace: &Trace,
    abs_window: Option<(i64, i64)>,
) -> NsysQueryResult<Option<GpuEventSource>> {
    if veloq_nsys_data::gpu_work_events::view_available(trace) {
        return Ok(Some(sidecar_gpu_event_source()));
    }

    // Full-trace gaps already pays the whole-trace scan cost; lazily
    // building the normalized event sidecar makes the first full query
    // useful for future exploratory gaps calls. Windowed cold queries
    // keep the direct local-frontier SQL path to avoid adding prep cost
    // to the common small-window workflow.
    if abs_window.is_none() {
        match veloq_nsys_data::gpu_work_events::ensure_sidecar(trace) {
            Ok(_) => {
                if veloq_nsys_data::gpu_work_events::view_available(trace) {
                    return Ok(Some(sidecar_gpu_event_source()));
                }
            }
            Err(err) => {
                log::warn!(
                    "gpu_work_events: lazy sidecar build skipped; falling back to raw gaps SQL: {err:#}"
                );
            }
        }
    }

    cold_gpu_event_source(trace)
}

fn sidecar_gpu_event_source() -> GpuEventSource {
    GpuEventSource {
        sql: r#"
            SELECT
                kind,
                row_id,
                device_id,
                stream_id,
                CAST('' AS VARCHAR) AS name,
                start_ns,
                end_ns
            FROM nsight.gpu_work_events
        "#
        .to_string(),
        needs_name_hydration: true,
    }
}

fn cold_gpu_event_source(trace: &Trace) -> NsysQueryResult<Option<GpuEventSource>> {
    let work = GpuWorkSet::from_data_definition()?;
    let mut subqueries: Vec<String> = Vec::new();
    for kind in work.present_in(trace) {
        subqueries.push(per_kind_select(kind)?);
    }
    if subqueries.is_empty() {
        return Ok(None);
    }
    Ok(Some(GpuEventSource {
        sql: subqueries.join(" UNION ALL "),
        needs_name_hydration: false,
    }))
}

fn parse_kind(s: &str) -> NsysQueryResult<EventKind> {
    match EventKind::parse(s) {
        Some(kind) => Ok(kind),
        None => Err(NsysQueryError::internal_sql_kind_tag_invalid("gaps", s)),
    }
}

/// SQL and bind params for a gaps row query plus its minimal count
/// query. The count query is only executed when the row query proves
/// `LIMIT` truncated the result set.
struct GapSqlQuery {
    rows_sql: String,
    rows_params: Vec<Value>,
    total_sql: String,
    total_params: Vec<Value>,
}

fn push_gap_filters(
    params: &mut Vec<Value>,
    abs_window: Option<(i64, i64)>,
    min_ns: i64,
) -> String {
    let mut where_parts: Vec<String> = Vec::new();
    if let Some((s, e)) = abs_window {
        where_parts.push("gap_start_ns < ? AND gap_end_ns > ?".to_string());
        params.push(Value::BigInt(e));
        params.push(Value::BigInt(s));
    }
    where_parts.push("gap_ns >= ?".to_string());
    params.push(Value::BigInt(min_ns));
    format!("WHERE {}", where_parts.join(" AND "))
}

/// Build the per-stream `--scope stream` SQL: `LEAD()` between
/// consecutive events on the same (device, stream).
fn build_stream_sql(
    union: &str,
    req: &GapsRequest,
    abs_window: Option<(i64, i64)>,
) -> NsysQueryResult<GapSqlQuery> {
    let (event_ctes, event_source, mut rows_params) =
        build_stream_event_input(union, req, abs_window);
    let where_clause = push_gap_filters(&mut rows_params, abs_window, req.min_ns);
    rows_params.push(Value::BigInt(limit_probe_value(req.limit)));

    let sort_spec = req
        .sort
        .clone()
        .unwrap_or_else(|| SortSpec::single("duration"));
    let order_by = gaps_sort_sql(&sort_spec)?;

    let rows_sql = format!(
        r#"
        WITH {event_ctes},
        sequenced AS (
            SELECT
                kind, row_id, device_id, stream_id, name, start_ns, end_ns,
                LEAD(start_ns) OVER w  AS next_start_ns,
                LEAD(row_id)   OVER w  AS next_row_id,
                LEAD(kind)     OVER w  AS next_kind,
                LEAD(name)     OVER w  AS next_name,
                LEAD(stream_id) OVER w AS next_stream_id
            FROM {event_source}
            WINDOW w AS (PARTITION BY device_id, stream_id ORDER BY start_ns, row_id)
        ),
        filtered AS (
            SELECT
                device_id, stream_id,
                end_ns AS gap_start_ns,
                next_start_ns AS gap_end_ns,
                next_start_ns - end_ns AS gap_ns,
                kind AS prev_kind, row_id AS prev_row_id, name AS prev_name,
                stream_id AS prev_stream_id,
                next_kind, next_row_id, next_name, next_stream_id
            FROM sequenced
            WHERE next_start_ns IS NOT NULL
        ),
        clipped AS (
            SELECT * FROM filtered {where_clause}
        )
        SELECT
            device_id, stream_id, gap_start_ns, gap_end_ns, gap_ns,
            prev_kind, prev_row_id, prev_name, prev_stream_id,
            next_kind, next_row_id, next_name, next_stream_id
        FROM clipped
        ORDER BY {order_by}
        LIMIT ?
        "#
    );

    let (total_event_ctes, total_event_source, mut total_params) =
        build_stream_event_input(union, req, abs_window);
    let total_where_clause = push_gap_filters(&mut total_params, abs_window, req.min_ns);
    let total_sql = format!(
        r#"
        WITH {total_event_ctes},
        sequenced AS (
            SELECT
                start_ns, end_ns,
                LEAD(start_ns) OVER w AS next_start_ns
            FROM {total_event_source}
            WINDOW w AS (PARTITION BY device_id, stream_id ORDER BY start_ns, row_id)
        ),
        filtered AS (
            SELECT
                end_ns AS gap_start_ns,
                next_start_ns AS gap_end_ns,
                next_start_ns - end_ns AS gap_ns
            FROM sequenced
            WHERE next_start_ns IS NOT NULL
        ),
        clipped AS (
            SELECT 1
            FROM filtered
            {total_where_clause}
        )
        SELECT CAST(COUNT(*) AS BIGINT) AS total_matched
        FROM clipped
        "#
    );

    Ok(GapSqlQuery {
        rows_sql,
        rows_params,
        total_sql,
        total_params,
    })
}

/// Build the `--scope device` (`partition_device=true`) or `--scope
/// trace` (`partition_device=false`) SQL: sweep-line over the
/// unioned event interval set, using `MAX(end_ns) OVER … ROWS
/// UNBOUNDED PRECEDING … 1 PRECEDING` to find when the next event
/// starts after the running maximum of all previous ends. Concurrent
/// events on different streams overlap correctly — the gap fires
/// only when the device (or trace) is actually idle.
///
/// The `arg_max(value, ord) OVER ...` window aggregates pick the
/// previous event's row_id / kind / name / stream_id at the same
/// instant `MAX(end_ns)` does, so the bracketing-event context is
/// taken from the event that actually closed out the running max.
fn build_unified_sql(
    union: &str,
    req: &GapsRequest,
    abs_window: Option<(i64, i64)>,
    partition_device: bool,
) -> NsysQueryResult<GapSqlQuery> {
    let partition = if partition_device {
        "PARTITION BY device_id ORDER BY start_ns, row_id"
    } else {
        "ORDER BY start_ns, row_id"
    };

    let (event_ctes, event_source, mut rows_params) =
        build_unified_event_input(union, req, abs_window, partition_device);
    let where_clause = push_gap_filters(&mut rows_params, abs_window, req.min_ns);
    rows_params.push(Value::BigInt(limit_probe_value(req.limit)));

    let sort_spec = req
        .sort
        .clone()
        .unwrap_or_else(|| SortSpec::single("duration"));
    let order_by = gaps_sort_sql(&sort_spec)?;

    // `stream_id` projection: under `--scope stream` it'd be the
    // partition key. Under unified scopes it's irrelevant at the
    // gap row level (the bracketing events on prev/next carry the
    // stream context). Project NULL so the row reader's i64 column
    // resolves to None.
    let stream_id_proj = "CAST(NULL AS BIGINT)";
    // Under `--scope trace` we also have no single device axis.
    let device_id_proj = if partition_device {
        "device_id"
    } else {
        "CAST(NULL AS INTEGER)"
    };

    let rows_sql = format!(
        r#"
        WITH {event_ctes},
        with_prev AS (
            SELECT
                device_id, stream_id, kind, row_id, name, start_ns, end_ns,
                MAX(end_ns)          OVER win AS prev_max_end,
                arg_max(row_id,    end_ns) OVER win AS prev_row_id,
                arg_max(kind,      end_ns) OVER win AS prev_kind,
                arg_max(name,      end_ns) OVER win AS prev_name,
                arg_max(stream_id, end_ns) OVER win AS prev_stream_id
            FROM {event_source}
            WINDOW win AS ({partition} ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        ),
        filtered AS (
            SELECT
                {device_id_proj} AS device_id,
                {stream_id_proj} AS stream_id,
                prev_max_end AS gap_start_ns,
                start_ns     AS gap_end_ns,
                start_ns - prev_max_end AS gap_ns,
                prev_kind, prev_row_id, prev_name, prev_stream_id,
                kind      AS next_kind,
                row_id    AS next_row_id,
                name      AS next_name,
                stream_id AS next_stream_id
            FROM with_prev
            WHERE prev_max_end IS NOT NULL AND start_ns > prev_max_end
        ),
        clipped AS (
            SELECT * FROM filtered {where_clause}
        )
        SELECT
            device_id, stream_id, gap_start_ns, gap_end_ns, gap_ns,
            prev_kind, prev_row_id, prev_name, prev_stream_id,
            next_kind, next_row_id, next_name, next_stream_id
        FROM clipped
        ORDER BY {order_by}
        LIMIT ?
        "#
    );

    let (total_event_ctes, total_event_source, mut total_params) =
        build_unified_event_input(union, req, abs_window, partition_device);
    let total_where_clause = push_gap_filters(&mut total_params, abs_window, req.min_ns);
    let total_sql = format!(
        r#"
        WITH {total_event_ctes},
        with_prev AS (
            SELECT
                start_ns, end_ns,
                MAX(end_ns) OVER win AS prev_max_end
            FROM {total_event_source}
            WINDOW win AS ({partition} ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        ),
        filtered AS (
            SELECT
                prev_max_end AS gap_start_ns,
                start_ns AS gap_end_ns,
                start_ns - prev_max_end AS gap_ns
            FROM with_prev
            WHERE prev_max_end IS NOT NULL AND start_ns > prev_max_end
        ),
        clipped AS (
            SELECT 1
            FROM filtered
            {total_where_clause}
        )
        SELECT CAST(COUNT(*) AS BIGINT) AS total_matched
        FROM clipped
        "#
    );

    Ok(GapSqlQuery {
        rows_sql,
        rows_params,
        total_sql,
        total_params,
    })
}

const GAP_EVENT_COLUMNS: &str = "kind, row_id, device_id, stream_id, name, start_ns, end_ns";
const GAP_EVENT_COLUMNS_E: &str =
    "e.kind, e.row_id, e.device_id, e.stream_id, e.name, e.start_ns, e.end_ns";

fn build_stream_event_input(
    union: &str,
    req: &GapsRequest,
    abs_window: Option<(i64, i64)>,
) -> (String, &'static str, Vec<Value>) {
    let mut scope_parts = Vec::new();
    let mut params = Vec::new();

    crate::kind_policy::LocationFilter {
        device: req.device,
        stream: req.stream,
    }
    .push_where(&mut scope_parts, &mut params);
    let scope_where = if scope_parts.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", scope_parts.join(" AND "))
    };

    let events_cte = format!("events AS NOT MATERIALIZED ({union})");
    let needs_scoped_events = abs_window.is_some() || !scope_parts.is_empty();
    if !needs_scoped_events {
        return (events_cte, "events", params);
    }

    let scoped_cte = format!(
        r#"
        scoped_events AS NOT MATERIALIZED (
            SELECT {GAP_EVENT_COLUMNS}
            FROM events
            {scope_where}
        )
        "#
    );

    let Some((s, e)) = abs_window else {
        return (
            format!("{events_cte}, {scoped_cte}"),
            "scoped_events",
            params,
        );
    };

    let local_cte = format!(
        r#"
        prefix_starts AS (
            SELECT device_id, stream_id, MAX(start_ns) AS start_ns
            FROM scoped_events
            WHERE start_ns <= ?
            GROUP BY device_id, stream_id
        ),
        prefix_rows AS (
            SELECT e.device_id, e.stream_id, e.start_ns, MAX(e.row_id) AS row_id
            FROM scoped_events e
            JOIN prefix_starts p
              ON e.device_id = p.device_id
             AND e.stream_id = p.stream_id
             AND e.start_ns = p.start_ns
            GROUP BY e.device_id, e.stream_id, e.start_ns
        ),
        prefix_events AS (
            SELECT {GAP_EVENT_COLUMNS_E}
            FROM scoped_events e
            JOIN prefix_rows p
              ON e.device_id = p.device_id
             AND e.stream_id = p.stream_id
             AND e.start_ns = p.start_ns
             AND e.row_id = p.row_id
        ),
        suffix_starts AS (
            SELECT device_id, stream_id, MIN(start_ns) AS start_ns
            FROM scoped_events
            WHERE start_ns >= ?
            GROUP BY device_id, stream_id
        ),
        suffix_rows AS (
            SELECT e.device_id, e.stream_id, e.start_ns, MIN(e.row_id) AS row_id
            FROM scoped_events e
            JOIN suffix_starts p
              ON e.device_id = p.device_id
             AND e.stream_id = p.stream_id
             AND e.start_ns = p.start_ns
            GROUP BY e.device_id, e.stream_id, e.start_ns
        ),
        suffix_events AS (
            SELECT {GAP_EVENT_COLUMNS_E}
            FROM scoped_events e
            JOIN suffix_rows p
              ON e.device_id = p.device_id
             AND e.stream_id = p.stream_id
             AND e.start_ns = p.start_ns
             AND e.row_id = p.row_id
        ),
        local_events AS (
            SELECT {GAP_EVENT_COLUMNS}
            FROM scoped_events
            WHERE start_ns < ? AND end_ns > ?
            UNION
            SELECT {GAP_EVENT_COLUMNS}
            FROM prefix_events
            UNION
            SELECT {GAP_EVENT_COLUMNS}
            FROM suffix_events
        )
        "#
    );

    params.push(Value::BigInt(s));
    params.push(Value::BigInt(e));
    params.push(Value::BigInt(e));
    params.push(Value::BigInt(s));

    (
        format!("{events_cte}, {scoped_cte}, {local_cte}"),
        "local_events",
        params,
    )
}

fn build_unified_event_input(
    union: &str,
    req: &GapsRequest,
    abs_window: Option<(i64, i64)>,
    partition_device: bool,
) -> (String, &'static str, Vec<Value>) {
    let mut scope_parts = Vec::new();
    let mut params = Vec::new();

    // `--stream` is rejected upstream under unified scopes; device only.
    crate::kind_policy::LocationFilter {
        device: req.device,
        stream: None,
    }
    .push_where(&mut scope_parts, &mut params);
    let scope_where = if scope_parts.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", scope_parts.join(" AND "))
    };

    let events_cte = format!("events AS NOT MATERIALIZED ({union})");
    let needs_scoped_events = abs_window.is_some() || !scope_parts.is_empty();
    if !needs_scoped_events {
        return (events_cte, "events", params);
    }

    let scoped_cte = format!(
        r#"
        scoped_events AS NOT MATERIALIZED (
            SELECT {GAP_EVENT_COLUMNS}
            FROM events
            {scope_where}
        )
        "#
    );

    let Some((s, e)) = abs_window else {
        return (
            format!("{events_cte}, {scoped_cte}"),
            "scoped_events",
            params,
        );
    };

    let (prefix_device_expr, suffix_device_expr, frontier_group, prefix_having, suffix_having) =
        if partition_device {
            ("device_id", "device_id", "GROUP BY device_id", "", "")
        } else {
            (
                "CAST(arg_max(device_id, end_ns) AS INTEGER)",
                "CAST(arg_min(device_id, start_ns) AS INTEGER)",
                "",
                "HAVING MAX(end_ns) IS NOT NULL",
                "HAVING MIN(start_ns) IS NOT NULL",
            )
        };
    let local_cte = format!(
        r#"
        prefix_events AS (
            SELECT
                arg_max(kind, end_ns) AS kind,
                arg_max(row_id, end_ns) AS row_id,
                {prefix_device_expr} AS device_id,
                CAST(arg_max(stream_id, end_ns) AS BIGINT) AS stream_id,
                arg_max(name, end_ns) AS name,
                CAST(arg_max(start_ns, end_ns) AS BIGINT) AS start_ns,
                MAX(end_ns) AS end_ns
            FROM scoped_events
            WHERE start_ns <= ?
            {frontier_group}
            {prefix_having}
        ),
        suffix_events AS (
            SELECT
                arg_min(kind, start_ns) AS kind,
                arg_min(row_id, start_ns) AS row_id,
                {suffix_device_expr} AS device_id,
                CAST(arg_min(stream_id, start_ns) AS BIGINT) AS stream_id,
                arg_min(name, start_ns) AS name,
                MIN(start_ns) AS start_ns,
                CAST(arg_min(end_ns, start_ns) AS BIGINT) AS end_ns
            FROM scoped_events
            WHERE start_ns >= ?
            {frontier_group}
            {suffix_having}
        ),
        local_events AS (
            SELECT {GAP_EVENT_COLUMNS}
            FROM scoped_events
            WHERE start_ns < ? AND end_ns > ?
            UNION
            SELECT {GAP_EVENT_COLUMNS}
            FROM prefix_events
            UNION
            SELECT {GAP_EVENT_COLUMNS}
            FROM suffix_events
        )
        "#
    );

    params.push(Value::BigInt(s));
    params.push(Value::BigInt(e));
    params.push(Value::BigInt(e));
    params.push(Value::BigInt(s));

    (
        format!("{events_cte}, {scoped_cte}, {local_cte}"),
        "local_events",
        params,
    )
}

/// SQL fragment emitting (kind, row_id, device_id, stream_id, name,
/// start_ns, end_ns) for one event table. Name resolution + memcpy
/// copyKind labels come from `kind_sql` so GPU work branches stay
/// in sync with the other commands.
///
/// Returns an error when called with a host-side kind. The normal
/// caller derives kinds from the shared NSys GPU work definition, but
/// the workspace's no-panic policy routes the precondition through
/// `Result` instead of `unreachable!`.
fn per_kind_select(kind: EventKind) -> NsysQueryResult<String> {
    if matches!(kind, EventKind::Runtime | EventKind::Osrt | EventKind::Nvtx) {
        return Err(NsysQueryError::internal_unsupported_kind(
            "gaps",
            kind.as_str(),
        ));
    }
    let table = kind.table();
    let label = kind.as_str();
    let name_expr = crate::kind_sql::display_name_expr(kind);
    let joins = crate::kind_sql::name_joins(kind);
    let dev = crate::kind_sql::GPU_DEVICE_ID_EXPR;
    let stm = crate::kind_sql::GPU_STREAM_ID_EXPR;
    Ok(format!(
        r#"
        SELECT
            '{label}' AS kind,
            t.rowid   AS row_id,
            {dev} AS device_id,
            {stm} AS stream_id,
            {name_expr} AS name,
            t.start   AS start_ns,
            t."end"   AS end_ns
        FROM nsight.{table} t {joins}
        "#
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use veloq_core::VeloqDiagnostic;

    fn gap_hydration_sql(duration_expr: &str, prev_kind_expr: &str) -> String {
        format!(
            "SELECT \
             0::INTEGER AS device_id, \
             CAST(NULL AS BIGINT) AS stream_id, \
             10::BIGINT AS gap_start_ns, \
             20::BIGINT AS gap_end_ns, \
             {duration_expr} AS gap_ns, \
             {prev_kind_expr} AS prev_kind, \
             1::BIGINT AS prev_row_id, \
             'prev' AS prev_name, \
             7::BIGINT AS prev_stream_id, \
             'kernel' AS next_kind, \
             2::BIGINT AS next_row_id, \
             'next' AS next_name, \
             8::BIGINT AS next_stream_id"
        )
    }

    fn stream_activity_hydration_sql(busy_expr: &str) -> String {
        format!(
            "SELECT \
             0::INTEGER AS device_id, \
             7::BIGINT AS stream_id, \
             {busy_expr} AS busy_ns"
        )
    }

    fn test_gap(start_ns: i64) -> Gap {
        Gap {
            key: format!("gap|dev:0|@{start_ns}"),
            device_id: Some(0),
            stream_id: None,
            start_ns,
            end_ns: start_ns + 10,
            duration_ns: 10,
            prev: GapNeighbor {
                row_id: RowId::new(EventKind::Kernel, start_ns),
                name: "prev".to_string(),
                timestamp_ns: start_ns,
                stream_id: 7,
            },
            next: GapNeighbor {
                row_id: RowId::new(EventKind::Kernel, start_ns + 1),
                name: "next".to_string(),
                timestamp_ns: start_ns + 10,
                stream_id: 8,
            },
        }
    }

    #[test]
    fn limit_saturation_uses_rows_when_not_truncated() {
        let mut gaps = vec![test_gap(1), test_gap(2)];

        assert_eq!(truncate_gap_rows_to_limit(&mut gaps, 3), Some(2));
        assert_eq!(gaps.len(), 2);
    }

    #[test]
    fn limit_saturation_truncates_and_defers_total_when_truncated() {
        let mut gaps = vec![test_gap(1), test_gap(2), test_gap(3)];

        assert_eq!(truncate_gap_rows_to_limit(&mut gaps, 2), None);
        assert_eq!(gaps.len(), 2);
        assert!(gaps.iter().all(|gap| gap.start_ns <= 2));
    }

    #[test]
    fn unified_window_input_pushes_scope_before_sweep() -> Result<()> {
        let req = GapsRequest {
            min_ns: 42,
            device: Some(2),
            limit: 7,
            ..Default::default()
        };
        let query = build_unified_sql(
            "SELECT * FROM synthetic_gpu_events",
            &req,
            Some((10, 20)),
            true,
        )?;

        assert!(query.rows_sql.contains("scoped_events AS"));
        assert!(query.rows_sql.contains("prefix_events AS"));
        assert!(query.rows_sql.contains("suffix_events AS"));
        assert!(query.rows_sql.contains("local_events AS"));
        assert!(query.rows_sql.contains("FROM local_events"));
        assert!(query.rows_sql.contains("GROUP BY device_id"));
        assert!(!query.rows_sql.contains("ROW_NUMBER()"));
        assert!(!query.rows_sql.contains("COUNT(*) OVER"));
        assert!(query.rows_sql.contains("WHERE start_ns < ? AND end_ns > ?"));
        assert!(query.total_sql.contains("FROM local_events"));
        assert!(query.total_sql.contains("MAX(end_ns) OVER win"));
        assert!(
            query
                .total_sql
                .contains("CAST(COUNT(*) AS BIGINT) AS total_matched")
        );
        assert_eq!(
            query.total_params,
            vec![
                Value::Int(2),
                Value::BigInt(10),
                Value::BigInt(20),
                Value::BigInt(20),
                Value::BigInt(10),
                Value::BigInt(20),
                Value::BigInt(10),
                Value::BigInt(42),
            ]
        );
        assert_eq!(
            query.rows_params,
            vec![
                Value::Int(2),
                Value::BigInt(10),
                Value::BigInt(20),
                Value::BigInt(20),
                Value::BigInt(10),
                Value::BigInt(20),
                Value::BigInt(10),
                Value::BigInt(42),
                Value::BigInt(8),
            ]
        );

        let full_trace_query = build_unified_sql(
            "SELECT * FROM synthetic_gpu_events",
            &GapsRequest {
                min_ns: 42,
                device: Some(2),
                limit: 7,
                ..Default::default()
            },
            None,
            true,
        )?;
        assert!(!full_trace_query.rows_sql.contains("COUNT(*) OVER"));
        assert!(full_trace_query.total_sql.contains("MAX(end_ns) OVER win"));
        assert!(!full_trace_query.total_sql.contains("arg_max("));
        assert_eq!(
            full_trace_query.total_params,
            vec![Value::Int(2), Value::BigInt(42)]
        );
        Ok(())
    }

    #[test]
    fn stream_window_input_pushes_scope_before_lead() -> Result<()> {
        let req = GapsRequest {
            scope: GapScope::Stream,
            min_ns: 42,
            device: Some(2),
            stream: Some(143),
            limit: 7,
            ..Default::default()
        };
        let query = build_stream_sql("SELECT * FROM synthetic_gpu_events", &req, Some((10, 20)))?;

        assert!(query.rows_sql.contains("scoped_events AS"));
        assert!(query.rows_sql.contains("prefix_events AS"));
        assert!(query.rows_sql.contains("suffix_events AS"));
        assert!(query.rows_sql.contains("local_events AS"));
        assert!(query.rows_sql.contains("FROM local_events"));
        assert!(query.rows_sql.contains("PARTITION BY device_id, stream_id"));
        assert!(!query.rows_sql.contains("COUNT(*) OVER"));
        assert!(query.rows_sql.contains("WHERE start_ns < ? AND end_ns > ?"));
        assert!(query.total_sql.contains("FROM local_events"));
        assert!(query.total_sql.contains("LEAD(start_ns) OVER w"));
        assert!(!query.total_sql.contains("LEAD(kind)"));
        assert!(
            query
                .total_sql
                .contains("CAST(COUNT(*) AS BIGINT) AS total_matched")
        );
        assert_eq!(
            query.total_params,
            vec![
                Value::Int(2),
                Value::BigInt(143),
                Value::BigInt(10),
                Value::BigInt(20),
                Value::BigInt(20),
                Value::BigInt(10),
                Value::BigInt(20),
                Value::BigInt(10),
                Value::BigInt(42),
            ]
        );
        assert_eq!(
            query.rows_params,
            vec![
                Value::Int(2),
                Value::BigInt(143),
                Value::BigInt(10),
                Value::BigInt(20),
                Value::BigInt(20),
                Value::BigInt(10),
                Value::BigInt(20),
                Value::BigInt(10),
                Value::BigInt(42),
                Value::BigInt(8),
            ]
        );

        let full_trace_query = build_stream_sql(
            "SELECT * FROM synthetic_gpu_events",
            &GapsRequest {
                scope: GapScope::Stream,
                min_ns: 42,
                device: Some(2),
                stream: Some(143),
                limit: 7,
                ..Default::default()
            },
            None,
        )?;
        assert!(!full_trace_query.rows_sql.contains("COUNT(*) OVER"));
        assert!(full_trace_query.total_sql.contains("LEAD(start_ns) OVER w"));
        assert!(!full_trace_query.total_sql.contains("LEAD(kind)"));
        assert_eq!(
            full_trace_query.total_params,
            vec![Value::Int(2), Value::BigInt(143), Value::BigInt(42)]
        );
        Ok(())
    }

    #[test]
    fn parse_min_duration_invalid_literal_returns_typed_error() -> anyhow::Result<()> {
        let err = match GapsRequest::parse_min_duration("bogus") {
            Ok(ns) => anyhow::bail!("expected invalid min duration to fail, got {ns} ns"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.gaps-min-duration-invalid");
        match err {
            crate::NsysQueryError::GapsMinDurationInvalid { value, .. } => {
                assert_eq!(value, "bogus");
            }
            other => anyhow::bail!("expected GapsMinDurationInvalid, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn parse_kind_unknown_tag_is_typed() -> Result<()> {
        let err = match parse_kind("bogus") {
            Ok(kind) => anyhow::bail!("expected unknown kind tag to fail, got {kind}"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.internal.sql-kind-tag-invalid");
        match err {
            crate::NsysQueryError::InternalSqlKindTagInvalid { verb, kind } => {
                assert_eq!(verb, "gaps");
                assert_eq!(kind, "bogus");
            }
            other => anyhow::bail!("expected InternalSqlKindTagInvalid, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn hydrate_gap_rows_prepare_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;

        let err = match hydrate_gap_rows(&conn, "SELECT * FROM", &[], GapScope::Device) {
            Ok(rows) => anyhow::bail!(
                "malformed gaps SQL should not hydrate successfully: {} rows",
                rows.len()
            ),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-prepare");
        assert!(matches!(
            err,
            crate::NsysQueryError::Sql {
                phase: crate::SqlPhase::Prepare,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn hydrate_gap_rows_query_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = "SELECT \
                   ? AS device_id, \
                   CAST(NULL AS BIGINT) AS stream_id, \
                   10::BIGINT AS gap_start_ns, \
                   20::BIGINT AS gap_end_ns, \
                   10::BIGINT AS gap_ns, \
                   'kernel' AS prev_kind, \
                   1::BIGINT AS prev_row_id, \
                   'prev' AS prev_name, \
                   7::BIGINT AS prev_stream_id, \
                   'kernel' AS next_kind, \
                   2::BIGINT AS next_row_id, \
                   'next' AS next_name, \
                   8::BIGINT AS next_stream_id";

        let err = match hydrate_gap_rows(&conn, sql, &[], GapScope::Device) {
            Ok(rows) => anyhow::bail!(
                "unbound gaps SQL parameter should not hydrate successfully: {} rows",
                rows.len()
            ),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-query");
        assert!(matches!(
            err,
            crate::NsysQueryError::Sql {
                phase: crate::SqlPhase::Query,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn hydrate_gap_rows_read_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = gap_hydration_sql("'not-duration'", "'kernel'");

        let err = match hydrate_gap_rows(&conn, &sql, &[], GapScope::Device) {
            Ok(rows) => anyhow::bail!(
                "malformed gaps row should not hydrate successfully: {} rows",
                rows.len()
            ),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-read");
        assert!(matches!(
            err,
            crate::NsysQueryError::Sql {
                phase: crate::SqlPhase::Read,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn hydrate_gap_rows_kind_tag_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = gap_hydration_sql("10::BIGINT", "'bogus'");

        let err = match hydrate_gap_rows(&conn, &sql, &[], GapScope::Device) {
            Ok(rows) => anyhow::bail!(
                "unknown gaps kind tag should not hydrate successfully: {} rows",
                rows.len()
            ),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.internal.sql-kind-tag-invalid");
        assert!(matches!(
            err,
            crate::NsysQueryError::InternalSqlKindTagInvalid { .. }
        ));
        Ok(())
    }

    #[test]
    fn hydrate_stream_activity_rows_prepare_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;

        let err = match hydrate_stream_activity_rows(&conn, "SELECT * FROM", &[], 100) {
            Ok(rows) => anyhow::bail!(
                "malformed stream-activity SQL should not hydrate successfully: {} rows",
                rows.len()
            ),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-prepare");
        assert!(matches!(
            err,
            crate::NsysQueryError::Sql {
                phase: crate::SqlPhase::Prepare,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn hydrate_stream_activity_rows_query_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = "SELECT \
                   ? AS device_id, \
                   7::BIGINT AS stream_id, \
                   10::BIGINT AS busy_ns";

        let err = match hydrate_stream_activity_rows(&conn, sql, &[], 100) {
            Ok(rows) => anyhow::bail!(
                "unbound stream-activity SQL parameter should not hydrate successfully: {} rows",
                rows.len()
            ),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-query");
        assert!(matches!(
            err,
            crate::NsysQueryError::Sql {
                phase: crate::SqlPhase::Query,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn hydrate_stream_activity_rows_read_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = stream_activity_hydration_sql("'not-busy'");

        let err = match hydrate_stream_activity_rows(&conn, &sql, &[], 100) {
            Ok(rows) => anyhow::bail!(
                "malformed stream-activity row should not hydrate successfully: {} rows",
                rows.len()
            ),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-read");
        assert!(matches!(
            err,
            crate::NsysQueryError::Sql {
                phase: crate::SqlPhase::Read,
                ..
            }
        ));
        Ok(())
    }
}
