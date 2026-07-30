//! `veloq graph-replays <trace>` — CUDA graph replay decomposition.
//!
//! CUDA graph captures appear in two NSys shapes:
//! - `--cuda-graph-trace=graph`: one `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`
//!   row is one replay. It has replay wall time but no node-level
//!   kernel/memcpy/memset decomposition.
//! - `--cuda-graph-trace=node`: graph-captured GPU work lands in the
//!   normal kernel/memcpy/memset tables with `graphNodeId` populated.
//!   Replays are keyed by the process-aware correlation identity
//!   `(native_pid, deviceId, contextId, correlationId)`.
//!
//! Raw `correlationId` is never used alone. Every public row carries
//! the [`veloq_nsys_data::SyntheticId`] display value for the full
//! process-aware identity.

use crate::{NsysQueryError, NsysQueryResult, RowId};
use duckdb::types::Value;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use veloq_core::sort::build_order_by;
use veloq_core::time::TimeWindow;
use veloq_core::{Direction, SortKeyDef, SortKeySpec, SortSpec};
use veloq_nsys_data::{SyntheticId, Trace};
use veloq_query::duckdb::list::{TotalCarrier, infallible_count_error, total_matched};
use veloq_query::sql::{name, total_matched_bigint_expr};

const RESIDENT_GRAPH_TRACE_TABLE: &str = "veloq_resident_graph_trace_rows";
const RESIDENT_GRAPH_NODE_TABLE: &str = "veloq_resident_graph_node_rows";
const RESIDENT_REPLAY_SUMMARY_TABLE: &str = "veloq_resident_graph_replay_summaries";
const RESIDENT_LAUNCHER_TABLE: &str = "veloq_resident_graph_replay_launchers";
const RESIDENT_BUSY_TABLE: &str = "veloq_resident_graph_replay_busy";
const RESIDENT_NODE_AGGREGATE_TABLE: &str = "veloq_resident_graph_replay_node_aggregates";

#[derive(Debug, Clone)]
pub struct GraphReplaysRequest {
    pub time_window: Option<TimeWindow>,
    /// Launch-scoped NVTX glob. Matches enclosing NVTX names around
    /// `cudaGraphLaunch%` runtime rows, then joins launches to replay
    /// work by `(process, device, context, correlationId)`.
    pub nvtx: Option<String>,
    pub process_id: Option<i64>,
    pub device: Option<i32>,
    pub sort: Option<SortSpec>,
    pub limit: usize,
    pub top_nodes_limit: usize,
}

impl Default for GraphReplaysRequest {
    fn default() -> Self {
        Self {
            time_window: None,
            nvtx: None,
            process_id: None,
            device: None,
            sort: None,
            limit: 20,
            top_nodes_limit: 10,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CaptureMode {
    GraphTrace,
    GraphNodes,
    None,
}

impl std::fmt::Display for CaptureMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::GraphTrace => "graph_trace",
            Self::GraphNodes => "graph_nodes",
            Self::None => "none",
        })
    }
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct GraphReplaysResponse {
    pub count: usize,
    pub total_matched: i64,
    pub capture_mode: CaptureMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_window_ns: Option<(i64, i64)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_scope: Option<String>,
    pub top_nodes_limit: usize,
    pub rows: Vec<GraphReplayRow>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct GraphReplayRow {
    /// List key for this replay. Includes the lossless synthetic
    /// correlation identity so processes reusing CUDA-local values
    /// stay distinct.
    pub key: String,
    pub capture_mode: CaptureMode,
    pub synthetic_id: String,
    pub process_id: i64,
    pub device_id: i32,
    pub context_id: i64,
    pub correlation_id: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub launcher_row_id: Option<RowId>,
    pub start_ns: i64,
    pub end_ns: i64,
    pub wall_ns: i64,
    pub sum_gpu_ns: i64,
    pub busy_ns: i64,
    pub idle_inside_replay_ns: i64,
    pub event_count: i64,
    pub kernel_count: i64,
    pub memcpy_count: i64,
    pub memset_count: i64,
    pub graph_trace_count: i64,
    pub stream_count: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_exec_id: Option<i64>,
    pub decomposition_available: bool,
    pub top_nodes: Vec<GraphReplayNode>,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct GraphReplayNode {
    pub graph_node_id: i64,
    pub kind: String,
    pub name: String,
    pub count: i64,
    pub stream_count: i64,
    pub start_ns: i64,
    pub end_ns: i64,
    pub wall_ns: i64,
    pub sum_ns: i64,
    pub max_ns: i64,
    pub sum_share_of_replay_wall: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortKey {
    Wall,
    Sum,
    Start,
    Count,
}

impl SortKeyDef for SortKey {
    fn specs() -> &'static [SortKeySpec<Self>] {
        &[
            SortKeySpec {
                variant: SortKey::Wall,
                canonical: "wall",
                aliases: &["wall_ns", "duration", "duration_ns"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SortKey::Sum,
                canonical: "sum",
                aliases: &["sum_gpu", "sum_gpu_ns", "total"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SortKey::Start,
                canonical: "start",
                aliases: &["start_ns"],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: SortKey::Count,
                canonical: "count",
                aliases: &["events", "event_count"],
                default_dir: Direction::Desc,
            },
        ]
    }
}

#[derive(Debug, Clone)]
struct ReplaySummary {
    process_id: i64,
    device_id: i32,
    context_id: i64,
    correlation_id: i64,
    start_ns: i64,
    end_ns: i64,
    sum_gpu_ns: i64,
    event_count: i64,
    kernel_count: i64,
    memcpy_count: i64,
    memset_count: i64,
    graph_trace_count: i64,
    stream_count: i64,
    graph_id: Option<i64>,
    graph_exec_id: Option<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ReplaySelection {
    process_id: i64,
    device_id: i32,
    context_id: i64,
    correlation_id: i64,
    start_ns: i64,
}

type ReplayDecomposition = HashMap<ReplaySelection, (i64, Vec<GraphReplayNode>)>;

impl ReplaySummary {
    fn selection(&self) -> ReplaySelection {
        ReplaySelection {
            process_id: self.process_id,
            device_id: self.device_id,
            context_id: self.context_id,
            correlation_id: self.correlation_id,
            start_ns: self.start_ns,
        }
    }
}

#[derive(Debug, Clone)]
struct NodeEvent {
    kind: String,
    name: String,
    graph_node_id: i64,
    stream_id: i64,
    start_ns: i64,
    end_ns: i64,
}

pub fn run<P: AsRef<Path>>(
    path: P,
    req: GraphReplaysRequest,
) -> NsysQueryResult<GraphReplaysResponse> {
    let trace = Trace::open(path).map_err(NsysQueryError::trace_open)?;
    run_with_trace(&trace, req)
}

pub fn run_with_trace(
    trace: &Trace,
    req: GraphReplaysRequest,
) -> NsysQueryResult<GraphReplaysResponse> {
    crate::check_limit(req.limit)?;
    if req.top_nodes_limit == 0 {
        return Err(NsysQueryError::GraphReplaysTopNodesTooSmall);
    }

    let abs_window = trace
        .resolve_window(req.time_window)
        .map_err(NsysQueryError::time_window_resolve)?;
    let mode = capture_mode(trace)?;

    let mut rows = match mode {
        CaptureMode::GraphTrace => query_graph_trace(trace, &req, abs_window)?,
        CaptureMode::GraphNodes => query_graph_nodes(trace, &req, abs_window)?,
        CaptureMode::None => Vec::new(),
    };

    let total_matched = total_matched::<i64, _>(&rows, TotalCarrier::First, |(_, total)| *total)
        .map_err(infallible_count_error)?;
    let summaries = rows
        .iter()
        .map(|(summary, _)| summary.clone())
        .collect::<Vec<_>>();
    let launchers = find_launchers(trace, &summaries)?;
    let mut resident_decomposition = match mode {
        CaptureMode::GraphNodes if abs_window.is_none() => {
            load_resident_decomposition(trace, &summaries, req.top_nodes_limit)?
        }
        CaptureMode::GraphTrace | CaptureMode::GraphNodes | CaptureMode::None => None,
    };
    let mut node_events = match (mode, resident_decomposition.is_some()) {
        (CaptureMode::GraphNodes, false) => load_node_events(trace, &summaries)?,
        (CaptureMode::GraphNodes, true) | (CaptureMode::GraphTrace | CaptureMode::None, _) => {
            HashMap::new()
        }
    };
    let mut out_rows = Vec::with_capacity(rows.len());
    for (summary, _) in rows.drain(..) {
        let selection = summary.selection();
        let launcher = launchers.get(&selection).copied();
        let synthetic = SyntheticId::pack(
            summary.process_id as u64,
            summary.device_id as u64,
            summary.context_id as u64,
            summary.correlation_id as u64,
        )
        .to_string();
        // Graph replay busy is replay-scoped, not the generic
        // GPU-busy interval set. `graph_trace` captures expose only
        // replay wall rows, while node-mode captures can union the
        // replay's node events.
        let (busy_ns, top_nodes, decomposition_available) = match mode {
            CaptureMode::GraphTrace => (summary.end_ns - summary.start_ns, Vec::new(), false),
            CaptureMode::GraphNodes => {
                if let Some((busy, nodes)) = resident_decomposition
                    .as_mut()
                    .and_then(|decomposition| decomposition.remove(&selection))
                {
                    (busy, nodes, true)
                } else {
                    let events = node_events.remove(&selection).unwrap_or_default();
                    let busy = busy_ns(events.iter().map(|e| (e.start_ns, e.end_ns)).collect());
                    let nodes = top_nodes(
                        &events,
                        summary.end_ns - summary.start_ns,
                        req.top_nodes_limit,
                    );
                    (busy, nodes, true)
                }
            }
            CaptureMode::None => (0, Vec::new(), false),
        };
        let wall_ns = summary.end_ns - summary.start_ns;
        out_rows.push(GraphReplayRow {
            key: format!("graph-replay|{synthetic}"),
            capture_mode: mode,
            synthetic_id: synthetic,
            process_id: summary.process_id,
            device_id: summary.device_id,
            context_id: summary.context_id,
            correlation_id: summary.correlation_id,
            launcher_row_id: launcher,
            start_ns: summary.start_ns,
            end_ns: summary.end_ns,
            wall_ns,
            sum_gpu_ns: summary.sum_gpu_ns,
            busy_ns,
            idle_inside_replay_ns: (wall_ns - busy_ns).max(0),
            event_count: summary.event_count,
            kernel_count: summary.kernel_count,
            memcpy_count: summary.memcpy_count,
            memset_count: summary.memset_count,
            graph_trace_count: summary.graph_trace_count,
            stream_count: summary.stream_count,
            graph_id: summary.graph_id,
            graph_exec_id: summary.graph_exec_id,
            decomposition_available,
            top_nodes,
        });
    }

    Ok(GraphReplaysResponse {
        count: out_rows.len(),
        total_matched,
        capture_mode: mode,
        time_window_ns: abs_window,
        nvtx_scope: req.nvtx.clone(),
        top_nodes_limit: req.top_nodes_limit,
        rows: out_rows,
    })
}

/// Materialize the normalized graph-replay evidence once in the resident
/// DuckDB connection. The table is private to the daemon session and is
/// covered by DuckDB's existing resident-memory accounting.
///
/// `Ok(false)` means the trace contains no replay evidence. Queries retain the
/// established source path in either case.
pub fn ensure_resident_index(trace: &Trace) -> NsysQueryResult<bool> {
    if resident_table_available(trace, RESIDENT_GRAPH_TRACE_TABLE)
        || resident_table_available(trace, RESIDENT_GRAPH_NODE_TABLE)
    {
        return Ok(true);
    }

    match capture_mode(trace)? {
        CaptureMode::GraphTrace => {
            let source = graph_trace_source_sql(trace);
            let sql = format!(
                "CREATE TEMP TABLE {RESIDENT_GRAPH_TRACE_TABLE} AS \
                 SELECT * FROM ({source}) \
                 ORDER BY process_id, device_id, context_id, correlation_id, start_ns, end_ns"
            );
            trace.conn().execute_batch(&sql).map_err(|source| {
                NsysQueryError::sql_query("graph-replays", "resident graph-trace build", source)
            })?;
            build_resident_summaries(trace, CaptureMode::GraphTrace)?;
            build_resident_launchers(trace)?;
            Ok(true)
        }
        CaptureMode::GraphNodes => {
            let source = source_node_event_subqueries(trace).join(" UNION ALL ");
            let sql = format!(
                "CREATE TEMP TABLE {RESIDENT_GRAPH_NODE_TABLE} AS \
                 SELECT * FROM ({source}) \
                 ORDER BY process_id, device_id, context_id, correlation_id, start_ns, end_ns, rowid"
            );
            trace.conn().execute_batch(&sql).map_err(|source| {
                NsysQueryError::sql_query("graph-replays", "resident graph-node build", source)
            })?;
            build_resident_summaries(trace, CaptureMode::GraphNodes)?;
            build_resident_decomposition(trace)?;
            build_resident_launchers(trace)?;
            Ok(true)
        }
        CaptureMode::None => Ok(false),
    }
}

fn capture_mode(trace: &Trace) -> NsysQueryResult<CaptureMode> {
    if resident_table_available(trace, RESIDENT_GRAPH_TRACE_TABLE) {
        return Ok(CaptureMode::GraphTrace);
    }
    if resident_table_available(trace, RESIDENT_GRAPH_NODE_TABLE) {
        return Ok(CaptureMode::GraphNodes);
    }

    if trace.table_exists("CUPTI_ACTIVITY_KIND_GRAPH_TRACE") {
        let count = count_capture_mode_rows(
            trace.conn(),
            "SELECT COUNT(*) FROM nsight.CUPTI_ACTIVITY_KIND_GRAPH_TRACE \
             WHERE correlationId IS NOT NULL AND start IS NOT NULL AND \"end\" IS NOT NULL",
        )?;
        if count > 0 {
            return Ok(CaptureMode::GraphTrace);
        }
    }

    if !node_event_subqueries(trace).is_empty() {
        let union = node_event_subqueries(trace).join(" UNION ALL ");
        let sql = format!("WITH event_rows AS ({union}) SELECT COUNT(*) FROM event_rows");
        let count = count_capture_mode_rows(trace.conn(), &sql)?;
        if count > 0 {
            return Ok(CaptureMode::GraphNodes);
        }
    }

    Ok(CaptureMode::None)
}

fn count_capture_mode_rows(conn: &duckdb::Connection, sql: &str) -> NsysQueryResult<i64> {
    conn.query_row(sql, [], |r| r.get(0))
        .map_err(|source| crate::NsysQueryError::sql_query("graph-replays", "capture-mode", source))
}

fn query_graph_trace(
    trace: &Trace,
    req: &GraphReplaysRequest,
    abs_window: Option<(i64, i64)>,
) -> NsysQueryResult<Vec<(ReplaySummary, i64)>> {
    let mut params = Vec::new();
    let (scope_cte, scoped_join) = launch_scope_sql(trace, req.nvtx.as_deref(), &mut params)?;
    let mut where_parts = vec![
        "t.correlation_id IS NOT NULL".to_string(),
        "t.start_ns IS NOT NULL".to_string(),
        "t.end_ns IS NOT NULL".to_string(),
    ];
    if let Some((start, end)) = abs_window {
        where_parts.push("t.end_ns > ? AND t.start_ns < ?".to_string());
        params.push(Value::BigInt(start));
        params.push(Value::BigInt(end));
    }
    if let Some(process_id) = req.process_id {
        where_parts.push("t.process_id = ?".to_string());
        params.push(Value::BigInt(process_id));
    }
    if let Some(device) = req.device {
        where_parts.push("t.device_id = ?".to_string());
        params.push(Value::Int(device));
    }
    let where_sql = where_parts.join(" AND ");
    let order_by = order_by_sql(req.sort.as_ref())?;
    params.push(Value::BigInt(req.limit as i64));
    let graph_trace_rows = graph_trace_source_sql(trace);

    let sql = format!(
        r#"
        WITH {scope_cte}
        graph_trace_rows AS ({graph_trace_rows}),
        base AS (
            SELECT
                t.process_id,
                t.device_id,
                t.context_id,
                t.correlation_id,
                t.start_ns,
                t.end_ns,
                CAST(t.end_ns - t.start_ns AS BIGINT) AS wall_ns,
                CAST(t.end_ns - t.start_ns AS BIGINT) AS sum_gpu_ns,
                CAST(1 AS BIGINT) AS event_count,
                CAST(0 AS BIGINT) AS kernel_count,
                CAST(0 AS BIGINT) AS memcpy_count,
                CAST(0 AS BIGINT) AS memset_count,
                CAST(1 AS BIGINT) AS graph_trace_count,
                CAST(1 AS BIGINT) AS stream_count,
                t.graph_id,
                t.graph_exec_id
            FROM graph_trace_rows t
            WHERE {where_sql}
        ),
        scoped AS (
            SELECT b.*
            FROM base b
            {scoped_join}
        )
        SELECT *,
               {total_matched}
        FROM scoped
        ORDER BY {order_by}
        LIMIT ?
        "#,
        total_matched = total_matched_bigint_expr(),
    );

    collect_replay_summaries(trace.conn(), &sql, &params)
}

fn query_graph_nodes(
    trace: &Trace,
    req: &GraphReplaysRequest,
    abs_window: Option<(i64, i64)>,
) -> NsysQueryResult<Vec<(ReplaySummary, i64)>> {
    if abs_window.is_none() && resident_table_available(trace, RESIDENT_REPLAY_SUMMARY_TABLE) {
        return query_resident_graph_node_summaries(trace, req);
    }
    let subqueries = node_event_subqueries(trace);
    if subqueries.is_empty() {
        return Ok(Vec::new());
    }
    let union = subqueries.join(" UNION ALL ");
    let mut params = Vec::new();
    let (scope_cte, scoped_join) = launch_scope_sql(trace, req.nvtx.as_deref(), &mut params)?;
    let mut where_parts = Vec::new();
    append_scope_filters(
        &mut where_parts,
        &mut params,
        abs_window,
        req.process_id,
        req.device,
    );
    let where_sql = if where_parts.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", where_parts.join(" AND "))
    };
    let order_by = order_by_sql(req.sort.as_ref())?;
    params.push(Value::BigInt(req.limit as i64));

    let sql = format!(
        r#"
        WITH {scope_cte}
        event_rows AS ({union}),
        replay_base AS (
            SELECT
                process_id,
                device_id,
                context_id,
                correlation_id,
                MIN(start_ns) AS start_ns,
                MAX(end_ns) AS end_ns,
                CAST(MAX(end_ns) - MIN(start_ns) AS BIGINT) AS wall_ns,
                CAST(SUM(end_ns - start_ns) AS BIGINT) AS sum_gpu_ns,
                CAST(COUNT(*) AS BIGINT) AS event_count,
                CAST(SUM(CASE WHEN kind = 'kernel' THEN 1 ELSE 0 END) AS BIGINT) AS kernel_count,
                CAST(SUM(CASE WHEN kind = 'memcpy' THEN 1 ELSE 0 END) AS BIGINT) AS memcpy_count,
                CAST(SUM(CASE WHEN kind = 'memset' THEN 1 ELSE 0 END) AS BIGINT) AS memset_count,
                CAST(0 AS BIGINT) AS graph_trace_count,
                CAST(COUNT(DISTINCT stream_id) AS BIGINT) AS stream_count,
                CAST(arbitrary(graph_id) AS BIGINT) AS graph_id,
                CAST(NULL AS BIGINT) AS graph_exec_id
            FROM event_rows
            {where_sql}
            GROUP BY process_id, device_id, context_id, correlation_id
        ),
        scoped AS (
            SELECT b.*
            FROM replay_base b
            {scoped_join}
        )
        SELECT *,
               {total_matched}
        FROM scoped
        ORDER BY {order_by}
        LIMIT ?
        "#,
        total_matched = total_matched_bigint_expr(),
    );

    collect_replay_summaries(trace.conn(), &sql, &params)
}

fn query_resident_graph_node_summaries(
    trace: &Trace,
    req: &GraphReplaysRequest,
) -> NsysQueryResult<Vec<(ReplaySummary, i64)>> {
    let mut params = Vec::new();
    let (scope_cte, scoped_join) = launch_scope_sql(trace, req.nvtx.as_deref(), &mut params)?;
    let mut where_parts = Vec::new();
    if let Some(process_id) = req.process_id {
        where_parts.push("b.process_id = ?".to_string());
        params.push(Value::BigInt(process_id));
    }
    if let Some(device) = req.device {
        where_parts.push("b.device_id = ?".to_string());
        params.push(Value::Int(device));
    }
    let where_sql = if where_parts.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", where_parts.join(" AND "))
    };
    let order_by = order_by_sql(req.sort.as_ref())?;
    params.push(Value::BigInt(req.limit as i64));
    let sql = format!(
        r#"
        WITH {scope_cte}
        scoped AS (
            SELECT b.*
            FROM {RESIDENT_REPLAY_SUMMARY_TABLE} b
            {scoped_join}
            {where_sql}
        )
        SELECT *,
               {total_matched}
        FROM scoped
        ORDER BY {order_by}
        LIMIT ?
        "#,
        total_matched = total_matched_bigint_expr(),
    );
    collect_replay_summaries(trace.conn(), &sql, &params)
}

fn collect_replay_summaries(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
) -> NsysQueryResult<Vec<(ReplaySummary, i64)>> {
    crate::query_sql::exec::query_rows(
        conn,
        sql,
        params,
        crate::query_sql::exec::GRAPH_REPLAYS_REPLAY_SUMMARY,
        replay_summary_row,
    )
}

fn replay_summary_row(row: &duckdb::Row<'_>) -> Result<(ReplaySummary, i64), duckdb::Error> {
    Ok((
        ReplaySummary {
            process_id: row.get("process_id")?,
            device_id: row.get("device_id")?,
            context_id: row.get("context_id")?,
            correlation_id: row.get("correlation_id")?,
            start_ns: row.get("start_ns")?,
            end_ns: row.get("end_ns")?,
            sum_gpu_ns: row.get("sum_gpu_ns")?,
            event_count: row.get("event_count")?,
            kernel_count: row.get("kernel_count")?,
            memcpy_count: row.get("memcpy_count")?,
            memset_count: row.get("memset_count")?,
            graph_trace_count: row.get("graph_trace_count")?,
            stream_count: row.get("stream_count")?,
            graph_id: row.get("graph_id")?,
            graph_exec_id: row.get("graph_exec_id")?,
        },
        row.get("total_matched")?,
    ))
}

fn graph_trace_source_sql(trace: &Trace) -> String {
    if resident_table_available(trace, RESIDENT_GRAPH_TRACE_TABLE) {
        return format!(
            "SELECT process_id, device_id, context_id, correlation_id, \
                    start_ns, end_ns, graph_id, graph_exec_id \
             FROM {RESIDENT_GRAPH_TRACE_TABLE}"
        );
    }
    let process = veloq_nsys_data::process_sql_projection(
        trace,
        "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
        "t",
        "proc",
        "t.start",
    );
    format!(
        "SELECT \
            {process_expr} AS process_id, \
            CAST(t.deviceId AS INTEGER) AS device_id, \
            CAST(t.contextId AS BIGINT) AS context_id, \
            CAST(t.correlationId AS BIGINT) AS correlation_id, \
            CAST(t.start AS BIGINT) AS start_ns, \
            CAST(t.\"end\" AS BIGINT) AS end_ns, \
            CAST(t.graphId AS BIGINT) AS graph_id, \
            CAST(t.graphExecId AS BIGINT) AS graph_exec_id \
         FROM nsight.CUPTI_ACTIVITY_KIND_GRAPH_TRACE t \
         {process_join} \
         WHERE t.correlationId IS NOT NULL \
           AND t.start IS NOT NULL \
           AND t.\"end\" IS NOT NULL",
        process_expr = process.expr,
        process_join = process.join,
    )
}

fn node_event_subqueries(trace: &Trace) -> Vec<String> {
    if resident_table_available(trace, RESIDENT_GRAPH_NODE_TABLE) {
        return vec![format!("SELECT * FROM {RESIDENT_GRAPH_NODE_TABLE}")];
    }
    source_node_event_subqueries(trace)
}

fn source_node_event_subqueries(trace: &Trace) -> Vec<String> {
    let mut out = Vec::new();
    if trace.table_exists("CUPTI_ACTIVITY_KIND_KERNEL") {
        let process = veloq_nsys_data::process_sql_projection(
            trace,
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "t",
            "proc",
            "t.start",
        );
        out.push(format!(
            r#"
            SELECT
                'kernel' AS kind,
                CAST(t.rowid AS BIGINT) AS rowid,
                {process_expr} AS process_id,
                CAST(t.deviceId AS INTEGER) AS device_id,
                CAST(t.contextId AS BIGINT) AS context_id,
                CAST(t.streamId AS BIGINT) AS stream_id,
                CAST(t.correlationId AS BIGINT) AS correlation_id,
                CAST(t.start AS BIGINT) AS start_ns,
                CAST(t."end" AS BIGINT) AS end_ns,
                CAST(t.graphId AS BIGINT) AS graph_id,
                CAST(t.graphNodeId AS BIGINT) AS graph_node_id,
                COALESCE(s.value, CONCAT('kernel:', CAST(t.shortName AS VARCHAR)), '<unnamed>') AS name
            FROM nsight.CUPTI_ACTIVITY_KIND_KERNEL t
            LEFT JOIN nsight.StringIds s ON t.shortName = s.id
            {process_join}
            WHERE t.graphNodeId IS NOT NULL
              AND t.correlationId IS NOT NULL
              AND t.deviceId IS NOT NULL
              AND t.contextId IS NOT NULL
              AND t.start IS NOT NULL
              AND t."end" IS NOT NULL
            "#,
            process_expr = process.expr,
            process_join = process.join,
        ));
    }
    if trace.table_exists("CUPTI_ACTIVITY_KIND_MEMCPY") {
        let process = veloq_nsys_data::process_sql_projection(
            trace,
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "t",
            "proc",
            "t.start",
        );
        out.push(format!(
            r#"
            SELECT
                'memcpy' AS kind,
                CAST(t.rowid AS BIGINT) AS rowid,
                {process_expr} AS process_id,
                CAST(t.deviceId AS INTEGER) AS device_id,
                CAST(t.contextId AS BIGINT) AS context_id,
                CAST(t.streamId AS BIGINT) AS stream_id,
                CAST(t.correlationId AS BIGINT) AS correlation_id,
                CAST(t.start AS BIGINT) AS start_ns,
                CAST(t."end" AS BIGINT) AS end_ns,
                CAST(NULL AS BIGINT) AS graph_id,
                CAST(t.graphNodeId AS BIGINT) AS graph_node_id,
                CONCAT('memcpy:', CAST(COALESCE(t.copyKind, -1) AS VARCHAR)) AS name
            FROM nsight.CUPTI_ACTIVITY_KIND_MEMCPY t
            {process_join}
            WHERE t.graphNodeId IS NOT NULL
              AND t.correlationId IS NOT NULL
              AND t.deviceId IS NOT NULL
              AND t.contextId IS NOT NULL
              AND t.start IS NOT NULL
              AND t."end" IS NOT NULL
            "#,
            process_expr = process.expr,
            process_join = process.join,
        ));
    }
    if trace.table_exists("CUPTI_ACTIVITY_KIND_MEMSET") {
        let process = veloq_nsys_data::process_sql_projection(
            trace,
            "CUPTI_ACTIVITY_KIND_MEMSET",
            "t",
            "proc",
            "t.start",
        );
        out.push(format!(
            r#"
            SELECT
                'memset' AS kind,
                CAST(t.rowid AS BIGINT) AS rowid,
                {process_expr} AS process_id,
                CAST(t.deviceId AS INTEGER) AS device_id,
                CAST(t.contextId AS BIGINT) AS context_id,
                CAST(t.streamId AS BIGINT) AS stream_id,
                CAST(t.correlationId AS BIGINT) AS correlation_id,
                CAST(t.start AS BIGINT) AS start_ns,
                CAST(t."end" AS BIGINT) AS end_ns,
                CAST(NULL AS BIGINT) AS graph_id,
                CAST(t.graphNodeId AS BIGINT) AS graph_node_id,
                'memset' AS name
            FROM nsight.CUPTI_ACTIVITY_KIND_MEMSET t
            {process_join}
            WHERE t.graphNodeId IS NOT NULL
              AND t.correlationId IS NOT NULL
              AND t.deviceId IS NOT NULL
              AND t.contextId IS NOT NULL
              AND t.start IS NOT NULL
              AND t."end" IS NOT NULL
            "#,
            process_expr = process.expr,
            process_join = process.join,
        ));
    }
    out
}

fn resident_table_available(trace: &Trace, table: &str) -> bool {
    trace
        .conn()
        .query_row(
            "SELECT EXISTS(\
                SELECT 1 FROM duckdb_tables() \
                WHERE table_name = ? AND temporary\
            )",
            [table],
            |row| row.get::<_, bool>(0),
        )
        .unwrap_or(false)
}

fn append_scope_filters(
    where_parts: &mut Vec<String>,
    params: &mut Vec<Value>,
    abs_window: Option<(i64, i64)>,
    process_id: Option<i64>,
    device: Option<i32>,
) {
    if let Some((start, end)) = abs_window {
        where_parts.push("end_ns > ? AND start_ns < ?".to_string());
        params.push(Value::BigInt(start));
        params.push(Value::BigInt(end));
    }
    crate::kind_policy::LocationFilter {
        process_id,
        device,
        stream: None,
    }
    .push_where(&mut *where_parts, &mut *params);
}

fn launch_scope_sql(
    trace: &Trace,
    nvtx: Option<&str>,
    params: &mut Vec<Value>,
) -> NsysQueryResult<(String, String)> {
    let Some(pattern) = nvtx else {
        return Ok((String::new(), String::new()));
    };
    for table in [
        "NVTX_EVENTS",
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        "TARGET_INFO_CUDA_CONTEXT_INFO",
    ] {
        if !trace.table_exists(table) {
            return Err(NsysQueryError::GraphReplaysNvtxPrereqTableMissing { table });
        }
    }
    let name_match = name::glob_like("nvtx_name", pattern);
    let path_match = name::glob_like("p.nvtx_path", pattern);
    params.extend(name_match.params);
    params.extend(path_match.params);
    let cte = format!(
        r#"
        matched_launches AS MATERIALIZED (
            SELECT DISTINCT
                CAST(((r.globalTid >> 24) & 16777215) AS BIGINT) AS process_id,
                CAST(c.deviceId AS INTEGER) AS device_id,
                CAST(c.contextId AS BIGINT) AS context_id,
                CAST(r.correlationId AS BIGINT) AS correlation_id
            FROM nsight.CUPTI_ACTIVITY_KIND_RUNTIME r
            LEFT JOIN nsight.StringIds rs ON r.nameId = rs.id
            JOIN nsight.TARGET_INFO_CUDA_CONTEXT_INFO c
              ON CAST(c.processId AS BIGINT) = CAST(((r.globalTid >> 24) & 16777215) AS BIGINT)
            WHERE r.correlationId IS NOT NULL
              AND COALESCE(rs.value, '') LIKE 'cudaGraphLaunch%'
              AND EXISTS (
                  SELECT 1
                  FROM (
                      SELECT
                          MAX(CASE WHEN {name_match_sql} THEN 1 ELSE 0 END) AS name_match,
                          string_agg(nvtx_name, '/' ORDER BY nvtx_start ASC, nvtx_end DESC, nvtx_rowid ASC) AS nvtx_path
                      FROM (
                          SELECT
                              n.rowid AS nvtx_rowid,
                              n.start AS nvtx_start,
                              COALESCE(n."end", n.start) AS nvtx_end,
                              COALESCE(n.text, ns.value, '<unnamed>') AS nvtx_name
                          FROM nsight.NVTX_EVENTS n
                          LEFT JOIN nsight.StringIds ns ON n.textId = ns.id
                          WHERE n.globalTid = r.globalTid
                            AND n.start <= r.start
                            AND COALESCE(n."end", n.start) >= r."end"
                      ) enclosing
                  ) p
                  WHERE p.name_match = 1
                     OR {path_match_sql}
              )
        ),
        "#,
        name_match_sql = name_match.sql,
        path_match_sql = path_match.sql,
    );
    Ok((
        cte,
        "JOIN matched_launches ml USING (process_id, device_id, context_id, correlation_id)"
            .to_string(),
    ))
}

fn order_by_sql(sort: Option<&SortSpec>) -> NsysQueryResult<String> {
    let default_sort = SortSpec::single("wall");
    let sort = sort.unwrap_or(&default_sort);
    let mut parts = Vec::new();
    for field in sort.fields() {
        let (key, dir) =
            SortKey::from_field(field).map_err(NsysQueryError::graph_replays_sort_invalid)?;
        let col = match key {
            SortKey::Wall => "wall_ns",
            SortKey::Sum => "sum_gpu_ns",
            SortKey::Start => "start_ns",
            SortKey::Count => "event_count",
        };
        parts.push((col, dir));
    }
    Ok(build_order_by(
        &parts,
        "process_id ASC, device_id ASC, context_id ASC, correlation_id",
    ))
}

fn selected_replays_cte(replays: &[ReplaySummary]) -> (String, Vec<Value>) {
    let values = std::iter::repeat_n("(?, ?, ?, ?, ?)", replays.len())
        .collect::<Vec<_>>()
        .join(", ");
    let mut params = Vec::with_capacity(replays.len() * 5);
    for replay in replays {
        params.extend([
            Value::BigInt(replay.process_id),
            Value::Int(replay.device_id),
            Value::BigInt(replay.context_id),
            Value::BigInt(replay.correlation_id),
            Value::BigInt(replay.start_ns),
        ]);
    }
    (
        format!(
            "selected_replays(\
                process_id, device_id, context_id, correlation_id, replay_start_ns\
             ) AS (VALUES {values})"
        ),
        params,
    )
}

fn find_launchers(
    trace: &Trace,
    replays: &[ReplaySummary],
) -> NsysQueryResult<HashMap<ReplaySelection, RowId>> {
    if replays.is_empty()
        || !trace.table_exists("CUPTI_ACTIVITY_KIND_RUNTIME")
        || !trace.table_exists("TARGET_INFO_CUDA_CONTEXT_INFO")
    {
        return Ok(HashMap::new());
    }
    let (selected_replays, params) = selected_replays_cte(replays);
    if resident_table_available(trace, RESIDENT_LAUNCHER_TABLE) {
        let sql = format!(
            "WITH {selected_replays} \
             SELECT \
                q.process_id, q.device_id, q.context_id, q.correlation_id, \
                q.replay_start_ns, l.launcher_rowid \
             FROM selected_replays q \
             JOIN {RESIDENT_LAUNCHER_TABLE} l \
               ON l.process_id = q.process_id \
              AND l.device_id = q.device_id \
              AND l.context_id = q.context_id \
              AND l.correlation_id = q.correlation_id \
              AND l.replay_start_ns = q.replay_start_ns"
        );
        return hydrate_launchers(trace.conn(), &sql, &params);
    }
    let sql = format!(
        r#"
        WITH {selected_replays},
        ranked AS (
            SELECT
                q.process_id,
                q.device_id,
                q.context_id,
                q.correlation_id,
                q.replay_start_ns,
                CAST(r.rowid AS BIGINT) AS launcher_rowid,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        q.process_id,
                        q.device_id,
                        q.context_id,
                        q.correlation_id,
                        q.replay_start_ns
                    ORDER BY
                        CASE WHEN COALESCE(s.value, '') LIKE 'cudaGraphLaunch%'
                             THEN 0 ELSE 1 END ASC,
                        CASE WHEN r.start <= q.replay_start_ns THEN 0 ELSE 1 END ASC,
                        ABS(r.start - q.replay_start_ns) ASC,
                        r.rowid ASC
                ) AS candidate_rank
            FROM selected_replays q
            JOIN nsight.TARGET_INFO_CUDA_CONTEXT_INFO c
              ON CAST(c.processId AS BIGINT) = q.process_id
             AND CAST(c.deviceId AS INTEGER) = q.device_id
             AND CAST(c.contextId AS BIGINT) = q.context_id
            JOIN nsight.CUPTI_ACTIVITY_KIND_RUNTIME r
              ON CAST(((r.globalTid >> 24) & 16777215) AS BIGINT) = q.process_id
             AND CAST(r.correlationId AS BIGINT) = q.correlation_id
            LEFT JOIN nsight.StringIds s ON r.nameId = s.id
        )
        SELECT
            process_id,
            device_id,
            context_id,
            correlation_id,
            replay_start_ns,
            launcher_rowid
        FROM ranked
        WHERE candidate_rank = 1
        "#,
    );
    hydrate_launchers(trace.conn(), &sql, &params)
}

fn build_resident_summaries(trace: &Trace, mode: CaptureMode) -> NsysQueryResult<()> {
    let select = match mode {
        CaptureMode::GraphTrace => format!(
            "SELECT \
                process_id, device_id, context_id, correlation_id, \
                start_ns, end_ns, \
                CAST(end_ns - start_ns AS BIGINT) AS wall_ns, \
                CAST(end_ns - start_ns AS BIGINT) AS sum_gpu_ns, \
                CAST(1 AS BIGINT) AS event_count, \
                CAST(0 AS BIGINT) AS kernel_count, \
                CAST(0 AS BIGINT) AS memcpy_count, \
                CAST(0 AS BIGINT) AS memset_count, \
                CAST(1 AS BIGINT) AS graph_trace_count, \
                CAST(1 AS BIGINT) AS stream_count, \
                graph_id, graph_exec_id \
             FROM {RESIDENT_GRAPH_TRACE_TABLE}"
        ),
        CaptureMode::GraphNodes => format!(
            "SELECT \
                process_id, device_id, context_id, correlation_id, \
                MIN(start_ns) AS start_ns, MAX(end_ns) AS end_ns, \
                CAST(MAX(end_ns) - MIN(start_ns) AS BIGINT) AS wall_ns, \
                CAST(SUM(end_ns - start_ns) AS BIGINT) AS sum_gpu_ns, \
                CAST(COUNT(*) AS BIGINT) AS event_count, \
                CAST(SUM(CASE WHEN kind = 'kernel' THEN 1 ELSE 0 END) AS BIGINT) AS kernel_count, \
                CAST(SUM(CASE WHEN kind = 'memcpy' THEN 1 ELSE 0 END) AS BIGINT) AS memcpy_count, \
                CAST(SUM(CASE WHEN kind = 'memset' THEN 1 ELSE 0 END) AS BIGINT) AS memset_count, \
                CAST(0 AS BIGINT) AS graph_trace_count, \
                CAST(COUNT(DISTINCT stream_id) AS BIGINT) AS stream_count, \
                CAST(arbitrary(graph_id) AS BIGINT) AS graph_id, \
                CAST(NULL AS BIGINT) AS graph_exec_id \
             FROM {RESIDENT_GRAPH_NODE_TABLE} \
             GROUP BY process_id, device_id, context_id, correlation_id"
        ),
        CaptureMode::None => return Ok(()),
    };
    let sql = format!(
        "CREATE TEMP TABLE {RESIDENT_REPLAY_SUMMARY_TABLE} AS \
         SELECT * FROM ({select}) \
         ORDER BY process_id, device_id, context_id, correlation_id, start_ns"
    );
    trace.conn().execute_batch(&sql).map_err(|source| {
        NsysQueryError::sql_query("graph-replays", "resident replay-summary build", source)
    })
}

fn build_resident_decomposition(trace: &Trace) -> NsysQueryResult<()> {
    let node_aggregate_sql = format!(
        r#"
        CREATE TEMP TABLE {RESIDENT_NODE_AGGREGATE_TABLE} AS
        SELECT
            e.process_id,
            e.device_id,
            e.context_id,
            e.correlation_id,
            s.start_ns AS replay_start_ns,
            s.wall_ns AS replay_wall_ns,
            e.graph_node_id,
            e.kind,
            e.name,
            CAST(COUNT(*) AS BIGINT) AS count,
            CAST(COUNT(DISTINCT e.stream_id) AS BIGINT) AS stream_count,
            CAST(MIN(e.start_ns) AS BIGINT) AS start_ns,
            CAST(MAX(e.end_ns) AS BIGINT) AS end_ns,
            CAST(SUM(e.end_ns - e.start_ns) AS BIGINT) AS sum_ns,
            CAST(MAX(e.end_ns - e.start_ns) AS BIGINT) AS max_ns
        FROM {RESIDENT_GRAPH_NODE_TABLE} e
        JOIN {RESIDENT_REPLAY_SUMMARY_TABLE} s
          USING (process_id, device_id, context_id, correlation_id)
        GROUP BY
            e.process_id,
            e.device_id,
            e.context_id,
            e.correlation_id,
            s.start_ns,
            s.wall_ns,
            e.graph_node_id,
            e.kind,
            e.name
        ORDER BY
            e.process_id,
            e.device_id,
            e.context_id,
            e.correlation_id,
            s.start_ns,
            sum_ns DESC,
            max_ns DESC,
            start_ns,
            e.graph_node_id
        "#
    );
    trace
        .conn()
        .execute_batch(&node_aggregate_sql)
        .map_err(|source| {
            NsysQueryError::sql_query("graph-replays", "resident node-aggregate build", source)
        })?;

    let busy_sql = format!(
        r#"
        CREATE TEMP TABLE {RESIDENT_BUSY_TABLE} AS
        WITH ordered AS (
            SELECT
                *,
                MAX(end_ns) OVER (
                    PARTITION BY process_id, device_id, context_id, correlation_id
                    ORDER BY start_ns, end_ns, rowid
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS prior_max_end
            FROM {RESIDENT_GRAPH_NODE_TABLE}
            WHERE end_ns > start_ns
        ),
        marked AS (
            SELECT
                *,
                CASE WHEN prior_max_end IS NULL OR start_ns > prior_max_end
                     THEN 1 ELSE 0 END AS island_start
            FROM ordered
        ),
        islanded AS (
            SELECT
                *,
                SUM(island_start) OVER (
                    PARTITION BY process_id, device_id, context_id, correlation_id
                    ORDER BY start_ns, end_ns, rowid
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS island_id
            FROM marked
        ),
        merged AS (
            SELECT
                process_id,
                device_id,
                context_id,
                correlation_id,
                island_id,
                MIN(start_ns) AS start_ns,
                MAX(end_ns) AS end_ns
            FROM islanded
            GROUP BY process_id, device_id, context_id, correlation_id, island_id
        )
        SELECT
            m.process_id,
            m.device_id,
            m.context_id,
            m.correlation_id,
            s.start_ns AS replay_start_ns,
            CAST(SUM(m.end_ns - m.start_ns) AS BIGINT) AS busy_ns
        FROM merged m
        JOIN {RESIDENT_REPLAY_SUMMARY_TABLE} s
          USING (process_id, device_id, context_id, correlation_id)
        GROUP BY
            m.process_id,
            m.device_id,
            m.context_id,
            m.correlation_id,
            s.start_ns
        ORDER BY
            m.process_id,
            m.device_id,
            m.context_id,
            m.correlation_id,
            s.start_ns
        "#
    );
    trace
        .conn()
        .execute_batch(&busy_sql)
        .map_err(|source| NsysQueryError::sql_query("graph-replays", "resident busy build", source))
}

fn build_resident_launchers(trace: &Trace) -> NsysQueryResult<()> {
    if !trace.table_exists("CUPTI_ACTIVITY_KIND_RUNTIME")
        || !trace.table_exists("TARGET_INFO_CUDA_CONTEXT_INFO")
    {
        return Ok(());
    }
    let sql = format!(
        r#"
        CREATE TEMP TABLE {RESIDENT_LAUNCHER_TABLE} AS
        WITH ranked AS (
            SELECT
                q.process_id,
                q.device_id,
                q.context_id,
                q.correlation_id,
                q.start_ns AS replay_start_ns,
                CAST(r.rowid AS BIGINT) AS launcher_rowid,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        q.process_id,
                        q.device_id,
                        q.context_id,
                        q.correlation_id,
                        q.start_ns
                    ORDER BY
                        CASE WHEN COALESCE(s.value, '') LIKE 'cudaGraphLaunch%'
                             THEN 0 ELSE 1 END ASC,
                        CASE WHEN r.start <= q.start_ns THEN 0 ELSE 1 END ASC,
                        ABS(r.start - q.start_ns) ASC,
                        r.rowid ASC
                ) AS candidate_rank
            FROM {RESIDENT_REPLAY_SUMMARY_TABLE} q
            JOIN nsight.TARGET_INFO_CUDA_CONTEXT_INFO c
              ON CAST(c.processId AS BIGINT) = q.process_id
             AND CAST(c.deviceId AS INTEGER) = q.device_id
             AND CAST(c.contextId AS BIGINT) = q.context_id
            JOIN nsight.CUPTI_ACTIVITY_KIND_RUNTIME r
              ON CAST(((r.globalTid >> 24) & 16777215) AS BIGINT) = q.process_id
             AND CAST(r.correlationId AS BIGINT) = q.correlation_id
            LEFT JOIN nsight.StringIds s ON r.nameId = s.id
        )
        SELECT
            process_id,
            device_id,
            context_id,
            correlation_id,
            replay_start_ns,
            launcher_rowid
        FROM ranked
        WHERE candidate_rank = 1
        ORDER BY process_id, device_id, context_id, correlation_id, replay_start_ns
        "#
    );
    trace.conn().execute_batch(&sql).map_err(|source| {
        NsysQueryError::sql_query("graph-replays", "resident launcher build", source)
    })
}

fn hydrate_launchers(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
) -> NsysQueryResult<HashMap<ReplaySelection, RowId>> {
    let rows = crate::query_sql::exec::query_rows(
        conn,
        sql,
        params,
        crate::query_sql::exec::GRAPH_REPLAYS_LAUNCHER_LOOKUP,
        |row| {
            Ok((
                replay_selection_row(row)?,
                RowId::new(crate::EventKind::Runtime, row.get("launcher_rowid")?),
            ))
        },
    )?;
    Ok(rows.into_iter().collect())
}

fn load_node_events(
    trace: &Trace,
    replays: &[ReplaySummary],
) -> NsysQueryResult<HashMap<ReplaySelection, Vec<NodeEvent>>> {
    if replays.is_empty() {
        return Ok(HashMap::new());
    }
    let subqueries = node_event_subqueries(trace);
    if subqueries.is_empty() {
        return Ok(HashMap::new());
    }
    let union = subqueries.join(" UNION ALL ");
    let (selected_replays, params) = selected_replays_cte(replays);
    let sql = format!(
        r#"
        WITH {selected_replays},
        event_rows AS ({union})
        SELECT
            q.process_id,
            q.device_id,
            q.context_id,
            q.correlation_id,
            q.replay_start_ns,
            e.kind,
            e.name,
            e.graph_node_id,
            e.stream_id,
            e.start_ns,
            e.end_ns
        FROM selected_replays q
        JOIN event_rows e
          ON e.process_id = q.process_id
         AND e.device_id = q.device_id
         AND e.context_id = q.context_id
         AND e.correlation_id = q.correlation_id
        ORDER BY
            q.process_id,
            q.device_id,
            q.context_id,
            q.correlation_id,
            q.replay_start_ns,
            e.start_ns,
            e.end_ns,
            e.rowid
        "#
    );
    hydrate_node_events(trace.conn(), &sql, &params)
}

fn load_resident_decomposition(
    trace: &Trace,
    replays: &[ReplaySummary],
    top_nodes_limit: usize,
) -> NsysQueryResult<Option<ReplayDecomposition>> {
    if replays.is_empty()
        || !resident_table_available(trace, RESIDENT_BUSY_TABLE)
        || !resident_table_available(trace, RESIDENT_NODE_AGGREGATE_TABLE)
    {
        return Ok(None);
    }
    let (selected_replays, mut params) = selected_replays_cte(replays);
    let busy_sql = format!(
        "WITH {selected_replays} \
         SELECT \
            q.process_id, q.device_id, q.context_id, q.correlation_id, \
            q.replay_start_ns, b.busy_ns \
         FROM selected_replays q \
         JOIN {RESIDENT_BUSY_TABLE} b \
           ON b.process_id = q.process_id \
          AND b.device_id = q.device_id \
          AND b.context_id = q.context_id \
          AND b.correlation_id = q.correlation_id \
          AND b.replay_start_ns = q.replay_start_ns"
    );
    let busy_rows = crate::query_sql::exec::query_rows(
        trace.conn(),
        &busy_sql,
        &params,
        crate::query_sql::exec::GRAPH_REPLAYS_NODE_EVENT,
        |row| Ok((replay_selection_row(row)?, row.get::<_, i64>("busy_ns")?)),
    )?;
    let mut decomposition = replays
        .iter()
        .map(|replay| (replay.selection(), (0, Vec::new())))
        .collect::<HashMap<_, _>>();
    for (selection, busy) in busy_rows {
        if let Some((resident_busy, _)) = decomposition.get_mut(&selection) {
            *resident_busy = busy;
        }
    }

    params.push(Value::BigInt(top_nodes_limit as i64));
    let node_sql = format!(
        r#"
        WITH {selected_replays},
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        process_id,
                        device_id,
                        context_id,
                        correlation_id,
                        replay_start_ns
                    ORDER BY
                        sum_ns DESC,
                        max_ns DESC,
                        start_ns ASC,
                        graph_node_id ASC
                ) AS node_rank
            FROM {RESIDENT_NODE_AGGREGATE_TABLE}
        )
        SELECT
            q.process_id,
            q.device_id,
            q.context_id,
            q.correlation_id,
            q.replay_start_ns,
            n.graph_node_id,
            n.kind,
            n.name,
            n.count,
            n.stream_count,
            n.start_ns,
            n.end_ns,
            n.sum_ns,
            n.max_ns,
            n.replay_wall_ns
        FROM selected_replays q
        JOIN ranked n
          ON n.process_id = q.process_id
         AND n.device_id = q.device_id
         AND n.context_id = q.context_id
         AND n.correlation_id = q.correlation_id
         AND n.replay_start_ns = q.replay_start_ns
        WHERE n.node_rank <= ?
        ORDER BY
            q.process_id,
            q.device_id,
            q.context_id,
            q.correlation_id,
            q.replay_start_ns,
            n.node_rank
        "#
    );
    let node_rows = crate::query_sql::exec::query_rows(
        trace.conn(),
        &node_sql,
        &params,
        crate::query_sql::exec::GRAPH_REPLAYS_NODE_EVENT,
        |row| {
            let replay_wall_ns = row.get::<_, i64>("replay_wall_ns")?;
            let start_ns = row.get::<_, i64>("start_ns")?;
            let end_ns = row.get::<_, i64>("end_ns")?;
            let sum_ns = row.get::<_, i64>("sum_ns")?;
            Ok((
                replay_selection_row(row)?,
                GraphReplayNode {
                    graph_node_id: row.get("graph_node_id")?,
                    kind: row.get("kind")?,
                    name: row.get("name")?,
                    count: row.get("count")?,
                    stream_count: row.get("stream_count")?,
                    start_ns,
                    end_ns,
                    wall_ns: end_ns - start_ns,
                    sum_ns,
                    max_ns: row.get("max_ns")?,
                    sum_share_of_replay_wall: if replay_wall_ns > 0 {
                        sum_ns as f64 / replay_wall_ns as f64
                    } else {
                        0.0
                    },
                },
            ))
        },
    )?;
    for (selection, node) in node_rows {
        decomposition
            .entry(selection)
            .or_insert_with(|| (0, Vec::new()))
            .1
            .push(node);
    }
    Ok(Some(decomposition))
}

fn hydrate_node_events(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
) -> NsysQueryResult<HashMap<ReplaySelection, Vec<NodeEvent>>> {
    let rows = crate::query_sql::exec::query_rows(
        conn,
        sql,
        params,
        crate::query_sql::exec::GRAPH_REPLAYS_NODE_EVENT,
        |row| Ok((replay_selection_row(row)?, node_event_row(row)?)),
    )?;
    let mut by_replay = HashMap::new();
    for (replay, event) in rows {
        by_replay.entry(replay).or_insert_with(Vec::new).push(event);
    }
    Ok(by_replay)
}

fn replay_selection_row(row: &duckdb::Row<'_>) -> Result<ReplaySelection, duckdb::Error> {
    Ok(ReplaySelection {
        process_id: row.get("process_id")?,
        device_id: row.get("device_id")?,
        context_id: row.get("context_id")?,
        correlation_id: row.get("correlation_id")?,
        start_ns: row.get("replay_start_ns")?,
    })
}

fn node_event_row(row: &duckdb::Row<'_>) -> Result<NodeEvent, duckdb::Error> {
    Ok(NodeEvent {
        kind: row.get("kind")?,
        name: row.get("name")?,
        graph_node_id: row.get("graph_node_id")?,
        stream_id: row.get("stream_id")?,
        start_ns: row.get("start_ns")?,
        end_ns: row.get("end_ns")?,
    })
}

fn busy_ns(mut intervals: Vec<(i64, i64)>) -> i64 {
    intervals.retain(|(s, e)| e > s);
    intervals.sort_unstable_by_key(|(s, e)| (*s, *e));
    let mut total = 0;
    let mut current: Option<(i64, i64)> = None;
    for (start, end) in intervals {
        match current {
            None => current = Some((start, end)),
            Some((cur_start, cur_end)) if start <= cur_end => {
                current = Some((cur_start, cur_end.max(end)));
            }
            Some((cur_start, cur_end)) => {
                total += cur_end - cur_start;
                current = Some((start, end));
            }
        }
    }
    if let Some((start, end)) = current {
        total += end - start;
    }
    total
}

fn top_nodes(events: &[NodeEvent], replay_wall_ns: i64, limit: usize) -> Vec<GraphReplayNode> {
    #[derive(Default)]
    struct Acc {
        count: i64,
        streams: HashSet<i64>,
        start_ns: i64,
        end_ns: i64,
        sum_ns: i64,
        max_ns: i64,
    }

    let mut map: HashMap<(i64, String, String), Acc> = HashMap::new();
    for e in events {
        let dur = e.end_ns - e.start_ns;
        let acc = map
            .entry((e.graph_node_id, e.kind.clone(), e.name.clone()))
            .or_insert_with(|| Acc {
                start_ns: e.start_ns,
                end_ns: e.end_ns,
                ..Default::default()
            });
        acc.count += 1;
        acc.streams.insert(e.stream_id);
        acc.start_ns = acc.start_ns.min(e.start_ns);
        acc.end_ns = acc.end_ns.max(e.end_ns);
        acc.sum_ns += dur;
        acc.max_ns = acc.max_ns.max(dur);
    }

    let mut nodes: Vec<GraphReplayNode> = map
        .into_iter()
        .map(|((graph_node_id, kind, name), acc)| GraphReplayNode {
            graph_node_id,
            kind,
            name,
            count: acc.count,
            stream_count: acc.streams.len() as i64,
            start_ns: acc.start_ns,
            end_ns: acc.end_ns,
            wall_ns: acc.end_ns - acc.start_ns,
            sum_ns: acc.sum_ns,
            max_ns: acc.max_ns,
            sum_share_of_replay_wall: if replay_wall_ns > 0 {
                acc.sum_ns as f64 / replay_wall_ns as f64
            } else {
                0.0
            },
        })
        .collect();
    nodes.sort_by(|a, b| {
        b.sum_ns
            .cmp(&a.sum_ns)
            .then_with(|| b.max_ns.cmp(&a.max_ns))
            .then_with(|| a.start_ns.cmp(&b.start_ns))
            .then_with(|| a.graph_node_id.cmp(&b.graph_node_id))
    });
    nodes.truncate(limit);
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use veloq_core::VeloqDiagnostic;

    fn replay_summary_hydration_sql(sum_expr: &str) -> String {
        format!(
            "SELECT \
             12345::BIGINT AS process_id, \
             0::INTEGER AS device_id, \
             1::BIGINT AS context_id, \
             2::BIGINT AS correlation_id, \
             10::BIGINT AS start_ns, \
             20::BIGINT AS end_ns, \
             {sum_expr} AS sum_gpu_ns, \
             1::BIGINT AS event_count, \
             1::BIGINT AS kernel_count, \
             0::BIGINT AS memcpy_count, \
             0::BIGINT AS memset_count, \
             0::BIGINT AS graph_trace_count, \
             1::BIGINT AS stream_count, \
             CAST(NULL AS BIGINT) AS graph_id, \
             CAST(NULL AS BIGINT) AS graph_exec_id, \
             1::BIGINT AS total_matched"
        )
    }

    fn node_event_hydration_sql(graph_node_expr: &str) -> String {
        format!(
            "SELECT \
             12345::BIGINT AS process_id, \
             0::INTEGER AS device_id, \
             1::BIGINT AS context_id, \
             2::BIGINT AS correlation_id, \
             10::BIGINT AS replay_start_ns, \
             'kernel' AS kind, \
             'node' AS name, \
             {graph_node_expr} AS graph_node_id, \
             7::BIGINT AS stream_id, \
             10::BIGINT AS start_ns, \
             20::BIGINT AS end_ns"
        )
    }

    fn launcher_hydration_sql(rowid_expr: &str) -> String {
        format!(
            "SELECT \
             12345::BIGINT AS process_id, \
             0::INTEGER AS device_id, \
             1::BIGINT AS context_id, \
             2::BIGINT AS correlation_id, \
             10::BIGINT AS replay_start_ns, \
             {rowid_expr} AS launcher_rowid"
        )
    }

    #[test]
    fn busy_ns_unions_overlaps() {
        assert_eq!(busy_ns(vec![(0, 10), (5, 15), (20, 25)]), 20);
    }

    #[test]
    fn capture_mode_count_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;

        let err = match count_capture_mode_rows(&conn, "SELECT * FROM") {
            Ok(count) => anyhow::bail!("malformed capture-mode SQL should fail, got {count}"),
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
    fn collect_replay_summaries_prepare_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;

        let err = match collect_replay_summaries(&conn, "SELECT * FROM", &[]) {
            Ok(rows) => anyhow::bail!(
                "malformed replay-summary SQL should not hydrate successfully: {} rows",
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
    fn collect_replay_summaries_query_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = "SELECT \
                   ? AS device_id, \
                   1::BIGINT AS context_id, \
                   2::BIGINT AS correlation_id, \
                   10::BIGINT AS start_ns, \
                   20::BIGINT AS end_ns, \
                   10::BIGINT AS sum_gpu_ns, \
                   1::BIGINT AS event_count, \
                   1::BIGINT AS kernel_count, \
                   0::BIGINT AS memcpy_count, \
                   0::BIGINT AS memset_count, \
                   0::BIGINT AS graph_trace_count, \
                   1::BIGINT AS stream_count, \
                   CAST(NULL AS BIGINT) AS graph_id, \
                   CAST(NULL AS BIGINT) AS graph_exec_id, \
                   1::BIGINT AS total_matched";

        let err = match collect_replay_summaries(&conn, sql, &[]) {
            Ok(rows) => anyhow::bail!(
                "unbound replay-summary SQL parameter should not hydrate successfully: {} rows",
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
    fn collect_replay_summaries_read_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = replay_summary_hydration_sql("'not-sum'");

        let err = match collect_replay_summaries(&conn, &sql, &[]) {
            Ok(rows) => anyhow::bail!(
                "malformed replay-summary row should not hydrate successfully: {} rows",
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
    fn hydrate_launchers_prepare_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;

        let err = match hydrate_launchers(&conn, "SELECT * FROM", &[]) {
            Ok(rows) => anyhow::bail!("malformed launcher SQL should fail, got {rows:?}"),
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
    fn hydrate_launchers_query_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;

        let err = match hydrate_launchers(&conn, "SELECT ? AS launcher_rowid", &[]) {
            Ok(rows) => anyhow::bail!("unbound launcher SQL parameter should fail, got {rows:?}"),
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
    fn hydrate_launchers_read_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = launcher_hydration_sql("'not-rowid'");

        let err = match hydrate_launchers(&conn, &sql, &[]) {
            Ok(rows) => anyhow::bail!("malformed launcher row should fail, got {rows:?}"),
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
    fn hydrate_node_events_prepare_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;

        let err = match hydrate_node_events(&conn, "SELECT * FROM", &[]) {
            Ok(rows) => anyhow::bail!(
                "malformed node-event SQL should not hydrate successfully: {} rows",
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
    fn hydrate_node_events_query_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = "SELECT \
                   ? AS kind, \
                   'node' AS name, \
                   1::BIGINT AS graph_node_id, \
                   7::BIGINT AS stream_id, \
                   10::BIGINT AS start_ns, \
                   20::BIGINT AS end_ns";

        let err = match hydrate_node_events(&conn, sql, &[]) {
            Ok(rows) => anyhow::bail!(
                "unbound node-event SQL parameter should not hydrate successfully: {} rows",
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
    fn hydrate_node_events_read_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = node_event_hydration_sql("'not-node-id'");

        let err = match hydrate_node_events(&conn, &sql, &[]) {
            Ok(rows) => anyhow::bail!(
                "malformed node-event row should not hydrate successfully: {} rows",
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
