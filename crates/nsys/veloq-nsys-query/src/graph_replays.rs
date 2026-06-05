//! `veloq graph-replays <trace>` — CUDA graph replay decomposition.
//!
//! CUDA graph captures appear in two NSys shapes:
//! - `--cuda-graph-trace=graph`: one `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`
//!   row is one replay. It has replay wall time but no node-level
//!   kernel/memcpy/memset decomposition.
//! - `--cuda-graph-trace=node`: graph-captured GPU work lands in the
//!   normal kernel/memcpy/memset tables with `graphNodeId` populated.
//!   Replays are keyed by the documented correlation triple
//!   `(deviceId, contextId, correlationId)`.
//!
//! Raw `correlationId` is never used alone. Every public row carries
//! the packed [`veloq_nsys_data::SyntheticId`] display value for the
//! full triple.

use crate::RowId;
use anyhow::{Context, Result};
use duckdb::types::Value;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use veloq_core::sort::build_order_by;
use veloq_core::time::TimeWindow;
use veloq_core::{Direction, SortKeyDef, SortKeySpec, SortSpec};
use veloq_nsys_data::{SyntheticId, Trace};

#[derive(Debug, Clone)]
pub struct GraphReplaysRequest {
    pub time_window: Option<TimeWindow>,
    /// Launch-scoped NVTX glob. Matches enclosing NVTX names around
    /// `cudaGraphLaunch%` runtime rows, then joins launches to replay
    /// work by `(device, context, correlationId)`.
    pub nvtx: Option<String>,
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
    /// List key for this replay. Includes the packed synthetic
    /// correlation id so two devices/processes reusing a raw
    /// `correlationId` stay distinct.
    pub key: String,
    pub capture_mode: CaptureMode,
    pub synthetic_id: String,
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

#[derive(Debug, Clone)]
struct NodeEvent {
    kind: String,
    name: String,
    graph_node_id: i64,
    stream_id: i64,
    start_ns: i64,
    end_ns: i64,
}

pub fn run<P: AsRef<Path>>(path: P, req: GraphReplaysRequest) -> Result<GraphReplaysResponse> {
    crate::check_limit(req.limit)?;
    if req.top_nodes_limit == 0 {
        anyhow::bail!("--top-nodes must be at least 1");
    }

    let trace = Trace::open(path)?;
    let abs_window = trace.resolve_window(req.time_window)?;
    let mode = capture_mode(&trace)?;

    let mut rows = match mode {
        CaptureMode::GraphTrace => query_graph_trace(&trace, &req, abs_window)?,
        CaptureMode::GraphNodes => query_graph_nodes(&trace, &req, abs_window)?,
        CaptureMode::None => Vec::new(),
    };

    let total_matched = rows.first().map(|(_, total)| *total).unwrap_or(0);
    let mut out_rows = Vec::with_capacity(rows.len());
    for (summary, _) in rows.drain(..) {
        let launcher = find_launcher(&trace, &summary)?;
        let synthetic = SyntheticId::pack(
            summary.device_id as u64,
            summary.context_id as u64,
            summary.correlation_id as u64,
        )
        .to_string();
        let (busy_ns, top_nodes, decomposition_available) = match mode {
            CaptureMode::GraphTrace => (summary.end_ns - summary.start_ns, Vec::new(), false),
            CaptureMode::GraphNodes => {
                let events = load_node_events(&trace, &summary)?;
                let busy = busy_ns(events.iter().map(|e| (e.start_ns, e.end_ns)).collect());
                let nodes = top_nodes(
                    &events,
                    summary.end_ns - summary.start_ns,
                    req.top_nodes_limit,
                );
                (busy, nodes, true)
            }
            CaptureMode::None => (0, Vec::new(), false),
        };
        let wall_ns = summary.end_ns - summary.start_ns;
        out_rows.push(GraphReplayRow {
            key: format!("graph-replay|{synthetic}"),
            capture_mode: mode,
            synthetic_id: synthetic,
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

fn capture_mode(trace: &Trace) -> Result<CaptureMode> {
    if trace.table_exists("CUPTI_ACTIVITY_KIND_GRAPH_TRACE") {
        let count: i64 = trace.conn().query_row(
            "SELECT COUNT(*) FROM nsight.CUPTI_ACTIVITY_KIND_GRAPH_TRACE \
             WHERE correlationId IS NOT NULL AND start IS NOT NULL AND \"end\" IS NOT NULL",
            [],
            |r| r.get(0),
        )?;
        if count > 0 {
            return Ok(CaptureMode::GraphTrace);
        }
    }

    if !node_event_subqueries(trace).is_empty() {
        let union = node_event_subqueries(trace).join(" UNION ALL ");
        let sql = format!("WITH event_rows AS ({union}) SELECT COUNT(*) FROM event_rows");
        let count: i64 = trace.conn().query_row(&sql, [], |r| r.get(0))?;
        if count > 0 {
            return Ok(CaptureMode::GraphNodes);
        }
    }

    Ok(CaptureMode::None)
}

fn query_graph_trace(
    trace: &Trace,
    req: &GraphReplaysRequest,
    abs_window: Option<(i64, i64)>,
) -> Result<Vec<(ReplaySummary, i64)>> {
    let mut params = Vec::new();
    let (scope_cte, scoped_join) = launch_scope_sql(trace, req.nvtx.as_deref(), &mut params)?;
    let mut where_parts = vec![
        "t.correlationId IS NOT NULL".to_string(),
        "t.start IS NOT NULL".to_string(),
        "t.\"end\" IS NOT NULL".to_string(),
    ];
    if let Some((start, end)) = abs_window {
        where_parts.push("t.\"end\" > ? AND t.start < ?".to_string());
        params.push(Value::BigInt(start));
        params.push(Value::BigInt(end));
    }
    if let Some(device) = req.device {
        where_parts.push("CAST(t.deviceId AS INTEGER) = ?".to_string());
        params.push(Value::Int(device));
    }
    let where_sql = where_parts.join(" AND ");
    let order_by = order_by_sql(req.sort.as_ref())?;
    params.push(Value::BigInt(req.limit as i64));

    let sql = format!(
        r#"
        WITH {scope_cte}
        base AS (
            SELECT
                CAST(t.deviceId AS INTEGER) AS device_id,
                CAST(t.contextId AS BIGINT) AS context_id,
                CAST(t.correlationId AS BIGINT) AS correlation_id,
                CAST(t.start AS BIGINT) AS start_ns,
                CAST(t."end" AS BIGINT) AS end_ns,
                CAST(t."end" - t.start AS BIGINT) AS wall_ns,
                CAST(t."end" - t.start AS BIGINT) AS sum_gpu_ns,
                CAST(1 AS BIGINT) AS event_count,
                CAST(0 AS BIGINT) AS kernel_count,
                CAST(0 AS BIGINT) AS memcpy_count,
                CAST(0 AS BIGINT) AS memset_count,
                CAST(1 AS BIGINT) AS graph_trace_count,
                CAST(1 AS BIGINT) AS stream_count,
                CAST(t.graphId AS BIGINT) AS graph_id,
                CAST(t.graphExecId AS BIGINT) AS graph_exec_id
            FROM nsight.CUPTI_ACTIVITY_KIND_GRAPH_TRACE t
            WHERE {where_sql}
        ),
        scoped AS (
            SELECT b.*
            FROM base b
            {scoped_join}
        )
        SELECT *,
               CAST(COUNT(*) OVER () AS BIGINT) AS total_matched
        FROM scoped
        ORDER BY {order_by}
        LIMIT ?
        "#
    );

    collect_replay_summaries(trace, &sql, &params).context("querying graph_trace replays")
}

fn query_graph_nodes(
    trace: &Trace,
    req: &GraphReplaysRequest,
    abs_window: Option<(i64, i64)>,
) -> Result<Vec<(ReplaySummary, i64)>> {
    let subqueries = node_event_subqueries(trace);
    if subqueries.is_empty() {
        return Ok(Vec::new());
    }
    let union = subqueries.join(" UNION ALL ");
    let mut params = Vec::new();
    let (scope_cte, scoped_join) = launch_scope_sql(trace, req.nvtx.as_deref(), &mut params)?;
    let mut where_parts = Vec::new();
    append_scope_filters(&mut where_parts, &mut params, abs_window, req.device);
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
            GROUP BY device_id, context_id, correlation_id
        ),
        scoped AS (
            SELECT b.*
            FROM replay_base b
            {scoped_join}
        )
        SELECT *,
               CAST(COUNT(*) OVER () AS BIGINT) AS total_matched
        FROM scoped
        ORDER BY {order_by}
        LIMIT ?
        "#
    );

    collect_replay_summaries(trace, &sql, &params).context("querying graph node replays")
}

fn collect_replay_summaries(
    trace: &Trace,
    sql: &str,
    params: &[Value],
) -> Result<Vec<(ReplaySummary, i64)>> {
    let mut stmt = trace.conn().prepare(sql)?;
    let params_ref = crate::bind(params);
    let mut rows = stmt.query(params_ref.as_slice())?;
    let mut out = Vec::new();
    while let Some(r) = rows.next()? {
        out.push((
            ReplaySummary {
                device_id: r.get("device_id")?,
                context_id: r.get("context_id")?,
                correlation_id: r.get("correlation_id")?,
                start_ns: r.get("start_ns")?,
                end_ns: r.get("end_ns")?,
                sum_gpu_ns: r.get("sum_gpu_ns")?,
                event_count: r.get("event_count")?,
                kernel_count: r.get("kernel_count")?,
                memcpy_count: r.get("memcpy_count")?,
                memset_count: r.get("memset_count")?,
                graph_trace_count: r.get("graph_trace_count")?,
                stream_count: r.get("stream_count")?,
                graph_id: r.get("graph_id")?,
                graph_exec_id: r.get("graph_exec_id")?,
            },
            r.get("total_matched")?,
        ));
    }
    Ok(out)
}

fn node_event_subqueries(trace: &Trace) -> Vec<String> {
    let mut out = Vec::new();
    if trace.table_exists("CUPTI_ACTIVITY_KIND_KERNEL") {
        out.push(
            r#"
            SELECT
                'kernel' AS kind,
                CAST(t.rowid AS BIGINT) AS rowid,
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
            WHERE t.graphNodeId IS NOT NULL
              AND t.correlationId IS NOT NULL
              AND t.deviceId IS NOT NULL
              AND t.contextId IS NOT NULL
              AND t.start IS NOT NULL
              AND t."end" IS NOT NULL
            "#
            .to_string(),
        );
    }
    if trace.table_exists("CUPTI_ACTIVITY_KIND_MEMCPY") {
        out.push(
            r#"
            SELECT
                'memcpy' AS kind,
                CAST(t.rowid AS BIGINT) AS rowid,
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
            WHERE t.graphNodeId IS NOT NULL
              AND t.correlationId IS NOT NULL
              AND t.deviceId IS NOT NULL
              AND t.contextId IS NOT NULL
              AND t.start IS NOT NULL
              AND t."end" IS NOT NULL
            "#
            .to_string(),
        );
    }
    if trace.table_exists("CUPTI_ACTIVITY_KIND_MEMSET") {
        out.push(
            r#"
            SELECT
                'memset' AS kind,
                CAST(t.rowid AS BIGINT) AS rowid,
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
            WHERE t.graphNodeId IS NOT NULL
              AND t.correlationId IS NOT NULL
              AND t.deviceId IS NOT NULL
              AND t.contextId IS NOT NULL
              AND t.start IS NOT NULL
              AND t."end" IS NOT NULL
            "#
            .to_string(),
        );
    }
    out
}

fn append_scope_filters(
    where_parts: &mut Vec<String>,
    params: &mut Vec<Value>,
    abs_window: Option<(i64, i64)>,
    device: Option<i32>,
) {
    if let Some((start, end)) = abs_window {
        where_parts.push("end_ns > ? AND start_ns < ?".to_string());
        params.push(Value::BigInt(start));
        params.push(Value::BigInt(end));
    }
    crate::kind_policy::LocationFilter {
        device,
        stream: None,
    }
    .push_where(&mut *where_parts, &mut *params);
}

fn launch_scope_sql(
    trace: &Trace,
    nvtx: Option<&str>,
    params: &mut Vec<Value>,
) -> Result<(String, String)> {
    let Some(pattern) = nvtx else {
        return Ok((String::new(), String::new()));
    };
    for table in [
        "NVTX_EVENTS",
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        "TARGET_INFO_CUDA_CONTEXT_INFO",
    ] {
        if !trace.table_exists(table) {
            anyhow::bail!(
                "graph-replays --nvtx requires `{table}`, which is not present in this trace"
            );
        }
    }
    let like_pattern = crate::search_glob_to_like(pattern);
    params.push(Value::Text(like_pattern.clone()));
    params.push(Value::Text(like_pattern));
    let cte = r#"
        matched_launches AS MATERIALIZED (
            SELECT DISTINCT
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
                          MAX(CASE WHEN nvtx_name LIKE ? ESCAPE '\' THEN 1 ELSE 0 END) AS name_match,
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
                     OR p.nvtx_path LIKE ? ESCAPE '\'
              )
        ),
        "#
    .to_string();
    Ok((
        cte,
        "JOIN matched_launches ml USING (device_id, context_id, correlation_id)".to_string(),
    ))
}

fn order_by_sql(sort: Option<&SortSpec>) -> Result<String> {
    let default_sort = SortSpec::single("wall");
    let sort = sort.unwrap_or(&default_sort);
    let mut parts = Vec::new();
    for field in sort.fields() {
        let (key, dir) = SortKey::from_field(field)?;
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
        "device_id ASC, context_id ASC, correlation_id",
    ))
}

fn find_launcher(trace: &Trace, replay: &ReplaySummary) -> Result<Option<RowId>> {
    if !trace.table_exists("CUPTI_ACTIVITY_KIND_RUNTIME")
        || !trace.table_exists("TARGET_INFO_CUDA_CONTEXT_INFO")
    {
        return Ok(None);
    }
    let sql = r#"
        SELECT CAST(r.rowid AS BIGINT) AS rowid
        FROM nsight.CUPTI_ACTIVITY_KIND_RUNTIME r
        LEFT JOIN nsight.StringIds s ON r.nameId = s.id
        JOIN nsight.TARGET_INFO_CUDA_CONTEXT_INFO c
          ON CAST(c.processId AS BIGINT) = CAST(((r.globalTid >> 24) & 16777215) AS BIGINT)
        WHERE CAST(c.deviceId AS INTEGER) = ?
          AND CAST(c.contextId AS BIGINT) = ?
          AND CAST(r.correlationId AS BIGINT) = ?
        ORDER BY
          CASE WHEN COALESCE(s.value, '') LIKE 'cudaGraphLaunch%' THEN 0 ELSE 1 END ASC,
          CASE WHEN r.start <= ? THEN 0 ELSE 1 END ASC,
          ABS(r.start - ?) ASC,
          r.rowid ASC
        LIMIT 1
        "#;
    let params = [
        Value::Int(replay.device_id),
        Value::BigInt(replay.context_id),
        Value::BigInt(replay.correlation_id),
        Value::BigInt(replay.start_ns),
        Value::BigInt(replay.start_ns),
    ];
    let mut stmt = trace.conn().prepare(sql)?;
    let bound = crate::bind(&params);
    let mut rows = stmt.query(bound.as_slice())?;
    Ok(if let Some(row) = rows.next()? {
        Some(RowId::new(crate::EventKind::Runtime, row.get("rowid")?))
    } else {
        None
    })
}

fn load_node_events(trace: &Trace, replay: &ReplaySummary) -> Result<Vec<NodeEvent>> {
    let subqueries = node_event_subqueries(trace);
    if subqueries.is_empty() {
        return Ok(Vec::new());
    }
    let union = subqueries.join(" UNION ALL ");
    let sql = format!(
        r#"
        WITH event_rows AS ({union})
        SELECT kind, name, graph_node_id, stream_id, start_ns, end_ns
        FROM event_rows
        WHERE device_id = ?
          AND context_id = ?
          AND correlation_id = ?
        ORDER BY start_ns ASC, end_ns ASC, rowid ASC
        "#
    );
    let params = [
        Value::Int(replay.device_id),
        Value::BigInt(replay.context_id),
        Value::BigInt(replay.correlation_id),
    ];
    let mut stmt = trace.conn().prepare(&sql)?;
    let bound = crate::bind(&params);
    let mut rows = stmt.query(bound.as_slice())?;
    let mut out = Vec::new();
    while let Some(r) = rows.next()? {
        out.push(NodeEvent {
            kind: r.get("kind")?,
            name: r.get("name")?,
            graph_node_id: r.get("graph_node_id")?,
            stream_id: r.get("stream_id")?,
            start_ns: r.get("start_ns")?,
            end_ns: r.get("end_ns")?,
        });
    }
    Ok(out)
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

    #[test]
    fn busy_ns_unions_overlaps() {
        assert_eq!(busy_ns(vec![(0, 10), (5, 15), (20, 25)]), 20);
    }
}
