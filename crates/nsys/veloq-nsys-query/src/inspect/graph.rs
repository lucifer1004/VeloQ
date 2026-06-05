//! `inspect graph:N` / `graph_node:N` / `graph_event:N` — CUDA-graph
//! lifecycle rows.
//!
//! - `graph` rows (CUPTI_ACTIVITY_KIND_GRAPH_TRACE) record GPU
//!   execution of a captured graph in `--cuda-graph-trace=graph`
//!   captures: the inner kernels don't appear in the kernel table.
//! - `graph_node` rows (CUDA_GRAPH_NODE_EVENTS) are node-creation
//!   metadata under `--cuda-graph-trace=node` captures; execution
//!   timing lives on the kernel/memcpy/memset rows that share the
//!   node's `graphNodeId`. Two best-effort subqueries enrich each
//!   node with its parent `graph_id` and `graph_exec_id`.
//! - `graph_event` rows (CUDA_GRAPH_EVENTS) are host-side API hook
//!   markers: `Graph Creation` (eventClass 95), `GraphExec Creation`
//!   (eventClass 94), and friends.

use crate::RowId;
use anyhow::{Context, Result};
use duckdb::Connection;
use duckdb::types::Value;
use serde::Serialize;

use super::{ColumnMap, EventDetails, maybe_col};

/// CUDA graph launch (`CUPTI_ACTIVITY_KIND_GRAPH_TRACE`). One row per
/// graph execution; in `--cuda-graph-trace=graph` captures the inner
/// kernels are rolled up and do not appear in
/// `CUPTI_ACTIVITY_KIND_KERNEL`.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct GraphDetails {
    /// Cross-trace key — equal to `row_id` stringified.
    pub key: String,
    pub row_id: RowId,
    pub start_ns: i64,
    pub end_ns: i64,
    pub duration_ns: i64,
    pub device_id: i32,
    pub context_id: i64,
    pub stream_id: i64,
    /// Captured-graph identity. Stable across launches of the same
    /// captured graph within one trace.
    pub graph_id: i64,
    /// One per `cudaGraphInstantiate`. Distinguishes graph executables
    /// that share a captured graph.
    pub graph_exec_id: i64,
    /// Shared with the launching `cudaGraphLaunch` runtime call.
    pub correlation_id: Option<i64>,
}

/// CUDA graph-node metadata (`CUDA_GRAPH_NODE_EVENTS`). One row per
/// node creation in `--cuda-graph-trace=node` captures. Execution
/// timing rides on the kernel/memcpy/memset rows whose `graphNodeId`
/// equals this id; this row carries the metadata only (which API
/// created the node, original id if cloned).
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct GraphNodeDetails {
    /// Cross-trace key — equal to `row_id` stringified.
    pub key: String,
    pub row_id: RowId,
    pub start_ns: i64,
    /// Most NODE_EVENTS rows are instantaneous (`end = start`); kept
    /// optional for parity with NVTX-style markers.
    pub end_ns: Option<i64>,
    /// `0` for the common instantaneous-marker case.
    pub duration_ns: i64,
    pub global_tid: Option<i64>,
    /// Node identity. Joins to `kernel.graphNodeId` /
    /// `memcpy.graphNodeId` / `memset.graphNodeId` for execution data.
    pub graph_node_id: i64,
    /// If this node was cloned (e.g. from a sub-graph template), the
    /// original node's id. `None` when the node was created directly.
    pub original_graph_node_id: Option<i64>,
    /// Parent captured-graph id. Looked up from the kernel table
    /// (any kernel row sharing this `graphNodeId` carries its
    /// `graphId`). `None` on traces that lack populated kernel
    /// rows for this node (e.g. graph-mode captures, or unused
    /// nodes that were created but never executed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_id: Option<i64>,
    /// Graph-executable id. Looked up via `CUDA_GRAPH_EVENTS`
    /// (`GraphExec Creation` event with matching `graphId`). `None`
    /// when `CUDA_GRAPH_EVENTS` is absent — i.e. on most pure
    /// `--cuda-graph-trace=node` captures.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_exec_id: Option<i64>,
}

/// CUDA graph-lifecycle event (`CUDA_GRAPH_EVENTS`). Host-side
/// construction marker emitted by NSys's CUDA API hook layer when a
/// graph is created or an exec is instantiated. Instantaneous
/// (`end == start`); the value is the eventClass label and the graph
/// identity columns.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct GraphEventDetails {
    /// Cross-trace key — equal to `row_id` stringified.
    pub key: String,
    pub row_id: RowId,
    pub start_ns: i64,
    /// Equals `start_ns` for instantaneous markers.
    pub end_ns: i64,
    pub duration_ns: i64,
    /// Host thread that called the CUDA API.
    pub global_tid: Option<i64>,
    /// Raw `eventClass` value (`94` = GraphExec Creation,
    /// `95` = Graph Creation).
    pub event_class: i64,
    /// Snake-case label derived from `event_class` via the shared
    /// [`crate::kind_sql::graph_event_class_label`] table.
    pub event_class_name: &'static str,
    /// Captured-graph identity.
    pub graph_id: i64,
    /// If this graph was cloned from another captured graph, the
    /// source graph's id.
    pub original_graph_id: Option<i64>,
    /// Set on `GraphExec Creation` events (the instantiated executable
    /// id). `None` on raw `Graph Creation` events.
    pub graph_exec_id: Option<i64>,
}

pub(super) fn query_graph(
    conn: &Connection,
    cols: &ColumnMap,
    id: RowId,
) -> Result<Option<EventDetails>> {
    const T: &str = "CUPTI_ACTIVITY_KIND_GRAPH_TRACE";
    if !cols.contains_key(T) {
        return Ok(None);
    }
    let corr = maybe_col(cols, T, "correlationId");
    let dev = crate::kind_sql::GPU_DEVICE_ID_EXPR;
    let ctx = crate::kind_sql::GPU_CONTEXT_ID_EXPR;
    let stm = crate::kind_sql::GPU_STREAM_ID_EXPR;
    let sql = format!(
        r#"
        SELECT
            t.start, t."end",
            {dev},
            {ctx},
            {stm},
            CAST(t.graphId AS BIGINT),
            CAST(t.graphExecId AS BIGINT),
            CAST({corr} AS BIGINT)
        FROM nsight.{T} t
        WHERE t.rowid = ?
        "#
    );
    let mut stmt = conn.prepare(&sql).context("prepare graph inspect")?;
    let mut rows = stmt.query([Value::BigInt(id.rowid)])?;
    let Some(r) = rows.next()? else {
        return Ok(None);
    };
    let start_ns: i64 = r.get(0)?;
    let end_ns: i64 = r.get(1)?;
    Ok(Some(EventDetails::Graph(GraphDetails {
        key: id.to_string(),
        row_id: id,
        start_ns,
        end_ns,
        duration_ns: end_ns - start_ns,
        device_id: r.get(2)?,
        context_id: r.get(3)?,
        stream_id: r.get(4)?,
        graph_id: r.get(5)?,
        graph_exec_id: r.get(6)?,
        correlation_id: r.get(7)?,
    })))
}

pub(super) fn query_graph_node(
    conn: &Connection,
    cols: &ColumnMap,
    id: RowId,
) -> Result<Option<EventDetails>> {
    const T: &str = "CUDA_GRAPH_NODE_EVENTS";
    if !cols.contains_key(T) {
        return Ok(None);
    }
    let orig = maybe_col(cols, T, "originalGraphNodeId");
    let gtid = maybe_col(cols, T, "globalTid");

    // Best-effort enrichment of the parent graph identity:
    //   - `graph_id`: pick `graphId` from any kernel row sharing the
    //     same `graphNodeId`. Memcpy/memset don't store `graphId` in
    //     the NSys schema so the kernel table is the canonical
    //     join. NULL when no kernel ran on this node (rare: nodes
    //     created but never executed).
    //   - `graph_exec_id`: chain through `CUDA_GRAPH_EVENTS` where
    //     `eventClass = 94` (`GraphExec Creation`) and `graphId`
    //     matches the one found above. Absent in pure
    //     `--cuda-graph-trace=node` captures (the API hook log
    //     table isn't produced) — in that case the field is NULL.
    let has_kernel = cols.contains_key("CUPTI_ACTIVITY_KIND_KERNEL");
    let has_graph_events = cols.contains_key("CUDA_GRAPH_EVENTS");
    let graph_id_subq = if has_kernel {
        "(SELECT k.graphId FROM nsight.CUPTI_ACTIVITY_KIND_KERNEL k \
          WHERE k.graphNodeId = t.graphNodeId LIMIT 1)"
    } else {
        "NULL"
    };
    let graph_exec_id_subq = if has_kernel && has_graph_events {
        "(SELECT ge.graphExecId FROM nsight.CUDA_GRAPH_EVENTS ge \
          WHERE ge.eventClass = 94 \
            AND ge.graphId = (SELECT k.graphId \
                              FROM nsight.CUPTI_ACTIVITY_KIND_KERNEL k \
                              WHERE k.graphNodeId = t.graphNodeId LIMIT 1) \
          LIMIT 1)"
    } else {
        "NULL"
    };
    let gtid_expr = veloq_nsys_data::sql_expr::u64_bits_to_i64(&gtid);

    let sql = format!(
        r#"
        SELECT
            t.start, t."end",
            CAST(t.graphNodeId AS BIGINT),
            CAST({orig} AS BIGINT),
            {gtid_expr},
            CAST({graph_id_subq} AS BIGINT),
            CAST({graph_exec_id_subq} AS BIGINT)
        FROM nsight.{T} t
        WHERE t.rowid = ?
        "#
    );
    let mut stmt = conn.prepare(&sql).context("prepare graph_node inspect")?;
    let mut rows = stmt.query([Value::BigInt(id.rowid)])?;
    let Some(r) = rows.next()? else {
        return Ok(None);
    };
    let start_ns: i64 = r.get(0)?;
    let end_ns: Option<i64> = r.get(1)?;
    let duration_ns = end_ns.map(|e| e - start_ns).unwrap_or(0);
    Ok(Some(EventDetails::GraphNode(GraphNodeDetails {
        key: id.to_string(),
        row_id: id,
        start_ns,
        end_ns,
        duration_ns,
        global_tid: r.get(4)?,
        graph_node_id: r.get(2)?,
        original_graph_node_id: r.get(3)?,
        graph_id: r.get(5)?,
        graph_exec_id: r.get(6)?,
    })))
}

pub(super) fn query_graph_event(
    conn: &Connection,
    cols: &ColumnMap,
    id: RowId,
) -> Result<Option<EventDetails>> {
    const T: &str = "CUDA_GRAPH_EVENTS";
    if !cols.contains_key(T) {
        return Ok(None);
    }
    let orig = maybe_col(cols, T, "originalGraphId");
    let gexec = maybe_col(cols, T, "graphExecId");
    let gtid = maybe_col(cols, T, "globalTid");
    let gtid_expr = veloq_nsys_data::sql_expr::u64_bits_to_i64(&gtid);
    let sql = format!(
        r#"
        SELECT
            t.start, t."end",
            CAST(t.eventClass AS BIGINT),
            CAST(t.graphId AS BIGINT),
            CAST({orig} AS BIGINT),
            CAST({gexec} AS BIGINT),
            {gtid_expr}
        FROM nsight.{T} t
        WHERE t.rowid = ?
        "#
    );
    let mut stmt = conn.prepare(&sql).context("prepare graph_event inspect")?;
    let mut rows = stmt.query([Value::BigInt(id.rowid)])?;
    let Some(r) = rows.next()? else {
        return Ok(None);
    };
    let start_ns: i64 = r.get(0)?;
    let end_ns: i64 = r.get(1)?;
    let event_class: i64 = r.get(2)?;
    Ok(Some(EventDetails::GraphEvent(GraphEventDetails {
        key: id.to_string(),
        row_id: id,
        start_ns,
        end_ns,
        duration_ns: end_ns - start_ns,
        global_tid: r.get(6)?,
        event_class,
        event_class_name: crate::kind_sql::graph_event_class_label(event_class),
        graph_id: r.get(3)?,
        original_graph_id: r.get(4)?,
        graph_exec_id: r.get(5)?,
    })))
}
