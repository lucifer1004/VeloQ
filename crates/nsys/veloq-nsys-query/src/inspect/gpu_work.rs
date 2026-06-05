//! `inspect kernel:N` / `memcpy:N` / `memset:N` — GPU-work rows.
//!
//! Three CUPTI activity tables share the same per-row shape: device,
//! context, stream, optional graph identity, and a `correlationId`
//! that links back to the launching `CUPTI_ACTIVITY_KIND_RUNTIME`
//! row. Kernels carry grid/block geometry and shared-memory sizing;
//! memcpy/memset add `bytes` plus a copy/value field.

use crate::{NvtxContext, RowId};
use anyhow::{Context, Result};
use duckdb::Connection;
use duckdb::types::Value;
use serde::Serialize;

use super::{ColumnMap, EventDetails, maybe_col, opt_string};

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct KernelDetails {
    /// Cross-trace key — equal to `row_id` stringified.
    pub key: String,
    pub row_id: RowId,
    pub short_name: Option<String>,
    pub demangled_name: Option<String>,
    pub start_ns: i64,
    pub end_ns: i64,
    pub duration_ns: i64,
    pub device_id: i32,
    pub context_id: i64,
    pub stream_id: i64,
    pub grid: [i64; 3],
    pub block: [i64; 3],
    pub registers_per_thread: Option<i64>,
    pub static_shared_memory: Option<i64>,
    pub dynamic_shared_memory: Option<i64>,
    pub correlation_id: Option<i64>,
    pub global_pid: Option<i64>,
    /// Captured-graph id when the kernel ran inside a CUDA graph
    /// (`--cuda-graph-trace=node` captures); `None` for eager kernels
    /// and for traces where the column is absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_id: Option<i64>,
    /// Node id within the captured graph; joins to `graph_node:<id>`
    /// via `CUDA_GRAPH_NODE_EVENTS.graphNodeId`. `None` for eager
    /// kernels.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_node_id: Option<i64>,
    /// Innermost NVTX range that was open on the launching host
    /// thread when the runtime call corresponding to this kernel
    /// fired. Populated automatically when the trace has NVTX +
    /// CUPTI_RUNTIME tables; `None` everywhere else.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_context: Option<NvtxContext>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct MemcpyDetails {
    /// Cross-trace key — equal to `row_id` stringified.
    pub key: String,
    pub row_id: RowId,
    pub start_ns: i64,
    pub end_ns: i64,
    pub duration_ns: i64,
    pub device_id: i32,
    pub context_id: i64,
    pub stream_id: i64,
    pub bytes: i64,
    pub copy_kind: i64,
    pub copy_kind_name: &'static str,
    pub correlation_id: Option<i64>,
    /// Node id when the memcpy ran inside a CUDA graph
    /// (`--cuda-graph-trace=node`). Memcpy/memset tables don't carry
    /// `graphId` directly — join `CUDA_GRAPH_NODE_EVENTS` or the
    /// kernel table on this id if you need the parent graph.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_node_id: Option<i64>,
    /// See [`KernelDetails::nvtx_context`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_context: Option<NvtxContext>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct MemsetDetails {
    /// Cross-trace key — equal to `row_id` stringified.
    pub key: String,
    pub row_id: RowId,
    pub start_ns: i64,
    pub end_ns: i64,
    pub duration_ns: i64,
    pub device_id: i32,
    pub context_id: i64,
    pub stream_id: i64,
    pub bytes: i64,
    pub value: Option<i64>,
    pub correlation_id: Option<i64>,
    /// Node id when the memset ran inside a CUDA graph
    /// (`--cuda-graph-trace=node`). See [`MemcpyDetails::graph_node_id`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_node_id: Option<i64>,
    /// See [`KernelDetails::nvtx_context`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_context: Option<NvtxContext>,
}

pub(super) fn query_kernel(
    conn: &Connection,
    cols: &ColumnMap,
    id: RowId,
) -> Result<Option<EventDetails>> {
    const T: &str = "CUPTI_ACTIVITY_KIND_KERNEL";
    if !cols.contains_key(T) {
        return Ok(None);
    }
    let reg = maybe_col(cols, T, "registersPerThread");
    let smem_static = maybe_col(cols, T, "staticSharedMemory");
    let smem_dyn = maybe_col(cols, T, "dynamicSharedMemory");
    let corr = maybe_col(cols, T, "correlationId");
    let gpid = maybe_col(cols, T, "globalPid");
    let gid = maybe_col(cols, T, "graphId");
    let gnid = maybe_col(cols, T, "graphNodeId");
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
            CAST(t.gridX AS BIGINT), CAST(t.gridY AS BIGINT), CAST(t.gridZ AS BIGINT),
            CAST(t.blockX AS BIGINT), CAST(t.blockY AS BIGINT), CAST(t.blockZ AS BIGINT),
            CAST({reg} AS BIGINT),
            CAST({smem_static} AS BIGINT),
            CAST({smem_dyn} AS BIGINT),
            CAST({corr} AS BIGINT),
            CAST({gpid} AS BIGINT),
            s_sh.value,
            s_dem.value,
            CAST({gid} AS BIGINT),
            CAST({gnid} AS BIGINT)
        FROM nsight.{T} t
        LEFT JOIN nsight.StringIds s_sh  ON t.shortName = s_sh.id
        LEFT JOIN nsight.StringIds s_dem ON t.demangledName = s_dem.id
        WHERE t.rowid = ?
        "#
    );
    let mut stmt = conn.prepare(&sql).context("prepare kernel inspect")?;
    let mut rows = stmt.query([Value::BigInt(id.rowid)])?;
    let Some(r) = rows.next()? else {
        return Ok(None);
    };
    let start_ns: i64 = r.get(0)?;
    let end_ns: i64 = r.get(1)?;
    Ok(Some(EventDetails::Kernel(KernelDetails {
        key: id.to_string(),
        row_id: id,
        short_name: opt_string(r, 16)?,
        demangled_name: opt_string(r, 17)?,
        start_ns,
        end_ns,
        duration_ns: end_ns - start_ns,
        device_id: r.get(2)?,
        context_id: r.get(3)?,
        stream_id: r.get(4)?,
        grid: [r.get(5)?, r.get(6)?, r.get(7)?],
        block: [r.get(8)?, r.get(9)?, r.get(10)?],
        registers_per_thread: r.get(11)?,
        static_shared_memory: r.get(12)?,
        dynamic_shared_memory: r.get(13)?,
        correlation_id: r.get(14)?,
        global_pid: r.get(15)?,
        graph_id: r.get(18)?,
        graph_node_id: r.get(19)?,
        nvtx_context: None,
    })))
}

pub(super) fn query_memcpy(
    conn: &Connection,
    cols: &ColumnMap,
    id: RowId,
) -> Result<Option<EventDetails>> {
    const T: &str = "CUPTI_ACTIVITY_KIND_MEMCPY";
    if !cols.contains_key(T) {
        return Ok(None);
    }
    let corr = maybe_col(cols, T, "correlationId");
    let gnid = maybe_col(cols, T, "graphNodeId");
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
            CAST(t.bytes AS BIGINT),
            CAST(t.copyKind AS BIGINT),
            CAST({corr} AS BIGINT),
            CAST({gnid} AS BIGINT)
        FROM nsight.{T} t
        WHERE t.rowid = ?
        "#
    );
    let mut stmt = conn.prepare(&sql).context("prepare memcpy inspect")?;
    let mut rows = stmt.query([Value::BigInt(id.rowid)])?;
    let Some(r) = rows.next()? else {
        return Ok(None);
    };
    let start_ns: i64 = r.get(0)?;
    let end_ns: i64 = r.get(1)?;
    let copy_kind: i64 = r.get(6)?;
    Ok(Some(EventDetails::Memcpy(MemcpyDetails {
        key: id.to_string(),
        row_id: id,
        start_ns,
        end_ns,
        duration_ns: end_ns - start_ns,
        device_id: r.get(2)?,
        context_id: r.get(3)?,
        stream_id: r.get(4)?,
        bytes: r.get(5)?,
        copy_kind,
        copy_kind_name: crate::kind_sql::copy_kind_label(copy_kind),
        correlation_id: r.get(7)?,
        graph_node_id: r.get(8)?,
        nvtx_context: None,
    })))
}

pub(super) fn query_memset(
    conn: &Connection,
    cols: &ColumnMap,
    id: RowId,
) -> Result<Option<EventDetails>> {
    const T: &str = "CUPTI_ACTIVITY_KIND_MEMSET";
    if !cols.contains_key(T) {
        return Ok(None);
    }
    let val = maybe_col(cols, T, "value");
    let corr = maybe_col(cols, T, "correlationId");
    let gnid = maybe_col(cols, T, "graphNodeId");
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
            CAST(t.bytes AS BIGINT),
            CAST({val} AS BIGINT),
            CAST({corr} AS BIGINT),
            CAST({gnid} AS BIGINT)
        FROM nsight.{T} t
        WHERE t.rowid = ?
        "#
    );
    let mut stmt = conn.prepare(&sql).context("prepare memset inspect")?;
    let mut rows = stmt.query([Value::BigInt(id.rowid)])?;
    let Some(r) = rows.next()? else {
        return Ok(None);
    };
    let start_ns: i64 = r.get(0)?;
    let end_ns: i64 = r.get(1)?;
    Ok(Some(EventDetails::Memset(MemsetDetails {
        key: id.to_string(),
        row_id: id,
        start_ns,
        end_ns,
        duration_ns: end_ns - start_ns,
        device_id: r.get(2)?,
        context_id: r.get(3)?,
        stream_id: r.get(4)?,
        bytes: r.get(5)?,
        value: r.get(6)?,
        correlation_id: r.get(7)?,
        graph_node_id: r.get(8)?,
        nvtx_context: None,
    })))
}
