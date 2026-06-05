//! `inspect runtime:N` / `osrt:N` / `nvtx:N` — host-side API rows.
//!
//! These three kinds live on the host thread, not the GPU: CUDA
//! runtime API calls, OS-runtime events, and NVTX range / marker
//! events. They share `globalTid` as their identity anchor and (for
//! Runtime / Osrt) a name resolved through `StringIds`. NVTX rows
//! carry a per-`(globalTid, domainId)` nesting depth computed once
//! per call into `Trace::nvtx_nesting()` and passed in by the
//! dispatcher.

use crate::RowId;
use anyhow::{Context, Result};
use duckdb::Connection;
use duckdb::types::Value;
use serde::Serialize;

use super::{ColumnMap, EventDetails};
use crate::event_ref::NvtxContext;

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct RuntimeDetails {
    /// Cross-trace key — equal to `row_id` stringified.
    pub key: String,
    pub row_id: RowId,
    pub name: String,
    pub start_ns: i64,
    pub end_ns: i64,
    pub duration_ns: i64,
    pub global_tid: i64,
    pub correlation_id: Option<i64>,
    /// Innermost NVTX range open on this runtime call's `global_tid`
    /// when it fired. Populated by `inspect` whenever an NVTX_EVENTS
    /// table is present; `None` for traces without NVTX or when no
    /// range was open.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_context: Option<NvtxContext>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct OsrtDetails {
    /// Cross-trace key — equal to `row_id` stringified.
    pub key: String,
    pub row_id: RowId,
    pub name: String,
    pub start_ns: i64,
    pub end_ns: i64,
    pub duration_ns: i64,
    pub global_tid: i64,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct NvtxDetails {
    /// Cross-trace key — equal to `row_id` stringified.
    pub key: String,
    pub row_id: RowId,
    pub name: String,
    pub start_ns: i64,
    pub end_ns: Option<i64>,
    pub duration_ns: Option<i64>,
    pub global_tid: i64,
    pub domain_id: i64,
    pub event_type: i64,
    /// Depth in the per-(global_tid, domain_id) nesting stack.
    /// 0 = outermost range. `None` only when nesting computation
    /// failed silently (shouldn't happen in practice).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth: Option<u8>,
    /// Full slash-joined NVTX hierarchy path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// Parent NVTX range row id when this range is nested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_row_id: Option<RowId>,
    /// Parent NVTX range name when this range is nested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_name: Option<String>,
}

pub(super) fn query_runtime(
    conn: &Connection,
    cols: &ColumnMap,
    id: RowId,
) -> Result<Option<EventDetails>> {
    const T: &str = "CUPTI_ACTIVITY_KIND_RUNTIME";
    if !cols.contains_key(T) {
        return Ok(None);
    }
    let global_tid = veloq_nsys_data::sql_expr::u64_bits_to_i64("t.globalTid");
    let sql = format!(
        r#"
        SELECT
            t.start, t."end",
            {global_tid},
            CAST(t.correlationId AS BIGINT),
            COALESCE(s.value, '<unknown>')
        FROM nsight.{T} t
        LEFT JOIN nsight.StringIds s ON t.nameId = s.id
        WHERE t.rowid = ?
        "#
    );
    let mut stmt = conn.prepare(&sql).context("prepare runtime inspect")?;
    let mut rows = stmt.query([Value::BigInt(id.rowid)])?;
    let Some(r) = rows.next()? else {
        return Ok(None);
    };
    let start_ns: i64 = r.get(0)?;
    let end_ns: i64 = r.get(1)?;
    Ok(Some(EventDetails::Runtime(RuntimeDetails {
        key: id.to_string(),
        row_id: id,
        name: r.get(4)?,
        start_ns,
        end_ns,
        duration_ns: end_ns - start_ns,
        global_tid: r.get(2)?,
        correlation_id: r.get(3)?,
        // Populated by the inspect dispatcher's `attach_nvtx_context`
        // pass when nesting was computed for the request.
        nvtx_context: None,
    })))
}

pub(super) fn query_osrt(
    conn: &Connection,
    cols: &ColumnMap,
    id: RowId,
) -> Result<Option<EventDetails>> {
    const T: &str = "OSRT_API";
    if !cols.contains_key(T) {
        return Ok(None);
    }
    let global_tid = veloq_nsys_data::sql_expr::u64_bits_to_i64("t.globalTid");
    let sql = format!(
        r#"
        SELECT
            t.start, t."end",
            {global_tid},
            COALESCE(s.value, '<unknown>')
        FROM nsight.{T} t
        LEFT JOIN nsight.StringIds s ON t.nameId = s.id
        WHERE t.rowid = ?
        "#
    );
    let mut stmt = conn.prepare(&sql).context("prepare osrt inspect")?;
    let mut rows = stmt.query([Value::BigInt(id.rowid)])?;
    let Some(r) = rows.next()? else {
        return Ok(None);
    };
    let start_ns: i64 = r.get(0)?;
    let end_ns: i64 = r.get(1)?;
    Ok(Some(EventDetails::Osrt(OsrtDetails {
        key: id.to_string(),
        row_id: id,
        name: r.get(3)?,
        start_ns,
        end_ns,
        duration_ns: end_ns - start_ns,
        global_tid: r.get(2)?,
    })))
}

pub(super) fn query_nvtx(
    conn: &Connection,
    cols: &ColumnMap,
    id: RowId,
    nesting: Option<&veloq_nsys_data::NvtxNesting>,
    tree: Option<&veloq_nsys_data::nvtx_tree::NvtxTree>,
) -> Result<Option<EventDetails>> {
    const T: &str = "NVTX_EVENTS";
    if !cols.contains_key(T) {
        return Ok(None);
    }
    let global_tid = veloq_nsys_data::sql_expr::u64_bits_to_i64("t.globalTid");
    let sql = format!(
        r#"
        SELECT
            t.start, t."end",
            {global_tid},
            CAST(COALESCE(t.domainId, 0) AS BIGINT),
            CAST(t.eventType AS BIGINT),
            COALESCE(t.text, s.value, '<unnamed>')
        FROM nsight.{T} t
        LEFT JOIN nsight.StringIds s ON t.textId = s.id
        WHERE t.rowid = ?
        "#
    );
    let mut stmt = conn.prepare(&sql).context("prepare nvtx inspect")?;
    let mut rows = stmt.query([Value::BigInt(id.rowid)])?;
    let Some(r) = rows.next()? else {
        return Ok(None);
    };
    let start_ns: i64 = r.get(0)?;
    let end_ns: Option<i64> = r.get(1)?;
    let depth = nesting.and_then(|m| m.get(&id.rowid).map(|e| e.depth));
    let tree_record = tree.and_then(|t| t.get(id.rowid));
    let path = tree_record.map(|rec| rec.path.clone());
    let parent_row_id = tree_record.and_then(|rec| {
        rec.parent_range_id
            .map(|rid| RowId::new(crate::EventKind::Nvtx, rid))
    });
    let parent_name = tree_record
        .and_then(|rec| rec.parent_range_id)
        .and_then(|rid| tree.and_then(|t| t.get(rid)))
        .map(|rec| rec.name.clone());
    Ok(Some(EventDetails::Nvtx(NvtxDetails {
        key: id.to_string(),
        row_id: id,
        name: r.get(5)?,
        start_ns,
        end_ns,
        duration_ns: end_ns.map(|e| e - start_ns),
        global_tid: r.get(2)?,
        domain_id: r.get(3)?,
        event_type: r.get(4)?,
        depth,
        path,
        parent_row_id,
        parent_name,
    })))
}
