//! `inspect sync:N` / `cuda_event:N` — synchronisation pair.
//!
//! `cudaEventSynchronize` / `cudaStreamWaitEvent` rows in
//! `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION` consume a placement from
//! `CUPTI_ACTIVITY_KIND_CUDA_EVENT` via `eventSyncId`. Grouping the
//! two kinds keeps that join-key contract visible in one file.

use crate::{NvtxContext, RowId};
use anyhow::{Context, Result};
use duckdb::Connection;
use duckdb::types::Value;
use serde::Serialize;

use super::{ColumnMap, EventDetails, maybe_col};

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SyncDetails {
    /// Cross-trace key — equal to `row_id` stringified.
    pub key: String,
    pub row_id: RowId,
    pub start_ns: i64,
    pub end_ns: i64,
    pub duration_ns: i64,
    pub device_id: i32,
    pub context_id: i64,
    /// Stream id — `0` (and a separate `stream_id_present: false` would
    /// be nice, but agents read the same field across kinds) for
    /// device-level synchronisation (`cudaDeviceSynchronize`).
    pub stream_id: i64,
    /// Raw `syncType` value (REFERENCES `ENUM_CUPTI_SYNC_TYPE`).
    pub sync_type: i64,
    /// Human-readable label derived from `sync_type` via the shared
    /// [`crate::kind_sql::sync_type_label`] table.
    pub sync_type_name: &'static str,
    pub correlation_id: Option<i64>,
    /// Pairs with `cuda_event.event_sync_id` to identify which
    /// `cudaEventRecord` placement this sync consumed. Populated for
    /// `cudaEventSynchronize` / `cudaStreamWaitEvent` syncs;
    /// typically `None` for `cudaStreamSynchronize` /
    /// `cudaDeviceSynchronize` which don't wait on a specific event.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_sync_id: Option<i64>,
    /// Innermost NVTX range that was open on the host thread when the
    /// runtime call corresponding to this sync fired. Populated
    /// automatically when the trace carries the needed tables;
    /// otherwise `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_context: Option<NvtxContext>,
}

/// `cudaEventRecord` activity (`CUPTI_ACTIVITY_KIND_CUDA_EVENT`).
/// One instantaneous row per recorded CUDA Event on a stream.
/// `event_sync_id` is the join key against
/// `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION.eventSyncId` — used to answer
/// "which sync consumed this placement" and the reverse.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct CudaEventDetails {
    /// Cross-trace key — equal to `row_id` stringified.
    pub key: String,
    pub row_id: RowId,
    pub start_ns: i64,
    pub device_id: i32,
    pub context_id: i64,
    pub stream_id: i64,
    /// Stable per-record id (the CUDA Event handle's identity).
    pub event_id: i64,
    /// Globally-unique placement id; pairs with the consuming sync's
    /// `event_sync_id`. `None` on rows that aren't part of a
    /// host-visible synchronisation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_sync_id: Option<i64>,
    /// Shared with the launching `cudaEventRecord` runtime call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<i64>,
}

pub(super) fn query_sync(
    conn: &Connection,
    cols: &ColumnMap,
    id: RowId,
) -> Result<Option<EventDetails>> {
    const T: &str = "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION";
    if !cols.contains_key(T) {
        return Ok(None);
    }
    let corr = maybe_col(cols, T, "correlationId");
    let esync = maybe_col(cols, T, "eventSyncId");
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
            CAST(t.syncType AS BIGINT),
            CAST({corr} AS BIGINT),
            CAST({esync} AS BIGINT)
        FROM nsight.{T} t
        WHERE t.rowid = ?
        "#
    );
    let mut stmt = conn.prepare(&sql).context("prepare sync inspect")?;
    let mut rows = stmt.query([Value::BigInt(id.rowid)])?;
    let Some(r) = rows.next()? else {
        return Ok(None);
    };
    let start_ns: i64 = r.get(0)?;
    let end_ns: i64 = r.get(1)?;
    let sync_type: i64 = r.get(5)?;
    Ok(Some(EventDetails::Sync(SyncDetails {
        key: id.to_string(),
        row_id: id,
        start_ns,
        end_ns,
        duration_ns: end_ns - start_ns,
        device_id: r.get(2)?,
        context_id: r.get(3)?,
        stream_id: r.get(4)?,
        sync_type,
        sync_type_name: crate::kind_sql::sync_type_label(sync_type),
        correlation_id: r.get(6)?,
        event_sync_id: r.get(7)?,
        nvtx_context: None,
    })))
}

pub(super) fn query_cuda_event(
    conn: &Connection,
    cols: &ColumnMap,
    id: RowId,
) -> Result<Option<EventDetails>> {
    const T: &str = "CUPTI_ACTIVITY_KIND_CUDA_EVENT";
    if !cols.contains_key(T) {
        return Ok(None);
    }
    let esync = maybe_col(cols, T, "eventSyncId");
    let corr = maybe_col(cols, T, "correlationId");
    let dev = crate::kind_sql::GPU_DEVICE_ID_EXPR;
    let ctx = crate::kind_sql::GPU_CONTEXT_ID_EXPR;
    let stm = crate::kind_sql::GPU_STREAM_ID_EXPR;
    let sql = format!(
        r#"
        SELECT
            t.timestamp,
            {dev},
            {ctx},
            {stm},
            CAST(t.eventId AS BIGINT),
            CAST({esync} AS BIGINT),
            CAST({corr} AS BIGINT)
        FROM nsight.{T} t
        WHERE t.rowid = ?
        "#
    );
    let mut stmt = conn.prepare(&sql).context("prepare cuda_event inspect")?;
    let mut rows = stmt.query([Value::BigInt(id.rowid)])?;
    let Some(r) = rows.next()? else {
        return Ok(None);
    };
    Ok(Some(EventDetails::CudaEvent(CudaEventDetails {
        key: id.to_string(),
        row_id: id,
        start_ns: r.get(0)?,
        device_id: r.get(1)?,
        context_id: r.get(2)?,
        stream_id: r.get(3)?,
        event_id: r.get(4)?,
        event_sync_id: r.get(5)?,
        correlation_id: r.get(6)?,
    })))
}
