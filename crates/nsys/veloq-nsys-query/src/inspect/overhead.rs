//! `inspect overhead:N` — profiling-overhead spans.
//!
//! Each `CUPTI_ACTIVITY_KIND_OVERHEAD` row records time CUPTI itself
//! spent doing bookkeeping (buffer flushes, instrumentation, driver
//! JIT). The fields are minimal — start/end, thread, overhead type,
//! optional correlation back to the runtime call that triggered it.

use crate::{NsysQueryResult, RowId};
use duckdb::Connection;
use serde::Serialize;

use super::{ColumnMap, EventDetails, map_inspect_read, maybe_col, query_inspect_row};

const INSPECT_OVERHEAD_SQL: &str = "overhead";

/// Profiling overhead (`CUPTI_ACTIVITY_KIND_OVERHEAD`). Each row is a
/// span of trace time spent by CUPTI itself — buffer flushes,
/// instrumentation, driver JIT, etc. Useful trust signal: "how much
/// of the trace's duration is the profiler observing?"
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct OverheadDetails {
    /// Cross-trace key — equal to `row_id` stringified.
    pub key: String,
    pub row_id: RowId,
    pub start_ns: i64,
    pub end_ns: i64,
    pub duration_ns: i64,
    pub global_tid: Option<i64>,
    /// Raw `overheadType` value (`ENUM_CUPTI_OVERHEAD_TYPE` id).
    pub overhead_type: i64,
    /// Snake-case label derived from `overhead_type` via the shared
    /// [`crate::kind_sql::overhead_type_label`] table.
    pub overhead_type_name: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<i64>,
}

pub(super) fn query_overhead(
    conn: &Connection,
    cols: &ColumnMap,
    id: RowId,
) -> NsysQueryResult<Option<EventDetails>> {
    const T: &str = "CUPTI_ACTIVITY_KIND_OVERHEAD";
    if !cols.contains_key(T) {
        return Ok(None);
    }
    let gtid = maybe_col(cols, T, "globalTid");
    let corr = maybe_col(cols, T, "correlationId");
    let gtid_expr = veloq_nsys_data::sql_expr::u64_bits_to_i64(&gtid);
    let sql = format!(
        r#"
        SELECT
            t.start, t."end",
            CAST(t.overheadType AS BIGINT),
            {gtid_expr},
            CAST({corr} AS BIGINT)
        FROM nsight.{T} t
        WHERE t.rowid = ?
        "#
    );
    query_inspect_row(conn, INSPECT_OVERHEAD_SQL, &sql, id, |r| {
        let start_ns: i64 = map_inspect_read(INSPECT_OVERHEAD_SQL, r.get(0))?;
        let end_ns: i64 = map_inspect_read(INSPECT_OVERHEAD_SQL, r.get(1))?;
        let overhead_type: i64 = map_inspect_read(INSPECT_OVERHEAD_SQL, r.get(2))?;
        Ok(EventDetails::Overhead(OverheadDetails {
            key: id.to_string(),
            row_id: id,
            start_ns,
            end_ns,
            duration_ns: end_ns - start_ns,
            global_tid: map_inspect_read(INSPECT_OVERHEAD_SQL, r.get(3))?,
            overhead_type,
            overhead_type_name: crate::kind_sql::overhead_type_label(overhead_type),
            correlation_id: map_inspect_read(INSPECT_OVERHEAD_SQL, r.get(4))?,
        }))
    })
}
