//! Shared NVTX→GPU-event projection CTE templates.
//!
//! Used by [`crate::slices`] instance and aggregate views. The CTE this
//! module builds is **not** the rowid-set filter
//! `nvtx_attribution` builds — that one produces
//! `attributed_<kind>_rowids` for `WHERE rowid IN (...)` use. This
//! one produces the full per-event tuple
//! `(kind, nvtx_rowid, device_id, stream_id, evt_start, evt_end, dur)`
//! that callers UNION ALL into a `gpu_events` set and fold over.
//!
//! The function is a string-template helper. It assumes the
//! surrounding `WITH` chain already defines:
//!
//! - `attributed_runtime(nvtx_rowid, correlationId, native_pid,
//!   device_id, context_id)` — runtime API calls that fired inside a
//!   matched NVTX range, sourced from the sidecar (so the trio
//!   `(device_id, context_id, correlationId)` is already resolved).
//!
//! No `ctx_for_pid` bridge is required at query time — the
//! sidecar carries `device_id` / `context_id` per attributed runtime
//! row, and the GPU JOIN below uses them directly.

use std::path::Path;

/// Quote a filesystem path for splicing into a DuckDB
/// `read_parquet('…')` literal. Doubles any embedded single quote
/// — parquet paths rarely contain them, but we don't want to assume.
pub fn quote_sidecar_path(path: &Path) -> String {
    path.to_string_lossy().replace('\'', "''")
}

/// Build the shared CTE that expands the runtime→NVTX-parent
/// sidecar into one row per `(runtime row, enclosing NVTX range)`
/// pair, via DuckDB's `UNNEST` over the per-runtime `nvtx_rowids` /
/// `nvtx_names` LIST columns. Used by every forward-attribution
/// surface (`nvtx_attribution::build`, `slices`).
///
/// Returns the CTE body — `<alias> AS (...)` — without a leading
/// `WITH` or trailing comma. Callers compose it into their own
/// `WITH` chain.
///
/// The projection is maximal so callers pick only the columns they
/// need; DuckDB's optimizer projection-prunes unused columns from
/// the parquet scan, so the wider shape costs nothing.
///
/// `sidecar_quoted` is the already-quoted SQL string literal content
/// — pass [`quote_sidecar_path`]'s output.
pub fn sidecar_expanded_cte(alias: &str, sidecar_quoted: &str) -> String {
    format!(
        r#"{alias} AS (
            SELECT rt_rowid                AS runtime_rowid,
                   correlation_id          AS correlationId,
                   native_pid,
                   device_id,
                   context_id,
                   UNNEST(nvtx_rowids)     AS nvtx_rowid,
                   UNNEST(nvtx_names)      AS nvtx_name
            FROM read_parquet('{sidecar_quoted}')
        )"#
    )
}

/// Build the per-kind CTE body that joins `attributed_runtime` rows
/// to the matching GPU activity table and emits the per-event tuple.
/// Returns the CTE body without a leading `WITH`/comma so the caller
/// can splice it into a comma-separated list.
///
/// Identical join shape across kernel / memcpy / memset; callers
/// pass the alias (e.g. `"gpu_kernels"`), the wire-level kind label
/// (`"kernel"`), and the backing CUPTI table name.
pub fn gpu_kind_cte(alias: &str, label: &str, table: &str) -> String {
    let dev = crate::kind_sql::GPU_DEVICE_ID_EXPR;
    let stm = crate::kind_sql::GPU_STREAM_ID_EXPR;
    // Attribution: `attributed_runtime` carries `(device_id,
    // context_id, correlationId)` directly from the sidecar, so the
    // GPU JOIN is a single trio match — no `ctx_for_pid` bridge.
    format!(
        r#"{alias} AS (
            SELECT '{label}' AS kind,
                   ar.nvtx_rowid,
                   {dev} AS device_id,
                   {stm} AS stream_id,
                   t.start  AS evt_start,
                   t."end"  AS evt_end,
                   (t."end" - t.start) AS dur
            FROM attributed_runtime ar
            JOIN nsight.{table} t
              ON t.correlationId              = ar.correlationId
             AND CAST(t.deviceId  AS INTEGER) = ar.device_id
             AND CAST(t.contextId AS BIGINT)  = ar.context_id
        )"#
    )
}
