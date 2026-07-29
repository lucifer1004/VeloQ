//! Reverse NVTX attribution — "which NVTX range was this event launched
//! inside?".
//!
//! The forward direction (an NVTX glob → all matching kernels/memcpys/
//! memsets) lives in [`crate::nvtx_attribution`] and powers `--nvtx` on `stats`
//! / `search`. This module's verbs (`inspect`, `search --with-nvtx`,
//! and any future per-event NVTX decoration) ask the inverse: given a
//! single CUPTI activity row (kernel/memcpy/memset/sync/runtime),
//! surface the innermost NVTX range that enclosed it.
//!
//! ## How this is computed
//!
//! Both directions share the
//! [`veloq_nsys_data::runtime_nvtx_parent`] sidecar. The sidecar
//! pre-computes, for every attributed runtime row, the full outer→
//! inner enclosing NVTX chain — a single `O((N_n + N_r) log + N_r ×
//! depth)` walk amortised across every NVTX-bearing verb on the
//! same trace. Reverse attribution reads the in-memory map via
//! [`veloq_nsys_data::runtime_nvtx_parent::build_or_load_index`]
//! (or [`load_if_present`] for cold single-row callers):
//!
//! [`load_if_present`]: veloq_nsys_data::runtime_nvtx_parent::load_if_present
//!
//! - **Runtime kind**: lookup by `rt_rowid` directly.
//! - **Kernel / Memcpy / Memset / Sync**: one small SQL fetches
//!   `(rowid, native_pid, deviceId, contextId, correlationId)` for the batch
//!   and feeds the tuple straight into
//!   `RuntimeNvtxParent::get_by_correlation`. The sidecar's
//!   key is the documented disambiguator
//!   `(native_pid, device_id, context_id, correlation_id)`.
//!
//! The sidecar reduces reverse attribution to one read of the parquet
//! sidecar plus an in-memory hashmap lookup, reusing the attribution
//! chain the sidecar already pre-computed.
//!
//! ## Why we still issue a small SQL for GPU kinds
//!
//! The sidecar is keyed by `(native_pid, device_id, context_id, correlation_id)`
//! while the caller has a CUPTI rowid. We need the rowid's
//! process-aware CUDA tuple to do the lookup — one
//! small `SELECT … WHERE rowid IN (?, ?, …)` against the GPU
//! activity table fetches those three columns for the batch.

use crate::{EventKind, NsysQueryResult, RowId};
use duckdb::types::Value;
use std::collections::HashMap;
use veloq_nsys_data::{NvtxNesting, RuntimeNvtxParent, Trace, runtime_nvtx_parent};

use crate::event_ref::NvtxContext;

/// Tables every reverse-attribution path needs regardless of source.
/// Without `NVTX_EVENTS` there's nothing to attribute to; without
/// `CUPTI_ACTIVITY_KIND_RUNTIME` there are no runtime rows for the
/// sidecar to walk against.
const CORE_PREREQ_TABLES: &[&str] = &["NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_RUNTIME"];

/// Additional table needed only when the reverse lookup target is a
/// GPU-side kind (kernel/memcpy/memset/sync). `correlationId` is only
/// unique within `(deviceId, contextId)`, so we need
/// `TARGET_INFO_CUDA_CONTEXT_INFO` to bridge those to `native_pid`
/// before consulting the sidecar's `by_correlation` map. Runtime
/// reverse lookup uses `rt_rowid` directly and doesn't need the
/// bridge.
const GPU_PREREQ_TABLE: &str = "TARGET_INFO_CUDA_CONTEXT_INFO";

/// True when the trace can support reverse attribution for `source`.
/// Callers (`inspect` always-on, `search --with-nvtx`) use this as a
/// cheap preflight: if false, skip the work and leave every event's
/// `nvtx_context` at `None` — same shape an absent overlap would
/// produce.
///
/// Source-aware so a runtime-only trace (NVTX + RUNTIME but no
/// `TARGET_INFO_CUDA_CONTEXT_INFO`) can still attribute Runtime
/// events even though kernel/memcpy/memset/sync lookups would
/// short-circuit.
pub fn trace_supports_reverse_for(trace: &Trace, source: Source) -> bool {
    if !CORE_PREREQ_TABLES.iter().all(|t| trace.table_exists(t)) {
        return false;
    }
    match source {
        Source::Runtime => true,
        Source::Kernel | Source::Memcpy | Source::Memset | Source::Sync => {
            trace.table_exists(GPU_PREREQ_TABLE)
        }
    }
}

/// Which event table to walk back from. The reverse query is
/// homogeneous for the four "GPU-side" kinds — they all carry
/// `rowid` + `correlationId` + `deviceId`/`contextId`, so a single
/// SQL template parameterised by table name handles all four.
/// Runtime is the odd kind: the event row is *itself* the
/// runtime-API anchor, so its reverse walk skips the SQL entirely
/// and goes straight from runtime row to sidecar entry via
/// `get_by_runtime`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Kernel,
    Memcpy,
    Memset,
    Sync,
    Runtime,
}

impl Source {
    fn try_from_kind(kind: EventKind) -> Option<Self> {
        match kind {
            EventKind::Kernel => Some(Self::Kernel),
            EventKind::Memcpy => Some(Self::Memcpy),
            EventKind::Memset => Some(Self::Memset),
            EventKind::Sync => Some(Self::Sync),
            EventKind::Runtime => Some(Self::Runtime),
            // Osrt has no correlationId and no equivalent
            // direct-attribution model (no first-class lookup
            // table); Nvtx is the source side of attribution and
            // can't attribute to itself. Graph* / CudaEvent /
            // Overhead / CpuSample stay out of the reverse model
            // until they get their own attribution stories.
            _ => None,
        }
    }

    fn table(self) -> &'static str {
        match self {
            Self::Kernel => "CUPTI_ACTIVITY_KIND_KERNEL",
            Self::Memcpy => "CUPTI_ACTIVITY_KIND_MEMCPY",
            Self::Memset => "CUPTI_ACTIVITY_KIND_MEMSET",
            Self::Sync => "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
            Self::Runtime => "CUPTI_ACTIVITY_KIND_RUNTIME",
        }
    }
}

/// Look up the innermost NVTX range covering one event by row_id.
///
/// Returns `Ok(None)` when:
/// - the trace lacks the prerequisite tables for this source
///   (`trace_supports_reverse_for` returned false);
/// - the row_id's kind isn't one we walk reverse for (Osrt/Nvtx/…);
/// - the sidecar has no entry for this rowid's `correlationId` (the
///   runtime call was outside every NVTX range, or the row's kind
///   couldn't be bridged to a runtime row).
///
/// All three cases are agent-observable as "no `nvtx_context` field"
/// in the JSON. The optional return keeps `inspect` linear — every
/// row_id gets exactly one lookup (or none, when the kind doesn't
/// qualify).
pub fn lookup_one(
    trace: &Trace,
    id: RowId,
    nesting: &NvtxNesting,
) -> NsysQueryResult<Option<NvtxContext>> {
    let Some(source) = Source::try_from_kind(id.kind) else {
        return Ok(None);
    };
    if !trace_supports_reverse_for(trace, source) {
        return Ok(None);
    }
    let batch = lookup_batch(trace, source, &[id.rowid], nesting)?;
    Ok(batch.into_iter().next().map(|(_rowid, ctx)| ctx))
}

/// Above this batch size, reverse lookup triggers a sidecar build
/// if one isn't present yet — the per-rowid bespoke SQL fallback
/// would otherwise scale linearly with batch size, while a sidecar
/// build is a constant-cost one-shot (~3.5s on a 2GB trace) that
/// amortises across the rest of the batch and every later NVTX-
/// bearing verb on this trace.
///
/// At or below the threshold (single-row inspect, "show me four
/// random hits"), we keep cold latency low: load the sidecar only
/// if it's already on disk, and fall back to a tight rowid-scoped
/// SQL CTE when not.
const SIDECAR_BUILD_THRESHOLD: usize = 4;

/// Batched reverse attribution for many row_ids of the same kind.
///
/// Used by `search --with-nvtx` after `search` has materialised the
/// row_ids list — we issue at most one small SQL per non-Runtime
/// kind in the batch (so a mixed kernel+memcpy result fans out to
/// at most three SQLs, not three hundred). Runtime kind needs no
/// SQL at all — the sidecar's `by_rt_rowid` map answers directly.
///
/// Returns a map `rowid → NvtxContext`. Rowids with no overlapping
/// NVTX range simply don't appear in the map; callers should treat
/// absence as "no context" (same convention as `lookup_one`'s `None`).
pub fn lookup_batch(
    trace: &Trace,
    source: Source,
    rowids: &[i64],
    nesting: &NvtxNesting,
) -> NsysQueryResult<HashMap<i64, NvtxContext>> {
    if rowids.is_empty() || !trace_supports_reverse_for(trace, source) {
        return Ok(HashMap::new());
    }
    if !trace.table_exists(source.table()) {
        return Ok(HashMap::new());
    }
    // Two cold-cache modes, picked by batch size:
    // - Small batch (`inspect kernel:N`): load sidecar if present,
    //   otherwise run a small rowid-scoped SQL CTE for just these
    //   rowids. No multi-second build cost for one-row decorations.
    // - Large batch (`search --with-nvtx`): build sidecar if absent
    //   (one-time cost), then in-memory lookup amortises across the
    //   rest of the batch and every later NVTX-bearing verb.
    let index = if rowids.len() <= SIDECAR_BUILD_THRESHOLD {
        runtime_nvtx_parent::load_if_present(trace).map_err(|source| {
            crate::NsysQueryError::nvtx_reverse_sidecar_load("load existing", source)
        })?
    } else {
        Some(
            runtime_nvtx_parent::build_or_load_index(trace).map_err(|source| {
                crate::NsysQueryError::nvtx_reverse_sidecar_load("build or load", source)
            })?,
        )
    };

    let Some(index) = index else {
        // Small batch + no sidecar → bespoke per-batch SQL, bounded by
        // the small `rowids.len()`.
        return cold_fallback(trace, source, rowids, nesting);
    };
    if index.is_empty() {
        return Ok(HashMap::new());
    }

    match source {
        Source::Runtime => Ok(lookup_runtime(&index, rowids, nesting)),
        Source::Kernel | Source::Memcpy | Source::Memset | Source::Sync => {
            lookup_gpu_kind(trace, source, rowids, &index, nesting)
        }
    }
}

/// Runtime kind: sidecar's `by_rt_rowid` map is directly keyed by the
/// rowid the caller already has. No SQL round-trip.
fn lookup_runtime(
    index: &RuntimeNvtxParent,
    rowids: &[i64],
    nesting: &NvtxNesting,
) -> HashMap<i64, NvtxContext> {
    let mut out: HashMap<i64, NvtxContext> = HashMap::with_capacity(rowids.len());
    for &rt_rowid in rowids {
        if let Some(entry) = index.get_by_runtime(rt_rowid)
            && let Some(ctx) = innermost_to_nvtx_context(entry, nesting)
        {
            out.insert(rt_rowid, ctx);
        }
    }
    out
}

/// GPU kinds (Kernel/Memcpy/Memset/Sync): one small SQL to map each
/// rowid to `(native_pid, deviceId, contextId, correlationId)`, then
/// in-memory lookup into the sidecar's `by_correlation`
/// map.
fn lookup_gpu_kind(
    trace: &Trace,
    source: Source,
    rowids: &[i64],
    index: &RuntimeNvtxParent,
    nesting: &NvtxNesting,
) -> NsysQueryResult<HashMap<i64, NvtxContext>> {
    let table = source.table();
    // DuckDB's `IN` accepts a parenthesised list of placeholders.
    let placeholders = std::iter::repeat_n("?", rowids.len())
        .collect::<Vec<_>>()
        .join(", ");
    let dev = crate::kind_sql::GPU_DEVICE_ID_EXPR;
    let ctx_expr = crate::kind_sql::GPU_CONTEXT_ID_EXPR;
    let process_join =
        veloq_nsys_data::process_lateral_join_sql(trace, table, "t", "proc", "t.start");
    let sql = format!(
        r#"SELECT t.rowid,
                  proc.process_id AS process_id,
                  {dev}         AS device_id,
                  {ctx_expr}    AS context_id,
                  t.correlationId
           FROM nsight.{table} t
           {process_join}
           WHERE t.rowid IN ({placeholders})
             AND t.correlationId IS NOT NULL"#
    );

    lookup_gpu_kind_with_sql(trace.conn(), &sql, rowids, index, nesting)
}

fn lookup_gpu_kind_with_sql(
    conn: &duckdb::Connection,
    sql: &str,
    rowids: &[i64],
    index: &RuntimeNvtxParent,
    nesting: &NvtxNesting,
) -> NsysQueryResult<HashMap<i64, NvtxContext>> {
    let params: Vec<Value> = rowids.iter().map(|&id| Value::BigInt(id)).collect();
    let rows = crate::query_sql::exec::query_rows(
        conn,
        sql,
        &params,
        crate::query_sql::exec::NVTX_REVERSE_GPU_LOOKUP,
        gpu_lookup_sql_row,
    )?;

    let mut out: HashMap<i64, NvtxContext> = HashMap::with_capacity(rowids.len());
    for row in rows {
        let Some(entry) = index.get_by_correlation(
            row.process_id,
            row.device_id,
            row.context_id,
            row.correlation_id,
        ) else {
            continue;
        };
        if let Some(ctx) = innermost_to_nvtx_context(entry, nesting) {
            out.insert(row.event_rowid, ctx);
        }
    }
    Ok(out)
}

struct GpuLookupSqlRow {
    event_rowid: i64,
    process_id: i64,
    device_id: i32,
    context_id: i64,
    correlation_id: i64,
}

fn gpu_lookup_sql_row(row: &duckdb::Row<'_>) -> Result<GpuLookupSqlRow, duckdb::Error> {
    Ok(GpuLookupSqlRow {
        event_rowid: row.get(0)?,
        process_id: row.get(1)?,
        device_id: row.get(2)?,
        context_id: row.get(3)?,
        correlation_id: row.get(4)?,
    })
}

/// Decorate the innermost enclosing range of `entry` with depth +
/// iter_index from the nesting map. Returns `None` if the entry has
/// no enclosing chain (sidecar invariant — should not happen, but
/// be defensive so we don't panic).
fn innermost_to_nvtx_context(
    entry: &veloq_nsys_data::RuntimeParentEntry,
    nesting: &NvtxNesting,
) -> Option<NvtxContext> {
    let innermost = entry.innermost()?;
    // Depth + iter_index come from the cached nesting map — the
    // sidecar has already chosen the innermost; we just decorate.
    // Absent rows (NVTX range computed before nesting wrote, or
    // out-of-window) fall back to default depth 0 / iter_index 0.
    let n = nesting
        .get(&innermost.nvtx_rowid)
        .copied()
        .unwrap_or_default();
    Some(NvtxContext {
        range_id: RowId::new(EventKind::Nvtx, innermost.nvtx_rowid),
        name: innermost.nvtx_name.clone(),
        depth: n.depth,
        // `0` is a valid iter_index (first occurrence); we surface
        // it always rather than skip-serializing the option, so
        // agents can rely on the field's presence when
        // `nvtx_context` is present.
        iter_index: Some(n.iter_index),
    })
}

/// Cold path: small batch, sidecar not yet built. Run the
/// containment CTE bounded to the requested rowids so we
/// don't pay a multi-second sidecar build for a one-row decoration.
/// Runtime kind needs no `ctx_for_pid` bridge (the event row IS the
/// runtime call); GPU kinds bridge via `(deviceId, contextId) →
/// processId` exactly as the sidecar build does internally.
fn cold_fallback(
    trace: &Trace,
    source: Source,
    rowids: &[i64],
    nesting: &NvtxNesting,
) -> NsysQueryResult<HashMap<i64, NvtxContext>> {
    let table = source.table();
    let placeholders = std::iter::repeat_n("?", rowids.len())
        .collect::<Vec<_>>()
        .join(", ");
    // Both branches end in the same `candidates` CTE that picks the
    // innermost NVTX range per `event_rowid` and a final
    // `SELECT … WHERE rn = 1`. Factor here so a future tie-break
    // change only needs touching once.
    //
    // The ORDER BY mirrors the warm sidecar walk's
    // `(start ASC, end DESC, rowid ASC)` outer→inner sort: `rn = 1`
    // is the range with the latest start, smallest end, largest
    // rowid — i.e. the innermost.
    const CANDIDATES_AND_PICK: &str = r#"
               candidates AS (
                   SELECT er.event_rowid,
                          n.rowid                                AS nvtx_rowid,
                          COALESCE(n.text, s.value, '<unnamed>') AS name,
                          ROW_NUMBER() OVER (
                              PARTITION BY er.event_rowid
                              ORDER BY n.start DESC, n."end" ASC, n.rowid DESC
                          ) AS rn
                   FROM event_runtime er
                   JOIN nsight.NVTX_EVENTS n
                     ON n.globalTid    = er.tid
                    AND n."end" IS NOT NULL
                    AND er.launch_start >= n.start
                    AND er.launch_end   <= n."end"
                   LEFT JOIN nsight.StringIds s ON n.textId = s.id
               )
               SELECT event_rowid, nvtx_rowid, name
               FROM candidates
               WHERE rn = 1"#;
    let sql = match source {
        Source::Runtime => format!(
            r#"WITH event_runtime AS (
                   SELECT t.rowid     AS event_rowid,
                          t.start     AS launch_start,
                          t."end"     AS launch_end,
                          t.globalTid AS tid
                   FROM nsight.{table} t
                   WHERE t.rowid IN ({placeholders})
                     AND t."end" IS NOT NULL
                     AND t.globalTid IS NOT NULL
               ),{CANDIDATES_AND_PICK}"#
        ),
        Source::Kernel | Source::Memcpy | Source::Memset | Source::Sync => {
            let dev = crate::kind_sql::GPU_DEVICE_ID_EXPR;
            let ctx_expr = crate::kind_sql::GPU_CONTEXT_ID_EXPR;
            let process_join =
                veloq_nsys_data::process_lateral_join_sql(trace, table, "t", "proc", "t.start");
            let runtime_pid = veloq_nsys_data::native_pid_sql("r.globalTid");
            format!(
                r#"WITH event_ctx AS (
                       SELECT t.rowid          AS event_rowid,
                              proc.process_id  AS process_id,
                              {dev}            AS device_id,
                              {ctx_expr}       AS context_id,
                              t.correlationId  AS correlationId
                       FROM nsight.{table} t
                       {process_join}
                       WHERE t.rowid IN ({placeholders})
                         AND t.correlationId IS NOT NULL
                   ),
                   event_runtime AS (
                       SELECT ec.event_rowid,
                              r.start                                              AS launch_start,
                              r."end"                                              AS launch_end,
                              r.globalTid                                          AS tid
                       FROM event_ctx ec
                       JOIN nsight.CUPTI_ACTIVITY_KIND_RUNTIME r
                         ON r.correlationId                                  = ec.correlationId
                        AND {runtime_pid} = ec.process_id
                   ),{CANDIDATES_AND_PICK}"#
            )
        }
    };

    cold_fallback_with_sql(trace.conn(), &sql, rowids, nesting)
}

fn cold_fallback_with_sql(
    conn: &duckdb::Connection,
    sql: &str,
    rowids: &[i64],
    nesting: &NvtxNesting,
) -> NsysQueryResult<HashMap<i64, NvtxContext>> {
    let params: Vec<Value> = rowids.iter().map(|&id| Value::BigInt(id)).collect();
    let rows = crate::query_sql::exec::query_rows(
        conn,
        sql,
        &params,
        crate::query_sql::exec::NVTX_REVERSE_COLD_FALLBACK,
        cold_fallback_sql_row,
    )?;

    let mut out: HashMap<i64, NvtxContext> = HashMap::with_capacity(rowids.len());
    for row in rows {
        let nesting_entry = nesting.get(&row.nvtx_rowid).copied().unwrap_or_default();
        out.insert(
            row.event_rowid,
            NvtxContext {
                range_id: RowId::new(EventKind::Nvtx, row.nvtx_rowid),
                name: row.name,
                depth: nesting_entry.depth,
                iter_index: Some(nesting_entry.iter_index),
            },
        );
    }
    Ok(out)
}

struct ColdFallbackSqlRow {
    event_rowid: i64,
    nvtx_rowid: i64,
    name: String,
}

fn cold_fallback_sql_row(row: &duckdb::Row<'_>) -> Result<ColdFallbackSqlRow, duckdb::Error> {
    Ok(ColdFallbackSqlRow {
        event_rowid: row.get(0)?,
        nvtx_rowid: row.get(1)?,
        name: row.get(2)?,
    })
}

/// Group row_ids by their CUPTI kind so a mixed-kind batch fans out to
/// at most one SQL per non-Runtime kind. Row_ids whose kind doesn't
/// qualify for reverse attribution are skipped entirely.
pub fn lookup_for_row_ids(
    trace: &Trace,
    ids: &[RowId],
    nesting: &NvtxNesting,
) -> NsysQueryResult<HashMap<RowId, NvtxContext>> {
    if ids.is_empty() {
        return Ok(HashMap::new());
    }
    // No global preflight — each per-source `lookup_batch` checks
    // its own prereqs. Pre-source-aware preflight, a trace with
    // NVTX + RUNTIME but no `TARGET_INFO_CUDA_CONTEXT_INFO` would
    // have short-circuited here even when the batch contained
    // Runtime rows that don't need the GPU bridge.
    let mut buckets: HashMap<Source, Vec<i64>> = HashMap::new();
    for id in ids {
        if let Some(source) = Source::try_from_kind(id.kind) {
            buckets.entry(source).or_default().push(id.rowid);
        }
    }
    let mut out: HashMap<RowId, NvtxContext> = HashMap::with_capacity(ids.len());
    for (source, rowids) in buckets {
        let batch = lookup_batch(trace, source, &rowids, nesting)?;
        for (rowid, ctx) in batch {
            out.insert(RowId::new(kind_of(source), rowid), ctx);
        }
    }
    Ok(out)
}

fn kind_of(source: Source) -> EventKind {
    match source {
        Source::Kernel => EventKind::Kernel,
        Source::Memcpy => EventKind::Memcpy,
        Source::Memset => EventKind::Memset,
        Source::Sync => EventKind::Sync,
        Source::Runtime => EventKind::Runtime,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use veloq_core::VeloqDiagnostic;

    #[test]
    fn lookup_gpu_kind_prepare_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let index = RuntimeNvtxParent::empty();
        let nesting = NvtxNesting::default();

        let err = match lookup_gpu_kind_with_sql(&conn, "SELECT * FROM", &[1], &index, &nesting) {
            Ok(rows) => anyhow::bail!(
                "malformed reverse-GPU SQL should not hydrate successfully: {} rows",
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
    fn lookup_gpu_kind_query_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let index = RuntimeNvtxParent::empty();
        let nesting = NvtxNesting::default();
        let sql = "SELECT \
                   ? AS event_rowid, \
                   ? AS device_id, \
                   0::BIGINT AS context_id, \
                   0::BIGINT AS correlation_id";

        let err = match lookup_gpu_kind_with_sql(&conn, sql, &[1], &index, &nesting) {
            Ok(rows) => anyhow::bail!(
                "unbound reverse-GPU SQL should not hydrate successfully: {} rows",
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
    fn lookup_gpu_kind_read_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let index = RuntimeNvtxParent::empty();
        let nesting = NvtxNesting::default();
        let sql = "SELECT \
                   'not-a-rowid' AS event_rowid, \
                   0::INTEGER AS device_id, \
                   0::BIGINT AS context_id, \
                   0::BIGINT AS correlation_id \
                   WHERE ? IS NOT NULL";

        let err = match lookup_gpu_kind_with_sql(&conn, sql, &[1], &index, &nesting) {
            Ok(rows) => anyhow::bail!(
                "malformed reverse-GPU row should not hydrate successfully: {} rows",
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
    fn cold_fallback_prepare_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let nesting = NvtxNesting::default();

        let err = match cold_fallback_with_sql(&conn, "SELECT * FROM", &[1], &nesting) {
            Ok(rows) => anyhow::bail!(
                "malformed reverse-cold SQL should not hydrate successfully: {} rows",
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
    fn cold_fallback_query_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let nesting = NvtxNesting::default();
        let sql = "SELECT \
                   ? AS event_rowid, \
                   ? AS nvtx_rowid, \
                   'range' AS name";

        let err = match cold_fallback_with_sql(&conn, sql, &[1], &nesting) {
            Ok(rows) => anyhow::bail!(
                "unbound reverse-cold SQL should not hydrate successfully: {} rows",
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
    fn cold_fallback_read_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let nesting = NvtxNesting::default();
        let sql = "SELECT \
                   'not-a-rowid' AS event_rowid, \
                   1::BIGINT AS nvtx_rowid, \
                   'range' AS name \
                   WHERE ? IS NOT NULL";

        let err = match cold_fallback_with_sql(&conn, sql, &[1], &nesting) {
            Ok(rows) => anyhow::bail!(
                "malformed reverse-cold row should not hydrate successfully: {} rows",
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
