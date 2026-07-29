//! `veloq correlate <row_id>...` — single-event causal-chain reverse lookup.
//!
//! For each input `row_id`, walks the
//! `(process, device, context, correlationId)` identity to find every
//! event causally related — typically: the CPU
//! runtime API call that launched the GPU work, plus the GPU events
//! that resulted. Useful as the "why is this kernel slow / what did
//! this `cudaLaunchKernel` produce" drill-down.
//!
//! Composes with `gaps` (find an idle gap → take `next.row_id` → call
//! `correlate` → see what CPU code was running) and with `slices`
//! (take a slice's high-cost row_id → trace its launcher).

use crate::query_sql::exec::{SqlLabel, query_rows, query_rows_fallible};
use crate::{EventRef, NsysQueryError, NsysQueryResult, RowId, row_id::EventKind};
use duckdb::types::Value;
use serde::Serialize;
use std::borrow::Cow;
use std::path::Path;
use veloq_nsys_data::{
    CorrelatedRowIds, CorrelationIndex, CudaProcessResolver, SyntheticId, Trace,
    native_pid_from_global_tid,
};

const CORRELATE_GPU_INFO_SQL: &str = "GPU correlation info";
const CORRELATE_RUNTIME_INFO_SQL: &str = "runtime correlation info";
const CORRELATE_OVERHEAD_INFO_SQL: &str = "overhead correlation info";
const CORRELATE_COLUMN_SCAN_SQL: &str = "overhead column scan";
const CORRELATE_SUMMARY_SQL: &str = "summary batch";

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct CorrelateResponse {
    /// Rows returned (= input row_ids resolved).
    pub count: usize,
    /// Same as `count` today — every requested row_id yields one
    /// `CorrelateResult` regardless of whether the correlation walk
    /// found anything. Kept for contract uniformity with stats /
    /// search / timeline.
    pub total_matched: usize,
    /// Canonical primary table. One row per requested input row_id;
    /// each carries the correlation chain (cpu/gpu/sync/graph events).
    pub rows: Vec<CorrelateResult>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct CorrelateResult {
    /// Key — equal to `row_id` stringified. Lets agents join
    /// correlate output against `inspect` / `search` rows.
    pub key: String,
    pub row_id: RowId,
    /// `true` if the input row had a `correlationId` and the
    /// (device, context) bridge resolved.
    pub correlation_found: bool,
    /// Opaque process-aware CUDA correlation identity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub synthetic_id: Option<String>,
    /// Owning native PID when correlation identity was resolved.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub process_id: Option<i64>,
    /// Raw correlationId from the NSys row.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<i64>,
    /// Flat event list — every event in the correlation group
    /// across all kinds (cpu / gpu / sync / graph), sorted by
    /// `start_ns`. Agents iterate one list and filter by
    /// `row_id` prefix (`kernel:`, `runtime:`, `sync:`, `graph:`)
    /// or `device_id` / `global_tid` rather than walking four
    /// parallel arrays. The per-kind buckets stay under
    /// [`CorrelateResultAuxiliary`] for callers who want them
    /// pre-split.
    pub events: Vec<EventRef>,
    pub auxiliary: CorrelateResultAuxiliary,
}

/// Per-kind buckets of the correlation chain. Convenient when an
/// agent already knows it only wants the GPU side (or the CPU
/// launcher) and doesn't want to filter the flat `events` list.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct CorrelateResultAuxiliary {
    /// CPU-side runtime API events with this correlationId. Typically
    /// exactly one (the launcher); kept as a list for symmetry.
    pub cpu_events: Vec<EventRef>,
    /// GPU-side events (kernel/memcpy/memset) with this correlationId.
    /// The input row, if GPU, appears here.
    pub gpu_events: Vec<EventRef>,
    /// Synchronisation events (cudaStreamSynchronize, etc.) with this
    /// correlationId. Typically empty for compute correlations and
    /// non-empty when the input is a sync row; the latter case answers
    /// "what was this sync waiting on" by looking at the matching
    /// `gpu_events` and `cpu_events` on the same correlation group.
    pub sync_events: Vec<EventRef>,
    /// CUDA-graph launches (`CUPTI_ACTIVITY_KIND_GRAPH_TRACE`) with this
    /// correlationId. Non-empty when the input is a graph row or a
    /// `cudaGraphLaunch` runtime call — answers "what graph did this
    /// launch execute" and "how long did the captured graph take."
    pub graph_events: Vec<EventRef>,
}

pub fn run<P: AsRef<Path>>(path: P, row_ids: &[RowId]) -> NsysQueryResult<CorrelateResponse> {
    let trace = Trace::open(path).map_err(NsysQueryError::trace_open)?;
    let index = trace
        .correlation_index()
        .map_err(NsysQueryError::correlation_index_load)?;
    // Load the column-presence map once and reuse across every
    // correlate_one call. fetch_summaries reads it to decide
    // `maybe_col` for the per-kind headline extras (kernel mangled
    // name, memset value, etc.) so correlate.events[] carries the
    // same EventRef shape as search.rows[].
    let cols = crate::column_map::load_standard(trace.conn())?;
    let process_resolver =
        CudaProcessResolver::build(&trace).map_err(NsysQueryError::correlation_index_load)?;

    let mut results = Vec::with_capacity(row_ids.len());
    for id in row_ids {
        results.push(correlate_one(
            &trace,
            &index,
            &process_resolver,
            &cols,
            *id,
        )?);
    }
    let count = results.len();
    Ok(CorrelateResponse {
        count,
        total_matched: count,
        rows: results,
    })
}

fn correlate_one(
    trace: &Trace,
    index: &CorrelationIndex,
    process_resolver: &CudaProcessResolver,
    cols: &crate::column_map::ColumnMap,
    id: RowId,
) -> NsysQueryResult<CorrelateResult> {
    // Step 1: extract the input row's correlation triple.
    let Some(info) = fetch_corr_info(trace, process_resolver, cols, id)? else {
        return Ok(CorrelateResult {
            key: id.to_string(),
            row_id: id,
            correlation_found: false,
            synthetic_id: None,
            process_id: None,
            correlation_id: None,
            events: Vec::new(),
            auxiliary: CorrelateResultAuxiliary {
                cpu_events: Vec::new(),
                gpu_events: Vec::new(),
                sync_events: Vec::new(),
                graph_events: Vec::new(),
            },
        });
    };

    // Step 2: index lookup. GPU rows borrow directly; runtime rows
    // route through `lookup_by_runtime` which returns a `Cow` so the
    // single-context fast path skips the allocation.
    let (group, syn_id): (Cow<'_, CorrelatedRowIds>, SyntheticId) = match info.bridge {
        Bridge::DevCtx {
            process,
            device,
            context,
        } => {
            let Some(g) = index.lookup(process, device, context, info.raw_corr) else {
                return Ok(not_found(id, info.raw_corr));
            };
            (
                Cow::Borrowed(g),
                SyntheticId::pack(process, device, context, info.raw_corr),
            )
        }
        Bridge::RuntimeTid { global_tid } => {
            let Some(merged) = index.lookup_by_runtime(global_tid, info.raw_corr) else {
                return Ok(not_found(id, info.raw_corr));
            };
            // Pick the (dev, ctx) candidate that actually had a non-empty
            // GPU bucket — that's the one the runtime call really targeted.
            // Fall back to the first context, or pid-stuffed fallback.
            let pid = native_pid_from_global_tid(global_tid) as u64;
            let candidates = index.contexts_for_pid(pid);
            let resolved = candidates
                .iter()
                .find(|&&(d, c)| {
                    index
                        .lookup(pid, d, c, info.raw_corr)
                        .map(|g| {
                            !(g.kernel.is_empty()
                                && g.memcpy.is_empty()
                                && g.memset.is_empty()
                                && g.sync.is_empty()
                                && g.graph.is_empty())
                        })
                        .unwrap_or(false)
                })
                .copied()
                .or_else(|| candidates.first().copied())
                .unwrap_or((0, pid));
            (
                merged,
                SyntheticId::pack(pid, resolved.0, resolved.1, info.raw_corr),
            )
        }
    };

    // Step 3: hydrate rowids into EventSummaries — one batched query
    // per kind, not one per rowid. A single CUDA Graph instantiation
    // can put 1000+ kernels under the same correlationId, so a
    // per-rowid query would issue 1000+ prepare+query cycles for that
    // case alone.
    let mut gpu_events = fetch_summaries(trace, cols, EventKind::Kernel, &group.kernel)?;
    gpu_events.extend(fetch_summaries(
        trace,
        cols,
        EventKind::Memcpy,
        &group.memcpy,
    )?);
    gpu_events.extend(fetch_summaries(
        trace,
        cols,
        EventKind::Memset,
        &group.memset,
    )?);
    let mut cpu_events = fetch_summaries(trace, cols, EventKind::Runtime, &group.runtime)?;
    let mut sync_events = fetch_summaries(trace, cols, EventKind::Sync, &group.sync)?;
    let mut graph_events = fetch_summaries(trace, cols, EventKind::Graph, &group.graph)?;
    let process_id = Some(syn_id.process() as i64);
    for event in cpu_events
        .iter_mut()
        .chain(gpu_events.iter_mut())
        .chain(sync_events.iter_mut())
        .chain(graph_events.iter_mut())
    {
        event.base_mut().process_id = process_id;
    }

    // Sort each side by start time so output reads chronologically.
    cpu_events.sort_by_key(|e| (e.base().start_ns, e.base().row_id.rowid));
    gpu_events.sort_by_key(|e| (e.base().start_ns, e.base().row_id.rowid));
    sync_events.sort_by_key(|e| (e.base().start_ns, e.base().row_id.rowid));
    graph_events.sort_by_key(|e| (e.base().start_ns, e.base().row_id.rowid));

    // Flat events list: stable sort-merge of the four per-kind
    // buckets by `start_ns`, ties broken by row_id_rowid for
    // determinism. Agents iterating `events` see the correlation
    // chain in chronological order across all kinds without having
    // to merge themselves.
    let mut events: Vec<EventRef> = cpu_events
        .iter()
        .chain(gpu_events.iter())
        .chain(sync_events.iter())
        .chain(graph_events.iter())
        .cloned()
        .collect();
    events.sort_by_key(|e| (e.base().start_ns, e.base().row_id.rowid));

    Ok(CorrelateResult {
        key: id.to_string(),
        row_id: id,
        correlation_found: true,
        synthetic_id: Some(syn_id.to_string()),
        process_id: Some(syn_id.process() as i64),
        correlation_id: Some(info.raw_corr as i64),
        events,
        auxiliary: CorrelateResultAuxiliary {
            cpu_events,
            gpu_events,
            sync_events,
            graph_events,
        },
    })
}

// ---- Step 1 helpers -------------------------------------------------------

enum Bridge {
    /// Process-aware CUDA identity resolved from the GPU event.
    DevCtx {
        process: u64,
        device: u64,
        context: u64,
    },
    /// Only `globalTid` available — runtime API rows. Resolved through
    /// TARGET_INFO_CUDA_CONTEXT_INFO.
    RuntimeTid { global_tid: i64 },
}

struct CorrInfo {
    raw_corr: u64,
    bridge: Bridge,
}

fn fetch_corr_info(
    trace: &Trace,
    process_resolver: &CudaProcessResolver,
    cols: &crate::column_map::ColumnMap,
    id: RowId,
) -> NsysQueryResult<Option<CorrInfo>> {
    // Dispatch on kind once; per-shape helpers exhaustively handle
    // their column projections without a second `match id.kind` that
    // would have to add an `unreachable!` arm.
    //
    // Sync rows carry native (deviceId, contextId, correlationId), same
    // shape as kernel/memcpy/memset — they go through the GPU bridge.
    //
    // Overhead rows record host-thread profiler bookkeeping; the
    // correlationId points at the runtime call they instrumented but
    // the table may not carry deviceId/contextId. Route through the
    // runtime-style globalTid bridge instead.
    match id.kind {
        EventKind::Kernel
        | EventKind::Memcpy
        | EventKind::Memset
        | EventKind::Sync
        | EventKind::Graph
        | EventKind::CudaEvent => fetch_gpu_corr_info(trace, process_resolver, cols, id),
        EventKind::Runtime => fetch_runtime_corr_info(trace, id),
        EventKind::Overhead => fetch_overhead_corr_info(trace, id),
        // No correlationId on these kinds — short-circuit.
        // CpuSample lives in COMPOSITE_EVENTS; nsys doesn't link it to
        // the CUPTI correlation graph, so there's no GPU work to walk to.
        EventKind::Osrt
        | EventKind::Nvtx
        | EventKind::GraphNode
        | EventKind::GraphEvent
        | EventKind::CpuSample => Ok(None),
    }
}

fn fetch_gpu_corr_info(
    trace: &Trace,
    process_resolver: &CudaProcessResolver,
    cols: &crate::column_map::ColumnMap,
    id: RowId,
) -> NsysQueryResult<Option<CorrInfo>> {
    let table = id.kind.table();
    let global_pid = crate::column_map::maybe_col(cols, table, "globalPid");
    let start_expr = if id.kind == EventKind::CudaEvent {
        "CAST(t.timestamp AS BIGINT)"
    } else {
        "CAST(t.start AS BIGINT)"
    };
    let sql = format!(
        "SELECT t.correlationId, \
                CAST(t.deviceId  AS BIGINT), \
                CAST(t.contextId AS BIGINT), \
                {start_expr}, \
                CAST({global_pid} AS BIGINT) \
         FROM nsight.{table} t WHERE t.rowid = ?"
    );
    fetch_gpu_corr_info_with_sql(trace.conn(), process_resolver, id, &sql)
}

fn fetch_gpu_corr_info_with_sql(
    conn: &duckdb::Connection,
    process_resolver: &CudaProcessResolver,
    id: RowId,
    sql: &str,
) -> NsysQueryResult<Option<CorrInfo>> {
    let params = [Value::BigInt(id.rowid)];
    let rows = query_rows(
        conn,
        sql,
        &params,
        SqlLabel::new("correlate", CORRELATE_GPU_INFO_SQL),
        gpu_corr_info_row,
    )?;
    let Some(row) = rows.into_iter().next() else {
        return Ok(None);
    };
    let Some(corr) = row.correlation_id else {
        return Ok(None);
    };
    Ok(Some(CorrInfo {
        raw_corr: corr as u64,
        bridge: Bridge::DevCtx {
            process: process_resolver
                .resolve_required(
                    id.kind.table(),
                    row.device_id as i32,
                    row.context_id,
                    Some(corr),
                    row.start_ns,
                    row.global_pid,
                )
                .map_err(NsysQueryError::data)? as u64,
            device: row.device_id as u64,
            context: row.context_id as u64,
        },
    }))
}

struct GpuCorrInfoRow {
    correlation_id: Option<i64>,
    device_id: i64,
    context_id: i64,
    start_ns: i64,
    global_pid: Option<i64>,
}

fn gpu_corr_info_row(row: &duckdb::Row<'_>) -> Result<GpuCorrInfoRow, duckdb::Error> {
    Ok(GpuCorrInfoRow {
        correlation_id: row.get(0)?,
        device_id: row.get(1)?,
        context_id: row.get(2)?,
        start_ns: row.get(3)?,
        global_pid: row.get(4)?,
    })
}

/// Overhead rows ride the runtime-style bridge: their `correlationId`
/// points at the runtime call that triggered the bookkeeping work,
/// and `globalTid` anchors the host thread. The table can lack
/// `deviceId`/`contextId` (and occasionally `globalTid`) on older
/// schemas, so we probe the columns first and short-circuit to
/// "uncorrelated" rather than blowing up at prepare time. Mirrors
/// what `inspect`'s overhead handler already does with `maybe_col`.
fn fetch_overhead_corr_info(trace: &Trace, id: RowId) -> NsysQueryResult<Option<CorrInfo>> {
    const T: &str = "CUPTI_ACTIVITY_KIND_OVERHEAD";
    let cols = nsight_columns(trace, T)?;
    if !cols.contains("correlationId") || !cols.contains("globalTid") {
        return Ok(None);
    }
    let global_tid = veloq_nsys_data::sql_expr::u64_bits_to_i64("t.globalTid");
    let sql = format!(
        "SELECT t.correlationId, {global_tid} \
         FROM nsight.{T} t WHERE t.rowid = ?"
    );
    fetch_overhead_corr_info_with_sql(trace.conn(), id, &sql)
}

fn fetch_overhead_corr_info_with_sql(
    conn: &duckdb::Connection,
    id: RowId,
    sql: &str,
) -> NsysQueryResult<Option<CorrInfo>> {
    let params = [Value::BigInt(id.rowid)];
    let rows = query_rows(
        conn,
        sql,
        &params,
        SqlLabel::new("correlate", CORRELATE_OVERHEAD_INFO_SQL),
        overhead_corr_info_row,
    )?;
    let Some(row) = rows.into_iter().next() else {
        return Ok(None);
    };
    let (Some(corr), Some(global_tid)) = (row.correlation_id, row.global_tid) else {
        return Ok(None);
    };
    Ok(Some(CorrInfo {
        raw_corr: corr as u64,
        bridge: Bridge::RuntimeTid { global_tid },
    }))
}

struct OverheadCorrInfoRow {
    correlation_id: Option<i64>,
    global_tid: Option<i64>,
}

fn overhead_corr_info_row(row: &duckdb::Row<'_>) -> Result<OverheadCorrInfoRow, duckdb::Error> {
    Ok(OverheadCorrInfoRow {
        correlation_id: row.get(0)?,
        global_tid: row.get(1)?,
    })
}

/// One-shot column probe for an `nsight.<table>`. Returns the set of
/// column names actually present, so callers can decide whether to
/// project a column or fall back. Used by the overhead correlate path
/// to avoid baking a `deviceId`/`contextId` assumption into the SQL.
///
/// Query by `table_schema = 'nsight'`, not `table_catalog`: see
/// [`crate::column_map::load_columns`] for the shared rationale —
/// `table_catalog` returns an empty set, silently downgrading every
/// overhead correlation to "uncorrelated".
fn nsight_columns(
    trace: &Trace,
    table: &str,
) -> NsysQueryResult<std::collections::HashSet<String>> {
    let sql = "SELECT column_name FROM information_schema.columns \
               WHERE table_schema = 'nsight' AND table_name = ?";
    nsight_columns_with_sql(trace.conn(), table, sql)
}

fn nsight_columns_with_sql(
    conn: &duckdb::Connection,
    table: &str,
    sql: &str,
) -> NsysQueryResult<std::collections::HashSet<String>> {
    let params = [Value::Text(table.to_string())];
    let rows = query_rows(
        conn,
        sql,
        &params,
        SqlLabel::new("correlate", CORRELATE_COLUMN_SCAN_SQL),
        column_name_row,
    )?;
    Ok(rows.into_iter().collect())
}

fn column_name_row(row: &duckdb::Row<'_>) -> Result<String, duckdb::Error> {
    row.get(0)
}

fn fetch_runtime_corr_info(trace: &Trace, id: RowId) -> NsysQueryResult<Option<CorrInfo>> {
    let global_tid = veloq_nsys_data::sql_expr::u64_bits_to_i64("t.globalTid");
    let sql = format!(
        "SELECT t.correlationId, {global_tid} \
         FROM nsight.CUPTI_ACTIVITY_KIND_RUNTIME t WHERE t.rowid = ?"
    );
    fetch_runtime_corr_info_with_sql(trace.conn(), id, &sql)
}

fn fetch_runtime_corr_info_with_sql(
    conn: &duckdb::Connection,
    id: RowId,
    sql: &str,
) -> NsysQueryResult<Option<CorrInfo>> {
    let params = [Value::BigInt(id.rowid)];
    let rows = query_rows(
        conn,
        sql,
        &params,
        SqlLabel::new("correlate", CORRELATE_RUNTIME_INFO_SQL),
        runtime_corr_info_row,
    )?;
    let Some(row) = rows.into_iter().next() else {
        return Ok(None);
    };
    let Some(corr) = row.correlation_id else {
        return Ok(None);
    };
    Ok(Some(CorrInfo {
        raw_corr: corr as u64,
        bridge: Bridge::RuntimeTid {
            global_tid: row.global_tid,
        },
    }))
}

struct RuntimeCorrInfoRow {
    correlation_id: Option<i64>,
    global_tid: i64,
}

fn runtime_corr_info_row(row: &duckdb::Row<'_>) -> Result<RuntimeCorrInfoRow, duckdb::Error> {
    Ok(RuntimeCorrInfoRow {
        correlation_id: row.get(0)?,
        global_tid: row.get(1)?,
    })
}

fn not_found(id: RowId, raw_corr: u64) -> CorrelateResult {
    CorrelateResult {
        key: id.to_string(),
        row_id: id,
        correlation_found: false,
        synthetic_id: None,
        process_id: None,
        correlation_id: Some(raw_corr as i64),
        events: Vec::new(),
        auxiliary: CorrelateResultAuxiliary {
            cpu_events: Vec::new(),
            gpu_events: Vec::new(),
            sync_events: Vec::new(),
            graph_events: Vec::new(),
        },
    }
}

// ---- Step 3: batched-by-kind hydration -----------------------------------

/// Chunk size for the `WHERE rowid IN (?, ?, ...)` placeholder list.
/// DuckDB accepts thousands of bind parameters per prepare, but chunking
/// keeps single-statement SQL strings bounded for memory and prepare
/// latency, and side-steps any per-driver placeholder ceiling we might
/// hit later.
const ROWID_BATCH: usize = 1024;

/// Hydrate every rowid for `kind` into an [`EventRef`] in O(ceil(N /
/// ROWID_BATCH)) round-trips. Empty input short-circuits.
///
/// Batching keeps correlate on a CUDA-graph correlation group (1000+
/// kernels sharing one `correlationId`) from issuing 1000+ DuckDB
/// prepares.
fn fetch_summaries(
    trace: &Trace,
    cols: &crate::column_map::ColumnMap,
    kind: EventKind,
    rowids: &[i64],
) -> NsysQueryResult<Vec<EventRef>> {
    if rowids.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(rowids.len());
    for chunk in rowids.chunks(ROWID_BATCH) {
        let sql = build_summary_sql(kind, cols, chunk.len())?;
        out.extend(hydrate_summary_batch(trace.conn(), &sql, kind, chunk)?);
    }
    Ok(out)
}

fn hydrate_summary_batch(
    conn: &duckdb::Connection,
    sql: &str,
    kind: EventKind,
    rowids: &[i64],
) -> NsysQueryResult<Vec<EventRef>> {
    let params: Vec<Value> = rowids.iter().map(|r| Value::BigInt(*r)).collect();
    query_rows_fallible(
        conn,
        sql,
        &params,
        SqlLabel::new("correlate", CORRELATE_SUMMARY_SQL),
        |row| hydrate_summary_row(row, kind),
    )
}

fn hydrate_summary_row(r: &duckdb::Row<'_>, kind: EventKind) -> NsysQueryResult<EventRef> {
    let rowid: i64 = r.get(0).map_err(|source| {
        crate::NsysQueryError::sql_read("correlate", CORRELATE_SUMMARY_SQL, source)
    })?;
    let row_id = RowId::new(kind, rowid);
    let base = crate::event_ref::EventRefBase {
        key: row_id.to_string(),
        row_id,
        name: r.get(1).map_err(|source| {
            crate::NsysQueryError::sql_read("correlate", CORRELATE_SUMMARY_SQL, source)
        })?,
        start_ns: r.get(2).map_err(|source| {
            crate::NsysQueryError::sql_read("correlate", CORRELATE_SUMMARY_SQL, source)
        })?,
        duration_ns: r.get(3).map_err(|source| {
            crate::NsysQueryError::sql_read("correlate", CORRELATE_SUMMARY_SQL, source)
        })?,
        process_id: None,
        device_id: r.get(4).map_err(|source| {
            crate::NsysQueryError::sql_read("correlate", CORRELATE_SUMMARY_SQL, source)
        })?,
        stream_id: r.get(5).map_err(|source| {
            crate::NsysQueryError::sql_read("correlate", CORRELATE_SUMMARY_SQL, source)
        })?,
        global_tid: r.get(6).map_err(|source| {
            crate::NsysQueryError::sql_read("correlate", CORRELATE_SUMMARY_SQL, source)
        })?,
        // correlate doesn't compute NVTX nesting for related events.
        depth: None,
        nvtx_context: None,
    };
    // Per-kind headline projection — same EventRef shape as
    // `search.rows[]` so an agent walking
    // `correlate.events[] | select(.type=="kernel") | .grid`
    // reads the same fields it would from search. Columns 7+ are the
    // per-kind extras; the SQL builder above matches the positions used here.
    let event_ref = match kind {
        EventKind::Kernel => {
            let grid_x: Option<i64> = r.get(7).map_err(|source| {
                crate::NsysQueryError::sql_read("correlate", CORRELATE_SUMMARY_SQL, source)
            })?;
            let grid_y: Option<i64> = r.get(8).map_err(|source| {
                crate::NsysQueryError::sql_read("correlate", CORRELATE_SUMMARY_SQL, source)
            })?;
            let grid_z: Option<i64> = r.get(9).map_err(|source| {
                crate::NsysQueryError::sql_read("correlate", CORRELATE_SUMMARY_SQL, source)
            })?;
            let block_x: Option<i64> = r.get(10).map_err(|source| {
                crate::NsysQueryError::sql_read("correlate", CORRELATE_SUMMARY_SQL, source)
            })?;
            let block_y: Option<i64> = r.get(11).map_err(|source| {
                crate::NsysQueryError::sql_read("correlate", CORRELATE_SUMMARY_SQL, source)
            })?;
            let block_z: Option<i64> = r.get(12).map_err(|source| {
                crate::NsysQueryError::sql_read("correlate", CORRELATE_SUMMARY_SQL, source)
            })?;
            let grid = match (grid_x, grid_y, grid_z) {
                (Some(x), Some(y), Some(z)) => Some([x, y, z]),
                _ => None,
            };
            let block = match (block_x, block_y, block_z) {
                (Some(x), Some(y), Some(z)) => Some([x, y, z]),
                _ => None,
            };
            let registers_per_thread = r.get(13).map_err(|source| {
                crate::NsysQueryError::sql_read("correlate", CORRELATE_SUMMARY_SQL, source)
            })?;
            let static_shared_memory = r.get(14).map_err(|source| {
                crate::NsysQueryError::sql_read("correlate", CORRELATE_SUMMARY_SQL, source)
            })?;
            let dynamic_shared_memory = r.get(15).map_err(|source| {
                crate::NsysQueryError::sql_read("correlate", CORRELATE_SUMMARY_SQL, source)
            })?;
            let demangled_name = crate::column_map::opt_string(r, 16).map_err(|source| {
                crate::NsysQueryError::sql_read("correlate", CORRELATE_SUMMARY_SQL, source)
            })?;
            let mangled_name = crate::column_map::opt_string(r, 17).map_err(|source| {
                crate::NsysQueryError::sql_read("correlate", CORRELATE_SUMMARY_SQL, source)
            })?;
            EventRef::Kernel(crate::event_ref::EventRefKernel {
                base,
                grid,
                block,
                registers_per_thread,
                static_shared_memory,
                dynamic_shared_memory,
                demangled_name,
                mangled_name,
            })
        }
        EventKind::Memcpy => {
            let bytes: Option<i64> = r.get(7).map_err(|source| {
                crate::NsysQueryError::sql_read("correlate", CORRELATE_SUMMARY_SQL, source)
            })?;
            let copy_kind: Option<i64> = r.get(8).map_err(|source| {
                crate::NsysQueryError::sql_read("correlate", CORRELATE_SUMMARY_SQL, source)
            })?;
            let copy_kind_name = copy_kind.map(crate::kind_sql::copy_kind_label);
            EventRef::Memcpy(crate::event_ref::EventRefMemcpy {
                base,
                bytes,
                copy_kind,
                copy_kind_name,
            })
        }
        EventKind::Memset => {
            let bytes: Option<i64> = r.get(7).map_err(|source| {
                crate::NsysQueryError::sql_read("correlate", CORRELATE_SUMMARY_SQL, source)
            })?;
            let value: Option<i64> = r.get(8).map_err(|source| {
                crate::NsysQueryError::sql_read("correlate", CORRELATE_SUMMARY_SQL, source)
            })?;
            EventRef::Memset(crate::event_ref::EventRefMemset { base, bytes, value })
        }
        // Other kinds correlate hydrates (Sync / Graph /
        // CudaEvent / Overhead / Runtime) carry only the base —
        // they're either GPU-side kinds without per-kind headlines
        // in this WI's scope or CPU-side host events.
        _ => EventRef::from_base(kind, base)?,
    };
    Ok(event_ref)
}

/// Per-kind SELECT used by `fetch_summaries`. The leading column is
/// always `rowid` so the Rust side can reconstruct the wire-format
/// `RowId`. Caller binds `n_placeholders` `i64`s for the `IN (...)`
/// list (one per rowid in the current chunk).
///
/// Returns an error for `Osrt`/`Nvtx`/`CpuSample`: correlate never
/// calls into these kinds (the index has no buckets for them), but
/// the workspace's no-panic policy means precondition violations
/// route through `Result` rather than `unreachable!`.
///
/// `cols` is consulted for kernel `maybe_col` probes (registers /
/// shared memory / mangledName) so older NSys schemas degrade
/// gracefully — same pattern used by `search::per_kind_select`.
fn build_summary_sql(
    kind: EventKind,
    cols: &crate::column_map::ColumnMap,
    n_placeholders: usize,
) -> NsysQueryResult<String> {
    debug_assert!(n_placeholders > 0);
    let placeholders = std::iter::repeat_n("?", n_placeholders)
        .collect::<Vec<_>>()
        .join(", ");
    let global_tid = veloq_nsys_data::sql_expr::u64_bits_to_i64("t.globalTid");
    match kind {
        EventKind::Kernel => {
            // Kernel — projects 18 columns: base 7 (rowid, name,
            // start, duration, dev, stream, NULL tid) + grid x/y/z
            // + block x/y/z + registers + static_shared +
            // dynamic_shared + demangled + mangled. Matches the
            // per-kind headline payload `search.rows[]` emits.
            const T: &str = "CUPTI_ACTIVITY_KIND_KERNEL";
            let name_expr = crate::kind_sql::display_name_expr(kind);
            let joins = crate::kind_sql::name_joins(kind);
            let dev = crate::kind_sql::GPU_DEVICE_ID_EXPR;
            let stm = crate::kind_sql::GPU_STREAM_ID_EXPR;
            let reg = crate::column_map::maybe_col(cols, T, "registersPerThread");
            let smem_static = crate::column_map::maybe_col(cols, T, "staticSharedMemory");
            let smem_dyn = crate::column_map::maybe_col(cols, T, "dynamicSharedMemory");
            let mangled_col = crate::column_map::maybe_col(cols, T, "mangledName");
            Ok(format!(
                "SELECT t.rowid, {name_expr}, \
                        t.start, t.\"end\" - t.start, \
                        {dev}, {stm}, CAST(NULL AS BIGINT), \
                        CAST(t.gridX AS BIGINT), CAST(t.gridY AS BIGINT), CAST(t.gridZ AS BIGINT), \
                        CAST(t.blockX AS BIGINT), CAST(t.blockY AS BIGINT), CAST(t.blockZ AS BIGINT), \
                        CAST({reg} AS BIGINT), \
                        CAST({smem_static} AS BIGINT), \
                        CAST({smem_dyn} AS BIGINT), \
                        s_dem.value, \
                        s_mng.value \
                 FROM nsight.{T} t {joins} \
                 LEFT JOIN nsight.StringIds s_mng ON s_mng.id = {mangled_col} \
                 WHERE t.rowid IN ({placeholders})"
            ))
        }
        EventKind::Memcpy => {
            // Memcpy — projects base 7 + (bytes, copyKind). 9 cols.
            let name_expr = crate::kind_sql::display_name_expr(kind);
            let joins = crate::kind_sql::name_joins(kind);
            let table = kind.table();
            let dev = crate::kind_sql::GPU_DEVICE_ID_EXPR;
            let stm = crate::kind_sql::GPU_STREAM_ID_EXPR;
            Ok(format!(
                "SELECT t.rowid, {name_expr}, \
                        t.start, t.\"end\" - t.start, \
                        {dev}, {stm}, CAST(NULL AS BIGINT), \
                        CAST(t.bytes AS BIGINT), \
                        CAST(t.copyKind AS BIGINT) \
                 FROM nsight.{table} t {joins} \
                 WHERE t.rowid IN ({placeholders})"
            ))
        }
        EventKind::Memset => {
            // Memset — projects base 7 + (bytes, value via
            // maybe_col). 9 cols. `value` is optional on older
            // NSys schemas.
            const T: &str = "CUPTI_ACTIVITY_KIND_MEMSET";
            let name_expr = crate::kind_sql::display_name_expr(kind);
            let joins = crate::kind_sql::name_joins(kind);
            let dev = crate::kind_sql::GPU_DEVICE_ID_EXPR;
            let stm = crate::kind_sql::GPU_STREAM_ID_EXPR;
            let val = crate::column_map::maybe_col(cols, T, "value");
            Ok(format!(
                "SELECT t.rowid, {name_expr}, \
                        t.start, t.\"end\" - t.start, \
                        {dev}, {stm}, CAST(NULL AS BIGINT), \
                        CAST(t.bytes AS BIGINT), \
                        CAST({val} AS BIGINT) \
                 FROM nsight.{T} t {joins} \
                 WHERE t.rowid IN ({placeholders})"
            ))
        }
        EventKind::Sync | EventKind::Graph => {
            // Sync / Graph — base 7 columns; correlate doesn't add
            // per-kind headlines for these.
            let name_expr = crate::kind_sql::display_name_expr(kind);
            let joins = crate::kind_sql::name_joins(kind);
            let table = kind.table();
            let dev = crate::kind_sql::GPU_DEVICE_ID_EXPR;
            let stm = crate::kind_sql::GPU_STREAM_ID_EXPR;
            Ok(format!(
                "SELECT t.rowid, {name_expr}, \
                        t.start, t.\"end\" - t.start, \
                        {dev}, {stm}, CAST(NULL AS BIGINT) \
                 FROM nsight.{table} t {joins} \
                 WHERE t.rowid IN ({placeholders})"
            ))
        }
        EventKind::CudaEvent => {
            // Instantaneous (no end column): project duration as 0,
            // use `t.timestamp` as the start. Name = cuda_event:<eventId>.
            let name_expr = crate::kind_sql::display_name_expr(kind);
            let table = kind.table();
            let dev = crate::kind_sql::GPU_DEVICE_ID_EXPR;
            let stm = crate::kind_sql::GPU_STREAM_ID_EXPR;
            Ok(format!(
                "SELECT t.rowid, {name_expr}, \
                        t.timestamp, 0, \
                        {dev}, \
                        {stm}, \
                        CAST(NULL AS BIGINT) \
                 FROM nsight.{table} t \
                 WHERE t.rowid IN ({placeholders})"
            ))
        }
        EventKind::Overhead => {
            // Has start/end + globalTid but NO deviceId/streamId.
            let name_expr = crate::kind_sql::display_name_expr(kind);
            let table = kind.table();
            Ok(format!(
                "SELECT t.rowid, {name_expr}, \
                        t.start, t.\"end\" - t.start, \
                        CAST(NULL AS INTEGER), \
                        CAST(NULL AS BIGINT), \
                        {global_tid} \
                 FROM nsight.{table} t \
                 WHERE t.rowid IN ({placeholders})"
            ))
        }
        EventKind::Runtime => Ok(format!(
            "SELECT t.rowid, \
                    COALESCE(s.value, '<unknown runtime>'), \
                    t.start, t.\"end\" - t.start, \
                    CAST(NULL AS INTEGER), \
                    CAST(NULL AS BIGINT), \
                    {global_tid} \
             FROM nsight.CUPTI_ACTIVITY_KIND_RUNTIME t \
             LEFT JOIN nsight.StringIds s ON t.nameId = s.id \
             WHERE t.rowid IN ({placeholders})"
        )),
        EventKind::Osrt
        | EventKind::Nvtx
        | EventKind::GraphNode
        | EventKind::GraphEvent
        | EventKind::CpuSample => Err(crate::NsysQueryError::internal_unsupported_kind(
            "correlate",
            kind.as_str(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use veloq_core::VeloqDiagnostic;

    fn kernel_row_id() -> RowId {
        RowId::new(EventKind::Kernel, 1)
    }

    #[test]
    fn fetch_gpu_corr_info_prepare_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let resolver = CudaProcessResolver::default();

        let err = match fetch_gpu_corr_info_with_sql(
            &conn,
            &resolver,
            kernel_row_id(),
            "SELECT * FROM",
        ) {
            Ok(_) => anyhow::bail!("malformed GPU corr-info SQL should not succeed"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-prepare");
        assert_eq!(
            err.sql_parts(),
            Some((
                "correlate",
                crate::SqlPhase::Prepare,
                CORRELATE_GPU_INFO_SQL
            ))
        );
        Ok(())
    }

    #[test]
    fn fetch_gpu_corr_info_query_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let resolver = CudaProcessResolver::default();
        let sql = "SELECT \
                   ? AS correlation_id, \
                   ? AS device_id, \
                   0::BIGINT AS context_id";

        let err = match fetch_gpu_corr_info_with_sql(&conn, &resolver, kernel_row_id(), sql) {
            Ok(_) => anyhow::bail!("unbound GPU corr-info SQL should not succeed"),
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
    fn fetch_gpu_corr_info_read_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let resolver = CudaProcessResolver::default();
        let sql = "SELECT \
                   'not-a-correlation' AS correlation_id, \
                   0::BIGINT AS device_id, \
                   0::BIGINT AS context_id \
                   WHERE ? IS NOT NULL";

        let err = match fetch_gpu_corr_info_with_sql(&conn, &resolver, kernel_row_id(), sql) {
            Ok(_) => anyhow::bail!("malformed GPU corr-info row should not succeed"),
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
    fn nsight_columns_prepare_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;

        let err =
            match nsight_columns_with_sql(&conn, "CUPTI_ACTIVITY_KIND_OVERHEAD", "SELECT * FROM") {
                Ok(cols) => anyhow::bail!(
                    "malformed correlate column-scan SQL should not succeed: {cols:?}"
                ),
                Err(err) => err,
            };

        assert_eq!(err.code().as_str(), "nsys.query.sql-prepare");
        assert_eq!(
            err.sql_parts(),
            Some((
                "correlate",
                crate::SqlPhase::Prepare,
                CORRELATE_COLUMN_SCAN_SQL
            ))
        );
        Ok(())
    }

    #[test]
    fn nsight_columns_query_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = "SELECT ? AS column_name WHERE ? IS NOT NULL";

        let err = match nsight_columns_with_sql(&conn, "CUPTI_ACTIVITY_KIND_OVERHEAD", sql) {
            Ok(cols) => {
                anyhow::bail!("unbound correlate column-scan SQL should not succeed: {cols:?}")
            }
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
    fn nsight_columns_read_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = "SELECT 1 AS column_name WHERE ? IS NOT NULL";

        let err = match nsight_columns_with_sql(&conn, "CUPTI_ACTIVITY_KIND_OVERHEAD", sql) {
            Ok(cols) => {
                anyhow::bail!("malformed correlate column-scan row should not succeed: {cols:?}")
            }
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
    fn hydrate_summary_batch_prepare_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;

        let err = match hydrate_summary_batch(&conn, "SELECT * FROM", EventKind::Runtime, &[1]) {
            Ok(rows) => anyhow::bail!(
                "malformed correlate summary SQL should not hydrate successfully: {} rows",
                rows.len()
            ),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-prepare");
        assert_eq!(
            err.sql_parts(),
            Some(("correlate", crate::SqlPhase::Prepare, CORRELATE_SUMMARY_SQL))
        );
        Ok(())
    }

    #[test]
    fn hydrate_summary_batch_query_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = "SELECT \
                   ? AS rowid, \
                   ? AS name, \
                   0::BIGINT AS start_ns, \
                   0::BIGINT AS duration_ns, \
                   CAST(NULL AS INTEGER) AS device_id, \
                   CAST(NULL AS BIGINT) AS stream_id, \
                   0::BIGINT AS global_tid";

        let err = match hydrate_summary_batch(&conn, sql, EventKind::Runtime, &[1]) {
            Ok(rows) => anyhow::bail!(
                "unbound correlate summary SQL should not hydrate successfully: {} rows",
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
    fn hydrate_summary_batch_read_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = "SELECT \
                   'not-a-rowid' AS rowid, \
                   'runtime' AS name, \
                   0::BIGINT AS start_ns, \
                   0::BIGINT AS duration_ns, \
                   CAST(NULL AS INTEGER) AS device_id, \
                   CAST(NULL AS BIGINT) AS stream_id, \
                   0::BIGINT AS global_tid \
                   WHERE ? IS NOT NULL";

        let err = match hydrate_summary_batch(&conn, sql, EventKind::Runtime, &[1]) {
            Ok(rows) => anyhow::bail!(
                "malformed correlate summary row should not hydrate successfully: {} rows",
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
