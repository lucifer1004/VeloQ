use super::gpu_activity::{DevCtxMap, DevCtxValue, collect_runtime_dev_ctx};
use super::{EnclosingNvtx, RuntimeParentEntry};
use crate::{NsysDataResult, Trace};
use rayon::prelude::*;
use std::collections::HashMap;

// ----- sweep ---------------------------------------------------------------

#[derive(Debug, Clone)]
pub(super) struct NvtxRangeRow {
    pub(super) rowid: i64,
    pub(super) start: i64,
    pub(super) end: i64,
    pub(super) name: String,
}

#[derive(Debug)]
pub(super) struct RuntimeRow {
    pub(super) rowid: i64,
    pub(super) correlation_id: Option<i64>,
    pub(super) native_pid: i64,
    pub(super) global_tid: i64,
    pub(super) start: i64,
    pub(super) end: i64,
    /// Derived from the matching GPU activity (kernel/memcpy/memset/
    /// sync) via the `(correlationId, native_pid)` join, where
    /// `native_pid` is mapped back to `(deviceId, contextId)` through
    /// `TARGET_INFO_CUDA_CONTEXT_INFO`. `None` when the runtime call
    /// has no GPU activity, no CUDA correlation, or the trace lacks
    /// the context-info table.
    pub(super) device_id: Option<i32>,
    pub(super) context_id: Option<i64>,
}

/// Compute the parent map from scratch. Same algorithm whether we're
/// building the sidecar on cold open or rebuilding after a trace edit.
pub(super) fn compute(trace: &Trace) -> NsysDataResult<Vec<RuntimeParentEntry>> {
    let nvtx_by_tid = collect_nvtx_by_tid(trace)?;
    if nvtx_by_tid.is_empty() {
        return Ok(Vec::new());
    }
    let runtime = collect_runtime_rows(trace)?;
    let walked = walk(&nvtx_by_tid, &runtime);
    // Backfill `device_id` / `context_id` from the GPU side. When
    // `TARGET_INFO_CUDA_CONTEXT_INFO` or every GPU activity table is
    // missing the map is empty — walked entries retain `None` for
    // both, the `by_correlation` map ends up empty, and runtime-side
    // `by_rt_rowid` attribution still works.
    let dev_ctx = collect_runtime_dev_ctx(trace)?;
    if dev_ctx.is_empty() {
        return Ok(walked);
    }
    Ok(merge_dev_ctx(walked, &dev_ctx))
}

/// Per attributed runtime row, look up its `(device_id, context_id)`
/// in `dev_ctx` and either set the fields in-place (the common case
/// of exactly one match) or fan out to multiple sidecar entries (one
/// per `(device, context)` candidate when CUPTI somehow presents the
/// same `(native_pid, correlation_id)` for multiple contexts).
///
/// Fan-out keeps the process-aware `by_correlation` map unambiguous on
/// every legitimate GPU row: each kernel/memcpy/memset/sync brings
/// its own `(native_pid, deviceId, contextId, correlationId)` and only ever
/// joins the one sidecar entry that matches. Runtime-side
/// `by_rt_rowid` collapses the fanout (the enclosing chain is
/// identical across copies, so any copy is correct for that map's
/// purpose).
pub(super) fn merge_dev_ctx(
    walked: Vec<RuntimeParentEntry>,
    dev_ctx: &DevCtxMap,
) -> Vec<RuntimeParentEntry> {
    let mut out: Vec<RuntimeParentEntry> = Vec::with_capacity(walked.len());
    for entry in walked {
        let lookup = entry
            .correlation_id
            .and_then(|corr| dev_ctx.get(&(entry.native_pid, corr)));
        match lookup {
            None => out.push(entry),
            Some(DevCtxValue::Single((dev, ctx))) => {
                let mut e = entry;
                e.device_id = Some(*dev);
                e.context_id = Some(*ctx);
                out.push(e);
            }
            Some(DevCtxValue::Many(many)) => {
                // Multi-candidate fan-out. Rare — only fires when
                // CUPTI presents an ambiguous `(native_pid,
                // correlationId) → (device, context)` mapping.
                // Cloning the enclosing chain is acceptable because
                // the rare case stays bounded by the number of
                // clashes, not by the trace size.
                for (dev, ctx) in many {
                    let mut e = entry.clone();
                    e.device_id = Some(*dev);
                    e.context_id = Some(*ctx);
                    out.push(e);
                }
            }
        }
    }
    out
}

pub(super) fn collect_nvtx_by_tid(
    trace: &Trace,
) -> NsysDataResult<HashMap<i64, Vec<NvtxRangeRow>>> {
    const TABLE: &str = "NVTX_EVENTS";
    if !trace.table_exists("NVTX_EVENTS") {
        return Ok(HashMap::new());
    }
    let global_tid = crate::sql_expr::u64_bits_to_i64("n.globalTid");
    let sql = format!(
        r#"SELECT n.rowid, n.start, n."end", {global_tid},
                        COALESCE(n.text, s.value, '<unnamed>') AS name
                 FROM nsight.NVTX_EVENTS n
                 LEFT JOIN nsight.StringIds s ON n.textId = s.id
                 WHERE n."end" IS NOT NULL
                   AND n.globalTid IS NOT NULL
                   AND n.start IS NOT NULL"#
    );
    let mut stmt = trace
        .conn()
        .prepare(&sql)
        .map_err(|source| crate::NsysDataError::nvtx_parent_rows_prepare(TABLE, source))?;
    let mut rows = stmt
        .query([])
        .map_err(|source| crate::NsysDataError::nvtx_parent_rows_query(TABLE, source))?;
    let mut by_tid: HashMap<i64, Vec<NvtxRangeRow>> = HashMap::new();
    while let Some(r) = rows
        .next()
        .map_err(|source| crate::NsysDataError::nvtx_parent_rows_read(TABLE, source))?
    {
        let rowid: i64 = r
            .get(0)
            .map_err(|source| crate::NsysDataError::nvtx_parent_rows_read(TABLE, source))?;
        let start: i64 = r
            .get(1)
            .map_err(|source| crate::NsysDataError::nvtx_parent_rows_read(TABLE, source))?;
        let end: i64 = r
            .get(2)
            .map_err(|source| crate::NsysDataError::nvtx_parent_rows_read(TABLE, source))?;
        let tid: i64 = r
            .get(3)
            .map_err(|source| crate::NsysDataError::nvtx_parent_rows_read(TABLE, source))?;
        let name: String = r
            .get(4)
            .map_err(|source| crate::NsysDataError::nvtx_parent_rows_read(TABLE, source))?;
        by_tid.entry(tid).or_default().push(NvtxRangeRow {
            rowid,
            start,
            end,
            name,
        });
    }
    for v in by_tid.values_mut() {
        v.sort_by_key(|r| r.start);
    }
    Ok(by_tid)
}

pub(super) fn collect_runtime_rows(trace: &Trace) -> NsysDataResult<Vec<RuntimeRow>> {
    const TABLE: &str = "CUPTI_ACTIVITY_KIND_RUNTIME";
    if !trace.table_exists("CUPTI_ACTIVITY_KIND_RUNTIME") {
        return Ok(Vec::new());
    }
    // `correlationId` is nullable — runtime calls without a CUDA
    // correlation (e.g. `cudaGetDeviceCount`) still need to be
    // attributed for runtime-side NVTX containment. Previous versions
    // filtered them out, silently dropping rows from
    // `--type runtime --group-by nvtx-parent`. The schema carries them
    // with `correlation_id = None`; the GPU-side `by_correlation` map
    // simply skips them.
    let global_tid = crate::sql_expr::u64_bits_to_i64("t.globalTid");
    let sql = format!(
        r#"SELECT t.rowid,
                        t.correlationId,
                        CAST(((t.globalTid >> 24) & 16777215) AS BIGINT) AS native_pid,
                        {global_tid},
                        t.start,
                        t."end"
                 FROM nsight.CUPTI_ACTIVITY_KIND_RUNTIME t
                 WHERE t."end" IS NOT NULL
                   AND t.globalTid IS NOT NULL"#
    );
    let mut stmt = trace
        .conn()
        .prepare(&sql)
        .map_err(|source| crate::NsysDataError::nvtx_parent_rows_prepare(TABLE, source))?;
    let mut rows = stmt
        .query([])
        .map_err(|source| crate::NsysDataError::nvtx_parent_rows_query(TABLE, source))?;
    let mut out: Vec<RuntimeRow> = Vec::new();
    while let Some(r) = rows
        .next()
        .map_err(|source| crate::NsysDataError::nvtx_parent_rows_read(TABLE, source))?
    {
        out.push(RuntimeRow {
            rowid: r
                .get(0)
                .map_err(|source| crate::NsysDataError::nvtx_parent_rows_read(TABLE, source))?,
            correlation_id: r
                .get(1)
                .map_err(|source| crate::NsysDataError::nvtx_parent_rows_read(TABLE, source))?,
            native_pid: r
                .get(2)
                .map_err(|source| crate::NsysDataError::nvtx_parent_rows_read(TABLE, source))?,
            global_tid: r
                .get(3)
                .map_err(|source| crate::NsysDataError::nvtx_parent_rows_read(TABLE, source))?,
            start: r
                .get(4)
                .map_err(|source| crate::NsysDataError::nvtx_parent_rows_read(TABLE, source))?,
            end: r
                .get(5)
                .map_err(|source| crate::NsysDataError::nvtx_parent_rows_read(TABLE, source))?,
            // Filled in by `collect_runtime_dev_ctx` upstream.
            device_id: None,
            context_id: None,
        });
    }
    Ok(out)
}

/// Per runtime row, collect *every* enclosing NVTX range on the same
/// `globalTid` (containment = `n.start <= r.start AND n.end >= r.end`).
/// Output order is outermost-first so callers can pick the innermost
/// via `.last()` without resorting.
///
/// ## Algorithm
///
/// A per-runtime walk-back from the per-tid NVTX upper-bound would be
/// O(N_runtime × N_nvtx_per_tid) — fine on shallow traces but quadratic
/// on traces with thousands of NVTX ranges per tid. Instead, the scan
/// is inverted: for each NVTX range, binary-search its containing slice
/// of pre-sorted-by-start runtime rows and append the range to each
/// row's enclosing list. Cost is O(N_nvtx × log N_runtime + matches),
/// three orders of magnitude less work on traces with thousands of
/// NVTX ranges per tid.
///
/// Rayon-parallelism is *across NVTX ranges* (the outer-loop unit of
/// work); per-runtime enclosing lists merge via per-thread shards
/// keyed by runtime index, reconciled at the end.
pub(super) fn walk(
    nvtx_by_tid: &HashMap<i64, Vec<NvtxRangeRow>>,
    runtime: &[RuntimeRow],
) -> Vec<RuntimeParentEntry> {
    if runtime.is_empty() || nvtx_by_tid.is_empty() {
        return Vec::new();
    }

    // Per-tid sorted runtime indices keyed by start. Each entry is
    // `(start, runtime_index)` so the partition_point gives a range
    // of original indices to decorate.
    let mut rt_sorted_by_tid: HashMap<i64, Vec<(i64, u32)>> = HashMap::new();
    for (i, r) in runtime.iter().enumerate() {
        rt_sorted_by_tid
            .entry(r.global_tid)
            .or_default()
            .push((r.start, i as u32));
    }
    for v in rt_sorted_by_tid.values_mut() {
        v.sort_unstable_by_key(|(s, _)| *s);
    }

    // Per-runtime-row enclosing chain in *insertion order*. We
    // append NVTX rows as we encounter them; outermost ranges have
    // smaller `start`, but we visit them in arbitrary order, so a
    // final per-row sort by start arranges them outer→inner.
    //
    // `Mutex` per shard would over-contend (millions of inserts).
    // Instead, each rayon worker produces a partial map keyed by
    // runtime index; we merge after the parallel section.
    use rayon::iter::IntoParallelIterator;
    let partials: Vec<Vec<(u32, NvtxLink)>> = nvtx_by_tid
        .iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .flat_map_iter(|(tid, ranges)| {
            let rt_sorted = rt_sorted_by_tid.get(tid).cloned().unwrap_or_default();
            ranges.iter().map(move |nvtx| {
                let mut links: Vec<(u32, NvtxLink)> = Vec::new();
                if rt_sorted.is_empty() {
                    return links;
                }
                // First runtime row with start >= nvtx.start.
                let lo = rt_sorted.partition_point(|(s, _)| *s < nvtx.start);
                for &(rt_start, rt_idx) in rt_sorted.iter().skip(lo) {
                    if rt_start > nvtx.end {
                        break;
                    }
                    // `.get()` (not index): `rt_idx` is in-bounds by
                    // construction; this satisfies clippy::indexing_slicing.
                    let Some(rt_row) = runtime.get(rt_idx as usize) else {
                        continue;
                    };
                    if rt_row.end <= nvtx.end {
                        links.push((
                            rt_idx,
                            NvtxLink {
                                start: nvtx.start,
                                end: nvtx.end,
                                rowid: nvtx.rowid,
                                // `name` is owned upstream; cloning
                                // here is unavoidable because the
                                // sidecar entries outlive the source
                                // rows. Strings are short
                                // (NVTX-range labels) so the cost is
                                // ~30 ns/clone.
                                name: nvtx.name.clone(),
                            },
                        ));
                    }
                }
                links
            })
        })
        .fold(Vec::new, |mut acc, mut chunk| {
            acc.append(&mut chunk);
            acc
        })
        .collect();

    // Bucket links into a per-runtime vector and finalise. Build via
    // `iter().map()` so each slot is its own freshly-allocated
    // `Vec<NvtxLink>` (the `vec![Vec::new(); n]` shorthand requires
    // `Clone` which `NvtxLink` doesn't implement).
    let mut chains: Vec<Vec<NvtxLink>> = (0..runtime.len()).map(|_| Vec::new()).collect();
    for partial in partials {
        for (rt_idx, link) in partial {
            if let Some(slot) = chains.get_mut(rt_idx as usize) {
                slot.push(link);
            }
        }
    }

    let mut out: Vec<RuntimeParentEntry> = Vec::with_capacity(runtime.len() / 10);
    for (i, mut chain) in chains.into_iter().enumerate() {
        if chain.is_empty() {
            continue;
        }
        // Outer→inner ordering: smaller start first; for ranges
        // sharing a start, the one with the *later* end (i.e. wider
        // interval) is outer. Tie-break on `rowid` for deterministic
        // chain shape across rebuilds.
        chain.sort_unstable_by(|a, b| {
            a.start
                .cmp(&b.start)
                .then(b.end.cmp(&a.end))
                .then(a.rowid.cmp(&b.rowid))
        });
        let enclosing: Vec<EnclosingNvtx> = chain
            .into_iter()
            .map(|l| EnclosingNvtx {
                nvtx_rowid: l.rowid,
                nvtx_name: l.name,
            })
            .collect();
        // `chains[i]` originated from `runtime[i]`; both vectors are
        // `runtime.len()` long, so a `.get()` here is just defensive.
        let Some(r) = runtime.get(i) else { continue };
        out.push(RuntimeParentEntry {
            rt_rowid: r.rowid,
            correlation_id: r.correlation_id,
            native_pid: r.native_pid,
            device_id: r.device_id,
            context_id: r.context_id,
            enclosing,
        });
    }
    out
}

/// Intermediate (runtime_index → enclosing NVTX) link produced by
/// the per-NVTX scan. Kept compact and `Send` so rayon shuffles them
/// across worker threads cheaply.
///
/// `start` and `end` are both carried for the final outer→inner
/// sort: NVTX ranges with the same start are disambiguated by end
/// (larger end = outer), and ties beyond that fall back to rowid so
/// the chain order is deterministic across rebuilds.
#[derive(Debug)]
struct NvtxLink {
    start: i64,
    end: i64,
    rowid: i64,
    name: String,
}
