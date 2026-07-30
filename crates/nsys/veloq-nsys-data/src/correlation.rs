//! Correlation index — `correlation_id ↔ (kernels, memcpys, runtimes)`.
//!
//! NSys's CUDA identifiers are process-local. To disambiguate rank-private
//! namespaces, the index keys every group by
//! `(native_pid, device_id, context_id, correlation_id)`.
//!
//! For runtime API events (`CUPTI_ACTIVITY_KIND_RUNTIME`) the source row
//! has `globalTid` but not `(device_id, context_id)`. We close that gap
//! via `TARGET_INFO_CUDA_CONTEXT_INFO`, which provides a
//! `(device_id, context_id) ↔ processId` mapping. The runtime row's
//! native PID is extracted from `globalTid` via
//! [`native_pid_from_global_tid`] (skip the source-domain byte:
//! `(id >> 24) & 0xFFFFFF`).
//!
//! ## Disk cache
//!
//! Building the index requires three full table scans plus a sort,
//! which is multi-second on large traces. Because veloq is a one-shot
//! CLI that agents call repeatedly against the same trace, the index
//! is persisted to `<trace_path>.veloq/correlation.bin` on first build.
//! Subsequent calls deserialise in milliseconds.
//!
//! Persistence goes through [`veloq_core::SidecarCache<CorrelationIndex>`];
//! [`CACHE_VERSION`] gates payload-shape changes (rebuild on
//! mismatch), and SidecarCache itself handles the source-fingerprint
//! invalidation, bincode wrapping, and atomic-rename write.

use crate::cuda_identity::{CudaProcessResolver, native_pid_from_global_tid};
use crate::{NsysDataResult, Trace};
use duckdb::Connection;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use veloq_core::SidecarCache;

// `CorrelatedRowIds` carries separate `sync` and `graph` buckets:
// synchronisation events carry a correlationId too (surfacing them lets
// `correlate` answer "what blocked this stream sync"), and graph_trace
// rows share `correlationId` with the launching `cudaGraphLaunch` runtime
// call (so `correlate kernel:N` on a graph launch surfaces the graph row
// alongside cpu/gpu events).
const CACHE_VERSION: u32 = 5;

/// Group of correlated table row ids for one synthetic correlation id.
/// Stored per-kind; lookups return all four lists.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CorrelatedRowIds {
    pub kernel: Vec<i64>,
    pub memcpy: Vec<i64>,
    pub memset: Vec<i64>,
    pub runtime: Vec<i64>,
    #[serde(default)]
    pub sync: Vec<i64>,
    #[serde(default)]
    pub graph: Vec<i64>,
}

impl CorrelatedRowIds {
    pub fn is_empty(&self) -> bool {
        self.kernel.is_empty()
            && self.memcpy.is_empty()
            && self.memset.is_empty()
            && self.runtime.is_empty()
            && self.sync.is_empty()
            && self.graph.is_empty()
    }
}

/// `correlation_id`-keyed index of CPU↔GPU related events.
///
/// Built by walking kernel/memcpy/memset/runtime tables once. Lookups
/// against the index are O(1) on the synthetic id.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CorrelationIndex {
    /// Distinct `(process_id, device_id, context_id)` mappings.
    contexts: HashSet<(u64, u64, u64)>,
    /// process_id → all (device_id, context_id) pairs that process owns.
    /// Multi-GPU runs (or any process with secondary contexts) put more
    /// than one entry here; runtime API rows must consider every candidate
    /// because the row itself doesn't carry (device, context).
    process_to_contexts: HashMap<u64, Vec<(u64, u64)>>,
    /// synthetic_id → all events with that correlation.
    groups: HashMap<SyntheticId, CorrelatedRowIds>,
}

#[derive(Debug, Clone, Default)]
pub struct CorrelationIndexStats {
    pub contexts: usize,
    pub processes: usize,
    pub unique_groups: usize,
    pub kernel_rows: usize,
    pub memcpy_rows: usize,
    pub memset_rows: usize,
    pub runtime_rows: usize,
    pub sync_rows: usize,
    pub graph_rows: usize,
}

/// Process-aware CUDA correlation identity.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub struct SyntheticId {
    process: u64,
    device: u64,
    context: u64,
    raw_corr: u64,
}

impl SyntheticId {
    /// Construct the canonical `(process, device, context, raw_corr)`
    /// identity. The name stays `pack` for source compatibility with
    /// existing internal callers; no lossy bit-packing is performed.
    #[inline]
    pub const fn pack(process: u64, device: u64, context: u64, raw_corr: u64) -> Self {
        Self {
            process,
            device,
            context,
            raw_corr,
        }
    }

    #[inline]
    pub const fn process(self) -> u64 {
        self.process
    }
}

impl std::fmt::Display for SyntheticId {
    /// Opaque, fixed-axis representation suitable for row keys and logs.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "p{:x}-d{:x}-c{:x}-r{:x}",
            self.process, self.device, self.context, self.raw_corr
        )
    }
}

#[inline]
fn synthetic_id(process: u64, device: u64, context: u64, raw_corr: u64) -> SyntheticId {
    SyntheticId::pack(process, device, context, raw_corr)
}

impl CorrelationIndex {
    /// Lookup by exact `(process, device, context, correlation_id)`.
    pub fn lookup(
        &self,
        process: u64,
        device: u64,
        context: u64,
        raw_corr: u64,
    ) -> Option<&CorrelatedRowIds> {
        self.groups
            .get(&synthetic_id(process, device, context, raw_corr))
    }

    /// Lookup by `correlation_id` when the caller has a runtime API event
    /// (only knows `globalTid`, not device/context). Resolves PID → all
    /// candidate (device, context) pairs via `TARGET_INFO_CUDA_CONTEXT_INFO`.
    ///
    /// In the common case (single context per PID) we can just borrow
    /// the matching group; only when multiple contexts contribute do
    /// we materialise a merged owned copy. Returning [`Cow`] keeps the
    /// hot path zero-copy without giving up correctness on multi-GPU
    /// runs.
    pub fn lookup_by_runtime(
        &self,
        global_tid: i64,
        raw_corr: u64,
    ) -> Option<Cow<'_, CorrelatedRowIds>> {
        let pid = native_pid_from_global_tid(global_tid) as u64;
        let candidates = self.process_to_contexts.get(&pid)?;
        // Single-borrow fast path: gather references to non-empty groups
        // first. If exactly one matches we hand it out as `Cow::Borrowed`
        // without any allocation.
        let mut hits: Vec<&CorrelatedRowIds> = Vec::with_capacity(candidates.len());
        for &(dev, ctx) in candidates {
            if let Some(g) = self.lookup(pid, dev, ctx, raw_corr) {
                hits.push(g);
            }
        }
        // Single-borrow fast path stays a borrow; multi-context fans
        // into a freshly-merged owned struct.
        let mut iter = hits.into_iter();
        let first = iter.next()?;
        match iter.next() {
            None => Some(Cow::Borrowed(first)),
            Some(second) => {
                let mut merged = CorrelatedRowIds::default();
                for g in std::iter::once(first)
                    .chain(std::iter::once(second))
                    .chain(iter)
                {
                    merged.kernel.extend_from_slice(&g.kernel);
                    merged.memcpy.extend_from_slice(&g.memcpy);
                    merged.memset.extend_from_slice(&g.memset);
                    merged.runtime.extend_from_slice(&g.runtime);
                    merged.sync.extend_from_slice(&g.sync);
                    merged.graph.extend_from_slice(&g.graph);
                }
                Some(Cow::Owned(merged))
            }
        }
    }

    /// All (device, context) pairs known for a process. Callers that want
    /// to display the resolved synthetic id alongside `lookup_by_runtime`
    /// results use this to enumerate candidates.
    pub fn contexts_for_pid(&self, pid: u64) -> &[(u64, u64)] {
        self.process_to_contexts
            .get(&pid)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn stats(&self) -> CorrelationIndexStats {
        let mut s = CorrelationIndexStats {
            contexts: self.contexts.len(),
            processes: self.process_to_contexts.len(),
            unique_groups: self.groups.len(),
            ..Default::default()
        };
        for g in self.groups.values() {
            s.kernel_rows += g.kernel.len();
            s.memcpy_rows += g.memcpy.len();
            s.memset_rows += g.memset.len();
            s.runtime_rows += g.runtime.len();
            s.sync_rows += g.sync.len();
            s.graph_rows += g.graph.len();
        }
        s
    }

    /// Build the index by scanning the trace. Does NOT consult or write
    /// the disk cache — use `build_or_load` for that.
    ///
    /// Uses Parquet-backed `nsight.<TABLE>` views, so a built-up trace
    /// runs the index build orders of magnitude faster than repeated
    /// raw table scans.
    pub fn build(trace: &Trace) -> NsysDataResult<Self> {
        let mut idx = Self::default();
        // TARGET_INFO_CUDA_CONTEXT_INFO is small; read it straight
        // from the attached DuckDB view.
        idx.load_context_process_maps(trace.conn())?;
        let resolver = CudaProcessResolver::build(trace)?;
        let items = collect_correlation_items(trace, &resolver, &idx.process_to_contexts)?;
        idx.merge_items(trace, items)?;
        Ok(idx)
    }

    fn load_context_process_maps(&mut self, conn: &Connection) -> NsysDataResult<()> {
        // Probe table existence cheaply (LIMIT 0). If absent, we can
        // still serve kernel/memcpy/memset lookups (which carry their
        // own device/context) — only runtime API resolution becomes
        // lossy. Log and continue.
        let probe = conn.execute(
            "SELECT 1 FROM nsight.TARGET_INFO_CUDA_CONTEXT_INFO LIMIT 0",
            [],
        );
        if probe.is_err() {
            log::warn!(
                "TARGET_INFO_CUDA_CONTEXT_INFO is missing — runtime↔GPU \
                 correlation will be limited to events with matching device/context"
            );
            return Ok(());
        }

        const TABLE: &str = "TARGET_INFO_CUDA_CONTEXT_INFO";
        let mut stmt = conn
            .prepare(
                "SELECT deviceId, contextId, processId FROM nsight.TARGET_INFO_CUDA_CONTEXT_INFO",
            )
            .map_err(|source| crate::NsysDataError::correlation_scan_prepare(TABLE, source))?;
        let mut rows = stmt
            .query([])
            .map_err(|source| crate::NsysDataError::correlation_scan_query(TABLE, source))?;
        while let Some(r) = rows
            .next()
            .map_err(|source| crate::NsysDataError::correlation_scan_read(TABLE, source))?
        {
            let device: i64 = r
                .get(0)
                .map_err(|source| crate::NsysDataError::correlation_scan_read(TABLE, source))?;
            let context: i64 = r
                .get(1)
                .map_err(|source| crate::NsysDataError::correlation_scan_read(TABLE, source))?;
            let process: i64 = r
                .get(2)
                .map_err(|source| crate::NsysDataError::correlation_scan_read(TABLE, source))?;
            let dev = device as u64;
            let ctx = context as u64;
            let pid = process as u64;
            self.contexts.insert((pid, dev, ctx));
            // Dedup: TARGET_INFO_CUDA_CONTEXT_INFO usually has one row per
            // (dev, ctx, pid) triple but be defensive against duplicates.
            let entry = self.process_to_contexts.entry(pid).or_default();
            if !entry.contains(&(dev, ctx)) {
                entry.push((dev, ctx));
            }
        }
        log::info!(
            "loaded {} CUDA context↔process mappings across {} processes",
            self.contexts.len(),
            self.process_to_contexts.len()
        );
        Ok(())
    }

    fn merge_items(
        &mut self,
        trace: &Trace,
        mut items: Vec<CorrelationItem>,
    ) -> NsysDataResult<()> {
        // Sort-merge: sort by syn_id then kind for stable grouping, then
        // linear-scan into per-syn_id buckets. O(N log N) sort + O(N)
        // group; performs well even on millions of items.
        let pool = trace.build_query_worker_pool()?;
        pool.install(|| {
            items.par_sort_unstable_by_key(|item| (item.syn_id, item.kind as u8));
        });

        let mut current_id = None;
        let mut current_group = CorrelatedRowIds::default();
        for item in items {
            if current_id != Some(item.syn_id) {
                if let Some(id) = current_id.replace(item.syn_id) {
                    self.groups.insert(id, std::mem::take(&mut current_group));
                }
            }
            current_group.push(item);
        }
        if let Some(id) = current_id {
            self.groups.insert(id, current_group);
        }
        Ok(())
    }

    // ---- disk cache ------------------------------------------------------

    /// Build or load the index. On first call: try cache, validate
    /// against source mtime/size, fall back to rebuild + atomic save.
    /// On subsequent calls (same process or fresh): hit the cache.
    pub fn build_or_load(trace: &Trace) -> NsysDataResult<Self> {
        let trace_path = trace.path();
        let cache = cache_handle(trace_path);
        let fp = crate::trace_artifact_fingerprint(trace_path).map_err(|source| {
            crate::NsysDataError::correlation_trace_fingerprint(trace_path.display(), source)
        })?;

        match cache.try_load(fp) {
            Ok(Some(idx)) => {
                // Demoted to debug: this fires on every warm call and
                // would otherwise clutter info-level output. First-build
                // progress (the slow path users care about) stays at info.
                log::debug!(
                    "correlation index loaded from cache: {} groups",
                    idx.groups.len()
                );
                return Ok(idx);
            }
            Ok(None) => {}
            Err(e) => {
                log::warn!(
                    "correlation cache at {} unusable ({e}); rebuilding",
                    cache.path().display()
                );
            }
        }

        let started = std::time::Instant::now();
        let idx = Self::build(trace)?;
        log::info!(
            "correlation index built in {:?}: {:?}",
            started.elapsed(),
            idx.stats()
        );

        if let Err(e) = cache.write(fp, &idx) {
            log::warn!(
                "failed to write correlation cache at {}: {e}",
                cache.path().display()
            );
        }
        Ok(idx)
    }
}

// ============================================================================
// Item collection
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum ItemKind {
    Kernel = 0,
    Memcpy = 1,
    Memset = 2,
    Runtime = 3,
    Sync = 4,
    Graph = 5,
}

#[derive(Debug, Clone, Copy)]
struct CorrelationItem {
    syn_id: SyntheticId,
    rowid: i64,
    kind: ItemKind,
}

impl CorrelatedRowIds {
    fn push(&mut self, item: CorrelationItem) {
        match item.kind {
            ItemKind::Kernel => self.kernel.push(item.rowid),
            ItemKind::Memcpy => self.memcpy.push(item.rowid),
            ItemKind::Memset => self.memset.push(item.rowid),
            ItemKind::Runtime => self.runtime.push(item.rowid),
            ItemKind::Sync => self.sync.push(item.rowid),
            ItemKind::Graph => self.graph.push(item.rowid),
        }
    }
}

fn collect_correlation_items(
    trace: &Trace,
    resolver: &CudaProcessResolver,
    process_to_contexts: &HashMap<u64, Vec<(u64, u64)>>,
) -> NsysDataResult<Vec<CorrelationItem>> {
    let mut items: Vec<CorrelationItem> = Vec::new();

    // Kernel/memcpy/memset/sync: resolve process and retain the native
    // (process, device, context, correlation_id) identity.
    collect_gpu_kind(
        trace,
        resolver,
        "CUPTI_ACTIVITY_KIND_KERNEL",
        ItemKind::Kernel,
        &mut items,
    )?;
    collect_gpu_kind(
        trace,
        resolver,
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        ItemKind::Memcpy,
        &mut items,
    )?;
    collect_gpu_kind(
        trace,
        resolver,
        "CUPTI_ACTIVITY_KIND_MEMSET",
        ItemKind::Memset,
        &mut items,
    )?;
    collect_gpu_kind(
        trace,
        resolver,
        "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
        ItemKind::Sync,
        &mut items,
    )?;
    // Graph_trace rows share correlationId with the host `cudaGraphLaunch`
    // call; same (process, device, context, correlationId) shape as kernels.
    collect_gpu_kind(
        trace,
        resolver,
        "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
        ItemKind::Graph,
        &mut items,
    )?;

    // Runtime: only `globalTid` available; resolve PID → all candidate
    // (device, context) pairs and emit one item under each.
    collect_runtime(trace, process_to_contexts, &mut items)?;

    log::info!("collected {} correlation items", items.len());
    Ok(items)
}

fn collect_gpu_kind(
    trace: &Trace,
    resolver: &CudaProcessResolver,
    table: &str,
    kind: ItemKind,
    out: &mut Vec<CorrelationItem>,
) -> NsysDataResult<()> {
    // Probe — table may be absent on partial traces.
    let probe_sql = format!("SELECT 1 FROM nsight.{table} LIMIT 0");
    let probe = trace.conn().execute(&probe_sql, []);
    if probe.is_err() {
        return Ok(());
    }
    let global_pid = if trace.table_has_column(table, "globalPid") {
        "CAST(globalPid AS BIGINT)"
    } else {
        "CAST(NULL AS BIGINT)"
    };
    let start = if trace.table_has_column(table, "start") {
        "CAST(start AS BIGINT)"
    } else if trace.table_has_column(table, "timestamp") {
        "CAST(timestamp AS BIGINT)"
    } else {
        "0::BIGINT"
    };
    let sql = format!(
        "SELECT rowid, correlationId, deviceId, contextId, {start}, {global_pid} \
         FROM nsight.{table} \
         WHERE correlationId IS NOT NULL"
    );
    let mut stmt = trace
        .conn()
        .prepare(&sql)
        .map_err(|source| crate::NsysDataError::correlation_scan_prepare(table, source))?;
    let mut rows = stmt
        .query([])
        .map_err(|source| crate::NsysDataError::correlation_scan_query(table, source))?;
    let mut n = 0;
    while let Some(r) = rows
        .next()
        .map_err(|source| crate::NsysDataError::correlation_scan_read(table, source))?
    {
        let rowid: i64 = r
            .get(0)
            .map_err(|source| crate::NsysDataError::correlation_scan_read(table, source))?;
        let corr: i64 = r
            .get(1)
            .map_err(|source| crate::NsysDataError::correlation_scan_read(table, source))?;
        let device: i64 = r
            .get(2)
            .map_err(|source| crate::NsysDataError::correlation_scan_read(table, source))?;
        let context: i64 = r
            .get(3)
            .map_err(|source| crate::NsysDataError::correlation_scan_read(table, source))?;
        let start_ns: i64 = r
            .get(4)
            .map_err(|source| crate::NsysDataError::correlation_scan_read(table, source))?;
        let global_pid: Option<i64> = r
            .get(5)
            .map_err(|source| crate::NsysDataError::correlation_scan_read(table, source))?;
        let process = resolver.resolve_required(
            table,
            device as i32,
            context,
            Some(corr),
            start_ns,
            global_pid,
        )? as u64;
        out.push(CorrelationItem {
            syn_id: synthetic_id(process, device as u64, context as u64, corr as u64),
            rowid,
            kind,
        });
        n += 1;
    }
    log::debug!("collected {n} {kind:?} correlation items from {table}");
    Ok(())
}

fn collect_runtime(
    trace: &Trace,
    process_to_contexts: &HashMap<u64, Vec<(u64, u64)>>,
    out: &mut Vec<CorrelationItem>,
) -> NsysDataResult<()> {
    let probe_sql = "SELECT 1 FROM nsight.CUPTI_ACTIVITY_KIND_RUNTIME LIMIT 0";
    let probe = trace.conn().execute(probe_sql, []);
    if probe.is_err() {
        return Ok(());
    }
    let global_tid = crate::sql_expr::u64_bits_to_i64("globalTid");
    let sql = format!(
        "SELECT rowid, correlationId, {global_tid} \
         FROM nsight.CUPTI_ACTIVITY_KIND_RUNTIME \
         WHERE correlationId IS NOT NULL AND globalTid IS NOT NULL"
    );
    const TABLE: &str = "CUPTI_ACTIVITY_KIND_RUNTIME";
    let mut stmt = trace
        .conn()
        .prepare(&sql)
        .map_err(|source| crate::NsysDataError::correlation_scan_prepare(TABLE, source))?;
    let mut rows = stmt
        .query([])
        .map_err(|source| crate::NsysDataError::correlation_scan_query(TABLE, source))?;
    let mut resolved = 0u64;
    let mut fanout = 0u64;
    let mut fallback = 0u64;
    while let Some(r) = rows
        .next()
        .map_err(|source| crate::NsysDataError::correlation_scan_read(TABLE, source))?
    {
        let rowid: i64 = r
            .get(0)
            .map_err(|source| crate::NsysDataError::correlation_scan_read(TABLE, source))?;
        let corr: i64 = r
            .get(1)
            .map_err(|source| crate::NsysDataError::correlation_scan_read(TABLE, source))?;
        let global_tid: i64 = r
            .get(2)
            .map_err(|source| crate::NsysDataError::correlation_scan_read(TABLE, source))?;
        let pid = native_pid_from_global_tid(global_tid) as u64;
        match process_to_contexts.get(&pid) {
            Some(ctxs) if !ctxs.is_empty() => {
                // Emit one item per candidate (dev, ctx). The runtime row
                // will be discoverable from any context's GPU work that
                // shares the raw_corr. In practice only one (dev, ctx)
                // bucket per raw_corr has GPU work, but indexing the
                // runtime row under all of them keeps lookups O(1).
                resolved += 1;
                if ctxs.len() > 1 {
                    fanout += 1;
                }
                for &(dev, ctx) in ctxs {
                    out.push(CorrelationItem {
                        syn_id: synthetic_id(pid, dev, ctx, corr as u64),
                        rowid,
                        kind: ItemKind::Runtime,
                    });
                }
            }
            _ => {
                // Fallback: stash PID into the context field. Won't
                // collide with real GPU events (which use real context
                // ids), but keeps the row searchable via `lookup_by_runtime`
                // if the caller later infers context.
                fallback += 1;
                out.push(CorrelationItem {
                    syn_id: synthetic_id(pid, 0, pid, corr as u64),
                    rowid,
                    kind: ItemKind::Runtime,
                });
            }
        };
    }
    log::debug!(
        "runtime correlation: {resolved} resolved ({fanout} multi-context), {fallback} fallback"
    );
    Ok(())
}

// ============================================================================
// Disk cache
// ============================================================================

fn cache_path_for(trace_path: &Path) -> PathBuf {
    veloq_core::artifact_dir_for(trace_path).join("correlation.bin")
}

/// Build the [`SidecarCache`] handle for a trace's correlation cache.
/// The cache layer carries path + version + label and dispatches the
/// load/write through bincode; the version constant lives here so a
/// payload-shape bump still gates rebuilds.
fn cache_handle(trace_path: &Path) -> SidecarCache<CorrelationIndex> {
    SidecarCache::new(
        cache_path_for(trace_path),
        CACHE_VERSION,
        "correlation cache",
    )
}

/// Best-effort cache file path. Public for status/reporting callers;
/// internal cache users should stay on `build_or_load`.
pub fn path_for(trace_path: &Path) -> PathBuf {
    cache_path_for(trace_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context, Result};
    use std::path::PathBuf;
    use tempfile::TempDir;
    use veloq_core::{SourceFingerprint, VeloqDiagnostic};

    fn parquet_fixture(tables: Vec<(&str, &str, Vec<&str>)>) -> Result<(TempDir, PathBuf)> {
        let dir = tempfile::tempdir()?;
        let pqtdir = dir.path().join("test_pqtdir");
        std::fs::create_dir_all(&pqtdir)?;
        let conn = Connection::open_in_memory()?;
        for (_, ddl, inserts) in &tables {
            conn.execute_batch(ddl)?;
            for insert in inserts {
                conn.execute_batch(insert)?;
            }
        }
        for (table, _, _) in &tables {
            let out = pqtdir.join(format!("{table}.parquet"));
            let out_lit = out.to_string_lossy().replace('\'', "''");
            conn.execute(
                &format!(r#"COPY (SELECT * FROM "{table}") TO '{out_lit}' (FORMAT PARQUET)"#),
                [],
            )?;
        }
        Ok((dir, pqtdir))
    }

    fn minimal_kernel_without_correlation() -> (&'static str, &'static str, Vec<&'static str>) {
        (
            "CUPTI_ACTIVITY_KIND_KERNEL",
            r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start BIGINT, "end" BIGINT)"#,
            Vec::new(),
        )
    }

    fn assert_correlation_scan_error(
        err: crate::NsysDataError,
        expected_code: &str,
        expected_table: &str,
    ) -> Result<()> {
        assert_eq!(err.code().as_str(), expected_code);
        let Some((area, _, label)) = err.duckdb_parts() else {
            anyhow::bail!("expected correlation scan DuckDB error, got {err:?}");
        };
        assert_eq!(area, "correlation scan");
        assert_eq!(label, expected_table);
        Ok(())
    }

    #[test]
    fn synthetic_id_layout() {
        let id = synthetic_id(1000, 1, 2, 3);
        assert_eq!(id, SyntheticId::pack(1000, 1, 2, 3));
        assert_eq!(id.process(), 1000);
        assert_eq!(id.to_string(), "p3e8-d1-c2-r3");
        assert_ne!(synthetic_id(1000, 0, 1, 42), synthetic_id(2000, 0, 1, 42));
    }

    #[test]
    fn native_pid_skips_source_domain_byte() {
        // pid=1000 in bits 24-47, source_domain=0x3B in bits 16-23,
        // native_tid in bits 0-15. Native PID extraction must skip the
        // domain byte.
        let global_tid = (1000i64 << 24) | (0x3B << 16) | 7;
        assert_eq!(native_pid_from_global_tid(global_tid), 1000);
    }

    #[test]
    fn build_or_load_missing_trace_fingerprint_error_is_typed() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture(vec![minimal_kernel_without_correlation()])?;
        let trace = Trace::open(&pqtdir)?;
        std::fs::remove_dir_all(&pqtdir)?;

        let err = match CorrelationIndex::build_or_load(&trace) {
            Ok(idx) => {
                anyhow::bail!("missing parquetdir should not build correlation index: {idx:?}")
            }
            Err(err) => err,
        };

        assert_eq!(
            err.code().as_str(),
            "nsys.data.correlation-trace-fingerprint"
        );
        match err {
            crate::NsysDataError::CorrelationTraceFingerprint { path, .. } => {
                assert!(path.contains("test_pqtdir"));
            }
            other => anyhow::bail!("expected CorrelationTraceFingerprint, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn build_or_load_context_scan_prepare_error_is_typed() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture(vec![
            minimal_kernel_without_correlation(),
            (
                "TARGET_INFO_CUDA_CONTEXT_INFO",
                "CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO (deviceId BIGINT, contextId BIGINT)",
                Vec::new(),
            ),
        ])?;
        let trace = Trace::open(&pqtdir)?;

        let err = match CorrelationIndex::build_or_load(&trace) {
            Ok(idx) => anyhow::bail!(
                "context table missing processId should not build correlation index: {idx:?}"
            ),
            Err(err) => err,
        };

        assert_correlation_scan_error(
            err,
            "nsys.data.duckdb-prepare",
            "TARGET_INFO_CUDA_CONTEXT_INFO",
        )
    }

    #[test]
    fn build_or_load_gpu_scan_prepare_error_is_typed() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture(vec![minimal_kernel_without_correlation()])?;
        let trace = Trace::open(&pqtdir)?;

        let err = match CorrelationIndex::build_or_load(&trace) {
            Ok(idx) => anyhow::bail!(
                "kernel table missing correlation columns should not build index: {idx:?}"
            ),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.data.duckdb-prepare");
        assert_eq!(
            err.duckdb_parts(),
            Some((
                "correlation scan",
                crate::DuckdbPhase::Prepare,
                "CUPTI_ACTIVITY_KIND_KERNEL",
            ))
        );
        Ok(())
    }

    #[test]
    fn build_or_load_gpu_scan_read_error_is_typed() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture(vec![(
            "CUPTI_ACTIVITY_KIND_KERNEL",
            r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start BIGINT, "end" BIGINT, correlationId TEXT, deviceId BIGINT, contextId BIGINT)"#,
            vec![
                r#"INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (start, "end", correlationId, deviceId, contextId) VALUES (0, 1, 'bad', 0, 7)"#,
            ],
        )])?;
        let trace = Trace::open(&pqtdir)?;

        let err = match CorrelationIndex::build_or_load(&trace) {
            Ok(idx) => {
                anyhow::bail!("bad GPU correlationId should not build correlation index: {idx:?}")
            }
            Err(err) => err,
        };

        assert_correlation_scan_error(err, "nsys.data.duckdb-read", "CUPTI_ACTIVITY_KIND_KERNEL")
    }

    #[test]
    fn build_or_load_runtime_scan_read_error_is_typed() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture(vec![(
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            r#"CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start BIGINT, "end" BIGINT, correlationId TEXT, globalTid BIGINT)"#,
            vec![
                r#"INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME (start, "end", correlationId, globalTid) VALUES (0, 1, 'bad', 7)"#,
            ],
        )])?;
        let trace = Trace::open(&pqtdir)?;

        let err = match CorrelationIndex::build_or_load(&trace) {
            Ok(idx) => anyhow::bail!(
                "bad runtime correlationId should not build correlation index: {idx:?}"
            ),
            Err(err) => err,
        };

        assert_correlation_scan_error(err, "nsys.data.duckdb-query", "CUPTI_ACTIVITY_KIND_RUNTIME")
    }

    #[test]
    fn lookup_returns_grouped_rows() -> Result<()> {
        let mut idx = CorrelationIndex::default();
        let syn = synthetic_id(1000, 2, 5, 999);
        idx.groups.insert(
            syn,
            CorrelatedRowIds {
                kernel: vec![100, 101],
                memcpy: vec![200],
                memset: vec![],
                runtime: vec![300],
                sync: vec![400],
                graph: vec![500],
            },
        );
        let got = idx.lookup(1000, 2, 5, 999).context("group should exist")?;
        assert_eq!(got.kernel, vec![100, 101]);
        assert_eq!(got.runtime, vec![300]);
        assert_eq!(got.sync, vec![400]);
        assert_eq!(got.graph, vec![500]);
        assert!(idx.lookup(1000, 2, 5, 888).is_none());
        Ok(())
    }

    #[test]
    fn lookup_by_runtime_resolves_via_process_map() -> Result<()> {
        let mut idx = CorrelationIndex::default();
        idx.process_to_contexts.insert(1000, vec![(3, 7)]);
        let syn = synthetic_id(1000, 3, 7, 42);
        idx.groups.insert(
            syn,
            CorrelatedRowIds {
                kernel: vec![55],
                ..Default::default()
            },
        );
        // Synthesize a globalTid with pid=1000, domain=0, tid=0
        let global_tid = 1000i64 << 24;
        let got = idx
            .lookup_by_runtime(global_tid, 42)
            .context("runtime lookup should resolve")?;
        assert_eq!(got.kernel, vec![55]);
        Ok(())
    }

    #[test]
    fn lookup_by_runtime_merges_multi_context() -> Result<()> {
        // A process with two CUDA contexts (e.g. multi-GPU run): runtime
        // walks must consider both candidates. With the bug, one of these
        // groups would be unreachable through lookup_by_runtime.
        let mut idx = CorrelationIndex::default();
        idx.process_to_contexts.insert(1000, vec![(0, 11), (1, 22)]);
        idx.groups.insert(
            synthetic_id(1000, 0, 11, 99),
            CorrelatedRowIds {
                kernel: vec![100],
                ..Default::default()
            },
        );
        idx.groups.insert(
            synthetic_id(1000, 1, 22, 99),
            CorrelatedRowIds {
                memcpy: vec![200],
                ..Default::default()
            },
        );

        let global_tid = 1000i64 << 24;
        let got = idx
            .lookup_by_runtime(global_tid, 99)
            .context("multi-context merge should produce a hit")?;
        assert_eq!(got.kernel, vec![100]);
        assert_eq!(got.memcpy, vec![200]);

        let ctxs = idx.contexts_for_pid(1000);
        assert!(ctxs.contains(&(0, 11)) && ctxs.contains(&(1, 22)));
        Ok(())
    }

    /// Verify a populated `CorrelationIndex` survives SidecarCache's
    /// bincode round-trip. The generic `SidecarCache::round_trip` test
    /// (in `veloq-core`) already covers the version-header + atomic
    /// rename path; this one pins the *payload shape* so a future
    /// `CorrelatedRowIds` field rename can't silently drop a bucket on
    /// warm callers.
    #[test]
    fn payload_round_trip_via_sidecar_cache() -> Result<()> {
        let dir = std::env::temp_dir().join(format!(
            "veloq-corr-cache-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir)?;
        let mut idx = CorrelationIndex::default();
        idx.contexts.insert((1000, 1, 1));
        idx.process_to_contexts.insert(1000, vec![(1, 1)]);
        idx.groups.insert(
            synthetic_id(1000, 1, 1, 7),
            CorrelatedRowIds {
                kernel: vec![10],
                memcpy: vec![20, 21],
                memset: vec![],
                runtime: vec![30],
                sync: vec![40],
                graph: vec![50],
            },
        );
        let cache: SidecarCache<CorrelationIndex> = SidecarCache::new(
            dir.join("correlation.bin"),
            CACHE_VERSION,
            "correlation cache",
        );
        let fp = SourceFingerprint {
            mtime_secs: 1234567,
            size: 999,
        };
        cache.write(fp, &idx)?;
        let back = cache
            .try_load(fp)?
            .context("just-written cache must load")?;
        assert_eq!(back.groups.len(), 1);
        let g = back
            .lookup(1000, 1, 1, 7)
            .context("group should survive round-trip")?;
        assert_eq!(g.kernel, vec![10]);
        assert_eq!(g.memcpy, vec![20, 21]);
        assert_eq!(g.sync, vec![40]);
        assert_eq!(g.graph, vec![50]);
        Ok(())
    }
}
