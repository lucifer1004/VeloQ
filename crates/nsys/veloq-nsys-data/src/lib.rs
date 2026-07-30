//! veloq-nsys-data — open an NSys trace via parquetdir and run small
//! metadata queries against it.
//!
//! veloq reads NSys traces only through the
//! parquetdir export (`nsys export -t parquetdir`). The
//! `<report>.veloq/parquetdir/<TABLE>.parquet` directory **is**
//! veloq's cache — no separate `veloq-parquet/` sidecar is built. SQLite
//! ingestion has been removed; the columnar-over-protobuf
//! principle stays intact, but the within-family choice is now parquet.
//!
//! ## Open lifecycle
//!
//! 1. `Trace::open(path)` accepts a `.nsys-rep` (auto-exported via
//!    `nsys_rep::resolve_trace`) or a `_pqtdir/` directory directly
//!    (test convenience and pre-exported workflows). The generated
//!    `<report>.veloq/parquetdir/` path is accepted as an alias for
//!    the owning `.nsys-rep`, not as a separate source identity.
//! 2. An in-memory DuckDB connection is opened with the `nsight`
//!    schema; for every `<TABLE>.parquet` file in the parquetdir, a
//!    view `nsight.<TABLE>` is created via `read_parquet(...)`.
//! 3. The schema adapter picks itself once and is cached on the
//!    handle.
//!
//! Heavier command implementations live in `veloq-nsys-query`; this
//! crate owns the `Trace` handle, sidecar caches, and the
//! `SchemaAdapter` trait.

pub mod adapter;
pub mod capabilities;
pub mod correlation;
pub mod cuda_identity;
pub mod error;
pub mod gpu_work;
pub mod gpu_work_events;
pub mod hardware;
pub mod meta_cache;
pub mod nsys_rep;
pub mod nvtx_nesting;
pub mod nvtx_stack;
pub mod nvtx_tree;
pub mod resident_identity;
pub mod runtime_nvtx_parent;
pub mod scope;
pub mod sidecar;
pub mod sidecar_registry;
pub mod sql_expr;
pub mod trace_map;

#[cfg(test)]
pub(crate) mod test_support;

pub use adapter::{
    AdapterChoice, AdapterMeta, AdapterStatus, DetectionMethod, SchemaAdapter, SchemaVersion,
    pick_adapter, table_exists,
};
pub use capabilities::CapabilityFlags;
pub use correlation::{CorrelatedRowIds, CorrelationIndex, CorrelationIndexStats, SyntheticId};
pub use cuda_identity::{
    CudaProcessResolver, ProcessSqlProjection, native_pid_from_global_tid, native_pid_sql,
    process_sql_projection,
};
pub use error::{DuckdbPhase, NsysDataError, NsysDataResult};
pub use gpu_work::{GPU_WORK_INTERVAL_COLUMNS, GPU_WORK_INTERVAL_KINDS, GpuWorkKind};
pub use gpu_work_events::{GPU_WORK_EVENTS_VERSION, GpuWorkEventRecord};
pub use hardware::{CpuInfo, DriverInfo, GpuInfo, HostInfo, NicInfo, SystemInfo};
pub use meta_cache::{META_CACHE_VERSION, PerTableEntry, TraceMetaCache};
pub use nvtx_nesting::{NvtxEntry, NvtxNesting};
pub use nvtx_tree::{NVTX_TREE_VERSION, NvtxTree, NvtxTreeRecord};
pub use resident_identity::{ResidentTraceIdentity, resident_trace_identity};
pub use runtime_nvtx_parent::{
    EnclosingNvtx, RUNTIME_NVTX_PARENT_VERSION, RuntimeNvtxParent, RuntimeParentEntry,
};
pub use sidecar_registry::{NsysSidecar, NsysSidecarStatus};
pub use veloq_core::time::TimeWindow;

use duckdb::Connection;
use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::UNIX_EPOCH;
use veloq_core::SourceFingerprint;

/// Tables veloq considers "primary execution": kernel/memcpy/memset/
/// runtime/sync. The trace's time origin for agent-facing
/// `--time-range` flags is derived from these (and excludes
/// OSRT/NVTX bootstrap markers, which NSys sometimes anchors to CUDA
/// driver init — landing them hundreds of seconds before any GPU work).
pub const PRIMARY_TABLES: &[&str] = &[
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "CUPTI_ACTIVITY_KIND_MEMSET",
    "CUPTI_ACTIVITY_KIND_RUNTIME",
    "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
];

/// All event tables veloq knows about. Used for the `full` span and for
/// per-table breakdowns.
pub const ALL_EVENT_TABLES: &[&str] = &[
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "CUPTI_ACTIVITY_KIND_MEMSET",
    "CUPTI_ACTIVITY_KIND_RUNTIME",
    "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
    "OSRT_API",
    "NVTX_EVENTS",
];

/// Sample/metric tables that carry time data but don't share the event-
/// table `start`/"end" column shape. Used only as a fallback in
/// [`Trace::read_origins`] when no event table contributed any rows —
/// e.g. `nsys profile --trace=none --nic-metrics=lf` captures, which
/// would otherwise anchor `--from`/`--to` at absolute zero. Each entry
/// is `(table, min_expr, max_expr)`; the exprs are spliced into SQL so
/// they must not include user input.
const SAMPLE_TABLE_SPECS: &[(&str, &str, &str)] = &[
    ("GPU_METRICS", "timestamp", "timestamp"),
    ("NET_NIC_METRIC", "start", r#"COALESCE("end", start)"#),
    ("COMPOSITE_EVENTS", "start", "start"),
    ("SCHED_EVENTS", "start", "start"),
];

/// Fingerprint the artifact NSys sidecars depend on.
///
/// `.nsys-rep` inputs keep the traditional source-file `(mtime, size)`
/// fingerprint. Direct `_pqtdir/` inputs need to follow their child
/// parquet files instead: rewriting `<TABLE>.parquet` in place may
/// leave the directory's own mtime/size unchanged, but derived
/// sidecars under `<input>.veloq/` must still invalidate.
pub(crate) fn trace_artifact_fingerprint(path: &Path) -> io::Result<SourceFingerprint> {
    if nsys_rep::is_parquetdir(path) && path.is_dir() {
        parquetdir_fingerprint(path)
    } else {
        SourceFingerprint::of_path(path)
    }
}

fn parquetdir_fingerprint(pqtdir: &Path) -> io::Result<SourceFingerprint> {
    let mut files = Vec::new();
    for entry in fs::read_dir(pqtdir)? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "parquet") {
            files.push(path);
        }
    }
    files.sort();

    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    hash_u64(&mut hash, files.len() as u64);
    let mut newest_mtime_secs = 0_i64;
    for path in files {
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            hash_u64(&mut hash, name.len() as u64);
            hash_bytes(&mut hash, name.as_bytes());
        }
        let meta = fs::metadata(&path)?;
        hash_u64(&mut hash, meta.len());
        if let Some((secs, nanos)) = modified_time_key(&meta) {
            newest_mtime_secs = newest_mtime_secs.max(secs);
            hash_i64(&mut hash, secs);
            hash_u32(&mut hash, nanos);
        }
    }

    Ok(SourceFingerprint {
        mtime_secs: newest_mtime_secs,
        size: hash,
    })
}

fn modified_time_key(meta: &fs::Metadata) -> Option<(i64, u32)> {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| (d.as_secs() as i64, d.subsec_nanos()))
}

fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *hash ^= u64::from(*b);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn hash_u64(hash: &mut u64, value: u64) {
    hash_bytes(hash, &value.to_le_bytes());
}

fn hash_i64(hash: &mut u64, value: i64) {
    hash_bytes(hash, &value.to_le_bytes());
}

fn hash_u32(hash: &mut u64, value: u32) {
    hash_bytes(hash, &value.to_le_bytes());
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct TraceOrigins {
    pub primary: TimeSpan,
    pub full: TimeSpan,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct TimeSpan {
    pub start_ns: i64,
    pub end_ns: i64,
}

impl TimeSpan {
    pub fn duration_ns(&self) -> i64 {
        self.end_ns - self.start_ns
    }
}

/// Handle to an opened NSys trace.
///
/// Wraps an in-memory DuckDB connection where every table the NSys
/// parquetdir export produced is exposed as a `nsight.<TABLE>` view
/// backed by `read_parquet('<pqtdir>/<TABLE>.parquet')`.
///
/// `source_path` carries the canonical source identity for derived
/// sidecars: either a `.nsys-rep` file or a direct `_pqtdir/` input.
/// Generated `<report>.veloq/parquetdir/` aliases resolve back to the
/// owning `.nsys-rep` so sidecars stay under one artifact root.
pub struct Trace {
    conn: Connection,
    /// Canonical source identity. Derived caches anchor under
    /// `<path>.veloq/`.
    source_path: PathBuf,
    /// Resolved parquetdir. For `.nsys-rep`, this is
    /// `<source>.veloq/parquetdir/`; for direct `_pqtdir/` input, this
    /// equals `source_path`.
    pqtdir_path: PathBuf,
    /// Tables present in the parquetdir (sorted). Computed once at
    /// open from a single `read_dir`.
    tables: Vec<String>,
    adapter: Arc<dyn SchemaAdapter>,
    detection_method: DetectionMethod,
    schema_version: Option<SchemaVersion>,
    meta_cache: OnceLock<TraceMetaCache>,
    query_worker_count: usize,
}

/// Trace-local Rayon pool enforcing the resolved query worker budget.
pub struct QueryWorkerPool {
    inner: rayon::ThreadPool,
}

impl QueryWorkerPool {
    pub fn install<OP, R>(&self, operation: OP) -> R
    where
        OP: FnOnce() -> R + Send,
        R: Send,
    {
        self.inner.install(operation)
    }
}

#[derive(Debug, Clone, Copy)]
struct QueryLimits {
    worker_threads: usize,
    memory_bytes: Option<u64>,
}

impl Trace {
    /// Open the trace.
    ///
    /// Accepts a `.nsys-rep` file (auto-exports to
    /// `<source>.veloq/parquetdir/`), an existing direct `_pqtdir/`
    /// directory, or the generated `<source>.veloq/parquetdir/` alias.
    /// Sets up the DuckDB `nsight` schema with views over every
    /// `<TABLE>.parquet` file in the parquetdir, then picks a schema
    /// adapter.
    pub fn open<P: AsRef<Path>>(path: P) -> NsysDataResult<Self> {
        Self::open_with_query_limits(path, None)
    }

    /// Open a daemon-resident trace with its worker budget and optional
    /// explicit query-memory ceiling.
    pub fn open_for_daemon<P: AsRef<Path>>(
        path: P,
        worker_threads: usize,
        memory_bytes: Option<u64>,
    ) -> NsysDataResult<Self> {
        Self::open_with_query_limits(
            path,
            Some(QueryLimits {
                worker_threads,
                memory_bytes,
            }),
        )
    }

    fn open_with_query_limits<P: AsRef<Path>>(
        path: P,
        query_limits: Option<QueryLimits>,
    ) -> NsysDataResult<Self> {
        let resolved = nsys_rep::resolve_trace(path.as_ref())?;
        let source_path = resolved.source_path;
        let pqtdir_path = resolved.pqtdir_path;
        let query_limits = query_limits.unwrap_or(QueryLimits {
            worker_threads: resolve_thread_count(),
            memory_bytes: None,
        });
        let (conn, tables) = open_nsight_duckdb(&pqtdir_path, query_limits)?;
        let choice = pick_adapter(&conn, &pqtdir_path)?;

        attach_nvtx_tree_view_if_present(&conn, &source_path)?;
        sidecar_registry::attach_optional_views(&conn, &source_path);

        Ok(Self {
            conn,
            source_path,
            pqtdir_path,
            tables,
            adapter: choice.adapter,
            detection_method: choice.method,
            schema_version: choice.schema_version,
            meta_cache: OnceLock::new(),
            query_worker_count: query_limits.worker_threads,
        })
    }

    /// Canonical source identity (`.nsys-rep` or direct `_pqtdir/`).
    /// Derived caches anchor under `<path>.veloq/`.
    pub fn path(&self) -> &Path {
        &self.source_path
    }

    /// Resolved parquetdir directory. Equals `path()` when the caller
    /// passed a parquetdir directly; otherwise
    /// `<source>.veloq/parquetdir/`.
    pub fn pqtdir_path(&self) -> &Path {
        &self.pqtdir_path
    }

    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    pub fn query_worker_count(&self) -> usize {
        self.query_worker_count
    }

    /// Build a local Rayon pool within this trace's machine-adaptive or
    /// explicitly configured worker budget.
    pub fn build_query_worker_pool(&self) -> NsysDataResult<QueryWorkerPool> {
        let inner = rayon::ThreadPoolBuilder::new()
            .num_threads(self.query_worker_count)
            .thread_name(|index| format!("veloq-query-{index}"))
            .build()
            .map_err(NsysDataError::query_worker_pool_build)?;
        Ok(QueryWorkerPool { inner })
    }

    /// Tables present in the parquetdir, sorted.
    pub fn tables(&self) -> &[String] {
        &self.tables
    }

    /// Does the trace contain `<table>.parquet`?
    pub fn has_table(&self, table: &str) -> bool {
        self.tables.iter().any(|t| t == table)
    }

    /// Path to `<table>.parquet` in the parquetdir. Callers that need
    /// to verify existence should call [`Self::has_table`] first.
    pub fn parquet_path(&self, table: &str) -> PathBuf {
        self.pqtdir_path.join(format!("{table}.parquet"))
    }

    /// Schema adapter picked on open.
    pub fn adapter(&self) -> &dyn SchemaAdapter {
        self.adapter.as_ref()
    }

    pub fn adapter_arc(&self) -> Arc<dyn SchemaAdapter> {
        Arc::clone(&self.adapter)
    }

    pub fn adapter_detection_method(&self) -> DetectionMethod {
        self.detection_method
    }

    pub fn schema_version(&self) -> Option<&SchemaVersion> {
        self.schema_version.as_ref()
    }

    /// Cheap "does this trace export this table?" check — alias for
    /// [`Self::has_table`] kept for call-site readability where the
    /// presence-of-parquet shape matters.
    pub fn table_exists(&self, table: &str) -> bool {
        self.has_table(table)
    }

    /// Does an attached NSys table expose a column?
    ///
    /// Table and column names come from VeloQ's internal schema inventory,
    /// never from user input. Probe failures conservatively report absence so
    /// optional-column paths can fall back to other identity evidence.
    pub fn table_has_column(&self, table: &str, column: &str) -> bool {
        self.conn
            .query_row(
                "SELECT COUNT(*) > 0 \
                 FROM information_schema.columns \
                 WHERE table_schema = 'nsight' \
                   AND table_name = ? \
                   AND column_name = ?",
                [table, column],
                |row| row.get(0),
            )
            .unwrap_or(false)
    }

    /// Resolve an optional caller-provided `TimeWindow` to absolute ns
    /// using this trace's primary origin.
    pub fn resolve_window(&self, w: Option<TimeWindow>) -> NsysDataResult<Option<(i64, i64)>> {
        let Some(w) = w else { return Ok(None) };
        let (origins, _) = self.read_origins()?;
        let resolved = w
            .absolute(origins.primary.start_ns)
            .map_err(NsysDataError::time_range_empty)?;
        Ok(Some(resolved))
    }

    /// List every table veloq can resolve in this trace — i.e. every
    /// `<TABLE>.parquet` file in the parquetdir.
    pub fn list_tables(&self) -> NsysDataResult<Vec<String>> {
        Ok(self.tables.clone())
    }

    pub fn correlation_index(&self) -> NsysDataResult<CorrelationIndex> {
        CorrelationIndex::build_or_load(self)
    }

    pub fn nvtx_nesting(&self) -> NsysDataResult<NvtxNesting> {
        if let Some(c) = self.meta_cache.get() {
            return Ok(c.nvtx_nesting.clone());
        }
        match meta_cache::try_load_existing(&self.source_path) {
            Ok(Some(c)) => {
                let nesting = c.nvtx_nesting.clone();
                let _ = self.meta_cache.set(c);
                return Ok(nesting);
            }
            Ok(None) => {}
            Err(e) => {
                log::debug!("meta cache unavailable for NVTX nesting reuse: {e:#}");
            }
        }
        self.compute_nvtx_nesting_uncached()
    }

    pub(crate) fn compute_nvtx_nesting_uncached(&self) -> NsysDataResult<NvtxNesting> {
        nvtx_nesting::compute(self)
    }

    pub fn meta_cache(&self) -> NsysDataResult<&TraceMetaCache> {
        if let Some(c) = self.meta_cache.get() {
            return Ok(c);
        }
        let built = meta_cache::build_or_load(self)?;
        let _ = self.meta_cache.set(built);
        self.meta_cache
            .get()
            .ok_or_else(NsysDataError::meta_cache_slot_uninitialised)
    }

    pub fn meta_cache_initialised(&self) -> bool {
        self.meta_cache.get().is_some()
    }

    /// Heap retained by lazily decoded source models beyond the base resident
    /// trace and DuckDB catalog estimate reserved at session admission.
    pub fn additional_resident_memory_estimate_bytes(&self) -> u64 {
        self.meta_cache
            .get()
            .map_or(0, TraceMetaCache::retained_heap_estimate_bytes)
    }

    /// Memory currently retained by DuckDB's own buffer manager.
    ///
    /// `duckdb_memory()` reports allocator-owned query-engine state by memory
    /// tag. It excludes profile bytes merely mapped or cached by the operating
    /// system, which keeps this estimate aligned with the daemon's explicit
    /// retained-memory boundary.
    pub fn query_engine_resident_memory_estimate_bytes(&self) -> NsysDataResult<u64> {
        let bytes = self
            .conn
            .query_row(
                "SELECT COALESCE(SUM(memory_usage_bytes), 0)::BIGINT FROM duckdb_memory()",
                [],
                |row| row.get::<_, i64>(0),
            )
            .map_err(NsysDataError::duckdb_resident_memory_read)?;
        Ok(u64::try_from(bytes).unwrap_or(u64::MAX))
    }

    /// Compute the trace's primary + full origins.
    pub fn read_origins(&self) -> NsysDataResult<(TraceOrigins, Vec<(&'static str, TimeSpan)>)> {
        let available: HashSet<&str> = self.tables.iter().map(String::as_str).collect();

        let mut per_table: Vec<(&'static str, TimeSpan)> = Vec::new();
        for &t in ALL_EVENT_TABLES {
            if !available.contains(t) {
                continue;
            }
            let sql =
                format!(r#"SELECT MIN(start), MAX(COALESCE("end", start)) FROM nsight."{t}""#);
            let mut stmt = self
                .conn
                .prepare(&sql)
                .map_err(|source| NsysDataError::trace_origins_prepare(t, source))?;
            let (mn, mx): (Option<i64>, Option<i64>) = stmt
                .query_row([], |r| Ok((r.get(0)?, r.get(1)?)))
                .map_err(|source| NsysDataError::trace_origins_read(t, source))?;
            if let (Some(start_ns), Some(end_ns)) = (mn, mx) {
                per_table.push((t, TimeSpan { start_ns, end_ns }));
            }
        }

        let combine = |tables: &[&'static str]| -> Option<TimeSpan> {
            let spans: Vec<TimeSpan> = per_table
                .iter()
                .filter(|(t, _)| tables.contains(t))
                .map(|(_, s)| *s)
                .collect();
            if spans.is_empty() {
                return None;
            }
            Some(TimeSpan {
                start_ns: spans.iter().map(|s| s.start_ns).min().unwrap_or(0),
                end_ns: spans.iter().map(|s| s.end_ns).max().unwrap_or(0),
            })
        };

        let event_primary = combine(PRIMARY_TABLES);
        let event_full = combine(ALL_EVENT_TABLES);

        // Metric/sample-only captures (e.g. `nsys profile --trace=none
        // --nic-metrics=lf`) have no event-table rows but still carry
        // real time-stamped samples.
        let sample_full = if event_full.is_none() {
            self.read_sample_table_span(&available)?
        } else {
            None
        };

        let full = event_full.or(sample_full).unwrap_or(TimeSpan {
            start_ns: 0,
            end_ns: 0,
        });
        let primary = event_primary.unwrap_or(full);

        Ok((TraceOrigins { primary, full }, per_table))
    }

    fn read_sample_table_span(
        &self,
        available: &HashSet<&str>,
    ) -> NsysDataResult<Option<TimeSpan>> {
        let mut lo: i64 = i64::MAX;
        let mut hi: i64 = i64::MIN;
        let mut any = false;
        for (t, min_col, max_expr) in SAMPLE_TABLE_SPECS {
            if !available.contains(t) {
                continue;
            }
            // Safe to splice: `min_col` / `max_expr` are static
            // string constants from `SAMPLE_TABLE_SPECS`, not user
            // input.
            let sql = format!(r#"SELECT MIN({min_col}), MAX({max_expr}) FROM nsight."{t}""#);
            let mut stmt = self
                .conn
                .prepare(&sql)
                .map_err(|source| NsysDataError::trace_sample_span_prepare(*t, source))?;
            let (mn, mx): (Option<i64>, Option<i64>) = stmt
                .query_row([], |r| Ok((r.get(0)?, r.get(1)?)))
                .map_err(|source| NsysDataError::trace_sample_span_read(*t, source))?;
            if let (Some(a), Some(b)) = (mn, mx) {
                lo = lo.min(a);
                hi = hi.max(b);
                any = true;
            }
        }
        Ok(if any {
            Some(TimeSpan {
                start_ns: lo,
                end_ns: hi,
            })
        } else {
            None
        })
    }

    /// Read NSys export metadata (schema version, product version, etc).
    pub fn read_export_metadata(&self) -> NsysDataResult<Vec<(String, String)>> {
        const TABLE: &str = "META_DATA_EXPORT";
        if !self.has_table(TABLE) {
            log::debug!("META_DATA_EXPORT absent — export metadata unavailable");
            return Ok(Vec::new());
        }
        let mut stmt = self
            .conn
            .prepare("SELECT name, value FROM nsight.META_DATA_EXPORT")
            .map_err(|source| NsysDataError::export_metadata_prepare(TABLE, source))?;
        let mut rows = stmt
            .query([])
            .map_err(|source| NsysDataError::export_metadata_query(TABLE, source))?;
        let mut out = Vec::new();
        while let Some(row) = rows
            .next()
            .map_err(|source| NsysDataError::export_metadata_read(TABLE, source))?
        {
            let k: String = row
                .get(0)
                .map_err(|source| NsysDataError::export_metadata_read(TABLE, source))?;
            let v: duckdb::types::Value = row
                .get(1)
                .map_err(|source| NsysDataError::export_metadata_read(TABLE, source))?;
            out.push((k, value_to_string(&v)));
        }
        Ok(out)
    }
}

/// Resolve veloq's DuckDB worker-thread count: `VELOQ_DUCKDB_THREADS`
/// (a positive integer) wins for benchmarking/overrides, otherwise the
/// shared query-engine default applies.
fn resolve_thread_count() -> usize {
    if let Some(n) = std::env::var("VELOQ_DUCKDB_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n >= 1)
    {
        return n;
    }
    veloq_core::default_query_worker_count()
}

/// The thread count is a veloq-internal integer, never user input, so the
/// `PRAGMA` literal is safe.
/// Open a fresh in-memory DuckDB, create the `nsight` schema, and
/// register a view for every `<TABLE>.parquet` in the parquetdir.
/// Returns the connection plus the sorted list of table names that
/// got a view.
fn open_nsight_duckdb(
    pqtdir: &Path,
    query_limits: QueryLimits,
) -> NsysDataResult<(Connection, Vec<String>)> {
    if !pqtdir.is_dir() {
        return Err(NsysDataError::parquetdir_not_found(pqtdir.display()));
    }
    let conn = Connection::open_in_memory().map_err(NsysDataError::duckdb_open_in_memory)?;
    conn.execute_batch(&format!("PRAGMA threads={}", query_limits.worker_threads))
        .map_err(NsysDataError::duckdb_thread_config)?;
    if let Some(memory_bytes) = query_limits.memory_bytes {
        conn.execute_batch(&format!("SET memory_limit = '{memory_bytes}B'"))
            .map_err(NsysDataError::duckdb_memory_config)?;
    }
    conn.execute_batch("CREATE SCHEMA IF NOT EXISTS nsight")
        .map_err(NsysDataError::duckdb_schema_create)?;

    let mut parquet_paths: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(pqtdir)
        .map_err(|source| NsysDataError::parquetdir_read(pqtdir.display(), source))?
    {
        let entry =
            entry.map_err(|source| NsysDataError::parquetdir_read(pqtdir.display(), source))?;
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "parquet") {
            parquet_paths.push(path);
        }
    }
    parquet_paths.sort();

    let mut tables: Vec<String> = Vec::with_capacity(parquet_paths.len());
    for parquet_path in &parquet_paths {
        let Some(stem) = parquet_path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(str::to_owned)
        else {
            continue;
        };
        let path_lit = parquet_path
            .to_str()
            .ok_or_else(|| NsysDataError::parquet_path_invalid_utf8(parquet_path.display()))?
            .replace('\'', "''");
        // `file_row_number AS rowid` gives every parquet row a
        // deterministic per-table identifier exposed through the JSON
        // envelope as `RowId.rowid`. Add 1 so the wire identifiers stay
        // 1-based and SQLite-compatible for scripts and rebuilt
        // sidecars.
        let table_ident = quote_sql_identifier(&stem);
        let sql = format!(
            "CREATE OR REPLACE VIEW nsight.{table_ident} AS \
             SELECT (file_row_number + 1) AS rowid, * \
             FROM read_parquet('{path_lit}', file_row_number = true)"
        );
        conn.execute(&sql, []).map_err(|source| {
            NsysDataError::parquet_view_create(stem.as_str(), parquet_path.display(), source)
        })?;
        tables.push(stem);
    }

    Ok((conn, tables))
}

pub(crate) fn quote_sql_identifier(identifier: &str) -> String {
    format!("\"{}\"", identifier.replace('"', "\"\""))
}

/// Register `nsight.nvtx_tree` over the sidecar parquet if one already
/// exists fresh on disk. No-op when the sidecar is absent or stale;
/// `open` must stay cheap, so this never triggers a sidecar build.
fn attach_nvtx_tree_view_if_present(conn: &Connection, source_path: &Path) -> NsysDataResult<()> {
    let sidecar = nvtx_tree::sidecar_path_for(source_path);
    if !sidecar.exists() || !nvtx_tree::sidecar_is_fresh_for_trace(source_path)? {
        return Ok(());
    }
    let Some(sql) = nvtx_tree::view_sql_for(&sidecar) else {
        log::warn!(
            "nvtx_tree: sidecar path is not valid UTF-8, skipping view registration: {}",
            sidecar.display(),
        );
        return Ok(());
    };
    conn.execute(&sql, [])
        .map_err(|source| NsysDataError::nvtx_tree_view_register(sidecar.display(), source))?;
    Ok(())
}

fn value_to_string(v: &duckdb::types::Value) -> String {
    use duckdb::types::Value;
    match v {
        Value::Null => String::new(),
        Value::Boolean(b) => b.to_string(),
        Value::TinyInt(i) => i.to_string(),
        Value::SmallInt(i) => i.to_string(),
        Value::Int(i) => i.to_string(),
        Value::BigInt(i) => i.to_string(),
        Value::HugeInt(i) => i.to_string(),
        Value::UTinyInt(i) => i.to_string(),
        Value::USmallInt(i) => i.to_string(),
        Value::UInt(i) => i.to_string(),
        Value::UBigInt(i) => i.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Double(f) => f.to_string(),
        Value::Text(s) => s.clone(),
        other => format!("{other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use std::path::PathBuf;
    use tempfile::TempDir;
    use veloq_core::VeloqDiagnostic;

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

    fn valid_empty_kernel() -> (&'static str, &'static str, Vec<&'static str>) {
        (
            "CUPTI_ACTIVITY_KIND_KERNEL",
            r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start BIGINT, "end" BIGINT)"#,
            Vec::new(),
        )
    }

    #[test]
    fn resolve_window_empty_range_error_is_typed() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture(vec![valid_empty_kernel()])?;
        let trace = Trace::open(&pqtdir)?;
        let window = TimeWindow::parse("10ns-5ns")?;

        let err = match trace.resolve_window(Some(window)) {
            Ok(resolved) => anyhow::bail!("inverted window should not resolve: {resolved:?}"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.data.time-range-empty");
        match err {
            NsysDataError::TimeRangeEmpty {
                source: veloq_core::time::TimeParseError::EmptyRange { start: 10, end: 5 },
            } => {}
            other => anyhow::bail!("expected TimeRangeEmpty with EmptyRange, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn open_invalid_parquet_file_error_is_typed() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let pqtdir = dir.path().join("bad_pqtdir");
        std::fs::create_dir_all(&pqtdir)?;
        std::fs::write(
            pqtdir.join("CUPTI_ACTIVITY_KIND_KERNEL.parquet"),
            b"not a parquet file",
        )?;

        let err = match Trace::open(&pqtdir) {
            Ok(_) => anyhow::bail!("invalid parquet file should fail while creating views"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.data.parquet-view-create");
        match err {
            NsysDataError::ParquetViewCreate { table, path, .. } => {
                assert_eq!(table, "CUPTI_ACTIVITY_KIND_KERNEL");
                assert!(path.contains("CUPTI_ACTIVITY_KIND_KERNEL.parquet"));
            }
            other => anyhow::bail!("expected ParquetViewCreate, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn open_parquet_filename_with_quote_creates_quoted_view() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture(vec![valid_empty_kernel()])?;
        let src = pqtdir.join("CUPTI_ACTIVITY_KIND_KERNEL.parquet");
        let quoted_table = "ODD\"TABLE";
        std::fs::copy(&src, pqtdir.join(format!("{quoted_table}.parquet")))?;

        let trace = Trace::open(&pqtdir)?;

        assert!(trace.has_table(quoted_table));
        assert_eq!(quote_sql_identifier(quoted_table), r#""ODD""TABLE""#);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn open_non_utf8_parquet_path_error_is_typed() -> Result<()> {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let dir = tempfile::tempdir()?;
        let pqtdir = dir
            .path()
            .join(OsString::from_vec(vec![b'p', b'q', b't', 0xff]));
        std::fs::create_dir_all(&pqtdir)?;
        std::fs::write(
            pqtdir.join("CUPTI_ACTIVITY_KIND_KERNEL.parquet"),
            b"not a parquet file",
        )?;

        let err = match Trace::open(&pqtdir) {
            Ok(_) => anyhow::bail!("non-UTF-8 parquet path should fail before view creation"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.data.parquet-path-invalid-utf8");
        match err {
            NsysDataError::ParquetPathInvalidUtf8 { path } => {
                assert!(path.contains("CUPTI_ACTIVITY_KIND_KERNEL.parquet"));
            }
            other => anyhow::bail!("expected ParquetPathInvalidUtf8, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn read_export_metadata_prepare_error_is_typed() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture(vec![
            valid_empty_kernel(),
            (
                "META_DATA_EXPORT",
                "CREATE TABLE META_DATA_EXPORT (name TEXT)",
                Vec::new(),
            ),
        ])?;
        let trace = Trace::open(&pqtdir)?;

        let err = match trace.read_export_metadata() {
            Ok(meta) => anyhow::bail!("malformed META_DATA_EXPORT should fail: {meta:?}"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.data.duckdb-prepare");
        assert_eq!(
            err.duckdb_parts(),
            Some(("export metadata", DuckdbPhase::Prepare, "META_DATA_EXPORT"))
        );
        Ok(())
    }

    #[test]
    fn read_export_metadata_read_error_is_typed() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture(vec![
            valid_empty_kernel(),
            (
                "META_DATA_EXPORT",
                "CREATE TABLE META_DATA_EXPORT (name BLOB, value TEXT)",
                vec!["INSERT INTO META_DATA_EXPORT (name, value) VALUES (BLOB 'abc', '1')"],
            ),
        ])?;
        let trace = Trace::open(&pqtdir)?;

        let err = match trace.read_export_metadata() {
            Ok(meta) => anyhow::bail!("wrong-typed META_DATA_EXPORT should fail: {meta:?}"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.data.duckdb-read");
        assert_eq!(
            err.duckdb_parts(),
            Some(("export metadata", DuckdbPhase::Read, "META_DATA_EXPORT"))
        );
        Ok(())
    }

    #[test]
    fn read_origins_event_table_prepare_error_is_typed() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture(vec![
            valid_empty_kernel(),
            (
                "OSRT_API",
                "CREATE TABLE OSRT_API (start BIGINT)",
                Vec::new(),
            ),
        ])?;
        let trace = Trace::open(&pqtdir)?;

        let err = match trace.read_origins() {
            Ok((origins, _)) => anyhow::bail!("malformed OSRT_API should fail: {origins:?}"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.data.duckdb-prepare");
        assert_eq!(
            err.duckdb_parts(),
            Some(("trace origins", DuckdbPhase::Prepare, "OSRT_API"))
        );
        Ok(())
    }

    #[test]
    fn read_origins_event_table_read_error_is_typed() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture(vec![
            valid_empty_kernel(),
            (
                "OSRT_API",
                r#"CREATE TABLE OSRT_API (start TEXT, "end" TEXT)"#,
                vec![r#"INSERT INTO OSRT_API (start, "end") VALUES ('a', 'b')"#],
            ),
        ])?;
        let trace = Trace::open(&pqtdir)?;

        let err = match trace.read_origins() {
            Ok((origins, _)) => anyhow::bail!("text OSRT_API span should fail: {origins:?}"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.data.duckdb-read");
        assert_eq!(
            err.duckdb_parts(),
            Some(("trace origins", DuckdbPhase::Read, "OSRT_API"))
        );
        Ok(())
    }

    #[test]
    fn read_origins_sample_table_prepare_error_is_typed() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture(vec![
            valid_empty_kernel(),
            (
                "GPU_METRICS",
                "CREATE TABLE GPU_METRICS (typeId BIGINT, metricId BIGINT, value DOUBLE)",
                Vec::new(),
            ),
        ])?;
        let trace = Trace::open(&pqtdir)?;

        let err = match trace.read_origins() {
            Ok((origins, _)) => anyhow::bail!("malformed GPU_METRICS should fail: {origins:?}"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.data.duckdb-prepare");
        assert_eq!(
            err.duckdb_parts(),
            Some(("trace sample span", DuckdbPhase::Prepare, "GPU_METRICS"))
        );
        Ok(())
    }

    #[test]
    fn read_origins_sample_table_read_error_is_typed() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture(vec![
            valid_empty_kernel(),
            (
                "GPU_METRICS",
                "CREATE TABLE GPU_METRICS (timestamp TEXT)",
                vec!["INSERT INTO GPU_METRICS (timestamp) VALUES ('not-ns')"],
            ),
        ])?;
        let trace = Trace::open(&pqtdir)?;

        let err = match trace.read_origins() {
            Ok((origins, _)) => anyhow::bail!("text GPU_METRICS span should fail: {origins:?}"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.data.duckdb-read");
        assert_eq!(
            err.duckdb_parts(),
            Some(("trace sample span", DuckdbPhase::Read, "GPU_METRICS"))
        );
        Ok(())
    }
}
