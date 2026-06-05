//! Per-trace NVTX-parent attribution index.
//!
//! For every runtime call on a host thread that sits inside one or
//! more NVTX ranges, this index records the full outermost→innermost
//! list of enclosing ranges. Runtime rows whose interval is not
//! contained by any NVTX range are simply absent — the sentinel "no
//! NVTX parent" is the LEFT-JOIN-miss state on the SQL side and the
//! `HashMap::get → None` state on the Rust side.
//!
//! ## Why a per-trace index
//!
//! The naive recipe (`runtime × NVTX_EVENTS` containment join +
//! per-event `ROW_NUMBER` window pick) is `O(N_runtime × N_nvtx/T)`
//! comparisons inside DuckDB. On a 21.8M-kernel / 2M-runtime / 34K-NVTX
//! production trace that's minutes. Sorting NVTX per-thread once and
//! binary-searching each runtime row brings the work down to
//! `O((N_n + N_r) log + N_r × depth)` — single-digit seconds.
//!
//! ## Shape of the cached artifact
//!
//! `<trace>.veloq/nvtx-parent.parquet` — one or more rows per
//! attributed runtime call (multi-context fan-out emits one row per
//! `(device, context)` candidate). SNAPPY-compressed, single row
//! group. Schema:
//!
//! | column           | type             | notes                                |
//! | ---------------- | ---------------- | ------------------------------------ |
//! | `rt_rowid`       | INT64            | `CUPTI_ACTIVITY_KIND_RUNTIME.rowid`. Not unique across rows when fan-out fired — collapse via `arbitrary(...)` / GROUP BY when reading runtime-side joins. |
//! | `correlation_id` | INT64 (nullable) | join key for kernel/memcpy/memset/sync; NULL when the runtime call has no CUDA correlation (e.g. `cudaGetDeviceCount`). Such rows are still attributed for runtime-side NVTX containment via `rt_rowid` lookup. |
//! | `native_pid`     | INT64            | derived from `runtime.globalTid >> 24` |
//! | `device_id`      | INT32 (nullable) | runtime row's resolved CUDA device, from the corresponding GPU activity. NULL when no GPU activity exists for this correlation or `TARGET_INFO_CUDA_CONTEXT_INFO` was absent at build time. |
//! | `context_id`    | INT64 (nullable) | runtime row's resolved CUDA context — same conditions as `device_id`. |
//! | `nvtx_rowids`    | LIST<INT64>      | outermost→innermost enclosing rowids |
//! | `nvtx_names`     | LIST<VARCHAR>    | outermost→innermost enclosing names  |
//!
//! Freshness/atomic publish via [`crate::sidecar`]; the version key is
//! `veloq.runtime_nvtx_parent.version` ([`RUNTIME_NVTX_PARENT_VERSION`]).
//!
//! ## Two directions, one sidecar
//!
//! Both attribution directions consume the same artifact:
//!
//! - **Reverse** (`inspect`, `search --with-nvtx`,
//!   `stats --group-by nvtx-parent`): "what's the innermost NVTX
//!   range covering this event?" → read `nvtx_rowids[-1]` /
//!   `nvtx_names[-1]` (last element).
//! - **Forward** (`stats --nvtx <pattern>`,
//!   `search --nvtx <pattern>`, `timeline --nvtx <pattern>`, `slices`):
//!   "which events were inside *any* NVTX range matching this
//!   pattern?" → `UNNEST(nvtx_names)` then
//!   `WHERE name LIKE …`. The all-enclosing list is necessary because
//!   a typical NVTX layout nests outer scopes around inner ones, and
//!   the user pattern targets a level that's frequently not the
//!   innermost.

use crate::Trace;
use anyhow::{Context, Result};
use arrow::array::{
    Array, ArrayRef, Int8Array, Int16Array, Int32Array, Int64Array, Int64Builder, ListArray,
    ListBuilder, StringArray, StringBuilder, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use veloq_core::SourceFingerprint;

/// Bump on every breaking schema change to the parquet sidecar (column
/// rename/removal, new mandatory column, fingerprint metadata format
/// change). Mismatched versions rebuild silently on next open.
///
/// Two load-bearing correctness properties of the current schema:
///   * `correlation_id` is nullable. A non-nullable column would filter
///     out runtime rows with `correlationId IS NULL`, dropping runtime
///     calls that don't emit GPU work (e.g. `cudaGetDeviceCount`) from
///     `--type runtime --group-by nvtx-parent`. Those rows are kept and
///     attribute via the `rt_rowid` map.
///   * Nullable `device_id` / `context_id` columns. Per the repo's
///     correlation model ([`crate::correlation`]) the
///     disambiguator for raw `correlationId` is
///     `(device_id, context_id, correlation_id)`, not
///     `(correlation_id, native_pid)`. Storing the device/context at
///     build time both matches that model and lets every query-time SQL
///     path drop the `ctx_for_pid` bridge through
///     `TARGET_INFO_CUDA_CONTEXT_INFO` — the GPU row's
///     `(deviceId, contextId, correlationId)` joins the sidecar directly.
pub const RUNTIME_NVTX_PARENT_VERSION: u32 = 1;

/// One enclosing NVTX range on the path from outermost to innermost
/// for a given runtime row. Owned `String` because the index outlives
/// the source rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnclosingNvtx {
    pub nvtx_rowid: i64,
    pub nvtx_name: String,
}

/// One mapping in the index. `enclosing` is sorted outermost-first so
/// the innermost range is `.last()`. A non-attributed runtime row
/// never appears in the index — the empty-enclosing case is *not* a
/// valid record.
///
/// `correlation_id` is `None` for runtime calls that don't emit GPU
/// work (e.g. `cudaGetDeviceCount`, `cudaDeviceCanAccessPeer`). Such
/// rows still attribute to NVTX ranges (containment is on
/// `globalTid`/start/end, not correlation), but they can't be the
/// target of a GPU-side reverse lookup.
///
/// `device_id` / `context_id` come from the corresponding GPU
/// activity (kernel/memcpy/memset/sync) whose `correlationId` matches
/// the runtime row's. Both are `None` when:
/// - the runtime call had no GPU activity (NULL correlation_id), OR
/// - the corresponding GPU activity isn't in the trace, OR
/// - `TARGET_INFO_CUDA_CONTEXT_INFO` was missing (so the disambiguating
///   bridge between `(device, context)` and `native_pid` couldn't run).
///
/// The [`RuntimeNvtxParent::by_correlation`] map only contains entries
/// where all three of `(device_id, context_id, correlation_id)` are
/// `Some` — that trio is the documented unique key per
/// [`crate::correlation`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeParentEntry {
    pub rt_rowid: i64,
    pub correlation_id: Option<i64>,
    pub native_pid: i64,
    pub device_id: Option<i32>,
    pub context_id: Option<i64>,
    pub enclosing: Vec<EnclosingNvtx>,
}

impl RuntimeParentEntry {
    /// Innermost (deepest) enclosing range. `None` would mean the
    /// record is malformed — sidecar build never emits an entry with
    /// no enclosing ranges. Callers should treat `None` defensively
    /// even so (returning a "no NVTX parent" result).
    pub fn innermost(&self) -> Option<&EnclosingNvtx> {
        self.enclosing.last()
    }

    /// Does any enclosing range's name satisfy `pred`? Used by the
    /// Rust-side forward filter when callers prefer not to round-trip
    /// through DuckDB (e.g. small test-runner queries).
    pub fn any_enclosing_name<F: Fn(&str) -> bool>(&self, pred: F) -> bool {
        self.enclosing.iter().any(|e| pred(&e.nvtx_name))
    }
}

/// In-memory NVTX-parent attribution index for a trace.
///
/// Two lookup paths so callers don't pay for the wrong key:
/// - `by_rt_rowid` — used when the caller has a runtime row (e.g.
///   `inspect runtime:N`, `correlate runtime:N`).
/// - `by_correlation` — used when the caller has a GPU-side row
///   (kernel/memcpy/memset/sync); keyed by the documented
///   disambiguator `(device_id, context_id, correlation_id)` so a
///   GPU row brings all three directly (no `ctx_for_pid` bridge
///   needed at lookup time).
///
/// Both maps share owned data via `Arc<RuntimeParentEntry>` so the
/// memory footprint is ~1× the underlying records regardless of how
/// many lookup paths surface.
pub struct RuntimeNvtxParent {
    by_rt_rowid: HashMap<i64, Arc<RuntimeParentEntry>>,
    by_correlation: HashMap<(i32, i64, i64), Arc<RuntimeParentEntry>>,
}

impl RuntimeNvtxParent {
    pub fn empty() -> Self {
        Self {
            by_rt_rowid: HashMap::new(),
            by_correlation: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.by_rt_rowid.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_rt_rowid.is_empty()
    }

    /// Parent chain of the runtime row at `rt_rowid`, if any.
    pub fn get_by_runtime(&self, rt_rowid: i64) -> Option<&RuntimeParentEntry> {
        self.by_rt_rowid.get(&rt_rowid).map(|a| a.as_ref())
    }

    /// Parent chain of any GPU-side event keyed by the disambiguating
    /// trio `(device_id, context_id, correlation_id)`. Per the
    /// repo's correlation model raw `correlationId` is only unique
    /// within `(device, context)`, so the GPU row's
    /// `(deviceId, contextId, correlationId)` is the natural lookup
    /// key — no `ctx_for_pid` bridge needed.
    pub fn get_by_correlation(
        &self,
        device_id: i32,
        context_id: i64,
        correlation_id: i64,
    ) -> Option<&RuntimeParentEntry> {
        self.by_correlation
            .get(&(device_id, context_id, correlation_id))
            .map(|a| a.as_ref())
    }

    fn from_records(records: Vec<RuntimeParentEntry>) -> Self {
        let mut by_rt_rowid: HashMap<i64, Arc<RuntimeParentEntry>> =
            HashMap::with_capacity(records.len());
        // GPU-side reverse lookup needs the full disambiguator trio.
        // Only entries with all three of `(device_id, context_id,
        // correlation_id)` Some populate it. Runtime-side lookups go
        // through `by_rt_rowid` which holds every attributed runtime
        // regardless of whether it has GPU activity.
        let mut by_correlation: HashMap<(i32, i64, i64), Arc<RuntimeParentEntry>> =
            HashMap::with_capacity(records.len());
        for r in records {
            let arc = Arc::new(r);
            by_rt_rowid.insert(arc.rt_rowid, Arc::clone(&arc));
            if let (Some(dev), Some(ctx), Some(corr)) =
                (arc.device_id, arc.context_id, arc.correlation_id)
            {
                by_correlation.insert((dev, ctx, corr), arc);
            }
        }
        Self {
            by_rt_rowid,
            by_correlation,
        }
    }
}

/// Filesystem path of the parquet sidecar for `trace_path`.
pub fn sidecar_path_for(trace_path: &Path) -> PathBuf {
    veloq_core::artifact_dir_for(trace_path).join("nvtx-parent.parquet")
}

/// Build the sidecar if missing or stale; return its path. The SQL
/// `read_parquet(…)` join in every NVTX-bearing verb consumes the
/// path directly.
///
/// Side effect: when the sidecar is rebuilt, the records are computed
/// and persisted; the in-memory index is *not* cached in the `Trace`
/// from this call. Callers that need the in-memory map should use
/// [`build_or_load_index`].
pub fn ensure_sidecar(trace: &Trace) -> Result<PathBuf> {
    let path = sidecar_path_for(trace.path());
    let fp = crate::trace_artifact_fingerprint(trace.path()).with_context(|| {
        format!(
            "stat trace `{}` for nvtx-parent fingerprint",
            trace.path().display()
        )
    })?;
    if sidecar_is_fresh(&path, fp)? {
        log::debug!(
            "runtime_nvtx_parent: warm sidecar at {} ({} bytes)",
            path.display(),
            fs::metadata(&path).map(|m| m.len()).unwrap_or(0),
        );
        return Ok(path);
    }
    let records = compute(trace).context("computing runtime→NVTX-parent map")?;
    write_parquet(&path, fp, &records)?;
    log::info!(
        "runtime_nvtx_parent: built sidecar at {} ({} entries)",
        path.display(),
        records.len()
    );
    Ok(path)
}

/// Build the sidecar (if missing or stale) and return the in-memory
/// [`RuntimeNvtxParent`] index. Callers that only need the SQL path
/// (e.g. `stats --group-by nvtx-parent`) should call [`ensure_sidecar`]
/// instead and let DuckDB scan the parquet directly.
pub fn build_or_load_index(trace: &Trace) -> Result<RuntimeNvtxParent> {
    let path = ensure_sidecar(trace)?;
    if !path.exists() {
        return Ok(RuntimeNvtxParent::empty());
    }
    let records = read_parquet(&path).context("loading nvtx-parent sidecar")?;
    Ok(RuntimeNvtxParent::from_records(records))
}

/// Load the in-memory index **only if** a fresh sidecar already exists
/// on disk; never trigger a build. Returns `Ok(None)` when the sidecar
/// is missing or stale, leaving the build decision to the caller.
///
/// Use this from cheap single-row verbs (e.g. `inspect kernel:N`) so
/// a cold cache doesn't force a multi-second build just to decorate
/// one row. Batched verbs (`search --with-nvtx`, `stats --group-by
/// nvtx-parent`) call [`build_or_load_index`] / [`ensure_sidecar`]
/// instead because they amortise the build cost across many lookups.
pub fn load_if_present(trace: &Trace) -> Result<Option<RuntimeNvtxParent>> {
    let path = sidecar_path_for(trace.path());
    let fp = crate::trace_artifact_fingerprint(trace.path()).with_context(|| {
        format!(
            "stat trace `{}` for nvtx-parent fingerprint",
            trace.path().display()
        )
    })?;
    if !sidecar_is_fresh(&path, fp)? {
        return Ok(None);
    }
    let records = read_parquet(&path).context("loading nvtx-parent sidecar")?;
    Ok(Some(RuntimeNvtxParent::from_records(records)))
}

// ----- sweep ---------------------------------------------------------------

#[derive(Debug, Clone)]
struct NvtxRangeRow {
    rowid: i64,
    start: i64,
    end: i64,
    name: String,
}

#[derive(Debug)]
struct RuntimeRow {
    rowid: i64,
    correlation_id: Option<i64>,
    native_pid: i64,
    global_tid: i64,
    start: i64,
    end: i64,
    /// Derived from the matching GPU activity (kernel/memcpy/memset/
    /// sync) via the `(correlationId, native_pid)` join, where
    /// `native_pid` is mapped back to `(deviceId, contextId)` through
    /// `TARGET_INFO_CUDA_CONTEXT_INFO`. `None` when the runtime call
    /// has no GPU activity, no CUDA correlation, or the trace lacks
    /// the context-info table.
    device_id: Option<i32>,
    context_id: Option<i64>,
}

/// Compute the parent map from scratch. Same algorithm whether we're
/// building the sidecar on cold open or rebuilding after a trace edit.
fn compute(trace: &Trace) -> Result<Vec<RuntimeParentEntry>> {
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
/// Fan-out keeps the trio-keyed `by_correlation` map unambiguous on
/// every legitimate GPU row: each kernel/memcpy/memset/sync brings
/// its own `(deviceId, contextId, correlationId)` and only ever
/// joins the one sidecar entry that matches. Runtime-side
/// `by_rt_rowid` collapses the fanout (the enclosing chain is
/// identical across copies, so any copy is correct for that map's
/// purpose).
fn merge_dev_ctx(walked: Vec<RuntimeParentEntry>, dev_ctx: &DevCtxMap) -> Vec<RuntimeParentEntry> {
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

/// All GPU activity tables that can populate the
/// `(native_pid, correlation_id) → (device_id, context_id)` map. The
/// runtime row's enclosing context comes from whichever of these
/// emitted the matching `correlationId`.
const GPU_ACTIVITY_TABLES: &[&str] = &[
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "CUPTI_ACTIVITY_KIND_MEMSET",
    "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
];

/// `(native_pid, correlation_id) → all matching (device_id, context_id)`
/// pairs found across the GPU activity tables.
///
/// Most `(native_pid, correlationId)` keys resolve to exactly one
/// `(device, context)`, so the common-case storage is `Single`
/// (inline, no allocation). When CUPTI presents multiple
/// `(device, context)` for the same key — a multi-context-clash
/// artifact — the entry promotes to `Many` and the merge step fans
/// out: one sidecar entry per candidate `(device, context)`.
///
/// Promotion is one-way; once `Many`, the entry stays `Many` even if
/// later inserts dedup. The expected churn is tiny so the asymmetric
/// transition is fine.
#[derive(Debug, Clone, PartialEq, Eq)]
enum DevCtxValue {
    Single((i32, i64)),
    Many(Vec<(i32, i64)>),
}

impl DevCtxValue {
    /// True if `dx` is already represented in this value.
    fn contains(&self, dx: &(i32, i64)) -> bool {
        match self {
            DevCtxValue::Single(existing) => existing == dx,
            DevCtxValue::Many(v) => v.contains(dx),
        }
    }

    /// Add `dx` if not already present, promoting `Single → Many` on
    /// the first divergent insert.
    fn push(&mut self, dx: (i32, i64)) {
        if self.contains(&dx) {
            return;
        }
        match self {
            DevCtxValue::Single(existing) => {
                *self = DevCtxValue::Many(vec![*existing, dx]);
            }
            DevCtxValue::Many(v) => v.push(dx),
        }
    }

    /// View as a slice for the merge step's dispatch.
    fn as_slice(&self) -> &[(i32, i64)] {
        match self {
            DevCtxValue::Single(x) => std::slice::from_ref(x),
            DevCtxValue::Many(v) => v.as_slice(),
        }
    }
}

type DevCtxMap = HashMap<(i64, i64), DevCtxValue>;

/// Collect `(native_pid, correlation_id) → (device_id, context_id)`.
///
/// Two paths, picked at runtime based on what's available:
///
/// Reads each present GPU activity table's parquet file via Arrow's
/// batched columnar reader, in parallel across tables, and joins
/// against an in-memory `ctx_for_pid` map. DuckDB→Rust row iteration
/// over ~27 M rows costs ~8 s on a 21.8 M-kernel trace; the columnar
/// path skips that handoff entirely.
///
/// Returns an empty map when the context-info table is absent or no
/// GPU activity table is present — callers treat empty as "no GPU
/// disambiguation available", collapsing to runtime-only attribution.
fn collect_runtime_dev_ctx(trace: &Trace) -> Result<DevCtxMap> {
    if !trace.table_exists("TARGET_INFO_CUDA_CONTEXT_INFO") {
        return Ok(HashMap::new());
    }
    let present: Vec<&'static str> = GPU_ACTIVITY_TABLES
        .iter()
        .copied()
        .filter(|t| trace.table_exists(t))
        .collect();
    if present.is_empty() {
        return Ok(HashMap::new());
    }
    let ctx_for_pid = read_ctx_for_pid(trace)?;
    let map = collect_via_parquet(trace, &present, &ctx_for_pid)?;
    // Count Single vs Many to surface the multi-context fan-out
    // case. CUPTI's documented model assigns process-unique
    // correlationIds across contexts, so `Many > 0` is unexpected
    // — a malformed trace, driver quirk, or assumption-violating
    // CUPTI build. We log a warning when it fires so the agent /
    // operator sees it (the fan-out itself attributes correctly;
    // the warning is just a heads-up that the trace tripped a code
    // path that's exercised by no production trace we've seen).
    let (single_count, many_count) = map.values().fold((0usize, 0usize), |(s, m), v| match v {
        DevCtxValue::Single(_) => (s + 1, m),
        DevCtxValue::Many(_) => (s, m + 1),
    });
    if many_count > 0 {
        log::warn!(
            "runtime_nvtx_parent: {} of {} dev/ctx entries had multi-context (native_pid, \
             correlationId) clashes — CUPTI usually emits unique correlationIds per process, \
             so this may indicate a malformed trace or unusual driver state; attribution \
             fans out each clashed runtime row to one sidecar entry per (device, context)",
            many_count,
            single_count + many_count,
        );
    } else {
        log::debug!(
            "runtime_nvtx_parent: dev_ctx breakdown {} Single / 0 Many (no multi-context clashes)",
            single_count,
        );
    }
    Ok(map)
}

/// `(device, context) → process_id` table from
/// `TARGET_INFO_CUDA_CONTEXT_INFO`. Small (one row per CUDA context),
/// so we just load it into a HashMap once.
fn read_ctx_for_pid(trace: &Trace) -> Result<HashMap<(i32, i64), i64>> {
    let mut stmt = trace.conn().prepare(
        r#"SELECT CAST(deviceId  AS INTEGER) AS device_id,
                  CAST(contextId AS BIGINT)  AS context_id,
                  CAST(processId AS BIGINT)  AS process_id
           FROM nsight.TARGET_INFO_CUDA_CONTEXT_INFO"#,
    )?;
    let mut rows = stmt.query([])?;
    let mut out = HashMap::new();
    while let Some(r) = rows.next()? {
        let dev: i32 = r.get(0)?;
        let ctx: i64 = r.get(1)?;
        let pid: i64 = r.get(2)?;
        out.insert((dev, ctx), pid);
    }
    Ok(out)
}

/// Build the dev/ctx map by reading each present GPU activity table's
/// parquet file directly. Caller guarantees every `tables` entry is
/// present in the parquetdir (filtered via `Trace::table_exists`).
///
/// Rayon-parallelises across tables so kernel (the largest) doesn't
/// gate memcpy / memset / sync; each table contributes a partial map
/// that's merged at the end.
fn collect_via_parquet(
    trace: &Trace,
    tables: &[&'static str],
    ctx_for_pid: &HashMap<(i32, i64), i64>,
) -> Result<DevCtxMap> {
    let paths: Vec<PathBuf> = tables.iter().map(|t| trace.parquet_path(t)).collect();
    let partials: Vec<DevCtxMap> = paths
        .par_iter()
        .map(|p| read_gpu_dev_ctx_parquet(p, ctx_for_pid))
        .collect::<Result<Vec<_>>>()?;
    let total: usize = partials.iter().map(|m| m.len()).sum();
    let mut out: DevCtxMap = HashMap::with_capacity(total);
    for p in partials {
        for (k, v) in p {
            match out.entry(k) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(v);
                }
                std::collections::hash_map::Entry::Occupied(mut e) => {
                    for dx in v.as_slice() {
                        e.get_mut().push(*dx);
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Read one GPU activity table's parquet file via Arrow's batched
/// columnar reader and project each row through `ctx_for_pid` to a
/// `(native_pid, correlation_id) → (device_id, context_id)` entry.
fn read_gpu_dev_ctx_parquet(
    path: &Path,
    ctx_for_pid: &HashMap<(i32, i64), i64>,
) -> Result<DevCtxMap> {
    let file = File::open(path)
        .with_context(|| format!("opening GPU activity parquet {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("opening parquet reader for {}", path.display()))?;
    let schema = builder.schema().clone();
    let corr_idx = schema
        .index_of("correlationId")
        .with_context(|| format!("{}: missing correlationId column", path.display()))?;
    let dev_idx = schema
        .index_of("deviceId")
        .with_context(|| format!("{}: missing deviceId column", path.display()))?;
    let ctx_idx = schema
        .index_of("contextId")
        .with_context(|| format!("{}: missing contextId column", path.display()))?;
    let reader = builder.build()?;

    let mut out: DevCtxMap = HashMap::new();
    for batch in reader {
        let batch = batch?;
        let corrs = batch.column(corr_idx).as_ref();
        let devs = batch.column(dev_idx).as_ref();
        let ctxs = batch.column(ctx_idx).as_ref();
        let n = batch.num_rows();
        out.reserve(n);
        for i in 0..n {
            let Some(corr) = parquet_integer_i64(corrs, i, "correlationId", path)? else {
                continue;
            };
            let Some(dev_i64) = parquet_integer_i64(devs, i, "deviceId", path)? else {
                continue;
            };
            let Some(ctx_id) = parquet_integer_i64(ctxs, i, "contextId", path)? else {
                continue;
            };
            let dev = i32::try_from(dev_i64).with_context(|| {
                format!(
                    "{}: deviceId value {dev_i64} does not fit Int32",
                    path.display()
                )
            })?;
            if let Some(&native_pid) = ctx_for_pid.get(&(dev, ctx_id)) {
                match out.entry((native_pid, corr)) {
                    std::collections::hash_map::Entry::Vacant(e) => {
                        e.insert(DevCtxValue::Single((dev, ctx_id)));
                    }
                    std::collections::hash_map::Entry::Occupied(mut e) => {
                        e.get_mut().push((dev, ctx_id));
                    }
                }
            }
        }
    }
    Ok(out)
}

fn parquet_integer_i64(
    array: &dyn Array,
    row: usize,
    column: &str,
    path: &Path,
) -> Result<Option<i64>> {
    if array.is_null(row) {
        return Ok(None);
    }
    if let Some(a) = array.as_any().downcast_ref::<Int8Array>() {
        return Ok(Some(i64::from(a.value(row))));
    }
    if let Some(a) = array.as_any().downcast_ref::<Int16Array>() {
        return Ok(Some(i64::from(a.value(row))));
    }
    if let Some(a) = array.as_any().downcast_ref::<Int32Array>() {
        return Ok(Some(i64::from(a.value(row))));
    }
    if let Some(a) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok(Some(a.value(row)));
    }
    if let Some(a) = array.as_any().downcast_ref::<UInt8Array>() {
        return Ok(Some(i64::from(a.value(row))));
    }
    if let Some(a) = array.as_any().downcast_ref::<UInt16Array>() {
        return Ok(Some(i64::from(a.value(row))));
    }
    if let Some(a) = array.as_any().downcast_ref::<UInt32Array>() {
        return Ok(Some(i64::from(a.value(row))));
    }
    if let Some(a) = array.as_any().downcast_ref::<UInt64Array>() {
        let value = a.value(row);
        return i64::try_from(value).map(Some).with_context(|| {
            format!(
                "{}: {column} value {value} does not fit Int64",
                path.display()
            )
        });
    }
    anyhow::bail!(
        "{}: {column} has unsupported integer type {:?}",
        path.display(),
        array.data_type()
    )
}

fn collect_nvtx_by_tid(trace: &Trace) -> Result<HashMap<i64, Vec<NvtxRangeRow>>> {
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
    let mut stmt = trace.conn().prepare(&sql)?;
    let mut rows = stmt.query([])?;
    let mut by_tid: HashMap<i64, Vec<NvtxRangeRow>> = HashMap::new();
    while let Some(r) = rows.next()? {
        let rowid: i64 = r.get(0)?;
        let start: i64 = r.get(1)?;
        let end: i64 = r.get(2)?;
        let tid: i64 = r.get(3)?;
        let name: String = r.get(4)?;
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

fn collect_runtime_rows(trace: &Trace) -> Result<Vec<RuntimeRow>> {
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
    let mut stmt = trace.conn().prepare(&sql)?;
    let mut rows = stmt.query([])?;
    let mut out: Vec<RuntimeRow> = Vec::new();
    while let Some(r) = rows.next()? {
        out.push(RuntimeRow {
            rowid: r.get(0)?,
            correlation_id: r.get(1)?,
            native_pid: r.get(2)?,
            global_tid: r.get(3)?,
            start: r.get(4)?,
            end: r.get(5)?,
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
fn walk(
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

// ----- parquet I/O ---------------------------------------------------------

fn parquet_schema() -> SchemaRef {
    // Arrow's default ListBuilder inner field is `Field::new("item",
    // …, true)` (nullable). The schema must match — we never write
    // nulls in practice, but the logical type carries the nullable
    // bit and a mismatch fails `RecordBatch::try_new`.
    let rowids_field = Arc::new(Field::new("item", DataType::Int64, true));
    let names_field = Arc::new(Field::new("item", DataType::Utf8, true));
    Arc::new(Schema::new(vec![
        Field::new("rt_rowid", DataType::Int64, false),
        // Nullable: runtime calls without a CUDA correlation
        // (e.g. `cudaGetDeviceCount`) write NULL here.
        Field::new("correlation_id", DataType::Int64, true),
        Field::new("native_pid", DataType::Int64, false),
        // Nullable: the runtime call's resolved GPU (device, context)
        // — both NULL when the call has no corresponding GPU
        // activity, when `TARGET_INFO_CUDA_CONTEXT_INFO` was absent
        // during build, or when no GPU activity table is present.
        Field::new("device_id", DataType::Int32, true),
        Field::new("context_id", DataType::Int64, true),
        Field::new("nvtx_rowids", DataType::List(rowids_field), false),
        Field::new("nvtx_names", DataType::List(names_field), false),
    ]))
}

const KV_VERSION: &str = "veloq.runtime_nvtx_parent.version";

fn write_parquet(path: &Path, fp: SourceFingerprint, records: &[RuntimeParentEntry]) -> Result<()> {
    let schema = parquet_schema();
    let mut rt_rowids: Vec<i64> = Vec::with_capacity(records.len());
    let mut corrs: Vec<Option<i64>> = Vec::with_capacity(records.len());
    let mut pids: Vec<i64> = Vec::with_capacity(records.len());
    let mut devs: Vec<Option<i32>> = Vec::with_capacity(records.len());
    let mut ctxs: Vec<Option<i64>> = Vec::with_capacity(records.len());

    // ListBuilders for the two list columns. The inner builder
    // accumulates items per row; calling `.append(true)` closes the
    // current row's list and starts a new one.
    let mut rowids_builder: ListBuilder<Int64Builder> = ListBuilder::new(Int64Builder::new());
    let mut names_builder: ListBuilder<StringBuilder> = ListBuilder::new(StringBuilder::new());

    for r in records {
        rt_rowids.push(r.rt_rowid);
        corrs.push(r.correlation_id);
        pids.push(r.native_pid);
        devs.push(r.device_id);
        ctxs.push(r.context_id);
        for e in &r.enclosing {
            rowids_builder.values().append_value(e.nvtx_rowid);
            names_builder.values().append_value(&e.nvtx_name);
        }
        rowids_builder.append(true);
        names_builder.append(true);
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(rt_rowids)),
        Arc::new(Int64Array::from(corrs)),
        Arc::new(Int64Array::from(pids)),
        Arc::new(Int32Array::from(devs)),
        Arc::new(Int64Array::from(ctxs)),
        Arc::new(rowids_builder.finish()),
        Arc::new(names_builder.finish()),
    ];
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns)
        .context("assembling RecordBatch for nvtx-parent sidecar")?;

    // Embed fingerprint + format version as parquet KV metadata. Warm
    // open reads only the footer (cheap) to validate before scanning.
    let kv = crate::sidecar::freshness_kv(KV_VERSION, RUNTIME_NVTX_PARENT_VERSION, fp);
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_key_value_metadata(Some(kv))
        .build();

    crate::sidecar::atomic_publish(path, |tmp| {
        let file = File::create(tmp)
            .with_context(|| format!("creating {} for nvtx-parent sidecar", tmp.display()))?;
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    })
}

fn read_parquet(path: &Path) -> Result<Vec<RuntimeParentEntry>> {
    let file = File::open(path)
        .with_context(|| format!("opening nvtx-parent sidecar {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;
    let mut out: Vec<RuntimeParentEntry> = Vec::new();
    for batch in reader {
        let batch = batch?;
        let rt = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("nvtx-parent: rt_rowid column missing/wrong type")?;
        let corr = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("nvtx-parent: correlation_id column missing/wrong type")?;
        let pid = batch
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("nvtx-parent: native_pid column missing/wrong type")?;
        let dev = batch
            .column(3)
            .as_any()
            .downcast_ref::<Int32Array>()
            .context("nvtx-parent: device_id column missing/wrong type")?;
        let ctx = batch
            .column(4)
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("nvtx-parent: context_id column missing/wrong type")?;
        let nvtx_rowids_col = batch
            .column(5)
            .as_any()
            .downcast_ref::<ListArray>()
            .context("nvtx-parent: nvtx_rowids column missing/wrong type")?;
        let nvtx_names_col = batch
            .column(6)
            .as_any()
            .downcast_ref::<ListArray>()
            .context("nvtx-parent: nvtx_names column missing/wrong type")?;
        for i in 0..batch.num_rows() {
            let ids_arr = nvtx_rowids_col.value(i);
            let ids = ids_arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .context("nvtx-parent: nvtx_rowids inner is not Int64")?;
            let names_arr = nvtx_names_col.value(i);
            let names = names_arr
                .as_any()
                .downcast_ref::<StringArray>()
                .context("nvtx-parent: nvtx_names inner is not Utf8")?;
            if ids.len() != names.len() {
                anyhow::bail!(
                    "nvtx-parent: row {i}: nvtx_rowids ({}) and nvtx_names ({}) length mismatch",
                    ids.len(),
                    names.len()
                );
            }
            let mut enclosing: Vec<EnclosingNvtx> = Vec::with_capacity(ids.len());
            for j in 0..ids.len() {
                enclosing.push(EnclosingNvtx {
                    nvtx_rowid: ids.value(j),
                    nvtx_name: names.value(j).to_string(),
                });
            }
            out.push(RuntimeParentEntry {
                rt_rowid: rt.value(i),
                correlation_id: if corr.is_null(i) {
                    None
                } else {
                    Some(corr.value(i))
                },
                native_pid: pid.value(i),
                device_id: if dev.is_null(i) {
                    None
                } else {
                    Some(dev.value(i))
                },
                context_id: if ctx.is_null(i) {
                    None
                } else {
                    Some(ctx.value(i))
                },
                enclosing,
            });
        }
    }
    Ok(out)
}

fn sidecar_is_fresh(path: &Path, fp: SourceFingerprint) -> Result<bool> {
    crate::sidecar::is_fresh(
        path,
        KV_VERSION,
        RUNTIME_NVTX_PARENT_VERSION,
        fp,
        "runtime_nvtx_parent",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(rowid: i64, start: i64, end: i64, name: &str) -> NvtxRangeRow {
        NvtxRangeRow {
            rowid,
            start,
            end,
            name: name.to_string(),
        }
    }

    fn rt(rowid: i64, corr: Option<i64>, pid: i64, tid: i64, start: i64, end: i64) -> RuntimeRow {
        RuntimeRow {
            rowid,
            correlation_id: corr,
            native_pid: pid,
            global_tid: tid,
            start,
            end,
            device_id: None,
            context_id: None,
        }
    }

    #[test]
    fn walk_collects_outer_to_inner_for_nested_ranges() -> Result<()> {
        let mut by_tid = HashMap::new();
        by_tid.insert(7, vec![n(1, 0, 100, "outer"), n(2, 40, 60, "inner")]);
        let out = walk(&by_tid, &[rt(10, Some(999), 42, 7, 45, 55)]);
        let first = out
            .first()
            .ok_or_else(|| anyhow::anyhow!("expected one entry"))?;
        assert_eq!(first.enclosing.len(), 2);
        // Outer first.
        let outer = first
            .enclosing
            .first()
            .ok_or_else(|| anyhow::anyhow!("missing outer"))?;
        assert_eq!(outer.nvtx_name, "outer");
        assert_eq!(outer.nvtx_rowid, 1);
        // Innermost = last.
        let innermost = first
            .innermost()
            .ok_or_else(|| anyhow::anyhow!("missing inner"))?;
        assert_eq!(innermost.nvtx_name, "inner");
        assert_eq!(innermost.nvtx_rowid, 2);
        Ok(())
    }

    /// P2 review guard: when two enclosing NVTX ranges share the
    /// same start, the one with the larger end is OUTER and must
    /// land earlier in the chain so `.last()` (innermost) is the
    /// tighter range.
    #[test]
    fn walk_orders_same_start_by_end_desc() -> Result<()> {
        let mut by_tid = HashMap::new();
        // Both start at 0; outer ends at 100, inner ends at 60.
        by_tid.insert(7, vec![n(1, 0, 100, "outer"), n(2, 0, 60, "inner")]);
        let out = walk(&by_tid, &[rt(10, Some(999), 42, 7, 30, 50)]);
        let first = out
            .first()
            .ok_or_else(|| anyhow::anyhow!("expected one entry"))?;
        assert_eq!(first.enclosing.len(), 2);
        let outer = first
            .enclosing
            .first()
            .ok_or_else(|| anyhow::anyhow!("missing outer"))?;
        assert_eq!(outer.nvtx_name, "outer", "outer must come first");
        let innermost = first
            .innermost()
            .ok_or_else(|| anyhow::anyhow!("missing innermost"))?;
        assert_eq!(
            innermost.nvtx_name, "inner",
            "innermost must be tighter range"
        );
        Ok(())
    }

    #[test]
    fn walk_skips_partial_overlap() {
        let mut by_tid = HashMap::new();
        by_tid.insert(7, vec![n(1, 0, 100, "outer")]);
        // Runtime exits past the NVTX end — not fully contained.
        let out = walk(&by_tid, &[rt(10, Some(999), 42, 7, 50, 150)]);
        assert!(out.is_empty());
    }

    #[test]
    fn any_enclosing_name_matches_outer_when_innermost_does_not() -> Result<()> {
        let mut by_tid = HashMap::new();
        by_tid.insert(
            7,
            vec![n(1, 0, 100, "training_step"), n(2, 40, 60, "fwd_pass")],
        );
        let out = walk(&by_tid, &[rt(10, Some(999), 42, 7, 45, 55)]);
        let first = out
            .first()
            .ok_or_else(|| anyhow::anyhow!("expected one entry"))?;
        // The semantic that v1 sidecar couldn't preserve: a pattern
        // matching the OUTER range must still attribute the contained
        // event, even though the innermost is something else.
        assert!(first.any_enclosing_name(|n| n.starts_with("training")));
        assert!(first.any_enclosing_name(|n| n == "fwd_pass"));
        assert!(!first.any_enclosing_name(|n| n.starts_with("eval")));
        Ok(())
    }

    #[test]
    fn parquet_roundtrip_preserves_records() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let path = tmp.path().join("rtnvtx.parquet");
        let records = vec![
            RuntimeParentEntry {
                rt_rowid: 1,
                correlation_id: Some(100),
                native_pid: 42,
                device_id: Some(0),
                context_id: Some(1),
                enclosing: vec![
                    EnclosingNvtx {
                        nvtx_rowid: 11,
                        nvtx_name: "iter_42".to_string(),
                    },
                    EnclosingNvtx {
                        nvtx_rowid: 12,
                        nvtx_name: "step_a".to_string(),
                    },
                ],
            },
            RuntimeParentEntry {
                rt_rowid: 2,
                correlation_id: Some(101),
                native_pid: 42,
                device_id: Some(0),
                context_id: Some(1),
                enclosing: vec![EnclosingNvtx {
                    nvtx_rowid: 11,
                    nvtx_name: "iter_42".to_string(),
                }],
            },
            // Runtime call without a CUDA correlation (e.g.
            // cudaGetDeviceCount). Must round-trip cleanly and only
            // surface in `by_rt_rowid`, never in `by_correlation`.
            RuntimeParentEntry {
                rt_rowid: 3,
                correlation_id: None,
                native_pid: 99,
                device_id: None,
                context_id: None,
                enclosing: vec![EnclosingNvtx {
                    nvtx_rowid: 21,
                    nvtx_name: "step_b".to_string(),
                }],
            },
        ];
        let fp = SourceFingerprint {
            mtime_secs: 1234567890,
            size: 4096,
        };
        write_parquet(&path, fp, &records)?;
        assert!(sidecar_is_fresh(&path, fp)?);
        let bumped = SourceFingerprint {
            mtime_secs: 1234567891,
            size: 4096,
        };
        assert!(
            !sidecar_is_fresh(&path, bumped)?,
            "mtime-change must invalidate"
        );
        let loaded = read_parquet(&path)?;
        assert_eq!(loaded, records);
        Ok(())
    }

    #[test]
    fn gpu_dev_ctx_reader_accepts_unsigned_nsys_integer_columns() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let path = tmp.path().join("gpu.parquet");
        let schema = Arc::new(Schema::new(vec![
            Field::new("correlationId", DataType::UInt32, true),
            Field::new("deviceId", DataType::UInt32, true),
            Field::new("contextId", DataType::UInt64, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(UInt32Array::from(vec![Some(77_u32), Some(88_u32)])),
                Arc::new(UInt32Array::from(vec![Some(0_u32), Some(1_u32)])),
                Arc::new(UInt64Array::from(vec![Some(123_u64), Some(999_u64)])),
            ],
        )?;
        {
            let file = File::create(&path)?;
            let mut writer = ArrowWriter::try_new(file, schema, None)?;
            writer.write(&batch)?;
            writer.close()?;
        }

        let mut ctx_for_pid = HashMap::new();
        ctx_for_pid.insert((0, 123), 4242);
        let out = read_gpu_dev_ctx_parquet(&path, &ctx_for_pid)?;
        assert_eq!(out.get(&(4242, 77)), Some(&DevCtxValue::Single((0, 123))));
        assert!(
            !out.contains_key(&(4242, 88)),
            "unmapped unsigned context should not produce an entry"
        );
        Ok(())
    }

    /// Schema invariant: runtime rows with `correlation_id = None` still
    /// show up under `by_rt_rowid` (so `--type runtime --group-by
    /// nvtx-parent` attributes them) but never under `by_correlation`
    /// (the GPU-side bridge only exists when there's a correlation
    /// to bridge through).
    #[test]
    fn none_correlation_is_absent_from_by_correlation_map() -> Result<()> {
        let records = vec![
            RuntimeParentEntry {
                rt_rowid: 1,
                correlation_id: Some(100),
                native_pid: 42,
                device_id: Some(0),
                context_id: Some(1),
                enclosing: vec![EnclosingNvtx {
                    nvtx_rowid: 11,
                    nvtx_name: "step".to_string(),
                }],
            },
            RuntimeParentEntry {
                rt_rowid: 2,
                correlation_id: None,
                native_pid: 42,
                device_id: None,
                context_id: None,
                enclosing: vec![EnclosingNvtx {
                    nvtx_rowid: 11,
                    nvtx_name: "step".to_string(),
                }],
            },
        ];
        let idx = RuntimeNvtxParent::from_records(records);
        // Both rows attributed.
        assert!(idx.get_by_runtime(1).is_some());
        assert!(idx.get_by_runtime(2).is_some());
        // Correlation lookup works for the correlated row via the
        // full disambiguator trio (device, context, correlation).
        assert!(idx.get_by_correlation(0, 1, 100).is_some());
        // The None-correlation row leaves no ghost entry.
        assert!(idx.get_by_correlation(0, 1, 0).is_none());
        Ok(())
    }

    /// `merge_dev_ctx` fan-out: a single attributed runtime row with
    /// an ambiguous `(native_pid, correlationId) → (device, context)`
    /// mapping must produce one sidecar entry per candidate `(D, X)`,
    /// preserving its enclosing chain across all copies.
    #[test]
    fn merge_dev_ctx_fans_out_on_ambiguous_correlation() -> Result<()> {
        let walked = vec![RuntimeParentEntry {
            rt_rowid: 1,
            correlation_id: Some(42),
            native_pid: 1000,
            device_id: None,
            context_id: None,
            enclosing: vec![EnclosingNvtx {
                nvtx_rowid: 11,
                nvtx_name: "step".to_string(),
            }],
        }];
        let mut dev_ctx: DevCtxMap = HashMap::new();
        // Two contexts in the same process both claim correlationId=42.
        dev_ctx.insert((1000, 42), DevCtxValue::Many(vec![(0, 1), (0, 2)]));

        let out = merge_dev_ctx(walked, &dev_ctx);
        assert_eq!(out.len(), 2, "ambiguous (D,X) must fan out");
        let dx_pairs: std::collections::BTreeSet<(i32, i64)> = out
            .iter()
            .map(|e| (e.device_id.unwrap_or(-1), e.context_id.unwrap_or(-1)))
            .collect();
        assert!(dx_pairs.contains(&(0, 1)));
        assert!(dx_pairs.contains(&(0, 2)));
        // Same rt_rowid, same enclosing on every copy.
        assert!(out.iter().all(|e| e.rt_rowid == 1));
        assert!(out.iter().all(|e| {
            e.enclosing.len() == 1
                && e.enclosing.first().map(|n| n.nvtx_name.as_str()) == Some("step")
        }));
        Ok(())
    }

    /// Common case: a single (D, X) candidate mutates the entry in
    /// place — no clone, no fanout.
    #[test]
    fn merge_dev_ctx_single_candidate_mutates_in_place() -> Result<()> {
        let walked = vec![RuntimeParentEntry {
            rt_rowid: 7,
            correlation_id: Some(100),
            native_pid: 1000,
            device_id: None,
            context_id: None,
            enclosing: vec![EnclosingNvtx {
                nvtx_rowid: 11,
                nvtx_name: "step".to_string(),
            }],
        }];
        let mut dev_ctx: DevCtxMap = HashMap::new();
        dev_ctx.insert((1000, 100), DevCtxValue::Single((0, 1)));
        let out = merge_dev_ctx(walked, &dev_ctx);
        assert_eq!(out.len(), 1);
        let first = out
            .first()
            .ok_or_else(|| anyhow::anyhow!("missing entry"))?;
        assert_eq!(first.device_id, Some(0));
        assert_eq!(first.context_id, Some(1));
        Ok(())
    }

    /// Schema invariant: multi-context within a single process with the
    /// same `correlationId` reused across contexts disambiguates
    /// through the `(device, context, correlation)` key.
    #[test]
    fn multi_context_same_correlation_disambiguates_by_device_context() -> Result<()> {
        let records = vec![
            RuntimeParentEntry {
                rt_rowid: 1,
                correlation_id: Some(42),
                native_pid: 1000,
                device_id: Some(0),
                context_id: Some(1),
                enclosing: vec![EnclosingNvtx {
                    nvtx_rowid: 11,
                    nvtx_name: "ctx1_scope".to_string(),
                }],
            },
            RuntimeParentEntry {
                rt_rowid: 2,
                correlation_id: Some(42),
                native_pid: 1000,
                device_id: Some(0),
                context_id: Some(2),
                enclosing: vec![EnclosingNvtx {
                    nvtx_rowid: 22,
                    nvtx_name: "ctx2_scope".to_string(),
                }],
            },
        ];
        let idx = RuntimeNvtxParent::from_records(records);
        let e1 = idx
            .get_by_correlation(0, 1, 42)
            .ok_or_else(|| anyhow::anyhow!("missing (0,1,42)"))?;
        let e2 = idx
            .get_by_correlation(0, 2, 42)
            .ok_or_else(|| anyhow::anyhow!("missing (0,2,42)"))?;
        assert_eq!(
            e1.innermost().map(|e| e.nvtx_name.as_str()),
            Some("ctx1_scope")
        );
        assert_eq!(
            e2.innermost().map(|e| e.nvtx_name.as_str()),
            Some("ctx2_scope")
        );
        Ok(())
    }
}
