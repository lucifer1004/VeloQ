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

mod compute;
mod gpu_activity;
mod parquet;

#[cfg(test)]
mod tests;

use crate::{NsysDataResult, Trace};
use compute::compute;
use parquet::{read_parquet, sidecar_is_fresh, write_parquet};
use std::collections::HashMap;
use std::fs;
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

fn source_fingerprint(trace_path: &Path) -> NsysDataResult<SourceFingerprint> {
    crate::trace_artifact_fingerprint(trace_path).map_err(|source| {
        crate::NsysDataError::nvtx_parent_trace_fingerprint(trace_path.display(), source)
    })
}

/// Build the sidecar if missing or stale; return its path. The SQL
/// `read_parquet(…)` join in every NVTX-bearing verb consumes the
/// path directly.
///
/// Side effect: when the sidecar is rebuilt, the records are computed
/// and persisted; the in-memory index is *not* cached in the `Trace`
/// from this call. Callers that need the in-memory map should use
/// [`build_or_load_index`].
pub fn ensure_sidecar(trace: &Trace) -> NsysDataResult<PathBuf> {
    Ok(ensure_sidecar_state(trace)?.path)
}

fn ensure_sidecar_state(
    trace: &Trace,
) -> NsysDataResult<crate::sidecar::FreshSidecar<Vec<RuntimeParentEntry>>> {
    let path = sidecar_path_for(trace.path());
    let fp = source_fingerprint(trace.path())?;
    let state = crate::sidecar::ensure_fresh_sidecar::<Vec<RuntimeParentEntry>>(
        path,
        fp,
        sidecar_is_fresh,
        || compute(trace),
        |path, fp, records| write_parquet(path, fp, records),
    )?;
    if let Some(records) = &state.rebuilt_records {
        log::info!(
            "runtime_nvtx_parent: built sidecar at {} ({} entries)",
            state.path.display(),
            records.len()
        );
    } else {
        log::debug!(
            "runtime_nvtx_parent: warm sidecar at {} ({} bytes)",
            state.path.display(),
            fs::metadata(&state.path).map(|m| m.len()).unwrap_or(0),
        );
    }
    Ok(state)
}

/// Build the sidecar (if missing or stale) and return the in-memory
/// [`RuntimeNvtxParent`] index. Callers that only need the SQL path
/// (e.g. `stats --group-by nvtx-parent`) should call [`ensure_sidecar`]
/// instead and let DuckDB scan the parquet directly.
pub fn build_or_load_index(trace: &Trace) -> NsysDataResult<RuntimeNvtxParent> {
    let state = ensure_sidecar_state(trace)?;
    if !state.path.exists() {
        return Ok(RuntimeNvtxParent::empty());
    }
    let records = match state.rebuilt_records {
        Some(records) => records,
        None => read_parquet(&state.path)?,
    };
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
pub fn load_if_present(trace: &Trace) -> NsysDataResult<Option<RuntimeNvtxParent>> {
    let path = sidecar_path_for(trace.path());
    let fp = source_fingerprint(trace.path())?;
    let records = crate::sidecar::load_if_fresh(&path, fp, sidecar_is_fresh, read_parquet)?;
    Ok(records.map(RuntimeNvtxParent::from_records))
}
