//! Per-trace flattened NVTX interval tree.
//!
//! One row per NVTX range, decorated with its parent range, depth in
//! the nesting stack, and a slash-joined `path` of ancestor names. Lets
//! every NVTX query collapse to a single parquet scan:
//!
//! ```sql
//! -- stack at timestamp T on globalTid X
//! SELECT path, depth, name
//! FROM nvtx_tree
//! WHERE global_tid = ? AND start <= ? AND (end IS NULL OR end > ?)
//! ORDER BY depth ASC;
//!
//! -- time per NVTX path across the whole trace
//! SELECT path, SUM(duration_ns), COUNT(*)
//! FROM nvtx_tree GROUP BY path ORDER BY 2 DESC LIMIT 20;
//! ```
//!
//! ## Why a per-trace index
//!
//! The naive containment join (`NVTX_EVENTS x NVTX_EVENTS` to
//! reconstruct parent links + recursive CTE for `path`) is
//! `O(N_nvtx^2 / T)` per query in DuckDB. The sidecar amortizes that
//! into a single sweep at build time and turns repeated queries into a
//! parquet scan with row-group skipping on `start`.
//!
//! ## Algorithm
//!
//! NSys's parquetdir export already collapses push/pop pairs (and
//! rangeStart/rangeEnd pairs) into a single row per range - every
//! `NVTX_EVENTS` row carries its own `start`, `end`, and resolved
//! name. The build pass therefore reuses the same stack-scan as
//! [`crate::nvtx_nesting`]: group rows by `(global_tid, domain_id)`,
//! sort by `start`, maintain a per-group stack of `(end_ns, range_id)`
//! sorted so `stack.last()` is the soonest-closing open range. At each
//! row we pop ranges that ended at or before the current `start`,
//! capture the new top of stack as `parent_range_id`, take `depth`
//! from the post-pop stack size, then push `(end, range_id)` if the
//! range has positive duration (instant markers / zero-duration ranges
//! don't nest, matching `nvtx_nesting`'s shape).
//!
//! `path` is materialized in a second pass over rows sorted by depth
//! ASC: each row's path is its parent's path plus `/` plus its own
//! name. Names containing `/` are escaped as `\/` so the join character
//! is unambiguous; readers that need to split a path must respect the
//! same escape.
//!
//! ## Shape of the cached artifact
//!
//! `<trace>.veloq/nvtx-tree.parquet` - one row per NVTX range with
//! `start IS NOT NULL`. SNAPPY-compressed, sorted by `(start ASC,
//! depth ASC)` so time-range predicates get parquet row-group skipping.
//! Schema (v1):
//!
//! | column            | type          | notes                              |
//! | ----------------- | ------------- | ---------------------------------- |
//! | `range_id`        | INT64         | original `NVTX_EVENTS.rowid`       |
//! | `parent_range_id` | INT64 (null)  | NULL at the top of the stack        |
//! | `depth`           | INT32         | 0 at the top                       |
//! | `domain_id`       | INT64         | `COALESCE(NVTX_EVENTS.domainId, 0)` |
//! | `name`            | VARCHAR       | `COALESCE(text, StringIds.value, '')` |
//! | `path`            | VARCHAR       | `/`-joined ancestor names ending in self; `/` in names escaped as `\/` |
//! | `start`           | INT64         | ns from session epoch              |
//! | `end`             | INT64 (null)  | NULL for instant markers           |
//! | `duration_ns`     | INT64 (null)  | `end - start`; NULL when `end` is NULL |
//! | `global_tid`      | INT64         | host thread that hosts the range   |
//!
//! Freshness/atomic publish via [`crate::sidecar`]; the version key is
//! `veloq.nvtx_tree.version` ([`NVTX_TREE_VERSION`]). `.nsys-rep` inputs
//! fingerprint the report file, while direct `_pqtdir/` inputs
//! fingerprint the child parquet files through
//! [`crate::trace_artifact_fingerprint`].

mod compute;
mod parquet;

#[cfg(test)]
mod tests;

pub(crate) use compute::nvtx_event_type_ids;

use crate::{NsysDataResult, Trace};
use compute::compute;
use parquet::{read_parquet, sidecar_is_fresh, write_parquet};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use veloq_core::SourceFingerprint;

/// Bump on every breaking schema change to the parquet sidecar.
/// Mismatched versions rebuild silently on next open.
pub const NVTX_TREE_VERSION: u32 = 1;

/// One NVTX range with parent linkage, depth, and ancestor path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NvtxTreeRecord {
    pub range_id: i64,
    pub parent_range_id: Option<i64>,
    pub depth: i32,
    pub domain_id: i64,
    pub name: String,
    pub path: String,
    pub start: i64,
    /// `None` for instant markers (`NVTX_EVENTS."end" IS NULL`).
    pub end: Option<i64>,
    pub duration_ns: Option<i64>,
    pub global_tid: i64,
}

/// In-memory flattened NVTX tree for a trace.
///
/// Carries auxiliary indices so callers can answer "stack at T" /
/// "ancestors of range R" without re-scanning the whole table.
pub struct NvtxTree {
    records: Vec<NvtxTreeRecord>,
    /// `range_id -> records index`. Each `range_id` is unique because
    /// it equals the source `NVTX_EVENTS.rowid`.
    by_id: HashMap<i64, usize>,
    /// `global_tid -> records indices sorted by start ASC` for the
    /// stack-at-T binary search. Populated lazily; the build pass
    /// emits rows pre-sorted by `(start, depth)` so the per-tid slice
    /// is just a filter, but we materialize the indices once here so
    /// repeated lookups are O(log N).
    by_tid_start: HashMap<i64, Vec<usize>>,
}

impl NvtxTree {
    pub fn empty() -> Self {
        Self {
            records: Vec::new(),
            by_id: HashMap::new(),
            by_tid_start: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn records(&self) -> &[NvtxTreeRecord] {
        &self.records
    }

    /// Range record for `range_id`, if present.
    pub fn get(&self, range_id: i64) -> Option<&NvtxTreeRecord> {
        self.by_id.get(&range_id).and_then(|i| self.records.get(*i))
    }

    /// Innermost stack on `global_tid` whose interval covers `t`
    /// (containment `start <= t < end`, with `end IS NULL` treated as
    /// `+inf`), returned outer->inner. Empty when the tid has no NVTX
    /// activity or no enclosing range at `t`.
    ///
    /// Well-formed NVTX ranges produce a single active chain. If a trace
    /// has unrelated covering chains on the same tid (for example,
    /// different domains or malformed overlaps), this chooses the
    /// deepest covering record and walks its covering parents so the
    /// result remains one chain.
    pub fn stack_at(&self, global_tid: i64, t: i64) -> Vec<&NvtxTreeRecord> {
        let Some(idxs) = self.by_tid_start.get(&global_tid) else {
            return Vec::new();
        };
        // First index whose start > t: the candidate window is
        // everything before it. NVTX nesting is correctly captured by
        // the depth/parent linkage already, so we just keep rows whose
        // interval covers `t`.
        let cut =
            idxs.partition_point(|&i| self.records.get(i).map(|r| r.start <= t).unwrap_or(false));
        let innermost = idxs
            .iter()
            .take(cut)
            .filter_map(|&i| self.records.get(i))
            .filter(|r| covers_timestamp(r, t))
            .max_by(|a, b| {
                a.depth
                    .cmp(&b.depth)
                    .then(a.start.cmp(&b.start))
                    .then(a.range_id.cmp(&b.range_id))
            });
        let Some(mut cursor) = innermost else {
            return Vec::new();
        };

        let mut out = Vec::new();
        let mut remaining = self.records.len();
        while remaining > 0 && cursor.global_tid == global_tid && covers_timestamp(cursor, t) {
            out.push(cursor);
            let Some(parent_id) = cursor.parent_range_id else {
                break;
            };
            let Some(parent) = self.get(parent_id) else {
                break;
            };
            cursor = parent;
            remaining -= 1;
        }
        out.reverse();
        out
    }

    /// Walk parent links from `range_id` to root, yielding the range
    /// itself first. Stops on missing parent linkage (a sidecar
    /// corruption or truncated build that lost an ancestor).
    pub fn ancestors(&self, range_id: i64) -> Vec<&NvtxTreeRecord> {
        let mut cursor = self.get(range_id);
        let mut out = Vec::new();
        while let Some(rec) = cursor {
            out.push(rec);
            cursor = rec.parent_range_id.and_then(|p| self.get(p));
        }
        out
    }

    fn from_records(records: Vec<NvtxTreeRecord>) -> Self {
        let mut by_id: HashMap<i64, usize> = HashMap::with_capacity(records.len());
        let mut by_tid_start: HashMap<i64, Vec<usize>> = HashMap::new();
        for (i, r) in records.iter().enumerate() {
            by_id.insert(r.range_id, i);
            by_tid_start.entry(r.global_tid).or_default().push(i);
        }
        for idxs in by_tid_start.values_mut() {
            idxs.sort_by_key(|&i| records.get(i).map(|r| r.start).unwrap_or(i64::MIN));
        }
        Self {
            records,
            by_id,
            by_tid_start,
        }
    }
}

fn covers_timestamp(record: &NvtxTreeRecord, t: i64) -> bool {
    record.start <= t && record.end.is_none_or(|e| e > t)
}

/// Filesystem path of the sidecar parquet under `<trace>.veloq/`.
pub fn sidecar_path_for(trace_path: &Path) -> PathBuf {
    veloq_core::artifact_dir_for(trace_path).join("nvtx-tree.parquet")
}

fn source_fingerprint(trace_path: &Path) -> NsysDataResult<SourceFingerprint> {
    crate::trace_artifact_fingerprint(trace_path).map_err(|source| {
        crate::NsysDataError::nvtx_tree_trace_fingerprint(trace_path.display(), source)
    })
}

pub(crate) fn sidecar_is_fresh_for_trace(trace_path: &Path) -> NsysDataResult<bool> {
    let path = sidecar_path_for(trace_path);
    let fp = source_fingerprint(trace_path)?;
    sidecar_is_fresh(&path, fp)
}

/// Build the sidecar if missing or stale; return its path. Also
/// registers `nsight.nvtx_tree` over the freshly-written sidecar on
/// `trace`'s connection so subsequent SQL on the same `Trace` handle
/// can query the view directly. SQL-only callers can therefore call
/// this once and then issue `SELECT ... FROM nsight.nvtx_tree`.
pub fn ensure_sidecar(trace: &Trace) -> NsysDataResult<PathBuf> {
    Ok(ensure_sidecar_state(trace)?.path)
}

fn ensure_sidecar_state(
    trace: &Trace,
) -> NsysDataResult<crate::sidecar::FreshSidecar<Vec<NvtxTreeRecord>>> {
    let path = sidecar_path_for(trace.path());
    let fp = source_fingerprint(trace.path())?;
    let state = crate::sidecar::ensure_fresh_sidecar::<Vec<NvtxTreeRecord>>(
        path,
        fp,
        sidecar_is_fresh,
        || compute(trace),
        |path, fp, records| write_parquet(path, fp, records),
    )?;
    if let Some(records) = &state.rebuilt_records {
        log::info!(
            "nvtx_tree: built sidecar at {} ({} ranges)",
            state.path.display(),
            records.len()
        );
    } else {
        log::debug!(
            "nvtx_tree: warm sidecar at {} ({} bytes)",
            state.path.display(),
            fs::metadata(&state.path).map(|m| m.len()).unwrap_or(0),
        );
    }
    attach_view(trace, &state.path)?;
    Ok(state)
}

fn attach_view(trace: &Trace, sidecar: &Path) -> NsysDataResult<()> {
    let Some(sql) = view_sql_for(sidecar) else {
        log::warn!(
            "nvtx_tree: sidecar path is not valid UTF-8, skipping view registration: {}",
            sidecar.display(),
        );
        return Ok(());
    };
    trace.conn().execute(&sql, []).map_err(|source| {
        crate::NsysDataError::nvtx_tree_view_register(sidecar.display(), source)
    })?;
    Ok(())
}

/// Build (if needed) and load all rows into memory.
pub fn build_or_load(trace: &Trace) -> NsysDataResult<NvtxTree> {
    let state = ensure_sidecar_state(trace)?;
    if !state.path.exists() {
        return Ok(NvtxTree::empty());
    }
    let records = match state.rebuilt_records {
        Some(records) => records,
        None => read_parquet(&state.path)?,
    };
    Ok(NvtxTree::from_records(records))
}

/// Load only if a fresh sidecar exists on disk; never trigger a build.
/// Mirrors `runtime_nvtx_parent::load_if_present` for cheap callers
/// that don't want to pay the build cost on a cold cache.
pub fn load_if_present(trace: &Trace) -> NsysDataResult<Option<NvtxTree>> {
    let path = sidecar_path_for(trace.path());
    let fp = source_fingerprint(trace.path())?;
    let records = crate::sidecar::load_if_fresh(&path, fp, sidecar_is_fresh, read_parquet)?;
    Ok(records.map(NvtxTree::from_records))
}

/// SQL fragment that registers a view named `nsight.nvtx_tree` over
/// the sidecar parquet, mirroring how `Trace::open` registers
/// parquetdir tables.
pub fn view_sql_for(sidecar_path: &Path) -> Option<String> {
    let lit = sidecar_path.to_str()?.replace('\'', "''");
    Some(format!(
        "CREATE OR REPLACE VIEW nsight.nvtx_tree AS \
         SELECT (file_row_number + 1) AS rowid, * \
         FROM read_parquet('{lit}', file_row_number = true)"
    ))
}
