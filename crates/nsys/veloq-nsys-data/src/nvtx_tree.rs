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

use crate::Trace;
use anyhow::{Context, Result};
use arrow::array::{Array, ArrayRef, Int32Array, Int64Array, StringArray, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;
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

fn source_fingerprint(trace_path: &Path) -> Result<SourceFingerprint> {
    crate::trace_artifact_fingerprint(trace_path).with_context(|| {
        format!(
            "stat trace `{}` for nvtx-tree fingerprint",
            trace_path.display()
        )
    })
}

pub(crate) fn sidecar_is_fresh_for_trace(trace_path: &Path) -> Result<bool> {
    let path = sidecar_path_for(trace_path);
    let fp = source_fingerprint(trace_path)?;
    sidecar_is_fresh(&path, fp)
}

/// Build the sidecar if missing or stale; return its path. Also
/// registers `nsight.nvtx_tree` over the freshly-written sidecar on
/// `trace`'s connection so subsequent SQL on the same `Trace` handle
/// can query the view directly. SQL-only callers can therefore call
/// this once and then issue `SELECT ... FROM nsight.nvtx_tree`.
pub fn ensure_sidecar(trace: &Trace) -> Result<PathBuf> {
    let path = sidecar_path_for(trace.path());
    let fp = source_fingerprint(trace.path())?;
    if sidecar_is_fresh(&path, fp)? {
        log::debug!(
            "nvtx_tree: warm sidecar at {} ({} bytes)",
            path.display(),
            fs::metadata(&path).map(|m| m.len()).unwrap_or(0),
        );
        attach_view(trace, &path)?;
        return Ok(path);
    }
    let records = compute(trace).context("computing NVTX tree")?;
    write_parquet(&path, fp, &records)?;
    log::info!(
        "nvtx_tree: built sidecar at {} ({} ranges)",
        path.display(),
        records.len()
    );
    attach_view(trace, &path)?;
    Ok(path)
}

fn attach_view(trace: &Trace, sidecar: &Path) -> Result<()> {
    let Some(sql) = view_sql_for(sidecar) else {
        log::warn!(
            "nvtx_tree: sidecar path is not valid UTF-8, skipping view registration: {}",
            sidecar.display(),
        );
        return Ok(());
    };
    trace.conn().execute(&sql, []).with_context(|| {
        format!(
            "registering nsight.nvtx_tree view from {}",
            sidecar.display()
        )
    })?;
    Ok(())
}

/// Build (if needed) and load all rows into memory.
pub fn build_or_load(trace: &Trace) -> Result<NvtxTree> {
    let path = ensure_sidecar(trace)?;
    if !path.exists() {
        return Ok(NvtxTree::empty());
    }
    let records = read_parquet(&path).context("loading nvtx-tree sidecar")?;
    Ok(NvtxTree::from_records(records))
}

/// Load only if a fresh sidecar exists on disk; never trigger a build.
/// Mirrors `runtime_nvtx_parent::load_if_present` for cheap callers
/// that don't want to pay the build cost on a cold cache.
pub fn load_if_present(trace: &Trace) -> Result<Option<NvtxTree>> {
    let path = sidecar_path_for(trace.path());
    let fp = source_fingerprint(trace.path())?;
    if !sidecar_is_fresh(&path, fp)? {
        return Ok(None);
    }
    let records = read_parquet(&path).context("loading nvtx-tree sidecar")?;
    Ok(Some(NvtxTree::from_records(records)))
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

// ----- compute -------------------------------------------------------------

#[derive(Debug, Clone)]
struct NvtxRow {
    rowid: i64,
    start: i64,
    end: Option<i64>,
    global_tid: i64,
    domain_id: i64,
    name: String,
}

fn trace_has_nvtx(trace: &Trace) -> bool {
    trace
        .conn()
        .execute("SELECT 1 FROM nsight.NVTX_EVENTS LIMIT 0", [])
        .is_ok()
}

/// Resolve NVTX `eventType` ids by NAME via the trace's own
/// `ENUM_NSYS_EVENT_TYPE` catalog. Version-robust: a future
/// nsys that renumbers the enum is followed automatically, so no frozen
/// magic-int table drives classification. Falls back to `default_ids` when
/// the catalog table is absent or matches nothing, so a trace without it
/// still resolves. `names` are fixed internal constants, so embedding them
/// as SQL string literals is injection-safe.
pub(crate) fn nvtx_event_type_ids(trace: &Trace, names: &[&str], default_ids: &[i64]) -> Vec<i64> {
    if !trace.has_table("ENUM_NSYS_EVENT_TYPE") {
        return default_ids.to_vec();
    }
    let list = names
        .iter()
        .map(|n| format!("'{n}'"))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT CAST(id AS BIGINT) FROM nsight.ENUM_NSYS_EVENT_TYPE WHERE name IN ({list})"
    );
    let resolved = (|| -> Result<Vec<i64>> {
        let mut stmt = trace.conn().prepare(&sql)?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(r) = rows.next()? {
            // Skip a NULL `id` rather than failing the whole query: one
            // NULL must not discard every legitimately-resolved id and
            // silently revert the entire set to `default_ids`, which
            // would defeat the by-name robustness.
            if let Some(id) = r.get::<_, Option<i64>>(0)? {
                out.push(id);
            }
        }
        Ok(out)
    })();
    match resolved {
        Ok(ids) if !ids.is_empty() => ids,
        _ => default_ids.to_vec(),
    }
}

fn collect_rows(trace: &Trace) -> Result<Vec<NvtxRow>> {
    let global_tid = crate::sql_expr::u64_bits_to_i64("n.globalTid");
    // Only true ranges enter the tree. Without this filter,
    // NvtxDomainCreate / NvtxMark rows (which also carry a start) pollute
    // the tree as bogus zero/instant "ranges". Resolved by name so the set
    // tracks the trace's own catalog.
    let range_ids = nvtx_event_type_ids(
        trace,
        &[
            "NvtxPushPopRange",
            "NvtxStartEndRange",
            "NvtxtPushPopRange",
            "NvtxtStartEndRange",
        ],
        &[59, 60, 70, 71],
    );
    let range_list = range_ids
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        r#"SELECT n.rowid,
                        n.start,
                        n."end",
                        {global_tid},
                        COALESCE(n.domainId, 0) AS domain_id,
                        COALESCE(n.text, s.value, '') AS name
                 FROM nsight.NVTX_EVENTS n
                 LEFT JOIN nsight.StringIds s ON n.textId = s.id
                 WHERE n.start IS NOT NULL
                   AND n.globalTid IS NOT NULL
                   AND n.eventType IN ({range_list})"#
    );
    let mut stmt = trace.conn().prepare(&sql)?;
    let mut rows = stmt.query([])?;
    let mut out = Vec::new();
    while let Some(r) = rows.next()? {
        out.push(NvtxRow {
            rowid: r.get(0)?,
            start: r.get(1)?,
            end: r.get(2)?,
            global_tid: r.get(3)?,
            domain_id: r.get(4)?,
            name: r.get(5)?,
        });
    }
    Ok(out)
}

fn compute(trace: &Trace) -> Result<Vec<NvtxTreeRecord>> {
    if !trace_has_nvtx(trace) {
        log::debug!("nvtx_tree: NVTX_EVENTS absent - empty result");
        return Ok(Vec::new());
    }
    let rows = collect_rows(trace).context("collecting NVTX rows for tree build")?;
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    Ok(compute_from_rows(rows))
}

/// Pure function over already-collected rows. Split out so unit tests
/// can exercise the algorithm without a DuckDB round-trip.
///
/// The two passes:
///
/// 1. Per-group (tid, domain) stack scan emits `(range_id,
///    parent_range_id, depth, ...)` partial records; rayon-parallel
///    across groups because each group's stack is independent.
/// 2. Topological path materialization: sort all rows by depth ASC,
///    accumulate `path = parent_path + '/' + escape(name)` keyed by
///    `range_id`. Single-threaded - the map is small and the work is
///    a few string allocations per row.
///
/// Output is sorted by `(start ASC, depth ASC)` so parquet row-group
/// skipping helps time-range queries.
fn compute_from_rows(rows: Vec<NvtxRow>) -> Vec<NvtxTreeRecord> {
    // Step 1: bucket by (tid, domain). BTreeMap for deterministic
    // group order in debug logs / golden tests.
    let mut groups: BTreeMap<(i64, i64), Vec<NvtxRow>> = BTreeMap::new();
    for row in rows {
        groups
            .entry((row.global_tid, row.domain_id))
            .or_default()
            .push(row);
    }

    // Step 2: per-group stack scan. Each group produces partial
    // records with `path` left empty - that's filled in step 3 once we
    // know every parent's name.
    let partials: Vec<Vec<NvtxTreeRecord>> = groups
        .into_par_iter()
        .map(|(_key, mut group)| stack_scan_group(&mut group))
        .collect();

    // Step 3: flatten partials, then materialize paths in a single
    // depth-ordered pass. Working maps owned by this scope so we can
    // release them once paths are written.
    let total: usize = partials.iter().map(Vec::len).sum();
    let mut records: Vec<NvtxTreeRecord> = Vec::with_capacity(total);
    for chunk in partials {
        records.extend(chunk);
    }
    materialize_paths(&mut records);

    // Step 4: sort for parquet row-group skipping.
    records.sort_by(|a, b| {
        a.start
            .cmp(&b.start)
            .then(a.depth.cmp(&b.depth))
            .then(a.range_id.cmp(&b.range_id))
    });
    records
}

/// Per-group stack scan. Mirrors `nvtx_nesting::compute_from_rows`'s
/// shape but additionally carries `range_id` on the stack so each
/// emitted record gets its `parent_range_id`.
fn stack_scan_group(group: &mut [NvtxRow]) -> Vec<NvtxTreeRecord> {
    // The nesting walk (sort-by-start + descending-end stack + touching-
    // boundary rule) is shared with `nvtx_nesting` via `nvtx_stack`; here
    // we just project each row to its tree record (path filled in step 3).
    crate::nvtx_stack::scan_group(group, |row, parent_range_id, depth| NvtxTreeRecord {
        range_id: row.rowid,
        parent_range_id,
        depth: i32::try_from(depth).unwrap_or(i32::MAX),
        domain_id: row.domain_id,
        name: row.name.clone(),
        path: String::new(),
        start: row.start,
        end: row.end,
        duration_ns: row.end.map(|e| e - row.start),
        global_tid: row.global_tid,
    })
}

impl crate::nvtx_stack::NvtxScanRow for NvtxRow {
    fn start(&self) -> i64 {
        self.start
    }
    fn end(&self) -> Option<i64> {
        self.end
    }
    fn rowid(&self) -> i64 {
        self.rowid
    }
}

/// Escape `/` in a name so the slash-joined `path` is unambiguous.
/// Backslashes get doubled first so the escape itself round-trips.
fn escape_name_for_path(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.chars() {
        match ch {
            '\\' => out.push_str(r"\\"),
            '/' => out.push_str(r"\/"),
            c => out.push(c),
        }
    }
    out
}

/// Fill `path` on every record. Records are read by `range_id` lookup,
/// so we process depth ASC and accumulate parent paths.
fn materialize_paths(records: &mut [NvtxTreeRecord]) {
    // Index into `records` by depth bucket so we can iterate
    // top-of-tree first without re-sorting the output (the final
    // sort happens after this).
    let mut by_depth: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for (i, rec) in records.iter().enumerate() {
        by_depth.entry(rec.depth).or_default().push(i);
    }

    // `range_id -> path` accumulator. Keeps owned `String`s because the
    // `path` field on each record needs an independent allocation
    // anyway.
    let mut paths: HashMap<i64, String> = HashMap::with_capacity(records.len());

    for (_depth, idxs) in by_depth {
        for idx in idxs {
            // Indexing into `records` here is safe - `idxs` was built
            // by enumerating the same slice and we never mutate its
            // length in this function.
            let Some(rec) = records.get(idx) else {
                continue;
            };
            let escaped = escape_name_for_path(&rec.name);
            let path = match rec.parent_range_id.and_then(|p| paths.get(&p).cloned()) {
                Some(parent_path) => format!("{parent_path}/{escaped}"),
                None => escaped,
            };
            paths.insert(rec.range_id, path.clone());
            if let Some(out) = records.get_mut(idx) {
                out.path = path;
            }
        }
    }
}

// ----- parquet I/O ---------------------------------------------------------

fn parquet_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("range_id", DataType::Int64, false),
        Field::new("parent_range_id", DataType::Int64, true),
        Field::new("depth", DataType::Int32, false),
        Field::new("domain_id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("path", DataType::Utf8, false),
        Field::new("start", DataType::Int64, false),
        Field::new("end", DataType::Int64, true),
        Field::new("duration_ns", DataType::Int64, true),
        Field::new("global_tid", DataType::Int64, false),
    ]))
}

const KV_VERSION: &str = "veloq.nvtx_tree.version";

fn write_parquet(path: &Path, fp: SourceFingerprint, records: &[NvtxTreeRecord]) -> Result<()> {
    let schema = parquet_schema();

    let mut range_ids: Vec<i64> = Vec::with_capacity(records.len());
    let mut parents: Vec<Option<i64>> = Vec::with_capacity(records.len());
    let mut depths: Vec<i32> = Vec::with_capacity(records.len());
    let mut domains: Vec<i64> = Vec::with_capacity(records.len());
    let mut names = StringBuilder::new();
    let mut paths = StringBuilder::new();
    let mut starts: Vec<i64> = Vec::with_capacity(records.len());
    let mut ends: Vec<Option<i64>> = Vec::with_capacity(records.len());
    let mut durations: Vec<Option<i64>> = Vec::with_capacity(records.len());
    let mut tids: Vec<i64> = Vec::with_capacity(records.len());

    for r in records {
        range_ids.push(r.range_id);
        parents.push(r.parent_range_id);
        depths.push(r.depth);
        domains.push(r.domain_id);
        names.append_value(&r.name);
        paths.append_value(&r.path);
        starts.push(r.start);
        ends.push(r.end);
        durations.push(r.duration_ns);
        tids.push(r.global_tid);
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(range_ids)),
        Arc::new(Int64Array::from(parents)),
        Arc::new(Int32Array::from(depths)),
        Arc::new(Int64Array::from(domains)),
        Arc::new(names.finish()),
        Arc::new(paths.finish()),
        Arc::new(Int64Array::from(starts)),
        Arc::new(Int64Array::from(ends)),
        Arc::new(Int64Array::from(durations)),
        Arc::new(Int64Array::from(tids)),
    ];
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns)
        .context("assembling RecordBatch for nvtx-tree sidecar")?;

    let kv = crate::sidecar::freshness_kv(KV_VERSION, NVTX_TREE_VERSION, fp);
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_key_value_metadata(Some(kv))
        .build();

    crate::sidecar::atomic_publish(path, |tmp| {
        let file = File::create(tmp)
            .with_context(|| format!("creating {} for nvtx-tree sidecar", tmp.display()))?;
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    })
}

fn read_parquet(path: &Path) -> Result<Vec<NvtxTreeRecord>> {
    let file = File::open(path)
        .with_context(|| format!("opening nvtx-tree sidecar {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;
    let mut out: Vec<NvtxTreeRecord> = Vec::new();
    for batch in reader {
        let batch = batch?;
        let range_ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("nvtx-tree: range_id column missing/wrong type")?;
        let parents = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("nvtx-tree: parent_range_id column missing/wrong type")?;
        let depths = batch
            .column(2)
            .as_any()
            .downcast_ref::<Int32Array>()
            .context("nvtx-tree: depth column missing/wrong type")?;
        let domains = batch
            .column(3)
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("nvtx-tree: domain_id column missing/wrong type")?;
        let names = batch
            .column(4)
            .as_any()
            .downcast_ref::<StringArray>()
            .context("nvtx-tree: name column missing/wrong type")?;
        let paths = batch
            .column(5)
            .as_any()
            .downcast_ref::<StringArray>()
            .context("nvtx-tree: path column missing/wrong type")?;
        let starts = batch
            .column(6)
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("nvtx-tree: start column missing/wrong type")?;
        let ends = batch
            .column(7)
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("nvtx-tree: end column missing/wrong type")?;
        let durations = batch
            .column(8)
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("nvtx-tree: duration_ns column missing/wrong type")?;
        let tids = batch
            .column(9)
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("nvtx-tree: global_tid column missing/wrong type")?;
        for i in 0..batch.num_rows() {
            out.push(NvtxTreeRecord {
                range_id: range_ids.value(i),
                parent_range_id: if parents.is_null(i) {
                    None
                } else {
                    Some(parents.value(i))
                },
                depth: depths.value(i),
                domain_id: domains.value(i),
                name: names.value(i).to_string(),
                path: paths.value(i).to_string(),
                start: starts.value(i),
                end: if ends.is_null(i) {
                    None
                } else {
                    Some(ends.value(i))
                },
                duration_ns: if durations.is_null(i) {
                    None
                } else {
                    Some(durations.value(i))
                },
                global_tid: tids.value(i),
            });
        }
    }
    Ok(out)
}

fn sidecar_is_fresh(path: &Path, fp: SourceFingerprint) -> Result<bool> {
    crate::sidecar::is_fresh(path, KV_VERSION, NVTX_TREE_VERSION, fp, "nvtx_tree")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(rowid: i64, start: i64, end: Option<i64>, tid: i64, domain: i64, name: &str) -> NvtxRow {
        NvtxRow {
            rowid,
            start,
            end,
            global_tid: tid,
            domain_id: domain,
            name: name.to_string(),
        }
    }

    fn tree_record(
        range_id: i64,
        parent_range_id: Option<i64>,
        depth: i32,
        domain_id: i64,
        name: &str,
        start: i64,
        end: i64,
    ) -> NvtxTreeRecord {
        NvtxTreeRecord {
            range_id,
            parent_range_id,
            depth,
            domain_id,
            name: name.to_string(),
            path: name.to_string(),
            start,
            end: Some(end),
            duration_ns: Some(end - start),
            global_tid: 7,
        }
    }

    fn find(records: &[NvtxTreeRecord], range_id: i64) -> Result<&NvtxTreeRecord> {
        records
            .iter()
            .find(|r| r.range_id == range_id)
            .with_context(|| format!("range_id {range_id} not present in records"))
    }

    /// Nested push/pop on a single thread -> outer first by depth,
    /// inner gets outer as parent, path is "outer/inner".
    #[test]
    fn nested_stack_emits_parent_and_path() -> Result<()> {
        let rows = vec![
            r(1, 0, Some(100), 7, 0, "outer"),
            r(2, 40, Some(60), 7, 0, "inner"),
        ];
        let out = compute_from_rows(rows);
        let outer = find(&out, 1)?;
        let inner = find(&out, 2)?;
        assert_eq!(outer.parent_range_id, None);
        assert_eq!(outer.depth, 0);
        assert_eq!(outer.path, "outer");
        assert_eq!(inner.parent_range_id, Some(1));
        assert_eq!(inner.depth, 1);
        assert_eq!(inner.path, "outer/inner");
        Ok(())
    }

    /// Unterminated range (end is None - an instant marker, in the
    /// codebase's terminology) gets the post-pop depth + parent but
    /// never pushes onto the stack. Subsequent ranges therefore see
    /// the same stack the marker did.
    #[test]
    fn unterminated_range_does_not_push() -> Result<()> {
        let rows = vec![
            r(1, 0, Some(100), 7, 0, "outer"),
            r(2, 10, None, 7, 0, "marker"),
            r(3, 20, Some(30), 7, 0, "inner"),
        ];
        let out = compute_from_rows(rows);
        assert_eq!(find(&out, 2)?.end, None);
        assert_eq!(find(&out, 2)?.duration_ns, None);
        assert_eq!(find(&out, 2)?.parent_range_id, Some(1));
        // `inner` must still see only `outer` on the stack (marker
        // didn't push), so depth=1 and parent=outer.
        assert_eq!(find(&out, 3)?.depth, 1);
        assert_eq!(find(&out, 3)?.parent_range_id, Some(1));
        assert_eq!(find(&out, 3)?.path, "outer/inner");
        Ok(())
    }

    /// Two threads each open their own stack - no cross-tid
    /// contamination even if their intervals overlap.
    #[test]
    fn per_tid_stacks_are_isolated() -> Result<()> {
        let rows = vec![
            r(1, 0, Some(100), 7, 0, "tid7_outer"),
            r(2, 50, Some(80), 7, 0, "tid7_inner"),
            r(3, 10, Some(90), 8, 0, "tid8_outer"),
            r(4, 20, Some(40), 8, 0, "tid8_inner"),
        ];
        let out = compute_from_rows(rows);
        assert_eq!(find(&out, 2)?.parent_range_id, Some(1));
        assert_eq!(find(&out, 4)?.parent_range_id, Some(3));
        assert_eq!(find(&out, 2)?.path, "tid7_outer/tid7_inner");
        assert_eq!(find(&out, 4)?.path, "tid8_outer/tid8_inner");
        Ok(())
    }

    /// Two domains on the same tid form independent stacks too,
    /// mirroring `nvtx_nesting`'s grouping.
    #[test]
    fn per_domain_stacks_are_isolated() -> Result<()> {
        let rows = vec![
            r(1, 0, Some(100), 7, 1, "domain1_outer"),
            r(2, 10, Some(20), 7, 2, "domain2_outer"),
            r(3, 30, Some(40), 7, 2, "domain2_inner"),
        ];
        let out = compute_from_rows(rows);
        // `domain2_inner` is on domain 2 - its parent is the outer in
        // domain 2, not the wider range in domain 1.
        assert_eq!(find(&out, 3)?.parent_range_id, None);
        // Same start ordering: domain 2 has the inner at 30, which
        // arrives after domain2_outer (10..20) has already closed; so
        // it's a root at depth 0 on its own stack.
        assert_eq!(find(&out, 3)?.depth, 0);
        Ok(())
    }

    /// Touching boundaries (`end == next.start`) do NOT nest - pin
    /// the same `<=` semantics that `nvtx_nesting` uses.
    #[test]
    fn touching_boundary_does_not_nest() -> Result<()> {
        let rows = vec![r(1, 0, Some(10), 1, 0, "a"), r(2, 10, Some(20), 1, 0, "b")];
        let out = compute_from_rows(rows);
        assert_eq!(find(&out, 2)?.parent_range_id, None);
        assert_eq!(find(&out, 2)?.depth, 0);
        Ok(())
    }

    /// Names containing `/` get escaped in `path` so the join character
    /// stays unambiguous.
    #[test]
    fn slashes_in_names_are_escaped() -> Result<()> {
        let rows = vec![
            r(1, 0, Some(100), 1, 0, "a/b"),
            r(2, 10, Some(20), 1, 0, "c\\d"),
        ];
        let out = compute_from_rows(rows);
        assert_eq!(find(&out, 1)?.path, r"a\/b");
        // Both the literal backslash from the name and the parent's
        // escaped slash round-trip cleanly.
        assert_eq!(find(&out, 2)?.path, r"a\/b/c\\d");
        Ok(())
    }

    /// `domain_id` is taken straight from the row; `compute_rows`
    /// collects NULL as 0 via `COALESCE` so the sidecar can stay
    /// non-nullable.
    #[test]
    fn domain_id_defaults_to_zero_when_unknown() -> Result<()> {
        // `collect_rows` does the COALESCE at SQL time; the algorithm
        // here just reads whatever it's given. So this test simulates
        // the post-COALESCE shape.
        let rows = vec![r(1, 0, Some(10), 1, 0, "default")];
        let out = compute_from_rows(rows);
        assert_eq!(find(&out, 1)?.domain_id, 0);
        Ok(())
    }

    /// Roundtrip preserves every field - including nullable end /
    /// duration / parent_range_id - and the fingerprint validates.
    #[test]
    fn parquet_roundtrip_preserves_records() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let path = tmp.path().join("nvtx_tree.parquet");
        let records = vec![
            NvtxTreeRecord {
                range_id: 1,
                parent_range_id: None,
                depth: 0,
                domain_id: 0,
                name: "outer".into(),
                path: "outer".into(),
                start: 0,
                end: Some(100),
                duration_ns: Some(100),
                global_tid: 7,
            },
            NvtxTreeRecord {
                range_id: 2,
                parent_range_id: Some(1),
                depth: 1,
                domain_id: 0,
                name: "inner".into(),
                path: "outer/inner".into(),
                start: 10,
                end: Some(20),
                duration_ns: Some(10),
                global_tid: 7,
            },
            // Instant marker: NULL end / duration; must round-trip.
            NvtxTreeRecord {
                range_id: 3,
                parent_range_id: Some(1),
                depth: 1,
                domain_id: 0,
                name: "marker".into(),
                path: "outer/marker".into(),
                start: 50,
                end: None,
                duration_ns: None,
                global_tid: 7,
            },
        ];
        let fp = SourceFingerprint {
            mtime_secs: 1_234_567_890,
            size: 4096,
        };
        write_parquet(&path, fp, &records)?;
        assert!(sidecar_is_fresh(&path, fp)?);
        let loaded = read_parquet(&path)?;
        assert_eq!(loaded, records);
        Ok(())
    }

    /// Fingerprint mismatch (different mtime or size) invalidates.
    #[test]
    fn mtime_or_size_change_invalidates_sidecar() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let path = tmp.path().join("nvtx_tree.parquet");
        let fp = SourceFingerprint {
            mtime_secs: 100,
            size: 200,
        };
        write_parquet(&path, fp, &[])?;
        assert!(sidecar_is_fresh(&path, fp)?);
        assert!(!sidecar_is_fresh(
            &path,
            SourceFingerprint {
                mtime_secs: 101,
                size: 200,
            },
        )?);
        assert!(!sidecar_is_fresh(
            &path,
            SourceFingerprint {
                mtime_secs: 100,
                size: 201,
            },
        )?);
        Ok(())
    }

    /// A sidecar written under a different `NVTX_TREE_VERSION` is
    /// stale even if the fingerprint matches - readers rebuild
    /// silently.
    #[test]
    fn version_mismatch_invalidates_sidecar() -> Result<()> {
        // Easiest way to simulate a version mismatch is to write a
        // parquet whose `KV_VERSION` key holds a different number,
        // then assert `sidecar_is_fresh` returns false.
        let tmp = tempfile::tempdir()?;
        let path = tmp.path().join("nvtx_tree.parquet");
        let schema = parquet_schema();
        // Empty batch - we only care about the KV metadata for this
        // freshness check.
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(Vec::<i64>::new())),
            Arc::new(Int64Array::from(Vec::<Option<i64>>::new())),
            Arc::new(Int32Array::from(Vec::<i32>::new())),
            Arc::new(Int64Array::from(Vec::<i64>::new())),
            Arc::new(StringBuilder::new().finish()),
            Arc::new(StringBuilder::new().finish()),
            Arc::new(Int64Array::from(Vec::<i64>::new())),
            Arc::new(Int64Array::from(Vec::<Option<i64>>::new())),
            Arc::new(Int64Array::from(Vec::<Option<i64>>::new())),
            Arc::new(Int64Array::from(Vec::<i64>::new())),
        ];
        let batch = RecordBatch::try_new(Arc::clone(&schema), columns)?;
        use parquet::file::metadata::KeyValue;
        let kv = vec![
            KeyValue::new(
                KV_VERSION.to_string(),
                Some((NVTX_TREE_VERSION + 1).to_string()),
            ),
            KeyValue::new(
                crate::sidecar::KV_MTIME.to_string(),
                Some("100".to_string()),
            ),
            KeyValue::new(crate::sidecar::KV_SIZE.to_string(), Some("200".to_string())),
        ];
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_key_value_metadata(Some(kv))
            .build();
        let file = File::create(&path)?;
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
        assert!(!sidecar_is_fresh(
            &path,
            SourceFingerprint {
                mtime_secs: 100,
                size: 200,
            },
        )?);
        Ok(())
    }

    /// `stack_at` returns ancestors outer->inner for a tid+timestamp
    /// inside several nested ranges, and empty when nothing covers
    /// the point.
    #[test]
    fn stack_at_returns_outer_to_inner_chain() {
        let rows = vec![
            r(1, 0, Some(100), 7, 0, "outer"),
            r(2, 40, Some(80), 7, 0, "mid"),
            r(3, 50, Some(60), 7, 0, "inner"),
        ];
        let records = compute_from_rows(rows);
        let tree = NvtxTree::from_records(records);

        let stack = tree.stack_at(7, 55);
        let names: Vec<&str> = stack.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, vec!["outer", "mid", "inner"]);

        // Outside any range.
        assert!(tree.stack_at(7, 200).is_empty());
        // Wrong tid.
        assert!(tree.stack_at(99, 55).is_empty());
    }

    /// When unrelated ranges cover the same timestamp on one tid, the
    /// public stack API still returns one deterministic parent chain
    /// rather than sibling rows that cannot form a stack.
    #[test]
    fn stack_at_returns_one_chain_for_unrelated_covering_ranges() {
        let records = vec![
            tree_record(1, None, 0, 1, "domain1_root", 0, 100),
            tree_record(2, None, 0, 2, "domain2_root", 10, 90),
            tree_record(3, Some(2), 1, 2, "domain2_child", 20, 80),
        ];
        let tree = NvtxTree::from_records(records);

        let stack = tree.stack_at(7, 50);
        let names: Vec<&str> = stack.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, vec!["domain2_root", "domain2_child"]);
    }

    /// `ancestors` walks up through parent links - emit self first,
    /// then parents toward the root.
    #[test]
    fn ancestors_walks_parent_chain() {
        let rows = vec![
            r(1, 0, Some(100), 7, 0, "outer"),
            r(2, 40, Some(80), 7, 0, "mid"),
            r(3, 50, Some(60), 7, 0, "inner"),
        ];
        let records = compute_from_rows(rows);
        let tree = NvtxTree::from_records(records);

        let names: Vec<&str> = tree.ancestors(3).iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, vec!["inner", "mid", "outer"]);
    }
}
