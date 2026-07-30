use super::NvtxTreeRecord;
use crate::{NsysDataResult, Trace};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};

// ----- compute -------------------------------------------------------------

#[derive(Debug, Clone)]
pub(super) struct NvtxRow {
    pub(super) rowid: i64,
    pub(super) start: i64,
    pub(super) end: Option<i64>,
    pub(super) global_tid: i64,
    pub(super) domain_id: i64,
    pub(super) name: String,
}

pub(super) fn trace_has_nvtx(trace: &Trace) -> bool {
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
    let resolved = (|| -> std::result::Result<Vec<i64>, duckdb::Error> {
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

pub(super) fn collect_rows(trace: &Trace) -> NsysDataResult<Vec<NvtxRow>> {
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
    let mut stmt = trace
        .conn()
        .prepare(&sql)
        .map_err(crate::NsysDataError::nvtx_tree_rows_prepare)?;
    let mut rows = stmt
        .query([])
        .map_err(crate::NsysDataError::nvtx_tree_rows_query)?;
    let mut out = Vec::new();
    while let Some(r) = rows
        .next()
        .map_err(crate::NsysDataError::nvtx_tree_rows_read)?
    {
        out.push(NvtxRow {
            rowid: r
                .get(0)
                .map_err(crate::NsysDataError::nvtx_tree_rows_read)?,
            start: r
                .get(1)
                .map_err(crate::NsysDataError::nvtx_tree_rows_read)?,
            end: r
                .get(2)
                .map_err(crate::NsysDataError::nvtx_tree_rows_read)?,
            global_tid: r
                .get(3)
                .map_err(crate::NsysDataError::nvtx_tree_rows_read)?,
            domain_id: r
                .get(4)
                .map_err(crate::NsysDataError::nvtx_tree_rows_read)?,
            name: r
                .get(5)
                .map_err(crate::NsysDataError::nvtx_tree_rows_read)?,
        });
    }
    Ok(out)
}

pub(super) fn compute(trace: &Trace) -> NsysDataResult<Vec<NvtxTreeRecord>> {
    if !trace_has_nvtx(trace) {
        log::debug!("nvtx_tree: NVTX_EVENTS absent - empty result");
        return Ok(Vec::new());
    }
    let rows = collect_rows(trace)?;
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let pool = trace.build_query_worker_pool()?;
    Ok(pool.install(|| compute_from_rows(rows)))
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
pub(super) fn compute_from_rows(rows: Vec<NvtxRow>) -> Vec<NvtxTreeRecord> {
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
