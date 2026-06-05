//! Schema-probe helpers shared by [`inspect`] and [`search`].
//!
//! NSys schema evolves over time: optional columns (`mangledName`,
//! `registersPerThread`, `globalPid`, …) appear in some report
//! versions but not others. A query that hard-codes those columns
//! breaks against older reports; a query that `LEFT JOIN`s every
//! optional table is slower than needed. The pattern these helpers
//! support is the middle path: probe `information_schema.columns`
//! once per run, then inject either the real column reference
//! (`t."<col>"`) or a literal `NULL` into the SQL string.
//!
//! [`inspect`]: crate::inspect
//! [`search`]: crate::search

use anyhow::Result;
use duckdb::Connection;
use std::collections::{HashMap, HashSet};

/// Map of `table_name` → set of column names present in that table.
/// Empty entries indicate "table absent in this trace"; callers can
/// still ask via [`has`] / [`maybe_col`] and get the right answer.
pub type ColumnMap = HashMap<&'static str, HashSet<String>>;

/// Probe every table named in `tables` against the `nsight` schema's
/// `information_schema.columns` view. Tables with zero present
/// columns are *omitted* from the returned map — callers should
/// treat `ColumnMap::get(name)` returning `None` as "table absent."
///
/// One round-trip prepares the statement; per-table queries reuse
/// it. Cheap enough to call once per verb invocation.
pub fn load_columns(conn: &Connection, tables: &[&'static str]) -> Result<ColumnMap> {
    let mut out: ColumnMap = HashMap::new();
    // Query by `table_schema`: `nsight` is a regular DuckDB schema, not
    // an attached catalog, so a `table_catalog = 'nsight'` filter returns
    // empty — a silent uncorrelated downgrade.
    let sql = "SELECT column_name FROM information_schema.columns \
               WHERE table_schema = 'nsight' AND table_name = ?";
    let mut stmt = conn.prepare(sql)?;
    for &t in tables {
        let mut rows = stmt.query([t])?;
        let mut cols: HashSet<String> = HashSet::new();
        while let Some(r) = rows.next()? {
            cols.insert(r.get::<_, String>(0)?);
        }
        if !cols.is_empty() {
            out.insert(t, cols);
        }
    }
    Ok(out)
}

/// The canonical table list every consumer probes today. Inspect
/// covers every kind it dispatches; search covers every kind it
/// projects. Both reach the same superset, so the list lives here.
pub const STANDARD_TABLES: &[&str] = &[
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "CUPTI_ACTIVITY_KIND_MEMSET",
    "CUPTI_ACTIVITY_KIND_RUNTIME",
    "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
    "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
    "CUDA_GRAPH_NODE_EVENTS",
    "CUDA_GRAPH_EVENTS",
    "CUPTI_ACTIVITY_KIND_CUDA_EVENT",
    "CUPTI_ACTIVITY_KIND_OVERHEAD",
    "OSRT_API",
    "NVTX_EVENTS",
    "COMPOSITE_EVENTS",
    "SAMPLING_CALLCHAINS",
    "ENUM_SAMPLING_THREAD_STATE",
];

/// Convenience: [`load_columns`] over [`STANDARD_TABLES`].
pub fn load_standard(conn: &Connection) -> Result<ColumnMap> {
    load_columns(conn, STANDARD_TABLES)
}

/// True iff `cols` has `col` present in `table`. Returns false for
/// both "column absent" and "table absent" — the maybe-NULL pattern
/// already conflates them.
pub fn has(cols: &ColumnMap, table: &'static str, col: &str) -> bool {
    cols.get(table).is_some_and(|s| s.contains(col))
}

/// SQL expression for a possibly-absent column. Returns `t."<col>"`
/// when present, `NULL` when absent. The caller injects the result
/// directly into a `format!` SQL string.
pub fn maybe_col(cols: &ColumnMap, table: &'static str, col: &str) -> String {
    if has(cols, table, col) {
        format!("t.\"{col}\"")
    } else {
        "NULL".to_string()
    }
}

/// Read a DuckDB column as `Option<String>` — `Null` and non-text
/// types collapse to `None`. Used by the maybe-NULL pattern when the
/// projected column was synthesised via [`maybe_col`].
pub fn opt_string(row: &duckdb::Row, idx: usize) -> Result<Option<String>, duckdb::Error> {
    match row.get_ref(idx)? {
        duckdb::types::ValueRef::Null => Ok(None),
        duckdb::types::ValueRef::Text(b) => Ok(Some(
            std::str::from_utf8(b).unwrap_or("<bad utf8>").to_string(),
        )),
        _ => Ok(None),
    }
}
