use super::sql::nvtx_style_label;
use super::{HIST_BOUNDARIES_NS, HistBucket, StatRow};
use crate::{EventKind, NsysQueryResult};
use duckdb::types::Value;
use std::collections::HashMap;
use veloq_nsys_data::{NvtxNesting, Trace};

/// Scope-wide window-function totals, repeated on every row by the
/// outer SELECT's `SUM(...) OVER ()` columns. Read once on the first
/// row and reused for the rest.
pub(super) struct StatsScope {
    pub(super) total_ns: i64,
    pub(super) total_count: i64,
    pub(super) total_groups: i64,
}

/// Run the prepared `sql` against `trace`, project each row into a
/// `StatRow`, and recover the scope-wide totals exposed by the window
/// functions in the outer SELECT. Carved out of `run` so the SELECT
/// alias -> struct field hydration is reviewable in isolation; bind
/// order and SQL assembly stay in the caller.
pub(super) fn hydrate_stats_rows(
    trace: &Trace,
    sql: &str,
    params: &[Value],
    hist: bool,
    nvtx_nesting: Option<&NvtxNesting>,
    domain_names: &HashMap<(i64, i64), String>,
) -> NsysQueryResult<(Vec<StatRow>, StatsScope)> {
    let rows = crate::query_sql::exec::query_rows(
        trace.conn(),
        sql,
        params,
        crate::query_sql::exec::STATS_AGGREGATE,
        |row| stats_sql_row(row, hist),
    )?;
    let scope = rows.first().map_or(
        StatsScope {
            total_ns: 0,
            total_count: 0,
            total_groups: 0,
        },
        StatsSqlRow::scope,
    );
    let out = rows
        .into_iter()
        .map(|row| stat_row_from_sql(row, nvtx_nesting, domain_names))
        .collect();
    Ok((out, scope))
}

struct StatsSqlRow {
    name: Option<String>,
    short_name_raw: Option<String>,
    kind: String,
    process_id: Option<i64>,
    device_id: Option<i32>,
    context_id: Option<i64>,
    stream_id: Option<i64>,
    graph_id: Option<i64>,
    graph_node_id: Option<i64>,
    count: i64,
    total_ns: i64,
    avg_ns: i64,
    min_ns: i64,
    max_ns: i64,
    p50_ns: i64,
    p95_ns: i64,
    p99_ns: i64,
    bytes_total: Option<i64>,
    gbps: Option<f64>,
    event_type: Option<i64>,
    nvtx_style_raw: Option<String>,
    nvtx_parent_rowid: Option<i64>,
    nvtx_parent_name_raw: Option<String>,
    nvtx_path_raw: Option<String>,
    nvtx_domain_id_raw: Option<i64>,
    nvtx_domain_pid_raw: Option<i64>,
    grid_x: Option<i64>,
    grid_y: Option<i64>,
    grid_z: Option<i64>,
    block_x: Option<i64>,
    block_y: Option<i64>,
    block_z: Option<i64>,
    scope_total_ns: i64,
    scope_total_count: i64,
    scope_total_groups: i64,
    histogram: Option<Vec<i64>>,
}

impl StatsSqlRow {
    fn scope(&self) -> StatsScope {
        StatsScope {
            total_ns: self.scope_total_ns,
            total_count: self.scope_total_count,
            total_groups: self.scope_total_groups,
        }
    }
}

fn stats_sql_row(row: &duckdb::Row<'_>, hist: bool) -> Result<StatsSqlRow, duckdb::Error> {
    // Read by SELECT alias rather than by integer index so a future
    // column reorder/insert in the outer SELECT can't silently shift
    // every value. Aliases match the column expressions built in `run`.
    let histogram = if hist {
        let n = HIST_BOUNDARIES_NS.len() + 1; // +1 tail
        let mut buckets = Vec::with_capacity(n);
        for i in 0..n {
            buckets.push(row.get::<_, i64>(format!("hist_b{i}").as_str())?);
        }
        Some(buckets)
    } else {
        None
    };

    Ok(StatsSqlRow {
        name: row.get("name")?,
        short_name_raw: row.get("short_name")?,
        kind: row.get("kind")?,
        process_id: row.get("process_id")?,
        device_id: row.get("device_id")?,
        context_id: row.get("context_id")?,
        stream_id: row.get("stream_id")?,
        graph_id: row.get("graph_id")?,
        graph_node_id: row.get("graph_node_id")?,
        count: row.get("count")?,
        total_ns: row.get("total_ns")?,
        avg_ns: row.get("avg_ns")?,
        min_ns: row.get("min_ns")?,
        max_ns: row.get("max_ns")?,
        p50_ns: row.get("p50_ns")?,
        p95_ns: row.get("p95_ns")?,
        p99_ns: row.get("p99_ns")?,
        bytes_total: row.get("bytes_total")?,
        gbps: row.get("gbps")?,
        event_type: row.get("event_type")?,
        nvtx_style_raw: row.get("nvtx_style")?,
        nvtx_parent_rowid: row.get("nvtx_parent_rowid")?,
        nvtx_parent_name_raw: row.get("nvtx_parent_name")?,
        nvtx_path_raw: row.get("nvtx_path")?,
        nvtx_domain_id_raw: row.get("nvtx_domain_id")?,
        nvtx_domain_pid_raw: row.get("nvtx_domain_pid")?,
        grid_x: row.get("grid_x")?,
        grid_y: row.get("grid_y")?,
        grid_z: row.get("grid_z")?,
        block_x: row.get("block_x")?,
        block_y: row.get("block_y")?,
        block_z: row.get("block_z")?,
        scope_total_ns: row.get("scope_total_ns")?,
        scope_total_count: row.get("scope_total_count")?,
        scope_total_groups: row.get("scope_total_groups")?,
        histogram,
    })
}

fn stat_row_from_sql(
    row: StatsSqlRow,
    nvtx_nesting: Option<&NvtxNesting>,
    domain_names: &HashMap<(i64, i64), String>,
) -> StatRow {
    let nvtx_style: Option<&'static str> = row.nvtx_style_raw.as_deref().map(nvtx_style_label);

    // Recover the typed EventKind from the SQL-side label so we can hand back
    // a stable `&'static str` without an open-coded string->string dispatch
    // table.
    let kind_static: &'static str = EventKind::parse(&row.kind)
        .map(EventKind::as_str)
        .unwrap_or("unknown");

    // Always populate `short_name` for kernel rows so the schema is stable
    // across `--group-by` modes and agents can roll demangled rows back to
    // their shortName group. For memcpy/memset it's redundant with `name`, so
    // omit it.
    let short_name = if kind_static == "kernel" {
        row.short_name_raw
    } else {
        None
    };

    // Row key: `(kind, name?, dev?, stream?, ctx?, graph?, graph_node?,
    // style?, nvtx?, grid?, block?)` pipe-joined. Only fields populated by
    // the active `--group-by` contribute.
    let mut key_parts = vec![kind_static.to_string()];
    if let Some(n) = row.name.as_deref() {
        key_parts.push(n.to_string());
    }
    if let Some(pid) = row.process_id {
        key_parts.push(format!("pid:{pid}"));
    }
    if let Some(d) = row.device_id {
        key_parts.push(format!("dev:{d}"));
    }
    if let Some(s) = row.stream_id {
        key_parts.push(format!("stream:{s}"));
    }
    if let Some(c) = row.context_id {
        key_parts.push(format!("ctx:{c}"));
    }
    if let Some(g) = row.graph_id {
        key_parts.push(format!("graph:{g}"));
    }
    if let Some(gn) = row.graph_node_id {
        key_parts.push(format!("graph_node:{gn}"));
    }
    if let Some(style) = nvtx_style {
        key_parts.push(format!("style:{style}"));
    }

    let nvtx_parent_key: Option<String> = row.nvtx_parent_name_raw.as_deref().map(|_| {
        row.nvtx_parent_rowid
            .map(|rid| format!("nvtx:{rid}"))
            .unwrap_or_else(|| crate::nvtx_parent::NO_NVTX_KEY.to_string())
    });
    if let Some(npk) = nvtx_parent_key.as_deref() {
        key_parts.push(npk.to_string());
    }

    let nvtx_path_key: Option<String> = row.nvtx_path_raw.as_deref().map(|path| {
        if path == crate::nvtx_parent::NO_NVTX_NAME {
            crate::nvtx_parent::NO_NVTX_PATH_KEY.to_string()
        } else {
            format!("nvtx-path:{path}")
        }
    });
    if let Some(npk) = nvtx_path_key.as_deref() {
        key_parts.push(npk.to_string());
    }

    let is_real_nvtx_path_row = row
        .nvtx_path_raw
        .as_deref()
        .is_some_and(|p| p != crate::nvtx_parent::NO_NVTX_NAME);
    let (domain_id, domain_pid, domain_name) = match (
        is_real_nvtx_path_row,
        row.nvtx_domain_id_raw,
        row.nvtx_domain_pid_raw,
    ) {
        (true, Some(did), Some(pid)) => {
            key_parts.push(format!("domain:{pid}:{did}"));
            let name = domain_names.get(&(pid, did)).cloned();
            (Some(did), Some(pid), name)
        }
        _ => (None, None, None),
    };

    if let (Some(gx), Some(gy), Some(gz), Some(bx), Some(by), Some(bz)) = (
        row.grid_x,
        row.grid_y,
        row.grid_z,
        row.block_x,
        row.block_y,
        row.block_z,
    ) {
        key_parts.push(format!("grid:{gx}x{gy}x{gz}"));
        key_parts.push(format!("block:{bx}x{by}x{bz}"));
    }
    let key = key_parts.join("|");

    // Depth comes from the per-trace `nvtx_nesting` map computed once for the
    // request; only populated when the row attributes to a real range.
    let nvtx_parent_depth: Option<u8> = match (row.nvtx_parent_rowid, nvtx_nesting) {
        (Some(rid), Some(map)) => map.get(&rid).map(|e| e.depth),
        _ => None,
    };

    StatRow {
        key,
        name: row.name,
        kind: kind_static,
        short_name,
        process_id: row.process_id,
        device_id: row.device_id,
        context_id: row.context_id,
        stream_id: row.stream_id,
        graph_id: row.graph_id,
        graph_node_id: row.graph_node_id,
        count: row.count,
        total_ns: row.total_ns,
        avg_ns: row.avg_ns,
        min_ns: row.min_ns,
        max_ns: row.max_ns,
        p50_ns: row.p50_ns,
        p95_ns: row.p95_ns,
        p99_ns: row.p99_ns,
        bytes_total: row.bytes_total,
        gbps: row.gbps,
        percentage: 0.0,
        histogram: row.histogram,
        event_type: row.event_type,
        nvtx_style,
        nvtx_parent_key,
        nvtx_parent_name: row.nvtx_parent_name_raw,
        nvtx_parent_depth,
        nvtx_path_key,
        nvtx_path: row.nvtx_path_raw,
        domain_id,
        domain_pid,
        domain_name,
        grid_x: row.grid_x,
        grid_y: row.grid_y,
        grid_z: row.grid_z,
        block_x: row.block_x,
        block_y: row.block_y,
        block_z: row.block_z,
    }
}

pub(super) fn build_bucket_schema() -> Vec<HistBucket> {
    let mut buckets = Vec::with_capacity(HIST_BOUNDARIES_NS.len() + 1);
    let mut prev: i64 = 0;
    for &b in HIST_BOUNDARIES_NS {
        buckets.push(HistBucket {
            lo: prev,
            hi: Some(b),
        });
        prev = b;
    }
    // Open-ended tail bucket
    buckets.push(HistBucket { lo: prev, hi: None });
    buckets
}
