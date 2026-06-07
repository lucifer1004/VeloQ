//! `veloq stats --by size` — bytes-as-aggregate-unit stats.
//!
//! Instead of aggregating over event duration
//! (`end - start`), this module aggregates over each event's `bytes`
//! column. Valid for memcpy and memset rows (the only kinds carrying
//! a `bytes` column); explicit non-memop kinds in `--type` error
//! up-front, while `KindFilter::All` (the implicit default) narrows
//! to memcpy + memset at SQL time.
//!
//! Lives in a sibling module rather than inside [`crate::stats`] so:
//!
//! 1. The wire shape (`StatsBySizeResponse` / `StatBySizeRow`)
//!    surfaces under a hidden schema target (`stats-by-size`) without
//!    leaking the `_bytes` fields into the public stats schema —
//!    schemars sees this as a separate type rather than the existing
//!    `StatsResponse` with renamed fields.
//! 2. Promotion to public ungate is a single change: move the
//!    `SchemaTarget` from `HIDDEN_TARGETS` to `TARGETS` and drop the
//!    CLI env gate. Today's `stats::run` is untouched.
//!
//! The query reuses the per-kind SQL templates from
//! `crate::stats::per_kind_subquery` so the kind dispatch and
//! column-presence probes stay single-sourced; only the aggregator
//! changes (SUM(bytes), AVG(bytes), QUANTILE_CONT(bytes, …) instead
//! of the duration counterparts).

use crate::query_sql::{
    event_scan::{EventScanFilterOptions, NvtxFilterPolicy, event_scan_filter},
    event_semantics::EventSemantics,
    sort::order_by,
};
use crate::{EventKind, KindFilter, NsysQueryError, NsysQueryResult};
use duckdb::types::Value;
use serde::Serialize;
use std::path::Path;
use veloq_core::{Direction, SortKeyDef, SortKeySpec, SortSpec, time::TimeWindow};
use veloq_nsys_data::Trace;
use veloq_query::sql::SqlFragment;

/// Kinds that carry a `bytes` column and are therefore aggregatable
/// under `--by size`. Mirrors the [`crate::stats::ALLOWED_KINDS`]
/// allow-list convention.
pub const ALLOWED_KINDS: [EventKind; 2] = [EventKind::Memcpy, EventKind::Memset];
const BYTES_PRESENT_PREDICATES: &[&str] = &["t.bytes IS NOT NULL"];

/// Sort keys this verb accepts. `bytes` / `bytes_total` are aliases
/// for `total`. The duration-axis keys from
/// [`crate::stats::SortKey`] (total_ns, p50_ns, gbps) are
/// deliberately omitted so a `--by size --sort gbps` request errors
/// at parse time — the column doesn't exist in this mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortKey {
    Total,
    Count,
    Avg,
    Min,
    Max,
    P50,
    P95,
    P99,
    Name,
    Device,
    Stream,
    Context,
}

impl SortKeyDef for SortKey {
    fn specs() -> &'static [SortKeySpec<Self>] {
        &[
            SortKeySpec {
                variant: SortKey::Total,
                canonical: "total",
                aliases: &["total_bytes", "bytes", "bytes_total"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SortKey::Count,
                canonical: "count",
                aliases: &[],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SortKey::Avg,
                canonical: "avg",
                aliases: &["avg_bytes"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SortKey::Min,
                canonical: "min",
                aliases: &["min_bytes"],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: SortKey::Max,
                canonical: "max",
                aliases: &["max_bytes"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SortKey::P50,
                canonical: "p50",
                aliases: &["p50_bytes"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SortKey::P95,
                canonical: "p95",
                aliases: &["p95_bytes"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SortKey::P99,
                canonical: "p99",
                aliases: &["p99_bytes"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SortKey::Name,
                canonical: "name",
                aliases: &[],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: SortKey::Device,
                canonical: "device",
                aliases: &["device_id", "dev"],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: SortKey::Stream,
                canonical: "stream",
                aliases: &["stream_id"],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: SortKey::Context,
                canonical: "context",
                aliases: &["context_id", "ctx"],
                default_dir: Direction::Asc,
            },
        ]
    }
}

impl SortKey {
    fn column(self) -> &'static str {
        match self {
            Self::Total => "total_bytes",
            Self::Count => "count",
            Self::Avg => "avg_bytes",
            Self::Min => "min_bytes",
            Self::Max => "max_bytes",
            Self::P50 => "p50_bytes",
            Self::P95 => "p95_bytes",
            Self::P99 => "p99_bytes",
            Self::Name => "name",
            Self::Device => "device_id",
            Self::Stream => "stream_id",
            Self::Context => "context_id",
        }
    }
}

#[derive(Debug, Clone)]
pub struct StatsBySizeRequest {
    /// Which kinds to aggregate. `KindFilter::All` resolves to
    /// memcpy + memset; explicit kinds outside [`ALLOWED_KINDS`]
    /// error in [`run`].
    pub kinds: KindFilter,
    pub group_by: crate::stats::GroupBy,
    pub time_window: Option<TimeWindow>,
    pub device: Option<i32>,
    pub stream: Option<i64>,
    pub sort: Option<SortSpec>,
    pub limit: usize,
}

impl Default for StatsBySizeRequest {
    fn default() -> Self {
        Self {
            kinds: KindFilter::All,
            group_by: crate::stats::GroupBy::default(),
            time_window: None,
            device: None,
            stream: None,
            sort: None,
            limit: 50,
        }
    }
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct StatsBySizeResponse {
    pub count: usize,
    pub total_matched: i64,
    /// Grand total *bytes* across the whole filtered scope. This is
    /// the denominator behind every row's `percentage`. Distinct from
    /// `StatsResponse.total_duration_ns` so JSON consumers can tell
    /// the two surfaces apart at the field-name level.
    pub total_bytes: i64,
    pub total_events: i64,
    pub time_window_ns: Option<(i64, i64)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_scope: Option<String>,
    pub rows: Vec<StatBySizeRow>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct StatBySizeRow {
    /// Same composite-key format as `StatsResponse.rows[].key` —
    /// `(kind, name?, dev?, stream?, ctx?)` pipe-joined. Lets agents
    /// `INDEX(.rows; .key)` cross-trace within a single mode without
    /// colliding on key shape across stats / stats-by-size.
    pub key: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(rename = "type")]
    pub kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub short_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_id: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_id: Option<i64>,
    pub count: i64,
    pub total_bytes: i64,
    pub avg_bytes: i64,
    pub min_bytes: i64,
    pub max_bytes: i64,
    pub p50_bytes: i64,
    pub p95_bytes: i64,
    pub p99_bytes: i64,
    pub percentage: f64,
}

pub fn run<P: AsRef<Path>>(
    path: P,
    req: StatsBySizeRequest,
) -> NsysQueryResult<StatsBySizeResponse> {
    let (trace, abs_window) = crate::open_scoped(path.as_ref(), req.limit, req.time_window)?;

    if let KindFilter::Only(v) = &req.kinds {
        for k in v {
            if !ALLOWED_KINDS.contains(k) {
                return Err(NsysQueryError::StatsBySizeKindNotAllowed { kind: k.as_str() });
            }
        }
    }
    // Defence-in-depth: hand-built `StatsBySizeRequest`s with axes
    // this verb doesn't implement yet would otherwise be silently
    // dropped by `build_group_keys` (which only consumes name +
    // device/context/stream). CLI callers already error in
    // `commands.rs`; library callers via the SDK go through here.
    if req.group_by.graph
        || req.group_by.graph_node
        || req.group_by.grid_block
        || req.group_by.nvtx_parent
        || req.group_by.nvtx_path
    {
        let unsupported = [
            ("graph", req.group_by.graph),
            ("graph_node", req.group_by.graph_node),
            ("grid_block", req.group_by.grid_block),
            ("nvtx-parent", req.group_by.nvtx_parent),
            ("nvtx-path", req.group_by.nvtx_path),
        ]
        .iter()
        .filter_map(|(name, on)| if *on { Some(*name) } else { None })
        .collect::<Vec<_>>()
        .join(", ");
        return Err(NsysQueryError::stats_by_size_group_by_unsupported(
            unsupported,
        ));
    }

    let requested = req.kinds.resolve(&ALLOWED_KINDS);
    let kinds: Vec<EventKind> = ALLOWED_KINDS
        .into_iter()
        .filter(|k| requested.contains(k))
        .filter(|k| trace.table_exists(k.table()))
        .collect();

    if kinds.is_empty() {
        return Ok(StatsBySizeResponse {
            count: 0,
            total_matched: 0,
            total_bytes: 0,
            total_events: 0,
            time_window_ns: abs_window,
            nvtx_scope: None,
            rows: Vec::new(),
        });
    }

    let mut subqueries: Vec<String> = Vec::with_capacity(kinds.len());
    let mut per_kind_params: Vec<Value> = Vec::new();
    for kind in &kinds {
        let fragment = per_kind_size_subquery(*kind, abs_window)?;
        subqueries.push(fragment.sql);
        per_kind_params.extend(fragment.params);
    }
    let union = subqueries.join(" UNION ALL ");

    // group-by axes — reuse the existing dispatch by hand-rolling the
    // small subset of fields we project. Identity axes (name + dim)
    // match stats; we deliberately skip the NVTX / graph axes because
    // bytes don't have a natural mapping there yet.
    let group_keys: Vec<&str> = build_group_keys(&req.group_by);
    let name_select = match req.group_by.name {
        crate::stats::NameAxis::ShortName | crate::stats::NameAxis::Demangled => {
            "short_name AS name"
        }
        crate::stats::NameAxis::Mangled => "short_name AS name",
        crate::stats::NameAxis::None => "CAST(NULL AS VARCHAR) AS name",
    };
    let short_name_select = match req.group_by.name {
        crate::stats::NameAxis::None => "CAST(NULL AS VARCHAR) AS short_name",
        _ => "short_name",
    };
    let device_select = if req.group_by.device {
        "device_id"
    } else {
        "CAST(NULL AS INTEGER) AS device_id"
    };
    let context_select = if req.group_by.context {
        "context_id"
    } else {
        "CAST(NULL AS BIGINT) AS context_id"
    };
    let stream_select = if req.group_by.stream {
        "stream_id"
    } else {
        "CAST(NULL AS BIGINT) AS stream_id"
    };
    let group_keys_sql = group_keys.join(", ");

    let sort_spec = req
        .sort
        .clone()
        .unwrap_or_else(|| SortSpec::single("total"));
    let order_by = sort_sql(&sort_spec)?;

    let mut location_where = String::new();
    let mut location_params: Vec<Value> = Vec::new();
    crate::kind_policy::LocationFilter {
        device: req.device,
        stream: req.stream,
    }
    .append_where(&mut location_where, &mut location_params);

    let sql = format!(
        r#"
        WITH events AS ({union}),
        grouped AS (
            SELECT
                {name_select},
                {short_name_select},
                kind,
                {device_select},
                {context_select},
                {stream_select},
                COUNT(*)                                       AS count,
                CAST(SUM(bytes) AS BIGINT)                     AS total_bytes,
                CAST(AVG(bytes) AS BIGINT)                     AS avg_bytes,
                MIN(bytes)                                     AS min_bytes,
                MAX(bytes)                                     AS max_bytes,
                CAST(quantile_disc(bytes, 0.50) AS BIGINT)     AS p50_bytes,
                CAST(quantile_disc(bytes, 0.95) AS BIGINT)     AS p95_bytes,
                CAST(quantile_disc(bytes, 0.99) AS BIGINT)     AS p99_bytes
            FROM events
            WHERE bytes IS NOT NULL AND bytes >= 0 {location_where}
            GROUP BY {group_keys_sql}
        )
        SELECT
            name, short_name, kind,
            device_id, context_id, stream_id,
            count,
            total_bytes, avg_bytes, min_bytes, max_bytes,
            p50_bytes, p95_bytes, p99_bytes,
            CAST(SUM(total_bytes) OVER () AS BIGINT) AS scope_total_bytes,
            CAST(SUM(count)       OVER () AS BIGINT) AS scope_total_count,
            CAST(COUNT(*)         OVER () AS BIGINT) AS scope_total_groups
        FROM grouped
        ORDER BY {order_by}
        LIMIT ?
        "#
    );

    let mut params: Vec<Value> = Vec::new();
    params.extend(per_kind_params);
    params.extend(location_params);
    params.push(Value::BigInt(req.limit as i64));

    let (mut out, scope) = hydrate_stats_by_size_rows(&trace, &sql, &params)?;

    if scope.total_bytes > 0 {
        for r in &mut out {
            r.percentage = (r.total_bytes as f64 / scope.total_bytes as f64) * 100.0;
        }
    }

    Ok(StatsBySizeResponse {
        count: out.len(),
        total_matched: scope.total_groups,
        total_bytes: scope.total_bytes,
        total_events: scope.total_count,
        time_window_ns: abs_window,
        nvtx_scope: None,
        rows: out,
    })
}

struct StatsBySizeScope {
    total_bytes: i64,
    total_count: i64,
    total_groups: i64,
}

fn hydrate_stats_by_size_rows(
    trace: &Trace,
    sql: &str,
    params: &[Value],
) -> NsysQueryResult<(Vec<StatBySizeRow>, StatsBySizeScope)> {
    let rows = crate::query_sql::exec::query_rows(
        trace.conn(),
        sql,
        params,
        crate::query_sql::exec::STATS_BY_SIZE_AGGREGATE,
        stats_by_size_sql_row,
    )?;

    let scope = rows.first().map_or(
        StatsBySizeScope {
            total_bytes: 0,
            total_count: 0,
            total_groups: 0,
        },
        StatsBySizeSqlRow::scope,
    );
    let out = rows.into_iter().map(stat_by_size_row).collect();
    Ok((out, scope))
}

struct StatsBySizeSqlRow {
    name: Option<String>,
    short_name: Option<String>,
    kind: String,
    device_id: Option<i32>,
    context_id: Option<i64>,
    stream_id: Option<i64>,
    count: i64,
    total_bytes: i64,
    avg_bytes: i64,
    min_bytes: i64,
    max_bytes: i64,
    p50_bytes: i64,
    p95_bytes: i64,
    p99_bytes: i64,
    scope_total_bytes: i64,
    scope_total_count: i64,
    scope_total_groups: i64,
}

impl StatsBySizeSqlRow {
    fn scope(&self) -> StatsBySizeScope {
        StatsBySizeScope {
            total_bytes: self.scope_total_bytes,
            total_count: self.scope_total_count,
            total_groups: self.scope_total_groups,
        }
    }
}

fn stats_by_size_sql_row(row: &duckdb::Row<'_>) -> Result<StatsBySizeSqlRow, duckdb::Error> {
    Ok(StatsBySizeSqlRow {
        name: row.get("name")?,
        short_name: row.get("short_name")?,
        kind: row.get("kind")?,
        device_id: row.get("device_id")?,
        context_id: row.get("context_id")?,
        stream_id: row.get("stream_id")?,
        count: row.get("count")?,
        total_bytes: row.get("total_bytes")?,
        avg_bytes: row.get("avg_bytes")?,
        min_bytes: row.get("min_bytes")?,
        max_bytes: row.get("max_bytes")?,
        p50_bytes: row.get("p50_bytes")?,
        p95_bytes: row.get("p95_bytes")?,
        p99_bytes: row.get("p99_bytes")?,
        scope_total_bytes: row.get("scope_total_bytes")?,
        scope_total_count: row.get("scope_total_count")?,
        scope_total_groups: row.get("scope_total_groups")?,
    })
}

fn stat_by_size_row(row: StatsBySizeSqlRow) -> StatBySizeRow {
    let kind_static: &'static str = EventKind::parse(&row.kind)
        .map(EventKind::as_str)
        .unwrap_or("unknown");
    let short_name = if kind_static == "kernel" {
        row.short_name
    } else {
        None
    };
    let mut key_parts = vec![kind_static.to_string()];
    if let Some(n) = row.name.as_deref() {
        key_parts.push(n.to_string());
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
    StatBySizeRow {
        key: key_parts.join("|"),
        name: row.name,
        kind: kind_static,
        short_name,
        device_id: row.device_id,
        context_id: row.context_id,
        stream_id: row.stream_id,
        count: row.count,
        total_bytes: row.total_bytes,
        avg_bytes: row.avg_bytes,
        min_bytes: row.min_bytes,
        max_bytes: row.max_bytes,
        p50_bytes: row.p50_bytes,
        p95_bytes: row.p95_bytes,
        p99_bytes: row.p99_bytes,
        percentage: 0.0,
    }
}

fn build_group_keys(g: &crate::stats::GroupBy) -> Vec<&'static str> {
    let mut keys: Vec<&'static str> = vec!["kind"];
    match g.name {
        crate::stats::NameAxis::ShortName
        | crate::stats::NameAxis::Demangled
        | crate::stats::NameAxis::Mangled => keys.push("short_name"),
        crate::stats::NameAxis::None => {}
    }
    if g.device {
        keys.push("device_id");
    }
    if g.context {
        keys.push("context_id");
    }
    if g.stream {
        keys.push("stream_id");
    }
    keys
}

fn sort_sql(spec: &SortSpec) -> NsysQueryResult<String> {
    order_by::<SortKey>(
        spec,
        SortKey::column,
        NsysQueryError::stats_by_size_sort_invalid,
        "total_bytes",
    )
}

fn per_kind_size_subquery(
    kind: EventKind,
    abs_window: Option<(i64, i64)>,
) -> NsysQueryResult<SqlFragment> {
    let sem = EventSemantics::new(kind);
    let bytes_expr = sem
        .size_bytes_expr()
        .ok_or_else(|| NsysQueryError::internal_unsupported_kind("stats-by-size", kind.as_str()))?;

    // Window semantics: include the event with its FULL bytes if its
    // interval overlaps the window; bytes are not proportionally clipped.
    let filter = event_scan_filter(
        sem,
        EventScanFilterOptions {
            abs_window,
            device: None,
            stream: None,
            nvtx_scope: crate::nvtx_attribution::NvtxScope::None,
            nvtx_policy: NvtxFilterPolicy::EmptyWhenUnsupported,
        },
        BYTES_PRESENT_PREDICATES,
    )?;
    let where_clause = filter.where_clause();
    let params = filter.into_params();

    let sql = format!(
        "SELECT {display_expr} AS display_name, \
                {short_expr}   AS short_name, \
                '{label}'      AS kind, \
                {bytes_expr}   AS bytes, \
                {dev}          AS device_id, \
                {ctx}          AS context_id, \
                {stm}          AS stream_id \
         FROM nsight.{table} t {join_clause} {where_clause}",
        display_expr = sem.display_name_expr(),
        short_expr = sem.short_name_expr(),
        label = sem.label(),
        dev = sem.device_expr(),
        ctx = sem.context_expr(),
        stm = sem.stream_expr(),
        table = sem.table(),
        join_clause = sem.name_joins(),
    );
    Ok(SqlFragment::new(sql, params))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use duckdb::Connection;
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

    fn minimal_trace() -> Result<(TempDir, Trace)> {
        let (dir, pqtdir) = parquet_fixture(vec![(
            "CUPTI_ACTIVITY_KIND_KERNEL",
            r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start BIGINT, "end" BIGINT)"#,
            Vec::new(),
        )])?;
        let trace = Trace::open(&pqtdir)?;
        Ok((dir, trace))
    }

    fn stats_by_size_hydration_sql(count_expr: &str) -> String {
        format!(
            "SELECT \
             CAST(NULL AS VARCHAR) AS name, \
             CAST(NULL AS VARCHAR) AS short_name, \
             'memcpy' AS kind, \
             CAST(NULL AS INTEGER) AS device_id, \
             CAST(NULL AS BIGINT) AS context_id, \
             CAST(NULL AS BIGINT) AS stream_id, \
             {count_expr} AS count, \
             1::BIGINT AS total_bytes, \
             1::BIGINT AS avg_bytes, \
             1::BIGINT AS min_bytes, \
             1::BIGINT AS max_bytes, \
             1::BIGINT AS p50_bytes, \
             1::BIGINT AS p95_bytes, \
             1::BIGINT AS p99_bytes, \
             1::BIGINT AS scope_total_bytes, \
             1::BIGINT AS scope_total_count, \
             1::BIGINT AS scope_total_groups"
        )
    }

    #[test]
    fn hydrate_stats_by_size_rows_prepare_error_is_typed() -> Result<()> {
        let (_dir, trace) = minimal_trace()?;

        let err = match hydrate_stats_by_size_rows(&trace, "SELECT * FROM", &[]) {
            Ok((rows, _)) => anyhow::bail!(
                "malformed stats-by-size SQL should not hydrate successfully: {} rows",
                rows.len()
            ),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-prepare");
        assert!(matches!(
            err,
            crate::NsysQueryError::Sql {
                phase: crate::SqlPhase::Prepare,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn hydrate_stats_by_size_rows_query_error_is_typed() -> Result<()> {
        let (_dir, trace) = minimal_trace()?;

        let err = match hydrate_stats_by_size_rows(&trace, "SELECT ? AS name", &[]) {
            Ok((rows, _)) => anyhow::bail!(
                "unbound stats-by-size SQL parameter should not hydrate successfully: {} rows",
                rows.len()
            ),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-query");
        assert!(matches!(
            err,
            crate::NsysQueryError::Sql {
                phase: crate::SqlPhase::Query,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn hydrate_stats_by_size_rows_read_error_is_typed() -> Result<()> {
        let (_dir, trace) = minimal_trace()?;
        let sql = stats_by_size_hydration_sql("'not-a-count'");

        let err = match hydrate_stats_by_size_rows(&trace, &sql, &[]) {
            Ok((rows, _)) => anyhow::bail!(
                "malformed stats-by-size row should not hydrate successfully: {} rows",
                rows.len()
            ),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-read");
        assert!(matches!(
            err,
            crate::NsysQueryError::Sql {
                phase: crate::SqlPhase::Read,
                ..
            }
        ));
        Ok(())
    }
}
