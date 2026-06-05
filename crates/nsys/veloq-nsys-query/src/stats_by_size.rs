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

use crate::{EventKind, KindFilter};
use anyhow::{Context, Result};
use duckdb::types::Value;
use serde::Serialize;
use std::path::Path;
use veloq_core::{Direction, SortKeyDef, SortKeySpec, SortSpec, time::TimeWindow};

/// Kinds that carry a `bytes` column and are therefore aggregatable
/// under `--by size`. Mirrors the [`crate::stats::ALLOWED_KINDS`]
/// allow-list convention.
pub const ALLOWED_KINDS: [EventKind; 2] = [EventKind::Memcpy, EventKind::Memset];

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

pub fn run<P: AsRef<Path>>(path: P, req: StatsBySizeRequest) -> Result<StatsBySizeResponse> {
    let (trace, abs_window) = crate::open_scoped(path.as_ref(), req.limit, req.time_window)?;

    if let KindFilter::Only(v) = &req.kinds {
        for k in v {
            if !ALLOWED_KINDS.contains(k) {
                anyhow::bail!(
                    "--by size only aggregates byte-carrying kinds \
                     (memcpy/memset); got `--type {}`. Drop the kind or \
                     unset --by size.",
                    k.as_str()
                );
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
        anyhow::bail!(
            "stats_by_size does not yet support --group-by \
             graph/graph_node/grid_block/nvtx-parent/nvtx-path. Supported axes \
             today are the name axis (short/demangled/mangled/no-name) \
             and device/context/stream."
        );
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
        let (sql, params) = per_kind_size_subquery(*kind, abs_window)?;
        subqueries.push(sql);
        per_kind_params.extend(params);
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

    let conn = trace.conn();
    let mut stmt = conn.prepare(&sql).context("prepare stats-by-size SQL")?;
    let params_ref = crate::bind(&params);
    let mut rows = stmt.query(params_ref.as_slice())?;

    let mut out: Vec<StatBySizeRow> = Vec::new();
    let mut scope_total_bytes: i64 = 0;
    let mut scope_total_count: i64 = 0;
    let mut scope_total_groups: i64 = 0;
    let mut scope_read = false;

    while let Some(row) = rows.next()? {
        let name: Option<String> = row.get("name")?;
        let short_name_raw: Option<String> = row.get("short_name")?;
        let kind: String = row.get("kind")?;
        let device_id: Option<i32> = row.get("device_id")?;
        let context_id: Option<i64> = row.get("context_id")?;
        let stream_id: Option<i64> = row.get("stream_id")?;
        let count: i64 = row.get("count")?;
        let total_bytes: i64 = row.get("total_bytes")?;
        let avg_bytes: i64 = row.get("avg_bytes")?;
        let min_bytes: i64 = row.get("min_bytes")?;
        let max_bytes: i64 = row.get("max_bytes")?;
        let p50_bytes: i64 = row.get("p50_bytes")?;
        let p95_bytes: i64 = row.get("p95_bytes")?;
        let p99_bytes: i64 = row.get("p99_bytes")?;
        if !scope_read {
            scope_total_bytes = row.get("scope_total_bytes")?;
            scope_total_count = row.get("scope_total_count")?;
            scope_total_groups = row.get("scope_total_groups")?;
            scope_read = true;
        }
        let kind_static: &'static str = EventKind::parse(&kind)
            .map(EventKind::as_str)
            .unwrap_or("unknown");
        let short_name = if kind_static == "kernel" {
            short_name_raw
        } else {
            None
        };
        let mut key_parts = vec![kind_static.to_string()];
        if let Some(n) = name.as_deref() {
            key_parts.push(n.to_string());
        }
        if let Some(d) = device_id {
            key_parts.push(format!("dev:{d}"));
        }
        if let Some(s) = stream_id {
            key_parts.push(format!("stream:{s}"));
        }
        if let Some(c) = context_id {
            key_parts.push(format!("ctx:{c}"));
        }
        let key = key_parts.join("|");
        out.push(StatBySizeRow {
            key,
            name,
            kind: kind_static,
            short_name,
            device_id,
            context_id,
            stream_id,
            count,
            total_bytes,
            avg_bytes,
            min_bytes,
            max_bytes,
            p50_bytes,
            p95_bytes,
            p99_bytes,
            percentage: 0.0,
        });
    }

    if scope_total_bytes > 0 {
        for r in &mut out {
            r.percentage = (r.total_bytes as f64 / scope_total_bytes as f64) * 100.0;
        }
    }

    Ok(StatsBySizeResponse {
        count: out.len(),
        total_matched: scope_total_groups,
        total_bytes: scope_total_bytes,
        total_events: scope_total_count,
        time_window_ns: abs_window,
        nvtx_scope: None,
        rows: out,
    })
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

fn sort_sql(spec: &SortSpec) -> Result<String> {
    let mut resolved: Vec<(&'static str, Direction)> = Vec::new();
    for f in spec.fields() {
        let (k, d) = SortKey::from_field(f)?;
        resolved.push((k.column(), d));
    }
    Ok(veloq_core::sort::build_order_by(&resolved, "total_bytes"))
}

fn per_kind_size_subquery(
    kind: EventKind,
    abs_window: Option<(i64, i64)>,
) -> Result<(String, Vec<Value>)> {
    let dev = crate::kind_sql::GPU_DEVICE_ID_EXPR;
    let ctx = crate::kind_sql::GPU_CONTEXT_ID_EXPR;
    let stm = crate::kind_sql::GPU_STREAM_ID_EXPR;
    let display_expr = crate::kind_sql::display_name_expr(kind);
    let short_expr = crate::kind_sql::short_name_expr(kind);
    let join_clause = crate::kind_sql::name_joins(kind);
    let label = kind.as_str();
    let table = kind.table();

    let mut params: Vec<Value> = Vec::new();
    let (bytes_expr, mut where_parts): (String, Vec<String>) = match abs_window {
        Some((start, end)) => {
            // Window semantics: an event is included with its FULL
            // bytes if its interval has any overlap with the window;
            // there is no proportional clip on bytes. Memops are not
            // uniform-rate DMAs in general, so a duration-weighted
            // scale would be a different approximation, not a more
            // precise one.
            params.push(Value::BigInt(end));
            params.push(Value::BigInt(start));
            (
                "CAST(t.bytes AS BIGINT)".to_string(),
                vec![r#"t.start < ? AND t."end" > ?"#.to_string()],
            )
        }
        None => ("CAST(t.bytes AS BIGINT)".to_string(), Vec::new()),
    };
    where_parts.push("t.bytes IS NOT NULL".to_string());
    let where_clause = format!("WHERE {}", where_parts.join(" AND "));

    let sql = format!(
        "SELECT {display_expr} AS display_name, \
                {short_expr}   AS short_name, \
                '{label}'      AS kind, \
                {bytes_expr}   AS bytes, \
                {dev}          AS device_id, \
                {ctx}          AS context_id, \
                {stm}          AS stream_id \
         FROM nsight.{table} t {join_clause} {where_clause}"
    );
    Ok((sql, params))
}
