//! `--type nic` source.
//!
//! Reads `NET_NIC_METRIC` (per-NIC interval samples) plus
//! `TARGET_INFO_NETWORK_METRICS` (counter dictionary),
//! `NIC_ID_MAP`, and `TARGET_INFO_NIC_INFO` (NIC identity). Summary
//! mode reports per `(NIC, port, counter)` min/max/mean/p50/p95/p99;
//! `--bucket Nms` switches to a long-form time series. NSys exports
//! NIC values as rates (`bytes/ms`, `packets/ms`, `ticks/ms`) or
//! already-averaged sizes (`bytes`), so bucket rollups use `mean`.

use crate::{NsysQueryError, NsysQueryResult};
use duckdb::types::Value;
use serde::Serialize;
use veloq_core::{Direction, SortKeyDef, SortKeySpec, SortSpec};
use veloq_nsys_data::Trace;
use veloq_query::duckdb::list as duckdb_list;
use veloq_query::sql::{name, total_matched_bigint_expr, window};

use super::{Coverage, MetricsCommon, NicMetricsBody, NicMetricsRequest, query_rows};

const NIC_COUNTERS_SQL: &str = "nic counters";
const NIC_BUCKETS_SQL: &str = "nic buckets";

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct NicCounterSummary {
    /// Cross-trace key. `nic_counter|nic:<nic_id>|port:<port_id>|metric:<metrics_idx>`.
    pub key: String,
    pub nic_id: i64,
    pub guid: i64,
    pub nic_name: String,
    pub global_id: i64,
    pub port_id: i64,
    pub metrics_list_id: i64,
    pub metrics_idx: i64,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    /// Aggregator used for `--bucket` rollups. NIC samples are rates
    /// or already-averaged sizes in current NSys exports, so this is
    /// `"mean"`.
    pub agg: &'static str,
    pub samples: i64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct NicBucketSample {
    /// Cross-trace key. `nic_bucket|<t_start_ns>|nic:<nic_id>|port:<port_id>|metric:<metrics_idx>`.
    pub key: String,
    pub t_start_ns: i64,
    pub t_end_ns: i64,
    pub nic_id: i64,
    pub guid: i64,
    pub nic_name: String,
    pub global_id: i64,
    pub port_id: i64,
    pub metrics_list_id: i64,
    pub metrics_idx: i64,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    pub agg: &'static str,
    pub value: f64,
    pub samples: i64,
}

pub(super) fn run_nic(
    trace: &Trace,
    req: &NicMetricsRequest,
    abs_window: Option<(i64, i64)>,
    trace_origin_ns: i64,
    trace_span_ns: (i64, i64),
) -> NsysQueryResult<NicMetricsBody> {
    require_nic_tables(trace)?;

    let counters = query_nic_counters(trace, req, abs_window)?;

    let (metrics_min, metrics_max, samples_total) = counters
        .iter()
        .fold((i64::MAX, i64::MIN, 0i64), |(lo, hi, n), c| {
            (lo.min(c.span_lo), hi.max(c.span_hi), n + c.summary.samples)
        });
    let max_gap_ns = counters.iter().filter_map(|c| c.max_gap_ns).max();
    let metrics_span_ns = if samples_total > 0 {
        Some((metrics_min, metrics_max))
    } else {
        None
    };

    let bucket_ns = req.common.bucket_ns;
    let (bucket_rows, total_buckets) = match bucket_ns {
        None => (Vec::new(), 0i64),
        Some(bucket) => query_nic_buckets(trace, req, abs_window, bucket, trace_origin_ns)?,
    };

    let mut summaries: Vec<NicCounterSummary> = counters.into_iter().map(|c| c.summary).collect();
    let summaries_pre_limit_len = summaries.len();
    if bucket_ns.is_none() {
        let sort_spec = req
            .common
            .sort
            .clone()
            .unwrap_or_else(|| SortSpec::single("name"));
        sort_counters(&mut summaries, &sort_spec)?;
        summaries.truncate(req.common.limit);
    } else {
        summaries.clear();
    }

    let mut buckets = bucket_rows;
    let total_buckets_before_limit = total_buckets;
    if bucket_ns.is_some() {
        buckets.truncate(req.common.limit);
    }

    let coverage = Coverage::compute(metrics_span_ns, trace_span_ns, samples_total, max_gap_ns);

    let (count, total_matched) = match bucket_ns {
        None => (summaries.len(), summaries_pre_limit_len as i64),
        Some(_) => (buckets.len(), total_buckets_before_limit),
    };

    Ok(NicMetricsBody {
        count,
        total_matched,
        rows: summaries,
        auxiliary: super::NicMetricsAuxiliary {
            common: MetricsCommon {
                trace_origin_ns,
                trace_span_ns,
                metrics_span_ns,
                coverage,
                time_window_ns: abs_window,
                bucket_ns,
            },
            counter_glob: req.counter_glob.clone(),
            buckets,
        },
    })
}

fn require_nic_tables(trace: &Trace) -> NsysQueryResult<()> {
    if !trace.table_exists("NET_NIC_METRIC") {
        return Err(NsysQueryError::MetricsNicTableMissing);
    }
    if !trace.table_exists("TARGET_INFO_NETWORK_METRICS") {
        return Err(NsysQueryError::MetricsNicDictionaryMissing);
    }
    if !trace.table_exists("NIC_ID_MAP") {
        return Err(NsysQueryError::MetricsNicIdMapMissing);
    }
    if !trace.table_exists("TARGET_INFO_NIC_INFO") {
        return Err(NsysQueryError::MetricsNicInfoMissing);
    }
    Ok(())
}

struct CounterWithSpan {
    summary: NicCounterSummary,
    span_lo: i64,
    span_hi: i64,
    max_gap_ns: Option<i64>,
}

fn query_nic_counters(
    trace: &Trace,
    req: &NicMetricsRequest,
    abs_window: Option<(i64, i64)>,
) -> NsysQueryResult<Vec<CounterWithSpan>> {
    let mut params: Vec<Value> = Vec::new();
    let dict_pred = if let Some(counter_glob) = req.counter_glob.as_deref() {
        let fragment = name::glob_like("name", counter_glob);
        params.extend(fragment.params);
        format!("WHERE {}", fragment.sql)
    } else {
        String::new()
    };

    let sample_scope = window::positive_interval_sample_scope("m.start", r#"m."end""#, abs_window);
    params.extend(sample_scope.params);

    let sql = format!(
        r#"
        WITH dict AS (
            SELECT metricsListId, metricsIdx, name, description, unit
            FROM nsight.TARGET_INFO_NETWORK_METRICS
            {dict_pred}
        ),
        ports AS (
            SELECT DISTINCT globalId, portId
            FROM nsight.NET_NIC_METRIC
        ),
        targets AS (
            SELECT
                map.nicId AS nic_id,
                nic.GUID AS guid,
                nic.name AS nic_name,
                p.globalId AS global_id,
                p.portId AS port_id,
                d.metricsListId AS metrics_list_id,
                d.metricsIdx AS metrics_idx,
                d.name AS metric_name,
                d.description AS description,
                d.unit AS unit
            FROM ports p
            JOIN nsight.NIC_ID_MAP map ON map.globalId = p.globalId
            JOIN nsight.TARGET_INFO_NIC_INFO nic ON nic.nicId = map.nicId
            CROSS JOIN dict d
        ),
        raw_samples AS (
            SELECT
                m.globalId AS global_id,
                m.portId AS port_id,
                m.metricsListId AS metrics_list_id,
                m.metricsIdx AS metrics_idx,
                {sample_start_expr} AS start_ns,
                {sample_end_expr} AS end_ns,
                CAST(m.value AS DOUBLE) AS value
            FROM nsight.NET_NIC_METRIC m
            JOIN dict d
              ON d.metricsListId = m.metricsListId
             AND d.metricsIdx = m.metricsIdx
            {sample_pred}
        ),
        samples AS (
            SELECT
                *,
                end_ns - start_ns AS clipped_ns
            FROM raw_samples
        ),
        sampled AS (
            SELECT
                *,
                LAG(end_ns) OVER (
                    PARTITION BY global_id, port_id, metrics_list_id, metrics_idx
                    ORDER BY start_ns, end_ns
                ) AS prev_end_ns
            FROM samples
        ),
        samples_with_gap AS (
            SELECT
                *,
                CASE
                    WHEN prev_end_ns IS NULL THEN NULL
                    ELSE GREATEST(start_ns - prev_end_ns, 0)
                END AS gap_ns
            FROM sampled
        )
        SELECT
            t.nic_id,
            t.guid,
            t.nic_name,
            t.global_id,
            t.port_id,
            t.metrics_list_id,
            t.metrics_idx,
            t.metric_name,
            t.description,
            t.unit,
            CAST(COUNT(s.start_ns) AS BIGINT) AS samples,
            COALESCE(MIN(s.value), 0.0) AS min_v,
            COALESCE(MAX(s.value), 0.0) AS max_v,
            COALESCE(
                SUM(s.value * CAST(s.clipped_ns AS DOUBLE))
                    / NULLIF(CAST(SUM(s.clipped_ns) AS DOUBLE), 0.0),
                0.0
            ) AS mean_v,
            COALESCE(APPROX_QUANTILE(s.value, 0.50), 0.0) AS p50_v,
            COALESCE(APPROX_QUANTILE(s.value, 0.95), 0.0) AS p95_v,
            COALESCE(APPROX_QUANTILE(s.value, 0.99), 0.0) AS p99_v,
            CAST(COALESCE(MIN(s.start_ns), 0) AS BIGINT) AS span_lo,
            CAST(COALESCE(MAX(s.end_ns), 0) AS BIGINT) AS span_hi,
            CAST(MAX(s.gap_ns) AS BIGINT) AS max_gap_ns
        FROM targets t
        LEFT JOIN samples_with_gap s
          ON s.global_id = t.global_id
         AND s.port_id = t.port_id
         AND s.metrics_list_id = t.metrics_list_id
         AND s.metrics_idx = t.metrics_idx
        GROUP BY
            t.nic_id,
            t.guid,
            t.nic_name,
            t.global_id,
            t.port_id,
            t.metrics_list_id,
            t.metrics_idx,
            t.metric_name,
            t.description,
            t.unit
        "#,
        sample_start_expr = sample_scope.start_expr,
        sample_end_expr = sample_scope.end_expr,
        sample_pred = sample_scope.where_clause,
    );

    let out = query_rows(trace, &sql, &params, NIC_COUNTERS_SQL, nic_counter_row)?;
    if out.is_empty()
        && let Some(g) = &req.counter_glob
    {
        return Err(NsysQueryError::metrics_nic_counter_no_match(g));
    }
    Ok(out)
}

fn nic_counter_row(row: &duckdb::Row<'_>) -> Result<CounterWithSpan, duckdb::Error> {
    let samples: i64 = row.get("samples")?;
    let nic_id: i64 = row.get("nic_id")?;
    let port_id: i64 = row.get("port_id")?;
    let metrics_idx: i64 = row.get("metrics_idx")?;
    Ok(CounterWithSpan {
        summary: NicCounterSummary {
            key: format!("nic_counter|nic:{nic_id}|port:{port_id}|metric:{metrics_idx}"),
            nic_id,
            guid: row.get("guid")?,
            nic_name: row.get("nic_name")?,
            global_id: row.get("global_id")?,
            port_id,
            metrics_list_id: row.get("metrics_list_id")?,
            metrics_idx,
            name: row.get("metric_name")?,
            description: row.get("description")?,
            unit: row.get("unit")?,
            agg: "mean",
            samples,
            min: row.get("min_v")?,
            max: row.get("max_v")?,
            mean: row.get("mean_v")?,
            p50: row.get("p50_v")?,
            p95: row.get("p95_v")?,
            p99: row.get("p99_v")?,
        },
        span_lo: if samples > 0 {
            row.get("span_lo")?
        } else {
            i64::MAX
        },
        span_hi: if samples > 0 {
            row.get("span_hi")?
        } else {
            i64::MIN
        },
        max_gap_ns: row.get("max_gap_ns")?,
    })
}

fn query_nic_buckets(
    trace: &Trace,
    req: &NicMetricsRequest,
    abs_window: Option<(i64, i64)>,
    bucket_ns: i64,
    primary_origin_ns: i64,
) -> NsysQueryResult<(Vec<NicBucketSample>, i64)> {
    let anchor = abs_window.map(|(s, _)| s).unwrap_or(primary_origin_ns);

    let mut params: Vec<Value> = Vec::new();
    let dict_pred = if let Some(counter_glob) = req.counter_glob.as_deref() {
        let fragment = name::glob_like("name", counter_glob);
        params.extend(fragment.params);
        format!("WHERE {}", fragment.sql)
    } else {
        String::new()
    };
    let sample_scope = window::positive_interval_sample_scope("m.start", r#"m."end""#, abs_window);
    params.extend(sample_scope.params);
    let bucket_start_expr = format!("bucket_idx * {bucket_ns} + {anchor}");
    let bucket_end_expr = format!("bucket_idx * {bucket_ns} + {anchor} + {bucket_ns}");
    let clipped_ns_expr = window::bucket_clipped_duration_expr(
        "start_ns",
        "end_ns",
        &bucket_start_expr,
        &bucket_end_expr,
    );

    let sql = format!(
        r#"
        WITH dict AS (
            SELECT metricsListId, metricsIdx, name, unit
            FROM nsight.TARGET_INFO_NETWORK_METRICS
            {dict_pred}
        ),
        samples AS (
            SELECT
                map.nicId AS nic_id,
                nic.GUID AS guid,
                nic.name AS nic_name,
                m.globalId AS global_id,
                m.portId AS port_id,
                m.metricsListId AS metrics_list_id,
                m.metricsIdx AS metrics_idx,
                d.name AS metric_name,
                d.unit AS unit,
                {sample_start_expr} AS start_ns,
                {sample_end_expr} AS end_ns,
                CAST(m.value AS DOUBLE) AS value
            FROM nsight.NET_NIC_METRIC m
            JOIN dict d
              ON d.metricsListId = m.metricsListId
             AND d.metricsIdx = m.metricsIdx
            JOIN nsight.NIC_ID_MAP map ON map.globalId = m.globalId
            JOIN nsight.TARGET_INFO_NIC_INFO nic ON nic.nicId = map.nicId
            {sample_pred}
        ),
        spans AS (
            SELECT
                s.nic_id,
                s.guid,
                s.nic_name,
                s.global_id,
                s.port_id,
                s.metrics_list_id,
                s.metrics_idx,
                s.metric_name,
                s.unit,
                CAST(b AS BIGINT) AS bucket_idx,
                s.start_ns,
                s.end_ns,
                s.value
            FROM samples s,
                 range(
                     CAST(FLOOR(CAST(s.start_ns - {anchor} AS DOUBLE) / {bucket}) AS BIGINT),
                     CAST(FLOOR(CAST(s.end_ns - 1 - {anchor} AS DOUBLE) / {bucket}) AS BIGINT) + 1
                 ) AS r(b)
        ),
        clipped AS (
            SELECT
                nic_id,
                guid,
                nic_name,
                global_id,
                port_id,
                metrics_list_id,
                metrics_idx,
                metric_name,
                unit,
                {bucket_start_expr} AS t_start,
                {bucket_end_expr} AS t_end,
                {clipped_ns_expr} AS clipped_ns,
                value
            FROM spans
        ),
        agg AS (
            SELECT
                nic_id,
                guid,
                nic_name,
                global_id,
                port_id,
                metrics_list_id,
                metrics_idx,
                metric_name,
                unit,
                t_start,
                t_end,
                SUM(value * CAST(clipped_ns AS DOUBLE))
                    / CAST(SUM(clipped_ns) AS DOUBLE) AS value,
                CAST(COUNT(*) AS BIGINT) AS samples
            FROM clipped
            WHERE clipped_ns > 0
            GROUP BY
                nic_id,
                guid,
                nic_name,
                global_id,
                port_id,
                metrics_list_id,
                metrics_idx,
                metric_name,
                unit,
                t_start,
                t_end
        )
        SELECT *,
               {total_matched}
        FROM agg
        ORDER BY t_start ASC, nic_id ASC, port_id ASC, metrics_list_id ASC, metrics_idx ASC
        LIMIT ?
        "#,
        bucket = bucket_ns,
        sample_start_expr = sample_scope.start_expr,
        sample_end_expr = sample_scope.end_expr,
        sample_pred = sample_scope.where_clause,
        total_matched = total_matched_bigint_expr(),
    );
    params.push(Value::BigInt(req.common.limit as i64));

    let rows = query_rows(trace, &sql, &params, NIC_BUCKETS_SQL, nic_bucket_row)?;
    duckdb_list::split_rows_and_total::<i64, _, _, _>(
        rows,
        duckdb_list::TotalCarrier::Last,
        |row| row.total_matched,
        duckdb_list::infallible_count_error,
        |row| Ok(row.bucket),
    )
}

struct NicBucketRow {
    bucket: NicBucketSample,
    total_matched: i64,
}

fn nic_bucket_row(row: &duckdb::Row<'_>) -> Result<NicBucketRow, duckdb::Error> {
    let t_start_ns: i64 = row.get("t_start")?;
    let nic_id: i64 = row.get("nic_id")?;
    let port_id: i64 = row.get("port_id")?;
    let metrics_idx: i64 = row.get("metrics_idx")?;
    Ok(NicBucketRow {
        bucket: NicBucketSample {
            key: format!(
                "nic_bucket|{t_start_ns}|nic:{nic_id}|port:{port_id}|metric:{metrics_idx}"
            ),
            t_start_ns,
            t_end_ns: row.get("t_end")?,
            nic_id,
            guid: row.get("guid")?,
            nic_name: row.get("nic_name")?,
            global_id: row.get("global_id")?,
            port_id,
            metrics_list_id: row.get("metrics_list_id")?,
            metrics_idx,
            name: row.get("metric_name")?,
            unit: row.get("unit")?,
            agg: "mean",
            value: row.get("value")?,
            samples: row.get("samples")?,
        },
        total_matched: row.get("total_matched")?,
    })
}

/// Sort axes the NIC counter-summary list supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NicCounterSortKey {
    Name,
    NicId,
    PortId,
    MetricsListId,
    MetricsIdx,
    Samples,
    Mean,
    Min,
    Max,
    P50,
    P95,
    P99,
}

impl SortKeyDef for NicCounterSortKey {
    fn specs() -> &'static [SortKeySpec<Self>] {
        &[
            SortKeySpec {
                variant: NicCounterSortKey::Name,
                canonical: "name",
                aliases: &[],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: NicCounterSortKey::NicId,
                canonical: "nic_id",
                aliases: &["nic", "nicid"],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: NicCounterSortKey::PortId,
                canonical: "port_id",
                aliases: &["port", "portid"],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: NicCounterSortKey::MetricsListId,
                canonical: "metrics_list_id",
                aliases: &["metricslistid"],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: NicCounterSortKey::MetricsIdx,
                canonical: "metrics_idx",
                aliases: &["metric", "metric_idx", "idx"],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: NicCounterSortKey::Samples,
                canonical: "samples",
                aliases: &[],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: NicCounterSortKey::Mean,
                canonical: "mean",
                aliases: &["avg"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: NicCounterSortKey::Min,
                canonical: "min",
                aliases: &[],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: NicCounterSortKey::Max,
                canonical: "max",
                aliases: &[],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: NicCounterSortKey::P50,
                canonical: "p50",
                aliases: &["median"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: NicCounterSortKey::P95,
                canonical: "p95",
                aliases: &[],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: NicCounterSortKey::P99,
                canonical: "p99",
                aliases: &[],
                default_dir: Direction::Desc,
            },
        ]
    }
}

fn sort_counters(out: &mut [NicCounterSummary], spec: &SortSpec) -> NsysQueryResult<()> {
    let resolved: Vec<(NicCounterSortKey, Direction)> = spec
        .fields()
        .iter()
        .map(|f| NicCounterSortKey::from_field(f).map_err(NsysQueryError::metrics_sort_invalid))
        .collect::<NsysQueryResult<_>>()?;
    veloq_core::sort_in_memory(
        out,
        &resolved,
        |k, a, b| match k {
            NicCounterSortKey::Name => a.name.cmp(&b.name),
            NicCounterSortKey::NicId => a.nic_id.cmp(&b.nic_id),
            NicCounterSortKey::PortId => a.port_id.cmp(&b.port_id),
            NicCounterSortKey::MetricsListId => a.metrics_list_id.cmp(&b.metrics_list_id),
            NicCounterSortKey::MetricsIdx => a.metrics_idx.cmp(&b.metrics_idx),
            NicCounterSortKey::Samples => a.samples.cmp(&b.samples),
            NicCounterSortKey::Mean => a.mean.total_cmp(&b.mean),
            NicCounterSortKey::Min => a.min.total_cmp(&b.min),
            NicCounterSortKey::Max => a.max.total_cmp(&b.max),
            NicCounterSortKey::P50 => a.p50.total_cmp(&b.p50),
            NicCounterSortKey::P95 => a.p95.total_cmp(&b.p95),
            NicCounterSortKey::P99 => a.p99.total_cmp(&b.p99),
        },
        |a, b| {
            a.nic_id
                .cmp(&b.nic_id)
                .then(a.port_id.cmp(&b.port_id))
                .then(a.metrics_list_id.cmp(&b.metrics_list_id))
                .then(a.metrics_idx.cmp(&b.metrics_idx))
        },
    );
    Ok(())
}
