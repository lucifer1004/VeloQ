//! `--type gpu` source.
//!
//! Reads `GPU_METRICS` (PM counter samples) + `TARGET_INFO_GPU_METRICS`
//! (the counter dictionary). Summary mode reports per-counter
//! min/max/mean/p50/p95/p99; `--bucket Nms` switches to a long-form
//! time series with the per-counter aggregator picked from the unit
//! suffix (`[Cycles Active]` / `[Requests]` roll up by sum, every
//! other unit by mean).

use anyhow::{Context, Result};
use duckdb::types::Value;
use serde::Serialize;
use veloq_core::{Direction, SortKeyDef, SortKeySpec, SortSpec};
use veloq_nsys_data::Trace;

use super::{Coverage, GpuMetricsBody, GpuMetricsRequest, MetricsCommon};

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct CounterSummary {
    /// Cross-trace key. `counter|type:<type_id>|metric:<metric_id>`
    /// — disambiguates a same-name counter across GPUs while still
    /// joining cleanly when two traces have matching `(type_id,
    /// metric_id)` pairs (the common case).
    pub key: String,
    pub metric_id: i64,
    pub type_id: i64,
    pub name: String,
    /// Parsed from the `[X]` suffix in `name` (e.g. `"Throughput %"`,
    /// `"Cycles Active"`). `None` when no `[...]` suffix is present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    /// Aggregator used for `--bucket` rollups. `"sum"` when the unit
    /// looks like a tally (`Cycles Active`, `Requests`), `"mean"`
    /// otherwise.
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
pub struct BucketSample {
    /// Cross-trace key. `bucket|<t_start_ns>|type:<type_id>|metric:<metric_id>`.
    pub key: String,
    pub t_start_ns: i64,
    pub t_end_ns: i64,
    /// `TARGET_INFO_GPU_METRICS.typeId` — uniquely identifies the
    /// `(provider, GPU)` pair the bucket came from. On
    /// `--gpu-metrics-devices=all` captures, the same `metric_id`
    /// (e.g. `7` for "SMs Active") repeats across every GPU; this
    /// field disambiguates them so an agent can either pivot per-GPU
    /// or aggregate across — agg layer can't, since values aren't
    /// summable across devices.
    pub type_id: i64,
    pub metric_id: i64,
    pub agg: &'static str,
    pub value: f64,
    pub samples: i64,
}

pub(super) fn run_gpu(
    trace: &Trace,
    req: &GpuMetricsRequest,
    abs_window: Option<(i64, i64)>,
    trace_origin_ns: i64,
    trace_span_ns: (i64, i64),
) -> Result<GpuMetricsBody> {
    if !trace.table_exists("GPU_METRICS") {
        anyhow::bail!(
            "metrics --type gpu requires `GPU_METRICS`, which is absent from this trace; \
             re-capture with `nsys profile --gpu-metrics-devices=…`"
        );
    }
    if !trace.table_exists("TARGET_INFO_GPU_METRICS") {
        anyhow::bail!(
            "metrics --type gpu requires `TARGET_INFO_GPU_METRICS` (counter dictionary); \
             likely a partial or corrupted nsys export"
        );
    }

    let counters = query_gpu_counters(trace, req, abs_window)?;

    // Pre-limit total before we truncate counters / buckets.
    let total_counters = counters.len() as i64;
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
        Some(bucket) => {
            query_gpu_buckets(trace, req, abs_window, bucket, &counters, trace_origin_ns)?
        }
    };

    let mut summaries: Vec<CounterSummary> = counters.into_iter().map(|c| c.summary).collect();
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
        None => {
            debug_assert_eq!(total_counters, summaries_pre_limit_len as i64);
            (summaries.len(), summaries_pre_limit_len as i64)
        }
        Some(_) => (buckets.len(), total_buckets_before_limit),
    };

    Ok(GpuMetricsBody {
        count,
        total_matched,
        rows: summaries,
        auxiliary: super::GpuMetricsAuxiliary {
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

/// Internal pairing: each GPU counter summary plus the
/// `(min, max)` timestamp span of its samples — kept private so the
/// public response shape stays a flat list, and `metrics_span_ns` /
/// coverage are derived once at the top.
struct CounterWithSpan {
    summary: CounterSummary,
    span_lo: i64,
    span_hi: i64,
    max_gap_ns: Option<i64>,
}

fn query_gpu_counters(
    trace: &Trace,
    req: &GpuMetricsRequest,
    abs_window: Option<(i64, i64)>,
) -> Result<Vec<CounterWithSpan>> {
    let mut params: Vec<Value> = Vec::new();

    // Counter dictionary: optionally filtered by glob via LIKE. We keep
    // the LIKE in SQL (instead of dict-then-filter in Rust) so the
    // GPU_METRICS join only walks rows for matching counters.
    let dict_pred = if req.counter_glob.is_some() {
        params.push(Value::Text(crate::search_glob_to_like(
            req.counter_glob.as_deref().unwrap_or(""),
        )));
        r#"WHERE metricName LIKE ? ESCAPE '\'"#
    } else {
        ""
    };

    // Time window — overlap-semantics here is just an inclusive
    // half-open interval on `timestamp` since samples are points.
    let (window_pred, window_params) = build_window_pred(abs_window);
    params.extend(window_params);

    let sql = format!(
        r#"
        WITH dict AS (
            SELECT typeId, metricId, metricName
            FROM nsight.TARGET_INFO_GPU_METRICS
            {dict_pred}
        ),
        samples AS (
            SELECT
                m.typeId,
                m.metricId,
                m.timestamp,
                CAST(m.value AS DOUBLE) AS value
            FROM nsight.GPU_METRICS m
            JOIN dict d ON d.typeId = m.typeId AND d.metricId = m.metricId
            {window_pred}
        ),
        sampled AS (
            SELECT
                *,
                LAG(timestamp) OVER (
                    PARTITION BY typeId, metricId
                    ORDER BY timestamp
                ) AS prev_timestamp
            FROM samples
        ),
        samples_with_gap AS (
            SELECT
                *,
                CASE
                    WHEN prev_timestamp IS NULL THEN NULL
                    ELSE timestamp - prev_timestamp
                END AS gap_ns
            FROM sampled
        )
        SELECT
            d.typeId AS type_id,
            d.metricId AS metric_id,
            d.metricName AS metric_name,
            CAST(COUNT(s.timestamp) AS BIGINT) AS samples,
            COALESCE(MIN(s.value), 0.0) AS min_v,
            COALESCE(MAX(s.value), 0.0) AS max_v,
            COALESCE(AVG(s.value), 0.0) AS mean_v,
            -- APPROX_QUANTILE (DuckDB's t-digest, ε ≈ 1%) instead of the
            -- exact `quantile_disc` that `stats` uses. Sample streams
            -- have unbounded cardinality (every counter × every tick
            -- across a long capture); exact percentiles would force a
            -- full sort per group on every call. APPROX is the right
            -- tradeoff at this row count and the wire format doesn't
            -- promise more than two decimals anyway.
            COALESCE(APPROX_QUANTILE(s.value, 0.50), 0.0) AS p50_v,
            COALESCE(APPROX_QUANTILE(s.value, 0.95), 0.0) AS p95_v,
            COALESCE(APPROX_QUANTILE(s.value, 0.99), 0.0) AS p99_v,
            CAST(COALESCE(MIN(s.timestamp), 0) AS BIGINT) AS span_lo,
            CAST(COALESCE(MAX(s.timestamp), 0) AS BIGINT) AS span_hi,
            CAST(MAX(s.gap_ns) AS BIGINT) AS max_gap_ns
        FROM dict d
        LEFT JOIN samples_with_gap s
          ON s.typeId = d.typeId AND s.metricId = d.metricId
        GROUP BY d.typeId, d.metricId, d.metricName
        -- No ORDER BY: `sort_counters` re-orders in Rust per the
        -- user's `--sort` spec, so any SQL ordering would be wasted.
        "#
    );

    let conn = trace.conn();
    let mut stmt = conn.prepare(&sql).context("preparing GPU metrics SQL")?;
    let params_ref = crate::bind(&params);
    let mut rows = stmt.query(params_ref.as_slice())?;

    let mut out: Vec<CounterWithSpan> = Vec::new();
    while let Some(r) = rows.next()? {
        let name: String = r.get("metric_name")?;
        let unit = parse_unit(&name);
        let agg = infer_agg(unit.as_deref());
        let samples: i64 = r.get("samples")?;
        let type_id: i64 = r.get("type_id")?;
        let metric_id: i64 = r.get("metric_id")?;
        out.push(CounterWithSpan {
            summary: CounterSummary {
                key: format!("counter|type:{type_id}|metric:{metric_id}"),
                type_id,
                metric_id,
                name,
                unit,
                agg,
                samples,
                min: r.get("min_v")?,
                max: r.get("max_v")?,
                mean: r.get("mean_v")?,
                p50: r.get("p50_v")?,
                p95: r.get("p95_v")?,
                p99: r.get("p99_v")?,
            },
            span_lo: if samples > 0 {
                r.get("span_lo")?
            } else {
                i64::MAX
            },
            span_hi: if samples > 0 {
                r.get("span_hi")?
            } else {
                i64::MIN
            },
            max_gap_ns: r.get("max_gap_ns")?,
        });
    }
    // A glob with no matching counter in the dictionary is a user
    // error worth surfacing — silently empty data hides typos.
    if out.is_empty()
        && let Some(g) = &req.counter_glob
    {
        anyhow::bail!(
            "no GPU counters match `--counter {g}`; \
                 run `veloq metrics <trace> --type gpu` (no --counter) to list available names"
        );
    }
    Ok(out)
}

fn query_gpu_buckets(
    trace: &Trace,
    req: &GpuMetricsRequest,
    abs_window: Option<(i64, i64)>,
    bucket_ns: i64,
    counters: &[CounterWithSpan],
    primary_origin_ns: i64,
) -> Result<(Vec<BucketSample>, i64)> {
    // Anchor buckets so user-typed offsets line up: window start if
    // given, primary origin otherwise. `primary_origin_ns` is plumbed
    // through from `run()` so we don't pay a second `read_origins()`
    // call here — it's cached on the handle but the redundant call
    // confused profiles. Matches timeline.rs's anchor logic.
    let anchor = abs_window.map(|(s, _)| s).unwrap_or(primary_origin_ns);

    let mut params: Vec<Value> = Vec::new();
    let dict_pred = if req.counter_glob.is_some() {
        params.push(Value::Text(crate::search_glob_to_like(
            req.counter_glob.as_deref().unwrap_or(""),
        )));
        r#"WHERE metricName LIKE ? ESCAPE '\'"#
    } else {
        ""
    };

    // Per-counter aggregator: build a CASE expression keyed by
    // metricId. SQL params would be cleaner, but DuckDB doesn't accept
    // a parameter inside a CASE label, so we splice the metric_id
    // integers (which we trust — they come straight from the dictionary
    // query we just ran, not from user input).
    let mut sum_metric_ids: Vec<i64> = Vec::new();
    for c in counters {
        if c.summary.agg == "sum" {
            sum_metric_ids.push(c.summary.metric_id);
        }
    }
    let agg_expr = if sum_metric_ids.is_empty() {
        "AVG(value)".to_string()
    } else {
        // Comma-joined integer list — safe because we built it
        // ourselves from trusted i64s. No string user input here.
        let ids = sum_metric_ids
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",");
        format!("CASE WHEN metricId IN ({ids}) THEN SUM(value) ELSE AVG(value) END")
    };

    let (window_pred, window_params) = build_window_pred(abs_window);
    params.extend(window_params);

    let sql = format!(
        r#"
        WITH dict AS (
            SELECT typeId, metricId
            FROM nsight.TARGET_INFO_GPU_METRICS
            {dict_pred}
        ),
        samples AS (
            SELECT
                m.typeId,
                m.metricId,
                m.timestamp,
                CAST(m.value AS DOUBLE) AS value
            FROM nsight.GPU_METRICS m
            JOIN dict d ON d.typeId = m.typeId AND d.metricId = m.metricId
            {window_pred}
        ),
        -- Carry `typeId` all the way through so multi-GPU captures
        -- (`--gpu-metrics-devices=all`) keep per-device buckets
        -- distinct. Without this two GPUs reporting metricId=7
        -- "SMs Active" silently collapse into one row whose value is
        -- their cross-GPU average.
        bucketed AS (
            SELECT
                typeId,
                metricId,
                CAST(FLOOR(CAST(timestamp - {anchor} AS DOUBLE) / {bucket}) AS BIGINT)
                    * {bucket} + {anchor} AS t_start,
                value
            FROM samples
        ),
        agg AS (
            SELECT
                typeId,
                metricId,
                t_start,
                t_start + {bucket} AS t_end,
                {agg_expr} AS value,
                CAST(COUNT(*) AS BIGINT) AS samples
            FROM bucketed
            GROUP BY typeId, metricId, t_start
        )
        SELECT *,
               CAST(COUNT(*) OVER () AS BIGINT) AS total_matched
        FROM agg
        ORDER BY t_start ASC, typeId ASC, metricId ASC
        LIMIT ?
        "#,
        bucket = bucket_ns,
    );
    params.push(Value::BigInt(req.common.limit as i64));

    let conn = trace.conn();
    let mut stmt = conn
        .prepare(&sql)
        .context("preparing GPU metrics bucket SQL")?;
    let params_ref = crate::bind(&params);
    let mut rows = stmt.query(params_ref.as_slice())?;

    // Build a metric_id → agg label map so each row carries the
    // aggregator that produced it. Matches what `--counter` summary
    // already exposes; lets a CSV consumer reconstruct semantics
    // without re-deriving from the name.
    let agg_by_id: std::collections::HashMap<i64, &'static str> = counters
        .iter()
        .map(|c| (c.summary.metric_id, c.summary.agg))
        .collect();

    let mut out: Vec<BucketSample> = Vec::new();
    let mut total_matched: i64 = 0;
    while let Some(r) = rows.next()? {
        let type_id: i64 = r.get("typeId")?;
        let metric_id: i64 = r.get("metricId")?;
        let agg = agg_by_id.get(&metric_id).copied().unwrap_or("mean");
        let t_start_ns: i64 = r.get("t_start")?;
        out.push(BucketSample {
            key: format!("bucket|{t_start_ns}|type:{type_id}|metric:{metric_id}"),
            t_start_ns,
            t_end_ns: r.get("t_end")?,
            type_id,
            metric_id,
            agg,
            value: r.get("value")?,
            samples: r.get("samples")?,
        });
        total_matched = r.get("total_matched")?;
    }
    Ok((out, total_matched))
}

fn build_window_pred(abs_window: Option<(i64, i64)>) -> (String, Vec<Value>) {
    match abs_window {
        Some((start, end)) => (
            "WHERE m.timestamp >= ? AND m.timestamp < ?".to_string(),
            vec![Value::BigInt(start), Value::BigInt(end)],
        ),
        None => (String::new(), Vec::new()),
    }
}

/// Parse the trailing `[X]` suffix in a counter name (`"SMs Active
/// [Throughput %]"` → `Some("Throughput %")`). Returns `None` when the
/// suffix is absent or malformed.
fn parse_unit(name: &str) -> Option<String> {
    let trimmed = name.trim_end();
    let close = trimmed.rfind(']')?;
    if close + 1 != trimmed.len() {
        return None;
    }
    let open = trimmed[..close].rfind('[')?;
    let body = trimmed[open + 1..close].trim();
    if body.is_empty() {
        None
    } else {
        Some(body.to_string())
    }
}

/// Choose the bucket aggregator for a counter based on its unit. Sum
/// for tallies (`Cycles Active`, `Requests`); mean for everything
/// else, which covers throughput percentages, frequencies, and the
/// per-warp averages nsys already pre-aggregates.
///
/// **Narrow allow-list, on purpose.** Real NSys counter sets include
/// other tally-shaped units (`Bytes`, `Instructions Issued`, …) that
/// we currently mean-aggregate. Starting conservative is safer than
/// silently summing the wrong things; each bucket row carries the
/// chosen `agg` token so downstream consumers can override
/// post-hoc. Follow-up audit-from-real-capture will broaden this.
fn infer_agg(unit: Option<&str>) -> &'static str {
    match unit {
        Some(u)
            if u.eq_ignore_ascii_case("Cycles Active") || u.eq_ignore_ascii_case("Requests") =>
        {
            "sum"
        }
        _ => "mean",
    }
}

/// Sort axes the gpu counter-summary list supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CounterSortKey {
    Name,
    MetricId,
    Samples,
    Mean,
    Min,
    Max,
    P50,
    P95,
    P99,
}

impl SortKeyDef for CounterSortKey {
    fn specs() -> &'static [SortKeySpec<Self>] {
        // Default direction matches what an agent would intuit:
        // identity axes (name, metric_id, min) ASC, magnitude axes
        // (samples, mean, max, percentiles) DESC.
        &[
            SortKeySpec {
                variant: CounterSortKey::Name,
                canonical: "name",
                aliases: &[],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: CounterSortKey::MetricId,
                canonical: "metric_id",
                aliases: &["metricid", "id"],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: CounterSortKey::Samples,
                canonical: "samples",
                aliases: &[],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: CounterSortKey::Mean,
                canonical: "mean",
                aliases: &["avg"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: CounterSortKey::Min,
                canonical: "min",
                aliases: &[],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: CounterSortKey::Max,
                canonical: "max",
                aliases: &[],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: CounterSortKey::P50,
                canonical: "p50",
                aliases: &["median"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: CounterSortKey::P95,
                canonical: "p95",
                aliases: &[],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: CounterSortKey::P99,
                canonical: "p99",
                aliases: &[],
                default_dir: Direction::Desc,
            },
        ]
    }
}

/// Sort the counter-summary list per the user's `--sort` spec. Keys
/// match what the JSON exposes — `name`, `metric_id`, `samples`,
/// `mean`, `min`, `max`, `p50`, `p95`, `p99`.
fn sort_counters(out: &mut [CounterSummary], spec: &SortSpec) -> Result<()> {
    let resolved: Vec<(CounterSortKey, Direction)> = spec
        .fields()
        .iter()
        .map(|f| CounterSortKey::from_field(f).map_err(Into::into))
        .collect::<Result<_>>()?;
    // f64 comparison: total_cmp() is total-ordering and ignores NaN
    // surprises — safer than partial_cmp for sort closures. Stable
    // tiebreak on metric_id ASC.
    veloq_core::sort_in_memory(
        out,
        &resolved,
        |k, a, b| match k {
            CounterSortKey::Name => a.name.cmp(&b.name),
            CounterSortKey::MetricId => a.metric_id.cmp(&b.metric_id),
            CounterSortKey::Samples => a.samples.cmp(&b.samples),
            CounterSortKey::Mean => a.mean.total_cmp(&b.mean),
            CounterSortKey::Min => a.min.total_cmp(&b.min),
            CounterSortKey::Max => a.max.total_cmp(&b.max),
            CounterSortKey::P50 => a.p50.total_cmp(&b.p50),
            CounterSortKey::P95 => a.p95.total_cmp(&b.p95),
            CounterSortKey::P99 => a.p99.total_cmp(&b.p99),
        },
        |a, b| a.metric_id.cmp(&b.metric_id),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_unit_extracts_bracketed_suffix() {
        assert_eq!(
            parse_unit("SMs Active [Throughput %]").as_deref(),
            Some("Throughput %")
        );
        assert_eq!(
            parse_unit("PCIe Read Requests to BAR1 [Requests]").as_deref(),
            Some("Requests")
        );
        assert_eq!(parse_unit("GPC Clock Frequency").as_deref(), None);
        assert_eq!(parse_unit("Weird [] suffix").as_deref(), None);
    }

    #[test]
    fn infer_agg_picks_sum_for_tallies() {
        assert_eq!(infer_agg(Some("Cycles Active")), "sum");
        assert_eq!(infer_agg(Some("Requests")), "sum");
        assert_eq!(infer_agg(Some("Throughput %")), "mean");
        assert_eq!(infer_agg(Some("MHz")), "mean");
        assert_eq!(infer_agg(None), "mean");
    }
}
