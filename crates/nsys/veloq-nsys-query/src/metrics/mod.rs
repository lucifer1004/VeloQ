//! `veloq metrics` — hardware-performance counter / CPU sample /
//! scheduler-event queries.
//!
//! Four sources today, each dispatched on `--type`:
//!
//! - **`--type gpu`** reads `GPU_METRICS` + `TARGET_INFO_GPU_METRICS`.
//!   Summary mode reports per-counter min/max/mean/p50/p95/p99;
//!   `--bucket Nms` switches to a long-form time series, with the
//!   aggregator picked from each counter's unit suffix
//!   (`[Cycles Active]` / `[Requests]` roll up by sum, everything
//!   else by mean).
//! - **`--type nic`** reads `NET_NIC_METRIC` +
//!   `TARGET_INFO_NETWORK_METRICS`, with NIC identity joined through
//!   `NIC_ID_MAP` + `TARGET_INFO_NIC_INFO`. Summary mode reports
//!   per-`(NIC, port, counter)` min/max/mean/p50/p95/p99;
//!   `--bucket Nms` returns a long-form `(bucket, NIC, port, counter)`
//!   time series. NSys exports NIC values as rates or already-averaged
//!   sizes, so bucket rollups use mean.
//! - **`--type cpu-sampling`** reads `COMPOSITE_EVENTS` +
//!   `SAMPLING_CALLCHAINS` + `StringIds`. Summary mode returns a
//!   hotspot histogram on the chosen `--group-by` axis
//!   (`symbol` / `tid` / `cpu` / `module`); `--bucket Nms` returns a
//!   long-form count-per-bucket-key time series. Carries three
//!   CPU-sampling-specific trust signals: `unresolved_leaf_share`,
//!   `kernel_leaf_share`, `truncated_stack_share`.
//! - **`--type cpu-sched`** reads `SCHED_EVENTS`. Summary mode
//!   returns a per-key on-cpu / off-cpu / ctx-switch breakdown on
//!   the chosen `--group-by` axis (`tid` / `cpu` / `state`);
//!   `--bucket Nms` returns on-cpu ns per `(t_start, key)` bucket.
//!   Carries two sched-specific trust signals:
//!   `unresolved_state_share` and `per_cpu_max_gap_ns`.
//!
//! Wire shape: [`MetricsResponse`] is a `#[serde(tag = "source")]`
//! tagged enum with one variant per source. Agents dispatching on
//! `--type gpu` see exactly the fields belonging to GPU — no empty
//! `hotspot: []` arrays from a sibling source. Truly shared
//! envelope facts (`coverage`, `metrics_span_ns`, `count`,
//! `total_matched`, `bucket_ns`, …) live under each variant's
//! `common: MetricsCommon` block, so the coverage gate is one
//! navigation step away regardless of which source you queried.
//!
use crate::query_sql::exec::SqlLabel;
use crate::{NsysQueryError, NsysQueryResult};
use serde::Serialize;
use std::path::Path;
use veloq_core::{SortSpec, time::TimeWindow};
use veloq_nsys_data::Trace;

mod cpu_sampling;
mod cpu_sched;
mod gpu;
mod nic;

pub use cpu_sampling::{CpuGroupBy, HotspotRow, HotspotSortKey};
pub use cpu_sched::{SchedGroupBy, SchedSortKey, SchedSummaryRow};
pub use gpu::{BucketSample, CounterSortKey, CounterSummary};
pub use nic::{NicBucketSample, NicCounterSortKey, NicCounterSummary};

// Re-export the public surface so downstream callers (CLI dispatch,
// tests) can `use veloq_nsys_query::metrics::*` without reaching into the
// per-source submodules.
//
// `parse_bucket` is the lone free function — it lives at the module
// root so all three request variants can share one entry point.

/// Selects which metric source veloq queries.
///
/// Used by the CLI layer to parse `--type` strings and dispatch into
/// the matching [`MetricsRequest`] variant. The response side does
/// not carry a `MetricSource` field — the source is the
/// [`MetricsResponse`] tag.
///
/// - `Gpu` reads GPU PM counter samples from `GPU_METRICS`.
/// - `Nic` reads NIC PM counter samples from `NET_NIC_METRIC`.
/// - `CpuSampling` reads CPU IP samples + callchains from
///   `COMPOSITE_EVENTS` + `SAMPLING_CALLCHAINS`.
/// - `CpuSched` reads context-switch events from `SCHED_EVENTS` —
///   a transition stream, not a sample stream, so it yields precise
///   on-cpu durations and context-switch counts.
///
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum MetricSource {
    Gpu,
    Nic,
    CpuSampling,
    CpuSched,
}

impl MetricSource {
    pub fn parse(s: &str) -> NsysQueryResult<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "gpu" => Ok(MetricSource::Gpu),
            "nic" | "network" => Ok(MetricSource::Nic),
            "cpu-sampling" | "cpu_sampling" | "cpu-ip" | "cpu_ip" => Ok(MetricSource::CpuSampling),
            "cpu-sched" | "cpu_sched" | "sched" => Ok(MetricSource::CpuSched),
            other => Err(NsysQueryError::metrics_unknown_source(other)),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            MetricSource::Gpu => "gpu",
            MetricSource::Nic => "nic",
            MetricSource::CpuSampling => "cpu-sampling",
            MetricSource::CpuSched => "cpu-sched",
        }
    }
}

/// Fields every `metrics` request carries, regardless of source.
///
/// `MetricsRequest::Gpu` / `CpuSampling` / `CpuSched` each embed one
/// `MetricsRequestCommon` so adding a new shared knob touches one
/// struct, not three.
#[derive(Debug, Clone)]
pub struct MetricsRequestCommon {
    /// Bucket width in ns for time-series output. `None` returns the
    /// summary view only.
    pub bucket_ns: Option<i64>,
    pub time_window: Option<TimeWindow>,
    /// Sort spec for the summary view. Ignored in bucketed mode
    /// (buckets are always time-ordered).
    pub sort: Option<SortSpec>,
    pub limit: usize,
}

impl Default for MetricsRequestCommon {
    fn default() -> Self {
        Self {
            bucket_ns: None,
            time_window: None,
            sort: None,
            limit: 1000,
        }
    }
}

/// Per-source request payload. The variant *is* the discriminator —
/// `--type gpu --group-by symbol` is a type error at construction
/// time rather than a runtime cross-source check. That construction-
/// time enforcement is the point of the per-source enum.
#[derive(Debug, Clone)]
pub enum MetricsRequest {
    Gpu(GpuMetricsRequest),
    Nic(NicMetricsRequest),
    CpuSampling(CpuSamplingRequest),
    CpuSched(CpuSchedRequest),
}

impl Default for MetricsRequest {
    fn default() -> Self {
        MetricsRequest::Gpu(GpuMetricsRequest::default())
    }
}

#[derive(Debug, Clone, Default)]
pub struct GpuMetricsRequest {
    /// Glob (`*` / `?`) applied to the counter's `metricName` (e.g.
    /// `"SMs Active*"`).
    pub counter_glob: Option<String>,
    pub common: MetricsRequestCommon,
}

#[derive(Debug, Clone, Default)]
pub struct NicMetricsRequest {
    /// Glob (`*` / `?`) applied to the network counter's `name` (e.g.
    /// `"IB: Bytes sent"`).
    pub counter_glob: Option<String>,
    pub common: MetricsRequestCommon,
}

#[derive(Debug, Clone, Default)]
pub struct CpuSamplingRequest {
    /// Aggregation axis (raw string from `--group-by`). Parsed
    /// downstream by `CpuGroupBy::parse`; `None` selects the default
    /// (`symbol`).
    pub group_by: Option<String>,
    /// Glob (`*` / `?`) applied to the leaf frame's `symbol` (with
    /// `--group-by symbol`) or `module` basename (with
    /// `--group-by module`).
    pub name_glob: Option<String>,
    /// Restrict to one CPU id (`COMPOSITE_EVENTS.cpu`).
    pub cpu: Option<i64>,
    /// Restrict to one thread (`globalTid`).
    pub tid: Option<i64>,
    pub common: MetricsRequestCommon,
}

#[derive(Debug, Clone, Default)]
pub struct CpuSchedRequest {
    /// Aggregation axis (raw string from `--group-by`). Parsed
    /// downstream by `SchedGroupBy::parse`; `None` selects the default
    /// (`tid`).
    pub group_by: Option<String>,
    /// Restrict to one CPU id (`SCHED_EVENTS.cpu`).
    pub cpu: Option<i64>,
    /// Restrict to one thread (`globalTid`).
    pub tid: Option<i64>,
    pub common: MetricsRequestCommon,
}

/// Parse a CLI `--bucket` string (`50ms`, `100us`, `1.2s`, …) into
/// ns. Mirrors `TimelineRequest::parse_interval` so both commands
/// reject the same shapes with the same wording.
pub fn parse_bucket(s: &str) -> crate::NsysQueryResult<i64> {
    crate::parse_positive_duration(s, "--bucket")
}

/// Per-source response. `#[serde(tag = "source")]` lifts the variant
/// name to the top of the JSON body, so agents see exactly the
/// fields belonging to their source — no `counters: []` for cpu-*,
/// no `hotspot: []` for gpu.
#[derive(Debug, Serialize, schemars::JsonSchema)]
#[serde(tag = "source", rename_all = "kebab-case")]
pub enum MetricsResponse {
    Gpu(GpuMetricsBody),
    Nic(NicMetricsBody),
    CpuSampling(CpuSamplingBody),
    CpuSched(CpuSchedBody),
}

/// Mode-independent envelope facts every metrics body's auxiliary
/// block carries. Nested under `common` (rather than `serde(flatten)`)
/// so the JSON Schema's `oneOf` discriminator stays clean — flatten +
/// tag interact poorly with schemars 1.x. `count` and
/// `total_matched` hoist to the body's top level (uniform with every
/// other response); the rest of the envelope-level signals
/// (coverage, span, window) stay in `common` under `auxiliary`.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct MetricsCommon {
    /// Primary origin (the anchor `--from`/`--to` apply against).
    /// Returned so an agent can convert raw sample timestamps to
    /// trace-relative offsets without making a second call.
    pub trace_origin_ns: i64,
    /// `(start, end)` over the trace's primary span — kernel / memcpy
    /// / memset / runtime / sync.
    pub trace_span_ns: (i64, i64),
    /// `(min, max)` over the metric-sample timestamps after filters,
    /// or `None` when zero samples match.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics_span_ns: Option<(i64, i64)>,
    pub coverage: Coverage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_window_ns: Option<(i64, i64)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bucket_ns: Option<i64>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct GpuMetricsBody {
    /// Rows returned (after `--limit`).
    pub count: usize,
    /// Rows matching the filter before `--limit`.
    pub total_matched: i64,
    /// Canonical primary table. Per-counter summary statistics
    /// when `--bucket` is not set; empty in bucket mode (the bucket
    /// time-series lives under `auxiliary.buckets`).
    pub rows: Vec<CounterSummary>,
    pub auxiliary: GpuMetricsAuxiliary,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct GpuMetricsAuxiliary {
    pub common: MetricsCommon,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub counter_glob: Option<String>,
    /// GPU long-form bucket series, one row per `(t_start, metric_id)`.
    /// Empty when `--bucket` is not set.
    pub buckets: Vec<BucketSample>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct NicMetricsBody {
    pub count: usize,
    pub total_matched: i64,
    /// Canonical primary table. Per-(NIC, port, counter) summary
    /// statistics when `--bucket` is not set; empty in bucket mode
    /// (the bucket time-series lives under `auxiliary.buckets`).
    pub rows: Vec<NicCounterSummary>,
    pub auxiliary: NicMetricsAuxiliary,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct NicMetricsAuxiliary {
    pub common: MetricsCommon,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub counter_glob: Option<String>,
    /// NIC long-form bucket series, one row per
    /// `(t_start, nic_id, port_id, metrics_idx)`. Empty when
    /// `--bucket` is not set.
    pub buckets: Vec<NicBucketSample>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct CpuSamplingBody {
    pub count: usize,
    pub total_matched: i64,
    /// Canonical primary table. Hotspot histogram when `--bucket`
    /// is not set; empty in bucket mode (the bucket time-series lives
    /// under `auxiliary.cpu_buckets`).
    pub rows: Vec<HotspotRow>,
    pub auxiliary: CpuSamplingAuxiliary,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct CpuSamplingAuxiliary {
    pub common: MetricsCommon,
    /// `--group-by` axis used (`"symbol"` / `"tid"` / `"cpu"` / `"module"`).
    pub group_by: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name_glob: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_filter: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tid_filter: Option<i64>,
    /// Trust signal: fraction of returned samples whose **leaf** frame
    /// has `unresolved=1`. High values mean "we're attributing to
    /// `<unresolved>` buckets a lot; consider better debuginfo /
    /// nsys's `--samples-per-backtrace`".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unresolved_leaf_share: Option<f64>,
    /// Trust signal: fraction of returned samples whose **leaf** frame
    /// has `kernelMode=1`. High values mean "CPU mostly inside
    /// syscalls — likely sleeping in futex_wait, etc., not burning
    /// user code".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_leaf_share: Option<f64>,
    /// Trust signal: fraction of returned samples whose deepest frame
    /// is the sentinel `"[Max depth]"` (NSys's stack-walk-truncation
    /// marker). High values mean stack walks didn't reach the thread
    /// entry — raise `nsys profile --samples-per-backtrace` for
    /// fuller stacks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncated_stack_share: Option<f64>,
    /// Long-form bucket series, one row per `(t_start, group-by key)`.
    /// Empty when `--bucket` is not set.
    pub cpu_buckets: Vec<CpuBucketSample>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct CpuSchedBody {
    pub count: usize,
    pub total_matched: i64,
    /// Canonical primary table. Per-key summary rows when `--bucket`
    /// is not set; empty in bucket mode (the bucket time-series lives
    /// under `auxiliary.cpu_buckets`). Shape depends on `--group-by`
    /// (tid / cpu / state) — see [`SchedSummaryRow`].
    pub rows: Vec<SchedSummaryRow>,
    pub auxiliary: CpuSchedAuxiliary,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct CpuSchedAuxiliary {
    pub common: MetricsCommon,
    /// `--group-by` axis used (`"tid"` / `"cpu"` / `"state"`).
    pub group_by: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_filter: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tid_filter: Option<i64>,
    /// Trust signal: fraction of returned SCHED_EVENTS rows whose
    /// `threadState` is `Unknown` (the kernel didn't tell us where
    /// the thread was going). High values mean the state axis is
    /// unreliable for this capture.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unresolved_state_share: Option<f64>,
    /// Trust signal: the maximum gap (ns) between consecutive
    /// SCHED_EVENTS samples on any single cpu, after filters. Very
    /// large gaps mean "this cpu's stream stopped logging" (sched
    /// buffer drops, or the cpu had no scheduling activity).
    /// `None` when fewer than two events were returned.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_cpu_max_gap_ns: Option<i64>,
    /// On-cpu ns per `(t_start, group-by key)` bucket. Empty when
    /// `--bucket` is not set.
    pub cpu_buckets: Vec<CpuBucketSample>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct Coverage {
    /// Primary count for the source's row set, after `--type`-specific
    /// filters. Source-specific meaning:
    ///
    /// - `--type gpu`: sum of `samples` across counters returned.
    /// - `--type nic`: sum of `samples` across NIC/port/counter rows.
    /// - `--type cpu-sampling`: count of per-leaf samples after
    ///   `--cpu` / `--tid` / `--from`/`--to` filters.
    /// - `--type cpu-sched`: count of `SCHED_EVENTS` rows after
    ///   the same scope filters.
    ///
    /// Reflects the active filters for whichever source this
    /// `Coverage` block lives under.
    pub samples_total: i64,
    /// Largest silent gap inside the filtered metric/event stream, in
    /// ns. For GPU metrics this is the max distance between consecutive
    /// point samples within one `(type_id, metric_id)` stream. For NIC
    /// metrics this is the max uncovered gap between consecutive
    /// interval samples within one `(global_id, port_id, counter)`
    /// stream. `None` when the source does not compute this signal or
    /// when fewer than two samples matched.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_gap_ns: Option<i64>,
    /// Metric/event span clipped to the coverage denominator. This is
    /// still first-to-last span coverage, not gap-removed coverage:
    /// a 1-second silence in the middle of the stream counts toward
    /// this number. For GPU/NIC metrics, use `max_gap_ns` to detect
    /// within-span dropouts.
    pub covered_ns: i64,
    /// Coverage denominator duration. Normally this is
    /// `trace_span_ns.1 - trace_span_ns.0`; metric-only traces fall
    /// back to `metrics_span_ns.1 - metrics_span_ns.0`.
    pub trace_ns: i64,
    /// `covered_ns / trace_ns`, clamped to `[0.0, 1.0]`. `0.0` when
    /// either denominator is zero. Read this as "span / coverage
    /// denominator", not "fraction observed with no gaps" — see
    /// `covered_ns`.
    pub ratio: f64,
}

/// One row in the cpu-sampling bucketed time series. `agg` is always
/// `"sum"` (sample counts have no other meaningful rollup); the
/// field is kept for shape symmetry with `BucketSample`.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct CpuBucketSample {
    pub t_start_ns: i64,
    pub t_end_ns: i64,
    pub key: String,
    pub agg: &'static str,
    pub value: f64,
    pub samples: i64,
}

pub fn run<P: AsRef<Path>>(path: P, req: MetricsRequest) -> NsysQueryResult<MetricsResponse> {
    let common = match &req {
        MetricsRequest::Gpu(r) => &r.common,
        MetricsRequest::Nic(r) => &r.common,
        MetricsRequest::CpuSampling(r) => &r.common,
        MetricsRequest::CpuSched(r) => &r.common,
    };
    crate::check_limit(common.limit)?;
    if let Some(b) = common.bucket_ns
        && b <= 0
    {
        return Err(NsysQueryError::MetricsBucketTooSmall { bucket_ns: b });
    }
    if common.bucket_ns.is_some() && common.sort.is_some() {
        return Err(NsysQueryError::MetricsSortWithBucket);
    }

    let trace = Trace::open(path).map_err(NsysQueryError::trace_open)?;
    // All metric tables (GPU_METRICS, SAMPLING_CALLCHAINS, COMPOSITE_EVENTS,
    // SCHED_EVENTS) resolve under `nsight.<TABLE>` directly.
    let abs_window = trace
        .resolve_window(common.time_window)
        .map_err(NsysQueryError::time_window_resolve)?;
    let (origins, _) = trace.read_origins().map_err(NsysQueryError::data)?;
    let trace_span = origins.primary;
    let trace_origin_ns = trace_span.start_ns;
    let trace_span_ns = (trace_span.start_ns, trace_span.end_ns);

    match req {
        MetricsRequest::Gpu(r) => Ok(MetricsResponse::Gpu(gpu::run_gpu(
            &trace,
            &r,
            abs_window,
            trace_origin_ns,
            trace_span_ns,
        )?)),
        MetricsRequest::Nic(r) => Ok(MetricsResponse::Nic(nic::run_nic(
            &trace,
            &r,
            abs_window,
            trace_origin_ns,
            trace_span_ns,
        )?)),
        MetricsRequest::CpuSampling(r) => Ok(MetricsResponse::CpuSampling(
            cpu_sampling::run_cpu_sampling(&trace, &r, abs_window, trace_origin_ns, trace_span_ns)?,
        )),
        MetricsRequest::CpuSched(r) => Ok(MetricsResponse::CpuSched(cpu_sched::run_cpu_sched(
            &trace,
            &r,
            abs_window,
            trace_origin_ns,
            trace_span_ns,
        )?)),
    }
}

pub(super) fn ratio(metrics_span: Option<(i64, i64)>, trace_span: (i64, i64)) -> f64 {
    let trace_ns = span_ns(trace_span);
    if trace_ns <= 0 {
        return 0.0;
    }
    let covered = covered_ns(metrics_span, trace_span);
    (covered as f64 / trace_ns as f64).clamp(0.0, 1.0)
}

fn span_ns(span: (i64, i64)) -> i64 {
    (span.1 - span.0).max(0)
}

fn coverage_denominator_span(
    metrics_span: Option<(i64, i64)>,
    trace_span: (i64, i64),
) -> (i64, i64) {
    if span_ns(trace_span) > 0 {
        trace_span
    } else {
        metrics_span.unwrap_or(trace_span)
    }
}

fn covered_ns(metrics_span: Option<(i64, i64)>, denominator_span: (i64, i64)) -> i64 {
    let denominator_ns = span_ns(denominator_span);
    if denominator_ns <= 0 {
        return 0;
    }
    let Some((metric_start, metric_end)) = metrics_span else {
        return 0;
    };
    let lo = metric_start.max(denominator_span.0);
    let hi = metric_end.min(denominator_span.1);
    (hi - lo).max(0)
}

/// Prepare `sql`, bind `params`, execute, and project each yielded row via
/// `hydrate` into a `Vec<T>`. Folds away the prepare → bind → loop
/// boilerplate that every metrics SQL call site replays — caller supplies
/// only the SQL body, param slice, error-context label, and per-row
/// projection.
///
/// `label` appears in the prepare-error context (e.g. `"cpu-sched
/// per-tid summary"`).
pub(super) fn query_rows<T>(
    trace: &Trace,
    sql: &str,
    params: &[duckdb::types::Value],
    label: &'static str,
    hydrate: impl FnMut(&duckdb::Row<'_>) -> Result<T, duckdb::Error>,
) -> NsysQueryResult<Vec<T>> {
    query_rows_on_conn(trace.conn(), sql, params, label, hydrate)
}

fn query_rows_on_conn<T>(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[duckdb::types::Value],
    label: &'static str,
    hydrate: impl FnMut(&duckdb::Row<'_>) -> Result<T, duckdb::Error>,
) -> NsysQueryResult<Vec<T>> {
    crate::query_sql::exec::query_rows(conn, sql, params, SqlLabel::new("metrics", label), hydrate)
}

pub(super) fn query_optional_row<T>(
    trace: &Trace,
    sql: &str,
    params: &[duckdb::types::Value],
    label: &'static str,
    hydrate: impl FnOnce(&duckdb::Row<'_>) -> Result<T, duckdb::Error>,
) -> NsysQueryResult<Option<T>> {
    query_optional_row_on_conn(trace.conn(), sql, params, label, hydrate)
}

fn query_optional_row_on_conn<T>(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[duckdb::types::Value],
    label: &'static str,
    hydrate: impl FnOnce(&duckdb::Row<'_>) -> Result<T, duckdb::Error>,
) -> NsysQueryResult<Option<T>> {
    crate::query_sql::exec::query_optional_row(
        conn,
        sql,
        params,
        SqlLabel::new("metrics", label),
        hydrate,
    )
}

impl Coverage {
    /// Build a `Coverage` from the three inputs every metrics source has
    /// at hand: the count of samples that survived its query filter, the
    /// first-to-last timestamp span of those samples (or `None` if none
    /// landed), and the overall trace span. Metric-only traces have an
    /// empty primary trace span, so coverage falls back to the metric
    /// span for its denominator.
    pub(super) fn compute(
        metrics_span: Option<(i64, i64)>,
        trace_span: (i64, i64),
        samples_total: i64,
        max_gap_ns: Option<i64>,
    ) -> Self {
        let denominator_span = coverage_denominator_span(metrics_span, trace_span);
        Self {
            samples_total,
            max_gap_ns,
            covered_ns: covered_ns(metrics_span, denominator_span),
            trace_ns: span_ns(denominator_span),
            ratio: ratio(metrics_span, denominator_span),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use veloq_core::VeloqDiagnostic;

    #[test]
    fn metric_source_parse() {
        assert!(matches!(MetricSource::parse("gpu"), Ok(MetricSource::Gpu)));
        assert!(matches!(MetricSource::parse("GPU"), Ok(MetricSource::Gpu)));
        assert!(matches!(MetricSource::parse("nic"), Ok(MetricSource::Nic)));
        assert!(matches!(
            MetricSource::parse("network"),
            Ok(MetricSource::Nic)
        ));
        assert!(MetricSource::parse("cpu").is_err());
    }

    #[test]
    fn ratio_zero_when_no_samples() {
        assert_eq!(ratio(None, (0, 1_000_000)), 0.0);
    }

    #[test]
    fn ratio_clamps_to_unit() {
        // metrics span is wider than primary span → still clamps to 1.0
        let r = ratio(Some((-100, 2_000_000)), (0, 1_000_000));
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn coverage_falls_back_to_metric_span_when_trace_span_empty() {
        let c = Coverage::compute(Some((100, 200)), (0, 0), 3, Some(10));
        assert_eq!(c.samples_total, 3);
        assert_eq!(c.max_gap_ns, Some(10));
        assert_eq!(c.covered_ns, 100);
        assert_eq!(c.trace_ns, 100);
        assert!((c.ratio - 1.0).abs() < 1e-9);
    }

    #[test]
    fn run_rejects_non_positive_bucket_before_opening_trace() -> anyhow::Result<()> {
        let req = MetricsRequest::Gpu(GpuMetricsRequest {
            common: MetricsRequestCommon {
                bucket_ns: Some(0),
                ..Default::default()
            },
            ..Default::default()
        });

        let err = match run("does-not-need-to-exist.nsys-rep", req) {
            Ok(_) => anyhow::bail!("expected invalid bucket to fail"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.metrics-bucket-too-small");
        assert!(matches!(
            err,
            crate::NsysQueryError::MetricsBucketTooSmall { bucket_ns: 0 }
        ));
        Ok(())
    }

    #[test]
    fn query_rows_prepare_error_is_typed() -> anyhow::Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;

        let err =
            match query_rows_on_conn(&conn, "SELECT * FROM", &[], "test-metrics", |_| Ok(0i64)) {
                Ok(rows) => anyhow::bail!("malformed metrics SQL should not succeed: {rows:?}"),
                Err(err) => err,
            };

        assert_eq!(err.code().as_str(), "nsys.query.sql-prepare");
        assert_eq!(
            err.sql_parts(),
            Some(("metrics", crate::SqlPhase::Prepare, "test-metrics"))
        );
        Ok(())
    }

    #[test]
    fn query_rows_query_error_is_typed() -> anyhow::Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = "SELECT ? AS value WHERE ? IS NOT NULL";
        let params = [duckdb::types::Value::BigInt(1)];

        let err = match query_rows_on_conn(&conn, sql, &params, "test-metrics", |_| Ok(0i64)) {
            Ok(rows) => anyhow::bail!("unbound metrics SQL should not succeed: {rows:?}"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-query");
        assert_eq!(
            err.sql_parts(),
            Some(("metrics", crate::SqlPhase::Query, "test-metrics"))
        );
        Ok(())
    }

    #[test]
    fn query_rows_read_error_is_typed() -> anyhow::Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;

        let err = match query_rows_on_conn(
            &conn,
            "SELECT 'not-an-int' AS value",
            &[],
            "test-metrics",
            |row| row.get::<_, i64>(0),
        ) {
            Ok(rows) => anyhow::bail!("malformed metrics row should not hydrate: {rows:?}"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-read");
        assert_eq!(
            err.sql_parts(),
            Some(("metrics", crate::SqlPhase::Read, "test-metrics"))
        );
        Ok(())
    }

    #[test]
    fn query_optional_row_read_error_is_typed() -> anyhow::Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;

        let err = match query_optional_row_on_conn(
            &conn,
            "SELECT 'not-an-int' AS value",
            &[],
            "test-metrics-optional",
            |row| row.get::<_, i64>(0),
        ) {
            Ok(row) => anyhow::bail!("malformed optional metrics row should not hydrate: {row:?}"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-read");
        assert_eq!(
            err.sql_parts(),
            Some(("metrics", crate::SqlPhase::Read, "test-metrics-optional"))
        );
        Ok(())
    }
}
