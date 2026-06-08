//! Integration tests against `metrics::run` on the synthetic
//! `with_gpu_metrics` fixture. Pins:
//!
//! - Per-counter summary: min/max/mean over a deterministic sample
//!   stream lets us assert exact values without floating-point fudge.
//! - Counter glob via `--counter`: matches by `metricName`.
//! - Time-window clipping: a window that excludes half the samples
//!   drops `samples` by half.
//! - Coverage block: ratio reflects observed-span / primary-span.
//! - Bucketed mode: aggregator selection (mean for `[Throughput %]`,
//!   sum for `[Requests]`) and bucket counts.
//!
//! No-panic policy: this file mirrors the rest of the suite — every
//! "I expect this to be present" assertion uses `ok_or_else` + `?`
//! instead of `.unwrap()` / `.expect()` / `[i]`. The workspace lint
//! config denies the panicking forms even in tests.

mod fixture;

use anyhow::{Result, anyhow, bail};
use duckdb::Connection;
use veloq_core::{SortSpec, time::TimeWindow};
use veloq_nsys_query::metrics::{
    BucketSample, CounterSummary, GpuMetricsBody, GpuMetricsRequest, MetricSource, MetricsRequest,
    MetricsResponse,
};

/// Helper: locate a counter by metric_id or fail the test with a
/// helpful message.
fn counter_by_id(counters: &[CounterSummary], id: i64) -> Result<&CounterSummary> {
    counters
        .iter()
        .find(|c| c.metric_id == id)
        .ok_or_else(|| anyhow!("expected counter with metric_id={id} in response"))
}

/// Helper: collect buckets for one counter, sorted by start time.
fn buckets_for(buckets: &[BucketSample], metric_id: i64) -> Vec<&BucketSample> {
    let mut v: Vec<&BucketSample> = buckets
        .iter()
        .filter(|b| b.metric_id == metric_id)
        .collect();
    v.sort_by_key(|b| b.t_start_ns);
    v
}

/// Build a GPU request via a closure-on-default builder.
fn gpu_req(build: impl FnOnce(&mut GpuMetricsRequest)) -> MetricsRequest {
    let mut r = GpuMetricsRequest::default();
    build(&mut r);
    MetricsRequest::Gpu(r)
}

/// Unwrap a [`MetricsResponse`] into its GPU body, or fail the test
/// loudly. Mirrors the rest of the suite's "no panics in tests"
/// policy — every helper returns `Result` so call sites use `?`.
fn expect_gpu(r: MetricsResponse) -> Result<GpuMetricsBody> {
    match r {
        MetricsResponse::Gpu(b) => Ok(b),
        MetricsResponse::Nic(_) => bail!("expected gpu variant, got nic"),
        MetricsResponse::CpuSampling(_) => bail!("expected gpu variant, got cpu-sampling"),
        MetricsResponse::CpuSched(_) => bail!("expected gpu variant, got cpu-sched"),
    }
}

#[test]
fn missing_table_errors_with_actionable_message() -> Result<()> {
    // Reuse minimal_gpu — no GPU_METRICS table. The error should name
    // the absent table + suggest a re-capture flag.
    let trace = fixture::minimal_gpu()?;
    let r = veloq_nsys_query::metrics::run(trace.path(), MetricsRequest::default());
    let err = match r {
        Ok(_) => bail!("expected metrics on a trace without GPU_METRICS to error"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(msg.contains("GPU_METRICS"), "got: {msg}");
    assert!(msg.contains("nsys profile"), "got: {msg}");
    Ok(())
}

#[test]
fn summary_covers_both_counters() -> Result<()> {
    let trace = fixture::with_gpu_metrics()?;
    let r = expect_gpu(veloq_nsys_query::metrics::run(
        trace.path(),
        MetricsRequest::default(),
    )?)?;

    assert_eq!(r.rows.len(), 2);
    assert_eq!(r.count, 2);
    assert_eq!(r.total_matched, 2);
    assert!(r.auxiliary.common.bucket_ns.is_none());
    assert!(r.auxiliary.buckets.is_empty());

    // Sorted by name ASC (default). PCIe < SMs alphabetically.
    let names: Vec<&str> = r.rows.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "PCIe Read Requests to BAR1 [Requests]",
            "SMs Active [Throughput %]"
        ]
    );

    // Counter 0 (Throughput %): values 0,10,…,90 → min=0, max=90, mean=45.
    let sms = counter_by_id(&r.rows, 0)?;
    assert_eq!(
        sms.key,
        format!("counter|type:{}|metric:{}", sms.type_id, sms.metric_id)
    );
    assert_eq!(sms.unit.as_deref(), Some("Throughput %"));
    assert_eq!(sms.agg, "mean");
    assert_eq!(sms.samples, 10);
    assert!((sms.min - 0.0).abs() < 1e-9);
    assert!((sms.max - 90.0).abs() < 1e-9);
    assert!((sms.mean - 45.0).abs() < 1e-9);

    // Counter 1 (Requests): constant 4 → mean=4, sum-agg.
    let pcie = counter_by_id(&r.rows, 1)?;
    assert_eq!(
        pcie.key,
        format!("counter|type:{}|metric:{}", pcie.type_id, pcie.metric_id)
    );
    assert_eq!(pcie.unit.as_deref(), Some("Requests"));
    assert_eq!(pcie.agg, "sum");
    assert_eq!(pcie.samples, 10);
    assert!((pcie.mean - 4.0).abs() < 1e-9);
    Ok(())
}

#[test]
fn coverage_reflects_metrics_span_over_primary_span() -> Result<()> {
    let trace = fixture::with_gpu_metrics()?;
    let r = expect_gpu(veloq_nsys_query::metrics::run(
        trace.path(),
        MetricsRequest::default(),
    )?)?;

    // Primary span: the lone anchor kernel 100ms..110ms = 10ms.
    // Metrics span: 10 samples 1ms apart, 100ms..109ms = 9ms.
    let (lo, hi) = r
        .auxiliary
        .common
        .metrics_span_ns
        .ok_or_else(|| anyhow!("expected samples to produce a metrics span"))?;
    assert_eq!(hi - lo, 9_000_000);
    assert_eq!(r.auxiliary.common.coverage.covered_ns, 9_000_000);
    assert_eq!(r.auxiliary.common.coverage.trace_ns, 10_000_000);
    assert!((r.auxiliary.common.coverage.ratio - 0.9).abs() < 1e-9);
    assert_eq!(r.auxiliary.common.coverage.samples_total, 20); // 10 per counter × 2
    assert_eq!(r.auxiliary.common.coverage.max_gap_ns, Some(1_000_000));
    Ok(())
}

#[test]
fn coverage_surfaces_mid_span_sample_gap() -> Result<()> {
    let trace = fixture::with_gpu_metrics()?;
    // The fixture path is a `_pqtdir/`; rewrite
    // GPU_METRICS.parquet in place with the middle-span rows dropped
    // so the coverage gap surfaces.
    let parquet = trace.path().join("GPU_METRICS.parquet");
    let parquet_lit = parquet.to_string_lossy().replace('\'', "''");
    let conn = Connection::open_in_memory().map_err(anyhow::Error::from)?;
    conn.execute(
        &format!(
            "CREATE TABLE GPU_METRICS AS SELECT * FROM read_parquet('{parquet_lit}') \
             WHERE NOT (timestamp >= ? AND timestamp < ?)"
        ),
        duckdb::params![103_000_000i64, 107_000_000i64],
    )?;
    conn.execute(
        &format!("COPY GPU_METRICS TO '{parquet_lit}' (FORMAT PARQUET)"),
        [],
    )?;
    drop(conn);

    let r = expect_gpu(veloq_nsys_query::metrics::run(
        trace.path(),
        MetricsRequest::default(),
    )?)?;

    assert_eq!(
        r.auxiliary.common.coverage.max_gap_ns,
        Some(5_000_000),
        "largest point-sample gap should expose the missing middle samples"
    );
    assert_eq!(r.auxiliary.common.coverage.samples_total, 12);
    Ok(())
}

#[test]
fn counter_glob_filters_to_one_counter() -> Result<()> {
    let trace = fixture::with_gpu_metrics()?;
    let req = gpu_req(|r| {
        r.counter_glob = Some("SMs Active*".to_string());
    });
    let r = expect_gpu(veloq_nsys_query::metrics::run(trace.path(), req)?)?;
    assert_eq!(r.rows.len(), 1);
    let only = r
        .rows
        .first()
        .ok_or_else(|| anyhow!("expected one counter row"))?;
    assert_eq!(only.metric_id, 0);
    assert!(only.name.starts_with("SMs Active"));
    Ok(())
}

#[test]
fn unmatched_glob_returns_actionable_error() -> Result<()> {
    let trace = fixture::with_gpu_metrics()?;
    let req = gpu_req(|r| {
        r.counter_glob = Some("DefinitelyNotACounter".to_string());
    });
    let r = veloq_nsys_query::metrics::run(trace.path(), req);
    let err = match r {
        Ok(_) => bail!("empty glob match must error rather than silently empty"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(msg.contains("no GPU counters match"), "got: {msg}");
    Ok(())
}

#[test]
fn time_window_clips_samples() -> Result<()> {
    // Window 100..105ms covers samples at 100/101/102/103/104 ms (5 each).
    let trace = fixture::with_gpu_metrics()?;
    let window = TimeWindow::parse("0-5ms")?;
    let req = gpu_req(move |r| {
        r.common.time_window = Some(window);
    });
    let r = expect_gpu(veloq_nsys_query::metrics::run(trace.path(), req)?)?;
    for c in &r.rows {
        assert_eq!(c.samples, 5, "counter {} should be clipped to 5", c.name);
    }
    // Coverage tracks the clipped sample span (100ms..104ms = 4ms).
    assert_eq!(r.auxiliary.common.coverage.samples_total, 10);
    let (lo, hi) = r
        .auxiliary
        .common
        .metrics_span_ns
        .ok_or_else(|| anyhow!("expected in-window samples"))?;
    assert_eq!(hi - lo, 4_000_000);
    Ok(())
}

#[test]
fn bucket_mode_emits_long_form_rows() -> Result<()> {
    let trace = fixture::with_gpu_metrics()?;
    // 5ms buckets across 10ms of samples → 2 buckets per counter × 2
    // counters = 4 rows. (Samples at 100..109ms; buckets [100,105)
    // and [105,110).)
    let req = gpu_req(|r| {
        r.common.bucket_ns = Some(5_000_000);
    });
    let r = expect_gpu(veloq_nsys_query::metrics::run(trace.path(), req)?)?;
    assert_eq!(r.auxiliary.common.bucket_ns, Some(5_000_000));
    assert_eq!(r.auxiliary.buckets.len(), 4);
    assert!(
        r.rows.is_empty(),
        "bucket mode should not also return counter summaries"
    );
    assert_eq!(r.count, 4);
    assert_eq!(r.total_matched, 4);

    // Counter 0 (Throughput %, mean agg): values 0..40 in bucket 1
    // (mean=20), 50..90 in bucket 2 (mean=70).
    let sms = buckets_for(&r.auxiliary.buckets, 0);
    assert_eq!(sms.len(), 2);
    let b0 = sms
        .first()
        .ok_or_else(|| anyhow!("missing bucket 0 for metric 0"))?;
    let b1 = sms
        .get(1)
        .ok_or_else(|| anyhow!("missing bucket 1 for metric 0"))?;
    assert_eq!(
        b0.key,
        format!(
            "bucket|{}|type:{}|metric:{}",
            b0.t_start_ns, b0.type_id, b0.metric_id
        )
    );
    assert_eq!(b0.agg, "mean");
    assert!((b0.value - 20.0).abs() < 1e-9);
    assert!((b1.value - 70.0).abs() < 1e-9);

    // Counter 1 (Requests, sum agg): 5 samples of 4 per bucket → sum=20.
    let pcie = buckets_for(&r.auxiliary.buckets, 1);
    assert_eq!(pcie.len(), 2);
    let p0 = pcie
        .first()
        .ok_or_else(|| anyhow!("missing bucket 0 for metric 1"))?;
    let p1 = pcie
        .get(1)
        .ok_or_else(|| anyhow!("missing bucket 1 for metric 1"))?;
    assert_eq!(
        p0.key,
        format!(
            "bucket|{}|type:{}|metric:{}",
            p0.t_start_ns, p0.type_id, p0.metric_id
        )
    );
    assert_eq!(p0.agg, "sum");
    assert!((p0.value - 20.0).abs() < 1e-9);
    assert!((p1.value - 20.0).abs() < 1e-9);
    Ok(())
}

#[test]
fn bucket_mode_rejects_sort() -> Result<()> {
    let trace = fixture::with_gpu_metrics()?;
    let req = gpu_req(|r| {
        r.common.bucket_ns = Some(5_000_000);
        r.common.sort = Some(SortSpec::single("mean"));
    });
    let r = veloq_nsys_query::metrics::run(trace.path(), req);
    let err = match r {
        Ok(_) => bail!("--sort with --bucket should be rejected"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("doesn't apply in bucketed mode"));
    Ok(())
}

#[test]
fn sort_metric_id_descending() -> Result<()> {
    let trace = fixture::with_gpu_metrics()?;
    let sort = SortSpec::parse("-metric_id")?;
    let req = gpu_req(move |r| {
        r.common.sort = Some(sort);
    });
    let r = expect_gpu(veloq_nsys_query::metrics::run(trace.path(), req)?)?;
    let first = r
        .rows
        .first()
        .ok_or_else(|| anyhow!("expected first counter"))?;
    let second = r
        .rows
        .get(1)
        .ok_or_else(|| anyhow!("expected second counter"))?;
    assert_eq!(first.metric_id, 1);
    assert_eq!(second.metric_id, 0);
    Ok(())
}

#[test]
fn multi_device_buckets_keep_typeid_distinct() -> Result<()> {
    // Two devices reporting the same metricId must stay in distinct
    // buckets — a typeId collapse would average them (10 + 90 → 50).
    let trace = fixture::with_gpu_metrics_multi_device()?;
    let req = gpu_req(|r| {
        // one bucket covers all 5 samples per device
        r.common.bucket_ns = Some(5_000_000);
    });
    let r = expect_gpu(veloq_nsys_query::metrics::run(trace.path(), req)?)?;
    // 2 typeIds × 1 metricId × 1 bucket = 2 rows.
    assert_eq!(r.auxiliary.buckets.len(), 2, "expected one row per device");
    let type_a: i64 = 281479271677952;
    let type_b: i64 = 281479271677953;
    let bucket_a = r
        .auxiliary
        .buckets
        .iter()
        .find(|b| b.type_id == type_a)
        .ok_or_else(|| anyhow!("missing bucket for type_a"))?;
    let bucket_b = r
        .auxiliary
        .buckets
        .iter()
        .find(|b| b.type_id == type_b)
        .ok_or_else(|| anyhow!("missing bucket for type_b"))?;
    assert_eq!(bucket_a.metric_id, 0);
    assert_eq!(bucket_b.metric_id, 0);
    assert!(
        (bucket_a.value - 10.0).abs() < 1e-9,
        "got {}",
        bucket_a.value
    );
    assert!(
        (bucket_b.value - 90.0).abs() < 1e-9,
        "got {}",
        bucket_b.value
    );
    assert_eq!(bucket_a.samples, 5);
    assert_eq!(bucket_b.samples, 5);
    Ok(())
}

#[test]
fn unknown_source_errors() -> Result<()> {
    let r = MetricSource::parse("definitely-not-a-source");
    let err = match r {
        Ok(_) => bail!("unknown metric source should fail"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(msg.contains("supported"), "got: {msg}");
    assert!(msg.contains("gpu"), "got: {msg}");
    assert!(msg.contains("nic"), "got: {msg}");
    Ok(())
}
