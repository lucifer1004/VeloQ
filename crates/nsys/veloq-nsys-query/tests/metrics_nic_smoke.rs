//! Integration tests for `metrics --type nic` against a synthetic
//! NSys-style `NET_NIC_METRIC` fixture. Pins the real export schema
//! observed from `nsys profile --nic-metrics=lf`: samples are interval
//! rows keyed by `(globalId, portId, metricsListId, metricsIdx)`, with
//! names and units coming from `TARGET_INFO_NETWORK_METRICS`.

mod fixture;

use anyhow::{Result, anyhow, bail};
use duckdb::Connection;
use veloq_core::time::TimeWindow;
use veloq_nsys_query::metrics::{
    MetricsRequest, MetricsResponse, NicBucketSample, NicCounterSummary, NicMetricsBody,
    NicMetricsRequest,
};

fn counter_by_idx(counters: &[NicCounterSummary], idx: i64) -> Result<&NicCounterSummary> {
    counters
        .iter()
        .find(|c| c.metrics_idx == idx)
        .ok_or_else(|| anyhow!("expected NIC counter metrics_idx={idx} in response"))
}

fn buckets_for_idx(buckets: &[NicBucketSample], idx: i64) -> Vec<&NicBucketSample> {
    let mut v: Vec<&NicBucketSample> = buckets.iter().filter(|b| b.metrics_idx == idx).collect();
    v.sort_by_key(|b| b.t_start_ns);
    v
}

fn nic_req(build: impl FnOnce(&mut NicMetricsRequest)) -> MetricsRequest {
    let mut r = NicMetricsRequest::default();
    build(&mut r);
    MetricsRequest::Nic(r)
}

fn expect_nic(r: MetricsResponse) -> Result<NicMetricsBody> {
    match r {
        MetricsResponse::Nic(b) => Ok(b),
        MetricsResponse::Gpu(_) => bail!("expected nic variant, got gpu"),
        MetricsResponse::CpuSampling(_) => bail!("expected nic variant, got cpu-sampling"),
        MetricsResponse::CpuSched(_) => bail!("expected nic variant, got cpu-sched"),
    }
}

#[test]
fn missing_table_errors_with_capture_hint() -> Result<()> {
    let trace = fixture::minimal_gpu()?;
    let r = veloq_nsys_query::metrics::run(
        trace.path(),
        MetricsRequest::Nic(NicMetricsRequest::default()),
    );
    let err = match r {
        Ok(_) => bail!("expected metrics on a trace without NET_NIC_METRIC to error"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(msg.contains("NET_NIC_METRIC"), "got: {msg}");
    assert!(msg.contains("--nic-metrics"), "got: {msg}");
    assert!(msg.contains("nsys status --network"), "got: {msg}");
    Ok(())
}

#[test]
fn summary_covers_nic_counters() -> Result<()> {
    let trace = fixture::with_nic_metrics()?;
    let r = expect_nic(veloq_nsys_query::metrics::run(
        trace.path(),
        nic_req(|_| {}),
    )?)?;

    assert_eq!(r.rows.len(), 2);
    assert_eq!(r.count, 2);
    assert_eq!(r.total_matched, 2);
    assert!(r.auxiliary.common.bucket_ns.is_none());
    assert!(r.auxiliary.buckets.is_empty());

    let names: Vec<&str> = r.rows.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, vec!["IB: Bytes sent", "IB: Send waits"]);

    let bytes = counter_by_idx(&r.rows, 6)?;
    assert_eq!(
        bytes.key,
        format!(
            "nic_counter|nic:{}|port:{}|metric:{}",
            bytes.nic_id, bytes.port_id, bytes.metrics_idx
        )
    );
    assert_eq!(bytes.nic_id, 0);
    assert_eq!(bytes.nic_name, "mlx5_0");
    assert_eq!(bytes.port_id, 0);
    assert_eq!(bytes.metrics_list_id, 0);
    assert_eq!(bytes.unit.as_deref(), Some("bytes/ms"));
    assert_eq!(bytes.agg, "mean");
    assert_eq!(bytes.samples, 10);
    assert!((bytes.min - 0.0).abs() < 1e-9);
    assert!((bytes.max - 90.0).abs() < 1e-9);
    assert!((bytes.mean - 45.0).abs() < 1e-9);

    let waits = counter_by_idx(&r.rows, 10)?;
    assert_eq!(
        waits.key,
        format!(
            "nic_counter|nic:{}|port:{}|metric:{}",
            waits.nic_id, waits.port_id, waits.metrics_idx
        )
    );
    assert_eq!(waits.unit.as_deref(), Some("ticks/ms"));
    assert_eq!(waits.samples, 10);
    assert!((waits.mean - 4.0).abs() < 1e-9);
    Ok(())
}

#[test]
fn coverage_uses_interval_span() -> Result<()> {
    let trace = fixture::with_nic_metrics()?;
    let r = expect_nic(veloq_nsys_query::metrics::run(
        trace.path(),
        nic_req(|_| {}),
    )?)?;

    let (lo, hi) = r
        .auxiliary
        .common
        .metrics_span_ns
        .ok_or_else(|| anyhow!("expected NIC samples to produce a metrics span"))?;
    assert_eq!(hi - lo, 10_000_000);
    assert_eq!(r.auxiliary.common.coverage.covered_ns, 10_000_000);
    assert_eq!(r.auxiliary.common.coverage.trace_ns, 10_000_000);
    assert!((r.auxiliary.common.coverage.ratio - 1.0).abs() < 1e-9);
    assert_eq!(r.auxiliary.common.coverage.samples_total, 20);
    assert_eq!(r.auxiliary.common.coverage.max_gap_ns, Some(0));
    Ok(())
}

#[test]
fn coverage_surfaces_mid_span_interval_gap() -> Result<()> {
    let trace = fixture::with_nic_metrics()?;
    // The fixture path is a `_pqtdir/`; rewrite
    // NET_NIC_METRIC.parquet in place with the middle-span rows
    // dropped so the coverage gap surfaces.
    let parquet = trace.path().join("NET_NIC_METRIC.parquet");
    let parquet_lit = parquet.to_string_lossy().replace('\'', "''");
    let conn = Connection::open_in_memory().map_err(anyhow::Error::from)?;
    conn.execute(
        &format!(
            "CREATE TABLE NET_NIC_METRIC AS SELECT * FROM read_parquet('{parquet_lit}') \
             WHERE NOT (start >= ? AND start < ?)"
        ),
        duckdb::params![103_000_000i64, 107_000_000i64],
    )?;
    conn.execute(
        &format!("COPY NET_NIC_METRIC TO '{parquet_lit}' (FORMAT PARQUET)"),
        [],
    )?;
    drop(conn);

    let r = expect_nic(veloq_nsys_query::metrics::run(
        trace.path(),
        nic_req(|_| {}),
    )?)?;

    assert_eq!(
        r.auxiliary.common.coverage.max_gap_ns,
        Some(4_000_000),
        "largest interval gap should expose the missing middle samples"
    );
    assert_eq!(r.auxiliary.common.coverage.samples_total, 12);
    assert!((r.auxiliary.common.coverage.ratio - 1.0).abs() < 1e-9);
    Ok(())
}

#[test]
fn counter_glob_filters_to_one_counter() -> Result<()> {
    let trace = fixture::with_nic_metrics()?;
    let r = expect_nic(veloq_nsys_query::metrics::run(
        trace.path(),
        nic_req(|r| {
            r.counter_glob = Some("IB: Bytes*".to_string());
        }),
    )?)?;
    assert_eq!(r.rows.len(), 1);
    let only = r
        .rows
        .first()
        .ok_or_else(|| anyhow!("expected one NIC counter row"))?;
    assert_eq!(only.metrics_idx, 6);
    assert_eq!(only.name, "IB: Bytes sent");
    Ok(())
}

#[test]
fn unmatched_glob_returns_actionable_error() -> Result<()> {
    let trace = fixture::with_nic_metrics()?;
    let r = veloq_nsys_query::metrics::run(
        trace.path(),
        nic_req(|r| {
            r.counter_glob = Some("DefinitelyNotANicCounter".to_string());
        }),
    );
    let err = match r {
        Ok(_) => bail!("empty glob match must error rather than silently empty"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(msg.contains("no NIC counters match"), "got: {msg}");
    Ok(())
}

#[test]
fn time_window_clips_interval_samples() -> Result<()> {
    let trace = fixture::with_nic_metrics()?;
    let window = TimeWindow::parse("0-5ms")?;
    let r = expect_nic(veloq_nsys_query::metrics::run(
        trace.path(),
        nic_req(move |r| {
            r.common.time_window = Some(window);
        }),
    )?)?;
    for c in &r.rows {
        assert_eq!(c.samples, 5, "counter {} should be clipped to 5", c.name);
    }
    let (lo, hi) = r
        .auxiliary
        .common
        .metrics_span_ns
        .ok_or_else(|| anyhow!("expected in-window NIC samples"))?;
    assert_eq!(hi - lo, 5_000_000);
    Ok(())
}

#[test]
fn bucket_mode_clips_intervals_to_time_window() -> Result<()> {
    let trace = fixture::with_nic_metrics()?;
    let window = TimeWindow::parse("500us-5ms")?;
    let r = expect_nic(veloq_nsys_query::metrics::run(
        trace.path(),
        nic_req(move |r| {
            r.counter_glob = Some("IB: Bytes*".to_string());
            r.common.time_window = Some(window);
            r.common.bucket_ns = Some(5_000_000);
        }),
    )?)?;

    assert_eq!(r.auxiliary.buckets.len(), 1);
    let b = r
        .auxiliary
        .buckets
        .first()
        .ok_or_else(|| anyhow!("missing clipped NIC bucket"))?;
    assert_eq!(b.t_start_ns, 100_500_000);
    assert_eq!(b.t_end_ns, 105_500_000);
    assert_eq!(b.samples, 5);
    assert!((b.value - (100.0 / 4.5)).abs() < 1e-9);
    Ok(())
}

#[test]
fn bucket_mode_emits_long_form_rows() -> Result<()> {
    let trace = fixture::with_nic_metrics()?;
    let r = expect_nic(veloq_nsys_query::metrics::run(
        trace.path(),
        nic_req(|r| {
            r.common.bucket_ns = Some(5_000_000);
        }),
    )?)?;
    assert_eq!(r.auxiliary.common.bucket_ns, Some(5_000_000));
    assert_eq!(r.auxiliary.buckets.len(), 4);
    assert!(
        r.rows.is_empty(),
        "bucket mode should not also return counter summaries"
    );
    assert_eq!(r.count, 4);
    assert_eq!(r.total_matched, 4);

    let bytes = buckets_for_idx(&r.auxiliary.buckets, 6);
    assert_eq!(bytes.len(), 2);
    let b0 = bytes
        .first()
        .ok_or_else(|| anyhow!("missing bucket 0 for NIC metric 6"))?;
    let b1 = bytes
        .get(1)
        .ok_or_else(|| anyhow!("missing bucket 1 for NIC metric 6"))?;
    assert_eq!(
        b0.key,
        format!(
            "nic_bucket|{}|nic:{}|port:{}|metric:{}",
            b0.t_start_ns, b0.nic_id, b0.port_id, b0.metrics_idx
        )
    );
    assert_eq!(b0.agg, "mean");
    assert_eq!(b0.name, "IB: Bytes sent");
    assert!((b0.value - 20.0).abs() < 1e-9);
    assert!((b1.value - 70.0).abs() < 1e-9);

    let waits = buckets_for_idx(&r.auxiliary.buckets, 10);
    assert_eq!(waits.len(), 2);
    let w0 = waits
        .first()
        .ok_or_else(|| anyhow!("missing bucket 0 for NIC metric 10"))?;
    let w1 = waits
        .get(1)
        .ok_or_else(|| anyhow!("missing bucket 1 for NIC metric 10"))?;
    assert_eq!(
        w0.key,
        format!(
            "nic_bucket|{}|nic:{}|port:{}|metric:{}",
            w0.t_start_ns, w0.nic_id, w0.port_id, w0.metrics_idx
        )
    );
    assert_eq!(w0.agg, "mean");
    assert!((w0.value - 4.0).abs() < 1e-9);
    assert!((w1.value - 4.0).abs() < 1e-9);
    Ok(())
}
