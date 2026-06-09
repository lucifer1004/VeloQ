//! Smoke tests for the `concurrency` verb. Asserts the
//! exact per-device / per-stream / compute-vs-copy numbers against the
//! worked-example fixture, including same-stream PDL overlap.

mod fixture;

use anyhow::{Result, anyhow};
use veloq_core::time::TimeWindow;
use veloq_nsys_query::concurrency::{ConcurrencyRequest, run};

#[test]
fn device_stream_and_compute_copy_overlap_match_rfc_example() -> Result<()> {
    let trace = fixture::concurrency_overlap()?;
    let r = run(trace.path(), ConcurrencyRequest::default())?;

    assert_eq!(r.count, 1);
    assert_eq!(r.total_matched, 1);
    let d = r
        .rows
        .first()
        .ok_or_else(|| anyhow!("expected one device row"))?;

    assert_eq!(d.key, "concurrency|dev:0");
    assert_eq!(d.device_id, 0);
    assert_eq!(d.sum_busy_ns, 210_000_000);
    assert_eq!(d.union_busy_ns, 120_000_000);
    assert_eq!(d.overlap_ns, 90_000_000);
    assert_eq!(d.max_concurrency, 3);

    assert_eq!(d.compute_vs_copy.compute_union_ns, 100_000_000);
    assert_eq!(d.compute_vs_copy.copy_union_ns, 40_000_000);
    assert_eq!(d.compute_vs_copy.compute_copy_overlap_ns, 20_000_000);

    // Streams ordered ascending by stream_id; no `key` on nested entries.
    let stream_ids: Vec<i64> = d.streams.iter().map(|s| s.stream_id).collect();
    assert_eq!(stream_ids, vec![7, 8]);

    let s7 = d
        .streams
        .iter()
        .find(|s| s.stream_id == 7)
        .ok_or_else(|| anyhow!("expected stream 7"))?;
    // Same-stream PDL overlap: K2 starts before K1 retires.
    assert_eq!(s7.sum_busy_ns, 110_000_000);
    assert_eq!(s7.union_busy_ns, 100_000_000);
    assert_eq!(s7.overlap_ns, 10_000_000);
    assert_eq!(s7.max_concurrency, 2);

    let s8 = d
        .streams
        .iter()
        .find(|s| s.stream_id == 8)
        .ok_or_else(|| anyhow!("expected stream 8"))?;
    // Compute/copy overlap on the same stream: kernel K3 overlaps memcpy M1.
    assert_eq!(s8.sum_busy_ns, 100_000_000);
    assert_eq!(s8.union_busy_ns, 90_000_000);
    assert_eq!(s8.overlap_ns, 10_000_000);
    assert_eq!(s8.max_concurrency, 2);

    Ok(())
}

#[test]
fn device_filter_keeps_only_requested_device() -> Result<()> {
    let trace = fixture::concurrency_two_devices()?;
    let all = run(trace.path(), ConcurrencyRequest::default())?;
    let all_devices: Vec<i32> = all.rows.iter().map(|row| row.device_id).collect();
    assert_eq!(all_devices, vec![0, 1]);

    let r = run(
        trace.path(),
        ConcurrencyRequest {
            device: Some(1),
            ..Default::default()
        },
    )?;
    assert_eq!(r.count, 1);
    assert_eq!(r.total_matched, 1);
    let d = r.rows.first().ok_or_else(|| anyhow!("expected device 1"))?;
    assert_eq!(d.device_id, 1);
    assert_eq!(d.union_busy_ns, 30_000_000);

    // A device that doesn't exist yields no rows (no empty/zero row).
    let empty = run(
        trace.path(),
        ConcurrencyRequest {
            device: Some(99),
            ..Default::default()
        },
    )?;
    assert_eq!(empty.count, 0);
    assert!(empty.rows.is_empty());
    Ok(())
}

#[test]
fn window_clips_the_measured_overlap() -> Result<()> {
    let trace = fixture::concurrency_overlap()?;
    let full = run(trace.path(), ConcurrencyRequest::default())?;
    let full_union = full
        .rows
        .first()
        .ok_or_else(|| anyhow!("expected device row"))?
        .union_busy_ns;

    // A sub-window must clip the union below the full span (and the
    // echoed window must be present). Origin-robust: only asserts the
    // clip direction, not exact ns.
    let r = run(
        trace.path(),
        ConcurrencyRequest {
            time_window: Some(TimeWindow::parse("0-50ms")?),
            ..Default::default()
        },
    )?;
    assert!(r.time_window_ns.is_some(), "echoed window present");
    let d = r
        .rows
        .first()
        .ok_or_else(|| anyhow!("expected device row"))?;
    assert!(d.union_busy_ns > 0, "some work in window");
    assert!(
        d.union_busy_ns < full_union,
        "window {} must clip below full union {}",
        d.union_busy_ns,
        full_union
    );
    // overlap_ns identity must still hold within the window.
    assert_eq!(d.overlap_ns, d.sum_busy_ns - d.union_busy_ns);
    Ok(())
}
