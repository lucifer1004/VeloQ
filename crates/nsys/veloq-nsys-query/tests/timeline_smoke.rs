//! Integration tests against `timeline::run` on the synthetic
//! GPU fixture. Pins bucket-clipping arithmetic (events spanning
//! bucket boundaries must contribute proportionally to each bucket)
//! and the anchor-to-primary-origin behavior.

mod fixture;

use anyhow::Result;
use veloq_core::time::TimeWindow;
use veloq_nsys_query::timeline::TimelineRequest;

#[test]
fn fifty_ms_bucket_captures_all_fixture_events() -> Result<()> {
    let trace = fixture::minimal_gpu()?;
    // Fixture spans 100..141.5ms; one 50ms bucket [100ms, 150ms]
    // captures everything: 4 kernels (22ms total), 2 memcpys (1ms),
    // 1 memset (0.2ms).
    let req = TimelineRequest {
        interval_ns: 50_000_000,
        ..Default::default()
    };
    let r = veloq_nsys_query::timeline::run(trace.path(), req)?;
    assert_eq!(r.count, 1);
    let b = r
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected one bucket"))?;
    assert_eq!(b.kernel_ns, 22_000_000); // 2*1ms + 2*10ms
    assert_eq!(b.memcpy_ns, 1_000_000); // 2*0.5ms
    assert_eq!(b.memset_ns, 200_000); // 0.2ms
    assert_eq!(b.total_ns, 23_200_000);
    assert_eq!(b.kernel_count, 4);
    assert_eq!(b.memcpy_count, 2);
    assert_eq!(b.memset_count, 1);
    Ok(())
}

#[test]
fn bucket_clipping_preserves_total_duration() -> Result<()> {
    let trace = fixture::minimal_gpu()?;
    // 8ms buckets, anchored at the primary origin (100ms): boundaries
    // 100/108/116/124/132/140/148ms. Several kernels and memcpys
    // straddle these — total per-kind summed across all buckets
    // must still equal the original event totals.
    let req = TimelineRequest {
        interval_ns: 8_000_000,
        ..Default::default()
    };
    let r = veloq_nsys_query::timeline::run(trace.path(), req)?;
    let sum_kernel: i64 = r.rows.iter().map(|b| b.kernel_ns).sum();
    let sum_memcpy: i64 = r.rows.iter().map(|b| b.memcpy_ns).sum();
    let sum_memset: i64 = r.rows.iter().map(|b| b.memset_ns).sum();
    assert_eq!(
        sum_kernel, 22_000_000,
        "all kernel ns must be attributed somewhere"
    );
    assert_eq!(sum_memcpy, 1_000_000);
    assert_eq!(sum_memset, 200_000);
    // Each bucket non-empty (no zero-total rows in output).
    for b in &r.rows {
        assert!(b.total_ns > 0, "empty buckets must be omitted");
    }
    Ok(())
}

#[test]
fn time_window_clips_events_before_bucket_bounds() -> Result<()> {
    let trace = fixture::minimal_gpu()?;
    // The 110..120ms slow kernel overlaps this window by only 2ms.
    // Timeline buckets must be generated from the clipped in-window
    // interval, not from the raw event start/end.
    let req = TimelineRequest {
        interval_ns: 8_000_000,
        time_window: Some(TimeWindow::parse("@118ms-@123ms")?),
        ..Default::default()
    };
    let r = veloq_nsys_query::timeline::run(trace.path(), req)?;
    assert_eq!(r.time_window_ns, Some((118_000_000, 123_000_000)));
    assert_eq!(r.count, 1, "rows: {:?}", r.rows);
    let b = r
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected one clipped bucket"))?;
    assert_eq!(b.start_ns, 118_000_000);
    assert_eq!(b.end_ns, 126_000_000);
    assert_eq!(b.kernel_ns, 2_000_000);
    assert_eq!(b.total_ns, 2_000_000);
    Ok(())
}

#[test]
fn rejects_zero_or_negative_bucket() -> Result<()> {
    let trace = fixture::minimal_gpu()?;
    let req = TimelineRequest {
        interval_ns: 0,
        ..Default::default()
    };
    assert!(veloq_nsys_query::timeline::run(trace.path(), req).is_err());
    Ok(())
}

/// P2 review guard: `timeline --type all --nvtx <pattern>` on a
/// graph-only trace (NVTX_EVENTS + RUNTIME + GRAPH_TRACE, no
/// kernel/memcpy/memset/sync) must not bail at
/// `nvtx_attribution::build` for "no attributable kinds in scope".
/// timeline should implicitly narrow non-attributable kinds (Graph)
/// the same way `search` does — the request collapses to an empty
/// result rather than an error.
#[test]
fn nvtx_with_graph_only_trace_returns_empty_not_error() -> Result<()> {
    use veloq_nsys_query::KindFilter;
    // KindFilter::All + --nvtx: Graph is in timeline's GPU-busy set
    // but not in the attributable set, so the narrowing should drop
    // it. Resulting kinds empty → response is empty, not an error.
    let trace = fixture::graph_only_with_nvtx()?;
    let req = TimelineRequest {
        interval_ns: 50_000_000,
        kinds: KindFilter::All,
        nvtx: Some("step*".into()),
        ..Default::default()
    };
    let resp = veloq_nsys_query::timeline::run(trace.path(), req)?;
    assert_eq!(resp.count, 0, "graph-only with --nvtx narrows to empty");
    assert_eq!(resp.rows.len(), 0);
    Ok(())
}
