//! Integration tests against `slices::run` on the synthetic
//! NVTX-attribution fixture. Specifically pins the `--time-range`
//! overlap semantics so slices/gaps/stats/search stay aligned —
//! a future refactor that re-introduces the old "entirely inside"
//! predicate would fail these tests.

mod fixture;

use anyhow::Result;
use veloq_core::time::TimeWindow;
use veloq_nsys_query::slices::{Slice, SlicesRequest, SlicesRow};

#[test]
fn time_range_uses_overlap_inclusion() -> Result<()> {
    let trace = fixture::nvtx_attribution()?;

    // Window 150..170ms is entirely inside range A [100..200ms]:
    // overlap-inclusion must count A, full-containment would miss it.
    let req = SlicesRequest {
        name: None,
        name_regex: None,
        time_window: Some(TimeWindow::parse("@150000000-@170000000")?),
        sort: None,
        limit: 10,
        ..Default::default()
    };
    let r = veloq_nsys_query::slices::run(trace.path(), req)?;
    assert_eq!(r.count, 1, "range A must be included via overlap");
    assert_eq!(r.total_matched, 1);
    Ok(())
}

#[test]
fn time_range_straddling_edge_qualifies() -> Result<()> {
    let trace = fixture::nvtx_attribution()?;

    // Window 180..220ms partially overlaps range A [100..200ms].
    let req = SlicesRequest {
        name: None,
        name_regex: None,
        time_window: Some(TimeWindow::parse("@180000000-@220000000")?),
        sort: None,
        limit: 10,
        ..Default::default()
    };
    let r = veloq_nsys_query::slices::run(trace.path(), req)?;
    assert_eq!(r.count, 1);
    Ok(())
}

#[test]
fn time_range_excludes_non_overlapping() -> Result<()> {
    let trace = fixture::nvtx_attribution()?;

    // Window 220..280ms falls in the gap between the two ranges and
    // overlaps neither — count must be 0.
    let req = SlicesRequest {
        name: None,
        name_regex: None,
        time_window: Some(TimeWindow::parse("@220000000-@280000000")?),
        sort: None,
        limit: 10,
        ..Default::default()
    };
    let r = veloq_nsys_query::slices::run(trace.path(), req)?;
    assert_eq!(r.count, 0);
    assert_eq!(r.total_matched, 0);
    Ok(())
}

#[test]
fn full_slice_bounds_reported_without_clipping() -> Result<()> {
    let trace = fixture::nvtx_attribution()?;

    // Tiny window inside range A. Slice's cpu bounds should still
    // report the full [100ms, 200ms] — overlap is for inclusion, not
    // for clipping the report.
    let req = SlicesRequest {
        name: None,
        name_regex: None,
        time_window: Some(TimeWindow::parse("@150000000-@151000000")?),
        sort: None,
        limit: 10,
        ..Default::default()
    };
    let r = veloq_nsys_query::slices::run(trace.path(), req)?;
    assert_eq!(r.count, 1);
    let slice = r
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("missing slice"))?;
    let SlicesRow::Instance(slice) = slice else {
        anyhow::bail!("expected instance slice row");
    };
    assert_eq!(slice.cpu.start_ns, 100_000_000);
    assert_eq!(slice.cpu.end_ns, 200_000_000);
    Ok(())
}

/// `--stream` scopes the GPU
/// attribution to one stream. The NVTX range itself always appears
/// (rows are ranges, not stream-scoped events) — `--stream` filters
/// which GPU events count toward its `attributed_*_ns`.
#[test]
fn stream_filter_scopes_gpu_attribution() -> Result<()> {
    let trace = fixture::nvtx_attribution_multistream()?;

    let slice_of = |stream: Option<i64>| -> Result<Slice> {
        let req = SlicesRequest {
            stream,
            limit: 10,
            ..Default::default()
        };
        let r = veloq_nsys_query::slices::run(trace.path(), req)?;
        assert_eq!(r.count, 1, "the one NVTX range is always present");
        let row = r
            .rows
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("missing slice"))?;
        let SlicesRow::Instance(slice) = row else {
            anyhow::bail!("expected instance slice row");
        };
        Ok(slice)
    };

    // No --stream: both kernels attributed (stream 7 = 10ms, stream 8 = 20ms).
    let all = slice_of(None)?;
    assert_eq!(all.attributed_kernel_ns, 30_000_000);
    let mut streams: Vec<i64> = all.gpu_attributed.iter().map(|g| g.stream_id).collect();
    streams.sort_unstable();
    assert_eq!(streams, vec![7, 8]);

    // --stream 7: only the 10ms stream-7 kernel counts.
    let s7 = slice_of(Some(7))?;
    assert_eq!(s7.attributed_kernel_ns, 10_000_000);
    let s7_streams: Vec<i64> = s7.gpu_attributed.iter().map(|g| g.stream_id).collect();
    assert_eq!(s7_streams, vec![7]);

    // --stream 8: only the 20ms stream-8 kernel counts.
    let s8 = slice_of(Some(8))?;
    assert_eq!(s8.attributed_kernel_ns, 20_000_000);

    // --stream 99: matches no GPU event; the range persists with zero
    // attribution (slices rows are NVTX ranges, so the row stays).
    let s99 = slice_of(Some(99))?;
    assert_eq!(s99.attributed_kernel_ns, 0);
    assert!(s99.gpu_attributed.is_empty());

    Ok(())
}
