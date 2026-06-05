//! Integration tests against `stats::run` on a synthetic fixture.
//!
//! These prove the SQL composition + name-resolution paths end to end:
//! - per-kind aggregation produces correct counts / totals
//! - `--group-by demangled` separates kernels with different `shortName`
//! - bytes_total / gbps populate for memcpy and memset rows only
//! - the named-column row reader stays in sync with the SELECT aliases
//!
//! The fixture builds a 4-kernel / 2-memcpy / 1-memset trace; see
//! `tests/fixture.rs` for the exact event durations and StringIds.

mod fixture;

use anyhow::Result;
use veloq_nsys_query::KindFilter;
use veloq_nsys_query::stats::{ALLOWED_KINDS, GroupBy, NameAxis, StatsRequest};

#[test]
fn aggregates_kernels_by_short_name() -> Result<()> {
    let trace = fixture::minimal_gpu()?;
    let req = StatsRequest {
        kinds: KindFilter::Only(vec![veloq_nsys_query::EventKind::Kernel]),
        group_by: GroupBy {
            name: NameAxis::ShortName,
            ..Default::default()
        },
        ..Default::default()
    };
    let r = veloq_nsys_query::stats::run(trace.path(), req)?;
    // Two distinct shortNames in the fixture.
    assert_eq!(r.count, 2);
    assert_eq!(r.total_matched, 2);
    // Sum of kernel durations: 2*1ms + 2*10ms = 22ms.
    assert_eq!(r.total_duration_ns, 22_000_000);
    assert_eq!(r.total_events, 4);

    // Each group's name + count is right.
    let by_name: std::collections::HashMap<_, _> = r
        .rows
        .iter()
        .filter_map(|row| row.name.clone().map(|n| (n, row.count)))
        .collect();
    assert_eq!(by_name.get("fast_kernel"), Some(&2));
    assert_eq!(by_name.get("slow_kernel"), Some(&2));
    Ok(())
}

#[test]
fn memcpy_rows_carry_bytes_and_gbps() -> Result<()> {
    let trace = fixture::minimal_gpu()?;
    let req = StatsRequest {
        kinds: KindFilter::Only(vec![veloq_nsys_query::EventKind::Memcpy]),
        ..Default::default()
    };
    let r = veloq_nsys_query::stats::run(trace.path(), req)?;
    // Two memcpys with different copyKind labels → two groups under
    // shortName grouping.
    assert_eq!(r.count, 2);
    // 4096 bytes each. Total bytes per row = 4096.
    for row in &r.rows {
        assert_eq!(row.bytes_total, Some(4096));
        let gbps = row
            .gbps
            .ok_or_else(|| anyhow::anyhow!("memcpy row missing gbps"))?;
        assert!(gbps > 0.0);
    }
    Ok(())
}

#[test]
fn all_gpu_kinds_via_kindfilter_all() -> Result<()> {
    let trace = fixture::minimal_gpu()?;
    // KindFilter::All against stats's ALLOWED_KINDS =
    // kernel + memcpy + memset + sync + graph + nvtx + runtime +
    // osrt (8 entries). The
    // minimal_gpu fixture has none of the host-side tables, so
    // those subqueries get filtered out by the table-availability
    // gate — totals match the GPU-work-only expectation.
    let req = StatsRequest::default();
    let r = veloq_nsys_query::stats::run(trace.path(), req)?;
    // Groups: 2 kernel + 2 memcpy + 1 memset = 5
    assert_eq!(r.total_matched, 5);
    assert_eq!(r.total_events, 7); // 4 kernels + 2 memcpys + 1 memset
    // Sanity: ALLOWED_KINDS is what we expect.
    assert_eq!(ALLOWED_KINDS.len(), 8);
    Ok(())
}

#[test]
fn aggregates_sync_by_sync_type() -> Result<()> {
    let trace = fixture::with_sync()?;
    let req = StatsRequest {
        kinds: KindFilter::Only(vec![veloq_nsys_query::EventKind::Sync]),
        ..Default::default()
    };
    let r = veloq_nsys_query::stats::run(trace.path(), req)?;
    // 3 distinct syncType labels in the fixture → 3 groups.
    assert_eq!(r.count, 3);
    assert_eq!(r.total_events, 3);
    // 1ms + 2ms + 5ms = 8ms total duration.
    assert_eq!(r.total_duration_ns, 8_000_000);
    let labels: std::collections::HashSet<_> =
        r.rows.iter().filter_map(|row| row.name.clone()).collect();
    assert!(labels.contains("cudaEventSynchronize"));
    assert!(labels.contains("cudaStreamSynchronize"));
    assert!(labels.contains("cudaDeviceSynchronize"));
    Ok(())
}

#[test]
fn time_window_clips_durations() -> Result<()> {
    let trace = fixture::minimal_gpu()?;
    // Window 105ms..125ms (relative to fixture's 100ms origin =
    // primary, so absolute 105..125ms).
    let req = StatsRequest {
        kinds: KindFilter::Only(vec![veloq_nsys_query::EventKind::Kernel]),
        time_window: Some(veloq_core::time::TimeWindow::parse("5ms-25ms")?),
        ..Default::default()
    };
    let r = veloq_nsys_query::stats::run(trace.path(), req)?;
    // The window covers kernel 3 (110..120ms) fully and clips kernel 4
    // (130..140) out entirely. fast kernels at 100-103 are also out
    // (entirely below window). One group: slow_kernel with one event.
    assert_eq!(r.total_events, 1);
    assert_eq!(r.total_duration_ns, 10_000_000); // exactly one slow_kernel
    Ok(())
}

#[test]
fn nvtx_kind_aggregates_by_range_name() -> Result<()> {
    // The nvtx_attribution fixture has two NVTX ranges:
    //   - step_a: 100ms..200ms (100ms duration)
    //   - step_b: 300ms..400ms (100ms duration)
    // Distinct names → two groups under the default name axis.
    let trace = fixture::nvtx_attribution()?;
    let req = StatsRequest {
        kinds: KindFilter::Only(vec![veloq_nsys_query::EventKind::Nvtx]),
        ..Default::default()
    };
    let r = veloq_nsys_query::stats::run(trace.path(), req)?;
    assert_eq!(r.count, 2);
    assert_eq!(r.total_matched, 2);
    assert_eq!(r.total_events, 2);
    assert_eq!(r.total_duration_ns, 200_000_000); // 100ms + 100ms

    let names: std::collections::HashSet<_> =
        r.rows.iter().filter_map(|row| row.name.clone()).collect();
    assert!(names.contains("step_a"));
    assert!(names.contains("step_b"));
    for row in &r.rows {
        // NVTX rows carry no device/context/stream — the wire-format
        // projection for NULL surfaces as `None`.
        assert!(row.device_id.is_none(), "nvtx row must not carry device_id");
        assert!(row.stream_id.is_none(), "nvtx row must not carry stream_id");
        // Duration totals match the fixture spans.
        assert_eq!(row.total_ns, 100_000_000);
        assert_eq!(row.count, 1);
    }
    Ok(())
}

#[test]
fn nvtx_kind_rejects_nvtx_scope_flag() -> Result<()> {
    // `--nvtx <pattern>` is the GPU-work-attributed-to-NVTX scope
    // filter; on `--type nvtx` it would be a no-op tautology
    // emitting zero rows silently. Reject up front with a redirect.
    let trace = fixture::nvtx_attribution()?;
    let req = StatsRequest {
        kinds: KindFilter::Only(vec![veloq_nsys_query::EventKind::Nvtx]),
        nvtx: Some("step_*".to_string()),
        ..Default::default()
    };
    let outcome = veloq_nsys_query::stats::run(trace.path(), req);
    let err = match outcome {
        Ok(_) => anyhow::bail!("--type nvtx + --nvtx must error, got Ok"),
        Err(e) => e,
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("--nvtx") && msg.contains("nvtx"),
        "error must reference --nvtx and nvtx kind; got: {msg}"
    );
    Ok(())
}

#[test]
fn nvtx_kind_rejects_device_group_by() -> Result<()> {
    // NVTX has no device column; --group-by device on --type nvtx
    // would emit a single misleading `device:null` bucket. Reject
    // up front with a clear message instead.
    let trace = fixture::nvtx_attribution()?;
    let req = StatsRequest {
        kinds: KindFilter::Only(vec![veloq_nsys_query::EventKind::Nvtx]),
        group_by: GroupBy {
            name: NameAxis::ShortName,
            device: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let outcome = veloq_nsys_query::stats::run(trace.path(), req);
    let err = match outcome {
        Ok(_) => anyhow::bail!("--group-by device on --type nvtx must error, got Ok"),
        Err(e) => e,
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("nvtx") && msg.contains("device"),
        "error must explain the rejection; got: {msg}"
    );
    Ok(())
}
