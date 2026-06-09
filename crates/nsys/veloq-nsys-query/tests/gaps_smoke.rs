//! Integration tests against `gaps::run` on synthetic fixtures.
//!
//! Covers the three scopes (`device` default / `stream` / `trace`),
//! the upstream rejection of meaningless flag combinations, and the
//! `auxiliary.streams[]` busy-ratio surface that stays useful across
//! every scope.

mod fixture;

use anyhow::Result;
use veloq_core::{SortSpec, time::TimeWindow};
use veloq_nsys_query::EventKind;
use veloq_nsys_query::gaps::{GapScope, GapsRequest, run};

#[test]
fn default_scope_is_device() {
    // Locks in the design decision: cold/no-flag callers get the
    // unified per-device view, not the per-stream one.
    assert_eq!(GapsRequest::default().scope, GapScope::Device);
}

#[test]
fn auxiliary_streams_carries_busy_ratio() -> Result<()> {
    // minimal_gpu places 4 kernels and 2 memcpys on (device=0,
    // stream=7) and 1 memset on the same stream. The busy_ratio
    // auxiliary is computed independently of scope, so the default
    // (device-unified) request returns the same per-stream summary.
    let trace = fixture::minimal_gpu()?;
    let r = run(
        trace.path(),
        GapsRequest {
            min_ns: 1,
            limit: 100,
            ..Default::default()
        },
    )?;
    let streams = &r.auxiliary.streams;
    let s = streams
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected one stream entry, got: {streams:?}"))?;
    assert_eq!(streams.len(), 1, "got: {streams:?}");
    assert_eq!(s.device_id, 0);
    assert_eq!(s.stream_id, 7);
    assert_eq!(s.key, "stream|dev:0|stream:7");
    assert!(
        s.span_ns > 0,
        "span_ns must be positive for a non-degenerate trace"
    );
    assert!(
        s.busy_ns > 0 && s.busy_ns <= s.span_ns,
        "busy_ns ({}) must fit in span_ns ({})",
        s.busy_ns,
        s.span_ns
    );
    assert!(
        (0.0..=1.0).contains(&s.busy_ratio),
        "busy_ratio out of range: {}",
        s.busy_ratio
    );
    Ok(())
}

#[test]
fn stream_scope_keeps_per_stream_filter() -> Result<()> {
    // --scope stream is the only scope where --stream <id> is
    // valid. Non-matching stream id → empty rows and empty aux.
    let trace = fixture::minimal_gpu()?;
    let r = run(
        trace.path(),
        GapsRequest {
            scope: GapScope::Stream,
            min_ns: 1,
            stream: Some(99),
            limit: 100,
            ..Default::default()
        },
    )?;
    assert!(r.rows.is_empty(), "no gaps on a stream the trace lacks");
    assert!(
        r.auxiliary.streams.is_empty(),
        "streams aux should mirror the empty scope, got: {:?}",
        r.auxiliary.streams
    );
    Ok(())
}

#[test]
fn stream_filter_rejected_under_device_scope() -> Result<()> {
    // --stream under the device scope would silently drop gaps
    // bracketed by events on other streams. Reject upfront.
    let trace = fixture::minimal_gpu()?;
    let outcome = run(
        trace.path(),
        GapsRequest {
            min_ns: 1,
            stream: Some(7),
            limit: 100,
            ..Default::default()
        },
    );
    let err = match outcome {
        Ok(_) => anyhow::bail!("--stream + --scope device must error, got Ok"),
        Err(e) => e,
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("--stream") && msg.contains("--scope"),
        "error must explain the rejection; got: {msg}"
    );
    Ok(())
}

#[test]
fn device_filter_rejected_under_trace_scope() -> Result<()> {
    // --device under --scope trace would hit a NULL device_id
    // projection and silently return empty. Reject upfront and
    // point the user at --scope device --device N instead.
    let trace = fixture::minimal_gpu()?;
    let outcome = run(
        trace.path(),
        GapsRequest {
            scope: GapScope::Trace,
            min_ns: 1,
            device: Some(0),
            limit: 100,
            ..Default::default()
        },
    );
    let err = match outcome {
        Ok(_) => anyhow::bail!("--device + --scope trace must error, got Ok"),
        Err(e) => e,
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("--device") && msg.contains("--scope trace"),
        "error must explain the rejection; got: {msg}"
    );
    Ok(())
}

#[test]
fn device_scope_emits_unified_keys() -> Result<()> {
    // Single-stream fixture: device-scope gaps still appear (the
    // 4 kernels + 2 memcpys + 1 memset have inter-event gaps),
    // but the row key is the device-scoped shape and stream_id
    // is `None` on the enclosing Gap (the prev/next neighbors
    // carry it).
    let trace = fixture::minimal_gpu()?;
    let r = run(
        trace.path(),
        GapsRequest {
            min_ns: 1, // catch everything
            limit: 100,
            ..Default::default()
        },
    )?;
    assert_eq!(r.scope, "device");
    let g = r
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected at least one device-scoped gap"))?;
    assert!(
        g.key.starts_with("gap|dev:") && !g.key.contains("stream:"),
        "device-scope key omits stream axis; got: {}",
        g.key
    );
    assert!(
        g.stream_id.is_none(),
        "device-scope rows omit Gap.stream_id; got: {:?}",
        g.stream_id
    );
    assert_eq!(g.device_id, Some(0));
    // Bracketing events always carry stream context.
    assert_eq!(g.prev.stream_id, 7);
    assert_eq!(g.next.stream_id, 7);
    Ok(())
}

#[test]
fn unified_time_window_preserves_cross_window_gap() -> Result<()> {
    // Window 105ms..109ms contains no GPU event body in minimal_gpu,
    // but it overlaps the idle gap between the 104.0..104.5ms memcpy
    // and the 110..120ms kernel. Local-window sweep input must keep
    // both bracketing frontier events.
    let trace = fixture::minimal_gpu()?;
    for scope in [GapScope::Device, GapScope::Trace] {
        let r = run(
            trace.path(),
            GapsRequest {
                scope,
                min_ns: 1,
                time_window: Some(TimeWindow::parse("@105ms-@109ms")?),
                limit: 100,
                ..Default::default()
            },
        )?;
        assert_eq!(r.total_matched, 1, "scope {scope:?}: {r:?}");
        let gap = r
            .rows
            .first()
            .ok_or_else(|| anyhow::anyhow!("scope {scope:?} returned no gap rows"))?;
        assert_eq!(gap.start_ns, 104_500_000);
        assert_eq!(gap.end_ns, 110_000_000);
        assert_eq!(gap.duration_ns, 5_500_000);
    }
    Ok(())
}

#[test]
fn stream_time_window_preserves_cross_window_gap() -> Result<()> {
    // Same cross-window gap as the unified-scope test, but under
    // per-stream LEAD semantics. The local window input must keep
    // the bracketing frontier events for the requested stream.
    let trace = fixture::minimal_gpu()?;
    let r = run(
        trace.path(),
        GapsRequest {
            scope: GapScope::Stream,
            device: Some(0),
            stream: Some(7),
            min_ns: 1,
            time_window: Some(TimeWindow::parse("@105ms-@109ms")?),
            limit: 100,
            ..Default::default()
        },
    )?;

    assert_eq!(r.total_matched, 1, "{r:?}");
    let gap = r
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("stream scope returned no gap rows"))?;
    assert_eq!(gap.key, "gap|dev:0|stream:7|@104500000");
    assert_eq!(gap.device_id, Some(0));
    assert_eq!(gap.stream_id, Some(7));
    assert_eq!(gap.start_ns, 104_500_000);
    assert_eq!(gap.end_ns, 110_000_000);
    assert_eq!(gap.duration_ns, 5_500_000);
    Ok(())
}

#[test]
fn warm_gpu_work_sidecar_preserves_windowed_gap_rows() -> Result<()> {
    let trace = fixture::minimal_gpu()?;
    let req = GapsRequest {
        scope: GapScope::Stream,
        device: Some(0),
        stream: Some(7),
        min_ns: 1,
        time_window: Some(TimeWindow::parse("@105ms-@109ms")?),
        limit: 100,
        ..Default::default()
    };

    let cold = run(trace.path(), req.clone())?;
    {
        let handle = veloq_nsys_data::Trace::open(trace.path())?;
        veloq_nsys_data::gpu_work_events::ensure_sidecar(&handle)?;
    }
    let warm = run(trace.path(), req)?;

    assert_eq!(warm.total_matched, cold.total_matched);
    assert_eq!(warm.rows.len(), cold.rows.len());
    let cold_gap = cold
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("cold query returned no rows"))?;
    let warm_gap = warm
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("warm query returned no rows"))?;
    assert_eq!(warm_gap.key, cold_gap.key);
    assert_eq!(warm_gap.start_ns, cold_gap.start_ns);
    assert_eq!(warm_gap.end_ns, cold_gap.end_ns);
    assert_eq!(warm_gap.duration_ns, cold_gap.duration_ns);
    assert_eq!(warm_gap.prev.row_id, cold_gap.prev.row_id);
    assert_eq!(warm_gap.prev.name, cold_gap.prev.name);
    assert_eq!(warm_gap.next.row_id, cold_gap.next.row_id);
    assert_eq!(warm_gap.next.name, cold_gap.next.name);
    // Pin the concrete resolved labels, not just cold == warm: a
    // both-paths-empty name regression would otherwise satisfy the
    // equality checks above vacuously. prev is the H2D memcpy ending at
    // 104.5ms; next is the slow kernel starting at 110ms.
    assert_eq!(cold_gap.prev.name, "cudaMemcpyHostToDevice");
    assert_eq!(cold_gap.next.name, "slow_kernel");
    Ok(())
}

#[test]
fn graph_trace_rows_count_as_busy_in_cold_and_warm_gaps() -> Result<()> {
    let trace = fixture::with_graph_trace()?;
    let req = GapsRequest {
        min_ns: 1,
        device: Some(0),
        time_window: Some(TimeWindow::parse("@90ms-@205ms")?),
        sort: Some(SortSpec::parse("start")?),
        limit: 100,
        ..Default::default()
    };

    let cold = run(trace.path(), req.clone())?;
    {
        let handle = veloq_nsys_data::Trace::open(trace.path())?;
        veloq_nsys_data::gpu_work_events::ensure_sidecar(&handle)?;
    }
    let warm = run(trace.path(), req)?;

    assert_eq!(cold.total_matched, 3, "cold rows: {:?}", cold.rows);
    assert_eq!(warm.total_matched, cold.total_matched);
    assert_eq!(warm.rows.len(), cold.rows.len());

    let starts: Vec<i64> = cold.rows.iter().map(|gap| gap.start_ns).collect();
    let ends: Vec<i64> = cold.rows.iter().map(|gap| gap.end_ns).collect();
    assert_eq!(starts, vec![110_000_000, 130_000_000, 150_000_000]);
    assert_eq!(ends, vec![120_000_000, 140_000_000, 200_000_000]);

    let kinds: Vec<(EventKind, EventKind)> = cold
        .rows
        .iter()
        .map(|gap| (gap.prev.row_id.kind, gap.next.row_id.kind))
        .collect();
    assert_eq!(
        kinds,
        vec![
            (EventKind::Graph, EventKind::Graph),
            (EventKind::Graph, EventKind::Graph),
            (EventKind::Graph, EventKind::Kernel),
        ]
    );

    // Pin the concrete resolved neighbor names so a both-paths-empty
    // hydration regression can't pass vacuously: graph rows resolve via
    // `'graph:' || graphId` (graphId=42), the eager kernel via its
    // demangled name.
    let names: Vec<(&str, &str)> = cold
        .rows
        .iter()
        .map(|gap| (gap.prev.name.as_str(), gap.next.name.as_str()))
        .collect();
    assert_eq!(
        names,
        vec![
            ("graph:42", "graph:42"),
            ("graph:42", "graph:42"),
            ("graph:42", "eager_kernel"),
        ],
        "cold rows: {:?}",
        cold.rows
    );

    for (cold_gap, warm_gap) in cold.rows.iter().zip(&warm.rows) {
        assert_eq!(warm_gap.key, cold_gap.key);
        assert_eq!(warm_gap.start_ns, cold_gap.start_ns);
        assert_eq!(warm_gap.end_ns, cold_gap.end_ns);
        assert_eq!(warm_gap.duration_ns, cold_gap.duration_ns);
        assert_eq!(warm_gap.prev.row_id, cold_gap.prev.row_id);
        assert_eq!(warm_gap.prev.name, cold_gap.prev.name);
        assert_eq!(warm_gap.next.row_id, cold_gap.next.row_id);
        assert_eq!(warm_gap.next.name, cold_gap.next.name);
    }
    Ok(())
}

#[test]
fn total_matched_is_exact_when_limit_truncates_rows() -> Result<()> {
    // minimal_gpu's stream 7 has six positive gaps. An unlimited query
    // returns them all; a small `limit` truncates the returned rows and
    // forces the separate COUNT(*) `total_sql` path (the LIMIT+1 probe
    // sees more rows than `limit`). Both must report the identical exact
    // `total_matched` — the rows query and the count query are built
    // from independent event-input CTEs and must stay in lock-step.
    let trace = fixture::minimal_gpu()?;
    let base = || GapsRequest {
        scope: GapScope::Stream,
        device: Some(0),
        stream: Some(7),
        min_ns: 1,
        limit: 100,
        ..Default::default()
    };

    let full = run(trace.path(), base())?;
    assert_eq!(full.rows.len(), 6, "unlimited query returns every gap row");
    assert_eq!(full.total_matched, 6, "unlimited total counts every gap");

    let truncated = run(trace.path(), GapsRequest { limit: 2, ..base() })?;
    assert_eq!(truncated.rows.len(), 2, "limit must cap returned rows");
    assert_eq!(
        truncated.total_matched, full.total_matched,
        "limit+1 truncation must still report the exact unlimited total"
    );
    Ok(())
}

#[test]
fn trace_scope_omits_both_axes() -> Result<()> {
    let trace = fixture::minimal_gpu()?;
    let r = run(
        trace.path(),
        GapsRequest {
            scope: GapScope::Trace,
            min_ns: 1,
            limit: 100,
            ..Default::default()
        },
    )?;
    assert_eq!(r.scope, "trace");
    if let Some(g) = r.rows.first() {
        assert!(g.device_id.is_none(), "trace-scope Gap omits device_id");
        assert!(g.stream_id.is_none(), "trace-scope Gap omits stream_id");
        assert!(
            g.key.starts_with("gap|@"),
            "trace-scope key has no axis prefix; got: {}",
            g.key
        );
    }
    Ok(())
}
