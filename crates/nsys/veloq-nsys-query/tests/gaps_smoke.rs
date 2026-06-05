//! Integration tests against `gaps::run` on synthetic fixtures.
//!
//! Covers the three scopes (`device` default / `stream` / `trace`),
//! the upstream rejection of meaningless flag combinations, and the
//! `auxiliary.streams[]` busy-ratio surface that stays useful across
//! every scope.

mod fixture;

use anyhow::Result;
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
