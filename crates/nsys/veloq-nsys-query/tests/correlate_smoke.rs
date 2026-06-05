//! Integration tests against `correlate::run` on synthetic fixtures.
//!
//! Specifically exercises the CUDA-graph case: one runtime call
//! `cudaGraphLaunch` correlates to many kernels via a shared
//! `correlationId`. The batched-rowid hydration in
//! `correlate::fetch_summaries` must (a) return them all and (b) not
//! blow up under the placeholder limit when N is large.

mod fixture;

use anyhow::{Result, anyhow};
use veloq_nsys_query::{EventKind, EventRef, RowId};

#[test]
fn correlate_runtime_returns_all_graph_kernels() -> Result<()> {
    let trace = fixture::cuda_graph(50)?;
    let runtime_row = RowId::new(EventKind::Runtime, 1);
    let r = veloq_nsys_query::correlate::run(trace.path(), &[runtime_row])?;
    assert_eq!(r.rows.len(), 1);
    let res = r
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected one result"))?;
    assert!(res.correlation_found, "correlation should resolve");
    assert_eq!(
        res.auxiliary.gpu_events.len(),
        50,
        "all graph kernels surfaced"
    );
    assert_eq!(
        res.auxiliary.cpu_events.len(),
        1,
        "the runtime itself comes back"
    );
    Ok(())
}

/// Wire-format parity with `search`: correlate's GPU kernel rows
/// MUST carry the same per-kind headline payload (grid / block /
/// registers / shared / demangled / mangled) as `search.rows[]`.
/// Without this regression test, a future refactor could quietly
/// revert correlate to base-only and the asymmetry would only
/// surface at agent runtime.
#[test]
fn correlate_kernel_rows_carry_per_kind_headlines() -> Result<()> {
    let trace = fixture::cuda_graph(3)?;
    let runtime_row = RowId::new(EventKind::Runtime, 1);
    let r = veloq_nsys_query::correlate::run(trace.path(), &[runtime_row])?;
    let res = r
        .rows
        .first()
        .ok_or_else(|| anyhow!("expected one result"))?;
    // Every gpu_event under this correlation is a kernel in this
    // fixture — assert each carries grid + block populated.
    for ev in &res.auxiliary.gpu_events {
        match ev {
            EventRef::Kernel(k) => {
                assert!(
                    k.grid.is_some(),
                    "kernel row should carry grid via per-kind headline"
                );
                assert!(k.block.is_some(), "kernel row should carry block");
            }
            other => {
                anyhow::bail!("expected Kernel variant in correlate gpu_events, got {other:?}")
            }
        }
    }
    Ok(())
}

/// Stress the chunked `IN (?, ?, ...)` path: 2500 kernels = 3 chunks
/// at `ROWID_BATCH = 1024`. Asserts correctness, not performance — the
/// batching payoff (2500 prepares collapse to 3) shows up in agent
/// latency, not unit-test wall-clock.
#[test]
fn correlate_runtime_with_giant_graph_group() -> Result<()> {
    let trace = fixture::cuda_graph(2500)?;
    let runtime_row = RowId::new(EventKind::Runtime, 1);
    let r = veloq_nsys_query::correlate::run(trace.path(), &[runtime_row])?;
    let res = r
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected one result"))?;
    assert!(res.correlation_found);
    assert_eq!(res.auxiliary.gpu_events.len(), 2500);
    Ok(())
}

/// Pin overhead → kernel correlation via the shared correlationId
/// bridge. Row 3 of `with_sync` carries correlationId=900 paired
/// with kernel row 1.
#[test]
fn correlate_overhead_with_real_correlation_finds_paired_kernel() -> Result<()> {
    let trace = fixture::with_sync()?;
    let overhead_row = RowId::new(EventKind::Overhead, 3);
    let r = veloq_nsys_query::correlate::run(trace.path(), &[overhead_row])?;
    let res = r
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected one result"))?;
    assert!(
        res.correlation_found,
        "overhead with correlationId=900 must correlate to the paired kernel; \
         pre-fix the column probe silently returned an empty set and routed \
         this to not-found"
    );
    assert!(
        !res.auxiliary.gpu_events.is_empty(),
        "the paired kernel must surface in gpu_events"
    );
    Ok(())
}

/// Overhead with NULL correlationId returns a clean not-found result
/// rather than a SQL prepare error — the path is routed through a
/// runtime-style bridge that short-circuits on NULL correlations and
/// doesn't depend on CUPTI_ACTIVITY_KIND_OVERHEAD carrying
/// `deviceId`/`contextId` (which the standard schema omits).
#[test]
fn correlate_overhead_handles_missing_device_context_columns() -> Result<()> {
    let trace = fixture::with_sync()?;
    let overhead_row = RowId::new(EventKind::Overhead, 1);
    let r = veloq_nsys_query::correlate::run(trace.path(), &[overhead_row])?;
    let res = r
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected one result"))?;
    assert!(
        !res.correlation_found,
        "overhead with NULL correlationId returns not-found, not an error"
    );
    Ok(())
}

/// (`cudaStreamSynchronize`, rowid=2 in the SYNCHRONIZATION table) sharing
/// `correlationId=900` with kernel rowid=1. Inspecting the sync row should
/// surface that kernel in `gpu_events` and the sync itself in `sync_events`.
#[test]
fn correlate_sync_to_kernel_via_shared_correlation() -> Result<()> {
    let trace = fixture::with_sync()?;
    // Sync row #2 carries syncType=3 (cudaStreamSynchronize) and the
    // shared correlationId.
    let sync_row = RowId::new(EventKind::Sync, 2);
    let r = veloq_nsys_query::correlate::run(trace.path(), &[sync_row])?;
    let res = r
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected one result"))?;
    assert!(
        res.correlation_found,
        "sync→kernel correlation should resolve"
    );
    // The kernel paired by correlationId surfaces on the GPU side.
    assert_eq!(
        res.auxiliary.gpu_events.len(),
        1,
        "the paired kernel surfaces"
    );
    // The originating sync row appears on the sync side (correlate walks
    // the index, not the input row, so the sync row itself rejoins via
    // its own correlationId bucket).
    assert_eq!(
        res.auxiliary.sync_events.len(),
        1,
        "the sync itself comes back"
    );
    Ok(())
}
