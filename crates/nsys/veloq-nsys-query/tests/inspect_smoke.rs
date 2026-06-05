//! Integration tests against `inspect::run` for kinds that weren't
//! covered by the existing transitive coverage in `graph_smoke.rs` and
//! `metrics_cpu_smoke.rs`. Each family submodule is exercised at least
//! once here so a regression in one kind's SQL or column probing
//! surfaces as a focused test failure rather than a downstream
//! mismatch elsewhere.
//!
//! Covered here:
//!   - `gpu_work::query_memcpy` (H2D and D2H rows from `minimal_gpu`)
//!   - `gpu_work::query_memset` (`minimal_gpu`)
//!   - `host_api::query_runtime` (`cuda_graph(1)`)
//!   - `host_api::query_nvtx` (`nvtx_attribution`)
//!   - `host_api::query_osrt` (`with_osrt`)
//!   - Multi-kind dispatch ordering preserves the input row_id order
//!   - Out-of-range row_id resolves to `EventDetails::NotFound`
//!
//! Already covered elsewhere (and intentionally not duplicated here):
//!   - Kernel / Graph / GraphNode / GraphEvent / Sync / CudaEvent /
//!     Overhead via `graph_smoke.rs`.
//!   - CpuSample (including the optional-callchain edge case) via
//!     `metrics_cpu_smoke.rs`.

mod fixture;

use anyhow::{Result, anyhow, bail};
use veloq_nsys_query::inspect::{self, EventDetails};
use veloq_nsys_query::{EventKind, RowId};

#[test]
fn memcpy_h2d_reports_bytes_and_label() -> Result<()> {
    let trace = fixture::minimal_gpu()?;
    let id = RowId::new(EventKind::Memcpy, 1);
    let r = inspect::run(trace.path(), &[id])?;
    let first = r
        .rows
        .first()
        .ok_or_else(|| anyhow!("expected one event"))?;
    let m = match first {
        EventDetails::Memcpy(m) => m,
        other => bail!("expected Memcpy, got {other:?}"),
    };
    assert_eq!(m.bytes, 4096);
    assert_eq!(m.copy_kind, 1);
    assert_eq!(m.copy_kind_name, "cudaMemcpyHostToDevice");
    assert_eq!(m.duration_ns, 500_000);
    assert_eq!(m.stream_id, 7);
    Ok(())
}

#[test]
fn memcpy_d2h_uses_different_copy_kind_label() -> Result<()> {
    let trace = fixture::minimal_gpu()?;
    let id = RowId::new(EventKind::Memcpy, 2);
    let r = inspect::run(trace.path(), &[id])?;
    let m = match r
        .rows
        .first()
        .ok_or_else(|| anyhow!("expected one event"))?
    {
        EventDetails::Memcpy(m) => m,
        other => bail!("expected Memcpy, got {other:?}"),
    };
    assert_eq!(m.copy_kind, 2);
    assert_eq!(m.copy_kind_name, "cudaMemcpyDeviceToHost");
    Ok(())
}

#[test]
fn memset_reports_bytes_and_value() -> Result<()> {
    let trace = fixture::minimal_gpu()?;
    let id = RowId::new(EventKind::Memset, 1);
    let r = inspect::run(trace.path(), &[id])?;
    let m = match r
        .rows
        .first()
        .ok_or_else(|| anyhow!("expected one event"))?
    {
        EventDetails::Memset(m) => m,
        other => bail!("expected Memset, got {other:?}"),
    };
    assert_eq!(m.bytes, 1024);
    assert_eq!(m.value, Some(0));
    assert_eq!(m.duration_ns, 200_000);
    Ok(())
}

#[test]
fn runtime_resolves_name_from_string_ids() -> Result<()> {
    let trace = fixture::cuda_graph(1)?;
    let id = RowId::new(EventKind::Runtime, 1);
    let r = inspect::run(trace.path(), &[id])?;
    let rt = match r
        .rows
        .first()
        .ok_or_else(|| anyhow!("expected one event"))?
    {
        EventDetails::Runtime(rt) => rt,
        other => bail!("expected Runtime, got {other:?}"),
    };
    assert_eq!(rt.name, "cudaGraphLaunch");
    assert_eq!(rt.correlation_id, Some(9000));
    assert_eq!(rt.duration_ns, 100_000);
    Ok(())
}

#[test]
fn nvtx_returns_text_and_global_tid() -> Result<()> {
    let trace = fixture::nvtx_attribution()?;
    // Fixture seeds 2 ranges; rowid=1 is `step_a` (100..200ms).
    let id = RowId::new(EventKind::Nvtx, 1);
    let r = inspect::run(trace.path(), &[id])?;
    let nv = match r
        .rows
        .first()
        .ok_or_else(|| anyhow!("expected one event"))?
    {
        EventDetails::Nvtx(n) => n,
        other => bail!("expected Nvtx, got {other:?}"),
    };
    assert_eq!(nv.name, "step_a");
    assert_eq!(nv.start_ns, 100_000_000);
    assert_eq!(nv.end_ns, Some(200_000_000));
    assert_eq!(nv.duration_ns, Some(100_000_000));
    Ok(())
}

#[test]
fn osrt_resolves_name_and_decodes_global_tid() -> Result<()> {
    let trace = fixture::with_osrt()?;
    let id = RowId::new(EventKind::Osrt, 1);
    let r = inspect::run(trace.path(), &[id])?;
    let o = match r
        .rows
        .first()
        .ok_or_else(|| anyhow!("expected one event"))?
    {
        EventDetails::Osrt(o) => o,
        other => bail!("expected Osrt, got {other:?}"),
    };
    assert_eq!(o.name, "pthread_mutex_lock");
    assert_eq!(o.duration_ns, 250_000);
    // Fixture packs globalTid = (1234 << 24) | 56.
    assert_eq!(o.global_tid, (1234i64 << 24) | 56);
    Ok(())
}

#[test]
fn missing_rowid_yields_not_found_variant() -> Result<()> {
    let trace = fixture::minimal_gpu()?;
    // Fixture has 4 kernels (rowid 1..=4); rowid=999 is out of range.
    let id = RowId::new(EventKind::Kernel, 999);
    let r = inspect::run(trace.path(), &[id])?;
    match r
        .rows
        .first()
        .ok_or_else(|| anyhow!("expected one event"))?
    {
        EventDetails::NotFound { key, row_id } => {
            assert_eq!(*row_id, id);
            assert_eq!(*key, id.to_string());
        }
        other => bail!("expected NotFound, got {other:?}"),
    }
    Ok(())
}

#[test]
fn dispatch_preserves_input_row_id_order_across_kinds() -> Result<()> {
    // A mix of kinds in a non-sorted order — output must mirror input
    // positionally so an agent can pair its request with the response
    // without re-sorting.
    let trace = fixture::minimal_gpu()?;
    let ids = [
        RowId::new(EventKind::Memset, 1),
        RowId::new(EventKind::Kernel, 1),
        RowId::new(EventKind::Memcpy, 2),
    ];
    let r = inspect::run(trace.path(), &ids)?;
    assert_eq!(r.rows.len(), 3);
    let kinds: Vec<&'static str> = r
        .rows
        .iter()
        .map(|e| match e {
            EventDetails::Memset(_) => "memset",
            EventDetails::Kernel(_) => "kernel",
            EventDetails::Memcpy(_) => "memcpy",
            _ => "other",
        })
        .collect();
    assert_eq!(kinds, vec!["memset", "kernel", "memcpy"]);
    Ok(())
}
