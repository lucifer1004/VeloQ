//! Shared kind-policy tests:
//! - Null-location filter policy (`--device` / `--stream` reject
//!   explicit null-location kinds; `KindFilter::All` narrows
//!   implicitly).
//! - `--group-by device/stream/context/graph/graph_node` set rule
//!   (errors only when EVERY explicit kind lacks the axis).
//! - `--nvtx` policy (explicit non-attributable kinds error;
//!   mixed-attributable sets pass).
//!
//! The fixtures here are intentionally small — most tests exercise
//! request-validation paths that bail before any SQL runs, so a
//! bare host_api fixture is enough.

mod fixture;

use anyhow::{Result, bail};
use veloq_nsys_query::stats::{GroupBy, StatsRequest};
use veloq_nsys_query::{EventKind, KindFilter, search::SearchRequest};

fn group_by_device() -> GroupBy {
    GroupBy {
        device: true,
        ..Default::default()
    }
}

// ---------- null-location filter policy --------------------------------------

#[test]
fn stats_explicit_runtime_with_device_errors() -> Result<()> {
    let trace = fixture::host_api()?;
    let r = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Runtime]),
            device: Some(0),
            ..Default::default()
        },
    );
    let msg = match r {
        Ok(_) => bail!("--type runtime --device 0 should error"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("--device") && msg.contains("runtime"),
        "error should mention --device + runtime; got `{msg}`"
    );
    Ok(())
}

#[test]
fn stats_default_all_with_device_is_positive() -> Result<()> {
    // `--device 0` with no explicit `--type` narrows implicitly to
    // location-bearing kinds and must not bail.
    let trace = fixture::minimal_gpu()?;
    let r = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            device: Some(0),
            ..Default::default()
        },
    )?;
    assert!(r.total_matched >= 1, "expected at least one row");
    Ok(())
}

#[test]
fn stats_mixed_kernel_runtime_with_device_errors() -> Result<()> {
    let trace = fixture::host_api()?;
    let r = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel, EventKind::Runtime]),
            device: Some(0),
            ..Default::default()
        },
    );
    assert!(
        r.is_err(),
        "mixed kernel,runtime + --device should error (runtime has no device)"
    );
    Ok(())
}

#[test]
fn stats_kernel_nvtx_with_device_errors() -> Result<()> {
    // Mixed-existing trap: kernel has device, nvtx doesn't. The
    // policy fires on any explicit null-location kind.
    let trace = fixture::nvtx_attribution()?;
    let r = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel, EventKind::Nvtx]),
            device: Some(0),
            ..Default::default()
        },
    );
    assert!(r.is_err(), "--type kernel,nvtx --device 0 must error");
    Ok(())
}

#[test]
fn stats_kernel_nvtx_with_stream_errors() -> Result<()> {
    let trace = fixture::nvtx_attribution()?;
    let r = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel, EventKind::Nvtx]),
            stream: Some(7),
            ..Default::default()
        },
    );
    assert!(r.is_err(), "--type kernel,nvtx --stream 7 must error");
    Ok(())
}

#[test]
fn search_explicit_nvtx_with_stream_errors() -> Result<()> {
    let trace = fixture::nvtx_nested()?;
    let r = veloq_nsys_query::search::run(
        trace.path(),
        SearchRequest {
            kinds: KindFilter::Only(vec![EventKind::Nvtx]),
            stream: Some(7),
            ..Default::default()
        },
    );
    assert!(r.is_err(), "--type nvtx --stream 7 must error");
    Ok(())
}

#[test]
fn search_overhead_stream_filter_errors() -> Result<()> {
    let trace = fixture::nvtx_attribution()?;
    let r = veloq_nsys_query::search::run(
        trace.path(),
        SearchRequest {
            kinds: KindFilter::Only(vec![EventKind::Overhead]),
            stream: Some(7),
            ..Default::default()
        },
    );
    assert!(
        r.is_err(),
        "--type overhead --stream 7 must error (overhead is null-location)"
    );
    Ok(())
}

#[test]
fn search_graph_node_device_filter_errors() -> Result<()> {
    let trace = fixture::nvtx_attribution()?;
    let r = veloq_nsys_query::search::run(
        trace.path(),
        SearchRequest {
            kinds: KindFilter::Only(vec![EventKind::GraphNode]),
            device: Some(0),
            ..Default::default()
        },
    );
    assert!(
        r.is_err(),
        "--type graph_node --device 0 must error (graph_node is null-location)"
    );
    Ok(())
}

// ---------- --group-by location-axis set rule -------------------------------

#[test]
fn stats_runtime_only_group_by_device_errors() -> Result<()> {
    let trace = fixture::host_api()?;
    let r = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Runtime]),
            group_by: group_by_device(),
            ..Default::default()
        },
    );
    assert!(
        r.is_err(),
        "--type runtime --group-by device must error (no device axis on runtime)"
    );
    Ok(())
}

#[test]
fn stats_runtime_osrt_group_by_device_errors() -> Result<()> {
    let trace = fixture::host_api()?;
    let r = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Runtime, EventKind::Osrt]),
            group_by: group_by_device(),
            ..Default::default()
        },
    );
    assert!(
        r.is_err(),
        "--type runtime,osrt --group-by device must error"
    );
    Ok(())
}

#[test]
fn stats_runtime_nvtx_group_by_device_errors() -> Result<()> {
    let trace = fixture::nvtx_attribution()?;
    let r = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Runtime, EventKind::Nvtx]),
            group_by: group_by_device(),
            ..Default::default()
        },
    );
    assert!(
        r.is_err(),
        "--type runtime,nvtx --group-by device must error"
    );
    Ok(())
}

#[test]
fn stats_kernel_runtime_group_by_device_positive() -> Result<()> {
    // Mixed-with-GPU: kernel HAS the device axis, so the set rule
    // does NOT fire even though runtime doesn't. Regression-fix
    // the WI explicitly calls out.
    let trace = fixture::nvtx_attribution()?;
    let r = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel, EventKind::Runtime]),
            group_by: group_by_device(),
            ..Default::default()
        },
    )?;
    assert!(
        r.total_matched >= 1,
        "mixed kernel,runtime --group-by device must succeed"
    );
    Ok(())
}

#[test]
fn stats_kernel_nvtx_group_by_device_positive() -> Result<()> {
    let trace = fixture::nvtx_attribution()?;
    let r = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel, EventKind::Nvtx]),
            group_by: group_by_device(),
            ..Default::default()
        },
    )?;
    assert!(
        r.total_matched >= 1,
        "mixed kernel,nvtx --group-by device must succeed (kernel splits, nvtx single-bucket)"
    );
    Ok(())
}

// ---------- --nvtx policy ----------------------------------------------------

#[test]
fn stats_sync_nvtx_scopes_via_attributed_sync_rowids() -> Result<()> {
    // Sync rows attribute via the (correlationId + ctx_for_pid) join
    // through the runtime call inside the NVTX range. The
    // nvtx_parent_attribution fixture has 3 sync rows: one inside
    // step_a, one inside step_b, one outside both ranges. `--nvtx
    // step_a` must narrow the scope to exactly the inside-step_a row.
    let trace = fixture::nvtx_parent_attribution()?;
    let unscoped = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Sync]),
            ..Default::default()
        },
    )?;
    let scoped = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Sync]),
            nvtx: Some("step_a".into()),
            ..Default::default()
        },
    )?;
    assert_eq!(unscoped.total_events, 3, "fixture seeds 3 sync rows");
    assert_eq!(
        scoped.total_events, 1,
        "--nvtx step_a must scope sync to exactly the inside-step_a row"
    );
    assert_eq!(
        scoped.nvtx_scope.as_deref(),
        Some("step_a"),
        "response must echo the user's pattern"
    );
    Ok(())
}

#[test]
fn stats_explicit_nvtx_kind_with_nvtx_pattern_errors() -> Result<()> {
    let trace = fixture::nvtx_attribution()?;
    let r = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Nvtx]),
            nvtx: Some("*".into()),
            ..Default::default()
        },
    );
    let msg = match r {
        Ok(_) => bail!("--type nvtx --nvtx '*' should error"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("nvtx") && msg.contains("experimental"),
        "error should mention experimental opt-in framing; got `{msg}`"
    );
    Ok(())
}

#[test]
fn stats_default_all_with_nvtx_pattern_positive() -> Result<()> {
    let trace = fixture::nvtx_attribution()?;
    let r = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            nvtx: Some("*".into()),
            ..Default::default()
        },
    )?;
    // KindFilter::All narrows implicitly to attributable set; the
    // fixture has 2 kernels matching → ≥1 row.
    assert!(r.total_matched >= 1);
    Ok(())
}

#[test]
fn stats_runtime_kernel_nvtx_positive() -> Result<()> {
    // Explicit mix of attributable kinds + --nvtx → policy allows.
    let trace = fixture::nvtx_attribution()?;
    let r = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Runtime, EventKind::Kernel]),
            nvtx: Some("*".into()),
            ..Default::default()
        },
    )?;
    assert!(
        r.total_matched >= 1,
        "runtime+kernel both attributable; --nvtx must succeed"
    );
    Ok(())
}

#[test]
fn stats_kernel_sync_nvtx_positive() -> Result<()> {
    // Both attributable; even with no sync rows in this fixture,
    // the bail must not fire.
    let trace = fixture::nvtx_attribution()?;
    let _r = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel, EventKind::Sync]),
            nvtx: Some("*".into()),
            ..Default::default()
        },
    )?;
    Ok(())
}

#[test]
fn stats_osrt_nvtx_errors() -> Result<()> {
    let trace = fixture::host_api()?;
    let r = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Osrt]),
            nvtx: Some("*".into()),
            ..Default::default()
        },
    );
    let msg = match r {
        Ok(_) => bail!("--type osrt --nvtx must error (osrt is non-attributable)"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("osrt") && msg.contains("experimental"),
        "osrt --nvtx error must use the experimental wording; got `{msg}`"
    );
    Ok(())
}

#[test]
fn stats_runtime_osrt_nvtx_errors() -> Result<()> {
    // Mixed: runtime IS attributable, osrt ISN'T. Policy errors
    // because the set contains a non-attributable kind.
    let trace = fixture::host_api()?;
    let r = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Runtime, EventKind::Osrt]),
            nvtx: Some("*".into()),
            ..Default::default()
        },
    );
    assert!(
        r.is_err(),
        "--type runtime,osrt --nvtx must error (osrt unattributable taints the set)"
    );
    Ok(())
}
