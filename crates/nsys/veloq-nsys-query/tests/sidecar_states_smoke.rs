//! Coverage for the three NVTX-parent sidecar states every NVTX-bearing
//! verb has to handle:
//!
//! 1. **Bypass** — small reverse-lookup batch (≤ `SIDECAR_BUILD_THRESHOLD`)
//!    AND no sidecar on disk → `nvtx_reverse::cold_fallback` SQL path
//!    runs, sidecar is NOT built.
//! 2. **Cold** — large reverse-lookup batch OR forward-attribution call
//!    OR any sidecar-required verb (slices, stats --group-by
//!    nvtx-parent, …) AND no sidecar on disk → sidecar is built once,
//!    then in-memory / parquet lookup serves the call.
//! 3. **Warm** — sidecar already on disk → loaded (or `read_parquet`-
//!    scanned) without a rebuild.
//!
//! Each test asserts the observable side effect (sidecar file present
//! and unchanged mtime) so a regression that silently rebuilds or
//! skips the path would fail loudly.
//!
//! The fixtures use tempdir-backed `_pqtdir/` directories — the
//! sidecar lands under the trace's artifact root as
//! `<trace>.veloq/nvtx-parent.parquet`. We delete it between cases to
//! sequence states deterministically.

mod fixture;

use anyhow::{Context, Result, anyhow};
use std::fs;
use std::path::PathBuf;
use std::thread::sleep;
use std::time::Duration;
use veloq_nsys_data::{Trace, runtime_nvtx_parent};
use veloq_nsys_query::inspect::EventDetails;
use veloq_nsys_query::search::SearchRequest;
use veloq_nsys_query::stats::{GroupBy, StatsRequest};
use veloq_nsys_query::{EventKind, KindFilter, RowId};

fn sidecar_path(trace_path: &std::path::Path) -> PathBuf {
    runtime_nvtx_parent::sidecar_path_for(trace_path)
}

fn delete_sidecar(trace_path: &std::path::Path) -> Result<()> {
    let p = sidecar_path(trace_path);
    if p.exists() {
        fs::remove_file(&p).with_context(|| format!("deleting {}", p.display()))?;
    }
    Ok(())
}

// ============================================================================
// REVERSE DIRECTION — inspect / search --with-nvtx
// ============================================================================

/// **Bypass state**: `inspect kernel:N` on a single rowid never
/// triggers a sidecar build. Cold latency stays low; the user pays
/// only the rowid-scoped SQL CTE.
#[test]
fn reverse_bypass_state_inspect_single_row_does_not_build_sidecar() -> Result<()> {
    let trace = fixture::nvtx_attribution()?;
    delete_sidecar(trace.path())?;
    assert!(
        !sidecar_path(trace.path()).exists(),
        "precondition: sidecar must be absent"
    );

    let resp = veloq_nsys_query::inspect::run(trace.path(), &[RowId::new(EventKind::Kernel, 1)])?;
    assert_eq!(resp.rows.len(), 1);
    let EventDetails::Kernel(k) = resp
        .rows
        .first()
        .ok_or_else(|| anyhow!("no rows returned"))?
    else {
        return Err(anyhow!("expected Kernel"));
    };
    let ctx = k
        .nvtx_context
        .as_ref()
        .ok_or_else(|| anyhow!("kernel:1 missing nvtx_context"))?;
    assert!(ctx.name == "step_a" || ctx.name == "step_b");

    // The bypass contract — sidecar STILL absent after a single-row
    // inspect on a cold trace.
    assert!(
        !sidecar_path(trace.path()).exists(),
        "single-row inspect must not trigger sidecar build (cold latency stays low)"
    );
    Ok(())
}

/// **Cold state**: `search --with-nvtx` on a cold trace returning
/// ≥ SIDECAR_BUILD_THRESHOLD hits triggers a sidecar build via the
/// actual public verb (not by calling `lookup_for_row_ids` with
/// padding). Subsequent calls would be warm.
#[test]
fn reverse_cold_state_search_with_nvtx_builds_sidecar() -> Result<()> {
    // The shared `nvtx_attribution` fixture has only 2 kernels —
    // too few to exceed the build threshold. Use a beefier
    // local fixture (8 kernels, all inside an NVTX range) so
    // `search --with-nvtx` actually crosses the threshold via real
    // hits rather than synthetic padding.
    let trace = fixture::many_kernels_in_nvtx(8)?;
    let trace_path = trace.path();
    delete_sidecar(trace_path)?;
    assert!(
        !sidecar_path(trace_path).exists(),
        "precondition: sidecar must be absent"
    );

    let resp = veloq_nsys_query::search::run(
        trace_path,
        SearchRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            with_nvtx: true,
            limit: 50,
            ..Default::default()
        },
    )?;
    // All 8 kernels matched and each one decorated with NVTX context.
    assert_eq!(resp.rows.len(), 8, "expected all 8 kernel hits");
    let decorated = resp
        .rows
        .iter()
        .filter(|hit| hit.base().nvtx_context.is_some())
        .count();
    assert_eq!(
        decorated, 8,
        "every search --with-nvtx hit must carry nvtx_context"
    );

    // The cold contract — sidecar now exists.
    assert!(
        sidecar_path(trace_path).exists(),
        "search --with-nvtx returning ≥ SIDECAR_BUILD_THRESHOLD hits on cold trace must build sidecar"
    );
    Ok(())
}

/// **Warm state**: with the sidecar pre-built, any reverse lookup —
/// including a single-row `inspect` — must use it (and must NOT
/// rebuild). Mtime invariance is the observable contract.
#[test]
fn reverse_warm_state_load_does_not_rebuild_sidecar() -> Result<()> {
    let trace = fixture::nvtx_attribution()?;
    delete_sidecar(trace.path())?;

    // Pre-build the sidecar.
    let t1 = Trace::open(trace.path())?;
    let path = runtime_nvtx_parent::ensure_sidecar(&t1)?;
    drop(t1);
    let mtime_before = fs::metadata(&path)?.modified()?;

    // Sleep past filesystem mtime resolution so a rebuild would
    // produce a visibly different mtime (some filesystems round to
    // 1s; this is conservative).
    sleep(Duration::from_millis(1100));

    // Single-row inspect (would hit the "load if present" branch).
    let resp = veloq_nsys_query::inspect::run(trace.path(), &[RowId::new(EventKind::Kernel, 1)])?;
    let EventDetails::Kernel(k) = resp
        .rows
        .first()
        .ok_or_else(|| anyhow!("no rows returned"))?
    else {
        return Err(anyhow!("expected Kernel"));
    };
    assert!(k.nvtx_context.is_some());

    // Batch lookup — would hit the "build_or_load_index" branch but
    // should NOT rebuild because the existing sidecar is fresh.
    let trace_handle = Trace::open(trace.path())?;
    let nesting = trace_handle.nvtx_nesting()?;
    let ids = vec![
        RowId::new(EventKind::Kernel, 1),
        RowId::new(EventKind::Kernel, 2),
        RowId::new(EventKind::Kernel, 99),
        RowId::new(EventKind::Kernel, 100),
        RowId::new(EventKind::Kernel, 101),
    ];
    let _ctxs = veloq_nsys_query::nvtx_reverse::lookup_for_row_ids(&trace_handle, &ids, &nesting)?;

    let mtime_after = fs::metadata(&path)?.modified()?;
    assert_eq!(
        mtime_before, mtime_after,
        "warm-state lookups must not rebuild the sidecar"
    );
    Ok(())
}

/// P2 review guard: when two NVTX ranges share a start, the cold
/// fallback SQL must agree with the warm sidecar walk on which is
/// "innermost". A cold single-row `inspect kernel:N` runs the
/// `cold_fallback` SQL path; a warm `search --with-nvtx` runs the
/// sidecar walk. Both must pick the same range as innermost.
#[test]
fn cold_and_warm_agree_on_same_start_innermost() -> Result<()> {
    let trace = fixture::same_start_nested_nvtx()?;
    let trace_path = trace.path();
    delete_sidecar(trace_path)?;
    assert!(!sidecar_path(trace_path).exists());

    // Cold path: single-row inspect → cold_fallback SQL.
    let cold_resp =
        veloq_nsys_query::inspect::run(trace_path, &[RowId::new(EventKind::Kernel, 1)])?;
    let EventDetails::Kernel(k_cold) = cold_resp.rows.first().ok_or_else(|| anyhow!("no row"))?
    else {
        return Err(anyhow!("expected Kernel"));
    };
    let cold_name = k_cold
        .nvtx_context
        .as_ref()
        .ok_or_else(|| anyhow!("cold missing nvtx_context"))?
        .name
        .clone();
    assert!(
        !sidecar_path(trace_path).exists(),
        "single-row inspect must not build sidecar (cold fallback path)"
    );

    // Warm path: pre-build sidecar, repeat inspect.
    let t = Trace::open(trace_path)?;
    runtime_nvtx_parent::ensure_sidecar(&t)?;
    drop(t);
    let warm_resp =
        veloq_nsys_query::inspect::run(trace_path, &[RowId::new(EventKind::Kernel, 1)])?;
    let EventDetails::Kernel(k_warm) = warm_resp.rows.first().ok_or_else(|| anyhow!("no row"))?
    else {
        return Err(anyhow!("expected Kernel"));
    };
    let warm_name = k_warm
        .nvtx_context
        .as_ref()
        .ok_or_else(|| anyhow!("warm missing nvtx_context"))?
        .name
        .clone();

    assert_eq!(
        cold_name, warm_name,
        "cold and warm must agree on innermost for same-start nested ranges \
         (cold picked {cold_name:?}, warm picked {warm_name:?})"
    );
    assert_eq!(
        warm_name, "inner",
        "innermost (tighter end) must win for same-start ranges"
    );
    Ok(())
}

// ============================================================================
// FORWARD DIRECTION — stats / search / timeline --nvtx <pattern>
// ============================================================================

/// **Cold state**: `stats --nvtx <pattern>` on a cold trace triggers
/// the sidecar build (forward attribution always needs the
/// outer→inner chain).
#[test]
fn forward_cold_state_stats_nvtx_builds_sidecar() -> Result<()> {
    let trace = fixture::nvtx_attribution()?;
    delete_sidecar(trace.path())?;

    let req = StatsRequest {
        nvtx: Some("step_*".into()),
        kinds: KindFilter::Only(vec![EventKind::Kernel]),
        ..Default::default()
    };
    let resp = veloq_nsys_query::stats::run(trace.path(), req)?;
    // Two kernels in this fixture, both inside step_a / step_b → both
    // attributed.
    assert!(
        !resp.rows.is_empty(),
        "expected at least one row from --nvtx 'step_*'"
    );

    assert!(
        sidecar_path(trace.path()).exists(),
        "forward --nvtx on cold trace must build sidecar"
    );
    Ok(())
}

/// **Warm state**: with sidecar present, `stats --nvtx` reads it
/// without rebuild.
#[test]
fn forward_warm_state_stats_nvtx_does_not_rebuild() -> Result<()> {
    let trace = fixture::nvtx_attribution()?;
    delete_sidecar(trace.path())?;
    // Pre-build.
    let t1 = Trace::open(trace.path())?;
    let path = runtime_nvtx_parent::ensure_sidecar(&t1)?;
    drop(t1);
    let mtime_before = fs::metadata(&path)?.modified()?;
    sleep(Duration::from_millis(1100));

    let req = StatsRequest {
        nvtx: Some("step_*".into()),
        kinds: KindFilter::Only(vec![EventKind::Kernel]),
        ..Default::default()
    };
    let _resp = veloq_nsys_query::stats::run(trace.path(), req)?;

    let mtime_after = fs::metadata(&path)?.modified()?;
    assert_eq!(
        mtime_before, mtime_after,
        "warm-state forward --nvtx must not rebuild the sidecar"
    );
    Ok(())
}

/// `search --nvtx <pattern>` — sanity that the forward path picks up
/// the same matches whether the sidecar starts cold or warm.
#[test]
fn forward_search_nvtx_cold_and_warm_agree() -> Result<()> {
    let trace = fixture::nvtx_attribution()?;
    delete_sidecar(trace.path())?;

    let req = || SearchRequest {
        nvtx: Some("step_*".into()),
        kinds: KindFilter::Only(vec![EventKind::Kernel]),
        limit: 50,
        ..Default::default()
    };
    let cold = veloq_nsys_query::search::run(trace.path(), req())?;
    assert!(sidecar_path(trace.path()).exists(), "cold call must build");
    // Same call against the now-warm sidecar.
    let warm = veloq_nsys_query::search::run(trace.path(), req())?;
    assert_eq!(
        cold.rows.len(),
        warm.rows.len(),
        "cold and warm forward attribution must agree on hit count"
    );
    Ok(())
}

// ============================================================================
// RUNTIME-ONLY (no TARGET_INFO_CUDA_CONTEXT_INFO)
// ============================================================================

/// A trace with NVTX_EVENTS + CUPTI_ACTIVITY_KIND_RUNTIME but no
/// `TARGET_INFO_CUDA_CONTEXT_INFO` (e.g. CUDA-API-only profile)
/// should still attribute runtime rows via every reverse path.
/// Pre-fix, the global preflight short-circuited and every lookup
/// returned `None` even though the runtime path doesn't need the
/// GPU bridge.
#[test]
fn runtime_only_trace_attributes_via_inspect_and_search() -> Result<()> {
    let trace = fixture::runtime_only_with_null_correlation()?;
    let trace_path = trace.path();
    // Reverse single-row: inspect runtime:N must get nvtx_context.
    let resp = veloq_nsys_query::inspect::run(trace_path, &[RowId::new(EventKind::Runtime, 1)])?;
    let EventDetails::Runtime(r) = resp
        .rows
        .first()
        .ok_or_else(|| anyhow!("no rows returned"))?
    else {
        return Err(anyhow!("expected Runtime variant"));
    };
    let ctx = r
        .nvtx_context
        .as_ref()
        .ok_or_else(|| anyhow!("runtime:1 missing nvtx_context on TARGET_INFO-less trace"))?;
    assert_eq!(ctx.name, "step");

    // Forward: stats --type runtime --nvtx <pattern> must succeed
    // without TARGET_INFO_CUDA_CONTEXT_INFO and emit attributed rows.
    let stats = veloq_nsys_query::stats::run(
        trace_path,
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Runtime]),
            nvtx: Some("step*".into()),
            ..Default::default()
        },
    )?;
    assert!(
        !stats.rows.is_empty(),
        "stats --type runtime --nvtx 'step*' must attribute the runtime row"
    );
    Ok(())
}

/// P1 guard: a runtime row with `correlationId IS NULL` must still
/// be attributable through the sidecar's `by_rt_rowid` lookup. We
/// pre-build the sidecar so the lookup actually consults it — a
/// cold-cache single-row `inspect` would otherwise route through
/// `cold_fallback` SQL, which has its own (separate) NULL handling
/// and wouldn't exercise the sidecar code path the name promises.
#[test]
fn runtime_row_with_null_correlation_id_is_still_attributed() -> Result<()> {
    let trace = fixture::runtime_only_with_null_correlation()?;
    let trace_path = trace.path();
    // Pre-build the sidecar so single-row inspect lands on the
    // warm path (by_rt_rowid), not the cold-fallback SQL.
    let t = Trace::open(trace_path)?;
    let sidecar = runtime_nvtx_parent::ensure_sidecar(&t)?;
    drop(t);
    assert!(
        sidecar.exists(),
        "sidecar must be built before inspect for this test to exercise by_rt_rowid"
    );

    let resp = veloq_nsys_query::inspect::run(trace_path, &[RowId::new(EventKind::Runtime, 1)])?;
    let EventDetails::Runtime(r) = resp
        .rows
        .first()
        .ok_or_else(|| anyhow!("no rows returned"))?
    else {
        return Err(anyhow!("expected Runtime variant"));
    };
    assert!(
        r.nvtx_context.is_some(),
        "runtime row with NULL correlationId must still attribute via by_rt_rowid lookup"
    );

    // Cross-check the in-memory map directly — the row exists under
    // `by_rt_rowid` but NOT under any `by_correlation` slot (since
    // correlation_id is None, the disambiguator trio can't form).
    let t = Trace::open(trace_path)?;
    let idx = runtime_nvtx_parent::build_or_load_index(&t)?;
    assert!(
        idx.get_by_runtime(1).is_some(),
        "by_rt_rowid must hold the NULL-correlation runtime row"
    );
    // No reasonable trio would resolve to this entry — verify a
    // representative probe misses cleanly.
    assert!(
        idx.get_by_correlation(0, 1, 0).is_none(),
        "by_correlation must not surface NULL-correlation entries under any trio"
    );
    Ok(())
}

// ============================================================================
// stats --group-by nvtx-parent
// ============================================================================

/// `stats --group-by nvtx-parent` always builds (or warm-loads) the
/// sidecar — there's no bypass mode for nvtx-parent because the join
/// requires the parquet directly.
#[test]
fn b2_cold_state_group_by_nvtx_parent_builds_sidecar() -> Result<()> {
    let trace = fixture::nvtx_attribution()?;
    delete_sidecar(trace.path())?;

    let req = StatsRequest {
        kinds: KindFilter::Only(vec![EventKind::Kernel]),
        group_by: GroupBy {
            nvtx_parent: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let resp = veloq_nsys_query::stats::run(trace.path(), req)?;
    assert!(!resp.rows.is_empty());
    assert!(sidecar_path(trace.path()).exists());
    Ok(())
}

#[test]
fn b2_warm_state_group_by_nvtx_parent_does_not_rebuild() -> Result<()> {
    let trace = fixture::nvtx_attribution()?;
    delete_sidecar(trace.path())?;
    let t1 = Trace::open(trace.path())?;
    let path = runtime_nvtx_parent::ensure_sidecar(&t1)?;
    drop(t1);
    let mtime_before = fs::metadata(&path)?.modified()?;
    sleep(Duration::from_millis(1100));

    let req = StatsRequest {
        kinds: KindFilter::Only(vec![EventKind::Kernel]),
        group_by: GroupBy {
            nvtx_parent: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let _resp = veloq_nsys_query::stats::run(trace.path(), req)?;

    let mtime_after = fs::metadata(&path)?.modified()?;
    assert_eq!(
        mtime_before, mtime_after,
        "warm-state B2 must not rebuild the sidecar"
    );
    Ok(())
}

// ============================================================================
// SLICES — instance and aggregate views
// ============================================================================

#[test]
fn slices_cold_state_builds_sidecar() -> Result<()> {
    let trace = fixture::nvtx_attribution()?;
    delete_sidecar(trace.path())?;

    let req = veloq_nsys_query::slices::SlicesRequest {
        name: Some("step_*".into()),
        ..Default::default()
    };
    let resp = veloq_nsys_query::slices::run(trace.path(), req)?;
    assert!(
        !resp.rows.is_empty(),
        "fixture has 2 ranges matching step_*"
    );
    assert!(sidecar_path(trace.path()).exists());
    Ok(())
}

// ============================================================================
// V2 SCHEMA SEMANTIC GUARD — all-enclosing forward attribution
// ============================================================================

/// **The reason v2 exists.** With nested NVTX (outer "training_step"
/// containing inner "fwd_pass"), a kernel inside `fwd_pass` is also
/// inside `training_step`. Forward attribution must include it under
/// either pattern — innermost-only (v1) would have missed
/// `training*`.
#[test]
fn forward_attribution_matches_outer_scope_of_nested_nvtx() -> Result<()> {
    let trace = fixture::nested_nvtx_with_kernel()?;
    let trace_path = trace.path();
    delete_sidecar(trace_path)?;

    // Outer-scope pattern — innermost ("fwd_pass") does NOT match,
    // but kernel is still attributed because outer ("training_step")
    // does.
    let outer = veloq_nsys_query::stats::run(
        trace_path,
        StatsRequest {
            nvtx: Some("training*".into()),
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            ..Default::default()
        },
    )?;
    assert!(
        !outer.rows.is_empty(),
        "outer-scope pattern 'training*' must attribute the contained kernel \
         (v1 sidecar's innermost-only model would have missed this)"
    );

    // Inner-scope pattern — innermost matches directly.
    let inner = veloq_nsys_query::stats::run(
        trace_path,
        StatsRequest {
            nvtx: Some("fwd*".into()),
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            ..Default::default()
        },
    )?;
    assert!(
        !inner.rows.is_empty(),
        "inner-scope pattern 'fwd*' must also attribute the contained kernel"
    );

    // Negative — pattern matches neither scope.
    let neither = veloq_nsys_query::stats::run(
        trace_path,
        StatsRequest {
            nvtx: Some("bwd*".into()),
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            ..Default::default()
        },
    )?;
    assert!(
        neither.rows.is_empty(),
        "pattern 'bwd*' matches neither scope; kernel must not be attributed"
    );
    Ok(())
}

#[test]
fn slices_warm_state_does_not_rebuild() -> Result<()> {
    let trace = fixture::nvtx_attribution()?;
    delete_sidecar(trace.path())?;
    let t1 = Trace::open(trace.path())?;
    let path = runtime_nvtx_parent::ensure_sidecar(&t1)?;
    drop(t1);
    let mtime_before = fs::metadata(&path)?.modified()?;
    sleep(Duration::from_millis(1100));

    let req = veloq_nsys_query::slices::SlicesRequest {
        name: Some("step_*".into()),
        ..Default::default()
    };
    let _ = veloq_nsys_query::slices::run(trace.path(), req)?;

    let mtime_after = fs::metadata(&path)?.modified()?;
    assert_eq!(mtime_before, mtime_after);
    Ok(())
}
