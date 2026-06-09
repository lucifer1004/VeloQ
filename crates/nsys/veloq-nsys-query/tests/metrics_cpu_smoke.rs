//! Integration tests against `metrics::run` + `inspect::run` for
//! `--type cpu-sampling`. Pins:
//!
//! - `--group-by symbol` rolls leaf frames up correctly, with the
//!   unresolved bucket separate per module.
//! - `--group-by tid` / `cpu` / `module` aggregate without needing
//!   the callchain table (except module, which does).
//! - Trust signals — `unresolved_leaf_share`, `kernel_leaf_share`,
//!   `truncated_stack_share` — reflect deterministic fixture content.
//! - `--from`/`--to` clips on `COMPOSITE_EVENTS.start`.
//! - `--cpu` / `--tid` scope filters.
//! - `--name` glob filters hotspot rows.
//! - `--bucket` produces long-form rows with `agg = "sum"`.
//! - `inspect cpu_sample:N` returns full callchain leaf-first plus
//!   the resolved thread-state name.
//! - `--type gpu` cross-flag rejection: passing `--group-by` errors.
//! - Missing-table error mentions the required nsys capture flag.
//!
//! No-panic policy: same idiom as `metrics_smoke.rs` — `ok_or_else`
//! plus `?` instead of `.unwrap()` / `.expect()` / `[i]`.

mod fixture;

use anyhow::{Result, anyhow, bail};
use veloq_core::time::TimeWindow;
use veloq_nsys_query::RowId;
use veloq_nsys_query::inspect::EventDetails;
use veloq_nsys_query::metrics::{
    CpuSamplingBody, CpuSamplingRequest, HotspotRow, MetricsRequest, MetricsResponse,
};

/// Build a cpu-sampling request via a closure-on-default builder.
fn cpu_req(build: impl FnOnce(&mut CpuSamplingRequest)) -> MetricsRequest {
    let mut r = CpuSamplingRequest::default();
    build(&mut r);
    MetricsRequest::CpuSampling(r)
}

/// Default cpu-sampling request — caller doesn't need to customise.
fn cpu_req_default() -> MetricsRequest {
    cpu_req(|_| {})
}

/// Unwrap a [`MetricsResponse`] into its cpu-sampling body or fail.
fn expect_cpu_sampling(r: MetricsResponse) -> Result<CpuSamplingBody> {
    match r {
        MetricsResponse::CpuSampling(b) => Ok(b),
        MetricsResponse::Gpu(_) => bail!("expected cpu-sampling variant, got gpu"),
        MetricsResponse::Nic(_) => bail!("expected cpu-sampling variant, got nic"),
        MetricsResponse::CpuSched(_) => bail!("expected cpu-sampling variant, got cpu-sched"),
    }
}

fn row_by_key<'a>(rows: &'a [HotspotRow], key: &str) -> Result<&'a HotspotRow> {
    rows.iter()
        .find(|r| r.key == key)
        .ok_or_else(|| anyhow!("expected hotspot row with key=`{key}`"))
}

#[test]
fn missing_composite_events_errors_with_capture_hint() -> Result<()> {
    let trace = fixture::minimal_gpu()?;
    let r = veloq_nsys_query::metrics::run(trace.path(), cpu_req_default());
    let err = match r {
        Ok(_) => bail!("expected missing-table error"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(msg.contains("COMPOSITE_EVENTS"), "got: {msg}");
    assert!(msg.contains("--sample"), "got: {msg}");
    Ok(())
}

#[test]
fn symbol_axis_collapses_unresolved_per_module() -> Result<()> {
    let trace = fixture::with_cpu_sampling()?;
    let r = expect_cpu_sampling(veloq_nsys_query::metrics::run(
        trace.path(),
        cpu_req_default(),
    )?)?;

    assert_eq!(r.auxiliary.group_by, "symbol");

    // 6 samples total: 4 func_a, 1 func_b, 1 <unresolved>.
    assert_eq!(r.auxiliary.common.coverage.samples_total, 6);

    let func_a = row_by_key(&r.rows, "func_a")?;
    assert_eq!(func_a.samples, 4);
    assert!((func_a.percentage - (4.0 * 100.0 / 6.0)).abs() < 1e-9);
    assert_eq!(func_a.module_name.as_deref(), Some("libapp.so"));
    assert_eq!(func_a.kernel_mode, Some(false));
    assert_eq!(func_a.unresolved, Some(false));
    assert_eq!(func_a.sample_row_id.as_deref(), Some("cpu_sample:1"));
    assert_eq!(func_a.sample_start_ns, Some(100_000_000));

    let func_b = row_by_key(&r.rows, "func_b")?;
    assert_eq!(func_b.samples, 1);
    assert_eq!(func_b.sample_row_id.as_deref(), Some("cpu_sample:3"));

    let unres = row_by_key(&r.rows, "<unresolved>@[kernel.kallsyms]")?;
    assert_eq!(unres.samples, 1);
    assert_eq!(unres.kernel_mode, Some(true));
    assert_eq!(unres.unresolved, Some(true));
    assert!(unres.symbol_name.is_none(), "unresolved row has no symbol");
    assert_eq!(unres.sample_row_id.as_deref(), Some("cpu_sample:4"));
    Ok(())
}

#[test]
fn stack_axis_groups_callchain_signatures() -> Result<()> {
    let trace = fixture::with_cpu_sampling()?;
    let req = cpu_req(|r| {
        r.group_by = Some("stack".to_string());
    });
    let r = expect_cpu_sampling(veloq_nsys_query::metrics::run(trace.path(), req)?)?;
    assert_eq!(r.auxiliary.group_by, "stack");
    assert_eq!(r.rows.len(), 3);

    let hot = r
        .rows
        .iter()
        .find(|row| row.samples == 4)
        .ok_or_else(|| anyhow!("expected 4-sample stack row"))?;
    assert!(hot.key.starts_with("stack|"));
    assert_eq!(hot.stack_hash.as_deref(), hot.key.strip_prefix("stack|"));
    assert_eq!(hot.stack_depth, Some(3));
    assert_eq!(hot.sample_row_id.as_deref(), Some("cpu_sample:1"));
    assert!(
        hot.stack_frames
            .iter()
            .any(|frame| frame == "func_a@libapp.so"),
        "frames: {:?}",
        hot.stack_frames
    );
    assert!(
        hot.stack_frames.iter().any(|frame| frame == "[Max depth]"),
        "frames: {:?}",
        hot.stack_frames
    );
    Ok(())
}

#[test]
fn stack_axis_name_glob_matches_any_frame() -> Result<()> {
    let trace = fixture::with_cpu_sampling()?;
    let req = cpu_req(|r| {
        r.group_by = Some("stack".to_string());
        r.name_glob = Some("*func_b*".to_string());
    });
    let r = expect_cpu_sampling(veloq_nsys_query::metrics::run(trace.path(), req)?)?;
    assert_eq!(r.rows.len(), 1);
    let row = r
        .rows
        .first()
        .ok_or_else(|| anyhow!("expected one stack row"))?;
    assert_eq!(row.samples, 1);
    assert!(
        row.stack_frames
            .iter()
            .any(|frame| frame == "func_b@libapp.so"),
        "frames: {:?}",
        row.stack_frames
    );
    Ok(())
}

#[test]
fn stack_axis_rejects_bucket_mode() -> Result<()> {
    let trace = fixture::with_cpu_sampling()?;
    let req = cpu_req(|r| {
        r.group_by = Some("stack".to_string());
        r.common.bucket_ns = Some(20_000_000);
    });
    let err = match veloq_nsys_query::metrics::run(trace.path(), req) {
        Ok(_) => bail!("expected stack bucket mode to error"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("--group-by stack does not support --bucket"),
        "got: {msg}"
    );
    Ok(())
}

#[test]
fn trust_signals_match_fixture_layout() -> Result<()> {
    let trace = fixture::with_cpu_sampling()?;
    let r = expect_cpu_sampling(veloq_nsys_query::metrics::run(
        trace.path(),
        cpu_req_default(),
    )?)?;
    // 1 of 6 leaves is unresolved + kernel; all 6 stacks end at "[Max depth]".
    let one_sixth = 1.0 / 6.0;
    let approx = |a: f64, b: f64| (a - b).abs() < 1e-9;
    let unres = r
        .auxiliary
        .unresolved_leaf_share
        .ok_or_else(|| anyhow!("expected unresolved_leaf_share"))?;
    let kern = r
        .auxiliary
        .kernel_leaf_share
        .ok_or_else(|| anyhow!("expected kernel_leaf_share"))?;
    let trunc = r
        .auxiliary
        .truncated_stack_share
        .ok_or_else(|| anyhow!("expected truncated_stack_share"))?;
    assert!(approx(unres, one_sixth), "got unresolved={unres}");
    assert!(approx(kern, one_sixth), "got kernel={kern}");
    assert!(approx(trunc, 1.0), "got truncated={trunc}");
    Ok(())
}

#[test]
fn tid_axis_groups_globaltid() -> Result<()> {
    let trace = fixture::with_cpu_sampling()?;
    let req = cpu_req(|r| {
        r.group_by = Some("tid".to_string());
    });
    let r = expect_cpu_sampling(veloq_nsys_query::metrics::run(trace.path(), req)?)?;
    assert_eq!(r.rows.len(), 2);
    let pid_a = 1234i64;
    let tid_a = 56i64;
    let gtid_a = (pid_a << 24) | tid_a;
    let row_a = row_by_key(&r.rows, &gtid_a.to_string())?;
    assert_eq!(row_a.samples, 4);
    assert_eq!(row_a.global_tid, Some(gtid_a));
    assert_eq!(row_a.pid, Some(pid_a));
    assert_eq!(row_a.tid, Some(tid_a));
    Ok(())
}

#[test]
fn name_on_numeric_axis_errors() -> Result<()> {
    // `--name` on tid / cpu axes used to silently no-op. Now it
    // hard-errors so agents don't burn time wondering why their
    // glob had no effect.
    let trace = fixture::with_cpu_sampling()?;
    for axis in ["tid", "cpu"] {
        let axis_owned = axis.to_string();
        let req = cpu_req(move |r| {
            r.group_by = Some(axis_owned);
            r.name_glob = Some("*foo*".to_string());
        });
        let err = match veloq_nsys_query::metrics::run(trace.path(), req) {
            Ok(_) => bail!("expected --name on {axis} axis to error"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(msg.contains("--name doesn't apply"), "got: {msg}");
    }
    Ok(())
}

#[test]
fn cpu_axis_groups_cpu() -> Result<()> {
    let trace = fixture::with_cpu_sampling()?;
    let req = cpu_req(|r| {
        r.group_by = Some("cpu".to_string());
    });
    let r = expect_cpu_sampling(veloq_nsys_query::metrics::run(trace.path(), req)?)?;
    let row_10 = row_by_key(&r.rows, "10")?;
    let row_11 = row_by_key(&r.rows, "11")?;
    assert_eq!(row_10.samples, 4);
    assert_eq!(row_10.cpu, Some(10));
    assert_eq!(row_11.samples, 2);
    Ok(())
}

#[test]
fn module_axis_basenames_paths() -> Result<()> {
    let trace = fixture::with_cpu_sampling()?;
    let req = cpu_req(|r| {
        r.group_by = Some("module".to_string());
    });
    let r = expect_cpu_sampling(veloq_nsys_query::metrics::run(trace.path(), req)?)?;
    // libapp.so: 5 (user-mode samples), kernel.kallsyms: 1.
    let libapp = row_by_key(&r.rows, "libapp.so")?;
    let kallsyms = row_by_key(&r.rows, "[kernel.kallsyms]")?;
    assert_eq!(libapp.samples, 5);
    assert_eq!(kallsyms.samples, 1);
    Ok(())
}

#[test]
fn time_window_clips_samples() -> Result<()> {
    let trace = fixture::with_cpu_sampling()?;
    // Samples at 100/110/120/130/140/150 ms; primary origin is
    // 100ms (the anchor kernel's start). `--from 0 --to 25ms`
    // (relative) captures 100..125 ms = samples 1, 2, 3.
    let window = TimeWindow::parse("0-25ms")?;
    let req = cpu_req(move |r| {
        r.common.time_window = Some(window);
    });
    let r = expect_cpu_sampling(veloq_nsys_query::metrics::run(trace.path(), req)?)?;
    assert_eq!(r.auxiliary.common.coverage.samples_total, 3);
    let func_a = row_by_key(&r.rows, "func_a")?;
    let func_b = row_by_key(&r.rows, "func_b")?;
    assert_eq!(func_a.samples, 2);
    assert_eq!(func_b.samples, 1);
    // No unresolved sample in this window.
    assert!(
        r.rows
            .iter()
            .all(|h| h.key != "<unresolved>@[kernel.kallsyms]")
    );
    Ok(())
}

#[test]
fn cpu_filter_scopes_to_one_core() -> Result<()> {
    let trace = fixture::with_cpu_sampling()?;
    let req = cpu_req(|r| {
        r.cpu = Some(11);
    });
    let r = expect_cpu_sampling(veloq_nsys_query::metrics::run(trace.path(), req)?)?;
    assert_eq!(r.auxiliary.common.coverage.samples_total, 2);
    assert_eq!(r.auxiliary.cpu_filter, Some(11));
    Ok(())
}

#[test]
fn tid_filter_scopes_to_one_thread() -> Result<()> {
    let trace = fixture::with_cpu_sampling()?;
    let pid_a = 1234i64;
    let tid_a = 56i64;
    let gtid_a = (pid_a << 24) | tid_a;
    let req = cpu_req(move |r| {
        r.tid = Some(gtid_a);
    });
    let r = expect_cpu_sampling(veloq_nsys_query::metrics::run(trace.path(), req)?)?;
    assert_eq!(r.auxiliary.common.coverage.samples_total, 4);
    assert_eq!(r.auxiliary.tid_filter, Some(gtid_a));
    Ok(())
}

#[test]
fn name_glob_filters_symbol_rows() -> Result<()> {
    let trace = fixture::with_cpu_sampling()?;
    let req = cpu_req(|r| {
        r.name_glob = Some("func_*".to_string());
    });
    let r = expect_cpu_sampling(veloq_nsys_query::metrics::run(trace.path(), req)?)?;
    // Glob matches `func_a` + `func_b`; excludes `<unresolved>@...`.
    assert!(
        r.rows.iter().all(|h| h.key.starts_with("func_")),
        "found non-matching key: {:?}",
        r.rows.iter().map(|h| &h.key).collect::<Vec<_>>()
    );
    assert_eq!(r.rows.len(), 2);
    Ok(())
}

#[test]
fn bucket_mode_emits_long_form_rows() -> Result<()> {
    let trace = fixture::with_cpu_sampling()?;
    let req = cpu_req(|r| {
        // 20ms buckets
        r.common.bucket_ns = Some(20_000_000);
    });
    let r = expect_cpu_sampling(veloq_nsys_query::metrics::run(trace.path(), req)?)?;
    assert_eq!(r.auxiliary.common.bucket_ns, Some(20_000_000));
    assert!(r.rows.is_empty(), "bucket mode skips hotspot");
    assert!(!r.auxiliary.cpu_buckets.is_empty());
    assert_eq!(r.count, r.auxiliary.cpu_buckets.len());
    assert_eq!(r.total_matched, r.auxiliary.cpu_buckets.len() as i64);
    // Every cpu bucket row carries agg="sum" and value == samples.
    for b in &r.auxiliary.cpu_buckets {
        assert!(
            [
                "func_a",
                "func_b",
                "<unresolved>@[kernel.kallsyms]",
                "<unresolved_addr_string>",
            ]
            .contains(&b.key.as_str()),
            "unexpected cpu-sampling bucket key `{}`",
            b.key
        );
        assert_eq!(b.agg, "sum");
        assert!((b.value - b.samples as f64).abs() < 1e-9);
    }
    Ok(())
}

// No runtime check rejects CPU flags on a GPU request: the tagged enum
// makes `MetricsRequest::Gpu` literally unable to carry `group_by` /
// `name` / `cpu` / `tid` fields, so the type system enforces the
// invariant. Cross-source CLI flag rejection lives in
// `crates/veloq/src/commands.rs::Cmd::Metrics`.

/// Regression: traces with `COMPOSITE_EVENTS` but no
/// `SAMPLING_CALLCHAINS` (captured with sampling on, backtrace off)
/// used to fail because the inspect path always prepared the chain
/// query. Surface the event row with an empty `callchain` instead.
#[test]
fn inspect_cpu_sample_empty_chain_when_sampling_callchains_missing() -> Result<()> {
    let trace = fixture::with_cpu_sampling_no_callchains()?;
    let id: RowId = "cpu_sample:1".parse()?;
    let resp = veloq_nsys_query::inspect::run(trace.path(), &[id])?;
    let first = resp
        .rows
        .first()
        .ok_or_else(|| anyhow!("expected one event"))?;
    let cs = match first {
        EventDetails::CpuSample(d) => d,
        other => bail!("expected CpuSample, got {other:?}"),
    };
    assert_eq!(cs.cpu, 10);
    assert_eq!(cs.thread_state, 1);
    assert_eq!(cs.thread_state_name.as_deref(), Some("Running"));
    assert!(
        cs.callchain.is_empty(),
        "callchain must be empty when SAMPLING_CALLCHAINS is absent"
    );
    Ok(())
}

#[test]
fn inspect_cpu_sample_returns_leaf_first_callchain() -> Result<()> {
    let trace = fixture::with_cpu_sampling()?;
    // sample id=4 is the unresolved kernel sample.
    let id: RowId = "cpu_sample:4".parse()?;
    let resp = veloq_nsys_query::inspect::run(trace.path(), &[id])?;
    let first = resp
        .rows
        .first()
        .ok_or_else(|| anyhow!("expected one event"))?;
    let cs = match first {
        EventDetails::CpuSample(d) => d,
        other => bail!("expected CpuSample, got {other:?}"),
    };
    assert_eq!(cs.cpu, 10);
    assert_eq!(cs.thread_state, 1);
    assert_eq!(cs.thread_state_name.as_deref(), Some("Running"));
    // Decoded pid/tid from the fixture's globalTid = (1234 << 24) | 56.
    assert_eq!(cs.pid, 1234);
    assert_eq!(cs.tid, 56);
    assert_eq!(cs.callchain.len(), 3);
    let leaf = cs
        .callchain
        .first()
        .ok_or_else(|| anyhow!("expected leaf frame"))?;
    assert_eq!(leaf.depth, 0);
    assert!(leaf.kernel_mode);
    assert!(leaf.unresolved);
    assert_eq!(leaf.module.as_deref(), Some("[kernel.kallsyms]"));
    let deepest = cs
        .callchain
        .last()
        .ok_or_else(|| anyhow!("expected deepest frame"))?;
    assert_eq!(deepest.depth, 2);
    assert_eq!(deepest.symbol.as_deref(), Some("[Max depth]"));
    Ok(())
}
