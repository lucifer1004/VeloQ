//! `veloq metrics --type cpu-sched` integration tests against the
//! `with_cpu_sched()` fixture. The fixture's layout is documented in
//! that function; expected aggregates here mirror it 1:1 so a mismatch
//! flags either a SQL bug or a fixture bug — read both ends before
//! "fixing" one.

use anyhow::{Result, anyhow, bail};
use veloq_nsys_query::metrics::{
    CpuSchedBody, CpuSchedRequest, MetricsRequest, MetricsResponse, SchedSummaryRow,
};

mod fixture;

/// Build a cpu-sched request via a closure-on-default builder.
fn sched_req(build: impl FnOnce(&mut CpuSchedRequest)) -> MetricsRequest {
    let mut r = CpuSchedRequest::default();
    build(&mut r);
    MetricsRequest::CpuSched(r)
}

fn sched_req_default() -> MetricsRequest {
    sched_req(|_| {})
}

/// Unwrap a [`MetricsResponse`] into its cpu-sched body or fail.
fn expect_cpu_sched(r: MetricsResponse) -> Result<CpuSchedBody> {
    match r {
        MetricsResponse::CpuSched(b) => Ok(b),
        MetricsResponse::Gpu(_) => bail!("expected cpu-sched variant, got gpu"),
        MetricsResponse::Nic(_) => bail!("expected cpu-sched variant, got nic"),
        MetricsResponse::CpuSampling(_) => bail!("expected cpu-sched variant, got cpu-sampling"),
    }
}

fn row_by_key<'a>(rows: &'a [SchedSummaryRow], key: &str) -> Result<&'a SchedSummaryRow> {
    rows.iter()
        .find(|r| r.key == key)
        .ok_or_else(|| anyhow!("expected sched row with key=`{key}`"))
}

#[test]
fn tid_axis_pairs_quanta_per_thread() -> Result<()> {
    let trace = fixture::with_cpu_sched()?;
    let r = expect_cpu_sched(veloq_nsys_query::metrics::run(
        trace.path(),
        sched_req_default(),
    )?)?;
    assert_eq!(r.auxiliary.group_by, "tid");
    assert_eq!(r.rows.len(), 2, "expected one row per thread");

    let pid: i64 = 1234;
    let gtid_a: i64 = (pid << 24) | 56;
    let gtid_b: i64 = (pid << 24) | 78;
    let row_a = row_by_key(&r.rows, &format!("tid:{gtid_a}"))?;
    let row_b = row_by_key(&r.rows, &format!("tid:{gtid_b}"))?;

    // A had 2 quanta of 10ms each.
    assert_eq!(row_a.on_cpu_ns, 20_000_000);
    assert_eq!(row_a.ctx_switches, 2);
    assert_eq!(row_a.avg_quantum_ns, Some(10_000_000));
    assert_eq!(row_a.global_tid, Some(gtid_a));
    assert_eq!(row_a.pid, Some(pid));
    assert_eq!(row_a.tid, Some(56));

    // B had 1 quantum of 20ms.
    assert_eq!(row_b.on_cpu_ns, 20_000_000);
    assert_eq!(row_b.ctx_switches, 1);
    assert_eq!(row_b.avg_quantum_ns, Some(20_000_000));
    assert_eq!(row_b.tid, Some(78));
    Ok(())
}

#[test]
fn cpu_axis_groups_per_core() -> Result<()> {
    let trace = fixture::with_cpu_sched()?;
    let req = sched_req(|r| {
        r.group_by = Some("cpu".to_string());
    });
    let r = expect_cpu_sched(veloq_nsys_query::metrics::run(trace.path(), req)?)?;
    assert_eq!(r.auxiliary.group_by, "cpu");
    assert_eq!(r.rows.len(), 2);

    let cpu_10 = row_by_key(&r.rows, "cpu:10")?;
    let cpu_11 = row_by_key(&r.rows, "cpu:11")?;
    // cpu 10 hosted only Thread A: two 10ms quanta = 20ms.
    assert_eq!(cpu_10.on_cpu_ns, 20_000_000);
    assert_eq!(cpu_10.ctx_switches, 2);
    assert_eq!(cpu_10.distinct_tids, Some(1));
    // cpu 11 hosted only Thread B: one 20ms quantum.
    assert_eq!(cpu_11.on_cpu_ns, 20_000_000);
    assert_eq!(cpu_11.ctx_switches, 1);
    assert_eq!(cpu_11.distinct_tids, Some(1));
    Ok(())
}

#[test]
fn state_axis_attributes_quanta_to_exit_state() -> Result<()> {
    let trace = fixture::with_cpu_sched()?;
    let req = sched_req(|r| {
        r.group_by = Some("state".to_string());
    });
    let r = expect_cpu_sched(veloq_nsys_query::metrics::run(trace.path(), req)?)?;
    assert_eq!(r.auxiliary.group_by, "state");
    assert_eq!(r.rows.len(), 2);

    let intr = row_by_key(&r.rows, "state:Interruptible")?;
    let unk = row_by_key(&r.rows, "state:Unknown")?;
    // Two sched-outs to Interruptible: A's first quantum (10ms) +
    // B's quantum (20ms) = 30ms on_cpu.
    assert_eq!(intr.on_cpu_ns, 30_000_000);
    assert_eq!(intr.ctx_switches, 2);
    // A's sched-out @210 → next sched-in @220 = 10ms in
    // Interruptible. B's sched-out @225 has no following sched-in
    // for the same tid, so no off_cpu contribution.
    assert_eq!(intr.off_cpu_ns, Some(10_000_000));
    assert_eq!(intr.state_id, Some(2));

    // One sched-out to Unknown: A's second quantum (10ms).
    assert_eq!(unk.on_cpu_ns, 10_000_000);
    assert_eq!(unk.ctx_switches, 1);
    assert_eq!(unk.state_id, Some(0));
    Ok(())
}

#[test]
fn trust_signals_match_fixture() -> Result<()> {
    let trace = fixture::with_cpu_sched()?;
    let r = expect_cpu_sched(veloq_nsys_query::metrics::run(
        trace.path(),
        sched_req_default(),
    )?)?;
    // 6 sched events total in the fixture.
    assert_eq!(r.auxiliary.common.coverage.samples_total, 6);
    // 1/6 events have threadState=Unknown (only A's last sched-out).
    let share = r
        .auxiliary
        .unresolved_state_share
        .ok_or_else(|| anyhow!("expected unresolved_state_share"))?;
    assert!((share - (1.0 / 6.0)).abs() < 1e-9, "got {share}");
    // cpu 11 has just two events 20ms apart — that's the biggest
    // single-cpu gap in the fixture.
    assert_eq!(r.auxiliary.per_cpu_max_gap_ns, Some(20_000_000));
    Ok(())
}

#[test]
fn bucket_mode_emits_clipped_long_form() -> Result<()> {
    let trace = fixture::with_cpu_sched()?;
    // 5ms buckets cleanly split each quantum into 2 buckets, so each
    // row's `value` should be exactly 5ms.
    let req = sched_req(|r| {
        r.common.bucket_ns = Some(5_000_000);
    });
    let r = expect_cpu_sched(veloq_nsys_query::metrics::run(trace.path(), req)?)?;
    assert_eq!(r.auxiliary.common.bucket_ns, Some(5_000_000));
    assert!(r.rows.is_empty(), "bucket mode skips summary");
    assert!(!r.auxiliary.cpu_buckets.is_empty());
    for b in &r.auxiliary.cpu_buckets {
        assert!(
            b.key.starts_with("tid:"),
            "default cpu-sched bucket key should use tid axis; got `{}`",
            b.key
        );
        assert_eq!(b.agg, "sum");
        // Every bucket carries strictly positive on-cpu time and
        // matches the bucket width (5ms) since quanta are exact
        // multiples of the bucket size in this fixture.
        assert!(b.value > 0.0);
        assert!(
            (b.value - 5_000_000.0).abs() < 1e-9,
            "got value {} samples {}",
            b.value,
            b.samples
        );
    }
    Ok(())
}

/// `total_matched` must reflect pre-LIMIT match count (via
/// `COUNT(*) OVER ()`), so callers can detect truncation rather than
/// seeing `total_matched == count` for free.
#[test]
fn bucket_total_matched_reflects_pre_limit_count() -> Result<()> {
    let trace = fixture::with_cpu_sched()?;
    // Under 5ms buckets with default tid grouping the fixture lays
    // down 8 (tid, t_start) rows: A has 4 buckets across its two
    // 10ms quanta and B has 4 buckets across its 20ms quantum. Cap
    // the limit at 2 so SQL truncates the result.
    let req = sched_req(|r| {
        r.common.bucket_ns = Some(5_000_000);
        r.common.limit = 2;
    });
    let r = expect_cpu_sched(veloq_nsys_query::metrics::run(trace.path(), req)?)?;
    assert_eq!(r.count, 2, "count reflects post-limit rows");
    assert_eq!(r.auxiliary.cpu_buckets.len(), 2);
    assert_eq!(
        r.total_matched, 8,
        "total_matched must reflect rows the WHERE matched, not what survived LIMIT"
    );
    Ok(())
}

#[test]
fn unknown_group_by_errors_with_expected_list() -> Result<()> {
    let trace = fixture::with_cpu_sched()?;
    let req = sched_req(|r| {
        r.group_by = Some("symbol".to_string());
    });
    let err = match veloq_nsys_query::metrics::run(trace.path(), req) {
        Ok(_) => bail!("expected `--group-by symbol` on cpu-sched to error"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(msg.contains("symbol"), "got: {msg}");
    assert!(msg.contains("tid, cpu, state"), "got: {msg}");
    Ok(())
}

// `name_flag_rejected_on_cpu_sched` removed: `CpuSchedRequest` no
// longer has a `name_glob` field, so the invalid combination is now
// a type error rather than a runtime check. The CLI-layer rejection
// for the `--name` flag lives in `commands.rs::Cmd::Metrics`.

#[test]
fn sort_default_is_on_cpu_desc() -> Result<()> {
    let trace = fixture::with_cpu_sched()?;
    let r = expect_cpu_sched(veloq_nsys_query::metrics::run(
        trace.path(),
        sched_req_default(),
    )?)?;
    // Default sort: on_cpu DESC. Both threads have equal on_cpu_ns
    // (20ms each), so the tiebreaker (`key` ASC) decides — the
    // smaller numeric globalTid wins.
    let first = r.rows.first().ok_or_else(|| anyhow!("no rows returned"))?;
    let second = r.rows.get(1).ok_or_else(|| anyhow!("only one row"))?;
    assert!(first.on_cpu_ns >= second.on_cpu_ns);
    Ok(())
}
