//! Functional smoke tests for `slices --aggregate`.
//!
//! Covers:
//! - SQL aggregate parity vs the per-instance `slices` view (jq-style
//!   equivalent): same fixture, the per-name sums and counts match.
//! - Small-`--limit` bias scenario: `slices --limit N` + manual
//!   group-by silently drops contributions when N < total_matched;
//!   `slices --aggregate` does not.

use anyhow::{Context, Result};
use duckdb::{Connection, params};
use std::collections::HashMap;
use std::path::PathBuf;
use tempfile::TempDir;
use veloq_nsys_query::slices::{
    Slice, SliceAggregate, SlicesAggregateGroupBy, SlicesRequest, SlicesRow, SlicesView,
};

mod fixture;

fn finalize_to_pqtdir(conn: &Connection, dir: TempDir) -> Result<(TempDir, PathBuf)> {
    let pqtdir = dir.path().join("test_pqtdir");
    std::fs::create_dir_all(&pqtdir).context("create parquetdir")?;
    let mut stmt = conn.prepare(
        "SELECT table_name FROM information_schema.tables \
         WHERE table_schema = 'main' AND table_type = 'BASE TABLE'",
    )?;
    let mut rows = stmt.query([])?;
    let mut tables = Vec::new();
    while let Some(r) = rows.next()? {
        tables.push(r.get::<_, String>(0)?);
    }
    for table in &tables {
        let out = pqtdir.join(format!("{table}.parquet"));
        let out_lit = out.to_string_lossy().replace('\'', "''");
        conn.execute(
            &format!(r#"COPY (SELECT * FROM "{table}") TO '{out_lit}' (FORMAT PARQUET)"#),
            [],
        )?;
    }
    Ok((dir, pqtdir))
}

/// Build a fixture with N NVTX instances of two repeated names
/// (`step_a` and `step_b`, alternating), each with one kernel of
/// known duration. Tests group-by-name aggregation directly.
fn repeating_names_fixture(per_name: usize) -> Result<(TempDir, PathBuf)> {
    let dir = tempfile::tempdir().context("tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    conn.execute_batch(
        r#"
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO (
            deviceId BIGINT, contextId BIGINT, processId BIGINT
        );
        CREATE TABLE NVTX_EVENTS (
            start BIGINT, "end" BIGINT, globalTid BIGINT,
            textId BIGINT, text TEXT,
            domainId BIGINT, eventType BIGINT
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start BIGINT, "end" BIGINT,
            globalTid BIGINT, correlationId BIGINT, nameId BIGINT
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start BIGINT, "end" BIGINT,
            deviceId BIGINT, contextId BIGINT, streamId BIGINT,
            shortName BIGINT, demangledName BIGINT,
            gridX BIGINT, gridY BIGINT, gridZ BIGINT,
            blockX BIGINT, blockY BIGINT, blockZ BIGINT,
            correlationId BIGINT, registersPerThread BIGINT,
            staticSharedMemory BIGINT, dynamicSharedMemory BIGINT,
            globalPid BIGINT
        );
        "#,
    )?;
    let pid: i64 = 12345;
    let global_tid: i64 = pid << 24;
    conn.execute(
        "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO (deviceId, contextId, processId) VALUES (?, ?, ?)",
        params![0i32, 1i64, pid],
    )?;
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1i64, "_kernel"],
    )?;

    // Range instances are spaced 1000 ns apart so they never overlap.
    // Each range is 100 ns wide; the runtime + kernel fall inside.
    // step_a kernels: 50 ns dur; step_b kernels: 30 ns dur.
    let mut next_corr: i64 = 1000;
    let mut base: i64 = 0;
    for i in 0..per_name {
        for (name, kdur) in [("step_a", 50i64), ("step_b", 30i64)] {
            let r_start = base;
            let r_end = r_start + 100;
            let k_start = r_start + 10;
            let k_end = k_start + kdur;
            conn.execute(
                "INSERT INTO NVTX_EVENTS \
                 (start, \"end\", globalTid, textId, text, domainId, eventType) \
                 VALUES (?, ?, ?, ?, ?, ?, ?)",
                params![r_start, r_end, global_tid, None::<i64>, name, 0i64, 60i64],
            )?;
            conn.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME \
                 (start, \"end\", globalTid, correlationId, nameId) \
                 VALUES (?, ?, ?, ?, ?)",
                params![r_start + 5, r_start + 8, global_tid, next_corr, None::<i64>,],
            )?;
            conn.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
                 (start, \"end\", deviceId, contextId, streamId, \
                  shortName, demangledName, gridX, gridY, gridZ, \
                  blockX, blockY, blockZ, correlationId, \
                  registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                params![
                    k_start, k_end, 0i32, 1i64, 7i64, 1i64, 1i64, 1i64, 1i64, 1i64, 128i64, 1i64,
                    1i64, next_corr, 32i64, 0i64, 0i64, 0i64,
                ],
            )?;
            next_corr += 1;
            base += 1000;
            let _ = i;
        }
    }
    finalize_to_pqtdir(&conn, dir)
}

fn instance_rows(resp: &veloq_nsys_query::slices::SlicesResponse) -> Vec<&Slice> {
    resp.rows
        .iter()
        .filter_map(|row| match row {
            SlicesRow::Instance(s) => Some(s),
            SlicesRow::Aggregate(_) => None,
        })
        .collect()
}

fn aggregate_rows(resp: &veloq_nsys_query::slices::SlicesResponse) -> Vec<&SliceAggregate> {
    resp.rows
        .iter()
        .filter_map(|row| match row {
            SlicesRow::Aggregate(r) => Some(r),
            SlicesRow::Instance(_) => None,
        })
        .collect()
}

fn aggregate_request(name: Option<String>, group_by: SlicesAggregateGroupBy) -> SlicesRequest {
    SlicesRequest {
        name,
        view: SlicesView::Aggregate,
        group_by,
        limit: 1000,
        ..Default::default()
    }
}

#[test]
fn aggregate_rows_match_slices_group_by_name() -> Result<()> {
    // Use the canonical 2-name fixture from the shared module — two
    // distinct names, one instance each. Aggregate instances must be 1
    // per name and totals must match the per-instance numbers.
    let trace = fixture::nvtx_attribution()?;
    let slices = veloq_nsys_query::slices::run(
        trace.path(),
        SlicesRequest {
            limit: 1000,
            ..Default::default()
        },
    )?;
    let aggregate = veloq_nsys_query::slices::run(
        trace.path(),
        aggregate_request(None, SlicesAggregateGroupBy::Name),
    )?;
    assert_eq!(aggregate.view, "aggregate");
    assert_eq!(aggregate.group_by, Some("name"));

    // jq-equivalent grouping of slices: SUM(attributed_kernel +
    // attributed_memcpy + attributed_memset) grouped by name.
    let mut by_name: HashMap<&str, (i64, i64)> = HashMap::new(); // name -> (count, total_ns)
    for s in instance_rows(&slices) {
        let total = s.attributed_kernel_ns + s.attributed_memcpy_ns + s.attributed_memset_ns;
        let e = by_name.entry(s.name.as_str()).or_insert((0, 0));
        e.0 += 1;
        e.1 += total;
    }

    // Build the same map from the aggregate response.
    let mut aggregate_map: HashMap<&str, (i64, i64)> = HashMap::new();
    for r in aggregate_rows(&aggregate) {
        aggregate_map.insert(r.name.as_str(), (r.instances, r.attributed_total_ns));
    }

    assert_eq!(by_name, aggregate_map, "per-name aggregates must match");
    Ok(())
}

#[test]
fn small_limit_jq_path_is_biased_aggregate_is_not() -> Result<()> {
    // 8 instances per name (16 total ranges). slices --limit 5 + a
    // jq-style grouping sees only 5 of the 16 ranges; the per-name
    // totals are biased. slices --aggregate returns full per-name
    // aggregates regardless of any group-row limit.
    let (_trace_dir, trace) = repeating_names_fixture(8)?;

    let biased = veloq_nsys_query::slices::run(
        &trace,
        SlicesRequest {
            name: Some("step_*".into()),
            limit: 5,
            ..Default::default()
        },
    )?;
    assert_eq!(biased.rows.len(), 5, "biased slices view captures 5 rows");
    // Total of biased rows is strictly less than the full attribution.
    let biased_total: i64 = instance_rows(&biased)
        .iter()
        .map(|s| s.attributed_kernel_ns + s.attributed_memcpy_ns + s.attributed_memset_ns)
        .sum();

    let aggregate = veloq_nsys_query::slices::run(
        &trace,
        aggregate_request(Some("step_*".into()), SlicesAggregateGroupBy::Name),
    )?;
    let aggregate_rows = aggregate_rows(&aggregate);
    assert_eq!(
        aggregate_rows.len(),
        2,
        "two distinct names group to 2 rows"
    );
    let aggregate_total: i64 = aggregate_rows.iter().map(|r| r.attributed_total_ns).sum();

    // Expected unbiased: 8 step_a kernels × 50 ns + 8 step_b × 30 ns = 640.
    assert_eq!(aggregate_total, 8 * 50 + 8 * 30);
    assert!(
        biased_total < aggregate_total,
        "biased jq path under-counts: biased={biased_total}, full={aggregate_total}"
    );

    // step_a / step_b counts are both 8, totals are 400 / 240.
    let by_name: HashMap<&str, &SliceAggregate> = aggregate_rows
        .iter()
        .map(|r| (r.name.as_str(), *r))
        .collect();
    let a = by_name.get("step_a").context("step_a row missing")?;
    let b = by_name.get("step_b").context("step_b row missing")?;
    assert_eq!(a.instances, 8);
    assert_eq!(a.attributed_total_ns, 400);
    assert_eq!(b.instances, 8);
    assert_eq!(b.attributed_total_ns, 240);
    Ok(())
}

#[test]
fn percentile_columns_are_finite_and_ordered() -> Result<()> {
    // p50 ≤ p99 — and both are finite (no NaN/Inf even when count is small).
    let (_trace_dir, trace) = repeating_names_fixture(4)?;
    let aggregate = veloq_nsys_query::slices::run(
        &trace,
        aggregate_request(Some("step_*".into()), SlicesAggregateGroupBy::Name),
    )?;
    for r in aggregate_rows(&aggregate) {
        assert!(r.p50_ns.is_finite() && r.p50_ns >= 0.0);
        assert!(r.p99_ns.is_finite() && r.p99_ns >= 0.0);
        assert!(
            r.p50_ns <= r.p99_ns,
            "p50 must be ≤ p99 for {} (got {} / {})",
            r.name,
            r.p50_ns,
            r.p99_ns
        );
    }
    Ok(())
}

#[test]
fn empty_pattern_match_returns_empty_rows() -> Result<()> {
    let trace = fixture::nvtx_attribution()?;
    let aggregate = veloq_nsys_query::slices::run(
        trace.path(),
        SlicesRequest {
            name: Some("does_not_exist_*".into()),
            view: SlicesView::Aggregate,
            limit: 100,
            ..Default::default()
        },
    )?;
    assert_eq!(aggregate.count, 0);
    assert_eq!(aggregate.total_matched, 0);
    assert!(aggregate.rows.is_empty());
    Ok(())
}
