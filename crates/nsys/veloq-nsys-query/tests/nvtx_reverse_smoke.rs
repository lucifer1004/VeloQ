//! Integration tests for reverse NVTX attribution — answering
//! "which NVTX range was this kernel/memcpy/memset/sync launched
//! inside?".
//!
//! Two surfaces under test:
//! - `inspect <row_id>` always attempts the reverse lookup; if the
//!   trace has NVTX_EVENTS + CUPTI_ACTIVITY_KIND_RUNTIME, the result
//!   carries `nvtx_context: { range_id, name, depth, iter_index }`.
//! - `search --with-nvtx` (opt-in) batches the same lookup for every
//!   GPU hit in the result.
//!
//! Both paths share the same SQL CTE (`crate::nvtx_reverse`); the
//! distinction at the surface is only "single rowid vs. IN list".

mod fixture;

use anyhow::{Context, Result, anyhow};
use duckdb::{Connection, params};
use std::path::PathBuf;
use veloq_nsys_query::inspect::EventDetails;
use veloq_nsys_query::search::SearchRequest;
use veloq_nsys_query::{EventKind, KindFilter, RowId};

/// `inspect kernel:N` over the two-range fixture must surface the
/// enclosing NVTX range on each kernel. The fixture lays down:
///
/// - NVTX `step_a` covering kernel correlationId 1001
/// - NVTX `step_b` covering kernel correlationId 1002
///
/// We don't hard-code which rowid is which (insertion order +
/// table-rowid mapping is stable, but the test reads names from
/// `nvtx_context` so the assertion is content-driven, not order-
/// dependent).
#[test]
fn inspect_kernel_populates_nvtx_context() -> Result<()> {
    let trace = fixture::nvtx_attribution()?;
    let resp = veloq_nsys_query::inspect::run(
        trace.path(),
        &[
            RowId::new(EventKind::Kernel, 1),
            RowId::new(EventKind::Kernel, 2),
        ],
    )?;
    assert_eq!(resp.rows.len(), 2);

    let mut names = std::collections::BTreeSet::new();
    for row in &resp.rows {
        let EventDetails::Kernel(k) = row else {
            return Err(anyhow!("expected Kernel variant, got {row:?}"));
        };
        let ctx = k
            .nvtx_context
            .as_ref()
            .ok_or_else(|| anyhow!("kernel:{} missing nvtx_context", k.row_id.rowid))?;
        names.insert(ctx.name.clone());
        // Single-range nesting + first occurrence: depth 0, iter 0.
        assert_eq!(ctx.depth, 0, "step_a / step_b are outermost ranges");
        assert_eq!(
            ctx.iter_index,
            Some(0),
            "each name appears once in this fixture"
        );
        // range_id points at a real nvtx row in the same trace.
        assert_eq!(ctx.range_id.kind, EventKind::Nvtx);
    }
    assert_eq!(
        names,
        ["step_a", "step_b"].iter().map(|s| s.to_string()).collect()
    );
    Ok(())
}

/// `search` without `--with-nvtx` leaves `nvtx_context: None` on every
/// hit. Pins the opt-in contract — agents that don't ask shouldn't pay
/// for the second SQL round-trip.
#[test]
fn search_without_flag_leaves_nvtx_context_empty() -> Result<()> {
    let trace = fixture::nvtx_attribution()?;
    let req = SearchRequest {
        kinds: KindFilter::Only(vec![EventKind::Kernel]),
        limit: 50,
        ..Default::default()
    };
    let r = veloq_nsys_query::search::run(trace.path(), req)?;
    assert!(!r.rows.is_empty(), "fixture has 2 kernels");
    for hit in &r.rows {
        let b = hit.base();
        assert!(
            b.nvtx_context.is_none(),
            "no --with-nvtx → nvtx_context must stay None on {}",
            b.row_id
        );
    }
    Ok(())
}

/// `search --with-nvtx` decorates every kernel hit with its enclosing
/// NVTX range. Same content as the inspect test, exercising the
/// batched code path.
#[test]
fn search_with_nvtx_decorates_every_kernel() -> Result<()> {
    let trace = fixture::nvtx_attribution()?;
    let req = SearchRequest {
        kinds: KindFilter::Only(vec![EventKind::Kernel]),
        limit: 50,
        with_nvtx: true,
        ..Default::default()
    };
    let r = veloq_nsys_query::search::run(trace.path(), req)?;
    assert_eq!(r.rows.len(), 2);

    let mut names = std::collections::BTreeSet::new();
    for hit in &r.rows {
        let b = hit.base();
        let ctx = b
            .nvtx_context
            .as_ref()
            .ok_or_else(|| anyhow!("{} missing nvtx_context with --with-nvtx", b.row_id))?;
        names.insert(ctx.name.clone());
    }
    assert_eq!(
        names,
        ["step_a", "step_b"].iter().map(|s| s.to_string()).collect()
    );
    Ok(())
}

/// Two processes share the same `correlationId` (42), each in its own
/// CUDA context and each inside its own NVTX range. Without the
/// `TARGET_INFO_CUDA_CONTEXT_INFO` bridge, the reverse SQL would
/// match both processes' runtime rows and `ROW_NUMBER()` could pick
/// the wrong process's NVTX range. The bridge restricts the runtime
/// join to the runtime whose native_pid matches the event's
/// `(device, context) → process_id` mapping, so each kernel gets its
/// own process's range.
#[test]
fn reverse_disambiguates_correlation_id_across_processes() -> Result<()> {
    let path = build_two_process_fixture()?;
    let resp = veloq_nsys_query::inspect::run(
        &path,
        &[
            RowId::new(EventKind::Kernel, 1), // process A
            RowId::new(EventKind::Kernel, 2), // process B
        ],
    )?;

    let mut by_rowid: std::collections::HashMap<i64, String> = std::collections::HashMap::new();
    for row in &resp.rows {
        let EventDetails::Kernel(k) = row else {
            return Err(anyhow!("expected Kernel variant"));
        };
        let ctx = k
            .nvtx_context
            .as_ref()
            .ok_or_else(|| anyhow!("kernel:{} missing nvtx_context", k.row_id.rowid))?;
        by_rowid.insert(k.row_id.rowid, ctx.name.clone());
    }
    assert_eq!(
        by_rowid.get(&1).map(String::as_str),
        Some("range_a"),
        "kernel in process A must resolve to A's NVTX range, not B's"
    );
    assert_eq!(
        by_rowid.get(&2).map(String::as_str),
        Some("range_b"),
        "kernel in process B must resolve to B's NVTX range, not A's"
    );
    Ok(())
}

/// Traces without `NVTX_EVENTS` (the common case for GPU-only
/// captures) must still produce a well-shaped response — the reverse
/// query is a best-effort decoration, not a hard requirement. Pin
/// "no NVTX → no nvtx_context, no error".
#[test]
fn inspect_without_nvtx_table_returns_no_context() -> Result<()> {
    let trace = fixture::minimal_gpu()?;
    let resp = veloq_nsys_query::inspect::run(trace.path(), &[RowId::new(EventKind::Kernel, 1)])?;
    assert_eq!(resp.rows.len(), 1);
    let EventDetails::Kernel(k) = resp
        .rows
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("missing kernel row"))?
    else {
        return Err(anyhow!("expected Kernel variant"));
    };
    assert!(
        k.nvtx_context.is_none(),
        "minimal_gpu has no NVTX_EVENTS — nvtx_context must be None"
    );
    Ok(())
}

/// Build a two-process synthetic trace where each process re-uses the
/// same `correlationId = 42` in its own CUDA context. The disambiguation
/// fixture lives next to the test (not `fixture.rs`) because it's the
/// only consumer.
///
/// Layout:
/// - process A: pid=1000, context_id=1, device=0
///   NVTX `range_a` [100..200ms] on globalTid_A, kernel rowid=1, corr=42
/// - process B: pid=2000, context_id=2, device=0
///   NVTX `range_b` [300..400ms] on globalTid_B, kernel rowid=2, corr=42
///
/// Without the context-info bridge the reverse SQL would join kernel:1's
/// correlationId=42 to *both* processes' runtime rows; the bridge
/// restricts it to the runtime whose native_pid matches A's
/// `(device=0, context=1) → process=1000`. NamedTempFile keeps the trace
/// alive for the duration of the test via the path being returned.
fn build_two_process_fixture() -> Result<PathBuf> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let pqtdir = dir.path().join("test_pqtdir");
    std::fs::create_dir_all(&pqtdir)?;
    // Leak the TempDir so the parquetdir outlives the test scope
    // without us threading a handle. Same caveat as the previous
    // NamedTempFile leak.
    std::mem::forget(dir);
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
        "#,
    )?;
    conn.execute_batch(fixture::KERNEL_TABLE_SQL)?;

    let pid_a: i64 = 1000;
    let pid_b: i64 = 2000;
    let gtid_a: i64 = pid_a << 24;
    let gtid_b: i64 = pid_b << 24;
    // Each process owns its own context on device 0. `correlationId`
    // values overlap (42 in both), so the bridge through
    // TARGET_INFO_CUDA_CONTEXT_INFO is the only thing keeping the
    // join correct.
    conn.execute(
        "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO (deviceId, contextId, processId) \
         VALUES (?, ?, ?)",
        params![0i32, 1i64, pid_a],
    )?;
    conn.execute(
        "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO (deviceId, contextId, processId) \
         VALUES (?, ?, ?)",
        params![0i32, 2i64, pid_b],
    )?;

    for (gtid, s, e, name) in &[
        (gtid_a, 100_000_000i64, 200_000_000i64, "range_a"),
        (gtid_b, 300_000_000i64, 400_000_000i64, "range_b"),
    ] {
        conn.execute(
            "INSERT INTO NVTX_EVENTS \
             (start, \"end\", globalTid, textId, text, domainId, eventType) \
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![*s, *e, *gtid, None::<i64>, *name, 0i64, 60i64],
        )?;
    }

    // Both runtime rows share correlationId 42 — distinguishable only
    // by `globalTid`'s native_pid bits.
    let corr: i64 = 42;
    for (gtid, runtime_start) in &[(gtid_a, 130_000_000i64), (gtid_b, 330_000_000i64)] {
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME \
             (start, \"end\", globalTid, correlationId, nameId) \
             VALUES (?, ?, ?, ?, ?)",
            params![
                *runtime_start,
                *runtime_start + 1_000_000i64,
                *gtid,
                corr,
                None::<i64>
            ],
        )?;
    }

    // kernel rowid 1 lives in (device=0, context=1) → process A
    // kernel rowid 2 lives in (device=0, context=2) → process B
    for (kernel_start, kernel_end, context_id) in &[
        (140_000_000i64, 150_000_000i64, 1i64),
        (340_000_000i64, 350_000_000i64, 2i64),
    ] {
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
             (start, \"end\", deviceId, contextId, streamId, \
              shortName, demangledName, gridX, gridY, gridZ, \
              blockX, blockY, blockZ, correlationId, \
              registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                *kernel_start,
                *kernel_end,
                0i32,
                *context_id,
                7i64,
                1i64,
                1i64,
                1i64,
                1i64,
                1i64,
                128i64,
                1i64,
                1i64,
                corr,
                32i64,
                0i64,
                0i64,
                0i64,
            ],
        )?;
    }

    // COPY each table to parquet.
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
    Ok(pqtdir)
}
