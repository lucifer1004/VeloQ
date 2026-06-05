//! Runtime / Osrt as first-class kinds:
//! - `stats --type runtime` / `stats --type osrt` smoke + name resolution
//! - `stats --collapse-versioned` strips `_v<digits>` suffixes
//! - `inspect runtime:N` carries `nvtx_context` when enclosed
//! - `search --with-nvtx` decorates runtime rows via the new
//!   `Source::Runtime` reverse-attribution branch

mod fixture;

use anyhow::{Context, Result, anyhow, bail};
use duckdb::{Connection, params};
use veloq_nsys_query::stats::StatsRequest;
use veloq_nsys_query::{EventKind, KindFilter, RowId, search::SearchRequest};

// ---------- stats: Runtime / Osrt as first-class ----------------------------

#[test]
fn stats_runtime_smoke() -> Result<()> {
    let trace = fixture::host_api()?;
    let req = StatsRequest {
        kinds: KindFilter::Only(vec![EventKind::Runtime]),
        ..Default::default()
    };
    let r = veloq_nsys_query::stats::run(trace.path(), req)?;
    // 3 distinct names → 3 groups. `cudaMalloc_v3020` is a separate
    // row from `cudaMalloc` without `--collapse-versioned`.
    assert_eq!(r.total_matched, 3);
    let names: Vec<&str> = r.rows.iter().filter_map(|r| r.name.as_deref()).collect();
    assert!(names.contains(&"cudaMalloc"));
    assert!(names.contains(&"cudaMalloc_v3020"));
    assert!(names.contains(&"cudaFree"));
    Ok(())
}

#[test]
fn stats_osrt_smoke() -> Result<()> {
    let trace = fixture::host_api()?;
    let req = StatsRequest {
        kinds: KindFilter::Only(vec![EventKind::Osrt]),
        ..Default::default()
    };
    let r = veloq_nsys_query::stats::run(trace.path(), req)?;
    assert_eq!(r.total_matched, 1);
    let row = r.rows.first().ok_or_else(|| anyhow!("missing osrt row"))?;
    assert_eq!(row.name.as_deref(), Some("read"));
    assert_eq!(row.kind, "osrt");
    Ok(())
}

// ---------- --collapse-versioned --------------------------------------------

#[test]
fn stats_collapse_versioned_folds_v_suffix() -> Result<()> {
    let trace = fixture::host_api()?;
    let baseline = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Runtime]),
            ..Default::default()
        },
    )?;
    let collapsed = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Runtime]),
            collapse_versioned: true,
            ..Default::default()
        },
    )?;
    // Without collapse: 3 buckets. With collapse: cudaMalloc +
    // cudaMalloc_v3020 fold → 2 buckets.
    assert_eq!(baseline.total_matched, 3);
    assert_eq!(collapsed.total_matched, 2);
    let names: Vec<&str> = collapsed
        .rows
        .iter()
        .filter_map(|r| r.name.as_deref())
        .collect();
    assert!(
        names.contains(&"cudaMalloc") && !names.contains(&"cudaMalloc_v3020"),
        "collapsed names should drop the _v<digits> suffix; got {names:?}"
    );
    Ok(())
}

// ---------- inspect runtime:N carries nvtx_context ---------------------------

#[test]
fn inspect_runtime_gets_nvtx_context_when_enclosed() -> Result<()> {
    // nvtx_attribution fixture has runtime rows inside `step_a` and
    // `step_b` respectively. Range "step_a" is [100ms, 200ms]; the
    // first runtime row starts at 120ms (lives inside step_a, not
    // step_b). The second runtime row starts at 320ms and lives
    // inside step_b. Pin both explicitly so the test catches any
    // future regression where ctx selection drifts.
    let trace = fixture::nvtx_attribution()?;
    let r = veloq_nsys_query::inspect::run(trace.path(), &[RowId::new(EventKind::Runtime, 1)])?;
    let first = r
        .rows
        .first()
        .ok_or_else(|| anyhow!("missing runtime row"))?;
    match first {
        veloq_nsys_query::inspect::EventDetails::Runtime(rt) => {
            let ctx = rt
                .nvtx_context
                .as_ref()
                .ok_or_else(|| anyhow!("runtime row should carry nvtx_context"))?;
            assert_eq!(
                ctx.name, "step_a",
                "rowid=1 runtime at t=120ms lives inside step_a [100..200ms]"
            );
            assert_eq!(ctx.depth, 0, "step_a is an outermost range");
        }
        other => bail!("expected Runtime variant, got {other:?}"),
    }
    let r2 = veloq_nsys_query::inspect::run(trace.path(), &[RowId::new(EventKind::Runtime, 2)])?;
    let second = r2
        .rows
        .first()
        .ok_or_else(|| anyhow!("missing runtime row 2"))?;
    let veloq_nsys_query::inspect::EventDetails::Runtime(rt2) = second else {
        bail!("expected Runtime variant for rowid=2")
    };
    let ctx2 = rt2
        .nvtx_context
        .as_ref()
        .ok_or_else(|| anyhow!("rowid=2 runtime should carry nvtx_context"))?;
    assert_eq!(ctx2.name, "step_b");
    Ok(())
}

#[test]
fn inspect_runtime_no_nvtx_context_when_no_nvtx_table() -> Result<()> {
    // host_api fixture has no NVTX_EVENTS table — `nvtx_context`
    // must be None.
    let trace = fixture::host_api()?;
    let r = veloq_nsys_query::inspect::run(trace.path(), &[RowId::new(EventKind::Runtime, 1)])?;
    let first = r
        .rows
        .first()
        .ok_or_else(|| anyhow!("missing runtime row"))?;
    match first {
        veloq_nsys_query::inspect::EventDetails::Runtime(rt) => {
            assert!(
                rt.nvtx_context.is_none(),
                "runtime in trace without NVTX must serialise as nvtx_context=None"
            );
        }
        other => bail!("expected Runtime variant, got {other:?}"),
    }
    Ok(())
}

#[test]
fn inspect_runtime_in_trace_with_nvtx_but_outside_range_is_none() -> Result<()> {
    // Trickier negative: the trace HAS NVTX ranges, but the runtime
    // row falls outside every one. Distinct from the "no NVTX table"
    // case above.
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
        "#,
    )?;
    let pid: i64 = 4242;
    let gtid: i64 = pid << 24;
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1i64, "cudaFree"],
    )?;
    conn.execute(
        "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO VALUES (?, ?, ?)",
        params![0i32, 1i64, pid],
    )?;
    // NVTX range at [1000..2000ns]; runtime sits at [5000..5010ns]
    // — strictly after the range ends.
    conn.execute(
        "INSERT INTO NVTX_EVENTS (start, \"end\", globalTid, textId, text, domainId, eventType) \
         VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![
            1000i64,
            2000i64,
            gtid,
            None::<i64>,
            "outside_run",
            0i64,
            60i64
        ],
    )?;
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME \
         (start, \"end\", globalTid, correlationId, nameId) \
         VALUES (?, ?, ?, ?, ?)",
        params![5000i64, 5010i64, gtid, 99i64, 1i64],
    )?;
    let pqtdir = dir.path().join("test_pqtdir");
    std::fs::create_dir_all(&pqtdir)?;
    for t in [
        "StringIds",
        "TARGET_INFO_CUDA_CONTEXT_INFO",
        "NVTX_EVENTS",
        "CUPTI_ACTIVITY_KIND_RUNTIME",
    ] {
        let out = pqtdir.join(format!("{t}.parquet"));
        let out_lit = out.to_string_lossy().replace('\'', "''");
        conn.execute(
            &format!(r#"COPY (SELECT * FROM "{t}") TO '{out_lit}' (FORMAT PARQUET)"#),
            [],
        )?;
    }

    let r = veloq_nsys_query::inspect::run(&pqtdir, &[RowId::new(EventKind::Runtime, 1)])?;
    let first = r.rows.first().ok_or_else(|| anyhow!("missing runtime"))?;
    match first {
        veloq_nsys_query::inspect::EventDetails::Runtime(rt) => {
            assert!(
                rt.nvtx_context.is_none(),
                "runtime outside every NVTX range must serialise as nvtx_context=None"
            );
        }
        other => bail!("expected Runtime variant, got {other:?}"),
    }
    let _keep = dir;
    Ok(())
}

// ---------- --with-nvtx exercises Source::Runtime reverse path ---------------

#[test]
fn search_nvtx_pattern_scopes_runtime_rows() -> Result<()> {
    // Fixture has runtime rows inside step_a and step_b; --nvtx
    // step_a must return only the inside-step_a row.
    let trace = fixture::nvtx_attribution()?;
    let r = veloq_nsys_query::search::run(
        trace.path(),
        SearchRequest {
            kinds: KindFilter::Only(vec![EventKind::Runtime]),
            nvtx: Some("step_a".to_string()),
            limit: 10,
            ..Default::default()
        },
    )?;
    assert_eq!(
        r.rows.len(),
        1,
        "--nvtx step_a on runtime must scope to the row inside step_a; got {} rows",
        r.rows.len()
    );
    let row = r
        .rows
        .first()
        .ok_or_else(|| anyhow!("expected one runtime row"))?;
    assert_eq!(row.base().row_id.kind, EventKind::Runtime);
    Ok(())
}

#[test]
fn search_nvtx_pattern_scopes_sync_rows() -> Result<()> {
    // 3 sync rows total (step_a / step_b / sentinel); --nvtx step_a
    // narrows to exactly the inside-step_a row via attributed_sync_rowids.
    let trace = fixture::nvtx_parent_attribution()?;
    let unscoped = veloq_nsys_query::search::run(
        trace.path(),
        SearchRequest {
            kinds: KindFilter::Only(vec![EventKind::Sync]),
            limit: 100,
            ..Default::default()
        },
    )?;
    let scoped = veloq_nsys_query::search::run(
        trace.path(),
        SearchRequest {
            kinds: KindFilter::Only(vec![EventKind::Sync]),
            nvtx: Some("step_a".to_string()),
            limit: 100,
            ..Default::default()
        },
    )?;
    // Fixture has 3 sync rows total (step_a / step_b / sentinel).
    // Pre-fix scoped == unscoped (3 rows) because the WHERE clause
    // was empty; post-fix scoped should narrow to exactly the 1 row
    // inside step_a.
    assert!(
        scoped.rows.len() < unscoped.rows.len(),
        "--nvtx step_a must narrow sync rows; got scoped={} vs unscoped={}",
        scoped.rows.len(),
        unscoped.rows.len()
    );
    assert_eq!(
        scoped.rows.len(),
        1,
        "--nvtx step_a expects exactly the inside-step_a sync row"
    );
    Ok(())
}

#[test]
fn search_with_nvtx_decorates_runtime_rows() -> Result<()> {
    // `--with-nvtx` must populate `nvtx_context` on runtime rows
    // that fall inside an NVTX range.
    let trace = fixture::nvtx_attribution()?;
    let r = veloq_nsys_query::search::run(
        trace.path(),
        SearchRequest {
            kinds: KindFilter::Only(vec![EventKind::Runtime]),
            with_nvtx: true,
            limit: 10,
            ..Default::default()
        },
    )?;
    assert!(
        !r.rows.is_empty(),
        "fixture has runtime rows; search must return them"
    );
    for ev in &r.rows {
        let ctx = ev
            .base()
            .nvtx_context
            .as_ref()
            .ok_or_else(|| anyhow!("--with-nvtx must decorate runtime rows"))?;
        assert!(
            ctx.name == "step_a" || ctx.name == "step_b",
            "runtime row's NVTX ctx name should be one of the fixture's ranges; got {}",
            ctx.name
        );
    }
    Ok(())
}
