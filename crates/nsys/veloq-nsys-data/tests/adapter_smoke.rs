//! End-to-end tests for schema adapter dispatch.
//!
//! Each fixture builds a parquetdir directly: tables go into an
//! in-memory DuckDB via `CREATE TABLE` + `INSERT`, then COPYed out to
//! `<tmpdir>/test_pqtdir/<TABLE>.parquet`. `Trace::open` consumes that
//! directory exactly like `nsys export -t parquetdir` would produce.

use anyhow::{Context, Result};
use duckdb::{Connection, params};
use std::path::{Path, PathBuf};
use tempfile::TempDir;
use veloq_core::VeloqDiagnostic;
use veloq_nsys_data::{DetectionMethod, Trace};

/// Owns the tempdir so the parquetdir outlives the test's
/// `Trace::open` call. Same shape as
/// `veloq-nsys-query/tests/fixture.rs::Fixture` — duplicated rather
/// than shared because pulling that into a separate fixtures crate is
/// unnecessary scope creep for one test module.
struct Fixture {
    path: PathBuf,
    _dir: TempDir,
}

impl Fixture {
    fn path(&self) -> &Path {
        &self.path
    }
}

/// COPY every user-created DuckDB table to `<dir>/test_pqtdir/<TABLE>.parquet`.
fn finalize_to_pqtdir(conn: &Connection, dir: TempDir) -> Result<Fixture> {
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
        )
        .with_context(|| format!("copy {table} to parquet"))?;
    }
    Ok(Fixture {
        path: pqtdir,
        _dir: dir,
    })
}

/// Modern 3.x export. `META_DATA_EXPORT` carries the version triple
/// `StandardAdapter::probe` looks for; the kernel table uses canonical
/// `start`/`"end"` columns.
fn minimal_v3() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;

    conn.execute_batch(
        r#"
        CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT);
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start BIGINT, "end" BIGINT,
            deviceId BIGINT, contextId BIGINT, streamId BIGINT,
            shortName BIGINT, demangledName BIGINT,
            gridX BIGINT, gridY BIGINT, gridZ BIGINT,
            blockX BIGINT, blockY BIGINT, blockZ BIGINT,
            correlationId BIGINT,
            registersPerThread BIGINT,
            staticSharedMemory BIGINT,
            dynamicSharedMemory BIGINT,
            globalPid BIGINT
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
            start BIGINT, "end" BIGINT,
            deviceId BIGINT, contextId BIGINT, streamId BIGINT,
            bytes BIGINT, copyKind BIGINT, correlationId BIGINT
        );
        "#,
    )
    .context("create minimal_v3 schema")?;

    for (key, value) in [
        ("EXPORT_SCHEMA_VERSION_MAJOR", "3"),
        ("EXPORT_SCHEMA_VERSION_MINOR", "22"),
        ("EXPORT_SCHEMA_VERSION_MICRO", "1"),
        ("EXPORT_PRODUCT_VERSION", "2025.4.1.136"),
    ] {
        conn.execute(
            "INSERT INTO META_DATA_EXPORT (name, value) VALUES (?, ?)",
            params![key, value],
        )?;
    }
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1i64, "demo_kernel"],
    )?;
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
         (start, \"end\", deviceId, contextId, streamId, shortName, demangledName, \
          gridX, gridY, gridZ, blockX, blockY, blockZ, correlationId, \
          registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            100_000_000i64,
            110_000_000i64,
            0i32,
            0i64,
            7i64,
            1i64,
            1i64,
            1i64,
            1i64,
            1i64,
            128i64,
            1i64,
            1i64,
            42i64,
            32i64,
            0i64,
            0i64,
            0i64,
        ],
    )?;
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY \
         (start, \"end\", deviceId, contextId, streamId, bytes, copyKind, correlationId) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            120_000_000i64,
            120_500_000i64,
            0i32,
            0i64,
            7i64,
            4096i64,
            1i64,
            43i64
        ],
    )?;

    finalize_to_pqtdir(&conn, dir)
}

fn minimal_v3_nic_metrics_only() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;

    conn.execute_batch(
        r#"
        CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT);
        CREATE TABLE NET_NIC_METRIC (
            start BIGINT, "end" BIGINT,
            globalId BIGINT, portId BIGINT,
            metricsListId BIGINT, metricsIdx BIGINT,
            value BIGINT
        );
        "#,
    )
    .context("create nic-metrics-only schema")?;

    for (key, value) in [
        ("EXPORT_SCHEMA_VERSION_MAJOR", "3"),
        ("EXPORT_SCHEMA_VERSION_MINOR", "22"),
        ("EXPORT_SCHEMA_VERSION_MICRO", "1"),
    ] {
        conn.execute(
            "INSERT INTO META_DATA_EXPORT (name, value) VALUES (?, ?)",
            params![key, value],
        )?;
    }
    conn.execute(
        "INSERT INTO NET_NIC_METRIC \
         (start, \"end\", globalId, portId, metricsListId, metricsIdx, value) \
         VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![100i64, 200i64, 0i64, 0i64, 0i64, 0i64, 42i64],
    )?;

    finalize_to_pqtdir(&conn, dir)
}

#[test]
fn standard_adapter_dispatches_for_v3_schema() -> Result<()> {
    let fixture = minimal_v3()?;
    let trace = Trace::open(fixture.path())?;
    assert_eq!(trace.adapter().metadata().id, "v3_standard");
    assert_eq!(
        trace.adapter_detection_method(),
        DetectionMethod::VersionMatch,
        "fixture carries META_DATA_EXPORT so version-match should fire"
    );
    let v = trace
        .schema_version()
        .context("schema_version present in fixture")?;
    assert_eq!((v.major, v.minor, v.micro), (3, 22, 1));
    Ok(())
}

#[test]
fn daemon_limited_trace_executes_queries() -> Result<()> {
    const RESERVED_QUERY_WORKERS: usize = 1;
    const RESERVED_QUERY_MEMORY_BYTES: u64 = 64 * 1024 * 1024;

    let fixture = minimal_v3()?;
    let trace = Trace::open_for_daemon(
        fixture.path(),
        RESERVED_QUERY_WORKERS,
        Some(RESERVED_QUERY_MEMORY_BYTES),
    )?;
    assert_eq!(trace.query_worker_count(), RESERVED_QUERY_WORKERS);
    let pool = trace.build_query_worker_pool()?;
    assert_eq!(
        pool.install(rayon::current_num_threads),
        RESERVED_QUERY_WORKERS
    );
    let table_count: i64 = trace.conn().query_row(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'nsight'",
        [],
        |row| row.get(0),
    )?;
    assert!(table_count > 0);
    Ok(())
}

#[test]
fn standard_adapter_accepts_v3_metric_only_schema() -> Result<()> {
    // Metric-only NSys exports (`--trace=none --nic-metrics=lf`) carry
    // METADATA + NET_NIC_METRIC and nothing else. StandardAdapter
    // recognises them by the metric-table presence path.
    let fixture = minimal_v3_nic_metrics_only()?;
    let trace = Trace::open(fixture.path())?;
    assert_eq!(trace.adapter().metadata().id, "v3_standard");
    Ok(())
}

#[test]
fn pick_adapter_bails_on_pre_v3_schema() -> Result<()> {
    // Pre-3.x exports lack the canonical 3.x table set. veloq no longer
    // ships a fallback adapter, so `Trace::open` should error out with
    // a clear message rather than silently picking something it doesn't
    // actually normalise.
    //
    // We synthesise a parquetdir that contains a non-canonical table
    // only — StandardAdapter's probe rejects it.
    let dir = tempfile::tempdir()?;
    let conn = Connection::open_in_memory()?;
    conn.execute_batch(
        r#"
        CREATE TABLE LEGACY_ACTIVITY (timestamp BIGINT, endTimestamp BIGINT);
        "#,
    )?;
    let fixture = finalize_to_pqtdir(&conn, dir)?;

    let result = Trace::open(fixture.path());
    let err = match result {
        Ok(_) => anyhow::bail!("pre-3.x parquetdir must not open"),
        Err(e) => e,
    };
    assert_eq!(err.code().as_str(), "nsys.data.schema-adapter-unmatched");
    assert!(
        err.to_string().contains("StandardAdapter")
            && err.to_string().contains("canonical 3.x columns"),
        "error should explain the schema mismatch; got: {err}"
    );
    Ok(())
}

#[test]
fn table_exists_helper_works_through_trace() -> Result<()> {
    let fixture = minimal_v3()?;
    let trace = Trace::open(fixture.path())?;
    assert!(trace.table_exists("CUPTI_ACTIVITY_KIND_KERNEL"));
    assert!(!trace.table_exists("BOGUS_TABLE_NAME"));
    Ok(())
}
