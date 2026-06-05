//! End-to-end smoke tests for the NVTX tree sidecar.
//!
//! Drive a synthetic NVTX_EVENTS table through the full build ->
//! parquet -> DuckDB-view pipeline and assert:
//!  - `ensure_sidecar` writes the parquet, populates the
//!    `nsight.nvtx_tree` view, and answers stack-at-T correctly via SQL.
//!  - A fresh `Trace::open` over the same parquetdir reuses the
//!    existing sidecar (no rebuild).
//!  - Rewriting a child parquet file invalidates the sidecar on the
//!    next `ensure_sidecar` call.

use anyhow::{Context, Result};
use duckdb::{Connection, params};
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::Duration;
use tempfile::TempDir;
use veloq_nsys_data::{Trace, nvtx_tree};

struct Fixture {
    path: PathBuf,
    _dir: TempDir,
}

impl Fixture {
    fn path(&self) -> &Path {
        &self.path
    }
}

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
        )?;
    }
    Ok(Fixture {
        path: pqtdir,
        _dir: dir,
    })
}

/// Synthetic NVTX-bearing trace:
///   tid 7, default domain:
///     rowid=1 outer    [0, 1000)
///     rowid=2 mid      [100, 800)   (parent = outer)
///     rowid=3 inner    [200, 300)   (parent = mid)
///   tid 8, default domain:
///     rowid=4 other    [500, 700)
fn nvtx_v3() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    conn.execute_batch(
        r#"
        CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT);
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS (
            rowid       BIGINT,
            start       BIGINT,
            "end"       BIGINT,
            eventType   BIGINT,
            globalTid   BIGINT,
            domainId    BIGINT,
            text        TEXT,
            textId      BIGINT
        );
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
        "#,
    )?;
    for (k, v) in [
        ("EXPORT_SCHEMA_VERSION_MAJOR", "3"),
        ("EXPORT_SCHEMA_VERSION_MINOR", "22"),
        ("EXPORT_SCHEMA_VERSION_MICRO", "1"),
    ] {
        conn.execute(
            "INSERT INTO META_DATA_EXPORT (name, value) VALUES (?, ?)",
            params![k, v],
        )?;
    }

    // Use inline `text` to avoid stringId management - the
    // `COALESCE(text, StringIds.value, '')` ladder will pick `text`
    // first.
    let events: &[(i64, i64, i64, i64, &str)] = &[
        (1, 0, 1000, 7, "outer"),
        (2, 100, 800, 7, "mid"),
        (3, 200, 300, 7, "inner"),
        (4, 500, 700, 8, "other"),
    ];
    for (rowid, start, end, tid, name) in events {
        conn.execute(
            "INSERT INTO NVTX_EVENTS (rowid, start, \"end\", eventType, globalTid, domainId, text, textId) \
             VALUES (?, ?, ?, 59, ?, ?, ?, NULL)",
            params![*rowid, *start, *end, *tid, 0i64, *name],
        )?;
    }

    // One dummy kernel so adapter detection has something to anchor.
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
         (start, \"end\", deviceId, contextId, streamId, shortName, demangledName, \
          gridX, gridY, gridZ, blockX, blockY, blockZ, correlationId, \
          registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            250i64,
            260i64,
            0i32,
            0i64,
            1i64,
            None::<i64>,
            None::<i64>,
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

    finalize_to_pqtdir(&conn, dir)
}

/// Trace whose `ENUM_NSYS_EVENT_TYPE` maps `NvtxPushPopRange` to a
/// NON-default id (159, not the fallback 59) and also contains a NULL-id
/// row. The two NVTX ranges are emitted at eventType 159. Exercises the
/// by-name eventType resolution: the tree captures these
/// ranges ONLY if `collect_rows` resolved the range eventType from the
/// catalog (159) rather than the hardcoded fallback `{59,60,70,71}`, and
/// only if a NULL `id` row did not poison the whole resolution.
fn nvtx_custom_eventtype_catalog() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    conn.execute_batch(
        r#"
        CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT);
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        CREATE TABLE ENUM_NSYS_EVENT_TYPE (id BIGINT, name TEXT, label TEXT);
        CREATE TABLE NVTX_EVENTS (
            rowid BIGINT, start BIGINT, "end" BIGINT, eventType BIGINT,
            globalTid BIGINT, domainId BIGINT, text TEXT, textId BIGINT
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start BIGINT, "end" BIGINT, deviceId BIGINT, contextId BIGINT,
            streamId BIGINT, shortName BIGINT, demangledName BIGINT,
            gridX BIGINT, gridY BIGINT, gridZ BIGINT,
            blockX BIGINT, blockY BIGINT, blockZ BIGINT, correlationId BIGINT,
            registersPerThread BIGINT, staticSharedMemory BIGINT,
            dynamicSharedMemory BIGINT, globalPid BIGINT
        );
        INSERT INTO META_DATA_EXPORT VALUES
            ('EXPORT_SCHEMA_VERSION_MAJOR','3'),
            ('EXPORT_SCHEMA_VERSION_MINOR','22'),
            ('EXPORT_SCHEMA_VERSION_MICRO','1');
        -- NvtxPushPopRange relocated to 159; a NULL-id row must be skipped,
        -- not abort the whole resolve.
        INSERT INTO ENUM_NSYS_EVENT_TYPE VALUES
            (159,'NvtxPushPopRange','NvtxPushPopRange'),
            (NULL,'NvtxBogusNullId','NvtxBogusNullId'),
            (75,'NvtxDomainCreate','NvtxDomainCreate');
        INSERT INTO NVTX_EVENTS (rowid,start,"end",eventType,globalTid,domainId,text,textId) VALUES
            (1, 0,    1000, 159, 7, 0, 'outer', NULL),
            (2, 100,  800,  159, 7, 0, 'inner', NULL);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL
            (start,"end",deviceId,contextId,streamId,shortName,demangledName,
             gridX,gridY,gridZ,blockX,blockY,blockZ,correlationId,
             registersPerThread,staticSharedMemory,dynamicSharedMemory,globalPid)
            VALUES (250,260,0,0,1,NULL,NULL,1,1,1,128,1,1,42,32,0,0,0);
        "#,
    )?;
    finalize_to_pqtdir(&conn, dir)
}

/// M1 regression: range eventType is resolved BY NAME from
/// the trace's own `ENUM_NSYS_EVENT_TYPE` catalog (here 159), and a NULL
/// `id` row in that catalog is skipped rather than collapsing the whole
/// resolution back to the hardcoded fallback. If either failed, the
/// 159-typed ranges would be filtered out and the tree would be empty.
#[test]
fn collect_rows_resolves_range_eventtype_by_name_tolerating_null_ids() -> Result<()> {
    let fix = nvtx_custom_eventtype_catalog()?;
    let trace = Trace::open(fix.path())?;
    let tree = nvtx_tree::build_or_load(&trace)?;
    let mut names: Vec<&str> = tree.records().iter().map(|r| r.name.as_str()).collect();
    names.sort_unstable();
    assert_eq!(
        names,
        vec!["inner", "outer"],
        "ranges at the catalog's NvtxPushPopRange id (159) must be captured; \
         got {names:?} (empty => by-name resolution or NULL-id skip regressed)"
    );
    Ok(())
}

#[test]
fn ensure_sidecar_builds_and_view_answers_stack_at_t() -> Result<()> {
    let fix = nvtx_v3()?;
    let sidecar_path = nvtx_tree::sidecar_path_for(fix.path());
    assert!(!sidecar_path.exists(), "sidecar should not pre-exist");

    let trace = Trace::open(fix.path())?;
    let path = nvtx_tree::ensure_sidecar(&trace)?;
    assert!(path.exists(), "ensure_sidecar should write the parquet");

    // The view is registered onto `trace`'s connection - query it
    // for the stack at t=250 on tid=7. Outer/mid/inner all cover
    // that point, ordered by depth ASC.
    let mut stmt = trace.conn().prepare(
        "SELECT name FROM nsight.nvtx_tree \
         WHERE global_tid = 7 \
           AND start <= 250 \
           AND (\"end\" IS NULL OR \"end\" > 250) \
         ORDER BY depth ASC",
    )?;
    let names: Vec<String> = stmt
        .query_map([], |r| r.get::<_, String>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(names, vec!["outer", "mid", "inner"]);

    // Path materialization: inner's path is "outer/mid/inner".
    let mut stmt = trace
        .conn()
        .prepare("SELECT path FROM nsight.nvtx_tree WHERE range_id = 3")?;
    let inner_path: String = stmt.query_row([], |r| r.get(0))?;
    assert_eq!(inner_path, "outer/mid/inner");

    // tid 8 has a single sibling range - stack at t=550 is just "other".
    let mut stmt = trace.conn().prepare(
        "SELECT name FROM nsight.nvtx_tree \
         WHERE global_tid = 8 \
           AND start <= 550 \
           AND (\"end\" IS NULL OR \"end\" > 550) \
         ORDER BY depth ASC",
    )?;
    let names: Vec<String> = stmt
        .query_map([], |r| r.get::<_, String>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(names, vec!["other"]);

    Ok(())
}

#[test]
fn warm_reopen_does_not_rebuild_sidecar() -> Result<()> {
    let fix = nvtx_v3()?;
    let sidecar_path = nvtx_tree::sidecar_path_for(fix.path());

    // Cold build.
    {
        let trace = Trace::open(fix.path())?;
        let _ = nvtx_tree::ensure_sidecar(&trace)?;
    }
    assert!(sidecar_path.exists());
    let mtime_cold = std::fs::metadata(&sidecar_path)?.modified()?;

    sleep(Duration::from_millis(1100));

    // Reopen - `Trace::open` should attach the existing view without
    // rebuilding, and a fresh `ensure_sidecar` call should hit the
    // warm path (no rewrite).
    {
        let trace = Trace::open(fix.path())?;
        // View was attached at open time - query should succeed
        // without calling `ensure_sidecar` again.
        let row_count: i64 =
            trace
                .conn()
                .query_row("SELECT COUNT(*) FROM nsight.nvtx_tree", [], |r| r.get(0))?;
        assert_eq!(row_count, 4);

        let _ = nvtx_tree::ensure_sidecar(&trace)?;
    }

    let mtime_warm = std::fs::metadata(&sidecar_path)?.modified()?;
    assert_eq!(
        mtime_cold, mtime_warm,
        "warm reopen must not rewrite the sidecar parquet"
    );
    Ok(())
}

#[test]
fn child_parquet_rewrite_invalidates_sidecar() -> Result<()> {
    let fix = nvtx_v3()?;
    let sidecar_path = nvtx_tree::sidecar_path_for(fix.path());

    {
        let trace = Trace::open(fix.path())?;
        let _ = nvtx_tree::ensure_sidecar(&trace)?;
    }
    let mtime_cold = std::fs::metadata(&sidecar_path)?.modified()?;
    sleep(Duration::from_millis(1100));

    // Rewrite one child parquet file in place. Direct `_pqtdir/`
    // inputs must fingerprint child parquet metadata rather than the
    // directory inode.
    {
        let parquet = fix.path().join("NVTX_EVENTS.parquet");
        let parquet_lit = parquet.to_string_lossy().replace('\'', "''");
        let conn = Connection::open_in_memory().context("open rewrite DuckDB")?;
        conn.execute(
            &format!(r#"CREATE TABLE NVTX_EVENTS AS SELECT * FROM read_parquet('{parquet_lit}')"#),
            [],
        )?;
        conn.execute(
            "INSERT INTO NVTX_EVENTS (rowid, start, \"end\", eventType, globalTid, domainId, text, textId) \
             VALUES (?, ?, ?, 59, ?, ?, ?, NULL)",
            params![5i64, 900i64, 950i64, 7i64, 0i64, "late"],
        )?;
        conn.execute(
            &format!(r#"COPY NVTX_EVENTS TO '{parquet_lit}' (FORMAT PARQUET)"#),
            [],
        )?;
    }

    {
        let trace = Trace::open(fix.path())?;
        let _ = nvtx_tree::ensure_sidecar(&trace)?;
    }
    let mtime_after = std::fs::metadata(&sidecar_path)?.modified()?;
    assert!(
        mtime_after > mtime_cold,
        "child parquet rewrite must trigger sidecar rebuild"
    );
    Ok(())
}

#[test]
fn open_without_sidecar_does_not_register_view() -> Result<()> {
    // `Trace::open` must never auto-build the sidecar - only attach
    // a view when one already exists on disk. Confirm querying the
    // view fails when the sidecar is absent.
    let fix = nvtx_v3()?;
    let trace = Trace::open(fix.path())?;
    let err = match trace.conn().query_row::<i64, _, _>(
        "SELECT COUNT(*) FROM nsight.nvtx_tree",
        [],
        |r| r.get(0),
    ) {
        Ok(_) => anyhow::bail!("query should fail when sidecar is absent"),
        Err(err) => err,
    };
    let msg = err.to_string().to_ascii_lowercase();
    assert!(
        msg.contains("nvtx_tree") || msg.contains("does not exist") || msg.contains("not found"),
        "error should reference the missing view: {msg}"
    );
    Ok(())
}
