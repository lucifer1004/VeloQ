//! Canonical fixture schema regression guard.
//!
//! [`fixture::setup_canonical_schema`] is the single chokepoint
//! every NSys-shape fixture should funnel through; the tests here
//! pin two invariants:
//!
//! 1. **The 24 canonical tables actually exist after setup** — guards
//!    against an entry getting dropped from `CANONICAL_TABLES` by
//!    accident.
//! 2. **No two entries collide on the same table name** — guards
//!    against a future entry duplicating a name; `setup_canonical_schema`
//!    would bail with "table already exists" but the failure mode
//!    would be runtime-only, not at definition time.
//! 3. **The `_minus` opt-out filters correctly** — single name,
//!    multiple names, unknown name (silent no-op).
//!
//! Also smoke-tests that the resulting database is openable by
//! `veloq_nsys_data::Trace::open` — the real-world consumer.

mod fixture;

use anyhow::Result;
use duckdb::Connection;
use std::collections::HashSet;

/// Names of every user-created table in the default DuckDB schema.
fn list_user_tables(conn: &Connection) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(
        "SELECT table_name FROM information_schema.tables \
         WHERE table_schema = 'main' AND table_type = 'BASE TABLE' \
         ORDER BY table_name",
    )?;
    let mut rows = stmt.query([])?;
    let mut out = Vec::new();
    while let Some(r) = rows.next()? {
        out.push(r.get::<_, String>(0)?);
    }
    Ok(out)
}

fn table_count(conn: &Connection, name: Option<&str>) -> Result<i64> {
    let sql = match name {
        Some(_) => {
            "SELECT COUNT(*) FROM information_schema.tables \
             WHERE table_schema = 'main' AND table_type = 'BASE TABLE' \
             AND table_name = ?1"
        }
        None => {
            "SELECT COUNT(*) FROM information_schema.tables \
             WHERE table_schema = 'main' AND table_type = 'BASE TABLE'"
        }
    };
    let mut stmt = conn.prepare(sql)?;
    let count: i64 = match name {
        Some(n) => stmt.query_row([n], |r| r.get(0))?,
        None => stmt.query_row([], |r| r.get(0))?,
    };
    Ok(count)
}

#[test]
fn canonical_schema_creates_every_listed_table() -> Result<()> {
    let conn = Connection::open_in_memory()?;
    fixture::setup_canonical_schema(&conn)?;
    let actual: HashSet<String> = list_user_tables(&conn)?.into_iter().collect();
    let expected: HashSet<String> = fixture::CANONICAL_TABLES
        .iter()
        .map(|(n, _)| (*n).to_string())
        .collect();
    let missing: Vec<&String> = expected.difference(&actual).collect();
    assert!(missing.is_empty(), "canonical schema missed: {:?}", missing);
    Ok(())
}

#[test]
fn canonical_table_names_are_distinct() -> Result<()> {
    let names: Vec<&str> = fixture::CANONICAL_TABLES.iter().map(|(n, _)| *n).collect();
    let unique: HashSet<&str> = names.iter().copied().collect();
    assert_eq!(
        names.len(),
        unique.len(),
        "duplicate table name in CANONICAL_TABLES: {:?}",
        names
    );
    Ok(())
}

#[test]
fn minus_filter_drops_named_table() -> Result<()> {
    let conn = Connection::open_in_memory()?;
    fixture::setup_canonical_schema_minus(&conn, &["NVTX_EVENTS"])?;
    assert_eq!(
        table_count(&conn, Some("NVTX_EVENTS"))?,
        0,
        "NVTX_EVENTS should be excluded"
    );
    assert_eq!(
        table_count(&conn, Some("CUPTI_ACTIVITY_KIND_KERNEL"))?,
        1,
        "non-excluded tables should still exist"
    );
    Ok(())
}

#[test]
fn minus_filter_drops_multiple_tables() -> Result<()> {
    let conn = Connection::open_in_memory()?;
    fixture::setup_canonical_schema_minus(&conn, &["NVTX_EVENTS", "OSRT_API"])?;
    for t in ["NVTX_EVENTS", "OSRT_API"] {
        assert_eq!(table_count(&conn, Some(t))?, 0, "{t} should be excluded");
    }
    Ok(())
}

#[test]
fn minus_filter_unknown_name_is_silent_no_op() -> Result<()> {
    let conn = Connection::open_in_memory()?;
    fixture::setup_canonical_schema_minus(&conn, &["NOT_A_REAL_TABLE"])?;
    let n = table_count(&conn, None)?;
    let expected = fixture::CANONICAL_TABLES.len() as i64;
    assert_eq!(n, expected, "unknown exclude name must not affect output");
    Ok(())
}

#[test]
fn canonical_schema_opens_via_trace() -> Result<()> {
    // End-to-end: every fixture's canonical schema, COPYed to parquet
    // and reopened through `Trace::open`. A canonical-schema parquetdir
    // (even with no rows) must open without erroring on a missing-table
    // or unsupported-adapter path.
    let dir = tempfile::tempdir()?;
    let conn = Connection::open_in_memory()?;
    fixture::setup_canonical_schema(&conn)?;
    let pqtdir = dir.path().join("test_pqtdir");
    std::fs::create_dir_all(&pqtdir)?;
    for (name, _) in fixture::CANONICAL_TABLES {
        let out = pqtdir.join(format!("{name}.parquet"));
        let out_lit = out.to_string_lossy().replace('\'', "''");
        conn.execute(
            &format!(r#"COPY (SELECT * FROM "{name}") TO '{out_lit}' (FORMAT PARQUET)"#),
            [],
        )?;
    }
    let _trace = veloq_nsys_data::Trace::open(&pqtdir)?;
    Ok(())
}
