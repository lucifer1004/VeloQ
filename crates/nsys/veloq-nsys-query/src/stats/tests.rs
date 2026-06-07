use super::hydrate::hydrate_stats_rows;
use anyhow::Result;
use duckdb::Connection;
use std::path::PathBuf;
use tempfile::TempDir;
use veloq_core::VeloqDiagnostic;
use veloq_nsys_data::Trace;

fn parquet_fixture(tables: Vec<(&str, &str, Vec<&str>)>) -> Result<(TempDir, PathBuf)> {
    let dir = tempfile::tempdir()?;
    let pqtdir = dir.path().join("test_pqtdir");
    std::fs::create_dir_all(&pqtdir)?;
    let conn = Connection::open_in_memory()?;
    for (_, ddl, inserts) in &tables {
        conn.execute_batch(ddl)?;
        for insert in inserts {
            conn.execute_batch(insert)?;
        }
    }
    for (table, _, _) in &tables {
        let out = pqtdir.join(format!("{table}.parquet"));
        let out_lit = out.to_string_lossy().replace('\'', "''");
        conn.execute(
            &format!(r#"COPY (SELECT * FROM "{table}") TO '{out_lit}' (FORMAT PARQUET)"#),
            [],
        )?;
    }
    Ok((dir, pqtdir))
}

fn minimal_trace() -> Result<(TempDir, Trace)> {
    let (dir, pqtdir) = parquet_fixture(vec![(
        "CUPTI_ACTIVITY_KIND_KERNEL",
        r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start BIGINT, "end" BIGINT)"#,
        Vec::new(),
    )])?;
    let trace = Trace::open(&pqtdir)?;
    Ok((dir, trace))
}

fn stats_hydration_sql(count_expr: &str) -> String {
    format!(
        "SELECT \
         CAST(NULL AS VARCHAR) AS name, \
         CAST(NULL AS VARCHAR) AS short_name, \
         'kernel' AS kind, \
         CAST(NULL AS INTEGER) AS device_id, \
         CAST(NULL AS BIGINT) AS context_id, \
         CAST(NULL AS BIGINT) AS stream_id, \
         CAST(NULL AS BIGINT) AS graph_id, \
         CAST(NULL AS BIGINT) AS graph_node_id, \
         {count_expr} AS count, \
         1::BIGINT AS total_ns, \
         1::BIGINT AS avg_ns, \
         1::BIGINT AS min_ns, \
         1::BIGINT AS max_ns, \
         1::BIGINT AS p50_ns, \
         1::BIGINT AS p95_ns, \
         1::BIGINT AS p99_ns, \
         CAST(NULL AS BIGINT) AS bytes_total, \
         CAST(NULL AS DOUBLE) AS gbps, \
         CAST(NULL AS BIGINT) AS event_type, \
         CAST(NULL AS VARCHAR) AS nvtx_style, \
         CAST(NULL AS BIGINT) AS nvtx_parent_rowid, \
         CAST(NULL AS VARCHAR) AS nvtx_parent_name, \
         CAST(NULL AS VARCHAR) AS nvtx_path, \
         CAST(NULL AS BIGINT) AS nvtx_domain_id, \
         CAST(NULL AS BIGINT) AS nvtx_domain_pid, \
         CAST(NULL AS BIGINT) AS grid_x, \
         CAST(NULL AS BIGINT) AS grid_y, \
         CAST(NULL AS BIGINT) AS grid_z, \
         CAST(NULL AS BIGINT) AS block_x, \
         CAST(NULL AS BIGINT) AS block_y, \
         CAST(NULL AS BIGINT) AS block_z, \
         1::BIGINT AS scope_total_ns, \
         1::BIGINT AS scope_total_count, \
         1::BIGINT AS scope_total_groups"
    )
}

#[test]
fn hydrate_stats_rows_prepare_error_is_typed() -> Result<()> {
    let (_dir, trace) = minimal_trace()?;

    let err = match hydrate_stats_rows(
        &trace,
        "SELECT * FROM",
        &[],
        false,
        None,
        &std::collections::HashMap::new(),
    ) {
        Ok((rows, _)) => anyhow::bail!(
            "malformed stats SQL should not hydrate successfully: {} rows",
            rows.len()
        ),
        Err(err) => err,
    };

    assert_eq!(err.code().as_str(), "nsys.query.sql-prepare");
    assert!(matches!(
        err,
        crate::NsysQueryError::Sql {
            phase: crate::SqlPhase::Prepare,
            ..
        }
    ));
    Ok(())
}

#[test]
fn hydrate_stats_rows_query_error_is_typed() -> Result<()> {
    let (_dir, trace) = minimal_trace()?;

    let err = match hydrate_stats_rows(
        &trace,
        "SELECT ? AS name",
        &[],
        false,
        None,
        &std::collections::HashMap::new(),
    ) {
        Ok((rows, _)) => anyhow::bail!(
            "unbound stats SQL parameter should not hydrate successfully: {} rows",
            rows.len()
        ),
        Err(err) => err,
    };

    assert_eq!(err.code().as_str(), "nsys.query.sql-query");
    assert!(matches!(
        err,
        crate::NsysQueryError::Sql {
            phase: crate::SqlPhase::Query,
            ..
        }
    ));
    Ok(())
}

#[test]
fn hydrate_stats_rows_read_error_is_typed() -> Result<()> {
    let (_dir, trace) = minimal_trace()?;
    let sql = stats_hydration_sql("'not-a-count'");

    let err = match hydrate_stats_rows(
        &trace,
        &sql,
        &[],
        false,
        None,
        &std::collections::HashMap::new(),
    ) {
        Ok((rows, _)) => anyhow::bail!(
            "malformed stats row should not hydrate successfully: {} rows",
            rows.len()
        ),
        Err(err) => err,
    };

    assert_eq!(err.code().as_str(), "nsys.query.sql-read");
    assert!(matches!(
        err,
        crate::NsysQueryError::Sql {
            phase: crate::SqlPhase::Read,
            ..
        }
    ));
    Ok(())
}
