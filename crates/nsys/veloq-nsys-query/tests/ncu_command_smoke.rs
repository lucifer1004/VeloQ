//! Smoke tests for generating an NCU rerun command from NSys metadata.

mod fixture;

use anyhow::{Context, Result};
use duckdb::{Connection, params};
use std::path::PathBuf;
use tempfile::TempDir;
use veloq_nsys_query::RowId;
use veloq_nsys_query::ncu_command::{EnvPolicy, NcuCommandRequest};

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

fn build_trace() -> Result<(TempDir, PathBuf)> {
    let dir = tempfile::tempdir().context("tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    conn.execute_batch(
        r#"
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        CREATE TABLE META_DATA_CAPTURE (name TEXT, value TEXT);
        CREATE TABLE PROCESSES (globalPid BIGINT, pid BIGINT, name TEXT);
        "#,
    )
    .context("create ncu-command fixture schema")?;
    conn.execute_batch(fixture::KERNEL_TABLE_SQL)
        .context("create kernel table")?;

    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1i64, "target_kernel"],
    )?;

    for (start, end) in [(100i64, 150i64), (200i64, 260i64)] {
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
             (start, \"end\", deviceId, contextId, streamId, \
              shortName, demangledName, gridX, gridY, gridZ, \
              blockX, blockY, blockZ, correlationId, \
              registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                start, end, 0i32, 0i64, 7i64, 1i64, 1i64, 2i64, 1i64, 1i64, 128i64, 1i64, 1i64,
                9i64, 32i64, 0i64, 0i64, 42i64,
            ],
        )?;
    }

    conn.execute(
        "INSERT INTO PROCESSES (globalPid, pid, name) VALUES (?, ?, ?)",
        params![42i64, 1234i64, "/opt/work/app"],
    )?;
    for (name, value) in [
        ("PROCESS_0:COMMAND", "/usr/bin/app"),
        ("PROCESS_0:ARGUMENT_0", "--size"),
        ("PROCESS_0:ARGUMENT_1", "128"),
        ("PROCESS_0:WORKING_DIR", "/workspace/case"),
        (
            "PROCESS_0:ENVIRONMENT_VARIABLE",
            "CUDA_VISIBLE_DEVICES=\"0\"",
        ),
        (
            "PROCESS_0:ENVIRONMENT_VARIABLE",
            "SECRET_TOKEN=\"do-not-emit\"",
        ),
    ] {
        conn.execute(
            "INSERT INTO META_DATA_CAPTURE (name, value) VALUES (?, ?)",
            params![name, value],
        )?;
    }
    finalize_to_pqtdir(&conn, dir)
}

#[test]
fn generates_name_skip_ncu_script_from_selected_kernel() -> Result<()> {
    let (_trace_dir, trace) = build_trace()?;
    let response = veloq_nsys_query::ncu_command::run(
        &trace,
        NcuCommandRequest {
            row_id: "kernel:2".parse::<RowId>()?,
            env_policy: EnvPolicy::Safe,
        },
    )?;

    assert_eq!(response.source_event.row_id.to_string(), "kernel:2");
    assert_eq!(response.selector.kernel_name_base, "function");
    assert_eq!(response.selector.kernel_name, "target_kernel");
    assert_eq!(response.selector.launch_skip, 1);
    assert_eq!(response.selector.launch_count, 1);
    assert_eq!(response.launch_recipe.command, "/usr/bin/app");
    assert_eq!(
        response.launch_recipe.args,
        vec!["--size".to_string(), "128".to_string()]
    );
    assert_eq!(response.launch_recipe.emitted_env_count, 1);
    assert_eq!(response.launch_recipe.redacted_env_count, 1);
    assert!(
        response.ncu.argv.windows(2).any(|w| {
            w.first().is_some_and(|v| v == "--kernel-name")
                && w.get(1).is_some_and(|v| v == "target_kernel")
        }),
        "missing kernel selector in argv: {:?}",
        response.ncu.argv
    );
    assert!(response.script.starts_with("#!/usr/bin/env bash"));
    assert!(response.script.contains("CUDA_VISIBLE_DEVICES=0"));
    assert!(!response.script.contains("do-not-emit"));
    assert!(response.script.contains("--launch-skip \\\n  1"));
    Ok(())
}

#[test]
fn works_when_kernel_table_is_parquet_cached() -> Result<()> {
    let (_trace_dir, trace) = build_trace()?;
    {
        let prepared = veloq_nsys_data::Trace::open(&trace)?;
        assert!(
            !prepared.tables().is_empty(),
            "fixture should produce at least one parquet table"
        );
    }

    let response = veloq_nsys_query::ncu_command::run(
        &trace,
        NcuCommandRequest {
            row_id: "kernel:2".parse::<RowId>()?,
            env_policy: EnvPolicy::Safe,
        },
    )?;
    assert_eq!(response.selector.launch_skip, 1);
    assert_eq!(response.selector.kernel_name, "target_kernel");
    Ok(())
}
