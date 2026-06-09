use super::{
    assert_error_code, build_graph_replay_trace, build_minimal_trace, finalize_to_pqtdir, run_veloq,
};
use anyhow::{Context, Result};
use duckdb::{Connection, params};
use serde_json::Value;
use std::path::PathBuf;
use tempfile::TempDir;

fn build_ncu_command_trace() -> Result<(TempDir, PathBuf)> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    conn.execute_batch(
        r#"
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        CREATE TABLE META_DATA_CAPTURE (name TEXT, value TEXT);
        CREATE TABLE PROCESSES (globalPid BIGINT, pid BIGINT, name TEXT);
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
            globalPid BIGINT,
            graphId BIGINT,
            graphNodeId BIGINT
        );
        "#,
    )
    .context("create ncu-command trace schema")?;
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
    ] {
        conn.execute(
            "INSERT INTO META_DATA_CAPTURE (name, value) VALUES (?, ?)",
            params![name, value],
        )?;
    }
    let pqtdir = finalize_to_pqtdir(&conn, &dir)?;
    Ok((dir, pqtdir))
}

#[test]
fn nsys_ncu_command_emits_json_recipe() -> Result<()> {
    let (_trace_dir, trace) = build_ncu_command_trace()?;
    let out = run_veloq(["nsys", "ncu-command", &trace.to_string_lossy(), "kernel:2"])?;
    assert!(
        out.status.success(),
        "ncu-command failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value =
        serde_json::from_slice(&out.stdout).context("ncu-command stdout must be valid JSON")?;
    assert_eq!(
        v.get("command").and_then(Value::as_str),
        Some("nsys.ncu-command"),
    );
    assert_eq!(
        v.pointer("/data/selector/launch_skip")
            .and_then(Value::as_i64),
        Some(1),
    );
    assert_eq!(
        v.pointer("/data/selector/kernel_name")
            .and_then(Value::as_str),
        Some("target_kernel"),
    );
    assert!(
        v.pointer("/data/script")
            .and_then(Value::as_str)
            .is_some_and(|s| s.contains("exec \\\n  env \\\n")),
        "script missing executable command: {v}"
    );
    Ok(())
}

#[test]
fn nsys_ncu_command_print_emits_only_shell_script() -> Result<()> {
    let (_trace_dir, trace) = build_ncu_command_trace()?;
    let out = run_veloq([
        "nsys",
        "ncu-command",
        &trace.to_string_lossy(),
        "kernel:2",
        "--print",
    ])?;
    assert!(
        out.status.success(),
        "ncu-command --print failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.starts_with("#!/usr/bin/env bash"));
    assert!(stdout.contains("--launch-count \\\n  1"));
    assert!(
        serde_json::from_slice::<Value>(&out.stdout).is_err(),
        "--print stdout must not be a JSON envelope"
    );
    Ok(())
}

#[test]
fn nsys_ncu_command_print_error_keeps_stdout_empty() -> Result<()> {
    let (_trace_dir, trace) = build_minimal_trace()?;
    let out = run_veloq([
        "nsys",
        "ncu-command",
        &trace.to_string_lossy(),
        "kernel:1",
        "--print",
    ])?;
    assert!(
        !out.status.success(),
        "ncu-command --print should fail when metadata is absent"
    );
    assert!(
        out.stdout.is_empty(),
        "--print failures must not write JSON stdout: {}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).starts_with("veloq:"),
        "stderr must carry the human-readable error"
    );
    Ok(())
}

#[test]
fn nsys_ncu_command_print_parse_error_keeps_stdout_empty() -> Result<()> {
    let out = run_veloq(["nsys", "ncu-command", "--print"])?;
    assert!(
        !out.status.success(),
        "missing ncu-command args should fail"
    );
    assert!(
        out.stdout.is_empty(),
        "--print parse failures must not write a JSON envelope: {}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(
        !out.stderr.is_empty(),
        "clap should render the parse error on stderr"
    );
    Ok(())
}

#[test]
fn nsys_ncu_command_rejects_table_format_without_print() -> Result<()> {
    let (_trace_dir, trace) = build_ncu_command_trace()?;
    let out = run_veloq([
        "nsys",
        "ncu-command",
        &trace.to_string_lossy(),
        "kernel:2",
        "--format",
        "table",
    ])?;
    assert!(
        !out.status.success(),
        "ncu-command table format should fail"
    );
    let v = assert_error_code(&out, "nsys.command.ncu-command-unsupported-format")?;
    assert_eq!(
        v.get("command").and_then(Value::as_str),
        Some("nsys.ncu-command"),
    );
    assert!(
        v.pointer("/error/message")
            .and_then(Value::as_str)
            .is_some_and(|s| s.contains("--print")),
        "error should point users to --print: {v}"
    );
    Ok(())
}

#[test]
fn nsys_query_ncu_command_unknown_env_has_specific_error_code() -> Result<()> {
    let (_trace_dir, trace) = build_ncu_command_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq([
        "nsys",
        "ncu-command",
        trace.as_str(),
        "kernel:2",
        "--env",
        "weird",
    ])?;
    let v = assert_error_code(&out, "nsys.query.ncu-command-unknown-env")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("weird") && message.contains("safe"),
        "message should name rejected and expected env policies: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_ncu_command_row_id_kind_has_specific_error_code() -> Result<()> {
    let (_trace_dir, trace) = build_ncu_command_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["nsys", "ncu-command", trace.as_str(), "runtime:1"])?;
    let v = assert_error_code(&out, "nsys.query.ncu-command-row-id-kind")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("runtime:1") && message.contains("kernel"),
        "message should point users at a kernel row id: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_ncu_command_kernel_table_missing_has_specific_error_code() -> Result<()> {
    let (_trace_dir, trace) = build_graph_replay_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["nsys", "ncu-command", trace.as_str(), "kernel:1"])?;
    let v = assert_error_code(&out, "nsys.query.ncu-command-kernel-table-missing")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("CUPTI_ACTIVITY_KIND_KERNEL"),
        "message should name the missing kernel table: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_ncu_command_metadata_missing_has_specific_error_code() -> Result<()> {
    let (_trace_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["nsys", "ncu-command", trace.as_str(), "kernel:1"])?;
    let v = assert_error_code(&out, "nsys.query.ncu-command-metadata-table-missing")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("META_DATA_CAPTURE") && message.contains("command"),
        "message should name the missing launch metadata table: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_ncu_command_kernel_not_found_has_specific_error_code() -> Result<()> {
    let (_trace_dir, trace) = build_ncu_command_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["nsys", "ncu-command", trace.as_str(), "kernel:99"])?;
    let v = assert_error_code(&out, "nsys.query.ncu-command-kernel-not-found")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("kernel:99") && message.contains("not found"),
        "message should name the missing kernel row id: {message}"
    );
    Ok(())
}
