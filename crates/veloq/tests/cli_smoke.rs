//! End-to-end smoke tests for the `veloq` binary.
//!
//! These tests spawn the actual binary (`env!("CARGO_BIN_EXE_veloq")`)
//! and validate the contracts spelled out in AGENTS.md §Design pillars:
//!
//!   1. Every success emits a JSON envelope on stdout with the keys
//!      `schema`, `command`, `trace`, `data`.
//!   2. Every handled failure emits a non-zero exit and a
//!      `{schema, command, trace, error}` envelope on stdout. JSON mode
//!      keeps stderr quiet; CSV/table mode also mirrors a human-readable
//!      line to stderr.
//!
//! The CLI is the glue layer between clap, the query crate, and the
//! JSON renderer — exactly where silent regressions hide. The lower
//! crates already have their own integration tests; these tests are
//! deliberately thin checks that the wiring works at all.
//!
//! NB: integration tests for a binary-only crate get the built path
//! via `env!("CARGO_BIN_EXE_<bin>")`, a Cargo built-in. No `assert_cmd`
//! dep needed.

use anyhow::{Context, Result, anyhow};
use duckdb::{Connection, params};
use serde_json::Value;
use std::path::PathBuf;
use std::process::{Command, Output};
use tempfile::TempDir;

/// COPY every user-created table in the in-memory DuckDB connection
/// out to `<dir>/test_pqtdir/<TABLE>.parquet` and return the
/// resulting parquetdir path. Mirrors `tests/fixture.rs::finalize_to_pqtdir`
/// but local to this binary-smoke crate so it doesn't pull in the
/// query crate's dev-dep graph.
fn finalize_to_pqtdir(conn: &Connection, dir: &TempDir) -> Result<PathBuf> {
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
    Ok(pqtdir)
}

/// Path to the freshly-built `veloq` binary, courtesy of Cargo.
fn veloq_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_veloq"))
}

/// Run `veloq <args>` and capture stdout / stderr / exit status.
fn run_veloq<I, S>(args: I) -> Result<Output>
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    Command::new(veloq_bin())
        .args(args)
        .output()
        .context("spawn veloq binary")
}

fn assert_error_code(out: &Output, expected: &str) -> Result<Value> {
    assert!(
        !out.status.success(),
        "expected non-zero exit for {expected}"
    );
    let v: Value =
        serde_json::from_slice(&out.stdout).context("error stdout must be valid JSON")?;
    assert_eq!(
        v.pointer("/error/code").and_then(Value::as_str),
        Some(expected),
        "unexpected error code for {expected}: {v}"
    );
    Ok(v)
}

fn run_veloq_with_env<I, S, E, K, V>(args: I, envs: E) -> Result<Output>
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
    E: IntoIterator<Item = (K, V)>,
    K: AsRef<std::ffi::OsStr>,
    V: AsRef<std::ffi::OsStr>,
{
    let mut cmd = Command::new(veloq_bin());
    cmd.args(args);
    for (key, value) in envs {
        cmd.env(key, value);
    }
    cmd.output().context("spawn veloq binary")
}

fn run_veloq_without_unstable<I, S>(args: I) -> Result<Output>
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    Command::new(veloq_bin())
        .args(args)
        .env_remove("VELOQ_UNSTABLE")
        .output()
        .context("spawn veloq binary")
}

/// Build a minimal parquetdir trace that `summary` can open: one
/// kernel row gives the trace a primary origin. Returns a `(TempDir,
/// parquetdir path)` pair — the `TempDir` keeps the underlying
/// directory alive for the duration of the test.
///
/// **Cross-crate sync**: the `CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL`
/// schema below mirrors `veloq-nsys-query/tests/fixture.rs::KERNEL_TABLE_SQL`
/// — if you add a column there, mirror it here. Both must agree with
/// the columns `Trace::open` + `summary::run` actually read.
fn build_minimal_trace() -> Result<(TempDir, PathBuf)> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    conn.execute_batch(
        r#"
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT);
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
    .context("create minimal trace schema")?;
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1i64, "smoke_kernel"],
    )?;
    conn.execute(
        "INSERT INTO META_DATA_EXPORT (name, value) VALUES (?, ?), (?, ?), (?, ?)",
        params![
            "EXPORT_SCHEMA_VERSION_MAJOR",
            "3",
            "EXPORT_SCHEMA_VERSION_MINOR",
            "0",
            "EXPORT_SCHEMA_VERSION_MICRO",
            "0",
        ],
    )?;
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
         (start, \"end\", deviceId, contextId, streamId, \
          shortName, demangledName, gridX, gridY, gridZ, \
          blockX, blockY, blockZ, correlationId, \
          registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            100_000_000i64,
            101_000_000i64,
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
            1i64,
            32i64,
            0i64,
            0i64,
            0i64,
        ],
    )?;
    let pqtdir = finalize_to_pqtdir(&conn, &dir)?;
    Ok((dir, pqtdir))
}

fn build_graph_replay_trace() -> Result<(TempDir, PathBuf)> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    conn.execute_batch(
        r#"
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT);
        CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO (
            deviceId BIGINT,
            contextId BIGINT,
            processId BIGINT
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start BIGINT,
            "end" BIGINT,
            globalTid BIGINT,
            correlationId BIGINT,
            nameId BIGINT
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (
            start BIGINT,
            "end" BIGINT,
            deviceId BIGINT,
            contextId BIGINT,
            streamId BIGINT,
            correlationId BIGINT,
            globalPid BIGINT,
            graphId BIGINT,
            graphExecId BIGINT
        );
        "#,
    )
    .context("create graph replay trace schema")?;
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1i64, "cudaGraphLaunch_v10000"],
    )?;
    conn.execute(
        "INSERT INTO META_DATA_EXPORT (name, value) VALUES (?, ?), (?, ?), (?, ?)",
        params![
            "EXPORT_SCHEMA_VERSION_MAJOR",
            "3",
            "EXPORT_SCHEMA_VERSION_MINOR",
            "0",
            "EXPORT_SCHEMA_VERSION_MICRO",
            "0",
        ],
    )?;
    conn.execute(
        "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO (deviceId, contextId, processId) \
         VALUES (?, ?, ?)",
        params![0i64, 1i64, 12345i64],
    )?;
    let global_tid = 12345i64 << 24;
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME \
         (start, \"end\", globalTid, correlationId, nameId) \
         VALUES (?, ?, ?, ?, ?)",
        params![99_950_000i64, 100_000_000i64, global_tid, 7100i64, 1i64],
    )?;
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE \
         (start, \"end\", deviceId, contextId, streamId, correlationId, globalPid, graphId, graphExecId) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            100_000_000i64,
            110_000_000i64,
            0i64,
            1i64,
            23i64,
            7100i64,
            0i64,
            42i64,
            43i64,
        ],
    )?;
    let pqtdir = finalize_to_pqtdir(&conn, &dir)?;
    Ok((dir, pqtdir))
}

/// Build the `.nsys-rep` + generated `<report>.veloq/parquetdir/`
/// shape that appears after the first run on a report.
fn build_generated_parquetdir_alias() -> Result<(TempDir, PathBuf, PathBuf)> {
    let (dir, direct_pqtdir) = build_minimal_trace()?;
    let report = dir.path().join("trace.nsys-rep");
    std::fs::write(&report, b"source").context("write report placeholder")?;
    let generated_pqtdir = veloq_core::artifact_dir_for(&report).join("parquetdir");
    copy_dir(&direct_pqtdir, &generated_pqtdir)?;
    Ok((dir, report, generated_pqtdir))
}

fn copy_dir(src: &std::path::Path, dst: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(dst).with_context(|| format!("create {}", dst.display()))?;
    for entry in std::fs::read_dir(src).with_context(|| format!("read {}", src.display()))? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir(&src_path, &dst_path)?;
        } else {
            std::fs::copy(&src_path, &dst_path).with_context(|| {
                format!("copy {} to {}", src_path.display(), dst_path.display())
            })?;
        }
    }
    Ok(())
}

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
fn help_exits_zero() -> Result<()> {
    let out = run_veloq(["--help"])?;
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Usage:"),
        "expected clap usage line in stdout; got: {stdout}"
    );
    Ok(())
}

#[test]
fn schema_endpoint_emits_envelope_without_trace() -> Result<()> {
    let out = run_veloq(["schema", "summary"])?;
    assert!(
        out.status.success(),
        "schema should succeed without a trace"
    );
    let v: Value =
        serde_json::from_slice(&out.stdout).context("schema stdout must be valid JSON")?;
    assert_eq!(
        v.get("command").and_then(Value::as_str),
        Some("nsys.schema"),
    );
    assert_eq!(
        v.get("source")
            .and_then(|s| s.get("kind"))
            .and_then(Value::as_str),
        Some("nsys"),
    );
    assert_eq!(
        v.get("source")
            .and_then(|s| s.get("version"))
            .and_then(Value::as_str),
        Some("v1"),
    );
    assert!(v.get("schema").is_some(), "envelope missing `schema` key");
    assert!(v.get("data").is_some(), "envelope missing `data` payload");
    assert!(
        v.get("trace").is_none(),
        "schema envelope must omit `trace` (meta endpoint): {v}",
    );
    Ok(())
}

#[test]
fn graph_replays_schema_endpoint_is_registered() -> Result<()> {
    let out = run_veloq(["schema", "graph-replays"])?;
    assert!(out.status.success());
    let v: Value = serde_json::from_slice(&out.stdout)?;
    assert_eq!(
        v.pointer("/data/target").and_then(Value::as_str),
        Some("graph-replays")
    );
    assert!(v.pointer("/data/schema").is_some());
    Ok(())
}

#[test]
fn nsys_namespace_routes_default_source_verbs() -> Result<()> {
    let (_trace_dir, trace) = build_minimal_trace()?;
    let out = run_veloq(["nsys", "summary", &trace.to_string_lossy()])?;
    assert!(
        out.status.success(),
        "nsys summary failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value =
        serde_json::from_slice(&out.stdout).context("nsys summary stdout must be valid JSON")?;
    assert_eq!(
        v.get("command").and_then(Value::as_str),
        Some("nsys.summary"),
    );
    assert_eq!(
        v.get("source")
            .and_then(|s| s.get("kind"))
            .and_then(Value::as_str),
        Some("nsys"),
    );

    let schema = run_veloq(["nsys", "schema", "summary"])?;
    assert!(
        schema.status.success(),
        "nsys schema failed: stderr={}",
        String::from_utf8_lossy(&schema.stderr)
    );
    let v: Value =
        serde_json::from_slice(&schema.stdout).context("nsys schema stdout must be valid JSON")?;
    assert_eq!(
        v.get("command").and_then(Value::as_str),
        Some("nsys.schema"),
    );
    assert!(
        v.get("trace").is_none(),
        "nsys schema envelope must omit trace: {v}"
    );
    Ok(())
}

#[test]
fn graph_replays_cli_renders_json_table_and_csv() -> Result<()> {
    let (_trace_dir, trace) = build_graph_replay_trace()?;
    let trace_arg = trace.to_string_lossy();

    let json = run_veloq(["graph-replays", trace_arg.as_ref(), "--limit", "1"])?;
    assert!(
        json.status.success(),
        "graph-replays JSON failed: stderr={}",
        String::from_utf8_lossy(&json.stderr)
    );
    let v: Value = serde_json::from_slice(&json.stdout)?;
    assert_eq!(
        v.get("command").and_then(Value::as_str),
        Some("nsys.graph-replays")
    );
    assert_eq!(
        v.pointer("/data/rows/0/capture_mode")
            .and_then(Value::as_str),
        Some("graph_trace")
    );

    let table = run_veloq([
        "--format",
        "table",
        "graph-replays",
        trace_arg.as_ref(),
        "--limit",
        "1",
    ])?;
    assert!(table.status.success());
    let table_stdout = String::from_utf8_lossy(&table.stdout);
    assert!(table_stdout.contains("graph_trace"));

    let csv = run_veloq([
        "--format",
        "csv",
        "graph-replays",
        trace_arg.as_ref(),
        "--limit",
        "1",
    ])?;
    assert!(csv.status.success());
    let csv_stdout = String::from_utf8_lossy(&csv.stdout);
    assert!(csv_stdout.contains("synthetic_id"));
    assert!(csv_stdout.contains("graph_trace"));
    Ok(())
}

#[test]
fn schema_endpoint_covers_cli_side_nsys_payloads() -> Result<()> {
    for target in ["prep", "correlation-stats", "ncu-command"] {
        let out = run_veloq(["schema", target])?;
        assert!(
            out.status.success(),
            "schema {target} should succeed: stderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
        let v: Value = serde_json::from_slice(&out.stdout)
            .with_context(|| format!("schema {target} stdout must be valid JSON"))?;
        assert_eq!(
            v.get("command").and_then(Value::as_str),
            Some("nsys.schema"),
        );
        assert_eq!(
            v.get("data")
                .and_then(|d| d.get("target"))
                .and_then(Value::as_str),
            Some(target),
        );
        assert!(
            v.get("data").and_then(|d| d.get("schema")).is_some(),
            "schema endpoint missing schema document for {target}: {v}"
        );
    }
    Ok(())
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

#[test]
fn nsys_metrics_gpu_rejects_cpu_flags_with_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["metrics", trace.as_str(), "--name", "foo"])?;
    let v = assert_error_code(&out, "nsys.command.metrics-cpu-flag-conflict")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--type gpu") && message.contains("--name"),
        "message should name the rejected CPU flag set: {message}"
    );
    Ok(())
}

#[test]
fn nsys_metrics_cpu_sampling_rejects_counter_with_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq([
        "metrics",
        trace.as_str(),
        "--type",
        "cpu-sampling",
        "--counter",
        "SM*",
    ])?;
    let v = assert_error_code(&out, "nsys.command.metrics-counter-flag-conflict")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--counter") && message.contains("--name"),
        "message should point cpu-sampling users at --name: {message}"
    );
    Ok(())
}

#[test]
fn nsys_metrics_cpu_sched_rejects_name_with_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq([
        "metrics",
        trace.as_str(),
        "--type",
        "cpu-sched",
        "--name",
        "foo",
    ])?;
    let v = assert_error_code(&out, "nsys.command.metrics-name-flag-conflict")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--name") && message.contains("cpu-sched"),
        "message should explain cpu-sched has no name field: {message}"
    );
    Ok(())
}

#[test]
fn summary_happy_path_emits_full_envelope() -> Result<()> {
    let (_trace_dir, trace) = build_minimal_trace()?;
    let out = run_veloq(["summary", &trace.to_string_lossy()])?;
    assert!(
        out.status.success(),
        "summary failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value =
        serde_json::from_slice(&out.stdout).context("summary stdout must be valid JSON")?;
    // envelope: schema/source/command/trace/data are required.
    // trace_span is optional — present iff the source could resolve a
    // primary time range from the meta-cache sidecar.
    for key in ["schema", "source", "command", "trace", "data"] {
        assert!(
            v.get(key).is_some(),
            "summary envelope missing `{key}`: {v}"
        );
    }
    assert_eq!(
        v.get("schema").and_then(Value::as_str),
        Some("v1"),
        "envelope schema must be `v1`",
    );
    assert_eq!(
        v.get("command").and_then(Value::as_str),
        Some("nsys.summary"),
    );
    assert_eq!(
        v.get("source")
            .and_then(|s| s.get("kind"))
            .and_then(Value::as_str),
        Some("nsys"),
    );
    assert_eq!(
        v.get("trace")
            .and_then(|t| t.get("kind"))
            .and_then(Value::as_str),
        Some("nsys"),
    );
    assert_eq!(
        v.get("trace")
            .and_then(|t| t.get("path"))
            .and_then(Value::as_str),
        Some(trace.to_string_lossy().as_ref())
    );
    Ok(())
}

#[test]
fn schema_envelope_advertises_version_and_omits_trace_span() -> Result<()> {
    // Meta verbs don't read a trace; the envelope must report `v1`
    // (current schema) AND omit `trace_span` (no trace to span).
    let out = run_veloq(["schema", "summary"])?;
    assert!(out.status.success());
    let v: Value =
        serde_json::from_slice(&out.stdout).context("schema stdout must be valid JSON")?;
    assert_eq!(v.get("schema").and_then(Value::as_str), Some("v1"));
    assert!(
        v.get("trace_span").is_none(),
        "meta-verb envelope must omit trace_span: {v}",
    );
    Ok(())
}

#[test]
fn info_probes_capabilities_for_parquetdir_traces() -> Result<()> {
    // `info` reports the cheap probe: source detection + filesystem
    // facts + (for `_pqtdir/` NSys traces) the same capability bitmap
    // `summary.auxiliary.capabilities` carries — computed via parquet
    // file stats, no DuckDB open.
    let (_trace_dir, trace) = build_minimal_trace()?;
    let trace_path = trace.to_string_lossy().to_string();
    let out = run_veloq(["info", &trace_path])?;
    assert!(
        out.status.success(),
        "info failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value = serde_json::from_slice(&out.stdout).context("info stdout must be valid JSON")?;
    let data = v.get("data").ok_or_else(|| anyhow!("missing data: {v}"))?;
    assert_eq!(
        data.get("detected_source").and_then(Value::as_str),
        Some("nsys"),
        "minimal parquetdir trace must detect as nsys: {data}"
    );
    assert_eq!(
        data.get("exists").and_then(Value::as_bool),
        Some(true),
        "trace must exist on disk: {data}"
    );
    let caps = data
        .get("capabilities")
        .ok_or_else(|| anyhow!("info missing capabilities for parquetdir NSys: {data}"))?;
    assert_eq!(
        caps.get("has_kernels").and_then(Value::as_bool),
        Some(true),
        "fixture has CUPTI_ACTIVITY_KIND_KERNEL: {caps}"
    );
    assert_eq!(
        caps.get("has_nic_metrics").and_then(Value::as_bool),
        Some(false),
        "fixture has no NIC tables — capability must be false: {caps}"
    );
    Ok(())
}

#[test]
fn info_probes_capabilities_for_generated_parquetdir_alias() -> Result<()> {
    let (_trace_dir, _report, generated_pqtdir) = build_generated_parquetdir_alias()?;
    let trace_path = generated_pqtdir.to_string_lossy().to_string();
    let out = run_veloq(["info", &trace_path])?;
    assert!(
        out.status.success(),
        "info failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value = serde_json::from_slice(&out.stdout).context("info stdout must be valid JSON")?;
    let data = v.get("data").ok_or_else(|| anyhow!("missing data: {v}"))?;
    assert_eq!(
        data.get("detected_source").and_then(Value::as_str),
        Some("nsys"),
        "generated parquetdir alias must detect as nsys: {data}"
    );
    let caps = data
        .get("capabilities")
        .ok_or_else(|| anyhow!("info missing capabilities for generated parquetdir: {data}"))?;
    assert_eq!(
        caps.get("has_kernels").and_then(Value::as_bool),
        Some(true),
        "generated parquetdir alias should probe table presence: {caps}"
    );
    Ok(())
}

#[test]
fn info_does_not_detect_orphan_generated_parquetdir_alias() -> Result<()> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let generated_pqtdir = dir.path().join("missing.nsys-rep.veloq/parquetdir");
    std::fs::create_dir_all(&generated_pqtdir).context("create generated parquetdir")?;
    std::fs::write(
        generated_pqtdir.join("CUPTI_ACTIVITY_KIND_KERNEL.parquet"),
        b"not parquet",
    )
    .context("write placeholder table")?;

    let trace_path = generated_pqtdir.to_string_lossy().to_string();
    let out = run_veloq(["info", &trace_path])?;
    assert!(
        out.status.success(),
        "info should stay a cheap filesystem probe: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value = serde_json::from_slice(&out.stdout).context("info stdout must be valid JSON")?;
    let data = v.get("data").ok_or_else(|| anyhow!("missing data: {v}"))?;
    assert!(
        data.get("detected_source").is_none(),
        "orphan generated parquetdir must not claim nsys: {data}"
    );
    assert_eq!(
        data.get("exists").and_then(Value::as_bool),
        Some(true),
        "info still reports the inspected path exists: {data}"
    );
    assert!(
        data.get("capabilities").is_none(),
        "orphan generated parquetdir must not emit capabilities: {data}"
    );
    Ok(())
}

#[test]
fn info_omits_capabilities_for_missing_trace() -> Result<()> {
    // A non-existent path detects as nsys (extension match) but the
    // capability probe is gated on `exists` — the response should
    // omit the field entirely rather than emit an all-false bitmap.
    let out = run_veloq(["info", "/nonexistent.sqlite"])?;
    assert!(
        out.status.success(),
        "info should succeed even on missing trace"
    );
    let v: Value = serde_json::from_slice(&out.stdout).context("info stdout must be valid JSON")?;
    let data = v.get("data").ok_or_else(|| anyhow!("missing data: {v}"))?;
    assert_eq!(data.get("exists").and_then(Value::as_bool), Some(false));
    assert!(
        data.get("capabilities").is_none(),
        "missing trace must not carry a capabilities bitmap: {data}"
    );
    Ok(())
}

#[test]
fn prep_on_generated_parquetdir_uses_owner_artifact_root() -> Result<()> {
    let (_trace_dir, report, generated_pqtdir) = build_generated_parquetdir_alias()?;
    let trace_path = generated_pqtdir.to_string_lossy().to_string();
    let owner_root = veloq_core::artifact_dir_for(&report);
    let alias_root = veloq_core::artifact_dir_for(&generated_pqtdir);

    let prep = run_veloq(["prep", &trace_path])?;
    assert!(
        prep.status.success(),
        "prep should accept generated parquetdir alias: {}",
        String::from_utf8_lossy(&prep.stderr)
    );
    let v: Value = serde_json::from_slice(&prep.stdout).context("prep stdout must be JSON")?;
    let data = v.get("data").ok_or_else(|| anyhow!("missing data: {v}"))?;
    assert_eq!(
        data.get("cache_root").and_then(Value::as_str),
        Some(owner_root.to_string_lossy().as_ref()),
        "prep must report the owning report cache root: {data}"
    );
    assert!(
        owner_root.join("meta.bin").is_file(),
        "prep should write meta.bin under the owning report cache root"
    );
    assert!(
        !alias_root.exists(),
        "generated parquetdir must not get its own nested cache root"
    );

    let status = run_veloq(["prep", "--status", &trace_path])?;
    assert!(
        status.status.success(),
        "prep --status should accept generated parquetdir alias: {}",
        String::from_utf8_lossy(&status.stderr)
    );
    let v: Value =
        serde_json::from_slice(&status.stdout).context("prep --status stdout must be JSON")?;
    let data = v.get("data").ok_or_else(|| anyhow!("missing data: {v}"))?;
    assert_eq!(
        data.get("cache_root").and_then(Value::as_str),
        Some(owner_root.to_string_lossy().as_ref()),
        "prep --status must inspect the owning report cache root: {data}"
    );
    Ok(())
}

#[test]
fn clean_removes_only_veloq_artifact_root() -> Result<()> {
    let (_trace_dir, trace) = build_minimal_trace()?;
    let trace_path = trace.to_string_lossy().to_string();
    let prep = run_veloq(["prep", &trace_path])?;
    assert!(
        prep.status.success(),
        "prep should create cache root: {}",
        String::from_utf8_lossy(&prep.stderr)
    );
    let cache_root = veloq_core::artifact_dir_for(&trace);
    assert!(
        cache_root.is_dir(),
        "prep should create artifact root {}",
        cache_root.display()
    );

    let dry = run_veloq(["clean", "--dry-run", &trace_path])?;
    assert!(
        dry.status.success(),
        "clean --dry-run failed: {}",
        String::from_utf8_lossy(&dry.stderr)
    );
    let v: Value =
        serde_json::from_slice(&dry.stdout).context("clean dry-run stdout must be JSON")?;
    let data = v.get("data").ok_or_else(|| anyhow!("missing data: {v}"))?;
    assert_eq!(v.get("command").and_then(Value::as_str), Some("clean"));
    assert_eq!(
        data.get("dry_run").and_then(Value::as_bool),
        Some(true),
        "dry-run flag must round-trip: {data}"
    );
    assert_eq!(
        data.get("removed").and_then(Value::as_bool),
        Some(false),
        "dry-run must not remove artifacts: {data}"
    );
    assert!(cache_root.is_dir(), "dry-run must leave cache root intact");

    let clean = run_veloq(["clean", &trace_path])?;
    assert!(
        clean.status.success(),
        "clean failed: {}",
        String::from_utf8_lossy(&clean.stderr)
    );
    let v: Value =
        serde_json::from_slice(&clean.stdout).context("clean stdout must be valid JSON")?;
    let data = v.get("data").ok_or_else(|| anyhow!("missing data: {v}"))?;
    assert_eq!(
        data.get("removed").and_then(Value::as_bool),
        Some(true),
        "clean should remove an existing artifact root: {data}"
    );
    assert!(
        !cache_root.exists(),
        "clean should remove only the artifact root"
    );
    assert!(trace.is_dir(), "direct parquetdir input must remain intact");
    Ok(())
}

#[test]
fn clean_generated_parquetdir_removes_owner_artifact_root() -> Result<()> {
    let (_trace_dir, report, generated_pqtdir) = build_generated_parquetdir_alias()?;
    let trace_path = generated_pqtdir.to_string_lossy().to_string();
    let owner_root = veloq_core::artifact_dir_for(&report);
    let alias_root = veloq_core::artifact_dir_for(&generated_pqtdir);

    let out = run_veloq(["clean", &trace_path])?;
    assert!(
        out.status.success(),
        "clean should accept generated parquetdir alias: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value = serde_json::from_slice(&out.stdout).context("clean stdout must be JSON")?;
    let data = v.get("data").ok_or_else(|| anyhow!("missing data: {v}"))?;
    assert_eq!(
        data.get("cache_root").and_then(Value::as_str),
        Some(owner_root.to_string_lossy().as_ref()),
        "clean must target the owning report cache root: {data}"
    );
    assert_eq!(
        data.get("removed").and_then(Value::as_bool),
        Some(true),
        "clean should remove the existing owner cache root: {data}"
    );
    assert!(report.is_file(), "clean must not remove the source report");
    assert!(
        !owner_root.exists(),
        "clean should remove the owning report cache root"
    );
    assert!(
        !alias_root.exists(),
        "clean should not create a nested alias cache root"
    );
    Ok(())
}

#[test]
fn prep_status_reports_cold_then_warm_state() -> Result<()> {
    // `--status` is the read-only inspection form. The parquetdir is
    // the input itself (always `present`); the
    // veloq-managed sidecar that flips cold→warm is the meta cache.
    let (_trace_dir, trace) = build_minimal_trace()?;
    let trace_path = trace.to_string_lossy().to_string();

    // Cold path: parquetdir exists (it IS the input), meta sidecar absent.
    let out = run_veloq(["prep", "--status", &trace_path])?;
    assert!(
        out.status.success(),
        "prep --status (cold) must exit 0: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value = serde_json::from_slice(&out.stdout)
        .context("prep --status (cold) stdout must be valid JSON")?;
    let data = v.get("data").ok_or_else(|| anyhow!("missing data: {v}"))?;
    let parquet = data
        .get("parquet_cache")
        .ok_or_else(|| anyhow!("missing parquet_cache: {data}"))?;
    assert_eq!(
        parquet.get("present").and_then(Value::as_bool),
        Some(true),
        "parquetdir is the input — must report present=true: {parquet}"
    );
    let tables = parquet
        .get("tables")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("missing parquet_cache.tables: {parquet}"))?;
    assert!(
        !tables.is_empty(),
        "parquet_cache.tables should list the input parquet tables: {parquet}"
    );
    let meta = data
        .get("meta_cache")
        .ok_or_else(|| anyhow!("missing meta_cache: {data}"))?;
    assert_eq!(
        meta.get("present").and_then(Value::as_bool),
        Some(false),
        "cold meta cache should not yet exist on disk: {meta}"
    );

    // Warm path: build the caches, then re-status.
    let prep = run_veloq(["prep", &trace_path])?;
    assert!(
        prep.status.success(),
        "prep build failed: {}",
        String::from_utf8_lossy(&prep.stderr)
    );
    let out = run_veloq(["prep", "--status", &trace_path])?;
    assert!(out.status.success());
    let v: Value = serde_json::from_slice(&out.stdout)
        .context("prep --status (warm) stdout must be valid JSON")?;
    let data = v.get("data").ok_or_else(|| anyhow!("missing data: {v}"))?;
    let parquet = data
        .get("parquet_cache")
        .ok_or_else(|| anyhow!("missing parquet_cache: {data}"))?;
    assert_eq!(
        parquet.get("present").and_then(Value::as_bool),
        Some(true),
        "parquet_cache.present must stay true: {parquet}"
    );
    let meta = data
        .get("meta_cache")
        .ok_or_else(|| anyhow!("missing meta_cache: {data}"))?;
    assert_eq!(
        meta.get("present").and_then(Value::as_bool),
        Some(true),
        "warm meta cache must be present on disk: {meta}"
    );
    assert_eq!(
        meta.get("fingerprint_match").and_then(Value::as_bool),
        Some(true),
        "warm meta cache must match fingerprint: {meta}"
    );
    // After a successful prep, the on-disk meta-cache format version
    // matches what this binary expects. The parquet cache no longer
    // carries a manifest version (it's nsys's own output).
    let meta_expected = meta
        .get("format_version_expected")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("missing meta format_version_expected: {meta}"))?;
    let meta_on_disk = meta
        .get("format_version_on_disk")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("missing meta format_version_on_disk after prep: {meta}"))?;
    assert_eq!(
        meta_expected, meta_on_disk,
        "warm meta cache version must match expected"
    );
    Ok(())
}

#[test]
fn cold_summary_emits_trace_span_on_first_run() -> Result<()> {
    // Regression: `summary` against a never-prepped trace used to
    // omit `trace_span` because `Source::compute_trace_span` only
    // consulted an existing sidecar. The verb itself builds the
    // sidecar; the emit boundary re-reads it so the very first
    // `summary` call hands agents a populated normalization
    // denominator (the field every diff / per-sec recipe needs).
    let (_trace_dir, trace) = build_minimal_trace()?;
    let trace_path = trace.to_string_lossy().to_string();
    // No `prep` here — this is the cold path.
    let out = run_veloq(["summary", &trace_path])?;
    assert!(
        out.status.success(),
        "cold summary failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value =
        serde_json::from_slice(&out.stdout).context("summary stdout must be valid JSON")?;
    let span = v
        .get("trace_span")
        .ok_or_else(|| anyhow::anyhow!("cold summary missing trace_span: {v}"))?;
    assert!(
        span.get("origin_ns").and_then(Value::as_i64).is_some(),
        "cold trace_span.origin_ns must be an i64: {span}"
    );
    assert!(
        span.get("span_ns").and_then(Value::as_i64).is_some(),
        "cold trace_span.span_ns must be an i64: {span}"
    );
    Ok(())
}

#[cfg(unix)]
fn run_cold_nsys_rep_with_fake_export(command: &str) -> Result<Output> {
    use std::os::unix::fs::PermissionsExt;

    let (trace_dir, direct_pqtdir) = build_minimal_trace()?;
    let report = trace_dir.path().join("cold.nsys-rep");
    std::fs::write(&report, b"source").context("write cold report placeholder")?;

    let fake_bin_dir = trace_dir.path().join("fake-bin");
    std::fs::create_dir_all(&fake_bin_dir).context("create fake nsys bin dir")?;
    let fake_nsys = fake_bin_dir.join("nsys");
    std::fs::write(
        &fake_nsys,
        r#"#!/bin/sh
if [ "$1" = "--version" ]; then
    echo "NVIDIA Nsight Systems version 2026.2.1.210-fake"
    exit 0
fi
if [ "$1" = "export" ] && [ "$2" = "--help" ]; then
    echo "Possible values are: sqlite, arrowdir, parquetdir"
    exit 0
fi
if [ "$1" != "export" ]; then
    echo "unexpected fake nsys invocation: $*" >&2
    exit 9
fi
out=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        -o|--output)
            shift
            out="$1"
            ;;
    esac
    shift
done
if [ -z "$out" ]; then
    echo "missing -o/--output" >&2
    exit 7
fi
printf 'fake stdout progress\n'
printf 'fake stderr diagnostic\n' >&2
mkdir -p "$out"
for f in "$VELOQ_FAKE_PQTDIR"/*.parquet; do
    cp "$f" "$out/"
done
"#,
    )
    .context("write fake nsys")?;
    let mut perms = std::fs::metadata(&fake_nsys)?.permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&fake_nsys, perms)?;

    let old_path = std::env::var_os("PATH").unwrap_or_default();
    let mut paths = vec![fake_bin_dir];
    paths.extend(std::env::split_paths(&old_path));
    let path_env: std::ffi::OsString = std::env::join_paths(paths)?;
    let trace_path = report.to_string_lossy().to_string();
    run_veloq_with_env(
        [command, &trace_path],
        [
            (std::ffi::OsString::from("PATH"), path_env),
            (
                std::ffi::OsString::from("VELOQ_FAKE_PQTDIR"),
                direct_pqtdir.as_os_str().to_os_string(),
            ),
        ],
    )
}

#[cfg(unix)]
fn assert_child_output_stays_off_stdout(out: &Output, command: &str) -> Result<()> {
    assert!(
        out.status.success(),
        "{command} failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.contains("fake stdout progress"),
        "child stdout must not contaminate JSON stdout: {stdout}"
    );
    let v: Value = serde_json::from_slice(&out.stdout)
        .with_context(|| format!("{command} stdout must be valid JSON"))?;
    assert_eq!(v.get("command").and_then(Value::as_str), Some(command),);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("fake stdout progress"),
        "captured child stdout should be replayed on veloq stderr: {stderr}"
    );
    assert!(
        stderr.contains("fake stderr diagnostic"),
        "captured child stderr should stay on veloq stderr: {stderr}"
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn cold_nsys_rep_summary_export_keeps_child_output_off_stdout() -> Result<()> {
    let out = run_cold_nsys_rep_with_fake_export("summary")?;
    assert_child_output_stays_off_stdout(&out, "nsys.summary")
}

#[cfg(unix)]
#[test]
fn cold_nsys_rep_prep_export_keeps_child_output_off_stdout() -> Result<()> {
    let out = run_cold_nsys_rep_with_fake_export("prep")?;
    assert_child_output_stays_off_stdout(&out, "nsys.prep")
}

#[test]
fn warm_summary_emits_trace_span_after_prep() -> Result<()> {
    // First call to `prep` writes the metadata cache so the
    // envelope-level `trace_span` becomes available on the next run.
    // This verifies the contract that warm traces carry the
    // normalization denominator agents need for cross-trace diff.
    let (_trace_dir, trace) = build_minimal_trace()?;
    let trace_path = trace.to_string_lossy().to_string();
    let prep = run_veloq(["prep", &trace_path])?;
    assert!(
        prep.status.success(),
        "prep failed: {}",
        String::from_utf8_lossy(&prep.stderr)
    );
    let out = run_veloq(["summary", &trace_path])?;
    assert!(out.status.success());
    let v: Value =
        serde_json::from_slice(&out.stdout).context("summary stdout must be valid JSON")?;
    let span = v
        .get("trace_span")
        .ok_or_else(|| anyhow::anyhow!("warm summary missing trace_span: {v}"))?;
    assert!(
        span.get("origin_ns").and_then(Value::as_i64).is_some(),
        "trace_span.origin_ns must be an i64: {span}"
    );
    assert!(
        span.get("span_ns").and_then(Value::as_i64).is_some(),
        "trace_span.span_ns must be an i64: {span}"
    );
    Ok(())
}

#[test]
fn missing_trace_in_json_mode_emits_error_envelope_with_quiet_stderr() -> Result<()> {
    // JSON is the documented default. Under the agent contract the
    // stdout envelope is the single source of truth; stderr stays
    // quiet so agents don't have to dedupe a "veloq: …" mirror.
    let out = run_veloq(["summary", "/nonexistent.sqlite"])?;
    assert!(
        !out.status.success(),
        "missing trace should yield non-zero exit"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.is_empty(),
        "JSON mode must keep stderr clean; got: {stderr}"
    );
    let v = assert_error_code(&out, "nsys.data.sqlite-input-unsupported")?;
    let error = v
        .get("error")
        .ok_or_else(|| anyhow!("stdout envelope missing `error`: {v}"))?;
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("error.message missing: {v}"))?;
    assert!(
        message.contains("/nonexistent.sqlite"),
        "error.message should mention the trace path; got: {message}"
    );
    assert!(
        v.get("data").is_none(),
        "error envelope must not carry `data`: {v}"
    );
    assert_eq!(
        v.get("command").and_then(Value::as_str),
        Some("nsys.summary"),
    );
    Ok(())
}

#[test]
fn nsys_stats_by_size_requires_unstable_with_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq_without_unstable(["stats", trace.as_str(), "--by", "size"])?;
    let v = assert_error_code(&out, "nsys.command.unstable-feature-disabled")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("VELOQ_UNSTABLE=1") && message.contains("--by size"),
        "message should name the env gate and hidden flag: {message}"
    );
    Ok(())
}

#[test]
fn nsys_stats_by_size_hist_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq_with_env(
        ["stats", trace.as_str(), "--by", "size", "--hist"],
        [("VELOQ_UNSTABLE", "1")],
    )?;
    let v = assert_error_code(&out, "nsys.command.stats-by-size-hist-unsupported")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--by size") && message.contains("histograms"),
        "message should explain the unsupported histogram combination: {message}"
    );
    Ok(())
}

#[test]
fn nsys_stats_by_size_nvtx_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq_with_env(
        ["stats", trace.as_str(), "--by", "size", "--nvtx", "phase*"],
        [("VELOQ_UNSTABLE", "1")],
    )?;
    let v = assert_error_code(&out, "nsys.command.stats-by-size-nvtx-unsupported")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--nvtx") && message.contains("--by size"),
        "message should explain the unsupported NVTX combination: {message}"
    );
    Ok(())
}

#[test]
fn nsys_stats_by_size_group_by_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq_with_env(
        [
            "stats",
            trace.as_str(),
            "--by",
            "size",
            "--group-by",
            "short,nvtx-path",
        ],
        [("VELOQ_UNSTABLE", "1")],
    )?;
    let v = assert_error_code(&out, "nsys.command.stats-by-size-group-by-unsupported")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("nvtx-path") && message.contains("--group-by"),
        "message should name the unsupported group-by axis: {message}"
    );
    Ok(())
}

#[test]
fn nsys_slices_path_group_by_requires_aggregate_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["slices", trace.as_str(), "--group-by", "path"])?;
    let v = assert_error_code(&out, "nsys.command.slices-group-by-requires-aggregate")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--group-by path") && message.contains("--aggregate"),
        "message should point users to --aggregate: {message}"
    );
    Ok(())
}

#[test]
fn nsys_metrics_unknown_source_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["metrics", trace.as_str(), "--type", "cpu"])?;
    let v = assert_error_code(&out, "nsys.command.metrics-unknown-source")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("cpu") && message.contains("cpu-sampling"),
        "message should name rejected and supported metric sources: {message}"
    );
    Ok(())
}

#[test]
fn nsys_metrics_invalid_bucket_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["metrics", trace.as_str(), "--bucket", "nope"])?;
    let v = assert_error_code(&out, "nsys.command.metrics-invalid-bucket")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--bucket") && message.contains("nope"),
        "message should name invalid --bucket value: {message}"
    );
    let chain = v
        .pointer("/error/chain")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    assert!(
        chain
            .iter()
            .filter_map(Value::as_str)
            .any(|entry| entry.contains("invalid --bucket")),
        "chain should keep bucket parser detail: {chain:?}"
    );
    Ok(())
}

#[test]
fn nsys_query_search_name_filter_conflict_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq([
        "search",
        trace.as_str(),
        "--name",
        "foo",
        "--name-regex",
        "foo",
    ])?;
    let v = assert_error_code(&out, "nsys.query.name-filter-conflict")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--name") && message.contains("--name-regex"),
        "message should name both conflicting filters: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_kind_location_filter_conflict_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["search", trace.as_str(), "--type", "nvtx", "--stream", "7"])?;
    let v = assert_error_code(&out, "nsys.query.kind-location-filter-conflict")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--stream") && message.contains("nvtx"),
        "message should name the conflicting location filter and kind: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_kind_nvtx_attribution_conflict_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq([
        "stats",
        trace.as_str(),
        "--type",
        "nvtx",
        "--all-devices",
        "--nvtx",
        "*",
    ])?;
    let v = assert_error_code(&out, "nsys.query.kind-nvtx-attribution-unsupported")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--nvtx") && message.contains("experimental"),
        "message should explain the unsupported NVTX attribution request: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_stats_group_by_unknown_token_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["stats", trace.as_str(), "--group-by", "mystery"])?;
    let v = assert_error_code(&out, "nsys.query.stats-group-by-unknown-token")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("mystery") && message.contains("short"),
        "message should name rejected and expected stats group-by axes: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_stats_group_by_name_axis_conflict_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["stats", trace.as_str(), "--group-by", "short,demangled"])?;
    let v = assert_error_code(&out, "nsys.query.stats-group-by-name-axis-conflict")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("short") && message.contains("demangled"),
        "message should name both name axes: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_stats_group_by_location_axis_conflict_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq([
        "stats",
        trace.as_str(),
        "--type",
        "runtime",
        "--all-devices",
        "--group-by",
        "device",
    ])?;
    let v = assert_error_code(&out, "nsys.query.stats-group-by-location-axis-conflict")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--type runtime") && message.contains("device"),
        "message should explain the CPU-side location-axis conflict: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_stats_grid_block_kind_conflict_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq([
        "stats",
        trace.as_str(),
        "--type",
        "kernel,memcpy",
        "--group-by",
        "grid_block",
    ])?;
    let v = assert_error_code(&out, "nsys.query.stats-grid-block-kind-conflict")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("grid_block") && message.contains("memcpy"),
        "message should name the non-kernel kind in the grid_block request: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_stats_nvtx_hierarchy_axes_conflict_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq([
        "stats",
        trace.as_str(),
        "--group-by",
        "nvtx-parent,nvtx-path",
    ])?;
    let v = assert_error_code(&out, "nsys.query.stats-nvtx-hierarchy-axis-conflict")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("nvtx-parent") && message.contains("nvtx-path"),
        "message should name the mutually exclusive NVTX hierarchy axes: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_stats_nvtx_hierarchy_graph_conflict_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["stats", trace.as_str(), "--group-by", "nvtx-parent,graph"])?;
    let v = assert_error_code(&out, "nsys.query.stats-nvtx-hierarchy-graph-axis-conflict")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("nvtx-parent") && message.contains("graph"),
        "message should explain the graph/NVTX hierarchy conflict: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_stats_nvtx_hierarchy_self_attribute_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq([
        "stats",
        trace.as_str(),
        "--type",
        "nvtx",
        "--all-devices",
        "--group-by",
        "nvtx-parent",
    ])?;
    let v = assert_error_code(&out, "nsys.query.stats-nvtx-hierarchy-self-attribute")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--type nvtx") && message.contains("self-attribute"),
        "message should explain the NVTX self-attribute request: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_stats_nvtx_hierarchy_prereq_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["stats", trace.as_str(), "--group-by", "nvtx-parent"])?;
    let v = assert_error_code(&out, "nsys.query.stats-nvtx-hierarchy-prereq-missing")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("nvtx-parent") && message.contains("NVTX_EVENTS"),
        "message should name the missing NVTX hierarchy prerequisite: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_timeline_interval_too_small_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["timeline", trace.as_str(), "--interval", "0ns"])?;
    let v = assert_error_code(&out, "nsys.query.timeline-interval-too-small")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--interval") && message.contains("positive"),
        "message should explain the minimum interval: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_timeline_invalid_interval_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["timeline", trace.as_str(), "--interval", "bogus"])?;
    let v = assert_error_code(&out, "nsys.query.timeline-interval-invalid")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--interval") && message.contains("bogus"),
        "message should name invalid --interval value: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_timeline_nvtx_prereq_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq([
        "timeline",
        trace.as_str(),
        "--interval",
        "1ms",
        "--nvtx",
        "*",
    ])?;
    let v = assert_error_code(&out, "nsys.query.nvtx-attribution-prereq-missing")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--nvtx") && message.contains("NVTX_EVENTS"),
        "message should name the missing NVTX attribution table: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_slices_unknown_group_by_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq([
        "slices",
        trace.as_str(),
        "--aggregate",
        "--group-by",
        "bogus",
    ])?;
    let v = assert_error_code(&out, "nsys.query.slices-unknown-group-by")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("bogus") && message.contains("name"),
        "message should name rejected and expected slices group-by axes: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_slices_missing_prereq_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["slices", trace.as_str()])?;
    let v = assert_error_code(&out, "nsys.query.slices-prereq-missing")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("NVTX_EVENTS") && message.contains("attribution"),
        "message should name the missing slices prerequisite table: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_metrics_sort_bucket_conflict_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq([
        "metrics",
        trace.as_str(),
        "--bucket",
        "1ms",
        "--sort",
        "count:desc",
    ])?;
    let v = assert_error_code(&out, "nsys.query.metrics-sort-bucket-conflict")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--sort") && message.contains("bucket"),
        "message should explain the bucket/sort conflict: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_metrics_gpu_missing_table_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["metrics", trace.as_str()])?;
    let v = assert_error_code(&out, "nsys.query.metrics-gpu-table-missing")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("GPU_METRICS") && message.contains("--gpu-metrics-devices"),
        "message should name the missing GPU metrics table and capture flag: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_metrics_nic_missing_table_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["metrics", trace.as_str(), "--type", "nic"])?;
    let v = assert_error_code(&out, "nsys.query.metrics-nic-table-missing")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("NET_NIC_METRIC") && message.contains("--nic-metrics"),
        "message should name the missing NIC metrics table and capture flag: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_metrics_cpu_sampling_missing_table_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["metrics", trace.as_str(), "--type", "cpu-sampling"])?;
    let v = assert_error_code(&out, "nsys.query.metrics-cpu-sampling-table-missing")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("COMPOSITE_EVENTS") && message.contains("--sample"),
        "message should name the missing CPU sampling table and capture flag: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_metrics_cpu_sched_missing_table_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["metrics", trace.as_str(), "--type", "cpu-sched"])?;
    let v = assert_error_code(&out, "nsys.query.metrics-cpu-sched-table-missing")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("SCHED_EVENTS") && message.contains("--cpuctxsw"),
        "message should name the missing CPU sched table and capture flag: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_gaps_invalid_scope_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["gaps", trace.as_str(), "--scope", "whole-job"])?;
    let v = assert_error_code(&out, "nsys.query.gaps-invalid-scope")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("whole-job") && message.contains("device"),
        "message should name rejected and expected scopes: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_gaps_invalid_min_duration_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["gaps", trace.as_str(), "--min-duration", "bogus"])?;
    let v = assert_error_code(&out, "nsys.query.gaps-min-duration-invalid")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--min-duration") && message.contains("bogus"),
        "message should name invalid --min-duration value: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_gaps_stream_scope_required_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["gaps", trace.as_str(), "--stream", "7"])?;
    let v = assert_error_code(&out, "nsys.query.gaps-stream-scope-required")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--stream") && message.contains("--scope stream"),
        "message should point stream filters at stream scope: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_gaps_device_trace_conflict_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["gaps", trace.as_str(), "--scope", "trace", "--device", "0"])?;
    let v = assert_error_code(&out, "nsys.query.gaps-device-scope-conflict")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--device 0") && message.contains("--scope trace"),
        "message should explain device filters conflict with trace scope: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_gaps_sort_stream_scope_required_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["gaps", trace.as_str(), "--sort", "stream"])?;
    let v = assert_error_code(&out, "nsys.query.gaps-sort-stream-scope-required")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--sort stream") && message.contains("--scope stream"),
        "message should point stream sort at stream scope: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_gaps_sort_device_trace_conflict_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq([
        "gaps",
        trace.as_str(),
        "--scope",
        "trace",
        "--all-devices",
        "--sort",
        "device",
    ])?;
    let v = assert_error_code(&out, "nsys.query.gaps-sort-device-scope-conflict")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--sort device") && message.contains("--scope trace"),
        "message should explain device sort conflicts with trace scope: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_graph_replays_top_nodes_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_graph_replay_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["graph-replays", trace.as_str(), "--top-nodes", "0"])?;
    let v = assert_error_code(&out, "nsys.query.graph-replays-top-nodes-too-small")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--top-nodes") && message.contains("1"),
        "message should name the minimum top-nodes value: {message}"
    );
    Ok(())
}

#[test]
fn nsys_query_graph_replays_nvtx_prereq_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_graph_replay_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["graph-replays", trace.as_str(), "--nvtx", "*"])?;
    let v = assert_error_code(&out, "nsys.query.graph-replays-nvtx-prereq-missing")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("graph-replays") && message.contains("NVTX_EVENTS"),
        "message should name the missing graph-replays NVTX table: {message}"
    );
    Ok(())
}

#[test]
fn nsys_missing_time_bound_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["stats", trace.as_str(), "--from", "1ms"])?;
    let v = assert_error_code(&out, "nsys.command.missing-time-bound")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--from") && message.contains("--to"),
        "message should name both time-bound flags: {message}"
    );
    Ok(())
}

#[test]
fn nsys_invalid_from_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["stats", trace.as_str(), "--from", "nope", "--to", "1ms"])?;
    let v = assert_error_code(&out, "nsys.command.invalid-from")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--from") && message.contains("nope"),
        "message should name invalid --from value: {message}"
    );
    Ok(())
}

#[test]
fn nsys_zero_limit_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["search", trace.as_str(), "--limit", "0"])?;
    let v = assert_error_code(&out, "nsys.command.limit-too-small")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--limit") && message.contains("0"),
        "message should name rejected limit: {message}"
    );
    Ok(())
}

#[test]
fn nsys_unknown_event_kind_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["stats", trace.as_str(), "--type", "bogus"])?;
    let v = assert_error_code(&out, "nsys.command.unknown-event-kind")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("bogus"),
        "message should name rejected event kind: {message}"
    );
    Ok(())
}

#[test]
fn nsys_event_kind_not_allowed_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq([
        "timeline",
        trace.as_str(),
        "--interval",
        "1ms",
        "--type",
        "sync",
    ])?;
    let v = assert_error_code(&out, "nsys.command.event-kind-not-allowed")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("sync") && message.contains("kernel"),
        "message should name rejected and allowed event kinds: {message}"
    );
    Ok(())
}

#[test]
fn nsys_empty_event_kind_list_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["stats", trace.as_str(), "--type", ","])?;
    let v = assert_error_code(&out, "nsys.command.empty-event-kind-list")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--type"),
        "message should name rejected flag: {message}"
    );
    Ok(())
}

#[test]
fn nsys_invalid_sort_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["stats", trace.as_str(), "--sort", "total:nope"])?;
    let v = assert_error_code(&out, "nsys.command.invalid-sort")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--sort") && message.contains("total:nope"),
        "message should name invalid --sort value: {message}"
    );
    let chain = v
        .pointer("/error/chain")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    assert!(
        chain
            .iter()
            .filter_map(Value::as_str)
            .any(|entry| entry.contains("unknown sort direction")),
        "chain should keep sort parser detail: {chain:?}"
    );
    Ok(())
}

#[test]
fn nsys_invalid_duration_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["search", trace.as_str(), "--duration", "nope"])?;
    let v = assert_error_code(&out, "nsys.command.invalid-duration")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("--duration") && message.contains("nope"),
        "message should name invalid --duration value: {message}"
    );
    Ok(())
}

#[test]
fn nsys_invalid_row_id_has_specific_error_code() -> Result<()> {
    let (_dir, trace) = build_minimal_trace()?;
    let trace = trace.to_string_lossy().into_owned();
    let out = run_veloq(["inspect", trace.as_str(), "no-colon"])?;
    let v = assert_error_code(&out, "nsys.command.invalid-row-id")?;
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        message.contains("row_id") && message.contains("no-colon"),
        "message should name invalid row_id: {message}"
    );
    Ok(())
}

#[test]
fn missing_trace_in_table_mode_mirrors_error_on_stderr() -> Result<()> {
    // Explicit `--format=table` is human-targeted; keep the stderr
    // mirror so terminal users see the cause without parsing JSON.
    let out = run_veloq(["--format", "table", "summary", "/nonexistent.sqlite"])?;
    assert!(
        !out.status.success(),
        "missing trace should yield non-zero exit"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.starts_with("veloq:"),
        "table mode must mirror `veloq: …` to stderr; got: {stderr}"
    );
    assert!(
        stderr.contains("/nonexistent.sqlite"),
        "stderr should mention the trace path; got: {stderr}"
    );
    Ok(())
}

#[test]
fn bogus_subcommand_routes_through_envelope() -> Result<()> {
    let out = run_veloq(["definitely-not-a-command"])?;
    assert!(
        !out.status.success(),
        "bogus subcommand should yield non-zero exit"
    );
    // JSON is the parse-error default — stderr stays quiet; the
    // unrecognized subcommand surfaces inside the stdout envelope's
    // error.message instead.
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.is_empty(),
        "JSON parse-error must keep stderr clean; got: {stderr}"
    );
    // stdout: envelope with `error.chain` mentioning clap's ErrorKind.
    let v: Value =
        serde_json::from_slice(&out.stdout).context("parse-error stdout must be valid JSON")?;
    let error = v
        .get("error")
        .ok_or_else(|| anyhow!("missing error: {v}"))?;
    let chain_entry = error
        .get("chain")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing chain[0]: {v}"))?;
    assert!(
        chain_entry.contains("InvalidSubcommand"),
        "chain[0] should mention clap::ErrorKind::InvalidSubcommand; got: {chain_entry}"
    );
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing error.message: {v}"))?;
    assert!(
        message.contains("definitely-not-a-command"),
        "error.message should echo the unrecognized subcommand; got: {message}"
    );
    Ok(())
}

#[test]
fn table_mode_parse_error_mirrors_error_on_stderr() -> Result<()> {
    let out = run_veloq(["--format", "table", "definitely-not-a-command"])?;
    assert!(
        !out.status.success(),
        "bogus subcommand should yield non-zero exit"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.starts_with("veloq:"),
        "table parse-error must mirror `veloq: ...` to stderr; got: {stderr}"
    );

    let v: Value =
        serde_json::from_slice(&out.stdout).context("parse-error stdout must be valid JSON")?;
    let message = v
        .get("error")
        .and_then(|e| e.get("message"))
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing error.message: {v}"))?;
    assert!(
        message.contains("definitely-not-a-command"),
        "error.message should echo the unrecognized subcommand; got: {message}"
    );
    Ok(())
}

#[test]
fn format_csv_dispatch_changes_stdout_shape() -> Result<()> {
    let (_trace_dir, trace) = build_minimal_trace()?;
    let out = run_veloq(["--format", "csv", "summary", &trace.to_string_lossy()])?;
    assert!(
        out.status.success(),
        "summary --format csv failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    // CSV output is comma-separated key=value-style lines, not JSON.
    // The cheapest invariant: not parseable as a JSON envelope.
    assert!(
        serde_json::from_str::<Value>(&stdout).is_err(),
        "--format csv must not emit JSON envelope; got: {stdout}"
    );
    Ok(())
}

#[test]
fn ncu_summary_envelope_has_single_data_layer() -> Result<()> {
    // Regression test for the v1-envelope migration bug where
    // `summarize_report_with` returned its own envelope-shaped
    // `ReportSummary`, which the dispatcher then wrapped *again* in
    // an `Envelope` — producing `.data.data.sources` instead of the
    // documented `.data.sources`.
    //
    // Uses the committed `source_metric_basic` fixture in-place (it
    // ships a committed `ncu_report` native sidecar), so the native
    // summary path serves NCU-free — a synthetic temp report would
    // have no committed sidecar and need NCU to build one.
    let trace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../ncu/veloq-ncu/tests/fixtures/source_metric_basic.ncu-rep");
    let out = run_veloq(["ncu", "summary", &trace.to_string_lossy()])?;
    assert!(
        out.status.success(),
        "ncu summary on the committed fixture should succeed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value =
        serde_json::from_slice(&out.stdout).context("ncu summary stdout must be valid JSON")?;

    // v1 envelope shape — qualified command, source kind, trace
    // kind, and `data` carrying the SummaryData directly.
    assert_eq!(
        v.get("command").and_then(Value::as_str),
        Some("ncu.summary"),
    );
    assert_eq!(
        v.get("source")
            .and_then(|s| s.get("kind"))
            .and_then(Value::as_str),
        Some("ncu"),
    );
    assert_eq!(
        v.get("trace")
            .and_then(|t| t.get("kind"))
            .and_then(Value::as_str),
        Some("ncu"),
    );

    let data = v
        .get("data")
        .ok_or_else(|| anyhow!("envelope missing `data`: {v}"))?;
    // The double-wrap regression would place a nested
    // `{schema, command, trace, data}` here instead of the canonical
    // fields. v4 native ncu summary: rows + count + total_matched +
    // auxiliary at the top of `.data`; the degraded `session` (NCU
    // version only), `ncu_version`, and `meta_cache_path` live inside
    // `auxiliary`; there is no `file_header_version`.
    for required in ["rows", "count", "total_matched", "auxiliary"] {
        assert!(
            data.get(required).is_some(),
            "data should carry canonical summary field `{required}`: {data}"
        );
    }
    let aux = data
        .get("auxiliary")
        .ok_or_else(|| anyhow!("missing auxiliary"))?;
    for required in ["session", "ncu_version", "meta_cache_path"] {
        assert!(
            aux.get(required).is_some(),
            "auxiliary should carry `{required}`: {aux}"
        );
    }
    assert!(
        data.get("data").is_none(),
        "data must NOT carry a nested `data` field (double-wrap regression): {data}"
    );
    Ok(())
}

#[test]
fn ncu_summary_csv_emits_native_totals_projection() -> Result<()> {
    // `ncu summary --format csv` renders the native totals projection
    // (section/key/value long format) NCU-free from the committed
    // `source_metric_basic` sidecar (the report itself is not committed;
    // build_or_load serves the sidecar). There is no `--page`: the
    // native model has no separate detail/raw/session pages.
    let trace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../ncu/veloq-ncu/tests/fixtures/source_metric_basic.ncu-rep");
    let out = run_veloq([
        "ncu",
        "summary",
        "--format",
        "csv",
        &trace.to_string_lossy(),
    ])?;
    assert!(
        out.status.success(),
        "ncu summary --format csv should succeed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("# command=ncu.summary"),
        "csv output should include command metadata: {stdout}"
    );
    assert!(
        stdout.contains("section,key,value"),
        "csv summary should emit the totals table header: {stdout}"
    );
    assert!(
        stdout.contains("launch_count"),
        "csv summary should emit a launch_count totals row: {stdout}"
    );
    assert!(
        serde_json::from_str::<Value>(&stdout).is_err(),
        "csv must not emit a JSON envelope; got: {stdout}"
    );
    Ok(())
}

#[test]
fn ncu_schema_endpoint_emits_envelope_without_trace() -> Result<()> {
    let out = run_veloq(["ncu", "schema", "summary"])?;
    assert!(
        out.status.success(),
        "ncu schema summary should succeed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value =
        serde_json::from_slice(&out.stdout).context("ncu schema stdout must be valid JSON")?;
    assert_eq!(v.get("command").and_then(Value::as_str), Some("ncu.schema"),);
    assert_eq!(
        v.get("source")
            .and_then(|s| s.get("kind"))
            .and_then(Value::as_str),
        Some("ncu"),
    );
    assert!(
        v.get("trace").is_none(),
        "ncu schema envelope must omit trace: {v}"
    );
    assert_eq!(
        v.get("data")
            .and_then(|d| d.get("target"))
            .and_then(Value::as_str),
        Some("summary"),
    );
    assert!(
        v.get("data").and_then(|d| d.get("schema")).is_some(),
        "ncu schema response missing schema document: {v}"
    );
    Ok(())
}

#[test]
fn schema_bad_target_omits_trace_field() -> Result<()> {
    // Regression test for `veloq schema <bad-target>` fabricating
    // `envelope.trace.path == ""` instead of omitting `trace` on the
    // error envelope (the success envelope omitted it correctly; the
    // failure path used the now-replaced `Cmd::trace_path -> &Path`
    // that returned `Path::new("")` for trace-less verbs).
    let out = run_veloq(["schema", "definitely-not-a-target"])?;
    assert!(
        !out.status.success(),
        "schema with bogus target should exit non-zero"
    );
    let v: Value =
        serde_json::from_slice(&out.stdout).context("schema error stdout must be valid JSON")?;
    // Qualified verb name + nsys source kind even on failure.
    assert_eq!(
        v.get("command").and_then(Value::as_str),
        Some("nsys.schema"),
    );
    assert_eq!(
        v.get("source")
            .and_then(|s| s.get("kind"))
            .and_then(Value::as_str),
        Some("nsys"),
    );
    // The bug: `trace` was present with `path: ""`. Fixed contract:
    // schema is a meta endpoint with no trace, so the field must be
    // absent on both success and failure.
    assert!(
        v.get("trace").is_none(),
        "schema error envelope must omit `trace`: {v}"
    );
    assert_eq!(
        v.pointer("/error/code").and_then(Value::as_str),
        Some("nsys.command.unknown-schema-target")
    );
    // Sanity: the error chain actually mentions the bogus target.
    let message = v
        .get("error")
        .and_then(|e| e.get("message"))
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing error.message: {v}"))?;
    assert!(
        message.contains("definitely-not-a-target"),
        "error.message should echo the bad target name; got: {message}"
    );
    Ok(())
}
