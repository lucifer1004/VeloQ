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

#[path = "cli_smoke/ncu.rs"]
mod ncu;
#[path = "cli_smoke/nsys_artifacts.rs"]
mod nsys_artifacts;
#[path = "cli_smoke/nsys_errors.rs"]
mod nsys_errors;
#[path = "cli_smoke/nsys_ncu_command.rs"]
mod nsys_ncu_command;
#[path = "cli_smoke/pytorch.rs"]
mod pytorch;
#[path = "cli_smoke/root.rs"]
mod root;

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

fn assert_schema_envelope(
    out: &Output,
    command: &str,
    source_kind: &str,
    source_version: &str,
    target: &str,
) -> Result<Value> {
    assert!(
        out.status.success(),
        "{command} {target} should succeed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value =
        serde_json::from_slice(&out.stdout).context("schema stdout must be valid JSON")?;
    assert_eq!(
        v.get("schema").and_then(Value::as_str),
        Some("v1"),
        "schema envelope must advertise v1: {v}",
    );
    assert_eq!(v.get("command").and_then(Value::as_str), Some(command));
    assert_eq!(
        v.pointer("/source/kind").and_then(Value::as_str),
        Some(source_kind),
        "schema envelope has wrong source kind: {v}",
    );
    assert_eq!(
        v.pointer("/source/version").and_then(Value::as_str),
        Some(source_version),
        "schema envelope has wrong source version: {v}",
    );
    assert!(
        v.get("trace").is_none(),
        "schema envelope must omit trace: {v}",
    );
    assert!(
        v.get("trace_span").is_none(),
        "schema envelope must omit trace_span: {v}",
    );
    assert_eq!(
        v.pointer("/data/target").and_then(Value::as_str),
        Some(target),
        "schema payload has wrong target: {v}",
    );
    let schema = v
        .pointer("/data/schema")
        .ok_or_else(|| anyhow!("schema response missing schema document: {v}"))?;
    assert!(
        schema.is_object(),
        "schema response must carry a JSON Schema object: {v}",
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

fn run_veloq_with_env_and_cwd<I, S, E, K, V>(
    args: I,
    envs: E,
    cwd: impl AsRef<std::path::Path>,
) -> Result<Output>
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
    E: IntoIterator<Item = (K, V)>,
    K: AsRef<std::ffi::OsStr>,
    V: AsRef<std::ffi::OsStr>,
{
    let mut cmd = Command::new(veloq_bin());
    cmd.args(args);
    cmd.current_dir(cwd);
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
        CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO (
            deviceId BIGINT, contextId BIGINT, processId BIGINT
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
        "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO \
         (deviceId, contextId, processId) VALUES (?, ?, ?)",
        params![0i32, 0i64, 12345i64],
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
