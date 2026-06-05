//! End-to-end smoke tests:
//! - `info` extended into a trace map (devices, processes with
//!   env-derived labels, NVTX summary, applicable_recipes)
//! - cold `.nsys-rep` falls back to basics + a `veloq prep` next step
//! - list verbs populate `meta.next_steps` on non-empty responses
//!
//! Each test builds an in-memory DuckDB, copies the tables out as a
//! parquetdir, and drives the freshly-built `veloq` binary.

use anyhow::{Context, Result};
use duckdb::{Connection, params};
use serde_json::Value;
use std::path::PathBuf;
use std::process::{Command, Output};
use tempfile::TempDir;

fn veloq_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_veloq"))
}

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

fn parse_stdout(out: &Output) -> Result<Value> {
    serde_json::from_slice(&out.stdout).context("veloq stdout must be valid JSON")
}

fn at<'a>(v: &'a Value, ptr: &str) -> Result<&'a Value> {
    v.pointer(ptr)
        .ok_or_else(|| anyhow::anyhow!("missing pointer `{ptr}` in {v}"))
}

/// COPY every user-created table to `<dir>/test_pqtdir/<TABLE>.parquet`.
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
        )?;
    }
    Ok(pqtdir)
}

fn install_minimal_export_metadata(conn: &Connection) -> Result<()> {
    conn.execute_batch("CREATE TABLE IF NOT EXISTS META_DATA_EXPORT (name TEXT, value TEXT);")?;
    for (k, v) in [
        ("EXPORT_SCHEMA_VERSION_MAJOR", "3"),
        ("EXPORT_SCHEMA_VERSION_MINOR", "0"),
        ("EXPORT_SCHEMA_VERSION_MICRO", "0"),
    ] {
        conn.execute(
            "INSERT INTO META_DATA_EXPORT (name, value) VALUES (?, ?)",
            params![k, v],
        )?;
    }
    Ok(())
}

/// Single-device, single-process trace with one kernel + a few NVTX
/// ranges so `info` has data to project into every trace-map field.
fn build_single_device_trace_with_nvtx() -> Result<(TempDir, PathBuf)> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    conn.execute_batch(
        r#"
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        CREATE TABLE TARGET_INFO_GPU (id BIGINT, cuDevice BIGINT, uuid TEXT);
        CREATE TABLE PROCESSES (globalPid BIGINT, pid BIGINT, name TEXT);
        CREATE TABLE NVTX_EVENTS (
            start BIGINT,
            "end" BIGINT,
            eventType BIGINT,
            globalTid BIGINT,
            domainId BIGINT,
            text TEXT,
            textId BIGINT
        );
        CREATE TABLE META_DATA_CAPTURE (name TEXT, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start BIGINT,
            "end" BIGINT,
            globalTid BIGINT,
            correlationId BIGINT,
            nameId BIGINT
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
    )?;
    install_minimal_export_metadata(&conn)?;
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1_i64, "smoke_kernel"],
    )?;
    conn.execute(
        "INSERT INTO TARGET_INFO_GPU (id, cuDevice, uuid) VALUES (?, ?, ?)",
        params![10_i64, 0_i64, "synthetic-gpu-0"],
    )?;
    conn.execute(
        "INSERT INTO PROCESSES (globalPid, pid, name) VALUES (?, ?, ?)",
        params![(4242_i64 << 24), 4242_i64, "synthetic-host"],
    )?;
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
         (start, \"end\", deviceId, contextId, streamId, \
          shortName, demangledName, gridX, gridY, gridZ, \
          blockX, blockY, blockZ, correlationId, \
          registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            100_000_000_i64,
            101_000_000_i64,
            0_i64,
            100_i64,
            7_i64,
            1_i64,
            1_i64,
            1_i64,
            1_i64,
            1_i64,
            128_i64,
            1_i64,
            1_i64,
            1_i64,
            32_i64,
            0_i64,
            0_i64,
            4242_i64 << 24,
        ],
    )?;
    // Three NVTX ranges on the same tid, default domain, two with the
    // same name so `top_paths` aggregation has a non-trivial winner.
    let gtid = 4242_i64 << 24;
    for (start, end, name) in [
        (10_000_000_i64, 60_000_000_i64, "outer_loop"),
        (20_000_000_i64, 40_000_000_i64, "inner_step"),
        (45_000_000_i64, 55_000_000_i64, "inner_step"),
    ] {
        conn.execute(
            "INSERT INTO NVTX_EVENTS (start, \"end\", eventType, globalTid, domainId, text, textId) \
             VALUES (?, ?, ?, ?, ?, ?, NULL)",
            params![start, end, 59_i64, gtid, 0_i64, name],
        )?;
    }
    let pqtdir = finalize_to_pqtdir(&conn, &dir)?;
    Ok((dir, pqtdir))
}

/// Two-device trace with rank-style env vars in META_DATA_CAPTURE so
/// process-label resolution has something to chew on.
fn build_two_device_trace_with_rank_env() -> Result<(TempDir, PathBuf)> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    conn.execute_batch(
        r#"
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        CREATE TABLE TARGET_INFO_GPU (id BIGINT, cuDevice BIGINT, uuid TEXT);
        CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO (
            deviceId BIGINT,
            contextId BIGINT,
            processId BIGINT
        );
        CREATE TABLE PROCESSES (globalPid BIGINT, pid BIGINT, name TEXT);
        CREATE TABLE META_DATA_CAPTURE (name TEXT, value TEXT);
        CREATE TABLE NVTX_EVENTS (
            start BIGINT,
            "end" BIGINT,
            eventType BIGINT,
            globalTid BIGINT,
            domainId BIGINT,
            text TEXT,
            textId BIGINT
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start BIGINT,
            "end" BIGINT,
            globalTid BIGINT,
            correlationId BIGINT,
            nameId BIGINT
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
    )?;
    install_minimal_export_metadata(&conn)?;
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1_i64, "smoke_kernel"],
    )?;
    for did in [0_i64, 1_i64] {
        conn.execute(
            "INSERT INTO TARGET_INFO_GPU (id, cuDevice, uuid) VALUES (?, ?, ?)",
            params![did + 10_i64, did, format!("synthetic-gpu-{did}")],
        )?;
    }
    conn.execute(
        "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO (deviceId, contextId, processId) \
         VALUES (?, ?, ?), (?, ?, ?)",
        params![0_i64, 100_i64, 4242_i64, 1_i64, 200_i64, 4343_i64],
    )?;
    conn.execute(
        "INSERT INTO PROCESSES (globalPid, pid, name) VALUES (?, ?, ?), (?, ?, ?)",
        params![
            (4242_i64 << 24),
            4242_i64,
            "synthetic-host-0",
            (4343_i64 << 24),
            4343_i64,
            "synthetic-host-1",
        ],
    )?;
    // Two PROCESS_N launch recipes, each carrying a RANK env var so the
    // trace-map projects two `launches[]` with `rank=0` / `rank=1`.
    for (name, value) in [
        ("PROCESS_0:COMMAND", "/usr/bin/synthetic"),
        ("PROCESS_0:ENVIRONMENT_VARIABLE", "RANK=0"),
        ("PROCESS_0:ENVIRONMENT_VARIABLE", "LOCAL_RANK=0"),
        ("PROCESS_1:COMMAND", "/usr/bin/synthetic"),
        ("PROCESS_1:ENVIRONMENT_VARIABLE", "RANK=1"),
        ("PROCESS_1:ENVIRONMENT_VARIABLE", "LOCAL_RANK=1"),
    ] {
        conn.execute(
            "INSERT INTO META_DATA_CAPTURE (name, value) VALUES (?, ?)",
            params![name, value],
        )?;
    }
    for (start, did, sid, pid) in [
        (100_000_000_i64, 0_i64, 7_i64, 4242_i64),
        (200_000_000_i64, 1_i64, 9_i64, 4343_i64),
    ] {
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
             (start, \"end\", deviceId, contextId, streamId, \
              shortName, demangledName, gridX, gridY, gridZ, \
              blockX, blockY, blockZ, correlationId, \
              registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                start,
                start + 1_000_000,
                did,
                if did == 0 { 100_i64 } else { 200_i64 },
                sid,
                1_i64,
                1_i64,
                1_i64,
                1_i64,
                1_i64,
                128_i64,
                1_i64,
                1_i64,
                1_i64,
                32_i64,
                0_i64,
                0_i64,
                pid << 24,
            ],
        )?;
    }
    let pqtdir = finalize_to_pqtdir(&conn, &dir)?;
    Ok((dir, pqtdir))
}

/// `info` on a 2-device fixture surfaces `devices.count == 2` and the
/// sorted device ids.
#[test]
fn info_surfaces_device_inventory() -> Result<()> {
    let (_dir, pqtdir) = build_two_device_trace_with_rank_env()?;
    let out = run_veloq(["info", pqtdir.to_string_lossy().as_ref()])?;
    assert!(
        out.status.success(),
        "exit={:?}; stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
    );
    let v = parse_stdout(&out)?;
    assert_eq!(at(&v, "/data/trace_map/devices/count")?.as_u64(), Some(2));
    let ids = at(&v, "/data/trace_map/devices/ids")?
        .as_array()
        .context("devices.ids must be an array")?;
    let ids: Vec<i64> = ids.iter().filter_map(Value::as_i64).collect();
    assert_eq!(ids, vec![0, 1], "device ids must be sorted ascending");
    Ok(())
}

/// `info` resolves rank-style env vars from `META_DATA_CAPTURE` into
/// per-launch labels.
#[test]
fn info_surfaces_rank_labels_from_env() -> Result<()> {
    let (_dir, pqtdir) = build_two_device_trace_with_rank_env()?;
    let out = run_veloq(["info", pqtdir.to_string_lossy().as_ref()])?;
    assert!(out.status.success());
    let v = parse_stdout(&out)?;
    let launches = at(&v, "/data/trace_map/processes/launches")?
        .as_array()
        .context("processes.launches must be an array")?;
    assert_eq!(launches.len(), 2, "expected one launch per PROCESS_N: {v}");
    let labels: Vec<&str> = launches
        .iter()
        .filter_map(|l| l.pointer("/label").and_then(Value::as_str))
        .collect();
    assert!(
        labels.iter().any(|l| l.contains("rank=0")),
        "labels should mention `rank=0`, got: {labels:?}",
    );
    assert!(
        labels.iter().any(|l| l.contains("rank=1")),
        "labels should mention `rank=1`, got: {labels:?}",
    );
    Ok(())
}

/// `info` on an NVTX-bearing fixture surfaces top_paths aggregation.
#[test]
fn info_surfaces_nvtx_top_paths() -> Result<()> {
    let (_dir, pqtdir) = build_single_device_trace_with_nvtx()?;
    let out = run_veloq(["info", pqtdir.to_string_lossy().as_ref()])?;
    assert!(
        out.status.success(),
        "exit={:?}; stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
    );
    let v = parse_stdout(&out)?;
    let nvtx = at(&v, "/data/trace_map/nvtx")?;
    let top_paths = at(nvtx, "/top_paths")?
        .as_array()
        .context("nvtx.top_paths must be an array")?;
    assert!(
        !top_paths.is_empty(),
        "expected non-empty top_paths from synthetic NVTX ranges: {v}",
    );
    let names: Vec<&str> = top_paths
        .iter()
        .filter_map(|p| p.pointer("/path").and_then(Value::as_str))
        .collect();
    assert!(
        names.iter().any(|n| n.contains("outer_loop")),
        "expected `outer_loop` in top_paths; got: {names:?}",
    );
    Ok(())
}

/// `info` on a path that doesn't resolve to a cached parquetdir falls
/// back to basics + a `meta.next_steps[0]` recommending `veloq prep`.
/// A `.nsys-rep` is the canonical cold input; we write an empty file
/// here so `info` sees the extension and emits the hint without paying
/// the export cost.
#[test]
fn info_on_cold_nsys_rep_emits_prep_next_step() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let cold = dir.path().join("synthetic.nsys-rep");
    std::fs::write(&cold, b"\x00\x00\x00\x00")?; // arbitrary bytes; veloq doesn't open the file
    let out = run_veloq(["info", cold.to_string_lossy().as_ref()])?;
    assert!(
        out.status.success(),
        "exit={:?}; stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
    );
    let v = parse_stdout(&out)?;
    // Basics still ship.
    assert_eq!(at(&v, "/data/exists")?.as_bool(), Some(true));
    assert_eq!(at(&v, "/data/extension")?.as_str(), Some("nsys-rep"));
    // No trace_map (cold path).
    assert!(
        v.pointer("/data/trace_map").is_none(),
        "cold .nsys-rep must not carry trace_map: {v}",
    );
    let cmd = at(&v, "/meta/next_steps/0/command")?
        .as_str()
        .context("next_steps[0].command must be a string")?;
    assert!(
        cmd.starts_with("veloq prep"),
        "next_steps must point at `veloq prep`; got: {cmd}",
    );
    Ok(())
}

/// `stats --device 0` on the 2-device fixture emits a `meta.next_steps`
/// entry derived from the top row.
#[test]
fn stats_emits_next_steps_on_non_empty_result() -> Result<()> {
    let (_dir, pqtdir) = build_two_device_trace_with_rank_env()?;
    let out = run_veloq(["stats", pqtdir.to_string_lossy().as_ref(), "--device", "0"])?;
    assert!(
        out.status.success(),
        "exit={:?}; stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
    );
    let v = parse_stdout(&out)?;
    let steps = at(&v, "/meta/next_steps")?
        .as_array()
        .context("meta.next_steps must be an array")?;
    assert!(
        !steps.is_empty(),
        "expected non-empty next_steps on a populated stats response: {v}",
    );
    let cmd = at(&v, "/meta/next_steps/0/command")?
        .as_str()
        .context("next_steps[0].command must be a string")?;
    assert!(
        cmd.starts_with("veloq search"),
        "stats next_step should drill into `veloq search`; got: {cmd}",
    );
    Ok(())
}
