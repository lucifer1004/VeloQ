//! End-to-end smoke tests for the scope foundation:
//! envelope meta block, the `--device` / `--all-devices`
//! ambiguity-refuse rule, the `device → native_pid` cross-axis bridge,
//! and the `veloq recipes` meta verb.
//!
//! Drives the freshly-built `veloq` binary via `Command::new(...)` so
//! every test exercises the CLI parser + dispatcher + envelope emit
//! path end-to-end.

use anyhow::{Context, Result};
use duckdb::{Connection, params};
use serde_json::Value;
use std::path::PathBuf;
use std::process::{Command, Output};
use tempfile::TempDir;

/// Path to the freshly-built `veloq` binary, courtesy of Cargo.
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

/// COPY every user-created table in the in-memory DuckDB connection
/// to `<dir>/test_pqtdir/<TABLE>.parquet` and return the parquetdir
/// path. Same shape as `cli_smoke.rs::finalize_to_pqtdir`; kept local
/// so this binary test doesn't share private state across files.
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

/// Build a synthetic two-process parquetdir trace. Two kernel rows, two
/// `PROCESSES` rows, and two
/// `TARGET_INFO_CUDA_CONTEXT_INFO` rows so the resolver can map
/// `deviceId = 0` → `native_pid = 4242` and `deviceId = 1` →
/// `native_pid = 4343`.
fn build_two_device_trace() -> Result<(TempDir, PathBuf)> {
    build_two_process_trace(1)
}

/// Same two-process trace, but both processes expose logical device 0.
fn build_colliding_logical_device_trace() -> Result<(TempDir, PathBuf)> {
    build_two_process_trace(0)
}

fn build_two_process_trace(second_logical_device: i64) -> Result<(TempDir, PathBuf)> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    conn.execute_batch(
        r#"
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT);
        CREATE TABLE TARGET_INFO_GPU (id BIGINT, cuDevice BIGINT, uuid TEXT);
        CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO (
            deviceId BIGINT,
            contextId BIGINT,
            processId BIGINT
        );
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
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1_i64, "smoke_kernel"],
    )?;
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
    for (physical_id, logical_device) in [(10_i64, 0_i64), (11_i64, second_logical_device)] {
        conn.execute(
            "INSERT INTO TARGET_INFO_GPU (id, cuDevice, uuid) VALUES (?, ?, ?)",
            params![
                physical_id,
                logical_device,
                format!("synthetic-gpu-{physical_id}")
            ],
        )?;
    }
    // The resolver surfaces the matched native_pid on
    // `meta.applied_scope.native_pid`, including when both processes
    // reuse logical ordinal 0.
    conn.execute(
        "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO (deviceId, contextId, processId) \
         VALUES (?, ?, ?), (?, ?, ?)",
        params![
            0_i64,
            100_i64,
            4242_i64,
            second_logical_device,
            200_i64,
            4343_i64
        ],
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
    // One kernel per process-local CUDA scope.
    for (start, did, context, sid, pid) in [
        (100_000_000_i64, 0_i64, 100_i64, 7_i64, 4242_i64),
        (
            200_000_000_i64,
            second_logical_device,
            200_i64,
            9_i64,
            4343_i64,
        ),
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
                context,
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

fn parse_stdout(out: &Output) -> Result<Value> {
    serde_json::from_slice(&out.stdout).context("veloq stdout must be valid JSON")
}

fn at<'a>(v: &'a Value, ptr: &str) -> Result<&'a Value> {
    v.pointer(ptr)
        .ok_or_else(|| anyhow::anyhow!("missing pointer `{ptr}` in {v}"))
}

/// Multi-device trace + no `--device` and no `--all-devices` → the
/// resolver refuses. The error envelope carries `meta.warnings` with
/// the structured `multi-device-ambiguous` code.
#[test]
fn stats_refuses_multi_device_without_flag() -> Result<()> {
    let (_dir, pqtdir) = build_two_device_trace()?;
    let out = run_veloq(["stats", pqtdir.to_string_lossy().as_ref()])?;
    assert_eq!(
        out.status.code(),
        Some(1),
        "expected exit 1, got {:?}; stdout={}; stderr={}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let v = parse_stdout(&out)?;
    assert_eq!(at(&v, "/schema")?.as_str(), Some("v1"));
    assert_eq!(
        at(&v, "/error/code")?.as_str(),
        Some("nsys.query.multi-device-ambiguous"),
        "got: {v}",
    );
    let hint = at(&v, "/error/hint")?
        .as_str()
        .context("error.hint must be a string")?;
    assert!(hint.contains("--all-devices"), "hint: {hint}");
    assert!(
        hint.contains("--process 4242") && hint.contains("--device 0"),
        "hint must contain an exact process/device recovery scope: {hint}",
    );
    let msg = at(&v, "/error/message")?
        .as_str()
        .context("error.message must be a string")?;
    assert!(msg.contains("2 CUDA process/device scopes"), "got: {msg}");
    assert!(
        msg.contains("--device") && msg.contains("--all-devices"),
        "message must mention both flags: {msg}",
    );
    // Structured warning code lives on `meta.warnings[0].code`.
    assert_eq!(
        at(&v, "/meta/warnings/0/code")?.as_str(),
        Some("multi-device-ambiguous"),
        "got: {v}",
    );
    let aggregate_command = at(&v, "/meta/next_steps/0/command")?
        .as_str()
        .context("first next_steps command must be a string")?;
    assert!(
        aggregate_command.contains("veloq stats")
            && aggregate_command.contains("--all-devices")
            && aggregate_command.contains(pqtdir.to_string_lossy().as_ref()),
        "aggregate next step should rerun this stats query: {aggregate_command}",
    );
    let device_command = at(&v, "/meta/next_steps/1/command")?
        .as_str()
        .context("second next_steps command must be a string")?;
    assert!(
        device_command.contains("veloq stats")
            && device_command.contains("--process 4242")
            && device_command.contains("--device 0")
            && device_command.contains(pqtdir.to_string_lossy().as_ref()),
        "device next step should rerun this stats query: {device_command}",
    );
    Ok(())
}

#[test]
fn colliding_device_zero_requires_process_and_exact_scope_succeeds() -> Result<()> {
    let (_dir, pqtdir) = build_colliding_logical_device_trace()?;

    let ambiguous = run_veloq(["stats", pqtdir.to_string_lossy().as_ref(), "--device", "0"])?;
    assert_eq!(ambiguous.status.code(), Some(1));
    let error = parse_stdout(&ambiguous)?;
    assert_eq!(
        at(&error, "/error/code")?.as_str(),
        Some("nsys.query.multi-device-ambiguous")
    );
    let exact_command = at(&error, "/meta/next_steps/1/command")?
        .as_str()
        .context("exact next step must be a string")?;
    assert!(
        exact_command.contains("--process 4242") && exact_command.contains("--device 0"),
        "exact recovery command must select both axes: {exact_command}"
    );

    let exact = run_veloq([
        "stats",
        pqtdir.to_string_lossy().as_ref(),
        "--process",
        "4242",
        "--device",
        "0",
    ])?;
    assert!(
        exact.status.success(),
        "exact process/device selection failed: stdout={}; stderr={}",
        String::from_utf8_lossy(&exact.stdout),
        String::from_utf8_lossy(&exact.stderr)
    );
    let response = parse_stdout(&exact)?;
    assert_eq!(at(&response, "/data/rows/0/count")?.as_u64(), Some(1));
    assert_eq!(
        at(&response, "/meta/applied_scope/native_pid")?.as_i64(),
        Some(4242)
    );
    assert_eq!(
        at(&response, "/meta/applied_scope/device")?.as_i64(),
        Some(0)
    );
    let follow_up = at(&response, "/meta/next_steps/0/command")?
        .as_str()
        .context("stats follow-up command must be a string")?;
    assert!(
        follow_up.contains("--process 4242") && follow_up.contains("--device 0"),
        "scoped follow-up must preserve both identity axes: {follow_up}"
    );
    Ok(())
}

#[test]
fn graph_replays_refuses_multi_device_without_flag() -> Result<()> {
    let (_dir, pqtdir) = build_two_device_trace()?;
    let out = run_veloq(["graph-replays", pqtdir.to_string_lossy().as_ref()])?;
    assert_eq!(out.status.code(), Some(1));
    let v = parse_stdout(&out)?;
    assert_eq!(
        at(&v, "/meta/warnings/0/code")?.as_str(),
        Some("multi-device-ambiguous"),
        "got: {v}",
    );
    let aggregate_command = at(&v, "/meta/next_steps/0/command")?
        .as_str()
        .context("first next_steps command must be a string")?;
    assert!(
        aggregate_command.contains("veloq graph-replays")
            && aggregate_command.contains("--all-devices"),
        "aggregate next step should preserve the graph-replays verb: {aggregate_command}",
    );
    let device_command = at(&v, "/meta/next_steps/1/command")?
        .as_str()
        .context("second next_steps command must be a string")?;
    assert!(
        device_command.contains("veloq graph-replays") && device_command.contains("--device 0"),
        "device next step should preserve the graph-replays verb: {device_command}",
    );
    Ok(())
}

/// Multi-device trace + `--device 0` → success. The envelope's
/// `meta.applied_scope` reports `device = 0` AND the cross-axis
/// `native_pid = 4242` (the host pid that ran on device 0, via the
/// `TARGET_INFO_CUDA_CONTEXT_INFO` bridge).
#[test]
fn unambiguous_device_preserves_process_optional_behavior() -> Result<()> {
    let (_dir, pqtdir) = build_two_device_trace()?;
    let out = run_veloq(["stats", pqtdir.to_string_lossy().as_ref(), "--device", "0"])?;
    assert!(
        out.status.success(),
        "an unambiguous bare --device must remain sufficient; exit {:?}; stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
    );
    let v = parse_stdout(&out)?;
    assert_eq!(at(&v, "/schema")?.as_str(), Some("v1"));
    assert_eq!(at(&v, "/meta/applied_scope/device")?.as_i64(), Some(0));
    assert_eq!(
        at(&v, "/meta/applied_scope/native_pid")?.as_i64(),
        Some(4242),
        "native_pid must echo the host pid that ran on the picked device: {v}",
    );
    // No opt-in aggregate → `aggregated_over` is empty / absent.
    assert!(
        v.pointer("/meta/applied_scope/aggregated_over").is_none()
            || at(&v, "/meta/applied_scope/aggregated_over")?
                .as_array()
                .is_some_and(|a| a.is_empty()),
        "no aggregation opt-in, but `aggregated_over` populated: {v}",
    );
    Ok(())
}

/// Multi-device trace + `--all-devices` → success aggregate.
/// `applied_scope.device` is absent (null), `aggregated_over =
/// ["process", "device"]`.
#[test]
fn stats_with_all_devices_marks_process_and_device_aggregation() -> Result<()> {
    let (_dir, pqtdir) = build_two_device_trace()?;
    let out = run_veloq(["stats", pqtdir.to_string_lossy().as_ref(), "--all-devices"])?;
    assert!(
        out.status.success(),
        "expected success, got exit {:?}; stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
    );
    let v = parse_stdout(&out)?;
    // `device` field absent (`None` serialises as omitted).
    assert!(
        v.pointer("/meta/applied_scope/device").is_none(),
        "expected no `device` field under --all-devices; got: {v}",
    );
    let agg = at(&v, "/meta/applied_scope/aggregated_over")?
        .as_array()
        .context("aggregated_over must be an array")?;
    assert_eq!(
        agg.iter().filter_map(Value::as_str).collect::<Vec<_>>(),
        vec!["process", "device"]
    );
    Ok(())
}

#[test]
fn concurrency_defaults_to_all_devices_on_multi_device_trace() -> Result<()> {
    let (_dir, pqtdir) = build_two_device_trace()?;
    let out = run_veloq(["concurrency", pqtdir.to_string_lossy().as_ref()])?;
    assert!(
        out.status.success(),
        "expected concurrency to default to per-device rows on a multi-device trace; exit={:?}; stdout={}; stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let v = parse_stdout(&out)?;
    assert_eq!(at(&v, "/data/count")?.as_u64(), Some(2));
    assert!(
        v.pointer("/meta/applied_scope/device").is_none(),
        "implicit all-device concurrency should not lock a single device: {v}",
    );
    assert_eq!(
        at(&v, "/meta/applied_scope/aggregated_over")?
            .as_array()
            .context("aggregated_over must be an array")?
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>(),
        vec!["process", "device"],
        "implicit all-device concurrency should mark both process-local axes: {v}",
    );
    Ok(())
}

#[test]
fn gaps_trace_scope_defaults_to_all_devices_on_multi_device_trace() -> Result<()> {
    let (_dir, pqtdir) = build_two_device_trace()?;
    let out = run_veloq([
        "gaps",
        pqtdir.to_string_lossy().as_ref(),
        "--scope",
        "trace",
        "--min-duration",
        "1ns",
    ])?;
    assert!(
        out.status.success(),
        "expected trace-scope gaps to default to all devices; exit={:?}; stdout={}; stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let v = parse_stdout(&out)?;
    assert_eq!(at(&v, "/data/scope")?.as_str(), Some("trace"));
    assert_eq!(
        at(&v, "/meta/applied_scope/aggregated_over")?
            .as_array()
            .context("aggregated_over must be an array")?
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>(),
        vec!["process", "device"],
        "trace-scope gaps should mark both implicit aggregate axes: {v}",
    );
    assert!(
        v.pointer("/data/rows/0/device_id").is_none(),
        "trace-scope gap rows should not carry a single device id: {v}",
    );
    Ok(())
}

#[test]
fn gaps_stream_scope_error_precedes_multi_device_ambiguity() -> Result<()> {
    let (_dir, pqtdir) = build_two_device_trace()?;
    let out = run_veloq(["gaps", pqtdir.to_string_lossy().as_ref(), "--stream", "7"])?;
    assert_eq!(
        out.status.code(),
        Some(1),
        "expected stream-scope error; stdout={}; stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let v = parse_stdout(&out)?;
    assert_eq!(
        at(&v, "/error/code")?.as_str(),
        Some("nsys.query.gaps-stream-scope-required"),
        "stream/scope validation should run before multi-device ambiguity: {v}",
    );
    assert!(
        v.pointer("/meta/warnings/0/code").is_none(),
        "parameter validation should not pretend to be an implicit all-device scope: {v}",
    );
    Ok(())
}

#[test]
fn stats_stream_scope_error_precedes_multi_device_ambiguity() -> Result<()> {
    let (_dir, pqtdir) = build_two_device_trace()?;
    let out = run_veloq(["stats", pqtdir.to_string_lossy().as_ref(), "--stream", "7"])?;
    assert_eq!(
        out.status.code(),
        Some(1),
        "expected stream/device scope error; stdout={}; stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let v = parse_stdout(&out)?;
    assert_eq!(
        at(&v, "/error/code")?.as_str(),
        Some("nsys.data.scope-stream-requires-device"),
        "--stream should not be masked by multi-device ambiguity: {v}",
    );
    assert!(
        v.pointer("/meta/warnings/0/code").is_none(),
        "stream parent-axis validation should not emit ambiguity warning: {v}",
    );
    Ok(())
}

#[test]
fn gaps_stream_scope_stream_filter_requires_device_before_ambiguity() -> Result<()> {
    let (_dir, pqtdir) = build_two_device_trace()?;
    let out = run_veloq([
        "gaps",
        pqtdir.to_string_lossy().as_ref(),
        "--scope",
        "stream",
        "--stream",
        "7",
    ])?;
    assert_eq!(
        out.status.code(),
        Some(1),
        "expected stream/device scope error; stdout={}; stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let v = parse_stdout(&out)?;
    assert_eq!(
        at(&v, "/error/code")?.as_str(),
        Some("nsys.data.scope-stream-requires-device"),
        "stream-scope gaps still require a device parent: {v}",
    );
    Ok(())
}

#[test]
fn stream_filter_requires_single_device_scope() -> Result<()> {
    let (_dir, pqtdir) = build_two_device_trace()?;
    let out = run_veloq([
        "stats",
        pqtdir.to_string_lossy().as_ref(),
        "--all-devices",
        "--stream",
        "7",
    ])?;
    assert_eq!(
        out.status.code(),
        Some(1),
        "expected stream/device scope error; stdout={}; stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let v = parse_stdout(&out)?;
    assert_eq!(
        at(&v, "/error/code")?.as_str(),
        Some("nsys.data.scope-stream-requires-device"),
        "--stream should not filter across all devices: {v}",
    );
    let message = at(&v, "/error/message")?
        .as_str()
        .context("error.message must be a string")?;
    assert!(
        message.contains("--stream 7") && message.contains("--device"),
        "message should explain the missing device scope: {message}",
    );
    Ok(())
}

#[test]
fn stats_group_by_stream_requires_device_parent_axis() -> Result<()> {
    let (_dir, pqtdir) = build_two_device_trace()?;
    let out = run_veloq([
        "stats",
        pqtdir.to_string_lossy().as_ref(),
        "--all-devices",
        "--group-by",
        "stream",
    ])?;
    assert_eq!(
        out.status.code(),
        Some(1),
        "expected group-by parent-axis error; stdout={}; stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let v = parse_stdout(&out)?;
    assert_eq!(
        at(&v, "/error/code")?.as_str(),
        Some("nsys.query.stats-group-by-device-parent-required"),
        "got: {v}",
    );
    let message = at(&v, "/error/message")?
        .as_str()
        .context("error.message must be a string")?;
    assert!(
        message.contains("--group-by stream") && message.contains("device,stream"),
        "message should point at safe comparison grouping: {message}",
    );
    Ok(())
}

#[test]
fn stats_all_devices_group_by_device_stream_is_valid_comparison() -> Result<()> {
    let (_dir, pqtdir) = build_two_device_trace()?;
    let out = run_veloq([
        "stats",
        pqtdir.to_string_lossy().as_ref(),
        "--all-devices",
        "--group-by",
        "device,stream",
    ])?;
    assert!(
        out.status.success(),
        "expected grouped comparison to succeed; exit={:?}; stdout={}; stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let v = parse_stdout(&out)?;
    assert_eq!(at(&v, "/data/count")?.as_u64(), Some(2));
    assert_eq!(
        at(&v, "/meta/applied_scope/aggregated_over")?
            .as_array()
            .context("aggregated_over must be an array")?
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>(),
        vec!["process", "device"],
        "all-device comparison should mark both process-local axes: {v}",
    );
    let rows = at(&v, "/data/rows")?
        .as_array()
        .context("data.rows must be an array")?;
    assert!(
        rows.iter()
            .any(|row| row.pointer("/device_id").and_then(Value::as_i64) == Some(0)),
        "expected a device 0 row: {v}",
    );
    assert!(
        rows.iter()
            .any(|row| row.pointer("/device_id").and_then(Value::as_i64) == Some(1)),
        "expected a device 1 row: {v}",
    );
    Ok(())
}

#[test]
fn stats_by_size_group_by_context_requires_device_parent_axis() -> Result<()> {
    let (_dir, pqtdir) = build_two_device_trace()?;
    let out = Command::new(veloq_bin())
        .env("VELOQ_UNSTABLE", "1")
        .args([
            "stats",
            pqtdir.to_string_lossy().as_ref(),
            "--by",
            "size",
            "--all-devices",
            "--group-by",
            "context",
        ])
        .output()
        .context("spawn veloq binary")?;
    assert_eq!(
        out.status.code(),
        Some(1),
        "expected group-by parent-axis error; stdout={}; stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let v = parse_stdout(&out)?;
    assert_eq!(
        at(&v, "/error/code")?.as_str(),
        Some("nsys.query.stats-group-by-device-parent-required"),
        "got: {v}",
    );
    Ok(())
}

/// `--device` and `--all-devices` together are rejected by clap at
/// parse time (per the `conflicts_with` arg-group declaration).
#[test]
fn device_and_all_devices_are_mutually_exclusive_at_parse_time() -> Result<()> {
    let (_dir, pqtdir) = build_two_device_trace()?;
    let out = run_veloq([
        "stats",
        pqtdir.to_string_lossy().as_ref(),
        "--device",
        "0",
        "--all-devices",
    ])?;
    assert_ne!(
        out.status.code(),
        Some(0),
        "expected non-zero exit on conflicting args",
    );
    let combined = String::from_utf8_lossy(&out.stdout) + String::from_utf8_lossy(&out.stderr);
    assert!(
        combined.contains("--all-devices") || combined.contains("--device"),
        "expected the parse error to mention the conflicting flag; got stdout+stderr: {combined}",
    );
    Ok(())
}

/// `slices` accepts the new scope flags now that `GpuLocationFilters`
/// is flattened into the variant. Clap-level smoke: just confirm the
/// parser doesn't reject the combination.
#[test]
fn slices_accepts_device_and_stream_flags() -> Result<()> {
    let (_dir, pqtdir) = build_two_device_trace()?;
    // The fixture has no NVTX events, so slices returns an empty list
    // — that's still a successful parse + run.
    let out = run_veloq([
        "slices",
        pqtdir.to_string_lossy().as_ref(),
        "--device",
        "0",
        "--stream",
        "7",
    ])?;
    assert!(
        out.status.success(),
        "expected `slices --device 0 --stream 7` to parse and run; exit={:?}; stdout={}; stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let v = parse_stdout(&out)?;
    assert_eq!(at(&v, "/meta/applied_scope/device")?.as_i64(), Some(0));
    assert_eq!(at(&v, "/meta/applied_scope/stream")?.as_i64(), Some(7));
    Ok(())
}

/// `veloq recipes` lists the registered recipes; the test asserts the
/// envelope shape and a non-empty registry without pinning the exact count.
#[test]
fn recipes_lists_registered_workflows() -> Result<()> {
    let out = run_veloq(["recipes"])?;
    assert!(
        out.status.success(),
        "exit={:?}; stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
    );
    let v = parse_stdout(&out)?;
    assert_eq!(at(&v, "/schema")?.as_str(), Some("v1"));
    assert_eq!(at(&v, "/source/kind")?.as_str(), Some("veloq"));
    assert_eq!(at(&v, "/command")?.as_str(), Some("recipes"));
    let count = at(&v, "/data/count")?
        .as_u64()
        .context("data.count must be unsigned")?;
    assert!(
        count >= 8,
        "expected the populated registry; got count {count}"
    );
    let rows = at(&v, "/data/rows")?
        .as_array()
        .context("data.rows must be an array")?;
    assert_eq!(rows.len() as u64, count);
    Ok(())
}

/// `veloq recipes <unknown-id>` returns an error envelope with exit
/// code 1 and a message naming the unknown id.
#[test]
fn recipes_show_unknown_id_errors() -> Result<()> {
    let out = run_veloq(["recipes", "no-such-recipe"])?;
    assert_eq!(out.status.code(), Some(1));
    assert!(
        out.stderr.is_empty(),
        "json meta error should keep stderr quiet; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v = parse_stdout(&out)?;
    let msg = at(&v, "/error/message")?
        .as_str()
        .context("error.message must be a string")?;
    assert_eq!(at(&v, "/error/code")?.as_str(), Some("meta.unknown-recipe"));
    assert_eq!(
        at(&v, "/error/hint")?.as_str(),
        Some("run `veloq recipes` to list registered ids")
    );
    assert!(
        msg.contains("no-such-recipe"),
        "error must name the missing id: {msg}",
    );
    Ok(())
}
