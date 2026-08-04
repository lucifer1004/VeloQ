use super::{assert_error_code, finalize_to_pqtdir, run_veloq};
use anyhow::{Context, Result};
use duckdb::Connection;
use serde_json::Value;
use std::path::PathBuf;
use tempfile::TempDir;

fn build_nvtx_only_trace() -> Result<(TempDir, PathBuf)> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    conn.execute_batch(
        r#"
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT);
        CREATE TABLE NVTX_EVENTS (
            start BIGINT,
            "end" BIGINT,
            eventType BIGINT,
            globalTid BIGINT,
            domainId BIGINT,
            text TEXT,
            textId BIGINT
        );
        INSERT INTO META_DATA_EXPORT VALUES
            ('EXPORT_SCHEMA_VERSION_MAJOR', '3'),
            ('EXPORT_SCHEMA_VERSION_MINOR', '0'),
            ('EXPORT_SCHEMA_VERSION_MICRO', '0');
        INSERT INTO NVTX_EVENTS VALUES
            (100, 200, 59, 70464307201, 7, 'host phase', NULL);
        "#,
    )?;
    let pqtdir = finalize_to_pqtdir(&conn, &dir)?;
    Ok((dir, pqtdir))
}

fn build_single_device_host_trace() -> Result<(TempDir, PathBuf)> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    conn.execute_batch(
        r#"
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO (
            deviceId BIGINT, contextId BIGINT, processId BIGINT
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start BIGINT, "end" BIGINT, deviceId BIGINT, contextId BIGINT,
            streamId BIGINT, shortName BIGINT, demangledName BIGINT,
            correlationId BIGINT, globalPid BIGINT
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start BIGINT, "end" BIGINT, globalTid BIGINT,
            correlationId BIGINT, nameId BIGINT
        );
        CREATE TABLE OSRT_API (
            start BIGINT, "end" BIGINT, globalTid BIGINT, nameId BIGINT
        );
        CREATE TABLE NVTX_EVENTS (
            start BIGINT, "end" BIGINT, eventType BIGINT, globalTid BIGINT,
            domainId BIGINT, text TEXT, textId BIGINT
        );
        INSERT INTO StringIds VALUES
            (1, 'kernel'), (2, 'cudaLaunchKernel'), (3, 'poll');
        INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO VALUES (0, 1, 4200);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
            (100, 110, 0, 1, 7, 1, 1, 9, 70464307200);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
            (20, 30, 70464307201, 9, 2);
        INSERT INTO OSRT_API VALUES (40, 50, 70464307201, 3);
        INSERT INTO NVTX_EVENTS VALUES
            (10, 60, 59, 70464307201, 0, 'host phase', NULL);
        "#,
    )?;
    let pqtdir = finalize_to_pqtdir(&conn, &dir)?;
    Ok((dir, pqtdir))
}

fn parse_success(out: &std::process::Output, command: &str) -> Result<Value> {
    assert!(
        out.status.success(),
        "{command} failed: stdout={}; stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let value: Value = serde_json::from_slice(&out.stdout)?;
    assert_eq!(value.get("command").and_then(Value::as_str), Some(command));
    Ok(value)
}

#[test]
fn nvtx_only_trace_supports_summary_search_stats_and_inspect() -> Result<()> {
    let (_dir, trace) = build_nvtx_only_trace()?;
    let trace = trace.to_string_lossy();

    let summary = parse_success(&run_veloq(["summary", trace.as_ref()])?, "nsys.summary")?;
    assert_eq!(
        summary.pointer("/data/rows/0/name").and_then(Value::as_str),
        Some("NVTX_EVENTS")
    );

    let search = parse_success(
        &run_veloq(["search", trace.as_ref(), "--type", "nvtx"])?,
        "nsys.search",
    )?;
    assert_eq!(
        search.pointer("/data/rows/0/type").and_then(Value::as_str),
        Some("nvtx")
    );
    let row_id = search
        .pointer("/data/rows/0/row_id")
        .and_then(Value::as_str)
        .context("NVTX search row must carry row_id")?;

    let stats = parse_success(
        &run_veloq(["stats", trace.as_ref(), "--type", "nvtx"])?,
        "nsys.stats",
    )?;
    assert_eq!(
        stats.pointer("/data/rows/0/type").and_then(Value::as_str),
        Some("nvtx")
    );

    let inspect = parse_success(
        &run_veloq(["inspect", trace.as_ref(), row_id])?,
        "nsys.inspect",
    )?;
    assert_eq!(
        inspect.pointer("/data/rows/0/type").and_then(Value::as_str),
        Some("nvtx")
    );
    Ok(())
}

#[test]
fn nvtx_only_slices_reports_its_missing_cuda_evidence() -> Result<()> {
    let (_dir, trace) = build_nvtx_only_trace()?;
    let out = run_veloq(["slices", trace.to_string_lossy().as_ref()])?;
    let value = assert_error_code(&out, "nsys.query.slices-prereq-missing")?;
    assert!(
        value
            .pointer("/error/message")
            .and_then(Value::as_str)
            .is_some_and(|message| message.contains("CUPTI_ACTIVITY_KIND_RUNTIME"))
    );
    Ok(())
}

#[test]
fn explicit_host_kinds_do_not_inherit_the_single_cuda_device() -> Result<()> {
    let (_dir, trace) = build_single_device_host_trace()?;
    let trace = trace.to_string_lossy();

    for verb in ["stats", "search"] {
        for kind in ["nvtx", "runtime", "osrt"] {
            let response = parse_success(
                &run_veloq([verb, trace.as_ref(), "--type", kind])?,
                &format!("nsys.{verb}"),
            )?;
            assert_eq!(
                response.pointer("/data/count").and_then(Value::as_u64),
                Some(1),
                "{verb} --type {kind}: {response}"
            );
            assert!(
                response.pointer("/meta/applied_scope/device").is_none(),
                "{verb} --type {kind} inherited a device: {response}"
            );
        }
    }
    Ok(())
}

#[test]
fn explicit_cuda_location_on_host_kind_is_still_rejected() -> Result<()> {
    let (_dir, trace) = build_single_device_host_trace()?;
    let out = run_veloq([
        "search",
        trace.to_string_lossy().as_ref(),
        "--type",
        "nvtx",
        "--device",
        "0",
    ])?;
    assert_error_code(&out, "nsys.query.kind-location-filter-conflict")?;
    Ok(())
}
