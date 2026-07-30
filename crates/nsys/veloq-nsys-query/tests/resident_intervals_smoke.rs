//! Differential coverage for the daemon session's disposable GPU interval
//! view. The same resident `Trace` first exercises the established paths,
//! then registers the TEMP view and repeats changing-argument scan queries.

mod fixture;

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use duckdb::{Connection, params};
use tempfile::TempDir;
use veloq_core::{SortSpec, artifact_dir_for, time::TimeWindow};
use veloq_nsys_data::Trace;
use veloq_nsys_query::{
    EventKind, KindFilter,
    concurrency::{ConcurrencyRequest, run_with_trace as run_concurrency},
    gaps::{GapsRequest, run_with_trace as run_gaps},
    resident_intervals,
    timeline::{TimelineRequest, run_with_trace as run_timeline},
};

#[test]
fn resident_view_reuses_one_exact_process_qualified_interval_set() -> Result<()> {
    let fixture = fixture::minimal_gpu()?;
    let trace = Trace::open(fixture.path())?;

    let timeline_request = TimelineRequest {
        interval_ns: 8_000_000,
        kinds: KindFilter::Only(vec![EventKind::Kernel, EventKind::Memcpy]),
        time_window: Some(TimeWindow::parse("@100ms-@145ms")?),
        process_id: Some(12345),
        device: Some(0),
        stream: Some(7),
        limit: 2,
        ..Default::default()
    };
    let concurrency_request = ConcurrencyRequest {
        process_id: Some(12345),
        device: Some(0),
        time_window: Some(TimeWindow::parse("@105ms-@135ms")?),
        limit: 1,
    };
    let gaps_request = GapsRequest {
        min_ns: 100_000,
        process_id: Some(12345),
        device: Some(0),
        time_window: Some(TimeWindow::parse("@100ms-@145ms")?),
        sort: Some(SortSpec::single("start")),
        limit: 2,
        ..Default::default()
    };

    let expected_timeline = serde_json::to_value(run_timeline(&trace, timeline_request.clone())?)?;
    let expected_concurrency =
        serde_json::to_value(run_concurrency(&trace, concurrency_request.clone())?)?;
    let expected_gaps = serde_json::to_value(run_gaps(&trace, gaps_request.clone())?)?;

    veloq_nsys_data::gpu_work_events::ensure_sidecar(&trace)?;
    let artifact_root = artifact_dir_for(trace.path());
    let artifacts_before = relative_files(&artifact_root)?;
    let first = resident_intervals::ensure(&trace)?
        .context("fixture intervals should be representable by the resident view")?;
    let second = resident_intervals::ensure(&trace)?
        .context("the already-built resident view should remain available")?;
    assert_eq!(
        first, second,
        "ensure must reuse the same session-local view"
    );
    assert!(first.accounted_bytes > 0);

    assert_eq!(
        serde_json::to_value(run_timeline(&trace, timeline_request)?)?,
        expected_timeline
    );
    assert_eq!(
        serde_json::to_value(run_concurrency(&trace, concurrency_request)?)?,
        expected_concurrency
    );
    assert_eq!(
        serde_json::to_value(run_gaps(&trace, gaps_request)?)?,
        expected_gaps
    );
    assert_eq!(
        relative_files(&artifact_root)?,
        artifacts_before,
        "the resident view must not publish a persistent artifact"
    );
    Ok(())
}

#[test]
fn unrepresentable_rows_leave_the_established_query_path_active() -> Result<()> {
    let (_root, path) = unrepresentable_trace()?;
    let trace = Trace::open(&path)?;
    let request = TimelineRequest {
        interval_ns: 10,
        ..Default::default()
    };
    let expected = serde_json::to_value(run_timeline(&trace, request.clone())?)?;
    trace.conn().execute_batch(
        "CREATE VIEW nsight.gpu_work_events AS \
         SELECT \
             'kernel'::VARCHAR AS kind, \
             rowid::BIGINT AS row_id, \
             CASE WHEN contextId = 1 THEN 12345::BIGINT ELSE NULL::BIGINT END AS process_id, \
             CAST(deviceId AS INTEGER) AS device_id, \
             CAST(streamId AS BIGINT) AS stream_id, \
             CAST(start AS BIGINT) AS start_ns, \
             CAST(\"end\" AS BIGINT) AS end_ns \
         FROM nsight.CUPTI_ACTIVITY_KIND_KERNEL",
    )?;

    assert!(
        resident_intervals::ensure(&trace)?.is_none(),
        "unresolved process ownership or a non-positive interval must reject the whole view"
    );
    assert_eq!(
        serde_json::to_value(run_timeline(&trace, request)?)?,
        expected,
        "a rejected resident build must preserve the established query result"
    );
    Ok(())
}

fn relative_files(root: &Path) -> Result<BTreeSet<PathBuf>> {
    let mut files = BTreeSet::new();
    if !root.exists() {
        return Ok(files);
    }
    collect_relative_files(root, root, &mut files)?;
    Ok(files)
}

fn collect_relative_files(
    root: &Path,
    current: &Path,
    files: &mut BTreeSet<PathBuf>,
) -> Result<()> {
    for entry in fs::read_dir(current)
        .with_context(|| format!("read artifact directory {}", current.display()))?
    {
        let path = entry?.path();
        if path.is_dir() {
            collect_relative_files(root, &path, files)?;
        } else {
            files.insert(path.strip_prefix(root)?.to_path_buf());
        }
    }
    Ok(())
}

fn unrepresentable_trace() -> Result<(TempDir, PathBuf)> {
    let root = tempfile::tempdir()?;
    let connection = Connection::open_in_memory()?;
    connection.execute_batch(fixture::KERNEL_TABLE_SQL)?;
    connection.execute_batch(
        "CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO (\
             deviceId BIGINT, contextId BIGINT, processId BIGINT\
         )",
    )?;
    connection.execute(
        "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO VALUES (?, ?, ?)",
        params![0_i32, 1_i64, 12345_i64],
    )?;
    for (start_ns, end_ns, context_id) in [(100_i64, 120_i64, 1_i64), (130, 140, 2), (150, 150, 1)]
    {
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (\
                 start, \"end\", deviceId, contextId, streamId, correlationId\
             ) VALUES (?, ?, ?, ?, ?, ?)",
            params![start_ns, end_ns, 0_i32, context_id, 7_i64, start_ns],
        )?;
    }

    let path = root.path().join("unrepresentable_pqtdir");
    fs::create_dir(&path)?;
    for table in [
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "TARGET_INFO_CUDA_CONTEXT_INFO",
    ] {
        let destination = path
            .join(format!("{table}.parquet"))
            .to_string_lossy()
            .replace('\'', "''");
        connection.execute_batch(&format!("COPY {table} TO '{destination}' (FORMAT PARQUET)"))?;
    }
    Ok((root, path))
}
