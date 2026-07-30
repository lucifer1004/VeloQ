//! Differential coverage for the daemon session's disposable GPU interval
//! index. The same resident `Trace` first exercises the established paths,
//! then builds the index and repeats changing-argument scan queries.

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
    concurrency::{
        ConcurrencyRequest, run_with_index as run_indexed_concurrency,
        run_with_trace as run_concurrency,
    },
    gaps::{GapScope, GapsRequest, run_with_index as run_indexed_gaps, run_with_trace as run_gaps},
    resident_intervals,
    timeline::{
        TimelineRequest, run_with_index as run_indexed_timeline, run_with_trace as run_timeline,
    },
};

const UNBOUNDED_TEST_CAPACITY_BYTES: u64 = u64::MAX;

#[test]
fn resident_index_reuses_one_exact_process_qualified_interval_set() -> Result<()> {
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
    assert!(
        resident_intervals::build(&trace, 0)?.is_none(),
        "an index that cannot fit the configured resident capacity must be bypassed"
    );
    let index = resident_intervals::build(&trace, UNBOUNDED_TEST_CAPACITY_BYTES)?
        .context("fixture intervals should be representable by the resident index")?;
    assert!(index.retained_memory_estimate_bytes() > 0);

    assert_eq!(
        serde_json::to_value(run_indexed_timeline(&trace, &index, timeline_request)?)?,
        expected_timeline
    );
    assert_eq!(
        serde_json::to_value(run_indexed_concurrency(
            &trace,
            &index,
            concurrency_request
        )?)?,
        expected_concurrency
    );
    assert_eq!(
        serde_json::to_value(run_indexed_gaps(&trace, &index, gaps_request)?)?,
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
fn resident_index_preserves_multistream_windows_and_all_gap_scopes() -> Result<()> {
    let fixture = fixture::concurrency_overlap()?;
    let trace = Trace::open(fixture.path())?;
    let index = build_index(&trace)?;

    for request in [
        TimelineRequest {
            interval_ns: 25_000_000,
            limit: 3,
            ..Default::default()
        },
        TimelineRequest {
            interval_ns: 17_000_000,
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            time_window: Some(TimeWindow::parse("@25ms-@105ms")?),
            process_id: Some(12345),
            device: Some(0),
            stream: Some(8),
            limit: 2,
            ..Default::default()
        },
    ] {
        assert_eq!(
            serde_json::to_value(run_indexed_timeline(&trace, &index, request.clone())?)?,
            serde_json::to_value(run_timeline(&trace, request)?)?
        );
    }

    for request in [
        ConcurrencyRequest::default(),
        ConcurrencyRequest {
            process_id: Some(12345),
            device: Some(0),
            time_window: Some(TimeWindow::parse("@30ms-@90ms")?),
            limit: 1,
        },
        ConcurrencyRequest {
            process_id: None,
            device: None,
            time_window: Some(TimeWindow::parse("@200ms-@300ms")?),
            limit: 10,
        },
    ] {
        assert_eq!(
            serde_json::to_value(run_indexed_concurrency(&trace, &index, request.clone())?)?,
            serde_json::to_value(run_concurrency(&trace, request)?)?
        );
    }

    for request in [
        GapsRequest {
            min_ns: 1,
            scope: GapScope::Device,
            sort: Some(SortSpec::parse("start:asc")?),
            ..Default::default()
        },
        GapsRequest {
            min_ns: 1,
            scope: GapScope::Stream,
            process_id: Some(12345),
            device: Some(0),
            stream: Some(8),
            time_window: Some(TimeWindow::parse("@30ms-@120ms")?),
            sort: Some(SortSpec::parse("duration:desc,start:asc")?),
            limit: 1,
        },
        GapsRequest {
            min_ns: 1,
            scope: GapScope::Trace,
            process_id: Some(12345),
            sort: Some(SortSpec::parse("start:desc")?),
            ..Default::default()
        },
    ] {
        assert_eq!(
            serde_json::to_value(run_indexed_gaps(&trace, &index, request.clone())?)?,
            serde_json::to_value(run_gaps(&trace, request)?)?
        );
    }
    Ok(())
}

#[test]
fn resident_index_preserves_multidevice_and_process_private_scopes() -> Result<()> {
    for fixture in [
        fixture::concurrency_two_devices()?,
        fixture::process_private_cuda_identity_collision()?,
    ] {
        let trace = Trace::open(fixture.path())?;
        let index = build_index(&trace)?;
        for process_id in [None, Some(1001), Some(2002), Some(12345)] {
            let concurrency = ConcurrencyRequest {
                process_id,
                device: None,
                time_window: None,
                limit: 10,
            };
            assert_eq!(
                serde_json::to_value(run_indexed_concurrency(
                    &trace,
                    &index,
                    concurrency.clone()
                )?)?,
                serde_json::to_value(run_concurrency(&trace, concurrency)?)?
            );

            let gaps = GapsRequest {
                min_ns: 1,
                scope: GapScope::Trace,
                process_id,
                sort: Some(SortSpec::single("start")),
                limit: 10,
                ..Default::default()
            };
            assert_eq!(
                serde_json::to_value(run_indexed_gaps(&trace, &index, gaps.clone())?)?,
                serde_json::to_value(run_gaps(&trace, gaps)?)?
            );
        }
    }
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
        resident_intervals::build(&trace, UNBOUNDED_TEST_CAPACITY_BYTES)?.is_none(),
        "unresolved process ownership or a non-positive interval must reject the whole index"
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

fn build_index(trace: &Trace) -> Result<resident_intervals::ResidentIntervalIndex> {
    veloq_nsys_data::gpu_work_events::ensure_sidecar(trace)?;
    resident_intervals::build(trace, UNBOUNDED_TEST_CAPACITY_BYTES)?
        .context("fixture intervals should build a resident index")
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
