use anyhow::Result;
use std::fs;
use std::path::Path;
use veloq_core::time::DurationFilter;
use veloq_pytorch_data::{EventType, TraceSet, build_or_load};
use veloq_pytorch_query::{
    EventFilterRequest, RankScope, TypeSelection, TypeToken, collectives, correlate, search, stats,
};

fn trace_json(rank: i64, offset_us: i64) -> String {
    format!(
        r#"{{
  "distributedInfo": {{ "rank": {rank}, "worker": "worker-{rank}" }},
  "traceEvents": [
    {{ "name": "ProfilerStep#1", "cat": "user_annotation", "ph": "X", "ts": {offset_us}, "dur": 1000, "pid": 1, "tid": 10, "args": {{ "rank": {rank} }} }},
    {{ "name": "aten::matmul", "cat": "cpu_op", "ph": "X", "ts": {cpu_us}, "dur": 200, "pid": 1, "tid": 10, "args": {{ "External id": 7, "Input Shapes": "[32,32]" }} }},
    {{ "name": "cudaLaunchKernel", "cat": "cuda_runtime", "ph": "X", "ts": {rt_us}, "dur": 20, "pid": 1, "tid": 10, "args": {{ "External id": 7, "correlation": 99 }} }},
    {{ "name": "void gemm_kernel", "cat": "kernel", "ph": "X", "ts": {kernel_us}, "dur": 300, "pid": 1, "tid": 7, "args": {{ "External id": 7, "correlation": 99, "device": 0, "stream": 7 }} }},
    {{ "name": "c10d::allreduce", "cat": "cpu_op", "ph": "X", "ts": {comm_us}, "dur": 100, "pid": 1, "tid": 10, "args": {{ "External id": 8, "rank": {rank} }} }},
    {{ "name": "ncclDevKernel_AllReduce", "cat": "kernel", "ph": "X", "ts": {nccl_us}, "dur": 200, "pid": 1, "tid": 8, "args": {{ "correlation": 100, "device": 0, "stream": 8, "rank": {rank} }} }}
  ]
}}"#,
        cpu_us = offset_us + 100,
        rt_us = offset_us + 150,
        kernel_us = offset_us + 200,
        comm_us = offset_us + 500,
        nccl_us = offset_us + 600,
    )
}

fn write_trace(path: &Path, rank: i64, offset_us: i64) -> Result<()> {
    fs::write(path, trace_json(rank, offset_us))?;
    Ok(())
}

fn single_trace() -> Result<(tempfile::TempDir, TraceSet)> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("rank0.pt.trace.json");
    write_trace(&path, 0, 0)?;
    let trace = build_or_load(&path)?;
    Ok((dir, trace))
}

fn multi_rank_trace() -> Result<(tempfile::TempDir, TraceSet)> {
    let dir = tempfile::tempdir()?;
    write_trace(&dir.path().join("rank0.pt.trace.json"), 0, 0)?;
    write_trace(&dir.path().join("rank1.pt.trace.json"), 1, 10_000)?;
    let trace = build_or_load(dir.path())?;
    Ok((dir, trace))
}

#[test]
fn search_returns_keyed_event_refs() -> Result<()> {
    let (_dir, trace) = single_trace()?;
    let response = search(
        &trace,
        EventFilterRequest {
            types: TypeSelection::Only([TypeToken::Event(EventType::Kernel)].into_iter().collect()),
            duration: Some(DurationFilter::Gte(1)),
            ..EventFilterRequest::default()
        },
    )?;
    assert!(response.count >= 1);
    assert!(response.rows.iter().all(|row| !row.key.is_empty()));
    assert!(response.rows.iter().all(|row| row.event_type == "kernel"));
    Ok(())
}

#[test]
fn multi_rank_search_requires_explicit_rank_scope() -> Result<()> {
    let (_dir, trace) = multi_rank_trace()?;
    let err = search(&trace, EventFilterRequest::default())
        .err()
        .ok_or_else(|| anyhow::anyhow!("expected multi-rank scope error"))?;
    assert!(err.to_string().contains("--rank"));

    let response = search(
        &trace,
        EventFilterRequest {
            rank_scope: RankScope {
                rank: None,
                all_ranks: true,
            },
            ..EventFilterRequest::default()
        },
    )?;
    assert!(response.count > 0);
    Ok(())
}

#[test]
fn correlate_answers_kernel_launch_cause() -> Result<()> {
    let (_dir, trace) = single_trace()?;
    let kernel = trace
        .events
        .iter()
        .find(|event| event.event_type == EventType::Kernel && event.name.contains("gemm"))
        .ok_or_else(|| anyhow::anyhow!("missing kernel event"))?;
    let response = correlate(&trace, std::slice::from_ref(&kernel.row_id));
    let row = response
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("missing correlate row"))?;
    assert!(row.events.iter().any(|event| event.name == "aten::matmul"));
    assert!(
        row.events
            .iter()
            .any(|event| event.name == "cudaLaunchKernel")
    );
    assert!(row.links.iter().any(|link| link.kind == "correlation"));
    Ok(())
}

#[test]
fn stats_can_group_comm_by_rank() -> Result<()> {
    let (_dir, trace) = multi_rank_trace()?;
    let response = stats(
        &trace,
        EventFilterRequest {
            types: TypeSelection::Only([TypeToken::Comm].into_iter().collect()),
            rank_scope: RankScope {
                rank: None,
                all_ranks: true,
            },
            ..EventFilterRequest::default()
        },
        &["comm-kind".to_string(), "rank".to_string()],
    )?;
    assert!(
        response
            .rows
            .iter()
            .any(|row| row.key.contains("all_reduce"))
    );
    assert!(response.rows.iter().any(|row| row.key.contains("rank:0")));
    assert!(response.rows.iter().any(|row| row.key.contains("rank:1")));
    Ok(())
}

#[test]
fn collectives_report_per_rank_skew() -> Result<()> {
    let (_dir, trace) = multi_rank_trace()?;
    let response = collectives(
        &trace,
        RankScope {
            rank: None,
            all_ranks: true,
        },
        None,
        10,
    );
    assert!(
        response
            .rows
            .iter()
            .any(|row| row.per_rank.len() == 2 && row.collective_kind == "all_reduce")
    );
    Ok(())
}
