use std::error::Error;
use std::fs;
use std::path::Path;
use veloq_core::VeloqDiagnostic;
use veloq_core::time::DurationFilter;
use veloq_pytorch_data::{EventType, TraceSet, build_or_load};
use veloq_pytorch_query::{
    EventFilterRequest, RankScope, SliceRow, TypeSelection, TypeToken, collectives, correlate,
    inspect, search, slices, stats, summary, timeline,
};

type Result<T = ()> = std::result::Result<T, Box<dyn Error>>;

fn test_error(message: &'static str) -> Box<dyn Error> {
    std::io::Error::other(message).into()
}

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

fn python_stack_trace() -> Result<(tempfile::TempDir, TraceSet)> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("with_stack.pt.trace.json");
    fs::write(
        &path,
        r#"{
  "traceEvents": [
    { "name": "<stdin>(1): <module>", "cat": "python_function", "ph": "X", "ts": 0, "dur": 1000, "pid": 1, "tid": 10, "args": { "Python id": 1, "Python parent id": null } },
    { "name": "train.py(20): train_step", "cat": "python_function", "ph": "X", "ts": 50, "dur": 800, "pid": 1, "tid": 10, "args": { "Python id": 2, "Python parent id": 1 } },
    { "name": "model.py(42): forward", "cat": "python_function", "ph": "X", "ts": 80, "dur": 300, "pid": 1, "tid": 10, "args": { "Python id": 3, "Python parent id": 2 } },
    { "name": "aten::matmul", "cat": "cpu_op", "ph": "X", "ts": 100, "dur": 100, "pid": 1, "tid": 10, "args": { "External id": 7 } },
    { "name": "cudaLaunchKernel", "cat": "cuda_runtime", "ph": "X", "ts": 130, "dur": 10, "pid": 1, "tid": 10, "args": { "External id": 7, "correlation": 99 } },
    { "name": "void gemm_kernel", "cat": "kernel", "ph": "X", "ts": 180, "dur": 200, "pid": 1, "tid": 8, "args": { "correlation": 99, "device": 0, "stream": 7 } }
  ]
}"#,
    )?;
    let trace = build_or_load(&path)?;
    Ok((dir, trace))
}

fn timeline_bucket_trace() -> Result<(tempfile::TempDir, TraceSet)> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("timeline_buckets.pt.trace.json");
    fs::write(
        &path,
        r#"{
  "traceEvents": [
    { "name": "aten::wide", "cat": "cpu_op", "ph": "X", "ts": 0, "dur": 300, "pid": 1, "tid": 10, "args": {} },
    { "name": "void kernel", "cat": "kernel", "ph": "X", "ts": 100, "dur": 200, "pid": 1, "tid": 8, "args": { "device": 0, "stream": 7 } },
    { "name": "ncclDevKernel_AllReduce", "cat": "kernel", "ph": "X", "ts": 250, "dur": 100, "pid": 1, "tid": 9, "args": { "device": 0, "stream": 8, "rank": 0 } }
  ]
}"#,
    )?;
    let trace = build_or_load(&path)?;
    Ok((dir, trace))
}

fn rank_collision_trace() -> Result<(tempfile::TempDir, TraceSet)> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("collision.pt.trace.json");
    fs::write(
        &path,
        r#"{
  "traceEvents": [
    { "name": "ProfilerStep#1", "cat": "user_annotation", "ph": "X", "ts": 0, "dur": 2000, "pid": 1, "tid": 10, "args": {} },
    { "name": "aten::rank0", "cat": "cpu_op", "ph": "X", "ts": 100, "dur": 100, "pid": 1, "tid": 10, "args": { "External id": 7, "rank": 0 } },
    { "name": "cudaLaunchKernel", "cat": "cuda_runtime", "ph": "X", "ts": 150, "dur": 10, "pid": 1, "tid": 10, "args": { "External id": 7, "correlation": 99, "rank": 0 } },
    { "name": "void gemm_rank0", "cat": "kernel", "ph": "X", "ts": 200, "dur": 100, "pid": 1, "tid": 8, "args": { "External id": 7, "correlation": 99, "rank": 0, "device": 0, "stream": 7 } },
    { "name": "aten::rank1", "cat": "cpu_op", "ph": "X", "ts": 1100, "dur": 100, "pid": 1, "tid": 11, "args": { "External id": 7, "rank": 1 } },
    { "name": "cudaLaunchKernel", "cat": "cuda_runtime", "ph": "X", "ts": 1150, "dur": 10, "pid": 1, "tid": 11, "args": { "External id": 7, "correlation": 99, "rank": 1 } },
    { "name": "void gemm_rank1", "cat": "kernel", "ph": "X", "ts": 1200, "dur": 100, "pid": 1, "tid": 9, "args": { "External id": 7, "correlation": 99, "rank": 1, "device": 0, "stream": 8 } }
  ]
}"#,
    )?;
    let trace = build_or_load(&path)?;
    Ok((dir, trace))
}

fn interleaved_rank_trace() -> Result<(tempfile::TempDir, TraceSet)> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("interleaved.pt.trace.json");
    fs::write(
        &path,
        r#"{
  "traceEvents": [
    { "name": "ProfilerStep#1", "cat": "user_annotation", "ph": "X", "ts": 0, "dur": 3000, "pid": 1, "tid": 10, "args": {} },
    { "name": "aten::rank0", "cat": "cpu_op", "ph": "X", "ts": 100, "dur": 100, "pid": 1, "tid": 10, "args": { "External id": 7, "rank": 0 } },
    { "name": "aten::rank1", "cat": "cpu_op", "ph": "X", "ts": 120, "dur": 100, "pid": 1, "tid": 11, "args": { "External id": 7, "rank": 1 } },
    { "name": "cudaLaunchKernel", "cat": "cuda_runtime", "ph": "X", "ts": 200, "dur": 10, "pid": 1, "tid": 10, "args": { "External id": 7, "correlation": 99, "rank": 0 } },
    { "name": "void gemm_rank0", "cat": "kernel", "ph": "X", "ts": 240, "dur": 100, "pid": 1, "tid": 8, "args": { "External id": 7, "correlation": 99, "rank": 0, "device": 0, "stream": 7 } },
    { "name": "cudaLaunchKernel", "cat": "cuda_runtime", "ph": "X", "ts": 300, "dur": 10, "pid": 1, "tid": 11, "args": { "External id": 7, "correlation": 100, "rank": 1 } },
    { "name": "void gemm_rank1", "cat": "kernel", "ph": "X", "ts": 340, "dur": 100, "pid": 1, "tid": 9, "args": { "External id": 7, "correlation": 100, "rank": 1, "device": 0, "stream": 8 } }
  ]
}"#,
    )?;
    let trace = build_or_load(&path)?;
    Ok((dir, trace))
}

fn unknown_rank_bridge_trace() -> Result<(tempfile::TempDir, TraceSet)> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("unknown_rank_bridge.pt.trace.json");
    fs::write(
        &path,
        r#"{
  "traceEvents": [
    { "name": "aten::rank0", "cat": "cpu_op", "ph": "X", "ts": 100, "dur": 100, "pid": 1, "tid": 10, "args": { "External id": 7, "rank": 0 } },
    { "name": "cudaLaunchKernel", "cat": "cuda_runtime", "ph": "X", "ts": 200, "dur": 10, "pid": 1, "tid": 10, "args": { "External id": 7, "correlation": 99 } },
    { "name": "void unknown_rank_kernel", "cat": "kernel", "ph": "X", "ts": 240, "dur": 100, "pid": 1, "tid": 8, "args": { "External id": 7, "correlation": 99, "device": 0, "stream": 7 } },
    { "name": "aten::rank1", "cat": "cpu_op", "ph": "X", "ts": 400, "dur": 100, "pid": 1, "tid": 11, "args": { "External id": 7, "rank": 1 } }
  ]
}"#,
    )?;
    let trace = build_or_load(&path)?;
    Ok((dir, trace))
}

fn leading_unknown_rank_bridge_trace() -> Result<(tempfile::TempDir, TraceSet)> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("leading_unknown.pt.trace.json");
    fs::write(
        &path,
        r#"{
  "traceEvents": [
    { "name": "cudaLaunchKernel", "cat": "cuda_runtime", "ph": "X", "ts": 100, "dur": 10, "pid": 1, "tid": 10, "args": { "External id": 7, "correlation": 99 } },
    { "name": "aten::rank0", "cat": "cpu_op", "ph": "X", "ts": 200, "dur": 100, "pid": 1, "tid": 10, "args": { "External id": 7, "rank": 0 } },
    { "name": "void gemm_rank0", "cat": "kernel", "ph": "X", "ts": 240, "dur": 100, "pid": 1, "tid": 8, "args": { "External id": 7, "correlation": 99, "rank": 0, "device": 0, "stream": 7 } },
    { "name": "aten::rank1", "cat": "cpu_op", "ph": "X", "ts": 400, "dur": 100, "pid": 1, "tid": 11, "args": { "External id": 7, "rank": 1 } },
    { "name": "void gemm_rank1", "cat": "kernel", "ph": "X", "ts": 440, "dur": 100, "pid": 1, "tid": 9, "args": { "External id": 7, "correlation": 99, "rank": 1, "device": 0, "stream": 8 } }
  ]
}"#,
    )?;
    let trace = build_or_load(&path)?;
    Ok((dir, trace))
}

fn rankless_step_trace() -> Result<(tempfile::TempDir, TraceSet)> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("trace.pt.trace.json");
    fs::write(
        &path,
        r#"{
  "traceEvents": [
    { "name": "ProfilerStep#1", "cat": "user_annotation", "ph": "X", "ts": 0, "dur": 1000, "pid": 1, "tid": 10, "args": {} },
    { "name": "aten::rank0", "cat": "cpu_op", "ph": "X", "ts": 100, "dur": 100, "pid": 1, "tid": 10, "args": { "External id": 7, "rank": 0 } },
    { "name": "void gemm_rank0", "cat": "kernel", "ph": "X", "ts": 200, "dur": 300, "pid": 1, "tid": 8, "args": { "External id": 7, "rank": 0, "device": 0, "stream": 7 } },
    { "name": "ncclDevKernel_AllReduce", "cat": "kernel", "ph": "X", "ts": 600, "dur": 200, "pid": 1, "tid": 9, "args": { "rank": 0, "device": 0, "stream": 8 } }
  ]
}"#,
    )?;
    let trace = build_or_load(&path)?;
    Ok((dir, trace))
}

fn repeated_slice_trace() -> Result<(tempfile::TempDir, TraceSet)> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("repeated_slices.pt.trace.json");
    fs::write(
        &path,
        r#"{
  "traceEvents": [
    { "name": "phase", "cat": "user_annotation", "ph": "X", "ts": 0, "dur": 100, "pid": 1, "tid": 10, "args": {} },
    { "name": "phase", "cat": "user_annotation", "ph": "X", "ts": 200, "dur": 100, "pid": 1, "tid": 10, "args": {} }
  ]
}"#,
    )?;
    let trace = build_or_load(&path)?;
    Ok((dir, trace))
}

fn nested_slice_trace() -> Result<(tempfile::TempDir, TraceSet)> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("nested_slices.pt.trace.json");
    fs::write(
        &path,
        r#"{
  "traceEvents": [
    { "name": "outer", "cat": "user_annotation", "ph": "X", "ts": 0, "dur": 500, "pid": 1, "tid": 10, "args": {} },
    { "name": "inner", "cat": "user_annotation", "ph": "X", "ts": 100, "dur": 100, "pid": 1, "tid": 10, "args": {} }
  ]
}"#,
    )?;
    let trace = build_or_load(&path)?;
    Ok((dir, trace))
}

fn extra_collective_kernel_trace() -> Result<(tempfile::TempDir, TraceSet)> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("extra_collective_kernel.pt.trace.json");
    fs::write(
        &path,
        r#"{
  "traceEvents": [
    { "name": "c10d::allreduce", "cat": "cpu_op", "ph": "X", "ts": 100, "dur": 100, "pid": 1, "tid": 10, "args": { "External id": 8, "rank": 0 } },
    { "name": "ncclDevKernel_AllReduce", "cat": "kernel", "ph": "X", "ts": 200, "dur": 200, "pid": 1, "tid": 8, "args": { "device": 0, "stream": 8, "rank": 0 } },
    { "name": "ncclDevKernel_AllReduce", "cat": "kernel", "ph": "X", "ts": 500, "dur": 200, "pid": 1, "tid": 8, "args": { "device": 0, "stream": 8, "rank": 0 } }
  ]
}"#,
    )?;
    let trace = build_or_load(&path)?;
    Ok((dir, trace))
}

fn multi_rank_collective_trace() -> Result<(tempfile::TempDir, TraceSet)> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("multi_rank_collective.pt.trace.json");
    fs::write(
        &path,
        r#"{
  "traceEvents": [
    { "name": "c10d::allreduce", "cat": "cpu_op", "ph": "X", "ts": 100, "dur": 100, "pid": 1, "tid": 10, "args": { "External id": 8, "rank": 0 } },
    { "name": "ncclDevKernel_AllReduce", "cat": "kernel", "ph": "X", "ts": 200, "dur": 200, "pid": 1, "tid": 8, "args": { "External id": 8, "device": 0, "stream": 8, "rank": 0 } },
    { "name": "c10d::allreduce", "cat": "cpu_op", "ph": "X", "ts": 1000, "dur": 100, "pid": 1, "tid": 11, "args": { "External id": 9, "rank": 1 } },
    { "name": "ncclDevKernel_AllReduce", "cat": "kernel", "ph": "X", "ts": 1100, "dur": 200, "pid": 1, "tid": 9, "args": { "External id": 9, "device": 0, "stream": 9, "rank": 1 } }
  ]
}"#,
    )?;
    let trace = build_or_load(&path)?;
    Ok((dir, trace))
}

fn reused_external_collective_trace() -> Result<(tempfile::TempDir, TraceSet)> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("reused_external_collective.pt.trace.json");
    fs::write(
        &path,
        r#"{
  "traceEvents": [
    { "name": "c10d::allreduce_first", "cat": "cpu_op", "ph": "X", "ts": 100, "dur": 100, "pid": 1, "tid": 10, "args": { "External id": 8, "rank": 0 } },
    { "name": "cudaLaunchKernel", "cat": "cuda_runtime", "ph": "X", "ts": 150, "dur": 10, "pid": 1, "tid": 10, "args": { "External id": 8, "correlation": 101, "rank": 0 } },
    { "name": "ncclDevKernel_AllReduce_first", "cat": "kernel", "ph": "X", "ts": 200, "dur": 200, "pid": 1, "tid": 8, "args": { "External id": 8, "correlation": 101, "device": 0, "stream": 8, "rank": 0 } },
    { "name": "c10d::allreduce_second", "cat": "cpu_op", "ph": "X", "ts": 500, "dur": 100, "pid": 1, "tid": 10, "args": { "External id": 8, "rank": 0 } },
    { "name": "cudaLaunchKernel", "cat": "cuda_runtime", "ph": "X", "ts": 550, "dur": 10, "pid": 1, "tid": 10, "args": { "External id": 8, "correlation": 102, "rank": 0 } },
    { "name": "ncclDevKernel_AllReduce_second", "cat": "kernel", "ph": "X", "ts": 600, "dur": 200, "pid": 1, "tid": 8, "args": { "External id": 8, "correlation": 102, "device": 0, "stream": 8, "rank": 0 } }
  ]
}"#,
    )?;
    let trace = build_or_load(&path)?;
    Ok((dir, trace))
}

fn reused_correlation_collective_trace() -> Result<(tempfile::TempDir, TraceSet)> {
    let dir = tempfile::tempdir()?;
    let path = dir
        .path()
        .join("reused_correlation_collective.pt.trace.json");
    fs::write(
        &path,
        r#"{
  "traceEvents": [
    { "name": "c10d::allreduce_first", "cat": "cpu_op", "ph": "X", "ts": 100, "dur": 100, "pid": 1, "tid": 10, "args": { "External id": 8, "rank": 0 } },
    { "name": "cudaLaunchKernel", "cat": "cuda_runtime", "ph": "X", "ts": 150, "dur": 10, "pid": 1, "tid": 10, "args": { "External id": 8, "correlation": 101, "rank": 0 } },
    { "name": "ncclDevKernel_AllReduce_first", "cat": "kernel", "ph": "X", "ts": 200, "dur": 200, "pid": 1, "tid": 8, "args": { "correlation": 101, "device": 0, "stream": 8, "rank": 0 } },
    { "name": "c10d::allreduce_second", "cat": "cpu_op", "ph": "X", "ts": 500, "dur": 100, "pid": 1, "tid": 10, "args": { "External id": 9, "rank": 0 } },
    { "name": "cudaLaunchKernel", "cat": "cuda_runtime", "ph": "X", "ts": 550, "dur": 10, "pid": 1, "tid": 10, "args": { "External id": 9, "correlation": 101, "rank": 0 } },
    { "name": "ncclDevKernel_AllReduce_second", "cat": "kernel", "ph": "X", "ts": 600, "dur": 200, "pid": 1, "tid": 8, "args": { "correlation": 101, "device": 0, "stream": 8, "rank": 0 } }
  ]
}"#,
    )?;
    let trace = build_or_load(&path)?;
    Ok((dir, trace))
}

fn runtime_driver_correlation_trace() -> Result<(tempfile::TempDir, TraceSet)> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("runtime_driver.pt.trace.json");
    fs::write(
        &path,
        r#"{
  "traceEvents": [
    { "name": "aten::matmul", "cat": "cpu_op", "ph": "X", "ts": 100, "dur": 100, "pid": 1, "tid": 10, "args": { "External id": 7, "rank": 0 } },
    { "name": "cudaLaunchKernel", "cat": "cuda_runtime", "ph": "X", "ts": 150, "dur": 10, "pid": 1, "tid": 10, "args": { "External id": 7, "correlation": 99, "rank": 0 } },
    { "name": "cuLaunchKernel", "cat": "cuda_driver", "ph": "X", "ts": 180, "dur": 10, "pid": 1, "tid": 10, "args": { "correlation": 99, "rank": 0 } },
    { "name": "void gemm_kernel", "cat": "kernel", "ph": "X", "ts": 220, "dur": 100, "pid": 1, "tid": 8, "args": { "correlation": 99, "rank": 0, "device": 0, "stream": 7 } }
  ]
}"#,
    )?;
    let trace = build_or_load(&path)?;
    Ok((dir, trace))
}

#[test]
fn search_returns_keyed_event_refs() -> Result<()> {
    let (_dir, trace) = single_trace()?;
    let response = search(
        &trace.query_trace(),
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
fn search_name_regex_matches_via_sql_path() -> Result<()> {
    let (_dir, trace) = single_trace()?;
    let response = search(
        &trace.query_trace(),
        EventFilterRequest {
            name_regex: Some(".*gemm.*".to_string()),
            ..EventFilterRequest::default()
        },
    )?;

    assert!(
        response
            .rows
            .iter()
            .any(|row| row.name == "void gemm_kernel")
    );
    assert!(response.rows.iter().all(|row| row.name.contains("gemm")));
    Ok(())
}

#[test]
fn summary_exposes_trace_schema_survey() -> Result<()> {
    let (_dir, trace) = single_trace()?;
    let response = summary(&trace);
    assert_eq!(response.auxiliary.schema_survey.raw_event_count, 6);
    assert_eq!(response.auxiliary.schema_survey.parsed_event_count, 6);
    assert_eq!(
        response
            .auxiliary
            .schema_survey
            .phase_counts
            .get("X")
            .copied(),
        Some(6)
    );
    assert_eq!(
        response
            .auxiliary
            .schema_survey
            .typed_arg_coverage
            .device_id,
        2
    );
    Ok(())
}

#[test]
fn inspect_returns_python_context_stack() -> Result<()> {
    let (_dir, trace) = python_stack_trace()?;
    let cpu = trace
        .events
        .iter()
        .find(|event| event.name == "aten::matmul")
        .ok_or_else(|| test_error("missing cpu op"))?;
    let response = inspect(&trace.query_trace(), std::slice::from_ref(&cpu.row_id))?;
    let row = response
        .rows
        .first()
        .ok_or_else(|| test_error("missing inspect row"))?;
    let event = row
        .event
        .as_ref()
        .ok_or_else(|| test_error("missing inspect event"))?;
    let context = event
        .python_context
        .as_ref()
        .ok_or_else(|| test_error("missing python context"))?;
    assert_eq!(context.name, "model.py(42): forward");
    let stack_names = event
        .python_stack
        .iter()
        .map(|frame| frame.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        stack_names,
        vec![
            "<stdin>(1): <module>",
            "train.py(20): train_step",
            "model.py(42): forward"
        ]
    );
    Ok(())
}

#[test]
fn stats_can_group_cpu_ops_by_python_path() -> Result<()> {
    let (_dir, trace) = python_stack_trace()?;
    let response = stats(
        &trace.query_trace(),
        EventFilterRequest {
            types: TypeSelection::Only([TypeToken::Event(EventType::CpuOp)].into_iter().collect()),
            ..EventFilterRequest::default()
        },
        &["python-path".to_string(), "name".to_string()],
    )?;
    let row = response
        .rows
        .iter()
        .find(|row| {
            row.axes
                .get("name")
                .is_some_and(|name| name == "aten::matmul")
        })
        .ok_or_else(|| test_error("missing python-path stats row"))?;
    let path = row
        .axes
        .get("python-path")
        .ok_or_else(|| test_error("missing python-path axis"))?;
    assert!(path.contains("train.py(20): train_step"), "path: {path}");
    assert!(path.contains("model.py(42): forward"), "path: {path}");
    Ok(())
}

#[test]
fn stats_python_axes_require_stack_capture() -> Result<()> {
    let (_dir, trace) = single_trace()?;
    let err = match stats(
        &trace.query_trace(),
        EventFilterRequest::default(),
        &["python-context".to_string()],
    ) {
        Ok(_) => return Err(test_error("expected missing stack error")),
        Err(err) => err,
    };
    let msg = err.to_string();
    assert!(msg.contains("with_stack=True"), "got: {msg}");
    assert_eq!(err.code().as_str(), "pytorch.query.python-stack-missing");
    Ok(())
}

#[test]
fn comm_cpu_ops_remain_cpu_op_refs() -> Result<()> {
    let (_dir, trace) = single_trace()?;
    let response = search(
        &trace.query_trace(),
        EventFilterRequest {
            types: TypeSelection::Only([TypeToken::Event(EventType::CpuOp)].into_iter().collect()),
            is_comm: true,
            ..EventFilterRequest::default()
        },
    )?;
    assert!(
        response
            .rows
            .iter()
            .any(|row| row.name == "c10d::allreduce" && row.event_type == "cpu-op")
    );
    Ok(())
}

#[test]
fn correlate_answers_kernel_launch_cause() -> Result<()> {
    let (_dir, trace) = single_trace()?;
    let kernel = trace
        .events
        .iter()
        .find(|event| event.event_type == EventType::Kernel && event.name.contains("gemm"))
        .ok_or_else(|| test_error("missing kernel event"))?;
    let response = correlate(&trace.query_trace(), std::slice::from_ref(&kernel.row_id))?;
    let row = response
        .rows
        .first()
        .ok_or_else(|| test_error("missing correlate row"))?;
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
fn correlate_preserves_runtime_driver_kernel_chain() -> Result<()> {
    let (_dir, trace) = runtime_driver_correlation_trace()?;
    let runtime = trace
        .events
        .iter()
        .find(|event| event.event_type == EventType::Runtime)
        .ok_or_else(|| test_error("missing runtime event"))?;
    let driver = trace
        .events
        .iter()
        .find(|event| event.event_type == EventType::Driver)
        .ok_or_else(|| test_error("missing driver event"))?;
    let kernel = trace
        .events
        .iter()
        .find(|event| event.event_type == EventType::Kernel)
        .ok_or_else(|| test_error("missing kernel event"))?;

    let response = correlate(&trace.query_trace(), std::slice::from_ref(&kernel.row_id))?;
    let row = response
        .rows
        .first()
        .ok_or_else(|| test_error("missing correlate row"))?;

    assert!(row.events.iter().any(|event| event.name == "aten::matmul"));
    assert!(
        row.events
            .iter()
            .any(|event| event.name == "cudaLaunchKernel" && event.event_type == "runtime")
    );
    assert!(
        row.events
            .iter()
            .any(|event| event.name == "cuLaunchKernel" && event.event_type == "driver")
    );
    assert!(row.links.iter().any(|link| {
        link.kind == "correlation"
            && link.from_row_id == runtime.row_id
            && link.to_row_id == kernel.row_id
    }));
    assert!(row.links.iter().any(|link| {
        link.kind == "correlation"
            && link.from_row_id == driver.row_id
            && link.to_row_id == kernel.row_id
    }));
    Ok(())
}

#[test]
fn correlate_from_cpu_preserves_runtime_driver_kernel_chain() -> Result<()> {
    let (_dir, trace) = runtime_driver_correlation_trace()?;
    let op = trace
        .events
        .iter()
        .find(|event| event.name == "aten::matmul")
        .ok_or_else(|| test_error("missing cpu op"))?;
    let runtime = trace
        .events
        .iter()
        .find(|event| event.event_type == EventType::Runtime)
        .ok_or_else(|| test_error("missing runtime event"))?;
    let driver = trace
        .events
        .iter()
        .find(|event| event.event_type == EventType::Driver)
        .ok_or_else(|| test_error("missing driver event"))?;
    let kernel = trace
        .events
        .iter()
        .find(|event| event.event_type == EventType::Kernel)
        .ok_or_else(|| test_error("missing kernel event"))?;

    let response = correlate(&trace.query_trace(), std::slice::from_ref(&op.row_id))?;
    let row = response
        .rows
        .first()
        .ok_or_else(|| test_error("missing correlate row"))?;

    assert!(
        row.events
            .iter()
            .any(|event| event.name == "cudaLaunchKernel" && event.event_type == "runtime")
    );
    assert!(
        row.events
            .iter()
            .any(|event| event.name == "cuLaunchKernel" && event.event_type == "driver")
    );
    assert!(
        row.events
            .iter()
            .any(|event| event.name == "void gemm_kernel")
    );
    assert!(row.links.iter().any(|link| {
        link.kind == "correlation"
            && link.from_row_id == runtime.row_id
            && link.to_row_id == kernel.row_id
    }));
    assert!(row.links.iter().any(|link| {
        link.kind == "correlation"
            && link.from_row_id == driver.row_id
            && link.to_row_id == kernel.row_id
    }));
    Ok(())
}

#[test]
fn correlate_does_not_cross_known_ranks_on_id_collision() -> Result<()> {
    let (_dir, trace) = rank_collision_trace()?;
    let kernel = trace
        .events
        .iter()
        .find(|event| event.event_type == EventType::Kernel && event.name.contains("rank1"))
        .ok_or_else(|| test_error("missing rank1 kernel event"))?;
    let response = correlate(&trace.query_trace(), std::slice::from_ref(&kernel.row_id))?;
    let row = response
        .rows
        .first()
        .ok_or_else(|| test_error("missing correlate row"))?;
    assert!(row.events.iter().any(|event| event.name == "aten::rank1"));
    assert!(!row.events.iter().any(|event| event.name == "aten::rank0"));
    Ok(())
}

#[test]
fn correlate_handles_interleaved_rank_external_ids() -> Result<()> {
    let (_dir, trace) = interleaved_rank_trace()?;
    let kernel = trace
        .events
        .iter()
        .find(|event| event.event_type == EventType::Kernel && event.name.contains("rank0"))
        .ok_or_else(|| test_error("missing rank0 kernel event"))?;
    let response = correlate(&trace.query_trace(), std::slice::from_ref(&kernel.row_id))?;
    let row = response
        .rows
        .first()
        .ok_or_else(|| test_error("missing correlate row"))?;
    assert!(row.events.iter().any(|event| event.name == "aten::rank0"));
    assert!(!row.events.iter().any(|event| event.name == "aten::rank1"));
    Ok(())
}

#[test]
fn correlate_does_not_use_unknown_rank_as_cross_rank_bridge() -> Result<()> {
    let (_dir, trace) = unknown_rank_bridge_trace()?;
    let op = trace
        .events
        .iter()
        .find(|event| event.name == "aten::rank1")
        .ok_or_else(|| test_error("missing rank1 op"))?;
    let response = correlate(&trace.query_trace(), std::slice::from_ref(&op.row_id))?;
    let row = response
        .rows
        .first()
        .ok_or_else(|| test_error("missing correlate row"))?;
    assert!(row.events.iter().any(|event| event.name == "aten::rank1"));
    assert!(!row.events.iter().any(|event| event.name == "aten::rank0"));
    assert!(
        !row.events
            .iter()
            .any(|event| event.name == "unknown_rank_kernel")
    );
    Ok(())
}

#[test]
fn correlate_does_not_use_leading_unknown_rank_as_cross_rank_bridge() -> Result<()> {
    let (_dir, trace) = leading_unknown_rank_bridge_trace()?;
    let op = trace
        .events
        .iter()
        .find(|event| event.name == "aten::rank1")
        .ok_or_else(|| test_error("missing rank1 op"))?;
    let response = correlate(&trace.query_trace(), std::slice::from_ref(&op.row_id))?;
    let row = response
        .rows
        .first()
        .ok_or_else(|| test_error("missing correlate row"))?;
    assert!(row.events.iter().any(|event| event.name == "aten::rank1"));
    assert!(!row.events.iter().any(|event| event.name == "aten::rank0"));
    assert!(!row.events.iter().any(|event| event.rank.is_none()));
    Ok(())
}

#[test]
fn correlate_does_not_cross_reused_external_id_in_same_rank() -> Result<()> {
    let (_dir, trace) = reused_external_collective_trace()?;
    let kernel = trace
        .events
        .iter()
        .find(|event| event.name == "ncclDevKernel_AllReduce_first")
        .ok_or_else(|| test_error("missing first nccl kernel"))?;
    let response = correlate(&trace.query_trace(), std::slice::from_ref(&kernel.row_id))?;
    let row = response
        .rows
        .first()
        .ok_or_else(|| test_error("missing correlate row"))?;
    assert!(
        row.events
            .iter()
            .any(|event| event.name == "c10d::allreduce_first")
    );
    assert!(
        !row.events
            .iter()
            .any(|event| event.name == "c10d::allreduce_second")
    );
    Ok(())
}

#[test]
fn correlate_does_not_cross_reused_correlation_id_in_same_rank() -> Result<()> {
    let (_dir, trace) = reused_correlation_collective_trace()?;
    let kernel = trace
        .events
        .iter()
        .find(|event| event.name == "ncclDevKernel_AllReduce_first")
        .ok_or_else(|| test_error("missing first nccl kernel"))?;
    let response = correlate(&trace.query_trace(), std::slice::from_ref(&kernel.row_id))?;
    let row = response
        .rows
        .first()
        .ok_or_else(|| test_error("missing correlate row"))?;
    assert!(
        row.events
            .iter()
            .any(|event| event.name == "c10d::allreduce_first")
    );
    assert!(
        !row.events
            .iter()
            .any(|event| event.name == "c10d::allreduce_second")
    );
    assert!(
        !row.events
            .iter()
            .any(|event| event.name == "ncclDevKernel_AllReduce_second")
    );
    Ok(())
}

#[test]
fn stats_can_group_comm_by_kind_and_rank() -> Result<()> {
    let (_dir, trace) = single_trace()?;
    let response = stats(
        &trace.query_trace(),
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
    Ok(())
}

#[test]
fn stats_key_order_is_stable_across_group_by_order() -> Result<()> {
    let (_dir, trace) = single_trace()?;
    let response = stats(
        &trace.query_trace(),
        EventFilterRequest {
            types: TypeSelection::Only([TypeToken::Event(EventType::CpuOp)].into_iter().collect()),
            ..EventFilterRequest::default()
        },
        &["type".to_string(), "name".to_string()],
    )?;
    let row = response
        .rows
        .iter()
        .find(|row| {
            row.axes
                .get("name")
                .is_some_and(|name| name == "aten::matmul")
        })
        .ok_or_else(|| test_error("missing aten::matmul stats row"))?;
    assert_eq!(row.key, "stats|name:aten::matmul|type:cpu-op");
    Ok(())
}

#[test]
fn stats_name_regex_matches_via_sql_path() -> Result<()> {
    let (_dir, trace) = single_trace()?;
    let response = stats(
        &trace.query_trace(),
        EventFilterRequest {
            name_regex: Some("aten::.*".to_string()),
            ..EventFilterRequest::default()
        },
        &["type".to_string(), "name".to_string()],
    )?;

    assert!(response.rows.iter().any(|row| {
        row.axes
            .get("name")
            .is_some_and(|name| name == "aten::matmul")
            && row
                .axes
                .get("type")
                .is_some_and(|event_type| event_type == "cpu-op")
    }));
    Ok(())
}

#[test]
fn stats_empty_group_by_has_no_row_when_filter_matches_nothing() -> Result<()> {
    let (_dir, trace) = single_trace()?;
    let response = stats(
        &trace.query_trace(),
        EventFilterRequest {
            name_glob: Some("does-not-exist*".to_string()),
            ..EventFilterRequest::default()
        },
        &[],
    )?;
    assert_eq!(response.count, 0);
    assert_eq!(response.total_matched, 0);
    assert!(response.rows.is_empty());
    Ok(())
}

#[test]
fn search_stream_filter_requires_device_scope() -> Result<()> {
    let (_dir, trace) = single_trace()?;
    let err = search(
        &trace.query_trace(),
        EventFilterRequest {
            stream: Some(7),
            ..EventFilterRequest::default()
        },
    )
    .err()
    .ok_or_else(|| test_error("expected stream parent-axis error"))?;
    assert_eq!(
        err.code().as_str(),
        "pytorch.query.local-filter-parent-required"
    );
    let message = err.to_string();
    assert!(
        message.contains("--stream 7") && message.contains("--device"),
        "message should explain the missing device parent: {message}",
    );
    Ok(())
}

#[test]
fn multi_rank_device_filter_requires_rank_scope() -> Result<()> {
    let (_dir, trace) = rank_collision_trace()?;
    let err = search(
        &trace.query_trace(),
        EventFilterRequest {
            rank_scope: RankScope {
                rank: None,
                all_ranks: true,
            },
            device: Some(0),
            ..EventFilterRequest::default()
        },
    )
    .err()
    .ok_or_else(|| test_error("expected device parent-axis error"))?;
    assert_eq!(
        err.code().as_str(),
        "pytorch.query.local-filter-parent-required"
    );
    let message = err.to_string();
    assert!(
        message.contains("--device 0") && message.contains("--rank"),
        "message should explain the missing rank parent: {message}",
    );
    Ok(())
}

#[test]
fn multi_rank_device_filter_error_precedes_generic_rank_scope_error() -> Result<()> {
    let (_dir, trace) = rank_collision_trace()?;
    let err = search(
        &trace.query_trace(),
        EventFilterRequest {
            device: Some(0),
            ..EventFilterRequest::default()
        },
    )
    .err()
    .ok_or_else(|| test_error("expected device parent-axis error"))?;
    assert_eq!(
        err.code().as_str(),
        "pytorch.query.local-filter-parent-required"
    );
    let message = err.to_string();
    assert!(
        message.contains("--device 0") && message.contains("--rank"),
        "message should explain the missing rank parent: {message}",
    );
    Ok(())
}

#[test]
fn multi_rank_stream_filter_error_precedes_generic_rank_scope_error() -> Result<()> {
    let (_dir, trace) = rank_collision_trace()?;
    let err = search(
        &trace.query_trace(),
        EventFilterRequest {
            stream: Some(7),
            ..EventFilterRequest::default()
        },
    )
    .err()
    .ok_or_else(|| test_error("expected stream parent-axis error"))?;
    assert_eq!(
        err.code().as_str(),
        "pytorch.query.local-filter-parent-required"
    );
    let message = err.to_string();
    assert!(
        message.contains("--stream 7")
            && message.contains("--rank")
            && message.contains("--device"),
        "message should explain the missing rank/device parents: {message}",
    );
    Ok(())
}

#[test]
fn multi_rank_stats_group_by_device_requires_rank_axis() -> Result<()> {
    let (_dir, trace) = rank_collision_trace()?;
    let err = stats(
        &trace.query_trace(),
        EventFilterRequest {
            rank_scope: RankScope {
                rank: None,
                all_ranks: true,
            },
            ..EventFilterRequest::default()
        },
        &["device".to_string()],
    )
    .err()
    .ok_or_else(|| test_error("expected stats group-by parent-axis error"))?;
    assert_eq!(
        err.code().as_str(),
        "pytorch.query.stats-group-by-parent-required"
    );
    let message = err.to_string();
    assert!(
        message.contains("--group-by device") && message.contains("rank,device"),
        "message should point at safe comparison grouping: {message}",
    );
    Ok(())
}

#[test]
fn multi_rank_stats_group_by_device_error_precedes_generic_rank_scope_error() -> Result<()> {
    let (_dir, trace) = rank_collision_trace()?;
    let err = stats(
        &trace.query_trace(),
        EventFilterRequest::default(),
        &["device".to_string()],
    )
    .err()
    .ok_or_else(|| test_error("expected stats group-by parent-axis error"))?;
    assert_eq!(
        err.code().as_str(),
        "pytorch.query.stats-group-by-parent-required"
    );
    let message = err.to_string();
    assert!(
        message.contains("--group-by device") && message.contains("rank,device"),
        "message should point at safe comparison grouping: {message}",
    );
    Ok(())
}

#[test]
fn stats_group_by_stream_requires_device_axis() -> Result<()> {
    let (_dir, trace) = rank_collision_trace()?;
    let err = stats(
        &trace.query_trace(),
        EventFilterRequest {
            rank_scope: RankScope {
                rank: Some(0),
                all_ranks: false,
            },
            ..EventFilterRequest::default()
        },
        &["stream".to_string()],
    )
    .err()
    .ok_or_else(|| test_error("expected stats stream parent-axis error"))?;
    assert_eq!(
        err.code().as_str(),
        "pytorch.query.stats-group-by-parent-required"
    );
    let message = err.to_string();
    assert!(
        message.contains("--group-by stream") && message.contains("device,stream"),
        "message should point at safe stream grouping: {message}",
    );
    Ok(())
}

#[test]
fn multi_rank_stats_allows_parent_axes_for_device_and_stream_comparison() -> Result<()> {
    let (_dir, trace) = rank_collision_trace()?;
    let response = stats(
        &trace.query_trace(),
        EventFilterRequest {
            rank_scope: RankScope {
                rank: None,
                all_ranks: true,
            },
            ..EventFilterRequest::default()
        },
        &[
            "rank".to_string(),
            "device".to_string(),
            "stream".to_string(),
        ],
    )?;
    assert!(response.count >= 2);
    assert!(response.rows.iter().any(|row| {
        row.axes.get("rank").is_some_and(|rank| rank == "0")
            && row.axes.get("device").is_some_and(|device| device == "0")
            && row.axes.get("stream").is_some_and(|stream| stream == "7")
    }));
    assert!(response.rows.iter().any(|row| {
        row.axes.get("rank").is_some_and(|rank| rank == "1")
            && row.axes.get("device").is_some_and(|device| device == "0")
            && row.axes.get("stream").is_some_and(|stream| stream == "8")
    }));
    Ok(())
}

#[test]
fn timeline_buckets_from_sidecar_preserve_overlap_totals() -> Result<()> {
    let (_dir, trace) = timeline_bucket_trace()?;
    let response = timeline(&trace.query_trace(), EventFilterRequest::default(), 100_000)?;

    assert_eq!(response.count, 4);
    assert_eq!(response.total_matched, 4);
    let bucket = response
        .rows
        .iter()
        .find(|row| row.key == "bucket|200000..300000")
        .ok_or_else(|| test_error("missing bucket 200000..300000"))?;
    assert_eq!(bucket.cpu_ns, 100_000);
    assert_eq!(bucket.gpu_ns, 150_000);
    assert_eq!(bucket.comm_ns, 50_000);
    assert_eq!(bucket.event_count, 3);
    assert_eq!(bucket.by_type_ns.get("cpu-op").copied(), Some(100_000));
    assert_eq!(bucket.by_type_ns.get("kernel").copied(), Some(150_000));
    Ok(())
}

#[test]
fn timeline_name_regex_matches_via_sql_path() -> Result<()> {
    let (_dir, trace) = timeline_bucket_trace()?;
    let response = timeline(
        &trace.query_trace(),
        EventFilterRequest {
            name_regex: Some(".*kernel.*".to_string()),
            ..EventFilterRequest::default()
        },
        100_000,
    )?;

    assert_eq!(response.total_matched, 2);
    assert_eq!(response.count, 2);
    assert!(response.rows.iter().all(|row| row.cpu_ns == 0));
    assert!(response.rows.iter().any(|row| row.gpu_ns > 0));
    Ok(())
}

#[test]
fn collectives_attach_single_trace_comm_evidence() -> Result<()> {
    let (_dir, trace) = single_trace()?;
    let response = collectives(
        &trace.query_trace(),
        RankScope {
            rank: None,
            all_ranks: true,
        },
        None,
        10,
    )?;
    let row = response
        .rows
        .iter()
        .find(|row| row.collective_kind == "all_reduce")
        .ok_or_else(|| test_error("missing all_reduce collective"))?;
    assert!(!response.auxiliary.cross_rank_skew);
    assert_eq!(row.skew_ns, None);
    let rank = row
        .per_rank
        .first()
        .ok_or_else(|| test_error("missing collective timing"))?;
    assert!(rank.cpu_row_id.is_some());
    assert!(!rank.kernel_row_ids.is_empty());
    assert!(rank.event_row_ids.len() >= 2);
    Ok(())
}

#[test]
fn collectives_have_unique_keys_across_ranks() -> Result<()> {
    let (_dir, trace) = multi_rank_collective_trace()?;
    let response = collectives(
        &trace.query_trace(),
        RankScope {
            rank: None,
            all_ranks: true,
        },
        None,
        10,
    )?;
    let keys = response
        .rows
        .iter()
        .map(|row| row.key.clone())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(keys.len(), response.rows.len());
    assert!(keys.iter().any(|key| key.contains("rank:0")));
    assert!(keys.iter().any(|key| key.contains("rank:1")));
    Ok(())
}

#[test]
fn collectives_requires_rank_scope_for_multi_rank_trace() -> Result<()> {
    let (_dir, trace) = multi_rank_collective_trace()?;
    let err = collectives(
        &trace.query_trace(),
        RankScope {
            rank: None,
            all_ranks: false,
        },
        None,
        10,
    )
    .err()
    .ok_or_else(|| test_error("expected collectives rank-scope error"))?;
    assert!(err.to_string().contains("--rank <n>"));
    assert_eq!(err.code().as_str(), "pytorch.query.rank-scope-required");
    Ok(())
}

#[test]
fn collectives_honor_rank_scope() -> Result<()> {
    let (_dir, trace) = multi_rank_collective_trace()?;
    let response = collectives(
        &trace.query_trace(),
        RankScope {
            rank: Some(1),
            all_ranks: false,
        },
        None,
        10,
    )?;
    assert!(!response.rows.is_empty());
    assert!(response.rows.iter().all(|row| row.key.contains("rank:1")));
    assert!(!response.rows.iter().any(|row| row.key.contains("rank:0")));
    Ok(())
}

#[test]
fn collectives_preserve_extra_unlinked_kernels() -> Result<()> {
    let (_dir, trace) = extra_collective_kernel_trace()?;
    let response = collectives(
        &trace.query_trace(),
        RankScope {
            rank: None,
            all_ranks: false,
        },
        None,
        10,
    )?;
    assert!(
        response
            .rows
            .iter()
            .any(|row| row.collective_kind == "all_reduce" && row.confidence == "ordinal")
    );
    assert!(
        response
            .rows
            .iter()
            .any(|row| row.collective_kind == "all_reduce" && row.confidence == "kernel-only")
    );
    Ok(())
}

#[test]
fn collectives_do_not_cross_reused_external_id_in_same_rank() -> Result<()> {
    let (_dir, trace) = reused_external_collective_trace()?;
    let response = collectives(
        &trace.query_trace(),
        RankScope {
            rank: None,
            all_ranks: false,
        },
        None,
        10,
    )?;
    let first_kernel = trace
        .events
        .iter()
        .find(|event| event.name == "ncclDevKernel_AllReduce_first")
        .ok_or_else(|| test_error("missing first nccl kernel"))?;
    let second_kernel = trace
        .events
        .iter()
        .find(|event| event.name == "ncclDevKernel_AllReduce_second")
        .ok_or_else(|| test_error("missing second nccl kernel"))?;
    let first = response
        .rows
        .iter()
        .find(|row| {
            row.per_rank
                .iter()
                .any(|timing| timing.name == "c10d::allreduce_first")
        })
        .ok_or_else(|| test_error("missing first collective row"))?;
    let first_timing = first
        .per_rank
        .first()
        .ok_or_else(|| test_error("missing first collective timing"))?;
    assert_eq!(first_timing.kernel_row_ids.len(), 1);
    assert_eq!(
        first_timing.kernel_row_ids,
        vec![first_kernel.row_id.clone()]
    );

    let second = response
        .rows
        .iter()
        .find(|row| {
            row.per_rank
                .iter()
                .any(|timing| timing.name == "c10d::allreduce_second")
        })
        .ok_or_else(|| test_error("missing second collective row"))?;
    let second_timing = second
        .per_rank
        .first()
        .ok_or_else(|| test_error("missing second collective timing"))?;
    assert_eq!(second_timing.kernel_row_ids.len(), 1);
    assert_eq!(
        second_timing.kernel_row_ids,
        vec![second_kernel.row_id.clone()]
    );
    Ok(())
}

#[test]
fn collectives_do_not_cross_reused_correlation_id_in_same_rank() -> Result<()> {
    let (_dir, trace) = reused_correlation_collective_trace()?;
    let response = collectives(
        &trace.query_trace(),
        RankScope {
            rank: None,
            all_ranks: false,
        },
        None,
        10,
    )?;
    let first_kernel = trace
        .events
        .iter()
        .find(|event| event.name == "ncclDevKernel_AllReduce_first")
        .ok_or_else(|| test_error("missing first nccl kernel"))?;
    let second_kernel = trace
        .events
        .iter()
        .find(|event| event.name == "ncclDevKernel_AllReduce_second")
        .ok_or_else(|| test_error("missing second nccl kernel"))?;
    let first = response
        .rows
        .iter()
        .find(|row| {
            row.per_rank
                .iter()
                .any(|timing| timing.name == "c10d::allreduce_first")
        })
        .ok_or_else(|| test_error("missing first collective row"))?;
    let first_timing = first
        .per_rank
        .first()
        .ok_or_else(|| test_error("missing first collective timing"))?;
    assert_eq!(first_timing.kernel_row_ids.len(), 1);
    assert_eq!(
        first_timing.kernel_row_ids,
        vec![first_kernel.row_id.clone()]
    );

    let second = response
        .rows
        .iter()
        .find(|row| {
            row.per_rank
                .iter()
                .any(|timing| timing.name == "c10d::allreduce_second")
        })
        .ok_or_else(|| test_error("missing second collective row"))?;
    let second_timing = second
        .per_rank
        .first()
        .ok_or_else(|| test_error("missing second collective timing"))?;
    assert_eq!(second_timing.kernel_row_ids.len(), 1);
    assert_eq!(
        second_timing.kernel_row_ids,
        vec![second_kernel.row_id.clone()]
    );
    Ok(())
}

#[test]
fn slices_attribute_rankless_steps_to_ranked_events() -> Result<()> {
    let (_dir, trace) = rankless_step_trace()?;
    let response = slices(
        &trace.query_trace(),
        EventFilterRequest {
            types: TypeSelection::Only([TypeToken::Event(EventType::Step)].into_iter().collect()),
            ..EventFilterRequest::default()
        },
        false,
        None,
    )?;
    let step = response
        .rows
        .iter()
        .find_map(|row| match row {
            veloq_pytorch_query::SliceRow::Instance(row) if row.name == "ProfilerStep#1" => {
                Some(row)
            }
            _ => None,
        })
        .ok_or_else(|| test_error("missing step slice"))?;
    assert_eq!(step.attributed_gpu_ns, 500_000);
    assert_eq!(step.attributed_comm_ns, 200_000);
    Ok(())
}

#[test]
fn slices_rank_scope_keeps_rankless_steps_with_ranked_children() -> Result<()> {
    let (_dir, trace) = rankless_step_trace()?;
    let response = slices(
        &trace.query_trace(),
        EventFilterRequest {
            types: TypeSelection::Only([TypeToken::Event(EventType::Step)].into_iter().collect()),
            rank_scope: RankScope {
                rank: Some(0),
                all_ranks: false,
            },
            ..EventFilterRequest::default()
        },
        false,
        None,
    )?;
    let step = response
        .rows
        .iter()
        .find_map(|row| match row {
            veloq_pytorch_query::SliceRow::Instance(row) if row.name == "ProfilerStep#1" => {
                Some(row)
            }
            _ => None,
        })
        .ok_or_else(|| test_error("missing rankless step slice"))?;
    assert_eq!(step.attributed_gpu_ns, 500_000);
    assert_eq!(step.attributed_comm_ns, 200_000);
    Ok(())
}

#[test]
fn slices_aggregate_total_matched_counts_groups() -> Result<()> {
    let (_dir, trace) = repeated_slice_trace()?;
    let response = slices(
        &trace.query_trace(),
        EventFilterRequest::default(),
        true,
        Some("name".to_string()),
    )?;
    assert_eq!(response.count, 1);
    assert_eq!(response.total_matched, 1);
    Ok(())
}

#[test]
fn slices_child_count_uses_sidecar_parent_links() -> Result<()> {
    let (_dir, trace) = nested_slice_trace()?;
    let response = slices(
        &trace.query_trace(),
        EventFilterRequest {
            name_glob: Some("outer".to_string()),
            ..EventFilterRequest::default()
        },
        false,
        None,
    )?;
    let row = response
        .rows
        .iter()
        .find_map(|row| match row {
            veloq_pytorch_query::SliceRow::Instance(row) => Some(row),
            _ => None,
        })
        .ok_or_else(|| test_error("missing nested outer slice"))?;
    assert_eq!(row.child_count, 1);
    Ok(())
}

#[test]
fn slices_respect_time_window() -> Result<()> {
    let (_dir, trace) = repeated_slice_trace()?;
    let response = slices(
        &trace.query_trace(),
        EventFilterRequest {
            time_window_ns: Some((150_000, 250_000)),
            ..EventFilterRequest::default()
        },
        false,
        None,
    )?;
    assert_eq!(response.count, 1);
    assert_eq!(response.total_matched, 1);
    assert_eq!(response.auxiliary.time_window_ns, Some((150_000, 250_000)));
    let row = response
        .rows
        .iter()
        .find_map(|row| match row {
            veloq_pytorch_query::SliceRow::Instance(row) => Some(row),
            _ => None,
        })
        .ok_or_else(|| test_error("missing windowed slice row"))?;
    assert_eq!(row.start_ns, 200_000);
    Ok(())
}

#[test]
fn slices_name_regex_matches_via_sql_path() -> Result<()> {
    let (_dir, trace) = repeated_slice_trace()?;
    let response = slices(
        &trace.query_trace(),
        EventFilterRequest {
            name_regex: Some("ph.*".to_string()),
            ..EventFilterRequest::default()
        },
        false,
        None,
    )?;

    assert_eq!(response.total_matched, 2);
    assert!(response.rows.iter().all(|row| match row {
        SliceRow::Instance(row) => row.name == "phase",
        SliceRow::Aggregate(_) => false,
    }));
    Ok(())
}

#[test]
fn slices_rank_scope_ignores_children_outside_time_window() -> Result<()> {
    let (_dir, trace) = rankless_step_trace()?;
    let response = slices(
        &trace.query_trace(),
        EventFilterRequest {
            types: TypeSelection::Only([TypeToken::Event(EventType::Step)].into_iter().collect()),
            rank_scope: RankScope {
                rank: Some(0),
                all_ranks: false,
            },
            time_window_ns: Some((850_000, 900_000)),
            ..EventFilterRequest::default()
        },
        false,
        None,
    )?;
    assert_eq!(response.count, 0);
    assert_eq!(response.total_matched, 0);
    Ok(())
}

#[test]
fn slices_attribution_is_clipped_to_time_window() -> Result<()> {
    let (_dir, trace) = rankless_step_trace()?;
    let response = slices(
        &trace.query_trace(),
        EventFilterRequest {
            types: TypeSelection::Only([TypeToken::Event(EventType::Step)].into_iter().collect()),
            time_window_ns: Some((250_000, 650_000)),
            ..EventFilterRequest::default()
        },
        false,
        None,
    )?;
    let step = response
        .rows
        .iter()
        .find_map(|row| match row {
            veloq_pytorch_query::SliceRow::Instance(row) if row.name == "ProfilerStep#1" => {
                Some(row)
            }
            _ => None,
        })
        .ok_or_else(|| test_error("missing windowed step slice"))?;
    assert_eq!(step.duration_ns, 1_000_000);
    assert_eq!(step.attributed_gpu_ns, 300_000);
    assert_eq!(step.attributed_comm_ns, 50_000);
    Ok(())
}

#[test]
fn slices_aggregate_reports_full_cpu_duration_under_time_window() -> Result<()> {
    let (_dir, trace) = rankless_step_trace()?;
    let response = slices(
        &trace.query_trace(),
        EventFilterRequest {
            types: TypeSelection::Only([TypeToken::Event(EventType::Step)].into_iter().collect()),
            time_window_ns: Some((250_000, 650_000)),
            ..EventFilterRequest::default()
        },
        true,
        Some("name".to_string()),
    )?;
    let row = response
        .rows
        .iter()
        .find_map(|row| match row {
            veloq_pytorch_query::SliceRow::Aggregate(row) if row.scope == "ProfilerStep#1" => {
                Some(row)
            }
            _ => None,
        })
        .ok_or_else(|| test_error("missing aggregate step slice"))?;
    assert_eq!(row.total_cpu_ns, 1_000_000);
    assert_eq!(row.total_gpu_ns, 300_000);
    assert_eq!(row.total_comm_ns, 50_000);
    Ok(())
}

#[test]
fn slices_rejects_unknown_aggregate_group_by() -> Result<()> {
    let (_dir, trace) = single_trace()?;
    let err = slices(
        &trace.query_trace(),
        EventFilterRequest {
            types: TypeSelection::Only([TypeToken::Event(EventType::Step)].into_iter().collect()),
            ..EventFilterRequest::default()
        },
        true,
        Some("rank".to_string()),
    )
    .err()
    .ok_or_else(|| test_error("expected group-by error"))?;
    assert!(err.to_string().contains("--group-by"));
    assert_eq!(err.code().as_str(), "pytorch.query.unknown-slices-group-by");
    Ok(())
}
