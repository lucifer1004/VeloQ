use anyhow::Result;
use flate2::{Compression, write::GzEncoder};
use std::fs;
use std::io::Write;
use std::path::Path;
use veloq_pytorch_data::{EventType, build_or_load, detect_path, sidecar_states};

fn trace_json(rank: i64, offset_us: i64) -> String {
    format!(
        r#"{{
  "schemaVersion": "1",
  "distributedInfo": {{ "rank": {rank}, "worker": "worker-{rank}" }},
  "cudaVersion": "12.4",
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

#[test]
fn detects_and_ingests_json_trace() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let trace_path = dir.path().join("worker0.pt.trace.json");
    write_trace(&trace_path, 0, 0)?;

    assert!(detect_path(&trace_path));
    let trace = build_or_load(&trace_path)?;
    assert_eq!(trace.files.len(), 1);
    assert!(
        trace
            .events
            .iter()
            .any(|event| event.event_type == EventType::CpuOp)
    );
    assert!(
        trace
            .events
            .iter()
            .any(|event| event.event_type == EventType::Kernel)
    );
    assert!(trace.capabilities.active_devices.contains(&0));
    assert!(trace.capabilities.has_comm_events);
    assert!(trace.envelope_trace_span().is_some());
    assert!(
        sidecar_states(&trace_path)
            .iter()
            .all(|sidecar| sidecar.present),
        "prep sidecars should be materialized"
    );
    Ok(())
}

#[test]
fn detects_and_ingests_gz_trace() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let trace_path = dir.path().join("worker0.pt.trace.json.gz");
    let file = fs::File::create(&trace_path)?;
    let mut encoder = GzEncoder::new(file, Compression::default());
    encoder.write_all(trace_json(0, 0).as_bytes())?;
    let _file = encoder.finish()?;

    assert!(detect_path(&trace_path));
    let trace = build_or_load(&trace_path)?;
    assert_eq!(trace.events.len(), 6);
    Ok(())
}

#[test]
fn directory_inputs_are_sorted_trace_sets() -> Result<()> {
    let dir = tempfile::tempdir()?;
    write_trace(&dir.path().join("rank1.pt.trace.json"), 1, 10_000)?;
    write_trace(&dir.path().join("rank0.pt.trace.json"), 0, 0)?;

    assert!(detect_path(dir.path()));
    let trace = build_or_load(dir.path())?;
    assert_eq!(trace.files.len(), 2);
    assert!(trace.is_multi_rank());
    assert!(
        trace
            .collectives
            .iter()
            .any(|group| group.per_rank.len() == 2)
    );
    let first_file = trace
        .files
        .first()
        .ok_or_else(|| anyhow::anyhow!("missing first trace file"))?;
    assert!(
        first_file.path.contains("rank0"),
        "sorted path order should drive stable indexes"
    );
    Ok(())
}
