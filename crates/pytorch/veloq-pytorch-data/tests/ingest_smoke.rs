use arrow::array::{ArrayRef, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use flate2::{Compression, write::GzEncoder};
use parquet::arrow::{ArrowWriter, arrow_reader::ParquetRecordBatchReaderBuilder};
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use veloq_core::VeloqDiagnostic;
use veloq_pytorch_data::{
    EventType, PytorchSidecar, build_or_load, build_or_load_query_trace, detect_path, prep_state,
    sidecar_path_for_artifact, sidecar_states,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

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

fn write_trace(path: &Path, rank: i64, offset_us: i64) -> TestResult {
    fs::write(path, trace_json(rank, offset_us))?;
    Ok(())
}

fn typed_sidecar_path(trace: &veloq_pytorch_data::TraceSet, sidecar: PytorchSidecar) -> PathBuf {
    sidecar_path_for_artifact(&trace.artifact_dir, sidecar)
}

fn query_metadata_path(trace: &veloq_pytorch_data::TraceSet) -> PathBuf {
    PathBuf::from(&trace.artifact_dir).join("query-meta.bin")
}

fn sidecar_ready(input: &Path, sidecar: PytorchSidecar) -> TestResult<bool> {
    sidecar_states(input)
        .into_iter()
        .find(|state| state.name == sidecar.name())
        .map(|state| state.present)
        .ok_or_else(|| {
            std::io::Error::other(format!("missing sidecar state {}", sidecar.name())).into()
        })
}

fn assert_parquet_has_column(path: &Path, column: &'static str) -> TestResult {
    let file = fs::File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    builder.schema().field_with_name(column)?;
    Ok(())
}

fn write_events_sidecar_missing_query_columns(path: &Path) -> TestResult {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new("row_id", DataType::Utf8, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(Vec::<String>::new())),
        Arc::new(StringArray::from(Vec::<String>::new())),
    ];
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns)?;
    let file = fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn count_for(map: &BTreeMap<String, usize>, key: &'static str) -> TestResult<usize> {
    map.get(key)
        .copied()
        .ok_or_else(|| std::io::Error::other(format!("missing count for {key}")).into())
}

#[test]
fn detects_and_ingests_json_trace() -> TestResult {
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
    assert_eq!(trace.schema_survey.raw_event_count, 6);
    assert_eq!(trace.schema_survey.parsed_event_count, 6);
    assert_eq!(trace.schema_survey.flow_marker_count, 0);
    assert_eq!(trace.schema_survey.skipped_event_count, 0);
    assert_eq!(count_for(&trace.schema_survey.phase_counts, "X")?, 6);
    assert_eq!(
        count_for(&trace.schema_survey.category_counts, "kernel")?,
        2
    );
    assert_eq!(
        count_for(&trace.schema_survey.arg_key_counts, "External id")?,
        4
    );
    assert_eq!(trace.schema_survey.typed_arg_coverage.rank, 6);
    assert_eq!(trace.schema_survey.typed_arg_coverage.device_id, 2);
    assert_eq!(trace.schema_survey.typed_arg_coverage.external_id, 4);
    assert_eq!(trace.schema_survey.typed_arg_coverage.correlation_id, 3);
    assert!(
        sidecar_states(&trace_path)
            .iter()
            .all(|sidecar| sidecar.present),
        "prep sidecars should be materialized"
    );
    assert!(prep_state(&trace_path)?.schema_survey.is_some());
    Ok(())
}

#[test]
fn query_trace_cache_does_not_require_full_meta_on_warm_path() -> TestResult {
    let dir = tempfile::tempdir()?;
    let trace_path = dir.path().join("worker0.pt.trace.json");
    write_trace(&trace_path, 0, 0)?;

    let trace = build_or_load(&trace_path)?;
    let query_meta_path = query_metadata_path(&trace);
    assert!(query_meta_path.exists());
    let meta_path = typed_sidecar_path(&trace, PytorchSidecar::Meta);
    fs::remove_file(&meta_path)?;

    let query_trace = build_or_load_query_trace(&trace_path)?;
    assert_eq!(query_trace.artifact_dir, trace.artifact_dir);
    assert_eq!(query_trace.trace_span, trace.trace_span);
    assert_eq!(
        query_trace.capabilities.event_count,
        trace.capabilities.event_count
    );
    assert!(
        !meta_path.exists(),
        "query trace warm path should not rebuild the full event cache"
    );
    Ok(())
}

#[test]
fn query_trace_rebuilds_unreadable_query_sidecar() -> TestResult {
    let dir = tempfile::tempdir()?;
    let trace_path = dir.path().join("worker0.pt.trace.json");
    write_trace(&trace_path, 0, 0)?;

    let trace = build_or_load(&trace_path)?;
    let events_path = typed_sidecar_path(&trace, PytorchSidecar::Events);
    fs::write(&events_path, b"not parquet")?;
    assert!(!sidecar_ready(&trace_path, PytorchSidecar::Events)?);

    let query_trace = build_or_load_query_trace(&trace_path)?;
    assert_eq!(query_trace.artifact_dir, trace.artifact_dir);
    assert_parquet_has_column(&events_path, "is_gpu_activity")?;
    Ok(())
}

#[test]
fn query_trace_rebuilds_query_sidecar_missing_required_columns() -> TestResult {
    let dir = tempfile::tempdir()?;
    let trace_path = dir.path().join("worker0.pt.trace.json");
    write_trace(&trace_path, 0, 0)?;

    let trace = build_or_load(&trace_path)?;
    let events_path = typed_sidecar_path(&trace, PytorchSidecar::Events);
    write_events_sidecar_missing_query_columns(&events_path)?;
    assert!(!sidecar_ready(&trace_path, PytorchSidecar::Events)?);

    let query_trace = build_or_load_query_trace(&trace_path)?;
    assert_eq!(query_trace.artifact_dir, trace.artifact_dir);
    assert_parquet_has_column(&events_path, "is_gpu_activity")?;
    Ok(())
}

#[test]
fn schema_survey_tracks_metadata_flow_and_skipped_events() -> TestResult {
    let dir = tempfile::tempdir()?;
    let trace_path = dir.path().join("survey.pt.trace.json");
    fs::write(
        &trace_path,
        r#"{
  "schemaVersion": "1",
  "deviceProperties": [{ "id": 0, "name": "cuda:0" }],
  "traceEvents": [
    { "name": "process_name", "ph": "M", "pid": 1, "args": { "name": "worker" } },
    { "name": "flow", "ph": "s", "ts": 10, "pid": 1, "tid": 10, "id": 1 },
    { "name": "aten::add", "cat": "cpu_op", "ph": "X", "ts": 20, "dur": 5, "pid": 1, "tid": 10, "args": { "External id": 1 } }
  ]
}"#,
    )?;

    let trace = build_or_load(&trace_path)?;
    let survey = &trace.schema_survey;
    assert_eq!(survey.raw_event_count, 3);
    assert_eq!(survey.parsed_event_count, 1);
    assert_eq!(survey.flow_marker_count, 1);
    assert_eq!(survey.skipped_event_count, 1);
    assert_eq!(count_for(&survey.phase_counts, "M")?, 1);
    assert_eq!(count_for(&survey.phase_counts, "s")?, 1);
    assert_eq!(count_for(&survey.phase_counts, "X")?, 1);
    let file = survey
        .files
        .iter()
        .find(|file| file.trace_index == 0)
        .ok_or_else(|| std::io::Error::other("missing file survey"))?;
    assert!(file.has_device_properties);
    assert!(
        file.top_level_keys
            .iter()
            .any(|key| key == "deviceProperties")
    );
    Ok(())
}

#[test]
fn collectives_sidecar_includes_confidence_column() -> TestResult {
    let dir = tempfile::tempdir()?;
    let trace_path = dir.path().join("worker0.pt.trace.json");
    write_trace(&trace_path, 0, 0)?;

    let trace = build_or_load(&trace_path)?;
    let file = fs::File::open(typed_sidecar_path(&trace, PytorchSidecar::Collectives))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    assert!(
        builder
            .schema()
            .fields()
            .iter()
            .any(|field| field.name() == "confidence")
    );
    for field in [
        "rank_ordinal",
        "row_id",
        "name",
        "rank_start_ns",
        "rank_duration_ns",
    ] {
        builder.schema().field_with_name(field)?;
    }
    assert!(builder.schema().field_with_name("skew_ns")?.is_nullable());
    Ok(())
}

#[test]
fn events_sidecar_includes_query_hot_columns() -> TestResult {
    let dir = tempfile::tempdir()?;
    let trace_path = dir.path().join("worker0.pt.trace.json");
    write_trace(&trace_path, 0, 0)?;

    let trace = build_or_load(&trace_path)?;
    let file = fs::File::open(typed_sidecar_path(&trace, PytorchSidecar::Events))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema();
    for field in [
        "trace_index",
        "original_index",
        "category",
        "phase",
        "pid",
        "tid",
        "comm_kind",
        "bytes",
        "shape",
        "parent_row_id",
        "step_row_id",
        "python_context_row_id",
        "python_context_name",
        "python_context_path",
        "is_gpu_activity",
    ] {
        assert!(
            schema.field_with_name(field).is_ok(),
            "events sidecar should include {field}"
        );
    }
    assert!(schema.field_with_name("category")?.is_nullable());
    assert!(schema.field_with_name("comm_kind")?.is_nullable());
    assert!(schema.field_with_name("bytes")?.is_nullable());
    assert!(
        schema
            .field_with_name("python_context_row_id")?
            .is_nullable()
    );
    assert!(schema.field_with_name("python_context_name")?.is_nullable());
    assert!(schema.field_with_name("python_context_path")?.is_nullable());
    assert!(!schema.field_with_name("trace_index")?.is_nullable());
    assert!(!schema.field_with_name("is_gpu_activity")?.is_nullable());
    Ok(())
}

#[test]
fn python_stack_context_is_derived_from_python_function_events() -> TestResult {
    let dir = tempfile::tempdir()?;
    let trace_path = dir.path().join("with_stack.pt.trace.json");
    fs::write(
        &trace_path,
        r#"{
  "traceEvents": [
    { "name": "<stdin>(1): <module>", "cat": "python_function", "ph": "X", "ts": 0, "dur": 1000, "pid": 1, "tid": 10, "args": { "Python id": 1, "Python parent id": null } },
    { "name": "train.py(20): train_step", "cat": "python_function", "ph": "X", "ts": 50, "dur": 800, "pid": 1, "tid": 10, "args": { "Python id": 2, "Python parent id": 1 } },
    { "name": "model.py(42): forward", "cat": "python_function", "ph": "X", "ts": 80, "dur": 300, "pid": 1, "tid": 10, "args": { "Python id": 3, "Python parent id": 2 } },
    { "name": "aten::matmul", "cat": "cpu_op", "ph": "X", "ts": 100, "dur": 100, "pid": 1, "tid": 10, "args": { "External id": 7 } }
  ]
}"#,
    )?;
    let trace = build_or_load(&trace_path)?;
    assert!(trace.capabilities.has_python_events);
    assert!(trace.capabilities.has_python_stack);
    let cpu = trace
        .events
        .iter()
        .find(|event| event.name == "aten::matmul")
        .ok_or_else(|| std::io::Error::other("missing cpu op"))?;
    assert_eq!(
        cpu.python_context_name.as_deref(),
        Some("model.py(42): forward")
    );
    let path = cpu
        .python_context_path
        .as_deref()
        .ok_or_else(|| std::io::Error::other("missing python context path"))?;
    assert!(path.contains("<stdin>(1): <module>"), "path: {path}");
    assert!(path.contains("train.py(20): train_step"), "path: {path}");
    assert!(path.contains("model.py(42): forward"), "path: {path}");
    Ok(())
}

#[test]
fn active_devices_come_from_gpu_activities() -> TestResult {
    let dir = tempfile::tempdir()?;
    let trace_path = dir.path().join("cpu_only.pt.trace.json");
    fs::write(
        &trace_path,
        r#"{
  "traceEvents": [
    { "name": "aten::cpu_only", "cat": "cpu_op", "ph": "X", "ts": 0, "dur": 10, "pid": 1, "tid": 10, "args": { "device": 7 } }
  ]
}"#,
    )?;

    let trace = build_or_load(&trace_path)?;
    assert!(trace.capabilities.active_devices.is_empty());
    Ok(())
}

#[test]
fn detects_and_ingests_gz_trace() -> TestResult {
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
fn directory_inputs_are_rejected_in_v0() -> TestResult {
    let dir = tempfile::tempdir()?;
    write_trace(&dir.path().join("rank1.pt.trace.json"), 1, 10_000)?;
    write_trace(&dir.path().join("rank0.pt.trace.json"), 0, 0)?;

    assert!(!detect_path(dir.path()));
    let Some(err) = build_or_load(dir.path()).err() else {
        return Err(std::io::Error::other("expected directory rejection").into());
    };
    assert!(err.to_string().contains("directory inputs"));
    assert_eq!(err.code().as_str(), "pytorch.input.directory-unsupported");
    Ok(())
}

#[test]
fn invalid_utf8_reports_pytorch_trace_json_context() -> TestResult {
    let dir = tempfile::tempdir()?;
    let trace_path = dir.path().join("bad.pt.trace.json");
    fs::write(&trace_path, [0xff])?;

    let Some(err) = build_or_load(&trace_path).err() else {
        return Err(std::io::Error::other("expected invalid utf-8 error").into());
    };
    assert!(err.to_string().contains("pytorch trace JSON"), "got: {err}");
    assert_eq!(err.code().as_str(), "encoding.utf8");
    Ok(())
}
