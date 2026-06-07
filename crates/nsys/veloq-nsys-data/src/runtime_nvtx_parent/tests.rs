use super::compute::{
    NvtxRangeRow, RuntimeRow, collect_nvtx_by_tid, collect_runtime_rows, merge_dev_ctx, walk,
};
use super::gpu_activity::{
    DevCtxMap, DevCtxValue, parquet_integer_i64, read_ctx_for_pid, read_gpu_dev_ctx_parquet,
};
use super::parquet::{parquet_schema, read_parquet, sidecar_is_fresh, write_parquet};
use super::{EnclosingNvtx, RuntimeNvtxParent, RuntimeParentEntry, source_fingerprint};
use crate::test_support::{parquet_fixture_with_rows, write_test_parquet};
use ::parquet::arrow::ArrowWriter;
use anyhow::Result;
use arrow::array::{
    Int32Array, Int64Array, Int64Builder, ListBuilder, StringArray, StringBuilder, UInt32Array,
    UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use veloq_core::{SourceFingerprint, VeloqDiagnostic};

fn n(rowid: i64, start: i64, end: i64, name: &str) -> NvtxRangeRow {
    NvtxRangeRow {
        rowid,
        start,
        end,
        name: name.to_string(),
    }
}

fn rt(rowid: i64, corr: Option<i64>, pid: i64, tid: i64, start: i64, end: i64) -> RuntimeRow {
    RuntimeRow {
        rowid,
        correlation_id: corr,
        native_pid: pid,
        global_tid: tid,
        start,
        end,
        device_id: None,
        context_id: None,
    }
}

#[test]
fn source_fingerprint_missing_trace_error_is_typed() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("missing_pqtdir");

    let err = match source_fingerprint(&path) {
        Ok(fp) => anyhow::bail!("missing trace should not fingerprint: {fp:?}"),
        Err(err) => err,
    };

    assert_eq!(
        err.code().as_str(),
        "nsys.data.nvtx-parent-trace-fingerprint"
    );
    match err {
        crate::NsysDataError::NvtxParentTraceFingerprint { path, .. } => {
            assert!(path.contains("missing_pqtdir"));
        }
        other => anyhow::bail!("expected NvtxParentTraceFingerprint, got {other:?}"),
    }
    Ok(())
}

#[test]
fn read_parquet_missing_file_error_is_typed() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("missing.parquet");

    let err = match read_parquet(&path) {
        Ok(rows) => anyhow::bail!("missing nvtx-parent sidecar should not load: {rows:?}"),
        Err(err) => err,
    };

    assert_eq!(err.code().as_str(), "nsys.data.nvtx-parent-sidecar-open");
    match err {
        crate::NsysDataError::NvtxParentSidecarOpen { path, .. } => {
            assert!(path.contains("missing.parquet"));
        }
        other => anyhow::bail!("expected NvtxParentSidecarOpen, got {other:?}"),
    }
    Ok(())
}

#[test]
fn read_parquet_invalid_file_error_is_typed() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("bad.parquet");
    std::fs::write(&path, b"not a parquet file")?;

    let err = match read_parquet(&path) {
        Ok(rows) => anyhow::bail!("invalid nvtx-parent sidecar should not load: {rows:?}"),
        Err(err) => err,
    };

    assert_eq!(err.code().as_str(), "nsys.data.nvtx-parent-reader-open");
    match err {
        crate::NsysDataError::NvtxParentReaderOpen { path, .. } => {
            assert!(path.contains("bad.parquet"));
        }
        other => anyhow::bail!("expected NvtxParentReaderOpen, got {other:?}"),
    }
    Ok(())
}

fn assert_nvtx_parent_rows_error(
    err: crate::NsysDataError,
    expected_code: &str,
    expected_table: &str,
) -> Result<()> {
    assert_eq!(err.code().as_str(), expected_code);
    let Some((area, _, label)) = err.duckdb_parts() else {
        anyhow::bail!("expected nvtx-parent rows DuckDB error, got {err:?}");
    };
    assert_eq!(area, "nvtx-parent rows");
    assert_eq!(label, expected_table);
    Ok(())
}

#[test]
fn ctx_for_pid_missing_column_error_is_typed_prepare() -> Result<()> {
    let (_dir, pqtdir) = parquet_fixture_with_rows(&[
        (
            "CUPTI_ACTIVITY_KIND_KERNEL",
            r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start BIGINT, "end" BIGINT)"#,
            Vec::new(),
        ),
        (
            "TARGET_INFO_CUDA_CONTEXT_INFO",
            "CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO (deviceId BIGINT)",
            Vec::new(),
        ),
    ])?;
    let trace = crate::Trace::open(&pqtdir)?;

    let err = match read_ctx_for_pid(&trace) {
        Ok(rows) => anyhow::bail!("missing context columns should fail: {rows:?}"),
        Err(err) => err,
    };

    assert_nvtx_parent_rows_error(
        err,
        "nsys.data.duckdb-prepare",
        "TARGET_INFO_CUDA_CONTEXT_INFO",
    )
}

#[test]
fn ctx_for_pid_bad_device_id_error_is_typed_query() -> Result<()> {
    let (_dir, pqtdir) = parquet_fixture_with_rows(&[
        (
            "CUPTI_ACTIVITY_KIND_KERNEL",
            r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start BIGINT, "end" BIGINT)"#,
            Vec::new(),
        ),
        (
            "TARGET_INFO_CUDA_CONTEXT_INFO",
            "CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO (deviceId TEXT, contextId BIGINT, processId BIGINT)",
            vec![
                "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO (deviceId, contextId, processId) VALUES ('bad', 7, 42)",
            ],
        ),
    ])?;
    let trace = crate::Trace::open(&pqtdir)?;

    let err = match read_ctx_for_pid(&trace) {
        Ok(rows) => anyhow::bail!("bad deviceId cast should fail: {rows:?}"),
        Err(err) => err,
    };

    assert_nvtx_parent_rows_error(
        err,
        "nsys.data.duckdb-query",
        "TARGET_INFO_CUDA_CONTEXT_INFO",
    )
}

#[test]
fn nvtx_rows_bad_start_error_is_typed_read() -> Result<()> {
    let (_dir, pqtdir) = parquet_fixture_with_rows(&[
        (
            "CUPTI_ACTIVITY_KIND_KERNEL",
            r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start BIGINT, "end" BIGINT)"#,
            Vec::new(),
        ),
        (
            "NVTX_EVENTS",
            r#"CREATE TABLE NVTX_EVENTS (start TEXT, "end" BIGINT, globalTid BIGINT, text TEXT, textId BIGINT)"#,
            vec![
                r#"INSERT INTO NVTX_EVENTS (start, "end", globalTid, text, textId) VALUES ('bad', 10, 7, 'outer', NULL)"#,
            ],
        ),
        (
            "StringIds",
            "CREATE TABLE StringIds (id BIGINT, value TEXT)",
            Vec::new(),
        ),
    ])?;
    let trace = crate::Trace::open(&pqtdir)?;

    let err = match collect_nvtx_by_tid(&trace) {
        Ok(rows) => anyhow::bail!("bad NVTX start should fail: {rows:?}"),
        Err(err) => err,
    };

    assert_nvtx_parent_rows_error(err, "nsys.data.duckdb-read", "NVTX_EVENTS")
}

#[test]
fn runtime_rows_bad_start_error_is_typed_read() -> Result<()> {
    let (_dir, pqtdir) = parquet_fixture_with_rows(&[
        (
            "CUPTI_ACTIVITY_KIND_KERNEL",
            r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start BIGINT, "end" BIGINT)"#,
            Vec::new(),
        ),
        (
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            r#"CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (correlationId BIGINT, globalTid BIGINT, start TEXT, "end" BIGINT)"#,
            vec![
                r#"INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME (correlationId, globalTid, start, "end") VALUES (99, 7, 'bad', 10)"#,
            ],
        ),
    ])?;
    let trace = crate::Trace::open(&pqtdir)?;

    let err = match collect_runtime_rows(&trace) {
        Ok(rows) => anyhow::bail!("bad runtime start should fail: {rows:?}"),
        Err(err) => err,
    };

    assert_nvtx_parent_rows_error(err, "nsys.data.duckdb-read", "CUPTI_ACTIVITY_KIND_RUNTIME")
}

#[test]
fn walk_collects_outer_to_inner_for_nested_ranges() -> Result<()> {
    let mut by_tid = HashMap::new();
    by_tid.insert(7, vec![n(1, 0, 100, "outer"), n(2, 40, 60, "inner")]);
    let out = walk(&by_tid, &[rt(10, Some(999), 42, 7, 45, 55)]);
    let first = out
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected one entry"))?;
    assert_eq!(first.enclosing.len(), 2);
    // Outer first.
    let outer = first
        .enclosing
        .first()
        .ok_or_else(|| anyhow::anyhow!("missing outer"))?;
    assert_eq!(outer.nvtx_name, "outer");
    assert_eq!(outer.nvtx_rowid, 1);
    // Innermost = last.
    let innermost = first
        .innermost()
        .ok_or_else(|| anyhow::anyhow!("missing inner"))?;
    assert_eq!(innermost.nvtx_name, "inner");
    assert_eq!(innermost.nvtx_rowid, 2);
    Ok(())
}

/// P2 review guard: when two enclosing NVTX ranges share the
/// same start, the one with the larger end is OUTER and must
/// land earlier in the chain so `.last()` (innermost) is the
/// tighter range.
#[test]
fn walk_orders_same_start_by_end_desc() -> Result<()> {
    let mut by_tid = HashMap::new();
    // Both start at 0; outer ends at 100, inner ends at 60.
    by_tid.insert(7, vec![n(1, 0, 100, "outer"), n(2, 0, 60, "inner")]);
    let out = walk(&by_tid, &[rt(10, Some(999), 42, 7, 30, 50)]);
    let first = out
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected one entry"))?;
    assert_eq!(first.enclosing.len(), 2);
    let outer = first
        .enclosing
        .first()
        .ok_or_else(|| anyhow::anyhow!("missing outer"))?;
    assert_eq!(outer.nvtx_name, "outer", "outer must come first");
    let innermost = first
        .innermost()
        .ok_or_else(|| anyhow::anyhow!("missing innermost"))?;
    assert_eq!(
        innermost.nvtx_name, "inner",
        "innermost must be tighter range"
    );
    Ok(())
}

#[test]
fn walk_skips_partial_overlap() {
    let mut by_tid = HashMap::new();
    by_tid.insert(7, vec![n(1, 0, 100, "outer")]);
    // Runtime exits past the NVTX end — not fully contained.
    let out = walk(&by_tid, &[rt(10, Some(999), 42, 7, 50, 150)]);
    assert!(out.is_empty());
}

#[test]
fn any_enclosing_name_matches_outer_when_innermost_does_not() -> Result<()> {
    let mut by_tid = HashMap::new();
    by_tid.insert(
        7,
        vec![n(1, 0, 100, "training_step"), n(2, 40, 60, "fwd_pass")],
    );
    let out = walk(&by_tid, &[rt(10, Some(999), 42, 7, 45, 55)]);
    let first = out
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected one entry"))?;
    // The semantic that v1 sidecar couldn't preserve: a pattern
    // matching the OUTER range must still attribute the contained
    // event, even though the innermost is something else.
    assert!(first.any_enclosing_name(|n| n.starts_with("training")));
    assert!(first.any_enclosing_name(|n| n == "fwd_pass"));
    assert!(!first.any_enclosing_name(|n| n.starts_with("eval")));
    Ok(())
}

#[test]
fn parquet_roundtrip_preserves_records() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("rtnvtx.parquet");
    let records = vec![
        RuntimeParentEntry {
            rt_rowid: 1,
            correlation_id: Some(100),
            native_pid: 42,
            device_id: Some(0),
            context_id: Some(1),
            enclosing: vec![
                EnclosingNvtx {
                    nvtx_rowid: 11,
                    nvtx_name: "iter_42".to_string(),
                },
                EnclosingNvtx {
                    nvtx_rowid: 12,
                    nvtx_name: "step_a".to_string(),
                },
            ],
        },
        RuntimeParentEntry {
            rt_rowid: 2,
            correlation_id: Some(101),
            native_pid: 42,
            device_id: Some(0),
            context_id: Some(1),
            enclosing: vec![EnclosingNvtx {
                nvtx_rowid: 11,
                nvtx_name: "iter_42".to_string(),
            }],
        },
        // Runtime call without a CUDA correlation (e.g.
        // cudaGetDeviceCount). Must round-trip cleanly and only
        // surface in `by_rt_rowid`, never in `by_correlation`.
        RuntimeParentEntry {
            rt_rowid: 3,
            correlation_id: None,
            native_pid: 99,
            device_id: None,
            context_id: None,
            enclosing: vec![EnclosingNvtx {
                nvtx_rowid: 21,
                nvtx_name: "step_b".to_string(),
            }],
        },
    ];
    let fp = SourceFingerprint {
        mtime_secs: 1234567890,
        size: 4096,
    };
    write_parquet(&path, fp, &records)?;
    assert!(sidecar_is_fresh(&path, fp)?);
    let bumped = SourceFingerprint {
        mtime_secs: 1234567891,
        size: 4096,
    };
    assert!(
        !sidecar_is_fresh(&path, bumped)?,
        "mtime-change must invalidate"
    );
    let loaded = read_parquet(&path)?;
    assert_eq!(loaded, records);
    Ok(())
}

#[test]
fn gpu_dev_ctx_reader_accepts_unsigned_nsys_integer_columns() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("gpu.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("correlationId", DataType::UInt32, true),
        Field::new("deviceId", DataType::UInt32, true),
        Field::new("contextId", DataType::UInt64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(UInt32Array::from(vec![Some(77_u32), Some(88_u32)])),
            Arc::new(UInt32Array::from(vec![Some(0_u32), Some(1_u32)])),
            Arc::new(UInt64Array::from(vec![Some(123_u64), Some(999_u64)])),
        ],
    )?;
    {
        let file = File::create(&path)?;
        let mut writer = ArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;
    }

    let mut ctx_for_pid = HashMap::new();
    ctx_for_pid.insert((0, 123), 4242);
    let out = read_gpu_dev_ctx_parquet(&path, &ctx_for_pid)?;
    assert_eq!(out.get(&(4242, 77)), Some(&DevCtxValue::Single((0, 123))));
    assert!(
        !out.contains_key(&(4242, 88)),
        "unmapped unsigned context should not produce an entry"
    );
    Ok(())
}

#[test]
fn gpu_dev_ctx_reader_missing_file_error_is_typed() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("missing-gpu.parquet");

    let err = match read_gpu_dev_ctx_parquet(&path, &HashMap::new()) {
        Ok(rows) => anyhow::bail!("missing GPU activity parquet should not load: {rows:?}"),
        Err(err) => err,
    };

    assert_eq!(
        err.code().as_str(),
        "nsys.data.nvtx-parent-gpu-activity-open"
    );
    match err {
        crate::NsysDataError::NvtxParentGpuActivityOpen { path, .. } => {
            assert!(path.contains("missing-gpu.parquet"));
        }
        other => anyhow::bail!("expected NvtxParentGpuActivityOpen, got {other:?}"),
    }
    Ok(())
}

#[test]
fn gpu_dev_ctx_reader_invalid_file_error_is_typed() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("bad-gpu.parquet");
    std::fs::write(&path, b"not a parquet file")?;

    let err = match read_gpu_dev_ctx_parquet(&path, &HashMap::new()) {
        Ok(rows) => anyhow::bail!("invalid GPU activity parquet should not load: {rows:?}"),
        Err(err) => err,
    };

    assert_eq!(
        err.code().as_str(),
        "nsys.data.nvtx-parent-gpu-activity-reader-open"
    );
    match err {
        crate::NsysDataError::NvtxParentGpuActivityReaderOpen { path, .. } => {
            assert!(path.contains("bad-gpu.parquet"));
        }
        other => anyhow::bail!("expected NvtxParentGpuActivityReaderOpen, got {other:?}"),
    }
    Ok(())
}

#[test]
fn gpu_dev_ctx_reader_rejects_missing_column_with_typed_error() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("gpu.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("correlationId", DataType::Int64, true),
        Field::new("deviceId", DataType::Int64, true),
    ]));
    write_test_parquet(
        &path,
        schema,
        vec![
            Arc::new(Int64Array::from(vec![Some(77)])),
            Arc::new(Int64Array::from(vec![Some(0)])),
        ],
    )?;

    let err = match read_gpu_dev_ctx_parquet(&path, &HashMap::new()) {
        Ok(rows) => anyhow::bail!("missing-column GPU activity parquet should fail: {rows:?}"),
        Err(err) => err,
    };

    assert_eq!(
        err.code().as_str(),
        "nsys.data.nvtx-parent-gpu-activity-column-missing"
    );
    match err {
        crate::NsysDataError::NvtxParentGpuActivityColumnMissing { column, .. } => {
            assert_eq!(column, "contextId");
        }
        other => anyhow::bail!("expected NvtxParentGpuActivityColumnMissing, got {other:?}"),
    }
    Ok(())
}

#[test]
fn gpu_dev_ctx_reader_rejects_device_id_overflow_with_typed_error() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("gpu.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("correlationId", DataType::Int64, true),
        Field::new("deviceId", DataType::Int64, true),
        Field::new("contextId", DataType::Int64, true),
    ]));
    let overflowing_device_id = i64::from(i32::MAX) + 1;
    write_test_parquet(
        &path,
        schema,
        vec![
            Arc::new(Int64Array::from(vec![Some(77)])),
            Arc::new(Int64Array::from(vec![Some(overflowing_device_id)])),
            Arc::new(Int64Array::from(vec![Some(123)])),
        ],
    )?;

    let err = match read_gpu_dev_ctx_parquet(&path, &HashMap::new()) {
        Ok(rows) => anyhow::bail!("overflowing deviceId should fail: {rows:?}"),
        Err(err) => err,
    };

    assert_eq!(err.code().as_str(), "nsys.data.nvtx-parent-int32-overflow");
    match err {
        crate::NsysDataError::NvtxParentInt32Overflow { column, value, .. } => {
            assert_eq!(column, "deviceId");
            assert_eq!(value, overflowing_device_id);
        }
        other => anyhow::bail!("expected NvtxParentInt32Overflow, got {other:?}"),
    }
    Ok(())
}

#[test]
fn optional_i64_rejects_non_integer_array_with_typed_error() -> Result<()> {
    let array = StringArray::from(vec![Some("not-an-integer")]);
    let err = match parquet_integer_i64(&array, 0, "contextId", Path::new("gpu.parquet")) {
        Ok(value) => anyhow::bail!("StringArray should not read as integer: {value:?}"),
        Err(err) => err,
    };

    assert_eq!(
        err.code().as_str(),
        "nsys.data.nvtx-parent-unsupported-integer-type"
    );
    match err {
        crate::NsysDataError::NvtxParentUnsupportedIntegerType {
            column, data_type, ..
        } => {
            assert_eq!(column, "contextId");
            assert!(data_type.contains("Utf8"));
        }
        other => anyhow::bail!("expected NvtxParentUnsupportedIntegerType, got {other:?}"),
    }
    Ok(())
}

#[test]
fn optional_i64_rejects_uint64_overflow_with_typed_error() -> Result<()> {
    let array = UInt64Array::from(vec![Some(u64::MAX)]);
    let err = match parquet_integer_i64(&array, 0, "contextId", Path::new("gpu.parquet")) {
        Ok(value) => anyhow::bail!("overflowing UInt64 should not read as integer: {value:?}"),
        Err(err) => err,
    };

    assert_eq!(
        err.code().as_str(),
        "nsys.data.nvtx-parent-integer-overflow"
    );
    match err {
        crate::NsysDataError::NvtxParentIntegerOverflow { column, value, .. } => {
            assert_eq!(column, "contextId");
            assert_eq!(value, u64::MAX);
        }
        other => anyhow::bail!("expected NvtxParentIntegerOverflow, got {other:?}"),
    }
    Ok(())
}

#[test]
fn read_parquet_rejects_mismatched_nvtx_lists_with_typed_error() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("nvtx-parent.parquet");
    let schema = parquet_schema();
    let mut rowids_builder: ListBuilder<Int64Builder> = ListBuilder::new(Int64Builder::new());
    rowids_builder.values().append_value(11);
    rowids_builder.values().append_value(22);
    rowids_builder.append(true);
    let mut names_builder: ListBuilder<StringBuilder> = ListBuilder::new(StringBuilder::new());
    names_builder.values().append_value("outer");
    names_builder.append(true);
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1])),
            Arc::new(Int64Array::from(vec![Some(100)])),
            Arc::new(Int64Array::from(vec![99])),
            Arc::new(Int32Array::from(vec![Some(0)])),
            Arc::new(Int64Array::from(vec![Some(1)])),
            Arc::new(rowids_builder.finish()),
            Arc::new(names_builder.finish()),
        ],
    )?;
    {
        let file = File::create(&path)?;
        let mut writer = ArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;
    }

    let err = match read_parquet(&path) {
        Ok(rows) => anyhow::bail!("mismatched list sidecar should not load: {rows:?}"),
        Err(err) => err,
    };

    assert_eq!(
        err.code().as_str(),
        "nsys.data.nvtx-parent-list-length-mismatch"
    );
    assert!(matches!(
        err,
        crate::NsysDataError::NvtxParentListLengthMismatch {
            row: 0,
            rowids_len: 2,
            names_len: 1
        }
    ));
    Ok(())
}

#[test]
fn read_parquet_rejects_wrong_column_type_with_typed_error() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("nvtx-parent.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("rt_rowid", DataType::Utf8, false),
        Field::new("correlation_id", DataType::Int64, true),
        Field::new("native_pid", DataType::Int64, false),
        Field::new("device_id", DataType::Int32, true),
        Field::new("context_id", DataType::Int64, true),
        Field::new(
            "nvtx_rowids",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            false,
        ),
        Field::new(
            "nvtx_names",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            false,
        ),
    ]));
    let mut rowids_builder: ListBuilder<Int64Builder> = ListBuilder::new(Int64Builder::new());
    rowids_builder.values().append_value(11);
    rowids_builder.append(true);
    let mut names_builder: ListBuilder<StringBuilder> = ListBuilder::new(StringBuilder::new());
    names_builder.values().append_value("outer");
    names_builder.append(true);
    write_test_parquet(
        &path,
        schema,
        vec![
            Arc::new(StringArray::from(vec!["not-int64"])),
            Arc::new(Int64Array::from(vec![Some(100)])),
            Arc::new(Int64Array::from(vec![99])),
            Arc::new(Int32Array::from(vec![Some(0)])),
            Arc::new(Int64Array::from(vec![Some(1)])),
            Arc::new(rowids_builder.finish()),
            Arc::new(names_builder.finish()),
        ],
    )?;

    let err = match read_parquet(&path) {
        Ok(rows) => anyhow::bail!("wrong-typed nvtx-parent sidecar should not load: {rows:?}"),
        Err(err) => err,
    };

    assert_eq!(
        err.code().as_str(),
        "nsys.data.nvtx-parent-column-type-mismatch"
    );
    match err {
        crate::NsysDataError::NvtxParentColumnTypeMismatch {
            column,
            expected,
            actual,
            ..
        } => {
            assert_eq!(column, "rt_rowid");
            assert_eq!(expected, "Int64");
            assert!(actual.contains("Utf8"));
        }
        other => anyhow::bail!("expected NvtxParentColumnTypeMismatch, got {other:?}"),
    }
    Ok(())
}

#[test]
fn read_parquet_rejects_wrong_list_inner_type_with_typed_error() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("nvtx-parent.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("rt_rowid", DataType::Int64, false),
        Field::new("correlation_id", DataType::Int64, true),
        Field::new("native_pid", DataType::Int64, false),
        Field::new("device_id", DataType::Int32, true),
        Field::new("context_id", DataType::Int64, true),
        Field::new(
            "nvtx_rowids",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            false,
        ),
        Field::new(
            "nvtx_names",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            false,
        ),
    ]));
    let mut rowids_builder: ListBuilder<StringBuilder> = ListBuilder::new(StringBuilder::new());
    rowids_builder.values().append_value("not-int64");
    rowids_builder.append(true);
    let mut names_builder: ListBuilder<StringBuilder> = ListBuilder::new(StringBuilder::new());
    names_builder.values().append_value("outer");
    names_builder.append(true);
    write_test_parquet(
        &path,
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1])),
            Arc::new(Int64Array::from(vec![Some(100)])),
            Arc::new(Int64Array::from(vec![99])),
            Arc::new(Int32Array::from(vec![Some(0)])),
            Arc::new(Int64Array::from(vec![Some(1)])),
            Arc::new(rowids_builder.finish()),
            Arc::new(names_builder.finish()),
        ],
    )?;

    let err = match read_parquet(&path) {
        Ok(rows) => anyhow::bail!("wrong-inner nvtx-parent sidecar should not load: {rows:?}"),
        Err(err) => err,
    };

    assert_eq!(
        err.code().as_str(),
        "nsys.data.nvtx-parent-column-type-mismatch"
    );
    match err {
        crate::NsysDataError::NvtxParentColumnTypeMismatch {
            column,
            expected,
            actual,
            ..
        } => {
            assert_eq!(column, "nvtx_rowids");
            assert_eq!(expected, "List<Int64>");
            assert!(actual.contains("Utf8"));
        }
        other => anyhow::bail!("expected NvtxParentColumnTypeMismatch, got {other:?}"),
    }
    Ok(())
}

#[test]
fn read_parquet_rejects_missing_column_with_typed_error() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("nvtx-parent.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("rt_rowid", DataType::Int64, false),
        Field::new("correlation_id", DataType::Int64, true),
        Field::new("native_pid", DataType::Int64, false),
        Field::new("device_id", DataType::Int32, true),
        Field::new("context_id", DataType::Int64, true),
        Field::new(
            "nvtx_rowids",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            false,
        ),
    ]));
    let mut rowids_builder: ListBuilder<Int64Builder> = ListBuilder::new(Int64Builder::new());
    rowids_builder.values().append_value(11);
    rowids_builder.append(true);
    write_test_parquet(
        &path,
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1])),
            Arc::new(Int64Array::from(vec![Some(100)])),
            Arc::new(Int64Array::from(vec![99])),
            Arc::new(Int32Array::from(vec![Some(0)])),
            Arc::new(Int64Array::from(vec![Some(1)])),
            Arc::new(rowids_builder.finish()),
        ],
    )?;

    let err = match read_parquet(&path) {
        Ok(rows) => anyhow::bail!("truncated nvtx-parent sidecar should not load: {rows:?}"),
        Err(err) => err,
    };

    assert_eq!(err.code().as_str(), "nsys.data.nvtx-parent-column-missing");
    match err {
        crate::NsysDataError::NvtxParentColumnMissing { column, .. } => {
            assert_eq!(column, "nvtx_names");
        }
        other => anyhow::bail!("expected NvtxParentColumnMissing, got {other:?}"),
    }
    Ok(())
}

/// Schema invariant: runtime rows with `correlation_id = None` still
/// show up under `by_rt_rowid` (so `--type runtime --group-by
/// nvtx-parent` attributes them) but never under `by_correlation`
/// (the GPU-side bridge only exists when there's a correlation
/// to bridge through).
#[test]
fn none_correlation_is_absent_from_by_correlation_map() -> Result<()> {
    let records = vec![
        RuntimeParentEntry {
            rt_rowid: 1,
            correlation_id: Some(100),
            native_pid: 42,
            device_id: Some(0),
            context_id: Some(1),
            enclosing: vec![EnclosingNvtx {
                nvtx_rowid: 11,
                nvtx_name: "step".to_string(),
            }],
        },
        RuntimeParentEntry {
            rt_rowid: 2,
            correlation_id: None,
            native_pid: 42,
            device_id: None,
            context_id: None,
            enclosing: vec![EnclosingNvtx {
                nvtx_rowid: 11,
                nvtx_name: "step".to_string(),
            }],
        },
    ];
    let idx = RuntimeNvtxParent::from_records(records);
    // Both rows attributed.
    assert!(idx.get_by_runtime(1).is_some());
    assert!(idx.get_by_runtime(2).is_some());
    // Correlation lookup works for the correlated row via the
    // full disambiguator trio (device, context, correlation).
    assert!(idx.get_by_correlation(0, 1, 100).is_some());
    // The None-correlation row leaves no ghost entry.
    assert!(idx.get_by_correlation(0, 1, 0).is_none());
    Ok(())
}

/// `merge_dev_ctx` fan-out: a single attributed runtime row with
/// an ambiguous `(native_pid, correlationId) → (device, context)`
/// mapping must produce one sidecar entry per candidate `(D, X)`,
/// preserving its enclosing chain across all copies.
#[test]
fn merge_dev_ctx_fans_out_on_ambiguous_correlation() -> Result<()> {
    let walked = vec![RuntimeParentEntry {
        rt_rowid: 1,
        correlation_id: Some(42),
        native_pid: 1000,
        device_id: None,
        context_id: None,
        enclosing: vec![EnclosingNvtx {
            nvtx_rowid: 11,
            nvtx_name: "step".to_string(),
        }],
    }];
    let mut dev_ctx: DevCtxMap = HashMap::new();
    // Two contexts in the same process both claim correlationId=42.
    dev_ctx.insert((1000, 42), DevCtxValue::Many(vec![(0, 1), (0, 2)]));

    let out = merge_dev_ctx(walked, &dev_ctx);
    assert_eq!(out.len(), 2, "ambiguous (D,X) must fan out");
    let dx_pairs: std::collections::BTreeSet<(i32, i64)> = out
        .iter()
        .map(|e| (e.device_id.unwrap_or(-1), e.context_id.unwrap_or(-1)))
        .collect();
    assert!(dx_pairs.contains(&(0, 1)));
    assert!(dx_pairs.contains(&(0, 2)));
    // Same rt_rowid, same enclosing on every copy.
    assert!(out.iter().all(|e| e.rt_rowid == 1));
    assert!(out.iter().all(|e| {
        e.enclosing.len() == 1 && e.enclosing.first().map(|n| n.nvtx_name.as_str()) == Some("step")
    }));
    Ok(())
}

/// Common case: a single (D, X) candidate mutates the entry in
/// place — no clone, no fanout.
#[test]
fn merge_dev_ctx_single_candidate_mutates_in_place() -> Result<()> {
    let walked = vec![RuntimeParentEntry {
        rt_rowid: 7,
        correlation_id: Some(100),
        native_pid: 1000,
        device_id: None,
        context_id: None,
        enclosing: vec![EnclosingNvtx {
            nvtx_rowid: 11,
            nvtx_name: "step".to_string(),
        }],
    }];
    let mut dev_ctx: DevCtxMap = HashMap::new();
    dev_ctx.insert((1000, 100), DevCtxValue::Single((0, 1)));
    let out = merge_dev_ctx(walked, &dev_ctx);
    assert_eq!(out.len(), 1);
    let first = out
        .first()
        .ok_or_else(|| anyhow::anyhow!("missing entry"))?;
    assert_eq!(first.device_id, Some(0));
    assert_eq!(first.context_id, Some(1));
    Ok(())
}

/// Schema invariant: multi-context within a single process with the
/// same `correlationId` reused across contexts disambiguates
/// through the `(device, context, correlation)` key.
#[test]
fn multi_context_same_correlation_disambiguates_by_device_context() -> Result<()> {
    let records = vec![
        RuntimeParentEntry {
            rt_rowid: 1,
            correlation_id: Some(42),
            native_pid: 1000,
            device_id: Some(0),
            context_id: Some(1),
            enclosing: vec![EnclosingNvtx {
                nvtx_rowid: 11,
                nvtx_name: "ctx1_scope".to_string(),
            }],
        },
        RuntimeParentEntry {
            rt_rowid: 2,
            correlation_id: Some(42),
            native_pid: 1000,
            device_id: Some(0),
            context_id: Some(2),
            enclosing: vec![EnclosingNvtx {
                nvtx_rowid: 22,
                nvtx_name: "ctx2_scope".to_string(),
            }],
        },
    ];
    let idx = RuntimeNvtxParent::from_records(records);
    let e1 = idx
        .get_by_correlation(0, 1, 42)
        .ok_or_else(|| anyhow::anyhow!("missing (0,1,42)"))?;
    let e2 = idx
        .get_by_correlation(0, 2, 42)
        .ok_or_else(|| anyhow::anyhow!("missing (0,2,42)"))?;
    assert_eq!(
        e1.innermost().map(|e| e.nvtx_name.as_str()),
        Some("ctx1_scope")
    );
    assert_eq!(
        e2.innermost().map(|e| e.nvtx_name.as_str()),
        Some("ctx2_scope")
    );
    Ok(())
}
