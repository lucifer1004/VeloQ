use super::{GPU_WORK_EVENTS_VERSION, GpuWorkEventRecord};
use crate::NsysDataResult;
use arrow::array::{Array, ArrayRef, Int32Array, Int64Array, StringArray, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use veloq_core::SourceFingerprint;

pub(super) fn parquet_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("kind", DataType::Utf8, false),
        Field::new("row_id", DataType::Int64, false),
        Field::new("process_id", DataType::Int64, false),
        Field::new("device_id", DataType::Int32, false),
        Field::new("stream_id", DataType::Int64, false),
        Field::new("start_ns", DataType::Int64, false),
        Field::new("end_ns", DataType::Int64, false),
    ]))
}

pub(super) const KV_VERSION: &str = "veloq.gpu_work_events.version";

pub(super) fn write_parquet(
    path: &Path,
    fp: SourceFingerprint,
    records: &[GpuWorkEventRecord],
) -> NsysDataResult<()> {
    let schema = parquet_schema();

    let mut kinds = StringBuilder::new();
    let mut row_ids: Vec<i64> = Vec::with_capacity(records.len());
    let mut processes: Vec<i64> = Vec::with_capacity(records.len());
    let mut devices: Vec<i32> = Vec::with_capacity(records.len());
    let mut streams: Vec<i64> = Vec::with_capacity(records.len());
    let mut starts: Vec<i64> = Vec::with_capacity(records.len());
    let mut ends: Vec<i64> = Vec::with_capacity(records.len());

    for r in records {
        kinds.append_value(&r.kind);
        row_ids.push(r.row_id);
        processes.push(r.process_id);
        devices.push(r.device_id);
        streams.push(r.stream_id);
        starts.push(r.start_ns);
        ends.push(r.end_ns);
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(kinds.finish()),
        Arc::new(Int64Array::from(row_ids)),
        Arc::new(Int64Array::from(processes)),
        Arc::new(Int32Array::from(devices)),
        Arc::new(Int64Array::from(streams)),
        Arc::new(Int64Array::from(starts)),
        Arc::new(Int64Array::from(ends)),
    ];
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns)
        .map_err(crate::NsysDataError::gpu_work_events_record_batch)?;

    let kv = crate::sidecar::freshness_kv(KV_VERSION, GPU_WORK_EVENTS_VERSION, fp);
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_key_value_metadata(Some(kv))
        .build();

    crate::sidecar::atomic_publish(path, |tmp| {
        let file = File::create(tmp).map_err(|source| {
            crate::NsysDataError::gpu_work_events_sidecar_create(tmp.display(), source)
        })?;
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).map_err(|source| {
            crate::NsysDataError::gpu_work_events_writer_open(tmp.display(), source)
        })?;
        writer.write(&batch).map_err(|source| {
            crate::NsysDataError::gpu_work_events_writer_write(tmp.display(), source)
        })?;
        writer.close().map_err(|source| {
            crate::NsysDataError::gpu_work_events_writer_close(tmp.display(), source)
        })?;
        Ok(())
    })
}

pub(super) fn read_parquet(path: &Path) -> NsysDataResult<Vec<GpuWorkEventRecord>> {
    let file = File::open(path).map_err(|source| {
        crate::NsysDataError::gpu_work_events_sidecar_open(path.display(), source)
    })?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|source| {
        crate::NsysDataError::gpu_work_events_reader_open(path.display(), source)
    })?;
    let reader = builder.build().map_err(|source| {
        crate::NsysDataError::gpu_work_events_reader_build(path.display(), source)
    })?;
    let mut out: Vec<GpuWorkEventRecord> = Vec::new();
    for batch in reader {
        let batch = batch.map_err(|source| {
            crate::NsysDataError::gpu_work_events_batch_read(path.display(), source)
        })?;
        let kinds = gpu_work_events_column::<StringArray>(&batch, 0, "kind", "Utf8", path)?;
        let row_ids = gpu_work_events_column::<Int64Array>(&batch, 1, "row_id", "Int64", path)?;
        let processes =
            gpu_work_events_column::<Int64Array>(&batch, 2, "process_id", "Int64", path)?;
        let devices = gpu_work_events_column::<Int32Array>(&batch, 3, "device_id", "Int32", path)?;
        let streams = gpu_work_events_column::<Int64Array>(&batch, 4, "stream_id", "Int64", path)?;
        let starts = gpu_work_events_column::<Int64Array>(&batch, 5, "start_ns", "Int64", path)?;
        let ends = gpu_work_events_column::<Int64Array>(&batch, 6, "end_ns", "Int64", path)?;
        for i in 0..batch.num_rows() {
            out.push(GpuWorkEventRecord {
                kind: kinds.value(i).to_string(),
                row_id: row_ids.value(i),
                process_id: processes.value(i),
                device_id: devices.value(i),
                stream_id: streams.value(i),
                start_ns: starts.value(i),
                end_ns: ends.value(i),
            });
        }
    }
    Ok(out)
}

fn gpu_work_events_column<'a, T: 'static>(
    batch: &'a RecordBatch,
    index: usize,
    column: &str,
    expected: &str,
    path: &Path,
) -> NsysDataResult<&'a T> {
    let array = batch.columns().get(index).ok_or_else(|| {
        crate::NsysDataError::gpu_work_events_column_missing(path.display(), column)
    })?;
    array.as_any().downcast_ref::<T>().ok_or_else(|| {
        crate::NsysDataError::gpu_work_events_column_type_mismatch(
            path.display(),
            column,
            expected,
            format!("{:?}", array.data_type()),
        )
    })
}

pub(super) fn sidecar_is_fresh(path: &Path, fp: SourceFingerprint) -> NsysDataResult<bool> {
    Ok(crate::sidecar::is_fresh(
        path,
        KV_VERSION,
        GPU_WORK_EVENTS_VERSION,
        fp,
        "gpu_work_events",
    )?)
}

pub(crate) fn format_version_on_disk(path: &Path) -> Option<u32> {
    let file = File::open(path).ok()?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).ok()?;
    find_version(
        builder
            .metadata()
            .file_metadata()
            .key_value_metadata()
            .map_or(&[], |kvs| kvs),
    )
}

fn find_version(kvs: &[KeyValue]) -> Option<u32> {
    kvs.iter()
        .find(|kv| kv.key == KV_VERSION)
        .and_then(|kv| kv.value.as_deref())
        .and_then(|value| value.parse().ok())
}
