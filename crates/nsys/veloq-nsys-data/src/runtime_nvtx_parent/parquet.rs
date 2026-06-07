use super::{EnclosingNvtx, RUNTIME_NVTX_PARENT_VERSION, RuntimeParentEntry};
use crate::NsysDataResult;
use arrow::array::{
    Array, ArrayRef, Int32Array, Int64Array, Int64Builder, ListArray, ListBuilder, StringArray,
    StringBuilder,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use veloq_core::SourceFingerprint;

// ----- parquet I/O ---------------------------------------------------------

pub(super) fn parquet_schema() -> SchemaRef {
    // Arrow's default ListBuilder inner field is `Field::new("item",
    // …, true)` (nullable). The schema must match — we never write
    // nulls in practice, but the logical type carries the nullable
    // bit and a mismatch fails `RecordBatch::try_new`.
    let rowids_field = Arc::new(Field::new("item", DataType::Int64, true));
    let names_field = Arc::new(Field::new("item", DataType::Utf8, true));
    Arc::new(Schema::new(vec![
        Field::new("rt_rowid", DataType::Int64, false),
        // Nullable: runtime calls without a CUDA correlation
        // (e.g. `cudaGetDeviceCount`) write NULL here.
        Field::new("correlation_id", DataType::Int64, true),
        Field::new("native_pid", DataType::Int64, false),
        // Nullable: the runtime call's resolved GPU (device, context)
        // — both NULL when the call has no corresponding GPU
        // activity, when `TARGET_INFO_CUDA_CONTEXT_INFO` was absent
        // during build, or when no GPU activity table is present.
        Field::new("device_id", DataType::Int32, true),
        Field::new("context_id", DataType::Int64, true),
        Field::new("nvtx_rowids", DataType::List(rowids_field), false),
        Field::new("nvtx_names", DataType::List(names_field), false),
    ]))
}

const KV_VERSION: &str = "veloq.runtime_nvtx_parent.version";

pub(super) fn write_parquet(
    path: &Path,
    fp: SourceFingerprint,
    records: &[RuntimeParentEntry],
) -> NsysDataResult<()> {
    let schema = parquet_schema();
    let mut rt_rowids: Vec<i64> = Vec::with_capacity(records.len());
    let mut corrs: Vec<Option<i64>> = Vec::with_capacity(records.len());
    let mut pids: Vec<i64> = Vec::with_capacity(records.len());
    let mut devs: Vec<Option<i32>> = Vec::with_capacity(records.len());
    let mut ctxs: Vec<Option<i64>> = Vec::with_capacity(records.len());

    // ListBuilders for the two list columns. The inner builder
    // accumulates items per row; calling `.append(true)` closes the
    // current row's list and starts a new one.
    let mut rowids_builder: ListBuilder<Int64Builder> = ListBuilder::new(Int64Builder::new());
    let mut names_builder: ListBuilder<StringBuilder> = ListBuilder::new(StringBuilder::new());

    for r in records {
        rt_rowids.push(r.rt_rowid);
        corrs.push(r.correlation_id);
        pids.push(r.native_pid);
        devs.push(r.device_id);
        ctxs.push(r.context_id);
        for e in &r.enclosing {
            rowids_builder.values().append_value(e.nvtx_rowid);
            names_builder.values().append_value(&e.nvtx_name);
        }
        rowids_builder.append(true);
        names_builder.append(true);
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(rt_rowids)),
        Arc::new(Int64Array::from(corrs)),
        Arc::new(Int64Array::from(pids)),
        Arc::new(Int32Array::from(devs)),
        Arc::new(Int64Array::from(ctxs)),
        Arc::new(rowids_builder.finish()),
        Arc::new(names_builder.finish()),
    ];
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns)
        .map_err(crate::NsysDataError::nvtx_parent_record_batch)?;

    // Embed fingerprint + format version as parquet KV metadata. Warm
    // open reads only the footer (cheap) to validate before scanning.
    let kv = crate::sidecar::freshness_kv(KV_VERSION, RUNTIME_NVTX_PARENT_VERSION, fp);
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_key_value_metadata(Some(kv))
        .build();

    crate::sidecar::atomic_publish(path, |tmp| {
        let file = File::create(tmp).map_err(|source| {
            crate::NsysDataError::nvtx_parent_sidecar_create(tmp.display(), source)
        })?;
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).map_err(|source| {
            crate::NsysDataError::nvtx_parent_writer_open(tmp.display(), source)
        })?;
        writer.write(&batch).map_err(|source| {
            crate::NsysDataError::nvtx_parent_writer_write(tmp.display(), source)
        })?;
        writer.close().map_err(|source| {
            crate::NsysDataError::nvtx_parent_writer_close(tmp.display(), source)
        })?;
        Ok(())
    })
}

pub(super) fn read_parquet(path: &Path) -> NsysDataResult<Vec<RuntimeParentEntry>> {
    let file = File::open(path)
        .map_err(|source| crate::NsysDataError::nvtx_parent_sidecar_open(path.display(), source))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|source| crate::NsysDataError::nvtx_parent_reader_open(path.display(), source))?;
    let reader = builder
        .build()
        .map_err(|source| crate::NsysDataError::nvtx_parent_reader_build(path.display(), source))?;
    let mut out: Vec<RuntimeParentEntry> = Vec::new();
    for batch in reader {
        let batch = batch.map_err(|source| {
            crate::NsysDataError::nvtx_parent_batch_read(path.display(), source)
        })?;
        let rt = nvtx_parent_column::<Int64Array>(&batch, 0, "rt_rowid", "Int64", path)?;
        let corr = nvtx_parent_column::<Int64Array>(&batch, 1, "correlation_id", "Int64", path)?;
        let pid = nvtx_parent_column::<Int64Array>(&batch, 2, "native_pid", "Int64", path)?;
        let dev = nvtx_parent_column::<Int32Array>(&batch, 3, "device_id", "Int32", path)?;
        let ctx = nvtx_parent_column::<Int64Array>(&batch, 4, "context_id", "Int64", path)?;
        let nvtx_rowids_col =
            nvtx_parent_column::<ListArray>(&batch, 5, "nvtx_rowids", "List<Int64>", path)?;
        let nvtx_names_col =
            nvtx_parent_column::<ListArray>(&batch, 6, "nvtx_names", "List<Utf8>", path)?;
        for i in 0..batch.num_rows() {
            let ids_arr = nvtx_rowids_col.value(i);
            let ids = ids_arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| {
                    crate::NsysDataError::nvtx_parent_column_type_mismatch(
                        path.display(),
                        "nvtx_rowids",
                        "List<Int64>",
                        format!("List<{:?}>", ids_arr.data_type()),
                    )
                })?;
            let names_arr = nvtx_names_col.value(i);
            let names = names_arr
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    crate::NsysDataError::nvtx_parent_column_type_mismatch(
                        path.display(),
                        "nvtx_names",
                        "List<Utf8>",
                        format!("List<{:?}>", names_arr.data_type()),
                    )
                })?;
            if ids.len() != names.len() {
                return Err(crate::NsysDataError::nvtx_parent_list_length_mismatch(
                    i,
                    ids.len(),
                    names.len(),
                ));
            }
            let mut enclosing: Vec<EnclosingNvtx> = Vec::with_capacity(ids.len());
            for j in 0..ids.len() {
                enclosing.push(EnclosingNvtx {
                    nvtx_rowid: ids.value(j),
                    nvtx_name: names.value(j).to_string(),
                });
            }
            out.push(RuntimeParentEntry {
                rt_rowid: rt.value(i),
                correlation_id: if corr.is_null(i) {
                    None
                } else {
                    Some(corr.value(i))
                },
                native_pid: pid.value(i),
                device_id: if dev.is_null(i) {
                    None
                } else {
                    Some(dev.value(i))
                },
                context_id: if ctx.is_null(i) {
                    None
                } else {
                    Some(ctx.value(i))
                },
                enclosing,
            });
        }
    }
    Ok(out)
}

fn nvtx_parent_column<'a, T: 'static>(
    batch: &'a RecordBatch,
    index: usize,
    column: &str,
    expected: &str,
    path: &Path,
) -> NsysDataResult<&'a T> {
    let array = batch
        .columns()
        .get(index)
        .ok_or_else(|| crate::NsysDataError::nvtx_parent_column_missing(path.display(), column))?;
    array.as_any().downcast_ref::<T>().ok_or_else(|| {
        crate::NsysDataError::nvtx_parent_column_type_mismatch(
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
        RUNTIME_NVTX_PARENT_VERSION,
        fp,
        "runtime_nvtx_parent",
    )?)
}
