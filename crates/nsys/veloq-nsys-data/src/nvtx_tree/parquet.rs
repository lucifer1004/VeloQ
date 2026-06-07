use super::{NVTX_TREE_VERSION, NvtxTreeRecord};
use crate::NsysDataResult;
use arrow::array::{Array, ArrayRef, Int32Array, Int64Array, StringArray, StringBuilder};
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
    Arc::new(Schema::new(vec![
        Field::new("range_id", DataType::Int64, false),
        Field::new("parent_range_id", DataType::Int64, true),
        Field::new("depth", DataType::Int32, false),
        Field::new("domain_id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("path", DataType::Utf8, false),
        Field::new("start", DataType::Int64, false),
        Field::new("end", DataType::Int64, true),
        Field::new("duration_ns", DataType::Int64, true),
        Field::new("global_tid", DataType::Int64, false),
    ]))
}

pub(super) const KV_VERSION: &str = "veloq.nvtx_tree.version";

pub(super) fn write_parquet(
    path: &Path,
    fp: SourceFingerprint,
    records: &[NvtxTreeRecord],
) -> NsysDataResult<()> {
    let schema = parquet_schema();

    let mut range_ids: Vec<i64> = Vec::with_capacity(records.len());
    let mut parents: Vec<Option<i64>> = Vec::with_capacity(records.len());
    let mut depths: Vec<i32> = Vec::with_capacity(records.len());
    let mut domains: Vec<i64> = Vec::with_capacity(records.len());
    let mut names = StringBuilder::new();
    let mut paths = StringBuilder::new();
    let mut starts: Vec<i64> = Vec::with_capacity(records.len());
    let mut ends: Vec<Option<i64>> = Vec::with_capacity(records.len());
    let mut durations: Vec<Option<i64>> = Vec::with_capacity(records.len());
    let mut tids: Vec<i64> = Vec::with_capacity(records.len());

    for r in records {
        range_ids.push(r.range_id);
        parents.push(r.parent_range_id);
        depths.push(r.depth);
        domains.push(r.domain_id);
        names.append_value(&r.name);
        paths.append_value(&r.path);
        starts.push(r.start);
        ends.push(r.end);
        durations.push(r.duration_ns);
        tids.push(r.global_tid);
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(range_ids)),
        Arc::new(Int64Array::from(parents)),
        Arc::new(Int32Array::from(depths)),
        Arc::new(Int64Array::from(domains)),
        Arc::new(names.finish()),
        Arc::new(paths.finish()),
        Arc::new(Int64Array::from(starts)),
        Arc::new(Int64Array::from(ends)),
        Arc::new(Int64Array::from(durations)),
        Arc::new(Int64Array::from(tids)),
    ];
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns)
        .map_err(crate::NsysDataError::nvtx_tree_record_batch)?;

    let kv = crate::sidecar::freshness_kv(KV_VERSION, NVTX_TREE_VERSION, fp);
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_key_value_metadata(Some(kv))
        .build();

    crate::sidecar::atomic_publish(path, |tmp| {
        let file = File::create(tmp).map_err(|source| {
            crate::NsysDataError::nvtx_tree_sidecar_create(tmp.display(), source)
        })?;
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))
            .map_err(|source| crate::NsysDataError::nvtx_tree_writer_open(tmp.display(), source))?;
        writer.write(&batch).map_err(|source| {
            crate::NsysDataError::nvtx_tree_writer_write(tmp.display(), source)
        })?;
        writer.close().map_err(|source| {
            crate::NsysDataError::nvtx_tree_writer_close(tmp.display(), source)
        })?;
        Ok(())
    })
}

pub(super) fn read_parquet(path: &Path) -> NsysDataResult<Vec<NvtxTreeRecord>> {
    let file = File::open(path)
        .map_err(|source| crate::NsysDataError::nvtx_tree_sidecar_open(path.display(), source))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|source| crate::NsysDataError::nvtx_tree_reader_open(path.display(), source))?;
    let reader = builder
        .build()
        .map_err(|source| crate::NsysDataError::nvtx_tree_reader_build(path.display(), source))?;
    let mut out: Vec<NvtxTreeRecord> = Vec::new();
    for batch in reader {
        let batch = batch
            .map_err(|source| crate::NsysDataError::nvtx_tree_batch_read(path.display(), source))?;
        let range_ids = nvtx_tree_column::<Int64Array>(&batch, 0, "range_id", "Int64", path)?;
        let parents = nvtx_tree_column::<Int64Array>(&batch, 1, "parent_range_id", "Int64", path)?;
        let depths = nvtx_tree_column::<Int32Array>(&batch, 2, "depth", "Int32", path)?;
        let domains = nvtx_tree_column::<Int64Array>(&batch, 3, "domain_id", "Int64", path)?;
        let names = nvtx_tree_column::<StringArray>(&batch, 4, "name", "Utf8", path)?;
        let paths = nvtx_tree_column::<StringArray>(&batch, 5, "path", "Utf8", path)?;
        let starts = nvtx_tree_column::<Int64Array>(&batch, 6, "start", "Int64", path)?;
        let ends = nvtx_tree_column::<Int64Array>(&batch, 7, "end", "Int64", path)?;
        let durations = nvtx_tree_column::<Int64Array>(&batch, 8, "duration_ns", "Int64", path)?;
        let tids = nvtx_tree_column::<Int64Array>(&batch, 9, "global_tid", "Int64", path)?;
        for i in 0..batch.num_rows() {
            out.push(NvtxTreeRecord {
                range_id: range_ids.value(i),
                parent_range_id: if parents.is_null(i) {
                    None
                } else {
                    Some(parents.value(i))
                },
                depth: depths.value(i),
                domain_id: domains.value(i),
                name: names.value(i).to_string(),
                path: paths.value(i).to_string(),
                start: starts.value(i),
                end: if ends.is_null(i) {
                    None
                } else {
                    Some(ends.value(i))
                },
                duration_ns: if durations.is_null(i) {
                    None
                } else {
                    Some(durations.value(i))
                },
                global_tid: tids.value(i),
            });
        }
    }
    Ok(out)
}

fn nvtx_tree_column<'a, T: 'static>(
    batch: &'a RecordBatch,
    index: usize,
    column: &str,
    expected: &str,
    path: &Path,
) -> NsysDataResult<&'a T> {
    let array = batch
        .columns()
        .get(index)
        .ok_or_else(|| crate::NsysDataError::nvtx_tree_column_missing(path.display(), column))?;
    array.as_any().downcast_ref::<T>().ok_or_else(|| {
        crate::NsysDataError::nvtx_tree_column_type_mismatch(
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
        NVTX_TREE_VERSION,
        fp,
        "nvtx_tree",
    )?)
}
