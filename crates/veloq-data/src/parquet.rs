use crate::{DataError, DataResult};
use ::parquet::arrow::ArrowWriter;
use ::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use ::parquet::file::metadata::KeyValue;
use ::parquet::file::properties::WriterProperties;
use arrow::array::ArrayRef;
use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use veloq_core::SourceFingerprint;

pub use crate::file::atomic_publish;

pub const KV_MTIME: &str = "veloq.source.mtime_secs";
pub const KV_SIZE: &str = "veloq.source.size";

pub fn freshness_kv(version_key: &str, version: u32, fp: SourceFingerprint) -> Vec<KeyValue> {
    vec![
        KeyValue::new(version_key.to_string(), Some(version.to_string())),
        KeyValue::new(KV_MTIME.to_string(), Some(fp.mtime_secs.to_string())),
        KeyValue::new(KV_SIZE.to_string(), Some(fp.size.to_string())),
    ]
}

pub fn is_fresh(
    path: &Path,
    version_key: &str,
    version: u32,
    fp: SourceFingerprint,
    label: &str,
) -> DataResult<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(_) => return Ok(false),
    };
    let builder = match ParquetRecordBatchReaderBuilder::try_new(file) {
        Ok(builder) => builder,
        Err(err) => {
            log::warn!(
                "{label}: ignoring unreadable sidecar at {}: {err}",
                path.display()
            );
            return Ok(false);
        }
    };
    let kvs = builder
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .cloned()
        .unwrap_or_default();
    let mut got_version: Option<u32> = None;
    let mut mtime: Option<i64> = None;
    let mut size: Option<u64> = None;
    for kv in &kvs {
        let Some(value) = &kv.value else { continue };
        if kv.key == version_key {
            got_version = value.parse().ok();
        } else if kv.key == KV_MTIME {
            mtime = value.parse().ok();
        } else if kv.key == KV_SIZE {
            size = value.parse().ok();
        }
    }
    let (Some(got_version), Some(mtime), Some(size)) = (got_version, mtime, size) else {
        return Ok(false);
    };
    Ok(got_version == version && mtime == fp.mtime_secs && size == fp.size)
}

pub fn write_record_batch_atomic(
    path: &Path,
    schema: Schema,
    columns: Vec<ArrayRef>,
    properties: Option<WriterProperties>,
) -> DataResult<()> {
    let schema = Arc::new(schema);
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns)
        .map_err(|source| DataError::build_parquet_batch(path, source))?;
    atomic_publish(path, |tmp| -> DataResult<()> {
        let file = fs::File::create(tmp).map_err(|source| DataError::write_file(tmp, source))?;
        let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), properties)
            .map_err(|source| DataError::open_parquet_writer(tmp, source))?;
        writer
            .write(&batch)
            .map_err(|source| DataError::write_parquet_batch(tmp, source))?;
        writer
            .close()
            .map_err(|source| DataError::close_parquet_writer(tmp, source))?;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;
    use arrow::datatypes::{DataType, Field};

    #[test]
    fn atomic_publish_promotes_on_success_and_leaves_no_tmp() -> DataResult<()> {
        let dir = tempfile::tempdir()
            .map_err(|source| DataError::create_dir(Path::new("temporary directory"), source))?;
        let path = dir.path().join("sub/side.parquet");
        atomic_publish(&path, |tmp| -> DataResult<()> {
            fs::write(tmp, b"payload").map_err(|source| DataError::write_file(tmp, source))
        })?;

        let payload = fs::read(&path).map_err(|source| DataError::read_file(&path, source))?;
        assert_eq!(payload, b"payload");
        assert!(!crate::file::sibling_tmp(&path).exists());
        Ok(())
    }

    #[test]
    fn atomic_publish_failure_leaves_previous_intact() -> DataResult<()> {
        let dir = tempfile::tempdir()
            .map_err(|source| DataError::create_dir(Path::new("temporary directory"), source))?;
        let path = dir.path().join("side.parquet");
        fs::write(&path, b"old").map_err(|source| DataError::write_file(&path, source))?;

        let err = atomic_publish(&path, |tmp| -> DataResult<()> {
            fs::write(tmp, b"new").map_err(|source| DataError::write_file(tmp, source))?;
            Err(DataError::write_file(
                tmp,
                std::io::Error::other("simulated build failure"),
            ))
        });

        assert!(err.is_err());
        let payload = fs::read(&path).map_err(|source| DataError::read_file(&path, source))?;
        assert_eq!(payload, b"old");
        Ok(())
    }

    #[test]
    fn is_fresh_matches_only_on_version_and_fingerprint() -> DataResult<()> {
        let dir = tempfile::tempdir()
            .map_err(|source| DataError::create_dir(Path::new("temporary directory"), source))?;
        let path = dir.path().join("side.parquet");
        let fp = SourceFingerprint {
            mtime_secs: 1_000,
            size: 4_096,
        };
        const VERSION_KEY: &str = "veloq.test.version";

        write_freshness_fixture(&path, VERSION_KEY, 3, fp)?;

        assert!(is_fresh(&path, VERSION_KEY, 3, fp, "test")?);
        assert!(!is_fresh(&path, VERSION_KEY, 4, fp, "test")?);
        assert!(!is_fresh(
            &path,
            VERSION_KEY,
            3,
            SourceFingerprint {
                mtime_secs: 999,
                size: 4_096,
            },
            "test",
        )?);
        assert!(!is_fresh(
            &path,
            VERSION_KEY,
            3,
            SourceFingerprint {
                mtime_secs: 1_000,
                size: 1,
            },
            "test",
        )?);
        Ok(())
    }

    fn write_freshness_fixture(
        path: &Path,
        version_key: &str,
        version: u32,
        fp: SourceFingerprint,
    ) -> DataResult<()> {
        let props = WriterProperties::builder()
            .set_key_value_metadata(Some(freshness_kv(version_key, version, fp)))
            .build();
        write_record_batch_atomic(
            path,
            Schema::new(vec![Field::new("x", DataType::Int64, false)]),
            vec![Arc::new(Int64Array::from(vec![1i64]))],
            Some(props),
        )
    }
}
