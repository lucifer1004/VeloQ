//! Shared per-trace parquet-sidecar scaffold.
//!
//! The NVTX-tree ([`crate::nvtx_tree`]) and runtime→enclosing-NVTX
//! ([`crate::runtime_nvtx_parent`]) sidecars are independent parquet
//! files derived from a `.nsys-rep` parquetdir, but they stamp+check
//! freshness and publish atomically the same way. This module is that
//! shared machinery:
//!
//! - [`freshness_kv`] — the version + source-fingerprint KV metadata
//!   every sidecar writes into its parquet footer.
//! - [`is_fresh`] — read the footer back and decide whether a cached
//!   sidecar matches the current source (else: rebuild).
//! - [`atomic_publish`] — write to `<path>.tmp` then `rename`, so a
//!   crashed/failed build never leaves a half-written sidecar in place.
//!
//! Only the per-sidecar **version key** (e.g. `veloq.nvtx_tree.version`)
//! differs between consumers; the source mtime/size keys are shared.

use anyhow::{Context, Result};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::metadata::KeyValue;
use std::fs::{self};
use std::path::{Path, PathBuf};
use veloq_core::SourceFingerprint;

/// Footer KV key for the source parquetdir's mtime (seconds). Shared by
/// every sidecar — freshness is keyed on the *source*, not the sidecar.
pub const KV_MTIME: &str = "veloq.source.mtime_secs";
/// Footer KV key for the source parquetdir's byte size.
pub const KV_SIZE: &str = "veloq.source.size";

/// The three KV-metadata pairs a per-trace sidecar stamps into its
/// parquet footer: a per-sidecar `version` under `version_key`, plus the
/// shared source mtime + size fingerprint. Splice into
/// `WriterProperties::set_key_value_metadata`.
pub fn freshness_kv(version_key: &str, version: u32, fp: SourceFingerprint) -> Vec<KeyValue> {
    vec![
        KeyValue::new(version_key.to_string(), Some(version.to_string())),
        KeyValue::new(KV_MTIME.to_string(), Some(fp.mtime_secs.to_string())),
        KeyValue::new(KV_SIZE.to_string(), Some(fp.size.to_string())),
    ]
}

/// True iff the parquet at `path` exists, is readable, and its footer
/// records `version` under `version_key` plus an mtime+size matching
/// `fp`. Missing / unreadable / version-or-fingerprint mismatch all
/// return `Ok(false)` (i.e. rebuild) rather than erroring — a cache is
/// never load-bearing. `label` names the consumer in the unreadable-file
/// warning.
pub fn is_fresh(
    path: &Path,
    version_key: &str,
    version: u32,
    fp: SourceFingerprint,
    label: &str,
) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return Ok(false),
    };
    let builder = match ParquetRecordBatchReaderBuilder::try_new(file) {
        Ok(b) => b,
        Err(e) => {
            log::warn!(
                "{label}: ignoring unreadable sidecar at {}: {e}",
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
        let Some(v) = &kv.value else { continue };
        if kv.key == version_key {
            got_version = v.parse().ok();
        } else if kv.key == KV_MTIME {
            mtime = v.parse().ok();
        } else if kv.key == KV_SIZE {
            size = v.parse().ok();
        }
    }
    let (Some(got_version), Some(mtime), Some(size)) = (got_version, mtime, size) else {
        return Ok(false);
    };
    Ok(got_version == version && mtime == fp.mtime_secs && size == fp.size)
}

/// Atomic publish: create `path`'s parent dir, invoke `write_to` against
/// a sibling `<path>.tmp`, then `rename` it onto `path`. A `write_to`
/// that errors leaves any existing sidecar at `path` untouched and the
/// `.tmp` is not promoted.
pub fn atomic_publish(path: &Path, write_to: impl FnOnce(&Path) -> Result<()>) -> Result<()> {
    let tmp = {
        let mut s = path.as_os_str().to_os_string();
        s.push(".tmp");
        PathBuf::from(s)
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    write_to(&tmp)?;
    fs::rename(&tmp, path).with_context(|| format!("publishing sidecar to {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_publish_promotes_on_success_and_leaves_no_tmp() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("sub/side.parquet");
        atomic_publish(&path, |tmp| fs::write(tmp, b"payload").map_err(Into::into))?;
        assert_eq!(fs::read(&path)?, b"payload");
        assert!(
            !dir.path().join("sub/side.parquet.tmp").exists(),
            "tmp must be renamed away"
        );
        Ok(())
    }

    #[test]
    fn atomic_publish_failure_leaves_previous_intact() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("side.parquet");
        fs::write(&path, b"old")?;
        let err = atomic_publish(&path, |tmp| {
            // Write the tmp, then fail before promotion.
            fs::write(tmp, b"new").ok();
            Err(anyhow::anyhow!("simulated build failure"))
        });
        assert!(err.is_err());
        // Previous sidecar untouched; no half-written promotion.
        assert_eq!(fs::read(&path)?, b"old");
        Ok(())
    }

    #[test]
    fn is_fresh_matches_only_on_version_and_fingerprint() -> Result<()> {
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use parquet::file::properties::WriterProperties;
        use std::sync::Arc;

        let dir = tempfile::tempdir()?;
        let path = dir.path().join("side.parquet");
        let fp = SourceFingerprint {
            mtime_secs: 1_000,
            size: 4_096,
        };
        const VK: &str = "veloq.test.version";

        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
        let write = |version: u32, fp: SourceFingerprint| -> Result<()> {
            let props = WriterProperties::builder()
                .set_key_value_metadata(Some(freshness_kv(VK, version, fp)))
                .build();
            let file = fs::File::create(&path)?;
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int64Array::from(vec![1i64]))],
            )?;
            let mut w = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props))?;
            w.write(&batch)?;
            w.close()?;
            Ok(())
        };

        write(3, fp)?;
        assert!(is_fresh(&path, VK, 3, fp, "test")?, "exact match");
        assert!(!is_fresh(&path, VK, 4, fp, "test")?, "version bump");
        assert!(
            !is_fresh(
                &path,
                VK,
                3,
                SourceFingerprint {
                    mtime_secs: 999,
                    size: 4_096
                },
                "test"
            )?,
            "mtime change"
        );
        assert!(
            !is_fresh(
                &path,
                VK,
                3,
                SourceFingerprint {
                    mtime_secs: 1_000,
                    size: 1
                },
                "test"
            )?,
            "size change"
        );
        Ok(())
    }
}
