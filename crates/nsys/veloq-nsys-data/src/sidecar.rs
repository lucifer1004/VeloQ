//! Shared per-trace parquet-sidecar scaffold.
//!
//! NSys keeps this module as its internal import surface, but the
//! source-neutral implementation lives in `veloq-data`.

pub use veloq_data::parquet::{KV_MTIME, KV_SIZE, atomic_publish, freshness_kv, is_fresh};

use crate::NsysDataResult;
use std::path::{Path, PathBuf};
use veloq_core::SourceFingerprint;

/// Result of ensuring a parquet sidecar exists and matches the source
/// fingerprint. `rebuilt_records` is populated only on a cold/stale build
/// so callers can log counts or reuse the just-computed records.
pub(crate) struct FreshSidecar<T> {
    pub path: PathBuf,
    pub rebuilt_records: Option<T>,
}

pub(crate) fn ensure_fresh_sidecar<T>(
    path: PathBuf,
    fingerprint: SourceFingerprint,
    is_sidecar_fresh: impl FnOnce(&Path, SourceFingerprint) -> NsysDataResult<bool>,
    compute: impl FnOnce() -> NsysDataResult<T>,
    write: impl FnOnce(&Path, SourceFingerprint, &T) -> NsysDataResult<()>,
) -> NsysDataResult<FreshSidecar<T>> {
    if is_sidecar_fresh(&path, fingerprint)? {
        return Ok(FreshSidecar {
            path,
            rebuilt_records: None,
        });
    }

    let records = compute()?;
    write(&path, fingerprint, &records)?;
    Ok(FreshSidecar {
        path,
        rebuilt_records: Some(records),
    })
}

pub(crate) fn load_if_fresh<T>(
    path: &Path,
    fingerprint: SourceFingerprint,
    is_sidecar_fresh: impl FnOnce(&Path, SourceFingerprint) -> NsysDataResult<bool>,
    load: impl FnOnce(&Path) -> NsysDataResult<T>,
) -> NsysDataResult<Option<T>> {
    if !is_sidecar_fresh(path, fingerprint)? {
        return Ok(None);
    }
    load(path).map(Some)
}
