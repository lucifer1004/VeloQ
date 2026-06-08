//! Registry for NSys-generated sidecars.
//!
//! The registry keeps source-owned artifact lifecycle metadata in one
//! place: path, version, freshness, optional DuckDB view attachment, and
//! prep/status reporting. Individual sidecar modules still own their
//! private record schema and build logic.

use crate::{NsysDataResult, Trace, gpu_work_events, meta_cache, nsys_rep};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

/// One source-owned generated artifact that may be prepared and reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NsysSidecar {
    MetaCache,
    GpuWorkEvents,
}

impl NsysSidecar {
    pub const fn id(self) -> &'static str {
        match self {
            Self::MetaCache => "meta-cache",
            Self::GpuWorkEvents => "gpu-work-events",
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::MetaCache => "meta.bin",
            Self::GpuWorkEvents => "gpu-work-events.parquet",
        }
    }

    pub const fn format_version_expected(self) -> u32 {
        match self {
            Self::MetaCache => meta_cache::META_CACHE_VERSION,
            Self::GpuWorkEvents => gpu_work_events::GPU_WORK_EVENTS_VERSION,
        }
    }

    pub fn path_for(self, source_path: &Path) -> PathBuf {
        match self {
            Self::MetaCache => meta_cache::path_for(source_path),
            Self::GpuWorkEvents => gpu_work_events::sidecar_path_for(source_path),
        }
    }

    pub fn status_for(self, source_path: &Path) -> NsysSidecarStatus {
        let path = self.path_for(source_path);
        let (present, size_bytes, mtime_secs) = file_state(&path);
        let format_version_on_disk = match self {
            Self::MetaCache => meta_cache::read_header(source_path)
                .ok()
                .flatten()
                .map(|header| header.version),
            Self::GpuWorkEvents => gpu_work_events::format_version_on_disk(&path),
        };
        let fingerprint_match = match self {
            Self::MetaCache => meta_cache::try_load_existing(source_path)
                .ok()
                .flatten()
                .is_some(),
            Self::GpuWorkEvents => {
                gpu_work_events::sidecar_is_fresh_for_trace(source_path).unwrap_or(false)
            }
        };
        NsysSidecarStatus {
            key: format!("sidecar|{}", self.id()),
            id: self.id().to_string(),
            name: self.name().to_string(),
            path: path.display().to_string(),
            present,
            size_bytes,
            mtime_secs,
            format_version_expected: self.format_version_expected(),
            format_version_on_disk,
            fingerprint_match,
        }
    }

    pub fn ensure(self, trace: &Trace) -> NsysDataResult<PathBuf> {
        match self {
            Self::MetaCache => {
                trace.meta_cache()?;
                Ok(meta_cache::path_for(trace.path()))
            }
            Self::GpuWorkEvents => gpu_work_events::ensure_sidecar(trace),
        }
    }

    fn attach_view_if_present(
        self,
        conn: &duckdb::Connection,
        source_path: &Path,
    ) -> NsysDataResult<()> {
        match self {
            Self::MetaCache => Ok(()),
            Self::GpuWorkEvents => gpu_work_events::attach_view_if_present(conn, source_path),
        }
    }
}

/// Sidecars warmed by `veloq prep`.
pub const PREP_SIDECARS: &[NsysSidecar] = &[NsysSidecar::MetaCache, NsysSidecar::GpuWorkEvents];

/// Read-only state for one registered sidecar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NsysSidecarStatus {
    pub key: String,
    pub id: String,
    pub name: String,
    pub path: String,
    pub present: bool,
    pub size_bytes: Option<u64>,
    pub mtime_secs: Option<i64>,
    pub format_version_expected: u32,
    pub format_version_on_disk: Option<u32>,
    pub fingerprint_match: bool,
}

pub fn sidecar_statuses(trace_path: &Path) -> Vec<NsysSidecarStatus> {
    let source_path = nsys_rep::sidecar_source_path(trace_path);
    PREP_SIDECARS
        .iter()
        .map(|sidecar| sidecar.status_for(&source_path))
        .collect()
}

pub fn ensure_prep_sidecars(trace: &Trace) -> NsysDataResult<Vec<NsysSidecarStatus>> {
    for sidecar in PREP_SIDECARS {
        sidecar.ensure(trace)?;
    }
    Ok(sidecar_statuses(trace.path()))
}

/// Attach registered optional views on trace open.
///
/// These views are accelerators, not authoritative inputs. A bad or
/// unreadable acceleration sidecar must not make unrelated commands fail;
/// commands that can use the sidecar keep their cold raw-table fallback.
pub(crate) fn attach_optional_views(conn: &duckdb::Connection, source_path: &Path) {
    for sidecar in PREP_SIDECARS {
        if let Err(err) = sidecar.attach_view_if_present(conn, source_path) {
            log::warn!(
                "{}: optional sidecar view attachment skipped: {err:#}",
                sidecar.id()
            );
        }
    }
}

fn file_state(path: &Path) -> (bool, Option<u64>, Option<i64>) {
    match std::fs::metadata(path) {
        Ok(meta) => {
            let mtime_secs = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64);
            (true, Some(meta.len()), mtime_secs)
        }
        Err(_) => (false, None, None),
    }
}
