use std::path::Path;

use crate::error::{NsysSourceError, NsysSourceResult};
use crate::payloads::{ParquetCacheStatus, PrepStatusPayload, SidecarStatus};

/// `veloq prep --status` — assemble the cache-status payload without
/// rebuilding anything. Reads filesystem metadata only. The
/// parquetdir has no manifest; its contents are
/// whatever `nsys export -t parquetdir` last wrote next to the trace.
pub(super) fn collect_prep_status(trace: &Path) -> NsysSourceResult<PrepStatusPayload> {
    // Where the parquetdir lives. For a `.nsys-rep`, that's
    // `<trace>.veloq/parquetdir/`; for a directly-passed `_pqtdir/`,
    // that's the input itself.
    let source_path = veloq_nsys_data::nsys_rep::sidecar_source_path(trace);
    let parquet_dir = if source_path.extension().and_then(|e| e.to_str()) == Some("nsys-rep") {
        veloq_nsys_data::nsys_rep::pqtdir_path_for(&source_path)
    } else {
        trace.to_path_buf()
    };
    let mut tables: Vec<String> = if parquet_dir.is_dir() {
        std::fs::read_dir(&parquet_dir)
            .map_err(|source| {
                NsysSourceError::prep_status_read_parquetdir(parquet_dir.display(), source)
            })?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "parquet"))
            .filter_map(|p| p.file_stem().and_then(|s| s.to_str()).map(str::to_owned))
            .collect()
    } else {
        Vec::new()
    };
    tables.sort();
    let parquet_status = ParquetCacheStatus {
        dir: parquet_dir.display().to_string(),
        present: parquet_dir.is_dir(),
        tables,
    };

    let meta_path = veloq_nsys_data::meta_cache::path_for(&source_path);
    let (present, size_bytes, mtime_secs) = match std::fs::metadata(&meta_path) {
        Ok(m) => {
            let mtime = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64);
            (true, Some(m.len()), mtime)
        }
        Err(_) => (false, None, None),
    };
    let meta_version_on_disk = veloq_nsys_data::meta_cache::read_header(&source_path)
        .ok()
        .flatten()
        .map(|h| h.version);
    // `try_load_existing` returns `Some(_)` only when the sidecar's
    // version + trace fingerprint both validate. Errors fold to
    // `false` so a corrupt or unreadable file shows up as
    // "present but not fingerprint-matching."
    let fingerprint_match = veloq_nsys_data::meta_cache::try_load_existing(&source_path)
        .ok()
        .flatten()
        .is_some();
    let meta_status = SidecarStatus {
        path: meta_path.display().to_string(),
        present,
        size_bytes,
        mtime_secs,
        format_version_expected: veloq_nsys_data::META_CACHE_VERSION,
        format_version_on_disk: meta_version_on_disk,
        fingerprint_match,
    };

    Ok(PrepStatusPayload {
        cache_root: veloq_core::artifact_dir_for(&source_path)
            .display()
            .to_string(),
        parquet_cache: parquet_status,
        meta_cache: meta_status,
    })
}
