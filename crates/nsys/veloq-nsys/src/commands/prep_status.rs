use std::path::Path;

use crate::error::{NsysSourceError, NsysSourceResult};
use crate::payloads::{ParquetCacheStatus, PrepAuxiliary, PrepPayload, PrepRow};

/// Assemble the `prep` / `prep --status` payload.
///
/// The caller decides whether sidecars were prepared first. This
/// function is read-only: it reports the current parquetdir + registered
/// sidecar state without building missing artifacts.
pub(super) fn collect_prep_response(
    trace: &Path,
    prepared: bool,
    elapsed_ms: u64,
) -> NsysSourceResult<PrepPayload> {
    let source_path = veloq_nsys_data::nsys_rep::sidecar_source_path(trace);
    let parquet_cache = collect_parquet_status(trace, &source_path)?;
    let rows = veloq_nsys_data::sidecar_registry::sidecar_statuses(&source_path)
        .into_iter()
        .map(PrepRow::from)
        .collect::<Vec<_>>();
    Ok(PrepPayload {
        count: rows.len(),
        total_matched: rows.len(),
        rows,
        auxiliary: PrepAuxiliary {
            cache_root: veloq_core::artifact_dir_for(&source_path)
                .display()
                .to_string(),
            parquet_cache,
            prepared,
            elapsed_ms,
        },
    })
}

fn collect_parquet_status(
    trace: &Path,
    source_path: &Path,
) -> NsysSourceResult<ParquetCacheStatus> {
    let parquet_dir = if source_path.extension().and_then(|e| e.to_str()) == Some("nsys-rep") {
        veloq_nsys_data::nsys_rep::pqtdir_path_for(source_path)
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
    Ok(ParquetCacheStatus {
        dir: parquet_dir.display().to_string(),
        present: parquet_dir.is_dir(),
        tables,
    })
}
