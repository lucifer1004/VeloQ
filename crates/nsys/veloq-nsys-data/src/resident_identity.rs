//! Side-effect-free identity and freshness inputs for resident NSys sessions.

use std::fs;
use std::mem;
use std::path::{Path, PathBuf};

use veloq_core::artifact_dir_for;

use crate::{gpu_work_events, nsys_rep, nvtx_tree, quote_sql_identifier};

const FRESHNESS_SIDECARS: &[&str] = &[
    "correlation.bin",
    "meta.bin",
    "gpu-work-events.parquet",
    "nvtx-parent.parquet",
    "nvtx-tree.parquet",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentTraceIdentity {
    pub canonical_source_path: PathBuf,
    pub freshness_key: String,
    pub resident_memory_estimate_bytes: u64,
}

/// Resolve aliases and fingerprint every persistent input that can affect an
/// already-open NSys session. This function performs metadata reads only: it
/// never exports a report or creates a sidecar.
pub fn resident_trace_identity(path: &Path) -> std::io::Result<ResidentTraceIdentity> {
    let source_path =
        nsys_rep::generated_parquetdir_owner(path).unwrap_or_else(|| path.to_path_buf());
    let canonical_source_path =
        fs::canonicalize(&source_path).unwrap_or_else(|_| source_path.clone());
    let parquetdir = if source_path.extension().is_some_and(|ext| ext == "nsys-rep") {
        nsys_rep::pqtdir_path_for(&source_path)
    } else {
        source_path.clone()
    };
    let artifact_root = artifact_dir_for(&source_path);

    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    hash_path(&mut hash, "source", &source_path)?;
    let parquet_paths = hash_parquetdir(&mut hash, &parquetdir)?;
    for name in FRESHNESS_SIDECARS {
        let sidecar = artifact_root.join(name);
        hash_path(&mut hash, name, &sidecar)?;
    }

    Ok(ResidentTraceIdentity {
        canonical_source_path,
        freshness_key: format!("{hash:016x}"),
        resident_memory_estimate_bytes: resident_structure_estimate(
            &source_path,
            &parquetdir,
            &artifact_root,
            &parquet_paths,
        ),
    })
}

fn hash_parquetdir(hash: &mut u64, path: &Path) -> std::io::Result<Vec<PathBuf>> {
    hash_path(hash, "parquetdir", path)?;
    if !path.is_dir() {
        return Ok(Vec::new());
    }
    let mut files = Vec::new();
    for entry in fs::read_dir(path)? {
        let path = entry?.path();
        if path.extension().is_some_and(|ext| ext == "parquet") {
            files.push(path);
        }
    }
    files.sort();
    hash_u64(hash, files.len() as u64);
    for file in &files {
        let name = file
            .file_name()
            .map(|name| name.to_string_lossy())
            .unwrap_or_default();
        hash_path(hash, &name, file)?;
    }
    Ok(files)
}

/// Estimate only the daemon-owned structures retained after opening a trace.
///
/// Parquet and sidecar payload bytes remain persistent inputs and are
/// deliberately excluded. DuckDB's reusable catalog state is represented by
/// the SQL definitions it retains for source and optional sidecar views.
fn resident_structure_estimate(
    source_path: &Path,
    parquetdir: &Path,
    artifact_root: &Path,
    parquet_paths: &[PathBuf],
) -> u64 {
    let mut bytes = mem::size_of::<crate::Trace>() as u64;
    bytes = add_path_bytes(bytes, source_path);
    bytes = add_path_bytes(bytes, parquetdir);
    bytes = add_usize(
        bytes,
        parquet_paths.len().saturating_mul(mem::size_of::<String>()),
    );
    bytes = add_usize(bytes, "CREATE SCHEMA IF NOT EXISTS nsight".len());

    for path in parquet_paths {
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        bytes = add_usize(bytes, stem.len());
        if let Some(sql) = parquet_view_sql(stem, path) {
            bytes = add_usize(bytes, sql.len());
        }
    }

    let gpu_work = artifact_root.join("gpu-work-events.parquet");
    if gpu_work.is_file()
        && let Some(sql) = gpu_work_events::view_sql_for(&gpu_work)
    {
        bytes = add_usize(bytes, sql.len());
    }
    let nvtx_tree = artifact_root.join("nvtx-tree.parquet");
    if nvtx_tree.is_file()
        && let Some(sql) = nvtx_tree::view_sql_for(&nvtx_tree)
    {
        bytes = add_usize(bytes, sql.len());
    }
    bytes
}

fn parquet_view_sql(stem: &str, path: &Path) -> Option<String> {
    let path_lit = path.to_str()?.replace('\'', "''");
    let table_ident = quote_sql_identifier(stem);
    Some(format!(
        "CREATE OR REPLACE VIEW nsight.{table_ident} AS \
         SELECT (file_row_number + 1) AS rowid, * \
         FROM read_parquet('{path_lit}', file_row_number = true)"
    ))
}

fn add_path_bytes(bytes: u64, path: &Path) -> u64 {
    add_usize(bytes, path.as_os_str().as_encoded_bytes().len())
}

fn add_usize(bytes: u64, additional: usize) -> u64 {
    bytes.saturating_add(u64::try_from(additional).unwrap_or(u64::MAX))
}

fn hash_path(hash: &mut u64, label: &str, path: &Path) -> std::io::Result<()> {
    hash_bytes(hash, label.as_bytes());
    match fs::metadata(path) {
        Ok(metadata) => {
            hash_bytes(hash, b"present");
            hash_u64(hash, metadata.len());
            if let Ok(modified) = metadata.modified()
                && let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH)
            {
                hash_u64(hash, duration.as_secs());
                hash_u64(hash, u64::from(duration.subsec_nanos()));
            }
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            hash_bytes(hash, b"missing");
            Ok(())
        }
        Err(error) => Err(error),
    }
}

fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn hash_u64(hash: &mut u64, value: u64) {
    hash_bytes(hash, &value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    #[test]
    fn generated_parquetdir_alias_uses_the_report_session_identity() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let report = directory.path().join("trace.nsys-rep");
        fs::write(&report, b"report")?;
        let alias = nsys_rep::pqtdir_path_for(&report);
        fs::create_dir_all(&alias)?;
        fs::write(alias.join("META_DATA_EXPORT.parquet"), b"parquet")?;

        let report_identity = resident_trace_identity(&report)?;
        let alias_identity = resident_trace_identity(&alias)?;

        assert_eq!(
            alias_identity.canonical_source_path,
            report_identity.canonical_source_path
        );
        assert_eq!(alias_identity.freshness_key, report_identity.freshness_key);
        Ok(())
    }

    #[test]
    fn replacing_a_retained_sidecar_changes_freshness() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let trace = directory.path().join("trace_pqtdir");
        fs::create_dir(&trace)?;
        let artifact_root = artifact_dir_for(&trace);
        fs::create_dir(&artifact_root)?;
        let sidecar = artifact_root.join("meta.bin");
        fs::write(&sidecar, b"first")?;
        let before = resident_trace_identity(&trace)?;

        fs::write(&sidecar, b"replacement")?;
        let after = resident_trace_identity(&trace)?;

        assert_ne!(after.freshness_key, before.freshness_key);
        assert_eq!(
            after.resident_memory_estimate_bytes, before.resident_memory_estimate_bytes,
            "persistent sidecar payload bytes are not daemon-resident state"
        );
        Ok(())
    }
}
