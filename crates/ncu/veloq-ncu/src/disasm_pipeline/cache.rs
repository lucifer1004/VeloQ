//! On-disk cache for per-cubin correlated disassembly.
//!
//! Each cubin's correlated payload lands at
//! `<report>.veloq/disasm/<sha>.correlated.json` alongside the raw
//! `<sha>.cubin` bytes. Format is plain JSON (one `CacheFile` wrapper
//! around a [`CorrelatedEntry`]); `schema` field at the top gates
//! version invalidation.
//!
//! JSON-on-disk (rather than the bincode SidecarCache the NSys crate
//! uses) is intentional: each correlated payload is small (≤ a few
//! hundred KB), and human-readable JSON makes debugging
//! nvdisasm-output edge cases immediate.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{NcuSourceError, NcuSourceResult};

use super::types::{CACHE_SCHEMA, CorrelatedEntry, clone_entry};

#[derive(Debug, Serialize, Deserialize)]
struct CacheFile {
    schema: u32,
    entry: CorrelatedEntry,
}

/// Compute the hex SHA-256 of `bytes`. Stable across runs and
/// platforms — used to key both the cubin sidecar file and the
/// disasm cache.
pub fn cubin_sha(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

struct TempArtifact(PathBuf);

impl Drop for TempArtifact {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn write_atomic(
    path: &Path,
    bytes: &[u8],
    write_error: impl Fn(&Path, std::io::Error) -> NcuSourceError,
    publish_error: impl Fn(&Path, std::io::Error) -> NcuSourceError,
) -> NcuSourceResult<()> {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut temp_name = path.as_os_str().to_owned();
    temp_name.push(format!(".tmp-{}-{seq}", std::process::id()));
    let temp = TempArtifact(PathBuf::from(temp_name));
    fs::write(&temp.0, bytes).map_err(|source| write_error(&temp.0, source))?;
    // Unix rename atomically replaces the destination. Windows rename does
    // not, so remove only there; readers may briefly observe a cache miss but
    // never a partially written artifact.
    #[cfg(windows)]
    if path.exists() {
        fs::remove_file(path).map_err(|source| publish_error(path, source))?;
    }
    fs::rename(&temp.0, path).map_err(|source| publish_error(path, source))?;
    Ok(())
}

/// Return the sidecar directory for a given trace path:
/// `<report>.veloq/disasm/`.
pub fn sidecar_dir(trace_path: &Path) -> PathBuf {
    veloq_core::artifact_dir_for(trace_path).join("disasm")
}

/// Materialize the cubin bytes to `<report>.veloq/disasm/<sha>.cubin`,
/// creating the sidecar directory if missing. Returns the cubin
/// path. Idempotent: reuses a verified file and atomically replaces a
/// missing or corrupted one.
pub fn extract_and_cache_cubin(
    trace_path: &Path,
    sha: &str,
    bytes: &[u8],
) -> NcuSourceResult<PathBuf> {
    let dir = sidecar_dir(trace_path);
    fs::create_dir_all(&dir)
        .map_err(|source| NcuSourceError::disasm_sidecar_dir_create(dir.display(), source))?;
    let cubin_path = dir.join(format!("{sha}.cubin"));
    let write_required = match fs::read(&cubin_path) {
        Ok(cached) => cubin_sha(&cached) != sha,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => true,
        Err(source) => {
            return Err(NcuSourceError::disasm_cubin_read(
                cubin_path.display(),
                source,
            ));
        }
    };
    if write_required {
        write_atomic(
            &cubin_path,
            bytes,
            |path, source| NcuSourceError::disasm_cubin_write(path.display(), source),
            |path, source| NcuSourceError::disasm_cubin_publish(path.display(), source),
        )?;
    }
    Ok(cubin_path)
}

/// Path of the correlated cache file for a given cubin SHA.
pub fn correlated_cache_path(trace_path: &Path, sha: &str) -> PathBuf {
    sidecar_dir(trace_path).join(format!("{sha}.correlated.json"))
}

/// Load a previously-cached `CorrelatedEntry` if `path` exists,
/// carries the current [`CACHE_SCHEMA`], and the stored cubin SHA and
/// `instruction_stride` match the caller's request. Returns
/// `Ok(None)` for missing, schema-mismatched, or stride-mismatched
/// files — all three force a fresh acquire.
pub fn load_cached(
    path: &Path,
    expected_cubin_sha: &str,
    instruction_stride: u64,
) -> NcuSourceResult<Option<CorrelatedEntry>> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(path)
        .map_err(|source| NcuSourceError::disasm_cache_read(path.display(), source))?;
    let raw: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(raw) => raw,
        Err(source) => {
            log::info!("correlated disasm cache could not be decoded: {source}; needs rebuild");
            return Ok(None);
        }
    };
    let schema = raw.get("schema").and_then(serde_json::Value::as_u64);
    if schema != Some(u64::from(CACHE_SCHEMA)) {
        return Ok(None);
    }
    let file: CacheFile = match serde_json::from_value(raw) {
        Ok(file) => file,
        Err(source) => {
            log::info!("correlated disasm cache shape is invalid: {source}; needs rebuild");
            return Ok(None);
        }
    };
    if file.entry.cubin_sha != expected_cubin_sha
        || file.entry.instruction_stride != instruction_stride
    {
        return Ok(None);
    }
    Ok(Some(file.entry))
}

/// Serialize `entry` and write it to the cache file at `path`.
pub fn write_cache(path: &Path, entry: &CorrelatedEntry) -> NcuSourceResult<()> {
    let file = CacheFile {
        schema: CACHE_SCHEMA,
        entry: clone_entry(entry),
    };
    let bytes = serde_json::to_vec_pretty(&file)
        .map_err(|source| NcuSourceError::disasm_cache_encode(path.display(), source))?;
    write_atomic(
        path,
        &bytes,
        |path, source| NcuSourceError::disasm_cache_write(path.display(), source),
        |path, source| NcuSourceError::disasm_cache_publish(path.display(), source),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disasm_pipeline::types::{KernelDisasm, SassInstruction};
    use anyhow::Result;

    // SASS instruction strides (bytes) used as test values: production
    // disasm uses 16; `OTHER_STRIDE` (8) exercises the mismatch path.
    const STRIDE: u64 = 16;
    const OTHER_STRIDE: u64 = 8;

    fn empty_entry() -> CorrelatedEntry {
        CorrelatedEntry {
            cubin_sha: String::new(),
            sm: None,
            instruction_stride: STRIDE,
            source_lineinfo_present: false,
            kernels: Vec::new(),
            ptx_lines: Vec::new(),
            source_index: Vec::new(),
            warnings: Vec::new(),
        }
    }

    #[test]
    fn cubin_sha_is_stable_hex() {
        let a = cubin_sha(b"hello");
        let b = cubin_sha(b"hello");
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
    }

    #[test]
    fn sidecar_dir_appends_suffix_to_trace_filename() {
        let dir = sidecar_dir(Path::new("/tmp/foo/report.ncu-rep"));
        assert_eq!(dir, Path::new("/tmp/foo/report.ncu-rep.veloq/disasm"));
    }

    #[test]
    fn extract_and_cache_cubin_round_trips() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "veloq-ncu-cache-test-{}.ncu-rep",
            std::process::id()
        ));
        let bytes = b"\x7fELF...fake cubin";
        let sha = cubin_sha(bytes);
        let cubin = extract_and_cache_cubin(&tmp, &sha, bytes)?;
        assert!(cubin.exists());
        let read_back = fs::read(&cubin)?;
        assert_eq!(read_back, bytes);
        fs::write(&cubin, b"truncated")?;
        let cubin2 = extract_and_cache_cubin(&tmp, &sha, bytes)?;
        assert_eq!(cubin, cubin2);
        assert_eq!(fs::read(&cubin2)?, bytes);
        fs::remove_dir_all(sidecar_dir(&tmp))?;
        Ok(())
    }

    #[test]
    fn load_cached_returns_none_when_missing() -> Result<()> {
        let path = std::env::temp_dir().join("veloq-ncu-nonexistent.correlated.json");
        let _ = fs::remove_file(&path);
        assert!(load_cached(&path, "unused", STRIDE)?.is_none());
        Ok(())
    }

    #[test]
    fn load_cached_round_trips_via_write_cache() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "veloq-ncu-cache-test-{}.correlated.json",
            std::process::id()
        ));
        let mut entry = empty_entry();
        entry.cubin_sha = "abc123".into();
        entry.sm = Some("sm_90".into());
        entry.kernels.push(KernelDisasm {
            key: "kernel-test".to_string(),
            function_name: "kernel".into(),
            start: 0,
            length: 32,
            instructions: vec![SassInstruction {
                address: 0,
                opcode: "NOP".into(),
                operands: String::new(),
                predicate: None,
                control_flow: false,
                source: None,
            }],
        });
        entry.warnings.push("nvdisasm warning : foo".into());
        write_cache(&path, &entry)?;
        let loaded = load_cached(&path, "abc123", STRIDE)?
            .ok_or_else(|| anyhow::anyhow!("just wrote it"))?;
        assert_eq!(loaded.cubin_sha, "abc123");
        assert_eq!(loaded.kernels.len(), 1);
        assert_eq!(loaded.warnings.len(), 1);
        assert!(load_cached(&path, "different", STRIDE)?.is_none());
        fs::remove_file(&path)?;
        Ok(())
    }

    #[test]
    fn load_cached_rebuilds_invalid_json() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "veloq-ncu-invalid-cache-test-{}.correlated.json",
            std::process::id()
        ));
        fs::write(&path, b"{partial")?;
        assert!(load_cached(&path, "abc", STRIDE)?.is_none());
        fs::remove_file(&path)?;
        Ok(())
    }

    #[test]
    fn load_cached_rejects_schema_mismatch() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "veloq-ncu-schema-test-{}.correlated.json",
            std::process::id()
        ));
        let raw = serde_json::json!({
            "schema": CACHE_SCHEMA + 99,
            "entry": {
                "cubin_sha": "xx", "sm": null,
                "instruction_stride": STRIDE,
                "source_lineinfo_present": false,
                "kernels": [], "ptx_lines": [], "source_index": [],
                "warnings": [],
            },
        });
        fs::write(&path, serde_json::to_vec(&raw)?)?;
        assert!(load_cached(&path, "xx", STRIDE)?.is_none());
        fs::remove_file(&path)?;
        Ok(())
    }

    #[test]
    fn load_cached_rejects_instruction_stride_mismatch() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "veloq-ncu-stride-test-{}.correlated.json",
            std::process::id()
        ));
        let mut entry = empty_entry();
        entry.cubin_sha = "abc".into();
        entry.instruction_stride = STRIDE;
        write_cache(&path, &entry)?;
        assert!(load_cached(&path, "abc", OTHER_STRIDE)?.is_none());
        fs::remove_file(&path)?;
        Ok(())
    }
}
