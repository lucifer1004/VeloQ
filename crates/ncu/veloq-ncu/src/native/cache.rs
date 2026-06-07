//! ncu_report-native ingest + cache.
//!
//! The sole NCU ingest path. Mirrors the NSys export-once + flock
//! pattern (`crates/nsys/veloq-nsys-data/src/nsys_rep.rs`):
//! lock-free fast path → flock → re-check → build.
//!
//! ## Freshness by content hash (not mtime/ctime)
//!
//! The cache is valid iff a sibling `ncu-native.sha256` marker records the
//! sha256 of the current `.ncu-rep`. This diverges deliberately from
//! `nsys_rep`'s ctime ordering: git checkout resets mtime/ctime, which
//! would invalidate a *committed* golden sidecar and force a rebuild —
//! and a rebuild requires NCU, breaking the NCU-free CI premise. A
//! content hash is checkout-stable, so committed goldens hit the cache.
//!
//! ## NCU dependency
//!
//! Building the cache runs the bundled Python helper against NVIDIA's
//! `ncu_report` API — NCU must be installed *at build time only*. A
//! content-hash match serves the cache with no NCU. A mismatch (or a
//! missing cache) with NCU absent is a structured error: veloq cannot
//! ingest a new/changed `.ncu-rep` without NCU.
//!
//! ## Discovery (cross-platform)
//!
//! [`locate_ncu_report`] returns the directory holding `ncu_report.py`
//! (set as `PYTHONPATH`); the interpreter is resolved separately in
//! [`run_helper`]. Precedence: the `VELOQ_NCU_REPORT_DIR` override, then
//! per-OS install roots (Linux `/usr/local`, `/opt/nvidia`, `/opt/cuda`;
//! macOS the `NVIDIA Nsight Compute*.app` bundle under `/Applications`;
//! Windows `Nsight Compute *` under the `Program Files` roots), newest by
//! natural version order, then `ncu` on `PATH` (no shell). The interpreter
//! is `VELOQ_PYTHON`, else `python3`/`python` on unix or
//! `python`/`python3`/`py -3` on Windows.
//!
//! ## Committed-sidecar mode (report absent)
//!
//! When the source `.ncu-rep` is *absent* but a committed sidecar
//! exists, the sidecar is authoritative — the export-once model taken
//! to its conclusion: ship the leak-free `<report>.veloq/` sidecar, not
//! the proprietary report (which embeds the capturing host's hostname /
//! NIC IPs and isn't always committable). Freshness isn't content-checked
//! in this mode (there's nothing to hash); a genuinely missing trace
//! with no sidecar still errors.

use sha2::{Digest, Sha256};
use std::ffi::OsString;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::error::{NcuSourceError, NcuSourceResult};

use super::{NATIVE_SCHEMA, NativeSidecar};

/// The export helper, bundled into the binary at compile time so veloq
/// ships self-contained (no separate script to distribute). Materialized
/// to a temp file and run via a discovered Python interpreter when a
/// (re)build is needed.
const HELPER_PY: &str = include_str!("../../scripts/ncu_export.py");

/// Override: a directory that directly contains
/// `ncu_report.py`. Set as `PYTHONPATH`; skips all platform discovery —
/// for containers, non-standard installs, and CI.
const ENV_NCU_REPORT_DIR: &str = "VELOQ_NCU_REPORT_DIR";

/// Override: the Python interpreter used to run the helper.
/// Set-but-unusable is a hard error (the candidate list is exactly this
/// one program — no silent fallthrough to the probe ladder).
const ENV_PYTHON: &str = "VELOQ_PYTHON";

const CACHE_NAME: &str = "ncu-native.json.gz";
const MARKER_NAME: &str = "ncu-native.sha256";
const LOCK_NAME: &str = "ncu-native.lock";

fn cache_path_for(report: &Path) -> PathBuf {
    veloq_core::artifact_dir_for(report).join(CACHE_NAME)
}

/// Public accessor for the native sidecar path, so verbs can echo
/// `auxiliary.meta_cache_path` without recomputing the layout.
pub fn path_for(report: &Path) -> PathBuf {
    cache_path_for(report)
}
fn marker_path_for(report: &Path) -> PathBuf {
    veloq_core::artifact_dir_for(report).join(MARKER_NAME)
}
fn lock_path_for(report: &Path) -> PathBuf {
    veloq_core::artifact_dir_for(report).join(LOCK_NAME)
}

/// Build or load the native sidecar. Content-hash fast path; otherwise
/// runs the helper under an export lock (requires NCU). See module docs.
pub fn build_or_load(report: &Path) -> NcuSourceResult<NativeSidecar> {
    let cache = cache_path_for(report);
    if !report.exists() {
        // Committed-sidecar mode: see module docs.
        if cache.is_file() {
            let sc = read_gz_sidecar(&cache)?;
            if sc.schema != NATIVE_SCHEMA {
                return Err(NcuSourceError::native_sidecar_schema_mismatch(
                    &cache,
                    sc.schema,
                    NATIVE_SCHEMA,
                ));
            }
            return Ok(sc);
        }
        return Err(NcuSourceError::trace_not_found(report));
    }
    let want = file_sha256(report)?;
    let marker = marker_path_for(report);

    if let Some(sc) = load_if_fresh(&cache, &marker, &want)? {
        return Ok(sc);
    }

    // Serialize concurrent (re)builds.
    let lock_path = lock_path_for(report);
    if let Some(parent) = lock_path.parent() {
        fs::create_dir_all(parent).map_err(|source| {
            NcuSourceError::native_artifact_dir_create(parent.display(), source)
        })?;
    }
    let lock_file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|source| {
            NcuSourceError::native_export_lockfile_open(lock_path.display(), source)
        })?;
    lock_file.lock().map_err(|source| {
        NcuSourceError::native_export_lock_acquire(lock_path.display(), source)
    })?;

    // A concurrent process may have built it while we waited.
    if let Some(sc) = load_if_fresh(&cache, &marker, &want)? {
        return Ok(sc);
    }

    // (Re)build requires NCU. Absent → structured error (don't serve a
    // stale cache for a changed report).
    let pythonpath = locate_ncu_report()
        .map_err(|source| NcuSourceError::native_ingest_unavailable(report, source))?;

    let payload = run_helper(report, &pythonpath)?;
    let sidecar: NativeSidecar =
        serde_json::from_str(&payload).map_err(NcuSourceError::native_helper_output_deserialize)?;
    if sidecar.schema != NATIVE_SCHEMA {
        return Err(NcuSourceError::native_helper_schema_mismatch(
            sidecar.schema,
            NATIVE_SCHEMA,
        ));
    }
    write_cache(&cache, &marker, &payload, &want)?;
    Ok(sidecar)
}

/// Load + validate the cache against the wanted content hash. `Ok(None)`
/// when the cache or marker is missing or the hash differs.
fn load_if_fresh(
    cache: &Path,
    marker: &Path,
    want: &str,
) -> NcuSourceResult<Option<NativeSidecar>> {
    if !cache.is_file() || !marker.is_file() {
        return Ok(None);
    }
    let got = fs::read_to_string(marker)
        .map_err(|source| NcuSourceError::native_cache_marker_read(marker.display(), source))?;
    if got.trim() != want {
        log::info!("ncu report changed since native sidecar was written; needs rebuild");
        return Ok(None);
    }
    let sc = read_gz_sidecar(cache)?;
    if sc.schema != NATIVE_SCHEMA {
        log::info!(
            "native sidecar schema {:?} != {NATIVE_SCHEMA}; needs rebuild",
            sc.schema
        );
        return Ok(None);
    }
    Ok(Some(sc))
}

/// Read + gunzip + deserialize a committed/cached native sidecar. Public
/// so tests and NCU-free callers can load a known-good golden directly.
pub fn read_gz_sidecar(path: &Path) -> NcuSourceResult<NativeSidecar> {
    let bytes = fs::read(path)
        .map_err(|source| NcuSourceError::native_sidecar_read(path.display(), source))?;
    let mut s = String::new();
    flate2::read::GzDecoder::new(bytes.as_slice())
        .read_to_string(&mut s)
        .map_err(|source| NcuSourceError::native_sidecar_gunzip(path.display(), source))?;
    serde_json::from_str(&s)
        .map_err(|source| NcuSourceError::native_sidecar_deserialize(path.display(), source))
}

fn write_cache(
    cache: &Path,
    marker: &Path,
    payload_json: &str,
    sha_hex: &str,
) -> NcuSourceResult<()> {
    if let Some(parent) = cache.parent() {
        fs::create_dir_all(parent).map_err(|source| {
            NcuSourceError::native_artifact_dir_create(parent.display(), source)
        })?;
    }
    // gzip the payload (mtime 0 → byte-reproducible, matching `gzip -n`).
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    enc.write_all(payload_json.as_bytes())
        .map_err(NcuSourceError::native_sidecar_gzip_write)?;
    let gz = enc
        .finish()
        .map_err(NcuSourceError::native_sidecar_gzip_finish)?;
    write_atomic(cache, &gz)?;
    write_atomic(marker, format!("{sha_hex}\n").as_bytes())?;
    log::info!(
        "wrote native sidecar: {} bytes -> {}",
        gz.len(),
        cache.display()
    );
    Ok(())
}

fn write_atomic(path: &Path, bytes: &[u8]) -> NcuSourceResult<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp_path = PathBuf::from(tmp);
    fs::write(&tmp_path, bytes)
        .map_err(|source| NcuSourceError::native_atomic_write(tmp_path.display(), source))?;
    fs::rename(&tmp_path, path).map_err(|source| {
        NcuSourceError::native_atomic_rename(tmp_path.display(), path.display(), source)
    })?;
    Ok(())
}

fn file_sha256(path: &Path) -> NcuSourceResult<String> {
    // `.ncu-rep` fixtures are a few MB; read whole rather than a buffered
    // loop so we avoid a slice index (clippy::indexing_slicing is denied).
    let bytes = fs::read(path)
        .map_err(|source| NcuSourceError::native_report_read(path.display(), source))?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

/// RAII cleanup for the materialized helper script — removes the temp
/// file on any exit path (including early `?` returns).
struct TempScript(PathBuf);
impl Drop for TempScript {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// Run the bundled helper against `report` with `PYTHONPATH` pointed at
/// the located `extras/python`. Returns the helper's stdout (the native
/// sidecar JSON). A non-zero exit surfaces the helper's stderr.
fn run_helper(report: &Path, pythonpath: &Path) -> NcuSourceResult<String> {
    // Unique per call (pid + process-global counter) so concurrent
    // in-process callers can't clobber each other's script; RAII removes
    // it even when `?` returns early.
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = TempScript(
        std::env::temp_dir().join(format!("veloq-ncu-export-{}-{seq}.py", std::process::id())),
    );
    fs::write(&tmp.0, HELPER_PY)
        .map_err(|source| NcuSourceError::native_helper_materialize(tmp.0.display(), source))?;

    // Try interpreter candidates in order. Advance to the next ONLY when
    // the interpreter binary itself is not found (spawn `NotFound`); a
    // helper that ran and exited non-zero surfaces verbatim.
    let candidates = python_candidates(std::env::var_os(ENV_PYTHON));
    let mut not_found = Vec::new();
    for (prog, pre_args) in &candidates {
        let out = Command::new(prog)
            .args(pre_args)
            .arg(&tmp.0)
            .arg(report)
            .env("PYTHONPATH", pythonpath)
            .output();
        match out {
            Ok(out) if out.status.success() => {
                return String::from_utf8(out.stdout)
                    .map_err(NcuSourceError::native_helper_stdout_utf8);
            }
            Ok(out) => {
                return Err(NcuSourceError::native_helper_failed(
                    prog,
                    out.status.to_string(),
                    String::from_utf8_lossy(&out.stderr).trim().to_string(),
                ));
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                not_found.push(prog.clone());
            }
            Err(e) => {
                return Err(NcuSourceError::native_helper_spawn(prog.clone(), e));
            }
        }
    }
    Err(NcuSourceError::native_python_missing(not_found.join(", ")))
}

/// Interpreter candidates in priority order: `(program, prefix_args)`.
/// `VELOQ_PYTHON` short-circuits to a single explicit choice (so a
/// set-but-missing override is a hard error, not a silent fallthrough).
fn python_candidates(override_py: Option<OsString>) -> Vec<(String, Vec<String>)> {
    if let Some(p) = override_py {
        return vec![(p.to_string_lossy().into_owned(), Vec::new())];
    }
    if cfg!(windows) {
        // Windows ships `python` / the `py` launcher; `python3` is rare.
        vec![
            ("python".to_string(), Vec::new()),
            ("python3".to_string(), Vec::new()),
            ("py".to_string(), vec!["-3".to_string()]),
        ]
    } else {
        vec![
            ("python3".to_string(), Vec::new()),
            ("python".to_string(), Vec::new()),
        ]
    }
}

/// Locate the directory that contains the `ncu_report` Python module, to
/// set as `PYTHONPATH`. Uniform return contract across platforms: always a
/// *directory* placed on `PYTHONPATH` (the interpreter is resolved
/// separately in [`run_helper`]). Precedence:
/// `VELOQ_NCU_REPORT_DIR` override → per-platform install roots → `ncu` on
/// `PATH`. Extends pre-deletion gate 4 to macOS/Windows.
pub fn locate_ncu_report() -> NcuSourceResult<PathBuf> {
    locate_ncu_report_impl(std::env::var_os(ENV_NCU_REPORT_DIR))
}

/// Discovery core, split out so the override path is unit-testable without
/// mutating process env.
fn locate_ncu_report_impl(override_dir: Option<OsString>) -> NcuSourceResult<PathBuf> {
    if let Some(dir) = override_dir {
        let p = PathBuf::from(dir);
        if p.join("ncu_report.py").is_file() {
            return Ok(p);
        }
        return Err(NcuSourceError::native_ncu_report_override_invalid(&p));
    }
    for (base, pattern) in platform_search_roots() {
        if let Some(p) = newest_glob_with_module(&base, &pattern) {
            return Ok(p);
        }
    }
    if let Some(p) = ncu_on_path_module_dir() {
        return Ok(p);
    }
    Err(NcuSourceError::native_ncu_report_module_missing(
        discovery_failure_message(),
    ))
}

/// `(base, glob-pattern)` pairs to search, per host OS. The
/// glob pattern is always written with `/` separators (the walker and
/// `Path::join` accept them on Windows too).
fn platform_search_roots() -> Vec<(PathBuf, String)> {
    #[cfg(target_os = "macos")]
    let (bases, patterns): (Vec<PathBuf>, &[&str]) = (
        vec![PathBuf::from("/Applications")],
        &[
            "NVIDIA Nsight Compute*.app/Contents/MacOS/python",
            "NVIDIA Nsight Compute*/extras/python",
        ],
    );
    #[cfg(target_os = "windows")]
    let (bases, patterns): (Vec<PathBuf>, &[&str]) = (
        windows_program_files()
            .into_iter()
            .map(|p| p.join("NVIDIA Corporation"))
            .collect(),
        &[
            "Nsight Compute */extras/python",
            "Nsight Compute*/extras/python",
        ],
    );
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let (bases, patterns): (Vec<PathBuf>, &[&str]) = (
        vec![
            PathBuf::from("/usr/local"),
            PathBuf::from("/opt/nvidia"),
            PathBuf::from("/opt/cuda"),
        ],
        &[
            "cuda-*/nsight-compute-*/extras/python",
            "nsight-compute-*/extras/python",
            "nsight-compute/*/extras/python",
            "nsight-compute/extras/python",
        ],
    );
    let mut out = Vec::new();
    for base in bases {
        for pat in patterns {
            out.push((base.clone(), (*pat).to_string()));
        }
    }
    out
}

/// Windows `Program Files` roots from the environment, with literal
/// fallbacks when the vars are unset.
#[cfg(target_os = "windows")]
fn windows_program_files() -> Vec<PathBuf> {
    let mut v = Vec::new();
    for var in ["ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"] {
        if let Some(val) = std::env::var_os(var) {
            v.push(PathBuf::from(val));
        }
    }
    if v.is_empty() {
        v.push(PathBuf::from(r"C:\Program Files"));
        v.push(PathBuf::from(r"C:\Program Files (x86)"));
    }
    v
}

/// Human-readable, platform-aware discovery failure: names the roots
/// actually searched plus the override.
fn discovery_failure_message() -> String {
    let roots: Vec<String> = platform_search_roots()
        .iter()
        .map(|(b, p)| format!("{}/{p}", b.display()))
        .collect();
    format!(
        "could not locate the ncu_report Python module. Set {ENV_NCU_REPORT_DIR} to the directory \
         containing ncu_report.py, or install Nsight Compute. Searched: [{}]; and `ncu` on PATH.",
        roots.join(", ")
    )
}

fn newest_glob_with_module(base: &Path, pattern: &str) -> Option<PathBuf> {
    // Minimal two-segment glob walker (no glob crate dep): split the
    // pattern on '/', expand each '*'-bearing segment against the dir.
    let mut frontier = vec![base.to_path_buf()];
    for seg in pattern.split('/') {
        let mut next = Vec::new();
        for dir in &frontier {
            if seg.contains('*') {
                if let Ok(rd) = fs::read_dir(dir) {
                    let mut matches: Vec<PathBuf> = rd
                        .flatten()
                        .map(|e| e.path())
                        .filter(|p| p.is_dir() && glob_match(seg, &name_of(p)))
                        .collect();
                    // Newest version dir first, by numeric/natural order:
                    // `2024.3.10` must beat `2024.3.2`,
                    // which a plain lexical sort gets wrong.
                    matches.sort_by(|a, b| {
                        version_key(&name_of(b))
                            .cmp(&version_key(&name_of(a)))
                            .then_with(|| name_of(b).cmp(&name_of(a)))
                    });
                    next.extend(matches);
                }
            } else {
                let cand = dir.join(seg);
                if cand.is_dir() {
                    next.push(cand);
                }
            }
        }
        frontier = next;
    }
    frontier
        .into_iter()
        .find(|p| p.join("ncu_report.py").is_file())
}

fn name_of(p: &Path) -> String {
    p.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string()
}

/// Single-`*` glob match (sufficient for the `prefix-*suffix` patterns
/// used here; the directory names never contain more than one wildcard).
fn glob_match(pattern: &str, name: &str) -> bool {
    match pattern.split_once('*') {
        None => pattern == name,
        Some((pre, suf)) => {
            name.len() >= pre.len() + suf.len() && name.starts_with(pre) && name.ends_with(suf)
        }
    }
}

/// Natural-order key: the sequence of integer runs in `name`, compared
/// lexicographically as numbers, so `...-2024.3.10` sorts above
/// `...-2024.3.2`. Non-digit separators delimit runs.
fn version_key(name: &str) -> Vec<u64> {
    let mut key = Vec::new();
    let mut cur = String::new();
    for ch in name.chars() {
        if ch.is_ascii_digit() {
            cur.push(ch);
        } else if !cur.is_empty() {
            key.push(cur.parse::<u64>().unwrap_or(0));
            cur.clear();
        }
    }
    if !cur.is_empty() {
        key.push(cur.parse::<u64>().unwrap_or(0));
    }
    key
}

/// Portable replacement for the old `sh -c "command -v ncu"` fallback:
/// search `PATH` for the `ncu` binary (no shell — works on
/// stock Windows), then walk up checking the layouts that hold the module
/// (`extras/python`, or the macOS bundle's `Contents/MacOS/python`).
fn ncu_on_path_module_dir() -> Option<PathBuf> {
    let exe = if cfg!(windows) { "ncu.exe" } else { "ncu" };
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let bin = dir.join(exe);
        if !bin.is_file() {
            continue;
        }
        let mut up = bin.parent();
        while let Some(d) = up {
            for sub in ["extras/python", "python", "../python"] {
                let cand = d.join(sub);
                if cand.join("ncu_report.py").is_file() {
                    return Some(cand);
                }
            }
            up = d.parent();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use veloq_core::VeloqDiagnostic;

    fn fixture() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/source_metric_basic.ncu-rep")
    }

    fn unique_temp_path(label: &str) -> PathBuf {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("veloq-ncu-{label}-{}-{seq}", std::process::id()))
    }

    fn write_test_sidecar(cache: &Path, schema: &str) -> Result<()> {
        use std::io::Write;
        let parent = cache
            .parent()
            .ok_or_else(|| anyhow::anyhow!("cache path has no parent"))?;
        fs::create_dir_all(parent)?;
        let payload = format!(
            r#"{{"schema":"{schema}","ncu_version":"test","session":{{"versions":[]}},"launches":[]}}"#
        );
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(payload.as_bytes())?;
        let gz = enc.finish()?;
        fs::write(cache, &gz)?;
        Ok(())
    }

    fn ncu_error_code(err: &NcuSourceError) -> &'static str {
        err.code().as_str()
    }

    /// Committed-sidecar mode:
    /// a `<report>.veloq/ncu-native.json.gz` with no source report present
    /// is served as authoritative, with no NCU and no freshness marker.
    /// Hermetic — writes a sidecar to a temp artifact dir whose `.ncu-rep`
    /// never exists, so it exercises the report-absent branch without
    /// depending on the committed tree's layout.
    #[test]
    fn build_or_load_serves_committed_sidecar_when_report_absent() -> Result<()> {
        let tmp = unique_temp_path("absent-report").with_extension("ncu-rep");
        let cache = cache_path_for(&tmp);
        let parent = cache
            .parent()
            .ok_or_else(|| anyhow::anyhow!("cache path has no parent"))?;
        let _ = fs::remove_dir_all(parent);
        write_test_sidecar(&cache, NATIVE_SCHEMA)?;

        assert!(
            !tmp.exists(),
            "the source report must be absent for this branch"
        );
        let sc = build_or_load(&tmp)?;
        assert_eq!(sc.schema, NATIVE_SCHEMA);

        fs::remove_dir_all(parent).ok();
        Ok(())
    }

    #[test]
    fn committed_sidecar_schema_mismatch_is_typed() -> Result<()> {
        let tmp = unique_temp_path("schema-mismatch").with_extension("ncu-rep");
        let cache = cache_path_for(&tmp);
        let parent = cache
            .parent()
            .ok_or_else(|| anyhow::anyhow!("cache path has no parent"))?;
        let _ = fs::remove_dir_all(parent);
        write_test_sidecar(&cache, "older-schema")?;

        assert!(
            !tmp.exists(),
            "the source report must be absent for committed-sidecar validation"
        );
        let err = build_or_load(&tmp)
            .err()
            .ok_or_else(|| anyhow::anyhow!("schema mismatch should error"))?;
        assert_eq!(
            ncu_error_code(&err),
            "ncu.input.native-sidecar-schema-mismatch"
        );

        fs::remove_dir_all(parent).ok();
        Ok(())
    }

    /// The committed `source_metric_basic` sidecar loads and carries the
    /// populated capture (≥2 launches) with no NCU — the real end-to-end
    /// committed-sidecar path the verb goldens ride on.
    #[test]
    fn committed_source_metric_sidecar_loads_without_ncu() -> Result<()> {
        let sc = build_or_load(&fixture())?;
        assert_eq!(sc.schema, NATIVE_SCHEMA);
        assert!(sc.launches.len() >= 2);
        Ok(())
    }

    #[test]
    fn glob_match_single_wildcard() {
        assert!(glob_match("nsight-compute-*", "nsight-compute-2026.1.1"));
        assert!(glob_match("cuda-*", "cuda-13.2"));
        assert!(!glob_match("nsight-compute-*", "nsight-systems-2026"));
        assert!(glob_match("extras", "extras"));
    }

    /// Version dirs order numerically, not lexically —
    /// `2024.3.10` must beat `2024.3.2` (the latent bug this fixes).
    #[test]
    fn version_key_orders_numerically() {
        assert!(
            version_key("nsight-compute-2024.3.10") > version_key("nsight-compute-2024.3.2"),
            "10 must sort above 2 within the same minor"
        );
        assert!(version_key("2024.3.2") < version_key("2025.1.0"));
        assert!(version_key("nsight-compute-2026.1.1") > version_key("nsight-compute-2026.1.0"));
    }

    /// The `VELOQ_NCU_REPORT_DIR` override returns the dir
    /// when it holds `ncu_report.py`, and errors when it does not — no
    /// silent fallthrough. Tested via the impl seam so no process-env
    /// mutation (unsafe in edition 2024) is needed.
    #[test]
    fn env_override_points_at_module_dir() -> Result<()> {
        let pid = std::process::id();
        let good = std::env::temp_dir().join(format!("veloq-ncu-report-ok-{pid}"));
        fs::create_dir_all(&good)?;
        fs::write(good.join("ncu_report.py"), b"# stub\n")?;
        let got = locate_ncu_report_impl(Some(good.clone().into_os_string()))?;
        assert_eq!(got, good);

        let empty = std::env::temp_dir().join(format!("veloq-ncu-report-empty-{pid}"));
        fs::create_dir_all(&empty)?;
        assert!(
            locate_ncu_report_impl(Some(empty.clone().into_os_string())).is_err(),
            "a dir without ncu_report.py must error, not fall through"
        );

        let _ = fs::remove_dir_all(&good);
        let _ = fs::remove_dir_all(&empty);
        Ok(())
    }

    #[test]
    fn env_override_error_is_typed() -> Result<()> {
        let empty = unique_temp_path("ncu-report-empty");
        let _ = fs::remove_dir_all(&empty);
        fs::create_dir_all(&empty)?;

        let err = locate_ncu_report_impl(Some(empty.clone().into_os_string()))
            .err()
            .ok_or_else(|| anyhow::anyhow!("invalid override dir should error"))?;
        assert_eq!(err.code().as_str(), "ncu.input.ncu-report-override-invalid");

        fs::remove_dir_all(&empty).ok();
        Ok(())
    }

    /// `VELOQ_PYTHON` short-circuits to one candidate; the
    /// default ladder is platform-correct.
    #[test]
    fn interpreter_candidates_respect_override_and_platform() {
        let forced = python_candidates(Some("my-python".into()));
        assert_eq!(forced.len(), 1);
        assert!(matches!(forced.first(), Some((p, args)) if p == "my-python" && args.is_empty()));

        let def = python_candidates(None);
        let names: Vec<&str> = def.iter().map(|(p, _)| p.as_str()).collect();
        assert!(!names.is_empty());
        if cfg!(windows) {
            assert_eq!(names.first(), Some(&"python"));
            assert!(names.contains(&"py"));
        } else {
            assert_eq!(names.first(), Some(&"python3"));
            assert!(names.contains(&"python"));
        }
    }

    /// The discovery-failure message names the override and is non-empty
    /// on every platform.
    #[test]
    fn discovery_failure_message_mentions_override() {
        let msg = discovery_failure_message();
        assert!(msg.contains(ENV_NCU_REPORT_DIR));
        assert!(msg.contains("ncu_report.py"));
    }
}
