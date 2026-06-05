//! Transparent `.nsys-rep` → parquetdir conversion.
//!
//! veloq requires `nsys ≥ 2024.6` (the release that introduced
//! `nsys export --type parquetdir`). The `.nsys-rep` input is exported
//! to a directory of per-table Parquet files under the report's
//! `<report>.veloq/` artifact root. The directory **is** veloq's
//! parquet cache — no separate `veloq-parquet/` sidecar is built.
//!
//! ## Layout
//!
//! ```text
//! <stem>.nsys-rep
//! <stem>.nsys-rep.veloq/
//!   parquetdir/                       # nsys-emitted, veloq-cached
//!     CUPTI_ACTIVITY_KIND_KERNEL.parquet
//!     NVTX_EVENTS.parquet
//!     ...                             # one .parquet per nsys table
//!   export.lock                       # advisory flock for export
//! ```
//!
//! ## Cache invalidation
//!
//! We follow `nsys_recipe.lib.data_reader._Loader.
//! validate_export_time`: ctime ordering of
//! `<stem>.nsys-rep.veloq/parquetdir/META_DATA_EXPORT.parquet` vs
//! the source `<stem>.nsys-rep`. This is strictly weaker than
//! the mtime+size invariant but keeps the exported parquet
//! files as the freshness source of truth.
//!
//! ## Capability check
//!
//! On the first conversion attempt per process, veloq probes
//! `nsys export --help type` for the literal substring `parquetdir`.
//! Older nsys (≤ 2024.5) lacks the format and veloq exits with an
//! actionable error asking the user to upgrade nsys.
//!
//! ## Concurrency
//!
//! Two veloq processes can race on a fresh `.nsys-rep`. The first to
//! acquire the flock at `<stem>.nsys-rep.veloq/export.lock` performs the export; the
//! second blocks, re-checks the cache, and short-circuits.

use anyhow::{Context, Result, bail};
use std::ffi::OsStr;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;
#[cfg(not(unix))]
use std::time::UNIX_EPOCH;
use veloq_core::{ARTIFACT_DIR_SUFFIX, artifact_dir_for};

/// Directory suffix matching nsys's own naming
/// (`nsys_recipe.lib.data_reader.ParquetLoader::output_suffix`).
pub const PQTDIR_SUFFIX: &str = "_pqtdir";
const PARQUETDIR_NAME: &str = "parquetdir";
const LOCK_NAME: &str = "export.lock";
const EXPORT_BASENAME: &str = "nsys-export";
const ETXTBSY: i32 = 26;
const TEXT_BUSY_RETRIES: usize = 20;

/// A "sentinel" table that any non-empty nsys export should produce.
/// Used to make the freshness check robust to traces that lack some
/// optional tables (e.g. `--trace=none --nic-metrics=lf` captures).
const SENTINEL_TABLE: &str = "META_DATA_EXPORT.parquet";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedTracePath {
    /// Canonical source identity for derived veloq sidecars.
    pub source_path: PathBuf,
    /// Directory of NSys-exported parquet tables opened by DuckDB.
    pub pqtdir_path: PathBuf,
}

/// Resolve an NSys input into source identity + parquetdir.
///
/// `.nsys-rep` inputs auto-export to `<report>.veloq/parquetdir/`.
/// Direct user-provided `_pqtdir/` inputs are source data in their own
/// right, so derived sidecars live under `<pqtdir>.veloq/`.
/// Generated `<report>.veloq/parquetdir/` paths are accepted as an
/// alias for the owning `.nsys-rep` so repeated calls cannot fork a
/// second `<report>.veloq/parquetdir.veloq/` cache root.
///
/// `.sqlite` inputs are rejected outright; veloq has no SQLite
/// ingestion path.
pub fn resolve_trace(path: &Path) -> Result<ResolvedTracePath> {
    if is_nsys_rep(path) {
        if !path.exists() {
            bail!("trace not found: {}", path.display());
        }
        return Ok(ResolvedTracePath {
            source_path: path.to_path_buf(),
            pqtdir_path: ensure_parquetdir(path)?,
        });
    }
    if let Some(owner) = generated_parquetdir_owner(path) {
        if !owner.exists() {
            bail!(
                "generated parquetdir belongs to missing .nsys-rep source: {}",
                owner.display()
            );
        }
        return Ok(ResolvedTracePath {
            source_path: owner.clone(),
            pqtdir_path: ensure_parquetdir(&owner)?,
        });
    }
    if is_parquetdir(path) {
        if !path.exists() {
            bail!("parquetdir not found: {}", path.display());
        }
        return Ok(ResolvedTracePath {
            source_path: path.to_path_buf(),
            pqtdir_path: path.to_path_buf(),
        });
    }
    if path.extension().is_some_and(|e| e == "sqlite") {
        bail!(
            "veloq does not read .sqlite traces directly. \
             Pass the original .nsys-rep instead.\n\
             Rejected: {}",
            path.display()
        );
    }
    // Otherwise pass through unchanged — useful for unit tests that
    // pass dummy paths the rest of the pipeline won't follow.
    Ok(ResolvedTracePath {
        source_path: path.to_path_buf(),
        pqtdir_path: path.to_path_buf(),
    })
}

/// Back-compat helper for callers that only need the parquetdir.
pub fn resolve_trace_path(path: &Path) -> Result<PathBuf> {
    Ok(resolve_trace(path)?.pqtdir_path)
}

fn is_nsys_rep(path: &Path) -> bool {
    matches!(path.extension().and_then(OsStr::to_str), Some("nsys-rep"))
}

pub(crate) fn is_parquetdir(path: &Path) -> bool {
    path.file_name()
        .and_then(OsStr::to_str)
        .is_some_and(|n| n.ends_with(PQTDIR_SUFFIX))
}

/// Return the `.nsys-rep` owner for veloq's generated
/// `<report>.veloq/parquetdir/` path.
///
/// This intentionally does not recognize arbitrary directories named
/// `parquetdir`; only the exact child of a `.veloq` artifact root is
/// a generated alias. Direct user exports keep the `_pqtdir` suffix.
pub fn generated_parquetdir_owner(path: &Path) -> Option<PathBuf> {
    let file_name = path.file_name().and_then(OsStr::to_str)?;
    if file_name != PARQUETDIR_NAME {
        return None;
    }
    let cache_root = path.parent()?;
    let cache_name = cache_root.file_name().and_then(OsStr::to_str)?;
    let owner_name = cache_name.strip_suffix(ARTIFACT_DIR_SUFFIX)?;
    let owner = cache_root
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map_or_else(|| PathBuf::from(owner_name), |p| p.join(owner_name));
    if is_nsys_rep(&owner) {
        Some(owner)
    } else {
        None
    }
}

/// True when `path` is veloq's generated parquetdir child and its
/// owning `.nsys-rep` source still exists.
pub fn is_valid_generated_parquetdir(path: &Path) -> bool {
    generated_parquetdir_owner(path).is_some_and(|owner| owner.exists())
}

/// Source identity used by sidecar readers that receive a raw CLI path
/// and must not trigger a cold `.nsys-rep` export.
pub fn sidecar_source_path(path: &Path) -> PathBuf {
    generated_parquetdir_owner(path).unwrap_or_else(|| path.to_path_buf())
}

/// Produce `<input>.veloq/parquetdir/` for `.nsys-rep` inputs.
pub fn pqtdir_path_for(nsys_rep: &Path) -> PathBuf {
    artifact_dir_for(nsys_rep).join(PARQUETDIR_NAME)
}

fn lock_path_for(nsys_rep: &Path) -> PathBuf {
    artifact_dir_for(nsys_rep).join(LOCK_NAME)
}

fn nsys_generated_pqtdir_paths_for(export_cwd: &Path) -> [PathBuf; 2] {
    let mut name = std::ffi::OsString::from(EXPORT_BASENAME);
    name.push(PQTDIR_SUFFIX);
    [export_cwd.join(EXPORT_BASENAME), export_cwd.join(name)]
}

fn ensure_parquetdir(nsys_rep: &Path) -> Result<PathBuf> {
    let pqtdir = pqtdir_path_for(nsys_rep);

    // Lock-free fast path.
    if cache_valid(nsys_rep, &pqtdir)? {
        return Ok(pqtdir);
    }

    let lock_path = lock_path_for(nsys_rep);
    if let Some(parent) = lock_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating artifact directory {}", parent.display()))?;
    }
    let lock_file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .with_context(|| format!("opening parquetdir export lockfile {}", lock_path.display()))?;
    lock_file
        .lock()
        .with_context(|| format!("flock {}", lock_path.display()))?;

    // Re-check cache validity under the lock — a concurrent process
    // may have produced the parquetdir while we waited.
    if cache_valid(nsys_rep, &pqtdir)? {
        return Ok(pqtdir);
    }
    if publish_existing_generated_export(nsys_rep, &pqtdir)? {
        return Ok(pqtdir);
    }
    probe_nsys_capability()?;
    export(nsys_rep, &pqtdir)?;
    Ok(pqtdir)
}

fn publish_existing_generated_export(source: &Path, pqtdir: &Path) -> Result<bool> {
    let cache_root = artifact_dir_for(source);
    for generated_path in nsys_generated_pqtdir_paths_for(&cache_root) {
        if !cache_valid(source, &generated_path)? {
            continue;
        }
        if pqtdir.exists() {
            fs::remove_dir_all(pqtdir)
                .with_context(|| format!("removing stale parquetdir {}", pqtdir.display()))?;
        }
        fs::rename(&generated_path, pqtdir).with_context(|| {
            format!(
                "publishing parquetdir {} to {}",
                generated_path.display(),
                pqtdir.display()
            )
        })?;
        return Ok(true);
    }
    Ok(false)
}

/// ctime ordering per
/// `nsys_recipe.lib.data_reader._Loader::validate_export_time`.
/// We check the directory's sentinel file rather than the directory's
/// own ctime (which on some filesystems doesn't advance on child
/// writes). On Unix, Rust's `created()` is birth time rather than
/// ctime, so use `MetadataExt::ctime` explicitly.
fn cache_valid(nsys_rep: &Path, pqtdir: &Path) -> Result<bool> {
    if !pqtdir.is_dir() {
        return Ok(false);
    }
    let sentinel = pqtdir.join(SENTINEL_TABLE);
    if !sentinel.is_file() {
        // Partial / empty parquetdir → not valid.
        return Ok(false);
    }
    let src_meta = fs::metadata(nsys_rep)
        .with_context(|| format!("stat .nsys-rep source {}", nsys_rep.display()))?;
    let sentinel_meta = fs::metadata(&sentinel)
        .with_context(|| format!("stat parquetdir sentinel {}", sentinel.display()))?;
    let (Some(src), Some(cached)) = (change_time_key(&src_meta), change_time_key(&sentinel_meta))
    else {
        return Ok(false);
    };
    Ok(cached >= src)
}

#[cfg(unix)]
fn change_time_key(meta: &fs::Metadata) -> Option<(i64, i64)> {
    use std::os::unix::fs::MetadataExt;
    Some((meta.ctime(), meta.ctime_nsec()))
}

#[cfg(not(unix))]
fn change_time_key(meta: &fs::Metadata) -> Option<(i64, i64)> {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| (d.as_secs() as i64, d.subsec_nanos() as i64))
}

/// Capability probe — cached for the process lifetime.
fn probe_nsys_capability() -> Result<()> {
    static PROBED: OnceLock<Result<(), String>> = OnceLock::new();
    let result = PROBED.get_or_init(|| match has_parquetdir_support(Path::new("nsys")) {
        Ok(true) => Ok(()),
        Ok(false) => {
            let detected = nsys_version_string().unwrap_or_else(|_| "unknown".into());
            Err(format!(
                "veloq requires nsys ≥ 2024.6 (parquetdir export not supported by this nsys).\n\
                 Detected: {detected}\n\
                 Upgrade nsys to 2024.6 or newer."
            ))
        }
        Err(e) => Err(format!(
            "could not probe `nsys export --help type` for parquetdir support — \
             ensure the Nsight Systems CLI is installed and on PATH ({e:#})"
        )),
    });
    result
        .as_ref()
        .map(|_| ())
        .map_err(|msg| anyhow::anyhow!("{msg}"))
}

/// Visible to tests so a fake-nsys can be exercised. Production
/// callers go through `probe_nsys_capability`.
pub(crate) fn has_parquetdir_support(nsys_bin: &Path) -> Result<bool> {
    // Retry on `ETXTBSY` (raw os error 26): on some setups (NFS-backed
    // bin dirs, just-written test fixtures, the kernel-side mmap+exec
    // serialisation window) the file is briefly busy right after we
    // write + chmod it. A handful of millisecond-scale retries is the
    // standard incantation; the production path stat's a stable
    // `nsys` binary on PATH that won't see this, but tests exec
    // freshly-written shell scripts and need the retry.
    let mut attempts = 0;
    let out = loop {
        match Command::new(nsys_bin)
            .arg("export")
            .arg("--help")
            .arg("type")
            .output()
        {
            Ok(out) => break out,
            Err(e) if retry_text_file_busy(&e, &mut attempts) => {
                continue;
            }
            Err(e) => {
                return Err(anyhow::Error::new(e)).with_context(|| {
                    format!("running `{} export --help type`", nsys_bin.display())
                });
            }
        }
    };
    // `nsys export --help type` writes to stdout on success; some
    // versions of the CLI also send help to stderr. Inspect both.
    let combined = {
        let mut s = String::from_utf8_lossy(&out.stdout).into_owned();
        s.push_str(&String::from_utf8_lossy(&out.stderr));
        s
    };
    Ok(combined.contains("parquetdir"))
}

fn nsys_version_string() -> Result<String> {
    let out = Command::new("nsys")
        .arg("--version")
        .output()
        .context("running `nsys --version`")?;
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        Ok(String::from_utf8_lossy(&out.stderr).trim().to_string())
    } else {
        Ok(s)
    }
}

fn export(source: &Path, pqtdir: &Path) -> Result<()> {
    export_with_command(source, pqtdir, Path::new("nsys"))
}

fn export_with_command(source: &Path, pqtdir: &Path, nsys_bin: &Path) -> Result<()> {
    log::info!(
        "exporting {} → parquetdir via `nsys export -t parquetdir` …",
        source.display()
    );
    let started = std::time::Instant::now();

    // `nsys export -t parquetdir` writes an output-derived directory
    // (`<output>/` on current NSys, `<output>_pqtdir/` on some older
    // exporters). Make that output explicit inside the artifact root,
    // then publish the generated directory at the stable `parquetdir/`
    // path used by veloq.
    let cache_root = artifact_dir_for(source);
    fs::create_dir_all(&cache_root)
        .with_context(|| format!("creating artifact directory {}", cache_root.display()))?;
    for generated_path in nsys_generated_pqtdir_paths_for(&cache_root) {
        if generated_path.is_dir() {
            fs::remove_dir_all(&generated_path).with_context(|| {
                format!(
                    "removing stale generated parquetdir {}",
                    generated_path.display()
                )
            })?;
        } else if generated_path.exists() {
            fs::remove_file(&generated_path).with_context(|| {
                format!(
                    "removing stale generated parquetdir placeholder {}",
                    generated_path.display()
                )
            })?;
        }
    }
    let source_arg = fs::canonicalize(source)
        .with_context(|| format!("canonicalizing .nsys-rep source {}", source.display()))?;
    let mut attempts = 0;
    let out = loop {
        match Command::new(nsys_bin)
            .current_dir(&cache_root)
            .arg("export")
            .arg("-t")
            .arg("parquetdir")
            .arg("-f")
            .arg("true")
            .arg("-o")
            .arg(EXPORT_BASENAME)
            .arg(&source_arg)
            .output()
        {
            Ok(out) => break out,
            Err(e) if retry_text_file_busy(&e, &mut attempts) => continue,
            Err(e) => {
                return Err(anyhow::Error::new(e)).with_context(
                    || "running `nsys export -t parquetdir` — ensure the Nsight Systems CLI is installed and on PATH",
                );
            }
        }
    };
    forward_child_output_to_stderr(&out);
    if !out.status.success() {
        bail!(
            "`nsys export -t parquetdir` failed (exit {:?}) for {}",
            out.status.code(),
            source.display()
        );
    }

    let generated_pqtdir = nsys_generated_pqtdir_paths_for(&cache_root)
        .into_iter()
        .find(|path| path.is_dir())
        .ok_or_else(|| {
            let expected = nsys_generated_pqtdir_paths_for(&cache_root)
                .into_iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(" or ");
            anyhow::anyhow!(
                "`nsys export -t parquetdir` reported success but produced no directory at {expected}"
            )
        })?;
    if generated_pqtdir != pqtdir {
        if pqtdir.exists() {
            fs::remove_dir_all(pqtdir)
                .with_context(|| format!("removing stale parquetdir {}", pqtdir.display()))?;
        }
        fs::rename(&generated_pqtdir, pqtdir).with_context(|| {
            format!(
                "publishing parquetdir {} to {}",
                generated_pqtdir.display(),
                pqtdir.display()
            )
        })?;
    }

    if !pqtdir.is_dir() {
        bail!(
            "`nsys export -t parquetdir` reported success but produced no directory at {}",
            pqtdir.display()
        );
    }

    log::info!(
        "parquetdir export done in {:?} ({} parquet files)",
        started.elapsed(),
        fs::read_dir(pqtdir)
            .map(|it| it
                .flatten()
                .filter(|e| e.path().extension().is_some_and(|x| x == "parquet"))
                .count())
            .unwrap_or(0),
    );
    Ok(())
}

fn forward_child_output_to_stderr(out: &Output) {
    let mut stderr = std::io::stderr().lock();
    for bytes in [&out.stdout, &out.stderr] {
        if bytes.is_empty() {
            continue;
        }
        if stderr.write_all(bytes).is_err() {
            return;
        }
        if !bytes.ends_with(b"\n") && !bytes.ends_with(b"\r") && stderr.write_all(b"\n").is_err() {
            return;
        }
    }
}

fn retry_text_file_busy(e: &std::io::Error, attempts: &mut usize) -> bool {
    if e.raw_os_error() == Some(ETXTBSY) && *attempts < TEXT_BUSY_RETRIES {
        *attempts += 1;
        std::thread::sleep(std::time::Duration::from_millis(25));
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::io::Write;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn non_nsys_rep_paths_are_passed_through() -> Result<()> {
        let pqtdir_like = Path::new("/tmp/trace_pqtdir");
        // is_parquetdir → returns true; but path doesn't exist → bail.
        let err = match resolve_trace_path(pqtdir_like) {
            Ok(path) => anyhow::bail!(
                "missing parquetdir should not resolve to {}",
                path.display()
            ),
            Err(err) => err,
        };
        assert!(format!("{err:#}").contains("not found"));

        // Unknown extension: pass-through.
        let bare = Path::new("/tmp/example");
        assert_eq!(resolve_trace_path(bare)?, bare);
        Ok(())
    }

    #[test]
    fn sqlite_input_is_rejected() -> Result<()> {
        let p = Path::new("/tmp/trace.sqlite");
        let err = match resolve_trace_path(p) {
            Ok(path) => anyhow::bail!("sqlite input should not resolve to {}", path.display()),
            Err(err) => err,
        };
        let msg = format!("{err:#}");
        assert!(msg.contains(".sqlite"));
        assert!(msg.contains(".nsys-rep"));
        Ok(())
    }

    #[test]
    fn extension_detection() {
        assert!(is_nsys_rep(Path::new("/tmp/foo.nsys-rep")));
        assert!(!is_nsys_rep(Path::new("/tmp/foo.sqlite")));
        assert!(!is_nsys_rep(Path::new("/tmp/foo")));
        // `.rep` alone is not enough — must be the full `.nsys-rep`.
        assert!(!is_nsys_rep(Path::new("/tmp/foo.rep")));
    }

    #[test]
    fn parquetdir_path_appends_artifact_root_to_report_filename() {
        let p = pqtdir_path_for(Path::new("/tmp/example.nsys-rep"));
        assert_eq!(p, Path::new("/tmp/example.nsys-rep.veloq/parquetdir"));
        // Filenames with extra dots keep the full report filename in
        // the artifact root.
        let p = pqtdir_path_for(Path::new("/tmp/trace.v1.nsys-rep"));
        assert_eq!(p, Path::new("/tmp/trace.v1.nsys-rep.veloq/parquetdir"));
    }

    #[test]
    fn generated_parquetdir_aliases_to_owning_report() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let source = dir.path().join("trace.nsys-rep");
        fs::write(&source, b"source")?;
        let pqtdir = pqtdir_path_for(&source);
        fs::create_dir_all(&pqtdir)?;
        fs::write(pqtdir.join(SENTINEL_TABLE), b"sentinel")?;

        assert_eq!(generated_parquetdir_owner(&pqtdir), Some(source.clone()));
        assert_eq!(sidecar_source_path(&pqtdir), source);

        let resolved = resolve_trace(&pqtdir)?;
        assert_eq!(resolved.source_path, sidecar_source_path(&pqtdir));
        assert_eq!(resolved.pqtdir_path, pqtdir);
        Ok(())
    }

    #[test]
    fn generated_parquetdir_requires_owning_report() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let pqtdir = dir.path().join("missing.nsys-rep.veloq/parquetdir");
        fs::create_dir_all(&pqtdir)?;

        assert_eq!(
            generated_parquetdir_owner(&pqtdir),
            Some(dir.path().join("missing.nsys-rep"))
        );
        assert!(
            !is_valid_generated_parquetdir(&pqtdir),
            "orphan generated parquetdir should not be detected as valid"
        );
        let err = match resolve_trace(&pqtdir) {
            Ok(path) => anyhow::bail!(
                "orphan generated parquetdir should not resolve to {}",
                path.pqtdir_path.display()
            ),
            Err(err) => err,
        };
        assert!(format!("{err:#}").contains("generated parquetdir"));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn capability_probe_detects_parquetdir() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let fake = dir.path().join("nsys");
        let mut f = fs::File::create(&fake)?;
        writeln!(
            f,
            r#"#!/bin/sh
echo "Possible values are: sqlite, hdf, text, json, info, arrow, arrowdir, parquetdir"
"#
        )?;
        // ETXTBSY on NFS / parallel-test setups: ensure the write
        // is durable before `exec`. fsync + drop + a stat re-read
        // are the standard incantation.
        f.sync_all()?;
        drop(f);
        let mut perms = fs::metadata(&fake)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake, perms)?;
        assert!(has_parquetdir_support(&fake)?);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn capability_probe_rejects_old_nsys() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let fake = dir.path().join("nsys");
        let mut f = fs::File::create(&fake)?;
        writeln!(
            f,
            r#"#!/bin/sh
echo "Possible values are: arrow, hdf, info, json, sqlite, text"
"#
        )?;
        // ETXTBSY on NFS / parallel-test setups: ensure the write
        // is durable before `exec`. fsync + drop + a stat re-read
        // are the standard incantation.
        f.sync_all()?;
        drop(f);
        let mut perms = fs::metadata(&fake)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake, perms)?;
        assert!(!has_parquetdir_support(&fake)?);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn export_writes_to_pqtdir_path() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let cwd = dir.path().join("caller-cwd");
        let traces = dir.path().join("traces");
        fs::create_dir_all(&cwd)?;
        fs::create_dir_all(&traces)?;
        let source = traces.join("trace.nsys-rep");
        fs::write(&source, b"source")?;
        let pqtdir = pqtdir_path_for(&source);

        let fake_nsys = dir.path().join("nsys");
        let mut f = fs::File::create(&fake_nsys)?;
        writeln!(
            f,
            r#"#!/bin/sh
# Mimic `nsys export -t parquetdir -o <output>`: write
# <output>_pqtdir/<TABLE>.parquet. The real exporter derives default
# output from the input path, so this test requires -o to ensure veloq
# controls artifact-root placement explicitly.
out=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        -o|--output)
            shift
            out="$1"
            ;;
    esac
    shift
done
if [ -z "$out" ]; then
    echo "missing -o/--output" >&2
    exit 7
fi
mkdir -p "${{out}}_pqtdir"
echo placeholder > "${{out}}_pqtdir/META_DATA_EXPORT.parquet"
"#
        )?;
        f.sync_all()?;
        drop(f);
        let mut perms = fs::metadata(&fake_nsys)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_nsys, perms)?;

        let old_cwd = std::env::current_dir()?;
        std::env::set_current_dir(&cwd)?;
        let export_result = export_with_command(&source, &pqtdir, &fake_nsys);
        std::env::set_current_dir(old_cwd)?;
        export_result?;
        assert!(
            pqtdir.is_dir(),
            "expected parquetdir at {}",
            pqtdir.display()
        );
        assert!(pqtdir.join("META_DATA_EXPORT.parquet").is_file());
        for generated_path in nsys_generated_pqtdir_paths_for(&artifact_dir_for(&source)) {
            assert!(
                !generated_path.exists(),
                "export should publish {} to parquetdir/",
                generated_path.display()
            );
        }
        assert!(
            !traces.join("trace_pqtdir").exists(),
            "export must not write a parquetdir next to the source report"
        );
        assert!(
            !cwd.join("trace_pqtdir").exists(),
            "export must not write beside caller cwd"
        );
        Ok(())
    }

    #[test]
    fn warm_retry_publishes_existing_generated_export_dirs() -> Result<()> {
        for candidate_idx in 0..2 {
            let dir = tempfile::tempdir()?;
            let source = dir.path().join("trace.nsys-rep");
            fs::write(&source, b"source")?;
            let pqtdir = pqtdir_path_for(&source);
            let cache_root = artifact_dir_for(&source);
            fs::create_dir_all(&cache_root)?;
            let generated = nsys_generated_pqtdir_paths_for(&cache_root)
                .get(candidate_idx)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing generated path candidate"))?;
            fs::create_dir_all(&generated)?;
            fs::write(generated.join(SENTINEL_TABLE), b"placeholder")?;

            assert!(publish_existing_generated_export(&source, &pqtdir)?);
            assert!(
                pqtdir.join(SENTINEL_TABLE).is_file(),
                "expected published sentinel at {}",
                pqtdir.display()
            );
            assert!(
                !generated.exists(),
                "generated export directory should be moved to stable cache"
            );
        }
        Ok(())
    }
}
