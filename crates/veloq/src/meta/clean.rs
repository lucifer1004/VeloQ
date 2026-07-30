//! `veloq clean <trace>` — remove VeloQ-generated products for one
//! report.
//!
//! The command removes only the shared `<trace>.veloq/` artifact root
//! defined by `veloq-core`. Passing VeloQ's generated
//! `<report>.veloq/parquetdir/` child cleans its parent artifact root.
//! It does not remove direct `_pqtdir/` inputs, `.nsys-rep` files,
//! `.ncu-rep` / `.ncu-repz` files, or any legacy sidecar names.

use clap::{Arg, ArgAction, ArgMatches, Command};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use veloq_core::{
    ARTIFACT_DIR_SUFFIX, EnvelopeTraceRef, OutputFormat, ProfileSource, artifact_dir_for,
};

use super::{MetaError, MetaResult, emit_meta_error, emit_or_error};

const VERB: &str = "clean";

#[derive(Debug, Serialize)]
struct CleanPayload {
    cache_root: String,
    existed: bool,
    dry_run: bool,
    removed: bool,
    files: u64,
    directories: u64,
    bytes: u64,
}

#[derive(Debug, Default)]
struct ArtifactStats {
    files: u64,
    directories: u64,
    bytes: u64,
}

impl ArtifactStats {
    fn add(&mut self, other: ArtifactStats) {
        self.files += other.files;
        self.directories += other.directories;
        self.bytes += other.bytes;
    }
}

pub fn cli() -> Command {
    Command::new(VERB)
        .about("Remove VeloQ-generated cache products for one report")
        .arg(
            Arg::new("trace")
                .required(true)
                .value_name("PATH")
                .help("Path to a trace/report artifact, or its .veloq cache root"),
        )
        .arg(
            Arg::new("dry-run")
                .long("dry-run")
                .action(ArgAction::SetTrue)
                .help("Report what would be removed without deleting it"),
        )
}

pub fn run(
    matches: &ArgMatches,
    sources: &[Arc<dyn ProfileSource>],
    fmt: OutputFormat,
) -> MetaResult<i32> {
    let trace_str = match matches.get_one::<String>("trace") {
        Some(s) => s,
        None => {
            let err = MetaError::missing_argument("trace");
            emit_meta_error(fmt, VERB, None, &err);
            return Ok(1);
        }
    };
    let trace = PathBuf::from(trace_str);
    let dry_run = matches.get_flag("dry-run");
    let detected = sources.iter().find(|s| s.detect(&trace)).map(|s| s.kind());
    let trace_ref = Some(EnvelopeTraceRef {
        kind: detected.unwrap_or("unknown"),
        path: trace.display().to_string(),
    });

    match clean(&trace, dry_run) {
        Ok(payload) => Ok(emit_or_error(fmt, VERB, trace_ref, None, payload)),
        Err(err) => {
            emit_meta_error(fmt, VERB, trace_ref, &err);
            Ok(1)
        }
    }
}

fn clean(trace: &Path, dry_run: bool) -> MetaResult<CleanPayload> {
    let cache_root = cache_root_arg(trace);
    let existed = cache_root.exists();
    let stats = if existed {
        ensure_removable_cache_root(&cache_root)?;
        measure(&cache_root)?
    } else {
        ArtifactStats::default()
    };

    if existed && !dry_run {
        fs::remove_dir_all(&cache_root).map_err(|source| MetaError::RemoveCacheRoot {
            path: path_string(&cache_root),
            source,
        })?;
    }

    Ok(CleanPayload {
        cache_root: cache_root.display().to_string(),
        existed,
        dry_run,
        removed: existed && !dry_run,
        files: stats.files,
        directories: stats.directories,
        bytes: stats.bytes,
    })
}

fn cache_root_arg(trace: &Path) -> PathBuf {
    if veloq_nsys::generated_parquetdir_owner(trace).is_some()
        && let Some(cache_root) = trace.parent()
    {
        return cache_root.to_path_buf();
    }
    if trace
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.ends_with(ARTIFACT_DIR_SUFFIX))
    {
        trace.to_path_buf()
    } else {
        artifact_dir_for(trace)
    }
}

fn ensure_removable_cache_root(cache_root: &Path) -> MetaResult<()> {
    let meta = fs::symlink_metadata(cache_root).map_err(|source| MetaError::StatCacheRoot {
        path: path_string(cache_root),
        source,
    })?;
    let file_type = meta.file_type();
    if file_type.is_symlink() {
        return Err(MetaError::CacheRootSymlink {
            path: path_string(cache_root),
        });
    }
    if !file_type.is_dir() {
        return Err(MetaError::CacheRootNotDirectory {
            path: path_string(cache_root),
        });
    }
    Ok(())
}

fn measure(path: &Path) -> MetaResult<ArtifactStats> {
    let meta = fs::symlink_metadata(path).map_err(|source| MetaError::StatArtifact {
        path: path_string(path),
        source,
    })?;
    let file_type = meta.file_type();
    if file_type.is_symlink() {
        return Ok(ArtifactStats {
            files: 1,
            directories: 0,
            bytes: meta.len(),
        });
    }
    if !file_type.is_dir() {
        return Ok(ArtifactStats {
            files: 1,
            directories: 0,
            bytes: meta.len(),
        });
    }

    let mut stats = ArtifactStats {
        files: 0,
        directories: 1,
        bytes: 0,
    };
    for entry in fs::read_dir(path).map_err(|source| MetaError::ReadDir {
        path: path_string(path),
        source,
    })? {
        let entry = entry.map_err(|source| MetaError::ReadDirEntry {
            path: path_string(path),
            source,
        })?;
        stats.add(measure(&entry.path())?);
    }
    Ok(stats)
}

fn path_string(path: &Path) -> String {
    path.display().to_string()
}
