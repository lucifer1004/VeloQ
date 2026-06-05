use crate::model::{FileFingerprint, InputFingerprint};
use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

pub fn detect_path(path: &Path) -> bool {
    if is_trace_file(path) {
        return true;
    }
    if !path.is_dir() {
        return false;
    }
    let Ok(entries) = fs::read_dir(path) else {
        return false;
    };
    entries
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .any(|child| is_trace_file(&child))
}

pub fn is_trace_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".pt.trace.json") || name.ends_with(".pt.trace.json.gz"))
}

pub(crate) fn discover_trace_files(input: &Path) -> Result<Vec<PathBuf>> {
    if input.is_file() {
        if is_trace_file(input) {
            return Ok(vec![input.to_path_buf()]);
        }
        anyhow::bail!(
            "pytorch source expects `.pt.trace.json` or `.pt.trace.json.gz`, got {}",
            input.display()
        );
    }
    if !input.is_dir() {
        anyhow::bail!("trace input does not exist: {}", input.display());
    }
    let mut files = Vec::new();
    for entry in
        fs::read_dir(input).with_context(|| format!("reading trace dir {}", input.display()))?
    {
        let entry = entry.with_context(|| format!("reading trace dir {}", input.display()))?;
        let path = entry.path();
        if path.is_file() && is_trace_file(&path) {
            files.push(path);
        }
    }
    files.sort_by_key(|path| path.display().to_string());
    if files.is_empty() {
        anyhow::bail!(
            "pytorch trace directory contains no `.pt.trace.json` or `.pt.trace.json.gz` files: {}",
            input.display()
        );
    }
    Ok(files)
}

pub(crate) fn fingerprint_for_files(files: &[PathBuf]) -> Result<InputFingerprint> {
    let mut out = Vec::with_capacity(files.len());
    for file in files {
        let metadata = fs::metadata(file).with_context(|| format!("stat {}", file.display()))?;
        let mtime_secs = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|duration| duration.as_secs() as i64)
            .unwrap_or(0);
        out.push(FileFingerprint {
            path: file.display().to_string(),
            mtime_secs,
            size: metadata.len(),
        });
    }
    Ok(InputFingerprint { files: out })
}

pub(crate) fn read_trace_text(file: &Path) -> Result<String> {
    let bytes = fs::read(file).with_context(|| format!("reading {}", file.display()))?;
    if file
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".gz"))
    {
        let mut decoder = GzDecoder::new(bytes.as_slice());
        let mut text = String::new();
        decoder
            .read_to_string(&mut text)
            .with_context(|| format!("decompressing {}", file.display()))?;
        return Ok(text);
    }
    String::from_utf8(bytes).with_context(|| format!("{} is not valid UTF-8 JSON", file.display()))
}
