use crate::index::finalize_trace_set;
use crate::ingest::parse_trace_set;
use crate::input::{discover_trace_files, fingerprint_for_files};
use crate::model::{InputFingerprint, PrepState, TraceSet};
use crate::sidecar::{materialize_sidecars, sibling_tmp, sidecar_states};
use crate::{CACHE_VERSION, SOURCE_KIND};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use veloq_core::{TraceSpan, artifact_dir_for};

#[derive(Serialize, Deserialize)]
struct CacheFile {
    version: u32,
    fingerprint: InputFingerprint,
    trace_set: TraceSet,
}

pub fn build_or_load(input: &Path) -> Result<TraceSet> {
    let files = discover_trace_files(input)?;
    let fingerprint = fingerprint_for_files(&files)?;
    if let Some(cached) = read_cache(input, &fingerprint)? {
        if !sidecar_states(input).iter().all(|sidecar| sidecar.present) {
            materialize_sidecars(&cached)?;
        }
        return Ok(cached);
    }

    let mut trace_set = parse_trace_set(input, &files, fingerprint)?;
    finalize_trace_set(&mut trace_set);
    write_cache(input, &trace_set)?;
    materialize_sidecars(&trace_set)?;
    Ok(trace_set)
}

pub fn prep_state(input: &Path) -> Result<PrepState> {
    let files = discover_trace_files(input)?;
    let fingerprint = fingerprint_for_files(&files)?;
    let cache_fresh = read_cache(input, &fingerprint)?.is_some();
    let artifact_dir = artifact_dir(input).display().to_string();
    Ok(PrepState {
        input_path: input.display().to_string(),
        artifact_dir,
        cache_version: CACHE_VERSION,
        cache_fresh,
        sidecars: sidecar_states(input),
    })
}

pub fn trace_span_for_path(input: &Path) -> Option<TraceSpan> {
    let files = discover_trace_files(input).ok()?;
    let fingerprint = fingerprint_for_files(&files).ok()?;
    read_cache(input, &fingerprint)
        .ok()
        .flatten()
        .and_then(|trace| trace.envelope_trace_span())
}

pub fn artifact_dir(input: &Path) -> PathBuf {
    artifact_dir_for(input).join(SOURCE_KIND)
}

fn cache_path(input: &Path) -> PathBuf {
    artifact_dir(input).join("meta.bin")
}

fn read_cache(input: &Path, fingerprint: &InputFingerprint) -> Result<Option<TraceSet>> {
    let path = cache_path(input);
    if !path.exists() {
        return Ok(None);
    }
    let bytes =
        fs::read(&path).with_context(|| format!("reading pytorch cache {}", path.display()))?;
    let (cache, _read): (CacheFile, _) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
            .with_context(|| format!("decoding pytorch cache {}", path.display()))?;
    if cache.version != CACHE_VERSION || cache.fingerprint != *fingerprint {
        return Ok(None);
    }
    Ok(Some(cache.trace_set))
}

fn write_cache(input: &Path, trace_set: &TraceSet) -> Result<()> {
    let path = cache_path(input);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating pytorch cache dir {}", parent.display()))?;
    }
    let cache = CacheFile {
        version: CACHE_VERSION,
        fingerprint: trace_set.fingerprint.clone(),
        trace_set: trace_set.clone(),
    };
    let bytes = bincode::serde::encode_to_vec(&cache, bincode::config::standard())
        .context("encoding pytorch cache")?;
    let tmp = sibling_tmp(&path);
    fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    fs::rename(&tmp, &path).with_context(|| format!("publishing {}", path.display()))?;
    Ok(())
}
