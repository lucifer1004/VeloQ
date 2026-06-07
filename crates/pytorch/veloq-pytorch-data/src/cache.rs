use crate::index::finalize_trace_set;
use crate::ingest::parse_trace_set;
use crate::input::{discover_trace_files, fingerprint_for_files};
use crate::model::{InputFingerprint, PrepState, QueryTrace, TraceSet};
use crate::sidecar::{
    PytorchSidecar, materialize_sidecars, query_sidecars_ready, sidecar_path_for_artifact,
    sidecar_states, sidecars_ready,
};
use crate::{CACHE_VERSION, PytorchDataError, PytorchDataResult, SOURCE_KIND};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use veloq_core::{TraceSpan, artifact_dir_for};
use veloq_data::file::atomic_publish;

const SIDECAR_SCHEMA_VERSION: u32 = 1;
const QUERY_METADATA_VERSION: u32 = 1;
const QUERY_METADATA_CACHE: &str = "query-meta.bin";

#[derive(Serialize, Deserialize)]
struct CacheFile {
    version: u32,
    sidecar_schema_version: u32,
    fingerprint: InputFingerprint,
    trace_set: TraceSet,
}

#[derive(Serialize, Deserialize)]
struct QueryMetadataFile {
    version: u32,
    cache_version: u32,
    sidecar_schema_version: u32,
    fingerprint: InputFingerprint,
    trace: QueryTrace,
}

pub fn build_or_load(input: &Path) -> PytorchDataResult<TraceSet> {
    let files = discover_trace_files(input)?;
    let fingerprint = fingerprint_for_files(&files)?;
    if let Some(cached) = read_cache(input, &fingerprint)? {
        if !sidecars_ready(input) {
            materialize_sidecars(&cached)?;
        }
        write_query_metadata(input, &cached.query_trace())?;
        return Ok(cached);
    }

    let mut trace_set = parse_trace_set(input, &files, fingerprint)?;
    finalize_trace_set(&mut trace_set);
    write_cache(input, &trace_set)?;
    materialize_sidecars(&trace_set)?;
    write_query_metadata(input, &trace_set.query_trace())?;
    Ok(trace_set)
}

pub fn build_or_load_query_trace(input: &Path) -> PytorchDataResult<QueryTrace> {
    let files = discover_trace_files(input)?;
    let fingerprint = fingerprint_for_files(&files)?;
    if query_sidecars_ready(input)
        && let Some(trace) = read_query_metadata(input, &fingerprint)?
    {
        return Ok(trace);
    }

    let trace_set = build_or_load(input)?;
    Ok(trace_set.query_trace())
}

pub fn prep_state(input: &Path) -> PytorchDataResult<PrepState> {
    let files = discover_trace_files(input)?;
    let fingerprint = fingerprint_for_files(&files)?;
    let cached = read_cache(input, &fingerprint)?;
    let cache_fresh = cached.is_some();
    let artifact_dir = artifact_dir(input).display().to_string();
    Ok(PrepState {
        input_path: input.display().to_string(),
        artifact_dir,
        cache_version: CACHE_VERSION,
        cache_fresh,
        sidecars: sidecar_states(input),
        schema_survey: cached.map(|trace| trace.schema_survey),
    })
}

pub fn trace_span_for_path(input: &Path) -> Option<TraceSpan> {
    let files = discover_trace_files(input).ok()?;
    let fingerprint = fingerprint_for_files(&files).ok()?;
    if let Some(trace) = read_query_metadata(input, &fingerprint).ok().flatten() {
        return trace.envelope_trace_span();
    }
    read_cache(input, &fingerprint)
        .ok()
        .flatten()
        .and_then(|trace| trace.envelope_trace_span())
}

pub fn artifact_dir(input: &Path) -> PathBuf {
    artifact_dir_for(input).join(SOURCE_KIND)
}

fn cache_path(input: &Path) -> PathBuf {
    sidecar_path_for_artifact(artifact_dir(input), PytorchSidecar::Meta)
}

fn query_metadata_path(input: &Path) -> PathBuf {
    artifact_dir(input).join(QUERY_METADATA_CACHE)
}

fn read_cache(input: &Path, fingerprint: &InputFingerprint) -> PytorchDataResult<Option<TraceSet>> {
    let path = cache_path(input);
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(&path).map_err(|source| PytorchDataError::read_cache(&path, source))?;
    let Ok(cache) = serde_json::from_slice::<CacheFile>(&bytes) else {
        return Ok(None);
    };
    if cache.version != CACHE_VERSION
        || cache.sidecar_schema_version != SIDECAR_SCHEMA_VERSION
        || cache.fingerprint != *fingerprint
    {
        return Ok(None);
    }
    Ok(Some(cache.trace_set))
}

fn read_query_metadata(
    input: &Path,
    fingerprint: &InputFingerprint,
) -> PytorchDataResult<Option<QueryTrace>> {
    let path = query_metadata_path(input);
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(&path).map_err(|source| PytorchDataError::read_cache(&path, source))?;
    let Ok(cache) = serde_json::from_slice::<QueryMetadataFile>(&bytes) else {
        return Ok(None);
    };
    if cache.version != QUERY_METADATA_VERSION
        || cache.cache_version != CACHE_VERSION
        || cache.sidecar_schema_version != SIDECAR_SCHEMA_VERSION
        || cache.fingerprint != *fingerprint
    {
        return Ok(None);
    }
    Ok(Some(cache.trace))
}

fn write_query_metadata(input: &Path, trace: &QueryTrace) -> PytorchDataResult<()> {
    let path = query_metadata_path(input);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|source| PytorchDataError::create_cache_dir(parent, source))?;
    }
    let cache = QueryMetadataFile {
        version: QUERY_METADATA_VERSION,
        cache_version: CACHE_VERSION,
        sidecar_schema_version: SIDECAR_SCHEMA_VERSION,
        fingerprint: trace.fingerprint.clone(),
        trace: trace.clone(),
    };
    let bytes = serde_json::to_vec(&cache).map_err(PytorchDataError::encode_cache)?;
    atomic_publish(&path, |tmp| -> PytorchDataResult<()> {
        fs::write(tmp, bytes).map_err(|source| PytorchDataError::write_cache(tmp, source))
    })
}

fn write_cache(input: &Path, trace_set: &TraceSet) -> PytorchDataResult<()> {
    let path = cache_path(input);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|source| PytorchDataError::create_cache_dir(parent, source))?;
    }
    let cache = CacheFile {
        version: CACHE_VERSION,
        sidecar_schema_version: SIDECAR_SCHEMA_VERSION,
        fingerprint: trace_set.fingerprint.clone(),
        trace_set: trace_set.clone(),
    };
    let bytes = serde_json::to_vec(&cache).map_err(PytorchDataError::encode_cache)?;
    atomic_publish(&path, |tmp| -> PytorchDataResult<()> {
        fs::write(tmp, bytes).map_err(|source| PytorchDataError::write_cache(tmp, source))
    })
}
