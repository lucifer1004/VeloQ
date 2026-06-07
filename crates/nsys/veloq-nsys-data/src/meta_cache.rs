//! On-disk metadata sidecar — `<trace>.veloq/meta.bin`.
//!
//! Caches the work `summary` (and its dependencies) do on cold open:
//! schema version, table list + counts + spans, hardware topology,
//! capability bitmap, NVTX nesting map. First call against a fresh
//! trace builds it (single-digit ms beyond the existing trace open);
//! every subsequent call deserialises a few KB of bincode and returns
//! instantly, so `summary`'s warm path is zero-SQL.
//!
//! ## Cache layout
//!
//! Goes through [`veloq_core::SidecarCache<TraceMetaCache>`] — the
//! shared bincode-with-version-header sidecar. Same on-disk shape
//! the correlation cache uses, with payload swapped for
//! [`TraceMetaCache`]. Invalidation (version mismatch, source-file
//! `(mtime, size)` mismatch, decode failure) and atomic-rename write
//! are SidecarCache's job; this module just supplies the path,
//! version, and label.
//!
//! ## What's cached vs not
//!
//! Cached: anything derivable from immutable trace state.
//! Capabilities, hardware, table list, per-table COUNT(*) and
//! MIN/MAX spans, NVTX nesting depths. These cost time to compute
//! but stay constant for a given trace file.
//!
//! Not cached: event-table data itself (lives in the Parquet cache),
//! correlation index (its own cache file), keeping the formats
//! separate lets each one evolve independently.

use crate::capabilities::CapabilityFlags;
use crate::hardware::HostInfo;
use crate::nvtx_nesting::NvtxEntry;
use crate::{NsysDataError, NsysDataResult, Trace};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use veloq_core::{SidecarCache, SourceFingerprint};

/// Bump on every breaking change to [`TraceMetaCache`]. Same
/// invalidation discipline as `correlation::CACHE_VERSION`: old
/// cache files with `version` != current rebuild silently on next
/// open.
pub const META_CACHE_VERSION: u32 = 1;

/// Everything the meta sidecar caches. All fields are direct
/// reflections of what other modules compute fresh; meta_cache
/// just glues the answers into one bincode blob.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TraceMetaCache {
    /// `SchemaVersion` parsed from `META_DATA_EXPORT`. `None` on
    /// older / partial exports (matches `Trace::schema_version()`).
    pub schema_version: Option<crate::SchemaVersion>,
    /// `EXPORT_PRODUCT_VERSION` string (raw, e.g. `"2025.4.1.136"`).
    pub product_version: Option<String>,
    /// Cheap-to-probe capability bitmap; same value
    /// [`CapabilityFlags::extract`] would produce.
    pub capabilities: CapabilityFlags,
    /// Hardware topology — empty Vec when `TARGET_INFO_SYSTEM_ENV`
    /// is absent.
    pub hosts: Vec<HostInfo>,
    /// Table names present in the attached `nsight` schema
    /// (lexically sorted). Mirror of `Trace::list_tables()`.
    pub available_tables: Vec<String>,
    /// Per-table `(name, row_count, start_ns, end_ns)` quadruple.
    /// The biggest single saving — large traces can take >100ms to
    /// `COUNT(*)` across all event tables.
    pub per_table: Vec<PerTableEntry>,
    /// Resolved trace origins (primary / full).
    pub origins: crate::TraceOrigins,
    /// NVTX nesting map (per-rowid depth + iter_index). Empty when
    /// the trace has no NVTX_EVENTS table; surfacing it lets warm
    /// `slices`, `search --type nvtx`, and reverse-attribution
    /// lookups skip the rayon scan entirely.
    pub nvtx_nesting: HashMap<i64, NvtxEntry>,
}

/// One row of [`TraceMetaCache::per_table`]. Keeps the field names
/// explicit so renaming any of them shows up as a real diff (and so
/// the bincode shape stays stable).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerTableEntry {
    pub name: String,
    pub row_count: i64,
    pub start_ns: i64,
    pub end_ns: i64,
}

/// Build the [`SidecarCache`] handle for a trace's `meta.bin`. The
/// cache itself is stateless — it carries path + version + label and
/// dispatches load/write through bincode. Kept on a free function so
/// both `build_or_load` and `try_load_existing` share one wiring.
fn cache_handle(trace_path: &Path) -> SidecarCache<TraceMetaCache> {
    SidecarCache::new(cache_path_for(trace_path), META_CACHE_VERSION, "meta cache")
}

/// Build or load the meta cache. First call after a trace open
/// (or whenever the trace file changes) rebuilds; subsequent calls
/// in fresh processes deserialise the sidecar.
pub fn build_or_load(trace: &Trace) -> NsysDataResult<TraceMetaCache> {
    let trace_path = trace.path();
    let cache = cache_handle(trace_path);
    let fp = source_fingerprint(trace_path)?;

    match cache.try_load(fp) {
        Ok(Some(payload)) => {
            log::debug!(
                "meta cache loaded from {} ({} tables, {} hosts, {} NVTX depths)",
                cache.path().display(),
                payload.per_table.len(),
                payload.hosts.len(),
                payload.nvtx_nesting.len()
            );
            return Ok(payload);
        }
        Ok(None) => {}
        Err(e) => {
            log::warn!(
                "meta cache at {} unusable ({e:#}); rebuilding",
                cache.path().display()
            );
        }
    }

    let started = std::time::Instant::now();
    let payload = build(trace)?;
    log::info!(
        "meta cache built in {:?}: {} tables, {} hosts, {} NVTX depths",
        started.elapsed(),
        payload.per_table.len(),
        payload.hosts.len(),
        payload.nvtx_nesting.len()
    );

    if let Err(e) = cache.write(fp, &payload) {
        log::warn!(
            "failed to write meta cache at {}: {e}",
            cache.path().display()
        );
    }
    Ok(payload)
}

/// Try to load a valid existing sidecar without rebuilding it.
///
/// Used both by hot-path helpers that already hold a `Trace` (via
/// `trace.path()`) and by the post-dispatch envelope-emit path
/// (which avoids opening a `Trace` just to read a few KB of bincode).
/// Returns `Ok(None)` when the sidecar is absent or fingerprint-stale.
pub fn try_load_existing(trace_path: &Path) -> NsysDataResult<Option<TraceMetaCache>> {
    let source_path = crate::nsys_rep::sidecar_source_path(trace_path);
    let fp = source_fingerprint(&source_path)?;
    cache_handle(&source_path)
        .try_load(fp)
        .map_err(NsysDataError::sidecar_operation)
}

/// Peek the on-disk header (format version + source fingerprint)
/// without decoding the payload. Used by `prep --status` to surface
/// the cache's version next to the current expected one — even for
/// a sidecar whose `try_load` would return `None` because of a
/// fingerprint mismatch.
pub fn read_header(trace_path: &Path) -> NsysDataResult<Option<veloq_core::SidecarHeader>> {
    cache_handle(trace_path)
        .read_header()
        .map_err(crate::NsysDataError::sidecar_header)
}

/// Project the meta sidecar's primary origin into a [`veloq_core::TraceSpan`]
/// (the envelope-level normalization denominator). Returns `None` if
/// the sidecar is absent, fingerprint-stale, or can't be read.
///
/// Single source of truth for envelope `trace_span`: both the
/// pre-dispatch hook (`NsysSource::compute_trace_span`) and the
/// post-dispatch emit refresh hand off here. Reads the file
/// directly — no `Trace::open` round-trip required.
pub fn trace_span_for_path(trace_path: &Path) -> Option<veloq_core::TraceSpan> {
    try_load_existing(trace_path)
        .ok()
        .flatten()
        .map(|m| veloq_core::TraceSpan {
            origin_ns: m.origins.primary.start_ns,
            span_ns: m.origins.primary.duration_ns(),
        })
}

fn build(trace: &Trace) -> NsysDataResult<TraceMetaCache> {
    let meta = trace.read_export_metadata()?;
    let schema_version = trace.schema_version().cloned();
    let product_version = meta
        .iter()
        .find(|(k, _)| k == "EXPORT_PRODUCT_VERSION")
        .map(|(_, v)| v.clone())
        .filter(|s| !s.is_empty());

    let capabilities = CapabilityFlags::extract(trace.pqtdir_path());
    let hosts = crate::hardware::extract(trace)?;
    let available_tables = trace.list_tables()?;

    // Per-table counts + spans. `read_origins` already does
    // MIN/MAX per table; pair it with COUNT(*) per table for the
    // full cached shape. Tables absent from `read_origins` (no
    // rows) get skipped entirely — matches `summary`'s current
    // behaviour.
    let (origins, per_table_spans) = trace.read_origins()?;
    let mut per_table: Vec<PerTableEntry> = Vec::with_capacity(per_table_spans.len());
    for (name, span) in per_table_spans {
        let row_count = count_rows(trace, name)?;
        per_table.push(PerTableEntry {
            name: name.to_string(),
            row_count,
            start_ns: span.start_ns,
            end_ns: span.end_ns,
        });
    }

    // NVTX nesting is the heaviest single component (one rayon
    // scan per (gtid, domain) group). Caching it pays for itself
    // after the second `slices` or `search --type nvtx` call.
    // `nvtx_nesting()` returns an empty map when NVTX_EVENTS is
    // absent — no special-casing needed.
    let nvtx_nesting = trace.compute_nvtx_nesting_uncached()?;

    Ok(TraceMetaCache {
        schema_version,
        product_version,
        capabilities,
        hosts,
        available_tables,
        per_table,
        origins,
        nvtx_nesting,
    })
}

fn count_rows(trace: &Trace, table: &str) -> NsysDataResult<i64> {
    // Parquetdir-only world: every table is a DuckDB view
    // over `<pqtdir>/<table>.parquet`. Parquet row-group metadata
    // makes `COUNT(*)` near-instant.
    let table_ident = crate::quote_sql_identifier(table);
    let sql = format!("SELECT COUNT(*) FROM nsight.{table_ident}");
    let mut stmt = trace
        .conn()
        .prepare(&sql)
        .map_err(|source| crate::NsysDataError::meta_cache_count_prepare(table, source))?;
    let count: i64 = stmt
        .query_row([], |row| row.get(0))
        .map_err(|source| crate::NsysDataError::meta_cache_count_read(table, source))?;
    Ok(count)
}

fn source_fingerprint(path: &Path) -> NsysDataResult<SourceFingerprint> {
    crate::trace_artifact_fingerprint(path)
        .map_err(|source| NsysDataError::trace_fingerprint_read(path.display(), source))
}

// ---- helper: where the cache lives -----------------------------------------

fn cache_path_for(trace_path: &Path) -> PathBuf {
    let source_path = crate::nsys_rep::sidecar_source_path(trace_path);
    veloq_core::artifact_dir_for(&source_path).join("meta.bin")
}

// ---- helper used by Trace ---------------------------------------------------

/// Best-effort cache file path. Public so the `prep` subcommand can
/// log it explicitly; internal callers stay on `build_or_load`.
pub fn path_for(trace_path: &Path) -> PathBuf {
    cache_path_for(trace_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use duckdb::Connection;
    use std::path::PathBuf;
    use tempfile::TempDir;
    use veloq_core::VeloqDiagnostic;

    fn minimal_trace() -> Result<(TempDir, PathBuf)> {
        let dir = tempfile::tempdir()?;
        let pqtdir = dir.path().join("test_pqtdir");
        std::fs::create_dir_all(&pqtdir)?;
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(
            r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start BIGINT, "end" BIGINT)"#,
        )?;
        let out = pqtdir.join("CUPTI_ACTIVITY_KIND_KERNEL.parquet");
        let out_lit = out.to_string_lossy().replace('\'', "''");
        conn.execute(
            &format!(
                r#"COPY (SELECT * FROM "CUPTI_ACTIVITY_KIND_KERNEL") TO '{out_lit}' (FORMAT PARQUET)"#
            ),
            [],
        )?;
        Ok((dir, pqtdir))
    }

    #[test]
    fn count_rows_missing_table_has_typed_prepare_error() -> Result<()> {
        let (_dir, pqtdir) = minimal_trace()?;
        let trace = Trace::open(&pqtdir)?;

        let err = match count_rows(&trace, "MISSING_TABLE") {
            Ok(count) => anyhow::bail!("missing table should not count rows: {count}"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.data.duckdb-prepare");
        assert_eq!(
            err.duckdb_parts(),
            Some(("meta cache", crate::DuckdbPhase::Prepare, "MISSING_TABLE"))
        );
        Ok(())
    }

    #[test]
    fn count_rows_read_error_code_is_stable() {
        let err = crate::NsysDataError::meta_cache_count_read(
            "CUPTI_ACTIVITY_KIND_KERNEL",
            duckdb_error(),
        );
        assert_eq!(err.code().as_str(), "nsys.data.duckdb-read");
        assert!(err.to_string().contains("CUPTI_ACTIVITY_KIND_KERNEL"));
    }

    fn duckdb_error() -> duckdb::Error {
        duckdb::Error::InvalidParameterName("test".to_string())
    }
}
