//! Schema-adapter registry — version dispatch on trace open.
//!
//! veloq attaches the NSys parquetdir as schema `nsight` and reads from
//! it in DuckDB. Query SQL across the workspace assumes the canonical
//! 3.x column shape (`start`/`"end"`, `correlationId`, dedicated
//! `streamId`/`contextId`). This module's job is to *confirm* that
//! shape on open via [`pick_adapter`] and bail cleanly otherwise —
//! agents see an envelope error instead of a downstream SQL crash on
//! a missing column.
//!
//! Today only [`StandardAdapter`] (NSys schema 3.x) is shipped. The
//! [`SchemaAdapter`] trait stays as the extension point: a future
//! schema fork lands as a new `XAdapter` here and registers in
//! [`pick_adapter`] / [`adapter_by_id`].

pub mod traits;
pub mod v3_standard;

pub use traits::{
    AdapterMeta, AdapterStatus, SchemaAdapter, SchemaVersion, get_schema_version, table_exists,
};
pub use v3_standard::StandardAdapter;

use crate::NsysDataResult;
use duckdb::Connection;
use std::path::Path;
use std::sync::Arc;

/// How an adapter was chosen. Surfaced internally for log lines and
/// the meta cache's optional provenance hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionMethod {
    /// Schema version matched an adapter's target range and the
    /// adapter's probe confirmed feature compatibility.
    VersionMatch,
    /// Schema version was missing or unrecognised; probe alone
    /// selected the adapter.
    FeatureProbe,
}

impl DetectionMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            DetectionMethod::VersionMatch => "version_match",
            DetectionMethod::FeatureProbe => "feature_probe",
        }
    }
}

/// Result of [`pick_adapter`] — the chosen adapter plus enough
/// provenance for `summary` to explain *how* it was chosen.
pub struct AdapterChoice {
    pub adapter: Arc<dyn SchemaAdapter>,
    pub method: DetectionMethod,
    /// Schema version read from `META_DATA_EXPORT`, when present.
    /// Callers route this into `summary.schema_version` so the
    /// version-read step doesn't run twice.
    pub schema_version: Option<SchemaVersion>,
}

/// Pick the schema adapter for the attached `nsight` schema.
///
/// Strategy:
///   1. Read schema version from `META_DATA_EXPORT` (cheap).
///   2. Version-dispatch: schema 3.x → [`StandardAdapter`], confirmed
///      by its probe.
///   3. If META_DATA_EXPORT is missing or the version doesn't match,
///      run `StandardAdapter::probe` directly so partial exports with
///      the canonical 3.x columns still open.
///   4. If even that fails, `bail!` — caller's `Trace::open` returns
///      the error through the JSON envelope.
pub fn pick_adapter(conn: &Connection, pqtdir: &Path) -> NsysDataResult<AdapterChoice> {
    let schema_version = get_schema_version(conn)?;

    if let Some(ref version) = schema_version
        && let Some(adapter) = select_by_version(version)
        && adapter.probe(pqtdir)
    {
        return Ok(AdapterChoice {
            adapter,
            method: DetectionMethod::VersionMatch,
            schema_version,
        });
    }

    // Version dispatch missed (no META_DATA_EXPORT, unknown major
    // version, or version-matched probe rejected the trace). Probe
    // StandardAdapter directly — if its canonical columns are present
    // we open even without META_DATA_EXPORT.
    let adapter: Arc<dyn SchemaAdapter> = Arc::new(StandardAdapter);
    if adapter.probe(pqtdir) {
        return Ok(AdapterChoice {
            adapter,
            method: DetectionMethod::FeatureProbe,
            schema_version,
        });
    }

    Err(crate::NsysDataError::SchemaAdapterUnmatched)
}

/// Adapter lookup by stable id. The meta cache uses this to
/// reconstruct the adapter from its persisted id on warm opens.
pub fn adapter_by_id(id: &str) -> Option<Arc<dyn SchemaAdapter>> {
    match id {
        "v3_standard" => Some(Arc::new(StandardAdapter)),
        _ => None,
    }
}

fn select_by_version(version: &SchemaVersion) -> Option<Arc<dyn SchemaAdapter>> {
    match version.major {
        // 3.x is the modern NSys export (2023+). Future major
        // versions will register their adapters here.
        3 => Some(Arc::new(StandardAdapter)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context, Result};

    #[test]
    fn adapter_by_id_round_trip() -> Result<()> {
        let v3 = adapter_by_id("v3_standard").context("v3_standard id is well-known")?;
        assert_eq!(v3.metadata().id, "v3_standard");
        assert!(adapter_by_id("bogus").is_none());
        Ok(())
    }

    #[test]
    fn detection_method_str_round_trip() {
        assert_eq!(DetectionMethod::VersionMatch.as_str(), "version_match");
        assert_eq!(DetectionMethod::FeatureProbe.as_str(), "feature_probe");
    }
}
