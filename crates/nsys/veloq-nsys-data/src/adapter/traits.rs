//! Schema-adapter trait and shared types.
//!
//! Different NSys releases use slightly different export schemas.
//! [`SchemaAdapter`] is the version-dispatch surface: every adapter
//! declares static [`AdapterMeta`] and implements a cheap [`probe`]
//! against the attached `nsight` connection. [`super::pick_adapter`]
//! walks the registered set and picks the first matching adapter (or
//! bails with a clear error). Only [`super::StandardAdapter`] (NSys
//! 3.x) is shipped today; the trait stays as the extension point for
//! future major-version forks.
//!
//! Query SQL reads `nsight.<TABLE>` directly. The adapter never
//! projects normalised views — its sole job is to gate `Trace::open`
//! on a recognised schema.
//!
//! [`probe`]: SchemaAdapter::probe

use anyhow::Result;
use duckdb::Connection;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Metadata describing a schema adapter — read by [`super::pick_adapter`]
/// for log lines and by [`super::adapter_by_id`] for warm-open reuse.
/// Not surfaced through any JSON envelope today; consumers are
/// internal.
#[derive(Debug, Clone, Copy)]
pub struct AdapterMeta {
    /// Stable identifier. Persisted in the metadata cache so warm
    /// calls can skip the probe.
    pub id: &'static str,
    /// Human-readable name (`"Standard Schema (3.x)"`, …).
    pub display_name: &'static str,
    /// One-line description of what the adapter looks for during probe.
    pub match_criteria: &'static str,
    /// Approximate NSys product versions the adapter targets. Free-text
    /// because NSys versioning isn't strictly semantic.
    pub target_versions: &'static str,
    /// Stability tier. Surfaced through log lines so operators can
    /// spot when a `Beta` adapter is in play.
    pub status: AdapterStatus,
}

/// Adapter compatibility tier. Plain-text `Display` (no emoji) —
/// retained on the trait so future adapters can mark themselves
/// `Beta` and operators can see the tier in log lines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterStatus {
    Stable,
    Beta,
}

impl AdapterStatus {
    /// Lowercase token. Used for log output so operators can
    /// string-grep without rendering.
    pub fn as_str(self) -> &'static str {
        match self {
            AdapterStatus::Stable => "stable",
            AdapterStatus::Beta => "beta",
        }
    }
}

impl std::fmt::Display for AdapterStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Schema-version triple parsed from `META_DATA_EXPORT`. Adapters use
/// it during `pick_adapter` dispatch; downstream consumers can echo
/// it back through `summary` for diagnostics. Serde derives are
/// here so the metadata cache (see `crate::meta_cache`) can persist it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaVersion {
    pub major: u32,
    pub minor: u32,
    pub micro: u32,
    /// `EXPORT_PRODUCT_VERSION` — e.g. `"2025.4.1.136"`. Optional
    /// because not every export sets it.
    pub product_version: Option<String>,
    /// `EXPORT_PRODUCT_DATE` — e.g. `"2025.4"`.
    pub product_date: Option<String>,
}

impl SchemaVersion {
    /// `(major, minor) ≥ (m, n)`. Convenience for adapters that want
    /// to constrain to a minimum schema.
    pub fn at_least(&self, major: u32, minor: u32) -> bool {
        self.major > major || (self.major == major && self.minor >= minor)
    }
}

impl std::fmt::Display for SchemaVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.micro)?;
        if let Some(p) = &self.product_version {
            write!(f, " (NSys {p})")?;
        }
        Ok(())
    }
}

/// The adapter contract. Every NSys schema variant veloq supports
/// ships an impl. Probe order is defined by [`super::pick_adapter`].
///
/// Implementations must be cheap to construct (zero-sized unit
/// structs in practice). They're held behind `Arc<dyn SchemaAdapter>`
/// on the [`crate::Trace`] handle.
pub trait SchemaAdapter: Send + Sync {
    /// Static metadata. Always returns the same value for a given impl.
    fn metadata(&self) -> AdapterMeta;

    /// Probe the parquetdir — return `true` iff this adapter can
    /// handle the trace. Cheap filesystem checks:
    /// look for the canonical-shape parquet files in the directory.
    /// Probes must be cheap because `pick_adapter` may try every
    /// adapter in sequence.
    fn probe(&self, pqtdir: &Path) -> bool;
}

// ============================================================================
// Helpers shared by every adapter
// ============================================================================

/// Cheap "is this table present" check — verifies that
/// `<pqtdir>/<table>.parquet` exists.
///
/// veloq reads NSys traces only through the
/// parquetdir export, so "did the trace capture this table?" reduces
/// to a filesystem stat. Independent of the DuckDB connection's
/// view set, which makes it the right answer for `capabilities`,
/// `hardware::extract`, adapter probes, etc.
pub fn table_exists(pqtdir: &Path, table: &str) -> bool {
    let file = pqtdir.join(format!("{table}.parquet"));
    file.is_file()
}

/// Read `META_DATA_EXPORT` and parse `EXPORT_SCHEMA_VERSION_*` into a
/// [`SchemaVersion`]. Returns `None` when the table is absent (some
/// very old exports) or when the major/minor/micro keys are missing.
///
/// `product_version` / `product_date` are best-effort: a present
/// schema-version triple with absent product-version is still
/// returned. Adapters consult this during version-first dispatch.
pub fn get_schema_version(conn: &Connection) -> Result<Option<SchemaVersion>> {
    // META_DATA_EXPORT resolves under the `nsight` schema once
    // `Trace::open` returns. An absent table (older or partial
    // export) is treated as no schema info rather than an error.
    let mut stmt = match conn.prepare(
        "SELECT name, value FROM nsight.META_DATA_EXPORT \
         WHERE name IN ('EXPORT_SCHEMA_VERSION_MAJOR', \
                        'EXPORT_SCHEMA_VERSION_MINOR', \
                        'EXPORT_SCHEMA_VERSION_MICRO', \
                        'EXPORT_PRODUCT_VERSION', \
                        'EXPORT_PRODUCT_DATE')",
    ) {
        Ok(stmt) => stmt,
        Err(_) => return Ok(None),
    };
    let mut rows = stmt.query([])?;

    let mut major: Option<u32> = None;
    let mut minor: Option<u32> = None;
    let mut micro: Option<u32> = None;
    let mut product_version: Option<String> = None;
    let mut product_date: Option<String> = None;

    while let Some(r) = rows.next()? {
        let key: String = r.get(0)?;
        // META_DATA_EXPORT.value alternates between TEXT and BIGINT
        // depending on which NSys subsystem wrote it; route through
        // duckdb's polymorphic Value rather than guessing.
        let raw: duckdb::types::Value = r.get(1)?;
        let value = value_to_string(&raw);
        match key.as_str() {
            "EXPORT_SCHEMA_VERSION_MAJOR" => major = value.parse().ok(),
            "EXPORT_SCHEMA_VERSION_MINOR" => minor = value.parse().ok(),
            "EXPORT_SCHEMA_VERSION_MICRO" => micro = value.parse().ok(),
            "EXPORT_PRODUCT_VERSION" if !value.is_empty() => {
                product_version = Some(value);
            }
            "EXPORT_PRODUCT_DATE" if !value.is_empty() => {
                product_date = Some(value);
            }
            _ => {}
        }
    }

    match (major, minor, micro) {
        (Some(major), Some(minor), Some(micro)) => Ok(Some(SchemaVersion {
            major,
            minor,
            micro,
            product_version,
            product_date,
        })),
        _ => Ok(None),
    }
}

fn value_to_string(v: &duckdb::types::Value) -> String {
    use duckdb::types::Value;
    match v {
        Value::Null => String::new(),
        Value::Text(s) => s.clone(),
        Value::BigInt(n) => n.to_string(),
        Value::Int(n) => n.to_string(),
        Value::SmallInt(n) => n.to_string(),
        Value::Float(n) => n.to_string(),
        Value::Double(n) => n.to_string(),
        Value::Boolean(b) => b.to_string(),
        other => format!("{other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_version_display_includes_product_when_present() {
        let v = SchemaVersion {
            major: 3,
            minor: 22,
            micro: 1,
            product_version: Some("2025.4.1.136".into()),
            product_date: None,
        };
        assert_eq!(v.to_string(), "3.22.1 (NSys 2025.4.1.136)");
    }

    #[test]
    fn schema_version_display_omits_missing_product() {
        let v = SchemaVersion {
            major: 3,
            minor: 22,
            micro: 1,
            product_version: None,
            product_date: None,
        };
        assert_eq!(v.to_string(), "3.22.1");
    }

    #[test]
    fn at_least_compares_major_first() {
        let v = SchemaVersion {
            major: 3,
            minor: 22,
            micro: 0,
            product_version: None,
            product_date: None,
        };
        assert!(v.at_least(3, 22));
        assert!(v.at_least(3, 0));
        assert!(v.at_least(2, 99));
        assert!(!v.at_least(3, 23));
        assert!(!v.at_least(4, 0));
    }

    #[test]
    fn adapter_status_str_is_plain_text() {
        // Tokens used by log lines — keep them lowercase + stable.
        assert_eq!(AdapterStatus::Stable.as_str(), "stable");
        assert_eq!(AdapterStatus::Beta.as_str(), "beta");
    }
}
