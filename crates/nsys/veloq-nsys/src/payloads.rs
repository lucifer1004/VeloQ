//! CLI-side response payloads — types that exist only on the wire,
//! not as a `veloq-nsys-query::*Response` (because the work happens in
//! `main.rs::run()` directly, not in `veloq-nsys-query`).
//!
//! Both derive [`schemars::JsonSchema`] so the [`crate::help`]
//! projector treats them identically to the response types from
//! `veloq-nsys-query`, and `veloq schema <target>` can emit their shapes
//! when those commands are looked up by name.

/// `veloq prep` / `veloq prep --status` response payload.
///
/// This is a canonical list response: each row is one registered NSys
/// sidecar and `auxiliary` carries command-level cache context.
#[derive(serde::Serialize, schemars::JsonSchema)]
pub struct PrepPayload {
    pub count: usize,
    pub total_matched: usize,
    pub rows: Vec<PrepRow>,
    pub auxiliary: PrepAuxiliary,
}

/// Parquet-cache directory state — describes the
/// `<report>.veloq/parquetdir/` export veloq reuses.
/// Validity is ctime-ordered against the source `.nsys-rep`; no
/// manifest is involved.
#[derive(serde::Serialize, schemars::JsonSchema)]
pub struct ParquetCacheStatus {
    /// Filesystem path of the parquetdir.
    pub dir: String,
    /// `true` iff the parquetdir exists on disk. Stale-vs-source
    /// invalidation runs at `Trace::open` time, not here.
    pub present: bool,
    /// `<TABLE>.parquet` files present in the parquetdir, sorted.
    pub tables: Vec<String>,
}

/// One registered NSys sidecar readiness row.
#[derive(serde::Serialize, schemars::JsonSchema)]
pub struct PrepRow {
    pub key: String,
    pub name: String,
    pub path: String,
    pub present: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    /// Filesystem mtime as seconds since UNIX epoch. `None` when
    /// the file is absent or the platform doesn't expose mtime.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mtime_secs: Option<i64>,
    /// Format version this veloq binary will write/expect.
    pub format_version_expected: u32,
    /// Format version actually present on disk. `None` when the
    /// sidecar is absent or the header is unreadable. Different
    /// from `format_version_expected` is the most common
    /// fingerprint-miss cause after a veloq upgrade.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format_version_on_disk: Option<u32>,
    pub fingerprint_match: bool,
}

impl From<veloq_nsys_data::NsysSidecarStatus> for PrepRow {
    fn from(status: veloq_nsys_data::NsysSidecarStatus) -> Self {
        Self {
            key: status.key,
            name: status.name,
            path: status.path,
            present: status.present,
            size_bytes: status.size_bytes,
            mtime_secs: status.mtime_secs,
            format_version_expected: status.format_version_expected,
            format_version_on_disk: status.format_version_on_disk,
            fingerprint_match: status.fingerprint_match,
        }
    }
}

#[derive(serde::Serialize, schemars::JsonSchema)]
pub struct PrepAuxiliary {
    /// Per-report artifact root that owns all veloq-generated files.
    pub cache_root: String,
    pub parquet_cache: ParquetCacheStatus,
    /// `true` for `prep`, `false` for `prep --status`.
    pub prepared: bool,
    pub elapsed_ms: u64,
}

/// `veloq correlation-stats` response payload. Row counts come back
/// per kind so agents can see which tables the correlation index
/// touched at build time.
#[derive(serde::Serialize, schemars::JsonSchema)]
pub struct CorrelationStatsPayload {
    pub elapsed_ms: u64,
    pub cache_present_after: bool,
    pub contexts: usize,
    pub processes: usize,
    pub unique_groups: usize,
    pub kernel_rows: usize,
    pub memcpy_rows: usize,
    pub memset_rows: usize,
    pub runtime_rows: usize,
    pub sync_rows: usize,
    pub graph_rows: usize,
}

/// `veloq schema <target>` response payload. The schema document is
/// already a `serde_json::Value` (built via `schemars::schema_for!`),
/// so the payload just pairs it with the target verb name.
#[derive(serde::Serialize)]
pub struct SchemaPayload {
    pub target: String,
    pub schema: serde_json::Value,
}
