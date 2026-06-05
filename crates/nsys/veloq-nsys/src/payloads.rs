//! CLI-side response payloads — types that exist only on the wire,
//! not as a `veloq-nsys-query::*Response` (because the work happens in
//! `main.rs::run()` directly, not in `veloq-nsys-query`).
//!
//! Both derive [`schemars::JsonSchema`] so the [`crate::help`]
//! projector treats them identically to the response types from
//! `veloq-nsys-query`, and `veloq schema <target>` can emit their shapes
//! when those commands are looked up by name.

/// `veloq prep` response payload. Mirrors the JSON we emit;
/// `meta_cache_path` lets scripts confirm the sidecar exists after
/// prep without recomputing the path themselves.
#[derive(serde::Serialize, schemars::JsonSchema)]
pub struct PrepPayload {
    pub elapsed_ms: u64,
    /// Per-report artifact root that owns all veloq-generated files.
    pub cache_root: String,
    pub parquet_tables: Vec<String>,
    pub meta_cache_path: String,
}

/// `veloq prep --status` response — read-only inspection of the
/// on-disk caches. Agents can call this before deciding whether
/// to run a heavy verb cold or pay the prep cost up front.
#[derive(serde::Serialize, schemars::JsonSchema)]
pub struct PrepStatusPayload {
    /// Per-report artifact root that owns all veloq-generated files.
    pub cache_root: String,
    pub parquet_cache: ParquetCacheStatus,
    pub meta_cache: SidecarStatus,
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

/// Generic sidecar (meta or correlation) state. Path is always
/// computed; size/mtime are populated only when the file is on
/// disk; `fingerprint_match` is `true` only when the cache itself
/// loaded cleanly under the trace's current fingerprint.
#[derive(serde::Serialize, schemars::JsonSchema)]
pub struct SidecarStatus {
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
