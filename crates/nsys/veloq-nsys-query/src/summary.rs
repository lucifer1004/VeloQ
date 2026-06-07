//! `veloq summary <trace>` — one-shot overview of a trace.
//!
//! Reports NSys version, total/per-table time ranges, the two trace
//! origins veloq tracks (primary = GPU-execution-anchored, full = all
//! events including OSRT/NVTX bootstrap), and per-table row counts.

use serde::Serialize;
use std::path::Path;
use veloq_nsys_data::{CapabilityFlags, Trace};

use crate::{NsysQueryError, NsysQueryResult};

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct Summary {
    /// Rows returned. Equal to `total_matched` (summary doesn't paginate).
    pub count: usize,
    pub total_matched: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub product_version: Option<String>,
    /// Canonical primary table. Each row is one event table with
    /// its row count and time span. `primary_time_range_ns` and
    /// `full_time_range_ns` are now expressed at the envelope level
    /// (`trace_span` for the primary; `auxiliary.full_time_range_ns`
    /// for the diagnostic full span).
    pub rows: Vec<TableSummary>,
    /// Non-row metadata: full-trace span (covers OSRT/NVTX bootstrap
    /// outside the primary table set) and capability bitmap.
    pub auxiliary: SummaryAuxiliary,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SummaryAuxiliary {
    /// Full event span, including OSRT/NVTX bootstrap markers (which
    /// NSys sometimes anchors hundreds of seconds before any GPU work).
    /// Useful for diagnostics; not the `--time-range` anchor — that's
    /// the envelope's `trace_span` (= the primary range).
    pub full_time_range_ns: TimeRange,
    /// What questions the trace can answer. Probed on open at <1ms.
    /// Agents read this *before* issuing heavy queries to know which
    /// kinds are queryable, whether `slices` will have NVTX data,
    /// whether `hardware` will return anything, etc. `None` when
    /// probing failed entirely (extremely unlikely — every probe is
    /// independent and `false` is the natural failure outcome).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<CapabilityFlags>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct TimeRange {
    pub start: i64,
    pub end: i64,
    pub duration: i64,
}

impl From<veloq_nsys_data::TimeSpan> for TimeRange {
    fn from(s: veloq_nsys_data::TimeSpan) -> Self {
        Self {
            start: s.start_ns,
            end: s.end_ns,
            duration: s.duration_ns(),
        }
    }
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct TableSummary {
    /// Cross-trace key. `table|<table_name>`.
    pub key: String,
    pub name: String,
    pub row_count: i64,
    pub start_ns: i64,
    pub end_ns: i64,
}

pub fn run<P: AsRef<Path>>(path: P) -> NsysQueryResult<Summary> {
    let trace = Trace::open(path).map_err(NsysQueryError::trace_open)?;

    // Hot path: pull everything from the metadata sidecar. First
    // call against a trace runs the COUNT(*) / MIN/MAX scans
    // (single-digit ms beyond the existing open) and persists the
    // result; subsequent calls — including fresh processes — read
    // a few KB of bincode and skip SQL entirely. The cache
    // invalidates on trace mtime/size change, so editing the trace
    // file behind veloq's back doesn't serve stale data.
    let meta = trace
        .meta_cache()
        .map_err(NsysQueryError::summary_meta_load)?;

    let per_table: Vec<TableSummary> = meta
        .per_table
        .iter()
        .map(|e| TableSummary {
            key: format!("table|{}", e.name),
            name: e.name.clone(),
            row_count: e.row_count,
            start_ns: e.start_ns,
            end_ns: e.end_ns,
        })
        .collect();

    // `schema_version` in the response is the `Display`-formatted
    // string ("3.22.1"), matching pre-cache behaviour. The cache
    // stores the structured `SchemaVersion`; rendering happens here
    // so a future response-shape change doesn't force a cache rebuild.
    let schema_version = meta
        .schema_version
        .as_ref()
        .map(|v| format!("{}.{}.{}", v.major, v.minor, v.micro));

    let count = per_table.len();
    Ok(Summary {
        count,
        total_matched: count,
        schema_version,
        product_version: meta.product_version.clone(),
        rows: per_table,
        auxiliary: SummaryAuxiliary {
            full_time_range_ns: meta.origins.full.into(),
            capabilities: Some(meta.capabilities.clone()),
        },
    })
}

// Note: meta selection, schema-version building, and row counting
// live in `veloq-nsys-data::meta_cache::build`, the single producer of
// these values. Keeping the SQL in one place lets the cache stay the
// authoritative shape and avoids a stale parallel implementation
// drifting on the response side.
