//! Lightweight NCU list verbs — `ncu ranges` / `graphs` / `sources`
//! — projecting the `ncu_report`-native sidecar to the list contract.
//!
//! Each verb projects the matching sidecar list to the contract:
//! `{ count, total_matched, rows: Vec<*Row>, auxiliary }`. Headline
//! columns only — heavy fields (metric arrays, embedded source
//! content, disasm payloads) stay on the underlying entry and reach
//! agents through the per-domain verb (`ncu inspect`, `ncu metrics`,
//! `ncu disasm`).
//!
//! row_id formats: `range:<idx>`, `graph:<idx>`, `source:<idx>`.
//! Indices are 0-based positions in the corresponding sidecar list.
//!
//! CMDLIST (OptiX command lists) is not ingested.

use serde::Serialize;
use std::path::Path;

use crate::error::{NcuSourceError, NcuSourceResult};
use crate::native;

/// Auxiliary block every list verb returns. Just the sidecar path so
/// scripts can confirm a warm cache. The per-verb response carries
/// the actual row data; this stays the same shape across all five
/// for jq-symmetric handling.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct ListsAuxiliary {
    pub meta_cache_path: String,
}

// ---- ncu ranges / graphs ---------------------------------------------------
//
// Native non-KERNEL workloads. The two verbs share one
// row shape: `ncu_report` exposes the same headline columns for each
// workload type — `name` + recovered `context_id` / `device_id` /
// `stream_id` + metric / rule counts. CMDLIST (OptiX command lists)
// is not ingested.

/// One non-KERNEL workload row (range / graph). `key` /
/// `row_id` carry the `<kind>:<idx>` discriminator.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct WorkloadRow {
    pub key: String,
    pub row_id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_id: Option<u64>,
    pub metric_count: usize,
    pub rule_count: usize,
}

fn workload_rows(
    workloads: &[native::NativeWorkload],
    kind: &str,
    limit: usize,
) -> Vec<WorkloadRow> {
    workloads
        .iter()
        .enumerate()
        .take(limit)
        .map(|(idx, w)| {
            let row_id = format!("{kind}:{idx}");
            WorkloadRow {
                key: row_id.clone(),
                row_id,
                name: w.name.clone(),
                context_id: w.context_id,
                device_id: w.device_id,
                stream_id: w.stream_id,
                metric_count: w.metric_count,
                rule_count: w.rule_count,
            }
        })
        .collect()
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct RangesResponse {
    pub count: usize,
    pub total_matched: usize,
    pub rows: Vec<WorkloadRow>,
    pub auxiliary: ListsAuxiliary,
}

pub fn ranges<P: AsRef<Path>>(path: P, limit: usize) -> NcuSourceResult<RangesResponse> {
    if limit == 0 {
        return Err(NcuSourceError::limit_too_small(limit));
    }
    let path = path.as_ref();
    let sidecar = native::cache::build_or_load(path)?;
    Ok(ranges_from_sidecar(
        &sidecar,
        native::cache::path_for(path).display().to_string(),
        limit,
    ))
}

/// Project `ranges` from an already-loaded sidecar. Split out so the
/// golden test can drive it from a committed sidecar via
/// [`crate::native::cache::read_gz_sidecar`] (the source `.ncu-rep`
/// embeds a hostname and is not committed).
pub fn ranges_from_sidecar(
    sidecar: &native::NativeSidecar,
    cache_path: String,
    limit: usize,
) -> RangesResponse {
    let rows = workload_rows(&sidecar.ranges, "range", limit);
    RangesResponse {
        count: rows.len(),
        total_matched: sidecar.ranges.len(),
        rows,
        auxiliary: ListsAuxiliary {
            meta_cache_path: cache_path,
        },
    }
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct GraphsResponse {
    pub count: usize,
    pub total_matched: usize,
    pub rows: Vec<WorkloadRow>,
    pub auxiliary: ListsAuxiliary,
}

pub fn graphs<P: AsRef<Path>>(path: P, limit: usize) -> NcuSourceResult<GraphsResponse> {
    if limit == 0 {
        return Err(NcuSourceError::limit_too_small(limit));
    }
    let path = path.as_ref();
    let sidecar = native::cache::build_or_load(path)?;
    Ok(graphs_from_sidecar(
        &sidecar,
        native::cache::path_for(path).display().to_string(),
        limit,
    ))
}

pub fn graphs_from_sidecar(
    sidecar: &native::NativeSidecar,
    cache_path: String,
    limit: usize,
) -> GraphsResponse {
    let rows = workload_rows(&sidecar.graphs, "graph", limit);
    GraphsResponse {
        count: rows.len(),
        total_matched: sidecar.graphs.len(),
        rows,
        auxiliary: ListsAuxiliary {
            meta_cache_path: cache_path,
        },
    }
}

// ---- ncu sources ----------------------------------------------------------

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SourcesResponse {
    pub count: usize,
    pub total_matched: usize,
    pub rows: Vec<SourceRow>,
    pub auxiliary: ListsAuxiliary,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SourceRow {
    /// Cross-trace key — `source:<launch_idx>`. Under the native
    /// model there is no standalone source table; each row is the
    /// cubin a kernel launch ran out of, so the index matches the
    /// launch's `launch:<idx>`.
    pub key: String,
    pub row_id: String,
    /// Compute capability label (`sm_120`, `sm_90`, …) recovered from
    /// the launch's `device__attribute_compute_capability_*` metrics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cuda_sm_name: Option<String>,
    /// Count of embedded source files (`--import-source`). `0` unless
    /// the capture imported source — `ncu_report` exposes no raw
    /// cubin/PTX byte sizes, so those columns are not populated.
    pub embedded_source_file_count: usize,
    /// `true` iff a non-empty SASS listing was captured for the
    /// launch's cubin (i.e. `ncu disasm --row-id launch:N` will have
    /// instructions to show).
    pub has_disasm: bool,
}

/// Native model has no standalone source/cubin table — `ncu_report`
/// binds each action directly to its source. We synthesize one source
/// row per launch (the cubin it ran out of), degraded:
/// `reference` / `ptx_bytes` / `cubin_bytes` / `sass_level_name` are
/// dropped (no raw-binary surface).
pub fn sources<P: AsRef<Path>>(path: P, limit: usize) -> NcuSourceResult<SourcesResponse> {
    if limit == 0 {
        return Err(NcuSourceError::limit_too_small(limit));
    }
    let path = path.as_ref();
    let sidecar = native::cache::build_or_load(path)?;
    let total = sidecar.launches.len();
    let rows: Vec<SourceRow> = sidecar
        .launches
        .iter()
        .enumerate()
        .take(limit)
        .map(|(idx, l)| {
            let row_id = format!("source:{idx}");
            SourceRow {
                key: row_id.clone(),
                row_id,
                cuda_sm_name: l.sm_label(),
                embedded_source_file_count: 0,
                has_disasm: l.has_disasm(),
            }
        })
        .collect();
    Ok(SourcesResponse {
        count: rows.len(),
        total_matched: total,
        rows,
        auxiliary: ListsAuxiliary {
            meta_cache_path: native::cache::path_for(path).display().to_string(),
        },
    })
}
