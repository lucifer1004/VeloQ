//! `veloq ncu metrics --counter <glob> [--per-launch]` — cross-launch
//! metric values projected for jq-style comparison.
//!
//! Two output shapes:
//!
//! - **Long format (default).** One row per `(launch, counter)`
//!   pair. `key = "launch:<idx>|counter:<name>"`, value is the
//!   scalar `MetricEntry.value` cast through `serde_json::Value`,
//!   plus `unit` / `value_type`. This is the natural shape for
//!   `jq -s 'group_by(.counter_name)'` and for two-trace joins via
//!   `INDEX(.data.rows; .key)`.
//! - **Wide format (`--per-launch`).** One row per launch, with a
//!   `counters: { <name>: <value>, ... }` map of every matched
//!   counter. Useful when an agent wants the value cluster per
//!   launch (e.g. dump 5 columns per row into a CSV).
//!
//! Per-instance breakdowns (one value per SM, one value per SASS
//! address) are kept off this verb — they belong on `ncu inspect`
//! where they don't drown out the cross-launch comparison.
//!
//! Reads the native sidecar ([`crate::native::cache`]).

use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;

use crate::error::{NcuSourceError, NcuSourceResult};
use crate::glob;
use crate::native::{NativeLaunch, cache};

/// CLI inputs for `ncu metrics`.
#[derive(Debug, Clone, Default)]
pub struct MetricsRequest {
    /// Glob over the metric `name` field. Required — `None` would
    /// dump every metric on every launch, which is `ncu inspect`'s
    /// job, not `metrics`.
    pub counter_glob: String,
    /// Glob over launches' demangled kernel signature (same shape
    /// `ncu launches --kernel <glob>` uses). `None` = all launches.
    pub kernel_glob: Option<String>,
    /// `true` = wide format (one row per launch, counters nested);
    /// `false` = long format (one row per `(launch, counter)` pair).
    pub per_launch: bool,
    pub limit: usize,
}

/// `veloq ncu metrics` response. `format` tells agents which row
/// variant to expect while preserving the `data.rows[]` list
/// contract.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct MetricsResponse {
    /// Rows returned (post `--limit`).
    pub count: usize,
    /// Rows matching the filter before `--limit` was applied.
    pub total_matched: usize,
    /// Long is the default cross-launch row shape; per-launch is the
    /// wide shape selected by `--per-launch`.
    pub format: MetricsFormat,
    /// Canonical primary table. Each row carries a stable `key`.
    pub rows: Vec<MetricRow>,
    pub auxiliary: MetricsAuxiliary,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MetricsFormat {
    Long,
    PerLaunch,
}

/// Long-vs-wide primary row variants. Agents should branch on
/// `data.format`, not infer the format from individual fields.
#[derive(Debug, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum MetricRow {
    Long(MetricLongRow),
    PerLaunch(MetricWideRow),
}

/// One `(launch, counter)` row in long format.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct MetricLongRow {
    /// Cross-trace key: `launch:<idx>|counter:<name>`. Two runs
    /// of the same workload yield matching keys when the launches
    /// resolve to the same flat index and the metric name agrees.
    pub key: String,
    pub launch_row_id: String,
    pub counter_name: String,
    /// JSON-typed value, mirroring `MetricEntry.value` (number /
    /// string / null).
    pub value: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    /// Discriminator: `"double"` / `"uint64"` / `"string"` / ...
    /// (matches `MetricEntry.value_type`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value_type: Option<String>,
}

/// One launch in wide format with every matched counter as a key
/// inside `counters`. Sorted name ordering keeps the JSON
/// reproducible across runs.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct MetricWideRow {
    /// Cross-trace key: `launch:<idx>`.
    pub key: String,
    pub row_id: String,
    pub kernel_demangled: String,
    /// `{ counter_name → value }` for every counter matching
    /// `--counter`. Sorted by name.
    pub counters: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct MetricsAuxiliary {
    pub counter_glob: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_glob: Option<String>,
    pub meta_cache_path: String,
}

pub fn run<P: AsRef<Path>>(path: P, req: MetricsRequest) -> NcuSourceResult<MetricsResponse> {
    if req.limit == 0 {
        return Err(NcuSourceError::limit_too_small(req.limit));
    }
    if counter_glob_is_empty(&req.counter_glob) {
        return Err(NcuSourceError::counter_glob_empty());
    }
    let path = path.as_ref();
    let sidecar = cache::build_or_load(path)?;
    let meta_cache_path = cache::path_for(path).display().to_string();

    let kernel_matcher = req.kernel_glob.as_deref().map(glob::compile);
    let counter_matcher = glob::compile(&req.counter_glob);

    if req.per_launch {
        let mut total_matched = 0usize;
        let mut rows: Vec<MetricWideRow> = Vec::new();
        for (idx, launch) in sidecar.launches.iter().enumerate() {
            if !matches_kernel(launch, kernel_matcher.as_ref()) {
                continue;
            }
            let counters = collect_counters(launch, &counter_matcher);
            if counters.is_empty() {
                continue;
            }
            total_matched += 1;
            if rows.len() >= req.limit {
                continue;
            }
            let row_id = format!("launch:{idx}");
            rows.push(MetricWideRow {
                key: row_id.clone(),
                row_id,
                kernel_demangled: launch.kernel_demangled.clone(),
                counters,
            });
        }
        Ok(MetricsResponse {
            count: rows.len(),
            total_matched,
            format: MetricsFormat::PerLaunch,
            rows: rows.into_iter().map(MetricRow::PerLaunch).collect(),
            auxiliary: MetricsAuxiliary {
                counter_glob: req.counter_glob,
                kernel_glob: req.kernel_glob,
                meta_cache_path,
            },
        })
    } else {
        let mut total_matched = 0usize;
        let mut rows: Vec<MetricLongRow> = Vec::new();
        for (idx, launch) in sidecar.launches.iter().enumerate() {
            if !matches_kernel(launch, kernel_matcher.as_ref()) {
                continue;
            }
            for metric in launch.metrics.iter() {
                let name = metric.name.as_str();
                if !counter_matcher.matches(name) {
                    continue;
                }
                total_matched += 1;
                if rows.len() >= req.limit {
                    continue;
                }
                rows.push(MetricLongRow {
                    key: format!("launch:{idx}|counter:{name}"),
                    launch_row_id: format!("launch:{idx}"),
                    counter_name: name.to_string(),
                    value: metric.value.clone(),
                    unit: metric.unit.clone(),
                    value_type: Some(metric.value_type.clone()),
                });
            }
        }
        Ok(MetricsResponse {
            count: rows.len(),
            total_matched,
            format: MetricsFormat::Long,
            rows: rows.into_iter().map(MetricRow::Long).collect(),
            auxiliary: MetricsAuxiliary {
                counter_glob: req.counter_glob,
                kernel_glob: req.kernel_glob,
                meta_cache_path,
            },
        })
    }
}

fn matches_kernel(launch: &NativeLaunch, matcher: Option<&glob::Matcher>) -> bool {
    let Some(m) = matcher else { return true };
    m.matches(&launch.kernel_demangled) || m.matches(&launch.kernel_mangled)
}

fn counter_glob_is_empty(counter: &str) -> bool {
    !counter.split(',').any(|part| !part.trim().is_empty())
}

fn collect_counters(
    launch: &NativeLaunch,
    matcher: &glob::Matcher,
) -> BTreeMap<String, serde_json::Value> {
    let mut out = BTreeMap::new();
    for metric in launch.metrics.iter() {
        let name = metric.name.as_str();
        if matcher.matches(name) {
            out.insert(name.to_string(), metric.value.clone());
        }
    }
    out
}
