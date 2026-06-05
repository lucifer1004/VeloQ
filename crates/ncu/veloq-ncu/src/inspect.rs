//! `veloq ncu inspect --row-id launch:<idx> ...` — full per-launch
//! details: metrics, rules, NVTX state, and the recovered identity
//! scalars.
//!
//! Mirrors `veloq inspect <kernel:N>` on the NSys side: an agent
//! reads `ncu launches`, picks rows of interest, then calls
//! `inspect` with their `row_id`s for the heavy details. Reads the
//! native sidecar ([`crate::native::cache`]), so inspect
//! is a fast deserialize + per-`row_id` index lookup.
//!
//! The heavy SASS listing stays on `ncu disasm`; inspect carries the
//! full metric + rule payload plus the recovered identity scalars.
//! The section catalog and cpu/python stacks are
//! dropped (no `ncu_report` equivalent).
//!
//! Out-of-range row_ids return a `NotFound` row so a partial batch
//! still produces a usable response — agents can pass a heuristic
//! id list without pre-filtering for existence.

use anyhow::{Context, Result};
use serde::Serialize;
use std::path::Path;

use crate::native::{NativeLaunch, NativeMetric, cache};

/// `veloq ncu inspect` response payload.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct InspectResponse {
    /// Rows returned (= input row_ids resolved).
    pub count: usize,
    /// Same as `count`; inspect doesn't paginate.
    pub total_matched: usize,
    /// Canonical primary table. One row per requested `row_id`,
    /// in input order. Out-of-range entries become `NotFound`.
    pub rows: Vec<LaunchDetailsRow>,
}

/// One inspected launch — the full metric + rule payload plus the
/// recovered identity scalars, with the `key` / `row_id`.
/// `NotFound` rows surface as a tagged-union sibling so the JSON
/// shape stays uniform across hits and misses.
///
/// The `Launch` variant boxes its body so the enum isn't permanently
/// sized for the heavy launch payload while the `NotFound` variant is
/// tiny (clippy's `large_enum_variant`). The box is one heap alloc
/// per matched row — negligible next to materializing the launch.
#[derive(Debug, Serialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LaunchDetailsRow {
    Launch(Box<LaunchDetails>),
    NotFound {
        key: String,
        row_id: String,
        reason: String,
    },
}

/// Full per-launch detail under the native model. Carries the
/// complete metric + rule arrays (the heavy payload inspect exists
/// for) plus the recovered identity scalars. The SASS listing is not
/// embedded — it stays on `ncu disasm` — but `has_disasm` /
/// `cubin_load_base` signal whether that verb will have output.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct LaunchDetails {
    pub key: String,
    pub row_id: String,
    pub kernel_demangled: String,
    pub kernel_mangled: String,
    pub kernel_function: String,
    pub grid_size: [u64; 3],
    pub block_size: [u64; 3],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cubin_load_base: Option<u64>,
    pub metric_count: usize,
    pub rule_count: usize,
    pub has_disasm: bool,
    pub metrics: Vec<NativeMetric>,
    /// Rule findings passed through verbatim from `ncu_report`.
    pub rules: Vec<serde_json::Value>,
}

impl LaunchDetails {
    fn from_launch(key: String, launch: &NativeLaunch) -> Self {
        LaunchDetails {
            row_id: key.clone(),
            key,
            kernel_demangled: launch.kernel_demangled.clone(),
            kernel_mangled: launch.kernel_mangled.clone(),
            kernel_function: launch.kernel_function.clone(),
            grid_size: launch.grid_size,
            block_size: launch.block_size,
            context_id: launch.metric_u64("launch__context_id"),
            device_id: launch.metric_u64("launch__device_id"),
            stream_id: launch.stream_id,
            cubin_load_base: launch.cubin_load_base,
            metric_count: launch.metrics.len(),
            rule_count: launch.rules.len(),
            has_disasm: launch.has_disasm(),
            metrics: launch.metrics.clone(),
            rules: launch.rules.clone(),
        }
    }
}

/// Resolve every `row_id` against the native sidecar's launches and
/// build the response in input order.
pub fn run<P: AsRef<Path>>(path: P, row_ids: &[String]) -> Result<InspectResponse> {
    anyhow::ensure!(
        !row_ids.is_empty(),
        "ncu inspect needs at least one --row-id"
    );
    let path = path.as_ref();
    let sidecar = cache::build_or_load(path)?;
    let n_launches = sidecar.launches.len();

    let mut rows: Vec<LaunchDetailsRow> = Vec::with_capacity(row_ids.len());
    for raw in row_ids {
        let row = match parse_launch_idx(raw) {
            Ok(idx) => match sidecar.launches.get(idx) {
                Some(launch) => LaunchDetailsRow::Launch(Box::new(LaunchDetails::from_launch(
                    raw.clone(),
                    launch,
                ))),
                None => LaunchDetailsRow::NotFound {
                    key: raw.clone(),
                    row_id: raw.clone(),
                    reason: format!(
                        "launch idx {idx} out of range ({n_launches} launches in this report)"
                    ),
                },
            },
            Err(e) => LaunchDetailsRow::NotFound {
                key: raw.clone(),
                row_id: raw.clone(),
                reason: format!("{e:#}"),
            },
        };
        rows.push(row);
    }

    let count = rows.len();
    Ok(InspectResponse {
        count,
        total_matched: count,
        rows,
    })
}

/// Parse the `"launch:<n>"` form. Future row-id kinds (range:N, ...)
/// join here; until then we reject anything else loudly so an agent
/// typo doesn't silently miss. Mirrors [`crate::row_id::parse_launch_idx`],
/// the shared parser used by the drill verbs.
fn parse_launch_idx(s: &str) -> Result<usize> {
    let (kind, idx) = s
        .split_once(':')
        .with_context(|| format!("expected `launch:<idx>`, got `{s}`"))?;
    anyhow::ensure!(
        kind == "launch",
        "ncu inspect currently supports only `launch:<idx>` row_ids (got `{kind}`)"
    );
    idx.parse::<usize>()
        .with_context(|| format!("invalid launch index `{idx}` in `{s}`"))
}
