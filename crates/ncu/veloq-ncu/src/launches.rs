//! `veloq ncu launches <file.ncu-rep>` — list every CUDA kernel
//! launch the report captured.
//!
//! Each row is the headline columns an agent needs to choose what
//! to drill into via [`crate::inspect`] — kernel name, grid/block
//! dims, context/device/stream, NVTX range stack. Full metrics and
//! rules stay on the (much larger) per-launch native sidecar entry
//! and surface through `ncu inspect --row-id launch:<idx>` only when
//! an agent asks.
//!
//! Reads the native sidecar ([`crate::native::cache`]).

use anyhow::Result;
use serde::Serialize;
use std::path::Path;

use crate::native::{NativeLaunch, cache};

/// CLI inputs for `ncu launches`. Built by `source.rs` from the
/// `Cmd::Launches` variant and handed to [`run`].
#[derive(Debug, Clone, Default)]
pub struct LaunchesRequest {
    pub kernel_glob: Option<String>,
    pub nvtx_range_glob: Option<String>,
    /// `(W, H, D)` extracted from `--grid WxHxD`. `0` on an axis
    /// means "any value matches" so an agent filtering by the
    /// dimensions they care about doesn't have to know the rest.
    pub grid: Option<[u64; 3]>,
    pub block: Option<[u64; 3]>,
    pub limit: usize,
}

/// `veloq ncu launches` response payload. Follows the canonical list contract:
/// rows-at-top + count + total_matched + auxiliary.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct LaunchesResponse {
    /// Rows returned (after `--limit`).
    pub count: usize,
    /// Rows matching the filter before `--limit`.
    pub total_matched: usize,
    /// Canonical primary table. Each row is one CUDA launch
    /// projected to its headline columns.
    pub rows: Vec<LaunchRow>,
    pub auxiliary: LaunchesAuxiliary,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct LaunchesAuxiliary {
    /// Glob filter echoed for diagnostics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_glob: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_range_glob: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grid_filter: Option<[u64; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_filter: Option<[u64; 3]>,
    /// Absolute path to the `<file>.veloq/ncu-native.json.gz` sidecar
    /// this response was sourced from.
    pub meta_cache_path: String,
}

/// One launch row — flat columns an agent can rank / filter on
/// without paying for the full metrics + rules payload. Keys
/// follow `"launch:<flat_idx>"` so a `jq INDEX` can pivot rows back
/// to the underlying launch entry in the sidecar.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct LaunchRow {
    /// Cross-trace key — equal to `row_id`.
    pub key: String,
    /// `"launch:<flat_idx>"` where `flat_idx` is the launch's
    /// position in the native sidecar's `launches` list (0-based).
    pub row_id: String,
    /// Demangled kernel signature. Empty when NCU couldn't demangle
    /// (rare, e.g. assembly kernels).
    pub kernel_demangled: String,
    /// Mangled kernel symbol — the raw `_Z…` name. Stable across
    /// the same binary across runs, so a useful cross-trace join
    /// key when demangled signatures vary by template parameters.
    pub kernel_mangled: String,
    /// Launch grid size — `[x, y, z]`.
    pub grid_size: [u64; 3],
    /// Launch block size — `[x, y, z]`.
    pub block_size: [u64; 3],
    /// CUDA context id (recovered from the `launch__context_id`
    /// metric).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_id: Option<u64>,
    /// CUDA device ordinal (from `launch__device_id`). New under the
    /// native model; useful for disambiguating multi-device captures.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_id: Option<u64>,
    /// `/`-joined names of every NVTX range active at the launch
    /// site (innermost last). `None` until the export helper emits
    /// NVTX state (`nvtx_state()`); the committed fixture captured none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_range_path: Option<String>,
}

/// Walk the cached native sidecar's `launches`, build a `LaunchRow` per
/// entry, then apply `--kernel` / `--nvtx-range` / `--grid` /
/// `--block` filters and the row cap. `count` is the post-limit
/// count, `total_matched` is the pre-limit match count so agents
/// can decide whether to raise `--limit` or narrow the filter.
pub fn run<P: AsRef<Path>>(path: P, req: LaunchesRequest) -> Result<LaunchesResponse> {
    anyhow::ensure!(
        req.limit > 0,
        "limit must be at least 1 (limit=0 suppresses every row including the total_matched / scope totals carried on them)"
    );
    let path = path.as_ref();
    let sidecar = cache::build_or_load(path)?;
    let meta_cache_path = cache::path_for(path).display().to_string();

    let kernel_matcher = req.kernel_glob.as_deref().map(crate::glob::compile);
    let nvtx_matcher = req.nvtx_range_glob.as_deref().map(crate::glob::compile);

    let mut total_matched = 0usize;
    let mut rows: Vec<LaunchRow> = Vec::new();
    for (idx, launch) in sidecar.launches.iter().enumerate() {
        // `ncu_report` does not yet surface NVTX state via the helper;
        // until it does, `nvtx_range_path` is `None` and a
        // `--nvtx-range` filter matches nothing.
        let nvtx_path: Option<String> = None;
        if !matches_kernel(launch, kernel_matcher.as_ref())
            || !matches_nvtx(nvtx_path.as_deref(), nvtx_matcher.as_ref())
            || !matches_dims(launch.grid_size, req.grid)
            || !matches_dims(launch.block_size, req.block)
        {
            continue;
        }
        total_matched += 1;
        if rows.len() >= req.limit {
            continue;
        }
        let row_id = format!("launch:{idx}");
        rows.push(LaunchRow {
            key: row_id.clone(),
            row_id,
            kernel_demangled: launch.kernel_demangled.clone(),
            kernel_mangled: launch.kernel_mangled.clone(),
            grid_size: launch.grid_size,
            block_size: launch.block_size,
            context_id: launch.metric_u64("launch__context_id"),
            device_id: launch.metric_u64("launch__device_id"),
            stream_id: launch.stream_id,
            nvtx_range_path: nvtx_path,
        });
    }

    Ok(LaunchesResponse {
        count: rows.len(),
        total_matched,
        rows,
        auxiliary: LaunchesAuxiliary {
            kernel_glob: req.kernel_glob,
            nvtx_range_glob: req.nvtx_range_glob,
            grid_filter: req.grid,
            block_filter: req.block,
            meta_cache_path,
        },
    })
}

/// Pre-compile `"WxHxD"` into a tuple. `0` on an axis means "any
/// value matches"; callers feed the result to [`matches_dims`].
pub fn parse_dims(s: &str) -> Result<[u64; 3]> {
    let parts: Vec<&str> = s.split(['x', 'X']).collect();
    let [x, y, z] = parts.as_slice() else {
        anyhow::bail!("expected `WxHxD` (got `{s}`); pad unused axes with 0 (e.g. `1024x1x1`)");
    };
    let parse_axis = |raw: &str| -> Result<u64> {
        raw.parse::<u64>()
            .map_err(|e| anyhow::anyhow!("invalid axis `{raw}` in `{s}`: {e}"))
    };
    Ok([parse_axis(x)?, parse_axis(y)?, parse_axis(z)?])
}

fn matches_kernel(launch: &NativeLaunch, matcher: Option<&crate::glob::Matcher>) -> bool {
    let Some(m) = matcher else { return true };
    m.matches(&launch.kernel_demangled) || m.matches(&launch.kernel_mangled)
}

fn matches_nvtx(path: Option<&str>, matcher: Option<&crate::glob::Matcher>) -> bool {
    let Some(m) = matcher else { return true };
    matches!(path, Some(p) if m.matches(p))
}

fn matches_dims(actual: [u64; 3], filter: Option<[u64; 3]>) -> bool {
    let Some(f) = filter else { return true };
    // 0 on a filter axis = wildcard.
    f.iter()
        .zip(actual.iter())
        .all(|(want, got)| *want == 0 || want == got)
}
