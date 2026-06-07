//! `ncu warp-stalls --row-id launch:<idx>` — per-source-line warp-stall
//! histogram from `timed_warp_samples()`.
//!
//! Projects the per-launch `warp_stalls` sidecar field (the raw periodic
//! warp-state stream the helper already aggregated to a per-PC ×
//! `StallReason` count). Three rollup axes:
//! - `line` (default): one row per `(file, line)` over attributed PCs.
//! - `sass`: one row per in-cubin PC (`rel_address`), source attached
//!   when known.
//! - `reason`: one row per `StallReason` across the whole kernel.
//!
//! `auxiliary` carries the kernel totals + the placement buckets, and
//! the **self-contained reconciliation identity** holds (no `ncu metrics`
//! reference needed): per launch,
//! `Σ row.total_samples(by sass) + unattributed_samples + out_of_cubin_samples
//!  == total_samples == len(timed_warp_samples())`.
//!
//! Extraction, not interpretation: raw reason counts + `not_issued`
//! only — no severity ranking, no percentages (a jq one-liner over the
//! counts), no remediation hints. Those live in the skill docs.

use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;

use crate::error::{NcuSourceError, NcuSourceResult};
use crate::glob;
use crate::native::{self, NativeSidecar, NativeWarpStallPc};

/// Rollup axis for `--by`.
#[derive(Debug, Clone, Copy)]
pub enum Axis {
    Line,
    Sass,
    Reason,
}

impl Axis {
    pub fn parse(s: &str) -> NcuSourceResult<Self> {
        match s {
            "line" => Ok(Axis::Line),
            "sass" => Ok(Axis::Sass),
            "reason" => Ok(Axis::Reason),
            other => Err(NcuSourceError::unknown_warp_stalls_axis(other)),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Axis::Line => "line",
            Axis::Sass => "sass",
            Axis::Reason => "reason",
        }
    }
}

#[derive(Debug, Clone)]
pub struct WarpStallsRequest {
    pub row_id: String,
    pub by: Axis,
    /// Restrict to source files matching this glob (line / sass axes).
    pub file_glob: Option<String>,
    pub limit: usize,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum WarpStallsRow {
    Line(WarpStallsLineRow),
    Sass(WarpStallsSassRow),
    Reason(WarpStallsReasonRow),
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct WarpStallsLineRow {
    /// `<file>:<line>` — the cross-trace key.
    pub key: String,
    pub file: String,
    pub line: u32,
    pub total_samples: u64,
    /// Sample count per `StallReason` name at this line.
    pub stalls: BTreeMap<String, u64>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct WarpStallsSassRow {
    /// `sass:<rel_address>` — the cross-trace key.
    pub key: String,
    pub rel_address: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    pub total_samples: u64,
    pub stalls: BTreeMap<String, u64>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct WarpStallsReasonRow {
    /// `reason:<name>` — the cross-trace key.
    pub key: String,
    pub reason: String,
    pub total_samples: u64,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct WarpStallsAuxiliary {
    /// `len(timed_warp_samples())` — the reconciliation total.
    pub total_samples: u64,
    /// SM-issue-stall signal: samples where no warp was issued.
    pub not_issued_samples: u64,
    /// In-cubin samples with no DWARF line (the `in_cubin_no_source`
    /// bucket); the line-axis rollup's missing budget.
    pub unattributed_samples: u64,
    /// Samples whose `pc` is not in this launch's cubin.
    pub out_of_cubin_samples: u64,
    /// Kernel-wide sample count per `StallReason` name.
    pub per_reason_totals: BTreeMap<String, u64>,
    pub meta_cache_path: String,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct WarpStallsResponse {
    pub axis: String,
    pub count: usize,
    pub total_matched: usize,
    pub rows: Vec<WarpStallsRow>,
    pub auxiliary: WarpStallsAuxiliary,
}

pub fn run<P: AsRef<Path>>(path: P, req: WarpStallsRequest) -> NcuSourceResult<WarpStallsResponse> {
    let sidecar = native::cache::build_or_load(path.as_ref())?;
    let cache_path = native::cache::path_for(path.as_ref()).display().to_string();
    run_on_sidecar(&sidecar, req, cache_path)
}

/// Core logic over an already-loaded native sidecar. Split from [`run`]
/// so the golden test can drive it from a committed sidecar via
/// [`crate::native::cache::read_gz_sidecar`] — the source `.ncu-rep` is
/// not committed (committed-sidecar mode).
pub fn run_on_sidecar(
    sidecar: &NativeSidecar,
    req: WarpStallsRequest,
    cache_path: String,
) -> NcuSourceResult<WarpStallsResponse> {
    if req.limit == 0 {
        return Err(NcuSourceError::limit_too_small(req.limit));
    }
    let idx = crate::row_id::parse_launch_idx(&req.row_id)?;
    let n_launches = sidecar.launches.len();
    if idx >= n_launches {
        return Err(NcuSourceError::launch_row_id_out_of_range(
            &req.row_id,
            idx,
            n_launches,
        ));
    }
    let Some(launch) = sidecar.launches.get(idx) else {
        return Err(NcuSourceError::launch_vanished_after_bounds_check(idx));
    };

    let file_matcher = req.file_glob.as_deref().map(glob::compile);

    // No warp sampling captured ⇒ empty-but-well-shaped response.
    let Some(ws) = launch.warp_stalls.as_ref() else {
        return Ok(WarpStallsResponse {
            axis: req.by.label().to_string(),
            count: 0,
            total_matched: 0,
            rows: Vec::new(),
            auxiliary: WarpStallsAuxiliary {
                total_samples: 0,
                not_issued_samples: 0,
                unattributed_samples: 0,
                out_of_cubin_samples: 0,
                per_reason_totals: BTreeMap::new(),
                meta_cache_path: cache_path,
            },
        });
    };

    let unattributed: u64 = ws
        .pcs
        .iter()
        .filter(|p| p.source.is_none())
        .map(pc_total)
        .sum();

    let aux = WarpStallsAuxiliary {
        total_samples: ws.total_samples,
        not_issued_samples: ws.not_issued_samples,
        unattributed_samples: unattributed,
        out_of_cubin_samples: ws.out_of_cubin_samples,
        per_reason_totals: ws.per_reason_totals.clone(),
        meta_cache_path: cache_path,
    };

    // Keys carry the `launch:<idx>|` prefix for cross-trace joins,
    // matching the `source-metrics` convention.
    let lk = format!("launch:{idx}");
    let rows: Vec<WarpStallsRow> = match req.by {
        Axis::Line => line_rows(&lk, &ws.pcs, file_matcher.as_ref()),
        Axis::Sass => sass_rows(&lk, &ws.pcs, file_matcher.as_ref()),
        Axis::Reason => reason_rows(&lk, &ws.per_reason_totals),
    };

    let total_matched = rows.len();
    let rows: Vec<WarpStallsRow> = rows.into_iter().take(req.limit).collect();

    Ok(WarpStallsResponse {
        axis: req.by.label().to_string(),
        count: rows.len(),
        total_matched,
        rows,
        auxiliary: aux,
    })
}

fn pc_total(p: &NativeWarpStallPc) -> u64 {
    p.reasons.values().copied().sum()
}

/// Merge `src` reason counts into `dst`.
fn merge_into(dst: &mut BTreeMap<String, u64>, src: &BTreeMap<String, u64>) {
    for (k, v) in src {
        let slot = dst.entry(k.clone()).or_insert(0);
        *slot = slot.saturating_add(*v);
    }
}

fn line_rows(
    lk: &str,
    pcs: &[NativeWarpStallPc],
    file: Option<&glob::Matcher>,
) -> Vec<WarpStallsRow> {
    let mut by_line: BTreeMap<(String, u32), BTreeMap<String, u64>> = BTreeMap::new();
    for p in pcs {
        let Some(src) = p.source.as_ref() else {
            continue; // in_cubin_no_source → unattributed (aux), not a line row
        };
        if let Some(m) = file
            && !m.matches(&src.file)
        {
            continue;
        }
        let bucket = by_line.entry((src.file.clone(), src.line)).or_default();
        merge_into(bucket, &p.reasons);
    }
    let mut rows: Vec<WarpStallsLineRow> = by_line
        .into_iter()
        .map(|((file, line), stalls)| WarpStallsLineRow {
            key: format!("{lk}|line:{file}:{line}"),
            file,
            line,
            total_samples: stalls.values().copied().sum(),
            stalls,
        })
        .collect();
    rows.sort_by(|a, b| {
        b.total_samples
            .cmp(&a.total_samples)
            .then_with(|| a.key.cmp(&b.key))
    });
    rows.into_iter().map(WarpStallsRow::Line).collect()
}

fn sass_rows(
    lk: &str,
    pcs: &[NativeWarpStallPc],
    file: Option<&glob::Matcher>,
) -> Vec<WarpStallsRow> {
    let mut rows: Vec<WarpStallsSassRow> = pcs
        .iter()
        .filter(|p| match (file, p.source.as_ref()) {
            (Some(m), Some(src)) => m.matches(&src.file),
            (Some(_), None) => false, // file filter excludes unattributed PCs
            (None, _) => true,
        })
        .map(|p| WarpStallsSassRow {
            key: format!("{lk}|sass:0x{:x}", p.rel_address.unwrap_or(0)),
            rel_address: p.rel_address,
            file: p.source.as_ref().map(|s| s.file.clone()),
            line: p.source.as_ref().map(|s| s.line),
            total_samples: pc_total(p),
            stalls: p.reasons.clone(),
        })
        .collect();
    rows.sort_by(|a, b| {
        b.total_samples
            .cmp(&a.total_samples)
            .then_with(|| a.rel_address.cmp(&b.rel_address))
    });
    rows.into_iter().map(WarpStallsRow::Sass).collect()
}

fn reason_rows(lk: &str, per_reason: &BTreeMap<String, u64>) -> Vec<WarpStallsRow> {
    let mut rows: Vec<WarpStallsReasonRow> = per_reason
        .iter()
        .map(|(reason, &total)| WarpStallsReasonRow {
            key: format!("{lk}|reason:{reason}"),
            reason: reason.clone(),
            total_samples: total,
        })
        .collect();
    rows.sort_by(|a, b| {
        b.total_samples
            .cmp(&a.total_samples)
            .then_with(|| a.reason.cmp(&b.reason))
    });
    rows.into_iter().map(WarpStallsRow::Reason).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context, Result};
    use veloq_core::VeloqDiagnostic;

    #[test]
    fn axis_parse_error_is_typed() -> Result<()> {
        let err = Axis::parse("pc").err().context("axis parse should fail")?;
        assert_eq!(err.code().as_str(), "ncu.command.unknown-warp-stalls-axis");
        Ok(())
    }
}
