//! `veloq ncu source-metrics` — per-source-line / per-SASS / per-file
//! NCU counter attribution.
//!
//! Joins per-PC `MetricInstance` values with the DWARF
//! source-line attribution from disassembly so agents can ask
//! questions like "which source lines have the most bank conflicts?"
//! without falling back to native `ncu --import T --page source --csv`.
//!
//! Source-attribution comes from the `ncu_report`
//! native sidecar: each per-PC instance carries a `placement` tag
//! (`attributed` / `in_cubin_no_source` / `out_of_cubin`) and line
//! attribution comes from the sidecar's `source_info`.
//!
//! Module layout — two pure submodules + this dispatch entry point:
//!
//! - [`additivity`] decides which counters are safe to SUM across
//!   SASS instructions on a row (`ncu_report` `metric_type` /
//!   `rollup_operation` / `metric_subtype`, suffix fallback).
//! - [`rollup`] performs the per-axis aggregation (`line`, `sass`,
//!   `file`) + the two-bucket unattributed accounting.
//!
//! The submodules take hand-built literals so they're unit-testable
//! without an `.ncu-rep` fixture.

pub mod additivity;
pub mod rollup;
pub mod units;

use crate::disasm_pipeline::SourceLineRef;
use crate::error::{NcuSourceError, NcuSourceResult};
use crate::glob;
use crate::native::{NativeLaunch, NativeMetric, NativeSidecar, Placement, cache};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;

/// Public response payload — `data` body of the envelope.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SourceMetricsResponse {
    /// `"line" | "sass" | "file"` — the active row variant.
    pub axis: &'static str,
    /// Rows returned (after `--limit`).
    pub count: usize,
    /// Rows produced by the rollup before `--limit` clipped them.
    pub total_matched: usize,
    /// Per-axis row collection. `#[serde(untagged)]` — the JSON
    /// shape is one of three documented.
    pub rows: Vec<SourceMetricsRow>,
    pub auxiliary: SourceMetricsAuxiliary,
}

/// A `#[serde(untagged)]` enum
/// discriminated by request at the verb level. The `axis` tag on the
/// envelope's `data` block (above) is the authoritative discriminator
/// for agents that can't pattern-match on field presence.
#[derive(Debug, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum SourceMetricsRow {
    Line(SourceMetricsLineRow),
    Sass(SourceMetricsSassRow),
    File(SourceMetricsFileRow),
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SourceMetricsLineRow {
    pub key: String,
    pub launch_row_id: String,
    pub file: String,
    pub line: u32,
    /// Sorted ascending, deduped — matches
    /// `DisasmAuxiliary.source_index[].sass_addresses`.
    pub sass_addresses: Vec<u64>,
    /// `{ start, end }` derivation over `sass_addresses` (min/max).
    /// Convenience field; `sass_addresses` is the canonical
    /// representation for filtered drills.
    pub sass_address_range: SassAddressRange,
    pub sass_count: u32,
    /// `BTreeMap<String, Option<f64>>`: `None` when no instance for
    /// this counter overlapped the row's SASS addresses; `Some(0.0)`
    /// when the counter applied but values summed to zero.
    pub counters: BTreeMap<String, Option<f64>>,
    /// Same key set as `counters` (invariant 3). Per-counter count
    /// of distinct SASS addresses on this row that had ≥1 instance.
    pub counter_coverage: BTreeMap<String, u32>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SourceMetricsSassRow {
    pub key: String,
    pub launch_row_id: String,
    /// Cubin-relative SASS address — matches
    /// `SassInstruction.address`.
    pub address: u64,
    pub opcode: String,
    pub operands: String,
    /// `None` for DWARF holes / compiler-inserted code.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceLineRef>,
    pub counters: BTreeMap<String, Option<f64>>,
    pub counter_coverage: BTreeMap<String, u32>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SourceMetricsFileRow {
    pub key: String,
    pub launch_row_id: String,
    pub file: String,
    pub line_count: u32,
    pub sass_count: u32,
    pub counters: BTreeMap<String, Option<f64>>,
    pub counter_coverage: BTreeMap<String, u32>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SassAddressRange {
    pub start: u64,
    pub end: u64,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SourceMetricsAuxiliary {
    pub row_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_demangled: Option<String>,
    pub counter_glob: String,
    /// Counters that matched `--counter` AND are source-attributable —
    /// `ncu_report` tagged ≥1 per-PC instance in-cubin (placement !=
    /// `out_of_cubin`). Empty when no match.
    pub matched_counters: Vec<String>,
    /// Counters matching `--counter` that were not source-attributable
    /// (no in-cubin per-PC instances) OR that the additivity rule
    /// rejected on line/file axes. Each carries a structured `reason`.
    pub skipped_counters: Vec<SkippedCounter>,
    /// Per-counter sum for in-cubin instances whose SASS instruction
    /// had `source: None` (DWARF hole). Dropped from `data.rows[]`
    /// on `--by line`/`--by file`; appears as a `source: null` row
    /// on `--by sass`.
    pub unattributed_sass_counter_totals: BTreeMap<String, f64>,
    /// Per-counter sum for instances whose `correlation_id` fell
    /// outside the cubin's `[load_base, load_base + length)`. Dropped
    /// from `data.rows[]` entirely.
    pub out_of_cubin_counter_totals: BTreeMap<String, f64>,
    /// Distinct in-cubin-no-source instance count summed into
    /// `unattributed_sass_counter_totals`.
    pub unattributed_sass_count: u32,
    /// Distinct out-of-cubin instance count summed into
    /// `out_of_cubin_counter_totals`.
    pub out_of_cubin_instance_count: u32,
    /// Free-form warnings (e.g. "report has no SourceCounters
    /// section" when no counters match).
    pub warnings: Vec<String>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SkippedCounter {
    pub name: String,
    pub reason: &'static str,
}

/// Request shape carried through to the runner.
#[derive(Debug, Clone)]
pub struct SourceMetricsRequest {
    pub row_id: String,
    pub counter_glob: String,
    pub by: Axis,
    pub file_glob: Option<String>,
    pub line: Option<u32>,
    pub sort: Option<String>,
    pub limit: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    Line,
    Sass,
    File,
}

impl Axis {
    pub fn parse(s: &str) -> NcuSourceResult<Self> {
        match s {
            "line" => Ok(Axis::Line),
            "sass" => Ok(Axis::Sass),
            "file" => Ok(Axis::File),
            other => Err(NcuSourceError::unknown_source_metrics_axis(other)),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Axis::Line => "line",
            Axis::Sass => "sass",
            Axis::File => "file",
        }
    }
}

/// Open the report (disasm-enabled walk), resolve the launch, run the
/// gate + rollup, and project the result into the wire shape.
pub fn run<P: AsRef<Path>>(
    path: P,
    req: SourceMetricsRequest,
) -> NcuSourceResult<SourceMetricsResponse> {
    validate_request(&req)?;
    let sidecar = cache::build_or_load(path.as_ref())?;
    run_on_sidecar(&sidecar, req)
}

/// Core source-metrics logic over an already-loaded native sidecar.
/// Split from [`run`] so tests can drive it from a committed golden
/// sidecar via [`crate::native::cache::read_gz_sidecar`] — the source
/// `.ncu-rep` is deliberately *not* committed (the binary embeds a
/// hostname; the native sidecar keeps only a version-only session
/// block — no host identifiers — so it is leak-free).
pub fn run_on_sidecar(
    sidecar: &NativeSidecar,
    req: SourceMetricsRequest,
) -> NcuSourceResult<SourceMetricsResponse> {
    validate_request(&req)?;
    let idx = crate::row_id::parse_launch_idx(&req.row_id)?;

    // Out-of-range index — error consistent with `disasm`.
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

    // The per-PC source attribution + placement tag
    // come straight off the sidecar (`ncu_report` source_info). A
    // launch with no source-correlated instances has no
    // `cubin_load_base` and nothing to attribute.
    if launch.cubin_load_base.is_none() {
        return Ok(empty_response(
            &req,
            launch,
            "launch has no cubin_load_base (no source-correlated metric instances present); \
             recapture with `--set source` or `--set full` to enable SourceCounters."
                .to_string(),
        ));
    }
    // The launch's SASS listing carries `ncu_report`'s `source_info`
    // per instruction — the authoritative line attribution (see [`rollup`]).
    let Some(disasm) = launch.disasm.as_ref().filter(|d| !d.is_empty()) else {
        return Ok(empty_response(
            &req,
            launch,
            "launch has no SASS listing (cubin is zero-byte or carried no instructions)"
                .to_string(),
        ));
    };

    // Compile the --counter glob — comma-separated; matches via OR.
    let counter_matchers: Vec<glob::Matcher> = req
        .counter_glob
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(glob::compile)
        .collect();
    if counter_matchers.is_empty() {
        return Err(NcuSourceError::counter_glob_empty());
    }
    let matches_any = |name: &str| counter_matchers.iter().any(|m| m.matches(name));

    // Pass 1: classify each requested metric. A counter is source-
    // attributable iff `ncu_report` tagged ≥1 of its per-PC instances
    // in-cubin (placement != out_of_cubin); additivity re-homes onto
    // the sidecar's classification with the suffix rule as fallback.
    let mut matched: Vec<(rollup::CounterSpec<'_>, &NativeMetric)> = Vec::new();
    let mut matched_counter_names: Vec<String> = Vec::new();
    let mut skipped: Vec<SkippedCounter> = Vec::new();

    for metric in &launch.metrics {
        let name = metric.name.as_str();
        if !matches_any(name) {
            continue;
        }
        if !is_source_attributable(metric) {
            skipped.push(SkippedCounter {
                name: name.to_string(),
                reason: "not-a-source-counter",
            });
            continue;
        }
        let additive = additivity::is_additive_native(metric);
        // Non-additive counters on line/file axes go to skipped;
        // on sass axis they pass through.
        if !additive && matches!(req.by, Axis::Line | Axis::File) {
            skipped.push(SkippedCounter {
                name: name.to_string(),
                reason: "non-additive-rollup",
            });
            continue;
        }
        matched.push((rollup::CounterSpec { name, additive }, metric));
        matched_counter_names.push(name.to_string());
    }

    let mut warnings = Vec::new();
    if matched.is_empty() && skipped.is_empty() {
        warnings.push(format!(
            "no metric names matched `--counter {}` for this launch",
            req.counter_glob
        ));
    }

    let rollup_output = rollup::rollup(rollup::RollupInput {
        counters: &matched,
        disasm,
    });

    // Apply axis-specific filters + project to the wire shape.
    let launch_row_id = req.row_id.clone();
    let (rows, total_matched) = match req.by {
        Axis::Line => project_line(&rollup_output, &req, &launch_row_id),
        Axis::Sass => project_sass(&rollup_output, &req, &launch_row_id),
        Axis::File => project_file(&rollup_output, &req, &launch_row_id),
    };

    let count = rows.len();

    Ok(SourceMetricsResponse {
        axis: req.by.label(),
        count,
        total_matched,
        rows,
        auxiliary: SourceMetricsAuxiliary {
            row_id: req.row_id,
            kernel_demangled: Some(launch.kernel_demangled.clone()).filter(|s| !s.is_empty()),
            counter_glob: req.counter_glob,
            matched_counters: matched_counter_names,
            skipped_counters: skipped,
            unattributed_sass_counter_totals: rollup_output.unattributed.unattributed_sass,
            out_of_cubin_counter_totals: rollup_output.unattributed.out_of_cubin,
            unattributed_sass_count: rollup_output.unattributed.unattributed_sass_count,
            out_of_cubin_instance_count: rollup_output.unattributed.out_of_cubin_instance_count,
            warnings,
        },
    })
}

fn validate_request(req: &SourceMetricsRequest) -> NcuSourceResult<()> {
    if req.limit == 0 {
        return Err(NcuSourceError::limit_too_small(req.limit));
    }
    if counter_glob_is_empty(&req.counter_glob) {
        return Err(NcuSourceError::counter_glob_empty());
    }
    if req.line.is_some() && req.file_glob.is_none() {
        return Err(NcuSourceError::SourceMetricsLineWithoutFile);
    }
    crate::row_id::parse_launch_idx(&req.row_id)?;
    Ok(())
}

fn counter_glob_is_empty(counter: &str) -> bool {
    !counter.split(',').any(|part| !part.trim().is_empty())
}

/// `true` when `ncu_report` tagged at least one of this metric's
/// per-PC instances as in-cubin (placement != out_of_cubin) — the
/// positive-evidence gate.
fn is_source_attributable(metric: &NativeMetric) -> bool {
    metric
        .instances
        .as_ref()
        .is_some_and(|insts| insts.iter().any(|i| i.placement != Placement::OutOfCubin))
}

fn empty_response(
    req: &SourceMetricsRequest,
    launch: &NativeLaunch,
    reason: String,
) -> SourceMetricsResponse {
    SourceMetricsResponse {
        axis: req.by.label(),
        count: 0,
        total_matched: 0,
        rows: Vec::new(),
        auxiliary: SourceMetricsAuxiliary {
            row_id: req.row_id.clone(),
            kernel_demangled: Some(launch.kernel_demangled.clone()).filter(|s| !s.is_empty()),
            counter_glob: req.counter_glob.clone(),
            matched_counters: Vec::new(),
            skipped_counters: Vec::new(),
            unattributed_sass_counter_totals: BTreeMap::new(),
            out_of_cubin_counter_totals: BTreeMap::new(),
            unattributed_sass_count: 0,
            out_of_cubin_instance_count: 0,
            warnings: vec![reason],
        },
    }
}

fn project_line(
    out: &rollup::RollupOutput,
    req: &SourceMetricsRequest,
    launch_row_id: &str,
) -> (Vec<SourceMetricsRow>, usize) {
    let mut rows: Vec<&rollup::LineRow> = out.line_rows.iter().collect();
    // --file / --line filters applied here, before sort + limit.
    if let Some(glob_pat) = req.file_glob.as_deref() {
        let matcher = glob::compile(glob_pat);
        rows.retain(|r| matcher.matches(&r.file));
    }
    if let Some(line) = req.line {
        rows.retain(|r| r.line == line);
    }
    let total = rows.len();
    sort_line_rows(&mut rows, req.sort.as_deref());
    rows.truncate(req.limit);
    let projected: Vec<SourceMetricsRow> = rows
        .into_iter()
        .map(|r| {
            let (start, end) = if r.sass_addresses.is_empty() {
                (0, 0)
            } else {
                (
                    *r.sass_addresses.first().unwrap_or(&0),
                    *r.sass_addresses.last().unwrap_or(&0),
                )
            };
            SourceMetricsRow::Line(SourceMetricsLineRow {
                key: format!("{launch_row_id}|line:{}:{}", r.file, r.line),
                launch_row_id: launch_row_id.to_string(),
                file: r.file.clone(),
                line: r.line,
                sass_addresses: r.sass_addresses.clone(),
                sass_address_range: SassAddressRange { start, end },
                sass_count: r.sass_addresses.len() as u32,
                counters: r.counters.clone(),
                counter_coverage: r.counter_coverage.clone(),
            })
        })
        .collect();
    (projected, total)
}

fn project_sass(
    out: &rollup::RollupOutput,
    req: &SourceMetricsRequest,
    launch_row_id: &str,
) -> (Vec<SourceMetricsRow>, usize) {
    let mut rows: Vec<&rollup::SassRow> = out.sass_rows.iter().collect();
    if let Some(glob_pat) = req.file_glob.as_deref() {
        let matcher = glob::compile(glob_pat);
        rows.retain(|r| match &r.source {
            Some(s) => matcher.matches(&s.file),
            None => false,
        });
    }
    if let Some(line) = req.line {
        rows.retain(|r| match &r.source {
            Some(s) => s.line == line,
            None => false,
        });
    }
    let total = rows.len();
    sort_sass_rows(&mut rows, req.sort.as_deref());
    rows.truncate(req.limit);
    let projected: Vec<SourceMetricsRow> = rows
        .into_iter()
        .map(|r| {
            SourceMetricsRow::Sass(SourceMetricsSassRow {
                key: format!("{launch_row_id}|sass:0x{:x}", r.address),
                launch_row_id: launch_row_id.to_string(),
                address: r.address,
                opcode: r.opcode.clone(),
                operands: r.operands.clone(),
                source: r.source.clone(),
                counters: r.counters.clone(),
                counter_coverage: r.counter_coverage.clone(),
            })
        })
        .collect();
    (projected, total)
}

fn project_file(
    out: &rollup::RollupOutput,
    req: &SourceMetricsRequest,
    launch_row_id: &str,
) -> (Vec<SourceMetricsRow>, usize) {
    let mut rows: Vec<&rollup::FileRow> = out.file_rows.iter().collect();
    if let Some(glob_pat) = req.file_glob.as_deref() {
        let matcher = glob::compile(glob_pat);
        rows.retain(|r| matcher.matches(&r.file));
    }
    let total = rows.len();
    sort_file_rows(&mut rows, req.sort.as_deref());
    rows.truncate(req.limit);
    let projected: Vec<SourceMetricsRow> = rows
        .into_iter()
        .map(|r| {
            SourceMetricsRow::File(SourceMetricsFileRow {
                key: format!("{launch_row_id}|file:{}", r.file),
                launch_row_id: launch_row_id.to_string(),
                file: r.file.clone(),
                line_count: r.line_count,
                sass_count: r.sass_count,
                counters: r.counters.clone(),
                counter_coverage: r.counter_coverage.clone(),
            })
        })
        .collect();
    (projected, total)
}

/// Parse a sort spec of the form `<counter>:asc|desc`. When no spec
/// is supplied, sort by the first counter (alphabetically first in
/// BTreeMap order) descending. Secondary key per axis keeps ties
/// deterministic.
fn sort_line_rows(rows: &mut Vec<&rollup::LineRow>, spec: Option<&str>) {
    sort_rows_by_counter(
        rows,
        spec,
        |r| &r.counters,
        |a, b| a.file.cmp(&b.file).then_with(|| a.line.cmp(&b.line)),
    );
}

fn sort_sass_rows(rows: &mut Vec<&rollup::SassRow>, spec: Option<&str>) {
    sort_rows_by_counter(
        rows,
        spec,
        |r| &r.counters,
        |a, b| a.address.cmp(&b.address),
    );
}

fn sort_file_rows(rows: &mut Vec<&rollup::FileRow>, spec: Option<&str>) {
    sort_rows_by_counter(rows, spec, |r| &r.counters, |a, b| a.file.cmp(&b.file));
}

/// Shared counter-value sort for the source-metrics axis rows. Sorts by
/// the `--sort` counter (descending by default; the first counter in
/// `BTreeMap` order when unspecified), with missing values sinking to
/// `NEG_INFINITY`, then by the per-axis `tiebreak`. Replaces the three
/// byte-identical `sort_{line,sass,file}_rows` bodies that differed only
/// in their tiebreak.
fn sort_rows_by_counter<R>(
    rows: &mut [&R],
    spec: Option<&str>,
    counters_of: impl Fn(&R) -> &std::collections::BTreeMap<String, Option<f64>>,
    tiebreak: impl Fn(&R, &R) -> std::cmp::Ordering,
) {
    let default = rows
        .first()
        .copied()
        .and_then(|r| counters_of(r).keys().next().cloned());
    let (counter, asc) = parse_sort_spec(spec, default);
    rows.sort_by(|&a, &b| {
        let val = |r: &R| {
            counter
                .as_ref()
                .and_then(|c| counters_of(r).get(c).copied().flatten())
                .unwrap_or(f64::NEG_INFINITY)
        };
        let (av, bv) = (val(a), val(b));
        let primary = if asc {
            av.partial_cmp(&bv)
        } else {
            bv.partial_cmp(&av)
        }
        .unwrap_or(std::cmp::Ordering::Equal);
        primary.then_with(|| tiebreak(a, b))
    });
}

fn parse_sort_spec(spec: Option<&str>, default: Option<String>) -> (Option<String>, bool) {
    match spec {
        Some(s) if !s.is_empty() => {
            let (name, dir) = match s.split_once(':') {
                Some((n, d)) => (n.to_string(), d),
                None => (s.to_string(), "desc"),
            };
            (Some(name), dir.eq_ignore_ascii_case("asc"))
        }
        _ => (default, false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context, Result};
    use veloq_core::VeloqDiagnostic;

    #[test]
    fn axis_parse_error_is_typed() -> Result<()> {
        let err = Axis::parse("pc").err().context("axis parse should fail")?;
        assert_eq!(
            err.code().as_str(),
            "ncu.command.unknown-source-metrics-axis"
        );
        Ok(())
    }
}
