//! ncu_report-native NCU model.
//!
//! The deserialized shape of the sidecar produced by
//! `scripts/ncu_export.py` (which drives NVIDIA's public `ncu_report`
//! Python API). The serde structs in this module ARE the helper↔Rust
//! field contract — the single source of truth.
//! Placement routing is on [`Placement`]; per-counter additivity in
//! [`crate::source_metrics`].
//!
//! The Rust ingest path ([`crate::native::cache`]) deserializes a
//! gzipped instance of this model from `<report>.veloq/ncu-native.json.gz`.

pub mod cache;
pub mod cubin;

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Schema tag the helper stamps; bump in lockstep with breaking changes
/// to the structs below and to `ncu_export.py`. The `metric_type` /
/// `metric_subtype` / `rollup` fields carry `ncu_report`'s enum *names*
/// (resolved Python-side from the live enum) rather than the
/// version-specific integer codes.
pub const NATIVE_SCHEMA: &str = "ncu-native-v1";

/// Top-level sidecar payload — the helper's stdout, deserialized.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NativeSidecar {
    /// Always [`NATIVE_SCHEMA`] for a sidecar this build understands.
    pub schema: String,
    /// `ncu_report.IContext.get_version()` — e.g. `"2026.1.1"`.
    pub ncu_version: String,
    pub session: NativeSession,
    pub launches: Vec<NativeLaunch>,
    /// Non-KERNEL workloads (RANGE / GRAPH), one array per `ncu_report`
    /// workload type. CMDLIST (OptiX
    /// command lists) is not ingested — out of veloq's CUDA scope.
    /// `#[serde(default, skip_serializing_if)]` so
    /// a KERNEL-only report emits no `ranges` / `graphs` keys.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ranges: Vec<NativeWorkload>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub graphs: Vec<NativeWorkload>,
    /// Set to `"degraded"` by the helper when it could not build a
    /// `metric_type` / `metric_subtype` / `rollup` reverse-map from the
    /// live `ncu_report` enum — the signature of a renamed/relocated enum
    /// container (a structural API change). Absent on a healthy sidecar.
    /// Converts what would be silent misclassification into a visible
    /// signal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub classification: Option<String>,
}

/// A non-KERNEL workload (RANGE / GRAPH). Lightweight vs
/// [`NativeLaunch`]: the list verbs surface only headline columns —
/// `name`, recovered `context_id` / `device_id` / `stream_id` (from
/// the `launch__*` metrics), and metric / rule counts. There is no
/// file-block framing (`block_index` / `thread_id`) and no section
/// catalog (`section_count`), as for launches.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NativeWorkload {
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

/// Degraded session block: `ncu_report` exposes only the NCU
/// version, not host/target/cmdline. Kept as a struct so a future API
/// that surfaces more can extend it without a wire reshape.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NativeSession {
    pub versions: Vec<NativeVersion>,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NativeVersion {
    pub provider: String,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NativeLaunch {
    pub kernel_demangled: String,
    pub kernel_mangled: String,
    pub kernel_function: String,
    pub grid_size: [u64; 3],
    pub block_size: [u64; 3],
    pub stream_id: Option<u64>,
    /// Runtime VA where this kernel's cubin loaded; `min(correlation_id)`
    /// over source-attributable instances. `None` when no source-
    /// correlated metric instances exist on the launch. SASS / instance
    /// `rel_address` values are `abs_pc - cubin_load_base`.
    pub cubin_load_base: Option<u64>,
    pub metrics: Vec<NativeMetric>,
    /// Rule findings, passed through verbatim from
    /// `ncu_report.IAction.rule_results_as_dicts()`. Kept as raw JSON
    /// rather than a fixed schema: the dict shape (focus_metrics,
    /// rule_message{message,title,type}, speedup_estimation, …) is NCU's
    /// and varies across versions; the `rules` verb projects what it needs.
    pub rules: Vec<serde_json::Value>,
    /// Full SASS listing (16-byte stride from `cubin_load_base`).
    /// `None` when `cubin_load_base` is `None`.
    pub disasm: Option<Vec<NativeInsn>>,
    /// Per-launch warp-stall histogram from `timed_warp_samples()`.
    /// `None` when no warp-state sampling was captured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warp_stalls: Option<NativeWarpStalls>,
}

/// Aggregated `timed_warp_samples()` for one launch: the
/// raw ~10^5-sample periodic warp-state stream collapsed to a per-PC ×
/// stall-reason count histogram. Per-sample timestamps are dropped.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NativeWarpStalls {
    /// `len(timed_warp_samples())` — the reconciliation total.
    pub total_samples: u64,
    /// Samples where the SM issued no warp (`not_issued == true`). A
    /// subset of `total_samples`, not part of the placement partition.
    pub not_issued_samples: u64,
    /// Samples whose `pc` is not a PC in this launch's cubin
    /// (`sass_by_pc(pc) == ""`).
    pub out_of_cubin_samples: u64,
    /// Sample count per `StallReason` name, over all samples.
    pub per_reason_totals: BTreeMap<String, u64>,
    /// One entry per distinct in-cubin sampled PC (attributed +
    /// `in_cubin_no_source`); `out_of_cubin` PCs fold into
    /// `out_of_cubin_samples`.
    pub pcs: Vec<NativeWarpStallPc>,
}

/// One in-cubin sampled PC's stall-reason breakdown.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NativeWarpStallPc {
    /// `pc - cubin_load_base`. `None` only when the base is unknown.
    pub rel_address: Option<u64>,
    /// `(file, line)` from `source_info(pc)`; `None` ⇒ `in_cubin_no_source`
    /// (a real PC with no DWARF line — routes to `unattributed`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<NativeSourceRef>,
    /// Sample count per `StallReason` name at this PC.
    pub reasons: BTreeMap<String, u64>,
}

/// `ncu_report.IMetric.metric_type()`, carried as the enum *name* the
/// helper resolves from the live `ncu_report` enum — not the
/// version-specific integer code. A name a future
/// `ncu` adds deserializes to [`MetricType::Unknown`] (→ the additivity
/// name-suffix fallback), never a hard error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MetricType {
    Counter,
    Throughput,
    Ratio,
    Other,
    /// Any `metric_type` name this build doesn't recognise.
    #[serde(other)]
    Unknown,
}

/// `ncu_report.IMetric.metric_subtype()` name. Only the subtypes the
/// additivity classifier cares about are enumerated; every other subtype
/// (and any future addition) lands on [`MetricSubtype::Unknown`], which
/// the classifier treats as "no decisive subtype" (falls through to the
/// `metric_type` check), exactly as a non-`{pct,ratio,per_second}` code did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MetricSubtype {
    Pct,
    Ratio,
    PerSecond,
    #[serde(other)]
    Unknown,
}

/// `ncu_report.IMetric.rollup_operation()` name. Only `Sum` gates
/// additivity; every other operation (and any future addition) lands on
/// [`RollupOp::Unknown`] and is treated as non-`Sum` (non-additive for a
/// counter), exactly as a non-`SUM` code was.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RollupOp {
    Sum,
    Avg,
    Min,
    Max,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NativeMetric {
    pub name: String,
    pub label: Option<String>,
    pub unit: Option<String>,
    /// Aggregate value typed by `value_type` — number or string.
    pub value: serde_json::Value,
    pub value_type: String,
    /// `ncu_report.IMetric.metric_type()` name (COUNTER/THROUGHPUT/RATIO/
    /// OTHER) — an additivity input. Name, not integer code:
    /// version-robust because the helper resolves it
    /// against the installed `ncu`'s live enum.
    pub metric_type: MetricType,
    /// Raw `metric_type()` integer code, retained as provenance so a
    /// renumber is detectable (the drift test compares codes). Absent
    /// only when the source lacked it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric_type_code: Option<i64>,
    /// `metric_subtype()` name (PCT / RATIO / PER_SECOND / …); `None` when
    /// the metric has no subtype.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric_subtype: Option<MetricSubtype>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric_subtype_code: Option<i64>,
    /// `rollup_operation()` name (SUM/AVG/MIN/MAX/…); `None` when the
    /// metric carries no rollup operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollup: Option<RollupOp>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollup_code: Option<i64>,
    /// Per-PC instances; present only on metrics carrying correlation IDs
    /// (the pcsamp / source-counter families). Absent otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instances: Option<Vec<NativeInstance>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NativeInstance {
    /// Absolute SASS PC the sample is correlated to.
    pub correlation_id: u64,
    /// `correlation_id - cubin_load_base`, or `None` when out of cubin /
    /// no base.
    pub rel_address: Option<u64>,
    pub value: serde_json::Value,
    pub placement: Placement,
}

/// Where a per-PC instance's `correlation_id` lands — the routing
/// computed by the positive-evidence gate and pre-tagged by the
/// helper. Closed vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Placement {
    /// `source_info(cid)` resolved to a (file, line).
    Attributed,
    /// A real PC in the cubin with no DWARF source line.
    InCubinNoSource,
    /// `cid` is not a PC in this cubin (e.g. the `warpsampling:` family's
    /// packed non-VA correlations).
    OutOfCubin,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NativeInsn {
    /// Cubin-relative offset (`abs_pc - cubin_load_base`).
    pub address: u64,
    pub opcode: String,
    pub operands: String,
    pub source: Option<NativeSourceRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NativeSourceRef {
    pub file: String,
    pub line: u32,
}

impl NativeLaunch {
    /// First metric on this launch with the exact `name`. `ncu_report`
    /// surfaces launch / device attributes (context, device, stream,
    /// compute capability) as ordinary metrics, so the kernel-path
    /// verbs recover those columns by name from here.
    pub fn metric(&self, name: &str) -> Option<&NativeMetric> {
        self.metrics.iter().find(|m| m.name == name)
    }

    /// Scalar metric value coerced to `u64` (the uint / double metric
    /// families serde-decode through `as_u64`). `None` when the metric
    /// is absent or not numeric.
    pub fn metric_u64(&self, name: &str) -> Option<u64> {
        self.metric(name).and_then(|m| m.value.as_u64())
    }

    /// `sm_<major><minor>` arch label from the `device__attribute_
    /// compute_capability_{major,minor}` metrics (e.g. 12 + 0 →
    /// `sm_120`). `None` when the capability metrics weren't captured.
    pub fn sm_label(&self) -> Option<String> {
        let major = self.metric_u64("device__attribute_compute_capability_major")?;
        let minor = self.metric_u64("device__attribute_compute_capability_minor")?;
        Some(format!("sm_{major}{minor}"))
    }

    /// `true` when a non-empty SASS listing was captured for the launch.
    pub fn has_disasm(&self) -> bool {
        self.disasm.as_ref().is_some_and(|d| !d.is_empty())
    }
}

/// `ncu summary` totals under the native model.
/// `range`/`graph` counts stay `0` until those workload types are
/// emitted. `cmdlist_count` is absent rather than a misleading
/// constant `0` — CMDLIST is OptiX-only and out of scope.
#[derive(Debug, Default, Serialize, schemars::JsonSchema)]
pub struct NativeTotals {
    pub launch_count: usize,
    pub range_count: usize,
    pub graph_count: usize,
    /// Sum of `metrics.len()` across launches.
    pub metric_count: usize,
    /// Sum of `rules.len()` across launches.
    pub rule_count: usize,
    /// Count of launches carrying a non-empty SASS listing.
    pub kernel_disasm_count: usize,
}

/// `Totals` lifted to a row with the canonical `key`, flattening
/// the totals (see [`NativeTotals`]) under the list contract.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct NativeTotalsRow {
    pub key: String,
    #[serde(flatten)]
    pub totals: NativeTotals,
}

/// Session block for the native `ncu summary`:
/// `ncu_report` exposes only the NCU version, not host / target /
/// cmdline — so no hostnames or install paths leak into output.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct NativeSessionInfo {
    pub versions: Vec<NativeVersion>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct NativeSummaryAuxiliary {
    pub session: NativeSessionInfo,
    /// `ncu_report.IContext.get_version()`.
    pub ncu_version: String,
    /// Path to the native sidecar (`<report>.veloq/ncu-native.json.gz`)
    /// this response was sourced from.
    pub meta_cache_path: String,
}

/// `ncu summary` response under the native model. The list
/// contract is `count` / `total_matched` / `rows` / `auxiliary`.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct NativeSummaryResponse {
    pub count: usize,
    pub total_matched: usize,
    pub rows: Vec<NativeTotalsRow>,
    pub auxiliary: NativeSummaryAuxiliary,
}

impl NativeSidecar {
    /// Aggregate the launch-derived [`NativeTotals`] for `ncu summary`.
    pub fn totals(&self) -> NativeTotals {
        let mut t = NativeTotals {
            launch_count: self.launches.len(),
            range_count: self.ranges.len(),
            graph_count: self.graphs.len(),
            ..NativeTotals::default()
        };
        for w in self.ranges.iter().chain(&self.graphs) {
            t.metric_count = t.metric_count.saturating_add(w.metric_count);
            t.rule_count = t.rule_count.saturating_add(w.rule_count);
        }
        for l in &self.launches {
            t.metric_count = t.metric_count.saturating_add(l.metrics.len());
            t.rule_count = t.rule_count.saturating_add(l.rules.len());
            if l.has_disasm() {
                t.kernel_disasm_count = t.kernel_disasm_count.saturating_add(1);
            }
        }
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Result, anyhow};
    use std::path::Path;

    fn golden() -> Result<NativeSidecar> {
        let p = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/source_metric_basic.ncu-rep.veloq/ncu-native.json.gz");
        cache::read_gz_sidecar(&p)
    }

    #[test]
    fn golden_deserializes_into_native_model() -> Result<()> {
        let sc = golden()?;
        assert_eq!(sc.schema, NATIVE_SCHEMA);
        assert!(sc.launches.len() >= 2, "expected >=2 launches");
        let l0 = sc.launches.first().ok_or_else(|| anyhow!("no launches"))?;
        assert!(!l0.kernel_function.is_empty());
        assert!(!l0.metrics.is_empty());
        Ok(())
    }

    /// The launches round-trip proof + the placement invariant,
    /// NCU-free (reads the committed golden).
    #[test]
    fn launch_one_pcsamp_placement_split_round_trips() -> Result<()> {
        let sc = golden()?;
        let l1 = sc.launches.get(1).ok_or_else(|| anyhow!("no launch:1"))?;
        assert!(
            l1.kernel_function.contains("synthetic_long_stall"),
            "launch:1 should be the pointer-chase kernel, got {:?}",
            l1.kernel_function
        );
        assert!(l1.cubin_load_base.is_some());
        assert!(l1.disasm.as_ref().is_some_and(|d| !d.is_empty()));

        let pcsamp = l1
            .metrics
            .iter()
            .find(|m| m.name == "smsp__pcsamp_warps_issue_stalled_long_scoreboard")
            .ok_or_else(|| anyhow!("pcsamp metric missing on launch:1"))?;
        let insts = pcsamp
            .instances
            .as_ref()
            .ok_or_else(|| anyhow!("pcsamp has no instances"))?;
        assert!(
            insts.iter().all(|i| i.placement == Placement::Attributed),
            "pcsamp instances should all be attributed"
        );

        let warp = l1
            .metrics
            .iter()
            .find(|m| m.name == "warpsampling:smsp__pcsamp_warps_issue_stalled_long_scoreboard")
            .ok_or_else(|| anyhow!("warpsampling family missing"))?;
        let warp_insts = warp
            .instances
            .as_ref()
            .ok_or_else(|| anyhow!("warpsampling has no instances"))?;
        assert!(
            warp_insts
                .iter()
                .all(|i| i.placement == Placement::OutOfCubin),
            "warpsampling family must be out_of_cubin (non-VA correlations)"
        );
        Ok(())
    }
}
