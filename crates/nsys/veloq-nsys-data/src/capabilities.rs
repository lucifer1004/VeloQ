//! Trace-capability self-description.
//!
//! Agents asking veloq about a trace usually want to know *what
//! questions are answerable* before issuing a heavy query. The
//! [`CapabilityFlags`] struct answers that: a boolean bitmap of
//! "which event tables exist" plus a handful of "what extra analyses
//! are possible" markers. Cheap to compute (each flag is a
//! `SELECT 1 ... LIMIT 0` probe), surfaced through `summary` so a
//! single `veloq summary` call tells an agent the supported
//! surface area.
//!
//! ## Curated, not exhaustive
//!
//! NSys exposes ~40 `Supports*` / `Has*` keys in
//! `TARGET_INFO_SYSTEM_ENV` (`SupportsLBR`, `IsGameRecordingModeOn`,
//! `SupportsXmcClients`, …). Most are capture-time properties of the
//! profiling host — they don't influence what veloq can *answer*.
//! Keep this list focused on flags that gate concrete agent
//! decisions:
//!
//! - which event kinds are queryable (`has_kernels` etc.)
//! - which command paths actually have data (`has_cuda_contexts` for
//!   runtime correlation, `has_nvtx` for `slices`/`--nvtx`)
//! - what optional analyses *would* succeed (`has_sampling`,
//!   `has_gpu_metrics`, `has_nic_metrics`)
//! - whether `hardware` will return anything (`has_target_info`)
//!
//! Future PRs can extend the struct additively; the JSON contract
//! ignores unknown keys on the consumer side (documented agent
//! behaviour), so adding a flag never breaks an existing agent.

use serde::{Deserialize, Serialize};

/// What this trace exposes. Every field is `false` by default
/// (`#[derive(Default)]` plus `#[serde(default)]` on the struct lets
/// it round-trip through serde even when a future version of veloq
/// writes the cache with a wider set of flags — unknown fields stay
/// false on deserialise, and missing fields default rather than
/// failing the parse).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct CapabilityFlags {
    // ---- Event tables ----
    /// `CUPTI_ACTIVITY_KIND_KERNEL` present — `stats`/`search` for
    /// `--type kernel` will return data.
    pub has_kernels: bool,
    /// `CUPTI_ACTIVITY_KIND_MEMCPY` present.
    pub has_memcpy: bool,
    /// `CUPTI_ACTIVITY_KIND_MEMSET` present.
    pub has_memset: bool,
    /// `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION` present — `stats --type
    /// sync` aggregates `cudaStreamSynchronize` et al.
    pub has_sync: bool,
    /// `CUPTI_ACTIVITY_KIND_RUNTIME` present — CPU-side CUDA API
    /// calls; required for `correlate` and `slices` attribution
    /// walks.
    pub has_runtime: bool,
    /// `OSRT_API` present — POSIX/OS runtime calls (pthread, poll, …).
    pub has_osrt: bool,
    /// `NVTX_EVENTS` present — required for `slices`, `--nvtx`, and
    /// the nesting-depth surfaces.
    pub has_nvtx: bool,
    /// `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` present — CUDA graph launches
    /// (one row per graph execution). When a workload captures work into
    /// graphs (`--cuda-graph-trace=graph`), kernels-inside-graphs are
    /// rolled up into these rows and do *not* appear in
    /// `CUPTI_ACTIVITY_KIND_KERNEL`; graph rows carry the only
    /// per-execution timing for that work.
    pub has_graph_trace: bool,
    /// `CUDA_GRAPH_NODE_EVENTS` present — per-node-within-a-graph
    /// metadata. Set when the workload was captured with
    /// `--cuda-graph-trace=node`. In that mode kernels-inside-graphs
    /// *are* present in `CUPTI_ACTIVITY_KIND_KERNEL` with `graphId` and
    /// `graphNodeId` populated, and this table holds per-node creation
    /// metadata. Mutually exclusive in practice with `has_graph_trace`
    /// — NSys produces one or the other depending on the capture flag.
    pub has_graph_nodes: bool,
    /// `CUDA_GRAPH_EVENTS` present — host-side graph construction
    /// events (`Graph Creation` / `GraphExec Creation`), captured via
    /// NSys's CUDA API hook layer. Distinct from `has_graph_trace`
    /// (which is GPU-side CUPTI execution data); both can coexist on
    /// the same trace in `=graph` mode.
    pub has_graph_events: bool,
    /// `CUPTI_ACTIVITY_KIND_CUDA_EVENT` present — every `cudaEventRecord`
    /// placement recorded as an instantaneous activity row. Pair via
    /// `eventSyncId` with the matching `cudaEventSynchronize` /
    /// `cudaStreamWaitEvent` in `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION`
    /// for cross-stream causality analysis.
    pub has_cuda_event_activity: bool,
    /// `CUPTI_ACTIVITY_KIND_OVERHEAD` present — CUPTI's own profiling
    /// overhead. Useful trust signal: "how much of this trace's
    /// duration is the profiler observing itself."
    pub has_overhead: bool,

    // ---- Index-supporting tables ----
    /// `TARGET_INFO_CUDA_CONTEXT_INFO` present — drives the
    /// `(device, context) ↔ processId` bridge used by `correlate`
    /// when the input is a runtime API row. Without it, runtime-side
    /// lookups fall back to a PID-only path and may miss matches on
    /// multi-context processes.
    pub has_cuda_contexts: bool,

    // ---- Optional / future-facing ----
    /// `SAMPLING_CALLCHAINS` present — per-sample stack frames.
    /// Required by `inspect cpu_sample:N` for the resolved callchain,
    /// and by `veloq metrics --type cpu-sampling --group-by symbol` /
    /// `module` for symbol attribution.
    pub has_sampling: bool,
    /// `COMPOSITE_EVENTS` present — CPU IP samples
    /// `(id, start, cpu, threadState, globalTid, cpuCycles)`. Pair
    /// with `has_sampling` to know whether the per-sample callchains
    /// are also present; in practice nsys writes both together when
    /// `--sample` is enabled at capture.
    pub has_composite_events: bool,
    /// `SCHED_EVENTS` present — context-switch transition stream
    /// (`sched-in` / `sched-out` events per `(cpu, globalTid)`).
    /// Required by `veloq metrics --type cpu-sched`. Populated by
    /// `nsys profile --cpuctxsw=process-tree` (or `system-wide`).
    pub has_sched_events: bool,
    /// `GPU_METRICS` + `TARGET_INFO_GPU_METRICS` (counter dictionary)
    /// both present — `veloq metrics --type gpu` has every table it
    /// requires. Set only when the full set is queryable so an agent
    /// can treat this flag as a green light, not just raw
    /// table-presence.
    pub has_gpu_metrics: bool,
    /// `NET_NIC_METRIC` + `TARGET_INFO_NETWORK_METRICS` (counter
    /// dictionary) + `NIC_ID_MAP` + `TARGET_INFO_NIC_INFO` all
    /// present — `veloq metrics --type nic` has every table it
    /// requires. Captured with `nsys profile --nic-metrics=lf` or
    /// `hf`. Set only when the full set is queryable so an agent can
    /// treat this flag as a green light, not just raw table-presence.
    pub has_nic_metrics: bool,
    /// `TARGET_INFO_SYSTEM_ENV` present — drives `veloq hardware`
    /// and the `summary.capabilities` field itself (flags below this
    /// point would all be `false` without it, but probing is cheap
    /// enough that we always run the full set).
    pub has_target_info: bool,
}

impl CapabilityFlags {
    /// Build a capability map by probing the attached trace. All
    /// probes are `SELECT 1 ... LIMIT 0` queries — cheap enough that
    /// callers can run this on every `Trace::open` without a sidecar
    /// cache. The result is also persisted in the metadata cache so
    /// the probe cost vanishes on warm calls.
    ///
    /// Takes a parquetdir path rather than a `Trace` so callers like
    /// `info` can probe without paying the DuckDB-open cost. The
    /// probe is just `<pqtdir>/<TABLE>.parquet`
    /// stat checks.
    pub fn extract(pqtdir: &std::path::Path) -> Self {
        let has = |t: &str| crate::adapter::table_exists(pqtdir, t);
        Self {
            has_kernels: has("CUPTI_ACTIVITY_KIND_KERNEL"),
            has_memcpy: has("CUPTI_ACTIVITY_KIND_MEMCPY"),
            has_memset: has("CUPTI_ACTIVITY_KIND_MEMSET"),
            has_sync: has("CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"),
            has_runtime: has("CUPTI_ACTIVITY_KIND_RUNTIME"),
            has_osrt: has("OSRT_API"),
            has_nvtx: has("NVTX_EVENTS"),
            has_graph_trace: has("CUPTI_ACTIVITY_KIND_GRAPH_TRACE"),
            has_graph_nodes: has("CUDA_GRAPH_NODE_EVENTS"),
            has_graph_events: has("CUDA_GRAPH_EVENTS"),
            has_cuda_event_activity: has("CUPTI_ACTIVITY_KIND_CUDA_EVENT"),
            has_overhead: has("CUPTI_ACTIVITY_KIND_OVERHEAD"),
            has_cuda_contexts: has("TARGET_INFO_CUDA_CONTEXT_INFO"),
            has_sampling: has("SAMPLING_CALLCHAINS"),
            has_composite_events: has("COMPOSITE_EVENTS"),
            has_sched_events: has("SCHED_EVENTS"),
            // The flag is the "can `metrics --type X` run?" answer,
            // not raw table-presence — gate on every table the query
            // path requires so `summary.capabilities.has_*_metrics`
            // never falsely promises a query that would then bail on
            // a missing dictionary / id-map.
            has_gpu_metrics: has("GPU_METRICS") && has("TARGET_INFO_GPU_METRICS"),
            has_nic_metrics: has("NET_NIC_METRIC")
                && has("TARGET_INFO_NETWORK_METRICS")
                && has("NIC_ID_MAP")
                && has("TARGET_INFO_NIC_INFO"),
            has_target_info: has("TARGET_INFO_SYSTEM_ENV"),
        }
    }

    /// Quick "any GPU event table exists" predicate. Used by
    /// downstream callers (and by future capability-gated docs) to
    /// short-circuit before issuing a multi-table query.
    pub fn any_gpu_events(&self) -> bool {
        self.has_kernels
            || self.has_memcpy
            || self.has_memset
            || self.has_sync
            || self.has_graph_trace
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_all_false() {
        let f = CapabilityFlags::default();
        assert!(!f.has_kernels);
        assert!(!f.has_target_info);
        assert!(!f.any_gpu_events());
    }

    #[test]
    fn any_gpu_events_picks_up_each_kind() {
        let mut f = CapabilityFlags::default();
        assert!(!f.any_gpu_events());
        f.has_kernels = true;
        assert!(f.any_gpu_events());
        f.has_kernels = false;
        f.has_memcpy = true;
        assert!(f.any_gpu_events());
        f.has_memcpy = false;
        f.has_memset = true;
        assert!(f.any_gpu_events());
        f.has_memset = false;
        f.has_sync = true;
        assert!(f.any_gpu_events());
    }

    /// Forward-compat: deserialising an older cache that's missing
    /// (say) the future `has_gpu_metrics` field should round-trip
    /// with `false` rather than fail. `#[derive(Default)]` + serde's
    /// implicit field defaults give us that for free; this test
    /// pins the behaviour so a future PR removing `Default` is loud.
    #[test]
    fn deserialises_missing_field_as_false() -> anyhow::Result<()> {
        let json = r#"{
            "has_kernels": true,
            "has_memcpy": true,
            "has_memset": false,
            "has_sync": false,
            "has_runtime": true,
            "has_osrt": false,
            "has_nvtx": true,
            "has_cuda_contexts": true,
            "has_sampling": false,
            "has_target_info": true
        }"#;
        let parsed: CapabilityFlags = serde_json::from_str(json)?;
        assert!(parsed.has_kernels);
        // `has_gpu_metrics` / `has_nic_metrics` were absent —
        // default false.
        assert!(!parsed.has_gpu_metrics);
        assert!(!parsed.has_nic_metrics);
        Ok(())
    }
}
