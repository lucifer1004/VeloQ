//! `veloq stats <trace>` — aggregated GPU work statistics.
//!
//! Returns one row per kernel/memcpy/memset *name*, with count, total
//! duration, distribution (min/max/p50/p95/p99), and percentage of
//! total. Optionally filters by event type and time window.

mod group_by;
mod hydrate;
mod sql;

#[cfg(test)]
mod tests;

pub use group_by::{GroupBy, NameAxis, SortKey};

use crate::{EventKind, KindFilter, NsysQueryError, NsysQueryResult};
use duckdb::types::Value;
use group_by::{GroupBySql, HistSql, resolve_name_axis, stats_sort_sql};
use hydrate::{build_bucket_schema, hydrate_stats_rows};
use serde::Serialize;
use sql::{NVTX_STYLE_EXPR, per_kind_subquery};
use std::path::Path;
use veloq_core::{SortSpec, time::TimeWindow};

/// Event kinds that `stats` is willing to aggregate. Library consumers
/// constructing `StatsRequest` by hand should pick from this set; CLI
/// callers go through `GpuFilters::kinds(&Self::ALLOWED_KINDS)`.
///
/// Sync is included because `cudaStreamSynchronize` / `cudaDeviceSynchronize`
/// durations are agent-actionable signals (CPU blocked waiting on GPU),
/// and aggregating by syncType is the natural way to see which sync
/// dominates a workload.
///
/// Graph is included because, in `--cuda-graph-trace=graph` captures
/// (the common default), kernels-inside-graphs do not appear in
/// `CUPTI_ACTIVITY_KIND_KERNEL` — the graph_trace row is the *only*
/// per-execution record for that work. Excluding it would silently
/// undercount GPU work on graph-heavy workloads (vLLM, TRT-LLM).
///
/// Nvtx is included so agents can ask "what's the per-step duration
/// distribution" without leaving stats. NVTX ranges are CPU-side
/// markers (start/end on the host thread; no device / stream), and
/// instant markers (`end IS NULL`) are excluded — they have no
/// duration. Mixing NVTX with GPU kinds via `KindFilter::All` still
/// works: the SQL UNION projects NULL for device / stream on NVTX
/// rows and the per-group totals stay correct, but agents who want
/// "GPU work only" should narrow with `--type kernel,memcpy,memset`.
///
/// **Aggregation caveat**: NVTX ranges sharing a name across multiple
/// host threads (or threads driving different GPUs) fold into one
/// group under the default `--group-by short`. There is no
/// device axis on NVTX_EVENTS — `--group-by device|context|stream|
/// graph|graph_node` on `--type nvtx` is rejected up-front rather
/// than emit a single `null` bucket. A future per-thread axis can
/// disambiguate; today, agents that need it should run separate
/// queries with `--time-range` narrowing per region of interest.
pub const ALLOWED_KINDS: [EventKind; 8] = [
    EventKind::Kernel,
    EventKind::Memcpy,
    EventKind::Memset,
    EventKind::Sync,
    EventKind::Graph,
    EventKind::Nvtx,
    EventKind::Runtime,
    EventKind::Osrt,
];

#[derive(Debug, Clone)]
pub struct StatsRequest {
    /// Which kinds to aggregate. Resolved against [`ALLOWED_KINDS`]
    /// (kernel/memcpy/memset) at run time; `KindFilter::All` covers
    /// the GPU set and `KindFilter::Only(...)` picks a subset. `run()`
    /// defends with a `bail!` if a hand-built `Only(...)` contains a
    /// non-GPU kind.
    pub kinds: KindFilter,
    pub group_by: GroupBy,
    pub time_window: Option<TimeWindow>,
    /// When set, only aggregate over GPU events causally attributable
    /// to NVTX ranges whose name matches this glob (`*`/`?`).
    pub nvtx: Option<String>,
    /// Restrict to one CUDA device (NSys `deviceId`).
    pub device: Option<i32>,
    /// Restrict to one CUDA stream (NSys `streamId`).
    pub stream: Option<i64>,
    /// When `true`, each row gains a `histogram` array of per-bucket
    /// event counts using `HIST_BOUNDARIES_NS`. Response also surfaces
    /// the bucket schema once at the top level.
    pub hist: bool,
    /// Sort specification. `None` falls back to the default
    /// (`total` descending).
    pub sort: Option<SortSpec>,
    pub limit: usize,
    /// When `true`, stats `--type runtime` folds API versions
    /// (e.g. `cudaMalloc_v3020`, `cudaMalloc_v2000`, `cudaMalloc`)
    /// into one bucket by stripping the `_v<digits>` suffix before
    /// grouping. Matches the nsys recipe `cuda_api_sum`'s substr
    /// trick. No-op for non-Runtime kinds. Opt-in (default
    /// `false`) so the unversioned view stays the default.
    pub collapse_versioned: bool,
}

/// Half-decade duration boundaries (ns). 17 boundaries → 18 buckets
/// covering from sub-10 ns to multi-second event durations. Half a
/// decade per bucket gives enough resolution to distinguish
/// kernel populations without making the response huge.
pub const HIST_BOUNDARIES_NS: &[i64] = &[
    10,
    32,
    100,
    316,
    1_000,
    3_162,
    10_000,
    31_623,
    100_000,
    316_228,
    1_000_000,
    3_162_278,
    10_000_000,
    31_622_776,
    100_000_000,
    316_227_766,
    1_000_000_000,
];

impl Default for StatsRequest {
    fn default() -> Self {
        Self {
            kinds: KindFilter::All,
            group_by: GroupBy::default(),
            time_window: None,
            nvtx: None,
            device: None,
            stream: None,
            hist: false,
            sort: None,
            limit: 50,
            collapse_versioned: false,
        }
    }
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct StatsResponse {
    /// Number of groups returned in `rows` (after `--limit`).
    pub count: usize,
    /// Total number of distinct **groups** produced by the scope's
    /// GROUP BY *before* `--limit` clipped the list. When
    /// `total_matched > count`, some groups were dropped — raise
    /// `--limit` or narrow the filter to see them. Same envelope
    /// convention every verb uses; for stats specifically, `rows`
    /// are groups, so "rows matched" and "groups matched" coincide.
    pub total_matched: i64,
    /// Grand total *event-duration* across the whole filtered scope
    /// (type filter + time window applied, but NOT clipped by `--limit`).
    /// This is the denominator behind every row's `percentage`.
    pub total_duration_ns: i64,
    /// Grand **event count** summed across every group — distinct from
    /// [`Self::total_matched`], which counts *groups*, not events.
    /// Named explicitly (`total_events`, not `total_count`) so it
    /// stays unambiguous next to the envelope-convention
    /// `total_matched` at the wire-format level.
    pub total_events: i64,
    /// Resolved time window, if any (absolute ns).
    pub time_window_ns: Option<(i64, i64)>,
    /// NVTX scoping in effect (the user's pattern, if `--nvtx` was set).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_scope: Option<String>,
    /// Half-open bucket boundaries `[lo, hi)` in ns, present iff the
    /// caller set `--hist`. The last entry has `hi: null`. Each row's
    /// `histogram` array has the same length as this list and is
    /// indexed by bucket position.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub histogram_buckets_ns: Option<Vec<HistBucket>>,
    /// True when the caller asked for `--group-by mangled` but the
    /// trace's `CUPTI_ACTIVITY_KIND_KERNEL` table has no `mangledName`
    /// column (older NSys schema). The query silently downgrades to
    /// `--group-by demangled`; the same fallback is also surfaced via
    /// a `log::info!` line on stderr for human consumers.
    /// JSON-only agents read this flag
    /// instead of parsing stderr. Omitted when no fallback occurred.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub mangled_axis_fallback: bool,
    pub rows: Vec<StatRow>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct HistBucket {
    pub lo: i64,
    /// `None` for the open-ended tail bucket.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hi: Option<i64>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct StatRow {
    /// Cross-trace key. Pipe-separated composite identity built
    /// from `(kind, name?, device_id?, stream_id?, context_id?,
    /// graph_id?, graph_node_id?, nvtx_style?, nvtx_parent?,
    /// nvtx_path?, grid_block?)` — exactly the fields `--group-by` activated. Two
    /// `stats` runs with the same `--group-by` produce matching keys
    /// for matching aggregation rows.
    pub key: String,
    /// The primary group key in the name axis — shortName, demangled
    /// signature, memcpy direction label, or memset label. Omitted in
    /// JSON when `--group-by` has `no-name`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(rename = "type")]
    pub kind: &'static str,
    /// Kernel shortName. Populated on every kernel row whose name axis
    /// is `short` or `demangled` (under `short` it equals `name`; under
    /// `demangled` it lets agents roll variants back to their shortName
    /// group). `None` for non-kernel kinds (memcpy/memset/sync/graph/
    /// nvtx) and when the name axis is `no-name`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub short_name: Option<String>,
    /// Physical-dimension columns. Each is populated only when the
    /// corresponding axis is part of `--group-by`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_id: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_id: Option<i64>,
    /// Captured-graph id. Populated only when `--group-by graph` is
    /// active *and* the kernel/memcpy/memset row has `graphId` set
    /// (i.e. ran inside a CUDA graph in a `=node` capture). `None`
    /// otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_id: Option<i64>,
    /// Per-node id. Populated only when `--group-by graph_node` is
    /// active.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_node_id: Option<i64>,
    pub count: i64,
    pub total_ns: i64,
    pub avg_ns: i64,
    pub min_ns: i64,
    pub max_ns: i64,
    pub p50_ns: i64,
    pub p95_ns: i64,
    pub p99_ns: i64,
    /// Total bytes transferred — only populated for memcpy/memset rows.
    /// `None` for kernel rows (no `bytes` column).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes_total: Option<i64>,
    /// Effective bandwidth in decimal GB/s (10^9 bytes/sec). Computed
    /// as `bytes_total / total_ns`. Same population rule as `bytes_total`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gbps: Option<f64>,
    pub percentage: f64,
    /// Per-bucket event counts, indexed by `histogram_buckets_ns`
    /// position on the response. Present iff `--hist` was set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub histogram: Option<Vec<i64>>,
    /// NVTX-only raw eventType value from `NVTX_EVENTS.eventType`.
    /// Mirrors `NvtxDetails.event_type` at inspect/host_api.rs. `None`
    /// on non-NVTX rows. Within a group, multiple raw values can fold
    /// into one bucket (e.g. 59 and 70 both produce
    /// `nvtx_style = "push_pop"`); the surfaced value is the minimum
    /// raw eventType seen in that bucket.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_type: Option<i64>,
    /// Derived label for NVTX `eventType`:
    /// `{59,70}` → `"push_pop"`,
    /// `{60,71}` → `"start_end"`,
    /// anything else (NVTX_PAYLOAD_*, instrumentation, future ints) →
    /// `"unknown"`. `None` on non-NVTX rows. Participates in the
    /// composite group key on NVTX rows so PushPop and StartEnd ranges
    /// with the same name split into distinct buckets — mirrors nsys
    /// `nvtx_sum`'s `GROUP BY tag, style`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_style: Option<&'static str>,
    /// Composite key for the innermost enclosing NVTX range — only
    /// populated when `--group-by nvtx-parent` is active.
    /// `"nvtx:<rowid>"` for events that fall inside a real range,
    /// `"nvtx:none"` for events outside every range. Lets agents
    /// `INDEX(.rows; .nvtx_parent_key)` across traces without a name
    /// collision.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_parent_key: Option<String>,
    /// Innermost enclosing NVTX range name (or the visible sentinel
    /// `"__no_nvtx__"` for events outside every range). Populated only
    /// when `--group-by nvtx-parent` is active.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_parent_name: Option<String>,
    /// Nesting depth of the innermost enclosing NVTX range — 0 for
    /// outermost ranges, 1 for ranges fully inside a single outer
    /// range, etc. Populated only when `--group-by nvtx-parent` is
    /// active AND the event attributes to a real range; left `None`
    /// for the no-NVTX sentinel so depth-0 doesn't collide with real
    /// outermost ranges.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_parent_depth: Option<u8>,
    /// Composite key for the full NVTX path — only populated when
    /// `--group-by nvtx-path` is active. `"nvtx-path:<path>"` for
    /// events that fall inside a real path, `"nvtx-path:none"` for
    /// events outside every range.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_path_key: Option<String>,
    /// Full slash-joined path of the innermost enclosing NVTX range,
    /// or the visible no-NVTX sentinel. Populated only when
    /// `--group-by nvtx-path` is active.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_path: Option<String>,
    /// Resolved NVTX domain identity of the enclosing range — the
    /// process-local handle `domainId`. Populated only
    /// when `--group-by nvtx-path` is active AND the row attributes to
    /// a real range; `None` for the no-NVTX sentinel (which has no
    /// enclosing range and therefore no domain).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain_id: Option<i64>,
    /// Owning process id of the enclosing range's domain, decoded
    /// `(global_tid >> 24) & 0xFFFFFF`. Pairs with
    /// `domain_id` to form the domain's true identity. `None` for the
    /// no-NVTX sentinel.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain_pid: Option<i64>,
    /// Resolved domain name when the `(pid, domain_id)` domain was
    /// registered via an `NvtxDomainCreate` event.
    /// Best-effort: `None` when unregistered (incl. the default domain
    /// id 0) or for the no-NVTX sentinel.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain_name: Option<String>,
    /// Kernel launch grid X dimension. Populated only when
    /// `--group-by grid_block` is active. Mirrors `EventRefKernel.grid`
    /// component 0 from the inspect/search surfaces.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grid_x: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grid_y: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grid_z: Option<i64>,
    /// Kernel launch block X dimension. Populated only when
    /// `--group-by grid_block` is active.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_x: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_y: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_z: Option<i64>,
}

pub fn run<P: AsRef<Path>>(path: P, req: StatsRequest) -> NsysQueryResult<StatsResponse> {
    let (trace, abs_window) = crate::open_scoped(path.as_ref(), req.limit, req.time_window)?;

    // Defence-in-depth: hand-built `StatsRequest`s with non-stats
    // kinds get rejected here too. CLI callers go through
    // `GpuFilters::kinds` with the `ALLOWED_KINDS` allow-list and
    // never reach this branch.
    if let KindFilter::Only(v) = &req.kinds {
        for k in v {
            if !ALLOWED_KINDS.contains(k) {
                return Err(crate::NsysQueryError::StatsKindNotAllowed { kind: k.as_str() });
            }
        }
    }

    // Shared `--device` / `--stream` policy: explicit null-location
    // kinds (Runtime/Osrt/Nvtx/GraphNode/GraphEvent/Overhead/
    // CpuSample) error rather than silently dropping when a location
    // filter is set. `KindFilter::All` continues to narrow implicitly
    // (today's "default just works" behaviour).
    crate::kind_policy::validate_location_filter(
        &req.kinds,
        crate::kind_policy::LocationFilter {
            device: req.device,
            stream: req.stream,
        },
        "stats",
    )?;

    // Shared `--nvtx` policy: explicit non-attributable kinds error
    // with a redirecting message. `resolve_nvtx_kinds` below repeats
    // this validation as part of its pipeline, so this early call is
    // strictly for *error precedence* — a request like `--nvtx p
    // --type osrt --group-by device` should land on the "--nvtx can't
    // scope --type osrt" message rather than the group-by-axis error
    // emitted by the location/grid_block/nvtx-parent checks below.
    crate::kind_policy::validate_nvtx_filter(&req.kinds, req.nvtx.as_deref(), "stats")?;

    // Set-level `--group-by location-axis` reject. The rule fires when
    // every kind in the explicit set is CPU-only — so
    // `--type runtime --group-by device` and `--type runtime,osrt
    // --group-by device` both error, while `--type kernel,runtime
    // --group-by device` is positive (the kernel rows fill the
    // device buckets, runtime rows land in a single null-device
    // bucket per group key — agents that don't want that can drop
    // runtime from the type set explicitly).
    let group_by_location_axis = req.group_by.device
        || req.group_by.context
        || req.group_by.stream
        || req.group_by.graph
        || req.group_by.graph_node;
    if group_by_location_axis
        && let KindFilter::Only(explicit) = &req.kinds
        && !explicit.is_empty()
        && explicit.iter().all(|k| !k.is_location_bearing())
    {
        let csv = explicit
            .iter()
            .map(|k| k.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(crate::NsysQueryError::stats_group_by_location_axis_conflict(csv));
    }

    // Policy: --group-by grid_block is kernel-only. The
    // CUPTI gridX/blockX columns live only on
    // CUPTI_ACTIVITY_KIND_KERNEL — projecting NULL for other kinds
    // would produce a single misleading null bucket. KindFilter::All
    // narrows implicitly to kernel at SQL time (other kinds drop out
    // via table_exists). Explicit non-kernel kinds error up-front.
    if req.group_by.grid_block
        && let KindFilter::Only(explicit) = &req.kinds
        && let Some(other) = explicit.iter().find(|k| !matches!(k, EventKind::Kernel))
    {
        return Err(crate::NsysQueryError::StatsGridBlockKindConflict {
            kind: other.as_str(),
        });
    }

    // Policy: nvtx-parent and nvtx-path are
    // mutually exclusive with the graph/graph_node axes (different
    // attribution model — NVTX is host-thread containment; CUDA-graph
    // captures are device-side state) and with --type nvtx
    // (self-attribute).
    let group_by_nvtx_hierarchy = req.group_by.nvtx_parent || req.group_by.nvtx_path;
    if group_by_nvtx_hierarchy {
        if req.group_by.nvtx_parent && req.group_by.nvtx_path {
            return Err(crate::NsysQueryError::StatsNvtxHierarchyAxesConflict);
        }
        let axis_name = if req.group_by.nvtx_path {
            "nvtx-path"
        } else {
            "nvtx-parent"
        };
        if req.group_by.graph || req.group_by.graph_node {
            return Err(crate::NsysQueryError::StatsNvtxHierarchyGraphAxisConflict { axis_name });
        }
        if let KindFilter::Only(explicit) = &req.kinds
            && explicit.iter().any(|k| matches!(k, EventKind::Nvtx))
        {
            return Err(crate::NsysQueryError::StatsNvtxHierarchySelfAttribute { axis_name });
        }
        // NVTX prereq tables: matching the --nvtx filter's contract.
        // `NVTX_EVENTS` and `CUPTI_ACTIVITY_KIND_RUNTIME` are
        // unconditional; without them the sidecar build can't
        // compute anything.
        for t in ["NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_RUNTIME"] {
            if !trace.table_exists(t) {
                return Err(
                    crate::NsysQueryError::StatsNvtxHierarchyPrereqTableMissing {
                        axis_name,
                        table: t,
                    },
                );
            }
        }
        // `TARGET_INFO_CUDA_CONTEXT_INFO` is required when a
        // GPU-side kind is *actually present in this trace and
        // requested*. Pure `--type runtime --group-by nvtx-parent`
        // doesn't need the bridge (it joins on `rt_rowid`); ditto
        // for `KindFilter::All` against a runtime-only trace where
        // the GPU activity tables don't exist.
        //
        // The check resolves the request against `ALLOWED_KINDS` and
        // probes table existence, so `KindFilter::All` against a
        // trace with no GPU activity tables collapses to runtime-
        // only and skips the bridge requirement — matching the SQL
        // path's actual behavior (it would emit only the runtime
        // subquery).
        let resolved = req.kinds.resolve(&ALLOWED_KINDS);
        let needs_ctx_bridge = resolved.iter().any(|k| {
            matches!(
                k,
                EventKind::Kernel | EventKind::Memcpy | EventKind::Memset | EventKind::Sync
            ) && trace.table_exists(match k {
                EventKind::Kernel => "CUPTI_ACTIVITY_KIND_KERNEL",
                EventKind::Memcpy => "CUPTI_ACTIVITY_KIND_MEMCPY",
                EventKind::Memset => "CUPTI_ACTIVITY_KIND_MEMSET",
                EventKind::Sync => "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
                _ => "",
            })
        });
        if needs_ctx_bridge && !trace.table_exists("TARGET_INFO_CUDA_CONTEXT_INFO") {
            return Err(crate::NsysQueryError::StatsNvtxHierarchyContextInfoMissing { axis_name });
        }
    }

    // Resolve `--type` + `--nvtx` via the shared helper: validates
    // explicit non-attributable kinds against `--nvtx`, resolves
    // against `ALLOWED_KINDS`, filters by table presence, and
    // implicitly narrows to the attributable set when `--nvtx` is
    // active (the contract that previously diverged across stats /
    // search / timeline).
    //
    // An additional narrowing layers on top: when
    // `--group-by grid_block` is active, narrow to kernel rows even
    // on the implicit KindFilter::All path. Other kinds would
    // project NULL grid/block and pollute the single null bucket.
    // Explicit non-kernel kinds are rejected up-front above.
    let kinds: Vec<EventKind> = crate::kind_policy::resolve_nvtx_kinds(
        &req.kinds,
        req.nvtx.as_deref(),
        &ALLOWED_KINDS,
        &trace,
        "stats",
    )?
    .into_iter()
    .filter(|k| !req.group_by.grid_block || matches!(k, EventKind::Kernel))
    .collect();

    let histogram_buckets_ns = if req.hist {
        Some(build_bucket_schema())
    } else {
        None
    };

    if kinds.is_empty() {
        return Ok(StatsResponse {
            count: 0,
            total_matched: 0,
            total_duration_ns: 0,
            total_events: 0,
            time_window_ns: abs_window,
            nvtx_scope: req.nvtx.clone(),
            histogram_buckets_ns,
            mangled_axis_fallback: false,
            rows: Vec::new(),
        });
    }

    let attribution = match req.nvtx.as_deref() {
        Some(p) => Some(crate::nvtx_attribution::build(p, &kinds, &trace)?),
        None => None,
    };

    // Probe schema once so optional columns (currently only
    // `mangledName` on the kernel table) can resolve to
    // a real ref or NULL without a per-kind reprobe inside
    // `per_kind_subquery`. The probe is cheap (one
    // information_schema query) and the result is also consulted to
    // pick the effective name axis when `--group-by mangled` would
    // otherwise hit an absent column.
    let columns = crate::column_map::load_columns(trace.conn(), &["CUPTI_ACTIVITY_KIND_KERNEL"])?;
    let axis_resolution = resolve_name_axis(req.group_by.name, &columns);
    let effective_group_by = GroupBy {
        name: axis_resolution.effective,
        ..req.group_by
    };

    // Each subquery carries its own parameter list so positional binds
    // can't drift across the UNION (see `per_kind_subquery`).
    let nvtx_scope = if attribution.is_some() {
        crate::nvtx_attribution::NvtxScope::Attributed
    } else {
        crate::nvtx_attribution::NvtxScope::None
    };

    // When an NVTX hierarchy group-by is active, ensure the
    // trace's per-runtime NVTX-parent sidecar is built (cold) or
    // fresh (warm). The sidecar lives in `veloq-nsys-data` (path
    // `<trace>.veloq/nvtx-parent.parquet`) and is shared across every
    // NVTX-bearing verb — building it once amortises across every
    // future `stats --group-by nvtx-parent` / `search --with-nvtx`
    // / `inspect <kind>:N` call on the same trace. Per-thread sorted
    // NVTX + binary-search-and-walk-back gives an
    // `O(N_runtime × log N_nvtx + matches)` build cost.
    let nvtx_parent_sidecar: Option<std::path::PathBuf> = if group_by_nvtx_hierarchy {
        let path = veloq_nsys_data::runtime_nvtx_parent::ensure_sidecar(&trace)
            .map_err(NsysQueryError::nvtx_parent_sidecar_ensure)?;
        Some(path)
    } else {
        None
    };
    // Resolve the `(pid, domainId) -> name` map once when the path axis
    // is active, so hydration can attach a resolved domain name to each
    // nvtx-path row. Names are
    // best-effort: if the resolver errors (e.g. a partial trace), degrade
    // to an empty map — domain *identity* still works, only the human
    // name is missing. Never fail the verb over a name lookup.
    let domain_names: std::collections::HashMap<(i64, i64), String> = if req.group_by.nvtx_path {
        veloq_nsys_data::nvtx_tree::ensure_sidecar(&trace)
            .map_err(NsysQueryError::nvtx_tree_load)?;
        veloq_nsys_data::trace_map::nvtx_domain_names(&trace).unwrap_or_default()
    } else {
        std::collections::HashMap::new()
    };

    let mut subqueries: Vec<String> = Vec::with_capacity(kinds.len());
    let mut per_kind_params: Vec<Value> = Vec::new();
    for kind in &kinds {
        let (sql, params) = per_kind_subquery(
            *kind,
            abs_window,
            nvtx_scope,
            req.collapse_versioned,
            &columns,
            nvtx_parent_sidecar.as_deref(),
            req.group_by.nvtx_path,
        )?;
        subqueries.push(sql);
        per_kind_params.extend(params);
    }
    let union = subqueries.join(" UNION ALL ");

    let group_by_sql = GroupBySql::for_axes(&effective_group_by);

    // When `--nvtx` is set, prepend the attribution CTE so the per-kind
    // subqueries can filter via `rowid IN attributed_<kind>_rowids`.
    let attribution_prefix = match &attribution {
        Some(att) => format!("{},", att.body),
        None => String::new(),
    };

    // Sort: default `total` descending preserves the original behaviour
    // exactly; user-supplied multi-field specs override.
    let sort_spec = req
        .sort
        .clone()
        .unwrap_or_else(|| SortSpec::single("total"));
    let order_by = stats_sort_sql(&sort_spec)?;

    let hist_sql = HistSql::build(req.hist);
    let (hist_grouped_cols, hist_outer_cols) =
        (hist_sql.grouped_cols.as_str(), hist_sql.outer_cols.as_str());
    let GroupBySql {
        name_select,
        short_name_select,
        device_select,
        context_select,
        stream_select,
        graph_select,
        graph_node_select,
        nvtx_parent_rowid_select,
        nvtx_parent_name_select,
        nvtx_path_select,
        nvtx_domain_id_select,
        nvtx_domain_pid_select,
        grid_x_select,
        grid_y_select,
        grid_z_select,
        block_x_select,
        block_y_select,
        block_z_select,
        group_keys_sql,
    } = &group_by_sql;

    // Window functions over the aggregated rows give us the *scope-wide*
    // totals (sum/count across all groups, not just the LIMITed slice).
    // Same single query, no extra round-trip.
    // Optional `--device` / `--stream` filters: each adds a positive
    // predicate to the pre-grouping WHERE so deviceId / streamId are
    // matched against bind parameters. Bind order is appended after
    // the per-kind windowed params (handled below).
    let mut location_where = String::new();
    let mut location_params: Vec<Value> = Vec::new();
    crate::kind_policy::LocationFilter {
        device: req.device,
        stream: req.stream,
    }
    .append_where(&mut location_where, &mut location_params);

    let sql = format!(
        r#"
        WITH {attribution_prefix} events AS ({union}),
        grouped AS (
            SELECT
                {name_select},
                {short_name_select},
                kind,
                {device_select},
                {context_select},
                {stream_select},
                {graph_select},
                {graph_node_select},
                {nvtx_parent_rowid_select},
                {nvtx_parent_name_select},
                {nvtx_path_select},
                {nvtx_domain_id_select},
                {nvtx_domain_pid_select},
                {grid_x_select},
                {grid_y_select},
                {grid_z_select},
                {block_x_select},
                {block_y_select},
                {block_z_select},
                -- nvtx_style is a derived label folding
                -- raw eventType ints into push_pop/start_end/unknown.
                -- Participates in GROUP BY for --type nvtx (NULL on
                -- non-NVTX kinds collapses into one bucket, leaving
                -- GPU group counts unchanged). event_type is the raw
                -- min-value within the bucket so agents can drill
                -- back into NSys's enum.
                {NVTX_STYLE_EXPR}                              AS nvtx_style,
                MIN(event_type)                                AS event_type,
                COUNT(*)                                       AS count,
                CAST(SUM(duration) AS BIGINT)                  AS total_ns,
                CAST(AVG(duration) AS BIGINT)                  AS avg_ns,
                MIN(duration)                                  AS min_ns,
                MAX(duration)                                  AS max_ns,
                CAST(quantile_disc(duration, 0.50) AS BIGINT)  AS p50_ns,
                CAST(quantile_disc(duration, 0.95) AS BIGINT)  AS p95_ns,
                CAST(quantile_disc(duration, 0.99) AS BIGINT)  AS p99_ns,
                CAST(SUM(bytes)    AS BIGINT)                  AS bytes_total
                {hist_grouped_cols}
            FROM events
            WHERE duration > 0 {location_where}
            GROUP BY {group_keys_sql}
        )
        SELECT
            name, short_name, kind,
            device_id, context_id, stream_id,
            graph_id, graph_node_id,
            nvtx_parent_rowid, nvtx_parent_name, nvtx_path,
            nvtx_domain_id, nvtx_domain_pid,
            grid_x, grid_y, grid_z, block_x, block_y, block_z,
            nvtx_style, event_type,
            count,
            total_ns, avg_ns, min_ns, max_ns, p50_ns, p95_ns, p99_ns,
            bytes_total,
            -- bytes / (ns × 1e-9) = bytes × 1e9 / ns. Decimal GB (10^9
            -- bytes), matching how PCIe / NVLink specs report bandwidth.
            CASE WHEN bytes_total IS NULL OR total_ns <= 0 THEN NULL
                 ELSE CAST(bytes_total AS DOUBLE) * 1e-9
                      / (CAST(total_ns AS DOUBLE) * 1e-9)
            END AS gbps,
            CAST(SUM(total_ns) OVER () AS BIGINT) AS scope_total_ns,
            CAST(SUM(count)    OVER () AS BIGINT) AS scope_total_count,
            CAST(COUNT(*)      OVER () AS BIGINT) AS scope_total_groups
            {hist_outer_cols}
        FROM grouped
        ORDER BY {order_by}
        LIMIT ?
        "#
    );

    // Bind order matches SQL position:
    //   1. attribution CTE params (one for the pattern glob), if `--nvtx`
    //   2. per-kind windowed params (carried alongside the SQL fragments
    //      by `per_kind_subquery`, so the bind can't drift)
    //   3. location_where params (--device / --stream)
    //   4. LIMIT param
    let mut params: Vec<Value> = Vec::new();
    if let Some(att) = &attribution {
        params.extend(att.params.iter().cloned());
    }
    params.extend(per_kind_params);
    params.extend(location_params);
    params.push(Value::BigInt(req.limit as i64));

    // When --group-by nvtx-parent is active, compute
    // the per-rowid NVTX nesting once so depth resolution during
    // hydration is a HashMap::get. Skipped on the no-axis path so we
    // don't pay for the cache build on the common case.
    let nvtx_nesting = if req.group_by.nvtx_parent && trace.table_exists("NVTX_EVENTS") {
        Some(
            trace
                .nvtx_nesting()
                .map_err(NsysQueryError::nvtx_nesting_load)?,
        )
    } else {
        None
    };
    let (mut out, scope) = hydrate_stats_rows(
        &trace,
        &sql,
        &params,
        req.hist,
        nvtx_nesting.as_ref(),
        &domain_names,
    )?;

    if scope.total_ns > 0 {
        for r in &mut out {
            r.percentage = (r.total_ns as f64 / scope.total_ns as f64) * 100.0;
        }
    }

    Ok(StatsResponse {
        count: out.len(),
        total_matched: scope.total_groups,
        total_duration_ns: scope.total_ns,
        total_events: scope.total_count,
        time_window_ns: abs_window,
        nvtx_scope: req.nvtx.clone(),
        histogram_buckets_ns,
        mangled_axis_fallback: axis_resolution.fell_back,
        rows: out,
    })
}
