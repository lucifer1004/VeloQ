use thiserror::Error;
use veloq_core::{ErrorCode, VeloqDiagnostic, sort::SortParseError};

pub type NsysQueryResult<T> = Result<T, NsysQueryError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqlPhase {
    Prepare,
    Query,
    Read,
}

impl SqlPhase {
    fn code(self) -> ErrorCode {
        match self {
            Self::Prepare => ErrorCode::new("nsys.query.sql-prepare"),
            Self::Query => ErrorCode::new("nsys.query.sql-query"),
            Self::Read => ErrorCode::new("nsys.query.sql-read"),
        }
    }
}

impl std::fmt::Display for SqlPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Prepare => "prepare",
            Self::Query => "execute",
            Self::Read => "read",
        })
    }
}

#[derive(Debug, Error)]
pub enum NsysQueryError {
    #[error("internal: {verb} cannot process `{kind}` rows after kind validation")]
    InternalUnsupportedKind {
        verb: &'static str,
        kind: &'static str,
    },

    #[error("internal: {verb} SQL returned unrecognised kind tag `{kind}`")]
    InternalSqlKindTagInvalid { verb: &'static str, kind: String },

    #[error("internal: {verb} NVTX attribution cannot process `{kind}` rows after kind validation")]
    InternalNvtxAttributionUnsupportedKind {
        verb: &'static str,
        kind: &'static str,
    },

    #[error("internal: {query} stats SQL returned no row")]
    InternalStatsRowMissing { query: &'static str },

    #[error("internal: ncu-command expected one launch recipe after count check ({selector})")]
    InternalNcuCommandLaunchRecipeSelectionMissing { selector: &'static str },

    #[error("internal: ncu-command {query} SQL returned no row")]
    InternalNcuCommandSqlRowMissing { query: &'static str },

    #[error("internal: slice builder for nvtx_rowid {row_id} disappeared between fold and emit")]
    InternalSliceBuilderMissing { row_id: i64 },

    #[error("reverse NVTX attribution failed to {mode} sidecar")]
    NvtxReverseSidecarLoad {
        mode: &'static str,
        #[source]
        source: Box<veloq_nsys_data::NsysDataError>,
    },

    #[error("{source}")]
    TraceOpen {
        #[source]
        source: Box<veloq_nsys_data::NsysDataError>,
    },

    #[error("{source}")]
    CorrelationIndexLoad {
        #[source]
        source: Box<veloq_nsys_data::NsysDataError>,
    },

    #[error("hardware extraction failed")]
    HardwareExtract {
        #[source]
        source: Box<veloq_nsys_data::NsysDataError>,
    },

    #[error("time-window resolution failed")]
    TimeWindowResolve {
        #[source]
        source: Box<veloq_nsys_data::NsysDataError>,
    },

    #[error("{source}")]
    Data {
        #[source]
        source: Box<veloq_nsys_data::NsysDataError>,
    },

    #[error("summary metadata load failed")]
    SummaryMetaLoad {
        #[source]
        source: Box<veloq_nsys_data::NsysDataError>,
    },

    #[error("NVTX-parent sidecar ensure failed")]
    NvtxParentSidecarEnsure {
        #[source]
        source: Box<veloq_nsys_data::NsysDataError>,
    },

    #[error("NVTX nesting load failed")]
    NvtxNestingLoad {
        #[source]
        source: Box<veloq_nsys_data::NsysDataError>,
    },

    #[error("NVTX tree load failed")]
    NvtxTreeLoad {
        #[source]
        source: Box<veloq_nsys_data::NsysDataError>,
    },

    #[error("{area} failed to {phase} {label} SQL")]
    Sql {
        area: &'static str,
        phase: SqlPhase,
        label: String,
        #[source]
        source: duckdb::Error,
    },

    #[error("--limit must be at least 1 (got {limit})")]
    LimitTooSmall { limit: usize },

    #[error("invalid {flag} `{value}`")]
    PositiveDurationInvalid {
        flag: String,
        value: String,
        #[source]
        source: veloq_core::time::TimeParseError,
    },

    #[error("{flag} must be positive (got {ns} ns)")]
    PositiveDurationTooSmall { flag: String, ns: i64 },

    #[error("`--name` and `--name-regex` are mutually exclusive; pick one")]
    NameFilterConflict,

    #[error(
        "search doesn't surface cpu_sample rows; use `veloq metrics --type cpu-sampling` or `veloq inspect cpu_sample:<id>` instead"
    )]
    SearchCpuSampleUnsupported,

    #[error("invalid search --sort")]
    SearchSortInvalid {
        #[source]
        source: SortParseError,
    },

    #[error("invalid --interval `{value}`")]
    TimelineIntervalInvalid {
        value: String,
        #[source]
        source: veloq_core::time::TimeParseError,
    },

    #[error("--interval must be positive (got {interval_ns} ns)")]
    TimelineIntervalTooSmall { interval_ns: i64 },

    #[error("timeline only buckets GPU kinds (kernel/memcpy/memset/graph); got `{kind}`")]
    TimelineKindNotAllowed { kind: &'static str },

    #[error("viz timeline requires --from and --to")]
    VizTimelineWindowRequired,

    #[error("unknown viz timeline track kind `{kind}`")]
    VizTimelineUnknownTrackKind { kind: String },

    #[error("invalid viz timeline track selector `{selector}`; expected name=value")]
    VizTimelineInvalidSelector { selector: String },

    #[error("unknown selector `{selector}` for viz timeline track `{kind}`")]
    VizTimelineUnknownSelector { kind: String, selector: String },

    #[error("viz timeline selector `{selector}` must be a non-negative integer")]
    VizTimelineSelectorNonNegativeInt { selector: String },

    #[error("viz timeline selector `{selector}` must be a positive integer")]
    VizTimelineSelectorPositiveInt { selector: String },

    #[error("viz timeline --highlight-kernels requires `top=<n>`")]
    VizTimelineHighlightTopRequired,

    #[error("unknown viz timeline highlight scope `{scope}` (expected: name, instance)")]
    VizTimelineUnknownHighlightScope { scope: String },

    #[error(
        "unknown viz timeline highlight metric `{metric}` (expected: duration, count, max-duration)"
    )]
    VizTimelineUnknownHighlightMetric { metric: String },

    #[error("viz timeline track `cuda-stream` requires `device=<id>`")]
    VizTimelineCudaStreamDeviceRequired,

    #[error("viz timeline track `cuda-stream` requires `stream=<id>`")]
    VizTimelineCudaStreamStreamRequired,

    #[error("viz timeline track `cuda-stream` does not accept `device=all`")]
    VizTimelineCudaStreamDeviceAll,

    #[error(
        "viz timeline device selector matches multiple process-local CUDA namespaces; add `process=<pid>` or use `device=all`"
    )]
    VizTimelineProcessRequired,

    #[error("visualization artifact failed")]
    VizTimelineArtifact {
        #[source]
        source: veloq_vis::VisualizationError,
    },

    #[error("unknown --group-by `{group_by}` for slices --aggregate (expected: name, path)")]
    SlicesUnknownGroupBy { group_by: String },

    #[error("invalid slices --sort")]
    SlicesSortInvalid {
        #[source]
        source: SortParseError,
    },

    #[error(
        "slices requires `{table}`, which is not present in this trace (NVTX attribution is unavailable)"
    )]
    SlicesPrereqTableMissing { table: &'static str },

    #[error(
        "slices requires at least one GPU event table (kernel/memcpy/memset), but none is present in this trace"
    )]
    SlicesGpuEventTableMissing,

    #[error("--nvtx attribution requires `{table}`, which is not present in this trace")]
    NvtxAttributionPrereqTableMissing { table: &'static str },

    #[error(
        "--nvtx attribution needs at least one attributable kind (kernel/memcpy/memset/sync/runtime); requested kinds don't match any present table"
    )]
    NvtxAttributionNoAttributableTableMatch,

    #[error(
        "--nvtx attribution on kernel/memcpy/memset/sync requires `TARGET_INFO_CUDA_CONTEXT_INFO`, which is not present in this trace (GPU activity rows cannot be bridged to runtime rows without the context-info table; the lookup would silently miss every kernel)"
    )]
    NvtxAttributionContextInfoMissing,

    #[error("graph-replays --nvtx requires `{table}`, which is not present in this trace")]
    GraphReplaysNvtxPrereqTableMissing { table: &'static str },

    #[error("invalid graph-replays --sort")]
    GraphReplaysSortInvalid {
        #[source]
        source: SortParseError,
    },

    #[error("--group-by name axis specified twice (`{previous}` and `{current}`); pick one")]
    StatsGroupByNameAxisConflict {
        previous: &'static str,
        current: &'static str,
    },

    #[error(
        "unknown --group-by token `{token}` (expected: short, demangled, mangled, no-name, device, stream, context, graph, graph_node, nvtx-parent, nvtx-path, grid_block)"
    )]
    StatsGroupByUnknownToken { token: String },

    #[error(
        "stats only aggregates duration-bearing kinds (kernel/memcpy/memset/sync/graph/nvtx); got `{kind}`"
    )]
    StatsKindNotAllowed { kind: &'static str },

    #[error(
        "stats: --group-by device/context/stream/graph/graph_node has no meaning for `--type {kinds}`; every kind in the set is CPU-side and carries no device/stream/graph columns. Group by name (the default) or mix in a GPU-side kind (kernel/memcpy/memset/sync/graph/cuda_event) to split GPU rows by device while keeping the CPU-side rows in their own null bucket."
    )]
    StatsGroupByLocationAxisConflict { kinds: String },

    #[error(
        "{verb}: --group-by {axis} requires a device parent axis because {axis} ids are device-local; use `--device <id>` for one device or include `device,{axis}` in --group-by for comparison"
    )]
    StatsGroupByDeviceParentRequired {
        verb: &'static str,
        axis: &'static str,
    },

    #[error(
        "stats: --group-by grid_block is kernel-only; gridX/Y/Z and blockX/Y/Z columns live on CUPTI_ACTIVITY_KIND_KERNEL and nowhere else. Got `--type {kind}` in the explicit kind set; either drop the non-kernel kinds or unset --group-by grid_block"
    )]
    StatsGridBlockKindConflict { kind: &'static str },

    #[error("invalid stats --sort")]
    StatsSortInvalid {
        #[source]
        source: SortParseError,
    },

    #[error(
        "--by size only aggregates byte-carrying kinds (memcpy/memset); got `--type {kind}`; drop the kind or unset --by size"
    )]
    StatsBySizeKindNotAllowed { kind: &'static str },

    #[error(
        "stats-by-size does not yet support --group-by {axes}; supported axes today are the name axis (short/demangled/mangled/no-name) and device/context/stream"
    )]
    StatsBySizeGroupByUnsupported { axes: String },

    #[error("invalid stats-by-size --sort")]
    StatsBySizeSortInvalid {
        #[source]
        source: SortParseError,
    },

    #[error(
        "stats: --group-by nvtx-parent and nvtx-path are mutually exclusive; pick rowid-level parent buckets or path-level hierarchy buckets"
    )]
    StatsNvtxHierarchyAxesConflict,

    #[error(
        "stats: --group-by {axis_name} is mutually exclusive with graph/graph_node; NVTX attribution walks host-thread containment, captured-graph axes walk device-side state. Pick one model per query"
    )]
    StatsNvtxHierarchyGraphAxisConflict { axis_name: &'static str },

    #[error(
        "stats: --group-by {axis_name} + --type nvtx is a self-attribute tautology; NVTX rows are the ranges NVTX hierarchy axes attribute other kinds to. Drop one of the two flags"
    )]
    StatsNvtxHierarchySelfAttribute { axis_name: &'static str },

    #[error("--group-by {axis_name} requires `{table}`, which is not present in this trace")]
    StatsNvtxHierarchyPrereqTableMissing {
        axis_name: &'static str,
        table: &'static str,
    },

    #[error(
        "--group-by {axis_name} on kernel/memcpy/memset/sync requires `TARGET_INFO_CUDA_CONTEXT_INFO`, which is not present in this trace (GPU activity rows cannot be bridged to runtime rows without the context-info table; the lookup would silently miss every kernel)"
    )]
    StatsNvtxHierarchyContextInfoMissing { axis_name: &'static str },

    #[error("unknown --env `{env}` (expected: none, safe, all)")]
    NcuCommandUnknownEnv { env: String },

    #[error(
        "ncu-command requires a CUDA kernel row id (got `{row_id}`); use `search --type kernel` first"
    )]
    NcuCommandRowIdKind { row_id: String },

    #[error("ncu-command requires `CUPTI_ACTIVITY_KIND_KERNEL`, which is absent from this trace")]
    NcuCommandKernelTableMissing,

    #[error(
        "ncu-command requires `META_DATA_CAPTURE` to recover the original command, argv, cwd, and env"
    )]
    NcuCommandMetadataTableMissing,

    #[error("kernel row `{row_id}` was not found")]
    NcuCommandKernelNotFound { row_id: String },

    #[error("kernel row `{row_id}` has neither a resolved shortName nor demangledName")]
    NcuCommandKernelNameMissing { row_id: String },

    #[error("META_DATA_CAPTURE contains no PROCESS_N:COMMAND launch recipe")]
    NcuCommandLaunchRecipeMissing,

    #[error(
        "multiple META_DATA_CAPTURE launch recipes match process `{process}`; cannot choose an NCU target command"
    )]
    NcuCommandAmbiguousProcessRecipe { process: String },

    #[error(
        "multiple META_DATA_CAPTURE launch recipes are present and none matched the selected kernel process"
    )]
    NcuCommandAmbiguousLaunchRecipe,

    #[error(
        "{verb}: {axes} cannot be combined with `--type {kinds}`; these kinds have no device/stream columns"
    )]
    KindLocationFilterConflict {
        verb: String,
        axes: &'static str,
        kinds: String,
    },

    #[error(
        "{verb}: --nvtx cannot scope `--type {kinds}`; NVTX attribution for these kinds is experimental and not yet implemented"
    )]
    KindNvtxAttributionUnsupported { verb: String, kinds: String },

    #[error("invalid --scope `{scope}`; expected device, stream, or trace")]
    GapsInvalidScope { scope: String },

    #[error("invalid --min-duration `{value}`")]
    GapsMinDurationInvalid {
        value: String,
        #[source]
        source: veloq_core::time::TimeParseError,
    },

    #[error("--min must be positive (got {min_ns} ns)")]
    GapsMinTooSmall { min_ns: i64 },

    #[error(
        "--stream <id> requires --scope stream; under --scope {scope} rows have no stream axis"
    )]
    GapsStreamRequiresStreamScope { scope: &'static str },

    #[error(
        "--device {device} is incompatible with --scope trace; use --scope device --device {device}"
    )]
    GapsDeviceInTraceScope { device: i32 },

    #[error(
        "--sort stream requires --scope stream; under --scope {scope} rows have no stream axis"
    )]
    GapsSortStreamRequiresStreamScope { scope: &'static str },

    #[error(
        "--sort device is incompatible with --scope trace; trace-scope gaps are not partitioned by device"
    )]
    GapsSortDeviceInTraceScope,

    #[error("invalid gaps --sort")]
    GapsSortInvalid {
        #[source]
        source: SortParseError,
    },

    #[error("--sort doesn't apply in bucketed mode; buckets are time-ordered")]
    MetricsSortWithBucket,

    #[error("--bucket must be positive (got {bucket_ns} ns)")]
    MetricsBucketTooSmall { bucket_ns: i64 },

    #[error("invalid metrics --sort")]
    MetricsSortInvalid {
        #[source]
        source: SortParseError,
    },

    #[error(
        "unknown `--type {metric_source}` for metrics (supported: gpu, nic, cpu-sampling, cpu-sched)"
    )]
    MetricsUnknownSource { metric_source: String },

    #[error(
        "metrics --type gpu requires `GPU_METRICS`, which is absent from this trace; re-capture with `nsys profile --gpu-metrics-devices=…`"
    )]
    MetricsGpuTableMissing,

    #[error(
        "metrics --type gpu requires `TARGET_INFO_GPU_METRICS` (counter dictionary); likely a partial or corrupted nsys export"
    )]
    MetricsGpuDictionaryMissing,

    #[error(
        "no GPU counters match `--counter {glob}`; run `veloq metrics <trace> --type gpu` (no --counter) to list available names"
    )]
    MetricsGpuCounterNoMatch { glob: String },

    #[error(
        "metrics --type nic requires `NET_NIC_METRIC`, which is absent from this trace; re-capture with `nsys profile --nic-metrics=lf` (or `hf`) and verify `nsys status --network` passes"
    )]
    MetricsNicTableMissing,

    #[error(
        "metrics --type nic requires `TARGET_INFO_NETWORK_METRICS` (counter dictionary); likely a partial or corrupted nsys export"
    )]
    MetricsNicDictionaryMissing,

    #[error(
        "metrics --type nic requires `NIC_ID_MAP` (globalId to nicId mapping); likely a partial or corrupted nsys export"
    )]
    MetricsNicIdMapMissing,

    #[error(
        "metrics --type nic requires `TARGET_INFO_NIC_INFO` (NIC identity); likely a partial or corrupted nsys export"
    )]
    MetricsNicInfoMissing,

    #[error(
        "no NIC counters match `--counter {glob}`; run `veloq metrics <trace> --type nic` (no --counter) to list available names"
    )]
    MetricsNicCounterNoMatch { glob: String },

    #[error(
        "metrics --type cpu-sampling requires `COMPOSITE_EVENTS`, which is absent from this trace; re-capture with `nsys profile --sample=process-tree`"
    )]
    MetricsCpuSamplingCompositeEventsMissing,

    #[error(
        "unknown --group-by `{axis}` for cpu-sampling (expected: symbol, tid, cpu, module, stack)"
    )]
    MetricsCpuSamplingUnknownGroupBy { axis: String },

    #[error(
        "--group-by {group_by} needs `SAMPLING_CALLCHAINS` (per-sample stacks), which is absent from this trace; either re-capture with stack sampling enabled or switch to `--group-by tid` / `cpu`"
    )]
    MetricsCpuSamplingCallchainsMissing { group_by: &'static str },

    #[error(
        "--group-by {group_by} bucketed mode needs `SAMPLING_CALLCHAINS`, which is absent from this trace"
    )]
    MetricsCpuSamplingBucketCallchainsMissing { group_by: &'static str },

    #[error(
        "--group-by stack does not support --bucket yet; use summary mode for stack aggregation or switch to symbol/module/tid/cpu for bucketed samples"
    )]
    MetricsCpuSamplingStackBucketUnsupported,

    #[error(
        "--name doesn't apply on --group-by {group_by} (keys are numeric); drop it or switch to `--group-by symbol` / `module` / `stack`"
    )]
    MetricsCpuSamplingNameOnNumericAxis { group_by: &'static str },

    #[error(
        "metrics --type cpu-sched requires `SCHED_EVENTS`, which is absent from this trace; re-capture with `nsys profile --cpuctxsw=process-tree` (or `system-wide`)"
    )]
    MetricsCpuSchedEventsMissing,

    #[error("unknown --group-by `{axis}` for cpu-sched (expected: tid, cpu, state)")]
    MetricsCpuSchedUnknownGroupBy { axis: String },

    #[error("--top-nodes must be at least 1 (got 0)")]
    GraphReplaysTopNodesTooSmall,
}

impl NsysQueryError {
    pub fn internal_unsupported_kind(verb: &'static str, kind: &'static str) -> Self {
        Self::InternalUnsupportedKind { verb, kind }
    }

    pub fn internal_sql_kind_tag_invalid(verb: &'static str, kind: &str) -> Self {
        Self::InternalSqlKindTagInvalid {
            verb,
            kind: kind.to_string(),
        }
    }

    pub fn internal_nvtx_attribution_unsupported_kind(
        verb: &'static str,
        kind: &'static str,
    ) -> Self {
        Self::InternalNvtxAttributionUnsupportedKind { verb, kind }
    }

    pub fn internal_stats_row_missing(query: &'static str) -> Self {
        Self::InternalStatsRowMissing { query }
    }

    pub fn internal_slice_builder_missing(row_id: i64) -> Self {
        Self::InternalSliceBuilderMissing { row_id }
    }

    pub fn nvtx_reverse_sidecar_load(
        mode: &'static str,
        source: veloq_nsys_data::NsysDataError,
    ) -> Self {
        Self::NvtxReverseSidecarLoad {
            mode,
            source: Box::new(source),
        }
    }

    pub fn trace_open(source: veloq_nsys_data::NsysDataError) -> Self {
        Self::TraceOpen {
            source: Box::new(source),
        }
    }

    pub fn correlation_index_load(source: veloq_nsys_data::NsysDataError) -> Self {
        Self::CorrelationIndexLoad {
            source: Box::new(source),
        }
    }

    pub fn hardware_extract(source: veloq_nsys_data::NsysDataError) -> Self {
        Self::HardwareExtract {
            source: Box::new(source),
        }
    }

    pub fn time_window_resolve(source: veloq_nsys_data::NsysDataError) -> Self {
        Self::TimeWindowResolve {
            source: Box::new(source),
        }
    }

    pub fn data(source: veloq_nsys_data::NsysDataError) -> Self {
        Self::Data {
            source: Box::new(source),
        }
    }

    pub fn summary_meta_load(source: veloq_nsys_data::NsysDataError) -> Self {
        Self::SummaryMetaLoad {
            source: Box::new(source),
        }
    }

    pub fn nvtx_parent_sidecar_ensure(source: veloq_nsys_data::NsysDataError) -> Self {
        Self::NvtxParentSidecarEnsure {
            source: Box::new(source),
        }
    }

    pub fn nvtx_nesting_load(source: veloq_nsys_data::NsysDataError) -> Self {
        Self::NvtxNestingLoad {
            source: Box::new(source),
        }
    }

    pub fn nvtx_tree_load(source: veloq_nsys_data::NsysDataError) -> Self {
        Self::NvtxTreeLoad {
            source: Box::new(source),
        }
    }

    pub fn sql(
        area: &'static str,
        phase: SqlPhase,
        label: impl Into<String>,
        source: duckdb::Error,
    ) -> Self {
        Self::Sql {
            area,
            phase,
            label: label.into(),
            source,
        }
    }

    pub fn sql_prepare(
        area: &'static str,
        label: impl Into<String>,
        source: duckdb::Error,
    ) -> Self {
        Self::sql(area, SqlPhase::Prepare, label, source)
    }

    pub fn sql_query(area: &'static str, label: impl Into<String>, source: duckdb::Error) -> Self {
        Self::sql(area, SqlPhase::Query, label, source)
    }

    pub fn sql_read(area: &'static str, label: impl Into<String>, source: duckdb::Error) -> Self {
        Self::sql(area, SqlPhase::Read, label, source)
    }

    pub fn sql_parts(&self) -> Option<(&'static str, SqlPhase, &str)> {
        match self {
            Self::Sql {
                area, phase, label, ..
            } => Some((*area, *phase, label.as_str())),
            _ => None,
        }
    }

    pub fn search_sort_invalid(source: SortParseError) -> Self {
        Self::SearchSortInvalid { source }
    }

    pub fn kind_location_filter_conflict(verb: &str, axes: &'static str, kinds: String) -> Self {
        Self::KindLocationFilterConflict {
            verb: verb.to_string(),
            axes,
            kinds,
        }
    }

    pub fn kind_nvtx_attribution_unsupported(verb: &str, kinds: String) -> Self {
        Self::KindNvtxAttributionUnsupported {
            verb: verb.to_string(),
            kinds,
        }
    }

    pub fn gaps_invalid_scope(scope: &str) -> Self {
        Self::GapsInvalidScope {
            scope: scope.to_string(),
        }
    }

    pub fn gaps_sort_invalid(source: SortParseError) -> Self {
        Self::GapsSortInvalid { source }
    }

    pub fn graph_replays_sort_invalid(source: SortParseError) -> Self {
        Self::GraphReplaysSortInvalid { source }
    }

    pub fn metrics_unknown_source(metric_source: &str) -> Self {
        Self::MetricsUnknownSource {
            metric_source: metric_source.to_string(),
        }
    }

    pub fn metrics_sort_invalid(source: SortParseError) -> Self {
        Self::MetricsSortInvalid { source }
    }

    pub fn metrics_gpu_counter_no_match(glob: &str) -> Self {
        Self::MetricsGpuCounterNoMatch {
            glob: glob.to_string(),
        }
    }

    pub fn metrics_nic_counter_no_match(glob: &str) -> Self {
        Self::MetricsNicCounterNoMatch {
            glob: glob.to_string(),
        }
    }

    pub fn metrics_cpu_sampling_unknown_group_by(axis: &str) -> Self {
        Self::MetricsCpuSamplingUnknownGroupBy {
            axis: axis.to_string(),
        }
    }

    pub fn metrics_cpu_sched_unknown_group_by(axis: &str) -> Self {
        Self::MetricsCpuSchedUnknownGroupBy {
            axis: axis.to_string(),
        }
    }

    pub fn slices_unknown_group_by(group_by: &str) -> Self {
        Self::SlicesUnknownGroupBy {
            group_by: group_by.to_string(),
        }
    }

    pub fn slices_sort_invalid(source: SortParseError) -> Self {
        Self::SlicesSortInvalid { source }
    }

    pub fn stats_group_by_unknown_token(token: &str) -> Self {
        Self::StatsGroupByUnknownToken {
            token: token.to_string(),
        }
    }

    pub fn stats_group_by_location_axis_conflict(kinds: String) -> Self {
        Self::StatsGroupByLocationAxisConflict { kinds }
    }

    pub fn stats_by_size_group_by_unsupported(axes: String) -> Self {
        Self::StatsBySizeGroupByUnsupported { axes }
    }

    pub fn stats_sort_invalid(source: SortParseError) -> Self {
        Self::StatsSortInvalid { source }
    }

    pub fn stats_by_size_sort_invalid(source: SortParseError) -> Self {
        Self::StatsBySizeSortInvalid { source }
    }

    pub fn ncu_command_unknown_env(env: &str) -> Self {
        Self::NcuCommandUnknownEnv {
            env: env.to_string(),
        }
    }

    pub fn ncu_command_row_id_kind(row_id: impl ToString) -> Self {
        Self::NcuCommandRowIdKind {
            row_id: row_id.to_string(),
        }
    }

    pub fn ncu_command_kernel_not_found(row_id: impl ToString) -> Self {
        Self::NcuCommandKernelNotFound {
            row_id: row_id.to_string(),
        }
    }

    pub fn ncu_command_kernel_name_missing(row_id: impl ToString) -> Self {
        Self::NcuCommandKernelNameMissing {
            row_id: row_id.to_string(),
        }
    }

    pub fn ncu_command_ambiguous_process_recipe(process: &str) -> Self {
        Self::NcuCommandAmbiguousProcessRecipe {
            process: process.to_string(),
        }
    }
}

impl VeloqDiagnostic for NsysQueryError {
    fn code(&self) -> ErrorCode {
        match self {
            Self::InternalUnsupportedKind { .. } => {
                ErrorCode::new("nsys.internal.unsupported-kind")
            }
            Self::InternalSqlKindTagInvalid { .. } => {
                ErrorCode::new("nsys.internal.sql-kind-tag-invalid")
            }
            Self::InternalNvtxAttributionUnsupportedKind { .. } => {
                ErrorCode::new("nsys.internal.nvtx-attribution-unsupported-kind")
            }
            Self::InternalStatsRowMissing { .. } => {
                ErrorCode::new("nsys.internal.stats-row-missing")
            }
            Self::InternalNcuCommandLaunchRecipeSelectionMissing { .. } => {
                ErrorCode::new("nsys.internal.ncu-command-launch-recipe-selection-missing")
            }
            Self::InternalNcuCommandSqlRowMissing { .. } => {
                ErrorCode::new("nsys.internal.ncu-command-sql-row-missing")
            }
            Self::InternalSliceBuilderMissing { .. } => {
                ErrorCode::new("nsys.internal.slice-builder-missing")
            }
            Self::NvtxReverseSidecarLoad { .. } => {
                ErrorCode::new("nsys.query.nvtx-reverse-sidecar-load")
            }
            Self::TraceOpen { source }
            | Self::CorrelationIndexLoad { source }
            | Self::Data { source } => source.code(),
            Self::HardwareExtract { .. } => ErrorCode::new("nsys.query.hardware-extract"),
            Self::TimeWindowResolve { .. } => ErrorCode::new("nsys.query.time-window-resolve"),
            Self::SummaryMetaLoad { .. } => ErrorCode::new("nsys.query.summary-meta-load"),
            Self::NvtxParentSidecarEnsure { .. } => {
                ErrorCode::new("nsys.query.nvtx-parent-sidecar-ensure")
            }
            Self::NvtxNestingLoad { .. } => ErrorCode::new("nsys.query.nvtx-nesting-load"),
            Self::NvtxTreeLoad { .. } => ErrorCode::new("nsys.query.nvtx-tree-load"),
            Self::Sql { phase, .. } => phase.code(),
            Self::LimitTooSmall { .. } => ErrorCode::new("nsys.query.limit-too-small"),
            Self::PositiveDurationInvalid { .. } => {
                ErrorCode::new("nsys.query.invalid-positive-duration")
            }
            Self::PositiveDurationTooSmall { .. } => {
                ErrorCode::new("nsys.query.positive-duration-too-small")
            }
            Self::NameFilterConflict => ErrorCode::new("nsys.query.name-filter-conflict"),
            Self::SearchCpuSampleUnsupported => {
                ErrorCode::new("nsys.query.search-cpu-sample-unsupported")
            }
            Self::SearchSortInvalid { .. } => ErrorCode::new("nsys.query.search-sort-invalid"),
            Self::TimelineIntervalInvalid { .. } => {
                ErrorCode::new("nsys.query.timeline-interval-invalid")
            }
            Self::TimelineIntervalTooSmall { .. } => {
                ErrorCode::new("nsys.query.timeline-interval-too-small")
            }
            Self::TimelineKindNotAllowed { .. } => {
                ErrorCode::new("nsys.query.timeline-kind-not-allowed")
            }
            Self::VizTimelineWindowRequired => {
                ErrorCode::new("nsys.query.viz-timeline-window-required")
            }
            Self::VizTimelineUnknownTrackKind { .. } => {
                ErrorCode::new("nsys.query.viz-timeline-unknown-track-kind")
            }
            Self::VizTimelineInvalidSelector { .. } => {
                ErrorCode::new("nsys.query.viz-timeline-invalid-selector")
            }
            Self::VizTimelineUnknownSelector { .. } => {
                ErrorCode::new("nsys.query.viz-timeline-unknown-selector")
            }
            Self::VizTimelineSelectorNonNegativeInt { .. } => {
                ErrorCode::new("nsys.query.viz-timeline-selector-non-negative-int")
            }
            Self::VizTimelineSelectorPositiveInt { .. } => {
                ErrorCode::new("nsys.query.viz-timeline-selector-positive-int")
            }
            Self::VizTimelineHighlightTopRequired => {
                ErrorCode::new("nsys.query.viz-timeline-highlight-top-required")
            }
            Self::VizTimelineUnknownHighlightScope { .. } => {
                ErrorCode::new("nsys.query.viz-timeline-unknown-highlight-scope")
            }
            Self::VizTimelineUnknownHighlightMetric { .. } => {
                ErrorCode::new("nsys.query.viz-timeline-unknown-highlight-metric")
            }
            Self::VizTimelineCudaStreamDeviceRequired => {
                ErrorCode::new("nsys.query.viz-timeline-cuda-stream-device-required")
            }
            Self::VizTimelineCudaStreamStreamRequired => {
                ErrorCode::new("nsys.query.viz-timeline-cuda-stream-stream-required")
            }
            Self::VizTimelineCudaStreamDeviceAll => {
                ErrorCode::new("nsys.query.viz-timeline-cuda-stream-device-all")
            }
            Self::VizTimelineProcessRequired => {
                ErrorCode::new("nsys.query.viz-timeline-process-required")
            }
            Self::VizTimelineArtifact { source } => source.code(),
            Self::SlicesUnknownGroupBy { .. } => {
                ErrorCode::new("nsys.query.slices-unknown-group-by")
            }
            Self::SlicesSortInvalid { .. } => ErrorCode::new("nsys.query.slices-sort-invalid"),
            Self::SlicesPrereqTableMissing { .. } => {
                ErrorCode::new("nsys.query.slices-prereq-missing")
            }
            Self::SlicesGpuEventTableMissing => {
                ErrorCode::new("nsys.query.slices-gpu-event-table-missing")
            }
            Self::NvtxAttributionPrereqTableMissing { .. } => {
                ErrorCode::new("nsys.query.nvtx-attribution-prereq-missing")
            }
            Self::NvtxAttributionNoAttributableTableMatch => {
                ErrorCode::new("nsys.query.nvtx-attribution-no-attributable-kind")
            }
            Self::NvtxAttributionContextInfoMissing => {
                ErrorCode::new("nsys.query.nvtx-attribution-context-info-missing")
            }
            Self::GraphReplaysNvtxPrereqTableMissing { .. } => {
                ErrorCode::new("nsys.query.graph-replays-nvtx-prereq-missing")
            }
            Self::GraphReplaysSortInvalid { .. } => {
                ErrorCode::new("nsys.query.graph-replays-sort-invalid")
            }
            Self::StatsGroupByNameAxisConflict { .. } => {
                ErrorCode::new("nsys.query.stats-group-by-name-axis-conflict")
            }
            Self::StatsGroupByUnknownToken { .. } => {
                ErrorCode::new("nsys.query.stats-group-by-unknown-token")
            }
            Self::StatsKindNotAllowed { .. } => ErrorCode::new("nsys.query.stats-kind-not-allowed"),
            Self::StatsGroupByLocationAxisConflict { .. } => {
                ErrorCode::new("nsys.query.stats-group-by-location-axis-conflict")
            }
            Self::StatsGroupByDeviceParentRequired { .. } => {
                ErrorCode::new("nsys.query.stats-group-by-device-parent-required")
            }
            Self::StatsGridBlockKindConflict { .. } => {
                ErrorCode::new("nsys.query.stats-grid-block-kind-conflict")
            }
            Self::StatsSortInvalid { .. } => ErrorCode::new("nsys.query.stats-sort-invalid"),
            Self::StatsBySizeKindNotAllowed { .. } => {
                ErrorCode::new("nsys.query.stats-by-size-kind-not-allowed")
            }
            Self::StatsBySizeGroupByUnsupported { .. } => {
                ErrorCode::new("nsys.query.stats-by-size-group-by-unsupported")
            }
            Self::StatsBySizeSortInvalid { .. } => {
                ErrorCode::new("nsys.query.stats-by-size-sort-invalid")
            }
            Self::StatsNvtxHierarchyAxesConflict => {
                ErrorCode::new("nsys.query.stats-nvtx-hierarchy-axis-conflict")
            }
            Self::StatsNvtxHierarchyGraphAxisConflict { .. } => {
                ErrorCode::new("nsys.query.stats-nvtx-hierarchy-graph-axis-conflict")
            }
            Self::StatsNvtxHierarchySelfAttribute { .. } => {
                ErrorCode::new("nsys.query.stats-nvtx-hierarchy-self-attribute")
            }
            Self::StatsNvtxHierarchyPrereqTableMissing { .. } => {
                ErrorCode::new("nsys.query.stats-nvtx-hierarchy-prereq-missing")
            }
            Self::StatsNvtxHierarchyContextInfoMissing { .. } => {
                ErrorCode::new("nsys.query.stats-nvtx-hierarchy-context-info-missing")
            }
            Self::NcuCommandUnknownEnv { .. } => {
                ErrorCode::new("nsys.query.ncu-command-unknown-env")
            }
            Self::NcuCommandRowIdKind { .. } => {
                ErrorCode::new("nsys.query.ncu-command-row-id-kind")
            }
            Self::NcuCommandKernelTableMissing => {
                ErrorCode::new("nsys.query.ncu-command-kernel-table-missing")
            }
            Self::NcuCommandMetadataTableMissing => {
                ErrorCode::new("nsys.query.ncu-command-metadata-table-missing")
            }
            Self::NcuCommandKernelNotFound { .. } => {
                ErrorCode::new("nsys.query.ncu-command-kernel-not-found")
            }
            Self::NcuCommandKernelNameMissing { .. } => {
                ErrorCode::new("nsys.query.ncu-command-kernel-name-missing")
            }
            Self::NcuCommandLaunchRecipeMissing => {
                ErrorCode::new("nsys.query.ncu-command-launch-recipe-missing")
            }
            Self::NcuCommandAmbiguousProcessRecipe { .. } => {
                ErrorCode::new("nsys.query.ncu-command-launch-recipe-ambiguous-process")
            }
            Self::NcuCommandAmbiguousLaunchRecipe => {
                ErrorCode::new("nsys.query.ncu-command-launch-recipe-ambiguous")
            }
            Self::KindLocationFilterConflict { .. } => {
                ErrorCode::new("nsys.query.kind-location-filter-conflict")
            }
            Self::KindNvtxAttributionUnsupported { .. } => {
                ErrorCode::new("nsys.query.kind-nvtx-attribution-unsupported")
            }
            Self::GapsInvalidScope { .. } => ErrorCode::new("nsys.query.gaps-invalid-scope"),
            Self::GapsMinDurationInvalid { .. } => {
                ErrorCode::new("nsys.query.gaps-min-duration-invalid")
            }
            Self::GapsMinTooSmall { .. } => ErrorCode::new("nsys.query.gaps-min-too-small"),
            Self::GapsStreamRequiresStreamScope { .. } => {
                ErrorCode::new("nsys.query.gaps-stream-scope-required")
            }
            Self::GapsDeviceInTraceScope { .. } => {
                ErrorCode::new("nsys.query.gaps-device-scope-conflict")
            }
            Self::GapsSortStreamRequiresStreamScope { .. } => {
                ErrorCode::new("nsys.query.gaps-sort-stream-scope-required")
            }
            Self::GapsSortDeviceInTraceScope => {
                ErrorCode::new("nsys.query.gaps-sort-device-scope-conflict")
            }
            Self::GapsSortInvalid { .. } => ErrorCode::new("nsys.query.gaps-sort-invalid"),
            Self::MetricsSortWithBucket => {
                ErrorCode::new("nsys.query.metrics-sort-bucket-conflict")
            }
            Self::MetricsBucketTooSmall { .. } => {
                ErrorCode::new("nsys.query.metrics-bucket-too-small")
            }
            Self::MetricsSortInvalid { .. } => ErrorCode::new("nsys.query.metrics-sort-invalid"),
            Self::MetricsUnknownSource { .. } => {
                ErrorCode::new("nsys.query.metrics-unknown-source")
            }
            Self::MetricsGpuTableMissing => ErrorCode::new("nsys.query.metrics-gpu-table-missing"),
            Self::MetricsGpuDictionaryMissing => {
                ErrorCode::new("nsys.query.metrics-gpu-dictionary-missing")
            }
            Self::MetricsGpuCounterNoMatch { .. } => {
                ErrorCode::new("nsys.query.metrics-gpu-counter-no-match")
            }
            Self::MetricsNicTableMissing => ErrorCode::new("nsys.query.metrics-nic-table-missing"),
            Self::MetricsNicDictionaryMissing => {
                ErrorCode::new("nsys.query.metrics-nic-dictionary-missing")
            }
            Self::MetricsNicIdMapMissing => ErrorCode::new("nsys.query.metrics-nic-id-map-missing"),
            Self::MetricsNicInfoMissing => ErrorCode::new("nsys.query.metrics-nic-info-missing"),
            Self::MetricsNicCounterNoMatch { .. } => {
                ErrorCode::new("nsys.query.metrics-nic-counter-no-match")
            }
            Self::MetricsCpuSamplingCompositeEventsMissing => {
                ErrorCode::new("nsys.query.metrics-cpu-sampling-table-missing")
            }
            Self::MetricsCpuSamplingUnknownGroupBy { .. } => {
                ErrorCode::new("nsys.query.metrics-cpu-sampling-unknown-group-by")
            }
            Self::MetricsCpuSamplingCallchainsMissing { .. } => {
                ErrorCode::new("nsys.query.metrics-cpu-sampling-callchains-missing")
            }
            Self::MetricsCpuSamplingBucketCallchainsMissing { .. } => {
                ErrorCode::new("nsys.query.metrics-cpu-sampling-callchains-missing")
            }
            Self::MetricsCpuSamplingStackBucketUnsupported => {
                ErrorCode::new("nsys.query.metrics-cpu-sampling-stack-bucket-unsupported")
            }
            Self::MetricsCpuSamplingNameOnNumericAxis { .. } => {
                ErrorCode::new("nsys.query.metrics-cpu-sampling-name-axis-conflict")
            }
            Self::MetricsCpuSchedEventsMissing => {
                ErrorCode::new("nsys.query.metrics-cpu-sched-table-missing")
            }
            Self::MetricsCpuSchedUnknownGroupBy { .. } => {
                ErrorCode::new("nsys.query.metrics-cpu-sched-unknown-group-by")
            }
            Self::GraphReplaysTopNodesTooSmall => {
                ErrorCode::new("nsys.query.graph-replays-top-nodes-too-small")
            }
        }
    }
}
