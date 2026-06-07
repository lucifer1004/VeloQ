use thiserror::Error;
use veloq_core::sort::SortParseError;
use veloq_core::{ErrorCode, VeloqDiagnostic};
use veloq_nsys_query::row_id::RowIdParseError;

pub type NsysSourceResult<T> = Result<T, NsysSourceError>;

#[derive(Debug, Error)]
pub enum NsysSourceError {
    #[error("internal: nsys verb missing trace path")]
    MissingTracePath,

    #[error("internal: schema handled before trace dispatch")]
    SchemaHandledBeforeTraceDispatch,

    #[error("unknown schema target `{target}`; expected one of: {expected}")]
    UnknownSchemaTarget { target: String, expected: String },

    #[error("`--from` and `--to` must be set together (got only one)")]
    MissingTimeBound,

    #[error("invalid --from `{value}`")]
    InvalidFrom {
        value: String,
        #[source]
        source: veloq_core::time::TimeParseError,
    },

    #[error("invalid --to `{value}`")]
    InvalidTo {
        value: String,
        #[source]
        source: veloq_core::time::TimeParseError,
    },

    #[error("--limit must be at least 1 (got {limit})")]
    LimitTooSmall { limit: usize },

    #[error("unknown event kind `{kind}`")]
    UnknownEventKind { kind: String },

    #[error("--type `{kind}` not allowed here (expected one of: {expected})")]
    EventKindNotAllowed { kind: String, expected: String },

    #[error("--type must list at least one event kind")]
    EmptyEventKindList,

    #[error("invalid --sort `{value}`")]
    InvalidSort {
        value: String,
        #[source]
        source: SortParseError,
    },

    #[error("invalid --duration `{value}`")]
    InvalidDuration {
        value: String,
        #[source]
        source: veloq_core::time::TimeParseError,
    },

    #[error("invalid row_id `{row_id}`")]
    InvalidRowId {
        row_id: String,
        #[source]
        source: RowIdParseError,
    },

    #[error("`{verb}` is experimental; set VELOQ_UNSTABLE=1 to opt in")]
    UnstableFeatureDisabled { verb: String },

    #[error(
        "histograms are not supported in --by size mode; byte histograms are not implemented yet"
    )]
    StatsBySizeHistogramUnsupported,

    #[error("--by size + --nvtx is not implemented yet; use --nvtx without --by size")]
    StatsBySizeNvtxUnsupported,

    #[error(
        "--by size does not support --group-by {axes}; supported axes are name/device/context/stream"
    )]
    StatsBySizeGroupByUnsupported { axes: String },

    #[error("slices --group-by {group_by} requires --aggregate")]
    SlicesGroupByRequiresAggregate { group_by: String },

    #[error(
        "ncu-command only supports JSON output (got `{fmt}`); use --print for a pipe-ready shell script"
    )]
    NcuCommandUnsupportedFormat { fmt: veloq_core::OutputFormat },

    #[error(
        "unknown metrics --type `{metric_source}`; expected gpu, nic, cpu-sampling, or cpu-sched"
    )]
    MetricsUnknownSource { metric_source: String },

    #[error("invalid metrics --bucket `{value}`")]
    MetricsInvalidBucket {
        value: String,
        #[source]
        source: Box<veloq_nsys_query::NsysQueryError>,
    },

    #[error("scope resolution could not open trace {path}")]
    ScopeTraceOpen {
        path: String,
        #[source]
        source: Box<veloq_nsys_data::NsysDataError>,
    },

    #[error("prep --status could not read parquetdir {path}")]
    PrepStatusReadParquetDir {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error(
        "metrics --type {metric_source} does not accept --group-by / --name / --cpu / --tid; drop them or switch to `--type cpu-sampling` / `cpu-sched`"
    )]
    MetricsCpuFlagsForCounterSource { metric_source: &'static str },

    #[error(
        "--counter is a gpu/nic flag (matches PM counter names); use `--name <glob>` to filter cpu-sampling rows"
    )]
    MetricsCounterFlagForCpuSampling,

    #[error(
        "--counter is a gpu/nic flag (matches PM counter names); cpu-sched has no name field to filter on"
    )]
    MetricsCounterFlagForCpuSched,

    #[error(
        "--name is a cpu-sampling flag (matches stack-frame symbols); cpu-sched has no name field — use --group-by / --tid / --cpu"
    )]
    MetricsNameFlagForCpuSched,

    #[error("serializing nsys schema target `{target}`")]
    SerializeSchema {
        target: &'static str,
        #[source]
        source: serde_json::Error,
    },

    #[error(transparent)]
    Tabular(#[from] veloq_core::TabularError),

    #[error("serializing nsys response envelope")]
    SerializeEnvelope {
        #[source]
        source: serde_json::Error,
    },

    #[error("{source}")]
    Query {
        #[source]
        source: Box<veloq_nsys_query::NsysQueryError>,
    },

    #[error("{source}")]
    Data {
        #[source]
        source: Box<veloq_nsys_data::NsysDataError>,
    },
}

impl NsysSourceError {
    pub fn unknown_schema_target(target: &str, expected: String) -> Self {
        Self::UnknownSchemaTarget {
            target: target.to_string(),
            expected,
        }
    }

    pub fn invalid_from(value: &str, source: veloq_core::time::TimeParseError) -> Self {
        Self::InvalidFrom {
            value: value.to_string(),
            source,
        }
    }

    pub fn invalid_to(value: &str, source: veloq_core::time::TimeParseError) -> Self {
        Self::InvalidTo {
            value: value.to_string(),
            source,
        }
    }

    pub fn limit_too_small(limit: usize) -> Self {
        Self::LimitTooSmall { limit }
    }

    pub fn unknown_event_kind(kind: &str) -> Self {
        Self::UnknownEventKind {
            kind: kind.to_string(),
        }
    }

    pub fn event_kind_not_allowed(kind: &str, expected: String) -> Self {
        Self::EventKindNotAllowed {
            kind: kind.to_string(),
            expected,
        }
    }

    pub fn invalid_sort(value: &str, source: SortParseError) -> Self {
        Self::InvalidSort {
            value: value.to_string(),
            source,
        }
    }

    pub fn invalid_duration(value: &str, source: veloq_core::time::TimeParseError) -> Self {
        Self::InvalidDuration {
            value: value.to_string(),
            source,
        }
    }

    pub fn invalid_row_id(row_id: &str, source: RowIdParseError) -> Self {
        Self::InvalidRowId {
            row_id: row_id.to_string(),
            source,
        }
    }

    pub fn unstable_feature_disabled(verb: &str) -> Self {
        Self::UnstableFeatureDisabled {
            verb: verb.to_string(),
        }
    }

    pub fn stats_by_size_group_by_unsupported(axes: String) -> Self {
        Self::StatsBySizeGroupByUnsupported { axes }
    }

    pub fn slices_group_by_requires_aggregate(group_by: &str) -> Self {
        Self::SlicesGroupByRequiresAggregate {
            group_by: group_by.to_string(),
        }
    }

    pub fn ncu_command_unsupported_format(fmt: veloq_core::OutputFormat) -> Self {
        Self::NcuCommandUnsupportedFormat { fmt }
    }

    pub fn metrics_unknown_source(metric_source: &str) -> Self {
        Self::MetricsUnknownSource {
            metric_source: metric_source.to_string(),
        }
    }

    pub fn metrics_invalid_bucket(value: &str, source: veloq_nsys_query::NsysQueryError) -> Self {
        Self::MetricsInvalidBucket {
            value: value.to_string(),
            source: Box::new(source),
        }
    }

    pub fn scope_trace_open(
        path: impl std::fmt::Display,
        source: veloq_nsys_data::NsysDataError,
    ) -> Self {
        Self::ScopeTraceOpen {
            path: path.to_string(),
            source: Box::new(source),
        }
    }

    pub fn prep_status_read_parquetdir(
        path: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::PrepStatusReadParquetDir {
            path: path.to_string(),
            source,
        }
    }

    pub fn serialize_schema(target: &'static str, source: serde_json::Error) -> Self {
        Self::SerializeSchema { target, source }
    }

    pub fn serialize_envelope(source: serde_json::Error) -> Self {
        Self::SerializeEnvelope { source }
    }

    pub fn query(source: veloq_nsys_query::NsysQueryError) -> Self {
        Self::Query {
            source: Box::new(source),
        }
    }

    pub fn data(source: veloq_nsys_data::NsysDataError) -> Self {
        Self::Data {
            source: Box::new(source),
        }
    }
}

impl From<veloq_nsys_data::NsysDataError> for NsysSourceError {
    fn from(source: veloq_nsys_data::NsysDataError) -> Self {
        Self::data(source)
    }
}

impl From<veloq_nsys_query::NsysQueryError> for NsysSourceError {
    fn from(source: veloq_nsys_query::NsysQueryError) -> Self {
        Self::query(source)
    }
}

impl VeloqDiagnostic for NsysSourceError {
    fn code(&self) -> ErrorCode {
        match self {
            Self::MissingTracePath => ErrorCode::new("nsys.internal.missing-trace-path"),
            Self::SchemaHandledBeforeTraceDispatch => {
                ErrorCode::new("nsys.internal.schema-handled-before-trace-dispatch")
            }
            Self::UnknownSchemaTarget { .. } => {
                ErrorCode::new("nsys.command.unknown-schema-target")
            }
            Self::MissingTimeBound => ErrorCode::new("nsys.command.missing-time-bound"),
            Self::InvalidFrom { .. } => ErrorCode::new("nsys.command.invalid-from"),
            Self::InvalidTo { .. } => ErrorCode::new("nsys.command.invalid-to"),
            Self::LimitTooSmall { .. } => ErrorCode::new("nsys.command.limit-too-small"),
            Self::UnknownEventKind { .. } => ErrorCode::new("nsys.command.unknown-event-kind"),
            Self::EventKindNotAllowed { .. } => {
                ErrorCode::new("nsys.command.event-kind-not-allowed")
            }
            Self::EmptyEventKindList => ErrorCode::new("nsys.command.empty-event-kind-list"),
            Self::InvalidSort { .. } => ErrorCode::new("nsys.command.invalid-sort"),
            Self::InvalidDuration { .. } => ErrorCode::new("nsys.command.invalid-duration"),
            Self::InvalidRowId { .. } => ErrorCode::new("nsys.command.invalid-row-id"),
            Self::UnstableFeatureDisabled { .. } => {
                ErrorCode::new("nsys.command.unstable-feature-disabled")
            }
            Self::StatsBySizeHistogramUnsupported => {
                ErrorCode::new("nsys.command.stats-by-size-hist-unsupported")
            }
            Self::StatsBySizeNvtxUnsupported => {
                ErrorCode::new("nsys.command.stats-by-size-nvtx-unsupported")
            }
            Self::StatsBySizeGroupByUnsupported { .. } => {
                ErrorCode::new("nsys.command.stats-by-size-group-by-unsupported")
            }
            Self::SlicesGroupByRequiresAggregate { .. } => {
                ErrorCode::new("nsys.command.slices-group-by-requires-aggregate")
            }
            Self::NcuCommandUnsupportedFormat { .. } => {
                ErrorCode::new("nsys.command.ncu-command-unsupported-format")
            }
            Self::MetricsUnknownSource { .. } => {
                ErrorCode::new("nsys.command.metrics-unknown-source")
            }
            Self::MetricsInvalidBucket { .. } => {
                ErrorCode::new("nsys.command.metrics-invalid-bucket")
            }
            Self::ScopeTraceOpen { .. } => ErrorCode::new("nsys.command.scope-trace-open"),
            Self::PrepStatusReadParquetDir { .. } => {
                ErrorCode::new("nsys.command.prep-status-read-parquetdir")
            }
            Self::MetricsCpuFlagsForCounterSource { .. } => {
                ErrorCode::new("nsys.command.metrics-cpu-flag-conflict")
            }
            Self::MetricsCounterFlagForCpuSampling | Self::MetricsCounterFlagForCpuSched => {
                ErrorCode::new("nsys.command.metrics-counter-flag-conflict")
            }
            Self::MetricsNameFlagForCpuSched => {
                ErrorCode::new("nsys.command.metrics-name-flag-conflict")
            }
            Self::SerializeSchema { .. } => ErrorCode::new("nsys.command.serialize-schema"),
            Self::Tabular(err) => err.code(),
            Self::SerializeEnvelope { .. } => ErrorCode::new("nsys.output.serialize-envelope"),
            Self::Query { source } => source.code(),
            Self::Data { source } => source.code(),
        }
    }
}
