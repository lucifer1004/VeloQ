use thiserror::Error;
use veloq_core::query::NameMatchError;
use veloq_core::time::TimeParseError;
use veloq_core::{ErrorCode, VeloqDiagnostic};

pub type PytorchQueryResult<T> = Result<T, PytorchQueryError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqlPhase {
    Prepare,
    Query,
    Read,
}

impl SqlPhase {
    fn code(self) -> ErrorCode {
        match self {
            Self::Prepare => ErrorCode::new("pytorch.query.sql-prepare"),
            Self::Query => ErrorCode::new("pytorch.query.sql-query"),
            Self::Read => ErrorCode::new("pytorch.query.sql-read"),
        }
    }

    fn action(self) -> &'static str {
        match self {
            Self::Prepare => "preparing",
            Self::Query => "querying",
            Self::Read => "reading",
        }
    }

    fn row_suffix(self) -> &'static str {
        match self {
            Self::Prepare | Self::Query => "",
            Self::Read => " row",
        }
    }
}

#[derive(Debug, Error)]
pub enum PytorchQueryError {
    #[error(
        "unknown pytorch --type `{token}`; expected one of: cpu-op, annotation, step, runtime, driver, kernel, memcpy, memset, memory, python, comm, all"
    )]
    UnknownType { token: String },
    #[error("--type must list at least one event type")]
    EmptyTypeSelection,
    #[error("pytorch trace has multiple ranks; use `--rank <n>` or `--all-ranks`")]
    MultiRankRequiresScope,
    #[error(
        "--limit must be at least 1 (limit=0 would suppress total_matched too); use `--limit 1` for one row + totals"
    )]
    LimitTooSmall,
    #[error("--name and --name-regex are mutually exclusive")]
    MutuallyExclusiveNameFilters,
    #[error("invalid --name glob `{pattern}`")]
    InvalidNameGlob {
        pattern: String,
        #[source]
        source: regex::Error,
    },
    #[error("invalid --name-regex `{pattern}`")]
    InvalidNameRegex {
        pattern: String,
        #[source]
        source: regex::Error,
    },
    #[error("`--from` and `--to` must be set together (got only one)")]
    MissingTimeBound,
    #[error("invalid --from `{value}`")]
    InvalidFrom {
        value: String,
        #[source]
        source: TimeParseError,
    },
    #[error("invalid --to `{value}`")]
    InvalidTo {
        value: String,
        #[source]
        source: TimeParseError,
    },
    #[error("time window end ({end} ns) must be greater than start ({start} ns)")]
    EmptyTimeWindow { start: i64, end: i64 },
    #[error("--interval must be greater than 0 ns")]
    IntervalTooSmall,
    #[error(
        "unknown pytorch stats --group-by axis `{axis}`; expected name,type,step,rank,device,stream,shape,comm-kind,python-context,python-path"
    )]
    UnknownStatsGroupBy { axis: String },
    #[error(
        "pytorch stats --group-by {axis} requires Python stack events, but this trace has none; re-capture with `torch.profiler.profile(..., with_stack=True)`"
    )]
    PythonStackMissing { axis: String },
    #[error("unknown pytorch slices --group-by axis `{axis}`; expected name or step")]
    UnknownSlicesGroupBy { axis: String },
    #[error("opening in-memory DuckDB connection")]
    SqlOpen {
        #[source]
        source: duckdb::Error,
    },
    #[error(
        "{action} pytorch {area} SQL{row_suffix}",
        action = phase.action(),
        row_suffix = phase.row_suffix()
    )]
    Sql {
        area: &'static str,
        phase: SqlPhase,
        label: String,
        #[source]
        source: duckdb::Error,
    },
    #[error("pytorch search SQL count does not fit in usize")]
    SearchCountOverflow {
        #[source]
        source: std::num::TryFromIntError,
    },
    #[error("pytorch stats SQL count does not fit in usize")]
    StatsCountOverflow {
        #[source]
        source: std::num::TryFromIntError,
    },
    #[error("pytorch timeline SQL count does not fit in usize")]
    TimelineCountOverflow {
        #[source]
        source: std::num::TryFromIntError,
    },
    #[error("pytorch slices SQL count does not fit in usize")]
    SlicesCountOverflow {
        #[source]
        source: std::num::TryFromIntError,
    },
    #[error("pytorch collectives SQL count does not fit in usize")]
    CollectivesCountOverflow {
        #[source]
        source: std::num::TryFromIntError,
    },
    #[error("decoding pytorch inspect {field} JSON sidecar value")]
    InspectJsonDecode {
        field: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("pytorch inspect trace_index does not fit in u32")]
    InspectTraceIndexOverflow {
        #[source]
        source: std::num::TryFromIntError,
    },
}

impl PytorchQueryError {
    pub fn unknown_type(token: &str) -> Self {
        Self::UnknownType {
            token: token.to_string(),
        }
    }

    pub fn invalid_name_glob(pattern: &str, source: regex::Error) -> Self {
        Self::InvalidNameGlob {
            pattern: pattern.to_string(),
            source,
        }
    }

    pub fn invalid_name_regex(pattern: &str, source: regex::Error) -> Self {
        Self::InvalidNameRegex {
            pattern: pattern.to_string(),
            source,
        }
    }

    pub fn invalid_from(value: &str, source: TimeParseError) -> Self {
        Self::InvalidFrom {
            value: value.to_string(),
            source,
        }
    }

    pub fn invalid_to(value: &str, source: TimeParseError) -> Self {
        Self::InvalidTo {
            value: value.to_string(),
            source,
        }
    }

    pub fn unknown_stats_group_by(axis: &str) -> Self {
        Self::UnknownStatsGroupBy {
            axis: axis.to_string(),
        }
    }

    pub fn python_stack_missing(axis: &str) -> Self {
        Self::PythonStackMissing {
            axis: axis.to_string(),
        }
    }

    pub fn unknown_slices_group_by(axis: &str) -> Self {
        Self::UnknownSlicesGroupBy {
            axis: axis.to_string(),
        }
    }

    pub(crate) fn from_name_match(source: NameMatchError) -> Self {
        match source {
            NameMatchError::Conflict => Self::MutuallyExclusiveNameFilters,
            NameMatchError::InvalidGlob { pattern, source } => {
                Self::InvalidNameGlob { pattern, source }
            }
            NameMatchError::InvalidRegex { pattern, source } => {
                Self::InvalidNameRegex { pattern, source }
            }
        }
    }

    pub(crate) fn sql_open(source: duckdb::Error) -> Self {
        Self::SqlOpen { source }
    }

    pub(crate) fn sql(
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

    pub(crate) fn sql_prepare(
        area: &'static str,
        label: impl Into<String>,
        source: duckdb::Error,
    ) -> Self {
        Self::sql(area, SqlPhase::Prepare, label, source)
    }

    pub(crate) fn sql_query(
        area: &'static str,
        label: impl Into<String>,
        source: duckdb::Error,
    ) -> Self {
        Self::sql(area, SqlPhase::Query, label, source)
    }

    pub(crate) fn sql_read(
        area: &'static str,
        label: impl Into<String>,
        source: duckdb::Error,
    ) -> Self {
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

    pub(crate) fn search_count_overflow(source: std::num::TryFromIntError) -> Self {
        Self::SearchCountOverflow { source }
    }

    pub(crate) fn stats_count_overflow(source: std::num::TryFromIntError) -> Self {
        Self::StatsCountOverflow { source }
    }

    pub(crate) fn timeline_count_overflow(source: std::num::TryFromIntError) -> Self {
        Self::TimelineCountOverflow { source }
    }

    pub(crate) fn slices_count_overflow(source: std::num::TryFromIntError) -> Self {
        Self::SlicesCountOverflow { source }
    }

    pub(crate) fn collectives_count_overflow(source: std::num::TryFromIntError) -> Self {
        Self::CollectivesCountOverflow { source }
    }

    pub(crate) fn inspect_json_decode(field: &str, source: serde_json::Error) -> Self {
        Self::InspectJsonDecode {
            field: field.to_string(),
            source,
        }
    }

    pub(crate) fn inspect_trace_index_overflow(source: std::num::TryFromIntError) -> Self {
        Self::InspectTraceIndexOverflow { source }
    }
}

impl VeloqDiagnostic for PytorchQueryError {
    fn code(&self) -> ErrorCode {
        match self {
            Self::UnknownType { .. } => ErrorCode::new("pytorch.query.unknown-type"),
            Self::EmptyTypeSelection => ErrorCode::new("pytorch.query.empty-type-selection"),
            Self::MultiRankRequiresScope => ErrorCode::new("pytorch.query.rank-scope-required"),
            Self::LimitTooSmall => ErrorCode::new("pytorch.query.limit-too-small"),
            Self::MutuallyExclusiveNameFilters => {
                ErrorCode::new("pytorch.query.name-filter-conflict")
            }
            Self::InvalidNameGlob { .. } => ErrorCode::new("pytorch.query.invalid-name-glob"),
            Self::InvalidNameRegex { .. } => ErrorCode::new("pytorch.query.invalid-name-regex"),
            Self::MissingTimeBound => ErrorCode::new("pytorch.query.missing-time-bound"),
            Self::InvalidFrom { .. } => ErrorCode::new("pytorch.query.invalid-from"),
            Self::InvalidTo { .. } => ErrorCode::new("pytorch.query.invalid-to"),
            Self::EmptyTimeWindow { .. } => ErrorCode::new("pytorch.query.empty-time-window"),
            Self::IntervalTooSmall => ErrorCode::new("pytorch.query.interval-too-small"),
            Self::UnknownStatsGroupBy { .. } => {
                ErrorCode::new("pytorch.query.unknown-stats-group-by")
            }
            Self::PythonStackMissing { .. } => ErrorCode::new("pytorch.query.python-stack-missing"),
            Self::UnknownSlicesGroupBy { .. } => {
                ErrorCode::new("pytorch.query.unknown-slices-group-by")
            }
            Self::SqlOpen { .. } => ErrorCode::new("pytorch.query.sql-open"),
            Self::Sql { phase, .. } => phase.code(),
            Self::SearchCountOverflow { .. } => {
                ErrorCode::new("pytorch.query.search-count-overflow")
            }
            Self::StatsCountOverflow { .. } => ErrorCode::new("pytorch.query.stats-count-overflow"),
            Self::TimelineCountOverflow { .. } => {
                ErrorCode::new("pytorch.query.timeline-count-overflow")
            }
            Self::SlicesCountOverflow { .. } => {
                ErrorCode::new("pytorch.query.slices-count-overflow")
            }
            Self::CollectivesCountOverflow { .. } => {
                ErrorCode::new("pytorch.query.collectives-count-overflow")
            }
            Self::InspectJsonDecode { .. } => ErrorCode::new("pytorch.query.inspect-json-decode"),
            Self::InspectTraceIndexOverflow { .. } => {
                ErrorCode::new("pytorch.query.inspect-trace-index-overflow")
            }
        }
    }
}
