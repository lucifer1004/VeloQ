use thiserror::Error;
use veloq_core::time::TimeParseError;
use veloq_core::{ErrorCode, OutputFormat, TabularError, VeloqDiagnostic};
use veloq_pytorch_data::PytorchDataError;
use veloq_pytorch_query::PytorchQueryError;

pub type PytorchCommandResult<T> = Result<T, PytorchCommandError>;
pub type PytorchSourceResult<T> = Result<T, PytorchSourceError>;

#[derive(Debug, Error)]
pub enum PytorchSourceError {
    #[error(transparent)]
    Command(#[from] PytorchCommandError),
    #[error(transparent)]
    Data(#[from] PytorchDataError),
    #[error(transparent)]
    Query(#[from] PytorchQueryError),
    #[error(transparent)]
    Tabular(#[from] TabularError),
    #[error("serializing pytorch response envelope")]
    SerializeEnvelope {
        #[source]
        source: serde_json::Error,
    },
}

impl PytorchSourceError {
    pub fn serialize_envelope(source: serde_json::Error) -> Self {
        Self::SerializeEnvelope { source }
    }
}

impl VeloqDiagnostic for PytorchSourceError {
    fn code(&self) -> ErrorCode {
        match self {
            Self::Command(error) => error.code(),
            Self::Data(error) => error.code(),
            Self::Query(error) => error.code(),
            Self::Tabular(error) => error.code(),
            Self::SerializeEnvelope { .. } => ErrorCode::new("pytorch.output.serialize-envelope"),
        }
    }
}

#[derive(Debug, Error)]
pub enum PytorchCommandError {
    #[error("veloq-pytorch schema currently supports only --format json (got `{fmt}`)")]
    UnsupportedSchemaFormat { fmt: OutputFormat },
    #[error(
        "unknown pytorch schema target `{target}`; expected one of: summary, search, inspect, stats, correlate, timeline, slices, collectives, prep"
    )]
    UnknownSchemaTarget { target: String },
    #[error("serializing pytorch schema target `{target}`")]
    SerializeSchema {
        target: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("internal: pytorch verb missing trace path")]
    MissingTracePath,
    #[error("internal: schema should have returned before trace dispatch")]
    SchemaDispatchedAsTraceCommand,
    #[error("invalid --duration `{value}`")]
    InvalidDuration {
        value: String,
        #[source]
        source: TimeParseError,
    },
    #[error("invalid --interval `{value}`")]
    InvalidInterval {
        value: String,
        #[source]
        source: TimeParseError,
    },
    #[error(
        "--limit must be at least 1 (limit=0 would suppress total_matched too); use `--limit 1` for one row + totals"
    )]
    LimitTooSmall,
}

impl PytorchCommandError {
    pub fn unsupported_schema_format(fmt: OutputFormat) -> Self {
        Self::UnsupportedSchemaFormat { fmt }
    }

    pub fn unknown_schema_target(target: &str) -> Self {
        Self::UnknownSchemaTarget {
            target: target.to_string(),
        }
    }

    pub fn serialize_schema(target: &str, source: serde_json::Error) -> Self {
        Self::SerializeSchema {
            target: target.to_string(),
            source,
        }
    }

    pub fn invalid_duration(value: &str, source: TimeParseError) -> Self {
        Self::InvalidDuration {
            value: value.to_string(),
            source,
        }
    }

    pub fn invalid_interval(value: &str, source: TimeParseError) -> Self {
        Self::InvalidInterval {
            value: value.to_string(),
            source,
        }
    }
}

impl VeloqDiagnostic for PytorchCommandError {
    fn code(&self) -> ErrorCode {
        match self {
            Self::UnsupportedSchemaFormat { .. } => {
                ErrorCode::new("pytorch.command.unsupported-schema-format")
            }
            Self::UnknownSchemaTarget { .. } => {
                ErrorCode::new("pytorch.command.unknown-schema-target")
            }
            Self::SerializeSchema { .. } => ErrorCode::new("pytorch.command.serialize-schema"),
            Self::MissingTracePath => ErrorCode::new("pytorch.command.missing-trace-path"),
            Self::SchemaDispatchedAsTraceCommand => {
                ErrorCode::new("pytorch.command.schema-dispatched-as-trace")
            }
            Self::InvalidDuration { .. } => ErrorCode::new("pytorch.command.invalid-duration"),
            Self::InvalidInterval { .. } => ErrorCode::new("pytorch.command.invalid-interval"),
            Self::LimitTooSmall => ErrorCode::new("pytorch.command.limit-too-small"),
        }
    }
}
