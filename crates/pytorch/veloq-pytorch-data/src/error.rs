use std::io;
use std::num::TryFromIntError;
use std::path::Path;
use thiserror::Error;
use veloq_core::{ErrorCode, VeloqDiagnostic};
use veloq_data::DataError;

pub type PytorchDataResult<T> = Result<T, PytorchDataError>;

#[derive(Debug, Error)]
pub enum PytorchDataError {
    #[error(transparent)]
    Data(#[from] DataError),
    #[error("pytorch source expects a Chrome trace `.json` or `.json.gz` file, got {path}")]
    UnsupportedTraceExtension { path: String },
    #[error("trace input does not exist: {path}")]
    InputDoesNotExist { path: String },
    #[error(
        "pytorch v0 supports one Chrome trace `.json` or `.json.gz` file; directory inputs are not supported yet: {path}"
    )]
    DirectoryInputsUnsupported { path: String },
    #[error("too many trace files in {path}")]
    TooManyTraceFiles {
        path: String,
        #[source]
        source: TryFromIntError,
    },
    #[error("parsing {path}")]
    ParseJson {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("pytorch trace root must be a JSON object")]
    TraceRootNotObject,
    #[error("pytorch trace missing `traceEvents` array")]
    MissingTraceEvents,
    #[error("trace event index does not fit in u64")]
    TraceEventIndexOverflow {
        #[source]
        source: TryFromIntError,
    },
    #[error("event count does not fit in u64")]
    EventCountOverflow {
        #[source]
        source: TryFromIntError,
    },
    #[error("reading pytorch cache {path}")]
    ReadCache {
        path: String,
        #[source]
        source: io::Error,
    },
    #[error("creating pytorch cache dir {path}")]
    CreateCacheDir {
        path: String,
        #[source]
        source: io::Error,
    },
    #[error("encoding pytorch cache")]
    EncodeCache {
        #[source]
        source: serde_json::Error,
    },
    #[error("writing pytorch cache {path}")]
    WriteCache {
        path: String,
        #[source]
        source: io::Error,
    },
}

impl PytorchDataError {
    pub fn unsupported_trace_extension(path: &Path) -> Self {
        Self::UnsupportedTraceExtension {
            path: display_path(path),
        }
    }

    pub fn input_does_not_exist(path: &Path) -> Self {
        Self::InputDoesNotExist {
            path: display_path(path),
        }
    }

    pub fn directory_inputs_unsupported(path: &Path) -> Self {
        Self::DirectoryInputsUnsupported {
            path: display_path(path),
        }
    }

    pub fn too_many_trace_files(path: &Path, source: TryFromIntError) -> Self {
        Self::TooManyTraceFiles {
            path: display_path(path),
            source,
        }
    }

    pub fn parse_json(path: &Path, source: serde_json::Error) -> Self {
        Self::ParseJson {
            path: display_path(path),
            source,
        }
    }

    pub fn trace_event_index_overflow(source: TryFromIntError) -> Self {
        Self::TraceEventIndexOverflow { source }
    }

    pub fn event_count_overflow(source: TryFromIntError) -> Self {
        Self::EventCountOverflow { source }
    }

    pub fn read_cache(path: &Path, source: io::Error) -> Self {
        Self::ReadCache {
            path: display_path(path),
            source,
        }
    }

    pub fn create_cache_dir(path: &Path, source: io::Error) -> Self {
        Self::CreateCacheDir {
            path: display_path(path),
            source,
        }
    }

    pub fn encode_cache(source: serde_json::Error) -> Self {
        Self::EncodeCache { source }
    }

    pub fn write_cache(path: &Path, source: io::Error) -> Self {
        Self::WriteCache {
            path: display_path(path),
            source,
        }
    }
}

impl VeloqDiagnostic for PytorchDataError {
    fn code(&self) -> ErrorCode {
        match self {
            Self::Data(error) => error.code(),
            Self::UnsupportedTraceExtension { .. } => {
                ErrorCode::new("pytorch.input.unsupported-extension")
            }
            Self::InputDoesNotExist { .. } => ErrorCode::new("pytorch.input.missing"),
            Self::DirectoryInputsUnsupported { .. } => {
                ErrorCode::new("pytorch.input.directory-unsupported")
            }
            Self::TooManyTraceFiles { .. } => ErrorCode::new("pytorch.input.too-many-files"),
            Self::ParseJson { .. } => ErrorCode::new("pytorch.trace.parse-json"),
            Self::TraceRootNotObject => ErrorCode::new("pytorch.trace.root-not-object"),
            Self::MissingTraceEvents => ErrorCode::new("pytorch.trace.missing-trace-events"),
            Self::TraceEventIndexOverflow { .. } => {
                ErrorCode::new("pytorch.trace.event-index-overflow")
            }
            Self::EventCountOverflow { .. } => ErrorCode::new("pytorch.trace.event-count-overflow"),
            Self::ReadCache { .. } => ErrorCode::new("pytorch.cache.read"),
            Self::CreateCacheDir { .. } => ErrorCode::new("pytorch.cache.create-dir"),
            Self::EncodeCache { .. } => ErrorCode::new("pytorch.cache.encode"),
            Self::WriteCache { .. } => ErrorCode::new("pytorch.cache.write"),
        }
    }
}

fn display_path(path: &Path) -> String {
    path.display().to_string()
}
