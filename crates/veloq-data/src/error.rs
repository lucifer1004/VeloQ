use std::io;
use std::path::Path;
use std::string::FromUtf8Error;
use thiserror::Error;
use veloq_core::{ErrorCode, VeloqDiagnostic};

pub type DataResult<T> = Result<T, DataError>;

#[derive(Debug, Error)]
pub enum DataError {
    #[error("reading {path}")]
    ReadFile {
        path: String,
        #[source]
        source: io::Error,
    },
    #[error("writing {path}")]
    WriteFile {
        path: String,
        #[source]
        source: io::Error,
    },
    #[error("stat {path}")]
    StatPath {
        path: String,
        #[source]
        source: io::Error,
    },
    #[error("creating {path}")]
    CreateDir {
        path: String,
        #[source]
        source: io::Error,
    },
    #[error("publishing {path}")]
    Publish {
        path: String,
        #[source]
        source: io::Error,
    },
    #[error("decompressing {path}")]
    DecompressGzip {
        path: String,
        #[source]
        source: io::Error,
    },
    #[error("{content} is not valid UTF-8: {path}")]
    DecodeUtf8 {
        path: String,
        content: String,
        #[source]
        source: FromUtf8Error,
    },
    #[error("building parquet batch for {path}")]
    BuildParquetBatch {
        path: String,
        #[source]
        source: arrow::error::ArrowError,
    },
    #[error("opening parquet writer for {path}")]
    OpenParquetWriter {
        path: String,
        #[source]
        source: parquet::errors::ParquetError,
    },
    #[error("writing parquet batch {path}")]
    WriteParquetBatch {
        path: String,
        #[source]
        source: parquet::errors::ParquetError,
    },
    #[error("closing parquet writer {path}")]
    CloseParquetWriter {
        path: String,
        #[source]
        source: parquet::errors::ParquetError,
    },
}

impl DataError {
    pub fn read_file(path: &Path, source: io::Error) -> Self {
        Self::ReadFile {
            path: display_path(path),
            source,
        }
    }

    pub fn write_file(path: &Path, source: io::Error) -> Self {
        Self::WriteFile {
            path: display_path(path),
            source,
        }
    }

    pub fn stat_path(path: &Path, source: io::Error) -> Self {
        Self::StatPath {
            path: display_path(path),
            source,
        }
    }

    pub fn create_dir(path: &Path, source: io::Error) -> Self {
        Self::CreateDir {
            path: display_path(path),
            source,
        }
    }

    pub fn publish(path: &Path, source: io::Error) -> Self {
        Self::Publish {
            path: display_path(path),
            source,
        }
    }

    pub fn decompress_gzip(path: &Path, source: io::Error) -> Self {
        Self::DecompressGzip {
            path: display_path(path),
            source,
        }
    }

    pub fn decode_utf8(path: &Path, content: impl Into<String>, source: FromUtf8Error) -> Self {
        Self::DecodeUtf8 {
            path: display_path(path),
            content: content.into(),
            source,
        }
    }

    pub fn build_parquet_batch(path: &Path, source: arrow::error::ArrowError) -> Self {
        Self::BuildParquetBatch {
            path: display_path(path),
            source,
        }
    }

    pub fn open_parquet_writer(path: &Path, source: parquet::errors::ParquetError) -> Self {
        Self::OpenParquetWriter {
            path: display_path(path),
            source,
        }
    }

    pub fn write_parquet_batch(path: &Path, source: parquet::errors::ParquetError) -> Self {
        Self::WriteParquetBatch {
            path: display_path(path),
            source,
        }
    }

    pub fn close_parquet_writer(path: &Path, source: parquet::errors::ParquetError) -> Self {
        Self::CloseParquetWriter {
            path: display_path(path),
            source,
        }
    }
}

impl VeloqDiagnostic for DataError {
    fn code(&self) -> ErrorCode {
        match self {
            Self::ReadFile { .. } => ErrorCode::IO_READ,
            Self::WriteFile { .. } => ErrorCode::IO_WRITE,
            Self::StatPath { .. } => ErrorCode::IO_STAT,
            Self::CreateDir { .. } => ErrorCode::IO_CREATE_DIR,
            Self::Publish { .. } => ErrorCode::IO_PUBLISH,
            Self::DecompressGzip { .. } => ErrorCode::COMPRESSION_GZIP,
            Self::DecodeUtf8 { .. } => ErrorCode::ENCODING_UTF8,
            Self::BuildParquetBatch { .. } => ErrorCode::ARROW_RECORD_BATCH,
            Self::OpenParquetWriter { .. } => ErrorCode::PARQUET_OPEN_WRITER,
            Self::WriteParquetBatch { .. } => ErrorCode::PARQUET_WRITE,
            Self::CloseParquetWriter { .. } => ErrorCode::PARQUET_CLOSE,
        }
    }
}

fn display_path(path: &Path) -> String {
    path.display().to_string()
}
