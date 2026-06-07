use serde::Serialize;
use std::borrow::Cow;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ErrorCode(&'static str);

impl ErrorCode {
    pub const IO_READ: Self = Self::new("io.read");
    pub const IO_WRITE: Self = Self::new("io.write");
    pub const IO_STAT: Self = Self::new("io.stat");
    pub const IO_CREATE_DIR: Self = Self::new("io.create-dir");
    pub const IO_REMOVE: Self = Self::new("io.remove");
    pub const IO_PUBLISH: Self = Self::new("io.publish");
    pub const ENCODING_UTF8: Self = Self::new("encoding.utf8");
    pub const COMPRESSION_GZIP: Self = Self::new("compression.gzip");
    pub const ARROW_RECORD_BATCH: Self = Self::new("arrow.record-batch");
    pub const PARQUET_OPEN_WRITER: Self = Self::new("parquet.open-writer");
    pub const PARQUET_WRITE: Self = Self::new("parquet.write");
    pub const PARQUET_CLOSE: Self = Self::new("parquet.close");
    pub const OUTPUT_CSV_HEADER: Self = Self::new("output.csv-header");
    pub const OUTPUT_CSV_ROW: Self = Self::new("output.csv-row");
    pub const OUTPUT_CSV_FLUSH: Self = Self::new("output.csv-flush");
    pub const SIDECAR_ENCODE: Self = Self::new("sidecar.encode");
    pub const SIDECAR_DECODE: Self = Self::new("sidecar.decode");

    pub const fn new(code: &'static str) -> Self {
        Self(code)
    }

    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl Serialize for ErrorCode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.0)
    }
}

pub trait VeloqDiagnostic: Error + Send + Sync + 'static {
    fn code(&self) -> ErrorCode;

    fn message(&self) -> Cow<'_, str> {
        Cow::Owned(self.to_string())
    }

    fn hint(&self) -> Option<Cow<'_, str>> {
        None
    }
}
