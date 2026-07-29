use thiserror::Error;
use veloq_core::{ErrorCode, VeloqDiagnostic};

pub type NsysDataResult<T> = Result<T, NsysDataError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DuckdbPhase {
    Prepare,
    Query,
    Read,
}

impl DuckdbPhase {
    fn code(self) -> ErrorCode {
        match self {
            Self::Prepare => ErrorCode::new("nsys.data.duckdb-prepare"),
            Self::Query => ErrorCode::new("nsys.data.duckdb-query"),
            Self::Read => ErrorCode::new("nsys.data.duckdb-read"),
        }
    }
}

impl std::fmt::Display for DuckdbPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Prepare => "prepare",
            Self::Query => "execute",
            Self::Read => "read",
        })
    }
}

#[derive(Debug, Error)]
pub enum NsysDataError {
    #[error("trace not found: {path}")]
    TraceNotFound { path: String },

    #[error("generated parquetdir belongs to missing .nsys-rep source: {source_path}")]
    GeneratedParquetdirSourceMissing { source_path: String },

    #[error("parquetdir not found: {path}")]
    ParquetdirNotFound { path: String },

    #[error("duckdb thread count could not be configured")]
    DuckdbThreadConfig {
        #[source]
        source: duckdb::Error,
    },

    #[error("duckdb in-memory connection could not be opened")]
    DuckdbOpenInMemory {
        #[source]
        source: duckdb::Error,
    },

    #[error("duckdb nsight schema could not be created")]
    DuckdbSchemaCreate {
        #[source]
        source: duckdb::Error,
    },

    #[error("parquetdir could not be read: {path}")]
    ParquetdirRead {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("parquet path is not valid UTF-8: {path}")]
    ParquetPathInvalidUtf8 { path: String },

    #[error("parquet table view could not be created for {table}")]
    ParquetViewCreate {
        table: String,
        path: String,
        #[source]
        source: duckdb::Error,
    },

    #[error("nvtx-tree view could not be registered from {path}")]
    NvtxTreeViewRegister {
        path: String,
        #[source]
        source: duckdb::Error,
    },

    #[error("gpu-work-events view could not be registered from {path}")]
    GpuWorkEventsViewRegister {
        path: String,
        #[source]
        source: duckdb::Error,
    },

    #[error("sidecar cache operation failed")]
    SidecarCache {
        #[source]
        source: veloq_data::DataError,
    },

    #[error("sidecar operation failed")]
    SidecarOperation {
        #[source]
        source: veloq_core::SidecarError,
    },

    #[error("sidecar cache header operation failed")]
    SidecarHeader {
        #[source]
        source: veloq_core::SidecarError,
    },

    #[error("trace artifact fingerprint could not be read: {path}")]
    TraceFingerprintRead {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error(
        "veloq does not read .sqlite traces directly: {path}; pass the original .nsys-rep instead"
    )]
    SqliteInputUnsupported { path: String },

    #[error("running `{command}` failed to start: {source}")]
    NsysExportSpawn {
        command: String,
        #[source]
        source: std::io::Error,
    },

    #[error("running `{command}` failed (exit {exit_code:?}) for {source_path}")]
    NsysExportFailed {
        command: String,
        exit_code: Option<i32>,
        source_path: String,
    },

    #[error("`nsys export -t parquetdir` reported success but produced no directory at {expected}")]
    NsysExportOutputMissing { expected: String },

    #[error("artifact directory could not be created: {path}")]
    ArtifactDirCreate {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("parquetdir export lockfile could not be opened: {path}")]
    NsysExportLockfileOpen {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("parquetdir export lock could not be acquired: {path}")]
    NsysExportLockAcquire {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("stale parquetdir could not be removed: {path}")]
    NsysParquetdirStaleRemove {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("parquetdir could not be published from {from} to {to}")]
    NsysParquetdirPublish {
        from: String,
        to: String,
        #[source]
        source: std::io::Error,
    },

    #[error(".nsys-rep source could not be stat'ed: {path}")]
    NsysCacheSourceStat {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("parquetdir sentinel could not be stat'ed: {path}")]
    NsysCacheSentinelStat {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("running `{command}` failed to start: {source}")]
    NsysParquetdirHelpSpawn {
        command: String,
        #[source]
        source: std::io::Error,
    },

    #[error("running `{command}` failed to start: {source}")]
    NsysVersionSpawn {
        command: String,
        #[source]
        source: std::io::Error,
    },

    #[error("stale generated parquetdir path could not be removed: {path}")]
    NsysExportGeneratedPathRemove {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error(".nsys-rep source could not be canonicalized: {path}")]
    NsysExportSourceCanonicalize {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error(
        "veloq requires nsys >= 2024.6 because this nsys does not support parquetdir export; detected: {detected}; upgrade nsys to 2024.6 or newer"
    )]
    NsysParquetdirUnsupported { detected: String },

    #[error(
        "could not probe `nsys export --help type` for parquetdir support: {message}; ensure the Nsight Systems CLI is installed and on PATH"
    )]
    NsysParquetdirProbeFailed { message: String },

    #[error(
        "internal: --device {device} and --all-devices are mutually exclusive but both were supplied"
    )]
    ScopeConflictingDeviceFlags { device: i32 },

    #[error(
        "--stream {stream} requires a single device scope; pass --device <id> or drop --stream"
    )]
    ScopeStreamRequiresDevice { stream: i64 },

    #[error(
        "CUDA process identity could not be resolved for {table} \
         (device={device_id}, context={context_id}, correlation={correlation_id:?}); \
         capture process/runtime metadata or use a complete Nsight Systems export"
    )]
    CudaProcessUnresolved {
        table: String,
        device_id: i32,
        context_id: i64,
        correlation_id: Option<i64>,
    },

    #[error("scope device probe requires {table}.{column}, which is not present in this trace")]
    ScopeDeviceProbeColumnMissing {
        table: String,
        column: String,
        #[source]
        source: duckdb::Error,
    },

    #[error("{area} failed to {phase} {label} DuckDB query")]
    Duckdb {
        area: &'static str,
        phase: DuckdbPhase,
        label: String,
        #[source]
        source: duckdb::Error,
    },

    #[error("internal: meta cache slot was not initialised after set/race")]
    MetaCacheSlotUninitialised,

    #[error("trace map probe requires {table}.{column}, which is not present in this trace")]
    TraceMapProbeColumnMissing {
        table: String,
        column: String,
        #[source]
        source: duckdb::Error,
    },

    #[error("--time-range resolves to an empty or inverted window: {source}")]
    TimeRangeEmpty {
        #[source]
        source: veloq_core::time::TimeParseError,
    },

    #[error("correlation cache could not fingerprint trace: {path}")]
    CorrelationTraceFingerprint {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("{label} is not an unsigned 64-bit integer: {value}")]
    HardwareInvalidU64Id {
        label: String,
        value: String,
        #[source]
        source: std::num::ParseIntError,
    },

    #[error("{path}: {column} has unsupported integer type {data_type}")]
    NvtxParentUnsupportedIntegerType {
        path: String,
        column: String,
        data_type: String,
    },

    #[error("{path}: {column} value {value} does not fit Int64")]
    NvtxParentIntegerOverflow {
        path: String,
        column: String,
        value: u64,
    },

    #[error(
        "nvtx-parent: row {row}: nvtx_rowids ({rowids_len}) and nvtx_names ({names_len}) length mismatch"
    )]
    NvtxParentListLengthMismatch {
        row: usize,
        rowids_len: usize,
        names_len: usize,
    },

    #[error("nvtx-parent sidecar missing {column} column in {path}")]
    NvtxParentColumnMissing { path: String, column: String },

    #[error("nvtx-parent sidecar column {column} in {path} has type {actual}; expected {expected}")]
    NvtxParentColumnTypeMismatch {
        path: String,
        column: String,
        expected: String,
        actual: String,
    },

    #[error("nvtx-parent GPU activity parquet missing {column} column in {path}")]
    NvtxParentGpuActivityColumnMissing { path: String, column: String },

    #[error("nvtx-parent GPU activity parquet could not be opened: {path}")]
    NvtxParentGpuActivityOpen {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("nvtx-parent GPU activity parquet reader could not be opened for {path}")]
    NvtxParentGpuActivityReaderOpen {
        path: String,
        #[source]
        source: parquet::errors::ParquetError,
    },

    #[error("nvtx-parent GPU activity parquet reader could not be built for {path}")]
    NvtxParentGpuActivityReaderBuild {
        path: String,
        #[source]
        source: parquet::errors::ParquetError,
    },

    #[error("nvtx-parent GPU activity parquet reader could not read batch from {path}")]
    NvtxParentGpuActivityBatchRead {
        path: String,
        #[source]
        source: arrow::error::ArrowError,
    },

    #[error("{path}: {column} value {value} does not fit Int32")]
    NvtxParentInt32Overflow {
        path: String,
        column: String,
        value: i64,
    },

    #[error("nvtx-parent could not fingerprint trace: {path}")]
    NvtxParentTraceFingerprint {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("nvtx-parent sidecar RecordBatch could not be assembled")]
    NvtxParentRecordBatch {
        #[source]
        source: arrow::error::ArrowError,
    },

    #[error("nvtx-parent sidecar temp file could not be created: {path}")]
    NvtxParentSidecarCreate {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("nvtx-parent parquet writer could not be opened for {path}")]
    NvtxParentWriterOpen {
        path: String,
        #[source]
        source: parquet::errors::ParquetError,
    },

    #[error("nvtx-parent parquet writer could not write batch to {path}")]
    NvtxParentWriterWrite {
        path: String,
        #[source]
        source: parquet::errors::ParquetError,
    },

    #[error("nvtx-parent parquet writer could not close {path}")]
    NvtxParentWriterClose {
        path: String,
        #[source]
        source: parquet::errors::ParquetError,
    },

    #[error("nvtx-parent sidecar could not be opened: {path}")]
    NvtxParentSidecarOpen {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("nvtx-parent parquet reader could not be opened for {path}")]
    NvtxParentReaderOpen {
        path: String,
        #[source]
        source: parquet::errors::ParquetError,
    },

    #[error("nvtx-parent parquet reader could not be built for {path}")]
    NvtxParentReaderBuild {
        path: String,
        #[source]
        source: parquet::errors::ParquetError,
    },

    #[error("nvtx-parent parquet reader could not read batch from {path}")]
    NvtxParentBatchRead {
        path: String,
        #[source]
        source: arrow::error::ArrowError,
    },

    #[error("nvtx-tree sidecar missing {column} column in {path}")]
    NvtxTreeColumnMissing { path: String, column: String },

    #[error("nvtx-tree sidecar column {column} in {path} has type {actual}; expected {expected}")]
    NvtxTreeColumnTypeMismatch {
        path: String,
        column: String,
        expected: String,
        actual: String,
    },

    #[error("nvtx-tree could not fingerprint trace: {path}")]
    NvtxTreeTraceFingerprint {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("nvtx-tree sidecar RecordBatch could not be assembled")]
    NvtxTreeRecordBatch {
        #[source]
        source: arrow::error::ArrowError,
    },

    #[error("nvtx-tree sidecar temp file could not be created: {path}")]
    NvtxTreeSidecarCreate {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("nvtx-tree parquet writer could not be opened for {path}")]
    NvtxTreeWriterOpen {
        path: String,
        #[source]
        source: parquet::errors::ParquetError,
    },

    #[error("nvtx-tree parquet writer could not write batch to {path}")]
    NvtxTreeWriterWrite {
        path: String,
        #[source]
        source: parquet::errors::ParquetError,
    },

    #[error("nvtx-tree parquet writer could not close {path}")]
    NvtxTreeWriterClose {
        path: String,
        #[source]
        source: parquet::errors::ParquetError,
    },

    #[error("nvtx-tree sidecar could not be opened: {path}")]
    NvtxTreeSidecarOpen {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("nvtx-tree parquet reader could not be opened for {path}")]
    NvtxTreeReaderOpen {
        path: String,
        #[source]
        source: parquet::errors::ParquetError,
    },

    #[error("nvtx-tree parquet reader could not be built for {path}")]
    NvtxTreeReaderBuild {
        path: String,
        #[source]
        source: parquet::errors::ParquetError,
    },

    #[error("nvtx-tree parquet reader could not read batch from {path}")]
    NvtxTreeBatchRead {
        path: String,
        #[source]
        source: arrow::error::ArrowError,
    },

    #[error("gpu-work-events sidecar missing {column} column in {path}")]
    GpuWorkEventsColumnMissing { path: String, column: String },

    #[error(
        "gpu-work-events sidecar column {column} in {path} has type {actual}; expected {expected}"
    )]
    GpuWorkEventsColumnTypeMismatch {
        path: String,
        column: String,
        expected: String,
        actual: String,
    },

    #[error("gpu-work-events could not fingerprint trace: {path}")]
    GpuWorkEventsTraceFingerprint {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("gpu-work-events sidecar RecordBatch could not be assembled")]
    GpuWorkEventsRecordBatch {
        #[source]
        source: arrow::error::ArrowError,
    },

    #[error("gpu-work-events sidecar temp file could not be created: {path}")]
    GpuWorkEventsSidecarCreate {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("gpu-work-events parquet writer could not be opened for {path}")]
    GpuWorkEventsWriterOpen {
        path: String,
        #[source]
        source: parquet::errors::ParquetError,
    },

    #[error("gpu-work-events parquet writer could not write batch to {path}")]
    GpuWorkEventsWriterWrite {
        path: String,
        #[source]
        source: parquet::errors::ParquetError,
    },

    #[error("gpu-work-events parquet writer could not close {path}")]
    GpuWorkEventsWriterClose {
        path: String,
        #[source]
        source: parquet::errors::ParquetError,
    },

    #[error("gpu-work-events sidecar could not be opened: {path}")]
    GpuWorkEventsSidecarOpen {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("gpu-work-events parquet reader could not be opened for {path}")]
    GpuWorkEventsReaderOpen {
        path: String,
        #[source]
        source: parquet::errors::ParquetError,
    },

    #[error("gpu-work-events parquet reader could not be built for {path}")]
    GpuWorkEventsReaderBuild {
        path: String,
        #[source]
        source: parquet::errors::ParquetError,
    },

    #[error("gpu-work-events parquet reader could not read batch from {path}")]
    GpuWorkEventsBatchRead {
        path: String,
        #[source]
        source: arrow::error::ArrowError,
    },

    #[error(
        "no schema adapter matched the trace; StandardAdapter did not recognise the canonical 3.x columns"
    )]
    SchemaAdapterUnmatched,
}

impl NsysDataError {
    pub fn trace_not_found(path: impl std::fmt::Display) -> Self {
        Self::TraceNotFound {
            path: path.to_string(),
        }
    }

    pub fn generated_parquetdir_source_missing(source: impl std::fmt::Display) -> Self {
        Self::GeneratedParquetdirSourceMissing {
            source_path: source.to_string(),
        }
    }

    pub fn parquetdir_not_found(path: impl std::fmt::Display) -> Self {
        Self::ParquetdirNotFound {
            path: path.to_string(),
        }
    }

    pub fn duckdb_thread_config(source: duckdb::Error) -> Self {
        Self::DuckdbThreadConfig { source }
    }

    pub fn duckdb_open_in_memory(source: duckdb::Error) -> Self {
        Self::DuckdbOpenInMemory { source }
    }

    pub fn duckdb_schema_create(source: duckdb::Error) -> Self {
        Self::DuckdbSchemaCreate { source }
    }

    pub fn parquetdir_read(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::ParquetdirRead {
            path: path.to_string(),
            source,
        }
    }

    pub fn parquet_path_invalid_utf8(path: impl std::fmt::Display) -> Self {
        Self::ParquetPathInvalidUtf8 {
            path: path.to_string(),
        }
    }

    pub fn parquet_view_create(
        table: impl Into<String>,
        path: impl std::fmt::Display,
        source: duckdb::Error,
    ) -> Self {
        Self::ParquetViewCreate {
            table: table.into(),
            path: path.to_string(),
            source,
        }
    }

    pub fn nvtx_tree_view_register(path: impl std::fmt::Display, source: duckdb::Error) -> Self {
        Self::NvtxTreeViewRegister {
            path: path.to_string(),
            source,
        }
    }

    pub fn gpu_work_events_view_register(
        path: impl std::fmt::Display,
        source: duckdb::Error,
    ) -> Self {
        Self::GpuWorkEventsViewRegister {
            path: path.to_string(),
            source,
        }
    }

    pub fn sidecar_cache(source: veloq_data::DataError) -> Self {
        Self::SidecarCache { source }
    }

    pub fn sidecar_operation(source: veloq_core::SidecarError) -> Self {
        Self::SidecarOperation { source }
    }

    pub fn sidecar_header(source: veloq_core::SidecarError) -> Self {
        Self::SidecarHeader { source }
    }

    pub fn trace_fingerprint_read(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::TraceFingerprintRead {
            path: path.to_string(),
            source,
        }
    }

    pub fn sqlite_input_unsupported(path: impl std::fmt::Display) -> Self {
        Self::SqliteInputUnsupported {
            path: path.to_string(),
        }
    }

    pub fn nsys_export_spawn(command: impl Into<String>, source: std::io::Error) -> Self {
        Self::NsysExportSpawn {
            command: command.into(),
            source,
        }
    }

    pub fn nsys_export_failed(
        command: impl Into<String>,
        exit_code: Option<i32>,
        source_path: impl std::fmt::Display,
    ) -> Self {
        Self::NsysExportFailed {
            command: command.into(),
            exit_code,
            source_path: source_path.to_string(),
        }
    }

    pub fn nsys_export_output_missing(expected: impl Into<String>) -> Self {
        Self::NsysExportOutputMissing {
            expected: expected.into(),
        }
    }

    pub fn artifact_dir_create(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::ArtifactDirCreate {
            path: path.to_string(),
            source,
        }
    }

    pub fn nsys_export_lockfile_open(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::NsysExportLockfileOpen {
            path: path.to_string(),
            source,
        }
    }

    pub fn nsys_export_lock_acquire(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::NsysExportLockAcquire {
            path: path.to_string(),
            source,
        }
    }

    pub fn nsys_parquetdir_stale_remove(
        path: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::NsysParquetdirStaleRemove {
            path: path.to_string(),
            source,
        }
    }

    pub fn nsys_parquetdir_publish(
        from: impl std::fmt::Display,
        to: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::NsysParquetdirPublish {
            from: from.to_string(),
            to: to.to_string(),
            source,
        }
    }

    pub fn nsys_cache_source_stat(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::NsysCacheSourceStat {
            path: path.to_string(),
            source,
        }
    }

    pub fn nsys_cache_sentinel_stat(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::NsysCacheSentinelStat {
            path: path.to_string(),
            source,
        }
    }

    pub fn nsys_parquetdir_help_spawn(command: impl Into<String>, source: std::io::Error) -> Self {
        Self::NsysParquetdirHelpSpawn {
            command: command.into(),
            source,
        }
    }

    pub fn nsys_version_spawn(command: impl Into<String>, source: std::io::Error) -> Self {
        Self::NsysVersionSpawn {
            command: command.into(),
            source,
        }
    }

    pub fn nsys_export_generated_path_remove(
        path: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::NsysExportGeneratedPathRemove {
            path: path.to_string(),
            source,
        }
    }

    pub fn nsys_export_source_canonicalize(
        path: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::NsysExportSourceCanonicalize {
            path: path.to_string(),
            source,
        }
    }

    pub fn nsys_parquetdir_unsupported(detected: impl Into<String>) -> Self {
        Self::NsysParquetdirUnsupported {
            detected: detected.into(),
        }
    }

    pub fn nsys_parquetdir_probe_failed(message: impl Into<String>) -> Self {
        Self::NsysParquetdirProbeFailed {
            message: message.into(),
        }
    }

    pub fn scope_conflicting_device_flags(device: i32) -> Self {
        Self::ScopeConflictingDeviceFlags { device }
    }

    pub fn scope_stream_requires_device(stream: i64) -> Self {
        Self::ScopeStreamRequiresDevice { stream }
    }

    pub fn cuda_process_unresolved(
        table: impl Into<String>,
        device_id: i32,
        context_id: i64,
        correlation_id: Option<i64>,
    ) -> Self {
        Self::CudaProcessUnresolved {
            table: table.into(),
            device_id,
            context_id,
            correlation_id,
        }
    }

    pub fn scope_device_probe_column_missing(
        table: impl Into<String>,
        column: impl Into<String>,
        source: duckdb::Error,
    ) -> Self {
        Self::ScopeDeviceProbeColumnMissing {
            table: table.into(),
            column: column.into(),
            source,
        }
    }

    pub fn duckdb(
        area: &'static str,
        phase: DuckdbPhase,
        label: impl Into<String>,
        source: duckdb::Error,
    ) -> Self {
        Self::Duckdb {
            area,
            phase,
            label: label.into(),
            source,
        }
    }

    pub fn duckdb_prepare(
        area: &'static str,
        label: impl Into<String>,
        source: duckdb::Error,
    ) -> Self {
        Self::duckdb(area, DuckdbPhase::Prepare, label, source)
    }

    pub fn duckdb_query(
        area: &'static str,
        label: impl Into<String>,
        source: duckdb::Error,
    ) -> Self {
        Self::duckdb(area, DuckdbPhase::Query, label, source)
    }

    pub fn duckdb_read(
        area: &'static str,
        label: impl Into<String>,
        source: duckdb::Error,
    ) -> Self {
        Self::duckdb(area, DuckdbPhase::Read, label, source)
    }

    pub fn duckdb_parts(&self) -> Option<(&'static str, DuckdbPhase, &str)> {
        match self {
            Self::Duckdb {
                area, phase, label, ..
            } => Some((*area, *phase, label.as_str())),
            _ => None,
        }
    }

    pub fn scope_device_probe_query(table: impl Into<String>, source: duckdb::Error) -> Self {
        Self::duckdb_query("scope device probe", table, source)
    }

    pub fn scope_device_probe_read(table: impl Into<String>, source: duckdb::Error) -> Self {
        Self::duckdb_read("scope device probe", table, source)
    }

    pub fn meta_cache_count_prepare(table: impl Into<String>, source: duckdb::Error) -> Self {
        Self::duckdb_prepare("meta cache", table, source)
    }

    pub fn meta_cache_count_read(table: impl Into<String>, source: duckdb::Error) -> Self {
        Self::duckdb_read("meta cache", table, source)
    }

    pub fn meta_cache_slot_uninitialised() -> Self {
        Self::MetaCacheSlotUninitialised
    }

    pub fn export_metadata_prepare(table: impl Into<String>, source: duckdb::Error) -> Self {
        Self::duckdb_prepare("export metadata", table, source)
    }

    pub fn export_metadata_query(table: impl Into<String>, source: duckdb::Error) -> Self {
        Self::duckdb_query("export metadata", table, source)
    }

    pub fn export_metadata_read(table: impl Into<String>, source: duckdb::Error) -> Self {
        Self::duckdb_read("export metadata", table, source)
    }

    pub fn trace_map_probe_column_missing(
        table: impl Into<String>,
        column: impl Into<String>,
        source: duckdb::Error,
    ) -> Self {
        Self::TraceMapProbeColumnMissing {
            table: table.into(),
            column: column.into(),
            source,
        }
    }

    pub fn trace_map_rows_query(table: impl Into<String>, source: duckdb::Error) -> Self {
        Self::duckdb_query("trace map", table, source)
    }

    pub fn trace_map_rows_read(table: impl Into<String>, source: duckdb::Error) -> Self {
        Self::duckdb_read("trace map", table, source)
    }

    pub fn trace_origins_prepare(table: impl Into<String>, source: duckdb::Error) -> Self {
        Self::duckdb_prepare("trace origins", table, source)
    }

    pub fn trace_origins_read(table: impl Into<String>, source: duckdb::Error) -> Self {
        Self::duckdb_read("trace origins", table, source)
    }

    pub fn trace_sample_span_prepare(table: impl Into<String>, source: duckdb::Error) -> Self {
        Self::duckdb_prepare("trace sample span", table, source)
    }

    pub fn trace_sample_span_read(table: impl Into<String>, source: duckdb::Error) -> Self {
        Self::duckdb_read("trace sample span", table, source)
    }

    pub fn time_range_empty(source: veloq_core::time::TimeParseError) -> Self {
        Self::TimeRangeEmpty { source }
    }

    pub fn nvtx_nesting_rows_prepare(source: duckdb::Error) -> Self {
        Self::duckdb_prepare("nvtx nesting", "rows", source)
    }

    pub fn nvtx_nesting_rows_query(source: duckdb::Error) -> Self {
        Self::duckdb_query("nvtx nesting", "rows", source)
    }

    pub fn nvtx_nesting_rows_read(source: duckdb::Error) -> Self {
        Self::duckdb_read("nvtx nesting", "rows", source)
    }

    pub fn nvtx_tree_rows_prepare(source: duckdb::Error) -> Self {
        Self::duckdb_prepare("nvtx tree", "rows", source)
    }

    pub fn nvtx_tree_rows_query(source: duckdb::Error) -> Self {
        Self::duckdb_query("nvtx tree", "rows", source)
    }

    pub fn nvtx_tree_rows_read(source: duckdb::Error) -> Self {
        Self::duckdb_read("nvtx tree", "rows", source)
    }

    pub fn correlation_trace_fingerprint(
        path: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::CorrelationTraceFingerprint {
            path: path.to_string(),
            source,
        }
    }

    pub fn correlation_scan_prepare(table: impl Into<String>, source: duckdb::Error) -> Self {
        Self::duckdb_prepare("correlation scan", table, source)
    }

    pub fn correlation_scan_query(table: impl Into<String>, source: duckdb::Error) -> Self {
        Self::duckdb_query("correlation scan", table, source)
    }

    pub fn correlation_scan_read(table: impl Into<String>, source: duckdb::Error) -> Self {
        Self::duckdb_read("correlation scan", table, source)
    }

    pub fn hardware_invalid_u64_id(
        label: impl Into<String>,
        value: impl Into<String>,
        source: std::num::ParseIntError,
    ) -> Self {
        Self::HardwareInvalidU64Id {
            label: label.into(),
            value: value.into(),
            source,
        }
    }

    pub fn hardware_rows_prepare(table: impl Into<String>, source: duckdb::Error) -> Self {
        Self::duckdb_prepare("hardware rows", table, source)
    }

    pub fn hardware_rows_query(table: impl Into<String>, source: duckdb::Error) -> Self {
        Self::duckdb_query("hardware rows", table, source)
    }

    pub fn hardware_rows_read(table: impl Into<String>, source: duckdb::Error) -> Self {
        Self::duckdb_read("hardware rows", table, source)
    }

    pub fn nvtx_parent_unsupported_integer_type(
        path: impl std::fmt::Display,
        column: impl Into<String>,
        data_type: impl Into<String>,
    ) -> Self {
        Self::NvtxParentUnsupportedIntegerType {
            path: path.to_string(),
            column: column.into(),
            data_type: data_type.into(),
        }
    }

    pub fn nvtx_parent_integer_overflow(
        path: impl std::fmt::Display,
        column: impl Into<String>,
        value: u64,
    ) -> Self {
        Self::NvtxParentIntegerOverflow {
            path: path.to_string(),
            column: column.into(),
            value,
        }
    }

    pub fn nvtx_parent_list_length_mismatch(
        row: usize,
        rowids_len: usize,
        names_len: usize,
    ) -> Self {
        Self::NvtxParentListLengthMismatch {
            row,
            rowids_len,
            names_len,
        }
    }

    pub fn nvtx_parent_column_missing(
        path: impl std::fmt::Display,
        column: impl Into<String>,
    ) -> Self {
        Self::NvtxParentColumnMissing {
            path: path.to_string(),
            column: column.into(),
        }
    }

    pub fn nvtx_parent_column_type_mismatch(
        path: impl std::fmt::Display,
        column: impl Into<String>,
        expected: impl Into<String>,
        actual: impl Into<String>,
    ) -> Self {
        Self::NvtxParentColumnTypeMismatch {
            path: path.to_string(),
            column: column.into(),
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    pub fn nvtx_parent_gpu_activity_column_missing(
        path: impl std::fmt::Display,
        column: impl Into<String>,
    ) -> Self {
        Self::NvtxParentGpuActivityColumnMissing {
            path: path.to_string(),
            column: column.into(),
        }
    }

    pub fn nvtx_parent_gpu_activity_open(
        path: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::NvtxParentGpuActivityOpen {
            path: path.to_string(),
            source,
        }
    }

    pub fn nvtx_parent_gpu_activity_reader_open(
        path: impl std::fmt::Display,
        source: parquet::errors::ParquetError,
    ) -> Self {
        Self::NvtxParentGpuActivityReaderOpen {
            path: path.to_string(),
            source,
        }
    }

    pub fn nvtx_parent_gpu_activity_reader_build(
        path: impl std::fmt::Display,
        source: parquet::errors::ParquetError,
    ) -> Self {
        Self::NvtxParentGpuActivityReaderBuild {
            path: path.to_string(),
            source,
        }
    }

    pub fn nvtx_parent_gpu_activity_batch_read(
        path: impl std::fmt::Display,
        source: arrow::error::ArrowError,
    ) -> Self {
        Self::NvtxParentGpuActivityBatchRead {
            path: path.to_string(),
            source,
        }
    }

    pub fn nvtx_parent_int32_overflow(
        path: impl std::fmt::Display,
        column: impl Into<String>,
        value: i64,
    ) -> Self {
        Self::NvtxParentInt32Overflow {
            path: path.to_string(),
            column: column.into(),
            value,
        }
    }

    pub fn nvtx_parent_rows_prepare(table: impl Into<String>, source: duckdb::Error) -> Self {
        Self::duckdb_prepare("nvtx-parent rows", table, source)
    }

    pub fn nvtx_parent_rows_query(table: impl Into<String>, source: duckdb::Error) -> Self {
        Self::duckdb_query("nvtx-parent rows", table, source)
    }

    pub fn nvtx_parent_rows_read(table: impl Into<String>, source: duckdb::Error) -> Self {
        Self::duckdb_read("nvtx-parent rows", table, source)
    }

    pub fn nvtx_parent_trace_fingerprint(
        path: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::NvtxParentTraceFingerprint {
            path: path.to_string(),
            source,
        }
    }

    pub fn nvtx_parent_record_batch(source: arrow::error::ArrowError) -> Self {
        Self::NvtxParentRecordBatch { source }
    }

    pub fn nvtx_parent_sidecar_create(
        path: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::NvtxParentSidecarCreate {
            path: path.to_string(),
            source,
        }
    }

    pub fn nvtx_parent_writer_open(
        path: impl std::fmt::Display,
        source: parquet::errors::ParquetError,
    ) -> Self {
        Self::NvtxParentWriterOpen {
            path: path.to_string(),
            source,
        }
    }

    pub fn nvtx_parent_writer_write(
        path: impl std::fmt::Display,
        source: parquet::errors::ParquetError,
    ) -> Self {
        Self::NvtxParentWriterWrite {
            path: path.to_string(),
            source,
        }
    }

    pub fn nvtx_parent_writer_close(
        path: impl std::fmt::Display,
        source: parquet::errors::ParquetError,
    ) -> Self {
        Self::NvtxParentWriterClose {
            path: path.to_string(),
            source,
        }
    }

    pub fn nvtx_parent_sidecar_open(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::NvtxParentSidecarOpen {
            path: path.to_string(),
            source,
        }
    }

    pub fn nvtx_parent_reader_open(
        path: impl std::fmt::Display,
        source: parquet::errors::ParquetError,
    ) -> Self {
        Self::NvtxParentReaderOpen {
            path: path.to_string(),
            source,
        }
    }

    pub fn nvtx_parent_reader_build(
        path: impl std::fmt::Display,
        source: parquet::errors::ParquetError,
    ) -> Self {
        Self::NvtxParentReaderBuild {
            path: path.to_string(),
            source,
        }
    }

    pub fn nvtx_parent_batch_read(
        path: impl std::fmt::Display,
        source: arrow::error::ArrowError,
    ) -> Self {
        Self::NvtxParentBatchRead {
            path: path.to_string(),
            source,
        }
    }

    pub fn nvtx_tree_column_missing(
        path: impl std::fmt::Display,
        column: impl Into<String>,
    ) -> Self {
        Self::NvtxTreeColumnMissing {
            path: path.to_string(),
            column: column.into(),
        }
    }

    pub fn nvtx_tree_column_type_mismatch(
        path: impl std::fmt::Display,
        column: impl Into<String>,
        expected: impl Into<String>,
        actual: impl Into<String>,
    ) -> Self {
        Self::NvtxTreeColumnTypeMismatch {
            path: path.to_string(),
            column: column.into(),
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    pub fn nvtx_tree_trace_fingerprint(
        path: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::NvtxTreeTraceFingerprint {
            path: path.to_string(),
            source,
        }
    }

    pub fn nvtx_tree_record_batch(source: arrow::error::ArrowError) -> Self {
        Self::NvtxTreeRecordBatch { source }
    }

    pub fn nvtx_tree_sidecar_create(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::NvtxTreeSidecarCreate {
            path: path.to_string(),
            source,
        }
    }

    pub fn nvtx_tree_writer_open(
        path: impl std::fmt::Display,
        source: parquet::errors::ParquetError,
    ) -> Self {
        Self::NvtxTreeWriterOpen {
            path: path.to_string(),
            source,
        }
    }

    pub fn nvtx_tree_writer_write(
        path: impl std::fmt::Display,
        source: parquet::errors::ParquetError,
    ) -> Self {
        Self::NvtxTreeWriterWrite {
            path: path.to_string(),
            source,
        }
    }

    pub fn nvtx_tree_writer_close(
        path: impl std::fmt::Display,
        source: parquet::errors::ParquetError,
    ) -> Self {
        Self::NvtxTreeWriterClose {
            path: path.to_string(),
            source,
        }
    }

    pub fn nvtx_tree_sidecar_open(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::NvtxTreeSidecarOpen {
            path: path.to_string(),
            source,
        }
    }

    pub fn nvtx_tree_reader_open(
        path: impl std::fmt::Display,
        source: parquet::errors::ParquetError,
    ) -> Self {
        Self::NvtxTreeReaderOpen {
            path: path.to_string(),
            source,
        }
    }

    pub fn nvtx_tree_reader_build(
        path: impl std::fmt::Display,
        source: parquet::errors::ParquetError,
    ) -> Self {
        Self::NvtxTreeReaderBuild {
            path: path.to_string(),
            source,
        }
    }

    pub fn nvtx_tree_batch_read(
        path: impl std::fmt::Display,
        source: arrow::error::ArrowError,
    ) -> Self {
        Self::NvtxTreeBatchRead {
            path: path.to_string(),
            source,
        }
    }

    pub fn gpu_work_events_rows_prepare(table: impl Into<String>, source: duckdb::Error) -> Self {
        Self::duckdb_prepare("gpu work events", table, source)
    }

    pub fn gpu_work_events_rows_query(table: impl Into<String>, source: duckdb::Error) -> Self {
        Self::duckdb_query("gpu work events", table, source)
    }

    pub fn gpu_work_events_rows_read(table: impl Into<String>, source: duckdb::Error) -> Self {
        Self::duckdb_read("gpu work events", table, source)
    }

    pub fn gpu_work_events_column_missing(
        path: impl std::fmt::Display,
        column: impl Into<String>,
    ) -> Self {
        Self::GpuWorkEventsColumnMissing {
            path: path.to_string(),
            column: column.into(),
        }
    }

    pub fn gpu_work_events_column_type_mismatch(
        path: impl std::fmt::Display,
        column: impl Into<String>,
        expected: impl Into<String>,
        actual: impl Into<String>,
    ) -> Self {
        Self::GpuWorkEventsColumnTypeMismatch {
            path: path.to_string(),
            column: column.into(),
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    pub fn gpu_work_events_trace_fingerprint(
        path: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::GpuWorkEventsTraceFingerprint {
            path: path.to_string(),
            source,
        }
    }

    pub fn gpu_work_events_record_batch(source: arrow::error::ArrowError) -> Self {
        Self::GpuWorkEventsRecordBatch { source }
    }

    pub fn gpu_work_events_sidecar_create(
        path: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::GpuWorkEventsSidecarCreate {
            path: path.to_string(),
            source,
        }
    }

    pub fn gpu_work_events_writer_open(
        path: impl std::fmt::Display,
        source: parquet::errors::ParquetError,
    ) -> Self {
        Self::GpuWorkEventsWriterOpen {
            path: path.to_string(),
            source,
        }
    }

    pub fn gpu_work_events_writer_write(
        path: impl std::fmt::Display,
        source: parquet::errors::ParquetError,
    ) -> Self {
        Self::GpuWorkEventsWriterWrite {
            path: path.to_string(),
            source,
        }
    }

    pub fn gpu_work_events_writer_close(
        path: impl std::fmt::Display,
        source: parquet::errors::ParquetError,
    ) -> Self {
        Self::GpuWorkEventsWriterClose {
            path: path.to_string(),
            source,
        }
    }

    pub fn gpu_work_events_sidecar_open(
        path: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::GpuWorkEventsSidecarOpen {
            path: path.to_string(),
            source,
        }
    }

    pub fn gpu_work_events_reader_open(
        path: impl std::fmt::Display,
        source: parquet::errors::ParquetError,
    ) -> Self {
        Self::GpuWorkEventsReaderOpen {
            path: path.to_string(),
            source,
        }
    }

    pub fn gpu_work_events_reader_build(
        path: impl std::fmt::Display,
        source: parquet::errors::ParquetError,
    ) -> Self {
        Self::GpuWorkEventsReaderBuild {
            path: path.to_string(),
            source,
        }
    }

    pub fn gpu_work_events_batch_read(
        path: impl std::fmt::Display,
        source: arrow::error::ArrowError,
    ) -> Self {
        Self::GpuWorkEventsBatchRead {
            path: path.to_string(),
            source,
        }
    }
}

impl From<veloq_data::DataError> for NsysDataError {
    fn from(source: veloq_data::DataError) -> Self {
        Self::sidecar_cache(source)
    }
}

impl VeloqDiagnostic for NsysDataError {
    fn code(&self) -> ErrorCode {
        match self {
            Self::TraceNotFound { .. } => ErrorCode::new("nsys.data.trace-not-found"),
            Self::GeneratedParquetdirSourceMissing { .. } => {
                ErrorCode::new("nsys.data.generated-parquetdir-source-missing")
            }
            Self::ParquetdirNotFound { .. } => ErrorCode::new("nsys.data.parquetdir-not-found"),
            Self::DuckdbThreadConfig { .. } => ErrorCode::new("nsys.data.duckdb-thread-config"),
            Self::DuckdbOpenInMemory { .. } => ErrorCode::new("nsys.data.duckdb-open-in-memory"),
            Self::DuckdbSchemaCreate { .. } => ErrorCode::new("nsys.data.duckdb-schema-create"),
            Self::ParquetdirRead { .. } => ErrorCode::new("nsys.data.parquetdir-read"),
            Self::ParquetPathInvalidUtf8 { .. } => {
                ErrorCode::new("nsys.data.parquet-path-invalid-utf8")
            }
            Self::ParquetViewCreate { .. } => ErrorCode::new("nsys.data.parquet-view-create"),
            Self::NvtxTreeViewRegister { .. } => {
                ErrorCode::new("nsys.data.nvtx-tree-view-register")
            }
            Self::GpuWorkEventsViewRegister { .. } => {
                ErrorCode::new("nsys.data.gpu-work-events-view-register")
            }
            Self::SidecarCache { .. } => ErrorCode::new("nsys.data.sidecar-cache"),
            Self::SidecarOperation { .. } => ErrorCode::new("nsys.data.sidecar-operation"),
            Self::SidecarHeader { .. } => ErrorCode::new("nsys.data.sidecar-header"),
            Self::TraceFingerprintRead { .. } => ErrorCode::new("nsys.data.trace-fingerprint-read"),
            Self::SqliteInputUnsupported { .. } => {
                ErrorCode::new("nsys.data.sqlite-input-unsupported")
            }
            Self::NsysExportSpawn { .. } => ErrorCode::new("nsys.data.nsys-export-spawn"),
            Self::NsysExportFailed { .. } => ErrorCode::new("nsys.data.nsys-export-failed"),
            Self::NsysExportOutputMissing { .. } => {
                ErrorCode::new("nsys.data.nsys-export-output-missing")
            }
            Self::ArtifactDirCreate { .. } => ErrorCode::new("nsys.data.artifact-dir-create"),
            Self::NsysExportLockfileOpen { .. } => {
                ErrorCode::new("nsys.data.nsys-export-lockfile-open")
            }
            Self::NsysExportLockAcquire { .. } => {
                ErrorCode::new("nsys.data.nsys-export-lock-acquire")
            }
            Self::NsysParquetdirStaleRemove { .. } => {
                ErrorCode::new("nsys.data.nsys-parquetdir-stale-remove")
            }
            Self::NsysParquetdirPublish { .. } => {
                ErrorCode::new("nsys.data.nsys-parquetdir-publish")
            }
            Self::NsysCacheSourceStat { .. } => ErrorCode::new("nsys.data.nsys-cache-source-stat"),
            Self::NsysCacheSentinelStat { .. } => {
                ErrorCode::new("nsys.data.nsys-cache-sentinel-stat")
            }
            Self::NsysParquetdirHelpSpawn { .. } => {
                ErrorCode::new("nsys.data.nsys-parquetdir-help-spawn")
            }
            Self::NsysVersionSpawn { .. } => ErrorCode::new("nsys.data.nsys-version-spawn"),
            Self::NsysExportGeneratedPathRemove { .. } => {
                ErrorCode::new("nsys.data.nsys-export-generated-path-remove")
            }
            Self::NsysExportSourceCanonicalize { .. } => {
                ErrorCode::new("nsys.data.nsys-export-source-canonicalize")
            }
            Self::NsysParquetdirUnsupported { .. } => {
                ErrorCode::new("nsys.data.nsys-parquetdir-unsupported")
            }
            Self::NsysParquetdirProbeFailed { .. } => {
                ErrorCode::new("nsys.data.nsys-parquetdir-probe-failed")
            }
            Self::ScopeConflictingDeviceFlags { .. } => {
                ErrorCode::new("nsys.data.scope-conflicting-device-flags")
            }
            Self::ScopeStreamRequiresDevice { .. } => {
                ErrorCode::new("nsys.data.scope-stream-requires-device")
            }
            Self::CudaProcessUnresolved { .. } => {
                ErrorCode::new("nsys.data.cuda-process-unresolved")
            }
            Self::ScopeDeviceProbeColumnMissing { .. } => {
                ErrorCode::new("nsys.data.scope-device-probe-column-missing")
            }
            Self::Duckdb { phase, .. } => phase.code(),
            Self::MetaCacheSlotUninitialised => {
                ErrorCode::new("nsys.data.meta-cache-slot-uninitialised")
            }
            Self::TraceMapProbeColumnMissing { .. } => {
                ErrorCode::new("nsys.data.trace-map-probe-column-missing")
            }
            Self::TimeRangeEmpty { .. } => ErrorCode::new("nsys.data.time-range-empty"),
            Self::CorrelationTraceFingerprint { .. } => {
                ErrorCode::new("nsys.data.correlation-trace-fingerprint")
            }
            Self::HardwareInvalidU64Id { .. } => {
                ErrorCode::new("nsys.data.hardware-invalid-u64-id")
            }
            Self::NvtxParentUnsupportedIntegerType { .. } => {
                ErrorCode::new("nsys.data.nvtx-parent-unsupported-integer-type")
            }
            Self::NvtxParentIntegerOverflow { .. } => {
                ErrorCode::new("nsys.data.nvtx-parent-integer-overflow")
            }
            Self::NvtxParentListLengthMismatch { .. } => {
                ErrorCode::new("nsys.data.nvtx-parent-list-length-mismatch")
            }
            Self::NvtxParentColumnMissing { .. } => {
                ErrorCode::new("nsys.data.nvtx-parent-column-missing")
            }
            Self::NvtxParentColumnTypeMismatch { .. } => {
                ErrorCode::new("nsys.data.nvtx-parent-column-type-mismatch")
            }
            Self::NvtxParentGpuActivityColumnMissing { .. } => {
                ErrorCode::new("nsys.data.nvtx-parent-gpu-activity-column-missing")
            }
            Self::NvtxParentGpuActivityOpen { .. } => {
                ErrorCode::new("nsys.data.nvtx-parent-gpu-activity-open")
            }
            Self::NvtxParentGpuActivityReaderOpen { .. } => {
                ErrorCode::new("nsys.data.nvtx-parent-gpu-activity-reader-open")
            }
            Self::NvtxParentGpuActivityReaderBuild { .. } => {
                ErrorCode::new("nsys.data.nvtx-parent-gpu-activity-reader-build")
            }
            Self::NvtxParentGpuActivityBatchRead { .. } => {
                ErrorCode::new("nsys.data.nvtx-parent-gpu-activity-batch-read")
            }
            Self::NvtxParentInt32Overflow { .. } => {
                ErrorCode::new("nsys.data.nvtx-parent-int32-overflow")
            }
            Self::NvtxParentTraceFingerprint { .. } => {
                ErrorCode::new("nsys.data.nvtx-parent-trace-fingerprint")
            }
            Self::NvtxParentRecordBatch { .. } => {
                ErrorCode::new("nsys.data.nvtx-parent-record-batch")
            }
            Self::NvtxParentSidecarCreate { .. } => {
                ErrorCode::new("nsys.data.nvtx-parent-sidecar-create")
            }
            Self::NvtxParentWriterOpen { .. } => {
                ErrorCode::new("nsys.data.nvtx-parent-writer-open")
            }
            Self::NvtxParentWriterWrite { .. } => {
                ErrorCode::new("nsys.data.nvtx-parent-writer-write")
            }
            Self::NvtxParentWriterClose { .. } => {
                ErrorCode::new("nsys.data.nvtx-parent-writer-close")
            }
            Self::NvtxParentSidecarOpen { .. } => {
                ErrorCode::new("nsys.data.nvtx-parent-sidecar-open")
            }
            Self::NvtxParentReaderOpen { .. } => {
                ErrorCode::new("nsys.data.nvtx-parent-reader-open")
            }
            Self::NvtxParentReaderBuild { .. } => {
                ErrorCode::new("nsys.data.nvtx-parent-reader-build")
            }
            Self::NvtxParentBatchRead { .. } => ErrorCode::new("nsys.data.nvtx-parent-batch-read"),
            Self::NvtxTreeColumnMissing { .. } => {
                ErrorCode::new("nsys.data.nvtx-tree-column-missing")
            }
            Self::NvtxTreeColumnTypeMismatch { .. } => {
                ErrorCode::new("nsys.data.nvtx-tree-column-type-mismatch")
            }
            Self::NvtxTreeTraceFingerprint { .. } => {
                ErrorCode::new("nsys.data.nvtx-tree-trace-fingerprint")
            }
            Self::NvtxTreeRecordBatch { .. } => ErrorCode::new("nsys.data.nvtx-tree-record-batch"),
            Self::NvtxTreeSidecarCreate { .. } => {
                ErrorCode::new("nsys.data.nvtx-tree-sidecar-create")
            }
            Self::NvtxTreeWriterOpen { .. } => ErrorCode::new("nsys.data.nvtx-tree-writer-open"),
            Self::NvtxTreeWriterWrite { .. } => ErrorCode::new("nsys.data.nvtx-tree-writer-write"),
            Self::NvtxTreeWriterClose { .. } => ErrorCode::new("nsys.data.nvtx-tree-writer-close"),
            Self::NvtxTreeSidecarOpen { .. } => ErrorCode::new("nsys.data.nvtx-tree-sidecar-open"),
            Self::NvtxTreeReaderOpen { .. } => ErrorCode::new("nsys.data.nvtx-tree-reader-open"),
            Self::NvtxTreeReaderBuild { .. } => ErrorCode::new("nsys.data.nvtx-tree-reader-build"),
            Self::NvtxTreeBatchRead { .. } => ErrorCode::new("nsys.data.nvtx-tree-batch-read"),
            Self::GpuWorkEventsColumnMissing { .. } => {
                ErrorCode::new("nsys.data.gpu-work-events-column-missing")
            }
            Self::GpuWorkEventsColumnTypeMismatch { .. } => {
                ErrorCode::new("nsys.data.gpu-work-events-column-type-mismatch")
            }
            Self::GpuWorkEventsTraceFingerprint { .. } => {
                ErrorCode::new("nsys.data.gpu-work-events-trace-fingerprint")
            }
            Self::GpuWorkEventsRecordBatch { .. } => {
                ErrorCode::new("nsys.data.gpu-work-events-record-batch")
            }
            Self::GpuWorkEventsSidecarCreate { .. } => {
                ErrorCode::new("nsys.data.gpu-work-events-sidecar-create")
            }
            Self::GpuWorkEventsWriterOpen { .. } => {
                ErrorCode::new("nsys.data.gpu-work-events-writer-open")
            }
            Self::GpuWorkEventsWriterWrite { .. } => {
                ErrorCode::new("nsys.data.gpu-work-events-writer-write")
            }
            Self::GpuWorkEventsWriterClose { .. } => {
                ErrorCode::new("nsys.data.gpu-work-events-writer-close")
            }
            Self::GpuWorkEventsSidecarOpen { .. } => {
                ErrorCode::new("nsys.data.gpu-work-events-sidecar-open")
            }
            Self::GpuWorkEventsReaderOpen { .. } => {
                ErrorCode::new("nsys.data.gpu-work-events-reader-open")
            }
            Self::GpuWorkEventsReaderBuild { .. } => {
                ErrorCode::new("nsys.data.gpu-work-events-reader-build")
            }
            Self::GpuWorkEventsBatchRead { .. } => {
                ErrorCode::new("nsys.data.gpu-work-events-batch-read")
            }
            Self::SchemaAdapterUnmatched => ErrorCode::new("nsys.data.schema-adapter-unmatched"),
        }
    }
}
