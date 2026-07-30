use thiserror::Error;
use veloq_core::{ErrorCode, OutputFormat, TabularError, VeloqDiagnostic};

pub type NcuSourceResult<T> = Result<T, NcuSourceError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcuIoOperation {
    CreateDir,
    Open,
    Lock,
    Read,
    Write,
    Publish,
    CompressGzip,
    DecompressGzip,
    DecompressZstd,
}

impl NcuIoOperation {
    fn code(self) -> ErrorCode {
        match self {
            Self::CreateDir => ErrorCode::new("ncu.input.io-create-dir"),
            Self::Open => ErrorCode::new("ncu.input.io-open"),
            Self::Lock => ErrorCode::new("ncu.input.io-lock"),
            Self::Read => ErrorCode::new("ncu.input.io-read"),
            Self::Write => ErrorCode::new("ncu.input.io-write"),
            Self::Publish => ErrorCode::new("ncu.input.io-publish"),
            Self::CompressGzip => ErrorCode::new("ncu.input.gzip-compress"),
            Self::DecompressGzip => ErrorCode::new("ncu.input.gzip-decompress"),
            Self::DecompressZstd => ErrorCode::new("ncu.input.zstd-decompress"),
        }
    }
}

impl std::fmt::Display for NcuIoOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CreateDir => f.write_str("create directory"),
            Self::Open => f.write_str("open"),
            Self::Lock => f.write_str("lock"),
            Self::Read => f.write_str("read"),
            Self::Write => f.write_str("write"),
            Self::Publish => f.write_str("publish"),
            Self::CompressGzip => f.write_str("gzip"),
            Self::DecompressGzip => f.write_str("gunzip"),
            Self::DecompressZstd => f.write_str("decompress zstd"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcuJsonOperation {
    Encode,
    Decode,
}

impl NcuJsonOperation {
    fn code(self) -> ErrorCode {
        match self {
            Self::Encode => ErrorCode::new("ncu.input.json-encode"),
            Self::Decode => ErrorCode::new("ncu.input.json-decode"),
        }
    }
}

impl std::fmt::Display for NcuJsonOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Encode => f.write_str("encode JSON"),
            Self::Decode => f.write_str("decode JSON"),
        }
    }
}

#[derive(Debug, Error)]
pub enum NcuSourceError {
    #[error("veloq-ncu schema currently supports only --format json (got `{fmt}`)")]
    UnsupportedSchemaFormat { fmt: OutputFormat },

    #[error("internal: ncu verb missing trace path")]
    MissingTracePath,

    #[error("ncu report `{trace}` not found")]
    TraceNotFound { trace: String },

    #[error("unknown ncu schema target `{target}`; expected one of: {expected}")]
    UnknownSchemaTarget { target: String, expected: String },

    #[error("serializing ncu schema target `{target}`")]
    SerializeSchema {
        target: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("invalid {flag} `{value}`; expected WxHxD with integer axes")]
    InvalidLaunchDims {
        flag: &'static str,
        value: String,
        #[source]
        source: Box<NcuSourceError>,
    },

    #[error(
        "expected `WxHxD` launch dimensions (got `{value}`); pad unused axes with 0 (e.g. `1024x1x1`)"
    )]
    LaunchDimsShape { value: String },

    #[error("invalid axis `{axis}` in launch dimensions `{value}`")]
    LaunchDimsAxis {
        value: String,
        axis: String,
        #[source]
        source: std::num::ParseIntError,
    },

    #[error("unknown --by axis `{axis}` for source-metrics; expected one of: line, sass, file")]
    UnknownSourceMetricsAxis { axis: String },

    #[error("unknown --by axis `{axis}` for warp-stalls; expected one of: line, sass, reason")]
    UnknownWarpStallsAxis { axis: String },

    #[error("--limit must be at least 1 (got {limit})")]
    LimitTooSmall { limit: usize },

    #[error("--counter must be a non-empty glob (comma-separated)")]
    CounterGlobEmpty,

    #[error("ncu inspect needs at least one --row-id")]
    InspectRowIdRequired,

    #[error("expected `launch:<idx>` row_id (got `{row_id}`)")]
    LaunchRowIdMissingColon { row_id: String },

    #[error("ncu commands currently support only `launch:<idx>` row_ids (got `{kind}`)")]
    LaunchRowIdKindUnsupported { kind: String },

    #[error("invalid launch index `{index}` in `{row_id}`")]
    LaunchRowIdIndexInvalid {
        row_id: String,
        index: String,
        #[source]
        source: std::num::ParseIntError,
    },

    #[error(
        "--row-id `{row_id}` launch index {launch_idx} out of range ({launch_count} launches in this report)"
    )]
    LaunchRowIdOutOfRange {
        row_id: String,
        launch_idx: usize,
        launch_count: usize,
    },

    #[error("internal: launch:{launch_idx} vanished after bounds check")]
    LaunchVanishedAfterBoundsCheck { launch_idx: usize },

    #[error("computed cubin length {computed} exceeds available bytes {available}")]
    CubinLengthExceedsAvailableBytes { computed: usize, available: usize },

    #[error("{area} failed to {operation}: {target}")]
    InputIo {
        area: &'static str,
        operation: NcuIoOperation,
        target: String,
        #[source]
        source: std::io::Error,
    },

    #[error("{area} failed to {operation}: {target}")]
    InputJson {
        area: &'static str,
        operation: NcuJsonOperation,
        target: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("{area} was not UTF-8")]
    InputUtf8 {
        area: &'static str,
        #[source]
        source: std::string::FromUtf8Error,
    },

    #[error("ELF candidate could not be parsed")]
    CubinElfParse {
        #[source]
        source: object::Error,
    },

    #[error("cubin length {length} does not fit usize")]
    CubinLengthOverflow {
        length: u64,
        #[source]
        source: std::num::TryFromIntError,
    },

    #[error("--line requires --file (you can't pin a line without a file)")]
    SourceMetricsLineWithoutFile,

    #[error("committed native sidecar `{path}` has schema `{actual}`, expected `{expected}`")]
    NativeSidecarSchemaMismatch {
        path: String,
        actual: String,
        expected: &'static str,
    },

    #[error(
        "native sidecar `{path}` was produced by ncu_report `{version}`, which cannot decode `.ncu-repz` reports"
    )]
    NativeSidecarUnsupportedCompressedReader { path: String, version: String },

    #[error(
        "cannot ingest `{report}` without Nsight Compute: {source}. A matching native sidecar would have been used, but none is fresh. Install NCU (provides the ncu_report Python module) or run `veloq ncu prep` on a machine with NCU, then commit/copy the <report>.veloq/ sidecar"
    )]
    NativeIngestUnavailable {
        report: String,
        #[source]
        source: Box<NcuSourceError>,
    },

    #[error("helper emitted schema `{actual}`, expected `{expected}`")]
    NativeHelperSchemaMismatch {
        actual: String,
        expected: &'static str,
    },

    #[error("ncu_report export helper failed under `{program}` ({status}): {stderr}")]
    NativeHelperFailed {
        program: String,
        status: String,
        stderr: String,
    },

    #[error("ncu_report export helper could not be spawned under `{program}`")]
    NativeHelperSpawn {
        program: String,
        #[source]
        source: std::io::Error,
    },

    #[error("invoking {bin} {args} on `{cubin}`")]
    DisasmToolSpawn {
        bin: &'static str,
        args: String,
        cubin: String,
        #[source]
        source: std::io::Error,
    },

    #[error("{bin} {args} on `{cubin}` exited with status {status}: {stderr}")]
    DisasmToolFailed {
        bin: &'static str,
        args: String,
        cubin: String,
        status: String,
        stderr: String,
    },

    #[error("{bin} {stream} was not UTF-8")]
    DisasmToolOutputUtf8 {
        bin: &'static str,
        stream: &'static str,
        #[source]
        source: std::string::FromUtf8Error,
    },

    #[error("nvdisasm --emit-json output could not be decoded")]
    NvdisasmJsonDecode {
        #[source]
        source: serde_json::Error,
    },

    #[error("nvdisasm JSON top-level was not an array")]
    NvdisasmTopLevelNotArray,

    #[error("nvdisasm JSON has no kernel array at index 1")]
    NvdisasmKernelArrayMissing,

    #[error("nvdisasm kernel[{index}] missing function-name")]
    NvdisasmKernelFunctionNameMissing { index: usize },

    #[error("nvdisasm kernel[{index}] missing start")]
    NvdisasmKernelStartMissing { index: usize },

    #[error(
        "no usable Python interpreter found (tried: {tried}); set VELOQ_PYTHON to a python3 that can `import ncu_report`"
    )]
    NativePythonMissing { tried: String },

    #[error("VELOQ_NCU_REPORT_DIR={dir} does not contain ncu_report.py")]
    NativeNcuReportOverrideInvalid { dir: String },

    #[error("{message}")]
    NativeNcuReportModuleMissing { message: String },

    #[error(transparent)]
    Tabular(#[from] TabularError),

    #[error("serializing ncu response envelope")]
    SerializeEnvelope {
        #[source]
        source: serde_json::Error,
    },
}

impl NcuSourceError {
    pub fn unknown_schema_target(target: &str, expected: String) -> Self {
        Self::UnknownSchemaTarget {
            target: target.to_string(),
            expected,
        }
    }

    pub fn trace_not_found(trace: &std::path::Path) -> Self {
        Self::TraceNotFound {
            trace: trace.display().to_string(),
        }
    }

    pub fn serialize_schema(target: &str, source: serde_json::Error) -> Self {
        Self::SerializeSchema {
            target: target.to_string(),
            source,
        }
    }

    pub fn invalid_launch_dims(flag: &'static str, value: &str, source: NcuSourceError) -> Self {
        Self::InvalidLaunchDims {
            flag,
            value: value.to_string(),
            source: Box::new(source),
        }
    }

    pub fn launch_dims_shape(value: &str) -> Self {
        Self::LaunchDimsShape {
            value: value.to_string(),
        }
    }

    pub fn launch_dims_axis(value: &str, axis: &str, source: std::num::ParseIntError) -> Self {
        Self::LaunchDimsAxis {
            value: value.to_string(),
            axis: axis.to_string(),
            source,
        }
    }

    pub fn unknown_source_metrics_axis(axis: &str) -> Self {
        Self::UnknownSourceMetricsAxis {
            axis: axis.to_string(),
        }
    }

    pub fn unknown_warp_stalls_axis(axis: &str) -> Self {
        Self::UnknownWarpStallsAxis {
            axis: axis.to_string(),
        }
    }

    pub fn limit_too_small(limit: usize) -> Self {
        Self::LimitTooSmall { limit }
    }

    pub fn counter_glob_empty() -> Self {
        Self::CounterGlobEmpty
    }

    pub fn launch_row_id_missing_colon(row_id: &str) -> Self {
        Self::LaunchRowIdMissingColon {
            row_id: row_id.to_string(),
        }
    }

    pub fn launch_row_id_kind_unsupported(kind: &str) -> Self {
        Self::LaunchRowIdKindUnsupported {
            kind: kind.to_string(),
        }
    }

    pub fn launch_row_id_index_invalid(
        row_id: &str,
        index: &str,
        source: std::num::ParseIntError,
    ) -> Self {
        Self::LaunchRowIdIndexInvalid {
            row_id: row_id.to_string(),
            index: index.to_string(),
            source,
        }
    }

    pub fn launch_row_id_out_of_range(
        row_id: &str,
        launch_idx: usize,
        launch_count: usize,
    ) -> Self {
        Self::LaunchRowIdOutOfRange {
            row_id: row_id.to_string(),
            launch_idx,
            launch_count,
        }
    }

    pub fn launch_vanished_after_bounds_check(launch_idx: usize) -> Self {
        Self::LaunchVanishedAfterBoundsCheck { launch_idx }
    }

    pub fn cubin_length_exceeds_available_bytes(computed: usize, available: usize) -> Self {
        Self::CubinLengthExceedsAvailableBytes {
            computed,
            available,
        }
    }

    fn input_io(
        area: &'static str,
        operation: NcuIoOperation,
        target: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::InputIo {
            area,
            operation,
            target: target.to_string(),
            source,
        }
    }

    fn input_json(
        area: &'static str,
        operation: NcuJsonOperation,
        target: impl std::fmt::Display,
        source: serde_json::Error,
    ) -> Self {
        Self::InputJson {
            area,
            operation,
            target: target.to_string(),
            source,
        }
    }

    fn input_utf8(area: &'static str, source: std::string::FromUtf8Error) -> Self {
        Self::InputUtf8 { area, source }
    }

    pub fn input_io_parts(&self) -> Option<(&'static str, NcuIoOperation, &str)> {
        match self {
            Self::InputIo {
                area,
                operation,
                target,
                ..
            } => Some((area, *operation, target)),
            _ => None,
        }
    }

    pub fn input_json_parts(&self) -> Option<(&'static str, NcuJsonOperation, &str)> {
        match self {
            Self::InputJson {
                area,
                operation,
                target,
                ..
            } => Some((area, *operation, target)),
            _ => None,
        }
    }

    pub fn cubin_report_read(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::input_io("cubin report", NcuIoOperation::Read, path, source)
    }

    pub fn cubin_report_zstd_decompress(
        path: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::input_io("cubin report", NcuIoOperation::DecompressZstd, path, source)
    }

    pub fn cubin_committed_dir_entry_read(
        path: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::input_io(
            "committed cubin directory entry",
            NcuIoOperation::Read,
            path,
            source,
        )
    }

    pub fn cubin_committed_read(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::input_io("committed cubin", NcuIoOperation::Read, path, source)
    }

    pub fn cubin_elf_parse(source: object::Error) -> Self {
        Self::CubinElfParse { source }
    }

    pub fn cubin_length_overflow(length: u64, source: std::num::TryFromIntError) -> Self {
        Self::CubinLengthOverflow { length, source }
    }

    pub fn native_sidecar_schema_mismatch(
        path: &std::path::Path,
        actual: impl Into<String>,
        expected: &'static str,
    ) -> Self {
        Self::NativeSidecarSchemaMismatch {
            path: path.display().to_string(),
            actual: actual.into(),
            expected,
        }
    }

    pub fn native_sidecar_unsupported_compressed_reader(
        path: &std::path::Path,
        version: &str,
    ) -> Self {
        Self::NativeSidecarUnsupportedCompressedReader {
            path: path.display().to_string(),
            version: version.to_string(),
        }
    }

    pub fn native_ingest_unavailable(report: &std::path::Path, source: NcuSourceError) -> Self {
        Self::NativeIngestUnavailable {
            report: report.display().to_string(),
            source: Box::new(source),
        }
    }

    pub fn native_helper_schema_mismatch(
        actual: impl Into<String>,
        expected: &'static str,
    ) -> Self {
        Self::NativeHelperSchemaMismatch {
            actual: actual.into(),
            expected,
        }
    }

    pub fn native_helper_failed(program: &str, status: String, stderr: String) -> Self {
        Self::NativeHelperFailed {
            program: program.to_string(),
            status,
            stderr,
        }
    }

    pub fn native_artifact_dir_create(
        path: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::input_io(
            "native artifact directory",
            NcuIoOperation::CreateDir,
            path,
            source,
        )
    }

    pub fn native_export_lockfile_open(
        path: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::input_io("native export lockfile", NcuIoOperation::Open, path, source)
    }

    pub fn native_export_lock_acquire(
        path: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::input_io("native export lock", NcuIoOperation::Lock, path, source)
    }

    pub fn native_cache_marker_read(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::input_io("native cache marker", NcuIoOperation::Read, path, source)
    }

    pub fn native_sidecar_read(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::input_io("native sidecar", NcuIoOperation::Read, path, source)
    }

    pub fn native_sidecar_gunzip(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::input_io(
            "native sidecar",
            NcuIoOperation::DecompressGzip,
            path,
            source,
        )
    }

    pub fn native_sidecar_deserialize(
        path: impl std::fmt::Display,
        source: serde_json::Error,
    ) -> Self {
        Self::input_json("native sidecar", NcuJsonOperation::Decode, path, source)
    }

    pub fn native_helper_output_deserialize(source: serde_json::Error) -> Self {
        Self::input_json(
            "ncu_report export helper output",
            NcuJsonOperation::Decode,
            "stdout",
            source,
        )
    }

    pub fn native_sidecar_gzip_write(source: std::io::Error) -> Self {
        Self::input_io(
            "native sidecar payload",
            NcuIoOperation::CompressGzip,
            "buffer",
            source,
        )
    }

    pub fn native_sidecar_gzip_finish(source: std::io::Error) -> Self {
        Self::input_io(
            "native sidecar gzip stream",
            NcuIoOperation::CompressGzip,
            "buffer",
            source,
        )
    }

    pub fn native_atomic_write(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::input_io("native temp file", NcuIoOperation::Write, path, source)
    }

    pub fn native_atomic_rename(
        from: impl std::fmt::Display,
        to: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::input_io(
            "native temp file",
            NcuIoOperation::Publish,
            format!("{from} -> {to}"),
            source,
        )
    }

    pub fn native_report_read(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::input_io("ncu report", NcuIoOperation::Read, path, source)
    }

    pub fn native_helper_materialize(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::input_io(
            "ncu_report export helper",
            NcuIoOperation::Write,
            path,
            source,
        )
    }

    pub fn native_helper_stdout_utf8(source: std::string::FromUtf8Error) -> Self {
        Self::input_utf8("ncu_report export helper stdout", source)
    }

    pub fn native_helper_spawn(program: impl Into<String>, source: std::io::Error) -> Self {
        Self::NativeHelperSpawn {
            program: program.into(),
            source,
        }
    }

    pub fn disasm_tool_spawn(
        bin: &'static str,
        cubin_path: &std::path::Path,
        args: &[&str],
        source: std::io::Error,
    ) -> Self {
        Self::DisasmToolSpawn {
            bin,
            args: format!("{args:?}"),
            cubin: cubin_path.display().to_string(),
            source,
        }
    }

    pub fn disasm_tool_failed(
        bin: &'static str,
        cubin_path: &std::path::Path,
        args: &[&str],
        status: String,
        stderr: String,
    ) -> Self {
        Self::DisasmToolFailed {
            bin,
            args: format!("{args:?}"),
            cubin: cubin_path.display().to_string(),
            status,
            stderr,
        }
    }

    pub fn disasm_tool_output_utf8(
        bin: &'static str,
        stream: &'static str,
        source: std::string::FromUtf8Error,
    ) -> Self {
        Self::DisasmToolOutputUtf8 {
            bin,
            stream,
            source,
        }
    }

    pub fn disasm_sidecar_dir_create(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::input_io(
            "disasm sidecar directory",
            NcuIoOperation::CreateDir,
            path,
            source,
        )
    }

    pub fn disasm_cubin_write(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::input_io("cubin sidecar", NcuIoOperation::Write, path, source)
    }

    pub fn disasm_cubin_read(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::input_io("cubin sidecar", NcuIoOperation::Read, path, source)
    }

    pub fn disasm_cubin_publish(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::input_io("cubin sidecar", NcuIoOperation::Publish, path, source)
    }

    pub fn disasm_cache_read(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::input_io(
            "correlated disasm cache",
            NcuIoOperation::Read,
            path,
            source,
        )
    }

    pub fn disasm_cache_decode(path: impl std::fmt::Display, source: serde_json::Error) -> Self {
        Self::input_json(
            "correlated disasm cache",
            NcuJsonOperation::Decode,
            path,
            source,
        )
    }

    pub fn disasm_cache_encode(path: impl std::fmt::Display, source: serde_json::Error) -> Self {
        Self::input_json(
            "correlated disasm cache",
            NcuJsonOperation::Encode,
            path,
            source,
        )
    }

    pub fn disasm_cache_write(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::input_io(
            "correlated disasm cache",
            NcuIoOperation::Write,
            path,
            source,
        )
    }

    pub fn disasm_cache_publish(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::input_io(
            "correlated disasm cache",
            NcuIoOperation::Publish,
            path,
            source,
        )
    }

    pub fn nvdisasm_json_decode(source: serde_json::Error) -> Self {
        Self::NvdisasmJsonDecode { source }
    }

    pub fn nvdisasm_kernel_function_name_missing(index: usize) -> Self {
        Self::NvdisasmKernelFunctionNameMissing { index }
    }

    pub fn nvdisasm_kernel_start_missing(index: usize) -> Self {
        Self::NvdisasmKernelStartMissing { index }
    }

    pub fn native_python_missing(tried: String) -> Self {
        Self::NativePythonMissing { tried }
    }

    pub fn native_ncu_report_override_invalid(dir: &std::path::Path) -> Self {
        Self::NativeNcuReportOverrideInvalid {
            dir: dir.display().to_string(),
        }
    }

    pub fn native_ncu_report_module_missing(message: String) -> Self {
        Self::NativeNcuReportModuleMissing { message }
    }

    pub fn serialize_envelope(source: serde_json::Error) -> Self {
        Self::SerializeEnvelope { source }
    }
}

impl VeloqDiagnostic for NcuSourceError {
    fn code(&self) -> ErrorCode {
        match self {
            Self::UnsupportedSchemaFormat { .. } => {
                ErrorCode::new("ncu.command.unsupported-schema-format")
            }
            Self::MissingTracePath => ErrorCode::new("ncu.command.missing-trace-path"),
            Self::TraceNotFound { .. } => ErrorCode::new("ncu.input.missing"),
            Self::UnknownSchemaTarget { .. } => ErrorCode::new("ncu.command.unknown-schema-target"),
            Self::SerializeSchema { .. } => ErrorCode::new("ncu.command.serialize-schema"),
            Self::InvalidLaunchDims { .. }
            | Self::LaunchDimsShape { .. }
            | Self::LaunchDimsAxis { .. } => ErrorCode::new("ncu.command.invalid-launch-dims"),
            Self::UnknownSourceMetricsAxis { .. } => {
                ErrorCode::new("ncu.command.unknown-source-metrics-axis")
            }
            Self::UnknownWarpStallsAxis { .. } => {
                ErrorCode::new("ncu.command.unknown-warp-stalls-axis")
            }
            Self::LimitTooSmall { .. } => ErrorCode::new("ncu.command.limit-too-small"),
            Self::CounterGlobEmpty => ErrorCode::new("ncu.command.empty-counter-glob"),
            Self::InspectRowIdRequired => ErrorCode::new("ncu.command.inspect-row-id-required"),
            Self::LaunchRowIdMissingColon { .. }
            | Self::LaunchRowIdKindUnsupported { .. }
            | Self::LaunchRowIdIndexInvalid { .. } => {
                ErrorCode::new("ncu.command.invalid-launch-row-id")
            }
            Self::LaunchRowIdOutOfRange { .. } => {
                ErrorCode::new("ncu.command.launch-row-id-out-of-range")
            }
            Self::LaunchVanishedAfterBoundsCheck { .. } => {
                ErrorCode::new("ncu.internal.launch-vanished-after-bounds-check")
            }
            Self::CubinLengthExceedsAvailableBytes { .. } => {
                ErrorCode::new("ncu.input.cubin-length-exceeds-available-bytes")
            }
            Self::InputIo { operation, .. } => operation.code(),
            Self::InputJson { operation, .. } => operation.code(),
            Self::InputUtf8 { .. } => ErrorCode::new("ncu.input.utf8"),
            Self::CubinElfParse { .. } => ErrorCode::new("ncu.input.cubin-elf-parse"),
            Self::CubinLengthOverflow { .. } => ErrorCode::new("ncu.input.cubin-length-overflow"),
            Self::SourceMetricsLineWithoutFile => {
                ErrorCode::new("ncu.command.source-metrics-line-without-file")
            }
            Self::NativeSidecarSchemaMismatch { .. } => {
                ErrorCode::new("ncu.input.native-sidecar-schema-mismatch")
            }
            Self::NativeSidecarUnsupportedCompressedReader { .. } => {
                ErrorCode::new("ncu.input.native-sidecar-unsupported-reader")
            }
            Self::NativeIngestUnavailable { .. } => ErrorCode::new("ncu.input.ingest-unavailable"),
            Self::NativeHelperSchemaMismatch { .. } => {
                ErrorCode::new("ncu.input.native-helper-schema-mismatch")
            }
            Self::NativeHelperFailed { .. } => ErrorCode::new("ncu.input.native-helper-failed"),
            Self::NativeHelperSpawn { .. } => ErrorCode::new("ncu.input.native-helper-spawn"),
            Self::DisasmToolSpawn { .. } => ErrorCode::new("ncu.input.disasm-tool-spawn"),
            Self::DisasmToolFailed { .. } => ErrorCode::new("ncu.input.disasm-tool-failed"),
            Self::DisasmToolOutputUtf8 { .. } => {
                ErrorCode::new("ncu.input.disasm-tool-output-utf8")
            }
            Self::NvdisasmJsonDecode { .. } => ErrorCode::new("ncu.input.nvdisasm-json-decode"),
            Self::NvdisasmTopLevelNotArray => {
                ErrorCode::new("ncu.input.nvdisasm-top-level-not-array")
            }
            Self::NvdisasmKernelArrayMissing => {
                ErrorCode::new("ncu.input.nvdisasm-kernel-array-missing")
            }
            Self::NvdisasmKernelFunctionNameMissing { .. } => {
                ErrorCode::new("ncu.input.nvdisasm-kernel-function-name-missing")
            }
            Self::NvdisasmKernelStartMissing { .. } => {
                ErrorCode::new("ncu.input.nvdisasm-kernel-start-missing")
            }
            Self::NativePythonMissing { .. } => ErrorCode::new("ncu.input.python-missing"),
            Self::NativeNcuReportOverrideInvalid { .. } => {
                ErrorCode::new("ncu.input.ncu-report-override-invalid")
            }
            Self::NativeNcuReportModuleMissing { .. } => {
                ErrorCode::new("ncu.input.ncu-report-module-missing")
            }
            Self::Tabular(err) => err.code(),
            Self::SerializeEnvelope { .. } => ErrorCode::new("ncu.output.serialize-envelope"),
        }
    }
}
