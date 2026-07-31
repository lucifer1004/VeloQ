use crate::model::{FileFingerprint, InputFingerprint};
use crate::{PytorchDataError, PytorchDataResult};
use std::path::{Path, PathBuf};
use veloq_data::file::{fingerprint_paths, read_text_maybe_gz_as};

pub fn detect_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".pt.trace.json") || name.ends_with(".pt.trace.json.gz"))
}

pub fn is_trace_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".json") || name.ends_with(".json.gz"))
}

pub(crate) fn discover_trace_files(input: &Path) -> PytorchDataResult<Vec<PathBuf>> {
    if input.is_file() {
        if is_trace_file(input) {
            return Ok(vec![input.to_path_buf()]);
        }
        return Err(PytorchDataError::unsupported_trace_extension(input));
    }
    if !input.is_dir() {
        return Err(PytorchDataError::input_does_not_exist(input));
    }
    Err(PytorchDataError::directory_inputs_unsupported(input))
}

pub(crate) fn fingerprint_for_files(files: &[PathBuf]) -> PytorchDataResult<InputFingerprint> {
    let files = fingerprint_paths(files)?
        .into_iter()
        .map(|fingerprint| FileFingerprint {
            path: fingerprint.path,
            mtime_secs: fingerprint.mtime_secs,
            mtime_nanos: fingerprint.mtime_nanos,
            size: fingerprint.size,
        })
        .collect();
    Ok(InputFingerprint { files })
}

pub(crate) fn read_trace_text(file: &Path) -> PytorchDataResult<String> {
    Ok(read_text_maybe_gz_as(file, "pytorch trace JSON")?)
}
