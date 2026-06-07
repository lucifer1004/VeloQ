use crate::{DataError, DataResult};
use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileFingerprint {
    pub path: String,
    pub mtime_secs: i64,
    pub mtime_nanos: u32,
    pub size: u64,
}

pub fn sibling_tmp(path: &Path) -> PathBuf {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    PathBuf::from(tmp)
}

pub fn atomic_publish<E>(
    path: &Path,
    write_to: impl FnOnce(&Path) -> Result<(), E>,
) -> Result<(), E>
where
    E: From<DataError>,
{
    let tmp = sibling_tmp(path);
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|source| E::from(DataError::create_dir(parent, source)))?;
    }
    write_to(&tmp)?;
    fs::rename(&tmp, path).map_err(|source| E::from(DataError::publish(path, source)))?;
    Ok(())
}

pub fn fingerprint_path(path: &Path) -> DataResult<FileFingerprint> {
    let metadata = fs::metadata(path).map_err(|source| DataError::stat_path(path, source))?;
    let (mtime_secs, mtime_nanos) = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| {
            (
                i64::try_from(duration.as_secs()).unwrap_or(i64::MAX),
                duration.subsec_nanos(),
            )
        })
        .unwrap_or((0, 0));
    Ok(FileFingerprint {
        path: path.display().to_string(),
        mtime_secs,
        mtime_nanos,
        size: metadata.len(),
    })
}

pub fn fingerprint_paths<P>(paths: &[P]) -> DataResult<Vec<FileFingerprint>>
where
    P: AsRef<Path>,
{
    paths
        .iter()
        .map(|path| fingerprint_path(path.as_ref()))
        .collect()
}

pub fn read_text_maybe_gz(path: &Path) -> DataResult<String> {
    read_text_maybe_gz_as(path, "text")
}

pub fn read_text_maybe_gz_as(path: &Path, content: impl Into<String>) -> DataResult<String> {
    let content = content.into();
    let bytes = fs::read(path).map_err(|source| DataError::read_file(path, source))?;
    if path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".gz"))
    {
        let mut decoder = GzDecoder::new(bytes.as_slice());
        let mut text = String::new();
        decoder
            .read_to_string(&mut text)
            .map_err(|source| DataError::decompress_gzip(path, source))?;
        return Ok(text);
    }
    String::from_utf8(bytes).map_err(|source| DataError::decode_utf8(path, content, source))
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::io::Write;

    #[test]
    fn read_text_maybe_gz_reads_plain_utf8() -> DataResult<()> {
        let dir = tempfile::tempdir()
            .map_err(|source| DataError::create_dir(Path::new("temporary directory"), source))?;
        let path = dir.path().join("trace.json");
        fs::write(&path, br#"{"traceEvents":[]}"#)
            .map_err(|source| DataError::write_file(&path, source))?;

        assert_eq!(read_text_maybe_gz(&path)?, r#"{"traceEvents":[]}"#);
        Ok(())
    }

    #[test]
    fn read_text_maybe_gz_decompresses_gzip_by_suffix() -> DataResult<()> {
        let dir = tempfile::tempdir()
            .map_err(|source| DataError::create_dir(Path::new("temporary directory"), source))?;
        let path = dir.path().join("trace.json.gz");
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder
            .write_all(br#"{"traceEvents":[]}"#)
            .map_err(|source| DataError::write_file(&path, source))?;
        let gzip = encoder
            .finish()
            .map_err(|source| DataError::write_file(&path, source))?;
        fs::write(&path, gzip).map_err(|source| DataError::write_file(&path, source))?;

        assert_eq!(read_text_maybe_gz(&path)?, r#"{"traceEvents":[]}"#);
        Ok(())
    }

    #[test]
    fn fingerprint_path_records_size_and_path() -> DataResult<()> {
        let dir = tempfile::tempdir()
            .map_err(|source| DataError::create_dir(Path::new("temporary directory"), source))?;
        let path = dir.path().join("trace.pt.trace.json");
        fs::write(&path, b"{}").map_err(|source| DataError::write_file(&path, source))?;

        let fingerprint = fingerprint_path(&path)?;
        assert_eq!(fingerprint.path, path.display().to_string());
        assert_eq!(fingerprint.size, 2);
        Ok(())
    }
}
