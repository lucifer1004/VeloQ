//! Filesystem layout for veloq-generated artifacts.
//!
//! Every generated product for a report is rooted under
//! `<report>.veloq/`. Source crates choose filenames inside that root,
//! but the root derivation itself stays shared so `veloq clean` can
//! remove products without knowing each backend's internals.

use std::path::{Path, PathBuf};

/// Suffix appended to the source artifact to form the generated cache
/// root.
pub const ARTIFACT_DIR_SUFFIX: &str = ".veloq";

/// Per-report directory that owns veloq-generated files.
///
/// Examples:
/// - `trace.nsys-rep` -> `trace.nsys-rep.veloq/`
/// - `trace_pqtdir` -> `trace_pqtdir.veloq/`
/// - `report.ncu-rep` -> `report.ncu-rep.veloq/`
pub fn artifact_dir_for(source: &Path) -> PathBuf {
    if let Some(file_name) = source.file_name() {
        let mut cache_name = file_name.to_owned();
        cache_name.push(ARTIFACT_DIR_SUFFIX);
        return source
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map_or_else(|| PathBuf::from(&cache_name), |p| p.join(&cache_name));
    }

    let mut p = source.as_os_str().to_owned();
    p.push(ARTIFACT_DIR_SUFFIX);
    PathBuf::from(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_dir_appends_suffix_to_file_or_directory_name() {
        assert_eq!(
            artifact_dir_for(Path::new("/tmp/trace.nsys-rep")),
            PathBuf::from("/tmp/trace.nsys-rep.veloq")
        );
        assert_eq!(
            artifact_dir_for(Path::new("/tmp/trace_pqtdir/")),
            PathBuf::from("/tmp/trace_pqtdir.veloq")
        );
        assert_eq!(
            artifact_dir_for(Path::new("trace_pqtdir/")),
            PathBuf::from("trace_pqtdir.veloq")
        );
    }
}
