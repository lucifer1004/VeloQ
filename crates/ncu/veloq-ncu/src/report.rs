use std::path::Path;

/// On-disk encoding of an Nsight Compute report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NcuReportFormat {
    Plain,
    Zstd,
}

impl NcuReportFormat {
    /// Detect a supported report from its path without opening it.
    pub(crate) fn detect(path: &Path) -> Option<Self> {
        match path.extension().and_then(|extension| extension.to_str()) {
            Some("ncu-rep") => Some(Self::Plain),
            Some("ncu-repz") => Some(Self::Zstd),
            _ => None,
        }
    }

    /// Whether an `ncu_report` release can decode this storage encoding.
    pub(crate) fn reader_supports(self, version: &str) -> bool {
        match self {
            Self::Plain => true,
            Self::Zstd => {
                release_version(version).is_some_and(|version| version >= MIN_ZSTD_READER_VERSION)
            }
        }
    }
}

const MIN_ZSTD_READER_VERSION: (u32, u32) = (2025, 4);

fn release_version(value: &str) -> Option<(u32, u32)> {
    let mut parts = value.split('.');
    Some((parts.next()?.parse().ok()?, parts.next()?.parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_supported_report_encodings() {
        assert_eq!(
            NcuReportFormat::detect(Path::new("report.ncu-rep")),
            Some(NcuReportFormat::Plain)
        );
        assert_eq!(
            NcuReportFormat::detect(Path::new("report.ncu-repz")),
            Some(NcuReportFormat::Zstd)
        );
        assert_eq!(
            NcuReportFormat::detect(Path::new("report.ncu-rep.veloq")),
            None
        );
        assert_eq!(NcuReportFormat::detect(Path::new("trace.nsys-rep")), None);
    }

    #[test]
    fn compressed_reports_require_a_capable_reader() {
        assert!(NcuReportFormat::Plain.reader_supports("2025.3.1"));
        assert!(!NcuReportFormat::Zstd.reader_supports("2025.3.1"));
        assert!(NcuReportFormat::Zstd.reader_supports("2025.4.0"));
        assert!(NcuReportFormat::Zstd.reader_supports("2026.2.1"));
        assert!(!NcuReportFormat::Zstd.reader_supports("unknown"));
    }
}
