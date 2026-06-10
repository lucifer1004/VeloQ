use thiserror::Error;
use veloq_core::{ErrorCode, VeloqDiagnostic};

#[derive(Debug, Error)]
pub enum VisualizationError {
    #[error("visualization time window must be positive (start_ns={start_ns}, end_ns={end_ns})")]
    NonPositiveWindow { start_ns: i64, end_ns: i64 },

    #[error("visualization artifact path must be relative and stay under the artifact root")]
    UnsafeRelativePath,

    #[error("visualization artifact filename must be a plain `.svg` filename")]
    UnsafeSvgFileName,

    #[error("failed to create visualization artifact directory `{path}`")]
    CreateArtifactDir {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to write visualization artifact `{path}`")]
    WriteArtifact {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to publish visualization artifact `{path}`")]
    PublishArtifact {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

impl VeloqDiagnostic for VisualizationError {
    fn code(&self) -> ErrorCode {
        match self {
            Self::NonPositiveWindow { .. } => ErrorCode::new("viz.non-positive-window"),
            Self::UnsafeRelativePath => ErrorCode::new("viz.unsafe-relative-path"),
            Self::UnsafeSvgFileName => ErrorCode::new("viz.unsafe-svg-filename"),
            Self::CreateArtifactDir { .. } => ErrorCode::new("viz.artifact-dir-create"),
            Self::WriteArtifact { .. } => ErrorCode::new("viz.artifact-write"),
            Self::PublishArtifact { .. } => ErrorCode::new("viz.artifact-publish"),
        }
    }
}
