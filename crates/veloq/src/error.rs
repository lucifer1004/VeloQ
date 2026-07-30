use crate::daemon::DaemonError;
use crate::meta::MetaError;
use std::borrow::Cow;
use thiserror::Error;
use veloq_core::{ErrorCode, SourceRunError, VeloqDiagnostic};

pub type CliResult<T> = Result<T, CliError>;

#[derive(Debug, Error)]
pub enum CliError {
    #[error("no subcommand selected (clap should have rejected this)")]
    NoSubcommand,

    #[error("default source `{kind}` not registered")]
    DefaultSourceNotRegistered { kind: &'static str },

    #[error("{source}")]
    SourceRun { source: SourceRunError },

    #[error("writing source command output")]
    SourceOutput {
        #[source]
        source: std::io::Error,
    },

    #[error(transparent)]
    Meta(#[from] MetaError),

    #[error(transparent)]
    Daemon(#[from] DaemonError),
}

impl CliError {
    pub fn source_run(source: SourceRunError) -> Self {
        Self::SourceRun { source }
    }

    pub fn source_output(source: std::io::Error) -> Self {
        Self::SourceOutput { source }
    }

    pub fn daemon(source: DaemonError) -> Self {
        Self::Daemon(source)
    }
}

impl VeloqDiagnostic for CliError {
    fn code(&self) -> ErrorCode {
        match self {
            Self::NoSubcommand => ErrorCode::new("cli.no-subcommand"),
            Self::DefaultSourceNotRegistered { .. } => {
                ErrorCode::new("cli.default-source-not-registered")
            }
            Self::SourceRun { .. } => ErrorCode::new("cli.source-run"),
            Self::SourceOutput { .. } => ErrorCode::new("cli.source-output"),
            Self::Meta(err) => err.code(),
            Self::Daemon(err) => err.code(),
        }
    }

    fn hint(&self) -> Option<Cow<'_, str>> {
        match self {
            Self::Meta(err) => err.hint(),
            Self::Daemon(err) => err.hint(),
            _ => None,
        }
    }
}
