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

    #[error(transparent)]
    Meta(#[from] MetaError),
}

impl CliError {
    pub fn source_run(source: SourceRunError) -> Self {
        Self::SourceRun { source }
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
            Self::Meta(err) => err.code(),
        }
    }

    fn hint(&self) -> Option<Cow<'_, str>> {
        match self {
            Self::Meta(err) => err.hint(),
            _ => None,
        }
    }
}
