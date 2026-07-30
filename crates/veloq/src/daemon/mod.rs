pub mod config;
pub mod lifecycle;
pub mod protocol;
pub mod routing;
pub mod runtime;
pub mod session;
pub mod state;

use std::borrow::Cow;
use std::sync::Arc;

use clap::{Arg, ArgMatches, Command};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use veloq_core::{
    EnvelopeError, ErrorCode, OutputFormat, ProfileSource, SourceRef, VeloqDiagnostic,
};

pub const DAEMON_SOURCE: SourceRef = SourceRef {
    kind: "veloq",
    version: "v1",
};
pub const DEFAULT_SOURCE: &str = "nsys";

pub type DaemonResult<T> = Result<T, DaemonError>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Error)]
#[serde(rename_all = "snake_case")]
pub enum DaemonError {
    #[error("{message}")]
    Absent { message: String },
    #[error("{message}")]
    Unresponsive { message: String },
    #[error("{message}")]
    Incompatible { message: String },
    #[error("{message}")]
    Unsupported { message: String },
    #[error("{message}")]
    ResourcePressure { message: String },
    #[error("{message}")]
    Cancelled { message: String },
    #[error("{message}")]
    ExecutionIndeterminate { message: String },
    #[error("{message}")]
    ConfigConflict { message: String },
    #[error("{message}")]
    InvalidConfig { message: String },
    #[error("{message}")]
    LifecycleFailed { message: String },
    #[error("daemon singleton ownership already exists")]
    OwnerExists,
}

impl DaemonError {
    pub fn absent(message: impl Into<String>) -> Self {
        Self::Absent {
            message: message.into(),
        }
    }

    pub fn unresponsive(message: impl Into<String>) -> Self {
        Self::Unresponsive {
            message: message.into(),
        }
    }

    pub fn incompatible(message: impl Into<String>) -> Self {
        Self::Incompatible {
            message: message.into(),
        }
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported {
            message: message.into(),
        }
    }

    pub fn resource_pressure(message: impl Into<String>) -> Self {
        Self::ResourcePressure {
            message: message.into(),
        }
    }

    pub fn cancelled(message: impl Into<String>) -> Self {
        Self::Cancelled {
            message: message.into(),
        }
    }

    pub fn execution_indeterminate(message: impl Into<String>) -> Self {
        Self::ExecutionIndeterminate {
            message: message.into(),
        }
    }

    pub fn config_conflict(message: impl Into<String>) -> Self {
        Self::ConfigConflict {
            message: message.into(),
        }
    }

    pub fn invalid_config(message: impl Into<String>) -> Self {
        Self::InvalidConfig {
            message: message.into(),
        }
    }

    pub fn lifecycle(message: impl Into<String>) -> Self {
        Self::LifecycleFailed {
            message: message.into(),
        }
    }

    pub fn owner_exists() -> Self {
        Self::OwnerExists
    }

    pub fn is_owner_exists(&self) -> bool {
        matches!(self, Self::OwnerExists)
    }
}

impl VeloqDiagnostic for DaemonError {
    fn code(&self) -> ErrorCode {
        ErrorCode::new(match self {
            Self::Absent { .. } => "daemon.absent",
            Self::Unresponsive { .. } => "daemon.unresponsive",
            Self::Incompatible { .. } => "daemon.incompatible",
            Self::Unsupported { .. } => "daemon.unsupported",
            Self::ResourcePressure { .. } => "daemon.resource-pressure",
            Self::Cancelled { .. } => "daemon.cancelled",
            Self::ExecutionIndeterminate { .. } => "daemon.execution-indeterminate",
            Self::ConfigConflict { .. } => "daemon.config-conflict",
            Self::InvalidConfig { .. } => "daemon.invalid-config",
            Self::LifecycleFailed { .. } | Self::OwnerExists => "daemon.lifecycle-failed",
        })
    }

    fn hint(&self) -> Option<Cow<'_, str>> {
        match self {
            Self::Absent { .. }
            | Self::Unresponsive { .. }
            | Self::Incompatible { .. }
            | Self::Unsupported { .. }
            | Self::ExecutionIndeterminate { .. } => Some(Cow::Borrowed(
                "use `--daemon off` on a new invocation to force independent one-shot execution",
            )),
            _ => None,
        }
    }
}

pub fn cli() -> Command {
    Command::new("daemon")
        .about("Manage the optional current-user local query daemon")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            config::start_args(
                Command::new("start")
                    .about("Start the current-user daemon or report the compatible live owner"),
            )
            .arg(config::timeout_arg()),
        )
        .subcommand(
            Command::new("stop")
                .about("Stop the current-user daemon")
                .arg(config::timeout_arg()),
        )
        .subcommand(
            Command::new("status")
                .about("Inspect current-user daemon state without mutation")
                .arg(config::timeout_arg()),
        )
        .subcommand(
            Command::new("__serve").hide(true).arg(
                Arg::new("owner-token")
                    .long("owner-token")
                    .required(true)
                    .hide(true),
            ),
        )
}

pub fn graft_source_commands(mut root: Command, sources: &[Arc<dyn ProfileSource>]) -> Command {
    for source in sources {
        let mut sub = source.cli();
        config::inject_query_args(&mut sub);
        if source.kind() == DEFAULT_SOURCE {
            for inner in sub.get_subcommands() {
                root = root.subcommand(inner.clone());
            }
        }
        root = root.subcommand(sub);
    }
    root
}

pub fn query_cli(sources: &[Arc<dyn ProfileSource>]) -> Command {
    let root = Command::new("veloq").arg(
        Arg::new("format")
            .long("format")
            .global(true)
            .default_value("json"),
    );
    graft_source_commands(root, sources)
}

pub fn run(
    matches: &ArgMatches,
    fmt: OutputFormat,
    sources: &[Arc<dyn ProfileSource>],
) -> DaemonResult<i32> {
    let (operation, operation_matches) = matches
        .subcommand()
        .ok_or_else(|| DaemonError::lifecycle("daemon lifecycle operation is required"))?;
    if operation == "__serve" {
        let token = operation_matches
            .get_one::<String>("owner-token")
            .ok_or_else(|| DaemonError::lifecycle("daemon owner token is required"))?;
        return runtime::serve(token, sources);
    }
    lifecycle::run(operation, operation_matches, fmt)
}

pub fn emit_error(error: &DaemonError, fmt: OutputFormat) {
    let envelope = EnvelopeError::from_diagnostic(
        Some(DAEMON_SOURCE),
        Some("daemon".to_string()),
        None,
        None,
        error,
    );
    if !matches!(fmt, OutputFormat::Json) {
        eprintln!("veloq: {error}");
    }
    if let Ok(rendered) = envelope.to_json_pretty() {
        println!("{rendered}");
    }
}
