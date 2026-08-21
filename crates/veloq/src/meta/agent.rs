//! `veloq agent` — install, update, uninstall, and diagnose VeloQ Agent Skills
//! integrations for supported agent runtimes.
//!
//! The native CLI orchestration lives in `agent-plugin-installer` so
//! other projects can reuse it. This module owns only VeloQ's command
//! surface, package validation, and JSON envelope projection.

use agent_plugin_installer::{
    AgentPluginError, AgentPluginOperation, AgentRuntime, AgentSelector as InstallerAgentSelector,
    BatchFailure, BatchOperationError, BatchStatus, DoctorStatus, FailurePolicy, InstallRequest,
    MarketplaceSource, OperationError, PluginRef, SourceUpdateRequest, UninstallRequest,
    UpdateRequest, check_operation as check_runtime_operation, doctor as doctor_runtime,
    install as install_runtime, uninstall as uninstall_runtime, update as update_runtime,
    update_from_source_many,
};
use clap::{Arg, ArgAction, ArgMatches, Command};
use serde::Serialize;
use std::path::{Path, PathBuf};
use veloq_core::OutputFormat;

use super::{MetaError, MetaResult, emit_meta_error, emit_or_error};

const VERB: &str = "agent";
const SELECTOR: &str = "agent";
const ALL: &str = "all";
const FROM_CHECKOUT: &str = "from-checkout";
const VELOQ_PLUGIN: PluginRef<'static> = PluginRef {
    selector: "veloq@veloq",
    name: "veloq",
};
const VELOQ_MARKETPLACE_SOURCE: &str = "lucifer1004/veloq";
const AGENT_VALUES: [&str; 3] = ["codex", "claude", "all"];

#[derive(Debug)]
struct RequiredPath {
    path: PathBuf,
    kind: RequiredPathKind,
}

impl RequiredPath {
    fn file(path: PathBuf) -> Self {
        Self {
            path,
            kind: RequiredPathKind::File,
        }
    }

    fn dir(path: PathBuf) -> Self {
        Self {
            path,
            kind: RequiredPathKind::Dir,
        }
    }
}

#[derive(Debug)]
enum RequiredPathKind {
    File,
    Dir,
}

#[derive(Serialize)]
struct AgentPayload {
    count: usize,
    total_matched: usize,
    rows: Vec<AgentRow>,
    auxiliary: AgentAuxiliary,
}

#[derive(Serialize)]
struct AgentRow {
    key: String,
    agent: &'static str,
    operation: &'static str,
    status: AgentStatus,
    cli: &'static str,
    commands: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    checkout: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

#[derive(Serialize)]
struct AgentAuxiliary {
    supported_agents: Vec<&'static str>,
}

#[derive(Serialize)]
#[serde(rename_all = "lowercase")]
enum AgentStatus {
    Ready,
    Installed,
    Updated,
    Uninstalled,
    Missing,
    Failed,
}

pub fn cli() -> Command {
    Command::new(VERB)
        .about("Install, update, uninstall, and diagnose VeloQ Agent Skills integrations")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            Command::new("doctor")
                .about("Report VeloQ Agent Skills integration readiness")
                .arg(agent_arg(false).help("Agent to inspect; defaults to all")),
        )
        .subcommand(
            Command::new("install")
                .about(
                    "Install VeloQ Agent Skills into a supported agent runtime \
                     (default source: Git marketplace lucifer1004/veloq)",
                )
                .arg(agent_arg(true))
                .arg(
                    Arg::new(FROM_CHECKOUT)
                        .long(FROM_CHECKOUT)
                        .value_name("PATH")
                        .help(
                            "Install from a local VeloQ checkout with plugin package metadata; \
                             omit to install from the Git marketplace lucifer1004/veloq",
                        ),
                ),
        )
        .subcommand(
            Command::new("update")
                .about("Update VeloQ Agent Skills through the selected agent runtime")
                .arg(agent_arg(true))
                .arg(
                    Arg::new(FROM_CHECKOUT)
                        .long(FROM_CHECKOUT)
                        .value_name("PATH")
                        .help(
                            "Update from a local VeloQ checkout; omit to update through the \
                             registered `veloq` Git marketplace (default source: \
                             lucifer1004/veloq)",
                        ),
                ),
        )
        .subcommand(
            Command::new("uninstall")
                .about("Uninstall VeloQ Agent Skills from a supported agent runtime")
                .arg(agent_arg(true)),
        )
}

pub fn run(matches: &ArgMatches, fmt: OutputFormat) -> MetaResult<i32> {
    let outcome = match matches.subcommand() {
        Some(("doctor", sub)) => doctor(sub),
        Some(("install", sub)) => install(sub),
        Some(("update", sub)) => update(sub),
        Some(("uninstall", sub)) => uninstall(sub),
        _ => Err(MetaError::missing_argument("agent subcommand")),
    };
    match outcome {
        Ok(payload) => Ok(emit_or_error(fmt, VERB, None, None, payload)),
        Err(err) => {
            emit_meta_error(fmt, VERB, None, &err);
            Ok(1)
        }
    }
}

fn agent_arg(required: bool) -> Arg {
    let mut arg = Arg::new(SELECTOR)
        .value_name("AGENT")
        .value_parser(AGENT_VALUES)
        .action(ArgAction::Set);
    if required {
        arg = arg
            .required(true)
            .help("Agent to operate on: codex, claude, or all");
    }
    arg
}

fn doctor(matches: &ArgMatches) -> MetaResult<AgentPayload> {
    let agents = selected_agents(matches.get_one::<String>(SELECTOR).map(String::as_str));
    let rows = agents
        .into_iter()
        .map(|agent| {
            let outcome = doctor_runtime(agent);
            AgentRow {
                key: key(agent),
                agent: agent.id(),
                operation: "doctor",
                status: match outcome.status {
                    DoctorStatus::Ready => AgentStatus::Ready,
                    DoctorStatus::Missing => AgentStatus::Missing,
                    DoctorStatus::Failed => AgentStatus::Failed,
                },
                cli: agent.cli(),
                commands: outcome.commands,
                checkout: None,
                message: outcome.message,
            }
        })
        .collect();
    Ok(payload(rows))
}

fn install(matches: &ArgMatches) -> MetaResult<AgentPayload> {
    let agents = selected_agents(matches.get_one::<String>(SELECTOR).map(String::as_str));
    if let Some(checkout) = matches.get_one::<String>(FROM_CHECKOUT).map(PathBuf::from) {
        for agent in &agents {
            validate_checkout(*agent, &checkout)?;
        }
        preflight_agents(&agents, AgentPluginOperation::Install)?;
        let mut rows = Vec::with_capacity(agents.len());
        for agent in agents {
            let install_checkout = install_checkout(agent, &checkout)?;
            let outcome = install_runtime(
                agent,
                InstallRequest::local(&install_checkout, VELOQ_PLUGIN),
            )
            .map_err(map_operation_error)?;
            rows.push(success_row(
                outcome.runtime,
                "install",
                AgentStatus::Installed,
                outcome.commands,
                Some(&checkout),
            ));
        }
        return Ok(payload(rows));
    }

    preflight_agents(&agents, AgentPluginOperation::Install)?;
    let mut rows = Vec::with_capacity(agents.len());
    for agent in agents {
        let outcome = install_runtime(
            agent,
            InstallRequest::new(
                MarketplaceSource::new(VELOQ_MARKETPLACE_SOURCE),
                VELOQ_PLUGIN,
            ),
        )
        .map_err(map_operation_error)?;
        let mut row = success_row(
            outcome.runtime,
            "install",
            AgentStatus::Installed,
            outcome.commands,
            None,
        );
        row.message = Some(format!(
            "installed from Git marketplace {VELOQ_MARKETPLACE_SOURCE}"
        ));
        rows.push(row);
    }
    Ok(payload(rows))
}

fn update(matches: &ArgMatches) -> MetaResult<AgentPayload> {
    let selected = matches.get_one::<String>(SELECTOR).map(String::as_str);
    let agents = selected_agents(selected);
    if let Some(checkout) = matches.get_one::<String>(FROM_CHECKOUT).map(PathBuf::from) {
        for agent in &agents {
            validate_checkout(*agent, &checkout)?;
        }
        let reporting_agent = agents.first().copied().unwrap_or(AgentRuntime::Codex);
        let source = checkout
            .canonicalize()
            .map_err(|_| package_missing(reporting_agent, &checkout, &checkout))?;
        let report = update_from_source_many(
            selected_agent_selector(selected),
            |_| SourceUpdateRequest::local(&source, VELOQ_PLUGIN),
            FailurePolicy::StopOnFailure,
        )
        .map_err(map_batch_installer_error)?;
        let rows = report
            .outcomes
            .into_iter()
            .map(|outcome| {
                success_row(
                    outcome.runtime,
                    "update",
                    AgentStatus::Updated,
                    mutation_commands(outcome.commands),
                    Some(&checkout),
                )
            })
            .collect();
        return Ok(payload(rows));
    }

    preflight_agents(&agents, AgentPluginOperation::Update)?;
    let mut rows = Vec::with_capacity(agents.len());
    for agent in agents {
        let outcome = update_runtime(
            agent,
            UpdateRequest::new(VELOQ_PLUGIN).with_marketplace_name("veloq"),
        )
        .map_err(map_operation_error)?;
        rows.push(success_row(
            outcome.runtime,
            "update",
            AgentStatus::Updated,
            outcome.commands,
            None,
        ));
    }
    Ok(payload(rows))
}

fn uninstall(matches: &ArgMatches) -> MetaResult<AgentPayload> {
    let agents = selected_agents(matches.get_one::<String>(SELECTOR).map(String::as_str));
    preflight_agents(&agents, AgentPluginOperation::Uninstall)?;
    let mut rows = Vec::with_capacity(agents.len());
    for agent in agents {
        let outcome = uninstall_runtime(agent, UninstallRequest::new(VELOQ_PLUGIN))
            .map_err(map_operation_error)?;
        rows.push(success_row(
            outcome.runtime,
            "uninstall",
            AgentStatus::Uninstalled,
            outcome.commands,
            None,
        ));
    }
    Ok(payload(rows))
}

fn preflight_agents(agents: &[AgentRuntime], operation: AgentPluginOperation) -> MetaResult<()> {
    for agent in agents {
        let outcome = check_runtime_operation(*agent, operation);
        match outcome.status {
            DoctorStatus::Ready => {}
            DoctorStatus::Missing => {
                return Err(MetaError::AgentCliMissing {
                    agent: outcome.runtime.id(),
                    cli: outcome.runtime.cli(),
                });
            }
            DoctorStatus::Failed => {
                return Err(MetaError::AgentCliFailed {
                    agent: outcome.runtime.id(),
                    phase: "preflight",
                    command: outcome.commands.join(" && "),
                    status: None,
                    stderr: outcome
                        .message
                        .unwrap_or_else(|| "agent preflight failed".to_string()),
                });
            }
        }
    }
    Ok(())
}

fn selected_agents(selector: Option<&str>) -> Vec<AgentRuntime> {
    selected_agent_selector(selector).runtimes().to_vec()
}

fn selected_agent_selector(selector: Option<&str>) -> InstallerAgentSelector {
    match selector.unwrap_or(ALL) {
        "codex" => InstallerAgentSelector::Codex,
        "claude" => InstallerAgentSelector::Claude,
        ALL => InstallerAgentSelector::All,
        _ => InstallerAgentSelector::All,
    }
}

fn payload(rows: Vec<AgentRow>) -> AgentPayload {
    let count = rows.len();
    AgentPayload {
        count,
        total_matched: count,
        rows,
        auxiliary: AgentAuxiliary {
            supported_agents: AgentRuntime::supported()
                .iter()
                .map(|agent| agent.id())
                .collect(),
        },
    }
}

fn validate_checkout(agent: AgentRuntime, checkout: &Path) -> MetaResult<()> {
    if !checkout.is_dir() {
        return Err(MetaError::AgentPackageMissing {
            agent: agent.id(),
            checkout: checkout.display().to_string(),
            missing: "checkout directory".to_string(),
        });
    }
    for required in package_requirements(agent, checkout) {
        match required.kind {
            RequiredPathKind::File if !required.path.is_file() => {
                return Err(package_missing(agent, checkout, &required.path));
            }
            RequiredPathKind::Dir if !required.path.is_dir() => {
                return Err(package_missing(agent, checkout, &required.path));
            }
            _ => {}
        }
    }
    Ok(())
}

fn package_requirements(agent: AgentRuntime, checkout: &Path) -> Vec<RequiredPath> {
    match agent {
        AgentRuntime::Codex => vec![
            RequiredPath::file(checkout.join(".agents/plugins/marketplace.json")),
            RequiredPath::file(checkout.join("plugins/veloq/.codex-plugin/plugin.json")),
            RequiredPath::dir(checkout.join("plugins/veloq/skills")),
            RequiredPath::file(
                checkout.join("plugins/veloq/skills/nsys-profile-analysis/SKILL.md"),
            ),
            RequiredPath::file(checkout.join("plugins/veloq/skills/ncu-profile-analysis/SKILL.md")),
            RequiredPath::file(
                checkout.join("plugins/veloq/skills/pytorch-profile-analysis/SKILL.md"),
            ),
        ],
        AgentRuntime::Claude => vec![
            RequiredPath::file(checkout.join(".claude-plugin/marketplace.json")),
            RequiredPath::file(checkout.join("plugins/veloq/.claude-plugin/plugin.json")),
            RequiredPath::dir(checkout.join("plugins/veloq/skills")),
            RequiredPath::file(
                checkout.join("plugins/veloq/skills/nsys-profile-analysis/SKILL.md"),
            ),
            RequiredPath::file(checkout.join("plugins/veloq/skills/ncu-profile-analysis/SKILL.md")),
            RequiredPath::file(
                checkout.join("plugins/veloq/skills/pytorch-profile-analysis/SKILL.md"),
            ),
        ],
    }
}

fn install_checkout(agent: AgentRuntime, checkout: &Path) -> MetaResult<PathBuf> {
    if !matches!(agent, AgentRuntime::Claude) {
        return Ok(checkout.to_path_buf());
    }
    checkout
        .canonicalize()
        .map_err(|_| package_missing(agent, checkout, checkout))
}

fn package_missing(agent: AgentRuntime, checkout: &Path, missing: &Path) -> MetaError {
    MetaError::AgentPackageMissing {
        agent: agent.id(),
        checkout: checkout.display().to_string(),
        missing: missing.display().to_string(),
    }
}

fn map_installer_error(err: AgentPluginError) -> MetaError {
    match err {
        AgentPluginError::CliMissing { runtime, cli } => MetaError::AgentCliMissing {
            agent: runtime,
            cli,
        },
        AgentPluginError::CliSpawnFailed {
            runtime,
            phase,
            command,
            reason,
        } => MetaError::AgentCliFailed {
            agent: runtime,
            phase,
            command,
            status: None,
            stderr: reason,
        },
        AgentPluginError::CliFailed {
            runtime,
            phase,
            command,
            status,
            stderr,
        } => MetaError::AgentCliFailed {
            agent: runtime,
            phase,
            command,
            status,
            stderr,
        },
        AgentPluginError::UnsupportedOption {
            runtime,
            option,
            reason,
        } => MetaError::AgentUnsupportedOption {
            agent: runtime,
            option,
            reason,
        },
    }
}

fn map_operation_error(err: OperationError) -> MetaError {
    map_installer_error(err.error)
}

fn map_batch_installer_error(err: BatchOperationError) -> MetaError {
    for outcome in err.into_report().outcomes {
        let Some(failure) = outcome.failure else {
            continue;
        };
        return match failure {
            BatchFailure::Validation(error) => map_installer_error(error),
            BatchFailure::Operation(error) => map_installer_error(error.error),
            BatchFailure::Preflight { .. } if outcome.status == BatchStatus::Missing => {
                MetaError::AgentCliMissing {
                    agent: outcome.runtime.id(),
                    cli: outcome.runtime.cli(),
                }
            }
            BatchFailure::Preflight { message } => MetaError::AgentCliFailed {
                agent: outcome.runtime.id(),
                phase: "preflight",
                command: outcome.commands.join(" && "),
                status: None,
                stderr: message,
            },
            other => MetaError::AgentCliFailed {
                agent: outcome.runtime.id(),
                phase: "update",
                command: outcome.commands.join(" && "),
                status: None,
                stderr: other.to_string(),
            },
        };
    }
    MetaError::AgentCliFailed {
        agent: "unknown",
        phase: "update",
        command: String::new(),
        status: None,
        stderr: "agent plugin batch operation failed without a runtime failure".to_string(),
    }
}

fn mutation_commands(commands: Vec<String>) -> Vec<String> {
    commands
        .into_iter()
        .filter(|command| !command.ends_with(" --help"))
        .collect()
}

fn success_row(
    agent: AgentRuntime,
    operation: &'static str,
    status: AgentStatus,
    commands: Vec<String>,
    checkout: Option<&Path>,
) -> AgentRow {
    AgentRow {
        key: key(agent),
        agent: agent.id(),
        operation,
        status,
        cli: agent.cli(),
        commands,
        checkout: checkout.map(|path| path.display().to_string()),
        message: None,
    }
}

fn key(agent: AgentRuntime) -> String {
    format!("agent|{}", agent.id())
}
