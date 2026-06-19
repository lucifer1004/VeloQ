//! `veloq agent` — install, update, uninstall, and diagnose VeloQ Agent Skills
//! integrations for supported agent runtimes.
//!
//! The native CLI orchestration lives in `agent-plugin-installer` so
//! other projects can reuse it. This module owns only VeloQ's command
//! surface, package validation, and JSON envelope projection.

use agent_plugin_installer::{
    AgentPluginError, AgentPluginOperation, AgentRuntime, DoctorStatus, InstallRequest, PluginRef,
    UninstallRequest, UpdateRequest, check_operation as check_runtime_operation,
    doctor as doctor_runtime, install as install_runtime, uninstall as uninstall_runtime,
    update as update_runtime,
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
                .about("Install VeloQ Agent Skills into a supported agent runtime")
                .arg(agent_arg(true))
                .arg(
                    Arg::new(FROM_CHECKOUT)
                        .long(FROM_CHECKOUT)
                        .value_name("PATH")
                        .required(true)
                        .help("Install from a local VeloQ checkout with plugin package metadata"),
                ),
        )
        .subcommand(
            Command::new("update")
                .about("Update VeloQ Agent Skills through the selected agent runtime")
                .arg(agent_arg(true)),
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
    let checkout = matches
        .get_one::<String>(FROM_CHECKOUT)
        .map(PathBuf::from)
        .ok_or_else(|| MetaError::missing_argument(FROM_CHECKOUT))?;
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
        .map_err(map_installer_error)?;
        rows.push(success_row(
            outcome.runtime,
            "install",
            AgentStatus::Installed,
            outcome.commands,
            Some(&checkout),
        ));
    }
    Ok(payload(rows))
}

fn update(matches: &ArgMatches) -> MetaResult<AgentPayload> {
    let agents = selected_agents(matches.get_one::<String>(SELECTOR).map(String::as_str));
    preflight_agents(&agents, AgentPluginOperation::Update)?;
    let mut rows = Vec::with_capacity(agents.len());
    for agent in agents {
        let outcome = update_runtime(
            agent,
            UpdateRequest::new(VELOQ_PLUGIN).with_marketplace_name("veloq"),
        )
        .map_err(map_installer_error)?;
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
            .map_err(map_installer_error)?;
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
    match selector.unwrap_or(ALL) {
        "codex" => vec![AgentRuntime::Codex],
        "claude" => vec![AgentRuntime::Claude],
        ALL => AgentRuntime::supported().to_vec(),
        _ => Vec::new(),
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
