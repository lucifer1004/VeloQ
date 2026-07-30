use std::fs;
use std::io::BufReader;
use std::process::{Command as ProcessCommand, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use clap::ArgMatches;
use interprocess::local_socket::traits::Stream as _;
use serde::Serialize;
use veloq_core::{Envelope, OutputFormat};

use super::config::{DaemonConfigRequest, DaemonLimits, lifecycle_timeout};
use super::protocol::{
    CONTROL_VERSION, ClientFrame, ControlOperation, PROTOCOL_VERSION, ServerFrame, read_frame,
    write_frame,
};
use super::runtime;
use super::session::{DaemonSnapshot, DaemonUsage, EvictionCounters, SessionStatus};
use super::state::{
    OwnerPhase, OwnerRecord, RuntimePaths, create_owner, new_owner_token, process_matches,
    process_resident_memory, read_owner, remove_owner, remove_stale_endpoint,
};
use super::{DAEMON_SOURCE, DaemonError, DaemonResult, emit_error};

const LIFECYCLE_POLL_INTERVAL: Duration = Duration::from_millis(10);
const NANOSECONDS_PER_MILLISECOND: u32 = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonState {
    Stopped,
    Starting,
    Ready,
    Stopping,
    Incompatible,
    Unresponsive,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DaemonRow {
    pub key: &'static str,
    pub operation: &'static str,
    pub state: DaemonState,
    pub compatible: Option<bool>,
    pub process_id: Option<u32>,
    pub veloq_version: Option<String>,
    pub protocol_version: Option<String>,
    pub limits: Option<DaemonLimits>,
    pub usage: Option<DaemonUsage>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DaemonAuxiliary {
    pub sessions: Vec<SessionStatus>,
    pub evictions: Option<EvictionCounters>,
}

struct StatusReport {
    owner: OwnerRecord,
    snapshot: DaemonSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DaemonPayload {
    pub count: usize,
    pub total_matched: usize,
    pub rows: Vec<DaemonRow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auxiliary: Option<DaemonAuxiliary>,
}

pub fn run(operation: &str, matches: &ArgMatches, fmt: OutputFormat) -> DaemonResult<i32> {
    let result = match operation {
        "start" => start(matches),
        "stop" => stop(matches),
        "status" => status(matches),
        other => Err(DaemonError::lifecycle(format!(
            "unknown daemon lifecycle operation `{other}`"
        ))),
    };
    match result {
        Ok(payload) => {
            let envelope = Envelope::new(
                DAEMON_SOURCE,
                "daemon".to_string(),
                None,
                None,
                None,
                payload,
            );
            match envelope.to_json_pretty() {
                Ok(rendered) => {
                    println!("{rendered}");
                    Ok(0)
                }
                Err(source) => {
                    let error = DaemonError::lifecycle(format!(
                        "cannot serialize daemon response: {source}"
                    ));
                    emit_error(&error, fmt);
                    Ok(1)
                }
            }
        }
        Err(error) => {
            emit_error(&error, fmt);
            Ok(1)
        }
    }
}

fn start(matches: &ArgMatches) -> DaemonResult<DaemonPayload> {
    let timeout = Duration::from_millis(lifecycle_timeout(matches)?);
    let deadline = Instant::now() + timeout;
    let request = DaemonConfigRequest::from_matches(matches)?;
    let requested_limits = request.effective();
    let paths = RuntimePaths::discover()?;

    loop {
        match read_owner(&paths)? {
            Some(owner) if process_matches(&owner) => {
                if !owner_is_compatible(&owner) {
                    return Err(DaemonError::incompatible(
                        "a live incompatible daemon owns the current-user endpoint",
                    ));
                }
                if request.conflicts_with(&owner.limits) {
                    return Err(DaemonError::config_conflict(
                        "the requested daemon settings differ from the live daemon",
                    ));
                }
                match owner.phase {
                    OwnerPhase::Stopping => {
                        wait_step(deadline)?;
                        continue;
                    }
                    OwnerPhase::Starting => {
                        if let Ok(report) = query_status(&paths, &owner, deadline) {
                            return Ok(single_payload(row_from_owner(
                                "start",
                                DaemonState::Ready,
                                Some(true),
                                Some(&report.owner),
                                Some(report.snapshot.usage),
                            )));
                        }
                        wait_step(deadline)?;
                        continue;
                    }
                    OwnerPhase::Ready => match query_status(&paths, &owner, deadline) {
                        Ok(report) => {
                            return Ok(single_payload(row_from_owner(
                                "start",
                                DaemonState::Ready,
                                Some(true),
                                Some(&report.owner),
                                Some(report.snapshot.usage),
                            )));
                        }
                        _ => {
                            if Instant::now() >= deadline {
                                return Err(DaemonError::lifecycle(
                                    "live daemon did not become queryable before the lifecycle deadline",
                                ));
                            }
                            wait_step(deadline)?;
                        }
                    },
                }
            }
            Some(owner) => {
                remove_stale_endpoint(&paths, &owner.token)?;
                remove_owner(&paths, &owner.token)?;
            }
            None => {
                if unattributed_endpoint_exists(&paths)? {
                    return Err(DaemonError::lifecycle(
                        "daemon endpoint exists without safely attributable singleton ownership",
                    ));
                }
                let token = new_owner_token()?;
                let owner = OwnerRecord::starting(token.clone(), requested_limits.clone())?;
                match create_owner(&paths, &owner) {
                    Ok(()) => {
                        if let Err(error) = spawn_daemon(&token) {
                            let _ = remove_owner(&paths, &token);
                            return Err(error);
                        }
                    }
                    Err(error) if error.is_owner_exists() => {
                        wait_step(deadline)?;
                        continue;
                    }
                    Err(error) => return Err(error),
                }
            }
        }
        if Instant::now() >= deadline {
            return Err(DaemonError::lifecycle(
                "daemon did not reach ready before the lifecycle deadline",
            ));
        }
        wait_step(deadline)?;
    }
}

fn stop(matches: &ArgMatches) -> DaemonResult<DaemonPayload> {
    let timeout = Duration::from_millis(lifecycle_timeout(matches)?);
    let deadline = Instant::now() + timeout;
    let paths = RuntimePaths::discover()?;
    loop {
        let Some(owner) = read_owner(&paths)? else {
            if unattributed_endpoint_exists(&paths)? {
                return Err(DaemonError::lifecycle(
                    "daemon endpoint exists without safely attributable singleton ownership",
                ));
            }
            return Ok(single_payload(row_from_owner(
                "stop",
                DaemonState::Stopped,
                None,
                None,
                None,
            )));
        };
        if !process_matches(&owner) {
            remove_stale_endpoint(&paths, &owner.token)?;
            remove_owner(&paths, &owner.token)?;
            continue;
        }
        match control(&paths, &owner, ControlOperation::Stop, deadline) {
            Ok(ServerFrame::Stopping { control_version }) if control_version == CONTROL_VERSION => {
            }
            Ok(_) | Err(_) => {
                if Instant::now() >= deadline {
                    return Err(DaemonError::lifecycle(
                        "live daemon did not acknowledge stop before the lifecycle deadline",
                    ));
                }
            }
        }
        while let Some(current) = read_owner(&paths)? {
            if current.token != owner.token {
                return Err(DaemonError::lifecycle(
                    "daemon singleton ownership changed while stop was waiting for release",
                ));
            }
            if !process_matches(&current) {
                remove_stale_endpoint(&paths, &current.token)?;
                remove_owner(&paths, &current.token)?;
                break;
            }
            wait_step(deadline)?;
        }
    }
}

fn status(matches: &ArgMatches) -> DaemonResult<DaemonPayload> {
    let timeout = Duration::from_millis(lifecycle_timeout(matches)?);
    let deadline = Instant::now() + timeout;
    let paths = RuntimePaths::discover()?;
    let Some(owner) = read_owner(&paths)? else {
        return Ok(status_payload(
            row_from_owner("status", DaemonState::Stopped, None, None, None),
            None,
        ));
    };
    if !process_matches(&owner) {
        return Ok(status_payload(
            row_from_owner("status", DaemonState::Stopped, None, None, None),
            None,
        ));
    }
    if !owner_is_compatible(&owner) {
        return Ok(status_payload(
            row_from_owner(
                "status",
                DaemonState::Incompatible,
                Some(false),
                Some(&owner),
                None,
            ),
            None,
        ));
    }
    match owner.phase {
        OwnerPhase::Starting => Ok(status_payload(
            row_from_owner(
                "status",
                DaemonState::Starting,
                Some(true),
                Some(&owner),
                Some(empty_usage(Some(owner.process_id))),
            ),
            None,
        )),
        OwnerPhase::Stopping => Ok(status_payload(
            row_from_owner(
                "status",
                DaemonState::Stopping,
                Some(true),
                Some(&owner),
                Some(empty_usage(Some(owner.process_id))),
            ),
            None,
        )),
        OwnerPhase::Ready => match query_status(&paths, &owner, deadline) {
            Ok(report) => {
                let snapshot = report.snapshot;
                Ok(status_payload(
                    row_from_owner(
                        "status",
                        DaemonState::Ready,
                        Some(true),
                        Some(&report.owner),
                        Some(snapshot.usage.clone()),
                    ),
                    Some(snapshot),
                ))
            }
            _ => Ok(status_payload(
                row_from_owner(
                    "status",
                    DaemonState::Unresponsive,
                    None,
                    Some(&owner),
                    None,
                ),
                None,
            )),
        },
    }
}

fn control(
    paths: &RuntimePaths,
    owner: &OwnerRecord,
    operation: ControlOperation,
    deadline: Instant,
) -> DaemonResult<ServerFrame> {
    let mut stream =
        runtime::connect(paths, &owner.token, remaining(deadline)?).map_err(|source| {
            DaemonError::unresponsive(format!("cannot connect to live daemon: {source}"))
        })?;
    write_frame(
        &mut stream,
        &ClientFrame::Control {
            control_version: CONTROL_VERSION.to_string(),
            owner_token: owner.token.clone(),
            operation,
            timeout_ms: duration_millis_ceil(remaining(deadline)?),
        },
    )
    .map_err(|source| {
        DaemonError::unresponsive(format!("cannot send daemon control request: {source}"))
    })?;
    stream
        .set_recv_timeout(Some(remaining(deadline)?))
        .map_err(|source| {
            DaemonError::unresponsive(format!(
                "cannot apply the remaining daemon lifecycle deadline: {source}"
            ))
        })?;
    read_frame(&mut BufReader::new(stream)).map_err(|source| {
        DaemonError::unresponsive(format!("cannot receive daemon control response: {source}"))
    })
}

fn query_status(
    paths: &RuntimePaths,
    owner: &OwnerRecord,
    deadline: Instant,
) -> DaemonResult<StatusReport> {
    let response = control(paths, owner, ControlOperation::Status, deadline)?;
    let ServerFrame::Status {
        control_version,
        process_id,
        veloq_version,
        protocol_version,
        limits,
        snapshot,
    } = response
    else {
        return Err(DaemonError::unresponsive(
            "the live daemon returned an invalid lifecycle status response",
        ));
    };
    let current = read_owner(paths)?.ok_or_else(|| {
        DaemonError::unresponsive("daemon ownership disappeared during lifecycle status exchange")
    })?;
    if current.token != owner.token
        || current.phase != OwnerPhase::Ready
        || control_version != CONTROL_VERSION
        || process_id != current.process_id
        || veloq_version != current.veloq_version
        || protocol_version != current.protocol_version
        || limits != current.limits
    {
        return Err(DaemonError::unresponsive(
            "the live daemon returned lifecycle state that does not match singleton ownership",
        ));
    }
    Ok(StatusReport {
        owner: current,
        snapshot: *snapshot,
    })
}

fn spawn_daemon(owner_token: &str) -> DaemonResult<()> {
    let executable = std::env::current_exe().map_err(|source| {
        DaemonError::lifecycle(format!("cannot locate the VeloQ executable: {source}"))
    })?;
    let mut command = ProcessCommand::new(executable);
    command
        .args(["daemon", "__serve", "--owner-token", owner_token])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    command.spawn().map(|_| ()).map_err(|source| {
        DaemonError::lifecycle(format!("cannot create the local daemon process: {source}"))
    })
}

fn owner_is_compatible(owner: &OwnerRecord) -> bool {
    owner.protocol_version == PROTOCOL_VERSION && owner.veloq_version == env!("CARGO_PKG_VERSION")
}

fn row_from_owner(
    operation: &'static str,
    state: DaemonState,
    compatible: Option<bool>,
    owner: Option<&OwnerRecord>,
    usage: Option<DaemonUsage>,
) -> DaemonRow {
    let expose_runtime = compatible == Some(true);
    DaemonRow {
        key: "daemon|local",
        operation,
        state,
        compatible,
        process_id: owner.map(|owner| owner.process_id),
        veloq_version: owner.map(|owner| owner.veloq_version.clone()),
        protocol_version: owner.map(|owner| owner.protocol_version.clone()),
        limits: expose_runtime
            .then(|| owner.map(|owner| owner.limits.clone()))
            .flatten(),
        usage: expose_runtime.then_some(usage).flatten(),
    }
}

fn single_payload(row: DaemonRow) -> DaemonPayload {
    DaemonPayload {
        count: 1,
        total_matched: 1,
        rows: vec![row],
        auxiliary: None,
    }
}

fn status_payload(row: DaemonRow, snapshot: Option<DaemonSnapshot>) -> DaemonPayload {
    let (sessions, evictions) = snapshot
        .map(|snapshot| (snapshot.sessions, Some(snapshot.evictions)))
        .unwrap_or_default();
    DaemonPayload {
        count: 1,
        total_matched: 1,
        rows: vec![row],
        auxiliary: Some(DaemonAuxiliary {
            sessions,
            evictions,
        }),
    }
}

fn empty_usage(process_id: Option<u32>) -> DaemonUsage {
    DaemonUsage {
        resident_sessions: 0,
        resident_memory_estimate_bytes: 0,
        active_requests: 0,
        queued_requests: 0,
        query_workers_reserved: 0,
        query_memory_reserved_bytes: 0,
        exact_response_entries: 0,
        cache_hits: 0,
        cache_misses: 0,
        process_resident_memory_bytes: process_id.and_then(process_resident_memory),
    }
}

fn unattributed_endpoint_exists(paths: &RuntimePaths) -> DaemonResult<bool> {
    #[cfg(unix)]
    {
        let entries = fs::read_dir(&paths.root).map_err(|source| {
            DaemonError::lifecycle(format!(
                "cannot inspect daemon endpoint directory: {source}"
            ))
        })?;
        for entry in entries {
            let entry = entry.map_err(|source| {
                DaemonError::lifecycle(format!("cannot inspect daemon endpoint: {source}"))
            })?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("daemon-v1-") && name.ends_with(".sock") {
                return Ok(true);
            }
        }
        Ok(false)
    }
    #[cfg(not(unix))]
    Ok(false)
}

fn remaining(deadline: Instant) -> DaemonResult<Duration> {
    deadline
        .checked_duration_since(Instant::now())
        .ok_or_else(|| {
            DaemonError::lifecycle("daemon lifecycle deadline expired before transition completed")
        })
}

fn duration_millis_ceil(duration: Duration) -> u64 {
    let millis = duration.as_millis();
    let rounded = millis.saturating_add(u128::from(
        duration.subsec_nanos() % NANOSECONDS_PER_MILLISECOND != 0,
    ));
    rounded.try_into().unwrap_or(u64::MAX)
}

fn wait_step(deadline: Instant) -> DaemonResult<()> {
    let remaining = remaining(deadline)?;
    thread::sleep(remaining.min(LIFECYCLE_POLL_INTERVAL));
    Ok(())
}
