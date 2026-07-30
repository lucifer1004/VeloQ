use std::io::{self, BufReader, Read};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use interprocess::local_socket::{
    ConnectOptions, ListenerNonblockingMode, ListenerOptions, prelude::*,
};
#[cfg(unix)]
use interprocess::os::unix::local_socket::ListenerOptionsExt;
use interprocess::{ConnectWaitMode, TryClone};
#[cfg(windows)]
use sysinfo::{Pid, ProcessesToUpdate, System};

use super::config::MAX_LIFECYCLE_TIMEOUT_MS;
use super::protocol::{
    CONTROL_VERSION, ClientFrame, ControlOperation, OUTPUT_CHUNK_BYTES, OutputStream,
    PROTOCOL_VERSION, ServerFrame, read_frame, write_frame,
};
use super::session::{
    AcceptOutcome, AdmissionFailure, DaemonEngine, ExactQueryKey, QueryReservation, SessionSpec,
};
use super::state::{OwnerPhase, RuntimePaths, read_owner, remove_owner, replace_owner};
use super::{DaemonError, DaemonResult};
use veloq_core::{
    OutputFormat, ProfileSource, SourceExecution, SourceRunResult, SourceSessionConfig,
};

const CONNECTION_IO_TIMEOUT: Duration = Duration::from_secs(30);
const ACCEPT_POLL_INTERVAL: Duration = Duration::from_millis(10);
const CLIENT_DISCONNECT_POLL_INTERVAL: Duration = Duration::from_millis(1);
const LIFECYCLE_CONNECTION_RESERVE: u64 = 1;

#[derive(Debug)]
struct ConnectionLimiter {
    active: AtomicU64,
    limit: u64,
}

impl ConnectionLimiter {
    fn new(limits: &super::config::DaemonLimits) -> Arc<Self> {
        Arc::new(Self {
            active: AtomicU64::new(0),
            limit: limits
                .max_concurrent_requests
                .saturating_add(limits.max_queued_requests)
                .saturating_add(LIFECYCLE_CONNECTION_RESERVE),
        })
    }

    fn try_acquire(self: &Arc<Self>) -> Option<ConnectionPermit> {
        let mut active = self.active.load(Ordering::Acquire);
        loop {
            if active >= self.limit {
                return None;
            }
            match self.active.compare_exchange_weak(
                active,
                active.saturating_add(1),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Some(ConnectionPermit {
                        limiter: Arc::clone(self),
                    });
                }
                Err(observed) => active = observed,
            }
        }
    }
}

struct ConnectionPermit {
    limiter: Arc<ConnectionLimiter>,
}

impl Drop for ConnectionPermit {
    fn drop(&mut self) {
        self.limiter.active.fetch_sub(1, Ordering::AcqRel);
    }
}

pub fn serve(owner_token: &str, sources: &[Arc<dyn ProfileSource>]) -> DaemonResult<i32> {
    let paths = RuntimePaths::discover()?;
    let owner = read_owner(&paths)?
        .ok_or_else(|| DaemonError::lifecycle("daemon ownership record is missing"))?;
    if owner.token != owner_token || owner.phase != OwnerPhase::Starting {
        return Err(DaemonError::lifecycle(
            "daemon ownership changed before the process became ready",
        ));
    }

    let name = paths.socket_name(owner_token)?;
    let options = ListenerOptions::new()
        .name(name)
        .reclaim_name(true)
        .try_overwrite(false);
    #[cfg(unix)]
    let options = options.mode(0o600);
    let listener = options.create_sync().map_err(|source| {
        DaemonError::lifecycle(format!(
            "cannot create current-user daemon endpoint: {source}"
        ))
    })?;

    let ready = owner.for_daemon()?;
    replace_owner(&paths, owner_token, &ready)?;
    let engine = DaemonEngine::new(ready.limits.clone());
    let paths = Arc::new(paths);
    let ready = Arc::new(ready);
    let sources = Arc::new(sources.to_vec());
    let limiter = ConnectionLimiter::new(&ready.limits);
    let (stop_sender, stop_receiver) = mpsc::sync_channel(1);
    listener
        .set_nonblocking(ListenerNonblockingMode::Accept)
        .map_err(|source| {
            DaemonError::lifecycle(format!(
                "cannot make daemon admission loop interruptible: {source}"
            ))
        })?;

    let shutdown_timeout = loop {
        if let Ok(timeout) = stop_receiver.try_recv() {
            break timeout;
        }

        match listener.accept() {
            Ok(mut stream) => {
                let Some(permit) = limiter.try_acquire() else {
                    log::debug!("daemon connection limit reached; dropping excess local client");
                    continue;
                };
                let paths = Arc::clone(&paths);
                let ready = Arc::clone(&ready);
                let engine = engine.clone();
                let sources = Arc::clone(&sources);
                let stop_sender = stop_sender.clone();
                std::thread::spawn(move || {
                    let _permit = permit;
                    let action =
                        handle_accepted_connection(&paths, &ready, &engine, &sources, &mut stream);
                    match action {
                        Ok(Some(timeout)) => {
                            let _ = stop_sender.try_send(timeout);
                        }
                        Ok(None) => {}
                        Err(source) => {
                            log::warn!("daemon local IPC request failed: {source}");
                            let stopping = read_owner(&paths).ok().flatten().is_some_and(|owner| {
                                owner.token == ready.token && owner.phase == OwnerPhase::Stopping
                            });
                            if stopping {
                                let _ = stop_sender.try_send(Duration::from_millis(
                                    ready.limits.shutdown_grace_ms,
                                ));
                            }
                        }
                    }
                });
            }
            Err(source) if source.kind() == io::ErrorKind::WouldBlock => {
                std::thread::sleep(ACCEPT_POLL_INTERVAL);
            }
            Err(source) => {
                log::warn!("daemon local IPC accept failed: {source}");
                std::thread::sleep(ACCEPT_POLL_INTERVAL);
            }
        }
    };

    drop(listener);
    let shutdown_deadline = Instant::now() + shutdown_timeout;
    let grace = shutdown_timeout.min(Duration::from_millis(ready.limits.shutdown_grace_ms));
    if !engine.wait_for_active_drain(grace) {
        engine.cancel_active_requests();
        let remaining = shutdown_deadline.saturating_duration_since(Instant::now());
        let _ = engine.wait_for_active_drain(remaining);
    }
    remove_owner(&paths, owner_token)?;
    Ok(0)
}

fn handle_accepted_connection(
    paths: &RuntimePaths,
    owner: &super::state::OwnerRecord,
    engine: &DaemonEngine,
    sources: &[Arc<dyn ProfileSource>],
    stream: &mut LocalSocketStream,
) -> DaemonResult<Option<Duration>> {
    if !peer_is_current_user(stream)? {
        return Ok(None);
    }
    stream
        .set_recv_timeout(Some(CONNECTION_IO_TIMEOUT))
        .and_then(|_| stream.set_send_timeout(Some(CONNECTION_IO_TIMEOUT)))
        .map_err(|source| {
            DaemonError::lifecycle(format!("cannot bound daemon IPC request: {source}"))
        })?;
    handle_connection(paths, owner, engine, sources, stream)
}

pub fn connect(
    paths: &RuntimePaths,
    owner_token: &str,
    timeout: Duration,
) -> io::Result<LocalSocketStream> {
    let name = paths
        .socket_name(owner_token)
        .map_err(|error| io::Error::other(error.to_string()))?;
    let stream = ConnectOptions::new()
        .name(name)
        .wait_mode(ConnectWaitMode::Timeout(timeout))
        .connect_sync()?;
    stream.set_recv_timeout(Some(timeout))?;
    stream.set_send_timeout(Some(timeout))?;
    Ok(stream)
}

fn handle_connection(
    paths: &RuntimePaths,
    owner: &super::state::OwnerRecord,
    engine: &DaemonEngine,
    sources: &[Arc<dyn ProfileSource>],
    stream: &mut LocalSocketStream,
) -> DaemonResult<Option<Duration>> {
    let mut reader = BufReader::new(stream);
    let first: ClientFrame = read_frame(&mut reader).map_err(|source| {
        DaemonError::lifecycle(format!("cannot read local daemon request: {source}"))
    })?;
    match first {
        ClientFrame::Control {
            control_version,
            owner_token,
            operation,
            timeout_ms,
        } => {
            if control_version != CONTROL_VERSION || owner_token != owner.token {
                return Ok(None);
            }
            match operation {
                ControlOperation::Status => {
                    write_frame(
                        reader.get_mut(),
                        &ServerFrame::Status {
                            control_version: CONTROL_VERSION.to_string(),
                            process_id: std::process::id(),
                            veloq_version: env!("CARGO_PKG_VERSION").to_string(),
                            protocol_version: PROTOCOL_VERSION.to_string(),
                            limits: owner.limits.clone(),
                            snapshot: Box::new(engine.snapshot()),
                        },
                    )
                    .map_err(|source| {
                        DaemonError::lifecycle(format!(
                            "cannot write daemon status response: {source}"
                        ))
                    })?;
                    Ok(None)
                }
                ControlOperation::Stop => {
                    engine.begin_shutdown();
                    let mut stopping = owner.clone();
                    stopping.phase = OwnerPhase::Stopping;
                    replace_owner(paths, &owner.token, &stopping)?;
                    write_frame(
                        reader.get_mut(),
                        &ServerFrame::Stopping {
                            control_version: CONTROL_VERSION.to_string(),
                        },
                    )
                    .map_err(|source| {
                        DaemonError::lifecycle(format!(
                            "cannot write daemon stop response: {source}"
                        ))
                    })?;
                    Ok(Some(Duration::from_millis(
                        timeout_ms.min(MAX_LIFECYCLE_TIMEOUT_MS),
                    )))
                }
            }
        }
        ClientFrame::Hello {
            protocol_version,
            veloq_version,
        } => {
            let compatible =
                protocol_version == PROTOCOL_VERSION && veloq_version == env!("CARGO_PKG_VERSION");
            write_frame(
                reader.get_mut(),
                &ServerFrame::Hello {
                    protocol_version: PROTOCOL_VERSION.to_string(),
                    veloq_version: env!("CARGO_PKG_VERSION").to_string(),
                    compatible,
                    capabilities: if compatible {
                        sources
                            .iter()
                            .flat_map(|source| {
                                source.daemon_command_verbs().iter().map(|verb| {
                                    super::protocol::Capability {
                                        source: source.kind().to_string(),
                                        command: format!("{}.{verb}", source.kind()),
                                    }
                                })
                            })
                            .collect()
                    } else {
                        Vec::new()
                    },
                },
            )
            .map_err(|source| {
                DaemonError::lifecycle(format!("cannot write daemon handshake: {source}"))
            })?;
            if !compatible {
                return Ok(None);
            }
            let request: ClientFrame = read_frame(&mut reader).map_err(|source| {
                DaemonError::lifecycle(format!("cannot read daemon execution request: {source}"))
            })?;
            if let ClientFrame::Query {
                request_id,
                source,
                command,
                invocation: request_invocation,
            } = request
            {
                if !request_invocation.environment_matches_current() {
                    write_rejection(
                        reader.get_mut(),
                        request_id,
                        DaemonError::incompatible(
                            "the client query environment differs from the daemon environment",
                        ),
                    )?;
                    return Ok(None);
                }
                let prepared = match prepare_query(sources, &source, &command, &request_invocation)
                {
                    Ok(prepared) => prepared,
                    Err(error) => {
                        write_rejection(reader.get_mut(), request_id, error)?;
                        return Ok(None);
                    }
                };
                write_frame(
                    reader.get_mut(),
                    &ServerFrame::Accepted {
                        request_id: request_id.clone(),
                    },
                )
                .map_err(|source| {
                    DaemonError::lifecycle(format!(
                        "cannot write daemon acceptance response: {source}"
                    ))
                })?;
                let execution = loop {
                    let admission =
                        prepared.admit(engine, sources, &request_id, &request_invocation);
                    let (accepted, exact_key, session_config, admitted_freshness_key) =
                        match admission {
                            Ok(PreparedAdmission::Cached(execution)) => {
                                write_completed_execution(
                                    reader.get_mut(),
                                    request_id,
                                    execution.as_ref(),
                                )?;
                                return Ok(None);
                            }
                            Ok(PreparedAdmission::Active {
                                accepted,
                                exact_key,
                                session_config,
                                admitted_freshness_key,
                            }) => (accepted, exact_key, session_config, admitted_freshness_key),
                            Err(failure) => {
                                write_admission_failure(reader.get_mut(), request_id, failure)?;
                                return Ok(None);
                            }
                        };
                    let active = match monitor_client_disconnect(
                        reader.get_ref(),
                        || {
                            let _ = engine.cancel(&request_id);
                        },
                        || accepted.wait_until_active(),
                    ) {
                        Ok(active) => active,
                        Err(AdmissionFailure::SessionInvalidated) => continue,
                        Err(failure) => {
                            write_admission_failure(reader.get_mut(), request_id, failure)?;
                            return Ok(None);
                        }
                    };
                    if !prepared.active_freshness_is_current(
                        &active,
                        sources,
                        admitted_freshness_key.as_deref(),
                    ) {
                        active.complete(None, None);
                        continue;
                    }
                    let active_execution = prepared.execute_active_monitoring_disconnect(
                        &active,
                        sources,
                        session_config,
                        reader.get_ref(),
                    );
                    if active.cancellation_requested() {
                        active.discard_resident_state_after_failure();
                        active.complete(None, None);
                        write_failed(
                            reader.get_mut(),
                            request_id,
                            DaemonError::cancelled("the accepted daemon request was cancelled"),
                        )
                        .map_err(|source| {
                            DaemonError::lifecycle(format!(
                                "cannot write daemon cancellation response: {source}"
                            ))
                        })?;
                        return Ok(None);
                    }
                    break match active_execution {
                        Ok((execution, additional_resident_bytes)) => {
                            let cacheable = prepared.refresh_active(
                                &active,
                                sources,
                                admitted_freshness_key.as_deref(),
                                additional_resident_bytes,
                            );
                            active.complete(
                                cacheable.then_some(exact_key).flatten(),
                                cacheable.then_some(&execution),
                            );
                            Ok(execution)
                        }
                        Err(error) => {
                            active.discard_resident_state_after_failure();
                            active.complete(None, None);
                            Err(error)
                        }
                    };
                };
                write_execution_result(reader.get_mut(), request_id, prepared.format, execution)?;
            }
            Ok(None)
        }
        _ => Ok(None),
    }
}

struct PreparedQuery {
    source_index: usize,
    matches: clap::ArgMatches,
    format: OutputFormat,
    context: veloq_core::SourceQueryContext,
    resolved_trace: std::path::PathBuf,
    terminal_width: Option<u16>,
}

impl PreparedQuery {
    fn execute_active_monitoring_disconnect(
        &self,
        active: &super::session::ActiveRequest,
        sources: &[Arc<dyn ProfileSource>],
        session_config: SourceSessionConfig,
        stream: &LocalSocketStream,
    ) -> SourceRunResult<(SourceExecution, u64)> {
        let cancellation = active.cancellation_token();
        monitor_client_disconnect(
            stream,
            || cancellation.cancel(),
            || self.execute_active(active, sources, session_config),
        )
    }

    fn execute_active(
        &self,
        active: &super::session::ActiveRequest,
        sources: &[Arc<dyn ProfileSource>],
        session_config: SourceSessionConfig,
    ) -> SourceRunResult<(SourceExecution, u64)> {
        let source = sources.get(self.source_index).ok_or_else(|| {
            Box::new(io::Error::other(
                "the selected daemon source is no longer registered",
            )) as veloq_core::SourceRunError
        })?;
        let source_matches = source_matches(&self.matches, source.kind());
        let cancellation = active.cancellation_token();
        let resident = veloq_core::tabular::with_terminal_width(self.terminal_width, || {
            active.execute_with_resident(
                || match source.open_daemon_session(&self.resolved_trace, session_config) {
                    Ok(session) => Ok(session),
                    Err(source) => {
                        log::warn!(
                            "daemon resident session could not be opened for {}: {source}",
                            self.resolved_trace.display()
                        );
                        Ok(None)
                    }
                },
                |session| session.execute(source_matches, self.format, &cancellation),
            )
        })?;
        match resident {
            Some(execution) => Ok(execution),
            None => veloq_core::tabular::with_terminal_width(self.terminal_width, || {
                source.execute_daemon_cancellable(
                    source_matches,
                    self.format,
                    &self.resolved_trace,
                    session_config,
                    &cancellation,
                )
            })
            .map(|execution| (execution, 0)),
        }
    }

    fn admit(
        &self,
        engine: &DaemonEngine,
        sources: &[Arc<dyn ProfileSource>],
        request_id: &str,
        invocation: &super::protocol::QueryInvocation,
    ) -> Result<PreparedAdmission, AdmissionFailure> {
        let source = sources.get(self.source_index);
        let identity = match source {
            Some(source) => match source.daemon_session_identity(&self.resolved_trace) {
                Ok(identity) => identity,
                Err(error) => {
                    log::warn!(
                        "daemon session identity is unavailable for {}: {error}",
                        self.resolved_trace.display()
                    );
                    None
                }
            },
            None => None,
        };
        let admitted_freshness_key = identity
            .as_ref()
            .map(|identity| identity.freshness_key.clone());
        let spec = source.zip(identity).map(|(source, identity)| SessionSpec {
            source_kind: source.kind().to_string(),
            source_version: source.version().to_string(),
            trace_kind: identity.trace_kind,
            canonical_trace_path: identity.canonical_trace_path.display().to_string(),
            configuration_key: identity.configuration_key,
            freshness_key: identity.freshness_key,
            resident_memory_estimate_bytes: identity.resident_memory_estimate_bytes,
        });
        let exact_key = spec.as_ref().map(|_| {
            ExactQueryKey::new(
                &self.context.command,
                self.format,
                invocation.semantic_key(
                    self.context.trace_path.as_deref().map(Path::as_os_str),
                    self.format,
                ),
            )
        });
        let concurrent = engine.limits().max_concurrent_requests;
        let workers = engine.limits().max_query_workers.div_ceil(concurrent);
        let memory = engine
            .limits()
            .max_query_memory_bytes
            .map(|bytes| bytes.div_ceil(concurrent));
        match engine.accept(
            request_id,
            spec,
            QueryReservation::new(workers, memory.unwrap_or(0)),
            exact_key.as_ref(),
        )? {
            AcceptOutcome::Cached(execution) => Ok(PreparedAdmission::Cached(execution)),
            AcceptOutcome::Accepted(accepted) => Ok(PreparedAdmission::Active {
                accepted,
                exact_key,
                session_config: SourceSessionConfig {
                    query_workers: workers,
                    query_memory_bytes: memory,
                },
                admitted_freshness_key,
            }),
        }
    }

    fn refresh_active(
        &self,
        active: &super::session::ActiveRequest,
        sources: &[Arc<dyn ProfileSource>],
        admitted_freshness_key: Option<&str>,
        additional_resident_bytes: u64,
    ) -> bool {
        let Some(admitted_freshness_key) = admitted_freshness_key else {
            return false;
        };
        let identity = sources
            .get(self.source_index)
            .and_then(|source| {
                source
                    .daemon_session_identity(&self.resolved_trace)
                    .map_err(|error| {
                        log::warn!(
                            "daemon post-query session identity is unavailable for {}: {error}",
                            self.resolved_trace.display()
                        );
                    })
                    .ok()
            })
            .flatten();
        let resident_memory_estimate_bytes = identity.as_ref().map_or(0, |identity| {
            identity
                .resident_memory_estimate_bytes
                .saturating_add(additional_resident_bytes)
        });
        active.refresh_resident_state(
            admitted_freshness_key,
            identity
                .as_ref()
                .map(|identity| identity.freshness_key.as_str()),
            resident_memory_estimate_bytes,
        )
    }

    fn active_freshness_is_current(
        &self,
        active: &super::session::ActiveRequest,
        sources: &[Arc<dyn ProfileSource>],
        admitted_freshness_key: Option<&str>,
    ) -> bool {
        let Some(admitted_freshness_key) = admitted_freshness_key else {
            return true;
        };
        let observed = sources.get(self.source_index).and_then(|source| {
            source
                .daemon_session_identity(&self.resolved_trace)
                .map_err(|error| {
                    log::warn!(
                        "daemon activation freshness could not be revalidated for {}: {error}",
                        self.resolved_trace.display()
                    );
                })
                .ok()
                .flatten()
        });
        if observed
            .as_ref()
            .map(|identity| identity.freshness_key.as_str())
            == Some(admitted_freshness_key)
        {
            return true;
        }
        active.refresh_resident_state(
            admitted_freshness_key,
            observed
                .as_ref()
                .map(|identity| identity.freshness_key.as_str()),
            0,
        );
        false
    }
}

fn monitor_client_disconnect<T: Send>(
    stream: &LocalSocketStream,
    on_disconnect: impl FnOnce(),
    operation: impl FnOnce() -> T + Send,
) -> T {
    let Ok(mut monitor) = stream.try_clone() else {
        return operation();
    };
    if monitor
        .set_recv_timeout(Some(CLIENT_DISCONNECT_POLL_INTERVAL))
        .is_err()
    {
        return operation();
    }
    std::thread::scope(|scope| {
        let operation = scope.spawn(operation);
        let mut probe = [0u8; 1];
        while !operation.is_finished() {
            match monitor.read(&mut probe) {
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock
                            | io::ErrorKind::TimedOut
                            | io::ErrorKind::Interrupted
                    ) => {}
                _ => {
                    on_disconnect();
                    break;
                }
            }
        }
        match operation.join() {
            Ok(result) => result,
            Err(panic) => std::panic::resume_unwind(panic),
        }
    })
}

enum PreparedAdmission {
    Cached(Arc<SourceExecution>),
    Active {
        accepted: super::session::AcceptedRequest,
        exact_key: Option<ExactQueryKey>,
        session_config: SourceSessionConfig,
        admitted_freshness_key: Option<String>,
    },
}

fn prepare_query(
    sources: &[Arc<dyn ProfileSource>],
    expected_source: &str,
    expected_command: &str,
    invocation: &super::protocol::QueryInvocation,
) -> DaemonResult<PreparedQuery> {
    let matches = super::query_cli(sources)
        .try_get_matches_from(invocation.decoded_arguments())
        .map_err(|source| {
            DaemonError::unsupported(format!(
                "the daemon could not parse the client invocation: {source}"
            ))
        })?;
    let fmt = matches
        .get_one::<String>("format")
        .map(String::as_str)
        .unwrap_or("json");
    let fmt =
        OutputFormat::parse(fmt).map_err(|source| DaemonError::unsupported(source.to_string()))?;
    let (sub_name, sub_matches) = matches.subcommand().ok_or_else(|| {
        DaemonError::unsupported("the daemon query invocation has no source command")
    })?;

    let (source_index, source, source_matches): (usize, &dyn ProfileSource, &clap::ArgMatches) =
        if let Some((index, source)) = sources
            .iter()
            .enumerate()
            .find(|(_, source)| source.kind() == sub_name)
        {
            (index, source.as_ref(), sub_matches)
        } else {
            let (index, source) = sources
                .iter()
                .enumerate()
                .find(|(_, source)| source.kind() == super::DEFAULT_SOURCE)
                .ok_or_else(|| {
                    DaemonError::unsupported("the daemon default source is not registered")
                })?;
            (index, source.as_ref(), &matches)
        };
    if source.kind() != expected_source {
        return Err(DaemonError::unsupported(
            "the parsed source does not match capability negotiation",
        ));
    }
    if !source.supports_daemon_command(expected_command) {
        return Err(DaemonError::unsupported(
            "the selected source command is not daemon-enabled",
        ));
    }
    let context = source.query_context(source_matches).map_err(|source| {
        DaemonError::unsupported(format!(
            "the daemon could not resolve query context: {source}"
        ))
    })?;
    if context.command != expected_command {
        return Err(DaemonError::unsupported(
            "the parsed command does not match capability negotiation",
        ));
    }
    let trace = context
        .trace_path
        .as_deref()
        .ok_or_else(|| DaemonError::unsupported("the daemon-enabled query has no trace path"))?;
    let cwd = invocation.decoded_cwd();
    if !cwd.is_absolute() {
        return Err(DaemonError::unsupported(
            "the daemon query working directory is not absolute",
        ));
    }
    let resolved_trace = if trace.is_absolute() {
        trace.to_path_buf()
    } else {
        cwd.join(trace)
    };
    Ok(PreparedQuery {
        source_index,
        matches,
        format: fmt,
        context,
        resolved_trace,
        terminal_width: invocation.terminal_width,
    })
}

fn source_matches<'a>(matches: &'a clap::ArgMatches, source_kind: &str) -> &'a clap::ArgMatches {
    match matches.subcommand() {
        Some((sub_name, sub_matches)) if sub_name == source_kind => sub_matches,
        _ => matches,
    }
}

fn write_rejection(
    stream: &mut LocalSocketStream,
    request_id: String,
    error: DaemonError,
) -> DaemonResult<()> {
    write_frame(stream, &ServerFrame::Rejected { request_id, error }).map_err(|source| {
        DaemonError::lifecycle(format!(
            "cannot write daemon pre-acceptance rejection: {source}"
        ))
    })
}

fn write_admission_failure(
    stream: &mut LocalSocketStream,
    request_id: String,
    failure: AdmissionFailure,
) -> DaemonResult<()> {
    let error = match failure {
        AdmissionFailure::ResourcePressure => DaemonError::resource_pressure(
            "the request could not enter active execution within daemon resource limits",
        ),
        AdmissionFailure::Cancelled | AdmissionFailure::ShuttingDown => {
            DaemonError::cancelled("the accepted daemon request was cancelled")
        }
        AdmissionFailure::SessionInvalidated
        | AdmissionFailure::DuplicateRequest
        | AdmissionFailure::UnknownRequest => DaemonError::execution_indeterminate(
            "the daemon could not establish unique execution ownership",
        ),
    };
    write_failed(stream, request_id, error).map_err(|source| {
        DaemonError::lifecycle(format!("cannot write daemon admission failure: {source}"))
    })
}

fn write_failed(
    stream: &mut impl io::Write,
    request_id: String,
    error: DaemonError,
) -> io::Result<()> {
    write_frame(stream, &ServerFrame::Failed { request_id, error })
}

fn write_execution_result(
    stream: &mut impl io::Write,
    request_id: String,
    format: OutputFormat,
    execution: SourceRunResult<SourceExecution>,
) -> DaemonResult<()> {
    let execution = match execution {
        Ok(execution) => execution,
        Err(source) => {
            let error = crate::error::CliError::source_run(source);
            crate::render_cli_diagnostic_execution(&error, format)
        }
    };
    write_completed_execution(stream, request_id, &execution)
}

fn write_completed_execution(
    stream: &mut impl io::Write,
    request_id: String,
    execution: &SourceExecution,
) -> DaemonResult<()> {
    write_output_chunks(
        stream,
        &request_id,
        OutputStream::Stdout,
        execution.stdout(),
    )?;
    write_output_chunks(
        stream,
        &request_id,
        OutputStream::Stderr,
        execution.stderr(),
    )?;
    write_frame(
        stream,
        &ServerFrame::Completed {
            request_id,
            exit_code: execution.exit_code(),
        },
    )
    .map_err(|source| {
        DaemonError::lifecycle(format!(
            "cannot write daemon query terminal response: {source}"
        ))
    })
}

fn write_output_chunks(
    stream: &mut impl io::Write,
    request_id: &str,
    output_stream: OutputStream,
    bytes: &[u8],
) -> DaemonResult<()> {
    for chunk in bytes.chunks(OUTPUT_CHUNK_BYTES) {
        write_frame(
            stream,
            &ServerFrame::OutputChunk {
                request_id: request_id.to_string(),
                stream: output_stream,
                bytes: chunk.to_vec(),
            },
        )
        .map_err(|source| {
            DaemonError::lifecycle(format!("cannot write daemon query output chunk: {source}"))
        })?;
    }
    Ok(())
}

#[cfg(unix)]
fn peer_is_current_user(stream: &LocalSocketStream) -> DaemonResult<bool> {
    let credentials = stream.peer_creds().map_err(|source| {
        DaemonError::lifecycle(format!(
            "cannot inspect local daemon peer credentials: {source}"
        ))
    })?;
    Ok(credentials.euid() == Some(unsafe { libc::geteuid() }))
}

#[cfg(windows)]
fn peer_is_current_user(stream: &LocalSocketStream) -> DaemonResult<bool> {
    let credentials = stream.peer_creds().map_err(|source| {
        DaemonError::lifecycle(format!(
            "cannot inspect local daemon peer credentials: {source}"
        ))
    })?;
    let peer_pid = credentials
        .pid()
        .ok_or_else(|| DaemonError::lifecycle("daemon peer process identity is unavailable"))?;
    let self_pid = std::process::id();
    let pids = [Pid::from_u32(peer_pid), Pid::from_u32(self_pid)];
    let mut system = System::new();
    system.refresh_processes(ProcessesToUpdate::Some(&pids), true);
    let peer_user = system
        .process(pids[0])
        .and_then(|process| process.user_id());
    let self_user = system
        .process(pids[1])
        .and_then(|process| process.user_id());
    Ok(peer_user.is_some() && peer_user == self_user)
}

#[cfg(not(any(unix, windows)))]
fn peer_is_current_user(_stream: &LocalSocketStream) -> DaemonResult<bool> {
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufReader, Cursor};

    #[test]
    fn completed_output_is_split_into_bounded_frames() -> DaemonResult<()> {
        let payload = vec![255; OUTPUT_CHUNK_BYTES.saturating_mul(5).saturating_add(17)];
        let execution = SourceExecution::from_parts(0, payload.clone(), Vec::new());
        let mut wire = Vec::new();
        write_completed_execution(&mut wire, "large".to_string(), &execution)?;

        let mut reconstructed = Vec::new();
        let mut reader = BufReader::new(Cursor::new(wire));
        loop {
            let frame: ServerFrame = read_frame(&mut reader)
                .map_err(|source| DaemonError::lifecycle(source.to_string()))?;
            match frame {
                ServerFrame::OutputChunk {
                    stream: OutputStream::Stdout,
                    bytes,
                    ..
                } => reconstructed.extend_from_slice(&bytes),
                ServerFrame::Completed { exit_code, .. } => {
                    assert_eq!(exit_code, 0);
                    break;
                }
                _ => {}
            }
        }
        assert_eq!(reconstructed, payload);
        Ok(())
    }

    #[test]
    fn known_source_failure_is_a_completed_cli_error() -> DaemonResult<()> {
        let mut wire = Vec::new();
        let failure: veloq_core::SourceRunError = io::Error::other("source failed").into();
        write_execution_result(
            &mut wire,
            "failure".to_string(),
            OutputFormat::Json,
            Err(failure),
        )?;

        let mut stdout = Vec::new();
        let mut reader = BufReader::new(Cursor::new(wire));
        let exit_code = loop {
            let frame: ServerFrame = read_frame(&mut reader)
                .map_err(|source| DaemonError::lifecycle(source.to_string()))?;
            match frame {
                ServerFrame::OutputChunk {
                    stream: OutputStream::Stdout,
                    bytes,
                    ..
                } => stdout.extend_from_slice(&bytes),
                ServerFrame::Completed {
                    exit_code: completed,
                    ..
                } => break completed,
                ServerFrame::Failed { .. } => {
                    return Err(DaemonError::lifecycle(
                        "known source failure used daemon failure transport",
                    ));
                }
                _ => {}
            }
        };
        assert_eq!(exit_code, 1);
        let payload: serde_json::Value = serde_json::from_slice(&stdout)
            .map_err(|source| DaemonError::lifecycle(source.to_string()))?;
        assert_eq!(
            payload
                .pointer("/error/code")
                .and_then(|value| value.as_str()),
            Some("cli.source-run")
        );
        Ok(())
    }
}
