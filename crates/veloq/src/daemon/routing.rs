use std::io::BufReader;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use clap::ArgMatches;
use interprocess::local_socket::prelude::*;
use veloq_core::{
    EnvelopeError, EnvelopeTraceRef, OutputFormat, ProfileSource, SourceExecution, SourceRunResult,
};

use super::DaemonError;
use super::config::{QueryRouting, RoutingMode};
use super::protocol::{
    ClientFrame, OutputStream, PROTOCOL_VERSION, QueryInvocation, RequestOwnership, ServerFrame,
    read_frame, write_frame,
};
use super::runtime;
use super::state::{OwnerPhase, RuntimePaths, process_matches, read_owner};

static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

pub fn execute_selected(
    source: &dyn ProfileSource,
    matches: &ArgMatches,
    fmt: OutputFormat,
) -> SourceRunResult<SourceExecution> {
    match super::config::query_routing(matches) {
        Ok(Some(routing)) => execute(source, matches, fmt, routing),
        Ok(None) => source.execute(matches, fmt),
        Err(error) => {
            let context = source.query_context(matches)?;
            Ok(render_query_error(source, &context, fmt, &error))
        }
    }
}

pub fn execute(
    source: &dyn ProfileSource,
    matches: &ArgMatches,
    fmt: OutputFormat,
    routing: QueryRouting,
) -> SourceRunResult<SourceExecution> {
    if routing.mode == RoutingMode::Off {
        return source.execute(matches, fmt);
    }

    let context = source.query_context(matches)?;
    if context.trace_path.is_none() {
        return source.execute(matches, fmt);
    }

    let invocation = match super::protocol::QueryInvocation::capture() {
        Ok(invocation) => invocation,
        Err(source_error) => {
            return route_failure(
                source,
                matches,
                fmt,
                routing.mode,
                context,
                DaemonError::unresponsive(format!(
                    "cannot capture the query invocation context: {source_error}"
                )),
            );
        }
    };
    let deadline = Instant::now() + Duration::from_millis(routing.connect_timeout_ms);
    let outcome = negotiate(source, &context.command, deadline);
    match outcome {
        Ok(Negotiated::Supported(stream)) => {
            match execute_daemon_query(stream, source, &context, invocation) {
                Ok(DaemonQueryOutcome::Completed(execution)) => Ok(execution),
                Ok(DaemonQueryOutcome::Rejected(error)) => {
                    if routing.mode == RoutingMode::Auto {
                        source.execute(matches, fmt)
                    } else {
                        Ok(render_query_error(source, &context, fmt, &error))
                    }
                }
                Err(error) => Ok(render_query_error(source, &context, fmt, &error)),
            }
        }
        Ok(Negotiated::Unsupported) => route_failure(
            source,
            matches,
            fmt,
            routing.mode,
            context,
            DaemonError::unsupported("the selected source command is not daemon-enabled"),
        ),
        Err(error) => route_failure(source, matches, fmt, routing.mode, context, error),
    }
}

enum Negotiated {
    Supported(LocalSocketStream),
    Unsupported,
}

enum DaemonQueryOutcome {
    Completed(SourceExecution),
    Rejected(DaemonError),
}

fn negotiate(
    source: &dyn ProfileSource,
    command: &str,
    deadline: Instant,
) -> Result<Negotiated, DaemonError> {
    let paths = RuntimePaths::discover()?;
    let Some(owner) = read_owner(&paths)? else {
        return Err(DaemonError::absent(
            "no live current-user daemon owns the local endpoint",
        ));
    };
    if !process_matches(&owner) {
        return Err(DaemonError::absent(
            "no live current-user daemon owns the local endpoint",
        ));
    }
    if owner.phase != OwnerPhase::Ready {
        return Err(DaemonError::unresponsive(
            "the live daemon is not ready for query capability exchange",
        ));
    }
    if owner.protocol_version != PROTOCOL_VERSION
        || owner.veloq_version != env!("CARGO_PKG_VERSION")
    {
        return Err(DaemonError::incompatible(
            "the live daemon is incompatible with this VeloQ client",
        ));
    }

    let timeout = remaining(deadline)?;
    let mut stream = runtime::connect(&paths, &owner.token, timeout).map_err(|source| {
        DaemonError::unresponsive(format!(
            "the live daemon did not accept a local connection: {source}"
        ))
    })?;
    write_frame(
        &mut stream,
        &ClientFrame::Hello {
            protocol_version: PROTOCOL_VERSION.to_string(),
            veloq_version: env!("CARGO_PKG_VERSION").to_string(),
        },
    )
    .map_err(|source| {
        DaemonError::unresponsive(format!(
            "the daemon capability exchange could not be sent: {source}"
        ))
    })?;
    let timeout = remaining(deadline)?;
    stream.set_recv_timeout(Some(timeout)).map_err(|source| {
        DaemonError::unresponsive(format!(
            "the daemon capability deadline could not be applied: {source}"
        ))
    })?;
    let mut reader = BufReader::new(stream);
    let response: ServerFrame = read_frame(&mut reader).map_err(|source| {
        DaemonError::unresponsive(format!(
            "the daemon capability exchange did not complete: {source}"
        ))
    })?;
    match response {
        ServerFrame::Hello {
            compatible,
            protocol_version,
            veloq_version,
            capabilities,
        } => {
            if !compatible
                || protocol_version != PROTOCOL_VERSION
                || veloq_version != env!("CARGO_PKG_VERSION")
            {
                return Err(DaemonError::incompatible(
                    "the live daemon rejected this client's private protocol",
                ));
            }
            if capabilities.iter().any(|capability| {
                capability.source == source.kind() && capability.command == command
            }) {
                Ok(Negotiated::Supported(reader.into_inner()))
            } else {
                Ok(Negotiated::Unsupported)
            }
        }
        _ => Err(DaemonError::unresponsive(
            "the live daemon returned an invalid capability response",
        )),
    }
}

fn execute_daemon_query(
    mut stream: LocalSocketStream,
    source: &dyn ProfileSource,
    context: &veloq_core::SourceQueryContext,
    invocation: QueryInvocation,
) -> Result<DaemonQueryOutcome, DaemonError> {
    let request_id = format!(
        "{:08x}-{:016x}",
        std::process::id(),
        NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed)
    );
    let mut ownership = RequestOwnership::new(request_id.clone());
    ownership.mark_transmitted();
    write_frame(
        &mut stream,
        &ClientFrame::Query {
            request_id: request_id.clone(),
            source: source.kind().to_string(),
            command: context.command.clone(),
            invocation,
        },
    )
    .map_err(|source| {
        DaemonError::execution_indeterminate(format!(
            "the daemon execution request may have been transmitted: {source}"
        ))
    })?;

    let mut reader = BufReader::new(stream);
    let first: ServerFrame = read_frame(&mut reader).map_err(|source| {
        DaemonError::execution_indeterminate(format!(
            "the daemon did not establish execution ownership: {source}"
        ))
    })?;
    ownership.observe(&first).map_err(|source| {
        DaemonError::execution_indeterminate(format!(
            "the daemon returned an invalid execution-ownership response: {source}"
        ))
    })?;
    match first {
        ServerFrame::Rejected { error, .. } => {
            if !ownership.permits_one_shot_fallback() {
                return Err(DaemonError::execution_indeterminate(
                    "the daemon rejection did not prove pre-acceptance ownership",
                ));
            }
            Ok(DaemonQueryOutcome::Rejected(error))
        }
        ServerFrame::Accepted { .. } => {
            reader.get_mut().set_recv_timeout(None).map_err(|source| {
                DaemonError::execution_indeterminate(format!(
                    "the accepted daemon request could not clear its handshake deadline: {source}"
                ))
            })?;
            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            loop {
                let response: ServerFrame = read_frame(&mut reader).map_err(|source| {
                    DaemonError::execution_indeterminate(format!(
                        "the accepted daemon request has no terminal outcome: {source}"
                    ))
                })?;
                ownership.observe(&response).map_err(|source| {
                    DaemonError::execution_indeterminate(format!(
                        "the daemon returned an invalid execution response: {source}"
                    ))
                })?;
                match response {
                    ServerFrame::OutputChunk { stream, bytes, .. } => {
                        let destination = match stream {
                            OutputStream::Stdout => &mut stdout,
                            OutputStream::Stderr => &mut stderr,
                        };
                        destination.try_reserve(bytes.len()).map_err(|source| {
                            DaemonError::execution_indeterminate(format!(
                                "the daemon query output cannot be retained by the client: {source}"
                            ))
                        })?;
                        destination.extend_from_slice(&bytes);
                    }
                    ServerFrame::Completed { exit_code, .. } => {
                        return Ok(DaemonQueryOutcome::Completed(SourceExecution::from_parts(
                            exit_code, stdout, stderr,
                        )));
                    }
                    ServerFrame::Failed { error, .. } => return Err(error),
                    _ => {
                        return Err(DaemonError::execution_indeterminate(
                            "the accepted daemon request returned a non-execution protocol frame",
                        ));
                    }
                }
            }
        }
        _ => Err(DaemonError::execution_indeterminate(
            "the daemon did not reject or accept the transmitted request",
        )),
    }
}

fn route_failure(
    source: &dyn ProfileSource,
    matches: &ArgMatches,
    fmt: OutputFormat,
    mode: RoutingMode,
    context: veloq_core::SourceQueryContext,
    error: DaemonError,
) -> SourceRunResult<SourceExecution> {
    if mode == RoutingMode::Auto
        && matches!(
            error,
            DaemonError::Absent { .. }
                | DaemonError::Unresponsive { .. }
                | DaemonError::Incompatible { .. }
                | DaemonError::Unsupported { .. }
        )
    {
        return source.execute(matches, fmt);
    }
    Ok(render_query_error(source, &context, fmt, &error))
}

fn render_query_error(
    source: &dyn ProfileSource,
    context: &veloq_core::SourceQueryContext,
    fmt: OutputFormat,
    error: &DaemonError,
) -> SourceExecution {
    let trace = context.trace_path.as_deref().map(|path| EnvelopeTraceRef {
        kind: source.kind(),
        path: path.display().to_string(),
    });
    let trace_span = context
        .trace_path
        .as_deref()
        .and_then(|path| source.compute_trace_span(path));
    let envelope = EnvelopeError::from_diagnostic(
        Some(source.source_ref()),
        Some(context.command.clone()),
        trace,
        trace_span,
        error,
    );
    let mut output = SourceExecution::new();
    output.set_exit_code(1);
    if context.raw_stdout {
        output.write_stderr_line(format!("veloq: {error}"));
        return output;
    }
    if !matches!(fmt, OutputFormat::Json) {
        output.write_stderr_line(format!("veloq: {error}"));
    }
    if let Ok(rendered) = envelope.to_json_pretty() {
        output.write_stdout_line(rendered);
    }
    output
}

fn remaining(deadline: Instant) -> Result<Duration, DaemonError> {
    deadline
        .checked_duration_since(Instant::now())
        .ok_or_else(|| {
            DaemonError::unresponsive(
                "the daemon capability exchange exceeded the connection deadline",
            )
        })
}
