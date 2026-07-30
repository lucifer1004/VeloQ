use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sysinfo::{Pid, ProcessesToUpdate, System};
use veloq_core::{
    CancellationToken, OutputFormat, ProfileSession, SourceExecution, SourceRunResult,
};

use super::config::DaemonLimits;
use super::protocol::SemanticInvocationKey;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionSpec {
    /// Axes that must match before any resident state can be reused.
    pub source_kind: String,
    pub source_version: String,
    pub trace_kind: String,
    pub canonical_trace_path: String,
    pub configuration_key: String,
    /// Opaque source-owned fingerprint covering input and artifact freshness.
    pub freshness_key: String,
    pub resident_memory_estimate_bytes: u64,
}

/// Complete semantic partition for an exact rendered response within one
/// freshness-validated session.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExactQueryKey {
    command: String,
    output_format: OutputFormat,
    invocation: SemanticInvocationKey,
}

impl ExactQueryKey {
    pub fn new(
        command: impl Into<String>,
        output_format: OutputFormat,
        invocation: SemanticInvocationKey,
    ) -> Self {
        Self {
            command: command.into(),
            output_format,
            invocation,
        }
    }

    fn retained_memory_estimate_bytes(&self) -> u64 {
        u64::try_from(std::mem::size_of::<Self>())
            .unwrap_or(u64::MAX)
            .saturating_add(u64::try_from(self.command.capacity()).unwrap_or(u64::MAX))
            .saturating_add(self.invocation.retained_heap_bytes())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryReservation {
    pub workers: u64,
    pub memory_bytes: u64,
}

impl QueryReservation {
    pub const fn new(workers: u64, memory_bytes: u64) -> Self {
        Self {
            workers,
            memory_bytes,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DaemonUsage {
    pub resident_sessions: u64,
    pub resident_memory_estimate_bytes: u64,
    pub active_requests: u64,
    pub queued_requests: u64,
    pub query_workers_reserved: u64,
    pub query_memory_reserved_bytes: u64,
    pub exact_response_entries: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub process_resident_memory_bytes: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvictionCounters {
    pub result_entries: u64,
    pub sessions: u64,
    pub freshness_invalidations: u64,
    pub idle_timeout_sessions: u64,
    pub session_limit_sessions: u64,
    pub resident_memory_sessions: u64,
    pub other_sessions: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSource {
    pub kind: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionTrace {
    pub kind: String,
    pub path: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionState {
    Idle,
    Queued,
    Active,
    Closing,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionStatus {
    pub key: String,
    pub session_id: String,
    pub source: SessionSource,
    pub trace: SessionTrace,
    pub state: SessionState,
    pub active_requests: u64,
    pub queued_requests: u64,
    pub resident_memory_estimate_bytes: u64,
    pub exact_response_entries: u64,
    pub exact_response_bytes_estimate: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub idle_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DaemonSnapshot {
    pub usage: DaemonUsage,
    pub sessions: Vec<SessionStatus>,
    pub evictions: EvictionCounters,
}

#[derive(Debug, Clone)]
pub enum AcceptOutcome {
    Cached(Arc<SourceExecution>),
    Accepted(AcceptedRequest),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionFailure {
    ResourcePressure,
    Cancelled,
    SessionInvalidated,
    ShuttingDown,
    DuplicateRequest,
    UnknownRequest,
}

#[derive(Debug, Clone)]
pub struct DaemonEngine {
    shared: Arc<Shared>,
}

#[derive(Debug)]
struct Shared {
    limits: DaemonLimits,
    state: Mutex<State>,
    changed: Condvar,
}

#[derive(Debug)]
struct State {
    next_session_id: u64,
    recency_clock: u64,
    sessions: BTreeMap<String, Session>,
    requests: HashMap<String, Request>,
    queue: VecDeque<String>,
    active_requests: u64,
    queued_requests: u64,
    query_workers_reserved: u64,
    query_memory_reserved_bytes: u64,
    cache_hits: u64,
    cache_misses: u64,
    evictions: EvictionCounters,
    shutting_down: bool,
}

#[derive(Debug)]
struct Session {
    id: String,
    configuration_key: String,
    freshness_key: String,
    source: SessionSource,
    trace: SessionTrace,
    source_resident_memory_bytes: u64,
    daemon_resident_memory_bytes: u64,
    active_requests: u64,
    queued_requests: u64,
    exact_results: HashMap<ExactQueryKey, ExactResult>,
    exact_response_bytes: u64,
    cache_hits: u64,
    cache_misses: u64,
    last_touch: Instant,
    last_touch_sequence: u64,
    closing_reason: Option<EvictionReason>,
    resident: Arc<ResidentSlot>,
}

struct ResidentSlot {
    session: Mutex<Option<Box<dyn ProfileSession>>>,
}

impl std::fmt::Debug for ResidentSlot {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let initialized = self
            .session
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_some();
        formatter
            .debug_struct("ResidentSlot")
            .field("initialized", &initialized)
            .finish()
    }
}

#[derive(Debug)]
struct ExactResult {
    execution: Arc<SourceExecution>,
    accounted_bytes: u64,
    last_touch_sequence: u64,
}

#[derive(Debug)]
struct Request {
    session_id: Option<String>,
    reservation: QueryReservation,
    status: RequestStatus,
    admission_deadline: Instant,
    cancellation: CancellationToken,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestStatus {
    Queued,
    Active,
    Cancelled,
    Expired,
    SessionInvalidated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvictionReason {
    Freshness,
    IdleTimeout,
    SessionLimit,
    ResidentMemory,
    ExecutionFailure,
}

#[derive(Debug, Clone)]
pub struct AcceptedRequest {
    engine: DaemonEngine,
    request_id: String,
    session_id: Option<String>,
}

#[derive(Debug)]
pub struct ActiveRequest {
    engine: DaemonEngine,
    request_id: String,
    session_id: Option<String>,
    cancellation: CancellationToken,
    finished: bool,
}

impl DaemonEngine {
    pub fn new(limits: DaemonLimits) -> Self {
        Self {
            shared: Arc::new(Shared {
                limits,
                state: Mutex::new(State {
                    next_session_id: 1,
                    recency_clock: 0,
                    sessions: BTreeMap::new(),
                    requests: HashMap::new(),
                    queue: VecDeque::new(),
                    active_requests: 0,
                    queued_requests: 0,
                    query_workers_reserved: 0,
                    query_memory_reserved_bytes: 0,
                    cache_hits: 0,
                    cache_misses: 0,
                    evictions: EvictionCounters::default(),
                    shutting_down: false,
                }),
                changed: Condvar::new(),
            }),
        }
    }

    pub fn limits(&self) -> &DaemonLimits {
        &self.shared.limits
    }

    pub fn accept<S>(
        &self,
        request_id: impl Into<String>,
        spec: S,
        reservation: QueryReservation,
        exact_cache_key: Option<&ExactQueryKey>,
    ) -> Result<AcceptOutcome, AdmissionFailure>
    where
        S: Into<Option<SessionSpec>>,
    {
        let spec = spec.into();
        let request_id = request_id.into();
        let now = Instant::now();
        let mut state = self.lock_state();
        if state.shutting_down {
            return Err(AdmissionFailure::ShuttingDown);
        }
        if state.requests.contains_key(&request_id) {
            return Err(AdmissionFailure::DuplicateRequest);
        }
        if reservation.workers > self.shared.limits.max_query_workers
            || self
                .shared
                .limits
                .max_query_memory_bytes
                .is_some_and(|limit| reservation.memory_bytes > limit)
        {
            return Err(AdmissionFailure::ResourcePressure);
        }

        purge_idle_sessions(&mut state, &self.shared.limits, now);
        let reusable = spec
            .as_ref()
            .and_then(|spec| reusable_session_id(&state, spec));
        let mut exact_cache_checked = false;
        if let (Some(session_id), Some(cache_key)) = (reusable.as_deref(), exact_cache_key) {
            exact_cache_checked = true;
            if let Some(execution) = lookup_exact(&mut state, session_id, cache_key, now) {
                return Ok(AcceptOutcome::Cached(execution));
            }
        }

        let global_resources_available =
            resources_available(&state, &self.shared.limits, reservation);
        let reusable_session_available = reusable
            .as_deref()
            .is_none_or(|session_id| session_available(&state, session_id));
        if !(global_resources_available && reusable_session_available)
            && (self.shared.limits.max_queued_requests == 0
                || self.shared.limits.admission_timeout_ms == 0
                || state.queued_requests >= self.shared.limits.max_queued_requests)
        {
            return Err(AdmissionFailure::ResourcePressure);
        }

        if let Some(spec) = spec.as_ref() {
            invalidate_stale_sessions(&mut state, spec);
            self.shared.changed.notify_all();
        }
        let session_id = match (reusable, spec.as_ref()) {
            (Some(session_id), _) => Some(session_id),
            (None, Some(spec)) => Some(admit_session(&mut state, &self.shared.limits, spec, now)?),
            (None, None) => None,
        };

        if !exact_cache_checked
            && let Some(cache_key) = exact_cache_key
            && let Some(session_id) = session_id.as_deref()
            && let Some(execution) = lookup_exact(&mut state, session_id, cache_key, now)
        {
            return Ok(AcceptOutcome::Cached(execution));
        }

        let can_activate = global_resources_available
            && session_id
                .as_deref()
                .is_none_or(|session_id| session_available(&state, session_id));
        let status = if can_activate {
            reserve_active(&mut state, session_id.as_deref(), reservation);
            RequestStatus::Active
        } else {
            state.queued_requests = state.queued_requests.saturating_add(1);
            if let Some(session_id) = session_id.as_deref()
                && let Some(session) = state.sessions.get_mut(session_id)
            {
                session.queued_requests = session.queued_requests.saturating_add(1);
            }
            state.queue.push_back(request_id.clone());
            RequestStatus::Queued
        };
        let _ = state.requests.insert(
            request_id.clone(),
            Request {
                session_id: session_id.clone(),
                reservation,
                status,
                admission_deadline: now
                    + Duration::from_millis(self.shared.limits.admission_timeout_ms),
                cancellation: CancellationToken::new(),
            },
        );
        Ok(AcceptOutcome::Accepted(AcceptedRequest {
            engine: self.clone(),
            request_id,
            session_id,
        }))
    }

    pub fn cancel(&self, request_id: &str) -> Result<(), AdmissionFailure> {
        let mut state = self.lock_state();
        let Some(request) = state.requests.get(request_id) else {
            return Err(AdmissionFailure::UnknownRequest);
        };
        let status = request.status;
        let session_id = request.session_id.clone();
        let cancellation = request.cancellation.clone();
        match status {
            RequestStatus::Queued => {
                remove_from_queue(&mut state.queue, request_id);
                state.queued_requests = state.queued_requests.saturating_sub(1);
                if let Some(session_id) = session_id.as_deref()
                    && let Some(session) = state.sessions.get_mut(session_id)
                {
                    session.queued_requests = session.queued_requests.saturating_sub(1);
                }
                if let Some(request) = state.requests.get_mut(request_id) {
                    request.status = RequestStatus::Cancelled;
                }
                if let Some(session_id) = session_id.as_deref() {
                    remove_closing_session_if_unused(&mut state, session_id);
                }
                promote_queued(&mut state, &self.shared.limits);
            }
            RequestStatus::Active => {}
            RequestStatus::Cancelled
            | RequestStatus::Expired
            | RequestStatus::SessionInvalidated => {}
        }
        self.shared.changed.notify_all();
        drop(state);
        cancellation.cancel();
        Ok(())
    }

    pub fn begin_shutdown(&self) {
        let mut state = self.lock_state();
        state.shutting_down = true;
        let queued: Vec<String> = state.queue.drain(..).collect();
        for request_id in queued {
            let Some(request) = state.requests.get(&request_id) else {
                continue;
            };
            let session_id = request.session_id.clone();
            state.queued_requests = state.queued_requests.saturating_sub(1);
            if let Some(session_id) = session_id.as_deref()
                && let Some(session) = state.sessions.get_mut(session_id)
            {
                session.queued_requests = session.queued_requests.saturating_sub(1);
            }
            if let Some(request) = state.requests.get_mut(&request_id) {
                request.status = RequestStatus::Cancelled;
                request.cancellation.cancel();
            }
        }
        self.shared.changed.notify_all();
    }

    pub fn wait_for_active_drain(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut state = self.lock_state();
        while state.active_requests > 0 {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return false;
            }
            let (next, wait) = self
                .shared
                .changed
                .wait_timeout(state, remaining)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state = next;
            if wait.timed_out() && state.active_requests > 0 {
                return false;
            }
        }
        true
    }

    pub fn cancel_active_requests(&self) {
        let state = self.lock_state();
        let cancellations = state
            .requests
            .values()
            .filter(|request| request.status == RequestStatus::Active)
            .map(|request| request.cancellation.clone())
            .collect::<Vec<_>>();
        self.shared.changed.notify_all();
        drop(state);
        for cancellation in cancellations {
            cancellation.cancel();
        }
    }

    pub fn snapshot(&self) -> DaemonSnapshot {
        let state = self.lock_state();
        snapshot(&state)
    }

    fn lock_state(&self) -> MutexGuard<'_, State> {
        self.shared
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl AcceptedRequest {
    #[cfg(test)]
    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    pub fn wait_until_active(self) -> Result<ActiveRequest, AdmissionFailure> {
        let mut state = self.engine.lock_state();
        loop {
            let Some(request) = state.requests.get(&self.request_id) else {
                return Err(AdmissionFailure::UnknownRequest);
            };
            match request.status {
                RequestStatus::Active => {
                    let cancellation = request.cancellation.clone();
                    return Ok(ActiveRequest {
                        engine: self.engine.clone(),
                        request_id: self.request_id,
                        session_id: self.session_id,
                        cancellation,
                        finished: false,
                    });
                }
                RequestStatus::Cancelled => {
                    state.requests.remove(&self.request_id);
                    return Err(AdmissionFailure::Cancelled);
                }
                RequestStatus::Expired => {
                    state.requests.remove(&self.request_id);
                    return Err(AdmissionFailure::ResourcePressure);
                }
                RequestStatus::SessionInvalidated => {
                    state.requests.remove(&self.request_id);
                    return Err(AdmissionFailure::SessionInvalidated);
                }
                RequestStatus::Queued => {
                    let deadline = request.admission_deadline;
                    let now = Instant::now();
                    if now >= deadline {
                        expire_queued_request(&mut state, &self.request_id);
                        state.requests.remove(&self.request_id);
                        promote_queued(&mut state, &self.engine.shared.limits);
                        self.engine.shared.changed.notify_all();
                        return Err(AdmissionFailure::ResourcePressure);
                    }
                    let timeout = deadline.saturating_duration_since(now);
                    let (next, _) = self
                        .engine
                        .shared
                        .changed
                        .wait_timeout(state, timeout)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    state = next;
                }
            }
        }
    }
}

impl ActiveRequest {
    #[cfg(test)]
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    pub fn cancellation_requested(&self) -> bool {
        self.cancellation.is_cancelled()
    }

    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    pub fn execute_with_resident(
        &self,
        initialize: impl FnOnce() -> SourceRunResult<Option<Box<dyn ProfileSession>>>,
        execute: impl FnOnce(&mut dyn ProfileSession) -> SourceRunResult<SourceExecution>,
    ) -> SourceRunResult<Option<(SourceExecution, u64)>> {
        let resident = {
            let state = self.engine.lock_state();
            self.session_id
                .as_deref()
                .and_then(|session_id| state.sessions.get(session_id))
                .map(|session| Arc::clone(&session.resident))
        };
        let Some(resident) = resident else {
            return Ok(None);
        };
        let mut session = resident
            .session
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if session.is_none() {
            *session = initialize()?;
        }
        match session.as_mut() {
            Some(session) => {
                let execution = execute(session.as_mut())?;
                let additional_bytes = session.additional_resident_memory_estimate_bytes();
                Ok(Some((execution, additional_bytes)))
            }
            None => Ok(None),
        }
    }

    /// Reconcile source-owned retained-state accounting after execution.
    ///
    /// A changed fingerprint closes the session instead of assigning the new
    /// fingerprint to a connection opened against older evidence.
    pub fn refresh_resident_state(
        &self,
        admitted_freshness_key: &str,
        observed_freshness_key: Option<&str>,
        resident_memory_estimate_bytes: u64,
    ) -> bool {
        let mut state = self.engine.lock_state();
        let Some(session_id) = self.session_id.as_deref() else {
            return false;
        };
        let freshness_unchanged = observed_freshness_key == Some(admitted_freshness_key);
        if !freshness_unchanged {
            close_session(&mut state, session_id, EvictionReason::Freshness);
            return false;
        }
        let Some(previous_bytes) = state
            .sessions
            .get(session_id)
            .map(|session| session.source_resident_memory_bytes)
        else {
            return false;
        };
        let additional = resident_memory_estimate_bytes.saturating_sub(previous_bytes);
        if additional > 0 {
            evict_results_until_memory_fits(&mut state, &self.engine.shared.limits, additional);
            evict_sessions_until_memory_fits(
                &mut state,
                &self.engine.shared.limits,
                additional,
                Some(session_id),
            );
        }
        if resident_memory(&state)
            .saturating_sub(previous_bytes)
            .saturating_add(resident_memory_estimate_bytes)
            > self.engine.shared.limits.max_resident_memory_bytes
        {
            close_session(&mut state, session_id, EvictionReason::ResidentMemory);
            return false;
        }
        if let Some(session) = state.sessions.get_mut(session_id) {
            session.source_resident_memory_bytes = resident_memory_estimate_bytes;
            true
        } else {
            false
        }
    }

    pub fn discard_resident_state_after_failure(&self) {
        let Some(session_id) = self.session_id.as_deref() else {
            return;
        };
        let mut state = self.engine.lock_state();
        close_session(&mut state, session_id, EvictionReason::ExecutionFailure);
    }

    pub fn complete(
        mut self,
        exact_cache_key: Option<ExactQueryKey>,
        execution: Option<&SourceExecution>,
    ) {
        let cached_execution = execution
            .filter(|execution| execution.exit_code() == 0)
            .filter(|_| !self.cancellation.is_cancelled())
            .cloned()
            .map(Arc::new);
        let mut state = self.engine.lock_state();
        if let (Some(session_id), Some(cache_key), Some(execution)) = (
            self.session_id.as_deref(),
            exact_cache_key,
            cached_execution,
        ) {
            insert_exact(
                &mut state,
                &self.engine.shared.limits,
                session_id,
                cache_key,
                execution,
            );
        }
        finish_active_request(&mut state, &self.engine.shared.limits, &self.request_id);
        self.finished = true;
        self.engine.shared.changed.notify_all();
    }
}

impl Drop for ActiveRequest {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let mut state = self.engine.lock_state();
        finish_active_request(&mut state, &self.engine.shared.limits, &self.request_id);
        self.engine.shared.changed.notify_all();
    }
}

fn reusable_session_id(state: &State, spec: &SessionSpec) -> Option<String> {
    state.sessions.values().find_map(|session| {
        (same_reuse_domain(session, spec)
            && session.freshness_key == spec.freshness_key
            && session.closing_reason.is_none())
        .then(|| session.id.clone())
    })
}

fn same_reuse_domain(session: &Session, spec: &SessionSpec) -> bool {
    session.source.kind == spec.source_kind
        && session.source.version == spec.source_version
        && session.trace.kind == spec.trace_kind
        && session.trace.path == spec.canonical_trace_path
        && session.configuration_key == spec.configuration_key
}

fn admit_session(
    state: &mut State,
    limits: &DaemonLimits,
    spec: &SessionSpec,
    now: Instant,
) -> Result<String, AdmissionFailure> {
    let session_id = format!("s{:016x}", state.next_session_id);
    let mut session = Session {
        id: session_id.clone(),
        configuration_key: spec.configuration_key.clone(),
        freshness_key: spec.freshness_key.clone(),
        source: SessionSource {
            kind: spec.source_kind.clone(),
            version: spec.source_version.clone(),
        },
        trace: SessionTrace {
            kind: spec.trace_kind.clone(),
            path: spec.canonical_trace_path.clone(),
        },
        source_resident_memory_bytes: spec.resident_memory_estimate_bytes,
        daemon_resident_memory_bytes: 0,
        active_requests: 0,
        queued_requests: 0,
        exact_results: HashMap::new(),
        exact_response_bytes: 0,
        cache_hits: 0,
        cache_misses: 0,
        last_touch: now,
        last_touch_sequence: 0,
        closing_reason: None,
        resident: Arc::new(ResidentSlot {
            session: Mutex::new(None),
        }),
    };
    session.daemon_resident_memory_bytes =
        daemon_session_memory_estimate(session_id.capacity(), &session);
    let accounted_bytes = session_memory(&session);
    if accounted_bytes > limits.max_resident_memory_bytes {
        return Err(AdmissionFailure::ResourcePressure);
    }
    make_room_for_session(
        state,
        limits,
        accounted_bytes,
        state.sessions.len() as u64 >= limits.max_sessions,
    );
    if state.sessions.len() as u64 >= limits.max_sessions
        || resident_memory(state).saturating_add(accounted_bytes) > limits.max_resident_memory_bytes
    {
        return Err(AdmissionFailure::ResourcePressure);
    }

    state.next_session_id = state.next_session_id.saturating_add(1);
    session.last_touch_sequence = tick(state);
    state.sessions.insert(session_id.clone(), session);
    Ok(session_id)
}

fn lookup_exact(
    state: &mut State,
    session_id: &str,
    cache_key: &ExactQueryKey,
    now: Instant,
) -> Option<Arc<SourceExecution>> {
    let sequence = tick(state);
    let session = state.sessions.get_mut(session_id)?;
    if session.active_requests > 0 {
        return None;
    }
    session.last_touch = now;
    session.last_touch_sequence = sequence;
    if let Some(entry) = session.exact_results.get_mut(cache_key) {
        entry.last_touch_sequence = sequence;
        session.cache_hits = session.cache_hits.saturating_add(1);
        state.cache_hits = state.cache_hits.saturating_add(1);
        return Some(Arc::clone(&entry.execution));
    }
    session.cache_misses = session.cache_misses.saturating_add(1);
    state.cache_misses = state.cache_misses.saturating_add(1);
    None
}

fn insert_exact(
    state: &mut State,
    limits: &DaemonLimits,
    session_id: &str,
    cache_key: ExactQueryKey,
    execution: Arc<SourceExecution>,
) {
    let accounted_bytes = u64::try_from(std::mem::size_of::<ExactResult>())
        .unwrap_or(u64::MAX)
        .saturating_add(cache_key.retained_memory_estimate_bytes())
        .saturating_add(execution.retained_memory_estimate_bytes());
    if accounted_bytes > limits.max_resident_memory_bytes {
        return;
    }

    if let Some(session) = state.sessions.get_mut(session_id)
        && let Some(previous) = session.exact_results.remove(&cache_key)
    {
        session.exact_response_bytes = session
            .exact_response_bytes
            .saturating_sub(previous.accounted_bytes);
    }
    evict_results_until_memory_fits(state, limits, accounted_bytes);
    evict_sessions_until_memory_fits(state, limits, accounted_bytes, Some(session_id));
    if resident_memory(state).saturating_add(accounted_bytes) > limits.max_resident_memory_bytes {
        return;
    }

    let sequence = tick(state);
    if let Some(session) = state.sessions.get_mut(session_id) {
        session.exact_response_bytes = session.exact_response_bytes.saturating_add(accounted_bytes);
        session.exact_results.insert(
            cache_key,
            ExactResult {
                execution,
                accounted_bytes,
                last_touch_sequence: sequence,
            },
        );
    }
}

fn resources_available(
    state: &State,
    limits: &DaemonLimits,
    reservation: QueryReservation,
) -> bool {
    state.active_requests < limits.max_concurrent_requests
        && state
            .query_workers_reserved
            .saturating_add(reservation.workers)
            <= limits.max_query_workers
        && limits.max_query_memory_bytes.is_none_or(|limit| {
            state
                .query_memory_reserved_bytes
                .saturating_add(reservation.memory_bytes)
                <= limit
        })
}

fn session_available(state: &State, session_id: &str) -> bool {
    state
        .sessions
        .get(session_id)
        .is_some_and(|session| session.active_requests == 0 && session.closing_reason.is_none())
}

fn reserve_active(state: &mut State, session_id: Option<&str>, reservation: QueryReservation) {
    state.active_requests = state.active_requests.saturating_add(1);
    state.query_workers_reserved = state
        .query_workers_reserved
        .saturating_add(reservation.workers);
    state.query_memory_reserved_bytes = state
        .query_memory_reserved_bytes
        .saturating_add(reservation.memory_bytes);
    let sequence = tick(state);
    if let Some(session_id) = session_id
        && let Some(session) = state.sessions.get_mut(session_id)
    {
        session.active_requests = session.active_requests.saturating_add(1);
        session.last_touch = Instant::now();
        session.last_touch_sequence = sequence;
    }
}

fn promote_queued(state: &mut State, limits: &DaemonLimits) {
    loop {
        let now = Instant::now();
        let queued = state.queue.iter().cloned().collect::<Vec<_>>();
        for request_id in &queued {
            let expired = state.requests.get(request_id).is_some_and(|request| {
                request.status == RequestStatus::Queued && now >= request.admission_deadline
            });
            if expired {
                expire_queued_request(state, request_id);
            } else if !state.requests.contains_key(request_id)
                || state
                    .requests
                    .get(request_id)
                    .is_some_and(|request| request.status != RequestStatus::Queued)
            {
                remove_from_queue(&mut state.queue, request_id);
            }
        }
        let candidate = state.queue.iter().position(|request_id| {
            state.requests.get(request_id).is_some_and(|request| {
                resources_available(state, limits, request.reservation)
                    && request
                        .session_id
                        .as_deref()
                        .is_none_or(|session_id| session_available(state, session_id))
            })
        });
        let Some(candidate) = candidate else {
            break;
        };
        let Some(request_id) = state.queue.remove(candidate) else {
            break;
        };
        let Some(request) = state.requests.get_mut(&request_id) else {
            continue;
        };
        request.status = RequestStatus::Active;
        let session_id = request.session_id.clone();
        let reservation = request.reservation;
        state.queued_requests = state.queued_requests.saturating_sub(1);
        if let Some(session_id) = session_id.as_deref()
            && let Some(session) = state.sessions.get_mut(session_id)
        {
            session.queued_requests = session.queued_requests.saturating_sub(1);
        }
        reserve_active(state, session_id.as_deref(), reservation);
    }
}

fn finish_active_request(state: &mut State, limits: &DaemonLimits, request_id: &str) {
    let Some(request) = state.requests.remove(request_id) else {
        return;
    };
    if request.status != RequestStatus::Active {
        return;
    }
    state.active_requests = state.active_requests.saturating_sub(1);
    state.query_workers_reserved = state
        .query_workers_reserved
        .saturating_sub(request.reservation.workers);
    state.query_memory_reserved_bytes = state
        .query_memory_reserved_bytes
        .saturating_sub(request.reservation.memory_bytes);
    let sequence = tick(state);
    if let Some(session_id) = request.session_id.as_deref() {
        if let Some(session) = state.sessions.get_mut(session_id) {
            session.active_requests = session.active_requests.saturating_sub(1);
            session.last_touch = Instant::now();
            session.last_touch_sequence = sequence;
        }
        remove_closing_session_if_unused(state, session_id);
    }
    promote_queued(state, limits);
}

fn expire_queued_request(state: &mut State, request_id: &str) {
    let Some(request) = state.requests.get(request_id) else {
        return;
    };
    if request.status != RequestStatus::Queued {
        return;
    }
    let session_id = request.session_id.clone();
    remove_from_queue(&mut state.queue, request_id);
    state.queued_requests = state.queued_requests.saturating_sub(1);
    if let Some(session_id) = session_id.as_deref()
        && let Some(session) = state.sessions.get_mut(session_id)
    {
        session.queued_requests = session.queued_requests.saturating_sub(1);
    }
    if let Some(request) = state.requests.get_mut(request_id) {
        request.status = RequestStatus::Expired;
    }
    if let Some(session_id) = session_id.as_deref() {
        remove_closing_session_if_unused(state, session_id);
    }
}

fn remove_from_queue(queue: &mut VecDeque<String>, request_id: &str) {
    if let Some(index) = queue.iter().position(|queued| queued == request_id) {
        queue.remove(index);
    }
}

fn invalidate_stale_sessions(state: &mut State, spec: &SessionSpec) {
    let stale: Vec<String> = state
        .sessions
        .values()
        .filter(|session| {
            same_reuse_domain(session, spec) && session.freshness_key != spec.freshness_key
        })
        .map(|session| session.id.clone())
        .collect();
    for session_id in stale {
        close_session(state, &session_id, EvictionReason::Freshness);
    }
}

fn close_session(state: &mut State, session_id: &str, reason: EvictionReason) {
    let queued = state
        .queue
        .iter()
        .filter(|request_id| {
            state.requests.get(*request_id).is_some_and(|request| {
                request.status == RequestStatus::Queued
                    && request.session_id.as_deref() == Some(session_id)
            })
        })
        .cloned()
        .collect::<Vec<_>>();
    for request_id in queued {
        remove_from_queue(&mut state.queue, &request_id);
        state.queued_requests = state.queued_requests.saturating_sub(1);
        if let Some(request) = state.requests.get_mut(&request_id) {
            request.status = RequestStatus::SessionInvalidated;
        }
    }
    if let Some(session) = state.sessions.get_mut(session_id) {
        session.queued_requests = 0;
        session.closing_reason = Some(reason);
    }
    remove_closing_session_if_unused(state, session_id);
}

fn purge_idle_sessions(state: &mut State, limits: &DaemonLimits, now: Instant) {
    if limits.idle_timeout_ms == 0 {
        return;
    }
    let timeout = Duration::from_millis(limits.idle_timeout_ms);
    let expired: Vec<String> = state
        .sessions
        .values()
        .filter(|session| {
            session.active_requests == 0
                && session.queued_requests == 0
                && now.saturating_duration_since(session.last_touch) >= timeout
        })
        .map(|session| session.id.clone())
        .collect();
    for session_id in expired {
        remove_session(state, &session_id, EvictionReason::IdleTimeout);
    }
}

fn make_room_for_session(
    state: &mut State,
    limits: &DaemonLimits,
    incoming_bytes: u64,
    session_count_full: bool,
) {
    if resident_memory(state).saturating_add(incoming_bytes) > limits.max_resident_memory_bytes {
        evict_results_until_memory_fits(state, limits, incoming_bytes);
    }
    while session_count_full_or_memory_exceeded(state, limits, incoming_bytes, session_count_full) {
        let reason = if state.sessions.len() as u64 >= limits.max_sessions {
            EvictionReason::SessionLimit
        } else {
            EvictionReason::ResidentMemory
        };
        let Some(candidate) = weighted_session_candidate(state) else {
            break;
        };
        remove_session(state, &candidate, reason);
    }
}

fn session_count_full_or_memory_exceeded(
    state: &State,
    limits: &DaemonLimits,
    incoming_bytes: u64,
    session_count_was_full: bool,
) -> bool {
    (session_count_was_full && state.sessions.len() as u64 >= limits.max_sessions)
        || resident_memory(state).saturating_add(incoming_bytes) > limits.max_resident_memory_bytes
}

fn evict_results_until_memory_fits(state: &mut State, limits: &DaemonLimits, incoming_bytes: u64) {
    while resident_memory(state).saturating_add(incoming_bytes) > limits.max_resident_memory_bytes {
        let candidate = state
            .sessions
            .iter()
            .flat_map(|(session_id, session)| {
                session.exact_results.iter().map(move |(cache_key, entry)| {
                    (
                        entry.last_touch_sequence,
                        session_id.clone(),
                        cache_key.clone(),
                    )
                })
            })
            .min();
        let Some((_, session_id, cache_key)) = candidate else {
            break;
        };
        remove_exact_result(state, &session_id, &cache_key);
    }
}

fn evict_sessions_until_memory_fits(
    state: &mut State,
    limits: &DaemonLimits,
    incoming_bytes: u64,
    protected_session_id: Option<&str>,
) {
    while resident_memory(state).saturating_add(incoming_bytes) > limits.max_resident_memory_bytes {
        let Some(candidate) = weighted_session_candidate_excluding(state, protected_session_id)
        else {
            break;
        };
        remove_session(state, &candidate, EvictionReason::ResidentMemory);
    }
}

fn weighted_session_candidate(state: &State) -> Option<String> {
    weighted_session_candidate_excluding(state, None)
}

fn weighted_session_candidate_excluding(state: &State, excluded: Option<&str>) -> Option<String> {
    state
        .sessions
        .values()
        .filter(|session| {
            session.active_requests == 0
                && session.queued_requests == 0
                && excluded != Some(session.id.as_str())
        })
        .max_by_key(|session| {
            // Larger age and retained cost both increase eviction priority.
            // The product protects a recently touched large session while
            // preventing it from dominating after it cools.
            let age = state
                .recency_clock
                .saturating_sub(session.last_touch_sequence)
                .saturating_add(1);
            let cost = session_memory(session).max(1);
            (
                session.closing_reason.is_some(),
                age.saturating_mul(cost),
                age,
                cost,
                std::cmp::Reverse(session.id.clone()),
            )
        })
        .map(|session| session.id.clone())
}

fn remove_exact_result(state: &mut State, session_id: &str, cache_key: &ExactQueryKey) {
    let Some(session) = state.sessions.get_mut(session_id) else {
        return;
    };
    let Some(entry) = session.exact_results.remove(cache_key) else {
        return;
    };
    session.exact_response_bytes = session
        .exact_response_bytes
        .saturating_sub(entry.accounted_bytes);
    state.evictions.result_entries = state.evictions.result_entries.saturating_add(1);
}

fn remove_session(state: &mut State, session_id: &str, reason: EvictionReason) {
    let Some(session) = state.sessions.remove(session_id) else {
        return;
    };
    state.evictions.result_entries = state
        .evictions
        .result_entries
        .saturating_add(session.exact_results.len() as u64);
    state.evictions.sessions = state.evictions.sessions.saturating_add(1);
    match reason {
        EvictionReason::Freshness => {
            state.evictions.freshness_invalidations =
                state.evictions.freshness_invalidations.saturating_add(1);
        }
        EvictionReason::IdleTimeout => {
            state.evictions.idle_timeout_sessions =
                state.evictions.idle_timeout_sessions.saturating_add(1);
        }
        EvictionReason::SessionLimit => {
            state.evictions.session_limit_sessions =
                state.evictions.session_limit_sessions.saturating_add(1);
        }
        EvictionReason::ResidentMemory => {
            state.evictions.resident_memory_sessions =
                state.evictions.resident_memory_sessions.saturating_add(1);
        }
        EvictionReason::ExecutionFailure => {
            state.evictions.other_sessions = state.evictions.other_sessions.saturating_add(1);
        }
    }
}

fn remove_closing_session_if_unused(state: &mut State, session_id: &str) {
    let reason = state.sessions.get(session_id).and_then(|session| {
        (session.active_requests == 0 && session.queued_requests == 0)
            .then_some(session.closing_reason)
            .flatten()
    });
    if let Some(reason) = reason {
        remove_session(state, session_id, reason);
    }
}

fn resident_memory(state: &State) -> u64 {
    state.sessions.values().fold(0u64, |total, session| {
        total.saturating_add(session_memory(session))
    })
}

fn session_memory(session: &Session) -> u64 {
    session
        .source_resident_memory_bytes
        .saturating_add(session.daemon_resident_memory_bytes)
        .saturating_add(session.exact_response_bytes)
}

fn daemon_session_memory_estimate(map_key_capacity: usize, session: &Session) -> u64 {
    u64::try_from(std::mem::size_of::<Session>())
        .unwrap_or(u64::MAX)
        .saturating_add(u64::try_from(std::mem::size_of::<String>()).unwrap_or(u64::MAX))
        .saturating_add(u64::try_from(map_key_capacity).unwrap_or(u64::MAX))
        .saturating_add(u64::try_from(session.id.capacity()).unwrap_or(u64::MAX))
        .saturating_add(u64::try_from(session.configuration_key.capacity()).unwrap_or(u64::MAX))
        .saturating_add(u64::try_from(session.freshness_key.capacity()).unwrap_or(u64::MAX))
        .saturating_add(u64::try_from(session.source.kind.capacity()).unwrap_or(u64::MAX))
        .saturating_add(u64::try_from(session.source.version.capacity()).unwrap_or(u64::MAX))
        .saturating_add(u64::try_from(session.trace.kind.capacity()).unwrap_or(u64::MAX))
        .saturating_add(u64::try_from(session.trace.path.capacity()).unwrap_or(u64::MAX))
        .saturating_add(u64::try_from(std::mem::size_of::<ResidentSlot>()).unwrap_or(u64::MAX))
}

fn tick(state: &mut State) -> u64 {
    state.recency_clock = state.recency_clock.saturating_add(1);
    state.recency_clock
}

fn snapshot(state: &State) -> DaemonSnapshot {
    let now = Instant::now();
    let sessions = state
        .sessions
        .values()
        .map(|session| {
            let state = if session.closing_reason.is_some() {
                SessionState::Closing
            } else if session.active_requests > 0 {
                SessionState::Active
            } else if session.queued_requests > 0 {
                SessionState::Queued
            } else {
                SessionState::Idle
            };
            SessionStatus {
                key: format!("daemon-session|{}", session.id),
                session_id: session.id.clone(),
                source: session.source.clone(),
                trace: session.trace.clone(),
                state,
                active_requests: session.active_requests,
                queued_requests: session.queued_requests,
                resident_memory_estimate_bytes: session_memory(session),
                exact_response_entries: session.exact_results.len() as u64,
                exact_response_bytes_estimate: session.exact_response_bytes,
                cache_hits: session.cache_hits,
                cache_misses: session.cache_misses,
                idle_ms: (state == SessionState::Idle).then(|| {
                    now.saturating_duration_since(session.last_touch)
                        .as_millis()
                        .try_into()
                        .unwrap_or(u64::MAX)
                }),
            }
        })
        .collect();
    DaemonSnapshot {
        usage: DaemonUsage {
            resident_sessions: state.sessions.len() as u64,
            resident_memory_estimate_bytes: resident_memory(state),
            active_requests: state.active_requests,
            queued_requests: state.queued_requests,
            query_workers_reserved: state.query_workers_reserved,
            query_memory_reserved_bytes: state.query_memory_reserved_bytes,
            exact_response_entries: state
                .sessions
                .values()
                .map(|session| session.exact_results.len() as u64)
                .sum(),
            cache_hits: state.cache_hits,
            cache_misses: state.cache_misses,
            process_resident_memory_bytes: process_resident_memory(),
        },
        sessions,
        evictions: state.evictions.clone(),
    }
}

fn process_resident_memory() -> Option<u64> {
    let pid = Pid::from_u32(std::process::id());
    let mut system = System::new();
    system.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
    system.process(pid).map(|process| process.memory())
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::panic,
    reason = "test failures need concise setup diagnostics"
)]
mod tests {
    use super::*;
    use crate::daemon::config::MIB_BYTES;
    use crate::daemon::protocol::{EncodedOsString, QueryInvocation};
    use std::ffi::OsStr;

    fn limits() -> DaemonLimits {
        DaemonLimits {
            max_sessions: 4,
            max_resident_memory_bytes: 4_096,
            max_concurrent_requests: 2,
            max_query_workers: 2,
            max_query_memory_bytes: Some(2_048),
            max_queued_requests: 2,
            admission_timeout_ms: 5_000,
            idle_timeout_ms: 0,
            shutdown_grace_ms: 100,
        }
    }

    fn spec(trace: &str, freshness: &str, bytes: u64) -> SessionSpec {
        SessionSpec {
            source_kind: "nsys".to_string(),
            source_version: "v4".to_string(),
            trace_kind: "nsys".to_string(),
            canonical_trace_path: format!("/profiles/{trace}"),
            configuration_key: "default-query-config".to_string(),
            freshness_key: freshness.to_string(),
            resident_memory_estimate_bytes: bytes,
        }
    }

    fn exact_key(command: &str) -> ExactQueryKey {
        exact_key_with(command, OutputFormat::Json, "default")
    }

    fn exact_key_with(command: &str, output_format: OutputFormat, context: &str) -> ExactQueryKey {
        let invocation = QueryInvocation {
            arguments: ["veloq", context]
                .into_iter()
                .map(|value| EncodedOsString::encode(OsStr::new(value)))
                .collect(),
            cwd: EncodedOsString::encode(OsStr::new(context)),
            environment: Vec::new(),
            terminal_width: Some(80),
        };
        ExactQueryKey::new(
            command,
            output_format,
            invocation.semantic_key(None, output_format),
        )
    }

    fn accepted(outcome: AcceptOutcome) -> AcceptedRequest {
        match outcome {
            AcceptOutcome::Accepted(request) => request,
            AcceptOutcome::Cached(_) => panic!("request unexpectedly hit the exact cache"),
        }
    }

    fn execution(body: &str) -> SourceExecution {
        let mut execution = SourceExecution::new();
        execution.write_stdout(body.as_bytes());
        execution
    }

    #[test]
    fn freshness_change_discards_cached_results_before_reuse() {
        let engine = DaemonEngine::new(limits());
        let request = accepted(
            engine
                .accept(
                    "build-v1",
                    spec("trace", "input-v1+artifacts-v1", 100),
                    QueryReservation::new(1, 128),
                    Some(&exact_key("summary")),
                )
                .expect("accept initial request"),
        );
        let active = request
            .wait_until_active()
            .expect("activate initial request");
        let result = execution("version one");
        active.complete(Some(exact_key("summary")), Some(&result));

        let cached = engine
            .accept(
                "hit-v1",
                spec("trace", "input-v1+artifacts-v1", 100),
                QueryReservation::new(1, 128),
                Some(&exact_key("summary")),
            )
            .expect("look up cached result");
        match cached {
            AcceptOutcome::Cached(hit) => assert_eq!(hit.stdout(), b"version one"),
            AcceptOutcome::Accepted(_) => panic!("unchanged freshness must reuse the exact result"),
        }

        let request = accepted(
            engine
                .accept(
                    "build-v2",
                    spec("trace", "input-v2+artifacts-v2", 100),
                    QueryReservation::new(1, 128),
                    Some(&exact_key("summary")),
                )
                .expect("accept fresh replacement"),
        );
        assert_ne!(request.session_id(), Some("s0000000000000001"));
        let snapshot = engine.snapshot();
        assert_eq!(snapshot.evictions.freshness_invalidations, 1);
        assert_eq!(snapshot.usage.exact_response_entries, 0);
    }

    #[test]
    fn pressure_discards_results_before_weighted_idle_sessions() {
        let mut configured = limits();
        configured.max_sessions = 2;
        configured.max_resident_memory_bytes = 6 * MIB_BYTES;
        let engine = DaemonEngine::new(configured);
        let source_bytes = MIB_BYTES + MIB_BYTES / 2;

        let first = accepted(
            engine
                .accept(
                    "first",
                    spec("first", "fresh", source_bytes),
                    QueryReservation::new(1, 128),
                    None,
                )
                .expect("accept first session"),
        )
        .wait_until_active()
        .expect("activate first session");
        let cached = execution(
            &"x".repeat(usize::try_from(2 * MIB_BYTES).expect("test allocation fits usize")),
        );
        first.complete(Some(exact_key("large-result")), Some(&cached));

        accepted(
            engine
                .accept(
                    "second",
                    spec("second", "fresh", source_bytes),
                    QueryReservation::new(1, 128),
                    None,
                )
                .expect("accept second session"),
        )
        .wait_until_active()
        .expect("activate second session")
        .complete(None, None);

        let third = accepted(
            engine
                .accept(
                    "third",
                    spec("third", "fresh", source_bytes),
                    QueryReservation::new(1, 128),
                    None,
                )
                .expect("admit third session after pressure relief"),
        );
        let snapshot = engine.snapshot();
        assert_eq!(snapshot.evictions.result_entries, 1);
        assert_eq!(snapshot.evictions.session_limit_sessions, 1);
        assert!(
            snapshot
                .sessions
                .iter()
                .any(|row| row.trace.path.ends_with("/second"))
        );
        assert!(
            snapshot
                .sessions
                .iter()
                .any(|row| row.trace.path.ends_with("/third"))
        );
        assert!(
            !snapshot
                .sessions
                .iter()
                .any(|row| row.trace.path.ends_with("/first"))
        );
        third
            .wait_until_active()
            .expect("activate third request")
            .complete(None, None);
    }

    #[test]
    fn lazy_resident_growth_is_accounted_and_evicts_an_idle_peer() {
        let mut configured = limits();
        configured.max_resident_memory_bytes = 3 * MIB_BYTES;
        let engine = DaemonEngine::new(configured);
        let source_bytes = MIB_BYTES;

        let growing = accepted(
            engine
                .accept(
                    "growing",
                    spec("growing", "fresh", source_bytes),
                    QueryReservation::new(1, 128),
                    None,
                )
                .expect("accept growing session"),
        )
        .wait_until_active()
        .expect("activate growing session");
        accepted(
            engine
                .accept(
                    "idle-peer",
                    spec("idle-peer", "fresh", source_bytes),
                    QueryReservation::new(1, 128),
                    None,
                )
                .expect("accept peer session"),
        )
        .wait_until_active()
        .expect("activate peer session")
        .complete(None, None);

        assert!(growing.refresh_resident_state("fresh", Some("fresh"), 2 * MIB_BYTES));
        let snapshot = engine.snapshot();
        assert!(snapshot.usage.resident_memory_estimate_bytes > 2 * source_bytes);
        assert_eq!(snapshot.evictions.resident_memory_sessions, 1);
        assert!(
            snapshot
                .sessions
                .iter()
                .any(|row| row.trace.path.ends_with("/growing"))
        );
        assert!(
            !snapshot
                .sessions
                .iter()
                .any(|row| row.trace.path.ends_with("/idle-peer"))
        );
        growing.complete(None, None);
    }

    #[test]
    fn lazy_resident_growth_over_the_ceiling_closes_its_session() {
        let mut configured = limits();
        configured.max_resident_memory_bytes = 2 * MIB_BYTES;
        let engine = DaemonEngine::new(configured);
        let growing = accepted(
            engine
                .accept(
                    "growing",
                    spec("growing", "fresh", MIB_BYTES),
                    QueryReservation::new(1, 128),
                    None,
                )
                .expect("accept growing session"),
        )
        .wait_until_active()
        .expect("activate growing session");

        assert!(!growing.refresh_resident_state("fresh", Some("fresh"), 3 * MIB_BYTES));
        assert_eq!(
            engine.snapshot().sessions.first().map(|row| row.state),
            Some(SessionState::Closing)
        );
        growing.complete(None, None);
        let snapshot = engine.snapshot();
        assert_eq!(snapshot.usage.resident_sessions, 0);
        assert_eq!(snapshot.evictions.resident_memory_sessions, 1);
    }

    #[test]
    fn queued_request_promotes_after_release_and_cancellation_is_observable() {
        let mut configured = limits();
        configured.max_concurrent_requests = 1;
        configured.max_query_workers = 1;
        configured.max_queued_requests = 1;
        let engine = DaemonEngine::new(configured);

        let first = accepted(
            engine
                .accept(
                    "active",
                    spec("first", "fresh", 100),
                    QueryReservation::new(1, 128),
                    None,
                )
                .expect("accept active request"),
        )
        .wait_until_active()
        .expect("activate first request");
        let queued = accepted(
            engine
                .accept(
                    "queued",
                    spec("second", "fresh", 100),
                    QueryReservation::new(1, 128),
                    None,
                )
                .expect("accept queued request"),
        );
        assert_eq!(engine.snapshot().usage.queued_requests, 1);
        assert!(matches!(
            engine.accept(
                "overflow",
                spec("third", "fresh", 100),
                QueryReservation::new(1, 128),
                None,
            ),
            Err(AdmissionFailure::ResourcePressure)
        ));

        let waiter = std::thread::spawn(move || queued.wait_until_active());
        first.complete(None, None);
        let promoted = waiter
            .join()
            .expect("queued waiter thread")
            .expect("promote queued request");
        engine
            .cancel(promoted.request_id())
            .expect("request cancellation");
        assert!(promoted.cancellation_requested());
    }

    #[test]
    fn stale_active_session_closes_only_after_its_request_finishes() {
        let engine = DaemonEngine::new(limits());
        let old = accepted(
            engine
                .accept(
                    "old",
                    spec("trace", "old", 100),
                    QueryReservation::new(1, 128),
                    None,
                )
                .expect("accept old request"),
        )
        .wait_until_active()
        .expect("activate old request");
        let new = accepted(
            engine
                .accept(
                    "new",
                    spec("trace", "new", 100),
                    QueryReservation::new(1, 128),
                    None,
                )
                .expect("accept replacement request"),
        )
        .wait_until_active()
        .expect("activate replacement request");

        let snapshot = engine.snapshot();
        assert_eq!(
            snapshot
                .sessions
                .iter()
                .filter(|row| row.state == SessionState::Closing)
                .count(),
            1
        );
        old.complete(None, None);
        assert_eq!(engine.snapshot().evictions.freshness_invalidations, 1);
        new.complete(None, None);
    }

    #[test]
    fn expired_queue_entry_is_not_promoted_when_capacity_reopens() {
        let mut configured = limits();
        configured.max_concurrent_requests = 1;
        configured.max_query_workers = 1;
        configured.admission_timeout_ms = 1;
        let engine = DaemonEngine::new(configured);

        let active = accepted(
            engine
                .accept(
                    "active",
                    spec("first", "fresh", 100),
                    QueryReservation::new(1, 128),
                    None,
                )
                .expect("accept active request"),
        )
        .wait_until_active()
        .expect("activate first request");
        let queued = accepted(
            engine
                .accept(
                    "queued",
                    spec("second", "fresh", 100),
                    QueryReservation::new(1, 128),
                    None,
                )
                .expect("accept queued request"),
        );
        std::thread::sleep(Duration::from_millis(5));
        active.complete(None, None);
        assert_eq!(
            queued.wait_until_active().map(|_| ()),
            Err(AdmissionFailure::ResourcePressure)
        );
        assert_eq!(engine.snapshot().usage.active_requests, 0);
    }

    #[test]
    fn idle_timeout_discards_only_inactive_sessions() {
        let mut configured = limits();
        configured.idle_timeout_ms = 1;
        let engine = DaemonEngine::new(configured);
        accepted(
            engine
                .accept(
                    "first",
                    spec("first", "fresh", 100),
                    QueryReservation::new(1, 128),
                    None,
                )
                .expect("accept first request"),
        )
        .wait_until_active()
        .expect("activate first request")
        .complete(None, None);

        std::thread::sleep(Duration::from_millis(5));
        let second = accepted(
            engine
                .accept(
                    "second",
                    spec("second", "fresh", 100),
                    QueryReservation::new(1, 128),
                    None,
                )
                .expect("accept after idle timeout"),
        );
        let snapshot = engine.snapshot();
        assert_eq!(snapshot.evictions.idle_timeout_sessions, 1);
        assert!(
            !snapshot
                .sessions
                .iter()
                .any(|row| row.trace.path.ends_with("/first"))
        );
        second
            .wait_until_active()
            .expect("activate second request")
            .complete(None, None);
    }

    #[test]
    fn daemon_restart_drops_memory_cache_without_affecting_execution() {
        let first_engine = DaemonEngine::new(limits());
        let request = accepted(
            first_engine
                .accept(
                    "build",
                    spec("trace", "fresh", 100),
                    QueryReservation::new(1, 128),
                    Some(&exact_key("summary")),
                )
                .expect("accept cache build"),
        );
        let result = execution("cached");
        request
            .wait_until_active()
            .expect("activate cache build")
            .complete(Some(exact_key("summary")), Some(&result));
        assert_eq!(first_engine.snapshot().usage.exact_response_entries, 1);

        let restarted = DaemonEngine::new(limits());
        let outcome = restarted
            .accept(
                "after-restart",
                spec("trace", "fresh", 100),
                QueryReservation::new(1, 128),
                Some(&exact_key("summary")),
            )
            .expect("execute after restart");
        assert!(matches!(outcome, AcceptOutcome::Accepted(_)));
        assert_eq!(restarted.snapshot().usage.cache_hits, 0);
    }

    #[test]
    fn exact_results_do_not_cross_command_format_or_invocation_axes() {
        let engine = DaemonEngine::new(limits());
        let summary_key = exact_key_with("nsys.summary", OutputFormat::Json, "context-a");
        let request = accepted(
            engine
                .accept(
                    "build",
                    spec("trace", "fresh", 100),
                    QueryReservation::new(1, 128),
                    Some(&summary_key),
                )
                .expect("accept exact result build"),
        );
        let result = execution("summary");
        request
            .wait_until_active()
            .expect("activate exact result build")
            .complete(Some(summary_key), Some(&result));

        for (request_id, key) in [
            (
                "different-command",
                exact_key_with("nsys.stats", OutputFormat::Json, "context-a"),
            ),
            (
                "different-format",
                exact_key_with("nsys.summary", OutputFormat::Csv, "context-a"),
            ),
            (
                "different-context",
                exact_key_with("nsys.summary", OutputFormat::Json, "context-b"),
            ),
        ] {
            let outcome = engine
                .accept(
                    request_id,
                    spec("trace", "fresh", 100),
                    QueryReservation::new(1, 128),
                    Some(&key),
                )
                .expect("accept distinct semantic request");
            accepted(outcome)
                .wait_until_active()
                .expect("activate distinct semantic request")
                .complete(None, None);
        }
        assert_eq!(engine.snapshot().usage.cache_hits, 0);
        assert_eq!(engine.snapshot().usage.cache_misses, 4);
    }

    #[test]
    fn admission_only_requests_are_bounded_without_creating_sessions() {
        let mut configured = limits();
        configured.max_concurrent_requests = 1;
        configured.max_query_workers = 1;
        let engine = DaemonEngine::new(configured);
        let active = accepted(
            engine
                .accept(
                    "sessionless-active",
                    None,
                    QueryReservation::new(1, 128),
                    None,
                )
                .expect("admit sessionless request"),
        )
        .wait_until_active()
        .expect("activate sessionless request");
        let queued = accepted(
            engine
                .accept(
                    "sessionless-queued",
                    None,
                    QueryReservation::new(1, 128),
                    None,
                )
                .expect("queue bounded sessionless request"),
        );
        let snapshot = engine.snapshot();
        assert_eq!(snapshot.usage.resident_sessions, 0);
        assert_eq!(snapshot.usage.active_requests, 1);
        assert_eq!(snapshot.usage.queued_requests, 1);
        active.complete(None, None);
        queued
            .wait_until_active()
            .expect("promote sessionless request")
            .complete(None, None);
    }

    #[test]
    fn one_resident_session_has_one_execution_slot_without_blocking_peers() {
        let engine = DaemonEngine::new(limits());
        let first = accepted(
            engine
                .accept(
                    "same-first",
                    spec("same", "fresh", 100),
                    QueryReservation::new(1, 128),
                    None,
                )
                .expect("admit first same-session request"),
        )
        .wait_until_active()
        .expect("activate first same-session request");
        let same_session = accepted(
            engine
                .accept(
                    "same-second",
                    spec("same", "fresh", 100),
                    QueryReservation::new(1, 128),
                    None,
                )
                .expect("queue second same-session request"),
        );
        let peer = accepted(
            engine
                .accept(
                    "peer",
                    spec("peer", "fresh", 100),
                    QueryReservation::new(1, 128),
                    None,
                )
                .expect("admit independent peer"),
        )
        .wait_until_active()
        .expect("activate independent peer");
        let snapshot = engine.snapshot();
        assert_eq!(snapshot.usage.active_requests, 2);
        assert_eq!(snapshot.usage.queued_requests, 1);
        peer.complete(None, None);
        first.complete(None, None);
        same_session
            .wait_until_active()
            .expect("promote same-session follower")
            .complete(None, None);
    }

    #[test]
    fn changed_post_query_freshness_closes_instead_of_relabeling_session() {
        let engine = DaemonEngine::new(limits());
        let active = accepted(
            engine
                .accept(
                    "mutated",
                    spec("trace", "before", 100),
                    QueryReservation::new(1, 128),
                    None,
                )
                .expect("admit request"),
        )
        .wait_until_active()
        .expect("activate request");
        assert!(!active.refresh_resident_state("before", Some("after"), 100));
        assert_eq!(
            engine.snapshot().sessions.first().map(|row| row.state),
            Some(SessionState::Closing)
        );
        active.complete(None, None);
        let snapshot = engine.snapshot();
        assert_eq!(snapshot.usage.resident_sessions, 0);
        assert_eq!(snapshot.evictions.freshness_invalidations, 1);
    }

    #[test]
    fn queued_request_is_detached_when_its_session_becomes_stale() {
        let mut configured = limits();
        configured.max_concurrent_requests = 1;
        configured.max_query_workers = 1;
        let engine = DaemonEngine::new(configured);
        let old_active = accepted(
            engine
                .accept(
                    "old-active",
                    spec("trace", "before", 100),
                    QueryReservation::new(1, 128),
                    None,
                )
                .expect("admit old active request"),
        )
        .wait_until_active()
        .expect("activate old request");
        let old_queued = accepted(
            engine
                .accept(
                    "old-queued",
                    spec("trace", "before", 100),
                    QueryReservation::new(1, 128),
                    None,
                )
                .expect("queue request on old session"),
        );
        let replacement = accepted(
            engine
                .accept(
                    "replacement",
                    spec("trace", "after", 100),
                    QueryReservation::new(1, 128),
                    None,
                )
                .expect("admit replacement session"),
        );

        assert!(matches!(
            old_queued.wait_until_active(),
            Err(AdmissionFailure::SessionInvalidated)
        ));
        assert_eq!(
            engine.snapshot().sessions.first().map(|row| row.state),
            Some(SessionState::Closing)
        );
        old_active.complete(None, None);
        replacement
            .wait_until_active()
            .expect("activate replacement request")
            .complete(None, None);
        assert_eq!(engine.snapshot().evictions.freshness_invalidations, 1);
    }

    #[test]
    fn resident_accounting_includes_daemon_owned_session_state() {
        let engine = DaemonEngine::new(limits());
        let active = accepted(
            engine
                .accept(
                    "structural-accounting",
                    spec("trace", "fresh", 0),
                    QueryReservation::new(1, 128),
                    None,
                )
                .expect("admit session"),
        )
        .wait_until_active()
        .expect("activate request");
        let accounted = engine.snapshot().usage.resident_memory_estimate_bytes;
        assert!(
            accounted
                >= u64::try_from(std::mem::size_of::<Session>())
                    .expect("session size fits resident accounting")
        );
        active.complete(None, None);
    }

    #[test]
    fn failed_resident_execution_discards_state_and_detaches_followers() {
        let engine = DaemonEngine::new(limits());
        let active = accepted(
            engine
                .accept(
                    "failed-active",
                    spec("trace", "fresh", 100),
                    QueryReservation::new(1, 128),
                    None,
                )
                .expect("admit active request"),
        )
        .wait_until_active()
        .expect("activate request");
        let queued = accepted(
            engine
                .accept(
                    "failed-follower",
                    spec("trace", "fresh", 100),
                    QueryReservation::new(1, 128),
                    None,
                )
                .expect("queue session follower"),
        );

        active.discard_resident_state_after_failure();
        assert!(matches!(
            queued.wait_until_active(),
            Err(AdmissionFailure::SessionInvalidated)
        ));
        active.complete(None, None);
        let snapshot = engine.snapshot();
        assert_eq!(snapshot.usage.resident_sessions, 0);
        assert_eq!(snapshot.evictions.other_sessions, 1);
    }

    #[test]
    fn failed_execution_is_never_exact_cached() {
        let engine = DaemonEngine::new(limits());
        let key = exact_key("summary");
        let active = accepted(
            engine
                .accept(
                    "failure",
                    spec("trace", "fresh", 100),
                    QueryReservation::new(1, 128),
                    Some(&key),
                )
                .expect("admit failed request"),
        )
        .wait_until_active()
        .expect("activate failed request");
        let mut failure = execution("handled failure");
        failure.set_exit_code(1);
        active.complete(Some(key.clone()), Some(&failure));
        assert_eq!(engine.snapshot().usage.exact_response_entries, 0);
        assert!(matches!(
            engine.accept(
                "retry",
                spec("trace", "fresh", 100),
                QueryReservation::new(1, 128),
                Some(&key),
            ),
            Ok(AcceptOutcome::Accepted(_))
        ));
    }

    #[test]
    fn active_cancellation_invokes_the_registered_source_interrupt() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let engine = DaemonEngine::new(limits());
        let active = accepted(
            engine
                .accept("cancel", None, QueryReservation::new(1, 128), None)
                .expect("admit request"),
        )
        .wait_until_active()
        .expect("activate request");
        let interrupted = Arc::new(AtomicBool::new(false));
        let observed = Arc::clone(&interrupted);
        active
            .cancellation_token()
            .register_interrupt(move || observed.store(true, Ordering::Release));
        engine.cancel_active_requests();
        assert!(active.cancellation_requested());
        assert!(interrupted.load(Ordering::Acquire));
        active.complete(None, None);
    }
}
