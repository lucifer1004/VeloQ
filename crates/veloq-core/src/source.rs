//! `ProfileSource` — the trait every profile backend implements so the
//! `veloq` binary can dispatch to it.
//!
//! A source is more than a parser: it owns its clap subcommand tree,
//! the trace-detection heuristic that lets meta verbs find it without
//! the user typing the source name, and the run glue that produces
//! output in whichever [`OutputFormat`] the user asked for.

use crate::diagnostic::{ErrorCode, VeloqDiagnostic};
use crate::envelope::{SourceRef, TraceSpan};
use std::error::Error;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use thiserror::Error;

pub type SourceRunError = Box<dyn Error + Send + Sync + 'static>;
pub type SourceRunResult<T> = Result<T, SourceRunError>;

/// Per-request cancellation shared by the daemon scheduler and a source.
///
/// Sources with an interruptible query engine register one callback when
/// execution starts. A cancellation that races with registration still
/// invokes the callback exactly once for that request.
#[derive(Clone, Default)]
pub struct CancellationToken {
    state: Arc<CancellationState>,
}

#[derive(Default)]
struct CancellationState {
    requested: AtomicBool,
    interrupt: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
}

impl std::fmt::Debug for CancellationToken {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CancellationToken")
            .field("requested", &self.is_cancelled())
            .finish_non_exhaustive()
    }
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_cancelled(&self) -> bool {
        self.state.requested.load(Ordering::Acquire)
    }

    pub fn cancel(&self) {
        if self.state.requested.swap(true, Ordering::AcqRel) {
            return;
        }
        let interrupt = self
            .state
            .interrupt
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        if let Some(interrupt) = interrupt {
            interrupt();
        }
    }

    pub fn register_interrupt(&self, interrupt: impl Fn() + Send + Sync + 'static) {
        let interrupt: Arc<dyn Fn() + Send + Sync> = Arc::new(interrupt);
        let mut registered = self
            .state
            .interrupt
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.is_cancelled() {
            drop(registered);
            interrupt();
        } else {
            *registered = Some(interrupt);
        }
    }
}

/// Source-owned state retained by one daemon session.
pub trait ProfileSession: Send {
    fn execute(
        &mut self,
        matches: &clap::ArgMatches,
        fmt: OutputFormat,
        cancellation: &CancellationToken,
    ) -> SourceRunResult<SourceExecution>;

    /// Source-owned memory retained in addition to the session identity's
    /// initial estimate. The daemon refreshes accounting after each request
    /// because a session may build disposable in-memory query state lazily.
    fn additional_resident_memory_estimate_bytes(&self) -> u64 {
        0
    }
}

/// Source-owned identity needed by frontends before query execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceQueryContext {
    pub command: String,
    pub trace_path: Option<PathBuf>,
    pub raw_stdout: bool,
}

/// Source-owned identity axes for deciding whether daemon memory state can be
/// reused across requests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceSessionIdentity {
    pub trace_kind: String,
    pub canonical_trace_path: PathBuf,
    pub configuration_key: String,
    pub freshness_key: String,
    pub resident_memory_estimate_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceSessionConfig {
    pub query_workers: u64,
    /// `None` leaves memory policy to the source query engine.
    pub query_memory_bytes: Option<u64>,
    /// Whole-daemon retained-memory ceiling. Sources use this only to refuse
    /// optional resident accelerators whose conservative retained-size bound
    /// cannot fit; daemon accounting remains authoritative after construction.
    pub resident_memory_bytes: u64,
}

/// Default worker budget shared by one-shot and daemon query engines.
///
/// DuckDB otherwise creates one worker per available CPU. The cap keeps worker
/// creation bounded on large hosts while following available parallelism on
/// smaller hosts.
pub fn default_query_worker_count() -> usize {
    const QUERY_WORKER_CAP: usize = 16;

    std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .min(QUERY_WORKER_CAP)
}

/// Process-independent output from one source command execution.
///
/// Source crates render their existing JSON, CSV, or table projection into
/// these buffers instead of writing process-global stdout/stderr directly.
/// The one-shot binary writes the buffers to the process streams; the daemon
/// transports the same bytes back to its client. Keeping the rendered bytes
/// here avoids a lossy deserialize/re-project hop for heterogeneous source
/// payloads and makes the execution ownership boundary explicit.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SourceExecution {
    exit_code: i32,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

impl SourceExecution {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_parts(exit_code: i32, stdout: Vec<u8>, stderr: Vec<u8>) -> Self {
        Self {
            exit_code,
            stdout,
            stderr,
        }
    }

    pub fn exit_code(&self) -> i32 {
        self.exit_code
    }

    pub fn set_exit_code(&mut self, exit_code: i32) {
        self.exit_code = exit_code;
    }

    pub fn stdout(&self) -> &[u8] {
        &self.stdout
    }

    pub fn stderr(&self) -> &[u8] {
        &self.stderr
    }

    pub fn write_stdout(&mut self, bytes: impl AsRef<[u8]>) {
        self.stdout.extend_from_slice(bytes.as_ref());
    }

    pub fn write_stdout_line(&mut self, line: impl AsRef<str>) {
        self.stdout.extend_from_slice(line.as_ref().as_bytes());
        self.stdout.push(b'\n');
    }

    pub fn write_stderr_line(&mut self, line: impl AsRef<str>) {
        self.stderr.extend_from_slice(line.as_ref().as_bytes());
        self.stderr.push(b'\n');
    }

    pub fn retained_memory_estimate_bytes(&self) -> u64 {
        u64::try_from(std::mem::size_of::<Self>())
            .unwrap_or(u64::MAX)
            .saturating_add(u64::try_from(self.stdout.capacity()).unwrap_or(u64::MAX))
            .saturating_add(u64::try_from(self.stderr.capacity()).unwrap_or(u64::MAX))
    }

    /// Project this execution onto process-owned streams for one-shot use.
    pub fn write_to_process(&self) -> io::Result<()> {
        let stderr = io::stderr();
        let mut stderr = stderr.lock();
        stderr.write_all(&self.stderr)?;
        stderr.flush()?;

        let stdout = io::stdout();
        let mut stdout = stdout.lock();
        stdout.write_all(&self.stdout)?;
        stdout.flush()
    }
}

#[derive(Debug, Error)]
pub enum OutputFormatError {
    #[error("unknown --format `{value}` (expected: json, csv, table)")]
    Unknown { value: String },
}

impl OutputFormatError {
    pub fn unknown(value: &str) -> Self {
        Self::Unknown {
            value: value.to_string(),
        }
    }
}

impl VeloqDiagnostic for OutputFormatError {
    fn code(&self) -> ErrorCode {
        match self {
            Self::Unknown { .. } => ErrorCode::new("cli.unknown-format"),
        }
    }
}

/// Output format every CLI invocation has to pick. JSON is the agent
/// contract; CSV / table are human-only conveniences. Lives in
/// `veloq-core` so sources don't each redefine the same enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OutputFormat {
    Json,
    Csv,
    Table,
}

impl OutputFormat {
    /// Parse the `--format` flag. Accepts `json`, `csv`, `table`,
    /// case-insensitive (plus `tbl` as a shorter alias for `table`).
    pub fn parse(s: &str) -> Result<Self, OutputFormatError> {
        let normalized = s.to_ascii_lowercase();
        match normalized.as_str() {
            "json" => Ok(Self::Json),
            "csv" => Ok(Self::Csv),
            "table" | "tbl" => Ok(Self::Table),
            _ => Err(OutputFormatError::unknown(s)),
        }
    }
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Json => "json",
            Self::Csv => "csv",
            Self::Table => "table",
        })
    }
}

/// One pluggable profile backend.
///
/// The `veloq` binary registers shared `ProfileSource` trait objects at
/// startup; CLI dispatch is the sum of every source's contribution
/// under a top-level `veloq <kind> …` namespace, plus a configured
/// default whose verbs are hoisted to `veloq <verb> …`.
///
/// `Send + Sync` is required only so a registry can be shared across
/// threads if a future server frontend wants to. The trait itself
/// makes no concurrency assumptions; methods take `&self`.
///
/// Each source owns its own rendering into [`SourceExecution`]. This avoids
/// forcing strongly typed response shapes through a `Deserialize` round trip
/// while keeping process-owned stdout/stderr outside the query engine.
pub trait ProfileSource: Send + Sync {
    /// Stable short name. Becomes the CLI namespace
    /// (`veloq <kind> …`) and lands in `envelope.source.kind`.
    /// Lowercase ASCII, no spaces.
    fn kind(&self) -> &'static str;

    /// Source-specific semver (`"v0"`, `"v1"`…). Bumps independently
    /// from the envelope schema version; lands in
    /// `envelope.source.version`.
    fn version(&self) -> &'static str;

    /// Combine `kind()` + `version()` into the envelope's source ref.
    fn source_ref(&self) -> SourceRef {
        SourceRef {
            kind: self.kind(),
            version: self.version(),
        }
    }

    /// Heuristic: does this path look like a trace this source can
    /// handle? Used by meta verbs (`veloq info <trace>`) to pick a
    /// source by trace extension or magic when the user didn't name
    /// one. Side-effect-free; should not open the file.
    fn detect(&self, trace: &Path) -> bool;

    /// Best-effort trace-wide `(origin_ns, span_ns)` an agent uses as
    /// the normalization denominator on cross-trace diffs. Called once
    /// per CLI invocation, after argument parsing and before verb
    /// dispatch; the result lands on the envelope (`trace_span`) so
    /// every response carries it.
    ///
    /// Default returns `None` — sources opt in when they can derive
    /// the answer cheaply (e.g. NSys reads from its metadata cache).
    /// Failures should return `None` rather than `Err`: a
    /// missing `trace_span` degrades agent normalization but should
    /// never block the verb itself. Implementations are responsible
    /// for logging any underlying error at warn level.
    fn compute_trace_span(&self, _trace: &Path) -> Option<TraceSpan> {
        None
    }

    /// The clap subcommand tree this source contributes. Returned as
    /// a built [`clap::Command`]; the top-level CLI grafts it under
    /// the source name (or hoists its subcommands when this is the
    /// configured default).
    ///
    /// Sources are free to compose their subtree however they like —
    /// `Command::new(self.kind()).subcommand(...)` is the obvious
    /// idiom but not required.
    fn cli(&self) -> clap::Command;

    /// Verb names this source can execute through the private local-daemon
    /// transport. The frontend qualifies them with `kind()` at the protocol
    /// boundary. An empty slice keeps the source one-shot only.
    fn daemon_command_verbs(&self) -> &'static [&'static str] {
        &[]
    }

    fn supports_daemon_command(&self, qualified_command: &str) -> bool {
        qualified_command
            .strip_prefix(self.kind())
            .and_then(|command| command.strip_prefix('.'))
            .is_some_and(|verb| self.daemon_command_verbs().contains(&verb))
    }

    /// Resolve a trace into source-owned daemon reuse identity without
    /// executing a query. Returning `None` keeps execution daemon-capable but
    /// bypasses resident and exact-response reuse for this request.
    fn daemon_session_identity(
        &self,
        _trace: &Path,
    ) -> SourceRunResult<Option<SourceSessionIdentity>> {
        Ok(None)
    }

    /// Open source-owned resident state after daemon admission. Returning
    /// `None` keeps the request on the ordinary source execution path.
    fn open_daemon_session(
        &self,
        _resolved_trace: &Path,
        _config: SourceSessionConfig,
    ) -> SourceRunResult<Option<Box<dyn ProfileSession>>> {
        Ok(None)
    }

    /// Resolve command and trace identity without executing the query.
    fn query_context(&self, matches: &clap::ArgMatches) -> SourceRunResult<SourceQueryContext>;

    /// Execute the dispatched verb and render its output in the requested
    /// format without writing process-global stdout or stderr.
    ///
    /// `matches` is the [`ArgMatches`] for this source's subtree
    /// (the result of parsing against [`Self::cli`]); sources need
    /// not handle their own namespace prefix.
    ///
    /// Return contract:
    /// - `Ok(SourceExecution { exit_code: 0, .. })` — verb succeeded.
    /// - `Ok(SourceExecution { exit_code: 1, .. })` — verb failed and the
    ///   source rendered its contextual `EnvelopeError`.
    /// - `Err(_)` — top-level / unhandled failure. The caller emits a
    ///   CLI-level error envelope (no verb/trace context) and exits 1.
    ///
    /// Splitting the "handled" case (envelope already on stdout) from
    /// the "unhandled" case keeps verb-level error envelopes
    /// agent-actionable without forcing a panic-style exit through
    /// `process::exit` from inside the source.
    ///
    /// [`ArgMatches`]: clap::ArgMatches
    fn execute(
        &self,
        matches: &clap::ArgMatches,
        fmt: OutputFormat,
    ) -> SourceRunResult<SourceExecution>;

    /// Execute through a daemon with the trace path already resolved against
    /// the client's working directory. Sources that advertise daemon commands
    /// override this when their normal `ArgMatches` retain a relative path.
    fn execute_daemon(
        &self,
        matches: &clap::ArgMatches,
        fmt: OutputFormat,
        _resolved_trace: &Path,
    ) -> SourceRunResult<SourceExecution> {
        self.execute(matches, fmt)
    }

    /// Daemon execution with a per-request cancellation signal. Sources with
    /// interruptible resident engines override the session path; this default
    /// still prevents work from starting after cancellation and discards a
    /// result that raced with cancellation.
    fn execute_daemon_cancellable(
        &self,
        matches: &clap::ArgMatches,
        fmt: OutputFormat,
        resolved_trace: &Path,
        _config: SourceSessionConfig,
        cancellation: &CancellationToken,
    ) -> SourceRunResult<SourceExecution> {
        if cancellation.is_cancelled() {
            return Err(io::Error::new(io::ErrorKind::Interrupted, "query cancelled").into());
        }
        let execution = self.execute_daemon(matches, fmt, resolved_trace)?;
        if cancellation.is_cancelled() {
            return Err(io::Error::new(io::ErrorKind::Interrupted, "query cancelled").into());
        }
        Ok(execution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Command;

    struct FakeSource;
    impl ProfileSource for FakeSource {
        fn kind(&self) -> &'static str {
            "fake"
        }
        fn version(&self) -> &'static str {
            "v1"
        }
        fn detect(&self, p: &Path) -> bool {
            p.extension().is_some_and(|e| e == "fake")
        }
        fn cli(&self) -> Command {
            Command::new("fake").subcommand(Command::new("ping"))
        }
        fn query_context(&self, m: &clap::ArgMatches) -> SourceRunResult<SourceQueryContext> {
            Ok(SourceQueryContext {
                command: format!(
                    "fake.{}",
                    m.subcommand_name()
                        .ok_or_else(|| std::io::Error::other("no subcommand"))?
                ),
                trace_path: None,
                raw_stdout: false,
            })
        }
        fn execute(
            &self,
            m: &clap::ArgMatches,
            _fmt: OutputFormat,
        ) -> SourceRunResult<SourceExecution> {
            let verb = m
                .subcommand_name()
                .ok_or_else(|| std::io::Error::other("no subcommand"))?;
            if verb != "ping" {
                return Err(
                    std::io::Error::other(format!("unexpected subcommand `{verb}`")).into(),
                );
            }
            let mut execution = SourceExecution::new();
            execution.write_stdout_line("pong");
            Ok(execution)
        }
    }

    #[test]
    fn source_ref_combines_kind_and_version() {
        let s = FakeSource;
        let r = s.source_ref();
        assert_eq!(r.kind, "fake");
        assert_eq!(r.version, "v1");
    }

    #[test]
    fn detect_matches_by_extension() {
        let s = FakeSource;
        assert!(s.detect(Path::new("/tmp/t.fake")));
        assert!(!s.detect(Path::new("/tmp/t.nsys-rep")));
        assert!(!s.detect(Path::new("/tmp/t")));
    }

    #[test]
    fn execute_dispatches_subcommand_without_process_io() -> SourceRunResult<()> {
        let s = FakeSource;
        let m = s.cli().try_get_matches_from(["fake", "ping"])?;
        let execution = s.execute(&m, OutputFormat::Json)?;
        assert_eq!(execution.exit_code(), 0);
        assert_eq!(execution.stdout(), b"pong\n");
        assert!(execution.stderr().is_empty());
        Ok(())
    }

    #[test]
    fn output_format_parse_round_trip() {
        for s in ["json", "JSON", "csv", "Csv", "table", "tbl"] {
            assert!(OutputFormat::parse(s).is_ok());
        }
        assert!(OutputFormat::parse("bogus").is_err());
    }
}
