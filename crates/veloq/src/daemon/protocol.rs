use std::ffi::{OsStr, OsString};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use veloq_core::OutputFormat;

use super::DaemonError;
use super::config::DaemonLimits;
use super::session::DaemonSnapshot;

pub const CONTROL_VERSION: &str = "1";
pub const PROTOCOL_VERSION: &str = "2";
pub const MAX_FRAME_BYTES: u64 = 64 * 1024 * 1024;
pub const OUTPUT_CHUNK_BYTES: usize = 1024 * 1024;
pub const QUERY_ENVIRONMENT_KEYS: &[&str] = &["PATH", "VELOQ_DUCKDB_THREADS", "VELOQ_UNSTABLE"];
const NON_SEMANTIC_VALUE_FLAGS: &[&str] = &["--daemon", "--daemon-connect-timeout-ms", "--format"];

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "encoding", rename_all = "snake_case")]
pub enum EncodedOsString {
    #[cfg(unix)]
    Unix { bytes: Vec<u8> },
    #[cfg(windows)]
    Windows { units: Vec<u16> },
}

impl EncodedOsString {
    pub fn encode(value: &OsStr) -> Self {
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            Self::Unix {
                bytes: value.as_bytes().to_vec(),
            }
        }
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStrExt;
            Self::Windows {
                units: value.encode_wide().collect(),
            }
        }
    }

    pub fn decode(&self) -> OsString {
        match self {
            #[cfg(unix)]
            Self::Unix { bytes } => {
                use std::os::unix::ffi::OsStringExt;
                OsString::from_vec(bytes.clone())
            }
            #[cfg(windows)]
            Self::Windows { units } => {
                use std::os::windows::ffi::OsStringExt;
                OsString::from_wide(units)
            }
        }
    }

    fn equals_ascii(&self, expected: &str) -> bool {
        match self {
            #[cfg(unix)]
            Self::Unix { bytes } => bytes == expected.as_bytes(),
            #[cfg(windows)]
            Self::Windows { units } => units.iter().copied().eq(expected.encode_utf16()),
        }
    }

    fn starts_with_ascii(&self, expected: &str) -> bool {
        match self {
            #[cfg(unix)]
            Self::Unix { bytes } => bytes.starts_with(expected.as_bytes()),
            #[cfg(windows)]
            Self::Windows { units } => {
                let expected = expected.encode_utf16().collect::<Vec<_>>();
                units.starts_with(&expected)
            }
        }
    }

    fn retained_heap_bytes(&self) -> u64 {
        let payload = match self {
            #[cfg(unix)]
            Self::Unix { bytes } => bytes.capacity(),
            #[cfg(windows)]
            Self::Windows { units } => units.capacity().saturating_mul(std::mem::size_of::<u16>()),
        };
        u64::try_from(payload).unwrap_or(u64::MAX)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvironmentValue {
    pub name: String,
    pub value: Option<EncodedOsString>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryInvocation {
    pub arguments: Vec<EncodedOsString>,
    pub cwd: EncodedOsString,
    pub environment: Vec<EnvironmentValue>,
    pub terminal_width: Option<u16>,
}

/// Invocation fields that can change a rendered response. Routing controls are
/// transport-only and are deliberately absent.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SemanticInvocationKey {
    arguments: Vec<EncodedOsString>,
    cwd: Option<EncodedOsString>,
    terminal_width: Option<u16>,
}

impl SemanticInvocationKey {
    pub fn retained_heap_bytes(&self) -> u64 {
        u64::try_from(
            self.arguments
                .capacity()
                .saturating_mul(std::mem::size_of::<EncodedOsString>()),
        )
        .unwrap_or(u64::MAX)
        .saturating_add(
            self.arguments
                .iter()
                .fold(0u64, |total, argument| {
                    total.saturating_add(argument.retained_heap_bytes())
                })
                .saturating_add(
                    self.cwd
                        .as_ref()
                        .map_or(0, EncodedOsString::retained_heap_bytes),
                ),
        )
    }
}

impl QueryInvocation {
    pub fn capture() -> io::Result<Self> {
        let cwd = std::env::current_dir()?;
        Ok(Self {
            arguments: std::env::args_os()
                .map(|value| EncodedOsString::encode(&value))
                .collect(),
            cwd: EncodedOsString::encode(cwd.as_os_str()),
            environment: QUERY_ENVIRONMENT_KEYS
                .iter()
                .map(|name| EnvironmentValue {
                    name: (*name).to_string(),
                    value: std::env::var_os(name)
                        .as_deref()
                        .map(EncodedOsString::encode),
                })
                .collect(),
            terminal_width: terminal_size::terminal_size().map(|(width, _)| width.0),
        })
    }

    pub fn decoded_arguments(&self) -> Vec<OsString> {
        self.arguments.iter().map(EncodedOsString::decode).collect()
    }

    pub fn decoded_cwd(&self) -> PathBuf {
        PathBuf::from(self.cwd.decode())
    }

    pub fn environment_matches_current(&self) -> bool {
        self.environment.len() == QUERY_ENVIRONMENT_KEYS.len()
            && self
                .environment
                .iter()
                .zip(QUERY_ENVIRONMENT_KEYS)
                .all(|(entry, expected_name)| {
                    entry.name == *expected_name
                        && std::env::var_os(expected_name)
                            == entry.value.as_ref().map(EncodedOsString::decode)
                })
    }

    pub fn semantic_key(
        &self,
        trace_argument: Option<&OsStr>,
        output_format: OutputFormat,
    ) -> SemanticInvocationKey {
        let arguments = self.arguments.get(1..).unwrap_or_default();
        let mut semantic = Vec::with_capacity(arguments.len());
        let mut index = 0;
        let encoded_trace = trace_argument.map(EncodedOsString::encode);
        let mut removed_trace = false;
        while let Some(argument) = arguments.get(index) {
            if NON_SEMANTIC_VALUE_FLAGS
                .iter()
                .any(|flag| argument.equals_ascii(flag))
            {
                index = index.saturating_add(2);
                continue;
            }
            if NON_SEMANTIC_VALUE_FLAGS.iter().any(|flag| {
                let mut prefix = String::with_capacity(flag.len() + 1);
                prefix.push_str(flag);
                prefix.push('=');
                argument.starts_with_ascii(&prefix)
            }) {
                index = index.saturating_add(1);
                continue;
            }
            if !removed_trace && encoded_trace.as_ref() == Some(argument) {
                removed_trace = true;
                index = index.saturating_add(1);
                continue;
            }
            semantic.push(argument.clone());
            index = index.saturating_add(1);
        }
        SemanticInvocationKey {
            arguments: semantic,
            cwd: trace_argument
                .is_some_and(|trace| PathBuf::from(trace).is_relative())
                .then(|| self.cwd.clone()),
            terminal_width: matches!(output_format, OutputFormat::Table)
                .then_some(self.terminal_width)
                .flatten(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capability {
    pub source: String,
    pub command: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientFrame {
    Hello {
        protocol_version: String,
        veloq_version: String,
    },
    Control {
        control_version: String,
        owner_token: String,
        operation: ControlOperation,
        timeout_ms: u64,
    },
    Query {
        request_id: String,
        source: String,
        command: String,
        invocation: QueryInvocation,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlOperation {
    Status,
    Stop,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerFrame {
    Hello {
        protocol_version: String,
        veloq_version: String,
        compatible: bool,
        capabilities: Vec<Capability>,
    },
    Status {
        control_version: String,
        process_id: u32,
        veloq_version: String,
        protocol_version: String,
        limits: DaemonLimits,
        snapshot: Box<DaemonSnapshot>,
    },
    Stopping {
        control_version: String,
    },
    Rejected {
        request_id: String,
        error: DaemonError,
    },
    Accepted {
        request_id: String,
    },
    OutputChunk {
        request_id: String,
        stream: OutputStream,
        bytes: Vec<u8>,
    },
    Completed {
        request_id: String,
        exit_code: i32,
    },
    Failed {
        request_id: String,
        error: DaemonError,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputStream {
    Stdout,
    Stderr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionOwnership {
    NotTransmitted,
    PreAcceptanceRejected,
    Accepted,
    Completed,
    Indeterminate,
}

impl ExecutionOwnership {
    pub fn permits_one_shot_fallback(self) -> bool {
        matches!(self, Self::NotTransmitted | Self::PreAcceptanceRejected)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestOwnership {
    request_id: String,
    state: ExecutionOwnership,
}

impl RequestOwnership {
    pub fn new(request_id: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            state: ExecutionOwnership::NotTransmitted,
        }
    }

    pub fn mark_transmitted(&mut self) {
        self.state = ExecutionOwnership::Indeterminate;
    }

    pub fn observe(&mut self, frame: &ServerFrame) -> io::Result<()> {
        let (request_id, next) = match frame {
            ServerFrame::Rejected { request_id, .. } => {
                (request_id, ExecutionOwnership::PreAcceptanceRejected)
            }
            ServerFrame::Accepted { request_id } => (request_id, ExecutionOwnership::Accepted),
            ServerFrame::OutputChunk { request_id, .. } => {
                if self.state != ExecutionOwnership::Accepted {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "daemon returned output before accepting the request",
                    ));
                }
                (request_id, ExecutionOwnership::Accepted)
            }
            ServerFrame::Completed { request_id, .. } | ServerFrame::Failed { request_id, .. } => {
                (request_id, ExecutionOwnership::Completed)
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "daemon returned a non-execution frame during request ownership tracking",
                ));
            }
        };
        if request_id != &self.request_id {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "daemon response request id does not match the transmitted request",
            ));
        }
        self.state = next;
        Ok(())
    }

    #[cfg(test)]
    pub fn state(&self) -> ExecutionOwnership {
        self.state
    }

    pub fn permits_one_shot_fallback(&self) -> bool {
        self.state.permits_one_shot_fallback()
    }
}

pub fn write_frame(writer: &mut impl Write, frame: &impl Serialize) -> io::Result<()> {
    let bytes = serde_json::to_vec(frame).map_err(io::Error::other)?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) >= MAX_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "daemon protocol frame exceeds the size limit",
        ));
    }
    writer.write_all(&bytes)?;
    writer.write_all(b"\n")?;
    writer.flush()
}

pub fn read_frame<T: DeserializeOwned>(reader: &mut impl BufRead) -> io::Result<T> {
    let mut bytes = Vec::new();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "daemon connection closed before a complete frame",
            ));
        }
        let take = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |index| index + 1);
        if bytes.len() as u64 + take as u64 > MAX_FRAME_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "daemon protocol frame exceeds the size limit",
            ));
        }
        let terminated = available.get(take.saturating_sub(1)) == Some(&b'\n');
        let chunk = available.get(..take).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "daemon protocol frame boundary exceeds the buffered input",
            )
        })?;
        bytes.extend_from_slice(chunk);
        reader.consume(take);
        if terminated {
            break;
        }
    }
    serde_json::from_slice(&bytes).map_err(io::Error::other)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufReader, Cursor};

    #[test]
    fn frame_round_trip_is_newline_delimited() -> io::Result<()> {
        let frame = ClientFrame::Hello {
            protocol_version: "1".into(),
            veloq_version: "0.5.1".into(),
        };
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &frame)?;
        assert_eq!(bytes.last(), Some(&b'\n'));
        let mut reader = BufReader::new(Cursor::new(bytes));
        let decoded: ClientFrame = read_frame(&mut reader)?;
        assert_eq!(decoded, frame);
        Ok(())
    }

    #[test]
    fn semantic_key_excludes_routing_and_irrelevant_process_context() {
        let invocation = QueryInvocation {
            arguments: [
                "veloq",
                "summary",
                "/tmp/trace.nsys-rep",
                "--daemon",
                "required",
                "--daemon-connect-timeout-ms=9",
                "--format",
                "json",
            ]
            .into_iter()
            .map(|value| EncodedOsString::encode(OsStr::new(value)))
            .collect(),
            cwd: EncodedOsString::encode(OsStr::new("/client/a")),
            environment: Vec::new(),
            terminal_width: Some(120),
        };
        let key =
            invocation.semantic_key(Some(OsStr::new("/tmp/trace.nsys-rep")), OutputFormat::Json);
        let equivalent = QueryInvocation {
            arguments: ["veloq", "summary", "/tmp/trace.nsys-rep", "--daemon=auto"]
                .into_iter()
                .map(|value| EncodedOsString::encode(OsStr::new(value)))
                .collect(),
            cwd: EncodedOsString::encode(OsStr::new("/client/b")),
            environment: Vec::new(),
            terminal_width: Some(80),
        }
        .semantic_key(Some(OsStr::new("/tmp/trace.nsys-rep")), OutputFormat::Json);
        assert_eq!(key, equivalent);

        let relative_a = QueryInvocation {
            arguments: ["veloq", "summary", "trace.nsys-rep"]
                .into_iter()
                .map(|value| EncodedOsString::encode(OsStr::new(value)))
                .collect(),
            cwd: EncodedOsString::encode(OsStr::new("/client/a")),
            environment: Vec::new(),
            terminal_width: Some(120),
        };
        let mut relative_b = relative_a.clone();
        relative_b.cwd = EncodedOsString::encode(OsStr::new("/client/b"));
        assert_ne!(
            relative_a.semantic_key(Some(OsStr::new("trace.nsys-rep")), OutputFormat::Json),
            relative_b.semantic_key(Some(OsStr::new("trace.nsys-rep")), OutputFormat::Json)
        );
        assert_ne!(
            relative_a.semantic_key(Some(OsStr::new("trace.nsys-rep")), OutputFormat::Table),
            QueryInvocation {
                terminal_width: Some(80),
                ..relative_a
            }
            .semantic_key(Some(OsStr::new("trace.nsys-rep")), OutputFormat::Table)
        );
    }

    #[test]
    fn environment_match_requires_the_complete_named_inventory() -> io::Result<()> {
        let captured = QueryInvocation::capture()?;
        assert!(captured.environment_matches_current());

        let mut missing = captured.clone();
        missing.environment.pop();
        assert!(!missing.environment_matches_current());

        let mut renamed = captured;
        renamed
            .environment
            .first_mut()
            .ok_or_else(|| io::Error::other("captured environment inventory is empty"))?
            .name = "UNEXPECTED".to_string();
        assert!(!renamed.environment_matches_current());
        Ok(())
    }

    #[test]
    fn ownership_accepts_output_chunks_only_after_acceptance() -> io::Result<()> {
        let mut rejected = RequestOwnership::new("rejected");
        rejected.mark_transmitted();
        rejected.observe(&ServerFrame::Rejected {
            request_id: "rejected".to_string(),
            error: DaemonError::unsupported("not accepted"),
        })?;
        assert!(rejected.permits_one_shot_fallback());

        let mut ownership = RequestOwnership::new("request");
        ownership.mark_transmitted();
        assert!(!ownership.permits_one_shot_fallback());
        ownership.observe(&ServerFrame::Accepted {
            request_id: "request".to_string(),
        })?;
        ownership.observe(&ServerFrame::OutputChunk {
            request_id: "request".to_string(),
            stream: OutputStream::Stdout,
            bytes: vec![1, 2, 3],
        })?;
        ownership.observe(&ServerFrame::Completed {
            request_id: "request".to_string(),
            exit_code: 0,
        })?;
        assert_eq!(ownership.state(), ExecutionOwnership::Completed);
        Ok(())
    }
}
