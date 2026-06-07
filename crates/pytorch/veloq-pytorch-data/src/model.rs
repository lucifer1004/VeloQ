use crate::SOURCE_KIND;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use veloq_core::{EnvelopeTraceRef, TraceSpan};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSet {
    pub input_path: String,
    pub artifact_dir: String,
    pub fingerprint: InputFingerprint,
    pub trace_span: Option<TimeRange>,
    pub files: Vec<TraceFile>,
    pub events: Vec<Event>,
    pub flows: Vec<FlowEdge>,
    pub links: Vec<EventLink>,
    pub collectives: Vec<CollectiveGroup>,
    pub capabilities: Capabilities,
    pub schema_survey: TraceSchemaSurvey,
}

impl TraceSet {
    pub fn query_trace(&self) -> QueryTrace {
        QueryTrace {
            input_path: self.input_path.clone(),
            artifact_dir: self.artifact_dir.clone(),
            fingerprint: self.fingerprint.clone(),
            trace_span: self.trace_span,
            capabilities: self.capabilities.clone(),
        }
    }

    pub fn trace_ref(&self) -> EnvelopeTraceRef {
        EnvelopeTraceRef {
            kind: SOURCE_KIND,
            path: self.input_path.clone(),
        }
    }

    pub fn envelope_trace_span(&self) -> Option<TraceSpan> {
        self.trace_span.map(|r| TraceSpan {
            origin_ns: r.start_ns,
            span_ns: r.duration_ns,
        })
    }

    pub fn is_multi_rank(&self) -> bool {
        self.capabilities.rank_count > 1
    }

    pub fn event_by_row_id(&self, row_id: &str) -> Option<&Event> {
        self.events.iter().find(|event| event.row_id == row_id)
    }

    pub fn file_by_index(&self, trace_index: u32) -> Option<&TraceFile> {
        self.files
            .iter()
            .find(|file| file.trace_index == trace_index)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryTrace {
    pub input_path: String,
    pub artifact_dir: String,
    pub fingerprint: InputFingerprint,
    pub trace_span: Option<TimeRange>,
    pub capabilities: Capabilities,
}

impl QueryTrace {
    pub fn trace_ref(&self) -> EnvelopeTraceRef {
        EnvelopeTraceRef {
            kind: SOURCE_KIND,
            path: self.input_path.clone(),
        }
    }

    pub fn envelope_trace_span(&self) -> Option<TraceSpan> {
        self.trace_span.map(|r| TraceSpan {
            origin_ns: r.start_ns,
            span_ns: r.duration_ns,
        })
    }

    pub fn is_multi_rank(&self) -> bool {
        self.capabilities.rank_count > 1
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct TimeRange {
    pub start_ns: i64,
    pub end_ns: i64,
    pub duration_ns: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InputFingerprint {
    pub files: Vec<FileFingerprint>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileFingerprint {
    pub path: String,
    pub mtime_secs: i64,
    pub mtime_nanos: u32,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceFile {
    pub key: String,
    pub trace_index: u32,
    pub path: String,
    pub rank: Option<i64>,
    pub worker: Option<String>,
    pub event_count: usize,
    pub schema_version: Option<String>,
    pub cuda_version: Option<String>,
    pub cupti_version: Option<String>,
    pub capture_flags: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EventType {
    CpuOp,
    Annotation,
    Step,
    Runtime,
    Driver,
    Kernel,
    Memcpy,
    Memset,
    Memory,
    Python,
    Comm,
}

impl EventType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CpuOp => "cpu-op",
            Self::Annotation => "annotation",
            Self::Step => "step",
            Self::Runtime => "runtime",
            Self::Driver => "driver",
            Self::Kernel => "kernel",
            Self::Memcpy => "memcpy",
            Self::Memset => "memset",
            Self::Memory => "memory",
            Self::Python => "python",
            Self::Comm => "comm",
        }
    }

    pub fn row_id_prefix(self) -> &'static str {
        match self {
            Self::CpuOp => "cpu_op",
            Self::Annotation => "annotation",
            Self::Step => "step",
            Self::Runtime => "runtime",
            Self::Driver => "driver",
            Self::Kernel => "kernel",
            Self::Memcpy => "memcpy",
            Self::Memset => "memset",
            Self::Memory => "memory",
            Self::Python => "python",
            Self::Comm => "comm",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub key: String,
    pub row_id: String,
    pub stable_index: u64,
    pub trace_index: u32,
    pub original_index: u64,
    pub event_type: EventType,
    pub name: String,
    pub category: Option<String>,
    pub phase: Option<String>,
    pub start_ns: i64,
    pub duration_ns: i64,
    pub end_ns: i64,
    pub rank: Option<i64>,
    pub worker: Option<String>,
    pub pid: Option<i64>,
    pub tid: Option<i64>,
    pub device_id: Option<i64>,
    pub stream_id: Option<i64>,
    pub external_id: Option<i64>,
    pub correlation_id: Option<i64>,
    pub step: Option<i64>,
    pub step_row_id: Option<String>,
    pub python_id: Option<i64>,
    pub python_parent_id: Option<i64>,
    pub python_context_row_id: Option<String>,
    pub python_context_name: Option<String>,
    pub python_context_path: Option<String>,
    pub is_comm: bool,
    pub comm_kind: Option<String>,
    pub bytes: Option<i64>,
    pub shape: Option<String>,
    pub parent_row_id: Option<String>,
    pub children_row_ids: Vec<String>,
    pub args: BTreeMap<String, Value>,
    pub raw: Value,
}

impl Event {
    pub fn is_gpu_activity(&self) -> bool {
        matches!(
            self.event_type,
            EventType::Kernel | EventType::Memcpy | EventType::Memset
        )
    }

    pub fn is_cpu_activity(&self) -> bool {
        !self.is_gpu_activity()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowEdge {
    pub key: String,
    pub flow_id: String,
    pub name: Option<String>,
    pub from_row_id: String,
    pub to_row_id: String,
    pub start_ns: i64,
    pub end_ns: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventLink {
    pub key: String,
    pub from_row_id: String,
    pub to_row_id: String,
    pub kind: String,
    pub confidence: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveGroup {
    pub key: String,
    pub collective_kind: String,
    pub step: Option<i64>,
    pub ordinal: u64,
    pub confidence: String,
    pub start_ns: i64,
    pub end_ns: i64,
    pub duration_ns: i64,
    pub skew_ns: Option<i64>,
    pub slow_rank: Option<i64>,
    pub per_rank: Vec<CollectiveRankTiming>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveRankTiming {
    pub rank: Option<i64>,
    pub row_id: String,
    pub cpu_row_id: Option<String>,
    pub kernel_row_ids: Vec<String>,
    pub event_row_ids: Vec<String>,
    pub name: String,
    pub start_ns: i64,
    pub duration_ns: i64,
    pub end_ns: i64,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Capabilities {
    pub trace_count: usize,
    pub rank_count: usize,
    pub worker_count: usize,
    pub event_count: usize,
    pub active_devices: Vec<i64>,
    pub has_cuda_runtime: bool,
    pub has_cuda_driver: bool,
    pub has_gpu_activity: bool,
    pub has_memory_events: bool,
    pub has_python_events: bool,
    pub has_python_stack: bool,
    pub has_comm_events: bool,
    pub has_steps: bool,
    pub has_flows: bool,
    pub cuda_versions: Vec<String>,
    pub cupti_versions: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PrepState {
    pub input_path: String,
    pub artifact_dir: String,
    pub cache_version: u32,
    pub cache_fresh: bool,
    pub sidecars: Vec<SidecarState>,
    pub schema_survey: Option<TraceSchemaSurvey>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SidecarState {
    pub key: String,
    pub name: String,
    pub path: String,
    pub present: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TraceSchemaSurvey {
    pub raw_event_count: usize,
    pub parsed_event_count: usize,
    pub flow_marker_count: usize,
    pub skipped_event_count: usize,
    pub files: Vec<TraceFileSchemaSurvey>,
    pub phase_counts: BTreeMap<String, usize>,
    pub category_counts: BTreeMap<String, usize>,
    pub event_type_counts: BTreeMap<String, usize>,
    pub arg_key_counts: BTreeMap<String, usize>,
    pub typed_arg_coverage: TypedArgCoverage,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TraceFileSchemaSurvey {
    pub trace_index: u32,
    pub raw_event_count: usize,
    pub parsed_event_count: usize,
    pub flow_marker_count: usize,
    pub skipped_event_count: usize,
    pub top_level_keys: Vec<String>,
    pub has_device_properties: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TypedArgCoverage {
    pub rank: usize,
    pub worker: usize,
    pub device_id: usize,
    pub stream_id: usize,
    pub external_id: usize,
    pub correlation_id: usize,
    pub step: usize,
    pub bytes: usize,
    pub shape: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct FlowMarker {
    pub(crate) trace_index: u32,
    pub(crate) flow_id: String,
    pub(crate) name: Option<String>,
    pub(crate) phase: String,
    pub(crate) start_ns: i64,
    pub(crate) pid: Option<i64>,
    pub(crate) tid: Option<i64>,
}
