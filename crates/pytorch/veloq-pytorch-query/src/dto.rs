use crate::scope::PytorchScope;
use schemars::JsonSchema;
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use veloq_pytorch_data::{Capabilities, Event, EventLink, TimeRange};

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct SummaryResponse {
    pub count: usize,
    pub total_matched: usize,
    pub rows: Vec<TraceFileRow>,
    pub auxiliary: SummaryAuxiliary,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct TraceFileRow {
    pub key: String,
    pub trace_index: u32,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rank: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker: Option<String>,
    pub event_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct SummaryAuxiliary {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_range_ns: Option<TimeRangeDto>,
    pub capabilities: CapabilitiesDto,
    pub artifact_dir: String,
    pub capture_flags: Vec<CaptureFlagsRow>,
}

#[derive(Debug, Clone, Copy, Serialize, JsonSchema)]
pub struct TimeRangeDto {
    pub start_ns: i64,
    pub end_ns: i64,
    pub duration_ns: i64,
}

impl From<TimeRange> for TimeRangeDto {
    fn from(range: TimeRange) -> Self {
        Self {
            start_ns: range.start_ns,
            end_ns: range.end_ns,
            duration_ns: range.duration_ns,
        }
    }
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct CaptureFlagsRow {
    pub trace_index: u32,
    pub flags: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct CapabilitiesDto {
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
    pub has_comm_events: bool,
    pub has_steps: bool,
    pub has_flows: bool,
    pub cuda_versions: Vec<String>,
    pub cupti_versions: Vec<String>,
}

impl From<&Capabilities> for CapabilitiesDto {
    fn from(capabilities: &Capabilities) -> Self {
        Self {
            trace_count: capabilities.trace_count,
            rank_count: capabilities.rank_count,
            worker_count: capabilities.worker_count,
            event_count: capabilities.event_count,
            active_devices: capabilities.active_devices.clone(),
            has_cuda_runtime: capabilities.has_cuda_runtime,
            has_cuda_driver: capabilities.has_cuda_driver,
            has_gpu_activity: capabilities.has_gpu_activity,
            has_memory_events: capabilities.has_memory_events,
            has_python_events: capabilities.has_python_events,
            has_comm_events: capabilities.has_comm_events,
            has_steps: capabilities.has_steps,
            has_flows: capabilities.has_flows,
            cuda_versions: capabilities.cuda_versions.clone(),
            cupti_versions: capabilities.cupti_versions.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct EventRef {
    pub key: String,
    pub row_id: String,
    #[serde(rename = "type")]
    pub event_type: String,
    pub name: String,
    pub start_ns: i64,
    pub duration_ns: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rank: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<i64>,
    #[serde(skip_serializing_if = "is_false", default)]
    pub is_comm: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external_id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<i64>,
}

impl From<&Event> for EventRef {
    fn from(event: &Event) -> Self {
        Self {
            key: event.key.clone(),
            row_id: event.row_id.clone(),
            event_type: event.event_type.as_str().to_string(),
            name: event.name.clone(),
            start_ns: event.start_ns,
            duration_ns: event.duration_ns,
            rank: event.rank,
            worker: event.worker.clone(),
            device_id: event.device_id,
            stream_id: event.stream_id,
            step: event.step,
            is_comm: event.is_comm,
            external_id: event.external_id,
            correlation_id: event.correlation_id,
        }
    }
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct SearchResponse {
    pub count: usize,
    pub total_matched: usize,
    pub rows: Vec<EventRef>,
    pub auxiliary: EventListAuxiliary,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct EventListAuxiliary {
    pub scope: PytorchScope,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_window_ns: Option<(i64, i64)>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct InspectResponse {
    pub count: usize,
    pub total_matched: usize,
    pub rows: Vec<InspectRow>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct InspectRow {
    pub key: String,
    pub row_id: String,
    pub found: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event: Option<EventDetails>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct EventDetails {
    pub reference: EventRef,
    pub trace_index: u32,
    pub original_index: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pid: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tid: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comm_kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shape: Option<String>,
    pub args: BTreeMap<String, Value>,
    pub typed_args: TypedArgs,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<EventRef>,
    pub children: Vec<EventRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<EventRef>,
    pub links: Vec<LinkRef>,
    pub raw: Value,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct TypedArgs {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external_id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rank: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<i64>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct LinkRef {
    pub key: String,
    pub from_row_id: String,
    pub to_row_id: String,
    pub kind: String,
    pub confidence: String,
}

impl From<&EventLink> for LinkRef {
    fn from(link: &EventLink) -> Self {
        Self {
            key: link.key.clone(),
            from_row_id: link.from_row_id.clone(),
            to_row_id: link.to_row_id.clone(),
            kind: link.kind.clone(),
            confidence: link.confidence.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct StatsResponse {
    pub count: usize,
    pub total_matched: usize,
    pub rows: Vec<StatsRow>,
    pub auxiliary: StatsAuxiliary,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct StatsAuxiliary {
    pub scope: PytorchScope,
    pub group_by: Vec<String>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct StatsRow {
    pub key: String,
    pub axes: BTreeMap<String, String>,
    pub count: usize,
    pub total_ns: i64,
    pub avg_ns: f64,
    pub min_ns: i64,
    pub max_ns: i64,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct CorrelateResponse {
    pub count: usize,
    pub total_matched: usize,
    pub rows: Vec<CorrelateRow>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct CorrelateRow {
    pub key: String,
    pub seed_row_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<EventRef>,
    pub events: Vec<EventRef>,
    pub links: Vec<LinkRef>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct TimelineResponse {
    pub count: usize,
    pub total_matched: usize,
    pub rows: Vec<TimelineBucketRow>,
    pub auxiliary: TimelineAuxiliary,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct TimelineAuxiliary {
    pub scope: PytorchScope,
    pub interval_ns: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_window_ns: Option<(i64, i64)>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct TimelineBucketRow {
    pub key: String,
    pub start_ns: i64,
    pub end_ns: i64,
    pub cpu_ns: i64,
    pub gpu_ns: i64,
    pub comm_ns: i64,
    pub event_count: usize,
    pub by_type_ns: BTreeMap<String, i64>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct SlicesResponse {
    pub count: usize,
    pub total_matched: usize,
    pub rows: Vec<SliceRow>,
    pub auxiliary: SlicesAuxiliary,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct SlicesAuxiliary {
    pub scope: PytorchScope,
    pub aggregate: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group_by: Option<String>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum SliceRow {
    Instance(SliceInstanceRow),
    Aggregate(SliceAggregateRow),
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct SliceInstanceRow {
    pub key: String,
    pub row_id: String,
    pub name: String,
    pub start_ns: i64,
    pub duration_ns: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rank: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<i64>,
    pub child_count: usize,
    pub attributed_gpu_ns: i64,
    pub attributed_comm_ns: i64,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct SliceAggregateRow {
    pub key: String,
    pub scope: String,
    pub instances: usize,
    pub total_cpu_ns: i64,
    pub total_gpu_ns: i64,
    pub total_comm_ns: i64,
    pub avg_cpu_ns: f64,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct CollectivesResponse {
    pub count: usize,
    pub total_matched: usize,
    pub rows: Vec<CollectiveRow>,
    pub auxiliary: CollectivesAuxiliary,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct CollectivesAuxiliary {
    pub scope: PytorchScope,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct CollectiveRow {
    pub key: String,
    pub collective_kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<i64>,
    pub ordinal: u64,
    pub confidence: String,
    pub start_ns: i64,
    pub duration_ns: i64,
    pub skew_ns: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slow_rank: Option<i64>,
    pub per_rank: Vec<CollectiveRankRow>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct CollectiveRankRow {
    pub rank: i64,
    pub row_id: String,
    pub name: String,
    pub start_ns: i64,
    pub duration_ns: i64,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct PrepResponse {
    pub count: usize,
    pub total_matched: usize,
    pub rows: Vec<PrepRow>,
    pub auxiliary: PrepAuxiliary,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct PrepRow {
    pub key: String,
    pub name: String,
    pub path: String,
    pub present: bool,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct PrepAuxiliary {
    pub input_path: String,
    pub artifact_dir: String,
    pub cache_version: u32,
    pub cache_fresh: bool,
    pub built: bool,
}

fn is_false(value: &bool) -> bool {
    !*value
}
