use crate::cache::artifact_dir;
use crate::classify::{
    classify_event, collective_kind_from_name, is_comm_event, parse_step_number,
};
use crate::index::build_flows;
use crate::input::read_trace_text;
use crate::metadata::{
    capture_flags, infer_rank_from_path, infer_worker_from_path, rank_from_top, version_from_top,
    worker_from_top,
};
use crate::model::{
    Capabilities, Event, EventType, FlowMarker, InputFingerprint, TraceFile, TraceSet,
};
use crate::survey::TraceSchemaSurveyBuilder;
use crate::value::{
    args_map, int_from_args, string_field, string_from_args, top_value_string, value_to_i64,
    value_to_ns, value_to_string,
};
use crate::{PytorchDataError, PytorchDataResult};
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

pub(crate) fn parse_trace_set(
    input: &Path,
    files: &[PathBuf],
    fingerprint: InputFingerprint,
) -> PytorchDataResult<TraceSet> {
    let mut trace_files = Vec::with_capacity(files.len());
    let mut events = Vec::new();
    let mut markers = Vec::new();
    let mut survey = TraceSchemaSurveyBuilder::default();

    for (trace_index_usize, file) in files.iter().enumerate() {
        let trace_index = u32::try_from(trace_index_usize)
            .map_err(|source| PytorchDataError::too_many_trace_files(input, source))?;
        let text = read_trace_text(file)?;
        let root: Value = serde_json::from_str(&text)
            .map_err(|source| PytorchDataError::parse_json(file, source))?;
        let top = root
            .as_object()
            .ok_or(PytorchDataError::TraceRootNotObject)?;
        let trace_events = top
            .get("traceEvents")
            .and_then(Value::as_array)
            .ok_or(PytorchDataError::MissingTraceEvents)?;
        survey.record_file_header(trace_index, top);
        let file_rank = rank_from_top(top).or_else(|| infer_rank_from_path(file));
        let file_worker = worker_from_top(top).or_else(|| infer_worker_from_path(file));
        let mut file_event_count = 0usize;

        for (original_index_usize, raw) in trace_events.iter().enumerate() {
            let raw_obj = raw.as_object();
            survey.record_raw_trace_event(trace_index, raw_obj);
            let Some(raw_obj) = raw_obj else {
                survey.record_skipped_event(trace_index);
                continue;
            };
            let original_index = u64::try_from(original_index_usize)
                .map_err(PytorchDataError::trace_event_index_overflow)?;
            if let Some(marker) = parse_flow_marker(trace_index, raw_obj) {
                survey.record_flow_marker(trace_index);
                markers.push(marker);
                continue;
            }
            let Some(event) = parse_event(
                trace_index,
                original_index,
                raw,
                raw_obj,
                file_rank,
                file_worker.clone(),
                u64::try_from(events.len()).map_err(PytorchDataError::event_count_overflow)?,
            )?
            else {
                survey.record_skipped_event(trace_index);
                continue;
            };
            survey.record_parsed_event(&event);
            file_event_count += 1;
            events.push(event);
        }

        trace_files.push(TraceFile {
            key: format!("trace|{}", file.display()),
            trace_index,
            path: file.display().to_string(),
            rank: file_rank,
            worker: file_worker,
            event_count: file_event_count,
            schema_version: top_value_string(top, &["schemaVersion", "schema_version"]),
            cuda_version: version_from_top(top, "cuda"),
            cupti_version: version_from_top(top, "cupti"),
            capture_flags: capture_flags(top),
        });
    }

    let flows = build_flows(&events, markers);
    Ok(TraceSet {
        input_path: input.display().to_string(),
        artifact_dir: artifact_dir(input).display().to_string(),
        fingerprint,
        trace_span: None,
        files: trace_files,
        events,
        flows,
        links: Vec::new(),
        collectives: Vec::new(),
        capabilities: Capabilities::default(),
        schema_survey: survey.finish(),
    })
}

fn parse_event(
    trace_index: u32,
    original_index: u64,
    raw: &Value,
    obj: &Map<String, Value>,
    file_rank: Option<i64>,
    file_worker: Option<String>,
    stable_index: u64,
) -> PytorchDataResult<Option<Event>> {
    let Some(start_ns) = obj.get("ts").and_then(value_to_ns) else {
        return Ok(None);
    };
    let duration_ns = obj.get("dur").and_then(value_to_ns).unwrap_or(0);
    let end_ns = start_ns.saturating_add(duration_ns);
    let name = string_field(obj, "name").unwrap_or_else(|| "unknown".to_string());
    let category = string_field(obj, "cat");
    let phase = string_field(obj, "ph");
    let args = args_map(obj.get("args"));
    let is_comm = is_comm_event(&name, category.as_deref(), &args);
    let event_type = classify_event(&name, category.as_deref(), &args, is_comm);
    let rank = int_from_args(&args, &["rank", "Rank", "global rank"]).or(file_rank);
    let worker = string_from_args(&args, &["worker", "worker_id", "Worker"])
        .or(file_worker)
        .or_else(|| {
            obj.get("pid")
                .and_then(value_to_i64)
                .map(|pid| format!("pid:{pid}"))
        });
    let device_id = int_from_args(
        &args,
        &[
            "device",
            "device_id",
            "Device Id",
            "device id",
            "Device",
            "cuda device",
        ],
    );
    let stream_id = int_from_args(&args, &["stream", "stream_id", "Stream Id", "stream id"]);
    let external_id = int_from_args(
        &args,
        &[
            "External id",
            "external id",
            "external_id",
            "externalId",
            "External ID",
        ],
    );
    let correlation_id = int_from_args(
        &args,
        &[
            "correlation",
            "Correlation",
            "correlation id",
            "Correlation ID",
            "correlationId",
        ],
    );
    let step = if event_type == EventType::Step {
        parse_step_number(&name)
    } else {
        int_from_args(&args, &["step", "Profiler Step", "profiler_step"])
    };
    let python_id = int_from_args(&args, &["Python id", "Python ID", "python_id"]);
    let python_parent_id = int_from_args(
        &args,
        &["Python parent id", "Python Parent ID", "python_parent_id"],
    );
    let bytes = int_from_args(&args, &["bytes", "Bytes", "Num Bytes", "num_bytes"]);
    let shape = string_from_args(
        &args,
        &[
            "Input Dims",
            "Input dims",
            "Input Shapes",
            "Input shapes",
            "input_shapes",
            "shape",
        ],
    );
    let row_prefix = event_type.row_id_prefix();
    let row_id = format!("{row_prefix}:{stable_index}");
    Ok(Some(Event {
        key: row_id.clone(),
        row_id,
        stable_index,
        trace_index,
        original_index,
        event_type,
        name,
        category,
        phase,
        start_ns,
        duration_ns,
        end_ns,
        rank,
        worker,
        pid: obj.get("pid").and_then(value_to_i64),
        tid: obj.get("tid").and_then(value_to_i64),
        device_id,
        stream_id,
        external_id,
        correlation_id,
        step,
        step_row_id: None,
        python_id,
        python_parent_id,
        python_context_row_id: None,
        python_context_name: None,
        python_context_path: None,
        is_comm,
        comm_kind: if is_comm {
            Some(collective_kind_from_name(
                string_field(obj, "name").as_deref().unwrap_or(""),
            ))
        } else {
            None
        },
        bytes,
        shape,
        parent_row_id: None,
        children_row_ids: Vec::new(),
        args,
        raw: raw.clone(),
    }))
}

fn parse_flow_marker(trace_index: u32, obj: &Map<String, Value>) -> Option<FlowMarker> {
    let phase = string_field(obj, "ph")?;
    if !matches!(phase.as_str(), "s" | "t" | "f") {
        return None;
    }
    let start_ns = obj.get("ts").and_then(value_to_ns)?;
    Some(FlowMarker {
        trace_index,
        flow_id: flow_id_from_obj(obj)?,
        name: string_field(obj, "name"),
        phase,
        start_ns,
        pid: obj.get("pid").and_then(value_to_i64),
        tid: obj.get("tid").and_then(value_to_i64),
    })
}

fn flow_id_from_obj(obj: &Map<String, Value>) -> Option<String> {
    obj.get("id")
        .and_then(value_to_string)
        .or_else(|| obj.get("id2").and_then(flow_id_from_id2))
}

fn flow_id_from_id2(value: &Value) -> Option<String> {
    value
        .as_object()
        .and_then(|obj| obj.get("global").or_else(|| obj.get("local")))
        .and_then(value_to_string)
}
