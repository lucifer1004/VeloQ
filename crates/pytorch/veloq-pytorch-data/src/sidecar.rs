use crate::PytorchDataResult;
use crate::cache::artifact_dir;
use crate::model::{CollectiveGroup, Event, EventLink, FlowEdge, SidecarState, TraceSet};
use arrow::array::{ArrayRef, BooleanArray, Int64Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use veloq_data::parquet::write_record_batch_atomic;

const EVENTS_COLUMNS: &[&str] = &[
    "key",
    "row_id",
    "stable_index",
    "type",
    "name",
    "start_ns",
    "duration_ns",
    "end_ns",
    "rank",
    "worker",
    "device_id",
    "stream_id",
    "step",
    "is_comm",
    "external_id",
    "correlation_id",
    "trace_index",
    "original_index",
    "category",
    "phase",
    "pid",
    "tid",
    "comm_kind",
    "bytes",
    "shape",
    "parent_row_id",
    "step_row_id",
    "python_context_row_id",
    "python_context_name",
    "python_context_path",
    "python_id",
    "python_parent_id",
    "is_gpu_activity",
    "raw_json",
];

const ARGS_COLUMNS: &[&str] = &["row_id", "arg_key", "arg_json"];
const FLOWS_COLUMNS: &[&str] = &[
    "key",
    "flow_id",
    "name",
    "from_row_id",
    "to_row_id",
    "start_ns",
    "end_ns",
];
const LINKS_COLUMNS: &[&str] = &["key", "from_row_id", "to_row_id", "kind", "confidence"];
const COLLECTIVES_COLUMNS: &[&str] = &[
    "key",
    "collective_kind",
    "step",
    "ordinal",
    "rank_count",
    "start_ns",
    "duration_ns",
    "skew_ns",
    "slow_rank",
    "confidence",
    "rank_ordinal",
    "rank",
    "row_id",
    "cpu_row_id",
    "kernel_row_ids",
    "event_row_ids",
    "name",
    "rank_start_ns",
    "rank_duration_ns",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PytorchSidecar {
    Meta,
    Events,
    Args,
    Flows,
    Links,
    Collectives,
}

impl PytorchSidecar {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Meta => "meta.bin",
            Self::Events => "events.parquet",
            Self::Args => "args.parquet",
            Self::Flows => "flows.parquet",
            Self::Links => "links.parquet",
            Self::Collectives => "collectives.parquet",
        }
    }

    const fn required_columns(self) -> &'static [&'static str] {
        match self {
            Self::Meta => &[],
            Self::Events => EVENTS_COLUMNS,
            Self::Args => ARGS_COLUMNS,
            Self::Flows => FLOWS_COLUMNS,
            Self::Links => LINKS_COLUMNS,
            Self::Collectives => COLLECTIVES_COLUMNS,
        }
    }
}

const ALL_SIDECARS: &[PytorchSidecar] = &[
    PytorchSidecar::Meta,
    PytorchSidecar::Events,
    PytorchSidecar::Args,
    PytorchSidecar::Flows,
    PytorchSidecar::Links,
    PytorchSidecar::Collectives,
];

const QUERY_SIDECARS: &[PytorchSidecar] = &[
    PytorchSidecar::Events,
    PytorchSidecar::Args,
    PytorchSidecar::Links,
    PytorchSidecar::Collectives,
];

pub fn sidecar_path_for_artifact(
    artifact_dir: impl AsRef<Path>,
    sidecar: PytorchSidecar,
) -> PathBuf {
    artifact_dir.as_ref().join(sidecar.name())
}

pub fn sidecar_states(input: &Path) -> Vec<SidecarState> {
    sidecar_paths(input)
        .into_iter()
        .map(|(sidecar, path)| {
            let name = sidecar.name();
            SidecarState {
                key: format!("sidecar|{name}"),
                name: name.to_string(),
                present: sidecar_is_ready(sidecar, &path),
                path: path.display().to_string(),
            }
        })
        .collect()
}

pub(crate) fn sidecars_ready(input: &Path) -> bool {
    sidecar_paths(input)
        .into_iter()
        .all(|(sidecar, path)| sidecar_is_ready(sidecar, &path))
}

pub(crate) fn query_sidecars_ready(input: &Path) -> bool {
    sidecar_paths_for(input, QUERY_SIDECARS)
        .into_iter()
        .all(|(sidecar, path)| sidecar_is_ready(sidecar, &path))
}

pub(crate) fn materialize_sidecars(trace_set: &TraceSet) -> PytorchDataResult<()> {
    let root = Path::new(&trace_set.artifact_dir);
    fs::create_dir_all(root).map_err(|source| veloq_data::DataError::create_dir(root, source))?;
    write_events_parquet(
        &sidecar_path_for_artifact(root, PytorchSidecar::Events),
        &trace_set.events,
    )?;
    write_args_parquet(
        &sidecar_path_for_artifact(root, PytorchSidecar::Args),
        &trace_set.events,
    )?;
    write_flows_parquet(
        &sidecar_path_for_artifact(root, PytorchSidecar::Flows),
        &trace_set.flows,
    )?;
    write_links_parquet(
        &sidecar_path_for_artifact(root, PytorchSidecar::Links),
        &trace_set.links,
    )?;
    write_collectives_parquet(
        &sidecar_path_for_artifact(root, PytorchSidecar::Collectives),
        &trace_set.collectives,
    )?;
    Ok(())
}

fn sidecar_paths(input: &Path) -> Vec<(PytorchSidecar, PathBuf)> {
    sidecar_paths_for(input, ALL_SIDECARS)
}

fn sidecar_paths_for(input: &Path, sidecars: &[PytorchSidecar]) -> Vec<(PytorchSidecar, PathBuf)> {
    let root = artifact_dir(input);
    sidecars
        .iter()
        .map(|sidecar| (*sidecar, sidecar_path_for_artifact(&root, *sidecar)))
        .collect()
}

fn sidecar_is_ready(sidecar: PytorchSidecar, path: &Path) -> bool {
    let required_columns = sidecar.required_columns();
    if required_columns.is_empty() {
        return path.is_file();
    }
    parquet_has_required_columns(sidecar, path, required_columns)
}

fn parquet_has_required_columns(
    sidecar: PytorchSidecar,
    path: &Path,
    required_columns: &[&str],
) -> bool {
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(err) => {
            if err.kind() == std::io::ErrorKind::NotFound {
                return false;
            }
            log::warn!(
                "pytorch {} sidecar is not readable at {}: {err}",
                sidecar.name(),
                path.display()
            );
            return false;
        }
    };
    let builder = match ParquetRecordBatchReaderBuilder::try_new(file) {
        Ok(builder) => builder,
        Err(err) => {
            log::warn!(
                "pytorch {} sidecar is not valid parquet at {}: {err}",
                sidecar.name(),
                path.display()
            );
            return false;
        }
    };
    let schema = builder.schema();
    for column in required_columns {
        if schema.field_with_name(column).is_err() {
            log::warn!(
                "pytorch {} sidecar at {} is missing required column {column}",
                sidecar.name(),
                path.display()
            );
            return false;
        }
    }
    true
}

fn write_events_parquet(path: &Path, events: &[Event]) -> PytorchDataResult<()> {
    let mut key = Vec::new();
    let mut row_id = Vec::new();
    let mut stable_index = Vec::new();
    let mut event_type = Vec::new();
    let mut name = Vec::new();
    let mut start_ns = Vec::new();
    let mut duration_ns = Vec::new();
    let mut end_ns = Vec::new();
    let mut rank = Vec::new();
    let mut worker = Vec::new();
    let mut device_id = Vec::new();
    let mut stream_id = Vec::new();
    let mut step = Vec::new();
    let mut is_comm = Vec::new();
    let mut external_id = Vec::new();
    let mut correlation_id = Vec::new();
    let mut trace_index = Vec::new();
    let mut original_index = Vec::new();
    let mut category = Vec::new();
    let mut phase = Vec::new();
    let mut pid = Vec::new();
    let mut tid = Vec::new();
    let mut comm_kind = Vec::new();
    let mut bytes = Vec::new();
    let mut shape = Vec::new();
    let mut parent_row_id = Vec::new();
    let mut step_row_id = Vec::new();
    let mut python_context_row_id = Vec::new();
    let mut python_context_name = Vec::new();
    let mut python_context_path = Vec::new();
    let mut python_id = Vec::new();
    let mut python_parent_id = Vec::new();
    let mut is_gpu_activity = Vec::new();
    let mut raw_json = Vec::new();
    for event in events {
        key.push(event.key.clone());
        row_id.push(event.row_id.clone());
        stable_index.push(event.stable_index);
        event_type.push(event.event_type.as_str().to_string());
        name.push(event.name.clone());
        start_ns.push(event.start_ns);
        duration_ns.push(event.duration_ns);
        end_ns.push(event.end_ns);
        rank.push(event.rank);
        worker.push(event.worker.clone());
        device_id.push(event.device_id);
        stream_id.push(event.stream_id);
        step.push(event.step);
        is_comm.push(event.is_comm);
        external_id.push(event.external_id);
        correlation_id.push(event.correlation_id);
        trace_index.push(i64::from(event.trace_index));
        original_index.push(event.original_index);
        category.push(event.category.clone());
        phase.push(event.phase.clone());
        pid.push(event.pid);
        tid.push(event.tid);
        comm_kind.push(event.comm_kind.clone());
        bytes.push(event.bytes);
        shape.push(event.shape.clone());
        parent_row_id.push(event.parent_row_id.clone());
        step_row_id.push(event.step_row_id.clone());
        python_context_row_id.push(event.python_context_row_id.clone());
        python_context_name.push(event.python_context_name.clone());
        python_context_path.push(event.python_context_path.clone());
        python_id.push(event.python_id);
        python_parent_id.push(event.python_parent_id);
        is_gpu_activity.push(event.is_gpu_activity());
        raw_json.push(event.raw.to_string());
    }
    write_parquet(
        path,
        Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("row_id", DataType::Utf8, false),
            Field::new("stable_index", DataType::UInt64, false),
            Field::new("type", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("start_ns", DataType::Int64, false),
            Field::new("duration_ns", DataType::Int64, false),
            Field::new("end_ns", DataType::Int64, false),
            Field::new("rank", DataType::Int64, true),
            Field::new("worker", DataType::Utf8, true),
            Field::new("device_id", DataType::Int64, true),
            Field::new("stream_id", DataType::Int64, true),
            Field::new("step", DataType::Int64, true),
            Field::new("is_comm", DataType::Boolean, false),
            Field::new("external_id", DataType::Int64, true),
            Field::new("correlation_id", DataType::Int64, true),
            Field::new("trace_index", DataType::Int64, false),
            Field::new("original_index", DataType::UInt64, false),
            Field::new("category", DataType::Utf8, true),
            Field::new("phase", DataType::Utf8, true),
            Field::new("pid", DataType::Int64, true),
            Field::new("tid", DataType::Int64, true),
            Field::new("comm_kind", DataType::Utf8, true),
            Field::new("bytes", DataType::Int64, true),
            Field::new("shape", DataType::Utf8, true),
            Field::new("parent_row_id", DataType::Utf8, true),
            Field::new("step_row_id", DataType::Utf8, true),
            Field::new("python_context_row_id", DataType::Utf8, true),
            Field::new("python_context_name", DataType::Utf8, true),
            Field::new("python_context_path", DataType::Utf8, true),
            Field::new("python_id", DataType::Int64, true),
            Field::new("python_parent_id", DataType::Int64, true),
            Field::new("is_gpu_activity", DataType::Boolean, false),
            Field::new("raw_json", DataType::Utf8, false),
        ]),
        vec![
            Arc::new(StringArray::from(key)),
            Arc::new(StringArray::from(row_id)),
            Arc::new(UInt64Array::from(stable_index)),
            Arc::new(StringArray::from(event_type)),
            Arc::new(StringArray::from(name)),
            Arc::new(Int64Array::from(start_ns)),
            Arc::new(Int64Array::from(duration_ns)),
            Arc::new(Int64Array::from(end_ns)),
            Arc::new(Int64Array::from(rank)),
            Arc::new(StringArray::from(worker)),
            Arc::new(Int64Array::from(device_id)),
            Arc::new(Int64Array::from(stream_id)),
            Arc::new(Int64Array::from(step)),
            Arc::new(BooleanArray::from(is_comm)),
            Arc::new(Int64Array::from(external_id)),
            Arc::new(Int64Array::from(correlation_id)),
            Arc::new(Int64Array::from(trace_index)),
            Arc::new(UInt64Array::from(original_index)),
            Arc::new(StringArray::from(category)),
            Arc::new(StringArray::from(phase)),
            Arc::new(Int64Array::from(pid)),
            Arc::new(Int64Array::from(tid)),
            Arc::new(StringArray::from(comm_kind)),
            Arc::new(Int64Array::from(bytes)),
            Arc::new(StringArray::from(shape)),
            Arc::new(StringArray::from(parent_row_id)),
            Arc::new(StringArray::from(step_row_id)),
            Arc::new(StringArray::from(python_context_row_id)),
            Arc::new(StringArray::from(python_context_name)),
            Arc::new(StringArray::from(python_context_path)),
            Arc::new(Int64Array::from(python_id)),
            Arc::new(Int64Array::from(python_parent_id)),
            Arc::new(BooleanArray::from(is_gpu_activity)),
            Arc::new(StringArray::from(raw_json)),
        ],
    )
}

fn write_args_parquet(path: &Path, events: &[Event]) -> PytorchDataResult<()> {
    let mut row_id = Vec::new();
    let mut arg_key = Vec::new();
    let mut arg_json = Vec::new();
    for event in events {
        for (key, value) in &event.args {
            row_id.push(event.row_id.clone());
            arg_key.push(key.clone());
            arg_json.push(value.to_string());
        }
    }
    write_parquet(
        path,
        Schema::new(vec![
            Field::new("row_id", DataType::Utf8, false),
            Field::new("arg_key", DataType::Utf8, false),
            Field::new("arg_json", DataType::Utf8, false),
        ]),
        vec![
            Arc::new(StringArray::from(row_id)),
            Arc::new(StringArray::from(arg_key)),
            Arc::new(StringArray::from(arg_json)),
        ],
    )
}

fn write_flows_parquet(path: &Path, flows: &[FlowEdge]) -> PytorchDataResult<()> {
    let mut key = Vec::new();
    let mut flow_id = Vec::new();
    let mut name = Vec::new();
    let mut from = Vec::new();
    let mut to = Vec::new();
    let mut start = Vec::new();
    let mut end = Vec::new();
    for flow in flows {
        key.push(flow.key.clone());
        flow_id.push(flow.flow_id.clone());
        name.push(flow.name.clone());
        from.push(flow.from_row_id.clone());
        to.push(flow.to_row_id.clone());
        start.push(flow.start_ns);
        end.push(flow.end_ns);
    }
    write_parquet(
        path,
        Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("flow_id", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("from_row_id", DataType::Utf8, false),
            Field::new("to_row_id", DataType::Utf8, false),
            Field::new("start_ns", DataType::Int64, false),
            Field::new("end_ns", DataType::Int64, false),
        ]),
        vec![
            Arc::new(StringArray::from(key)),
            Arc::new(StringArray::from(flow_id)),
            Arc::new(StringArray::from(name)),
            Arc::new(StringArray::from(from)),
            Arc::new(StringArray::from(to)),
            Arc::new(Int64Array::from(start)),
            Arc::new(Int64Array::from(end)),
        ],
    )
}

fn write_links_parquet(path: &Path, links: &[EventLink]) -> PytorchDataResult<()> {
    let mut key = Vec::new();
    let mut from = Vec::new();
    let mut to = Vec::new();
    let mut kind = Vec::new();
    let mut confidence = Vec::new();
    for link in links {
        key.push(link.key.clone());
        from.push(link.from_row_id.clone());
        to.push(link.to_row_id.clone());
        kind.push(link.kind.clone());
        confidence.push(link.confidence.clone());
    }
    write_parquet(
        path,
        Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("from_row_id", DataType::Utf8, false),
            Field::new("to_row_id", DataType::Utf8, false),
            Field::new("kind", DataType::Utf8, false),
            Field::new("confidence", DataType::Utf8, false),
        ]),
        vec![
            Arc::new(StringArray::from(key)),
            Arc::new(StringArray::from(from)),
            Arc::new(StringArray::from(to)),
            Arc::new(StringArray::from(kind)),
            Arc::new(StringArray::from(confidence)),
        ],
    )
}

fn write_collectives_parquet(
    path: &Path,
    collectives: &[CollectiveGroup],
) -> PytorchDataResult<()> {
    let mut key = Vec::new();
    let mut kind = Vec::new();
    let mut step = Vec::new();
    let mut ordinal = Vec::new();
    let mut rank_count = Vec::new();
    let mut start = Vec::new();
    let mut duration = Vec::new();
    let mut skew = Vec::new();
    let mut slow_rank = Vec::new();
    let mut confidence = Vec::new();
    let mut rank_ordinal = Vec::new();
    let mut rank = Vec::new();
    let mut row_id = Vec::new();
    let mut cpu_row_id = Vec::new();
    let mut kernel_row_ids = Vec::new();
    let mut event_row_ids = Vec::new();
    let mut name = Vec::new();
    let mut rank_start = Vec::new();
    let mut rank_duration = Vec::new();
    for collective in collectives {
        for (timing_idx, timing) in collective.per_rank.iter().enumerate() {
            key.push(collective.key.clone());
            kind.push(collective.collective_kind.clone());
            step.push(collective.step);
            ordinal.push(collective.ordinal);
            rank_count.push(i64::try_from(collective.per_rank.len()).unwrap_or(i64::MAX));
            start.push(collective.start_ns);
            duration.push(collective.duration_ns);
            skew.push(collective.skew_ns);
            slow_rank.push(collective.slow_rank);
            confidence.push(collective.confidence.clone());
            rank_ordinal.push(u64::try_from(timing_idx).unwrap_or(u64::MAX));
            rank.push(timing.rank);
            row_id.push(timing.row_id.clone());
            cpu_row_id.push(timing.cpu_row_id.clone());
            kernel_row_ids.push(timing.kernel_row_ids.join(","));
            event_row_ids.push(timing.event_row_ids.join(","));
            name.push(timing.name.clone());
            rank_start.push(timing.start_ns);
            rank_duration.push(timing.duration_ns);
        }
    }
    write_parquet(
        path,
        Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("collective_kind", DataType::Utf8, false),
            Field::new("step", DataType::Int64, true),
            Field::new("ordinal", DataType::UInt64, false),
            Field::new("rank_count", DataType::Int64, false),
            Field::new("start_ns", DataType::Int64, false),
            Field::new("duration_ns", DataType::Int64, false),
            Field::new("skew_ns", DataType::Int64, true),
            Field::new("slow_rank", DataType::Int64, true),
            Field::new("confidence", DataType::Utf8, false),
            Field::new("rank_ordinal", DataType::UInt64, false),
            Field::new("rank", DataType::Int64, true),
            Field::new("row_id", DataType::Utf8, false),
            Field::new("cpu_row_id", DataType::Utf8, true),
            Field::new("kernel_row_ids", DataType::Utf8, false),
            Field::new("event_row_ids", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("rank_start_ns", DataType::Int64, false),
            Field::new("rank_duration_ns", DataType::Int64, false),
        ]),
        vec![
            Arc::new(StringArray::from(key)),
            Arc::new(StringArray::from(kind)),
            Arc::new(Int64Array::from(step)),
            Arc::new(UInt64Array::from(ordinal)),
            Arc::new(Int64Array::from(rank_count)),
            Arc::new(Int64Array::from(start)),
            Arc::new(Int64Array::from(duration)),
            Arc::new(Int64Array::from(skew)),
            Arc::new(Int64Array::from(slow_rank)),
            Arc::new(StringArray::from(confidence)),
            Arc::new(UInt64Array::from(rank_ordinal)),
            Arc::new(Int64Array::from(rank)),
            Arc::new(StringArray::from(row_id)),
            Arc::new(StringArray::from(cpu_row_id)),
            Arc::new(StringArray::from(kernel_row_ids)),
            Arc::new(StringArray::from(event_row_ids)),
            Arc::new(StringArray::from(name)),
            Arc::new(Int64Array::from(rank_start)),
            Arc::new(Int64Array::from(rank_duration)),
        ],
    )
}

fn write_parquet(path: &Path, schema: Schema, columns: Vec<ArrayRef>) -> PytorchDataResult<()> {
    Ok(write_record_batch_atomic(path, schema, columns, None)?)
}
