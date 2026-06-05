use crate::cache::artifact_dir;
use crate::model::{CollectiveGroup, Event, EventLink, FlowEdge, SidecarState, TraceSet};
use anyhow::{Context, Result};
use arrow::array::{ArrayRef, BooleanArray, Int64Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub fn sidecar_states(input: &Path) -> Vec<SidecarState> {
    sidecar_paths(input)
        .into_iter()
        .map(|(name, path)| SidecarState {
            key: format!("sidecar|{name}"),
            name: name.to_string(),
            present: path.exists(),
            path: path.display().to_string(),
        })
        .collect()
}

pub(crate) fn materialize_sidecars(trace_set: &TraceSet) -> Result<()> {
    let root = PathBuf::from(&trace_set.artifact_dir);
    fs::create_dir_all(&root).with_context(|| format!("creating {}", root.display()))?;
    write_events_parquet(&root.join("events.parquet"), &trace_set.events)?;
    write_args_parquet(&root.join("args.parquet"), &trace_set.events)?;
    write_flows_parquet(&root.join("flows.parquet"), &trace_set.flows)?;
    write_links_parquet(&root.join("links.parquet"), &trace_set.links)?;
    write_collectives_parquet(&root.join("collectives.parquet"), &trace_set.collectives)?;
    Ok(())
}

pub(crate) fn sibling_tmp(path: &Path) -> PathBuf {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    PathBuf::from(tmp)
}

fn sidecar_paths(input: &Path) -> Vec<(&'static str, PathBuf)> {
    let root = artifact_dir(input);
    vec![
        ("meta.bin", root.join("meta.bin")),
        ("events.parquet", root.join("events.parquet")),
        ("args.parquet", root.join("args.parquet")),
        ("flows.parquet", root.join("flows.parquet")),
        ("links.parquet", root.join("links.parquet")),
        ("collectives.parquet", root.join("collectives.parquet")),
    ]
}

fn write_events_parquet(path: &Path, events: &[Event]) -> Result<()> {
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
        ],
    )
}

fn write_args_parquet(path: &Path, events: &[Event]) -> Result<()> {
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

fn write_flows_parquet(path: &Path, flows: &[FlowEdge]) -> Result<()> {
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

fn write_links_parquet(path: &Path, links: &[EventLink]) -> Result<()> {
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

fn write_collectives_parquet(path: &Path, collectives: &[CollectiveGroup]) -> Result<()> {
    let mut key = Vec::new();
    let mut kind = Vec::new();
    let mut step = Vec::new();
    let mut ordinal = Vec::new();
    let mut rank_count = Vec::new();
    let mut start = Vec::new();
    let mut duration = Vec::new();
    let mut skew = Vec::new();
    let mut slow_rank = Vec::new();
    for collective in collectives {
        key.push(collective.key.clone());
        kind.push(collective.collective_kind.clone());
        step.push(collective.step);
        ordinal.push(collective.ordinal);
        rank_count.push(i64::try_from(collective.per_rank.len()).unwrap_or(i64::MAX));
        start.push(collective.start_ns);
        duration.push(collective.duration_ns);
        skew.push(collective.skew_ns);
        slow_rank.push(collective.slow_rank);
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
            Field::new("skew_ns", DataType::Int64, false),
            Field::new("slow_rank", DataType::Int64, true),
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
        ],
    )
}

fn write_parquet(path: &Path, schema: Schema, columns: Vec<ArrayRef>) -> Result<()> {
    let schema = Arc::new(schema);
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns)
        .with_context(|| format!("building parquet batch for {}", path.display()))?;
    let tmp = sibling_tmp(path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let file = fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), None)
        .with_context(|| format!("opening parquet writer for {}", tmp.display()))?;
    writer
        .write(&batch)
        .with_context(|| format!("writing parquet batch {}", tmp.display()))?;
    writer
        .close()
        .with_context(|| format!("closing parquet writer {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| format!("publishing {}", path.display()))?;
    Ok(())
}
