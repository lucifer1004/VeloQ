use crate::dto::{EventDetails, EventRef, InspectResponse, InspectRow, LinkRef, TypedArgs};
use veloq_pytorch_data::TraceSet;

pub fn inspect(trace: &TraceSet, row_ids: &[String]) -> InspectResponse {
    let rows = row_ids
        .iter()
        .map(|row_id| inspect_one(trace, row_id))
        .collect::<Vec<_>>();
    InspectResponse {
        count: rows.len(),
        total_matched: rows.len(),
        rows,
    }
}

fn inspect_one(trace: &TraceSet, row_id: &str) -> InspectRow {
    let Some(event) = trace.event_by_row_id(row_id) else {
        return InspectRow {
            key: row_id.to_string(),
            row_id: row_id.to_string(),
            found: false,
            event: None,
        };
    };
    let parent = event
        .parent_row_id
        .as_deref()
        .and_then(|id| trace.event_by_row_id(id))
        .map(EventRef::from);
    let step = event
        .step_row_id
        .as_deref()
        .and_then(|id| trace.event_by_row_id(id))
        .map(EventRef::from);
    let children = event
        .children_row_ids
        .iter()
        .filter_map(|id| trace.event_by_row_id(id))
        .map(EventRef::from)
        .collect();
    let links = trace
        .links
        .iter()
        .filter(|link| link.from_row_id == event.row_id || link.to_row_id == event.row_id)
        .map(LinkRef::from)
        .collect();
    InspectRow {
        key: row_id.to_string(),
        row_id: row_id.to_string(),
        found: true,
        event: Some(EventDetails {
            reference: EventRef::from(event),
            trace_index: event.trace_index,
            original_index: event.original_index,
            category: event.category.clone(),
            phase: event.phase.clone(),
            pid: event.pid,
            tid: event.tid,
            comm_kind: event.comm_kind.clone(),
            bytes: event.bytes,
            shape: event.shape.clone(),
            args: event.args.clone(),
            typed_args: TypedArgs {
                external_id: event.external_id,
                correlation_id: event.correlation_id,
                device_id: event.device_id,
                stream_id: event.stream_id,
                rank: event.rank,
                step: event.step,
            },
            parent,
            children,
            step,
            links,
            raw: event.raw.clone(),
        }),
    }
}
