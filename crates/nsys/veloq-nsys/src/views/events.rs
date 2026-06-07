use veloq_core::tabular::{TabularView, cell_opt};
use veloq_nsys_query::{EventRef, correlate::CorrelateResponse, inspect::InspectResponse};

pub fn inspect_view(data: &InspectResponse) -> TabularView {
    // Per-event-kind formatting lives on `EventDetails::summary_row`,
    // next to the detail structs. This module's job is just to drop
    // each `SummaryRow` into the TabularView grid.
    let mut v = TabularView::new(vec![
        "row_id",
        "type",
        "name",
        "start_ns",
        "duration_ns",
        "device_id",
        "stream_id",
        "details",
    ]);
    for e in &data.rows {
        let s = e.summary_row();
        v.push_row(vec![
            s.row_id,
            s.kind.to_string(),
            s.name,
            s.start_ns,
            s.duration_ns,
            cell_opt(s.device_id),
            cell_opt(s.stream_id),
            s.details,
        ]);
    }
    v
}

pub fn correlate_view(data: &CorrelateResponse) -> TabularView {
    // One row per related event, prefixed with the originating row_id so
    // the agent (or human) can group sub-events back to their parent
    // input. cpu/gpu side discriminator stays as a column.
    let mut v = TabularView::new(vec![
        "input_row_id",
        "side",
        "row_id",
        "name",
        "start_ns",
        "duration_ns",
        "device_id",
        "stream_id",
        "global_tid",
        "correlation_id",
        "synthetic_id",
    ]);
    for r in &data.rows {
        let parent = r.row_id.to_string();
        let corr = r.correlation_id.map(|n| n.to_string()).unwrap_or_default();
        let syn = r.synthetic_id.clone().unwrap_or_default();
        let aux = &r.auxiliary;
        push_event_rows(&mut v, &parent, "cpu", &corr, &syn, &aux.cpu_events);
        push_event_rows(&mut v, &parent, "gpu", &corr, &syn, &aux.gpu_events);
        push_event_rows(&mut v, &parent, "sync", &corr, &syn, &aux.sync_events);
        push_event_rows(&mut v, &parent, "graph", &corr, &syn, &aux.graph_events);
        if aux.cpu_events.is_empty()
            && aux.gpu_events.is_empty()
            && aux.sync_events.is_empty()
            && aux.graph_events.is_empty()
        {
            v.push_row(vec![
                parent,
                "self".to_string(),
                r.row_id.to_string(),
                if r.correlation_found {
                    "(no related events)".into()
                } else {
                    "(no correlation)".into()
                },
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                corr,
                syn,
            ]);
        }
    }
    v
}

fn push_event_rows(
    v: &mut TabularView,
    parent: &str,
    side: &str,
    corr: &str,
    syn: &str,
    events: &[EventRef],
) {
    for e in events {
        let b = e.base();
        v.push_row(vec![
            parent.to_string(),
            side.to_string(),
            b.row_id.to_string(),
            b.name.clone(),
            b.start_ns.to_string(),
            b.duration_ns.to_string(),
            cell_opt(b.device_id),
            cell_opt(b.stream_id),
            cell_opt(b.global_tid),
            corr.to_string(),
            syn.to_string(),
        ]);
    }
}
