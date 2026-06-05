use crate::dto::{CorrelateResponse, CorrelateRow, EventRef, LinkRef};
use std::collections::BTreeSet;
use veloq_pytorch_data::{EventType, TraceSet};

pub fn correlate(trace: &TraceSet, row_ids: &[String]) -> CorrelateResponse {
    let rows = row_ids
        .iter()
        .map(|row_id| correlate_one(trace, row_id))
        .collect::<Vec<_>>();
    CorrelateResponse {
        count: rows.len(),
        total_matched: rows.len(),
        rows,
    }
}

fn correlate_one(trace: &TraceSet, row_id: &str) -> CorrelateRow {
    let seed = trace.event_by_row_id(row_id);
    let mut row_ids = BTreeSet::new();
    if let Some(seed) = seed {
        row_ids.insert(seed.row_id.clone());
        if let Some(parent) = &seed.parent_row_id {
            row_ids.insert(parent.clone());
        }
        if let Some(step) = &seed.step_row_id {
            row_ids.insert(step.clone());
        }
        for child in &seed.children_row_ids {
            row_ids.insert(child.clone());
        }
        for event in &trace.events {
            let same_external = seed.external_id.is_some() && seed.external_id == event.external_id;
            let same_correlation =
                seed.correlation_id.is_some() && seed.correlation_id == event.correlation_id;
            let same_step = seed.step.is_some()
                && seed.step == event.step
                && event.event_type == EventType::Step;
            if same_external || same_correlation || same_step {
                row_ids.insert(event.row_id.clone());
            }
        }
        for link in &trace.links {
            if link.from_row_id == seed.row_id {
                row_ids.insert(link.to_row_id.clone());
            }
            if link.to_row_id == seed.row_id {
                row_ids.insert(link.from_row_id.clone());
            }
        }
    }

    let mut events = row_ids
        .iter()
        .filter_map(|id| trace.event_by_row_id(id))
        .map(EventRef::from)
        .collect::<Vec<_>>();
    events.sort_by_key(|event| (event.start_ns, event.row_id.clone()));
    let event_ids = events
        .iter()
        .map(|event| event.row_id.clone())
        .collect::<BTreeSet<_>>();
    let links = trace
        .links
        .iter()
        .filter(|link| event_ids.contains(&link.from_row_id) && event_ids.contains(&link.to_row_id))
        .map(LinkRef::from)
        .collect();
    CorrelateRow {
        key: row_id.to_string(),
        seed_row_id: row_id.to_string(),
        seed: seed.map(EventRef::from),
        events,
        links,
    }
}
