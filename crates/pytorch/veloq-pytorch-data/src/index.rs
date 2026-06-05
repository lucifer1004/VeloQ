use crate::classify::collective_kind_from_name;
use crate::model::{
    Capabilities, CollectiveGroup, CollectiveRankTiming, Event, EventLink, EventType, FlowEdge,
    FlowMarker, TimeRange, TraceSet,
};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn finalize_trace_set(trace_set: &mut TraceSet) {
    assign_step_context(&mut trace_set.events);
    assign_parent_children(&mut trace_set.events);
    trace_set.links = build_links(&trace_set.events, &trace_set.flows);
    trace_set.collectives = build_collectives(&trace_set.events);
    trace_set.trace_span = compute_trace_span(&trace_set.events);
    trace_set.capabilities = compute_capabilities(trace_set);
}

fn compute_trace_span(events: &[Event]) -> Option<TimeRange> {
    let mut start: Option<i64> = None;
    let mut end: Option<i64> = None;
    for event in events {
        start = Some(start.map_or(event.start_ns, |v| v.min(event.start_ns)));
        end = Some(end.map_or(event.end_ns, |v| v.max(event.end_ns)));
    }
    let (Some(start_ns), Some(end_ns)) = (start, end) else {
        return None;
    };
    Some(TimeRange {
        start_ns,
        end_ns,
        duration_ns: end_ns.saturating_sub(start_ns),
    })
}

fn assign_step_context(events: &mut [Event]) {
    let steps: Vec<(Option<i64>, i64, i64, i64, String)> = events
        .iter()
        .filter(|event| event.event_type == EventType::Step)
        .filter_map(|event| {
            event.step.map(|step| {
                (
                    event.rank,
                    event.start_ns,
                    event.end_ns,
                    step,
                    event.row_id.clone(),
                )
            })
        })
        .collect();
    for event in events.iter_mut() {
        if event.event_type == EventType::Step {
            event.step_row_id = Some(event.row_id.clone());
            continue;
        }
        let mut best: Option<(i64, i64, String)> = None;
        for (rank, start_ns, end_ns, step, row_id) in &steps {
            if *rank != event.rank || event.start_ns < *start_ns || event.end_ns > *end_ns {
                continue;
            }
            let span = end_ns.saturating_sub(*start_ns);
            let replace = best
                .as_ref()
                .is_none_or(|(best_span, _, _)| span < *best_span);
            if replace {
                best = Some((span, *step, row_id.clone()));
            }
        }
        if let Some((_span, step, row_id)) = best {
            event.step = Some(step);
            event.step_row_id = Some(row_id);
        }
    }
}

fn assign_parent_children(events: &mut [Event]) {
    let mut indexed: Vec<usize> = events
        .iter()
        .enumerate()
        .filter(|(_, event)| event.is_cpu_activity() && event.duration_ns > 0)
        .map(|(idx, _)| idx)
        .collect();
    indexed.sort_by(|a, b| {
        let Some(ea) = events.get(*a) else {
            return std::cmp::Ordering::Equal;
        };
        let Some(eb) = events.get(*b) else {
            return std::cmp::Ordering::Equal;
        };
        (
            ea.rank,
            ea.tid,
            ea.start_ns,
            std::cmp::Reverse(ea.duration_ns),
            ea.stable_index,
        )
            .cmp(&(
                eb.rank,
                eb.tid,
                eb.start_ns,
                std::cmp::Reverse(eb.duration_ns),
                eb.stable_index,
            ))
    });

    let mut parent_by_idx: BTreeMap<usize, usize> = BTreeMap::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut current_group: Option<(Option<i64>, Option<i64>)> = None;
    for idx in indexed {
        let Some(event) = events.get(idx) else {
            continue;
        };
        let group = (event.rank, event.tid);
        if current_group != Some(group) {
            stack.clear();
            current_group = Some(group);
        }
        while let Some(parent_idx) = stack.last().copied() {
            let Some(parent) = events.get(parent_idx) else {
                let _ = stack.pop();
                continue;
            };
            if parent.end_ns >= event.end_ns && parent.start_ns <= event.start_ns {
                break;
            }
            let _ = stack.pop();
        }
        if let Some(parent_idx) = stack.last().copied()
            && parent_idx != idx
        {
            parent_by_idx.insert(idx, parent_idx);
        }
        stack.push(idx);
    }

    let mut child_edges: Vec<(usize, String)> = Vec::new();
    for (child_idx, parent_idx) in &parent_by_idx {
        let parent_row_id = events.get(*parent_idx).map(|event| event.row_id.clone());
        let child_row_id = events.get(*child_idx).map(|event| event.row_id.clone());
        if let (Some(parent_row_id), Some(child_row_id)) = (parent_row_id, child_row_id) {
            if let Some(child) = events.get_mut(*child_idx) {
                child.parent_row_id = Some(parent_row_id);
            }
            child_edges.push((*parent_idx, child_row_id));
        }
    }
    for (parent_idx, child_row_id) in child_edges {
        if let Some(parent) = events.get_mut(parent_idx) {
            parent.children_row_ids.push(child_row_id);
        }
    }
}

pub(crate) fn build_flows(events: &[Event], markers: Vec<FlowMarker>) -> Vec<FlowEdge> {
    let mut by_flow: BTreeMap<(u32, String), Vec<FlowMarker>> = BTreeMap::new();
    for marker in markers {
        by_flow
            .entry((marker.trace_index, marker.flow_id.clone()))
            .or_default()
            .push(marker);
    }

    let mut out = Vec::new();
    for ((_, flow_id), mut group) in by_flow {
        group.sort_by_key(|marker| (marker.start_ns, marker.phase.clone()));
        let mut previous: Option<(FlowMarker, String)> = None;
        for marker in group {
            let Some(row_id) = nearest_event_for_marker(events, &marker) else {
                continue;
            };
            if let Some((prev_marker, prev_row_id)) = previous
                && prev_row_id != row_id
            {
                let idx = out.len();
                out.push(FlowEdge {
                    key: format!("flow|{flow_id}|{idx}"),
                    flow_id: flow_id.clone(),
                    name: marker.name.clone().or(prev_marker.name),
                    from_row_id: prev_row_id,
                    to_row_id: row_id.clone(),
                    start_ns: prev_marker.start_ns,
                    end_ns: marker.start_ns,
                });
            }
            previous = Some((marker, row_id));
        }
    }
    out
}

fn nearest_event_for_marker(events: &[Event], marker: &FlowMarker) -> Option<String> {
    let mut best: Option<(i64, String)> = None;
    for event in events {
        if event.trace_index != marker.trace_index
            || event.pid != marker.pid
            || event.tid != marker.tid
        {
            continue;
        }
        let contains = event.start_ns <= marker.start_ns && event.end_ns >= marker.start_ns;
        if !contains {
            continue;
        }
        let span = event.duration_ns.max(0);
        let replace = best.as_ref().is_none_or(|(best_span, _)| span < *best_span);
        if replace {
            best = Some((span, event.row_id.clone()));
        }
    }
    best.map(|(_, row_id)| row_id)
}

fn build_links(events: &[Event], flows: &[FlowEdge]) -> Vec<EventLink> {
    let mut triples: BTreeSet<(String, String, String, String)> = BTreeSet::new();

    for event in events {
        if let Some(parent) = &event.parent_row_id {
            triples.insert((
                parent.clone(),
                event.row_id.clone(),
                "nesting".to_string(),
                "interval".to_string(),
            ));
        }
        if let Some(step) = &event.step_row_id
            && step != &event.row_id
        {
            triples.insert((
                step.clone(),
                event.row_id.clone(),
                "step".to_string(),
                "interval".to_string(),
            ));
        }
    }

    link_groups(
        events,
        |event| event.external_id.map(|id| id.to_string()),
        "external",
        "external-id",
        &mut triples,
    );
    link_runtime_to_gpu_by_correlation(events, &mut triples);
    for flow in flows {
        triples.insert((
            flow.from_row_id.clone(),
            flow.to_row_id.clone(),
            "flow".to_string(),
            "ac2g".to_string(),
        ));
    }

    triples
        .into_iter()
        .map(|(from_row_id, to_row_id, kind, confidence)| EventLink {
            key: format!("link|{kind}|{from_row_id}|{to_row_id}"),
            from_row_id,
            to_row_id,
            kind,
            confidence,
        })
        .collect()
}

fn link_groups(
    events: &[Event],
    key_fn: impl Fn(&Event) -> Option<String>,
    kind: &str,
    confidence: &str,
    triples: &mut BTreeSet<(String, String, String, String)>,
) {
    let mut groups: BTreeMap<String, Vec<&Event>> = BTreeMap::new();
    for event in events {
        if let Some(key) = key_fn(event) {
            groups.entry(key).or_default().push(event);
        }
    }
    for (_key, mut group) in groups {
        group.sort_by_key(|event| (event.start_ns, event.stable_index));
        let mut previous: Option<&Event> = None;
        for event in group {
            if let Some(prev) = previous
                && prev.row_id != event.row_id
            {
                triples.insert((
                    prev.row_id.clone(),
                    event.row_id.clone(),
                    kind.to_string(),
                    confidence.to_string(),
                ));
            }
            previous = Some(event);
        }
    }
}

fn link_runtime_to_gpu_by_correlation(
    events: &[Event],
    triples: &mut BTreeSet<(String, String, String, String)>,
) {
    let mut groups: BTreeMap<i64, Vec<&Event>> = BTreeMap::new();
    for event in events {
        if let Some(corr) = event.correlation_id {
            groups.entry(corr).or_default().push(event);
        }
    }
    for (_corr, group) in groups {
        for from in &group {
            if !matches!(from.event_type, EventType::Runtime | EventType::Driver) {
                continue;
            }
            for to in &group {
                if !to.is_gpu_activity() {
                    continue;
                }
                triples.insert((
                    from.row_id.clone(),
                    to.row_id.clone(),
                    "correlation".to_string(),
                    "correlation-id".to_string(),
                ));
            }
        }
    }
}

fn build_collectives(events: &[Event]) -> Vec<CollectiveGroup> {
    let mut per_rank: BTreeMap<(i64, Option<i64>, String), Vec<&Event>> = BTreeMap::new();
    for event in events {
        if !event.is_comm {
            continue;
        }
        let rank = event.rank.unwrap_or(0);
        let kind = event
            .comm_kind
            .clone()
            .unwrap_or_else(|| collective_kind_from_name(&event.name));
        per_rank
            .entry((rank, event.step, kind))
            .or_default()
            .push(event);
    }

    let mut by_collective: BTreeMap<(String, Option<i64>, u64), Vec<CollectiveRankTiming>> =
        BTreeMap::new();
    for ((rank, step, kind), mut events_for_rank) in per_rank {
        events_for_rank.sort_by_key(|event| (event.start_ns, event.stable_index));
        for (ordinal_usize, event) in events_for_rank.into_iter().enumerate() {
            let Ok(ordinal) = u64::try_from(ordinal_usize) else {
                continue;
            };
            by_collective
                .entry((kind.clone(), step, ordinal))
                .or_default()
                .push(CollectiveRankTiming {
                    rank,
                    row_id: event.row_id.clone(),
                    name: event.name.clone(),
                    start_ns: event.start_ns,
                    duration_ns: event.duration_ns,
                    end_ns: event.end_ns,
                });
        }
    }

    let mut out = Vec::new();
    for ((kind, step, ordinal), mut timings) in by_collective {
        timings.sort_by_key(|timing| timing.rank);
        let mut start: Option<i64> = None;
        let mut end: Option<i64> = None;
        let mut first_start: Option<i64> = None;
        let mut last_start: Option<i64> = None;
        let mut slow_rank: Option<(i64, i64)> = None;
        for timing in &timings {
            start = Some(start.map_or(timing.start_ns, |value| value.min(timing.start_ns)));
            end = Some(end.map_or(timing.end_ns, |value| value.max(timing.end_ns)));
            first_start =
                Some(first_start.map_or(timing.start_ns, |value| value.min(timing.start_ns)));
            last_start =
                Some(last_start.map_or(timing.start_ns, |value| value.max(timing.start_ns)));
            let replace = slow_rank
                .as_ref()
                .is_none_or(|(_, duration)| timing.duration_ns > *duration);
            if replace {
                slow_rank = Some((timing.rank, timing.duration_ns));
            }
        }
        let start_ns = start.unwrap_or(0);
        let end_ns = end.unwrap_or(start_ns);
        let skew_ns = match (first_start, last_start) {
            (Some(first), Some(last)) => last.saturating_sub(first),
            _ => 0,
        };
        let step_key = step
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string());
        out.push(CollectiveGroup {
            key: format!("collective|{kind}|step:{step_key}|ordinal:{ordinal}"),
            collective_kind: kind,
            step,
            ordinal,
            confidence: "ordinal".to_string(),
            start_ns,
            end_ns,
            duration_ns: end_ns.saturating_sub(start_ns),
            skew_ns,
            slow_rank: slow_rank.map(|(rank, _)| rank),
            per_rank: timings,
        });
    }
    out.sort_by_key(|group| {
        (
            std::cmp::Reverse(group.duration_ns),
            group.start_ns,
            group.key.clone(),
        )
    });
    out
}

fn compute_capabilities(trace_set: &TraceSet) -> Capabilities {
    let mut ranks = BTreeSet::new();
    let mut workers = BTreeSet::new();
    let mut devices = BTreeSet::new();
    let mut cuda_versions = BTreeSet::new();
    let mut cupti_versions = BTreeSet::new();
    let mut out = Capabilities {
        trace_count: trace_set.files.len(),
        event_count: trace_set.events.len(),
        ..Capabilities::default()
    };
    for file in &trace_set.files {
        if let Some(rank) = file.rank {
            ranks.insert(rank);
        }
        if let Some(worker) = &file.worker {
            workers.insert(worker.clone());
        }
        if let Some(version) = &file.cuda_version {
            cuda_versions.insert(version.clone());
        }
        if let Some(version) = &file.cupti_version {
            cupti_versions.insert(version.clone());
        }
    }
    for event in &trace_set.events {
        if let Some(rank) = event.rank {
            ranks.insert(rank);
        }
        if let Some(worker) = &event.worker {
            workers.insert(worker.clone());
        }
        if let Some(device) = event.device_id {
            devices.insert(device);
        }
        out.has_cuda_runtime |= event.event_type == EventType::Runtime;
        out.has_cuda_driver |= event.event_type == EventType::Driver;
        out.has_gpu_activity |= event.is_gpu_activity();
        out.has_memory_events |= event.event_type == EventType::Memory;
        out.has_python_events |= event.event_type == EventType::Python;
        out.has_comm_events |= event.is_comm;
        out.has_steps |= event.event_type == EventType::Step;
    }
    out.rank_count = ranks
        .len()
        .max(if trace_set.files.len() > 1 { 1 } else { 0 });
    out.worker_count = workers.len();
    out.active_devices = devices.into_iter().collect();
    out.has_flows = !trace_set.flows.is_empty();
    out.cuda_versions = cuda_versions.into_iter().collect();
    out.cupti_versions = cupti_versions.into_iter().collect();
    out
}
