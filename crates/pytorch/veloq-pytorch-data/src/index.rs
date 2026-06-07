use crate::classify::collective_kind_from_name;
use crate::model::{
    Capabilities, CollectiveGroup, CollectiveRankTiming, Event, EventLink, EventType, FlowEdge,
    FlowMarker, TimeRange, TraceSet,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub(crate) fn finalize_trace_set(trace_set: &mut TraceSet) {
    assign_step_context(&mut trace_set.events);
    assign_parent_children(&mut trace_set.events);
    assign_python_context(&mut trace_set.events);
    trace_set.links = build_links(&trace_set.events, &trace_set.flows);
    trace_set.collectives = build_collectives(&trace_set.events, &trace_set.links);
    trace_set.trace_span = compute_trace_span(&trace_set.events);
    trace_set.capabilities = compute_capabilities(trace_set);
}

fn assign_python_context(events: &mut [Event]) {
    let parent_by_idx = python_parent_indices(events);
    let context_by_idx = nearest_python_context_indices(events);
    let paths = python_context_paths(events, &parent_by_idx);
    let context_rows = context_by_idx
        .iter()
        .filter_map(|(idx, context_idx)| {
            let context = events.get(*context_idx)?;
            Some((
                *idx,
                context.row_id.clone(),
                context.name.clone(),
                paths
                    .get(context_idx)
                    .cloned()
                    .unwrap_or_else(|| context.name.clone()),
            ))
        })
        .collect::<Vec<_>>();
    for (idx, row_id, name, path) in context_rows {
        if let Some(event) = events.get_mut(idx) {
            event.python_context_row_id = Some(row_id);
            event.python_context_name = Some(name);
            event.python_context_path = Some(path);
        }
    }
}

fn python_parent_indices(events: &[Event]) -> BTreeMap<usize, usize> {
    let mut by_python_id: BTreeMap<(u32, Option<i64>, Option<i64>, i64), usize> = BTreeMap::new();
    for (idx, event) in events.iter().enumerate() {
        if event.event_type != EventType::Python {
            continue;
        }
        if let Some(python_id) = event.python_id {
            by_python_id.insert((event.trace_index, event.pid, event.tid, python_id), idx);
        }
    }

    let mut out = BTreeMap::new();
    for (idx, event) in events.iter().enumerate() {
        if event.event_type != EventType::Python {
            continue;
        }
        let Some(parent_id) = event.python_parent_id else {
            continue;
        };
        let Some(parent_idx) =
            by_python_id.get(&(event.trace_index, event.pid, event.tid, parent_id))
        else {
            continue;
        };
        if *parent_idx != idx {
            out.insert(idx, *parent_idx);
        }
    }
    out
}

fn nearest_python_context_indices(events: &[Event]) -> BTreeMap<usize, usize> {
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
            ea.trace_index,
            ea.pid,
            ea.tid,
            ea.start_ns,
            std::cmp::Reverse(ea.duration_ns),
            ea.stable_index,
        )
            .cmp(&(
                eb.trace_index,
                eb.pid,
                eb.tid,
                eb.start_ns,
                std::cmp::Reverse(eb.duration_ns),
                eb.stable_index,
            ))
    });

    let mut out = BTreeMap::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut current_group: Option<(u32, Option<i64>, Option<i64>)> = None;
    for idx in indexed {
        let Some(event) = events.get(idx) else {
            continue;
        };
        let group = (event.trace_index, event.pid, event.tid);
        if current_group != Some(group) {
            stack.clear();
            current_group = Some(group);
        }
        while let Some(parent_idx) = stack.last().copied() {
            let Some(parent) = events.get(parent_idx) else {
                let _ = stack.pop();
                continue;
            };
            if parent.start_ns <= event.start_ns && parent.end_ns >= event.end_ns {
                break;
            }
            let _ = stack.pop();
        }
        if event.event_type == EventType::Python {
            out.insert(idx, idx);
            stack.push(idx);
        } else if let Some(context_idx) = stack.last().copied() {
            out.insert(idx, context_idx);
        }
    }
    out
}

fn python_context_paths(
    events: &[Event],
    parent_by_idx: &BTreeMap<usize, usize>,
) -> BTreeMap<usize, String> {
    let row_to_idx = events
        .iter()
        .enumerate()
        .map(|(idx, event)| (event.row_id.clone(), idx))
        .collect::<BTreeMap<_, _>>();
    let mut out = BTreeMap::new();
    for (idx, event) in events.iter().enumerate() {
        if event.event_type != EventType::Python {
            continue;
        }
        let path = python_stack_indices(idx, events, parent_by_idx, &row_to_idx)
            .into_iter()
            .filter_map(|frame_idx| events.get(frame_idx).map(|frame| frame.name.clone()))
            .collect::<Vec<_>>()
            .join(" > ");
        out.insert(idx, path);
    }
    out
}

fn python_stack_indices(
    context_idx: usize,
    events: &[Event],
    parent_by_idx: &BTreeMap<usize, usize>,
    row_to_idx: &BTreeMap<String, usize>,
) -> Vec<usize> {
    let mut stack = Vec::new();
    let mut seen = BTreeSet::new();
    let mut current = Some(context_idx);
    while let Some(idx) = current {
        if !seen.insert(idx) {
            break;
        }
        let Some(event) = events.get(idx) else {
            break;
        };
        if event.event_type != EventType::Python {
            break;
        }
        stack.push(idx);
        current = parent_by_idx
            .get(&idx)
            .copied()
            .or_else(|| python_interval_parent_idx(event, row_to_idx, events));
    }
    stack.reverse();
    stack
}

fn python_interval_parent_idx(
    event: &Event,
    row_to_idx: &BTreeMap<String, usize>,
    events: &[Event],
) -> Option<usize> {
    let parent = event.parent_row_id.as_ref()?;
    let parent_idx = row_to_idx.get(parent).copied()?;
    events
        .get(parent_idx)
        .filter(|parent| parent.event_type == EventType::Python)
        .map(|_| parent_idx)
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

fn ranks_compatible(left: Option<i64>, right: Option<i64>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left == right,
        _ => true,
    }
}

fn ranks_linkable_in_group(
    left: Option<i64>,
    right: Option<i64>,
    known_ranks: &BTreeSet<i64>,
) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left == right,
        (None, None) => true,
        _ => known_ranks.len() <= 1,
    }
}

fn assign_step_context(events: &mut [Event]) {
    let steps: Vec<(u32, Option<i64>, i64, i64, i64, String)> = events
        .iter()
        .filter(|event| event.event_type == EventType::Step)
        .filter_map(|event| {
            event.step.map(|step| {
                (
                    event.trace_index,
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
        for (trace_index, rank, start_ns, end_ns, step, row_id) in &steps {
            if *trace_index != event.trace_index
                || !ranks_compatible(*rank, event.rank)
                || event.start_ns < *start_ns
                || event.end_ns > *end_ns
            {
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
            ea.trace_index,
            ea.rank,
            ea.tid,
            ea.start_ns,
            std::cmp::Reverse(ea.duration_ns),
            ea.stable_index,
        )
            .cmp(&(
                eb.trace_index,
                eb.rank,
                eb.tid,
                eb.start_ns,
                std::cmp::Reverse(eb.duration_ns),
                eb.stable_index,
            ))
    });

    let mut parent_by_idx: BTreeMap<usize, usize> = BTreeMap::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut current_group: Option<(u32, Option<i64>, Option<i64>)> = None;
    for idx in indexed {
        let Some(event) = events.get(idx) else {
            continue;
        };
        let group = (event.trace_index, event.rank, event.tid);
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
        |event| event.external_id.map(|id| scoped_numeric_key(event, id)),
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
        let known_ranks = group
            .iter()
            .filter_map(|event| event.rank)
            .collect::<BTreeSet<_>>();
        let mut previous_by_rank: BTreeMap<Option<i64>, &Event> = BTreeMap::new();
        for event in group {
            if let Some(prev) = previous_by_rank.get(&event.rank).copied() {
                insert_rank_link(prev, event, &known_ranks, kind, confidence, triples);
            }
            match event.rank {
                Some(_) => {
                    if let Some(prev) = previous_by_rank.get(&None).copied() {
                        insert_rank_link(prev, event, &known_ranks, kind, confidence, triples);
                    }
                }
                None => {
                    if let Some(rank) = known_ranks.iter().next()
                        && let Some(prev) = previous_by_rank.get(&Some(*rank)).copied()
                    {
                        insert_rank_link(prev, event, &known_ranks, kind, confidence, triples);
                    }
                }
            }
            previous_by_rank.insert(event.rank, event);
        }
    }
}

fn insert_rank_link(
    prev: &Event,
    event: &Event,
    known_ranks: &BTreeSet<i64>,
    kind: &str,
    confidence: &str,
    triples: &mut BTreeSet<(String, String, String, String)>,
) {
    if prev.row_id == event.row_id || !ranks_linkable_in_group(prev.rank, event.rank, known_ranks) {
        return;
    }
    triples.insert((
        prev.row_id.clone(),
        event.row_id.clone(),
        kind.to_string(),
        confidence.to_string(),
    ));
}

fn link_runtime_to_gpu_by_correlation(
    events: &[Event],
    triples: &mut BTreeSet<(String, String, String, String)>,
) {
    let mut groups: BTreeMap<String, Vec<&Event>> = BTreeMap::new();
    for event in events {
        if let Some(corr) = event.correlation_id {
            groups
                .entry(scoped_numeric_key(event, corr))
                .or_default()
                .push(event);
        }
    }
    for (_corr, group) in groups {
        let known_ranks = group
            .iter()
            .filter_map(|event| event.rank)
            .collect::<BTreeSet<_>>();
        for gpu in group.iter().filter(|event| event.is_gpu_activity()) {
            for launcher_type in [EventType::Runtime, EventType::Driver] {
                let mut best: Option<&Event> = None;
                for launcher in group
                    .iter()
                    .copied()
                    .filter(|event| event.event_type == launcher_type)
                {
                    if launcher.start_ns > gpu.start_ns
                        || !ranks_linkable_in_group(launcher.rank, gpu.rank, &known_ranks)
                    {
                        continue;
                    }
                    let replace = best.as_ref().is_none_or(|current| {
                        (launcher.start_ns, launcher.stable_index)
                            > (current.start_ns, current.stable_index)
                    });
                    if replace {
                        best = Some(launcher);
                    }
                }
                if let Some(from) = best {
                    triples.insert((
                        from.row_id.clone(),
                        gpu.row_id.clone(),
                        "correlation".to_string(),
                        "correlation-id".to_string(),
                    ));
                }
            }
        }
    }
}

fn rank_key(value: Option<i64>) -> String {
    value
        .map(|rank| rank.to_string())
        .unwrap_or_else(|| "none".to_string())
}

fn scoped_numeric_key(event: &Event, id: i64) -> String {
    format!("trace:{}|id:{id}", event.trace_index)
}

fn build_collectives(events: &[Event], links: &[EventLink]) -> Vec<CollectiveGroup> {
    let events_by_row_id = events
        .iter()
        .map(|event| (event.row_id.clone(), event))
        .collect::<BTreeMap<_, _>>();
    let linked_gpu_by_cpu = linked_comm_gpu_by_cpu(events, links, &events_by_row_id);
    let mut primary_by_group: BTreeMap<CollectiveKey, Vec<&Event>> = BTreeMap::new();
    let mut gpu_by_group: BTreeMap<CollectiveKey, Vec<&Event>> = BTreeMap::new();
    for event in events {
        if !event.is_comm {
            continue;
        }
        let key = collective_key(event);
        if is_collective_primary(event) {
            primary_by_group.entry(key).or_default().push(event);
        } else if event.is_gpu_activity() {
            gpu_by_group.entry(key).or_default().push(event);
        }
    }
    for events_for_group in primary_by_group.values_mut() {
        events_for_group.sort_by_key(|event| (event.start_ns, event.stable_index));
    }
    for events_for_group in gpu_by_group.values_mut() {
        events_for_group.sort_by_key(|event| (event.start_ns, event.stable_index));
    }

    let mut used_gpu = BTreeSet::new();
    let mut out = Vec::new();
    for (key, primary_events) in &primary_by_group {
        let fallback_gpu = gpu_by_group.get(key);
        for (ordinal_usize, primary) in primary_events.iter().enumerate() {
            let Ok(ordinal) = u64::try_from(ordinal_usize) else {
                continue;
            };
            let mut kernel_row_ids = linked_gpu_by_cpu
                .get(&primary.row_id)
                .cloned()
                .unwrap_or_default();
            let linked = !kernel_row_ids.is_empty();
            if kernel_row_ids.is_empty()
                && let Some(gpu_events) = fallback_gpu
                && let Some(gpu) = gpu_events.get(ordinal_usize)
            {
                kernel_row_ids.push(gpu.row_id.clone());
            }
            kernel_row_ids.sort();
            kernel_row_ids.dedup();
            for row_id in &kernel_row_ids {
                used_gpu.insert(row_id.clone());
            }
            let confidence = if linked {
                "link"
            } else if kernel_row_ids.is_empty() {
                "cpu-only"
            } else {
                "ordinal"
            };
            out.push(collective_group_from_parts(
                key,
                ordinal,
                primary,
                Some(primary.row_id.clone()),
                kernel_row_ids,
                confidence,
                &events_by_row_id,
            ));
        }
    }

    for (key, gpu_events) in &gpu_by_group {
        let primary_count = primary_by_group.get(key).map(Vec::len).unwrap_or(0);
        let mut kernel_only_count = 0usize;
        for (ordinal_usize, gpu) in gpu_events.iter().enumerate() {
            if used_gpu.contains(&gpu.row_id) {
                continue;
            }
            let ordinal_usize = if primary_count == 0 {
                ordinal_usize
            } else {
                let ordinal = primary_count.saturating_add(kernel_only_count);
                kernel_only_count = kernel_only_count.saturating_add(1);
                ordinal
            };
            let Ok(ordinal) = u64::try_from(ordinal_usize) else {
                continue;
            };
            out.push(collective_group_from_parts(
                key,
                ordinal,
                gpu,
                None,
                vec![gpu.row_id.clone()],
                "kernel-only",
                &events_by_row_id,
            ));
        }
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CollectiveKey {
    rank: Option<i64>,
    step: Option<i64>,
    kind: String,
}

fn collective_key(event: &Event) -> CollectiveKey {
    CollectiveKey {
        rank: event.rank,
        step: event.step,
        kind: event
            .comm_kind
            .clone()
            .unwrap_or_else(|| collective_kind_from_name(&event.name)),
    }
}

fn is_collective_primary(event: &Event) -> bool {
    event.is_comm && matches!(event.event_type, EventType::Comm | EventType::CpuOp)
}

fn linked_comm_gpu_by_cpu(
    events: &[Event],
    links: &[EventLink],
    events_by_row_id: &BTreeMap<String, &Event>,
) -> BTreeMap<String, Vec<String>> {
    let primary_events = events
        .iter()
        .filter(|event| is_collective_primary(event))
        .collect::<Vec<_>>();
    let mut adjacency: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for link in links {
        if !matches!(link.kind.as_str(), "external" | "correlation" | "flow") {
            continue;
        }
        adjacency
            .entry(link.from_row_id.clone())
            .or_default()
            .push(link.to_row_id.clone());
    }

    let mut out = BTreeMap::new();
    for primary in primary_events {
        let primary_key = collective_key(primary);
        let mut queue = VecDeque::from([primary.row_id.clone()]);
        let mut visited = BTreeSet::new();
        let mut gpu_ids = BTreeSet::new();
        while let Some(row_id) = queue.pop_front() {
            if !visited.insert(row_id.clone()) {
                continue;
            }
            if let Some(event) = events_by_row_id.get(&row_id)
                && event.is_comm
                && event.is_gpu_activity()
                && collective_key(event) == primary_key
            {
                gpu_ids.insert(row_id.clone());
            }
            if let Some(next_ids) = adjacency.get(&row_id) {
                for next_id in next_ids {
                    if next_id != &primary.row_id
                        && let Some(next_event) = events_by_row_id.get(next_id)
                        && is_collective_primary(next_event)
                    {
                        continue;
                    }
                    if !visited.contains(next_id) {
                        queue.push_back(next_id.clone());
                    }
                }
            }
        }
        if !gpu_ids.is_empty() {
            out.insert(primary.row_id.clone(), gpu_ids.into_iter().collect());
        }
    }
    out
}

fn collective_group_from_parts(
    key: &CollectiveKey,
    ordinal: u64,
    anchor: &Event,
    cpu_row_id: Option<String>,
    kernel_row_ids: Vec<String>,
    confidence: &str,
    events_by_row_id: &BTreeMap<String, &Event>,
) -> CollectiveGroup {
    let mut event_row_ids = BTreeSet::new();
    if let Some(row_id) = &cpu_row_id {
        event_row_ids.insert(row_id.clone());
    }
    for row_id in &kernel_row_ids {
        event_row_ids.insert(row_id.clone());
    }
    if event_row_ids.is_empty() {
        event_row_ids.insert(anchor.row_id.clone());
    }

    let mut start: Option<i64> = None;
    let mut end: Option<i64> = None;
    for row_id in &event_row_ids {
        if let Some(event) = events_by_row_id.get(row_id) {
            start = Some(start.map_or(event.start_ns, |value| value.min(event.start_ns)));
            end = Some(end.map_or(event.end_ns, |value| value.max(event.end_ns)));
        }
    }
    let start_ns = start.unwrap_or(anchor.start_ns);
    let end_ns = end.unwrap_or(anchor.end_ns);
    let step_key = key
        .step
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_string());
    let rank_key = rank_key(key.rank);
    let timing = CollectiveRankTiming {
        rank: key.rank,
        row_id: anchor.row_id.clone(),
        cpu_row_id,
        kernel_row_ids,
        event_row_ids: event_row_ids.into_iter().collect(),
        name: anchor.name.clone(),
        start_ns,
        duration_ns: end_ns.saturating_sub(start_ns),
        end_ns,
    };
    CollectiveGroup {
        key: format!(
            "collective|{}|rank:{rank_key}|step:{step_key}|ordinal:{ordinal}",
            key.kind
        ),
        collective_kind: key.kind.clone(),
        step: key.step,
        ordinal,
        confidence: confidence.to_string(),
        start_ns,
        end_ns,
        duration_ns: end_ns.saturating_sub(start_ns),
        skew_ns: None,
        slow_rank: None,
        per_rank: vec![timing],
    }
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
        if event.is_gpu_activity()
            && let Some(device) = event.device_id
        {
            devices.insert(device);
        }
        out.has_cuda_runtime |= event.event_type == EventType::Runtime;
        out.has_cuda_driver |= event.event_type == EventType::Driver;
        out.has_gpu_activity |= event.is_gpu_activity();
        out.has_memory_events |= event.event_type == EventType::Memory;
        out.has_python_events |= event.event_type == EventType::Python;
        out.has_python_stack |= event.event_type == EventType::Python && event.python_id.is_some();
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
