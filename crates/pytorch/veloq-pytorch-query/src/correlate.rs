use crate::PytorchQueryResult;
use crate::dto::{CorrelateResponse, CorrelateRow, LinkRef};
use crate::query_sql::{
    event_row::{EventSqlRow, event_sql_row, link_sql_row},
    exec::{self, SqlLabel, SqlVerb},
    inspect as inspect_sql, sidecar,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use veloq_pytorch_data::{PytorchSidecar, QueryTrace};

pub fn correlate(trace: &QueryTrace, row_ids: &[String]) -> PytorchQueryResult<CorrelateResponse> {
    let events_path = sidecar::path(&trace.artifact_dir, PytorchSidecar::Events);
    let links_path = sidecar::path(&trace.artifact_dir, PytorchSidecar::Links);
    let conn = exec::open_connection()?;
    let mut event_cache = BTreeMap::new();
    let mut adjacent_links_cache = BTreeMap::new();
    let rows = row_ids
        .iter()
        .map(|row_id| {
            correlate_one(
                &conn,
                &events_path,
                &links_path,
                row_id,
                &mut event_cache,
                &mut adjacent_links_cache,
            )
        })
        .collect::<PytorchQueryResult<Vec<_>>>()?;
    Ok(CorrelateResponse {
        count: rows.len(),
        total_matched: rows.len(),
        rows,
    })
}

fn correlate_one(
    conn: &duckdb::Connection,
    events_path: &str,
    links_path: &str,
    row_id: &str,
    event_cache: &mut BTreeMap<String, EventSqlRow>,
    adjacent_links_cache: &mut BTreeMap<String, Vec<LinkRef>>,
) -> PytorchQueryResult<CorrelateRow> {
    let seed = load_event_cached(conn, events_path, event_cache, row_id)?;
    let mut row_ids = BTreeSet::new();
    let mut causal_link_keys = BTreeSet::new();
    if let Some(seed) = &seed {
        row_ids.insert(seed.row_id.clone());
        if let Some(parent) = &seed.parent_row_id {
            row_ids.insert(parent.clone());
        }
        if let Some(step) = &seed.step_row_id {
            row_ids.insert(step.clone());
        }
        for child in load_children_for_parent(conn, events_path, event_cache, &seed.row_id)? {
            row_ids.insert(child.row_id);
        }
        expand_causal_links(
            conn,
            events_path,
            links_path,
            event_cache,
            adjacent_links_cache,
            seed,
            &mut row_ids,
            &mut causal_link_keys,
        )?;
    }

    let missing_ids = row_ids
        .iter()
        .filter(|id| !event_cache.contains_key(id.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    event_cache.extend(load_events_by_row_ids(conn, events_path, &missing_ids)?);
    let mut events = row_ids
        .iter()
        .filter_map(|id| event_cache.get(id))
        .map(EventSqlRow::event_ref)
        .collect::<Vec<_>>();
    events.sort_by_key(|event| (event.start_ns, event.row_id.clone()));
    let event_ids = events
        .iter()
        .map(|event| event.row_id.clone())
        .collect::<BTreeSet<_>>();
    let links = load_links_for_event_set(conn, links_path, &event_ids)?
        .into_iter()
        .filter(|link| {
            matches!(link.kind.as_str(), "nesting" | "step") || causal_link_keys.contains(&link.key)
        })
        .collect();
    Ok(CorrelateRow {
        key: row_id.to_string(),
        seed_row_id: row_id.to_string(),
        seed: seed.map(|event| event.event_ref()),
        events,
        links,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "correlate traversal needs shared sidecar caches plus the seed/output sets"
)]
fn expand_causal_links(
    conn: &duckdb::Connection,
    events_path: &str,
    links_path: &str,
    event_cache: &mut BTreeMap<String, EventSqlRow>,
    adjacent_links_cache: &mut BTreeMap<String, Vec<LinkRef>>,
    seed: &EventSqlRow,
    row_ids: &mut BTreeSet<String>,
    causal_link_keys: &mut BTreeSet<String>,
) -> PytorchQueryResult<()> {
    let seed_row_id = seed.row_id.clone();
    let mut queue = initial_queue(seed);
    let mut visited = BTreeSet::new();
    while let Some(state) = queue.pop_front() {
        if !visited.insert(state.clone()) {
            continue;
        }
        let Some(current) = load_event_cached(conn, events_path, event_cache, &state.row_id)?
        else {
            continue;
        };
        let adjacent_links =
            load_adjacent_links_cached(conn, links_path, adjacent_links_cache, &state.row_id)?;
        for link in adjacent_links {
            if !matches!(link.kind.as_str(), "external" | "correlation" | "flow") {
                continue;
            }
            let next = if link.from_row_id == state.row_id {
                Some((link.to_row_id.clone(), LinkDirection::Forward))
            } else if link.to_row_id == state.row_id {
                Some((link.from_row_id.clone(), LinkDirection::Reverse))
            } else {
                None
            };
            if let Some((next, direction)) = next {
                let Some(next_event) = load_event_cached(conn, events_path, event_cache, &next)?
                else {
                    continue;
                };
                if !can_follow_causal_link(
                    &seed_row_id,
                    &current,
                    &next_event,
                    &link.kind,
                    direction,
                    state.intent,
                ) {
                    continue;
                }
                causal_link_keys.insert(link.key.clone());
                row_ids.insert(next.clone());
                include_precise_launch_siblings(
                    conn,
                    events_path,
                    links_path,
                    event_cache,
                    adjacent_links_cache,
                    &next_event,
                    &link.kind,
                    direction,
                    state.intent,
                    row_ids,
                    causal_link_keys,
                )?;
                let next_state = TraversalState {
                    row_id: next,
                    intent: state.intent,
                };
                if !visited.contains(&next_state)
                    && should_expand_event(&seed_row_id, &next_event, state.intent)
                {
                    queue.push_back(next_state);
                }
            }
        }
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "correlate traversal needs shared sidecar caches plus the current edge state"
)]
fn include_precise_launch_siblings(
    conn: &duckdb::Connection,
    events_path: &str,
    links_path: &str,
    event_cache: &mut BTreeMap<String, EventSqlRow>,
    adjacent_links_cache: &mut BTreeMap<String, Vec<LinkRef>>,
    next_event: &EventSqlRow,
    kind: &str,
    direction: LinkDirection,
    intent: TraversalIntent,
    row_ids: &mut BTreeSet<String>,
    causal_link_keys: &mut BTreeSet<String>,
) -> PytorchQueryResult<()> {
    if kind != "correlation"
        || !matches!(
            (direction, intent),
            (LinkDirection::Forward, TraversalIntent::Downstream)
        )
        || !next_event.is_gpu_activity
    {
        return Ok(());
    }

    let sibling_links =
        load_links_to_cached(conn, links_path, adjacent_links_cache, &next_event.row_id)?;
    for sibling_link in sibling_links {
        if sibling_link.kind != "correlation" || sibling_link.to_row_id != next_event.row_id {
            continue;
        }
        let Some(sibling) =
            load_event_cached(conn, events_path, event_cache, &sibling_link.from_row_id)?
        else {
            continue;
        };
        if sibling.is_runtime_or_driver() {
            row_ids.insert(sibling.row_id);
            causal_link_keys.insert(sibling_link.key);
        }
    }
    Ok(())
}

fn initial_queue(seed: &EventSqlRow) -> VecDeque<TraversalState> {
    let mut queue = VecDeque::new();
    if seed.is_gpu_activity {
        queue.push_back(TraversalState::new(&seed.row_id, TraversalIntent::Upstream));
    } else if seed.is_runtime_or_driver() {
        queue.push_back(TraversalState::new(&seed.row_id, TraversalIntent::Upstream));
        queue.push_back(TraversalState::new(
            &seed.row_id,
            TraversalIntent::Downstream,
        ));
    } else {
        queue.push_back(TraversalState::new(
            &seed.row_id,
            TraversalIntent::Downstream,
        ));
    }
    queue
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct TraversalState {
    row_id: String,
    intent: TraversalIntent,
}

impl TraversalState {
    fn new(row_id: &str, intent: TraversalIntent) -> Self {
        Self {
            row_id: row_id.to_string(),
            intent,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum TraversalIntent {
    Upstream,
    Downstream,
}

#[derive(Clone, Copy)]
enum LinkDirection {
    Forward,
    Reverse,
}

fn can_follow_causal_link(
    seed_row_id: &str,
    current: &EventSqlRow,
    next: &EventSqlRow,
    kind: &str,
    direction: LinkDirection,
    intent: TraversalIntent,
) -> bool {
    match kind {
        "correlation" | "flow" => can_follow_precise_link(direction, intent),
        "external" => can_follow_external_link(seed_row_id, current, next, direction, intent),
        _ => false,
    }
}

fn can_follow_precise_link(direction: LinkDirection, intent: TraversalIntent) -> bool {
    matches!(
        (direction, intent),
        (LinkDirection::Forward, TraversalIntent::Downstream)
            | (LinkDirection::Reverse, TraversalIntent::Upstream)
    )
}

fn can_follow_external_link(
    seed_row_id: &str,
    current: &EventSqlRow,
    next: &EventSqlRow,
    direction: LinkDirection,
    intent: TraversalIntent,
) -> bool {
    if is_causal_root(current) && current.row_id != seed_row_id {
        return false;
    }
    match (direction, intent) {
        (LinkDirection::Forward, TraversalIntent::Downstream) => {
            !current.is_gpu_activity
                && !is_non_seed_causal_root(next, seed_row_id)
                && next.start_ns >= current.start_ns
        }
        (LinkDirection::Reverse, TraversalIntent::Upstream) => {
            !is_causal_root(current)
                && (current.row_id == seed_row_id || !current.is_gpu_activity)
                && next.start_ns <= current.start_ns
        }
        _ => false,
    }
}

fn should_expand_event(seed_row_id: &str, event: &EventSqlRow, intent: TraversalIntent) -> bool {
    if event.row_id == seed_row_id {
        return true;
    }
    match intent {
        TraversalIntent::Upstream => !is_causal_root(event),
        TraversalIntent::Downstream => !event.is_gpu_activity && !is_causal_root(event),
    }
}

fn is_causal_root(event: &EventSqlRow) -> bool {
    event.is_cpu_activity() && !event.is_runtime_or_driver()
}

fn is_non_seed_causal_root(event: &EventSqlRow, seed_row_id: &str) -> bool {
    event.row_id != seed_row_id && is_causal_root(event)
}

fn load_event_cached(
    conn: &duckdb::Connection,
    events_path: &str,
    event_cache: &mut BTreeMap<String, EventSqlRow>,
    row_id: &str,
) -> PytorchQueryResult<Option<EventSqlRow>> {
    if let Some(event) = event_cache.get(row_id) {
        return Ok(Some(event.clone()));
    }
    let row_ids = vec![row_id.to_string()];
    let loaded = load_events_by_row_ids(conn, events_path, &row_ids)?;
    let event = loaded.get(row_id).cloned();
    event_cache.extend(loaded);
    Ok(event)
}

fn load_events_by_row_ids(
    conn: &duckdb::Connection,
    events_path: &str,
    row_ids: &[String],
) -> PytorchQueryResult<BTreeMap<String, EventSqlRow>> {
    let Some(query) = inspect_sql::events_by_row_ids_sql(events_path, row_ids) else {
        return Ok(BTreeMap::new());
    };
    let rows = exec::query_rows_on(
        conn,
        &query.sql,
        &query.params,
        SqlLabel::new(SqlVerb::Correlate, "events"),
        event_sql_row,
    )?;
    Ok(rows
        .into_iter()
        .map(|event| (event.row_id.clone(), event))
        .collect())
}

fn load_children_for_parent(
    conn: &duckdb::Connection,
    events_path: &str,
    event_cache: &mut BTreeMap<String, EventSqlRow>,
    parent_row_id: &str,
) -> PytorchQueryResult<Vec<EventSqlRow>> {
    let parent_ids = vec![parent_row_id.to_string()];
    let Some(query) = inspect_sql::children_sql(events_path, &parent_ids) else {
        return Ok(Vec::new());
    };
    let rows = exec::query_rows_on(
        conn,
        &query.sql,
        &query.params,
        SqlLabel::new(SqlVerb::Correlate, "children"),
        event_sql_row,
    )?;
    for child in &rows {
        event_cache.insert(child.row_id.clone(), child.clone());
    }
    Ok(rows)
}

fn load_adjacent_links_cached(
    conn: &duckdb::Connection,
    links_path: &str,
    adjacent_links_cache: &mut BTreeMap<String, Vec<LinkRef>>,
    row_id: &str,
) -> PytorchQueryResult<Vec<LinkRef>> {
    if let Some(links) = adjacent_links_cache.get(row_id) {
        return Ok(links.clone());
    }
    let row_ids = vec![row_id.to_string()];
    let Some(query) = inspect_sql::links_sql(links_path, &row_ids) else {
        return Ok(Vec::new());
    };
    let links = exec::query_rows_on(
        conn,
        &query.sql,
        &query.params,
        SqlLabel::new(SqlVerb::Correlate, "links"),
        link_sql_row,
    )?;
    adjacent_links_cache.insert(row_id.to_string(), links.clone());
    Ok(links)
}

fn load_links_to_cached(
    conn: &duckdb::Connection,
    links_path: &str,
    adjacent_links_cache: &mut BTreeMap<String, Vec<LinkRef>>,
    row_id: &str,
) -> PytorchQueryResult<Vec<LinkRef>> {
    if let Some(links) = adjacent_links_cache.get(row_id) {
        return Ok(links
            .iter()
            .filter(|link| link.to_row_id == row_id)
            .cloned()
            .collect());
    }
    let row_ids = vec![row_id.to_string()];
    let Some(query) = inspect_sql::links_to_sql(links_path, &row_ids) else {
        return Ok(Vec::new());
    };
    exec::query_rows_on(
        conn,
        &query.sql,
        &query.params,
        SqlLabel::new(SqlVerb::Correlate, "links-to"),
        link_sql_row,
    )
}

fn load_links_for_event_set(
    conn: &duckdb::Connection,
    links_path: &str,
    event_ids: &BTreeSet<String>,
) -> PytorchQueryResult<Vec<LinkRef>> {
    let row_ids = event_ids.iter().cloned().collect::<Vec<_>>();
    let Some(query) = inspect_sql::links_sql(links_path, &row_ids) else {
        return Ok(Vec::new());
    };
    let links = exec::query_rows_on(
        conn,
        &query.sql,
        &query.params,
        SqlLabel::new(SqlVerb::Correlate, "result-links"),
        link_sql_row,
    )?;
    Ok(links
        .into_iter()
        .filter(|link| event_ids.contains(&link.from_row_id) && event_ids.contains(&link.to_row_id))
        .collect())
}
