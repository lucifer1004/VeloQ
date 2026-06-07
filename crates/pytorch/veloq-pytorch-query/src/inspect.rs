use crate::dto::{EventDetails, EventRef, InspectResponse, InspectRow, LinkRef, TypedArgs};
use crate::query_sql::{
    event_row::{EventSqlRow, arg_sql_row, event_sql_row, link_sql_row},
    exec::{self, SqlLabel, SqlVerb},
    inspect as inspect_sql, sidecar,
};
use crate::{PytorchQueryError, PytorchQueryResult};
use serde_json::Value;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use veloq_pytorch_data::{PytorchSidecar, QueryTrace};

pub fn inspect(trace: &QueryTrace, row_ids: &[String]) -> PytorchQueryResult<InspectResponse> {
    if row_ids.is_empty() {
        return Ok(InspectResponse {
            count: 0,
            total_matched: 0,
            rows: Vec::new(),
        });
    }

    let events_path = sidecar::path(&trace.artifact_dir, PytorchSidecar::Events);
    let args_path = sidecar::path(&trace.artifact_dir, PytorchSidecar::Args);
    let links_path = sidecar::path(&trace.artifact_dir, PytorchSidecar::Links);
    let conn = exec::open_connection()?;
    let seed_ids = unique_row_ids(row_ids.iter());
    let mut event_cache = load_events_by_row_ids(&conn, &events_path, &seed_ids)?;
    let found_seed_ids = seed_ids
        .iter()
        .filter(|row_id| event_cache.contains_key(row_id.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if found_seed_ids.is_empty() {
        let rows = row_ids
            .iter()
            .map(|row_id| missing_row(row_id))
            .collect::<Vec<_>>();
        return Ok(InspectResponse {
            count: rows.len(),
            total_matched: rows.len(),
            rows,
        });
    }

    let related_ids = direct_related_ids(&event_cache, &found_seed_ids);
    let related_events = load_events_by_row_ids(&conn, &events_path, &related_ids)?;
    event_cache.extend(related_events);

    let children_by_parent = load_children(&conn, &events_path, &found_seed_ids)?;
    for children in children_by_parent.values() {
        for child in children {
            event_cache.insert(child.row_id.clone(), child.clone());
        }
    }
    let args_by_row = load_args(&conn, &args_path, &found_seed_ids)?;
    let links_by_row = load_links(&conn, &links_path, &found_seed_ids)?;
    let mut rows = Vec::new();
    for row_id in row_ids {
        rows.push(inspect_one(
            &conn,
            &events_path,
            row_id,
            &mut event_cache,
            &children_by_parent,
            &args_by_row,
            &links_by_row,
        )?);
    }
    Ok(InspectResponse {
        count: rows.len(),
        total_matched: rows.len(),
        rows,
    })
}

fn inspect_one(
    conn: &duckdb::Connection,
    events_path: &str,
    row_id: &str,
    event_cache: &mut BTreeMap<String, EventSqlRow>,
    children_by_parent: &BTreeMap<String, Vec<EventSqlRow>>,
    args_by_row: &BTreeMap<String, BTreeMap<String, Value>>,
    links_by_row: &BTreeMap<String, Vec<LinkRef>>,
) -> PytorchQueryResult<InspectRow> {
    let Some(event) = event_cache.get(row_id).cloned() else {
        return Ok(missing_row(row_id));
    };
    let parent = event
        .parent_row_id
        .as_deref()
        .and_then(|id| event_cache.get(id))
        .map(EventSqlRow::event_ref);
    let step = event
        .step_row_id
        .as_deref()
        .and_then(|id| event_cache.get(id))
        .map(EventSqlRow::event_ref);
    let python_context = event
        .python_context_row_id
        .as_deref()
        .and_then(|id| event_cache.get(id))
        .map(EventSqlRow::event_ref);
    let python_stack = python_stack(conn, events_path, &event, event_cache)?;
    let children = children_by_parent
        .get(row_id)
        .map(|children| {
            children
                .iter()
                .map(EventSqlRow::event_ref)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let links = links_by_row.get(row_id).cloned().unwrap_or_default();
    let args = args_by_row.get(row_id).cloned().unwrap_or_default();
    Ok(InspectRow {
        key: row_id.to_string(),
        row_id: row_id.to_string(),
        found: true,
        event: Some(EventDetails {
            reference: event.event_ref(),
            trace_index: event.trace_index_u32()?,
            original_index: event.original_index,
            category: event.category.clone(),
            phase: event.phase.clone(),
            pid: event.pid,
            tid: event.tid,
            comm_kind: event.comm_kind.clone(),
            bytes: event.bytes,
            shape: event.shape.clone(),
            args,
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
            python_context,
            python_stack,
            links,
            raw: event.raw_value()?,
        }),
    })
}

fn missing_row(row_id: &str) -> InspectRow {
    InspectRow {
        key: row_id.to_string(),
        row_id: row_id.to_string(),
        found: false,
        event: None,
    }
}

fn python_stack(
    conn: &duckdb::Connection,
    events_path: &str,
    event: &EventSqlRow,
    event_cache: &mut BTreeMap<String, EventSqlRow>,
) -> PytorchQueryResult<Vec<EventRef>> {
    let Some(context_row_id) = event.python_context_row_id.as_deref() else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    let mut current = load_event_cached(conn, events_path, event_cache, context_row_id)?;
    let mut seen = BTreeSet::new();
    while let Some(frame) = current {
        if frame.event_type != "python" || !seen.insert(frame.row_id.clone()) {
            break;
        }
        out.push(frame.event_ref());
        current = if let Some(parent_id) = frame.python_parent_id {
            load_python_parent_cached(conn, events_path, event_cache, &frame, parent_id)?
        } else {
            None
        };
        if current.is_none()
            && let Some(parent_row_id) = frame.parent_row_id.as_deref()
        {
            current = load_event_cached(conn, events_path, event_cache, parent_row_id)?
                .filter(|parent| parent.event_type == "python");
        }
    }
    out.reverse();
    Ok(out)
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

fn load_python_parent_cached(
    conn: &duckdb::Connection,
    events_path: &str,
    event_cache: &mut BTreeMap<String, EventSqlRow>,
    frame: &EventSqlRow,
    parent_id: i64,
) -> PytorchQueryResult<Option<EventSqlRow>> {
    if let Some(parent) = event_cache
        .values()
        .find(|candidate| candidate.matches_python_identity(frame, parent_id))
        .cloned()
    {
        return Ok(Some(parent));
    }
    let query = inspect_sql::python_parent_sql(
        events_path,
        frame.trace_index,
        frame.pid,
        frame.tid,
        parent_id,
    );
    let parent = exec::query_optional_row_on(
        conn,
        &query.sql,
        &query.params,
        SqlLabel::new(SqlVerb::Inspect, "python-parent"),
        event_sql_row,
    )?;
    if let Some(parent) = &parent {
        event_cache.insert(parent.row_id.clone(), parent.clone());
    }
    Ok(parent)
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
        SqlLabel::new(SqlVerb::Inspect, "events"),
        event_sql_row,
    )?;
    Ok(rows
        .into_iter()
        .map(|event| (event.row_id.clone(), event))
        .collect())
}

fn load_children(
    conn: &duckdb::Connection,
    events_path: &str,
    parent_row_ids: &[String],
) -> PytorchQueryResult<BTreeMap<String, Vec<EventSqlRow>>> {
    let Some(query) = inspect_sql::children_sql(events_path, parent_row_ids) else {
        return Ok(BTreeMap::new());
    };
    let rows = exec::query_rows_on(
        conn,
        &query.sql,
        &query.params,
        SqlLabel::new(SqlVerb::Inspect, "children"),
        event_sql_row,
    )?;
    let mut out: BTreeMap<String, Vec<EventSqlRow>> = BTreeMap::new();
    for row in rows {
        if let Some(parent_row_id) = row.parent_row_id.clone() {
            out.entry(parent_row_id).or_default().push(row);
        }
    }
    Ok(out)
}

fn load_links(
    conn: &duckdb::Connection,
    links_path: &str,
    row_ids: &[String],
) -> PytorchQueryResult<BTreeMap<String, Vec<LinkRef>>> {
    let Some(query) = inspect_sql::links_sql(links_path, row_ids) else {
        return Ok(BTreeMap::new());
    };
    let rows = exec::query_rows_on(
        conn,
        &query.sql,
        &query.params,
        SqlLabel::new(SqlVerb::Inspect, "links"),
        link_sql_row,
    )?;
    let row_id_set = row_ids.iter().cloned().collect::<BTreeSet<_>>();
    let mut out: BTreeMap<String, Vec<LinkRef>> = BTreeMap::new();
    for link in rows {
        if row_id_set.contains(&link.from_row_id) {
            out.entry(link.from_row_id.clone())
                .or_default()
                .push(link.clone());
        }
        if link.to_row_id != link.from_row_id && row_id_set.contains(&link.to_row_id) {
            out.entry(link.to_row_id.clone()).or_default().push(link);
        }
    }
    Ok(out)
}

fn load_args(
    conn: &duckdb::Connection,
    args_path: &str,
    row_ids: &[String],
) -> PytorchQueryResult<BTreeMap<String, BTreeMap<String, Value>>> {
    let Some(query) = inspect_sql::args_sql(args_path, row_ids) else {
        return Ok(BTreeMap::new());
    };
    let rows = exec::query_rows_on(
        conn,
        &query.sql,
        &query.params,
        SqlLabel::new(SqlVerb::Inspect, "args"),
        arg_sql_row,
    )?;
    let mut out: BTreeMap<String, BTreeMap<String, Value>> = BTreeMap::new();
    for row in rows {
        let value = serde_json::from_str(&row.arg_json)
            .map_err(|source| PytorchQueryError::inspect_json_decode("arg", source))?;
        out.entry(row.row_id)
            .or_default()
            .insert(row.arg_key, value);
    }
    Ok(out)
}

fn direct_related_ids(
    event_cache: &BTreeMap<String, EventSqlRow>,
    seed_ids: &[String],
) -> Vec<String> {
    let mut ids = BTreeSet::new();
    for seed_id in seed_ids {
        let Some(event) = event_cache.get(seed_id) else {
            continue;
        };
        ids.extend(event.parent_row_id.iter().cloned());
        ids.extend(event.step_row_id.iter().cloned());
        ids.extend(event.python_context_row_id.iter().cloned());
    }
    ids.into_iter().collect()
}

fn unique_row_ids<'a>(row_ids: impl Iterator<Item = &'a String>) -> Vec<String> {
    row_ids
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}
