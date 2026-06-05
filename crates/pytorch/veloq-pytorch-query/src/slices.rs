use crate::dto::{SliceAggregateRow, SliceInstanceRow, SliceRow, SlicesAuxiliary, SlicesResponse};
use crate::filter::{CompiledFilters, EventFilterRequest, event_matches_scope, require_rank_scope};
use anyhow::Result;
use std::collections::BTreeMap;
use veloq_pytorch_data::{Event, EventType, TraceSet};

pub fn slices(
    trace: &TraceSet,
    request: EventFilterRequest,
    aggregate: bool,
    group_by: Option<String>,
) -> Result<SlicesResponse> {
    require_rank_scope(trace, request.rank_scope)?;
    let compiled = CompiledFilters::new(&request)?;
    let mut instances = trace
        .events
        .iter()
        .filter(|event| matches!(event.event_type, EventType::Step | EventType::Annotation))
        .filter(|event| event_matches_scope(event, &request))
        .filter(|event| compiled.matches_name(&event.name))
        .map(|event| slice_instance(trace, event))
        .collect::<Vec<_>>();
    instances.sort_by_key(|row| (row.start_ns, row.row_id.clone()));
    let total_matched = instances.len();
    if aggregate {
        let group_by = group_by.unwrap_or_else(|| "name".to_string());
        let mut groups: BTreeMap<String, Vec<SliceInstanceRow>> = BTreeMap::new();
        for row in instances {
            let key = if group_by == "step" {
                row.step
                    .map(|step| step.to_string())
                    .unwrap_or_else(|| "none".to_string())
            } else {
                row.name.clone()
            };
            groups.entry(key).or_default().push(row);
        }
        let mut rows = Vec::new();
        for (scope, rows_for_scope) in groups {
            rows.push(SliceRow::Aggregate(slice_aggregate(scope, rows_for_scope)));
        }
        rows.sort_by_key(|row| match row {
            SliceRow::Aggregate(row) => (std::cmp::Reverse(row.total_cpu_ns), row.key.clone()),
            SliceRow::Instance(row) => (std::cmp::Reverse(row.duration_ns), row.key.clone()),
        });
        rows.truncate(request.limit);
        return Ok(SlicesResponse {
            count: rows.len(),
            total_matched,
            rows,
            auxiliary: SlicesAuxiliary {
                scope: request.rank_scope.echo(request.step),
                aggregate,
                group_by: Some(group_by),
            },
        });
    }
    instances.truncate(request.limit);
    Ok(SlicesResponse {
        count: instances.len(),
        total_matched,
        rows: instances.into_iter().map(SliceRow::Instance).collect(),
        auxiliary: SlicesAuxiliary {
            scope: request.rank_scope.echo(request.step),
            aggregate,
            group_by: None,
        },
    })
}

fn slice_instance(trace: &TraceSet, event: &Event) -> SliceInstanceRow {
    let mut gpu_ns = 0i64;
    let mut comm_ns = 0i64;
    for candidate in &trace.events {
        if candidate.row_id == event.row_id || candidate.rank != event.rank {
            continue;
        }
        if candidate.start_ns < event.start_ns || candidate.end_ns > event.end_ns {
            continue;
        }
        if candidate.is_gpu_activity() {
            gpu_ns = gpu_ns.saturating_add(candidate.duration_ns);
        }
        if candidate.is_comm {
            comm_ns = comm_ns.saturating_add(candidate.duration_ns);
        }
    }
    SliceInstanceRow {
        key: format!("slice|{}|@{}", event.name, event.start_ns),
        row_id: event.row_id.clone(),
        name: event.name.clone(),
        start_ns: event.start_ns,
        duration_ns: event.duration_ns,
        rank: event.rank,
        step: event.step,
        child_count: event.children_row_ids.len(),
        attributed_gpu_ns: gpu_ns,
        attributed_comm_ns: comm_ns,
    }
}

fn slice_aggregate(scope: String, rows: Vec<SliceInstanceRow>) -> SliceAggregateRow {
    let instances = rows.len();
    let mut total_cpu_ns = 0i64;
    let mut total_gpu_ns = 0i64;
    let mut total_comm_ns = 0i64;
    for row in &rows {
        total_cpu_ns = total_cpu_ns.saturating_add(row.duration_ns);
        total_gpu_ns = total_gpu_ns.saturating_add(row.attributed_gpu_ns);
        total_comm_ns = total_comm_ns.saturating_add(row.attributed_comm_ns);
    }
    let avg_cpu_ns = if instances == 0 {
        0.0
    } else {
        total_cpu_ns as f64 / instances as f64
    };
    SliceAggregateRow {
        key: format!("scope|{scope}"),
        scope,
        instances,
        total_cpu_ns,
        total_gpu_ns,
        total_comm_ns,
        avg_cpu_ns,
    }
}
