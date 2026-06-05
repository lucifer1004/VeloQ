use crate::dto::{StatsAuxiliary, StatsResponse, StatsRow};
use crate::filter::{EventFilterRequest, filtered_events, require_rank_scope};
use anyhow::Result;
use std::collections::BTreeMap;
use veloq_pytorch_data::{Event, TraceSet};

#[derive(Default)]
struct StatsAcc {
    count: usize,
    total_ns: i64,
    min_ns: Option<i64>,
    max_ns: Option<i64>,
}

pub fn stats(
    trace: &TraceSet,
    request: EventFilterRequest,
    group_by: &[String],
) -> Result<StatsResponse> {
    require_rank_scope(trace, request.rank_scope)?;
    validate_group_by(group_by)?;
    let mut groups: BTreeMap<Vec<(String, String)>, StatsAcc> = BTreeMap::new();
    for event in filtered_events(trace, &request)? {
        let axes = group_axes(event, group_by);
        let acc = groups.entry(axes).or_default();
        acc.count += 1;
        acc.total_ns = acc.total_ns.saturating_add(event.duration_ns);
        acc.min_ns = Some(
            acc.min_ns
                .map_or(event.duration_ns, |value| value.min(event.duration_ns)),
        );
        acc.max_ns = Some(
            acc.max_ns
                .map_or(event.duration_ns, |value| value.max(event.duration_ns)),
        );
    }

    let total_matched = groups.len();
    let mut rows = groups
        .into_iter()
        .map(|(axes_vec, acc)| {
            let axes = axes_vec.into_iter().collect::<BTreeMap<_, _>>();
            let count = acc.count;
            let avg_ns = if count == 0 {
                0.0
            } else {
                acc.total_ns as f64 / count as f64
            };
            StatsRow {
                key: stats_key(&axes),
                axes,
                count,
                total_ns: acc.total_ns,
                avg_ns,
                min_ns: acc.min_ns.unwrap_or(0),
                max_ns: acc.max_ns.unwrap_or(0),
            }
        })
        .collect::<Vec<_>>();
    rows.sort_by_key(|row| (std::cmp::Reverse(row.total_ns), row.key.clone()));
    rows.truncate(request.limit);
    Ok(StatsResponse {
        count: rows.len(),
        total_matched,
        rows,
        auxiliary: StatsAuxiliary {
            scope: request.rank_scope.echo(request.step),
            group_by: group_by.to_vec(),
        },
    })
}

fn validate_group_by(group_by: &[String]) -> Result<()> {
    for axis in group_by {
        match axis.as_str() {
            "name" | "type" | "step" | "rank" | "device" | "stream" | "shape" | "comm-kind" => {}
            other => anyhow::bail!(
                "unknown pytorch stats --group-by axis `{other}`; expected name,type,step,rank,device,stream,shape,comm-kind"
            ),
        }
    }
    Ok(())
}

fn group_axes(event: &Event, group_by: &[String]) -> Vec<(String, String)> {
    group_by
        .iter()
        .map(|axis| {
            let value = match axis.as_str() {
                "name" => event.name.clone(),
                "type" => event.event_type.as_str().to_string(),
                "step" => option_i64(event.step),
                "rank" => option_i64(event.rank),
                "device" => option_i64(event.device_id),
                "stream" => option_i64(event.stream_id),
                "shape" => event.shape.clone().unwrap_or_else(|| "none".to_string()),
                "comm-kind" => event
                    .comm_kind
                    .clone()
                    .unwrap_or_else(|| "none".to_string()),
                _ => "unknown".to_string(),
            };
            (axis.clone(), value)
        })
        .collect()
}

fn stats_key(axes: &BTreeMap<String, String>) -> String {
    let suffix = axes
        .iter()
        .map(|(axis, value)| format!("{axis}:{value}"))
        .collect::<Vec<_>>()
        .join("|");
    format!("stats|{suffix}")
}

fn option_i64(value: Option<i64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_string())
}
