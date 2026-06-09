use crate::dto::{StatsAuxiliary, StatsResponse, StatsRow};
use crate::filter::{EventFilterRequest, limit_ref, validate_event_scope};
use crate::query_sql::{
    event_filter,
    exec::{self, SqlLabel, SqlVerb},
    sidecar,
};
use crate::{PytorchQueryError, PytorchQueryResult};
use std::collections::BTreeMap;
use veloq_core::{AxisParentError, AxisUsage};
use veloq_pytorch_data::{Capabilities, PytorchSidecar, QueryTrace};
use veloq_query::duckdb::list::{TotalCarrier, count_from_i64, split_rows_and_total};

const NO_AXES: &[&str] = &[];
const RANK_AXIS: &[&str] = &["rank"];
const DEVICE_AXIS: &[&str] = &["device"];
const RANK_DEVICE_AXES: &[&str] = &["rank", "device"];

pub fn stats(
    trace: &QueryTrace,
    request: EventFilterRequest,
    group_by: &[String],
) -> PytorchQueryResult<StatsResponse> {
    validate_group_by(&trace.capabilities, group_by)?;
    validate_group_by_scope(trace, &request, group_by)?;
    validate_event_scope(trace, &request)?;
    stats_sql(trace, request, group_by)
}

fn stats_sql(
    trace: &QueryTrace,
    request: EventFilterRequest,
    group_by: &[String],
) -> PytorchQueryResult<StatsResponse> {
    limit_ref(request.limit)?;
    let events_path = sidecar::path(&trace.artifact_dir, PytorchSidecar::Events);
    let query = event_filter::stats_sql(&events_path, &request, group_by)?;
    let (rows, total_matched) = query_stats_rows(&query.sql, &query.params, group_by)?;
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

fn query_stats_rows(
    sql: &str,
    params: &[duckdb::types::Value],
    group_by: &[String],
) -> PytorchQueryResult<(Vec<StatsRow>, usize)> {
    let raw_rows = exec::query_rows(
        sql,
        params,
        SqlLabel::new(SqlVerb::Stats, "aggregate"),
        |row| stats_sql_row(row, group_by),
    )?;
    split_rows_and_total::<usize, _, _, _>(
        raw_rows,
        TotalCarrier::First,
        |row| row.total_matched,
        crate::PytorchQueryError::stats_count_overflow,
        stats_row_from_sql,
    )
}

struct StatsSqlRow {
    axes: BTreeMap<String, String>,
    count: i64,
    total_ns: i64,
    avg_ns: f64,
    min_ns: i64,
    max_ns: i64,
    total_matched: i64,
}

fn stats_sql_row(row: &duckdb::Row<'_>, group_by: &[String]) -> Result<StatsSqlRow, duckdb::Error> {
    let mut axes = BTreeMap::new();
    for (idx, axis) in group_by.iter().enumerate() {
        let value: String = row.get(format!("axis_{idx}").as_str())?;
        axes.insert(axis.clone(), value);
    }
    Ok(StatsSqlRow {
        axes,
        count: row.get("count")?,
        total_ns: row.get("total_ns")?,
        avg_ns: row.get("avg_ns")?,
        min_ns: row.get("min_ns")?,
        max_ns: row.get("max_ns")?,
        total_matched: row.get("total_matched")?,
    })
}

fn stats_row_from_sql(row: StatsSqlRow) -> PytorchQueryResult<StatsRow> {
    let count = usize_count(row.count)?;
    Ok(StatsRow {
        key: stats_key(&row.axes),
        axes: row.axes,
        count,
        total_ns: row.total_ns,
        avg_ns: row.avg_ns,
        min_ns: row.min_ns,
        max_ns: row.max_ns,
    })
}

fn usize_count(value: i64) -> PytorchQueryResult<usize> {
    count_from_i64(value, crate::PytorchQueryError::stats_count_overflow)
}

fn validate_group_by(capabilities: &Capabilities, group_by: &[String]) -> PytorchQueryResult<()> {
    for axis in group_by {
        match axis.as_str() {
            "name" | "type" | "step" | "rank" | "device" | "stream" | "shape" | "comm-kind" => {}
            "python-context" | "python-path" => {
                if !capabilities.has_python_stack {
                    return Err(PytorchQueryError::python_stack_missing(axis));
                }
            }
            other => return Err(PytorchQueryError::unknown_stats_group_by(other)),
        }
    }
    Ok(())
}

fn validate_group_by_scope(
    trace: &QueryTrace,
    request: &EventFilterRequest,
    group_by: &[String],
) -> PytorchQueryResult<()> {
    let has_device = has_axis(group_by, "device");
    let has_stream = has_axis(group_by, "stream");
    let usage = stats_axis_usage(trace, request, group_by);

    if has_device && let Err(err) = usage.validate_projection("device", RANK_AXIS) {
        return Err(stats_group_by_parent_error(&err));
    }

    if has_stream && let Err(err) = usage.validate_projection("stream", RANK_DEVICE_AXES) {
        return Err(stats_group_by_parent_error(&err));
    }

    Ok(())
}

fn stats_axis_usage<'a>(
    trace: &QueryTrace,
    request: &EventFilterRequest,
    group_by: &[String],
) -> AxisUsage<'a> {
    let fixed = match (
        !trace.is_multi_rank() || request.rank_scope.rank.is_some(),
        request.device.is_some(),
    ) {
        (true, true) => RANK_DEVICE_AXES,
        (true, false) => RANK_AXIS,
        (false, true) => DEVICE_AXIS,
        (false, false) => NO_AXES,
    };
    let projected = match (has_axis(group_by, "rank"), has_axis(group_by, "device")) {
        (true, true) => RANK_DEVICE_AXES,
        (true, false) => RANK_AXIS,
        (false, true) => DEVICE_AXIS,
        (false, false) => NO_AXES,
    };
    AxisUsage::new(fixed, projected)
}

fn stats_group_by_parent_error(err: &AxisParentError) -> PytorchQueryError {
    match (
        err.axis(),
        err.missing_contains("rank"),
        err.missing_contains("device"),
    ) {
        ("device", true, _) => {
            PytorchQueryError::stats_group_by_parent_required("device", "rank", "rank,device")
        }
        ("stream", true, true) => PytorchQueryError::stats_group_by_parent_required(
            "stream",
            "rank,device",
            "rank,device,stream",
        ),
        ("stream", true, false) => PytorchQueryError::stats_group_by_parent_required(
            "stream",
            "rank",
            "rank,device,stream",
        ),
        ("stream", false, true) | ("stream", false, false) => {
            PytorchQueryError::stats_group_by_parent_required("stream", "device", "device,stream")
        }
        _ => PytorchQueryError::stats_group_by_parent_required(
            err.axis(),
            "parent axis",
            "the parent axis",
        ),
    }
}

fn has_axis(group_by: &[String], axis: &str) -> bool {
    group_by.iter().any(|candidate| candidate == axis)
}

fn stats_key(axes: &BTreeMap<String, String>) -> String {
    let suffix = axes
        .iter()
        .map(|(axis, value)| format!("{axis}:{value}"))
        .collect::<Vec<_>>()
        .join("|");
    format!("stats|{suffix}")
}
