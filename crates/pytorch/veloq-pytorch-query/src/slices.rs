use crate::dto::{SliceAggregateRow, SliceInstanceRow, SliceRow, SlicesAuxiliary, SlicesResponse};
use crate::filter::{EventFilterRequest, limit_ref, require_rank_scope};
use crate::query_sql::{
    exec::{self, SqlLabel, SqlVerb},
    sidecar, slices as slices_sql,
};
use crate::{PytorchQueryError, PytorchQueryResult};
use veloq_pytorch_data::{PytorchSidecar, QueryTrace};
use veloq_query::duckdb::list::{TotalCarrier, count_from_i64, split_rows_and_total};

pub fn slices(
    trace: &QueryTrace,
    request: EventFilterRequest,
    aggregate: bool,
    group_by: Option<String>,
) -> PytorchQueryResult<SlicesResponse> {
    require_rank_scope(trace, request.rank_scope)?;
    slices_sql(trace, request, aggregate, group_by)
}

fn slices_sql(
    trace: &QueryTrace,
    request: EventFilterRequest,
    aggregate: bool,
    group_by: Option<String>,
) -> PytorchQueryResult<SlicesResponse> {
    let events_path = sidecar::path(&trace.artifact_dir, PytorchSidecar::Events);
    if aggregate {
        let group_by = group_by.unwrap_or_else(|| "name".to_string());
        let query = slices_sql::aggregate_sql(&events_path, &request, &group_by)?;
        limit_ref(request.limit)?;
        let (rows, total_matched) = query_aggregate_rows(&query.sql, &query.params)?;
        return Ok(SlicesResponse {
            count: rows.len(),
            total_matched,
            rows: rows.into_iter().map(SliceRow::Aggregate).collect(),
            auxiliary: SlicesAuxiliary {
                scope: request.rank_scope.echo(request.step),
                aggregate,
                group_by: Some(group_by),
                time_window_ns: request.time_window_ns,
            },
        });
    }

    let query = slices_sql::instance_sql(&events_path, &request)?;
    limit_ref(request.limit)?;
    let (rows, total_matched) = query_instance_rows(&query.sql, &query.params)?;
    Ok(SlicesResponse {
        count: rows.len(),
        total_matched,
        rows: rows.into_iter().map(SliceRow::Instance).collect(),
        auxiliary: SlicesAuxiliary {
            scope: request.rank_scope.echo(request.step),
            aggregate,
            group_by: None,
            time_window_ns: request.time_window_ns,
        },
    })
}

struct InstanceSqlRow {
    key: String,
    row_id: String,
    name: String,
    start_ns: i64,
    duration_ns: i64,
    rank: Option<i64>,
    step: Option<i64>,
    child_count: i64,
    attributed_gpu_ns: i64,
    attributed_comm_ns: i64,
    total_matched: i64,
}

struct AggregateSqlRow {
    key: String,
    scope: String,
    instances: i64,
    total_cpu_ns: i64,
    total_gpu_ns: i64,
    total_comm_ns: i64,
    avg_cpu_ns: f64,
    total_matched: i64,
}

fn query_instance_rows(
    sql: &str,
    params: &[duckdb::types::Value],
) -> PytorchQueryResult<(Vec<SliceInstanceRow>, usize)> {
    let raw_rows = exec::query_rows(
        sql,
        params,
        SqlLabel::new(SqlVerb::Slices, "instance"),
        instance_sql_row,
    )?;
    split_rows_and_total::<usize, _, _, _>(
        raw_rows,
        TotalCarrier::First,
        |row| row.total_matched,
        PytorchQueryError::slices_count_overflow,
        instance_from_sql,
    )
}

fn query_aggregate_rows(
    sql: &str,
    params: &[duckdb::types::Value],
) -> PytorchQueryResult<(Vec<SliceAggregateRow>, usize)> {
    let raw_rows = exec::query_rows(
        sql,
        params,
        SqlLabel::new(SqlVerb::Slices, "aggregate"),
        aggregate_sql_row,
    )?;
    split_rows_and_total::<usize, _, _, _>(
        raw_rows,
        TotalCarrier::First,
        |row| row.total_matched,
        PytorchQueryError::slices_count_overflow,
        aggregate_from_sql,
    )
}

fn instance_sql_row(row: &duckdb::Row<'_>) -> Result<InstanceSqlRow, duckdb::Error> {
    Ok(InstanceSqlRow {
        key: row.get("key")?,
        row_id: row.get("row_id")?,
        name: row.get("name")?,
        start_ns: row.get("start_ns")?,
        duration_ns: row.get("duration_ns")?,
        rank: row.get("rank")?,
        step: row.get("step")?,
        child_count: row.get("child_count")?,
        attributed_gpu_ns: row.get("attributed_gpu_ns")?,
        attributed_comm_ns: row.get("attributed_comm_ns")?,
        total_matched: row.get("total_matched")?,
    })
}

fn aggregate_sql_row(row: &duckdb::Row<'_>) -> Result<AggregateSqlRow, duckdb::Error> {
    Ok(AggregateSqlRow {
        key: row.get("key")?,
        scope: row.get("scope")?,
        instances: row.get("instances")?,
        total_cpu_ns: row.get("total_cpu_ns")?,
        total_gpu_ns: row.get("total_gpu_ns")?,
        total_comm_ns: row.get("total_comm_ns")?,
        avg_cpu_ns: row.get("avg_cpu_ns")?,
        total_matched: row.get("total_matched")?,
    })
}

fn instance_from_sql(row: InstanceSqlRow) -> PytorchQueryResult<SliceInstanceRow> {
    Ok(SliceInstanceRow {
        key: row.key,
        row_id: row.row_id,
        name: row.name,
        start_ns: row.start_ns,
        duration_ns: row.duration_ns,
        rank: row.rank,
        step: row.step,
        child_count: usize_count(row.child_count)?,
        attributed_gpu_ns: row.attributed_gpu_ns,
        attributed_comm_ns: row.attributed_comm_ns,
    })
}

fn aggregate_from_sql(row: AggregateSqlRow) -> PytorchQueryResult<SliceAggregateRow> {
    Ok(SliceAggregateRow {
        key: row.key,
        scope: row.scope,
        instances: usize_count(row.instances)?,
        total_cpu_ns: row.total_cpu_ns,
        total_gpu_ns: row.total_gpu_ns,
        total_comm_ns: row.total_comm_ns,
        avg_cpu_ns: row.avg_cpu_ns,
    })
}

fn usize_count(value: i64) -> PytorchQueryResult<usize> {
    count_from_i64(value, PytorchQueryError::slices_count_overflow)
}
