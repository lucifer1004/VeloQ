use crate::PytorchQueryResult;
use crate::dto::{EventListAuxiliary, EventRef, SearchResponse};
use crate::filter::{EventFilterRequest, limit_ref, validate_event_scope};
use crate::query_sql::{
    event_filter,
    exec::{self, SqlLabel, SqlVerb},
    sidecar,
};
use veloq_pytorch_data::{PytorchSidecar, QueryTrace};
use veloq_query::duckdb::list as duckdb_list;

pub fn search(
    trace: &QueryTrace,
    request: EventFilterRequest,
) -> PytorchQueryResult<SearchResponse> {
    validate_event_scope(trace, &request)?;
    limit_ref(request.limit)?;
    let events_path = sidecar::path(&trace.artifact_dir, PytorchSidecar::Events);
    let query = event_filter::search_sql(&events_path, &request)?;
    let (rows, total_matched) = query_search_rows(&query.sql, &query.params)?;
    Ok(SearchResponse {
        count: rows.len(),
        total_matched,
        rows,
        auxiliary: EventListAuxiliary {
            scope: request.rank_scope.echo(request.step),
            time_window_ns: request.time_window_ns,
        },
    })
}

fn query_search_rows(
    sql: &str,
    params: &[duckdb::types::Value],
) -> PytorchQueryResult<(Vec<EventRef>, usize)> {
    let raw_rows = exec::query_rows(
        sql,
        params,
        SqlLabel::new(SqlVerb::Search, "events"),
        search_sql_row,
    )?;
    duckdb_list::split_rows_and_total::<usize, _, _, _>(
        raw_rows,
        duckdb_list::TotalCarrier::First,
        |row| row.total_matched,
        crate::PytorchQueryError::search_count_overflow,
        |row| Ok(row.event),
    )
}

struct SearchSqlRow {
    event: EventRef,
    total_matched: i64,
}

fn search_sql_row(row: &duckdb::Row<'_>) -> Result<SearchSqlRow, duckdb::Error> {
    Ok(SearchSqlRow {
        event: EventRef {
            key: row.get("key")?,
            row_id: row.get("row_id")?,
            event_type: row.get("event_type")?,
            name: row.get("name")?,
            start_ns: row.get("start_ns")?,
            duration_ns: row.get("duration_ns")?,
            rank: row.get("rank")?,
            worker: row.get("worker")?,
            device_id: row.get("device_id")?,
            stream_id: row.get("stream_id")?,
            step: row.get("step")?,
            is_comm: row.get("is_comm")?,
            external_id: row.get("external_id")?,
            correlation_id: row.get("correlation_id")?,
            comm_kind: row.get("comm_kind")?,
            bytes: row.get("bytes")?,
            shape: row.get("shape")?,
        },
        total_matched: row.get("total_matched")?,
    })
}
