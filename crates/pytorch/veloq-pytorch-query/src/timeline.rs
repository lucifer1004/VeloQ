use crate::dto::{TimelineAuxiliary, TimelineBucketRow, TimelineResponse};
use crate::filter::{EventFilterRequest, limit_ref, validate_event_scope};
use crate::query_sql::{
    event_filter,
    exec::{self, SqlLabel, SqlVerb},
    sidecar,
};
use crate::{PytorchQueryError, PytorchQueryResult};
use std::collections::BTreeMap;
use veloq_core::TimelineBucket;
use veloq_pytorch_data::{PytorchSidecar, QueryTrace};
use veloq_query::duckdb::list::{TotalCarrier, count_from_i64, total_matched};

#[derive(Default)]
struct BucketAcc {
    cpu_ns: i64,
    gpu_ns: i64,
    comm_ns: i64,
    event_count: usize,
    by_type_ns: BTreeMap<String, i64>,
}

pub fn timeline(
    trace: &QueryTrace,
    request: EventFilterRequest,
    interval_ns: i64,
) -> PytorchQueryResult<TimelineResponse> {
    validate_event_scope(trace, &request)?;
    if interval_ns <= 0 {
        return Err(PytorchQueryError::IntervalTooSmall);
    }
    timeline_sql(trace, request, interval_ns)
}

fn timeline_sql(
    trace: &QueryTrace,
    request: EventFilterRequest,
    interval_ns: i64,
) -> PytorchQueryResult<TimelineResponse> {
    limit_ref(request.limit)?;
    let origin = trace.trace_span.map(|span| span.start_ns).unwrap_or(0);
    let window = request.time_window_ns.or_else(|| {
        trace
            .trace_span
            .map(|span| (span.start_ns, span.end_ns.max(span.start_ns)))
    });
    let (window_start, window_end) = window.unwrap_or((origin, origin));
    let events_path = sidecar::path(&trace.artifact_dir, PytorchSidecar::Events);
    let query = event_filter::timeline_sql(
        &events_path,
        &request,
        origin,
        window_start,
        window_end,
        interval_ns,
    )?;
    let (rows, total_matched) = query_timeline_rows(&query.sql, &query.params, interval_ns)?;
    Ok(TimelineResponse {
        count: rows.len(),
        total_matched,
        rows,
        auxiliary: TimelineAuxiliary {
            scope: request.rank_scope.echo(request.step),
            interval_ns,
            time_window_ns: request.time_window_ns,
        },
    })
}

fn query_timeline_rows(
    sql: &str,
    params: &[duckdb::types::Value],
    interval_ns: i64,
) -> PytorchQueryResult<(Vec<TimelineBucketRow>, usize)> {
    let raw_rows = exec::query_rows(
        sql,
        params,
        SqlLabel::new(SqlVerb::Timeline, "aggregate"),
        timeline_sql_row,
    )?;
    let total_matched =
        total_matched::<usize, _>(&raw_rows, TotalCarrier::First, |row| row.total_matched)
            .map_err(PytorchQueryError::timeline_count_overflow)?;
    let mut buckets: BTreeMap<i64, BucketAcc> = BTreeMap::new();
    for row in raw_rows {
        let event_count = usize_count(row.event_count)?;

        let acc = buckets.entry(row.bucket_start).or_default();
        acc.cpu_ns = acc.cpu_ns.saturating_add(row.cpu_ns);
        acc.gpu_ns = acc.gpu_ns.saturating_add(row.gpu_ns);
        acc.comm_ns = acc.comm_ns.saturating_add(row.comm_ns);
        acc.event_count = acc.event_count.saturating_add(event_count);
        let entry = acc.by_type_ns.entry(row.event_type).or_default();
        *entry = entry.saturating_add(row.type_ns);
    }

    let rows = buckets
        .into_iter()
        .map(|(bucket_start, acc)| {
            let bucket = TimelineBucket::new(bucket_start, interval_ns);
            TimelineBucketRow {
                key: bucket.key(),
                start_ns: bucket.start_ns,
                end_ns: bucket.end_ns(),
                cpu_ns: acc.cpu_ns,
                gpu_ns: acc.gpu_ns,
                comm_ns: acc.comm_ns,
                event_count: acc.event_count,
                by_type_ns: acc.by_type_ns,
            }
        })
        .collect::<Vec<_>>();
    Ok((rows, total_matched))
}

struct TimelineSqlRow {
    bucket_start: i64,
    event_type: String,
    type_ns: i64,
    cpu_ns: i64,
    gpu_ns: i64,
    comm_ns: i64,
    event_count: i64,
    total_matched: i64,
}

fn timeline_sql_row(row: &duckdb::Row<'_>) -> Result<TimelineSqlRow, duckdb::Error> {
    Ok(TimelineSqlRow {
        bucket_start: row.get("bucket_start")?,
        event_type: row.get("type")?,
        type_ns: row.get("type_ns")?,
        cpu_ns: row.get("cpu_ns")?,
        gpu_ns: row.get("gpu_ns")?,
        comm_ns: row.get("comm_ns")?,
        event_count: row.get("event_count")?,
        total_matched: row.get("total_matched")?,
    })
}

fn usize_count(value: i64) -> PytorchQueryResult<usize> {
    count_from_i64(value, PytorchQueryError::timeline_count_overflow)
}
