use crate::dto::{CollectiveRankRow, CollectiveRow, CollectivesAuxiliary, CollectivesResponse};
use crate::filter::{limit_ref, require_rank_scope};
use crate::query_sql::{
    collectives as collectives_sql,
    exec::{self, SqlLabel, SqlVerb},
    sidecar,
};
use crate::scope::RankScope;
use crate::{PytorchQueryError, PytorchQueryResult};
use veloq_pytorch_data::{PytorchSidecar, QueryTrace};
use veloq_query::duckdb::list::{TotalCarrier, total_matched};

pub fn collectives(
    trace: &QueryTrace,
    rank_scope: RankScope,
    step: Option<i64>,
    limit: usize,
) -> PytorchQueryResult<CollectivesResponse> {
    require_rank_scope(trace, rank_scope)?;
    limit_ref(limit)?;
    let collectives_path = sidecar::path(&trace.artifact_dir, PytorchSidecar::Collectives);
    let query = collectives_sql::collectives_sql(&collectives_path, rank_scope, step, limit);
    let sql_rows = exec::query_rows(
        &query.sql,
        &query.params,
        SqlLabel::new(SqlVerb::Collectives, "rows"),
        collective_sql_row,
    )?;
    let (rows, total_matched) = collectives_from_sql_rows(sql_rows)?;
    Ok(CollectivesResponse {
        count: rows.len(),
        total_matched,
        rows,
        auxiliary: CollectivesAuxiliary {
            scope: rank_scope.echo(step),
            cross_rank_skew: false,
        },
    })
}

#[derive(Debug)]
struct CollectiveSqlRow {
    key: String,
    collective_kind: String,
    step: Option<i64>,
    ordinal: u64,
    confidence: String,
    start_ns: i64,
    duration_ns: i64,
    skew_ns: Option<i64>,
    slow_rank: Option<i64>,
    total_matched: i64,
    rank: Option<i64>,
    row_id: String,
    cpu_row_id: Option<String>,
    kernel_row_ids: String,
    event_row_ids: String,
    name: String,
    rank_start_ns: i64,
    rank_duration_ns: i64,
}

fn collectives_from_sql_rows(
    rows: Vec<CollectiveSqlRow>,
) -> PytorchQueryResult<(Vec<CollectiveRow>, usize)> {
    let total_matched =
        total_matched::<usize, _>(&rows, TotalCarrier::First, |row| row.total_matched)
            .map_err(PytorchQueryError::collectives_count_overflow)?;
    let mut out: Vec<CollectiveRow> = Vec::new();
    for row in rows {
        let rank_row = CollectiveRankRow {
            rank: row.rank,
            row_id: row.row_id,
            cpu_row_id: row.cpu_row_id,
            kernel_row_ids: split_row_ids(&row.kernel_row_ids),
            event_row_ids: split_row_ids(&row.event_row_ids),
            name: row.name,
            start_ns: row.rank_start_ns,
            duration_ns: row.rank_duration_ns,
        };
        match out.last_mut() {
            Some(current) if current.key == row.key => {
                current.per_rank.push(rank_row);
            }
            _ => out.push(CollectiveRow {
                key: row.key,
                collective_kind: row.collective_kind,
                step: row.step,
                ordinal: row.ordinal,
                confidence: row.confidence,
                start_ns: row.start_ns,
                duration_ns: row.duration_ns,
                skew_ns: row.skew_ns,
                slow_rank: row.slow_rank,
                per_rank: vec![rank_row],
            }),
        }
    }
    Ok((out, total_matched))
}

fn split_row_ids(value: &str) -> Vec<String> {
    if value.is_empty() {
        return Vec::new();
    }
    value.split(',').map(str::to_string).collect()
}

fn collective_sql_row(row: &duckdb::Row<'_>) -> Result<CollectiveSqlRow, duckdb::Error> {
    Ok(CollectiveSqlRow {
        key: row.get("key")?,
        collective_kind: row.get("collective_kind")?,
        step: row.get("step")?,
        ordinal: row.get("ordinal")?,
        confidence: row.get("confidence")?,
        start_ns: row.get("start_ns")?,
        duration_ns: row.get("duration_ns")?,
        skew_ns: row.get("skew_ns")?,
        slow_rank: row.get("slow_rank")?,
        total_matched: row.get("total_matched")?,
        rank: row.get("rank")?,
        row_id: row.get("row_id")?,
        cpu_row_id: row.get("cpu_row_id")?,
        kernel_row_ids: row.get("kernel_row_ids")?,
        event_row_ids: row.get("event_row_ids")?,
        name: row.get("name")?,
        rank_start_ns: row.get("rank_start_ns")?,
        rank_duration_ns: row.get("rank_duration_ns")?,
    })
}
