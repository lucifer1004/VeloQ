use super::SqlQuery;
use crate::scope::RankScope;
use duckdb::types::Value;
use veloq_query::sql::{SqlFilter, total_matched_expr};

pub(crate) fn collectives_sql(
    collectives_path: &str,
    rank_scope: RankScope,
    step: Option<i64>,
    limit: usize,
) -> SqlQuery {
    let mut filter = SqlFilter::default();
    if let Some(step) = step {
        filter.push_predicate("step = ?");
        filter.push_param(Value::BigInt(step));
    }
    if let Some(rank) = rank_scope.rank {
        filter.push_predicate("rank = ?");
        filter.push_param(Value::BigInt(rank));
    }
    let where_clause = filter.where_clause();
    let sql = format!(
        r#"
WITH source AS (
  SELECT *
  FROM read_parquet(?)
),
matching_keys AS (
  SELECT DISTINCT key
  FROM source
  {where_clause}
),
groups AS (
  SELECT
    source.key,
    source.collective_kind,
    source.step,
    source.ordinal,
    source.confidence,
    source.start_ns,
    source.duration_ns,
    source.skew_ns,
    source.slow_rank
  FROM source
  JOIN matching_keys USING (key)
  GROUP BY
    source.key,
    source.collective_kind,
    source.step,
    source.ordinal,
    source.confidence,
    source.start_ns,
    source.duration_ns,
    source.skew_ns,
    source.slow_rank
),
ranked_groups AS (
  SELECT
    *,
    {total_matched}
  FROM groups
  ORDER BY duration_ns DESC, key ASC
  LIMIT ?
)
SELECT
  ranked_groups.key,
  ranked_groups.collective_kind,
  ranked_groups.step,
  ranked_groups.ordinal,
  ranked_groups.confidence,
  ranked_groups.start_ns,
  ranked_groups.duration_ns,
  ranked_groups.skew_ns,
  ranked_groups.slow_rank,
  ranked_groups.total_matched,
  source.rank_ordinal,
  source.rank,
  source.row_id,
  source.cpu_row_id,
  source.kernel_row_ids,
  source.event_row_ids,
  source.name,
  source.rank_start_ns,
  source.rank_duration_ns
FROM ranked_groups
JOIN source USING (key)
ORDER BY ranked_groups.duration_ns DESC, ranked_groups.key ASC, source.rank_ordinal ASC
"#,
        total_matched = total_matched_expr(),
    );
    let mut params = vec![Value::Text(collectives_path.to_string())];
    params.extend(filter.into_params());
    params.push(Value::BigInt(limit as i64));
    SqlQuery { sql, params }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collectives_sql_keeps_filter_bind_order() {
        let query = collectives_sql(
            "collectives.parquet",
            RankScope {
                rank: Some(2),
                all_ranks: false,
            },
            Some(7),
            50,
        );

        assert!(query.sql.contains("FROM read_parquet(?)"));
        assert!(query.sql.contains("step = ?"));
        assert!(query.sql.contains("rank = ?"));
        assert_eq!(
            query.params,
            vec![
                Value::Text("collectives.parquet".to_string()),
                Value::BigInt(7),
                Value::BigInt(2),
                Value::BigInt(50),
            ]
        );
    }
}
