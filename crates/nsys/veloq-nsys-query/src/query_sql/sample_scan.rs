use duckdb::types::Value;
use veloq_query::sql::{SqlFilter, SqlFragment, window};

/// Shared builder for source-table samples filtered by time, CPU, and
/// decoded global thread id.
pub(crate) fn filtered_cte(
    cte_name: &str,
    table: &str,
    select_projection: &str,
    cpu_filter: Option<i64>,
    tid_filter: Option<i64>,
    abs_window: Option<(i64, i64)>,
) -> SqlFragment {
    let mut filter = SqlFilter::default();
    if let Some(fragment) = window::point_filter("start", abs_window) {
        filter.push_fragment(fragment);
    }
    if let Some(cpu) = cpu_filter {
        filter.push_predicate("CAST(cpu AS BIGINT) = ?");
        filter.push_param(Value::BigInt(cpu));
    }
    if let Some(tid) = tid_filter {
        filter.push_predicate(format!(
            "{} = ?",
            veloq_nsys_data::sql_expr::u64_bits_to_i64("globalTid")
        ));
        filter.push_param(Value::BigInt(tid));
    }

    let where_clause = filter.where_clause();
    SqlFragment::new(
        format!(
            "{cte_name} AS (
                SELECT {select_projection}
                FROM nsight.{table}
                {where_clause}
            )"
        ),
        filter.into_params(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filtered_cte_keeps_window_cpu_tid_param_order() {
        let fragment = filtered_cte(
            "filtered_samples",
            "COMPOSITE_EVENTS",
            "id, start, cpu, globalTid",
            Some(7),
            Some(42),
            Some((100, 200)),
        );

        assert!(fragment.sql.contains("filtered_samples AS"));
        assert!(
            fragment
                .sql
                .contains("WHERE start >= ? AND start < ? AND CAST(cpu AS BIGINT) = ?")
        );
        assert!(
            fragment
                .sql
                .contains("CASE WHEN globalTid IS NULL THEN NULL")
        );
        assert!(
            fragment
                .sql
                .contains("ELSE CAST(globalTid AS BIGINT) END = ?")
        );
        assert_eq!(
            fragment.params,
            vec![
                Value::BigInt(100),
                Value::BigInt(200),
                Value::BigInt(7),
                Value::BigInt(42),
            ]
        );
    }

    #[test]
    fn filtered_cte_omits_where_when_unfiltered() {
        let fragment = filtered_cte("filtered_sched", "SCHED_EVENTS", "start", None, None, None);

        assert!(!fragment.sql.contains("WHERE"));
        assert!(fragment.params.is_empty());
    }
}
