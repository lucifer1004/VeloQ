use crate::sql::SqlFragment;
use duckdb::types::Value;
use veloq_core::WindowRef;

/// Half-open point-sample predicate: `expr >= window_start AND expr < window_end`.
///
/// Bind order is always `start, end`.
pub fn point_filter(expr: &str, window: Option<(i64, i64)>) -> Option<SqlFragment> {
    let (start, end) = WindowRef::from_option(window).bounds()?;
    Some(SqlFragment::new(
        format!("{expr} >= ? AND {expr} < ?"),
        vec![Value::BigInt(start), Value::BigInt(end)],
    ))
}

/// `alias.start < window_end AND alias."end" > window_start`.
///
/// Bind order is always `end, start`.
pub fn overlap_filter(alias: &str, window: Option<(i64, i64)>) -> Option<SqlFragment> {
    let (start, end) = WindowRef::from_option(window).bounds()?;
    Some(SqlFragment::new(
        format!(r#"{alias}.start < ? AND {alias}."end" > ?"#),
        vec![Value::BigInt(end), Value::BigInt(start)],
    ))
}

/// Overlap predicate for callers that already projected start/end
/// expressions into an outer CTE.
///
/// Bind order is always `end, start`.
pub fn overlap_filter_expr(
    start_expr: &str,
    end_expr: &str,
    window: Option<(i64, i64)>,
) -> Option<SqlFragment> {
    let (start, end) = WindowRef::from_option(window).bounds()?;
    Some(SqlFragment::new(
        format!("{start_expr} < ? AND {end_expr} > ?"),
        vec![Value::BigInt(end), Value::BigInt(start)],
    ))
}

/// Clipped duration expression for a `t`-style start/end interval.
///
/// Bind order is `end, start`; when paired with [`overlap_filter`] and
/// appended before it, the total order is `end, start, end, start`.
pub fn clipped_duration_expr(alias: &str, window: Option<(i64, i64)>) -> SqlFragment {
    match WindowRef::from_option(window).bounds() {
        Some((start, end)) => SqlFragment::new(
            format!(r#"LEAST({alias}."end", ?) - GREATEST({alias}.start, ?)"#),
            vec![Value::BigInt(end), Value::BigInt(start)],
        ),
        None => SqlFragment::new(format!(r#"({alias}."end" - {alias}.start)"#), Vec::new()),
    }
}

/// Duration clipped to a bucket interval.
pub fn bucket_clipped_duration_expr(
    start_expr: &str,
    end_expr: &str,
    bucket_start_expr: &str,
    bucket_end_expr: &str,
) -> String {
    format!("LEAST({end_expr}, {bucket_end_expr}) - GREATEST({start_expr}, {bucket_start_expr})")
}

/// Scope for positive interval samples that may be clipped to a user
/// time window before aggregation.
#[derive(Debug, Clone, PartialEq)]
pub struct IntervalSampleScope {
    pub start_expr: String,
    pub end_expr: String,
    pub where_clause: String,
    pub params: Vec<Value>,
}

/// Positive interval sample scope.
///
/// Always rejects invalid intervals with
/// `start_expr >= 0 AND end_expr > start_expr`. When a window is present,
/// projected start/end expressions are clipped before downstream bucket
/// math. Bind order is `start, end` for clipped expressions, then
/// `start, end` for the overlap predicate.
pub fn positive_interval_sample_scope(
    start_expr: &str,
    end_expr: &str,
    window: Option<(i64, i64)>,
) -> IntervalSampleScope {
    let mut predicates = vec![format!("{start_expr} >= 0 AND {end_expr} > {start_expr}")];
    let mut params = Vec::new();

    let (scoped_start_expr, scoped_end_expr) = match WindowRef::from_option(window).bounds() {
        Some((start, end)) => {
            params.push(Value::BigInt(start));
            params.push(Value::BigInt(end));
            predicates.push(format!("{end_expr} > ? AND {start_expr} < ?"));
            params.push(Value::BigInt(start));
            params.push(Value::BigInt(end));
            (
                format!("GREATEST({start_expr}, ?)"),
                format!("LEAST({end_expr}, ?)"),
            )
        }
        None => (start_expr.to_string(), end_expr.to_string()),
    };

    IntervalSampleScope {
        start_expr: scoped_start_expr,
        end_expr: scoped_end_expr,
        where_clause: format!("WHERE {}", predicates.join(" AND ")),
        params,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_filter_binds_start_then_end() -> anyhow::Result<()> {
        let fragment = point_filter("m.timestamp", Some((10, 20)))
            .ok_or_else(|| anyhow::anyhow!("window should emit fragment"))?;
        assert_eq!(fragment.sql, "m.timestamp >= ? AND m.timestamp < ?");
        assert_eq!(fragment.params, vec![Value::BigInt(10), Value::BigInt(20)]);
        Ok(())
    }

    #[test]
    fn overlap_filter_binds_end_then_start() -> anyhow::Result<()> {
        let fragment = overlap_filter("t", Some((10, 20)))
            .ok_or_else(|| anyhow::anyhow!("window should emit fragment"))?;
        assert_eq!(fragment.sql, r#"t.start < ? AND t."end" > ?"#);
        assert_eq!(fragment.params, vec![Value::BigInt(20), Value::BigInt(10)]);
        Ok(())
    }

    #[test]
    fn overlap_expr_binds_end_then_start() -> anyhow::Result<()> {
        let fragment = overlap_filter_expr("start_ns", "end_ns", Some((11, 22)))
            .ok_or_else(|| anyhow::anyhow!("window should emit fragment"))?;
        assert_eq!(fragment.sql, "start_ns < ? AND end_ns > ?");
        assert_eq!(fragment.params, vec![Value::BigInt(22), Value::BigInt(11)]);
        Ok(())
    }

    #[test]
    fn clipped_duration_binds_end_then_start() {
        let fragment = clipped_duration_expr("t", Some((10, 20)));
        assert_eq!(fragment.sql, r#"LEAST(t."end", ?) - GREATEST(t.start, ?)"#);
        assert_eq!(fragment.params, vec![Value::BigInt(20), Value::BigInt(10)]);
    }

    #[test]
    fn absent_window_emits_plain_duration_without_params() {
        let fragment = clipped_duration_expr("t", None);
        assert_eq!(fragment.sql, r#"(t."end" - t.start)"#);
        assert!(fragment.params.is_empty());
    }

    #[test]
    fn bucket_clipped_duration_uses_bucket_bounds_without_params() {
        assert_eq!(
            bucket_clipped_duration_expr("s", "e", "bucket_start", "bucket_end"),
            "LEAST(e, bucket_end) - GREATEST(s, bucket_start)"
        );
    }

    #[test]
    fn positive_interval_sample_scope_clips_and_binds_start_end_pairs() {
        let scope =
            positive_interval_sample_scope("m.timestamp", "m.timestamp + m.duration", Some((5, 9)));

        assert_eq!(scope.start_expr, "GREATEST(m.timestamp, ?)");
        assert_eq!(scope.end_expr, "LEAST(m.timestamp + m.duration, ?)");
        assert_eq!(
            scope.where_clause,
            "WHERE m.timestamp >= 0 AND m.timestamp + m.duration > m.timestamp AND m.timestamp + m.duration > ? AND m.timestamp < ?"
        );
        assert_eq!(
            scope.params,
            vec![
                Value::BigInt(5),
                Value::BigInt(9),
                Value::BigInt(5),
                Value::BigInt(9)
            ]
        );
    }

    #[test]
    fn positive_interval_sample_scope_without_window_only_validates_interval() {
        let scope = positive_interval_sample_scope("start", "end", None);

        assert_eq!(scope.start_expr, "start");
        assert_eq!(scope.end_expr, "end");
        assert_eq!(scope.where_clause, "WHERE start >= 0 AND end > start");
        assert!(scope.params.is_empty());
    }
}
