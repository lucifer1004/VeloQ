use super::{SqlQuery, push_name_filter};
use crate::filter::EventFilterRequest;
use crate::{PytorchQueryError, PytorchQueryResult};
use duckdb::types::Value;
use veloq_core::WindowRef;
use veloq_query::sql::{SqlFilter, SqlFragment, total_matched_expr, window};

pub(crate) fn instance_sql(
    events_path: &str,
    request: &EventFilterRequest,
) -> PytorchQueryResult<SqlQuery> {
    let mut common = common_instance_ctes(events_path, request)?;
    let sql = format!(
        r#"
{ctes},
ranked AS (
  SELECT
    *,
    {total_matched}
  FROM instances
  ORDER BY start_ns ASC, row_id ASC
  LIMIT ?
)
SELECT
  key,
  row_id,
  name,
  start_ns,
  duration_ns,
  rank,
  step,
  child_count,
  attributed_gpu_ns,
  attributed_comm_ns,
  total_matched
FROM ranked
ORDER BY start_ns ASC, row_id ASC
"#,
        ctes = common.ctes,
        total_matched = total_matched_expr(),
    );
    common.params.push(Value::BigInt(request.limit as i64));
    Ok(SqlQuery {
        sql,
        params: common.params,
    })
}

pub(crate) fn aggregate_sql(
    events_path: &str,
    request: &EventFilterRequest,
    group_by: &str,
) -> PytorchQueryResult<SqlQuery> {
    let mut common = common_instance_ctes(events_path, request)?;
    let scope_expr = aggregate_scope_expr(group_by)?;
    let sql = format!(
        r#"
{ctes},
grouped AS (
  SELECT
    'scope|' || scope AS key,
    scope,
    CAST(COUNT(*) AS BIGINT) AS instances,
    CAST(SUM(duration_ns) AS BIGINT) AS total_cpu_ns,
    CAST(SUM(attributed_gpu_ns) AS BIGINT) AS total_gpu_ns,
    CAST(SUM(attributed_comm_ns) AS BIGINT) AS total_comm_ns,
    AVG(duration_ns) AS avg_cpu_ns
  FROM (
    SELECT
      {scope_expr} AS scope,
      duration_ns,
      attributed_gpu_ns,
      attributed_comm_ns
    FROM instances
  )
  GROUP BY scope
),
ranked AS (
  SELECT
    *,
    {total_matched}
  FROM grouped
  ORDER BY total_cpu_ns DESC, key ASC
  LIMIT ?
)
SELECT
  key,
  scope,
  instances,
  total_cpu_ns,
  total_gpu_ns,
  total_comm_ns,
  avg_cpu_ns,
  total_matched
FROM ranked
ORDER BY total_cpu_ns DESC, key ASC
"#,
        ctes = common.ctes,
        total_matched = total_matched_expr(),
    );
    common.params.push(Value::BigInt(request.limit as i64));
    Ok(SqlQuery {
        sql,
        params: common.params,
    })
}

struct CommonInstanceCtes {
    ctes: String,
    params: Vec<Value>,
}

fn common_instance_ctes(
    events_path: &str,
    request: &EventFilterRequest,
) -> PytorchQueryResult<CommonInstanceCtes> {
    let slice_filter = slice_filter(request)?;
    let slice_where_clause = slice_filter.where_clause();
    let clip = attribution_clip_expr(request.time_window_ns);
    let attribution_scope = attribution_scope_filter(request);
    let attribution_scope_predicates = attribution_scope.sql;
    let ctes = format!(
        r#"
WITH events AS (
  SELECT *
  FROM read_parquet(?)
),
filtered_slices AS (
  SELECT
    s.row_id,
    s.name,
    s.start_ns,
    s.duration_ns,
    s.end_ns,
    s.rank,
    s.device_id,
    s.stream_id,
    s.step,
    s.trace_index
  FROM events s
  {slice_where_clause}
),
child_counts AS (
  SELECT
    parent_row_id AS row_id,
    CAST(COUNT(*) AS BIGINT) AS child_count
  FROM events
  WHERE parent_row_id IS NOT NULL
  GROUP BY parent_row_id
),
attribution_candidates AS (
  SELECT
    s.row_id,
    c.is_gpu_activity,
    c.is_comm,
    {clip_expr} AS overlap_ns
  FROM filtered_slices s
  JOIN events c ON
    c.row_id <> s.row_id
    AND c.trace_index = s.trace_index
    AND c.start_ns >= s.start_ns
    AND c.end_ns <= s.end_ns
    AND (c.is_gpu_activity OR c.is_comm)
    AND {attribution_scope_predicates}
),
attribution_rows AS (
  SELECT
    row_id,
    CAST(COALESCE(SUM(CASE WHEN is_gpu_activity THEN overlap_ns ELSE 0 END), 0) AS BIGINT) AS attributed_gpu_ns,
    CAST(COALESCE(SUM(CASE WHEN is_comm THEN overlap_ns ELSE 0 END), 0) AS BIGINT) AS attributed_comm_ns
  FROM attribution_candidates
  GROUP BY row_id
),
instances AS (
  SELECT
    'slice|' || s.name || '|@' || CAST(s.start_ns AS VARCHAR) AS key,
    s.row_id,
    s.name,
    s.start_ns,
    s.duration_ns,
    s.rank,
    s.step,
    COALESCE(cc.child_count, 0) AS child_count,
    COALESCE(a.attributed_gpu_ns, 0) AS attributed_gpu_ns,
    COALESCE(a.attributed_comm_ns, 0) AS attributed_comm_ns
  FROM filtered_slices s
  LEFT JOIN child_counts cc USING (row_id)
  LEFT JOIN attribution_rows a USING (row_id)
)"#,
        clip_expr = clip.sql
    );
    let mut params = vec![Value::Text(events_path.to_string())];
    params.extend(slice_filter.into_params());
    params.extend(clip.params);
    params.extend(attribution_scope.params);
    Ok(CommonInstanceCtes { ctes, params })
}

fn slice_filter(request: &EventFilterRequest) -> PytorchQueryResult<SqlFilter> {
    let mut filter = SqlFilter::default();
    filter.push_predicate("s.type IN ('step', 'annotation')");
    push_name_filter(&mut filter, "s.name", request)?;
    if let Some(fragment) =
        window::overlap_filter_expr("s.start_ns", "s.end_ns", request.time_window_ns)
    {
        filter.push_fragment(fragment);
    }
    if let Some(step) = request.step {
        filter.push_predicate("s.step = ?");
        filter.push_param(Value::BigInt(step));
    }
    if let Some(rank) = request.rank_scope.rank {
        filter.push_predicate("(s.rank IS NULL OR s.rank = ?)");
        filter.push_param(Value::BigInt(rank));
    }
    if let Some(device) = request.device {
        filter.push_predicate("(s.device_id IS NULL OR s.device_id = ?)");
        filter.push_param(Value::BigInt(device));
    }
    if let Some(stream) = request.stream {
        filter.push_predicate("(s.stream_id IS NULL OR s.stream_id = ?)");
        filter.push_param(Value::BigInt(stream));
    }
    if let Some(fragment) = slice_scope_filter(request) {
        filter.push_fragment(fragment);
    }
    Ok(filter)
}

fn slice_scope_filter(request: &EventFilterRequest) -> Option<SqlFragment> {
    if request.rank_scope.rank.is_none() && request.device.is_none() && request.stream.is_none() {
        return None;
    }

    let mut params = Vec::new();
    let direct = event_scope_predicates("s", request, &mut params, false);
    let mut child_predicates = vec![
        "child.row_id <> s.row_id".to_string(),
        "child.trace_index = s.trace_index".to_string(),
        "child.start_ns >= s.start_ns".to_string(),
        "child.end_ns <= s.end_ns".to_string(),
    ];
    if let Some(fragment) =
        window::overlap_filter_expr("child.start_ns", "child.end_ns", request.time_window_ns)
    {
        child_predicates.push(fragment.sql);
        params.extend(fragment.params);
    }
    child_predicates.extend(event_scope_predicates("child", request, &mut params, false));

    Some(SqlFragment::new(
        format!(
            "(({}) OR EXISTS (SELECT 1 FROM events child WHERE {}))",
            direct.join(" AND "),
            child_predicates.join(" AND ")
        ),
        params,
    ))
}

fn attribution_scope_filter(request: &EventFilterRequest) -> SqlFragment {
    let mut params = Vec::new();
    let mut predicates = vec![
        attribution_duration_predicate(request.time_window_ns, &mut params),
        "(c.rank IS NULL OR s.rank IS NULL OR c.rank = s.rank)".to_string(),
    ];
    if request.rank_scope.rank.is_some() {
        predicates.pop();
    }
    predicates.extend(event_scope_predicates("c", request, &mut params, true));
    SqlFragment::new(predicates.join(" AND "), params)
}

fn event_scope_predicates(
    alias: &str,
    request: &EventFilterRequest,
    params: &mut Vec<Value>,
    include_step: bool,
) -> Vec<String> {
    let mut predicates = Vec::new();
    if let Some(rank) = request.rank_scope.rank {
        predicates.push(format!("{alias}.rank = ?"));
        params.push(Value::BigInt(rank));
    }
    if let Some(device) = request.device {
        predicates.push(format!("{alias}.device_id = ?"));
        params.push(Value::BigInt(device));
    }
    if let Some(stream) = request.stream {
        predicates.push(format!("{alias}.stream_id = ?"));
        params.push(Value::BigInt(stream));
    }
    if include_step && let Some(step) = request.step {
        predicates.push(format!("{alias}.step = ?"));
        params.push(Value::BigInt(step));
    }
    predicates
}

fn attribution_clip_expr(window: Option<(i64, i64)>) -> SqlFragment {
    match WindowRef::from_option(window).bounds() {
        Some((start, end)) => SqlFragment::new(
            "LEAST(GREATEST(c.end_ns, c.start_ns), ?) - GREATEST(c.start_ns, ?)",
            vec![Value::BigInt(end), Value::BigInt(start)],
        ),
        None => SqlFragment::new("GREATEST(c.end_ns, c.start_ns) - c.start_ns", Vec::new()),
    }
}

fn attribution_duration_predicate(window: Option<(i64, i64)>, params: &mut Vec<Value>) -> String {
    match WindowRef::from_option(window).bounds() {
        Some((start, end)) => {
            params.push(Value::BigInt(end));
            params.push(Value::BigInt(start));
            "c.start_ns < ? AND GREATEST(c.end_ns, c.start_ns) > ?".to_string()
        }
        None => "GREATEST(c.end_ns, c.start_ns) > c.start_ns".to_string(),
    }
}

fn aggregate_scope_expr(group_by: &str) -> PytorchQueryResult<&'static str> {
    match group_by {
        "name" => Ok("name"),
        "step" => Ok("COALESCE(CAST(step AS VARCHAR), 'none')"),
        other => Err(PytorchQueryError::unknown_slices_group_by(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scope::RankScope;

    #[test]
    fn instance_sql_keeps_filter_and_attribution_bind_order() -> PytorchQueryResult<()> {
        let sql = instance_sql(
            "events.parquet",
            &EventFilterRequest {
                name_glob: Some("Profiler*".to_string()),
                time_window_ns: Some((100, 200)),
                rank_scope: RankScope {
                    rank: Some(3),
                    all_ranks: false,
                },
                device: Some(0),
                stream: Some(7),
                step: Some(2),
                limit: 5,
                ..EventFilterRequest::default()
            },
        )?;

        assert!(sql.sql.contains("FROM read_parquet(?)"));
        assert!(sql.sql.contains("child.trace_index = s.trace_index"));
        assert!(sql.sql.contains("AND (c.is_gpu_activity OR c.is_comm)"));
        assert_eq!(
            sql.params,
            vec![
                Value::Text("events.parquet".to_string()),
                Value::Text("Profiler%".to_string()),
                Value::BigInt(200),
                Value::BigInt(100),
                Value::BigInt(2),
                Value::BigInt(3),
                Value::BigInt(0),
                Value::BigInt(7),
                Value::BigInt(3),
                Value::BigInt(0),
                Value::BigInt(7),
                Value::BigInt(200),
                Value::BigInt(100),
                Value::BigInt(3),
                Value::BigInt(0),
                Value::BigInt(7),
                Value::BigInt(200),
                Value::BigInt(100),
                Value::BigInt(200),
                Value::BigInt(100),
                Value::BigInt(3),
                Value::BigInt(0),
                Value::BigInt(7),
                Value::BigInt(2),
                Value::BigInt(5),
            ]
        );
        Ok(())
    }

    #[test]
    fn instance_sql_pushes_regex_to_duckdb() -> PytorchQueryResult<()> {
        let sql = instance_sql(
            "events.parquet",
            &EventFilterRequest {
                name_regex: Some("ProfilerStep#.*".to_string()),
                limit: 5,
                ..EventFilterRequest::default()
            },
        )?;

        assert!(sql.sql.contains("regexp_matches(s.name, ?)"));
        assert_eq!(
            sql.params,
            vec![
                Value::Text("events.parquet".to_string()),
                Value::Text("ProfilerStep#.*".to_string()),
                Value::BigInt(5),
            ]
        );
        Ok(())
    }

    #[test]
    fn aggregate_sql_rejects_unknown_group_by_after_filter_validation() -> PytorchQueryResult<()> {
        let err = aggregate_sql(
            "events.parquet",
            &EventFilterRequest {
                name_glob: Some("step*".to_string()),
                ..EventFilterRequest::default()
            },
            "rank",
        )
        .err()
        .ok_or(PytorchQueryError::LimitTooSmall)?;

        assert!(matches!(
            err,
            PytorchQueryError::UnknownSlicesGroupBy { .. }
        ));
        Ok(())
    }
}
