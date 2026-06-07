use super::{SqlQuery, push_name_filter};
use crate::filter::{EventFilterRequest, TypeSelection, TypeToken};
use crate::{PytorchQueryError, PytorchQueryResult};
use duckdb::types::Value;
use veloq_query::sql::{SqlFilter, total_matched_expr, window};

pub(crate) fn search_sql(
    events_path: &str,
    request: &EventFilterRequest,
) -> PytorchQueryResult<SqlQuery> {
    let filter = build_filter(request)?;
    let where_clause = filter.where_clause();
    let sql = format!(
        r#"
SELECT
  key,
  row_id,
  type AS event_type,
  name,
  start_ns,
  duration_ns,
  rank,
  worker,
  device_id,
  stream_id,
  step,
  is_comm,
  external_id,
  correlation_id,
  comm_kind,
  bytes,
  shape,
  total_matched
FROM (
  SELECT
    key,
    row_id,
    type,
    name,
    start_ns,
    duration_ns,
    rank,
    worker,
    device_id,
    stream_id,
    step,
    is_comm,
    external_id,
    correlation_id,
    comm_kind,
    bytes,
    shape,
    stable_index,
    {total_matched}
  FROM read_parquet(?)
  {where_clause}
  ORDER BY start_ns ASC, stable_index ASC
  LIMIT ?
) AS ranked
ORDER BY start_ns ASC, stable_index ASC
"#,
        total_matched = total_matched_expr(),
    );
    let mut params = vec![Value::Text(events_path.to_string())];
    params.append(&mut filter.into_params());
    params.push(Value::BigInt(request.limit as i64));
    Ok(SqlQuery { sql, params })
}

pub(crate) fn stats_sql(
    events_path: &str,
    request: &EventFilterRequest,
    group_by: &[String],
) -> PytorchQueryResult<SqlQuery> {
    let filter = build_filter(request)?;
    let where_clause = filter.where_clause();
    let axes = stats_axes(group_by)?;
    let select_axes = if axes.is_empty() {
        String::new()
    } else {
        format!(
            "{},",
            axes.iter()
                .enumerate()
                .map(|(idx, axis)| format!("{} AS axis_{idx}", axis.sql_expr))
                .collect::<Vec<_>>()
                .join(",\n    ")
        )
    };
    let group_tail_clause = if axes.is_empty() {
        "HAVING COUNT(*) > 0".to_string()
    } else {
        format!(
            "GROUP BY {}",
            (1..=axes.len())
                .map(|idx| idx.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let sort_key_expr = stats_sort_key_expr(&axes);
    let sql = format!(
        r#"
WITH grouped AS (
  SELECT
    {select_axes}
    COUNT(*) AS count,
    SUM(duration_ns) AS total_ns,
    AVG(duration_ns) AS avg_ns,
    MIN(duration_ns) AS min_ns,
    MAX(duration_ns) AS max_ns
  FROM read_parquet(?)
  {where_clause}
  {group_tail_clause}
),
ranked AS (
  SELECT
    *,
    {sort_key_expr} AS sort_key,
    {total_matched}
  FROM grouped
  ORDER BY total_ns DESC, sort_key ASC
  LIMIT ?
)
SELECT *
FROM ranked
ORDER BY total_ns DESC, sort_key ASC
"#,
        total_matched = total_matched_expr(),
    );
    let mut params = vec![Value::Text(events_path.to_string())];
    params.extend(filter.into_params());
    params.push(Value::BigInt(request.limit as i64));
    Ok(SqlQuery { sql, params })
}

pub(crate) fn timeline_sql(
    events_path: &str,
    request: &EventFilterRequest,
    origin_ns: i64,
    window_start_ns: i64,
    window_end_ns: i64,
    interval_ns: i64,
) -> PytorchQueryResult<SqlQuery> {
    let filter = build_filter(request)?;
    let where_clause = filter.where_clause();
    let sql = format!(
        r#"
WITH filtered AS (
  SELECT
    type,
    start_ns,
    GREATEST(end_ns, start_ns) AS event_end_ns,
    is_comm,
    is_gpu_activity
  FROM read_parquet(?)
  {where_clause}
),
bounded AS (
  SELECT
    type,
    GREATEST(start_ns, ?) AS clipped_start,
    LEAST(event_end_ns, ?) AS clipped_end,
    is_comm,
    is_gpu_activity
  FROM filtered
),
clipped AS (
  SELECT *
  FROM bounded
  WHERE clipped_end > clipped_start
),
bucketed AS (
  SELECT
    bucket.bucket_start,
    type,
    is_comm,
    is_gpu_activity,
    GREATEST(
      0,
      LEAST(clipped_end, bucket.bucket_start + ?) - GREATEST(clipped_start, bucket.bucket_start)
    ) AS overlap_ns
  FROM (
    SELECT
      *,
      ? + CAST(FLOOR((clipped_start - ?) / ?) AS BIGINT) * ? AS first_bucket_start
    FROM clipped
  ) AS events
  CROSS JOIN LATERAL range(first_bucket_start, clipped_end, ?) AS bucket(bucket_start)
),
type_rows AS (
  SELECT
    bucket_start,
    type,
    SUM(overlap_ns) AS type_ns,
    SUM(CASE WHEN is_gpu_activity THEN overlap_ns ELSE 0 END) AS gpu_ns,
    SUM(CASE WHEN NOT is_gpu_activity THEN overlap_ns ELSE 0 END) AS cpu_ns,
    SUM(CASE WHEN is_comm THEN overlap_ns ELSE 0 END) AS comm_ns,
    COUNT(*) AS event_count
  FROM bucketed
  WHERE overlap_ns > 0
  GROUP BY bucket_start, type
),
bucket_rows AS (
  SELECT bucket_start
  FROM type_rows
  GROUP BY bucket_start
),
ranked_buckets AS (
  SELECT
    bucket_start,
    {total_matched}
  FROM bucket_rows
  ORDER BY bucket_start ASC
  LIMIT ?
)
SELECT
  tr.bucket_start,
  tr.type,
  tr.type_ns,
  tr.cpu_ns,
  tr.gpu_ns,
  tr.comm_ns,
  tr.event_count,
  rb.total_matched
FROM type_rows tr
JOIN ranked_buckets rb USING (bucket_start)
ORDER BY tr.bucket_start ASC, tr.type ASC
"#,
        total_matched = total_matched_expr(),
    );
    let mut params = vec![Value::Text(events_path.to_string())];
    params.extend(filter.into_params());
    params.extend([
        Value::BigInt(window_start_ns),
        Value::BigInt(window_end_ns),
        Value::BigInt(interval_ns),
        Value::BigInt(origin_ns),
        Value::BigInt(origin_ns),
        Value::BigInt(interval_ns),
        Value::BigInt(interval_ns),
        Value::BigInt(interval_ns),
        Value::BigInt(request.limit as i64),
    ]);
    Ok(SqlQuery { sql, params })
}

pub(crate) fn build_filter(request: &EventFilterRequest) -> PytorchQueryResult<SqlFilter> {
    let mut filter = SqlFilter::default();
    push_type_filter(&mut filter, &request.types);
    if request.is_comm {
        filter.push_predicate("is_comm = TRUE");
    }
    if let Some(rank) = request.rank_scope.rank {
        filter.push_predicate("rank = ?");
        filter.push_param(Value::BigInt(rank));
    }
    if let Some(device) = request.device {
        filter.push_predicate("device_id = ?");
        filter.push_param(Value::BigInt(device));
    }
    if let Some(stream) = request.stream {
        filter.push_predicate("stream_id = ?");
        filter.push_param(Value::BigInt(stream));
    }
    if let Some(step) = request.step {
        filter.push_predicate("step = ?");
        filter.push_param(Value::BigInt(step));
    }
    push_name_filter(&mut filter, "name", request)?;
    if let Some(duration) = request.duration {
        filter.push_predicate(duration.sql("duration_ns"));
        for value in duration.sql_params() {
            filter.push_param(Value::BigInt(value));
        }
    }
    if let Some(fragment) =
        window::overlap_filter_expr("start_ns", "end_ns", request.time_window_ns)
    {
        filter.push_fragment(fragment);
    }
    Ok(filter)
}

struct StatsAxis {
    name: String,
    sql_expr: &'static str,
}

fn stats_axes(group_by: &[String]) -> PytorchQueryResult<Vec<StatsAxis>> {
    group_by
        .iter()
        .map(|axis| {
            Ok(StatsAxis {
                name: axis.clone(),
                sql_expr: stats_axis_expr(axis)?,
            })
        })
        .collect()
}

fn stats_axis_expr(axis: &str) -> PytorchQueryResult<&'static str> {
    match axis {
        "name" => Ok("COALESCE(name, 'none')"),
        "type" => Ok("COALESCE(type, 'none')"),
        "step" => Ok("COALESCE(CAST(step AS VARCHAR), 'none')"),
        "rank" => Ok("COALESCE(CAST(rank AS VARCHAR), 'none')"),
        "device" => Ok("COALESCE(CAST(device_id AS VARCHAR), 'none')"),
        "stream" => Ok("COALESCE(CAST(stream_id AS VARCHAR), 'none')"),
        "shape" => Ok("COALESCE(shape, 'none')"),
        "comm-kind" => Ok("COALESCE(comm_kind, 'none')"),
        "python-context" => Ok("COALESCE(python_context_name, 'none')"),
        "python-path" => Ok("COALESCE(python_context_path, 'none')"),
        other => Err(PytorchQueryError::unknown_stats_group_by(other)),
    }
}

fn stats_sort_key_expr(axes: &[StatsAxis]) -> String {
    if axes.is_empty() {
        return "'stats|'".to_string();
    }
    let mut keyed_aliases = axes
        .iter()
        .enumerate()
        .map(|(idx, axis)| (axis.name.as_str(), format!("axis_{idx}")))
        .collect::<Vec<_>>();
    keyed_aliases.sort_by(|(lhs, _), (rhs, _)| lhs.cmp(rhs));
    keyed_aliases.dedup_by(|(lhs, _), (rhs, _)| lhs == rhs);
    let suffix = keyed_aliases
        .into_iter()
        .map(|(axis, alias)| format!("'{axis}:' || {alias}"))
        .collect::<Vec<_>>()
        .join(" || '|' || ");
    format!("'stats|' || {suffix}")
}

fn push_type_filter(filter: &mut SqlFilter, selection: &TypeSelection) {
    let TypeSelection::Only(tokens) = selection else {
        return;
    };
    let mut type_params = Vec::new();
    let mut has_comm = false;
    for token in tokens {
        match token {
            TypeToken::Event(event_type) => {
                type_params.push(Value::Text(event_type.as_str().to_string()));
            }
            TypeToken::Comm => has_comm = true,
        }
    }
    let mut parts = Vec::new();
    if !type_params.is_empty() {
        let placeholders = std::iter::repeat_n("?", type_params.len())
            .collect::<Vec<_>>()
            .join(", ");
        parts.push(format!("type IN ({placeholders})"));
    }
    if has_comm {
        parts.push("is_comm = TRUE".to_string());
    }
    if !parts.is_empty() {
        filter.push_predicate(format!("({})", parts.join(" OR ")));
        for param in type_params {
            filter.push_param(param);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::TypeSelection;
    use crate::scope::RankScope;
    use veloq_core::time::DurationFilter;
    use veloq_pytorch_data::EventType;

    #[test]
    fn search_sql_keeps_bind_order() -> PytorchQueryResult<()> {
        let sql = search_sql(
            "events.parquet",
            &EventFilterRequest {
                types: TypeSelection::Only(
                    [TypeToken::Event(EventType::Kernel), TypeToken::Comm]
                        .into_iter()
                        .collect(),
                ),
                name_glob: Some("nccl*".to_string()),
                duration: Some(DurationFilter::Gte(10)),
                time_window_ns: Some((100, 200)),
                rank_scope: RankScope {
                    rank: Some(3),
                    all_ranks: false,
                },
                device: Some(0),
                stream: Some(7),
                step: Some(2),
                is_comm: true,
                limit: 50,
                ..EventFilterRequest::default()
            },
        )?;

        assert!(sql.sql.contains("FROM read_parquet(?)"));
        assert!(
            !sql.sql
                .contains(&format!("SELECT *, {}", total_matched_expr()))
        );
        assert_eq!(
            sql.params,
            vec![
                Value::Text("events.parquet".to_string()),
                Value::Text("kernel".to_string()),
                Value::BigInt(3),
                Value::BigInt(0),
                Value::BigInt(7),
                Value::BigInt(2),
                Value::Text("nccl%".to_string()),
                Value::BigInt(10),
                Value::BigInt(200),
                Value::BigInt(100),
                Value::BigInt(50),
            ]
        );
        Ok(())
    }

    #[test]
    fn search_sql_pushes_regex_to_duckdb() -> PytorchQueryResult<()> {
        let sql = search_sql(
            "events.parquet",
            &EventFilterRequest {
                name_regex: Some("aten::.*".to_string()),
                limit: 10,
                ..EventFilterRequest::default()
            },
        )?;

        assert!(sql.sql.contains("regexp_matches(name, ?)"));
        assert_eq!(
            sql.params,
            vec![
                Value::Text("events.parquet".to_string()),
                Value::Text("aten::.*".to_string()),
                Value::BigInt(10),
            ]
        );
        Ok(())
    }

    #[test]
    fn stats_sql_builds_group_key_in_axis_name_order() -> PytorchQueryResult<()> {
        let sql = stats_sql(
            "events.parquet",
            &EventFilterRequest {
                name_glob: Some("aten::*".to_string()),
                limit: 10,
                ..EventFilterRequest::default()
            },
            &["type".to_string(), "name".to_string()],
        )?;

        assert!(sql.sql.contains("COALESCE(type, 'none') AS axis_0"));
        assert!(sql.sql.contains("COALESCE(name, 'none') AS axis_1"));
        assert!(
            sql.sql
                .contains("'stats|' || 'name:' || axis_1 || '|' || 'type:' || axis_0")
        );
        assert_eq!(
            sql.params,
            vec![
                Value::Text("events.parquet".to_string()),
                Value::Text("aten::%".to_string()),
                Value::BigInt(10),
            ]
        );
        Ok(())
    }

    #[test]
    fn timeline_sql_keeps_clip_and_bucket_bind_order() -> PytorchQueryResult<()> {
        let sql = timeline_sql(
            "events.parquet",
            &EventFilterRequest {
                name_glob: Some("aten::*".to_string()),
                time_window_ns: Some((100, 200)),
                limit: 5,
                ..EventFilterRequest::default()
            },
            10,
            100,
            200,
            50,
        )?;

        assert!(sql.sql.contains("CROSS JOIN LATERAL range"));
        assert_eq!(
            sql.params,
            vec![
                Value::Text("events.parquet".to_string()),
                Value::Text("aten::%".to_string()),
                Value::BigInt(200),
                Value::BigInt(100),
                Value::BigInt(100),
                Value::BigInt(200),
                Value::BigInt(50),
                Value::BigInt(10),
                Value::BigInt(10),
                Value::BigInt(50),
                Value::BigInt(50),
                Value::BigInt(50),
                Value::BigInt(5),
            ]
        );
        Ok(())
    }
}
