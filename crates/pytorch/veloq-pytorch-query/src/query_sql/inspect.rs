use super::SqlQuery;
use duckdb::types::Value;

const EVENT_COLUMNS: &str = r#"
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
  trace_index,
  original_index,
  category,
  phase,
  pid,
  tid,
  parent_row_id,
  step_row_id,
  python_context_row_id,
  python_id,
  python_parent_id,
  is_gpu_activity,
  raw_json
"#;

pub(crate) fn events_by_row_ids_sql(events_path: &str, row_ids: &[String]) -> Option<SqlQuery> {
    let placeholders = placeholders(row_ids.len())?;
    let sql = format!(
        r#"
SELECT
{EVENT_COLUMNS}
FROM read_parquet(?)
WHERE row_id IN ({placeholders})
ORDER BY start_ns ASC, row_id ASC
"#
    );
    let mut params = vec![Value::Text(events_path.to_string())];
    params.extend(row_ids.iter().cloned().map(Value::Text));
    Some(SqlQuery { sql, params })
}

pub(crate) fn children_sql(events_path: &str, parent_row_ids: &[String]) -> Option<SqlQuery> {
    let placeholders = placeholders(parent_row_ids.len())?;
    let sql = format!(
        r#"
SELECT
{EVENT_COLUMNS}
FROM read_parquet(?)
WHERE parent_row_id IN ({placeholders})
ORDER BY parent_row_id ASC, start_ns ASC, row_id ASC
"#
    );
    let mut params = vec![Value::Text(events_path.to_string())];
    params.extend(parent_row_ids.iter().cloned().map(Value::Text));
    Some(SqlQuery { sql, params })
}

pub(crate) fn links_sql(links_path: &str, row_ids: &[String]) -> Option<SqlQuery> {
    let from_placeholders = placeholders(row_ids.len())?;
    let to_placeholders = placeholders(row_ids.len())?;
    let sql = format!(
        r#"
SELECT
  key,
  from_row_id,
  to_row_id,
  kind,
  confidence
FROM read_parquet(?)
WHERE from_row_id IN ({from_placeholders})
   OR to_row_id IN ({to_placeholders})
ORDER BY from_row_id ASC, to_row_id ASC, kind ASC, confidence ASC
"#
    );
    let mut params = vec![Value::Text(links_path.to_string())];
    params.extend(row_ids.iter().cloned().map(Value::Text));
    params.extend(row_ids.iter().cloned().map(Value::Text));
    Some(SqlQuery { sql, params })
}

pub(crate) fn links_to_sql(links_path: &str, row_ids: &[String]) -> Option<SqlQuery> {
    let placeholders = placeholders(row_ids.len())?;
    let sql = format!(
        r#"
SELECT
  key,
  from_row_id,
  to_row_id,
  kind,
  confidence
FROM read_parquet(?)
WHERE to_row_id IN ({placeholders})
ORDER BY from_row_id ASC, to_row_id ASC, kind ASC, confidence ASC
"#
    );
    let mut params = vec![Value::Text(links_path.to_string())];
    params.extend(row_ids.iter().cloned().map(Value::Text));
    Some(SqlQuery { sql, params })
}

pub(crate) fn args_sql(args_path: &str, row_ids: &[String]) -> Option<SqlQuery> {
    let placeholders = placeholders(row_ids.len())?;
    let sql = format!(
        r#"
SELECT
  row_id,
  arg_key,
  arg_json
FROM read_parquet(?)
WHERE row_id IN ({placeholders})
ORDER BY row_id ASC, arg_key ASC
"#
    );
    let mut params = vec![Value::Text(args_path.to_string())];
    params.extend(row_ids.iter().cloned().map(Value::Text));
    Some(SqlQuery { sql, params })
}

pub(crate) fn python_parent_sql(
    events_path: &str,
    trace_index: i64,
    pid: Option<i64>,
    tid: Option<i64>,
    python_id: i64,
) -> SqlQuery {
    let mut predicates = vec![
        "type = 'python'".to_string(),
        "trace_index = ?".to_string(),
        "python_id = ?".to_string(),
    ];
    let mut params = vec![
        Value::Text(events_path.to_string()),
        Value::BigInt(trace_index),
        Value::BigInt(python_id),
    ];
    push_optional_i64_predicate(&mut predicates, &mut params, "pid", pid);
    push_optional_i64_predicate(&mut predicates, &mut params, "tid", tid);
    let where_clause = predicates.join(" AND ");
    let sql = format!(
        r#"
SELECT
{EVENT_COLUMNS}
FROM read_parquet(?)
WHERE {where_clause}
ORDER BY start_ns ASC, row_id ASC
LIMIT 1
"#
    );
    SqlQuery { sql, params }
}

fn push_optional_i64_predicate(
    predicates: &mut Vec<String>,
    params: &mut Vec<Value>,
    column: &'static str,
    value: Option<i64>,
) {
    if let Some(value) = value {
        predicates.push(format!("{column} = ?"));
        params.push(Value::BigInt(value));
    } else {
        predicates.push(format!("{column} IS NULL"));
    }
}

fn placeholders(count: usize) -> Option<String> {
    if count == 0 {
        return None;
    }
    Some(
        std::iter::repeat_n("?", count)
            .collect::<Vec<_>>()
            .join(", "),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_by_row_ids_sql_keeps_values_parameterized() -> Result<(), Box<dyn std::error::Error>>
    {
        let row_ids = vec!["cpu_op:1".to_string(), "kernel:2".to_string()];
        let query = events_by_row_ids_sql("events.parquet", &row_ids)
            .ok_or_else(|| std::io::Error::other("non-empty row ids should produce SQL"))?;

        assert!(query.sql.contains("FROM read_parquet(?)"));
        assert!(query.sql.contains("row_id IN (?, ?)"));
        assert_eq!(
            query.params,
            vec![
                Value::Text("events.parquet".to_string()),
                Value::Text("cpu_op:1".to_string()),
                Value::Text("kernel:2".to_string()),
            ]
        );
        Ok(())
    }

    #[test]
    fn python_parent_sql_handles_null_pid_tid_without_params() {
        let query = python_parent_sql("events.parquet", 3, None, None, 7);

        assert!(query.sql.contains("pid IS NULL"));
        assert!(query.sql.contains("tid IS NULL"));
        assert_eq!(
            query.params,
            vec![
                Value::Text("events.parquet".to_string()),
                Value::BigInt(3),
                Value::BigInt(7),
            ]
        );
    }
}
