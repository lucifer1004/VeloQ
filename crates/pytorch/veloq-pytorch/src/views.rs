use std::collections::{BTreeMap, BTreeSet};
use veloq_core::tabular::{TabularResult, TabularView, push_count_meta};

pub fn generic_rows_view<T: serde::Serialize>(response: &T) -> TabularView {
    let value = serde_json::to_value(response).unwrap_or(serde_json::Value::Null);
    let count = value
        .get("count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let total = value
        .get("total_matched")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let rows = value
        .get("rows")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();

    let mut columns = BTreeSet::new();
    let mut flattened = Vec::new();
    for row in rows {
        let mut map = BTreeMap::new();
        flatten_row("", &row, &mut map);
        for column in map.keys() {
            columns.insert(column.clone());
        }
        flattened.push(map);
    }
    if columns.is_empty() {
        columns.insert("key".to_string());
    }
    let columns_vec = columns.into_iter().collect::<Vec<_>>();
    let mut view = TabularView::new(columns_vec.clone());
    push_count_meta(&mut view, count, total);
    for row in flattened {
        let cells = columns_vec
            .iter()
            .map(|column| row.get(column).cloned().unwrap_or_default())
            .collect();
        view.push_row(cells);
    }
    view
}

fn flatten_row(prefix: &str, value: &serde_json::Value, out: &mut BTreeMap<String, String>) {
    match value {
        serde_json::Value::Object(obj) => {
            for (key, value) in obj {
                let next = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                match value {
                    serde_json::Value::Object(_) => {
                        out.insert(next, value.to_string());
                    }
                    serde_json::Value::Array(_) => {
                        out.insert(next, value.to_string());
                    }
                    _ => flatten_row(&next, value, out),
                }
            }
        }
        serde_json::Value::Null => {
            out.insert(prefix.to_string(), String::new());
        }
        serde_json::Value::Bool(v) => {
            out.insert(prefix.to_string(), v.to_string());
        }
        serde_json::Value::Number(v) => {
            out.insert(prefix.to_string(), v.to_string());
        }
        serde_json::Value::String(v) => {
            out.insert(prefix.to_string(), v.clone());
        }
        serde_json::Value::Array(_) => {
            out.insert(prefix.to_string(), value.to_string());
        }
    }
}

pub fn emit_tabular<T: serde::Serialize>(
    response: &T,
    command: &str,
    trace: &str,
    fmt: veloq_core::OutputFormat,
) -> TabularResult<()> {
    let view = generic_rows_view(response);
    match fmt {
        veloq_core::OutputFormat::Json => {}
        veloq_core::OutputFormat::Csv => veloq_core::tabular::emit_csv(&view, command, trace)?,
        veloq_core::OutputFormat::Table => veloq_core::tabular::emit_table(&view, command, trace)?,
    }
    Ok(())
}
