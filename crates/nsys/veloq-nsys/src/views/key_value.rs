use crate::payloads::PrepPayload;
use veloq_core::tabular::{TabularView, cell_opt, push_count_meta};

/// Key/value flattener for responses without a natural row list
/// (correlation-stats). Uses serde to walk the JSON shape so we don't
/// have to keep the projection in sync with the struct definition.
pub fn key_value_view<T: serde::Serialize>(data: &T) -> TabularView {
    let value = serde_json::to_value(data).unwrap_or(serde_json::Value::Null);
    let mut v = TabularView::new(vec!["field", "value"]);
    if let serde_json::Value::Object(map) = value {
        for (k, vv) in map {
            v.push_row(vec![k, render_value(&vv)]);
        }
    } else {
        v.push_row(vec!["value".to_string(), render_value(&value)]);
    }
    v
}

pub fn prep_view(data: &PrepPayload) -> TabularView {
    let mut v = TabularView::new(vec![
        "key",
        "name",
        "present",
        "fingerprint_match",
        "format_version_expected",
        "format_version_on_disk",
        "size_bytes",
        "path",
    ]);
    for row in &data.rows {
        v.push_row(vec![
            row.key.clone(),
            row.name.clone(),
            row.present.to_string(),
            row.fingerprint_match.to_string(),
            row.format_version_expected.to_string(),
            cell_opt(row.format_version_on_disk),
            cell_opt(row.size_bytes),
            row.path.clone(),
        ]);
    }
    push_count_meta(&mut v, data.count, data.total_matched);
    v.push_meta("cache_root", data.auxiliary.cache_root.clone());
    v.push_meta("prepared", data.auxiliary.prepared.to_string());
    v.push_meta("elapsed_ms", data.auxiliary.elapsed_ms.to_string());
    v.push_meta(
        "parquet_cache_present",
        data.auxiliary.parquet_cache.present.to_string(),
    );
    v.push_meta(
        "parquet_cache_tables",
        data.auxiliary.parquet_cache.tables.len().to_string(),
    );
    v
}

fn render_value(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => String::new(),
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Array(a) => {
            // Compact: comma-joined scalars; objects/arrays fall back to JSON.
            let scalar_only = a.iter().all(|x| {
                !matches!(
                    x,
                    serde_json::Value::Array(_) | serde_json::Value::Object(_)
                )
            });
            if scalar_only {
                a.iter().map(render_value).collect::<Vec<_>>().join(", ")
            } else {
                serde_json::to_string(v).unwrap_or_default()
            }
        }
        serde_json::Value::Object(_) => serde_json::to_string(v).unwrap_or_default(),
    }
}
