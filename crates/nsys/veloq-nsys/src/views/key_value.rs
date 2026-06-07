use veloq_core::tabular::TabularView;

/// Key/value flattener for responses without a natural row list
/// (prep, correlation-stats). Uses serde to walk the JSON shape so we
/// don't have to keep the projection in sync with the struct definition.
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
