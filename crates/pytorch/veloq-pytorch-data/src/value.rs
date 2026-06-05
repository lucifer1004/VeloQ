use serde_json::{Map, Value};
use std::collections::BTreeMap;

pub(crate) fn args_map(value: Option<&Value>) -> BTreeMap<String, Value> {
    value
        .and_then(Value::as_object)
        .map(|obj| {
            obj.iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect()
        })
        .unwrap_or_default()
}

pub(crate) fn string_field(obj: &Map<String, Value>, key: &str) -> Option<String> {
    obj.get(key).and_then(value_to_string)
}

pub(crate) fn top_value_string(obj: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(value) = obj.get(*key).and_then(value_to_string) {
            return Some(value);
        }
    }
    None
}

pub(crate) fn value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

pub(crate) fn value_to_i64(value: &Value) -> Option<i64> {
    value.as_i64().or_else(|| {
        value
            .as_u64()
            .and_then(|value| i64::try_from(value).ok())
            .or_else(|| value.as_f64().map(|value| value as i64))
    })
}

pub(crate) fn value_to_ns(value: &Value) -> Option<i64> {
    value_to_i64(value)
        .map(|value| value.saturating_mul(1_000))
        .or_else(|| value.as_f64().map(|value| (value * 1_000.0) as i64))
}

pub(crate) fn int_from_args(args: &BTreeMap<String, Value>, keys: &[&str]) -> Option<i64> {
    for key in keys {
        if let Some(value) = args.get(*key).and_then(value_to_i64) {
            return Some(value);
        }
    }
    None
}

pub(crate) fn string_from_args(args: &BTreeMap<String, Value>, keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(value) = args.get(*key).and_then(value_to_string) {
            return Some(value);
        }
    }
    None
}

pub(crate) fn value_string_contains(value: &Value, needle: &str) -> bool {
    value_to_string(value)
        .map(|text| text.to_ascii_lowercase().contains(needle))
        .unwrap_or(false)
}
