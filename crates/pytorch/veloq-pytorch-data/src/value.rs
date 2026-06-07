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
    let number = value.as_number()?;
    if let Some(value) = number.as_i64() {
        return Some(value.saturating_mul(1_000));
    }
    if let Some(value) = number.as_u64() {
        return Some(
            i64::try_from(value)
                .unwrap_or(i64::MAX)
                .saturating_mul(1_000),
        );
    }
    number.as_f64().and_then(f64_us_to_ns)
}

fn f64_us_to_ns(value: f64) -> Option<i64> {
    if !value.is_finite() {
        return None;
    }
    let ns = (value * 1_000.0).round();
    if ns >= i64::MAX as f64 {
        Some(i64::MAX)
    } else if ns <= i64::MIN as f64 {
        Some(i64::MIN)
    } else {
        Some(ns as i64)
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn value_to_ns_preserves_fractional_microseconds() {
        assert_eq!(value_to_ns(&json!(1.5)), Some(1_500));
        assert_eq!(value_to_ns(&json!(0.125)), Some(125));
        assert_eq!(value_to_ns(&json!(2)), Some(2_000));
    }
}
