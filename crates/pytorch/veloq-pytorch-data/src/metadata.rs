use crate::value::{value_to_i64, value_to_string};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::path::Path;

pub(crate) fn rank_from_top(top: &Map<String, Value>) -> Option<i64> {
    top.get("distributedInfo")
        .and_then(Value::as_object)
        .and_then(|obj| obj.get("rank"))
        .and_then(value_to_i64)
        .or_else(|| top.get("rank").and_then(value_to_i64))
}

pub(crate) fn worker_from_top(top: &Map<String, Value>) -> Option<String> {
    top.get("distributedInfo")
        .and_then(Value::as_object)
        .and_then(|obj| obj.get("worker"))
        .and_then(value_to_string)
        .or_else(|| top.get("worker").and_then(value_to_string))
}

pub(crate) fn infer_rank_from_path(path: &Path) -> Option<i64> {
    let text = path.file_name()?.to_str()?.to_ascii_lowercase();
    let chars: Vec<char> = text.chars().collect();
    for (idx, _) in chars.iter().enumerate() {
        let tail: String = chars.iter().skip(idx).collect();
        if let Some(rest) = tail.strip_prefix("rank") {
            let digits: String = rest
                .chars()
                .skip_while(|c| matches!(c, '_' | '-' | '=' | '.'))
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if let Ok(rank) = digits.parse::<i64>() {
                return Some(rank);
            }
        }
    }
    None
}

pub(crate) fn infer_worker_from_path(path: &Path) -> Option<String> {
    let text = path.file_stem()?.to_str()?;
    Some(text.to_string()).filter(|s| !s.is_empty())
}

pub(crate) fn version_from_top(top: &Map<String, Value>, needle: &str) -> Option<String> {
    let needle = needle.to_ascii_lowercase();
    for (key, value) in top {
        let lower_key = key.to_ascii_lowercase();
        if lower_key.contains(&needle)
            && lower_key.contains("version")
            && let Some(version) = value_to_string(value)
        {
            return Some(version);
        }
    }
    None
}

pub(crate) fn capture_flags(top: &Map<String, Value>) -> BTreeMap<String, Value> {
    let mut out = BTreeMap::new();
    for (key, value) in top {
        let lower = key.to_ascii_lowercase();
        if key == "traceEvents" || key == "deviceProperties" {
            continue;
        }
        if lower.contains("profile") || lower.contains("capture") || lower.contains("with_") {
            out.insert(key.clone(), value.clone());
        }
    }
    out
}
