use crate::model::EventType;
use crate::value::value_string_contains;
use serde_json::Value;
use std::collections::BTreeMap;

pub(crate) fn classify_event(
    name: &str,
    category: Option<&str>,
    args: &BTreeMap<String, Value>,
    is_comm: bool,
) -> EventType {
    let lower_name = name.to_ascii_lowercase();
    let lower_cat = category.unwrap_or("").to_ascii_lowercase();
    if lower_name.starts_with("profilerstep#") {
        return EventType::Step;
    }
    if lower_cat.contains("kernel") || lower_name.starts_with("nccldevkernel") {
        return EventType::Kernel;
    }
    if lower_cat.contains("gpu_memcpy") || lower_cat.contains("memcpy") {
        return EventType::Memcpy;
    }
    if lower_cat.contains("gpu_memset") || lower_cat.contains("memset") {
        return EventType::Memset;
    }
    if lower_cat.contains("cuda_runtime") {
        return EventType::Runtime;
    }
    if lower_cat.contains("cuda_driver") {
        return EventType::Driver;
    }
    if lower_cat.contains("user_annotation") {
        return EventType::Annotation;
    }
    if lower_cat.contains("python") {
        return EventType::Python;
    }
    if lower_cat.contains("memory")
        || args.contains_key("Device Type")
        || args.contains_key("Total Allocated")
    {
        return EventType::Memory;
    }
    if is_comm {
        return EventType::Comm;
    }
    if lower_cat.contains("cpu_op") || args.contains_key("External id") {
        return EventType::CpuOp;
    }
    EventType::CpuOp
}

pub(crate) fn is_comm_event(
    name: &str,
    category: Option<&str>,
    args: &BTreeMap<String, Value>,
) -> bool {
    let name = name.to_ascii_lowercase();
    let category = category.unwrap_or("").to_ascii_lowercase();
    name.contains("c10d::")
        || name.contains("record_param_comms")
        || name.contains("nccl")
        || category.contains("communication")
        || args.iter().any(|(key, value)| {
            key.to_ascii_lowercase().contains("comm") || value_string_contains(value, "nccl")
        })
}

pub(crate) fn collective_kind_from_name(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    if lower.contains("all_reduce") || lower.contains("allreduce") {
        "all_reduce"
    } else if lower.contains("all_gather") || lower.contains("allgather") {
        "all_gather"
    } else if lower.contains("reduce_scatter") || lower.contains("reducescatter") {
        "reduce_scatter"
    } else if lower.contains("broadcast") {
        "broadcast"
    } else if lower.contains("barrier") {
        "barrier"
    } else if lower.contains("send") {
        "send"
    } else if lower.contains("recv") || lower.contains("receive") {
        "recv"
    } else {
        "unknown"
    }
    .to_string()
}

pub(crate) fn parse_step_number(name: &str) -> Option<i64> {
    name.strip_prefix("ProfilerStep#")
        .and_then(|rest| rest.trim().parse::<i64>().ok())
}
