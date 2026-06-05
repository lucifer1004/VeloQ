//! Metric-name → base-unit inference.
//!
//! Pure string analysis: NCU's section metadata often omits `Unit`
//! because the official tooling can derive it from the metric name.
//! This covers the common grammar so views keep useful unit labels
//! and the `source-metrics` additivity gate can recognise rate-family
//! counters — all without scaling the raw metric values.

/// Infer the base display unit NCU would normally get from its metric
/// database. Section metadata often omits the unit because the official
/// tooling can derive it from the metric name; this covers the common
/// grammar so CSV/table views keep useful unit labels without scaling
/// the raw metric values.
pub fn infer_metric_unit(name: &str) -> Option<&'static str> {
    let name = canonical_metric_name(name);

    if is_percent_metric(name) {
        return Some("percent");
    }

    if is_nanosecond_metric(name) {
        return Some("nsecond");
    }

    if name.ends_with(".per_second") {
        let base = infer_counter_unit(name.trim_end_matches(".per_second"))?;
        return match base {
            "cycle" => Some("hertz"),
            "byte" => Some("byte_per_second"),
            "sector" => Some("sector_per_second"),
            "request" => Some("request_per_second"),
            "inst" => Some("inst_per_second"),
            "warp" => Some("warp_per_second"),
            "thread" => Some("thread_per_second"),
            _ => None,
        };
    }

    if name.ends_with(".per_cycle_active") || name.ends_with(".per_cycle_elapsed") {
        let stem = name
            .trim_end_matches(".per_cycle_active")
            .trim_end_matches(".per_cycle_elapsed");
        let base = infer_counter_unit(stem)?;
        return match base {
            // NCU displays active/eligible warp averages as warps, not
            // warp/cycle, because the denominator is part of the
            // averaging mode rather than the physical unit.
            "warp" => Some("warp"),
            "byte" => Some("byte_per_cycle"),
            "sector" => Some("sector_per_cycle"),
            "inst" => Some("inst_per_cycle"),
            "thread" => Some("thread_per_cycle"),
            _ => None,
        };
    }

    if name.ends_with(".peak_sustained") {
        let stem = name.trim_end_matches(".peak_sustained");
        let base = infer_counter_unit(stem)?;
        return match base {
            "byte" => Some("byte_per_cycle"),
            "sector" => Some("sector_per_cycle"),
            "inst" => Some("inst_per_cycle"),
            "warp" => Some("warp"),
            _ => Some(base),
        };
    }

    infer_counter_unit(name)
}

fn is_percent_metric(name: &str) -> bool {
    name.ends_with(".pct")
        || name.contains(".pct_")
        || name.ends_with("_pct")
        || name.contains("_pct_")
}

fn is_nanosecond_metric(name: &str) -> bool {
    name == "gpu__time_duration.sum"
        || name.ends_with("__time_duration.sum")
        || name.ends_with("_interval_time")
}

fn infer_counter_unit(name: &str) -> Option<&'static str> {
    let metric = canonical_metric_name(name);
    if metric.starts_with("device__attribute_") {
        return None;
    }

    if let Some(unit) = infer_launch_unit(metric) {
        return Some(unit);
    }

    if metric.contains("pmsampler_dropped_samples") || metric.contains("pmsampler_merged_samples") {
        return Some("sample");
    }
    if metric.contains("replayer_passes") || metric.contains("pcsamp_aggregated_passes") {
        return Some("pass");
    }
    if metric.contains("branch_targets")
        || metric.contains("pcsamp_warps_issue_stalled_branch_resolving")
    {
        return Some("branches");
    }
    if metric.contains("bytes") || metric.ends_with("_byte") {
        return Some("byte");
    }
    if metric.contains("sectors") || metric.contains("_sector") {
        return Some("sector");
    }
    if metric.contains("requests") || metric.contains("_request") {
        return Some("request");
    }
    if metric.contains("cycles") || metric.contains("_cycle") {
        return Some("cycle");
    }
    if metric.contains("warps") || metric.contains("_warp") {
        return Some("warp");
    }
    if metric.contains("__inst")
        || metric.starts_with("inst_")
        || metric.contains("_inst_")
        || metric.ends_with("_inst")
        || metric.ends_with("_inst_executed")
    {
        return Some("inst");
    }
    if metric.contains("threads") || metric.contains("_thread") {
        return Some("thread");
    }

    None
}

fn canonical_metric_name(name: &str) -> &str {
    let mut offset = 0;
    for part in name.split('.') {
        if looks_like_metric_start(part) {
            return &name[offset..];
        }
        offset += part.len() + 1;
    }
    name
}

fn looks_like_metric_start(part: &str) -> bool {
    part.contains("__")
        || part.starts_with("inst_")
        || part.starts_with("thread_inst")
        || part.starts_with("memory_")
}

fn infer_launch_unit(metric: &str) -> Option<&'static str> {
    match metric {
        "launch__block_dim_x" | "launch__block_dim_y" | "launch__block_dim_z" => Some("block"),
        "launch__occupancy_limit_blocks"
        | "launch__occupancy_limit_registers"
        | "launch__occupancy_limit_shared_mem"
        | "launch__occupancy_limit_warps" => Some("block"),
        "launch__registers_per_thread" | "launch__registers_per_thread_allocated" => {
            Some("register_per_thread")
        }
        "launch__shared_mem_config_size" => Some("byte"),
        "launch__shared_mem_per_block"
        | "launch__shared_mem_per_block_allocated"
        | "launch__shared_mem_per_block_driver"
        | "launch__shared_mem_per_block_dynamic"
        | "launch__shared_mem_per_block_static" => Some("byte_per_block"),
        "launch__thread_count" => Some("thread"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infers_common_ncu_base_units_from_metric_names() {
        let cases = [
            ("gpu__time_duration.sum", Some("nsecond")),
            ("SM_A.TriageAC.sm__cycles_active.avg", Some("cycle")),
            ("dram__bytes.sum.per_second", Some("byte_per_second")),
            ("gpc__cycles_elapsed.avg.per_second", Some("hertz")),
            (
                "sm__inst_executed.avg.per_cycle_active",
                Some("inst_per_cycle"),
            ),
            (
                "lts__t_sectors.avg.per_cycle_elapsed",
                Some("sector_per_cycle"),
            ),
            (
                "launch__shared_mem_per_block_static",
                Some("byte_per_block"),
            ),
            ("launch__registers_per_thread", Some("register_per_thread")),
            ("smsp__pcsamp_aggregated_passes", Some("pass")),
            ("profiler__pmsampler_dropped_samples", Some("sample")),
            ("smsp__branch_targets_threads_divergent", Some("branches")),
            ("device__attribute_clock_rate", None),
        ];

        for (name, expected) in cases {
            assert_eq!(infer_metric_unit(name), expected, "metric {name}");
        }
    }
}
