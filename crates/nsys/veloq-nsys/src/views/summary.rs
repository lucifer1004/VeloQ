use veloq_core::tabular::TabularView;
use veloq_nsys_query::summary::Summary;

pub fn summary_view(data: &Summary) -> TabularView {
    let mut v = TabularView::new(vec!["table", "row_count", "start_ns", "end_ns"]);
    for t in &data.rows {
        v.push_row(vec![
            t.name.clone(),
            t.row_count.to_string(),
            t.start_ns.to_string(),
            t.end_ns.to_string(),
        ]);
    }
    if let Some(s) = &data.schema_version {
        v.push_meta("schema_version", s.clone());
    }
    if let Some(p) = &data.product_version {
        v.push_meta("product_version", p.clone());
    }
    v.push_meta(
        "full_time_range_ns",
        format!(
            "{}-{} (dur {})",
            data.auxiliary.full_time_range_ns.start,
            data.auxiliary.full_time_range_ns.end,
            data.auxiliary.full_time_range_ns.duration
        ),
    );
    if let Some(c) = &data.auxiliary.capabilities {
        // Compact one-line listing of `true` flags — agents reading
        // JSON get the structured shape; table consumers get a
        // human-readable summary that doesn't need vertical sprawl.
        let mut active: Vec<&str> = Vec::new();
        if c.has_kernels {
            active.push("kernels");
        }
        if c.has_memcpy {
            active.push("memcpy");
        }
        if c.has_memset {
            active.push("memset");
        }
        if c.has_sync {
            active.push("sync");
        }
        if c.has_runtime {
            active.push("runtime");
        }
        if c.has_osrt {
            active.push("osrt");
        }
        if c.has_nvtx {
            active.push("nvtx");
        }
        if c.has_cuda_contexts {
            active.push("cuda_contexts");
        }
        if c.has_sampling {
            active.push("sampling");
        }
        if c.has_composite_events {
            active.push("composite_events");
        }
        if c.has_gpu_metrics {
            active.push("gpu_metrics");
        }
        if c.has_nic_metrics {
            active.push("nic_metrics");
        }
        if c.has_target_info {
            active.push("target_info");
        }
        v.push_meta("capabilities", active.join(", "));
    }
    v
}
