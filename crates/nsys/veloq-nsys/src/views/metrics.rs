use crate::views::meta::push_time_window_meta;
use veloq_core::tabular::{
    DISPLAY_PRECISION, TabularView, cell_opt, push_count_meta, push_optional_meta,
};
use veloq_nsys_query::metrics::MetricsResponse;

/// Metrics view dispatch — each variant pulls its own column set
/// because the tabular shape diverges sharply per source (GPU
/// counters, CPU hotspot, sched summary). Common envelope facts
/// (`coverage`, `trace_span_ns`, …) flow through one helper so each
/// variant's view stays focused on its own columns.
pub fn metrics_view(data: &MetricsResponse) -> TabularView {
    match data {
        MetricsResponse::Gpu(b) => gpu_metrics_view(b),
        MetricsResponse::Nic(b) => nic_metrics_view(b),
        MetricsResponse::CpuSampling(b) => cpu_sampling_metrics_view(b),
        MetricsResponse::CpuSched(b) => cpu_sched_metrics_view(b),
    }
}

fn push_common_meta(
    v: &mut TabularView,
    count: usize,
    total_matched: i64,
    common: &veloq_nsys_query::metrics::MetricsCommon,
) {
    push_count_meta(v, count, total_matched);
    v.push_meta("samples_total", common.coverage.samples_total.to_string());
    if let Some(gap) = common.coverage.max_gap_ns {
        v.push_meta("max_gap_ns", gap.to_string());
    }
    v.push_meta(
        "coverage",
        format!(
            "{} / {} ns ({:.*})",
            common.coverage.covered_ns,
            common.coverage.trace_ns,
            DISPLAY_PRECISION,
            common.coverage.ratio
        ),
    );
    if let Some((lo, hi)) = common.metrics_span_ns {
        v.push_meta("metrics_span_ns", format!("{lo}-{hi}"));
    }
    v.push_meta(
        "trace_span_ns",
        format!("{}-{}", common.trace_span_ns.0, common.trace_span_ns.1),
    );
    push_optional_meta(v, "bucket_ns", common.bucket_ns);
    push_time_window_meta(v, common.time_window_ns);
}

fn gpu_metrics_view(body: &veloq_nsys_query::metrics::GpuMetricsBody) -> TabularView {
    let mut v = if body.auxiliary.common.bucket_ns.is_some() {
        let mut v = TabularView::new(vec![
            "t_start_ns",
            "t_end_ns",
            "type_id",
            "metric_id",
            "agg",
            "value",
            "samples",
        ]);
        for b in &body.auxiliary.buckets {
            v.push_row(vec![
                b.t_start_ns.to_string(),
                b.t_end_ns.to_string(),
                b.type_id.to_string(),
                b.metric_id.to_string(),
                b.agg.to_string(),
                format!("{:.*}", DISPLAY_PRECISION, b.value),
                b.samples.to_string(),
            ]);
        }
        v
    } else {
        let mut v = TabularView::new(vec![
            "metric_id",
            "name",
            "unit",
            "agg",
            "samples",
            "min",
            "max",
            "mean",
            "p50",
            "p95",
            "p99",
        ]);
        for c in &body.rows {
            v.push_row(vec![
                c.metric_id.to_string(),
                c.name.clone(),
                c.unit.clone().unwrap_or_default(),
                c.agg.to_string(),
                c.samples.to_string(),
                format!("{:.*}", DISPLAY_PRECISION, c.min),
                format!("{:.*}", DISPLAY_PRECISION, c.max),
                format!("{:.*}", DISPLAY_PRECISION, c.mean),
                format!("{:.*}", DISPLAY_PRECISION, c.p50),
                format!("{:.*}", DISPLAY_PRECISION, c.p95),
                format!("{:.*}", DISPLAY_PRECISION, c.p99),
            ]);
        }
        v
    };
    v.push_meta("source", "gpu".to_string());
    push_common_meta(
        &mut v,
        body.count,
        body.total_matched,
        &body.auxiliary.common,
    );
    push_optional_meta(
        &mut v,
        "counter_glob",
        body.auxiliary.counter_glob.as_deref(),
    );
    v
}

fn nic_metrics_view(body: &veloq_nsys_query::metrics::NicMetricsBody) -> TabularView {
    let mut v = if body.auxiliary.common.bucket_ns.is_some() {
        let mut v = TabularView::new(vec![
            "t_start_ns",
            "t_end_ns",
            "nic_id",
            "nic_name",
            "port_id",
            "metrics_list_id",
            "metrics_idx",
            "name",
            "unit",
            "agg",
            "value",
            "samples",
        ]);
        for b in &body.auxiliary.buckets {
            v.push_row(vec![
                b.t_start_ns.to_string(),
                b.t_end_ns.to_string(),
                b.nic_id.to_string(),
                b.nic_name.clone(),
                b.port_id.to_string(),
                b.metrics_list_id.to_string(),
                b.metrics_idx.to_string(),
                b.name.clone(),
                b.unit.clone().unwrap_or_default(),
                b.agg.to_string(),
                format!("{:.*}", DISPLAY_PRECISION, b.value),
                b.samples.to_string(),
            ]);
        }
        v
    } else {
        let mut v = TabularView::new(vec![
            "nic_id",
            "nic_name",
            "port_id",
            "metrics_list_id",
            "metrics_idx",
            "name",
            "unit",
            "agg",
            "samples",
            "min",
            "max",
            "mean",
            "p50",
            "p95",
            "p99",
        ]);
        for c in &body.rows {
            v.push_row(vec![
                c.nic_id.to_string(),
                c.nic_name.clone(),
                c.port_id.to_string(),
                c.metrics_list_id.to_string(),
                c.metrics_idx.to_string(),
                c.name.clone(),
                c.unit.clone().unwrap_or_default(),
                c.agg.to_string(),
                c.samples.to_string(),
                format!("{:.*}", DISPLAY_PRECISION, c.min),
                format!("{:.*}", DISPLAY_PRECISION, c.max),
                format!("{:.*}", DISPLAY_PRECISION, c.mean),
                format!("{:.*}", DISPLAY_PRECISION, c.p50),
                format!("{:.*}", DISPLAY_PRECISION, c.p95),
                format!("{:.*}", DISPLAY_PRECISION, c.p99),
            ]);
        }
        v
    };
    v.push_meta("source", "nic".to_string());
    push_common_meta(
        &mut v,
        body.count,
        body.total_matched,
        &body.auxiliary.common,
    );
    push_optional_meta(
        &mut v,
        "counter_glob",
        body.auxiliary.counter_glob.as_deref(),
    );
    v
}

fn cpu_sampling_metrics_view(body: &veloq_nsys_query::metrics::CpuSamplingBody) -> TabularView {
    let mut v = if body.auxiliary.common.bucket_ns.is_some() {
        cpu_buckets_view(&body.auxiliary.cpu_buckets)
    } else {
        let mut v = TabularView::new(vec![
            "key",
            "samples",
            "percentage",
            "sample_row_id",
            "sample_start_ns",
            "symbol",
            "module",
            "kernel",
            "unresolved",
            "cpu",
            "global_tid",
            "pid",
            "tid",
            "stack_hash",
            "stack_depth",
            "stack_frames",
        ]);
        for h in &body.rows {
            v.push_row(vec![
                h.key.clone(),
                h.samples.to_string(),
                format!("{:.*}", DISPLAY_PRECISION, h.percentage),
                h.sample_row_id.clone().unwrap_or_default(),
                cell_opt(h.sample_start_ns),
                h.symbol_name.clone().unwrap_or_default(),
                h.module_name.clone().unwrap_or_default(),
                h.kernel_mode.map(|b| b.to_string()).unwrap_or_default(),
                h.unresolved.map(|b| b.to_string()).unwrap_or_default(),
                cell_opt(h.cpu),
                cell_opt(h.global_tid),
                cell_opt(h.pid),
                cell_opt(h.tid),
                h.stack_hash.clone().unwrap_or_default(),
                cell_opt(h.stack_depth),
                h.stack_frames.join(" | "),
            ]);
        }
        v
    };
    v.push_meta("source", "cpu-sampling".to_string());
    push_common_meta(
        &mut v,
        body.count,
        body.total_matched,
        &body.auxiliary.common,
    );
    v.push_meta("group_by", body.auxiliary.group_by.to_string());
    push_optional_meta(&mut v, "name_glob", body.auxiliary.name_glob.as_deref());
    push_optional_meta(&mut v, "cpu_filter", body.auxiliary.cpu_filter);
    push_optional_meta(&mut v, "tid_filter", body.auxiliary.tid_filter);
    if let Some(r) = body.auxiliary.unresolved_leaf_share {
        v.push_meta(
            "unresolved_leaf_share",
            format!("{r:.*}", DISPLAY_PRECISION),
        );
    }
    if let Some(r) = body.auxiliary.kernel_leaf_share {
        v.push_meta("kernel_leaf_share", format!("{r:.*}", DISPLAY_PRECISION));
    }
    if let Some(r) = body.auxiliary.truncated_stack_share {
        v.push_meta(
            "truncated_stack_share",
            format!("{r:.*}", DISPLAY_PRECISION),
        );
    }
    v
}

fn cpu_sched_metrics_view(body: &veloq_nsys_query::metrics::CpuSchedBody) -> TabularView {
    let mut v = if body.auxiliary.common.bucket_ns.is_some() {
        cpu_buckets_view(&body.auxiliary.cpu_buckets)
    } else {
        let mut v = TabularView::new(vec![
            "key",
            "on_cpu_ns",
            "off_cpu_ns",
            "ctx_switches",
            "avg_quantum_ns",
            "observed_span_ns",
            "cpu",
            "global_tid",
            "pid",
            "tid",
            "state_id",
            "state_name",
            "distinct_tids",
        ]);
        for r in &body.rows {
            v.push_row(vec![
                r.key.clone(),
                r.on_cpu_ns.to_string(),
                cell_opt(r.off_cpu_ns),
                r.ctx_switches.to_string(),
                cell_opt(r.avg_quantum_ns),
                r.observed_span_ns.to_string(),
                cell_opt(r.cpu),
                cell_opt(r.global_tid),
                cell_opt(r.pid),
                cell_opt(r.tid),
                cell_opt(r.state_id),
                r.state_name.clone().unwrap_or_default(),
                cell_opt(r.distinct_tids),
            ]);
        }
        v
    };
    v.push_meta("source", "cpu-sched".to_string());
    push_common_meta(
        &mut v,
        body.count,
        body.total_matched,
        &body.auxiliary.common,
    );
    v.push_meta("group_by", body.auxiliary.group_by.to_string());
    push_optional_meta(&mut v, "cpu_filter", body.auxiliary.cpu_filter);
    push_optional_meta(&mut v, "tid_filter", body.auxiliary.tid_filter);
    if let Some(r) = body.auxiliary.unresolved_state_share {
        v.push_meta(
            "unresolved_state_share",
            format!("{r:.*}", DISPLAY_PRECISION),
        );
    }
    if let Some(g) = body.auxiliary.per_cpu_max_gap_ns {
        v.push_meta("per_cpu_max_gap_ns", g.to_string());
    }
    v
}

fn cpu_buckets_view(buckets: &[veloq_nsys_query::metrics::CpuBucketSample]) -> TabularView {
    let mut v = TabularView::new(vec![
        "t_start_ns",
        "t_end_ns",
        "key",
        "agg",
        "value",
        "samples",
    ]);
    for b in buckets {
        v.push_row(vec![
            b.t_start_ns.to_string(),
            b.t_end_ns.to_string(),
            b.key.clone(),
            b.agg.to_string(),
            format!("{:.*}", DISPLAY_PRECISION, b.value),
            b.samples.to_string(),
        ]);
    }
    v
}
