use crate::views::meta::{push_nvtx_scope_meta, push_time_window_meta};
use veloq_core::tabular::{TabularView, cell_opt, push_count_meta, push_optional_meta};
use veloq_nsys_query::{
    graph_replays::GraphReplaysResponse,
    slices::{SlicesResponse, SlicesRow},
};

pub fn graph_replays_view(data: &GraphReplaysResponse) -> TabularView {
    let mut v = TabularView::new(vec![
        "synthetic_id",
        "capture_mode",
        "device_id",
        "context_id",
        "correlation_id",
        "launcher_row_id",
        "start_ns",
        "end_ns",
        "wall_ns",
        "sum_gpu_ns",
        "busy_ns",
        "idle_inside_replay_ns",
        "event_count",
        "kernel_count",
        "memcpy_count",
        "memset_count",
        "stream_count",
        "graph_id",
        "graph_exec_id",
        "decomposition_available",
        "top_node_name",
        "top_node_sum_ns",
    ]);
    for r in &data.rows {
        let top = r.top_nodes.first();
        v.push_row(vec![
            r.synthetic_id.clone(),
            r.capture_mode.to_string(),
            r.device_id.to_string(),
            r.context_id.to_string(),
            r.correlation_id.to_string(),
            r.launcher_row_id
                .map(|id| id.to_string())
                .unwrap_or_default(),
            r.start_ns.to_string(),
            r.end_ns.to_string(),
            r.wall_ns.to_string(),
            r.sum_gpu_ns.to_string(),
            r.busy_ns.to_string(),
            r.idle_inside_replay_ns.to_string(),
            r.event_count.to_string(),
            r.kernel_count.to_string(),
            r.memcpy_count.to_string(),
            r.memset_count.to_string(),
            r.stream_count.to_string(),
            cell_opt(r.graph_id),
            cell_opt(r.graph_exec_id),
            r.decomposition_available.to_string(),
            top.map(|n| n.name.clone()).unwrap_or_default(),
            top.map(|n| n.sum_ns.to_string()).unwrap_or_default(),
        ]);
    }
    v.push_meta("capture_mode", data.capture_mode.to_string());
    push_count_meta(&mut v, data.count, data.total_matched);
    v.push_meta("top_nodes_limit", data.top_nodes_limit.to_string());
    push_time_window_meta(&mut v, data.time_window_ns);
    push_nvtx_scope_meta(&mut v, data.nvtx_scope.as_deref());
    v
}

pub fn slices_view(data: &SlicesResponse) -> TabularView {
    if data.view == "aggregate" {
        let mut v = TabularView::new(vec![
            "name",
            "path",
            "instances",
            "attributed_total_ns",
            "p50_ns",
            "p99_ns",
        ]);
        for row in &data.rows {
            let SlicesRow::Aggregate(r) = row else {
                continue;
            };
            v.push_row(vec![
                r.name.clone(),
                r.path.clone().unwrap_or_default(),
                r.instances.to_string(),
                r.attributed_total_ns.to_string(),
                format!("{:.0}", r.p50_ns),
                format!("{:.0}", r.p99_ns),
            ]);
        }
        push_slices_meta(&mut v, data);
        return v;
    }

    // Flatten: one row per (slice, device, stream). Slice-level fields
    // repeat across its sub-rows; the agent can collapse them again on
    // (row_id) if they care, and humans see the relationship clearly.
    let mut v = TabularView::new(vec![
        "slice_row_id",
        "slice_name",
        "cpu_start_ns",
        "cpu_end_ns",
        "cpu_duration_ns",
        "device_id",
        "stream_id",
        "gpu_start_ns",
        "gpu_end_ns",
        "kernel_ns",
        "kernel_count",
        "memcpy_ns",
        "memcpy_count",
        "memset_ns",
        "memset_count",
    ]);
    for row in &data.rows {
        let SlicesRow::Instance(s) = row else {
            continue;
        };
        if s.gpu_attributed.is_empty() {
            // Slice with no GPU work attributed — still emit one row so
            // the user sees it. Stream columns become blank.
            v.push_row(vec![
                s.row_id.to_string(),
                s.name.clone(),
                s.cpu.start_ns.to_string(),
                s.cpu.end_ns.to_string(),
                s.cpu.duration_ns.to_string(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                "0".into(),
                "0".into(),
                "0".into(),
                "0".into(),
                "0".into(),
                "0".into(),
            ]);
        } else {
            for g in &s.gpu_attributed {
                v.push_row(vec![
                    s.row_id.to_string(),
                    s.name.clone(),
                    s.cpu.start_ns.to_string(),
                    s.cpu.end_ns.to_string(),
                    s.cpu.duration_ns.to_string(),
                    g.device_id.to_string(),
                    g.stream_id.to_string(),
                    g.start_ns.to_string(),
                    g.end_ns.to_string(),
                    g.kernel_ns.to_string(),
                    g.kernel_count.to_string(),
                    g.memcpy_ns.to_string(),
                    g.memcpy_count.to_string(),
                    g.memset_ns.to_string(),
                    g.memset_count.to_string(),
                ]);
            }
        }
    }
    push_slices_meta(&mut v, data);
    v
}

fn push_slices_meta(v: &mut TabularView, data: &SlicesResponse) {
    v.push_meta("view", data.view.to_string());
    push_optional_meta(v, "group_by", data.group_by);
    push_optional_meta(v, "name", data.name.as_deref());
    push_optional_meta(v, "name_regex", data.name_regex.as_deref());
    v.push_meta("attribution", data.attribution.to_string());
    push_count_meta(v, data.rows.len(), data.total_matched);
    push_time_window_meta(v, data.time_window_ns);
}
