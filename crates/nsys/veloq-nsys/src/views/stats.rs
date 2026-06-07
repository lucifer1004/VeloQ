use crate::views::meta::{push_nvtx_scope_meta, push_time_window_meta};
use veloq_core::tabular::{DISPLAY_PRECISION, TabularView, cell_opt, push_count_meta};
use veloq_nsys_query::{stats::StatsResponse, stats_by_size::StatsBySizeResponse};

pub fn stats_view(data: &StatsResponse) -> TabularView {
    // Identity columns mirror StatRow's group-key fields so the
    // table/CSV output can discriminate rows whose differences live
    // only in the group axes added after the initial schema (graph,
    // graph_node, grid_block, nvtx_parent, nvtx_path, nvtx_style,
    // event_type).
    // Without them, two rows that JSON serialises distinctly can
    // collapse into indistinguishable-looking duplicates under
    // --format table. Empty cells for axes the request didn't
    // activate match the JSON `skip_serializing_if = "Option::is_none"`
    // convention.
    let mut v = TabularView::new(vec![
        "type",
        "name",
        "short_name",
        "device_id",
        "context_id",
        "stream_id",
        "graph_id",
        "graph_node_id",
        "nvtx_parent_name",
        "nvtx_parent_depth",
        "nvtx_path",
        "nvtx_style",
        "event_type",
        "grid_x",
        "grid_y",
        "grid_z",
        "block_x",
        "block_y",
        "block_z",
        "count",
        "total_ns",
        "avg_ns",
        "min_ns",
        "max_ns",
        "p50_ns",
        "p95_ns",
        "p99_ns",
        "bytes_total",
        "gbps",
        "percentage",
    ]);
    for r in &data.rows {
        v.push_row(vec![
            r.kind.to_string(),
            r.name.clone().unwrap_or_default(),
            r.short_name.clone().unwrap_or_default(),
            cell_opt(r.device_id),
            cell_opt(r.context_id),
            cell_opt(r.stream_id),
            cell_opt(r.graph_id),
            cell_opt(r.graph_node_id),
            r.nvtx_parent_name.clone().unwrap_or_default(),
            cell_opt(r.nvtx_parent_depth),
            r.nvtx_path.clone().unwrap_or_default(),
            r.nvtx_style.unwrap_or_default().to_string(),
            cell_opt(r.event_type),
            cell_opt(r.grid_x),
            cell_opt(r.grid_y),
            cell_opt(r.grid_z),
            cell_opt(r.block_x),
            cell_opt(r.block_y),
            cell_opt(r.block_z),
            r.count.to_string(),
            r.total_ns.to_string(),
            r.avg_ns.to_string(),
            r.min_ns.to_string(),
            r.max_ns.to_string(),
            r.p50_ns.to_string(),
            r.p95_ns.to_string(),
            r.p99_ns.to_string(),
            cell_opt(r.bytes_total),
            cell_opt(r.gbps.map(|x| format!("{x:.*}", DISPLAY_PRECISION))),
            format!("{:.*}", DISPLAY_PRECISION, r.percentage),
        ]);
    }
    push_count_meta(&mut v, data.count, data.total_matched);
    v.push_meta("total_duration_ns", data.total_duration_ns.to_string());
    v.push_meta("total_events", data.total_events.to_string());
    push_time_window_meta(&mut v, data.time_window_ns);
    push_nvtx_scope_meta(&mut v, data.nvtx_scope.as_deref());
    if data.histogram_buckets_ns.is_some() {
        // Histogram doesn't fit a single-cell column gracefully; point
        // the user at JSON if they asked for --hist with a tabular format.
        // Underscore-prefixed key flags this as a human-readable hint
        // rather than a structured meta field — scripts can skip
        // anything starting with `_`.
        v.push_meta(
            "_note",
            "histogram values are JSON-only; rerun with `--format json` to see them",
        );
    }
    v
}

pub fn stats_by_size_view(data: &StatsBySizeResponse) -> TabularView {
    let mut v = TabularView::new(vec![
        "type",
        "name",
        "short_name",
        "device_id",
        "context_id",
        "stream_id",
        "count",
        "total_bytes",
        "avg_bytes",
        "min_bytes",
        "max_bytes",
        "p50_bytes",
        "p95_bytes",
        "p99_bytes",
        "percentage",
    ]);
    for r in &data.rows {
        v.push_row(vec![
            r.kind.to_string(),
            r.name.clone().unwrap_or_default(),
            r.short_name.clone().unwrap_or_default(),
            cell_opt(r.device_id),
            cell_opt(r.context_id),
            cell_opt(r.stream_id),
            r.count.to_string(),
            r.total_bytes.to_string(),
            r.avg_bytes.to_string(),
            r.min_bytes.to_string(),
            r.max_bytes.to_string(),
            r.p50_bytes.to_string(),
            r.p95_bytes.to_string(),
            r.p99_bytes.to_string(),
            format!("{:.*}", DISPLAY_PRECISION, r.percentage),
        ]);
    }
    push_count_meta(&mut v, data.count, data.total_matched);
    v.push_meta("total_bytes", data.total_bytes.to_string());
    v.push_meta("total_events", data.total_events.to_string());
    push_time_window_meta(&mut v, data.time_window_ns);
    v
}
