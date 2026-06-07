use crate::views::meta::{push_nvtx_scope_meta, push_time_window_meta};
use veloq_core::tabular::{TabularView, cell_opt, push_count_meta};
use veloq_nsys_query::{
    concurrency::ConcurrencyResponse, gaps::GapsResponse, search::SearchResponse,
    timeline::TimelineResponse,
};

pub fn search_view(data: &SearchResponse) -> TabularView {
    let mut v = TabularView::new(vec![
        "row_id",
        "name",
        "start_ns",
        "duration_ns",
        "device_id",
        "stream_id",
        "global_tid",
    ]);
    for e in &data.rows {
        let b = e.base();
        v.push_row(vec![
            b.row_id.to_string(),
            b.name.clone(),
            b.start_ns.to_string(),
            b.duration_ns.to_string(),
            cell_opt(b.device_id),
            cell_opt(b.stream_id),
            cell_opt(b.global_tid),
        ]);
    }
    push_count_meta(&mut v, data.count, data.total_matched);
    push_time_window_meta(&mut v, data.time_window_ns);
    push_nvtx_scope_meta(&mut v, data.nvtx_scope.as_deref());
    v
}

pub fn timeline_view(data: &TimelineResponse) -> TabularView {
    let mut v = TabularView::new(vec![
        "start_ns",
        "end_ns",
        "total_ns",
        "kernel_ns",
        "memcpy_ns",
        "memset_ns",
        "graph_ns",
        "count",
        "kernel_count",
        "memcpy_count",
        "memset_count",
        "graph_count",
    ]);
    for b in &data.rows {
        v.push_row(vec![
            b.start_ns.to_string(),
            b.end_ns.to_string(),
            b.total_ns.to_string(),
            b.kernel_ns.to_string(),
            b.memcpy_ns.to_string(),
            b.memset_ns.to_string(),
            b.graph_ns.to_string(),
            b.count.to_string(),
            b.kernel_count.to_string(),
            b.memcpy_count.to_string(),
            b.memset_count.to_string(),
            b.graph_count.to_string(),
        ]);
    }
    v.push_meta("interval_ns", data.interval_ns.to_string());
    push_count_meta(&mut v, data.count, data.total_matched);
    push_time_window_meta(&mut v, data.time_window_ns);
    push_nvtx_scope_meta(&mut v, data.nvtx_scope.as_deref());
    v
}

pub fn concurrency_view(data: &ConcurrencyResponse) -> TabularView {
    // Per-device summary. The full per-stream breakdown stays in the
    // JSON payload; the table collapses it to a stream count (same
    // "flatten nested to a scalar for tabular" choice as graph_replays).
    let mut v = TabularView::new(vec![
        "device_id",
        "sum_busy_ns",
        "union_busy_ns",
        "overlap_ns",
        "max_concurrency",
        "compute_union_ns",
        "copy_union_ns",
        "compute_copy_overlap_ns",
        "streams",
    ]);
    for d in &data.rows {
        v.push_row(vec![
            d.device_id.to_string(),
            d.sum_busy_ns.to_string(),
            d.union_busy_ns.to_string(),
            d.overlap_ns.to_string(),
            d.max_concurrency.to_string(),
            d.compute_vs_copy.compute_union_ns.to_string(),
            d.compute_vs_copy.copy_union_ns.to_string(),
            d.compute_vs_copy.compute_copy_overlap_ns.to_string(),
            d.streams.len().to_string(),
        ]);
    }
    push_count_meta(&mut v, data.count, data.total_matched);
    push_time_window_meta(&mut v, data.time_window_ns);
    v
}

pub fn gaps_view(data: &GapsResponse) -> TabularView {
    // Under unified scopes (device / trace) `g.device_id` /
    // `g.stream_id` are `None`; the bracketing events' stream lives
    // on prev/next, so render them too. Empty cells for the absent
    // axes keep the column count stable.
    let mut v = TabularView::new(vec![
        "device_id",
        "stream_id",
        "start_ns",
        "end_ns",
        "duration_ns",
        "prev_row_id",
        "prev_stream",
        "prev_name",
        "next_row_id",
        "next_stream",
        "next_name",
    ]);
    for g in &data.rows {
        v.push_row(vec![
            g.device_id.map(|d| d.to_string()).unwrap_or_default(),
            g.stream_id.map(|s| s.to_string()).unwrap_or_default(),
            g.start_ns.to_string(),
            g.end_ns.to_string(),
            g.duration_ns.to_string(),
            g.prev.row_id.to_string(),
            g.prev.stream_id.to_string(),
            g.prev.name.clone(),
            g.next.row_id.to_string(),
            g.next.stream_id.to_string(),
            g.next.name.clone(),
        ]);
    }
    v.push_meta("scope", data.scope.to_string());
    v.push_meta("min_ns", data.min_ns.to_string());
    push_count_meta(&mut v, data.count, data.total_matched);
    push_time_window_meta(&mut v, data.time_window_ns);
    v
}
