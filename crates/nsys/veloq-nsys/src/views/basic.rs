use crate::views::meta::{push_nvtx_scope_meta, push_time_window_meta};
use veloq_core::tabular::{TabularView, cell_opt, push_count_meta};
use veloq_nsys_query::{
    concurrency::ConcurrencyResponse, gaps::GapsResponse, search::SearchResponse,
    timeline::TimelineResponse, viz_timeline::VizTimelineResponse,
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

pub fn viz_timeline_view(data: &VizTimelineResponse) -> TabularView {
    let mut v = TabularView::new(vec![
        "path",
        "format",
        "track_count",
        "rendered_item_count",
        "total_item_count",
        "density_item_count",
        "density_bin_count",
        "density_duration_ns",
        "omitted_explicit_item_count",
        "aggregated",
        "omitted_track_count",
        "suppressed_label_count",
        "truncated_label_count",
    ]);
    for row in &data.rows {
        v.push_row(vec![
            row.path.clone(),
            row.format.clone(),
            row.track_count.to_string(),
            row.rendered_item_count.to_string(),
            row.total_item_count.to_string(),
            row.density_item_count.to_string(),
            row.density_bin_count.to_string(),
            row.density_duration_ns.to_string(),
            row.omitted_explicit_item_count.to_string(),
            row.aggregated.to_string(),
            row.omitted_track_count.to_string(),
            row.suppressed_label_count.to_string(),
            row.truncated_label_count.to_string(),
        ]);
    }
    if let Some(row) = data.rows.first() {
        let [start, end] = row.time_window_ns;
        v.push_meta("time_window_ns", format!("{start}-{end}"));
    }
    push_count_meta(&mut v, data.count, data.total_matched);
    v.push_meta(
        "requested_tracks",
        data.auxiliary.requested_tracks.len().to_string(),
    );
    v.push_meta(
        "resolved_tracks",
        data.auxiliary.resolved_tracks.len().to_string(),
    );
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use veloq_nsys_query::viz_timeline::{
        VizLabelPolicyEcho, VizRenderPolicyEcho, VizTimelineAuxiliary, VizTimelineFigureRow,
    };
    use veloq_vis::{VizLabelPolicy, VizRenderPolicy};

    #[test]
    fn viz_timeline_view_emits_time_window_meta_once() -> anyhow::Result<()> {
        let data = VizTimelineResponse {
            count: 2,
            total_matched: 2,
            rows: vec![figure_row("a.svg", [10, 20]), figure_row("b.svg", [10, 20])],
            auxiliary: VizTimelineAuxiliary {
                requested_tracks: Vec::new(),
                resolved_tracks: Vec::new(),
                requested_highlights: Vec::new(),
                resolved_highlights: Vec::new(),
                unresolved_highlights: Vec::new(),
                render_policy: VizRenderPolicyEcho::from(&VizRenderPolicy::default()),
                label_policy: VizLabelPolicyEcho::from(&VizLabelPolicy::default()),
            },
        };

        let view = viz_timeline_view(&data);
        let time_window_meta = view
            .meta
            .iter()
            .filter(|(key, _)| key == "time_window_ns")
            .collect::<Vec<_>>();
        assert_eq!(time_window_meta.len(), 1);
        let (_, value) = time_window_meta
            .first()
            .ok_or_else(|| anyhow::anyhow!("missing time_window_ns meta"))?;
        assert_eq!(value.as_str(), "10-20");
        Ok(())
    }

    fn figure_row(path: &str, time_window_ns: [i64; 2]) -> VizTimelineFigureRow {
        VizTimelineFigureRow {
            key: format!("figure|{path}"),
            path: path.to_string(),
            format: "svg".to_string(),
            time_window_ns,
            track_count: 0,
            rendered_item_count: 0,
            total_item_count: 0,
            density_item_count: 0,
            density_bin_count: 0,
            density_duration_ns: 0,
            omitted_explicit_item_count: 0,
            aggregated: false,
            omitted_track_count: 0,
            suppressed_label_count: 0,
            truncated_label_count: 0,
        }
    }
}
