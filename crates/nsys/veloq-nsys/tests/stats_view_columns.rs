//! stats_view's tabular columns must include every StatRow identity
//! axis — without them, `--format table/csv` collapses rows that
//! differ only on the newer group-by axes into duplicates.

use std::collections::HashSet;
use veloq_nsys::views;
use veloq_nsys_query::stats::{StatRow, StatsResponse};

#[test]
fn stats_view_carries_every_identity_axis() {
    let empty = StatsResponse {
        count: 0,
        total_matched: 0,
        total_duration_ns: 0,
        total_events: 0,
        time_window_ns: None,
        nvtx_scope: None,
        histogram_buckets_ns: None,
        mangled_axis_fallback: false,
        rows: Vec::<StatRow>::new(),
    };
    let view = views::stats_view(&empty);
    let cols: HashSet<&str> = view.columns.iter().map(String::as_str).collect();

    // Pre-existing identity columns.
    for required in [
        "type",
        "name",
        "short_name",
        "device_id",
        "context_id",
        "stream_id",
    ] {
        assert!(
            cols.contains(required),
            "stats_view missing baseline column `{required}`"
        );
    }
    // Extended grouping axes — each row's identity depends on these
    // when the corresponding axis is active.
    for required in [
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
    ] {
        assert!(
            cols.contains(required),
            "stats_view missing identity axis `{required}` — rows that differ \
             only on this axis collapse into indistinguishable CSV output"
        );
    }
}
