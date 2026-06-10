use crate::{VizInterval, VizRole, VizScene, VizTimeWindow, VizTrack};

pub(super) struct Layout {
    pub(super) width: f64,
    pub(super) height: f64,
    pub(super) label_width: f64,
    pub(super) plot_width: f64,
    pub(super) top: f64,
    track_offsets: Vec<f64>,
    track_area_height: f64,
    pub(super) highlight_rows: usize,
}

const TRACK_BASE_HEIGHT: f64 = 22.0;
const SUBLANE_HEIGHT: f64 = 16.0;

impl Layout {
    pub(super) fn new(width_px: u32, track_lane_counts: &[usize], highlights: usize) -> Self {
        let width = f64::from(width_px.max(480));
        let label_width = 240.0;
        let top = 34.0;
        let highlight_rows = highlights.min(6);
        let mut track_offsets = Vec::with_capacity(track_lane_counts.len());
        let mut track_area_height = 0.0;
        for lane_count in track_lane_counts {
            track_offsets.push(track_area_height);
            track_area_height += track_height(*lane_count);
        }
        track_area_height = track_area_height.max(TRACK_BASE_HEIGHT);
        let height = top + track_area_height + 42.0 + highlight_rows as f64 * 16.0;
        Self {
            width,
            height,
            label_width,
            plot_width: width - label_width - 20.0,
            top,
            track_offsets,
            track_area_height,
            highlight_rows,
        }
    }

    pub(super) fn track_y(&self, row: usize) -> f64 {
        self.top + self.track_offsets.get(row).copied().unwrap_or_default() + 4.0
    }

    pub(super) fn track_lane_y(&self, row: usize, lane: usize) -> f64 {
        self.track_y(row) + lane as f64 * SUBLANE_HEIGHT
    }

    pub(super) fn plot_bottom(&self) -> f64 {
        self.top + self.track_area_height + 2.0
    }

    pub(super) fn legend_y(&self) -> f64 {
        self.height - 28.0 - self.highlight_rows as f64 * 16.0
    }

    pub(super) fn highlight_legend_y(&self, row: usize) -> f64 {
        self.legend_y() + 16.0 * (row + 1) as f64
    }
}

fn track_height(lanes: usize) -> f64 {
    TRACK_BASE_HEIGHT + lanes.saturating_sub(1) as f64 * SUBLANE_HEIGHT
}
pub(super) struct LaneLayout {
    pub(super) lanes: Vec<usize>,
    pub(super) track_lane_counts: Vec<usize>,
}

pub(super) fn assign_lanes(
    tracks: &[&VizTrack],
    items: &[&VizInterval],
    scene: &VizScene,
) -> LaneLayout {
    let mut lanes = vec![0; items.len()];
    let mut track_lane_counts = vec![1; tracks.len()];

    for (track_idx, track) in tracks.iter().enumerate() {
        if !track_uses_sublanes(track.role) {
            continue;
        }
        let mut track_items = items
            .iter()
            .enumerate()
            .filter(|(_, item)| item.track_key == track.key)
            .collect::<Vec<_>>();
        track_items.sort_by(|(a_idx, a), (b_idx, b)| {
            a.start_ns
                .cmp(&b.start_ns)
                .then_with(|| a.end_ns.cmp(&b.end_ns))
                .then_with(|| a_idx.cmp(b_idx))
        });

        let mut lane_ends: Vec<i64> = Vec::new();
        for (item_idx, item) in track_items {
            let clipped_start = item.start_ns.max(scene.time_window.start_ns);
            let clipped_end = item.end_ns.min(scene.time_window.end_ns);
            if clipped_end <= clipped_start {
                continue;
            }
            let lane = if let Some(lane) = lane_ends.iter().position(|end| *end <= clipped_start) {
                if let Some(end) = lane_ends.get_mut(lane) {
                    *end = clipped_end;
                }
                lane
            } else {
                lane_ends.push(clipped_end);
                lane_ends.len().saturating_sub(1)
            };
            if let Some(item_lane) = lanes.get_mut(item_idx) {
                *item_lane = lane;
            }
        }
        if let Some(track_lanes) = track_lane_counts.get_mut(track_idx) {
            *track_lanes = lane_ends.len().max(1);
        }
    }

    LaneLayout {
        lanes,
        track_lane_counts,
    }
}

fn track_uses_sublanes(role: VizRole) -> bool {
    matches!(role, VizRole::Detail | VizRole::Annotation)
}

pub(super) fn overlaps_window(item: &VizInterval, window: VizTimeWindow) -> bool {
    item.end_ns > window.start_ns && item.start_ns < window.end_ns && item.end_ns > item.start_ns
}
