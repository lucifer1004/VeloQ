use schemars::JsonSchema;
use serde::Serialize;
use veloq_core::time::TimeWindow;
use veloq_vis::{VizAxis, VizHighlight, VizHighlightScore, VizLabelPolicy, VizRenderPolicy};

#[derive(Debug, Clone, Default)]
pub struct VizTimelineRequest {
    pub time_window: Option<TimeWindow>,
    pub tracks: Vec<String>,
    pub highlight_kernels: Vec<String>,
    pub render_policy: VizRenderPolicy,
    pub label_policy: VizLabelPolicy,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct VizTimelineResponse {
    pub count: usize,
    pub total_matched: usize,
    pub rows: Vec<VizTimelineFigureRow>,
    pub auxiliary: VizTimelineAuxiliary,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct VizTimelineFigureRow {
    pub key: String,
    pub path: String,
    pub format: String,
    pub time_window_ns: [i64; 2],
    pub track_count: usize,
    pub rendered_item_count: usize,
    pub total_item_count: usize,
    pub density_item_count: usize,
    pub density_bin_count: usize,
    pub density_duration_ns: i64,
    pub omitted_explicit_item_count: usize,
    pub aggregated: bool,
    pub omitted_track_count: usize,
    pub suppressed_label_count: usize,
    pub truncated_label_count: usize,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct VizTimelineAuxiliary {
    pub requested_tracks: Vec<String>,
    pub resolved_tracks: Vec<VizResolvedTrack>,
    pub requested_highlights: Vec<String>,
    pub resolved_highlights: Vec<VizResolvedHighlight>,
    pub unresolved_highlights: Vec<VizUnresolvedHighlight>,
    pub render_policy: VizRenderPolicyEcho,
    pub label_policy: VizLabelPolicyEcho,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct VizResolvedTrack {
    pub track_key: String,
    pub label: String,
    pub kind: String,
    pub role: String,
    pub axes: Vec<VizAxis>,
    pub source_axes: Vec<VizAxis>,
    pub placement_axes: Vec<VizAxis>,
    pub placement_source: String,
    pub render_policy: String,
    pub label_policy: String,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct VizResolvedHighlight {
    pub key: String,
    pub rank: usize,
    pub color: String,
    pub label: String,
    pub full_name: String,
    pub scope: String,
    pub metric: String,
    pub score: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_total: Option<i64>,
    pub total_duration_ns: i64,
    pub instance_count: usize,
    pub max_duration_ns: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub row_id: Option<String>,
}

impl VizResolvedHighlight {
    pub(super) fn to_scene_highlight(&self) -> VizHighlight {
        VizHighlight {
            key: self.key.clone(),
            label: self.label.clone(),
            full_label: self.full_name.clone(),
            color: self.color.clone(),
            rank: Some(self.rank),
            scope: Some(self.scope.clone()),
            score: Some(VizHighlightScore {
                metric: self.metric.clone(),
                value: self.score,
                total: self.score_total,
            }),
        }
    }
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct VizUnresolvedHighlight {
    pub spec: String,
    pub reason: String,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct VizRenderPolicyEcho {
    pub width_px: u32,
    pub max_tracks: usize,
    pub max_items: usize,
    pub min_interval_px: f64,
    pub density_bin_px: f64,
    pub aggregation: String,
}

impl From<&VizRenderPolicy> for VizRenderPolicyEcho {
    fn from(policy: &VizRenderPolicy) -> Self {
        Self {
            width_px: policy.width_px,
            max_tracks: policy.max_tracks,
            max_items: policy.max_items,
            min_interval_px: policy.min_interval_px,
            density_bin_px: policy.density_bin_px,
            aggregation: policy.aggregation.to_string(),
        }
    }
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct VizLabelPolicyEcho {
    pub mode: String,
    pub min_label_px: f64,
}

impl From<&VizLabelPolicy> for VizLabelPolicyEcho {
    fn from(policy: &VizLabelPolicy) -> Self {
        Self {
            mode: policy.mode.to_string(),
            min_label_px: policy.min_label_px,
        }
    }
}
