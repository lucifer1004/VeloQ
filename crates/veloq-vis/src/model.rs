use schemars::JsonSchema;
use serde::Serialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
pub struct VizTimeWindow {
    pub start_ns: i64,
    pub end_ns: i64,
}

impl VizTimeWindow {
    pub fn span_ns(self) -> i64 {
        self.end_ns.saturating_sub(self.start_ns)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, JsonSchema)]
pub struct VizAxis {
    pub name: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, JsonSchema)]
pub struct VizHighlight {
    pub key: String,
    pub label: String,
    pub full_label: String,
    pub color: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rank: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<VizHighlightScore>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, JsonSchema)]
pub struct VizHighlightScore {
    pub metric: String,
    pub value: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum VizRole {
    Group,
    Summary,
    Detail,
    Annotation,
    Overlay,
}

impl std::fmt::Display for VizRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Group => "group",
            Self::Summary => "summary",
            Self::Detail => "detail",
            Self::Annotation => "annotation",
            Self::Overlay => "overlay",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, JsonSchema)]
pub struct VizTrack {
    pub key: String,
    pub label: String,
    pub kind: String,
    pub role: VizRole,
    pub depth: usize,
    pub axes: Vec<VizAxis>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, JsonSchema)]
pub struct VizTrackMetadata {
    pub track_key: String,
    pub label: String,
    pub kind: String,
    pub role: String,
    pub axes: Vec<VizAxis>,
    pub source_axes: Vec<VizAxis>,
    pub placement_axes: Vec<VizAxis>,
    pub placement_source: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, JsonSchema)]
pub struct VizInterval {
    pub track_key: String,
    pub start_ns: i64,
    pub end_ns: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub row_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub class: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<VizRole>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub highlight_key: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum VizAggregation {
    None,
    ItemLimit,
    DensityBins,
}

impl std::fmt::Display for VizAggregation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::None => "none",
            Self::ItemLimit => "item_limit",
            Self::DensityBins => "density_bins",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct VizRenderPolicy {
    pub width_px: u32,
    pub max_tracks: usize,
    pub max_items: usize,
    pub min_interval_px: f64,
    pub density_bin_px: f64,
    pub aggregation: VizAggregation,
}

impl Default for VizRenderPolicy {
    fn default() -> Self {
        Self {
            width_px: 1200,
            max_tracks: 64,
            max_items: 5000,
            min_interval_px: 1.0,
            density_bin_px: 2.0,
            aggregation: VizAggregation::DensityBins,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum VizLabelMode {
    Auto,
    Hide,
}

impl std::fmt::Display for VizLabelMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Auto => "auto",
            Self::Hide => "hide",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct VizLabelPolicy {
    pub mode: VizLabelMode,
    pub min_label_px: f64,
}

impl Default for VizLabelPolicy {
    fn default() -> Self {
        Self {
            mode: VizLabelMode::Auto,
            min_label_px: 48.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct VizSceneMetadata {
    pub command: String,
    pub source_kind: String,
    pub source_version: String,
    pub tracks: Vec<VizTrackMetadata>,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct VizScene {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<VizSceneMetadata>,
    pub time_window: VizTimeWindow,
    pub tracks: Vec<VizTrack>,
    pub intervals: Vec<VizInterval>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub highlights: Vec<VizHighlight>,
    pub render_policy: VizRenderPolicy,
    pub label_policy: VizLabelPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvgRenderResult {
    pub svg: String,
    pub summary: SvgRenderSummary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, JsonSchema)]
pub struct SvgRenderSummary {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrittenSvgArtifact {
    pub path: PathBuf,
    pub relative_path: String,
    pub format: &'static str,
}
