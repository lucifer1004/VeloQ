//! Source-neutral static visualization primitives.
//!
//! Source crates own evidence extraction and track semantics. This
//! module owns only the portable scene shape, a small SVG renderer, and
//! artifact-root publishing for generated report figures.

use crate::{ErrorCode, VeloqDiagnostic};
use schemars::JsonSchema;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};
use thiserror::Error;

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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum VizAggregation {
    None,
    ItemLimit,
}

impl std::fmt::Display for VizAggregation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::None => "none",
            Self::ItemLimit => "item_limit",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct VizRenderPolicy {
    pub width_px: u32,
    pub max_tracks: usize,
    pub max_items: usize,
    pub min_interval_px: f64,
    pub aggregation: VizAggregation,
}

impl Default for VizRenderPolicy {
    fn default() -> Self {
        Self {
            width_px: 1200,
            max_tracks: 64,
            max_items: 5000,
            min_interval_px: 1.0,
            aggregation: VizAggregation::ItemLimit,
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
    pub max_chars: usize,
}

impl Default for VizLabelPolicy {
    fn default() -> Self {
        Self {
            mode: VizLabelMode::Auto,
            min_label_px: 48.0,
            max_chars: 32,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct VizScene {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub time_window: VizTimeWindow,
    pub tracks: Vec<VizTrack>,
    pub intervals: Vec<VizInterval>,
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

#[derive(Debug, Error)]
pub enum VisualizationError {
    #[error("visualization time window must be positive (start_ns={start_ns}, end_ns={end_ns})")]
    NonPositiveWindow { start_ns: i64, end_ns: i64 },

    #[error("visualization artifact path must be relative and stay under the artifact root")]
    UnsafeRelativePath,

    #[error("visualization artifact filename must be a plain `.svg` filename")]
    UnsafeSvgFileName,

    #[error("failed to create visualization artifact directory `{path}`")]
    CreateArtifactDir {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to write visualization artifact `{path}`")]
    WriteArtifact {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to publish visualization artifact `{path}`")]
    PublishArtifact {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

impl VeloqDiagnostic for VisualizationError {
    fn code(&self) -> ErrorCode {
        match self {
            Self::NonPositiveWindow { .. } => ErrorCode::new("viz.non-positive-window"),
            Self::UnsafeRelativePath => ErrorCode::new("viz.unsafe-relative-path"),
            Self::UnsafeSvgFileName => ErrorCode::new("viz.unsafe-svg-filename"),
            Self::CreateArtifactDir { .. } => ErrorCode::new("viz.artifact-dir-create"),
            Self::WriteArtifact { .. } => ErrorCode::new("viz.artifact-write"),
            Self::PublishArtifact { .. } => ErrorCode::new("viz.artifact-publish"),
        }
    }
}

pub fn render_svg(scene: &VizScene) -> Result<SvgRenderResult, VisualizationError> {
    if scene.time_window.end_ns <= scene.time_window.start_ns {
        return Err(VisualizationError::NonPositiveWindow {
            start_ns: scene.time_window.start_ns,
            end_ns: scene.time_window.end_ns,
        });
    }

    let rendered_tracks: Vec<&VizTrack> = scene
        .tracks
        .iter()
        .take(scene.render_policy.max_tracks)
        .collect();
    let rendered_track_keys: BTreeSet<&str> = rendered_tracks
        .iter()
        .map(|track| track.key.as_str())
        .collect();
    let track_rows = track_row_map(&rendered_tracks);
    let total_items = scene
        .intervals
        .iter()
        .filter(|item| rendered_track_keys.contains(item.track_key.as_str()))
        .filter(|item| overlaps_window(item, scene.time_window))
        .count();
    let item_limit = scene.render_policy.max_items;
    let items_to_render: Vec<&VizInterval> = scene
        .intervals
        .iter()
        .filter(|item| rendered_track_keys.contains(item.track_key.as_str()))
        .filter(|item| overlaps_window(item, scene.time_window))
        .take(item_limit)
        .collect();

    let layout = Layout::new(scene.render_policy.width_px, rendered_tracks.len());
    let span = scene.time_window.span_ns() as f64;
    let scale = layout.plot_width / span;
    let mut suppressed_label_count = 0usize;
    let mut truncated_label_count = 0usize;

    let mut svg = String::new();
    push_svg_header(&mut svg, &layout, scene);
    push_ticks(&mut svg, &layout, scene.time_window);
    push_track_labels(&mut svg, &layout, &rendered_tracks);
    push_legend(&mut svg, &layout, &items_to_render);

    for item in &items_to_render {
        let Some(row) = track_rows.get(item.track_key.as_str()) else {
            continue;
        };
        let clipped_start = item.start_ns.max(scene.time_window.start_ns);
        let clipped_end = item.end_ns.min(scene.time_window.end_ns);
        if clipped_end <= clipped_start {
            continue;
        }
        let x = layout.label_width + ((clipped_start - scene.time_window.start_ns) as f64 * scale);
        let raw_width = (clipped_end - clipped_start) as f64 * scale;
        let y = layout.track_y(*row);
        let color = item_color(item.class.as_deref());
        let Some(track) = rendered_tracks.get(*row) else {
            continue;
        };
        let track_role = track.role;
        let item_role = item.role.unwrap_or(track_role);
        let style = item_style(track_role, item_role, item.class.as_deref());
        let title = tooltip_for(item);
        if raw_width >= scene.render_policy.min_interval_px {
            push_rect(
                &mut svg,
                RectDraw {
                    x,
                    y: y + style.y_offset,
                    width: raw_width.max(1.0),
                    height: style.height,
                    color,
                    opacity: style.opacity,
                    role: item_role,
                    class: item.class.as_deref(),
                    title: title.as_deref(),
                },
            );
        } else {
            push_tick(
                &mut svg,
                x,
                y,
                color,
                style.opacity,
                item_role,
                title.as_deref(),
            );
        }

        if scene.label_policy.mode == VizLabelMode::Hide {
            if item.label.is_some() {
                suppressed_label_count += 1;
            }
            continue;
        }
        let Some(label) = item.label.as_deref() else {
            continue;
        };
        if suppress_role_label(track_role, item_role) {
            suppressed_label_count += 1;
            continue;
        }
        let Some((label, truncated)) = fit_interval_label(label, raw_width, &scene.label_policy)
        else {
            suppressed_label_count += 1;
            continue;
        };
        if truncated {
            truncated_label_count += 1;
        }
        push_interval_label(&mut svg, x + 3.0, y + 13.0, &label);
    }

    if total_items > item_limit {
        push_note(
            &mut svg,
            layout.label_width,
            layout.height - 10.0,
            &format!(
                "rendered {} of {} selected items",
                items_to_render.len(),
                total_items
            ),
        );
    }
    svg.push_str("</svg>\n");

    Ok(SvgRenderResult {
        svg,
        summary: SvgRenderSummary {
            track_count: rendered_tracks.len(),
            rendered_item_count: items_to_render.len(),
            total_item_count: total_items,
            aggregated: total_items > item_limit,
            omitted_track_count: scene.tracks.len().saturating_sub(rendered_tracks.len()),
            suppressed_label_count,
            truncated_label_count,
        },
    })
}

pub fn write_svg_artifact(
    artifact_root: &Path,
    relative_dir: &Path,
    file_name: &str,
    svg: &str,
) -> Result<WrittenSvgArtifact, VisualizationError> {
    validate_relative_path(relative_dir)?;
    validate_svg_file_name(file_name)?;

    let dir = artifact_root.join(relative_dir);
    std::fs::create_dir_all(&dir).map_err(|source| VisualizationError::CreateArtifactDir {
        path: dir.display().to_string(),
        source,
    })?;

    let path = dir.join(file_name);
    let tmp_path = path.with_file_name(format!("{file_name}.tmp.{}", std::process::id()));
    std::fs::write(&tmp_path, svg).map_err(|source| VisualizationError::WriteArtifact {
        path: tmp_path.display().to_string(),
        source,
    })?;
    std::fs::rename(&tmp_path, &path).map_err(|source| {
        let _ = std::fs::remove_file(&tmp_path);
        VisualizationError::PublishArtifact {
            path: path.display().to_string(),
            source,
        }
    })?;

    let relative_path = relative_dir
        .join(file_name)
        .to_string_lossy()
        .replace('\\', "/");
    Ok(WrittenSvgArtifact {
        path,
        relative_path,
        format: "svg",
    })
}

struct Layout {
    width: f64,
    height: f64,
    label_width: f64,
    plot_width: f64,
    top: f64,
    row_height: f64,
}

impl Layout {
    fn new(width_px: u32, tracks: usize) -> Self {
        let width = f64::from(width_px.max(480));
        let label_width = 240.0;
        let top = 34.0;
        let row_height = 22.0;
        let height = top + row_height * tracks.max(1) as f64 + 42.0;
        Self {
            width,
            height,
            label_width,
            plot_width: width - label_width - 20.0,
            top,
            row_height,
        }
    }

    fn track_y(&self, row: usize) -> f64 {
        self.top + row as f64 * self.row_height + 4.0
    }
}

fn validate_relative_path(path: &Path) -> Result<(), VisualizationError> {
    if path.is_absolute() {
        return Err(VisualizationError::UnsafeRelativePath);
    }
    for component in path.components() {
        match component {
            Component::Normal(_) => {}
            Component::CurDir => {}
            _ => return Err(VisualizationError::UnsafeRelativePath),
        }
    }
    Ok(())
}

fn validate_svg_file_name(file_name: &str) -> Result<(), VisualizationError> {
    let path = Path::new(file_name);
    let mut components = path.components();
    let Some(Component::Normal(_)) = components.next() else {
        return Err(VisualizationError::UnsafeSvgFileName);
    };
    if components.next().is_some() || path.extension().and_then(|e| e.to_str()) != Some("svg") {
        return Err(VisualizationError::UnsafeSvgFileName);
    }
    Ok(())
}

fn track_row_map<'a>(tracks: &[&'a VizTrack]) -> BTreeMap<&'a str, usize> {
    let mut out = BTreeMap::new();
    for (idx, track) in tracks.iter().enumerate() {
        out.insert(track.key.as_str(), idx);
    }
    out
}

fn overlaps_window(item: &VizInterval, window: VizTimeWindow) -> bool {
    item.end_ns > window.start_ns && item.start_ns < window.end_ns && item.end_ns > item.start_ns
}

fn push_svg_header(svg: &mut String, layout: &Layout, scene: &VizScene) {
    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{:.0}" height="{:.0}" viewBox="0 0 {:.0} {:.0}" role="img">"#,
        layout.width, layout.height, layout.width, layout.height
    ));
    svg.push('\n');
    if let Some(title) = &scene.title {
        svg.push_str("<title>");
        svg.push_str(&escape_xml(title));
        svg.push_str("</title>\n");
    }
    svg.push_str(
        r#"<style>
text{font-family:"DejaVu Sans","Liberation Sans",Arial,sans-serif;font-size:11px;fill:#17202a}
.axis{stroke:#c9d1d9;stroke-width:1}
.track-line{stroke:#edf1f5;stroke-width:1}
.track-label{font-size:11px;fill:#2b3440}
.track-label-group{font-weight:600;fill:#111827}
.track-label-summary{fill:#4b5563}
.interval-label{font-size:10px;fill:#ffffff;pointer-events:none}
.note{font-size:10px;fill:#6b7280}
.legend-label{font-size:10px;fill:#4b5563}
</style>
"#,
    );
    svg.push_str(&format!(
        r##"<rect x="0" y="0" width="{:.0}" height="{:.0}" fill="#ffffff"/>"##,
        layout.width, layout.height
    ));
    svg.push('\n');
}

fn push_ticks(svg: &mut String, layout: &Layout, window: VizTimeWindow) {
    let y1 = layout.top - 10.0;
    let y2 = layout.height - 38.0;
    svg.push_str(&format!(
        r#"<line class="axis" x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}"/>"#,
        layout.label_width, y1, layout.label_width, y2
    ));
    svg.push('\n');
    svg.push_str(&format!(
        r#"<line class="axis" x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}"/>"#,
        layout.label_width + layout.plot_width,
        y1,
        layout.label_width + layout.plot_width,
        y2
    ));
    svg.push('\n');
    push_note(
        svg,
        layout.label_width,
        16.0,
        &format_time_tick(window.start_ns),
    );
    push_note_anchored(
        svg,
        layout.label_width + layout.plot_width,
        16.0,
        &format_time_tick(window.end_ns),
        "end",
    );
}

fn push_track_labels(svg: &mut String, layout: &Layout, tracks: &[&VizTrack]) {
    for (idx, track) in tracks.iter().enumerate() {
        let y = layout.track_y(idx);
        let label_x = 8.0 + track.depth as f64 * 14.0;
        let role = track.role.to_string();
        svg.push_str(&format!(
            r#"<line class="track-line track-line-{role}" data-track-role="{role}" x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}"/>"#,
            layout.label_width,
            y + 7.0,
            layout.width - 10.0,
            y + 7.0
        ));
        svg.push('\n');
        svg.push_str(&format!(
            r#"<text class="track-label track-label-{role}" data-track-role="{role}" x="{:.1}" y="{:.1}">{}</text>"#,
            label_x,
            y + 12.0,
            escape_xml(&track.label)
        ));
        svg.push('\n');
    }
}

fn push_legend(svg: &mut String, layout: &Layout, items: &[&VizInterval]) {
    let mut classes = BTreeSet::new();
    for item in items {
        if let Some(class) = item.class.as_deref() {
            classes.insert(canonical_legend_class(class));
        }
    }
    if classes.is_empty() {
        return;
    }

    let mut x = layout.label_width;
    let y = layout.height - 28.0;
    for class in ordered_legend_classes(&classes) {
        let color = item_color(Some(class));
        let opacity = legend_opacity(class);
        svg.push_str(&format!(
            r#"<rect x="{x:.1}" y="{:.1}" width="10" height="10" rx="1.5" fill="{color}" fill-opacity="{opacity:.2}"/>"#,
            y - 8.0
        ));
        svg.push('\n');
        svg.push_str(&format!(
            r#"<text class="legend-label" x="{:.1}" y="{y:.1}">{}</text>"#,
            x + 14.0,
            escape_xml(legend_label(class))
        ));
        svg.push('\n');
        x += legend_width(class);
        if x > layout.label_width + layout.plot_width - 72.0 {
            break;
        }
    }
}

fn ordered_legend_classes<'a>(classes: &BTreeSet<&'a str>) -> Vec<&'a str> {
    let preferred = ["kernel", "memcpy", "memset", "graph", "gap", "api", "nvtx"];
    let mut out = Vec::new();
    for class in preferred {
        if classes.contains(class) {
            out.push(class);
        }
    }
    for class in classes {
        if !preferred.contains(class) {
            out.push(class);
        }
    }
    out
}

fn legend_label(class: &str) -> &str {
    match class {
        "kernel" => "kernel",
        "memcpy" => "memcpy",
        "memset" => "memset",
        "graph" => "graph",
        "gap" => "idle gap",
        "api" | "runtime" => "CUDA API",
        "nvtx" => "NVTX",
        _ => class,
    }
}

fn canonical_legend_class(class: &str) -> &str {
    match class {
        "runtime" => "api",
        _ => class,
    }
}

fn legend_width(class: &str) -> f64 {
    28.0 + estimate_text_width(legend_label(class), 10.0)
}

fn legend_opacity(class: &str) -> f64 {
    match class {
        "gap" => 0.30,
        _ => item_opacity(Some(class)),
    }
}

struct RectDraw<'a> {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    color: &'a str,
    opacity: f64,
    role: VizRole,
    class: Option<&'a str>,
    title: Option<&'a str>,
}

fn push_rect(svg: &mut String, rect: RectDraw<'_>) {
    let x = rect.x;
    let y = rect.y;
    let width = rect.width;
    let height = rect.height;
    let color = rect.color;
    let opacity = rect.opacity;
    let role = rect.role.to_string();
    let class_attr = rect.class.unwrap_or("unknown");
    svg.push_str(&format!(
        r#"<rect class="interval interval-{class_attr} interval-role-{role}" data-role="{role}" data-class="{class_attr}" x="{x:.1}" y="{y:.1}" width="{width:.1}" height="{height:.1}" rx="1.5" fill="{color}" fill-opacity="{opacity:.2}">"#
    ));
    if let Some(title) = rect.title {
        svg.push_str("<title>");
        svg.push_str(&escape_xml(title));
        svg.push_str("</title>");
    }
    svg.push_str("</rect>\n");
}

fn push_tick(
    svg: &mut String,
    x: f64,
    y: f64,
    color: &str,
    opacity: f64,
    role: VizRole,
    title: Option<&str>,
) {
    let role = role.to_string();
    svg.push_str(&format!(
        r#"<line class="interval-tick interval-role-{role}" data-role="{role}" x1="{x:.1}" y1="{y:.1}" x2="{x:.1}" y2="{:.1}" stroke="{color}" stroke-opacity="{opacity:.2}" stroke-width="1.2">"#,
        y + 14.0
    ));
    if let Some(title) = title {
        svg.push_str("<title>");
        svg.push_str(&escape_xml(title));
        svg.push_str("</title>");
    }
    svg.push_str("</line>\n");
}

fn push_interval_label(svg: &mut String, x: f64, y: f64, label: &str) {
    svg.push_str(&format!(
        r#"<text class="interval-label" x="{x:.1}" y="{y:.1}">{}</text>"#,
        escape_xml(label)
    ));
    svg.push('\n');
}

fn push_note(svg: &mut String, x: f64, y: f64, text: &str) {
    push_note_anchored(svg, x, y, text, "start");
}

fn push_note_anchored(svg: &mut String, x: f64, y: f64, text: &str, anchor: &str) {
    svg.push_str(&format!(
        r#"<text class="note" x="{x:.1}" y="{y:.1}" text-anchor="{anchor}">{}</text>"#,
        escape_xml(text)
    ));
    svg.push('\n');
}

fn tooltip_for(item: &VizInterval) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(row_id) = &item.row_id {
        parts.push(row_id.clone());
    }
    if let Some(label) = &item.label {
        parts.push(label.clone());
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" | "))
    }
}

struct ItemStyle {
    height: f64,
    y_offset: f64,
    opacity: f64,
}

fn item_style(track_role: VizRole, item_role: VizRole, class: Option<&str>) -> ItemStyle {
    match (track_role, item_role) {
        (_, VizRole::Overlay) => ItemStyle {
            height: 14.0,
            y_offset: 0.0,
            opacity: 0.30,
        },
        (VizRole::Summary, _) => ItemStyle {
            height: 8.0,
            y_offset: 3.0,
            opacity: item_opacity(class).min(0.58),
        },
        _ => ItemStyle {
            height: 14.0,
            y_offset: 0.0,
            opacity: item_opacity(class),
        },
    }
}

fn suppress_role_label(track_role: VizRole, item_role: VizRole) -> bool {
    matches!(track_role, VizRole::Summary) || matches!(item_role, VizRole::Overlay)
}

fn fit_interval_label(
    label: &str,
    raw_width: f64,
    policy: &VizLabelPolicy,
) -> Option<(String, bool)> {
    if raw_width < policy.min_label_px || policy.max_chars == 0 {
        return None;
    }
    let width_chars = ((raw_width - 8.0) / 6.4).floor();
    if width_chars < 4.0 {
        return None;
    }
    let max_chars = policy.max_chars.min(width_chars as usize);
    Some(truncate_label(label, max_chars))
}

fn truncate_label(label: &str, max_chars: usize) -> (String, bool) {
    let total_chars = label.chars().count();
    if total_chars <= max_chars {
        return (label.to_string(), false);
    }
    if max_chars <= 3 {
        return (String::new(), true);
    }
    let mut out = String::new();
    for ch in label.chars().take(max_chars - 3) {
        out.push(ch);
    }
    out.push_str("...");
    (out, true)
}

fn estimate_text_width(label: &str, font_px: f64) -> f64 {
    label.chars().count() as f64 * font_px * 0.58
}

fn format_time_tick(ns: i64) -> String {
    let abs = ns.saturating_abs();
    if abs < 1_000 {
        format!("{ns} ns")
    } else if abs < 1_000_000 {
        format!("{} us", format_decimal(ns as f64 / 1_000.0, 3))
    } else if abs < 1_000_000_000 {
        format!("{} ms", format_decimal(ns as f64 / 1_000_000.0, 3))
    } else {
        format!("{} s", format_decimal(ns as f64 / 1_000_000_000.0, 3))
    }
}

fn format_decimal(value: f64, decimals: usize) -> String {
    let mut out = format!("{value:.decimals$}");
    if out.contains('.') {
        while out.ends_with('0') {
            out.pop();
        }
        if out.ends_with('.') {
            out.pop();
        }
    }
    out
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn item_color(class: Option<&str>) -> &'static str {
    match class {
        Some("kernel") => "#2563eb",
        Some("memcpy") => "#0891b2",
        Some("memset") => "#0d9488",
        Some("graph") => "#7c3aed",
        Some("gap") => "#ef4444",
        Some("api" | "runtime") => "#64748b",
        Some("nvtx") => "#16a34a",
        _ => "#334155",
    }
}

fn item_opacity(class: Option<&str>) -> f64 {
    match class {
        Some("gap") => 0.72,
        Some("nvtx") => 0.88,
        _ => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene() -> VizScene {
        VizScene {
            title: Some("test".to_string()),
            time_window: VizTimeWindow {
                start_ns: 100,
                end_ns: 200,
            },
            tracks: vec![VizTrack {
                key: "gpu|0".to_string(),
                label: "GPU 0".to_string(),
                kind: "gpu".to_string(),
                role: VizRole::Detail,
                depth: 0,
                axes: vec![VizAxis {
                    name: "device".to_string(),
                    value: "0".to_string(),
                }],
            }],
            intervals: vec![VizInterval {
                track_key: "gpu|0".to_string(),
                start_ns: 110,
                end_ns: 150,
                label: Some("kernel<&>".to_string()),
                row_id: Some("kernel:1".to_string()),
                class: Some("kernel".to_string()),
                role: None,
            }],
            render_policy: VizRenderPolicy::default(),
            label_policy: VizLabelPolicy::default(),
        }
    }

    #[test]
    fn render_svg_escapes_labels_and_counts_items() -> anyhow::Result<()> {
        let rendered = render_svg(&scene())?;
        assert_eq!(rendered.summary.track_count, 1);
        assert_eq!(rendered.summary.rendered_item_count, 1);
        assert_eq!(rendered.summary.total_item_count, 1);
        assert!(rendered.svg.contains("kernel&lt;&amp;&gt;"));
        assert!(rendered.svg.contains("<svg"));
        Ok(())
    }

    #[test]
    fn render_svg_reports_omitted_tracks_and_item_limit() -> anyhow::Result<()> {
        let mut scene = scene();
        scene.tracks.push(VizTrack {
            key: "gpu|1".to_string(),
            label: "GPU 1".to_string(),
            kind: "gpu".to_string(),
            role: VizRole::Detail,
            depth: 0,
            axes: vec![],
        });
        scene.render_policy.max_tracks = 1;
        scene.render_policy.max_items = 0;
        let rendered = render_svg(&scene)?;
        assert_eq!(rendered.summary.omitted_track_count, 1);
        assert_eq!(rendered.summary.rendered_item_count, 0);
        assert_eq!(rendered.summary.total_item_count, 1);
        assert!(rendered.summary.aggregated);
        Ok(())
    }

    #[test]
    fn render_svg_exposes_roles_and_suppresses_summary_overlay_labels() -> anyhow::Result<()> {
        let mut scene = scene();
        scene.tracks = vec![
            VizTrack {
                key: "gpu-device|dev:0".to_string(),
                label: "GPU 0".to_string(),
                kind: "gpu-device".to_string(),
                role: VizRole::Group,
                depth: 0,
                axes: vec![],
            },
            VizTrack {
                key: "gpu-summary|dev:0".to_string(),
                label: "busy summary".to_string(),
                kind: "gpu-summary".to_string(),
                role: VizRole::Summary,
                depth: 1,
                axes: vec![],
            },
        ];
        scene.intervals = vec![
            VizInterval {
                track_key: "gpu-summary|dev:0".to_string(),
                start_ns: 110,
                end_ns: 130,
                label: Some("summary kernel".to_string()),
                row_id: Some("kernel:1".to_string()),
                class: Some("kernel".to_string()),
                role: None,
            },
            VizInterval {
                track_key: "gpu-summary|dev:0".to_string(),
                start_ns: 130,
                end_ns: 150,
                label: Some("idle".to_string()),
                row_id: None,
                class: Some("gap".to_string()),
                role: Some(VizRole::Overlay),
            },
        ];

        let rendered = render_svg(&scene)?;
        assert!(rendered.svg.contains("data-track-role=\"group\""));
        assert!(rendered.svg.contains("data-track-role=\"summary\""));
        assert!(rendered.svg.contains("data-role=\"overlay\""));
        assert!(!rendered.svg.contains(">summary kernel</text>"));
        assert!(!rendered.svg.contains(">idle</text>"));
        assert_eq!(rendered.summary.suppressed_label_count, 2);
        Ok(())
    }

    #[test]
    fn write_svg_artifact_returns_portable_relative_path() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let written = write_svg_artifact(
            dir.path(),
            Path::new("figures/nsys"),
            "timeline.svg",
            "<svg/>",
        )?;
        assert_eq!(written.relative_path, "figures/nsys/timeline.svg");
        assert_eq!(written.format, "svg");
        assert!(written.path.exists());
        Ok(())
    }

    #[test]
    fn write_svg_artifact_rejects_parent_relative_dir() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let err = match write_svg_artifact(dir.path(), Path::new("../x"), "timeline.svg", "<svg/>")
        {
            Ok(_) => anyhow::bail!("unsafe path should fail"),
            Err(err) => err,
        };
        assert!(matches!(err, VisualizationError::UnsafeRelativePath));
        Ok(())
    }
}
