use serde::Serialize;
use std::collections::BTreeSet;

use crate::{
    VizHighlight, VizInterval, VizLabelPolicy, VizRenderPolicy, VizRole, VizScene,
    VizSceneMetadata, VizTimeWindow, VizTrack,
};

use super::layout::Layout;
use super::style::{item_color, item_opacity};
use super::text::{escape_xml, estimate_text_width, format_time_tick, truncate_label};

pub(super) fn push_svg_header(svg: &mut String, layout: &Layout, scene: &VizScene) {
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
    push_metadata(svg, scene);
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

#[derive(Serialize)]
struct SvgMetadata<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    scene: Option<&'a VizSceneMetadata>,
    time_window: VizTimeWindow,
    tracks: Vec<SvgTrackMetadata<'a>>,
    render_policy: &'a VizRenderPolicy,
    label_policy: &'a VizLabelPolicy,
}

#[derive(Serialize)]
struct SvgTrackMetadata<'a> {
    track_key: &'a str,
    role: String,
}

fn push_metadata(svg: &mut String, scene: &VizScene) {
    let payload = SvgMetadata {
        scene: scene.metadata.as_ref(),
        time_window: scene.time_window,
        tracks: scene
            .tracks
            .iter()
            .map(|track| SvgTrackMetadata {
                track_key: track.key.as_str(),
                role: track.role.to_string(),
            })
            .collect(),
        render_policy: &scene.render_policy,
        label_policy: &scene.label_policy,
    };
    let Ok(json) = serde_json::to_string(&payload) else {
        return;
    };
    svg.push_str(r#"<metadata id="veloq-viz-metadata" type="application/json">"#);
    svg.push_str(&escape_xml(&json));
    svg.push_str("</metadata>\n");
}

pub(super) fn push_ticks(svg: &mut String, layout: &Layout, window: VizTimeWindow) {
    let y1 = layout.top - 10.0;
    let y2 = layout.plot_bottom();
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

pub(super) fn push_track_labels(svg: &mut String, layout: &Layout, tracks: &[&VizTrack]) {
    for (idx, track) in tracks.iter().enumerate() {
        let y = layout.track_y(idx);
        let label_x = 8.0 + track.depth as f64 * 14.0;
        let role = track.role.to_string();
        let key = escape_xml(&track.key);
        svg.push_str(&format!(
            r#"<line class="track-line track-line-{role}" data-track-key="{key}" data-track-role="{role}" x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}"/>"#,
            layout.label_width,
            y + 7.0,
            layout.width - 10.0,
            y + 7.0
        ));
        svg.push('\n');
        svg.push_str(&format!(
            r#"<text class="track-label track-label-{role}" data-track-key="{key}" data-track-role="{role}" x="{:.1}" y="{:.1}">{}</text>"#,
            label_x,
            y + 12.0,
            escape_xml(&track.label)
        ));
        svg.push('\n');
    }
}

pub(super) fn push_legend(svg: &mut String, layout: &Layout, items: &[&VizInterval]) {
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
    let y = layout.legend_y();
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

pub(super) fn push_highlight_legend(
    svg: &mut String,
    layout: &Layout,
    highlights: &[VizHighlight],
) {
    for (idx, highlight) in highlights.iter().take(layout.highlight_rows).enumerate() {
        let y = layout.highlight_legend_y(idx);
        svg.push_str("<g class=\"highlight-legend-item\">");
        svg.push_str("<title>");
        svg.push_str(&escape_xml(&highlight.full_label));
        svg.push_str("</title>");
        svg.push_str(&format!(
            r##"<rect x="{:.1}" y="{:.1}" width="10" height="10" rx="1.5" fill="{}" stroke="#111827" stroke-width="0.6"/>"##,
            layout.label_width,
            y - 8.0,
            escape_xml(&highlight.color)
        ));
        svg.push('\n');
        svg.push_str(&format!(
            r#"<text class="legend-label" x="{:.1}" y="{y:.1}">{}</text>"#,
            layout.label_width + 14.0,
            escape_xml(&highlight_legend_label(highlight, layout.plot_width))
        ));
        svg.push_str("</g>\n");
    }
}

fn highlight_legend_label(highlight: &VizHighlight, plot_width: f64) -> String {
    let prefix = highlight.rank.map_or_else(
        || highlight.label.clone(),
        |rank| format!("#{rank} {}", highlight.label),
    );
    let max_chars = ((plot_width - 24.0) / 6.0).floor().max(24.0) as usize;
    truncate_label(&prefix, max_chars).0
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

pub(super) struct RectDraw<'a> {
    pub(super) x: f64,
    pub(super) y: f64,
    pub(super) width: f64,
    pub(super) height: f64,
    pub(super) color: &'a str,
    pub(super) opacity: f64,
    pub(super) role: VizRole,
    pub(super) class: Option<&'a str>,
    pub(super) highlight_key: Option<&'a str>,
    pub(super) title: Option<&'a str>,
}

pub(super) struct TickDraw<'a> {
    pub(super) x: f64,
    pub(super) y: f64,
    pub(super) color: &'a str,
    pub(super) opacity: f64,
    pub(super) role: VizRole,
    pub(super) highlight_key: Option<&'a str>,
    pub(super) title: Option<&'a str>,
}

pub(super) fn push_rect(svg: &mut String, rect: RectDraw<'_>) {
    let x = rect.x;
    let y = rect.y;
    let width = rect.width;
    let height = rect.height;
    let color = rect.color;
    let opacity = rect.opacity;
    let role = rect.role.to_string();
    let class_attr = rect.class.unwrap_or("unknown");
    let highlight_class = if rect.highlight_key.is_some() {
        " interval-highlighted"
    } else {
        ""
    };
    let highlight_attr = rect.highlight_key.map_or_else(String::new, |key| {
        format!(
            r##" data-highlight-key="{}" stroke="#111827" stroke-width="0.8""##,
            escape_xml(key)
        )
    });
    svg.push_str(&format!(
        r#"<rect class="interval interval-{class_attr} interval-role-{role}{highlight_class}" data-role="{role}" data-class="{class_attr}"{highlight_attr} x="{x:.1}" y="{y:.1}" width="{width:.1}" height="{height:.1}" rx="1.5" fill="{color}" fill-opacity="{opacity:.2}">"#
    ));
    if let Some(title) = rect.title {
        svg.push_str("<title>");
        svg.push_str(&escape_xml(title));
        svg.push_str("</title>");
    }
    svg.push_str("</rect>\n");
}

pub(super) fn push_tick(svg: &mut String, tick: TickDraw<'_>) {
    let x = tick.x;
    let y = tick.y;
    let color = tick.color;
    let opacity = tick.opacity;
    let role = tick.role.to_string();
    let highlight_class = if tick.highlight_key.is_some() {
        " interval-highlighted"
    } else {
        ""
    };
    let highlight_attr = tick.highlight_key.map_or_else(String::new, |key| {
        format!(r#" data-highlight-key="{}""#, escape_xml(key))
    });
    let stroke_width = if tick.highlight_key.is_some() {
        2.0
    } else {
        1.2
    };
    svg.push_str(&format!(
        r#"<line class="interval-tick interval-role-{role}{highlight_class}" data-role="{role}"{highlight_attr} x1="{x:.1}" y1="{y:.1}" x2="{x:.1}" y2="{:.1}" stroke="{color}" stroke-opacity="{opacity:.2}" stroke-width="{stroke_width:.1}">"#,
        y + 14.0
    ));
    if let Some(title) = tick.title {
        svg.push_str("<title>");
        svg.push_str(&escape_xml(title));
        svg.push_str("</title>");
    }
    svg.push_str("</line>\n");
}

pub(super) fn push_interval_label(svg: &mut String, x: f64, y: f64, label: &str) {
    svg.push_str(&format!(
        r#"<text class="interval-label" x="{x:.1}" y="{y:.1}">{}</text>"#,
        escape_xml(label)
    ));
    svg.push('\n');
}

pub(super) fn push_note(svg: &mut String, x: f64, y: f64, text: &str) {
    push_note_anchored(svg, x, y, text, "start");
}

fn push_note_anchored(svg: &mut String, x: f64, y: f64, text: &str, anchor: &str) {
    svg.push_str(&format!(
        r#"<text class="note" x="{x:.1}" y="{y:.1}" text-anchor="{anchor}">{}</text>"#,
        escape_xml(text)
    ));
    svg.push('\n');
}
