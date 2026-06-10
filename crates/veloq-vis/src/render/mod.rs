mod layout;
mod painter;
mod style;
#[cfg(test)]
mod tests;
mod text;

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    SvgRenderResult, SvgRenderSummary, VisualizationError, VizHighlight, VizInterval, VizLabelMode,
    VizScene, VizTrack,
};
use layout::{Layout, assign_lanes, overlaps_window};
use painter::{
    RectDraw, TickDraw, push_highlight_legend, push_interval_label, push_legend, push_note,
    push_rect, push_svg_header, push_tick, push_ticks, push_track_labels,
};
use style::{item_color, item_style, suppress_role_label};
use text::{fit_interval_label, tooltip_for};

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

    let lane_layout = assign_lanes(&rendered_tracks, &items_to_render, scene);
    let layout = Layout::new(
        scene.render_policy.width_px,
        &lane_layout.track_lane_counts,
        scene.highlights.len(),
    );
    let span = scene.time_window.span_ns() as f64;
    let scale = layout.plot_width / span;
    let highlights = highlight_map(&scene.highlights);
    let mut suppressed_label_count = 0usize;
    let mut truncated_label_count = 0usize;

    let mut svg = String::new();
    push_svg_header(&mut svg, &layout, scene);
    push_ticks(&mut svg, &layout, scene.time_window);
    push_track_labels(&mut svg, &layout, &rendered_tracks);
    push_legend(&mut svg, &layout, &items_to_render);
    push_highlight_legend(&mut svg, &layout, &scene.highlights);

    for (item_idx, item) in items_to_render.iter().enumerate() {
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
        let lane = lane_layout.lanes.get(item_idx).copied().unwrap_or_default();
        let y = layout.track_lane_y(*row, lane);
        let highlight = item
            .highlight_key
            .as_deref()
            .and_then(|key| highlights.get(key));
        let color =
            highlight.map_or_else(|| item_color(item.class.as_deref()), |h| h.color.as_str());
        let Some(track) = rendered_tracks.get(*row) else {
            continue;
        };
        let track_role = track.role;
        let item_role = item.role.unwrap_or(track_role);
        let style = item_style(track_role, item_role, item.class.as_deref());
        let title = tooltip_for(item, highlight.copied());
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
                    highlight_key: item.highlight_key.as_deref(),
                    title: title.as_deref(),
                },
            );
        } else {
            push_tick(
                &mut svg,
                TickDraw {
                    x,
                    y,
                    color,
                    opacity: style.opacity,
                    role: item_role,
                    highlight_key: item.highlight_key.as_deref(),
                    title: title.as_deref(),
                },
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

fn track_row_map<'a>(tracks: &[&'a VizTrack]) -> BTreeMap<&'a str, usize> {
    let mut out = BTreeMap::new();
    for (idx, track) in tracks.iter().enumerate() {
        out.insert(track.key.as_str(), idx);
    }
    out
}

fn highlight_map(highlights: &[VizHighlight]) -> BTreeMap<&str, &VizHighlight> {
    highlights
        .iter()
        .map(|highlight| (highlight.key.as_str(), highlight))
        .collect()
}
