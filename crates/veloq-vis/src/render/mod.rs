mod layout;
mod painter;
mod style;
#[cfg(test)]
mod tests;
mod text;

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    SvgRenderResult, SvgRenderSummary, VisualizationError, VizAggregation, VizHighlight,
    VizInterval, VizLabelMode, VizRole, VizScene, VizTimeWindow, VizTrack,
};
use layout::{Layout, assign_lanes, overlaps_window};
use painter::{
    DensityBinDraw, IntervalLabelDraw, RectDraw, TickDraw, push_density_bin, push_highlight_legend,
    push_interval_label, push_legend, push_note, push_rect, push_svg_header, push_tick, push_ticks,
    push_track_labels,
};
use style::{
    INTERVAL_LABEL_BASELINE_OFFSET_PX, INTERVAL_LABEL_PADDING_X, density_opacity, item_color,
    item_style, suppress_role_label,
};
use text::{DensityTooltipBreakdown, density_tooltip_for, fit_interval_label, tooltip_for};

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
    let candidate_items: Vec<&VizInterval> = scene
        .intervals
        .iter()
        .filter(|item| rendered_track_keys.contains(item.track_key.as_str()))
        .filter(|item| overlaps_window(item, scene.time_window))
        .collect();
    let total_items = candidate_items.len();
    let highlights = highlight_map(&scene.highlights);

    let selection = select_render_items(&candidate_items, &track_rows, &rendered_tracks, scene);
    let lane_layout = assign_lanes(&rendered_tracks, &selection.explicit_items, scene);
    let layout = Layout::new(
        scene.render_policy.width_px,
        &lane_layout.track_lane_counts,
        scene.highlights.len(),
    );
    let span = scene.time_window.span_ns() as f64;
    let scale = layout.plot_width / span;
    let mut suppressed_label_count = 0usize;
    let mut truncated_label_count = 0usize;

    let mut svg = String::new();
    push_svg_header(&mut svg, &layout, scene);
    push_ticks(&mut svg, &layout, scene.time_window);
    push_track_labels(&mut svg, &layout, &rendered_tracks);
    push_legend(&mut svg, &layout, &candidate_items);
    push_highlight_legend(&mut svg, &layout, &scene.highlights);

    for bin in &selection.density_bins {
        let Some(row) = track_rows.get(bin.track_key.as_str()) else {
            continue;
        };
        let clipped_start = bin.start_ns.max(scene.time_window.start_ns);
        let clipped_end = bin.end_ns.min(scene.time_window.end_ns);
        if clipped_end <= clipped_start {
            continue;
        }
        let x = layout.label_width + ((clipped_start - scene.time_window.start_ns) as f64 * scale);
        let raw_width = (clipped_end - clipped_start) as f64 * scale;
        let Some(track) = rendered_tracks.get(*row) else {
            continue;
        };
        let style = item_style(track.role, bin.role, Some(&bin.dominant_class));
        let hit_y = layout.track_y(*row) + style.y_offset;
        let heat_height = density_heat_height(track.role, bin.role);
        let y = hit_y + (style.height - heat_height) / 2.0;
        let breakdown = bin
            .breakdown
            .iter()
            .map(|class| DensityTooltipBreakdown {
                class: class.class.as_str(),
                count: class.count,
                total_duration_ns: class.total_duration_ns,
            })
            .collect::<Vec<_>>();
        let title = density_tooltip_for(
            &bin.dominant_class,
            bin.count,
            bin.total_duration_ns,
            bin.max_duration_ns,
            bin.start_ns,
            bin.end_ns,
            &breakdown,
        );
        let color = bin
            .dominant_highlight_key
            .as_deref()
            .and_then(|key| {
                highlights
                    .get(key)
                    .map(|highlight| highlight.color.as_str())
            })
            .unwrap_or_else(|| item_color(Some(&bin.dominant_class)));
        push_density_bin(
            &mut svg,
            DensityBinDraw {
                x,
                y,
                hit_y,
                width: raw_width.max(1.0),
                height: heat_height,
                hit_height: style.height,
                color,
                opacity: density_opacity(bin.total_duration_ns, bin.end_ns - bin.start_ns),
                class: &bin.dominant_class,
                highlight_key: bin.dominant_highlight_key.as_deref(),
                dashed: !matches!(track.role, VizRole::Summary),
                count: bin.count,
                total_duration_ns: bin.total_duration_ns,
                title: &title,
            },
        );
    }

    for (item_idx, item) in selection.explicit_items.iter().enumerate() {
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
        push_interval_label(
            &mut svg,
            IntervalLabelDraw {
                id: item_idx,
                x: x + INTERVAL_LABEL_PADDING_X,
                y: y + INTERVAL_LABEL_BASELINE_OFFSET_PX,
                clip_x: x,
                clip_y: y + style.y_offset,
                clip_width: raw_width.max(1.0),
                clip_height: style.height,
                label: &label,
            },
        );
    }

    if selection.density_item_count > 0 || selection.omitted_explicit_item_count > 0 {
        push_note(
            &mut svg,
            layout.label_width,
            layout.height - 10.0,
            &format!(
                "rendered {} visual items from {} selected items ({} density bins)",
                selection.rendered_visual_item_count(),
                total_items,
                selection.density_bins.len()
            ),
        );
    }
    svg.push_str("</svg>\n");

    Ok(SvgRenderResult {
        svg,
        summary: SvgRenderSummary {
            track_count: rendered_tracks.len(),
            rendered_item_count: selection.rendered_visual_item_count(),
            total_item_count: total_items,
            density_item_count: selection.density_item_count,
            density_bin_count: selection.density_bins.len(),
            density_duration_ns: selection.density_duration_ns,
            omitted_explicit_item_count: selection.omitted_explicit_item_count,
            aggregated: selection.density_item_count > 0
                || selection.omitted_explicit_item_count > 0,
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

struct RenderSelection<'a> {
    explicit_items: Vec<&'a VizInterval>,
    density_bins: Vec<DensityBin>,
    density_item_count: usize,
    density_duration_ns: i64,
    omitted_explicit_item_count: usize,
}

impl RenderSelection<'_> {
    fn rendered_visual_item_count(&self) -> usize {
        self.explicit_items
            .len()
            .saturating_add(self.density_bins.len())
    }
}

#[derive(Debug, Clone)]
struct DensityBin {
    track_key: String,
    dominant_class: String,
    dominant_highlight_key: Option<String>,
    breakdown: Vec<DensityClassBreakdown>,
    role: VizRole,
    start_ns: i64,
    end_ns: i64,
    count: usize,
    total_duration_ns: i64,
    max_duration_ns: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct DensityBinKey {
    track_key: String,
    bin_index: u64,
}

#[derive(Debug, Clone)]
struct DensityAccumulator {
    role: VizRole,
    start_ns: i64,
    end_ns: i64,
    count: usize,
    total_duration_ns: i64,
    max_duration_ns: i64,
    breakdown: BTreeMap<String, DensityClassAccumulator>,
    visual_breakdown: BTreeMap<DensityVisualKey, DensityClassAccumulator>,
}

#[derive(Debug, Clone)]
struct DensityClassAccumulator {
    count: usize,
    total_duration_ns: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct DensityVisualKey {
    class: String,
    highlight_key: Option<String>,
}

#[derive(Debug, Clone)]
struct DensityClassBreakdown {
    class: String,
    count: usize,
    total_duration_ns: i64,
}

struct RenderCandidate<'a> {
    item: &'a VizInterval,
    clipped_start: i64,
    clipped_end: i64,
    raw_width: f64,
    item_role: VizRole,
}

#[derive(Clone, Copy)]
struct DensityThreshold {
    interval_px: f64,
    bin_px: f64,
}

fn select_render_items<'a>(
    candidates: &[&'a VizInterval],
    track_rows: &BTreeMap<&str, usize>,
    rendered_tracks: &[&VizTrack],
    scene: &VizScene,
) -> RenderSelection<'a> {
    let density_bin_px = effective_density_bin_px(scene);
    let layout_width = f64::from(scene.render_policy.width_px.max(480));
    let plot_width = layout_width - 240.0 - 20.0;
    let scale = plot_width / scene.time_window.span_ns() as f64;
    let prepared = prepare_render_candidates(candidates, track_rows, rendered_tracks, scene, scale);
    if scene.render_policy.max_items == 0 {
        return RenderSelection {
            explicit_items: Vec::new(),
            density_bins: Vec::new(),
            density_item_count: 0,
            density_duration_ns: 0,
            omitted_explicit_item_count: prepared.len(),
        };
    }
    let density_threshold = choose_density_threshold(&prepared, density_bin_px, scene, scale);
    let mut explicit_items = Vec::new();
    let mut density = BTreeMap::<DensityBinKey, DensityAccumulator>::new();
    let mut density_item_count = 0usize;
    let mut density_duration_ns = 0i64;
    let mut omitted_explicit_item_count = 0usize;

    for candidate in &prepared {
        if should_omit_annotation_tick(candidate, scene.render_policy.min_interval_px) {
            omitted_explicit_item_count = omitted_explicit_item_count.saturating_add(1);
            continue;
        }
        if should_density_bin(
            candidate,
            density_threshold,
            scene.render_policy.min_interval_px,
        ) {
            let Some(threshold) = density_threshold else {
                explicit_items.push(candidate.item);
                continue;
            };
            let bin_index = density_bin_index(
                candidate.clipped_start,
                candidate.clipped_end,
                scene.time_window,
                scale,
                threshold.bin_px,
            );
            let class = candidate
                .item
                .class
                .as_deref()
                .unwrap_or("unknown")
                .to_string();
            let key = DensityBinKey {
                track_key: candidate.item.track_key.clone(),
                bin_index,
            };
            let (bin_start_ns, bin_end_ns) =
                density_bin_bounds(bin_index, scene.time_window, scale, threshold.bin_px);
            let duration_ns = candidate.clipped_end - candidate.clipped_start;
            let entry = density.entry(key).or_insert(DensityAccumulator {
                start_ns: bin_start_ns,
                end_ns: bin_end_ns,
                role: candidate.item_role,
                count: 0,
                total_duration_ns: 0,
                max_duration_ns: 0,
                breakdown: BTreeMap::new(),
                visual_breakdown: BTreeMap::new(),
            });
            entry.count = entry.count.saturating_add(1);
            entry.total_duration_ns = entry.total_duration_ns.saturating_add(duration_ns);
            entry.max_duration_ns = entry.max_duration_ns.max(duration_ns);
            let class_entry =
                entry
                    .breakdown
                    .entry(class.clone())
                    .or_insert(DensityClassAccumulator {
                        count: 0,
                        total_duration_ns: 0,
                    });
            class_entry.count = class_entry.count.saturating_add(1);
            class_entry.total_duration_ns =
                class_entry.total_duration_ns.saturating_add(duration_ns);
            let visual_entry = entry
                .visual_breakdown
                .entry(DensityVisualKey {
                    class,
                    highlight_key: candidate.item.highlight_key.clone(),
                })
                .or_insert(DensityClassAccumulator {
                    count: 0,
                    total_duration_ns: 0,
                });
            visual_entry.count = visual_entry.count.saturating_add(1);
            visual_entry.total_duration_ns =
                visual_entry.total_duration_ns.saturating_add(duration_ns);
            density_item_count = density_item_count.saturating_add(1);
            density_duration_ns = density_duration_ns.saturating_add(duration_ns);
        } else {
            explicit_items.push(candidate.item);
        }
    }

    let item_limit_omitted_count = if density_threshold.is_none() {
        explicit_items
            .len()
            .saturating_sub(scene.render_policy.max_items)
    } else {
        0
    };
    omitted_explicit_item_count =
        omitted_explicit_item_count.saturating_add(item_limit_omitted_count);
    let explicit_items = if density_threshold.is_none() {
        limit_explicit_items(explicit_items, scene.render_policy.max_items)
    } else {
        explicit_items
    };
    let density_bins = density
        .into_iter()
        .map(|(key, value)| {
            let dominant_visual = dominant_density_visual(&value.visual_breakdown);
            let dominant_class = dominant_visual
                .map(|visual| visual.class.clone())
                .unwrap_or_else(|| dominant_density_class(&value.breakdown));
            let dominant_highlight_key =
                dominant_visual.and_then(|visual| visual.highlight_key.clone());
            DensityBin {
                track_key: key.track_key,
                dominant_class,
                dominant_highlight_key,
                breakdown: density_class_breakdown(value.breakdown),
                role: value.role,
                start_ns: value.start_ns,
                end_ns: value.end_ns,
                count: value.count,
                total_duration_ns: value.total_duration_ns,
                max_duration_ns: value.max_duration_ns,
            }
        })
        .collect();

    RenderSelection {
        explicit_items,
        density_bins,
        density_item_count,
        density_duration_ns,
        omitted_explicit_item_count,
    }
}

fn prepare_render_candidates<'a>(
    candidates: &[&'a VizInterval],
    track_rows: &BTreeMap<&str, usize>,
    rendered_tracks: &[&VizTrack],
    scene: &VizScene,
    scale: f64,
) -> Vec<RenderCandidate<'a>> {
    let mut out = Vec::new();
    for item in candidates {
        let Some(row) = track_rows.get(item.track_key.as_str()) else {
            continue;
        };
        let Some(track) = rendered_tracks.get(*row) else {
            continue;
        };
        let clipped_start = item.start_ns.max(scene.time_window.start_ns);
        let clipped_end = item.end_ns.min(scene.time_window.end_ns);
        if clipped_end <= clipped_start {
            continue;
        }
        out.push(RenderCandidate {
            item,
            clipped_start,
            clipped_end,
            raw_width: (clipped_end - clipped_start) as f64 * scale,
            item_role: item.role.unwrap_or(track.role),
        });
    }
    out
}

fn choose_density_threshold(
    candidates: &[RenderCandidate<'_>],
    density_bin_px: Option<f64>,
    scene: &VizScene,
    scale: f64,
) -> Option<DensityThreshold> {
    let bin_px = density_bin_px?;
    if candidates.is_empty() {
        return Some(DensityThreshold {
            interval_px: scene.render_policy.min_interval_px,
            bin_px,
        });
    }

    let base_threshold = scene.render_policy.min_interval_px;
    let base_count = rendered_count_for_threshold(
        candidates,
        base_threshold,
        base_threshold,
        scene,
        scale,
        bin_px,
    );
    if base_count <= scene.render_policy.max_items {
        return Some(DensityThreshold {
            interval_px: base_threshold,
            bin_px,
        });
    }

    let max_width = candidates
        .iter()
        .filter(|candidate| is_density_eligible(candidate))
        .map(|candidate| candidate.raw_width)
        .fold(base_threshold, f64::max);
    let mut low = base_threshold;
    let mut high = max_width.max(base_threshold) + 1.0;
    for _ in 0..32 {
        let mid = (low + high) / 2.0;
        let count =
            rendered_count_for_threshold(candidates, mid, base_threshold, scene, scale, bin_px);
        if count <= scene.render_policy.max_items {
            high = mid;
        } else {
            low = mid;
        }
    }
    Some(DensityThreshold {
        interval_px: high,
        bin_px,
    })
}

fn rendered_count_for_threshold(
    candidates: &[RenderCandidate<'_>],
    threshold: f64,
    base_threshold: f64,
    scene: &VizScene,
    scale: f64,
    bin_px: f64,
) -> usize {
    let mut explicit_count = 0usize;
    let mut density_bins = BTreeSet::<DensityBinKey>::new();
    for candidate in candidates {
        if should_omit_annotation_tick(candidate, base_threshold) {
            continue;
        }
        if should_density_bin(
            candidate,
            Some(DensityThreshold {
                interval_px: threshold,
                bin_px,
            }),
            base_threshold,
        ) {
            density_bins.insert(DensityBinKey {
                track_key: candidate.item.track_key.clone(),
                bin_index: density_bin_index(
                    candidate.clipped_start,
                    candidate.clipped_end,
                    scene.time_window,
                    scale,
                    bin_px,
                ),
            });
        } else {
            explicit_count = explicit_count.saturating_add(1);
        }
    }
    explicit_count.saturating_add(density_bins.len())
}

fn limit_explicit_items(
    mut explicit_items: Vec<&VizInterval>,
    max_items: usize,
) -> Vec<&VizInterval> {
    if explicit_items.len() <= max_items {
        return explicit_items;
    }
    if max_items == 0 {
        return Vec::new();
    }
    explicit_items.sort_by(|left, right| {
        left.start_ns
            .cmp(&right.start_ns)
            .then_with(|| left.end_ns.cmp(&right.end_ns))
            .then_with(|| left.track_key.cmp(&right.track_key))
            .then_with(|| left.row_id.cmp(&right.row_id))
    });

    if max_items == 1 {
        let mid = explicit_items.len() / 2;
        return explicit_items
            .get(mid)
            .copied()
            .into_iter()
            .collect::<Vec<_>>();
    }

    let last_item = explicit_items.len().saturating_sub(1);
    let last_slot = max_items.saturating_sub(1);
    (0..max_items)
        .filter_map(|slot| {
            let index = slot.saturating_mul(last_item) / last_slot;
            explicit_items.get(index).copied()
        })
        .collect()
}

fn dominant_density_class(breakdown: &BTreeMap<String, DensityClassAccumulator>) -> String {
    breakdown
        .iter()
        .max_by(|(left_class, left), (right_class, right)| {
            left.total_duration_ns
                .cmp(&right.total_duration_ns)
                .then_with(|| left.count.cmp(&right.count))
                .then_with(|| right_class.cmp(left_class))
        })
        .map(|(class, _)| class.clone())
        .unwrap_or_else(|| "unknown".to_string())
}

fn dominant_density_visual(
    breakdown: &BTreeMap<DensityVisualKey, DensityClassAccumulator>,
) -> Option<&DensityVisualKey> {
    breakdown
        .iter()
        .max_by(|(left_key, left), (right_key, right)| {
            left.total_duration_ns
                .cmp(&right.total_duration_ns)
                .then_with(|| left.count.cmp(&right.count))
                .then_with(|| {
                    left_key
                        .highlight_key
                        .is_some()
                        .cmp(&right_key.highlight_key.is_some())
                })
                .then_with(|| right_key.cmp(left_key))
        })
        .map(|(key, _)| key)
}

fn density_class_breakdown(
    breakdown: BTreeMap<String, DensityClassAccumulator>,
) -> Vec<DensityClassBreakdown> {
    let mut classes = breakdown
        .into_iter()
        .map(|(class, value)| DensityClassBreakdown {
            class,
            count: value.count,
            total_duration_ns: value.total_duration_ns,
        })
        .collect::<Vec<_>>();
    classes.sort_by(|left, right| {
        right
            .total_duration_ns
            .cmp(&left.total_duration_ns)
            .then_with(|| right.count.cmp(&left.count))
            .then_with(|| left.class.cmp(&right.class))
    });
    classes
}

fn effective_density_bin_px(scene: &VizScene) -> Option<f64> {
    if scene.render_policy.aggregation != VizAggregation::DensityBins {
        return None;
    }
    let bin_px = scene.render_policy.density_bin_px;
    if bin_px.is_finite() && bin_px > 0.0 {
        Some(bin_px)
    } else {
        None
    }
}

fn should_density_bin(
    candidate: &RenderCandidate<'_>,
    threshold: Option<DensityThreshold>,
    base_threshold: f64,
) -> bool {
    let Some(threshold) = threshold else {
        return false;
    };
    if !is_density_eligible(candidate) {
        return false;
    }
    if candidate.raw_width < base_threshold {
        return true;
    }
    threshold.interval_px > base_threshold && candidate.raw_width < threshold.interval_px
}

fn is_density_eligible(candidate: &RenderCandidate<'_>) -> bool {
    !matches!(candidate.item_role, VizRole::Annotation | VizRole::Overlay)
}

fn should_omit_annotation_tick(candidate: &RenderCandidate<'_>, min_interval_px: f64) -> bool {
    matches!(candidate.item_role, VizRole::Annotation) && candidate.raw_width < min_interval_px
}

fn density_heat_height(track_role: VizRole, item_role: VizRole) -> f64 {
    match (track_role, item_role) {
        (VizRole::Summary, _) => 4.0,
        _ => 4.0,
    }
}

fn density_bin_index(
    clipped_start: i64,
    clipped_end: i64,
    window: VizTimeWindow,
    scale: f64,
    bin_px: f64,
) -> u64 {
    let midpoint = clipped_start + (clipped_end - clipped_start) / 2;
    let offset_px = (midpoint - window.start_ns) as f64 * scale;
    (offset_px / bin_px).floor().max(0.0) as u64
}

fn density_bin_bounds(
    bin_index: u64,
    window: VizTimeWindow,
    scale: f64,
    bin_px: f64,
) -> (i64, i64) {
    let start_offset_ns = ((bin_index as f64 * bin_px) / scale).floor() as i64;
    let end_offset_ns = (((bin_index.saturating_add(1)) as f64 * bin_px) / scale).ceil() as i64;
    let start_ns = window.start_ns.saturating_add(start_offset_ns);
    let end_ns = window.start_ns.saturating_add(end_offset_ns);
    (start_ns.max(window.start_ns), end_ns.min(window.end_ns))
}
