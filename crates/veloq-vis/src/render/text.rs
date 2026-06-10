use veloq_core::time::format_duration_ns;

use crate::{VizHighlight, VizHighlightScore, VizInterval, VizLabelPolicy};

use super::style::{INTERVAL_LABEL_FONT_PX, INTERVAL_LABEL_MIN_WIDTH_PX, INTERVAL_LABEL_PADDING_X};

pub(super) fn tooltip_for(item: &VizInterval, highlight: Option<&VizHighlight>) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(row_id) = &item.row_id {
        parts.push(row_id.clone());
    }
    if let Some(label) = &item.label {
        parts.push(label.clone());
    }
    if let Some(highlight) = highlight {
        parts.push(format!("highlight: {}", highlight.full_label));
        if let Some(score) = &highlight.score {
            parts.push(format!("score: {}", format_highlight_score(score)));
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" | "))
    }
}

pub(super) struct DensityTooltipBreakdown<'a> {
    pub(super) class: &'a str,
    pub(super) count: usize,
    pub(super) total_duration_ns: i64,
}

pub(super) fn density_tooltip_for(
    dominant_class: &str,
    count: usize,
    total_duration_ns: i64,
    max_duration_ns: i64,
    start_ns: i64,
    end_ns: i64,
    breakdown: &[DensityTooltipBreakdown<'_>],
) -> String {
    let mut parts = vec![format!(
        "density: {count} intervals | dominant {dominant_class} | total {} | max {} | bin {}..{}",
        format_duration_ns(total_duration_ns),
        format_duration_ns(max_duration_ns),
        format_duration_ns(start_ns),
        format_duration_ns(end_ns)
    )];
    if breakdown.len() > 1 {
        let classes = breakdown
            .iter()
            .map(|class| {
                format!(
                    "{}: {}x/{}",
                    class.class,
                    class.count,
                    format_duration_ns(class.total_duration_ns)
                )
            })
            .collect::<Vec<_>>();
        parts.push(format!("breakdown: {}", classes.join(", ")));
    }
    parts.join(" | ")
}

pub(super) fn fit_interval_label(
    label: &str,
    raw_width: f64,
    policy: &VizLabelPolicy,
) -> Option<(String, bool)> {
    // Text is clipped to the interval bar, so the string can be longer than
    // the visible span. This threshold only prevents noisy one-letter labels.
    let min_visible_width = INTERVAL_LABEL_MIN_WIDTH_PX.max(policy.min_label_px);
    if raw_width < min_visible_width {
        return None;
    }
    let available_width = (raw_width - INTERVAL_LABEL_PADDING_X).max(0.0);
    let truncated = estimate_text_width(label, INTERVAL_LABEL_FONT_PX) > available_width;
    Some((label.to_string(), truncated))
}

pub(super) fn truncate_label(label: &str, max_chars: usize) -> (String, bool) {
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

pub(super) fn estimate_text_width(label: &str, font_px: f64) -> f64 {
    // ~0.58 average character-to-font-size width ratio.
    label.chars().count() as f64 * font_px * 0.58
}

pub(super) fn format_highlight_score(score: &VizHighlightScore) -> String {
    let label = match score.metric.as_str() {
        "total_duration_ns" => format!("total {}", format_duration_ns(score.value)),
        "instance_count" => format!("count {}x", score.value),
        "max_duration_ns" => format!("max {}", format_duration_ns(score.value)),
        _ => format!("score {}", score.value),
    };
    match score.metric.as_str() {
        "total_duration_ns" | "instance_count" => format_score_share(label, score),
        _ => label,
    }
}

fn format_score_share(label: String, score: &VizHighlightScore) -> String {
    let Some(total) = score.total else {
        return label;
    };
    if total <= 0 {
        return label;
    }
    let percentage = score.value as f64 * 100.0 / total as f64;
    format!("{label} ({percentage:.1}%)")
}

pub(super) fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}
