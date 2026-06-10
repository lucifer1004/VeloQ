use veloq_core::time::format_duration_ns;

use crate::{VizHighlight, VizHighlightScore, VizInterval, VizLabelPolicy};

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
pub(super) fn fit_interval_label(
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
