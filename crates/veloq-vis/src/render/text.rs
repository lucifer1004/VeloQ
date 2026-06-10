use crate::{VizHighlight, VizInterval, VizLabelPolicy};

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

pub(super) fn format_time_tick(ns: i64) -> String {
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

pub(super) fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}
