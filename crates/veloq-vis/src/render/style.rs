use crate::VizRole;

pub(super) struct ItemStyle {
    pub(super) height: f64,
    pub(super) y_offset: f64,
    pub(super) opacity: f64,
}

pub(super) fn item_style(
    track_role: VizRole,
    item_role: VizRole,
    class: Option<&str>,
) -> ItemStyle {
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

pub(super) fn suppress_role_label(track_role: VizRole, item_role: VizRole) -> bool {
    matches!(track_role, VizRole::Summary) || matches!(item_role, VizRole::Overlay)
}
pub(super) fn item_color(class: Option<&str>) -> &'static str {
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

pub(super) fn item_opacity(class: Option<&str>) -> f64 {
    match class {
        Some("gap") => 0.72,
        Some("nvtx") => 0.88,
        _ => 1.0,
    }
}
