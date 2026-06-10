use std::collections::{BTreeMap, BTreeSet};

use veloq_vis::{VizHighlight, VizTrack};

use crate::EventKind;

use super::events::TimelineEvent;
use super::keys::{gpu_summary_track_key, stream_track_key};
use super::spec::{HighlightMetric, HighlightScope, KernelHighlightSpec};
use super::types::{VizResolvedHighlight, VizUnresolvedHighlight};

#[derive(Debug, Default)]
pub(super) struct ResolvedKernelHighlights {
    pub(super) response_highlights: Vec<VizResolvedHighlight>,
    pub(super) scene_highlights: Vec<VizHighlight>,
    pub(super) unresolved_highlights: Vec<VizUnresolvedHighlight>,
    pub(super) assignments: HighlightAssignments,
}

#[derive(Debug, Default)]
pub(super) struct HighlightAssignments {
    by_full_name: BTreeMap<String, String>,
    by_row_id: BTreeMap<String, String>,
}

impl HighlightAssignments {
    pub(super) fn for_event(&self, event: &TimelineEvent) -> Option<String> {
        if event.kind != EventKind::Kernel {
            return None;
        }
        let row_id = event.row_id.to_string();
        self.by_row_id
            .get(&row_id)
            .or_else(|| self.by_full_name.get(&event.full_name))
            .cloned()
    }
}

#[derive(Debug, Clone, Default)]
struct KernelAggregate {
    label: Option<String>,
    total_duration_ns: i64,
    instance_count: usize,
    max_duration_ns: i64,
}

impl KernelAggregate {
    fn add(&mut self, label: &str, duration_ns: i64) {
        if self.label.is_none() {
            self.label = Some(label.to_string());
        }
        self.total_duration_ns = self.total_duration_ns.saturating_add(duration_ns);
        self.instance_count = self.instance_count.saturating_add(1);
        self.max_duration_ns = self.max_duration_ns.max(duration_ns);
    }
}

const HIGHLIGHT_COLORS: &[&str] = &[
    "#f97316", "#a855f7", "#e11d48", "#ca8a04", "#9333ea", "#dc2626", "#4f46e5", "#0f766e",
];

pub(super) fn resolve_kernel_highlights(
    specs: &[KernelHighlightSpec],
    tracks: &[VizTrack],
    gpu_events: &[TimelineEvent],
) -> ResolvedKernelHighlights {
    let candidates = highlight_candidate_kernels(tracks, gpu_events);
    let mut out = ResolvedKernelHighlights::default();
    for (spec_idx, spec) in specs.iter().enumerate() {
        let before = out.response_highlights.len();
        match spec.scope {
            HighlightScope::Name => {
                resolve_name_highlights(spec_idx, spec, &candidates, &mut out);
            }
            HighlightScope::Instance => {
                resolve_instance_highlights(spec_idx, spec, &candidates, &mut out);
            }
        }
        if out.response_highlights.len() == before {
            out.unresolved_highlights.push(VizUnresolvedHighlight {
                spec: spec.raw.clone(),
                reason: "no_matching_kernel_events".to_string(),
            });
        }
    }
    out.scene_highlights = out
        .response_highlights
        .iter()
        .map(VizResolvedHighlight::to_scene_highlight)
        .collect();
    out
}

fn highlight_candidate_kernels<'a>(
    tracks: &[VizTrack],
    gpu_events: &'a [TimelineEvent],
) -> Vec<&'a TimelineEvent> {
    let track_keys: BTreeSet<&str> = tracks.iter().map(|track| track.key.as_str()).collect();
    gpu_events
        .iter()
        .filter(|event| event.kind == EventKind::Kernel)
        .filter(|event| kernel_event_has_rendered_gpu_track(event, &track_keys))
        .collect()
}

fn kernel_event_has_rendered_gpu_track(event: &TimelineEvent, track_keys: &BTreeSet<&str>) -> bool {
    if let Some(device) = event.device_id {
        let key = gpu_summary_track_key(device);
        if track_keys.contains(key.as_str()) {
            return true;
        }
    }
    if let (Some(device), Some(stream)) = (event.device_id, event.stream_id) {
        let key = stream_track_key(device, stream);
        if track_keys.contains(key.as_str()) {
            return true;
        }
    }
    false
}

fn resolve_name_highlights(
    spec_idx: usize,
    spec: &KernelHighlightSpec,
    candidates: &[&TimelineEvent],
    out: &mut ResolvedKernelHighlights,
) {
    let mut aggregates: BTreeMap<String, KernelAggregate> = BTreeMap::new();
    for event in candidates {
        aggregates
            .entry(event.full_name.clone())
            .or_default()
            .add(&event.name, event.duration_ns());
    }
    let mut ranked: Vec<(String, KernelAggregate)> = aggregates.into_iter().collect();
    ranked.sort_by(|a, b| {
        spec.metric
            .score(
                b.1.total_duration_ns,
                b.1.instance_count,
                b.1.max_duration_ns,
            )
            .cmp(&spec.metric.score(
                a.1.total_duration_ns,
                a.1.instance_count,
                a.1.max_duration_ns,
            ))
            .then_with(|| a.0.cmp(&b.0))
    });
    let score_total = aggregate_score_total(spec.metric, &ranked);
    for (idx, (name, agg)) in ranked.into_iter().take(spec.top).enumerate() {
        let rank = idx + 1;
        let key = format!("kernel-highlight|name|spec:{spec_idx}|rank:{rank}");
        out.assignments
            .by_full_name
            .insert(name.clone(), key.clone());
        let label = agg.label.unwrap_or_else(|| name.clone());
        let score = spec.metric.score(
            agg.total_duration_ns,
            agg.instance_count,
            agg.max_duration_ns,
        );
        out.response_highlights.push(VizResolvedHighlight {
            key,
            rank,
            color: highlight_color(out.response_highlights.len()),
            label,
            full_name: name,
            scope: spec.scope.as_str().to_string(),
            metric: spec.metric.as_str().to_string(),
            score,
            score_total,
            total_duration_ns: agg.total_duration_ns,
            instance_count: agg.instance_count,
            max_duration_ns: agg.max_duration_ns,
            row_id: None,
        });
    }
}

fn resolve_instance_highlights(
    spec_idx: usize,
    spec: &KernelHighlightSpec,
    candidates: &[&TimelineEvent],
    out: &mut ResolvedKernelHighlights,
) {
    let score_total = instance_score_total(spec.metric, candidates);
    let mut ranked = candidates.to_vec();
    ranked.sort_by(|a, b| {
        let a_duration = a.duration_ns();
        let b_duration = b.duration_ns();
        spec.metric
            .score(b_duration, 1, b_duration)
            .cmp(&spec.metric.score(a_duration, 1, a_duration))
            .then_with(|| b_duration.cmp(&a_duration))
            .then_with(|| a.start_ns.cmp(&b.start_ns))
            .then_with(|| a.row_id.to_string().cmp(&b.row_id.to_string()))
    });
    for (idx, event) in ranked.into_iter().take(spec.top).enumerate() {
        let rank = idx + 1;
        let row_id = event.row_id.to_string();
        let key = format!("kernel-highlight|instance|spec:{spec_idx}|rank:{rank}");
        let duration_ns = event.duration_ns();
        out.assignments
            .by_row_id
            .insert(row_id.clone(), key.clone());
        out.response_highlights.push(VizResolvedHighlight {
            key,
            rank,
            color: highlight_color(out.response_highlights.len()),
            label: event.name.clone(),
            full_name: event.full_name.clone(),
            scope: spec.scope.as_str().to_string(),
            metric: spec.metric.as_str().to_string(),
            score: spec.metric.score(duration_ns, 1, duration_ns),
            score_total,
            total_duration_ns: duration_ns,
            instance_count: 1,
            max_duration_ns: duration_ns,
            row_id: Some(row_id),
        });
    }
}

fn aggregate_score_total(
    metric: HighlightMetric,
    ranked: &[(String, KernelAggregate)],
) -> Option<i64> {
    match metric {
        HighlightMetric::Duration => Some(ranked.iter().fold(0i64, |total, (_, agg)| {
            total.saturating_add(agg.total_duration_ns)
        })),
        HighlightMetric::Count => Some(ranked.iter().fold(0i64, |total, (_, agg)| {
            let count = i64::try_from(agg.instance_count).unwrap_or(i64::MAX);
            total.saturating_add(count)
        })),
        HighlightMetric::MaxDuration => None,
    }
}

fn instance_score_total(metric: HighlightMetric, candidates: &[&TimelineEvent]) -> Option<i64> {
    match metric {
        HighlightMetric::Duration => Some(candidates.iter().fold(0i64, |total, event| {
            total.saturating_add(event.duration_ns())
        })),
        HighlightMetric::Count => Some(i64::try_from(candidates.len()).unwrap_or(i64::MAX)),
        HighlightMetric::MaxDuration => None,
    }
}

fn highlight_color(idx: usize) -> String {
    HIGHLIGHT_COLORS
        .get(idx % HIGHLIGHT_COLORS.len())
        .copied()
        .unwrap_or("#f97316")
        .to_string()
}
