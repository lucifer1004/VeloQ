//! NSys static timeline SVG figure export.
//!
//! This module owns NSys track semantics and evidence extraction. The
//! source-neutral scene shape and SVG writer live in `veloq-vis`.

use crate::query_sql::{
    event_scan::{EventScanFilterOptions, NvtxFilterPolicy, event_scan_filter},
    event_semantics::EventSemantics,
    exec, gpu_work,
};
use crate::{EventKind, NsysQueryError, NsysQueryResult, RowId};
use schemars::JsonSchema;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use veloq_core::artifact_dir_for;
use veloq_core::time::TimeWindow;
use veloq_nsys_data::Trace;
use veloq_query::sql::SqlFragment;
use veloq_vis::{
    SvgRenderSummary, VizAxis, VizHighlight, VizInterval, VizLabelPolicy, VizRenderPolicy, VizRole,
    VizScene, VizTimeWindow, VizTrack, render_svg, write_svg_artifact,
};

pub const DEFAULT_TOP_STREAMS: usize = 8;

#[derive(Debug, Clone, Default)]
pub struct VizTimelineRequest {
    pub time_window: Option<TimeWindow>,
    pub tracks: Vec<String>,
    pub highlight_kernels: Vec<String>,
    pub render_policy: VizRenderPolicy,
    pub label_policy: VizLabelPolicy,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct VizTimelineResponse {
    pub count: usize,
    pub total_matched: usize,
    pub rows: Vec<VizTimelineFigureRow>,
    pub auxiliary: VizTimelineAuxiliary,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct VizTimelineFigureRow {
    pub key: String,
    pub path: String,
    pub format: String,
    pub time_window_ns: [i64; 2],
    pub track_count: usize,
    pub rendered_item_count: usize,
    pub total_item_count: usize,
    pub aggregated: bool,
    pub omitted_track_count: usize,
    pub suppressed_label_count: usize,
    pub truncated_label_count: usize,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct VizTimelineAuxiliary {
    pub requested_tracks: Vec<String>,
    pub resolved_tracks: Vec<VizResolvedTrack>,
    pub requested_highlights: Vec<String>,
    pub resolved_highlights: Vec<VizResolvedHighlight>,
    pub unresolved_highlights: Vec<VizUnresolvedHighlight>,
    pub render_policy: VizRenderPolicyEcho,
    pub label_policy: VizLabelPolicyEcho,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct VizResolvedTrack {
    pub track_key: String,
    pub label: String,
    pub kind: String,
    pub role: String,
    pub axes: Vec<VizAxis>,
    pub render_policy: String,
    pub label_policy: String,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct VizResolvedHighlight {
    pub key: String,
    pub rank: usize,
    pub color: String,
    pub label: String,
    pub full_name: String,
    pub scope: String,
    pub metric: String,
    pub total_duration_ns: i64,
    pub instance_count: usize,
    pub max_duration_ns: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub row_id: Option<String>,
}

impl VizResolvedHighlight {
    fn to_scene_highlight(&self) -> VizHighlight {
        VizHighlight {
            key: self.key.clone(),
            label: self.label.clone(),
            full_label: self.full_name.clone(),
            color: self.color.clone(),
            rank: Some(self.rank),
            scope: Some(self.scope.clone()),
        }
    }
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct VizUnresolvedHighlight {
    pub spec: String,
    pub reason: String,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct VizRenderPolicyEcho {
    pub width_px: u32,
    pub max_tracks: usize,
    pub max_items: usize,
    pub min_interval_px: f64,
    pub aggregation: String,
}

impl From<&VizRenderPolicy> for VizRenderPolicyEcho {
    fn from(policy: &VizRenderPolicy) -> Self {
        Self {
            width_px: policy.width_px,
            max_tracks: policy.max_tracks,
            max_items: policy.max_items,
            min_interval_px: policy.min_interval_px,
            aggregation: policy.aggregation.to_string(),
        }
    }
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct VizLabelPolicyEcho {
    pub mode: String,
    pub min_label_px: f64,
    pub max_chars: usize,
}

impl From<&VizLabelPolicy> for VizLabelPolicyEcho {
    fn from(policy: &VizLabelPolicy) -> Self {
        Self {
            mode: policy.mode.to_string(),
            min_label_px: policy.min_label_px,
            max_chars: policy.max_chars,
        }
    }
}

pub fn default_track_specs() -> Vec<String> {
    vec![
        "gpu:device=all".to_string(),
        format!("cuda-streams:device=all,top={DEFAULT_TOP_STREAMS}"),
        "gaps-overlay:device=all".to_string(),
    ]
}

pub fn run<P: AsRef<Path>>(
    path: P,
    req: VizTimelineRequest,
) -> NsysQueryResult<VizTimelineResponse> {
    let trace = Trace::open(path).map_err(NsysQueryError::trace_open)?;
    let Some(window_req) = req.time_window else {
        return Err(NsysQueryError::VizTimelineWindowRequired);
    };
    let (start_ns, end_ns) = trace
        .resolve_window(Some(window_req))
        .map_err(NsysQueryError::time_window_resolve)?
        .ok_or(NsysQueryError::VizTimelineWindowRequired)?;
    let time_window = VizTimeWindow { start_ns, end_ns };

    let requested_tracks = if req.tracks.is_empty() {
        default_track_specs()
    } else {
        req.tracks.clone()
    };
    let parsed_tracks = requested_tracks
        .iter()
        .map(|spec| TrackSpec::parse(spec))
        .collect::<NsysQueryResult<Vec<_>>>()?;
    let parsed_highlights = req
        .highlight_kernels
        .iter()
        .map(|spec| KernelHighlightSpec::parse(spec))
        .collect::<NsysQueryResult<Vec<_>>>()?;

    let gpu_events = query_gpu_events(&trace, (start_ns, end_ns))?;
    let api_events = if parsed_tracks
        .iter()
        .any(|spec| spec.kind == TrackKind::CudaApi)
    {
        query_runtime_events(&trace, (start_ns, end_ns))?
    } else {
        Vec::new()
    };
    let nvtx_events = if parsed_tracks
        .iter()
        .any(|spec| spec.kind == TrackKind::Nvtx)
    {
        query_nvtx_events(&trace, (start_ns, end_ns))?
    } else {
        Vec::new()
    };

    let resolved = resolve_tracks(
        &parsed_tracks,
        &gpu_events,
        !api_events.is_empty(),
        !nvtx_events.is_empty(),
    )?;
    let highlights = resolve_kernel_highlights(&parsed_highlights, &resolved.tracks, &gpu_events);
    let intervals = build_intervals(
        &resolved.tracks,
        &parsed_tracks,
        &gpu_events,
        &api_events,
        &nvtx_events,
        &highlights.assignments,
    );
    let scene = VizScene {
        title: Some("NSys timeline".to_string()),
        time_window,
        tracks: resolved.tracks.clone(),
        intervals,
        highlights: highlights.scene_highlights,
        render_policy: req.render_policy.clone(),
        label_policy: req.label_policy.clone(),
    };
    let rendered =
        render_svg(&scene).map_err(|source| NsysQueryError::VizTimelineArtifact { source })?;

    let fingerprint = request_fingerprint(
        start_ns,
        end_ns,
        &requested_tracks,
        &req.highlight_kernels,
        &req.render_policy,
        &req.label_policy,
    );
    let file_name = format!("timeline-{start_ns}-{end_ns}-{fingerprint}.svg");
    let artifact_root = artifact_dir_for(trace.path());
    let written = write_svg_artifact(
        &artifact_root,
        Path::new("figures/nsys/timeline"),
        &file_name,
        &rendered.svg,
    )
    .map_err(|source| NsysQueryError::VizTimelineArtifact { source })?;

    let row = figure_row(
        start_ns,
        end_ns,
        &fingerprint,
        written.relative_path,
        &rendered.summary,
    );
    Ok(VizTimelineResponse {
        count: 1,
        total_matched: 1,
        rows: vec![row],
        auxiliary: VizTimelineAuxiliary {
            requested_tracks,
            resolved_tracks: resolved.response_tracks,
            requested_highlights: req.highlight_kernels,
            resolved_highlights: highlights.response_highlights,
            unresolved_highlights: highlights.unresolved_highlights,
            render_policy: VizRenderPolicyEcho::from(&req.render_policy),
            label_policy: VizLabelPolicyEcho::from(&req.label_policy),
        },
    })
}

fn figure_row(
    start_ns: i64,
    end_ns: i64,
    fingerprint: &str,
    path: String,
    summary: &SvgRenderSummary,
) -> VizTimelineFigureRow {
    VizTimelineFigureRow {
        key: format!("figure|timeline|{start_ns}..{end_ns}|req:{fingerprint}"),
        path,
        format: "svg".to_string(),
        time_window_ns: [start_ns, end_ns],
        track_count: summary.track_count,
        rendered_item_count: summary.rendered_item_count,
        total_item_count: summary.total_item_count,
        aggregated: summary.aggregated,
        omitted_track_count: summary.omitted_track_count,
        suppressed_label_count: summary.suppressed_label_count,
        truncated_label_count: summary.truncated_label_count,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrackKind {
    Gpu,
    CudaStreams,
    CudaStream,
    CudaApi,
    Nvtx,
    GapsOverlay,
}

impl TrackKind {
    fn parse(raw: &str) -> NsysQueryResult<Self> {
        match raw {
            "gpu" => Ok(Self::Gpu),
            "cuda-streams" => Ok(Self::CudaStreams),
            "cuda-stream" => Ok(Self::CudaStream),
            "cuda-api" => Ok(Self::CudaApi),
            "nvtx" => Ok(Self::Nvtx),
            "gaps-overlay" => Ok(Self::GapsOverlay),
            _ => Err(NsysQueryError::VizTimelineUnknownTrackKind {
                kind: raw.to_string(),
            }),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Gpu => "gpu",
            Self::CudaStreams => "cuda-streams",
            Self::CudaStream => "cuda-stream",
            Self::CudaApi => "cuda-api",
            Self::Nvtx => "nvtx",
            Self::GapsOverlay => "gaps-overlay",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DeviceSelector {
    All,
    One(i32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TrackSpec {
    kind: TrackKind,
    device: Option<DeviceSelector>,
    stream: Option<i64>,
    top: Option<usize>,
    depth: Option<usize>,
}

impl TrackSpec {
    fn parse(raw: &str) -> NsysQueryResult<Self> {
        let (kind_raw, selectors_raw) = match raw.split_once(':') {
            Some((kind, selectors)) => (kind.trim(), Some(selectors)),
            None => (raw.trim(), None),
        };
        let kind = TrackKind::parse(kind_raw)?;
        let mut spec = Self {
            kind,
            device: None,
            stream: None,
            top: None,
            depth: None,
        };
        if let Some(selectors) = selectors_raw {
            for selector in selectors.split(',') {
                let selector = selector.trim();
                if selector.is_empty() {
                    continue;
                }
                let (name, value) = selector.split_once('=').ok_or_else(|| {
                    NsysQueryError::VizTimelineInvalidSelector {
                        selector: selector.to_string(),
                    }
                })?;
                spec.apply_selector(name.trim(), value.trim())?;
            }
        }
        spec.validate()?;
        Ok(spec)
    }

    fn apply_selector(&mut self, name: &str, value: &str) -> NsysQueryResult<()> {
        match name {
            "device" if self.kind_accepts_selector("device") => {
                self.device = Some(parse_device_selector(value)?);
                Ok(())
            }
            "stream" if self.kind_accepts_selector("stream") => {
                self.stream = Some(parse_non_negative_i64("stream", value)?);
                Ok(())
            }
            "top" if self.kind_accepts_selector("top") => {
                self.top = Some(parse_positive_usize("top", value)?);
                Ok(())
            }
            "depth" if self.kind_accepts_selector("depth") => {
                self.depth = Some(parse_positive_usize("depth", value)?);
                Ok(())
            }
            _ => Err(NsysQueryError::VizTimelineUnknownSelector {
                kind: self.kind.as_str().to_string(),
                selector: name.to_string(),
            }),
        }
    }

    fn kind_accepts_selector(&self, selector: &str) -> bool {
        matches!(
            (self.kind, selector),
            (TrackKind::Gpu, "device")
                | (TrackKind::CudaStreams, "device" | "top")
                | (TrackKind::CudaStream, "device" | "stream")
                | (TrackKind::Nvtx, "depth")
                | (TrackKind::GapsOverlay, "device")
        )
    }

    fn validate(&self) -> NsysQueryResult<()> {
        if self.kind == TrackKind::CudaStream {
            match self.device {
                Some(DeviceSelector::One(_)) => {}
                Some(DeviceSelector::All) => {
                    return Err(NsysQueryError::VizTimelineCudaStreamDeviceAll);
                }
                None => return Err(NsysQueryError::VizTimelineCudaStreamDeviceRequired),
            }
            if self.stream.is_none() {
                return Err(NsysQueryError::VizTimelineCudaStreamStreamRequired);
            }
        }
        Ok(())
    }
}

fn parse_device_selector(value: &str) -> NsysQueryResult<DeviceSelector> {
    if value == "all" {
        return Ok(DeviceSelector::All);
    }
    let raw = parse_non_negative_i64("device", value)?;
    let device =
        i32::try_from(raw).map_err(|_| NsysQueryError::VizTimelineSelectorNonNegativeInt {
            selector: "device".to_string(),
        })?;
    Ok(DeviceSelector::One(device))
}

fn parse_non_negative_i64(selector: &str, value: &str) -> NsysQueryResult<i64> {
    let parsed =
        value
            .parse::<i64>()
            .map_err(|_| NsysQueryError::VizTimelineSelectorNonNegativeInt {
                selector: selector.to_string(),
            })?;
    if parsed < 0 {
        return Err(NsysQueryError::VizTimelineSelectorNonNegativeInt {
            selector: selector.to_string(),
        });
    }
    Ok(parsed)
}

fn parse_positive_usize(selector: &str, value: &str) -> NsysQueryResult<usize> {
    let parsed =
        value
            .parse::<usize>()
            .map_err(|_| NsysQueryError::VizTimelineSelectorPositiveInt {
                selector: selector.to_string(),
            })?;
    if parsed == 0 {
        return Err(NsysQueryError::VizTimelineSelectorPositiveInt {
            selector: selector.to_string(),
        });
    }
    Ok(parsed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HighlightScope {
    Name,
    Instance,
}

impl HighlightScope {
    fn parse(raw: &str) -> NsysQueryResult<Self> {
        match raw {
            "name" => Ok(Self::Name),
            "instance" => Ok(Self::Instance),
            _ => Err(NsysQueryError::VizTimelineUnknownHighlightScope {
                scope: raw.to_string(),
            }),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Name => "name",
            Self::Instance => "instance",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HighlightMetric {
    Duration,
    Count,
    MaxDuration,
}

impl HighlightMetric {
    fn parse(raw: &str) -> NsysQueryResult<Self> {
        match raw {
            "duration" | "total-duration" | "total_duration_ns" => Ok(Self::Duration),
            "count" | "instance-count" | "instance_count" => Ok(Self::Count),
            "max-duration" | "max_duration" | "max_duration_ns" => Ok(Self::MaxDuration),
            _ => Err(NsysQueryError::VizTimelineUnknownHighlightMetric {
                metric: raw.to_string(),
            }),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Duration => "total_duration_ns",
            Self::Count => "instance_count",
            Self::MaxDuration => "max_duration_ns",
        }
    }

    fn score(self, total_duration_ns: i64, instance_count: usize, max_duration_ns: i64) -> i64 {
        match self {
            Self::Duration => total_duration_ns,
            Self::Count => i64::try_from(instance_count).unwrap_or(i64::MAX),
            Self::MaxDuration => max_duration_ns,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct KernelHighlightSpec {
    raw: String,
    top: usize,
    scope: HighlightScope,
    metric: HighlightMetric,
}

impl KernelHighlightSpec {
    fn parse(raw: &str) -> NsysQueryResult<Self> {
        let mut top = None;
        let mut scope = HighlightScope::Name;
        let mut metric = HighlightMetric::Duration;
        for selector in raw.split(',') {
            let selector = selector.trim();
            if selector.is_empty() {
                continue;
            }
            let (name, value) = selector.split_once('=').ok_or_else(|| {
                NsysQueryError::VizTimelineInvalidSelector {
                    selector: selector.to_string(),
                }
            })?;
            match name.trim() {
                "top" => {
                    top = Some(parse_positive_usize("highlight-kernels.top", value.trim())?);
                }
                "scope" => {
                    scope = HighlightScope::parse(value.trim())?;
                }
                "by" | "metric" => {
                    metric = HighlightMetric::parse(value.trim())?;
                }
                other => {
                    return Err(NsysQueryError::VizTimelineUnknownSelector {
                        kind: "highlight-kernels".to_string(),
                        selector: other.to_string(),
                    });
                }
            }
        }
        Ok(Self {
            raw: raw.to_string(),
            top: top.ok_or(NsysQueryError::VizTimelineHighlightTopRequired)?,
            scope,
            metric,
        })
    }
}

#[derive(Debug, Default)]
struct ResolvedKernelHighlights {
    response_highlights: Vec<VizResolvedHighlight>,
    scene_highlights: Vec<VizHighlight>,
    unresolved_highlights: Vec<VizUnresolvedHighlight>,
    assignments: HighlightAssignments,
}

#[derive(Debug, Default)]
struct HighlightAssignments {
    by_full_name: BTreeMap<String, String>,
    by_row_id: BTreeMap<String, String>,
}

impl HighlightAssignments {
    fn for_event(&self, event: &TimelineEvent) -> Option<String> {
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

fn resolve_kernel_highlights(
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
    for (idx, (name, agg)) in ranked.into_iter().take(spec.top).enumerate() {
        let rank = idx + 1;
        let key = format!("kernel-highlight|name|spec:{spec_idx}|rank:{rank}");
        out.assignments
            .by_full_name
            .insert(name.clone(), key.clone());
        let label = agg.label.unwrap_or_else(|| name.clone());
        out.response_highlights.push(VizResolvedHighlight {
            key,
            rank,
            color: highlight_color(out.response_highlights.len()),
            label,
            full_name: name,
            scope: spec.scope.as_str().to_string(),
            metric: spec.metric.as_str().to_string(),
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
            total_duration_ns: duration_ns,
            instance_count: 1,
            max_duration_ns: duration_ns,
            row_id: Some(row_id),
        });
    }
}

fn highlight_color(idx: usize) -> String {
    HIGHLIGHT_COLORS
        .get(idx % HIGHLIGHT_COLORS.len())
        .copied()
        .unwrap_or("#f97316")
        .to_string()
}

#[derive(Debug, Clone)]
struct TimelineEvent {
    row_id: RowId,
    kind: EventKind,
    name: String,
    full_name: String,
    start_ns: i64,
    end_ns: i64,
    device_id: Option<i32>,
    stream_id: Option<i64>,
}

impl TimelineEvent {
    fn duration_ns(&self) -> i64 {
        self.end_ns.saturating_sub(self.start_ns)
    }
}

#[derive(Debug)]
struct ResolvedTracks {
    tracks: Vec<VizTrack>,
    response_tracks: Vec<VizResolvedTrack>,
}

fn resolve_tracks(
    specs: &[TrackSpec],
    gpu_events: &[TimelineEvent],
    has_api_events: bool,
    has_nvtx_events: bool,
) -> NsysQueryResult<ResolvedTracks> {
    let mut tracks = Vec::new();
    let mut response_tracks = Vec::new();
    let mut seen = BTreeSet::new();
    for spec in specs {
        match spec.kind {
            TrackKind::Gpu => {
                for device in devices_for(spec.device.as_ref(), gpu_events) {
                    push_device_group(&mut tracks, &mut response_tracks, &mut seen, device);
                    push_track(
                        &mut tracks,
                        &mut response_tracks,
                        &mut seen,
                        TrackDef::new(
                            gpu_summary_track_key(device),
                            "busy summary".to_string(),
                            "gpu-summary",
                            vec![axis("device", device)],
                        )
                        .depth(1)
                        .role(VizRole::Summary),
                    );
                }
            }
            TrackKind::CudaStreams => {
                for (device, stream) in top_streams_for(spec, gpu_events) {
                    push_device_group(&mut tracks, &mut response_tracks, &mut seen, device);
                    push_track(
                        &mut tracks,
                        &mut response_tracks,
                        &mut seen,
                        TrackDef::new(
                            stream_track_key(device, stream),
                            format!("stream {stream}"),
                            "cuda-stream",
                            vec![axis("device", device), axis("stream", stream)],
                        )
                        .depth(1)
                        .role(VizRole::Detail),
                    );
                }
            }
            TrackKind::CudaStream => {
                let Some(DeviceSelector::One(device)) = spec.device else {
                    return Err(NsysQueryError::VizTimelineCudaStreamDeviceRequired);
                };
                let Some(stream) = spec.stream else {
                    return Err(NsysQueryError::VizTimelineCudaStreamStreamRequired);
                };
                push_device_group(&mut tracks, &mut response_tracks, &mut seen, device);
                push_track(
                    &mut tracks,
                    &mut response_tracks,
                    &mut seen,
                    TrackDef::new(
                        stream_track_key(device, stream),
                        format!("stream {stream}"),
                        "cuda-stream",
                        vec![axis("device", device), axis("stream", stream)],
                    )
                    .depth(1)
                    .role(VizRole::Detail),
                );
            }
            TrackKind::CudaApi => {
                if has_api_events {
                    push_track(
                        &mut tracks,
                        &mut response_tracks,
                        &mut seen,
                        TrackDef::new(
                            "cuda-api".to_string(),
                            "CUDA API".to_string(),
                            "cuda-api",
                            vec![],
                        )
                        .role(VizRole::Annotation),
                    );
                }
            }
            TrackKind::Nvtx => {
                if has_nvtx_events {
                    let depth = spec.depth.unwrap_or(1);
                    push_track(
                        &mut tracks,
                        &mut response_tracks,
                        &mut seen,
                        TrackDef::new(
                            "nvtx|depth:1".to_string(),
                            "NVTX".to_string(),
                            "nvtx",
                            vec![axis("depth", depth)],
                        )
                        .role(VizRole::Annotation),
                    );
                }
            }
            TrackKind::GapsOverlay => {}
        }
    }
    sort_resolved_tracks(&mut tracks, &mut response_tracks);
    Ok(ResolvedTracks {
        tracks,
        response_tracks,
    })
}

struct TrackDef {
    key: String,
    label: String,
    kind: String,
    axes: Vec<VizAxis>,
    depth: usize,
    role: VizRole,
}

impl TrackDef {
    fn new(key: String, label: String, kind: impl Into<String>, axes: Vec<VizAxis>) -> Self {
        Self {
            key,
            label,
            kind: kind.into(),
            axes,
            depth: 0,
            role: VizRole::Detail,
        }
    }

    fn depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    fn role(mut self, role: VizRole) -> Self {
        self.role = role;
        self
    }
}

fn push_device_group(
    tracks: &mut Vec<VizTrack>,
    response_tracks: &mut Vec<VizResolvedTrack>,
    seen: &mut BTreeSet<String>,
    device: i32,
) {
    push_track(
        tracks,
        response_tracks,
        seen,
        TrackDef::new(
            device_group_track_key(device),
            format!("GPU {device}"),
            "gpu-device",
            vec![axis("device", device)],
        )
        .role(VizRole::Group),
    );
}

fn push_track(
    tracks: &mut Vec<VizTrack>,
    response_tracks: &mut Vec<VizResolvedTrack>,
    seen: &mut BTreeSet<String>,
    def: TrackDef,
) {
    if !seen.insert(def.key.clone()) {
        return;
    }
    tracks.push(VizTrack {
        key: def.key.clone(),
        label: def.label.clone(),
        kind: def.kind.clone(),
        role: def.role,
        depth: def.depth,
        axes: def.axes.clone(),
    });
    response_tracks.push(VizResolvedTrack {
        track_key: def.key,
        label: def.label,
        kind: def.kind,
        role: def.role.to_string(),
        axes: def.axes,
        render_policy: "default".to_string(),
        label_policy: "default".to_string(),
    });
}

fn sort_resolved_tracks(tracks: &mut [VizTrack], response_tracks: &mut [VizResolvedTrack]) {
    let insertion = tracks
        .iter()
        .enumerate()
        .map(|(idx, track)| (track.key.clone(), idx))
        .collect::<BTreeMap<_, _>>();
    tracks.sort_by(|a, b| {
        track_order_key(
            a.role,
            &a.axes,
            insertion.get(&a.key).copied().unwrap_or(usize::MAX),
        )
        .cmp(&track_order_key(
            b.role,
            &b.axes,
            insertion.get(&b.key).copied().unwrap_or(usize::MAX),
        ))
    });
    response_tracks.sort_by(|a, b| {
        response_track_order_key(a, &insertion).cmp(&response_track_order_key(b, &insertion))
    });
}

fn response_track_order_key(
    track: &VizResolvedTrack,
    insertion: &BTreeMap<String, usize>,
) -> (u8, i32, u8, usize) {
    let role = match track.role.as_str() {
        "group" => VizRole::Group,
        "summary" => VizRole::Summary,
        "detail" => VizRole::Detail,
        "annotation" => VizRole::Annotation,
        "overlay" => VizRole::Overlay,
        _ => VizRole::Detail,
    };
    track_order_key(
        role,
        &track.axes,
        insertion
            .get(&track.track_key)
            .copied()
            .unwrap_or(usize::MAX),
    )
}

fn track_order_key(role: VizRole, axes: &[VizAxis], insertion: usize) -> (u8, i32, u8, usize) {
    let device = axis_i32(axes, "device");
    if let Some(device) = device
        && matches!(role, VizRole::Group | VizRole::Summary | VizRole::Detail)
    {
        return (0, device, visual_role_rank(role), insertion);
    }
    (1, 0, visual_role_rank(role), insertion)
}

fn visual_role_rank(role: VizRole) -> u8 {
    match role {
        VizRole::Group => 0,
        VizRole::Summary => 1,
        VizRole::Detail => 2,
        VizRole::Overlay => 3,
        VizRole::Annotation => 4,
    }
}

fn axis_i32(axes: &[VizAxis], name: &str) -> Option<i32> {
    axes.iter()
        .find(|axis| axis.name == name)
        .and_then(|axis| axis.value.parse::<i32>().ok())
}

fn devices_for(selector: Option<&DeviceSelector>, events: &[TimelineEvent]) -> Vec<i32> {
    match selector {
        Some(DeviceSelector::One(device)) => vec![*device],
        Some(DeviceSelector::All) | None => events
            .iter()
            .filter_map(|event| event.device_id)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
    }
}

fn top_streams_for(spec: &TrackSpec, events: &[TimelineEvent]) -> Vec<(i32, i64)> {
    let top = spec.top.unwrap_or(DEFAULT_TOP_STREAMS);
    let devices: Option<BTreeSet<i32>> = match spec.device.as_ref() {
        Some(DeviceSelector::One(device)) => Some(BTreeSet::from([*device])),
        Some(DeviceSelector::All) | None => None,
    };
    let mut busy: BTreeMap<(i32, i64), i64> = BTreeMap::new();
    for event in events {
        let (Some(device), Some(stream)) = (event.device_id, event.stream_id) else {
            continue;
        };
        if let Some(devices) = &devices
            && !devices.contains(&device)
        {
            continue;
        }
        let duration = event.end_ns.saturating_sub(event.start_ns);
        *busy.entry((device, stream)).or_insert(0) += duration;
    }
    let mut streams: Vec<((i32, i64), i64)> = busy.into_iter().collect();
    streams.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    streams
        .into_iter()
        .take(top)
        .map(|((device, stream), _)| (device, stream))
        .collect()
}

fn build_intervals(
    tracks: &[VizTrack],
    specs: &[TrackSpec],
    gpu_events: &[TimelineEvent],
    api_events: &[TimelineEvent],
    nvtx_events: &[TimelineEvent],
    highlights: &HighlightAssignments,
) -> Vec<VizInterval> {
    let track_keys: BTreeSet<&str> = tracks.iter().map(|track| track.key.as_str()).collect();
    let mut out = Vec::new();
    for event in gpu_events {
        if let Some(device) = event.device_id {
            let key = gpu_summary_track_key(device);
            if track_keys.contains(key.as_str()) {
                out.push(interval_for_event(key, event, highlights));
            }
        }
        if let (Some(device), Some(stream)) = (event.device_id, event.stream_id) {
            let key = stream_track_key(device, stream);
            if track_keys.contains(key.as_str()) {
                out.push(interval_for_event(key, event, highlights));
            }
        }
    }
    if specs.iter().any(|spec| spec.kind == TrackKind::GapsOverlay) {
        out.extend(gap_overlay_intervals(tracks, gpu_events));
    }
    if track_keys.contains("cuda-api") {
        out.extend(
            api_events
                .iter()
                .map(|event| interval_for_event("cuda-api".to_string(), event, highlights)),
        );
    }
    if track_keys.contains("nvtx|depth:1") {
        out.extend(
            nvtx_events
                .iter()
                .map(|event| interval_for_event("nvtx|depth:1".to_string(), event, highlights)),
        );
    }
    out.sort_by(|a, b| {
        a.start_ns
            .cmp(&b.start_ns)
            .then_with(|| a.track_key.cmp(&b.track_key))
    });
    out
}

fn interval_for_event(
    track_key: String,
    event: &TimelineEvent,
    highlights: &HighlightAssignments,
) -> VizInterval {
    VizInterval {
        track_key,
        start_ns: event.start_ns,
        end_ns: event.end_ns,
        label: Some(event.name.clone()),
        row_id: Some(event.row_id.to_string()),
        class: Some(event.kind.as_str().to_string()),
        role: None,
        highlight_key: highlights.for_event(event),
    }
}

fn gap_overlay_intervals(tracks: &[VizTrack], events: &[TimelineEvent]) -> Vec<VizInterval> {
    let gpu_track_devices: BTreeSet<i32> = tracks
        .iter()
        .filter(|track| track.kind == "gpu-summary")
        .filter_map(|track| {
            track
                .axes
                .iter()
                .find(|axis| axis.name == "device")
                .and_then(|axis| axis.value.parse::<i32>().ok())
        })
        .collect();
    let mut by_device: BTreeMap<i32, Vec<&TimelineEvent>> = BTreeMap::new();
    for event in events {
        if let Some(device) = event.device_id
            && gpu_track_devices.contains(&device)
        {
            by_device.entry(device).or_default().push(event);
        }
    }
    let mut out = Vec::new();
    for (device, mut events) in by_device {
        events.sort_by(|a, b| {
            a.start_ns
                .cmp(&b.start_ns)
                .then_with(|| a.end_ns.cmp(&b.end_ns))
        });
        let mut running_end: Option<i64> = None;
        for event in events {
            if let Some(prev_end) = running_end
                && event.start_ns > prev_end
            {
                out.push(VizInterval {
                    track_key: gpu_summary_track_key(device),
                    start_ns: prev_end,
                    end_ns: event.start_ns,
                    label: Some("idle".to_string()),
                    row_id: None,
                    class: Some("gap".to_string()),
                    role: Some(VizRole::Overlay),
                    highlight_key: None,
                });
            }
            running_end = Some(running_end.map_or(event.end_ns, |end| end.max(event.end_ns)));
        }
    }
    out
}

fn query_gpu_events(trace: &Trace, abs_window: (i64, i64)) -> NsysQueryResult<Vec<TimelineEvent>> {
    let work = gpu_work::GpuWorkSet::from_data_definition()?;
    let kinds = work.present_in(trace);
    if kinds.is_empty() {
        return Ok(Vec::new());
    }
    let mut subqueries = Vec::new();
    let mut params = Vec::new();
    for kind in kinds {
        let fragment = interval_select(kind, abs_window, None, None)?;
        subqueries.push(fragment.sql);
        params.extend(fragment.params);
    }
    let sql = subqueries.join(" UNION ALL ");
    exec::query_rows_fallible(
        trace.conn(),
        &sql,
        &params,
        exec::SqlLabel::new("viz-timeline", "gpu-events"),
        timeline_event_row,
    )
}

fn query_runtime_events(
    trace: &Trace,
    abs_window: (i64, i64),
) -> NsysQueryResult<Vec<TimelineEvent>> {
    if !trace.table_exists(EventKind::Runtime.table()) {
        return Ok(Vec::new());
    }
    let fragment = interval_select(EventKind::Runtime, abs_window, None, None)?;
    exec::query_rows_fallible(
        trace.conn(),
        &fragment.sql,
        &fragment.params,
        exec::SqlLabel::new("viz-timeline", "cuda-api-events"),
        timeline_event_row,
    )
}

fn query_nvtx_events(trace: &Trace, abs_window: (i64, i64)) -> NsysQueryResult<Vec<TimelineEvent>> {
    if !trace.table_exists(EventKind::Nvtx.table()) {
        return Ok(Vec::new());
    }
    let fragment = interval_select(EventKind::Nvtx, abs_window, None, None)?;
    exec::query_rows_fallible(
        trace.conn(),
        &fragment.sql,
        &fragment.params,
        exec::SqlLabel::new("viz-timeline", "nvtx-events"),
        timeline_event_row,
    )
}

fn interval_select(
    kind: EventKind,
    abs_window: (i64, i64),
    device: Option<i32>,
    stream: Option<i64>,
) -> NsysQueryResult<SqlFragment> {
    let sem = EventSemantics::new(kind);
    let mut intrinsic = Vec::new();
    if matches!(kind, EventKind::Nvtx) {
        intrinsic.push(r#"t."end" IS NOT NULL"#);
    }
    let filter = event_scan_filter(
        sem,
        EventScanFilterOptions {
            abs_window: Some(abs_window),
            device,
            stream,
            nvtx_scope: crate::nvtx_attribution::NvtxScope::None,
            nvtx_policy: NvtxFilterPolicy::EmptyWhenUnsupported,
        },
        &intrinsic,
    )?;
    let where_clause = filter.where_clause();
    let sql = format!(
        r#"
        SELECT
            '{label}' AS kind,
            t.rowid AS row_id_num,
            {short_name_expr} AS name,
            {full_name_expr} AS full_name,
            t.start AS start_ns,
            COALESCE(t."end", t.start) AS end_ns,
            {device_expr} AS device_id,
            {stream_expr} AS stream_id
        FROM nsight.{table} t {joins}
        {where_clause}
        "#,
        label = sem.label(),
        short_name_expr = sem.short_name_expr(),
        full_name_expr = sem.display_name_expr(),
        device_expr = sem.device_expr(),
        stream_expr = sem.stream_expr(),
        table = sem.table(),
        joins = sem.name_joins(),
    );
    Ok(SqlFragment::new(sql, filter.into_params()))
}

fn timeline_event_row(row: &duckdb::Row<'_>) -> NsysQueryResult<TimelineEvent> {
    let kind_raw: String = row.get("kind").map_err(viz_timeline_row_read)?;
    let kind = EventKind::parse(&kind_raw)
        .ok_or_else(|| NsysQueryError::internal_sql_kind_tag_invalid("viz-timeline", &kind_raw))?;
    let rowid: i64 = row.get("row_id_num").map_err(viz_timeline_row_read)?;
    Ok(TimelineEvent {
        row_id: RowId::new(kind, rowid),
        kind,
        name: row.get("name").map_err(viz_timeline_row_read)?,
        full_name: row.get("full_name").map_err(viz_timeline_row_read)?,
        start_ns: row.get("start_ns").map_err(viz_timeline_row_read)?,
        end_ns: row.get("end_ns").map_err(viz_timeline_row_read)?,
        device_id: row.get("device_id").map_err(viz_timeline_row_read)?,
        stream_id: row.get("stream_id").map_err(viz_timeline_row_read)?,
    })
}

fn viz_timeline_row_read(source: duckdb::Error) -> NsysQueryError {
    NsysQueryError::sql_read("viz-timeline", "event-row", source)
}

fn stream_track_key(device: i32, stream: i64) -> String {
    format!("cuda-stream|dev:{device}|stream:{stream}")
}

fn device_group_track_key(device: i32) -> String {
    format!("gpu-device|dev:{device}")
}

fn gpu_summary_track_key(device: i32) -> String {
    format!("gpu-summary|dev:{device}")
}

fn axis(name: &str, value: impl ToString) -> VizAxis {
    VizAxis {
        name: name.to_string(),
        value: value.to_string(),
    }
}

fn request_fingerprint(
    start_ns: i64,
    end_ns: i64,
    tracks: &[String],
    highlights: &[String],
    render_policy: &VizRenderPolicy,
    label_policy: &VizLabelPolicy,
) -> String {
    let mut hash = Fnv1a64::new();
    hash.push(&start_ns.to_string());
    hash.push(&end_ns.to_string());
    for track in tracks {
        hash.push(track);
    }
    for highlight in highlights {
        hash.push(highlight);
    }
    hash.push(&render_policy.width_px.to_string());
    hash.push(&render_policy.max_tracks.to_string());
    hash.push(&render_policy.max_items.to_string());
    hash.push(&render_policy.min_interval_px.to_string());
    hash.push(&render_policy.aggregation.to_string());
    hash.push(&label_policy.mode.to_string());
    hash.push(&label_policy.min_label_px.to_string());
    hash.push(&label_policy.max_chars.to_string());
    format!("{:016x}", hash.finish())
}

struct Fnv1a64 {
    state: u64,
}

impl Fnv1a64 {
    fn new() -> Self {
        Self {
            state: 0xcbf29ce484222325,
        }
    }

    fn push(&mut self, value: &str) {
        for byte in value.as_bytes().iter().copied().chain([0]) {
            self.state ^= u64::from(byte);
            self.state = self.state.wrapping_mul(0x100000001b3);
        }
    }

    fn finish(self) -> u64 {
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NsysQueryError;
    use veloq_core::VeloqDiagnostic;

    #[test]
    fn parses_default_track_specs() -> anyhow::Result<()> {
        for spec in default_track_specs() {
            TrackSpec::parse(&spec)?;
        }
        Ok(())
    }

    #[test]
    fn cuda_stream_requires_device_and_stream() -> anyhow::Result<()> {
        let err = match TrackSpec::parse("cuda-stream:stream=7") {
            Ok(_) => anyhow::bail!("missing device should fail"),
            Err(err) => err,
        };
        assert_eq!(
            err.code().as_str(),
            "nsys.query.viz-timeline-cuda-stream-device-required"
        );

        let err = match TrackSpec::parse("cuda-stream:device=all,stream=7") {
            Ok(_) => anyhow::bail!("device=all should fail for exact stream"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            NsysQueryError::VizTimelineCudaStreamDeviceAll
        ));
        Ok(())
    }

    #[test]
    fn fingerprint_is_stable_for_equivalent_input() {
        let render = VizRenderPolicy::default();
        let label = VizLabelPolicy::default();
        let tracks = default_track_specs();
        assert_eq!(
            request_fingerprint(1, 2, &tracks, &[], &render, &label),
            request_fingerprint(1, 2, &tracks, &[], &render, &label)
        );
    }

    #[test]
    fn parses_kernel_highlight_specs() -> anyhow::Result<()> {
        let spec = KernelHighlightSpec::parse("top=3")?;
        assert_eq!(spec.top, 3);
        assert_eq!(spec.scope, HighlightScope::Name);
        assert_eq!(spec.metric, HighlightMetric::Duration);

        let spec = KernelHighlightSpec::parse("top=2,scope=instance,by=max-duration")?;
        assert_eq!(spec.top, 2);
        assert_eq!(spec.scope, HighlightScope::Instance);
        assert_eq!(spec.metric, HighlightMetric::MaxDuration);

        let err = match KernelHighlightSpec::parse("scope=name") {
            Ok(_) => anyhow::bail!("missing top should fail"),
            Err(err) => err,
        };
        assert_eq!(
            err.code().as_str(),
            "nsys.query.viz-timeline-highlight-top-required"
        );
        Ok(())
    }

    #[test]
    fn name_highlights_rank_by_full_name_but_label_with_short_name() -> anyhow::Result<()> {
        let tracks = vec![VizTrack {
            key: gpu_summary_track_key(0),
            label: "busy summary".to_string(),
            kind: "gpu-summary".to_string(),
            role: VizRole::Summary,
            depth: 1,
            axes: vec![axis("device", 0)],
        }];
        let events = vec![
            event_named(1, 0, 7, 0, 100, "short", "void short<int>()"),
            event_named(2, 0, 7, 100, 400, "short", "void short<double>()"),
            event_named(3, 0, 7, 400, 450, "other", "void other()"),
        ];
        let spec = KernelHighlightSpec::parse("top=1,scope=name")?;
        let resolved = resolve_kernel_highlights(&[spec], &tracks, &events);
        let highlight = resolved
            .response_highlights
            .first()
            .ok_or_else(|| anyhow::anyhow!("expected one highlight"))?;

        assert_eq!(highlight.label, "short");
        assert_eq!(highlight.full_name, "void short<double>()");
        assert_eq!(highlight.total_duration_ns, 300);
        assert_eq!(highlight.instance_count, 1);
        assert_eq!(
            resolved.assignments.for_event(
                events
                    .get(1)
                    .ok_or_else(|| anyhow::anyhow!("expected second event"))?
            ),
            Some(highlight.key.clone())
        );
        assert_eq!(
            resolved.assignments.for_event(
                events
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("expected first event"))?
            ),
            None
        );
        Ok(())
    }

    #[test]
    fn instance_highlights_attach_row_id() -> anyhow::Result<()> {
        let tracks = vec![VizTrack {
            key: stream_track_key(0, 7),
            label: "stream 7".to_string(),
            kind: "cuda-stream".to_string(),
            role: VizRole::Detail,
            depth: 1,
            axes: vec![axis("device", 0), axis("stream", 7)],
        }];
        let events = vec![
            event_named(1, 0, 7, 0, 100, "fast", "void fast()"),
            event_named(2, 0, 7, 100, 400, "slow", "void slow()"),
            event_named(3, 0, 8, 400, 800, "other", "void other()"),
        ];
        let spec = KernelHighlightSpec::parse("top=1,scope=instance")?;
        let resolved = resolve_kernel_highlights(&[spec], &tracks, &events);
        let highlight = resolved
            .response_highlights
            .first()
            .ok_or_else(|| anyhow::anyhow!("expected one highlight"))?;

        assert_eq!(highlight.label, "slow");
        assert_eq!(highlight.full_name, "void slow()");
        assert_eq!(highlight.row_id.as_deref(), Some("kernel:2"));
        assert_eq!(
            resolved.assignments.for_event(
                events
                    .get(1)
                    .ok_or_else(|| anyhow::anyhow!("expected second event"))?
            ),
            Some(highlight.key.clone())
        );
        assert_eq!(
            resolved.assignments.for_event(
                events
                    .get(2)
                    .ok_or_else(|| anyhow::anyhow!("expected third event"))?
            ),
            None
        );
        Ok(())
    }

    #[test]
    fn resolved_tracks_group_streams_under_their_device() -> anyhow::Result<()> {
        let specs = vec![
            TrackSpec::parse("gpu:device=all")?,
            TrackSpec::parse("cuda-streams:device=all,top=4")?,
        ];
        let events = vec![
            event(1, 0, 7, 0, 100),
            event(2, 1, 3, 0, 100),
            event(3, 0, 8, 0, 50),
            event(4, 1, 4, 0, 50),
        ];
        let resolved = resolve_tracks(&specs, &events, false, false)?;
        let rows = resolved
            .response_tracks
            .iter()
            .map(|track| (track.label.as_str(), track.role.as_str()))
            .collect::<Vec<_>>();

        assert_eq!(
            rows,
            vec![
                ("GPU 0", "group"),
                ("busy summary", "summary"),
                ("stream 7", "detail"),
                ("stream 8", "detail"),
                ("GPU 1", "group"),
                ("busy summary", "summary"),
                ("stream 3", "detail"),
                ("stream 4", "detail"),
            ]
        );
        Ok(())
    }

    fn event(rowid: i64, device: i32, stream: i64, start_ns: i64, end_ns: i64) -> TimelineEvent {
        event_named(
            rowid,
            device,
            stream,
            start_ns,
            end_ns,
            "kernel",
            "void kernel()",
        )
    }

    fn event_named(
        rowid: i64,
        device: i32,
        stream: i64,
        start_ns: i64,
        end_ns: i64,
        name: &str,
        full_name: &str,
    ) -> TimelineEvent {
        TimelineEvent {
            row_id: RowId::new(EventKind::Kernel, rowid),
            kind: EventKind::Kernel,
            name: name.to_string(),
            full_name: full_name.to_string(),
            start_ns,
            end_ns,
            device_id: Some(device),
            stream_id: Some(stream),
        }
    }
}
