//! NSys static timeline SVG figure export.
//!
//! This module owns NSys track semantics and evidence extraction. The
//! source-neutral scene shape and SVG writer live in `veloq-core`.

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
use veloq_core::time::TimeWindow;
use veloq_core::{
    SvgRenderSummary, VizAxis, VizInterval, VizLabelPolicy, VizRenderPolicy, VizRole, VizScene,
    VizTimeWindow, VizTrack, artifact_dir_for, render_svg, write_svg_artifact,
};
use veloq_nsys_data::Trace;
use veloq_query::sql::SqlFragment;

pub const DEFAULT_TOP_STREAMS: usize = 8;

#[derive(Debug, Clone, Default)]
pub struct VizTimelineRequest {
    pub time_window: Option<TimeWindow>,
    pub tracks: Vec<String>,
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
    let intervals = build_intervals(
        &resolved.tracks,
        &parsed_tracks,
        &gpu_events,
        &api_events,
        &nvtx_events,
    );
    let scene = VizScene {
        title: Some("NSys timeline".to_string()),
        time_window,
        tracks: resolved.tracks.clone(),
        intervals,
        render_policy: req.render_policy.clone(),
        label_policy: req.label_policy.clone(),
    };
    let rendered =
        render_svg(&scene).map_err(|source| NsysQueryError::VizTimelineArtifact { source })?;

    let fingerprint = request_fingerprint(
        start_ns,
        end_ns,
        &requested_tracks,
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

#[derive(Debug, Clone)]
struct TimelineEvent {
    row_id: RowId,
    kind: EventKind,
    name: String,
    start_ns: i64,
    end_ns: i64,
    device_id: Option<i32>,
    stream_id: Option<i64>,
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
) -> Vec<VizInterval> {
    let track_keys: BTreeSet<&str> = tracks.iter().map(|track| track.key.as_str()).collect();
    let mut out = Vec::new();
    for event in gpu_events {
        if let Some(device) = event.device_id {
            let key = gpu_summary_track_key(device);
            if track_keys.contains(key.as_str()) {
                out.push(interval_for_event(key, event));
            }
        }
        if let (Some(device), Some(stream)) = (event.device_id, event.stream_id) {
            let key = stream_track_key(device, stream);
            if track_keys.contains(key.as_str()) {
                out.push(interval_for_event(key, event));
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
                .map(|event| interval_for_event("cuda-api".to_string(), event)),
        );
    }
    if track_keys.contains("nvtx|depth:1") {
        out.extend(
            nvtx_events
                .iter()
                .map(|event| interval_for_event("nvtx|depth:1".to_string(), event)),
        );
    }
    out.sort_by(|a, b| {
        a.start_ns
            .cmp(&b.start_ns)
            .then_with(|| a.track_key.cmp(&b.track_key))
    });
    out
}

fn interval_for_event(track_key: String, event: &TimelineEvent) -> VizInterval {
    VizInterval {
        track_key,
        start_ns: event.start_ns,
        end_ns: event.end_ns,
        label: Some(event.name.clone()),
        row_id: Some(event.row_id.to_string()),
        class: Some(event.kind.as_str().to_string()),
        role: None,
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
            {name_expr} AS name,
            t.start AS start_ns,
            COALESCE(t."end", t.start) AS end_ns,
            {device_expr} AS device_id,
            {stream_expr} AS stream_id
        FROM nsight.{table} t {joins}
        {where_clause}
        "#,
        label = sem.label(),
        name_expr = sem.display_name_expr(),
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
    render_policy: &VizRenderPolicy,
    label_policy: &VizLabelPolicy,
) -> String {
    let mut hash = Fnv1a64::new();
    hash.push(&start_ns.to_string());
    hash.push(&end_ns.to_string());
    for track in tracks {
        hash.push(track);
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
            request_fingerprint(1, 2, &tracks, &render, &label),
            request_fingerprint(1, 2, &tracks, &render, &label)
        );
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
        TimelineEvent {
            row_id: RowId::new(EventKind::Kernel, rowid),
            kind: EventKind::Kernel,
            name: "kernel".to_string(),
            start_ns,
            end_ns,
            device_id: Some(device),
            stream_id: Some(stream),
        }
    }
}
