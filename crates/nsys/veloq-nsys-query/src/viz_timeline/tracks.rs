use std::collections::{BTreeMap, BTreeSet};

use veloq_vis::{VizAxis, VizRole, VizSceneMetadata, VizTrack, VizTrackMetadata};

use crate::{NsysQueryError, NsysQueryResult};

use super::events::TimelineEvent;
use super::keys::{
    axis, axis_i32, axis_i64, cuda_api_track_key, device_group_track_key, gpu_summary_track_key,
    nvtx_track_key, stream_track_key,
};
use super::spec::{DeviceSelector, TrackKind, TrackSpec};
use super::types::VizResolvedTrack;
use super::{DEFAULT_TOP_STREAMS, NSYS_SOURCE_KIND, NSYS_SOURCE_VERSION, VIZ_TIMELINE_COMMAND};

#[derive(Debug)]
pub(super) struct ResolvedTracks {
    pub(super) tracks: Vec<VizTrack>,
    pub(super) response_tracks: Vec<VizResolvedTrack>,
}

pub(super) fn resolve_tracks(
    specs: &[TrackSpec],
    gpu_events: &[TimelineEvent],
    api_events: &[TimelineEvent],
    nvtx_events: &[TimelineEvent],
    trace_cuda_scopes: &BTreeSet<(i64, i32)>,
) -> NsysQueryResult<ResolvedTracks> {
    let mut tracks = Vec::new();
    let mut response_tracks = Vec::new();
    let mut seen = BTreeSet::new();
    for spec in specs {
        match spec.kind {
            TrackKind::Gpu => {
                for (process, device) in scopes_for(spec, gpu_events, trace_cuda_scopes)? {
                    push_device_group(
                        &mut tracks,
                        &mut response_tracks,
                        &mut seen,
                        process,
                        device,
                    );
                    push_track(
                        &mut tracks,
                        &mut response_tracks,
                        &mut seen,
                        TrackDef::new(
                            gpu_summary_track_key(process, device),
                            "busy summary".to_string(),
                            "gpu-summary",
                            vec![axis("process", process), axis("device", device)],
                        )
                        .depth(1)
                        .role(VizRole::Summary),
                    );
                }
            }
            TrackKind::CudaStreams => {
                for (process, device, stream) in
                    top_streams_for(spec, gpu_events, trace_cuda_scopes)?
                {
                    push_device_group(
                        &mut tracks,
                        &mut response_tracks,
                        &mut seen,
                        process,
                        device,
                    );
                    push_track(
                        &mut tracks,
                        &mut response_tracks,
                        &mut seen,
                        TrackDef::new(
                            stream_track_key(process, device, stream),
                            format!("stream {stream}"),
                            "cuda-stream",
                            vec![
                                axis("process", process),
                                axis("device", device),
                                axis("stream", stream),
                            ],
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
                let scopes = scopes_for(spec, gpu_events, trace_cuda_scopes)?;
                let Some(&(process, _)) = scopes.first() else {
                    continue;
                };
                push_device_group(
                    &mut tracks,
                    &mut response_tracks,
                    &mut seen,
                    process,
                    device,
                );
                push_track(
                    &mut tracks,
                    &mut response_tracks,
                    &mut seen,
                    TrackDef::new(
                        stream_track_key(process, device, stream),
                        format!("stream {stream}"),
                        "cuda-stream",
                        vec![
                            axis("process", process),
                            axis("device", device),
                            axis("stream", stream),
                        ],
                    )
                    .depth(1)
                    .role(VizRole::Detail),
                );
            }
            TrackKind::CudaApi => {
                for process in api_events
                    .iter()
                    .filter_map(|event| event.process_id)
                    .filter(|process| spec.process.is_none_or(|wanted| wanted == *process))
                    .collect::<BTreeSet<_>>()
                {
                    push_track(
                        &mut tracks,
                        &mut response_tracks,
                        &mut seen,
                        TrackDef::new(
                            cuda_api_track_key(process),
                            format!("PID {process} / CUDA API"),
                            "cuda-api",
                            vec![axis("process", process)],
                        )
                        .placement_source("fallback")
                        .role(VizRole::Annotation),
                    );
                }
            }
            TrackKind::Nvtx => {
                if !nvtx_events.is_empty() {
                    let depth = spec.depth.unwrap_or(1);
                    let depth_events = nvtx_events
                        .iter()
                        .filter(|event| event.nvtx_depth == Some(depth))
                        .collect::<Vec<_>>();
                    for (process, device) in scopes_for_nvtx(spec, &depth_events) {
                        push_device_group(
                            &mut tracks,
                            &mut response_tracks,
                            &mut seen,
                            process,
                            device,
                        );
                        push_track(
                            &mut tracks,
                            &mut response_tracks,
                            &mut seen,
                            TrackDef::new(
                                nvtx_track_key(depth, process, Some(device)),
                                "NVTX".to_string(),
                                "nvtx",
                                vec![
                                    axis("process", process),
                                    axis("device", device),
                                    axis("depth", depth),
                                ],
                            )
                            .source_axes(vec![axis("process", process), axis("depth", depth)])
                            .placement_axes(vec![axis("process", process), axis("device", device)])
                            .placement_source("nvtx_parent_sidecar")
                            .depth(1)
                            .role(VizRole::Annotation),
                        );
                    }
                    for process in depth_events
                        .iter()
                        .filter(|event| event.device_id.is_none())
                        .filter_map(|event| event.process_id)
                        .filter(|process| spec.process.is_none_or(|wanted| wanted == *process))
                        .collect::<BTreeSet<_>>()
                    {
                        push_track(
                            &mut tracks,
                            &mut response_tracks,
                            &mut seen,
                            TrackDef::new(
                                nvtx_track_key(depth, process, None),
                                "NVTX".to_string(),
                                "nvtx",
                                vec![axis("process", process), axis("depth", depth)],
                            )
                            .placement_axes(vec![])
                            .placement_source("fallback")
                            .role(VizRole::Annotation),
                        );
                    }
                }
            }
            TrackKind::GapsOverlay => {
                // Overlays do not create tracks, but their selectors are
                // still scope-bearing and must receive the same trace-wide
                // logical-device ambiguity validation as GPU tracks.
                scopes_for(spec, gpu_events, trace_cuda_scopes)?;
            }
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
    source_axes: Vec<VizAxis>,
    placement_axes: Vec<VizAxis>,
    placement_source: String,
    depth: usize,
    role: VizRole,
}

impl TrackDef {
    fn new(key: String, label: String, kind: impl Into<String>, axes: Vec<VizAxis>) -> Self {
        let source_axes = axes.clone();
        let placement_axes = axes.clone();
        Self {
            key,
            label,
            kind: kind.into(),
            axes,
            source_axes,
            placement_axes,
            placement_source: "native".to_string(),
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

    fn source_axes(mut self, axes: Vec<VizAxis>) -> Self {
        self.source_axes = axes;
        self
    }

    fn placement_axes(mut self, axes: Vec<VizAxis>) -> Self {
        self.placement_axes = axes;
        self
    }

    fn placement_source(mut self, source: impl Into<String>) -> Self {
        self.placement_source = source.into();
        self
    }
}

fn push_device_group(
    tracks: &mut Vec<VizTrack>,
    response_tracks: &mut Vec<VizResolvedTrack>,
    seen: &mut BTreeSet<String>,
    process: i64,
    device: i32,
) {
    push_track(
        tracks,
        response_tracks,
        seen,
        TrackDef::new(
            device_group_track_key(process, device),
            format!("PID {process} / GPU {device}"),
            "gpu-device",
            vec![axis("process", process), axis("device", device)],
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
        source_axes: def.source_axes,
        placement_axes: def.placement_axes,
        placement_source: def.placement_source,
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
) -> (u8, i64, i32, u8, usize) {
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

fn track_order_key(role: VizRole, axes: &[VizAxis], insertion: usize) -> (u8, i64, i32, u8, usize) {
    let process = axis_i64(axes, "process").unwrap_or(0);
    let device = axis_i32(axes, "device");
    if let Some(device) = device
        && matches!(
            role,
            VizRole::Group | VizRole::Summary | VizRole::Detail | VizRole::Annotation
        )
    {
        return (0, process, device, visual_role_rank(role), insertion);
    }
    (1, process, 0, visual_role_rank(role), insertion)
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

fn scopes_for(
    spec: &TrackSpec,
    events: &[TimelineEvent],
    trace_cuda_scopes: &BTreeSet<(i64, i32)>,
) -> NsysQueryResult<Vec<(i64, i32)>> {
    if let (Some(process), Some(DeviceSelector::One(device))) = (spec.process, &spec.device) {
        return Ok(vec![(process, *device)]);
    }
    if spec.process.is_none()
        && let Some(DeviceSelector::One(device)) = spec.device
        && trace_cuda_scopes
            .iter()
            .filter(|(_, candidate_device)| *candidate_device == device)
            .map(|(pid, _)| *pid)
            .collect::<BTreeSet<_>>()
            .len()
            > 1
    {
        return Err(NsysQueryError::VizTimelineProcessRequired);
    }
    let scopes = events
        .iter()
        .filter_map(|event| Some((event.process_id?, event.device_id?)))
        .filter(|(process, device)| {
            spec.process.is_none_or(|wanted| wanted == *process)
                && match spec.device {
                    Some(DeviceSelector::One(wanted)) => wanted == *device,
                    Some(DeviceSelector::All) | None => true,
                }
        })
        .collect::<BTreeSet<_>>();
    Ok(scopes.into_iter().collect())
}

fn scopes_for_nvtx(spec: &TrackSpec, events: &[&TimelineEvent]) -> Vec<(i64, i32)> {
    events
        .iter()
        .filter_map(|event| Some((event.process_id?, event.device_id?)))
        .filter(|(process, _)| spec.process.is_none_or(|wanted| wanted == *process))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(super) fn scene_metadata(tracks: &[VizResolvedTrack]) -> VizSceneMetadata {
    VizSceneMetadata {
        command: VIZ_TIMELINE_COMMAND.to_string(),
        source_kind: NSYS_SOURCE_KIND.to_string(),
        source_version: NSYS_SOURCE_VERSION.to_string(),
        tracks: tracks.iter().map(track_metadata).collect(),
    }
}

fn track_metadata(track: &VizResolvedTrack) -> VizTrackMetadata {
    VizTrackMetadata {
        track_key: track.track_key.clone(),
        label: track.label.clone(),
        kind: track.kind.clone(),
        role: track.role.clone(),
        axes: track.axes.clone(),
        source_axes: track.source_axes.clone(),
        placement_axes: track.placement_axes.clone(),
        placement_source: track.placement_source.clone(),
    }
}

fn top_streams_for(
    spec: &TrackSpec,
    events: &[TimelineEvent],
    trace_cuda_scopes: &BTreeSet<(i64, i32)>,
) -> NsysQueryResult<Vec<(i64, i32, i64)>> {
    let top = spec.top.unwrap_or(DEFAULT_TOP_STREAMS);
    let scopes = scopes_for(spec, events, trace_cuda_scopes)?;
    let scope_set = scopes.into_iter().collect::<BTreeSet<_>>();
    let mut busy: BTreeMap<(i64, i32, i64), i64> = BTreeMap::new();
    for event in events {
        let (Some(process), Some(device), Some(stream)) =
            (event.process_id, event.device_id, event.stream_id)
        else {
            continue;
        };
        if !scope_set.contains(&(process, device)) {
            continue;
        }
        let duration = event.end_ns.saturating_sub(event.start_ns);
        *busy.entry((process, device, stream)).or_insert(0) += duration;
    }
    let mut streams: Vec<((i64, i32, i64), i64)> = busy.into_iter().collect();
    streams.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    Ok(streams
        .into_iter()
        .take(top)
        .map(|((process, device, stream), _)| (process, device, stream))
        .collect())
}
