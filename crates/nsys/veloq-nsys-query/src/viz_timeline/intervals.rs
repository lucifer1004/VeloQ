use std::collections::{BTreeMap, BTreeSet};

use veloq_vis::{VizInterval, VizRole, VizTrack};

use super::events::TimelineEvent;
use super::highlights::HighlightAssignments;
use super::keys::{axis_i32, axis_usize, gpu_summary_track_key, stream_track_key};
use super::spec::{TrackKind, TrackSpec};

pub(super) fn build_intervals(
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
    let nvtx_tracks = tracks
        .iter()
        .filter(|track| track.kind == "nvtx")
        .map(|track| {
            (
                track.key.clone(),
                axis_i32(&track.axes, "device"),
                axis_usize(&track.axes, "depth"),
            )
        })
        .collect::<Vec<_>>();
    for event in nvtx_events {
        for (track_key, track_device, track_depth) in &nvtx_tracks {
            if *track_device == event.device_id && *track_depth == event.nvtx_depth {
                out.push(interval_for_event(track_key.clone(), event, highlights));
            }
        }
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
