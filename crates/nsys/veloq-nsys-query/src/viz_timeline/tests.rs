use super::default_track_specs;
use super::events::TimelineEvent;
use super::fingerprint::request_fingerprint;
use super::highlights::{HighlightAssignments, resolve_kernel_highlights};
use super::intervals::build_intervals;
use super::keys::{axis, gpu_summary_track_key, stream_track_key};
use super::spec::{HighlightMetric, HighlightScope, KernelHighlightSpec, TrackSpec};
use super::tracks::resolve_tracks;
use crate::{EventKind, NsysQueryError, RowId};
use veloq_core::VeloqDiagnostic;
use veloq_vis::{VizAxis, VizLabelPolicy, VizRenderPolicy, VizRole, VizTrack};

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
    let resolved = resolve_tracks(&specs, &events, false, &[])?;
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
    let stream = resolved
        .response_tracks
        .iter()
        .find(|track| track.track_key == "cuda-stream|dev:0|stream:7")
        .ok_or_else(|| anyhow::anyhow!("expected stream track"))?;
    assert_eq!(
        stream.source_axes,
        vec![axis("device", 0), axis("stream", 7)]
    );
    assert_eq!(
        stream.placement_axes,
        vec![axis("device", 0), axis("stream", 7)]
    );
    assert_eq!(stream.placement_source, "native");
    Ok(())
}

#[test]
fn nvtx_depth_track_uses_dynamic_key_for_routing() -> anyhow::Result<()> {
    let specs = vec![TrackSpec::parse("nvtx:depth=3")?];
    let nvtx_events = vec![
        TimelineEvent {
            row_id: RowId::new(EventKind::Nvtx, 41),
            kind: EventKind::Nvtx,
            name: "depth2".to_string(),
            full_name: "depth2".to_string(),
            start_ns: 0,
            end_ns: 5,
            device_id: None,
            stream_id: None,
            nvtx_depth: Some(2),
        },
        TimelineEvent {
            row_id: RowId::new(EventKind::Nvtx, 42),
            kind: EventKind::Nvtx,
            name: "depth3".to_string(),
            full_name: "depth3".to_string(),
            start_ns: 10,
            end_ns: 20,
            device_id: None,
            stream_id: None,
            nvtx_depth: Some(3),
        },
    ];
    let resolved = resolve_tracks(&specs, &[], false, &nvtx_events)?;
    let track = resolved
        .response_tracks
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected resolved NVTX track"))?;
    assert_eq!(track.track_key, "nvtx|depth:3");
    assert_eq!(track.source_axes, vec![axis("depth", 3)]);
    assert_eq!(track.placement_axes, Vec::<VizAxis>::new());
    assert_eq!(track.placement_source, "fallback");

    let intervals = build_intervals(
        &resolved.tracks,
        &specs,
        &[],
        &[],
        &nvtx_events,
        &HighlightAssignments::default(),
    );
    let interval = intervals
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected routed NVTX interval"))?;
    assert_eq!(interval.track_key, "nvtx|depth:3");
    assert_eq!(interval.row_id.as_deref(), Some("nvtx:42"));
    assert_eq!(intervals.len(), 1);
    Ok(())
}

#[test]
fn device_attributed_nvtx_tracks_group_under_devices() -> anyhow::Result<()> {
    let specs = vec![TrackSpec::parse("nvtx:depth=1")?];
    let nvtx_events = vec![
        TimelineEvent {
            row_id: RowId::new(EventKind::Nvtx, 1),
            kind: EventKind::Nvtx,
            name: "dev0".to_string(),
            full_name: "dev0".to_string(),
            start_ns: 0,
            end_ns: 10,
            device_id: Some(0),
            stream_id: None,
            nvtx_depth: Some(1),
        },
        TimelineEvent {
            row_id: RowId::new(EventKind::Nvtx, 2),
            kind: EventKind::Nvtx,
            name: "dev1".to_string(),
            full_name: "dev1".to_string(),
            start_ns: 0,
            end_ns: 10,
            device_id: Some(1),
            stream_id: None,
            nvtx_depth: Some(1),
        },
    ];
    let resolved = resolve_tracks(&specs, &[], false, &nvtx_events)?;
    let rows = resolved
        .response_tracks
        .iter()
        .map(|track| (track.track_key.as_str(), track.role.as_str()))
        .collect::<Vec<_>>();

    assert_eq!(
        rows,
        vec![
            ("gpu-device|dev:0", "group"),
            ("nvtx|depth:1|dev:0", "annotation"),
            ("gpu-device|dev:1", "group"),
            ("nvtx|depth:1|dev:1", "annotation"),
        ]
    );

    let intervals = build_intervals(
        &resolved.tracks,
        &specs,
        &[],
        &[],
        &nvtx_events,
        &HighlightAssignments::default(),
    );
    let keys = intervals
        .iter()
        .map(|interval| interval.track_key.as_str())
        .collect::<Vec<_>>();

    assert_eq!(keys, vec!["nvtx|depth:1|dev:0", "nvtx|depth:1|dev:1"]);
    let device_track = resolved
        .response_tracks
        .iter()
        .find(|track| track.track_key == "nvtx|depth:1|dev:0")
        .ok_or_else(|| anyhow::anyhow!("expected device-attributed NVTX track"))?;
    assert_eq!(device_track.source_axes, vec![axis("depth", 1)]);
    assert_eq!(device_track.placement_axes, vec![axis("device", 0)]);
    assert_eq!(device_track.placement_source, "nvtx_parent_sidecar");
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
        nvtx_depth: None,
    }
}
