//! NSys static timeline SVG figure export.
//!
//! This module owns NSys track semantics and evidence extraction. The
//! source-neutral scene shape and SVG writer live in `veloq-vis`.

mod events;
mod fingerprint;
mod highlights;
mod intervals;
mod keys;
mod spec;
#[cfg(test)]
mod tests;
mod tracks;
mod types;

use std::path::Path;

use veloq_core::artifact_dir_for;
use veloq_nsys_data::Trace;
use veloq_vis::{SvgRenderSummary, VizScene, VizTimeWindow, render_svg, write_svg_artifact};

use crate::{NsysQueryError, NsysQueryResult};
use events::{query_gpu_events, query_nvtx_events, query_runtime_events};
use fingerprint::request_fingerprint;
use highlights::resolve_kernel_highlights;
use intervals::build_intervals;
use spec::{KernelHighlightSpec, TrackKind, TrackSpec};
use tracks::{resolve_tracks, scene_metadata};

pub use types::{
    VizLabelPolicyEcho, VizRenderPolicyEcho, VizResolvedHighlight, VizResolvedTrack,
    VizTimelineAuxiliary, VizTimelineFigureRow, VizTimelineRequest, VizTimelineResponse,
    VizUnresolvedHighlight,
};

pub const DEFAULT_TOP_STREAMS: usize = 8;
const VIZ_TIMELINE_COMMAND: &str = "nsys.viz.timeline";
const NSYS_SOURCE_KIND: &str = "nsys";
const NSYS_SOURCE_VERSION: &str = "v3";

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
        &nvtx_events,
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
        metadata: Some(scene_metadata(&resolved.response_tracks)),
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
        density_item_count: summary.density_item_count,
        density_bin_count: summary.density_bin_count,
        density_duration_ns: summary.density_duration_ns,
        omitted_explicit_item_count: summary.omitted_explicit_item_count,
        aggregated: summary.aggregated,
        omitted_track_count: summary.omitted_track_count,
        suppressed_label_count: summary.suppressed_label_count,
        truncated_label_count: summary.truncated_label_count,
    }
}
