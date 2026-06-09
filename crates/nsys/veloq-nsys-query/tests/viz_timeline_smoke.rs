//! Smoke tests for NSys static timeline SVG artifacts.

mod fixture;

use anyhow::Result;
use std::collections::BTreeSet;
use veloq_core::{VeloqDiagnostic, artifact_dir_for, time::TimeWindow};
use veloq_nsys_query::viz_timeline::{VizTimelineRequest, run};

#[test]
fn default_tracks_write_svg_artifact() -> Result<()> {
    let trace = fixture::minimal_gpu()?;
    let resp = run(
        trace.path(),
        VizTimelineRequest {
            time_window: Some(TimeWindow::parse("@100ms-@150ms")?),
            ..Default::default()
        },
    )?;

    assert_eq!(resp.count, 1);
    assert_eq!(resp.total_matched, 1);
    assert_eq!(resp.auxiliary.requested_tracks.len(), 3);
    assert!(
        resp.auxiliary.resolved_tracks.len() >= 2,
        "expected gpu + stream tracks, got {:?}",
        resp.auxiliary.resolved_tracks.len()
    );
    let roles: BTreeSet<&str> = resp
        .auxiliary
        .resolved_tracks
        .iter()
        .map(|track| track.role.as_str())
        .collect();
    assert!(
        roles.contains("group"),
        "expected device group role: {roles:?}"
    );
    assert!(
        roles.contains("summary"),
        "expected GPU busy summary role: {roles:?}"
    );
    assert!(
        roles.contains("detail"),
        "expected CUDA stream detail role: {roles:?}"
    );
    let row = resp
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected one figure row"))?;
    assert!(
        row.key
            .starts_with("figure|timeline|100000000..150000000|req:"),
        "unexpected key: {}",
        row.key
    );
    assert_eq!(row.format, "svg");
    assert_eq!(row.time_window_ns, [100_000_000, 150_000_000]);
    assert!(row.rendered_item_count > 0);
    assert!(row.total_item_count >= row.rendered_item_count);
    assert!(
        row.path.starts_with("figures/nsys/timeline/"),
        "path should be artifact-root relative, got {}",
        row.path
    );
    assert!(row.path.ends_with(".svg"));
    assert!(
        artifact_dir_for(trace.path()).join(&row.path).exists(),
        "svg artifact should exist under artifact root"
    );
    let svg = std::fs::read_to_string(artifact_dir_for(trace.path()).join(&row.path))?;
    assert!(svg.contains("data-track-role=\"group\""));
    assert!(svg.contains("data-track-role=\"summary\""));
    assert!(svg.contains("data-track-role=\"detail\""));
    Ok(())
}

#[test]
fn missing_window_is_structured_error() -> Result<()> {
    let trace = fixture::minimal_gpu()?;
    let err = match run(trace.path(), VizTimelineRequest::default()) {
        Ok(resp) => anyhow::bail!("missing window should fail, got {resp:?}"),
        Err(err) => err,
    };
    assert_eq!(
        err.code().as_str(),
        "nsys.query.viz-timeline-window-required"
    );
    Ok(())
}
