use super::*;
use crate::{
    VizAxis, VizHighlight, VizInterval, VizLabelPolicy, VizRenderPolicy, VizRole, VizScene,
    VizTimeWindow, VizTrack,
};

fn scene() -> VizScene {
    VizScene {
        title: Some("test".to_string()),
        metadata: None,
        time_window: VizTimeWindow {
            start_ns: 100,
            end_ns: 200,
        },
        tracks: vec![VizTrack {
            key: "gpu|0".to_string(),
            label: "GPU 0".to_string(),
            kind: "gpu".to_string(),
            role: VizRole::Detail,
            depth: 0,
            axes: vec![VizAxis {
                name: "device".to_string(),
                value: "0".to_string(),
            }],
        }],
        intervals: vec![VizInterval {
            track_key: "gpu|0".to_string(),
            start_ns: 110,
            end_ns: 150,
            label: Some("kernel<&>".to_string()),
            row_id: Some("kernel:1".to_string()),
            class: Some("kernel".to_string()),
            role: None,
            highlight_key: None,
        }],
        highlights: vec![],
        render_policy: VizRenderPolicy::default(),
        label_policy: VizLabelPolicy::default(),
    }
}

#[test]
fn render_svg_escapes_labels_and_counts_items() -> anyhow::Result<()> {
    let rendered = render_svg(&scene())?;
    assert_eq!(rendered.summary.track_count, 1);
    assert_eq!(rendered.summary.rendered_item_count, 1);
    assert_eq!(rendered.summary.total_item_count, 1);
    assert!(rendered.svg.contains("kernel&lt;&amp;&gt;"));
    assert!(rendered.svg.contains("<svg"));
    assert!(rendered.svg.contains("veloq-viz-metadata"));
    assert!(rendered.svg.contains("data-track-key=\"gpu|0\""));
    Ok(())
}

#[test]
fn render_svg_reports_omitted_tracks_and_item_limit() -> anyhow::Result<()> {
    let mut scene = scene();
    scene.tracks.push(VizTrack {
        key: "gpu|1".to_string(),
        label: "GPU 1".to_string(),
        kind: "gpu".to_string(),
        role: VizRole::Detail,
        depth: 0,
        axes: vec![],
    });
    scene.render_policy.max_tracks = 1;
    scene.render_policy.max_items = 0;
    let rendered = render_svg(&scene)?;
    assert_eq!(rendered.summary.omitted_track_count, 1);
    assert_eq!(rendered.summary.rendered_item_count, 0);
    assert_eq!(rendered.summary.total_item_count, 1);
    assert!(rendered.summary.aggregated);
    Ok(())
}

#[test]
fn render_svg_exposes_roles_and_suppresses_summary_overlay_labels() -> anyhow::Result<()> {
    let mut scene = scene();
    scene.tracks = vec![
        VizTrack {
            key: "gpu-device|dev:0".to_string(),
            label: "GPU 0".to_string(),
            kind: "gpu-device".to_string(),
            role: VizRole::Group,
            depth: 0,
            axes: vec![],
        },
        VizTrack {
            key: "gpu-summary|dev:0".to_string(),
            label: "busy summary".to_string(),
            kind: "gpu-summary".to_string(),
            role: VizRole::Summary,
            depth: 1,
            axes: vec![],
        },
    ];
    scene.intervals = vec![
        VizInterval {
            track_key: "gpu-summary|dev:0".to_string(),
            start_ns: 110,
            end_ns: 130,
            label: Some("summary kernel".to_string()),
            row_id: Some("kernel:1".to_string()),
            class: Some("kernel".to_string()),
            role: None,
            highlight_key: None,
        },
        VizInterval {
            track_key: "gpu-summary|dev:0".to_string(),
            start_ns: 130,
            end_ns: 150,
            label: Some("idle".to_string()),
            row_id: None,
            class: Some("gap".to_string()),
            role: Some(VizRole::Overlay),
            highlight_key: None,
        },
    ];

    let rendered = render_svg(&scene)?;
    assert!(rendered.svg.contains("data-track-role=\"group\""));
    assert!(rendered.svg.contains("data-track-role=\"summary\""));
    assert!(rendered.svg.contains("data-role=\"overlay\""));
    assert!(!rendered.svg.contains(">summary kernel</text>"));
    assert!(!rendered.svg.contains(">idle</text>"));
    assert_eq!(rendered.summary.suppressed_label_count, 2);
    Ok(())
}

#[test]
fn render_svg_packs_overlapping_detail_intervals_into_sublanes() -> anyhow::Result<()> {
    let mut scene = scene();
    scene.time_window = VizTimeWindow {
        start_ns: 0,
        end_ns: 100,
    };
    scene.intervals = vec![
        VizInterval {
            track_key: "gpu|0".to_string(),
            start_ns: 0,
            end_ns: 70,
            label: Some("first".to_string()),
            row_id: Some("kernel:1".to_string()),
            class: Some("kernel".to_string()),
            role: None,
            highlight_key: None,
        },
        VizInterval {
            track_key: "gpu|0".to_string(),
            start_ns: 20,
            end_ns: 90,
            label: Some("second".to_string()),
            row_id: Some("kernel:2".to_string()),
            class: Some("kernel".to_string()),
            role: None,
            highlight_key: None,
        },
    ];

    let rendered = render_svg(&scene)?;

    assert!(rendered.svg.contains(">first</text>"));
    assert!(rendered.svg.contains(">second</text>"));
    assert!(rendered.svg.contains(r#"y="38.0""#));
    assert!(rendered.svg.contains(r#"y="54.0""#));
    Ok(())
}

#[test]
fn render_svg_packs_overlapping_annotation_intervals_into_sublanes() -> anyhow::Result<()> {
    let mut scene = scene();
    scene.tracks = vec![VizTrack {
        key: "nvtx|depth:1".to_string(),
        label: "NVTX".to_string(),
        kind: "nvtx".to_string(),
        role: VizRole::Annotation,
        depth: 0,
        axes: vec![],
    }];
    scene.time_window = VizTimeWindow {
        start_ns: 0,
        end_ns: 100,
    };
    scene.intervals = vec![
        VizInterval {
            track_key: "nvtx|depth:1".to_string(),
            start_ns: 0,
            end_ns: 80,
            label: Some("outer".to_string()),
            row_id: Some("nvtx:1".to_string()),
            class: Some("nvtx".to_string()),
            role: None,
            highlight_key: None,
        },
        VizInterval {
            track_key: "nvtx|depth:1".to_string(),
            start_ns: 20,
            end_ns: 60,
            label: Some("inner".to_string()),
            row_id: Some("nvtx:2".to_string()),
            class: Some("nvtx".to_string()),
            role: None,
            highlight_key: None,
        },
    ];

    let rendered = render_svg(&scene)?;

    assert!(rendered.svg.contains(">outer</text>"));
    assert!(rendered.svg.contains(">inner</text>"));
    assert!(rendered.svg.contains(r#"y="38.0""#));
    assert!(rendered.svg.contains(r#"y="54.0""#));
    Ok(())
}

#[test]
fn render_svg_marks_highlights_and_keeps_separate_legend() -> anyhow::Result<()> {
    let mut scene = scene();
    let Some(interval) = scene.intervals.first_mut() else {
        anyhow::bail!("test scene must contain an interval");
    };
    interval.highlight_key = Some("hot-kernel-1".to_string());
    scene.highlights = vec![VizHighlight {
        key: "hot-kernel-1".to_string(),
        label: "very_long_kernel_name_for_legend".to_string(),
        full_label: "very_long_kernel_name_for_legend<&>".to_string(),
        color: "#f97316".to_string(),
        rank: Some(1),
        scope: Some("name".to_string()),
    }];

    let rendered = render_svg(&scene)?;

    assert_eq!(rendered.summary.rendered_item_count, 1);
    assert!(rendered.svg.contains("data-highlight-key=\"hot-kernel-1\""));
    assert!(rendered.svg.contains("interval-highlighted"));
    assert!(rendered.svg.contains("highlight-legend-item"));
    assert!(rendered.svg.contains("kernel</text>"));
    assert!(rendered.svg.contains("#1 very_long_kernel_name_for_legend"));
    assert!(
        rendered
            .svg
            .contains("very_long_kernel_name_for_legend&lt;&amp;&gt;")
    );
    Ok(())
}
