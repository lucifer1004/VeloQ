use super::*;
use crate::{
    VizAggregation, VizAxis, VizHighlight, VizHighlightScore, VizInterval, VizLabelPolicy,
    VizRenderPolicy, VizRole, VizScene, VizTimeWindow, VizTrack,
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
    assert_eq!(rendered.summary.density_item_count, 0);
    assert_eq!(rendered.summary.density_bin_count, 0);
    assert!(rendered.svg.contains("kernel&lt;&amp;&gt;"));
    assert!(rendered.svg.contains("<svg"));
    assert!(rendered.svg.contains("veloq-viz-metadata"));
    assert!(rendered.svg.contains("data-track-key=\"gpu|0\""));
    Ok(())
}

#[test]
fn render_svg_sanitizes_interval_class_tokens() -> anyhow::Result<()> {
    let mut scene = scene();
    let Some(interval) = scene.intervals.first_mut() else {
        anyhow::bail!("test scene must contain an interval");
    };
    interval.class = Some("kernel\" onclick=\"alert(1)".to_string());

    let rendered = render_svg(&scene)?;

    assert!(rendered.svg.contains("interval-kernel-onclick-alert-1"));
    assert!(
        rendered
            .svg
            .contains("data-class=\"kernel&quot; onclick=&quot;alert(1)\"")
    );
    assert!(!rendered.svg.contains("data-class=\"kernel\" onclick="));
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
    assert_eq!(rendered.summary.omitted_explicit_item_count, 1);
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
fn render_svg_aggregates_subpixel_items_across_full_window() -> anyhow::Result<()> {
    let mut scene = scene();
    scene.time_window = VizTimeWindow {
        start_ns: 0,
        end_ns: 10_000,
    };
    scene.render_policy.width_px = 480;
    scene.render_policy.max_items = 1;
    scene.intervals = (0..10)
        .map(|idx| VizInterval {
            track_key: "gpu|0".to_string(),
            start_ns: 100 + idx * 900,
            end_ns: 101 + idx * 900,
            label: Some(format!("kernel-{idx}")),
            row_id: Some(format!("kernel:{idx}")),
            class: Some("kernel".to_string()),
            role: None,
            highlight_key: None,
        })
        .collect();

    let rendered = render_svg(&scene)?;

    assert_eq!(rendered.summary.total_item_count, 10);
    assert_eq!(rendered.summary.density_item_count, 10);
    assert_eq!(rendered.summary.omitted_explicit_item_count, 0);
    assert!(rendered.summary.density_bin_count > 1);
    assert_eq!(
        rendered.summary.rendered_item_count,
        rendered.summary.density_bin_count
    );
    assert!(rendered.summary.aggregated);
    assert!(rendered.svg.contains("density-bin-kernel"));
    assert!(rendered.svg.contains("data-density-count="));
    assert!(rendered.svg.contains("density:"));
    Ok(())
}

#[test]
fn render_svg_density_bins_do_not_cross_tracks() -> anyhow::Result<()> {
    let mut scene = scene();
    scene.time_window = VizTimeWindow {
        start_ns: 0,
        end_ns: 10_000,
    };
    scene.render_policy.width_px = 480;
    scene.tracks.push(VizTrack {
        key: "gpu|1".to_string(),
        label: "GPU 1".to_string(),
        kind: "gpu".to_string(),
        role: VizRole::Detail,
        depth: 0,
        axes: vec![VizAxis {
            name: "device".to_string(),
            value: "1".to_string(),
        }],
    });
    scene.intervals = vec![
        VizInterval {
            track_key: "gpu|0".to_string(),
            start_ns: 100,
            end_ns: 101,
            label: Some("kernel-0".to_string()),
            row_id: Some("kernel:0".to_string()),
            class: Some("kernel".to_string()),
            role: None,
            highlight_key: None,
        },
        VizInterval {
            track_key: "gpu|1".to_string(),
            start_ns: 100,
            end_ns: 101,
            label: Some("kernel-1".to_string()),
            row_id: Some("kernel:1".to_string()),
            class: Some("kernel".to_string()),
            role: None,
            highlight_key: None,
        },
    ];

    let rendered = render_svg(&scene)?;

    assert_eq!(rendered.summary.density_item_count, 2);
    assert_eq!(rendered.summary.density_bin_count, 2);
    assert_eq!(rendered.summary.rendered_item_count, 2);
    Ok(())
}

#[test]
fn render_svg_density_bins_merge_classes_within_track_bin() -> anyhow::Result<()> {
    let mut scene = scene();
    scene.time_window = VizTimeWindow {
        start_ns: 0,
        end_ns: 10_000,
    };
    scene.render_policy.width_px = 480;
    scene.render_policy.density_bin_px = 48.0;
    scene.intervals = vec![
        VizInterval {
            track_key: "gpu|0".to_string(),
            start_ns: 100,
            end_ns: 101,
            label: Some("kernel-0".to_string()),
            row_id: Some("kernel:0".to_string()),
            class: Some("kernel".to_string()),
            role: None,
            highlight_key: None,
        },
        VizInterval {
            track_key: "gpu|0".to_string(),
            start_ns: 200,
            end_ns: 201,
            label: Some("memcpy-0".to_string()),
            row_id: Some("memcpy:0".to_string()),
            class: Some("memcpy".to_string()),
            role: None,
            highlight_key: None,
        },
        VizInterval {
            track_key: "gpu|0".to_string(),
            start_ns: 300,
            end_ns: 301,
            label: Some("kernel-1".to_string()),
            row_id: Some("kernel:1".to_string()),
            class: Some("kernel".to_string()),
            role: None,
            highlight_key: None,
        },
    ];

    let rendered = render_svg(&scene)?;

    assert_eq!(rendered.summary.density_item_count, 3);
    assert_eq!(rendered.summary.density_bin_count, 1);
    assert_eq!(rendered.summary.rendered_item_count, 1);
    assert_eq!(rendered.svg.matches("class=\"density-bin ").count(), 1);
    assert!(rendered.svg.contains("density-bin-kernel"));
    assert!(rendered.svg.contains("data-density-count=\"3\""));
    assert!(rendered.svg.contains("stroke-dasharray=\"2 3\""));
    assert!(!rendered.svg.contains("density-label"));
    assert!(rendered.svg.contains("dominant kernel"));
    assert!(
        rendered
            .svg
            .contains("breakdown: kernel: 2x/2 ns, memcpy: 1x/1 ns")
    );
    Ok(())
}

#[test]
fn render_svg_summary_density_uses_solid_occupancy_band() -> anyhow::Result<()> {
    let mut scene = scene();
    let Some(track) = scene.tracks.first_mut() else {
        anyhow::bail!("test scene must contain a track");
    };
    track.role = VizRole::Summary;
    track.kind = "gpu-summary".to_string();
    scene.time_window = VizTimeWindow {
        start_ns: 0,
        end_ns: 10_000,
    };
    scene.render_policy.width_px = 480;
    scene.render_policy.density_bin_px = 48.0;
    scene.intervals = (0..3)
        .map(|idx| VizInterval {
            track_key: "gpu|0".to_string(),
            start_ns: 100 + idx * 100,
            end_ns: 101 + idx * 100,
            label: Some(format!("kernel-{idx}")),
            row_id: Some(format!("kernel:{idx}")),
            class: Some("kernel".to_string()),
            role: None,
            highlight_key: None,
        })
        .collect();

    let rendered = render_svg(&scene)?;

    assert_eq!(rendered.summary.density_item_count, 3);
    assert_eq!(rendered.summary.density_bin_count, 1);
    assert!(rendered.svg.contains("class=\"density-heat\""));
    assert!(!rendered.svg.contains("stroke-dasharray=\"2 3\""));
    Ok(())
}

#[test]
fn render_svg_density_bins_preserve_highlight_marker() -> anyhow::Result<()> {
    let mut scene = scene();
    scene.time_window = VizTimeWindow {
        start_ns: 0,
        end_ns: 10_000,
    };
    scene.render_policy.width_px = 480;
    scene.render_policy.density_bin_px = 48.0;
    scene.intervals = vec![
        VizInterval {
            track_key: "gpu|0".to_string(),
            start_ns: 100,
            end_ns: 101,
            label: Some("hot".to_string()),
            row_id: Some("kernel:1".to_string()),
            class: Some("kernel".to_string()),
            role: None,
            highlight_key: Some("hot-kernel".to_string()),
        },
        VizInterval {
            track_key: "gpu|0".to_string(),
            start_ns: 200,
            end_ns: 201,
            label: Some("cold".to_string()),
            row_id: Some("kernel:2".to_string()),
            class: Some("kernel".to_string()),
            role: None,
            highlight_key: None,
        },
    ];
    scene.highlights = vec![VizHighlight {
        key: "hot-kernel".to_string(),
        label: "hot".to_string(),
        full_label: "hot".to_string(),
        color: "#f97316".to_string(),
        rank: Some(1),
        scope: Some("name".to_string()),
        score: None,
    }];

    let rendered = render_svg(&scene)?;

    assert_eq!(rendered.summary.total_item_count, 2);
    assert_eq!(rendered.summary.density_item_count, 2);
    assert_eq!(rendered.summary.density_bin_count, 1);
    assert_eq!(rendered.summary.rendered_item_count, 1);
    assert!(rendered.svg.contains("data-highlight-key=\"hot-kernel\""));
    assert!(rendered.svg.contains("density-highlighted"));
    assert!(!rendered.svg.contains("interval-highlighted"));
    assert!(rendered.svg.contains("density-bin-kernel"));
    Ok(())
}

#[test]
fn render_svg_can_disable_density_and_fall_back_to_item_limit() -> anyhow::Result<()> {
    let mut scene = scene();
    scene.time_window = VizTimeWindow {
        start_ns: 0,
        end_ns: 10_000,
    };
    scene.render_policy.width_px = 480;
    scene.render_policy.max_items = 2;
    scene.render_policy.aggregation = VizAggregation::ItemLimit;
    scene.intervals = (0..5)
        .map(|idx| VizInterval {
            track_key: "gpu|0".to_string(),
            start_ns: 100 + idx * 100,
            end_ns: 101 + idx * 100,
            label: Some(format!("kernel-{idx}")),
            row_id: Some(format!("kernel:{idx}")),
            class: Some("kernel".to_string()),
            role: None,
            highlight_key: None,
        })
        .collect();

    let rendered = render_svg(&scene)?;

    assert_eq!(rendered.summary.total_item_count, 5);
    assert_eq!(rendered.summary.density_item_count, 0);
    assert_eq!(rendered.summary.density_bin_count, 0);
    assert_eq!(rendered.summary.rendered_item_count, 2);
    assert_eq!(rendered.summary.omitted_explicit_item_count, 3);
    assert!(rendered.svg.contains("interval-tick"));
    assert!(!rendered.svg.contains("density-bin"));
    Ok(())
}

#[test]
fn render_svg_raises_density_threshold_to_fit_item_budget() -> anyhow::Result<()> {
    let mut scene = scene();
    scene.time_window = VizTimeWindow {
        start_ns: 0,
        end_ns: 1_000,
    };
    scene.render_policy.width_px = 480;
    scene.render_policy.max_items = 4;
    scene.render_policy.density_bin_px = 120.0;
    scene.intervals = (0..10)
        .map(|idx| {
            let start_ns = idx * 90;
            VizInterval {
                track_key: "gpu|0".to_string(),
                start_ns,
                end_ns: start_ns + 10,
                label: Some(format!("kernel-{idx}")),
                row_id: Some(format!("kernel:{idx}")),
                class: Some("kernel".to_string()),
                role: None,
                highlight_key: None,
            }
        })
        .collect();

    let rendered = render_svg(&scene)?;

    assert_eq!(rendered.summary.total_item_count, 10);
    assert_eq!(rendered.summary.density_item_count, 10);
    assert!(rendered.summary.density_bin_count <= 4);
    assert_eq!(
        rendered.summary.rendered_item_count,
        rendered.summary.density_bin_count
    );
    assert_eq!(rendered.summary.omitted_explicit_item_count, 0);
    assert!(rendered.svg.contains("density-bin-kernel"));
    assert!(!rendered.svg.contains("data-highlight-key=\"hot-kernel\""));
    Ok(())
}

#[test]
fn render_svg_density_bins_can_aggregate_highlights_to_fit_item_budget() -> anyhow::Result<()> {
    let mut scene = scene();
    scene.time_window = VizTimeWindow {
        start_ns: 0,
        end_ns: 1_000,
    };
    scene.render_policy.width_px = 480;
    scene.render_policy.max_items = 4;
    scene.render_policy.density_bin_px = 120.0;
    scene.intervals = (0..10)
        .map(|idx| {
            let start_ns = idx * 90;
            VizInterval {
                track_key: "gpu|0".to_string(),
                start_ns,
                end_ns: start_ns + 10,
                label: Some(format!("kernel-{idx}")),
                row_id: Some(format!("kernel:{idx}")),
                class: Some("kernel".to_string()),
                role: None,
                highlight_key: Some("hot-kernel".to_string()),
            }
        })
        .collect();
    scene.highlights = vec![VizHighlight {
        key: "hot-kernel".to_string(),
        label: "hot".to_string(),
        full_label: "hot".to_string(),
        color: "#f97316".to_string(),
        rank: Some(1),
        scope: Some("name".to_string()),
        score: None,
    }];

    let rendered = render_svg(&scene)?;

    assert_eq!(rendered.summary.total_item_count, 10);
    assert_eq!(rendered.summary.density_item_count, 10);
    assert!(rendered.summary.density_bin_count <= 4);
    assert_eq!(
        rendered.summary.rendered_item_count,
        rendered.summary.density_bin_count
    );
    assert_eq!(rendered.summary.omitted_explicit_item_count, 0);
    assert!(rendered.svg.contains("data-highlight-key=\"hot-kernel\""));
    assert!(rendered.svg.contains("density-highlighted"));
    assert!(!rendered.svg.contains("interval-highlighted"));
    assert!(rendered.svg.contains("density-bin-kernel"));
    Ok(())
}

#[test]
fn render_svg_keeps_annotation_intervals_explicit_for_labels() -> anyhow::Result<()> {
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
        end_ns: 10_000,
    };
    scene.render_policy.width_px = 480;
    scene.render_policy.max_items = 1;
    scene.render_policy.density_bin_px = 48.0;
    scene.intervals = vec![
        VizInterval {
            track_key: "nvtx|depth:1".to_string(),
            start_ns: 100,
            end_ns: 5_000,
            label: Some("outer".to_string()),
            row_id: Some("nvtx:1".to_string()),
            class: Some("nvtx".to_string()),
            role: None,
            highlight_key: None,
        },
        VizInterval {
            track_key: "nvtx|depth:1".to_string(),
            start_ns: 5_100,
            end_ns: 9_000,
            label: Some("inner".to_string()),
            row_id: Some("nvtx:2".to_string()),
            class: Some("nvtx".to_string()),
            role: None,
            highlight_key: None,
        },
    ];

    let rendered = render_svg(&scene)?;

    assert_eq!(rendered.summary.total_item_count, 2);
    assert_eq!(rendered.summary.density_item_count, 0);
    assert_eq!(rendered.summary.density_bin_count, 0);
    assert_eq!(rendered.summary.rendered_item_count, 2);
    assert!(rendered.svg.contains(">outer</text>"));
    assert!(rendered.svg.contains(">inner</text>"));
    assert!(!rendered.svg.contains("density-bin-nvtx"));
    Ok(())
}

#[test]
fn render_svg_keeps_sub_label_width_annotation_bars() -> anyhow::Result<()> {
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
        end_ns: 10_000,
    };
    scene.render_policy.width_px = 480;
    scene.label_policy.min_label_px = 160.0;
    scene.intervals = vec![
        VizInterval {
            track_key: "nvtx|depth:1".to_string(),
            start_ns: 100,
            end_ns: 5_000,
            label: Some("phase".to_string()),
            row_id: Some("nvtx:1".to_string()),
            class: Some("nvtx".to_string()),
            role: None,
            highlight_key: None,
        },
        VizInterval {
            track_key: "nvtx|depth:1".to_string(),
            start_ns: 5_100,
            end_ns: 9_000,
            label: Some("short-label".to_string()),
            row_id: Some("nvtx:2".to_string()),
            class: Some("nvtx".to_string()),
            role: None,
            highlight_key: None,
        },
    ];

    let rendered = render_svg(&scene)?;

    assert_eq!(rendered.summary.total_item_count, 2);
    assert_eq!(rendered.summary.rendered_item_count, 2);
    assert_eq!(rendered.summary.omitted_explicit_item_count, 0);
    assert!(rendered.svg.contains("nvtx:1"));
    assert!(rendered.svg.contains("nvtx:2"));
    assert!(!rendered.svg.contains(">phase</text>"));
    assert!(!rendered.svg.contains(">short-label</text>"));
    assert_eq!(rendered.summary.suppressed_label_count, 2);
    assert_eq!(rendered.summary.truncated_label_count, 0);
    assert!(
        !rendered
            .svg
            .contains("interval-tick interval-role-annotation")
    );
    Ok(())
}

#[test]
fn render_svg_uses_available_width_for_compact_annotation_prefixes() -> anyhow::Result<()> {
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
        end_ns: 10_000,
    };
    scene.render_policy.width_px = 480;
    scene.label_policy.min_label_px = 1.0;
    scene.intervals = vec![VizInterval {
        track_key: "nvtx|depth:1".to_string(),
        start_ns: 0,
        end_ns: 2_000,
        label: Some("execute_context".to_string()),
        row_id: Some("nvtx:1".to_string()),
        class: Some("nvtx".to_string()),
        role: None,
        highlight_key: None,
    }];

    let rendered = render_svg(&scene)?;

    assert!(rendered.svg.contains(">execute_context</text>"));
    assert!(!rendered.svg.contains(">execute_c</text>"));
    assert_eq!(rendered.summary.suppressed_label_count, 0);
    assert_eq!(rendered.summary.truncated_label_count, 1);
    Ok(())
}

#[test]
fn render_svg_honors_min_label_width_policy() -> anyhow::Result<()> {
    let mut scene = scene();
    scene.label_policy.min_label_px = 10_000.0;
    scene.intervals = vec![VizInterval {
        track_key: "gpu|0".to_string(),
        start_ns: 110,
        end_ns: 150,
        label: Some("kernel".to_string()),
        row_id: Some("kernel:1".to_string()),
        class: Some("kernel".to_string()),
        role: None,
        highlight_key: None,
    }];

    let rendered = render_svg(&scene)?;

    assert!(rendered.svg.contains("kernel:1"));
    assert!(!rendered.svg.contains(r#"<text class="interval-label""#));
    assert_eq!(rendered.summary.suppressed_label_count, 1);
    Ok(())
}

#[test]
fn render_svg_uses_available_space_for_long_interval_labels() -> anyhow::Result<()> {
    let mut scene = scene();
    scene.time_window = VizTimeWindow {
        start_ns: 0,
        end_ns: 10_000,
    };
    scene.render_policy.width_px = 480;
    scene.label_policy.min_label_px = 1.0;
    scene.intervals = vec![VizInterval {
        track_key: "gpu|0".to_string(),
        start_ns: 0,
        end_ns: 10_000,
        label: Some("very_long_kernel_name".to_string()),
        row_id: Some("kernel:1".to_string()),
        class: Some("kernel".to_string()),
        role: None,
        highlight_key: None,
    }];

    let rendered = render_svg(&scene)?;

    assert!(rendered.svg.contains(">very_long_kernel_name</text>"));
    assert_eq!(rendered.summary.suppressed_label_count, 0);
    assert_eq!(rendered.summary.truncated_label_count, 0);
    Ok(())
}

#[test]
fn render_svg_omits_subpixel_annotation_ticks() -> anyhow::Result<()> {
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
        end_ns: 10_000,
    };
    scene.render_policy.width_px = 480;
    scene.intervals = vec![VizInterval {
        track_key: "nvtx|depth:1".to_string(),
        start_ns: 5_100,
        end_ns: 5_101,
        label: Some("tick".to_string()),
        row_id: Some("nvtx:2".to_string()),
        class: Some("nvtx".to_string()),
        role: None,
        highlight_key: None,
    }];

    let rendered = render_svg(&scene)?;

    assert_eq!(rendered.summary.total_item_count, 1);
    assert_eq!(rendered.summary.rendered_item_count, 0);
    assert_eq!(rendered.summary.omitted_explicit_item_count, 1);
    assert!(!rendered.svg.contains("nvtx:2"));
    assert!(
        !rendered
            .svg
            .contains("interval-tick interval-role-annotation")
    );
    Ok(())
}

#[test]
fn render_svg_samples_limited_explicit_items_across_window() -> anyhow::Result<()> {
    let mut scene = scene();
    scene.time_window = VizTimeWindow {
        start_ns: 0,
        end_ns: 1_000,
    };
    scene.render_policy.width_px = 480;
    scene.render_policy.max_items = 3;
    scene.render_policy.aggregation = VizAggregation::ItemLimit;
    scene.intervals = (0..7)
        .map(|idx| {
            let start_ns = idx * 100;
            VizInterval {
                track_key: "gpu|0".to_string(),
                start_ns,
                end_ns: start_ns + 50,
                label: Some(format!("kernel-{idx}")),
                row_id: Some(format!("kernel:{idx}")),
                class: Some("kernel".to_string()),
                role: None,
                highlight_key: None,
            }
        })
        .collect();

    let rendered = render_svg(&scene)?;

    assert_eq!(rendered.summary.total_item_count, 7);
    assert_eq!(rendered.summary.rendered_item_count, 3);
    assert_eq!(rendered.summary.omitted_explicit_item_count, 4);
    assert!(rendered.svg.contains("kernel:0"));
    assert!(rendered.svg.contains("kernel:3"));
    assert!(rendered.svg.contains("kernel:6"));
    assert!(!rendered.svg.contains("kernel:1"));
    assert!(!rendered.svg.contains("kernel:2"));
    assert!(!rendered.svg.contains("kernel:4"));
    assert!(!rendered.svg.contains("kernel:5"));
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
        score: Some(VizHighlightScore {
            metric: "total_duration_ns".to_string(),
            value: 1_500_000,
            total: Some(5_000_000),
        }),
    }];

    let rendered = render_svg(&scene)?;

    assert_eq!(rendered.summary.rendered_item_count, 1);
    assert!(rendered.svg.contains("data-highlight-key=\"hot-kernel-1\""));
    assert!(rendered.svg.contains("interval-highlighted"));
    assert!(rendered.svg.contains("highlight-legend-item"));
    assert!(rendered.svg.contains("kernel</text>"));
    assert!(rendered.svg.contains("#1 very_long_kernel_name_for_legend"));
    assert!(rendered.svg.contains("total 1.5 ms (30.0%)"));
    assert!(
        rendered
            .svg
            .contains("very_long_kernel_name_for_legend&lt;&amp;&gt;")
    );
    Ok(())
}
