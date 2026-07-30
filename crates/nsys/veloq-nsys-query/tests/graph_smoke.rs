//! Integration tests for `EventKind::Graph` plumbing.
//!
//! Covers the `--cuda-graph-trace=graph` shape (kernels-inside-graphs
//! are rolled up into graph_trace rows; the kernel table holds only
//! eager-mode kernels). The `with_graph_trace` fixture mirrors the
//! shape of a `--cuda-graph-trace=graph` capture.

mod fixture;

use anyhow::Result;
use veloq_core::time::TimeWindow;
use veloq_nsys_query::{EventKind, KindFilter, RowId};

#[test]
fn stats_rolls_up_graphs_by_graph_id() -> Result<()> {
    let trace = fixture::with_graph_trace()?;
    let req = veloq_nsys_query::stats::StatsRequest {
        kinds: KindFilter::Only(vec![EventKind::Graph]),
        ..Default::default()
    };
    let r = veloq_nsys_query::stats::run(trace.path(), req)?;
    // Three launches of one captured graph → one row at graph:42 with count=3.
    assert_eq!(r.count, 1, "single graphId rolls up to one row");
    let row = r
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected one stats row"))?;
    assert_eq!(row.name.as_deref(), Some("graph:42"));
    assert_eq!(row.kind, "graph");
    assert_eq!(row.count, 3);
    assert_eq!(row.total_ns, 30_000_000, "3 × 10ms launches");
    Ok(())
}

#[test]
fn stats_default_kinds_include_graph() -> Result<()> {
    let trace = fixture::with_graph_trace()?;
    let r = veloq_nsys_query::stats::run(trace.path(), Default::default())?;
    // 1 eager kernel + 1 graph row (3 launches collapse on
    // graph:<graphId>) + 1 NVTX range + 1 runtime call.
    assert_eq!(r.total_matched, 4);
    let graph = r
        .rows
        .iter()
        .find(|r| r.kind == "graph")
        .ok_or_else(|| anyhow::anyhow!("expected a graph row in default stats"))?;
    assert_eq!(graph.name.as_deref(), Some("graph:42"));
    let nvtx = r
        .rows
        .iter()
        .find(|r| r.kind == "nvtx")
        .ok_or_else(|| anyhow::anyhow!("expected an nvtx row in default stats"))?;
    assert_eq!(nvtx.name.as_deref(), Some("frame"));
    let runtime = r
        .rows
        .iter()
        .find(|r| r.kind == "runtime")
        .ok_or_else(|| anyhow::anyhow!("expected a runtime row in default stats (post-A1)"))?;
    assert_eq!(runtime.name.as_deref(), Some("cudaGraphLaunch_v10000"));
    Ok(())
}

#[test]
fn search_returns_graph_hits_by_name_pattern() -> Result<()> {
    let trace = fixture::with_graph_trace()?;
    let req = veloq_nsys_query::search::SearchRequest {
        kinds: KindFilter::Only(vec![EventKind::Graph]),
        name_glob: Some("graph:*".to_string()),
        limit: 10,
        ..Default::default()
    };
    let r = veloq_nsys_query::search::run(trace.path(), req)?;
    // Three launch rows match the glob.
    assert_eq!(r.rows.len(), 3);
    for h in &r.rows {
        let b = h.base();
        assert_eq!(b.row_id.kind, EventKind::Graph);
        assert_eq!(b.name, "graph:42");
    }
    Ok(())
}

#[test]
fn correlate_from_launch_returns_graph_row() -> Result<()> {
    let trace = fixture::with_graph_trace()?;
    // First runtime row in the fixture is the cudaGraphLaunch with
    // correlationId 7100 → graph_trace rowid 1 shares that correlation.
    let launch = RowId::new(EventKind::Runtime, 1);
    let r = veloq_nsys_query::correlate::run(trace.path(), &[launch])?;
    let res = r
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected one result"))?;
    assert!(res.correlation_found, "launch should correlate");
    assert_eq!(
        res.auxiliary.graph_events.len(),
        1,
        "one graph for this launch"
    );
    let g = res
        .auxiliary
        .graph_events
        .first()
        .ok_or_else(|| anyhow::anyhow!("graph_events should be non-empty"))?;
    let gb = g.base();
    assert_eq!(gb.row_id.kind, EventKind::Graph);
    assert_eq!(gb.name, "graph:42");
    Ok(())
}

#[test]
fn correlate_from_graph_returns_launching_runtime() -> Result<()> {
    let trace = fixture::with_graph_trace()?;
    // Inspect the other direction: from the graph_trace row find its
    // launcher. First graph_trace rowid = 1.
    let graph = RowId::new(EventKind::Graph, 1);
    let r = veloq_nsys_query::correlate::run(trace.path(), &[graph])?;
    let res = r
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected one result"))?;
    assert!(res.correlation_found);
    assert_eq!(res.auxiliary.cpu_events.len(), 1, "launching runtime call");
    let cpu = res
        .auxiliary
        .cpu_events
        .first()
        .ok_or_else(|| anyhow::anyhow!("cpu_events should be non-empty"))?;
    assert_eq!(cpu.base().name, "cudaGraphLaunch_v10000");
    Ok(())
}

#[test]
fn inspect_graph_row_returns_graph_details() -> Result<()> {
    let trace = fixture::with_graph_trace()?;
    let id = RowId::new(EventKind::Graph, 1);
    let r = veloq_nsys_query::inspect::run(trace.path(), &[id])?;
    let first = r
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected one event"))?;
    match first {
        veloq_nsys_query::inspect::EventDetails::Graph(g) => {
            assert_eq!(g.graph_id, 42);
            assert_eq!(g.graph_exec_id, 43);
            assert_eq!(g.device_id, 0);
            assert_eq!(g.stream_id, 23);
            assert_eq!(g.duration_ns, 10_000_000);
            assert_eq!(g.correlation_id, Some(7100));
        }
        other => anyhow::bail!("expected EventDetails::Graph, got {other:?}"),
    }
    Ok(())
}

#[test]
fn timeline_graph_ns_separated_from_kernel_ns() -> Result<()> {
    let trace = fixture::with_graph_trace()?;
    let req = veloq_nsys_query::timeline::TimelineRequest {
        interval_ns: 1_000_000, // 1ms buckets — same as the default in other tests
        kinds: KindFilter::Only(vec![EventKind::Kernel, EventKind::Graph]),
        ..Default::default()
    };
    let r = veloq_nsys_query::timeline::run(trace.path(), req)?;
    // At least one bucket has non-zero graph_ns.
    let any_graph_bucket = r.rows.iter().any(|b| b.graph_ns > 0);
    assert!(any_graph_bucket, "graph_ns must be reported per bucket");
    // Sum across buckets: graphs total 30ms, eager kernel total 1ms.
    let graph_sum: i64 = r.rows.iter().map(|b| b.graph_ns).sum();
    let kernel_sum: i64 = r.rows.iter().map(|b| b.kernel_ns).sum();
    assert_eq!(graph_sum, 30_000_000);
    assert_eq!(kernel_sum, 1_000_000);
    Ok(())
}

/// Regression: `veloq timeline --type all --nvtx <pattern>` previously
/// crashed on traces that contained graph_trace rows. The GPU-busy set
/// includes `Graph`, NVTX attribution skips it, but `per_kind_select`
/// errored under `nvtx_scope.is_attributed()` for the `Graph` arm.
/// Now it emits `WHERE FALSE` so the subquery contributes zero rows.
#[test]
fn timeline_nvtx_skips_graph_without_crashing() -> Result<()> {
    let trace = fixture::with_graph_trace()?;
    let req = veloq_nsys_query::timeline::TimelineRequest {
        interval_ns: 1_000_000,
        kinds: KindFilter::All,
        nvtx: Some("frame".to_string()),
        ..Default::default()
    };
    let r = veloq_nsys_query::timeline::run(trace.path(), req)?;
    // The eager kernel (1ms at 200ms–201ms) is outside the NVTX range
    // (90ms..210ms ✓ — it's inside) and attributed via its runtime call.
    // graph_trace rows are NVTX-opaque, so graph_ns is zero in every
    // bucket even though the launches sit inside the NVTX range.
    let total_graph: i64 = r.rows.iter().map(|b| b.graph_ns).sum();
    assert_eq!(total_graph, 0, "graph_ns must be zero under --nvtx");
    Ok(())
}

#[test]
fn capability_bit_set_when_graph_trace_present() -> Result<()> {
    let trace = fixture::with_graph_trace()?;
    let s = veloq_nsys_query::summary::run(trace.path())?;
    let caps = s
        .auxiliary
        .capabilities
        .ok_or_else(|| anyhow::anyhow!("summary missing capabilities"))?;
    assert!(caps.has_graph_trace, "has_graph_trace should be true");
    Ok(())
}

#[test]
fn graph_replays_graph_trace_returns_wall_rows_without_nodes() -> Result<()> {
    let trace = fixture::with_graph_trace()?;
    let r = veloq_nsys_query::graph_replays::run(
        trace.path(),
        veloq_nsys_query::graph_replays::GraphReplaysRequest::default(),
    )?;
    assert_eq!(r.capture_mode.to_string(), "graph_trace");
    assert_eq!(r.total_matched, 3);
    assert_eq!(r.rows.len(), 3);
    let first = r
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected first replay"))?;
    assert_eq!(first.wall_ns, 10_000_000);
    assert_eq!(first.sum_gpu_ns, 10_000_000);
    assert_eq!(first.busy_ns, 10_000_000);
    assert_eq!(first.graph_trace_count, 1);
    assert_eq!(first.graph_id, Some(42));
    assert!(!first.decomposition_available);
    assert!(first.top_nodes.is_empty());
    assert_eq!(
        first.launcher_row_id.map(|id| id.kind),
        Some(EventKind::Runtime)
    );
    Ok(())
}

#[test]
fn graph_replays_nvtx_scopes_graph_trace_by_launch_runtime() -> Result<()> {
    let trace = fixture::with_graph_trace()?;
    let r = veloq_nsys_query::graph_replays::run(
        trace.path(),
        veloq_nsys_query::graph_replays::GraphReplaysRequest {
            nvtx: Some("frame".to_string()),
            ..Default::default()
        },
    )?;
    assert_eq!(
        r.rows.len(),
        3,
        "graph_trace rows should match through cudaGraphLaunch runtime containment"
    );
    assert_eq!(r.nvtx_scope.as_deref(), Some("frame"));
    Ok(())
}

#[test]
fn graph_replays_time_window_overlaps_without_clipping_duration() -> Result<()> {
    let trace = fixture::with_graph_trace()?;
    let r = veloq_nsys_query::graph_replays::run(
        trace.path(),
        veloq_nsys_query::graph_replays::GraphReplaysRequest {
            time_window: Some(TimeWindow::parse("@125ms-@126ms")?),
            ..Default::default()
        },
    )?;
    assert_eq!(r.rows.len(), 1);
    assert_eq!(
        r.rows
            .first()
            .ok_or_else(|| anyhow::anyhow!("expected overlapped replay"))?
            .wall_ns,
        10_000_000,
        "reported replay duration must stay full, not clipped to the window"
    );
    Ok(())
}

// ===== Node-mode (--cuda-graph-trace=node) ===================================
//
// Node-mode captures replace `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` with
// `CUDA_GRAPH_NODE_EVENTS` and inject graph-captured kernels into the
// regular kernel table with `graphId` / `graphNodeId` populated. The
// `with_graph_nodes` fixture mirrors that shape.

#[test]
fn capability_bit_set_when_node_events_present() -> Result<()> {
    let trace = fixture::with_graph_nodes()?;
    let s = veloq_nsys_query::summary::run(trace.path())?;
    let caps = s
        .auxiliary
        .capabilities
        .ok_or_else(|| anyhow::anyhow!("summary missing capabilities"))?;
    assert!(caps.has_graph_nodes, "has_graph_nodes should be true");
    // Mutually exclusive with graph-mode in the canonical capture flow.
    assert!(!caps.has_graph_trace, "node-mode lacks GRAPH_TRACE");
    Ok(())
}

#[test]
fn stats_group_by_graph_rolls_kernels_by_graph_id() -> Result<()> {
    let trace = fixture::with_graph_nodes()?;
    let req = veloq_nsys_query::stats::StatsRequest {
        kinds: KindFilter::Only(vec![EventKind::Kernel]),
        group_by: veloq_nsys_query::stats::GroupBy::from_arg("no-name,graph")?,
        ..Default::default()
    };
    let r = veloq_nsys_query::stats::run(trace.path(), req)?;
    // Expect 2 rows: one for graphId=42 (6 kernels totaling 45ms),
    // one for graphId=NULL (the eager kernel, 2ms).
    let graph_row = r
        .rows
        .iter()
        .find(|r| r.graph_id == Some(42))
        .ok_or_else(|| anyhow::anyhow!("missing graphId=42 row"))?;
    assert_eq!(graph_row.count, 6);
    assert_eq!(graph_row.total_ns, 45_000_000, "3 × (5ms + 10ms) = 45ms");
    let eager_row = r
        .rows
        .iter()
        .find(|r| r.graph_id.is_none())
        .ok_or_else(|| anyhow::anyhow!("missing eager (graphId NULL) row"))?;
    assert_eq!(eager_row.count, 1);
    assert_eq!(eager_row.total_ns, 2_000_000);
    Ok(())
}

#[test]
fn stats_group_by_graph_node_breaks_out_per_node() -> Result<()> {
    let trace = fixture::with_graph_nodes()?;
    let req = veloq_nsys_query::stats::StatsRequest {
        kinds: KindFilter::Only(vec![EventKind::Kernel]),
        group_by: veloq_nsys_query::stats::GroupBy::from_arg("no-name,graph_node")?,
        ..Default::default()
    };
    let r = veloq_nsys_query::stats::run(trace.path(), req)?;
    let node_1001 = r
        .rows
        .iter()
        .find(|r| r.graph_node_id == Some(1001))
        .ok_or_else(|| anyhow::anyhow!("missing node 1001"))?;
    assert_eq!(node_1001.count, 3, "node 1001 replayed 3 times");
    assert_eq!(node_1001.total_ns, 15_000_000, "3 × 5ms");
    let node_1002 = r
        .rows
        .iter()
        .find(|r| r.graph_node_id == Some(1002))
        .ok_or_else(|| anyhow::anyhow!("missing node 1002"))?;
    assert_eq!(node_1002.count, 3, "node 1002 replayed 3 times");
    assert_eq!(node_1002.total_ns, 30_000_000, "3 × 10ms");
    Ok(())
}

#[test]
fn inspect_kernel_surfaces_graph_fields_in_node_mode() -> Result<()> {
    let trace = fixture::with_graph_nodes()?;
    // Kernel rowid 1 is the first graph-captured kernel (node 1001).
    let id = RowId::new(EventKind::Kernel, 1);
    let r = veloq_nsys_query::inspect::run(trace.path(), &[id])?;
    let first = r
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected one event"))?;
    match first {
        veloq_nsys_query::inspect::EventDetails::Kernel(k) => {
            assert_eq!(k.graph_id, Some(42));
            assert_eq!(k.graph_node_id, Some(1001));
        }
        other => anyhow::bail!("expected EventDetails::Kernel, got {other:?}"),
    }
    Ok(())
}

#[test]
fn inspect_eager_kernel_omits_graph_fields() -> Result<()> {
    let trace = fixture::with_graph_nodes()?;
    // Kernel rowid 7 is the eager kernel (inserted after the 6 graph ones).
    let id = RowId::new(EventKind::Kernel, 7);
    let r = veloq_nsys_query::inspect::run(trace.path(), &[id])?;
    let first = r
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected one event"))?;
    match first {
        veloq_nsys_query::inspect::EventDetails::Kernel(k) => {
            assert!(k.graph_id.is_none(), "eager kernel should have no graph_id");
            assert!(
                k.graph_node_id.is_none(),
                "eager kernel should have no graph_node_id"
            );
        }
        other => anyhow::bail!("expected EventDetails::Kernel, got {other:?}"),
    }
    Ok(())
}

#[test]
fn inspect_graph_node_returns_node_metadata() -> Result<()> {
    let trace = fixture::with_graph_nodes()?;
    let id = RowId::new(EventKind::GraphNode, 1);
    let r = veloq_nsys_query::inspect::run(trace.path(), &[id])?;
    let first = r
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected one event"))?;
    match first {
        veloq_nsys_query::inspect::EventDetails::GraphNode(n) => {
            assert_eq!(n.graph_node_id, 1001);
            assert!(n.original_graph_node_id.is_none());
        }
        other => anyhow::bail!("expected EventDetails::GraphNode, got {other:?}"),
    }
    Ok(())
}

#[test]
fn search_returns_graph_node_metadata_hits() -> Result<()> {
    let trace = fixture::with_graph_nodes()?;
    let req = veloq_nsys_query::search::SearchRequest {
        kinds: KindFilter::Only(vec![EventKind::GraphNode]),
        limit: 10,
        ..Default::default()
    };
    let r = veloq_nsys_query::search::run(trace.path(), req)?;
    assert_eq!(r.rows.len(), 2, "two distinct nodes in fixture");
    for h in &r.rows {
        let b = h.base();
        assert_eq!(b.row_id.kind, EventKind::GraphNode);
        assert!(b.name.starts_with("node:"));
    }
    Ok(())
}

#[test]
fn inspect_graph_node_enriches_with_parent_graph_id() -> Result<()> {
    // The with_graph_nodes fixture's NODE_EVENTS rows reference
    // graphNodeId 1001/1002. The kernel table has rows with the same
    // graphNodeId and graphId=42, so `query_graph_node` should
    // resolve `graph_id: Some(42)` via the correlated lookup.
    let trace = fixture::with_graph_nodes()?;
    let id = RowId::new(EventKind::GraphNode, 1);
    let r = veloq_nsys_query::inspect::run(trace.path(), &[id])?;
    let first = r
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected one event"))?;
    match first {
        veloq_nsys_query::inspect::EventDetails::GraphNode(n) => {
            assert_eq!(
                n.graph_id,
                Some(42),
                "graph_id should be resolved from kernel-table join"
            );
            // graph_exec_id requires CUDA_GRAPH_EVENTS — absent in the
            // pure node-mode fixture, so this stays None.
            assert!(n.graph_exec_id.is_none());
        }
        other => anyhow::bail!("expected EventDetails::GraphNode, got {other:?}"),
    }
    Ok(())
}

#[test]
fn graph_replays_node_mode_groups_by_correlation_and_orders_top_nodes() -> Result<()> {
    let trace = fixture::with_graph_nodes()?;
    let r = veloq_nsys_query::graph_replays::run(
        trace.path(),
        veloq_nsys_query::graph_replays::GraphReplaysRequest::default(),
    )?;
    assert_eq!(r.capture_mode.to_string(), "graph_nodes");
    assert_eq!(r.total_matched, 3);
    assert_eq!(r.rows.len(), 3);
    let first = r
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected first replay"))?;
    assert_eq!(first.event_count, 2);
    assert_eq!(first.kernel_count, 2);
    assert_eq!(first.wall_ns, 15_500_000);
    assert_eq!(first.sum_gpu_ns, 15_000_000);
    assert_eq!(first.busy_ns, 15_000_000);
    assert_eq!(first.idle_inside_replay_ns, 500_000);
    assert!(first.decomposition_available);
    assert!(r.rows.iter().all(|row| {
        row.launcher_row_id
            .is_some_and(|id| id.kind == EventKind::Runtime)
    }));
    assert_eq!(
        r.rows
            .iter()
            .filter_map(|row| row.launcher_row_id.map(|id| id.rowid))
            .collect::<std::collections::HashSet<_>>()
            .len(),
        3
    );
    assert_eq!(first.top_nodes.len(), 2);
    let first_node = first
        .top_nodes
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected first top graph node"))?;
    let second_node = first
        .top_nodes
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("expected second top graph node"))?;
    assert_eq!(first_node.graph_node_id, 1002);
    assert_eq!(first_node.sum_ns, 10_000_000);
    assert_eq!(second_node.graph_node_id, 1001);
    Ok(())
}

#[test]
fn graph_replays_node_mode_nvtx_filters_by_launch_runtime() -> Result<()> {
    let trace = fixture::with_graph_nodes()?;
    let r = veloq_nsys_query::graph_replays::run(
        trace.path(),
        veloq_nsys_query::graph_replays::GraphReplaysRequest {
            nvtx: Some("frame".to_string()),
            ..Default::default()
        },
    )?;
    assert_eq!(r.rows.len(), 3);
    Ok(())
}

#[test]
fn graph_replays_same_raw_correlation_on_two_devices_does_not_merge() -> Result<()> {
    let trace = fixture::graph_trace_reused_correlation_two_devices()?;
    let r = veloq_nsys_query::graph_replays::run(
        trace.path(),
        veloq_nsys_query::graph_replays::GraphReplaysRequest {
            device: None,
            ..Default::default()
        },
    )?;
    assert_eq!(r.rows.len(), 2);
    assert_eq!(r.total_matched, 2);
    let first = r
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected first replay"))?;
    let second = r
        .rows
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("expected second replay"))?;
    assert_ne!(first.synthetic_id, second.synthetic_id);
    assert_eq!(first.correlation_id, second.correlation_id);
    Ok(())
}

#[test]
fn resident_graph_trace_index_preserves_varying_query_responses() -> Result<()> {
    let fixture = fixture::with_graph_trace()?;
    let trace = veloq_nsys_data::Trace::open(fixture.path())?;
    let mut requests = vec![
        veloq_nsys_query::graph_replays::GraphReplaysRequest::default(),
        veloq_nsys_query::graph_replays::GraphReplaysRequest {
            time_window: Some(TimeWindow::parse("@125ms-@126ms")?),
            sort: Some(veloq_core::SortSpec::parse("start:asc")?),
            limit: 1,
            ..Default::default()
        },
        veloq_nsys_query::graph_replays::GraphReplaysRequest {
            nvtx: Some("frame".to_string()),
            top_nodes_limit: 1,
            ..Default::default()
        },
    ];
    for sort in ["wall:asc", "sum:desc", "start:desc", "count:asc"] {
        requests.push(veloq_nsys_query::graph_replays::GraphReplaysRequest {
            process_id: Some(12345),
            device: Some(0),
            sort: Some(veloq_core::SortSpec::parse(sort)?),
            ..Default::default()
        });
    }
    for request in requests {
        let one_shot = veloq_nsys_query::graph_replays::run(fixture.path(), request.clone())?;
        assert!(veloq_nsys_query::graph_replays::ensure_resident_index(
            &trace
        )?);
        let resident = veloq_nsys_query::graph_replays::run_with_trace(&trace, request)?;
        assert_eq!(
            serde_json::to_vec(&resident)?,
            serde_json::to_vec(&one_shot)?
        );
    }
    Ok(())
}

#[test]
fn resident_graph_node_index_preserves_varying_query_responses() -> Result<()> {
    let fixture = fixture::with_graph_nodes()?;
    let trace = veloq_nsys_data::Trace::open(fixture.path())?;
    let mut requests = vec![
        veloq_nsys_query::graph_replays::GraphReplaysRequest::default(),
        veloq_nsys_query::graph_replays::GraphReplaysRequest {
            time_window: Some(TimeWindow::parse("@200ms-@216ms")?),
            sort: Some(veloq_core::SortSpec::parse("sum:desc,start:asc")?),
            limit: 2,
            top_nodes_limit: 1,
            ..Default::default()
        },
        veloq_nsys_query::graph_replays::GraphReplaysRequest {
            nvtx: Some("frame".to_string()),
            ..Default::default()
        },
    ];
    for sort in ["wall:asc", "sum:desc", "start:desc", "count:asc"] {
        requests.push(veloq_nsys_query::graph_replays::GraphReplaysRequest {
            process_id: Some(12345),
            device: Some(0),
            sort: Some(veloq_core::SortSpec::parse(sort)?),
            top_nodes_limit: 2,
            ..Default::default()
        });
    }
    for request in requests {
        let one_shot = veloq_nsys_query::graph_replays::run(fixture.path(), request.clone())?;
        assert!(veloq_nsys_query::graph_replays::ensure_resident_index(
            &trace
        )?);
        let resident = veloq_nsys_query::graph_replays::run_with_trace(&trace, request)?;
        assert_eq!(
            serde_json::to_vec(&resident)?,
            serde_json::to_vec(&one_shot)?
        );
    }
    Ok(())
}

#[test]
fn resident_graph_index_absence_preserves_established_empty_response() -> Result<()> {
    let fixture = fixture::minimal_gpu()?;
    let trace = veloq_nsys_data::Trace::open(fixture.path())?;
    let request = veloq_nsys_query::graph_replays::GraphReplaysRequest::default();
    let one_shot = veloq_nsys_query::graph_replays::run(fixture.path(), request.clone())?;
    assert!(!veloq_nsys_query::graph_replays::ensure_resident_index(
        &trace
    )?);
    let resident = veloq_nsys_query::graph_replays::run_with_trace(&trace, request)?;
    assert_eq!(
        serde_json::to_vec(&resident)?,
        serde_json::to_vec(&one_shot)?
    );
    Ok(())
}

// ===== Graph-events (CUDA_GRAPH_EVENTS lifecycle) ============================

#[test]
fn capability_bit_set_when_graph_events_present() -> Result<()> {
    let trace = fixture::with_graph_trace()?;
    let s = veloq_nsys_query::summary::run(trace.path())?;
    let caps = s
        .auxiliary
        .capabilities
        .ok_or_else(|| anyhow::anyhow!("summary missing capabilities"))?;
    assert!(caps.has_graph_events, "has_graph_events should be true");
    Ok(())
}

#[test]
fn search_returns_graph_event_hits() -> Result<()> {
    let trace = fixture::with_graph_trace()?;
    let req = veloq_nsys_query::search::SearchRequest {
        kinds: KindFilter::Only(vec![EventKind::GraphEvent]),
        limit: 10,
        ..Default::default()
    };
    let r = veloq_nsys_query::search::run(trace.path(), req)?;
    // The fixture has 3 GRAPH_EVENTS rows: 2 Graph Creations + 1
    // GraphExec Creation.
    assert_eq!(r.rows.len(), 3);
    let names: Vec<&str> = r.rows.iter().map(|h| h.base().name.as_str()).collect();
    assert!(names.contains(&"graph_creation"));
    assert!(names.contains(&"graph_exec_creation"));
    Ok(())
}

#[test]
fn search_graph_event_filterable_by_subtype_name() -> Result<()> {
    let trace = fixture::with_graph_trace()?;
    let req = veloq_nsys_query::search::SearchRequest {
        kinds: KindFilter::Only(vec![EventKind::GraphEvent]),
        name_glob: Some("graph_exec_creation".to_string()),
        limit: 10,
        ..Default::default()
    };
    let r = veloq_nsys_query::search::run(trace.path(), req)?;
    assert_eq!(r.rows.len(), 1, "single GraphExec Creation in fixture");
    Ok(())
}

#[test]
fn inspect_graph_event_returns_lifecycle_metadata() -> Result<()> {
    let trace = fixture::with_graph_trace()?;
    // First GRAPH_EVENTS row is a Graph Creation for graphId=41.
    let id = RowId::new(EventKind::GraphEvent, 1);
    let r = veloq_nsys_query::inspect::run(trace.path(), &[id])?;
    let first = r
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected one event"))?;
    match first {
        veloq_nsys_query::inspect::EventDetails::GraphEvent(g) => {
            assert_eq!(g.event_class, 95);
            assert_eq!(g.event_class_name, "graph_creation");
            assert_eq!(g.graph_id, 41);
            assert!(g.graph_exec_id.is_none(), "raw Graph Creation has no exec");
        }
        other => anyhow::bail!("expected EventDetails::GraphEvent, got {other:?}"),
    }

    // Third row is the GraphExec Creation for graphId=42.
    let id = RowId::new(EventKind::GraphEvent, 3);
    let r = veloq_nsys_query::inspect::run(trace.path(), &[id])?;
    let third = r
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected one event"))?;
    match third {
        veloq_nsys_query::inspect::EventDetails::GraphEvent(g) => {
            assert_eq!(g.event_class, 94);
            assert_eq!(g.event_class_name, "graph_exec_creation");
            assert_eq!(g.graph_id, 42);
            assert_eq!(g.graph_exec_id, Some(43));
        }
        other => anyhow::bail!("expected EventDetails::GraphEvent, got {other:?}"),
    }
    Ok(())
}

#[test]
fn correlate_graph_event_short_circuits() -> Result<()> {
    // GRAPH_EVENTS rows have no correlationId — correlate must
    // return a not-found result, never an internal hydration bail.
    let trace = fixture::with_graph_trace()?;
    let id = RowId::new(EventKind::GraphEvent, 1);
    let r = veloq_nsys_query::correlate::run(trace.path(), &[id])?;
    let res = r
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected one result"))?;
    assert!(!res.correlation_found);
    assert!(res.auxiliary.gpu_events.is_empty());
    assert!(res.auxiliary.cpu_events.is_empty());
    Ok(())
}

// ===== CUDA Event activity + Overhead + sync.event_sync_id enrichment =======

#[test]
fn capability_bits_set_for_cuda_event_and_overhead() -> Result<()> {
    let trace = fixture::with_sync()?;
    let s = veloq_nsys_query::summary::run(trace.path())?;
    let caps = s
        .auxiliary
        .capabilities
        .ok_or_else(|| anyhow::anyhow!("summary missing capabilities"))?;
    assert!(caps.has_cuda_event_activity);
    assert!(caps.has_overhead);
    Ok(())
}

#[test]
fn search_returns_cuda_event_hits() -> Result<()> {
    let trace = fixture::with_sync()?;
    let req = veloq_nsys_query::search::SearchRequest {
        kinds: KindFilter::Only(vec![EventKind::CudaEvent]),
        limit: 10,
        ..Default::default()
    };
    let r = veloq_nsys_query::search::run(trace.path(), req)?;
    assert_eq!(r.rows.len(), 2, "two cuda_event rows in fixture");
    for h in &r.rows {
        let b = h.base();
        assert_eq!(b.row_id.kind, EventKind::CudaEvent);
        assert!(b.name.starts_with("cuda_event:"));
    }
    Ok(())
}

#[test]
fn inspect_cuda_event_returns_details() -> Result<()> {
    let trace = fixture::with_sync()?;
    let id = RowId::new(EventKind::CudaEvent, 1);
    let r = veloq_nsys_query::inspect::run(trace.path(), &[id])?;
    let first = r
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected one event"))?;
    match first {
        veloq_nsys_query::inspect::EventDetails::CudaEvent(c) => {
            assert_eq!(c.event_id, 42);
            assert_eq!(c.event_sync_id, Some(777));
            assert_eq!(c.stream_id, 7);
        }
        other => anyhow::bail!("expected EventDetails::CudaEvent, got {other:?}"),
    }
    Ok(())
}

#[test]
fn inspect_sync_surfaces_event_sync_id() -> Result<()> {
    let trace = fixture::with_sync()?;
    // Sync rowid 1 is cudaEventSynchronize with eventSyncId=777,
    // which pairs with cuda_event:1's event_sync_id=777.
    let id = RowId::new(EventKind::Sync, 1);
    let r = veloq_nsys_query::inspect::run(trace.path(), &[id])?;
    let first = r
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected one event"))?;
    match first {
        veloq_nsys_query::inspect::EventDetails::Sync(s) => {
            assert_eq!(s.sync_type_name, "cudaEventSynchronize");
            assert_eq!(
                s.event_sync_id,
                Some(777),
                "sync should expose event_sync_id for cross-event pairing"
            );
        }
        other => anyhow::bail!("expected EventDetails::Sync, got {other:?}"),
    }
    Ok(())
}

#[test]
fn search_returns_overhead_hits_with_type_label() -> Result<()> {
    let trace = fixture::with_sync()?;
    let req = veloq_nsys_query::search::SearchRequest {
        kinds: KindFilter::Only(vec![EventKind::Overhead]),
        limit: 10,
        ..Default::default()
    };
    let r = veloq_nsys_query::search::run(trace.path(), req)?;
    // The `with_sync` fixture now carries three overhead spans —
    // two with NULL correlationId (rowids 1+2) and a third with a
    // real correlationId for the overhead-correlation regression
    // (see `correlate_overhead_with_real_correlation_finds_paired_kernel`).
    assert_eq!(r.rows.len(), 3);
    let names: Vec<&str> = r.rows.iter().map(|h| h.base().name.as_str()).collect();
    assert!(names.contains(&"cupti_instrumentation"));
    assert!(names.contains(&"command_buffer_full"));
    assert!(names.contains(&"driver_compiler"));
    Ok(())
}

#[test]
fn inspect_overhead_returns_details_with_label() -> Result<()> {
    let trace = fixture::with_sync()?;
    let id = RowId::new(EventKind::Overhead, 1);
    let r = veloq_nsys_query::inspect::run(trace.path(), &[id])?;
    let first = r
        .rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("expected one event"))?;
    match first {
        veloq_nsys_query::inspect::EventDetails::Overhead(o) => {
            assert_eq!(o.overhead_type, 4);
            assert_eq!(o.overhead_type_name, "cupti_instrumentation");
            assert_eq!(o.duration_ns, 100_000);
        }
        other => anyhow::bail!("expected EventDetails::Overhead, got {other:?}"),
    }
    Ok(())
}
