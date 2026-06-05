//! Per-command flatteners — turn each Response type into a `TabularView`
//! for the CSV/table output formats. JSON keeps the original nested
//! shape; everything else funnels through this module.
//!
//! Design: each command has one "primary list" (rows / events / gaps /
//! slices / per_table / …). The primary list becomes the table body;
//! anything else from the envelope (counts, time window, NVTX scope)
//! becomes `meta` lines. When a response is non-tabular (e.g. `prep`
//! or `correlation-stats`), we emit a small `field` / `value`
//! key-value grid so the format is still meaningful.

use crate::format::{DISPLAY_PRECISION, TabularView, cell_opt};
use veloq_nsys_query::{
    EventRef,
    concurrency::ConcurrencyResponse,
    correlate::CorrelateResponse,
    gaps::GapsResponse,
    graph_replays::GraphReplaysResponse,
    hardware::HardwareResponse,
    inspect::InspectResponse,
    metrics::MetricsResponse,
    search::SearchResponse,
    slices::{SlicesResponse, SlicesRow},
    stats::StatsResponse,
    stats_by_size::StatsBySizeResponse,
    summary::Summary,
    timeline::TimelineResponse,
};

pub fn stats_view(data: &StatsResponse) -> TabularView {
    // Identity columns mirror StatRow's group-key fields so the
    // table/CSV output can discriminate rows whose differences live
    // only in the group axes added after the initial schema (graph,
    // graph_node, grid_block, nvtx_parent, nvtx_path, nvtx_style,
    // event_type).
    // Without them, two rows that JSON serialises distinctly can
    // collapse into indistinguishable-looking duplicates under
    // --format table. Empty cells for axes the request didn't
    // activate match the JSON `skip_serializing_if = "Option::is_none"`
    // convention.
    let mut v = TabularView::new(vec![
        "type",
        "name",
        "short_name",
        "device_id",
        "context_id",
        "stream_id",
        "graph_id",
        "graph_node_id",
        "nvtx_parent_name",
        "nvtx_parent_depth",
        "nvtx_path",
        "nvtx_style",
        "event_type",
        "grid_x",
        "grid_y",
        "grid_z",
        "block_x",
        "block_y",
        "block_z",
        "count",
        "total_ns",
        "avg_ns",
        "min_ns",
        "max_ns",
        "p50_ns",
        "p95_ns",
        "p99_ns",
        "bytes_total",
        "gbps",
        "percentage",
    ]);
    for r in &data.rows {
        v.push_row(vec![
            r.kind.to_string(),
            r.name.clone().unwrap_or_default(),
            r.short_name.clone().unwrap_or_default(),
            cell_opt(r.device_id),
            cell_opt(r.context_id),
            cell_opt(r.stream_id),
            cell_opt(r.graph_id),
            cell_opt(r.graph_node_id),
            r.nvtx_parent_name.clone().unwrap_or_default(),
            cell_opt(r.nvtx_parent_depth),
            r.nvtx_path.clone().unwrap_or_default(),
            r.nvtx_style.unwrap_or_default().to_string(),
            cell_opt(r.event_type),
            cell_opt(r.grid_x),
            cell_opt(r.grid_y),
            cell_opt(r.grid_z),
            cell_opt(r.block_x),
            cell_opt(r.block_y),
            cell_opt(r.block_z),
            r.count.to_string(),
            r.total_ns.to_string(),
            r.avg_ns.to_string(),
            r.min_ns.to_string(),
            r.max_ns.to_string(),
            r.p50_ns.to_string(),
            r.p95_ns.to_string(),
            r.p99_ns.to_string(),
            cell_opt(r.bytes_total),
            cell_opt(r.gbps.map(|x| format!("{x:.*}", DISPLAY_PRECISION))),
            format!("{:.*}", DISPLAY_PRECISION, r.percentage),
        ]);
    }
    v.push_meta("count", data.count.to_string());
    v.push_meta("total_matched", data.total_matched.to_string());
    v.push_meta("total_duration_ns", data.total_duration_ns.to_string());
    v.push_meta("total_events", data.total_events.to_string());
    if let Some((s, e)) = data.time_window_ns {
        v.push_meta("time_window_ns", format!("{s}-{e}"));
    }
    if let Some(scope) = &data.nvtx_scope {
        v.push_meta("nvtx_scope", scope.clone());
    }
    if data.histogram_buckets_ns.is_some() {
        // Histogram doesn't fit a single-cell column gracefully; point
        // the user at JSON if they asked for --hist with a tabular format.
        // Underscore-prefixed key flags this as a human-readable hint
        // rather than a structured meta field — scripts can skip
        // anything starting with `_`.
        v.push_meta(
            "_note",
            "histogram values are JSON-only; rerun with `--format json` to see them",
        );
    }
    v
}

pub fn stats_by_size_view(data: &StatsBySizeResponse) -> TabularView {
    let mut v = TabularView::new(vec![
        "type",
        "name",
        "short_name",
        "device_id",
        "context_id",
        "stream_id",
        "count",
        "total_bytes",
        "avg_bytes",
        "min_bytes",
        "max_bytes",
        "p50_bytes",
        "p95_bytes",
        "p99_bytes",
        "percentage",
    ]);
    for r in &data.rows {
        v.push_row(vec![
            r.kind.to_string(),
            r.name.clone().unwrap_or_default(),
            r.short_name.clone().unwrap_or_default(),
            cell_opt(r.device_id),
            cell_opt(r.context_id),
            cell_opt(r.stream_id),
            r.count.to_string(),
            r.total_bytes.to_string(),
            r.avg_bytes.to_string(),
            r.min_bytes.to_string(),
            r.max_bytes.to_string(),
            r.p50_bytes.to_string(),
            r.p95_bytes.to_string(),
            r.p99_bytes.to_string(),
            format!("{:.*}", DISPLAY_PRECISION, r.percentage),
        ]);
    }
    v.push_meta("count", data.count.to_string());
    v.push_meta("total_matched", data.total_matched.to_string());
    v.push_meta("total_bytes", data.total_bytes.to_string());
    v.push_meta("total_events", data.total_events.to_string());
    if let Some((s, e)) = data.time_window_ns {
        v.push_meta("time_window_ns", format!("{s}-{e}"));
    }
    v
}

pub fn search_view(data: &SearchResponse) -> TabularView {
    let mut v = TabularView::new(vec![
        "row_id",
        "name",
        "start_ns",
        "duration_ns",
        "device_id",
        "stream_id",
        "global_tid",
    ]);
    for e in &data.rows {
        let b = e.base();
        v.push_row(vec![
            b.row_id.to_string(),
            b.name.clone(),
            b.start_ns.to_string(),
            b.duration_ns.to_string(),
            cell_opt(b.device_id),
            cell_opt(b.stream_id),
            cell_opt(b.global_tid),
        ]);
    }
    v.push_meta("count", data.count.to_string());
    v.push_meta("total_matched", data.total_matched.to_string());
    if let Some((s, e)) = data.time_window_ns {
        v.push_meta("time_window_ns", format!("{s}-{e}"));
    }
    if let Some(scope) = &data.nvtx_scope {
        v.push_meta("nvtx_scope", scope.clone());
    }
    v
}

pub fn timeline_view(data: &TimelineResponse) -> TabularView {
    let mut v = TabularView::new(vec![
        "start_ns",
        "end_ns",
        "total_ns",
        "kernel_ns",
        "memcpy_ns",
        "memset_ns",
        "graph_ns",
        "count",
        "kernel_count",
        "memcpy_count",
        "memset_count",
        "graph_count",
    ]);
    for b in &data.rows {
        v.push_row(vec![
            b.start_ns.to_string(),
            b.end_ns.to_string(),
            b.total_ns.to_string(),
            b.kernel_ns.to_string(),
            b.memcpy_ns.to_string(),
            b.memset_ns.to_string(),
            b.graph_ns.to_string(),
            b.count.to_string(),
            b.kernel_count.to_string(),
            b.memcpy_count.to_string(),
            b.memset_count.to_string(),
            b.graph_count.to_string(),
        ]);
    }
    v.push_meta("interval_ns", data.interval_ns.to_string());
    v.push_meta("count", data.count.to_string());
    v.push_meta("total_matched", data.total_matched.to_string());
    if let Some((s, e)) = data.time_window_ns {
        v.push_meta("time_window_ns", format!("{s}-{e}"));
    }
    if let Some(scope) = &data.nvtx_scope {
        v.push_meta("nvtx_scope", scope.clone());
    }
    v
}

pub fn concurrency_view(data: &ConcurrencyResponse) -> TabularView {
    // Per-device summary. The full per-stream breakdown stays in the
    // JSON payload; the table collapses it to a stream count (same
    // "flatten nested to a scalar for tabular" choice as graph_replays).
    let mut v = TabularView::new(vec![
        "device_id",
        "sum_busy_ns",
        "union_busy_ns",
        "overlap_ns",
        "max_concurrency",
        "compute_union_ns",
        "copy_union_ns",
        "compute_copy_overlap_ns",
        "streams",
    ]);
    for d in &data.rows {
        v.push_row(vec![
            d.device_id.to_string(),
            d.sum_busy_ns.to_string(),
            d.union_busy_ns.to_string(),
            d.overlap_ns.to_string(),
            d.max_concurrency.to_string(),
            d.compute_vs_copy.compute_union_ns.to_string(),
            d.compute_vs_copy.copy_union_ns.to_string(),
            d.compute_vs_copy.compute_copy_overlap_ns.to_string(),
            d.streams.len().to_string(),
        ]);
    }
    v.push_meta("count", data.count.to_string());
    v.push_meta("total_matched", data.total_matched.to_string());
    if let Some((s, e)) = data.time_window_ns {
        v.push_meta("time_window_ns", format!("{s}-{e}"));
    }
    v
}

pub fn gaps_view(data: &GapsResponse) -> TabularView {
    // Under unified scopes (device / trace) `g.device_id` /
    // `g.stream_id` are `None`; the bracketing events' stream lives
    // on prev/next, so render them too. Empty cells for the absent
    // axes keep the column count stable.
    let mut v = TabularView::new(vec![
        "device_id",
        "stream_id",
        "start_ns",
        "end_ns",
        "duration_ns",
        "prev_row_id",
        "prev_stream",
        "prev_name",
        "next_row_id",
        "next_stream",
        "next_name",
    ]);
    for g in &data.rows {
        v.push_row(vec![
            g.device_id.map(|d| d.to_string()).unwrap_or_default(),
            g.stream_id.map(|s| s.to_string()).unwrap_or_default(),
            g.start_ns.to_string(),
            g.end_ns.to_string(),
            g.duration_ns.to_string(),
            g.prev.row_id.to_string(),
            g.prev.stream_id.to_string(),
            g.prev.name.clone(),
            g.next.row_id.to_string(),
            g.next.stream_id.to_string(),
            g.next.name.clone(),
        ]);
    }
    v.push_meta("scope", data.scope.to_string());
    v.push_meta("min_ns", data.min_ns.to_string());
    v.push_meta("count", data.count.to_string());
    v.push_meta("total_matched", data.total_matched.to_string());
    if let Some((s, e)) = data.time_window_ns {
        v.push_meta("time_window_ns", format!("{s}-{e}"));
    }
    v
}

pub fn graph_replays_view(data: &GraphReplaysResponse) -> TabularView {
    let mut v = TabularView::new(vec![
        "synthetic_id",
        "capture_mode",
        "device_id",
        "context_id",
        "correlation_id",
        "launcher_row_id",
        "start_ns",
        "end_ns",
        "wall_ns",
        "sum_gpu_ns",
        "busy_ns",
        "idle_inside_replay_ns",
        "event_count",
        "kernel_count",
        "memcpy_count",
        "memset_count",
        "stream_count",
        "graph_id",
        "graph_exec_id",
        "decomposition_available",
        "top_node_name",
        "top_node_sum_ns",
    ]);
    for r in &data.rows {
        let top = r.top_nodes.first();
        v.push_row(vec![
            r.synthetic_id.clone(),
            r.capture_mode.to_string(),
            r.device_id.to_string(),
            r.context_id.to_string(),
            r.correlation_id.to_string(),
            r.launcher_row_id
                .map(|id| id.to_string())
                .unwrap_or_default(),
            r.start_ns.to_string(),
            r.end_ns.to_string(),
            r.wall_ns.to_string(),
            r.sum_gpu_ns.to_string(),
            r.busy_ns.to_string(),
            r.idle_inside_replay_ns.to_string(),
            r.event_count.to_string(),
            r.kernel_count.to_string(),
            r.memcpy_count.to_string(),
            r.memset_count.to_string(),
            r.stream_count.to_string(),
            cell_opt(r.graph_id),
            cell_opt(r.graph_exec_id),
            r.decomposition_available.to_string(),
            top.map(|n| n.name.clone()).unwrap_or_default(),
            top.map(|n| n.sum_ns.to_string()).unwrap_or_default(),
        ]);
    }
    v.push_meta("capture_mode", data.capture_mode.to_string());
    v.push_meta("count", data.count.to_string());
    v.push_meta("total_matched", data.total_matched.to_string());
    v.push_meta("top_nodes_limit", data.top_nodes_limit.to_string());
    if let Some((s, e)) = data.time_window_ns {
        v.push_meta("time_window_ns", format!("{s}-{e}"));
    }
    if let Some(scope) = &data.nvtx_scope {
        v.push_meta("nvtx_scope", scope.clone());
    }
    v
}

pub fn slices_view(data: &SlicesResponse) -> TabularView {
    if data.view == "aggregate" {
        let mut v = TabularView::new(vec![
            "name",
            "path",
            "instances",
            "attributed_total_ns",
            "p50_ns",
            "p99_ns",
        ]);
        for row in &data.rows {
            let SlicesRow::Aggregate(r) = row else {
                continue;
            };
            v.push_row(vec![
                r.name.clone(),
                r.path.clone().unwrap_or_default(),
                r.instances.to_string(),
                r.attributed_total_ns.to_string(),
                format!("{:.0}", r.p50_ns),
                format!("{:.0}", r.p99_ns),
            ]);
        }
        push_slices_meta(&mut v, data);
        return v;
    }

    // Flatten: one row per (slice, device, stream). Slice-level fields
    // repeat across its sub-rows; the agent can collapse them again on
    // (row_id) if they care, and humans see the relationship clearly.
    let mut v = TabularView::new(vec![
        "slice_row_id",
        "slice_name",
        "cpu_start_ns",
        "cpu_end_ns",
        "cpu_duration_ns",
        "device_id",
        "stream_id",
        "gpu_start_ns",
        "gpu_end_ns",
        "kernel_ns",
        "kernel_count",
        "memcpy_ns",
        "memcpy_count",
        "memset_ns",
        "memset_count",
    ]);
    for row in &data.rows {
        let SlicesRow::Instance(s) = row else {
            continue;
        };
        if s.gpu_attributed.is_empty() {
            // Slice with no GPU work attributed — still emit one row so
            // the user sees it. Stream columns become blank.
            v.push_row(vec![
                s.row_id.to_string(),
                s.name.clone(),
                s.cpu.start_ns.to_string(),
                s.cpu.end_ns.to_string(),
                s.cpu.duration_ns.to_string(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                "0".into(),
                "0".into(),
                "0".into(),
                "0".into(),
                "0".into(),
                "0".into(),
            ]);
        } else {
            for g in &s.gpu_attributed {
                v.push_row(vec![
                    s.row_id.to_string(),
                    s.name.clone(),
                    s.cpu.start_ns.to_string(),
                    s.cpu.end_ns.to_string(),
                    s.cpu.duration_ns.to_string(),
                    g.device_id.to_string(),
                    g.stream_id.to_string(),
                    g.start_ns.to_string(),
                    g.end_ns.to_string(),
                    g.kernel_ns.to_string(),
                    g.kernel_count.to_string(),
                    g.memcpy_ns.to_string(),
                    g.memcpy_count.to_string(),
                    g.memset_ns.to_string(),
                    g.memset_count.to_string(),
                ]);
            }
        }
    }
    push_slices_meta(&mut v, data);
    v
}

fn push_slices_meta(v: &mut TabularView, data: &SlicesResponse) {
    v.push_meta("view", data.view.to_string());
    if let Some(group_by) = data.group_by {
        v.push_meta("group_by", group_by.to_string());
    }
    if let Some(name) = &data.name {
        v.push_meta("name", name.clone());
    }
    if let Some(re) = &data.name_regex {
        v.push_meta("name_regex", re.clone());
    }
    v.push_meta("attribution", data.attribution.to_string());
    v.push_meta("count", data.rows.len().to_string());
    v.push_meta("total_matched", data.total_matched.to_string());
    if let Some((s, e)) = data.time_window_ns {
        v.push_meta("time_window_ns", format!("{s}-{e}"));
    }
}

pub fn summary_view(data: &Summary) -> TabularView {
    let mut v = TabularView::new(vec!["table", "row_count", "start_ns", "end_ns"]);
    for t in &data.rows {
        v.push_row(vec![
            t.name.clone(),
            t.row_count.to_string(),
            t.start_ns.to_string(),
            t.end_ns.to_string(),
        ]);
    }
    if let Some(s) = &data.schema_version {
        v.push_meta("schema_version", s.clone());
    }
    if let Some(p) = &data.product_version {
        v.push_meta("product_version", p.clone());
    }
    v.push_meta(
        "full_time_range_ns",
        format!(
            "{}-{} (dur {})",
            data.auxiliary.full_time_range_ns.start,
            data.auxiliary.full_time_range_ns.end,
            data.auxiliary.full_time_range_ns.duration
        ),
    );
    if let Some(c) = &data.auxiliary.capabilities {
        // Compact one-line listing of `true` flags — agents reading
        // JSON get the structured shape; table consumers get a
        // human-readable summary that doesn't need vertical sprawl.
        let mut active: Vec<&str> = Vec::new();
        if c.has_kernels {
            active.push("kernels");
        }
        if c.has_memcpy {
            active.push("memcpy");
        }
        if c.has_memset {
            active.push("memset");
        }
        if c.has_sync {
            active.push("sync");
        }
        if c.has_runtime {
            active.push("runtime");
        }
        if c.has_osrt {
            active.push("osrt");
        }
        if c.has_nvtx {
            active.push("nvtx");
        }
        if c.has_cuda_contexts {
            active.push("cuda_contexts");
        }
        if c.has_sampling {
            active.push("sampling");
        }
        if c.has_composite_events {
            active.push("composite_events");
        }
        if c.has_gpu_metrics {
            active.push("gpu_metrics");
        }
        if c.has_nic_metrics {
            active.push("nic_metrics");
        }
        if c.has_target_info {
            active.push("target_info");
        }
        v.push_meta("capabilities", active.join(", "));
    }
    v
}

pub fn inspect_view(data: &InspectResponse) -> TabularView {
    // Per-event-kind formatting lives on `EventDetails::summary_row`,
    // next to the detail structs. This module's job is just to drop
    // each `SummaryRow` into the TabularView grid.
    let mut v = TabularView::new(vec![
        "row_id",
        "type",
        "name",
        "start_ns",
        "duration_ns",
        "device_id",
        "stream_id",
        "details",
    ]);
    for e in &data.rows {
        let s = e.summary_row();
        v.push_row(vec![
            s.row_id,
            s.kind.to_string(),
            s.name,
            s.start_ns,
            s.duration_ns,
            cell_opt(s.device_id),
            cell_opt(s.stream_id),
            s.details,
        ]);
    }
    v
}

pub fn correlate_view(data: &CorrelateResponse) -> TabularView {
    // One row per related event, prefixed with the originating row_id so
    // the agent (or human) can group sub-events back to their parent
    // input. cpu/gpu side discriminator stays as a column.
    let mut v = TabularView::new(vec![
        "input_row_id",
        "side",
        "row_id",
        "name",
        "start_ns",
        "duration_ns",
        "device_id",
        "stream_id",
        "global_tid",
        "correlation_id",
        "synthetic_id",
    ]);
    for r in &data.rows {
        let parent = r.row_id.to_string();
        let corr = r.correlation_id.map(|n| n.to_string()).unwrap_or_default();
        let syn = r.synthetic_id.clone().unwrap_or_default();
        let aux = &r.auxiliary;
        push_event_rows(&mut v, &parent, "cpu", &corr, &syn, &aux.cpu_events);
        push_event_rows(&mut v, &parent, "gpu", &corr, &syn, &aux.gpu_events);
        push_event_rows(&mut v, &parent, "sync", &corr, &syn, &aux.sync_events);
        push_event_rows(&mut v, &parent, "graph", &corr, &syn, &aux.graph_events);
        if aux.cpu_events.is_empty()
            && aux.gpu_events.is_empty()
            && aux.sync_events.is_empty()
            && aux.graph_events.is_empty()
        {
            v.push_row(vec![
                parent,
                "self".to_string(),
                r.row_id.to_string(),
                if r.correlation_found {
                    "(no related events)".into()
                } else {
                    "(no correlation)".into()
                },
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                corr,
                syn,
            ]);
        }
    }
    v
}

fn push_event_rows(
    v: &mut TabularView,
    parent: &str,
    side: &str,
    corr: &str,
    syn: &str,
    events: &[EventRef],
) {
    for e in events {
        let b = e.base();
        v.push_row(vec![
            parent.to_string(),
            side.to_string(),
            b.row_id.to_string(),
            b.name.clone(),
            b.start_ns.to_string(),
            b.duration_ns.to_string(),
            cell_opt(b.device_id),
            cell_opt(b.stream_id),
            cell_opt(b.global_tid),
            corr.to_string(),
            syn.to_string(),
        ]);
    }
}

/// Hardware view — one row per GPU across every host, plus
/// per-host metadata as `meta` lines. Multi-GPU traces produce
/// one row per device; single-host single-GPU traces still get a
/// row so the table is never empty when a GPU exists. NIC and
/// system/CPU info land on `meta` to keep the row schema stable
/// (a NIC list isn't comparable in shape to the GPU table).
pub fn hardware_view(data: &HardwareResponse) -> TabularView {
    let mut v = TabularView::new(vec![
        "host",
        "gpu_id",
        "gpu_name",
        "chip",
        "compute",
        "sms",
        "vram_bytes",
        "bus",
    ]);
    for host in &data.rows {
        let host_label = host
            .system
            .as_ref()
            .and_then(|s| s.hostname.clone())
            .unwrap_or_else(|| format!("host#{:04x}", host.hw_host_id));
        if host.gpus.is_empty() {
            // Surface host-only rows so a CPU-only profile still
            // appears in the table view rather than vanishing.
            v.push_row(vec![
                host_label.clone(),
                String::new(),
                "<no GPUs>".into(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
            ]);
        }
        for gpu in &host.gpus {
            let compute = match (gpu.compute_major, gpu.compute_minor) {
                (Some(a), Some(b)) => format!("{a}.{b}"),
                _ => String::new(),
            };
            v.push_row(vec![
                host_label.clone(),
                gpu.id.to_string(),
                gpu.name.clone(),
                gpu.chip_name.clone().unwrap_or_default(),
                compute,
                cell_opt(gpu.sm_count),
                cell_opt(gpu.total_memory),
                gpu.bus_location.clone().unwrap_or_default(),
            ]);
        }

        // Per-host meta lines — system/cpu/driver/nics summarised as
        // free-text so the agent (and humans) get every signal
        // without inflating the row schema.
        if let Some(sys) = &host.system {
            if let Some(ref h) = sys.hostname {
                v.push_meta(format!("{host_label}.hostname"), h.clone());
            }
            if let Some(ref os) = sys.os_description {
                v.push_meta(format!("{host_label}.os"), os.clone());
            }
            if let Some(ref k) = sys.kernel_version {
                v.push_meta(format!("{host_label}.kernel"), k.clone());
            }
        }
        if let Some(cpu) = &host.cpu {
            v.push_meta(
                format!("{host_label}.cpu"),
                match cpu.core_count {
                    Some(n) => format!("{} ({} cores)", cpu.model, n),
                    None => cpu.model.clone(),
                },
            );
        }
        if let Some(drv) = &host.drivers {
            // Parsed CUDA version is the agent-actionable one
            // (`13.0` not `13000`); fall back to raw when the
            // string isn't an integer-encoded version.
            if let Some(parsed) = drv.cuda_version_parsed() {
                v.push_meta(format!("{host_label}.cuda"), parsed);
            } else if let Some(ref raw) = drv.cuda_driver_version {
                v.push_meta(format!("{host_label}.cuda_raw"), raw.clone());
            }
            if let Some(ref nv) = drv.nv_driver_version {
                v.push_meta(format!("{host_label}.nv_driver"), nv.clone());
            }
        }
        for nic in &host.nics {
            v.push_meta(
                format!("{host_label}.nic{}", nic.id),
                format!(
                    "{} vendor={} device={}",
                    nic.name,
                    cell_opt(nic.vendor_id),
                    cell_opt(nic.device_id)
                ),
            );
        }
    }
    v.push_meta("host_count", data.rows.len().to_string());
    v
}

/// Metrics view dispatch — each variant pulls its own column set
/// because the tabular shape diverges sharply per source (GPU
/// counters, CPU hotspot, sched summary). Common envelope facts
/// (`coverage`, `trace_span_ns`, …) flow through one helper so each
/// variant's view stays focused on its own columns.
pub fn metrics_view(data: &MetricsResponse) -> TabularView {
    match data {
        MetricsResponse::Gpu(b) => gpu_metrics_view(b),
        MetricsResponse::Nic(b) => nic_metrics_view(b),
        MetricsResponse::CpuSampling(b) => cpu_sampling_metrics_view(b),
        MetricsResponse::CpuSched(b) => cpu_sched_metrics_view(b),
    }
}

fn push_common_meta(
    v: &mut TabularView,
    count: usize,
    total_matched: i64,
    common: &veloq_nsys_query::metrics::MetricsCommon,
) {
    v.push_meta("count", count.to_string());
    v.push_meta("total_matched", total_matched.to_string());
    v.push_meta("samples_total", common.coverage.samples_total.to_string());
    if let Some(gap) = common.coverage.max_gap_ns {
        v.push_meta("max_gap_ns", gap.to_string());
    }
    v.push_meta(
        "coverage",
        format!(
            "{} / {} ns ({:.*})",
            common.coverage.covered_ns,
            common.coverage.trace_ns,
            DISPLAY_PRECISION,
            common.coverage.ratio
        ),
    );
    if let Some((lo, hi)) = common.metrics_span_ns {
        v.push_meta("metrics_span_ns", format!("{lo}-{hi}"));
    }
    v.push_meta(
        "trace_span_ns",
        format!("{}-{}", common.trace_span_ns.0, common.trace_span_ns.1),
    );
    if let Some(b) = common.bucket_ns {
        v.push_meta("bucket_ns", b.to_string());
    }
    if let Some((s, e)) = common.time_window_ns {
        v.push_meta("time_window_ns", format!("{s}-{e}"));
    }
}

fn gpu_metrics_view(body: &veloq_nsys_query::metrics::GpuMetricsBody) -> TabularView {
    let mut v = if body.auxiliary.common.bucket_ns.is_some() {
        let mut v = TabularView::new(vec![
            "t_start_ns",
            "t_end_ns",
            "type_id",
            "metric_id",
            "agg",
            "value",
            "samples",
        ]);
        for b in &body.auxiliary.buckets {
            v.push_row(vec![
                b.t_start_ns.to_string(),
                b.t_end_ns.to_string(),
                b.type_id.to_string(),
                b.metric_id.to_string(),
                b.agg.to_string(),
                format!("{:.*}", DISPLAY_PRECISION, b.value),
                b.samples.to_string(),
            ]);
        }
        v
    } else {
        let mut v = TabularView::new(vec![
            "metric_id",
            "name",
            "unit",
            "agg",
            "samples",
            "min",
            "max",
            "mean",
            "p50",
            "p95",
            "p99",
        ]);
        for c in &body.rows {
            v.push_row(vec![
                c.metric_id.to_string(),
                c.name.clone(),
                c.unit.clone().unwrap_or_default(),
                c.agg.to_string(),
                c.samples.to_string(),
                format!("{:.*}", DISPLAY_PRECISION, c.min),
                format!("{:.*}", DISPLAY_PRECISION, c.max),
                format!("{:.*}", DISPLAY_PRECISION, c.mean),
                format!("{:.*}", DISPLAY_PRECISION, c.p50),
                format!("{:.*}", DISPLAY_PRECISION, c.p95),
                format!("{:.*}", DISPLAY_PRECISION, c.p99),
            ]);
        }
        v
    };
    v.push_meta("source", "gpu".to_string());
    push_common_meta(
        &mut v,
        body.count,
        body.total_matched,
        &body.auxiliary.common,
    );
    if let Some(g) = &body.auxiliary.counter_glob {
        v.push_meta("counter_glob", g.clone());
    }
    v
}

fn nic_metrics_view(body: &veloq_nsys_query::metrics::NicMetricsBody) -> TabularView {
    let mut v = if body.auxiliary.common.bucket_ns.is_some() {
        let mut v = TabularView::new(vec![
            "t_start_ns",
            "t_end_ns",
            "nic_id",
            "nic_name",
            "port_id",
            "metrics_list_id",
            "metrics_idx",
            "name",
            "unit",
            "agg",
            "value",
            "samples",
        ]);
        for b in &body.auxiliary.buckets {
            v.push_row(vec![
                b.t_start_ns.to_string(),
                b.t_end_ns.to_string(),
                b.nic_id.to_string(),
                b.nic_name.clone(),
                b.port_id.to_string(),
                b.metrics_list_id.to_string(),
                b.metrics_idx.to_string(),
                b.name.clone(),
                b.unit.clone().unwrap_or_default(),
                b.agg.to_string(),
                format!("{:.*}", DISPLAY_PRECISION, b.value),
                b.samples.to_string(),
            ]);
        }
        v
    } else {
        let mut v = TabularView::new(vec![
            "nic_id",
            "nic_name",
            "port_id",
            "metrics_list_id",
            "metrics_idx",
            "name",
            "unit",
            "agg",
            "samples",
            "min",
            "max",
            "mean",
            "p50",
            "p95",
            "p99",
        ]);
        for c in &body.rows {
            v.push_row(vec![
                c.nic_id.to_string(),
                c.nic_name.clone(),
                c.port_id.to_string(),
                c.metrics_list_id.to_string(),
                c.metrics_idx.to_string(),
                c.name.clone(),
                c.unit.clone().unwrap_or_default(),
                c.agg.to_string(),
                c.samples.to_string(),
                format!("{:.*}", DISPLAY_PRECISION, c.min),
                format!("{:.*}", DISPLAY_PRECISION, c.max),
                format!("{:.*}", DISPLAY_PRECISION, c.mean),
                format!("{:.*}", DISPLAY_PRECISION, c.p50),
                format!("{:.*}", DISPLAY_PRECISION, c.p95),
                format!("{:.*}", DISPLAY_PRECISION, c.p99),
            ]);
        }
        v
    };
    v.push_meta("source", "nic".to_string());
    push_common_meta(
        &mut v,
        body.count,
        body.total_matched,
        &body.auxiliary.common,
    );
    if let Some(g) = &body.auxiliary.counter_glob {
        v.push_meta("counter_glob", g.clone());
    }
    v
}

fn cpu_sampling_metrics_view(body: &veloq_nsys_query::metrics::CpuSamplingBody) -> TabularView {
    let mut v = if body.auxiliary.common.bucket_ns.is_some() {
        cpu_buckets_view(&body.auxiliary.cpu_buckets)
    } else {
        let mut v = TabularView::new(vec![
            "key",
            "samples",
            "percentage",
            "symbol",
            "module",
            "kernel",
            "unresolved",
            "cpu",
            "global_tid",
            "pid",
            "tid",
        ]);
        for h in &body.rows {
            v.push_row(vec![
                h.key.clone(),
                h.samples.to_string(),
                format!("{:.*}", DISPLAY_PRECISION, h.percentage),
                h.symbol_name.clone().unwrap_or_default(),
                h.module_name.clone().unwrap_or_default(),
                h.kernel_mode.map(|b| b.to_string()).unwrap_or_default(),
                h.unresolved.map(|b| b.to_string()).unwrap_or_default(),
                cell_opt(h.cpu),
                cell_opt(h.global_tid),
                cell_opt(h.pid),
                cell_opt(h.tid),
            ]);
        }
        v
    };
    v.push_meta("source", "cpu-sampling".to_string());
    push_common_meta(
        &mut v,
        body.count,
        body.total_matched,
        &body.auxiliary.common,
    );
    v.push_meta("group_by", body.auxiliary.group_by.to_string());
    if let Some(g) = &body.auxiliary.name_glob {
        v.push_meta("name_glob", g.clone());
    }
    if let Some(c) = body.auxiliary.cpu_filter {
        v.push_meta("cpu_filter", c.to_string());
    }
    if let Some(t) = body.auxiliary.tid_filter {
        v.push_meta("tid_filter", t.to_string());
    }
    if let Some(r) = body.auxiliary.unresolved_leaf_share {
        v.push_meta(
            "unresolved_leaf_share",
            format!("{r:.*}", DISPLAY_PRECISION),
        );
    }
    if let Some(r) = body.auxiliary.kernel_leaf_share {
        v.push_meta("kernel_leaf_share", format!("{r:.*}", DISPLAY_PRECISION));
    }
    if let Some(r) = body.auxiliary.truncated_stack_share {
        v.push_meta(
            "truncated_stack_share",
            format!("{r:.*}", DISPLAY_PRECISION),
        );
    }
    v
}

fn cpu_sched_metrics_view(body: &veloq_nsys_query::metrics::CpuSchedBody) -> TabularView {
    let mut v = if body.auxiliary.common.bucket_ns.is_some() {
        cpu_buckets_view(&body.auxiliary.cpu_buckets)
    } else {
        let mut v = TabularView::new(vec![
            "key",
            "on_cpu_ns",
            "off_cpu_ns",
            "ctx_switches",
            "avg_quantum_ns",
            "observed_span_ns",
            "cpu",
            "global_tid",
            "pid",
            "tid",
            "state_id",
            "state_name",
            "distinct_tids",
        ]);
        for r in &body.rows {
            v.push_row(vec![
                r.key.clone(),
                r.on_cpu_ns.to_string(),
                cell_opt(r.off_cpu_ns),
                r.ctx_switches.to_string(),
                cell_opt(r.avg_quantum_ns),
                r.observed_span_ns.to_string(),
                cell_opt(r.cpu),
                cell_opt(r.global_tid),
                cell_opt(r.pid),
                cell_opt(r.tid),
                cell_opt(r.state_id),
                r.state_name.clone().unwrap_or_default(),
                cell_opt(r.distinct_tids),
            ]);
        }
        v
    };
    v.push_meta("source", "cpu-sched".to_string());
    push_common_meta(
        &mut v,
        body.count,
        body.total_matched,
        &body.auxiliary.common,
    );
    v.push_meta("group_by", body.auxiliary.group_by.to_string());
    if let Some(c) = body.auxiliary.cpu_filter {
        v.push_meta("cpu_filter", c.to_string());
    }
    if let Some(t) = body.auxiliary.tid_filter {
        v.push_meta("tid_filter", t.to_string());
    }
    if let Some(r) = body.auxiliary.unresolved_state_share {
        v.push_meta(
            "unresolved_state_share",
            format!("{r:.*}", DISPLAY_PRECISION),
        );
    }
    if let Some(g) = body.auxiliary.per_cpu_max_gap_ns {
        v.push_meta("per_cpu_max_gap_ns", g.to_string());
    }
    v
}

fn cpu_buckets_view(buckets: &[veloq_nsys_query::metrics::CpuBucketSample]) -> TabularView {
    let mut v = TabularView::new(vec![
        "t_start_ns",
        "t_end_ns",
        "key",
        "agg",
        "value",
        "samples",
    ]);
    for b in buckets {
        v.push_row(vec![
            b.t_start_ns.to_string(),
            b.t_end_ns.to_string(),
            b.key.clone(),
            b.agg.to_string(),
            format!("{:.*}", DISPLAY_PRECISION, b.value),
            b.samples.to_string(),
        ]);
    }
    v
}

/// Key/value flattener for responses without a natural row list
/// (prep, correlation-stats). Uses serde to walk the JSON shape so we
/// don't have to keep the projection in sync with the struct definition.
pub fn key_value_view<T: serde::Serialize>(data: &T) -> TabularView {
    let value = serde_json::to_value(data).unwrap_or(serde_json::Value::Null);
    let mut v = TabularView::new(vec!["field", "value"]);
    if let serde_json::Value::Object(map) = value {
        for (k, vv) in map {
            v.push_row(vec![k, render_value(&vv)]);
        }
    } else {
        v.push_row(vec!["value".to_string(), render_value(&value)]);
    }
    v
}

fn render_value(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => String::new(),
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Array(a) => {
            // Compact: comma-joined scalars; objects/arrays fall back to JSON.
            let scalar_only = a.iter().all(|x| {
                !matches!(
                    x,
                    serde_json::Value::Array(_) | serde_json::Value::Object(_)
                )
            });
            if scalar_only {
                a.iter().map(render_value).collect::<Vec<_>>().join(", ")
            } else {
                serde_json::to_string(v).unwrap_or_default()
            }
        }
        serde_json::Value::Object(_) => serde_json::to_string(v).unwrap_or_default(),
    }
}
