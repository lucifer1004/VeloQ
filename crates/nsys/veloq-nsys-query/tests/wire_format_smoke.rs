//! Snapshot-ish smoke tests: feed every top-level response type into
//! [`wire_format_for`] and assert the rendered output mentions the
//! canonical field names. The projector + JsonSchema derives are the
//! SSOT for response shape; this file is the cheap watchdog that
//! catches whole-branch breakage early (e.g. someone strips
//! `JsonSchema` from a struct or schemars 2.x changes the schema
//! emit and the projector loses a field).
//!
//! Assertions are deliberately loose — string contains, not exact
//! match — because the *format* of the wire-format text can evolve
//! (we may add docs inline, change spacing, etc.) without that
//! being a regression. What we don't tolerate is a canonical field
//! disappearing.

use veloq_core::wire_format::wire_format_for;
use veloq_nsys_query::correlate::CorrelateResponse;
use veloq_nsys_query::gaps::GapsResponse;
use veloq_nsys_query::graph_replays::GraphReplaysResponse;
use veloq_nsys_query::hardware::HardwareResponse;
use veloq_nsys_query::inspect::{EventDetails, InspectResponse};
use veloq_nsys_query::metrics::MetricsResponse;
use veloq_nsys_query::search::SearchResponse;
use veloq_nsys_query::slices::SlicesResponse;
use veloq_nsys_query::stats::StatsResponse;
use veloq_nsys_query::summary::Summary;
use veloq_nsys_query::timeline::TimelineResponse;
use veloq_nsys_query::viz_timeline::VizTimelineResponse;

/// Helper: every snapshot assertion checks that a list of canonical
/// field names appears somewhere in the projected text (root or
/// definitions). Fails the test verbosely with the full output if any
/// field is missing — makes debugging derive-or-projector breakage
/// quick.
fn assert_contains_all(text: &str, expected: &[&str]) {
    let missing: Vec<&&str> = expected.iter().filter(|f| !text.contains(**f)).collect();
    assert!(
        missing.is_empty(),
        "wire-format output is missing canonical fields {missing:?}\n\
         full output:\n{text}"
    );
}

#[test]
fn summary_response_projects() {
    // per_table → rows (canonical primary table); primary_time_range_ns
    // moves to envelope `trace_span`; full_time_range_ns + capabilities
    // live under `auxiliary`. The projector renders the auxiliary struct
    // by name so `SummaryAuxiliary` appears in definitions.
    let text = wire_format_for::<Summary>().render();
    assert_contains_all(
        &text,
        &[
            "schema_version",
            "product_version",
            "rows",
            "auxiliary",
            "full_time_range_ns",
            "capabilities",
        ],
    );
}

#[test]
fn stats_response_projects() {
    let text = wire_format_for::<StatsResponse>().render();
    assert_contains_all(
        &text,
        &[
            "count",
            "total_matched",
            "total_duration_ns",
            "total_events",
            "rows",
            "total_ns",
            "avg_ns",
            "p50_ns",
            "p95_ns",
            "p99_ns",
            // NVTX style group key on stats nvtx rows.
            "event_type",
            "nvtx_style",
            // nvtx-parent axis.
            "nvtx_parent_key",
            "nvtx_parent_name",
            "nvtx_parent_depth",
            "nvtx_path_key",
            "nvtx_path",
            // grid_block axis.
            "grid_x",
            "grid_y",
            "grid_z",
            "block_x",
            "block_y",
            "block_z",
        ],
    );
}

#[test]
fn search_response_projects() {
    // events → rows; key is the cross-trace join field.
    let text = wire_format_for::<SearchResponse>().render();
    assert_contains_all(
        &text,
        &[
            "count",
            "total_matched",
            "rows",
            "key",
            "row_id",
            "start_ns",
            "duration_ns",
        ],
    );
}

#[test]
fn inspect_response_projects() {
    // events → rows. EventDetails is a serde-tagged union over 13
    // variants (kernel/memcpy/.../cpu_sample + NotFound). The projector
    // emits each variant as a definition and the root references
    // EventDetails.
    let text = wire_format_for::<InspectResponse>().render();
    assert_contains_all(&text, &["rows", "EventDetails"]);
}

#[test]
fn event_details_variants_project() {
    // Each EventKind has its own *Details struct — make sure all show
    // up as named definitions in the projection (or at least the most
    // common 6). This catches "someone added a new EventKind but
    // forgot the JsonSchema derive on the new Details struct."
    let text = wire_format_for::<EventDetails>().render();
    assert_contains_all(
        &text,
        &[
            "KernelDetails",
            "MemcpyDetails",
            "MemsetDetails",
            "RuntimeDetails",
            "NvtxDetails",
            "SyncDetails",
            "CpuSampleDetails",
            "parent_row_id",
            "parent_name",
        ],
    );
}

#[test]
fn correlate_response_projects() {
    // The response carries `rows`; each CorrelateResult has a flat
    // `events: Vec<EventRef>` at top-level, with the per-kind buckets
    // (cpu_events/gpu_events/sync_events/graph_events) under
    // `auxiliary`. Agents typically iterate `events` and filter by
    // row_id prefix; the per-kind buckets stay as a convenience.
    let text = wire_format_for::<CorrelateResponse>().render();
    assert_contains_all(
        &text,
        &[
            "rows",
            "correlation_found",
            "events",
            "auxiliary",
            "cpu_events",
            "gpu_events",
            "sync_events",
            "graph_events",
            "EventRef",
        ],
    );
}

#[test]
fn gaps_response_projects() {
    // gaps → rows; key is the cross-trace join field;
    // auxiliary.streams[] surfaces per-(device, stream) busy_ratio
    // so agents can pre-filter idle streams.
    let text = wire_format_for::<GapsResponse>().render();
    assert_contains_all(
        &text,
        &[
            "min_ns",
            "count",
            "total_matched",
            "rows",
            "key",
            "duration_ns",
            "prev",
            "next",
            "auxiliary",
            "streams",
            "busy_ns",
            "span_ns",
            "busy_ratio",
        ],
    );
}

#[test]
fn graph_replays_response_projects() {
    let text = wire_format_for::<GraphReplaysResponse>().render();
    assert_contains_all(
        &text,
        &[
            "count",
            "total_matched",
            "capture_mode",
            "top_nodes_limit",
            "rows",
            "synthetic_id",
            "wall_ns",
            "sum_gpu_ns",
            "busy_ns",
            "idle_inside_replay_ns",
            "top_nodes",
            "sum_share_of_replay_wall",
        ],
    );
}

#[test]
fn timeline_response_projects() {
    // buckets → rows.
    let text = wire_format_for::<TimelineResponse>().render();
    assert_contains_all(
        &text,
        &[
            "interval_ns",
            "count",
            "rows",
            "total_ns",
            "kernel_ns",
            "memcpy_ns",
            "memset_ns",
            "graph_ns",
        ],
    );
}

#[test]
fn viz_timeline_response_projects() {
    let text = wire_format_for::<VizTimelineResponse>().render();
    assert_contains_all(
        &text,
        &[
            "count",
            "total_matched",
            "rows",
            "key",
            "path",
            "format",
            "time_window_ns",
            "track_count",
            "rendered_item_count",
            "total_item_count",
            "aggregated",
            "omitted_track_count",
            "suppressed_label_count",
            "truncated_label_count",
            "auxiliary",
            "requested_tracks",
            "resolved_tracks",
            "role",
            "render_policy",
            "label_policy",
        ],
    );
}

#[test]
fn slices_response_projects() {
    // slices → rows. `view` distinguishes instance rows from
    // aggregate rows.
    let text = wire_format_for::<SlicesResponse>().render();
    assert_contains_all(
        &text,
        &[
            "attribution",
            "view",
            "group_by",
            "rows",
            "cpu",
            "gpu_attributed",
            "attributed_kernel_ns",
            "instances",
            "attributed_total_ns",
            "p99_ns",
        ],
    );
}

#[test]
fn hardware_response_projects() {
    // hosts → rows. HostInfo references SystemInfo / DriverInfo /
    // CpuInfo / GpuInfo / NicInfo — pick a handful to make sure the
    // JsonSchema graph walks through into veloq-nsys-data.
    let text = wire_format_for::<HardwareResponse>().render();
    assert_contains_all(&text, &["rows", "GpuInfo", "CpuInfo", "NicInfo"]);
}

#[test]
fn metrics_response_projects() {
    // Each mode's summary table is `rows`; per-mode bucket arrays
    // and the shared `common` block move under `auxiliary`. The mode
    // bodies (GpuMetricsBody / NicMetricsBody / CpuSamplingBody /
    // CpuSchedBody) all hoist `count` + `total_matched` to the body
    // top level, matching the rest of the wire format.
    let text = wire_format_for::<MetricsResponse>().render();
    assert_contains_all(
        &text,
        &[
            "source",
            "trace_origin_ns",
            "trace_span_ns",
            "metrics_span_ns",
            "coverage",
            "rows",
            "auxiliary",
            "buckets",
            "cpu_buckets",
            "nic_id",
            "metrics_idx",
            "unresolved_leaf_share",
            "kernel_leaf_share",
            "truncated_stack_share",
        ],
    );
}

#[test]
fn metrics_coverage_block_carries_its_fields() {
    // The Coverage struct is the trust-signal anchor; pin its field
    // set explicitly so an accidental rename loses no info on the wire.
    let text = wire_format_for::<MetricsResponse>().render();
    assert_contains_all(&text, &["samples_total", "covered_ns", "trace_ns", "ratio"]);
}

// ---------------------------------------------------------------------------
// Contract: every primary `rows[]` element carries a stable `key`.
//
// The string-contains tests above can drift (renames, doc comments
// matching the same substring). The structural check below walks the
// schema directly: locate `properties.rows`, resolve `items` through
// `$ref` and `oneOf` (the inspect tagged union), assert every leaf
// schema declares a `key: string` property. Catches "someone added a
// new Response type and forgot the key field" without a hand-edited
// allow-list.
// ---------------------------------------------------------------------------

fn check_rows_have_key<T: schemars::JsonSchema>() -> anyhow::Result<()> {
    let type_name = std::any::type_name::<T>();
    let root = serde_json::Value::from(schemars::schema_for!(T));
    let defs = root
        .get("$defs")
        .and_then(serde_json::Value::as_object)
        .cloned()
        .unwrap_or_default();
    let resolve = |v: &serde_json::Value| -> serde_json::Value {
        let Some(refstr) = v.get("$ref").and_then(serde_json::Value::as_str) else {
            return v.clone();
        };
        let name = refstr.rsplit('/').next().unwrap_or(refstr);
        defs.get(name).cloned().unwrap_or_else(|| v.clone())
    };

    let rows = root
        .get("properties")
        .and_then(|p| p.get("rows"))
        .ok_or_else(|| anyhow::anyhow!("{type_name}: schema missing rows property"))?;
    let items = rows
        .get("items")
        .ok_or_else(|| anyhow::anyhow!("{type_name}: rows.items missing"))?;
    let item = resolve(items);

    // `oneOf` is how schemars renders serde-tagged enums (e.g.
    // EventDetails). `anyOf` covers untagged unions such as
    // SlicesResponse rows. Every variant must independently carry
    // `key`.
    let variants = item
        .get("oneOf")
        .or_else(|| item.get("anyOf"))
        .and_then(serde_json::Value::as_array);
    if let Some(variants) = variants {
        for (i, v) in variants.iter().enumerate() {
            let resolved = resolve(v);
            if resolved
                .get("properties")
                .and_then(|p| p.get("key"))
                .is_none()
            {
                anyhow::bail!(
                    "{type_name}: rows[] variant #{i} lacks a `key` field — \
                     every row must carry a stable cross-trace join key"
                );
            }
        }
        return Ok(());
    }

    if item.get("properties").and_then(|p| p.get("key")).is_none() {
        anyhow::bail!(
            "{type_name}: rows[] item lacks a `key` field — every row \
             must carry a stable cross-trace join key"
        );
    }
    Ok(())
}

#[test]
fn every_primary_rows_item_carries_key() -> anyhow::Result<()> {
    check_rows_have_key::<Summary>()?;
    check_rows_have_key::<StatsResponse>()?;
    check_rows_have_key::<SearchResponse>()?;
    check_rows_have_key::<InspectResponse>()?;
    check_rows_have_key::<CorrelateResponse>()?;
    check_rows_have_key::<GapsResponse>()?;
    check_rows_have_key::<GraphReplaysResponse>()?;
    check_rows_have_key::<TimelineResponse>()?;
    check_rows_have_key::<VizTimelineResponse>()?;
    check_rows_have_key::<SlicesResponse>()?;
    check_rows_have_key::<HardwareResponse>()?;
    // MetricsResponse is a serde-tagged union over four mode bodies;
    // its `rows` lives inside each body, not at the response root.
    // Cover the four bodies directly so the structural assertion
    // sees the rows[] schema.
    check_rows_have_key::<veloq_nsys_query::metrics::GpuMetricsBody>()?;
    check_rows_have_key::<veloq_nsys_query::metrics::NicMetricsBody>()?;
    check_rows_have_key::<veloq_nsys_query::metrics::CpuSamplingBody>()?;
    check_rows_have_key::<veloq_nsys_query::metrics::CpuSchedBody>()?;
    Ok(())
}
