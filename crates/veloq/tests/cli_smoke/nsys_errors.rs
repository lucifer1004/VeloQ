use super::{
    assert_error_code, build_graph_replay_trace, build_minimal_trace, run_veloq,
    run_veloq_with_env, run_veloq_without_unstable,
};
use anyhow::{Result, anyhow};
use serde_json::Value;
use std::process::Output;

fn argv_with_trace(args: &[&str], trace: String) -> Result<Vec<String>> {
    let mut iter = args.iter();
    let verb = iter
        .next()
        .ok_or_else(|| anyhow!("internal: nsys error smoke test missing verb"))?;
    let mut argv = Vec::with_capacity(args.len() + 1);
    argv.push((*verb).to_owned());
    argv.push(trace);
    argv.extend(iter.map(|arg| (*arg).to_owned()));
    Ok(argv)
}

fn run_with_minimal_trace(args: &[&str]) -> Result<Output> {
    let (_dir, trace) = build_minimal_trace()?;
    run_veloq(argv_with_trace(args, trace.to_string_lossy().into_owned())?)
}

fn run_with_minimal_trace_env(args: &[&str], envs: &[(&str, &str)]) -> Result<Output> {
    let (_dir, trace) = build_minimal_trace()?;
    run_veloq_with_env(
        argv_with_trace(args, trace.to_string_lossy().into_owned())?,
        envs.iter().copied(),
    )
}

fn run_with_minimal_trace_without_unstable(args: &[&str]) -> Result<Output> {
    let (_dir, trace) = build_minimal_trace()?;
    run_veloq_without_unstable(argv_with_trace(args, trace.to_string_lossy().into_owned())?)
}

fn run_with_graph_replay_trace(args: &[&str]) -> Result<Output> {
    let (_dir, trace) = build_graph_replay_trace()?;
    run_veloq(argv_with_trace(args, trace.to_string_lossy().into_owned())?)
}

fn assert_message_contains(v: &Value, needles: &[&str], reason: &str) {
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    for needle in needles {
        assert!(message.contains(needle), "{reason}: {message}");
    }
}

fn assert_chain_contains(v: &Value, needle: &str, reason: &str) {
    let chain = v
        .pointer("/error/chain")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    assert!(
        chain
            .iter()
            .filter_map(Value::as_str)
            .any(|entry| entry.contains(needle)),
        "{reason}: {chain:?}"
    );
}

#[test]
fn nsys_metrics_gpu_rejects_cpu_flags_with_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["metrics", "--name", "foo"])?;
    let v = assert_error_code(&out, "nsys.command.metrics-cpu-flag-conflict")?;
    assert_message_contains(
        &v,
        &["--type gpu", "--name"],
        "message should name the rejected CPU flag set",
    );
    Ok(())
}

#[test]
fn nsys_metrics_cpu_sampling_rejects_counter_with_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["metrics", "--type", "cpu-sampling", "--counter", "SM*"])?;
    let v = assert_error_code(&out, "nsys.command.metrics-counter-flag-conflict")?;
    assert_message_contains(
        &v,
        &["--counter", "--name"],
        "message should point cpu-sampling users at --name",
    );
    Ok(())
}

#[test]
fn nsys_metrics_cpu_sched_rejects_name_with_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["metrics", "--type", "cpu-sched", "--name", "foo"])?;
    let v = assert_error_code(&out, "nsys.command.metrics-name-flag-conflict")?;
    assert_message_contains(
        &v,
        &["--name", "cpu-sched"],
        "message should explain cpu-sched has no name field",
    );
    Ok(())
}

#[test]
fn nsys_stats_by_size_requires_unstable_with_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace_without_unstable(&["stats", "--by", "size"])?;
    let v = assert_error_code(&out, "nsys.command.unstable-feature-disabled")?;
    assert_message_contains(
        &v,
        &["VELOQ_UNSTABLE=1", "--by size"],
        "message should name the env gate and hidden flag",
    );
    Ok(())
}

#[test]
fn nsys_stats_by_size_hist_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace_env(
        &["stats", "--by", "size", "--hist"],
        &[("VELOQ_UNSTABLE", "1")],
    )?;
    let v = assert_error_code(&out, "nsys.command.stats-by-size-hist-unsupported")?;
    assert_message_contains(
        &v,
        &["--by size", "histograms"],
        "message should explain the unsupported histogram combination",
    );
    Ok(())
}

#[test]
fn nsys_stats_by_size_nvtx_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace_env(
        &["stats", "--by", "size", "--nvtx", "phase*"],
        &[("VELOQ_UNSTABLE", "1")],
    )?;
    let v = assert_error_code(&out, "nsys.command.stats-by-size-nvtx-unsupported")?;
    assert_message_contains(
        &v,
        &["--nvtx", "--by size"],
        "message should explain the unsupported NVTX combination",
    );
    Ok(())
}

#[test]
fn nsys_stats_by_size_group_by_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace_env(
        &["stats", "--by", "size", "--group-by", "short,nvtx-path"],
        &[("VELOQ_UNSTABLE", "1")],
    )?;
    let v = assert_error_code(&out, "nsys.command.stats-by-size-group-by-unsupported")?;
    assert_message_contains(
        &v,
        &["nvtx-path", "--group-by"],
        "message should name the unsupported group-by axis",
    );
    Ok(())
}

#[test]
fn nsys_slices_path_group_by_requires_aggregate_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["slices", "--group-by", "path"])?;
    let v = assert_error_code(&out, "nsys.command.slices-group-by-requires-aggregate")?;
    assert_message_contains(
        &v,
        &["--group-by path", "--aggregate"],
        "message should point users to --aggregate",
    );
    Ok(())
}

#[test]
fn nsys_metrics_unknown_source_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["metrics", "--type", "cpu"])?;
    let v = assert_error_code(&out, "nsys.command.metrics-unknown-source")?;
    assert_message_contains(
        &v,
        &["cpu", "cpu-sampling"],
        "message should name rejected and supported metric sources",
    );
    Ok(())
}

#[test]
fn nsys_metrics_invalid_bucket_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["metrics", "--bucket", "nope"])?;
    let v = assert_error_code(&out, "nsys.command.metrics-invalid-bucket")?;
    assert_message_contains(
        &v,
        &["--bucket", "nope"],
        "message should name invalid --bucket value",
    );
    assert_chain_contains(
        &v,
        "invalid --bucket",
        "chain should keep bucket parser detail",
    );
    Ok(())
}

#[test]
fn nsys_query_search_name_filter_conflict_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["search", "--name", "foo", "--name-regex", "foo"])?;
    let v = assert_error_code(&out, "nsys.query.name-filter-conflict")?;
    assert_message_contains(
        &v,
        &["--name", "--name-regex"],
        "message should name both conflicting filters",
    );
    Ok(())
}

#[test]
fn nsys_query_kind_location_filter_conflict_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["search", "--type", "nvtx", "--stream", "7"])?;
    let v = assert_error_code(&out, "nsys.query.kind-location-filter-conflict")?;
    assert_message_contains(
        &v,
        &["--stream", "nvtx"],
        "message should name the conflicting location filter and kind",
    );
    Ok(())
}

#[test]
fn nsys_query_kind_nvtx_attribution_conflict_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["stats", "--type", "nvtx", "--all-devices", "--nvtx", "*"])?;
    let v = assert_error_code(&out, "nsys.query.kind-nvtx-attribution-unsupported")?;
    assert_message_contains(
        &v,
        &["--nvtx", "experimental"],
        "message should explain the unsupported NVTX attribution request",
    );
    Ok(())
}

#[test]
fn nsys_query_stats_group_by_unknown_token_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["stats", "--group-by", "mystery"])?;
    let v = assert_error_code(&out, "nsys.query.stats-group-by-unknown-token")?;
    assert_message_contains(
        &v,
        &["mystery", "short"],
        "message should name rejected and expected stats group-by axes",
    );
    Ok(())
}

#[test]
fn nsys_query_stats_group_by_name_axis_conflict_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["stats", "--group-by", "short,demangled"])?;
    let v = assert_error_code(&out, "nsys.query.stats-group-by-name-axis-conflict")?;
    assert_message_contains(
        &v,
        &["short", "demangled"],
        "message should name both name axes",
    );
    Ok(())
}

#[test]
fn nsys_query_stats_group_by_location_axis_conflict_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&[
        "stats",
        "--type",
        "runtime",
        "--all-devices",
        "--group-by",
        "device",
    ])?;
    let v = assert_error_code(&out, "nsys.query.stats-group-by-location-axis-conflict")?;
    assert_message_contains(
        &v,
        &["--type runtime", "device"],
        "message should explain the CPU-side location-axis conflict",
    );
    Ok(())
}

#[test]
fn nsys_query_stats_grid_block_kind_conflict_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&[
        "stats",
        "--type",
        "kernel,memcpy",
        "--group-by",
        "grid_block",
    ])?;
    let v = assert_error_code(&out, "nsys.query.stats-grid-block-kind-conflict")?;
    assert_message_contains(
        &v,
        &["grid_block", "memcpy"],
        "message should name the non-kernel kind in the grid_block request",
    );
    Ok(())
}

#[test]
fn nsys_query_stats_nvtx_hierarchy_axes_conflict_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["stats", "--group-by", "nvtx-parent,nvtx-path"])?;
    let v = assert_error_code(&out, "nsys.query.stats-nvtx-hierarchy-axis-conflict")?;
    assert_message_contains(
        &v,
        &["nvtx-parent", "nvtx-path"],
        "message should name the mutually exclusive NVTX hierarchy axes",
    );
    Ok(())
}

#[test]
fn nsys_query_stats_nvtx_hierarchy_graph_conflict_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["stats", "--group-by", "nvtx-parent,graph"])?;
    let v = assert_error_code(&out, "nsys.query.stats-nvtx-hierarchy-graph-axis-conflict")?;
    assert_message_contains(
        &v,
        &["nvtx-parent", "graph"],
        "message should explain the graph/NVTX hierarchy conflict",
    );
    Ok(())
}

#[test]
fn nsys_query_stats_nvtx_hierarchy_self_attribute_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&[
        "stats",
        "--type",
        "nvtx",
        "--all-devices",
        "--group-by",
        "nvtx-parent",
    ])?;
    let v = assert_error_code(&out, "nsys.query.stats-nvtx-hierarchy-self-attribute")?;
    assert_message_contains(
        &v,
        &["--type nvtx", "self-attribute"],
        "message should explain the NVTX self-attribute request",
    );
    Ok(())
}

#[test]
fn nsys_query_stats_nvtx_hierarchy_prereq_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["stats", "--group-by", "nvtx-parent"])?;
    let v = assert_error_code(&out, "nsys.query.stats-nvtx-hierarchy-prereq-missing")?;
    assert_message_contains(
        &v,
        &["nvtx-parent", "NVTX_EVENTS"],
        "message should name the missing NVTX hierarchy prerequisite",
    );
    Ok(())
}

#[test]
fn nsys_query_timeline_interval_too_small_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["timeline", "--interval", "0ns"])?;
    let v = assert_error_code(&out, "nsys.query.timeline-interval-too-small")?;
    assert_message_contains(
        &v,
        &["--interval", "positive"],
        "message should explain the minimum interval",
    );
    Ok(())
}

#[test]
fn nsys_query_timeline_invalid_interval_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["timeline", "--interval", "bogus"])?;
    let v = assert_error_code(&out, "nsys.query.timeline-interval-invalid")?;
    assert_message_contains(
        &v,
        &["--interval", "bogus"],
        "message should name invalid --interval value",
    );
    Ok(())
}

#[test]
fn nsys_query_timeline_nvtx_prereq_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["timeline", "--interval", "1ms", "--nvtx", "*"])?;
    let v = assert_error_code(&out, "nsys.query.nvtx-attribution-prereq-missing")?;
    assert_message_contains(
        &v,
        &["--nvtx", "NVTX_EVENTS"],
        "message should name the missing NVTX attribution table",
    );
    Ok(())
}

#[test]
fn nsys_query_slices_unknown_group_by_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["slices", "--aggregate", "--group-by", "bogus"])?;
    let v = assert_error_code(&out, "nsys.query.slices-unknown-group-by")?;
    assert_message_contains(
        &v,
        &["bogus", "name"],
        "message should name rejected and expected slices group-by axes",
    );
    Ok(())
}

#[test]
fn nsys_query_slices_missing_prereq_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["slices"])?;
    let v = assert_error_code(&out, "nsys.query.slices-prereq-missing")?;
    assert_message_contains(
        &v,
        &["NVTX_EVENTS", "attribution"],
        "message should name the missing slices prerequisite table",
    );
    Ok(())
}

#[test]
fn nsys_query_metrics_sort_bucket_conflict_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["metrics", "--bucket", "1ms", "--sort", "count:desc"])?;
    let v = assert_error_code(&out, "nsys.query.metrics-sort-bucket-conflict")?;
    assert_message_contains(
        &v,
        &["--sort", "bucket"],
        "message should explain the bucket/sort conflict",
    );
    Ok(())
}

#[test]
fn nsys_query_metrics_gpu_missing_table_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["metrics"])?;
    let v = assert_error_code(&out, "nsys.query.metrics-gpu-table-missing")?;
    assert_message_contains(
        &v,
        &["GPU_METRICS", "--gpu-metrics-devices"],
        "message should name the missing GPU metrics table and capture flag",
    );
    Ok(())
}

#[test]
fn nsys_query_metrics_nic_missing_table_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["metrics", "--type", "nic"])?;
    let v = assert_error_code(&out, "nsys.query.metrics-nic-table-missing")?;
    assert_message_contains(
        &v,
        &["NET_NIC_METRIC", "--nic-metrics"],
        "message should name the missing NIC metrics table and capture flag",
    );
    Ok(())
}

#[test]
fn nsys_query_metrics_cpu_sampling_missing_table_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["metrics", "--type", "cpu-sampling"])?;
    let v = assert_error_code(&out, "nsys.query.metrics-cpu-sampling-table-missing")?;
    assert_message_contains(
        &v,
        &["COMPOSITE_EVENTS", "--sample"],
        "message should name the missing CPU sampling table and capture flag",
    );
    Ok(())
}

#[test]
fn nsys_query_metrics_cpu_sched_missing_table_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["metrics", "--type", "cpu-sched"])?;
    let v = assert_error_code(&out, "nsys.query.metrics-cpu-sched-table-missing")?;
    assert_message_contains(
        &v,
        &["SCHED_EVENTS", "--cpuctxsw"],
        "message should name the missing CPU sched table and capture flag",
    );
    Ok(())
}

#[test]
fn nsys_query_gaps_invalid_scope_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["gaps", "--scope", "whole-job"])?;
    let v = assert_error_code(&out, "nsys.query.gaps-invalid-scope")?;
    assert_message_contains(
        &v,
        &["whole-job", "device"],
        "message should name rejected and expected scopes",
    );
    Ok(())
}

#[test]
fn nsys_query_gaps_invalid_min_duration_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["gaps", "--min-duration", "bogus"])?;
    let v = assert_error_code(&out, "nsys.query.gaps-min-duration-invalid")?;
    assert_message_contains(
        &v,
        &["--min-duration", "bogus"],
        "message should name invalid --min-duration value",
    );
    Ok(())
}

#[test]
fn nsys_query_gaps_stream_scope_required_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["gaps", "--stream", "7"])?;
    let v = assert_error_code(&out, "nsys.query.gaps-stream-scope-required")?;
    assert_message_contains(
        &v,
        &["--stream", "--scope stream"],
        "message should point stream filters at stream scope",
    );
    Ok(())
}

#[test]
fn nsys_query_gaps_device_trace_conflict_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["gaps", "--scope", "trace", "--device", "0"])?;
    let v = assert_error_code(&out, "nsys.query.gaps-device-scope-conflict")?;
    assert_message_contains(
        &v,
        &["--device 0", "--scope trace"],
        "message should explain device filters conflict with trace scope",
    );
    Ok(())
}

#[test]
fn nsys_query_gaps_sort_stream_scope_required_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["gaps", "--sort", "stream"])?;
    let v = assert_error_code(&out, "nsys.query.gaps-sort-stream-scope-required")?;
    assert_message_contains(
        &v,
        &["--sort stream", "--scope stream"],
        "message should point stream sort at stream scope",
    );
    Ok(())
}

#[test]
fn nsys_query_gaps_sort_device_trace_conflict_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&[
        "gaps",
        "--scope",
        "trace",
        "--all-devices",
        "--sort",
        "device",
    ])?;
    let v = assert_error_code(&out, "nsys.query.gaps-sort-device-scope-conflict")?;
    assert_message_contains(
        &v,
        &["--sort device", "--scope trace"],
        "message should explain device sort conflicts with trace scope",
    );
    Ok(())
}

#[test]
fn nsys_query_graph_replays_top_nodes_has_specific_error_code() -> Result<()> {
    let out = run_with_graph_replay_trace(&["graph-replays", "--top-nodes", "0"])?;
    let v = assert_error_code(&out, "nsys.query.graph-replays-top-nodes-too-small")?;
    assert_message_contains(
        &v,
        &["--top-nodes", "1"],
        "message should name the minimum top-nodes value",
    );
    Ok(())
}

#[test]
fn nsys_query_graph_replays_nvtx_prereq_has_specific_error_code() -> Result<()> {
    let out = run_with_graph_replay_trace(&["graph-replays", "--nvtx", "*"])?;
    let v = assert_error_code(&out, "nsys.query.graph-replays-nvtx-prereq-missing")?;
    assert_message_contains(
        &v,
        &["graph-replays", "NVTX_EVENTS"],
        "message should name the missing graph-replays NVTX table",
    );
    Ok(())
}

#[test]
fn nsys_missing_time_bound_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["stats", "--from", "1ms"])?;
    let v = assert_error_code(&out, "nsys.command.missing-time-bound")?;
    assert_message_contains(
        &v,
        &["--from", "--to"],
        "message should name both time-bound flags",
    );
    Ok(())
}

#[test]
fn nsys_invalid_from_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["stats", "--from", "nope", "--to", "1ms"])?;
    let v = assert_error_code(&out, "nsys.command.invalid-from")?;
    assert_message_contains(
        &v,
        &["--from", "nope"],
        "message should name invalid --from value",
    );
    Ok(())
}

#[test]
fn nsys_zero_limit_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["search", "--limit", "0"])?;
    let v = assert_error_code(&out, "nsys.command.limit-too-small")?;
    assert_message_contains(&v, &["--limit", "0"], "message should name rejected limit");
    Ok(())
}

#[test]
fn nsys_unknown_event_kind_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["stats", "--type", "bogus"])?;
    let v = assert_error_code(&out, "nsys.command.unknown-event-kind")?;
    assert_message_contains(&v, &["bogus"], "message should name rejected event kind");
    Ok(())
}

#[test]
fn nsys_event_kind_not_allowed_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["timeline", "--interval", "1ms", "--type", "sync"])?;
    let v = assert_error_code(&out, "nsys.command.event-kind-not-allowed")?;
    assert_message_contains(
        &v,
        &["sync", "kernel"],
        "message should name rejected and allowed event kinds",
    );
    Ok(())
}

#[test]
fn nsys_empty_event_kind_list_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["stats", "--type", ","])?;
    let v = assert_error_code(&out, "nsys.command.empty-event-kind-list")?;
    assert_message_contains(&v, &["--type"], "message should name rejected flag");
    Ok(())
}

#[test]
fn nsys_invalid_sort_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["stats", "--sort", "total:nope"])?;
    let v = assert_error_code(&out, "nsys.command.invalid-sort")?;
    assert_message_contains(
        &v,
        &["--sort", "total:nope"],
        "message should name invalid --sort value",
    );
    assert_chain_contains(
        &v,
        "unknown sort direction",
        "chain should keep sort parser detail",
    );
    Ok(())
}

#[test]
fn nsys_invalid_duration_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["search", "--duration", "nope"])?;
    let v = assert_error_code(&out, "nsys.command.invalid-duration")?;
    assert_message_contains(
        &v,
        &["--duration", "nope"],
        "message should name invalid --duration value",
    );
    Ok(())
}

#[test]
fn nsys_invalid_row_id_has_specific_error_code() -> Result<()> {
    let out = run_with_minimal_trace(&["inspect", "no-colon"])?;
    let v = assert_error_code(&out, "nsys.command.invalid-row-id")?;
    assert_message_contains(
        &v,
        &["row_id", "no-colon"],
        "message should name invalid row_id",
    );
    Ok(())
}
