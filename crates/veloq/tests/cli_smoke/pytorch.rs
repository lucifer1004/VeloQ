use super::{assert_error_code, run_veloq};
use anyhow::{Context, Result, anyhow};
use serde_json::Value;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

fn write_multi_rank_pytorch_trace(dir: &TempDir) -> Result<PathBuf> {
    let path = dir.path().join("multi_rank.pt.trace.json");
    std::fs::write(
        &path,
        r#"{
  "traceEvents": [
    { "name": "c10d::allreduce", "cat": "cpu_op", "ph": "X", "ts": 100, "dur": 100, "pid": 1, "tid": 10, "args": { "External id": 8, "rank": 0 } },
    { "name": "ncclDevKernel_AllReduce", "cat": "kernel", "ph": "X", "ts": 200, "dur": 200, "pid": 1, "tid": 8, "args": { "External id": 8, "device": 0, "stream": 8, "rank": 0 } },
    { "name": "c10d::allreduce", "cat": "cpu_op", "ph": "X", "ts": 1000, "dur": 100, "pid": 1, "tid": 11, "args": { "External id": 9, "rank": 1 } },
    { "name": "ncclDevKernel_AllReduce", "cat": "kernel", "ph": "X", "ts": 1100, "dur": 200, "pid": 1, "tid": 9, "args": { "External id": 9, "device": 0, "stream": 9, "rank": 1 } }
  ]
}"#,
    )
    .context("write multi-rank pytorch trace")?;
    Ok(path)
}

fn assert_pytorch_rank_scope_error(
    out: &std::process::Output,
    command: &str,
    trace_arg: &str,
) -> Result<Value> {
    let v = assert_error_code(out, "pytorch.query.rank-scope-required")?;
    assert!(
        out.stderr.is_empty(),
        "json error mode should keep stderr empty; stderr={}",
        String::from_utf8_lossy(&out.stderr),
    );
    assert_eq!(
        v.pointer("/command").and_then(Value::as_str),
        Some(command),
        "got: {v}",
    );
    let hint = v
        .pointer("/error/hint")
        .and_then(Value::as_str)
        .context("error.hint must be a string")?;
    assert!(
        hint.contains("--all-ranks") && hint.contains("--rank 0"),
        "hint must mention both recovery flags: {hint}",
    );
    assert_eq!(
        v.pointer("/meta/warnings/0/code").and_then(Value::as_str),
        Some("multi-rank-ambiguous"),
        "got: {v}",
    );
    let verb = command
        .strip_prefix("pytorch.")
        .ok_or_else(|| anyhow!("pytorch command must be qualified: {command}"))?;
    let aggregate_command = v
        .pointer("/meta/next_steps/0/command")
        .and_then(Value::as_str)
        .context("first next_steps command must be a string")?;
    assert!(
        aggregate_command.contains(&format!("veloq pytorch {verb}"))
            && aggregate_command.contains(trace_arg)
            && aggregate_command.contains("--all-ranks"),
        "aggregate next step should rerun the pytorch query: {aggregate_command}",
    );
    let rank_command = v
        .pointer("/meta/next_steps/1/command")
        .and_then(Value::as_str)
        .context("second next_steps command must be a string")?;
    assert!(
        rank_command.contains(&format!("veloq pytorch {verb}"))
            && rank_command.contains(trace_arg)
            && rank_command.contains("--rank 0"),
        "rank next step should rerun the pytorch query: {rank_command}",
    );
    Ok(v)
}

#[test]
fn pytorch_schema_endpoint_emits_envelope_without_trace() -> Result<()> {
    let out = run_veloq(["pytorch", "schema", "summary"])?;
    assert!(
        out.status.success(),
        "pytorch schema summary should succeed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value =
        serde_json::from_slice(&out.stdout).context("pytorch schema stdout must be valid JSON")?;
    assert_eq!(
        v.get("command").and_then(Value::as_str),
        Some("pytorch.schema"),
    );
    assert_eq!(
        v.get("source")
            .and_then(|s| s.get("kind"))
            .and_then(Value::as_str),
        Some("pytorch"),
    );
    assert_eq!(
        v.get("source")
            .and_then(|s| s.get("version"))
            .and_then(Value::as_str),
        Some("v0"),
    );
    assert!(
        v.get("trace").is_none(),
        "pytorch schema envelope must omit trace: {v}"
    );
    assert_eq!(
        v.get("data")
            .and_then(|d| d.get("target"))
            .and_then(Value::as_str),
        Some("summary"),
    );
    assert!(
        v.get("data").and_then(|d| d.get("schema")).is_some(),
        "pytorch schema response missing schema document: {v}"
    );
    Ok(())
}

#[test]
fn pytorch_summary_emits_source_identity_and_trace_rows() -> Result<()> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let trace = write_multi_rank_pytorch_trace(&dir)?;
    let trace_arg = trace.to_string_lossy().into_owned();
    let out = run_veloq(["pytorch", "summary", trace_arg.as_str()])?;
    assert!(
        out.status.success(),
        "pytorch summary failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value =
        serde_json::from_slice(&out.stdout).context("pytorch summary stdout must be valid JSON")?;
    assert_eq!(
        v.get("command").and_then(Value::as_str),
        Some("pytorch.summary"),
    );
    assert_eq!(
        v.get("source")
            .and_then(|s| s.get("kind"))
            .and_then(Value::as_str),
        Some("pytorch"),
    );
    assert_eq!(
        v.get("source")
            .and_then(|s| s.get("version"))
            .and_then(Value::as_str),
        Some("v0"),
    );
    assert_eq!(
        v.pointer("/trace/path").and_then(Value::as_str),
        Some(trace_arg.as_str()),
    );
    assert!(
        v.get("trace_span").is_some(),
        "pytorch summary should expose trace_span after ingest: {v}"
    );
    let rows = v
        .pointer("/data/rows")
        .and_then(Value::as_array)
        .context("summary rows must be an array")?;
    let row = rows.first().context("summary must return one trace row")?;
    let expected_key = format!("trace|{trace_arg}");
    assert_eq!(
        row.get("key").and_then(Value::as_str),
        Some(expected_key.as_str()),
    );
    assert_eq!(
        v.pointer("/data/auxiliary/capabilities/rank_count")
            .and_then(Value::as_u64),
        Some(2),
    );
    assert_eq!(
        v.pointer("/data/total_matched").and_then(Value::as_u64),
        Some(1),
    );
    Ok(())
}

#[test]
fn pytorch_search_and_stats_rank_scope_errors_are_recoverable() -> Result<()> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let trace = write_multi_rank_pytorch_trace(&dir)?;
    let trace_arg = trace.to_string_lossy().into_owned();

    let search = run_veloq(["pytorch", "search", trace_arg.as_str()])?;
    assert_pytorch_rank_scope_error(&search, "pytorch.search", trace_arg.as_str())?;

    let stats = run_veloq(["pytorch", "stats", trace_arg.as_str()])?;
    assert_pytorch_rank_scope_error(&stats, "pytorch.stats", trace_arg.as_str())?;
    Ok(())
}

#[test]
fn pytorch_collectives_rank_scope_error_is_recoverable() -> Result<()> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let trace = write_multi_rank_pytorch_trace(&dir)?;
    let trace_arg = trace.to_string_lossy().into_owned();
    let out = run_veloq(["pytorch", "collectives", trace_arg.as_str()])?;
    assert_pytorch_rank_scope_error(&out, "pytorch.collectives", trace_arg.as_str())?;
    Ok(())
}

#[test]
fn pytorch_prep_status_reports_without_building_sidecars() -> Result<()> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let trace = write_multi_rank_pytorch_trace(&dir)?;
    let trace_arg = trace.to_string_lossy().into_owned();
    let out = run_veloq(["pytorch", "prep", trace_arg.as_str(), "--status"])?;
    assert!(
        out.status.success(),
        "pytorch prep --status failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value =
        serde_json::from_slice(&out.stdout).context("pytorch prep stdout must be valid JSON")?;
    assert_eq!(
        v.get("command").and_then(Value::as_str),
        Some("pytorch.prep"),
    );
    assert_eq!(
        v.pointer("/data/auxiliary/built").and_then(Value::as_bool),
        Some(false),
    );
    assert_eq!(
        v.pointer("/data/auxiliary/cache_fresh")
            .and_then(Value::as_bool),
        Some(false),
    );
    let rows = v
        .pointer("/data/rows")
        .and_then(Value::as_array)
        .context("prep rows must be an array")?;
    assert!(
        rows.iter()
            .all(|row| row.get("present").and_then(Value::as_bool) == Some(false)),
        "prep --status must not build sidecars: {v}"
    );
    let artifact_dir = v
        .pointer("/data/auxiliary/artifact_dir")
        .and_then(Value::as_str)
        .context("prep auxiliary must carry artifact_dir")?;
    assert!(
        !Path::new(artifact_dir).exists(),
        "prep --status should not create artifact dir: {artifact_dir}"
    );
    Ok(())
}

#[test]
fn pytorch_input_errors_emit_handled_envelopes() -> Result<()> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let unsupported = dir.path().join("trace.json");
    std::fs::write(&unsupported, r#"{ "traceEvents": [] }"#)?;
    let unsupported_arg = unsupported.to_string_lossy().into_owned();
    let unsupported_out = run_veloq(["pytorch", "summary", unsupported_arg.as_str()])?;
    let unsupported_v = assert_error_code(&unsupported_out, "pytorch.input.unsupported-extension")?;
    assert_eq!(
        unsupported_v.get("command").and_then(Value::as_str),
        Some("pytorch.summary"),
    );
    assert!(
        unsupported_out.stderr.is_empty(),
        "json handled errors should keep stderr empty; stderr={}",
        String::from_utf8_lossy(&unsupported_out.stderr)
    );

    let invalid = dir.path().join("invalid.pt.trace.json");
    std::fs::write(&invalid, "{")?;
    let invalid_arg = invalid.to_string_lossy().into_owned();
    let invalid_out = run_veloq(["pytorch", "summary", invalid_arg.as_str()])?;
    let invalid_v = assert_error_code(&invalid_out, "pytorch.trace.parse-json")?;
    assert_eq!(
        invalid_v.get("command").and_then(Value::as_str),
        Some("pytorch.summary"),
    );

    let directory_arg = dir.path().to_string_lossy().into_owned();
    let directory_out = run_veloq(["pytorch", "summary", directory_arg.as_str()])?;
    let directory_v = assert_error_code(&directory_out, "pytorch.input.directory-unsupported")?;
    assert_eq!(
        directory_v.get("command").and_then(Value::as_str),
        Some("pytorch.summary"),
    );
    Ok(())
}
