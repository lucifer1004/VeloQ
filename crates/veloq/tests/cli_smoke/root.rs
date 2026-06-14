use super::{
    assert_error_code, assert_schema_envelope, build_graph_replay_trace, build_minimal_trace,
    run_veloq, run_veloq_with_env, run_veloq_with_env_and_cwd,
};
use anyhow::{Context, Result, anyhow};
use serde_json::Value;
use std::path::{Path, PathBuf};

#[test]
fn help_exits_zero() -> Result<()> {
    let out = run_veloq(["--help"])?;
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Usage:"),
        "expected clap usage line in stdout; got: {stdout}"
    );
    Ok(())
}

#[test]
fn nsys_schema_endpoint_emits_standard_meta_envelope() -> Result<()> {
    let out = run_veloq(["schema", "summary"])?;
    let _ = assert_schema_envelope(&out, "nsys.schema", "nsys", "v3", "summary")?;
    Ok(())
}

#[test]
fn nsys_graph_replays_schema_endpoint_is_registered() -> Result<()> {
    let out = run_veloq(["schema", "graph-replays"])?;
    let _ = assert_schema_envelope(&out, "nsys.schema", "nsys", "v3", "graph-replays")?;
    Ok(())
}

#[test]
fn nsys_viz_timeline_schema_endpoint_is_registered() -> Result<()> {
    let out = run_veloq(["schema", "viz.timeline"])?;
    let _ = assert_schema_envelope(&out, "nsys.schema", "nsys", "v3", "viz.timeline")?;
    Ok(())
}

#[test]
fn nsys_viz_timeline_requires_paired_window_bounds() -> Result<()> {
    for (provided, missing) in [("--from", "--to"), ("--to", "--from")] {
        let out = run_veloq(["viz", "timeline", "trace.nsys-rep", provided, "@100ms"])?;
        assert!(
            !out.status.success(),
            "single-sided {provided} should fail during clap parsing"
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.is_empty(),
            "JSON parse-error must keep stderr clean; got: {stderr}"
        );
        let v: Value =
            serde_json::from_slice(&out.stdout).context("parse-error stdout must be valid JSON")?;
        let message = v
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("missing error.message: {v}"))?;
        assert!(
            message.contains(missing),
            "single-sided {provided} should require {missing}; got: {message}"
        );
        let chain_entry = v
            .get("error")
            .and_then(|e| e.get("chain"))
            .and_then(Value::as_array)
            .and_then(|a| a.first())
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("missing error.chain[0]: {v}"))?;
        assert!(
            chain_entry.contains("MissingRequiredArgument"),
            "single-sided {provided} should be a clap missing-argument error; got: {chain_entry}"
        );
    }
    Ok(())
}

#[test]
fn agent_doctor_reports_canonical_rows() -> Result<()> {
    let out = run_veloq(["agent", "doctor"])?;
    assert!(
        out.status.success(),
        "agent doctor failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value =
        serde_json::from_slice(&out.stdout).context("agent doctor stdout must be JSON")?;
    assert_eq!(v.get("command").and_then(Value::as_str), Some("agent"));
    assert_eq!(
        v.pointer("/source/kind").and_then(Value::as_str),
        Some("veloq")
    );
    assert_eq!(v.pointer("/data/count").and_then(Value::as_u64), Some(2));
    let rows = v
        .pointer("/data/rows")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("agent doctor rows missing: {v}"))?;
    for agent in ["codex", "claude"] {
        let row = rows
            .iter()
            .find(|row| row.get("agent").and_then(Value::as_str) == Some(agent))
            .ok_or_else(|| anyhow!("missing agent row {agent}: {v}"))?;
        let expected_key = format!("agent|{agent}");
        assert_eq!(
            row.get("key").and_then(Value::as_str),
            Some(expected_key.as_str())
        );
        assert_eq!(row.get("operation").and_then(Value::as_str), Some("doctor"));
        assert!(
            matches!(
                row.get("status").and_then(Value::as_str),
                Some("ready" | "missing" | "failed")
            ),
            "unexpected agent doctor status: {row}"
        );
    }
    Ok(())
}

#[test]
fn agent_install_missing_checkout_errors_before_cli() -> Result<()> {
    let out = run_veloq([
        "agent",
        "install",
        "codex",
        "--from-checkout",
        "/no/such/veloq-checkout",
    ])?;
    let v = assert_error_code(&out, "meta.agent.package-missing")?;
    assert_eq!(v.get("command").and_then(Value::as_str), Some("agent"));
    assert!(
        String::from_utf8_lossy(&out.stderr).is_empty(),
        "JSON mode should keep stderr quiet"
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn agent_install_codex_uses_native_cli_without_leaking_child_stdout() -> Result<()> {
    let temp = tempfile::tempdir().context("create agent install tempdir")?;
    let bin_dir = temp.path().join("bin");
    std::fs::create_dir_all(&bin_dir).context("create fake bin dir")?;
    let log = temp.path().join("codex.log");
    let fake_codex = bin_dir.join("codex");
    write_executable(
        &fake_codex,
        r#"#!/bin/sh
printf '%s\n' "$*" >> "$VELOQ_FAKE_CODEX_LOG"
printf '%s\n' "child stdout must stay captured"
printf '%s\n' "child stderr must stay captured" >&2
exit 0
"#,
    )?;
    let repo = repo_root()?;
    let out = run_veloq_with_env(
        [
            "agent",
            "install",
            "codex",
            "--from-checkout",
            repo.to_string_lossy().as_ref(),
        ],
        [
            ("PATH", bin_dir.to_string_lossy().to_string()),
            ("VELOQ_FAKE_CODEX_LOG", log.to_string_lossy().to_string()),
            (
                "CODEX_HOME",
                temp.path().join("codex-home").to_string_lossy().to_string(),
            ),
        ],
    )?;
    assert!(
        out.status.success(),
        "agent install codex failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.contains("child stdout must stay captured"),
        "child stdout leaked into VeloQ stdout: {stdout}"
    );
    let v: Value = serde_json::from_slice(&out.stdout).context("agent install stdout JSON")?;
    let row = v
        .pointer("/data/rows")
        .and_then(Value::as_array)
        .and_then(|rows| rows.first())
        .ok_or_else(|| anyhow!("missing install row: {v}"))?;
    assert_eq!(row.get("agent").and_then(Value::as_str), Some("codex"));
    assert_eq!(row.get("status").and_then(Value::as_str), Some("installed"));
    assert_eq!(
        row.get("operation").and_then(Value::as_str),
        Some("install")
    );
    let commands = row
        .get("commands")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("missing commands: {row}"))?;
    assert_eq!(commands.len(), 2);

    let log = std::fs::read_to_string(&log).context("read fake codex log")?;
    assert!(
        log.contains("plugin marketplace add"),
        "marketplace add command not invoked: {log}"
    );
    assert!(
        log.contains("plugin add veloq@veloq"),
        "plugin add command not invoked: {log}"
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn agent_install_claude_uses_canonical_checkout_and_mirror() -> Result<()> {
    let temp = tempfile::tempdir().context("create agent install tempdir")?;
    let bin_dir = temp.path().join("bin");
    std::fs::create_dir_all(&bin_dir).context("create fake bin dir")?;
    let log = temp.path().join("claude.log");
    let fake_claude = bin_dir.join("claude");
    write_executable(
        &fake_claude,
        r#"#!/bin/sh
printf '%s\n' "$*" >> "$VELOQ_FAKE_CLAUDE_LOG"
printf '%s\n' "child stdout must stay captured"
printf '%s\n' "child stderr must stay captured" >&2
exit 0
"#,
    )?;
    let repo = repo_root()?;
    let marketplace: Value = serde_json::from_str(
        &std::fs::read_to_string(repo.join(".claude-plugin/marketplace.json"))
            .context("read Claude marketplace manifest")?,
    )
    .context("parse Claude marketplace manifest")?;
    assert_eq!(
        marketplace
            .pointer("/plugins/0/source")
            .and_then(Value::as_str),
        Some("./plugins/veloq")
    );
    assert_symlink_target(
        &repo.join(".agents/skills"),
        Path::new("../plugins/veloq/skills"),
    )?;
    assert_symlink_target(
        &repo.join(".codex-plugin/plugin.json"),
        Path::new("../plugins/veloq/.codex-plugin/plugin.json"),
    )?;
    assert_symlink_target(
        &repo.join(".claude-plugin/plugin.json"),
        Path::new("../plugins/veloq/.claude-plugin/plugin.json"),
    )?;
    let canonical_repo = repo
        .canonicalize()
        .with_context(|| format!("canonicalize {}", repo.display()))?;
    let out = run_veloq_with_env_and_cwd(
        ["agent", "install", "claude", "--from-checkout", "."],
        [
            ("PATH", bin_dir.to_string_lossy().to_string()),
            ("VELOQ_FAKE_CLAUDE_LOG", log.to_string_lossy().to_string()),
        ],
        &repo,
    )?;
    assert!(
        out.status.success(),
        "agent install claude failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.contains("child stdout must stay captured"),
        "child stdout leaked into VeloQ stdout: {stdout}"
    );
    let v: Value = serde_json::from_slice(&out.stdout).context("agent install stdout JSON")?;
    let row = v
        .pointer("/data/rows")
        .and_then(Value::as_array)
        .and_then(|rows| rows.first())
        .ok_or_else(|| anyhow!("missing install row: {v}"))?;
    assert_eq!(row.get("agent").and_then(Value::as_str), Some("claude"));
    assert_eq!(row.get("status").and_then(Value::as_str), Some("installed"));

    let log = std::fs::read_to_string(&log).context("read fake claude log")?;
    assert!(
        log.contains("plugin marketplace add --help"),
        "marketplace add preflight command not invoked: {log}"
    );
    assert!(
        log.contains("plugin install --help"),
        "plugin install preflight command not invoked: {log}"
    );
    let expected_marketplace = format!("plugin marketplace add {}", canonical_repo.display());
    assert!(
        log.contains(&expected_marketplace),
        "claude marketplace add should receive canonical checkout path; expected {expected_marketplace}; got: {log}"
    );
    assert!(
        log.contains("plugin install veloq@veloq"),
        "plugin install command not invoked: {log}"
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn agent_install_all_preflights_before_side_effects() -> Result<()> {
    let temp = tempfile::tempdir().context("create agent install tempdir")?;
    let bin_dir = temp.path().join("bin");
    std::fs::create_dir_all(&bin_dir).context("create fake bin dir")?;
    let log = temp.path().join("codex.log");
    let fake_codex = bin_dir.join("codex");
    write_executable(
        &fake_codex,
        r#"#!/bin/sh
printf '%s\n' "$*" >> "$VELOQ_FAKE_CODEX_LOG"
exit 0
"#,
    )?;
    let repo = repo_root()?;
    let out = run_veloq_with_env(
        [
            "agent",
            "install",
            "all",
            "--from-checkout",
            repo.to_string_lossy().as_ref(),
        ],
        [
            ("PATH", bin_dir.to_string_lossy().to_string()),
            ("VELOQ_FAKE_CODEX_LOG", log.to_string_lossy().to_string()),
            (
                "CODEX_HOME",
                temp.path().join("codex-home").to_string_lossy().to_string(),
            ),
        ],
    )?;
    let _ = assert_error_code(&out, "meta.agent.cli-missing")?;
    let log = std::fs::read_to_string(&log).context("read fake codex log")?;
    assert!(
        log.lines()
            .all(|line| line.ends_with("--help") || line == "plugin --help"),
        "preflight should run only help commands before failing; got: {log}"
    );
    assert!(
        !log.lines().any(|line| line == "plugin add veloq@veloq"),
        "codex install side effect ran before claude preflight failed: {log}"
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn agent_uninstall_codex_uses_native_cli_without_leaking_child_stdout() -> Result<()> {
    let temp = tempfile::tempdir().context("create agent uninstall tempdir")?;
    let bin_dir = temp.path().join("bin");
    std::fs::create_dir_all(&bin_dir).context("create fake bin dir")?;
    let log = temp.path().join("codex.log");
    let fake_codex = bin_dir.join("codex");
    write_executable(
        &fake_codex,
        r#"#!/bin/sh
printf '%s\n' "$*" >> "$VELOQ_FAKE_CODEX_LOG"
printf '%s\n' "child stdout must stay captured"
printf '%s\n' "child stderr must stay captured" >&2
exit 0
"#,
    )?;
    let out = run_veloq_with_env(
        ["agent", "uninstall", "codex"],
        [
            ("PATH", bin_dir.to_string_lossy().to_string()),
            ("VELOQ_FAKE_CODEX_LOG", log.to_string_lossy().to_string()),
            (
                "CODEX_HOME",
                temp.path().join("codex-home").to_string_lossy().to_string(),
            ),
        ],
    )?;
    assert!(
        out.status.success(),
        "agent uninstall codex failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.contains("child stdout must stay captured"),
        "child stdout leaked into VeloQ stdout: {stdout}"
    );
    let v: Value = serde_json::from_slice(&out.stdout).context("agent uninstall stdout JSON")?;
    let row = v
        .pointer("/data/rows")
        .and_then(Value::as_array)
        .and_then(|rows| rows.first())
        .ok_or_else(|| anyhow!("missing uninstall row: {v}"))?;
    assert_eq!(row.get("agent").and_then(Value::as_str), Some("codex"));
    assert_eq!(
        row.get("status").and_then(Value::as_str),
        Some("uninstalled")
    );
    assert_eq!(
        row.get("operation").and_then(Value::as_str),
        Some("uninstall")
    );
    let commands = row
        .get("commands")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("missing commands: {row}"))?;
    assert_eq!(commands.len(), 1);

    let log = std::fs::read_to_string(&log).context("read fake codex log")?;
    assert!(
        log.contains("plugin remove --help"),
        "uninstall preflight command not invoked: {log}"
    );
    assert!(
        log.contains("plugin remove veloq@veloq"),
        "plugin remove command not invoked: {log}"
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn agent_uninstall_all_preflights_before_side_effects() -> Result<()> {
    let temp = tempfile::tempdir().context("create agent uninstall tempdir")?;
    let bin_dir = temp.path().join("bin");
    std::fs::create_dir_all(&bin_dir).context("create fake bin dir")?;
    let log = temp.path().join("codex.log");
    let fake_codex = bin_dir.join("codex");
    write_executable(
        &fake_codex,
        r#"#!/bin/sh
printf '%s\n' "$*" >> "$VELOQ_FAKE_CODEX_LOG"
exit 0
"#,
    )?;
    let out = run_veloq_with_env(
        ["agent", "uninstall", "all"],
        [
            ("PATH", bin_dir.to_string_lossy().to_string()),
            ("VELOQ_FAKE_CODEX_LOG", log.to_string_lossy().to_string()),
            (
                "CODEX_HOME",
                temp.path().join("codex-home").to_string_lossy().to_string(),
            ),
        ],
    )?;
    let _ = assert_error_code(&out, "meta.agent.cli-missing")?;
    let log = std::fs::read_to_string(&log).context("read fake codex log")?;
    assert_eq!(
        log.trim(),
        "plugin remove --help",
        "uninstall preflight should run only the operation-specific help command before failing; got: {log}"
    );
    assert!(
        !log.lines().any(|line| line == "plugin remove veloq@veloq"),
        "codex uninstall side effect ran before claude preflight failed: {log}"
    );
    Ok(())
}

fn repo_root() -> Result<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| anyhow!("failed to resolve repo root from {}", manifest.display()))
}

#[cfg(unix)]
fn assert_symlink_target(path: &Path, expected: &Path) -> Result<()> {
    let metadata =
        std::fs::symlink_metadata(path).with_context(|| format!("stat {}", path.display()))?;
    assert!(
        metadata.file_type().is_symlink(),
        "expected symlink at {}",
        path.display()
    );
    let actual =
        std::fs::read_link(path).with_context(|| format!("readlink {}", path.display()))?;
    assert_eq!(
        actual,
        expected,
        "unexpected symlink target for {}",
        path.display()
    );
    Ok(())
}

#[cfg(unix)]
fn write_executable(path: &Path, body: &str) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    std::fs::write(path, body).with_context(|| format!("write {}", path.display()))?;
    let mut permissions = std::fs::metadata(path)
        .with_context(|| format!("stat {}", path.display()))?
        .permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(path, permissions)
        .with_context(|| format!("chmod {}", path.display()))?;
    Ok(())
}

#[test]
fn nsys_viz_timeline_writes_svg_artifact() -> Result<()> {
    let (_trace_dir, trace) = build_minimal_trace()?;
    let trace_arg = trace.to_string_lossy();
    let out = run_veloq([
        "viz",
        "timeline",
        trace_arg.as_ref(),
        "--from",
        "@100ms",
        "--to",
        "@102ms",
        "--highlight-kernels",
        "top=1,scope=name",
    ])?;
    assert!(
        out.status.success(),
        "viz timeline failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value =
        serde_json::from_slice(&out.stdout).context("viz timeline stdout must be JSON")?;
    assert_eq!(
        v.get("command").and_then(Value::as_str),
        Some("nsys.viz.timeline")
    );
    let row = v
        .pointer("/data/rows/0")
        .ok_or_else(|| anyhow!("missing figure row: {v}"))?;
    let path = row
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing figure path: {row}"))?;
    assert!(
        path.starts_with("figures/nsys/timeline/") && path.ends_with(".svg"),
        "figure path must be artifact-root relative SVG path, got {path}"
    );
    assert_eq!(row.get("format").and_then(Value::as_str), Some("svg"));
    let highlight = v
        .pointer("/data/auxiliary/resolved_highlights/0")
        .ok_or_else(|| anyhow!("missing resolved highlight: {v}"))?;
    assert_eq!(
        highlight.get("label").and_then(Value::as_str),
        Some("smoke_kernel")
    );
    assert_eq!(highlight.get("scope").and_then(Value::as_str), Some("name"));
    assert_eq!(
        highlight.get("metric").and_then(Value::as_str),
        Some("total_duration_ns")
    );
    assert_eq!(
        highlight.get("score").and_then(Value::as_i64),
        Some(1_000_000)
    );
    assert_eq!(
        highlight.get("score_total").and_then(Value::as_i64),
        Some(1_000_000)
    );
    assert!(
        row.get("rendered_item_count")
            .and_then(Value::as_u64)
            .unwrap_or_default()
            > 0,
        "figure should render at least one item: {row}"
    );
    assert!(
        veloq_core::artifact_dir_for(&trace).join(path).exists(),
        "SVG artifact should exist under the trace artifact root"
    );
    Ok(())
}

#[test]
fn nsys_namespace_routes_default_source_verbs() -> Result<()> {
    let (_trace_dir, trace) = build_minimal_trace()?;
    let out = run_veloq(["nsys", "summary", &trace.to_string_lossy()])?;
    assert!(
        out.status.success(),
        "nsys summary failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value =
        serde_json::from_slice(&out.stdout).context("nsys summary stdout must be valid JSON")?;
    assert_eq!(
        v.get("command").and_then(Value::as_str),
        Some("nsys.summary"),
    );
    assert_eq!(
        v.get("source")
            .and_then(|s| s.get("kind"))
            .and_then(Value::as_str),
        Some("nsys"),
    );

    let schema = run_veloq(["nsys", "schema", "summary"])?;
    let _ = assert_schema_envelope(&schema, "nsys.schema", "nsys", "v3", "summary")?;
    Ok(())
}

#[test]
fn nsys_graph_replays_cli_renders_json_table_and_csv() -> Result<()> {
    let (_trace_dir, trace) = build_graph_replay_trace()?;
    let trace_arg = trace.to_string_lossy();

    let json = run_veloq(["graph-replays", trace_arg.as_ref(), "--limit", "1"])?;
    assert!(
        json.status.success(),
        "graph-replays JSON failed: stderr={}",
        String::from_utf8_lossy(&json.stderr)
    );
    let v: Value = serde_json::from_slice(&json.stdout)?;
    assert_eq!(
        v.get("command").and_then(Value::as_str),
        Some("nsys.graph-replays")
    );
    assert_eq!(
        v.pointer("/data/rows/0/capture_mode")
            .and_then(Value::as_str),
        Some("graph_trace")
    );

    let table = run_veloq([
        "--format",
        "table",
        "graph-replays",
        trace_arg.as_ref(),
        "--limit",
        "1",
    ])?;
    assert!(table.status.success());
    let table_stdout = String::from_utf8_lossy(&table.stdout);
    assert!(table_stdout.contains("graph_trace"));

    let csv = run_veloq([
        "--format",
        "csv",
        "graph-replays",
        trace_arg.as_ref(),
        "--limit",
        "1",
    ])?;
    assert!(csv.status.success());
    let csv_stdout = String::from_utf8_lossy(&csv.stdout);
    assert!(csv_stdout.contains("synthetic_id"));
    assert!(csv_stdout.contains("graph_trace"));
    Ok(())
}

#[test]
fn nsys_schema_endpoint_covers_cli_side_payloads() -> Result<()> {
    for target in ["prep", "correlation-stats", "ncu-command"] {
        let out = run_veloq(["schema", target])?;
        let _ = assert_schema_envelope(&out, "nsys.schema", "nsys", "v3", target)?;
    }
    Ok(())
}

#[test]
fn nsys_summary_happy_path_emits_full_envelope() -> Result<()> {
    let (_trace_dir, trace) = build_minimal_trace()?;
    let out = run_veloq(["summary", &trace.to_string_lossy()])?;
    assert!(
        out.status.success(),
        "summary failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value =
        serde_json::from_slice(&out.stdout).context("summary stdout must be valid JSON")?;
    // envelope: schema/source/command/trace/data are required.
    // trace_span is optional — present iff the source could resolve a
    // primary time range from the meta-cache sidecar.
    for key in ["schema", "source", "command", "trace", "data"] {
        assert!(
            v.get(key).is_some(),
            "summary envelope missing `{key}`: {v}"
        );
    }
    assert_eq!(
        v.get("schema").and_then(Value::as_str),
        Some("v1"),
        "envelope schema must be `v1`",
    );
    assert_eq!(
        v.get("command").and_then(Value::as_str),
        Some("nsys.summary"),
    );
    assert_eq!(
        v.get("source")
            .and_then(|s| s.get("kind"))
            .and_then(Value::as_str),
        Some("nsys"),
    );
    assert_eq!(
        v.get("source")
            .and_then(|s| s.get("version"))
            .and_then(Value::as_str),
        Some("v3"),
    );
    assert_eq!(
        v.get("trace")
            .and_then(|t| t.get("kind"))
            .and_then(Value::as_str),
        Some("nsys"),
    );
    assert_eq!(
        v.get("trace")
            .and_then(|t| t.get("path"))
            .and_then(Value::as_str),
        Some(trace.to_string_lossy().as_ref())
    );
    Ok(())
}

#[test]
fn missing_trace_in_json_mode_emits_error_envelope_with_quiet_stderr() -> Result<()> {
    // JSON is the documented default. Under the agent contract the
    // stdout envelope is the single source of truth; stderr stays
    // quiet so agents don't have to dedupe a "veloq: …" mirror.
    let out = run_veloq(["summary", "/nonexistent.sqlite"])?;
    assert!(
        !out.status.success(),
        "missing trace should yield non-zero exit"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.is_empty(),
        "JSON mode must keep stderr clean; got: {stderr}"
    );
    let v = assert_error_code(&out, "nsys.data.sqlite-input-unsupported")?;
    let error = v
        .get("error")
        .ok_or_else(|| anyhow!("stdout envelope missing `error`: {v}"))?;
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("error.message missing: {v}"))?;
    assert!(
        message.contains("/nonexistent.sqlite"),
        "error.message should mention the trace path; got: {message}"
    );
    assert!(
        v.get("data").is_none(),
        "error envelope must not carry `data`: {v}"
    );
    assert_eq!(
        v.get("command").and_then(Value::as_str),
        Some("nsys.summary"),
    );
    Ok(())
}

#[test]
fn missing_trace_in_table_mode_mirrors_error_on_stderr() -> Result<()> {
    // Explicit `--format=table` is human-targeted; keep the stderr
    // mirror so terminal users see the cause without parsing JSON.
    let out = run_veloq(["--format", "table", "summary", "/nonexistent.sqlite"])?;
    assert!(
        !out.status.success(),
        "missing trace should yield non-zero exit"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.starts_with("veloq:"),
        "table mode must mirror `veloq: …` to stderr; got: {stderr}"
    );
    assert!(
        stderr.contains("/nonexistent.sqlite"),
        "stderr should mention the trace path; got: {stderr}"
    );
    Ok(())
}

#[test]
fn bogus_subcommand_routes_through_envelope() -> Result<()> {
    let out = run_veloq(["definitely-not-a-command"])?;
    assert!(
        !out.status.success(),
        "bogus subcommand should yield non-zero exit"
    );
    // JSON is the parse-error default — stderr stays quiet; the
    // unrecognized subcommand surfaces inside the stdout envelope's
    // error.message instead.
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.is_empty(),
        "JSON parse-error must keep stderr clean; got: {stderr}"
    );
    // stdout: envelope with `error.chain` mentioning clap's ErrorKind.
    let v: Value =
        serde_json::from_slice(&out.stdout).context("parse-error stdout must be valid JSON")?;
    let error = v
        .get("error")
        .ok_or_else(|| anyhow!("missing error: {v}"))?;
    let chain_entry = error
        .get("chain")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing chain[0]: {v}"))?;
    assert!(
        chain_entry.contains("InvalidSubcommand"),
        "chain[0] should mention clap::ErrorKind::InvalidSubcommand; got: {chain_entry}"
    );
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing error.message: {v}"))?;
    assert!(
        message.contains("definitely-not-a-command"),
        "error.message should echo the unrecognized subcommand; got: {message}"
    );
    Ok(())
}

#[test]
fn table_mode_parse_error_mirrors_error_on_stderr() -> Result<()> {
    let out = run_veloq(["--format", "table", "definitely-not-a-command"])?;
    assert!(
        !out.status.success(),
        "bogus subcommand should yield non-zero exit"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.starts_with("veloq:"),
        "table parse-error must mirror `veloq: ...` to stderr; got: {stderr}"
    );

    let v: Value =
        serde_json::from_slice(&out.stdout).context("parse-error stdout must be valid JSON")?;
    let message = v
        .get("error")
        .and_then(|e| e.get("message"))
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing error.message: {v}"))?;
    assert!(
        message.contains("definitely-not-a-command"),
        "error.message should echo the unrecognized subcommand; got: {message}"
    );
    Ok(())
}

#[test]
fn format_csv_dispatch_changes_stdout_shape() -> Result<()> {
    let (_trace_dir, trace) = build_minimal_trace()?;
    let out = run_veloq(["--format", "csv", "summary", &trace.to_string_lossy()])?;
    assert!(
        out.status.success(),
        "summary --format csv failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    // CSV output is comma-separated key=value-style lines, not JSON.
    // The cheapest invariant: not parseable as a JSON envelope.
    assert!(
        serde_json::from_str::<Value>(&stdout).is_err(),
        "--format csv must not emit JSON envelope; got: {stdout}"
    );
    Ok(())
}

#[test]
fn nsys_schema_bad_target_omits_trace_field() -> Result<()> {
    // Regression test for `veloq schema <bad-target>` fabricating
    // `envelope.trace.path == ""` instead of omitting `trace` on the
    // error envelope (the success envelope omitted it correctly; the
    // failure path used the now-replaced `Cmd::trace_path -> &Path`
    // that returned `Path::new("")` for trace-less verbs).
    let out = run_veloq(["schema", "definitely-not-a-target"])?;
    assert!(
        !out.status.success(),
        "schema with bogus target should exit non-zero"
    );
    let v: Value =
        serde_json::from_slice(&out.stdout).context("schema error stdout must be valid JSON")?;
    // Qualified verb name + nsys source kind even on failure.
    assert_eq!(
        v.get("command").and_then(Value::as_str),
        Some("nsys.schema"),
    );
    assert_eq!(
        v.get("source")
            .and_then(|s| s.get("kind"))
            .and_then(Value::as_str),
        Some("nsys"),
    );
    assert_eq!(
        v.get("source")
            .and_then(|s| s.get("version"))
            .and_then(Value::as_str),
        Some("v3"),
    );
    // The bug: `trace` was present with `path: ""`. Fixed contract:
    // schema is a meta endpoint with no trace, so the field must be
    // absent on both success and failure.
    assert!(
        v.get("trace").is_none(),
        "schema error envelope must omit `trace`: {v}"
    );
    assert_eq!(
        v.pointer("/error/code").and_then(Value::as_str),
        Some("nsys.command.unknown-schema-target")
    );
    // Sanity: the error chain actually mentions the bogus target.
    let message = v
        .get("error")
        .and_then(|e| e.get("message"))
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing error.message: {v}"))?;
    assert!(
        message.contains("definitely-not-a-target"),
        "error.message should echo the bad target name; got: {message}"
    );
    Ok(())
}
