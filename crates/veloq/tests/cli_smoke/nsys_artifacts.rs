use super::{build_minimal_trace, run_veloq, run_veloq_with_env};
use anyhow::{Context, Result, anyhow};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Output;
use tempfile::TempDir;

/// Build the `.nsys-rep` + generated `<report>.veloq/parquetdir/`
/// shape that appears after the first run on a report.
fn build_generated_parquetdir_alias() -> Result<(TempDir, PathBuf, PathBuf)> {
    let (dir, direct_pqtdir) = build_minimal_trace()?;
    let report = dir.path().join("trace.nsys-rep");
    std::fs::write(&report, b"source").context("write report placeholder")?;
    let generated_pqtdir = veloq_core::artifact_dir_for(&report).join("parquetdir");
    copy_dir(&direct_pqtdir, &generated_pqtdir)?;
    Ok((dir, report, generated_pqtdir))
}

fn copy_dir(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst).with_context(|| format!("create {}", dst.display()))?;
    for entry in std::fs::read_dir(src).with_context(|| format!("read {}", src.display()))? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir(&src_path, &dst_path)?;
        } else {
            std::fs::copy(&src_path, &dst_path).with_context(|| {
                format!("copy {} to {}", src_path.display(), dst_path.display())
            })?;
        }
    }
    Ok(())
}

#[test]
fn info_probes_capabilities_for_parquetdir_traces() -> Result<()> {
    // `info` reports the cheap probe: source detection + filesystem
    // facts + (for `_pqtdir/` NSys traces) the same capability bitmap
    // `summary.auxiliary.capabilities` carries — computed via parquet
    // file stats, no DuckDB open.
    let (_trace_dir, trace) = build_minimal_trace()?;
    let trace_path = trace.to_string_lossy().to_string();
    let out = run_veloq(["info", &trace_path])?;
    assert!(
        out.status.success(),
        "info failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value = serde_json::from_slice(&out.stdout).context("info stdout must be valid JSON")?;
    let data = v.get("data").ok_or_else(|| anyhow!("missing data: {v}"))?;
    assert_eq!(
        data.get("detected_source").and_then(Value::as_str),
        Some("nsys"),
        "minimal parquetdir trace must detect as nsys: {data}"
    );
    assert_eq!(
        data.get("exists").and_then(Value::as_bool),
        Some(true),
        "trace must exist on disk: {data}"
    );
    let caps = data
        .get("capabilities")
        .ok_or_else(|| anyhow!("info missing capabilities for parquetdir NSys: {data}"))?;
    assert_eq!(
        caps.get("has_kernels").and_then(Value::as_bool),
        Some(true),
        "fixture has CUPTI_ACTIVITY_KIND_KERNEL: {caps}"
    );
    assert_eq!(
        caps.get("has_nic_metrics").and_then(Value::as_bool),
        Some(false),
        "fixture has no NIC tables — capability must be false: {caps}"
    );
    Ok(())
}

#[test]
fn info_probes_capabilities_for_generated_parquetdir_alias() -> Result<()> {
    let (_trace_dir, _report, generated_pqtdir) = build_generated_parquetdir_alias()?;
    let trace_path = generated_pqtdir.to_string_lossy().to_string();
    let out = run_veloq(["info", &trace_path])?;
    assert!(
        out.status.success(),
        "info failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value = serde_json::from_slice(&out.stdout).context("info stdout must be valid JSON")?;
    let data = v.get("data").ok_or_else(|| anyhow!("missing data: {v}"))?;
    assert_eq!(
        data.get("detected_source").and_then(Value::as_str),
        Some("nsys"),
        "generated parquetdir alias must detect as nsys: {data}"
    );
    let caps = data
        .get("capabilities")
        .ok_or_else(|| anyhow!("info missing capabilities for generated parquetdir: {data}"))?;
    assert_eq!(
        caps.get("has_kernels").and_then(Value::as_bool),
        Some(true),
        "generated parquetdir alias should probe table presence: {caps}"
    );
    Ok(())
}

#[test]
fn info_does_not_detect_orphan_generated_parquetdir_alias() -> Result<()> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let generated_pqtdir = dir.path().join("missing.nsys-rep.veloq/parquetdir");
    std::fs::create_dir_all(&generated_pqtdir).context("create generated parquetdir")?;
    std::fs::write(
        generated_pqtdir.join("CUPTI_ACTIVITY_KIND_KERNEL.parquet"),
        b"not parquet",
    )
    .context("write placeholder table")?;

    let trace_path = generated_pqtdir.to_string_lossy().to_string();
    let out = run_veloq(["info", &trace_path])?;
    assert!(
        out.status.success(),
        "info should stay a cheap filesystem probe: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value = serde_json::from_slice(&out.stdout).context("info stdout must be valid JSON")?;
    let data = v.get("data").ok_or_else(|| anyhow!("missing data: {v}"))?;
    assert!(
        data.get("detected_source").is_none(),
        "orphan generated parquetdir must not claim nsys: {data}"
    );
    assert_eq!(
        data.get("exists").and_then(Value::as_bool),
        Some(true),
        "info still reports the inspected path exists: {data}"
    );
    assert!(
        data.get("capabilities").is_none(),
        "orphan generated parquetdir must not emit capabilities: {data}"
    );
    Ok(())
}

#[test]
fn info_omits_capabilities_for_missing_trace() -> Result<()> {
    // A non-existent path detects as nsys (extension match) but the
    // capability probe is gated on `exists` — the response should
    // omit the field entirely rather than emit an all-false bitmap.
    let out = run_veloq(["info", "/nonexistent.sqlite"])?;
    assert!(
        out.status.success(),
        "info should succeed even on missing trace"
    );
    let v: Value = serde_json::from_slice(&out.stdout).context("info stdout must be valid JSON")?;
    let data = v.get("data").ok_or_else(|| anyhow!("missing data: {v}"))?;
    assert_eq!(data.get("exists").and_then(Value::as_bool), Some(false));
    assert!(
        data.get("capabilities").is_none(),
        "missing trace must not carry a capabilities bitmap: {data}"
    );
    Ok(())
}

#[test]
fn prep_on_generated_parquetdir_uses_owner_artifact_root() -> Result<()> {
    let (_trace_dir, report, generated_pqtdir) = build_generated_parquetdir_alias()?;
    let trace_path = generated_pqtdir.to_string_lossy().to_string();
    let owner_root = veloq_core::artifact_dir_for(&report);
    let alias_root = veloq_core::artifact_dir_for(&generated_pqtdir);

    let prep = run_veloq(["prep", &trace_path])?;
    assert!(
        prep.status.success(),
        "prep should accept generated parquetdir alias: {}",
        String::from_utf8_lossy(&prep.stderr)
    );
    let v: Value = serde_json::from_slice(&prep.stdout).context("prep stdout must be JSON")?;
    let data = v.get("data").ok_or_else(|| anyhow!("missing data: {v}"))?;
    assert_eq!(
        data.get("cache_root").and_then(Value::as_str),
        Some(owner_root.to_string_lossy().as_ref()),
        "prep must report the owning report cache root: {data}"
    );
    assert!(
        owner_root.join("meta.bin").is_file(),
        "prep should write meta.bin under the owning report cache root"
    );
    assert!(
        !alias_root.exists(),
        "generated parquetdir must not get its own nested cache root"
    );

    let status = run_veloq(["prep", "--status", &trace_path])?;
    assert!(
        status.status.success(),
        "prep --status should accept generated parquetdir alias: {}",
        String::from_utf8_lossy(&status.stderr)
    );
    let v: Value =
        serde_json::from_slice(&status.stdout).context("prep --status stdout must be JSON")?;
    let data = v.get("data").ok_or_else(|| anyhow!("missing data: {v}"))?;
    assert_eq!(
        data.get("cache_root").and_then(Value::as_str),
        Some(owner_root.to_string_lossy().as_ref()),
        "prep --status must inspect the owning report cache root: {data}"
    );
    Ok(())
}

#[test]
fn clean_removes_only_veloq_artifact_root() -> Result<()> {
    let (_trace_dir, trace) = build_minimal_trace()?;
    let trace_path = trace.to_string_lossy().to_string();
    let prep = run_veloq(["prep", &trace_path])?;
    assert!(
        prep.status.success(),
        "prep should create cache root: {}",
        String::from_utf8_lossy(&prep.stderr)
    );
    let cache_root = veloq_core::artifact_dir_for(&trace);
    assert!(
        cache_root.is_dir(),
        "prep should create artifact root {}",
        cache_root.display()
    );

    let dry = run_veloq(["clean", "--dry-run", &trace_path])?;
    assert!(
        dry.status.success(),
        "clean --dry-run failed: {}",
        String::from_utf8_lossy(&dry.stderr)
    );
    let v: Value =
        serde_json::from_slice(&dry.stdout).context("clean dry-run stdout must be JSON")?;
    let data = v.get("data").ok_or_else(|| anyhow!("missing data: {v}"))?;
    assert_eq!(v.get("command").and_then(Value::as_str), Some("clean"));
    assert_eq!(
        data.get("dry_run").and_then(Value::as_bool),
        Some(true),
        "dry-run flag must round-trip: {data}"
    );
    assert_eq!(
        data.get("removed").and_then(Value::as_bool),
        Some(false),
        "dry-run must not remove artifacts: {data}"
    );
    assert!(cache_root.is_dir(), "dry-run must leave cache root intact");

    let clean = run_veloq(["clean", &trace_path])?;
    assert!(
        clean.status.success(),
        "clean failed: {}",
        String::from_utf8_lossy(&clean.stderr)
    );
    let v: Value =
        serde_json::from_slice(&clean.stdout).context("clean stdout must be valid JSON")?;
    let data = v.get("data").ok_or_else(|| anyhow!("missing data: {v}"))?;
    assert_eq!(
        data.get("removed").and_then(Value::as_bool),
        Some(true),
        "clean should remove an existing artifact root: {data}"
    );
    assert!(
        !cache_root.exists(),
        "clean should remove only the artifact root"
    );
    assert!(trace.is_dir(), "direct parquetdir input must remain intact");
    Ok(())
}

#[test]
fn clean_generated_parquetdir_removes_owner_artifact_root() -> Result<()> {
    let (_trace_dir, report, generated_pqtdir) = build_generated_parquetdir_alias()?;
    let trace_path = generated_pqtdir.to_string_lossy().to_string();
    let owner_root = veloq_core::artifact_dir_for(&report);
    let alias_root = veloq_core::artifact_dir_for(&generated_pqtdir);

    let out = run_veloq(["clean", &trace_path])?;
    assert!(
        out.status.success(),
        "clean should accept generated parquetdir alias: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value = serde_json::from_slice(&out.stdout).context("clean stdout must be JSON")?;
    let data = v.get("data").ok_or_else(|| anyhow!("missing data: {v}"))?;
    assert_eq!(
        data.get("cache_root").and_then(Value::as_str),
        Some(owner_root.to_string_lossy().as_ref()),
        "clean must target the owning report cache root: {data}"
    );
    assert_eq!(
        data.get("removed").and_then(Value::as_bool),
        Some(true),
        "clean should remove the existing owner cache root: {data}"
    );
    assert!(report.is_file(), "clean must not remove the source report");
    assert!(
        !owner_root.exists(),
        "clean should remove the owning report cache root"
    );
    assert!(
        !alias_root.exists(),
        "clean should not create a nested alias cache root"
    );
    Ok(())
}

#[test]
fn prep_status_reports_cold_then_warm_state() -> Result<()> {
    // `--status` is the read-only inspection form. The parquetdir is
    // the input itself (always `present`); the
    // veloq-managed sidecar that flips cold→warm is the meta cache.
    let (_trace_dir, trace) = build_minimal_trace()?;
    let trace_path = trace.to_string_lossy().to_string();

    // Cold path: parquetdir exists (it IS the input), meta sidecar absent.
    let out = run_veloq(["prep", "--status", &trace_path])?;
    assert!(
        out.status.success(),
        "prep --status (cold) must exit 0: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value = serde_json::from_slice(&out.stdout)
        .context("prep --status (cold) stdout must be valid JSON")?;
    let data = v.get("data").ok_or_else(|| anyhow!("missing data: {v}"))?;
    let parquet = data
        .get("parquet_cache")
        .ok_or_else(|| anyhow!("missing parquet_cache: {data}"))?;
    assert_eq!(
        parquet.get("present").and_then(Value::as_bool),
        Some(true),
        "parquetdir is the input — must report present=true: {parquet}"
    );
    let tables = parquet
        .get("tables")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("missing parquet_cache.tables: {parquet}"))?;
    assert!(
        !tables.is_empty(),
        "parquet_cache.tables should list the input parquet tables: {parquet}"
    );
    let meta = data
        .get("meta_cache")
        .ok_or_else(|| anyhow!("missing meta_cache: {data}"))?;
    assert_eq!(
        meta.get("present").and_then(Value::as_bool),
        Some(false),
        "cold meta cache should not yet exist on disk: {meta}"
    );

    // Warm path: build the caches, then re-status.
    let prep = run_veloq(["prep", &trace_path])?;
    assert!(
        prep.status.success(),
        "prep build failed: {}",
        String::from_utf8_lossy(&prep.stderr)
    );
    let out = run_veloq(["prep", "--status", &trace_path])?;
    assert!(out.status.success());
    let v: Value = serde_json::from_slice(&out.stdout)
        .context("prep --status (warm) stdout must be valid JSON")?;
    let data = v.get("data").ok_or_else(|| anyhow!("missing data: {v}"))?;
    let parquet = data
        .get("parquet_cache")
        .ok_or_else(|| anyhow!("missing parquet_cache: {data}"))?;
    assert_eq!(
        parquet.get("present").and_then(Value::as_bool),
        Some(true),
        "parquet_cache.present must stay true: {parquet}"
    );
    let meta = data
        .get("meta_cache")
        .ok_or_else(|| anyhow!("missing meta_cache: {data}"))?;
    assert_eq!(
        meta.get("present").and_then(Value::as_bool),
        Some(true),
        "warm meta cache must be present on disk: {meta}"
    );
    assert_eq!(
        meta.get("fingerprint_match").and_then(Value::as_bool),
        Some(true),
        "warm meta cache must match fingerprint: {meta}"
    );
    // After a successful prep, the on-disk meta-cache format version
    // matches what this binary expects. The parquet cache no longer
    // carries a manifest version (it's nsys's own output).
    let meta_expected = meta
        .get("format_version_expected")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("missing meta format_version_expected: {meta}"))?;
    let meta_on_disk = meta
        .get("format_version_on_disk")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("missing meta format_version_on_disk after prep: {meta}"))?;
    assert_eq!(
        meta_expected, meta_on_disk,
        "warm meta cache version must match expected"
    );
    Ok(())
}

#[test]
fn cold_summary_emits_trace_span_on_first_run() -> Result<()> {
    // Regression: `summary` against a never-prepped trace used to
    // omit `trace_span` because `Source::compute_trace_span` only
    // consulted an existing sidecar. The verb itself builds the
    // sidecar; the emit boundary re-reads it so the very first
    // `summary` call hands agents a populated normalization
    // denominator (the field every diff / per-sec recipe needs).
    let (_trace_dir, trace) = build_minimal_trace()?;
    let trace_path = trace.to_string_lossy().to_string();
    // No `prep` here — this is the cold path.
    let out = run_veloq(["summary", &trace_path])?;
    assert!(
        out.status.success(),
        "cold summary failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value =
        serde_json::from_slice(&out.stdout).context("summary stdout must be valid JSON")?;
    let span = v
        .get("trace_span")
        .ok_or_else(|| anyhow::anyhow!("cold summary missing trace_span: {v}"))?;
    assert!(
        span.get("origin_ns").and_then(Value::as_i64).is_some(),
        "cold trace_span.origin_ns must be an i64: {span}"
    );
    assert!(
        span.get("span_ns").and_then(Value::as_i64).is_some(),
        "cold trace_span.span_ns must be an i64: {span}"
    );
    Ok(())
}

#[cfg(unix)]
fn run_cold_nsys_rep_with_fake_export(command: &str) -> Result<Output> {
    use std::os::unix::fs::PermissionsExt;

    let (trace_dir, direct_pqtdir) = build_minimal_trace()?;
    let report = trace_dir.path().join("cold.nsys-rep");
    std::fs::write(&report, b"source").context("write cold report placeholder")?;

    let fake_bin_dir = trace_dir.path().join("fake-bin");
    std::fs::create_dir_all(&fake_bin_dir).context("create fake nsys bin dir")?;
    let fake_nsys = fake_bin_dir.join("nsys");
    std::fs::write(
        &fake_nsys,
        r#"#!/bin/sh
if [ "$1" = "--version" ]; then
    echo "NVIDIA Nsight Systems version 2026.2.1.210-fake"
    exit 0
fi
if [ "$1" = "export" ] && [ "$2" = "--help" ]; then
    echo "Possible values are: sqlite, arrowdir, parquetdir"
    exit 0
fi
if [ "$1" != "export" ]; then
    echo "unexpected fake nsys invocation: $*" >&2
    exit 9
fi
out=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        -o|--output)
            shift
            out="$1"
            ;;
    esac
    shift
done
if [ -z "$out" ]; then
    echo "missing -o/--output" >&2
    exit 7
fi
printf 'fake stdout progress\n'
printf 'fake stderr diagnostic\n' >&2
mkdir -p "$out"
for f in "$VELOQ_FAKE_PQTDIR"/*.parquet; do
    cp "$f" "$out/"
done
"#,
    )
    .context("write fake nsys")?;
    let mut perms = std::fs::metadata(&fake_nsys)?.permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&fake_nsys, perms)?;

    let old_path = std::env::var_os("PATH").unwrap_or_default();
    let mut paths = vec![fake_bin_dir];
    paths.extend(std::env::split_paths(&old_path));
    let path_env: std::ffi::OsString = std::env::join_paths(paths)?;
    let trace_path = report.to_string_lossy().to_string();
    run_veloq_with_env(
        [command, &trace_path],
        [
            (std::ffi::OsString::from("PATH"), path_env),
            (
                std::ffi::OsString::from("VELOQ_FAKE_PQTDIR"),
                direct_pqtdir.as_os_str().to_os_string(),
            ),
        ],
    )
}

#[cfg(unix)]
fn assert_child_output_stays_off_stdout(out: &Output, command: &str) -> Result<()> {
    assert!(
        out.status.success(),
        "{command} failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.contains("fake stdout progress"),
        "child stdout must not contaminate JSON stdout: {stdout}"
    );
    let v: Value = serde_json::from_slice(&out.stdout)
        .with_context(|| format!("{command} stdout must be valid JSON"))?;
    assert_eq!(v.get("command").and_then(Value::as_str), Some(command),);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("fake stdout progress"),
        "captured child stdout should be replayed on veloq stderr: {stderr}"
    );
    assert!(
        stderr.contains("fake stderr diagnostic"),
        "captured child stderr should stay on veloq stderr: {stderr}"
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn cold_nsys_rep_summary_export_keeps_child_output_off_stdout() -> Result<()> {
    let out = run_cold_nsys_rep_with_fake_export("summary")?;
    assert_child_output_stays_off_stdout(&out, "nsys.summary")
}

#[cfg(unix)]
#[test]
fn cold_nsys_rep_prep_export_keeps_child_output_off_stdout() -> Result<()> {
    let out = run_cold_nsys_rep_with_fake_export("prep")?;
    assert_child_output_stays_off_stdout(&out, "nsys.prep")
}

#[test]
fn warm_summary_emits_trace_span_after_prep() -> Result<()> {
    // First call to `prep` writes the metadata cache so the
    // envelope-level `trace_span` becomes available on the next run.
    // This verifies the contract that warm traces carry the
    // normalization denominator agents need for cross-trace diff.
    let (_trace_dir, trace) = build_minimal_trace()?;
    let trace_path = trace.to_string_lossy().to_string();
    let prep = run_veloq(["prep", &trace_path])?;
    assert!(
        prep.status.success(),
        "prep failed: {}",
        String::from_utf8_lossy(&prep.stderr)
    );
    let out = run_veloq(["summary", &trace_path])?;
    assert!(out.status.success());
    let v: Value =
        serde_json::from_slice(&out.stdout).context("summary stdout must be valid JSON")?;
    let span = v
        .get("trace_span")
        .ok_or_else(|| anyhow::anyhow!("warm summary missing trace_span: {v}"))?;
    assert!(
        span.get("origin_ns").and_then(Value::as_i64).is_some(),
        "trace_span.origin_ns must be an i64: {span}"
    );
    assert!(
        span.get("span_ns").and_then(Value::as_i64).is_some(),
        "trace_span.span_ns must be an i64: {span}"
    );
    Ok(())
}
