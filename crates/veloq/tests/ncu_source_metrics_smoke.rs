//! Tier-2 end-to-end smoke for `ncu source-metrics`.
//!
//! Two fixtures exercise complementary paths:
//!
//! * `vector_add_basic.ncu-rep` — `--set base` capture, no source
//!   counters. Covers wrong-input and empty-result paths.
//! * `source_metric_basic.ncu-rep` — `--set full` capture with
//!   `nvcc -lineinfo`. Covers the populated `--by line` / `--by sass`
//!   / `--by file` axes plus the gate-coverage AC
//!   (both `Section.SourceMetrics` and `ProfilerSourceMetricTable`
//!   body-item paths exercised).
//!
//! Each test launches the freshly-built `veloq` binary so the full
//! envelope → dispatch → response pipeline is exercised.

use anyhow::{Context, Result};
use serde_json::Value;
use std::path::PathBuf;
use std::process::{Command, Output};

fn veloq_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_veloq"))
}

fn fixtures_dir() -> Result<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let crates_root = manifest
        .parent()
        .ok_or_else(|| anyhow::anyhow!("workspace layout: veloq is under crates/"))?;
    Ok(crates_root.join("ncu/veloq-ncu/tests/fixtures"))
}

/// `--set base` capture, no source counters. Used for empty-result /
/// wrong-input coverage.
fn fixture() -> Result<PathBuf> {
    Ok(fixtures_dir()?.join("vector_add_basic.ncu-rep"))
}

/// `--set full` capture with `nvcc -lineinfo`. Carries populated
/// source counters that resolve to (file, line) attributions.
fn populated_fixture() -> Result<PathBuf> {
    Ok(fixtures_dir()?.join("source_metric_basic.ncu-rep"))
}

/// Bank-conflict counter name. Same metric the user-agent
/// investigation used.
const BANK_CONFLICT_COUNTER: &str = "derived__memory_l1_conflicts_shared_nway";

fn run_veloq<I, S>(args: I) -> Result<Output>
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    Command::new(veloq_bin())
        .args(args)
        .output()
        .context("spawn veloq binary")
}

fn parse_stdout(out: &Output) -> Result<Value> {
    serde_json::from_slice(&out.stdout).context("veloq stdout must be valid JSON")
}

fn at<'a>(v: &'a Value, ptr: &str) -> Result<&'a Value> {
    v.pointer(ptr)
        .ok_or_else(|| anyhow::anyhow!("missing pointer `{ptr}` in {v}"))
}

/// `--counter` is required (matches `ncu metrics`). A
/// cold invocation without it returns a clap-level error.
#[test]
fn missing_counter_flag_is_clap_level_error() -> Result<()> {
    let trace = fixture()?;
    let out = run_veloq([
        "ncu",
        "source-metrics",
        trace.to_string_lossy().as_ref(),
        "--row-id",
        "launch:0",
    ])?;
    // clap emits non-zero. The error text mentions --counter.
    assert!(!out.status.success());
    let combined = String::from_utf8_lossy(&out.stdout) + String::from_utf8_lossy(&out.stderr);
    assert!(
        combined.contains("--counter"),
        "expected the clap error to mention --counter; got: {combined}",
    );
    Ok(())
}

/// `--by something_else` is rejected at the verb dispatch with a
/// structured EnvelopeError.
#[test]
fn unknown_axis_emits_structured_error() -> Result<()> {
    let trace = fixture()?;
    let out = run_veloq([
        "ncu",
        "source-metrics",
        trace.to_string_lossy().as_ref(),
        "--row-id",
        "launch:0",
        "--counter",
        "x",
        "--by",
        "instructions",
    ])?;
    assert_eq!(out.status.code(), Some(1));
    let v = parse_stdout(&out)?;
    assert_eq!(
        at(&v, "/error/code")?.as_str(),
        Some("ncu.command.unknown-source-metrics-axis")
    );
    let msg = at(&v, "/error/message")?
        .as_str()
        .context("error.message must be a string")?;
    assert!(
        msg.contains("--by axis"),
        "expected message to mention `--by axis`, got: {msg}",
    );
    Ok(())
}

/// `--line` without `--file` is rejected with a structured error.
#[test]
fn line_without_file_is_rejected() -> Result<()> {
    let trace = fixture()?;
    let out = run_veloq([
        "ncu",
        "source-metrics",
        trace.to_string_lossy().as_ref(),
        "--row-id",
        "launch:0",
        "--counter",
        "*",
        "--line",
        "42",
    ])?;
    assert_eq!(out.status.code(), Some(1));
    let v = parse_stdout(&out)?;
    assert_eq!(
        at(&v, "/error/code")?.as_str(),
        Some("ncu.command.source-metrics-line-without-file")
    );
    let msg = at(&v, "/error/message")?
        .as_str()
        .context("error.message must be a string")?;
    assert!(
        msg.contains("--line") && msg.contains("--file"),
        "expected error to mention both flags; got: {msg}",
    );
    Ok(())
}

/// Non-`launch:N` row_id is rejected at the row-id parser, matching
/// disasm's wording.
#[test]
fn non_launch_row_id_is_rejected() -> Result<()> {
    let trace = fixture()?;
    let out = run_veloq([
        "ncu",
        "source-metrics",
        trace.to_string_lossy().as_ref(),
        "--row-id",
        "range:0",
        "--counter",
        "*",
    ])?;
    assert_eq!(out.status.code(), Some(1));
    let v = parse_stdout(&out)?;
    assert_eq!(
        at(&v, "/error/code")?.as_str(),
        Some("ncu.command.invalid-launch-row-id")
    );
    let msg = at(&v, "/error/message")?
        .as_str()
        .context("error.message must be a string")?;
    assert!(
        msg.contains("launch:<idx>"),
        "expected the row-id parser to mention `launch:<idx>`; got: {msg}",
    );
    Ok(())
}

#[test]
fn comma_only_counter_glob_is_rejected() -> Result<()> {
    let trace = fixture()?;
    let out = run_veloq([
        "ncu",
        "source-metrics",
        trace.to_string_lossy().as_ref(),
        "--row-id",
        "launch:0",
        "--counter",
        ",",
    ])?;
    assert_eq!(out.status.code(), Some(1));
    let v = parse_stdout(&out)?;
    assert_eq!(
        at(&v, "/error/code")?.as_str(),
        Some("ncu.command.empty-counter-glob")
    );
    let msg = at(&v, "/error/message")?
        .as_str()
        .context("error.message must be a string")?;
    assert!(
        msg.contains("--counter"),
        "expected empty-counter error to mention `--counter`; got: {msg}",
    );
    Ok(())
}

/// Against the existing fixture (no source counters), the verb
/// returns a clean envelope with an `auxiliary.warnings` entry
/// explaining the gap — a reader should see *why* the result is empty,
/// not just an empty rows list with no explanation.
#[test]
fn empty_result_envelope_explains_why() -> Result<()> {
    let trace = fixture()?;
    let out = run_veloq([
        "ncu",
        "source-metrics",
        trace.to_string_lossy().as_ref(),
        "--row-id",
        "launch:0",
        "--counter",
        "derived__*",
    ])?;
    assert!(
        out.status.success(),
        "expected envelope-level success on empty result; exit={:?}; stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
    );
    let v = parse_stdout(&out)?;
    assert_eq!(at(&v, "/schema")?.as_str(), Some("v1"));
    assert_eq!(at(&v, "/source/kind")?.as_str(), Some("ncu"));
    assert_eq!(at(&v, "/command")?.as_str(), Some("ncu.source-metrics"));
    let count = at(&v, "/data/count")?.as_u64().unwrap_or(0);
    let warnings = at(&v, "/data/auxiliary/warnings")?
        .as_array()
        .context("auxiliary.warnings must be an array")?;
    // Either rows are empty AND a warning explains why, OR (when a
    // future fixture lands the source counters) rows are non-empty —
    // the test pins the contract that empty-result is never silent.
    if count == 0 {
        assert!(
            !warnings.is_empty(),
            "empty result must carry an explanatory auxiliary.warnings; got: {v}",
        );
    }
    Ok(())
}

/// `ncu schema source-metrics` returns the wire-format schema. Pins
/// the schema-target registration.
#[test]
fn schema_target_includes_source_metrics() -> Result<()> {
    let out = run_veloq(["ncu", "schema", "source-metrics"])?;
    assert!(
        out.status.success(),
        "exit={:?}; stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
    );
    let v = parse_stdout(&out)?;
    assert_eq!(at(&v, "/data/target")?.as_str(), Some("source-metrics"));
    // Schema document is present (don't probe its shape beyond
    // "non-null object").
    assert!(v.pointer("/data/schema").is_some());
    Ok(())
}

// ---- populated-fixture tests (source_metric_basic.ncu-rep) ----

/// `--by line` against the populated fixture returns ≥1 row with the
/// motivating-case counter resolved as Some(f64), counter_coverage
/// populated, and sass_addresses non-empty. Pins the load-bearing
/// derived__* additive-double-valued path that the prior
/// "uint64-only ⇒ additive" heuristic would have silently dropped.
#[test]
fn populated_fixture_by_line_resolves_motivating_counter() -> Result<()> {
    let trace = populated_fixture()?;
    let out = run_veloq([
        "ncu",
        "source-metrics",
        trace.to_string_lossy().as_ref(),
        "--row-id",
        "launch:0",
        "--counter",
        BANK_CONFLICT_COUNTER,
    ])?;
    assert!(
        out.status.success(),
        "exit={:?}; stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
    );
    let v = parse_stdout(&out)?;
    assert_eq!(at(&v, "/data/axis")?.as_str(), Some("line"));
    let rows = at(&v, "/data/rows")?
        .as_array()
        .context("data.rows must be an array")?;
    assert!(
        !rows.is_empty(),
        "expected ≥1 line row on populated fixture; got: {v}",
    );
    // Top row carries the motivating counter as a real number, and
    // counter_coverage is populated for the same key.
    let row = rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("rows non-empty by previous assertion"))?;
    let counters = at(row, "/counters")?
        .as_object()
        .context("counters must be an object")?;
    let coverage = at(row, "/counter_coverage")?
        .as_object()
        .context("counter_coverage must be an object")?;
    assert!(
        counters.contains_key(BANK_CONFLICT_COUNTER),
        "counters missing motivating-case key; got: {counters:?}",
    );
    assert_eq!(
        counters.keys().collect::<Vec<_>>(),
        coverage.keys().collect::<Vec<_>>(),
        "counters / counter_coverage must share an ordered key set",
    );
    let v_val = counters
        .get(BANK_CONFLICT_COUNTER)
        .and_then(|x| x.as_f64())
        .context("motivating counter must resolve to f64 (not null) on a populated fixture")?;
    assert!(v_val >= 0.0);
    let sass_addresses = at(row, "/sass_addresses")?
        .as_array()
        .context("sass_addresses must be an array")?;
    assert!(
        !sass_addresses.is_empty(),
        "expected ≥1 SASS address backing the line row",
    );
    // sass_addresses is sorted ascending + deduped (invariant 4)
    // — pin it.
    let raw: Vec<i64> = sass_addresses.iter().filter_map(Value::as_i64).collect();
    let strictly_ascending = raw
        .iter()
        .zip(raw.iter().skip(1))
        .all(|(prev, next)| prev < next);
    assert!(
        strictly_ascending,
        "sass_addresses must be strictly ascending"
    );
    Ok(())
}

/// `--by sass` returns one row per cubin-relative SASS address with
/// the counter value verbatim. Identity rollup — pin the per-PC
/// shape.
#[test]
fn populated_fixture_by_sass_emits_per_pc_rows() -> Result<()> {
    let trace = populated_fixture()?;
    let out = run_veloq([
        "ncu",
        "source-metrics",
        trace.to_string_lossy().as_ref(),
        "--row-id",
        "launch:0",
        "--counter",
        BANK_CONFLICT_COUNTER,
        "--by",
        "sass",
    ])?;
    assert!(out.status.success());
    let v = parse_stdout(&out)?;
    assert_eq!(at(&v, "/data/axis")?.as_str(), Some("sass"));
    let rows = at(&v, "/data/rows")?
        .as_array()
        .context("data.rows must be an array")?;
    assert!(!rows.is_empty(), "expected ≥1 sass row");
    // Every row has an `address` key; the motivating counter is
    // present (numeric or null).
    for r in rows {
        assert!(r.pointer("/address").is_some(), "row missing address: {r}");
        let counters = at(r, "/counters")?.as_object().context("counters object")?;
        assert!(counters.contains_key(BANK_CONFLICT_COUNTER));
    }
    Ok(())
}

/// `--by file` collapses every line row in the fixture into one row
/// per source file; `line_count` and `sass_count` are populated.
#[test]
fn populated_fixture_by_file_aggregates_lines() -> Result<()> {
    let trace = populated_fixture()?;
    let out = run_veloq([
        "ncu",
        "source-metrics",
        trace.to_string_lossy().as_ref(),
        "--row-id",
        "launch:0",
        "--counter",
        BANK_CONFLICT_COUNTER,
        "--by",
        "file",
    ])?;
    assert!(out.status.success());
    let v = parse_stdout(&out)?;
    assert_eq!(at(&v, "/data/axis")?.as_str(), Some("file"));
    let rows = at(&v, "/data/rows")?
        .as_array()
        .context("data.rows must be an array")?;
    assert!(!rows.is_empty(), "expected ≥1 file row");
    for r in rows {
        let line_count = at(r, "/line_count")?
            .as_u64()
            .context("line_count must be unsigned")?;
        let sass_count = at(r, "/sass_count")?
            .as_u64()
            .context("sass_count must be unsigned")?;
        assert!(line_count >= 1);
        assert!(sass_count >= line_count, "sass_count must ≥ line_count");
    }
    Ok(())
}

/// `--counter A,B` and `--counter B,A` produce byte-identical
/// `data.rows[]` — pins the invariant 5 cross-counter key
/// collation. Asserts via serde_json::to_vec equality, not
/// structural compare.
#[test]
fn populated_fixture_counter_arg_permutation_is_byte_identical() -> Result<()> {
    let trace = populated_fixture()?;
    // The fixture carries multiple derived__memory_l1_* counters;
    // pick two that resolve, then assert order-invariance.
    let a = BANK_CONFLICT_COUNTER;
    let b = "derived__memory_l1_wavefronts_shared_excessive";
    let run = |order: &str| -> Result<Vec<u8>> {
        let out = run_veloq([
            "ncu",
            "source-metrics",
            trace.to_string_lossy().as_ref(),
            "--row-id",
            "launch:0",
            "--counter",
            order,
        ])?;
        assert!(out.status.success(), "exit={:?}", out.status.code());
        let v = parse_stdout(&out)?;
        let rows = at(&v, "/data/rows")?.clone();
        Ok(serde_json::to_vec(&rows)?)
    };
    let ab = run(&format!("{a},{b}"))?;
    let ba = run(&format!("{b},{a}"))?;
    assert_eq!(ab, ba, "A,B vs B,A must serialize byte-identically",);
    Ok(())
}

/// Gate-coverage assertion: the fixture must
/// exercise both the `Section.SourceMetrics` path AND the
/// `ProfilerSourceMetricTable` body-item path. The decoded summary
/// surfaces both as `section.source_metrics` (the first path) and
/// `section.metrics` entries with `location: "body"` whose name
/// appears in the launch's per-PC metric instances (the second
/// path). Both must be non-empty.
#[test]
fn populated_fixture_exercises_both_gate_paths() -> Result<()> {
    let trace = populated_fixture()?;
    let out = run_veloq([
        "ncu",
        "inspect",
        trace.to_string_lossy().as_ref(),
        "--row-id",
        "launch:1",
    ])?;
    assert!(out.status.success(), "exit={:?}", out.status.code());
    let v = parse_stdout(&out)?;
    // Native model: there is no section catalog; instead `ncu inspect`
    // carries the full metric list with each per-PC instance pre-tagged
    // by `placement`. Both placement paths must appear on the
    // pointer-chase launch: the `smsp__pcsamp_*` source-counter family
    // resolves to source (`attributed`), while the `warpsampling:`
    // family carries non-VA correlations (`out_of_cubin`).
    let metrics = at(&v, "/data/rows/0/metrics")?
        .as_array()
        .context("metrics must be an array")?;
    let mut placements: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for m in metrics {
        if let Some(insts) = m.pointer("/instances").and_then(Value::as_array) {
            for i in insts {
                if let Some(p) = i.pointer("/placement").and_then(Value::as_str) {
                    placements.insert(p);
                }
            }
        }
    }
    assert!(
        placements.contains("attributed"),
        "gate-path 1 (source-attributed instances) must appear; saw {placements:?}",
    );
    assert!(
        placements.contains("out_of_cubin"),
        "gate-path 2 (out-of-cubin correlations) must appear; saw {placements:?}",
    );
    Ok(())
}

/// End-to-end populated-rows AC. `launch:1`
/// is the pointer-chase kernel; the portable
/// `*pcsamp_warps_issue_stalled_*` glob picks up both the
/// source-attributable `smsp__pcsamp_*` (pm-category) family and the
/// non-source-attributable `warpsampling:smsp__pcsamp_*`
/// (sampler-category) family. The positive-evidence gate routes the
/// pm family into `matched_counters` (rows contribute) and the
/// sampler family into `skipped_counters` (rejected by cubin-range
/// check).
#[test]
fn populated_fixture_pcsamp_populates_rows_and_routes_warpsampling_to_skipped() -> Result<()> {
    let trace = populated_fixture()?;
    let out = run_veloq([
        "ncu",
        "source-metrics",
        trace.to_string_lossy().as_ref(),
        "--row-id",
        "launch:1",
        "--counter",
        "*pcsamp_warps_issue_stalled_*",
        "--by",
        "line",
        "--limit",
        "10",
    ])?;
    if !out.status.success() {
        anyhow::bail!(
            "verb exited non-zero: stdout={} stderr={}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let env = parse_stdout(&out)?;

    let count = at(&env, "/data/count")?.as_u64().unwrap_or(0);
    assert!(
        count > 0,
        "expected non-empty rows on launch:1 pointer-chase kernel; got: {env}"
    );

    let matched = at(&env, "/data/auxiliary/matched_counters")?
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("matched_counters must be an array"))?;
    let matched_pm = matched
        .iter()
        .filter_map(Value::as_str)
        .filter(|n| !n.starts_with("warpsampling:"))
        .count();
    let matched_warpsampling = matched
        .iter()
        .filter_map(Value::as_str)
        .filter(|n| n.starts_with("warpsampling:"))
        .count();
    assert!(
        matched_pm > 0,
        "expected ≥1 unprefixed smsp__pcsamp_* counter in matched_counters; got: {matched:?}"
    );
    assert_eq!(
        matched_warpsampling, 0,
        "warpsampling: family should be in skipped, not matched; got matched: {matched:?}"
    );

    let skipped = at(&env, "/data/auxiliary/skipped_counters")?
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("skipped_counters must be an array"))?;
    let skipped_warpsampling = skipped
        .iter()
        .filter(|s| {
            s.pointer("/name")
                .and_then(Value::as_str)
                .map(|n| n.starts_with("warpsampling:"))
                .unwrap_or(false)
                && s.pointer("/reason").and_then(Value::as_str) == Some("not-a-source-counter")
        })
        .count();
    assert!(
        skipped_warpsampling > 0,
        "expected warpsampling: family in skipped_counters with reason not-a-source-counter; got: {skipped:?}"
    );
    Ok(())
}

// ---- warp-stalls verb ------------------------------------------

/// `ncu warp-stalls` end-to-end through the real binary. The committed
/// `source_metric_basic` sidecar carries no `warp_stalls` (it predates
/// the field), so this exercises the CLI dispatch + envelope + the
/// empty-but-well-shaped path (no warp sampling captured ⇒ not an
/// error). Populated-data correctness is locked by the in-process
/// `verb_goldens` warp-stalls golden + reconciliation tests.
#[test]
fn warp_stalls_empty_is_well_shaped() -> Result<()> {
    let trace = populated_fixture()?;
    let out = run_veloq([
        "ncu",
        "warp-stalls",
        "--row-id",
        "launch:0",
        &trace.to_string_lossy(),
    ])?;
    assert!(
        out.status.success(),
        "warp-stalls should succeed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v = parse_stdout(&out)?;
    assert_eq!(at(&v, "/command")?.as_str(), Some("ncu.warp-stalls"));
    assert_eq!(at(&v, "/data/axis")?.as_str(), Some("line"));
    assert_eq!(at(&v, "/data/count")?.as_u64(), Some(0));
    // auxiliary is present and well-shaped even with no samples.
    assert_eq!(at(&v, "/data/auxiliary/total_samples")?.as_u64(), Some(0));
    at(&v, "/data/auxiliary/per_reason_totals")?;
    Ok(())
}

/// Unknown `--by` axis is a structured error, mirroring source-metrics.
#[test]
fn warp_stalls_unknown_axis_errors() -> Result<()> {
    let trace = populated_fixture()?;
    let out = run_veloq([
        "ncu",
        "warp-stalls",
        "--row-id",
        "launch:0",
        "--by",
        "bogus",
        &trace.to_string_lossy(),
    ])?;
    assert!(!out.status.success(), "bogus axis should fail");
    let v = parse_stdout(&out)?;
    assert_eq!(
        at(&v, "/error/code")?.as_str(),
        Some("ncu.command.unknown-warp-stalls-axis")
    );
    assert!(
        at(&v, "/error/message")?
            .as_str()
            .is_some_and(|m| m.contains("unknown --by axis")),
        "expected structured axis error: {v}"
    );
    Ok(())
}

#[test]
fn warp_stalls_out_of_range_row_id_errors() -> Result<()> {
    let trace = populated_fixture()?;
    let out = run_veloq([
        "ncu",
        "warp-stalls",
        "--row-id",
        "launch:9999",
        &trace.to_string_lossy(),
    ])?;
    assert!(!out.status.success(), "out-of-range row-id should fail");
    let v = parse_stdout(&out)?;
    assert_eq!(
        at(&v, "/error/code")?.as_str(),
        Some("ncu.command.launch-row-id-out-of-range")
    );
    assert!(
        at(&v, "/error/message")?
            .as_str()
            .is_some_and(|m| m.contains("launch:9999") && m.contains("out of range")),
        "expected structured out-of-range error: {v}"
    );
    Ok(())
}
