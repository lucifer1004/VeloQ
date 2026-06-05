//! Tier-2 end-to-end smoke for NCU detail-verb tabular projectors.
//!
//! Each NCU detail verb now accepts `--format json|csv|table`. JSON
//! stays the agent contract; CSV/table are flat projections of the
//! same `data.rows[]` (nested objects → dotted-key columns,
//! `BTreeMap` fields → one column per resolved key, scalar arrays →
//! semicolon-joined single column). These tests pin:
//!
//! 1. Every verb's CSV output carries a non-empty header line on a
//!    fixture run.
//! 2. Every verb's `--format table` returns a zero exit.
//! 3. Cross-format consistency: for verbs whose row struct is the
//!    natural-tabular shape, the CSV header columns equal the
//!    flattened JSON keys of `data.rows[0]` (one nesting level +
//!    BTreeMap expansion). Verbs that intentionally diverge (inspect
//!    is a narrow projection; disasm projects one row per SASS
//!    instruction inside `data.rows[*].instructions`) get a dedicated
//!    check tailored to their projection rule.
//!
//! Each test launches the freshly-built `veloq` binary so the full
//! envelope → dispatch → projector pipeline is exercised.

use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
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

fn vector_add() -> Result<PathBuf> {
    Ok(fixtures_dir()?.join("vector_add_basic.ncu-rep"))
}

fn source_metric_fixture() -> Result<PathBuf> {
    Ok(fixtures_dir()?.join("source_metric_basic.ncu-rep"))
}

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

fn assert_zero(out: &Output, ctx: &str) -> Result<()> {
    if !out.status.success() {
        anyhow::bail!(
            "{ctx} exited with {:?}\nstdout: {}\nstderr: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(())
}

fn csv_header(stdout: &str) -> Result<Vec<String>> {
    let header_line = stdout
        .lines()
        .find(|l| !l.starts_with('#') && !l.is_empty())
        .ok_or_else(|| anyhow::anyhow!("csv has no non-comment, non-empty lines"))?;
    Ok(header_line.split(',').map(|s| s.to_string()).collect())
}

/// Flatten one nesting level + BTreeMap expansion of a JSON object,
/// matching the cross-format consistency contract in
/// the AC: nested Objects expand to dotted-key
/// columns; arrays of scalars and arrays of objects stay as one key.
fn flatten_one_level_keys(v: &Value) -> BTreeSet<String> {
    let mut keys = BTreeSet::new();
    if let Value::Object(map) = v {
        for (k, vv) in map {
            match vv {
                Value::Object(inner) => {
                    if inner.is_empty() {
                        keys.insert(k.clone());
                    } else {
                        for sk in inner.keys() {
                            keys.insert(format!("{k}.{sk}"));
                        }
                    }
                }
                _ => {
                    keys.insert(k.clone());
                }
            }
        }
    }
    keys
}

/// Run `veloq ncu <verb>` in JSON mode, return the parsed envelope.
fn fetch_json(trace: &Path, verb: &str, extra: &[&str]) -> Result<Value> {
    let mut args: Vec<String> = vec![
        "ncu".to_string(),
        verb.to_string(),
        trace.to_string_lossy().to_string(),
        "--format".to_string(),
        "json".to_string(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    let out = run_veloq(args)?;
    assert_zero(&out, &format!("ncu {verb} --format json"))?;
    serde_json::from_slice(&out.stdout).context("veloq stdout must be valid JSON")
}

/// Run `veloq ncu <verb>` in CSV mode, return the parsed header.
fn fetch_csv_header(trace: &Path, verb: &str, extra: &[&str]) -> Result<Vec<String>> {
    let mut args: Vec<String> = vec![
        "ncu".to_string(),
        verb.to_string(),
        trace.to_string_lossy().to_string(),
        "--format".to_string(),
        "csv".to_string(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    let out = run_veloq(args)?;
    assert_zero(&out, &format!("ncu {verb} --format csv"))?;
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    csv_header(&text)
}

/// Run `veloq ncu <verb>` in table mode and just assert zero exit.
fn fetch_table_ok(trace: &Path, verb: &str, extra: &[&str]) -> Result<()> {
    let mut args: Vec<String> = vec![
        "ncu".to_string(),
        verb.to_string(),
        trace.to_string_lossy().to_string(),
        "--format".to_string(),
        "table".to_string(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    let out = run_veloq(args)?;
    assert_zero(&out, &format!("ncu {verb} --format table"))
}

/// Pull `data.rows[0]` from a successful envelope; returns `None`
/// when the response carries no rows (empty result).
fn first_row(env: &Value) -> Option<&Value> {
    env.pointer("/data/rows/0")
}

/// Cross-format consistency check used by verbs whose `data.rows[]`
/// element is the natural tabular row. The CSV header must be a
/// **superset** of the flattened (one nesting level + BTreeMap-
/// expanded) JSON keys of `data.rows[0]`: every JSON key must
/// resolve to a CSV column, but the CSV may also surface
/// `#[serde(skip_serializing_if = "Option::is_none")]` columns
/// that the current fixture happens to leave `None` (e.g.
/// `nvtx_range_path` on a launch with no NVTX context, or
/// `source.column` on a SASS instruction whose DWARF carries
/// line-but-not-column info). The asymmetry is by design — JSON
/// elides absent Optionals, CSV always reserves the column so
/// downstream spreadsheets see the schema. The CSV/JSON parity
/// contract is "no JSON key silently drops from CSV", which is what
/// column-drift detection actually needs.
fn assert_csv_covers_json(trace: &Path, verb: &str, extra: &[&str]) -> Result<()> {
    let env = fetch_json(trace, verb, extra)?;
    let csv = fetch_csv_header(trace, verb, extra)?;
    fetch_table_ok(trace, verb, extra)?;
    let Some(row) = first_row(&env) else {
        // Empty row set: only ensure the CSV still produced a non-
        // empty header. There is no `data.rows[0]` anchor here.
        assert!(
            !csv.is_empty(),
            "ncu {verb} csv has no header even with empty rows"
        );
        return Ok(());
    };
    let json_keys = flatten_one_level_keys(row);
    let csv_set: BTreeSet<String> = csv.into_iter().collect();
    let missing: BTreeSet<&String> = json_keys.difference(&csv_set).collect();
    assert!(
        missing.is_empty(),
        "csv header missing JSON keys for ncu {verb}: missing {missing:?}, \
         csv has {csv_set:?}, json has {json_keys:?}"
    );
    Ok(())
}

// ---- per-verb smokes ------------------------------------------------------

#[test]
fn ranges_csv_table_emit_and_consistent() -> Result<()> {
    let trace = vector_add()?;
    assert_csv_covers_json(&trace, "ranges", &[])
}

#[test]
fn graphs_csv_table_emit_and_consistent() -> Result<()> {
    let trace = vector_add()?;
    assert_csv_covers_json(&trace, "graphs", &[])
}

#[test]
fn sources_csv_table_emit_and_consistent() -> Result<()> {
    let trace = vector_add()?;
    assert_csv_covers_json(&trace, "sources", &[])
}

#[test]
fn launches_csv_table_emit_and_consistent() -> Result<()> {
    let trace = vector_add()?;
    assert_csv_covers_json(&trace, "launches", &[])
}

#[test]
fn metrics_long_csv_table_emit_and_consistent() -> Result<()> {
    let trace = source_metric_fixture()?;
    assert_csv_covers_json(&trace, "metrics", &["--counter", "sm__cycles_active*"])
}

#[test]
fn metrics_wide_csv_table_emit_and_consistent() -> Result<()> {
    let trace = source_metric_fixture()?;
    assert_csv_covers_json(
        &trace,
        "metrics",
        &["--counter", "sm__cycles_active*", "--per-launch"],
    )
}

#[test]
fn source_metrics_line_csv_table_emit_and_consistent() -> Result<()> {
    let trace = source_metric_fixture()?;
    assert_csv_covers_json(
        &trace,
        "source-metrics",
        &[
            "--row-id",
            "launch:0",
            "--counter",
            "derived__memory_l1_conflicts_shared_nway",
            "--by",
            "line",
        ],
    )
}

#[test]
fn source_metrics_sass_csv_table_emit_and_consistent() -> Result<()> {
    let trace = source_metric_fixture()?;
    assert_csv_covers_json(
        &trace,
        "source-metrics",
        &[
            "--row-id",
            "launch:0",
            "--counter",
            "derived__memory_l1_conflicts_shared_nway",
            "--by",
            "sass",
        ],
    )
}

#[test]
fn source_metrics_file_csv_table_emit_and_consistent() -> Result<()> {
    let trace = source_metric_fixture()?;
    assert_csv_covers_json(
        &trace,
        "source-metrics",
        &[
            "--row-id",
            "launch:0",
            "--counter",
            "derived__memory_l1_conflicts_shared_nway",
            "--by",
            "file",
        ],
    )
}

// ---- heterogeneous-row verbs (inspect / disasm) --------------------------

#[test]
fn inspect_csv_emits_wide_header_with_variant_tag() -> Result<()> {
    // Inspect is the documented heterogeneous-row case (see
    // `inspect_view_v2` in crates/ncu/veloq-ncu/src/views.rs): one
    // wide CSV with NULL columns for fields the variant doesn't
    // carry. The CSV header is a narrow projection of the full
    // LaunchEntry payload — a strict cross-format key-set equality
    // doesn't hold here, but the CSV header must include the
    // variant `type` discriminator and a handful of identity columns.
    let trace = vector_add()?;
    let csv = fetch_csv_header(&trace, "inspect", &["--row-id", "launch:0"])?;
    fetch_table_ok(&trace, "inspect", &["--row-id", "launch:0"])?;
    for required in ["type", "key", "row_id", "kernel_demangled", "grid_size"] {
        assert!(
            csv.iter().any(|c| c == required),
            "inspect csv header missing required column `{required}`: {csv:?}"
        );
    }
    Ok(())
}

#[test]
fn inspect_csv_renders_not_found_row_with_null_columns() -> Result<()> {
    let trace = vector_add()?;
    let args: Vec<String> = vec![
        "ncu".to_string(),
        "inspect".to_string(),
        trace.to_string_lossy().to_string(),
        "--format".to_string(),
        "csv".to_string(),
        "--row-id".to_string(),
        "launch:9999".to_string(),
    ];
    let out = run_veloq(args)?;
    assert_zero(&out, "ncu inspect not_found csv")?;
    let text = String::from_utf8_lossy(&out.stdout);
    // The not-found row should land in the CSV; the `reason` column
    // is the variant-specific column carrying the diagnostic.
    let lines: Vec<&str> = text.lines().filter(|l| !l.starts_with('#')).collect();
    assert!(
        lines.len() >= 2,
        "expected header + at least one row, got: {text}"
    );
    let header = csv_header(&text)?;
    let type_idx = header
        .iter()
        .position(|h| h == "type")
        .ok_or_else(|| anyhow::anyhow!("`type` column missing from header"))?;
    let reason_idx = header
        .iter()
        .position(|h| h == "reason")
        .ok_or_else(|| anyhow::anyhow!("`reason` column missing from header"))?;
    let row = lines
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("expected at least one data row, got: {text}"))?;
    let cells: Vec<&str> = row.split(',').collect();
    assert_eq!(cells.get(type_idx).copied(), Some("not_found"));
    assert!(
        cells
            .get(reason_idx)
            .copied()
            .map(|c| !c.is_empty())
            .unwrap_or(false),
        "reason cell empty: {row}"
    );
    Ok(())
}

#[test]
fn disasm_csv_one_row_per_sass_instruction() -> Result<()> {
    // disasm's `data.rows[]` carries one KernelDisasm per kernel,
    // each with a nested Vec<SassInstruction>. The tabular view
    // denormalises: one CSV row per SASS instruction with parent-
    // kernel identity columns. So the cross-format check here looks
    // at `data.rows[0].instructions[0]` (one SASS instruction) and
    // confirms its flattened keys are present in the CSV header.
    let trace = source_metric_fixture()?;
    let env = fetch_json(&trace, "disasm", &["--row-id", "launch:0"])?;
    let csv = fetch_csv_header(&trace, "disasm", &["--row-id", "launch:0"])?;
    fetch_table_ok(&trace, "disasm", &["--row-id", "launch:0"])?;
    let Some(kernel) = env.pointer("/data/rows/0") else {
        // Empty disasm — still must have produced a header.
        assert!(!csv.is_empty(), "disasm csv has no header");
        return Ok(());
    };
    let Some(insn) = kernel.pointer("/instructions/0") else {
        // No SASS instructions in the first kernel — fine, header
        // present is enough.
        assert!(!csv.is_empty(), "disasm csv has no header");
        return Ok(());
    };
    let insn_keys = flatten_one_level_keys(insn);
    let csv_set: BTreeSet<String> = csv.iter().cloned().collect();
    // Header should contain the kernel identity columns the
    // projector denormalises.
    for kernel_col in [
        "kernel_key",
        "kernel_function_name",
        "kernel_start",
        "kernel_length",
    ] {
        assert!(
            csv_set.contains(kernel_col),
            "disasm csv header missing kernel identity column `{kernel_col}`: {csv_set:?}"
        );
    }
    // Every SASS instruction key (post one-level flatten) must
    // appear in the CSV header. The reverse doesn't hold (CSV
    // includes the kernel-* columns).
    for k in &insn_keys {
        assert!(
            csv_set.contains(k),
            "disasm csv missing SASS instruction column `{k}`: csv {csv_set:?} vs insn {insn_keys:?}"
        );
    }
    Ok(())
}

#[test]
fn schema_rejects_non_json_format() -> Result<()> {
    // ncu schema is the JSON-only exception by design. Confirm that
    // `--format csv` still rejects with the standard error envelope
    // rather than slipping through the new dispatch path.
    let out = run_veloq(["ncu", "schema", "launches", "--format", "csv"])?;
    assert!(
        !out.status.success(),
        "schema --format csv should exit non-zero"
    );
    let env: Value =
        serde_json::from_slice(&out.stdout).context("error envelope must still be valid JSON")?;
    assert!(env.pointer("/error").is_some());
    Ok(())
}
