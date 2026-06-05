//! End-to-end smoke tests:
//! - `--help` reorder (Recipes block + Common-flags matrix appear
//!   above the Response schema)
//! - narrow-window and empty-with-scope guardrails populate
//!   `meta.warnings[]`
//! - registered recipes pass a CLI-parser smoke

use anyhow::{Context, Result};
use duckdb::{Connection, params};
use serde_json::Value;
use std::path::PathBuf;
use std::process::{Command, Output};
use tempfile::TempDir;

fn veloq_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_veloq"))
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

fn parse_stdout(out: &Output) -> Result<Value> {
    serde_json::from_slice(&out.stdout).context("veloq stdout must be valid JSON")
}

fn at<'a>(v: &'a Value, ptr: &str) -> Result<&'a Value> {
    v.pointer(ptr)
        .ok_or_else(|| anyhow::anyhow!("missing pointer `{ptr}` in {v}"))
}

fn finalize_to_pqtdir(conn: &Connection, dir: &TempDir) -> Result<PathBuf> {
    let pqtdir = dir.path().join("test_pqtdir");
    std::fs::create_dir_all(&pqtdir).context("create parquetdir")?;
    let mut stmt = conn.prepare(
        "SELECT table_name FROM information_schema.tables \
         WHERE table_schema = 'main' AND table_type = 'BASE TABLE'",
    )?;
    let mut rows = stmt.query([])?;
    let mut tables = Vec::new();
    while let Some(r) = rows.next()? {
        tables.push(r.get::<_, String>(0)?);
    }
    for table in &tables {
        let out = pqtdir.join(format!("{table}.parquet"));
        let out_lit = out.to_string_lossy().replace('\'', "''");
        conn.execute(
            &format!(r#"COPY (SELECT * FROM "{table}") TO '{out_lit}' (FORMAT PARQUET)"#),
            [],
        )?;
    }
    Ok(pqtdir)
}

fn install_minimal_export_metadata(conn: &Connection) -> Result<()> {
    conn.execute_batch("CREATE TABLE IF NOT EXISTS META_DATA_EXPORT (name TEXT, value TEXT);")?;
    for (k, v) in [
        ("EXPORT_SCHEMA_VERSION_MAJOR", "3"),
        ("EXPORT_SCHEMA_VERSION_MINOR", "0"),
        ("EXPORT_SCHEMA_VERSION_MICRO", "0"),
    ] {
        conn.execute(
            "INSERT INTO META_DATA_EXPORT (name, value) VALUES (?, ?)",
            params![k, v],
        )?;
    }
    Ok(())
}

/// Wide-span trace (10 s of kernel activity on one device) so the
/// narrow-window guardrail can fire on a sub-1% window.
fn build_long_span_trace() -> Result<(TempDir, PathBuf)> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    conn.execute_batch(
        r#"
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        CREATE TABLE TARGET_INFO_GPU (id BIGINT, cuDevice BIGINT, uuid TEXT);
        CREATE TABLE PROCESSES (globalPid BIGINT, pid BIGINT, name TEXT);
        CREATE TABLE NVTX_EVENTS (
            start BIGINT, "end" BIGINT, eventType BIGINT,
            globalTid BIGINT, domainId BIGINT, text TEXT, textId BIGINT
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start BIGINT, "end" BIGINT,
            globalTid BIGINT, correlationId BIGINT, nameId BIGINT
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start BIGINT, "end" BIGINT,
            deviceId BIGINT, contextId BIGINT, streamId BIGINT,
            shortName BIGINT, demangledName BIGINT,
            gridX BIGINT, gridY BIGINT, gridZ BIGINT,
            blockX BIGINT, blockY BIGINT, blockZ BIGINT,
            correlationId BIGINT,
            registersPerThread BIGINT,
            staticSharedMemory BIGINT,
            dynamicSharedMemory BIGINT,
            globalPid BIGINT,
            graphId BIGINT, graphNodeId BIGINT
        );
        "#,
    )?;
    install_minimal_export_metadata(&conn)?;
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1_i64, "smoke_kernel"],
    )?;
    conn.execute(
        "INSERT INTO TARGET_INFO_GPU (id, cuDevice, uuid) VALUES (?, ?, ?)",
        params![10_i64, 0_i64, "synthetic-gpu-0"],
    )?;
    conn.execute(
        "INSERT INTO PROCESSES (globalPid, pid, name) VALUES (?, ?, ?)",
        params![(4242_i64 << 24), 4242_i64, "synthetic-host"],
    )?;
    // 10-second span: kernels every 100 ms across 10 s. Plenty of
    // dynamic range for the narrow-window check.
    for i in 0..100_i64 {
        let start = i * 100_000_000;
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
             (start, \"end\", deviceId, contextId, streamId, \
              shortName, demangledName, gridX, gridY, gridZ, \
              blockX, blockY, blockZ, correlationId, \
              registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                start,
                start + 50_000_000,
                0_i64,
                100_i64,
                7_i64,
                1_i64,
                1_i64,
                1_i64,
                1_i64,
                1_i64,
                128_i64,
                1_i64,
                1_i64,
                1_i64,
                32_i64,
                0_i64,
                0_i64,
                4242_i64 << 24,
            ],
        )?;
    }
    let pqtdir = finalize_to_pqtdir(&conn, &dir)?;
    Ok((dir, pqtdir))
}

/// `veloq stats --help` projects a Recipes-for-this-verb block and a
/// Common-flags matrix above the Response schema. The exact text
/// changes as recipes evolve; we just assert the *ordering* relative
/// to "Response (.data):".
#[test]
fn stats_help_orders_recipes_above_response() -> Result<()> {
    let out = run_veloq(["stats", "--help"])?;
    assert!(out.status.success());
    let s = String::from_utf8_lossy(&out.stdout);
    let recipes = s
        .find("Recipes for this verb")
        .context("stats --help must surface a Recipes block")?;
    let common_flags = s
        .find("Common flags:")
        .context("stats --help must surface a Common-flags matrix")?;
    let response = s
        .find("Response (.data):")
        .context("stats --help still emits the response schema")?;
    assert!(recipes < response, "Recipes must precede Response (.data)");
    assert!(
        common_flags < response,
        "Common flags must precede Response (.data)"
    );
    assert!(
        s.contains("--device <N>"),
        "Common flags must spell out --device <N>; got:\n{s}",
    );
    Ok(())
}

/// `veloq ncu source-metrics --help` runs through the NCU help
/// projector (`crates/ncu/veloq-ncu/src/help.rs`, separate from the
/// NSys projector covered above). The projector injects only a
/// Recipes block — NCU verbs don't share the location/window/nvtx
/// flag family that drives the Common-flags matrix on the NSys side.
/// This test pins that the new `source-metrics` verb is wired into
/// `NCU_VERBS` and the registered recipes (`source-line-hotspots`,
/// `source-instruction-walk`) actually surface in its help output.
#[test]
fn ncu_source_metrics_help_surfaces_recipes_block() -> Result<()> {
    let out = run_veloq(["ncu", "source-metrics", "--help"])?;
    assert!(
        out.status.success(),
        "exit={:?}; stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
    );
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(
        s.contains("Recipes for this verb"),
        "ncu source-metrics --help must surface a Recipes block; \
         regression in NCU_VERBS list or the help projector. Output:\n{s}",
    );
    // Two recipes target source-metrics today; the help projector
    // must list both. The exact title text comes from the registry
    // so an assertion on the id is the load-bearing pin.
    assert!(
        s.contains("source-line-hotspots"),
        "expected `source-line-hotspots` recipe id in help output; got:\n{s}",
    );
    assert!(
        s.contains("source-instruction-walk"),
        "expected `source-instruction-walk` recipe id in help output; got:\n{s}",
    );
    Ok(())
}

/// `veloq recipes` projects every registered recipe. WI-C populates
/// the canonical set; assert it's non-empty and contains the
/// `nvtx-breakdown` recipe specifically (a stable anchor).
#[test]
fn recipes_list_contains_expected_canonical_ids() -> Result<()> {
    let out = run_veloq(["recipes"])?;
    assert!(out.status.success());
    let v = parse_stdout(&out)?;
    let rows = at(&v, "/data/rows")?
        .as_array()
        .context("data.rows must be an array")?;
    let ids: Vec<&str> = rows
        .iter()
        .filter_map(|r| r.pointer("/id").and_then(Value::as_str))
        .collect();
    for expected in ["nvtx-breakdown", "gpu-idle-audit", "memcpy-asymmetry"] {
        assert!(
            ids.contains(&expected),
            "expected `{expected}` in recipe ids; got: {ids:?}",
        );
    }
    Ok(())
}

/// Every recipe's `body` (fetched via `veloq recipes <id>`) parses as
/// a sequence of shell-style lines where each non-blank, non-comment,
/// non-continuation line starts with `veloq ` or a recognised shell
/// scaffolding token. Catches accidental command-name typos.
#[test]
fn every_recipe_body_starts_lines_with_veloq_or_shell_scaffolding() -> Result<()> {
    // First gather the recipe ids from the list payload.
    let list = run_veloq(["recipes"])?;
    assert!(list.status.success());
    let v = parse_stdout(&list)?;
    let ids: Vec<String> = at(&v, "/data/rows")?
        .as_array()
        .context("data.rows must be an array")?
        .iter()
        .filter_map(|r| r.pointer("/id").and_then(Value::as_str).map(str::to_string))
        .collect();
    assert!(!ids.is_empty());
    let allowed_prefixes = ["veloq ", "for ", "do", "done", "#"];
    for id in &ids {
        let detail = run_veloq(["recipes", id])?;
        assert!(
            detail.status.success(),
            "veloq recipes {id} failed; stderr: {}",
            String::from_utf8_lossy(&detail.stderr),
        );
        let dv = parse_stdout(&detail)?;
        let body = at(&dv, "/data/body")?
            .as_str()
            .context("data.body must be a string")?;
        // Backslash line-continuation: every line that immediately
        // follows one ending in `\` is the tail of the previous
        // logical command. We only validate the start of *logical*
        // lines so the indented `--flag value` continuations don't
        // trip the prefix check.
        let mut continuation = false;
        for line in body.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continuation = false;
                continue;
            }
            if !continuation {
                assert!(
                    allowed_prefixes.iter().any(|p| trimmed.starts_with(p)),
                    "recipe `{id}` body line `{line}` doesn't start with a recognised prefix",
                );
            }
            continuation = trimmed.ends_with('\\');
        }
    }
    Ok(())
}

/// Narrow time window vs. the trace span produces a
/// `meta.warnings[0].code == "narrow-window"` hint. Use `@<ns>` so the
/// resolver lands the window absolutely (relative endpoints anchor on
/// the trace origin and might end up wider than 1% in tests).
#[test]
fn narrow_window_warning_fires() -> Result<()> {
    let (_dir, pqtdir) = build_long_span_trace()?;
    // 1-microsecond window vs. ~10 s span = 0.00001% — well under 1%.
    let out = run_veloq([
        "stats",
        pqtdir.to_string_lossy().as_ref(),
        "--from",
        "@1000",
        "--to",
        "@2000",
    ])?;
    assert!(
        out.status.success(),
        "exit={:?}; stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
    );
    let v = parse_stdout(&out)?;
    let warnings = at(&v, "/meta/warnings")?
        .as_array()
        .context("meta.warnings must be an array")?;
    let codes: Vec<&str> = warnings
        .iter()
        .filter_map(|w| w.pointer("/code").and_then(Value::as_str))
        .collect();
    assert!(
        codes.contains(&"narrow-window"),
        "expected `narrow-window` warning; got codes: {codes:?}; full envelope: {v}",
    );
    Ok(())
}

/// Empty result under an explicit scope filter produces an
/// `empty-with-scope` warning.
#[test]
fn empty_with_scope_warning_fires() -> Result<()> {
    let (_dir, pqtdir) = build_long_span_trace()?;
    // Stream 999 doesn't exist in the fixture — the device filter
    // resolves but the stream filter excludes every row.
    let out = run_veloq([
        "stats",
        pqtdir.to_string_lossy().as_ref(),
        "--device",
        "0",
        "--stream",
        "999",
    ])?;
    assert!(
        out.status.success(),
        "exit={:?}; stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
    );
    let v = parse_stdout(&out)?;
    let warnings = at(&v, "/meta/warnings")?
        .as_array()
        .context("meta.warnings must be an array")?;
    let codes: Vec<&str> = warnings
        .iter()
        .filter_map(|w| w.pointer("/code").and_then(Value::as_str))
        .collect();
    assert!(
        codes.contains(&"empty-with-scope"),
        "expected `empty-with-scope` warning; got codes: {codes:?}; full envelope: {v}",
    );
    Ok(())
}
