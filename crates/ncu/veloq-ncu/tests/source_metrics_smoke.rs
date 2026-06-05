//! Smoke tests for `ncu source-metrics` using the populated source
//! counter fixture.

use anyhow::{Result, anyhow, bail};
use veloq_ncu::source_metrics::{
    self, Axis, SkippedCounter, SourceMetricsRequest, SourceMetricsRow,
};

fn fixture() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/source_metric_basic.ncu-rep")
}

fn request(counter_glob: &str, by: Axis, limit: usize) -> SourceMetricsRequest {
    SourceMetricsRequest {
        row_id: "launch:0".to_string(),
        counter_glob: counter_glob.to_string(),
        by,
        file_glob: None,
        line: None,
        sort: None,
        limit,
    }
}

#[test]
fn line_axis_returns_source_rows_for_instruction_counters() -> Result<()> {
    let r = source_metrics::run(
        fixture(),
        request("inst_executed,thread_inst_executed*", Axis::Line, 5),
    )?;

    assert_eq!(r.axis, "line");
    assert!(r.count > 0, "expected populated line rows");
    assert!(r.total_matched >= r.count);
    assert!(
        r.auxiliary
            .matched_counters
            .iter()
            .any(|name| name == "inst_executed"),
        "expected inst_executed to pass source-attribution gate: {:?}",
        r.auxiliary.matched_counters
    );

    let row = r
        .rows
        .first()
        .ok_or_else(|| anyhow!("expected at least one source-metrics row"))?;
    let SourceMetricsRow::Line(row) = row else {
        bail!("expected line row variant");
    };
    assert!(row.key.starts_with("launch:0|line:"));
    assert!(row.file.ends_with("synthetic.cu"));
    assert!(row.sass_count > 0);
    assert!(row.counters.contains_key("inst_executed"));
    assert_eq!(
        row.counters.keys().collect::<Vec<_>>(),
        row.counter_coverage.keys().collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn sass_axis_populates_instruction_text() -> Result<()> {
    let r = source_metrics::run(fixture(), request("inst_executed", Axis::Sass, 20))?;

    assert_eq!(r.axis, "sass");
    assert!(r.count > 0, "expected populated sass rows");
    let mut saw_instruction = false;
    for row in &r.rows {
        let SourceMetricsRow::Sass(row) = row else {
            bail!("expected sass row variant");
        };
        assert!(row.key.starts_with("launch:0|sass:0x"));
        if !row.opcode.is_empty() {
            saw_instruction = true;
        }
    }
    assert!(
        saw_instruction,
        "expected at least one sass row to carry opcode text"
    );
    Ok(())
}

#[test]
fn file_axis_aggregates_line_rows() -> Result<()> {
    let r = source_metrics::run(fixture(), request("inst_executed", Axis::File, 10))?;

    assert_eq!(r.axis, "file");
    assert!(r.count > 0, "expected populated file rows");
    let row = r
        .rows
        .first()
        .ok_or_else(|| anyhow!("expected at least one file row"))?;
    let SourceMetricsRow::File(row) = row else {
        bail!("expected file row variant");
    };
    assert!(row.key.starts_with("launch:0|file:"));
    assert!(row.file.ends_with("synthetic.cu"));
    assert!(row.line_count > 0);
    assert!(row.sass_count > 0);
    assert!(row.counters.contains_key("inst_executed"));
    Ok(())
}

#[test]
fn pcsamp_stall_counters_skip_when_fixture_has_no_per_pc_instances() -> Result<()> {
    let r = source_metrics::run(
        fixture(),
        request("smsp__pcsamp_warps_issue_stalled_*", Axis::Line, 5),
    )?;

    assert_eq!(r.count, 0);
    assert_eq!(r.total_matched, 0);
    assert!(
        r.auxiliary.matched_counters.is_empty(),
        "pcsamp stall counters should not be reported as source-attributed in this fixture"
    );
    assert!(
        r.auxiliary
            .skipped_counters
            .iter()
            .any(is_not_source_counter),
        "expected skipped not-a-source-counter diagnostics: {:?}",
        r.auxiliary.skipped_counters
    );
    Ok(())
}

fn is_not_source_counter(counter: &SkippedCounter) -> bool {
    counter.reason == "not-a-source-counter"
}

fn launch1_request(counter_glob: &str, by: Axis, limit: usize) -> SourceMetricsRequest {
    SourceMetricsRequest {
        row_id: "launch:1".to_string(),
        counter_glob: counter_glob.to_string(),
        by,
        file_glob: None,
        line: None,
        sort: None,
        limit,
    }
}

/// PC-sampling populated-rows AC.
///
/// `launch:1` is the pointer-chase kernel that runs long enough for
/// PC sampling to accumulate per-PC instances. NCU emits two distinct
/// counter families that share the suffix `pcsamp_warps_issue_stalled_*`:
///
/// * `smsp__pcsamp_*` (category=pm, uint64 values, correlation_id =
///   SASS VA inside the cubin) — source-attributable.
/// * `warpsampling:smsp__pcsamp_*` (category=sampler, double values,
///   correlation_id = packed non-VA quantity) — NOT
///   source-attributable.
///
/// Veloq's positive-evidence gate accepts the first and rejects the
/// second by correlation-range check. The portable glob below relies
/// on the gate doing the routing — no name-based filtering.
#[test]
fn pcsamp_pointer_chase_kernel_populates_lines_and_skips_warpsampling_family() -> Result<()> {
    let r = source_metrics::run(
        fixture(),
        launch1_request("*pcsamp_warps_issue_stalled_*", Axis::Line, 10),
    )?;

    assert_eq!(r.axis, "line");
    assert!(r.count > 0, "expected populated line rows on launch:1");

    let matched_pm: Vec<&String> = r
        .auxiliary
        .matched_counters
        .iter()
        .filter(|n| !n.starts_with("warpsampling:"))
        .collect();
    assert!(
        !matched_pm.is_empty(),
        "expected unprefixed smsp__pcsamp_* counters in matched_counters; got: {:?}",
        r.auxiliary.matched_counters
    );

    let matched_warpsampling = r
        .auxiliary
        .matched_counters
        .iter()
        .filter(|n| n.starts_with("warpsampling:"))
        .count();
    assert_eq!(
        matched_warpsampling, 0,
        "warpsampling: family should be skipped by the cubin-range gate, not matched"
    );

    let skipped_warpsampling = r
        .auxiliary
        .skipped_counters
        .iter()
        .filter(|s| s.name.starts_with("warpsampling:") && s.reason == "not-a-source-counter")
        .count();
    assert!(
        skipped_warpsampling > 0,
        "expected warpsampling: family to land in skipped_counters; got: {:?}",
        r.auxiliary.skipped_counters
    );

    // The pointer-chase load line should be the dominant
    // long_scoreboard stall site.
    let mut max_stalls = 0.0_f64;
    for row in &r.rows {
        let SourceMetricsRow::Line(line) = row else {
            continue;
        };
        if let Some(Some(v)) = line
            .counters
            .get("smsp__pcsamp_warps_issue_stalled_long_scoreboard")
            && *v > max_stalls
        {
            max_stalls = *v;
        }
    }
    assert!(
        max_stalls > 0.0,
        "expected at least one line with long_scoreboard > 0 on the pointer-chase kernel"
    );
    Ok(())
}

#[test]
fn pcsamp_pointer_chase_kernel_sass_axis_carries_opcodes() -> Result<()> {
    let r = source_metrics::run(
        fixture(),
        launch1_request("*pcsamp_warps_issue_stalled_*", Axis::Sass, 30),
    )?;
    assert_eq!(r.axis, "sass");
    assert!(r.count > 0, "expected populated sass rows on launch:1");
    let with_opcode = r.rows.iter().any(|row| {
        if let SourceMetricsRow::Sass(s) = row {
            !s.opcode.is_empty() && !s.counters.is_empty()
        } else {
            false
        }
    });
    assert!(
        with_opcode,
        "expected at least one SASS row to carry opcode + counters"
    );
    Ok(())
}

#[test]
fn pcsamp_pointer_chase_kernel_file_axis_aggregates() -> Result<()> {
    let r = source_metrics::run(
        fixture(),
        launch1_request("*pcsamp_warps_issue_stalled_*", Axis::File, 5),
    )?;
    assert_eq!(r.axis, "file");
    assert!(r.count > 0, "expected populated file rows on launch:1");
    Ok(())
}

/// End-to-end coverage of the
/// `in_cubin_no_source` -> `unattributed_sass` routing on REAL
/// placement tags. The fixture is a committed *sidecar only*
/// (`unattributed_basic.ncu-native.json.gz`) — its `.ncu-rep` is not
/// committed because the binary embeds a hostname; the native sidecar
/// keeps only a version-only session block (no host identifiers), so it
/// is leak-free. NOTE: with no parent `.ncu-rep` this sidecar is a
/// frozen, non-regenerable-in-tree golden (the test is NCU-free by
/// design). It was captured from a kernel built WITHOUT `-lineinfo`, so
/// every pcsamp PC is in-cubin but
/// carries no DWARF line -> placement `in_cubin_no_source` -> the whole
/// budget lands in `unattributed_sass`, with zero line rows.
#[test]
fn no_lineinfo_fixture_fills_unattributed_sass_and_reconciles() -> Result<()> {
    const COUNTER: &str = "smsp__pcsamp_sample_count";
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/unattributed_basic.ncu-native.json.gz");
    let sc = veloq_ncu::native::cache::read_gz_sidecar(&path)?;

    let r = source_metrics::run_on_sidecar(
        &sc,
        SourceMetricsRequest {
            row_id: "launch:0".to_string(),
            counter_glob: COUNTER.to_string(),
            by: Axis::Line,
            file_glob: None,
            line: None,
            sort: None,
            limit: 100,
        },
    )?;

    // No DWARF -> no attributed line rows; the budget is unattributed.
    assert_eq!(
        r.count, 0,
        "no -lineinfo should yield zero attributed line rows"
    );
    let unattr = r
        .auxiliary
        .unattributed_sass_counter_totals
        .get(COUNTER)
        .copied()
        .ok_or_else(|| anyhow!("expected a non-empty unattributed_sass bucket"))?;
    assert!(
        unattr > 0.0,
        "unattributed_sass must be non-zero, got {unattr}"
    );
    assert!(r.auxiliary.unattributed_sass_count > 0);
    assert!(
        r.auxiliary.out_of_cubin_counter_totals.is_empty(),
        "pcsamp PCs are all in-cubin -> out_of_cubin bucket stays empty"
    );

    // Reconciliation identity: Σ rows + unattributed + out_of_cubin ==
    // the kernel-level total (the metric's aggregate value in the
    // sidecar). Self-contained — no `.ncu-rep` / `ncu metrics` needed.
    let launch = sc.launches.first().ok_or_else(|| anyhow!("no launch:0"))?;
    let kernel_total = launch
        .metric(COUNTER)
        .and_then(|m| m.value.as_f64())
        .ok_or_else(|| anyhow!("counter total missing on launch"))?;
    let row_sum: f64 = r
        .rows
        .iter()
        .map(|row| match row {
            SourceMetricsRow::Line(l) => l.counters.get(COUNTER).copied().flatten().unwrap_or(0.0),
            _ => 0.0,
        })
        .sum();
    let oob = r
        .auxiliary
        .out_of_cubin_counter_totals
        .get(COUNTER)
        .copied()
        .unwrap_or(0.0);
    assert!(
        (row_sum + unattr + oob - kernel_total).abs() < 1e-6,
        "reconciliation failed: rows({row_sum}) + unattr({unattr}) + oob({oob}) != total({kernel_total})"
    );
    Ok(())
}
