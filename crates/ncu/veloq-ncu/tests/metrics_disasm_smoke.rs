//! Smoke tests for `ncu metrics` (long + wide) and `ncu disasm`.
//!
//! The bundled `vector_add_basic.ncu-rep` fixture has one launch
//! with ~600 metrics; that's enough to exercise the filter +
//! projection on `metrics`. The fixture's cubin is zero-byte, so
//! `disasm` exercises the "no SASS" path — the warning surfaces
//! through `auxiliary.warnings` rather than failing the verb.

use anyhow::{Result, anyhow, bail};
use veloq_ncu::disasm;
use veloq_ncu::metrics::{self, MetricRow, MetricsFormat, MetricsRequest};

fn fixture() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vector_add_basic.ncu-rep")
}

#[test]
fn metrics_long_format_emits_launch_counter_rows_with_keys() -> Result<()> {
    let r = metrics::run(
        fixture(),
        MetricsRequest {
            counter_glob: "launch__*".to_string(),
            limit: 1000,
            ..Default::default()
        },
    )?;
    assert!(r.count > 0, "expected at least one launch__* metric");
    assert_eq!(r.count, r.total_matched);
    assert!(
        matches!(r.format, MetricsFormat::Long),
        "expected long format by default"
    );
    let row = r
        .rows
        .first()
        .ok_or_else(|| anyhow!("expected at least one metric row"))?;
    let MetricRow::Long(row) = row else {
        bail!("expected long row variant");
    };
    assert!(
        row.key.starts_with("launch:0|counter:launch__"),
        "key should be `launch:<idx>|counter:<name>`: {}",
        row.key
    );
    assert_eq!(row.launch_row_id, "launch:0");
    assert!(row.counter_name.starts_with("launch__"));
    assert_eq!(r.auxiliary.counter_glob, "launch__*");
    assert!(!r.auxiliary.meta_cache_path.is_empty());
    Ok(())
}

#[test]
fn metrics_per_launch_format_clusters_counters_under_launch() -> Result<()> {
    let r = metrics::run(
        fixture(),
        MetricsRequest {
            counter_glob: "launch__grid_*".to_string(),
            per_launch: true,
            limit: 100,
            ..Default::default()
        },
    )?;
    assert_eq!(r.count, 1, "fixture has one launch");
    assert!(
        matches!(r.format, MetricsFormat::PerLaunch),
        "expected wide format with --per-launch"
    );
    let row = r.rows.first().ok_or_else(|| anyhow!("expected one row"))?;
    let MetricRow::PerLaunch(row) = row else {
        bail!("expected per-launch row variant");
    };
    assert_eq!(row.row_id, "launch:0");
    assert!(
        row.counters.contains_key("launch__grid_size"),
        "expected launch__grid_size in counters map: {:?}",
        row.counters.keys().collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn metrics_unknown_glob_returns_empty_rows_not_error() -> Result<()> {
    let r = metrics::run(
        fixture(),
        MetricsRequest {
            counter_glob: "no_such_counter_*".to_string(),
            limit: 100,
            ..Default::default()
        },
    )?;
    assert_eq!(r.count, 0);
    assert_eq!(r.total_matched, 0);
    Ok(())
}

#[test]
fn disasm_zero_byte_cubin_returns_empty_rows_with_warning() -> Result<()> {
    // vector_add_basic ships with cubin_bytes=0 — disasm can't
    // produce SASS, but the verb should degrade gracefully via the
    // auxiliary.warnings channel rather than erroring out.
    let r = disasm::run(fixture(), "launch:0")?;
    assert_eq!(r.count, 0);
    assert_eq!(r.total_matched, 0);
    assert_eq!(r.auxiliary.row_id, "launch:0");
    // A zero-byte cubin yields no SASS, which surfaces a non-empty
    // warnings list. Exact wording isn't pinned; the contract is
    // "warnings explain why rows is empty".
    assert!(
        !r.auxiliary.warnings.is_empty(),
        "expected at least one diagnostic warning explaining the empty disasm: {:?}",
        r.auxiliary.warnings
    );
    Ok(())
}

#[test]
fn disasm_rejects_unknown_row_id_kind() -> Result<()> {
    match disasm::run(fixture(), "range:0") {
        Ok(_) => bail!("expected error for `range:0` row_id"),
        Err(err) => {
            let msg = format!("{err:#}");
            assert!(
                msg.contains("launch:<idx>"),
                "error should hint at the supported form: {msg}"
            );
        }
    }
    Ok(())
}
