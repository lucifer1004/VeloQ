//! Smoke tests for `ncu launches` + `ncu inspect`, both running
//! off the bundled `vector_add_basic.ncu-rep` fixture (one launch).
//!
//! Covers the contract (rows + count + auxiliary), the headline
//! columns on `LaunchRow`, and the round-trip from a `launches`
//! row's `row_id` back to a full `LaunchEntry` through `inspect`.

use anyhow::{Result, anyhow, bail};
use veloq_ncu::inspect::{self, LaunchDetailsRow};
use veloq_ncu::launches::{self, LaunchesRequest};

fn fixture() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vector_add_basic.ncu-rep")
}

#[test]
fn launches_lists_every_launch_with_v2_headline_columns() -> Result<()> {
    let r = launches::run(
        fixture(),
        LaunchesRequest {
            limit: 100,
            ..Default::default()
        },
    )?;
    assert_eq!(r.count, 1);
    assert_eq!(r.total_matched, 1);
    let row = r
        .rows
        .first()
        .ok_or_else(|| anyhow!("expected one launch row"))?;
    assert_eq!(row.row_id, "launch:0");
    assert_eq!(row.key, "launch:0");
    assert_eq!(
        row.kernel_demangled,
        "vector_add(const float *, const float *, float *, int)"
    );
    assert_eq!(row.grid_size, [4096, 1, 1]);
    assert_eq!(row.block_size, [256, 1, 1]);
    // auxiliary echoes the (empty) filter set + a non-empty cache path.
    assert!(r.auxiliary.kernel_glob.is_none());
    assert!(!r.auxiliary.meta_cache_path.is_empty());
    Ok(())
}

#[test]
fn launches_kernel_glob_filters_by_demangled_or_mangled() -> Result<()> {
    let hit = launches::run(
        fixture(),
        LaunchesRequest {
            kernel_glob: Some("*vector_add*".into()),
            limit: 100,
            ..Default::default()
        },
    )?;
    assert_eq!(hit.count, 1, "demangled glob should match the one launch");

    let miss = launches::run(
        fixture(),
        LaunchesRequest {
            kernel_glob: Some("*NoSuchKernel*".into()),
            limit: 100,
            ..Default::default()
        },
    )?;
    assert_eq!(miss.count, 0);
    assert_eq!(miss.total_matched, 0);
    Ok(())
}

#[test]
fn launches_grid_filter_uses_zero_as_wildcard() -> Result<()> {
    // `4096x0x0` should match — first axis exact, others wildcards.
    let r = launches::run(
        fixture(),
        LaunchesRequest {
            grid: Some([4096, 0, 0]),
            limit: 100,
            ..Default::default()
        },
    )?;
    assert_eq!(r.count, 1);

    // `1024x0x0` should miss — first axis doesn't match.
    let r = launches::run(
        fixture(),
        LaunchesRequest {
            grid: Some([1024, 0, 0]),
            limit: 100,
            ..Default::default()
        },
    )?;
    assert_eq!(r.count, 0);
    Ok(())
}

#[test]
fn inspect_resolves_launch_row_id_to_full_details() -> Result<()> {
    let r = inspect::run(fixture(), &["launch:0".into()])?;
    assert_eq!(r.count, 1);
    let row = r
        .rows
        .first()
        .ok_or_else(|| anyhow!("expected one inspect row"))?;
    let details = match row {
        LaunchDetailsRow::Launch(d) => d,
        LaunchDetailsRow::NotFound { reason, .. } => {
            bail!("expected Launch, got NotFound: {reason}")
        }
    };
    assert_eq!(details.key, "launch:0");
    assert_eq!(details.row_id, "launch:0");
    // Native model: fields are flat, there is no section catalog, and
    // metrics/rules are carried in full with recovered counts.
    assert_eq!(
        details.kernel_demangled,
        "vector_add(const float *, const float *, float *, int)"
    );
    assert!(details.metric_count > 0);
    assert_eq!(details.rule_count, details.rules.len());
    assert_eq!(details.metric_count, details.metrics.len());
    Ok(())
}

#[test]
fn inspect_out_of_range_row_id_returns_not_found_not_error() -> Result<()> {
    let r = inspect::run(fixture(), &["launch:42".into()])?;
    assert_eq!(r.count, 1);
    match r.rows.first().ok_or_else(|| anyhow!("expected one row"))? {
        LaunchDetailsRow::NotFound { row_id, reason, .. } => {
            assert_eq!(row_id, "launch:42");
            assert!(
                reason.contains("out of range"),
                "reason should explain why ({reason})"
            );
        }
        LaunchDetailsRow::Launch(_) => bail!("expected NotFound for out-of-range row_id"),
    }
    Ok(())
}

#[test]
fn inspect_rejects_unknown_row_id_kinds_as_not_found() -> Result<()> {
    let r = inspect::run(fixture(), &["range:0".into()])?;
    match r.rows.first().ok_or_else(|| anyhow!("expected one row"))? {
        LaunchDetailsRow::NotFound { reason, .. } => {
            assert!(
                reason.contains("launch:<idx>"),
                "should hint at the supported form ({reason})"
            );
        }
        LaunchDetailsRow::Launch(_) => bail!("expected NotFound for `range:0` row_id"),
    }
    Ok(())
}
