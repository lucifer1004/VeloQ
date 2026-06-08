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

fn assert_not_found_row(
    row: &LaunchDetailsRow,
    expected_row_id: &str,
    reason_needle: &str,
) -> Result<()> {
    match row {
        LaunchDetailsRow::NotFound {
            key,
            row_id,
            reason,
        } => {
            assert_eq!(key, expected_row_id);
            assert_eq!(row_id, expected_row_id);
            assert!(
                reason.contains(reason_needle),
                "reason should mention `{reason_needle}` ({reason})"
            );
        }
        LaunchDetailsRow::Launch(_) => bail!("expected NotFound for `{expected_row_id}` row_id"),
    }
    Ok(())
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
    assert_not_found_row(
        r.rows.first().ok_or_else(|| anyhow!("expected one row"))?,
        "launch:42",
        "out of range",
    )?;
    Ok(())
}

#[test]
fn inspect_malformed_row_ids_return_not_found_not_error() -> Result<()> {
    let row_ids = vec!["launch0".to_string(), "launch:abc".to_string()];
    let r = inspect::run(fixture(), &row_ids)?;
    assert_eq!(r.count, row_ids.len());
    assert_eq!(r.total_matched, row_ids.len());

    let mut rows = r.rows.iter();
    assert_not_found_row(
        rows.next().ok_or_else(|| anyhow!("expected first row"))?,
        "launch0",
        "expected",
    )?;
    assert_not_found_row(
        rows.next().ok_or_else(|| anyhow!("expected second row"))?,
        "launch:abc",
        "invalid launch index",
    )?;
    assert!(
        rows.next().is_none(),
        "inspect should return one row per requested row_id"
    );
    Ok(())
}

#[test]
fn inspect_rejects_unknown_row_id_kinds_as_not_found() -> Result<()> {
    let r = inspect::run(fixture(), &["range:0".into()])?;
    assert_not_found_row(
        r.rows.first().ok_or_else(|| anyhow!("expected one row"))?,
        "range:0",
        "launch:<idx>",
    )?;
    Ok(())
}
