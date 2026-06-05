//! `--group-by mangled` axis:
//! - kernel rows split per mangled symbol, even when shortName and
//!   demangled signature collide
//! - missing `mangledName` column falls back to demangled axis with
//!   `mangled_axis_fallback = true` on the response
//! - the `mangled` token is recognised by the GroupBy parser

mod fixture;

use anyhow::Result;
use veloq_nsys_query::stats::{GroupBy, NameAxis, StatsRequest};
use veloq_nsys_query::{EventKind, KindFilter};

#[test]
fn mangled_axis_splits_kernels_that_share_demangled() -> Result<()> {
    let trace = fixture::kernel_with_mangled_names()?;

    // Demangled axis: 2 rows (rows 1+2 fold; row 3 distinct).
    let dem = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            group_by: GroupBy {
                name: NameAxis::Demangled,
                ..Default::default()
            },
            ..Default::default()
        },
    )?;
    assert_eq!(
        dem.total_matched,
        2,
        "demangled axis folds same-demangled rows; got rows: {:?}",
        dem.rows
            .iter()
            .map(|r| (r.name.clone(), r.count))
            .collect::<Vec<_>>()
    );
    assert!(!dem.mangled_axis_fallback);

    // Mangled axis: 3 rows (each mangled symbol unique).
    let mng = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            group_by: GroupBy {
                name: NameAxis::Mangled,
                ..Default::default()
            },
            ..Default::default()
        },
    )?;
    assert_eq!(
        mng.total_matched,
        3,
        "mangled axis preserves per-symbol identity; got rows: {:?}",
        mng.rows
            .iter()
            .map(|r| (r.name.clone(), r.count))
            .collect::<Vec<_>>()
    );
    assert!(!mng.mangled_axis_fallback);

    // Mangled names surface as the row's `name` field.
    let names: std::collections::HashSet<_> =
        mng.rows.iter().filter_map(|r| r.name.clone()).collect();
    assert!(names.contains("_Z8MyKernelIiEvPi"), "got {names:?}");
    assert!(names.contains("_Z8MyKernelIiEvPiS_"), "got {names:?}");
    assert!(names.contains("_Z11OtherKernelPf"), "got {names:?}");
    Ok(())
}

#[test]
fn mangled_axis_keeps_non_kernel_rows_distinct() -> Result<()> {
    // Non-kernel rows under --group-by mangled must carry their
    // display-name fallback (not NULL); composite keys must stay
    // distinct so each bucket is joinable.
    let trace = fixture::minimal_gpu()?;
    let r = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Memcpy, EventKind::Memset]),
            group_by: GroupBy {
                name: NameAxis::Mangled,
                ..Default::default()
            },
            ..Default::default()
        },
    )?;
    // No row may carry a NULL name under --group-by mangled on
    // these kinds — display_name fallback always resolves.
    for row in &r.rows {
        assert!(
            row.name.is_some(),
            "non-kernel row missing name under --group-by mangled: {row:?}"
        );
    }
    // Composite keys must be distinct across rows.
    let keys: std::collections::HashSet<_> = r.rows.iter().map(|x| x.key.clone()).collect();
    assert_eq!(
        keys.len(),
        r.rows.len(),
        "duplicate composite keys under --group-by mangled: {:?}",
        r.rows
            .iter()
            .map(|x| (x.kind, x.key.clone()))
            .collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn mangled_axis_falls_back_to_demangled_when_column_absent() -> Result<()> {
    // Trace without `mangledName` downgrades to Demangled silently
    // and surfaces `mangled_axis_fallback = true` on the response.
    let trace = fixture::minimal_gpu()?;
    let mng = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            group_by: GroupBy {
                name: NameAxis::Mangled,
                ..Default::default()
            },
            ..Default::default()
        },
    )?;
    assert!(
        mng.mangled_axis_fallback,
        "trace without mangledName column must report axis fallback"
    );
    // The downgrade lands on the demangled axis, so we expect the
    // same row count as an explicit Demangled request.
    let dem = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            group_by: GroupBy {
                name: NameAxis::Demangled,
                ..Default::default()
            },
            ..Default::default()
        },
    )?;
    assert_eq!(mng.total_matched, dem.total_matched);
    assert!(!dem.mangled_axis_fallback);
    Ok(())
}
