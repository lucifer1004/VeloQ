//! `--group-by grid_block`:
//! - kernel rows split by launch shape (grid×block) even when names match
//! - StatRow surfaces grid_x/y/z and block_x/y/z fields when axis active
//! - composes with the name axis (demangled,grid_block) and physical
//!   axes (device,grid_block)
//! - non-kernel kinds in --type error up-front (kernel-only columns)

mod fixture;

use anyhow::{Result, anyhow};
use veloq_nsys_query::stats::{GroupBy, NameAxis, StatsRequest};
use veloq_nsys_query::{EventKind, KindFilter};

fn grid_block_axis() -> GroupBy {
    GroupBy {
        grid_block: true,
        ..Default::default()
    }
}

#[test]
fn grid_block_splits_same_name_kernels_with_different_shapes() -> Result<()> {
    let trace = fixture::kernels_with_launch_configs()?;

    // Without the axis: all 4 kernels share shortName → 1 row.
    let baseline = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            ..Default::default()
        },
    )?;
    assert_eq!(baseline.total_matched, 1);
    for row in &baseline.rows {
        // Non-axis rows must NOT surface grid/block fields.
        assert!(
            row.grid_x.is_none() && row.block_x.is_none(),
            "non-axis row leaked grid/block"
        );
    }

    // With grid_block axis: 3 distinct launch shapes.
    let by_shape = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            group_by: grid_block_axis(),
            ..Default::default()
        },
    )?;
    assert_eq!(
        by_shape.total_matched,
        3,
        "grid_block axis splits launch configs; got {:?}",
        by_shape
            .rows
            .iter()
            .map(|r| (
                r.grid_x, r.grid_y, r.grid_z, r.block_x, r.block_y, r.block_z
            ))
            .collect::<Vec<_>>()
    );

    // Sum of bucket totals must equal trace-wide total.
    let bucket_total: i64 = by_shape.rows.iter().map(|r| r.total_ns).sum();
    assert_eq!(bucket_total, baseline.total_duration_ns);

    // Each row must surface all six axis fields.
    for row in &by_shape.rows {
        assert!(row.grid_x.is_some(), "grid_x missing");
        assert!(row.grid_y.is_some());
        assert!(row.grid_z.is_some());
        assert!(row.block_x.is_some());
        assert!(row.block_y.is_some());
        assert!(row.block_z.is_some());
        // Composite key carries grid:/block: segments.
        assert!(row.key.contains("grid:"), "key missing grid: segment");
        assert!(row.key.contains("block:"), "key missing block: segment");
    }
    Ok(())
}

#[test]
fn composes_with_demangled_name_axis() -> Result<()> {
    // demangled,grid_block: all four kernels share demangledName=1, so
    // the cardinality still comes from the grid_block axis alone (3 rows).
    let trace = fixture::kernels_with_launch_configs()?;
    let r = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            group_by: GroupBy {
                name: NameAxis::Demangled,
                grid_block: true,
                ..Default::default()
            },
            ..Default::default()
        },
    )?;
    assert_eq!(r.total_matched, 3);
    for row in &r.rows {
        assert!(row.name.is_some());
        assert!(row.grid_x.is_some());
    }
    Ok(())
}

#[test]
fn composes_with_device_axis() -> Result<()> {
    // device + grid_block: only one device in the fixture, so we still
    // get 3 rows but each one carries device_id=0.
    let trace = fixture::kernels_with_launch_configs()?;
    let r = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            group_by: GroupBy {
                grid_block: true,
                device: true,
                ..Default::default()
            },
            ..Default::default()
        },
    )?;
    assert_eq!(r.total_matched, 3);
    for row in &r.rows {
        assert_eq!(row.device_id, Some(0));
        assert!(row.grid_x.is_some());
    }
    Ok(())
}

#[test]
fn rejects_grid_block_with_non_kernel_kind() -> Result<()> {
    let trace = fixture::kernels_with_launch_configs()?;
    let outcome = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Memcpy]),
            group_by: grid_block_axis(),
            ..Default::default()
        },
    );
    let err = match outcome {
        Ok(_) => return Err(anyhow!("expected reject for memcpy+grid_block, got Ok")),
        Err(e) => e,
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("kernel-only") && msg.contains("memcpy"),
        "got: {msg}"
    );
    Ok(())
}

#[test]
fn grid_block_narrows_kindfilter_all_to_kernel() -> Result<()> {
    // With KindFilter::All + grid_block axis, stats narrows to kernel
    // implicitly (other kinds drop out via the .filter() pre-SQL).
    // Result must match the explicit `--type kernel` count.
    let trace = fixture::kernels_with_launch_configs()?;
    let r_all = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::All,
            group_by: grid_block_axis(),
            ..Default::default()
        },
    )?;
    let r_kernel = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            group_by: grid_block_axis(),
            ..Default::default()
        },
    )?;
    assert_eq!(r_all.total_matched, r_kernel.total_matched);
    assert_eq!(r_all.total_events, r_kernel.total_events);
    Ok(())
}
