//! Regression coverage for process-private CUDA namespaces.
//!
//! Both ranks reuse the exact local `(device, context, stream,
//! correlationId)` tuple. Every query surface must retain the native
//! process as an identity axis.

mod fixture;

use anyhow::{Context, Result};
use std::collections::{BTreeMap, BTreeSet};
use veloq_nsys_data::Trace;
use veloq_nsys_data::scope::{ResolveError, ScopeRequest, resolve_scope};
use veloq_nsys_data::trace_map;
use veloq_nsys_query::concurrency::ConcurrencyRequest;
use veloq_nsys_query::graph_replays::GraphReplaysRequest;
use veloq_nsys_query::search::SearchRequest;
use veloq_nsys_query::slices::{SlicesAggregateGroupBy, SlicesRequest, SlicesRow, SlicesView};
use veloq_nsys_query::{EventKind, KindFilter, RowId};

#[test]
fn graph_replays_do_not_merge_process_private_device_zero() -> Result<()> {
    let trace = fixture::process_private_cuda_identity_collision()?;
    let response =
        veloq_nsys_query::graph_replays::run(trace.path(), GraphReplaysRequest::default())?;

    assert_eq!(response.rows.len(), 2);
    assert_eq!(response.total_matched, 2);
    assert_eq!(
        response
            .rows
            .iter()
            .map(|row| row.process_id)
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([1001, 2002])
    );
    for row in &response.rows {
        assert_eq!(row.device_id, 0);
        assert_eq!(row.context_id, 1);
        assert_eq!(row.correlation_id, 42);
        assert_eq!(row.wall_ns, 10_000_000);
        assert!(
            row.key
                .contains(&format!("p{:x}-d0-c1-r2a", row.process_id))
        );
    }
    let [first, second] = response.rows.as_slice() else {
        anyhow::bail!("expected exactly two graph replay rows");
    };
    assert_ne!(first.synthetic_id, second.synthetic_id);
    Ok(())
}

#[test]
fn resident_graph_replays_preserve_process_private_device_zero() -> Result<()> {
    let fixture = fixture::process_private_cuda_identity_collision()?;
    let one_shot =
        veloq_nsys_query::graph_replays::run(fixture.path(), GraphReplaysRequest::default())?;
    let trace = Trace::open(fixture.path())?;
    assert!(veloq_nsys_query::graph_replays::ensure_resident_index(
        &trace
    )?);
    let resident =
        veloq_nsys_query::graph_replays::run_with_trace(&trace, GraphReplaysRequest::default())?;
    assert_eq!(
        serde_json::to_vec(&resident)?,
        serde_json::to_vec(&one_shot)?
    );
    assert_eq!(
        resident
            .rows
            .iter()
            .map(|row| (row.process_id, row.device_id))
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([(1001, 0), (2002, 0)])
    );
    Ok(())
}

#[test]
fn correlate_keeps_each_rank_chain_private() -> Result<()> {
    let trace = fixture::process_private_cuda_identity_collision()?;
    let response = veloq_nsys_query::correlate::run(
        trace.path(),
        &[
            RowId::new(EventKind::Kernel, 1),
            RowId::new(EventKind::Kernel, 2),
        ],
    )?;

    for (result, expected_pid) in response.rows.iter().zip([1001, 2002]) {
        assert_eq!(result.process_id, Some(expected_pid));
        assert_eq!(result.auxiliary.cpu_events.len(), 1);
        assert_eq!(result.auxiliary.gpu_events.len(), 1);
        assert!(
            result
                .events
                .iter()
                .all(|event| event.base().process_id == Some(expected_pid))
        );
    }
    Ok(())
}

#[test]
fn reverse_nvtx_lookup_uses_process_aware_correlation_identity() -> Result<()> {
    let trace = fixture::process_private_cuda_identity_collision()?;
    let response = veloq_nsys_query::search::run(
        trace.path(),
        SearchRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            with_nvtx: true,
            limit: 10,
            ..Default::default()
        },
    )?;

    let by_pid = response
        .rows
        .iter()
        .map(|event| {
            let base = event.base();
            Ok((
                base.process_id.context("kernel must carry process_id")?,
                base.nvtx_context
                    .as_ref()
                    .context("kernel must carry nvtx_context")?
                    .name
                    .clone(),
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    assert_eq!(by_pid.get(&1001).map(String::as_str), Some("rank0_step"));
    assert_eq!(by_pid.get(&2002).map(String::as_str), Some("rank1_step"));
    Ok(())
}

#[test]
fn per_device_aggregation_partitions_by_process() -> Result<()> {
    let trace = fixture::process_private_cuda_identity_collision()?;
    let response = veloq_nsys_query::concurrency::run(trace.path(), ConcurrencyRequest::default())?;

    assert_eq!(response.rows.len(), 2);
    assert_eq!(
        response
            .rows
            .iter()
            .map(|row| row.key.as_str())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["concurrency|pid:1001|dev:0", "concurrency|pid:2002|dev:0",])
    );
    assert!(
        response
            .rows
            .iter()
            .all(|row| row.union_busy_ns == 10_000_000)
    );
    Ok(())
}

#[test]
fn same_named_slice_aggregates_remain_process_scoped() -> Result<()> {
    let trace = fixture::process_private_cuda_identity_collision()?;
    let response = veloq_nsys_query::slices::run(
        trace.path(),
        SlicesRequest {
            name: Some("shared_step".to_string()),
            view: SlicesView::Aggregate,
            group_by: SlicesAggregateGroupBy::Name,
            ..Default::default()
        },
    )?;

    let rows = response
        .rows
        .iter()
        .map(|row| match row {
            SlicesRow::Aggregate(row) => Ok((row.process_id, row.key.as_str())),
            SlicesRow::Instance(_) => anyhow::bail!("expected aggregate slice row"),
        })
        .collect::<Result<BTreeSet<_>>>()?;
    assert_eq!(
        rows,
        BTreeSet::from([
            (1001, "scope|pid:1001|shared_step"),
            (2002, "scope|pid:2002|shared_step"),
        ])
    );
    Ok(())
}

#[test]
fn device_zero_requires_process_when_ordinals_collide() -> Result<()> {
    let fixture = fixture::process_private_cuda_identity_collision()?;
    let trace = Trace::open(fixture.path())?;

    let ambiguous = resolve_scope(
        &trace,
        ScopeRequest {
            device: Some(0),
            ..Default::default()
        },
    );
    assert!(matches!(ambiguous, Err(ResolveError::Ambiguous(_))));

    let resolved = resolve_scope(
        &trace,
        ScopeRequest {
            process: Some(1001),
            device: Some(0),
            ..Default::default()
        },
    )?;
    assert_eq!(resolved.applied.native_pid, Some(1001));
    assert_eq!(resolved.applied.device, Some(0));
    Ok(())
}

#[test]
fn trace_map_preserves_duplicate_logical_zero_without_inventing_physical_ids() -> Result<()> {
    let fixture = fixture::process_private_cuda_identity_collision()?;
    let trace = Trace::open(fixture.path())?;
    let map = trace_map::build(&trace, trace_map::NVTX_TOP_PATHS_DEFAULT)?;

    assert!(map.devices.physical.is_none());
    assert_eq!(
        map.devices
            .logical_scopes
            .iter()
            .map(|scope| (scope.process_id, scope.device_id))
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([(1001, 0), (2002, 0)])
    );
    Ok(())
}
