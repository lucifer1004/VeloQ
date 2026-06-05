//! NVTX path-aware CLI semantics.

mod fixture;

use anyhow::{Result, anyhow, bail};
use veloq_nsys_query::inspect::{self, EventDetails};
use veloq_nsys_query::slices::{
    SliceAggregate, SlicesAggregateGroupBy, SlicesRequest, SlicesRow, SlicesView,
};
use veloq_nsys_query::stats::{GroupBy, StatsRequest};
use veloq_nsys_query::{EventKind, KindFilter, RowId};

fn path_axis() -> GroupBy {
    GroupBy {
        nvtx_path: true,
        ..Default::default()
    }
}

#[test]
fn stats_nvtx_path_keeps_same_leaf_ranges_distinct() -> Result<()> {
    let trace = fixture::same_leaf_nested_nvtx_paths()?;
    let response = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            group_by: path_axis(),
            ..Default::default()
        },
    )?;

    assert_eq!(response.total_matched, 2);
    let paths = response
        .rows
        .iter()
        .map(|r| r.nvtx_path.as_deref().unwrap_or(""))
        .collect::<std::collections::HashSet<_>>();
    assert!(paths.contains("phase_a/work"), "got {paths:?}");
    assert!(paths.contains("phase_b/work"), "got {paths:?}");
    for row in &response.rows {
        assert!(row.nvtx_parent_name.is_none());
        assert!(
            row.nvtx_path_key
                .as_deref()
                .is_some_and(|key| key.starts_with("nvtx-path:phase_")),
            "bad path key: {:?}",
            row.nvtx_path_key
        );
        assert!(row.key.contains("nvtx-path:"));
    }
    Ok(())
}

/// Two ranges with the
/// SAME leaf name + SAME parent chain but DIFFERENT `(pid, domainId)`
/// domains must produce two distinct rows. The keys differ only by the
/// `domain:<pid>:<id>` component, and each row carries its resolved
/// `domain_id` / `domain_pid` (+ name, since both domains are
/// registered here).
#[test]
fn stats_nvtx_path_keeps_same_path_distinct_domains_distinct() -> Result<()> {
    let trace = fixture::same_leaf_same_parent_distinct_domains()?;
    let response = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            group_by: path_axis(),
            ..Default::default()
        },
    )?;

    // Both rows share the identical `phase/work` path...
    assert_eq!(response.total_matched, 2, "rows: {:?}", response.rows);
    for row in &response.rows {
        assert_eq!(row.nvtx_path.as_deref(), Some("phase/work"));
        // ...and an enclosing range, so each carries a domain identity.
        assert!(
            row.domain_id.is_some() && row.domain_pid.is_some(),
            "expected domain identity on row {:?}",
            row.key
        );
        assert!(
            row.key.contains("domain:"),
            "expected domain key component in {:?}",
            row.key
        );
    }

    // The two rows are kept distinct purely by their domain component.
    let keys = response
        .rows
        .iter()
        .map(|r| r.key.clone())
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(keys.len(), 2, "domains must not collapse: {keys:?}");

    // Each (pid, domain_id) pair resolves to its registered name.
    let mut by_domain: std::collections::HashMap<(i64, i64), Option<String>> =
        std::collections::HashMap::new();
    for row in &response.rows {
        let pid = row
            .domain_pid
            .ok_or_else(|| anyhow!("missing domain_pid"))?;
        let did = row.domain_id.ok_or_else(|| anyhow!("missing domain_id"))?;
        by_domain.insert((pid, did), row.domain_name.clone());
    }
    assert_eq!(
        by_domain.get(&(12345, 1)),
        Some(&Some("alpha".to_string())),
        "by_domain: {by_domain:?}"
    );
    assert_eq!(
        by_domain.get(&(67890, 2)),
        Some(&Some("beta".to_string())),
        "by_domain: {by_domain:?}"
    );
    Ok(())
}

/// A no-NVTX sentinel row
/// (a kernel outside any range) MUST NOT carry a domain-identity
/// component — no `domain:` key part and `domain_id`/`domain_pid`/
/// `domain_name` all `None` — while the enclosed kernel's row does.
#[test]
fn stats_nvtx_path_sentinel_row_has_no_domain() -> Result<()> {
    let trace = fixture::nvtx_path_enclosed_and_sentinel()?;
    let response = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            group_by: path_axis(),
            ..Default::default()
        },
    )?;

    assert_eq!(response.total_matched, 2, "rows: {:?}", response.rows);
    let enclosed = response
        .rows
        .iter()
        .find(|r| r.nvtx_path.as_deref() == Some("work"))
        .ok_or_else(|| anyhow!("missing enclosed 'work' row: {:?}", response.rows))?;
    assert_eq!(enclosed.domain_id, Some(1));
    assert_eq!(enclosed.domain_pid, Some(4242));
    assert_eq!(enclosed.domain_name.as_deref(), Some("alpha"));
    assert!(
        enclosed.key.contains("domain:4242:1"),
        "key: {}",
        enclosed.key
    );

    let sentinel = response
        .rows
        .iter()
        .find(|r| r.nvtx_path.as_deref() != Some("work"))
        .ok_or_else(|| anyhow!("missing sentinel row: {:?}", response.rows))?;
    assert!(
        sentinel.domain_id.is_none(),
        "sentinel domain_id: {:?}",
        sentinel
    );
    assert!(sentinel.domain_pid.is_none());
    assert!(sentinel.domain_name.is_none());
    assert!(
        !sentinel.key.contains("domain:"),
        "sentinel key must carry no domain component: {}",
        sentinel.key
    );
    Ok(())
}

fn aggregate_rows(resp: &veloq_nsys_query::slices::SlicesResponse) -> Vec<&SliceAggregate> {
    resp.rows
        .iter()
        .filter_map(|row| match row {
            SlicesRow::Aggregate(r) => Some(r),
            SlicesRow::Instance(_) => None,
        })
        .collect()
}

#[test]
fn slices_aggregate_path_mode_does_not_collapse_same_leaf_ranges() -> Result<()> {
    let trace = fixture::same_leaf_nested_nvtx_paths()?;

    let by_name = veloq_nsys_query::slices::run(
        trace.path(),
        SlicesRequest {
            name: Some("work".to_string()),
            view: SlicesView::Aggregate,
            limit: 100,
            ..Default::default()
        },
    )?;
    assert_eq!(by_name.group_by, Some("name"));
    assert_eq!(by_name.total_matched, 1);
    assert_eq!(
        aggregate_rows(&by_name).first().map(|r| r.instances),
        Some(2)
    );

    let by_path = veloq_nsys_query::slices::run(
        trace.path(),
        SlicesRequest {
            name: Some("work".to_string()),
            view: SlicesView::Aggregate,
            group_by: SlicesAggregateGroupBy::Path,
            limit: 100,
            ..Default::default()
        },
    )?;
    assert_eq!(by_path.group_by, Some("path"));
    assert_eq!(by_path.total_matched, 2);
    let paths = aggregate_rows(&by_path)
        .into_iter()
        .map(|r| r.path.as_deref().unwrap_or(""))
        .collect::<std::collections::HashSet<_>>();
    assert!(paths.contains("phase_a/work"), "got {paths:?}");
    assert!(paths.contains("phase_b/work"), "got {paths:?}");
    Ok(())
}

#[test]
fn inspect_nvtx_exposes_path_and_parent() -> Result<()> {
    let trace = fixture::same_leaf_nested_nvtx_paths()?;
    let id = RowId::new(EventKind::Nvtx, 2);
    let response = inspect::run(trace.path(), &[id])?;
    let event = response
        .rows
        .first()
        .ok_or_else(|| anyhow!("expected one inspect row"))?;
    let nvtx = match event {
        EventDetails::Nvtx(n) => n,
        other => bail!("expected Nvtx, got {other:?}"),
    };

    assert_eq!(nvtx.name, "work");
    assert_eq!(nvtx.depth, Some(1));
    assert_eq!(nvtx.path.as_deref(), Some("phase_a/work"));
    assert_eq!(nvtx.parent_row_id, Some(RowId::new(EventKind::Nvtx, 1)));
    assert_eq!(nvtx.parent_name.as_deref(), Some("phase_a"));
    Ok(())
}
