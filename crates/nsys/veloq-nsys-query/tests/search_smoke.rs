//! Integration tests for `search`.
//!
//! Focus right now is NVTX depth surfacing — the nesting computation
//! lives in `veloq-nsys-data::nvtx_nesting`, but the wiring through
//! `search` proves agents can ask "give me only root spans" via
//! `--type nvtx` and post-filter `depth == 0`.

mod fixture;

use anyhow::Result;
use std::collections::HashMap;
use veloq_nsys_query::search::SearchRequest;
use veloq_nsys_query::{EventKind, EventRef, KindFilter};

#[test]
fn nvtx_search_surfaces_nesting_depth() -> Result<()> {
    let trace = fixture::nvtx_nested()?;
    let req = SearchRequest {
        kinds: KindFilter::Only(vec![EventKind::Nvtx]),
        limit: 50,
        ..Default::default()
    };
    let r = veloq_nsys_query::search::run(trace.path(), req)?;

    // Index hits by name for readable assertions. Multiple ranges share
    // a (gtid, domain), so depths must reflect the nested layout
    // documented on `fixture::nvtx_nested`.
    let by_name: HashMap<&str, &EventRef> =
        r.rows.iter().map(|h| (h.base().name.as_str(), h)).collect();

    assert_eq!(
        by_name.get("outer").and_then(|h| h.base().depth),
        Some(0),
        "outer range is the root span"
    );
    assert_eq!(
        by_name.get("inner").and_then(|h| h.base().depth),
        Some(1),
        "inner sits one level under outer"
    );
    assert_eq!(
        by_name.get("leaf").and_then(|h| h.base().depth),
        Some(2),
        "leaf sits inside both outer and inner"
    );
    // `marker` is an instant marker (end IS NULL). The search SQL
    // filters those out via `WHERE t."end" IS NOT NULL` in the NVTX
    // arm, so the marker shouldn't appear at all — proves the
    // existing filter still holds after the depth wiring.
    assert!(
        !by_name.contains_key("marker"),
        "instant markers stay filtered out of search NVTX results"
    );
    assert_eq!(
        by_name.get("sibling").and_then(|h| h.base().depth),
        Some(0),
        "sibling after outer is also a root span"
    );

    Ok(())
}

/// Skipping the nesting computation when no NVTX kind is requested
/// keeps cold-path latency unchanged. The behavioural assertion is
/// that GPU-only searches still return hits *without* a `depth` field
/// — populated only for `--type nvtx`.
#[test]
fn gpu_search_omits_depth_field() -> Result<()> {
    let trace = fixture::minimal_gpu()?;
    let req = SearchRequest {
        kinds: KindFilter::Only(vec![EventKind::Kernel]),
        limit: 20,
        ..Default::default()
    };
    let r = veloq_nsys_query::search::run(trace.path(), req)?;
    assert!(!r.rows.is_empty());
    for hit in &r.rows {
        assert!(
            hit.base().depth.is_none(),
            "kernel hits must not carry NVTX depth"
        );
    }
    Ok(())
}

/// `--name-regex` drives the StringId pre-filter path (the `name_match_ids`
/// CTE + the per-kind `demangledName`/`shortName` membership predicate).
/// It must return exactly the rows whose resolved name matches — the
/// pre-filter is only a pruning superset, the regex stays authoritative.
/// `minimal_gpu` has 2×`fast_kernel` + 2×`slow_kernel`.
#[test]
fn name_regex_prefilter_matches_resolved_names() -> Result<()> {
    let trace = fixture::minimal_gpu()?;
    let run = |re: &str| -> Result<_> {
        veloq_nsys_query::search::run(
            trace.path(),
            SearchRequest {
                kinds: KindFilter::Only(vec![EventKind::Kernel]),
                name_regex: Some(re.to_string()),
                limit: 20,
                ..Default::default()
            },
        )
    };

    let fast = run("^fast")?;
    assert_eq!(fast.count, 2, "two fast_kernel rows match");
    assert_eq!(fast.total_matched, 2);
    for hit in &fast.rows {
        assert_eq!(hit.base().name, "fast_kernel");
    }

    let all = run("kernel")?;
    assert_eq!(all.count, 4, "all four kernels contain 'kernel'");
    assert_eq!(all.total_matched, 4);

    let none = run("no_such_kernel_xyz")?;
    assert_eq!(none.count, 0);
    assert_eq!(none.total_matched, 0);
    assert!(none.rows.is_empty());

    Ok(())
}
