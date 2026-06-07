//! `stats --by size` — byte-aggregator
//! verb under the hidden StatsBySizeResponse shape.
//!
//! - memcpy + memset rows aggregate over the `bytes` column
//! - explicit non-memop kinds reject up-front
//! - KindFilter::All narrows implicitly to memcpy + memset
//! - --sort key validity: bytes/total parse; *_bytes parse; gbps
//!   rejects; total_ns rejects
//! - per-row percentage = row.total_bytes / response.total_bytes

mod fixture;

use anyhow::{Result, anyhow};
use veloq_core::VeloqDiagnostic;
use veloq_nsys_query::stats_by_size::{StatsBySizeRequest, run};
use veloq_nsys_query::{EventKind, KindFilter, NsysQueryError};

fn assert_query_error_code(err: &NsysQueryError, expected: &str) {
    assert_eq!(err.code().as_str(), expected);
}

#[test]
fn memcpy_rows_aggregate_by_bytes() -> Result<()> {
    let trace = fixture::minimal_gpu()?;
    let r = run(
        trace.path(),
        StatsBySizeRequest {
            kinds: KindFilter::Only(vec![EventKind::Memcpy]),
            ..Default::default()
        },
    )?;
    assert!(
        r.total_bytes > 0,
        "memcpy rows should sum to a positive total_bytes"
    );
    for row in &r.rows {
        assert_eq!(row.kind, "memcpy");
        assert!(row.total_bytes > 0);
        // Percentage adds up to ~100% (within float epsilon) across
        // all returned rows of the same kind.
        assert!(row.percentage >= 0.0 && row.percentage <= 100.0 + 1e-6);
    }
    let pct_sum: f64 = r.rows.iter().map(|x| x.percentage).sum();
    assert!(
        (pct_sum - 100.0).abs() < 1e-6,
        "percentages must sum to ~100; got {pct_sum}"
    );
    Ok(())
}

#[test]
fn rejects_non_memop_kind_explicit() -> Result<()> {
    let trace = fixture::minimal_gpu()?;
    let outcome = run(
        trace.path(),
        StatsBySizeRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            ..Default::default()
        },
    );
    let err = match outcome {
        Ok(_) => return Err(anyhow!("expected reject for kernel under --by size")),
        Err(e) => e,
    };
    assert_query_error_code(&err, "nsys.query.stats-by-size-kind-not-allowed");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("byte-carrying") || msg.contains("memcpy/memset"),
        "got: {msg}"
    );
    Ok(())
}

#[test]
fn kindfilter_all_narrows_to_memops() -> Result<()> {
    let trace = fixture::minimal_gpu()?;
    // KindFilter::All resolves to memcpy + memset only under --by size.
    let r_all = run(
        trace.path(),
        StatsBySizeRequest {
            kinds: KindFilter::All,
            ..Default::default()
        },
    )?;
    // Explicit memcpy + memset must yield the same total.
    let r_explicit = run(
        trace.path(),
        StatsBySizeRequest {
            kinds: KindFilter::Only(vec![EventKind::Memcpy, EventKind::Memset]),
            ..Default::default()
        },
    )?;
    assert_eq!(r_all.total_bytes, r_explicit.total_bytes);
    assert_eq!(r_all.total_events, r_explicit.total_events);
    Ok(())
}

#[test]
fn sort_key_accepts_bytes_and_aliases() -> Result<()> {
    use veloq_core::SortSpec;
    let trace = fixture::minimal_gpu()?;
    for key in ["bytes", "total", "total_bytes", "avg_bytes", "p95_bytes"] {
        let _ = run(
            trace.path(),
            StatsBySizeRequest {
                sort: Some(SortSpec::parse(key)?),
                ..Default::default()
            },
        )?;
    }
    Ok(())
}

#[test]
fn rejects_unsupported_group_by_axes() -> Result<()> {
    // SDK callers handing in axes the byte path doesn't implement
    // must get a clear reject, not a response that looks grouped
    // but isn't.
    let trace = fixture::minimal_gpu()?;
    let unsupported: &[veloq_nsys_query::stats::GroupBy] = &[
        veloq_nsys_query::stats::GroupBy {
            graph: true,
            ..Default::default()
        },
        veloq_nsys_query::stats::GroupBy {
            graph_node: true,
            ..Default::default()
        },
        veloq_nsys_query::stats::GroupBy {
            grid_block: true,
            ..Default::default()
        },
        veloq_nsys_query::stats::GroupBy {
            nvtx_parent: true,
            ..Default::default()
        },
        veloq_nsys_query::stats::GroupBy {
            nvtx_path: true,
            ..Default::default()
        },
    ];
    for gb in unsupported {
        let outcome = run(
            trace.path(),
            StatsBySizeRequest {
                group_by: *gb,
                ..Default::default()
            },
        );
        match outcome {
            Ok(_) => {
                return Err(anyhow!(
                    "expected reject for unsupported axis on group_by {gb:?}"
                ));
            }
            Err(e) => {
                assert_query_error_code(&e, "nsys.query.stats-by-size-group-by-unsupported");
                let msg = format!("{e:#}");
                assert!(
                    msg.contains("does not yet support") || msg.contains("not yet"),
                    "got: {msg}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn sort_key_rejects_duration_keys() -> Result<()> {
    // Duration-axis keys are valid for `stats` but the byte-axis path
    // has no matching column — the request must error (parse-time or
    // run-time), never silently fall through to a working query.
    use veloq_core::SortSpec;
    let trace = fixture::minimal_gpu()?;
    for key in ["gbps", "total_ns", "p50_ns"] {
        // Parse may itself error if the key isn't in the by-size
        // SortKey set; that's an acceptable reject. Otherwise the
        // SortSpec is built and `run` must reject.
        let outcome = match SortSpec::parse(key) {
            Err(_) => continue,
            Ok(spec) => run(
                trace.path(),
                StatsBySizeRequest {
                    sort: Some(spec),
                    ..Default::default()
                },
            ),
        };
        if outcome.is_ok() {
            return Err(anyhow!(
                "expected --by size to reject duration-axis sort key `{key}` \
                 (the column doesn't exist in this mode); got Ok"
            ));
        }
    }
    Ok(())
}
