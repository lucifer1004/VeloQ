//! NVTX style group key:
//! - `stats --type nvtx` splits same-name PushPop / StartEnd ranges
//!   into distinct rows via the derived `nvtx_style` label
//! - unknown raw `eventType` ints fold into one `"unknown"` bucket
//!   rather than spawning a row-per-int
//! - non-NVTX rows surface `event_type=None` and `nvtx_style=None`
//!   (GPU group counts must stay unchanged)

mod fixture;

use anyhow::{Result, anyhow};
use veloq_nsys_query::stats::StatsRequest;
use veloq_nsys_query::{EventKind, KindFilter};

#[test]
fn stats_nvtx_splits_pushpop_and_startend_with_same_name() -> Result<()> {
    let trace = fixture::nvtx_styles()?;
    let r = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Nvtx]),
            ..Default::default()
        },
    )?;

    // Three buckets: `iteration|push_pop` (eventType ∈ {59,70}),
    // `iteration|start_end` (eventType ∈ {60,71}), `weird|unknown`
    // (eventType 99). Five raw NVTX rows in the fixture.
    assert_eq!(r.total_matched, 3);
    assert_eq!(r.total_events, 5);

    let by_style: std::collections::HashMap<(String, &'static str), &_> = r
        .rows
        .iter()
        .filter_map(|row| {
            let name = row.name.clone()?;
            let style = row.nvtx_style?;
            Some(((name, style), row))
        })
        .collect();

    let pp = by_style
        .get(&("iteration".into(), "push_pop"))
        .ok_or_else(|| anyhow!("missing push_pop bucket"))?;
    assert_eq!(pp.count, 2, "push_pop fold contains eventType ∈ {{59, 70}}");
    // event_type surfaces the MIN raw value seen in the bucket.
    assert_eq!(pp.event_type, Some(59));
    assert!(
        pp.key.contains("style:push_pop"),
        "composite key must carry style suffix; got {}",
        pp.key
    );

    let se = by_style
        .get(&("iteration".into(), "start_end"))
        .ok_or_else(|| anyhow!("missing start_end bucket"))?;
    assert_eq!(se.count, 2);
    assert_eq!(se.event_type, Some(60));
    assert!(se.key.contains("style:start_end"), "got {}", se.key);

    let unk = by_style
        .get(&("weird".into(), "unknown"))
        .ok_or_else(|| anyhow!("missing unknown bucket"))?;
    assert_eq!(unk.count, 1);
    assert_eq!(unk.event_type, Some(99));
    assert!(unk.key.contains("style:unknown"), "got {}", unk.key);

    Ok(())
}

#[test]
fn stats_non_nvtx_rows_carry_no_style_or_event_type() -> Result<()> {
    // The minimal_gpu fixture has no NVTX_EVENTS table; kernel rows
    // must serialise `nvtx_style=None` and `event_type=None`: adding
    // the always-on group key must not change GPU rows.
    let trace = fixture::minimal_gpu()?;
    let r = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            ..Default::default()
        },
    )?;
    assert!(
        !r.rows.is_empty(),
        "fixture has kernels; stats must return them"
    );
    for row in &r.rows {
        assert!(
            row.nvtx_style.is_none(),
            "kernel rows must not surface nvtx_style; got {:?}",
            row.nvtx_style
        );
        assert!(
            row.event_type.is_none(),
            "kernel rows must not surface event_type; got {:?}",
            row.event_type
        );
        assert!(
            !row.key.contains("style:"),
            "kernel key must not carry style suffix; got {}",
            row.key
        );
    }
    Ok(())
}

#[test]
fn stats_unknown_event_type_folds_into_one_bucket() -> Result<()> {
    // The fixture has one "weird" range at eventType=99. If we ever
    // regressed to GROUP BY raw event_type (instead of the derived
    // nvtx_style label), this would spawn a row per int. Pinning the
    // single-bucket behaviour here catches that.
    let trace = fixture::nvtx_styles()?;
    let r = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Nvtx]),
            ..Default::default()
        },
    )?;
    let unknown_rows: Vec<_> = r
        .rows
        .iter()
        .filter(|row| row.nvtx_style == Some("unknown"))
        .collect();
    assert_eq!(
        unknown_rows.len(),
        1,
        "all unknown eventType ints must fold into one `unknown` bucket"
    );
    Ok(())
}
