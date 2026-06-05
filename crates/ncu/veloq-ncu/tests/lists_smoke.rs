//! Smoke tests for the lightweight NCU list verbs.
//!
//! `sources` / `ranges` / `graphs` all read the `ncu_report` native
//! sidecar (`<file>.veloq/ncu-native.json.gz`). veloq has
//! no `blocks` or `cmdlists` verb — CMDLIST is OptiX command lists,
//! outside veloq's CUDA scope, and neither has a meaningful
//! `ncu_report` surface.
//!
//! The bundled `vector_add_basic.ncu-rep` fixture has 1 launch, a
//! zero-byte cubin, and 0 ranges / graphs — covering the non-empty
//! `sources` row and the empty-but-valid shape of the rest.

use anyhow::{Result, anyhow};
use veloq_ncu::lists;

fn fixture() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vector_add_basic.ncu-rep")
}

#[test]
fn sources_lists_one_source_with_native_shape() -> Result<()> {
    let r = lists::sources(fixture(), 100)?;
    // One synthesized source row per launch under the native model.
    assert_eq!(r.count, 1);
    assert_eq!(r.total_matched, 1);
    let row = r
        .rows
        .first()
        .ok_or_else(|| anyhow!("expected one source row"))?;
    assert_eq!(row.row_id, "source:0");
    assert_eq!(row.key, "source:0");
    // sm recovered from the launch's compute-capability metrics.
    assert_eq!(row.cuda_sm_name.as_deref(), Some("sm_100"));
    // vector_add_basic has a zero-byte cubin — no SASS captured, so
    // the disasm flag stays false (raw-binary byte counts are dropped
    // under the native model).
    assert!(!row.has_disasm);
    assert!(!r.auxiliary.meta_cache_path.is_empty());
    Ok(())
}

#[test]
fn ranges_graphs_are_empty_but_well_shaped() -> Result<()> {
    let ranges = lists::ranges(fixture(), 100)?;
    assert_eq!(ranges.count, 0);
    assert_eq!(ranges.total_matched, 0);
    assert!(ranges.rows.is_empty());

    let graphs = lists::graphs(fixture(), 100)?;
    assert_eq!(graphs.count, 0);
    assert_eq!(graphs.total_matched, 0);
    assert!(graphs.rows.is_empty());

    // auxiliary echoes the same shape regardless of emptiness.
    assert!(!ranges.auxiliary.meta_cache_path.is_empty());
    assert!(!graphs.auxiliary.meta_cache_path.is_empty());
    Ok(())
}
