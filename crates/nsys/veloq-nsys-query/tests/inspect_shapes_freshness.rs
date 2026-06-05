//! Asserts `references/inspect-shapes.md` is in sync with the
//! projected `EventDetails` schema. When this fails, run
//!
//!   cargo run --release -p veloq-nsys-query --example gen_inspect_shapes -- write
//!
//! and commit the regenerated file. The error message in the
//! `panic!` below repeats this hint.

use anyhow::Result;
use std::path::PathBuf;

#[test]
fn inspect_shapes_md_matches_projection() -> Result<()> {
    let expected = veloq_nsys_query::docgen::inspect_shapes_body();
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join(".claude/skills/nsys-profile-analysis/references/inspect-shapes.md");
    let on_disk = std::fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("can't read {}: {e}", path.display()))?;
    if on_disk != expected {
        // Find the first divergence so the diff isn't a wall of text.
        let div = on_disk
            .lines()
            .zip(expected.lines())
            .enumerate()
            .find(|(_, (a, b))| a != b)
            .map(|(i, (a, b))| format!("line {i}: on-disk=`{a}` projected=`{b}`"))
            .unwrap_or_else(|| {
                format!(
                    "lengths differ: on-disk={}b projected={}b",
                    on_disk.len(),
                    expected.len()
                )
            });
        anyhow::bail!(
            "inspect-shapes.md is out of sync with EventDetails projection.\n\
             First divergence: {div}\n\
             Regenerate: cargo run --release -p veloq-nsys-query --example gen_inspect_shapes -- write"
        );
    }
    Ok(())
}
