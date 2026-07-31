//! Opt-in ncu-version drift detector.
//!
//! Default CI is NCU-free (it replays committed sidecars), so `ncu_report`
//! API drift and enum renumbering would otherwise surface only in the
//! field. This test closes that gap: when run on a box with Nsight Compute
//! installed (gated behind `VELOQ_NCU_LIVE=1`), it runs the bundled export
//! helper LIVE against the one committed real report and compares the fresh
//! sidecar to the committed one. It catches:
//!
//! - **API-method drift** — a renamed/removed `ncu_report` method makes the
//!   helper exit non-zero (surfaced here as a failure with the helper's
//!   stderr).
//! - **Enum-name drift** — a renamed enum container empties the helper's
//!   reverse map, which stamps a `classification: "degraded"` marker
//!   (asserted absent below).
//! - **Enum renumbering** — a silent-corruption hazard: the per-metric
//!   `*_code` provenance must still agree with the
//!   committed fixture, and the additivity classification must be unchanged.
//!
//! Note on what is NOT compared: in the committed fixture the
//! `metric_subtype` / `rollup` *names* are placeholders
//! (`"unknown"`) for codes whose live names weren't reproducible offline.
//! We therefore compare the classify-relevant invariants — `metric_type`
//! name, all `*_code` values, and `is_additive_native` — not the advisory
//! subtype/rollup name strings.
//!
//! Run: `VELOQ_NCU_LIVE=1 cargo test --release -p veloq-ncu --test ncu_live_drift`

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, anyhow, bail};
use veloq_ncu::native::{NativeSidecar, cache};
use veloq_ncu::source_metrics::additivity::is_additive_native;

fn report_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vector_add_basic.ncu-rep")
}

fn helper_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/ncu_export.py")
}

/// Run the bundled helper live against `report`, returning the parsed
/// sidecar. Mirrors the cache module's discovery + interpreter handling
/// but always re-exports (the cache fast-path would serve the committed
/// sidecar without touching `ncu_report`).
fn run_helper_live(report: &Path) -> Result<NativeSidecar> {
    let pythonpath = cache::locate_ncu_report().context("resolve ncu_report import path")?;
    let python = std::env::var("VELOQ_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let mut command = Command::new(&python);
    command.arg(helper_path()).arg(report);
    if let Some(path) = pythonpath {
        command.env("VELOQ_NCU_REPORT_DIR", path);
    }
    let out = command.output().context("spawn export helper")?;
    if !out.status.success() {
        bail!(
            "export helper failed ({}):\n{}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    serde_json::from_slice(&out.stdout).context("helper stdout is a valid ncu-native sidecar")
}

#[test]
fn live_helper_sidecar_matches_committed() -> Result<()> {
    if std::env::var_os("VELOQ_NCU_LIVE").is_none() {
        eprintln!(
            "skipping: set VELOQ_NCU_LIVE=1 on a box with Nsight Compute to run the drift check"
        );
        return Ok(());
    }

    let report = report_path();
    let committed = cache::read_gz_sidecar(&cache::path_for(&report))
        .context("read committed vector_add_basic sidecar")?;
    let live = run_helper_live(&report)?;

    // A structural enum-API change collapses the reverse map -> degraded.
    if live.classification.as_deref() == Some("degraded") {
        bail!("live helper emitted a degraded sidecar: ncu_report metric enum container changed");
    }
    assert_eq!(
        live.launches.len(),
        committed.launches.len(),
        "launch count drifted"
    );

    for (li, (cl, ll)) in committed
        .launches
        .iter()
        .zip(live.launches.iter())
        .enumerate()
    {
        let live_by_name: HashMap<&str, _> =
            ll.metrics.iter().map(|m| (m.name.as_str(), m)).collect();
        assert_eq!(
            cl.metrics.len(),
            ll.metrics.len(),
            "launch {li}: metric count drifted"
        );
        for cm in &cl.metrics {
            let lm = live_by_name.get(cm.name.as_str()).ok_or_else(|| {
                anyhow!("launch {li}: metric {} vanished from live export", cm.name)
            })?;

            // metric_type name is reproducible offline -> assert equal.
            assert_eq!(
                cm.metric_type, lm.metric_type,
                "launch {li}: metric_type name drifted for {} (regenerate the committed fixture if this is an intended ncu upgrade)",
                cm.name
            );
            // Raw enum codes: the renumber detector.
            assert_eq!(
                (cm.metric_type_code, cm.metric_subtype_code, cm.rollup_code),
                (lm.metric_type_code, lm.metric_subtype_code, lm.rollup_code),
                "launch {li}: enum *code* drifted for {} — ncu renumbered an enum; regenerate the committed fixture and verify the additivity classifier still matches names",
                cm.name
            );
            // The load-bearing invariant: additivity classification unchanged.
            assert_eq!(
                is_additive_native(cm),
                is_additive_native(lm),
                "launch {li}: additivity classification drifted for {}",
                cm.name
            );
        }
    }
    Ok(())
}

#[test]
fn live_helper_compressed_sidecar_matches_plain() -> Result<()> {
    if std::env::var_os("VELOQ_NCU_LIVE").is_none() {
        eprintln!(
            "skipping: set VELOQ_NCU_LIVE=1 on a box with Nsight Compute to run the drift check"
        );
        return Ok(());
    }

    let report = report_path();
    let plain = run_helper_live(&report)?;
    let raw = fs::read(&report).context("read plain NCU report")?;
    let compressed =
        zstd::stream::encode_all(raw.as_slice(), 0).context("compress NCU report fixture")?;
    let compressed_path = std::env::temp_dir().join(format!(
        "veloq-ncu-live-{}-compressed.ncu-repz",
        std::process::id()
    ));
    fs::write(&compressed_path, compressed).context("write compressed NCU report fixture")?;
    let loaded = run_helper_live(&compressed_path);
    let _ = fs::remove_file(&compressed_path);
    let loaded = loaded?;

    assert_eq!(
        serde_json::to_value(loaded)?,
        serde_json::to_value(plain)?,
        "compressed and plain reports must produce the same native sidecar"
    );
    Ok(())
}
