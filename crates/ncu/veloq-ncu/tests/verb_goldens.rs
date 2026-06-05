//! Frozen-golden regression oracle for the `ncu_report`-native
//! verbs. Each verb runs in-process against the
//! committed `source_metric_basic` fixture in **committed-sidecar mode**:
//! the proprietary `.ncu-rep` is NOT committed (it embedded a hostname +
//! internal NIC IPs) — only the leak-free
//! `<report>.veloq/` artifacts are: the native sidecar
//! (`ncu-native.json.gz`), the per-cubin disasm cache
//! (`<sha>.correlated.json`), and the extracted cubin (`<sha>.cubin`,
//! clean compiled code). `build_or_load` serves the sidecar with the
//! report absent and disasm sources the committed cubin, so this is
//! **NCU-free and nvdisasm-free**: goldens are diffed without invoking
//! any external tool.
//!
//! Goldens live in `tests/goldens/ncu-<verb>.json`. Regenerate after
//! an intentional wire change with `UPDATE_GOLDENS=1 cargo test
//! --release -p veloq-ncu --test verb_goldens` and review the diff.
//!
//! Scope: launches / sources / metrics (narrow glob) / disasm — the
//! verbs with small, reviewable output. `inspect` (full metric dump)
//! and `summary` (native totals) are covered by targeted
//! assertions in `launches_inspect_smoke.rs` and the veloq
//! `ncu_summary_envelope_has_single_data_layer` test respectively;
//! a full golden of inspect's ~4.4k-entry metric arrays would not be
//! reviewable.

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::path::{Path, PathBuf};

use veloq_ncu::disasm;
use veloq_ncu::launches::{self, LaunchesRequest};
use veloq_ncu::lists;
use veloq_ncu::metrics::{self, MetricsRequest};
use veloq_ncu::source_metrics::{self, Axis, SourceMetricsRequest};
use veloq_ncu::warp_stalls::{self, WarpStallsRequest};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/source_metric_basic.ncu-rep")
}

fn goldens_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/goldens")
}

/// Normalize volatile fields so committed goldens are stable and
/// carry no local paths: `meta_cache_path` is an absolute path under
/// `CARGO_MANIFEST_DIR`, so keep only the report-relative tail (the
/// part after `fixtures/`). Guards the AGENTS.md no-local-leak rule.
fn normalize(v: &mut Value) {
    match v {
        Value::Object(map) => {
            for (k, val) in map.iter_mut() {
                if k == "meta_cache_path"
                    && let Value::String(s) = val
                    && let Some(idx) = s.find("fixtures/")
                    && let Some(tail) = s.get(idx + "fixtures/".len()..)
                {
                    let tail = tail.to_string();
                    *s = tail;
                } else {
                    normalize(val);
                }
            }
        }
        Value::Array(arr) => arr.iter_mut().for_each(normalize),
        _ => {}
    }
}

/// Serialize `response`, normalize, and compare to the committed
/// golden. `UPDATE_GOLDENS=1` rewrites the golden instead.
fn check_golden<T: Serialize>(name: &str, response: &T) -> Result<()> {
    let mut value = serde_json::to_value(response).context("serialize response")?;
    normalize(&mut value);
    let pretty = serde_json::to_string_pretty(&value).context("pretty-print")?;
    let path = goldens_dir().join(format!("ncu-{name}.json"));

    if std::env::var("UPDATE_GOLDENS").is_ok() {
        std::fs::create_dir_all(goldens_dir()).context("create goldens dir")?;
        std::fs::write(&path, format!("{pretty}\n"))
            .with_context(|| format!("write golden {}", path.display()))?;
        return Ok(());
    }

    let expected = std::fs::read_to_string(&path).with_context(|| {
        format!(
            "missing golden {} (regenerate with UPDATE_GOLDENS=1)",
            path.display()
        )
    })?;
    anyhow::ensure!(
        expected.trim_end() == pretty,
        "golden mismatch for `{name}` ({}). Run with UPDATE_GOLDENS=1 to refresh after an intentional wire change.",
        path.display()
    );
    Ok(())
}

#[test]
fn launches_matches_golden() -> Result<()> {
    let r = launches::run(
        fixture(),
        LaunchesRequest {
            limit: 100,
            ..Default::default()
        },
    )?;
    check_golden("launches", &r)
}

#[test]
fn sources_matches_golden() -> Result<()> {
    let r = lists::sources(fixture(), 100)?;
    check_golden("sources", &r)
}

#[test]
fn metrics_narrow_glob_matches_golden() -> Result<()> {
    // Narrow glob keeps the golden small + reviewable; the row shape
    // (key / value / unit / value_type) is what the oracle locks.
    let r = metrics::run(
        fixture(),
        MetricsRequest {
            counter_glob: "launch__grid_size".to_string(),
            limit: 100,
            ..Default::default()
        },
    )?;
    check_golden("metrics", &r)
}

#[test]
fn disasm_launch1_matches_golden() -> Result<()> {
    // Serves from the committed `<sha>.correlated.json` cache — no
    // nvdisasm/cuobjdump invocation. Locks the proto-free hybrid
    // output (predicate + control_flow + source attribution).
    let r = disasm::run(fixture(), "launch:1")?;
    check_golden("disasm", &r)
}

#[test]
fn source_metrics_line_matches_golden() -> Result<()> {
    // Native source-metrics: placement-driven two-bucket split + line
    // attribution from the sidecar's `source_info`. NCU-free and
    // nvdisasm-free (no disasm pipeline). Locks the per-line rollup +
    // the reconciliation budget (rows + unattributed + out_of_cubin).
    let r = source_metrics::run(
        fixture(),
        SourceMetricsRequest {
            row_id: "launch:1".to_string(),
            counter_glob: "smsp__pcsamp_warps_issue_stalled_long_scoreboard".to_string(),
            by: Axis::Line,
            file_glob: None,
            line: None,
            sort: None,
            limit: 100,
        },
    )?;
    check_golden("source-metrics-line", &r)
}

/// Native non-KERNEL workloads. Sidecar-only
/// fixtures (`graph_basic` / `range_basic`) — the source `.ncu-rep`s
/// embed a hostname; the native sidecars are leak-free (version-only
/// session). Driven via `*_from_sidecar` + `read_gz_sidecar`, NCU-free.
fn read_sidecar(stem: &str) -> Result<veloq_ncu::native::NativeSidecar> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(format!("{stem}.ncu-native.json.gz"));
    veloq_ncu::native::cache::read_gz_sidecar(&p)
}

#[test]
fn graphs_matches_golden() -> Result<()> {
    let sc = read_sidecar("graph_basic")?;
    let r = lists::graphs_from_sidecar(&sc, "graph_basic.ncu-native.json.gz".to_string(), 100);
    check_golden("graphs", &r)
}

#[test]
fn ranges_matches_golden() -> Result<()> {
    let sc = read_sidecar("range_basic")?;
    let r = lists::ranges_from_sidecar(&sc, "range_basic.ncu-native.json.gz".to_string(), 100);
    check_golden("ranges", &r)
}

/// `warp-stalls --by line`. Sidecar-only fixture
/// (`warp_stalls_basic`, a stall-heavy pointer-chase captured with warp
/// sampling) — leak-free (no `.ncu-rep`; only `/tmp`-pathed source).
#[test]
fn warp_stalls_line_matches_golden() -> Result<()> {
    let sc = read_sidecar("warp_stalls_basic")?;
    let r = warp_stalls::run_on_sidecar(
        &sc,
        WarpStallsRequest {
            row_id: "launch:0".to_string(),
            by: warp_stalls::Axis::Line,
            file_glob: None,
            limit: 100,
        },
        "warp_stalls_basic.ncu-native.json.gz".to_string(),
    )?;
    check_golden("warp-stalls-line", &r)
}

/// Self-contained reconciliation identity: per launch,
/// `Σ by-sass row totals + out_of_cubin == total_samples ==
/// len(timed_warp_samples())`, and the by-line partition
/// `Σ line rows + unattributed + out_of_cubin == total_samples`. No
/// `ncu metrics` reference — the sample count is the oracle.
#[test]
fn warp_stalls_reconciles() -> Result<()> {
    use veloq_ncu::warp_stalls::WarpStallsRow;
    let sc = read_sidecar("warp_stalls_basic")?;
    let req = |by| WarpStallsRequest {
        row_id: "launch:0".to_string(),
        by,
        file_glob: None,
        limit: 100_000,
    };
    let cache = "warp_stalls_basic.ncu-native.json.gz".to_string();

    let sass = warp_stalls::run_on_sidecar(&sc, req(warp_stalls::Axis::Sass), cache.clone())?;
    let total = sass.auxiliary.total_samples;
    let oob = sass.auxiliary.out_of_cubin_samples;
    let unattr = sass.auxiliary.unattributed_samples;
    anyhow::ensure!(total > 0, "fixture should carry warp samples");

    let sass_sum: u64 = sass
        .rows
        .iter()
        .map(|r| match r {
            WarpStallsRow::Sass(s) => s.total_samples,
            _ => 0,
        })
        .sum();
    anyhow::ensure!(
        sass_sum + oob == total,
        "by-sass: {sass_sum} + oob {oob} != total {total}"
    );

    let line = warp_stalls::run_on_sidecar(&sc, req(warp_stalls::Axis::Line), cache.clone())?;
    let line_sum: u64 = line
        .rows
        .iter()
        .map(|r| match r {
            WarpStallsRow::Line(l) => l.total_samples,
            _ => 0,
        })
        .sum();
    anyhow::ensure!(
        line_sum + unattr + oob == total,
        "by-line: {line_sum} + unattr {unattr} + oob {oob} != total {total}"
    );

    // `--by reason` totals partition the whole stream.
    let reason = warp_stalls::run_on_sidecar(&sc, req(warp_stalls::Axis::Reason), cache)?;
    let reason_sum: u64 = reason
        .rows
        .iter()
        .map(|r| match r {
            WarpStallsRow::Reason(x) => x.total_samples,
            _ => 0,
        })
        .sum();
    anyhow::ensure!(
        reason_sum == total,
        "by-reason: {reason_sum} != total {total}"
    );
    anyhow::ensure!(
        reason.auxiliary.not_issued_samples <= total,
        "not_issued must be a subset of total"
    );
    Ok(())
}
