//! Structural assertion: every NCU response's primary `rows[]` element
//! carries a `key` field. Canonical list contract — keeps the
//! cross-trace jq join recipes working across the full surface.
//!
//! Mirrors `veloq-nsys-query/tests/wire_format_smoke.rs`; the test
//! walks `schema_for!(T)` rather than asserting against rendered
//! strings so adding a new response type that forgot `key` still
//! fails the test without a hand-edited allow-list.

use veloq_ncu::disasm::DisasmResponse;
use veloq_ncu::inspect::InspectResponse as NcuInspectResponse;
use veloq_ncu::launches::LaunchesResponse;
use veloq_ncu::lists::{GraphsResponse, RangesResponse, SourcesResponse};
use veloq_ncu::metrics::MetricsResponse;
use veloq_ncu::native::NativeSummaryResponse;
use veloq_ncu::source_metrics::SourceMetricsResponse;
use veloq_ncu::warp_stalls::WarpStallsResponse;

fn check_rows_have_key<T: schemars::JsonSchema>() -> anyhow::Result<()> {
    let type_name = std::any::type_name::<T>();
    let root = serde_json::Value::from(schemars::schema_for!(T));
    let defs = root
        .get("$defs")
        .and_then(serde_json::Value::as_object)
        .cloned()
        .unwrap_or_default();
    let resolve = |v: &serde_json::Value| -> serde_json::Value {
        let Some(refstr) = v.get("$ref").and_then(serde_json::Value::as_str) else {
            return v.clone();
        };
        let name = refstr.rsplit('/').next().unwrap_or(refstr);
        defs.get(name).cloned().unwrap_or_else(|| v.clone())
    };

    let rows = root
        .get("properties")
        .and_then(|p| p.get("rows"))
        .ok_or_else(|| anyhow::anyhow!("{type_name}: schema missing rows property"))?;
    let items = rows
        .get("items")
        .ok_or_else(|| anyhow::anyhow!("{type_name}: rows.items missing"))?;
    let item = resolve(items);

    let variants = item
        .get("oneOf")
        .or_else(|| item.get("anyOf"))
        .and_then(serde_json::Value::as_array);
    if let Some(variants) = variants {
        for (i, v) in variants.iter().enumerate() {
            let resolved = resolve(v);
            if resolved
                .get("properties")
                .and_then(|p| p.get("key"))
                .is_none()
            {
                anyhow::bail!("{type_name}: rows[] variant #{i} lacks a `key` field");
            }
        }
        return Ok(());
    }

    if item.get("properties").and_then(|p| p.get("key")).is_none() {
        anyhow::bail!("{type_name}: rows[] item lacks a `key` field");
    }
    Ok(())
}

#[test]
fn every_primary_rows_item_carries_key() -> anyhow::Result<()> {
    check_rows_have_key::<NativeSummaryResponse>()?;
    check_rows_have_key::<LaunchesResponse>()?;
    check_rows_have_key::<NcuInspectResponse>()?;
    check_rows_have_key::<DisasmResponse>()?;
    check_rows_have_key::<RangesResponse>()?;
    check_rows_have_key::<GraphsResponse>()?;
    check_rows_have_key::<SourcesResponse>()?;
    check_rows_have_key::<MetricsResponse>()?;
    check_rows_have_key::<SourceMetricsResponse>()?;
    check_rows_have_key::<WarpStallsResponse>()?;
    Ok(())
}
