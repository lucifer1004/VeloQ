//! `veloq ncu schema <target>` — strict JSON Schema for NCU response
//! payloads.
//!
//! Mirrors the NSys schema endpoint: the schema is generated from the
//! same Rust type that `serde_json` serializes on successful command
//! output, then wrapped in the shared envelope by `source.rs`.

use anyhow::Result;

use crate::disasm::DisasmResponse;
use crate::inspect::InspectResponse;
use crate::launches::LaunchesResponse;
use crate::lists::{GraphsResponse, RangesResponse, SourcesResponse};
use crate::metrics::MetricsResponse;
use crate::native::NativeSummaryResponse;
use crate::source_metrics::SourceMetricsResponse;
use crate::warp_stalls::WarpStallsResponse;

/// Response payload for `veloq ncu schema <target>`.
#[derive(serde::Serialize)]
pub struct SchemaPayload {
    pub target: String,
    pub schema: serde_json::Value,
}

/// Dispatch `target` to the matching NCU response payload type.
pub fn schema_value_for(target: &str) -> Result<serde_json::Value> {
    let value = match target {
        "summary" => serde_json::to_value(schemars::schema_for!(NativeSummaryResponse))?,
        "launches" => serde_json::to_value(schemars::schema_for!(LaunchesResponse))?,
        "inspect" => serde_json::to_value(schemars::schema_for!(InspectResponse))?,
        "metrics" => serde_json::to_value(schemars::schema_for!(MetricsResponse))?,
        "disasm" => serde_json::to_value(schemars::schema_for!(DisasmResponse))?,
        "ranges" => serde_json::to_value(schemars::schema_for!(RangesResponse))?,
        "graphs" => serde_json::to_value(schemars::schema_for!(GraphsResponse))?,
        "sources" => serde_json::to_value(schemars::schema_for!(SourcesResponse))?,
        "source-metrics" => serde_json::to_value(schemars::schema_for!(SourceMetricsResponse))?,
        "warp-stalls" => serde_json::to_value(schemars::schema_for!(WarpStallsResponse))?,
        other => anyhow::bail!(
            "unknown ncu schema target `{other}`; expected one of: \
             summary, launches, inspect, metrics, disasm, ranges, graphs, sources, source-metrics, warp-stalls"
        ),
    };
    Ok(value)
}
