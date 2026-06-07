//! `veloq ncu schema <target>` — strict JSON Schema for NCU response
//! payloads.
//!
//! Mirrors the NSys schema endpoint: the schema is generated from the
//! same Rust type that `serde_json` serializes on successful command
//! output, then wrapped in the shared envelope by `source.rs`.

use crate::disasm::DisasmResponse;
use crate::error::{NcuSourceError, NcuSourceResult};
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
pub fn schema_value_for(target: &str) -> NcuSourceResult<serde_json::Value> {
    let schema = match target {
        "summary" => schemars::schema_for!(NativeSummaryResponse),
        "launches" => schemars::schema_for!(LaunchesResponse),
        "inspect" => schemars::schema_for!(InspectResponse),
        "metrics" => schemars::schema_for!(MetricsResponse),
        "disasm" => schemars::schema_for!(DisasmResponse),
        "ranges" => schemars::schema_for!(RangesResponse),
        "graphs" => schemars::schema_for!(GraphsResponse),
        "sources" => schemars::schema_for!(SourcesResponse),
        "source-metrics" => schemars::schema_for!(SourceMetricsResponse),
        "warp-stalls" => schemars::schema_for!(WarpStallsResponse),
        other => return Err(NcuSourceError::unknown_schema_target(other)),
    };
    serde_json::to_value(schema).map_err(|source| NcuSourceError::serialize_schema(target, source))
}
