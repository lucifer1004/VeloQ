//! `veloq ncu schema <target>` — strict JSON Schema for NCU response
//! payloads.
//!
//! Mirrors the NSys schema endpoint: the schema is generated from the
//! same Rust type that `serde_json` serializes on successful command
//! output, then wrapped in the shared envelope by `source.rs`.

use crate::error::NcuSourceResult;

/// Response payload for `veloq ncu schema <target>`.
#[derive(serde::Serialize)]
pub struct SchemaPayload {
    pub target: String,
    pub schema: serde_json::Value,
}

/// Dispatch `target` to the matching NCU response payload type.
pub fn schema_value_for(target: &str) -> NcuSourceResult<serde_json::Value> {
    crate::schema_targets::resolve(target)
}
