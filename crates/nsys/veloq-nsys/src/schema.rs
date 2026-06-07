//! `veloq schema <target>` — meta endpoint that emits the strict
//! JSON Schema for one subcommand's response body.
//!
//! SSOT for tooling that wants machine-validated wire format: the
//! schema comes from `schemars::schema_for!(T)` on the same Rust
//! struct `serde_json` serialises out of, so the schema and the
//! actual responses cannot drift.
//!
//! Wire shape rides the standard v1 [`Envelope`] (with `trace`
//! omitted, since this call doesn't read a trace). The payload is
//! [`SchemaPayload`] from [`crate::payloads`].
//!
//! Target dispatch and the visible/hidden registry live in
//! [`crate::schema_targets`]; this module is a thin re-export.
//!
//! [`Envelope`]: veloq_core::Envelope
//! [`SchemaPayload`]: crate::payloads::SchemaPayload

use crate::error::NsysSourceResult;
use crate::schema_targets;

/// Dispatch `target` to the matching response type and return its
/// JSON Schema as a `serde_json::Value`. Adding a new subcommand
/// means adding one entry in [`schema_targets::TARGETS`] — schemars +
/// clap handle the rest of the wiring on their own.
pub fn schema_value_for(target: &str) -> NsysSourceResult<serde_json::Value> {
    schema_targets::resolve(target)
}
