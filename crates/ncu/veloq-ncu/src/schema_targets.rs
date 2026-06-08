//! Shared registry of `veloq ncu schema <target>` entries.
//!
//! This is the NCU schema-target SSOT. The resolver, error messages,
//! CLI help, and drift tests all consume this registry so the target
//! list cannot silently diverge.

use crate::disasm::DisasmResponse;
use crate::error::{NcuSourceError, NcuSourceResult};
use crate::inspect::InspectResponse;
use crate::launches::LaunchesResponse;
use crate::lists::{GraphsResponse, RangesResponse, SourcesResponse};
use crate::metrics::MetricsResponse;
use crate::native::NativeSummaryResponse;
use crate::source_metrics::SourceMetricsResponse;
use crate::warp_stalls::WarpStallsResponse;

pub struct SchemaTarget {
    pub name: &'static str,
    pub schema_fn: fn() -> NcuSourceResult<serde_json::Value>,
}

macro_rules! target {
    ($name:expr, $ty:ty) => {
        SchemaTarget {
            name: $name,
            schema_fn: || {
                serde_json::to_value(schemars::schema_for!($ty))
                    .map_err(|source| NcuSourceError::serialize_schema($name, source))
            },
        }
    };
}

pub const TARGETS: &[SchemaTarget] = &[
    target!("summary", NativeSummaryResponse),
    target!("launches", LaunchesResponse),
    target!("inspect", InspectResponse),
    target!("metrics", MetricsResponse),
    target!("disasm", DisasmResponse),
    target!("ranges", RangesResponse),
    target!("graphs", GraphsResponse),
    target!("sources", SourcesResponse),
    target!("source-metrics", SourceMetricsResponse),
    target!("warp-stalls", WarpStallsResponse),
];

pub fn render_target_list() -> String {
    TARGETS
        .iter()
        .map(|target| target.name)
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn resolve(name: &str) -> NcuSourceResult<serde_json::Value> {
    if let Some(target) = TARGETS.iter().find(|target| target.name == name) {
        return (target.schema_fn)();
    }
    Err(NcuSourceError::unknown_schema_target(
        name,
        render_target_list(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    #[test]
    fn registry_resolves_every_target() -> Result<()> {
        for target in TARGETS {
            let value = (target.schema_fn)()?;
            assert!(
                value.is_object(),
                "schema for {} should be a JSON object",
                target.name
            );
        }
        Ok(())
    }

    #[test]
    fn resolve_unknown_lists_every_target() -> Result<()> {
        let msg = match resolve("does-not-exist") {
            Ok(_) => anyhow::bail!("resolve of unknown target should return Err"),
            Err(error) => error.to_string(),
        };
        for target in TARGETS {
            assert!(msg.contains(target.name), "error must list {}", target.name);
        }
        Ok(())
    }
}
