//! Shared registry of `veloq pytorch schema <target>` entries.
//!
//! This is the PyTorch schema-target SSOT. The resolver, error
//! messages, CLI help, and drift tests all consume this registry so the
//! target list cannot silently diverge.

use crate::{PytorchCommandError, PytorchCommandResult};
use veloq_pytorch_query::{
    CollectivesResponse, CorrelateResponse, InspectResponse, PrepResponse, SearchResponse,
    SlicesResponse, StatsResponse, SummaryResponse, TimelineResponse,
};

pub struct SchemaTarget {
    pub name: &'static str,
    pub schema_fn: fn() -> PytorchCommandResult<serde_json::Value>,
}

macro_rules! target {
    ($name:expr, $ty:ty) => {
        SchemaTarget {
            name: $name,
            schema_fn: || {
                serde_json::to_value(schemars::schema_for!($ty))
                    .map_err(|source| PytorchCommandError::serialize_schema($name, source))
            },
        }
    };
}

pub const TARGETS: &[SchemaTarget] = &[
    target!("summary", SummaryResponse),
    target!("search", SearchResponse),
    target!("inspect", InspectResponse),
    target!("stats", StatsResponse),
    target!("correlate", CorrelateResponse),
    target!("timeline", TimelineResponse),
    target!("slices", SlicesResponse),
    target!("collectives", CollectivesResponse),
    target!("prep", PrepResponse),
];

pub fn render_target_list() -> String {
    TARGETS
        .iter()
        .map(|target| target.name)
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn resolve(name: &str) -> PytorchCommandResult<serde_json::Value> {
    if let Some(target) = TARGETS.iter().find(|target| target.name == name) {
        return (target.schema_fn)();
    }
    Err(PytorchCommandError::unknown_schema_target(
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
