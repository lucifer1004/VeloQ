//! Shared registry of `veloq schema <target>` entries — SSOT consumed
//! by [`crate::schema::schema_value_for`], the `cli::Cmd::Schema` arg
//! help (injected at runtime by [`crate::help::inject_long_about`]),
//! and [`crate::help::long_about_schema`]'s "Valid targets:" line.
//!
//! Adding a new public verb means adding one row in [`TARGETS`]; the
//! three consuming sites all derive from this slice, so they cannot
//! drift. The drift regression test in
//! `tests/schema_targets_drift.rs` enforces the invariant.
//!
//! Hidden targets live in [`HIDDEN_TARGETS`] and are routed only when
//! `VELOQ_UNSTABLE=1` is set in the calling environment, per the
//! hidden-exposure rule. Public targets live in [`TARGETS`].

use crate::error::{NsysSourceError, NsysSourceResult};
use crate::payloads::{CorrelationStatsPayload, PrepPayload, PrepStatusPayload};
use veloq_nsys_query::{
    concurrency::ConcurrencyResponse, correlate::CorrelateResponse, gaps::GapsResponse,
    graph_replays::GraphReplaysResponse, hardware::HardwareResponse, inspect::InspectResponse,
    metrics::MetricsResponse, ncu_command::NcuCommandResponse, search::SearchResponse,
    slices::SlicesResponse, stats::StatsResponse, stats_by_size::StatsBySizeResponse,
    summary::Summary, timeline::TimelineResponse,
};

/// One row of the schema-target registry. `schema_fn` returns the
/// JSON Schema document for the named response type; bodies are
/// non-capturing closures so the slice can be `const`.
pub struct SchemaTarget {
    pub name: &'static str,
    pub schema_fn: fn() -> NsysSourceResult<serde_json::Value>,
}

macro_rules! target {
    ($name:expr, $ty:ty) => {
        SchemaTarget {
            name: $name,
            schema_fn: || {
                serde_json::to_value(schemars::schema_for!($ty))
                    .map_err(|source| NsysSourceError::serialize_schema($name, source))
            },
        }
    };
}

/// Always-visible schema targets. Order is significant: it
/// determines the help output, which is asserted byte-stable.
pub const TARGETS: &[SchemaTarget] = &[
    target!("summary", Summary),
    target!("stats", StatsResponse),
    target!("search", SearchResponse),
    target!("inspect", InspectResponse),
    target!("correlate", CorrelateResponse),
    target!("graph-replays", GraphReplaysResponse),
    target!("ncu-command", NcuCommandResponse),
    target!("concurrency", ConcurrencyResponse),
    target!("gaps", GapsResponse),
    target!("timeline", TimelineResponse),
    target!("slices", SlicesResponse),
    target!("hardware", HardwareResponse),
    target!("metrics", MetricsResponse),
    target!("prep", PrepPayload),
    target!("prep-status", PrepStatusPayload),
    target!("correlation-stats", CorrelationStatsPayload),
];

/// Targets resolved only when `VELOQ_UNSTABLE=1`.
pub const HIDDEN_TARGETS: &[SchemaTarget] = &[target!("stats-by-size", StatsBySizeResponse)];

const UNSTABLE_ENV: &str = "VELOQ_UNSTABLE";

fn unstable_enabled() -> bool {
    std::env::var(UNSTABLE_ENV)
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Targets currently visible to the user given the live process
/// environment. `HIDDEN_TARGETS` slice is appended when
/// `VELOQ_UNSTABLE=1`.
pub fn visible_targets() -> Vec<&'static SchemaTarget> {
    let mut out: Vec<&'static SchemaTarget> = TARGETS.iter().collect();
    if unstable_enabled() {
        out.extend(HIDDEN_TARGETS.iter());
    }
    out
}

/// Comma-separated list of visible target names. Used by the
/// `cli.rs` arg help injection and the `help.rs` long_about_schema
/// renderer so neither hand-maintains the list.
pub fn render_target_list() -> String {
    visible_targets()
        .iter()
        .map(|t| t.name)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Resolve a target name to its JSON Schema document. Hidden
/// targets resolve only when `VELOQ_UNSTABLE=1`; otherwise they
/// behave as if absent.
pub fn resolve(name: &str) -> NsysSourceResult<serde_json::Value> {
    if let Some(t) = TARGETS.iter().find(|t| t.name == name) {
        return (t.schema_fn)();
    }
    if unstable_enabled()
        && let Some(t) = HIDDEN_TARGETS.iter().find(|t| t.name == name)
    {
        return (t.schema_fn)();
    }
    Err(NsysSourceError::unknown_schema_target(
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
        for entry in TARGETS {
            let v = (entry.schema_fn)()?;
            assert!(
                v.is_object(),
                "schema for {} should be a JSON object",
                entry.name
            );
        }
        Ok(())
    }

    #[test]
    fn prep_status_in_registry() {
        // Regression: prep-status must be in the TARGETS registry so
        // schema.rs, cli.rs, and help.rs all stay in sync. This test
        // closes that drift gap — do not delete.
        assert!(
            TARGETS.iter().any(|t| t.name == "prep-status"),
            "prep-status must be in TARGETS"
        );
    }

    #[test]
    fn resolve_unknown_lists_visible_targets() -> Result<()> {
        let msg = match resolve("does-not-exist") {
            Ok(_) => anyhow::bail!("resolve of unknown target should return Err"),
            Err(e) => e.to_string(),
        };
        for entry in TARGETS {
            assert!(msg.contains(entry.name), "error must list {}", entry.name);
        }
        Ok(())
    }
}
