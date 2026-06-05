//! Shared request-validation policies for the kind-aware verbs
//! (`stats`, `search`). Two silent-drop traps these policies prevent:
//!
//! 1. `--device <n>` / `--stream <n>` filters added to a request
//!    whose `--type` includes a kind without those columns
//!    (Nvtx / Runtime / Osrt / GraphNode / GraphEvent / Overhead).
//!    The projected `device_id` / `stream_id` are NULL for those
//!    kinds, so the SQL would silently drop the rows. Explicit
//!    null-location kinds are rejected with a redirecting error, and
//!    `--type all` narrows implicitly.
//!
//! 2. `--nvtx <pattern>` on a kind that isn't attributable (Osrt /
//!    Nvtx / Graph* / CudaEvent / Overhead / CpuSample). A `FALSE`
//!    clause would return zero rows, so explicit non-attributable
//!    kinds are rejected and `--type all` narrows implicitly to the
//!    attributable set.
//!
//! Housing these here (rather than re-validating in both `stats.rs`
//! and `search.rs`) keeps the wording — and the precise set rules —
//! identical across verbs.

use crate::{EventKind, KindFilter, nvtx_attribution};
use anyhow::Result;
use duckdb::types::Value;
use veloq_nsys_data::Trace;

/// Possible `--device` / `--stream` filter on the request. Used by
/// [`validate_location_filter`] so the caller doesn't have to
/// stringify presence flags itself.
#[derive(Debug, Clone, Copy, Default)]
pub struct LocationFilter {
    pub device: Option<i32>,
    pub stream: Option<i64>,
}

impl LocationFilter {
    pub fn any(self) -> bool {
        self.device.is_some() || self.stream.is_some()
    }

    /// Push the canonical `device_id = ?` / `stream_id = ?` predicates
    /// (device first, to fix the bind order) and their bound params onto
    /// a verb's WHERE accumulator. The bare `col = ?` form already
    /// excludes NULL device/stream rows by SQL semantics, so no
    /// `IS NOT NULL` guard is needed. Single source for the predicate
    /// text + `Value` typing shared across the verbs.
    pub fn push_where(self, parts: &mut Vec<String>, params: &mut Vec<Value>) {
        if let Some(d) = self.device {
            parts.push("device_id = ?".to_string());
            params.push(Value::Int(d));
        }
        if let Some(s) = self.stream {
            parts.push("stream_id = ?".to_string());
            params.push(Value::BigInt(s));
        }
    }

    /// String-accumulator variant for verbs that splice a single
    /// ` AND <pred>` run into their SQL (e.g. `stats`). Same predicates,
    /// same bind order as [`LocationFilter::push_where`].
    pub fn append_where(self, sql: &mut String, params: &mut Vec<Value>) {
        let mut parts = Vec::new();
        self.push_where(&mut parts, params);
        for p in parts {
            sql.push_str(" AND ");
            sql.push_str(&p);
        }
    }
}

/// Reject explicit null-location kinds combined with `--device` /
/// `--stream` filters. The complementary case — `KindFilter::All` +
/// `--device 0` — is allowed: the user typed no `--type`, so we
/// narrow implicitly to location-bearing kinds at SQL time and let
/// the request go through.
///
/// `verb` is the verb name (`"stats"` / `"search"`) so the error
/// quotes the actual command name.
pub fn validate_location_filter(
    kinds: &KindFilter,
    location: LocationFilter,
    verb: &str,
) -> Result<()> {
    if !location.any() {
        return Ok(());
    }
    let explicit = match kinds {
        KindFilter::Only(v) => v,
        // KindFilter::All narrows implicitly — no explicit
        // null-location kind to call out.
        KindFilter::All => return Ok(()),
    };
    let offenders: Vec<&'static str> = explicit
        .iter()
        .filter(|k| !k.is_location_bearing())
        .map(|k| k.as_str())
        .collect();
    if offenders.is_empty() {
        return Ok(());
    }
    let axes = match (location.device.is_some(), location.stream.is_some()) {
        (true, true) => "--device / --stream",
        (true, false) => "--device",
        (false, true) => "--stream",
        (false, false) => unreachable_axes(),
    };
    let kinds_csv = offenders.join(", ");
    anyhow::bail!(
        "{verb}: {axes} cannot be combined with `--type {kinds_csv}` — \
         these kinds are CPU-side host-thread events with no \
         device/stream columns. Drop `--type` to narrow to \
         location-bearing kinds (kernel/memcpy/memset/sync/graph/\
         cuda_event), or remove the location filter."
    )
}

// `unreachable!()` is denied workspace-wide; this returns the
// fallback string instead. The static-typed callsites above
// guarantee one of the four arms hits.
fn unreachable_axes() -> &'static str {
    "--device / --stream"
}

/// Single chokepoint for `--type` + `--nvtx` resolution across
/// every NVTX-bearing verb (`stats`, `search`, `timeline`). The
/// canonical pipeline is:
///
/// 1. **Validate explicit non-attributable kinds**: if `--nvtx` is
///    set and the user wrote `--type X` for a kind that can't be
///    attributed (Osrt / Graph* / CudaEvent / Overhead / CpuSample),
///    bail with a redirecting error.
/// 2. **Resolve to a concrete kind list** against the verb's
///    allow-list (`allowed`).
/// 3. **Filter by table presence**: kinds whose backing CUPTI table
///    isn't in the trace drop out silently.
/// 4. **(Optional) Implicit attributable narrowing**: when `--nvtx`
///    is set and the user wrote `--type all` (`KindFilter::All`),
///    drop non-attributable kinds without erroring — they're
///    implicitly out of scope, not user-asserted.
///
/// Pre-consolidation each verb hand-rolled some subset of these
/// steps and diverged: stats was missing the implicit narrowing
/// (calling `nvtx_attribution::build` with non-attributable kinds
/// like Graph could bail), timeline was missing the validator
/// (explicit non-attributable kinds slipped through). This helper
/// is the one place future attribution-set changes need touching.
pub fn resolve_nvtx_kinds(
    kinds: &KindFilter,
    nvtx: Option<&str>,
    allowed: &[EventKind],
    trace: &Trace,
    verb: &str,
) -> Result<Vec<EventKind>> {
    validate_nvtx_filter(kinds, nvtx, verb)?;
    let nvtx_requested = nvtx.is_some();
    let requested = kinds.resolve(allowed);
    Ok(allowed
        .iter()
        .copied()
        .filter(|k| requested.contains(k))
        .filter(|k| trace.table_exists(k.table()))
        .filter(|k| !nvtx_requested || nvtx_attribution::is_attributable(*k))
        .collect())
}

/// Reject explicit non-attributable kinds combined with `--nvtx`.
/// `KindFilter::All` narrows implicitly to the attributable set per
/// [`nvtx_attribution::is_attributable`].
///
/// Prefer [`resolve_nvtx_kinds`] when you also need the resolved
/// kind list — it threads validation + resolution through one call.
pub fn validate_nvtx_filter(kinds: &KindFilter, nvtx: Option<&str>, verb: &str) -> Result<()> {
    if nvtx.is_none() {
        return Ok(());
    }
    let explicit = match kinds {
        KindFilter::Only(v) => v,
        KindFilter::All => return Ok(()),
    };
    let offenders: Vec<&'static str> = explicit
        .iter()
        .filter(|k| !nvtx_attribution::is_attributable(**k))
        .map(|k| k.as_str())
        .collect();
    if offenders.is_empty() {
        return Ok(());
    }
    let kinds_csv = offenders.join(", ");
    anyhow::bail!(
        "{verb}: --nvtx cannot scope `--type {kinds_csv}` — NVTX \
         attribution for these kinds is experimental and not yet \
         implemented (the attributable set today is kernel/memcpy/\
         memset/sync/runtime). Drop the kind from `--type` or \
         remove `--nvtx` to widen the scope; future opt-in via the \
         workspace's experimental gate may extend the attributable \
         set."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_where_emits_canonical_predicates_in_device_then_stream_order() {
        let mut parts = Vec::new();
        let mut params = Vec::new();
        LocationFilter {
            device: Some(3),
            stream: Some(7),
        }
        .push_where(&mut parts, &mut params);
        assert_eq!(
            parts.iter().map(String::as_str).collect::<Vec<_>>(),
            ["device_id = ?", "stream_id = ?"]
        );
        assert!(matches!(params.first(), Some(Value::Int(3))));
        assert!(matches!(params.get(1), Some(Value::BigInt(7))));
    }

    #[test]
    fn push_where_device_only_and_empty() {
        let mut parts = Vec::new();
        let mut params = Vec::new();
        LocationFilter {
            device: Some(0),
            stream: None,
        }
        .push_where(&mut parts, &mut params);
        assert_eq!(
            parts.iter().map(String::as_str).collect::<Vec<_>>(),
            ["device_id = ?"]
        );
        assert_eq!(params.len(), 1);

        let mut p2 = Vec::new();
        let mut q2 = Vec::new();
        LocationFilter::default().push_where(&mut p2, &mut q2);
        assert!(p2.is_empty() && q2.is_empty());
    }

    #[test]
    fn append_where_joins_with_leading_and() {
        let mut sql = String::new();
        let mut params = Vec::new();
        LocationFilter {
            device: Some(1),
            stream: Some(2),
        }
        .append_where(&mut sql, &mut params);
        assert_eq!(sql, " AND device_id = ? AND stream_id = ?");
        assert_eq!(params.len(), 2);
    }
}
