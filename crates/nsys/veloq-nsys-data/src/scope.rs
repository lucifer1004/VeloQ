//! Scope-ambiguity resolver for list verbs.
//!
//! Every veloq list verb that operates on event rows
//! takes the same scope filter — `--device <N>` (existing on
//! `stats`/`search`/`gaps`/`timeline`, added to `slices` in v3) — plus
//! the `--all-devices` aggregator opt-in. This module owns the rule:
//!
//! 1. *Single device:* resolver returns the unique value automatically.
//!    No flag required.
//! 2. *Multiple devices, `--device` unset, `--all-devices` unset:*
//!    refuse with [`AmbiguityError`]. The verb dispatch surfaces this
//!    as an `EnvelopeError` carrying a `multi-device-ambiguous` warning
//!    code so the agent reads the refusal in
//!    structured form.
//! 3. *Multiple devices, `--device <id>` set:* resolver returns the
//!    single resolved device PLUS the native_pid that ran work on it
//!    (looked up via `TARGET_INFO_CUDA_CONTEXT_INFO`). The cross-axis
//!    bridge is the load-bearing TP-dedup mechanism: each host process
//!    emits NVTX on its own `globalTid`, so filtering by the resolved
//!    native_pid is what deduplicates the per-process rows in
//!    `slices` on a TP workload.
//! 4. *Multiple devices, `--all-devices` set:* aggregate response;
//!    `applied_scope.aggregated_over = ["device"]`.
//!
//! `--stream` does not participate in ambiguity refusal — cross-stream
//! sum on one device is not the wrong-answer footgun this resolver
//! addresses.
//!
//! ## Layering
//!
//! Lives in `veloq-nsys-data` (not `veloq`) because the SQL probes
//! (`TARGET_INFO_GPU`, `TARGET_INFO_CUDA_CONTEXT_INFO`) and the scope
//! axes (`deviceId`) are NSys-specific. The output type
//! [`veloq_core::AppliedScope`] stays cross-source so per-source
//! resolvers all emit the same envelope shape.

use crate::Trace;
use anyhow::{Context, Result};
use std::collections::HashSet;
use thiserror::Error;
use veloq_core::{AppliedScope, Warning, WarningCode, WarningSeverity};

/// Caller-supplied scope inputs the resolver actually reasons about.
/// Mirrors the public fields of
/// `veloq_nsys::filters::GpuLocationFilters`. The verb dispatch fills
/// the rest of `AppliedScope` (`kind`, `nvtx_pattern`, `time_window_ns`)
/// from its own parsed args after the resolver returns — those don't
/// participate in ambiguity refusal so they don't belong here.
#[derive(Debug, Default, Clone)]
pub struct ScopeRequest {
    pub device: Option<i32>,
    pub stream: Option<i64>,
    pub all_devices: bool,
}

/// Outcome of [`resolve_scope`]: the verb either gets an `AppliedScope`
/// to populate `envelope.meta.applied_scope` AND to feed back into its
/// SQL WHERE clauses, or it gets an [`AmbiguityError`] to surface as a
/// structured `EnvelopeError`.
#[derive(Debug)]
pub struct ResolvedScope {
    /// Populated unconditionally — verbs assign this into the
    /// envelope's `meta.applied_scope` block.
    pub applied: AppliedScope,
}

/// Refusal payload for an ambiguous query — multi-device trace with no
/// `--device` and no `--all-devices` opt-in. Holds a human-readable
/// message + the `WarningCode` that goes into the error envelope's
/// `meta.warnings`.
#[derive(Debug, Error)]
#[error("{message}")]
pub struct AmbiguityError {
    pub message: String,
    pub warning: Warning,
}

impl AmbiguityError {
    fn multi_device(n: usize) -> Self {
        let message = format!(
            "trace has {n} devices; pass `--device <id>` or `--all-devices` to scope the query"
        );
        Self {
            warning: Warning {
                severity: WarningSeverity::Warn,
                code: WarningCode::MultiDeviceAmbiguous,
                message: message.clone(),
            },
            message,
        }
    }
}

/// Resolve the scope for a query. Reads the trace's device set, picks
/// the resolution path per the case analysis above, and (when a single
/// device is locked in) cross-references `TARGET_INFO_CUDA_CONTEXT_INFO`
/// to surface the native_pid that ran on it.
///
/// Returns `Ok(ResolvedScope)` when the query is unambiguous (with or
/// without a flag) — caller assigns `.applied` into the envelope's
/// `meta.applied_scope` and uses the resolved fields in its SQL WHERE
/// clauses.
///
/// Returns `Err(AmbiguityError)` when the trace has >1 device and the
/// user gave neither `--device` nor `--all-devices`. The verb's
/// `run()` converts the error into an `EnvelopeError` whose
/// `meta.warnings` carries the structured `multi-device-ambiguous`
/// code.
pub fn resolve_scope(trace: &Trace, req: ScopeRequest) -> Result<ResolvedScope, ResolveError> {
    let devices = device_set(trace).map_err(ResolveError::Probe)?;

    // Case analysis. Note that `req.device` and
    // `req.all_devices` are mutually exclusive at the clap level; we
    // still defensively double-check here so internal callers that
    // bypass clap can't pass both.
    if req.device.is_some() && req.all_devices {
        return Err(ResolveError::Probe(anyhow::anyhow!(
            "internal: --device and --all-devices are mutually exclusive but both were supplied"
        )));
    }

    // Case analysis split by `all_devices` first so the match stays
    // exhaustive without a wildcard arm (each branch enumerates every
    // `(device, device_count)` pair it can hit).
    let (resolved_device, aggregated_over): (Option<i32>, Vec<String>) = if req.all_devices {
        // (Case 4) `--all-devices` opts into the aggregate. `--device`
        // is rejected at clap level (`conflicts_with`); defensive check
        // here catches internal callers that bypass clap.
        if let Some(d) = req.device {
            return Err(ResolveError::Probe(anyhow::anyhow!(
                "internal: --device {d} + --all-devices reached the resolver — \
                 clap should have rejected this at parse time"
            )));
        }
        (None, vec!["device".to_string()])
    } else {
        match (req.device, devices.len()) {
            // Zero-device trace (no events at all). Resolver returns
            // None; verb's SQL will produce an empty result — the
            // guardrail layer in WI-C will warn.
            (None, 0) => (None, Vec::new()),

            // (Case 1) Single device: auto-resolve to the unique value.
            (None, 1) => (devices.into_iter().next(), Vec::new()),

            // (Case 2) Multi-device, no `--device` → refuse.
            (None, n) => {
                return Err(ResolveError::Ambiguous(AmbiguityError::multi_device(n)));
            }

            // (Case 3) Explicit `--device <N>` — single-device traces
            // accept it as long as it matches; multi-device traces
            // accept any value. We do NOT pre-validate that `d` is in
            // the device set: picking a device
            // with zero events is a valid request that returns an
            // empty success envelope with `empty-with-scope` warning.
            (Some(d), _) => (Some(d), Vec::new()),
        }
    };

    // Cross-axis bridge: when a single device is locked in, look up
    // the native_pid(s) that ran on it. In a TP workload this is the
    // single host-thread process; in PP setups it can be multiple.
    // We surface ONE native_pid on `applied_scope.native_pid` for the
    // v1 contract; the verb's host-thread SQL uses that value to
    // dedupe TP-replica rows.
    let native_pid = match resolved_device {
        Some(d) => native_pid_for_device(trace, d).unwrap_or(None),
        None => None,
    };

    Ok(ResolvedScope {
        applied: AppliedScope {
            device: resolved_device,
            stream: req.stream,
            native_pid,
            // kind / nvtx_pattern / time_window_ns are filled by the
            // verb dispatch after `resolve_scope` returns. The resolver
            // never reads them.
            kind: None,
            nvtx_pattern: None,
            time_window_ns: None,
            aggregated_over,
        },
    })
}

/// Outcome wrapper so callers can distinguish "structured refusal"
/// (gets a `meta.warnings` entry on the error envelope) from "probe
/// failure" (the trace itself can't be read).
#[derive(Debug, Error)]
pub enum ResolveError {
    #[error(transparent)]
    Ambiguous(AmbiguityError),
    #[error("probing scope: {0:#}")]
    Probe(anyhow::Error),
}

/// Set of distinct `deviceId`s present in the trace's GPU event
/// tables. Reads `TARGET_INFO_GPU.cuDevice` first (preferred — one row
/// per CUDA device the runtime knew about); falls back to `SELECT DISTINCT
/// deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL` when the inventory table
/// is absent.
fn device_set(trace: &Trace) -> Result<HashSet<i32>> {
    let mut out = HashSet::new();

    if trace.has_table("TARGET_INFO_GPU") {
        let mut stmt = trace
            .conn()
            .prepare("SELECT CAST(cuDevice AS INTEGER) FROM nsight.TARGET_INFO_GPU")
            .context("preparing TARGET_INFO_GPU device probe")?;
        let mut rows = stmt.query([])?;
        while let Some(r) = rows.next()? {
            let id: Option<i32> = r.get(0)?;
            if let Some(id) = id {
                out.insert(id);
            }
        }
        if !out.is_empty() {
            return Ok(out);
        }
    }

    // Fallback to DISTINCT scans over location-bearing activity
    // tables. Kernel is the most common table, but graph-trace-only
    // captures can legitimately have no kernel rows while still
    // spanning multiple devices.
    for table in [
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "CUPTI_ACTIVITY_KIND_MEMSET",
        "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
        "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
    ] {
        if !trace.has_table(table) {
            continue;
        }
        let sql = format!(
            "SELECT DISTINCT CAST(deviceId AS INTEGER) \
             FROM nsight.\"{table}\" \
             WHERE deviceId IS NOT NULL"
        );
        let mut stmt = trace
            .conn()
            .prepare(&sql)
            .with_context(|| format!("preparing DISTINCT deviceId fallback probe for {table}"))?;
        let mut rows = stmt.query([])?;
        while let Some(r) = rows.next()? {
            let id: i32 = r.get(0)?;
            out.insert(id);
        }
    }

    Ok(out)
}

/// Map `deviceId -> native_pid` via `TARGET_INFO_CUDA_CONTEXT_INFO`
/// joined with the high-24-bit native_pid extraction used everywhere
/// else (per `AGENTS.md` globalTid bit layout / `decode_global_tid`).
///
/// Returns the *first* native_pid found for the device — in a TP
/// workload there is exactly one host process per device, so "first"
/// = "the one that ran on this device". On PP / shared-device
/// workloads multiple native_pids can run on one device; we surface
/// the first match because the v1 `applied_scope.native_pid` field
/// carries a single value. Future ADR will extend if PP becomes a
/// target workload.
///
/// `Ok(None)` when `TARGET_INFO_CUDA_CONTEXT_INFO` is absent (older
/// nsys captures), or no row matches the device — both are valid
/// "host pid unknown" states and the caller surfaces `null` to the
/// agent.
fn native_pid_for_device(trace: &Trace, device: i32) -> Result<Option<i64>> {
    if !trace.has_table("TARGET_INFO_CUDA_CONTEXT_INFO") {
        return Ok(None);
    }
    let mut stmt = trace.conn().prepare(
        "SELECT CAST(processId AS BIGINT) FROM nsight.TARGET_INFO_CUDA_CONTEXT_INFO \
         WHERE CAST(deviceId AS INTEGER) = ? \
         ORDER BY processId ASC LIMIT 1",
    )?;
    let mut rows = stmt.query([device])?;
    Ok(if let Some(row) = rows.next()? {
        Some(row.get::<_, i64>(0)?)
    } else {
        None
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ambiguity_error_carries_multi_device_warning_code() {
        let e = AmbiguityError::multi_device(4);
        assert_eq!(
            e.warning.code as u8,
            WarningCode::MultiDeviceAmbiguous as u8,
            "code variant must be MultiDeviceAmbiguous"
        );
        assert!(matches!(e.warning.severity, WarningSeverity::Warn));
        assert!(e.message.contains("4 devices"), "message: {}", e.message);
        assert!(
            e.message.contains("--device") && e.message.contains("--all-devices"),
            "message must mention both flags: {}",
            e.message
        );
    }
}
