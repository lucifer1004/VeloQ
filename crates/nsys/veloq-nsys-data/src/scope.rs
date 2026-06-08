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
//! 5. *Multiple devices, command opts into implicit all-device scope:*
//!    same outcome as `--all-devices`. This is reserved for commands
//!    whose primary response already has an explicit device axis, or
//!    whose selected scope is trace-wide.
//!
//! `--stream` is valid only after the resolver has locked a single
//! device. CUDA stream identifiers are not a useful cross-device
//! scope axis for agent queries; `--all-devices --stream N` is rejected
//! instead of silently filtering stream id N on every device.
//!
//! ## Layering
//!
//! Lives in `veloq-nsys-data` (not `veloq`) because the SQL probes
//! (`TARGET_INFO_GPU`, `TARGET_INFO_CUDA_CONTEXT_INFO`) and the scope
//! axes (`deviceId`) are NSys-specific. The output type
//! [`veloq_core::AppliedScope`] stays cross-source so per-source
//! resolvers all emit the same envelope shape.

use crate::{NsysDataResult, Trace};
use std::collections::HashSet;
use thiserror::Error;
use veloq_core::{AppliedScope, AxisUsage, Warning, WarningCode, WarningSeverity};

const NO_AXES: &[&str] = &[];
const DEVICE_AXIS: &[&str] = &["device"];

/// Caller-supplied scope inputs the resolver actually reasons about.
/// Mirrors the public fields of
/// `veloq_nsys::filters::GpuLocationFilters` plus the internal
/// `implicit_all_devices` opt-in for commands whose response shape is
/// already trace-wide or per-device. The verb dispatch fills the rest
/// of `AppliedScope` (`kind`, `nvtx_pattern`, `time_window_ns`) from
/// its own parsed args after the resolver returns — those don't
/// participate in ambiguity refusal so they don't belong here.
#[derive(Debug, Default, Clone)]
pub struct ScopeRequest {
    pub device: Option<i32>,
    pub stream: Option<i64>,
    pub all_devices: bool,
    pub implicit_all_devices: bool,
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
pub fn resolve_scope(
    trace: &Trace,
    req: ScopeRequest,
) -> std::result::Result<ResolvedScope, ResolveError> {
    // Case analysis. Note that `req.device` and
    // `req.all_devices` are mutually exclusive at the clap level; we
    // still defensively double-check here so internal callers that
    // bypass clap can't pass both.
    if let (true, Some(device)) = (req.all_devices, req.device) {
        return Err(ResolveError::probe(
            crate::NsysDataError::scope_conflicting_device_flags(device),
        ));
    }

    // Case analysis split by `all_devices` first so the match stays
    // exhaustive without a wildcard arm (each branch enumerates every
    // `(device, device_count)` pair it can hit).
    let (resolved_device, aggregated_over): (Option<i32>, Vec<String>) = if req.all_devices {
        // (Case 4) `--all-devices` opts into the aggregate.
        (None, vec!["device".to_string()])
    } else {
        let devices = device_set(trace).map_err(ResolveError::probe)?;
        match (req.device, devices.len()) {
            // Zero-device trace (no events at all). Resolver returns
            // None; verb's SQL will produce an empty result — the
            // guardrail layer in WI-C will warn.
            (None, 0) => (None, Vec::new()),

            // (Case 1) Single device: auto-resolve to the unique value.
            (None, 1) => (devices.into_iter().next(), Vec::new()),

            // Multi-device, no `--device`: `--stream` is more specific
            // than the generic ambiguity error because `--all-devices`
            // is not a valid recovery for a stream-local filter.
            (None, n) => {
                if let Some(err) = stream_parent_error(req.stream, NO_AXES) {
                    return Err(ResolveError::probe(err));
                }
                if req.implicit_all_devices {
                    (None, vec!["device".to_string()])
                } else {
                    return Err(ResolveError::Ambiguous(AmbiguityError::multi_device(n)));
                }
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

    let fixed_axes = if resolved_device.is_some() {
        DEVICE_AXIS
    } else {
        NO_AXES
    };
    if let Some(err) = stream_parent_error(req.stream, fixed_axes) {
        return Err(ResolveError::probe(err));
    }

    // Cross-axis bridge: when a single device is locked in, look up
    // the native_pid(s) that ran on it. In a TP workload this is the
    // single host-thread process; in PP setups it can be multiple.
    // We surface ONE native_pid on `applied_scope.native_pid` for the
    // v1 contract; the verb's host-thread SQL uses that value to
    // dedupe TP-replica rows.
    let native_pid = match resolved_device {
        Some(d) => native_pid_for_device(trace, d).map_err(ResolveError::probe)?,
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

fn stream_parent_error(
    stream: Option<i64>,
    fixed_axes: &[&'static str],
) -> Option<crate::NsysDataError> {
    let stream = stream?;
    AxisUsage::new(fixed_axes, NO_AXES)
        .validate_filter("stream", DEVICE_AXIS)
        .err()
        .map(|_| crate::NsysDataError::scope_stream_requires_device(stream))
}

/// Outcome wrapper so callers can distinguish "structured refusal"
/// (gets a `meta.warnings` entry on the error envelope) from "probe
/// failure" (the trace itself can't be read).
#[derive(Debug, Error)]
pub enum ResolveError {
    #[error(transparent)]
    Ambiguous(AmbiguityError),
    #[error("probing scope: {0:#}")]
    Probe(#[source] Box<crate::NsysDataError>),
}

impl ResolveError {
    fn probe(source: crate::NsysDataError) -> Self {
        Self::Probe(Box::new(source))
    }
}

/// Set of distinct `deviceId`s present in the trace's GPU event
/// tables. Reads `TARGET_INFO_GPU.cuDevice` first (preferred — one row
/// per CUDA device the runtime knew about); falls back to `SELECT DISTINCT
/// deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL` when the inventory table
/// is absent.
fn device_set(trace: &Trace) -> NsysDataResult<HashSet<i32>> {
    let mut out = HashSet::new();

    if trace.has_table("TARGET_INFO_GPU") {
        let mut stmt = trace
            .conn()
            .prepare("SELECT CAST(cuDevice AS INTEGER) FROM nsight.TARGET_INFO_GPU")
            .map_err(|source| {
                crate::NsysDataError::scope_device_probe_column_missing(
                    "TARGET_INFO_GPU",
                    "cuDevice",
                    source,
                )
            })?;
        let mut rows = stmt.query([]).map_err(|source| {
            crate::NsysDataError::scope_device_probe_query("TARGET_INFO_GPU", source)
        })?;
        while let Some(r) = rows.next().map_err(|source| {
            crate::NsysDataError::scope_device_probe_read("TARGET_INFO_GPU", source)
        })? {
            let id: Option<i32> = r.get(0).map_err(|source| {
                crate::NsysDataError::scope_device_probe_read("TARGET_INFO_GPU", source)
            })?;
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
        let mut stmt = trace.conn().prepare(&sql).map_err(|source| {
            crate::NsysDataError::scope_device_probe_column_missing(table, "deviceId", source)
        })?;
        let mut rows = stmt
            .query([])
            .map_err(|source| crate::NsysDataError::scope_device_probe_query(table, source))?;
        while let Some(r) = rows
            .next()
            .map_err(|source| crate::NsysDataError::scope_device_probe_read(table, source))?
        {
            let id: i32 = r
                .get(0)
                .map_err(|source| crate::NsysDataError::scope_device_probe_read(table, source))?;
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
fn native_pid_for_device(trace: &Trace, device: i32) -> NsysDataResult<Option<i64>> {
    if !trace.has_table("TARGET_INFO_CUDA_CONTEXT_INFO") {
        return Ok(None);
    }
    let mut stmt = trace
        .conn()
        .prepare(
            "SELECT CAST(processId AS BIGINT) FROM nsight.TARGET_INFO_CUDA_CONTEXT_INFO \
         WHERE CAST(deviceId AS INTEGER) = ? \
         ORDER BY processId ASC LIMIT 1",
        )
        .map_err(|source| {
            crate::NsysDataError::scope_device_probe_column_missing(
                "TARGET_INFO_CUDA_CONTEXT_INFO",
                "deviceId/processId",
                source,
            )
        })?;
    let mut rows = stmt.query([device]).map_err(|source| {
        crate::NsysDataError::scope_device_probe_query("TARGET_INFO_CUDA_CONTEXT_INFO", source)
    })?;
    Ok(
        if let Some(row) = rows.next().map_err(|source| {
            crate::NsysDataError::scope_device_probe_read("TARGET_INFO_CUDA_CONTEXT_INFO", source)
        })? {
            Some(row.get::<_, i64>(0).map_err(|source| {
                crate::NsysDataError::scope_device_probe_read(
                    "TARGET_INFO_CUDA_CONTEXT_INFO",
                    source,
                )
            })?)
        } else {
            None
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use duckdb::Connection;
    use std::path::PathBuf;
    use tempfile::TempDir;
    use veloq_core::VeloqDiagnostic;

    fn parquet_fixture(tables: &[(&str, &str)]) -> Result<(TempDir, PathBuf)> {
        let tables_with_rows = tables
            .iter()
            .map(|(table, ddl)| (*table, *ddl, Vec::new()))
            .collect::<Vec<_>>();
        parquet_fixture_with_rows(&tables_with_rows)
    }

    fn parquet_fixture_with_rows(tables: &[(&str, &str, Vec<&str>)]) -> Result<(TempDir, PathBuf)> {
        let dir = tempfile::tempdir()?;
        let pqtdir = dir.path().join("test_pqtdir");
        std::fs::create_dir_all(&pqtdir)?;
        let conn = Connection::open_in_memory()?;
        for (_, ddl, inserts) in tables {
            conn.execute_batch(ddl)?;
            for insert in inserts {
                conn.execute_batch(insert)?;
            }
        }
        for (table, _, _) in tables {
            let out = pqtdir.join(format!("{table}.parquet"));
            let out_lit = out.to_string_lossy().replace('\'', "''");
            conn.execute(
                &format!(r#"COPY (SELECT * FROM "{table}") TO '{out_lit}' (FORMAT PARQUET)"#),
                [],
            )?;
        }
        Ok((dir, pqtdir))
    }

    fn assert_scope_probe_query_error(
        err: crate::NsysDataError,
        expected_table: &str,
    ) -> Result<()> {
        assert_eq!(err.code().as_str(), "nsys.data.duckdb-query");
        assert_eq!(
            err.duckdb_parts(),
            Some((
                "scope device probe",
                crate::DuckdbPhase::Query,
                expected_table
            ))
        );
        Ok(())
    }

    fn downcast_scope_probe_error(err: ResolveError) -> Result<crate::NsysDataError> {
        match err {
            ResolveError::Probe(err) => Ok(*err),
            ResolveError::Ambiguous(err) => {
                anyhow::bail!("expected scope probe error, got ambiguity: {err}")
            }
        }
    }

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

    #[test]
    fn conflicting_device_scope_flags_have_typed_data_error() {
        let err = crate::NsysDataError::scope_conflicting_device_flags(7);
        assert_eq!(
            err.code().as_str(),
            "nsys.data.scope-conflicting-device-flags"
        );
        assert!(err.to_string().contains("--device 7"));
        assert!(err.to_string().contains("--all-devices"));
    }

    #[test]
    fn implicit_all_devices_resolves_multi_device_trace() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture_with_rows(&[
            (
                "CUPTI_ACTIVITY_KIND_KERNEL",
                r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start BIGINT, "end" BIGINT)"#,
                Vec::new(),
            ),
            (
                "TARGET_INFO_GPU",
                "CREATE TABLE TARGET_INFO_GPU (cuDevice BIGINT)",
                vec![
                    "INSERT INTO TARGET_INFO_GPU (cuDevice) VALUES (0)",
                    "INSERT INTO TARGET_INFO_GPU (cuDevice) VALUES (1)",
                ],
            ),
        ])?;
        let trace = Trace::open(&pqtdir)?;

        let strict = resolve_scope(&trace, ScopeRequest::default());
        assert!(
            matches!(strict, Err(ResolveError::Ambiguous(_))),
            "strict multi-device request should still refuse: {strict:?}",
        );

        let resolved = resolve_scope(
            &trace,
            ScopeRequest {
                implicit_all_devices: true,
                ..ScopeRequest::default()
            },
        )?;
        assert!(
            resolved.applied.device.is_none(),
            "implicit all-device scope must not lock one device: {resolved:?}",
        );
        let axis = resolved
            .applied
            .aggregated_over
            .first()
            .ok_or_else(|| anyhow::anyhow!("expected aggregated_over device axis"))?;
        assert_eq!(axis, "device");
        Ok(())
    }

    #[test]
    fn stream_requires_single_resolved_device_scope() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture_with_rows(&[
            (
                "CUPTI_ACTIVITY_KIND_KERNEL",
                r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start BIGINT, "end" BIGINT)"#,
                Vec::new(),
            ),
            (
                "TARGET_INFO_GPU",
                "CREATE TABLE TARGET_INFO_GPU (cuDevice BIGINT)",
                vec![
                    "INSERT INTO TARGET_INFO_GPU (cuDevice) VALUES (0)",
                    "INSERT INTO TARGET_INFO_GPU (cuDevice) VALUES (1)",
                ],
            ),
        ])?;
        let trace = Trace::open(&pqtdir)?;

        let err = match resolve_scope(
            &trace,
            ScopeRequest {
                stream: Some(7),
                all_devices: true,
                ..ScopeRequest::default()
            },
        ) {
            Ok(scope) => anyhow::bail!("stream across all devices should fail: {scope:?}"),
            Err(err) => downcast_scope_probe_error(err)?,
        };
        assert_eq!(
            err.code().as_str(),
            "nsys.data.scope-stream-requires-device"
        );

        let scoped = resolve_scope(
            &trace,
            ScopeRequest {
                device: Some(0),
                stream: Some(7),
                ..ScopeRequest::default()
            },
        )?;
        assert_eq!(scoped.applied.device, Some(0));
        assert_eq!(scoped.applied.stream, Some(7));
        Ok(())
    }

    #[test]
    fn target_info_gpu_probe_missing_cudevice_has_typed_data_error() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture(&[
            (
                "CUPTI_ACTIVITY_KIND_KERNEL",
                r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start BIGINT, "end" BIGINT)"#,
            ),
            (
                "TARGET_INFO_GPU",
                "CREATE TABLE TARGET_INFO_GPU (id BIGINT)",
            ),
        ])?;
        let trace = Trace::open(&pqtdir)?;

        let err = match resolve_scope(&trace, ScopeRequest::default()) {
            Ok(scope) => anyhow::bail!("malformed TARGET_INFO_GPU should fail: {scope:?}"),
            Err(err) => downcast_scope_probe_error(err)?,
        };

        assert_eq!(
            err.code().as_str(),
            "nsys.data.scope-device-probe-column-missing"
        );
        match err {
            crate::NsysDataError::ScopeDeviceProbeColumnMissing { table, column, .. } => {
                assert_eq!(table, "TARGET_INFO_GPU");
                assert_eq!(column, "cuDevice");
            }
            other => anyhow::bail!("expected ScopeDeviceProbeColumnMissing, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn target_info_gpu_probe_bad_cudevice_has_typed_query_error() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture_with_rows(&[
            (
                "CUPTI_ACTIVITY_KIND_KERNEL",
                r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start BIGINT, "end" BIGINT)"#,
                Vec::new(),
            ),
            (
                "TARGET_INFO_GPU",
                "CREATE TABLE TARGET_INFO_GPU (cuDevice TEXT)",
                vec!["INSERT INTO TARGET_INFO_GPU (cuDevice) VALUES ('bad')"],
            ),
        ])?;
        let trace = Trace::open(&pqtdir)?;

        let err = match resolve_scope(&trace, ScopeRequest::default()) {
            Ok(scope) => anyhow::bail!("invalid TARGET_INFO_GPU should fail: {scope:?}"),
            Err(err) => downcast_scope_probe_error(err)?,
        };

        assert_scope_probe_query_error(err, "TARGET_INFO_GPU")
    }

    #[test]
    fn fallback_device_probe_missing_deviceid_has_typed_data_error() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture(&[(
            "CUPTI_ACTIVITY_KIND_KERNEL",
            r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start BIGINT, "end" BIGINT)"#,
        )])?;
        let trace = Trace::open(&pqtdir)?;

        let err = match resolve_scope(&trace, ScopeRequest::default()) {
            Ok(scope) => anyhow::bail!("malformed kernel table should fail: {scope:?}"),
            Err(err) => downcast_scope_probe_error(err)?,
        };

        assert_eq!(
            err.code().as_str(),
            "nsys.data.scope-device-probe-column-missing"
        );
        match err {
            crate::NsysDataError::ScopeDeviceProbeColumnMissing { table, column, .. } => {
                assert_eq!(table, "CUPTI_ACTIVITY_KIND_KERNEL");
                assert_eq!(column, "deviceId");
            }
            other => anyhow::bail!("expected ScopeDeviceProbeColumnMissing, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn fallback_device_probe_bad_deviceid_has_typed_query_error() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture_with_rows(&[(
            "CUPTI_ACTIVITY_KIND_KERNEL",
            r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start BIGINT, "end" BIGINT, deviceId TEXT)"#,
            vec![
                r#"INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (start, "end", deviceId) VALUES (0, 1, 'bad')"#,
            ],
        )])?;
        let trace = Trace::open(&pqtdir)?;

        let err = match resolve_scope(&trace, ScopeRequest::default()) {
            Ok(scope) => anyhow::bail!("invalid kernel deviceId should fail: {scope:?}"),
            Err(err) => downcast_scope_probe_error(err)?,
        };

        assert_scope_probe_query_error(err, "CUPTI_ACTIVITY_KIND_KERNEL")
    }

    #[test]
    fn native_pid_probe_missing_context_processid_surfaces_typed_error() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture_with_rows(&[
            (
                "CUPTI_ACTIVITY_KIND_KERNEL",
                r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start BIGINT, "end" BIGINT)"#,
                Vec::new(),
            ),
            (
                "TARGET_INFO_GPU",
                "CREATE TABLE TARGET_INFO_GPU (cuDevice BIGINT)",
                vec!["INSERT INTO TARGET_INFO_GPU (cuDevice) VALUES (0)"],
            ),
            (
                "TARGET_INFO_CUDA_CONTEXT_INFO",
                "CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO (deviceId BIGINT)",
                Vec::new(),
            ),
        ])?;
        let trace = Trace::open(&pqtdir)?;

        let err = match resolve_scope(
            &trace,
            ScopeRequest {
                device: Some(0),
                ..ScopeRequest::default()
            },
        ) {
            Ok(scope) => anyhow::bail!("malformed context table should fail: {scope:?}"),
            Err(err) => downcast_scope_probe_error(err)?,
        };

        assert_eq!(
            err.code().as_str(),
            "nsys.data.scope-device-probe-column-missing"
        );
        match err {
            crate::NsysDataError::ScopeDeviceProbeColumnMissing { table, column, .. } => {
                assert_eq!(table, "TARGET_INFO_CUDA_CONTEXT_INFO");
                assert_eq!(column, "deviceId/processId");
            }
            other => anyhow::bail!("expected ScopeDeviceProbeColumnMissing, got {other:?}"),
        }
        Ok(())
    }
}
