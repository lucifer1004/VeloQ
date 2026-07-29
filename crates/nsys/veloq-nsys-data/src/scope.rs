//! Scope-ambiguity resolver for list verbs.
//!
//! Every veloq list verb that operates on event rows
//! takes the same scope filters — `--process <PID>` and `--device <N>`
//! (`stats`/`search`/`gaps`/`timeline`/`slices` and process-sensitive
//! graph/viz views) — plus
//! the `--all-devices` aggregator opt-in. This module owns the rule:
//!
//! 1. *Single process/device scope:* resolver returns both values automatically.
//!    No flag required.
//! 2. *Multiple devices, `--device` unset, `--all-devices` unset:*
//!    refuse with [`AmbiguityError`]. The verb dispatch surfaces this
//!    as an `EnvelopeError` carrying a `multi-device-ambiguous` warning
//!    code so the agent reads the refusal in
//!    structured form.
//! 3. *Colliding logical ordinals:* `--device <id>` alone refuses when
//!    multiple processes expose that same ordinal; `--process <PID>`
//!    disambiguates it.
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
    pub process: Option<i64>,
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

/// Refusal payload for a query matching multiple process/device scopes
/// without an exact selector or all-device opt-in. Holds a human-readable
/// message, structured warning, and exact recovery candidates.
#[derive(Debug, Error)]
#[error("{message}")]
pub struct AmbiguityError {
    pub message: String,
    pub warning: Warning,
    /// Stable process-local CUDA scopes that matched the request.
    /// The command layer uses the first pair for an executable exact
    /// recovery suggestion instead of suggesting another ambiguous
    /// bare device ordinal.
    pub candidate_scopes: Vec<(i64, i32)>,
    /// Preserve a caller-supplied process when suggesting an intentional
    /// all-device aggregate.
    pub requested_process: Option<i64>,
}

impl AmbiguityError {
    fn multi_device(mut candidate_scopes: Vec<(i64, i32)>, requested_process: Option<i64>) -> Self {
        candidate_scopes.sort_unstable();
        candidate_scopes.dedup();
        let n = candidate_scopes.len();
        let message = format!(
            "trace has {n} CUDA process/device scopes; pass `--process <pid>` with \
             `--device <id>`, or `--all-devices` to scope the query"
        );
        Self {
            warning: Warning {
                severity: WarningSeverity::Warn,
                code: WarningCode::MultiDeviceAmbiguous,
                message: message.clone(),
            },
            message,
            candidate_scopes,
            requested_process,
        }
    }
}

/// Resolve the scope for a query. Reads the trace's process/device set, picks
/// the resolution path per the case analysis above, and (when a single
/// device is locked in) cross-references `TARGET_INFO_CUDA_CONTEXT_INFO`
/// to surface the native_pid that ran on it.
///
/// Returns `Ok(ResolvedScope)` when the query is unambiguous (with or
/// without a flag) — caller assigns `.applied` into the envelope's
/// `meta.applied_scope` and uses the resolved fields in its SQL WHERE
/// clauses.
///
/// Returns `Err(AmbiguityError)` when the request matches more than one
/// process/device scope without all-device aggregation. The verb's
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

    let scopes = cuda_scope_set(trace).map_err(ResolveError::probe)?;
    let matching: HashSet<(i64, i32)> = scopes
        .iter()
        .copied()
        .filter(|(pid, _device)| req.process.is_none_or(|wanted| wanted == *pid))
        .filter(|(_pid, device)| req.device.is_none_or(|wanted| wanted == *device))
        .collect();
    let matching_process_count = matching
        .iter()
        .map(|(pid, _)| *pid)
        .collect::<HashSet<_>>()
        .len();

    let (native_pid, resolved_device, aggregated_over): (Option<i64>, Option<i32>, Vec<String>) =
        if req.all_devices {
            let mut axes = vec!["device".to_string()];
            if req.process.is_none() && matching_process_count > 1 {
                axes.insert(0, "process".to_string());
            }
            (req.process, None, axes)
        } else {
            match matching.len() {
                0 => (req.process, req.device, Vec::new()),
                1 => {
                    let (pid, device) = matching.into_iter().next().expect("one scope");
                    (Some(pid), Some(device), Vec::new())
                }
                _ => {
                    if let Some(err) = stream_parent_error(req.stream, NO_AXES) {
                        return Err(ResolveError::probe(err));
                    }
                    if req.implicit_all_devices {
                        let mut axes = Vec::new();
                        if req.process.is_none() && matching_process_count > 1 {
                            axes.push("process".to_string());
                        }
                        if req.device.is_none() {
                            axes.push("device".to_string());
                        }
                        (req.process, req.device, axes)
                    } else {
                        let candidate_scopes = matching.iter().copied().collect();
                        return Err(ResolveError::Ambiguous(AmbiguityError::multi_device(
                            candidate_scopes,
                            req.process,
                        )));
                    }
                }
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

/// Set of distinct process-local CUDA scopes `(native_pid, deviceId)`.
///
/// Context metadata is authoritative when present. Activity scans are
/// the fallback and use the same process resolver as correlation and
/// sidecar builders. Physical `TARGET_INFO_GPU` rows are validated here
/// but never fabricated into process-local logical scopes.
pub fn cuda_scope_set(trace: &Trace) -> NsysDataResult<HashSet<(i64, i32)>> {
    let mut out = HashSet::new();

    if trace.has_table("TARGET_INFO_CUDA_CONTEXT_INFO") {
        let mut stmt = trace
            .conn()
            .prepare(
                "SELECT DISTINCT CAST(processId AS BIGINT), CAST(deviceId AS INTEGER) \
                 FROM nsight.TARGET_INFO_CUDA_CONTEXT_INFO",
            )
            .map_err(|source| {
                crate::NsysDataError::scope_device_probe_column_missing(
                    "TARGET_INFO_CUDA_CONTEXT_INFO",
                    "deviceId/processId",
                    source,
                )
            })?;
        let mut rows = stmt.query([]).map_err(|source| {
            crate::NsysDataError::scope_device_probe_query("TARGET_INFO_CUDA_CONTEXT_INFO", source)
        })?;
        while let Some(r) = rows.next().map_err(|source| {
            crate::NsysDataError::scope_device_probe_read("TARGET_INFO_CUDA_CONTEXT_INFO", source)
        })? {
            let pid: i64 = r.get(0).map_err(|source| {
                crate::NsysDataError::scope_device_probe_read(
                    "TARGET_INFO_CUDA_CONTEXT_INFO",
                    source,
                )
            })?;
            let device: i32 = r.get(1).map_err(|source| {
                crate::NsysDataError::scope_device_probe_read(
                    "TARGET_INFO_CUDA_CONTEXT_INFO",
                    source,
                )
            })?;
            out.insert((pid, device));
        }
        if !out.is_empty() {
            return Ok(out);
        }
    }

    // Validate and retain the target-info ordinal inventory before
    // scanning activity. It has no owning-process axis, so activity
    // rows remain the preferred source of process-aware scopes; target
    // rows are added afterward only for ordinals with no activity.
    let mut target_devices = HashSet::new();
    if trace.has_table("TARGET_INFO_GPU") {
        let mut stmt = trace
            .conn()
            .prepare("SELECT DISTINCT CAST(cuDevice AS INTEGER) FROM nsight.TARGET_INFO_GPU")
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
        while let Some(row) = rows.next().map_err(|source| {
            crate::NsysDataError::scope_device_probe_read("TARGET_INFO_GPU", source)
        })? {
            if let Some(device) = row.get::<_, Option<i32>>(0).map_err(|source| {
                crate::NsysDataError::scope_device_probe_read("TARGET_INFO_GPU", source)
            })? {
                target_devices.insert(device);
            }
        }
    }

    let resolver = crate::CudaProcessResolver::build(trace)?;
    for table in crate::GPU_WORK_INTERVAL_KINDS
        .iter()
        .map(|kind| kind.table)
        .chain(std::iter::once("CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"))
    {
        if !trace.has_table(table) {
            continue;
        }
        // A partial activity export can omit the location columns while
        // TARGET_INFO_GPU still supplies a usable ordinal inventory.
        // Preserve that fallback instead of failing the whole probe.
        if !trace.table_has_column(table, "deviceId") && !target_devices.is_empty() {
            continue;
        }
        let start_col = if trace.table_has_column(table, "start") {
            "CAST(start AS BIGINT)"
        } else {
            "0::BIGINT"
        };
        let corr_col = if trace.table_has_column(table, "correlationId") {
            "CAST(correlationId AS BIGINT)"
        } else {
            "CAST(NULL AS BIGINT)"
        };
        let global_pid_col = if trace.table_has_column(table, "globalPid") {
            "CAST(globalPid AS BIGINT)"
        } else {
            "CAST(NULL AS BIGINT)"
        };
        let context_col = if trace.table_has_column(table, "contextId") {
            "CAST(contextId AS BIGINT)"
        } else {
            "0::BIGINT"
        };
        let sql = format!(
            "SELECT DISTINCT CAST(deviceId AS INTEGER), {context_col}, \
                    {corr_col}, {start_col}, {global_pid_col} \
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
            let device: i32 = r
                .get(0)
                .map_err(|source| crate::NsysDataError::scope_device_probe_read(table, source))?;
            let context: i64 = r
                .get(1)
                .map_err(|source| crate::NsysDataError::scope_device_probe_read(table, source))?;
            let correlation: Option<i64> = r
                .get(2)
                .map_err(|source| crate::NsysDataError::scope_device_probe_read(table, source))?;
            let start: i64 = r
                .get(3)
                .map_err(|source| crate::NsysDataError::scope_device_probe_read(table, source))?;
            let global_pid: Option<i64> = r
                .get(4)
                .map_err(|source| crate::NsysDataError::scope_device_probe_read(table, source))?;
            let process = resolver.resolve_required(
                table,
                device,
                context,
                correlation,
                start,
                global_pid,
            )?;
            out.insert((process, device));
        }
    }

    Ok(out)
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
        let e =
            AmbiguityError::multi_device(vec![(1001, 0), (2002, 0), (3003, 1), (4004, 1)], None);
        assert_eq!(
            e.warning.code as u8,
            WarningCode::MultiDeviceAmbiguous as u8,
            "code variant must be MultiDeviceAmbiguous"
        );
        assert!(matches!(e.warning.severity, WarningSeverity::Warn));
        assert!(
            e.message.contains("4 CUDA process/device scopes"),
            "message: {}",
            e.message
        );
        assert!(
            e.message.contains("--device") && e.message.contains("--all-devices"),
            "message must mention both flags: {}",
            e.message
        );
        assert_eq!(e.candidate_scopes.first(), Some(&(1001, 0)));
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
                "TARGET_INFO_CUDA_CONTEXT_INFO",
                "CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO (\
                    deviceId BIGINT, contextId BIGINT, processId BIGINT)",
                vec![
                    "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO VALUES (0, 1, 1001)",
                    "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO VALUES (1, 1, 1001)",
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
    fn activity_global_pid_recovers_process_collision_without_context_table() -> Result<()> {
        let pid0 = 1001_i64;
        let (_dir, pqtdir) = parquet_fixture_with_rows(&[
            (
                "TARGET_INFO_GPU",
                "CREATE TABLE TARGET_INFO_GPU (cuDevice BIGINT)",
                vec!["INSERT INTO TARGET_INFO_GPU (cuDevice) VALUES (0)"],
            ),
            (
                "PROCESSES",
                "CREATE TABLE PROCESSES (globalPid BIGINT, pid BIGINT)",
                vec![
                    "INSERT INTO PROCESSES (globalPid, pid) \
                     VALUES (1001::BIGINT * 16777216, 1001)",
                    "INSERT INTO PROCESSES (globalPid, pid) \
                     VALUES (2002::BIGINT * 16777216, 2002)",
                ],
            ),
            (
                "CUPTI_ACTIVITY_KIND_KERNEL",
                "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (\
                    start BIGINT, \"end\" BIGINT, deviceId BIGINT, contextId BIGINT, \
                    correlationId BIGINT, globalPid BIGINT)",
                vec![
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES \
                     (10, 11, 0, 1, 42, 1001::BIGINT * 16777216)",
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES \
                     (20, 21, 0, 1, 42, 2002::BIGINT * 16777216)",
                ],
            ),
        ])?;
        let trace = Trace::open(&pqtdir)?;

        assert!(matches!(
            resolve_scope(
                &trace,
                ScopeRequest {
                    device: Some(0),
                    ..ScopeRequest::default()
                }
            ),
            Err(ResolveError::Ambiguous(_))
        ));
        let exact = resolve_scope(
            &trace,
            ScopeRequest {
                process: Some(pid0),
                device: Some(0),
                ..ScopeRequest::default()
            },
        )?;
        assert_eq!(exact.applied.native_pid, Some(pid0));
        assert_eq!(exact.applied.device, Some(0));
        Ok(())
    }

    #[test]
    fn inactive_physical_device_is_not_fabricated_into_logical_scope() -> Result<()> {
        let pid = 1001_i64;
        let (_dir, pqtdir) = parquet_fixture_with_rows(&[
            (
                "TARGET_INFO_GPU",
                "CREATE TABLE TARGET_INFO_GPU (cuDevice BIGINT)",
                vec![
                    "INSERT INTO TARGET_INFO_GPU (cuDevice) VALUES (0)",
                    "INSERT INTO TARGET_INFO_GPU (cuDevice) VALUES (1)",
                ],
            ),
            (
                "PROCESSES",
                "CREATE TABLE PROCESSES (globalPid BIGINT, pid BIGINT)",
                vec![
                    "INSERT INTO PROCESSES (globalPid, pid) \
                     VALUES (1001::BIGINT * 16777216, 1001)",
                ],
            ),
            (
                "CUPTI_ACTIVITY_KIND_KERNEL",
                "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (\
                    start BIGINT, \"end\" BIGINT, deviceId BIGINT, contextId BIGINT, \
                    correlationId BIGINT, globalPid BIGINT)",
                vec![
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES \
                     (10, 11, 0, 1, 42, 1001::BIGINT * 16777216)",
                ],
            ),
        ])?;
        let trace = Trace::open(&pqtdir)?;
        let scopes = cuda_scope_set(&trace)?;
        assert_eq!(
            scopes,
            HashSet::from([(pid, 0)]),
            "logical scopes require ownership evidence; physical inventory is separate"
        );
        Ok(())
    }

    #[test]
    fn unresolved_activity_process_is_a_typed_error() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture_with_rows(&[(
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (\
                start BIGINT, \"end\" BIGINT, deviceId BIGINT, contextId BIGINT, \
                correlationId BIGINT)",
            vec!["INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (10, 11, 0, 1, 42)"],
        )])?;
        let trace = Trace::open(&pqtdir)?;

        let err = match resolve_scope(&trace, ScopeRequest::default()) {
            Ok(scope) => anyhow::bail!("unresolved CUDA ownership should fail: {scope:?}"),
            Err(err) => downcast_scope_probe_error(err)?,
        };
        assert_eq!(err.code().as_str(), "nsys.data.cuda-process-unresolved");
        assert!(matches!(
            err,
            crate::NsysDataError::CudaProcessUnresolved {
                device_id: 0,
                context_id: 1,
                correlation_id: Some(42),
                ..
            }
        ));
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
