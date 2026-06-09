//! Normalized GPU work event sidecar for gaps-style interval queries.
//!
//! `<trace>.veloq/gpu-work-events.parquet` contains the minimal
//! kernel / memcpy / memset / graph-trace interval surface shared by
//! gap planning: event kind, source row id, device, stream, start, and
//! end. It does not store display names; query crates hydrate names
//! from the source tables after applying LIMIT so name semantics stay
//! centralized.
//!
//! Freshness and atomic publish follow [[RFC-0005]] via
//! [`crate::sidecar`]. The sidecar is registered as
//! `nsight.gpu_work_events` when fresh on `Trace::open`, and
//! [`ensure_sidecar`] registers the view after a build.

mod parquet;

use crate::{NsysDataResult, Trace};
use parquet::{read_parquet, sidecar_is_fresh, write_parquet};
use std::fs;
use std::path::{Path, PathBuf};
use veloq_core::SourceFingerprint;

/// Bump on every breaking schema change to the parquet sidecar.
/// Mismatched versions rebuild silently on next use.
pub const GPU_WORK_EVENTS_VERSION: u32 = 1;

/// One duration-bearing GPU work interval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuWorkEventRecord {
    pub kind: String,
    pub row_id: i64,
    pub device_id: i32,
    pub stream_id: i64,
    pub start_ns: i64,
    pub end_ns: i64,
}

struct GpuWorkKind {
    label: &'static str,
    table: &'static str,
}

const GPU_WORK_KINDS: &[GpuWorkKind] = &[
    GpuWorkKind {
        label: "kernel",
        table: "CUPTI_ACTIVITY_KIND_KERNEL",
    },
    GpuWorkKind {
        label: "memcpy",
        table: "CUPTI_ACTIVITY_KIND_MEMCPY",
    },
    GpuWorkKind {
        label: "memset",
        table: "CUPTI_ACTIVITY_KIND_MEMSET",
    },
    GpuWorkKind {
        label: "graph",
        table: "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
    },
];

/// Filesystem path of the sidecar parquet under `<trace>.veloq/`.
pub fn sidecar_path_for(trace_path: &Path) -> PathBuf {
    veloq_core::artifact_dir_for(trace_path).join("gpu-work-events.parquet")
}

fn source_fingerprint(trace_path: &Path) -> NsysDataResult<SourceFingerprint> {
    crate::trace_artifact_fingerprint(trace_path).map_err(|source| {
        crate::NsysDataError::gpu_work_events_trace_fingerprint(trace_path.display(), source)
    })
}

pub(crate) fn sidecar_is_fresh_for_trace(trace_path: &Path) -> NsysDataResult<bool> {
    let path = sidecar_path_for(trace_path);
    let fp = source_fingerprint(trace_path)?;
    sidecar_is_fresh(&path, fp)
}

pub(crate) fn format_version_on_disk(path: &Path) -> Option<u32> {
    parquet::format_version_on_disk(path)
}

/// Build the sidecar if missing or stale; return its path and register
/// `nsight.gpu_work_events` on this trace handle.
pub fn ensure_sidecar(trace: &Trace) -> NsysDataResult<PathBuf> {
    Ok(ensure_sidecar_state(trace)?.path)
}

fn ensure_sidecar_state(
    trace: &Trace,
) -> NsysDataResult<crate::sidecar::FreshSidecar<Vec<GpuWorkEventRecord>>> {
    let path = sidecar_path_for(trace.path());
    let fp = source_fingerprint(trace.path())?;
    let state = crate::sidecar::ensure_fresh_sidecar::<Vec<GpuWorkEventRecord>>(
        path,
        fp,
        sidecar_is_fresh,
        || compute(trace),
        |path, fp, records| write_parquet(path, fp, records),
    )?;
    if let Some(records) = &state.rebuilt_records {
        log::info!(
            "gpu_work_events: built sidecar at {} ({} events)",
            state.path.display(),
            records.len()
        );
    } else {
        log::debug!(
            "gpu_work_events: warm sidecar at {} ({} bytes)",
            state.path.display(),
            fs::metadata(&state.path).map(|m| m.len()).unwrap_or(0),
        );
    }
    attach_view(trace, &state.path)?;
    Ok(state)
}

/// Load only if a fresh sidecar exists on disk; never trigger a build.
pub fn load_if_present(trace: &Trace) -> NsysDataResult<Option<Vec<GpuWorkEventRecord>>> {
    let path = sidecar_path_for(trace.path());
    let fp = source_fingerprint(trace.path())?;
    crate::sidecar::load_if_fresh(&path, fp, sidecar_is_fresh, read_parquet)
}

/// Does this trace handle currently expose `nsight.gpu_work_events`?
pub fn view_available(trace: &Trace) -> bool {
    trace
        .conn()
        .execute("SELECT 1 FROM nsight.gpu_work_events LIMIT 0", [])
        .is_ok()
}

pub(crate) fn attach_view_if_present(
    conn: &duckdb::Connection,
    source_path: &Path,
) -> NsysDataResult<()> {
    let sidecar = sidecar_path_for(source_path);
    if !sidecar.exists() || !sidecar_is_fresh_for_trace(source_path)? {
        return Ok(());
    }
    attach_view_path(conn, &sidecar)
}

fn attach_view(trace: &Trace, sidecar: &Path) -> NsysDataResult<()> {
    attach_view_path(trace.conn(), sidecar)
}

fn attach_view_path(conn: &duckdb::Connection, sidecar: &Path) -> NsysDataResult<()> {
    let Some(sql) = view_sql_for(sidecar) else {
        log::warn!(
            "gpu_work_events: sidecar path is not valid UTF-8, skipping view registration: {}",
            sidecar.display(),
        );
        return Ok(());
    };
    conn.execute(&sql, []).map_err(|source| {
        crate::NsysDataError::gpu_work_events_view_register(sidecar.display(), source)
    })?;
    Ok(())
}

/// SQL fragment that registers a view named `nsight.gpu_work_events`
/// over the sidecar parquet.
pub fn view_sql_for(sidecar_path: &Path) -> Option<String> {
    let lit = sidecar_path.to_str()?.replace('\'', "''");
    Some(format!(
        "CREATE OR REPLACE VIEW nsight.gpu_work_events AS \
         SELECT (file_row_number + 1) AS rowid, * \
         FROM read_parquet('{lit}', file_row_number = true)"
    ))
}

fn compute(trace: &Trace) -> NsysDataResult<Vec<GpuWorkEventRecord>> {
    let mut records = Vec::new();
    for kind in GPU_WORK_KINDS {
        collect_kind(trace, kind, &mut records)?;
    }
    records.sort_by(|a, b| {
        a.start_ns
            .cmp(&b.start_ns)
            .then(a.device_id.cmp(&b.device_id))
            .then(a.stream_id.cmp(&b.stream_id))
            .then(a.kind.cmp(&b.kind))
            .then(a.row_id.cmp(&b.row_id))
    });
    Ok(records)
}

fn collect_kind(
    trace: &Trace,
    kind: &GpuWorkKind,
    out: &mut Vec<GpuWorkEventRecord>,
) -> NsysDataResult<()> {
    if !trace.has_table(kind.table) {
        return Ok(());
    }
    let table = crate::quote_sql_identifier(kind.table);
    let sql = format!(
        r#"
        SELECT
            t.rowid AS row_id,
            CAST(t.deviceId AS INTEGER) AS device_id,
            CAST(COALESCE(t.streamId, 0) AS BIGINT) AS stream_id,
            t.start AS start_ns,
            t."end" AS end_ns
        FROM nsight.{table} t
        ORDER BY start_ns, row_id
        "#
    );
    let mut stmt = trace
        .conn()
        .prepare(&sql)
        .map_err(|source| crate::NsysDataError::gpu_work_events_rows_prepare(kind.table, source))?;
    let mut rows = stmt
        .query([])
        .map_err(|source| crate::NsysDataError::gpu_work_events_rows_query(kind.table, source))?;
    while let Some(row) = rows
        .next()
        .map_err(|source| crate::NsysDataError::gpu_work_events_rows_read(kind.table, source))?
    {
        out.push(GpuWorkEventRecord {
            kind: kind.label.to_string(),
            row_id: row.get(0).map_err(|source| {
                crate::NsysDataError::gpu_work_events_rows_read(kind.table, source)
            })?,
            device_id: row.get(1).map_err(|source| {
                crate::NsysDataError::gpu_work_events_rows_read(kind.table, source)
            })?,
            stream_id: row.get(2).map_err(|source| {
                crate::NsysDataError::gpu_work_events_rows_read(kind.table, source)
            })?,
            start_ns: row.get(3).map_err(|source| {
                crate::NsysDataError::gpu_work_events_rows_read(kind.table, source)
            })?,
            end_ns: row.get(4).map_err(|source| {
                crate::NsysDataError::gpu_work_events_rows_read(kind.table, source)
            })?,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::parquet_fixture_with_rows;
    use anyhow::Result;
    use veloq_core::VeloqDiagnostic;

    fn gpu_fixture() -> Result<(tempfile::TempDir, PathBuf)> {
        parquet_fixture_with_rows(&[
            (
                "CUPTI_ACTIVITY_KIND_KERNEL",
                r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                    start BIGINT, "end" BIGINT, deviceId BIGINT, streamId BIGINT
                )"#,
                vec![
                    r#"INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL
                       (start, "end", deviceId, streamId)
                       VALUES (100, 110, 0, 7)"#,
                ],
            ),
            (
                "CUPTI_ACTIVITY_KIND_MEMCPY",
                r#"CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
                    start BIGINT, "end" BIGINT, deviceId BIGINT, streamId BIGINT
                )"#,
                vec![
                    r#"INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY
                       (start, "end", deviceId, streamId)
                       VALUES (120, 130, 0, NULL)"#,
                ],
            ),
            (
                "CUPTI_ACTIVITY_KIND_MEMSET",
                r#"CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (
                    start BIGINT, "end" BIGINT, deviceId BIGINT, streamId BIGINT
                )"#,
                vec![
                    r#"INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET
                       (start, "end", deviceId, streamId)
                       VALUES (90, 95, 1, 9)"#,
                ],
            ),
            (
                "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
                r#"CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (
                    start BIGINT, "end" BIGINT, deviceId BIGINT, streamId BIGINT
                )"#,
                vec![
                    r#"INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE
                       (start, "end", deviceId, streamId)
                       VALUES (140, 150, 0, 23)"#,
                ],
            ),
        ])
    }

    #[test]
    fn ensure_sidecar_builds_loads_and_attaches_view() -> Result<()> {
        let (_dir, pqtdir) = gpu_fixture()?;
        let trace = Trace::open(&pqtdir)?;
        assert!(!view_available(&trace));

        let path = ensure_sidecar(&trace)?;
        assert!(
            path.exists(),
            "sidecar path should exist: {}",
            path.display()
        );
        assert!(view_available(&trace));

        let records =
            load_if_present(&trace)?.ok_or_else(|| anyhow::anyhow!("fresh sidecar should load"))?;
        assert_eq!(records.len(), 4, "records: {records:?}");
        let memcpy = records
            .iter()
            .find(|r| r.kind == "memcpy")
            .ok_or_else(|| anyhow::anyhow!("missing memcpy record: {records:?}"))?;
        assert_eq!(memcpy.stream_id, 0);
        let graph = records
            .iter()
            .find(|r| r.kind == "graph")
            .ok_or_else(|| anyhow::anyhow!("missing graph record: {records:?}"))?;
        assert_eq!(graph.device_id, 0);
        assert_eq!(graph.stream_id, 23);

        let reopened = Trace::open(&pqtdir)?;
        assert!(view_available(&reopened));
        let count: i64 = reopened.conn().query_row(
            "SELECT COUNT(*) FROM nsight.gpu_work_events",
            [],
            |row| row.get(0),
        )?;
        assert_eq!(count, 4);
        Ok(())
    }

    #[test]
    fn source_fingerprint_missing_trace_error_is_typed() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let path = tmp.path().join("missing_pqtdir");

        let err = match source_fingerprint(&path) {
            Ok(fp) => anyhow::bail!("missing trace should not fingerprint: {fp:?}"),
            Err(err) => err,
        };

        assert_eq!(
            err.code().as_str(),
            "nsys.data.gpu-work-events-trace-fingerprint"
        );
        Ok(())
    }
}
