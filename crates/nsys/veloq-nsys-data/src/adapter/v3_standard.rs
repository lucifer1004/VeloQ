//! Adapter for NSys schema 3.x (NSight Systems 2023.1+).
//!
//! Modern NSys exports use canonical column names: `start` / `"end"`
//! on event tables, dedicated `streamId` / `contextId` columns,
//! `correlationId` not `corrId`. Some valid exports contain only
//! metric tables (`nsys profile --trace=none --nic-metrics=lf`, for
//! example), so the probe accepts any known 3.x event or metric table
//! being present in the parquetdir. Query SQL across the workspace
//! reads `nsight.<TABLE>` directly under that assumption; this
//! adapter's job is to *confirm* the canonical shape on open and bail
//! otherwise.
//!
//! The probe runs against parquet footers in the
//! parquetdir, not a SQLite file.

use super::traits::{AdapterMeta, AdapterStatus, SchemaAdapter, table_exists};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::Path;

/// Schema-3.x adapter. Zero-sized; instances live on the trace handle
/// as `Arc<dyn SchemaAdapter>`.
pub struct StandardAdapter;

/// Canonical table/column sets that mark a trace as schema 3.x.
/// Presence of *any* listed table with its canonical columns is enough
/// for `probe` to claim the trace.
const PROBE_TABLES: &[(&str, &[&str])] = &[
    ("CUPTI_ACTIVITY_KIND_KERNEL", &["start", "end"]),
    ("CUPTI_ACTIVITY_KIND_MEMCPY", &["start", "end"]),
    ("CUPTI_ACTIVITY_KIND_MEMSET", &["start", "end"]),
    ("CUPTI_ACTIVITY_KIND_RUNTIME", &["start", "end"]),
    ("CUPTI_ACTIVITY_KIND_SYNCHRONIZATION", &["start", "end"]),
    (
        "NVTX_EVENTS",
        &[
            "start",
            "end",
            "eventType",
            "globalTid",
            "domainId",
            "text",
            "textId",
        ],
    ),
    ("GPU_METRICS", &["timestamp", "typeId", "metricId", "value"]),
    (
        "NET_NIC_METRIC",
        &[
            "start",
            "end",
            "globalId",
            "portId",
            "metricsListId",
            "metricsIdx",
            "value",
        ],
    ),
    (
        "COMPOSITE_EVENTS",
        &["id", "start", "cpu", "threadState", "globalTid"],
    ),
    (
        "SCHED_EVENTS",
        &["start", "cpu", "globalTid", "threadState"],
    ),
];

impl SchemaAdapter for StandardAdapter {
    fn metadata(&self) -> AdapterMeta {
        AdapterMeta {
            id: "v3_standard",
            display_name: "Standard Schema (3.x)",
            match_criteria: "known NSys 3.x event/metric parquet with canonical columns",
            target_versions: "NSight Systems 2023.1+",
            status: AdapterStatus::Stable,
        }
    }

    fn probe(&self, pqtdir: &Path) -> bool {
        PROBE_TABLES
            .iter()
            .any(|(table, columns)| table_has_columns(pqtdir, table, columns))
    }
}

fn table_has_columns(pqtdir: &Path, table: &str, columns: &[&str]) -> bool {
    if !table_exists(pqtdir, table) {
        return false;
    }
    let path = pqtdir.join(format!("{table}.parquet"));
    let Ok(file) = File::open(path) else {
        return false;
    };
    let Ok(builder) = ParquetRecordBatchReaderBuilder::try_new(file) else {
        return false;
    };
    let schema = builder.schema();
    columns.iter().all(|column| schema.index_of(column).is_ok())
}

/// Helper for downstream tests that want to know what the adapter
/// reports without exercising probe (e.g. dispatch tests with a fake
/// connection). Kept here rather than in `tests/` so the static
/// metadata is in one place.
#[cfg(test)]
pub(crate) fn _adapter_id_for_tests() -> &'static str {
    StandardAdapter.metadata().id
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context, Result};
    use duckdb::Connection;

    #[test]
    fn metadata_advertises_stable() {
        let m = StandardAdapter.metadata();
        assert_eq!(m.id, "v3_standard");
        assert_eq!(m.status, AdapterStatus::Stable);
    }

    #[test]
    fn helper_id_is_metadata_id() {
        assert_eq!(_adapter_id_for_tests(), "v3_standard");
    }

    #[test]
    fn probe_signature_compiles() {
        fn _check(a: &dyn SchemaAdapter, p: &Path) -> bool {
            a.probe(p)
        }
    }

    #[test]
    fn probe_accepts_kernel_only_pqtdir() -> Result<()> {
        let dir = tempfile::tempdir()?;
        write_parquet(
            dir.path(),
            "CUPTI_ACTIVITY_KIND_KERNEL",
            r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start BIGINT, "end" BIGINT);"#,
        )?;
        assert!(StandardAdapter.probe(dir.path()));
        Ok(())
    }

    #[test]
    fn probe_accepts_metric_only_pqtdir() -> Result<()> {
        let dir = tempfile::tempdir()?;
        write_parquet(
            dir.path(),
            "GPU_METRICS",
            "CREATE TABLE GPU_METRICS (timestamp BIGINT, typeId BIGINT, metricId BIGINT, value DOUBLE);",
        )?;
        assert!(StandardAdapter.probe(dir.path()));
        Ok(())
    }

    #[test]
    fn probe_accepts_nvtx_only_pqtdir() -> Result<()> {
        let dir = tempfile::tempdir()?;
        write_parquet(
            dir.path(),
            "NVTX_EVENTS",
            r#"CREATE TABLE NVTX_EVENTS (
                start BIGINT, "end" BIGINT, eventType BIGINT,
                globalTid BIGINT, domainId BIGINT, text TEXT, textId BIGINT
            );"#,
        )?;
        assert!(StandardAdapter.probe(dir.path()));
        Ok(())
    }

    #[test]
    fn probe_rejects_nvtx_table_missing_canonical_columns() -> Result<()> {
        let dir = tempfile::tempdir()?;
        write_parquet(
            dir.path(),
            "NVTX_EVENTS",
            r#"CREATE TABLE NVTX_EVENTS (
                start BIGINT, "end" BIGINT, globalTid BIGINT,
                domainId BIGINT, text TEXT, textId BIGINT
            );"#,
        )?;
        assert!(!StandardAdapter.probe(dir.path()));
        Ok(())
    }

    #[test]
    fn probe_rejects_empty_pqtdir() -> Result<()> {
        let dir = tempfile::tempdir()?;
        assert!(!StandardAdapter.probe(dir.path()));
        Ok(())
    }

    #[test]
    fn probe_rejects_table_with_legacy_columns() -> Result<()> {
        let dir = tempfile::tempdir()?;
        write_parquet(
            dir.path(),
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (timestamp BIGINT, endTimestamp BIGINT);",
        )?;
        assert!(!StandardAdapter.probe(dir.path()));
        Ok(())
    }

    fn write_parquet(pqtdir: &Path, table: &str, ddl: &str) -> Result<()> {
        let conn = Connection::open_in_memory().context("open in-memory DuckDB")?;
        conn.execute_batch(ddl)?;
        let out = pqtdir.join(format!("{table}.parquet"));
        let out_lit = out.to_string_lossy().replace('\'', "''");
        conn.execute(
            &format!(r#"COPY (SELECT * FROM "{table}") TO '{out_lit}' (FORMAT PARQUET)"#),
            [],
        )
        .with_context(|| format!("copy {table} to parquet"))?;
        Ok(())
    }
}
