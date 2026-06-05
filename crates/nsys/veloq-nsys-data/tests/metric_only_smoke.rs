//! End-to-end coverage for two related metric-only behaviours:
//!
//! 1. [`Trace::read_origins`] anchors `--from`/`--to` on the first
//!    metric sample instead of absolute zero when no event table has
//!    any rows. Without this, captures from
//!    `nsys profile --trace=none --nic-metrics=lf` would treat
//!    `--from 0` as t=0 ns rather than first-sample time, off by
//!    hundreds of seconds on real captures.
//! 2. [`CapabilityFlags::has_nic_metrics`] only flips true when every
//!    table `metrics --type nic` requires is present (`NET_NIC_METRIC`,
//!    `TARGET_INFO_NETWORK_METRICS`, `NIC_ID_MAP`, and
//!    `TARGET_INFO_NIC_INFO`) — so `summary` can't promise a query that
//!    would then bail on a missing dictionary or id-map.
//!
//! Same fixture shape as `hardware_smoke.rs`: a DuckDB-built
//! parquetdir opened through `Trace::open` so adapter dispatch runs.

use anyhow::{Context, Result};
use duckdb::{Connection, params};
use std::path::{Path, PathBuf};
use tempfile::TempDir;
use veloq_nsys_data::{CapabilityFlags, Trace};

struct Fixture {
    path: PathBuf,
    _dir: TempDir,
}

impl Fixture {
    fn path(&self) -> &Path {
        &self.path
    }
}

fn finalize_to_pqtdir(conn: &Connection, dir: TempDir) -> Result<Fixture> {
    let pqtdir = dir.path().join("test_pqtdir");
    std::fs::create_dir_all(&pqtdir).context("create parquetdir")?;
    let mut stmt = conn.prepare(
        "SELECT table_name FROM information_schema.tables \
         WHERE table_schema = 'main' AND table_type = 'BASE TABLE'",
    )?;
    let mut rows = stmt.query([])?;
    let mut tables = Vec::new();
    while let Some(r) = rows.next()? {
        tables.push(r.get::<_, String>(0)?);
    }
    for table in &tables {
        let out = pqtdir.join(format!("{table}.parquet"));
        let out_lit = out.to_string_lossy().replace('\'', "''");
        conn.execute(
            &format!(r#"COPY (SELECT * FROM "{table}") TO '{out_lit}' (FORMAT PARQUET)"#),
            [],
        )?;
    }
    Ok(Fixture {
        path: pqtdir,
        _dir: dir,
    })
}

/// Schema 3.x meta-block — picked up by `StandardAdapter::probe`'s
/// version-match path so `Trace::open` succeeds on metric-only files.
fn write_v3_meta(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
        CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT);
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        "#,
    )
    .context("create meta tables")?;
    for (k, v) in [
        ("EXPORT_SCHEMA_VERSION_MAJOR", "3"),
        ("EXPORT_SCHEMA_VERSION_MINOR", "22"),
        ("EXPORT_SCHEMA_VERSION_MICRO", "1"),
        ("EXPORT_PRODUCT_VERSION", "2025.4.1.136"),
    ] {
        conn.execute(
            "INSERT INTO META_DATA_EXPORT (name, value) VALUES (?, ?)",
            params![k, v],
        )?;
    }
    Ok(())
}

/// Metric-only NIC capture (`nsys profile --trace=none
/// --nic-metrics=lf`): no event tables, only `NET_NIC_METRIC` plus
/// the dictionary / id-map / nic-info tables `metrics --type nic`
/// requires. First sample lands at 1.2 s so the anchor test is
/// distinguishable from the broken behaviour (t=0).
fn nic_metric_only() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    write_v3_meta(&conn)?;
    conn.execute_batch(
        r#"
        CREATE TABLE NET_NIC_METRIC (
            start BIGINT, "end" BIGINT,
            globalId BIGINT, portId BIGINT,
            metricsListId BIGINT, metricsIdx BIGINT,
            value REAL
        );
        CREATE TABLE TARGET_INFO_NETWORK_METRICS (
            metricsListId BIGINT, metricsIdx BIGINT,
            name TEXT, description TEXT, unit TEXT
        );
        CREATE TABLE NIC_ID_MAP (
            globalId BIGINT, nicId BIGINT
        );
        CREATE TABLE TARGET_INFO_NIC_INFO (
            GUID BIGINT, nicId BIGINT, name TEXT,
            deviceId BIGINT, vendorId BIGINT
        );
        "#,
    )
    .context("nic metric schema")?;
    for (start, end_ns) in [
        (1_200_000_000_i64, 1_201_000_000_i64),
        (1_300_000_000_i64, 1_301_000_000_i64),
        (1_500_000_000_i64, 1_501_000_000_i64),
    ] {
        conn.execute(
            "INSERT INTO NET_NIC_METRIC \
             (start, \"end\", globalId, portId, metricsListId, metricsIdx, value) \
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![start, end_ns, 1i64, 1i64, 1i64, 0i64, 1.0_f64],
        )?;
    }
    finalize_to_pqtdir(&conn, dir)
}

/// Same shape as [`nic_metric_only`] but missing the dictionary
/// table — used to confirm `has_nic_metrics` is the AND, not just
/// `NET_NIC_METRIC` table-presence.
fn nic_metric_partial() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    write_v3_meta(&conn)?;
    conn.execute_batch(
        r#"
        CREATE TABLE NET_NIC_METRIC (
            start BIGINT, "end" BIGINT,
            globalId BIGINT, portId BIGINT,
            metricsListId BIGINT, metricsIdx BIGINT,
            value REAL
        );
        CREATE TABLE NIC_ID_MAP (
            globalId BIGINT, nicId BIGINT
        );
        CREATE TABLE TARGET_INFO_NIC_INFO (
            GUID BIGINT, nicId BIGINT, name TEXT,
            deviceId BIGINT, vendorId BIGINT
        );
        "#,
    )
    .context("partial nic metric schema")?;
    conn.execute(
        "INSERT INTO NET_NIC_METRIC \
         (start, \"end\", globalId, portId, metricsListId, metricsIdx, value) \
         VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![
            1_200_000_000_i64,
            1_201_000_000_i64,
            1i64,
            1i64,
            1i64,
            0i64,
            1.0_f64
        ],
    )?;
    finalize_to_pqtdir(&conn, dir)
}

/// GPU metric-only capture (`nsys profile --trace=none --gpu-metrics-…`):
/// only `GPU_METRICS` (with `timestamp`, not `start`/`"end"`) plus the
/// counter dictionary `TARGET_INFO_GPU_METRICS`. Samples start at
/// 2.0 s so the origin fallback test is independent of the NIC fixture.
fn gpu_metric_only() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    write_v3_meta(&conn)?;
    conn.execute_batch(
        r#"
        CREATE TABLE GPU_METRICS (
            timestamp BIGINT, typeId BIGINT, metricId BIGINT, value REAL
        );
        CREATE TABLE TARGET_INFO_GPU_METRICS (
            typeId BIGINT, metricId BIGINT, metricName TEXT,
            description TEXT
        );
        "#,
    )
    .context("gpu metric schema")?;
    for ts in [2_000_000_000_i64, 2_100_000_000_i64, 2_500_000_000_i64] {
        conn.execute(
            "INSERT INTO GPU_METRICS (timestamp, typeId, metricId, value) VALUES (?, ?, ?, ?)",
            params![ts, 1i64, 7i64, 0.5_f64],
        )?;
    }
    finalize_to_pqtdir(&conn, dir)
}

#[test]
fn read_origins_anchors_on_first_nic_sample_when_no_event_tables() -> Result<()> {
    let fx = nic_metric_only()?;
    let trace = Trace::open(fx.path())?;
    let (origins, per_table) = trace.read_origins()?;

    // Event-table per_table stays empty — the fallback fires through
    // sample tables, not by polluting per_table on normal traces.
    assert!(
        per_table.is_empty(),
        "metric-only trace should not surface event-table per_table entries; got {per_table:?}"
    );
    assert_eq!(origins.primary.start_ns, 1_200_000_000);
    assert_eq!(origins.primary.end_ns, 1_501_000_000);
    assert_eq!(origins.full.start_ns, 1_200_000_000);
    assert_eq!(origins.full.end_ns, 1_501_000_000);
    Ok(())
}

#[test]
fn read_origins_handles_gpu_metric_only_timestamp_column() -> Result<()> {
    // Regression: `GPU_METRICS` uses `timestamp`, not `start`/`"end"`.
    // The fallback's per-spec min/max columns must respect that.
    let fx = gpu_metric_only()?;
    let trace = Trace::open(fx.path())?;
    let (origins, _per_table) = trace.read_origins()?;
    assert_eq!(origins.primary.start_ns, 2_000_000_000);
    assert_eq!(origins.primary.end_ns, 2_500_000_000);
    Ok(())
}

#[test]
fn has_nic_metrics_requires_full_table_set() -> Result<()> {
    let full = nic_metric_only()?;
    let partial = nic_metric_partial()?;

    let full_caps = CapabilityFlags::extract(full.path());
    let partial_caps = CapabilityFlags::extract(partial.path());

    assert!(
        full_caps.has_nic_metrics,
        "NIC fixture has every required table; has_nic_metrics should be true"
    );
    assert!(
        !partial_caps.has_nic_metrics,
        "partial NIC fixture lacks TARGET_INFO_NETWORK_METRICS; \
         has_nic_metrics must be false so summary can't false-promise the query"
    );
    Ok(())
}

#[test]
fn has_gpu_metrics_requires_counter_dictionary() -> Result<()> {
    // Full GPU set → true.
    let full = gpu_metric_only()?;
    let full_caps = CapabilityFlags::extract(full.path());
    assert!(full_caps.has_gpu_metrics);

    // Drop the dictionary → false. Build a separate fixture rather
    // than mutating to keep the original `full` fixture untouched.
    let dir = tempfile::tempdir().context("tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    write_v3_meta(&conn)?;
    conn.execute_batch(
        r#"CREATE TABLE GPU_METRICS (
            timestamp BIGINT, typeId BIGINT, metricId BIGINT, value REAL
        );"#,
    )
    .context("gpu metric schema (no dict)")?;
    let fixture = finalize_to_pqtdir(&conn, dir)?;
    let caps = CapabilityFlags::extract(fixture.path());
    assert!(
        !caps.has_gpu_metrics,
        "GPU_METRICS without TARGET_INFO_GPU_METRICS dictionary should not flip has_gpu_metrics"
    );
    Ok(())
}
