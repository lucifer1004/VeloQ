//! End-to-end test for hardware topology extraction.
//!
//! Builds a synthetic NSys parquetdir with the `TARGET_INFO_*` tables
//! populated, opens it through `Trace::open` (so adapter dispatch
//! runs), and asserts the extracted topology surfaces the expected
//! host / CPU / GPU / NIC values.

use anyhow::{Context, Result};
use duckdb::{Connection, params};
use std::path::{Path, PathBuf};
use tempfile::TempDir;
use veloq_nsys_data::{CapabilityFlags, Trace, hardware};

/// Owns the tempdir so the parquetdir outlives the Trace::open call.
struct Fixture {
    path: PathBuf,
    _dir: TempDir,
}

fn finalize_to_pqtdir(conn: &Connection, dir: TempDir) -> Result<Fixture> {
    let pqtdir = dir.path().join("test_pqtdir");
    std::fs::create_dir_all(&pqtdir).context("create parquetdir")?;
    let mut stmt = conn.prepare(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' AND table_type = 'BASE TABLE'",
    )?;
    let mut rows = stmt.query([])?;
    let mut tables = Vec::new();
    while let Some(r) = rows.next()? {
        tables.push(r.get::<_, String>(0)?);
    }
    for table in &tables {
        let out = pqtdir.join(format!("{table}.parquet"));
        let out_lit = out.to_string_lossy().replace('\x27', "''");
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

impl Fixture {
    fn path(&self) -> &Path {
        &self.path
    }
}

/// Single-host trace with one CPU, one H100-class GPU, and one
/// Mellanox NIC. `META_DATA_EXPORT` carries schema 3.22.1 so
/// adapter dispatch picks `StandardAdapter` and reads the real
/// `start`/`"end"` columns on event tables.
fn with_target_info() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;

    conn.execute_batch(
        r#"
        CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT);
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start BIGINT, "end" BIGINT,
            deviceId BIGINT, contextId BIGINT, streamId BIGINT,
            shortName BIGINT, demangledName BIGINT,
            gridX BIGINT, gridY BIGINT, gridZ BIGINT,
            blockX BIGINT, blockY BIGINT, blockZ BIGINT,
            correlationId BIGINT,
            registersPerThread BIGINT,
            staticSharedMemory BIGINT,
            dynamicSharedMemory BIGINT,
            globalPid BIGINT
        );
        CREATE TABLE TARGET_INFO_SYSTEM_ENV (
            globalVid UBIGINT, name TEXT, value TEXT
        );
        CREATE TABLE TARGET_INFO_GPU (
            vmId UBIGINT, id BIGINT, cuDevice BIGINT, name TEXT, uuid TEXT,
            chipName TEXT, totalMemory UBIGINT, smCount BIGINT,
            computeMajor BIGINT, computeMinor BIGINT, busLocation TEXT
        );
        CREATE TABLE TARGET_INFO_NIC_INFO (
            GUID UBIGINT, nicId BIGINT, name TEXT,
            deviceId BIGINT, vendorId BIGINT
        );
        "#,
    )
    .context("schema")?;

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
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
         (start, \"end\", deviceId, contextId, streamId, shortName, demangledName, \
          gridX, gridY, gridZ, blockX, blockY, blockZ, correlationId, \
          registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            100_000_000i64,
            110_000_000i64,
            0i32,
            0i64,
            7i64,
            None::<i64>,
            None::<i64>,
            1i64,
            1i64,
            1i64,
            128i64,
            1i64,
            1i64,
            42i64,
            32i64,
            0i64,
            0i64,
            0i64,
        ],
    )?;

    // High-bit vm/global ids exercise NSys parquetdir's UBIGINT
    // boundary; reading these through i64 would fail.
    let global_vid: u64 = (0xE0A5_u64 << 48) | 0x1234_u64;
    let global_vid_text = global_vid.to_string();
    for (name, value) in [
        ("Hostname", "test-host"),
        ("OsDescription", "Ubuntu 22.04.3 LTS"),
        ("KernelVersion", "6.5.0-test"),
        ("HardwarePlatform", "x86_64"),
        ("SoftwarePlatform", "L4X"),
        ("SoftwareReleaseVersion", "6.5.0-test-generic"),
        ("CpuDescription", "AMD EPYC 7763 64-Core Processor"),
        ("CpuArchitecture", "x86_64"),
        ("CpuCores", "128"),
        ("CpuSpeedMhz", "2450"),
        ("CudaDriverVersion", "13000"),
        ("NvDriverVersion", "580.95.05"),
    ] {
        conn.execute(
            "INSERT INTO TARGET_INFO_SYSTEM_ENV (globalVid, name, value) \
             VALUES (CAST(? AS UBIGINT), ?, ?)",
            params![global_vid_text.as_str(), name, value],
        )?;
    }

    // One H100 attached to the same host.
    conn.execute(
        "INSERT INTO TARGET_INFO_GPU \
         (vmId, id, cuDevice, name, uuid, chipName, totalMemory, smCount, \
          computeMajor, computeMinor, busLocation) \
         VALUES (CAST(? AS UBIGINT), ?, ?, ?, ?, ?, CAST(? AS UBIGINT), ?, ?, ?, ?)",
        params![
            global_vid_text.as_str(),
            99i64,
            0i64,
            "NVIDIA H100 80GB HBM3",
            "72aa5c49-8c76-a7a9-2124-ab9a2ff11ab1",
            "GH100",
            "85017624576",
            132i64,
            9i64,
            0i64,
            "0000:45:00.0",
        ],
    )?;

    // One Mellanox NIC.
    conn.execute(
        "INSERT INTO TARGET_INFO_NIC_INFO \
         (GUID, nicId, name, deviceId, vendorId) \
         VALUES (CAST(? AS UBIGINT), ?, ?, ?, ?)",
        params![global_vid_text.as_str(), 0i64, "mlx5_0", 4131i64, 5555i64],
    )?;

    finalize_to_pqtdir(&conn, dir)
}

#[test]
fn extracts_full_topology_for_single_host() -> Result<()> {
    let fixture = with_target_info()?;
    let trace = Trace::open(fixture.path())?;
    let hosts = hardware::extract(&trace)?;

    assert_eq!(hosts.len(), 1, "single-host fixture expected");
    let host = hosts.first().context("first host")?;
    assert_eq!(host.hw_host_id, 0xE0A5);
    assert_eq!(host.vm_id, (0xE0A5_u64 << 48) | 0x1234_u64);

    let system = host.system.as_ref().context("system info present")?;
    assert_eq!(system.hostname.as_deref(), Some("test-host"));
    assert_eq!(system.os_description.as_deref(), Some("Ubuntu 22.04.3 LTS"));
    assert_eq!(system.hardware_platform.as_deref(), Some("x86_64"));

    let cpu = host.cpu.as_ref().context("cpu info present")?;
    assert_eq!(cpu.model, "AMD EPYC 7763 64-Core Processor");
    assert_eq!(cpu.core_count, Some(128));
    assert_eq!(cpu.clock_mhz, Some(2450));

    let drivers = host.drivers.as_ref().context("drivers present")?;
    assert_eq!(drivers.cuda_driver_version.as_deref(), Some("13000"));
    // Parsed form decodes the int packing — `13000` = `13.0`.
    assert_eq!(drivers.cuda_version_parsed().as_deref(), Some("13.0"));

    assert_eq!(host.gpus.len(), 1, "one GPU on this host");
    let gpu = host.gpus.first().context("first GPU")?;
    assert_eq!(gpu.id, 99);
    assert_eq!(gpu.name, "NVIDIA H100 80GB HBM3");
    assert_eq!(gpu.chip_name.as_deref(), Some("GH100"));
    assert_eq!(gpu.sm_count, Some(132));
    assert_eq!(gpu.compute_major, Some(9));
    assert_eq!(gpu.compute_minor, Some(0));

    assert_eq!(host.nics.len(), 1, "one NIC on single-host trace");
    let nic = host.nics.first().context("first NIC")?;
    assert_eq!(nic.name, "mlx5_0");
    assert_eq!(nic.vendor_id, Some(5555));

    Ok(())
}

#[test]
fn empty_vec_when_target_info_absent() -> Result<()> {
    // Build a fixture without TARGET_INFO_SYSTEM_ENV. `Trace::open`
    // should still succeed (adapter dispatch picks v3_standard
    // because the kernel table has start/"end" columns), but
    // `extract` returns an empty Vec rather than failing.
    let dir = tempfile::tempdir().context("tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    conn.execute_batch(
        r#"
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start BIGINT, "end" BIGINT,
            deviceId BIGINT, contextId BIGINT, streamId BIGINT,
            shortName BIGINT, demangledName BIGINT,
            gridX BIGINT, gridY BIGINT, gridZ BIGINT,
            blockX BIGINT, blockY BIGINT, blockZ BIGINT,
            correlationId BIGINT,
            registersPerThread BIGINT,
            staticSharedMemory BIGINT,
            dynamicSharedMemory BIGINT,
            globalPid BIGINT
        );
        "#,
    )?;
    let fixture = finalize_to_pqtdir(&conn, dir)?;

    let trace = Trace::open(fixture.path())?;
    let hosts = hardware::extract(&trace)?;
    assert!(hosts.is_empty(), "no SYSTEM_ENV → empty topology");
    Ok(())
}

#[test]
fn capabilities_reflect_table_presence() -> Result<()> {
    let fixture = with_target_info()?;
    let trace = Trace::open(fixture.path())?;
    let caps = CapabilityFlags::extract(trace.path());

    // The fixture only has the kernel event table — capability flags
    // for the other event tables should all be false.
    assert!(caps.has_kernels);
    assert!(!caps.has_memcpy);
    assert!(!caps.has_memset);
    assert!(!caps.has_sync);
    assert!(!caps.has_runtime);
    assert!(!caps.has_osrt);
    assert!(!caps.has_nvtx);
    assert!(!caps.has_cuda_contexts);
    assert!(!caps.has_sampling);
    assert!(!caps.has_gpu_metrics);
    assert!(!caps.has_nic_metrics);

    // But TARGET_INFO_SYSTEM_ENV is present — hardware queries will
    // return data.
    assert!(caps.has_target_info);

    // The convenience predicate composes from the four GPU kinds.
    assert!(caps.any_gpu_events());
    Ok(())
}
