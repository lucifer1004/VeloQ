//! Hardware topology — CPU / GPU / NIC inventory of the profiled host(s).
//!
//! NSys writes per-host system info into `TARGET_INFO_SYSTEM_ENV`
//! (key-value rows), `TARGET_INFO_GPU` (one row per CUDA device),
//! and `TARGET_INFO_NIC_INFO` (network interfaces, mostly InfiniBand
//! NICs). veloq exposes the pivoted shape so agents asking "what
//! hardware ran this" get one structured response per host instead
//! of having to issue three separate `inspect`-style queries.
//!
//! Hardware here is *just* hardware — capture-time capability flags
//! (`Supports*` / `Has*` keys in `TARGET_INFO_SYSTEM_ENV`) collapse
//! into the curated module-level [`crate::CapabilityFlags`] and stay
//! out of [`HostInfo`]. Querying through explicit `match` arms (no
//! implicit `if let Ok && let Ok` chains) satisfies the workspace's
//! no-panic lints.

use crate::{NsysDataResult, Trace};
use serde::{Deserialize, Serialize};

/// One profiled host (single-node traces have exactly one).
///
/// **Caveat on field-skip annotations:** `HostInfo` is persisted to
/// the bincode-encoded metadata cache. bincode is positional
/// — it doesn't tag fields by name, so `#[serde(skip_serializing_if
/// = "…")]` *cannot* be used here: the encoder would skip bytes the
/// decoder still expects to read, misaligning every following field.
/// Optional fields therefore round-trip as explicit `null` (Option
/// = None) / `[]` (Vec::new) in JSON output too. Agents reading the
/// hardware response see the same keys consistently.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct HostInfo {
    /// Cross-trace key. `host|<hostname>` when SystemInfo has a
    /// hostname; otherwise `host|vm:<vm_id>` so the row still has a
    /// stable identity. Populated at extract time so callers that go
    /// through the meta-cache sidecar see the same string a fresh
    /// extract would.
    pub key: String,
    /// Hardware host id — bits 48-63 of NSys `globalVid` / `globalTid`.
    /// Useful for correlating runtime rows on multi-host traces.
    pub hw_host_id: u16,
    /// NSys-internal VM id (raw `TARGET_INFO_SYSTEM_ENV.globalVid`).
    pub vm_id: u64,
    /// OS / platform attributes. `None` only on partial exports
    /// where every queried key is absent.
    pub system: Option<SystemInfo>,
    /// CUDA / NVIDIA / OFED driver versions.
    pub drivers: Option<DriverInfo>,
    /// CPU description. `None` when NSys didn't record any CPU
    /// attribute.
    pub cpu: Option<CpuInfo>,
    /// GPU devices. Empty Vec on hosts with no CUDA-visible GPU.
    pub gpus: Vec<GpuInfo>,
    /// Network interfaces (mostly InfiniBand on cluster traces).
    pub nics: Vec<NicInfo>,
}

/// Operating-system + platform attributes from `TARGET_INFO_SYSTEM_ENV`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SystemInfo {
    pub hostname: Option<String>,
    pub system_uid: Option<String>,
    /// e.g. `"Ubuntu 22.04.3 LTS"`.
    pub os_description: Option<String>,
    pub kernel_version: Option<String>,
    /// e.g. `"x86_64"`.
    pub hardware_platform: Option<String>,
    pub software_platform: Option<String>,
    pub software_release_version: Option<String>,
}

impl SystemInfo {
    /// `true` iff at least one field carries a value — used by
    /// `extract` to decide between `Some(SystemInfo)` and `None`.
    fn has_any(&self) -> bool {
        self.hostname.is_some()
            || self.system_uid.is_some()
            || self.os_description.is_some()
            || self.kernel_version.is_some()
            || self.hardware_platform.is_some()
            || self.software_platform.is_some()
            || self.software_release_version.is_some()
    }
}

/// Driver versions. CUDA's `cuda_driver_version` is an integer
/// encoding (`major * 1000 + minor * 10`); [`DriverInfo::cuda_version_parsed`]
/// produces the canonical `"13.0"` shape.
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DriverInfo {
    pub cuda_driver_version: Option<String>,
    pub nv_driver_version: Option<String>,
    /// OFED / InfiniBand driver — present only on cluster traces.
    pub ofed_driver_version: Option<String>,
}

impl DriverInfo {
    fn has_any(&self) -> bool {
        self.cuda_driver_version.is_some()
            || self.nv_driver_version.is_some()
            || self.ofed_driver_version.is_some()
    }

    /// Decode `cuda_driver_version`'s `major*1000 + minor*10` packing
    /// into `"<major>.<minor>"`. Returns `None` when the field is
    /// missing or unparseable.
    pub fn cuda_version_parsed(&self) -> Option<String> {
        let raw = self.cuda_driver_version.as_ref()?;
        let n: u32 = raw.parse().ok()?;
        let major = n / 1000;
        let minor = (n % 1000) / 10;
        Some(format!("{major}.{minor}"))
    }
}

/// CPU description. `model` is mandatory in the `Some(_)` case —
/// `core_count` and `clock_mhz` round-trip as `Option<u32>` because
/// some NSys exports omit them.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CpuInfo {
    pub model: String,
    pub core_count: Option<u32>,
    pub clock_mhz: Option<u32>,
    pub architecture: Option<String>,
}

/// GPU device entry. `name` is the human-readable string NSys
/// writes (e.g. `"NVIDIA H100 PCIe 80GB"`); `chip_name` is the
/// codename (`"GH100"`). Compute capability splits across
/// `compute_major` / `compute_minor`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct GpuInfo {
    pub id: u32,
    pub name: String,
    pub uuid: Option<String>,
    pub chip_name: Option<String>,
    pub total_memory: Option<u64>,
    pub sm_count: Option<u32>,
    pub compute_major: Option<u32>,
    pub compute_minor: Option<u32>,
    /// PCI bus location (e.g. `"0000:1d:00.0"`).
    pub bus_location: Option<String>,
}

/// Network interface. Vendor IDs (e.g. `5555` for Mellanox) round
/// through unchanged — agents that want human-readable vendor names
/// can run a separate lookup; veloq doesn't bundle a PCI ID database.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NicInfo {
    pub id: u32,
    pub name: String,
    pub device_id: Option<u32>,
    pub vendor_id: Option<u32>,
}

/// Pull the full hardware topology. Returns `Ok(vec![])` (not an
/// error) when `TARGET_INFO_SYSTEM_ENV` is absent — that's a partial
/// export, not a failure mode, and the empty Vec is the correct
/// response for a JSON consumer.
pub fn extract(trace: &Trace) -> NsysDataResult<Vec<HostInfo>> {
    if !trace.table_exists("TARGET_INFO_SYSTEM_ENV") {
        log::debug!("TARGET_INFO_SYSTEM_ENV absent — hardware topology unavailable");
        return Ok(Vec::new());
    }

    // Step 1: pivot the key-value rows into one row per host.
    // The MAX(CASE WHEN name = 'X' THEN value END) idiom is what
    // NSys's own reports use — preserves the per-host grouping
    // (`globalVid`) while keeping the projection schema stable
    // across NSys versions (extra `name`s show up as new columns we
    // just ignore).
    let mut hosts: Vec<HostInfo> = Vec::new();
    let mut stmt = trace.conn().prepare(SYSTEM_ENV_PIVOT).map_err(|source| {
        crate::NsysDataError::hardware_rows_prepare("TARGET_INFO_SYSTEM_ENV", source)
    })?;
    let mut rows = stmt.query([]).map_err(|source| {
        crate::NsysDataError::hardware_rows_query("TARGET_INFO_SYSTEM_ENV", source)
    })?;
    while let Some(row) = rows.next().map_err(|source| {
        crate::NsysDataError::hardware_rows_read("TARGET_INFO_SYSTEM_ENV", source)
    })? {
        let global_vid_text: String = row.get(0).map_err(|source| {
            crate::NsysDataError::hardware_rows_read("TARGET_INFO_SYSTEM_ENV", source)
        })?;
        let global_vid = parse_u64_id(&global_vid_text, "TARGET_INFO_SYSTEM_ENV.globalVid")?;
        // NSys packs hw_host_id into bits 48-63.
        let hw_host_id = ((global_vid >> 48) & 0xFFFF) as u16;

        let system = SystemInfo {
            hostname: row.get(1).map_err(|source| {
                crate::NsysDataError::hardware_rows_read("TARGET_INFO_SYSTEM_ENV", source)
            })?,
            system_uid: row.get(2).map_err(|source| {
                crate::NsysDataError::hardware_rows_read("TARGET_INFO_SYSTEM_ENV", source)
            })?,
            os_description: row.get(3).map_err(|source| {
                crate::NsysDataError::hardware_rows_read("TARGET_INFO_SYSTEM_ENV", source)
            })?,
            kernel_version: row.get(4).map_err(|source| {
                crate::NsysDataError::hardware_rows_read("TARGET_INFO_SYSTEM_ENV", source)
            })?,
            hardware_platform: row.get(5).map_err(|source| {
                crate::NsysDataError::hardware_rows_read("TARGET_INFO_SYSTEM_ENV", source)
            })?,
            software_platform: row.get(6).map_err(|source| {
                crate::NsysDataError::hardware_rows_read("TARGET_INFO_SYSTEM_ENV", source)
            })?,
            software_release_version: row.get(7).map_err(|source| {
                crate::NsysDataError::hardware_rows_read("TARGET_INFO_SYSTEM_ENV", source)
            })?,
        };
        let system = if system.has_any() { Some(system) } else { None };

        let cpu_cores: Option<i64> = row.get(8).map_err(|source| {
            crate::NsysDataError::hardware_rows_read("TARGET_INFO_SYSTEM_ENV", source)
        })?;
        let cpu_speed_mhz: Option<i64> = row.get(9).map_err(|source| {
            crate::NsysDataError::hardware_rows_read("TARGET_INFO_SYSTEM_ENV", source)
        })?;
        let cpu_arch: Option<String> = row.get(10).map_err(|source| {
            crate::NsysDataError::hardware_rows_read("TARGET_INFO_SYSTEM_ENV", source)
        })?;
        let cpu_desc: Option<String> = row.get(11).map_err(|source| {
            crate::NsysDataError::hardware_rows_read("TARGET_INFO_SYSTEM_ENV", source)
        })?;
        // CPU only surfaces when NSys recorded the description; the
        // `model` field carries the verbatim string. `core_count`
        // and `clock_mhz` round-trip as `Option` so partial exports
        // (description present, cores/MHz missing) still show the
        // CPU name.
        let cpu = cpu_desc.map(|model| CpuInfo {
            model,
            core_count: cpu_cores.map(|n| n as u32),
            clock_mhz: cpu_speed_mhz.map(|n| n as u32),
            architecture: cpu_arch,
        });

        let drivers = DriverInfo {
            cuda_driver_version: row.get(12).map_err(|source| {
                crate::NsysDataError::hardware_rows_read("TARGET_INFO_SYSTEM_ENV", source)
            })?,
            nv_driver_version: row.get(13).map_err(|source| {
                crate::NsysDataError::hardware_rows_read("TARGET_INFO_SYSTEM_ENV", source)
            })?,
            ofed_driver_version: row.get(14).map_err(|source| {
                crate::NsysDataError::hardware_rows_read("TARGET_INFO_SYSTEM_ENV", source)
            })?,
        };
        let drivers = if drivers.has_any() {
            Some(drivers)
        } else {
            None
        };

        let key = match system.as_ref().and_then(|s| s.hostname.as_deref()) {
            Some(hostname) => format!("host|{hostname}"),
            None => format!("host|vm:{global_vid}"),
        };
        hosts.push(HostInfo {
            key,
            hw_host_id,
            vm_id: global_vid,
            system,
            drivers,
            cpu,
            gpus: Vec::new(),
            nics: Vec::new(),
        });
    }
    drop(rows);
    drop(stmt);

    // Step 2: attach GPUs. Multiple GPUs per host are common; rows
    // are grouped via `vmId` (the same identifier `globalVid`
    // points at on the system_env side).
    if trace.table_exists("TARGET_INFO_GPU") {
        attach_gpus(trace, &mut hosts)?;
    } else {
        log::debug!("TARGET_INFO_GPU absent — GPU inventory empty");
    }

    // Step 3: attach NICs. Single-host traces (the common case) get
    // every NIC; multi-host attribution is unresolved in NSys's
    // schema — there's no GUID → vmId mapping anywhere in the
    // exports we've seen.
    if trace.table_exists("TARGET_INFO_NIC_INFO") {
        attach_nics(trace, &mut hosts)?;
    } else {
        log::debug!("TARGET_INFO_NIC_INFO absent — NIC inventory empty");
    }

    Ok(hosts)
}

fn attach_gpus(trace: &Trace, hosts: &mut [HostInfo]) -> NsysDataResult<()> {
    const TABLE: &str = "TARGET_INFO_GPU";
    let vm_id_expr = crate::sql_expr::u64_decimal_string("vmId");
    let mut stmt = trace
        .conn()
        .prepare(&format!(
            "SELECT {vm_id_expr}, id, name, uuid, chipName, totalMemory, smCount, \
                computeMajor, computeMinor, busLocation \
         FROM nsight.TARGET_INFO_GPU \
         ORDER BY vmId, id"
        ))
        .map_err(|source| crate::NsysDataError::hardware_rows_prepare(TABLE, source))?;
    let mut rows = stmt
        .query([])
        .map_err(|source| crate::NsysDataError::hardware_rows_query(TABLE, source))?;
    while let Some(row) = rows
        .next()
        .map_err(|source| crate::NsysDataError::hardware_rows_read(TABLE, source))?
    {
        let vm_id_text: String = row
            .get(0)
            .map_err(|source| crate::NsysDataError::hardware_rows_read(TABLE, source))?;
        let vm_id = parse_u64_id(&vm_id_text, "TARGET_INFO_GPU.vmId")?;
        let hw_host_id = ((vm_id >> 48) & 0xFFFF) as u16;
        let id: i64 = row
            .get(1)
            .map_err(|source| crate::NsysDataError::hardware_rows_read(TABLE, source))?;
        let name: Option<String> = row
            .get(2)
            .map_err(|source| crate::NsysDataError::hardware_rows_read(TABLE, source))?;
        let uuid: Option<String> = row
            .get(3)
            .map_err(|source| crate::NsysDataError::hardware_rows_read(TABLE, source))?;
        let chip_name: Option<String> = row
            .get(4)
            .map_err(|source| crate::NsysDataError::hardware_rows_read(TABLE, source))?;
        let total_memory: Option<i64> = row
            .get(5)
            .map_err(|source| crate::NsysDataError::hardware_rows_read(TABLE, source))?;
        let sm_count: Option<i64> = row
            .get(6)
            .map_err(|source| crate::NsysDataError::hardware_rows_read(TABLE, source))?;
        let compute_major: Option<i64> = row
            .get(7)
            .map_err(|source| crate::NsysDataError::hardware_rows_read(TABLE, source))?;
        let compute_minor: Option<i64> = row
            .get(8)
            .map_err(|source| crate::NsysDataError::hardware_rows_read(TABLE, source))?;
        let bus_location: Option<String> = row
            .get(9)
            .map_err(|source| crate::NsysDataError::hardware_rows_read(TABLE, source))?;

        let Some(name) = name else {
            // No name → no point surfacing the GPU. Real traces
            // always have one; this is defence against a partial
            // export.
            continue;
        };

        let gpu = GpuInfo {
            id: id as u32,
            name,
            uuid,
            chip_name,
            total_memory: total_memory.map(|n| n as u64),
            sm_count: sm_count.map(|n| n as u32),
            compute_major: compute_major.map(|n| n as u32),
            compute_minor: compute_minor.map(|n| n as u32),
            bus_location,
        };

        match hosts.iter_mut().find(|h| h.hw_host_id == hw_host_id) {
            Some(host) => host.gpus.push(gpu),
            None => log::warn!(
                "GPU vmId={vm_id} (hw_host_id=0x{hw_host_id:04X}) has no matching host in SYSTEM_ENV"
            ),
        }
    }
    Ok(())
}

fn attach_nics(trace: &Trace, hosts: &mut [HostInfo]) -> NsysDataResult<()> {
    const TABLE: &str = "TARGET_INFO_NIC_INFO";
    let guid_expr = crate::sql_expr::u64_decimal_string("GUID");
    let mut stmt = trace
        .conn()
        .prepare(&format!(
            "SELECT {guid_expr}, nicId, name, deviceId, vendorId \
         FROM nsight.TARGET_INFO_NIC_INFO \
         ORDER BY nicId, GUID"
        ))
        .map_err(|source| crate::NsysDataError::hardware_rows_prepare(TABLE, source))?;
    let mut rows = stmt
        .query([])
        .map_err(|source| crate::NsysDataError::hardware_rows_query(TABLE, source))?;
    // NSys's schema doesn't expose a NIC → host mapping. Single-host
    // traces get every NIC attached to the only host; multi-host
    // traces skip NIC attribution to avoid mis-assigning interfaces
    // to the wrong node.
    let single_host = hosts.len() == 1;
    while let Some(row) = rows
        .next()
        .map_err(|source| crate::NsysDataError::hardware_rows_read(TABLE, source))?
    {
        // Read GUID even though we don't use it — proves the column
        // is present and forces a per-row probe of the table shape.
        let _guid: String = row
            .get(0)
            .map_err(|source| crate::NsysDataError::hardware_rows_read(TABLE, source))?;
        let id: i64 = row
            .get(1)
            .map_err(|source| crate::NsysDataError::hardware_rows_read(TABLE, source))?;
        let name: Option<String> = row
            .get(2)
            .map_err(|source| crate::NsysDataError::hardware_rows_read(TABLE, source))?;
        let device_id: Option<i64> = row
            .get(3)
            .map_err(|source| crate::NsysDataError::hardware_rows_read(TABLE, source))?;
        let vendor_id: Option<i64> = row
            .get(4)
            .map_err(|source| crate::NsysDataError::hardware_rows_read(TABLE, source))?;

        if !single_host {
            // One log line per multi-host trace, not per row.
            log::debug!("skipping NIC inventory — multi-host trace has no GUID→host mapping");
            break;
        }

        let Some(name) = name else { continue };
        let nic = NicInfo {
            id: id as u32,
            name,
            device_id: device_id.map(|n| n as u32),
            vendor_id: vendor_id.map(|n| n as u32),
        };
        // `single_host` is true here, so `hosts.first_mut()` is
        // guaranteed `Some`; route through `if let` to satisfy the
        // workspace's no-unwrap policy.
        if let Some(host) = hosts.first_mut() {
            host.nics.push(nic);
        }
    }
    Ok(())
}

fn parse_u64_id(value: &str, label: &str) -> NsysDataResult<u64> {
    value
        .parse::<u64>()
        .map_err(|source| crate::NsysDataError::hardware_invalid_u64_id(label, value, source))
}

/// Pivot SQL for `TARGET_INFO_SYSTEM_ENV`. NSys stores host info as
/// (name, value) rows; we pivot into one row per `globalVid` so the
/// Rust side reads positionally. Adding a new field means: add the
/// `MAX(CASE WHEN name = 'NewKey' THEN value END)` clause + a new
/// `row.get(N)` in `extract`. Keep the column indices and the
/// `row.get` calls aligned.
const SYSTEM_ENV_PIVOT: &str = "\
    SELECT CAST(globalVid AS VARCHAR), \
           MAX(CASE WHEN name = 'Hostname'              THEN value END)              AS hostname, \
           MAX(CASE WHEN name = 'SystemUID'             THEN value END)              AS system_uid, \
           MAX(CASE WHEN name = 'OsDescription'         THEN value END)              AS os_desc, \
           MAX(CASE WHEN name = 'KernelVersion'         THEN value END)              AS kernel_ver, \
           MAX(CASE WHEN name = 'HardwarePlatform'      THEN value END)              AS hw_platform, \
           MAX(CASE WHEN name = 'SoftwarePlatform'      THEN value END)              AS sw_platform, \
           MAX(CASE WHEN name = 'SoftwareReleaseVersion' THEN value END)             AS sw_release, \
           MAX(CASE WHEN name = 'CpuCores'              THEN CAST(value AS BIGINT) END) AS cpu_cores, \
           MAX(CASE WHEN name = 'CpuSpeedMhz'           THEN CAST(value AS BIGINT) END) AS cpu_speed_mhz, \
           MAX(CASE WHEN name = 'CpuArchitecture'       THEN value END)              AS cpu_arch, \
           MAX(CASE WHEN name = 'CpuDescription'        THEN value END)              AS cpu_desc, \
           MAX(CASE WHEN name = 'CudaDriverVersion'     THEN value END)              AS cuda_drv, \
           MAX(CASE WHEN name = 'NvDriverVersion'       THEN value END)              AS nv_drv, \
           MAX(CASE WHEN name = 'OfedDriverVersion'     THEN value END)              AS ofed_drv \
    FROM nsight.TARGET_INFO_SYSTEM_ENV \
    GROUP BY globalVid";

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use duckdb::Connection;
    use std::path::PathBuf;
    use tempfile::TempDir;
    use veloq_core::VeloqDiagnostic;

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

    fn assert_hardware_rows_error(
        err: crate::NsysDataError,
        expected_code: &str,
        expected_table: &str,
    ) -> Result<()> {
        assert_eq!(err.code().as_str(), expected_code);
        let Some((area, _, label)) = err.duckdb_parts() else {
            anyhow::bail!("expected hardware rows DuckDB error, got {err:?}");
        };
        assert_eq!(area, "hardware rows");
        assert_eq!(label, expected_table);
        Ok(())
    }

    #[test]
    fn cuda_version_parsed_decodes_int_encoding() {
        let d = DriverInfo {
            cuda_driver_version: Some("13000".into()),
            ..Default::default()
        };
        assert_eq!(d.cuda_version_parsed().as_deref(), Some("13.0"));

        let d = DriverInfo {
            cuda_driver_version: Some("12050".into()),
            ..Default::default()
        };
        assert_eq!(d.cuda_version_parsed().as_deref(), Some("12.5"));

        let d = DriverInfo {
            cuda_driver_version: None,
            ..Default::default()
        };
        assert!(d.cuda_version_parsed().is_none());

        let d = DriverInfo {
            cuda_driver_version: Some("not-a-number".into()),
            ..Default::default()
        };
        assert!(d.cuda_version_parsed().is_none());
    }

    #[test]
    fn system_info_has_any_detects_partial_population() {
        let mut s = SystemInfo::default();
        assert!(!s.has_any());
        s.hostname = Some("box".into());
        assert!(s.has_any());
    }

    #[test]
    fn parse_u64_id_rejects_invalid_value_with_typed_error() -> Result<()> {
        let err = match parse_u64_id("not-a-number", "TARGET_INFO_GPU.vmId") {
            Ok(value) => anyhow::bail!("invalid u64 id should not parse as {value}"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.data.hardware-invalid-u64-id");
        match err {
            crate::NsysDataError::HardwareInvalidU64Id { label, value, .. } => {
                assert_eq!(label, "TARGET_INFO_GPU.vmId");
                assert_eq!(value, "not-a-number");
            }
            other => anyhow::bail!("expected HardwareInvalidU64Id, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn system_env_bad_cpu_cores_has_typed_query_error() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture_with_rows(&[
            (
                "CUPTI_ACTIVITY_KIND_KERNEL",
                r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start BIGINT, "end" BIGINT)"#,
                Vec::new(),
            ),
            (
                "TARGET_INFO_SYSTEM_ENV",
                "CREATE TABLE TARGET_INFO_SYSTEM_ENV (globalVid BIGINT, name TEXT, value TEXT)",
                vec![
                    "INSERT INTO TARGET_INFO_SYSTEM_ENV (globalVid, name, value) VALUES (0, 'Hostname', 'box')",
                    "INSERT INTO TARGET_INFO_SYSTEM_ENV (globalVid, name, value) VALUES (0, 'CpuCores', 'bad')",
                ],
            ),
        ])?;
        let trace = Trace::open(&pqtdir)?;

        let err = match extract(&trace) {
            Ok(rows) => anyhow::bail!("bad CpuCores cast should fail: {rows:?}"),
            Err(err) => err,
        };

        assert_hardware_rows_error(err, "nsys.data.duckdb-query", "TARGET_INFO_SYSTEM_ENV")
    }

    #[test]
    fn gpu_missing_vmid_has_typed_prepare_error() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture_with_rows(&[
            (
                "CUPTI_ACTIVITY_KIND_KERNEL",
                r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start BIGINT, "end" BIGINT)"#,
                Vec::new(),
            ),
            (
                "TARGET_INFO_SYSTEM_ENV",
                "CREATE TABLE TARGET_INFO_SYSTEM_ENV (globalVid BIGINT, name TEXT, value TEXT)",
                vec![
                    "INSERT INTO TARGET_INFO_SYSTEM_ENV (globalVid, name, value) VALUES (0, 'Hostname', 'box')",
                ],
            ),
            (
                "TARGET_INFO_GPU",
                "CREATE TABLE TARGET_INFO_GPU (id BIGINT, name TEXT)",
                Vec::new(),
            ),
        ])?;
        let trace = Trace::open(&pqtdir)?;

        let err = match extract(&trace) {
            Ok(rows) => anyhow::bail!("missing GPU vmId should fail: {rows:?}"),
            Err(err) => err,
        };

        assert_hardware_rows_error(err, "nsys.data.duckdb-prepare", "TARGET_INFO_GPU")
    }

    #[test]
    fn gpu_bad_id_has_typed_read_error() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture_with_rows(&[
            (
                "CUPTI_ACTIVITY_KIND_KERNEL",
                r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start BIGINT, "end" BIGINT)"#,
                Vec::new(),
            ),
            (
                "TARGET_INFO_SYSTEM_ENV",
                "CREATE TABLE TARGET_INFO_SYSTEM_ENV (globalVid BIGINT, name TEXT, value TEXT)",
                vec![
                    "INSERT INTO TARGET_INFO_SYSTEM_ENV (globalVid, name, value) VALUES (0, 'Hostname', 'box')",
                ],
            ),
            (
                "TARGET_INFO_GPU",
                "CREATE TABLE TARGET_INFO_GPU (vmId BIGINT, id TEXT, name TEXT, uuid TEXT, chipName TEXT, totalMemory BIGINT, smCount BIGINT, computeMajor BIGINT, computeMinor BIGINT, busLocation TEXT)",
                vec![
                    "INSERT INTO TARGET_INFO_GPU (vmId, id, name, uuid, chipName, totalMemory, smCount, computeMajor, computeMinor, busLocation) VALUES (0, 'bad', 'GPU', NULL, NULL, NULL, NULL, NULL, NULL, NULL)",
                ],
            ),
        ])?;
        let trace = Trace::open(&pqtdir)?;

        let err = match extract(&trace) {
            Ok(rows) => anyhow::bail!("bad GPU id should fail: {rows:?}"),
            Err(err) => err,
        };

        assert_hardware_rows_error(err, "nsys.data.duckdb-read", "TARGET_INFO_GPU")
    }
}
