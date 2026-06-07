//! Trace-map projection used by `veloq info`'s extended response.
//!
//! The agent-facing introspection surface has five
//! touchpoints; this module owns touchpoint 1 ("what is this trace?").
//! The output is a structured snapshot of:
//!
//! - **devices** — count + CUDA ordinals, from
//!   `TARGET_INFO_GPU.cuDevice` (preferred) or the
//!   `CUPTI_ACTIVITY_KIND_KERNEL` DISTINCT fallback.
//! - **processes** — count + sorted native pids + per-launch env-derived
//!   labels (`rank=`, `local_rank=`, `slurm_procid=`,
//!   `mpi_comm_world_rank=`, `pmi_rank=`) recovered from
//!   `META_DATA_CAPTURE.PROCESS_N:ENVIRONMENT_VARIABLE` rows.
//! - **nvtx** — domain inventory + top paths by `SUM(duration_ns)`
//!   projected from the NVTX-tree sidecar; `None` when the trace lacks
//!   NVTX or the sidecar isn't built (caller falls back gracefully).
//!
//! The projection is *cheap on a warm cache*: every probe is either a
//! `LIMIT N` on a parquet-backed view or a slice of the in-memory NVTX
//! tree. `info`'s sub-100ms contract is preserved when the path is a
//! direct `_pqtdir/` input or a generated parquetdir alias; cold
//! `.nsys-rep` callers do not reach this module (`info` keeps the
//! basics-only payload).

use crate::nvtx_tree::NvtxTree;
use crate::{NsysDataResult, Trace};
use std::collections::{BTreeMap, HashMap};

/// Top-K cutoff for `nvtx.top_paths`. Surfacing the long tail would
/// blow up `info`'s payload on traces with thousands of distinct NVTX
/// paths; ten is enough for agents to see the hierarchy and pick a
/// scope. Public so tests and future tuning work agree on the cap.
pub const NVTX_TOP_PATHS_DEFAULT: usize = 10;

/// Snapshot returned by [`build`]. Each block is independently optional
/// (the nvtx one is `Option`; devices/processes are always present but
/// can be empty when the source tables are missing).
#[derive(Debug, Clone)]
pub struct TraceMap {
    pub devices: DeviceInventory,
    pub processes: ProcessInventory,
    pub nvtx: Option<NvtxSummary>,
}

/// Distinct CUDA device ids seen in the trace.
#[derive(Debug, Clone, Default)]
pub struct DeviceInventory {
    pub count: usize,
    /// Ascending order — keeps test assertions and human inspection
    /// stable.
    pub ids: Vec<i32>,
}

/// Process inventory: distinct native pids that produced rows, plus a
/// list of launch recipes recovered from `META_DATA_CAPTURE`.
#[derive(Debug, Clone, Default)]
pub struct ProcessInventory {
    pub count: usize,
    pub native_pids: Vec<i64>,
    pub launches: Vec<ProcessLaunch>,
}

/// One `PROCESS_N` launch recipe with the rank-style env vars summarised.
#[derive(Debug, Clone)]
pub struct ProcessLaunch {
    /// `PROCESS_N` index as recorded by nsys.
    pub index: u32,
    /// Best-effort label: `"rank=0,local_rank=0"` style when any of the
    /// known rank-bearing env vars was set; falls back to the short
    /// command basename, or `"process_N"` as a last resort.
    pub label: String,
    /// Raw command string from `PROCESS_N:COMMAND`, when present. Kept
    /// here so callers can render their own labels if the env summary
    /// is empty.
    pub command: Option<String>,
}

/// NVTX-tree projection: which domains the trace uses and the top paths
/// by total duration.
#[derive(Debug, Clone, Default)]
pub struct NvtxSummary {
    pub domains: Vec<NvtxDomain>,
    pub top_paths: Vec<NvtxTopPath>,
}

/// One NVTX domain, identified per process. `domainId` is
/// a process-local handle assigned by NVTX in creation order, so the same
/// `id` means different domains in different processes; identity is the
/// `(pid, id)` pair. `name` is best-effort — recovered from the process's
/// own `NvtxDomainCreate` row; default-domain ranges land with `id == 0`
/// and `name == None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NvtxDomain {
    /// Owning process id, decoded `(global_tid >> 24) & 0xFFFFFF`.
    pub pid: i64,
    pub id: i64,
    pub name: Option<String>,
}

/// One aggregated NVTX path: total wall-clock under it across all
/// instances, with the instance count for context.
#[derive(Debug, Clone)]
pub struct NvtxTopPath {
    pub path: String,
    pub total_ns: i64,
    pub instances: u64,
}

/// Build the full trace map. Caller picks `nvtx_top_k`
/// ([`NVTX_TOP_PATHS_DEFAULT`] is the reasonable default).
///
/// All four sub-queries (`device_ids`, `process_pids`, `launches`,
/// `nvtx`) are independent — none of them errors out on missing tables;
/// each returns an empty inventory instead so an agent sees the honest
/// "this trace doesn't have it" shape rather than an opaque failure.
pub fn build(trace: &Trace, nvtx_top_k: usize) -> NsysDataResult<TraceMap> {
    let devices = build_devices(trace)?;
    let processes = build_processes(trace)?;
    let nvtx = build_nvtx(trace, nvtx_top_k)?;
    Ok(TraceMap {
        devices,
        processes,
        nvtx,
    })
}

fn build_devices(trace: &Trace) -> NsysDataResult<DeviceInventory> {
    let mut ids = collect_device_ids(trace)?;
    ids.sort_unstable();
    ids.dedup();
    Ok(DeviceInventory {
        count: ids.len(),
        ids,
    })
}

fn collect_device_ids(trace: &Trace) -> NsysDataResult<Vec<i32>> {
    if trace.has_table("TARGET_INFO_GPU") {
        let mut stmt = trace
            .conn()
            .prepare("SELECT CAST(cuDevice AS INTEGER) FROM nsight.TARGET_INFO_GPU")
            .map_err(|source| {
                crate::NsysDataError::trace_map_probe_column_missing(
                    "TARGET_INFO_GPU",
                    "cuDevice",
                    source,
                )
            })?;
        let mut rows = stmt.query([]).map_err(|source| {
            crate::NsysDataError::trace_map_rows_query("TARGET_INFO_GPU", source)
        })?;
        let mut out = Vec::new();
        while let Some(r) = rows.next().map_err(|source| {
            crate::NsysDataError::trace_map_rows_read("TARGET_INFO_GPU", source)
        })? {
            let id: Option<i32> = r.get(0).map_err(|source| {
                crate::NsysDataError::trace_map_rows_read("TARGET_INFO_GPU", source)
            })?;
            if let Some(id) = id {
                out.push(id);
            }
        }
        if !out.is_empty() {
            return Ok(out);
        }
    }
    if !trace.has_table("CUPTI_ACTIVITY_KIND_KERNEL") {
        return Ok(Vec::new());
    }
    let mut stmt = trace
        .conn()
        .prepare(
            "SELECT DISTINCT CAST(deviceId AS INTEGER) \
             FROM nsight.CUPTI_ACTIVITY_KIND_KERNEL \
             WHERE deviceId IS NOT NULL",
        )
        .map_err(|source| {
            crate::NsysDataError::trace_map_probe_column_missing(
                "CUPTI_ACTIVITY_KIND_KERNEL",
                "deviceId",
                source,
            )
        })?;
    let mut rows = stmt.query([]).map_err(|source| {
        crate::NsysDataError::trace_map_rows_query("CUPTI_ACTIVITY_KIND_KERNEL", source)
    })?;
    let mut out = Vec::new();
    while let Some(r) = rows.next().map_err(|source| {
        crate::NsysDataError::trace_map_rows_read("CUPTI_ACTIVITY_KIND_KERNEL", source)
    })? {
        out.push(r.get::<_, i32>(0).map_err(|source| {
            crate::NsysDataError::trace_map_rows_read("CUPTI_ACTIVITY_KIND_KERNEL", source)
        })?);
    }
    Ok(out)
}

fn build_processes(trace: &Trace) -> NsysDataResult<ProcessInventory> {
    let mut native_pids = collect_native_pids(trace)?;
    native_pids.sort_unstable();
    native_pids.dedup();
    let launches = collect_launches(trace)?;
    Ok(ProcessInventory {
        count: native_pids.len(),
        native_pids,
        launches,
    })
}

fn collect_native_pids(trace: &Trace) -> NsysDataResult<Vec<i64>> {
    if trace.has_table("PROCESSES") {
        let mut stmt = trace
            .conn()
            .prepare(
                "SELECT DISTINCT CAST(pid AS BIGINT) FROM nsight.PROCESSES WHERE pid IS NOT NULL",
            )
            .map_err(|source| {
                crate::NsysDataError::trace_map_probe_column_missing("PROCESSES", "pid", source)
            })?;
        let mut rows = stmt
            .query([])
            .map_err(|source| crate::NsysDataError::trace_map_rows_query("PROCESSES", source))?;
        let mut out = Vec::new();
        while let Some(r) = rows
            .next()
            .map_err(|source| crate::NsysDataError::trace_map_rows_read("PROCESSES", source))?
        {
            out.push(r.get::<_, i64>(0).map_err(|source| {
                crate::NsysDataError::trace_map_rows_read("PROCESSES", source)
            })?);
        }
        if !out.is_empty() {
            return Ok(out);
        }
    }
    if trace.has_table("TARGET_INFO_CUDA_CONTEXT_INFO") {
        let mut stmt = trace
            .conn()
            .prepare(
                "SELECT DISTINCT CAST(processId AS BIGINT) \
                 FROM nsight.TARGET_INFO_CUDA_CONTEXT_INFO \
                 WHERE processId IS NOT NULL",
            )
            .map_err(|source| {
                crate::NsysDataError::trace_map_probe_column_missing(
                    "TARGET_INFO_CUDA_CONTEXT_INFO",
                    "processId",
                    source,
                )
            })?;
        let mut rows = stmt.query([]).map_err(|source| {
            crate::NsysDataError::trace_map_rows_query("TARGET_INFO_CUDA_CONTEXT_INFO", source)
        })?;
        let mut out = Vec::new();
        while let Some(r) = rows.next().map_err(|source| {
            crate::NsysDataError::trace_map_rows_read("TARGET_INFO_CUDA_CONTEXT_INFO", source)
        })? {
            out.push(r.get::<_, i64>(0).map_err(|source| {
                crate::NsysDataError::trace_map_rows_read("TARGET_INFO_CUDA_CONTEXT_INFO", source)
            })?);
        }
        return Ok(out);
    }
    Ok(Vec::new())
}

/// Env-var names whose value we surface in process labels. Order matters
/// — the produced label lists them in declaration order so the rendered
/// string is stable across runs.
const RANK_ENV_KEYS: &[&str] = &[
    "RANK",
    "LOCAL_RANK",
    "SLURM_PROCID",
    "MPI_COMM_WORLD_RANK",
    "PMI_RANK",
];

fn collect_launches(trace: &Trace) -> NsysDataResult<Vec<ProcessLaunch>> {
    if !trace.has_table("META_DATA_CAPTURE") {
        return Ok(Vec::new());
    }
    let mut stmt = trace
        .conn()
        .prepare(
            "SELECT name, value FROM nsight.META_DATA_CAPTURE \
             WHERE name LIKE 'PROCESS\\_%:%' ESCAPE '\\'",
        )
        .map_err(|source| {
            crate::NsysDataError::trace_map_probe_column_missing(
                "META_DATA_CAPTURE",
                "name/value",
                source,
            )
        })?;
    let mut rows = stmt.query([]).map_err(|source| {
        crate::NsysDataError::trace_map_rows_query("META_DATA_CAPTURE", source)
    })?;
    // Aggregate every PROCESS_N:FIELD row into a per-index builder so we
    // can render labels independent of insertion order.
    let mut builders: BTreeMap<u32, LaunchBuilder> = BTreeMap::new();
    while let Some(r) = rows
        .next()
        .map_err(|source| crate::NsysDataError::trace_map_rows_read("META_DATA_CAPTURE", source))?
    {
        let key: String = r.get(0).map_err(|source| {
            crate::NsysDataError::trace_map_rows_read("META_DATA_CAPTURE", source)
        })?;
        let value: String = r.get(1).map_err(|source| {
            crate::NsysDataError::trace_map_rows_read("META_DATA_CAPTURE", source)
        })?;
        let Some((idx, field)) = split_process_key(&key) else {
            continue;
        };
        let entry = builders.entry(idx).or_default();
        match field {
            "COMMAND" => entry.command = Some(value),
            "ENVIRONMENT_VARIABLE" => {
                if let Some((k, v)) = parse_env_assignment(&value)
                    && RANK_ENV_KEYS.contains(&k.as_str())
                {
                    entry.env.insert(k, v);
                }
            }
            _ => {}
        }
    }
    let mut out = Vec::with_capacity(builders.len());
    for (idx, b) in builders {
        out.push(b.into_launch(idx));
    }
    Ok(out)
}

#[derive(Default)]
struct LaunchBuilder {
    command: Option<String>,
    env: HashMap<String, String>,
}

impl LaunchBuilder {
    fn into_launch(self, index: u32) -> ProcessLaunch {
        let label = render_label(index, self.command.as_deref(), &self.env);
        ProcessLaunch {
            index,
            label,
            command: self.command,
        }
    }
}

fn render_label(index: u32, command: Option<&str>, env: &HashMap<String, String>) -> String {
    let parts: Vec<String> = RANK_ENV_KEYS
        .iter()
        .filter_map(|k| {
            env.get(*k)
                .map(|v| format!("{}={}", k.to_ascii_lowercase(), v))
        })
        .collect();
    if !parts.is_empty() {
        return parts.join(",");
    }
    if let Some(cmd) = command {
        let basename = cmd.rsplit('/').next().unwrap_or(cmd).trim();
        if !basename.is_empty() {
            return basename.to_string();
        }
    }
    format!("process_{index}")
}

fn split_process_key(key: &str) -> Option<(u32, &str)> {
    let rest = key.strip_prefix("PROCESS_")?;
    let (idx_str, field) = rest.split_once(':')?;
    let idx = idx_str.parse::<u32>().ok()?;
    Some((idx, field))
}

fn parse_env_assignment(raw: &str) -> Option<(String, String)> {
    // Nsys serialises env entries as either `KEY=VALUE` or
    // `KEY="VALUE"`. Strip the optional surrounding quotes so a label
    // like `rank="0"` doesn't end up with quote characters in the
    // rendered payload.
    let (k, v) = raw.split_once('=')?;
    let v = v.trim();
    let v = v.strip_prefix('"').unwrap_or(v);
    let v = v.strip_suffix('"').unwrap_or(v);
    Some((k.to_string(), v.to_string()))
}

fn build_nvtx(trace: &Trace, top_k: usize) -> NsysDataResult<Option<NvtxSummary>> {
    if !trace.has_table("NVTX_EVENTS") {
        return Ok(None);
    }
    // Prefer an already-built sidecar (the warm path); fall back to a
    // build_or_load only when the sidecar is absent or stale, so a cold
    // call from `info` doesn't double-pay any other code path that
    // would build the tree anyway.
    let tree = match crate::nvtx_tree::load_if_present(trace)? {
        Some(t) => t,
        None => crate::nvtx_tree::build_or_load(trace)?,
    };
    let domains = collect_domains(trace, &tree)?;
    let top_paths = top_paths_from_tree(&tree, top_k);
    if domains.is_empty() && top_paths.is_empty() {
        return Ok(None);
    }
    Ok(Some(NvtxSummary { domains, top_paths }))
}

/// Decode the owning process id from a (already bit-reinterpreted)
/// `global_tid`: `(global_tid >> 24) & 0xFFFFFF` — the project's canonical
/// PID decode (mirrors `veloq_nsys_query`).
fn pid_of(global_tid: i64) -> i64 {
    (global_tid >> 24) & 0xFFFFFF
}

fn collect_domains(trace: &Trace, tree: &NvtxTree) -> NsysDataResult<Vec<NvtxDomain>> {
    // Domain identity is (pid, domainId): `domainId` is a process-local
    // handle, so the same id in two processes is two distinct domains.
    // Collect the distinct pairs actually present in the
    // tree, then attach each process's own resolved name.
    let names = nvtx_domain_names(trace)?;
    // Inventory = domains that have ranges in the tree UNION domains that
    // were registered (a `NvtxDomainCreate` with no ranges captured is
    // still a declared domain worth listing). Without the
    // registration keys, filtering the tree to ranges-only would drop
    // registered-but-unused domains from the inventory.
    let mut pairs: Vec<(i64, i64)> = tree
        .records()
        .iter()
        .map(|r| (pid_of(r.global_tid), r.domain_id))
        .collect();
    pairs.extend(names.keys().copied());
    Ok(domains_from(&pairs, &names))
}

/// Build the `(pid, domainId) -> name` map from the trace's
/// `NvtxDomainCreate` rows. The create event type is
/// resolved by NAME via the catalog (not a hardcoded int — `33` is
/// `NvtxCategory`, never domain-create); the name comes from the inline
/// `text` or, when registered, `StringIds.value`. Keyed by `(pid,
/// domainId)` so one process's registration cannot clobber another's.
///
/// Public so the query crate can domain-qualify `stats --group-by
/// nvtx-path` rows with the resolved domain name. Names are best-effort: a domain with no
/// `NvtxDomainCreate` row simply has no entry here.
pub fn nvtx_domain_names(trace: &Trace) -> NsysDataResult<HashMap<(i64, i64), String>> {
    let mut names = HashMap::new();
    let create_ids = crate::nvtx_tree::nvtx_event_type_ids(trace, &["NvtxDomainCreate"], &[75]);
    let create_list = create_ids
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let global_tid = crate::sql_expr::u64_bits_to_i64("n.globalTid");
    let sql = format!(
        "SELECT {global_tid}, CAST(n.domainId AS BIGINT), COALESCE(n.text, s.value) \
         FROM nsight.NVTX_EVENTS n \
         LEFT JOIN nsight.StringIds s ON n.textId = s.id \
         WHERE n.eventType IN ({create_list}) \
           AND n.globalTid IS NOT NULL \
           AND COALESCE(n.text, s.value) IS NOT NULL"
    );
    if let Ok(mut stmt) = trace.conn().prepare(&sql) {
        let mut rows = stmt
            .query([])
            .map_err(|source| crate::NsysDataError::trace_map_rows_query("NVTX_EVENTS", source))?;
        while let Some(r) = rows
            .next()
            .map_err(|source| crate::NsysDataError::trace_map_rows_read("NVTX_EVENTS", source))?
        {
            let gtid: i64 = r.get(0).map_err(|source| {
                crate::NsysDataError::trace_map_rows_read("NVTX_EVENTS", source)
            })?;
            let id: i64 = r.get(1).map_err(|source| {
                crate::NsysDataError::trace_map_rows_read("NVTX_EVENTS", source)
            })?;
            let name: String = r.get(2).map_err(|source| {
                crate::NsysDataError::trace_map_rows_read("NVTX_EVENTS", source)
            })?;
            if !name.is_empty() {
                names.insert((pid_of(gtid), id), name);
            }
        }
    }
    Ok(names)
}

/// Pure projection of distinct `(pid, domainId)` pairs into the domain
/// list, attaching each pair's resolved name. Split out so the
/// per-process identity is unit-testable without a DB,
/// mirroring [`crate::nvtx_tree`]'s `compute_from_rows`.
fn domains_from(pairs: &[(i64, i64)], names: &HashMap<(i64, i64), String>) -> Vec<NvtxDomain> {
    let mut seen: Vec<(i64, i64)> = pairs.to_vec();
    seen.sort_unstable();
    seen.dedup();
    seen.into_iter()
        .map(|(pid, id)| NvtxDomain {
            pid,
            id,
            name: names.get(&(pid, id)).cloned(),
        })
        .collect()
}

fn top_paths_from_tree(tree: &NvtxTree, top_k: usize) -> Vec<NvtxTopPath> {
    let mut agg: HashMap<&str, (i64, u64)> = HashMap::new();
    for r in tree.records() {
        if let Some(d) = r.duration_ns {
            let entry = agg.entry(r.path.as_str()).or_insert((0, 0));
            entry.0 = entry.0.saturating_add(d);
            entry.1 = entry.1.saturating_add(1);
        }
    }
    let mut paths: Vec<NvtxTopPath> = agg
        .into_iter()
        .map(|(path, (total_ns, instances))| NvtxTopPath {
            path: path.to_string(),
            total_ns,
            instances,
        })
        .collect();
    paths.sort_by(|a, b| {
        b.total_ns
            .cmp(&a.total_ns)
            .then(b.instances.cmp(&a.instances))
            .then(a.path.cmp(&b.path))
    });
    paths.truncate(top_k);
    paths
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

    fn malformed_trace(extra_tables: &[(&str, &str)]) -> Result<(TempDir, PathBuf)> {
        let mut tables = vec![(
            "CUPTI_ACTIVITY_KIND_KERNEL",
            r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start BIGINT, "end" BIGINT)"#,
        )];
        tables.extend_from_slice(extra_tables);
        parquet_fixture(&tables)
    }

    fn assert_trace_map_column_error(
        err: crate::NsysDataError,
        expected_table: &str,
        expected_column: &str,
    ) -> Result<()> {
        assert_eq!(
            err.code().as_str(),
            "nsys.data.trace-map-probe-column-missing"
        );
        match err {
            crate::NsysDataError::TraceMapProbeColumnMissing { table, column, .. } => {
                assert_eq!(table, expected_table);
                assert_eq!(column, expected_column);
            }
            other => anyhow::bail!("expected TraceMapProbeColumnMissing, got {other:?}"),
        }
        Ok(())
    }

    fn assert_trace_map_rows_query_error(
        err: crate::NsysDataError,
        expected_table: &str,
    ) -> Result<()> {
        assert_eq!(err.code().as_str(), "nsys.data.duckdb-query");
        let Some((area, phase, label)) = err.duckdb_parts() else {
            anyhow::bail!("expected trace map DuckDB error, got {err:?}");
        };
        assert_eq!(area, "trace map");
        assert_eq!(phase, crate::DuckdbPhase::Query);
        assert_eq!(label, expected_table);
        Ok(())
    }

    /// Domain identity is `(pid, domainId)`. The same
    /// `domainId` registered in two processes must stay two distinct
    /// domains, each keeping its own process's name — never merged, never
    /// clobbered. `domainId` is a process-local handle, so even identical
    /// names across processes are distinct entries.
    #[test]
    fn domains_keyed_by_pid_and_id_never_merge_or_clobber() {
        let mut names = HashMap::new();
        // Same domainId=1 in two processes, DIFFERENT names.
        names.insert((100, 1), "MPI".to_string());
        names.insert((200, 1), "NCCL".to_string());
        // Same name "CCCL" registered in two processes (domainId 2).
        names.insert((100, 2), "CCCL".to_string());
        names.insert((200, 2), "CCCL".to_string());

        // Pairs as they would appear from tree records (with a dup of (100,1)).
        let pairs = vec![
            (100, 0), // default domain, no name
            (100, 1),
            (100, 1),
            (200, 1),
            (100, 2),
            (200, 2),
        ];
        let domains = domains_from(&pairs, &names);

        // Five distinct (pid, id) identities; the (100,1) dup collapses.
        assert_eq!(domains.len(), 5, "got {domains:?}");
        let find = |pid: i64, id: i64| {
            domains
                .iter()
                .find(|d| d.pid == pid && d.id == id)
                .and_then(|d| d.name.as_deref())
        };
        // domainId=1 means different domains in different processes — no clobber.
        assert_eq!(find(100, 1), Some("MPI"));
        assert_eq!(find(200, 1), Some("NCCL"));
        // Same name in two processes stays two entries.
        assert_eq!(find(100, 2), Some("CCCL"));
        assert_eq!(find(200, 2), Some("CCCL"));
        // Default domain is nameless.
        assert_eq!(find(100, 0), None);
    }

    #[test]
    fn pid_decode_matches_canonical_shift() {
        // (global_tid >> 24) & 0xFFFFFF — same as veloq_nsys_query.
        assert_eq!(pid_of(281718749659136), 14530);

        // High-bit-set globalTid: `u64_bits_to_i64` reinterprets the bits,
        // so the value reaches `pid_of` as a *negative* i64. Rust's `>>`
        // on i64 is arithmetic (sign-extends), DuckDB's is logical — but
        // the `& 0xFFFFFF` mask keeps only original bits [24,48), which
        // both fills leave untouched, so the two decodes agree. Lock that
        // a negative global_tid still yields the masked pid.
        let bits: u64 = 0x8000_0000_0000_0000 | ((14530u64) << 24);
        let gtid = i64::from_ne_bytes(bits.to_ne_bytes());
        assert!(gtid < 0, "test value must be a negative i64");
        assert_eq!(pid_of(gtid), 14530);
    }

    #[test]
    fn device_probe_missing_target_info_gpu_cudevice_has_typed_error() -> Result<()> {
        let (_dir, pqtdir) = malformed_trace(&[(
            "TARGET_INFO_GPU",
            "CREATE TABLE TARGET_INFO_GPU (id BIGINT)",
        )])?;
        let trace = crate::Trace::open(&pqtdir)?;

        let err = match collect_device_ids(&trace) {
            Ok(ids) => anyhow::bail!("malformed TARGET_INFO_GPU should fail: {ids:?}"),
            Err(err) => err,
        };

        assert_trace_map_column_error(err, "TARGET_INFO_GPU", "cuDevice")
    }

    #[test]
    fn device_probe_bad_target_info_gpu_cudevice_has_typed_query_error() -> Result<()> {
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
        let trace = crate::Trace::open(&pqtdir)?;

        let err = match collect_device_ids(&trace) {
            Ok(ids) => anyhow::bail!("invalid TARGET_INFO_GPU should fail: {ids:?}"),
            Err(err) => err,
        };

        assert_trace_map_rows_query_error(err, "TARGET_INFO_GPU")
    }

    #[test]
    fn device_probe_missing_kernel_deviceid_has_typed_error() -> Result<()> {
        let (_dir, pqtdir) = malformed_trace(&[])?;
        let trace = crate::Trace::open(&pqtdir)?;

        let err = match collect_device_ids(&trace) {
            Ok(ids) => anyhow::bail!("malformed kernel table should fail: {ids:?}"),
            Err(err) => err,
        };

        assert_trace_map_column_error(err, "CUPTI_ACTIVITY_KIND_KERNEL", "deviceId")
    }

    #[test]
    fn process_probe_missing_processes_pid_has_typed_error() -> Result<()> {
        let (_dir, pqtdir) =
            malformed_trace(&[("PROCESSES", "CREATE TABLE PROCESSES (name TEXT)")])?;
        let trace = crate::Trace::open(&pqtdir)?;

        let err = match collect_native_pids(&trace) {
            Ok(pids) => anyhow::bail!("malformed PROCESSES should fail: {pids:?}"),
            Err(err) => err,
        };

        assert_trace_map_column_error(err, "PROCESSES", "pid")
    }

    #[test]
    fn process_probe_bad_processes_pid_has_typed_query_error() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture_with_rows(&[
            (
                "CUPTI_ACTIVITY_KIND_KERNEL",
                r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start BIGINT, "end" BIGINT)"#,
                Vec::new(),
            ),
            (
                "PROCESSES",
                "CREATE TABLE PROCESSES (pid TEXT)",
                vec!["INSERT INTO PROCESSES (pid) VALUES ('bad')"],
            ),
        ])?;
        let trace = crate::Trace::open(&pqtdir)?;

        let err = match collect_native_pids(&trace) {
            Ok(pids) => anyhow::bail!("invalid PROCESSES should fail: {pids:?}"),
            Err(err) => err,
        };

        assert_trace_map_rows_query_error(err, "PROCESSES")
    }

    #[test]
    fn process_probe_missing_context_processid_has_typed_error() -> Result<()> {
        let (_dir, pqtdir) = malformed_trace(&[(
            "TARGET_INFO_CUDA_CONTEXT_INFO",
            "CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO (deviceId BIGINT)",
        )])?;
        let trace = crate::Trace::open(&pqtdir)?;

        let err = match collect_native_pids(&trace) {
            Ok(pids) => anyhow::bail!("malformed context table should fail: {pids:?}"),
            Err(err) => err,
        };

        assert_trace_map_column_error(err, "TARGET_INFO_CUDA_CONTEXT_INFO", "processId")
    }

    #[test]
    fn launch_probe_missing_value_has_typed_error() -> Result<()> {
        let (_dir, pqtdir) = malformed_trace(&[(
            "META_DATA_CAPTURE",
            "CREATE TABLE META_DATA_CAPTURE (name TEXT)",
        )])?;
        let trace = crate::Trace::open(&pqtdir)?;

        let err = match collect_launches(&trace) {
            Ok(launches) => anyhow::bail!("malformed capture table should fail: {launches:?}"),
            Err(err) => err,
        };

        assert_trace_map_column_error(err, "META_DATA_CAPTURE", "name/value")
    }

    #[test]
    fn context_probe_bad_processid_has_typed_query_error() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture_with_rows(&[
            (
                "CUPTI_ACTIVITY_KIND_KERNEL",
                r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start BIGINT, "end" BIGINT)"#,
                Vec::new(),
            ),
            (
                "TARGET_INFO_CUDA_CONTEXT_INFO",
                "CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO (processId TEXT)",
                vec!["INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO (processId) VALUES ('bad')"],
            ),
        ])?;
        let trace = crate::Trace::open(&pqtdir)?;

        let err = match collect_native_pids(&trace) {
            Ok(pids) => anyhow::bail!("invalid context table should fail: {pids:?}"),
            Err(err) => err,
        };

        assert_trace_map_rows_query_error(err, "TARGET_INFO_CUDA_CONTEXT_INFO")
    }

    #[test]
    fn split_process_key_parses_well_formed_inputs() {
        assert_eq!(split_process_key("PROCESS_0:COMMAND"), Some((0, "COMMAND")));
        assert_eq!(
            split_process_key("PROCESS_12:ENVIRONMENT_VARIABLE"),
            Some((12, "ENVIRONMENT_VARIABLE"))
        );
        assert_eq!(split_process_key("not_a_process_key"), None);
        assert_eq!(split_process_key("PROCESS_x:COMMAND"), None);
    }

    #[test]
    fn parse_env_assignment_strips_quotes() -> anyhow::Result<()> {
        assert_eq!(
            parse_env_assignment("RANK=0")
                .ok_or_else(|| anyhow::anyhow!("expected RANK=0 to parse"))?,
            ("RANK".to_string(), "0".to_string())
        );
        assert_eq!(
            parse_env_assignment("LOCAL_RANK=\"3\"")
                .ok_or_else(|| anyhow::anyhow!("expected LOCAL_RANK=\"3\" to parse"))?,
            ("LOCAL_RANK".to_string(), "3".to_string())
        );
        assert_eq!(parse_env_assignment("no_equals_here"), None);
        Ok(())
    }

    #[test]
    fn render_label_prefers_rank_env_in_declaration_order() {
        let mut env = HashMap::new();
        env.insert("LOCAL_RANK".to_string(), "3".to_string());
        env.insert("RANK".to_string(), "5".to_string());
        // Declaration order in `RANK_ENV_KEYS` is RANK then LOCAL_RANK.
        assert_eq!(
            render_label(0, Some("/x/bin/app"), &env),
            "rank=5,local_rank=3"
        );
    }

    #[test]
    fn render_label_falls_back_to_command_basename() {
        let env = HashMap::new();
        assert_eq!(render_label(0, Some("/opt/work/app"), &env), "app");
    }

    #[test]
    fn render_label_falls_back_to_process_index_when_nothing_known() {
        let env = HashMap::new();
        assert_eq!(render_label(7, None, &env), "process_7");
    }
}
