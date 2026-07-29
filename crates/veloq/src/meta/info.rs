//! `veloq info <trace>` — first-touch trace map.
//!
//! Two response shapes, picked per the discoverability rule
//! that `info` is the agent's first-look introspection:
//!
//! - **Warm parquetdir** (direct `_pqtdir/` input *or* a generated
//!   `<trace>.veloq/parquetdir/` alias for a cached `.nsys-rep`): we
//!   open the trace and emit the full map — `capabilities`,
//!   `trace_map.{devices,processes,nvtx}`, and the
//!   `applicable_recipes` filtered by trace shape. Stays sub-100ms
//!   because every probe is a `LIMIT N` on a parquet view or a slice
//!   of the in-memory NVTX-tree sidecar.
//!
//! - **Cold `.nsys-rep`** (no cached parquetdir yet): we keep the
//!   basics (`detected_source`, `exists`, `size_bytes`,
//!   `extension`) and add a `meta.next_steps[]` entry pointing at
//!   `veloq prep <path>`. Avoids the multi-second cold-export cost
//!   that would silently break the sub-100ms contract.
//!
//! Non-NSys paths and missing files keep emitting only the basics —
//! the trace-map blocks are NSys-specific so a stray `info` on an NCU
//! report or an unknown file stays well-behaved.

use clap::{Arg, ArgMatches, Command};
use serde::Serialize;
use std::path::{Path, PathBuf};
use veloq_core::{EnvelopeTraceRef, NextStep, OutputFormat, ProfileSource, ResponseMeta};
use veloq_nsys::CapabilityFlags;
use veloq_nsys::trace_map::{
    DeviceInventory, LogicalDeviceScope, NVTX_TOP_PATHS_DEFAULT, NvtxDomain, NvtxSummary,
    NvtxTopPath, PhysicalDeviceInventory, ProcessInventory, ProcessLaunch, TraceMap,
};

use super::{MetaError, MetaResult, emit_meta_error, emit_or_error};
use veloq_core::recipes::{self, TraceShape};

const VERB: &str = "info";

#[derive(Serialize)]
struct InfoPayload {
    /// Detected source kind (`"nsys"`, `"ncu"`, …), or `None` when
    /// no registered source claims the path.
    #[serde(skip_serializing_if = "Option::is_none")]
    detected_source: Option<&'static str>,
    /// File extension after the last `.`, lowercased. `None` for
    /// extensionless paths.
    #[serde(skip_serializing_if = "Option::is_none")]
    extension: Option<String>,
    /// `true` when `std::fs::metadata` succeeds. A `false` here is
    /// the most common reason a verb against this trace would fail.
    exists: bool,
    /// Size in bytes on disk when `exists`.
    #[serde(skip_serializing_if = "Option::is_none")]
    size_bytes: Option<u64>,
    /// Capability bitmap — `true` per (table-present /
    /// query-can-run) predicate. Same shape as
    /// `summary.auxiliary.capabilities`. Populated only when
    /// `detected_source == "nsys"`, the trace exists on disk, and
    /// the path is a direct `_pqtdir/` or VeloQ's generated
    /// `<report>.veloq/parquetdir/` alias.
    #[serde(skip_serializing_if = "Option::is_none")]
    capabilities: Option<CapabilityFlags>,
    /// Trace-shape projection: devices, processes (with rank-style
    /// labels), and an NVTX summary. Populated under the same warm-
    /// parquetdir gate as `capabilities`.
    #[serde(skip_serializing_if = "Option::is_none")]
    trace_map: Option<TraceMapOut>,
    /// Recipe ids whose `trace_shape` predicates this trace satisfies.
    /// Empty when no recipe matches; absent when the trace map wasn't
    /// computed (cold `.nsys-rep`, non-NSys path).
    #[serde(skip_serializing_if = "Option::is_none")]
    applicable_recipes: Option<Vec<&'static str>>,
}

#[derive(Serialize)]
struct TraceMapOut {
    devices: DevicesOut,
    processes: ProcessesOut,
    #[serde(skip_serializing_if = "Option::is_none")]
    nvtx: Option<NvtxOut>,
}

#[derive(Serialize)]
struct DevicesOut {
    #[serde(skip_serializing_if = "Option::is_none")]
    physical: Option<PhysicalDevicesOut>,
    logical_scopes: Vec<LogicalDeviceScopeOut>,
}

#[derive(Serialize)]
struct PhysicalDevicesOut {
    count: usize,
    ids: Vec<i32>,
}

#[derive(Serialize)]
struct LogicalDeviceScopeOut {
    process_id: i64,
    device_id: i32,
}

#[derive(Serialize)]
struct ProcessesOut {
    count: usize,
    native_pids: Vec<i64>,
    launches: Vec<LaunchOut>,
}

#[derive(Serialize)]
struct LaunchOut {
    index: u32,
    label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    command: Option<String>,
}

#[derive(Serialize)]
struct NvtxOut {
    domains: Vec<NvtxDomainOut>,
    top_paths: Vec<NvtxTopPathOut>,
}

#[derive(Serialize)]
struct NvtxDomainOut {
    /// Owning process id — domain identity is `(pid, id)`;
    /// `id` alone is a process-local handle.
    pid: i64,
    id: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

#[derive(Serialize)]
struct NvtxTopPathOut {
    path: String,
    total_ns: i64,
    instances: u64,
}

pub fn cli() -> Command {
    Command::new(VERB)
        .about("Identify which veloq source can read a trace (and basic filesystem facts)")
        .arg(
            Arg::new("trace")
                .required(true)
                .value_name("PATH")
                .help("Path to the trace artifact"),
        )
}

pub fn run(
    matches: &ArgMatches,
    sources: &[Box<dyn ProfileSource>],
    fmt: OutputFormat,
) -> MetaResult<i32> {
    let trace_str = match matches.get_one::<String>("trace") {
        Some(s) => s,
        None => {
            let err = MetaError::missing_argument("trace");
            emit_meta_error(fmt, VERB, None, &err);
            return Ok(1);
        }
    };
    let trace = PathBuf::from(trace_str);

    let detected = sources.iter().find(|s| s.detect(&trace)).map(|s| s.kind());

    let metadata = std::fs::metadata(&trace).ok();
    let extension = trace
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase());

    // The full trace map is computed only when we're pointing at an
    // already-exported parquetdir: either a direct `_pqtdir/` input
    // or the generated `<report>.veloq/parquetdir/` alias. Cold
    // `.nsys-rep` keeps the basics-only shape and surfaces a `prep`
    // hint via `meta.next_steps` instead.
    let is_direct_pqtdir = trace
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.ends_with("_pqtdir"));
    let is_generated_pqtdir = veloq_nsys::is_valid_generated_parquetdir(&trace);
    let parquetdir_ready = is_direct_pqtdir || is_generated_pqtdir;

    let mut capabilities = None;
    let mut trace_map = None;
    let mut applicable_recipes = None;
    let mut meta_next_steps: Vec<NextStep> = Vec::new();

    if detected == Some("nsys") && metadata.is_some() {
        if parquetdir_ready {
            let caps = CapabilityFlags::extract(&trace);
            // Open the trace once; the trace_map builder reuses the
            // connection for every probe so the parquet views are
            // registered exactly once.
            match veloq_nsys::Trace::open(&trace) {
                Ok(trace_handle) => {
                    match veloq_nsys::trace_map::build(&trace_handle, NVTX_TOP_PATHS_DEFAULT) {
                        Ok(map) => {
                            let shape = trace_shape(&caps, &map);
                            applicable_recipes = Some(applicable_recipe_ids(&shape));
                            trace_map = Some(project_trace_map(map));
                        }
                        Err(err) => {
                            // Trace-map failure shouldn't take down
                            // the whole info envelope — surface it
                            // as a hint instead so capabilities still
                            // ship.
                            log::warn!("trace_map build failed: {err:#}");
                        }
                    }
                }
                Err(err) => {
                    log::warn!("trace open failed during info: {err:#}");
                }
            }
            capabilities = Some(caps);
        } else if extension.as_deref() == Some("nsys-rep") {
            // Cold `.nsys-rep` — point the agent at the `prep` step
            // that unlocks the full map without making `info` itself
            // pay the export cost.
            meta_next_steps.push(NextStep {
                hint: "Run `veloq prep` to export the parquet sidecar; \
                       a follow-up `info` then surfaces devices, processes, \
                       and NVTX top paths."
                    .to_string(),
                command: format!("veloq prep {}", trace.display()),
            });
        }
    }

    let payload = InfoPayload {
        detected_source: detected,
        extension,
        exists: metadata.is_some(),
        size_bytes: metadata.as_ref().map(|m| m.len()),
        capabilities,
        trace_map,
        applicable_recipes,
    };

    let trace_ref = Some(EnvelopeTraceRef {
        kind: detected.unwrap_or("unknown"),
        path: trace_path_string(&trace),
    });

    let meta = if meta_next_steps.is_empty() {
        None
    } else {
        Some(ResponseMeta {
            next_steps: meta_next_steps,
            ..ResponseMeta::default()
        })
    };

    Ok(emit_or_error(fmt, VERB, trace_ref, meta, payload))
}

fn trace_path_string(p: &Path) -> String {
    p.display().to_string()
}

fn trace_shape(caps: &CapabilityFlags, map: &TraceMap) -> TraceShape {
    TraceShape {
        has_kernels: caps.has_kernels,
        has_memcpy: caps.has_memcpy,
        has_nvtx: caps.has_nvtx,
        has_target_info: caps.has_target_info,
        multi_device: map
            .devices
            .physical
            .as_ref()
            .is_some_and(|physical| physical.count > 1),
        multi_process: map.processes.count > 1,
        has_graph_trace: caps.has_graph_trace,
        has_graph_nodes: caps.has_graph_nodes,
    }
}

fn applicable_recipe_ids(shape: &TraceShape) -> Vec<&'static str> {
    recipes::recipes_for_trace_shape(shape)
        .map(|r| r.id)
        .collect()
}

fn project_trace_map(map: TraceMap) -> TraceMapOut {
    TraceMapOut {
        devices: project_devices(map.devices),
        processes: project_processes(map.processes),
        nvtx: map.nvtx.map(project_nvtx),
    }
}

fn project_devices(d: DeviceInventory) -> DevicesOut {
    DevicesOut {
        physical: d.physical.map(project_physical_devices),
        logical_scopes: d
            .logical_scopes
            .into_iter()
            .map(project_logical_scope)
            .collect(),
    }
}

fn project_physical_devices(d: PhysicalDeviceInventory) -> PhysicalDevicesOut {
    PhysicalDevicesOut {
        count: d.count,
        ids: d.ids,
    }
}

fn project_logical_scope(scope: LogicalDeviceScope) -> LogicalDeviceScopeOut {
    LogicalDeviceScopeOut {
        process_id: scope.process_id,
        device_id: scope.device_id,
    }
}

fn project_processes(p: ProcessInventory) -> ProcessesOut {
    ProcessesOut {
        count: p.count,
        native_pids: p.native_pids,
        launches: p.launches.into_iter().map(project_launch).collect(),
    }
}

fn project_launch(l: ProcessLaunch) -> LaunchOut {
    LaunchOut {
        index: l.index,
        label: l.label,
        command: l.command,
    }
}

fn project_nvtx(n: NvtxSummary) -> NvtxOut {
    NvtxOut {
        domains: n.domains.into_iter().map(project_domain).collect(),
        top_paths: n.top_paths.into_iter().map(project_top_path).collect(),
    }
}

fn project_domain(d: NvtxDomain) -> NvtxDomainOut {
    NvtxDomainOut {
        pid: d.pid,
        id: d.id,
        name: d.name,
    }
}

fn project_top_path(t: NvtxTopPath) -> NvtxTopPathOut {
    NvtxTopPathOut {
        path: t.path,
        total_ns: t.total_ns,
        instances: t.instances,
    }
}
