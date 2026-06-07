//! `veloq inspect <trace> <row_id> [<row_id> …]` — full event details.
//!
//! Returns one structured record per requested row_id. Fields are
//! per-kind (kernel has grid/block/regs, memcpy has bytes/copyKind,
//! etc.). Missing optional columns from older NSys schemas come back as
//! JSON `null`.
//!
//! Per-kind query logic lives in family submodules below; this file
//! owns the public envelope (`EventDetails` / `InspectResponse`), the
//! tabular `summary_row` projection, the `run` dispatcher, and the
//! schema-probe helpers shared by every family.

mod cpu_sample;
mod gpu_work;
mod graph;
mod host_api;
mod overhead;
mod sync;

pub use cpu_sample::{CallchainFrame, CpuSampleDetails};
pub use gpu_work::{KernelDetails, MemcpyDetails, MemsetDetails};
pub use graph::{GraphDetails, GraphEventDetails, GraphNodeDetails};
pub use host_api::{NvtxDetails, OsrtDetails, RuntimeDetails};
pub use overhead::OverheadDetails;
pub use sync::{CudaEventDetails, SyncDetails};

use crate::column_map::{self, ColumnMap, maybe_col, opt_string};
use crate::query_sql::exec::{SqlLabel, query_optional_row_fallible};
use crate::{EventKind, NsysQueryError, NsysQueryResult, RowId};
use duckdb::types::Value;
use serde::Serialize;
use std::path::Path;
use veloq_nsys_data::Trace;

#[derive(Debug, Serialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum EventDetails {
    Kernel(KernelDetails),
    Memcpy(MemcpyDetails),
    Memset(MemsetDetails),
    Runtime(RuntimeDetails),
    Osrt(OsrtDetails),
    Nvtx(NvtxDetails),
    Sync(SyncDetails),
    Graph(GraphDetails),
    #[serde(rename = "graph_node")]
    GraphNode(GraphNodeDetails),
    #[serde(rename = "graph_event")]
    GraphEvent(GraphEventDetails),
    #[serde(rename = "cuda_event")]
    CudaEvent(CudaEventDetails),
    Overhead(OverheadDetails),
    #[serde(rename = "cpu_sample")]
    CpuSample(CpuSampleDetails),
    /// Returned when a `row_id` doesn't exist in the table it claims.
    #[serde(rename = "not_found")]
    NotFound {
        /// Cross-trace key — equal to `row_id` stringified.
        key: String,
        row_id: RowId,
    },
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct InspectResponse {
    /// Rows returned (= input row_ids resolved).
    pub count: usize,
    /// Same as `count` today — `inspect` doesn't paginate; every
    /// requested row_id yields one row.
    pub total_matched: usize,
    /// Canonical primary table. Each row is the full details for
    /// one requested row_id.
    pub rows: Vec<EventDetails>,
}

/// Compact projection of an [`EventDetails`] for tabular/CSV output.
/// Lives next to the dispatch in this module so the per-kind field
/// flattening stays in one place — formatting consumers (`veloq`'s
/// `views::inspect_view`) just call [`EventDetails::summary_row`] and
/// don't need to match the variants themselves.
pub struct SummaryRow {
    pub row_id: String,
    pub kind: &'static str,
    pub name: String,
    pub start_ns: String,
    pub duration_ns: String,
    pub device_id: Option<i64>,
    pub stream_id: Option<i64>,
    pub details: String,
}

impl EventDetails {
    pub fn summary_row(&self) -> SummaryRow {
        match self {
            EventDetails::Kernel(k) => SummaryRow {
                row_id: k.row_id.to_string(),
                kind: "kernel",
                name: k
                    .demangled_name
                    .clone()
                    .or_else(|| k.short_name.clone())
                    .unwrap_or_else(|| "<anonymous>".into()),
                start_ns: k.start_ns.to_string(),
                duration_ns: k.duration_ns.to_string(),
                device_id: Some(k.device_id as i64),
                stream_id: Some(k.stream_id),
                details: format!(
                    "grid=[{},{},{}] block=[{},{},{}]",
                    k.grid[0], k.grid[1], k.grid[2], k.block[0], k.block[1], k.block[2]
                ),
            },
            EventDetails::Memcpy(m) => SummaryRow {
                row_id: m.row_id.to_string(),
                kind: "memcpy",
                name: m.copy_kind_name.to_string(),
                start_ns: m.start_ns.to_string(),
                duration_ns: m.duration_ns.to_string(),
                device_id: Some(m.device_id as i64),
                stream_id: Some(m.stream_id),
                details: format!("bytes={}", m.bytes),
            },
            EventDetails::Memset(m) => SummaryRow {
                row_id: m.row_id.to_string(),
                kind: "memset",
                name: "cudaMemset".to_string(),
                start_ns: m.start_ns.to_string(),
                duration_ns: m.duration_ns.to_string(),
                device_id: Some(m.device_id as i64),
                stream_id: Some(m.stream_id),
                details: format!("bytes={}", m.bytes),
            },
            EventDetails::Runtime(r) => SummaryRow {
                row_id: r.row_id.to_string(),
                kind: "runtime",
                name: r.name.clone(),
                start_ns: r.start_ns.to_string(),
                duration_ns: r.duration_ns.to_string(),
                device_id: None,
                stream_id: None,
                details: format!("global_tid={}", r.global_tid),
            },
            EventDetails::Osrt(o) => SummaryRow {
                row_id: o.row_id.to_string(),
                kind: "osrt",
                name: o.name.clone(),
                start_ns: o.start_ns.to_string(),
                duration_ns: o.duration_ns.to_string(),
                device_id: None,
                stream_id: None,
                details: format!("global_tid={}", o.global_tid),
            },
            EventDetails::Nvtx(n) => SummaryRow {
                row_id: n.row_id.to_string(),
                kind: "nvtx",
                name: n.name.clone(),
                start_ns: n.start_ns.to_string(),
                duration_ns: n.duration_ns.map(|d| d.to_string()).unwrap_or_default(),
                device_id: None,
                stream_id: None,
                details: format!(
                    "global_tid={} domain={} event_type={}",
                    n.global_tid, n.domain_id, n.event_type
                ),
            },
            EventDetails::Sync(s) => SummaryRow {
                row_id: s.row_id.to_string(),
                kind: "sync",
                name: s.sync_type_name.to_string(),
                start_ns: s.start_ns.to_string(),
                duration_ns: s.duration_ns.to_string(),
                device_id: Some(s.device_id as i64),
                stream_id: Some(s.stream_id),
                details: format!("sync_type={}", s.sync_type),
            },
            EventDetails::Graph(g) => SummaryRow {
                row_id: g.row_id.to_string(),
                kind: "graph",
                name: format!("graph:{}", g.graph_id),
                start_ns: g.start_ns.to_string(),
                duration_ns: g.duration_ns.to_string(),
                device_id: Some(g.device_id as i64),
                stream_id: Some(g.stream_id),
                details: format!("graph_exec={}", g.graph_exec_id),
            },
            EventDetails::GraphNode(n) => SummaryRow {
                row_id: n.row_id.to_string(),
                kind: "graph_node",
                name: format!("node:{}", n.graph_node_id),
                start_ns: n.start_ns.to_string(),
                duration_ns: n.duration_ns.to_string(),
                device_id: None,
                stream_id: None,
                details: {
                    let mut parts: Vec<String> = Vec::new();
                    if let Some(g) = n.graph_id {
                        parts.push(format!("graph={g}"));
                    }
                    if let Some(o) = n.original_graph_node_id {
                        parts.push(format!("orig={o}"));
                    }
                    parts.join(" ")
                },
            },
            EventDetails::GraphEvent(g) => SummaryRow {
                row_id: g.row_id.to_string(),
                kind: "graph_event",
                name: g.event_class_name.to_string(),
                start_ns: g.start_ns.to_string(),
                duration_ns: g.duration_ns.to_string(),
                device_id: None,
                stream_id: None,
                details: {
                    let mut parts: Vec<String> = vec![format!("graph={}", g.graph_id)];
                    if let Some(e) = g.graph_exec_id {
                        parts.push(format!("exec={e}"));
                    }
                    parts.join(" ")
                },
            },
            EventDetails::CudaEvent(c) => SummaryRow {
                row_id: c.row_id.to_string(),
                kind: "cuda_event",
                name: format!("cuda_event:{}", c.event_id),
                start_ns: c.start_ns.to_string(),
                duration_ns: "0".to_string(),
                device_id: Some(c.device_id as i64),
                stream_id: Some(c.stream_id),
                details: match c.event_sync_id {
                    Some(s) => format!("sync_id={s}"),
                    None => String::new(),
                },
            },
            EventDetails::Overhead(o) => SummaryRow {
                row_id: o.row_id.to_string(),
                kind: "overhead",
                name: o.overhead_type_name.to_string(),
                start_ns: o.start_ns.to_string(),
                duration_ns: o.duration_ns.to_string(),
                device_id: None,
                stream_id: None,
                details: format!("type={}", o.overhead_type),
            },
            EventDetails::CpuSample(c) => SummaryRow {
                row_id: c.row_id.to_string(),
                kind: "cpu_sample",
                name: c
                    .callchain
                    .first()
                    .and_then(|f| f.symbol.clone())
                    .unwrap_or_else(|| "<unresolved>".to_string()),
                start_ns: c.start_ns.to_string(),
                duration_ns: "0".to_string(),
                device_id: None,
                stream_id: None,
                details: format!(
                    "cpu={} pid={} tid={} state={} depth={}",
                    c.cpu,
                    c.pid,
                    c.tid,
                    c.thread_state_name.as_deref().unwrap_or("?"),
                    c.callchain.len()
                ),
            },
            EventDetails::NotFound { key: _, row_id } => SummaryRow {
                row_id: row_id.to_string(),
                kind: "not_found",
                name: String::new(),
                start_ns: String::new(),
                duration_ns: String::new(),
                device_id: None,
                stream_id: None,
                details: String::new(),
            },
        }
    }
}

pub fn run<P: AsRef<Path>>(path: P, row_ids: &[RowId]) -> NsysQueryResult<InspectResponse> {
    // inspect reads a few individual CUPTI/NVTX/graph rows; `Trace::open`
    // already exposes every `nsight.<TABLE>`, so no extra setup is needed.
    let trace = Trace::open(path).map_err(NsysQueryError::trace_open)?;
    let columns = column_map::load_standard(trace.conn())?;

    // Compute NVTX nesting once when *any* requested row_id either is
    // an NVTX hit (for `depth` on NvtxDetails) or is a kind that the
    // reverse-attribution path can populate (`nvtx_context`).
    // Amortises the rayon scan over all row_ids in the batch.
    let needs_nesting = row_ids.iter().any(|id| {
        id.kind == EventKind::Nvtx
            || matches!(
                id.kind,
                EventKind::Kernel
                    | EventKind::Memcpy
                    | EventKind::Memset
                    | EventKind::Sync
                    | EventKind::Runtime
            )
    });
    let nesting = if needs_nesting {
        Some(
            trace
                .nvtx_nesting()
                .map_err(NsysQueryError::nvtx_nesting_load)?,
        )
    } else {
        None
    };
    let needs_nvtx_tree =
        row_ids.iter().any(|id| id.kind == EventKind::Nvtx) && trace.table_exists("NVTX_EVENTS");
    let nvtx_tree = if needs_nvtx_tree {
        Some(
            veloq_nsys_data::nvtx_tree::build_or_load(&trace)
                .map_err(NsysQueryError::nvtx_tree_load)?,
        )
    } else {
        None
    };

    let mut events = Vec::with_capacity(row_ids.len());
    for id in row_ids {
        let details = match id.kind {
            EventKind::Kernel => gpu_work::query_kernel(trace.conn(), &columns, *id)?,
            EventKind::Memcpy => gpu_work::query_memcpy(trace.conn(), &columns, *id)?,
            EventKind::Memset => gpu_work::query_memset(trace.conn(), &columns, *id)?,
            EventKind::Runtime => host_api::query_runtime(trace.conn(), &columns, *id)?,
            EventKind::Osrt => host_api::query_osrt(trace.conn(), &columns, *id)?,
            EventKind::Nvtx => host_api::query_nvtx(
                trace.conn(),
                &columns,
                *id,
                nesting.as_ref(),
                nvtx_tree.as_ref(),
            )?,
            EventKind::Sync => sync::query_sync(trace.conn(), &columns, *id)?,
            EventKind::Graph => graph::query_graph(trace.conn(), &columns, *id)?,
            EventKind::GraphNode => graph::query_graph_node(trace.conn(), &columns, *id)?,
            EventKind::GraphEvent => graph::query_graph_event(trace.conn(), &columns, *id)?,
            EventKind::CudaEvent => sync::query_cuda_event(trace.conn(), &columns, *id)?,
            EventKind::Overhead => overhead::query_overhead(trace.conn(), &columns, *id)?,
            EventKind::CpuSample => cpu_sample::query_cpu_sample(trace.conn(), &columns, *id)?,
        }
        .unwrap_or_else(|| EventDetails::NotFound {
            key: id.to_string(),
            row_id: *id,
        });
        events.push(details);
    }

    // Reverse NVTX attribution — default-on for inspect: any
    // kernel/memcpy/memset/sync row_id in the batch gets its innermost
    // enclosing NVTX range surfaced as `nvtx_context`. One SQL per
    // kind present, so a 100-row mixed batch fans out to ≤4 SQLs.
    if let Some(nesting_map) = nesting.as_ref() {
        let contexts = crate::nvtx_reverse::lookup_for_row_ids(&trace, row_ids, nesting_map)?;
        for event in &mut events {
            attach_nvtx_context(event, &contexts);
        }
    }

    let count = events.len();
    Ok(InspectResponse {
        count,
        total_matched: count,
        rows: events,
    })
}

/// Drop the looked-up `NvtxContext` into the matching variant's
/// `nvtx_context` slot. Variants that don't carry the field
/// (`NotFound`, `Osrt`, `Nvtx`, `Graph*`, `CudaEvent`, `Overhead`,
/// `CpuSample`) silently ignore the lookup — the reverse query only
/// emitted contexts for kinds it knows how to walk anyway, so the
/// keyed-by-RowId map is naturally sparse on those.
fn attach_nvtx_context(
    event: &mut EventDetails,
    contexts: &std::collections::HashMap<RowId, crate::NvtxContext>,
) {
    match event {
        EventDetails::Kernel(k) => k.nvtx_context = contexts.get(&k.row_id).cloned(),
        EventDetails::Memcpy(m) => m.nvtx_context = contexts.get(&m.row_id).cloned(),
        EventDetails::Memset(m) => m.nvtx_context = contexts.get(&m.row_id).cloned(),
        EventDetails::Sync(s) => s.nvtx_context = contexts.get(&s.row_id).cloned(),
        EventDetails::Runtime(r) => r.nvtx_context = contexts.get(&r.row_id).cloned(),
        _ => {}
    }
}

fn query_inspect_row(
    conn: &duckdb::Connection,
    kind: &'static str,
    sql: &str,
    id: RowId,
    hydrate: impl FnOnce(&duckdb::Row<'_>) -> NsysQueryResult<EventDetails>,
) -> NsysQueryResult<Option<EventDetails>> {
    let params = [Value::BigInt(id.rowid)];
    query_optional_row_fallible(conn, sql, &params, SqlLabel::new("inspect", kind), hydrate)
}

pub(super) fn map_inspect_read<T>(
    kind: &'static str,
    result: std::result::Result<T, duckdb::Error>,
) -> NsysQueryResult<T> {
    result.map_err(|source| crate::NsysQueryError::sql_read("inspect", kind, source))
}

// Schema-probe helpers (`ColumnMap`, `load_columns`, `has`,
// `maybe_col`, `opt_string`) live in `crate::column_map` — both
// `inspect` and `search` consume them.

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use veloq_core::VeloqDiagnostic;

    fn row_id() -> RowId {
        RowId::new(EventKind::Kernel, 1)
    }

    #[test]
    fn query_inspect_row_prepare_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;

        let err = match query_inspect_row(&conn, "test_kind", "SELECT * FROM", row_id(), |_| {
            Err(crate::NsysQueryError::internal_sql_kind_tag_invalid(
                "inspect",
                "unexpected_hydrate",
            ))
        }) {
            Ok(_) => anyhow::bail!("malformed inspect SQL should not succeed"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-prepare");
        assert_eq!(
            err.sql_parts(),
            Some(("inspect", crate::SqlPhase::Prepare, "test_kind"))
        );
        Ok(())
    }

    #[test]
    fn query_inspect_row_query_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = "SELECT ? AS value WHERE ? IS NOT NULL";

        let err = match query_inspect_row(&conn, "test_kind", sql, row_id(), |_| {
            Err(crate::NsysQueryError::internal_sql_kind_tag_invalid(
                "inspect",
                "unexpected_hydrate",
            ))
        }) {
            Ok(_) => anyhow::bail!("unbound inspect SQL should not succeed"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-query");
        assert_eq!(
            err.sql_parts(),
            Some(("inspect", crate::SqlPhase::Query, "test_kind"))
        );
        Ok(())
    }

    #[test]
    fn query_inspect_row_read_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = "SELECT 'not-an-int' AS value WHERE ? IS NOT NULL";

        let err = match query_inspect_row(&conn, "test_kind", sql, row_id(), |row| {
            let _: i64 = map_inspect_read("test_kind", row.get(0))?;
            Ok(EventDetails::NotFound {
                key: row_id().to_string(),
                row_id: row_id(),
            })
        }) {
            Ok(_) => anyhow::bail!("malformed inspect row should not hydrate"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-read");
        assert_eq!(
            err.sql_parts(),
            Some(("inspect", crate::SqlPhase::Read, "test_kind"))
        );
        Ok(())
    }
}
