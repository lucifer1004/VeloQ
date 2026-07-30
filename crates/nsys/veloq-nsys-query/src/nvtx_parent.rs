//! Rank-and-pick-innermost NVTX parent attribution — SQL plumbing.
//!
//! Provides `stats --group-by nvtx-parent`,
//! a forward-attribution axis: every kernel/memcpy/memset/sync/runtime
//! event rolls up under the innermost NVTX range that fully contains
//! it (or under the visible sentinel `__no_nvtx__` when no enclosing
//! range exists).
//!
//! The actual attribution index — the runtime→NVTX-parent map — lives
//! in [`veloq_nsys_data::runtime_nvtx_parent`] and is persisted as
//! `<trace>.veloq/nvtx-parent.parquet`. This module is the SQL-side
//! adapter: it asks the data crate for the sidecar path, wires
//! `read_parquet('…')` into a `LEFT JOIN`, and provides the column
//! expressions stats.rs splices into the per-kind subquery.

use crate::{EventKind, NsysQueryError, NsysQueryResult};
use std::path::Path;
use veloq_nsys_data::Trace;

/// View / table name the per-kind events query joins against. Used
/// only for documentation/log lines; the actual SQL splices a
/// `read_parquet(…)` literal so DuckDB pushes filter predicates into
/// the scan.
pub fn parent_view_name(kind: EventKind) -> Option<&'static str> {
    match kind {
        EventKind::Kernel => Some("kernel_nvtx_parent_view"),
        EventKind::Memcpy => Some("memcpy_nvtx_parent_view"),
        EventKind::Memset => Some("memset_nvtx_parent_view"),
        EventKind::Sync => Some("sync_nvtx_parent_view"),
        EventKind::Runtime => Some("runtime_nvtx_parent_view"),
        _ => None,
    }
}

/// True iff `kind` has a parent-attribution path. Kinds without a path
/// (Graph, GraphNode, Osrt, Overhead, GraphEvent, CudaEvent, CpuSample,
/// Nvtx) always land in the no-NVTX sentinel.
pub fn has_parent_path(kind: EventKind) -> bool {
    parent_view_name(kind).is_some()
}

/// SQL-level sentinel for events that don't attribute to any NVTX
/// range. Visible to JSON consumers; agents filter "events outside
/// every range" via `.nvtx_parent_name == "__no_nvtx__"`.
pub const NO_NVTX_NAME: &str = "__no_nvtx__";

/// SQL-level sentinel for the `nvtx_parent_key` composite suffix.
pub const NO_NVTX_KEY: &str = "nvtx:none";

/// SQL-level sentinel for the `nvtx_path_key` composite suffix.
pub const NO_NVTX_PATH_KEY: &str = "nvtx-path:none";

/// Ensure the trace's NVTX-parent sidecar is built (cold) or current
/// (warm). Called once per `stats --group-by nvtx-parent` request,
/// before any per-kind subquery emits its `LEFT JOIN`. The sidecar
/// is shared across all attributable kinds in this query and across
/// every NVTX-bearing verb on the same trace.
pub fn ensure_sidecar(trace: &Trace) -> NsysQueryResult<()> {
    veloq_nsys_data::runtime_nvtx_parent::ensure_sidecar(trace)
        .map_err(NsysQueryError::nvtx_parent_sidecar_ensure)?;
    Ok(())
}

/// Emit the SQL JOIN snippet that decorates a per-kind events
/// subquery with `nvtx_parent_rowid` + `nvtx_parent_name` columns,
/// projecting the sentinel when no match exists. Returns
/// `(rowid_expr, name_expr, path_expr, domain_id_expr, domain_pid_expr,
/// join_clause)`.
///
/// `domain_id_expr` / `domain_pid_expr` project the innermost enclosing
/// range's NVTX domain identity `(pid, domainId)` from the joined
/// `nvtx_tree` row when `include_path` is set; `pid` is the canonical decode
/// `(global_tid >> 24) & 0xFFFFFF`. Both are typed NULL when the path
/// axis is inactive or the event has no enclosing range.
///
/// `sidecar_path` is the absolute path of the parquet sidecar. We
/// splice it literally rather than building a temp view so DuckDB's
/// parquet reader gets filter pushdown for free; the path is escaped
/// via [`crate::nvtx_projection::quote_sidecar_path`].
pub fn join_clause(
    trace: &Trace,
    kind: EventKind,
    sidecar_path: &Path,
    include_path: bool,
) -> (String, String, String, String, String, String) {
    let Some(_view) = parent_view_name(kind) else {
        return (
            "CAST(NULL AS BIGINT)".to_string(),
            format!("'{NO_NVTX_NAME}'"),
            format!("'{NO_NVTX_NAME}'"),
            "CAST(NULL AS BIGINT)".to_string(),
            "CAST(NULL AS BIGINT)".to_string(),
            String::new(),
        );
    };
    let quoted = crate::nvtx_projection::quote_sidecar_path(sidecar_path);
    let read = format!("read_parquet('{quoted}')");
    let parent_join = match kind {
        EventKind::Runtime => {
            // Runtime side joins on `rt_rowid` alone, so the
            // sidecar's multi-context fan-out (one row per
            // ambiguous `(device, context)` candidate, all sharing
            // the same `rt_rowid` and identical enclosing chain)
            // would double-count the runtime event under a plain
            // LEFT JOIN. Collapse fanout copies into one row per
            // `rt_rowid` via a deduped subquery — the
            // device/context fields aren't read on this path
            // anyway, and `nvtx_rowids` / `nvtx_names` are
            // identical across copies so `arbitrary()` is exact.
            format!(
                "LEFT JOIN (\
                   SELECT rt_rowid, \
                          arbitrary(nvtx_rowids) AS nvtx_rowids, \
                          arbitrary(nvtx_names)  AS nvtx_names \
                   FROM {read} \
                   GROUP BY rt_rowid\
                 ) np ON np.rt_rowid = t.rowid"
            )
        }
        EventKind::Kernel | EventKind::Memcpy | EventKind::Memset | EventKind::Sync => {
            let process = veloq_nsys_data::process_sql_projection(
                trace,
                kind.table(),
                "t",
                "cuda_proc",
                "t.start",
            );
            format!(
                "{process_join} \
                 LEFT JOIN {read} np \
                   ON np.native_pid     = {process_expr} \
                  AND np.device_id      = {dev} \
                  AND np.context_id     = {ctx} \
                  AND np.correlation_id = t.correlationId",
                process_join = process.join,
                process_expr = process.expr,
                dev = crate::kind_sql::GPU_DEVICE_ID_EXPR,
                ctx = crate::kind_sql::GPU_CONTEXT_ID_EXPR,
            )
        }
        _ => String::new(),
    };
    let path_join = if include_path {
        " LEFT JOIN nsight.nvtx_tree nt \
            ON nt.range_id = list_extract(np.nvtx_rowids, -1)"
            .to_string()
    } else {
        String::new()
    };
    let join = format!("{parent_join}{path_join}");
    // Sidecar: `nvtx_rowids` and `nvtx_names` are LIST<...>
    // columns ordered outermost→innermost. nvtx-parent wants the
    // innermost, which is `list[-1]` in DuckDB. The outer COALESCE on the name
    // handles the LEFT-JOIN miss (no sidecar row → list_extract
    // returns NULL).
    let (domain_id_expr, domain_pid_expr) = if include_path {
        (
            "CAST(nt.domain_id AS BIGINT)".to_string(),
            // Decode the owning pid from the tree row's globalTid — the
            // project's canonical shift. 16777215 == 0xFFFFFF.
            "CAST(((nt.global_tid >> 24) & 16777215) AS BIGINT)".to_string(),
        )
    } else {
        (
            "CAST(NULL AS BIGINT)".to_string(),
            "CAST(NULL AS BIGINT)".to_string(),
        )
    };
    (
        "list_extract(np.nvtx_rowids, -1)".to_string(),
        format!("COALESCE(list_extract(np.nvtx_names, -1), '{NO_NVTX_NAME}')"),
        if include_path {
            format!("COALESCE(nt.path, '{NO_NVTX_NAME}')")
        } else {
            "CAST(NULL AS VARCHAR)".to_string()
        },
        domain_id_expr,
        domain_pid_expr,
        join,
    )
}
