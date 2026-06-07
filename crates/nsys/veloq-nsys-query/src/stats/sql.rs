use crate::query_sql::event_scan::stats_event_scan;
use crate::{EventKind, NsysQueryResult};
use duckdb::types::Value;

/// SQL fragment selecting (display_name, short_name, kind, duration) for one event table.
///
/// `display_name` is the leaf identity used by `--group-by demangled`
/// (demangled for kernels, label for memcpy/memset). `short_name` is
/// always the shortName for kernels, and identical to display_name for
/// memcpy/memset.
///
/// When `windowed` is true, four positional `?` parameters must be
/// bound in this order: `end, start, end, start`.
///
/// When `nvtx_scope` is `Attributed`, the WHERE clause includes a
/// rowid-IN filter against `attributed_<kind>_rowids` (a CTE that must
/// already be in scope). No additional params are bound.
/// Build the per-kind subquery body **and** its bind parameters as a
/// pair, so the caller can't get out of sync with positional `?`s.
/// Returning the body and params together prevents a placeholder added
/// here from silently misaligning every kind's bind slots across the
/// surrounding UNION ALL.
pub(super) fn per_kind_subquery(
    kind: EventKind,
    abs_window: Option<(i64, i64)>,
    nvtx_scope: crate::nvtx_attribution::NvtxScope,
    collapse_versioned: bool,
    columns: &crate::column_map::ColumnMap,
    nvtx_parent_sidecar: Option<&std::path::Path>,
    include_nvtx_path: bool,
) -> NsysQueryResult<(String, Vec<Value>)> {
    let scan = stats_event_scan(kind, abs_window, nvtx_scope, collapse_versioned)?;

    // Mangled-name projection: real value for kernels (StringIds
    // probe degrades to NULL on schemas missing the column), display
    // name for non-kernel kinds (preserves per-name identity so the
    // axis doesn't collapse memcpy/sync/runtime/NVTX into a single
    // NULL bucket).
    let (mangled_expr, mangled_join): (String, String) = if matches!(kind, EventKind::Kernel) {
        let mangled_col =
            crate::column_map::maybe_col(columns, "CUPTI_ACTIVITY_KIND_KERNEL", "mangledName");
        let join = format!("LEFT JOIN nsight.StringIds s_mng ON s_mng.id = {mangled_col}");
        ("s_mng.value".to_string(), join)
    } else {
        (scan.raw_display_expr.to_string(), String::new())
    };
    // When `--group-by nvtx-parent` is active,
    // LEFT JOIN against the trace-wide parquet sidecar built by
    // `veloq_nsys_data::runtime_nvtx_parent`. Events outside every
    // NVTX range fall back to the sentinel via COALESCE. Kinds
    // without an attribution path (Graph/Osrt/Nvtx) project the
    // sentinel inline.
    let (
        parent_rowid_expr,
        parent_name_expr,
        parent_path_expr,
        domain_id_expr,
        domain_pid_expr,
        parent_join,
    ) = match nvtx_parent_sidecar {
        Some(path) => crate::nvtx_parent::join_clause(kind, path, include_nvtx_path),
        None => (
            "CAST(NULL AS BIGINT)".to_string(),
            "CAST(NULL AS VARCHAR)".to_string(),
            "CAST(NULL AS VARCHAR)".to_string(),
            "CAST(NULL AS BIGINT)".to_string(),
            "CAST(NULL AS BIGINT)".to_string(),
            String::new(),
        ),
    };
    // Kernel-only grid/block projection. Only the kernel
    // CUPTI table carries gridX/Y/Z + blockX/Y/Z; for every other
    // kind we project typed NULLs so the UNION ALL stays homogeneous.
    // The axis is rejected upstream for non-kernel kinds so a single-
    // NULL-bucket row is never produced.
    let (grid_x_e, grid_y_e, grid_z_e, block_x_e, block_y_e, block_z_e) =
        if matches!(kind, EventKind::Kernel) {
            (
                "CAST(t.gridX  AS BIGINT)",
                "CAST(t.gridY  AS BIGINT)",
                "CAST(t.gridZ  AS BIGINT)",
                "CAST(t.blockX AS BIGINT)",
                "CAST(t.blockY AS BIGINT)",
                "CAST(t.blockZ AS BIGINT)",
            )
        } else {
            (
                "CAST(NULL AS BIGINT)",
                "CAST(NULL AS BIGINT)",
                "CAST(NULL AS BIGINT)",
                "CAST(NULL AS BIGINT)",
                "CAST(NULL AS BIGINT)",
                "CAST(NULL AS BIGINT)",
            )
        };
    let sql = format!(
        "SELECT {display_expr} AS display_name, \
                {short_expr}   AS short_name, \
                {mangled_expr} AS mangled_name, \
                '{label}'      AS kind, \
                {duration_expr} AS duration, \
                {dev}        AS device_id, \
                {ctx}        AS context_id, \
                {stm}        AS stream_id, \
                {bytes_expr}                       AS bytes, \
                {graph_id_expr}                    AS graph_id, \
                {graph_node_id_expr}               AS graph_node_id, \
                {event_type_expr}                  AS event_type, \
                {parent_rowid_expr}                AS nvtx_parent_rowid, \
                {parent_name_expr}                 AS nvtx_parent_name, \
                {parent_path_expr}                 AS nvtx_path, \
                {domain_id_expr}                   AS nvtx_domain_id, \
                {domain_pid_expr}                  AS nvtx_domain_pid, \
                {grid_x_e}                         AS grid_x, \
                {grid_y_e}                         AS grid_y, \
                {grid_z_e}                         AS grid_z, \
                {block_x_e}                        AS block_x, \
                {block_y_e}                        AS block_y, \
                {block_z_e}                        AS block_z \
         FROM nsight.{table} t {join_clause} {mangled_join} {parent_join} {where_clause}",
        display_expr = scan.display_expr.as_str(),
        short_expr = scan.short_expr.as_str(),
        label = scan.label,
        duration_expr = scan.duration_expr.as_str(),
        dev = scan.device_expr,
        ctx = scan.context_expr,
        stm = scan.stream_expr,
        bytes_expr = scan.bytes_expr,
        graph_id_expr = scan.graph_id_expr,
        graph_node_id_expr = scan.graph_node_id_expr,
        event_type_expr = scan.event_type_expr,
        table = scan.table,
        join_clause = scan.name_joins,
        where_clause = scan.where_clause.as_str(),
    );
    Ok((sql, scan.params))
}

/// SQL expression mapping raw NVTX `eventType` ints to the derived
/// style label. Mirrors `nvtx_style_label` Rust-side. `NULL` on
/// non-NVTX rows so the column collapses into a single bucket for
/// GROUP BY without splitting GPU rows.
///
/// The numeric constants come from NSys's
/// `enum NvtxEventType` (NSys SDK; see Nsight Systems documentation):
///
/// * 59, 70 → PushPop range (legacy + extended payload)
/// * 60, 71 → StartEnd range (legacy + extended payload)
///
/// Anything else (NVTX_RESOURCE_*, NVTX_DOMAIN_*, future enum
/// extensions) lands in `"unknown"` rather than spawning bucket-
/// per-int — this keeps the group count bounded as nsys adds new
/// event types.
pub(super) const NVTX_STYLE_EXPR: &str = "CASE \
    WHEN event_type IS NULL THEN NULL \
    WHEN event_type IN (59, 70) THEN 'push_pop' \
    WHEN event_type IN (60, 71) THEN 'start_end' \
    ELSE 'unknown' \
END";

/// Rust-side mirror of `NVTX_STYLE_EXPR` for derived response fields.
/// Used to coerce the SQL-emitted `nvtx_style` VARCHAR back into a
/// `&'static str` so consumers don't carry an unbounded String.
pub(super) fn nvtx_style_label(raw: &str) -> &'static str {
    match raw {
        "push_pop" => "push_pop",
        "start_end" => "start_end",
        _ => "unknown",
    }
}
