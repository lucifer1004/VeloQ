//! Per-event-kind SQL + label fragments shared across query commands.
//!
//! Without this, the same "what's this event called" projection ended
//! up open-coded in five places (stats, search, gaps, slices, correlate)
//! and the memcpy `copyKind → label` CASE drifted between sites (some
//! missed `HostToArray` / `ArrayToHost`). Touching kernel name
//! resolution or the memcpy copyKind set here now changes every command
//! at once.
//!
//! The SQL fragments assume two conventions in the surrounding query:
//! - the event row aliased as `t`
//! - kernel queries pre-join `nsight.StringIds` as `s_sh` (shortName)
//!   and `s_dem` (demangledName), via [`KERNEL_STRINGIDS_JOINS`].

use crate::EventKind;

/// `LEFT JOIN`s needed to resolve kernel names (shortName + demangled).
/// Other kinds don't need any joins for their name expressions.
pub const KERNEL_STRINGIDS_JOINS: &str = "\
    LEFT JOIN nsight.StringIds s_sh  ON t.shortName     = s_sh.id \
    LEFT JOIN nsight.StringIds s_dem ON t.demangledName = s_dem.id";

/// `LEFT JOIN` resolving NVTX_EVENTS range names. `text` is the
/// inline-string fast path; `textId` indirects through StringIds for
/// ranges that pre-interned. The `COALESCE` in
/// [`NVTX_TEXT_EXPR`] picks whichever is non-NULL.
pub const NVTX_STRINGIDS_JOIN: &str = "\
    LEFT JOIN nsight.StringIds s_text ON t.textId = s_text.id";

/// Display name for an NVTX range. Mirrors the projection used in
/// `inspect` and `search --type nvtx` so the same range text shows
/// up under each verb.
pub const NVTX_TEXT_EXPR: &str = "COALESCE(t.text, s_text.value, '<unnamed nvtx range>')";

/// Canonical SQL expression for the GPU device id off a `t`-aliased
/// CUPTI activity row. Used in SELECT lists (as `… AS device_id`) and
/// in WHERE predicates (as `… = ?`). Keeping the cast type in one
/// place means a future schema change (e.g. INT16 device ids) only
/// touches this constant.
pub const GPU_DEVICE_ID_EXPR: &str = "CAST(t.deviceId AS INTEGER)";

/// Canonical SQL expression for the CUDA context id off a `t`-aliased
/// CUPTI activity row.
pub const GPU_CONTEXT_ID_EXPR: &str = "CAST(t.contextId AS BIGINT)";

/// Canonical SQL expression for the GPU stream id off a `t`-aliased
/// CUPTI activity row. `COALESCE(..., 0)` matches the wire-format
/// convention that stream-less rows surface as `stream_id = 0`.
pub const GPU_STREAM_ID_EXPR: &str = "CAST(COALESCE(t.streamId, 0) AS BIGINT)";

/// CASE expression mapping `t.copyKind` to a human-readable label.
/// Single source of truth — keep `copy_kind_label` in sync with this
/// match list.
pub const MEMCPY_COPYKIND_CASE: &str = "\
    CASE t.copyKind \
        WHEN 1 THEN 'cudaMemcpyHostToDevice' \
        WHEN 2 THEN 'cudaMemcpyDeviceToHost' \
        WHEN 3 THEN 'cudaMemcpyHostToArray' \
        WHEN 4 THEN 'cudaMemcpyArrayToHost' \
        WHEN 8 THEN 'cudaMemcpyDeviceToDevice' \
        ELSE 'cudaMemcpy' \
    END";

/// Rust-side mirror of [`MEMCPY_COPYKIND_CASE`] for inspect-style hydration
/// after a SELECT has already pulled `copyKind`. Must stay aligned with
/// the SQL match list above.
pub fn copy_kind_label(k: i64) -> &'static str {
    match k {
        1 => "cudaMemcpyHostToDevice",
        2 => "cudaMemcpyDeviceToHost",
        3 => "cudaMemcpyHostToArray",
        4 => "cudaMemcpyArrayToHost",
        8 => "cudaMemcpyDeviceToDevice",
        _ => "cudaMemcpy",
    }
}

/// CASE expression mapping `t.syncType` (REFERENCES `ENUM_CUPTI_SYNC_TYPE`)
/// to a human-readable label. NSys exports the enum table only on some
/// schemas, so we hardcode the mapping. Keep [`sync_type_label`] aligned
/// with this match list.
pub const SYNC_TYPE_CASE: &str = "\
    CASE t.syncType \
        WHEN 1 THEN 'cudaEventSynchronize' \
        WHEN 2 THEN 'cudaStreamWaitEvent' \
        WHEN 3 THEN 'cudaStreamSynchronize' \
        WHEN 4 THEN 'cudaDeviceSynchronize' \
        ELSE 'cudaSync' \
    END";

/// Rust-side mirror of [`SYNC_TYPE_CASE`] for inspect-style hydration
/// after a SELECT has already pulled `syncType`. Must stay aligned with
/// the SQL match list above.
pub fn sync_type_label(k: i64) -> &'static str {
    match k {
        1 => "cudaEventSynchronize",
        2 => "cudaStreamWaitEvent",
        3 => "cudaStreamSynchronize",
        4 => "cudaDeviceSynchronize",
        _ => "cudaSync",
    }
}

/// CASE expression mapping `t.eventClass` to a human-readable label
/// for `CUDA_GRAPH_EVENTS` rows. The table mixes two event types
/// (94 = `GraphExec Creation`, 95 = `Graph Creation`); we project a
/// snake_case wire token so `--name 'graph_creation'` cleanly filters
/// to one sub-type. Keep [`graph_event_class_label`] aligned with
/// this match list.
pub const GRAPH_EVENT_CLASS_CASE: &str = "\
    CASE t.eventClass \
        WHEN 94 THEN 'graph_exec_creation' \
        WHEN 95 THEN 'graph_creation' \
        ELSE 'graph_event_other' \
    END";

/// Rust-side mirror of [`GRAPH_EVENT_CLASS_CASE`] for inspect-style
/// hydration. Must stay aligned with the SQL match list above.
pub fn graph_event_class_label(k: i64) -> &'static str {
    match k {
        94 => "graph_exec_creation",
        95 => "graph_creation",
        _ => "graph_event_other",
    }
}

/// CASE expression mapping `t.overheadType` to a snake_case wire
/// token. Mirrors the `ENUM_CUPTI_OVERHEAD_TYPE` enum from CUPTI
/// (ids 0..8). Keep [`overhead_type_label`] aligned.
pub const OVERHEAD_TYPE_CASE: &str = "\
    CASE t.overheadType \
        WHEN 0 THEN 'undefined' \
        WHEN 1 THEN 'unknown' \
        WHEN 2 THEN 'driver_compiler' \
        WHEN 3 THEN 'cupti_buffer_flush' \
        WHEN 4 THEN 'cupti_instrumentation' \
        WHEN 5 THEN 'cupti_resource' \
        WHEN 6 THEN 'runtime_module_loading' \
        WHEN 7 THEN 'lazy_function_loading' \
        WHEN 8 THEN 'command_buffer_full' \
        ELSE 'overhead_other' \
    END";

/// Rust-side mirror of [`OVERHEAD_TYPE_CASE`] for inspect hydration.
/// Must stay aligned with the SQL match list above.
pub fn overhead_type_label(k: i64) -> &'static str {
    match k {
        0 => "undefined",
        1 => "unknown",
        2 => "driver_compiler",
        3 => "cupti_buffer_flush",
        4 => "cupti_instrumentation",
        5 => "cupti_resource",
        6 => "runtime_module_loading",
        7 => "lazy_function_loading",
        8 => "command_buffer_full",
        _ => "overhead_other",
    }
}

/// `LEFT JOIN`s needed to resolve a runtime API name. Shared StringIds
/// resolution for the Runtime stats kind, distinct alias so it can
/// co-exist with the OSRT join below.
pub const RUNTIME_STRINGIDS_JOIN: &str = "LEFT JOIN nsight.StringIds s_rt ON t.nameId = s_rt.id";

/// `LEFT JOIN`s needed to resolve an OSRT API name (pthread/I/O syscall).
/// Same pattern as Runtime, distinct alias so the two can co-exist in a
/// future UNION ALL that includes both.
pub const OSRT_STRINGIDS_JOIN: &str = "LEFT JOIN nsight.StringIds s_os ON t.nameId = s_os.id";

/// "Display name" SQL: prefers demangled for kernels (the "which variant
/// is this" question agents ask), kind-label for memcpy/memset, sync-type
/// label for sync, resolved API name for Runtime/Osrt. CpuSample stays
/// empty because it isn't a first-class stats kind.
pub fn display_name_expr(kind: EventKind) -> &'static str {
    match kind {
        EventKind::Kernel => "COALESCE(s_dem.value, s_sh.value, '<anonymous kernel>')",
        EventKind::Memcpy => MEMCPY_COPYKIND_CASE,
        EventKind::Memset => "'cudaMemset'",
        EventKind::Sync => SYNC_TYPE_CASE,
        EventKind::Graph => "'graph:' || CAST(t.graphId AS VARCHAR)",
        EventKind::GraphNode => "'node:' || CAST(t.graphNodeId AS VARCHAR)",
        EventKind::GraphEvent => GRAPH_EVENT_CLASS_CASE,
        EventKind::CudaEvent => "'cuda_event:' || CAST(t.eventId AS VARCHAR)",
        EventKind::Overhead => OVERHEAD_TYPE_CASE,
        EventKind::Nvtx => NVTX_TEXT_EXPR,
        EventKind::Runtime => "COALESCE(s_rt.value, '<unknown runtime>')",
        EventKind::Osrt => "COALESCE(s_os.value, '<unknown osrt>')",
        EventKind::CpuSample => "''",
    }
}

/// "Short name" SQL: prefers shortName for kernels so demangled-grouped
/// rows can still roll up to a shortName bucket. Identical to
/// [`display_name_expr`] for memcpy/memset/sync/graph — each `graphId`
/// gets its own short bucket so the default `stats` rollup shows
/// per-captured-graph timing rather than collapsing every graph to a
/// single row.
pub fn short_name_expr(kind: EventKind) -> &'static str {
    match kind {
        EventKind::Kernel => "COALESCE(s_sh.value, s_dem.value, '<anonymous kernel>')",
        EventKind::Memcpy => MEMCPY_COPYKIND_CASE,
        EventKind::Memset => "'cudaMemset'",
        EventKind::Sync => SYNC_TYPE_CASE,
        EventKind::Graph => "'graph:' || CAST(t.graphId AS VARCHAR)",
        EventKind::GraphNode => "'graph_node'",
        EventKind::GraphEvent => GRAPH_EVENT_CLASS_CASE,
        EventKind::CudaEvent => "'cuda_event'",
        EventKind::Overhead => OVERHEAD_TYPE_CASE,
        EventKind::Nvtx => NVTX_TEXT_EXPR,
        EventKind::Runtime => "COALESCE(s_rt.value, '<unknown runtime>')",
        EventKind::Osrt => "COALESCE(s_os.value, '<unknown osrt>')",
        EventKind::CpuSample => "''",
    }
}

/// JOIN clauses needed for this kind's name resolution. Empty string
/// for kinds whose name comes from a literal or `CASE` expression.
pub fn name_joins(kind: EventKind) -> &'static str {
    match kind {
        EventKind::Kernel => KERNEL_STRINGIDS_JOINS,
        EventKind::Nvtx => NVTX_STRINGIDS_JOIN,
        EventKind::Runtime => RUNTIME_STRINGIDS_JOIN,
        EventKind::Osrt => OSRT_STRINGIDS_JOIN,
        _ => "",
    }
}
