//! Shared NVTX→GPU attribution CTE.
//!
//! Several commands (`stats --nvtx`, `search --nvtx`,
//! `timeline --nvtx`) restrict their aggregation/filter to GPU events
//! *causally attributable* to NVTX ranges matching a name pattern.
//! The attribution chain is identical across them, so we build the
//! CTE once here.
//!
//! The produced CTE exposes up to five views in the surrounding SQL:
//! - `attributed_kernel_rowids(rowid)`
//! - `attributed_memcpy_rowids(rowid)`
//! - `attributed_memset_rowids(rowid)`
//! - `attributed_sync_rowids(rowid)`
//! - `attributed_runtime_rowids(rowid)`
//!
//! Each lists table rowids of events whose `correlationId` (or, for
//! Runtime, full-interval containment in an NVTX range on the same
//! `globalTid`) traces back to a host thread inside a matching NVTX
//! range. Same walk `slices` uses, distilled to id lists for use as
//! `WHERE t.rowid IN (...)` filters. Only the views the caller
//! actually asked for are materialised.
//!
//! ## How `attributed_runtime` is computed (sidecar)
//!
//! The shared `<trace>.veloq/nvtx-parent.parquet` sidecar stores, for
//! every attributed runtime row, the full outermost→innermost chain
//! of enclosing NVTX ranges. Forward attribution reads the sidecar
//! once and `UNNEST`s the chain so each runtime row contributes one
//! `(nvtx_rowid, nvtx_name)` per enclosing range. The pattern filter
//! (`WHERE nvtx_name LIKE ? ESCAPE '\'`) then picks any runtime row
//! whose chain contains a matching name — including outer scopes,
//! which the innermost-only path would have missed.
//!
//! The sidecar pays the expensive containment walk once per trace at
//! build time, so the forward CTE shrinks to a
//! `read_parquet(...) UNNEST WHERE LIKE` instead of a per-call
//! containment join against `NVTX_EVENTS × CUPTI_ACTIVITY_KIND_RUNTIME`.
//! Downstream `attributed_<kind>_rowids` CTEs (which join
//! `attributed_runtime × ctx_for_pid × <cupti-kind>`) are unchanged.

use crate::EventKind;
use anyhow::{Context, Result};
use duckdb::types::Value;
use veloq_nsys_data::{Trace, runtime_nvtx_parent};

/// Whether a per-kind subquery should be constrained by NVTX
/// attribution. Replaces `nvtx_attributed: bool` parameters that left
/// call sites looking like `per_kind_subquery(kind, abs_window, true)`
/// with no hint at what `true` meant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvtxScope {
    /// Don't constrain — return all events for the kind.
    None,
    /// Constrain to events whose `rowid` appears in the attributed view
    /// for this kind (see [`KERNEL_VIEW`], [`MEMCPY_VIEW`],
    /// [`MEMSET_VIEW`]). The CTE built by [`build`] must already be in
    /// the surrounding SQL's `WITH` block.
    Attributed,
}

impl NvtxScope {
    pub fn is_attributed(self) -> bool {
        matches!(self, Self::Attributed)
    }
}

/// Names of the row-id views produced by the CTE. Callers splice these
/// into their own SQL with `WHERE t.rowid IN (SELECT rowid FROM <view>)`.
pub const KERNEL_VIEW: &str = "attributed_kernel_rowids";
pub const MEMCPY_VIEW: &str = "attributed_memcpy_rowids";
pub const MEMSET_VIEW: &str = "attributed_memset_rowids";
pub const SYNC_VIEW: &str = "attributed_sync_rowids";
pub const RUNTIME_VIEW: &str = "attributed_runtime_rowids";

/// Tables every attribution path needs regardless of kind. The
/// sidecar build also reads them; we preflight here so callers see
/// a structured error instead of a parquet/SQL surprise later.
const CORE_PREREQ_TABLES: &[&str] = &["NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_RUNTIME"];

/// Additional table needed when a GPU-side kind is in scope. Without
/// `TARGET_INFO_CUDA_CONTEXT_INFO` the sidecar build can't
/// resolve `(device, context)` per attributed runtime row, so GPU
/// rows would silently fall into the no-match path. Bail explicitly
/// rather than degrade silently. Pure runtime requests don't need
/// the bridge.
const GPU_PREREQ_TABLE: &str = "TARGET_INFO_CUDA_CONTEXT_INFO";

/// Result of building the attribution CTE.
pub struct AttributionCte {
    /// CTE body **without** the leading `WITH` keyword and with no
    /// trailing comma. The caller composes the final SQL like:
    /// `format!("WITH {cte}, my_other_cte AS (...) SELECT ...")` or
    /// `format!("WITH {cte} SELECT ...")` when there are no extra CTEs.
    pub body: String,
    /// Positional parameters in bind order. Always: the LIKE-glob
    /// for the pattern, followed (if `pattern_only` is `false`) by
    /// the NVTX range time-window bounds.
    pub params: Vec<Value>,
}

/// Build the CTE for an NVTX attribution scope.
///
/// `pattern` is a shell-style glob (`*`/`?`) against the NVTX range
/// name (`COALESCE(text, StringIds.value)`). It's escape-safe via
/// `crate::search_glob_to_like`.
///
/// `kinds` enumerates the GPU kinds the caller will need rowid views for.
/// Only those GPU `attributed_<kind>_rowids` materialised views are
/// emitted (so a trace missing e.g. `CUPTI_ACTIVITY_KIND_MEMSET` won't
/// fail when the caller never asked for memset).
///
/// `trace` is consulted via [`Trace::table_exists`] for each
/// prerequisite plus the kinds the caller actually requested; this
/// function returns an error if any prerequisite is missing or if the
/// caller's requested kinds all lack a backing table.
pub fn build(pattern: &str, kinds: &[EventKind], trace: &Trace) -> Result<AttributionCte> {
    for t in CORE_PREREQ_TABLES {
        if !trace.table_exists(t) {
            anyhow::bail!("--nvtx attribution requires `{t}`, which is not present in this trace");
        }
    }

    let want_kernel =
        kinds.contains(&EventKind::Kernel) && trace.table_exists("CUPTI_ACTIVITY_KIND_KERNEL");
    let want_memcpy =
        kinds.contains(&EventKind::Memcpy) && trace.table_exists("CUPTI_ACTIVITY_KIND_MEMCPY");
    let want_memset =
        kinds.contains(&EventKind::Memset) && trace.table_exists("CUPTI_ACTIVITY_KIND_MEMSET");
    // The attributable set includes Sync and Runtime: both carry
    // correlationId (Sync via the same three-key join the GPU kinds
    // use; Runtime via direct full-interval containment on globalTid).
    // Any of the five kinds is enough to make `--nvtx` meaningful, so a
    // `--type sync --nvtx '*'` request attributes cleanly rather than
    // hitting the bail below.
    let want_sync = kinds.contains(&EventKind::Sync)
        && trace.table_exists("CUPTI_ACTIVITY_KIND_SYNCHRONIZATION");
    let want_runtime = kinds.contains(&EventKind::Runtime);
    if !(want_kernel || want_memcpy || want_memset || want_sync || want_runtime) {
        anyhow::bail!(
            "--nvtx attribution needs at least one attributable kind \
             (kernel/memcpy/memset/sync/runtime); requested kinds don't \
             match any present table"
        );
    }

    // `TARGET_INFO_CUDA_CONTEXT_INFO` is required when a GPU-side
    // kind is in scope: without it the sidecar build can't resolve
    // `(device, context)` per attributed runtime row, so the
    // downstream trio-keyed GPU joins silently miss every kernel.
    // Pure `--type runtime --nvtx <pattern>` doesn't need it (the
    // runtime CTE joins on `runtime_rowid`).
    let needs_ctx_bridge = want_kernel || want_memcpy || want_memset || want_sync;
    if needs_ctx_bridge && !trace.table_exists(GPU_PREREQ_TABLE) {
        anyhow::bail!(
            "--nvtx attribution on kernel/memcpy/memset/sync requires `{GPU_PREREQ_TABLE}`, \
             which is not present in this trace (GPU activity rows cannot be bridged to \
             runtime rows without the context-info table; the lookup would silently miss \
             every kernel)"
        );
    }

    // Build (or warm-load) the shared sidecar, then splice its path
    // into the shared sidecar-expansion CTE. DuckDB pushes the LIKE
    // predicate down through the parquet scan; for a typical 2M-row
    // sidecar that's tens of ms of warm work.
    let sidecar_path = runtime_nvtx_parent::ensure_sidecar(trace)
        .context("building NVTX-parent attribution sidecar for forward filter")?;
    let sidecar_quoted = crate::nvtx_projection::quote_sidecar_path(&sidecar_path);
    let attributed_runtime_cte =
        crate::nvtx_projection::sidecar_expanded_cte("attributed_runtime", &sidecar_quoted);

    let params = vec![Value::Text(crate::search_glob_to_like(pattern))];

    // `matched_runtime` narrows the UNNESTed sidecar to runtime rows
    // whose chain contains a name matching the pattern — outer
    // scopes included. Downstream GPU-side CTEs JOIN it directly on
    // the documented disambiguator `(device, context, correlationId)`;
    // no `ctx_for_pid` bridge at query time (the sidecar carries
    // `device_id` / `context_id` per entry).
    let prefix = format!(
        r#"
        {attributed_runtime_cte},
        matched_runtime AS (
            SELECT runtime_rowid,
                   nvtx_rowid,
                   correlationId,
                   native_pid,
                   device_id,
                   context_id
            FROM attributed_runtime
            WHERE nvtx_name LIKE ? ESCAPE '\'
        )"#
    );

    // `AS MATERIALIZED` is load-bearing: DuckDB's default is to inline
    // CTEs, duplicating this parameterized subquery through multiple
    // references and risking corrupted parameter slots. Materialising
    // forces single evaluation + cached referencing.
    let kernel_cte = r#",
        attributed_kernel_rowids AS MATERIALIZED (
            SELECT DISTINCT k.rowid AS rowid
            FROM matched_runtime mr
            JOIN nsight.CUPTI_ACTIVITY_KIND_KERNEL k
              ON k.correlationId              = mr.correlationId
             AND CAST(k.deviceId  AS INTEGER) = mr.device_id
             AND CAST(k.contextId AS BIGINT)  = mr.context_id
        )"#;
    let memcpy_cte = r#",
        attributed_memcpy_rowids AS MATERIALIZED (
            SELECT DISTINCT t.rowid AS rowid
            FROM matched_runtime mr
            JOIN nsight.CUPTI_ACTIVITY_KIND_MEMCPY t
              ON t.correlationId              = mr.correlationId
             AND CAST(t.deviceId  AS INTEGER) = mr.device_id
             AND CAST(t.contextId AS BIGINT)  = mr.context_id
        )"#;
    let memset_cte = r#",
        attributed_memset_rowids AS MATERIALIZED (
            SELECT DISTINCT t.rowid AS rowid
            FROM matched_runtime mr
            JOIN nsight.CUPTI_ACTIVITY_KIND_MEMSET t
              ON t.correlationId              = mr.correlationId
             AND CAST(t.deviceId  AS INTEGER) = mr.device_id
             AND CAST(t.contextId AS BIGINT)  = mr.context_id
        )"#;
    // Sync attribution joins on the same trio as the other GPU
    // kinds — sync rows carry deviceId/contextId from CUPTI
    // alongside correlationId.
    let sync_cte = r#",
        attributed_sync_rowids AS MATERIALIZED (
            SELECT DISTINCT t.rowid AS rowid
            FROM matched_runtime mr
            JOIN nsight.CUPTI_ACTIVITY_KIND_SYNCHRONIZATION t
              ON t.correlationId              = mr.correlationId
             AND CAST(t.deviceId  AS INTEGER) = mr.device_id
             AND CAST(t.contextId AS BIGINT)  = mr.context_id
        )"#;
    // Runtime attribution is the simplest of the five: matched_runtime
    // already carries the runtime row id; just project it. DISTINCT
    // collapses duplicates produced by the UNNEST (same runtime row
    // can match multiple enclosing names).
    let runtime_cte = r#",
        attributed_runtime_rowids AS MATERIALIZED (
            SELECT DISTINCT runtime_rowid AS rowid FROM matched_runtime
        )"#;

    let mut body = prefix;
    if want_kernel {
        body.push_str(kernel_cte);
    }
    if want_memcpy {
        body.push_str(memcpy_cte);
    }
    if want_memset {
        body.push_str(memset_cte);
    }
    if want_sync {
        body.push_str(sync_cte);
    }
    if want_runtime {
        body.push_str(runtime_cte);
    }

    Ok(AttributionCte { body, params })
}

/// Filter clause that restricts a kernel/memcpy/memset subquery to
/// attributed rowids. Caller wraps in their own WHERE/AND scaffolding.
pub fn filter_clause(view_name: &str, alias: &str) -> String {
    format!("{alias}.rowid IN (SELECT rowid FROM {view_name})")
}

/// True iff `kind` carries a path to NVTX attribution. Kernel /
/// Memcpy / Memset / Sync attribute via the
/// `(correlationId, device_id, context_id)` join through
/// `attributed_runtime` + `ctx_for_pid`; Runtime attributes via
/// full-interval containment on `globalTid` directly. All other
/// kinds — Osrt (no correlationId), Nvtx (the source, not the
/// target), Graph / GraphNode / GraphEvent / CudaEvent / Overhead /
/// CpuSample — are *not* attributable in v1 of the attribution
/// model. Shared between `stats` and `search` so the implicit
/// narrowing of `--type all` + `--nvtx` stays consistent across
/// both verbs.
pub fn is_attributable(kind: EventKind) -> bool {
    matches!(
        kind,
        EventKind::Kernel
            | EventKind::Memcpy
            | EventKind::Memset
            | EventKind::Sync
            | EventKind::Runtime
    )
}
