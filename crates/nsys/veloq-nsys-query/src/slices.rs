//! `veloq slices --pattern <glob>` — NVTX-range attribution views.
//!
//! The default view returns one row per NVTX range matching `pattern`:
//! - **cpu**: the raw NVTX timestamps (host thread wall-time)
//! - **gpu_attributed**: per-(device, stream) bounding boxes of the GPU
//!   work *causally attributable* to launches issued inside the range
//!
//! `slices --aggregate` keeps the same attribution walk but aggregates
//! matching range instances into scope-level distribution rows.
//!
//! The attribution walks:
//!   NVTX range R on tid T, [s, e]
//!     → runtime API rows in [s, e] with globalTid=T  (correlationId, native_pid)
//!     → kernel / memcpy / memset rows with matching (device, context, correlationId)
//!
//! The (device, context, correlationId) triple is what disambiguates
//! raw `correlationId` across processes — see
//! `veloq-nsys-data::correlation` for the same logic in single-event form.

use crate::query_sql::{
    exec::{SqlLabel, query_rows, query_rows_with_context},
    sort::order_by,
};
use crate::{NsysQueryError, NsysQueryResult, RowId, row_id::EventKind};
use duckdb::types::Value;
use serde::Serialize;
use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use veloq_core::{Direction, NameFilterRef, SortKeyDef, SortKeySpec, SortSpec, time::TimeWindow};
use veloq_nsys_data::{NvtxNesting, Trace, runtime_nvtx_parent};
use veloq_query::duckdb::list as duckdb_list;
use veloq_query::sql::{name, total_matched_bigint_expr, window};

// Per-kind CTE body for the NVTX projection lives in
// `crate::nvtx_projection::gpu_kind_cte` so instance and aggregate views
// share the same attribution SQL.
use crate::nvtx_projection::gpu_kind_cte;

const SLICES_INSTANCE_SQL: &str = "instance";
const SLICES_AGGREGATE_SQL: &str = "aggregate";

#[derive(Debug, Clone)]
pub struct SlicesRequest {
    /// Glob-style NVTX range name (`*`/`?`). `None` matches every
    /// NVTX range. Mutually exclusive with `name_regex`.
    pub name: Option<String>,
    /// Regex NVTX range name (DuckDB `regexp_matches`, PCRE-ish).
    /// Mutually exclusive with `name`.
    pub name_regex: Option<String>,
    pub time_window: Option<TimeWindow>,
    /// Sort spec. `None` falls back to `start` ascending — chronological,
    /// which is what iter-to-iter comparison wants by default.
    pub sort: Option<SortSpec>,
    /// Response view. Default is one row per matched NVTX range
    /// instance; aggregate view groups matching instances by name/path.
    pub view: SlicesView,
    /// Aggregate grouping axis. Ignored by the default instance view.
    pub group_by: SlicesAggregateGroupBy,
    pub limit: usize,
    /// Scope the gpu_attributed attribution to one
    /// CUDA device. `None` means "every device" — which on a
    /// multi-device trace must be paired with an `--all-devices`
    /// opt-in upstream (the scope resolver in `veloq-nsys-data::scope`
    /// refuses otherwise).
    pub device: Option<i32>,
    /// Plain stream filter on the gpu_attributed attribution.
    pub stream: Option<i64>,
    /// Cross-axis bridge from `--device`: the native_pid that ran on
    /// the chosen device, looked up by the resolver via
    /// `TARGET_INFO_CUDA_CONTEXT_INFO`. When `Some`, the NVTX-side
    /// filter narrows `matched_nvtx` to ranges whose `globalTid`
    /// high-24-bits equal this pid — the TP-replica deduplication.
    /// Named `native_pid` (OS-level pid) rather than `rank` (the
    /// distributed-runtime cohort index from `RANK` / `SLURM_PROCID` /
    /// `MPI_COMM_WORLD_RANK` env vars), which veloq does not currently
    /// resolve.
    pub native_pid: Option<i64>,
}

impl Default for SlicesRequest {
    fn default() -> Self {
        Self {
            name: None,
            name_regex: None,
            time_window: None,
            sort: None,
            view: SlicesView::Instance,
            group_by: SlicesAggregateGroupBy::Name,
            limit: 100,
            device: None,
            stream: None,
            native_pid: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SlicesView {
    /// One row per matched NVTX range instance.
    #[default]
    Instance,
    /// One row per aggregated NVTX scope group.
    Aggregate,
}

impl SlicesView {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Instance => "instance",
            Self::Aggregate => "aggregate",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SlicesAggregateGroupBy {
    /// One row per NVTX leaf name.
    #[default]
    Name,
    /// One row per full NVTX hierarchy path.
    Path,
}

impl SlicesAggregateGroupBy {
    pub fn parse(s: &str) -> NsysQueryResult<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "name" | "leaf" | "nvtx-name" | "nvtx_name" => Ok(Self::Name),
            "path" | "nvtx-path" | "nvtx_path" => Ok(Self::Path),
            other => Err(NsysQueryError::slices_unknown_group_by(other)),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Name => "name",
            Self::Path => "path",
        }
    }
}

impl fmt::Display for SlicesAggregateGroupBy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Sort axes `slices` supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortKey {
    /// CPU range start time (default ASC — chronological).
    Start,
    /// CPU range duration.
    CpuDuration,
    /// Sum of attributed kernel ns across all streams in the slice.
    AttributedKernel,
    /// Sum of attributed kernel + memcpy + memset ns.
    AttributedTotal,
    /// NVTX range name.
    Name,
}

impl SortKeyDef for SortKey {
    fn specs() -> &'static [SortKeySpec<Self>] {
        &[
            SortKeySpec {
                variant: SortKey::Start,
                canonical: "start",
                aliases: &["time", "cpu_start"],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: SortKey::CpuDuration,
                canonical: "cpu_duration",
                aliases: &["cpu_dur", "duration", "dur"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SortKey::AttributedKernel,
                canonical: "attributed_kernel",
                aliases: &["kernel_ns", "kernel"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SortKey::AttributedTotal,
                canonical: "attributed_total",
                aliases: &["attributed", "total"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SortKey::Name,
                canonical: "name",
                aliases: &[],
                default_dir: Direction::Asc,
            },
        ]
    }
}

impl SortKey {
    /// Column name in the `per_range` CTE.
    fn column(self) -> &'static str {
        match self {
            Self::Start => "r_start",
            Self::CpuDuration => "cpu_duration_ns",
            Self::AttributedKernel => "attributed_kernel_ns",
            Self::AttributedTotal => "attributed_total_ns",
            Self::Name => "name",
        }
    }
}

fn slices_sort_sql(spec: &SortSpec) -> NsysQueryResult<String> {
    // r_start as tiebreaker for determinism — only kicks in when the
    // primary key truly ties (rare for the aggregate columns; possible
    // for `name`).
    order_by::<SortKey>(
        spec,
        SortKey::column,
        NsysQueryError::slices_sort_invalid,
        "r_start, nvtx_rowid",
    )
}

/// Sort axes `slices --aggregate` supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateSortKey {
    Total,
    Instances,
    P50,
    P99,
    Name,
    Path,
}

impl SortKeyDef for AggregateSortKey {
    fn specs() -> &'static [SortKeySpec<Self>] {
        &[
            SortKeySpec {
                variant: AggregateSortKey::Total,
                canonical: "total",
                aliases: &["attributed_total", "attributed_total_ns", "total_ns", "sum"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: AggregateSortKey::Instances,
                canonical: "instances",
                aliases: &["count"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: AggregateSortKey::P50,
                canonical: "p50",
                aliases: &["p50_ns", "typical"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: AggregateSortKey::P99,
                canonical: "p99",
                aliases: &["p99_ns", "tail"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: AggregateSortKey::Name,
                canonical: "name",
                aliases: &["scope"],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: AggregateSortKey::Path,
                canonical: "path",
                aliases: &["nvtx_path", "nvtx-path"],
                default_dir: Direction::Asc,
            },
        ]
    }
}

impl AggregateSortKey {
    fn column(self) -> &'static str {
        match self {
            Self::Total => "attributed_total_ns",
            Self::Instances => "instances",
            Self::P50 => "p50_ns",
            Self::P99 => "p99_ns",
            Self::Name => "name",
            Self::Path => "path",
        }
    }
}

fn aggregate_sort_sql(spec: &SortSpec) -> NsysQueryResult<String> {
    let order = order_by::<AggregateSortKey>(
        spec,
        AggregateSortKey::column,
        NsysQueryError::slices_sort_invalid,
        "path",
    )?;
    Ok(format!("{order}, name ASC"))
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SlicesResponse {
    /// Always `"correlation"` in v0 — the only attribution model we
    /// surface. A future `--mode overlap` would emit `"overlap"` here.
    pub attribution: &'static str,
    /// Active response view: `"instance"` for the default per-range
    /// view, `"aggregate"` for scope-level distribution rows.
    pub view: &'static str,
    /// Active aggregate group key. Present only when `view = "aggregate"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group_by: Option<&'static str>,
    /// Glob filter that was applied, if any. Echoes `--name`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Regex filter that was applied, if any. Echoes `--name-regex`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name_regex: Option<String>,
    /// Slices returned (after LIMIT).
    pub count: usize,
    /// NVTX ranges matching the name filter before LIMIT was applied.
    pub total_matched: i64,
    /// Resolved `--time-range`, if any (absolute ns). slices uses
    /// **overlap** semantics: an NVTX range qualifies if it
    /// intersects the window, matching `stats`/`search`/`gaps`. The
    /// slice's full CPU bounds are still reported (no clipping). Always
    /// serialised (as `null` when no window was set) for cross-command
    /// schema parity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_window_ns: Option<(i64, i64)>,
    /// Canonical primary table. Rows are one NVTX-attributed slice
    /// in the default view, or one scope aggregate row in aggregate view.
    pub rows: Vec<SlicesRow>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum SlicesRow {
    Instance(Slice),
    Aggregate(SliceAggregate),
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct Slice {
    /// Cross-trace key. `slice|<name>|@<cpu.start_ns>`. Two runs of
    /// the same workload yield matching keys when the NVTX names and
    /// per-iter wall-clock origin line up; agents can pre-normalize
    /// timestamps using envelope `trace_span.origin_ns`.
    pub key: String,
    pub row_id: RowId,
    pub name: String,
    pub cpu: CpuSpan,
    pub gpu_attributed: Vec<GpuStreamSpan>,
    pub attributed_kernel_ns: i64,
    pub attributed_kernel_count: i64,
    pub attributed_memcpy_ns: i64,
    pub attributed_memcpy_count: i64,
    pub attributed_memset_ns: i64,
    pub attributed_memset_count: i64,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SliceAggregate {
    /// Cross-trace key. `scope|<name>` for leaf-name aggregates or
    /// `scope|path:<path>` for full-path aggregates.
    pub key: String,
    pub name: String,
    /// Full NVTX hierarchy path. Populated only in path aggregate mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// Number of NVTX range instances aggregated into this scope row.
    pub instances: i64,
    /// Sum of per-instance attributed GPU total ns.
    pub attributed_total_ns: i64,
    /// Median of per-instance attributed GPU total ns.
    pub p50_ns: f64,
    /// p99 of per-instance attributed GPU total ns.
    pub p99_ns: f64,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct CpuSpan {
    pub global_tid: i64,
    pub start_ns: i64,
    pub end_ns: i64,
    pub duration_ns: i64,
    /// Nesting depth of this NVTX range relative to other ranges on the
    /// same `(global_tid, domain_id)` stack. 0 = outermost. Useful for
    /// filtering "give me only root spans" in iter-to-iter comparison.
    /// `None` when nesting wasn't computed (e.g. trace had no NVTX).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nesting_depth: Option<u8>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct GpuStreamSpan {
    pub device_id: i32,
    pub stream_id: i64,
    pub start_ns: i64,
    pub end_ns: i64,
    pub kernel_ns: i64,
    pub kernel_count: i64,
    pub memcpy_ns: i64,
    pub memcpy_count: i64,
    pub memset_ns: i64,
    pub memset_count: i64,
}

pub fn run<P: AsRef<Path>>(path: P, req: SlicesRequest) -> NsysQueryResult<SlicesResponse> {
    crate::check_limit(req.limit)?;
    let trace = Trace::open(path).map_err(NsysQueryError::trace_open)?;

    // NVTX_EVENTS / RUNTIME / CONTEXT_INFO are all required for the
    // attribution walk to make sense. They're each optional in the NSys
    // export schema; if any is missing, bail with a structured error
    // matching stats/search/`--nvtx`. Returning success-with-empty here
    // would conflate "no NVTX ranges match the pattern" with "this
    // trace was exported without NVTX at all" — those are very different
    // diagnoses for an agent.
    for t in [
        "NVTX_EVENTS",
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        "TARGET_INFO_CUDA_CONTEXT_INFO",
    ] {
        if !trace.table_exists(t) {
            return Err(NsysQueryError::SlicesPrereqTableMissing { table: t });
        }
    }
    let has_kernel = trace.table_exists("CUPTI_ACTIVITY_KIND_KERNEL");
    let has_memcpy = trace.table_exists("CUPTI_ACTIVITY_KIND_MEMCPY");
    let has_memset = trace.table_exists("CUPTI_ACTIVITY_KIND_MEMSET");
    if !(has_kernel || has_memcpy || has_memset) {
        return Err(NsysQueryError::SlicesGpuEventTableMissing);
    }

    // Time window scopes the NVTX RANGES. Inclusion uses **overlap**
    // semantics — a range qualifies if any part of it intersects the
    // window — matching `stats`/`search`/`gaps`. (The earlier
    // "entirely inside" predicate was the odd one out across veloq's
    // four time-windowed commands.) The slice's full CPU bounds are
    // still reported as-is; we don't clip the reporting to the window,
    // just the inclusion.
    let abs_window = trace
        .resolve_window(req.time_window)
        .map_err(NsysQueryError::time_window_resolve)?;

    let name_filter = NameFilterRef::from_optional(req.name.as_deref(), req.name_regex.as_deref())
        .map_err(|_| NsysQueryError::NameFilterConflict)?;
    let mut params: Vec<Value> = Vec::new();
    let name_fragment = name::predicate_or_bound_true("COALESCE(n.text, s.value, '')", name_filter);
    let name_predicate = name_fragment.sql;
    params.extend(name_fragment.params);

    let time_filter = if let Some(fragment) = window::overlap_filter("n", abs_window) {
        // overlap: range_start < window_end AND range_end > window_start
        params.extend(fragment.params);
        format!("AND {}", fragment.sql)
    } else {
        String::new()
    };

    // Rank filter (cross-axis bridge from `--device <N>`): the
    // resolver in `veloq-nsys-data::scope` supplies the native_pid
    // that ran on device N. When set, restrict `matched_nvtx` to NVTX
    // ranges whose host thread belongs to that process — the
    // TP-replica deduplication.
    //
    // Extracting native_pid: `(globalTid >> 24) & 0xFFFFFF` per
    // AGENTS.md's globalTid bit layout (24-bit PID in bits 24..47).
    let host_pid_filter: String = if let Some(pid) = req.native_pid {
        params.push(Value::BigInt(pid));
        "AND ((n.globalTid >> 24) & 16777215) = ?".to_string()
    } else {
        String::new()
    };

    // `--device` filters on the attributed (sidecar) side via
    // `se.device_id`; `--stream` is not carried on the sidecar, so it
    // filters the flattened GPU event rows in the `aggregated` CTE via
    // `e.stream_id` (uniform scope filters). Each binds one
    // positional `?` in device-then-stream order, matching their SQL
    // positions (`attributed_runtime` precedes `aggregated`).
    let mut attrib_filters = String::new();
    if let Some(d) = req.device {
        params.push(Value::Int(d));
        attrib_filters.push_str(" AND se.device_id = ?");
    }
    let mut gpu_stream_filter = String::new();
    if let Some(s) = req.stream {
        params.push(Value::BigInt(s));
        gpu_stream_filter.push_str(" AND e.stream_id = ?");
    }

    if req.view == SlicesView::Aggregate {
        return run_aggregate(
            &trace,
            req,
            abs_window,
            &name_predicate,
            &time_filter,
            &host_pid_filter,
            &attrib_filters,
            &gpu_stream_filter,
            params,
            has_kernel,
            has_memcpy,
            has_memset,
        );
    }

    // Resolve --sort and compute the ORDER BY body for the ranked CTE.
    let sort_spec = req
        .sort
        .clone()
        .unwrap_or_else(|| SortSpec::single("start"));
    let order_by = slices_sort_sql(&sort_spec)?;

    // LIMIT applies to distinct NVTX ranges (via the `ranked` CTE),
    // not the inner per-stream/per-kind sub-rows. `total_matched` is
    // carried separately by the COUNT(*) OVER () window, so no +1 trick.
    params.push(Value::BigInt(req.limit as i64));

    // Build (or warm-load) the shared NVTX-parent sidecar — slices
    // reuses the same runtime→enclosing-NVTX walk that powers
    // `stats --group-by nvtx-parent`, `--nvtx <pattern>`, and
    // `inspect`. The sidecar replaces a per-call containment join
    // against `CUPTI_ACTIVITY_KIND_RUNTIME` with a JOIN against
    // `read_parquet('<sidecar>')` UNNESTed by enclosing-rowid lists.
    let sidecar_path = runtime_nvtx_parent::ensure_sidecar(&trace)
        .map_err(NsysQueryError::nvtx_parent_sidecar_ensure)?;
    let sidecar_quoted = crate::nvtx_projection::quote_sidecar_path(&sidecar_path);
    let sidecar_expanded_cte =
        crate::nvtx_projection::sidecar_expanded_cte("sidecar_expanded", &sidecar_quoted);

    // The query is split into a sequence of CTEs that mirror the
    // attribution walk; the outer SELECT pivots into one row per
    // (range, device, stream, kind) tuple which the Rust side folds.
    //
    // LIMIT runs after the user's `--sort` so e.g. `--sort
    // attributed_kernel --limit 5` actually picks the 5 heaviest
    // ranges (not the 5 chronologically earliest). The `per_range` CTE
    // computes the per-NVTX-range aggregates used as sort keys; the
    // `ranked` CTE applies the sort + limit and assigns `rn` so the
    // final output preserves user order across the per-stream/kind
    // sub-rows.
    //
    // The per-kind CTEs (gpu_kernels/gpu_memcpys/gpu_memsets) are only
    // emitted when their backing table exists — older or stripped traces
    // sometimes lack MEMSET, and unconditional reference would fail
    // with a SQL catalog error before the WHERE filter even ran.
    let mut gpu_event_ctes: Vec<String> = Vec::with_capacity(3);
    let mut gpu_event_unions: Vec<&str> = Vec::with_capacity(3);
    if has_kernel {
        gpu_event_ctes.push(gpu_kind_cte(
            "gpu_kernels",
            "kernel",
            "CUPTI_ACTIVITY_KIND_KERNEL",
        ));
        gpu_event_unions.push("SELECT * FROM gpu_kernels");
    }
    if has_memcpy {
        gpu_event_ctes.push(gpu_kind_cte(
            "gpu_memcpys",
            "memcpy",
            "CUPTI_ACTIVITY_KIND_MEMCPY",
        ));
        gpu_event_unions.push("SELECT * FROM gpu_memcpys");
    }
    if has_memset {
        gpu_event_ctes.push(gpu_kind_cte(
            "gpu_memsets",
            "memset",
            "CUPTI_ACTIVITY_KIND_MEMSET",
        ));
        gpu_event_unions.push("SELECT * FROM gpu_memsets");
    }
    let gpu_kind_ctes = gpu_event_ctes.join(",\n");
    let gpu_events_union = gpu_event_unions.join("\n            UNION ALL ");
    let sql = format!(
        r#"
        WITH matched_nvtx AS (
            SELECT
                n.rowid                          AS nvtx_rowid,
                n.globalTid                      AS tid,
                n.start                          AS r_start,
                n."end"                          AS r_end,
                COALESCE(n.text, s.value, '<unnamed>') AS name
            FROM nsight.NVTX_EVENTS n
            LEFT JOIN nsight.StringIds s ON n.textId = s.id
            WHERE {name_predicate}
              AND n."end" IS NOT NULL
              AND n.globalTid IS NOT NULL
              {time_filter}
              {host_pid_filter}
        ),
        {sidecar_expanded_cte},
        attributed_runtime AS (
            -- Join the shared sidecar's outer→inner-expanded view
            -- against pattern-filtered NVTX ranges. The downstream
            -- GPU projection (`nvtx_projection::gpu_kind_cte`)
            -- reads `nvtx_rowid` / `correlationId` / `device_id` /
            -- `context_id` from this CTE shape.
            SELECT m.nvtx_rowid,
                   se.correlationId,
                   se.native_pid,
                   se.device_id,
                   se.context_id
            FROM matched_nvtx m
            JOIN sidecar_expanded se
              ON se.nvtx_rowid = m.nvtx_rowid
            WHERE 1=1 {attrib_filters}
        ),
        {gpu_kind_ctes},
        gpu_events AS (
            {gpu_events_union}
        ),
        aggregated AS (
            SELECT
                e.nvtx_rowid,
                e.device_id,
                e.stream_id,
                e.kind,
                MIN(e.evt_start) AS gpu_start,
                MAX(e.evt_end)   AS gpu_end,
                CAST(SUM(e.dur) AS BIGINT) AS busy_ns,
                CAST(COUNT(*)   AS BIGINT) AS event_count
            FROM gpu_events e
            WHERE 1=1 {gpu_stream_filter}
            GROUP BY e.nvtx_rowid, e.device_id, e.stream_id, e.kind
        ),
        per_range AS (
            SELECT
                m.nvtx_rowid,
                m.name,
                m.tid,
                m.r_start,
                m.r_end,
                CAST(m.r_end - m.r_start AS BIGINT) AS cpu_duration_ns,
                CAST(COALESCE(SUM(CASE WHEN a.kind = 'kernel' THEN a.busy_ns ELSE 0 END), 0) AS BIGINT) AS attributed_kernel_ns,
                CAST(COALESCE(SUM(CASE WHEN a.kind = 'memcpy' THEN a.busy_ns ELSE 0 END), 0) AS BIGINT) AS attributed_memcpy_ns,
                CAST(COALESCE(SUM(CASE WHEN a.kind = 'memset' THEN a.busy_ns ELSE 0 END), 0) AS BIGINT) AS attributed_memset_ns,
                CAST(COALESCE(SUM(a.busy_ns), 0) AS BIGINT) AS attributed_total_ns
            FROM matched_nvtx m
            LEFT JOIN aggregated a ON a.nvtx_rowid = m.nvtx_rowid
            GROUP BY m.nvtx_rowid, m.name, m.tid, m.r_start, m.r_end
        ),
        ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (ORDER BY {order_by}) AS rn,
                   CAST(COUNT(*) OVER () AS BIGINT)        AS total_matched
            FROM per_range
        ),
        selected AS (
            SELECT * FROM ranked WHERE rn <= ?
        )
        SELECT
            s.nvtx_rowid,
            s.name,
            s.tid,
            s.r_start,
            s.r_end,
            a.device_id,
            a.stream_id,
            a.kind,
            a.gpu_start,
            a.gpu_end,
            a.busy_ns,
            a.event_count,
            s.total_matched
        FROM selected s
        LEFT JOIN aggregated a ON a.nvtx_rowid = s.nvtx_rowid
        ORDER BY s.rn, a.device_id, a.stream_id, a.kind
        "#
    );

    let (slices, total_matched) = hydrate_slice_rows(&trace, &sql, &params)?;

    Ok(SlicesResponse {
        attribution: "correlation",
        view: SlicesView::Instance.as_str(),
        group_by: None,
        name: req.name,
        name_regex: req.name_regex,
        count: slices.len(),
        total_matched,
        time_window_ns: abs_window,
        rows: slices.into_iter().map(SlicesRow::Instance).collect(),
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "SQL builder: every predicate/param slot is positional; an args-struct here would just move the long list one frame outward"
)]
fn run_aggregate(
    trace: &Trace,
    req: SlicesRequest,
    abs_window: Option<(i64, i64)>,
    name_predicate: &str,
    time_filter: &str,
    host_pid_filter: &str,
    attrib_filters: &str,
    gpu_stream_filter: &str,
    mut params: Vec<Value>,
    has_kernel: bool,
    has_memcpy: bool,
    has_memset: bool,
) -> NsysQueryResult<SlicesResponse> {
    let sort_spec = req
        .sort
        .clone()
        .unwrap_or_else(|| SortSpec::single("total"));
    let order_by = aggregate_sort_sql(&sort_spec)?;
    params.push(Value::BigInt(req.limit as i64));

    let sidecar_path = runtime_nvtx_parent::ensure_sidecar(trace)
        .map_err(NsysQueryError::nvtx_parent_sidecar_ensure)?;
    let sidecar_quoted = crate::nvtx_projection::quote_sidecar_path(&sidecar_path);
    let sidecar_expanded_cte =
        crate::nvtx_projection::sidecar_expanded_cte("sidecar_expanded", &sidecar_quoted);

    let (path_join, path_expr, grouped_name_expr, grouped_path_expr, group_by_cols) =
        match req.group_by {
            SlicesAggregateGroupBy::Name => (
                "",
                "CAST(NULL AS VARCHAR)",
                "name",
                "CAST(NULL AS VARCHAR)",
                "name",
            ),
            SlicesAggregateGroupBy::Path => {
                veloq_nsys_data::nvtx_tree::ensure_sidecar(trace)
                    .map_err(NsysQueryError::nvtx_tree_load)?;
                (
                    "LEFT JOIN nsight.nvtx_tree nt ON nt.range_id = n.rowid",
                    "COALESCE(nt.path, COALESCE(n.text, s.value, '<unnamed>'))",
                    "arbitrary(name)",
                    "path",
                    "path",
                )
            }
        };

    let mut gpu_event_ctes: Vec<String> = Vec::with_capacity(3);
    let mut gpu_event_unions: Vec<&str> = Vec::with_capacity(3);
    if has_kernel {
        gpu_event_ctes.push(gpu_kind_cte(
            "gpu_kernels",
            "kernel",
            "CUPTI_ACTIVITY_KIND_KERNEL",
        ));
        gpu_event_unions.push("SELECT * FROM gpu_kernels");
    }
    if has_memcpy {
        gpu_event_ctes.push(gpu_kind_cte(
            "gpu_memcpys",
            "memcpy",
            "CUPTI_ACTIVITY_KIND_MEMCPY",
        ));
        gpu_event_unions.push("SELECT * FROM gpu_memcpys");
    }
    if has_memset {
        gpu_event_ctes.push(gpu_kind_cte(
            "gpu_memsets",
            "memset",
            "CUPTI_ACTIVITY_KIND_MEMSET",
        ));
        gpu_event_unions.push("SELECT * FROM gpu_memsets");
    }
    let gpu_kind_ctes = gpu_event_ctes.join(",\n");
    let gpu_events_union = gpu_event_unions.join("\n            UNION ALL ");

    let sql = format!(
        r#"
        WITH matched_nvtx AS (
            SELECT
                n.rowid                          AS nvtx_rowid,
                n.globalTid                      AS tid,
                n.start                          AS r_start,
                n."end"                          AS r_end,
                COALESCE(n.text, s.value, '<unnamed>') AS name,
                {path_expr}                      AS path
            FROM nsight.NVTX_EVENTS n
            LEFT JOIN nsight.StringIds s ON n.textId = s.id
            {path_join}
            WHERE {name_predicate}
              AND n."end" IS NOT NULL
              AND n.globalTid IS NOT NULL
              {time_filter}
              {host_pid_filter}
        ),
        {sidecar_expanded_cte},
        attributed_runtime AS (
            SELECT m.nvtx_rowid,
                   se.correlationId,
                   se.native_pid,
                   se.device_id,
                   se.context_id
            FROM matched_nvtx m
            JOIN sidecar_expanded se
              ON se.nvtx_rowid = m.nvtx_rowid
            WHERE 1=1 {attrib_filters}
        ),
        {gpu_kind_ctes},
        gpu_events AS (
            {gpu_events_union}
        ),
        aggregated AS (
            SELECT
                e.nvtx_rowid,
                CAST(SUM(e.dur) AS BIGINT) AS busy_ns
            FROM gpu_events e
            WHERE 1=1 {gpu_stream_filter}
            GROUP BY e.nvtx_rowid
        ),
        per_range AS (
            SELECT
                m.nvtx_rowid,
                m.name,
                m.path,
                CAST(COALESCE(a.busy_ns, 0) AS BIGINT) AS attributed_total_ns
            FROM matched_nvtx m
            LEFT JOIN aggregated a ON a.nvtx_rowid = m.nvtx_rowid
        ),
        per_group AS (
            SELECT
                {grouped_name_expr} AS name,
                {grouped_path_expr} AS path,
                CAST(COUNT(*) AS BIGINT) AS instances,
                CAST(SUM(attributed_total_ns) AS BIGINT) AS attributed_total_ns,
                QUANTILE_CONT(attributed_total_ns, 0.50) AS p50_ns,
                QUANTILE_CONT(attributed_total_ns, 0.99) AS p99_ns
            FROM per_range
            GROUP BY {group_by_cols}
        ),
        ranked AS (
            SELECT *,
                   {total_matched}
            FROM per_group
        )
        SELECT
            name, path, instances, attributed_total_ns, p50_ns, p99_ns, total_matched
        FROM ranked
        ORDER BY {order_by}
        LIMIT ?
        "#,
        total_matched = total_matched_bigint_expr(),
    );

    let (out, total_matched) =
        hydrate_aggregate_rows(trace.conn(), &sql, &params, req.limit, req.group_by)?;

    Ok(SlicesResponse {
        attribution: "correlation",
        view: SlicesView::Aggregate.as_str(),
        group_by: Some(req.group_by.as_str()),
        name: req.name,
        name_regex: req.name_regex,
        count: out.len(),
        total_matched,
        time_window_ns: abs_window,
        rows: out.into_iter().map(SlicesRow::Aggregate).collect(),
    })
}

/// Run the prepared `sql`, fold the (range × device × stream × kind)
/// rows into per-range `Slice` structs, and recover `total_matched`
/// from the SQL-side window function. Carved out of `run` so the
/// fold plus NVTX-nesting lookup is reviewable in isolation; bind
/// order and SQL assembly stay in the caller.
///
/// The fold preserves input order via the auxiliary `order` Vec: SQL
/// returns rows sorted by `rn ASC, device_id, stream_id, kind`, so
/// the first time a `nvtx_rowid` appears it dictates the slice's
/// position in the response.
fn hydrate_aggregate_rows(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
    limit: usize,
    group_by: SlicesAggregateGroupBy,
) -> NsysQueryResult<(Vec<SliceAggregate>, i64)> {
    let rows = query_rows(
        conn,
        sql,
        params,
        SqlLabel::new("slices", SLICES_AGGREGATE_SQL),
        slice_aggregate_sql_row,
    )?;
    let (out, total_matched) = duckdb_list::split_rows_and_total::<i64, _, _, _>(
        rows,
        duckdb_list::TotalCarrier::First,
        |row| row.total_matched,
        duckdb_list::infallible_count_error,
        |row| Ok(slice_aggregate_from_sql_row(row, group_by)),
    )?;
    debug_assert!(out.len() <= limit);
    Ok((out, total_matched))
}

struct SliceAggregateSqlRow {
    name: String,
    path: Option<String>,
    instances: i64,
    attributed_total_ns: i64,
    p50_ns: f64,
    p99_ns: f64,
    total_matched: i64,
}

fn slice_aggregate_sql_row(row: &duckdb::Row<'_>) -> Result<SliceAggregateSqlRow, duckdb::Error> {
    Ok(SliceAggregateSqlRow {
        name: row.get(0)?,
        path: row.get(1)?,
        instances: row.get(2)?,
        attributed_total_ns: row.get(3)?,
        p50_ns: row.get(4)?,
        p99_ns: row.get(5)?,
        total_matched: row.get(6)?,
    })
}

fn slice_aggregate_from_sql_row(
    row: SliceAggregateSqlRow,
    group_by: SlicesAggregateGroupBy,
) -> SliceAggregate {
    let key = match (group_by, row.path.as_deref()) {
        (SlicesAggregateGroupBy::Path, Some(p)) => format!("scope|path:{p}"),
        _ => format!("scope|{}", row.name),
    };
    SliceAggregate {
        key,
        name: row.name,
        path: row.path,
        instances: row.instances,
        attributed_total_ns: row.attributed_total_ns,
        p50_ns: row.p50_ns,
        p99_ns: row.p99_ns,
    }
}

fn hydrate_slice_rows(
    trace: &Trace,
    sql: &str,
    params: &[Value],
) -> NsysQueryResult<(Vec<Slice>, i64)> {
    hydrate_slice_rows_with_nesting(trace.conn(), sql, params, || {
        trace
            .nvtx_nesting()
            .map_err(NsysQueryError::nvtx_nesting_load)
    })
}

fn hydrate_slice_rows_with_nesting<F>(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
    load_nesting: F,
) -> NsysQueryResult<(Vec<Slice>, i64)>
where
    F: FnOnce() -> NsysQueryResult<NvtxNesting>,
{
    let (rows, nesting) = query_rows_with_context(
        conn,
        sql,
        params,
        SqlLabel::new("slices", SLICES_INSTANCE_SQL),
        load_nesting,
        |row, _nesting| slice_sql_row(row),
    )?;

    // Fold (range, device, stream, kind) rows into per-range structs.
    // Keyed by nvtx_rowid → Slice; per-stream sub-key (device, stream).
    let mut slices_by_id: HashMap<i64, SliceBuilder> = HashMap::new();
    let mut order: Vec<i64> = Vec::new();
    let mut last_row_id: Option<i64> = None;

    let total_matched =
        duckdb_list::total_matched::<i64, _>(&rows, duckdb_list::TotalCarrier::Last, |row| {
            row.total_matched
        })
        .map_err(duckdb_list::infallible_count_error)?;
    for row in rows {
        if last_row_id != Some(row.nvtx_rowid) {
            order.push(row.nvtx_rowid);
            last_row_id = Some(row.nvtx_rowid);
        }

        let builder = slices_by_id.entry(row.nvtx_rowid).or_insert_with(|| {
            SliceBuilder::new(
                row.nvtx_rowid,
                row.name.clone(),
                row.tid,
                row.r_start,
                row.r_end,
                nesting.get(&row.nvtx_rowid).map(|e| e.depth),
            )
        });

        if let (Some(dev), Some(stream), Some(kind)) =
            (row.device_id, row.stream_id, row.kind.as_deref())
        {
            builder.add_aggregate(
                dev,
                stream,
                kind,
                row.gpu_start.unwrap_or(0),
                row.gpu_end.unwrap_or(0),
                row.busy_ns.unwrap_or(0),
                row.event_count.unwrap_or(0),
            );
        }
    }

    // SQL already enforced rn <= limit; fold into Slices. Every id in
    // `order` was inserted into `slices_by_id` while reading the
    // rows, so the lookup can't legitimately miss — but we still
    // route a missing entry through a structured error rather than
    // panicking, in case a future refactor drifts the invariant.
    let mut slices: Vec<Slice> = Vec::with_capacity(order.len());
    for id in order {
        let Some(builder) = slices_by_id.remove(&id) else {
            return Err(NsysQueryError::internal_slice_builder_missing(id));
        };
        slices.push(builder.build());
    }

    Ok((slices, total_matched))
}

struct SliceSqlRow {
    nvtx_rowid: i64,
    name: String,
    tid: i64,
    r_start: i64,
    r_end: i64,
    device_id: Option<i32>,
    stream_id: Option<i64>,
    kind: Option<String>,
    gpu_start: Option<i64>,
    gpu_end: Option<i64>,
    busy_ns: Option<i64>,
    event_count: Option<i64>,
    total_matched: i64,
}

fn slice_sql_row(row: &duckdb::Row<'_>) -> Result<SliceSqlRow, duckdb::Error> {
    Ok(SliceSqlRow {
        nvtx_rowid: row.get(0)?,
        name: row.get(1)?,
        tid: row.get(2)?,
        r_start: row.get(3)?,
        r_end: row.get(4)?,
        device_id: row.get(5)?,
        stream_id: row.get(6)?,
        kind: row.get(7)?,
        gpu_start: row.get(8)?,
        gpu_end: row.get(9)?,
        busy_ns: row.get(10)?,
        event_count: row.get(11)?,
        total_matched: row.get(12)?,
    })
}

// ---- internal builder -----------------------------------------------------

struct SliceBuilder {
    row_id: i64,
    name: String,
    tid: i64,
    r_start: i64,
    r_end: i64,
    nesting_depth: Option<u8>,
    per_stream: HashMap<(i32, i64), StreamAcc>,
}

#[derive(Default)]
struct StreamAcc {
    gpu_start: i64,
    gpu_end: i64,
    kernel_ns: i64,
    kernel_count: i64,
    memcpy_ns: i64,
    memcpy_count: i64,
    memset_ns: i64,
    memset_count: i64,
    seen: bool,
}

impl SliceBuilder {
    fn new(
        row_id: i64,
        name: String,
        tid: i64,
        r_start: i64,
        r_end: i64,
        nesting_depth: Option<u8>,
    ) -> Self {
        Self {
            row_id,
            name,
            tid,
            r_start,
            r_end,
            nesting_depth,
            per_stream: HashMap::new(),
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "per-(device, stream, kind) aggregate row hydration: each column is a positional slot; an args-struct here would just move the column list one frame outward"
    )]
    fn add_aggregate(
        &mut self,
        device_id: i32,
        stream_id: i64,
        kind: &str,
        gpu_start: i64,
        gpu_end: i64,
        busy_ns: i64,
        event_count: i64,
    ) {
        let acc = self.per_stream.entry((device_id, stream_id)).or_default();
        if !acc.seen {
            acc.gpu_start = gpu_start;
            acc.gpu_end = gpu_end;
            acc.seen = true;
        } else {
            acc.gpu_start = acc.gpu_start.min(gpu_start);
            acc.gpu_end = acc.gpu_end.max(gpu_end);
        }
        match kind {
            "kernel" => {
                acc.kernel_ns = acc.kernel_ns.saturating_add(busy_ns);
                acc.kernel_count = acc.kernel_count.saturating_add(event_count);
            }
            "memcpy" => {
                acc.memcpy_ns = acc.memcpy_ns.saturating_add(busy_ns);
                acc.memcpy_count = acc.memcpy_count.saturating_add(event_count);
            }
            "memset" => {
                acc.memset_ns = acc.memset_ns.saturating_add(busy_ns);
                acc.memset_count = acc.memset_count.saturating_add(event_count);
            }
            _ => {}
        }
    }

    fn build(self) -> Slice {
        let mut gpu_attributed: Vec<GpuStreamSpan> = self
            .per_stream
            .into_iter()
            .map(|((device_id, stream_id), acc)| GpuStreamSpan {
                device_id,
                stream_id,
                start_ns: acc.gpu_start,
                end_ns: acc.gpu_end,
                kernel_ns: acc.kernel_ns,
                kernel_count: acc.kernel_count,
                memcpy_ns: acc.memcpy_ns,
                memcpy_count: acc.memcpy_count,
                memset_ns: acc.memset_ns,
                memset_count: acc.memset_count,
            })
            .collect();
        gpu_attributed.sort_by_key(|s| (s.device_id, s.stream_id));

        let mut tot_kernel_ns = 0i64;
        let mut tot_kernel_count = 0i64;
        let mut tot_memcpy_ns = 0i64;
        let mut tot_memcpy_count = 0i64;
        let mut tot_memset_ns = 0i64;
        let mut tot_memset_count = 0i64;
        for s in &gpu_attributed {
            tot_kernel_ns += s.kernel_ns;
            tot_kernel_count += s.kernel_count;
            tot_memcpy_ns += s.memcpy_ns;
            tot_memcpy_count += s.memcpy_count;
            tot_memset_ns += s.memset_ns;
            tot_memset_count += s.memset_count;
        }

        let key = format!("slice|{}|@{}", self.name, self.r_start);
        Slice {
            key,
            row_id: RowId::new(EventKind::Nvtx, self.row_id),
            name: self.name,
            cpu: CpuSpan {
                global_tid: self.tid,
                start_ns: self.r_start,
                end_ns: self.r_end,
                duration_ns: self.r_end - self.r_start,
                nesting_depth: self.nesting_depth,
            },
            gpu_attributed,
            attributed_kernel_ns: tot_kernel_ns,
            attributed_kernel_count: tot_kernel_count,
            attributed_memcpy_ns: tot_memcpy_ns,
            attributed_memcpy_count: tot_memcpy_count,
            attributed_memset_ns: tot_memset_ns,
            attributed_memset_count: tot_memset_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use veloq_core::VeloqDiagnostic;

    fn empty_nesting() -> NsysQueryResult<NvtxNesting> {
        Ok(HashMap::new())
    }

    #[test]
    fn hydrate_aggregate_rows_prepare_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;

        let err = match hydrate_aggregate_rows(
            &conn,
            "SELECT * FROM",
            &[],
            1,
            SlicesAggregateGroupBy::Name,
        ) {
            Ok(rows) => anyhow::bail!(
                "malformed slices aggregate SQL should not hydrate successfully: {} rows",
                rows.0.len()
            ),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-prepare");
        assert_eq!(
            err.sql_parts(),
            Some(("slices", crate::SqlPhase::Prepare, SLICES_AGGREGATE_SQL))
        );
        Ok(())
    }

    #[test]
    fn hydrate_aggregate_rows_query_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = "SELECT ? AS name";

        let err = match hydrate_aggregate_rows(&conn, sql, &[], 1, SlicesAggregateGroupBy::Name) {
            Ok(rows) => anyhow::bail!(
                "unbound slices aggregate SQL should not hydrate successfully: {} rows",
                rows.0.len()
            ),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-query");
        assert!(matches!(
            err,
            crate::NsysQueryError::Sql {
                phase: crate::SqlPhase::Query,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn hydrate_aggregate_rows_read_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = "SELECT \
                   'scope' AS name, \
                   NULL::VARCHAR AS path, \
                   'not-instances' AS instances";

        let err = match hydrate_aggregate_rows(&conn, sql, &[], 1, SlicesAggregateGroupBy::Name) {
            Ok(rows) => anyhow::bail!(
                "malformed slices aggregate row should not hydrate successfully: {} rows",
                rows.0.len()
            ),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-read");
        assert!(matches!(
            err,
            crate::NsysQueryError::Sql {
                phase: crate::SqlPhase::Read,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn hydrate_slice_rows_prepare_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;

        let err = match hydrate_slice_rows_with_nesting(&conn, "SELECT * FROM", &[], empty_nesting)
        {
            Ok(rows) => anyhow::bail!(
                "malformed slices instance SQL should not hydrate successfully: {} rows",
                rows.0.len()
            ),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-prepare");
        assert_eq!(
            err.sql_parts(),
            Some(("slices", crate::SqlPhase::Prepare, SLICES_INSTANCE_SQL))
        );
        Ok(())
    }

    #[test]
    fn hydrate_slice_rows_query_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = "SELECT ? AS nvtx_rowid";

        let err = match hydrate_slice_rows_with_nesting(&conn, sql, &[], empty_nesting) {
            Ok(rows) => anyhow::bail!(
                "unbound slices instance SQL should not hydrate successfully: {} rows",
                rows.0.len()
            ),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-query");
        assert!(matches!(
            err,
            crate::NsysQueryError::Sql {
                phase: crate::SqlPhase::Query,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn hydrate_slice_rows_read_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = "SELECT 'not-a-rowid' AS nvtx_rowid";

        let err = match hydrate_slice_rows_with_nesting(&conn, sql, &[], empty_nesting) {
            Ok(rows) => anyhow::bail!(
                "malformed slices instance row should not hydrate successfully: {} rows",
                rows.0.len()
            ),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-read");
        assert!(matches!(
            err,
            crate::NsysQueryError::Sql {
                phase: crate::SqlPhase::Read,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn internal_slice_builder_missing_has_typed_code() {
        let err = crate::NsysQueryError::internal_slice_builder_missing(42);

        assert_eq!(err.code().as_str(), "nsys.internal.slice-builder-missing");
        assert!(matches!(
            err,
            crate::NsysQueryError::InternalSliceBuilderMissing { row_id: 42 }
        ));
    }
}
