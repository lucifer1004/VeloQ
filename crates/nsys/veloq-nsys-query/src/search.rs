//! `veloq search <trace> ...` — filter events into a list of `row_id`s
//! plus a few headline columns. Designed as the `inspect` entry-point.

use duckdb::types::Value;
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;
use veloq_core::{
    Direction, NameFilterRef, SortKeyDef, SortKeySpec, SortSpec,
    time::{DurationFilter, TimeWindow},
};
use veloq_nsys_data::NvtxNesting;

use crate::column_map::{self, ColumnMap, maybe_col};
use crate::event_ref::{
    EventRefBase, EventRefKernel, EventRefMemcpy, EventRefMemset, EventRefNvtx,
};
use crate::query_sql::{
    event_semantics::EventSemantics,
    exec::{SqlLabel, query_rows},
    sort::order_by,
};
use crate::{EventKind, EventRef, KindFilter, NsysQueryError, NsysQueryResult, RowId};
use veloq_query::duckdb::list as duckdb_list;
use veloq_query::sql::{
    name, total_matched_bigint_expr, where_clause as build_where_clause, window,
};

// =============================================================================
// Per-kind headline columns
//
// Each per-kind subquery projects a uniform 24-column shape (the 8
// shared columns + the per-kind headline fields for jq
// reach-through). Each kind's subquery fills its applicable
// columns; the rest are CAST(NULL AS …). The constants below pad
// the irrelevant slots for non-extended kinds so each subquery's
// SELECT list stays readable.
// =============================================================================

/// NULL pad for the 11 kernel-only headline columns (positions
/// 8..=18 in the outer row tuple).
const NULL_KERNEL_COLS: &str = "\
    CAST(NULL AS BIGINT) AS grid_x, CAST(NULL AS BIGINT) AS grid_y, CAST(NULL AS BIGINT) AS grid_z, \
    CAST(NULL AS BIGINT) AS block_x, CAST(NULL AS BIGINT) AS block_y, CAST(NULL AS BIGINT) AS block_z, \
    CAST(NULL AS BIGINT) AS registers_per_thread, \
    CAST(NULL AS BIGINT) AS static_shared_memory, \
    CAST(NULL AS BIGINT) AS dynamic_shared_memory, \
    CAST(NULL AS VARCHAR) AS demangled_name, \
    CAST(NULL AS VARCHAR) AS mangled_name";

/// NULL pad for the 3 memcpy/memset-only columns (positions 19..=21).
const NULL_MEMOP_COLS: &str = "\
    CAST(NULL AS BIGINT) AS bytes, \
    CAST(NULL AS BIGINT) AS copy_kind, \
    CAST(NULL AS BIGINT) AS memset_value";

/// NULL pad for the 2 nvtx-only columns (positions 22..=23).
const NULL_NVTX_COLS: &str = "\
    CAST(NULL AS BIGINT) AS event_type, \
    CAST(NULL AS BIGINT) AS domain_id";

// Column-index constants for the row builder. Must mirror the
// projection order in per_kind_select + the outer SELECT.
const COL_KIND: usize = 0;
const COL_ROW_ID_NUM: usize = 1;
const COL_NAME: usize = 2;
const COL_START_NS: usize = 3;
const COL_DURATION_NS: usize = 4;
const COL_DEVICE_ID: usize = 5;
const COL_STREAM_ID: usize = 6;
const COL_GLOBAL_TID: usize = 7;
const COL_GRID_X: usize = 8;
const COL_GRID_Y: usize = 9;
const COL_GRID_Z: usize = 10;
const COL_BLOCK_X: usize = 11;
const COL_BLOCK_Y: usize = 12;
const COL_BLOCK_Z: usize = 13;
const COL_REGISTERS_PER_THREAD: usize = 14;
const COL_STATIC_SHARED_MEMORY: usize = 15;
const COL_DYNAMIC_SHARED_MEMORY: usize = 16;
const COL_DEMANGLED_NAME: usize = 17;
const COL_MANGLED_NAME: usize = 18;
const COL_BYTES: usize = 19;
const COL_COPY_KIND: usize = 20;
const COL_MEMSET_VALUE: usize = 21;
const COL_EVENT_TYPE: usize = 22;
const COL_DOMAIN_ID: usize = 23;

const SEARCH_RANK_SQL: &str = "rank";
const SEARCH_HYDRATE_SQL: &str = "hydrate";

#[derive(Debug, Clone)]
pub struct SearchRequest {
    /// Which event tables to include. `KindFilter::All` covers every
    /// kind search knows about; `KindFilter::Only(...)` picks a subset.
    pub kinds: KindFilter,
    /// Wildcard-style name pattern. `*` matches any sequence, `?` any
    /// single char. `None` = no name filter. Mutually exclusive with
    /// `name_regex` at the CLI layer.
    pub name_glob: Option<String>,
    /// Regex-style name pattern (DuckDB `regexp_matches` semantics —
    /// PCRE-ish). `None` = no name filter. Mutually exclusive with
    /// `name_glob`.
    pub name_regex: Option<String>,
    /// Restrict by event duration (e.g. `>1ms`, `100us-1ms`).
    pub duration: Option<DurationFilter>,
    /// Restrict to events overlapping this window (relative to trace start).
    pub time_window: Option<TimeWindow>,
    /// When set, only return GPU events causally attributed to NVTX
    /// ranges matching this glob. Non-GPU kinds (runtime/osrt/nvtx) get
    /// implicitly dropped from `kinds` since attribution doesn't apply.
    pub nvtx: Option<String>,
    /// Restrict to one native process owning the CUDA namespace.
    pub process_id: Option<i64>,
    /// Restrict to one CUDA device (NSys `deviceId`). Only the GPU
    /// kinds (kernel/memcpy/memset) carry a deviceId; runtime/osrt/nvtx
    /// rows pass through this filter unchanged.
    pub device: Option<i32>,
    /// Restrict to one CUDA stream. Same kind scoping as `device`.
    pub stream: Option<i64>,
    /// Sort specification (one or more `key[:dir]` fields).
    /// `None` falls back to the default (`start` ascending).
    pub sort: Option<SortSpec>,
    /// Max events to return.
    pub limit: usize,
    /// Populate every kernel/memcpy/memset/sync hit's `nvtx_context`
    /// via the reverse-attribution query. Off by default — a
    /// large result set otherwise adds one extra SQL per CUPTI kind
    /// in the response. Orthogonal to `--nvtx`: `--nvtx` filters the
    /// result set to GPU work *attributed* to a name pattern;
    /// `--with-nvtx` decorates each row that's already in the
    /// result with the innermost enclosing range.
    pub with_nvtx: bool,
}

impl Default for SearchRequest {
    /// Mirror the CLI defaults so library callers using
    /// `SearchRequest::default()` get a working request (not the
    /// derive's `limit: 0`, which would silently empty the response
    /// and zero out `total_matched`).
    fn default() -> Self {
        Self {
            kinds: KindFilter::All,
            name_glob: None,
            name_regex: None,
            duration: None,
            time_window: None,
            nvtx: None,
            process_id: None,
            device: None,
            stream: None,
            sort: None,
            limit: 100,
            with_nvtx: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortKey {
    #[default]
    Start,
    Duration,
    Name,
}

impl SortKeyDef for SortKey {
    fn specs() -> &'static [SortKeySpec<Self>] {
        // `start` leads (chronological is the obvious default).
        // `duration` is the regression-hunt axis; `name` is the
        // alphabetical browse axis.
        &[
            SortKeySpec {
                variant: SortKey::Start,
                canonical: "start",
                aliases: &["time"],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: SortKey::Duration,
                canonical: "duration",
                aliases: &["dur"],
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
    /// SQL column name that this key sorts on. Tiebreakers
    /// (`row_id_num`) are appended by the caller so each command
    /// stays deterministic across runs.
    fn primary_column(self) -> &'static str {
        match self {
            Self::Start => "start_ns",
            Self::Duration => "duration_ns",
            Self::Name => "name",
        }
    }
}

/// Build the SQL `ORDER BY` body for search, using
/// `veloq_core::sort::build_order_by` for the shared format.
fn sort_sql(spec: &SortSpec) -> NsysQueryResult<String> {
    order_by::<SortKey>(
        spec,
        SortKey::primary_column,
        NsysQueryError::search_sort_invalid,
        "row_id_num",
    )
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SearchResponse {
    /// Rows returned (after LIMIT).
    pub count: usize,
    /// Rows matching the filter, before LIMIT was applied. When
    /// `total_matched > count`, raise `--limit` or narrow the filter.
    pub total_matched: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_window_ns: Option<(i64, i64)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_scope: Option<String>,
    /// Canonical primary table. Each row is one matched event,
    /// shaped as a shared [`EventRef`] (same type `correlate.events`,
    /// future `ncu launches`, and `inspect`-time references use).
    pub rows: Vec<EventRef>,
}

pub fn run<P: AsRef<Path>>(path: P, req: SearchRequest) -> NsysQueryResult<SearchResponse> {
    let (trace, abs_window) = crate::open_scoped(path.as_ref(), req.limit, req.time_window)?;

    if let KindFilter::Only(kinds) = &req.kinds
        && kinds.contains(&EventKind::CpuSample)
    {
        return Err(NsysQueryError::SearchCpuSampleUnsupported);
    }

    // Shared `--device` / `--stream` policy (see [`crate::kind_policy`]
    // for the rule and the wording rationale).
    crate::kind_policy::validate_location_filter(
        &req.kinds,
        crate::kind_policy::LocationFilter {
            process_id: req.process_id,
            device: req.device,
            stream: req.stream,
        },
        "search",
    )?;
    // Validate `--type` + `--nvtx`, resolve to a concrete kind list,
    // filter by table presence, and (when `--nvtx` is set) implicitly
    // narrow to the attributable set. The shared helper in
    // `kind_policy` keeps this pipeline identical across stats /
    // search / timeline.
    let kinds = crate::kind_policy::resolve_nvtx_kinds(
        &req.kinds,
        req.nvtx.as_deref(),
        EventKind::ALL,
        &trace,
        "search",
    )?;
    if kinds.is_empty() {
        return Ok(SearchResponse {
            count: 0,
            total_matched: 0,
            time_window_ns: abs_window,
            nvtx_scope: req.nvtx.clone(),
            rows: Vec::new(),
        });
    }

    // Build the attribution CTE *after* the kind filter so it only
    // emits views for tables we actually need + are present.
    let attribution = match req.nvtx.as_deref() {
        Some(p) => Some(crate::nvtx_attribution::build(p, &kinds, &trace)?),
        None => None,
    };

    // Search runs in two phases for late materialization. The slow part
    // of a wide scan is decompressing the headline columns and resolving
    // names through the StringIds joins for *every* matched row before
    // LIMIT can trim them. Stage 1 ranks a *thin* projection (rowid +
    // the sort/filter columns, no joins unless a name filter/sort needs
    // them) to find the surviving `(kind, rowid)`s and `total_matched`;
    // stage 2 then materializes the full headline payload for only those
    // ≤limit survivors. See [`per_kind_rank_select`] / [`per_kind_select`].
    let nvtx_scope = if attribution.is_some() {
        crate::nvtx_attribution::NvtxScope::Attributed
    } else {
        crate::nvtx_attribution::NvtxScope::None
    };

    // Build outer filters
    let mut where_parts: Vec<String> = Vec::new();
    let mut params: Vec<Value> = Vec::new();

    let name_filter =
        NameFilterRef::from_optional(req.name_glob.as_deref(), req.name_regex.as_deref())
            .map_err(|_| NsysQueryError::NameFilterConflict)?;
    if let Some(fragment) = name::predicate("name", name_filter) {
        where_parts.push(fragment.sql);
        params.extend(fragment.params);
    }

    crate::kind_policy::LocationFilter {
        process_id: req.process_id,
        device: req.device,
        stream: req.stream,
    }
    .push_where(&mut where_parts, &mut params);

    if let Some(d) = req.duration {
        where_parts.push(d.sql("duration_ns"));
        for n in d.sql_params() {
            params.push(Value::BigInt(n));
        }
    }

    if let Some(fragment) =
        window::overlap_filter_expr("start_ns", "(start_ns + duration_ns)", abs_window)
    {
        where_parts.push(fragment.sql);
        params.extend(fragment.params);
    }

    // duration sanity: drop events with non-positive duration (instant
    // NVTX markers etc.). `graph_node`, `graph_event`, and `cuda_event`
    // are metadata/instantaneous-marker kinds — exempt them so search
    // returns their rows.
    where_parts.push(
        "(duration_ns > 0 OR kind IN ('graph_node', 'graph_event', 'cuda_event'))".to_string(),
    );

    let where_clause = format!("WHERE {}", where_parts.join(" AND "));

    // Default = bare "start" if caller didn't supply a SortSpec.
    let sort_spec = match req.sort.as_ref() {
        Some(s) => s.clone(),
        None => SortSpec::single("start"),
    };
    let order_by = sort_sql(&sort_spec)?;

    // ---- Stage 1: rank a thin projection. Project `name` (and so join
    // StringIds) only when a name filter or a name sort reads it;
    // otherwise the rank scan touches just rowid + the numeric
    // sort/filter columns, leaving the expensive joins to stage 2.
    let include_name =
        req.name_glob.is_some() || req.name_regex.is_some() || sort_uses_name(&sort_spec);

    // Name pre-filter: when a name filter is active and the kind set
    // includes a StringId-named kind (kernel/runtime/osrt), resolve the
    // matching StringIds once into a CTE so those kinds can prune to
    // candidate rows *before* their name joins, instead of resolving names
    // across the whole table. The outer `name LIKE/regexp ?` stays the
    // authoritative filter — the per-kind predicate is a proven superset
    // (see [`per_kind_rank_select`]), so this only accelerates and can't
    // change the result set.
    let has_stringid_kind = kinds
        .iter()
        .any(|k| matches!(k, EventKind::Kernel | EventKind::Runtime | EventKind::Osrt));
    // The pre-filter targets the expensive-outer-filter case:
    // `--name-regex`, where DuckDB would otherwise run PCRE matching on
    // every row's resolved name. Resolving the matching StringIds once
    // lets the planner prune the scan via the id-membership semi-join
    // ahead of both the StringIds name joins and the per-row regexp. A
    // glob's `LIKE` is cheap, so the planner leaves the name join in front
    // of it and the semi-join wouldn't prune — there the pre-filter is
    // pure overhead, so it's skipped. The outer name filter stays
    // authoritative either way; the per-kind predicate is only a superset.
    let name_match_cte: Option<(String, Value)> = match (&req.name_regex, has_stringid_kind) {
        (Some(re), true) => Some((
            "name_match_ids AS (SELECT id FROM nsight.StringIds WHERE regexp_matches(value, ?))"
                .to_string(),
            Value::Text(re.clone()),
        )),
        _ => None,
    };
    let use_prefilter = name_match_cte.is_some();

    let mut rank_subqueries = Vec::with_capacity(kinds.len());
    let mut rank_subquery_params = Vec::new();
    for k in &kinds {
        let fragment = crate::query_sql::event_scan::search_rank_select(
            &trace,
            *k,
            nvtx_scope,
            include_name,
            use_prefilter,
        )?;
        rank_subqueries.push(fragment.sql);
        rank_subquery_params.extend(fragment.params);
    }
    let rank_union = rank_subqueries.join(" UNION ALL ");

    // Combine the (optional) pre-filter CTE and the (optional) attribution
    // CTEs into one WITH; param order below follows this textual order.
    let mut cte_parts: Vec<String> = Vec::new();
    if let Some((cte, _)) = &name_match_cte {
        cte_parts.push(cte.clone());
    }
    if let Some(att) = &attribution {
        cte_parts.push(att.body.clone());
    }
    let attribution_prefix = if cte_parts.is_empty() {
        String::new()
    } else {
        format!("WITH {} ", cte_parts.join(", "))
    };

    let rank_sql = format!(
        r#"
        {attribution_prefix}
        SELECT kind, row_id_num, process_id,
               {total_matched}
        FROM ({rank_union})
        {where_clause}
        ORDER BY {order_by}
        LIMIT ?
        "#,
        total_matched = total_matched_bigint_expr(),
    );

    // Bind order matches SQL position: pre-filter CTE param (if any),
    // attribution CTE params (if `--nvtx`), the outer-filter params, LIMIT.
    let mut rank_params: Vec<Value> = Vec::new();
    if let Some((_, p)) = &name_match_cte {
        rank_params.push(p.clone());
    }
    if let Some(att) = &attribution {
        rank_params.extend(att.params.iter().cloned());
    }
    rank_params.extend(rank_subquery_params);
    rank_params.extend(params.iter().cloned());
    rank_params.push(Value::BigInt(req.limit as i64));

    let conn = trace.conn();
    let (survivors, total_matched) = hydrate_ranked_survivors(conn, &rank_sql, &rank_params)?;

    // NVTX nesting is computed at most once per search call. Build it
    // for either reason: NVTX hits in the result set need their `depth`
    // populated, and `--with-nvtx` (the reverse-attribution decoration)
    // wants depth + iter_index for every GPU hit. Skip otherwise —
    // leaving `depth`/`nvtx_context` on every row as `None`.
    let needs_nesting = kinds.contains(&EventKind::Nvtx) || req.with_nvtx;
    let nesting = if needs_nesting {
        Some(
            trace
                .nvtx_nesting()
                .map_err(NsysQueryError::nvtx_nesting_load)?,
        )
    } else {
        None
    };

    if survivors.is_empty() {
        return Ok(SearchResponse {
            count: 0,
            total_matched,
            time_window_ns: abs_window,
            nvtx_scope: req.nvtx.clone(),
            rows: Vec::new(),
        });
    }

    // ---- Stage 2: materialize the full headline payload for the ≤limit
    // survivors only. Each surviving kind's rowids are pushed into that
    // kind's own scan (`t.rowid IN (...)`) ahead of the StringIds joins,
    // so the wide columns/joins touch only the rows we keep. Filtering
    // per kind (not over the union) avoids cross-kind rowid collisions —
    // a kernel and a memcpy can share a per-table rowid.
    let cols = column_map::load_standard(conn)?;
    let mut wide_subqueries = Vec::new();
    for k in &kinds {
        let ids: Vec<i64> = survivors
            .iter()
            .filter(|(sk, _, _)| sk == k)
            .map(|(_, r, _)| *r)
            .collect();
        if ids.is_empty() {
            continue;
        }
        wide_subqueries.push(per_kind_select(
            *k,
            crate::nvtx_attribution::NvtxScope::None,
            &cols,
            Some(&ids),
        )?);
    }
    let wide_union = wide_subqueries.join(" UNION ALL ");

    let sql = format!(
        r#"
        SELECT kind, row_id_num, name, start_ns, duration_ns,
               device_id, stream_id, global_tid,
               grid_x, grid_y, grid_z,
               block_x, block_y, block_z,
               registers_per_thread,
               static_shared_memory,
               dynamic_shared_memory,
               demangled_name, mangled_name,
               bytes, copy_kind, memset_value,
               event_type, domain_id
        FROM ({wide_union})
        "#
    );

    let mut by_id = hydrate_search_rows(conn, &sql, survivors.len(), nesting.as_ref())?;

    // Re-apply the stage-1 ordering: stage 2 fetched survivors by rowid
    // (arbitrary order), so walk the ranked survivor list and pull each
    // materialized row back out in that order.
    let mut events: Vec<EventRef> = Vec::with_capacity(survivors.len());
    for (k, r, process_id) in &survivors {
        if let Some(ev) = by_id.remove(&RowId::new(*k, *r)) {
            let mut ev = ev;
            ev.base_mut().process_id = *process_id;
            events.push(ev);
        }
    }

    // Opt-in batched reverse-attribution. Decorates kernel/memcpy/
    // memset/sync hits with the innermost NVTX range that was open on
    // their launching thread. Other kinds in the result silently pass
    // through with `nvtx_context: None`. The `needs_nesting` branch
    // above already built the map when `with_nvtx` was set, so we
    // just reuse it.
    if req.with_nvtx
        && let Some(nesting_map) = nesting.as_ref()
    {
        let row_ids: Vec<crate::RowId> = events.iter().map(|e| e.base().row_id).collect();
        let contexts = crate::nvtx_reverse::lookup_for_row_ids(&trace, &row_ids, nesting_map)?;
        for ev in &mut events {
            let row_id = ev.base().row_id;
            if let Some(ctx) = contexts.get(&row_id) {
                ev.base_mut().nvtx_context = Some(ctx.clone());
            }
        }
    }

    Ok(SearchResponse {
        count: events.len(),
        total_matched,
        time_window_ns: abs_window,
        nvtx_scope: req.nvtx.clone(),
        rows: events,
    })
}

fn hydrate_ranked_survivors(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
) -> NsysQueryResult<(Vec<(EventKind, i64, Option<i64>)>, i64)> {
    let rows = query_rows(
        conn,
        sql,
        params,
        SqlLabel::new("search", SEARCH_RANK_SQL),
        search_rank_sql_row,
    )?;
    duckdb_list::split_rows_and_total::<i64, _, _, _>(
        rows,
        duckdb_list::TotalCarrier::First,
        |row| row.total_matched,
        duckdb_list::infallible_count_error,
        |row| {
            Ok((
                parse_search_sql_kind(&row.kind)?,
                row.rowid_num,
                row.process_id,
            ))
        },
    )
}

struct SearchRankSqlRow {
    kind: String,
    rowid_num: i64,
    process_id: Option<i64>,
    total_matched: i64,
}

fn search_rank_sql_row(row: &duckdb::Row<'_>) -> Result<SearchRankSqlRow, duckdb::Error> {
    Ok(SearchRankSqlRow {
        kind: row.get(0)?,
        rowid_num: row.get(1)?,
        process_id: row.get(2)?,
        total_matched: row.get(3)?,
    })
}

fn hydrate_search_rows(
    conn: &duckdb::Connection,
    sql: &str,
    survivor_count: usize,
    nesting: Option<&NvtxNesting>,
) -> NsysQueryResult<HashMap<RowId, EventRef>> {
    let mut by_id: HashMap<RowId, EventRef> = HashMap::with_capacity(survivor_count);
    let rows = query_rows(
        conn,
        sql,
        &[],
        SqlLabel::new("search", SEARCH_HYDRATE_SQL),
        search_sql_row,
    )?;
    for row in rows {
        let (row_id, event_ref) = event_ref_from_sql_row(row, nesting)?;
        by_id.insert(row_id, event_ref);
    }
    Ok(by_id)
}

struct SearchSqlRow {
    kind: String,
    rowid_num: i64,
    name: String,
    start_ns: i64,
    duration_ns: i64,
    device_id: Option<i32>,
    stream_id: Option<i64>,
    global_tid: Option<i64>,
    grid_x: Option<i64>,
    grid_y: Option<i64>,
    grid_z: Option<i64>,
    block_x: Option<i64>,
    block_y: Option<i64>,
    block_z: Option<i64>,
    registers_per_thread: Option<i64>,
    static_shared_memory: Option<i64>,
    dynamic_shared_memory: Option<i64>,
    demangled_name: Option<String>,
    mangled_name: Option<String>,
    bytes: Option<i64>,
    copy_kind: Option<i64>,
    memset_value: Option<i64>,
    event_type: Option<i64>,
    domain_id: Option<i64>,
}

fn search_sql_row(row: &duckdb::Row<'_>) -> Result<SearchSqlRow, duckdb::Error> {
    Ok(SearchSqlRow {
        kind: row.get(COL_KIND)?,
        rowid_num: row.get(COL_ROW_ID_NUM)?,
        name: row.get(COL_NAME)?,
        start_ns: row.get(COL_START_NS)?,
        duration_ns: row.get(COL_DURATION_NS)?,
        device_id: row.get(COL_DEVICE_ID)?,
        stream_id: row.get(COL_STREAM_ID)?,
        global_tid: row.get(COL_GLOBAL_TID)?,
        grid_x: row.get(COL_GRID_X)?,
        grid_y: row.get(COL_GRID_Y)?,
        grid_z: row.get(COL_GRID_Z)?,
        block_x: row.get(COL_BLOCK_X)?,
        block_y: row.get(COL_BLOCK_Y)?,
        block_z: row.get(COL_BLOCK_Z)?,
        registers_per_thread: row.get(COL_REGISTERS_PER_THREAD)?,
        static_shared_memory: row.get(COL_STATIC_SHARED_MEMORY)?,
        dynamic_shared_memory: row.get(COL_DYNAMIC_SHARED_MEMORY)?,
        demangled_name: row.get(COL_DEMANGLED_NAME)?,
        mangled_name: row.get(COL_MANGLED_NAME)?,
        bytes: row.get(COL_BYTES)?,
        copy_kind: row.get(COL_COPY_KIND)?,
        memset_value: row.get(COL_MEMSET_VALUE)?,
        event_type: row.get(COL_EVENT_TYPE)?,
        domain_id: row.get(COL_DOMAIN_ID)?,
    })
}

fn event_ref_from_sql_row(
    row: SearchSqlRow,
    nesting: Option<&NvtxNesting>,
) -> NsysQueryResult<(RowId, EventRef)> {
    let kind = parse_search_sql_kind(&row.kind)?;
    // Populate `depth` only for NVTX hits, and only when we actually
    // computed the nesting map. Lookup miss (e.g. an instant marker
    // whose row predates the nesting scan, though `compute` covers
    // those) falls back to `None` rather than spelling a default.
    let depth = match (kind, nesting) {
        (EventKind::Nvtx, Some(map)) => map.get(&row.rowid_num).map(|e| e.depth),
        _ => None,
    };

    let row_id = RowId::new(kind, row.rowid_num);
    let base = EventRefBase {
        key: row_id.to_string(),
        row_id,
        name: row.name,
        start_ns: row.start_ns,
        duration_ns: row.duration_ns,
        process_id: None,
        device_id: row.device_id,
        stream_id: row.stream_id,
        global_tid: row.global_tid,
        depth,
        // `nvtx_context` is populated by the `--with-nvtx`
        // post-decoration pass below; left None on construction.
        nvtx_context: None,
    };

    let event_ref = match kind {
        EventKind::Kernel => EventRef::Kernel(EventRefKernel {
            base,
            grid: build_xyz(row.grid_x, row.grid_y, row.grid_z),
            block: build_xyz(row.block_x, row.block_y, row.block_z),
            registers_per_thread: row.registers_per_thread,
            static_shared_memory: row.static_shared_memory,
            dynamic_shared_memory: row.dynamic_shared_memory,
            demangled_name: row.demangled_name,
            mangled_name: row.mangled_name,
        }),
        EventKind::Memcpy => {
            let copy_kind_name = row.copy_kind.map(crate::kind_sql::copy_kind_label);
            EventRef::Memcpy(EventRefMemcpy {
                base,
                bytes: row.bytes,
                copy_kind: row.copy_kind,
                copy_kind_name,
            })
        }
        EventKind::Memset => EventRef::Memset(EventRefMemset {
            base,
            bytes: row.bytes,
            value: row.memset_value,
        }),
        EventKind::Nvtx => EventRef::Nvtx(EventRefNvtx {
            base,
            event_type: row.event_type,
            domain_id: row.domain_id,
        }),
        // Non-extended kinds carry just the shared base.
        _ => EventRef::from_base(kind, base)?,
    };
    Ok((row_id, event_ref))
}

fn parse_search_sql_kind(kind: &str) -> NsysQueryResult<EventKind> {
    let Some(kind) = EventKind::parse(kind) else {
        return Err(NsysQueryError::internal_sql_kind_tag_invalid(
            "search", kind,
        ));
    };
    Ok(kind)
}

/// Read three columns at `xi`/`yi`/`zi` into an `Option<[i64; 3]>`.
/// All three present → `Some(...)`; any NULL → `None`. Used to
/// assemble kernel `grid`/`block` triples in the row builder.
fn build_xyz(x: Option<i64>, y: Option<i64>, z: Option<i64>) -> Option<[i64; 3]> {
    match (x, y, z) {
        (Some(x), Some(y), Some(z)) => Some([x, y, z]),
        _ => None,
    }
}

fn per_kind_select(
    kind: EventKind,
    nvtx_scope: crate::nvtx_attribution::NvtxScope,
    cols: &ColumnMap,
    rowid_filter: Option<&[i64]>,
) -> NsysQueryResult<String> {
    let sem = EventSemantics::new(kind);
    let table = sem.table();
    let label = sem.label();
    let dev = sem.device_expr();
    let stm = sem.stream_expr();
    // Predicates pushed into this kind's scan ahead of the StringIds
    // joins: the NVTX attribution constraint (when `--nvtx` is active and
    // the kind is attributable — Kernel/Memcpy/Memset/Sync/Runtime) and,
    // in stage 2, an explicit survivor-rowid filter. Both reference only
    // base columns, so DuckDB pushes them down to the scan, keeping the
    // joins off every non-matching row.
    let mut base_preds: Vec<String> = Vec::new();
    if nvtx_scope.is_attributed()
        && let Some(p) = sem.attribution_filter("t")
    {
        base_preds.push(p);
    }
    if let Some(ids) = rowid_filter {
        base_preds.push(rowid_in_list(ids));
    }
    let where_clause = build_where_clause(&base_preds);
    let global_tid = veloq_nsys_data::sql_expr::u64_bits_to_i64("t.globalTid");
    Ok(match kind {
        EventKind::Kernel => {
            // Kernel — the extended kind with the largest headline
            // payload (grid/block/registers/shared/demangled/mangled).
            // grid* / block* are mandatory in the CUPTI table;
            // registers / shared memory / mangled are probed via
            // `maybe_col` so older NSys schemas degrade to NULL.
            const T: &str = "CUPTI_ACTIVITY_KIND_KERNEL";
            let name_expr = sem.display_name_expr();
            let joins = sem.name_joins();
            let reg = maybe_col(cols, T, "registersPerThread");
            let smem_static = maybe_col(cols, T, "staticSharedMemory");
            let smem_dyn = maybe_col(cols, T, "dynamicSharedMemory");
            // mangledName: the join key resolves to NULL when the
            // column is absent, leaving the StringIds lookup
            // empty → projected mangled_name = NULL. Cleaner than
            // gating the join itself.
            let mangled_col = maybe_col(cols, T, "mangledName");
            format!(
                r#"
                SELECT
                    '{label}' AS kind,
                    t.rowid AS row_id_num,
                    {name_expr} AS name,
                    t.start AS start_ns,
                    (t."end" - t.start) AS duration_ns,
                    {dev} AS device_id,
                    {stm} AS stream_id,
                    CAST(NULL AS BIGINT) AS global_tid,
                    CAST(t.gridX AS BIGINT) AS grid_x,
                    CAST(t.gridY AS BIGINT) AS grid_y,
                    CAST(t.gridZ AS BIGINT) AS grid_z,
                    CAST(t.blockX AS BIGINT) AS block_x,
                    CAST(t.blockY AS BIGINT) AS block_y,
                    CAST(t.blockZ AS BIGINT) AS block_z,
                    CAST({reg} AS BIGINT) AS registers_per_thread,
                    CAST({smem_static} AS BIGINT) AS static_shared_memory,
                    CAST({smem_dyn} AS BIGINT) AS dynamic_shared_memory,
                    s_dem.value AS demangled_name,
                    s_mng.value AS mangled_name,
                    {NULL_MEMOP_COLS},
                    {NULL_NVTX_COLS}
                FROM nsight.{table} t {joins}
                LEFT JOIN nsight.StringIds s_mng ON s_mng.id = {mangled_col}
                {where_clause}
                "#,
            )
        }
        EventKind::Memcpy => {
            // Memcpy — bytes + copyKind are mandatory in the CUPTI
            // table. Name comes from kind_sql so the copyKind →
            // label CASE stays single-sourced.
            let name_expr = sem.display_name_expr();
            let joins = sem.name_joins();
            format!(
                r#"
                SELECT
                    '{label}' AS kind,
                    t.rowid AS row_id_num,
                    {name_expr} AS name,
                    t.start AS start_ns,
                    (t."end" - t.start) AS duration_ns,
                    {dev} AS device_id,
                    {stm} AS stream_id,
                    CAST(NULL AS BIGINT) AS global_tid,
                    {NULL_KERNEL_COLS},
                    CAST(t.bytes AS BIGINT) AS bytes,
                    CAST(t.copyKind AS BIGINT) AS copy_kind,
                    CAST(NULL AS BIGINT) AS memset_value,
                    {NULL_NVTX_COLS}
                FROM nsight.{table} t {joins}
                {where_clause}
                "#,
            )
        }
        EventKind::Memset => {
            // Memset — bytes is mandatory; the `value` column is
            // optional on older NSys schemas, so it goes through
            // maybe_col.
            const T: &str = "CUPTI_ACTIVITY_KIND_MEMSET";
            let name_expr = sem.display_name_expr();
            let joins = sem.name_joins();
            let val = maybe_col(cols, T, "value");
            format!(
                r#"
                SELECT
                    '{label}' AS kind,
                    t.rowid AS row_id_num,
                    {name_expr} AS name,
                    t.start AS start_ns,
                    (t."end" - t.start) AS duration_ns,
                    {dev} AS device_id,
                    {stm} AS stream_id,
                    CAST(NULL AS BIGINT) AS global_tid,
                    {NULL_KERNEL_COLS},
                    CAST(t.bytes AS BIGINT) AS bytes,
                    CAST(NULL AS BIGINT) AS copy_kind,
                    CAST({val} AS BIGINT) AS memset_value,
                    {NULL_NVTX_COLS}
                FROM nsight.{table} t {joins}
                {where_clause}
                "#,
            )
        }
        EventKind::Sync | EventKind::Graph => {
            // Sync/Graph — base-only kinds that still need
            // kind_sql-driven naming + joins, but have no
            // per-kind headline payload.
            let name_expr = sem.display_name_expr();
            let joins = sem.name_joins();
            format!(
                r#"
                SELECT
                    '{label}' AS kind,
                    t.rowid AS row_id_num,
                    {name_expr} AS name,
                    t.start AS start_ns,
                    (t."end" - t.start) AS duration_ns,
                    {dev} AS device_id,
                    {stm} AS stream_id,
                    CAST(NULL AS BIGINT) AS global_tid,
                    {NULL_KERNEL_COLS},
                    {NULL_MEMOP_COLS},
                    {NULL_NVTX_COLS}
                FROM nsight.{table} t {joins}
                {where_clause}
                "#,
            )
        }
        EventKind::Runtime => format!(
            r#"
            SELECT
                '{label}' AS kind,
                t.rowid AS row_id_num,
                COALESCE(s.value, '<unknown runtime>') AS name,
                t.start AS start_ns,
                (t."end" - t.start) AS duration_ns,
                CAST(NULL AS INTEGER) AS device_id,
                CAST(NULL AS BIGINT) AS stream_id,
                {global_tid} AS global_tid,
                {NULL_KERNEL_COLS},
                {NULL_MEMOP_COLS},
                {NULL_NVTX_COLS}
            FROM nsight.{table} t
            LEFT JOIN nsight.StringIds s ON t.nameId = s.id
            {where_clause}
            "#,
        ),
        EventKind::Osrt => format!(
            r#"
            SELECT
                '{label}' AS kind,
                t.rowid AS row_id_num,
                COALESCE(s.value, '<unknown osrt>') AS name,
                t.start AS start_ns,
                (t."end" - t.start) AS duration_ns,
                CAST(NULL AS INTEGER) AS device_id,
                CAST(NULL AS BIGINT) AS stream_id,
                {global_tid} AS global_tid,
                {NULL_KERNEL_COLS},
                {NULL_MEMOP_COLS},
                {NULL_NVTX_COLS}
            FROM nsight.{table} t
            LEFT JOIN nsight.StringIds s ON t.nameId = s.id
            {where_clause}
            "#,
        ),
        EventKind::Nvtx => {
            // NVTX — the extended kind; surfaces eventType + domainId
            // (both mandatory in NVTX_EVENTS) so agents can filter by
            // domain or distinguish PushPop/Mark/Range without an
            // inspect roundtrip. `t."end" IS NOT NULL` is intrinsic
            // (drops instant markers) and ANDs with any rowid filter.
            let mut preds = base_preds.clone();
            preds.push(r#"t."end" IS NOT NULL"#.to_string());
            let where_clause = build_where_clause(&preds);
            format!(
                r#"
                SELECT
                    '{label}' AS kind,
                    t.rowid AS row_id_num,
                    COALESCE(t.text, s.value, '<unnamed nvtx>') AS name,
                    t.start AS start_ns,
                    (t."end" - t.start) AS duration_ns,
                    CAST(NULL AS INTEGER) AS device_id,
                    CAST(NULL AS BIGINT) AS stream_id,
                    {global_tid} AS global_tid,
                    {NULL_KERNEL_COLS},
                    {NULL_MEMOP_COLS},
                    CAST(t.eventType AS BIGINT) AS event_type,
                    CAST(t.domainId AS BIGINT) AS domain_id
                FROM nsight.{table} t
                LEFT JOIN nsight.StringIds s ON t.textId = s.id
                {where_clause}
                "#,
            )
        }
        EventKind::GraphNode => format!(
            r#"
            SELECT
                '{label}' AS kind,
                t.rowid AS row_id_num,
                'node:' || CAST(t.graphNodeId AS VARCHAR) AS name,
                t.start AS start_ns,
                (t."end" - t.start) AS duration_ns,
                CAST(NULL AS INTEGER) AS device_id,
                CAST(NULL AS BIGINT) AS stream_id,
                {global_tid} AS global_tid,
                {NULL_KERNEL_COLS},
                {NULL_MEMOP_COLS},
                {NULL_NVTX_COLS}
            FROM nsight.{table} t
            {where_clause}
            "#,
        ),
        EventKind::GraphEvent => {
            // Name = snake_case eventClass label via the shared CASE.
            // Both creation sub-types fall under the single graph_event
            // kind; agents pick a sub-type via `--name graph_creation`
            // or `--name graph_exec_creation`.
            let name_expr = sem.display_name_expr();
            format!(
                r#"
                SELECT
                    '{label}' AS kind,
                    t.rowid AS row_id_num,
                    {name_expr} AS name,
                    t.start AS start_ns,
                    (t."end" - t.start) AS duration_ns,
                    CAST(NULL AS INTEGER) AS device_id,
                    CAST(NULL AS BIGINT) AS stream_id,
                {global_tid} AS global_tid,
                    {NULL_KERNEL_COLS},
                    {NULL_MEMOP_COLS},
                    {NULL_NVTX_COLS}
                FROM nsight.{table} t
                {where_clause}
                "#,
            )
        }
        EventKind::CudaEvent => {
            // CUDA_EVENT rows have a single `timestamp` column (no
            // end). Project duration_ns = 0 and rely on the duration-
            // filter exemption (below) to keep them searchable.
            let name_expr = sem.display_name_expr();
            format!(
                r#"
                SELECT
                    '{label}' AS kind,
                    t.rowid AS row_id_num,
                    {name_expr} AS name,
                    t.timestamp AS start_ns,
                    0 AS duration_ns,
                    {dev} AS device_id,
                    {stm} AS stream_id,
                    CAST(NULL AS BIGINT) AS global_tid,
                    {NULL_KERNEL_COLS},
                    {NULL_MEMOP_COLS},
                    {NULL_NVTX_COLS}
                FROM nsight.{table} t
                {where_clause}
                "#,
            )
        }
        EventKind::Overhead => {
            // Has start/end + globalTid; no deviceId/streamId.
            let name_expr = sem.display_name_expr();
            format!(
                r#"
                SELECT
                    '{label}' AS kind,
                    t.rowid AS row_id_num,
                    {name_expr} AS name,
                    t.start AS start_ns,
                    (t."end" - t.start) AS duration_ns,
                    CAST(NULL AS INTEGER) AS device_id,
                    CAST(NULL AS BIGINT) AS stream_id,
                    {global_tid} AS global_tid,
                    {NULL_KERNEL_COLS},
                    {NULL_MEMOP_COLS},
                    {NULL_NVTX_COLS}
                FROM nsight.{table} t
                {where_clause}
                "#,
            )
        }
        // `CpuSample` is excluded from `EventKind::ALL`, so
        // `GpuFilters::kinds` rejects `--type cpu_sample` before we
        // get here. The bail keeps the match exhaustive and surfaces
        // a clear redirect if a future caller routes around the
        // upstream allow-list (e.g. a library consumer hand-building
        // a request).
        EventKind::CpuSample => {
            return Err(NsysQueryError::SearchCpuSampleUnsupported);
        }
    })
}

/// `t.rowid IN (...)` over an explicit set of survivor rowids. Rowids are
/// veloq-internal table identifiers (`file_row_number + 1`), never user
/// input, so inlining them as integer literals is safe — and an IN-list
/// of literals is what lets DuckDB push the filter into the parquet scan
/// ahead of the StringIds joins (stage 2's whole point).
fn rowid_in_list(ids: &[i64]) -> String {
    let mut joined = String::new();
    for (i, id) in ids.iter().enumerate() {
        if i > 0 {
            joined.push(',');
        }
        joined.push_str(&id.to_string());
    }
    format!("t.rowid IN ({joined})")
}

/// Does this sort spec read the `name` column? Stage 1 only projects
/// (and joins for) `name` when a filter or a sort actually needs it.
fn sort_uses_name(spec: &SortSpec) -> bool {
    spec.fields()
        .iter()
        .any(|f| matches!(SortKey::from_field(f), Ok((SortKey::Name, _))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use veloq_core::VeloqDiagnostic;
    use veloq_core::query::glob_sql_like;

    #[test]
    fn glob_translation() {
        assert_eq!(glob_sql_like("foo*bar"), "foo%bar");
        assert_eq!(glob_sql_like("ker?el"), "ker_el");
        // SQL specials inside the glob get escaped so they're literal:
        assert_eq!(glob_sql_like("100%_case"), "100\\%\\_case");
    }

    #[test]
    fn hydrate_ranked_survivors_prepare_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;

        let err = match hydrate_ranked_survivors(&conn, "SELECT * FROM", &[]) {
            Ok(rows) => anyhow::bail!(
                "malformed search rank SQL should not hydrate successfully: {} rows",
                rows.0.len()
            ),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-prepare");
        assert_eq!(
            err.sql_parts(),
            Some(("search", crate::SqlPhase::Prepare, SEARCH_RANK_SQL))
        );
        Ok(())
    }

    #[test]
    fn hydrate_ranked_survivors_query_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = "SELECT ? AS kind, 1::BIGINT AS row_id_num, 1::BIGINT AS total_matched";

        let err = match hydrate_ranked_survivors(&conn, sql, &[]) {
            Ok(rows) => anyhow::bail!(
                "unbound search rank SQL should not hydrate successfully: {} rows",
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
    fn hydrate_ranked_survivors_read_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql =
            "SELECT 'kernel' AS kind, 'not-a-rowid' AS row_id_num, 1::BIGINT AS total_matched";

        let err = match hydrate_ranked_survivors(&conn, sql, &[]) {
            Ok(rows) => anyhow::bail!(
                "malformed search rank row should not hydrate successfully: {} rows",
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
    fn hydrate_ranked_survivors_kind_tag_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = "SELECT 'not_a_kind' AS kind, 1::BIGINT AS row_id_num, \
                          NULL::BIGINT AS process_id, 1::BIGINT AS total_matched";

        let err = match hydrate_ranked_survivors(&conn, sql, &[]) {
            Ok(rows) => anyhow::bail!(
                "unknown search rank kind should not hydrate successfully: {} rows",
                rows.0.len()
            ),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.internal.sql-kind-tag-invalid");
        assert!(matches!(
            err,
            crate::NsysQueryError::InternalSqlKindTagInvalid { .. }
        ));
        Ok(())
    }

    #[test]
    fn hydrate_search_rows_prepare_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;

        let err = match hydrate_search_rows(&conn, "SELECT * FROM", 1, None) {
            Ok(rows) => anyhow::bail!(
                "malformed search hydrate SQL should not hydrate successfully: {} rows",
                rows.len()
            ),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-prepare");
        assert_eq!(
            err.sql_parts(),
            Some(("search", crate::SqlPhase::Prepare, SEARCH_HYDRATE_SQL))
        );
        Ok(())
    }

    #[test]
    fn hydrate_search_rows_query_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;

        let err = match hydrate_search_rows(&conn, "SELECT ? AS kind", 1, None) {
            Ok(rows) => anyhow::bail!(
                "unbound search hydrate SQL should not hydrate successfully: {} rows",
                rows.len()
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
    fn hydrate_search_rows_read_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = "SELECT 'kernel' AS kind, 'not-a-rowid' AS row_id_num";

        let err = match hydrate_search_rows(&conn, sql, 1, None) {
            Ok(rows) => anyhow::bail!(
                "malformed search hydrate row should not hydrate successfully: {} rows",
                rows.len()
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
}
