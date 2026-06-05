//! `veloq search <trace> ...` — filter events into a list of `row_id`s
//! plus a few headline columns. Designed as the `inspect` entry-point.

use anyhow::{Context, Result};
use duckdb::types::Value;
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;
use veloq_core::{
    Direction, SortKeyDef, SortKeySpec, SortSpec,
    time::{DurationFilter, TimeWindow},
};

use crate::column_map::{self, ColumnMap, maybe_col, opt_string};
use crate::event_ref::{
    EventRefBase, EventRefKernel, EventRefMemcpy, EventRefMemset, EventRefNvtx,
};
use crate::{EventKind, EventRef, KindFilter, RowId};

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
fn sort_sql(spec: &SortSpec) -> anyhow::Result<String> {
    let mut resolved: Vec<(&'static str, Direction)> = Vec::new();
    for f in spec.fields() {
        let (key, dir) = SortKey::from_field(f)?;
        resolved.push((key.primary_column(), dir));
    }
    Ok(veloq_core::sort::build_order_by(&resolved, "row_id_num"))
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

pub fn run<P: AsRef<Path>>(path: P, req: SearchRequest) -> Result<SearchResponse> {
    let (trace, abs_window) = crate::open_scoped(path.as_ref(), req.limit, req.time_window)?;

    // Shared `--device` / `--stream` policy (see [`crate::kind_policy`]
    // for the rule and the wording rationale).
    crate::kind_policy::validate_location_filter(
        &req.kinds,
        crate::kind_policy::LocationFilter {
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

    if req.name_glob.is_some() && req.name_regex.is_some() {
        anyhow::bail!("`--name` and `--name-regex` are mutually exclusive; pick one");
    }
    if let Some(glob) = &req.name_glob {
        // ESCAPE takes exactly one character in DuckDB. Both Rust source
        // and SQL need a single backslash, hence the doubled escape here.
        where_parts.push(r"name LIKE ? ESCAPE '\'".to_string());
        params.push(Value::Text(crate::search_glob_to_like(glob)));
    }
    if let Some(re) = &req.name_regex {
        // DuckDB's regexp_matches is PCRE-flavoured. We pass the
        // pattern verbatim; no special escaping needed beyond what
        // the user already wrote.
        where_parts.push("regexp_matches(name, ?)".to_string());
        params.push(Value::Text(re.clone()));
    }

    crate::kind_policy::LocationFilter {
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

    if let Some((s, e)) = abs_window {
        // overlap predicate: event_start < window_end AND event_end > window_start
        where_parts.push("start_ns < ? AND (start_ns + duration_ns) > ?".to_string());
        params.push(Value::BigInt(e));
        params.push(Value::BigInt(s));
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
    for k in &kinds {
        rank_subqueries.push(per_kind_rank_select(
            *k,
            nvtx_scope,
            include_name,
            use_prefilter,
        )?);
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
        SELECT kind, row_id_num,
               CAST(COUNT(*) OVER () AS BIGINT) AS total_matched
        FROM ({rank_union})
        {where_clause}
        ORDER BY {order_by}
        LIMIT ?
        "#
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
    rank_params.extend(params.iter().cloned());
    rank_params.push(Value::BigInt(req.limit as i64));

    let conn = trace.conn();
    let mut survivors: Vec<(EventKind, i64)> = Vec::with_capacity(req.limit);
    let mut total_matched: i64 = 0;
    {
        let mut stmt = conn
            .prepare(&rank_sql)
            .context("failed to prepare search rank SQL")?;
        let bound = crate::bind(&rank_params);
        let mut rows = stmt.query(bound.as_slice())?;
        while let Some(row) = rows.next()? {
            let kind_str: String = row.get(0)?;
            let kind = EventKind::parse(&kind_str)
                .with_context(|| format!("unrecognised kind `{kind_str}` from SQL"))?;
            let rowid_num: i64 = row.get(1)?;
            // `total_matched` is identical on every row (window over the
            // unbounded partition); the last write wins.
            total_matched = row.get(2)?;
            survivors.push((kind, rowid_num));
        }
    }

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
                .context("computing NVTX nesting depth for search")?,
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
    let cols = column_map::load_standard(conn)
        .context("loading schema column map for search subqueries")?;
    let mut wide_subqueries = Vec::new();
    for k in &kinds {
        let ids: Vec<i64> = survivors
            .iter()
            .filter(|(sk, _)| sk == k)
            .map(|(_, r)| *r)
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

    let mut by_id: HashMap<RowId, EventRef> = HashMap::with_capacity(survivors.len());
    let mut stmt = conn.prepare(&sql).context("failed to prepare search SQL")?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let kind_str: String = row.get(COL_KIND)?;
        let kind = EventKind::parse(&kind_str)
            .with_context(|| format!("unrecognised kind `{kind_str}` from SQL"))?;
        let rowid_num: i64 = row.get(COL_ROW_ID_NUM)?;
        let name: String = row.get(COL_NAME)?;
        let start_ns: i64 = row.get(COL_START_NS)?;
        let duration_ns: i64 = row.get(COL_DURATION_NS)?;
        let device_id: Option<i32> = row.get(COL_DEVICE_ID)?;
        let stream_id: Option<i64> = row.get(COL_STREAM_ID)?;
        let global_tid: Option<i64> = row.get(COL_GLOBAL_TID)?;

        // Populate `depth` only for NVTX hits, and only when we actually
        // computed the nesting map. Lookup miss (e.g. an instant marker
        // whose row predates the nesting scan, though `compute` covers
        // those) falls back to `None` rather than spelling a default.
        let depth = match (kind, nesting.as_ref()) {
            (EventKind::Nvtx, Some(map)) => map.get(&rowid_num).map(|e| e.depth),
            _ => None,
        };

        let row_id = RowId::new(kind, rowid_num);
        let base = EventRefBase {
            key: row_id.to_string(),
            row_id,
            name,
            start_ns,
            duration_ns,
            device_id,
            stream_id,
            global_tid,
            depth,
            // `nvtx_context` is populated by the `--with-nvtx`
            // post-decoration pass below; left None on construction.
            nvtx_context: None,
        };

        let event_ref = match kind {
            EventKind::Kernel => {
                let grid = build_xyz(row, COL_GRID_X, COL_GRID_Y, COL_GRID_Z)?;
                let block = build_xyz(row, COL_BLOCK_X, COL_BLOCK_Y, COL_BLOCK_Z)?;
                let registers_per_thread = row.get(COL_REGISTERS_PER_THREAD)?;
                let static_shared_memory = row.get(COL_STATIC_SHARED_MEMORY)?;
                let dynamic_shared_memory = row.get(COL_DYNAMIC_SHARED_MEMORY)?;
                let demangled_name = opt_string(row, COL_DEMANGLED_NAME)?;
                let mangled_name = opt_string(row, COL_MANGLED_NAME)?;
                EventRef::Kernel(EventRefKernel {
                    base,
                    grid,
                    block,
                    registers_per_thread,
                    static_shared_memory,
                    dynamic_shared_memory,
                    demangled_name,
                    mangled_name,
                })
            }
            EventKind::Memcpy => {
                let bytes: Option<i64> = row.get(COL_BYTES)?;
                let copy_kind: Option<i64> = row.get(COL_COPY_KIND)?;
                let copy_kind_name = copy_kind.map(crate::kind_sql::copy_kind_label);
                EventRef::Memcpy(EventRefMemcpy {
                    base,
                    bytes,
                    copy_kind,
                    copy_kind_name,
                })
            }
            EventKind::Memset => {
                let bytes: Option<i64> = row.get(COL_BYTES)?;
                let value: Option<i64> = row.get(COL_MEMSET_VALUE)?;
                EventRef::Memset(EventRefMemset { base, bytes, value })
            }
            EventKind::Nvtx => {
                let event_type: Option<i64> = row.get(COL_EVENT_TYPE)?;
                let domain_id: Option<i64> = row.get(COL_DOMAIN_ID)?;
                EventRef::Nvtx(EventRefNvtx {
                    base,
                    event_type,
                    domain_id,
                })
            }
            // Non-extended kinds carry just the shared base.
            _ => EventRef::from_base(kind, base)?,
        };
        by_id.insert(row_id, event_ref);
    }

    // Re-apply the stage-1 ordering: stage 2 fetched survivors by rowid
    // (arbitrary order), so walk the ranked survivor list and pull each
    // materialized row back out in that order.
    let mut events: Vec<EventRef> = Vec::with_capacity(survivors.len());
    for (k, r) in &survivors {
        if let Some(ev) = by_id.remove(&RowId::new(*k, *r)) {
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
        let contexts = crate::nvtx_reverse::lookup_for_row_ids(&trace, &row_ids, nesting_map)
            .context("batched NVTX reverse attribution for search")?;
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

/// Read three columns at `xi`/`yi`/`zi` into an `Option<[i64; 3]>`.
/// All three present → `Some(...)`; any NULL → `None`. Used to
/// assemble kernel `grid`/`block` triples in the row builder.
fn build_xyz(row: &duckdb::Row, xi: usize, yi: usize, zi: usize) -> Result<Option<[i64; 3]>> {
    let x: Option<i64> = row.get(xi)?;
    let y: Option<i64> = row.get(yi)?;
    let z: Option<i64> = row.get(zi)?;
    Ok(match (x, y, z) {
        (Some(x), Some(y), Some(z)) => Some([x, y, z]),
        _ => None,
    })
}

fn per_kind_select(
    kind: EventKind,
    nvtx_scope: crate::nvtx_attribution::NvtxScope,
    cols: &ColumnMap,
    rowid_filter: Option<&[i64]>,
) -> Result<String> {
    let table = kind.table();
    let label = kind.as_str();
    let dev = crate::kind_sql::GPU_DEVICE_ID_EXPR;
    let stm = crate::kind_sql::GPU_STREAM_ID_EXPR;
    // Predicates pushed into this kind's scan ahead of the StringIds
    // joins: the NVTX attribution constraint (when `--nvtx` is active and
    // the kind is attributable — Kernel/Memcpy/Memset/Sync/Runtime) and,
    // in stage 2, an explicit survivor-rowid filter. Both reference only
    // base columns, so DuckDB pushes them down to the scan, keeping the
    // joins off every non-matching row.
    let mut base_preds: Vec<String> = Vec::new();
    if let Some(p) = attribution_pred(kind, nvtx_scope) {
        base_preds.push(p);
    }
    if let Some(ids) = rowid_filter {
        base_preds.push(rowid_in_list(ids));
    }
    let where_clause = build_where(&base_preds);
    let global_tid = veloq_nsys_data::sql_expr::u64_bits_to_i64("t.globalTid");
    Ok(match kind {
        EventKind::Kernel => {
            // Kernel — the extended kind with the largest headline
            // payload (grid/block/registers/shared/demangled/mangled).
            // grid* / block* are mandatory in the CUPTI table;
            // registers / shared memory / mangled are probed via
            // `maybe_col` so older NSys schemas degrade to NULL.
            const T: &str = "CUPTI_ACTIVITY_KIND_KERNEL";
            let name_expr = crate::kind_sql::display_name_expr(kind);
            let joins = crate::kind_sql::name_joins(kind);
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
            let name_expr = crate::kind_sql::display_name_expr(kind);
            let joins = crate::kind_sql::name_joins(kind);
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
            let name_expr = crate::kind_sql::display_name_expr(kind);
            let joins = crate::kind_sql::name_joins(kind);
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
            let name_expr = crate::kind_sql::display_name_expr(kind);
            let joins = crate::kind_sql::name_joins(kind);
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
            let where_clause = build_where(&preds);
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
            let name_expr = crate::kind_sql::display_name_expr(kind);
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
            let name_expr = crate::kind_sql::display_name_expr(kind);
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
            let name_expr = crate::kind_sql::display_name_expr(kind);
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
        EventKind::CpuSample => anyhow::bail!(
            "internal: search doesn't surface cpu_sample rows; \
             use `veloq metrics --type cpu-sampling` or \
             `veloq inspect cpu_sample:<id>` instead"
        ),
    })
}

/// Thin per-kind projection for stage-1 ranking: rowid plus the columns
/// the sort/filter touch — `name` only when `include_name`. Mirrors
/// [`per_kind_select`]'s start/duration/device/stream/name and intrinsic
/// filters per kind exactly, so the ranked set matches what stage 2
/// re-materializes — but without the headline columns or (when a name
/// isn't read) any StringIds join.
fn per_kind_rank_select(
    kind: EventKind,
    nvtx_scope: crate::nvtx_attribution::NvtxScope,
    include_name: bool,
    use_prefilter: bool,
) -> Result<String> {
    if matches!(kind, EventKind::CpuSample) {
        anyhow::bail!(
            "internal: search doesn't surface cpu_sample rows; \
             use `veloq metrics --type cpu-sampling` or \
             `veloq inspect cpu_sample:<id>` instead"
        );
    }
    let table = kind.table();
    let label = kind.as_str();
    let dev = crate::kind_sql::GPU_DEVICE_ID_EXPR;
    let stm = crate::kind_sql::GPU_STREAM_ID_EXPR;

    // device/stream carry only on the GPU kinds (matches per_kind_select).
    let (dev_expr, stm_expr) = match kind {
        EventKind::Kernel
        | EventKind::Memcpy
        | EventKind::Memset
        | EventKind::Sync
        | EventKind::Graph
        | EventKind::CudaEvent => (dev, stm),
        _ => ("CAST(NULL AS INTEGER)", "CAST(NULL AS BIGINT)"),
    };
    // CUDA_EVENT rows have only a `timestamp` (no end); everything else is
    // a start/end span.
    let (start_expr, dur_expr) = match kind {
        EventKind::CudaEvent => ("t.timestamp", "0"),
        _ => ("t.start", r#"(t."end" - t.start)"#),
    };

    // Name expression + its joins, only when a name filter/sort reads it.
    // Each resolves to the SAME value per_kind_select projects, so
    // filtering/ordering on it is identical.
    let (name_proj, name_joins) = if include_name {
        let expr = match kind {
            // per_kind_select hand-rolls NVTX's name with the
            // '<unnamed nvtx>' fallback (display_name_expr uses a
            // different literal); match it here for an identical value.
            EventKind::Nvtx => "COALESCE(t.text, s_text.value, '<unnamed nvtx>')".to_string(),
            _ => crate::kind_sql::display_name_expr(kind).to_string(),
        };
        (
            format!(", {expr} AS name"),
            crate::kind_sql::name_joins(kind),
        )
    } else {
        (String::new(), "")
    };

    // Same constraints per_kind_select applies (attribution when scoped,
    // the NVTX end-not-null intrinsic), so the ranked rowids are exactly
    // the ones stage 2 can re-fetch.
    let mut preds: Vec<String> = Vec::new();
    if let Some(p) = attribution_pred(kind, nvtx_scope) {
        preds.push(p);
    }
    if matches!(kind, EventKind::Nvtx) {
        preds.push(r#"t."end" IS NOT NULL"#.to_string());
    }
    // Name pre-filter (see the `name_match_ids` CTE in `run`): a pushed-
    // down superset of "this row's resolved name matches the pattern",
    // referencing only the raw id columns so DuckDB prunes the scan before
    // the StringIds joins. A `NULL` id component means the COALESCE falls
    // through to the next source (or the literal fallback), so those rows
    // must pass the pre-filter and are settled by the authoritative outer
    // `name LIKE/regexp` filter. Only the StringId-named kinds get this.
    if use_prefilter {
        match kind {
            EventKind::Kernel => preds.push(
                "(t.demangledName IN (SELECT id FROM name_match_ids) \
                 OR t.shortName IN (SELECT id FROM name_match_ids) \
                 OR t.demangledName IS NULL OR t.shortName IS NULL)"
                    .to_string(),
            ),
            EventKind::Runtime | EventKind::Osrt => preds.push(
                "(t.nameId IN (SELECT id FROM name_match_ids) OR t.nameId IS NULL)".to_string(),
            ),
            _ => {}
        }
    }
    let where_clause = build_where(&preds);

    Ok(format!(
        r#"
        SELECT
            '{label}' AS kind,
            t.rowid AS row_id_num,
            {start_expr} AS start_ns,
            {dur_expr} AS duration_ns,
            {dev_expr} AS device_id,
            {stm_expr} AS stream_id{name_proj}
        FROM nsight.{table} t {name_joins}
        {where_clause}
        "#
    ))
}

/// NVTX attribution predicate for one kind: when `--nvtx` scoping is
/// active and the kind is attributable (Kernel/Memcpy/Memset/Sync/
/// Runtime), constrain its scan to the rowids the attribution CTE
/// resolved. Non-attributable kinds — and the unscoped case — get `None`.
fn attribution_pred(
    kind: EventKind,
    nvtx_scope: crate::nvtx_attribution::NvtxScope,
) -> Option<String> {
    if !nvtx_scope.is_attributed() {
        return None;
    }
    let view = match kind {
        EventKind::Kernel => crate::nvtx_attribution::KERNEL_VIEW,
        EventKind::Memcpy => crate::nvtx_attribution::MEMCPY_VIEW,
        EventKind::Memset => crate::nvtx_attribution::MEMSET_VIEW,
        EventKind::Sync => crate::nvtx_attribution::SYNC_VIEW,
        EventKind::Runtime => crate::nvtx_attribution::RUNTIME_VIEW,
        _ => return None,
    };
    Some(format!("t.rowid IN (SELECT rowid FROM {view})"))
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

/// Join predicates into a `WHERE` clause, or the empty string if none.
fn build_where(preds: &[String]) -> String {
    if preds.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", preds.join(" AND "))
    }
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
    use crate::search_glob_to_like;

    #[test]
    fn glob_translation() {
        assert_eq!(search_glob_to_like("foo*bar"), "foo%bar");
        assert_eq!(search_glob_to_like("ker?el"), "ker_el");
        // SQL specials inside the glob get escaped so they're literal:
        assert_eq!(search_glob_to_like("100%_case"), "100\\%\\_case");
    }
}
