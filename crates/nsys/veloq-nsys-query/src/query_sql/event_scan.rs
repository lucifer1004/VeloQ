use crate::query_sql::event_semantics::EventSemantics;
use crate::{EventKind, NsysQueryError, NsysQueryResult};
use duckdb::types::Value;
use veloq_query::sql::{SqlFilter, SqlFragment, window};

const NO_INTRINSIC_PREDICATES: &[&str] = &[];
const NVTX_RANGE_PREDICATES: &[&str] = &[r#"t."end" IS NOT NULL"#];

/// Reusable stats scan facts for one event table.
///
/// This is deliberately narrower than a stats query builder: it owns the
/// common per-kind scan semantics, while `stats` keeps its group-by, NVTX
/// parent, mangled-name, and grid/block projections local.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StatsEventScan {
    pub(crate) table: &'static str,
    pub(crate) label: &'static str,
    pub(crate) display_expr: String,
    pub(crate) short_expr: String,
    pub(crate) raw_display_expr: &'static str,
    pub(crate) name_joins: &'static str,
    pub(crate) duration_expr: String,
    pub(crate) device_expr: &'static str,
    pub(crate) context_expr: &'static str,
    pub(crate) stream_expr: &'static str,
    pub(crate) bytes_expr: &'static str,
    pub(crate) graph_id_expr: &'static str,
    pub(crate) graph_node_id_expr: &'static str,
    pub(crate) event_type_expr: &'static str,
    pub(crate) where_clause: String,
    pub(crate) params: Vec<Value>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct EventScanFilterOptions<'a> {
    pub(crate) abs_window: Option<(i64, i64)>,
    pub(crate) device: Option<i32>,
    pub(crate) stream: Option<i64>,
    pub(crate) nvtx_scope: crate::nvtx_attribution::NvtxScope,
    pub(crate) nvtx_policy: NvtxFilterPolicy<'a>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum NvtxFilterPolicy<'a> {
    EmptyWhenUnsupported,
    ErrorUnlessKindIn {
        verb: &'static str,
        allowed: &'a [EventKind],
    },
}

/// Thin per-kind projection for search-like ranking scans.
///
/// This intentionally avoids rich headline columns. Callers materialize
/// those later after `LIMIT` has selected survivor rowids.
pub(crate) fn search_rank_select(
    trace: &veloq_nsys_data::Trace,
    kind: EventKind,
    nvtx_scope: crate::nvtx_attribution::NvtxScope,
    include_name: bool,
    use_prefilter: bool,
) -> NsysQueryResult<SqlFragment> {
    if matches!(kind, EventKind::CpuSample) {
        return Err(NsysQueryError::SearchCpuSampleUnsupported);
    }

    let sem = EventSemantics::new(kind);
    let start_col = if matches!(kind, EventKind::CudaEvent) {
        "t.timestamp"
    } else {
        "t.start"
    };
    let process =
        veloq_nsys_data::process_sql_projection(trace, sem.table(), "t", "event_proc", start_col);
    let (start_expr, duration_expr) = start_duration_exprs(kind);
    let (name_projection, name_joins) = name_projection(sem, include_name);
    let mut filters = SqlFilter::default();

    if nvtx_scope.is_attributed()
        && let Some(predicate) = sem.attribution_filter("t")
    {
        filters.push_predicate(predicate);
    }
    if matches!(kind, EventKind::Nvtx) {
        filters.push_predicate(r#"t."end" IS NOT NULL"#);
    }
    if use_prefilter && let Some(predicate) = name_match_prefilter(kind) {
        filters.push_predicate(predicate);
    }

    let where_clause = filters.where_clause();
    let sql = format!(
        r#"
        SELECT
            '{label}' AS kind,
            t.rowid AS row_id_num,
            {start_expr} AS start_ns,
            {duration_expr} AS duration_ns,
            {process_expr} AS process_id,
            {device_expr} AS device_id,
            {stream_expr} AS stream_id{name_projection}
        FROM nsight.{table} t {name_joins} {process_join}
        {where_clause}
        "#,
        label = sem.label(),
        process_expr = process.expr,
        process_join = process.join,
        device_expr = sem.device_expr(),
        stream_expr = sem.stream_expr(),
        table = sem.table(),
    );
    Ok(SqlFragment::new(sql, filters.into_params()))
}

/// Build the common scan semantics for one `stats` event table.
///
/// When a time window is present the returned params are ordered as:
/// clipped-duration `end,start`, then overlap-filter `end,start`.
pub(crate) fn stats_event_scan(
    kind: EventKind,
    abs_window: Option<(i64, i64)>,
    nvtx_scope: crate::nvtx_attribution::NvtxScope,
    collapse_versioned: bool,
) -> NsysQueryResult<StatsEventScan> {
    ensure_stats_scan_kind(kind)?;
    let sem = EventSemantics::new(kind);
    let (display_expr, short_expr) = stats_name_exprs(sem, collapse_versioned);

    let duration = window::clipped_duration_expr("t", abs_window);
    let duration_expr = duration.sql;
    let mut params = duration.params;

    let filter = stats_filter(sem, abs_window, nvtx_scope)?;
    let where_clause = filter.where_clause();
    params.extend(filter.into_params());

    Ok(StatsEventScan {
        table: sem.table(),
        label: sem.label(),
        display_expr,
        short_expr,
        raw_display_expr: sem.display_name_expr(),
        name_joins: sem.name_joins(),
        duration_expr,
        device_expr: sem.device_expr(),
        context_expr: sem.context_expr(),
        stream_expr: sem.stream_expr(),
        bytes_expr: sem.stats_bytes_expr(),
        graph_id_expr: sem.graph_id_expr(),
        graph_node_id_expr: sem.graph_node_id_expr(),
        event_type_expr: sem.event_type_expr(),
        where_clause,
        params,
    })
}

/// Common per-event scan filters.
///
/// Bind order is always time window `end,start`, then device, then stream.
/// NVTX attribution predicates do not introduce bind params.
pub(crate) fn event_scan_filter(
    sem: EventSemantics,
    options: EventScanFilterOptions<'_>,
    intrinsic_predicates: &[&str],
) -> NsysQueryResult<SqlFilter> {
    let mut filter = SqlFilter::default();

    if let Some(fragment) = window::overlap_filter("t", options.abs_window) {
        filter.push_fragment(fragment);
    }
    if let Some(device) = options.device {
        filter.push_predicate(format!("{} = ?", sem.device_expr()));
        filter.push_param(Value::Int(device));
    }
    if let Some(stream) = options.stream {
        filter.push_predicate(format!("{} = ?", sem.stream_expr()));
        filter.push_param(Value::BigInt(stream));
    }
    for predicate in intrinsic_predicates {
        filter.push_predicate(*predicate);
    }
    push_nvtx_attribution_filter(&mut filter, sem, options.nvtx_scope, options.nvtx_policy)?;

    Ok(filter)
}

fn ensure_stats_scan_kind(kind: EventKind) -> NsysQueryResult<()> {
    match kind {
        EventKind::Kernel
        | EventKind::Memcpy
        | EventKind::Memset
        | EventKind::Sync
        | EventKind::Graph
        | EventKind::Nvtx
        | EventKind::Runtime
        | EventKind::Osrt => Ok(()),
        EventKind::GraphNode
        | EventKind::GraphEvent
        | EventKind::CudaEvent
        | EventKind::Overhead
        | EventKind::CpuSample => Err(NsysQueryError::internal_unsupported_kind(
            "stats",
            kind.as_str(),
        )),
    }
}

fn stats_name_exprs(sem: EventSemantics, collapse_versioned: bool) -> (String, String) {
    if collapse_versioned && matches!(sem.kind(), EventKind::Runtime) {
        // Runtime-only API-version collapse, matching nsys's
        // `cuda_api_sum` recipe (`cudaMalloc_v3020` -> `cudaMalloc`).
        let display = sem.display_name_expr();
        let short = sem.short_name_expr();
        (
            format!("regexp_replace({display}, '_v[0-9]+$', '')"),
            format!("regexp_replace({short}, '_v[0-9]+$', '')"),
        )
    } else {
        (
            sem.display_name_expr().to_string(),
            sem.short_name_expr().to_string(),
        )
    }
}

fn stats_filter(
    sem: EventSemantics,
    abs_window: Option<(i64, i64)>,
    nvtx_scope: crate::nvtx_attribution::NvtxScope,
) -> NsysQueryResult<SqlFilter> {
    let intrinsic = if matches!(sem.kind(), EventKind::Nvtx) {
        // NVTX marks have a NULL end and no duration.
        NVTX_RANGE_PREDICATES
    } else {
        NO_INTRINSIC_PREDICATES
    };
    event_scan_filter(
        sem,
        EventScanFilterOptions {
            abs_window,
            device: None,
            stream: None,
            nvtx_scope,
            nvtx_policy: NvtxFilterPolicy::EmptyWhenUnsupported,
        },
        intrinsic,
    )
}

fn push_nvtx_attribution_filter(
    filter: &mut SqlFilter,
    sem: EventSemantics,
    nvtx_scope: crate::nvtx_attribution::NvtxScope,
    policy: NvtxFilterPolicy<'_>,
) -> NsysQueryResult<()> {
    if !nvtx_scope.is_attributed() {
        return Ok(());
    }

    if let NvtxFilterPolicy::ErrorUnlessKindIn { verb, allowed } = policy
        && !allowed.contains(&sem.kind())
    {
        return Err(NsysQueryError::internal_nvtx_attribution_unsupported_kind(
            verb,
            sem.label(),
        ));
    }

    match sem.attribution_filter("t") {
        Some(predicate) => filter.push_predicate(predicate),
        // Kinds without an attribution path emit no rows under `--nvtx`,
        // preserving the surrounding UNION ALL shape.
        None => filter.push_predicate("FALSE"),
    }

    Ok(())
}

fn start_duration_exprs(kind: EventKind) -> (&'static str, &'static str) {
    match kind {
        EventKind::CudaEvent => ("t.timestamp", "0"),
        _ => ("t.start", r#"(t."end" - t.start)"#),
    }
}

fn name_projection(sem: EventSemantics, include_name: bool) -> (String, &'static str) {
    if !include_name {
        return (String::new(), "");
    }
    (
        format!(", {} AS name", search_name_expr(sem)),
        sem.name_joins(),
    )
}

fn search_name_expr(sem: EventSemantics) -> &'static str {
    match sem.kind() {
        EventKind::Nvtx => "COALESCE(t.text, s_text.value, '<unnamed nvtx>')",
        _ => sem.display_name_expr(),
    }
}

fn name_match_prefilter(kind: EventKind) -> Option<&'static str> {
    match kind {
        EventKind::Kernel => Some(
            "(t.demangledName IN (SELECT id FROM name_match_ids) \
             OR t.shortName IN (SELECT id FROM name_match_ids) \
             OR t.demangledName IS NULL OR t.shortName IS NULL)",
        ),
        EventKind::Runtime | EventKind::Osrt => {
            Some("(t.nameId IN (SELECT id FROM name_match_ids) OR t.nameId IS NULL)")
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use duckdb::Connection;
    use tempfile::TempDir;
    use veloq_nsys_data::Trace;

    fn minimal_trace() -> Result<(TempDir, Trace)> {
        let dir = tempfile::tempdir()?;
        let pqtdir = dir.path().join("test_pqtdir");
        std::fs::create_dir_all(&pqtdir)?;
        let conn = Connection::open_in_memory()?;
        let tables = [
            (
                "CUPTI_ACTIVITY_KIND_KERNEL",
                r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start BIGINT, "end" BIGINT)"#,
            ),
            (
                "CUPTI_ACTIVITY_KIND_CUDA_EVENT",
                "CREATE TABLE CUPTI_ACTIVITY_KIND_CUDA_EVENT (timestamp BIGINT)",
            ),
            (
                "NVTX_EVENTS",
                r#"CREATE TABLE NVTX_EVENTS (start BIGINT, "end" BIGINT)"#,
            ),
        ];
        for (table, ddl) in tables {
            conn.execute_batch(ddl)?;
            let out = pqtdir.join(format!("{table}.parquet"));
            let out_lit = out.to_string_lossy().replace('\'', "''");
            conn.execute(
                &format!(r#"COPY (SELECT * FROM "{table}") TO '{out_lit}' (FORMAT PARQUET)"#),
                [],
            )?;
        }
        let trace = Trace::open(&pqtdir)?;
        Ok((dir, trace))
    }

    #[test]
    fn scan_filter_orders_window_location_and_nvtx_predicates() -> Result<()> {
        let filter = event_scan_filter(
            EventSemantics::new(EventKind::Kernel),
            EventScanFilterOptions {
                abs_window: Some((10, 20)),
                device: Some(3),
                stream: Some(7),
                nvtx_scope: crate::nvtx_attribution::NvtxScope::Attributed,
                nvtx_policy: NvtxFilterPolicy::EmptyWhenUnsupported,
            },
            &[],
        )?;

        assert_eq!(
            filter.where_clause(),
            format!(
                r#"WHERE t.start < ? AND t."end" > ? AND {} = ? AND {} = ? AND {}"#,
                crate::kind_sql::GPU_DEVICE_ID_EXPR,
                crate::kind_sql::GPU_STREAM_ID_EXPR,
                crate::nvtx_attribution::filter_clause(crate::nvtx_attribution::KERNEL_VIEW, "t")
            )
        );
        assert_eq!(
            filter.into_params(),
            vec![
                Value::BigInt(20),
                Value::BigInt(10),
                Value::Int(3),
                Value::BigInt(7)
            ]
        );
        Ok(())
    }

    #[test]
    fn scan_filter_inserts_intrinsic_predicates_before_nvtx() -> Result<()> {
        let filter = event_scan_filter(
            EventSemantics::new(EventKind::Memcpy),
            EventScanFilterOptions {
                abs_window: None,
                device: None,
                stream: None,
                nvtx_scope: crate::nvtx_attribution::NvtxScope::Attributed,
                nvtx_policy: NvtxFilterPolicy::EmptyWhenUnsupported,
            },
            &["t.bytes IS NOT NULL"],
        )?;

        assert!(
            filter
                .where_clause()
                .starts_with("WHERE t.bytes IS NOT NULL AND")
        );
        assert!(filter.into_params().is_empty());
        Ok(())
    }

    #[test]
    fn scan_filter_can_error_for_policy_disallowed_nvtx_kind() {
        let result = event_scan_filter(
            EventSemantics::new(EventKind::Sync),
            EventScanFilterOptions {
                abs_window: None,
                device: None,
                stream: None,
                nvtx_scope: crate::nvtx_attribution::NvtxScope::Attributed,
                nvtx_policy: NvtxFilterPolicy::ErrorUnlessKindIn {
                    verb: "timeline",
                    allowed: &[EventKind::Kernel],
                },
            },
            &[],
        );

        assert!(matches!(
            result,
            Err(NsysQueryError::InternalNvtxAttributionUnsupportedKind {
                verb: "timeline",
                kind: "sync"
            })
        ));
    }

    #[test]
    fn scan_filter_emits_false_for_unattributable_allowed_kind() -> Result<()> {
        let filter = event_scan_filter(
            EventSemantics::new(EventKind::Graph),
            EventScanFilterOptions {
                abs_window: None,
                device: None,
                stream: None,
                nvtx_scope: crate::nvtx_attribution::NvtxScope::Attributed,
                nvtx_policy: NvtxFilterPolicy::EmptyWhenUnsupported,
            },
            &[],
        )?;

        assert_eq!(filter.where_clause(), "WHERE FALSE");
        assert!(filter.into_params().is_empty());
        Ok(())
    }

    #[test]
    fn stats_scan_binds_duration_params_before_overlap_params() -> Result<()> {
        let scan = stats_event_scan(
            EventKind::Kernel,
            Some((10, 20)),
            crate::nvtx_attribution::NvtxScope::None,
            false,
        )?;
        assert_eq!(
            scan.duration_expr,
            r#"LEAST(t."end", ?) - GREATEST(t.start, ?)"#
        );
        assert_eq!(scan.where_clause, r#"WHERE t.start < ? AND t."end" > ?"#);
        assert_eq!(
            scan.params,
            vec![
                Value::BigInt(20),
                Value::BigInt(10),
                Value::BigInt(20),
                Value::BigInt(10)
            ]
        );
        assert_eq!(scan.device_expr, crate::kind_sql::GPU_DEVICE_ID_EXPR);
        Ok(())
    }

    #[test]
    fn stats_scan_collapses_runtime_names_only_for_display_axes() -> Result<()> {
        let scan = stats_event_scan(
            EventKind::Runtime,
            None,
            crate::nvtx_attribution::NvtxScope::None,
            true,
        )?;
        assert!(scan.display_expr.starts_with("regexp_replace("));
        assert!(scan.short_expr.starts_with("regexp_replace("));
        assert_eq!(
            scan.raw_display_expr,
            "COALESCE(s_rt.value, '<unknown runtime>')"
        );
        assert_eq!(scan.device_expr, "CAST(NULL AS INTEGER)");
        assert!(scan.params.is_empty());
        Ok(())
    }

    #[test]
    fn stats_scan_filters_nvtx_instants_and_opaque_attribution() -> Result<()> {
        let nvtx = stats_event_scan(
            EventKind::Nvtx,
            None,
            crate::nvtx_attribution::NvtxScope::None,
            false,
        )?;
        assert_eq!(nvtx.where_clause, r#"WHERE t."end" IS NOT NULL"#);

        let graph = stats_event_scan(
            EventKind::Graph,
            None,
            crate::nvtx_attribution::NvtxScope::Attributed,
            false,
        )?;
        assert_eq!(graph.where_clause, "WHERE FALSE");
        assert!(graph.params.is_empty());
        Ok(())
    }

    #[test]
    fn rank_select_uses_cuda_event_timestamp_and_zero_duration() -> Result<()> {
        let (_dir, trace) = minimal_trace()?;
        let fragment = search_rank_select(
            &trace,
            EventKind::CudaEvent,
            crate::nvtx_attribution::NvtxScope::None,
            false,
            false,
        )?;
        assert!(fragment.sql.contains("t.timestamp AS start_ns"));
        assert!(fragment.sql.contains("0 AS duration_ns"));
        assert!(fragment.sql.contains(crate::kind_sql::GPU_DEVICE_ID_EXPR));
        assert!(fragment.params.is_empty());
        Ok(())
    }

    #[test]
    fn rank_select_adds_nvtx_intrinsic_filter_and_search_name() -> Result<()> {
        let (_dir, trace) = minimal_trace()?;
        let fragment = search_rank_select(
            &trace,
            EventKind::Nvtx,
            crate::nvtx_attribution::NvtxScope::None,
            true,
            false,
        )?;
        assert!(
            fragment
                .sql
                .contains("COALESCE(t.text, s_text.value, '<unnamed nvtx>') AS name")
        );
        assert!(fragment.sql.contains(r#"WHERE t."end" IS NOT NULL"#));
        assert!(fragment.params.is_empty());
        Ok(())
    }

    #[test]
    fn rank_select_adds_attribution_and_prefilter() -> Result<()> {
        let (_dir, trace) = minimal_trace()?;
        let fragment = search_rank_select(
            &trace,
            EventKind::Kernel,
            crate::nvtx_attribution::NvtxScope::Attributed,
            true,
            true,
        )?;
        assert!(fragment.sql.contains("attributed_kernel_rowids"));
        assert!(fragment.sql.contains("name_match_ids"));
        assert!(
            fragment
                .sql
                .contains(crate::kind_sql::KERNEL_STRINGIDS_JOINS)
        );
        assert!(fragment.params.is_empty());
        Ok(())
    }
}
