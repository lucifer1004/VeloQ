use crate::query_sql::{
    event_scan::{EventScanFilterOptions, NvtxFilterPolicy, event_scan_filter},
    event_semantics::EventSemantics,
    exec, gpu_work,
};
use crate::{EventKind, NsysQueryError, NsysQueryResult, RowId};
use veloq_nsys_data::{Trace, runtime_nvtx_parent};
use veloq_query::sql::SqlFragment;

#[derive(Debug, Clone)]
pub(super) struct TimelineEvent {
    pub(super) row_id: RowId,
    pub(super) kind: EventKind,
    pub(super) name: String,
    pub(super) full_name: String,
    pub(super) start_ns: i64,
    pub(super) end_ns: i64,
    pub(super) device_id: Option<i32>,
    pub(super) stream_id: Option<i64>,
    pub(super) nvtx_depth: Option<usize>,
}

impl TimelineEvent {
    pub(super) fn duration_ns(&self) -> i64 {
        self.end_ns.saturating_sub(self.start_ns)
    }
}
pub(super) fn query_gpu_events(
    trace: &Trace,
    abs_window: (i64, i64),
) -> NsysQueryResult<Vec<TimelineEvent>> {
    let work = gpu_work::GpuWorkSet::from_data_definition()?;
    let kinds = work.present_in(trace);
    if kinds.is_empty() {
        return Ok(Vec::new());
    }
    let mut subqueries = Vec::new();
    let mut params = Vec::new();
    for kind in kinds {
        let fragment = interval_select(kind, abs_window, None, None)?;
        subqueries.push(fragment.sql);
        params.extend(fragment.params);
    }
    let sql = subqueries.join(" UNION ALL ");
    exec::query_rows_fallible(
        trace.conn(),
        &sql,
        &params,
        exec::SqlLabel::new("viz-timeline", "gpu-events"),
        timeline_event_row,
    )
}

pub(super) fn query_runtime_events(
    trace: &Trace,
    abs_window: (i64, i64),
) -> NsysQueryResult<Vec<TimelineEvent>> {
    if !trace.table_exists(EventKind::Runtime.table()) {
        return Ok(Vec::new());
    }
    let fragment = interval_select(EventKind::Runtime, abs_window, None, None)?;
    exec::query_rows_fallible(
        trace.conn(),
        &fragment.sql,
        &fragment.params,
        exec::SqlLabel::new("viz-timeline", "cuda-api-events"),
        timeline_event_row,
    )
}

pub(super) fn query_nvtx_events(
    trace: &Trace,
    abs_window: (i64, i64),
) -> NsysQueryResult<Vec<TimelineEvent>> {
    if !trace.table_exists(EventKind::Nvtx.table()) {
        return Ok(Vec::new());
    }
    let sem = EventSemantics::new(EventKind::Nvtx);
    let intrinsic = vec![r#"t."end" IS NOT NULL"#];
    let filter = event_scan_filter(
        sem,
        EventScanFilterOptions {
            abs_window: Some(abs_window),
            device: None,
            stream: None,
            nvtx_scope: crate::nvtx_attribution::NvtxScope::None,
            nvtx_policy: NvtxFilterPolicy::EmptyWhenUnsupported,
        },
        &intrinsic,
    )?;
    let where_clause = filter.where_clause();
    let sidecar_path = runtime_nvtx_parent::ensure_sidecar(trace)
        .map_err(NsysQueryError::nvtx_parent_sidecar_ensure)?;
    let sidecar_quoted = crate::nvtx_projection::quote_sidecar_path(&sidecar_path);
    let sidecar_expanded_cte =
        crate::nvtx_projection::sidecar_expanded_cte("sidecar_expanded", &sidecar_quoted);
    let sql = format!(
        r#"
        WITH {sidecar_expanded_cte},
        nvtx_devices AS (
            SELECT nvtx_rowid,
                   CAST(device_id AS INTEGER) AS device_id
            FROM sidecar_expanded
            WHERE device_id IS NOT NULL
            GROUP BY nvtx_rowid, device_id
        )
        SELECT
            '{label}' AS kind,
            t.rowid AS row_id_num,
            {short_name_expr} AS name,
            {full_name_expr} AS full_name,
            t.start AS start_ns,
            COALESCE(t."end", t.start) AS end_ns,
            nvtx_dev.device_id AS device_id,
            CAST(NULL AS BIGINT) AS stream_id
        FROM nsight.{table} t {joins}
        LEFT JOIN nvtx_devices nvtx_dev
          ON nvtx_dev.nvtx_rowid = t.rowid
        {where_clause}
        "#,
        label = sem.label(),
        short_name_expr = sem.short_name_expr(),
        full_name_expr = sem.display_name_expr(),
        table = sem.table(),
        joins = sem.name_joins(),
    );
    let mut rows = exec::query_rows_fallible(
        trace.conn(),
        &sql,
        &filter.into_params(),
        exec::SqlLabel::new("viz-timeline", "nvtx-events"),
        timeline_event_row,
    )?;
    let nesting = trace
        .nvtx_nesting()
        .map_err(NsysQueryError::nvtx_nesting_load)?;
    for row in &mut rows {
        row.nvtx_depth = nesting
            .get(&row.row_id.rowid)
            .map(|entry| usize::from(entry.depth) + 1);
    }
    Ok(rows)
}

fn interval_select(
    kind: EventKind,
    abs_window: (i64, i64),
    device: Option<i32>,
    stream: Option<i64>,
) -> NsysQueryResult<SqlFragment> {
    let sem = EventSemantics::new(kind);
    let mut intrinsic = Vec::new();
    if matches!(kind, EventKind::Nvtx) {
        intrinsic.push(r#"t."end" IS NOT NULL"#);
    }
    let filter = event_scan_filter(
        sem,
        EventScanFilterOptions {
            abs_window: Some(abs_window),
            device,
            stream,
            nvtx_scope: crate::nvtx_attribution::NvtxScope::None,
            nvtx_policy: NvtxFilterPolicy::EmptyWhenUnsupported,
        },
        &intrinsic,
    )?;
    let where_clause = filter.where_clause();
    let sql = format!(
        r#"
        SELECT
            '{label}' AS kind,
            t.rowid AS row_id_num,
            {short_name_expr} AS name,
            {full_name_expr} AS full_name,
            t.start AS start_ns,
            COALESCE(t."end", t.start) AS end_ns,
            {device_expr} AS device_id,
            {stream_expr} AS stream_id
        FROM nsight.{table} t {joins}
        {where_clause}
        "#,
        label = sem.label(),
        short_name_expr = sem.short_name_expr(),
        full_name_expr = sem.display_name_expr(),
        device_expr = sem.device_expr(),
        stream_expr = sem.stream_expr(),
        table = sem.table(),
        joins = sem.name_joins(),
    );
    Ok(SqlFragment::new(sql, filter.into_params()))
}

fn timeline_event_row(row: &duckdb::Row<'_>) -> NsysQueryResult<TimelineEvent> {
    let kind_raw: String = row.get("kind").map_err(viz_timeline_row_read)?;
    let kind = EventKind::parse(&kind_raw)
        .ok_or_else(|| NsysQueryError::internal_sql_kind_tag_invalid("viz-timeline", &kind_raw))?;
    let rowid: i64 = row.get("row_id_num").map_err(viz_timeline_row_read)?;
    Ok(TimelineEvent {
        row_id: RowId::new(kind, rowid),
        kind,
        name: row.get("name").map_err(viz_timeline_row_read)?,
        full_name: row.get("full_name").map_err(viz_timeline_row_read)?,
        start_ns: row.get("start_ns").map_err(viz_timeline_row_read)?,
        end_ns: row.get("end_ns").map_err(viz_timeline_row_read)?,
        device_id: row.get("device_id").map_err(viz_timeline_row_read)?,
        stream_id: row.get("stream_id").map_err(viz_timeline_row_read)?,
        nvtx_depth: None,
    })
}

fn viz_timeline_row_read(source: duckdb::Error) -> NsysQueryError {
    NsysQueryError::sql_read("viz-timeline", "event-row", source)
}
