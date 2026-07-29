//! `veloq timeline <trace> --interval Nms` — time-bucketed GPU activity.
//!
//! Counts and total event-duration per fixed-width time bucket. Events
//! straddling bucket boundaries are **clipped**: a 5ms kernel running
//! 99.5ms..104.5ms contributes 0.5ms to bucket `[95ms, 100ms)` and
//! 4.5ms to bucket `[100ms, 105ms)`. `total_ns` is the **sum** of those
//! clipped per-event durations, not their union: when streams run
//! concurrently within a bucket it double-counts the overlap and can
//! exceed the bucket width. It answers "how much kernel/copy work was
//! issued in this window" (timeline plots, saturation trends), not "how
//! long the device was busy" — for true union busy/idle time use
//! `concurrency` (per-process/device union + overlap) or `gaps` (idle bubbles).
//!
//! Bucket alignment: aligned to multiples of `interval_ns` from the
//! window start (or the trace's primary origin if no `--time-range`).
//! Buckets with zero contribution are omitted from the response —
//! `total_matched` counts only the non-empty buckets.

use crate::query_sql::{
    event_scan::{EventScanFilterOptions, NvtxFilterPolicy, event_scan_filter},
    event_semantics::EventSemantics,
    exec, gpu_work,
};
use crate::{EventKind, KindFilter, NsysQueryError, NsysQueryResult};
use duckdb::types::Value;
use serde::Serialize;
use std::path::Path;
use veloq_core::{time::TimeWindow, timeline_bucket_key};
use veloq_nsys_data::Trace;
use veloq_query::duckdb::list as duckdb_list;
use veloq_query::sql::{SqlFragment, total_matched_bigint_expr, window};

#[derive(Debug, Clone)]
pub struct TimelineRequest {
    /// Bucket width in nanoseconds. Must be positive.
    pub interval_ns: i64,
    pub kinds: KindFilter,
    pub time_window: Option<TimeWindow>,
    /// Optional NVTX-attribution scope (glob against NVTX range name).
    pub nvtx: Option<String>,
    /// Restrict to one native process owning the CUDA namespace.
    pub process_id: Option<i64>,
    /// Restrict to one CUDA device (NSys `deviceId`).
    pub device: Option<i32>,
    /// Restrict to one CUDA stream (NSys `streamId`).
    pub stream: Option<i64>,
    /// Max buckets to return.
    pub limit: usize,
}

impl Default for TimelineRequest {
    fn default() -> Self {
        Self {
            interval_ns: 1_000_000, // 1ms default
            kinds: KindFilter::All,
            time_window: None,
            nvtx: None,
            process_id: None,
            device: None,
            stream: None,
            limit: 1000,
        }
    }
}

impl TimelineRequest {
    /// Parse the `--interval` CLI string (`1ms`, `100us`, `1.2s`, …) into ns.
    pub fn parse_interval(s: &str) -> NsysQueryResult<i64> {
        let ns = veloq_core::time::parse_duration_ns(s).map_err(|source| {
            NsysQueryError::TimelineIntervalInvalid {
                value: s.to_string(),
                source,
            }
        })?;
        if ns <= 0 {
            return Err(NsysQueryError::TimelineIntervalTooSmall { interval_ns: ns });
        }
        Ok(ns)
    }
}

pub struct TimelineKindPolicy {
    allowed: Vec<EventKind>,
}

impl TimelineKindPolicy {
    pub fn from_gpu_work_definition() -> NsysQueryResult<Self> {
        Ok(Self {
            allowed: gpu_work::GpuWorkSet::from_data_definition()?
                .kinds()
                .to_vec(),
        })
    }

    pub fn allowed(&self) -> &[EventKind] {
        &self.allowed
    }

    fn validate_explicit(&self, kinds: &KindFilter) -> NsysQueryResult<()> {
        if let KindFilter::Only(v) = kinds {
            for k in v {
                if !self.allowed.contains(k) {
                    return Err(NsysQueryError::TimelineKindNotAllowed { kind: k.as_str() });
                }
            }
        }
        Ok(())
    }

    fn resolve(
        &self,
        kinds: &KindFilter,
        nvtx: Option<&str>,
        trace: &Trace,
    ) -> NsysQueryResult<Vec<EventKind>> {
        self.validate_explicit(kinds)?;
        crate::kind_policy::resolve_nvtx_kinds(kinds, nvtx, self.allowed(), trace, "timeline")
    }
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct TimelineResponse {
    pub interval_ns: i64,
    /// Non-empty buckets returned (after `--limit`).
    pub count: usize,
    /// Non-empty buckets produced before `--limit`.
    pub total_matched: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_window_ns: Option<(i64, i64)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_scope: Option<String>,
    /// Canonical primary table. Each row is one non-empty time bucket.
    pub rows: Vec<Bucket>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct Bucket {
    /// Cross-trace key. `bucket|<start_ns>..<end_ns>`. Aligns
    /// two timelines bucket-for-bucket when interval and origin match;
    /// agents pre-normalize using envelope `trace_span.origin_ns` when
    /// they don't.
    pub key: String,
    pub start_ns: i64,
    pub end_ns: i64,
    /// Sum of clipped per-event durations in this bucket — not the
    /// interval union. Can exceed the bucket width when streams overlap
    /// (it double-counts concurrency); use `concurrency` for union time.
    pub total_ns: i64,
    pub kernel_ns: i64,
    pub memcpy_ns: i64,
    pub memset_ns: i64,
    /// CUDA-graph wall-time in this bucket — sum of `graph_trace` row
    /// durations (clipped to the bucket). When non-zero, the trace
    /// captured work into graphs and this share would otherwise be
    /// invisible to per-kernel aggregation.
    pub graph_ns: i64,
    pub count: i64,
    pub kernel_count: i64,
    pub memcpy_count: i64,
    pub memset_count: i64,
    pub graph_count: i64,
}

pub fn run<P: AsRef<Path>>(path: P, req: TimelineRequest) -> NsysQueryResult<TimelineResponse> {
    crate::check_limit(req.limit)?;
    if req.interval_ns <= 0 {
        return Err(NsysQueryError::TimelineIntervalTooSmall {
            interval_ns: req.interval_ns,
        });
    }

    let trace = Trace::open(path).map_err(NsysQueryError::trace_open)?;

    let abs_window = trace
        .resolve_window(req.time_window)
        .map_err(NsysQueryError::time_window_resolve)?;

    // Filter requested kinds against timeline's allow-list (GPU-busy
    // interval kinds only). The shared `--type` + `--nvtx` resolver in
    // `kind_policy` does the rest: explicit non-attributable kinds
    // bail when `--nvtx` is set, `KindFilter::All` narrows
    // implicitly to the attributable set, and missing tables drop
    // out silently.
    let kind_policy = TimelineKindPolicy::from_gpu_work_definition()?;
    let kinds = kind_policy.resolve(&req.kinds, req.nvtx.as_deref(), &trace)?;
    if kinds.is_empty() {
        return Ok(TimelineResponse {
            interval_ns: req.interval_ns,
            count: 0,
            total_matched: 0,
            time_window_ns: abs_window,
            nvtx_scope: req.nvtx.clone(),
            rows: Vec::new(),
        });
    }

    // Anchor: if --time-range given, use its start; else the trace's
    // primary origin. This makes bucket starts line up with relative
    // time offsets the user typed.
    let anchor = match abs_window {
        Some((s, _)) => s,
        None => {
            trace
                .read_origins()
                .map_err(NsysQueryError::data)?
                .0
                .primary
                .start_ns
        }
    };

    let attribution = match req.nvtx.as_deref() {
        Some(p) => Some(crate::nvtx_attribution::build(p, &kinds, &trace)?),
        None => None,
    };
    let nvtx_scope = if attribution.is_some() {
        crate::nvtx_attribution::NvtxScope::Attributed
    } else {
        crate::nvtx_attribution::NvtxScope::None
    };

    // Per-kind event SELECTs feeding the UNION ALL.
    let mut subqueries: Vec<String> = Vec::with_capacity(kinds.len());
    let mut per_kind_params: Vec<Value> = Vec::new();
    for kind in &kinds {
        let fragment = per_kind_select(
            &trace,
            *kind,
            abs_window,
            nvtx_scope,
            req.process_id,
            req.device,
            req.stream,
            kind_policy.allowed(),
        )?;
        subqueries.push(fragment.sql);
        per_kind_params.extend(fragment.params);
    }
    let union = subqueries.join(" UNION ALL ");

    let attribution_prefix = match &attribution {
        Some(att) => format!("{},", att.body),
        None => String::new(),
    };

    // The bucket-generator: range(bucket_start_low, bucket_end_high, interval_ns).
    // We compute the low/high directly from the events' extent so the
    // cross-join doesn't generate buckets covering empty timeline regions.
    let bucket_ns = req.interval_ns;
    let bucket_end_expr = format!("b.bucket_start + {bucket_ns}");
    let clipped_ns_expr = window::bucket_clipped_duration_expr(
        "e.start_ns",
        "e.end_ns",
        "b.bucket_start",
        &bucket_end_expr,
    );
    let sql = format!(
        r#"
        WITH {attribution_prefix} events AS ({union}),
        bounds AS (
            SELECT MIN(start_ns) AS lo, MAX(end_ns) AS hi
            FROM events
            WHERE end_ns > start_ns
        ),
        bucket_range AS (
            SELECT
                CAST(FLOOR(CAST(lo - {anchor} AS DOUBLE) / {bucket}) AS BIGINT) * {bucket} + {anchor} AS bs_start,
                (CAST(FLOOR(CAST(hi - {anchor} AS DOUBLE) / {bucket}) AS BIGINT) + 1) * {bucket} + {anchor} AS bs_end
            FROM bounds
        ),
        buckets AS (
            SELECT r.range AS bucket_start FROM bucket_range, range(bs_start, bs_end, {bucket}) AS r(range)
        ),
        clipped AS (
            SELECT
                b.bucket_start,
                b.bucket_start + {bucket} AS bucket_end,
                e.kind,
                GREATEST(
                    0,
                    {clipped_ns_expr}
                ) AS clipped_ns
            FROM buckets b
            JOIN events e
              ON e.start_ns < b.bucket_start + {bucket}
             AND e.end_ns   > b.bucket_start
        ),
        agg AS (
            SELECT
                bucket_start,
                bucket_end,
                CAST(SUM(clipped_ns) AS BIGINT) AS total_ns,
                CAST(SUM(CASE WHEN kind = 'kernel' THEN clipped_ns ELSE 0 END) AS BIGINT) AS kernel_ns,
                CAST(SUM(CASE WHEN kind = 'memcpy' THEN clipped_ns ELSE 0 END) AS BIGINT) AS memcpy_ns,
                CAST(SUM(CASE WHEN kind = 'memset' THEN clipped_ns ELSE 0 END) AS BIGINT) AS memset_ns,
                CAST(SUM(CASE WHEN kind = 'graph'  THEN clipped_ns ELSE 0 END) AS BIGINT) AS graph_ns,
                CAST(COUNT(*) AS BIGINT)                                              AS count,
                CAST(SUM(CASE WHEN kind = 'kernel' THEN 1 ELSE 0 END) AS BIGINT)      AS kernel_count,
                CAST(SUM(CASE WHEN kind = 'memcpy' THEN 1 ELSE 0 END) AS BIGINT)      AS memcpy_count,
                CAST(SUM(CASE WHEN kind = 'memset' THEN 1 ELSE 0 END) AS BIGINT)      AS memset_count,
                CAST(SUM(CASE WHEN kind = 'graph'  THEN 1 ELSE 0 END) AS BIGINT)      AS graph_count
            FROM clipped
            WHERE clipped_ns > 0
            GROUP BY bucket_start, bucket_end
        )
        SELECT *,
               {total_matched}
        FROM agg
        ORDER BY bucket_start
        LIMIT ?
        "#,
        bucket = bucket_ns,
        anchor = anchor,
        total_matched = total_matched_bigint_expr(),
    );

    // Bind order:
    //   1. attribution CTE param (the pattern glob), if --nvtx
    //   2. per-kind params (clipped bounds, filters) from per_kind_select
    //   3. LIMIT
    let mut params: Vec<Value> = Vec::new();
    if let Some(att) = &attribution {
        params.extend(att.params.iter().cloned());
    }
    params.extend(per_kind_params);
    params.push(Value::BigInt(req.limit as i64));

    let (buckets, total_matched) = hydrate_timeline_rows(&trace, &sql, &params)?;

    Ok(TimelineResponse {
        interval_ns: req.interval_ns,
        count: buckets.len(),
        total_matched,
        time_window_ns: abs_window,
        nvtx_scope: req.nvtx.clone(),
        rows: buckets,
    })
}

fn hydrate_timeline_rows(
    trace: &Trace,
    sql: &str,
    params: &[Value],
) -> NsysQueryResult<(Vec<Bucket>, i64)> {
    let rows = exec::query_rows(
        trace.conn(),
        sql,
        params,
        exec::TIMELINE_AGGREGATE,
        timeline_sql_row,
    )?;
    duckdb_list::split_rows_and_total::<i64, _, _, _>(
        rows,
        duckdb_list::TotalCarrier::First,
        |row| row.total_matched,
        duckdb_list::infallible_count_error,
        |row| Ok(row.bucket),
    )
}

struct TimelineSqlRow {
    bucket: Bucket,
    total_matched: i64,
}

fn timeline_sql_row(row: &duckdb::Row<'_>) -> Result<TimelineSqlRow, duckdb::Error> {
    let start_ns: i64 = row.get("bucket_start")?;
    let end_ns: i64 = row.get("bucket_end")?;
    Ok(TimelineSqlRow {
        bucket: Bucket {
            key: timeline_bucket_key(start_ns, end_ns),
            start_ns,
            end_ns,
            total_ns: row.get("total_ns")?,
            kernel_ns: row.get("kernel_ns")?,
            memcpy_ns: row.get("memcpy_ns")?,
            memset_ns: row.get("memset_ns")?,
            graph_ns: row.get("graph_ns")?,
            count: row.get("count")?,
            kernel_count: row.get("kernel_count")?,
            memcpy_count: row.get("memcpy_count")?,
            memset_count: row.get("memset_count")?,
            graph_count: row.get("graph_count")?,
        },
        total_matched: row.get("total_matched")?,
    })
}

/// Per-kind event SELECT contributing `(kind, start_ns, end_ns)` to the
/// `events` CTE. When a time window is present, the projected bounds
/// are clipped before bucket generation, so buckets and `total_ns`
/// reflect in-window work only.
fn per_kind_select(
    trace: &Trace,
    kind: EventKind,
    abs_window: Option<(i64, i64)>,
    nvtx_scope: crate::nvtx_attribution::NvtxScope,
    process_id: Option<i64>,
    device: Option<i32>,
    stream: Option<i64>,
    allowed_kinds: &[EventKind],
) -> NsysQueryResult<SqlFragment> {
    if matches!(kind, EventKind::Runtime | EventKind::Osrt | EventKind::Nvtx) {
        return Err(NsysQueryError::internal_unsupported_kind(
            "timeline",
            kind.as_str(),
        ));
    }
    let sem = EventSemantics::new(kind);

    let (start_expr, end_expr, mut params) = match abs_window {
        Some((s, e)) => (
            "GREATEST(t.start, ?)".to_string(),
            r#"LEAST(t."end", ?)"#.to_string(),
            vec![Value::BigInt(s), Value::BigInt(e)],
        ),
        None => ("t.start".to_string(), r#"t."end""#.to_string(), Vec::new()),
    };

    let mut filter = event_scan_filter(
        sem,
        EventScanFilterOptions {
            abs_window,
            device,
            stream,
            nvtx_scope,
            nvtx_policy: NvtxFilterPolicy::ErrorUnlessKindIn {
                verb: "timeline",
                allowed: allowed_kinds,
            },
        },
        &[],
    )?;
    let process =
        veloq_nsys_data::process_sql_projection(trace, sem.table(), "t", "event_proc", "t.start");
    if let Some(process_id) = process_id {
        filter.push_predicate(format!("{} = ?", process.expr));
        filter.push_param(Value::BigInt(process_id));
    }
    let where_clause = filter.where_clause();
    params.extend(filter.into_params());

    let sql = format!(
        "SELECT '{label}' AS kind, {start_expr} AS start_ns, {end_expr} AS end_ns \
         FROM nsight.{table} t {process_join} {where_clause}",
        label = sem.label(),
        table = sem.table(),
        process_join = process.join,
    );
    Ok(SqlFragment::new(sql, params))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use duckdb::Connection;
    use std::path::PathBuf;
    use tempfile::TempDir;
    use veloq_core::VeloqDiagnostic;

    fn parquet_fixture(tables: Vec<(&str, &str, Vec<&str>)>) -> Result<(TempDir, PathBuf)> {
        let dir = tempfile::tempdir()?;
        let pqtdir = dir.path().join("test_pqtdir");
        std::fs::create_dir_all(&pqtdir)?;
        let conn = Connection::open_in_memory()?;
        for (_, ddl, inserts) in &tables {
            conn.execute_batch(ddl)?;
            for insert in inserts {
                conn.execute_batch(insert)?;
            }
        }
        for (table, _, _) in &tables {
            let out = pqtdir.join(format!("{table}.parquet"));
            let out_lit = out.to_string_lossy().replace('\'', "''");
            conn.execute(
                &format!(r#"COPY (SELECT * FROM "{table}") TO '{out_lit}' (FORMAT PARQUET)"#),
                [],
            )?;
        }
        Ok((dir, pqtdir))
    }

    fn minimal_trace() -> Result<(TempDir, Trace)> {
        let (dir, pqtdir) = parquet_fixture(vec![(
            "CUPTI_ACTIVITY_KIND_KERNEL",
            r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start BIGINT, "end" BIGINT)"#,
            Vec::new(),
        )])?;
        let trace = Trace::open(&pqtdir)?;
        Ok((dir, trace))
    }

    fn timeline_hydration_sql(total_ns_expr: &str) -> String {
        format!(
            "SELECT \
             0::BIGINT AS bucket_start, \
             10::BIGINT AS bucket_end, \
             {total_ns_expr} AS total_ns, \
             1::BIGINT AS kernel_ns, \
             0::BIGINT AS memcpy_ns, \
             0::BIGINT AS memset_ns, \
             0::BIGINT AS graph_ns, \
             1::BIGINT AS count, \
             1::BIGINT AS kernel_count, \
             0::BIGINT AS memcpy_count, \
             0::BIGINT AS memset_count, \
             0::BIGINT AS graph_count, \
             1::BIGINT AS total_matched"
        )
    }

    #[test]
    fn windowed_per_kind_select_binds_clip_params_before_overlap_params() -> Result<()> {
        // The windowed projection clips bounds with GREATEST/LEAST in
        // the SELECT (binding window start,end) ahead of the overlap
        // WHERE clause (which binds end,start). Lock the exact bind
        // vector + ordering so a future refactor can't silently desync
        // the placeholders against their SQL positions.
        let allowed = [
            EventKind::Kernel,
            EventKind::Memcpy,
            EventKind::Memset,
            EventKind::Graph,
        ];
        let (_dir, trace) = minimal_trace()?;
        let fragment = per_kind_select(
            &trace,
            EventKind::Kernel,
            Some((10, 20)),
            crate::nvtx_attribution::NvtxScope::None,
            None,
            None,
            None,
            &allowed,
        )?;
        assert_eq!(
            fragment.params,
            vec![
                Value::BigInt(10), // GREATEST(t.start, ?) — clip start
                Value::BigInt(20), // LEAST(t."end", ?)    — clip end
                Value::BigInt(20), // WHERE t.start < ?     — overlap end
                Value::BigInt(10), // WHERE t."end" > ?     — overlap start
            ]
        );
        let clip_pos = fragment
            .sql
            .find("GREATEST(t.start, ?)")
            .ok_or_else(|| anyhow::anyhow!("clip expr missing from SQL: {}", fragment.sql))?;
        let where_pos = fragment
            .sql
            .find("WHERE")
            .ok_or_else(|| anyhow::anyhow!("WHERE missing from SQL: {}", fragment.sql))?;
        assert!(
            clip_pos < where_pos,
            "clip placeholders must precede WHERE placeholders: {}",
            fragment.sql
        );
        Ok(())
    }

    #[test]
    fn parse_interval_invalid_literal_returns_typed_error() -> anyhow::Result<()> {
        let err = match TimelineRequest::parse_interval("bogus") {
            Ok(ns) => anyhow::bail!("expected invalid interval to fail, got {ns} ns"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.timeline-interval-invalid");
        match err {
            crate::NsysQueryError::TimelineIntervalInvalid { value, .. } => {
                assert_eq!(value, "bogus");
            }
            other => anyhow::bail!("expected TimelineIntervalInvalid, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn hydrate_timeline_rows_prepare_error_is_typed() -> Result<()> {
        let (_dir, trace) = minimal_trace()?;

        let err = match hydrate_timeline_rows(&trace, "SELECT * FROM", &[]) {
            Ok((rows, _)) => anyhow::bail!(
                "malformed timeline SQL should not hydrate successfully: {} rows",
                rows.len()
            ),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-prepare");
        assert!(matches!(
            err,
            crate::NsysQueryError::Sql {
                phase: crate::SqlPhase::Prepare,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn hydrate_timeline_rows_query_error_is_typed() -> Result<()> {
        let (_dir, trace) = minimal_trace()?;

        let err = match hydrate_timeline_rows(&trace, "SELECT ? AS bucket_start", &[]) {
            Ok((rows, _)) => anyhow::bail!(
                "unbound timeline SQL parameter should not hydrate successfully: {} rows",
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
    fn hydrate_timeline_rows_read_error_is_typed() -> Result<()> {
        let (_dir, trace) = minimal_trace()?;
        let sql = timeline_hydration_sql("'not-total'");

        let err = match hydrate_timeline_rows(&trace, &sql, &[]) {
            Ok((rows, _)) => anyhow::bail!(
                "malformed timeline row should not hydrate successfully: {} rows",
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
