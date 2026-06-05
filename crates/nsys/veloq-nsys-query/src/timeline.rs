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
//! `concurrency` (per-device union + overlap) or `gaps` (idle bubbles).
//!
//! Bucket alignment: aligned to multiples of `interval_ns` from the
//! window start (or the trace's primary origin if no `--time-range`).
//! Buckets with zero contribution are omitted from the response —
//! `total_matched` counts only the non-empty buckets.

use crate::{EventKind, KindFilter};
use anyhow::{Context, Result};
use duckdb::types::Value;
use serde::Serialize;
use std::path::Path;
use veloq_core::time::TimeWindow;
use veloq_nsys_data::Trace;

/// Kinds `timeline` is willing to bucket. GPU-busy-time kinds only:
/// `Sync` is intentionally excluded because synchronisation events are
/// CPU-side waits, not GPU activity. `Graph` is included because, in
/// `--cuda-graph-trace=graph` captures, graph_trace rows are the only
/// per-execution record of work that ran on the GPU during a graph
/// launch.
pub const ALLOWED_KINDS: [EventKind; 4] = [
    EventKind::Kernel,
    EventKind::Memcpy,
    EventKind::Memset,
    EventKind::Graph,
];

#[derive(Debug, Clone)]
pub struct TimelineRequest {
    /// Bucket width in nanoseconds. Must be positive.
    pub interval_ns: i64,
    pub kinds: KindFilter,
    pub time_window: Option<TimeWindow>,
    /// Optional NVTX-attribution scope (glob against NVTX range name).
    pub nvtx: Option<String>,
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
            device: None,
            stream: None,
            limit: 1000,
        }
    }
}

impl TimelineRequest {
    /// Parse the `--interval` CLI string (`1ms`, `100us`, `1.2s`, …) into ns.
    pub fn parse_interval(s: &str) -> Result<i64> {
        crate::parse_positive_duration(s, "--interval")
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

pub fn run<P: AsRef<Path>>(path: P, req: TimelineRequest) -> Result<TimelineResponse> {
    crate::check_limit(req.limit)?;
    if req.interval_ns <= 0 {
        anyhow::bail!("--interval must be positive (got {} ns)", req.interval_ns);
    }

    let trace = Trace::open(path)?;

    let abs_window = trace.resolve_window(req.time_window)?;

    // Filter requested kinds against timeline's allow-list (GPU
    // only). The shared `--type` + `--nvtx` resolver in
    // `kind_policy` does the rest: explicit non-attributable kinds
    // bail when `--nvtx` is set, `KindFilter::All` narrows
    // implicitly to the attributable set, and missing tables drop
    // out silently.
    if let KindFilter::Only(v) = &req.kinds {
        for k in v {
            if !ALLOWED_KINDS.contains(k) {
                anyhow::bail!(
                    "timeline only buckets GPU kinds (kernel/memcpy/memset/graph); got `{}`",
                    k.as_str()
                );
            }
        }
    }
    let kinds = crate::kind_policy::resolve_nvtx_kinds(
        &req.kinds,
        req.nvtx.as_deref(),
        &ALLOWED_KINDS,
        &trace,
        "timeline",
    )?;
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
        None => trace.read_origins()?.0.primary.start_ns,
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
        let (sql, params) = per_kind_select(*kind, abs_window, nvtx_scope, req.device, req.stream)?;
        subqueries.push(sql);
        per_kind_params.extend(params);
    }
    let union = subqueries.join(" UNION ALL ");

    let attribution_prefix = match &attribution {
        Some(att) => format!("{},", att.body),
        None => String::new(),
    };

    // The bucket-generator: range(bucket_start_low, bucket_end_high, interval_ns).
    // We compute the low/high directly from the events' extent so the
    // cross-join doesn't generate buckets covering empty timeline regions.
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
                    LEAST(e.end_ns, b.bucket_start + {bucket}) - GREATEST(e.start_ns, b.bucket_start)
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
               CAST(COUNT(*) OVER () AS BIGINT) AS total_matched
        FROM agg
        ORDER BY bucket_start
        LIMIT ?
        "#,
        bucket = req.interval_ns,
        anchor = anchor,
    );

    // Bind order:
    //   1. attribution CTE param (the pattern glob), if --nvtx
    //   2. per-kind windowed params (carried by per_kind_select)
    //   3. LIMIT
    let mut params: Vec<Value> = Vec::new();
    if let Some(att) = &attribution {
        params.extend(att.params.iter().cloned());
    }
    params.extend(per_kind_params);
    params.push(Value::BigInt(req.limit as i64));

    let conn = trace.conn();
    let mut stmt = conn.prepare(&sql).context("preparing timeline SQL")?;
    let params_ref = crate::bind(&params);
    let mut rows = stmt.query(params_ref.as_slice())?;

    let mut buckets: Vec<Bucket> = Vec::new();
    let mut total_matched: i64 = 0;
    while let Some(r) = rows.next()? {
        let start_ns: i64 = r.get("bucket_start")?;
        let end_ns: i64 = r.get("bucket_end")?;
        buckets.push(Bucket {
            key: format!("bucket|{start_ns}..{end_ns}"),
            start_ns,
            end_ns,
            total_ns: r.get("total_ns")?,
            kernel_ns: r.get("kernel_ns")?,
            memcpy_ns: r.get("memcpy_ns")?,
            memset_ns: r.get("memset_ns")?,
            graph_ns: r.get("graph_ns")?,
            count: r.get("count")?,
            kernel_count: r.get("kernel_count")?,
            memcpy_count: r.get("memcpy_count")?,
            memset_count: r.get("memset_count")?,
            graph_count: r.get("graph_count")?,
        });
        total_matched = r.get("total_matched")?;
    }

    Ok(TimelineResponse {
        interval_ns: req.interval_ns,
        count: buckets.len(),
        total_matched,
        time_window_ns: abs_window,
        nvtx_scope: req.nvtx.clone(),
        rows: buckets,
    })
}

/// Per-kind event SELECT contributing `(kind, start_ns, end_ns)` to the
/// `events` CTE. Mirrors stats's `per_kind_subquery` shape (the same
/// windowed-clip pattern); time-window + device + stream WHERE
/// clauses bind their respective params in order.
fn per_kind_select(
    kind: EventKind,
    abs_window: Option<(i64, i64)>,
    nvtx_scope: crate::nvtx_attribution::NvtxScope,
    device: Option<i32>,
    stream: Option<i64>,
) -> Result<(String, Vec<Value>)> {
    if matches!(kind, EventKind::Runtime | EventKind::Osrt | EventKind::Nvtx) {
        anyhow::bail!(
            "internal: timeline only buckets GPU kinds; got `{}`",
            kind.as_str()
        );
    }
    let table = kind.table();
    let label = kind.as_str();

    let mut params: Vec<Value> = Vec::new();
    let mut where_parts: Vec<String> = Vec::new();

    if let Some((start, end)) = abs_window {
        // Drop events entirely outside the window. Per-event clipping
        // happens in the `clipped` CTE via GREATEST/LEAST.
        where_parts.push(r#"t.start < ? AND t."end" > ?"#.to_string());
        params.push(Value::BigInt(end));
        params.push(Value::BigInt(start));
    }
    if let Some(dev) = device {
        where_parts.push(format!("{} = ?", crate::kind_sql::GPU_DEVICE_ID_EXPR));
        params.push(Value::Int(dev));
    }
    if let Some(stm) = stream {
        where_parts.push(format!("{} = ?", crate::kind_sql::GPU_STREAM_ID_EXPR));
        params.push(Value::BigInt(stm));
    }
    if nvtx_scope.is_attributed() {
        // Mirrors `stats::per_kind_subquery`: kinds that NVTX attribution
        // doesn't cover (graph_trace rolls up captured work that may not
        // sit under any current NVTX scope) emit `WHERE FALSE` so their
        // UNION ALL slot produces zero rows, instead of bailing. Without
        // this, `--type all --nvtx` crashes on traces that contain
        // graph_trace data because `ALLOWED_KINDS` includes `Graph`.
        let view: Option<&'static str> = match kind {
            EventKind::Kernel => Some(crate::nvtx_attribution::KERNEL_VIEW),
            EventKind::Memcpy => Some(crate::nvtx_attribution::MEMCPY_VIEW),
            EventKind::Memset => Some(crate::nvtx_attribution::MEMSET_VIEW),
            EventKind::Graph => None,
            EventKind::Runtime
            | EventKind::Osrt
            | EventKind::Nvtx
            | EventKind::Sync
            | EventKind::GraphNode
            | EventKind::GraphEvent
            | EventKind::CudaEvent
            | EventKind::Overhead
            | EventKind::CpuSample => {
                anyhow::bail!("internal: NVTX attribution unsupported for `{}`", label)
            }
        };
        match view {
            Some(v) => where_parts.push(crate::nvtx_attribution::filter_clause(v, "t")),
            None => where_parts.push("FALSE".to_string()),
        }
    }
    let where_clause = if where_parts.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", where_parts.join(" AND "))
    };

    let sql = format!(
        "SELECT '{label}' AS kind, t.start AS start_ns, t.\"end\" AS end_ns \
         FROM nsight.{table} t {where_clause}"
    );
    Ok((sql, params))
}
