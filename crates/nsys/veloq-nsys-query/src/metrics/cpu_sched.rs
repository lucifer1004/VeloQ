//! `--type cpu-sched` source.
//!
//! Reads `SCHED_EVENTS` (a transition stream — sched-in/out events
//! paired into precise on-cpu durations) and rolls up per-key
//! summaries on three axes (`tid` / `cpu` / `state`) or bucketed
//! time series. Trust signals: `unresolved_state_share`,
//! `per_cpu_max_gap_ns` (the response's `coverage` block is filled
//! by `run_cpu_sched` itself).

use crate::{NsysQueryError, NsysQueryResult};
use duckdb::types::Value;
use serde::Serialize;
use veloq_core::{Direction, SortKeyDef, SortKeySpec, SortSpec};
use veloq_nsys_data::Trace;
use veloq_query::duckdb::list as duckdb_list;
use veloq_query::sql::{total_matched_bigint_expr, window};

use super::{Coverage, CpuBucketSample, CpuSchedBody, CpuSchedRequest, MetricsCommon};

const CPU_SCHED_STATS_SQL: &str = "cpu-sched stats";
const CPU_SCHED_BUCKETS_SQL: &str = "cpu-sched buckets";

/// `--group-by` axis for `--type cpu-sched`.
///
/// `Tid` (default) yields per-thread on-cpu / off-cpu / ctx-switch
/// breakdown. `Cpu` yields per-cpu utilization. `State` yields
/// per-target-state aggregation: where threads went when sched'd out
/// (Running / Waiting / Mutex / …) — only as good as the kernel's
/// state labelling, which is often sparse (mostly `Unknown` in some
/// captures).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedGroupBy {
    Tid,
    Cpu,
    State,
}

impl SchedGroupBy {
    pub fn parse(s: &str) -> NsysQueryResult<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "tid" | "thread" => Ok(Self::Tid),
            "cpu" | "core" => Ok(Self::Cpu),
            "state" => Ok(Self::State),
            other => Err(NsysQueryError::metrics_cpu_sched_unknown_group_by(other)),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Tid => "tid",
            Self::Cpu => "cpu",
            Self::State => "state",
        }
    }
}

/// One row in the cpu-sched summary view. Axis-specific fields are
/// populated per `--group-by`:
///
/// - `tid` axis: `global_tid` + decoded `pid` / `tid` + `observed_span_ns`
/// - `cpu` axis: `cpu`, `distinct_tids`, and `off_cpu_ns` reads as
///   "idle ns on this cpu within its observation window"
/// - `state` axis: `state_id` + `state_name`, and `off_cpu_ns` reads
///   as "total time threads spent in this state before being
///   sched'd back in"
///
/// The `key` column is always present and stringifies the row's
/// identity (`"tid:12345"` / `"cpu:7"` / `"state:Mutex"`) so an agent
/// can sort and dedupe deterministically without a per-axis schema
/// swap.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SchedSummaryRow {
    pub key: String,
    /// Total run-time accumulated under this key. For `tid` / `cpu`
    /// this is the sum of paired sched-in → sched-out deltas. For
    /// `state` this is the sum of run-quanta that exited *to* this
    /// state — "how much CPU did threads burn before yielding to X?".
    pub on_cpu_ns: i64,
    /// Axis-specific off-cpu accounting (see struct doc). `None`
    /// when the row had no usable interval (single event, etc.).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub off_cpu_ns: Option<i64>,
    /// Number of context-switch events (sched-in for tid/cpu axes;
    /// sched-out for state axis) attributed to this key.
    pub ctx_switches: i64,
    /// `on_cpu_ns / ctx_switches`, rounded down. `None` when the row
    /// has zero paired quanta even though some events exist.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_quantum_ns: Option<i64>,
    /// `(max_event_ns - min_event_ns)` over events bound to this
    /// key — the local observation window for this row. Useful as a
    /// denominator for ad-hoc utilization math.
    pub observed_span_ns: i64,

    /// Present on `--group-by tid` axis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub global_tid: Option<i64>,
    /// Decoded process id — `(global_tid >> 24) & 0xFFFFFF`.
    /// See [`crate::decode_global_tid`] for the full bit layout.
    /// Present on `--group-by tid` axis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pid: Option<i64>,
    /// Decoded thread id — `global_tid & 0xFFFF`. Present on
    /// `--group-by tid` axis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tid: Option<i64>,
    /// Present on `--group-by cpu` axis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu: Option<i64>,
    /// Number of distinct globalTid values seen on this cpu. Present
    /// on `--group-by cpu` axis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distinct_tids: Option<i64>,
    /// Raw `threadState` enum id (FK to
    /// `ENUM_SAMPLING_THREAD_STATE.id`). Present on `--group-by state`
    /// axis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_id: Option<i64>,
    /// Human label from `ENUM_SAMPLING_THREAD_STATE.name`
    /// (e.g. `"Running"` / `"Interruptible"` / `"Unknown"`). Present
    /// on `--group-by state` axis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_name: Option<String>,
}

/// Fields shared by every `SchedSummaryRow` axis: the timing/count
/// numbers each per-axis SQL produces. Lets the per-axis constructors
/// take one struct's worth of "common" data plus their axis-specific
/// columns — adding a new common field touches one constructor list
/// instead of three call sites that each spelled out the same Nones.
pub(super) struct SchedCommon {
    pub on_cpu_ns: i64,
    pub off_cpu_ns: Option<i64>,
    pub ctx_switches: i64,
    pub avg_quantum_ns: Option<i64>,
    pub observed_span_ns: i64,
}

impl SchedSummaryRow {
    /// Build a `--group-by tid` axis row. `pid` / `tid` are decoded
    /// from `global_tid` via [`crate::decode_global_tid`] so callers
    /// don't have to remember the bit layout.
    pub(super) fn for_tid(global_tid: i64, common: SchedCommon) -> Self {
        let (pid, tid) = crate::decode_global_tid(global_tid);
        Self {
            key: format!("tid:{global_tid}"),
            on_cpu_ns: common.on_cpu_ns,
            off_cpu_ns: common.off_cpu_ns,
            ctx_switches: common.ctx_switches,
            avg_quantum_ns: common.avg_quantum_ns,
            observed_span_ns: common.observed_span_ns,
            global_tid: Some(global_tid),
            pid: Some(pid),
            tid: Some(tid),
            cpu: None,
            distinct_tids: None,
            state_id: None,
            state_name: None,
        }
    }

    /// Build a `--group-by cpu` axis row.
    pub(super) fn for_cpu(cpu: i64, distinct_tids: i64, common: SchedCommon) -> Self {
        Self {
            key: format!("cpu:{cpu}"),
            on_cpu_ns: common.on_cpu_ns,
            off_cpu_ns: common.off_cpu_ns,
            ctx_switches: common.ctx_switches,
            avg_quantum_ns: common.avg_quantum_ns,
            observed_span_ns: common.observed_span_ns,
            global_tid: None,
            pid: None,
            tid: None,
            cpu: Some(cpu),
            distinct_tids: Some(distinct_tids),
            state_id: None,
            state_name: None,
        }
    }

    /// Build a `--group-by state` axis row. `observed_span_ns` from
    /// `common` is ignored — observed-span isn't a per-state concept,
    /// so the row reports `0` rather than fabricating one. Agents
    /// that need a denominator should use the response's
    /// `metrics_span_ns`.
    pub(super) fn for_state(
        state_id: Option<i64>,
        state_name: String,
        common: SchedCommon,
    ) -> Self {
        Self {
            key: format!("state:{state_name}"),
            on_cpu_ns: common.on_cpu_ns,
            off_cpu_ns: common.off_cpu_ns,
            ctx_switches: common.ctx_switches,
            avg_quantum_ns: common.avg_quantum_ns,
            observed_span_ns: 0,
            global_tid: None,
            pid: None,
            tid: None,
            cpu: None,
            distinct_tids: None,
            state_id,
            state_name: Some(state_name),
        }
    }
}

/// `SCHED_EVENTS`-based per-key summary.
///
/// SCHED_EVENTS is a *transition stream*: each row is a sched-in
/// (`isSchedIn=1`) or sched-out (`isSchedIn=0`) event. Pairing
/// adjacent (`isSchedIn=1` → next event on the same `(globalTid, cpu)`)
/// gives a precise run-quantum; that's the basis for every per-axis
/// rollup below. Off-cpu accounting is axis-specific:
///
/// - `tid`: off_cpu_ns = observed_span - on_cpu_ns
/// - `cpu`: off_cpu_ns = idle time on this cpu within its window
/// - `state`: off_cpu_ns = sum of (sched-out with this state → next
///   sched-in for same tid). Reads as "time threads spent in
///   state X".
pub(super) fn run_cpu_sched(
    trace: &Trace,
    req: &CpuSchedRequest,
    abs_window: Option<(i64, i64)>,
    trace_origin_ns: i64,
    trace_span_ns: (i64, i64),
) -> NsysQueryResult<CpuSchedBody> {
    if !trace.table_exists("SCHED_EVENTS") {
        return Err(NsysQueryError::MetricsCpuSchedEventsMissing);
    }

    let group_by = match req.group_by.as_deref() {
        Some(s) => SchedGroupBy::parse(s)?,
        None => SchedGroupBy::Tid,
    };

    let stats = query_sched_stats(trace, req, abs_window)?;

    let rows = if req.common.bucket_ns.is_none() {
        match group_by {
            SchedGroupBy::Tid => query_sched_summary_tid(trace, req, abs_window)?,
            SchedGroupBy::Cpu => query_sched_summary_cpu(trace, req, abs_window)?,
            SchedGroupBy::State => query_sched_summary_state(trace, req, abs_window)?,
        }
    } else {
        Vec::new()
    };

    // --sort applies to summary mode only; bucket mode is always
    // time-ordered. Defaults: tid axis sorts by on_cpu_ns DESC
    // ("biggest hog first"), cpu axis by on_cpu_ns DESC, state axis
    // by ctx_switches DESC.
    let mut rows = rows;
    if req.common.bucket_ns.is_none() {
        let default_spec = SortSpec::single("on_cpu");
        let sort_spec = req.common.sort.as_ref().unwrap_or(&default_spec);
        sort_sched(&mut rows, sort_spec)?;
    }
    let rows_pre_limit = rows.len() as i64;
    if req.common.bucket_ns.is_none() {
        rows.truncate(req.common.limit);
    }

    // SQL applies `LIMIT ?` and projects `COUNT(*) OVER ()` so
    // `buckets_pre_limit` reflects rows the WHERE clause matched
    // before the limit truncated them. Computing the count from
    // `cpu_buckets.len()` after SQL would silently equal `count`
    // any time matched > limit, defeating the trust signal.
    let (cpu_buckets, buckets_pre_limit) = match req.common.bucket_ns {
        None => (Vec::new(), 0),
        Some(bucket_ns) => {
            query_sched_buckets(trace, req, abs_window, bucket_ns, group_by, trace_origin_ns)?
        }
    };

    let metrics_span_ns = stats.span;
    let coverage = Coverage::compute(metrics_span_ns, trace_span_ns, stats.events_total, None);
    let unresolved_state_share = if stats.events_total > 0 {
        Some((stats.n_unknown_state as f64 / stats.events_total as f64).clamp(0.0, 1.0))
    } else {
        None
    };

    let (count, total_matched) = match req.common.bucket_ns {
        None => (rows.len(), rows_pre_limit),
        Some(_) => (cpu_buckets.len(), buckets_pre_limit),
    };

    Ok(CpuSchedBody {
        count,
        total_matched,
        rows,
        auxiliary: super::CpuSchedAuxiliary {
            common: MetricsCommon {
                trace_origin_ns,
                trace_span_ns,
                metrics_span_ns,
                coverage,
                time_window_ns: abs_window,
                bucket_ns: req.common.bucket_ns,
            },
            group_by: group_by.as_str(),
            cpu_filter: req.cpu,
            tid_filter: req.tid,
            unresolved_state_share,
            per_cpu_max_gap_ns: stats.per_cpu_max_gap,
            cpu_buckets,
        },
    })
}

/// Trust-signal aggregates over the filtered `SCHED_EVENTS` row set —
/// total count, time span, unknown-state count, and the max gap
/// between consecutive events on any single cpu. One query so we hit
/// `SCHED_EVENTS` once.
struct SchedStats {
    events_total: i64,
    span: Option<(i64, i64)>,
    n_unknown_state: i64,
    /// `None` when fewer than two events were returned (max LAG over a
    /// 1-row set is undefined). Otherwise the maximum `start - LAG(start)`
    /// across all `(cpu)` partitions.
    per_cpu_max_gap: Option<i64>,
}

fn query_sched_stats(
    trace: &Trace,
    req: &CpuSchedRequest,
    abs_window: Option<(i64, i64)>,
) -> NsysQueryResult<SchedStats> {
    let (filtered_cte, params) = build_sched_filtered_cte(req, abs_window);
    let sql = format!(
        r#"
        WITH {filtered_cte},
        gaps AS (
            SELECT
                start - LAG(start) OVER (PARTITION BY cpu ORDER BY start) AS gap_ns
            FROM filtered_sched
        )
        SELECT
            CAST((SELECT COUNT(*) FROM filtered_sched) AS BIGINT) AS events_total,
            CAST((SELECT COALESCE(MIN(start), 0) FROM filtered_sched) AS BIGINT) AS span_lo,
            CAST((SELECT COALESCE(MAX(start), 0) FROM filtered_sched) AS BIGINT) AS span_hi,
            CAST((SELECT COUNT(*) FROM filtered_sched
                  WHERE COALESCE(threadState, 0) = 0) AS BIGINT) AS n_unknown,
            CAST((SELECT MAX(gap_ns) FROM gaps) AS BIGINT) AS max_gap
        "#
    );
    super::query_optional_row(trace, &sql, &params, CPU_SCHED_STATS_SQL, sched_stats_row)?
        .ok_or_else(|| crate::NsysQueryError::internal_stats_row_missing("cpu-sched"))
}

fn sched_stats_row(row: &duckdb::Row<'_>) -> Result<SchedStats, duckdb::Error> {
    let events_total: i64 = row.get("events_total")?;
    let span = if events_total > 0 {
        Some((row.get("span_lo")?, row.get("span_hi")?))
    } else {
        None
    };
    // MAX(gap_ns) returns NULL when the windowed LAG had no predecessor
    // on any partition (only one row per cpu, or zero rows). Read as
    // SQL NULL → `None` so the wire format doesn't carry a misleading
    // zero.
    let per_cpu_max_gap: Option<i64> = row.get("max_gap")?;
    Ok(SchedStats {
        events_total,
        span,
        n_unknown_state: row.get("n_unknown")?,
        per_cpu_max_gap,
    })
}

/// Shared filtered-events CTE — applied uniformly across summary /
/// bucket / trust-signal queries so they all aggregate over the same
/// row set. Returns the CTE body (without `WITH` prefix) plus the
/// parameter list to bind in order.
fn build_sched_filtered_cte(
    req: &CpuSchedRequest,
    abs_window: Option<(i64, i64)>,
) -> (String, Vec<Value>) {
    let global_tid = veloq_nsys_data::sql_expr::u64_bits_to_i64("globalTid");
    let fragment = crate::query_sql::sample_scan::filtered_cte(
        "filtered_sched",
        "SCHED_EVENTS",
        &format!(
            "start, \
             CAST(cpu AS BIGINT) AS cpu, \
             isSchedIn, \
             {global_tid} AS globalTid, \
             threadState"
        ),
        req.cpu,
        req.tid,
        abs_window,
    );
    (fragment.sql, fragment.params)
}

/// Build the paired-quanta CTE: each `(cpu, globalTid)` partition is
/// scanned in start order; consecutive sched-in → sched-out events
/// produce one quantum. `next_state` carries the sched-out's
/// `threadState` so the state-axis rollup can read where the thread
/// went without a second pass.
fn quanta_cte() -> &'static str {
    r#"quanta AS (
        SELECT
            cpu,
            globalTid,
            start AS quantum_start,
            next_start AS quantum_end,
            next_start - start AS on_cpu_ns,
            next_state AS exit_state
        FROM (
            SELECT
                start,
                cpu,
                isSchedIn,
                globalTid,
                threadState,
                LEAD(start) OVER w AS next_start,
                LEAD(isSchedIn) OVER w AS next_is_in,
                LEAD(threadState) OVER w AS next_state
            FROM filtered_sched
            WINDOW w AS (PARTITION BY globalTid, cpu ORDER BY start)
        )
        -- A valid quantum is sched-in followed by the next event on
        -- the same (tid, cpu) being sched-out. The same-cpu LEAD
        -- preserves the invariant: thread migrations break the
        -- partition, so a sched-out elsewhere wouldn't pair up here.
        WHERE isSchedIn = 1
          AND next_is_in = 0
          AND next_start IS NOT NULL
    )"#
}

fn query_sched_summary_tid(
    trace: &Trace,
    req: &CpuSchedRequest,
    abs_window: Option<(i64, i64)>,
) -> NsysQueryResult<Vec<SchedSummaryRow>> {
    let (filtered_cte, params) = build_sched_filtered_cte(req, abs_window);
    let quanta = quanta_cte();
    let sql = format!(
        r#"
        WITH {filtered_cte},
        {quanta},
        per_tid_on AS (
            SELECT
                globalTid,
                CAST(SUM(on_cpu_ns) AS BIGINT) AS on_cpu_ns,
                CAST(COUNT(*)        AS BIGINT) AS ctx_switches
            FROM quanta
            GROUP BY globalTid
        ),
        per_tid_span AS (
            SELECT
                globalTid,
                CAST(MIN(start) AS BIGINT) AS first_seen,
                CAST(MAX(start) AS BIGINT) AS last_seen
            FROM filtered_sched
            GROUP BY globalTid
        )
        SELECT
            COALESCE(o.globalTid, s.globalTid) AS globalTid,
            COALESCE(o.on_cpu_ns, 0)           AS on_cpu_ns,
            COALESCE(o.ctx_switches, 0)        AS ctx_switches,
            (s.last_seen - s.first_seen)       AS observed_span_ns
        FROM per_tid_span s
        FULL OUTER JOIN per_tid_on o ON o.globalTid = s.globalTid
        "#
    );
    super::query_rows(trace, &sql, &params, "cpu-sched per-tid summary", |r| {
        let global_tid: i64 = r.get("globalTid")?;
        let on_cpu_ns: i64 = r.get("on_cpu_ns")?;
        let ctx_switches: i64 = r.get("ctx_switches")?;
        let observed_span_ns: i64 = r.get("observed_span_ns")?;
        let off_cpu_ns = (observed_span_ns - on_cpu_ns).max(0);
        let avg_quantum_ns = if ctx_switches > 0 {
            Some(on_cpu_ns / ctx_switches)
        } else {
            None
        };
        Ok(SchedSummaryRow::for_tid(
            global_tid,
            SchedCommon {
                on_cpu_ns,
                off_cpu_ns: Some(off_cpu_ns),
                ctx_switches,
                avg_quantum_ns,
                observed_span_ns,
            },
        ))
    })
}

fn query_sched_summary_cpu(
    trace: &Trace,
    req: &CpuSchedRequest,
    abs_window: Option<(i64, i64)>,
) -> NsysQueryResult<Vec<SchedSummaryRow>> {
    let (filtered_cte, params) = build_sched_filtered_cte(req, abs_window);
    let quanta = quanta_cte();
    let sql = format!(
        r#"
        WITH {filtered_cte},
        {quanta},
        per_cpu_on AS (
            SELECT
                cpu,
                CAST(SUM(on_cpu_ns)               AS BIGINT) AS on_cpu_ns,
                CAST(COUNT(*)                     AS BIGINT) AS ctx_switches,
                CAST(COUNT(DISTINCT globalTid)    AS BIGINT) AS distinct_tids
            FROM quanta
            GROUP BY cpu
        ),
        per_cpu_span AS (
            SELECT
                cpu,
                CAST(MIN(start) AS BIGINT) AS first_seen,
                CAST(MAX(start) AS BIGINT) AS last_seen
            FROM filtered_sched
            GROUP BY cpu
        )
        SELECT
            COALESCE(o.cpu, s.cpu)             AS cpu,
            COALESCE(o.on_cpu_ns, 0)           AS on_cpu_ns,
            COALESCE(o.ctx_switches, 0)        AS ctx_switches,
            COALESCE(o.distinct_tids, 0)       AS distinct_tids,
            (s.last_seen - s.first_seen)       AS observed_span_ns
        FROM per_cpu_span s
        FULL OUTER JOIN per_cpu_on o ON o.cpu = s.cpu
        "#
    );
    super::query_rows(trace, &sql, &params, "cpu-sched per-cpu summary", |r| {
        let cpu: i64 = r.get("cpu")?;
        let on_cpu_ns: i64 = r.get("on_cpu_ns")?;
        let ctx_switches: i64 = r.get("ctx_switches")?;
        let distinct_tids: i64 = r.get("distinct_tids")?;
        let observed_span_ns: i64 = r.get("observed_span_ns")?;
        // For the cpu axis, off_cpu_ns is the idle-on-this-cpu time
        // within its observation window. Subtract sum of run-quanta.
        let off_cpu_ns = (observed_span_ns - on_cpu_ns).max(0);
        let avg_quantum_ns = if ctx_switches > 0 {
            Some(on_cpu_ns / ctx_switches)
        } else {
            None
        };
        Ok(SchedSummaryRow::for_cpu(
            cpu,
            distinct_tids,
            SchedCommon {
                on_cpu_ns,
                off_cpu_ns: Some(off_cpu_ns),
                ctx_switches,
                avg_quantum_ns,
                observed_span_ns,
            },
        ))
    })
}

fn query_sched_summary_state(
    trace: &Trace,
    req: &CpuSchedRequest,
    abs_window: Option<(i64, i64)>,
) -> NsysQueryResult<Vec<SchedSummaryRow>> {
    let (filtered_cte, params) = build_sched_filtered_cte(req, abs_window);
    let quanta = quanta_cte();
    let sql = format!(
        r#"
        WITH {filtered_cte},
        {quanta},
        sched_out_durations AS (
            -- For each sched-out, find the next sched-in for the
            -- SAME globalTid on ANY cpu (state-axis cares about the
            -- thread, not which cpu it ran on next). The state at
            -- the sched-out is the state the thread entered.
            --
            -- LEAD must look at the full per-tid event sequence —
            -- if we filter `isSchedIn = 0` first, the window
            -- function sees only sched-outs and `next_is_in` is
            -- always 0. So compute LEAD over the whole sequence and
            -- filter to sched-outs in the outer SELECT.
            SELECT exit_state, out_start, next_event, next_is_in
            FROM (
                SELECT
                    threadState AS exit_state,
                    start AS out_start,
                    isSchedIn,
                    LEAD(start) OVER w AS next_event,
                    LEAD(isSchedIn) OVER w AS next_is_in
                FROM filtered_sched
                WINDOW w AS (PARTITION BY globalTid ORDER BY start)
            )
            WHERE isSchedIn = 0
        ),
        per_state_off AS (
            SELECT
                exit_state,
                CAST(SUM(
                    CASE WHEN next_is_in = 1
                         THEN next_event - out_start
                         ELSE 0 END
                ) AS BIGINT) AS off_cpu_ns,
                CAST(COUNT(*) AS BIGINT) AS ctx_out_count
            FROM sched_out_durations
            GROUP BY exit_state
        ),
        per_state_on AS (
            SELECT
                exit_state,
                CAST(SUM(on_cpu_ns) AS BIGINT) AS on_cpu_ns,
                CAST(COUNT(*)       AS BIGINT) AS ctx_in_count
            FROM quanta
            GROUP BY exit_state
        )
        SELECT
            COALESCE(off.exit_state, ons.exit_state) AS state_id,
            COALESCE(en.name, 'unknown')             AS state_name,
            COALESCE(ons.on_cpu_ns, 0)               AS on_cpu_ns,
            COALESCE(off.off_cpu_ns, 0)              AS off_cpu_ns,
            COALESCE(off.ctx_out_count, 0)           AS ctx_switches,
            COALESCE(ons.ctx_in_count, 0)            AS pairing_count
        FROM per_state_off off
        FULL OUTER JOIN per_state_on ons
            ON ons.exit_state = off.exit_state
        LEFT JOIN nsight.ENUM_SAMPLING_THREAD_STATE en
            ON en.id = COALESCE(off.exit_state, ons.exit_state)
        "#
    );
    super::query_rows(trace, &sql, &params, "cpu-sched per-state summary", |r| {
        let state_id: Option<i64> = r.get("state_id")?;
        let state_name: String = r.get("state_name")?;
        let on_cpu_ns: i64 = r.get("on_cpu_ns")?;
        let off_cpu_ns: i64 = r.get("off_cpu_ns")?;
        let ctx_switches: i64 = r.get("ctx_switches")?;
        let pairing_count: i64 = r.get("pairing_count")?;
        let avg_quantum_ns = if pairing_count > 0 {
            Some(on_cpu_ns / pairing_count)
        } else {
            None
        };
        Ok(SchedSummaryRow::for_state(
            state_id,
            state_name,
            SchedCommon {
                on_cpu_ns,
                off_cpu_ns: Some(off_cpu_ns),
                ctx_switches,
                avg_quantum_ns,
                // for_state ignores observed_span_ns; pass 0 to be explicit.
                observed_span_ns: 0,
            },
        ))
    })
}

/// Bucket-mode time series: per `(t_start, group-by key)` on-cpu ns.
/// The bucket grid is anchored on `trace_origin_ns` so buckets line up
/// with what other commands already use. Quanta straddling a bucket
/// boundary are clipped on each side, so each bucket's `value`
/// measures actual in-bucket on-cpu time.
fn query_sched_buckets(
    trace: &Trace,
    req: &CpuSchedRequest,
    abs_window: Option<(i64, i64)>,
    bucket_ns: i64,
    group_by: SchedGroupBy,
    trace_origin_ns: i64,
) -> NsysQueryResult<(Vec<CpuBucketSample>, i64)> {
    let (filtered_cte, mut params) = build_sched_filtered_cte(req, abs_window);
    let quanta = quanta_cte();
    // Key expression per axis. The state axis joins
    // `ENUM_SAMPLING_THREAD_STATE` and uses the human label so the
    // bucket rows match the summary rows' `key`.
    let (key_expr, extra_join) = match group_by {
        SchedGroupBy::Tid => ("'tid:' || globalTid".to_string(), String::new()),
        SchedGroupBy::Cpu => ("'cpu:' || cpu".to_string(), String::new()),
        SchedGroupBy::State => (
            "'state:' || COALESCE(en.name, 'unknown')".to_string(),
            "LEFT JOIN nsight.ENUM_SAMPLING_THREAD_STATE en ON en.id = exit_state".to_string(),
        ),
    };
    params.push(Value::BigInt(req.common.limit as i64));
    let bucket_start_expr = format!("bucket_idx * {bucket_ns} + {trace_origin_ns}");
    let bucket_end_expr = format!("bucket_idx * {bucket_ns} + {trace_origin_ns} + {bucket_ns}");
    let clipped_ns_expr = window::bucket_clipped_duration_expr(
        "quantum_start",
        "quantum_end",
        &bucket_start_expr,
        &bucket_end_expr,
    );
    let sql = format!(
        r#"
        WITH {filtered_cte},
        {quanta},
        quanta_keyed AS (
            SELECT
                {key_expr} AS key,
                quantum_start,
                quantum_end
            FROM quanta
            {extra_join}
        ),
        spans AS (
            -- Bucket grid: emit every bucket touched by each quantum.
            -- DuckDB's range() is half-open like a Rust range; the
            -- math drops `quantum` into bucket coords, then we
            -- generate the integer-indexed buckets it overlaps.
            SELECT
                qk.key,
                CAST(b AS BIGINT) AS bucket_idx,
                qk.quantum_start,
                qk.quantum_end
            FROM quanta_keyed qk,
                 range(
                     CAST(FLOOR(CAST(qk.quantum_start - {anchor} AS DOUBLE) / {bucket}) AS BIGINT),
                     CAST(FLOOR(CAST(qk.quantum_end - 1 - {anchor} AS DOUBLE) / {bucket}) AS BIGINT) + 1
                 ) AS r(b)
        ),
        clipped AS (
            SELECT
                key,
                {bucket_start_expr} AS t_start,
                {bucket_end_expr} AS t_end,
                -- Clip the quantum to this bucket's bounds.
                {clipped_ns_expr} AS clipped_ns
            FROM spans
        ),
        agg AS (
            SELECT
                key,
                t_start,
                t_end,
                CAST(SUM(clipped_ns) AS DOUBLE) AS value,
                CAST(COUNT(*) AS BIGINT) AS samples
            FROM clipped
            WHERE clipped_ns > 0
            GROUP BY key, t_start, t_end
        )
        SELECT *,
               {total_matched}
        FROM agg
        ORDER BY t_start ASC, key ASC
        LIMIT ?
        "#,
        anchor = trace_origin_ns,
        bucket = bucket_ns,
        total_matched = total_matched_bigint_expr(),
    );
    let rows = super::query_rows(
        trace,
        &sql,
        &params,
        CPU_SCHED_BUCKETS_SQL,
        sched_bucket_row,
    )?;
    duckdb_list::split_rows_and_total::<i64, _, _, _>(
        rows,
        duckdb_list::TotalCarrier::Last,
        |row| row.total_matched,
        duckdb_list::infallible_count_error,
        |row| Ok(row.bucket),
    )
}

struct SchedBucketRow {
    bucket: CpuBucketSample,
    total_matched: i64,
}

fn sched_bucket_row(row: &duckdb::Row<'_>) -> Result<SchedBucketRow, duckdb::Error> {
    Ok(SchedBucketRow {
        bucket: CpuBucketSample {
            t_start_ns: row.get("t_start")?,
            t_end_ns: row.get("t_end")?,
            key: row.get("key")?,
            agg: "sum",
            value: row.get("value")?,
            samples: row.get("samples")?,
        },
        total_matched: row.get("total_matched")?,
    })
}

/// Sort axes the cpu-sched summary list supports. The default is
/// `on_cpu` DESC ("biggest hog first" for tid/cpu; "most accumulated
/// run-time" for state).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedSortKey {
    OnCpu,
    OffCpu,
    CtxSwitches,
    AvgQuantum,
    ObservedSpan,
    Pid,
    Tid,
    Cpu,
    Key,
}

impl SortKeyDef for SchedSortKey {
    fn specs() -> &'static [SortKeySpec<Self>] {
        &[
            SortKeySpec {
                variant: SchedSortKey::OnCpu,
                canonical: "on_cpu",
                aliases: &["on_cpu_ns", "samples"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SchedSortKey::OffCpu,
                canonical: "off_cpu",
                aliases: &["off_cpu_ns"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SchedSortKey::CtxSwitches,
                canonical: "ctx_switches",
                aliases: &["switches", "ctx"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SchedSortKey::AvgQuantum,
                canonical: "avg_quantum",
                aliases: &["avg_quantum_ns", "quantum"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SchedSortKey::ObservedSpan,
                canonical: "observed_span",
                aliases: &["observed_span_ns", "span"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SchedSortKey::Pid,
                canonical: "pid",
                aliases: &[],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: SchedSortKey::Tid,
                canonical: "tid",
                aliases: &[],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: SchedSortKey::Cpu,
                canonical: "cpu",
                aliases: &[],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: SchedSortKey::Key,
                canonical: "key",
                aliases: &["name"],
                default_dir: Direction::Asc,
            },
        ]
    }
}

fn sort_sched(out: &mut [SchedSummaryRow], spec: &SortSpec) -> NsysQueryResult<()> {
    let resolved: Vec<(SchedSortKey, Direction)> = spec
        .fields()
        .iter()
        .map(|f| SchedSortKey::from_field(f).map_err(NsysQueryError::metrics_sort_invalid))
        .collect::<NsysQueryResult<_>>()?;
    // Stable tiebreaker on `key` ASC.
    veloq_core::sort_in_memory(
        out,
        &resolved,
        |k, a, b| match k {
            SchedSortKey::OnCpu => a.on_cpu_ns.cmp(&b.on_cpu_ns),
            SchedSortKey::OffCpu => a.off_cpu_ns.unwrap_or(0).cmp(&b.off_cpu_ns.unwrap_or(0)),
            SchedSortKey::CtxSwitches => a.ctx_switches.cmp(&b.ctx_switches),
            SchedSortKey::AvgQuantum => a
                .avg_quantum_ns
                .unwrap_or(0)
                .cmp(&b.avg_quantum_ns.unwrap_or(0)),
            SchedSortKey::ObservedSpan => a.observed_span_ns.cmp(&b.observed_span_ns),
            SchedSortKey::Pid => a.pid.unwrap_or(0).cmp(&b.pid.unwrap_or(0)),
            SchedSortKey::Tid => a.tid.unwrap_or(0).cmp(&b.tid.unwrap_or(0)),
            SchedSortKey::Cpu => a.cpu.unwrap_or(0).cmp(&b.cpu.unwrap_or(0)),
            SchedSortKey::Key => a.key.cmp(&b.key),
        },
        |a, b| a.key.cmp(&b.key),
    );
    Ok(())
}
