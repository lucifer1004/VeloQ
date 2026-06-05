//! `veloq gaps --min Nms` — GPU idle-bubble detection.
//!
//! Three scopes, picked via `--scope`:
//!
//! - **`device` (default)**: per device, gap = window where *no
//!   stream* was running a kernel / memcpy / memset. Multi-stream
//!   workloads see only the real device-wide idle bubbles —
//!   streams running concurrently don't produce phantom gaps on
//!   their idle peers.
//! - **`stream`**: per (device, stream), gap = window between
//!   consecutive events on that stream. Useful for "is this
//!   specific stream getting starved" diagnostics; not the right
//!   default because long-idle streams dominate output.
//! - **`trace`**: across all devices, gap = window where no device
//!   was running GPU work. Useful for distributed / multi-GPU jobs
//!   where the goal is "find moments where the whole rig was idle."
//!
//! Each gap carries the bracketing events (`prev` / `next`) so the
//! agent can `inspect` them or anchor follow-up correlate queries.
//! Under unified scopes (device / trace) the bracketing events may
//! live on different streams; the neighbor's `stream_id` is surfaced
//! so the agent can see which streams induced the boundary.
//!
//! ## Concurrency model
//!
//! `--scope device` and `--scope trace` use a sweep-line
//! (`MAX(end_ns) OVER ... ROWS UNBOUNDED PRECEDING ... 1 PRECEDING`)
//! that handles overlapping events correctly: a gap fires only when
//! the next event's `start_ns` exceeds the running max of all
//! previous events' `end_ns`. `--scope stream` keeps the existing
//! per-stream `LEAD()` formulation, where overlap on a single
//! stream (rare in real captures) produces a non-positive gap that
//! the threshold filter drops.

use anyhow::{Context, Result};
use duckdb::types::Value;
use serde::Serialize;
use std::path::Path;
use veloq_core::{
    Direction, SortKeyDef, SortKeySpec, SortSpec,
    time::{TimeWindow, parse_duration_ns},
};
use veloq_nsys_data::Trace;

use crate::{EventKind, RowId};

/// Aggregation scope for the gap computation. See the module doc
/// for the per-scope semantics; the request's other filters
/// (`device`, `stream`, `time_window`) compose on top.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GapScope {
    /// Per device, unified across streams. The default — answers
    /// "where was *this device's* GPU idle?" without phantom gaps
    /// on long-idle peer streams.
    #[default]
    Device,
    /// Per (device, stream); kept for the rare "is this specific
    /// stream getting starved" question.
    Stream,
    /// Trace-wide, unified across devices. Gap = window where no
    /// device ran GPU work. Useful for multi-GPU jobs that want to
    /// see whole-rig idle bubbles.
    Trace,
}

impl GapScope {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "device" => Ok(Self::Device),
            "stream" => Ok(Self::Stream),
            "trace" => Ok(Self::Trace),
            other => anyhow::bail!("invalid --scope `{other}`; expected device / stream / trace"),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Device => "device",
            Self::Stream => "stream",
            Self::Trace => "trace",
        }
    }
}

#[derive(Debug, Clone)]
pub struct GapsRequest {
    /// Minimum gap duration to report, in nanoseconds.
    pub min_ns: i64,
    /// Aggregation scope — see [`GapScope`]. Default `Device`.
    pub scope: GapScope,
    /// Optional `device_id` filter (NSys `deviceId`). Valid under
    /// every scope; rejected under `--scope trace` only when the
    /// filter would semantically conflict with the unified view.
    pub device: Option<i32>,
    /// Optional `stream_id` filter (NSys `streamId`). Only valid
    /// under `--scope stream` — under unified scopes there's no
    /// per-stream output to filter on. Use the row's `prev` / `next`
    /// neighbors and their `stream_id` to drill down post-hoc.
    pub stream: Option<i64>,
    /// Optional window — restricts to gaps whose start lies inside it.
    pub time_window: Option<TimeWindow>,
    /// Sort spec. `None` falls back to `duration` descending (biggest
    /// bubbles first — the default before sort was customisable).
    pub sort: Option<SortSpec>,
    /// Max rows to return.
    pub limit: usize,
}

impl Default for GapsRequest {
    fn default() -> Self {
        Self {
            min_ns: 1_000_000, // 1ms
            scope: GapScope::default(),
            device: None,
            stream: None,
            time_window: None,
            sort: None,
            limit: 100,
        }
    }
}

/// Sort axes `gaps` supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortKey {
    Duration,
    Start,
    Device,
    Stream,
}

impl SortKeyDef for SortKey {
    fn specs() -> &'static [SortKeySpec<Self>] {
        &[
            SortKeySpec {
                variant: SortKey::Duration,
                canonical: "duration",
                aliases: &["dur", "gap", "gap_ns"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SortKey::Start,
                canonical: "start",
                aliases: &["time", "start_ns"],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: SortKey::Device,
                canonical: "device",
                aliases: &["device_id", "dev"],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: SortKey::Stream,
                canonical: "stream",
                aliases: &["stream_id"],
                default_dir: Direction::Asc,
            },
        ]
    }
}

impl SortKey {
    fn column(self) -> &'static str {
        match self {
            Self::Duration => "gap_ns",
            Self::Start => "gap_start_ns",
            Self::Device => "device_id",
            Self::Stream => "stream_id",
        }
    }
}

fn gaps_sort_sql(spec: &SortSpec) -> anyhow::Result<String> {
    let mut resolved = Vec::new();
    for f in spec.fields() {
        let (k, d) = SortKey::from_field(f)?;
        resolved.push((k.column(), d));
    }
    // gap_start_ns as tiebreaker — guarantees deterministic order for
    // ties on the primary key.
    Ok(veloq_core::sort::build_order_by(&resolved, "gap_start_ns"))
}

impl GapsRequest {
    /// Parse the `--min-duration` CLI string ("1ms", "100us", "1.2s",
    /// "42ns") into ns.
    pub fn parse_min_duration(s: &str) -> Result<i64> {
        parse_duration_ns(s).with_context(|| format!("invalid --min-duration `{s}`"))
    }
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct GapsResponse {
    pub min_ns: i64,
    /// Echoed scope (`device` / `stream` / `trace`) so agents that
    /// see the response in isolation know which gap semantic
    /// produced it — the per-scope `key` format also differs.
    pub scope: &'static str,
    /// Gaps returned (after LIMIT).
    pub count: usize,
    /// Gaps matching `--min` + filters before LIMIT was applied.
    pub total_matched: i64,
    /// Resolved `--time-range`, if any (absolute ns).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_window_ns: Option<(i64, i64)>,
    /// Canonical primary table. Each row is one GPU idle bubble.
    pub rows: Vec<Gap>,
    /// Stream-level summary covering the same `(device, stream)`
    /// scope as `rows`. Surfaced so agents can pre-filter on
    /// `busy_ratio` (e.g. "ignore streams that ran <5% of the
    /// window") before iterating gaps — long-idle streams produce
    /// massive trace-wide gaps that dominate the response otherwise.
    pub auxiliary: GapsAuxiliary,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct GapsAuxiliary {
    /// One entry per `(device_id, stream_id)` that has at least one
    /// event in the gaps scope (after `--device` / `--stream` /
    /// `--time-range` filtering). Sorted by `(device_id, stream_id)`
    /// ascending so two traces with the same topology produce
    /// matching index positions for jq joins.
    pub streams: Vec<StreamActivity>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct StreamActivity {
    /// Cross-trace key. `stream|dev:<device_id>|stream:<stream_id>`.
    pub key: String,
    pub device_id: i32,
    pub stream_id: i64,
    /// Sum of in-scope event durations on this stream. Events
    /// straddling the `--time-range` boundary contribute only their
    /// in-window slice — matches how `stats` clips durations.
    pub busy_ns: i64,
    /// Wall-clock span covered by this scope: `--time-range`
    /// duration if set, otherwise the trace's primary origin span.
    pub span_ns: i64,
    /// `busy_ns / span_ns`. 1.0 = saturated; near 0 = effectively
    /// idle. `f64::NAN` when `span_ns == 0` (degenerate trace);
    /// agents should treat NaN as "unknown" rather than divide.
    pub busy_ratio: f64,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct Gap {
    /// Cross-trace key. Format depends on scope:
    /// - `device` → `gap|dev:<device_id>|@<start_ns>`
    /// - `stream` → `gap|dev:<device_id>|stream:<stream_id>|@<start_ns>`
    /// - `trace`  → `gap|@<start_ns>`
    ///
    /// Two runs of the same workload produce matching keys at
    /// matching axes; agents pre-normalize using envelope
    /// `trace_span.origin_ns`.
    pub key: String,
    /// `None` only under `--scope trace`, where the gap is a
    /// trace-wide bubble with no single device axis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_id: Option<i32>,
    /// `Some` only under `--scope stream`. Under `device` / `trace`
    /// the bracketing events may live on different streams; consult
    /// `prev.stream_id` / `next.stream_id` instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_id: Option<i64>,
    pub start_ns: i64,
    pub end_ns: i64,
    pub duration_ns: i64,
    pub prev: GapNeighbor,
    pub next: GapNeighbor,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct GapNeighbor {
    pub row_id: RowId,
    pub name: String,
    /// For `prev`: the event's `end_ns` (= gap start). For `next`: the
    /// event's `start_ns` (= gap end). Same value as the enclosing
    /// `Gap`'s endpoint — exposed here so the agent doesn't have to
    /// recompute.
    pub timestamp_ns: i64,
    /// Stream the bracketing event ran on. Under unified scopes
    /// (`device` / `trace`) this is the only place the stream
    /// shows up — the enclosing `Gap.stream_id` is `None` there.
    pub stream_id: i64,
}

pub fn run<P: AsRef<Path>>(path: P, req: GapsRequest) -> Result<GapsResponse> {
    crate::check_limit(req.limit)?;
    if req.min_ns <= 0 {
        anyhow::bail!("--min must be positive (got {} ns)", req.min_ns);
    }
    // --stream filter only makes sense under per-stream scope.
    // Under unified scopes (device / trace) we'd silently drop
    // gaps bracketed by events on other streams — confusing rather
    // than useful.
    if req.stream.is_some() && req.scope != GapScope::Stream {
        anyhow::bail!(
            "--stream <id> requires `--scope stream`; under \
             `--scope {scope}` events from every stream contribute to \
             the gap computation. Drop --stream or switch scope.",
            scope = req.scope.as_str()
        );
    }
    // --device under --scope trace: the row-level device_id is
    // projected NULL (a trace-scope gap spans every device), so a
    // `device_id = ?` filter would silently drop every row. Reject
    // upfront instead. Use `--scope device --device N` for per-device.
    if let (Some(dev), GapScope::Trace) = (req.device, req.scope) {
        anyhow::bail!(
            "--device {dev} is incompatible with `--scope trace` \
             (trace-scope gaps span every device); use \
             `--scope device --device {dev}` for per-device gaps."
        );
    }
    // Sort by `stream` requires per-stream scope. Sort by `device`
    // is meaningless under `trace` (single all-device partition).
    if let Some(s) = &req.sort {
        for f in s.fields() {
            let (k, _) = SortKey::from_field(f)?;
            match (k, req.scope) {
                (SortKey::Stream, scope) if scope != GapScope::Stream => anyhow::bail!(
                    "--sort stream requires `--scope stream`; under `--scope {}` rows have no stream axis",
                    scope.as_str()
                ),
                (SortKey::Device, GapScope::Trace) => anyhow::bail!(
                    "--sort device is meaningless under `--scope trace` (gaps are not partitioned by device)"
                ),
                _ => {}
            }
        }
    }

    let trace = Trace::open(path)?;
    let abs_window = trace.resolve_window(req.time_window)?;

    // Kernel / memcpy / memset are the GPU work that keeps the device
    // busy. Runtime API calls (cudaLaunchKernel, cudaMemcpyAsync) are
    // CPU-side and don't count; sync events are CPU blocking and
    // intentionally excluded from the "device in flight" definition.
    let mut subqueries: Vec<String> = Vec::new();
    for kind in [EventKind::Kernel, EventKind::Memcpy, EventKind::Memset] {
        if trace.table_exists(kind.table()) {
            subqueries.push(per_kind_select(kind)?);
        }
    }
    if subqueries.is_empty() {
        return Ok(GapsResponse {
            min_ns: req.min_ns,
            scope: req.scope.as_str(),
            count: 0,
            total_matched: 0,
            time_window_ns: abs_window,
            rows: Vec::new(),
            auxiliary: GapsAuxiliary {
                streams: Vec::new(),
            },
        });
    }
    let union = subqueries.join(" UNION ALL ");

    let (sql, params) = match req.scope {
        GapScope::Stream => build_stream_sql(&union, &req, abs_window)?,
        GapScope::Device => {
            build_unified_sql(&union, &req, abs_window, /*partition_device=*/ true)?
        }
        GapScope::Trace => {
            build_unified_sql(&union, &req, abs_window, /*partition_device=*/ false)?
        }
    };

    let conn = trace.conn();
    let mut stmt = conn.prepare(&sql).context("preparing gaps SQL")?;
    let params_ref = crate::bind(&params);
    let mut rows = stmt.query(params_ref.as_slice())?;

    let mut gaps: Vec<Gap> = Vec::with_capacity(req.limit);
    let mut total_matched: i64 = 0;
    while let Some(r) = rows.next()? {
        // Column layout is the same across all three SQL paths:
        //   0 device_id (i32, NULLABLE under scope=trace)
        //   1 stream_id (i64, NULLABLE under scope=device|trace)
        //   2 start_ns  3 end_ns  4 duration_ns
        //   5 prev_kind 6 prev_row 7 prev_name 8 prev_stream_id
        //   9 next_kind 10 next_row 11 next_name 12 next_stream_id
        //   13 total_matched
        let device_id: Option<i32> = r.get(0)?;
        let stream_id: Option<i64> = r.get(1)?;
        let start_ns: i64 = r.get(2)?;
        let end_ns: i64 = r.get(3)?;
        let duration_ns: i64 = r.get(4)?;
        let prev_kind: String = r.get(5)?;
        let prev_row_num: i64 = r.get(6)?;
        let prev_name: String = r.get(7)?;
        let prev_stream_id: i64 = r.get(8)?;
        let next_kind: String = r.get(9)?;
        let next_row_num: i64 = r.get(10)?;
        let next_name: String = r.get(11)?;
        let next_stream_id: i64 = r.get(12)?;
        total_matched = r.get(13)?;

        let key = match (req.scope, device_id, stream_id) {
            (GapScope::Stream, Some(d), Some(s)) => format!("gap|dev:{d}|stream:{s}|@{start_ns}"),
            (GapScope::Device, Some(d), _) => format!("gap|dev:{d}|@{start_ns}"),
            (GapScope::Trace, _, _) => format!("gap|@{start_ns}"),
            // Shouldn't happen under correct SQL — scope-stream
            // implies both device + stream are populated, scope-device
            // implies device is populated. Fall back rather than bail
            // so a SQL quirk doesn't kill the whole response.
            _ => format!("gap|@{start_ns}"),
        };

        gaps.push(Gap {
            key,
            device_id,
            stream_id,
            start_ns,
            end_ns,
            duration_ns,
            prev: GapNeighbor {
                row_id: RowId::new(parse_kind(&prev_kind)?, prev_row_num),
                name: prev_name,
                timestamp_ns: start_ns,
                stream_id: prev_stream_id,
            },
            next: GapNeighbor {
                row_id: RowId::new(parse_kind(&next_kind)?, next_row_num),
                name: next_name,
                timestamp_ns: end_ns,
                stream_id: next_stream_id,
            },
        });
    }

    let span_ns = match abs_window {
        Some((s, e)) => (e - s).max(0),
        None => {
            let (origins, _) = trace.read_origins()?;
            origins.primary.duration_ns().max(0)
        }
    };
    let streams = compute_stream_activity(&trace, &union, &req, abs_window, span_ns)?;

    Ok(GapsResponse {
        min_ns: req.min_ns,
        scope: req.scope.as_str(),
        count: gaps.len(),
        total_matched,
        time_window_ns: abs_window,
        rows: gaps,
        auxiliary: GapsAuxiliary { streams },
    })
}

/// Per-(device, stream) busy time. Same `(device, stream, time-window)`
/// filters as the gap query — keeps the auxiliary scope consistent
/// with `rows` so a `--stream 7` request returns one stream's
/// activity, not all of them.
fn compute_stream_activity(
    trace: &Trace,
    union: &str,
    req: &GapsRequest,
    abs_window: Option<(i64, i64)>,
    span_ns: i64,
) -> Result<Vec<StreamActivity>> {
    // Window-clip the durations the same way `stats` does so events
    // straddling the boundary don't over-count: an event 0..200ms
    // inside a 50..150ms window contributes 100ms, not 200ms. The
    // outer WHERE still uses the overlap predicate so we keep events
    // whose body touches the window.
    let (duration_expr, mut where_parts, mut params): (String, Vec<String>, Vec<Value>) =
        match abs_window {
            Some((s, e)) => {
                let p = vec![
                    Value::BigInt(e),
                    Value::BigInt(s),
                    Value::BigInt(e),
                    Value::BigInt(s),
                ];
                (
                    "LEAST(end_ns, ?) - GREATEST(start_ns, ?)".to_string(),
                    vec!["start_ns < ? AND end_ns > ?".to_string()],
                    p,
                )
            }
            None => ("end_ns - start_ns".to_string(), Vec::new(), Vec::new()),
        };

    crate::kind_policy::LocationFilter {
        device: req.device,
        stream: req.stream,
    }
    .push_where(&mut where_parts, &mut params);
    let where_clause = if where_parts.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", where_parts.join(" AND "))
    };

    let sql = format!(
        r#"
        WITH events AS ({union})
        SELECT
            device_id,
            stream_id,
            CAST(SUM({duration_expr}) AS BIGINT) AS busy_ns
        FROM events
        {where_clause}
        GROUP BY device_id, stream_id
        ORDER BY device_id, stream_id
        "#
    );

    let conn = trace.conn();
    let mut stmt = conn
        .prepare(&sql)
        .context("preparing stream-activity SQL")?;
    let params_ref = crate::bind(&params);
    let mut rows = stmt.query(params_ref.as_slice())?;

    let mut out = Vec::new();
    while let Some(r) = rows.next()? {
        let device_id: i32 = r.get(0)?;
        let stream_id: i64 = r.get(1)?;
        let busy_ns: i64 = r.get::<_, Option<i64>>(2)?.unwrap_or(0);
        let busy_ratio = if span_ns > 0 {
            (busy_ns as f64) / (span_ns as f64)
        } else {
            f64::NAN
        };
        out.push(StreamActivity {
            key: format!("stream|dev:{device_id}|stream:{stream_id}"),
            device_id,
            stream_id,
            busy_ns,
            span_ns,
            busy_ratio,
        });
    }
    Ok(out)
}

fn parse_kind(s: &str) -> Result<EventKind> {
    EventKind::parse(s).with_context(|| format!("unrecognised kind tag `{s}` from SQL"))
}

/// SQL fragment emitting (kind, row_id, device_id, stream_id, name,
/// start_ns, end_ns) for one event table. Name resolution + memcpy
/// copyKind labels come from `kind_sql` so the three GPU branches stay
/// in sync with the other commands.
///
/// Build the per-stream `--scope stream` SQL: `LEAD()` between
/// consecutive events on the same (device, stream). Every emitted
/// column matches the 14-column layout the row reader expects.
fn build_stream_sql(
    union: &str,
    req: &GapsRequest,
    abs_window: Option<(i64, i64)>,
) -> Result<(String, Vec<Value>)> {
    let mut where_parts: Vec<String> = Vec::new();
    let mut params: Vec<Value> = Vec::new();

    crate::kind_policy::LocationFilter {
        device: req.device,
        stream: req.stream,
    }
    .push_where(&mut where_parts, &mut params);
    if let Some((s, e)) = abs_window {
        where_parts.push("gap_start_ns < ? AND gap_end_ns > ?".to_string());
        params.push(Value::BigInt(e));
        params.push(Value::BigInt(s));
    }
    where_parts.push("gap_ns >= ?".to_string());
    params.push(Value::BigInt(req.min_ns));
    let where_clause = format!("WHERE {}", where_parts.join(" AND "));
    params.push(Value::BigInt(req.limit as i64));

    let sort_spec = req
        .sort
        .clone()
        .unwrap_or_else(|| SortSpec::single("duration"));
    let order_by = gaps_sort_sql(&sort_spec)?;

    let sql = format!(
        r#"
        WITH events AS ({union}),
        sequenced AS (
            SELECT
                kind, row_id, device_id, stream_id, name, start_ns, end_ns,
                LEAD(start_ns) OVER w  AS next_start_ns,
                LEAD(row_id)   OVER w  AS next_row_id,
                LEAD(kind)     OVER w  AS next_kind,
                LEAD(name)     OVER w  AS next_name,
                LEAD(stream_id) OVER w AS next_stream_id
            FROM events
            WINDOW w AS (PARTITION BY device_id, stream_id ORDER BY start_ns, row_id)
        ),
        filtered AS (
            SELECT
                device_id, stream_id,
                end_ns AS gap_start_ns,
                next_start_ns AS gap_end_ns,
                next_start_ns - end_ns AS gap_ns,
                kind AS prev_kind, row_id AS prev_row_id, name AS prev_name,
                stream_id AS prev_stream_id,
                next_kind, next_row_id, next_name, next_stream_id
            FROM sequenced
            WHERE next_start_ns IS NOT NULL
        ),
        clipped AS (
            SELECT * FROM filtered {where_clause}
        )
        SELECT
            device_id, stream_id, gap_start_ns, gap_end_ns, gap_ns,
            prev_kind, prev_row_id, prev_name, prev_stream_id,
            next_kind, next_row_id, next_name, next_stream_id,
            CAST(COUNT(*) OVER () AS BIGINT) AS total_matched
        FROM clipped
        ORDER BY {order_by}
        LIMIT ?
        "#
    );
    Ok((sql, params))
}

/// Build the `--scope device` (`partition_device=true`) or `--scope
/// trace` (`partition_device=false`) SQL: sweep-line over the
/// unioned event interval set, using `MAX(end_ns) OVER … ROWS
/// UNBOUNDED PRECEDING … 1 PRECEDING` to find when the next event
/// starts after the running maximum of all previous ends. Concurrent
/// events on different streams overlap correctly — the gap fires
/// only when the device (or trace) is actually idle.
///
/// The `arg_max(value, ord) OVER ...` window aggregates pick the
/// previous event's row_id / kind / name / stream_id at the same
/// instant `MAX(end_ns)` does, so the bracketing-event context is
/// taken from the event that actually closed out the running max.
fn build_unified_sql(
    union: &str,
    req: &GapsRequest,
    abs_window: Option<(i64, i64)>,
    partition_device: bool,
) -> Result<(String, Vec<Value>)> {
    let partition = if partition_device {
        "PARTITION BY device_id ORDER BY start_ns, row_id"
    } else {
        "ORDER BY start_ns, row_id"
    };

    let mut where_parts: Vec<String> = Vec::new();
    let mut params: Vec<Value> = Vec::new();

    // `--stream` is rejected upstream under unified scopes; device only.
    crate::kind_policy::LocationFilter {
        device: req.device,
        stream: None,
    }
    .push_where(&mut where_parts, &mut params);
    if let Some((s, e)) = abs_window {
        where_parts.push("gap_start_ns < ? AND gap_end_ns > ?".to_string());
        params.push(Value::BigInt(e));
        params.push(Value::BigInt(s));
    }
    where_parts.push("gap_ns >= ?".to_string());
    params.push(Value::BigInt(req.min_ns));
    let where_clause = format!("WHERE {}", where_parts.join(" AND "));
    params.push(Value::BigInt(req.limit as i64));

    let sort_spec = req
        .sort
        .clone()
        .unwrap_or_else(|| SortSpec::single("duration"));
    let order_by = gaps_sort_sql(&sort_spec)?;

    // `stream_id` projection: under `--scope stream` it'd be the
    // partition key. Under unified scopes it's irrelevant at the
    // gap row level (the bracketing events on prev/next carry the
    // stream context). Project NULL so the row reader's i64 column
    // resolves to None.
    let stream_id_proj = "CAST(NULL AS BIGINT)";
    // Under `--scope trace` we also have no single device axis.
    let device_id_proj = if partition_device {
        "device_id"
    } else {
        "CAST(NULL AS INTEGER)"
    };

    let sql = format!(
        r#"
        WITH events AS ({union}),
        with_prev AS (
            SELECT
                device_id, stream_id, kind, row_id, name, start_ns, end_ns,
                MAX(end_ns)          OVER win AS prev_max_end,
                arg_max(row_id,    end_ns) OVER win AS prev_row_id,
                arg_max(kind,      end_ns) OVER win AS prev_kind,
                arg_max(name,      end_ns) OVER win AS prev_name,
                arg_max(stream_id, end_ns) OVER win AS prev_stream_id
            FROM events
            WINDOW win AS ({partition} ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        ),
        filtered AS (
            SELECT
                {device_id_proj} AS device_id,
                {stream_id_proj} AS stream_id,
                prev_max_end AS gap_start_ns,
                start_ns     AS gap_end_ns,
                start_ns - prev_max_end AS gap_ns,
                prev_kind, prev_row_id, prev_name, prev_stream_id,
                kind      AS next_kind,
                row_id    AS next_row_id,
                name      AS next_name,
                stream_id AS next_stream_id
            FROM with_prev
            WHERE prev_max_end IS NOT NULL AND start_ns > prev_max_end
        ),
        clipped AS (
            SELECT * FROM filtered {where_clause}
        )
        SELECT
            device_id, stream_id, gap_start_ns, gap_end_ns, gap_ns,
            prev_kind, prev_row_id, prev_name, prev_stream_id,
            next_kind, next_row_id, next_name, next_stream_id,
            CAST(COUNT(*) OVER () AS BIGINT) AS total_matched
        FROM clipped
        ORDER BY {order_by}
        LIMIT ?
        "#
    );
    Ok((sql, params))
}

/// Returns an error when called with a non-GPU kind. Upstream filters
/// to kernel/memcpy/memset, but the workspace's no-panic policy
/// routes the precondition through `Result` instead of `unreachable!`.
fn per_kind_select(kind: EventKind) -> Result<String> {
    if matches!(kind, EventKind::Runtime | EventKind::Osrt | EventKind::Nvtx) {
        anyhow::bail!(
            "internal: gaps only inspects GPU stream events; got `{}`",
            kind.as_str()
        );
    }
    let table = kind.table();
    let label = kind.as_str();
    let name_expr = crate::kind_sql::display_name_expr(kind);
    let joins = crate::kind_sql::name_joins(kind);
    let dev = crate::kind_sql::GPU_DEVICE_ID_EXPR;
    let stm = crate::kind_sql::GPU_STREAM_ID_EXPR;
    Ok(format!(
        r#"
        SELECT
            '{label}' AS kind,
            t.rowid   AS row_id,
            {dev} AS device_id,
            {stm} AS stream_id,
            {name_expr} AS name,
            t.start   AS start_ns,
            t."end"   AS end_ns
        FROM nsight.{table} t {joins}
        "#
    ))
}
