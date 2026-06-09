//! `veloq concurrency <trace>` — GPU kernel/transfer overlap extraction.
//!
//! Overlap is the union-versus-sum of GPU event
//! intervals: per device (and per stream) we report `sum_busy_ns`, the
//! interval-`union_busy_ns`, their difference `overlap_ns`, and the peak
//! `max_concurrency`, plus a compute-versus-copy block. The union and
//! the concurrency degree are the parts an agent cannot reconstruct in
//! jq from per-event rows — exposing them is the point of the verb.
//!
//! The interval sweep ([`measures`]) generalizes
//! [`crate::graph_replays`]'s `busy_ns`: it makes no same-stream-serial
//! assumption, so same-stream Programmatic Dependent Launch (PDL)
//! overlap is counted, not dropped. Compute/copy overlap falls out of
//! inclusion-exclusion: `compute_union + copy_union − device_union`.

use duckdb::types::Value;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;
use veloq_core::time::TimeWindow;
use veloq_nsys_data::Trace;

use crate::{
    NsysQueryError, NsysQueryResult,
    query_sql::gpu_work::{GpuWorkClass, GpuWorkSet},
};

#[derive(Debug, Clone)]
pub struct ConcurrencyRequest {
    /// Optional `device_id` filter (NSys `deviceId`). Overlap is always
    /// per device; this just restricts which device rows are emitted.
    pub device: Option<i32>,
    /// Optional window — measures overlap over each event's clipped
    /// portion (overlap-inclusion, like the other windowed verbs).
    pub time_window: Option<TimeWindow>,
    /// Max device rows to return.
    pub limit: usize,
}

impl Default for ConcurrencyRequest {
    fn default() -> Self {
        Self {
            device: None,
            time_window: None,
            limit: 100,
        }
    }
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct ConcurrencyResponse {
    /// Device rows returned (after LIMIT).
    pub count: usize,
    /// Devices with at least one in-scope event, before LIMIT.
    pub total_matched: i64,
    /// Resolved `--from`/`--to` window, if any (absolute ns).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_window_ns: Option<(i64, i64)>,
    /// Canonical primary list — one row per device, ascending `device_id`.
    pub rows: Vec<DeviceConcurrency>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct DeviceConcurrency {
    /// Cross-trace key. `concurrency|dev:<device_id>`.
    pub key: String,
    pub device_id: i32,
    /// Sum of in-scope event durations on this device.
    pub sum_busy_ns: i64,
    /// Wall time during which at least one in-scope event ran.
    pub union_busy_ns: i64,
    /// `sum_busy_ns − union_busy_ns` — time covered by more than one
    /// event; `0` when fully serial.
    pub overlap_ns: i64,
    /// Peak number of in-scope events open at any instant.
    pub max_concurrency: i64,
    /// Unioned compute / copy busy time and their overlap.
    pub compute_vs_copy: ComputeVsCopy,
    /// Per-stream breakdown, ascending `stream_id`. A stream's
    /// `overlap_ns` is its same-stream (e.g. PDL) overlap.
    pub streams: Vec<StreamConcurrency>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct ComputeVsCopy {
    /// Unioned busy time of compute (kernel + graph) events.
    pub compute_union_ns: i64,
    /// Unioned busy time of copy (memcpy + memset) events.
    pub copy_union_ns: i64,
    /// Wall time during which at least one compute event and at least
    /// one copy event were simultaneously running.
    pub compute_copy_overlap_ns: i64,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct StreamConcurrency {
    pub stream_id: i64,
    pub sum_busy_ns: i64,
    pub union_busy_ns: i64,
    pub overlap_ns: i64,
    pub max_concurrency: i64,
}

/// Sum, union, and peak-concurrency over a set of half-open intervals.
/// Intervals that exactly touch (`a.end == b.start`) merge for `union`
/// and do not count as simultaneously open, so back-to-back serial
/// events yield `overlap = 0` and do not inflate `max_concurrency`.
///
/// All three measures fall out of a single endpoint sweep: `sum` is a
/// scan, `union` and `peak` share one sort of the 2N start/end points.
/// At an equal timestamp the ending edge (`-1`) sorts before the
/// starting edge (`+1`), so back-to-back intervals neither inflate the
/// peak nor open a spurious gap in the union — the closing region and
/// the reopening region meet at the shared boundary and sum to the same
/// covered span as a merge would.
fn measures(intervals: &[(i64, i64)]) -> (i64, i64, i64) {
    let sum: i64 = intervals.iter().map(|(s, e)| e - s).sum();

    let mut points: Vec<(i64, i8)> = Vec::with_capacity(intervals.len() * 2);
    for (s, e) in intervals {
        points.push((*s, 1));
        points.push((*e, -1));
    }
    points.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let mut open = 0i64;
    let mut peak = 0i64;
    let mut union = 0i64;
    let mut region_start = 0i64;
    for (t, delta) in points {
        // A new busy region opens when the first interval lifts `open`
        // off zero; it closes when `open` returns to zero.
        if open == 0 && delta == 1 {
            region_start = t;
        }
        open += i64::from(delta);
        if open > peak {
            peak = open;
        }
        if open == 0 {
            union += t - region_start;
        }
    }

    (sum, union, peak)
}

/// Unioned busy time only — the single measure the compute/copy blocks
/// need (their `sum`/`peak` are discarded). Cheaper than [`measures`]:
/// a sort by start plus a merge sweep, with no endpoint-doubling for the
/// peak. `intervals` is sorted in place.
fn union_only(intervals: &mut [(i64, i64)]) -> i64 {
    intervals.sort_unstable_by_key(|(s, _)| *s);
    let mut union = 0i64;
    let mut cur: Option<(i64, i64)> = None;
    for &(s, e) in intervals.iter() {
        match cur {
            None => cur = Some((s, e)),
            // Overlapping or exactly touching → extend the open run.
            Some((cs, ce)) if s <= ce => cur = Some((cs, ce.max(e))),
            Some((cs, ce)) => {
                union += ce - cs;
                cur = Some((s, e));
            }
        }
    }
    if let Some((cs, ce)) = cur {
        union += ce - cs;
    }
    union
}

/// Build the windowed interval-fetch SQL across the GPU-busy kinds that
/// exist in the trace. Projects `(device_id, stream_id, is_compute,
/// start_ns, end_ns)`, clipping each event to the window and keeping
/// only events whose body overlaps it.
fn fetch_sql(
    trace: &Trace,
    abs_window: Option<(i64, i64)>,
) -> NsysQueryResult<(String, Vec<Value>)> {
    let dev = crate::kind_sql::GPU_DEVICE_ID_EXPR;
    let stm = crate::kind_sql::GPU_STREAM_ID_EXPR;
    let work = GpuWorkSet::from_data_definition()?;
    let mut subqueries: Vec<String> = Vec::new();
    for kind in work.present_in(trace) {
        let table = kind.table();
        let compute = match work.class(kind)? {
            GpuWorkClass::Compute => 1,
            GpuWorkClass::Copy => 0,
        };
        subqueries.push(format!(
            "SELECT {dev} AS device_id, {stm} AS stream_id, \
             {compute} AS is_compute, t.start AS start_ns, t.\"end\" AS end_ns \
             FROM nsight.{table} t"
        ));
    }
    if subqueries.is_empty() {
        return Ok((String::new(), Vec::new()));
    }
    let union = subqueries.join(" UNION ALL ");

    let (start_expr, end_expr, where_clause, params): (String, String, String, Vec<Value>) =
        match abs_window {
            Some((s, e)) => (
                "GREATEST(start_ns, ?)".to_string(),
                "LEAST(end_ns, ?)".to_string(),
                "WHERE start_ns < ? AND end_ns > ?".to_string(),
                vec![
                    Value::BigInt(s),
                    Value::BigInt(e),
                    Value::BigInt(e),
                    Value::BigInt(s),
                ],
            ),
            None => (
                "start_ns".to_string(),
                "end_ns".to_string(),
                String::new(),
                Vec::new(),
            ),
        };

    let sql = format!(
        "WITH events AS ({union}) \
         SELECT device_id, stream_id, is_compute, \
         {start_expr} AS s_ns, {end_expr} AS e_ns \
         FROM events {where_clause}"
    );
    Ok((sql, params))
}

/// Per-device accumulator of clipped intervals.
#[derive(Default)]
struct DeviceAccum {
    all: Vec<(i64, i64)>,
    compute: Vec<(i64, i64)>,
    copy: Vec<(i64, i64)>,
    streams: BTreeMap<i64, Vec<(i64, i64)>>,
}

fn hydrate_concurrency_intervals(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
    device: Option<i32>,
) -> NsysQueryResult<BTreeMap<i32, DeviceAccum>> {
    let mut by_device: BTreeMap<i32, DeviceAccum> = BTreeMap::new();
    let rows = crate::query_sql::exec::query_rows(
        conn,
        sql,
        params,
        crate::query_sql::exec::CONCURRENCY_INTERVAL,
        concurrency_interval_row,
    )?;
    for row in rows {
        // A clipped interval can be empty (touches the window edge);
        // drop those so they don't add zero-width noise.
        if row.end_ns <= row.start_ns {
            continue;
        }
        if device.is_some_and(|dev| row.device_id != dev) {
            continue;
        }
        let acc = by_device.entry(row.device_id).or_default();
        acc.all.push((row.start_ns, row.end_ns));
        if row.is_compute != 0 {
            acc.compute.push((row.start_ns, row.end_ns));
        } else {
            acc.copy.push((row.start_ns, row.end_ns));
        }
        acc.streams
            .entry(row.stream_id)
            .or_default()
            .push((row.start_ns, row.end_ns));
    }
    Ok(by_device)
}

struct ConcurrencyIntervalRow {
    device_id: i32,
    stream_id: i64,
    is_compute: i32,
    start_ns: i64,
    end_ns: i64,
}

fn concurrency_interval_row(
    row: &duckdb::Row<'_>,
) -> Result<ConcurrencyIntervalRow, duckdb::Error> {
    Ok(ConcurrencyIntervalRow {
        device_id: row.get(0)?,
        stream_id: row.get(1)?,
        is_compute: row.get(2)?,
        start_ns: row.get(3)?,
        end_ns: row.get(4)?,
    })
}

pub fn run<P: AsRef<Path>>(
    path: P,
    req: ConcurrencyRequest,
) -> NsysQueryResult<ConcurrencyResponse> {
    crate::check_limit(req.limit)?;

    let trace = Trace::open(path).map_err(NsysQueryError::trace_open)?;
    let abs_window = trace
        .resolve_window(req.time_window)
        .map_err(NsysQueryError::time_window_resolve)?;

    let (sql, params) = fetch_sql(&trace, abs_window)?;
    if sql.is_empty() {
        return Ok(ConcurrencyResponse {
            count: 0,
            total_matched: 0,
            time_window_ns: abs_window,
            rows: Vec::new(),
        });
    }

    let by_device = hydrate_concurrency_intervals(trace.conn(), &sql, &params, req.device)?;

    let total_matched = by_device.len() as i64;

    let mut device_rows: Vec<DeviceConcurrency> = Vec::with_capacity(by_device.len());
    for (device_id, mut acc) in by_device {
        let (sum_busy_ns, union_busy_ns, max_concurrency) = measures(&acc.all);
        let compute_union_ns = union_only(&mut acc.compute);
        let copy_union_ns = union_only(&mut acc.copy);
        // Inclusion-exclusion: the time both a compute and a copy event
        // ran is |compute| + |copy| − |compute ∪ copy|, and
        // compute ∪ copy is exactly the device union.
        let compute_copy_overlap_ns = compute_union_ns + copy_union_ns - union_busy_ns;

        let streams: Vec<StreamConcurrency> = acc
            .streams
            .into_iter()
            .map(|(stream_id, iv)| {
                let (sum, union, maxc) = measures(&iv);
                StreamConcurrency {
                    stream_id,
                    sum_busy_ns: sum,
                    union_busy_ns: union,
                    overlap_ns: sum - union,
                    max_concurrency: maxc,
                }
            })
            .collect();

        device_rows.push(DeviceConcurrency {
            key: format!("concurrency|dev:{device_id}"),
            device_id,
            sum_busy_ns,
            union_busy_ns,
            overlap_ns: sum_busy_ns - union_busy_ns,
            max_concurrency,
            compute_vs_copy: ComputeVsCopy {
                compute_union_ns,
                copy_union_ns,
                compute_copy_overlap_ns,
            },
            streams,
        });
    }

    device_rows.truncate(req.limit);

    Ok(ConcurrencyResponse {
        count: device_rows.len(),
        total_matched,
        time_window_ns: abs_window,
        rows: device_rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use veloq_core::VeloqDiagnostic;

    #[test]
    fn measures_serial_back_to_back_has_no_overlap() {
        // [0,10) then [10,20): contiguous union 20, no overlap, peak 1.
        let (sum, union, peak) = measures(&[(0, 10), (10, 20)]);
        assert_eq!(sum, 20);
        assert_eq!(union, 20);
        assert_eq!(sum - union, 0);
        assert_eq!(peak, 1);
    }

    #[test]
    fn measures_counts_overlap_and_peak() {
        // [0,60),[50,100),[30,90): sum 170, union [0,100)=100, overlap
        // 70, and at [50,60) all three are open → peak 3.
        let (sum, union, peak) = measures(&[(0, 60), (50, 100), (30, 90)]);
        assert_eq!(sum, 170);
        assert_eq!(union, 100);
        assert_eq!(sum - union, 70);
        assert_eq!(peak, 3);
    }

    #[test]
    fn measures_same_stream_pdl_overlap_is_counted() {
        // PDL: successor [50,100) starts before predecessor [0,60)
        // retires → 10 of overlap, peak 2 (no serial assumption).
        let (sum, union, peak) = measures(&[(0, 60), (50, 100)]);
        assert_eq!(sum, 110);
        assert_eq!(union, 100);
        assert_eq!(sum - union, 10);
        assert_eq!(peak, 2);
    }

    #[test]
    fn hydrate_concurrency_intervals_prepare_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;

        let err = match hydrate_concurrency_intervals(&conn, "SELECT * FROM", &[], None) {
            Ok(rows) => anyhow::bail!(
                "malformed concurrency SQL should not hydrate successfully: {} devices",
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
    fn hydrate_concurrency_intervals_query_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = "SELECT \
                   ? AS device_id, \
                   0::BIGINT AS stream_id, \
                   1::INTEGER AS is_compute, \
                   0::BIGINT AS s_ns, \
                   10::BIGINT AS e_ns";

        let err = match hydrate_concurrency_intervals(&conn, sql, &[], None) {
            Ok(rows) => anyhow::bail!(
                "unbound concurrency SQL parameter should not hydrate successfully: {} devices",
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
    fn hydrate_concurrency_intervals_read_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = "SELECT \
                   'not-a-device' AS device_id, \
                   0::BIGINT AS stream_id, \
                   1::INTEGER AS is_compute, \
                   0::BIGINT AS s_ns, \
                   10::BIGINT AS e_ns";

        let err = match hydrate_concurrency_intervals(&conn, sql, &[], None) {
            Ok(rows) => anyhow::bail!(
                "malformed concurrency row should not hydrate successfully: {} devices",
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
