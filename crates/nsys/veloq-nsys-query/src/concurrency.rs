//! `veloq concurrency <trace>` — GPU kernel/transfer overlap extraction.
//!
//! Overlap is the union-versus-sum of GPU event
//! intervals: per process-local device (and per stream) we report `sum_busy_ns`, the
//! interval-`union_busy_ns`, their difference `overlap_ns`, and the peak
//! `max_concurrency`, plus a compute-versus-copy block. The union and
//! the concurrency degree are the parts an agent cannot reconstruct in
//! jq from per-event rows — exposing them is the point of the verb.
//!
//! The interval sweep makes no same-stream-serial assumption,
//! so same-stream Programmatic Dependent Launch (PDL) overlap is
//! counted, not dropped. Compute/copy overlap falls out of
//! inclusion-exclusion: `compute_union + copy_union − device_union`.

use duckdb::types::Value;
use rayon::prelude::*;
use serde::Serialize;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
use std::path::Path;
use veloq_core::time::TimeWindow;
use veloq_nsys_data::Trace;

use crate::{
    NsysQueryError, NsysQueryResult,
    query_sql::gpu_work::{GpuWorkClass, GpuWorkSet},
};

#[derive(Debug, Clone)]
pub struct ConcurrencyRequest {
    /// Optional native process filter.
    pub process_id: Option<i64>,
    /// Optional `device_id` filter (NSys `deviceId`). Overlap is always
    /// per process/device; this restricts which rows are emitted.
    pub device: Option<i32>,
    /// Optional window — measures overlap over each event's clipped
    /// portion (overlap-inclusion, like the other windowed verbs).
    pub time_window: Option<TimeWindow>,
    /// Max process/device rows to return.
    pub limit: usize,
}

impl Default for ConcurrencyRequest {
    fn default() -> Self {
        Self {
            process_id: None,
            device: None,
            time_window: None,
            limit: 100,
        }
    }
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct ConcurrencyResponse {
    /// Process/device rows returned (after LIMIT).
    pub count: usize,
    /// Process/device scopes with at least one in-scope event, before LIMIT.
    pub total_matched: i64,
    /// Resolved `--from`/`--to` window, if any (absolute ns).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_window_ns: Option<(i64, i64)>,
    /// Canonical primary list — one row per process/device, ascending
    /// `(process_id, device_id)`.
    pub rows: Vec<DeviceConcurrency>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct DeviceConcurrency {
    /// Cross-trace key. `concurrency|pid:<pid>|dev:<device_id>`.
    pub key: String,
    pub process_id: i64,
    pub device_id: i32,
    /// Sum of in-scope event durations on this process-local device.
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

/// Build the windowed interval-fetch SQL across the GPU-busy kinds that
/// exist in the trace. Projects `(device_id, stream_id, is_compute,
/// start_ns, end_ns)`, clipping each event to the window and keeping
/// only events whose body overlaps it.
fn fetch_sql(
    trace: &Trace,
    abs_window: Option<(i64, i64)>,
    process_id: Option<i64>,
    device: Option<i32>,
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
        let process =
            veloq_nsys_data::process_sql_projection(trace, table, "t", "event_proc", "t.start");
        subqueries.push(format!(
            "SELECT {process_expr} AS process_id, {dev} AS device_id, {stm} AS stream_id, \
             {compute} AS is_compute, t.start AS start_ns, t.\"end\" AS end_ns \
             FROM nsight.{table} t {process_join}",
            process_expr = process.expr,
            process_join = process.join,
        ));
    }
    if subqueries.is_empty() {
        return Ok((String::new(), Vec::new()));
    }
    let union = subqueries.join(" UNION ALL ");

    let (start_expr, end_expr, mut where_parts, mut params): (
        String,
        String,
        Vec<String>,
        Vec<Value>,
    ) = match abs_window {
        Some((s, e)) => (
            "GREATEST(start_ns, ?)".to_string(),
            "LEAST(end_ns, ?)".to_string(),
            vec!["start_ns < ? AND end_ns > ?".to_string()],
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
            Vec::new(),
            Vec::new(),
        ),
    };
    if let Some(process_id) = process_id {
        where_parts.push("process_id = ?".to_string());
        params.push(Value::BigInt(process_id));
    }
    if let Some(device) = device {
        where_parts.push("device_id = ?".to_string());
        params.push(Value::Int(device));
    }
    let where_clause = if where_parts.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", where_parts.join(" AND "))
    };

    let sql = format!(
        "WITH events AS ({union}) \
         SELECT process_id, device_id, stream_id, is_compute, \
         {start_expr} AS s_ns, {end_expr} AS e_ns \
         FROM events {where_clause}"
    );
    Ok((sql, params))
}

struct ConcurrencyInterval {
    process_id: i64,
    device_id: i32,
    stream_id: i64,
    is_compute: i32,
    start_ns: i64,
    end_ns: i64,
}

fn load_concurrency_intervals(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
) -> NsysQueryResult<Vec<ConcurrencyInterval>> {
    crate::query_sql::exec::query_rows(
        conn,
        sql,
        params,
        crate::query_sql::exec::CONCURRENCY_INTERVAL,
        |row| {
            Ok(ConcurrencyInterval {
                process_id: row.get(0)?,
                device_id: row.get(1)?,
                stream_id: row.get(2)?,
                is_compute: row.get(3)?,
                start_ns: row.get(4)?,
                end_ns: row.get(5)?,
            })
        },
    )
}

#[derive(Default)]
struct UnionSweep {
    total_ns: i64,
    current: Option<(i64, i64)>,
}

impl UnionSweep {
    fn push_sorted(&mut self, start_ns: i64, end_ns: i64) {
        match self.current {
            None => self.current = Some((start_ns, end_ns)),
            Some((current_start, current_end)) if start_ns <= current_end => {
                self.current = Some((current_start, current_end.max(end_ns)));
            }
            Some((current_start, current_end)) => {
                self.total_ns += current_end - current_start;
                self.current = Some((start_ns, end_ns));
            }
        }
    }

    fn finish(mut self) -> i64 {
        if let Some((start_ns, end_ns)) = self.current.take() {
            self.total_ns += end_ns - start_ns;
        }
        self.total_ns
    }
}

#[derive(Default)]
struct IntervalSweep {
    sum_ns: i64,
    union: UnionSweep,
    active_ends: BinaryHeap<Reverse<i64>>,
    open_count: i64,
    peak: i64,
}

impl IntervalSweep {
    fn push_sorted(&mut self, start_ns: i64, end_ns: i64) {
        self.sum_ns += end_ns - start_ns;
        self.union.push_sorted(start_ns, end_ns);
        while self.active_ends.peek().is_some_and(|end| end.0 <= start_ns) {
            self.active_ends.pop();
            self.open_count -= 1;
        }
        self.active_ends.push(Reverse(end_ns));
        self.open_count += 1;
        self.peak = self.peak.max(self.open_count);
    }

    fn finish(self) -> (i64, i64, i64) {
        (self.sum_ns, self.union.finish(), self.peak)
    }
}

fn aggregate_sorted_device(
    process_id: i64,
    device_id: i32,
    intervals: &[ConcurrencyInterval],
) -> DeviceConcurrency {
    let mut device = IntervalSweep::default();
    let mut compute = UnionSweep::default();
    let mut copy = UnionSweep::default();
    let mut streams = BTreeMap::<i64, IntervalSweep>::new();

    for interval in intervals {
        device.push_sorted(interval.start_ns, interval.end_ns);
        if interval.is_compute != 0 {
            compute.push_sorted(interval.start_ns, interval.end_ns);
        } else {
            copy.push_sorted(interval.start_ns, interval.end_ns);
        }
        streams
            .entry(interval.stream_id)
            .or_default()
            .push_sorted(interval.start_ns, interval.end_ns);
    }

    let (sum_busy_ns, union_busy_ns, max_concurrency) = device.finish();
    let compute_union_ns = compute.finish();
    let copy_union_ns = copy.finish();
    DeviceConcurrency {
        key: format!("concurrency|pid:{process_id}|dev:{device_id}"),
        process_id,
        device_id,
        sum_busy_ns,
        union_busy_ns,
        overlap_ns: sum_busy_ns - union_busy_ns,
        max_concurrency,
        compute_vs_copy: ComputeVsCopy {
            compute_union_ns,
            copy_union_ns,
            compute_copy_overlap_ns: compute_union_ns + copy_union_ns - union_busy_ns,
        },
        streams: streams
            .into_iter()
            .map(|(stream_id, sweep)| {
                let (sum_busy_ns, union_busy_ns, max_concurrency) = sweep.finish();
                StreamConcurrency {
                    stream_id,
                    sum_busy_ns,
                    union_busy_ns,
                    overlap_ns: sum_busy_ns - union_busy_ns,
                    max_concurrency,
                }
            })
            .collect(),
    }
}

fn optimized_run(
    trace: &Trace,
    req: &ConcurrencyRequest,
    abs_window: Option<(i64, i64)>,
) -> NsysQueryResult<ConcurrencyResponse> {
    let (sql, params) = fetch_sql(trace, abs_window, req.process_id, req.device)?;
    if sql.is_empty() {
        return Ok(ConcurrencyResponse {
            count: 0,
            total_matched: 0,
            time_window_ns: abs_window,
            rows: Vec::new(),
        });
    }
    let mut intervals = load_concurrency_intervals(trace.conn(), &sql, &params)?;
    intervals.retain(|interval| interval.end_ns > interval.start_ns);
    let pool = trace
        .build_query_worker_pool()
        .map_err(NsysQueryError::data)?;
    pool.install(|| intervals.par_sort_unstable_by_key(concurrency_interval_order));

    let groups = intervals
        .chunk_by(|left, right| {
            left.process_id == right.process_id && left.device_id == right.device_id
        })
        .collect::<Vec<_>>();
    let total_matched = groups.len() as i64;
    let aggregate = || {
        groups
            .par_iter()
            .take(req.limit)
            .filter_map(|group| {
                group
                    .first()
                    .map(|first| aggregate_sorted_device(first.process_id, first.device_id, group))
            })
            .collect()
    };
    let rows: Vec<DeviceConcurrency> = pool.install(aggregate);

    Ok(ConcurrencyResponse {
        count: rows.len(),
        total_matched,
        time_window_ns: abs_window,
        rows,
    })
}

fn concurrency_interval_order(interval: &ConcurrencyInterval) -> (i64, i32, i64, i64, i64) {
    (
        interval.process_id,
        interval.device_id,
        interval.start_ns,
        interval.end_ns,
        interval.stream_id,
    )
}

pub fn run<P: AsRef<Path>>(
    path: P,
    req: ConcurrencyRequest,
) -> NsysQueryResult<ConcurrencyResponse> {
    crate::check_limit(req.limit)?;
    let trace = Trace::open(path).map_err(NsysQueryError::trace_open)?;
    run_after_limit(&trace, req)
}

pub fn run_with_trace(
    trace: &Trace,
    req: ConcurrencyRequest,
) -> NsysQueryResult<ConcurrencyResponse> {
    crate::check_limit(req.limit)?;
    run_after_limit(trace, req)
}

pub fn run_with_index(
    trace: &Trace,
    index: &crate::resident_intervals::ResidentIntervalIndex,
    req: ConcurrencyRequest,
) -> NsysQueryResult<ConcurrencyResponse> {
    crate::check_limit(req.limit)?;
    let abs_window = trace
        .resolve_window(req.time_window)
        .map_err(NsysQueryError::time_window_resolve)?;
    let devices = index
        .selected_devices(req.process_id, req.device)
        .collect::<Vec<_>>();
    let pool = trace
        .build_query_worker_pool()
        .map_err(NsysQueryError::data)?;
    let activities = pool.install(|| {
        devices
            .par_iter()
            .filter_map(|device| {
                let activity = device.activity(abs_window);
                if activity.sum_busy_ns > 0 {
                    Some((device.process_id(), device.device_id(), activity))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    });
    let total_matched = activities.len() as i64;
    let rows = activities
        .into_iter()
        .take(req.limit)
        .map(|(process_id, device_id, activity)| {
            device_activity_row(process_id, device_id, activity)
        })
        .collect::<Vec<_>>();
    Ok(ConcurrencyResponse {
        count: rows.len(),
        total_matched,
        time_window_ns: abs_window,
        rows,
    })
}

fn device_activity_row(
    process_id: i64,
    device_id: i32,
    activity: crate::resident_intervals::DeviceActivityEvidence,
) -> DeviceConcurrency {
    DeviceConcurrency {
        key: format!("concurrency|pid:{process_id}|dev:{device_id}"),
        process_id,
        device_id,
        sum_busy_ns: activity.sum_busy_ns,
        union_busy_ns: activity.union_busy_ns,
        overlap_ns: activity.sum_busy_ns - activity.union_busy_ns,
        max_concurrency: activity.max_concurrency,
        compute_vs_copy: ComputeVsCopy {
            compute_union_ns: activity.compute_union_ns,
            copy_union_ns: activity.copy_union_ns,
            compute_copy_overlap_ns: activity.compute_union_ns + activity.copy_union_ns
                - activity.union_busy_ns,
        },
        streams: activity
            .streams
            .into_iter()
            .map(|stream| StreamConcurrency {
                stream_id: stream.stream_id,
                sum_busy_ns: stream.sum_busy_ns,
                union_busy_ns: stream.union_busy_ns,
                overlap_ns: stream.sum_busy_ns - stream.union_busy_ns,
                max_concurrency: stream.max_concurrency,
            })
            .collect(),
    }
}

fn run_after_limit(trace: &Trace, req: ConcurrencyRequest) -> NsysQueryResult<ConcurrencyResponse> {
    let abs_window = trace
        .resolve_window(req.time_window)
        .map_err(NsysQueryError::time_window_resolve)?;

    optimized_run(trace, &req, abs_window)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context as _;

    #[test]
    fn sorted_sweep_preserves_half_open_overlap_math() -> anyhow::Result<()> {
        let intervals = [
            ConcurrencyInterval {
                process_id: 12345,
                device_id: 0,
                stream_id: 7,
                is_compute: 1,
                start_ns: 0,
                end_ns: 60,
            },
            ConcurrencyInterval {
                process_id: 12345,
                device_id: 0,
                stream_id: 8,
                is_compute: 0,
                start_ns: 30,
                end_ns: 90,
            },
            ConcurrencyInterval {
                process_id: 12345,
                device_id: 0,
                stream_id: 7,
                is_compute: 1,
                start_ns: 50,
                end_ns: 100,
            },
            ConcurrencyInterval {
                process_id: 12345,
                device_id: 0,
                stream_id: 8,
                is_compute: 0,
                start_ns: 100,
                end_ns: 110,
            },
        ];
        let device = aggregate_sorted_device(12345, 0, &intervals);

        assert_eq!(device.sum_busy_ns, 180);
        assert_eq!(device.union_busy_ns, 110);
        assert_eq!(device.overlap_ns, 70);
        assert_eq!(device.max_concurrency, 3);
        assert_eq!(device.compute_vs_copy.compute_union_ns, 100);
        assert_eq!(device.compute_vs_copy.copy_union_ns, 70);
        assert_eq!(device.compute_vs_copy.compute_copy_overlap_ns, 60);
        assert_eq!(device.streams.len(), 2);
        let first_stream = device.streams.first().context("missing first stream")?;
        assert_eq!(first_stream.stream_id, 7);
        assert_eq!(first_stream.union_busy_ns, 100);
        assert_eq!(first_stream.max_concurrency, 2);
        let second_stream = device.streams.get(1).context("missing second stream")?;
        assert_eq!(second_stream.stream_id, 8);
        assert_eq!(second_stream.union_busy_ns, 70);
        assert_eq!(second_stream.max_concurrency, 1);
        Ok(())
    }
}
