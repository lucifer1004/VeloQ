//! Session-local NSys GPU interval index for daemon-routed scan queries.
//!
//! The index is derived from the already registered, fresh
//! `gpu-work-events` sidecar. It owns no persistent artifact and disappears
//! with the daemon session. Rows are partitioned by native process before
//! device, ordered by start time inside each device, and accompanied by
//! overlap frontiers plus reusable device, stream, process, and trace gap
//! evidence.

use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
use std::mem::size_of;
use veloq_nsys_data::Trace;

use crate::query_sql::exec;
use crate::{EventKind, NsysQueryError, NsysQueryResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuInterval {
    pub kind: EventKind,
    pub row_id: i64,
    pub stream_id: i64,
    pub start_ns: i64,
    pub end_ns: i64,
}

impl GpuInterval {
    pub(crate) fn duration_ns(self) -> i64 {
        self.end_ns - self.start_ns
    }

    fn clipped(self, window: Option<(i64, i64)>) -> Option<Self> {
        let Some((from_ns, to_ns)) = window else {
            return Some(self);
        };
        if self.start_ns >= to_ns || self.end_ns <= from_ns {
            return None;
        }
        Some(Self {
            start_ns: self.start_ns.max(from_ns),
            end_ns: self.end_ns.min(to_ns),
            ..self
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IntervalBoundary {
    pub kind: EventKind,
    pub row_id: i64,
    pub stream_id: i64,
}

impl From<GpuInterval> for IntervalBoundary {
    fn from(interval: GpuInterval) -> Self {
        Self {
            kind: interval.kind,
            row_id: interval.row_id,
            stream_id: interval.stream_id,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GapEvidence {
    pub start_ns: i64,
    pub end_ns: i64,
    pub prev: IntervalBoundary,
    pub next: IntervalBoundary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LocalGapRef {
    prev_index: u32,
    next_index: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ProcessIntervalLocation {
    device_index: u32,
    interval_index: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ProcessGapRef {
    prev: ProcessIntervalLocation,
    next: ProcessIntervalLocation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TraceIntervalLocation {
    process_index: u32,
    device_index: u32,
    interval_index: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TraceGapRef {
    prev: TraceIntervalLocation,
    next: TraceIntervalLocation,
}

impl GapEvidence {
    pub(crate) fn duration_ns(self) -> i64 {
        self.end_ns - self.start_ns
    }

    pub(crate) fn overlaps(self, window: Option<(i64, i64)>) -> bool {
        window.is_none_or(|(from_ns, to_ns)| self.start_ns < to_ns && self.end_ns > from_ns)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StreamActivityEvidence {
    pub stream_id: i64,
    pub sum_busy_ns: i64,
    pub union_busy_ns: i64,
    pub max_concurrency: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeviceActivityEvidence {
    pub sum_busy_ns: i64,
    pub union_busy_ns: i64,
    pub max_concurrency: i64,
    pub compute_union_ns: i64,
    pub copy_union_ns: i64,
    pub streams: Vec<StreamActivityEvidence>,
}

#[derive(Debug)]
pub(crate) struct StreamPartition {
    stream_id: i64,
    sum_busy_ns: i64,
    gaps: Vec<LocalGapRef>,
}

impl StreamPartition {
    pub(crate) fn stream_id(&self) -> i64 {
        self.stream_id
    }

    pub(crate) fn sum_busy_ns(&self) -> i64 {
        self.sum_busy_ns
    }

    fn retained_memory_estimate_bytes(&self) -> u64 {
        vec_capacity_bytes::<LocalGapRef>(self.gaps.capacity())
    }
}

#[derive(Debug)]
pub(crate) struct DevicePartition {
    process_id: i64,
    device_id: i32,
    intervals: Vec<GpuInterval>,
    prefix_max_end: Vec<i64>,
    full_activity: DeviceActivityEvidence,
    streams: Vec<StreamPartition>,
    gaps: Vec<LocalGapRef>,
}

impl DevicePartition {
    fn build(process_id: i64, device_id: i32, intervals: Vec<GpuInterval>) -> Option<Self> {
        u32::try_from(intervals.len()).ok()?;
        let mut prefix_max_end = Vec::with_capacity(intervals.len());
        let mut max_end = i64::MIN;
        for interval in &intervals {
            max_end = max_end.max(interval.end_ns);
            prefix_max_end.push(max_end);
        }

        let full_activity = activity_for_sorted(intervals.iter().copied());
        let gaps = unified_local_gap_refs(&intervals)?;

        let mut by_stream = BTreeMap::<i64, Vec<usize>>::new();
        for (interval_index, interval) in intervals.iter().enumerate() {
            by_stream
                .entry(interval.stream_id)
                .or_default()
                .push(interval_index);
        }
        let streams = by_stream
            .into_iter()
            .map(|(stream_id, interval_indices)| {
                Some(StreamPartition {
                    stream_id,
                    sum_busy_ns: interval_indices
                        .iter()
                        .filter_map(|index| intervals.get(*index))
                        .copied()
                        .map(GpuInterval::duration_ns)
                        .sum(),
                    gaps: stream_gap_refs(&intervals, &interval_indices)?,
                })
            })
            .collect::<Option<Vec<_>>>()?;

        Some(Self {
            process_id,
            device_id,
            intervals,
            prefix_max_end,
            full_activity,
            streams,
            gaps,
        })
    }

    pub(crate) fn process_id(&self) -> i64 {
        self.process_id
    }

    pub(crate) fn device_id(&self) -> i32 {
        self.device_id
    }

    pub(crate) fn intervals(
        &self,
        window: Option<(i64, i64)>,
    ) -> impl Iterator<Item = GpuInterval> + '_ {
        self.candidate_slice(window)
            .iter()
            .filter_map(move |interval| interval.clipped(window))
    }

    pub(crate) fn activity(&self, window: Option<(i64, i64)>) -> DeviceActivityEvidence {
        match window {
            None => self.full_activity.clone(),
            Some(_) => activity_for_sorted(self.intervals(window)),
        }
    }

    pub(crate) fn streams(&self) -> &[StreamPartition] {
        &self.streams
    }

    pub(crate) fn gaps(&self) -> impl Iterator<Item = GapEvidence> + '_ {
        self.gaps
            .iter()
            .filter_map(|gap| resolve_local_gap(&self.intervals, *gap))
    }

    pub(crate) fn stream_gaps<'a>(
        &'a self,
        stream: &'a StreamPartition,
    ) -> impl Iterator<Item = GapEvidence> + 'a {
        stream
            .gaps
            .iter()
            .filter_map(|gap| resolve_local_gap(&self.intervals, *gap))
    }

    fn candidate_slice(&self, window: Option<(i64, i64)>) -> &[GpuInterval] {
        let Some((from_ns, to_ns)) = window else {
            return &self.intervals;
        };
        let start = self
            .prefix_max_end
            .partition_point(|max_end| *max_end <= from_ns);
        let end = self
            .intervals
            .partition_point(|interval| interval.start_ns < to_ns);
        self.intervals.get(start..end).unwrap_or(&[])
    }

    fn retained_memory_estimate_bytes(&self) -> u64 {
        vec_capacity_bytes::<GpuInterval>(self.intervals.capacity())
            .saturating_add(vec_capacity_bytes::<i64>(self.prefix_max_end.capacity()))
            .saturating_add(vec_capacity_bytes::<StreamActivityEvidence>(
                self.full_activity.streams.capacity(),
            ))
            .saturating_add(vec_capacity_bytes::<StreamPartition>(
                self.streams.capacity(),
            ))
            .saturating_add(
                self.streams
                    .iter()
                    .map(StreamPartition::retained_memory_estimate_bytes)
                    .fold(0, u64::saturating_add),
            )
            .saturating_add(vec_capacity_bytes::<LocalGapRef>(self.gaps.capacity()))
    }
}

#[derive(Debug)]
pub(crate) struct ProcessPartition {
    process_id: i64,
    devices: Vec<DevicePartition>,
    trace_gaps: Vec<ProcessGapRef>,
}

impl ProcessPartition {
    pub(crate) fn devices(&self) -> &[DevicePartition] {
        &self.devices
    }

    fn retained_memory_estimate_bytes(&self) -> u64 {
        vec_capacity_bytes::<DevicePartition>(self.devices.capacity())
            .saturating_add(
                self.devices
                    .iter()
                    .map(DevicePartition::retained_memory_estimate_bytes)
                    .fold(0, u64::saturating_add),
            )
            .saturating_add(vec_capacity_bytes::<ProcessGapRef>(
                self.trace_gaps.capacity(),
            ))
    }
}

/// Disposable interval index retained by one NSys daemon session.
#[derive(Debug)]
pub struct ResidentIntervalIndex {
    processes: Vec<ProcessPartition>,
    trace_gaps: Vec<TraceGapRef>,
    retained_memory_estimate_bytes: u64,
}

impl ResidentIntervalIndex {
    pub(crate) fn selected_devices(
        &self,
        process_id: Option<i64>,
        device_id: Option<i32>,
    ) -> impl Iterator<Item = &DevicePartition> {
        self.processes
            .iter()
            .filter(move |process| process_id.is_none_or(|pid| process.process_id == pid))
            .flat_map(ProcessPartition::devices)
            .filter(move |device| device_id.is_none_or(|id| device.device_id == id))
    }

    pub(crate) fn visit_trace_gaps(
        &self,
        process_id: Option<i64>,
        mut visit: impl FnMut(GapEvidence),
    ) {
        match process_id {
            Some(process_id) => {
                if let Some(process) = self
                    .processes
                    .iter()
                    .find(|process| process.process_id == process_id)
                {
                    for gap in &process.trace_gaps {
                        if let Some(gap) = resolve_process_gap(process, *gap) {
                            visit(gap);
                        }
                    }
                }
            }
            None => {
                for gap in &self.trace_gaps {
                    if let Some(gap) = resolve_trace_gap(&self.processes, *gap) {
                        visit(gap);
                    }
                }
            }
        }
    }

    pub fn retained_memory_estimate_bytes(&self) -> u64 {
        self.retained_memory_estimate_bytes
    }
}

#[derive(Debug, Clone, Copy)]
struct SourceInterval {
    process_id: i64,
    device_id: i32,
    interval: GpuInterval,
}

#[derive(Debug, Clone, Copy)]
struct RawSourceInterval {
    kind_code: i32,
    row_id: i64,
    process_id: i64,
    device_id: i32,
    stream_id: i64,
    start_ns: i64,
    end_ns: i64,
}

/// Build the index from an already registered fresh sidecar.
///
/// `Ok(None)` means the sidecar is absent, contains unsupported rows, has no
/// work, or cannot fit beneath the daemon's existing retained-memory ceiling.
pub fn build(
    trace: &Trace,
    resident_memory_ceiling_bytes: u64,
) -> NsysQueryResult<Option<ResidentIntervalIndex>> {
    if !veloq_nsys_data::gpu_work_events::view_available(trace) || has_unsupported_rows(trace)? {
        return Ok(None);
    }

    let row_count = trace
        .conn()
        .query_row("SELECT COUNT(*) FROM nsight.gpu_work_events", [], |row| {
            row.get::<_, i64>(0)
        })
        .map_err(|source| NsysQueryError::sql_query("resident intervals", "row-count", source))?;
    if row_count <= 0 {
        return Ok(None);
    }
    let row_count = u64::try_from(row_count).unwrap_or(u64::MAX);
    if conservative_retained_bound(row_count) > resident_memory_ceiling_bytes {
        log::debug!(
            "resident intervals: conservative retained bound exceeds daemon capacity; using established query paths"
        );
        return Ok(None);
    }

    let raw = exec::query_rows(
        trace.conn(),
        "SELECT \
            CASE kind \
                WHEN 'kernel' THEN 0 \
                WHEN 'memcpy' THEN 1 \
                WHEN 'memset' THEN 2 \
                WHEN 'graph' THEN 3 \
                ELSE -1 \
            END::INTEGER AS kind_code, \
            CAST(row_id AS BIGINT), CAST(process_id AS BIGINT), \
            CAST(device_id AS INTEGER), CAST(stream_id AS BIGINT), \
            CAST(start_ns AS BIGINT), CAST(end_ns AS BIGINT) \
         FROM nsight.gpu_work_events",
        &[],
        exec::RESIDENT_INTERVAL,
        |row| {
            Ok(RawSourceInterval {
                kind_code: row.get(0)?,
                row_id: row.get(1)?,
                process_id: row.get(2)?,
                device_id: row.get(3)?,
                stream_id: row.get(4)?,
                start_ns: row.get(5)?,
                end_ns: row.get(6)?,
            })
        },
    )?;
    let mut source = raw
        .into_iter()
        .map(|raw| {
            Ok(SourceInterval {
                process_id: raw.process_id,
                device_id: raw.device_id,
                interval: GpuInterval {
                    kind: work_kind_from_code(raw.kind_code)?,
                    row_id: raw.row_id,
                    stream_id: raw.stream_id,
                    start_ns: raw.start_ns,
                    end_ns: raw.end_ns,
                },
            })
        })
        .collect::<NsysQueryResult<Vec<_>>>()?;

    let pool = trace
        .build_query_worker_pool()
        .map_err(NsysQueryError::data)?;
    pool.install(|| {
        source.par_sort_unstable_by_key(|event| {
            (
                event.process_id,
                event.device_id,
                event.interval.start_ns,
                event.interval.row_id,
                event.interval.kind.as_str(),
            )
        });
    });

    let mut grouped = Vec::<(i64, i32, Vec<GpuInterval>)>::new();
    for event in source {
        match grouped.last_mut() {
            Some((process_id, device_id, intervals))
                if *process_id == event.process_id && *device_id == event.device_id =>
            {
                intervals.push(event.interval);
            }
            _ => grouped.push((event.process_id, event.device_id, vec![event.interval])),
        }
    }
    let devices = pool.install(|| {
        grouped
            .into_par_iter()
            .map(|(process_id, device_id, intervals)| {
                DevicePartition::build(process_id, device_id, intervals)
            })
            .collect::<Vec<_>>()
    });
    let Some(devices) = devices.into_iter().collect::<Option<Vec<_>>>() else {
        log::debug!(
            "resident intervals: a partition exceeds compact index capacity; using established query paths"
        );
        return Ok(None);
    };

    let mut processes = Vec::<ProcessPartition>::new();
    for device in devices {
        match processes.last_mut() {
            Some(process) if process.process_id == device.process_id => {
                process.devices.push(device);
            }
            _ => processes.push(ProcessPartition {
                process_id: device.process_id,
                devices: vec![device],
                trace_gaps: Vec::new(),
            }),
        }
    }
    let process_gap_refs = pool.install(|| {
        processes
            .par_iter()
            .map(|process| process_trace_gap_refs(&process.devices))
            .collect::<Vec<_>>()
    });
    let Some(process_gap_refs) = process_gap_refs.into_iter().collect::<Option<Vec<_>>>() else {
        return Ok(None);
    };
    for (process, gaps) in processes.iter_mut().zip(process_gap_refs) {
        process.trace_gaps = gaps;
    }
    let Some(trace_gaps) = trace_gap_refs(&processes) else {
        return Ok(None);
    };

    let mut index = ResidentIntervalIndex {
        processes,
        trace_gaps,
        retained_memory_estimate_bytes: 0,
    };
    index.retained_memory_estimate_bytes = index_memory_bytes(&index);
    if index.retained_memory_estimate_bytes > resident_memory_ceiling_bytes {
        log::debug!(
            "resident intervals: measured retained size exceeds daemon capacity; using established query paths"
        );
        return Ok(None);
    }
    Ok(Some(index))
}

fn activity_for_sorted(intervals: impl IntoIterator<Item = GpuInterval>) -> DeviceActivityEvidence {
    let mut device = ActivitySweep::default();
    let mut compute = UnionSweep::default();
    let mut copy = UnionSweep::default();
    let mut streams = BTreeMap::<i64, ActivitySweep>::new();

    for interval in intervals {
        device.push_sorted(interval.start_ns, interval.end_ns);
        if matches!(interval.kind, EventKind::Kernel | EventKind::Graph) {
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
    DeviceActivityEvidence {
        sum_busy_ns,
        union_busy_ns,
        max_concurrency,
        compute_union_ns: compute.finish(),
        copy_union_ns: copy.finish(),
        streams: streams
            .into_iter()
            .map(|(stream_id, sweep)| {
                let (sum_busy_ns, union_busy_ns, max_concurrency) = sweep.finish();
                StreamActivityEvidence {
                    stream_id,
                    sum_busy_ns,
                    union_busy_ns,
                    max_concurrency,
                }
            })
            .collect(),
    }
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
struct ActivitySweep {
    sum_ns: i64,
    union: UnionSweep,
    active_ends: BinaryHeap<Reverse<i64>>,
    open_count: i64,
    peak: i64,
}

impl ActivitySweep {
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

fn stream_gap_refs(
    intervals: &[GpuInterval],
    interval_indices: &[usize],
) -> Option<Vec<LocalGapRef>> {
    interval_indices
        .windows(2)
        .map(|pair| {
            let prev_index = pair.first().copied()?;
            let next_index = pair.get(1).copied()?;
            let previous = intervals.get(prev_index)?;
            let next = intervals.get(next_index)?;
            Some((next.start_ns > previous.end_ns).then_some(LocalGapRef {
                prev_index: u32::try_from(prev_index).ok()?,
                next_index: u32::try_from(next_index).ok()?,
            }))
        })
        .collect::<Option<Vec<_>>>()
        .map(|gaps| gaps.into_iter().flatten().collect())
}

fn unified_local_gap_refs(intervals: &[GpuInterval]) -> Option<Vec<LocalGapRef>> {
    let mut gaps = Vec::new();
    let mut frontier_index: Option<usize> = None;
    for (event_index, event) in intervals.iter().enumerate() {
        if let Some(previous_index) = frontier_index {
            let Some(previous) = intervals.get(previous_index) else {
                continue;
            };
            if event.start_ns > previous.end_ns {
                gaps.push(LocalGapRef {
                    prev_index: u32::try_from(previous_index).ok()?,
                    next_index: u32::try_from(event_index).ok()?,
                });
            }
            if event.end_ns > previous.end_ns {
                frontier_index = Some(event_index);
            }
        } else {
            frontier_index = Some(event_index);
        }
    }
    Some(gaps)
}

#[derive(Debug, Clone, Copy)]
struct LocatedInterval<L> {
    process_id: i64,
    device_id: i32,
    location: L,
    interval: GpuInterval,
}

fn unified_gap_pairs<L: Copy>(mut intervals: Vec<LocatedInterval<L>>) -> Vec<(L, L)> {
    intervals.sort_unstable_by_key(|event| {
        (
            event.interval.start_ns,
            event.interval.row_id,
            event.interval.kind.as_str(),
            event.process_id,
            event.device_id,
        )
    });
    let mut gaps = Vec::new();
    let mut frontier: Option<LocatedInterval<L>> = None;
    for event in intervals {
        if let Some(previous) = frontier {
            if event.interval.start_ns > previous.interval.end_ns {
                gaps.push((previous.location, event.location));
            }
            if event.interval.end_ns > previous.interval.end_ns {
                frontier = Some(event);
            }
        } else {
            frontier = Some(event);
        }
    }
    gaps
}

fn process_trace_gap_refs(devices: &[DevicePartition]) -> Option<Vec<ProcessGapRef>> {
    let intervals =
        devices
            .iter()
            .enumerate()
            .flat_map(|(device_index, device)| {
                device.intervals.iter().copied().enumerate().map(
                    move |(interval_index, interval)| {
                        Some(LocatedInterval {
                            process_id: device.process_id,
                            device_id: device.device_id,
                            location: ProcessIntervalLocation {
                                device_index: u32::try_from(device_index).ok()?,
                                interval_index: u32::try_from(interval_index).ok()?,
                            },
                            interval,
                        })
                    },
                )
            })
            .collect::<Option<Vec<_>>>()?;
    Some(
        unified_gap_pairs(intervals)
            .into_iter()
            .map(|(prev, next)| ProcessGapRef { prev, next })
            .collect(),
    )
}

fn trace_gap_refs(processes: &[ProcessPartition]) -> Option<Vec<TraceGapRef>> {
    let intervals = processes
        .iter()
        .enumerate()
        .flat_map(|(process_index, process)| {
            process
                .devices
                .iter()
                .enumerate()
                .flat_map(move |(device_index, device)| {
                    device.intervals.iter().copied().enumerate().map(
                        move |(interval_index, interval)| {
                            Some(LocatedInterval {
                                process_id: device.process_id,
                                device_id: device.device_id,
                                location: TraceIntervalLocation {
                                    process_index: u32::try_from(process_index).ok()?,
                                    device_index: u32::try_from(device_index).ok()?,
                                    interval_index: u32::try_from(interval_index).ok()?,
                                },
                                interval,
                            })
                        },
                    )
                })
        })
        .collect::<Option<Vec<_>>>()?;
    Some(
        unified_gap_pairs(intervals)
            .into_iter()
            .map(|(prev, next)| TraceGapRef { prev, next })
            .collect(),
    )
}

fn resolve_local_gap(intervals: &[GpuInterval], gap: LocalGapRef) -> Option<GapEvidence> {
    gap_evidence(
        *intervals.get(usize::try_from(gap.prev_index).ok()?)?,
        *intervals.get(usize::try_from(gap.next_index).ok()?)?,
    )
}

fn resolve_process_gap(process: &ProcessPartition, gap: ProcessGapRef) -> Option<GapEvidence> {
    let previous = process
        .devices
        .get(usize::try_from(gap.prev.device_index).ok()?)?
        .intervals
        .get(usize::try_from(gap.prev.interval_index).ok()?)?;
    let next = process
        .devices
        .get(usize::try_from(gap.next.device_index).ok()?)?
        .intervals
        .get(usize::try_from(gap.next.interval_index).ok()?)?;
    gap_evidence(*previous, *next)
}

fn resolve_trace_gap(processes: &[ProcessPartition], gap: TraceGapRef) -> Option<GapEvidence> {
    let resolve = |location: TraceIntervalLocation| {
        processes
            .get(usize::try_from(location.process_index).ok()?)?
            .devices
            .get(usize::try_from(location.device_index).ok()?)?
            .intervals
            .get(usize::try_from(location.interval_index).ok()?)
            .copied()
    };
    gap_evidence(resolve(gap.prev)?, resolve(gap.next)?)
}

fn gap_evidence(previous: GpuInterval, next: GpuInterval) -> Option<GapEvidence> {
    (next.start_ns > previous.end_ns).then_some(GapEvidence {
        start_ns: previous.end_ns,
        end_ns: next.start_ns,
        prev: previous.into(),
        next: next.into(),
    })
}

fn work_kind_from_code(code: i32) -> NsysQueryResult<EventKind> {
    match code {
        0 => Ok(EventKind::Kernel),
        1 => Ok(EventKind::Memcpy),
        2 => Ok(EventKind::Memset),
        3 => Ok(EventKind::Graph),
        _ => Err(NsysQueryError::internal_sql_kind_tag_invalid(
            "resident intervals",
            &code.to_string(),
        )),
    }
}

fn has_unsupported_rows(trace: &Trace) -> NsysQueryResult<bool> {
    trace
        .conn()
        .query_row(
            "SELECT EXISTS(\
                SELECT 1 FROM nsight.gpu_work_events \
                WHERE process_id IS NULL OR end_ns <= start_ns \
                LIMIT 1\
            )",
            [],
            |row| row.get(0),
        )
        .map_err(|source| NsysQueryError::sql_query("resident intervals", "eligibility", source))
}

fn conservative_retained_bound(row_count: u64) -> u64 {
    let per_row = size_of::<GpuInterval>()
        .saturating_add(size_of::<i64>())
        .saturating_add(size_of::<LocalGapRef>().saturating_mul(2))
        .saturating_add(size_of::<ProcessGapRef>())
        .saturating_add(size_of::<TraceGapRef>());
    row_count.saturating_mul(u64::try_from(per_row).unwrap_or(u64::MAX))
}

fn index_memory_bytes(index: &ResidentIntervalIndex) -> u64 {
    u64::try_from(size_of::<ResidentIntervalIndex>())
        .unwrap_or(u64::MAX)
        .saturating_add(vec_capacity_bytes::<ProcessPartition>(
            index.processes.capacity(),
        ))
        .saturating_add(
            index
                .processes
                .iter()
                .map(ProcessPartition::retained_memory_estimate_bytes)
                .fold(0, u64::saturating_add),
        )
        .saturating_add(vec_capacity_bytes::<TraceGapRef>(
            index.trace_gaps.capacity(),
        ))
}

fn vec_capacity_bytes<T>(capacity: usize) -> u64 {
    u64::try_from(capacity)
        .unwrap_or(u64::MAX)
        .saturating_mul(u64::try_from(size_of::<T>()).unwrap_or(u64::MAX))
}
