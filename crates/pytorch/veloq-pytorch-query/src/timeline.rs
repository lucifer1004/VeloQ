use crate::dto::{TimelineAuxiliary, TimelineBucketRow, TimelineResponse};
use crate::filter::{EventFilterRequest, filtered_events, require_rank_scope};
use anyhow::Result;
use std::collections::BTreeMap;
use veloq_pytorch_data::{Event, TraceSet};

#[derive(Default)]
struct BucketAcc {
    cpu_ns: i64,
    gpu_ns: i64,
    comm_ns: i64,
    event_count: usize,
    by_type_ns: BTreeMap<String, i64>,
}

pub fn timeline(
    trace: &TraceSet,
    request: EventFilterRequest,
    interval_ns: i64,
) -> Result<TimelineResponse> {
    require_rank_scope(trace, request.rank_scope)?;
    if interval_ns <= 0 {
        anyhow::bail!("--interval must be greater than 0 ns");
    }
    let origin = trace.trace_span.map(|span| span.start_ns).unwrap_or(0);
    let window = request.time_window_ns.or_else(|| {
        trace
            .trace_span
            .map(|span| (span.start_ns, span.end_ns.max(span.start_ns)))
    });
    let (window_start, window_end) = window.unwrap_or((origin, origin));
    let mut buckets: BTreeMap<i64, BucketAcc> = BTreeMap::new();
    for event in filtered_events(trace, &request)? {
        add_event_to_buckets(
            event,
            interval_ns,
            origin,
            window_start,
            window_end,
            &mut buckets,
        );
    }
    let total_matched = buckets.len();
    let mut rows = Vec::new();
    for (bucket_start, acc) in buckets {
        rows.push(TimelineBucketRow {
            key: format!(
                "bucket|{}..{}",
                bucket_start,
                bucket_start.saturating_add(interval_ns)
            ),
            start_ns: bucket_start,
            end_ns: bucket_start.saturating_add(interval_ns),
            cpu_ns: acc.cpu_ns,
            gpu_ns: acc.gpu_ns,
            comm_ns: acc.comm_ns,
            event_count: acc.event_count,
            by_type_ns: acc.by_type_ns,
        });
    }
    rows.truncate(request.limit);
    Ok(TimelineResponse {
        count: rows.len(),
        total_matched,
        rows,
        auxiliary: TimelineAuxiliary {
            scope: request.rank_scope.echo(request.step),
            interval_ns,
            time_window_ns: request.time_window_ns,
        },
    })
}

fn add_event_to_buckets(
    event: &Event,
    interval_ns: i64,
    origin: i64,
    window_start: i64,
    window_end: i64,
    buckets: &mut BTreeMap<i64, BucketAcc>,
) {
    let start = event.start_ns.max(window_start);
    let end = event.end_ns.max(event.start_ns).min(window_end);
    if end <= start {
        return;
    }
    let offset = start.saturating_sub(origin);
    let mut bucket_start = origin.saturating_add((offset / interval_ns) * interval_ns);
    while bucket_start < end {
        let bucket_end = bucket_start.saturating_add(interval_ns);
        let overlap = end.min(bucket_end).saturating_sub(start.max(bucket_start));
        if overlap > 0 {
            let acc = buckets.entry(bucket_start).or_default();
            if event.is_gpu_activity() {
                acc.gpu_ns = acc.gpu_ns.saturating_add(overlap);
            } else {
                acc.cpu_ns = acc.cpu_ns.saturating_add(overlap);
            }
            if event.is_comm {
                acc.comm_ns = acc.comm_ns.saturating_add(overlap);
            }
            acc.event_count += 1;
            let type_key = event.event_type.as_str().to_string();
            let entry = acc.by_type_ns.entry(type_key).or_default();
            *entry = entry.saturating_add(overlap);
        }
        bucket_start = bucket_start.saturating_add(interval_ns);
    }
}
