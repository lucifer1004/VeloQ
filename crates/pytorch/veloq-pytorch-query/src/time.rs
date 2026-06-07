use crate::{PytorchQueryError, PytorchQueryResult};
use veloq_pytorch_data::{QueryTrace, TimeRange};

pub fn resolve_time_window(
    trace: &QueryTrace,
    from: Option<&str>,
    to: Option<&str>,
) -> PytorchQueryResult<Option<(i64, i64)>> {
    resolve_time_window_from_span(trace.trace_span, from, to)
}

fn resolve_time_window_from_span(
    trace_span: Option<TimeRange>,
    from: Option<&str>,
    to: Option<&str>,
) -> PytorchQueryResult<Option<(i64, i64)>> {
    match (from, to) {
        (None, None) => Ok(None),
        (Some(_), None) | (None, Some(_)) => Err(PytorchQueryError::MissingTimeBound),
        (Some(from), Some(to)) => {
            let origin = trace_span.map(|span| span.start_ns).unwrap_or(0);
            let start = veloq_core::time::TimePoint::parse(from)
                .map_err(|source| PytorchQueryError::invalid_from(from, source))?
                .resolve(origin);
            let end = veloq_core::time::TimePoint::parse(to)
                .map_err(|source| PytorchQueryError::invalid_to(to, source))?
                .resolve(origin);
            if end <= start {
                return Err(PytorchQueryError::EmptyTimeWindow { start, end });
            }
            Ok(Some((start, end)))
        }
    }
}

pub fn parse_group_by(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|axis| !axis.is_empty())
        .map(ToString::to_string)
        .collect()
}
