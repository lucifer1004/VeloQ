use crate::dto::{EventListAuxiliary, EventRef, SearchResponse};
use crate::filter::{EventFilterRequest, filtered_events, require_rank_scope};
use anyhow::Result;
use veloq_pytorch_data::TraceSet;

pub fn search(trace: &TraceSet, request: EventFilterRequest) -> Result<SearchResponse> {
    require_rank_scope(trace, request.rank_scope)?;
    let events = filtered_events(trace, &request)?;
    let total_matched = events.len();
    let rows = events
        .into_iter()
        .take(request.limit)
        .map(EventRef::from)
        .collect::<Vec<_>>();
    Ok(SearchResponse {
        count: rows.len(),
        total_matched,
        rows,
        auxiliary: EventListAuxiliary {
            scope: request.rank_scope.echo(request.step),
            time_window_ns: request.time_window_ns,
        },
    })
}
