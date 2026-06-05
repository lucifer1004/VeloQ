use crate::dto::{CollectiveRankRow, CollectiveRow, CollectivesAuxiliary, CollectivesResponse};
use crate::scope::RankScope;
use veloq_pytorch_data::{CollectiveGroup, TraceSet};

pub fn collectives(
    trace: &TraceSet,
    rank_scope: RankScope,
    step: Option<i64>,
    limit: usize,
) -> CollectivesResponse {
    let mut rows = trace
        .collectives
        .iter()
        .filter(|group| step.is_none_or(|step| group.step == Some(step)))
        .map(collective_row)
        .collect::<Vec<_>>();
    rows.sort_by_key(|row| (std::cmp::Reverse(row.duration_ns), row.key.clone()));
    let total_matched = rows.len();
    rows.truncate(limit);
    CollectivesResponse {
        count: rows.len(),
        total_matched,
        rows,
        auxiliary: CollectivesAuxiliary {
            scope: rank_scope.echo(step),
        },
    }
}

fn collective_row(group: &CollectiveGroup) -> CollectiveRow {
    CollectiveRow {
        key: group.key.clone(),
        collective_kind: group.collective_kind.clone(),
        step: group.step,
        ordinal: group.ordinal,
        confidence: group.confidence.clone(),
        start_ns: group.start_ns,
        duration_ns: group.duration_ns,
        skew_ns: group.skew_ns,
        slow_rank: group.slow_rank,
        per_rank: group
            .per_rank
            .iter()
            .map(|rank| CollectiveRankRow {
                rank: rank.rank,
                row_id: rank.row_id.clone(),
                name: rank.name.clone(),
                start_ns: rank.start_ns,
                duration_ns: rank.duration_ns,
            })
            .collect(),
    }
}
