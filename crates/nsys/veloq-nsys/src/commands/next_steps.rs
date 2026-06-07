use veloq_core::NextStep;
use veloq_nsys_data::scope::ResolvedScope;

/// Append `--device N` to a hint command when the resolver locked in a
/// concrete device. On single-device traces the user didn't type the
/// flag; the suggestion includes it anyway so the agent can copy-paste
/// without forgetting the scope on a multi-device follow-up.
fn device_suffix(scope: &ResolvedScope) -> String {
    match scope.applied.device {
        Some(d) => format!(" --device {d}"),
        None => String::new(),
    }
}

/// Top-row follow-up for `slices` in instance view: inspect the NVTX
/// range itself, which expands the GPU-attributed kernels under it.
/// `inspect` is row-id-keyed so the resolved scope doesn't carry into
/// the follow-up command — agents that need to keep scoping just re-run
/// `slices --device N` themselves.
pub(super) fn slices_instance_next_steps(
    rows: &[veloq_nsys_query::slices::SlicesRow],
) -> Vec<NextStep> {
    use veloq_nsys_query::slices::SlicesRow;
    let Some(SlicesRow::Instance(top)) = rows.first() else {
        return Vec::new();
    };
    vec![NextStep {
        hint: format!(
            "Drill into the top NVTX range `{}` to see every GPU kernel \
             attributed under it.",
            top.name
        ),
        command: format!("veloq inspect {}", top.row_id),
    }]
}

/// Top-row follow-up for `slices` in aggregate view: re-run `slices`
/// scoped to the heaviest aggregate's NVTX name so the agent sees each
/// instance's contribution.
pub(super) fn slices_aggregate_next_steps(
    rows: &[veloq_nsys_query::slices::SlicesRow],
    scope: &ResolvedScope,
) -> Vec<NextStep> {
    use veloq_nsys_query::slices::SlicesRow;
    let Some(SlicesRow::Aggregate(top)) = rows.first() else {
        return Vec::new();
    };
    let pattern = top.path.as_deref().unwrap_or(&top.name);
    vec![NextStep {
        hint: format!(
            "List per-instance contributions for the heaviest aggregate \
             (`{pattern}`)."
        ),
        command: format!(
            "veloq slices <trace> --name '{}'{}",
            top.name,
            device_suffix(scope)
        ),
    }]
}

/// Top-row follow-up for `stats`: stats rows are aggregates without a
/// row_id, so the natural drill-down is to `search` the same kind for
/// individual events.
pub(super) fn stats_next_steps(
    rows: &[veloq_nsys_query::stats::StatRow],
    scope: &ResolvedScope,
) -> Vec<NextStep> {
    let Some(top) = rows.first() else {
        return Vec::new();
    };
    let name_clause = match top.name.as_deref() {
        Some(n) if !n.is_empty() => format!(" --name '{n}'"),
        _ => String::new(),
    };
    vec![NextStep {
        hint: format!(
            "List individual events behind the top stats row (`{}`).",
            top.name.as_deref().unwrap_or(top.kind)
        ),
        command: format!(
            "veloq search <trace> --type {}{}{}",
            top.kind,
            name_clause,
            device_suffix(scope)
        ),
    }]
}

/// Top-row follow-up for `search`: drill into the headline row with
/// `inspect`.
pub(super) fn search_next_steps(
    rows: &[veloq_nsys_query::event_ref::EventRef],
    _scope: &ResolvedScope,
) -> Vec<NextStep> {
    let Some(top) = rows.first() else {
        return Vec::new();
    };
    vec![NextStep {
        hint: format!(
            "Expand the top hit `{}` with the full headline + NVTX \
             context.",
            top.base().name
        ),
        command: format!("veloq inspect {}", top.base().row_id),
    }]
}

/// Top-row follow-up for `gaps`: inspect both the event that ended the
/// previous activity and the event that started the next one.
pub(super) fn gaps_next_steps(rows: &[veloq_nsys_query::gaps::Gap]) -> Vec<NextStep> {
    let Some(top) = rows.first() else {
        return Vec::new();
    };
    vec![NextStep {
        hint: format!(
            "Inspect the events bracketing the largest gap \
             (`{}` → `{}`).",
            top.prev.name, top.next.name
        ),
        command: format!("veloq inspect {} {}", top.prev.row_id, top.next.row_id),
    }]
}
