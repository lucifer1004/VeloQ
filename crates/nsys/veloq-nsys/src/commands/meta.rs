use std::path::Path;
use veloq_core::{NextStep, ResponseMeta, TraceSpan, Warning, guards};
use veloq_nsys_data::scope::ResolvedScope;

/// Wrap a resolved scope into the envelope's optional `meta` block.
/// The verb dispatch fills `kind` / `nvtx_pattern` / `time_window_ns`
/// from its own parsed args (the resolver leaves them None) before
/// constructing the meta block. `next_steps` carries the verb's
/// follow-up hints (computed from the top result row); pass `vec![]`
/// when the verb has nothing to suggest (empty result, or a verb that
/// hasn't been wired yet).
pub(super) fn meta_with_scope(
    scope: &ResolvedScope,
    kind: Option<String>,
    nvtx_pattern: Option<String>,
    time_window_ns: Option<(i64, i64)>,
    next_steps: Vec<NextStep>,
    warnings: Vec<Warning>,
) -> Option<ResponseMeta> {
    let mut applied = scope.applied.clone();
    applied.kind = kind;
    applied.nvtx_pattern = nvtx_pattern;
    applied.time_window_ns = time_window_ns;
    Some(ResponseMeta {
        applied_scope: Some(applied),
        next_steps,
        warnings,
    })
}

/// Compose every guardrail check into one warning list. Each
/// `check_*` returns `Option<Warning>`; we collect the `Some`s in
/// declaration order so the wire shape is deterministic across runs.
/// `applied` is the resolved scope projected with kind/nvtx/window
/// already populated — same value the envelope ships, so the guards
/// see exactly what the user asked for.
///
/// The trace span is re-read from the meta sidecar when the pre-
/// dispatch hook returned `None` (e.g. cold parquetdir on the first
/// call). The query path above will have built the sidecar by now,
/// so this is the same warm read `emit_with_meta` performs after
/// dispatch.
pub(super) fn run_guards(
    row_count: usize,
    applied: &veloq_core::AppliedScope,
    trace: &Path,
    trace_span: Option<TraceSpan>,
) -> Vec<Warning> {
    // The narrow-window guard needs the trace span as a denominator.
    // Pre-dispatch we may have None (no warm sidecar); re-read it now,
    // and as a last resort build it eagerly so the guard fires on
    // cold-first-run too. Skip the build when there's no time window
    // to compare against — the guard wouldn't fire anyway and we'd
    // pay the cost for nothing.
    let span = if applied.time_window_ns.is_some() {
        trace_span
            .or_else(|| veloq_nsys_data::meta_cache::trace_span_for_path(trace))
            .or_else(|| resolve_trace_span_eager(trace))
            .map(|s| s.span_ns)
    } else {
        None
    };
    let mut warnings = Vec::new();
    if let Some(w) = guards::check_time_window(applied.time_window_ns, span) {
        warnings.push(w);
    }
    if let Some(w) = guards::check_empty_result(row_count, applied) {
        warnings.push(w);
    }
    warnings
}

/// Build (or warm-load) the meta sidecar so the guards have a non-
/// `None` trace span to compare against. Best-effort — any failure
/// silently leaves the span as `None` (the narrow-window guard then
/// simply doesn't fire, which is the conservative default).
fn resolve_trace_span_eager(trace: &Path) -> Option<TraceSpan> {
    let trace_handle = veloq_nsys_data::Trace::open(trace).ok()?;
    let cache = trace_handle.meta_cache().ok()?;
    Some(TraceSpan {
        origin_ns: cache.origins.primary.start_ns,
        span_ns: cache.origins.primary.duration_ns(),
    })
}

/// Materialise the projected `AppliedScope` that a verb will assign to
/// its envelope — same kind/nvtx/window fields the meta builder fills
/// — so the guards see exactly what the agent will read back.
pub(super) fn projected_scope(
    scope: &ResolvedScope,
    kind: Option<&str>,
    nvtx_pattern: Option<&str>,
    time_window_ns: Option<(i64, i64)>,
) -> veloq_core::AppliedScope {
    let mut applied = scope.applied.clone();
    applied.kind = kind.map(str::to_string);
    applied.nvtx_pattern = nvtx_pattern.map(str::to_string);
    applied.time_window_ns = time_window_ns;
    applied
}
