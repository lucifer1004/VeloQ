//! Silent-failure guardrails for list-verb responses.
//!
//! Touchpoint 5 ("why nothing / weird number?"):
//! list verbs that return surprising shapes — zero rows under an
//! explicit scope filter, or a time-window suspiciously narrow
//! against the trace span — append a `Warning` to the envelope's
//! `meta.warnings[]` instead of leaving the agent to guess.
//!
//! These helpers live in `veloq-core` (not `veloq`) so each
//! profile-source crate can wire them directly from its verb
//! dispatch.
//!
//! ## Single-source-of-truth thresholds
//!
//! Tuning lives here: the "very narrow" cutoff is 1% of the
//! trace span, the "empty under scope" rule fires only when at
//! least one scope filter was explicitly set. Both are conservative
//! to avoid noise — they should never fire on a well-formed query
//! that returns a real zero-row result (e.g. a verb on a trace
//! that genuinely has no matching events).

use crate::meta::{AppliedScope, Warning, WarningCode, WarningSeverity};

/// Threshold below which the time-window guard considers the window
/// "very narrow" relative to the trace span. 1% is intentionally
/// conservative — agents that asked for a deliberate `--from 0 --to
/// 100ms` window on a 10s trace will still trip it (1.0%), which is
/// fine: the warning isn't an error, just a hint.
pub const NARROW_WINDOW_RATIO_PERCENT: i64 = 1;

/// Build a `narrow-window` warning when `window_ns` is below
/// [`NARROW_WINDOW_RATIO_PERCENT`] of `trace_span_ns`. Returns `None`
/// when either input is missing/non-positive (no warning to emit) or
/// the window is comfortably wide.
///
/// The check is purely numeric — it doesn't reason about *why* the
/// window is small. Agents that genuinely want a small slice (e.g.
/// timeline at 100ms intervals over 10s) are pinged anyway and can
/// ignore the warning; agents that typo'd `--from 100 --to 200` on a
/// multi-second trace see exactly the prod they need.
pub fn check_time_window(
    window_ns: Option<(i64, i64)>,
    trace_span_ns: Option<i64>,
) -> Option<Warning> {
    let (start, end) = window_ns?;
    let span = trace_span_ns?;
    if span <= 0 {
        return None;
    }
    let width = end.saturating_sub(start);
    if width <= 0 {
        return None;
    }
    // `width * 100` can overflow on absurd inputs; saturate so the
    // comparison stays in the i64 domain without panicking.
    let ratio = width.saturating_mul(100) / span;
    if ratio >= NARROW_WINDOW_RATIO_PERCENT {
        return None;
    }
    Some(Warning {
        severity: WarningSeverity::Warn,
        code: WarningCode::NarrowWindow,
        message: format!(
            "time window {width} ns is under {NARROW_WINDOW_RATIO_PERCENT}% of the \
             trace span ({span} ns); double-check units — `--from 100 --to 200` is \
             100 ns, not 100 ms. Use unit suffixes (1ms, 10us) or the `@<ns>` \
             absolute marker."
        ),
    })
}

/// Build an `empty-with-scope` warning when a list verb returns zero
/// rows AND the caller asked for at least one scope filter (device,
/// stream, native_pid, kind, NVTX pattern, or time window). Returns
/// `None` when the row count is positive or when no scope filter was
/// explicitly set — empty result with no scope is its own honest
/// answer.
///
/// `aggregated_over` is *not* a scope filter: `--all-devices` is an
/// opt-in aggregation, not a narrowing predicate.
pub fn check_empty_result(row_count: usize, scope: &AppliedScope) -> Option<Warning> {
    if row_count > 0 {
        return None;
    }
    let mut filters: Vec<&'static str> = Vec::new();
    if scope.device.is_some() {
        filters.push("--device");
    }
    if scope.stream.is_some() {
        filters.push("--stream");
    }
    if scope.native_pid.is_some() {
        filters.push("native_pid");
    }
    if scope.kind.is_some() {
        filters.push("--type");
    }
    if scope.nvtx_pattern.is_some() {
        filters.push("--nvtx");
    }
    if scope.time_window_ns.is_some() {
        filters.push("--from/--to");
    }
    if filters.is_empty() {
        return None;
    }
    let joined = filters.join(", ");
    Some(Warning {
        severity: WarningSeverity::Warn,
        code: WarningCode::EmptyWithScope,
        message: format!(
            "scope filter ({joined}) excluded every row. Drop a filter or widen the \
             query — an empty list often means the scope didn't match anything in \
             this trace."
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn narrow_window_skips_when_inputs_missing() {
        assert!(check_time_window(None, Some(1_000_000_000)).is_none());
        assert!(check_time_window(Some((0, 100)), None).is_none());
    }

    #[test]
    fn narrow_window_skips_when_ratio_above_threshold() {
        // 100 ms over a 1 s span = 10% — well above 1%.
        assert!(check_time_window(Some((0, 100_000_000)), Some(1_000_000_000)).is_none());
    }

    #[test]
    fn narrow_window_fires_when_ratio_below_threshold() -> anyhow::Result<()> {
        // 1 us over a 1 s span = 0.0001% — well below 1%.
        let w = check_time_window(Some((0, 1_000)), Some(1_000_000_000))
            .ok_or_else(|| anyhow::anyhow!("narrow window must trip the guard"))?;
        assert!(matches!(w.code, WarningCode::NarrowWindow));
        assert!(matches!(w.severity, WarningSeverity::Warn));
        Ok(())
    }

    #[test]
    fn narrow_window_handles_inverted_endpoints() {
        // end <= start → no width → no warning (caller already
        // bailed earlier on the SQL side; the guard just stays
        // silent).
        assert!(check_time_window(Some((100, 50)), Some(1_000)).is_none());
    }

    #[test]
    fn empty_result_skips_when_rows_present() {
        let scope = AppliedScope {
            device: Some(0),
            ..AppliedScope::default()
        };
        assert!(check_empty_result(1, &scope).is_none());
    }

    #[test]
    fn empty_result_skips_when_no_filters_set() {
        let scope = AppliedScope::default();
        assert!(check_empty_result(0, &scope).is_none());
    }

    #[test]
    fn empty_result_fires_when_device_filter_excluded_everything() -> anyhow::Result<()> {
        let scope = AppliedScope {
            device: Some(7),
            ..AppliedScope::default()
        };
        let w = check_empty_result(0, &scope).ok_or_else(|| {
            anyhow::anyhow!("zero rows under explicit --device must trip the guard")
        })?;
        assert!(matches!(w.code, WarningCode::EmptyWithScope));
        assert!(
            w.message.contains("--device"),
            "message must mention the failing filter: {}",
            w.message
        );
        Ok(())
    }

    #[test]
    fn empty_result_ignores_aggregated_over_marker() {
        // --all-devices populates `aggregated_over` but no narrowing
        // filter — guard should stay silent.
        let scope = AppliedScope {
            aggregated_over: vec!["device".to_string()],
            ..AppliedScope::default()
        };
        assert!(check_empty_result(0, &scope).is_none());
    }
}
