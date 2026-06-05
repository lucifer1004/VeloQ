//! Additivity rule for `--by line` / `--by file` rollup.
//!
//! A counter is **non-additive** iff at least one of:
//!
//! 1. [`crate::source_metrics::units::infer_metric_unit`] returns a unit
//!    string containing `"per_"` — the rate families (`.per_second`,
//!    `.per_cycle_active`, `.per_cycle_elapsed`).
//! 2. The name matches [`is_percent_metric`]'s rules
//!    (`.pct`, `.pct_*`, `_pct`, `_pct_*`).
//! 3. The name ends in one of four explicit suffixes: `.ratio`,
//!    `.avg`, `.max`, `.min`.
//!
//! Anchoring on the existing `infer_metric_unit` predicate is
//! deliberate: any new rate suffix NCU adds in a future release is
//! caught the moment that helper learns it, with no parallel
//! maintenance loop here.

use crate::native::{MetricSubtype, MetricType, NativeMetric, RollupOp};
use crate::source_metrics::units::infer_metric_unit;

const EXPLICIT_NON_ADDITIVE_SUFFIXES: &[&str] = &[".ratio", ".avg", ".max", ".min"];

/// Additivity for a native metric: use
/// `ncu_report`'s own `metric_type` /
/// `rollup_operation` / `metric_subtype` classification where it
/// suffices, falling back to the name-suffix rule for counters
/// `ncu_report` doesn't classify (`metric_type` is `Other`/`Unknown`, or
/// `rollup` is absent). The fallback is not a parallel policy — it's the
/// same [`is_additive`] kept as the documented backstop. Matching is on
/// the `ncu_report` enum *names* (resolved Python-side from the live
/// enum), so an `ncu` version that renumbers its enum codes cannot
/// silently flip this classification, and a name a future `ncu` *adds*
/// resolves to the `Unknown` arm (→ this same fallback).
pub fn is_additive_native(metric: &NativeMetric) -> bool {
    match classify(metric) {
        Some(additive) => additive,
        None => is_additive(&metric.name),
    }
}

/// `Some(additive)` when `ncu_report` classifies the metric
/// authoritatively; `None` when it doesn't (caller falls back to the
/// name-suffix rule). A percent / ratio / per-second subtype is never
/// SUM-additive on a source line; a THROUGHPUT or RATIO base type
/// isn't either; a COUNTER is additive iff its rollup is SUM.
fn classify(metric: &NativeMetric) -> Option<bool> {
    if matches!(
        metric.metric_subtype,
        Some(MetricSubtype::Pct | MetricSubtype::Ratio | MetricSubtype::PerSecond)
    ) {
        return Some(false);
    }
    match metric.metric_type {
        MetricType::Ratio | MetricType::Throughput => Some(false),
        MetricType::Counter => match metric.rollup {
            Some(RollupOp::Sum) => Some(true),
            Some(_) => Some(false),
            None => None,
        },
        MetricType::Other | MetricType::Unknown => None,
    }
}

/// `true` when summing this counter across SASS instructions on a
/// source line (or file) is meaningful.
pub fn is_additive(metric_name: &str) -> bool {
    !is_non_additive(metric_name)
}

/// `true` when summing this counter would be meaningless. See module
/// docs for the rule.
pub fn is_non_additive(metric_name: &str) -> bool {
    if is_percent_metric(metric_name) {
        return true;
    }
    if matches_per_family(metric_name) {
        return true;
    }
    EXPLICIT_NON_ADDITIVE_SUFFIXES
        .iter()
        .any(|s| metric_name.ends_with(s))
}

/// Mirror of the private predicate in [`crate::source_metrics::units`].
/// Kept in sync — both modules use the same rule.
fn is_percent_metric(name: &str) -> bool {
    name.ends_with(".pct")
        || name.contains(".pct_")
        || name.ends_with("_pct")
        || name.contains("_pct_")
}

/// `true` when this metric is a rate-family counter. Catches the
/// canonical NCU rate suffixes directly, AND falls back to
/// `infer_metric_unit` returning a `*_per_*` unit so any future
/// rate variant the helper learns is automatically picked up here.
/// `hertz` (returned by the helper for `cycle/.per_second`
/// combinations) is also recognised — it's a rate unit that doesn't
/// happen to spell out `per_`.
fn matches_per_family(name: &str) -> bool {
    if name.ends_with(".per_second")
        || name.ends_with(".per_cycle_active")
        || name.ends_with(".per_cycle_elapsed")
    {
        return true;
    }
    matches!(infer_metric_unit(name), Some(u) if u.contains("per_") || u == "hertz")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::{MetricSubtype, MetricType, NativeMetric, RollupOp};

    fn metric(
        name: &str,
        ty: MetricType,
        subtype: Option<MetricSubtype>,
        rollup: Option<RollupOp>,
    ) -> NativeMetric {
        NativeMetric {
            name: name.to_string(),
            label: None,
            unit: None,
            value: serde_json::Value::Null,
            value_type: "double".to_string(),
            metric_type: ty,
            metric_type_code: None,
            metric_subtype: subtype,
            metric_subtype_code: None,
            rollup,
            rollup_code: None,
            instances: None,
        }
    }

    /// Name-based classification: a
    /// COUNTER with SUM rollup is additive; THROUGHPUT/RATIO and the
    /// pct/ratio/per_second subtypes are not; `Unknown`/absent routes to
    /// the name-suffix fallback rather than guessing.
    #[test]
    fn classify_matches_enum_names() {
        // COUNTER + SUM => additive.
        assert!(is_additive_native(&metric(
            "smsp__sass_thread_inst_executed.sum",
            MetricType::Counter,
            None,
            Some(RollupOp::Sum),
        )));
        // COUNTER + non-SUM rollup => non-additive.
        assert!(!is_additive_native(&metric(
            "x.avg",
            MetricType::Counter,
            None,
            Some(RollupOp::Avg),
        )));
        // THROUGHPUT / RATIO base type => non-additive regardless of name.
        assert!(!is_additive_native(&metric(
            "additive_looking_name",
            MetricType::Throughput,
            None,
            None,
        )));
        // pct subtype => non-additive even on a COUNTER.
        assert!(!is_additive_native(&metric(
            "x",
            MetricType::Counter,
            Some(MetricSubtype::Pct),
            Some(RollupOp::Sum),
        )));
        // Unknown type (a future ncu addition) => name-suffix fallback:
        // an additive-looking name stays additive.
        assert!(is_additive_native(&metric(
            "derived__memory_l1_conflicts_shared_nway",
            MetricType::Unknown,
            None,
            None,
        )));
        // Unknown type + non-additive *name* => fallback rejects it.
        assert!(!is_additive_native(&metric(
            "x.per_second",
            MetricType::Unknown,
            None,
            None,
        )));
        // COUNTER with no rollup => fallback (rollup absent is not decisive).
        assert!(is_additive_native(&metric(
            "smsp__sass_thread_inst_executed.sum",
            MetricType::Counter,
            None,
            None,
        )));
    }

    #[test]
    fn motivating_case_is_additive() {
        // The user-agent investigation used this counter
        // (double-valued instances, additive name).
        // The earlier "uint64-only ⇒ additive" heuristic would have
        // silently dropped it. The rule must accept it.
        assert!(is_additive("derived__memory_l1_conflicts_shared_nway"));
    }

    #[test]
    fn additive_canonical_names() {
        assert!(is_additive("smsp__sass_thread_inst_executed.sum"));
        assert!(is_additive("derived__memory_l1_wavefronts_shared"));
        assert!(is_additive("smsp__sass_inst_executed_op_global_ld.sum"));
    }

    #[test]
    fn rejects_pct_suffixes() {
        assert!(is_non_additive(
            "sm__cycles_active.avg.pct_of_peak_sustained_elapsed"
        ));
        assert!(is_non_additive("foo.pct"));
        assert!(is_non_additive("smsp__pct_of_warps"));
        assert!(is_non_additive("foo_pct"));
    }

    #[test]
    fn rejects_per_family_via_infer_metric_unit() {
        // .per_second / .per_cycle_active / .per_cycle_elapsed all
        // come back from infer_metric_unit as *_per_* units; the
        // rule catches them via that helper, not a parallel list.
        assert!(is_non_additive("sm__cycles_active.per_second"));
        assert!(is_non_additive("smsp__inst_executed.per_cycle_active"));
        assert!(is_non_additive("smsp__inst_executed.per_cycle_elapsed"));
    }

    #[test]
    fn rejects_explicit_four_suffixes() {
        assert!(is_non_additive("smsp__some_metric.ratio"));
        assert!(is_non_additive("smsp__some_metric.avg"));
        assert!(is_non_additive("smsp__some_metric.max"));
        assert!(is_non_additive("smsp__some_metric.min"));
    }

    #[test]
    fn is_additive_is_complement_of_is_non_additive() {
        for name in [
            "derived__memory_l1_conflicts_shared_nway",
            "sm__cycles_active.pct_of_peak_sustained_elapsed",
            "sm__cycles_active.per_second",
            "smsp__inst_executed.per_cycle_active",
            "smsp__inst_executed.per_cycle_elapsed",
            "smsp__some_metric.ratio",
            "smsp__some_metric.avg",
            "smsp__some_metric.max",
            "smsp__some_metric.min",
            "smsp__sass_thread_inst_executed.sum",
        ] {
            assert_eq!(is_additive(name), !is_non_additive(name), "name = {name}");
        }
    }
}
