use anyhow::Result;
use std::collections::BTreeSet;
use std::path::Path;
use veloq_core::{ProfileSource, VeloqDiagnostic};
use veloq_ncu::NcuSource;
use veloq_ncu::native::{MetricSubtype, MetricType, NativeMetric, RollupOp};

#[test]
fn source_identity_and_detection_match_wire_contract() {
    let source = NcuSource;
    assert_eq!(source.kind(), "ncu");
    assert_eq!(source.version(), "v1");
    assert!(source.detect(Path::new("report.ncu-rep")));
    assert!(!source.detect(Path::new("trace.nsys-rep")));
    assert!(!source.detect(Path::new("worker0.pt.trace.json")));
    assert!(!source.detect(Path::new("report.ncu-rep.veloq")));
}

#[test]
fn stable_command_surface_matches_rfc_0008() {
    let actual: BTreeSet<String> = NcuSource
        .cli()
        .get_subcommands()
        .map(|cmd| cmd.get_name().to_string())
        .collect();
    let expected: BTreeSet<String> = [
        "disasm",
        "graphs",
        "inspect",
        "launches",
        "metrics",
        "ranges",
        "schema",
        "source-metrics",
        "sources",
        "summary",
        "warp-stalls",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    assert_eq!(actual, expected);
}

#[test]
fn schema_targets_are_registered() -> Result<()> {
    for target in [
        "summary",
        "launches",
        "inspect",
        "metrics",
        "disasm",
        "ranges",
        "graphs",
        "sources",
        "source-metrics",
        "warp-stalls",
    ] {
        let schema = veloq_ncu::schema::schema_value_for(target)?;
        assert!(schema.is_object(), "{target} schema should be an object");
    }
    Ok(())
}

#[test]
fn unknown_schema_target_has_command_error_code() -> Result<()> {
    let err = veloq_ncu::schema::schema_value_for("bogus")
        .err()
        .ok_or_else(|| anyhow::anyhow!("expected schema target error"))?;
    assert_eq!(err.code().as_str(), "ncu.command.unknown-schema-target");
    Ok(())
}

#[test]
fn metric_enum_fields_use_names_unknown_fallback_and_preserve_codes() -> Result<()> {
    let metric: NativeMetric = serde_json::from_value(serde_json::json!({
        "name": "future_metric",
        "label": "Future Metric",
        "unit": "unit",
        "value": 42.0,
        "value_type": "double",
        "metric_type": "future_metric_type",
        "metric_type_code": 9001,
        "metric_subtype": "future_metric_subtype",
        "metric_subtype_code": 9002,
        "rollup": "future_rollup",
        "rollup_code": 9003
    }))?;

    assert_eq!(metric.metric_type, MetricType::Unknown);
    assert_eq!(metric.metric_type_code, Some(9001));
    assert_eq!(metric.metric_subtype, Some(MetricSubtype::Unknown));
    assert_eq!(metric.metric_subtype_code, Some(9002));
    assert_eq!(metric.rollup, Some(RollupOp::Unknown));
    assert_eq!(metric.rollup_code, Some(9003));

    let rendered = serde_json::to_value(metric)?;
    assert_eq!(
        rendered
            .get("metric_type")
            .and_then(serde_json::Value::as_str),
        Some("unknown")
    );
    assert_eq!(
        rendered
            .get("metric_type_code")
            .and_then(serde_json::Value::as_i64),
        Some(9001)
    );
    assert_eq!(
        rendered
            .get("metric_subtype")
            .and_then(serde_json::Value::as_str),
        Some("unknown")
    );
    assert_eq!(
        rendered
            .get("metric_subtype_code")
            .and_then(serde_json::Value::as_i64),
        Some(9002)
    );
    assert_eq!(
        rendered.get("rollup").and_then(serde_json::Value::as_str),
        Some("unknown")
    );
    assert_eq!(
        rendered
            .get("rollup_code")
            .and_then(serde_json::Value::as_i64),
        Some(9003)
    );
    Ok(())
}
