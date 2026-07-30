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
    assert!(source.detect(Path::new("report.ncu-repz")));
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
    for target in veloq_ncu::schema_targets::TARGETS {
        let schema = veloq_ncu::schema::schema_value_for(target.name)?;
        assert!(
            schema.is_object(),
            "{} schema should be an object",
            target.name
        );
    }
    Ok(())
}

#[test]
fn schema_target_arg_help_lists_every_registry_target() -> Result<()> {
    let source = NcuSource;
    let schema = source
        .cli()
        .find_subcommand("schema")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("schema subcommand not found"))?;
    let long_about = schema
        .get_long_about()
        .map(|about| about.to_string())
        .unwrap_or_default();
    let help = schema
        .get_arguments()
        .find(|arg| arg.get_id() == "target")
        .and_then(clap::Arg::get_help)
        .map(|help| help.to_string())
        .unwrap_or_default();
    for target in veloq_ncu::schema_targets::TARGETS {
        assert!(
            help.contains(target.name),
            "schema target arg help missing `{}`",
            target.name
        );
        assert!(
            long_about.contains(target.name),
            "schema long_about missing `{}`",
            target.name
        );
    }
    Ok(())
}

#[test]
fn unknown_schema_target_has_command_error_code() -> Result<()> {
    let err = veloq_ncu::schema::schema_value_for("bogus")
        .err()
        .ok_or_else(|| anyhow::anyhow!("expected schema target error"))?;
    assert_eq!(err.code().as_str(), "ncu.command.unknown-schema-target");
    let msg = err.to_string();
    for target in veloq_ncu::schema_targets::TARGETS {
        assert!(
            msg.contains(target.name),
            "unknown target error should list `{}`",
            target.name
        );
    }
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
