use super::{assert_error_code, assert_schema_envelope, run_veloq};
use anyhow::{Context, Result, anyhow};
use serde_json::Value;

#[test]
fn ncu_summary_envelope_has_single_data_layer() -> Result<()> {
    // Regression test for the v1-envelope migration bug where
    // `summarize_report_with` returned its own envelope-shaped
    // `ReportSummary`, which the dispatcher then wrapped *again* in
    // an `Envelope` — producing `.data.data.sources` instead of the
    // documented `.data.sources`.
    //
    // Uses the committed `source_metric_basic` fixture in-place (it
    // ships a committed `ncu_report` native sidecar), so the native
    // summary path serves NCU-free — a synthetic temp report would
    // have no committed sidecar and need NCU to build one.
    let trace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../ncu/veloq-ncu/tests/fixtures/source_metric_basic.ncu-rep");
    let out = run_veloq(["ncu", "summary", &trace.to_string_lossy()])?;
    assert!(
        out.status.success(),
        "ncu summary on the committed fixture should succeed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value =
        serde_json::from_slice(&out.stdout).context("ncu summary stdout must be valid JSON")?;

    // v1 envelope shape — qualified command, source kind, trace
    // kind, and `data` carrying the SummaryData directly.
    assert_eq!(
        v.get("command").and_then(Value::as_str),
        Some("ncu.summary"),
    );
    assert_eq!(
        v.get("source")
            .and_then(|s| s.get("kind"))
            .and_then(Value::as_str),
        Some("ncu"),
    );
    assert_eq!(
        v.get("trace")
            .and_then(|t| t.get("kind"))
            .and_then(Value::as_str),
        Some("ncu"),
    );

    let data = v
        .get("data")
        .ok_or_else(|| anyhow!("envelope missing `data`: {v}"))?;
    // The double-wrap regression would place a nested
    // `{schema, command, trace, data}` here instead of the canonical
    // fields. v4 native ncu summary: rows + count + total_matched +
    // auxiliary at the top of `.data`; the degraded `session` (NCU
    // version only), `ncu_version`, and `meta_cache_path` live inside
    // `auxiliary`; there is no `file_header_version`.
    for required in ["rows", "count", "total_matched", "auxiliary"] {
        assert!(
            data.get(required).is_some(),
            "data should carry canonical summary field `{required}`: {data}"
        );
    }
    let aux = data
        .get("auxiliary")
        .ok_or_else(|| anyhow!("missing auxiliary"))?;
    for required in ["session", "ncu_version", "meta_cache_path"] {
        assert!(
            aux.get(required).is_some(),
            "auxiliary should carry `{required}`: {aux}"
        );
    }
    assert!(
        data.get("data").is_none(),
        "data must NOT carry a nested `data` field (double-wrap regression): {data}"
    );
    Ok(())
}

#[test]
fn ncu_summary_csv_emits_native_totals_projection() -> Result<()> {
    // `ncu summary --format csv` renders the native totals projection
    // (section/key/value long format) NCU-free from the committed
    // `source_metric_basic` sidecar (the report itself is not committed;
    // build_or_load serves the sidecar). There is no `--page`: the
    // native model has no separate detail/raw/session pages.
    let trace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../ncu/veloq-ncu/tests/fixtures/source_metric_basic.ncu-rep");
    let out = run_veloq([
        "ncu",
        "summary",
        "--format",
        "csv",
        &trace.to_string_lossy(),
    ])?;
    assert!(
        out.status.success(),
        "ncu summary --format csv should succeed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("# command=ncu.summary"),
        "csv output should include command metadata: {stdout}"
    );
    assert!(
        stdout.contains("section,key,value"),
        "csv summary should emit the totals table header: {stdout}"
    );
    assert!(
        stdout.contains("launch_count"),
        "csv summary should emit a launch_count totals row: {stdout}"
    );
    assert!(
        serde_json::from_str::<Value>(&stdout).is_err(),
        "csv must not emit a JSON envelope; got: {stdout}"
    );
    Ok(())
}

#[test]
fn ncu_schema_endpoint_emits_standard_meta_envelope() -> Result<()> {
    let out = run_veloq(["ncu", "schema", "summary"])?;
    let _ = assert_schema_envelope(&out, "ncu.schema", "ncu", "v1", "summary")?;
    Ok(())
}

#[test]
fn ncu_schema_bad_target_has_source_context_and_no_trace_span() -> Result<()> {
    let out = run_veloq(["ncu", "schema", "definitely-not-a-target"])?;
    let v = assert_error_code(&out, "ncu.command.unknown-schema-target")?;
    assert!(
        out.stderr.is_empty(),
        "json error mode should keep stderr empty; stderr={}",
        String::from_utf8_lossy(&out.stderr),
    );
    assert_eq!(v.get("command").and_then(Value::as_str), Some("ncu.schema"),);
    assert_eq!(
        v.get("source")
            .and_then(|s| s.get("kind"))
            .and_then(Value::as_str),
        Some("ncu"),
    );
    assert_eq!(
        v.get("source")
            .and_then(|s| s.get("version"))
            .and_then(Value::as_str),
        Some("v1"),
    );
    assert!(
        v.get("trace").is_none(),
        "ncu schema error envelope must omit trace: {v}"
    );
    assert!(
        v.get("trace_span").is_none(),
        "ncu schema error envelope must omit trace_span: {v}"
    );
    let message = v
        .get("error")
        .and_then(|e| e.get("message"))
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing error.message: {v}"))?;
    assert!(
        message.contains("definitely-not-a-target"),
        "error.message should echo the bad target name; got: {message}"
    );
    Ok(())
}
