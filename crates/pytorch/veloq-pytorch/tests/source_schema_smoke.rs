use anyhow::Result;
use std::fs;
use veloq_core::ProfileSource;
use veloq_pytorch::PytorchSource;

#[test]
fn source_detects_pytorch_inputs() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let trace_path = dir.path().join("worker0.pt.trace.json");
    fs::write(&trace_path, r#"{"traceEvents":[]}"#)?;
    let source = PytorchSource;
    assert!(source.detect(&trace_path));
    assert!(source.detect(dir.path()));
    assert!(!source.detect(&dir.path().join("report.ncu-rep")));
    Ok(())
}

#[test]
fn schema_targets_are_registered() -> Result<()> {
    for target in [
        "summary",
        "search",
        "inspect",
        "stats",
        "correlate",
        "timeline",
        "slices",
        "collectives",
        "prep",
    ] {
        let schema = veloq_pytorch::schema::schema_value_for(target)?;
        assert!(schema.is_object(), "{target} schema should be an object");
    }
    Ok(())
}
