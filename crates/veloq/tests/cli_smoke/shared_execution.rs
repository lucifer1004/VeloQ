use super::{build_minimal_trace, run_veloq};
use anyhow::{Context, Result, anyhow};
use std::path::Path;
use tempfile::TempDir;
use veloq_core::{OutputFormat, ProfileSource};
use veloq_ncu::NcuSource;
use veloq_nsys::NsysSource;
use veloq_pytorch::PytorchSource;

fn assert_direct_execution_matches_cli(
    source: &dyn ProfileSource,
    source_args: &[String],
    cli_args: &[String],
    format: OutputFormat,
) -> Result<()> {
    let matches = source
        .cli()
        .try_get_matches_from(source_args)
        .context("parse source command")?;
    let direct = source
        .execute(&matches, format)
        .map_err(|err| anyhow!("execute source command: {err}"))?;
    let process = run_veloq(cli_args)?;

    assert_eq!(
        process.status.code(),
        Some(direct.exit_code()),
        "one-shot exit status diverged for {cli_args:?}"
    );
    assert_eq!(
        process.stdout,
        direct.stdout(),
        "one-shot stdout diverged for {cli_args:?}"
    );
    assert_eq!(
        process.stderr,
        direct.stderr(),
        "one-shot stderr diverged for {cli_args:?}"
    );
    Ok(())
}

fn write_pytorch_trace(dir: &TempDir) -> Result<String> {
    let path = dir.path().join("shared-execution.pt.trace.json");
    std::fs::write(
        &path,
        r#"{
  "traceEvents": [
    { "name": "aten::matmul", "cat": "cpu_op", "ph": "X", "ts": 100, "dur": 80, "pid": 1, "tid": 10, "args": { "External id": 8 } },
    { "name": "matmul_kernel", "cat": "kernel", "ph": "X", "ts": 200, "dur": 120, "pid": 1, "tid": 8, "args": { "External id": 8, "device": 0, "stream": 7 } }
  ]
}"#,
    )
    .context("write PyTorch trace")?;
    Ok(path.to_string_lossy().into_owned())
}

fn ncu_fixture() -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../ncu/veloq-ncu/tests/fixtures/source_metric_basic.ncu-rep")
        .to_string_lossy()
        .into_owned()
}

#[test]
fn source_neutral_execution_preserves_representative_successes() -> Result<()> {
    let (_nsys_dir, nsys_trace) = build_minimal_trace()?;
    let nsys_trace = nsys_trace.to_string_lossy().into_owned();
    assert_direct_execution_matches_cli(
        &NsysSource,
        &["nsys".into(), "summary".into(), nsys_trace.clone()],
        &["nsys".into(), "summary".into(), nsys_trace],
        OutputFormat::Json,
    )?;

    let ncu_trace = ncu_fixture();
    assert_direct_execution_matches_cli(
        &NcuSource,
        &["ncu".into(), "summary".into(), ncu_trace.clone()],
        &["ncu".into(), "summary".into(), ncu_trace],
        OutputFormat::Json,
    )?;

    let pytorch_dir = tempfile::tempdir().context("create PyTorch tempdir")?;
    let pytorch_trace = write_pytorch_trace(&pytorch_dir)?;
    assert_direct_execution_matches_cli(
        &PytorchSource,
        &["pytorch".into(), "summary".into(), pytorch_trace.clone()],
        &["pytorch".into(), "summary".into(), pytorch_trace],
        OutputFormat::Json,
    )
}

#[test]
fn source_neutral_execution_preserves_handled_error_context() -> Result<()> {
    for (source, namespace, trace) in [
        (
            &NsysSource as &dyn ProfileSource,
            "nsys",
            "missing.nsys-rep",
        ),
        (&NcuSource as &dyn ProfileSource, "ncu", "missing.ncu-rep"),
        (
            &PytorchSource as &dyn ProfileSource,
            "pytorch",
            "missing.pt.trace.json",
        ),
    ] {
        let args = vec![
            namespace.to_string(),
            "summary".to_string(),
            trace.to_string(),
        ];
        assert_direct_execution_matches_cli(source, &args, &args, OutputFormat::Json)?;
    }
    Ok(())
}

#[test]
fn source_neutral_execution_preserves_non_json_projection() -> Result<()> {
    let ncu_trace = ncu_fixture();
    assert_direct_execution_matches_cli(
        &NcuSource,
        &["ncu".into(), "summary".into(), ncu_trace.clone()],
        &[
            "--format".into(),
            "csv".into(),
            "ncu".into(),
            "summary".into(),
            ncu_trace,
        ],
        OutputFormat::Csv,
    )
}
