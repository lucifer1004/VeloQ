use anyhow::{Context, Result};
use clap::FromArgMatches;
use std::fs;
use std::path::Path;
use veloq_core::ProfileSource;
use veloq_core::{EnvelopeError, OutputFormat, SourceRef, VeloqDiagnostic};
use veloq_pytorch::PytorchSource;
use veloq_pytorch::cli::Cmd;

#[test]
fn source_detects_pytorch_inputs() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let trace_path = dir.path().join("worker0.pt.trace.json");
    fs::write(&trace_path, r#"{"traceEvents":[]}"#)?;
    let source = PytorchSource;
    assert!(source.detect(&trace_path));
    assert!(!source.detect(dir.path()));
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

#[test]
fn unknown_schema_target_has_command_error_code() -> Result<()> {
    let err = veloq_pytorch::schema::schema_value_for("bogus")
        .err()
        .ok_or_else(|| anyhow::anyhow!("expected schema target error"))?;
    assert_eq!(err.code().as_str(), "pytorch.command.unknown-schema-target");
    Ok(())
}

#[test]
fn collectives_accepts_rank_scope_flags() -> Result<()> {
    let source = PytorchSource;
    let matches = source.cli().try_get_matches_from([
        "pytorch",
        "collectives",
        "worker0.pt.trace.json",
        "--rank",
        "1",
    ])?;
    let cmd = Cmd::from_arg_matches(&matches)?;
    let Cmd::Collectives {
        rank, all_ranks, ..
    } = cmd
    else {
        anyhow::bail!("expected collectives command");
    };
    assert_eq!(rank, Some(1));
    assert!(!all_ranks);

    let matches = source.cli().try_get_matches_from([
        "pytorch",
        "collectives",
        "worker0.pt.trace.json",
        "--all-ranks",
    ])?;
    let cmd = Cmd::from_arg_matches(&matches)?;
    let Cmd::Collectives {
        rank, all_ranks, ..
    } = cmd
    else {
        anyhow::bail!("expected collectives command");
    };
    assert_eq!(rank, None);
    assert!(all_ranks);
    Ok(())
}

#[test]
fn data_errors_survive_anyhow_context_for_diagnostic_projection() -> Result<()> {
    let data_err =
        veloq_pytorch_data::PytorchDataError::directory_inputs_unsupported(Path::new("traces"));
    let wrapped = Err::<(), _>(data_err)
        .context("running pytorch summary")
        .err()
        .ok_or_else(|| anyhow::anyhow!("expected wrapped error"))?;
    let data_err = wrapped
        .downcast_ref::<veloq_pytorch_data::PytorchDataError>()
        .ok_or_else(|| anyhow::anyhow!("expected pytorch data error downcast"))?;

    let env = EnvelopeError::from_diagnostic(
        Some(SourceRef {
            kind: "pytorch",
            version: "v0",
        }),
        Some("pytorch.summary".to_string()),
        None,
        None,
        data_err,
    );
    let json: serde_json::Value = serde_json::from_str(&env.to_json()?)?;
    assert_eq!(
        json.pointer("/error/code")
            .and_then(serde_json::Value::as_str),
        Some("pytorch.input.directory-unsupported")
    );
    let message = data_err.to_string();
    assert_eq!(
        json.pointer("/error/message")
            .and_then(serde_json::Value::as_str),
        Some(message.as_str())
    );
    assert_eq!(
        data_err.code().as_str(),
        "pytorch.input.directory-unsupported"
    );
    Ok(())
}

#[test]
fn query_errors_survive_anyhow_context_for_diagnostic_projection() -> Result<()> {
    let query_err = veloq_pytorch_query::PytorchQueryError::MultiRankRequiresScope;
    let wrapped = Err::<(), _>(query_err)
        .context("running pytorch collectives")
        .err()
        .ok_or_else(|| anyhow::anyhow!("expected wrapped error"))?;
    let query_err = wrapped
        .downcast_ref::<veloq_pytorch_query::PytorchQueryError>()
        .ok_or_else(|| anyhow::anyhow!("expected pytorch query error downcast"))?;

    let env = EnvelopeError::from_diagnostic(
        Some(SourceRef {
            kind: "pytorch",
            version: "v0",
        }),
        Some("pytorch.collectives".to_string()),
        None,
        None,
        query_err,
    );
    let json: serde_json::Value = serde_json::from_str(&env.to_json()?)?;
    assert_eq!(
        json.pointer("/error/code")
            .and_then(serde_json::Value::as_str),
        Some("pytorch.query.rank-scope-required")
    );
    let message = query_err.to_string();
    assert_eq!(
        json.pointer("/error/message")
            .and_then(serde_json::Value::as_str),
        Some(message.as_str())
    );
    assert_eq!(
        query_err.code().as_str(),
        "pytorch.query.rank-scope-required"
    );
    Ok(())
}

#[test]
fn command_errors_survive_anyhow_context_for_diagnostic_projection() -> Result<()> {
    let command_err =
        veloq_pytorch::PytorchCommandError::unsupported_schema_format(OutputFormat::Csv);
    let wrapped = Err::<(), _>(command_err)
        .context("running pytorch schema")
        .err()
        .ok_or_else(|| anyhow::anyhow!("expected wrapped error"))?;
    let command_err = wrapped
        .downcast_ref::<veloq_pytorch::PytorchCommandError>()
        .ok_or_else(|| anyhow::anyhow!("expected pytorch command error downcast"))?;

    let env = EnvelopeError::from_diagnostic(
        Some(SourceRef {
            kind: "pytorch",
            version: "v0",
        }),
        Some("pytorch.schema".to_string()),
        None,
        None,
        command_err,
    );
    let json: serde_json::Value = serde_json::from_str(&env.to_json()?)?;
    assert_eq!(
        json.pointer("/error/code")
            .and_then(serde_json::Value::as_str),
        Some("pytorch.command.unsupported-schema-format")
    );
    let message = command_err.to_string();
    assert_eq!(
        json.pointer("/error/message")
            .and_then(serde_json::Value::as_str),
        Some(message.as_str())
    );
    assert_eq!(
        command_err.code().as_str(),
        "pytorch.command.unsupported-schema-format"
    );
    Ok(())
}
