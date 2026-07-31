use crate::PytorchSourceError;
use crate::cli::Cmd;
use crate::commands;
use clap::{ArgMatches, Command, FromArgMatches, Subcommand};
use std::path::Path;
use veloq_core::{
    EnvelopeError, EnvelopeTraceRef, NextStep, OutputFormat, ProfileSource, ResponseMeta,
    SourceExecution, SourceQueryContext, SourceRef, SourceRunResult, TraceSpan, Warning,
    WarningCode, WarningSeverity, shell_quote,
};
use veloq_pytorch_query::PytorchQueryError;

pub struct PytorchSource;

impl PytorchSource {
    pub const KIND: &'static str = "pytorch";
    pub const VERSION: &'static str = "v0";

    pub(crate) fn source_ref() -> SourceRef {
        SourceRef {
            kind: Self::KIND,
            version: Self::VERSION,
        }
    }

    pub(crate) fn trace_ref(trace: &Path) -> EnvelopeTraceRef {
        EnvelopeTraceRef {
            kind: Self::KIND,
            path: trace.display().to_string(),
        }
    }
}

impl ProfileSource for PytorchSource {
    fn kind(&self) -> &'static str {
        Self::KIND
    }

    fn version(&self) -> &'static str {
        Self::VERSION
    }

    fn detect(&self, trace: &Path) -> bool {
        veloq_pytorch_data::detect_path(trace)
    }

    fn compute_trace_span(&self, trace: &Path) -> Option<TraceSpan> {
        veloq_pytorch_data::trace_span_for_path(trace)
    }

    fn cli(&self) -> Command {
        let parent = Command::new(Self::KIND)
            .about("Query PyTorch Kineto/Profiler Chrome trace .json(.gz) files")
            .subcommand_required(true)
            .arg_required_else_help(true);
        crate::help::inject_long_about(Cmd::augment_subcommands(parent))
    }

    fn query_context(&self, matches: &ArgMatches) -> SourceRunResult<SourceQueryContext> {
        let cmd = Cmd::from_arg_matches(matches)?;
        Ok(SourceQueryContext {
            command: format!("{}.{}", Self::KIND, cmd.name()),
            trace_path: cmd.trace_path().map(Path::to_path_buf),
            raw_stdout: false,
        })
    }

    fn execute(&self, matches: &ArgMatches, fmt: OutputFormat) -> SourceRunResult<SourceExecution> {
        let cmd = Cmd::from_arg_matches(matches)?;
        let verb = cmd.name();
        let trace_path = cmd.trace_path().map(Path::to_path_buf);
        let mut output = SourceExecution::new();
        match commands::run(cmd, trace_path.as_deref(), fmt, &mut output) {
            Ok(code) => {
                output.set_exit_code(code);
                Ok(output)
            }
            Err(err) => {
                let span = trace_path
                    .as_deref()
                    .and_then(|path| self.compute_trace_span(path));
                emit_err(verb, trace_path.as_deref(), span, &err, fmt, &mut output);
                output.set_exit_code(1);
                Ok(output)
            }
        }
    }
}

fn emit_err(
    verb: &str,
    trace: Option<&Path>,
    trace_span: Option<TraceSpan>,
    err: &PytorchSourceError,
    fmt: OutputFormat,
    output: &mut SourceExecution,
) {
    match err {
        PytorchSourceError::Command(err) => {
            emit_diagnostic(verb, trace, trace_span, err, fmt, output)
        }
        PytorchSourceError::Data(err) => emit_diagnostic(verb, trace, trace_span, err, fmt, output),
        PytorchSourceError::Query(rank_err @ PytorchQueryError::MultiRankRequiresScope) => {
            emit_rank_scope_error(verb, trace, trace_span, rank_err, fmt, output);
        }
        PytorchSourceError::Query(err) => {
            emit_diagnostic(verb, trace, trace_span, err, fmt, output)
        }
        PytorchSourceError::Tabular(err) => {
            emit_diagnostic(verb, trace, trace_span, err, fmt, output)
        }
        PytorchSourceError::SerializeEnvelope { .. } => {
            emit_diagnostic(verb, trace, trace_span, err, fmt, output);
        }
    }
}

fn emit_rank_scope_error(
    verb: &str,
    trace: Option<&Path>,
    trace_span: Option<TraceSpan>,
    err: &PytorchQueryError,
    fmt: OutputFormat,
    output: &mut SourceExecution,
) {
    use veloq_core::VeloqDiagnostic;
    // Source the message, code, and hint from the typed error's
    // VeloqDiagnostic impl so this enriched envelope cannot drift from
    // the generic `emit_diagnostic` path for the same variant.
    let message = err.to_string();
    let trace_arg = trace
        .map(|path| shell_quote(&path.display().to_string()))
        .unwrap_or_else(|| "<trace>".to_string());
    let mut env = EnvelopeError::new(
        Some(PytorchSource::source_ref()),
        Some(format!("{}.{}", PytorchSource::KIND, verb)),
        trace.map(PytorchSource::trace_ref),
        trace_span,
        Some(ResponseMeta {
            next_steps: vec![
                NextStep {
                    hint: "Aggregate intentionally across every PyTorch rank.".to_string(),
                    command: format!("veloq pytorch {verb} {trace_arg} --all-ranks"),
                },
                NextStep {
                    hint: "Inspect one PyTorch rank before comparing peers.".to_string(),
                    command: format!("veloq pytorch {verb} {trace_arg} --rank 0"),
                },
            ],
            warnings: vec![Warning {
                severity: WarningSeverity::Warn,
                code: WarningCode::MultiRankAmbiguous,
                message: message.clone(),
            }],
            ..ResponseMeta::default()
        }),
        message.clone(),
        Vec::new(),
    );
    env.error.code = Some(err.code());
    env.error.hint = err.hint().map(|hint| hint.into_owned());
    if !matches!(fmt, OutputFormat::Json) {
        output.write_stderr_line(format!("veloq: {message}"));
    }
    if let Ok(s) = env.to_json_pretty() {
        output.write_stdout_line(s);
    }
}

fn emit_diagnostic<E>(
    verb: &str,
    trace: Option<&Path>,
    trace_span: Option<TraceSpan>,
    err: &E,
    fmt: OutputFormat,
    output: &mut SourceExecution,
) where
    E: veloq_core::VeloqDiagnostic,
{
    let envelope = EnvelopeError::from_diagnostic(
        Some(PytorchSource::source_ref()),
        Some(format!("{}.{verb}", PytorchSource::KIND)),
        trace.map(PytorchSource::trace_ref),
        trace_span,
        err,
    );
    if !matches!(fmt, OutputFormat::Json) {
        output.write_stderr_line(format!("veloq: {err}"));
    }
    if let Ok(rendered) = envelope.to_json_pretty() {
        output.write_stdout_line(rendered);
    }
}
