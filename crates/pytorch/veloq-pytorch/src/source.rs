use crate::PytorchSourceError;
use crate::cli::Cmd;
use crate::commands;
use clap::{ArgMatches, Command, FromArgMatches, Subcommand};
use std::path::Path;
use veloq_core::{
    EnvelopeTraceRef, OutputFormat, ProfileSource, SourceRef, SourceRunResult, TraceSpan,
    write_diagnostic_error_envelope,
};

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
            .about("PyTorch Kineto/Profiler trace query verbs")
            .subcommand_required(true)
            .arg_required_else_help(true);
        Cmd::augment_subcommands(parent)
    }

    fn run(&self, matches: &ArgMatches, fmt: OutputFormat) -> SourceRunResult<i32> {
        let cmd = Cmd::from_arg_matches(matches)?;
        let verb = cmd.name();
        let trace_path = cmd.trace_path().map(Path::to_path_buf);
        match commands::run(cmd, trace_path.as_deref(), fmt) {
            Ok(code) => Ok(code),
            Err(err) => {
                let span = trace_path
                    .as_deref()
                    .and_then(|path| self.compute_trace_span(path));
                emit_err(verb, trace_path.as_deref(), span, &err, fmt);
                Ok(1)
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
) {
    match err {
        PytorchSourceError::Command(err) => emit_diagnostic(verb, trace, trace_span, err, fmt),
        PytorchSourceError::Data(err) => emit_diagnostic(verb, trace, trace_span, err, fmt),
        PytorchSourceError::Query(err) => emit_diagnostic(verb, trace, trace_span, err, fmt),
        PytorchSourceError::Tabular(err) => emit_diagnostic(verb, trace, trace_span, err, fmt),
        PytorchSourceError::SerializeEnvelope { .. } => {
            emit_diagnostic(verb, trace, trace_span, err, fmt);
        }
    }
}

fn emit_diagnostic<E>(
    verb: &str,
    trace: Option<&Path>,
    trace_span: Option<TraceSpan>,
    err: &E,
    fmt: OutputFormat,
) where
    E: veloq_core::VeloqDiagnostic,
{
    write_diagnostic_error_envelope(
        PytorchSource::source_ref(),
        verb,
        trace.map(PytorchSource::trace_ref),
        trace_span,
        err,
        fmt,
    );
}
