//! `veloq` — agent-friendly profile-query CLI.
//!
//! Thin dispatch shell. The binary owns:
//!
//! - the top-level clap parser (the global `--format` flag);
//! - the registry of `ProfileSource` impls (today: NSys, hoisted to
//!   the top level because it's the configured default and also
//!   available under `veloq nsys …`; NCU under `veloq ncu …`;
//!   PyTorch/Kineto under `veloq pytorch …`);
//! - dispatch from the parsed `ArgMatches` to the matching source's
//!   shared execution boundary.
//!
//! Everything else — per-verb arg parsing, query execution, envelope
//! emit, CSV/table rendering, error envelope writing — lives in the
//! source's own crate.

mod daemon;
mod error;
mod meta;

use clap::{Arg, ArgMatches, Command};
use error::{CliError, CliResult};
use std::sync::Arc;
use veloq_core::{EnvelopeError, OutputFormat, ProfileSource, SourceExecution, VeloqDiagnostic};
use veloq_ncu::NcuSource;
use veloq_nsys::NsysSource;
use veloq_pytorch::PytorchSource;

// mimalloc for the binary only — veloq's hot paths (DuckDB/Arrow
// materialization, JSON serialization, rayon NVTX grouping) are
// allocation-heavy. Scoped here so the libraries stay allocator-agnostic.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// The configured default source. Its subcommands are hoisted to the
/// top level so users can keep typing `veloq stats <trace>` without
/// the `nsys` namespace prefix. Every non-default source contributes
/// a `veloq <kind> <verb>` namespace.
const DEFAULT_SOURCE: &str = daemon::DEFAULT_SOURCE;

fn main() {
    let sources = registered_sources();

    let parser = build_parser(&sources);
    let matches = match parser.try_get_matches() {
        Ok(m) => m,
        Err(e) => {
            if raw_stdout_parse_error_mode() {
                let _ = e.print();
                std::process::exit(e.exit_code());
            }
            // Parse errors fire before clap has resolved global args.
            // Best-effort scan argv for an explicit `--format` so
            // human-targeted parse failures still get the stderr mirror.
            // Logger isn't initialized yet — irrelevant; clap doesn't log.
            veloq_nsys::output::emit_parse_error(&e, parse_error_output_format());
            std::process::exit(e.exit_code());
        }
    };

    let default_format = String::from("json");
    let fmt_str: &String = matches
        .get_one::<String>("format")
        .unwrap_or(&default_format);
    let fmt = match OutputFormat::parse(fmt_str) {
        Ok(f) => f,
        Err(err) => {
            // Same as parse errors: `--format` itself is bad, fall
            // back to the documented default for stderr policy.
            emit_cli_diagnostic_error(&err, OutputFormat::Json);
            std::process::exit(1);
        }
    };

    // Now we know `fmt` — initialize the logger. In JSON mode (the
    // agent contract, also the default), drop the default filter to
    // `error` so library `log::warn!` calls don't duplicate the
    // failure chain that already lives in the stdout envelope
    // (dual-channel policy: see the error-envelope helpers). A legitimate
    // hard error still surfaces; everything chattier requires explicit
    // `RUST_LOG=…` opt-in. CSV / table users keep the chatty filter
    // (warn + per-crate info) so first-time-on-a-trace runs show
    // Parquet build progress for humans.
    let default_filter = if matches!(fmt, OutputFormat::Json) {
        "error"
    } else {
        "warn,veloq_core=info,veloq_nsys_data=info,veloq_nsys_query=info,veloq_nsys=info,\
         veloq_pytorch_data=info,veloq_pytorch_query=info,veloq_pytorch=info,veloq=info"
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default_filter))
        .init();

    let code = match dispatch(&sources, &matches, fmt) {
        Ok(c) => c,
        Err(err) => {
            // Top-level dispatch failure — no source-level envelope
            // was written. Emit a CLI-level error envelope (without
            // source/verb/trace context) so agents still have one
            // JSON document to parse.
            emit_cli_diagnostic_error(&err, fmt);
            1
        }
    };
    std::process::exit(code);
}

/// The registry is the single source of truth for what verbs the CLI
/// offers. Each source contributes its own `clap::Command` subtree;
/// we graft them under the top-level parser below. NSys is the
/// configured default (hoisted to top level) and still has an
/// explicit `veloq nsys ...` namespace; NCU gets `veloq ncu ...` and
/// PyTorch gets `veloq pytorch ...`.
fn registered_sources() -> Vec<Arc<dyn ProfileSource>> {
    vec![
        Arc::new(NsysSource),
        Arc::new(NcuSource),
        Arc::new(PytorchSource),
    ]
}

fn emit_cli_diagnostic_error<E>(err: &E, fmt: OutputFormat)
where
    E: VeloqDiagnostic,
{
    let execution = render_cli_diagnostic_execution(err, fmt);
    let _ = execution.write_to_process();
}

pub(crate) fn render_cli_diagnostic_execution<E>(err: &E, fmt: OutputFormat) -> SourceExecution
where
    E: VeloqDiagnostic,
{
    let env = EnvelopeError::from_diagnostic(None, None, None, None, err);
    let mut execution = SourceExecution::new();
    execution.set_exit_code(1);
    if !matches!(fmt, OutputFormat::Json) {
        execution.write_stderr_line(format!("veloq: {err}"));
    }
    if let Ok(s) = env.to_json_pretty() {
        execution.write_stdout_line(s);
    }
    execution
}

/// `veloq nsys ncu-command --print` is explicitly pipe-oriented:
/// stdout must contain only the generated shell script. Clap errors
/// happen before source dispatch, so detect that raw mode from argv
/// here and let clap print only its native stderr usage.
fn raw_stdout_parse_error_mode() -> bool {
    let mut saw_print = false;
    let mut command_path = Vec::new();
    let mut skip_next = false;
    for arg in std::env::args_os().skip(1) {
        if skip_next {
            skip_next = false;
            continue;
        }
        if arg == "--print" {
            saw_print = true;
            continue;
        }
        if arg == "--format" {
            skip_next = true;
            continue;
        }
        let arg = arg.to_string_lossy();
        if arg.starts_with("--format=") || arg.starts_with('-') {
            continue;
        }
        command_path.push(arg.into_owned());
    }
    if !saw_print {
        return false;
    }
    matches!(
        command_path.as_slice(),
        [cmd, ..] if cmd == "ncu-command"
    ) || matches!(
        command_path.as_slice(),
        [source, cmd, ..] if source == "nsys" && cmd == "ncu-command"
    )
}

fn parse_error_output_format() -> OutputFormat {
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--format" {
            if let Some(value) = args.next().and_then(|value| value.into_string().ok())
                && let Ok(fmt) = OutputFormat::parse(&value)
            {
                return fmt;
            }
            continue;
        }
        let arg = arg.to_string_lossy();
        if let Some(value) = arg.strip_prefix("--format=")
            && let Ok(fmt) = OutputFormat::parse(value)
        {
            return fmt;
        }
    }
    OutputFormat::Json
}

/// Build the top-level `clap::Command` by composing the global
/// `--format` flag with each source's contributed subcommand tree.
/// The configured default source's verbs are hoisted to the top
/// level so they don't require a namespace prefix; every source also
/// ends up under `veloq <kind> <verb>`.
fn build_parser(sources: &[Arc<dyn ProfileSource>]) -> Command {
    let mut root = Command::new("veloq")
        .version(env!("CARGO_PKG_VERSION"))
        .about("Agent-friendly profile-query CLI (JSON on stdout)")
        .arg(
            Arg::new("format")
                .long("format")
                .global(true)
                .default_value("json")
                .help(
                    "Output format. JSON is the default agent contract; \
                     csv / table flatten the response's primary list with envelope \
                     metadata in header comments.",
                ),
        )
        .subcommand_required(true)
        .arg_required_else_help(true);

    root = daemon::graft_source_commands(root, sources);

    // Meta verbs (`veloq info`, `veloq sources`) sit at the top
    // level alongside the hoisted default-source verbs. Adding a
    // meta verb means one entry in `meta::cli()`.
    for meta_cmd in meta::cli() {
        root = root.subcommand(meta_cmd);
    }
    root = root.subcommand(daemon::cli());

    root
}

/// Route the parsed `ArgMatches` through the matching source's shared
/// execution boundary, then project the result onto one-shot process I/O.
/// For source namespaces we expect `veloq <kind> <verb>` (two levels
/// deep). For the default source's hoisted aliases we look up the
/// verb name as the subcommand at the root.
fn dispatch(
    sources: &[Arc<dyn ProfileSource>],
    matches: &ArgMatches,
    fmt: OutputFormat,
) -> CliResult<i32> {
    let (sub_name, sub_matches) = matches.subcommand().ok_or(CliError::NoSubcommand)?;

    if sub_name == "daemon" {
        return daemon::run(sub_matches, fmt, sources).map_err(CliError::daemon);
    }

    // Meta verbs come first — they're owned by the binary, not by
    // any profile source. Success responses are JSON-only; `--format`
    // still controls whether handled errors get a stderr mirror.
    if meta::is_meta(sub_name) {
        return meta::run(sub_name, sub_matches, sources, fmt).map_err(CliError::from);
    }

    // Source namespace: `veloq <kind> <verb>` (two levels deep).
    for source in sources {
        if source.kind() == sub_name {
            let execution = daemon::routing::execute_selected(source.as_ref(), sub_matches, fmt)
                .map_err(CliError::source_run)?;
            return project_source_execution(execution);
        }
    }

    // Otherwise the subcommand is a default-source verb (hoisted to
    // the top level). Find the default source and let it run.
    let default = sources.iter().find(|s| s.kind() == DEFAULT_SOURCE).ok_or(
        CliError::DefaultSourceNotRegistered {
            kind: DEFAULT_SOURCE,
        },
    )?;
    let execution = daemon::routing::execute_selected(default.as_ref(), matches, fmt)
        .map_err(CliError::source_run)?;
    project_source_execution(execution)
}

fn project_source_execution(execution: SourceExecution) -> CliResult<i32> {
    let exit_code = execution.exit_code();
    execution
        .write_to_process()
        .map_err(CliError::source_output)?;
    Ok(exit_code)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::path::Path;

    #[test]
    fn registered_source_kinds_and_namespaces_are_unique() {
        let sources = registered_sources();
        let mut kinds = BTreeSet::new();
        let mut namespaces = BTreeSet::new();

        for source in &sources {
            let kind = source.kind();
            assert!(kinds.insert(kind), "duplicate source kind `{kind}`");

            let cli = source.cli();
            let namespace = cli.get_name();
            assert_eq!(
                namespace, kind,
                "source namespace `{namespace}` must match kind `{kind}`"
            );
            assert!(
                namespaces.insert(namespace.to_string()),
                "duplicate source namespace `{namespace}`"
            );
        }

        assert!(
            kinds.contains(DEFAULT_SOURCE),
            "default source `{DEFAULT_SOURCE}` is not registered"
        );
    }

    #[test]
    fn automatic_detection_claims_representative_inputs_once() {
        let sources = registered_sources();
        for (path, expected) in [
            ("trace.nsys-rep", Some("nsys")),
            ("trace_pqtdir", Some("nsys")),
            ("report.ncu-rep", Some("ncu")),
            ("report.ncu-repz", Some("ncu")),
            ("worker0.pt.trace.json", Some("pytorch")),
            ("worker0.pt.trace.json.gz", Some("pytorch")),
            ("trace.sqlite", None),
            ("trace.json", None),
        ] {
            let claims: Vec<&str> = sources
                .iter()
                .filter(|source| source.detect(Path::new(path)))
                .map(|source| source.kind())
                .collect();
            assert!(
                claims.len() <= 1,
                "automatic detection overlap for `{path}`: {claims:?}"
            );
            assert_eq!(
                claims.first().copied(),
                expected,
                "unexpected automatic detection claim for `{path}`"
            );
        }
    }
}
