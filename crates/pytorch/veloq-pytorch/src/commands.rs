//! Per-command dispatch for the PyTorch source.
//!
//! The source entry point owns identity and clap integration; this module
//! translates typed CLI variants into query requests and renders responses.

use crate::cli::{Cmd, CommonArgs, EventArgs, ScopeArgs};
use crate::schema::{SchemaPayload, schema_value_for};
use crate::source::PytorchSource;
use crate::views::emit_tabular;
use anyhow::{Context, Result};
use std::path::Path;
use veloq_core::{Envelope, OutputFormat, TraceSpan};
use veloq_pytorch_query::{EventFilterRequest, RankScope};

pub(crate) fn run(cmd: Cmd, trace_path: Option<&Path>, fmt: OutputFormat) -> Result<i32> {
    let verb = cmd.name();
    let qualified = format!("{}.{verb}", PytorchSource::KIND);

    if let Cmd::Schema { target } = &cmd {
        if fmt != OutputFormat::Json {
            anyhow::bail!(
                "veloq-pytorch schema currently supports only --format json (got `{fmt}`)"
            );
        }
        let schema = schema_value_for(target)?;
        veloq_core::emit_envelope(
            PytorchSource::source_ref(),
            qualified,
            None,
            None,
            None,
            SchemaPayload {
                target: target.clone(),
                schema,
            },
        )?;
        return Ok(0);
    }

    let trace_path =
        trace_path.ok_or_else(|| anyhow::anyhow!("internal: pytorch verb missing trace path"))?;

    dispatch_trace_command(cmd, &qualified, trace_path, fmt)
        .with_context(|| format!("running pytorch {verb}"))
}

fn dispatch_trace_command(
    cmd: Cmd,
    qualified: &str,
    trace_path: &Path,
    fmt: OutputFormat,
) -> Result<i32> {
    match cmd {
        Cmd::Summary { .. } => {
            let trace = veloq_pytorch_data::build_or_load(trace_path)?;
            let response = veloq_pytorch_query::summary(&trace);
            emit_response(
                qualified,
                trace_path,
                trace.envelope_trace_span(),
                fmt,
                response,
            )
        }
        Cmd::Search { filters, .. } => {
            let trace = veloq_pytorch_data::build_or_load(trace_path)?;
            let request = event_request(&trace, &filters, 100)?;
            let response = veloq_pytorch_query::search(&trace, request)?;
            emit_response(
                qualified,
                trace_path,
                trace.envelope_trace_span(),
                fmt,
                response,
            )
        }
        Cmd::Inspect { row_ids, .. } => {
            let trace = veloq_pytorch_data::build_or_load(trace_path)?;
            let response = veloq_pytorch_query::inspect(&trace, &row_ids);
            emit_response(
                qualified,
                trace_path,
                trace.envelope_trace_span(),
                fmt,
                response,
            )
        }
        Cmd::Stats {
            group_by, filters, ..
        } => {
            let trace = veloq_pytorch_data::build_or_load(trace_path)?;
            let request = event_request(&trace, &filters, 50)?;
            let axes = veloq_pytorch_query::parse_group_by(&group_by);
            let response = veloq_pytorch_query::stats(&trace, request, &axes)?;
            emit_response(
                qualified,
                trace_path,
                trace.envelope_trace_span(),
                fmt,
                response,
            )
        }
        Cmd::Correlate { row_ids, .. } => {
            let trace = veloq_pytorch_data::build_or_load(trace_path)?;
            let response = veloq_pytorch_query::correlate(&trace, &row_ids);
            emit_response(
                qualified,
                trace_path,
                trace.envelope_trace_span(),
                fmt,
                response,
            )
        }
        Cmd::Timeline {
            interval, filters, ..
        } => {
            let trace = veloq_pytorch_data::build_or_load(trace_path)?;
            let request = event_request(&trace, &filters, 1000)?;
            let interval_ns = veloq_core::time::parse_duration_ns(&interval)
                .with_context(|| format!("invalid --interval `{interval}`"))?;
            let response = veloq_pytorch_query::timeline(&trace, request, interval_ns)?;
            emit_response(
                qualified,
                trace_path,
                trace.envelope_trace_span(),
                fmt,
                response,
            )
        }
        Cmd::Slices {
            name,
            name_regex,
            aggregate,
            group_by,
            scope,
            common,
            ..
        } => {
            let trace = veloq_pytorch_data::build_or_load(trace_path)?;
            let request = slice_request(&trace, name, name_regex, scope, common)?;
            let group_by = if aggregate { Some(group_by) } else { None };
            let response = veloq_pytorch_query::slices(&trace, request, aggregate, group_by)?;
            emit_response(
                qualified,
                trace_path,
                trace.envelope_trace_span(),
                fmt,
                response,
            )
        }
        Cmd::Collectives { step, limit, .. } => {
            veloq_pytorch_query::ensure_trace_dir(trace_path)?;
            let limit = checked_limit(Some(limit), 100)?;
            let trace = veloq_pytorch_data::build_or_load(trace_path)?;
            let response = veloq_pytorch_query::collectives(
                &trace,
                RankScope {
                    rank: None,
                    all_ranks: true,
                },
                step,
                limit,
            );
            emit_response(
                qualified,
                trace_path,
                trace.envelope_trace_span(),
                fmt,
                response,
            )
        }
        Cmd::Prep { status, .. } => {
            if !status {
                let _ = veloq_pytorch_data::build_or_load(trace_path)?;
            }
            let state = veloq_pytorch_data::prep_state(trace_path)?;
            let response = veloq_pytorch_query::prep_response(state, !status);
            let span = veloq_pytorch_data::trace_span_for_path(trace_path);
            emit_response(qualified, trace_path, span, fmt, response)
        }
        Cmd::Schema { .. } => {
            anyhow::bail!("internal: schema should have returned before trace dispatch")
        }
    }
}

fn event_request(
    trace: &veloq_pytorch_data::TraceSet,
    args: &EventArgs,
    default_limit: usize,
) -> Result<EventFilterRequest> {
    Ok(EventFilterRequest {
        types: veloq_pytorch_query::parse_type_selection(&args.types)?,
        name_glob: args.name.clone(),
        name_regex: args.name_regex.clone(),
        duration: args
            .duration
            .as_deref()
            .map(veloq_core::time::DurationFilter::parse)
            .transpose()
            .context("invalid --duration")?,
        time_window_ns: veloq_pytorch_query::resolve_time_window(
            trace,
            args.common.from.as_deref(),
            args.common.to.as_deref(),
        )?,
        rank_scope: rank_scope(args.scope),
        device: args.scope.device,
        stream: args.scope.stream,
        step: args.scope.step,
        is_comm: args.is_comm,
        limit: checked_limit(args.common.limit, default_limit)?,
    })
}

fn slice_request(
    trace: &veloq_pytorch_data::TraceSet,
    name: Option<String>,
    name_regex: Option<String>,
    scope: ScopeArgs,
    common: CommonArgs,
) -> Result<EventFilterRequest> {
    Ok(EventFilterRequest {
        types: veloq_pytorch_query::parse_type_selection("step,annotation")?,
        name_glob: name,
        name_regex,
        duration: None,
        time_window_ns: veloq_pytorch_query::resolve_time_window(
            trace,
            common.from.as_deref(),
            common.to.as_deref(),
        )?,
        rank_scope: rank_scope(scope),
        device: scope.device,
        stream: scope.stream,
        step: scope.step,
        is_comm: false,
        limit: checked_limit(common.limit, 100)?,
    })
}

fn rank_scope(scope: ScopeArgs) -> RankScope {
    RankScope {
        rank: scope.rank,
        all_ranks: scope.all_ranks,
    }
}

fn checked_limit(limit: Option<usize>, default_limit: usize) -> Result<usize> {
    let limit = limit.unwrap_or(default_limit);
    if limit == 0 {
        anyhow::bail!(
            "--limit must be at least 1 (limit=0 would suppress total_matched too); use `--limit 1` for one row + totals"
        );
    }
    Ok(limit)
}

fn emit_response<T: serde::Serialize>(
    qualified: &str,
    trace: &Path,
    trace_span: Option<TraceSpan>,
    fmt: OutputFormat,
    response: T,
) -> Result<i32> {
    let trace_str = trace.display().to_string();
    match fmt {
        OutputFormat::Json => {
            let env = Envelope::new(
                PytorchSource::source_ref(),
                qualified.to_string(),
                Some(PytorchSource::trace_ref(trace)),
                trace_span,
                None,
                response,
            );
            println!("{}", env.to_json_pretty()?);
        }
        OutputFormat::Csv | OutputFormat::Table => {
            emit_tabular(&response, qualified, &trace_str, fmt)?;
        }
    }
    Ok(0)
}
