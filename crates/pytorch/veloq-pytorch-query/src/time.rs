use anyhow::{Context, Result};
use std::path::Path;
use veloq_pytorch_data::TraceSet;

pub fn resolve_time_window(
    trace: &TraceSet,
    from: Option<&str>,
    to: Option<&str>,
) -> Result<Option<(i64, i64)>> {
    match (from, to) {
        (None, None) => Ok(None),
        (Some(_), None) | (None, Some(_)) => {
            anyhow::bail!("`--from` and `--to` must be set together (got only one)")
        }
        (Some(from), Some(to)) => {
            let origin = trace.trace_span.map(|span| span.start_ns).unwrap_or(0);
            let start = veloq_core::time::TimePoint::parse(from)
                .with_context(|| format!("invalid --from `{from}`"))?
                .resolve(origin);
            let end = veloq_core::time::TimePoint::parse(to)
                .with_context(|| format!("invalid --to `{to}`"))?
                .resolve(origin);
            if end <= start {
                anyhow::bail!("time window end ({end} ns) must be greater than start ({start} ns)");
            }
            Ok(Some((start, end)))
        }
    }
}

pub fn parse_group_by(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|axis| !axis.is_empty())
        .map(ToString::to_string)
        .collect()
}

pub fn ensure_trace_dir(path: &Path) -> Result<()> {
    if !path.is_dir() {
        anyhow::bail!("pytorch collectives requires a trace directory input");
    }
    Ok(())
}
