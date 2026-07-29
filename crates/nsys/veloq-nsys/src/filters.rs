//! Shared CLI argument groups used by multiple subcommands.
//!
//! Every subcommand-specific arg stays on its own variant; the ones
//! that *are* genuinely shared end up here so the parsing helpers live
//! in one place and adding a new subcommand is `#[command(flatten)]`.

use clap::Args;
use std::path::PathBuf;
use veloq_core::time::{TimePoint, TimeWindow};
use veloq_nsys_query::{EventKind, KindFilter};

use crate::error::NsysSourceError;

/// The trace-path positional, lifted out so every subcommand can
/// `#[command(flatten)]` it. Keeping the parsing rule + docstring in
/// one place means a future input-shape change
/// touches one file.
#[derive(Args, Debug, Clone)]
pub struct TraceArg {
    /// Path to an NSys `.nsys-rep` file or pre-exported `_pqtdir/`
    /// parquet directory. Positional, required.
    pub trace: PathBuf,
}

/// Args shared by every time-windowed command — `--from`/`--to`
/// pair plus row capping.
///
/// Conventional CLI dialect: `--from` and `--to` separately rather
/// than a single `--time-range A-B`. The pair plays well with shell
/// composition (one flag per arg) and matches how the rest of the
/// observability ecosystem talks about time windows
/// (Prometheus, kubectl logs, gcloud logging).
///
/// Endpoints accept time literals (`1.2s`, `100ms`, `100us`, `42ns`)
/// anchored to the trace's primary origin, or absolute ns prefixed
/// with `@` (`@-185s` selects ns -185_000_000_000).
#[derive(Args, Debug, Clone, Default)]
pub struct CommonFilters {
    /// Start of the time window. Pair with `--to`; setting only one
    /// is an error.
    #[arg(long, value_name = "TIME")]
    pub from: Option<String>,

    /// End of the time window. Pair with `--from`.
    #[arg(long, value_name = "TIME")]
    pub to: Option<String>,

    /// Max rows to return. Default differs per command (50 for stats,
    /// 100 for search/slices/gaps, 1000 for timeline).
    #[arg(long)]
    pub limit: Option<usize>,
}

impl CommonFilters {
    /// Combine `--from` / `--to` into a [`TimeWindow`]. Returns `Ok(None)`
    /// when neither is set; errors when exactly one is set, or when
    /// either endpoint is malformed. The caller absolutises via
    /// `veloq_nsys_data::Trace::resolve_window`.
    pub fn time_window(&self) -> crate::error::NsysSourceResult<Option<TimeWindow>> {
        match (self.from.as_deref(), self.to.as_deref()) {
            (None, None) => Ok(None),
            (Some(_), None) | (None, Some(_)) => Err(NsysSourceError::MissingTimeBound),
            (Some(from), Some(to)) => {
                let start = TimePoint::parse(from)
                    .map_err(|source| NsysSourceError::invalid_from(from, source))?;
                let end = TimePoint::parse(to)
                    .map_err(|source| NsysSourceError::invalid_to(to, source))?;
                Ok(Some(TimeWindow { start, end }))
            }
        }
    }

    /// Resolve `--limit` against the caller's default, rejecting `0`.
    /// A zero limit silently breaks the `total_matched` / scope-totals
    /// fields (those are read off rows the SQL never produces), so we
    /// surface it as a CLI error rather than letting bogus totals
    /// escape into the JSON envelope.
    pub fn limit_or(&self, default: usize) -> crate::error::NsysSourceResult<usize> {
        let n = self.limit.unwrap_or(default);
        if n == 0 {
            return Err(NsysSourceError::limit_too_small(n));
        }
        Ok(n)
    }
}

/// Uniform scope filters shared by list verbs that aggregate or list GPU
/// events (`stats`, `search`, `slices`, `gaps`, `timeline`).
/// `--device <N>` scopes **both** axes — `deviceId` on GPU
/// events AND the native_pids that ran on device N on host-thread
/// events — via the existing `TARGET_INFO_CUDA_CONTEXT_INFO` bridge.
/// `--all-devices` opts back into the cross-device aggregate when the
/// silent-sum behavior is genuinely what the caller wants. Most commands
/// refuse multi-device traces with neither flag set; trace-wide command
/// modes can explicitly opt into an implicit all-device scope.
#[derive(Args, Debug, Clone, Default)]
pub struct GpuLocationFilters {
    /// Restrict to one native OS process. Required with `--device`
    /// when rank-private CUDA namespaces reuse the same logical ordinal.
    #[arg(long = "process", value_name = "PID")]
    pub process: Option<i64>,

    /// Restrict to one CUDA device (NSys `deviceId`). On multi-device
    /// traces, most commands require this unless `--all-devices` is
    /// set; mutually exclusive with `--all-devices`. Also scopes
    /// host-thread events (NVTX ranges, runtime API calls, slices' CPU
    /// bounds) to the native_pid(s) that ran on this device.
    #[arg(long, value_name = "DEV_ID", conflicts_with = "all_devices")]
    pub device: Option<i32>,

    /// Restrict to one CUDA stream (NSys `streamId`). Requires a
    /// single resolved device; on multi-device traces pass `--device`
    /// as well.
    #[arg(long, value_name = "STREAM_ID")]
    pub stream: Option<i64>,

    /// Opt into the cross-device aggregate on multi-device traces.
    /// Mutually exclusive with `--device`. Some trace-wide command
    /// modes imply this when no device is selected.
    #[arg(long = "all-devices", default_value_t = false)]
    pub all_devices: bool,
}

/// Device-only scope filters for list verbs that aggregate across
/// streams by construction. Most commands use the same multi-device
/// ambiguity policy as [`GpuLocationFilters`], but commands with a
/// natural per-device output can imply all-device scope. This group
/// intentionally does not expose `--stream`.
#[derive(Args, Debug, Clone, Default)]
pub struct DeviceLocationFilters {
    /// Restrict to one native OS process. Required with `--device`
    /// when rank-private CUDA namespaces reuse the same logical ordinal.
    #[arg(long = "process", value_name = "PID")]
    pub process: Option<i64>,

    /// Restrict to one CUDA device (NSys `deviceId`). On multi-device
    /// traces, strict commands require this unless `--all-devices` is
    /// set; mutually exclusive with `--all-devices`.
    #[arg(long, value_name = "DEV_ID", conflicts_with = "all_devices")]
    pub device: Option<i32>,

    /// Opt into the cross-device aggregate on multi-device traces.
    /// Mutually exclusive with `--device`. Commands with a natural
    /// per-device output can imply this when no device is selected.
    #[arg(long = "all-devices", default_value_t = false)]
    pub all_devices: bool,
}

/// Args shared by `stats` and `search` — event-kind selection and
/// NVTX-scope filter. (slices uses `--name` directly, since its
/// pattern is required and defines the whole query.)
#[derive(Args, Debug, Clone, Default)]
pub struct GpuFilters {
    /// Comma-separated event kinds, or `all`. Allowed values depend on
    /// the subcommand: `stats` accepts `kernel`, `memcpy`, `memset`,
    /// `sync`, `runtime`, `osrt`, `graph`, `nvtx`; `search` accepts the
    /// same set. `--type nvtx` rejects `--group-by device|context|
    /// stream|graph|graph_node` and `--nvtx` (NVTX has no device axis).
    #[arg(long = "type", default_value = "all")]
    pub kinds: String,

    /// Restrict to GPU events causally attributed to NVTX ranges
    /// matching this glob (`*`/`?`). Same correlation walk as `slices`.
    /// See AGENTS.md → "NVTX projection" for the model.
    #[arg(long)]
    pub nvtx: Option<String>,
}

impl GpuFilters {
    /// Parse `--type` against the caller's allow-list.
    /// - `"all"` (the default) returns [`KindFilter::All`] — the
    ///   downstream request resolves `All` against its own allow-list
    ///   (stats uses `ALLOWED_KINDS`, search uses `EventKind::ALL`).
    /// - Explicit token lists return [`KindFilter::Only`]; tokens not
    ///   in `allowed` are rejected with a friendly message.
    pub fn kinds(&self, allowed: &[EventKind]) -> crate::error::NsysSourceResult<KindFilter> {
        let raw = self.kinds.trim();
        if raw.eq_ignore_ascii_case("all") {
            return Ok(KindFilter::All);
        }
        let mut out: Vec<EventKind> = Vec::new();
        for tok in raw.split(',') {
            let tok = tok.trim();
            if tok.is_empty() {
                continue;
            }
            let k =
                EventKind::parse(tok).ok_or_else(|| NsysSourceError::unknown_event_kind(tok))?;
            if !allowed.contains(&k) {
                let names: Vec<&str> = allowed.iter().map(|k| k.as_str()).collect();
                return Err(NsysSourceError::event_kind_not_allowed(
                    tok,
                    names.join(", "),
                ));
            }
            if !out.contains(&k) {
                out.push(k);
            }
        }
        if out.is_empty() {
            return Err(NsysSourceError::EmptyEventKindList);
        }
        Ok(KindFilter::Only(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    fn gpu(kinds: &str) -> GpuFilters {
        GpuFilters {
            kinds: kinds.to_string(),
            nvtx: None,
        }
    }

    #[test]
    fn kinds_all_returns_filter_all() -> Result<()> {
        let allowed = [EventKind::Kernel, EventKind::Memcpy, EventKind::Memset];
        assert_eq!(gpu("all").kinds(&allowed)?, KindFilter::All);
        Ok(())
    }

    #[test]
    fn kinds_csv_parses_and_dedupes() -> Result<()> {
        let allowed = EventKind::ALL;
        let got = gpu("kernel, memcpy, kernel").kinds(allowed)?;
        assert_eq!(
            got,
            KindFilter::Only(vec![EventKind::Kernel, EventKind::Memcpy])
        );
        Ok(())
    }

    #[test]
    fn kinds_rejects_disallowed_for_subset() {
        let allowed = [EventKind::Kernel, EventKind::Memcpy, EventKind::Memset];
        let result = gpu("runtime").kinds(&allowed);
        let s = match result {
            Ok(_) => String::from("(no error)"),
            Err(e) => e.to_string(),
        };
        assert!(s.contains("not allowed here"), "got: {s}");
        assert!(s.contains("kernel"), "got: {s}");
    }

    #[test]
    fn kinds_rejects_unknown_token() {
        let allowed = EventKind::ALL;
        let result = gpu("bogus").kinds(allowed);
        let s = match result {
            Ok(_) => String::from("(no error)"),
            Err(e) => e.to_string(),
        };
        assert!(s.contains("unknown event kind"), "got: {s}");
    }

    #[test]
    fn common_limit_or_uses_default_when_unset() -> Result<()> {
        let c = CommonFilters::default();
        assert_eq!(c.limit_or(50)?, 50);
        let c2 = CommonFilters {
            limit: Some(7),
            ..Default::default()
        };
        assert_eq!(c2.limit_or(50)?, 7);
        Ok(())
    }

    #[test]
    fn common_limit_or_rejects_zero() {
        let c = CommonFilters {
            limit: Some(0),
            ..Default::default()
        };
        assert!(c.limit_or(50).is_err());
    }

    #[test]
    fn time_window_requires_both_endpoints() {
        let c = CommonFilters {
            from: Some("1s".into()),
            ..Default::default()
        };
        assert!(c.time_window().is_err(), "missing --to must error");
        let c = CommonFilters {
            to: Some("2s".into()),
            ..Default::default()
        };
        assert!(c.time_window().is_err(), "missing --from must error");
    }

    #[test]
    fn time_window_parses_both_endpoints() -> Result<()> {
        let c = CommonFilters {
            from: Some("1s".into()),
            to: Some("2s".into()),
            ..Default::default()
        };
        let w = c
            .time_window()?
            .ok_or_else(|| anyhow::anyhow!("expected Some(TimeWindow)"))?;
        let (a, b) = w.absolute(0)?;
        assert_eq!((a, b), (1_000_000_000, 2_000_000_000));
        Ok(())
    }
}
