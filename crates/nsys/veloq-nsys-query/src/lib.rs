//! veloq-nsys-query — per-subcommand query implementations.
//!
//! Each subcommand owns one module here. Phase 0 ships `summary`;
//! `stats`, `search`, `inspect`, `timeline`, `gaps`, `correlate` follow.

pub mod column_map;
pub mod concurrency;
pub mod correlate;
pub mod docgen;
pub mod error;
pub mod event_ref;
pub mod gaps;
pub mod graph_replays;
pub mod hardware;
pub mod inspect;
pub mod kind_filter;
pub mod kind_policy;
pub mod kind_sql;
pub mod metrics;
pub mod ncu_command;
pub mod nvtx_attribution;
pub mod nvtx_parent;
pub mod nvtx_projection;
pub mod nvtx_reverse;
mod query_sql;
pub mod row_id;
pub mod search;
pub mod slices;
pub mod stats;
pub mod stats_by_size;
pub mod summary;
pub mod timeline;
pub mod viz_timeline;

pub use error::{NsysQueryError, NsysQueryResult, SqlPhase};
pub use event_ref::{EventRef, NvtxContext};
pub use kind_filter::KindFilter;
pub use row_id::{EventKind, RowId};

/// Reject `limit == 0` at the public-API boundary. The CLI also
/// guards via `CommonFilters::limit_or`, but library callers can
/// hand-build a request with `limit: 0`, which silently zeroes
/// `total_matched` (the count comes off SQL rows that LIMIT 0
/// suppressed). Call this at the top of every `run()`.
pub fn check_limit(limit: usize) -> NsysQueryResult<()> {
    veloq_core::LimitRef::new(limit)
        .map(|_| ())
        .map_err(|_| NsysQueryError::LimitTooSmall { limit })
}

/// Shared verb preamble: validate the limit, open the trace, and resolve
/// the `--from/--to` window to an absolute `(start_ns, end_ns)`. Used by
/// the verbs whose `run()` opens with exactly this sequence (`stats` /
/// `search` / `stats_by_size`). Verbs that interleave other validation
/// between these steps — `gaps`' `--min` check, `timeline`'s `--interval`
/// check, `slices`' deferred window resolution — keep their own preamble
/// so error precedence is unchanged.
pub fn open_scoped(
    path: &std::path::Path,
    limit: usize,
    window: Option<veloq_core::time::TimeWindow>,
) -> NsysQueryResult<(veloq_nsys_data::Trace, Option<(i64, i64)>)> {
    check_limit(limit)?;
    let trace = veloq_nsys_data::Trace::open(path).map_err(NsysQueryError::trace_open)?;
    let abs_window = trace
        .resolve_window(window)
        .map_err(NsysQueryError::time_window_resolve)?;
    Ok((trace, abs_window))
}

/// NSys records modules as absolute paths
/// (`/usr/lib/x86_64-linux-gnu/libc.so.6`) or Windows-style
/// (`C:\Windows\system32\foo.dll`). For hotspot tables / callchains
/// agents (and humans) want the basename — `libc.so.6` /
/// `foo.dll`. Centralised here so the `metrics --type cpu-sampling`
/// path and `inspect cpu_sample:N` agree on what "module name"
/// means without two copies of the slice-on-`/` logic drifting.
pub fn module_basename(path: &str) -> String {
    path.rsplit(['/', '\\']).next().unwrap_or("").to_string()
}

/// Decode an nsys `globalTid` into `(pid, tid)`. NSys packs four
/// fields into the 64-bit slot:
///
/// | bits 48-63 | bits 24-47 | bits 16-23      | bits 0-15  |
/// | HW/Host ID | Native PID | Source Domain   | Native TID |
///
/// **TID is 16 bits, not 24.** The middle 8 bits carry the source-
/// domain id (`0x00` = OSRT tracer, `0x3B` = CUDA driver, …); using
/// `>> 24` for PID extraction (instead of `>> 16`) skips that byte
/// so the same PID lands consistently whether you're reading
/// `PROCESSES.globalPid` (OSRT) or `ThreadNames.globalTid` (CUDA).
/// A naive `>> 16` would land a domain-shifted "pid" that disagrees
/// across tables by a constant offset.
///
/// Centralised here so future call sites have one place to update if
/// the layout ever shifts; both `metrics --type cpu-sampling` and
/// `inspect cpu_sample:N` go through this helper.
pub fn decode_global_tid(global_tid: i64) -> (i64, i64) {
    let pid = (global_tid >> 24) & 0xFFFFFF;
    let tid = global_tid & 0xFFFF;
    (pid, tid)
}

/// Parse a CLI duration flag (`100us` / `1.2s` / `42ns` / …) into ns,
/// rejecting non-positive results. Wraps
/// [`veloq_core::time::parse_duration_ns`] with a flag-name aware
/// typed error and a "must be positive" guard. Used by every
/// command that accepts a bucket/interval-like duration flag.
///
/// `gaps::parse_min_duration` intentionally does *not* go through
/// this — `--min-duration 0ns` (the default) means "no minimum",
/// which is a meaningful filter even though it isn't positive.
pub fn parse_positive_duration(s: &str, flag: &str) -> NsysQueryResult<i64> {
    let ns = veloq_core::time::parse_duration_ns(s).map_err(|source| {
        NsysQueryError::PositiveDurationInvalid {
            flag: flag.to_string(),
            value: s.to_string(),
            source,
        }
    })?;
    if ns <= 0 {
        return Err(NsysQueryError::PositiveDurationTooSmall {
            flag: flag.to_string(),
            ns,
        });
    }
    Ok(ns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use veloq_core::VeloqDiagnostic;

    #[test]
    fn check_limit_zero_returns_typed_error() -> anyhow::Result<()> {
        let err = match check_limit(0) {
            Ok(()) => anyhow::bail!("expected check_limit(0) to fail"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.limit-too-small");
        assert!(matches!(err, NsysQueryError::LimitTooSmall { limit: 0 }));
        Ok(())
    }

    #[test]
    fn parse_positive_duration_invalid_literal_returns_typed_error() -> anyhow::Result<()> {
        let err = match parse_positive_duration("bogus", "--bucket") {
            Ok(ns) => anyhow::bail!("expected invalid duration to fail, got {ns} ns"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.invalid-positive-duration");
        match err {
            NsysQueryError::PositiveDurationInvalid { flag, value, .. } => {
                assert_eq!(flag, "--bucket");
                assert_eq!(value, "bogus");
            }
            other => anyhow::bail!("expected PositiveDurationInvalid, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn parse_positive_duration_zero_returns_typed_error() -> anyhow::Result<()> {
        let err = match parse_positive_duration("0ns", "--interval") {
            Ok(ns) => anyhow::bail!("expected zero duration to fail, got {ns} ns"),
            Err(err) => err,
        };

        assert_eq!(
            err.code().as_str(),
            "nsys.query.positive-duration-too-small"
        );
        assert!(matches!(
            err,
            NsysQueryError::PositiveDurationTooSmall {
                flag,
                ns: 0
            } if flag == "--interval"
        ));
        Ok(())
    }
}
