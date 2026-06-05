//! veloq-nsys-query — per-subcommand query implementations.
//!
//! Each subcommand owns one module here. Phase 0 ships `summary`;
//! `stats`, `search`, `inspect`, `timeline`, `gaps`, `correlate` follow.

pub mod column_map;
pub mod concurrency;
pub mod correlate;
pub mod docgen;
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
pub mod row_id;
pub mod search;
pub mod slices;
pub mod stats;
pub mod stats_by_size;
pub mod summary;
pub mod timeline;

pub use event_ref::{EventRef, NvtxContext};
pub use kind_filter::KindFilter;
pub use row_id::{EventKind, RowId};

/// Reject `limit == 0` at the public-API boundary. The CLI also
/// guards via `CommonFilters::limit_or`, but library callers can
/// hand-build a request with `limit: 0`, which silently zeroes
/// `total_matched` (the count comes off SQL rows that LIMIT 0
/// suppressed). Call this at the top of every `run()`.
pub fn check_limit(limit: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        limit > 0,
        "limit must be at least 1 (limit=0 suppresses every row including the \
         total_matched / scope totals carried on them)"
    );
    Ok(())
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
) -> anyhow::Result<(veloq_nsys_data::Trace, Option<(i64, i64)>)> {
    check_limit(limit)?;
    let trace = veloq_nsys_data::Trace::open(path)?;
    let abs_window = trace.resolve_window(window)?;
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
/// context message and a "must be positive" guard. Used by every
/// command that accepts a bucket/interval-like duration flag.
///
/// `gaps::parse_min_duration` intentionally does *not* go through
/// this — `--min-duration 0ns` (the default) means "no minimum",
/// which is a meaningful filter even though it isn't positive.
pub fn parse_positive_duration(s: &str, flag: &str) -> anyhow::Result<i64> {
    use anyhow::Context;
    let ns =
        veloq_core::time::parse_duration_ns(s).with_context(|| format!("invalid {flag} `{s}`"))?;
    anyhow::ensure!(ns > 0, "{flag} must be positive (got {ns} ns)");
    Ok(ns)
}

/// Coerce a slice of `duckdb::types::Value` into the slice-of-trait-objects
/// shape that `duckdb::Statement::query` requires for positional binding.
///
/// The result must outlive the `query` call (the returned `Rows` holds the
/// `&dyn ToSql` references), so bind it to a local rather than passing it
/// inline as a temporary.
pub fn bind(params: &[duckdb::types::Value]) -> Vec<&dyn duckdb::ToSql> {
    params.iter().map(|v| v as &dyn duckdb::ToSql).collect()
}

/// Convert shell-style `*`/`?` wildcards to SQL `LIKE` patterns, escaping
/// the SQL-special `%`/`_`/`\` chars in the input so they're literal.
/// Used by every command that takes a `--name` / `--pattern` glob.
///
/// Pairs with [`sql_like_match`] for the same pattern grammar applied
/// in Rust — the two MUST agree, so they live side-by-side here.
pub fn search_glob_to_like(glob: &str) -> String {
    let mut out = String::with_capacity(glob.len());
    for ch in glob.chars() {
        match ch {
            '*' => out.push('%'),
            '?' => out.push('_'),
            '%' | '_' | '\\' => {
                out.push('\\');
                out.push(ch);
            }
            other => out.push(other),
        }
    }
    out
}

/// Apply the same `%` / `_` / `\<lit>` LIKE pattern grammar in Rust.
/// Used by commands that need to post-filter rows in memory (e.g.
/// when SQL-side filtering would force per-axis SQL divergence in
/// `metrics --type cpu-sampling`). Input pattern comes from
/// [`search_glob_to_like`] so the grammar matches 1:1.
///
/// Supports `%` (zero-or-more), `_` (exactly-one), and `\` as a
/// literal-next escape. Anything not listed is matched as a literal.
pub fn sql_like_match(s: &str, pattern: &str) -> bool {
    let s_bytes: Vec<char> = s.chars().collect();
    let p_bytes: Vec<char> = pattern.chars().collect();
    fn rec(s: &[char], p: &[char]) -> bool {
        if p.is_empty() {
            return s.is_empty();
        }
        let (head, tail) = match p.split_first() {
            Some((h, t)) => (*h, t),
            None => return s.is_empty(),
        };
        match head {
            '%' => {
                // empty match here…
                if rec(s, tail) {
                    return true;
                }
                // …or consume one char and try again.
                match s.split_first() {
                    Some((_, srest)) => rec(srest, p),
                    None => false,
                }
            }
            '_' => match s.split_first() {
                Some((_, srest)) => rec(srest, tail),
                None => false,
            },
            '\\' => match (tail.split_first(), s.split_first()) {
                (Some((lit, prest)), Some((sc, srest))) if *lit == *sc => rec(srest, prest),
                _ => false,
            },
            other => match s.split_first() {
                Some((sc, srest)) if *sc == other => rec(srest, tail),
                _ => false,
            },
        }
    }
    rec(&s_bytes, &p_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip the glob through [`search_glob_to_like`] and assert
    /// the Rust matcher agrees with the expected result. Catches drift
    /// between the two halves of the LIKE pipeline: a future edit to
    /// either function that changes the `*`/`?`/`%`/`_`/`\` grammar
    /// will fail here before it can quietly produce wrong filter
    /// results in `metrics --type cpu-sampling`'s post-query path.
    #[test]
    fn glob_to_like_round_trips_through_sql_like_match() {
        // (glob, candidate, expected). Cover: literal, `*`, `?`,
        // anchored start/end, LIKE-special chars (`%`/`_`) escaped
        // through the conversion, and the `\<lit>` escape branch.
        let cases: &[(&str, &str, bool)] = &[
            ("foo", "foo", true),
            ("foo", "foox", false),
            ("foo*", "foo", true),
            ("foo*", "foobar", true),
            ("foo*", "bar", false),
            ("*foo", "barfoo", true),
            ("*foo*", "abarfooz", true),
            ("f?o", "foo", true),
            ("f?o", "fxxo", false),
            // `*` / `?` are *only* meaningful as wildcards; embedded
            // SQL-special `%` / `_` from the user input must round-trip
            // as literals.
            ("100%", "100%", true),
            ("100%", "100x", false),
            ("a_b", "a_b", true),
            ("a_b", "axb", false),
            // Backslash in the user input becomes an escaped backslash
            // in the LIKE pattern; the matcher must consume it the same way.
            ("a\\b", "a\\b", true),
        ];
        for (glob, candidate, expected) in cases {
            let like = search_glob_to_like(glob);
            assert_eq!(
                sql_like_match(candidate, &like),
                *expected,
                "glob=`{glob}` -> like=`{like}` vs candidate=`{candidate}`"
            );
        }
    }
}
