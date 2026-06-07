//! Time literal parsing for CLI args.
//!
//! Accepted single durations: `42ns`, `100us`, `1500ms`, `1.2s`.
//! Accepted ranges: `<dur>-<dur>` (e.g. `1.2s-1.5s`).
//!
//! Ranges are interpreted as **offsets from trace start**. Callers add
//! the trace's `start_ns` before issuing SQL.

use std::num::ParseFloatError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TimeParseError {
    #[error("empty time literal")]
    Empty,
    #[error("unknown time unit in `{0}` (expected ns/us/ms/s)")]
    UnknownUnit(String),
    #[error("invalid number in `{lit}`: {source}")]
    BadNumber {
        lit: String,
        #[source]
        source: ParseFloatError,
    },
    #[error("range must be `start-end`, got `{0}`")]
    BadRange(String),
    #[error("range end ({end} ns) must be greater than start ({start} ns)")]
    EmptyRange { start: i64, end: i64 },
    #[error("time literal `{0}` overflows i64 nanoseconds")]
    Overflow(String),
}

/// Parse a single duration literal into nanoseconds.
///
/// Examples: `"42ns"` → 42, `"1.2s"` → 1_200_000_000.
pub fn parse_duration_ns(s: &str) -> Result<i64, TimeParseError> {
    let s = s.trim();
    if s.is_empty() {
        return Err(TimeParseError::Empty);
    }

    // The number part is the leading run of digits / decimal point,
    // plus optionally a single leading sign. A `-` / `+` after the
    // first character is a unit-side glyph at best and a malformed
    // input at worst — let f64::parse reject it. (Previously we
    // silently consumed inline signs into the number portion, which
    // let `1-2ms` mis-split into ("1-2", "ms").)
    let mut split = 0usize;
    for (i, c) in s.char_indices() {
        let in_number = c.is_ascii_digit() || c == '.' || ((c == '-' || c == '+') && i == 0);
        if in_number {
            split = i + c.len_utf8();
        } else {
            break;
        }
    }
    let (num_part, unit_part) = s.split_at(split);
    let unit = unit_part.trim().to_ascii_lowercase();

    let value: f64 = num_part
        .parse()
        .map_err(|e: ParseFloatError| TimeParseError::BadNumber {
            lit: s.to_string(),
            source: e,
        })?;
    if !value.is_finite() {
        return Err(TimeParseError::Overflow(s.to_string()));
    }

    let multiplier_ns: f64 = match unit.as_str() {
        "ns" | "" => 1.0,
        "us" | "µs" => 1_000.0,
        "ms" => 1_000_000.0,
        "s" => 1_000_000_000.0,
        _ => return Err(TimeParseError::UnknownUnit(unit_part.to_string())),
    };

    // Explicit overflow check: `as i64` would silently saturate at
    // i64::MAX (or 0 for NaN, already caught above), producing nonsense
    // windows. Reject anything outside the representable range.
    let ns = value * multiplier_ns;
    if !ns.is_finite() || ns > i64::MAX as f64 || ns < i64::MIN as f64 {
        return Err(TimeParseError::Overflow(s.to_string()));
    }
    Ok(ns as i64)
}

/// Endpoint of a `TimeWindow`. Two anchors:
/// - `Relative(ns)` — offset from the trace's primary origin (default
///   for bare literals like `1.2s`).
/// - `Absolute(ns)` — absolute ns (literal prefixed with `@`, e.g.
///   `@-185s`). Useful when the agent has read raw timestamps from
///   `summary`'s `per_table` and wants to address them directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimePoint {
    Relative(i64),
    Absolute(i64),
}

impl TimePoint {
    pub fn resolve(&self, origin_ns: i64) -> i64 {
        match *self {
            TimePoint::Relative(off) => origin_ns + off,
            TimePoint::Absolute(abs) => abs,
        }
    }

    pub fn parse(s: &str) -> Result<Self, TimeParseError> {
        let s = s.trim();
        if let Some(rest) = s.strip_prefix('@') {
            Ok(TimePoint::Absolute(parse_duration_ns(rest)?))
        } else {
            Ok(TimePoint::Relative(parse_duration_ns(s)?))
        }
    }
}

/// Half-open time window. Endpoints are anchored as `TimePoint`s and
/// resolved to absolute ns at query time.
#[derive(Debug, Clone, Copy)]
pub struct TimeWindow {
    pub start: TimePoint,
    pub end: TimePoint,
}

impl TimeWindow {
    /// Parse a `start-end` literal. Each endpoint independently honours
    /// the `@` prefix, so you can mix anchors (e.g. `@0-1s` = from
    /// absolute 0 to 1s after primary origin) if you really need to.
    pub fn parse(s: &str) -> Result<Self, TimeParseError> {
        let s = s.trim();
        // Find the dash that separates the two endpoints. Skip a
        // possible leading `@` and a possible numeric sign so a literal
        // like `@-185s-@-180s` still parses.
        let mut search_start = 0usize;
        let bytes = s.as_bytes();
        if bytes.first() == Some(&b'@') {
            search_start = 1;
        }
        if matches!(bytes.get(search_start), Some(b'-' | b'+')) {
            search_start += 1;
        }
        let dash = s[search_start..]
            .find('-')
            .map(|i| i + search_start)
            .ok_or_else(|| TimeParseError::BadRange(s.to_string()))?;
        let (lhs, rhs_with_dash) = s.split_at(dash);
        let rhs = &rhs_with_dash[1..];

        let start = TimePoint::parse(lhs)?;
        let end = TimePoint::parse(rhs)?;
        Ok(Self { start, end })
    }

    /// Resolve to absolute ns by anchoring `Relative` endpoints against
    /// `primary_origin_ns`. Returns `(start_ns, end_ns)`.
    pub fn absolute(&self, primary_origin_ns: i64) -> Result<(i64, i64), TimeParseError> {
        let s = self.start.resolve(primary_origin_ns);
        let e = self.end.resolve(primary_origin_ns);
        if e <= s {
            return Err(TimeParseError::EmptyRange { start: s, end: e });
        }
        Ok((s, e))
    }
}

/// Duration-on-an-event predicate, parsed from a CLI string like
/// `">1ms"` or `"100us-1ms"`. Used by `search --duration`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurationFilter {
    /// `> threshold_ns`
    Gt(i64),
    /// `>= threshold_ns`
    Gte(i64),
    /// `< threshold_ns`
    Lt(i64),
    /// `<= threshold_ns`
    Lte(i64),
    /// `min_ns ..= max_ns`
    Range { min_ns: i64, max_ns: i64 },
}

impl DurationFilter {
    /// Render as a SQL fragment in terms of `duration_col`. The fragment
    /// is parameterised (uses `?`); call `.sql_params()` to get the bind
    /// list in the matching order.
    pub fn sql(&self, duration_col: &str) -> String {
        match self {
            DurationFilter::Gt(_) => format!("{duration_col} > ?"),
            DurationFilter::Gte(_) => format!("{duration_col} >= ?"),
            DurationFilter::Lt(_) => format!("{duration_col} < ?"),
            DurationFilter::Lte(_) => format!("{duration_col} <= ?"),
            DurationFilter::Range { .. } => {
                format!("{duration_col} BETWEEN ? AND ?")
            }
        }
    }

    pub fn sql_params(&self) -> Vec<i64> {
        match *self {
            DurationFilter::Gt(n) => vec![n],
            DurationFilter::Gte(n) => vec![n],
            DurationFilter::Lt(n) => vec![n],
            DurationFilter::Lte(n) => vec![n],
            DurationFilter::Range { min_ns, max_ns } => vec![min_ns, max_ns],
        }
    }

    pub fn matches(&self, duration_ns: i64) -> bool {
        match *self {
            DurationFilter::Gt(ns) => duration_ns > ns,
            DurationFilter::Gte(ns) => duration_ns >= ns,
            DurationFilter::Lt(ns) => duration_ns < ns,
            DurationFilter::Lte(ns) => duration_ns <= ns,
            DurationFilter::Range { min_ns, max_ns } => {
                duration_ns >= min_ns && duration_ns <= max_ns
            }
        }
    }

    pub fn parse(s: &str) -> Result<Self, TimeParseError> {
        let s = s.trim();
        if s.is_empty() {
            return Err(TimeParseError::Empty);
        }

        // Range form: `<dur>-<dur>` (and not starting with `>` or `<`).
        // Skip past the first codepoint when searching for the separator
        // dash — the first char could be a leading numeric sign and
        // won't itself be the separator. Char-aware indexing keeps
        // multibyte input (e.g. `µs`) from byte-slicing into the middle
        // of a codepoint and panicking before the JSON error envelope.
        if !s.starts_with('<') && !s.starts_with('>') {
            let after_first = s.char_indices().nth(1).map(|(i, _)| i).unwrap_or(s.len());
            if let Some(dash) = s[after_first..].find('-').map(|i| i + after_first) {
                let (lhs, rhs) = s.split_at(dash);
                let rhs = &rhs[1..]; // skip the dash itself (ASCII, 1 byte)
                let min_ns = parse_duration_ns(lhs)?;
                let max_ns = parse_duration_ns(rhs)?;
                if max_ns < min_ns {
                    return Err(TimeParseError::EmptyRange {
                        start: min_ns,
                        end: max_ns,
                    });
                }
                return Ok(DurationFilter::Range { min_ns, max_ns });
            }
            // Bare literal with no operator is ambiguous (equal? at least?).
            // Force the user to be explicit.
            return Err(TimeParseError::BadRange(format!(
                "{s} — duration filter needs a comparator (`>`, `>=`, `<`, `<=`) or a range (`min-max`)"
            )));
        }

        // Comparator form. Local typed enum drops the stringly-typed
        // `op` round-trip and makes the final `match` exhaustive without
        // an unreachable arm.
        enum Op {
            Gt,
            Ge,
            Lt,
            Le,
        }
        let (op, rest) = if let Some(rest) = s.strip_prefix(">=") {
            (Op::Ge, rest)
        } else if let Some(rest) = s.strip_prefix("<=") {
            (Op::Le, rest)
        } else if let Some(rest) = s.strip_prefix('>') {
            (Op::Gt, rest)
        } else if let Some(rest) = s.strip_prefix('<') {
            (Op::Lt, rest)
        } else {
            return Err(TimeParseError::BadRange(s.to_string()));
        };
        let n = parse_duration_ns(rest.trim())?;
        Ok(match op {
            Op::Gt => DurationFilter::Gt(n),
            Op::Ge => DurationFilter::Gte(n),
            Op::Lt => DurationFilter::Lt(n),
            Op::Le => DurationFilter::Lte(n),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations() -> anyhow::Result<()> {
        assert_eq!(parse_duration_ns("42ns")?, 42);
        assert_eq!(parse_duration_ns("100us")?, 100_000);
        assert_eq!(parse_duration_ns("1.5ms")?, 1_500_000);
        assert_eq!(parse_duration_ns("1.2s")?, 1_200_000_000);
        Ok(())
    }

    #[test]
    fn ranges() -> anyhow::Result<()> {
        let w = TimeWindow::parse("1.2s-1.5s")?;
        assert_eq!(w.start, TimePoint::Relative(1_200_000_000));
        assert_eq!(w.end, TimePoint::Relative(1_500_000_000));
        let (a, b) = w.absolute(186_198_374)?;
        assert_eq!(a, 1_386_198_374);
        assert_eq!(b, 1_686_198_374);
        Ok(())
    }

    #[test]
    fn absolute_endpoints() -> anyhow::Result<()> {
        let w = TimeWindow::parse("@-185s-@-180s")?;
        assert_eq!(w.start, TimePoint::Absolute(-185_000_000_000));
        assert_eq!(w.end, TimePoint::Absolute(-180_000_000_000));
        // origin is ignored for absolute endpoints
        let (a, b) = w.absolute(123)?;
        assert_eq!((a, b), (-185_000_000_000, -180_000_000_000));
        Ok(())
    }

    #[test]
    fn mixed_anchors() -> anyhow::Result<()> {
        let w = TimeWindow::parse("@-1s-2s")?;
        assert_eq!(w.start, TimePoint::Absolute(-1_000_000_000));
        assert_eq!(w.end, TimePoint::Relative(2_000_000_000));
        let (a, b) = w.absolute(5_000_000_000)?;
        assert_eq!((a, b), (-1_000_000_000, 7_000_000_000));
        Ok(())
    }

    #[test]
    fn bad_unit() {
        assert!(matches!(
            parse_duration_ns("5min"),
            Err(TimeParseError::UnknownUnit(_))
        ));
    }

    #[test]
    fn overflow_rejected() {
        // `9999999999999s` * 1e9 ns/s > i64::MAX
        assert!(matches!(
            parse_duration_ns("9999999999999s"),
            Err(TimeParseError::Overflow(_))
        ));
        // Scientific notation isn't supported by the lexer (the number
        // part stops at `e`), so this lands as UnknownUnit rather than
        // Overflow — fine either way, just not a number we'd silently
        // saturate.
        assert!(parse_duration_ns("1e30s").is_err());
    }

    #[test]
    fn inline_signs_rejected() {
        // Earlier impl silently split `1-2ms` as ("1-2", "ms"); now `-`
        // mid-literal stays with the unit half and forces BadNumber/Unit.
        assert!(parse_duration_ns("1-2ms").is_err());
    }

    #[test]
    fn multibyte_input_doesnt_panic() {
        // Regression: `µ` is two bytes, so the old byte-indexed `s[1..]`
        // lookup in `DurationFilter::parse` would split into the middle
        // of the codepoint and panic. Now it should fall through to a
        // structured error.
        assert!(DurationFilter::parse("µs").is_err());
        assert!(DurationFilter::parse("µ-1ms").is_err());
    }

    #[test]
    fn empty_range_rejected_at_resolve() -> anyhow::Result<()> {
        // The parser is structural — it builds endpoints without
        // touching the trace origin. The empty/inverted check moved to
        // `.absolute()` so absolute endpoints (which only get a true
        // ordering once resolved) are checked uniformly.
        let w = TimeWindow::parse("1s-1s")?;
        assert!(matches!(
            w.absolute(0),
            Err(TimeParseError::EmptyRange { .. })
        ));
        Ok(())
    }

    #[test]
    fn duration_filter_comparators() -> anyhow::Result<()> {
        assert_eq!(
            DurationFilter::parse(">1ms")?,
            DurationFilter::Gt(1_000_000)
        );
        assert_eq!(
            DurationFilter::parse(">=100us")?,
            DurationFilter::Gte(100_000)
        );
        assert_eq!(
            DurationFilter::parse("<1s")?,
            DurationFilter::Lt(1_000_000_000)
        );
        assert_eq!(DurationFilter::parse("<=42ns")?, DurationFilter::Lte(42));
        Ok(())
    }

    #[test]
    fn duration_filter_range() -> anyhow::Result<()> {
        assert_eq!(
            DurationFilter::parse("100us-1ms")?,
            DurationFilter::Range {
                min_ns: 100_000,
                max_ns: 1_000_000
            }
        );
        Ok(())
    }

    #[test]
    fn duration_filter_matches_typed_values() {
        assert!(DurationFilter::Gt(10).matches(11));
        assert!(!DurationFilter::Gt(10).matches(10));
        assert!(DurationFilter::Gte(10).matches(10));
        assert!(DurationFilter::Lt(10).matches(9));
        assert!(!DurationFilter::Lt(10).matches(10));
        assert!(DurationFilter::Lte(10).matches(10));
        assert!(
            DurationFilter::Range {
                min_ns: 10,
                max_ns: 20
            }
            .matches(15)
        );
    }

    #[test]
    fn duration_filter_bare_literal_rejected() {
        assert!(DurationFilter::parse("1ms").is_err());
    }
}
