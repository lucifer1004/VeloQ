//! Shared `--sort` parsing utility.
//!
//! Every subcommand that exposes `--sort` consumes the same string
//! syntax. A `--sort` value is a comma-separated list of fields; each
//! field is `key[:asc|:desc]` (or `-key` / `+key` shorthand). The set
//! of valid keys differs per command, so each command parses the
//! shared `SortSpec` and then maps each field's `key` string onto its
//! own typed enum.
//!
//! Accepted forms (single field):
//! - `total`              — key only, command's default direction
//! - `total:desc`         — explicit direction
//! - `total:asc`
//! - `-total`             — shorthand for `key:desc`
//! - `+name`              — shorthand for `key:asc`
//!
//! Multi-field (tiebreakers, left-to-right):
//! - `total,name`         — total then name (each uses its key's default dir)
//! - `-total,+count`      — total DESC, count ASC
//!
//! Whitespace around each field and the direction token is trimmed.

use thiserror::Error;

/// Ascending vs descending. Each command's per-key default is applied
/// when the caller didn't say (e.g., `total` → `Desc`, `name` → `Asc`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Asc,
    Desc,
}

impl Direction {
    /// SQL keyword form, for splicing into `ORDER BY` clauses.
    pub fn sql(self) -> &'static str {
        match self {
            Direction::Asc => "ASC",
            Direction::Desc => "DESC",
        }
    }

    /// Orient a forward (ascending) comparison per this direction:
    /// `Asc` keeps it, `Desc` reverses it.
    pub fn apply(self, ord: std::cmp::Ordering) -> std::cmp::Ordering {
        match self {
            Direction::Asc => ord,
            Direction::Desc => ord.reverse(),
        }
    }
}

/// Multi-key in-memory sort shared by the summary sorters (NSys metrics
/// gpu/nic/cpu_sched/cpu_sampling; NCU source-metrics / warp-stalls
/// rows). For each `(key, direction)` in order, compare with `cmp` and
/// return the first non-`Equal` result oriented by the direction; fall
/// back to `tiebreak`. Centralizes the spec-loop + `Asc/Desc` reverse +
/// tiebreak fallthrough each site used to hand-roll — callers supply
/// only the per-key comparison.
pub fn sort_in_memory<T, K>(
    rows: &mut [T],
    keys: &[(K, Direction)],
    cmp: impl Fn(&K, &T, &T) -> std::cmp::Ordering,
    tiebreak: impl Fn(&T, &T) -> std::cmp::Ordering,
) {
    rows.sort_by(|a, b| {
        for (k, dir) in keys {
            let ord = cmp(k, a, b);
            if ord != std::cmp::Ordering::Equal {
                return dir.apply(ord);
            }
        }
        tiebreak(a, b)
    });
}

/// One parsed atom of a `--sort` value: the `key` plus an optional
/// caller-supplied direction. The `key` is lowercased and trimmed;
/// the direction is `None` when the caller didn't override.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortField {
    pub key: String,
    pub direction: Option<Direction>,
}

impl SortField {
    /// Parse a single field. Multi-field strings should go through
    /// `SortSpec::parse` instead — it splits on commas and calls this.
    pub fn parse(raw: &str) -> Result<Self, SortParseError> {
        let raw = raw.trim();
        if raw.is_empty() {
            return Err(SortParseError::EmptyField);
        }
        // `-key` / `+key` shorthand binds direction first.
        if let Some(rest) = raw.strip_prefix('-') {
            return Self::new(rest.trim(), Some(Direction::Desc));
        }
        if let Some(rest) = raw.strip_prefix('+') {
            return Self::new(rest.trim(), Some(Direction::Asc));
        }
        if let Some((k, d)) = raw.split_once(':') {
            let dir = match d.trim().to_ascii_lowercase().as_str() {
                "asc" | "ascending" | "" => Direction::Asc,
                "desc" | "descending" => Direction::Desc,
                other => return Err(SortParseError::BadDirection(other.to_string())),
            };
            return Self::new(k.trim(), Some(dir));
        }
        Self::new(raw, None)
    }

    fn new(key: &str, direction: Option<Direction>) -> Result<Self, SortParseError> {
        if key.is_empty() {
            return Err(SortParseError::EmptyField);
        }
        Ok(Self {
            key: key.to_ascii_lowercase(),
            direction,
        })
    }

    /// Resolve `direction`, picking the per-key default when the caller
    /// didn't explicitly set one.
    pub fn direction_or(&self, default: Direction) -> Direction {
        self.direction.unwrap_or(default)
    }
}

/// The whole `--sort` value: an ordered list of `SortField`s.
///
/// Empty inputs are rejected at parse time so commands can trust that
/// a non-empty list reached them. CLI defaults supply at least one
/// field (e.g., stats's default is `"total"`), so empty input from
/// users is the only way to hit `EmptySpec`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortSpec(pub Vec<SortField>);

impl SortSpec {
    pub fn parse(raw: &str) -> Result<Self, SortParseError> {
        let raw = raw.trim();
        if raw.is_empty() {
            return Err(SortParseError::EmptySpec);
        }
        let mut fields = Vec::new();
        for part in raw.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            fields.push(SortField::parse(part)?);
        }
        if fields.is_empty() {
            return Err(SortParseError::EmptySpec);
        }
        Ok(Self(fields))
    }

    pub fn fields(&self) -> &[SortField] {
        &self.0
    }

    /// Hand-construct a one-field spec from a static key, without going
    /// through `parse`. Used by per-command defaults where parsing a
    /// constant would otherwise need `.unwrap()` / `.expect()` on
    /// known-good input.
    pub fn single(key: &str) -> Self {
        Self(vec![SortField {
            key: key.to_string(),
            direction: None,
        }])
    }
}

#[derive(Debug, Error)]
pub enum SortParseError {
    #[error("--sort spec is empty")]
    EmptySpec,
    #[error("--sort field is empty (check for stray commas)")]
    EmptyField,
    #[error("unknown sort direction `{0}` (expected `asc` or `desc`)")]
    BadDirection(String),
    #[error("unknown --sort key `{key}` (expected: {expected})")]
    UnknownKey { key: String, expected: String },
}

/// One row in a per-command sort-key table. Names the variant the row
/// resolves to, its canonical key string, any user-typed aliases, and
/// the direction applied when the caller didn't say `:asc`/`:desc`.
///
/// `canonical` is what `--help` advertises and what error messages
/// list. `aliases` covers ergonomics (`total_ns` for `total`, `dur` for
/// `duration`) — they parse identically.
#[derive(Debug, Clone, Copy)]
pub struct SortKeySpec<T: 'static> {
    pub variant: T,
    pub canonical: &'static str,
    pub aliases: &'static [&'static str],
    pub default_dir: Direction,
}

/// Single source of truth for a command's `--sort` vocabulary.
///
/// Each per-command `SortKey` enum implements this with a static table
/// listing every accepted key, its aliases, and its default direction.
/// [`Self::from_field`] and [`Self::help_text`] are derived from that table, so the
/// parser, the `--help` block, and the error message can never drift.
///
/// The trait deliberately exposes only the table — sorting on rows
/// (whether SQL `ORDER BY` text or in-memory comparators) stays in each
/// command's module because the column mapping is command-specific.
pub trait SortKeyDef: Sized + Copy + 'static {
    /// Per-key table. Order matters: it controls the `--help` listing
    /// order and the order canonical names appear in error messages.
    fn specs() -> &'static [SortKeySpec<Self>];

    /// Resolve a parsed `--sort` field into the typed variant + the
    /// caller-or-default direction. Returns
    /// [`SortParseError::UnknownKey`] when the user typed a key the
    /// table doesn't know; the error names every canonical key so
    /// agents reading the message don't need `--help`.
    fn from_field(field: &SortField) -> Result<(Self, Direction), SortParseError> {
        for spec in Self::specs() {
            if spec.canonical == field.key || spec.aliases.iter().any(|a| *a == field.key) {
                return Ok((spec.variant, field.direction_or(spec.default_dir)));
            }
        }
        let expected: Vec<&'static str> = Self::specs().iter().map(|s| s.canonical).collect();
        Err(SortParseError::UnknownKey {
            key: field.key.clone(),
            expected: expected.join(", "),
        })
    }

    /// One-line, comma-separated rendering for the `--help` block.
    /// Every key gets `(DIRECTION)`; the first row also carries
    /// `default` so an agent reading help knows what `--sort key`
    /// without `:asc`/`:desc` resolves to.
    fn help_text() -> String {
        let mut out = String::new();
        for (i, spec) in Self::specs().iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(spec.canonical);
            if i == 0 {
                out.push_str(" (default ");
                out.push_str(spec.default_dir.sql());
                out.push(')');
            } else {
                out.push_str(" (");
                out.push_str(spec.default_dir.sql());
                out.push(')');
            }
        }
        out
    }
}

/// Build a SQL `ORDER BY ...` body (the text after `ORDER BY`) from a
/// resolved sequence of `(column_expr, direction)` pairs plus an
/// optional deterministic tiebreaker column.
///
/// The tiebreaker inherits the direction of the last user-specified
/// field — so a user-facing DESC sort doesn't break visual
/// consistency by silently falling back to ASC on tied rows. Pass an
/// empty `tiebreaker_column` to skip the tiebreaker entirely.
pub fn build_order_by(parts: &[(&str, Direction)], tiebreaker_column: &str) -> String {
    let mut out: Vec<String> = parts
        .iter()
        .map(|(col, dir)| format!("{col} {}", dir.sql()))
        .collect();
    if !tiebreaker_column.is_empty() {
        let tail_dir = parts.last().map(|(_, d)| *d).unwrap_or(Direction::Asc);
        out.push(format!("{tiebreaker_column} {}", tail_dir.sql()));
    }
    out.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sort_in_memory_orders_by_key_then_tiebreak() {
        // (group, val, id): sort by val DESC, tiebreak id ASC.
        let mut rows = vec![(1, 5, 30), (1, 5, 10), (2, 9, 20)];
        sort_in_memory(
            &mut rows,
            &[("val", Direction::Desc)],
            |k, a, b| match *k {
                "val" => a.1.cmp(&b.1),
                _ => std::cmp::Ordering::Equal,
            },
            |a, b| a.2.cmp(&b.2),
        );
        assert_eq!(rows, vec![(2, 9, 20), (1, 5, 10), (1, 5, 30)]);
    }

    #[test]
    fn sort_in_memory_multi_key_asc() {
        let mut rows = vec![(1, 5), (2, 5), (1, 3)];
        sort_in_memory(
            &mut rows,
            &[("g", Direction::Asc), ("v", Direction::Asc)],
            |k, a, b| match *k {
                "g" => a.0.cmp(&b.0),
                "v" => a.1.cmp(&b.1),
                _ => std::cmp::Ordering::Equal,
            },
            |_, _| std::cmp::Ordering::Equal,
        );
        assert_eq!(rows, vec![(1, 3), (1, 5), (2, 5)]);
    }

    // Helpers keep the no-panic policy honest: `.get(i)` + `?` instead
    // of `[i]` / `.unwrap()` on the tested fields, so a regression in
    // the parser would fail the test rather than panic and skip the
    // assertion that follows.
    fn field_at(spec: &SortSpec, i: usize) -> anyhow::Result<&SortField> {
        spec.0
            .get(i)
            .ok_or_else(|| anyhow::anyhow!("sort spec has no field at index {i}"))
    }

    #[test]
    fn single_bare_key_has_no_direction() -> anyhow::Result<()> {
        let s = SortSpec::parse("total")?;
        assert_eq!(s.0.len(), 1);
        let f0 = field_at(&s, 0)?;
        assert_eq!(f0.key, "total");
        assert!(f0.direction.is_none());
        Ok(())
    }

    #[test]
    fn colon_direction_parses() -> anyhow::Result<()> {
        let s = SortSpec::parse("name:desc")?;
        assert_eq!(field_at(&s, 0)?.direction, Some(Direction::Desc));
        let s = SortSpec::parse("Count:ASC")?;
        let f0 = field_at(&s, 0)?;
        assert_eq!(f0.key, "count"); // normalised
        assert_eq!(f0.direction, Some(Direction::Asc));
        Ok(())
    }

    #[test]
    fn dash_plus_prefix_shorthand() -> anyhow::Result<()> {
        let s = SortSpec::parse("-duration")?;
        assert_eq!(field_at(&s, 0)?.direction, Some(Direction::Desc));
        let s = SortSpec::parse(" + start ")?;
        let f0 = field_at(&s, 0)?;
        assert_eq!(f0.key, "start");
        assert_eq!(f0.direction, Some(Direction::Asc));
        Ok(())
    }

    #[test]
    fn multi_field_parses_each_independently() -> anyhow::Result<()> {
        let s = SortSpec::parse("-total, +name, p99:desc")?;
        assert_eq!(s.0.len(), 3);
        let f0 = field_at(&s, 0)?;
        let f1 = field_at(&s, 1)?;
        let f2 = field_at(&s, 2)?;
        assert_eq!(f0.key, "total");
        assert_eq!(f0.direction, Some(Direction::Desc));
        assert_eq!(f1.key, "name");
        assert_eq!(f1.direction, Some(Direction::Asc));
        assert_eq!(f2.key, "p99");
        assert_eq!(f2.direction, Some(Direction::Desc));
        Ok(())
    }

    #[test]
    fn bad_direction_rejected() {
        assert!(matches!(
            SortSpec::parse("total:nope"),
            Err(SortParseError::BadDirection(_))
        ));
    }

    #[test]
    fn empty_inputs_rejected() -> anyhow::Result<()> {
        assert!(matches!(
            SortSpec::parse(""),
            Err(SortParseError::EmptySpec)
        ));
        assert!(matches!(
            SortSpec::parse("   "),
            Err(SortParseError::EmptySpec)
        ));
        // Trailing comma is tolerated as long as at least one field is real.
        assert_eq!(SortSpec::parse("total,")?.0.len(), 1);
        Ok(())
    }

    /// Toy `SortKeyDef` impl exercising the table-driven lookup +
    /// help-text rendering. Mirrors the per-command shape (a few
    /// keys, mixed defaults, an alias) without depending on any
    /// real command's vocabulary.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ToyKey {
        Total,
        Name,
        Duration,
    }

    impl SortKeyDef for ToyKey {
        fn specs() -> &'static [SortKeySpec<Self>] {
            &[
                SortKeySpec {
                    variant: ToyKey::Total,
                    canonical: "total",
                    aliases: &["total_ns"],
                    default_dir: Direction::Desc,
                },
                SortKeySpec {
                    variant: ToyKey::Name,
                    canonical: "name",
                    aliases: &[],
                    default_dir: Direction::Asc,
                },
                SortKeySpec {
                    variant: ToyKey::Duration,
                    canonical: "duration",
                    aliases: &["dur"],
                    default_dir: Direction::Desc,
                },
            ]
        }
    }

    #[test]
    fn sort_key_def_resolves_canonical() -> Result<(), SortParseError> {
        let f = SortField::parse("total")?;
        let (k, d) = ToyKey::from_field(&f)?;
        assert_eq!(k, ToyKey::Total);
        assert_eq!(d, Direction::Desc); // per-key default
        Ok(())
    }

    #[test]
    fn sort_key_def_resolves_alias() -> Result<(), SortParseError> {
        let f = SortField::parse("total_ns:asc")?;
        let (k, d) = ToyKey::from_field(&f)?;
        assert_eq!(k, ToyKey::Total);
        assert_eq!(d, Direction::Asc); // caller override beats default
        Ok(())
    }

    #[test]
    fn sort_key_def_rejects_unknown_with_expected_list() -> Result<(), SortParseError> {
        let f = SortField::parse("nope")?;
        let err = match ToyKey::from_field(&f) {
            Err(e) => e.to_string(),
            Ok(_) => {
                return Err(SortParseError::BadDirection(
                    "nope unexpectedly resolved".to_string(),
                ));
            }
        };
        assert!(err.contains("unknown --sort key `nope`"), "got: {err}");
        assert!(err.contains("total"));
        assert!(err.contains("name"));
        assert!(err.contains("duration"));
        Ok(())
    }

    #[test]
    fn sort_key_def_help_text_marks_first_as_default() {
        let s = ToyKey::help_text();
        assert_eq!(s, "total (default DESC), name (ASC), duration (DESC)");
    }

    #[test]
    fn direction_or_default() -> anyhow::Result<()> {
        let s = SortSpec::parse("total")?;
        assert_eq!(
            field_at(&s, 0)?.direction_or(Direction::Desc),
            Direction::Desc
        );
        let s = SortSpec::parse("total:asc")?;
        assert_eq!(
            field_at(&s, 0)?.direction_or(Direction::Desc),
            Direction::Asc
        );
        Ok(())
    }
}
