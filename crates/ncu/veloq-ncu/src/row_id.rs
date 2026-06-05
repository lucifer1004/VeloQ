//! Shared `launch:<idx>` parser for the NCU drill verbs (`disasm`,
//! `source-metrics`).
//!
//! The parser is shared so the drill verbs keep one
//! parser and stable error wording.

use anyhow::{Context, Result};

/// Parse a `launch:<idx>` row_id into the launch index, using the same
/// error wording across NCU drill verbs so the error envelope stays
/// consistent. Only `launch:<idx>` is accepted.
pub(crate) fn parse_launch_idx(s: &str) -> Result<usize> {
    let (kind, idx) = s
        .split_once(':')
        .with_context(|| format!("expected `launch:<idx>`, got `{s}`"))?;
    anyhow::ensure!(
        kind == "launch",
        "ncu drill verbs currently support only `launch:<idx>` row_ids (got `{kind}`)"
    );
    idx.parse::<usize>()
        .with_context(|| format!("invalid launch index `{idx}` in `{s}`"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_well_formed_launch_id() -> Result<()> {
        assert_eq!(parse_launch_idx("launch:0")?, 0);
        assert_eq!(parse_launch_idx("launch:42")?, 42);
        Ok(())
    }

    fn assert_rejects(input: &str, needle: &str) -> Result<()> {
        let err = parse_launch_idx(input)
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected `{input}` to fail to parse"))?;
        anyhow::ensure!(
            err.to_string().contains(needle),
            "expected error to contain `{needle}`; got: {err}"
        );
        Ok(())
    }

    #[test]
    fn rejects_missing_colon() -> Result<()> {
        assert_rejects("launch0", "expected")
    }

    #[test]
    fn rejects_unknown_kind() -> Result<()> {
        assert_rejects("range:0", "launch:<idx>")
    }

    #[test]
    fn rejects_non_numeric_index() -> Result<()> {
        assert_rejects("launch:abc", "invalid launch index")
    }
}
