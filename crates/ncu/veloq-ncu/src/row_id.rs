//! Shared `launch:<idx>` parser for the NCU drill verbs (`disasm`,
//! `source-metrics`).
//!
//! The parser is shared so the drill verbs keep one
//! parser and stable error wording.

use crate::error::{NcuSourceError, NcuSourceResult};

/// Parse a `launch:<idx>` row_id into the launch index, using the same
/// error wording across NCU drill verbs so the error envelope stays
/// consistent. Only `launch:<idx>` is accepted.
pub(crate) fn parse_launch_idx(s: &str) -> NcuSourceResult<usize> {
    let Some((kind, idx)) = s.split_once(':') else {
        return Err(NcuSourceError::launch_row_id_missing_colon(s));
    };
    if kind != "launch" {
        return Err(NcuSourceError::launch_row_id_kind_unsupported(kind));
    }
    idx.parse::<usize>()
        .map_err(|source| NcuSourceError::launch_row_id_index_invalid(s, idx, source))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use veloq_core::VeloqDiagnostic;

    #[test]
    fn parses_well_formed_launch_id() -> Result<()> {
        assert_eq!(parse_launch_idx("launch:0")?, 0);
        assert_eq!(parse_launch_idx("launch:42")?, 42);
        Ok(())
    }

    fn assert_rejects(input: &str, needle: &str) -> Result<NcuSourceError> {
        let err = parse_launch_idx(input)
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected `{input}` to fail to parse"))?;
        anyhow::ensure!(
            err.to_string().contains(needle),
            "expected error to contain `{needle}`; got: {err}"
        );
        Ok(err)
    }

    fn assert_typed_invalid_row_id(input: &str, needle: &str) -> Result<NcuSourceError> {
        let err = assert_rejects(input, needle)?;
        assert_eq!(err.code().as_str(), "ncu.command.invalid-launch-row-id");
        Ok(err)
    }

    #[test]
    fn rejects_missing_colon() -> Result<()> {
        let typed = assert_typed_invalid_row_id("launch0", "expected")?;
        assert!(matches!(
            typed,
            NcuSourceError::LaunchRowIdMissingColon { .. }
        ));
        Ok(())
    }

    #[test]
    fn rejects_unknown_kind() -> Result<()> {
        let typed = assert_typed_invalid_row_id("range:0", "launch:<idx>")?;
        assert!(matches!(
            typed,
            NcuSourceError::LaunchRowIdKindUnsupported { .. }
        ));
        Ok(())
    }

    #[test]
    fn rejects_non_numeric_index() -> Result<()> {
        let typed = assert_typed_invalid_row_id("launch:abc", "invalid launch index")?;
        assert!(matches!(
            typed,
            NcuSourceError::LaunchRowIdIndexInvalid { .. }
        ));
        Ok(())
    }
}
