use crate::{NsysQueryError, NsysQueryResult};
use veloq_core::{Direction, SortKeyDef, SortSpec, sort::SortParseError};

/// Resolve a command-specific [`SortSpec`] into an SQL `ORDER BY` body.
///
/// Callers provide their typed key-to-column mapping and invalid-sort
/// error mapper, so the accepted keys and diagnostics remain local to the
/// verb while the `SortKeyDef::from_field` loop stays single-sourced.
pub(crate) fn order_by<K>(
    spec: &SortSpec,
    column: impl Fn(K) -> &'static str,
    invalid_sort: fn(SortParseError) -> NsysQueryError,
    tiebreaker: &str,
) -> NsysQueryResult<String>
where
    K: SortKeyDef,
{
    let mut resolved: Vec<(&'static str, Direction)> = Vec::new();
    for field in spec.fields() {
        let (key, direction) = K::from_field(field).map_err(invalid_sort)?;
        resolved.push((column(key), direction));
    }
    Ok(veloq_core::sort::build_order_by(&resolved, tiebreaker))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use veloq_core::{SortKeySpec, VeloqDiagnostic};

    #[derive(Clone, Copy)]
    enum TestSortKey {
        Total,
        Name,
    }

    impl SortKeyDef for TestSortKey {
        fn specs() -> &'static [SortKeySpec<Self>] {
            &[
                SortKeySpec {
                    variant: TestSortKey::Total,
                    canonical: "total",
                    aliases: &["sum"],
                    default_dir: Direction::Desc,
                },
                SortKeySpec {
                    variant: TestSortKey::Name,
                    canonical: "name",
                    aliases: &[],
                    default_dir: Direction::Asc,
                },
            ]
        }
    }

    fn column(key: TestSortKey) -> &'static str {
        match key {
            TestSortKey::Total => "total_ns",
            TestSortKey::Name => "name",
        }
    }

    #[test]
    fn order_by_resolves_alias_defaults_and_tiebreaker() -> Result<()> {
        let spec = SortSpec::parse("sum,name:desc")?;
        let order =
            order_by::<TestSortKey>(&spec, column, NsysQueryError::search_sort_invalid, "row_id")?;
        assert_eq!(order, "total_ns DESC, name DESC, row_id DESC");
        Ok(())
    }

    #[test]
    fn invalid_sort_uses_callers_error_mapper() -> Result<()> {
        let spec = SortSpec::single("missing");
        let err = match order_by::<TestSortKey>(
            &spec,
            column,
            NsysQueryError::stats_by_size_sort_invalid,
            "row_id",
        ) {
            Ok(order) => anyhow::bail!("unknown key should fail, got order {order}"),
            Err(err) => err,
        };
        assert_eq!(err.code().as_str(), "nsys.query.stats-by-size-sort-invalid");
        Ok(())
    }
}
