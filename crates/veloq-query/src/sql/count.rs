/// SQL window-count expression for list queries that carry
/// `total_matched` on each returned row.
pub fn total_matched_expr() -> &'static str {
    "COUNT(*) OVER () AS total_matched"
}

/// SQL window-count expression forced to DuckDB BIGINT.
pub fn total_matched_bigint_expr() -> &'static str {
    "CAST(COUNT(*) OVER () AS BIGINT) AS total_matched"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_matched_expr_keeps_wire_alias() {
        assert_eq!(total_matched_expr(), "COUNT(*) OVER () AS total_matched");
    }

    #[test]
    fn total_matched_bigint_expr_keeps_wire_alias_and_cast() {
        assert_eq!(
            total_matched_bigint_expr(),
            "CAST(COUNT(*) OVER () AS BIGINT) AS total_matched"
        );
    }
}
