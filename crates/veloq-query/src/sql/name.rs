use crate::sql::SqlFragment;
use duckdb::types::Value;
use veloq_core::NameFilterRef;

pub fn glob_like(expr: &str, pattern: &str) -> SqlFragment {
    SqlFragment::new(
        format!(r#"{expr} LIKE ? ESCAPE '\'"#),
        vec![Value::Text(veloq_core::query::glob_sql_like(pattern))],
    )
}

pub fn regex_matches(expr: &str, pattern: &str) -> SqlFragment {
    SqlFragment::new(
        format!("regexp_matches({expr}, ?)"),
        vec![Value::Text(pattern.to_string())],
    )
}

pub fn predicate(expr: &str, filter: NameFilterRef<'_>) -> Option<SqlFragment> {
    match filter {
        NameFilterRef::Any => None,
        NameFilterRef::Glob(pattern) => Some(glob_like(expr, pattern)),
        NameFilterRef::Regex(pattern) => Some(regex_matches(expr, pattern)),
    }
}

pub fn predicate_or_bound_true(expr: &str, filter: NameFilterRef<'_>) -> SqlFragment {
    predicate(expr, filter).unwrap_or_else(|| {
        SqlFragment::new("(? IS NOT NULL OR TRUE)", vec![Value::Text(String::new())])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_like_uses_canonical_like_escape_and_param() {
        let fragment = glob_like("name", "foo*_%");

        assert_eq!(fragment.sql, r#"name LIKE ? ESCAPE '\'"#);
        assert_eq!(fragment.params, vec![Value::Text("foo%\\_\\%".to_string())]);
    }

    #[test]
    fn regex_matches_passes_pattern_verbatim() {
        let fragment = regex_matches("COALESCE(name, '')", "foo|bar");

        assert_eq!(fragment.sql, "regexp_matches(COALESCE(name, ''), ?)");
        assert_eq!(fragment.params, vec![Value::Text("foo|bar".to_string())]);
    }

    #[test]
    fn predicate_returns_none_for_unfiltered_names() -> anyhow::Result<()> {
        let filter = NameFilterRef::from_optional(None, None)?;

        assert!(predicate("name", filter).is_none());
        Ok(())
    }

    #[test]
    fn bound_true_preserves_legacy_single_param_layout() -> anyhow::Result<()> {
        let fragment = predicate_or_bound_true("name", NameFilterRef::from_optional(None, None)?);

        assert_eq!(fragment.sql, "(? IS NOT NULL OR TRUE)");
        assert_eq!(fragment.params, vec![Value::Text(String::new())]);
        Ok(())
    }
}
