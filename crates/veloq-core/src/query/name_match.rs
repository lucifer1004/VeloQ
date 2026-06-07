use regex::Regex;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum NameMatchError {
    #[error("name glob and regex filters are mutually exclusive")]
    Conflict,
    #[error("invalid name glob `{pattern}`")]
    InvalidGlob {
        pattern: String,
        #[source]
        source: regex::Error,
    },
    #[error("invalid name regex `{pattern}`")]
    InvalidRegex {
        pattern: String,
        #[source]
        source: regex::Error,
    },
}

#[derive(Debug, Clone)]
pub struct NameMatcher {
    glob: Option<Regex>,
    regex: Option<Regex>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NameFilterRef<'a> {
    Any,
    Glob(&'a str),
    Regex(&'a str),
}

impl<'a> NameFilterRef<'a> {
    pub fn from_optional(
        glob: Option<&'a str>,
        regex: Option<&'a str>,
    ) -> Result<Self, NameMatchError> {
        match (glob, regex) {
            (Some(_), Some(_)) => Err(NameMatchError::Conflict),
            (Some(pattern), None) => Ok(Self::Glob(pattern)),
            (None, Some(pattern)) => Ok(Self::Regex(pattern)),
            (None, None) => Ok(Self::Any),
        }
    }

    pub fn sql_like_pattern(self) -> Option<String> {
        match self {
            Self::Glob(pattern) => Some(glob_sql_like(pattern)),
            Self::Any | Self::Regex(_) => None,
        }
    }

    pub fn compile_matcher(self) -> Result<NameMatcher, NameMatchError> {
        match self {
            Self::Any => Ok(NameMatcher {
                glob: None,
                regex: None,
            }),
            Self::Glob(pattern) => Ok(NameMatcher {
                glob: Some(compile_glob(pattern)?),
                regex: None,
            }),
            Self::Regex(pattern) => Ok(NameMatcher {
                glob: None,
                regex: Some(compile_regex(pattern)?),
            }),
        }
    }
}

impl NameMatcher {
    pub fn new(glob: Option<&str>, regex: Option<&str>) -> Result<Self, NameMatchError> {
        NameFilterRef::from_optional(glob, regex)?.compile_matcher()
    }

    pub fn matches(&self, name: &str) -> bool {
        self.glob.as_ref().is_none_or(|regex| regex.is_match(name))
            && self.regex.as_ref().is_none_or(|regex| regex.is_match(name))
    }
}

pub fn glob_regex(pattern: &str) -> String {
    let mut out = String::from("^");
    for ch in pattern.chars() {
        match ch {
            '*' => out.push_str(".*"),
            '?' => out.push('.'),
            _ => out.push_str(&regex::escape(&ch.to_string())),
        }
    }
    out.push('$');
    out
}

/// Project the canonical VeloQ shell-style glob grammar (`*` / `?`) into
/// SQL `LIKE` pattern grammar, escaping SQL-special `%` / `_` / `\` chars
/// in the input so they stay literal.
///
/// This returns only a bindable pattern string. Source crates still own
/// the SQL fragment, the `ESCAPE '\'` clause, and backend-specific bind
/// values.
pub fn glob_sql_like(pattern: &str) -> String {
    let mut out = String::with_capacity(pattern.len());
    for ch in pattern.chars() {
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

/// Apply the SQL `LIKE` pattern grammar in Rust for patterns produced by
/// [`glob_sql_like`].
///
/// This is for source code that needs to post-filter rows in memory while
/// keeping the same name semantics as SQL-backed filtering. It does not
/// parse SQL clauses.
pub fn sql_like_matches(s: &str, pattern: &str) -> bool {
    let s_chars: Vec<char> = s.chars().collect();
    let pattern_chars: Vec<char> = pattern.chars().collect();

    fn rec(s: &[char], pattern: &[char]) -> bool {
        let (head, tail) = match pattern.split_first() {
            Some((head, tail)) => (*head, tail),
            None => return s.is_empty(),
        };

        match head {
            '%' => {
                if rec(s, tail) {
                    return true;
                }
                match s.split_first() {
                    Some((_, s_tail)) => rec(s_tail, pattern),
                    None => false,
                }
            }
            '_' => match s.split_first() {
                Some((_, s_tail)) => rec(s_tail, tail),
                None => false,
            },
            '\\' => match (tail.split_first(), s.split_first()) {
                (Some((literal, pattern_tail)), Some((s_head, s_tail))) if *literal == *s_head => {
                    rec(s_tail, pattern_tail)
                }
                _ => false,
            },
            literal => match s.split_first() {
                Some((s_head, s_tail)) if *s_head == literal => rec(s_tail, tail),
                _ => false,
            },
        }
    }

    rec(&s_chars, &pattern_chars)
}

fn compile_glob(pattern: &str) -> Result<Regex, NameMatchError> {
    let regex_pattern = glob_regex(pattern);
    Regex::new(&regex_pattern).map_err(|source| NameMatchError::InvalidGlob {
        pattern: pattern.to_string(),
        source,
    })
}

fn compile_regex(pattern: &str) -> Result<Regex, NameMatchError> {
    Regex::new(pattern).map_err(|source| NameMatchError::InvalidRegex {
        pattern: pattern.to_string(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_matches_full_name() -> anyhow::Result<()> {
        let matcher = NameMatcher::new(Some("aten::*"), None)?;
        assert!(matcher.matches("aten::matmul"));
        assert!(!matcher.matches("cuda::matmul"));
        Ok(())
    }

    #[test]
    fn question_mark_matches_one_char() -> anyhow::Result<()> {
        let matcher = NameMatcher::new(Some("ker?el"), None)?;
        assert!(matcher.matches("kernel"));
        assert!(!matcher.matches("kerXXel"));
        Ok(())
    }

    #[test]
    fn regex_matches_when_supplied() -> anyhow::Result<()> {
        let matcher = NameMatcher::new(None, Some("kernel|memcpy"))?;
        assert!(matcher.matches("kernel"));
        assert!(matcher.matches("cuda_memcpy"));
        assert!(!matcher.matches("runtime"));
        Ok(())
    }

    #[test]
    fn glob_and_regex_conflict() {
        assert!(matches!(
            NameMatcher::new(Some("*"), Some(".*")),
            Err(NameMatchError::Conflict)
        ));
    }

    #[test]
    fn name_filter_ref_selects_one_mode() -> anyhow::Result<()> {
        assert_eq!(
            NameFilterRef::from_optional(None, None)?,
            NameFilterRef::Any
        );
        assert_eq!(
            NameFilterRef::from_optional(Some("foo*"), None)?,
            NameFilterRef::Glob("foo*")
        );
        assert_eq!(
            NameFilterRef::from_optional(None, Some("foo|bar"))?,
            NameFilterRef::Regex("foo|bar")
        );
        Ok(())
    }

    #[test]
    fn name_filter_ref_reports_conflict() {
        assert!(matches!(
            NameFilterRef::from_optional(Some("*"), Some(".*")),
            Err(NameMatchError::Conflict)
        ));
    }

    #[test]
    fn name_filter_ref_projects_glob_to_sql_like_only() -> anyhow::Result<()> {
        assert_eq!(
            NameFilterRef::from_optional(Some("foo*"), None)?.sql_like_pattern(),
            Some("foo%".to_string())
        );
        assert_eq!(
            NameFilterRef::from_optional(None, Some("foo.*"))?.sql_like_pattern(),
            None
        );
        assert_eq!(
            NameFilterRef::from_optional(None, None)?.sql_like_pattern(),
            None
        );
        Ok(())
    }

    #[test]
    fn invalid_regex_is_typed() {
        assert!(matches!(
            NameMatcher::new(None, Some("[")),
            Err(NameMatchError::InvalidRegex { pattern, .. }) if pattern == "["
        ));
    }

    #[test]
    fn glob_regex_escapes_regex_metacharacters() {
        assert_eq!(glob_regex("a.b*"), "^a\\.b.*$");
    }

    #[test]
    fn glob_sql_like_converts_wildcards_and_escapes_sql_special_chars() {
        assert_eq!(glob_sql_like("foo*bar"), "foo%bar");
        assert_eq!(glob_sql_like("ker?el"), "ker_el");
        assert_eq!(glob_sql_like("100%_case"), "100\\%\\_case");
        assert_eq!(glob_sql_like("a\\b"), "a\\\\b");
    }

    #[test]
    fn glob_sql_like_round_trips_through_sql_like_matches() {
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
            ("100%", "100%", true),
            ("100%", "100x", false),
            ("a_b", "a_b", true),
            ("a_b", "axb", false),
            ("a\\b", "a\\b", true),
        ];

        for (glob, candidate, expected) in cases {
            let pattern = glob_sql_like(glob);
            assert_eq!(
                sql_like_matches(candidate, &pattern),
                *expected,
                "glob=`{glob}` -> pattern=`{pattern}` vs candidate=`{candidate}`"
            );
        }
    }
}
