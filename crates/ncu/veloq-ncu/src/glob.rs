//! Shell-style `*`/`?` glob matcher, in-Rust only.
//!
//! NCU's filter args (`--kernel`, `--nvtx-range`, ...) post-filter
//! rows in memory rather than running through a query language,
//! so we don't need the SQL-LIKE escape pass that
//! `veloq_nsys_query::search_glob_to_like` does. This module is the
//! NCU-side complement: take a glob, compile it once, match many.
//!
//! Grammar: `*` matches any sequence (possibly empty), `?` matches
//! exactly one char, every other byte matches literally. No escape
//! sequence — NCU strings are user-supplied globs, not SQL.

/// Compiled glob. Cheap to construct (one allocation for the char
/// buffer); call [`Matcher::matches`] N times after that.
pub struct Matcher {
    pattern: Vec<char>,
}

/// Compile `pattern` into a [`Matcher`].
pub fn compile(pattern: &str) -> Matcher {
    Matcher {
        pattern: pattern.chars().collect(),
    }
}

impl Matcher {
    /// `true` iff `s` matches the compiled glob.
    pub fn matches(&self, s: &str) -> bool {
        let s_chars: Vec<char> = s.chars().collect();
        rec(&s_chars, &self.pattern)
    }
}

fn rec(s: &[char], p: &[char]) -> bool {
    let (head, tail) = match p.split_first() {
        Some((h, t)) => (*h, t),
        None => return s.is_empty(),
    };
    match head {
        '*' => {
            if rec(s, tail) {
                return true;
            }
            match s.split_first() {
                Some((_, srest)) => rec(srest, p),
                None => false,
            }
        }
        '?' => match s.split_first() {
            Some((_, srest)) => rec(srest, tail),
            None => false,
        },
        lit => match s.split_first() {
            Some((sc, srest)) if *sc == lit => rec(srest, tail),
            _ => false,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_match() {
        assert!(compile("step_42").matches("step_42"));
        assert!(!compile("step_42").matches("step_43"));
    }

    #[test]
    fn star_matches_any_sequence_including_empty() {
        let m = compile("*decode*");
        assert!(m.matches("decode"));
        assert!(m.matches("pre_decode_post"));
        assert!(!m.matches("encode"));
    }

    #[test]
    fn question_matches_exactly_one_char() {
        let m = compile("step_?");
        assert!(m.matches("step_0"));
        assert!(m.matches("step_9"));
        assert!(!m.matches("step_"));
        assert!(!m.matches("step_42"));
    }

    #[test]
    fn nested_path_segments_with_slash() {
        // Used by `--nvtx-range '*step/*decode*'`.
        let m = compile("*step/*decode*");
        assert!(m.matches("outer/step/inner/decode/run"));
        assert!(!m.matches("step/encode"));
    }
}
