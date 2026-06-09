/// Quote one command word for POSIX-style shells.
///
/// Used only for agent-facing `meta.next_steps.command` strings. It is
/// intentionally small: safe words are left bare; every other word is
/// single-quoted with embedded single quotes escaped as `'\''`.
pub fn shell_quote(word: &str) -> String {
    if !word.is_empty()
        && word.bytes().all(|b| {
            b.is_ascii_alphanumeric() || matches!(b, b'/' | b'.' | b'_' | b'-' | b':' | b'@')
        })
    {
        return word.to_string();
    }

    let mut out = String::with_capacity(word.len() + 2);
    out.push('\'');
    for ch in word.chars() {
        if ch == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(ch);
        }
    }
    out.push('\'');
    out
}

#[cfg(test)]
mod tests {
    use super::shell_quote;

    #[test]
    fn leaves_safe_words_bare() {
        assert_eq!(shell_quote("demo.nsys-rep"), "demo.nsys-rep");
        assert_eq!(
            shell_quote("target/trace.veloq/parquetdir"),
            "target/trace.veloq/parquetdir"
        );
        assert_eq!(shell_quote("kernel:390"), "kernel:390");
    }

    #[test]
    fn quotes_spaces_and_single_quotes() {
        assert_eq!(shell_quote("a b"), "'a b'");
        assert_eq!(shell_quote("a'b"), "'a'\\''b'");
        assert_eq!(shell_quote(""), "''");
    }
}
