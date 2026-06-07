use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum LimitError {
    #[error("limit must be at least 1")]
    TooSmall { limit: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LimitRef {
    value: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LimitedRows<T> {
    pub rows: Vec<T>,
    pub total_matched: usize,
}

impl LimitRef {
    pub fn new(limit: usize) -> Result<Self, LimitError> {
        if limit == 0 {
            return Err(LimitError::TooSmall { limit });
        }
        Ok(Self { value: limit })
    }

    pub fn value(self) -> usize {
        self.value
    }

    /// Apply this limit to an already sorted row set while preserving the
    /// pre-limit row count for canonical `total_matched` fields.
    pub fn apply<T>(self, mut rows: Vec<T>) -> LimitedRows<T> {
        let total_matched = rows.len();
        rows.truncate(self.value);
        LimitedRows {
            rows,
            total_matched,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_limit_is_rejected() {
        assert_eq!(LimitRef::new(0), Err(LimitError::TooSmall { limit: 0 }));
    }

    #[test]
    fn value_returns_original_limit() -> anyhow::Result<()> {
        assert_eq!(LimitRef::new(3)?.value(), 3);
        Ok(())
    }

    #[test]
    fn apply_truncates_and_preserves_total_matched() -> anyhow::Result<()> {
        let limited = LimitRef::new(2)?.apply(vec!["a", "b", "c"]);

        assert_eq!(limited.rows, vec!["a", "b"]);
        assert_eq!(limited.total_matched, 3);
        Ok(())
    }

    #[test]
    fn apply_keeps_short_rows_unchanged() -> anyhow::Result<()> {
        let limited = LimitRef::new(5)?.apply(vec![1, 2]);

        assert_eq!(limited.rows, vec![1, 2]);
        assert_eq!(limited.total_matched, 2);
        Ok(())
    }
}
