use std::convert::Infallible;
use std::num::TryFromIntError;

/// Row that carries a SQL `COUNT(*) OVER () AS total_matched` value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TotalCarrier {
    First,
    Last,
}

fn carrier_row<T>(rows: &[T], carrier: TotalCarrier) -> Option<&T> {
    match carrier {
        TotalCarrier::First => rows.first(),
        TotalCarrier::Last => rows.last(),
    }
}

pub fn infallible_count_error<E>(err: Infallible) -> E {
    match err {}
}

pub trait TotalCount: Sized {
    type Error;

    fn try_from_i64(value: i64) -> Result<Self, Self::Error>;
}

impl TotalCount for i64 {
    type Error = Infallible;

    fn try_from_i64(value: i64) -> Result<Self, Self::Error> {
        Ok(value)
    }
}

impl TotalCount for usize {
    type Error = TryFromIntError;

    fn try_from_i64(value: i64) -> Result<Self, Self::Error> {
        usize::try_from(value)
    }
}

/// Convert a non-negative SQL count into a `usize` at the query boundary.
pub fn count_from_i64<E>(
    value: i64,
    overflow: impl FnOnce(TryFromIntError) -> E,
) -> Result<usize, E> {
    usize::try_from(value).map_err(overflow)
}

/// Recover `COUNT(*) OVER () AS total_matched` from a typed carrier row.
///
/// SQL list queries typically return the total on every row after `LIMIT`.
/// Empty result sets have no carrier row, so the total is zero.
pub fn total_matched<Count, T>(
    rows: &[T],
    carrier: TotalCarrier,
    total_matched: impl FnOnce(&T) -> i64,
) -> Result<Count, Count::Error>
where
    Count: TotalCount,
{
    match carrier_row(rows, carrier) {
        Some(row) => Count::try_from_i64(total_matched(row)),
        None => Count::try_from_i64(0),
    }
}

/// Split hydrated SQL rows into wire rows plus a caller-computed total.
///
/// The total is computed while the raw rows are still available as a slice,
/// then each raw row is consumed and mapped into its final response shape.
pub fn split_rows_and_computed_total<T, U, Total, E>(
    rows: Vec<T>,
    total: impl FnOnce(&[T]) -> Result<Total, E>,
    map_row: impl FnMut(T) -> Result<U, E>,
) -> Result<(Vec<U>, Total), E> {
    let total = total(&rows)?;
    let rows = rows.into_iter().map(map_row).collect::<Result<_, _>>()?;
    Ok((rows, total))
}

/// Split hydrated rows and recover a typed total from a typed carrier row.
pub fn split_rows_and_total<Count, T, U, E>(
    rows: Vec<T>,
    carrier: TotalCarrier,
    total_matched_of: impl FnOnce(&T) -> i64,
    count_error: impl FnOnce(Count::Error) -> E,
    map_row: impl FnMut(T) -> Result<U, E>,
) -> Result<(Vec<U>, Count), E>
where
    Count: TotalCount,
{
    split_rows_and_computed_total(
        rows,
        |rows| total_matched::<Count, T>(rows, carrier, total_matched_of).map_err(count_error),
        map_row,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_matched_converts_carrier_row_or_zero() -> anyhow::Result<()> {
        #[derive(Debug)]
        struct Row {
            total_matched: i64,
        }

        let rows = [Row { total_matched: 3 }, Row { total_matched: 5 }];
        let first = total_matched::<usize, _>(&rows, TotalCarrier::First, |row| row.total_matched)
            .map_err(anyhow::Error::new)?;
        let last = total_matched::<usize, _>(&rows, TotalCarrier::Last, |row| row.total_matched)
            .map_err(anyhow::Error::new)?;
        let empty = total_matched::<usize, Row>(&[], TotalCarrier::First, |row| row.total_matched)
            .map_err(anyhow::Error::new)?;

        assert_eq!(first, 3);
        assert_eq!(last, 5);
        assert_eq!(empty, 0);
        Ok(())
    }

    #[test]
    fn total_matched_uses_typed_carrier() -> anyhow::Result<()> {
        #[derive(Debug)]
        struct Row {
            total_matched: i64,
        }

        let rows = [Row { total_matched: 3 }, Row { total_matched: 5 }];

        assert_eq!(
            total_matched::<i64, _>(&rows, TotalCarrier::First, |row| row.total_matched)
                .map_err(infallible_count_error::<anyhow::Error>)?,
            3
        );
        assert_eq!(
            total_matched::<i64, _>(&rows, TotalCarrier::Last, |row| row.total_matched)
                .map_err(infallible_count_error::<anyhow::Error>)?,
            5
        );
        assert_eq!(
            total_matched::<i64, Row>(&[], TotalCarrier::First, |row| row.total_matched)
                .map_err(infallible_count_error::<anyhow::Error>)?,
            0
        );
        assert_eq!(
            total_matched::<i64, Row>(&[], TotalCarrier::Last, |row| row.total_matched)
                .map_err(infallible_count_error::<anyhow::Error>)?,
            0
        );
        Ok(())
    }

    #[test]
    fn split_rows_and_computed_total_computes_total_before_mapping() -> anyhow::Result<()> {
        #[derive(Debug)]
        struct Row {
            value: i64,
            total_matched: i64,
        }

        let rows = vec![
            Row {
                value: 1,
                total_matched: 2,
            },
            Row {
                value: 2,
                total_matched: 2,
            },
        ];
        let (values, total) = split_rows_and_computed_total(
            rows,
            |rows| {
                total_matched::<i64, _>(rows, TotalCarrier::First, |row| row.total_matched)
                    .map_err(infallible_count_error::<anyhow::Error>)
            },
            |row| Ok::<_, anyhow::Error>(row.value * 10),
        )?;

        assert_eq!(values, vec![10, 20]);
        assert_eq!(total, 2);
        Ok(())
    }

    #[test]
    fn split_rows_and_total_carrier_helpers_map_rows() -> anyhow::Result<()> {
        #[derive(Debug)]
        struct Row {
            value: i64,
            total_matched: i64,
        }

        let rows = vec![
            Row {
                value: 1,
                total_matched: 3,
            },
            Row {
                value: 2,
                total_matched: 5,
            },
        ];
        let (first_values, first_total) = split_rows_and_total::<i64, _, _, _>(
            rows,
            TotalCarrier::First,
            |row| row.total_matched,
            infallible_count_error,
            |row| Ok::<_, anyhow::Error>(row.value * 10),
        )?;
        assert_eq!(first_values, vec![10, 20]);
        assert_eq!(first_total, 3);

        let rows = vec![
            Row {
                value: 1,
                total_matched: 3,
            },
            Row {
                value: 2,
                total_matched: 5,
            },
        ];
        let (last_values, last_total) = split_rows_and_total::<i64, _, _, _>(
            rows,
            TotalCarrier::Last,
            |row| row.total_matched,
            infallible_count_error,
            |row| Ok::<_, anyhow::Error>(row.value * 10),
        )?;
        assert_eq!(last_values, vec![10, 20]);
        assert_eq!(last_total, 5);

        let rows = vec![Row {
            value: 1,
            total_matched: 2,
        }];
        let (usize_values, usize_total) = split_rows_and_total::<usize, _, _, _>(
            rows,
            TotalCarrier::First,
            |row| row.total_matched,
            anyhow::Error::new,
            |row| Ok::<_, anyhow::Error>(row.value * 10),
        )?;
        assert_eq!(usize_values, vec![10]);
        assert_eq!(usize_total, 2);
        Ok(())
    }
}
