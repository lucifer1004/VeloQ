use duckdb::types::Value;

/// Coerce DuckDB values into the trait-object slice shape expected by
/// `Statement::query`.
///
/// The returned values borrow `params`; bind the returned vector to a local
/// for at least as long as the `Rows` value returned by DuckDB is alive.
fn bind(params: &[Value]) -> Vec<&dyn duckdb::ToSql> {
    params.iter().map(|v| v as &dyn duckdb::ToSql).collect()
}

/// Source-local SQL label/error mapper used by the labeled query helpers.
///
/// The shared DuckDB row loops stay source-neutral: each profile source owns
/// the concrete label type and maps prepare/query/read failures into its own
/// diagnostic enum.
pub trait SqlErrorMapper<E>: Copy {
    fn prepare_error(self, source: duckdb::Error) -> E;
    fn query_error(self, source: duckdb::Error) -> E;
    fn read_error(self, source: duckdb::Error) -> E;
}

fn collect_rows<T, E>(
    rows: &mut duckdb::Rows<'_>,
    mut read_error: impl FnMut(duckdb::Error) -> E,
    mut hydrate: impl FnMut(&duckdb::Row<'_>) -> Result<T, duckdb::Error>,
) -> Result<Vec<T>, E> {
    let mut out = Vec::new();
    while let Some(row) = rows.next().map_err(&mut read_error)? {
        out.push(hydrate(row).map_err(&mut read_error)?);
    }
    Ok(out)
}

fn collect_rows_fallible<T, E>(
    rows: &mut duckdb::Rows<'_>,
    mut read_error: impl FnMut(duckdb::Error) -> E,
    mut hydrate: impl FnMut(&duckdb::Row<'_>) -> Result<T, E>,
) -> Result<Vec<T>, E> {
    let mut out = Vec::new();
    while let Some(row) = rows.next().map_err(&mut read_error)? {
        out.push(hydrate(row)?);
    }
    Ok(out)
}

/// Prepare `sql`, bind `params`, execute, and hydrate all rows.
///
/// Error constructors stay with the caller so source-specific diagnostics and
/// error codes do not leak into this shared crate.
fn query_rows<T, E, PrepareError, QueryError, ReadError, Hydrate>(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
    prepare_error: PrepareError,
    query_error: QueryError,
    read_error: ReadError,
    hydrate: Hydrate,
) -> Result<Vec<T>, E>
where
    PrepareError: FnOnce(duckdb::Error) -> E,
    QueryError: FnOnce(duckdb::Error) -> E,
    ReadError: FnMut(duckdb::Error) -> E,
    Hydrate: FnMut(&duckdb::Row<'_>) -> Result<T, duckdb::Error>,
{
    let mut stmt = conn.prepare(sql).map_err(prepare_error)?;
    let bound = bind(params);
    let mut rows = stmt.query(bound.as_slice()).map_err(query_error)?;
    collect_rows(&mut rows, read_error, hydrate)
}

/// Prepare `sql`, bind `params`, execute, and hydrate all rows using a
/// source-local SQL label for phase-specific diagnostics.
pub fn query_rows_labeled<T, E, Label, Hydrate>(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
    label: Label,
    hydrate: Hydrate,
) -> Result<Vec<T>, E>
where
    Label: SqlErrorMapper<E>,
    Hydrate: FnMut(&duckdb::Row<'_>) -> Result<T, duckdb::Error>,
{
    query_rows(
        conn,
        sql,
        params,
        |source| label.prepare_error(source),
        |source| label.query_error(source),
        |source| label.read_error(source),
        hydrate,
    )
}

/// Bind `params`, execute an already-prepared statement, and hydrate all rows.
///
/// Use this when one statement is intentionally reused across several
/// parameter sets so callers can keep that prepare lifecycle while sharing the
/// query/read row loop.
fn query_rows_prepared<T, E, QueryError, ReadError, Hydrate>(
    stmt: &mut duckdb::Statement<'_>,
    params: &[Value],
    query_error: QueryError,
    read_error: ReadError,
    hydrate: Hydrate,
) -> Result<Vec<T>, E>
where
    QueryError: FnOnce(duckdb::Error) -> E,
    ReadError: FnMut(duckdb::Error) -> E,
    Hydrate: FnMut(&duckdb::Row<'_>) -> Result<T, duckdb::Error>,
{
    let bound = bind(params);
    let mut rows = stmt.query(bound.as_slice()).map_err(query_error)?;
    collect_rows(&mut rows, read_error, hydrate)
}

/// Bind `params`, execute an already-prepared statement, and hydrate all rows
/// using a source-local SQL label for query/read diagnostics.
pub fn query_rows_prepared_labeled<T, E, Label, Hydrate>(
    stmt: &mut duckdb::Statement<'_>,
    params: &[Value],
    label: Label,
    hydrate: Hydrate,
) -> Result<Vec<T>, E>
where
    Label: SqlErrorMapper<E>,
    Hydrate: FnMut(&duckdb::Row<'_>) -> Result<T, duckdb::Error>,
{
    query_rows_prepared(
        stmt,
        params,
        |source| label.query_error(source),
        |source| label.read_error(source),
        hydrate,
    )
}

/// Prepare `sql`, bind `params`, execute, and hydrate at most one row.
///
/// Multiple matching rows are intentionally ignored after the first, matching
/// the common `rows.next()` single-row probe pattern.
fn query_optional_row<T, E, PrepareError, QueryError, ReadError, Hydrate>(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
    prepare_error: PrepareError,
    query_error: QueryError,
    mut read_error: ReadError,
    hydrate: Hydrate,
) -> Result<Option<T>, E>
where
    PrepareError: FnOnce(duckdb::Error) -> E,
    QueryError: FnOnce(duckdb::Error) -> E,
    ReadError: FnMut(duckdb::Error) -> E,
    Hydrate: FnOnce(&duckdb::Row<'_>) -> Result<T, duckdb::Error>,
{
    let mut stmt = conn.prepare(sql).map_err(prepare_error)?;
    let bound = bind(params);
    let mut rows = stmt.query(bound.as_slice()).map_err(query_error)?;
    let Some(row) = rows.next().map_err(&mut read_error)? else {
        return Ok(None);
    };
    Ok(Some(hydrate(row).map_err(&mut read_error)?))
}

/// Prepare `sql`, bind `params`, execute, and hydrate at most one row using a
/// source-local SQL label for phase-specific diagnostics.
pub fn query_optional_row_labeled<T, E, Label, Hydrate>(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
    label: Label,
    hydrate: Hydrate,
) -> Result<Option<T>, E>
where
    Label: SqlErrorMapper<E>,
    Hydrate: FnOnce(&duckdb::Row<'_>) -> Result<T, duckdb::Error>,
{
    query_optional_row(
        conn,
        sql,
        params,
        |source| label.prepare_error(source),
        |source| label.query_error(source),
        |source| label.read_error(source),
        hydrate,
    )
}

/// Prepare `sql`, bind `params`, execute, and hydrate at most one row while
/// allowing the hydrator to return caller-domain errors.
///
/// Use this for single-row probes whose row-to-domain conversion can fail for
/// reasons beyond DuckDB column reads.
fn query_optional_row_fallible<T, E, PrepareError, QueryError, ReadError, Hydrate>(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
    prepare_error: PrepareError,
    query_error: QueryError,
    mut read_error: ReadError,
    hydrate: Hydrate,
) -> Result<Option<T>, E>
where
    PrepareError: FnOnce(duckdb::Error) -> E,
    QueryError: FnOnce(duckdb::Error) -> E,
    ReadError: FnMut(duckdb::Error) -> E,
    Hydrate: FnOnce(&duckdb::Row<'_>) -> Result<T, E>,
{
    let mut stmt = conn.prepare(sql).map_err(prepare_error)?;
    let bound = bind(params);
    let mut rows = stmt.query(bound.as_slice()).map_err(query_error)?;
    let Some(row) = rows.next().map_err(&mut read_error)? else {
        return Ok(None);
    };
    Ok(Some(hydrate(row)?))
}

/// Prepare `sql`, bind `params`, execute, and hydrate at most one row while
/// preserving caller-domain hydrate errors and using a source-local SQL label.
pub fn query_optional_row_fallible_labeled<T, E, Label, Hydrate>(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
    label: Label,
    hydrate: Hydrate,
) -> Result<Option<T>, E>
where
    Label: SqlErrorMapper<E>,
    Hydrate: FnOnce(&duckdb::Row<'_>) -> Result<T, E>,
{
    query_optional_row_fallible(
        conn,
        sql,
        params,
        |source| label.prepare_error(source),
        |source| label.query_error(source),
        |source| label.read_error(source),
        hydrate,
    )
}

/// Prepare `sql`, bind `params`, execute, and hydrate all rows while allowing
/// the hydrator to return caller-domain errors.
///
/// Use this when row hydration can fail for reasons beyond DuckDB column reads
/// and those errors should not be wrapped as read-phase SQL diagnostics.
fn query_rows_fallible<T, E, PrepareError, QueryError, ReadError, Hydrate>(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
    prepare_error: PrepareError,
    query_error: QueryError,
    read_error: ReadError,
    hydrate: Hydrate,
) -> Result<Vec<T>, E>
where
    PrepareError: FnOnce(duckdb::Error) -> E,
    QueryError: FnOnce(duckdb::Error) -> E,
    ReadError: FnMut(duckdb::Error) -> E,
    Hydrate: FnMut(&duckdb::Row<'_>) -> Result<T, E>,
{
    let mut stmt = conn.prepare(sql).map_err(prepare_error)?;
    let bound = bind(params);
    let mut rows = stmt.query(bound.as_slice()).map_err(query_error)?;
    collect_rows_fallible(&mut rows, read_error, hydrate)
}

/// Prepare `sql`, bind `params`, execute, and hydrate all rows while
/// preserving caller-domain hydrate errors and using a source-local SQL label.
pub fn query_rows_fallible_labeled<T, E, Label, Hydrate>(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
    label: Label,
    hydrate: Hydrate,
) -> Result<Vec<T>, E>
where
    Label: SqlErrorMapper<E>,
    Hydrate: FnMut(&duckdb::Row<'_>) -> Result<T, E>,
{
    query_rows_fallible(
        conn,
        sql,
        params,
        |source| label.prepare_error(source),
        |source| label.query_error(source),
        |source| label.read_error(source),
        hydrate,
    )
}

/// Prepare `sql`, bind `params`, execute, load one caller-provided context,
/// and hydrate all rows.
///
/// This keeps call sites that need a one-time sidecar/context load after
/// DuckDB query setup but before row reads on the same shared execution path as
/// [`query_rows`].
#[expect(
    clippy::too_many_arguments,
    reason = "callers pass source-specific prepare/query/read mappers plus a context loader and row hydrator"
)]
fn query_rows_with_context<T, C, E, PrepareError, QueryError, ReadError, LoadContext, Hydrate>(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
    prepare_error: PrepareError,
    query_error: QueryError,
    mut read_error: ReadError,
    load_context: LoadContext,
    mut hydrate: Hydrate,
) -> Result<(Vec<T>, C), E>
where
    PrepareError: FnOnce(duckdb::Error) -> E,
    QueryError: FnOnce(duckdb::Error) -> E,
    ReadError: FnMut(duckdb::Error) -> E,
    LoadContext: FnOnce() -> Result<C, E>,
    Hydrate: FnMut(&duckdb::Row<'_>, &C) -> Result<T, duckdb::Error>,
{
    let mut stmt = conn.prepare(sql).map_err(prepare_error)?;
    let bound = bind(params);
    let mut rows = stmt.query(bound.as_slice()).map_err(query_error)?;
    let context = load_context()?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().map_err(&mut read_error)? {
        out.push(hydrate(row, &context).map_err(&mut read_error)?);
    }
    Ok((out, context))
}

/// Prepare `sql`, bind `params`, execute, load one caller-provided context,
/// and hydrate all rows using a source-local SQL label.
pub fn query_rows_with_context_labeled<T, C, E, Label, LoadContext, Hydrate>(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
    label: Label,
    load_context: LoadContext,
    hydrate: Hydrate,
) -> Result<(Vec<T>, C), E>
where
    Label: SqlErrorMapper<E>,
    LoadContext: FnOnce() -> Result<C, E>,
    Hydrate: FnMut(&duckdb::Row<'_>, &C) -> Result<T, duckdb::Error>,
{
    query_rows_with_context(
        conn,
        sql,
        params,
        |source| label.prepare_error(source),
        |source| label.query_error(source),
        |source| label.read_error(source),
        load_context,
        hydrate,
    )
}

/// Open an in-memory DuckDB connection, then prepare, execute, and hydrate all
/// rows through [`query_rows`].
fn query_rows_in_memory<T, E, OpenError, PrepareError, QueryError, ReadError, Hydrate>(
    sql: &str,
    params: &[Value],
    open_error: OpenError,
    prepare_error: PrepareError,
    query_error: QueryError,
    read_error: ReadError,
    hydrate: Hydrate,
) -> Result<Vec<T>, E>
where
    OpenError: FnOnce(duckdb::Error) -> E,
    PrepareError: FnOnce(duckdb::Error) -> E,
    QueryError: FnOnce(duckdb::Error) -> E,
    ReadError: FnMut(duckdb::Error) -> E,
    Hydrate: FnMut(&duckdb::Row<'_>) -> Result<T, duckdb::Error>,
{
    let conn = duckdb::Connection::open_in_memory().map_err(open_error)?;
    query_rows(
        &conn,
        sql,
        params,
        prepare_error,
        query_error,
        read_error,
        hydrate,
    )
}

/// Open an in-memory DuckDB connection, then prepare, execute, and hydrate all
/// rows using a source-local SQL label.
pub fn query_rows_in_memory_labeled<T, E, OpenError, Label, Hydrate>(
    sql: &str,
    params: &[Value],
    open_error: OpenError,
    label: Label,
    hydrate: Hydrate,
) -> Result<Vec<T>, E>
where
    OpenError: FnOnce(duckdb::Error) -> E,
    Label: SqlErrorMapper<E>,
    Hydrate: FnMut(&duckdb::Row<'_>) -> Result<T, duckdb::Error>,
{
    query_rows_in_memory(
        sql,
        params,
        open_error,
        |source| label.prepare_error(source),
        |source| label.query_error(source),
        |source| label.read_error(source),
        hydrate,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Eq)]
    struct LabelError {
        phase: &'static str,
        label: &'static str,
    }

    #[derive(Debug, Clone, Copy)]
    struct TestLabel(&'static str);

    impl SqlErrorMapper<LabelError> for TestLabel {
        fn prepare_error(self, _source: duckdb::Error) -> LabelError {
            LabelError {
                phase: "prepare",
                label: self.0,
            }
        }

        fn query_error(self, _source: duckdb::Error) -> LabelError {
            LabelError {
                phase: "query",
                label: self.0,
            }
        }

        fn read_error(self, _source: duckdb::Error) -> LabelError {
            LabelError {
                phase: "read",
                label: self.0,
            }
        }
    }

    #[test]
    fn query_rows_maps_prepare_query_read_and_hydrates_rows() -> anyhow::Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = "SELECT CAST(? AS BIGINT) + 1 AS value";
        let params = [Value::BigInt(41)];
        let rows = query_rows(
            &conn,
            sql,
            &params,
            anyhow::Error::new,
            anyhow::Error::new,
            anyhow::Error::new,
            |row| row.get::<_, i64>("value"),
        )?;

        assert_eq!(rows, vec![42]);
        Ok(())
    }

    #[test]
    fn query_rows_labeled_maps_phase_and_label() -> anyhow::Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let err = match query_rows_labeled(
            &conn,
            "SELECT * FROM",
            &[],
            TestLabel("semantic-label"),
            |_| Ok(0i64),
        ) {
            Ok(rows) => anyhow::bail!("malformed SQL should not hydrate: {rows:?}"),
            Err(err) => err,
        };

        assert_eq!(
            err,
            LabelError {
                phase: "prepare",
                label: "semantic-label"
            }
        );
        Ok(())
    }

    #[test]
    fn query_rows_with_context_loads_context_before_hydrating_rows() -> anyhow::Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let (rows, context) = query_rows_with_context(
            &conn,
            "SELECT CAST(? AS BIGINT) AS value",
            &[Value::BigInt(4)],
            anyhow::Error::new,
            anyhow::Error::new,
            anyhow::Error::new,
            || Ok::<_, anyhow::Error>(10i64),
            |row, context| {
                let value = row.get::<_, i64>("value")?;
                Ok(value + *context)
            },
        )?;

        assert_eq!(context, 10);
        assert_eq!(rows, vec![14]);
        Ok(())
    }

    #[test]
    fn query_rows_prepared_reuses_statement_for_different_params() -> anyhow::Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let mut stmt = conn.prepare("SELECT CAST(? AS BIGINT) + 1 AS value")?;

        let first = query_rows_prepared(
            &mut stmt,
            &[Value::BigInt(1)],
            anyhow::Error::new,
            anyhow::Error::new,
            |row| row.get::<_, i64>("value"),
        )?;
        let second = query_rows_prepared(
            &mut stmt,
            &[Value::BigInt(4)],
            anyhow::Error::new,
            anyhow::Error::new,
            |row| row.get::<_, i64>("value"),
        )?;

        assert_eq!(first, vec![2]);
        assert_eq!(second, vec![5]);
        Ok(())
    }

    #[test]
    fn query_optional_row_hydrates_first_or_none() -> anyhow::Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let rows = query_optional_row(
            &conn,
            "SELECT CAST(? AS BIGINT) AS value UNION ALL SELECT 99",
            &[Value::BigInt(3)],
            anyhow::Error::new,
            anyhow::Error::new,
            anyhow::Error::new,
            |row| row.get::<_, i64>("value"),
        )?;
        let none = query_optional_row(
            &conn,
            "SELECT CAST(? AS BIGINT) AS value WHERE FALSE",
            &[Value::BigInt(3)],
            anyhow::Error::new,
            anyhow::Error::new,
            anyhow::Error::new,
            |row| row.get::<_, i64>("value"),
        )?;

        assert_eq!(rows, Some(3));
        assert_eq!(none, None);
        Ok(())
    }

    #[test]
    fn query_optional_row_fallible_preserves_hydrate_domain_error() -> anyhow::Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let err = match query_optional_row_fallible(
            &conn,
            "SELECT CAST(? AS BIGINT) AS value",
            &[Value::BigInt(1)],
            anyhow::Error::new,
            anyhow::Error::new,
            anyhow::Error::new,
            |row| {
                let value = row.get::<_, i64>("value").map_err(anyhow::Error::new)?;
                if value == 1 {
                    anyhow::bail!("custom optional hydrate error");
                }
                Ok(value)
            },
        ) {
            Ok(row) => anyhow::bail!("custom optional hydrate error should fail: {row:?}"),
            Err(err) => err,
        };

        assert_eq!(err.to_string(), "custom optional hydrate error");
        Ok(())
    }

    #[test]
    fn query_rows_fallible_preserves_hydrate_domain_error() -> anyhow::Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let err = match query_rows_fallible(
            &conn,
            "SELECT CAST(? AS BIGINT) AS value",
            &[Value::BigInt(1)],
            anyhow::Error::new,
            anyhow::Error::new,
            anyhow::Error::new,
            |row| {
                let value = row.get::<_, i64>("value").map_err(anyhow::Error::new)?;
                if value == 1 {
                    anyhow::bail!("custom hydrate error");
                }
                Ok(value)
            },
        ) {
            Ok(rows) => anyhow::bail!("custom hydrate error should fail: {rows:?}"),
            Err(err) => err,
        };

        assert_eq!(err.to_string(), "custom hydrate error");
        Ok(())
    }

    #[test]
    fn query_rows_in_memory_opens_and_hydrates_rows() -> anyhow::Result<()> {
        let rows = query_rows_in_memory(
            "SELECT CAST(? AS BIGINT) AS value",
            &[Value::BigInt(7)],
            anyhow::Error::new,
            anyhow::Error::new,
            anyhow::Error::new,
            anyhow::Error::new,
            |row| row.get::<_, i64>("value"),
        )?;

        assert_eq!(rows, vec![7]);
        Ok(())
    }
}
