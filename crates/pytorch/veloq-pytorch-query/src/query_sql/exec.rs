use crate::{PytorchQueryError, PytorchQueryResult};
use duckdb::types::Value;
use veloq_query::duckdb::exec as duckdb_exec;

#[derive(Debug, Clone, Copy)]
pub(crate) enum SqlVerb {
    Collectives,
    Correlate,
    Inspect,
    Search,
    Stats,
    Timeline,
    Slices,
}

impl SqlVerb {
    fn area(self) -> &'static str {
        match self {
            Self::Collectives => "collectives",
            Self::Correlate => "correlate",
            Self::Inspect => "inspect",
            Self::Search => "search",
            Self::Stats => "stats",
            Self::Timeline => "timeline",
            Self::Slices => "slices",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SqlLabel<'a> {
    verb: SqlVerb,
    label: &'a str,
}

impl<'a> SqlLabel<'a> {
    pub(crate) const fn new(verb: SqlVerb, label: &'a str) -> Self {
        Self { verb, label }
    }
}

impl duckdb_exec::SqlErrorMapper<PytorchQueryError> for SqlLabel<'_> {
    fn prepare_error(self, source: duckdb::Error) -> PytorchQueryError {
        PytorchQueryError::sql_prepare(self.verb.area(), self.label, source)
    }

    fn query_error(self, source: duckdb::Error) -> PytorchQueryError {
        PytorchQueryError::sql_query(self.verb.area(), self.label, source)
    }

    fn read_error(self, source: duckdb::Error) -> PytorchQueryError {
        PytorchQueryError::sql_read(self.verb.area(), self.label, source)
    }
}

pub(crate) fn open_connection() -> PytorchQueryResult<duckdb::Connection> {
    duckdb::Connection::open_in_memory().map_err(PytorchQueryError::sql_open)
}

pub(crate) fn query_rows_on<T>(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
    label: SqlLabel<'_>,
    hydrate: impl FnMut(&duckdb::Row<'_>) -> Result<T, duckdb::Error>,
) -> PytorchQueryResult<Vec<T>> {
    duckdb_exec::query_rows_labeled(conn, sql, params, label, hydrate)
}

pub(crate) fn query_optional_row_on<T>(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
    label: SqlLabel<'_>,
    hydrate: impl FnOnce(&duckdb::Row<'_>) -> Result<T, duckdb::Error>,
) -> PytorchQueryResult<Option<T>> {
    duckdb_exec::query_optional_row_labeled(conn, sql, params, label, hydrate)
}

pub(crate) fn query_rows<T>(
    sql: &str,
    params: &[Value],
    label: SqlLabel<'_>,
    hydrate: impl FnMut(&duckdb::Row<'_>) -> Result<T, duckdb::Error>,
) -> PytorchQueryResult<Vec<T>> {
    duckdb_exec::query_rows_in_memory_labeled(
        sql,
        params,
        PytorchQueryError::sql_open,
        label,
        hydrate,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use veloq_core::VeloqDiagnostic;

    #[test]
    fn query_rows_prepare_error_uses_label_verb() -> Result<(), Box<dyn std::error::Error>> {
        let err = match query_rows(
            "SELECT * FROM",
            &[],
            SqlLabel::new(SqlVerb::Search, "events"),
            |_| Ok(0i64),
        ) {
            Ok(rows) => return Err(format!("malformed SQL should not hydrate: {rows:?}").into()),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "pytorch.query.sql-prepare");
        assert_eq!(err.to_string(), "preparing pytorch search SQL");
        assert_eq!(
            err.sql_parts(),
            Some(("search", crate::SqlPhase::Prepare, "events"))
        );
        Ok(())
    }

    #[test]
    fn query_rows_read_error_uses_label_verb() -> Result<(), Box<dyn std::error::Error>> {
        let err = match query_rows(
            "SELECT 'not-an-int' AS value",
            &[],
            SqlLabel::new(SqlVerb::Timeline, "aggregate"),
            |row| row.get::<_, i64>(0),
        ) {
            Ok(rows) => return Err(format!("malformed row should not hydrate: {rows:?}").into()),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "pytorch.query.sql-read");
        assert_eq!(err.to_string(), "reading pytorch timeline SQL row");
        assert_eq!(
            err.sql_parts(),
            Some(("timeline", crate::SqlPhase::Read, "aggregate"))
        );
        Ok(())
    }
}
