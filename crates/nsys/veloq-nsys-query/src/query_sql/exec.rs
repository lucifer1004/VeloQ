use crate::{NsysQueryError, NsysQueryResult};
use duckdb::types::Value;
use veloq_query::duckdb::exec as duckdb_exec;

#[derive(Debug, Clone, Copy)]
pub(crate) struct SqlLabel {
    area: &'static str,
    label: &'static str,
}

impl SqlLabel {
    pub(crate) const fn new(area: &'static str, label: &'static str) -> Self {
        Self { area, label }
    }
}

impl duckdb_exec::SqlErrorMapper<NsysQueryError> for SqlLabel {
    fn prepare_error(self, source: duckdb::Error) -> NsysQueryError {
        NsysQueryError::sql_prepare(self.area, self.label, source)
    }

    fn query_error(self, source: duckdb::Error) -> NsysQueryError {
        NsysQueryError::sql_query(self.area, self.label, source)
    }

    fn read_error(self, source: duckdb::Error) -> NsysQueryError {
        NsysQueryError::sql_read(self.area, self.label, source)
    }
}

pub(crate) const CONCURRENCY_INTERVAL: SqlLabel = SqlLabel::new("concurrency", "interval");
pub(crate) const GAPS_GAP: SqlLabel = SqlLabel::new("gaps", "gap");
pub(crate) const GAPS_NAME_LOOKUP: SqlLabel = SqlLabel::new("gaps", "name-lookup");
pub(crate) const GAPS_STREAM_ACTIVITY: SqlLabel = SqlLabel::new("gaps", "stream-activity");
pub(crate) const GRAPH_REPLAYS_LAUNCHER_LOOKUP: SqlLabel =
    SqlLabel::new("graph-replays", "launcher lookup");
pub(crate) const GRAPH_REPLAYS_NODE_EVENT: SqlLabel = SqlLabel::new("graph-replays", "node-event");
pub(crate) const GRAPH_REPLAYS_REPLAY_SUMMARY: SqlLabel =
    SqlLabel::new("graph-replays", "replay-summary");
pub(crate) const NVTX_REVERSE_COLD_FALLBACK: SqlLabel =
    SqlLabel::new("reverse NVTX attribution", "cold fallback");
pub(crate) const NVTX_REVERSE_GPU_LOOKUP: SqlLabel =
    SqlLabel::new("reverse NVTX attribution", "GPU lookup");
pub(crate) const STATS_AGGREGATE: SqlLabel = SqlLabel::new("stats", "aggregate");
pub(crate) const STATS_BY_SIZE_AGGREGATE: SqlLabel = SqlLabel::new("stats-by-size", "aggregate");
pub(crate) const TIMELINE_AGGREGATE: SqlLabel = SqlLabel::new("timeline", "aggregate");

pub(crate) fn query_rows<T>(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
    label: SqlLabel,
    hydrate: impl FnMut(&duckdb::Row<'_>) -> Result<T, duckdb::Error>,
) -> NsysQueryResult<Vec<T>> {
    duckdb_exec::query_rows_labeled(conn, sql, params, label, hydrate)
}

pub(crate) fn query_rows_prepared<T>(
    stmt: &mut duckdb::Statement<'_>,
    params: &[Value],
    label: SqlLabel,
    hydrate: impl FnMut(&duckdb::Row<'_>) -> Result<T, duckdb::Error>,
) -> NsysQueryResult<Vec<T>> {
    duckdb_exec::query_rows_prepared_labeled(stmt, params, label, hydrate)
}

pub(crate) fn query_optional_row<T>(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
    label: SqlLabel,
    hydrate: impl FnOnce(&duckdb::Row<'_>) -> Result<T, duckdb::Error>,
) -> NsysQueryResult<Option<T>> {
    duckdb_exec::query_optional_row_labeled(conn, sql, params, label, hydrate)
}

pub(crate) fn query_rows_fallible<T>(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
    label: SqlLabel,
    hydrate: impl FnMut(&duckdb::Row<'_>) -> NsysQueryResult<T>,
) -> NsysQueryResult<Vec<T>> {
    duckdb_exec::query_rows_fallible_labeled(conn, sql, params, label, hydrate)
}

pub(crate) fn query_optional_row_fallible<T>(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
    label: SqlLabel,
    hydrate: impl FnOnce(&duckdb::Row<'_>) -> NsysQueryResult<T>,
) -> NsysQueryResult<Option<T>> {
    duckdb_exec::query_optional_row_fallible_labeled(conn, sql, params, label, hydrate)
}

pub(crate) fn query_rows_with_context<T, C>(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
    label: SqlLabel,
    load_context: impl FnOnce() -> NsysQueryResult<C>,
    hydrate: impl FnMut(&duckdb::Row<'_>, &C) -> Result<T, duckdb::Error>,
) -> NsysQueryResult<(Vec<T>, C)> {
    duckdb_exec::query_rows_with_context_labeled(conn, sql, params, label, load_context, hydrate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use veloq_core::VeloqDiagnostic;

    #[test]
    fn query_rows_prepare_error_uses_label() -> anyhow::Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let label = SqlLabel::new("test-area", "test-label");

        let err = match query_rows(&conn, "SELECT * FROM", &[], label, |_| Ok(0i64)) {
            Ok(rows) => anyhow::bail!("malformed SQL should not hydrate: {rows:?}"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-prepare");
        assert_eq!(
            err.sql_parts(),
            Some(("test-area", crate::SqlPhase::Prepare, "test-label"))
        );
        Ok(())
    }

    #[test]
    fn query_rows_read_error_uses_label() -> anyhow::Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let label = SqlLabel::new("test-area", "test-label");

        let err = match query_rows(&conn, "SELECT 'not-an-int' AS value", &[], label, |row| {
            row.get::<_, i64>(0)
        }) {
            Ok(rows) => anyhow::bail!("malformed row should not hydrate: {rows:?}"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-read");
        assert_eq!(
            err.sql_parts(),
            Some(("test-area", crate::SqlPhase::Read, "test-label"))
        );
        Ok(())
    }
}
