//! `inspect kernel:N` / `memcpy:N` / `memset:N` — GPU-work rows.
//!
//! Three CUPTI activity tables share the same per-row shape: device,
//! context, stream, optional graph identity, and a `correlationId`
//! that links back to the launching `CUPTI_ACTIVITY_KIND_RUNTIME`
//! row. Kernels carry grid/block geometry and shared-memory sizing;
//! memcpy/memset add `bytes` plus a copy/value field.

use crate::{NsysQueryResult, NvtxContext, RowId};
use duckdb::Connection;
use serde::Serialize;

use super::{ColumnMap, EventDetails, maybe_col, opt_string, query_inspect_row};

const INSPECT_KERNEL_SQL: &str = "kernel";
const INSPECT_MEMCPY_SQL: &str = "memcpy";
const INSPECT_MEMSET_SQL: &str = "memset";

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct KernelDetails {
    /// Cross-trace key — equal to `row_id` stringified.
    pub key: String,
    pub row_id: RowId,
    pub short_name: Option<String>,
    pub demangled_name: Option<String>,
    pub start_ns: i64,
    pub end_ns: i64,
    pub duration_ns: i64,
    pub device_id: i32,
    pub context_id: i64,
    pub stream_id: i64,
    pub grid: [i64; 3],
    pub block: [i64; 3],
    pub registers_per_thread: Option<i64>,
    pub static_shared_memory: Option<i64>,
    pub dynamic_shared_memory: Option<i64>,
    pub correlation_id: Option<i64>,
    pub global_pid: Option<i64>,
    /// Captured-graph id when the kernel ran inside a CUDA graph
    /// (`--cuda-graph-trace=node` captures); `None` for eager kernels
    /// and for traces where the column is absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_id: Option<i64>,
    /// Node id within the captured graph; joins to `graph_node:<id>`
    /// via `CUDA_GRAPH_NODE_EVENTS.graphNodeId`. `None` for eager
    /// kernels.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_node_id: Option<i64>,
    /// Innermost NVTX range that was open on the launching host
    /// thread when the runtime call corresponding to this kernel
    /// fired. Populated automatically when the trace has NVTX +
    /// CUPTI_RUNTIME tables; `None` everywhere else.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_context: Option<NvtxContext>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct MemcpyDetails {
    /// Cross-trace key — equal to `row_id` stringified.
    pub key: String,
    pub row_id: RowId,
    pub start_ns: i64,
    pub end_ns: i64,
    pub duration_ns: i64,
    pub device_id: i32,
    pub context_id: i64,
    pub stream_id: i64,
    pub bytes: i64,
    pub copy_kind: i64,
    pub copy_kind_name: &'static str,
    pub correlation_id: Option<i64>,
    /// Node id when the memcpy ran inside a CUDA graph
    /// (`--cuda-graph-trace=node`). Memcpy/memset tables don't carry
    /// `graphId` directly — join `CUDA_GRAPH_NODE_EVENTS` or the
    /// kernel table on this id if you need the parent graph.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_node_id: Option<i64>,
    /// See [`KernelDetails::nvtx_context`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_context: Option<NvtxContext>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct MemsetDetails {
    /// Cross-trace key — equal to `row_id` stringified.
    pub key: String,
    pub row_id: RowId,
    pub start_ns: i64,
    pub end_ns: i64,
    pub duration_ns: i64,
    pub device_id: i32,
    pub context_id: i64,
    pub stream_id: i64,
    pub bytes: i64,
    pub value: Option<i64>,
    pub correlation_id: Option<i64>,
    /// Node id when the memset ran inside a CUDA graph
    /// (`--cuda-graph-trace=node`). See [`MemcpyDetails::graph_node_id`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_node_id: Option<i64>,
    /// See [`KernelDetails::nvtx_context`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_context: Option<NvtxContext>,
}

pub(super) fn query_kernel(
    conn: &Connection,
    cols: &ColumnMap,
    id: RowId,
) -> NsysQueryResult<Option<EventDetails>> {
    const T: &str = "CUPTI_ACTIVITY_KIND_KERNEL";
    if !cols.contains_key(T) {
        return Ok(None);
    }
    let reg = maybe_col(cols, T, "registersPerThread");
    let smem_static = maybe_col(cols, T, "staticSharedMemory");
    let smem_dyn = maybe_col(cols, T, "dynamicSharedMemory");
    let corr = maybe_col(cols, T, "correlationId");
    let gpid = maybe_col(cols, T, "globalPid");
    let gid = maybe_col(cols, T, "graphId");
    let gnid = maybe_col(cols, T, "graphNodeId");
    let dev = crate::kind_sql::GPU_DEVICE_ID_EXPR;
    let ctx = crate::kind_sql::GPU_CONTEXT_ID_EXPR;
    let stm = crate::kind_sql::GPU_STREAM_ID_EXPR;

    let sql = format!(
        r#"
        SELECT
            t.start, t."end",
            {dev},
            {ctx},
            {stm},
            CAST(t.gridX AS BIGINT), CAST(t.gridY AS BIGINT), CAST(t.gridZ AS BIGINT),
            CAST(t.blockX AS BIGINT), CAST(t.blockY AS BIGINT), CAST(t.blockZ AS BIGINT),
            CAST({reg} AS BIGINT),
            CAST({smem_static} AS BIGINT),
            CAST({smem_dyn} AS BIGINT),
            CAST({corr} AS BIGINT),
            CAST({gpid} AS BIGINT),
            s_sh.value,
            s_dem.value,
            CAST({gid} AS BIGINT),
            CAST({gnid} AS BIGINT)
        FROM nsight.{T} t
        LEFT JOIN nsight.StringIds s_sh  ON t.shortName = s_sh.id
        LEFT JOIN nsight.StringIds s_dem ON t.demangledName = s_dem.id
        WHERE t.rowid = ?
        "#
    );
    query_kernel_with_sql(conn, id, &sql)
}

fn query_kernel_with_sql(
    conn: &Connection,
    id: RowId,
    sql: &str,
) -> NsysQueryResult<Option<EventDetails>> {
    query_inspect_row(conn, INSPECT_KERNEL_SQL, sql, id, |r| {
        let start_ns: i64 = map_inspect_read(INSPECT_KERNEL_SQL, r.get(0))?;
        let end_ns: i64 = map_inspect_read(INSPECT_KERNEL_SQL, r.get(1))?;
        Ok(EventDetails::Kernel(KernelDetails {
            key: id.to_string(),
            row_id: id,
            short_name: map_inspect_read(INSPECT_KERNEL_SQL, opt_string(r, 16))?,
            demangled_name: map_inspect_read(INSPECT_KERNEL_SQL, opt_string(r, 17))?,
            start_ns,
            end_ns,
            duration_ns: end_ns - start_ns,
            device_id: map_inspect_read(INSPECT_KERNEL_SQL, r.get(2))?,
            context_id: map_inspect_read(INSPECT_KERNEL_SQL, r.get(3))?,
            stream_id: map_inspect_read(INSPECT_KERNEL_SQL, r.get(4))?,
            grid: [
                map_inspect_read(INSPECT_KERNEL_SQL, r.get(5))?,
                map_inspect_read(INSPECT_KERNEL_SQL, r.get(6))?,
                map_inspect_read(INSPECT_KERNEL_SQL, r.get(7))?,
            ],
            block: [
                map_inspect_read(INSPECT_KERNEL_SQL, r.get(8))?,
                map_inspect_read(INSPECT_KERNEL_SQL, r.get(9))?,
                map_inspect_read(INSPECT_KERNEL_SQL, r.get(10))?,
            ],
            registers_per_thread: map_inspect_read(INSPECT_KERNEL_SQL, r.get(11))?,
            static_shared_memory: map_inspect_read(INSPECT_KERNEL_SQL, r.get(12))?,
            dynamic_shared_memory: map_inspect_read(INSPECT_KERNEL_SQL, r.get(13))?,
            correlation_id: map_inspect_read(INSPECT_KERNEL_SQL, r.get(14))?,
            global_pid: map_inspect_read(INSPECT_KERNEL_SQL, r.get(15))?,
            graph_id: map_inspect_read(INSPECT_KERNEL_SQL, r.get(18))?,
            graph_node_id: map_inspect_read(INSPECT_KERNEL_SQL, r.get(19))?,
            nvtx_context: None,
        }))
    })
}

pub(super) fn query_memcpy(
    conn: &Connection,
    cols: &ColumnMap,
    id: RowId,
) -> NsysQueryResult<Option<EventDetails>> {
    const T: &str = "CUPTI_ACTIVITY_KIND_MEMCPY";
    if !cols.contains_key(T) {
        return Ok(None);
    }
    let corr = maybe_col(cols, T, "correlationId");
    let gnid = maybe_col(cols, T, "graphNodeId");
    let dev = crate::kind_sql::GPU_DEVICE_ID_EXPR;
    let ctx = crate::kind_sql::GPU_CONTEXT_ID_EXPR;
    let stm = crate::kind_sql::GPU_STREAM_ID_EXPR;
    let sql = format!(
        r#"
        SELECT
            t.start, t."end",
            {dev},
            {ctx},
            {stm},
            CAST(t.bytes AS BIGINT),
            CAST(t.copyKind AS BIGINT),
            CAST({corr} AS BIGINT),
            CAST({gnid} AS BIGINT)
        FROM nsight.{T} t
        WHERE t.rowid = ?
        "#
    );
    query_memcpy_with_sql(conn, id, &sql)
}

fn query_memcpy_with_sql(
    conn: &Connection,
    id: RowId,
    sql: &str,
) -> NsysQueryResult<Option<EventDetails>> {
    query_inspect_row(conn, INSPECT_MEMCPY_SQL, sql, id, |r| {
        let start_ns: i64 = map_inspect_read(INSPECT_MEMCPY_SQL, r.get(0))?;
        let end_ns: i64 = map_inspect_read(INSPECT_MEMCPY_SQL, r.get(1))?;
        let copy_kind: i64 = map_inspect_read(INSPECT_MEMCPY_SQL, r.get(6))?;
        Ok(EventDetails::Memcpy(MemcpyDetails {
            key: id.to_string(),
            row_id: id,
            start_ns,
            end_ns,
            duration_ns: end_ns - start_ns,
            device_id: map_inspect_read(INSPECT_MEMCPY_SQL, r.get(2))?,
            context_id: map_inspect_read(INSPECT_MEMCPY_SQL, r.get(3))?,
            stream_id: map_inspect_read(INSPECT_MEMCPY_SQL, r.get(4))?,
            bytes: map_inspect_read(INSPECT_MEMCPY_SQL, r.get(5))?,
            copy_kind,
            copy_kind_name: crate::kind_sql::copy_kind_label(copy_kind),
            correlation_id: map_inspect_read(INSPECT_MEMCPY_SQL, r.get(7))?,
            graph_node_id: map_inspect_read(INSPECT_MEMCPY_SQL, r.get(8))?,
            nvtx_context: None,
        }))
    })
}

pub(super) fn query_memset(
    conn: &Connection,
    cols: &ColumnMap,
    id: RowId,
) -> NsysQueryResult<Option<EventDetails>> {
    const T: &str = "CUPTI_ACTIVITY_KIND_MEMSET";
    if !cols.contains_key(T) {
        return Ok(None);
    }
    let val = maybe_col(cols, T, "value");
    let corr = maybe_col(cols, T, "correlationId");
    let gnid = maybe_col(cols, T, "graphNodeId");
    let dev = crate::kind_sql::GPU_DEVICE_ID_EXPR;
    let ctx = crate::kind_sql::GPU_CONTEXT_ID_EXPR;
    let stm = crate::kind_sql::GPU_STREAM_ID_EXPR;
    let sql = format!(
        r#"
        SELECT
            t.start, t."end",
            {dev},
            {ctx},
            {stm},
            CAST(t.bytes AS BIGINT),
            CAST({val} AS BIGINT),
            CAST({corr} AS BIGINT),
            CAST({gnid} AS BIGINT)
        FROM nsight.{T} t
        WHERE t.rowid = ?
        "#
    );
    query_memset_with_sql(conn, id, &sql)
}

fn query_memset_with_sql(
    conn: &Connection,
    id: RowId,
    sql: &str,
) -> NsysQueryResult<Option<EventDetails>> {
    query_inspect_row(conn, INSPECT_MEMSET_SQL, sql, id, |r| {
        let start_ns: i64 = map_inspect_read(INSPECT_MEMSET_SQL, r.get(0))?;
        let end_ns: i64 = map_inspect_read(INSPECT_MEMSET_SQL, r.get(1))?;
        Ok(EventDetails::Memset(MemsetDetails {
            key: id.to_string(),
            row_id: id,
            start_ns,
            end_ns,
            duration_ns: end_ns - start_ns,
            device_id: map_inspect_read(INSPECT_MEMSET_SQL, r.get(2))?,
            context_id: map_inspect_read(INSPECT_MEMSET_SQL, r.get(3))?,
            stream_id: map_inspect_read(INSPECT_MEMSET_SQL, r.get(4))?,
            bytes: map_inspect_read(INSPECT_MEMSET_SQL, r.get(5))?,
            value: map_inspect_read(INSPECT_MEMSET_SQL, r.get(6))?,
            correlation_id: map_inspect_read(INSPECT_MEMSET_SQL, r.get(7))?,
            graph_node_id: map_inspect_read(INSPECT_MEMSET_SQL, r.get(8))?,
            nvtx_context: None,
        }))
    })
}

fn map_inspect_read<T>(
    kind: &'static str,
    result: std::result::Result<T, duckdb::Error>,
) -> NsysQueryResult<T> {
    result.map_err(|source| crate::NsysQueryError::sql_read("inspect", kind, source))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use veloq_core::VeloqDiagnostic;

    fn id(kind: crate::EventKind) -> RowId {
        RowId::new(kind, 1)
    }

    fn assert_error(
        result: NsysQueryResult<Option<EventDetails>>,
        expected_code: &str,
        expected_kind: &'static str,
    ) -> Result<crate::NsysQueryError> {
        let err = match result {
            Ok(_) => anyhow::bail!("malformed inspect SQL should not succeed"),
            Err(err) => err,
        };
        assert_eq!(err.code().as_str(), expected_code);
        let Some((area, _, label)) = err.sql_parts() else {
            anyhow::bail!("expected inspect SQL error, got {err:?}");
        };
        assert_eq!(area, "inspect");
        assert_eq!(label, expected_kind);
        Ok(err)
    }

    #[test]
    fn query_kernel_prepare_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;

        let err = assert_error(
            query_kernel_with_sql(&conn, id(crate::EventKind::Kernel), "SELECT * FROM"),
            "nsys.query.sql-prepare",
            INSPECT_KERNEL_SQL,
        )?;

        assert!(matches!(
            err,
            crate::NsysQueryError::Sql {
                phase: crate::SqlPhase::Prepare,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn query_kernel_query_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = "SELECT ? AS start_ns WHERE ? IS NOT NULL";

        let err = assert_error(
            query_kernel_with_sql(&conn, id(crate::EventKind::Kernel), sql),
            "nsys.query.sql-query",
            INSPECT_KERNEL_SQL,
        )?;

        assert!(matches!(
            err,
            crate::NsysQueryError::Sql {
                phase: crate::SqlPhase::Query,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn query_kernel_read_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = "SELECT 'not-a-start' AS start_ns WHERE ? IS NOT NULL";

        let err = assert_error(
            query_kernel_with_sql(&conn, id(crate::EventKind::Kernel), sql),
            "nsys.query.sql-read",
            INSPECT_KERNEL_SQL,
        )?;

        assert!(matches!(
            err,
            crate::NsysQueryError::Sql {
                phase: crate::SqlPhase::Read,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn query_memcpy_prepare_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;

        let err = assert_error(
            query_memcpy_with_sql(&conn, id(crate::EventKind::Memcpy), "SELECT * FROM"),
            "nsys.query.sql-prepare",
            INSPECT_MEMCPY_SQL,
        )?;

        assert!(matches!(
            err,
            crate::NsysQueryError::Sql {
                phase: crate::SqlPhase::Prepare,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn query_memcpy_query_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = "SELECT ? AS start_ns WHERE ? IS NOT NULL";

        let err = assert_error(
            query_memcpy_with_sql(&conn, id(crate::EventKind::Memcpy), sql),
            "nsys.query.sql-query",
            INSPECT_MEMCPY_SQL,
        )?;

        assert!(matches!(
            err,
            crate::NsysQueryError::Sql {
                phase: crate::SqlPhase::Query,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn query_memcpy_read_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = "SELECT 'not-a-start' AS start_ns WHERE ? IS NOT NULL";

        let err = assert_error(
            query_memcpy_with_sql(&conn, id(crate::EventKind::Memcpy), sql),
            "nsys.query.sql-read",
            INSPECT_MEMCPY_SQL,
        )?;

        assert!(matches!(
            err,
            crate::NsysQueryError::Sql {
                phase: crate::SqlPhase::Read,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn query_memset_prepare_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;

        let err = assert_error(
            query_memset_with_sql(&conn, id(crate::EventKind::Memset), "SELECT * FROM"),
            "nsys.query.sql-prepare",
            INSPECT_MEMSET_SQL,
        )?;

        assert!(matches!(
            err,
            crate::NsysQueryError::Sql {
                phase: crate::SqlPhase::Prepare,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn query_memset_query_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = "SELECT ? AS start_ns WHERE ? IS NOT NULL";

        let err = assert_error(
            query_memset_with_sql(&conn, id(crate::EventKind::Memset), sql),
            "nsys.query.sql-query",
            INSPECT_MEMSET_SQL,
        )?;

        assert!(matches!(
            err,
            crate::NsysQueryError::Sql {
                phase: crate::SqlPhase::Query,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn query_memset_read_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = "SELECT 'not-a-start' AS start_ns WHERE ? IS NOT NULL";

        let err = assert_error(
            query_memset_with_sql(&conn, id(crate::EventKind::Memset), sql),
            "nsys.query.sql-read",
            INSPECT_MEMSET_SQL,
        )?;

        assert!(matches!(
            err,
            crate::NsysQueryError::Sql {
                phase: crate::SqlPhase::Read,
                ..
            }
        ));
        Ok(())
    }
}
