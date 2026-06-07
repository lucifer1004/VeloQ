//! `inspect cpu_sample:N` — CPU IP sample + resolved backtrace.
//!
//! One `COMPOSITE_EVENTS` row joined to its `SAMPLING_CALLCHAINS`
//! stack. The callchain SELECT runs only when the optional callchain
//! table is present; otherwise the sample comes back with an empty
//! `callchain` (traces captured with sampling on, backtrace off).

use crate::query_sql::exec::{SqlLabel, query_rows_fallible};
use crate::{NsysQueryResult, RowId};
use duckdb::Connection;
use duckdb::types::Value;
use serde::Serialize;

use super::{ColumnMap, EventDetails, map_inspect_read, maybe_col, opt_string, query_inspect_row};

const INSPECT_CPU_SAMPLE_SQL: &str = "cpu_sample";
const INSPECT_CPU_SAMPLE_CALLCHAIN_SQL: &str = "cpu_sample_callchain";

/// CPU IP sample (`COMPOSITE_EVENTS` row + joined
/// `SAMPLING_CALLCHAINS` stack). One sample = one timestamp on one
/// (cpu, thread) plus the resolved backtrace from leaf to outermost
/// recorded frame.
///
/// **Stack convention**: `callchain[0]` is the leaf (the function the
/// CPU was *currently executing*); higher indices walk outward
/// toward the thread entry point. The deepest frame is often
/// `"[Max depth]"` — NSys's sentinel meaning "stack walk truncated"
/// (raise capture-side `--samples-per-backtrace` to see further).
///
/// **`row_id` quirk**: `cpu_sample:N` uses `COMPOSITE_EVENTS.id` (the
/// natural join key to `SAMPLING_CALLCHAINS`), not the implicit table
/// row number. On typical exports these align, but the column-id
/// reading is the authoritative one and what callers should bind
/// against.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct CpuSampleDetails {
    /// Cross-trace key — equal to `row_id` stringified.
    pub key: String,
    pub row_id: RowId,
    pub start_ns: i64,
    pub cpu: i64,
    pub global_tid: i64,
    /// Decoded process id — `(global_tid >> 24) & 0xFFFFFF`.
    /// See [`crate::decode_global_tid`] for the full bit layout and
    /// the source-domain "59 offset" caveat.
    pub pid: i64,
    /// Decoded thread id — `global_tid & 0xFFFF`. NSys's TID slot is
    /// 16 bits; bits 16..23 carry the source-domain id, not more TID
    /// bits. See [`crate::decode_global_tid`].
    pub tid: i64,
    /// Raw `threadState` value (`ENUM_SAMPLING_THREAD_STATE` id).
    /// Almost always `1` (Running) on COMPOSITE_EVENTS rows since
    /// NSys takes IP samples only on running threads; the other
    /// states show up on `SCHED_EVENTS` instead.
    pub thread_state: i64,
    /// Label from `ENUM_SAMPLING_THREAD_STATE.name` — PascalCase
    /// (`"Running"`, `"Interruptible"`, `"OSRuntime"`, …),
    /// passed through verbatim from the nsys enum table. `None`
    /// when the enum row is absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thread_state_name: Option<String>,
    /// `=1` periodic IP sample, `=0` context-switch event sample.
    /// Most traces have `=1` exclusively.
    pub cpu_cycles: i64,
    /// Leaf at index 0; the array is empty when the sample's stack
    /// couldn't be reconstructed.
    pub callchain: Vec<CallchainFrame>,
}

/// One resolved frame in a CPU sample's backtrace.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct CallchainFrame {
    /// 0 = leaf (currently-executing function); larger values walk
    /// outward toward the thread entry.
    pub depth: i64,
    /// Resolved symbol name (`StringIds.value` joined from
    /// `SAMPLING_CALLCHAINS.symbol`). `None` when the frame has no
    /// debug info — caller can render the raw `ip` instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol: Option<String>,
    /// Module **basename** (just the file name; full path stripped).
    /// `None` when the symbol's module column is null.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub module: Option<String>,
    /// True when the frame was in kernel mode (inside a syscall).
    pub kernel_mode: bool,
    /// True when nsys couldn't resolve `symbol` to a name. Pair with
    /// `ip` for the raw address.
    pub unresolved: bool,
    /// Original IP rendered as a hex string. Always populated; useful
    /// for unresolved frames or when symbolisation is ambiguous.
    pub ip: String,
}

/// `cpu_sample:N` → resolves to a `COMPOSITE_EVENTS` row keyed by
/// `id = N` (the natural join key, not the implicit table row number)
/// plus a stack walk via `SAMPLING_CALLCHAINS`. The event SELECT
/// runs first (cheap; needed even when callchains are missing);
/// the callchain SELECT runs only when `SAMPLING_CALLCHAINS` is
/// present, returning an empty `callchain` otherwise.
pub(super) fn query_cpu_sample(
    conn: &Connection,
    cols: &ColumnMap,
    id: RowId,
) -> NsysQueryResult<Option<EventDetails>> {
    const T: &str = "COMPOSITE_EVENTS";
    if !cols.contains_key(T) {
        return Ok(None);
    }
    let cycles = maybe_col(cols, T, "cpuCycles");
    let global_tid = veloq_nsys_data::sql_expr::u64_bits_to_i64("t.globalTid");

    let sql = format!(
        r#"
        SELECT
            t.start,
            CAST(t.cpu AS BIGINT),
            {global_tid},
            CAST(t.threadState AS BIGINT),
            CAST({cycles} AS BIGINT),
            e.name
        FROM nsight.{T} t
        LEFT JOIN nsight.ENUM_SAMPLING_THREAD_STATE e ON e.id = t.threadState
        WHERE t.id = ?
        "#
    );
    query_inspect_row(conn, INSPECT_CPU_SAMPLE_SQL, &sql, id, |r| {
        let start_ns: i64 = map_inspect_read(INSPECT_CPU_SAMPLE_SQL, r.get(0))?;
        let cpu: i64 = map_inspect_read(INSPECT_CPU_SAMPLE_SQL, r.get(1))?;
        let global_tid: i64 = map_inspect_read(INSPECT_CPU_SAMPLE_SQL, r.get(2))?;
        let thread_state: i64 = map_inspect_read(INSPECT_CPU_SAMPLE_SQL, r.get(3))?;
        // `cpuCycles` is normally INTEGER but the column is optional on
        // older nsys schemas (`maybe_col` projected NULL when absent).
        // Read as `Option<i64>` then coalesce so we surface NULL-as-0
        // explicitly instead of swallowing a type-mismatch error.
        let cpu_cycles: i64 =
            map_inspect_read(INSPECT_CPU_SAMPLE_SQL, r.get::<_, Option<i64>>(4))?.unwrap_or(0);
        let thread_state_name: Option<String> =
            map_inspect_read(INSPECT_CPU_SAMPLE_SQL, opt_string(r, 5))?;

        // Callchain: each frame's symbol + module joined to StringIds.
        // Order by stackDepth ASC — in this trace's schema depth=0 is the
        // leaf (currently-executing frame) and grows outward; we emit in
        // the same order so JSON consumers see leaf-first.
        //
        // SAMPLING_CALLCHAINS is optional: traces captured without backtrace
        // collection have COMPOSITE_EVENTS but no callchain table. Surface
        // the event row with an empty callchain rather than erroring.
        let callchain = if cols.contains_key("SAMPLING_CALLCHAINS") {
            let chain_sql = "
            SELECT
                CAST(c.stackDepth AS BIGINT),
                s.value AS symbol_name,
                m.value AS module_name,
                CAST(COALESCE(c.kernelMode, 0) AS BIGINT),
                CAST(COALESCE(c.unresolved, 0) AS BIGINT),
                CAST(COALESCE(c.originalIP, 0) AS BIGINT)
            FROM nsight.SAMPLING_CALLCHAINS c
            LEFT JOIN nsight.StringIds s ON s.id = c.symbol
            LEFT JOIN nsight.StringIds m ON m.id = c.module
            WHERE c.id = ?
            ORDER BY c.stackDepth ASC
        ";
            let params = [Value::BigInt(id.rowid)];
            query_rows_fallible(
                conn,
                chain_sql,
                &params,
                SqlLabel::new("inspect", INSPECT_CPU_SAMPLE_CALLCHAIN_SQL),
                callchain_frame_row,
            )?
        } else {
            Vec::new()
        };

        let (pid, tid) = crate::decode_global_tid(global_tid);
        Ok(EventDetails::CpuSample(CpuSampleDetails {
            key: id.to_string(),
            row_id: id,
            start_ns,
            cpu,
            global_tid,
            pid,
            tid,
            thread_state,
            thread_state_name,
            cpu_cycles,
            callchain,
        }))
    })
}

fn callchain_frame_row(row: &duckdb::Row<'_>) -> NsysQueryResult<CallchainFrame> {
    let depth: i64 = map_inspect_read(INSPECT_CPU_SAMPLE_CALLCHAIN_SQL, row.get(0))?;
    let symbol = map_inspect_read(INSPECT_CPU_SAMPLE_CALLCHAIN_SQL, opt_string(row, 1))?;
    let module_full = map_inspect_read(INSPECT_CPU_SAMPLE_CALLCHAIN_SQL, opt_string(row, 2))?;
    let kernel_mode_int: i64 = map_inspect_read(INSPECT_CPU_SAMPLE_CALLCHAIN_SQL, row.get(3))?;
    let unresolved_int: i64 = map_inspect_read(INSPECT_CPU_SAMPLE_CALLCHAIN_SQL, row.get(4))?;
    let original_ip: i64 = map_inspect_read(INSPECT_CPU_SAMPLE_CALLCHAIN_SQL, row.get(5))?;
    Ok(CallchainFrame {
        depth,
        symbol,
        module: module_full.as_deref().map(crate::module_basename),
        kernel_mode: kernel_mode_int != 0,
        unresolved: unresolved_int != 0,
        // i64 -> u64 via `as` preserves the bit pattern, which is
        // what we want for displaying negative-coded kernel addresses.
        ip: format!("0x{:x}", original_ip as u64),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use std::collections::HashSet;
    use veloq_core::VeloqDiagnostic;

    fn row_id() -> RowId {
        RowId::new(crate::EventKind::CpuSample, 7)
    }

    fn base_columns(include_callchain: bool) -> ColumnMap {
        let mut cols = ColumnMap::new();
        cols.insert(
            "COMPOSITE_EVENTS",
            [
                "id",
                "start",
                "cpu",
                "globalTid",
                "threadState",
                "cpuCycles",
            ]
            .into_iter()
            .map(String::from)
            .collect::<HashSet<_>>(),
        );
        if include_callchain {
            cols.insert(
                "SAMPLING_CALLCHAINS",
                [
                    "id",
                    "stackDepth",
                    "symbol",
                    "module",
                    "kernelMode",
                    "unresolved",
                    "originalIP",
                ]
                .into_iter()
                .map(String::from)
                .collect::<HashSet<_>>(),
            );
        }
        cols
    }

    fn create_sample(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "
            CREATE SCHEMA nsight;
            CREATE TABLE nsight.COMPOSITE_EVENTS (
                id BIGINT,
                start BIGINT,
                cpu BIGINT,
                globalTid BIGINT,
                threadState BIGINT,
                cpuCycles BIGINT
            );
            CREATE TABLE nsight.ENUM_SAMPLING_THREAD_STATE (
                id BIGINT,
                name TEXT
            );
            INSERT INTO nsight.COMPOSITE_EVENTS
                VALUES (7, 11, 2, 16777217, 1, 99);
            INSERT INTO nsight.ENUM_SAMPLING_THREAD_STATE
                VALUES (1, 'Running');
            ",
        )?;
        Ok(())
    }

    #[test]
    fn cpu_sample_callchain_prepare_error_is_typed() -> Result<()> {
        let conn = Connection::open_in_memory()?;
        create_sample(&conn)?;

        let err = match query_cpu_sample(&conn, &base_columns(true), row_id()) {
            Ok(row) => anyhow::bail!("missing callchain table should not succeed: {row:?}"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-prepare");
        assert_eq!(
            err.sql_parts(),
            Some((
                "inspect",
                crate::SqlPhase::Prepare,
                INSPECT_CPU_SAMPLE_CALLCHAIN_SQL
            ))
        );
        Ok(())
    }

    #[test]
    fn cpu_sample_callchain_query_error_is_typed() -> Result<()> {
        let conn = Connection::open_in_memory()?;
        create_sample(&conn)?;
        conn.execute_batch(
            "
            CREATE TABLE nsight.StringIds (
                id BIGINT,
                value TEXT
            );
            CREATE TABLE nsight.SAMPLING_CALLCHAINS (
                id BIGINT,
                stackDepth TEXT,
                symbol BIGINT,
                module BIGINT,
                kernelMode BIGINT,
                unresolved BIGINT,
                originalIP BIGINT
            );
            INSERT INTO nsight.SAMPLING_CALLCHAINS
                VALUES (7, 'not-an-int', NULL, NULL, 0, 0, 42);
            ",
        )?;

        let err = match query_cpu_sample(&conn, &base_columns(true), row_id()) {
            Ok(row) => anyhow::bail!("malformed callchain row should not succeed: {row:?}"),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-query");
        assert_eq!(
            err.sql_parts(),
            Some((
                "inspect",
                crate::SqlPhase::Query,
                INSPECT_CPU_SAMPLE_CALLCHAIN_SQL
            ))
        );
        Ok(())
    }
}
