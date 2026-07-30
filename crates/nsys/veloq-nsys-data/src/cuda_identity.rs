//! Process-aware identity for CUDA activity rows.
//!
//! CUDA ordinals and handles are process-local. In particular, two rank
//! processes that each expose one physical GPU through `CUDA_VISIBLE_DEVICES`
//! can both emit `(device=0, context=1, stream=7, correlation=42)`. This
//! module resolves the owning native PID before callers group or correlate
//! those rows.

use crate::{NsysDataResult, Trace};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy)]
struct RuntimeAnchor {
    start_ns: i64,
    end_ns: i64,
}

#[derive(Debug, Clone, Copy)]
enum NativeProcessEvidence {
    Unique(i64),
    Ambiguous,
}

/// In-memory resolver shared by sidecar and correlation-index builders.
#[derive(Debug, Default)]
pub struct CudaProcessResolver {
    global_to_native: HashMap<i64, NativeProcessEvidence>,
    context_processes: HashMap<(i32, i64), Vec<i64>>,
    runtime_anchors: HashMap<(i64, i64), Vec<RuntimeAnchor>>,
    known_processes: HashSet<i64>,
}

impl CudaProcessResolver {
    pub fn build(trace: &Trace) -> NsysDataResult<Self> {
        let mut out = Self::default();
        out.load_processes(trace)?;
        out.load_contexts(trace)?;
        out.load_runtime_anchors(trace)?;
        Ok(out)
    }

    /// Resolve a CUDA activity row to its native PID.
    ///
    /// Resolution order is:
    /// 1. the activity's `globalPid` bridged through `PROCESSES`;
    /// 2. a unique `(device, context) -> process` mapping;
    /// 3. the process whose same-correlation runtime call is temporally
    ///    closest to the activity;
    /// 4. the sole process known to the trace;
    /// 5. unresolved (`None`) when the trace does not contain enough
    ///    ownership evidence.
    pub fn resolve(
        &self,
        device_id: i32,
        context_id: i64,
        correlation_id: Option<i64>,
        start_ns: i64,
        global_pid: Option<i64>,
    ) -> Option<i64> {
        if let Some(global_pid) = global_pid {
            match self.global_to_native.get(&global_pid) {
                Some(NativeProcessEvidence::Unique(native_pid)) => return Some(*native_pid),
                Some(NativeProcessEvidence::Ambiguous) => return None,
                None => {}
            }
            // Some partial exports carry a native pid directly but omit
            // PROCESSES. Accept it only when another trace table confirms
            // that pid, avoiding an ungrounded global-id interpretation.
            if self.known_processes.contains(&global_pid) {
                return Some(global_pid);
            }
        }

        let candidates = self
            .context_processes
            .get(&(device_id, context_id))
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        if candidates.len() == 1 {
            return candidates.first().copied();
        }

        if let Some(correlation_id) = correlation_id
            && let Some(pid) = self.closest_runtime_process(candidates, correlation_id, start_ns)
        {
            return Some(pid);
        }

        if self.known_processes.len() == 1 {
            return self.known_processes.iter().copied().next();
        }
        None
    }

    /// Resolve an activity row whose downstream identity requires a native PID.
    pub fn resolve_required(
        &self,
        table: &str,
        device_id: i32,
        context_id: i64,
        correlation_id: Option<i64>,
        start_ns: i64,
        global_pid: Option<i64>,
    ) -> NsysDataResult<i64> {
        self.resolve(device_id, context_id, correlation_id, start_ns, global_pid)
            .ok_or_else(|| {
                crate::NsysDataError::cuda_process_unresolved(
                    table,
                    device_id,
                    context_id,
                    correlation_id,
                )
            })
    }

    fn closest_runtime_process(
        &self,
        candidates: &[i64],
        correlation_id: i64,
        activity_start_ns: i64,
    ) -> Option<i64> {
        let mut scored: Vec<(bool, i64, i64)> = Vec::new();
        for &pid in candidates {
            let Some(anchors) = self.runtime_anchors.get(&(pid, correlation_id)) else {
                continue;
            };
            let best = anchors
                .iter()
                .map(|anchor| {
                    let launch_end = anchor.end_ns.max(anchor.start_ns);
                    (
                        launch_end > activity_start_ns,
                        launch_end.abs_diff(activity_start_ns) as i64,
                    )
                })
                .min();
            if let Some((after_activity, distance)) = best {
                scored.push((after_activity, distance, pid));
            }
        }
        scored.sort_unstable();
        let first = scored.first().copied()?;
        if scored
            .get(1)
            .is_some_and(|second| (second.0, second.1) == (first.0, first.1))
        {
            return None;
        }
        Some(first.2)
    }

    fn load_processes(&mut self, trace: &Trace) -> NsysDataResult<()> {
        if !trace.table_exists("PROCESSES")
            || !trace.table_has_column("PROCESSES", "globalPid")
            || !trace.table_has_column("PROCESSES", "pid")
        {
            return Ok(());
        }
        const TABLE: &str = "PROCESSES";
        let mut stmt = trace
            .conn()
            .prepare(
                "SELECT CAST(globalPid AS BIGINT), CAST(pid AS BIGINT) \
                 FROM nsight.PROCESSES \
                 WHERE globalPid IS NOT NULL AND pid IS NOT NULL",
            )
            .map_err(|source| crate::NsysDataError::correlation_scan_prepare(TABLE, source))?;
        let mut rows = stmt
            .query([])
            .map_err(|source| crate::NsysDataError::correlation_scan_query(TABLE, source))?;
        while let Some(row) = rows
            .next()
            .map_err(|source| crate::NsysDataError::correlation_scan_read(TABLE, source))?
        {
            let global_pid: i64 = row
                .get(0)
                .map_err(|source| crate::NsysDataError::correlation_scan_read(TABLE, source))?;
            let native_pid: i64 = row
                .get(1)
                .map_err(|source| crate::NsysDataError::correlation_scan_read(TABLE, source))?;
            match self.global_to_native.entry(global_pid) {
                Entry::Vacant(entry) => {
                    entry.insert(NativeProcessEvidence::Unique(native_pid));
                }
                Entry::Occupied(mut entry) => {
                    if matches!(
                        entry.get(),
                        NativeProcessEvidence::Unique(existing) if *existing != native_pid
                    ) {
                        entry.insert(NativeProcessEvidence::Ambiguous);
                    }
                }
            }
            self.known_processes.insert(native_pid);
        }
        Ok(())
    }

    fn load_contexts(&mut self, trace: &Trace) -> NsysDataResult<()> {
        if !trace.table_exists("TARGET_INFO_CUDA_CONTEXT_INFO") {
            return Ok(());
        }
        const TABLE: &str = "TARGET_INFO_CUDA_CONTEXT_INFO";
        let mut stmt = trace
            .conn()
            .prepare(
                "SELECT CAST(deviceId AS INTEGER), CAST(contextId AS BIGINT), \
                        CAST(processId AS BIGINT) \
                 FROM nsight.TARGET_INFO_CUDA_CONTEXT_INFO",
            )
            .map_err(|source| crate::NsysDataError::correlation_scan_prepare(TABLE, source))?;
        let mut rows = stmt
            .query([])
            .map_err(|source| crate::NsysDataError::correlation_scan_query(TABLE, source))?;
        while let Some(row) = rows
            .next()
            .map_err(|source| crate::NsysDataError::correlation_scan_read(TABLE, source))?
        {
            let device_id: i32 = row
                .get(0)
                .map_err(|source| crate::NsysDataError::correlation_scan_read(TABLE, source))?;
            let context_id: i64 = row
                .get(1)
                .map_err(|source| crate::NsysDataError::correlation_scan_read(TABLE, source))?;
            let process_id: i64 = row
                .get(2)
                .map_err(|source| crate::NsysDataError::correlation_scan_read(TABLE, source))?;
            let entry = self
                .context_processes
                .entry((device_id, context_id))
                .or_default();
            if !entry.contains(&process_id) {
                entry.push(process_id);
            }
            self.known_processes.insert(process_id);
        }
        for processes in self.context_processes.values_mut() {
            processes.sort_unstable();
        }
        Ok(())
    }

    fn load_runtime_anchors(&mut self, trace: &Trace) -> NsysDataResult<()> {
        if !trace.table_exists("CUPTI_ACTIVITY_KIND_RUNTIME") {
            return Ok(());
        }
        const TABLE: &str = "CUPTI_ACTIVITY_KIND_RUNTIME";
        let global_tid = crate::sql_expr::u64_bits_to_i64("globalTid");
        let sql = format!(
            "SELECT CAST(correlationId AS BIGINT), {global_tid}, \
                    CAST(start AS BIGINT), CAST(COALESCE(\"end\", start) AS BIGINT) \
             FROM nsight.CUPTI_ACTIVITY_KIND_RUNTIME \
             WHERE correlationId IS NOT NULL AND globalTid IS NOT NULL"
        );
        let mut stmt = trace
            .conn()
            .prepare(&sql)
            .map_err(|source| crate::NsysDataError::correlation_scan_prepare(TABLE, source))?;
        let mut rows = stmt
            .query([])
            .map_err(|source| crate::NsysDataError::correlation_scan_query(TABLE, source))?;
        while let Some(row) = rows
            .next()
            .map_err(|source| crate::NsysDataError::correlation_scan_read(TABLE, source))?
        {
            let correlation_id: i64 = row
                .get(0)
                .map_err(|source| crate::NsysDataError::correlation_scan_read(TABLE, source))?;
            let global_tid: i64 = row
                .get(1)
                .map_err(|source| crate::NsysDataError::correlation_scan_read(TABLE, source))?;
            let start_ns: i64 = row
                .get(2)
                .map_err(|source| crate::NsysDataError::correlation_scan_read(TABLE, source))?;
            let end_ns: i64 = row
                .get(3)
                .map_err(|source| crate::NsysDataError::correlation_scan_read(TABLE, source))?;
            let native_pid = native_pid_from_global_tid(global_tid);
            self.runtime_anchors
                .entry((native_pid, correlation_id))
                .or_default()
                .push(RuntimeAnchor { start_ns, end_ns });
            self.known_processes.insert(native_pid);
        }
        Ok(())
    }
}

/// Native PID encoded in an NSys `globalTid`.
#[inline]
pub fn native_pid_from_global_tid(global_tid: i64) -> i64 {
    (global_tid >> 24) & 0xFF_FFFF
}

fn cuda_activity_process_projection(
    trace: &Trace,
    table: &str,
    event_alias: &str,
    join_alias: &str,
    start_expr: &str,
) -> ProcessSqlProjection {
    let direct_capable = trace.table_has_column(table, "globalPid")
        && trace.table_exists("PROCESSES")
        && trace.table_has_column("PROCESSES", "globalPid")
        && trace.table_has_column("PROCESSES", "pid");
    let context_capable = trace.table_exists("TARGET_INFO_CUDA_CONTEXT_INFO");
    let runtime_capable = context_capable
        && trace.table_exists("CUPTI_ACTIVITY_KIND_RUNTIME")
        && trace.table_has_column(table, "correlationId");

    if !direct_capable && !context_capable {
        return ProcessSqlProjection {
            expr: "CAST(NULL AS BIGINT)".to_string(),
            join: String::new(),
        };
    }

    let direct_alias = format!("{join_alias}_direct");
    let context_alias = format!("{join_alias}_context");
    let runtime_alias = format!("{join_alias}_runtime");
    let mut joins = String::new();

    let direct_expr = if direct_capable {
        joins.push_str(&format!(
            "LEFT JOIN (\
                 SELECT globalPid, \
                        CASE WHEN COUNT(DISTINCT CAST(pid AS BIGINT)) = 1 \
                             THEN MIN(CAST(pid AS BIGINT)) END AS process_id, \
                        COUNT(DISTINCT CAST(pid AS BIGINT)) AS process_count \
                 FROM nsight.PROCESSES \
                 WHERE globalPid IS NOT NULL AND pid IS NOT NULL \
                 GROUP BY globalPid\
             ) {direct_alias} \
               ON {direct_alias}.globalPid = {event_alias}.globalPid "
        ));
        format!("{direct_alias}.process_id")
    } else {
        "CAST(NULL AS BIGINT)".to_string()
    };
    let direct_missing = if direct_capable {
        format!("{direct_alias}.process_count IS NULL")
    } else {
        "TRUE".to_string()
    };

    let context_expr = if context_capable {
        joins.push_str(&format!(
            "LEFT JOIN (\
                 SELECT CAST(deviceId AS INTEGER) AS device_id, \
                        CAST(contextId AS BIGINT) AS context_id, \
                        MIN(CAST(processId AS BIGINT)) AS process_id \
                 FROM nsight.TARGET_INFO_CUDA_CONTEXT_INFO \
                 GROUP BY CAST(deviceId AS INTEGER), CAST(contextId AS BIGINT) \
                 HAVING COUNT(DISTINCT CAST(processId AS BIGINT)) = 1\
             ) {context_alias} \
               ON {direct_missing} \
              AND {context_alias}.device_id = CAST({event_alias}.deviceId AS INTEGER) \
              AND {context_alias}.context_id = CAST({event_alias}.contextId AS BIGINT) "
        ));
        format!("{context_alias}.process_id")
    } else {
        "CAST(NULL AS BIGINT)".to_string()
    };

    let runtime_expr = if runtime_capable {
        let runtime_pid = native_pid_sql("r.globalTid");
        joins.push_str(&format!(
            "LEFT JOIN LATERAL (\
                 WITH runtime_candidates AS (\
                     SELECT CAST(c.processId AS BIGINT) AS process_id, \
                            CASE WHEN COALESCE(r.\"end\", r.start) > {start_expr} \
                                 THEN 1 ELSE 0 END AS after_activity, \
                            ABS(CAST(COALESCE(r.\"end\", r.start) AS BIGINT) \
                                - CAST({start_expr} AS BIGINT)) AS distance \
                     FROM nsight.TARGET_INFO_CUDA_CONTEXT_INFO c \
                     JOIN nsight.CUPTI_ACTIVITY_KIND_RUNTIME r \
                       ON r.correlationId = {event_alias}.correlationId \
                      AND {runtime_pid} = CAST(c.processId AS BIGINT) \
                     WHERE {direct_missing} \
                       AND {context_expr} IS NULL \
                       AND CAST(c.deviceId AS INTEGER) \
                           = CAST({event_alias}.deviceId AS INTEGER) \
                       AND CAST(c.contextId AS BIGINT) \
                           = CAST({event_alias}.contextId AS BIGINT)\
                 ), \
                 ranked AS (\
                     SELECT process_id, \
                            DENSE_RANK() OVER (\
                                ORDER BY after_activity, distance\
                            ) AS score_rank \
                     FROM runtime_candidates\
                 ) \
                 SELECT MIN(process_id) AS process_id \
                 FROM ranked \
                 WHERE score_rank = 1 \
                 HAVING COUNT(DISTINCT process_id) = 1\
             ) {runtime_alias} ON TRUE "
        ));
        format!("{runtime_alias}.process_id")
    } else {
        "CAST(NULL AS BIGINT)".to_string()
    };

    ProcessSqlProjection {
        expr: format!("COALESCE({direct_expr}, {context_expr}, {runtime_expr})"),
        join: joins,
    }
}

pub fn native_pid_sql(global_tid_expr: &str) -> String {
    format!("CAST((({global_tid_expr} >> 24) & 16777215) AS BIGINT)")
}

/// SQL projection of an event row's native process identity.
///
/// `expr` is always a BIGINT-compatible expression and `join` is the
/// optional relation needed by that expression. CUDA activity tables use
/// the process-aware resolver; host-thread tables decode `globalTid`;
/// process tables with `globalPid` bridge through `PROCESSES`.
pub struct ProcessSqlProjection {
    pub expr: String,
    pub join: String,
}

pub fn process_sql_projection(
    trace: &Trace,
    table: &str,
    event_alias: &str,
    join_alias: &str,
    start_expr: &str,
) -> ProcessSqlProjection {
    if trace.table_has_column(table, "deviceId") && trace.table_has_column(table, "contextId") {
        return cuda_activity_process_projection(trace, table, event_alias, join_alias, start_expr);
    }
    if trace.table_has_column(table, "globalTid") {
        return ProcessSqlProjection {
            expr: native_pid_sql(&format!("{event_alias}.globalTid")),
            join: String::new(),
        };
    }
    if trace.table_has_column(table, "globalPid")
        && trace.table_exists("PROCESSES")
        && trace.table_has_column("PROCESSES", "globalPid")
        && trace.table_has_column("PROCESSES", "pid")
    {
        return ProcessSqlProjection {
            expr: format!("CAST({join_alias}.pid AS BIGINT)"),
            join: format!(
                "LEFT JOIN nsight.PROCESSES {join_alias} \
                 ON {join_alias}.globalPid = {event_alias}.globalPid"
            ),
        };
    }
    ProcessSqlProjection {
        expr: "CAST(NULL AS BIGINT)".to_string(),
        join: String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::parquet_fixture_with_rows;
    use anyhow::Result;

    #[test]
    fn unresolved_identity_is_not_fabricated_as_pid_zero() {
        let resolver = CudaProcessResolver::default();
        assert_eq!(resolver.resolve(0, 1, Some(42), 100, None), None);
    }

    #[test]
    fn equal_runtime_candidates_remain_unresolved_in_memory_and_sql() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture_with_rows(&[
            (
                "TARGET_INFO_CUDA_CONTEXT_INFO",
                "CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO (\
                    deviceId BIGINT, contextId BIGINT, processId BIGINT)",
                vec![
                    "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO VALUES (0, 1, 1001)",
                    "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO VALUES (0, 1, 2002)",
                ],
            ),
            (
                "CUPTI_ACTIVITY_KIND_RUNTIME",
                "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (\
                    start BIGINT, \"end\" BIGINT, globalTid BIGINT, correlationId BIGINT)",
                vec![
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME \
                     VALUES (90, 95, 1001::BIGINT << 24, 42)",
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME \
                     VALUES (90, 95, 2002::BIGINT << 24, 42)",
                ],
            ),
            (
                "CUPTI_ACTIVITY_KIND_KERNEL",
                "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (\
                    start BIGINT, \"end\" BIGINT, deviceId BIGINT, contextId BIGINT, \
                    correlationId BIGINT)",
                vec!["INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (100, 110, 0, 1, 42)"],
            ),
        ])?;
        let trace = Trace::open(&pqtdir)?;
        let resolver = CudaProcessResolver::build(&trace)?;
        assert_eq!(resolver.resolve(0, 1, Some(42), 100, None), None);

        let process =
            process_sql_projection(&trace, "CUPTI_ACTIVITY_KIND_KERNEL", "t", "proc", "t.start");
        let sql = format!(
            "SELECT {expr} \
             FROM nsight.CUPTI_ACTIVITY_KIND_KERNEL t {join}",
            expr = process.expr,
            join = process.join,
        );
        let process_id: Option<i64> = trace.conn().query_row(&sql, [], |row| row.get(0))?;
        assert_eq!(process_id, None);
        Ok(())
    }

    #[test]
    fn sql_projection_follows_ordered_process_evidence() -> Result<()> {
        let (_dir, pqtdir) = parquet_fixture_with_rows(&[
            (
                "PROCESSES",
                "CREATE TABLE PROCESSES (globalPid BIGINT, pid BIGINT)",
                vec![
                    "INSERT INTO PROCESSES VALUES (111, 1001)",
                    "INSERT INTO PROCESSES VALUES (222, 1001)",
                    "INSERT INTO PROCESSES VALUES (222, 2002)",
                ],
            ),
            (
                "TARGET_INFO_CUDA_CONTEXT_INFO",
                "CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO (\
                    deviceId BIGINT, contextId BIGINT, processId BIGINT)",
                vec![
                    "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO VALUES (0, 1, 1001)",
                    "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO VALUES (0, 1, 2002)",
                    "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO VALUES (0, 2, 2002)",
                    "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO VALUES (0, 3, 1001)",
                    "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO VALUES (0, 3, 2002)",
                    "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO VALUES (0, 4, 1001)",
                ],
            ),
            (
                "CUPTI_ACTIVITY_KIND_RUNTIME",
                "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (\
                    start BIGINT, \"end\" BIGINT, globalTid BIGINT, correlationId BIGINT)",
                vec![
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME \
                     VALUES (90, 95, 1001::BIGINT << 24, 43)",
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME \
                     VALUES (70, 80, 2002::BIGINT << 24, 43)",
                ],
            ),
            (
                "CUPTI_ACTIVITY_KIND_KERNEL",
                "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (\
                    start BIGINT, \"end\" BIGINT, deviceId BIGINT, contextId BIGINT, \
                    correlationId BIGINT, globalPid BIGINT)",
                vec![
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
                     VALUES (100, 110, 0, 1, 42, 111)",
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
                     VALUES (100, 110, 0, 2, 42, NULL)",
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
                     VALUES (100, 110, 0, 3, 43, NULL)",
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
                     VALUES (100, 110, 0, 4, 44, 222)",
                ],
            ),
        ])?;
        let trace = Trace::open(&pqtdir)?;
        let resolver = CudaProcessResolver::build(&trace)?;
        let expected = [
            resolver.resolve(0, 1, Some(42), 100, Some(111)),
            resolver.resolve(0, 2, Some(42), 100, None),
            resolver.resolve(0, 3, Some(43), 100, None),
            resolver.resolve(0, 4, Some(44), 100, Some(222)),
        ];
        assert_eq!(expected, [Some(1001), Some(2002), Some(1001), None]);

        let process =
            process_sql_projection(&trace, "CUPTI_ACTIVITY_KIND_KERNEL", "t", "proc", "t.start");
        let sql = format!(
            "SELECT {expr} \
             FROM nsight.CUPTI_ACTIVITY_KIND_KERNEL t {join} \
             ORDER BY t.rowid",
            expr = process.expr,
            join = process.join,
        );
        let mut stmt = trace.conn().prepare(&sql)?;
        let actual = stmt
            .query_map([], |row| row.get::<_, Option<i64>>(0))?
            .collect::<duckdb::Result<Vec<_>>>()?;
        assert_eq!(actual, expected);
        Ok(())
    }
}
