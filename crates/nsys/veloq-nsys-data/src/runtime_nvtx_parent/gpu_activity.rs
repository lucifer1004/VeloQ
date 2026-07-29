use crate::{CudaProcessResolver, NsysDataResult, Trace};
#[cfg(test)]
use arrow::array::{
    Array, Int8Array, Int16Array, Int32Array, Int64Array, UInt8Array, UInt16Array, UInt32Array,
    UInt64Array,
};
#[cfg(test)]
use arrow::datatypes::Schema;
#[cfg(test)]
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::HashMap;
#[cfg(test)]
use std::fs::File;
#[cfg(test)]
use std::path::Path;

/// All GPU activity tables that can populate the
/// `(native_pid, correlation_id) → (device_id, context_id)` map. The
/// runtime row's enclosing context comes from whichever of these
/// emitted the matching `correlationId`.
const GPU_ACTIVITY_TABLES: &[&str] = &[
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "CUPTI_ACTIVITY_KIND_MEMSET",
    "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
];

/// `(native_pid, correlation_id) → all matching (device_id, context_id)`
/// pairs found across the GPU activity tables.
///
/// Most `(native_pid, correlationId)` keys resolve to exactly one
/// `(device, context)`, so the common-case storage is `Single`
/// (inline, no allocation). When CUPTI presents multiple
/// `(device, context)` for the same key — a multi-context-clash
/// artifact — the entry promotes to `Many` and the merge step fans
/// out: one sidecar entry per candidate `(device, context)`.
///
/// Promotion is one-way; once `Many`, the entry stays `Many` even if
/// later inserts dedup. The expected churn is tiny so the asymmetric
/// transition is fine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum DevCtxValue {
    Single((i32, i64)),
    Many(Vec<(i32, i64)>),
}

impl DevCtxValue {
    /// True if `dx` is already represented in this value.
    fn contains(&self, dx: &(i32, i64)) -> bool {
        match self {
            DevCtxValue::Single(existing) => existing == dx,
            DevCtxValue::Many(v) => v.contains(dx),
        }
    }

    /// Add `dx` if not already present, promoting `Single → Many` on
    /// the first divergent insert.
    fn push(&mut self, dx: (i32, i64)) {
        if self.contains(&dx) {
            return;
        }
        match self {
            DevCtxValue::Single(existing) => {
                *self = DevCtxValue::Many(vec![*existing, dx]);
            }
            DevCtxValue::Many(v) => v.push(dx),
        }
    }
}

pub(super) type DevCtxMap = HashMap<(i64, i64), DevCtxValue>;

/// Collect `(native_pid, correlation_id) → (device_id, context_id)`.
///
/// Process identity is resolved from activity `globalPid` where
/// available, then from process-aware context/runtime evidence. This
/// deliberately avoids the old single-valued `(device, context) -> pid`
/// map, which overwrote rank-private ordinal collisions.
pub(super) fn collect_runtime_dev_ctx(trace: &Trace) -> NsysDataResult<DevCtxMap> {
    let present: Vec<&'static str> = GPU_ACTIVITY_TABLES
        .iter()
        .copied()
        .filter(|t| trace.table_exists(t))
        .collect();
    if present.is_empty() {
        return Ok(HashMap::new());
    }
    let resolver = CudaProcessResolver::build(trace)?;
    let mut map = HashMap::new();
    for table in present {
        collect_table(trace, &resolver, table, &mut map)?;
    }
    // Count Single vs Many to surface the multi-context fan-out
    // case. CUPTI's documented model assigns process-unique
    // correlationIds across contexts, so `Many > 0` is unexpected
    // — a malformed trace, driver quirk, or assumption-violating
    // CUPTI build. We log a warning when it fires so the agent /
    // operator sees it (the fan-out itself attributes correctly;
    // the warning is just a heads-up that the trace tripped a code
    // path that's exercised by no production trace we've seen).
    let (single_count, many_count) = map.values().fold((0usize, 0usize), |(s, m), v| match v {
        DevCtxValue::Single(_) => (s + 1, m),
        DevCtxValue::Many(_) => (s, m + 1),
    });
    if many_count > 0 {
        log::warn!(
            "runtime_nvtx_parent: {} of {} dev/ctx entries had multi-context (native_pid, \
             correlationId) clashes — CUPTI usually emits unique correlationIds per process, \
             so this may indicate a malformed trace or unusual driver state; attribution \
             fans out each clashed runtime row to one sidecar entry per (device, context)",
            many_count,
            single_count + many_count,
        );
    } else {
        log::debug!(
            "runtime_nvtx_parent: dev_ctx breakdown {} Single / 0 Many (no multi-context clashes)",
            single_count,
        );
    }
    Ok(map)
}

fn collect_table(
    trace: &Trace,
    resolver: &CudaProcessResolver,
    table: &str,
    out: &mut DevCtxMap,
) -> NsysDataResult<()> {
    let global_pid = if trace.table_has_column(table, "globalPid") {
        "CAST(globalPid AS BIGINT)"
    } else {
        "CAST(NULL AS BIGINT)"
    };
    let sql = format!(
        "SELECT CAST(correlationId AS BIGINT), CAST(deviceId AS INTEGER), \
                CAST(contextId AS BIGINT), CAST(start AS BIGINT), {global_pid} \
         FROM nsight.{table} \
         WHERE correlationId IS NOT NULL"
    );
    let mut stmt = trace
        .conn()
        .prepare(&sql)
        .map_err(|source| crate::NsysDataError::nvtx_parent_rows_prepare(table, source))?;
    let mut rows = stmt
        .query([])
        .map_err(|source| crate::NsysDataError::nvtx_parent_rows_query(table, source))?;
    while let Some(row) = rows
        .next()
        .map_err(|source| crate::NsysDataError::nvtx_parent_rows_read(table, source))?
    {
        let correlation_id: i64 = row
            .get(0)
            .map_err(|source| crate::NsysDataError::nvtx_parent_rows_read(table, source))?;
        let device_id: i32 = row
            .get(1)
            .map_err(|source| crate::NsysDataError::nvtx_parent_rows_read(table, source))?;
        let context_id: i64 = row
            .get(2)
            .map_err(|source| crate::NsysDataError::nvtx_parent_rows_read(table, source))?;
        let start_ns: i64 = row
            .get(3)
            .map_err(|source| crate::NsysDataError::nvtx_parent_rows_read(table, source))?;
        let global_pid: Option<i64> = row
            .get(4)
            .map_err(|source| crate::NsysDataError::nvtx_parent_rows_read(table, source))?;
        let process_id = resolver.resolve_required(
            table,
            device_id,
            context_id,
            Some(correlation_id),
            start_ns,
            global_pid,
        )?;
        let dx = (device_id, context_id);
        match out.entry((process_id, correlation_id)) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(DevCtxValue::Single(dx));
            }
            std::collections::hash_map::Entry::Occupied(mut entry) => entry.get_mut().push(dx),
        }
    }
    Ok(())
}

/// `(device, context) → process_id[]` table from
/// `TARGET_INFO_CUDA_CONTEXT_INFO`. Small (one row per CUDA context),
/// so we just load it into a multimap once. The value must remain
/// multi-valued because rank-private CUDA namespaces can reuse both
/// device and context ordinals.
#[cfg(test)]
pub(super) fn read_ctx_for_pid(trace: &Trace) -> NsysDataResult<HashMap<(i32, i64), Vec<i64>>> {
    const TABLE: &str = "TARGET_INFO_CUDA_CONTEXT_INFO";
    let mut stmt = trace
        .conn()
        .prepare(
            r#"SELECT CAST(deviceId  AS INTEGER) AS device_id,
                  CAST(contextId AS BIGINT)  AS context_id,
                  CAST(processId AS BIGINT)  AS process_id
           FROM nsight.TARGET_INFO_CUDA_CONTEXT_INFO"#,
        )
        .map_err(|source| crate::NsysDataError::nvtx_parent_rows_prepare(TABLE, source))?;
    let mut rows = stmt
        .query([])
        .map_err(|source| crate::NsysDataError::nvtx_parent_rows_query(TABLE, source))?;
    let mut out = HashMap::new();
    while let Some(r) = rows
        .next()
        .map_err(|source| crate::NsysDataError::nvtx_parent_rows_read(TABLE, source))?
    {
        let dev: i32 = r
            .get(0)
            .map_err(|source| crate::NsysDataError::nvtx_parent_rows_read(TABLE, source))?;
        let ctx: i64 = r
            .get(1)
            .map_err(|source| crate::NsysDataError::nvtx_parent_rows_read(TABLE, source))?;
        let pid: i64 = r
            .get(2)
            .map_err(|source| crate::NsysDataError::nvtx_parent_rows_read(TABLE, source))?;
        let pids = out.entry((dev, ctx)).or_insert_with(Vec::new);
        if !pids.contains(&pid) {
            pids.push(pid);
        }
    }
    Ok(out)
}

/// Build the dev/ctx map by reading each present GPU activity table's
/// parquet file directly. Caller guarantees every `tables` entry is
/// present in the parquetdir (filtered via `Trace::table_exists`).
///
/// Rayon-parallelises across tables so kernel (the largest) doesn't
/// gate memcpy / memset / sync; each table contributes a partial map
/// that's merged at the end.
/// Read one GPU activity table's parquet file via Arrow's batched
/// columnar reader and project each row through `ctx_for_pid` to a
/// `(native_pid, correlation_id) → (device_id, context_id)` entry.
#[cfg(test)]
pub(super) fn read_gpu_dev_ctx_parquet(
    path: &Path,
    ctx_for_pid: &HashMap<(i32, i64), Vec<i64>>,
) -> NsysDataResult<DevCtxMap> {
    let file = File::open(path).map_err(|source| {
        crate::NsysDataError::nvtx_parent_gpu_activity_open(path.display(), source)
    })?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|source| {
        crate::NsysDataError::nvtx_parent_gpu_activity_reader_open(path.display(), source)
    })?;
    let schema = builder.schema().clone();
    let corr_idx = gpu_activity_column_index(&schema, path, "correlationId")?;
    let dev_idx = gpu_activity_column_index(&schema, path, "deviceId")?;
    let ctx_idx = gpu_activity_column_index(&schema, path, "contextId")?;
    let reader = builder.build().map_err(|source| {
        crate::NsysDataError::nvtx_parent_gpu_activity_reader_build(path.display(), source)
    })?;

    let mut out: DevCtxMap = HashMap::new();
    for batch in reader {
        let batch = batch.map_err(|source| {
            crate::NsysDataError::nvtx_parent_gpu_activity_batch_read(path.display(), source)
        })?;
        let corrs = batch.column(corr_idx).as_ref();
        let devs = batch.column(dev_idx).as_ref();
        let ctxs = batch.column(ctx_idx).as_ref();
        let n = batch.num_rows();
        out.reserve(n);
        for i in 0..n {
            let Some(corr) = parquet_integer_i64(corrs, i, "correlationId", path)? else {
                continue;
            };
            let Some(dev_i64) = parquet_integer_i64(devs, i, "deviceId", path)? else {
                continue;
            };
            let Some(ctx_id) = parquet_integer_i64(ctxs, i, "contextId", path)? else {
                continue;
            };
            let dev = i32::try_from(dev_i64).map_err(|_| {
                crate::NsysDataError::nvtx_parent_int32_overflow(
                    path.display(),
                    "deviceId",
                    dev_i64,
                )
            })?;
            if let Some(native_pids) = ctx_for_pid.get(&(dev, ctx_id)) {
                for &native_pid in native_pids {
                    match out.entry((native_pid, corr)) {
                        std::collections::hash_map::Entry::Vacant(e) => {
                            e.insert(DevCtxValue::Single((dev, ctx_id)));
                        }
                        std::collections::hash_map::Entry::Occupied(mut e) => {
                            e.get_mut().push((dev, ctx_id));
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
fn gpu_activity_column_index(
    schema: &Schema,
    path: &Path,
    column: &'static str,
) -> NsysDataResult<usize> {
    schema.index_of(column).map_err(|_| {
        crate::NsysDataError::nvtx_parent_gpu_activity_column_missing(path.display(), column)
    })
}

#[cfg(test)]
pub(super) fn parquet_integer_i64(
    array: &dyn Array,
    row: usize,
    column: &str,
    path: &Path,
) -> NsysDataResult<Option<i64>> {
    if array.is_null(row) {
        return Ok(None);
    }
    if let Some(a) = array.as_any().downcast_ref::<Int8Array>() {
        return Ok(Some(i64::from(a.value(row))));
    }
    if let Some(a) = array.as_any().downcast_ref::<Int16Array>() {
        return Ok(Some(i64::from(a.value(row))));
    }
    if let Some(a) = array.as_any().downcast_ref::<Int32Array>() {
        return Ok(Some(i64::from(a.value(row))));
    }
    if let Some(a) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok(Some(a.value(row)));
    }
    if let Some(a) = array.as_any().downcast_ref::<UInt8Array>() {
        return Ok(Some(i64::from(a.value(row))));
    }
    if let Some(a) = array.as_any().downcast_ref::<UInt16Array>() {
        return Ok(Some(i64::from(a.value(row))));
    }
    if let Some(a) = array.as_any().downcast_ref::<UInt32Array>() {
        return Ok(Some(i64::from(a.value(row))));
    }
    if let Some(a) = array.as_any().downcast_ref::<UInt64Array>() {
        let value = a.value(row);
        return match i64::try_from(value) {
            Ok(value) => Ok(Some(value)),
            Err(_) => Err(crate::NsysDataError::nvtx_parent_integer_overflow(
                path.display(),
                column,
                value,
            )),
        };
    }
    Err(crate::NsysDataError::nvtx_parent_unsupported_integer_type(
        path.display(),
        column,
        format!("{:?}", array.data_type()),
    ))
}
