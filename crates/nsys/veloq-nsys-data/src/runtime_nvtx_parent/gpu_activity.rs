use crate::{NsysDataResult, Trace};
use arrow::array::{
    Array, Int8Array, Int16Array, Int32Array, Int64Array, UInt8Array, UInt16Array, UInt32Array,
    UInt64Array,
};
use arrow::datatypes::Schema;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

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

    /// View as a slice for the merge step's dispatch.
    fn as_slice(&self) -> &[(i32, i64)] {
        match self {
            DevCtxValue::Single(x) => std::slice::from_ref(x),
            DevCtxValue::Many(v) => v.as_slice(),
        }
    }
}

pub(super) type DevCtxMap = HashMap<(i64, i64), DevCtxValue>;

/// Collect `(native_pid, correlation_id) → (device_id, context_id)`.
///
/// Two paths, picked at runtime based on what's available:
///
/// Reads each present GPU activity table's parquet file via Arrow's
/// batched columnar reader, in parallel across tables, and joins
/// against an in-memory `ctx_for_pid` map. DuckDB→Rust row iteration
/// over ~27 M rows costs ~8 s on a 21.8 M-kernel trace; the columnar
/// path skips that handoff entirely.
///
/// Returns an empty map when the context-info table is absent or no
/// GPU activity table is present — callers treat empty as "no GPU
/// disambiguation available", collapsing to runtime-only attribution.
pub(super) fn collect_runtime_dev_ctx(trace: &Trace) -> NsysDataResult<DevCtxMap> {
    if !trace.table_exists("TARGET_INFO_CUDA_CONTEXT_INFO") {
        return Ok(HashMap::new());
    }
    let present: Vec<&'static str> = GPU_ACTIVITY_TABLES
        .iter()
        .copied()
        .filter(|t| trace.table_exists(t))
        .collect();
    if present.is_empty() {
        return Ok(HashMap::new());
    }
    let ctx_for_pid = read_ctx_for_pid(trace)?;
    let map = collect_via_parquet(trace, &present, &ctx_for_pid)?;
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

/// `(device, context) → process_id` table from
/// `TARGET_INFO_CUDA_CONTEXT_INFO`. Small (one row per CUDA context),
/// so we just load it into a HashMap once.
pub(super) fn read_ctx_for_pid(trace: &Trace) -> NsysDataResult<HashMap<(i32, i64), i64>> {
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
        out.insert((dev, ctx), pid);
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
fn collect_via_parquet(
    trace: &Trace,
    tables: &[&'static str],
    ctx_for_pid: &HashMap<(i32, i64), i64>,
) -> NsysDataResult<DevCtxMap> {
    let paths: Vec<PathBuf> = tables.iter().map(|t| trace.parquet_path(t)).collect();
    let partials: Vec<DevCtxMap> = paths
        .par_iter()
        .map(|p| read_gpu_dev_ctx_parquet(p, ctx_for_pid))
        .collect::<NsysDataResult<Vec<_>>>()?;
    let total: usize = partials.iter().map(|m| m.len()).sum();
    let mut out: DevCtxMap = HashMap::with_capacity(total);
    for p in partials {
        for (k, v) in p {
            match out.entry(k) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(v);
                }
                std::collections::hash_map::Entry::Occupied(mut e) => {
                    for dx in v.as_slice() {
                        e.get_mut().push(*dx);
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Read one GPU activity table's parquet file via Arrow's batched
/// columnar reader and project each row through `ctx_for_pid` to a
/// `(native_pid, correlation_id) → (device_id, context_id)` entry.
pub(super) fn read_gpu_dev_ctx_parquet(
    path: &Path,
    ctx_for_pid: &HashMap<(i32, i64), i64>,
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
            if let Some(&native_pid) = ctx_for_pid.get(&(dev, ctx_id)) {
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
    Ok(out)
}

fn gpu_activity_column_index(
    schema: &Schema,
    path: &Path,
    column: &'static str,
) -> NsysDataResult<usize> {
    schema.index_of(column).map_err(|_| {
        crate::NsysDataError::nvtx_parent_gpu_activity_column_missing(path.display(), column)
    })
}

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
