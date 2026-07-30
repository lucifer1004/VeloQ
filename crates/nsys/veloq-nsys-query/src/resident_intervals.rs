//! Session-local normalized GPU intervals for daemon-routed scan queries.
//!
//! The view is a DuckDB TEMP VIEW owned by the resident [`Trace`] connection.
//! It projects the already registered, process-qualified GPU-work sidecar and
//! disappears with the daemon session. `timeline`, `concurrency`, and `gaps`
//! can therefore share one interval contract without copying the full dataset
//! into daemon memory or adding another sidecar lifecycle.

use veloq_nsys_data::Trace;

use crate::query_sql::gpu_work::{GpuWorkClass, GpuWorkSet};
use crate::{NsysQueryError, NsysQueryResult};

const TABLE: &str = "veloq_resident_gpu_intervals";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentIntervalViewInfo {
    pub accounted_bytes: u64,
}

/// Register the resident view once for this connection.
///
/// `Ok(None)` means no fresh GPU-work sidecar is registered, or that sidecar
/// contains a row the resident contract cannot safely represent. Callers
/// retain the established source or sidecar query path for that session.
pub fn ensure(trace: &Trace) -> NsysQueryResult<Option<ResidentIntervalViewInfo>> {
    if available(trace) {
        return view_info().map(Some);
    }

    if !veloq_nsys_data::gpu_work_events::view_available(trace) {
        return Ok(None);
    }

    if has_unsupported_rows(trace)? {
        return Ok(None);
    }

    let select = sidecar_select(&GpuWorkSet::from_data_definition()?)?;
    let definition = format!("CREATE TEMP VIEW {TABLE} AS {select}");
    trace
        .conn()
        .execute_batch(&definition)
        .map_err(|source| NsysQueryError::sql_query("resident intervals", "register", source))?;

    Ok(Some(ResidentIntervalViewInfo {
        accounted_bytes: definition.len().try_into().unwrap_or(u64::MAX),
    }))
}

fn sidecar_select(work: &GpuWorkSet) -> NsysQueryResult<String> {
    let mut compute_labels = Vec::new();
    for kind in work.kinds() {
        if matches!(work.class(*kind)?, GpuWorkClass::Compute) {
            compute_labels.push(format!("'{}'", kind.as_str()));
        }
    }
    let compute_labels = compute_labels.join(", ");
    Ok(format!(
        "SELECT \
            CAST(kind AS VARCHAR) AS kind, \
            CAST(row_id AS BIGINT) AS row_id, \
            CAST(process_id AS BIGINT) AS process_id, \
            CAST(device_id AS INTEGER) AS device_id, \
            CAST(stream_id AS BIGINT) AS stream_id, \
            CAST(start_ns AS BIGINT) AS start_ns, \
            CAST(end_ns AS BIGINT) AS end_ns, \
            CASE WHEN kind IN ({compute_labels}) THEN 1 ELSE 0 END::INTEGER AS is_compute \
         FROM nsight.gpu_work_events"
    ))
}

pub(crate) fn available(trace: &Trace) -> bool {
    trace
        .conn()
        .query_row(
            "SELECT EXISTS(\
                SELECT 1 FROM duckdb_views() \
                WHERE view_name = ? AND temporary\
            )",
            [TABLE],
            |row| row.get::<_, bool>(0),
        )
        .unwrap_or(false)
}

pub(crate) const fn table_name() -> &'static str {
    TABLE
}

fn view_info() -> NsysQueryResult<ResidentIntervalViewInfo> {
    let select = sidecar_select(&GpuWorkSet::from_data_definition()?)?;
    let definition = format!("CREATE TEMP VIEW {TABLE} AS {select}");
    Ok(ResidentIntervalViewInfo {
        accounted_bytes: definition.len().try_into().unwrap_or(u64::MAX),
    })
}

fn has_unsupported_rows(trace: &Trace) -> NsysQueryResult<bool> {
    trace
        .conn()
        .query_row(
            "SELECT EXISTS(\
                SELECT 1 FROM nsight.gpu_work_events \
                WHERE process_id IS NULL OR end_ns <= start_ns \
                LIMIT 1\
            )",
            [],
            |row| row.get(0),
        )
        .map_err(|source| NsysQueryError::sql_query("resident intervals", "eligibility", source))
}
