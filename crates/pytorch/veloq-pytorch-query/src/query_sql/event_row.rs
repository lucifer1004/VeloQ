use crate::dto::{EventRef, LinkRef};
use crate::{PytorchQueryError, PytorchQueryResult};
use serde_json::Value;

#[derive(Debug, Clone)]
pub(crate) struct EventSqlRow {
    pub(crate) key: String,
    pub(crate) row_id: String,
    pub(crate) event_type: String,
    pub(crate) name: String,
    pub(crate) start_ns: i64,
    pub(crate) duration_ns: i64,
    pub(crate) rank: Option<i64>,
    pub(crate) worker: Option<String>,
    pub(crate) device_id: Option<i64>,
    pub(crate) stream_id: Option<i64>,
    pub(crate) step: Option<i64>,
    pub(crate) is_comm: bool,
    pub(crate) external_id: Option<i64>,
    pub(crate) correlation_id: Option<i64>,
    pub(crate) comm_kind: Option<String>,
    pub(crate) bytes: Option<i64>,
    pub(crate) shape: Option<String>,
    pub(crate) trace_index: i64,
    pub(crate) original_index: u64,
    pub(crate) category: Option<String>,
    pub(crate) phase: Option<String>,
    pub(crate) pid: Option<i64>,
    pub(crate) tid: Option<i64>,
    pub(crate) parent_row_id: Option<String>,
    pub(crate) step_row_id: Option<String>,
    pub(crate) python_context_row_id: Option<String>,
    pub(crate) python_id: Option<i64>,
    pub(crate) python_parent_id: Option<i64>,
    pub(crate) is_gpu_activity: bool,
    pub(crate) raw_json: String,
}

impl EventSqlRow {
    pub(crate) fn event_ref(&self) -> EventRef {
        EventRef {
            key: self.key.clone(),
            row_id: self.row_id.clone(),
            event_type: self.event_type.clone(),
            name: self.name.clone(),
            start_ns: self.start_ns,
            duration_ns: self.duration_ns,
            rank: self.rank,
            worker: self.worker.clone(),
            device_id: self.device_id,
            stream_id: self.stream_id,
            step: self.step,
            is_comm: self.is_comm,
            external_id: self.external_id,
            correlation_id: self.correlation_id,
            comm_kind: self.comm_kind.clone(),
            bytes: self.bytes,
            shape: self.shape.clone(),
        }
    }

    pub(crate) fn raw_value(&self) -> PytorchQueryResult<Value> {
        serde_json::from_str(&self.raw_json)
            .map_err(|source| PytorchQueryError::inspect_json_decode("raw", source))
    }

    pub(crate) fn trace_index_u32(&self) -> PytorchQueryResult<u32> {
        u32::try_from(self.trace_index).map_err(PytorchQueryError::inspect_trace_index_overflow)
    }

    pub(crate) fn matches_python_identity(&self, frame: &EventSqlRow, python_id: i64) -> bool {
        self.event_type == "python"
            && self.trace_index == frame.trace_index
            && self.pid == frame.pid
            && self.tid == frame.tid
            && self.python_id == Some(python_id)
    }

    pub(crate) fn is_cpu_activity(&self) -> bool {
        !self.is_gpu_activity
    }

    pub(crate) fn is_runtime_or_driver(&self) -> bool {
        matches!(self.event_type.as_str(), "runtime" | "driver")
    }
}

pub(crate) fn event_sql_row(row: &duckdb::Row<'_>) -> Result<EventSqlRow, duckdb::Error> {
    Ok(EventSqlRow {
        key: row.get("key")?,
        row_id: row.get("row_id")?,
        event_type: row.get("event_type")?,
        name: row.get("name")?,
        start_ns: row.get("start_ns")?,
        duration_ns: row.get("duration_ns")?,
        rank: row.get("rank")?,
        worker: row.get("worker")?,
        device_id: row.get("device_id")?,
        stream_id: row.get("stream_id")?,
        step: row.get("step")?,
        is_comm: row.get("is_comm")?,
        external_id: row.get("external_id")?,
        correlation_id: row.get("correlation_id")?,
        comm_kind: row.get("comm_kind")?,
        bytes: row.get("bytes")?,
        shape: row.get("shape")?,
        trace_index: row.get("trace_index")?,
        original_index: row.get("original_index")?,
        category: row.get("category")?,
        phase: row.get("phase")?,
        pid: row.get("pid")?,
        tid: row.get("tid")?,
        parent_row_id: row.get("parent_row_id")?,
        step_row_id: row.get("step_row_id")?,
        python_context_row_id: row.get("python_context_row_id")?,
        python_id: row.get("python_id")?,
        python_parent_id: row.get("python_parent_id")?,
        is_gpu_activity: row.get("is_gpu_activity")?,
        raw_json: row.get("raw_json")?,
    })
}

pub(crate) fn link_sql_row(row: &duckdb::Row<'_>) -> Result<LinkRef, duckdb::Error> {
    Ok(LinkRef {
        key: row.get("key")?,
        from_row_id: row.get("from_row_id")?,
        to_row_id: row.get("to_row_id")?,
        kind: row.get("kind")?,
        confidence: row.get("confidence")?,
    })
}

pub(crate) struct ArgSqlRow {
    pub(crate) row_id: String,
    pub(crate) arg_key: String,
    pub(crate) arg_json: String,
}

pub(crate) fn arg_sql_row(row: &duckdb::Row<'_>) -> Result<ArgSqlRow, duckdb::Error> {
    Ok(ArgSqlRow {
        row_id: row.get("row_id")?,
        arg_key: row.get("arg_key")?,
        arg_json: row.get("arg_json")?,
    })
}
