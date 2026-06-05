use crate::dto::{
    CapabilitiesDto, CaptureFlagsRow, SummaryAuxiliary, SummaryResponse, TraceFileRow,
};
use veloq_pytorch_data::{TraceFile, TraceSet};

pub fn summary(trace: &TraceSet) -> SummaryResponse {
    let rows: Vec<TraceFileRow> = trace.files.iter().map(trace_file_row).collect();
    let count = rows.len();
    SummaryResponse {
        count,
        total_matched: count,
        rows,
        auxiliary: SummaryAuxiliary {
            time_range_ns: trace.trace_span.map(Into::into),
            capabilities: CapabilitiesDto::from(&trace.capabilities),
            artifact_dir: trace.artifact_dir.clone(),
            capture_flags: trace
                .files
                .iter()
                .map(|file| CaptureFlagsRow {
                    trace_index: file.trace_index,
                    flags: file.capture_flags.clone(),
                })
                .collect(),
        },
    }
}

fn trace_file_row(file: &TraceFile) -> TraceFileRow {
    TraceFileRow {
        key: file.key.clone(),
        trace_index: file.trace_index,
        path: file.path.clone(),
        rank: file.rank,
        worker: file.worker.clone(),
        event_count: file.event_count,
        schema_version: file.schema_version.clone(),
    }
}
