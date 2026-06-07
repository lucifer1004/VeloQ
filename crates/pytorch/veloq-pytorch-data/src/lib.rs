//! PyTorch/Kineto Chrome trace ingest for the `pytorch` VeloQ source.
//!
//! The crate stays below the CLI/query layer: it discovers single-trace
//! inputs, parses Chrome `traceEvents`, classifies a
//! typed event model, builds nesting/correlation/collective indexes, and
//! persists a source-local cache under `<input>.veloq/pytorch/`.

mod cache;
mod classify;
mod error;
mod index;
mod ingest;
mod input;
mod metadata;
mod model;
mod sidecar;
mod survey;
mod value;

pub use cache::{
    artifact_dir, build_or_load, build_or_load_query_trace, prep_state, trace_span_for_path,
};
pub use error::{PytorchDataError, PytorchDataResult};
pub use input::{detect_path, is_trace_file};
pub use model::{
    Capabilities, CollectiveGroup, CollectiveRankTiming, Event, EventLink, EventType,
    FileFingerprint, FlowEdge, InputFingerprint, PrepState, QueryTrace, SidecarState, TimeRange,
    TraceFile, TraceFileSchemaSurvey, TraceSchemaSurvey, TraceSet, TypedArgCoverage,
};
pub use sidecar::{PytorchSidecar, sidecar_path_for_artifact, sidecar_states};

pub const SOURCE_KIND: &str = "pytorch";
pub(crate) const CACHE_VERSION: u32 = 1;
