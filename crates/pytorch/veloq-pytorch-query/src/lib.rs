//! Query layer for PyTorch/Kineto traces.
//!
//! Each public verb owns a module, matching the NSys query crate style.
//! Shared request filtering, scope echoing, and wire DTOs stay separate so
//! verb modules carry only query behavior.

pub mod collectives;
pub mod correlate;
pub mod dto;
pub mod error;
pub mod filter;
pub mod inspect;
pub mod prep;
mod query_sql;
pub mod scope;
pub mod search;
pub mod slices;
pub mod stats;
pub mod summary;
pub mod time;
pub mod timeline;

pub use collectives::collectives;
pub use correlate::correlate;
pub use dto::{
    CapabilitiesDto, CaptureFlagsRow, CollectiveRankRow, CollectiveRow, CollectivesAuxiliary,
    CollectivesResponse, CorrelateResponse, CorrelateRow, EventDetails, EventListAuxiliary,
    EventRef, InspectResponse, InspectRow, LinkRef, PrepAuxiliary, PrepResponse, PrepRow,
    SearchResponse, SliceAggregateRow, SliceInstanceRow, SliceRow, SlicesAuxiliary, SlicesResponse,
    StatsAuxiliary, StatsResponse, StatsRow, SummaryAuxiliary, SummaryResponse, TimeRangeDto,
    TimelineAuxiliary, TimelineBucketRow, TimelineResponse, TraceFileRow, TraceFileSchemaSurveyDto,
    TraceSchemaSurveyDto, TypedArgCoverageDto, TypedArgs,
};
pub use error::{PytorchQueryError, PytorchQueryResult, SqlPhase};
pub use filter::{EventFilterRequest, TypeSelection, TypeToken, parse_type_selection};
pub use inspect::inspect;
pub use prep::prep_response;
pub use scope::{PytorchScope, RankScope};
pub use search::search;
pub use slices::slices;
pub use stats::stats;
pub use summary::summary;
pub use time::{parse_group_by, resolve_time_window};
pub use timeline::timeline;
