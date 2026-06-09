//! Source-neutral query semantics.
//!
//! These helpers define VeloQ's cross-source query grammar and small
//! projections of that grammar into backend pattern languages. They do
//! not build SQL statements, own bind values, know DuckDB types, or know
//! NSys/PyTorch event schemas. Source crates adapt these semantics to
//! their own storage backend.

mod axis;
mod limit;
mod name_match;
mod time_window;
mod timeline_bucket;

pub use axis::{AxisParentError, AxisUsage};
pub use limit::{LimitError, LimitRef, LimitedRows};
pub use name_match::{
    NameFilterRef, NameMatchError, NameMatcher, glob_regex, glob_sql_like, sql_like_matches,
};
pub use time_window::{
    WindowRef, clipped_interval, clipped_interval_duration_ns, interval_overlaps,
};
pub use timeline_bucket::{TimelineBucket, timeline_bucket_key};
