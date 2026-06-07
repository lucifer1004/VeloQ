//! Per-command flatteners — turn each Response type into a `TabularView`
//! for the CSV/table output formats. JSON keeps the original nested
//! shape; everything else funnels through this module.
//!
//! Design: each command has one "primary list" (rows / events / gaps /
//! slices / per_table / …). The primary list becomes the table body;
//! anything else from the envelope (counts, time window, NVTX scope)
//! becomes `meta` lines. When a response is non-tabular (e.g. `prep`
//! or `correlation-stats`), we emit a small `field` / `value`
//! key-value grid so the format is still meaningful.

mod basic;
mod events;
mod graph_slices;
mod hardware;
mod key_value;
mod meta;
mod metrics;
mod stats;
mod summary;

pub use basic::{concurrency_view, gaps_view, search_view, timeline_view};
pub use events::{correlate_view, inspect_view};
pub use graph_slices::{graph_replays_view, slices_view};
pub use hardware::hardware_view;
pub use key_value::key_value_view;
pub use metrics::metrics_view;
pub use stats::{stats_by_size_view, stats_view};
pub use summary::summary_view;
