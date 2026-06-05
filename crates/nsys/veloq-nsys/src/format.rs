//! Output-format re-exports for NSys command rendering.
//!
//! The generic CSV/table machinery lives in `veloq-core`; NSys keeps
//! this module so existing command/view code can continue importing
//! `crate::format::*`.

pub use veloq_core::OutputFormat as Format;
pub use veloq_core::tabular::{DISPLAY_PRECISION, TabularView, cell_opt, emit_csv, emit_table};
