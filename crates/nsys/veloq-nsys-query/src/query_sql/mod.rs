//! Small SQL helpers shared by NSys query verbs.
//!
//! This is intentionally a fragment layer, not a query builder. Each verb
//! still owns its main SQL shape; helpers here carry repeated event-kind,
//! window, and sort semantics with bind parameters kept next to the SQL
//! text that introduced them.

pub(crate) mod event_scan;
pub(crate) mod event_semantics;
pub(crate) mod exec;
pub(crate) mod gpu_work;
pub(crate) mod sample_scan;
pub(crate) mod sort;
