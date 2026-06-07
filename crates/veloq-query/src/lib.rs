//! DuckDB-backed query helpers shared by VeloQ profile backends.
//!
//! This crate is intentionally a small SQL fragment and execution layer. It
//! does not encode source-specific profile semantics; NSys, PyTorch, and
//! future backends keep their event/table meanings in their own query crates.

pub mod duckdb;
pub mod sql;
