//! Source-neutral data processing helpers for VeloQ backends.
//!
//! Keep this crate free of profile-source semantics. It may know how to
//! fingerprint files, read compressed text, and publish Parquet sidecars, but
//! it must not depend on NSys, NCU, PyTorch, or a future source crate.

pub mod error;
pub mod file;
pub mod parquet;

pub use error::{DataError, DataResult};
