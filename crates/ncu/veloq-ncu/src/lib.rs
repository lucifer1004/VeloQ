//! `veloq-ncu` — read Nsight Compute `.ncu-rep` reports.
//!
//! Ingestion goes through NVIDIA's official `ncu_report` Python API:
//! a bundled helper exports each report to a leak-free
//! native sidecar under `<report>.veloq/` ([`native`]), and every verb
//! reads that sidecar. No NVIDIA proto schemas are vendored — Nsight
//! Compute is required only at prep/ingest time (it provides the
//! `ncu_report` module); query-time is NCU-free. Disasm fidelity comes
//! from an ELF scan of the embedded cubins fed to the existing
//! nvdisasm/cuobjdump pipeline ([`disasm_pipeline`]).
//!
//! The data crate stays parallel to `veloq-nsys-data` rather than nested
//! inside it: the NSys and NCU stacks share nothing concrete today
//! (no schema, no query layer, no time origin), and the only thing
//! we want to share — output formatting and the response envelope —
//! lives upstream in `veloq-core`.

pub mod cli;
pub mod disasm;
pub mod disasm_pipeline;
pub mod error;
pub mod glob;
pub mod help;
pub mod inspect;
pub mod launches;
pub mod lists;
pub mod metrics;
pub mod native;
pub mod row_id;
pub mod schema;
pub mod source;
pub mod source_metrics;
pub mod views;
pub mod warp_stalls;

pub use cli::Cmd;
pub use error::{NcuSourceError, NcuSourceResult};
pub use source::NcuSource;
