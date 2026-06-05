//! veloq-nsys — Nsight Systems profile source for the veloq CLI.
//!
//! This crate owns the entire NSys subcommand surface: the clap arg
//! types (`cli::Cmd` + the shared `filters` arg groups), the
//! per-command dispatch (`commands::run`), the CSV/table output
//! adapters (`format` + `views`), the `--help` long_about builder
//! (`help`), and the schema-endpoint plumbing (`schema`). The
//! top-level `veloq` binary parses against `cli::Cmd` and hands the
//! parsed value to `commands::run`.
//!
//! The split exists so each profile source lives in its own crate with
//! its own internal modules, while the binary stays a thin
//! registry/dispatch shell.

pub mod cli;
pub mod commands;
pub mod filters;
pub mod format;
pub mod help;
pub mod output;
pub mod payloads;
pub mod schema;
pub mod schema_targets;
pub mod source;
pub mod views;

pub use cli::Cmd;
pub use source::NsysSource;
// Re-export the capability bitmap so meta verbs (`veloq/src/meta`)
// can run the same probe `summary.auxiliary.capabilities` exposes
// without taking a direct dep on `veloq-nsys-data`.
pub use veloq_nsys_data::{
    CapabilityFlags, Trace,
    nsys_rep::{generated_parquetdir_owner, is_valid_generated_parquetdir},
    trace_map,
};
