//! veloq-core — cross-source primitives for the veloq CLI family.
//!
//! Intentionally tiny: the source-qualified response [`Envelope`] +
//! [`EnvelopeError`], the [`ProfileSource`] trait every backend
//! implements, and a few cross-source utilities (sort spec, time
//! literals, wire-format projector). Source-specific types live in
//! their respective source crates.

pub mod artifacts;
pub mod envelope;
pub mod guards;
pub mod meta;
pub mod recipes;
pub mod sidecar;
pub mod sort;
pub mod source;
pub mod tabular;
pub mod time;
pub mod wire_format;

pub use artifacts::{ARTIFACT_DIR_SUFFIX, artifact_dir_for};
pub use envelope::{
    ENVELOPE_VERSION, Envelope, EnvelopeError, EnvelopeErrorDetails, EnvelopeTraceRef, SourceRef,
    TraceSpan, emit_envelope, write_error_envelope,
};
pub use meta::{AppliedScope, NextStep, ResponseMeta, Warning, WarningCode, WarningSeverity};
pub use sidecar::{SidecarCache, SidecarHeader, SourceFingerprint};
pub use sort::{Direction, SortField, SortKeyDef, SortKeySpec, SortSpec, sort_in_memory};
pub use source::{OutputFormat, ProfileSource};
