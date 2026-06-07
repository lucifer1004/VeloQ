use std::path::Path;
use veloq_core::{OutputFormat, TraceSpan};
use veloq_nsys_data::scope::{ResolveError, ResolvedScope, ScopeRequest, resolve_scope};

use crate::error::{NsysSourceError, NsysSourceResult};
use crate::filters::{DeviceLocationFilters, GpuLocationFilters};
use crate::output::emit_ambiguity_error;

/// Open the trace, resolve scope, and either return the resolved
/// scope to the caller or emit a structured ambiguity-refusal error
/// envelope and signal the caller to stop.
///
/// When the trace has >1 device and the user gave
/// neither `--device` nor `--all-devices`, the resolver refuses and
/// veloq returns `Ok(None)`. The error envelope (including the
/// `multi-device-ambiguous` warning code) has
/// already been written to stdout; the caller's verb arm should
/// `return Ok(())` immediately.
///
/// `Ok(Some(scope))` means the query is unambiguous and the verb
/// should continue, using `scope.applied` for both the envelope's
/// `meta.applied_scope` block and any SQL WHERE clauses it builds.
pub(super) fn resolve_or_refuse(
    trace_path: &Path,
    fmt: OutputFormat,
    verb: &str,
    trace_span: Option<TraceSpan>,
    req: ScopeRequest,
) -> NsysSourceResult<Option<ResolvedScope>> {
    let trace_handle = veloq_nsys_data::Trace::open(trace_path)
        .map_err(|source| NsysSourceError::scope_trace_open(trace_path.display(), source))?;
    match resolve_scope(&trace_handle, req) {
        Ok(scope) => Ok(Some(scope)),
        Err(ResolveError::Ambiguous(amb)) => {
            emit_ambiguity_error(verb, trace_path, trace_span, &amb, fmt);
            Ok(None)
        }
        Err(ResolveError::Probe(e)) => Err(NsysSourceError::from(*e)),
    }
}

/// Convenience: build a `ScopeRequest` from the CLI args a list verb
/// pulled out of its `Cmd::*` variant. Keeps per-verb arms terse.
pub(super) fn scope_request_from(location: &GpuLocationFilters) -> ScopeRequest {
    ScopeRequest {
        device: location.device,
        stream: location.stream,
        all_devices: location.all_devices,
    }
}

pub(super) fn scope_request_from_device(location: &DeviceLocationFilters) -> ScopeRequest {
    ScopeRequest {
        device: location.device,
        stream: None,
        all_devices: location.all_devices,
    }
}
