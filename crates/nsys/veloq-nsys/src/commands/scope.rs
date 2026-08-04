use std::path::Path;
use veloq_core::{AppliedScope, OutputFormat, SourceExecution, TraceSpan};
use veloq_nsys_data::scope::{ResolveError, ResolvedScope, ScopeRequest, resolve_scope};
use veloq_nsys_query::KindFilter;

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
    resident_trace: Option<&veloq_nsys_data::Trace>,
    fmt: OutputFormat,
    verb: &str,
    trace_span: Option<TraceSpan>,
    req: ScopeRequest,
    output: &mut SourceExecution,
) -> NsysSourceResult<Option<ResolvedScope>> {
    let opened;
    let trace_handle = match resident_trace {
        Some(trace) => trace,
        None => {
            opened = veloq_nsys_data::Trace::open(trace_path).map_err(|source| {
                NsysSourceError::scope_trace_open(trace_path.display(), source)
            })?;
            &opened
        }
    };
    match resolve_scope(trace_handle, req) {
        Ok(scope) => Ok(Some(scope)),
        Err(ResolveError::Ambiguous(amb)) => {
            emit_ambiguity_error(verb, trace_path, trace_span, &amb, fmt, output);
            Ok(None)
        }
        Err(ResolveError::Probe(e)) => Err(NsysSourceError::from(*e)),
    }
}

pub(super) struct KindScopeRequest<'a> {
    pub kinds: &'a KindFilter,
    pub location: &'a GpuLocationFilters,
}

/// Resolve a kind-aware list query without fabricating a CUDA location
/// filter for an explicitly host-only kind set.
///
/// The ordinary resolver auto-selects the sole process/device scope so
/// CUDA queries work without flags. Runtime, OS runtime, NVTX, and the
/// other non-location-bearing kinds have no device/stream columns, so
/// feeding that inferred device back into their SQL is invalid. Keep an
/// explicit process selector, but leave the CUDA axes unset when the user
/// did not request them. Explicit CUDA location flags still take the
/// ordinary resolver and validation path unchanged.
pub(super) fn resolve_kind_scope_or_refuse(
    trace_path: &Path,
    resident_trace: Option<&veloq_nsys_data::Trace>,
    fmt: OutputFormat,
    verb: &str,
    trace_span: Option<TraceSpan>,
    req: KindScopeRequest<'_>,
    output: &mut SourceExecution,
) -> NsysSourceResult<Option<ResolvedScope>> {
    let explicitly_host_only = matches!(
        req.kinds,
        KindFilter::Only(explicit)
            if !explicit.is_empty() && explicit.iter().all(|kind| !kind.is_location_bearing())
    );
    let has_cuda_location_request =
        req.location.device.is_some() || req.location.stream.is_some() || req.location.all_devices;

    if explicitly_host_only && !has_cuda_location_request {
        return Ok(Some(ResolvedScope {
            applied: AppliedScope {
                device: None,
                stream: None,
                native_pid: req.location.process,
                kind: None,
                nvtx_pattern: None,
                time_window_ns: None,
                aggregated_over: Vec::new(),
            },
        }));
    }

    resolve_or_refuse(
        trace_path,
        resident_trace,
        fmt,
        verb,
        trace_span,
        scope_request_from(req.location),
        output,
    )
}

/// Convenience: build a `ScopeRequest` from the CLI args a list verb
/// pulled out of its `Cmd::*` variant. Keeps per-verb arms terse.
pub(super) fn scope_request_from(location: &GpuLocationFilters) -> ScopeRequest {
    ScopeRequest {
        process: location.process,
        device: location.device,
        stream: location.stream,
        all_devices: location.all_devices,
        implicit_all_devices: false,
    }
}

pub(super) fn scope_request_from_device(location: &DeviceLocationFilters) -> ScopeRequest {
    ScopeRequest {
        process: location.process,
        device: location.device,
        stream: None,
        all_devices: location.all_devices,
        implicit_all_devices: false,
    }
}

pub(super) fn scope_request_from_device_with_implicit_all(
    location: &DeviceLocationFilters,
) -> ScopeRequest {
    ScopeRequest {
        process: location.process,
        device: location.device,
        stream: None,
        all_devices: location.all_devices,
        implicit_all_devices: true,
    }
}
