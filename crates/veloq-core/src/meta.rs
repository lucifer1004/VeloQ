//! Envelope-level metadata block (`Envelope.meta`) — applied scope echo,
//! next-step hints, and silent-failure warnings.
//!
//! The envelope at schema `"v1"` carries an optional
//! top-level `meta: { applied_scope?, next_steps?, warnings? }` block
//! that every list verb populates with the same shape, so an agent
//! reads one structural location regardless of source.
//!
//! ## Wire-shape contract
//!
//! - `Envelope.meta` is `Option<ResponseMeta>`; omitted from serialized
//!   JSON when None. Meta verbs that don't operate on event rows
//!   (`info`, `sources`, `schema`) emit `meta: None` as the default.
//! - Every list verb that accepts scope filters (`stats`, `search`,
//!   `slices`, `gaps`, `timeline`) MUST emit a non-null
//!   `meta.applied_scope` on success. A successful envelope from those
//!   verbs without `meta.applied_scope` is a bug. The
//!   `wire_format_smoke` test suite enforces this structurally.
//! - The companion ADR guarantees that on a multi-device
//!   trace, a successful response either picked a single resolved
//!   scope (`device: Some(0)`) or explicitly opted into aggregation
//!   (`aggregated_over: ["device"]`). The combination "applied_scope
//!   present but every field null and `aggregated_over` empty" is
//!   impossible — the resolver refuses the query before reaching this
//!   point.

use serde::Serialize;

/// Top-level metadata carried alongside every list response. All three
/// sub-fields are independently optional; `ResponseMeta` itself is
/// `Option`-wrapped on the envelope and omitted entirely when empty.
#[derive(Debug, Default, Clone, Serialize, schemars::JsonSchema)]
pub struct ResponseMeta {
    /// Per-verb echo of the scope filters that produced this response.
    /// Required (non-None) on success envelopes from every list verb
    /// that accepts location filters; omitted on meta verbs that don't
    /// operate on a trace.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub applied_scope: Option<AppliedScope>,

    /// Up to three suggested follow-up commands, each pointing at the
    /// next obvious operation given this response. Capped to keep the
    /// meta block scannable for an agent that surfaces it to a human.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub next_steps: Vec<NextStep>,

    /// Risk-on-the-answer notices fired by the guardrail layer (e.g.
    /// `narrow-window`, `empty-with-scope`). A warning is emitted
    /// alongside a successful response; it never represents query
    /// failure (which uses `EnvelopeError`).
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub warnings: Vec<Warning>,
}

impl ResponseMeta {
    /// `true` iff every sub-field is empty. Used by serialisers that
    /// want to omit the whole `meta` field rather than emit `meta: {}`.
    pub fn is_empty(&self) -> bool {
        self.applied_scope.is_none() && self.next_steps.is_empty() && self.warnings.is_empty()
    }
}

/// Per-verb echo of the scope that was applied to produce the response.
///
/// §applied_scope: each field carries the **single
/// resolved value** the verb applied. Absent (`None`) means the filter
/// was not set by the user *and* the trace did not require it (only a
/// single value was available). The `aggregated_over` vector enumerates
/// axes the user explicitly opted into aggregation for via
/// `--all-devices` — when an axis name appears there, the corresponding
/// scalar field is `None` and the response is the aggregate.
#[derive(Debug, Default, Clone, Serialize, schemars::JsonSchema)]
pub struct AppliedScope {
    /// Resolved CUDA `deviceId`. `None` when not set; appears in
    /// `aggregated_over` when `--all-devices` was used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<i32>,
    /// Resolved CUDA `streamId`. Streams are exempt from ambiguity
    /// refusal; this is plain echo.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<i64>,
    /// Resolved native_pid (high 24 bits of `globalTid`) of the host
    /// process that ran work on the resolved device. **Resolver output,
    /// not a user input** — populated by the device→native_pid bridge
    /// when the caller picked `--device <N>`. `None` when no device
    /// filter was set, when `TARGET_INFO_CUDA_CONTEXT_INFO` is absent,
    /// or when `--all-devices` was used. The wire field is
    /// deliberately named `native_pid` (the OS-level concept) rather
    /// than `rank` (the distributed-runtime cohort index from `RANK` /
    /// `SLURM_PROCID` / `MPI_COMM_WORLD_RANK` env vars), which veloq
    /// does not currently resolve. A future ADR may add a
    /// `cohort_rank` field once that resolution lands.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_pid: Option<i64>,
    /// Event-kind filter (`--type kernel,memcpy,...`). Comma-joined for
    /// the wire; absent when default `all`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// NVTX glob pattern (`--nvtx 'step_*'`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_pattern: Option<String>,
    /// Resolved `--time-range`, if any (absolute ns).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_window_ns: Option<(i64, i64)>,
    /// Named axes the user opted into aggregation for; subset of
    /// `["device"]`. Empty = no opt-in aggregation.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub aggregated_over: Vec<String>,
}

/// One suggested follow-up command.
///
/// §next_steps: `command` is a complete, ready-to-run
/// `veloq <verb> ...` invocation string. Template owners are
/// responsible for shell-safe quoting of user-derived values embedded
/// in it (NVTX names, file paths, row_ids). Emitters MUST NOT inject
/// jq, pipelines, or non-veloq commands.
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct NextStep {
    /// One-sentence agent-facing explanation of *why* this step.
    pub hint: String,
    /// `veloq <verb> ...` invocation string. Shell-safe.
    pub command: String,
}

/// Risk-on-the-answer notice. Emitted alongside successful responses,
/// not in place of them.
///
/// §warnings: `severity` is the closed enum
/// `"info" | "warn"`; `code` is a namespaced [`WarningCode`] string;
/// `message` is human-readable and not required to be stable.
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct Warning {
    pub severity: WarningSeverity,
    pub code: WarningCode,
    pub message: String,
}

/// Closed enum for the `severity` field. Serialises as a lowercase
/// string `"info"` or `"warn"`.
#[derive(Debug, Clone, Copy, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum WarningSeverity {
    /// Informational notice; the answer is correct but the agent may
    /// want to double-check (e.g. low PM-counter coverage).
    Info,
    /// Material risk on the answer (e.g. silent empty result, suspect
    /// time-window).
    Warn,
}

/// Closed enum for the `code` field. Serialises as the kebab-case
/// string variant. New codes are additive; existing codes are stable
/// across versions.
#[derive(Debug, Clone, Copy, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum WarningCode {
    /// `--from`/`--to` window covers <1% of trace_span and was parsed
    /// from raw-integer endpoints (no unit suffix, no `@` prefix).
    /// Suggests the agent forgot units.
    NarrowWindow,
    /// Query returned zero rows but a scope filter was active. The
    /// filter likely excluded all rows; suggest broadening.
    EmptyWithScope,
    /// Trace has more than one device and the agent did not pick a
    /// scope or opt into aggregation. Emitted as `meta.warnings` on an
    /// error envelope.
    MultiDeviceAmbiguous,
    /// PyTorch trace has more than one distributed rank and the agent
    /// did not pick a rank or opt into aggregation. Emitted as
    /// `meta.warnings` on an error envelope.
    MultiRankAmbiguous,
    /// PM-counter coverage on the queried time window is below the
    /// "trust" threshold (per `metrics --type gpu|nic` semantics).
    CoverageLow,
    /// Trace claimed a schema version veloq doesn't recognise; verb
    /// fell back to a permissive adapter.
    SchemaFallback,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn at<'a>(v: &'a Value, ptr: &str) -> anyhow::Result<&'a Value> {
        v.pointer(ptr)
            .ok_or_else(|| anyhow::anyhow!("missing pointer `{ptr}` in {v}"))
    }

    #[test]
    fn empty_meta_serialises_all_optional_fields_absent() -> anyhow::Result<()> {
        let m = ResponseMeta::default();
        assert!(m.is_empty());
        let json = serde_json::to_value(&m)?;
        // All three sub-fields are absent from the serialised object.
        assert!(json.get("applied_scope").is_none());
        assert!(json.get("next_steps").is_none());
        assert!(json.get("warnings").is_none());
        Ok(())
    }

    #[test]
    fn applied_scope_round_trip_with_aggregated_over() -> anyhow::Result<()> {
        let m = ResponseMeta {
            applied_scope: Some(AppliedScope {
                device: None,
                stream: None,
                native_pid: Some(1234),
                kind: Some("kernel".into()),
                nvtx_pattern: Some("step_*".into()),
                time_window_ns: Some((0, 1_000_000_000)),
                aggregated_over: vec!["device".into()],
            }),
            ..ResponseMeta::default()
        };
        let json = serde_json::to_value(&m)?;
        assert_eq!(at(&json, "/applied_scope/native_pid")?.as_i64(), Some(1234));
        assert_eq!(
            at(&json, "/applied_scope/aggregated_over/0")?.as_str(),
            Some("device")
        );
        // `device` is None → omitted from the wire.
        assert!(json.pointer("/applied_scope/device").is_none());
        Ok(())
    }

    #[test]
    fn warning_severity_serialises_lowercase() -> anyhow::Result<()> {
        let w = Warning {
            severity: WarningSeverity::Warn,
            code: WarningCode::EmptyWithScope,
            message: "filter excluded all rows".into(),
        };
        let json = serde_json::to_value(&w)?;
        assert_eq!(at(&json, "/severity")?.as_str(), Some("warn"));
        assert_eq!(at(&json, "/code")?.as_str(), Some("empty-with-scope"));
        Ok(())
    }

    #[test]
    fn warning_code_kebab_case_for_every_variant() -> anyhow::Result<()> {
        let codes = [
            (WarningCode::NarrowWindow, "narrow-window"),
            (WarningCode::EmptyWithScope, "empty-with-scope"),
            (WarningCode::MultiDeviceAmbiguous, "multi-device-ambiguous"),
            (WarningCode::MultiRankAmbiguous, "multi-rank-ambiguous"),
            (WarningCode::CoverageLow, "coverage-low"),
            (WarningCode::SchemaFallback, "schema-fallback"),
        ];
        for (c, expected) in codes {
            let v = serde_json::to_value(c)?;
            assert_eq!(
                v.as_str(),
                Some(expected),
                "{c:?} should serialise as {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn next_step_round_trip() -> anyhow::Result<()> {
        let n = NextStep {
            hint: "drill into top row".into(),
            command: "veloq inspect path/to/trace nvtx:42".into(),
        };
        let json = serde_json::to_value(&n)?;
        assert_eq!(at(&json, "/hint")?.as_str(), Some("drill into top row"));
        assert_eq!(
            at(&json, "/command")?.as_str(),
            Some("veloq inspect path/to/trace nvtx:42")
        );
        Ok(())
    }
}
