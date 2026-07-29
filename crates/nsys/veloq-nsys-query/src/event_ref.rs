//! [`EventRef`] — the shared "row that references one trace event"
//! shape every list-of-events response returns.
//!
//! `inspect` / `search` / `correlate` all return this one type, so an
//! agent sees the same fields in the same positions and can write one
//! jq snippet that lifts an event reference out of any of them.
//!
//! `EventRef` is a `#[serde(tag = "type")]` enum so every row carries a
//! top-level `type` discriminator (matching the `EventDetails`
//! convention used by [`crate::inspect`]) and so a few headline
//! columns — kernel grid/block/registers/shared/mangled/demangled,
//! memcpy/memset bytes, nvtx event_type/domain — surface in the
//! row itself rather than requiring a follow-up `inspect` call. The
//! remaining 8 kinds carry the shared payload via [`EventRefBase`].
//!
//! The optional [`NvtxContext`] block lands on every variant's
//! shared base. It carries the innermost NVTX range that contained
//! the event, plus the range's nesting depth and a 0-based
//! `iter_index` among same-name repeats.

use crate::{EventKind, NsysQueryError, NsysQueryResult, RowId};
use serde::Serialize;

/// Shared fields on every `EventRef` variant. Lives in a separate
/// struct so the per-kind variants `#[serde(flatten)]` it instead of
/// re-typing the 10 fields each time.
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct EventRefBase {
    /// Cross-trace key — equal to `row_id` stringified. Lets
    /// agents `INDEX(.events; .key)` against `inspect` / `correlate`
    /// output without recomputing the kind:rowid string.
    pub key: String,
    pub row_id: RowId,
    pub name: String,
    pub start_ns: i64,
    pub duration_ns: i64,
    /// Native process owning this event's CUDA or host-thread namespace.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub process_id: Option<i64>,
    /// GPU events: device id. CPU events / NVTX: `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_id: Option<i32>,
    /// GPU events: stream id. CPU events / NVTX: `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_id: Option<i64>,
    /// CPU / NVTX events: serialised global thread id. GPU events: `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub global_tid: Option<i64>,
    /// NVTX-only: depth in the per-(global_tid, domain_id) nesting
    /// stack. 0 = outermost. `None` for non-NVTX events and when the
    /// nesting computation was skipped.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth: Option<u8>,
    /// Innermost NVTX range that contained this event. Populated by
    /// `inspect` (always when an NVTX_EVENTS table is present) and by
    /// `search --with-nvtx` (opt-in). `None` everywhere else; agents
    /// reading `.nvtx_context` should handle absence gracefully.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_context: Option<NvtxContext>,
}

/// Per-kind kernel headline. `grid` / `block` are 3-tuples; the
/// remaining fields are sparse (older NSys schemas may omit
/// `registersPerThread`, `mangledName`, etc., in which case they
/// serialise as absent).
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct EventRefKernel {
    #[serde(flatten)]
    pub base: EventRefBase,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grid: Option<[i64; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block: Option<[i64; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub registers_per_thread: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub static_shared_memory: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic_shared_memory: Option<i64>,
    /// Full C++-demangled symbol — already projected by `inspect`,
    /// surfaced here so agents don't need a second hop for
    /// readability sort axes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub demangled_name: Option<String>,
    /// Raw mangled symbol (absent on older NSys reports that don't ship
    /// a `mangledName` column).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mangled_name: Option<String>,
}

/// Per-kind memcpy headline. `copy_kind` is the integer enum value;
/// `copy_kind_name` is the resolved label (`"HtoD"`, `"DtoH"`, etc.).
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct EventRefMemcpy {
    #[serde(flatten)]
    pub base: EventRefBase,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub copy_kind: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub copy_kind_name: Option<&'static str>,
}

/// Per-kind memset headline.
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct EventRefMemset {
    #[serde(flatten)]
    pub base: EventRefBase,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<i64>,
    /// The byte-value being written. `None` on older NSys reports
    /// missing the `value` column.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<i64>,
}

/// Per-kind NVTX headline — distinguishes `PushPop` ranges from
/// `Mark` events and exposes the user-defined domain id so agents
/// can filter by domain without a second query.
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct EventRefNvtx {
    #[serde(flatten)]
    pub base: EventRefBase,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_type: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain_id: Option<i64>,
}

/// Reference to one event row by stable identity (`row_id`), plus
/// the headline columns every consumer wants without paying for
/// `inspect` on each entry.
///
/// Every list-of-events surface (search, correlate's flat
/// events, future `ncu launches`, …) returns `Vec<EventRef>` so
/// agents can write one jq recipe (`INDEX(.data.rows; .key)` etc.)
/// regardless of which verb produced the list. The `type`
/// discriminator (`"kernel"`, `"memcpy"`, `"sync"`, …) selects
/// which variant payload follows; absent kind-specific fields
/// (e.g. `grid` on a `sync` row) serialise as missing rather than
/// `null`.
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum EventRef {
    Kernel(EventRefKernel),
    Memcpy(EventRefMemcpy),
    Memset(EventRefMemset),
    Nvtx(EventRefNvtx),
    Sync(EventRefBase),
    Runtime(EventRefBase),
    Osrt(EventRefBase),
    Graph(EventRefBase),
    #[serde(rename = "graph_node")]
    GraphNode(EventRefBase),
    #[serde(rename = "graph_event")]
    GraphEvent(EventRefBase),
    #[serde(rename = "cuda_event")]
    CudaEvent(EventRefBase),
    Overhead(EventRefBase),
}

impl EventRef {
    /// Borrow the [`EventRefBase`] underneath every variant. Lets
    /// downstream consumers (correlate, view rendering, sort) reach
    /// the shared fields without matching on the discriminator.
    pub fn base(&self) -> &EventRefBase {
        match self {
            EventRef::Kernel(k) => &k.base,
            EventRef::Memcpy(m) => &m.base,
            EventRef::Memset(m) => &m.base,
            EventRef::Nvtx(n) => &n.base,
            EventRef::Sync(b)
            | EventRef::Runtime(b)
            | EventRef::Osrt(b)
            | EventRef::Graph(b)
            | EventRef::GraphNode(b)
            | EventRef::GraphEvent(b)
            | EventRef::CudaEvent(b)
            | EventRef::Overhead(b) => b,
        }
    }

    /// Mutable accessor for [`EventRefBase`]. Used by the
    /// `--with-nvtx` post-decoration pass to attach `nvtx_context`
    /// to rows that already exist in the response.
    pub fn base_mut(&mut self) -> &mut EventRefBase {
        match self {
            EventRef::Kernel(k) => &mut k.base,
            EventRef::Memcpy(m) => &mut m.base,
            EventRef::Memset(m) => &mut m.base,
            EventRef::Nvtx(n) => &mut n.base,
            EventRef::Sync(b)
            | EventRef::Runtime(b)
            | EventRef::Osrt(b)
            | EventRef::Graph(b)
            | EventRef::GraphNode(b)
            | EventRef::GraphEvent(b)
            | EventRef::CudaEvent(b)
            | EventRef::Overhead(b) => b,
        }
    }

    /// Wrap an [`EventRefBase`] in the variant matching `kind`, with
    /// every kind-specific field defaulted to `None`. Used by
    /// callers (correlate's `fetch_summaries`) that don't project
    /// the per-kind headline columns and only have the shared base
    /// available. Errors when `kind` is `CpuSample`, which has no
    /// EventRef variant.
    pub fn from_base(kind: EventKind, base: EventRefBase) -> NsysQueryResult<Self> {
        Ok(match kind {
            EventKind::Kernel => EventRef::Kernel(EventRefKernel {
                base,
                grid: None,
                block: None,
                registers_per_thread: None,
                static_shared_memory: None,
                dynamic_shared_memory: None,
                demangled_name: None,
                mangled_name: None,
            }),
            EventKind::Memcpy => EventRef::Memcpy(EventRefMemcpy {
                base,
                bytes: None,
                copy_kind: None,
                copy_kind_name: None,
            }),
            EventKind::Memset => EventRef::Memset(EventRefMemset {
                base,
                bytes: None,
                value: None,
            }),
            EventKind::Nvtx => EventRef::Nvtx(EventRefNvtx {
                base,
                event_type: None,
                domain_id: None,
            }),
            EventKind::Sync => EventRef::Sync(base),
            EventKind::Runtime => EventRef::Runtime(base),
            EventKind::Osrt => EventRef::Osrt(base),
            EventKind::Graph => EventRef::Graph(base),
            EventKind::GraphNode => EventRef::GraphNode(base),
            EventKind::GraphEvent => EventRef::GraphEvent(base),
            EventKind::CudaEvent => EventRef::CudaEvent(base),
            EventKind::Overhead => EventRef::Overhead(base),
            EventKind::CpuSample => {
                return Err(NsysQueryError::SearchCpuSampleUnsupported);
            }
        })
    }
}

/// The innermost NVTX range containing an event, with enough fields
/// for an agent to attribute the event to a per-iteration scope
/// without a second query.
///
/// `iter_index` is the 0-based ordinal of this range among repeats
/// with the same (tid, domain, name) — answers "which iteration?"
/// directly when an agent walks a `search` over multiple `step_*`
/// instances.
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct NvtxContext {
    pub range_id: RowId,
    pub name: String,
    pub depth: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iter_index: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Result, anyhow};
    use serde_json::Value;

    fn base(kind: EventKind, rowid: i64, name: &str) -> EventRefBase {
        let row_id = RowId::new(kind, rowid);
        EventRefBase {
            key: row_id.to_string(),
            row_id,
            name: name.to_string(),
            start_ns: 100,
            duration_ns: 25,
            process_id: None,
            device_id: None,
            stream_id: None,
            global_tid: None,
            depth: None,
            nvtx_context: None,
        }
    }

    fn string_field<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
        value
            .get(field)
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("missing string field `{field}` in {value}"))
    }

    fn i64_field(value: &Value, field: &str) -> Result<i64> {
        value
            .get(field)
            .and_then(Value::as_i64)
            .ok_or_else(|| anyhow!("missing integer field `{field}` in {value}"))
    }

    #[test]
    fn event_ref_contract_serializes_tagged_kernel_with_shared_and_headline_fields() -> Result<()> {
        let mut base = base(EventKind::Kernel, 7, "kernel_name");
        base.device_id = Some(0);
        base.stream_id = Some(3);
        let value = serde_json::to_value(EventRef::Kernel(EventRefKernel {
            base,
            grid: Some([1, 2, 3]),
            block: Some([4, 5, 6]),
            registers_per_thread: Some(32),
            static_shared_memory: Some(128),
            dynamic_shared_memory: Some(256),
            demangled_name: Some("void kernel_name()".to_string()),
            mangled_name: Some("_Z11kernel_namev".to_string()),
        }))?;

        assert_eq!(string_field(&value, "type")?, "kernel");
        assert_eq!(string_field(&value, "key")?, "kernel:7");
        assert_eq!(string_field(&value, "row_id")?, "kernel:7");
        assert_eq!(string_field(&value, "name")?, "kernel_name");
        assert_eq!(i64_field(&value, "start_ns")?, 100);
        assert_eq!(i64_field(&value, "duration_ns")?, 25);
        assert_eq!(i64_field(&value, "device_id")?, 0);
        assert_eq!(i64_field(&value, "stream_id")?, 3);
        assert_eq!(i64_field(&value, "registers_per_thread")?, 32);
        assert_eq!(i64_field(&value, "static_shared_memory")?, 128);
        assert_eq!(i64_field(&value, "dynamic_shared_memory")?, 256);
        assert_eq!(
            string_field(&value, "demangled_name")?,
            "void kernel_name()"
        );
        assert_eq!(string_field(&value, "mangled_name")?, "_Z11kernel_namev");

        let grid: Vec<i64> = value
            .get("grid")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("missing grid array in {value}"))?
            .iter()
            .filter_map(Value::as_i64)
            .collect();
        assert_eq!(grid, vec![1, 2, 3]);
        let block: Vec<i64> = value
            .get("block")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("missing block array in {value}"))?
            .iter()
            .filter_map(Value::as_i64)
            .collect();
        assert_eq!(block, vec![4, 5, 6]);
        Ok(())
    }

    #[test]
    fn event_ref_contract_serializes_kind_specific_optional_headlines() -> Result<()> {
        let memcpy = serde_json::to_value(EventRef::Memcpy(EventRefMemcpy {
            base: base(EventKind::Memcpy, 8, "memcpy_h2d"),
            bytes: Some(4096),
            copy_kind: Some(1),
            copy_kind_name: Some("cudaMemcpyHostToDevice"),
        }))?;
        assert_eq!(string_field(&memcpy, "type")?, "memcpy");
        assert_eq!(string_field(&memcpy, "row_id")?, "memcpy:8");
        assert_eq!(i64_field(&memcpy, "bytes")?, 4096);
        assert_eq!(i64_field(&memcpy, "copy_kind")?, 1);
        assert_eq!(
            string_field(&memcpy, "copy_kind_name")?,
            "cudaMemcpyHostToDevice"
        );

        let memset = serde_json::to_value(EventRef::Memset(EventRefMemset {
            base: base(EventKind::Memset, 9, "memset"),
            bytes: Some(1024),
            value: Some(0),
        }))?;
        assert_eq!(string_field(&memset, "type")?, "memset");
        assert_eq!(string_field(&memset, "row_id")?, "memset:9");
        assert_eq!(i64_field(&memset, "bytes")?, 1024);
        assert_eq!(i64_field(&memset, "value")?, 0);

        let nvtx = serde_json::to_value(EventRef::Nvtx(EventRefNvtx {
            base: base(EventKind::Nvtx, 10, "step"),
            event_type: Some(60),
            domain_id: Some(2),
        }))?;
        assert_eq!(string_field(&nvtx, "type")?, "nvtx");
        assert_eq!(string_field(&nvtx, "row_id")?, "nvtx:10");
        assert_eq!(i64_field(&nvtx, "event_type")?, 60);
        assert_eq!(i64_field(&nvtx, "domain_id")?, 2);
        Ok(())
    }

    #[test]
    fn event_ref_contract_omits_unavailable_optional_fields() -> Result<()> {
        let value = serde_json::to_value(EventRef::Runtime(base(
            EventKind::Runtime,
            11,
            "cudaLaunchKernel",
        )))?;
        assert_eq!(string_field(&value, "type")?, "runtime");
        assert_eq!(string_field(&value, "key")?, "runtime:11");
        assert_eq!(string_field(&value, "row_id")?, "runtime:11");
        for absent in [
            "device_id",
            "stream_id",
            "global_tid",
            "depth",
            "nvtx_context",
            "grid",
            "bytes",
            "event_type",
        ] {
            assert!(
                value.get(absent).is_none(),
                "optional field `{absent}` should be omitted when unavailable: {value}"
            );
        }
        Ok(())
    }

    #[test]
    fn event_ref_contract_from_base_rejects_cpu_sample() {
        let outcome = EventRef::from_base(
            EventKind::CpuSample,
            base(EventKind::CpuSample, 12, "sample"),
        );
        assert!(matches!(
            outcome,
            Err(NsysQueryError::SearchCpuSampleUnsupported)
        ));
    }
}
