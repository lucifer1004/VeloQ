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
