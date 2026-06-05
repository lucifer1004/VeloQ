//! Wire-format event identifiers.
//!
//! On the JSON surface every event is identified by a `RowId`
//! serialised as `"<kind>:<rowid>"` (e.g. `"kernel:1234"`). The
//! type-tagged form makes responses self-describing and lets `inspect`
//! dispatch without an extra column.

use serde::{Deserialize, Serialize, Serializer, de::Deserializer};
use std::fmt;
use std::str::FromStr;
use thiserror::Error;

/// The event tables veloq surfaces. Each maps to one NSys table.
///
/// **CpuSample is special.** It denotes a row in `COMPOSITE_EVENTS`
/// (one CPU IP sample with its callchain). Unlike every other kind,
/// it's intentionally **omitted from `EventKind::ALL`** — so `stats`
/// and `search`, which use `ALL` as their permissive allow-list,
/// reject `--type cpu_sample` rather than silently mis-aggregating
/// over a fundamentally different shape (point samples, not
/// intervals; no end_ns, no duration, no name column). The only path
/// that surfaces CpuSample is `inspect cpu_sample:N`, which joins
/// `COMPOSITE_EVENTS` × `SAMPLING_CALLCHAINS` × `StringIds` for the
/// resolved stack. Primary aggregation entry is
/// `veloq metrics --type cpu-sampling`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventKind {
    Kernel,
    Memcpy,
    Memset,
    Runtime,
    Osrt,
    Nvtx,
    Sync,
    Graph,
    GraphNode,
    GraphEvent,
    CudaEvent,
    Overhead,
    /// `COMPOSITE_EVENTS` row — one CPU IP sample plus a callchain
    /// chained from `SAMPLING_CALLCHAINS`. **Not** in `ALL`; see the
    /// type-level note above.
    CpuSample,
}

impl EventKind {
    pub const ALL: &'static [EventKind] = &[
        EventKind::Kernel,
        EventKind::Memcpy,
        EventKind::Memset,
        EventKind::Runtime,
        EventKind::Osrt,
        EventKind::Nvtx,
        EventKind::Sync,
        EventKind::Graph,
        EventKind::GraphNode,
        EventKind::GraphEvent,
        EventKind::CudaEvent,
        EventKind::Overhead,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            EventKind::Kernel => "kernel",
            EventKind::Memcpy => "memcpy",
            EventKind::Memset => "memset",
            EventKind::Runtime => "runtime",
            EventKind::Osrt => "osrt",
            EventKind::Nvtx => "nvtx",
            EventKind::Sync => "sync",
            EventKind::Graph => "graph",
            EventKind::GraphNode => "graph_node",
            EventKind::GraphEvent => "graph_event",
            EventKind::CudaEvent => "cuda_event",
            EventKind::Overhead => "overhead",
            EventKind::CpuSample => "cpu_sample",
        }
    }

    /// True iff `self`'s backing table carries `deviceId` /
    /// `streamId` (and `contextId` where applicable), letting it
    /// participate in `--device` / `--stream` filters and `--group-by
    /// device|stream|context` axes. The complement are *CPU-only
    /// host-thread* events that only have `globalTid` — running them
    /// through a location filter silently empties their bucket. The
    /// shared null-location filter policy uses this
    /// predicate to error on explicit `--type runtime --device 0`
    /// instead of dropping the kind silently.
    pub fn is_location_bearing(self) -> bool {
        matches!(
            self,
            EventKind::Kernel
                | EventKind::Memcpy
                | EventKind::Memset
                | EventKind::Sync
                | EventKind::Graph
                | EventKind::CudaEvent
        )
    }

    pub fn table(self) -> &'static str {
        match self {
            EventKind::Kernel => "CUPTI_ACTIVITY_KIND_KERNEL",
            EventKind::Memcpy => "CUPTI_ACTIVITY_KIND_MEMCPY",
            EventKind::Memset => "CUPTI_ACTIVITY_KIND_MEMSET",
            EventKind::Runtime => "CUPTI_ACTIVITY_KIND_RUNTIME",
            EventKind::Osrt => "OSRT_API",
            EventKind::Nvtx => "NVTX_EVENTS",
            EventKind::Sync => "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
            EventKind::Graph => "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
            EventKind::GraphNode => "CUDA_GRAPH_NODE_EVENTS",
            EventKind::GraphEvent => "CUDA_GRAPH_EVENTS",
            EventKind::CudaEvent => "CUPTI_ACTIVITY_KIND_CUDA_EVENT",
            EventKind::Overhead => "CUPTI_ACTIVITY_KIND_OVERHEAD",
            EventKind::CpuSample => "COMPOSITE_EVENTS",
        }
    }

    pub fn parse(s: &str) -> Option<EventKind> {
        match s.to_ascii_lowercase().as_str() {
            "kernel" | "kernels" => Some(EventKind::Kernel),
            "memcpy" => Some(EventKind::Memcpy),
            "memset" => Some(EventKind::Memset),
            "runtime" | "cuda_api" | "cuda" => Some(EventKind::Runtime),
            "osrt" | "osrt_api" => Some(EventKind::Osrt),
            "nvtx" => Some(EventKind::Nvtx),
            "sync" | "synchronization" => Some(EventKind::Sync),
            "graph" | "graph_trace" | "cuda_graph" => Some(EventKind::Graph),
            "graph_node" | "graphnode" | "node" => Some(EventKind::GraphNode),
            "graph_event" | "graph_events" | "graph_lifecycle" => Some(EventKind::GraphEvent),
            "cuda_event" | "cudaevent" | "cuda_event_record" => Some(EventKind::CudaEvent),
            "overhead" | "cupti_overhead" | "profiler_overhead" => Some(EventKind::Overhead),
            "cpu_sample" | "cpu_samples" | "cpusample" => Some(EventKind::CpuSample),
            _ => None,
        }
    }
}

impl fmt::Display for EventKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A type-tagged event identifier. Wire format: `"<kind>:<rowid>"`.
///
/// The underlying SQLite-compatible rowid is preserved verbatim — no
/// bit-packing arithmetic is applied here. This keeps the wire form
/// trivial to round-trip through scripts and other tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RowId {
    pub kind: EventKind,
    pub rowid: i64,
}

impl RowId {
    pub fn new(kind: EventKind, rowid: i64) -> Self {
        Self { kind, rowid }
    }
}

impl fmt::Display for RowId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.kind.as_str(), self.rowid)
    }
}

#[derive(Debug, Error)]
pub enum RowIdParseError {
    #[error(
        "invalid row_id `{0}`: missing `<kind>:<rowid>` separator. \
         Expected e.g. `kernel:1234`, `memcpy:567`, or `runtime:42`."
    )]
    NoColon(String),
    #[error(
        "invalid row_id: unknown event kind `{0}`. \
         Expected one of: kernel, memcpy, memset, runtime, osrt, nvtx, sync, graph, graph_node, graph_event, cuda_event, overhead, cpu_sample."
    )]
    UnknownKind(String),
    #[error("invalid row_id `{0}`: rowid portion is not a base-10 integer")]
    BadRowid(String),
}

impl FromStr for RowId {
    type Err = RowIdParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (kind_str, rowid_str) = s
            .split_once(':')
            .ok_or_else(|| RowIdParseError::NoColon(s.to_string()))?;
        let kind = EventKind::parse(kind_str)
            .ok_or_else(|| RowIdParseError::UnknownKind(kind_str.to_string()))?;
        let rowid: i64 = rowid_str
            .parse()
            .map_err(|_| RowIdParseError::BadRowid(s.to_string()))?;
        Ok(RowId { kind, rowid })
    }
}

impl Serialize for RowId {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        ser.collect_str(self)
    }
}

/// Hand-rolled because the actual wire form is a `"<kind>:<rowid>"`
/// string (see `Serialize` above), not the struct layout the auto
/// derive would emit. The schema reflects what callers will see —
/// a plain string with a documented pattern.
impl schemars::JsonSchema for RowId {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "RowId".into()
    }

    fn json_schema(_gen: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "string",
            "description": "Event identifier of the form `<kind>:<sqlite-compatible-rowid>` \
                            (e.g. `kernel:1234`, `cpu_sample:42`). Self-describing — \
                            the kind prefix tells `inspect` which schema to expect.",
            "pattern": "^[a-z_]+:[0-9]+$"
        })
    }
}

impl<'de> Deserialize<'de> for RowId {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let s = <&str as Deserialize>::deserialize(de)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() -> anyhow::Result<()> {
        let id = RowId::new(EventKind::Kernel, 1234);
        assert_eq!(id.to_string(), "kernel:1234");
        assert_eq!("kernel:1234".parse::<RowId>()?, id);
        Ok(())
    }

    #[test]
    fn json_round_trip() -> anyhow::Result<()> {
        let id = RowId::new(EventKind::Memcpy, 42);
        let s = serde_json::to_string(&id)?;
        assert_eq!(s, r#""memcpy:42""#);
        let back: RowId = serde_json::from_str(&s)?;
        assert_eq!(back, id);
        Ok(())
    }

    #[test]
    fn bad_inputs() {
        assert!(matches!(
            "no-colon".parse::<RowId>(),
            Err(RowIdParseError::NoColon(_))
        ));
        assert!(matches!(
            "bogus:1".parse::<RowId>(),
            Err(RowIdParseError::UnknownKind(_))
        ));
        assert!(matches!(
            "kernel:abc".parse::<RowId>(),
            Err(RowIdParseError::BadRowid(_))
        ));
    }
}
