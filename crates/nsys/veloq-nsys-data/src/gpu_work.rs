//! Shared NSys GPU work interval semantics.
//!
//! These are duration-bearing device work tables with the same minimal
//! interval shape: `start`, `"end"`, `deviceId`, and `streamId`.
//! Query-side verbs and derived sidecars use this list as the source of
//! truth for "GPU busy" interval inputs.

/// Common interval columns used by every GPU work table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuWorkIntervalColumns {
    pub start_ns: &'static str,
    pub end_ns: &'static str,
    pub device_id: &'static str,
    pub stream_id: &'static str,
}

pub const GPU_WORK_INTERVAL_COLUMNS: GpuWorkIntervalColumns = GpuWorkIntervalColumns {
    start_ns: "start",
    end_ns: "end",
    device_id: "deviceId",
    stream_id: "streamId",
};

/// One NSys table that represents duration-bearing GPU work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuWorkKind {
    /// Stable VeloQ event kind label, matching query-side `EventKind`.
    pub label: &'static str,
    /// NSys backing table.
    pub table: &'static str,
}

pub const GPU_WORK_INTERVAL_KINDS: &[GpuWorkKind] = &[
    GpuWorkKind {
        label: "kernel",
        table: "CUPTI_ACTIVITY_KIND_KERNEL",
    },
    GpuWorkKind {
        label: "memcpy",
        table: "CUPTI_ACTIVITY_KIND_MEMCPY",
    },
    GpuWorkKind {
        label: "memset",
        table: "CUPTI_ACTIVITY_KIND_MEMSET",
    },
    GpuWorkKind {
        label: "graph",
        table: "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
    },
];
