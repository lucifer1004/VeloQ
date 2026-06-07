//! `veloq hardware <trace>` — CPU / GPU / NIC inventory.
//!
//! Thin wrapper around [`veloq_nsys_data::hardware::extract`]: opens the
//! trace, pulls the topology, wraps in a response envelope so the
//! CLI can emit it through the standard JSON contract.
//!
//! Returning a struct (rather than re-exporting `Vec<HostInfo>`)
//! gives the response a stable top-level shape — agents always read
//! `data.rows[]` (the canonical primary table; each row is one
//! profiled host), and we can add response-level fields
//! (`total_hosts`, `elapsed_ms`, …) later without breaking consumers.

use serde::Serialize;
use std::path::Path;
use veloq_nsys_data::{HostInfo, Trace};

use crate::{NsysQueryError, NsysQueryResult};

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct HardwareResponse {
    /// Rows returned. Equal to `total_matched` today (no filtering
    /// on `hardware`); the field is here for contract uniformity.
    pub count: usize,
    /// Same as `count` today.
    pub total_matched: usize,
    /// Canonical primary table. Each row is one profiled host.
    /// Empty Vec (not an error) when `TARGET_INFO_SYSTEM_ENV` is
    /// absent — agents should treat "empty rows" as "no hardware
    /// info available" rather than retrying. Pair with
    /// `summary.capabilities.has_target_info` for the upstream signal.
    pub rows: Vec<HostInfo>,
}

pub fn run<P: AsRef<Path>>(path: P) -> NsysQueryResult<HardwareResponse> {
    // Hardware extraction queries small TARGET_INFO_* tables only.
    let trace = Trace::open(path).map_err(NsysQueryError::trace_open)?;
    let hosts =
        veloq_nsys_data::hardware::extract(&trace).map_err(NsysQueryError::hardware_extract)?;
    let count = hosts.len();
    Ok(HardwareResponse {
        count,
        total_matched: count,
        rows: hosts,
    })
}
