use clap::{Arg, ArgMatches, Command};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use sysinfo::System;

use super::{DaemonError, DaemonResult};

pub const DEFAULT_LIFECYCLE_TIMEOUT_MS: u64 = 30_000;
pub const DEFAULT_CONNECT_TIMEOUT_MS: u64 = 1_000;

pub(crate) const MIB_BYTES: u64 = 1_048_576;
const MAX_U32: u64 = u32::MAX as u64;
const MAX_MIB: u64 = 17_592_186_044_415;
const MAX_LONG_TIMEOUT_MS: u64 = 31_536_000_000;
pub const MAX_LIFECYCLE_TIMEOUT_MS: u64 = 600_000;

const DEFAULT_MAX_SESSIONS: u64 = 8;
const DEFAULT_MAX_CONCURRENT_REQUESTS: u64 = 1;
const DEFAULT_MAX_QUEUED_REQUESTS: u64 = 32;
const DEFAULT_ADMISSION_TIMEOUT_MS: u64 = 30_000;
const DEFAULT_IDLE_TIMEOUT_MS: u64 = 900_000;
const DEFAULT_SHUTDOWN_GRACE_MS: u64 = 30_000;
const RESIDENT_MEMORY_CAPACITY_DIVISOR: u64 = 16;

fn default_resident_memory_mib() -> u64 {
    static LIMIT: OnceLock<u64> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        let mut system = System::new();
        system.refresh_memory();
        let capacity_bytes = system
            .cgroup_limits()
            .map_or_else(|| system.total_memory(), |limits| limits.total_memory);
        let capacity_mib = (capacity_bytes / MIB_BYTES).clamp(1, MAX_MIB);
        (capacity_mib / RESIDENT_MEMORY_CAPACITY_DIVISOR).max(1)
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DaemonLimits {
    pub max_sessions: u64,
    pub max_resident_memory_bytes: u64,
    pub max_concurrent_requests: u64,
    pub max_query_workers: u64,
    pub max_query_memory_bytes: Option<u64>,
    pub max_queued_requests: u64,
    pub admission_timeout_ms: u64,
    pub idle_timeout_ms: u64,
    pub shutdown_grace_ms: u64,
}

impl Default for DaemonLimits {
    fn default() -> Self {
        Self {
            max_sessions: DEFAULT_MAX_SESSIONS,
            max_resident_memory_bytes: default_resident_memory_mib() * MIB_BYTES,
            max_concurrent_requests: DEFAULT_MAX_CONCURRENT_REQUESTS,
            max_query_workers: veloq_core::default_query_worker_count() as u64,
            max_query_memory_bytes: None,
            max_queued_requests: DEFAULT_MAX_QUEUED_REQUESTS,
            admission_timeout_ms: DEFAULT_ADMISSION_TIMEOUT_MS,
            idle_timeout_ms: DEFAULT_IDLE_TIMEOUT_MS,
            shutdown_grace_ms: DEFAULT_SHUTDOWN_GRACE_MS,
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct DaemonConfigRequest {
    pub max_sessions: Option<u64>,
    pub max_resident_memory_mib: Option<u64>,
    pub max_concurrent_requests: Option<u64>,
    pub max_query_workers: Option<u64>,
    pub max_query_memory_mib: Option<u64>,
    pub max_queued_requests: Option<u64>,
    pub admission_timeout_ms: Option<u64>,
    pub idle_timeout_ms: Option<u64>,
    pub shutdown_grace_ms: Option<u64>,
}

impl DaemonConfigRequest {
    pub fn from_matches(matches: &ArgMatches) -> DaemonResult<Self> {
        let request = Self {
            max_sessions: parse_optional(matches, "max-sessions", 1, MAX_U32)?,
            max_resident_memory_mib: parse_optional(
                matches,
                "max-resident-memory-mib",
                1,
                MAX_MIB,
            )?,
            max_concurrent_requests: parse_optional(
                matches,
                "max-concurrent-requests",
                1,
                MAX_U32,
            )?,
            max_query_workers: parse_optional(matches, "max-query-workers", 1, MAX_U32)?,
            max_query_memory_mib: parse_optional(matches, "max-query-memory-mib", 1, MAX_MIB)?,
            max_queued_requests: parse_optional(matches, "max-queued-requests", 0, MAX_U32)?,
            admission_timeout_ms: parse_optional(
                matches,
                "admission-timeout-ms",
                0,
                MAX_LONG_TIMEOUT_MS,
            )?,
            idle_timeout_ms: parse_optional(matches, "idle-timeout-ms", 0, MAX_LONG_TIMEOUT_MS)?,
            shutdown_grace_ms: parse_optional(
                matches,
                "shutdown-grace-ms",
                0,
                MAX_LONG_TIMEOUT_MS,
            )?,
        };
        if request
            .max_concurrent_requests
            .unwrap_or(DEFAULT_MAX_CONCURRENT_REQUESTS)
            > 1
            && request.max_query_memory_mib.is_none()
        {
            return Err(DaemonError::invalid_config(
                "--max-query-memory-mib is required when --max-concurrent-requests is greater than 1",
            ));
        }
        Ok(request)
    }

    pub fn effective(&self) -> DaemonLimits {
        let defaults = DaemonLimits::default();
        DaemonLimits {
            max_sessions: self.max_sessions.unwrap_or(defaults.max_sessions),
            max_resident_memory_bytes: self
                .max_resident_memory_mib
                .map(|value| value * MIB_BYTES)
                .unwrap_or(defaults.max_resident_memory_bytes),
            max_concurrent_requests: self
                .max_concurrent_requests
                .unwrap_or(defaults.max_concurrent_requests),
            max_query_workers: self.max_query_workers.unwrap_or(defaults.max_query_workers),
            max_query_memory_bytes: self.max_query_memory_mib.map(|value| value * MIB_BYTES),
            max_queued_requests: self
                .max_queued_requests
                .unwrap_or(defaults.max_queued_requests),
            admission_timeout_ms: self
                .admission_timeout_ms
                .unwrap_or(defaults.admission_timeout_ms),
            idle_timeout_ms: self.idle_timeout_ms.unwrap_or(defaults.idle_timeout_ms),
            shutdown_grace_ms: self.shutdown_grace_ms.unwrap_or(defaults.shutdown_grace_ms),
        }
    }

    pub fn conflicts_with(&self, effective: &DaemonLimits) -> bool {
        self.max_sessions
            .is_some_and(|value| value != effective.max_sessions)
            || self.max_resident_memory_mib.is_some_and(|value| {
                value.saturating_mul(MIB_BYTES) != effective.max_resident_memory_bytes
            })
            || self
                .max_concurrent_requests
                .is_some_and(|value| value != effective.max_concurrent_requests)
            || self
                .max_query_workers
                .is_some_and(|value| value != effective.max_query_workers)
            || self.max_query_memory_mib.is_some_and(|value| {
                Some(value.saturating_mul(MIB_BYTES)) != effective.max_query_memory_bytes
            })
            || self
                .max_queued_requests
                .is_some_and(|value| value != effective.max_queued_requests)
            || self
                .admission_timeout_ms
                .is_some_and(|value| value != effective.admission_timeout_ms)
            || self
                .idle_timeout_ms
                .is_some_and(|value| value != effective.idle_timeout_ms)
            || self
                .shutdown_grace_ms
                .is_some_and(|value| value != effective.shutdown_grace_ms)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingMode {
    Auto,
    Off,
    Required,
}

impl RoutingMode {
    pub fn parse(value: &str) -> DaemonResult<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "off" => Ok(Self::Off),
            "required" => Ok(Self::Required),
            _ => Err(DaemonError::invalid_config(format!(
                "--daemon must be auto, off, or required; got `{value}`"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryRouting {
    pub mode: RoutingMode,
    pub connect_timeout_ms: u64,
}

pub fn query_routing(matches: &ArgMatches) -> DaemonResult<Option<QueryRouting>> {
    if let Ok(Some(value)) = matches.try_get_one::<String>("daemon") {
        let timeout = matches
            .try_get_one::<String>("daemon-connect-timeout-ms")
            .ok()
            .flatten()
            .ok_or_else(|| DaemonError::invalid_config("missing daemon connection timeout"))?;
        return Ok(Some(QueryRouting {
            mode: RoutingMode::parse(value)?,
            connect_timeout_ms: parse_decimal(
                "--daemon-connect-timeout-ms",
                timeout,
                1,
                MAX_LIFECYCLE_TIMEOUT_MS,
            )?,
        }));
    }
    if let Some((_, child)) = matches.subcommand() {
        return query_routing(child);
    }
    Ok(None)
}

pub fn lifecycle_timeout(matches: &ArgMatches) -> DaemonResult<u64> {
    let value = matches
        .get_one::<String>("timeout-ms")
        .ok_or_else(|| DaemonError::invalid_config("missing lifecycle timeout"))?;
    parse_decimal("--timeout-ms", value, 1, MAX_LIFECYCLE_TIMEOUT_MS)
}

pub fn inject_query_args(command: &mut Command) {
    if command.get_subcommands().next().is_some() {
        for child in command.get_subcommands_mut() {
            inject_query_args(child);
        }
        return;
    }
    if command.get_name() == "schema" {
        return;
    }
    let updated = command
        .clone()
        .arg(
            Arg::new("daemon")
                .long("daemon")
                .default_value("auto")
                .value_name("MODE")
                .help("Query routing: auto, off, or required"),
        )
        .arg(
            Arg::new("daemon-connect-timeout-ms")
                .long("daemon-connect-timeout-ms")
                .default_value(DEFAULT_CONNECT_TIMEOUT_MS.to_string())
                .value_name("N")
                .allow_hyphen_values(true)
                .help("Local daemon connection and capability deadline in milliseconds"),
        );
    *command = updated;
}

pub fn start_args(command: Command) -> Command {
    command
        .arg(resource_arg(
            "max-sessions",
            DEFAULT_MAX_SESSIONS,
            "Maximum resident profile sessions",
        ))
        .arg(resource_arg(
            "max-resident-memory-mib",
            default_resident_memory_mib(),
            "Daemon-accounted resident memory ceiling in MiB",
        ))
        .arg(resource_arg(
            "max-concurrent-requests",
            DEFAULT_MAX_CONCURRENT_REQUESTS,
            "Maximum active query requests",
        ))
        .arg(resource_arg(
            "max-query-workers",
            veloq_core::default_query_worker_count() as u64,
            "Maximum aggregate query-worker permits",
        ))
        .arg(
            Arg::new("max-query-memory-mib")
                .long("max-query-memory-mib")
                .value_name("N")
                .allow_hyphen_values(true)
                .help(
                    "Maximum aggregate active-query memory budgets in MiB; omitted uses the source query engine default and requires one active request",
                ),
        )
        .arg(resource_arg(
            "max-queued-requests",
            DEFAULT_MAX_QUEUED_REQUESTS,
            "Maximum accepted requests waiting for admission",
        ))
        .arg(resource_arg(
            "admission-timeout-ms",
            DEFAULT_ADMISSION_TIMEOUT_MS,
            "Maximum admission queue wait in milliseconds",
        ))
        .arg(resource_arg(
            "idle-timeout-ms",
            DEFAULT_IDLE_TIMEOUT_MS,
            "Idle session discard eligibility in milliseconds",
        ))
        .arg(resource_arg(
            "shutdown-grace-ms",
            DEFAULT_SHUTDOWN_GRACE_MS,
            "Grace for active requests during daemon stop",
        ))
}

pub fn timeout_arg() -> Arg {
    Arg::new("timeout-ms")
        .long("timeout-ms")
        .default_value(DEFAULT_LIFECYCLE_TIMEOUT_MS.to_string())
        .value_name("N")
        .allow_hyphen_values(true)
        .help("Lifecycle deadline in milliseconds (1..=600000)")
}

fn resource_arg(id: &'static str, default: u64, help: &'static str) -> Arg {
    Arg::new(id)
        .long(id)
        .value_name("N")
        .allow_hyphen_values(true)
        .help(format!("{help}; default {default}"))
}

fn parse_optional(
    matches: &ArgMatches,
    id: &'static str,
    min: u64,
    max: u64,
) -> DaemonResult<Option<u64>> {
    matches
        .get_one::<String>(id)
        .map(|value| parse_decimal(&format!("--{id}"), value, min, max))
        .transpose()
}

fn parse_decimal(flag: &str, value: &str, min: u64, max: u64) -> DaemonResult<u64> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(DaemonError::invalid_config(format!(
            "{flag} must be a decimal integer from {min} through {max}; got `{value}`"
        )));
    }
    let parsed = value.parse::<u64>().map_err(|_| {
        DaemonError::invalid_config(format!(
            "{flag} must be a decimal integer from {min} through {max}; got `{value}`"
        ))
    })?;
    if !(min..=max).contains(&parsed) {
        return Err(DaemonError::invalid_config(format!(
            "{flag} must be from {min} through {max}; got `{value}`"
        )));
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimal_parser_rejects_signs_and_bounds() {
        assert!(parse_decimal("--x", "+1", 1, 10).is_err());
        assert!(parse_decimal("--x", "-1", 1, 10).is_err());
        assert!(parse_decimal("--x", "0", 1, 10).is_err());
        assert!(parse_decimal("--x", "11", 1, 10).is_err());
        assert_eq!(parse_decimal("--x", "10", 1, 10).ok(), Some(10));
    }
}
