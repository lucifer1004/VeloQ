//! clap subcommand surface for the NSys profile source.
//!
//! Lives in `veloq-nsys` (not the binary) so the top-level `veloq`
//! parser can graft this subtree under any namespace it wants
//! (today: hoisted to the top level because NSys is the default
//! source; tomorrow: `veloq nsys <verb>` once the registry
//! dispatcher lands). Either way the `Cmd` enum and its arg groups
//! stay in one place.

use crate::filters::{
    CommonFilters, DeviceLocationFilters, GpuFilters, GpuLocationFilters, TraceArg,
};
use clap::{Subcommand, ValueEnum};
use std::path::Path;

/// Aggregation unit for `stats`. `Ns` is the public, duration-based
/// default; `Size` is the experimental byte-aggregator gated behind
/// `VELOQ_UNSTABLE=1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum StatsBy {
    /// Aggregate over event durations in nanoseconds. Default.
    #[default]
    Ns,
    /// Aggregate over event byte counts for memcpy/memset rows.
    /// Hidden flag value; requires `VELOQ_UNSTABLE=1` at the CLI
    /// dispatch layer.
    Size,
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Print a one-shot overview of the trace (version, duration, tables).
    Summary {
        #[command(flatten)]
        trace_arg: TraceArg,
    },

    /// Aggregate kernel/memcpy/memset/sync/runtime/osrt/graph/nvtx events by name.
    Stats {
        #[command(flatten)]
        trace_arg: TraceArg,

        /// Comma-separated grouping axes. Name axis (one of `short`,
        /// `demangled`, `mangled`, `no-name`) is mutually exclusive;
        /// physical axes (`device`, `stream`, `context`, `graph`,
        /// `graph_node`, `grid_block`) and NVTX hierarchy axes
        /// (`nvtx-parent`, `nvtx-path`) compose freely. Default
        /// `short` matches the original "one row per shortName" view.
        /// `mangled` falls back to `demangled` on older NSys schemas
        /// (no `mangledName` column); the fallback is reported via
        /// `mangled_axis_fallback` on the response. `grid_block` is
        /// kernel-only (gridX/Y/Z + blockX/Y/Z columns); explicit
        /// non-kernel kinds with this axis error up-front. Examples:
        /// `device`, `demangled,device`, `mangled,device`,
        /// `demangled,grid_block`, `nvtx-path`, `no-name,device,stream`.
        /// `stream` and `context` are device-local: use `--device`
        /// for one device, or include `device` in this list for
        /// cross-device comparison.
        #[arg(long, default_value = "short")]
        group_by: String,

        /// Add per-row event-duration histogram (half-decade buckets,
        /// 10 ns to 1 s). Response surfaces the shared bucket schema
        /// once at the top level as `histogram_buckets_ns`.
        #[arg(long)]
        hist: bool,

        /// For `--type runtime` only: fold API versions (e.g.
        /// `cudaMalloc_v3020` and `cudaMalloc_v2000` and `cudaMalloc`)
        /// into one bucket by stripping the `_v<digits>` suffix
        /// before grouping. Matches the nsys recipe `cuda_api_sum`
        /// behaviour. No-op for non-Runtime kinds. Opt-in so the
        /// unversioned view stays the default.
        #[arg(long = "collapse-versioned")]
        collapse_versioned: bool,

        /// Sort spec — one or more `key[:asc|:desc]` (or `-key` / `+key`)
        /// fields, comma-separated. Accepted keys: total, count, avg,
        /// min, max, p50, p95, p99, bytes, gbps, name, device, stream,
        /// context. Default `total` (DESC).
        #[arg(long, default_value = "total")]
        sort: String,

        /// (Hidden) Aggregation unit. Default `ns` (duration aggregates,
        /// the public path). `size` switches to byte-aggregating
        /// memcpy/memset rows under the experimental
        /// `StatsBySizeResponse` shape — requires `VELOQ_UNSTABLE=1`.
        #[arg(long, value_enum, default_value_t = StatsBy::Ns, hide = true)]
        by: StatsBy,

        #[command(flatten)]
        gpu: GpuFilters,

        #[command(flatten)]
        location: GpuLocationFilters,

        #[command(flatten)]
        common: CommonFilters,
    },

    /// Filter events. Returns a list of `row_id`s plus headline columns.
    Search {
        #[command(flatten)]
        trace_arg: TraceArg,

        /// Glob-style name pattern (`*` matches any sequence, `?` any
        /// single char). SQL `%`/`_` characters are escaped. Mutually
        /// exclusive with `--name-regex`.
        #[arg(long)]
        name: Option<String>,

        /// Regex name pattern (DuckDB `regexp_matches`, PCRE-ish).
        /// Mutually exclusive with `--name`.
        #[arg(long = "name-regex", value_name = "REGEX")]
        name_regex: Option<String>,

        /// Duration filter: `>1ms`, `<=100us`, `>=42ns`, or range `100us-1ms`.
        #[arg(long)]
        duration: Option<String>,

        /// Sort spec — one or more `key[:asc|:desc]` fields. Accepted
        /// keys: start (default, ASC), duration (DESC), name (ASC).
        #[arg(long, default_value = "start")]
        sort: String,

        /// Decorate every kernel/memcpy/memset/sync hit with the
        /// innermost NVTX range that was open on its launching thread
        /// (`nvtx_context: { range_id, name, depth, iter_index }`).
        /// Off by default — adds one extra SQL per CUPTI kind in the
        /// result; opt in when the diff target is "which step did
        /// this slow kernel belong to". Different lever from `--nvtx`,
        /// which *filters* the result set; this one decorates rows
        /// that are already in it.
        #[arg(long = "with-nvtx")]
        with_nvtx: bool,

        #[command(flatten)]
        gpu: GpuFilters,

        #[command(flatten)]
        location: GpuLocationFilters,

        #[command(flatten)]
        common: CommonFilters,
    },

    /// Fetch full details for one or more events by `row_id`
    /// (e.g. `kernel:1234`).
    Inspect {
        #[command(flatten)]
        trace_arg: TraceArg,

        /// One or more row ids in `<kind>:<rowid>` form.
        #[arg(required = true)]
        row_ids: Vec<String>,
    },

    /// Walk the CPU↔GPU causal chain for one or more events by `row_id`.
    Correlate {
        #[command(flatten)]
        trace_arg: TraceArg,

        /// One or more row ids in `<kind>:<rowid>` form.
        #[arg(required = true)]
        row_ids: Vec<String>,
    },

    /// List CUDA graph replays and, in node-capture mode, the graph
    /// nodes/kernels dominating each replay.
    GraphReplays {
        #[command(flatten)]
        trace_arg: TraceArg,

        #[command(flatten)]
        location: DeviceLocationFilters,

        /// Restrict to graph launches whose runtime call is inside an
        /// enclosing NVTX range matching this glob.
        #[arg(long)]
        nvtx: Option<String>,

        /// Sort spec — accepted keys: wall (default, DESC), sum (DESC),
        /// start (ASC), count (DESC).
        #[arg(long, default_value = "wall:desc")]
        sort: String,

        /// Max nodes to include inside each replay row's JSON
        /// `top_nodes` list. Table/CSV flatten only the top node.
        #[arg(long = "top-nodes", default_value_t = 10)]
        top_nodes: usize,

        #[command(flatten)]
        common: CommonFilters,
    },

    /// Generate an Nsight Compute rerun command for one CUDA kernel
    /// event selected from the NSys timeline.
    NcuCommand {
        #[command(flatten)]
        trace_arg: TraceArg,

        /// Kernel row id in `<kind>:<rowid>` form, e.g. `kernel:1234`.
        #[arg(value_name = "ROW_ID")]
        row_id: String,

        /// Emit only a pipe-ready bash script on stdout. Errors stay
        /// on stderr so `veloq nsys ncu-command ... --print | bash`
        /// never feeds a JSON error envelope to the shell.
        #[arg(long)]
        print: bool,

        /// Environment to include in the script: `none`, `safe`, or
        /// `all`. Sensitive-looking variable names are always omitted.
        #[arg(long, default_value = "safe", value_name = "POLICY")]
        env: String,
    },

    /// Build (or load) the correlation index and report per-kind row stats.
    CorrelationStats {
        #[command(flatten)]
        trace_arg: TraceArg,
    },

    /// Prepare a trace for fast queries: export to a parquetdir and warm the metadata cache (`--status` inspects cache state without building).
    Prep {
        #[command(flatten)]
        trace_arg: TraceArg,

        /// Report cache state without building anything. Exits with
        /// `0` regardless of cache state; check `parquet_cache.present`
        /// / `parquet_cache.tables` and `meta_cache.fingerprint_match`
        /// / `meta_cache.format_version_on_disk` in the response.
        #[arg(long)]
        status: bool,
    },

    /// GPU overlap: per-device union vs sum busy time, peak concurrency, per-stream + compute/copy overlap.
    Concurrency {
        #[command(flatten)]
        trace_arg: TraceArg,

        #[command(flatten)]
        location: DeviceLocationFilters,

        #[command(flatten)]
        common: CommonFilters,
    },

    /// Find GPU idle bubbles per device, per stream, or across the trace (`--scope`).
    Gaps {
        #[command(flatten)]
        trace_arg: TraceArg,

        /// Aggregation scope. `device` (default) | `stream` | `trace`.
        /// Trace scope implies all devices when no device is selected.
        #[arg(long, default_value = "device", value_name = "SCOPE")]
        scope: String,

        /// Minimum gap duration to report. Accepts the same forms as
        /// `--from`/`--to` endpoints (`1ms`, `100us`, `1.2s`, `42ns`).
        #[arg(long, default_value = "1ms", value_name = "TIME")]
        min_duration: String,

        #[command(flatten)]
        location: GpuLocationFilters,

        /// Sort spec — one or more `key[:asc|:desc]` fields. Accepted
        /// keys: duration (default, DESC), start (ASC), device (ASC),
        /// stream (ASC). `stream` requires `--scope stream`; `device`
        /// is meaningless under `--scope trace`.
        #[arg(long, default_value = "duration")]
        sort: String,

        #[command(flatten)]
        common: CommonFilters,
    },

    /// Time-bucketed GPU activity: per-bucket busy ns, per-kind breakdown, and event counts.
    Timeline {
        #[command(flatten)]
        trace_arg: TraceArg,

        /// Bucket width. Same forms as `--from`/`--to` endpoints
        /// (`1ms`, `100us`, `1.2s`, `42ns`). Required.
        #[arg(long, value_name = "TIME")]
        interval: String,

        #[command(flatten)]
        gpu: GpuFilters,

        #[command(flatten)]
        location: GpuLocationFilters,

        #[command(flatten)]
        common: CommonFilters,
    },

    /// Per-NVTX-range CPU bounds plus attributed GPU work; `--aggregate` for per-scope distributions.
    Slices {
        #[command(flatten)]
        trace_arg: TraceArg,

        /// Glob-style NVTX range name (`*` matches any sequence, `?`
        /// any single char). Omit to match every NVTX range.
        /// Mutually exclusive with `--name-regex`.
        #[arg(long)]
        name: Option<String>,

        /// Regex NVTX range name (DuckDB `regexp_matches`, PCRE-ish).
        /// Mutually exclusive with `--name`.
        #[arg(long = "name-regex", value_name = "REGEX")]
        name_regex: Option<String>,

        /// Sort spec — one or more `key[:asc|:desc]` fields. Accepted
        /// instance-view keys: start (default, ASC), cpu_duration
        /// (DESC), attributed_kernel (DESC), attributed_total (DESC),
        /// name (ASC). In --aggregate mode: total (default, DESC),
        /// instances (DESC), p50 / p99 (DESC), name/path (ASC).
        #[arg(long, default_value = "")]
        sort: String,

        /// Aggregate matching NVTX range instances into per-scope rows.
        /// Default output remains one row per individual range.
        #[arg(long)]
        aggregate: bool,

        /// Aggregate axis: `name` groups by leaf NVTX range
        /// name; `path` groups by the full NVTX hierarchy path. Only
        /// meaningful with `--aggregate`.
        #[arg(long = "group-by", default_value = "name")]
        group_by: String,

        #[command(flatten)]
        location: GpuLocationFilters,

        #[command(flatten)]
        common: CommonFilters,
    },

    /// Profiled hardware inventory (CPU, CUDA GPUs, NICs) from the trace's `TARGET_INFO_*` tables.
    Hardware {
        #[command(flatten)]
        trace_arg: TraceArg,
    },

    /// Hardware perf-counter / CPU-sample / CPU-sched queries (`--type`); summary or `--bucket` time series.
    Metrics {
        #[command(flatten)]
        trace_arg: TraceArg,

        /// Metric source: `gpu` (default), `nic`, `cpu-sampling`,
        /// `cpu-sched`.
        #[arg(long = "type", default_value = "gpu")]
        source: String,

        // ---- gpu + nic ----
        /// `--type gpu`: glob (`*`/`?`) over the counter's
        /// `metricName` (e.g. `"SMs Active*"`). `--type nic`: glob
        /// over the network counter name (e.g. `"IB: Bytes sent"`).
        /// Omit to keep every counter on the trace.
        #[arg(long)]
        counter: Option<String>,

        // ---- cpu-sampling + cpu-sched ----
        /// Aggregation axis. `--type cpu-sampling`: `symbol` (default,
        /// leaf-frame function — `perf top` ergonomics), `tid`, `cpu`,
        /// `module`, `stack`. `--type cpu-sched`: `tid` (default), `cpu`, `state`.
        #[arg(long = "group-by", value_name = "AXIS")]
        group_by: Option<String>,

        /// `--type cpu-sampling`: glob (`*`/`?`) over the leaf
        /// frame's symbol name (with `--group-by symbol`), module
        /// basename (with `--group-by module`), or any frame in a stack
        /// (with `--group-by stack`). cpu-sched has no name
        /// field — this flag is rejected there.
        #[arg(long)]
        name: Option<String>,

        /// Restrict to one CPU id. Applies to cpu-sampling
        /// (`COMPOSITE_EVENTS.cpu`) and cpu-sched (`SCHED_EVENTS.cpu`).
        #[arg(long, value_name = "CPU_ID")]
        cpu: Option<i64>,

        /// Restrict to one thread (`globalTid` — see `summary`
        /// per-table for the actual ids). Applies to cpu-sampling and
        /// cpu-sched.
        #[arg(long, value_name = "TID")]
        tid: Option<i64>,

        // ---- shared ----
        /// Bucket width for time-series mode (`50ms`, `100us`,
        /// `1.2s`, …). Without this the response is the summary
        /// view; with it, samples roll up into fixed-width buckets.
        /// `--type gpu` aggregator: `mean` (or `sum` for
        /// `[Cycles Active]` / `[Requests]`); nic: `mean`
        /// (NSys exports rates/averages); cpu-sampling: `sum`
        /// (sample counts); cpu-sched: `sum` (clipped on-cpu ns per
        /// bucket).
        #[arg(long, value_name = "TIME")]
        bucket: Option<String>,

        /// Sort spec — applies to the summary view only. See
        /// `metrics --help` Sort keys block for the full per-type key
        /// list (one source-of-truth, regenerated from the parser).
        #[arg(long)]
        sort: Option<String>,

        #[command(flatten)]
        common: CommonFilters,
    },

    /// Emit the strict JSON Schema for one subcommand's `.data` response body. Meta endpoint — reads no trace.
    Schema {
        /// Subcommand whose response schema to print. The valid
        /// target list is injected at runtime by
        /// `help::inject_long_about` from the shared
        /// `schema_targets::TARGETS` registry, so this doc-comment
        /// stays target-list-free to keep the SSOT honest.
        target: String,
    },
}

impl Cmd {
    /// Stable subcommand label used in the JSON envelope's `command`
    /// field. `Cmd::Stats` inspects `by` so the size-mode envelope
    /// reports `stats-by-size` (matching the schema target) and
    /// success/error envelopes agree for one invocation.
    pub fn name(&self) -> &'static str {
        match self {
            Cmd::Summary { .. } => "summary",
            Cmd::Stats { by, .. } => match by {
                StatsBy::Ns => "stats",
                StatsBy::Size => "stats-by-size",
            },
            Cmd::Search { .. } => "search",
            Cmd::Inspect { .. } => "inspect",
            Cmd::Correlate { .. } => "correlate",
            Cmd::GraphReplays { .. } => "graph-replays",
            Cmd::NcuCommand { .. } => "ncu-command",
            Cmd::CorrelationStats { .. } => "correlation-stats",
            Cmd::Prep { .. } => "prep",
            Cmd::Concurrency { .. } => "concurrency",
            Cmd::Gaps { .. } => "gaps",
            Cmd::Timeline { .. } => "timeline",
            Cmd::Slices { .. } => "slices",
            Cmd::Hardware { .. } => "hardware",
            Cmd::Metrics { .. } => "metrics",
            Cmd::Schema { .. } => "schema",
        }
    }

    /// Path to the trace this invocation operates on. Available
    /// even before the command runs, so error envelopes can carry
    /// the trace path on parse-level failures (e.g. invalid
    /// `--limit`). `Cmd::Schema` doesn't read a trace and returns
    /// `None` — callers should omit `envelope.trace` entirely in
    /// that case rather than fabricate an empty path.
    pub fn trace_path(&self) -> Option<&Path> {
        match self {
            Cmd::Summary { trace_arg }
            | Cmd::CorrelationStats { trace_arg }
            | Cmd::Prep { trace_arg, .. }
            | Cmd::Hardware { trace_arg } => Some(&trace_arg.trace),
            Cmd::Stats { trace_arg, .. }
            | Cmd::Search { trace_arg, .. }
            | Cmd::Inspect { trace_arg, .. }
            | Cmd::Correlate { trace_arg, .. }
            | Cmd::GraphReplays { trace_arg, .. }
            | Cmd::NcuCommand { trace_arg, .. }
            | Cmd::Concurrency { trace_arg, .. }
            | Cmd::Gaps { trace_arg, .. }
            | Cmd::Timeline { trace_arg, .. }
            | Cmd::Slices { trace_arg, .. }
            | Cmd::Metrics { trace_arg, .. } => Some(&trace_arg.trace),
            Cmd::Schema { .. } => None,
        }
    }

    /// Whether this invocation intentionally writes non-JSON success
    /// output to stdout. Used by the source error path to keep stdout
    /// pipe-safe for shell execution.
    pub fn raw_stdout(&self) -> bool {
        matches!(self, Cmd::NcuCommand { print: true, .. })
    }
}
