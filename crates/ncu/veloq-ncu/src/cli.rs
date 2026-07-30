//! clap subcommand surface for the NCU profile source.
//!
//! Covers the NCU verbs — `summary`, `launches`, `inspect`, `metrics`,
//! `disasm`, `ranges`, `graphs`, `sources`, `source-metrics`,
//! `warp-stalls`, and `schema` — reached through the top-level `veloq`
//! binary's registry dispatch, so agents use one tool and one envelope
//! contract across NSys and NCU.

use clap::Subcommand;
use std::path::{Path, PathBuf};

#[derive(Subcommand)]
pub enum Cmd {
    /// Summarize an NCU kernel report — launch-derived totals
    /// (launch / range / graph / metric / rule / kernel-disasm counts)
    /// plus the NCU version. Per-launch metrics, rules, SASS / PTX /
    /// source correlation, and embedded source live in the dedicated
    /// verbs (`ncu launches`, `ncu inspect`, `ncu metrics`, `ncu
    /// disasm`, `ncu sources`, `ncu source-metrics`). JSON emits the
    /// full payload; `--format csv|table` renders a totals + session
    /// projection.
    Summary {
        /// Path to the `.ncu-rep` or `.ncu-repz` file.
        trace: PathBuf,
    },

    /// List CUDA kernel launches in the report. Each row is the
    /// headline columns an agent needs to pick an entry to drill
    /// into via `ncu inspect --row-id launch:<idx>`; metrics, rules,
    /// and NVTX state land there. Reads the
    /// `<file>.veloq/ncu-native.json.gz` sidecar.
    Launches {
        /// Path to the `.ncu-rep` or `.ncu-repz` file.
        trace: PathBuf,

        /// Glob (`*` / `?`) over the launch's demangled kernel
        /// signature. Matches against `kernel_demangled` first,
        /// falling back to `kernel_mangled` when the demangled name
        /// is empty.
        #[arg(long, value_name = "GLOB")]
        kernel: Option<String>,

        /// Glob over the NVTX range stack at the launch site. Joins
        /// each launch's `core.nvtx.start_end_ranges[].name` with
        /// `/` (innermost last) so an agent can filter against
        /// `*step/*decode*` etc.
        #[arg(long = "nvtx-range", value_name = "GLOB")]
        nvtx_range: Option<String>,

        /// Restrict to launches with a matching `grid_size`. Format:
        /// `WxHxD` (e.g. `1024x1x1`). Zero in any axis matches any
        /// value on that axis.
        #[arg(long, value_name = "WxHxD")]
        grid: Option<String>,

        /// Restrict to launches with a matching `block_size`. Same
        /// format as `--grid`.
        #[arg(long, value_name = "WxHxD")]
        block: Option<String>,

        /// Max launches to return.
        #[arg(long, default_value_t = 100)]
        limit: usize,
    },

    /// Fetch full per-launch details for one or more `launch:<idx>`
    /// row_ids: metrics, rules, NVTX state, and the recovered identity
    /// scalars. Mirrors `veloq inspect <kernel:N>` on the NSys side.
    Inspect {
        /// Path to the `.ncu-rep` or `.ncu-repz` file.
        trace: PathBuf,

        /// `launch:<idx>` row_id (one or more) as returned by
        /// `ncu launches`. Out-of-range idxs return a `NotFound`
        /// row so a partial batch still produces a usable response.
        #[arg(long = "row-id", value_name = "ROW_ID", required = true)]
        row_ids: Vec<String>,
    },

    /// Cross-launch metric values projected for jq-style comparison.
    /// Required `--counter <glob>` matches metric names; long-format
    /// (default) emits one row per `(launch, counter)` pair,
    /// `--per-launch` switches to the wide shape with every matched
    /// counter nested under each launch. The primary list is always
    /// `data.rows[]`; branch on `data.format` (`long` or
    /// `per_launch`) before reading row fields.
    Metrics {
        /// Path to the `.ncu-rep` or `.ncu-repz` file.
        trace: PathBuf,

        /// Glob over metric `name` field. Required — `--counter *`
        /// is allowed when an agent really wants everything.
        #[arg(long, value_name = "GLOB", required = true)]
        counter: String,

        /// Optional kernel-glob narrowing (matches demangled or
        /// mangled).
        #[arg(long, value_name = "GLOB")]
        kernel: Option<String>,

        /// Switch to the wide shape: one row per launch with a
        /// nested `counters: { name → value }` map.
        #[arg(long = "per-launch")]
        per_launch: bool,

        /// Max rows to return.
        #[arg(long, default_value_t = 1000)]
        limit: usize,
    },

    /// SASS / PTX / source-index correlation for the cubin a single
    /// launch ran out of. Resolves `--row-id launch:<idx>` to the
    /// launch's `source_ref`, then drives the existing per-cubin
    /// disasm pipeline. The acquisition cache lives under
    /// `<file>.veloq/disasm/<sha>.correlated.json`, so second+
    /// calls for the same cubin skip the nvdisasm / cuobjdump
    /// invocations.
    Disasm {
        /// Path to the `.ncu-rep` or `.ncu-repz` file.
        trace: PathBuf,

        /// `launch:<idx>` row_id as returned by `ncu launches`.
        #[arg(long = "row-id", value_name = "ROW_ID", required = true)]
        row_id: String,
    },

    /// List range workloads (captured under `ncu --replay-mode
    /// range` / `app-range`). Headline rows; full details via a
    /// future `ncu inspect range:<idx>`.
    Ranges {
        /// Path to the `.ncu-rep` or `.ncu-repz` file.
        trace: PathBuf,
        #[arg(long, default_value_t = 100)]
        limit: usize,
    },

    /// List CUDA-graph workloads (captured under `ncu
    /// --graph-profiling graph`).
    Graphs {
        /// Path to the `.ncu-rep` or `.ncu-repz` file.
        trace: PathBuf,
        #[arg(long, default_value_t = 100)]
        limit: usize,
    },

    /// List source / cubin metadata entries — one row per launch (the
    /// cubin it ran out of), with `cuda_sm_name`,
    /// `embedded_source_file_count`, and a `has_disasm` flag indicating
    /// whether the SASS/PTX pipeline output is already cached for that
    /// cubin. (`ncu_report` exposes no raw cubin/PTX byte sizes.)
    Sources {
        /// Path to the `.ncu-rep` or `.ncu-repz` file.
        trace: PathBuf,
        #[arg(long, default_value_t = 100)]
        limit: usize,
    },

    /// Per-source-line / per-SASS / per-file metric attribution for
    /// one CUDA kernel launch. Joins per-PC `MetricInstance` values
    /// with the DWARF source-line attribution from disasm so agents
    /// can answer "which source lines have the most bank conflicts?"
    SourceMetrics {
        /// Path to the `.ncu-rep` or `.ncu-repz` file.
        trace: PathBuf,

        /// `launch:<idx>` row_id as returned by `ncu launches`.
        /// Non-launch row_ids are rejected — they
        /// carry no `cubin_load_base`, so the SASS-PC join is
        /// ambiguous.
        #[arg(long = "row-id", value_name = "ROW_ID", required = true)]
        row_id: String,

        /// Counter selector glob (comma-separated). Required, matching
        /// the existing `ncu metrics --counter` contract; the canonical
        /// invocation lives in the `source-line-hotspots` recipe.
        #[arg(long, value_name = "GLOB", required = true)]
        counter: String,

        /// Rollup axis. `line` (default) aggregates by `(file, line)`;
        /// `sass` is per-PC identity; `file` aggregates each file's
        /// lines.
        #[arg(long, value_name = "AXIS", default_value = "line")]
        by: String,

        /// Restrict rows to source files matching this glob. Applied
        /// before sort + limit.
        #[arg(long, value_name = "GLOB")]
        file: Option<String>,

        /// Restrict to one source line. Requires `--file`.
        #[arg(long, value_name = "N")]
        line: Option<u32>,

        /// Sort spec; default is the first matched counter, descending.
        #[arg(long, value_name = "SPEC")]
        sort: Option<String>,

        /// Max rows returned.
        #[arg(long, default_value_t = 100)]
        limit: usize,
    },

    /// Per-source-line warp-stall-reason histogram for one kernel
    /// launch, from NCU's `timed_warp_samples` (the raw periodic
    /// warp-state stream). Answers "which source
    /// lines incur which stall reasons, and how much." Counts are
    /// raw sample counts; compute percentages in jq if wanted.
    WarpStalls {
        /// Path to the `.ncu-rep` or `.ncu-repz` file.
        trace: PathBuf,

        /// `launch:<idx>` row_id as returned by `ncu launches`.
        #[arg(long = "row-id", value_name = "ROW_ID", required = true)]
        row_id: String,

        /// Rollup axis. `line` (default) aggregates attributed PCs by
        /// `(file, line)`; `sass` is per in-cubin PC; `reason`
        /// collapses to one row per `StallReason` across the kernel.
        #[arg(long, value_name = "AXIS", default_value = "line")]
        by: String,

        /// Restrict rows to source files matching this glob (line /
        /// sass axes).
        #[arg(long, value_name = "GLOB")]
        file: Option<String>,

        /// Max rows returned.
        #[arg(long, default_value_t = 100)]
        limit: usize,
    },

    /// Emit the strict JSON Schema for one NCU response payload.
    /// Meta endpoint — does not read a trace.
    Schema {
        /// Subcommand whose response schema to print. The valid target
        /// list is injected at runtime from `schema_targets::TARGETS`.
        target: String,
    },
}

impl Cmd {
    /// Trace path this verb operates on. Used to populate the
    /// envelope's `trace.path` and as the input to the report
    /// reader.
    pub fn trace_path(&self) -> Option<&Path> {
        match self {
            Cmd::Summary { trace, .. } => Some(trace),
            Cmd::Launches { trace, .. } => Some(trace),
            Cmd::Inspect { trace, .. } => Some(trace),
            Cmd::Metrics { trace, .. } => Some(trace),
            Cmd::Disasm { trace, .. } => Some(trace),
            Cmd::Ranges { trace, .. } => Some(trace),
            Cmd::Graphs { trace, .. } => Some(trace),
            Cmd::Sources { trace, .. } => Some(trace),
            Cmd::SourceMetrics { trace, .. } => Some(trace),
            Cmd::WarpStalls { trace, .. } => Some(trace),
            Cmd::Schema { .. } => None,
        }
    }

    /// Stable verb label used in the qualified `envelope.command`
    /// (`ncu.summary`, …). Mirrors the kebab-case clap names.
    pub fn name(&self) -> &'static str {
        match self {
            Cmd::Summary { .. } => "summary",
            Cmd::Launches { .. } => "launches",
            Cmd::Inspect { .. } => "inspect",
            Cmd::Metrics { .. } => "metrics",
            Cmd::Disasm { .. } => "disasm",
            Cmd::Ranges { .. } => "ranges",
            Cmd::Graphs { .. } => "graphs",
            Cmd::Sources { .. } => "sources",
            Cmd::SourceMetrics { .. } => "source-metrics",
            Cmd::WarpStalls { .. } => "warp-stalls",
            Cmd::Schema { .. } => "schema",
        }
    }
}
