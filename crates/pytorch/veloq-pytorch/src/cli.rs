use clap::{Args, Subcommand};
use std::path::{Path, PathBuf};

#[derive(Subcommand)]
pub enum Cmd {
    /// Summarize one PyTorch trace file.
    Summary {
        /// Path to a Chrome trace `.json` or `.json.gz` file.
        trace: PathBuf,
    },

    /// Filter PyTorch/Kineto events into typed event refs.
    Search {
        /// Path to a Chrome trace `.json` or `.json.gz` file.
        trace: PathBuf,
        #[command(flatten)]
        filters: EventArgs,
    },

    /// Fetch full details for one or more row ids.
    Inspect {
        /// Path to a Chrome trace `.json` or `.json.gz` file.
        trace: PathBuf,
        #[arg(required = true)]
        row_ids: Vec<String>,
    },

    /// Aggregate event durations and counts by one or more axes.
    Stats {
        /// Path to a Chrome trace `.json` or `.json.gz` file.
        trace: PathBuf,
        /// Comma-separated axes: name,type,step,rank,device,stream,shape,comm-kind,python-context,python-path.
        /// In multi-rank traces, `device` must be paired with `rank`;
        /// `stream` must be paired with both `rank` and `device`.
        #[arg(long, default_value = "name")]
        group_by: String,
        #[command(flatten)]
        filters: EventArgs,
    },

    /// Walk CPU op / runtime / driver / GPU / flow causal chains.
    Correlate {
        /// Path to a Chrome trace `.json` or `.json.gz` file.
        trace: PathBuf,
        #[arg(required = true)]
        row_ids: Vec<String>,
    },

    /// Bucket CPU, GPU, and communication time.
    Timeline {
        /// Path to a Chrome trace `.json` or `.json.gz` file.
        trace: PathBuf,
        /// Bucket width, e.g. `1ms`, `100us`, `1s`.
        #[arg(long, value_name = "TIME", required = true)]
        interval: String,
        #[command(flatten)]
        filters: EventArgs,
    },

    /// Query ProfilerStep and user annotation ranges.
    Slices {
        /// Path to a Chrome trace `.json` or `.json.gz` file.
        trace: PathBuf,
        /// Glob-style annotation/step name pattern.
        #[arg(long)]
        name: Option<String>,
        /// Regex annotation/step name pattern.
        #[arg(long = "name-regex", value_name = "REGEX")]
        name_regex: Option<String>,
        /// Aggregate range instances by name or step.
        #[arg(long)]
        aggregate: bool,
        /// Aggregate grouping axis in `--aggregate` mode: name or step.
        #[arg(long = "group-by", default_value = "name")]
        group_by: String,
        #[command(flatten)]
        scope: ScopeArgs,
        #[command(flatten)]
        common: CommonArgs,
    },

    /// Analyze communication collectives within one PyTorch trace file.
    Collectives {
        /// Path to a Chrome trace `.json` or `.json.gz` file.
        trace: PathBuf,
        /// Restrict groups to one profiler step.
        #[arg(long)]
        step: Option<i64>,
        /// Restrict to one PyTorch distributed rank.
        #[arg(long, conflicts_with = "all_ranks")]
        rank: Option<i64>,
        /// Opt into reporting all ranks when a trace contains multiple ranks.
        #[arg(long = "all-ranks", default_value_t = false)]
        all_ranks: bool,
        /// Max collectives to return.
        #[arg(long, default_value_t = 100)]
        limit: usize,
    },

    /// Build or inspect PyTorch sidecars.
    Prep {
        /// Path to a Chrome trace `.json` or `.json.gz` file.
        trace: PathBuf,
        /// Report cache state without building sidecars.
        #[arg(long)]
        status: bool,
    },

    /// Emit the JSON Schema for one PyTorch response payload.
    Schema {
        /// Subcommand whose response schema to print. The valid target
        /// list is injected at runtime from `schema_targets::TARGETS`.
        target: String,
    },
}

#[derive(Args, Debug, Clone, Default)]
pub struct CommonArgs {
    /// Start of the time window. Pair with `--to`.
    #[arg(long, value_name = "TIME")]
    pub from: Option<String>,

    /// End of the time window. Pair with `--from`.
    #[arg(long, value_name = "TIME")]
    pub to: Option<String>,

    /// Max rows to return.
    #[arg(long)]
    pub limit: Option<usize>,
}

#[derive(Args, Debug, Clone, Copy, Default)]
pub struct ScopeArgs {
    /// Restrict to one PyTorch distributed rank.
    #[arg(long, conflicts_with = "all_ranks")]
    pub rank: Option<i64>,

    /// Opt into cross-rank aggregation/listing.
    #[arg(long = "all-ranks", default_value_t = false)]
    pub all_ranks: bool,

    /// Restrict to one CUDA device id. On multi-rank traces this
    /// requires `--rank` because device ids are rank-local.
    #[arg(long)]
    pub device: Option<i64>,

    /// Restrict to one CUDA stream id. Requires a single device, and
    /// on multi-rank traces a single rank.
    #[arg(long)]
    pub stream: Option<i64>,

    /// Restrict to one ProfilerStep number.
    #[arg(long)]
    pub step: Option<i64>,
}

#[derive(Args, Debug, Clone, Default)]
pub struct EventArgs {
    /// Event types, comma-separated, or `all`.
    #[arg(long = "type", default_value = "all")]
    pub types: String,

    /// Glob-style name pattern.
    #[arg(long)]
    pub name: Option<String>,

    /// Regex name pattern.
    #[arg(long = "name-regex", value_name = "REGEX")]
    pub name_regex: Option<String>,

    /// Duration filter: `>1ms`, `<=100us`, or `100us-1ms`.
    #[arg(long)]
    pub duration: Option<String>,

    /// Keep only communication-related events. Combine with `--type kernel` for NCCL kernels.
    #[arg(long = "is-comm")]
    pub is_comm: bool,

    #[command(flatten)]
    pub scope: ScopeArgs,

    #[command(flatten)]
    pub common: CommonArgs,
}

impl Cmd {
    pub fn trace_path(&self) -> Option<&Path> {
        match self {
            Cmd::Summary { trace }
            | Cmd::Search { trace, .. }
            | Cmd::Inspect { trace, .. }
            | Cmd::Stats { trace, .. }
            | Cmd::Correlate { trace, .. }
            | Cmd::Timeline { trace, .. }
            | Cmd::Slices { trace, .. }
            | Cmd::Collectives { trace, .. }
            | Cmd::Prep { trace, .. } => Some(trace),
            Cmd::Schema { .. } => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Cmd::Summary { .. } => "summary",
            Cmd::Search { .. } => "search",
            Cmd::Inspect { .. } => "inspect",
            Cmd::Stats { .. } => "stats",
            Cmd::Correlate { .. } => "correlate",
            Cmd::Timeline { .. } => "timeline",
            Cmd::Slices { .. } => "slices",
            Cmd::Collectives { .. } => "collectives",
            Cmd::Prep { .. } => "prep",
            Cmd::Schema { .. } => "schema",
        }
    }
}
