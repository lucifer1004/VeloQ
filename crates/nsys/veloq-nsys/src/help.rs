//! `--help` long_about composition for every subcommand.
//!
//! SSOT for response shape: each per-command `long_about_<cmd>()`
//! calls [`wire_format_for::<ResponseT>`] to project the actual Rust
//! struct into a compact TS-shorthand block. Static text (one-line
//! blurb, sort keys, example invocations) lives in module-level
//! constants so the per-command function stays a 3-line composer.
//! `--help` for any subcommand is therefore guaranteed in-sync with
//! the wire format.
//!
//! [`inject_long_about`] is the entry point `main()`
//! calls once at startup; it augments `derive(Parser)`'s
//! auto-generated `clap::Command` tree with the projected long_about
//! text per subcommand.

use veloq_core::SortKeyDef;
use veloq_core::recipes::recipes_for_verb;
use veloq_core::wire_format::wire_format_for;

use crate::payloads::{CorrelationStatsPayload, PrepPayload};
use crate::schema_targets;

/// Composition options for a verb's `--help` block. Lets each per-verb
/// projector say "this verb takes the location/window/nvtx filter
/// family" without duplicating the matrix text — the projector adds
/// the corresponding Common-flags entries automatically.
#[derive(Default)]
struct LongAboutOpts<'a> {
    /// Verb name (the clap subcommand string, e.g. `"stats"`). Used to
    /// project the Recipes-for-this-verb block from the recipe
    /// registry's `related_verbs` field. Default-empty when the
    /// projector doesn't want a Recipes block (meta-like verbs that
    /// don't appear in recipes).
    verb: &'a str,
    /// Per-verb sort-key documentation, projected from the `SortKey`
    /// enum's `help_text()`. None when the verb has no sortable rows.
    sort_keys: Option<&'a str>,
    /// `true` when the verb accepts `--device` / `--all-devices`. Adds
    /// the device + all-devices rows to the Common flags matrix.
    has_location: bool,
    /// `true` when the verb accepts `--stream`.
    has_stream: bool,
    /// `true` when the verb accepts `--from` / `--to`. Adds the
    /// time-window row.
    has_time_window: bool,
    /// `true` when the verb accepts `--nvtx`. Adds the NVTX scope row.
    has_nvtx: bool,
    /// 2-3 example invocations, surfaced under the Examples section.
    examples: &'a [&'a str],
}

/// Wraps the projected schema, optional sort-key reference, the
/// Recipes-for-this-verb block (discoverability
/// touchpoint 2), the Common-flags matrix, and example invocations
/// into a single multi-paragraph `long_about` block suitable for clap.
/// `T` is the subcommand's response payload type — anything
/// `#[derive(JsonSchema)]`.
///
/// **Order matters.** Recipes and Common flags appear *before* the
/// Response schema / Sort keys / Examples blocks, so the first
/// screenful an agent sees on `--help` is the workflow guidance — not
/// the wire-format reference. Reference material stays in place for
/// the agent that scrolls to it.
fn long_about_for<T: schemars::JsonSchema>(blurb: &str, opts: LongAboutOpts<'_>) -> String {
    let wf = wire_format_for::<T>();
    let mut out = String::new();
    out.push_str(blurb);

    // 1. Recipes-for-this-verb. Registry lookup is build-time-baked, so
    // this is a cheap iterator over a static slice. Empty result =
    // block omitted entirely (no blank "Recipes for this verb:" line).
    let recipes: Vec<_> = if opts.verb.is_empty() {
        Vec::new()
    } else {
        recipes_for_verb(opts.verb).collect()
    };
    if !recipes.is_empty() {
        out.push_str(
            "\n\nRecipes for this verb (run `veloq recipes <id>` for the canonical command):\n",
        );
        for r in &recipes {
            out.push_str(&format!("  {:<28} {}\n", r.id, r.title));
        }
        if out.ends_with('\n') {
            out.pop();
        }
    }

    // 2. Common-flags matrix. The projector says which families this
    // verb takes; we render the canonical one-liner for each.
    let common_rows = common_flag_rows(&opts);
    if !common_rows.is_empty() {
        out.push_str("\n\nCommon flags:\n");
        for (flag, desc) in &common_rows {
            out.push_str(&format!("  {flag:<18}{desc}\n"));
        }
        if out.ends_with('\n') {
            out.pop();
        }
    }

    out.push_str("\n\nResponse (.data):\n");
    out.push_str(&indent_block(&wf.render(), "  "));
    if let Some(keys) = opts.sort_keys {
        out.push_str("\n\nSort keys:\n");
        out.push_str(&indent_block(keys, "  "));
    }
    if !opts.examples.is_empty() {
        out.push_str("\n\nExamples:\n");
        for ex in opts.examples {
            out.push_str("  ");
            out.push_str(ex);
            out.push('\n');
        }
        // Trim trailing newline so the help body doesn't end with a
        // blank line (clap already adds its own).
        if out.ends_with('\n') {
            out.pop();
        }
    }
    out
}

/// Render the Common-flags matrix for the verb. Returns
/// `(flag, description)` pairs in canonical order; the caller renders
/// them as a two-column block.
fn common_flag_rows(opts: &LongAboutOpts<'_>) -> Vec<(&'static str, &'static str)> {
    let mut rows = Vec::new();
    if opts.has_location {
        rows.push((
            "--device <N>",
            "Restrict to one CUDA device (required on multi-device traces; \
             also scopes host-thread events via the context-info bridge).",
        ));
        rows.push((
            "--all-devices",
            "Opt into the cross-device aggregate when that's the intent.",
        ));
        if opts.has_stream {
            rows.push((
                "--stream <S>",
                "Restrict to one CUDA stream (use with --device for per-stream views).",
            ));
        }
    }
    if opts.has_time_window {
        rows.push((
            "--from / --to",
            "Time window. Accepts duration literals (1.2s, 100ms, 100us, 42ns) and the \
             `@<ns>` absolute marker.",
        ));
    }
    if opts.has_nvtx {
        rows.push((
            "--nvtx '<glob>'",
            "NVTX-scoped attribution. Use the leaf range name (`*step*`) or full path \
             (`outer/inner`).",
        ));
    }
    rows
}

fn indent_block(text: &str, prefix: &str) -> String {
    text.lines()
        .map(|l| {
            if l.is_empty() {
                String::new()
            } else {
                format!("{prefix}{l}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// ===== per-command long_about =============================================

const SUMMARY_BLURB: &str =
    "Print a one-shot overview of the trace (version, adapter, capabilities, per-table span).";
const SUMMARY_EXAMPLES: &[&str] = &[
    "veloq summary T",
    "veloq summary T | jq '.data.auxiliary.capabilities'",
];

fn long_about_summary() -> String {
    use veloq_nsys_query::summary::Summary;
    long_about_for::<Summary>(
        SUMMARY_BLURB,
        LongAboutOpts {
            verb: "summary",
            examples: SUMMARY_EXAMPLES,
            ..Default::default()
        },
    )
}

const STATS_BLURB: &str = "Aggregate kernel/memcpy/memset/sync/runtime/osrt/graph/nvtx events by \
                          name + optional --group-by axes (device, stream, context, graph, \
                          graph_node, grid_block, nvtx-parent, nvtx-path). Each row carries count + duration distribution + percentage \
                          of total. `--type nvtx` is CPU-side ranges only and rejects \
                          device/stream/context/graph axes. `--type runtime` + \
                          `--collapse-versioned` folds API versions (e.g. cudaMalloc_v3020 → \
                          cudaMalloc) into one bucket.";
const STATS_EXAMPLES: &[&str] = &[
    "veloq stats T --limit 10",
    "veloq stats T --group-by demangled --limit 20",
    "veloq stats T --type kernel --group-by nvtx-path --limit 20",
    "veloq stats T --type memcpy --sort gbps:desc",
    "veloq stats T --type sync --limit 10",
];

fn long_about_stats() -> String {
    use veloq_nsys_query::stats::{SortKey, StatsResponse};
    let keys = SortKey::help_text();
    long_about_for::<StatsResponse>(
        STATS_BLURB,
        LongAboutOpts {
            verb: "stats",
            sort_keys: Some(&keys),
            has_location: true,
            has_stream: true,
            has_time_window: true,
            has_nvtx: true,
            examples: STATS_EXAMPLES,
        },
    )
}

const SEARCH_BLURB: &str = "Filter events by name/regex/duration/window — returns row_id list + headline columns. \
     row_ids round-trip into `inspect` or `correlate` verbatim.";
const SEARCH_EXAMPLES: &[&str] = &[
    "veloq search T --name '*flash_attn*' --limit 20",
    "veloq search T --name-regex '^cudaMalloc.*' --type runtime",
    "veloq search T --duration '>1ms' --type kernel --sort duration:desc",
];

fn long_about_search() -> String {
    use veloq_nsys_query::search::{SearchResponse, SortKey};
    let keys = SortKey::help_text();
    long_about_for::<SearchResponse>(
        SEARCH_BLURB,
        LongAboutOpts {
            verb: "search",
            sort_keys: Some(&keys),
            has_location: true,
            has_stream: true,
            has_time_window: true,
            has_nvtx: true,
            examples: SEARCH_EXAMPLES,
        },
    )
}

const INSPECT_BLURB: &str = "Fetch full details for one or more events by row_id. \
                            The response is a heterogeneous array — each element's `type` tag \
                            tells you which kind-specific schema to expect. Full per-kind \
                            field sets live in references/inspect-shapes.md (auto-generated \
                            from the same JsonSchema derives this command emits via \
                            `veloq schema inspect`).";
const INSPECT_EXAMPLES: &[&str] = &[
    "veloq inspect T kernel:1234",
    "veloq inspect T cpu_sample:42",
    "veloq inspect T kernel:1 memcpy:5 nvtx:140",
];

fn long_about_inspect() -> String {
    use veloq_nsys_query::inspect::InspectResponse;
    long_about_for::<InspectResponse>(
        INSPECT_BLURB,
        LongAboutOpts {
            verb: "inspect",
            examples: INSPECT_EXAMPLES,
            ..Default::default()
        },
    )
}

const CORRELATE_BLURB: &str = "Walk the CPU↔GPU causal chain anchored on a row_id. Returns the runtime API call that \
     launched the event plus every event sharing its `correlationId` (kernels, memcpys, syncs, \
     graph rows). Use after `gaps` (drill a bubble) or `slices` (drill a slow attributed row).";
const CORRELATE_EXAMPLES: &[&str] = &[
    "veloq correlate T kernel:1234",
    "veloq correlate T sync:42",
    "veloq correlate T runtime:71 kernel:5",
];

fn long_about_correlate() -> String {
    use veloq_nsys_query::correlate::CorrelateResponse;
    long_about_for::<CorrelateResponse>(
        CORRELATE_BLURB,
        LongAboutOpts {
            verb: "correlate",
            examples: CORRELATE_EXAMPLES,
            ..Default::default()
        },
    )
}

const GRAPH_REPLAYS_BLURB: &str = "List CUDA graph replays. In graph-trace capture mode, each \
     row is one CUPTI graph execution with wall time but no internal decomposition. In node \
     capture mode, replays are grouped by (device, context, correlationId) and include the \
     top graph nodes/kernels by summed GPU time. `--nvtx` is launch-scoped: it matches \
     cudaGraphLaunch runtime calls inside NVTX ranges, then joins those launches to replay work.";
const GRAPH_REPLAYS_EXAMPLES: &[&str] = &[
    "veloq graph-replays T --device 0 --limit 20",
    "veloq graph-replays T --device 0 --sort sum:desc --top-nodes 5",
    "veloq graph-replays T --device 0 --nvtx '*frame*'",
];

fn long_about_graph_replays() -> String {
    use veloq_nsys_query::graph_replays::{GraphReplaysResponse, SortKey};
    let keys = SortKey::help_text();
    long_about_for::<GraphReplaysResponse>(
        GRAPH_REPLAYS_BLURB,
        LongAboutOpts {
            verb: "graph-replays",
            sort_keys: Some(&keys),
            has_location: true,
            has_stream: false,
            has_time_window: true,
            has_nvtx: true,
            examples: GRAPH_REPLAYS_EXAMPLES,
        },
    )
}

const NCU_COMMAND_BLURB: &str = "Generate a native `ncu` rerun command for one selected CUDA kernel \
     event. The command recovers the captured app argv/cwd from NSys metadata, counts earlier \
     matching kernel launches, and emits `--kernel-name`, `--launch-skip`, and `--launch-count 1`. \
     JSON returns the structured recipe; `--print` writes only the pipe-ready bash script.";
const NCU_COMMAND_EXAMPLES: &[&str] = &[
    "veloq nsys ncu-command T kernel:1234",
    "veloq nsys ncu-command T kernel:1234 --print",
    "veloq nsys ncu-command T kernel:1234 --print | bash",
];

fn long_about_ncu_command() -> String {
    use veloq_nsys_query::ncu_command::NcuCommandResponse;
    long_about_for::<NcuCommandResponse>(
        NCU_COMMAND_BLURB,
        LongAboutOpts {
            verb: "ncu-command",
            examples: NCU_COMMAND_EXAMPLES,
            ..Default::default()
        },
    )
}

const CORRELATION_STATS_BLURB: &str = "Build (or load) the correlation index and report row-count stats per kind. Mostly useful \
     as a smoke handle while NVTX-scoped commands land; the index is reused by `stats --nvtx`.";
const CORRELATION_STATS_EXAMPLES: &[&str] = &["veloq correlation-stats T"];

fn long_about_correlation_stats() -> String {
    long_about_for::<CorrelationStatsPayload>(
        CORRELATION_STATS_BLURB,
        LongAboutOpts {
            verb: "correlation-stats",
            examples: CORRELATION_STATS_EXAMPLES,
            ..Default::default()
        },
    )
}

const PREP_BLURB: &str = "Prepare a trace for fast queries: export every nsys table to a parquetdir at \
     `<trace>.veloq/parquetdir/` (if not already present and fresh) and warm the metadata \
     cache. Other commands auto-prep on first heavy use; agents that know they'll query the \
     same trace many times benefit from running this first explicitly. `--status` is the \
     read-only form: it reports parquetdir presence / table names and meta-cache fingerprint \
     state without building anything — a cheap pre-flight check before scripting a batch of \
     heavy verbs.";
const PREP_EXAMPLES: &[&str] = &["veloq prep T"];

fn long_about_prep() -> String {
    long_about_for::<PrepPayload>(
        PREP_BLURB,
        LongAboutOpts {
            verb: "prep",
            examples: PREP_EXAMPLES,
            ..Default::default()
        },
    )
}

const CONCURRENCY_BLURB: &str = "GPU kernel/transfer overlap, extracted as union-vs-sum of event \
     intervals. Per device: sum_busy_ns (Σ durations), union_busy_ns (wall time ≥1 event ran), \
     overlap_ns (= sum − union; >0 means concurrency), and max_concurrency (peak simultaneous \
     events). Each device row nests a per-stream breakdown — a stream's own overlap_ns is its \
     same-stream (e.g. Programmatic Dependent Launch) overlap — plus a compute_vs_copy block \
     (compute = kernel/graph, copy = memcpy/memset). Extraction only: no ratio or verdict; compute \
     any ratio in jq. Reported per device, never summed across devices.";
const CONCURRENCY_EXAMPLES: &[&str] = &[
    "veloq concurrency T",
    "veloq concurrency T --device 0",
    "veloq concurrency T | jq '.data.rows[] | {device_id, overlap_ns, max_concurrency}'",
];

fn long_about_concurrency() -> String {
    use veloq_nsys_query::concurrency::ConcurrencyResponse;
    long_about_for::<ConcurrencyResponse>(
        CONCURRENCY_BLURB,
        LongAboutOpts {
            verb: "concurrency",
            has_location: true,
            has_time_window: true,
            examples: CONCURRENCY_EXAMPLES,
            ..Default::default()
        },
    )
}

const GAPS_BLURB: &str = "Find GPU idle bubbles. Three scopes via `--scope`:\n\
                         \n\
                         - `device` (default): per device, gap = window where no stream was \
                         running GPU work. Cross-stream concurrency is accounted for — long-idle \
                         peer streams don't produce phantom gaps.\n\
                         - `stream`: per (device, stream), gap = window between consecutive \
                         events on that stream. Use for per-stream starvation diagnostics.\n\
                         - `trace`: across all devices, gap = window where no device ran GPU \
                         work. Multi-GPU rig idle analysis.\n\
                         \n\
                         Each gap reports duration + `prev`/`next` events (with stream context). \
                         Under unified scopes the bracketing events may live on different \
                         streams — `prev.stream_id` / `next.stream_id` make that visible.";
const GAPS_EXAMPLES: &[&str] = &[
    "veloq gaps T --min-duration 1ms --limit 20",
    "veloq gaps T --scope stream --device 0 --stream 7 --sort start:asc",
    "veloq gaps T --scope trace --min-duration 5ms",
];

fn long_about_gaps() -> String {
    use veloq_nsys_query::gaps::{GapsResponse, SortKey};
    let keys = SortKey::help_text();
    long_about_for::<GapsResponse>(
        GAPS_BLURB,
        LongAboutOpts {
            verb: "gaps",
            sort_keys: Some(&keys),
            has_location: true,
            has_stream: true,
            has_time_window: true,
            examples: GAPS_EXAMPLES,
            ..Default::default()
        },
    )
}

const TIMELINE_BLURB: &str = "Time-bucketed GPU activity. Each fixed-width bucket reports `total_ns` + per-kind \
     breakdown + event counts. Events straddling bucket edges are clipped. `total_ns` is the SUM of \
     clipped per-event durations, not their union: under concurrent streams it double-counts overlap \
     and can exceed the bucket width — it measures work issued per window, not device-busy time. For \
     true union busy/idle time use `concurrency` or `gaps`. Useful for timeline plots and saturation trends.";
const TIMELINE_EXAMPLES: &[&str] = &[
    "veloq timeline T --interval 100ms --limit 100",
    "veloq timeline T --interval 1s --type kernel,memcpy",
];

fn long_about_timeline() -> String {
    use veloq_nsys_query::timeline::TimelineResponse;
    long_about_for::<TimelineResponse>(
        TIMELINE_BLURB,
        LongAboutOpts {
            verb: "timeline",
            has_location: true,
            has_stream: true,
            has_time_window: true,
            has_nvtx: true,
            examples: TIMELINE_EXAMPLES,
            ..Default::default()
        },
    )
}

const SLICES_BLURB: &str = "For each NVTX range matching --name, return CPU host-thread bounds \
                           + per-(device, stream) GPU work attributed via correlationId. \
                           The headline regression-hunt tool: compare two iterations' \
                           `attributed_kernel_ns`. Add --aggregate to aggregate matching range \
                           instances by scope name/path and return instances + attributed_total_ns \
                           + p50/p99 distribution rows. Aggregate sort keys: total, instances, \
                           p50/typical, p99/tail, name, path.";
const SLICES_EXAMPLES: &[&str] = &[
    "veloq slices T --name '*step*' --limit 10",
    "veloq slices T --name-regex 'iter_[0-9]+' --sort attributed_kernel:desc",
    "veloq slices T --name 'iter_*' --aggregate --sort tail:desc --limit 20",
    "veloq slices T --aggregate --group-by path --sort path:asc --limit 200",
];

fn long_about_slices() -> String {
    use veloq_nsys_query::slices::{SlicesResponse, SortKey};
    let keys = SortKey::help_text();
    long_about_for::<SlicesResponse>(
        SLICES_BLURB,
        LongAboutOpts {
            verb: "slices",
            sort_keys: Some(&keys),
            has_location: true,
            has_stream: true,
            has_time_window: true,
            examples: SLICES_EXAMPLES,
            ..Default::default()
        },
    )
}

const HARDWARE_BLURB: &str = "Profiled CPU / GPU / NIC inventory from the trace's `TARGET_INFO_*` tables. Returns \
     `{ rows: [] }` (not an error) when those tables are absent — gate on \
     `summary.auxiliary.capabilities.has_target_info` first.";
const HARDWARE_EXAMPLES: &[&str] = &[
    "veloq hardware T",
    "veloq hardware T | jq '.data.rows[0].gpus'",
];

fn long_about_hardware() -> String {
    use veloq_nsys_query::hardware::HardwareResponse;
    long_about_for::<HardwareResponse>(
        HARDWARE_BLURB,
        LongAboutOpts {
            verb: "hardware",
            examples: HARDWARE_EXAMPLES,
            ..Default::default()
        },
    )
}

const METRICS_BLURB: &str = "Hardware performance-counter / CPU sample / CPU sched queries. \
     `--type gpu` reads GPU_METRICS + TARGET_INFO_GPU_METRICS; \
     `--type nic` reads NET_NIC_METRIC + TARGET_INFO_NETWORK_METRICS; \
     `--type cpu-sampling` reads COMPOSITE_EVENTS + SAMPLING_CALLCHAINS; \
     `--type cpu-sched` reads SCHED_EVENTS (a transition stream — sched-in/out \
     events paired into precise on-cpu durations). Every response carries a \
     `coverage` block — read `coverage.ratio` and GPU/NIC `coverage.max_gap_ns` \
     before trusting values; nsys's metric/sample/sched buffers can silently drop data on long traces. \
     cpu-sampling adds three more trust signals: `unresolved_leaf_share`, \
     `kernel_leaf_share`, `truncated_stack_share`. cpu-sched adds two: \
     `unresolved_state_share`, `per_cpu_max_gap_ns`.";
const METRICS_EXAMPLES: &[&str] = &[
    "veloq metrics T --type gpu --limit 8 --sort=mean:desc",
    "veloq metrics T --type gpu --counter '*Throughput*' --bucket 50ms",
    "veloq metrics T --type nic --counter 'IB: Bytes*' --bucket 50ms",
    "veloq metrics T --type cpu-sampling --limit 20",
    "veloq metrics T --type cpu-sampling --group-by tid --limit 10",
    "veloq metrics T --type cpu-sched --group-by tid --limit 10",
    "veloq metrics T --type cpu-sched --bucket 1s",
];

fn long_about_metrics() -> String {
    use veloq_nsys_query::metrics::{
        CounterSortKey, HotspotSortKey, MetricsResponse, NicCounterSortKey, SchedSortKey,
    };
    // Three `--type`s consume distinct vocabularies; project each so
    // the agent sees what each mode accepts.
    let keys = format!(
        "--type gpu: {}\n--type nic: {}\n--type cpu-sampling: {}\n--type cpu-sched: {}",
        CounterSortKey::help_text(),
        NicCounterSortKey::help_text(),
        HotspotSortKey::help_text(),
        SchedSortKey::help_text(),
    );
    long_about_for::<MetricsResponse>(
        METRICS_BLURB,
        LongAboutOpts {
            verb: "metrics",
            sort_keys: Some(&keys),
            has_location: true,
            has_time_window: true,
            examples: METRICS_EXAMPLES,
            ..Default::default()
        },
    )
}

const SCHEMA_BLURB: &str = "Meta endpoint — emits the strict JSON Schema for one subcommand's response body. \
     SSOT for tooling that wants machine-validated wire format; the schema is generated \
     from the same Rust structs `serde_json` serialises out of, so it cannot drift. \
     Envelope shape is `{ schema, source, command, data: { target, schema } }` (no `trace` field — this \
     call doesn't read a trace).";
const SCHEMA_EXAMPLES: &[&str] = &[
    "veloq schema stats",
    "veloq schema metrics | jq '.data.schema.$defs.Coverage'",
    "veloq schema inspect | jq '.data.schema.$defs | keys'",
];

/// `long_about_schema` is handcrafted — the schema endpoint's
/// *response shape* is itself a JSON Schema document (variable per
/// target), so there's no fixed Rust struct to project. The text
/// describes the envelope contract and lists valid targets pulled
/// from the shared [`schema_targets::TARGETS`] registry.
pub fn long_about_schema() -> String {
    let mut out = String::new();
    out.push_str(SCHEMA_BLURB);
    out.push_str("\n\nValid targets: ");
    out.push_str(&schema_targets::render_target_list());
    out.push('.');
    out.push_str("\n\nResponse envelope:\n  ");
    out.push_str(
        "{ schema: \"v1\", source: { kind: \"nsys\", version: \"v1\" }, \
         command: \"nsys.schema\", data: { target: <string>, schema: <JSON Schema document> } }",
    );
    out.push_str("\n\nExamples:\n");
    for ex in SCHEMA_EXAMPLES {
        out.push_str("  ");
        out.push_str(ex);
        out.push('\n');
    }
    if out.ends_with('\n') {
        out.pop();
    }
    out
}

/// Help text shown on the `schema` subcommand's `target` positional
/// arg, generated from the shared registry so the per-arg help and
/// the long_about cannot drift.
pub(crate) fn schema_target_arg_help() -> String {
    format!(
        "Subcommand whose response schema to print. One of: {}",
        schema_targets::render_target_list()
    )
}

/// Patch every NSys subcommand's `long_about` onto the supplied
/// top-level [`clap::Command`] tree. Called once at the start of
/// `main()` with the parser the binary built from its own
/// `derive(Parser)`; the resulting tree is what `--help` and parsing
/// both see.
///
/// Takes the command by value (and returns it) rather than building
/// the tree itself because the binary owns the top-level structure
/// (global `--format`, subcommand registry); this helper is just the
/// projector that swaps in SSOT-derived `long_about` text on the
/// NSys subcommands.
pub fn inject_long_about(cmd: clap::Command) -> clap::Command {
    cmd.mut_subcommand("summary", |c| c.long_about(long_about_summary()))
        .mut_subcommand("stats", |c| c.long_about(long_about_stats()))
        .mut_subcommand("search", |c| c.long_about(long_about_search()))
        .mut_subcommand("inspect", |c| c.long_about(long_about_inspect()))
        .mut_subcommand("correlate", |c| c.long_about(long_about_correlate()))
        .mut_subcommand("graph-replays", |c| {
            c.long_about(long_about_graph_replays())
        })
        .mut_subcommand("ncu-command", |c| c.long_about(long_about_ncu_command()))
        .mut_subcommand("correlation-stats", |c| {
            c.long_about(long_about_correlation_stats())
        })
        .mut_subcommand("prep", |c| c.long_about(long_about_prep()))
        .mut_subcommand("concurrency", |c| c.long_about(long_about_concurrency()))
        .mut_subcommand("gaps", |c| c.long_about(long_about_gaps()))
        .mut_subcommand("timeline", |c| c.long_about(long_about_timeline()))
        .mut_subcommand("slices", |c| c.long_about(long_about_slices()))
        .mut_subcommand("hardware", |c| c.long_about(long_about_hardware()))
        .mut_subcommand("metrics", |c| c.long_about(long_about_metrics()))
        .mut_subcommand("schema", |c| {
            c.long_about(long_about_schema())
                .mut_arg("target", |a| a.help(schema_target_arg_help()))
        })
}
