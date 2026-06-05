//! `veloq stats <trace>` — aggregated GPU work statistics.
//!
//! Returns one row per kernel/memcpy/memset *name*, with count, total
//! duration, distribution (min/max/p50/p95/p99), and percentage of
//! total. Optionally filters by event type and time window.

use crate::{EventKind, KindFilter};
use anyhow::{Context, Result};
use duckdb::types::Value;
use serde::Serialize;
use std::path::Path;
use veloq_core::{Direction, SortKeyDef, SortKeySpec, SortSpec, time::TimeWindow};
use veloq_nsys_data::Trace;

/// Event kinds that `stats` is willing to aggregate. Library consumers
/// constructing `StatsRequest` by hand should pick from this set; CLI
/// callers go through `GpuFilters::kinds(&Self::ALLOWED_KINDS)`.
///
/// Sync is included because `cudaStreamSynchronize` / `cudaDeviceSynchronize`
/// durations are agent-actionable signals (CPU blocked waiting on GPU),
/// and aggregating by syncType is the natural way to see which sync
/// dominates a workload.
///
/// Graph is included because, in `--cuda-graph-trace=graph` captures
/// (the common default), kernels-inside-graphs do not appear in
/// `CUPTI_ACTIVITY_KIND_KERNEL` — the graph_trace row is the *only*
/// per-execution record for that work. Excluding it would silently
/// undercount GPU work on graph-heavy workloads (vLLM, TRT-LLM).
///
/// Nvtx is included so agents can ask "what's the per-step duration
/// distribution" without leaving stats. NVTX ranges are CPU-side
/// markers (start/end on the host thread; no device / stream), and
/// instant markers (`end IS NULL`) are excluded — they have no
/// duration. Mixing NVTX with GPU kinds via `KindFilter::All` still
/// works: the SQL UNION projects NULL for device / stream on NVTX
/// rows and the per-group totals stay correct, but agents who want
/// "GPU work only" should narrow with `--type kernel,memcpy,memset`.
///
/// **Aggregation caveat**: NVTX ranges sharing a name across multiple
/// host threads (or threads driving different GPUs) fold into one
/// group under the default `--group-by short`. There is no
/// device axis on NVTX_EVENTS — `--group-by device|context|stream|
/// graph|graph_node` on `--type nvtx` is rejected up-front rather
/// than emit a single `null` bucket. A future per-thread axis can
/// disambiguate; today, agents that need it should run separate
/// queries with `--time-range` narrowing per region of interest.
pub const ALLOWED_KINDS: [EventKind; 8] = [
    EventKind::Kernel,
    EventKind::Memcpy,
    EventKind::Memset,
    EventKind::Sync,
    EventKind::Graph,
    EventKind::Nvtx,
    EventKind::Runtime,
    EventKind::Osrt,
];

/// Identity dimension: how the kernel name folds across rows.
/// Mutually exclusive (only one can be active in any `--group-by`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NameAxis {
    /// One row per shortName (and per kind). Default.
    #[default]
    ShortName,
    /// One row per template instantiation — for kernels, that's the
    /// fully demangled signature; for memcpy/memset, identical to
    /// ShortName.
    Demangled,
    /// One row per mangled (raw Itanium ABI / MSVC) symbol — for
    /// kernels with template parameters, this is the wire-format
    /// the compiler emits. Distinct from [`NameAxis::Demangled`]: two
    /// link-distinct kernels can demangle to the same C++ signature
    /// (e.g., differing on a `__restrict__` or `volatile` qualifier
    /// stripped during demangling), and the mangled axis preserves
    /// that distinction. Falls back to [`NameAxis::Demangled`] on
    /// traces missing the `mangledName` column (older NSys schemas)
    /// — see [`StatsResponse::mangled_axis_fallback`].
    Mangled,
    /// No name axis at all — rows roll up across kernels of the same
    /// kind. Use this for "per-device totals" / "per-stream totals"
    /// when the kernel breakdown doesn't matter.
    None,
}

/// Compositional grouping for `stats`.
///
/// `--group-by` accepts a comma-separated list of tokens. At most one
/// of {short, demangled, mangled, no-name} is allowed (the name layer).
/// The physical-dimension tokens {device, context, stream} can be
/// combined freely. Order doesn't matter.
///
/// | tokens (any order)            | rows emitted (per kind)                |
/// | ----------------------------- | -------------------------------------- |
/// | (default = `short`)           | one per shortName                      |
/// | `demangled`                   | one per demangled signature            |
/// | `mangled`                     | one per mangled symbol                 |
/// | `device`                      | one per device, rolled across kernels  |
/// | `demangled,device`            | one per (demangled, device)            |
/// | `mangled,device`              | one per (mangled, device)              |
/// | `short,device,stream`         | one per (shortName, device, stream)    |
/// | `no-name,device,stream`       | one per (device, stream)               |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GroupBy {
    pub name: NameAxis,
    pub device: bool,
    pub context: bool,
    pub stream: bool,
    /// Group by captured-graph id (kernel column `graphId`). Useful on
    /// `--cuda-graph-trace=node` traces to answer "how much time per
    /// captured graph." Always paired with `kind` so eager (NULL
    /// `graphId`) and graph-captured rows don't mix.
    pub graph: bool,
    /// Group by per-node id (kernel/memcpy/memset column
    /// `graphNodeId`). Per-node-within-a-graph breakdown — the unique
    /// signal for node-mode captures.
    pub graph_node: bool,
    /// Group by innermost enclosing NVTX range.
    /// Mutually exclusive with `graph` / `graph_node` (different
    /// attribution model — NVTX is host-thread containment; graph
    /// captures are device-side capture state). Events with no
    /// enclosing NVTX range fold into the visible sentinel
    /// `nvtx_parent_name = "__no_nvtx__"`.
    pub nvtx_parent: bool,
    /// Group by the full slash-joined path of the innermost enclosing
    /// NVTX range. Same
    /// attribution model as `nvtx_parent`, but collapses repeated
    /// range instances by hierarchy path rather than by source rowid.
    pub nvtx_path: bool,
    /// Group by `(gridX,Y,Z, blockX,Y,Z)` launch config.
    /// Kernel-only — error rather than
    /// project NULL buckets for non-kernel kinds. Composes with the
    /// name axis: `--group-by demangled,grid_block` splits each
    /// kernel signature into one row per launch shape.
    pub grid_block: bool,
}

/// Sort axes `stats` supports. Keys map to aggregate columns in the
/// `grouped` CTE; the caller can override direction via `:asc`/`:desc`/
/// `-`/`+`. Default direction picked per-key to match the common case
/// (e.g., `total` → DESC, `name` → ASC).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortKey {
    Total,
    Count,
    Avg,
    Min,
    Max,
    P50,
    P95,
    P99,
    Bytes,
    Gbps,
    Name,
    Device,
    Stream,
    Context,
    Graph,
    GraphNode,
}

impl SortKeyDef for SortKey {
    fn specs() -> &'static [SortKeySpec<Self>] {
        // Per-key default direction is what `--sort key` resolves to
        // when the caller didn't say `:asc`/`:desc`. Aggregate axes
        // (total, count, p99, …) default DESC because the natural
        // question is "biggest first"; identity axes (name, device,
        // stream) default ASC because they're for browsing.
        //
        // The order of this table controls how `--help` lists the
        // keys and what shows up first in the `expected: ...` part
        // of the error message — keep aggregates before identities
        // so the most useful keys lead.
        &[
            SortKeySpec {
                variant: SortKey::Total,
                canonical: "total",
                aliases: &["total_ns"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SortKey::Count,
                canonical: "count",
                aliases: &[],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SortKey::Avg,
                canonical: "avg",
                aliases: &["avg_ns"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SortKey::Min,
                canonical: "min",
                aliases: &["min_ns"],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: SortKey::Max,
                canonical: "max",
                aliases: &["max_ns"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SortKey::P50,
                canonical: "p50",
                aliases: &["p50_ns"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SortKey::P95,
                canonical: "p95",
                aliases: &["p95_ns"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SortKey::P99,
                canonical: "p99",
                aliases: &["p99_ns"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SortKey::Bytes,
                canonical: "bytes",
                aliases: &["bytes_total"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SortKey::Gbps,
                canonical: "gbps",
                aliases: &[],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: SortKey::Name,
                canonical: "name",
                aliases: &[],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: SortKey::Device,
                canonical: "device",
                aliases: &["device_id", "dev"],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: SortKey::Stream,
                canonical: "stream",
                aliases: &["stream_id"],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: SortKey::Context,
                canonical: "context",
                aliases: &["context_id", "ctx"],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: SortKey::Graph,
                canonical: "graph",
                aliases: &["graph_id"],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: SortKey::GraphNode,
                canonical: "graph_node",
                aliases: &["graph_node_id", "node"],
                default_dir: Direction::Asc,
            },
        ]
    }
}

impl SortKey {
    fn column(self) -> &'static str {
        match self {
            Self::Total => "total_ns",
            Self::Count => "count",
            Self::Avg => "avg_ns",
            Self::Min => "min_ns",
            Self::Max => "max_ns",
            Self::P50 => "p50_ns",
            Self::P95 => "p95_ns",
            Self::P99 => "p99_ns",
            Self::Bytes => "bytes_total",
            Self::Gbps => "gbps",
            Self::Name => "name",
            Self::Device => "device_id",
            Self::Stream => "stream_id",
            Self::Context => "context_id",
            Self::Graph => "graph_id",
            Self::GraphNode => "graph_node_id",
        }
    }
}

/// SQL fragments derived from a `GroupBy` request, kept out of
/// `run()` so the dispatch lives in one place. Each `_select` field is
/// the column expression for the outer SELECT projection; `group_keys_sql`
/// is the `GROUP BY` body.
struct GroupBySql {
    name_select: &'static str,
    short_name_select: &'static str,
    device_select: &'static str,
    context_select: &'static str,
    stream_select: &'static str,
    graph_select: &'static str,
    graph_node_select: &'static str,
    /// SELECT fragment for the innermost-NVTX-range rowid. Either a
    /// passthrough of the `events.nvtx_parent_rowid` column (axis
    /// active) or a typed NULL literal (axis inactive).
    nvtx_parent_rowid_select: &'static str,
    /// SELECT fragment for the parent NVTX range name. When the axis
    /// is active, projects the COALESCE-sentinel string from `events`;
    /// otherwise NULL. Wrapped in `--group-by nvtx-parent` so the
    /// non-axis paths keep their existing schema.
    nvtx_parent_name_select: &'static str,
    /// SELECT fragment for the full NVTX path. Populated only when
    /// `--group-by nvtx-path` is active.
    nvtx_path_select: &'static str,
    /// SELECT fragments for the enclosing range's NVTX domain identity
    /// `(domainId, pid)`. Populated only when `--group-by nvtx-path` is
    /// active; both participate in GROUP BY so same-name ranges in
    /// distinct `(pid, domainId)` domains do not collapse.
    nvtx_domain_id_select: &'static str,
    nvtx_domain_pid_select: &'static str,
    /// 6 SELECT fragments for the kernel launch grid/block tuple
    /// — when the `grid_block` axis is active these passthrough the
    /// per-kind columns; otherwise they project typed NULLs so the
    /// outer SELECT shape stays stable.
    grid_x_select: &'static str,
    grid_y_select: &'static str,
    grid_z_select: &'static str,
    block_x_select: &'static str,
    block_y_select: &'static str,
    block_z_select: &'static str,
    group_keys_sql: String,
}

/// Decision recorded by [`resolve_name_axis`] so the response and
/// `--group-by` SQL builder agree on whether to use Mangled or fall
/// back. `mangled_axis_fallback` on the response surfaces the same
/// signal to consumers.
#[derive(Debug, Clone, Copy)]
struct NameAxisResolution {
    effective: NameAxis,
    fell_back: bool,
}

/// `NameAxis::Mangled` falls back to `Demangled` when the trace's
/// `CUPTI_ACTIVITY_KIND_KERNEL` table lacks a `mangledName` column
/// (older NSys schemas). The fallback is logged via `log::info!` so
/// it lands on stderr in human modes and gets suppressed by
/// `2> /dev/null` in JSON pipelines.
/// The response carries `mangled_axis_fallback: true` so JSON consumers
/// who care can still detect it without parsing stderr.
fn resolve_name_axis(
    requested: NameAxis,
    columns: &crate::column_map::ColumnMap,
) -> NameAxisResolution {
    if !matches!(requested, NameAxis::Mangled) {
        return NameAxisResolution {
            effective: requested,
            fell_back: false,
        };
    }
    if crate::column_map::has(columns, "CUPTI_ACTIVITY_KIND_KERNEL", "mangledName") {
        return NameAxisResolution {
            effective: NameAxis::Mangled,
            fell_back: false,
        };
    }
    log::info!(
        "stats: --group-by mangled falling back to demangled — \
         trace's CUPTI_ACTIVITY_KIND_KERNEL has no mangledName column \
         (older NSys schema)"
    );
    NameAxisResolution {
        effective: NameAxis::Demangled,
        fell_back: true,
    }
}

impl GroupBySql {
    fn for_axes(g: &GroupBy) -> Self {
        // `kind` is ALWAYS grouped so we never silently mix
        // kernel+memcpy totals; everything else is conditional on the
        // request.
        //
        // `nvtx_style` (derived label from event_type via
        // [`NVTX_STYLE_EXPR`]) is always part of the group key. For
        // non-NVTX rows it's NULL, so all GPU rows collapse into one
        // null bucket per group — counts/totals unchanged. For NVTX
        // rows it splits PushPop vs StartEnd ranges with the same name
        // into distinct buckets, mirroring nsys's `nvtx_sum`.
        let mut group_keys: Vec<String> = vec!["kind".to_string(), NVTX_STYLE_EXPR.to_string()];
        let (name_select, short_name_select) = match g.name {
            NameAxis::ShortName => {
                group_keys.push("short_name".to_string());
                // name and short_name are the same value here, but
                // keeping short_name in the response preserves a
                // stable schema across `--group-by` modes (agents
                // always find it on kernel rows). No aggregator
                // needed — short_name is in GROUP BY.
                ("short_name AS name", "short_name")
            }
            NameAxis::Demangled => {
                group_keys.push("display_name".to_string());
                group_keys.push("short_name".to_string());
                // short_name carried so agent can roll demangled rows
                // back. Both columns are in GROUP BY, so no
                // aggregator needed.
                ("display_name AS name", "short_name")
            }
            NameAxis::Mangled => {
                // Group on the raw mangled symbol (preserves link
                // identity even when two kernels demangle the same).
                // short_name stays in GROUP BY so the response
                // carries the shortName the mangled row rolls up to,
                // mirroring the Demangled axis. mangled→demangled is
                // ABI-deterministic so pinning short_name keeps the
                // demangle-id stable across runs.
                group_keys.push("mangled_name".to_string());
                group_keys.push("short_name".to_string());
                ("mangled_name AS name", "short_name")
            }
            NameAxis::None => (
                "CAST(NULL AS VARCHAR) AS name",
                "CAST(NULL AS VARCHAR) AS short_name",
            ),
        };
        let device_select = if g.device {
            group_keys.push("device_id".to_string());
            "device_id"
        } else {
            "CAST(NULL AS INTEGER) AS device_id"
        };
        let context_select = if g.context {
            group_keys.push("context_id".to_string());
            "context_id"
        } else {
            "CAST(NULL AS BIGINT) AS context_id"
        };
        let stream_select = if g.stream {
            group_keys.push("stream_id".to_string());
            "stream_id"
        } else {
            "CAST(NULL AS BIGINT) AS stream_id"
        };
        let graph_select = if g.graph {
            group_keys.push("graph_id".to_string());
            "graph_id"
        } else {
            "CAST(NULL AS BIGINT) AS graph_id"
        };
        let graph_node_select = if g.graph_node {
            group_keys.push("graph_node_id".to_string());
            "graph_node_id"
        } else {
            "CAST(NULL AS BIGINT) AS graph_node_id"
        };
        let (nvtx_parent_rowid_select, nvtx_parent_name_select) = if g.nvtx_parent {
            // Both columns come from the per-kind LEFT JOIN inside
            // per_kind_subquery: real values for events that landed
            // in a parent range, NULL/sentinel otherwise.
            group_keys.push("nvtx_parent_rowid".to_string());
            group_keys.push("nvtx_parent_name".to_string());
            ("nvtx_parent_rowid", "nvtx_parent_name")
        } else {
            (
                "CAST(NULL AS BIGINT) AS nvtx_parent_rowid",
                "CAST(NULL AS VARCHAR) AS nvtx_parent_name",
            )
        };
        let nvtx_path_select = if g.nvtx_path {
            group_keys.push("nvtx_path".to_string());
            "nvtx_path"
        } else {
            "CAST(NULL AS VARCHAR) AS nvtx_path"
        };
        // Domain identity is part of the path axis: two same-name /
        // same-parent ranges in distinct (pid, domainId) domains must
        // not collapse. Both columns join GROUP BY when
        // the path axis is active; NULL otherwise.
        let (nvtx_domain_id_select, nvtx_domain_pid_select) = if g.nvtx_path {
            group_keys.push("nvtx_domain_id".to_string());
            group_keys.push("nvtx_domain_pid".to_string());
            ("nvtx_domain_id", "nvtx_domain_pid")
        } else {
            (
                "CAST(NULL AS BIGINT) AS nvtx_domain_id",
                "CAST(NULL AS BIGINT) AS nvtx_domain_pid",
            )
        };
        // Each axis column needs its own GROUP BY entry
        // so two kernels with different (grid, block) shapes don't
        // fold into one row. Stays kernel-scoped — the axis caller
        // validated earlier that only kernel rows reach here.
        let (grid_x_select, grid_y_select, grid_z_select) = if g.grid_block {
            group_keys.push("grid_x".to_string());
            group_keys.push("grid_y".to_string());
            group_keys.push("grid_z".to_string());
            ("grid_x", "grid_y", "grid_z")
        } else {
            (
                "CAST(NULL AS BIGINT) AS grid_x",
                "CAST(NULL AS BIGINT) AS grid_y",
                "CAST(NULL AS BIGINT) AS grid_z",
            )
        };
        let (block_x_select, block_y_select, block_z_select) = if g.grid_block {
            group_keys.push("block_x".to_string());
            group_keys.push("block_y".to_string());
            group_keys.push("block_z".to_string());
            ("block_x", "block_y", "block_z")
        } else {
            (
                "CAST(NULL AS BIGINT) AS block_x",
                "CAST(NULL AS BIGINT) AS block_y",
                "CAST(NULL AS BIGINT) AS block_z",
            )
        };
        Self {
            name_select,
            short_name_select,
            device_select,
            context_select,
            stream_select,
            graph_select,
            graph_node_select,
            nvtx_parent_rowid_select,
            nvtx_parent_name_select,
            nvtx_path_select,
            nvtx_domain_id_select,
            nvtx_domain_pid_select,
            grid_x_select,
            grid_y_select,
            grid_z_select,
            block_x_select,
            block_y_select,
            block_z_select,
            group_keys_sql: group_keys.join(", "),
        }
    }
}

/// SQL fragments for the optional duration-histogram columns. Two
/// outputs: the bucket aggregators inside the `grouped` CTE, and the
/// passthrough projection in the outer SELECT.
struct HistSql {
    grouped_cols: String,
    outer_cols: String,
}

impl HistSql {
    /// Build histogram column SQL when `enabled`; otherwise return
    /// empty strings so the splice in `run()` stays neutral.
    fn build(enabled: bool) -> Self {
        if !enabled {
            return Self {
                grouped_cols: String::new(),
                outer_cols: String::new(),
            };
        }
        let mut g = String::new();
        let mut o = String::new();
        let mut prev: Option<i64> = None;
        for (i, &b) in HIST_BOUNDARIES_NS.iter().enumerate() {
            let cond = match prev {
                None => format!("duration > 0 AND duration < {b}"),
                Some(p) => format!("duration >= {p} AND duration < {b}"),
            };
            g.push_str(&format!(
                ", CAST(SUM(CASE WHEN {cond} THEN 1 ELSE 0 END) AS BIGINT) AS hist_b{i}"
            ));
            o.push_str(&format!(", hist_b{i}"));
            prev = Some(b);
        }
        let tail_idx = HIST_BOUNDARIES_NS.len();
        let last = HIST_BOUNDARIES_NS.last().copied().unwrap_or(0);
        g.push_str(&format!(
            ", CAST(SUM(CASE WHEN duration >= {last} THEN 1 ELSE 0 END) AS BIGINT) AS hist_b{tail_idx}"
        ));
        o.push_str(&format!(", hist_b{tail_idx}"));
        Self {
            grouped_cols: g,
            outer_cols: o,
        }
    }
}

fn stats_sort_sql(spec: &SortSpec) -> anyhow::Result<String> {
    let mut resolved: Vec<(&'static str, Direction)> = Vec::new();
    for f in spec.fields() {
        let (k, d) = SortKey::from_field(f)?;
        resolved.push((k.column(), d));
    }
    // total_ns as the last-resort tiebreaker keeps output deterministic
    // even when the user sorts by something like `name` and several
    // groups happen to share a name (shouldn't happen, but cheap).
    Ok(veloq_core::sort::build_order_by(&resolved, "total_ns"))
}

impl GroupBy {
    pub fn from_arg(s: &str) -> anyhow::Result<Self> {
        let mut out = Self::default();
        let mut name_seen: Option<&'static str> = None;
        for raw in s.split(',') {
            let tok = raw.trim().to_ascii_lowercase();
            match tok.as_str() {
                "" => continue,
                "short" | "shortname" | "short_name" => {
                    if let Some(prev) = name_seen {
                        anyhow::bail!(
                            "--group-by name axis specified twice (`{prev}` and `short`); pick one"
                        );
                    }
                    out.name = NameAxis::ShortName;
                    name_seen = Some("short");
                }
                "demangled" | "demangled_name" | "variant" => {
                    if let Some(prev) = name_seen {
                        anyhow::bail!(
                            "--group-by name axis specified twice (`{prev}` and `demangled`); pick one"
                        );
                    }
                    out.name = NameAxis::Demangled;
                    name_seen = Some("demangled");
                }
                "mangled" | "mangled_name" => {
                    if let Some(prev) = name_seen {
                        anyhow::bail!(
                            "--group-by name axis specified twice (`{prev}` and `mangled`); pick one"
                        );
                    }
                    out.name = NameAxis::Mangled;
                    name_seen = Some("mangled");
                }
                "no-name" | "noname" | "none" => {
                    if let Some(prev) = name_seen {
                        anyhow::bail!(
                            "--group-by name axis specified twice (`{prev}` and `no-name`); pick one"
                        );
                    }
                    out.name = NameAxis::None;
                    name_seen = Some("no-name");
                }
                "device" | "dev" => out.device = true,
                "stream" => out.stream = true,
                "context" | "ctx" => out.context = true,
                "graph" | "graph_id" => out.graph = true,
                "graph_node" | "graphnode" | "graph_node_id" | "node" => out.graph_node = true,
                "nvtx-parent" | "nvtx_parent" => out.nvtx_parent = true,
                "nvtx-path" | "nvtx_path" => out.nvtx_path = true,
                "grid_block" | "grid-block" | "gridblock" => out.grid_block = true,
                other => anyhow::bail!(
                    "unknown --group-by token `{other}` (expected: short, demangled, mangled, no-name, device, stream, context, graph, graph_node, nvtx-parent, nvtx-path, grid_block)"
                ),
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod group_by_tests {
    use super::*;

    #[test]
    fn default_is_shortname_no_dims() -> anyhow::Result<()> {
        let g = GroupBy::from_arg("short")?;
        assert_eq!(
            g,
            GroupBy {
                name: NameAxis::ShortName,
                ..Default::default()
            }
        );
        Ok(())
    }

    #[test]
    fn parses_dimensions() -> anyhow::Result<()> {
        let g = GroupBy::from_arg("device,stream")?;
        assert!(g.device && g.stream && !g.context);
        // No explicit name → default ShortName
        assert_eq!(g.name, NameAxis::ShortName);
        Ok(())
    }

    #[test]
    fn no_name_axis() -> anyhow::Result<()> {
        let g = GroupBy::from_arg("no-name,device")?;
        assert_eq!(g.name, NameAxis::None);
        assert!(g.device);
        Ok(())
    }

    #[test]
    fn rejects_two_name_axes() {
        assert!(GroupBy::from_arg("short,demangled").is_err());
        assert!(GroupBy::from_arg("demangled,no-name").is_err());
        assert!(GroupBy::from_arg("mangled,demangled").is_err());
    }

    #[test]
    fn rejects_unknown_token() {
        assert!(GroupBy::from_arg("nonsense").is_err());
    }

    /// Table-driven coverage for every axis token + its aliases. One
    /// row per supported token; the body asserts the matching boolean
    /// or NameAxis variant on the parsed GroupBy. Replaces per-WI
    /// parser smoke tests that each re-tested a single token in
    /// isolation.
    #[test]
    fn parses_every_token() -> anyhow::Result<()> {
        type AxisCheck = fn(&GroupBy) -> bool;
        // (input, predicate on the parsed GroupBy)
        let cases: &[(&str, AxisCheck)] = &[
            ("short", |g| g.name == NameAxis::ShortName),
            ("shortname", |g| g.name == NameAxis::ShortName),
            ("short_name", |g| g.name == NameAxis::ShortName),
            ("demangled", |g| g.name == NameAxis::Demangled),
            ("demangled_name", |g| g.name == NameAxis::Demangled),
            ("variant", |g| g.name == NameAxis::Demangled),
            ("mangled", |g| g.name == NameAxis::Mangled),
            ("mangled_name", |g| g.name == NameAxis::Mangled),
            ("no-name", |g| g.name == NameAxis::None),
            ("noname", |g| g.name == NameAxis::None),
            ("none", |g| g.name == NameAxis::None),
            ("device", |g| g.device),
            ("dev", |g| g.device),
            ("stream", |g| g.stream),
            ("context", |g| g.context),
            ("ctx", |g| g.context),
            ("graph", |g| g.graph),
            ("graph_id", |g| g.graph),
            ("graph_node", |g| g.graph_node),
            ("graphnode", |g| g.graph_node),
            ("graph_node_id", |g| g.graph_node),
            ("node", |g| g.graph_node),
            ("nvtx-parent", |g| g.nvtx_parent),
            ("nvtx_parent", |g| g.nvtx_parent),
            ("nvtx-path", |g| g.nvtx_path),
            ("nvtx_path", |g| g.nvtx_path),
            ("grid_block", |g| g.grid_block),
            ("grid-block", |g| g.grid_block),
            ("gridblock", |g| g.grid_block),
        ];
        for (tok, pred) in cases {
            let g = GroupBy::from_arg(tok)?;
            assert!(pred(&g), "token `{tok}` did not set expected axis: {g:?}");
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct StatsRequest {
    /// Which kinds to aggregate. Resolved against [`ALLOWED_KINDS`]
    /// (kernel/memcpy/memset) at run time; `KindFilter::All` covers
    /// the GPU set and `KindFilter::Only(...)` picks a subset. `run()`
    /// defends with a `bail!` if a hand-built `Only(...)` contains a
    /// non-GPU kind.
    pub kinds: KindFilter,
    pub group_by: GroupBy,
    pub time_window: Option<TimeWindow>,
    /// When set, only aggregate over GPU events causally attributable
    /// to NVTX ranges whose name matches this glob (`*`/`?`).
    pub nvtx: Option<String>,
    /// Restrict to one CUDA device (NSys `deviceId`).
    pub device: Option<i32>,
    /// Restrict to one CUDA stream (NSys `streamId`).
    pub stream: Option<i64>,
    /// When `true`, each row gains a `histogram` array of per-bucket
    /// event counts using `HIST_BOUNDARIES_NS`. Response also surfaces
    /// the bucket schema once at the top level.
    pub hist: bool,
    /// Sort specification. `None` falls back to the default
    /// (`total` descending).
    pub sort: Option<SortSpec>,
    pub limit: usize,
    /// When `true`, stats `--type runtime` folds API versions
    /// (e.g. `cudaMalloc_v3020`, `cudaMalloc_v2000`, `cudaMalloc`)
    /// into one bucket by stripping the `_v<digits>` suffix before
    /// grouping. Matches the nsys recipe `cuda_api_sum`'s substr
    /// trick. No-op for non-Runtime kinds. Opt-in (default
    /// `false`) so the unversioned view stays the default.
    pub collapse_versioned: bool,
}

/// Half-decade duration boundaries (ns). 17 boundaries → 18 buckets
/// covering from sub-10 ns to multi-second event durations. Half a
/// decade per bucket gives enough resolution to distinguish
/// kernel populations without making the response huge.
pub const HIST_BOUNDARIES_NS: &[i64] = &[
    10,
    32,
    100,
    316,
    1_000,
    3_162,
    10_000,
    31_623,
    100_000,
    316_228,
    1_000_000,
    3_162_278,
    10_000_000,
    31_622_776,
    100_000_000,
    316_227_766,
    1_000_000_000,
];

impl Default for StatsRequest {
    fn default() -> Self {
        Self {
            kinds: KindFilter::All,
            group_by: GroupBy::default(),
            time_window: None,
            nvtx: None,
            device: None,
            stream: None,
            hist: false,
            sort: None,
            limit: 50,
            collapse_versioned: false,
        }
    }
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct StatsResponse {
    /// Number of groups returned in `rows` (after `--limit`).
    pub count: usize,
    /// Total number of distinct **groups** produced by the scope's
    /// GROUP BY *before* `--limit` clipped the list. When
    /// `total_matched > count`, some groups were dropped — raise
    /// `--limit` or narrow the filter to see them. Same envelope
    /// convention every verb uses; for stats specifically, `rows`
    /// are groups, so "rows matched" and "groups matched" coincide.
    pub total_matched: i64,
    /// Grand total *event-duration* across the whole filtered scope
    /// (type filter + time window applied, but NOT clipped by `--limit`).
    /// This is the denominator behind every row's `percentage`.
    pub total_duration_ns: i64,
    /// Grand **event count** summed across every group — distinct from
    /// [`Self::total_matched`], which counts *groups*, not events.
    /// Named explicitly (`total_events`, not `total_count`) so it
    /// stays unambiguous next to the envelope-convention
    /// `total_matched` at the wire-format level.
    pub total_events: i64,
    /// Resolved time window, if any (absolute ns).
    pub time_window_ns: Option<(i64, i64)>,
    /// NVTX scoping in effect (the user's pattern, if `--nvtx` was set).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_scope: Option<String>,
    /// Half-open bucket boundaries `[lo, hi)` in ns, present iff the
    /// caller set `--hist`. The last entry has `hi: null`. Each row's
    /// `histogram` array has the same length as this list and is
    /// indexed by bucket position.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub histogram_buckets_ns: Option<Vec<HistBucket>>,
    /// True when the caller asked for `--group-by mangled` but the
    /// trace's `CUPTI_ACTIVITY_KIND_KERNEL` table has no `mangledName`
    /// column (older NSys schema). The query silently downgrades to
    /// `--group-by demangled`; the same fallback is also surfaced via
    /// a `log::info!` line on stderr for human consumers.
    /// JSON-only agents read this flag
    /// instead of parsing stderr. Omitted when no fallback occurred.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub mangled_axis_fallback: bool,
    pub rows: Vec<StatRow>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct HistBucket {
    pub lo: i64,
    /// `None` for the open-ended tail bucket.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hi: Option<i64>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct StatRow {
    /// Cross-trace key. Pipe-separated composite identity built
    /// from `(kind, name?, device_id?, stream_id?, context_id?,
    /// graph_id?, graph_node_id?, nvtx_style?, nvtx_parent?,
    /// nvtx_path?, grid_block?)` — exactly the fields `--group-by` activated. Two
    /// `stats` runs with the same `--group-by` produce matching keys
    /// for matching aggregation rows.
    pub key: String,
    /// The primary group key in the name axis — shortName, demangled
    /// signature, memcpy direction label, or memset label. Omitted in
    /// JSON when `--group-by` has `no-name`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(rename = "type")]
    pub kind: &'static str,
    /// Kernel shortName. Populated on every kernel row whose name axis
    /// is `short` or `demangled` (under `short` it equals `name`; under
    /// `demangled` it lets agents roll variants back to their shortName
    /// group). `None` for non-kernel kinds (memcpy/memset/sync/graph/
    /// nvtx) and when the name axis is `no-name`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub short_name: Option<String>,
    /// Physical-dimension columns. Each is populated only when the
    /// corresponding axis is part of `--group-by`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_id: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_id: Option<i64>,
    /// Captured-graph id. Populated only when `--group-by graph` is
    /// active *and* the kernel/memcpy/memset row has `graphId` set
    /// (i.e. ran inside a CUDA graph in a `=node` capture). `None`
    /// otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_id: Option<i64>,
    /// Per-node id. Populated only when `--group-by graph_node` is
    /// active.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_node_id: Option<i64>,
    pub count: i64,
    pub total_ns: i64,
    pub avg_ns: i64,
    pub min_ns: i64,
    pub max_ns: i64,
    pub p50_ns: i64,
    pub p95_ns: i64,
    pub p99_ns: i64,
    /// Total bytes transferred — only populated for memcpy/memset rows.
    /// `None` for kernel rows (no `bytes` column).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes_total: Option<i64>,
    /// Effective bandwidth in decimal GB/s (10^9 bytes/sec). Computed
    /// as `bytes_total / total_ns`. Same population rule as `bytes_total`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gbps: Option<f64>,
    pub percentage: f64,
    /// Per-bucket event counts, indexed by `histogram_buckets_ns`
    /// position on the response. Present iff `--hist` was set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub histogram: Option<Vec<i64>>,
    /// NVTX-only raw eventType value from `NVTX_EVENTS.eventType`.
    /// Mirrors `NvtxDetails.event_type` at inspect/host_api.rs. `None`
    /// on non-NVTX rows. Within a group, multiple raw values can fold
    /// into one bucket (e.g. 59 and 70 both produce
    /// `nvtx_style = "push_pop"`); the surfaced value is the minimum
    /// raw eventType seen in that bucket.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_type: Option<i64>,
    /// Derived label for NVTX `eventType`:
    /// `{59,70}` → `"push_pop"`,
    /// `{60,71}` → `"start_end"`,
    /// anything else (NVTX_PAYLOAD_*, instrumentation, future ints) →
    /// `"unknown"`. `None` on non-NVTX rows. Participates in the
    /// composite group key on NVTX rows so PushPop and StartEnd ranges
    /// with the same name split into distinct buckets — mirrors nsys
    /// `nvtx_sum`'s `GROUP BY tag, style`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_style: Option<&'static str>,
    /// Composite key for the innermost enclosing NVTX range — only
    /// populated when `--group-by nvtx-parent` is active.
    /// `"nvtx:<rowid>"` for events that fall inside a real range,
    /// `"nvtx:none"` for events outside every range. Lets agents
    /// `INDEX(.rows; .nvtx_parent_key)` across traces without a name
    /// collision.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_parent_key: Option<String>,
    /// Innermost enclosing NVTX range name (or the visible sentinel
    /// `"__no_nvtx__"` for events outside every range). Populated only
    /// when `--group-by nvtx-parent` is active.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_parent_name: Option<String>,
    /// Nesting depth of the innermost enclosing NVTX range — 0 for
    /// outermost ranges, 1 for ranges fully inside a single outer
    /// range, etc. Populated only when `--group-by nvtx-parent` is
    /// active AND the event attributes to a real range; left `None`
    /// for the no-NVTX sentinel so depth-0 doesn't collide with real
    /// outermost ranges.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_parent_depth: Option<u8>,
    /// Composite key for the full NVTX path — only populated when
    /// `--group-by nvtx-path` is active. `"nvtx-path:<path>"` for
    /// events that fall inside a real path, `"nvtx-path:none"` for
    /// events outside every range.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_path_key: Option<String>,
    /// Full slash-joined path of the innermost enclosing NVTX range,
    /// or the visible no-NVTX sentinel. Populated only when
    /// `--group-by nvtx-path` is active.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvtx_path: Option<String>,
    /// Resolved NVTX domain identity of the enclosing range — the
    /// process-local handle `domainId`. Populated only
    /// when `--group-by nvtx-path` is active AND the row attributes to
    /// a real range; `None` for the no-NVTX sentinel (which has no
    /// enclosing range and therefore no domain).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain_id: Option<i64>,
    /// Owning process id of the enclosing range's domain, decoded
    /// `(global_tid >> 24) & 0xFFFFFF`. Pairs with
    /// `domain_id` to form the domain's true identity. `None` for the
    /// no-NVTX sentinel.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain_pid: Option<i64>,
    /// Resolved domain name when the `(pid, domain_id)` domain was
    /// registered via an `NvtxDomainCreate` event.
    /// Best-effort: `None` when unregistered (incl. the default domain
    /// id 0) or for the no-NVTX sentinel.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain_name: Option<String>,
    /// Kernel launch grid X dimension. Populated only when
    /// `--group-by grid_block` is active. Mirrors `EventRefKernel.grid`
    /// component 0 from the inspect/search surfaces.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grid_x: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grid_y: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grid_z: Option<i64>,
    /// Kernel launch block X dimension. Populated only when
    /// `--group-by grid_block` is active.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_x: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_y: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_z: Option<i64>,
}

pub fn run<P: AsRef<Path>>(path: P, req: StatsRequest) -> Result<StatsResponse> {
    let (trace, abs_window) = crate::open_scoped(path.as_ref(), req.limit, req.time_window)?;

    // Defence-in-depth: hand-built `StatsRequest`s with non-stats
    // kinds get rejected here too. CLI callers go through
    // `GpuFilters::kinds` with the `ALLOWED_KINDS` allow-list and
    // never reach this branch.
    if let KindFilter::Only(v) = &req.kinds {
        for k in v {
            if !ALLOWED_KINDS.contains(k) {
                anyhow::bail!(
                    "stats only aggregates duration-bearing kinds \
                     (kernel/memcpy/memset/sync/graph/nvtx); got `{}`",
                    k.as_str()
                );
            }
        }
    }

    // Shared `--device` / `--stream` policy: explicit null-location
    // kinds (Runtime/Osrt/Nvtx/GraphNode/GraphEvent/Overhead/
    // CpuSample) error rather than silently dropping when a location
    // filter is set. `KindFilter::All` continues to narrow implicitly
    // (today's "default just works" behaviour).
    crate::kind_policy::validate_location_filter(
        &req.kinds,
        crate::kind_policy::LocationFilter {
            device: req.device,
            stream: req.stream,
        },
        "stats",
    )?;

    // Shared `--nvtx` policy: explicit non-attributable kinds error
    // with a redirecting message. `resolve_nvtx_kinds` below repeats
    // this validation as part of its pipeline, so this early call is
    // strictly for *error precedence* — a request like `--nvtx p
    // --type osrt --group-by device` should land on the "--nvtx can't
    // scope --type osrt" message rather than the group-by-axis error
    // emitted by the location/grid_block/nvtx-parent checks below.
    crate::kind_policy::validate_nvtx_filter(&req.kinds, req.nvtx.as_deref(), "stats")?;

    // Set-level `--group-by location-axis` reject. The rule fires when
    // every kind in the explicit set is CPU-only — so
    // `--type runtime --group-by device` and `--type runtime,osrt
    // --group-by device` both error, while `--type kernel,runtime
    // --group-by device` is positive (the kernel rows fill the
    // device buckets, runtime rows land in a single null-device
    // bucket per group key — agents that don't want that can drop
    // runtime from the type set explicitly).
    let group_by_location_axis = req.group_by.device
        || req.group_by.context
        || req.group_by.stream
        || req.group_by.graph
        || req.group_by.graph_node;
    if group_by_location_axis
        && let KindFilter::Only(explicit) = &req.kinds
        && !explicit.is_empty()
        && explicit.iter().all(|k| !k.is_location_bearing())
    {
        let csv = explicit
            .iter()
            .map(|k| k.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        anyhow::bail!(
            "stats: --group-by device/context/stream/graph/graph_node \
             has no meaning for `--type {csv}` — every kind in the \
             set is CPU-side and carries no device/stream/graph \
             columns. Group by name (the default) or mix in a \
             GPU-side kind (kernel/memcpy/memset/sync/graph/\
             cuda_event) to split GPU rows by device while keeping \
             the CPU-side rows in their own null bucket."
        );
    }

    // Policy: --group-by grid_block is kernel-only. The
    // CUPTI gridX/blockX columns live only on
    // CUPTI_ACTIVITY_KIND_KERNEL — projecting NULL for other kinds
    // would produce a single misleading null bucket. KindFilter::All
    // narrows implicitly to kernel at SQL time (other kinds drop out
    // via table_exists). Explicit non-kernel kinds error up-front.
    if req.group_by.grid_block
        && let KindFilter::Only(explicit) = &req.kinds
        && let Some(other) = explicit.iter().find(|k| !matches!(k, EventKind::Kernel))
    {
        anyhow::bail!(
            "stats: --group-by grid_block is kernel-only — \
             gridX/Y/Z and blockX/Y/Z columns live on \
             CUPTI_ACTIVITY_KIND_KERNEL and nowhere else. Got \
             `--type {}` in the explicit kind set; either drop the \
             non-kernel kinds or unset --group-by grid_block.",
            other.as_str()
        );
    }

    // Policy: nvtx-parent and nvtx-path are
    // mutually exclusive with the graph/graph_node axes (different
    // attribution model — NVTX is host-thread containment; CUDA-graph
    // captures are device-side state) and with --type nvtx
    // (self-attribute).
    let group_by_nvtx_hierarchy = req.group_by.nvtx_parent || req.group_by.nvtx_path;
    if group_by_nvtx_hierarchy {
        if req.group_by.nvtx_parent && req.group_by.nvtx_path {
            anyhow::bail!(
                "stats: --group-by nvtx-parent and nvtx-path are mutually exclusive — \
                 pick rowid-level parent buckets or path-level hierarchy buckets"
            );
        }
        let axis_name = if req.group_by.nvtx_path {
            "nvtx-path"
        } else {
            "nvtx-parent"
        };
        if req.group_by.graph || req.group_by.graph_node {
            anyhow::bail!(
                "stats: --group-by {axis_name} is mutually exclusive with \
                 graph/graph_node — NVTX attribution walks host-thread \
                 containment, captured-graph axes walk device-side state. \
                 Pick one model per query."
            );
        }
        if let KindFilter::Only(explicit) = &req.kinds
            && explicit.iter().any(|k| matches!(k, EventKind::Nvtx))
        {
            anyhow::bail!(
                "stats: --group-by {axis_name} + --type nvtx is a \
                 self-attribute tautology — NVTX rows ARE the ranges \
                 NVTX hierarchy axes attribute other kinds to. Drop one of the \
                 two flags."
            );
        }
        // NVTX prereq tables: matching the --nvtx filter's contract.
        // `NVTX_EVENTS` and `CUPTI_ACTIVITY_KIND_RUNTIME` are
        // unconditional; without them the sidecar build can't
        // compute anything.
        for t in ["NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_RUNTIME"] {
            if !trace.table_exists(t) {
                anyhow::bail!(
                    "--group-by {axis_name} requires `{t}`, which is not \
                     present in this trace"
                );
            }
        }
        // `TARGET_INFO_CUDA_CONTEXT_INFO` is required when a
        // GPU-side kind is *actually present in this trace and
        // requested*. Pure `--type runtime --group-by nvtx-parent`
        // doesn't need the bridge (it joins on `rt_rowid`); ditto
        // for `KindFilter::All` against a runtime-only trace where
        // the GPU activity tables don't exist.
        //
        // The check resolves the request against `ALLOWED_KINDS` and
        // probes table existence, so `KindFilter::All` against a
        // trace with no GPU activity tables collapses to runtime-
        // only and skips the bridge requirement — matching the SQL
        // path's actual behavior (it would emit only the runtime
        // subquery).
        let resolved = req.kinds.resolve(&ALLOWED_KINDS);
        let needs_ctx_bridge = resolved.iter().any(|k| {
            matches!(
                k,
                EventKind::Kernel | EventKind::Memcpy | EventKind::Memset | EventKind::Sync
            ) && trace.table_exists(match k {
                EventKind::Kernel => "CUPTI_ACTIVITY_KIND_KERNEL",
                EventKind::Memcpy => "CUPTI_ACTIVITY_KIND_MEMCPY",
                EventKind::Memset => "CUPTI_ACTIVITY_KIND_MEMSET",
                EventKind::Sync => "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
                _ => "",
            })
        });
        if needs_ctx_bridge && !trace.table_exists("TARGET_INFO_CUDA_CONTEXT_INFO") {
            anyhow::bail!(
                "--group-by {axis_name} on kernel/memcpy/memset/sync requires \
                 `TARGET_INFO_CUDA_CONTEXT_INFO`, which is not present in this trace \
                 (GPU activity rows cannot be bridged to runtime rows without the \
                 context-info table; the lookup would silently miss every kernel)"
            );
        }
    }

    // Resolve `--type` + `--nvtx` via the shared helper: validates
    // explicit non-attributable kinds against `--nvtx`, resolves
    // against `ALLOWED_KINDS`, filters by table presence, and
    // implicitly narrows to the attributable set when `--nvtx` is
    // active (the contract that previously diverged across stats /
    // search / timeline).
    //
    // An additional narrowing layers on top: when
    // `--group-by grid_block` is active, narrow to kernel rows even
    // on the implicit KindFilter::All path. Other kinds would
    // project NULL grid/block and pollute the single null bucket.
    // Explicit non-kernel kinds are rejected up-front above.
    let kinds: Vec<EventKind> = crate::kind_policy::resolve_nvtx_kinds(
        &req.kinds,
        req.nvtx.as_deref(),
        &ALLOWED_KINDS,
        &trace,
        "stats",
    )?
    .into_iter()
    .filter(|k| !req.group_by.grid_block || matches!(k, EventKind::Kernel))
    .collect();

    let histogram_buckets_ns = if req.hist {
        Some(build_bucket_schema())
    } else {
        None
    };

    if kinds.is_empty() {
        return Ok(StatsResponse {
            count: 0,
            total_matched: 0,
            total_duration_ns: 0,
            total_events: 0,
            time_window_ns: abs_window,
            nvtx_scope: req.nvtx.clone(),
            histogram_buckets_ns,
            mangled_axis_fallback: false,
            rows: Vec::new(),
        });
    }

    let attribution = match req.nvtx.as_deref() {
        Some(p) => Some(crate::nvtx_attribution::build(p, &kinds, &trace)?),
        None => None,
    };

    // Probe schema once so optional columns (currently only
    // `mangledName` on the kernel table) can resolve to
    // a real ref or NULL without a per-kind reprobe inside
    // `per_kind_subquery`. The probe is cheap (one
    // information_schema query) and the result is also consulted to
    // pick the effective name axis when `--group-by mangled` would
    // otherwise hit an absent column.
    let columns = crate::column_map::load_columns(trace.conn(), &["CUPTI_ACTIVITY_KIND_KERNEL"])?;
    let axis_resolution = resolve_name_axis(req.group_by.name, &columns);
    let effective_group_by = GroupBy {
        name: axis_resolution.effective,
        ..req.group_by
    };

    // Each subquery carries its own parameter list so positional binds
    // can't drift across the UNION (see `per_kind_subquery`).
    let nvtx_scope = if attribution.is_some() {
        crate::nvtx_attribution::NvtxScope::Attributed
    } else {
        crate::nvtx_attribution::NvtxScope::None
    };

    // When an NVTX hierarchy group-by is active, ensure the
    // trace's per-runtime NVTX-parent sidecar is built (cold) or
    // fresh (warm). The sidecar lives in `veloq-nsys-data` (path
    // `<trace>.veloq/nvtx-parent.parquet`) and is shared across every
    // NVTX-bearing verb — building it once amortises across every
    // future `stats --group-by nvtx-parent` / `search --with-nvtx`
    // / `inspect <kind>:N` call on the same trace. Per-thread sorted
    // NVTX + binary-search-and-walk-back gives an
    // `O(N_runtime × log N_nvtx + matches)` build cost.
    let nvtx_parent_sidecar: Option<std::path::PathBuf> = if group_by_nvtx_hierarchy {
        let path = veloq_nsys_data::runtime_nvtx_parent::ensure_sidecar(&trace)
            .context("building NVTX-parent attribution sidecar")?;
        Some(path)
    } else {
        None
    };
    // Resolve the `(pid, domainId) -> name` map once when the path axis
    // is active, so hydration can attach a resolved domain name to each
    // nvtx-path row. Names are
    // best-effort: if the resolver errors (e.g. a partial trace), degrade
    // to an empty map — domain *identity* still works, only the human
    // name is missing. Never fail the verb over a name lookup.
    let domain_names: std::collections::HashMap<(i64, i64), String> = if req.group_by.nvtx_path {
        veloq_nsys_data::nvtx_tree::ensure_sidecar(&trace)
            .context("building NVTX tree sidecar for stats --group-by nvtx-path")?;
        veloq_nsys_data::trace_map::nvtx_domain_names(&trace).unwrap_or_default()
    } else {
        std::collections::HashMap::new()
    };

    let mut subqueries: Vec<String> = Vec::with_capacity(kinds.len());
    let mut per_kind_params: Vec<Value> = Vec::new();
    for kind in &kinds {
        let (sql, params) = per_kind_subquery(
            *kind,
            abs_window,
            nvtx_scope,
            req.collapse_versioned,
            &columns,
            nvtx_parent_sidecar.as_deref(),
            req.group_by.nvtx_path,
        )?;
        subqueries.push(sql);
        per_kind_params.extend(params);
    }
    let union = subqueries.join(" UNION ALL ");

    let group_by_sql = GroupBySql::for_axes(&effective_group_by);

    // When `--nvtx` is set, prepend the attribution CTE so the per-kind
    // subqueries can filter via `rowid IN attributed_<kind>_rowids`.
    let attribution_prefix = match &attribution {
        Some(att) => format!("{},", att.body),
        None => String::new(),
    };

    // Sort: default `total` descending preserves the original behaviour
    // exactly; user-supplied multi-field specs override.
    let sort_spec = req
        .sort
        .clone()
        .unwrap_or_else(|| SortSpec::single("total"));
    let order_by = stats_sort_sql(&sort_spec)?;

    let hist_sql = HistSql::build(req.hist);
    let (hist_grouped_cols, hist_outer_cols) =
        (hist_sql.grouped_cols.as_str(), hist_sql.outer_cols.as_str());
    let GroupBySql {
        name_select,
        short_name_select,
        device_select,
        context_select,
        stream_select,
        graph_select,
        graph_node_select,
        nvtx_parent_rowid_select,
        nvtx_parent_name_select,
        nvtx_path_select,
        nvtx_domain_id_select,
        nvtx_domain_pid_select,
        grid_x_select,
        grid_y_select,
        grid_z_select,
        block_x_select,
        block_y_select,
        block_z_select,
        group_keys_sql,
    } = &group_by_sql;

    // Window functions over the aggregated rows give us the *scope-wide*
    // totals (sum/count across all groups, not just the LIMITed slice).
    // Same single query, no extra round-trip.
    // Optional `--device` / `--stream` filters: each adds a positive
    // predicate to the pre-grouping WHERE so deviceId / streamId are
    // matched against bind parameters. Bind order is appended after
    // the per-kind windowed params (handled below).
    let mut location_where = String::new();
    let mut location_params: Vec<Value> = Vec::new();
    crate::kind_policy::LocationFilter {
        device: req.device,
        stream: req.stream,
    }
    .append_where(&mut location_where, &mut location_params);

    let sql = format!(
        r#"
        WITH {attribution_prefix} events AS ({union}),
        grouped AS (
            SELECT
                {name_select},
                {short_name_select},
                kind,
                {device_select},
                {context_select},
                {stream_select},
                {graph_select},
                {graph_node_select},
                {nvtx_parent_rowid_select},
                {nvtx_parent_name_select},
                {nvtx_path_select},
                {nvtx_domain_id_select},
                {nvtx_domain_pid_select},
                {grid_x_select},
                {grid_y_select},
                {grid_z_select},
                {block_x_select},
                {block_y_select},
                {block_z_select},
                -- nvtx_style is a derived label folding
                -- raw eventType ints into push_pop/start_end/unknown.
                -- Participates in GROUP BY for --type nvtx (NULL on
                -- non-NVTX kinds collapses into one bucket, leaving
                -- GPU group counts unchanged). event_type is the raw
                -- min-value within the bucket so agents can drill
                -- back into NSys's enum.
                {NVTX_STYLE_EXPR}                              AS nvtx_style,
                MIN(event_type)                                AS event_type,
                COUNT(*)                                       AS count,
                CAST(SUM(duration) AS BIGINT)                  AS total_ns,
                CAST(AVG(duration) AS BIGINT)                  AS avg_ns,
                MIN(duration)                                  AS min_ns,
                MAX(duration)                                  AS max_ns,
                CAST(quantile_disc(duration, 0.50) AS BIGINT)  AS p50_ns,
                CAST(quantile_disc(duration, 0.95) AS BIGINT)  AS p95_ns,
                CAST(quantile_disc(duration, 0.99) AS BIGINT)  AS p99_ns,
                CAST(SUM(bytes)    AS BIGINT)                  AS bytes_total
                {hist_grouped_cols}
            FROM events
            WHERE duration > 0 {location_where}
            GROUP BY {group_keys_sql}
        )
        SELECT
            name, short_name, kind,
            device_id, context_id, stream_id,
            graph_id, graph_node_id,
            nvtx_parent_rowid, nvtx_parent_name, nvtx_path,
            nvtx_domain_id, nvtx_domain_pid,
            grid_x, grid_y, grid_z, block_x, block_y, block_z,
            nvtx_style, event_type,
            count,
            total_ns, avg_ns, min_ns, max_ns, p50_ns, p95_ns, p99_ns,
            bytes_total,
            -- bytes / (ns × 1e-9) = bytes × 1e9 / ns. Decimal GB (10^9
            -- bytes), matching how PCIe / NVLink specs report bandwidth.
            CASE WHEN bytes_total IS NULL OR total_ns <= 0 THEN NULL
                 ELSE CAST(bytes_total AS DOUBLE) * 1e-9
                      / (CAST(total_ns AS DOUBLE) * 1e-9)
            END AS gbps,
            CAST(SUM(total_ns) OVER () AS BIGINT) AS scope_total_ns,
            CAST(SUM(count)    OVER () AS BIGINT) AS scope_total_count,
            CAST(COUNT(*)      OVER () AS BIGINT) AS scope_total_groups
            {hist_outer_cols}
        FROM grouped
        ORDER BY {order_by}
        LIMIT ?
        "#
    );

    // Bind order matches SQL position:
    //   1. attribution CTE params (one for the pattern glob), if `--nvtx`
    //   2. per-kind windowed params (carried alongside the SQL fragments
    //      by `per_kind_subquery`, so the bind can't drift)
    //   3. location_where params (--device / --stream)
    //   4. LIMIT param
    let mut params: Vec<Value> = Vec::new();
    if let Some(att) = &attribution {
        params.extend(att.params.iter().cloned());
    }
    params.extend(per_kind_params);
    params.extend(location_params);
    params.push(Value::BigInt(req.limit as i64));

    // When --group-by nvtx-parent is active, compute
    // the per-rowid NVTX nesting once so depth resolution during
    // hydration is a HashMap::get. Skipped on the no-axis path so we
    // don't pay for the cache build on the common case.
    let nvtx_nesting = if req.group_by.nvtx_parent && trace.table_exists("NVTX_EVENTS") {
        Some(trace.nvtx_nesting()?)
    } else {
        None
    };
    let (mut out, scope) = hydrate_stats_rows(
        &trace,
        &sql,
        &params,
        req.hist,
        nvtx_nesting.as_ref(),
        &domain_names,
    )?;

    if scope.total_ns > 0 {
        for r in &mut out {
            r.percentage = (r.total_ns as f64 / scope.total_ns as f64) * 100.0;
        }
    }

    Ok(StatsResponse {
        count: out.len(),
        total_matched: scope.total_groups,
        total_duration_ns: scope.total_ns,
        total_events: scope.total_count,
        time_window_ns: abs_window,
        nvtx_scope: req.nvtx.clone(),
        histogram_buckets_ns,
        mangled_axis_fallback: axis_resolution.fell_back,
        rows: out,
    })
}

/// Scope-wide window-function totals, repeated on every row by the
/// outer SELECT's `SUM(...) OVER ()` columns. Read once on the first
/// row and reused for the rest.
struct StatsScope {
    total_ns: i64,
    total_count: i64,
    total_groups: i64,
}

/// Run the prepared `sql` against `trace`, project each row into a
/// `StatRow`, and recover the scope-wide totals exposed by the window
/// functions in the outer SELECT. Carved out of `run` so the SELECT
/// alias → struct field hydration is reviewable in isolation; bind
/// order and SQL assembly stay in the caller.
fn hydrate_stats_rows(
    trace: &Trace,
    sql: &str,
    params: &[Value],
    hist: bool,
    nvtx_nesting: Option<&veloq_nsys_data::NvtxNesting>,
    domain_names: &std::collections::HashMap<(i64, i64), String>,
) -> Result<(Vec<StatRow>, StatsScope)> {
    let mut stmt = trace
        .conn()
        .prepare(sql)
        .context("failed to prepare stats SQL")?;
    let params_ref = crate::bind(params);
    let mut rows = stmt.query(params_ref.as_slice())?;

    let mut out: Vec<StatRow> = Vec::new();
    let mut scope = StatsScope {
        total_ns: 0,
        total_count: 0,
        total_groups: 0,
    };
    let mut scope_read = false;
    while let Some(row) = rows.next()? {
        // Read by SELECT alias rather than by integer index so a future
        // column reorder/insert in the outer SELECT can't silently
        // shift every value. Aliases match the column expressions
        // built in `run`.
        let name: Option<String> = row.get("name")?;
        let short_name_raw: Option<String> = row.get("short_name")?;
        let kind: String = row.get("kind")?;
        let device_id: Option<i32> = row.get("device_id")?;
        let context_id: Option<i64> = row.get("context_id")?;
        let stream_id: Option<i64> = row.get("stream_id")?;
        let graph_id: Option<i64> = row.get("graph_id")?;
        let graph_node_id: Option<i64> = row.get("graph_node_id")?;
        let count: i64 = row.get("count")?;
        let total_ns: i64 = row.get("total_ns")?;
        let avg_ns: i64 = row.get("avg_ns")?;
        let min_ns: i64 = row.get("min_ns")?;
        let max_ns: i64 = row.get("max_ns")?;
        let p50_ns: i64 = row.get("p50_ns")?;
        let p95_ns: i64 = row.get("p95_ns")?;
        let p99_ns: i64 = row.get("p99_ns")?;
        let bytes_total: Option<i64> = row.get("bytes_total")?;
        let gbps: Option<f64> = row.get("gbps")?;
        let event_type: Option<i64> = row.get("event_type")?;
        let nvtx_style_raw: Option<String> = row.get("nvtx_style")?;
        let nvtx_style: Option<&'static str> = nvtx_style_raw.as_deref().map(nvtx_style_label);
        let nvtx_parent_rowid: Option<i64> = row.get("nvtx_parent_rowid")?;
        let nvtx_parent_name_raw: Option<String> = row.get("nvtx_parent_name")?;
        let nvtx_path_raw: Option<String> = row.get("nvtx_path")?;
        let nvtx_domain_id_raw: Option<i64> = row.get("nvtx_domain_id")?;
        let nvtx_domain_pid_raw: Option<i64> = row.get("nvtx_domain_pid")?;
        let grid_x: Option<i64> = row.get("grid_x")?;
        let grid_y: Option<i64> = row.get("grid_y")?;
        let grid_z: Option<i64> = row.get("grid_z")?;
        let block_x: Option<i64> = row.get("block_x")?;
        let block_y: Option<i64> = row.get("block_y")?;
        let block_z: Option<i64> = row.get("block_z")?;
        if !scope_read {
            scope.total_ns = row.get("scope_total_ns")?;
            scope.total_count = row.get("scope_total_count")?;
            scope.total_groups = row.get("scope_total_groups")?;
            scope_read = true;
        }

        // Histogram columns (when --hist) carry the `hist_bN` aliases
        // emitted by `HistSql::build`. Reading by name keeps the
        // hydration in sync with the SELECT.
        let histogram = if hist {
            let n = HIST_BOUNDARIES_NS.len() + 1; // +1 tail
            let mut buckets = Vec::with_capacity(n);
            for i in 0..n {
                buckets.push(row.get::<_, i64>(format!("hist_b{i}").as_str())?);
            }
            Some(buckets)
        } else {
            None
        };

        // Recover the typed EventKind from the SQL-side label so we can
        // hand back a stable `&'static str` without an open-coded
        // string→string dispatch table.
        let kind_static: &'static str = EventKind::parse(&kind)
            .map(EventKind::as_str)
            .unwrap_or("unknown");

        // Always populate `short_name` for kernel rows so the schema is
        // stable across `--group-by` modes and agents can roll demangled
        // rows back to their shortName group. For memcpy/memset it's
        // redundant with `name`, so omit it.
        let short_name = if kind_static == "kernel" {
            short_name_raw
        } else {
            None
        };

        // Row key: `(kind, name?, dev?, stream?, ctx?, graph?,
        // graph_node?, style?, nvtx?, grid?, block?)` pipe-joined.
        // Only fields populated by the active `--group-by` contribute
        // — that way two `stats` runs with the same flags produce
        // matching keys for matching rows, and runs with different
        // `--group-by` won't accidentally collide on a shorter key.
        let mut key_parts = vec![kind_static.to_string()];
        if let Some(n) = name.as_deref() {
            key_parts.push(n.to_string());
        }
        if let Some(d) = device_id {
            key_parts.push(format!("dev:{d}"));
        }
        if let Some(s) = stream_id {
            key_parts.push(format!("stream:{s}"));
        }
        if let Some(c) = context_id {
            key_parts.push(format!("ctx:{c}"));
        }
        if let Some(g) = graph_id {
            key_parts.push(format!("graph:{g}"));
        }
        if let Some(gn) = graph_node_id {
            key_parts.push(format!("graph_node:{gn}"));
        }
        // Include `nvtx_style` in the composite key so two
        // NVTX ranges sharing a name but differing on PushPop vs
        // StartEnd produce distinct keys cross-trace. Skipped for
        // non-NVTX rows (nvtx_style is None there), keeping the GPU
        // key shape backward-compatible.
        if let Some(style) = nvtx_style {
            key_parts.push(format!("style:{style}"));
        }
        // Include `nvtx_parent_key` in the composite key
        // so events rolled up under the same NVTX range produce
        // matching keys cross-trace. `nvtx:none` is the sentinel for
        // events outside every range; non-axis rows (nvtx_parent_name
        // is None) don't contribute to the key, preserving back-compat.
        let nvtx_parent_key: Option<String> = nvtx_parent_name_raw.as_deref().map(|_| {
            nvtx_parent_rowid
                .map(|rid| format!("nvtx:{rid}"))
                .unwrap_or_else(|| crate::nvtx_parent::NO_NVTX_KEY.to_string())
        });
        if let Some(npk) = nvtx_parent_key.as_deref() {
            key_parts.push(npk.to_string());
        }
        let nvtx_path_key: Option<String> = nvtx_path_raw.as_deref().map(|path| {
            if path == crate::nvtx_parent::NO_NVTX_NAME {
                crate::nvtx_parent::NO_NVTX_PATH_KEY.to_string()
            } else {
                format!("nvtx-path:{path}")
            }
        });
        if let Some(npk) = nvtx_path_key.as_deref() {
            key_parts.push(npk.to_string());
        }
        // Domain identity is part of the nvtx-path axis: two same-name / same-parent ranges
        // in distinct (pid, domainId) domains must produce distinct rows.
        // Only real nvtx-path rows carry it — the no-NVTX sentinel has no
        // enclosing range and MUST NOT get a domain-identity component,
        // so identity fields stay None and no key part is added. The
        // domain key goes RIGHT AFTER the nvtx-path component so key
        // ordering is stable.
        let is_real_nvtx_path_row = nvtx_path_raw
            .as_deref()
            .is_some_and(|p| p != crate::nvtx_parent::NO_NVTX_NAME);
        let (domain_id, domain_pid, domain_name) = match (
            is_real_nvtx_path_row,
            nvtx_domain_id_raw,
            nvtx_domain_pid_raw,
        ) {
            (true, Some(did), Some(pid)) => {
                key_parts.push(format!("domain:{pid}:{did}"));
                let name = domain_names.get(&(pid, did)).cloned();
                (Some(did), Some(pid), name)
            }
            _ => (None, None, None),
        };
        // Launch-shape key segment so two kernels with
        // different (grid, block) shapes produce distinct keys
        // cross-trace. Only contributes when the axis is active (all
        // six values populated together); empty / NULL drops the
        // segment, preserving back-compat for non-axis rows.
        if let (Some(gx), Some(gy), Some(gz), Some(bx), Some(by), Some(bz)) =
            (grid_x, grid_y, grid_z, block_x, block_y, block_z)
        {
            key_parts.push(format!("grid:{gx}x{gy}x{gz}"));
            key_parts.push(format!("block:{bx}x{by}x{bz}"));
        }
        let key = key_parts.join("|");

        // Depth comes from the per-trace `nvtx_nesting` map computed
        // once for the request; only populated when the row attributes
        // to a real range (sentinel stays at None so depth=0 doesn't
        // collide with real outermost ranges).
        let nvtx_parent_depth: Option<u8> = match (nvtx_parent_rowid, nvtx_nesting) {
            (Some(rid), Some(map)) => map.get(&rid).map(|e| e.depth),
            _ => None,
        };

        out.push(StatRow {
            key,
            name,
            kind: kind_static,
            short_name,
            device_id,
            context_id,
            stream_id,
            graph_id,
            graph_node_id,
            count,
            total_ns,
            avg_ns,
            min_ns,
            max_ns,
            p50_ns,
            p95_ns,
            p99_ns,
            bytes_total,
            gbps,
            percentage: 0.0,
            histogram,
            event_type,
            nvtx_style,
            nvtx_parent_key,
            nvtx_parent_name: nvtx_parent_name_raw,
            nvtx_parent_depth,
            nvtx_path_key,
            nvtx_path: nvtx_path_raw,
            domain_id,
            domain_pid,
            domain_name,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
        });
    }
    Ok((out, scope))
}

fn build_bucket_schema() -> Vec<HistBucket> {
    let mut buckets = Vec::with_capacity(HIST_BOUNDARIES_NS.len() + 1);
    let mut prev: i64 = 0;
    for &b in HIST_BOUNDARIES_NS {
        buckets.push(HistBucket {
            lo: prev,
            hi: Some(b),
        });
        prev = b;
    }
    // Open-ended tail bucket
    buckets.push(HistBucket { lo: prev, hi: None });
    buckets
}

/// SQL fragment selecting (display_name, short_name, kind, duration) for one event table.
///
/// `display_name` is the leaf identity used by `--group-by demangled`
/// (demangled for kernels, label for memcpy/memset). `short_name` is
/// always the shortName for kernels, and identical to display_name for
/// memcpy/memset.
///
/// When `windowed` is true, four positional `?` parameters must be
/// bound in this order: `end, start, end, start`.
///
/// When `nvtx_scope` is `Attributed`, the WHERE clause includes a
/// rowid-IN filter against `attributed_<kind>_rowids` (a CTE that must
/// already be in scope). No additional params are bound.
/// Build the per-kind subquery body **and** its bind parameters as a
/// pair, so the caller can't get out of sync with positional `?`s.
/// Returning the body and params together prevents a placeholder added
/// here from silently misaligning every kind's bind slots across the
/// surrounding UNION ALL.
fn per_kind_subquery(
    kind: EventKind,
    abs_window: Option<(i64, i64)>,
    nvtx_scope: crate::nvtx_attribution::NvtxScope,
    collapse_versioned: bool,
    columns: &crate::column_map::ColumnMap,
    nvtx_parent_sidecar: Option<&std::path::Path>,
    include_nvtx_path: bool,
) -> Result<(String, Vec<Value>)> {
    // ALLOWED_KINDS is enforced upstream — Runtime/Osrt/GraphNode/
    // GraphEvent/CudaEvent/Overhead/CpuSample never reach here in
    // practice. The workspace's no-panic policy routes the invariant
    // violation through `Result` instead of an `unreachable!`.
    //
    // For Sync, Graph, and Nvtx, `attributed_view` is `None`: NVTX
    // attribution is defined as GPU work *causally* attributable to
    // a host-thread NVTX range. Sync events sit on the runtime API
    // side of that walk; graph_trace rows roll up work that may have
    // been captured outside any current NVTX scope; NVTX rows ARE
    // the attribution source and can't attribute to themselves.
    // With `--nvtx`, all three kinds are excluded entirely (see the
    // FALSE clause below).
    //
    // Per-kind dispatch:
    //   - attributed_view: which `attributed_<kind>_rowids` CTE
    //     filters this kind when `--nvtx` is set (None → kind is
    //     NVTX-opaque or, for Nvtx itself, redundant)
    //   - bytes_expr: SQL fragment for the bandwidth column
    //   - graph_id_expr / graph_node_id_expr: columns to project for
    //     `--group-by graph` / `--group-by graph_node`. Kernel rows
    //     carry both; memcpy/memset only graphNodeId; sync/graph/nvtx
    //     have neither.
    let (attributed_view, bytes_expr, graph_id_expr, graph_node_id_expr) = match kind {
        EventKind::Kernel => (
            Some(crate::nvtx_attribution::KERNEL_VIEW),
            "CAST(NULL AS BIGINT)",
            "CAST(t.graphId AS BIGINT)",
            "CAST(t.graphNodeId AS BIGINT)",
        ),
        EventKind::Memcpy => (
            Some(crate::nvtx_attribution::MEMCPY_VIEW),
            "CAST(COALESCE(t.bytes, 0) AS BIGINT)",
            "CAST(NULL AS BIGINT)",
            "CAST(t.graphNodeId AS BIGINT)",
        ),
        EventKind::Memset => (
            Some(crate::nvtx_attribution::MEMSET_VIEW),
            "CAST(COALESCE(t.bytes, 0) AS BIGINT)",
            "CAST(NULL AS BIGINT)",
            "CAST(t.graphNodeId AS BIGINT)",
        ),
        EventKind::Sync => (
            Some(crate::nvtx_attribution::SYNC_VIEW),
            "CAST(NULL AS BIGINT)",
            "CAST(NULL AS BIGINT)",
            "CAST(NULL AS BIGINT)",
        ),
        EventKind::Graph => (
            None,
            "CAST(NULL AS BIGINT)",
            "CAST(t.graphId AS BIGINT)",
            "CAST(NULL AS BIGINT)",
        ),
        EventKind::Nvtx => (
            None,
            "CAST(NULL AS BIGINT)",
            "CAST(NULL AS BIGINT)",
            "CAST(NULL AS BIGINT)",
        ),
        // Runtime + Osrt are CPU-side host-thread events: no device,
        // no stream, no context, no graph linkage. They are first-class
        // stats kinds with a
        // five-NULL physical-axis projection so they UNION ALL cleanly
        // with the GPU kinds above. Runtime is NVTX-attributable via
        // the dedicated `attributed_runtime_rowids` view (full-interval
        // containment on globalTid). Osrt rows have no correlationId
        // and don't sit on a runtime API boundary, so they stay out of
        // the attribution path — `--nvtx` filters Osrt to empty
        // (matching today's Nvtx behaviour).
        EventKind::Runtime => (
            Some(crate::nvtx_attribution::RUNTIME_VIEW),
            "CAST(NULL AS BIGINT)",
            "CAST(NULL AS BIGINT)",
            "CAST(NULL AS BIGINT)",
        ),
        EventKind::Osrt => (
            None,
            "CAST(NULL AS BIGINT)",
            "CAST(NULL AS BIGINT)",
            "CAST(NULL AS BIGINT)",
        ),
        EventKind::GraphNode
        | EventKind::GraphEvent
        | EventKind::CudaEvent
        | EventKind::Overhead
        | EventKind::CpuSample => anyhow::bail!(
            "internal: stats does not aggregate `{}` rows; \
             they are reachable via `search` / `inspect` instead",
            kind.as_str()
        ),
    };
    // Runtime-only API-version collapse — when `collapse_versioned`
    // is set we wrap the resolved name in DuckDB's `regexp_replace`
    // to strip trailing `_v<digits>` suffixes (`cudaMalloc_v3020` →
    // `cudaMalloc`). The wrap is applied to BOTH display_name and
    // short_name so the rollup buckets agree, matching nsys's
    // `cuda_api_sum` recipe.
    let (display_expr, short_expr): (String, String) =
        if collapse_versioned && matches!(kind, EventKind::Runtime) {
            let inner_display = crate::kind_sql::display_name_expr(kind);
            let inner_short = crate::kind_sql::short_name_expr(kind);
            (
                format!("regexp_replace({inner_display}, '_v[0-9]+$', '')"),
                format!("regexp_replace({inner_short}, '_v[0-9]+$', '')"),
            )
        } else {
            (
                crate::kind_sql::display_name_expr(kind).to_string(),
                crate::kind_sql::short_name_expr(kind).to_string(),
            )
        };
    let join_clause = crate::kind_sql::name_joins(kind);
    let label = kind.as_str();
    let table = kind.table();

    let mut params: Vec<Value> = Vec::new();
    let (duration_expr, mut where_parts): (String, Vec<String>) = match abs_window {
        Some((start, end)) => {
            // Clip duration to the window: an event straddling the edge
            // contributes only the in-window portion. WHERE filters out
            // events entirely outside the window.
            params.push(Value::BigInt(end));
            params.push(Value::BigInt(start));
            params.push(Value::BigInt(end));
            params.push(Value::BigInt(start));
            (
                r#"LEAST(t."end", ?) - GREATEST(t.start, ?)"#.to_string(),
                vec![r#"t.start < ? AND t."end" > ?"#.to_string()],
            )
        }
        None => (r#"(t."end" - t.start)"#.to_string(), Vec::new()),
    };

    // NVTX instant markers (`Mark` events have a NULL end) carry no
    // duration; keep only ranges (`end IS NOT NULL`) so zero-length
    // samples don't pollute averages and percentiles.
    if matches!(kind, EventKind::Nvtx) {
        where_parts.push(r#"t."end" IS NOT NULL"#.to_string());
    }

    if nvtx_scope.is_attributed() {
        match attributed_view {
            Some(view) => where_parts.push(crate::nvtx_attribution::filter_clause(view, "t")),
            // Sync/Graph/Nvtx aren't attributable to an NVTX range —
            // emit a contradiction so the UNION ALL row produces no
            // events. (For Nvtx specifically, the `--nvtx` flag is
            // for cross-attributing GPU work; on the NVTX rows
            // themselves it would be a no-op tautology.)
            None => where_parts.push("FALSE".to_string()),
        }
    }
    let where_clause = if where_parts.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", where_parts.join(" AND "))
    };

    // NVTX, Runtime, and Osrt rows are CPU-side host-thread events
    // with no device / context / stream columns on their backing
    // tables. Project NULL of the matching SQL type so the UNION ALL
    // columns stay homogeneous with the GPU kinds. (Sync and Graph
    // both have GPU-side `deviceId` / `streamId` already.)
    let cpu_only_axes = matches!(kind, EventKind::Nvtx | EventKind::Runtime | EventKind::Osrt);
    let (dev, ctx, stm) = if cpu_only_axes {
        (
            "CAST(NULL AS INTEGER)",
            "CAST(NULL AS BIGINT)",
            "CAST(NULL AS BIGINT)",
        )
    } else {
        (
            crate::kind_sql::GPU_DEVICE_ID_EXPR,
            crate::kind_sql::GPU_CONTEXT_ID_EXPR,
            crate::kind_sql::GPU_STREAM_ID_EXPR,
        )
    };
    // NVTX-only raw eventType projection. Non-NVTX rows project a
    // typed NULL so the UNION ALL stays homogeneous. The
    // {59,70}→push_pop and {60,71}→start_end folding is a derived label in
    // the `grouped` CTE — see `NVTX_STYLE_EXPR`.
    let event_type_expr = if matches!(kind, EventKind::Nvtx) {
        "CAST(t.eventType AS BIGINT)"
    } else {
        "CAST(NULL AS BIGINT)"
    };
    // Mangled-name projection: real value for kernels (StringIds
    // probe degrades to NULL on schemas missing the column), display
    // name for non-kernel kinds (preserves per-name identity so the
    // axis doesn't collapse memcpy/sync/runtime/NVTX into a single
    // NULL bucket).
    let (mangled_expr, mangled_join): (String, String) = if matches!(kind, EventKind::Kernel) {
        let mangled_col =
            crate::column_map::maybe_col(columns, "CUPTI_ACTIVITY_KIND_KERNEL", "mangledName");
        let join = format!("LEFT JOIN nsight.StringIds s_mng ON s_mng.id = {mangled_col}");
        ("s_mng.value".to_string(), join)
    } else {
        (
            crate::kind_sql::display_name_expr(kind).to_string(),
            String::new(),
        )
    };
    // When `--group-by nvtx-parent` is active,
    // LEFT JOIN against the trace-wide parquet sidecar built by
    // `veloq_nsys_data::runtime_nvtx_parent`. Events outside every
    // NVTX range fall back to the sentinel via COALESCE. Kinds
    // without an attribution path (Graph/Osrt/Nvtx) project the
    // sentinel inline.
    let (
        parent_rowid_expr,
        parent_name_expr,
        parent_path_expr,
        domain_id_expr,
        domain_pid_expr,
        parent_join,
    ) = match nvtx_parent_sidecar {
        Some(path) => crate::nvtx_parent::join_clause(kind, path, include_nvtx_path),
        None => (
            "CAST(NULL AS BIGINT)".to_string(),
            "CAST(NULL AS VARCHAR)".to_string(),
            "CAST(NULL AS VARCHAR)".to_string(),
            "CAST(NULL AS BIGINT)".to_string(),
            "CAST(NULL AS BIGINT)".to_string(),
            String::new(),
        ),
    };
    // Kernel-only grid/block projection. Only the kernel
    // CUPTI table carries gridX/Y/Z + blockX/Y/Z; for every other
    // kind we project typed NULLs so the UNION ALL stays homogeneous.
    // The axis is rejected upstream for non-kernel kinds so a single-
    // NULL-bucket row is never produced.
    let (grid_x_e, grid_y_e, grid_z_e, block_x_e, block_y_e, block_z_e) =
        if matches!(kind, EventKind::Kernel) {
            (
                "CAST(t.gridX  AS BIGINT)",
                "CAST(t.gridY  AS BIGINT)",
                "CAST(t.gridZ  AS BIGINT)",
                "CAST(t.blockX AS BIGINT)",
                "CAST(t.blockY AS BIGINT)",
                "CAST(t.blockZ AS BIGINT)",
            )
        } else {
            (
                "CAST(NULL AS BIGINT)",
                "CAST(NULL AS BIGINT)",
                "CAST(NULL AS BIGINT)",
                "CAST(NULL AS BIGINT)",
                "CAST(NULL AS BIGINT)",
                "CAST(NULL AS BIGINT)",
            )
        };
    let sql = format!(
        "SELECT {display_expr} AS display_name, \
                {short_expr}   AS short_name, \
                {mangled_expr} AS mangled_name, \
                '{label}'      AS kind, \
                {duration_expr} AS duration, \
                {dev}        AS device_id, \
                {ctx}        AS context_id, \
                {stm}        AS stream_id, \
                {bytes_expr}                       AS bytes, \
                {graph_id_expr}                    AS graph_id, \
                {graph_node_id_expr}               AS graph_node_id, \
                {event_type_expr}                  AS event_type, \
                {parent_rowid_expr}                AS nvtx_parent_rowid, \
                {parent_name_expr}                 AS nvtx_parent_name, \
                {parent_path_expr}                 AS nvtx_path, \
                {domain_id_expr}                   AS nvtx_domain_id, \
                {domain_pid_expr}                  AS nvtx_domain_pid, \
                {grid_x_e}                         AS grid_x, \
                {grid_y_e}                         AS grid_y, \
                {grid_z_e}                         AS grid_z, \
                {block_x_e}                        AS block_x, \
                {block_y_e}                        AS block_y, \
                {block_z_e}                        AS block_z \
         FROM nsight.{table} t {join_clause} {mangled_join} {parent_join} {where_clause}"
    );
    Ok((sql, params))
}

/// SQL expression mapping raw NVTX `eventType` ints to the derived
/// style label. Mirrors `nvtx_style_label` Rust-side. `NULL` on
/// non-NVTX rows so the column collapses into a single bucket for
/// GROUP BY without splitting GPU rows.
///
/// The numeric constants come from NSys's
/// `enum NvtxEventType` (NSys SDK; see Nsight Systems documentation):
///
/// * 59, 70 → PushPop range (legacy + extended payload)
/// * 60, 71 → StartEnd range (legacy + extended payload)
///
/// Anything else (NVTX_RESOURCE_*, NVTX_DOMAIN_*, future enum
/// extensions) lands in `"unknown"` rather than spawning bucket-
/// per-int — this keeps the group count bounded as nsys adds new
/// event types.
const NVTX_STYLE_EXPR: &str = "CASE \
    WHEN event_type IS NULL THEN NULL \
    WHEN event_type IN (59, 70) THEN 'push_pop' \
    WHEN event_type IN (60, 71) THEN 'start_end' \
    ELSE 'unknown' \
END";

/// Rust-side mirror of `NVTX_STYLE_EXPR` for derived response fields.
/// Used to coerce the SQL-emitted `nvtx_style` VARCHAR back into a
/// `&'static str` so consumers don't carry an unbounded String.
fn nvtx_style_label(raw: &str) -> &'static str {
    match raw {
        "push_pop" => "push_pop",
        "start_end" => "start_end",
        _ => "unknown",
    }
}
