use super::HIST_BOUNDARIES_NS;
use super::sql::NVTX_STYLE_EXPR;
use crate::{NsysQueryError, NsysQueryResult};
use veloq_core::{AxisUsage, Direction, SortKeyDef, SortKeySpec, SortSpec};

const NO_AXES: &[&str] = &[];
const DEVICE_AXIS: &[&str] = &["device"];

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
pub(super) struct GroupBySql {
    pub(super) name_select: &'static str,
    pub(super) short_name_select: &'static str,
    pub(super) device_select: &'static str,
    pub(super) context_select: &'static str,
    pub(super) stream_select: &'static str,
    pub(super) graph_select: &'static str,
    pub(super) graph_node_select: &'static str,
    /// SELECT fragment for the innermost-NVTX-range rowid. Either a
    /// passthrough of the `events.nvtx_parent_rowid` column (axis
    /// active) or a typed NULL literal (axis inactive).
    pub(super) nvtx_parent_rowid_select: &'static str,
    /// SELECT fragment for the parent NVTX range name. When the axis
    /// is active, projects the COALESCE-sentinel string from `events`;
    /// otherwise NULL. Wrapped in `--group-by nvtx-parent` so the
    /// non-axis paths keep their existing schema.
    pub(super) nvtx_parent_name_select: &'static str,
    /// SELECT fragment for the full NVTX path. Populated only when
    /// `--group-by nvtx-path` is active.
    pub(super) nvtx_path_select: &'static str,
    /// SELECT fragments for the enclosing range's NVTX domain identity
    /// `(domainId, pid)`. Populated only when `--group-by nvtx-path` is
    /// active; both participate in GROUP BY so same-name ranges in
    /// distinct `(pid, domainId)` domains do not collapse.
    pub(super) nvtx_domain_id_select: &'static str,
    pub(super) nvtx_domain_pid_select: &'static str,
    /// 6 SELECT fragments for the kernel launch grid/block tuple
    /// — when the `grid_block` axis is active these passthrough the
    /// per-kind columns; otherwise they project typed NULLs so the
    /// outer SELECT shape stays stable.
    pub(super) grid_x_select: &'static str,
    pub(super) grid_y_select: &'static str,
    pub(super) grid_z_select: &'static str,
    pub(super) block_x_select: &'static str,
    pub(super) block_y_select: &'static str,
    pub(super) block_z_select: &'static str,
    pub(super) group_keys_sql: String,
}

/// Decision recorded by [`resolve_name_axis`] so the response and
/// `--group-by` SQL builder agree on whether to use Mangled or fall
/// back. `mangled_axis_fallback` on the response surfaces the same
/// signal to consumers.
#[derive(Debug, Clone, Copy)]
pub(super) struct NameAxisResolution {
    pub(super) effective: NameAxis,
    pub(super) fell_back: bool,
}

/// `NameAxis::Mangled` falls back to `Demangled` when the trace's
/// `CUPTI_ACTIVITY_KIND_KERNEL` table lacks a `mangledName` column
/// (older NSys schemas). The fallback is logged via `log::info!` so
/// it lands on stderr in human modes and gets suppressed by
/// `2> /dev/null` in JSON pipelines.
/// The response carries `mangled_axis_fallback: true` so JSON consumers
/// who care can still detect it without parsing stderr.
pub(super) fn resolve_name_axis(
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
    pub(super) fn for_axes(g: &GroupBy) -> Self {
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
pub(super) struct HistSql {
    pub(super) grouped_cols: String,
    pub(super) outer_cols: String,
}

impl HistSql {
    /// Build histogram column SQL when `enabled`; otherwise return
    /// empty strings so the splice in `run()` stays neutral.
    pub(super) fn build(enabled: bool) -> Self {
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

pub(super) fn stats_sort_sql(spec: &SortSpec) -> NsysQueryResult<String> {
    // total_ns as the last-resort tiebreaker keeps output deterministic
    // even when the user sorts by something like `name` and several
    // groups happen to share a name (shouldn't happen, but cheap).
    crate::query_sql::sort::order_by::<SortKey>(
        spec,
        SortKey::column,
        NsysQueryError::stats_sort_invalid,
        "total_ns",
    )
}

impl GroupBy {
    /// Validate local child axes whose ids are only meaningful under a
    /// device parent. `--device <id>` fixes the parent; otherwise the
    /// parent must be projected as part of the group key.
    pub(crate) fn validate_device_parent_axes(
        self,
        verb: &'static str,
        device: Option<i32>,
    ) -> NsysQueryResult<()> {
        let fixed = if device.is_some() {
            DEVICE_AXIS
        } else {
            NO_AXES
        };
        let projected = if self.device { DEVICE_AXIS } else { NO_AXES };
        let usage = AxisUsage::new(fixed, projected);
        let result = if self.stream {
            usage.validate_projection("stream", DEVICE_AXIS)
        } else if self.context {
            usage.validate_projection("context", DEVICE_AXIS)
        } else {
            Ok(())
        };
        result.map_err(
            |err| crate::NsysQueryError::StatsGroupByDeviceParentRequired {
                verb,
                axis: err.axis(),
            },
        )
    }

    pub fn from_arg(s: &str) -> NsysQueryResult<Self> {
        let mut out = Self::default();
        let mut name_seen: Option<&'static str> = None;
        for raw in s.split(',') {
            let tok = raw.trim().to_ascii_lowercase();
            match tok.as_str() {
                "" => continue,
                "short" | "shortname" | "short_name" => {
                    if let Some(prev) = name_seen {
                        return Err(crate::NsysQueryError::StatsGroupByNameAxisConflict {
                            previous: prev,
                            current: "short",
                        });
                    }
                    out.name = NameAxis::ShortName;
                    name_seen = Some("short");
                }
                "demangled" | "demangled_name" | "variant" => {
                    if let Some(prev) = name_seen {
                        return Err(crate::NsysQueryError::StatsGroupByNameAxisConflict {
                            previous: prev,
                            current: "demangled",
                        });
                    }
                    out.name = NameAxis::Demangled;
                    name_seen = Some("demangled");
                }
                "mangled" | "mangled_name" => {
                    if let Some(prev) = name_seen {
                        return Err(crate::NsysQueryError::StatsGroupByNameAxisConflict {
                            previous: prev,
                            current: "mangled",
                        });
                    }
                    out.name = NameAxis::Mangled;
                    name_seen = Some("mangled");
                }
                "no-name" | "noname" | "none" => {
                    if let Some(prev) = name_seen {
                        return Err(crate::NsysQueryError::StatsGroupByNameAxisConflict {
                            previous: prev,
                            current: "no-name",
                        });
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
                other => {
                    return Err(crate::NsysQueryError::stats_group_by_unknown_token(other));
                }
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
