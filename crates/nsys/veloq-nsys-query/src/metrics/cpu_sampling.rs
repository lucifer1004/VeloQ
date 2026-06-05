//! `--type cpu-sampling` source.
//!
//! Reads `COMPOSITE_EVENTS` (periodic IP samples) plus
//! `SAMPLING_CALLCHAINS` (per-sample stack frames) and rolls up
//! per-key hotspot histograms on four axes (`symbol` / `module` /
//! `tid` / `cpu`) or bucketed sample counts. Trust signals:
//! `unresolved_leaf_share`, `kernel_leaf_share`,
//! `truncated_stack_share`.

use anyhow::{Context, Result};
use duckdb::types::Value;
use serde::Serialize;
use veloq_core::{Direction, SortKeyDef, SortKeySpec, SortSpec};
use veloq_nsys_data::Trace;

use super::{Coverage, CpuBucketSample, CpuSamplingBody, CpuSamplingRequest, MetricsCommon};

/// `--group-by` axis for `--type cpu-sampling`. The default is
/// `Symbol` (leaf-frame function — `perf top` ergonomics). Other
/// axes don't need the `SAMPLING_CALLCHAINS` join: `Tid` / `Cpu`
/// aggregate purely on `COMPOSITE_EVENTS` columns, and `Module`
/// joins only the leaf frame's module string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuGroupBy {
    Symbol,
    Tid,
    Cpu,
    Module,
}

impl CpuGroupBy {
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "symbol" => Ok(Self::Symbol),
            "tid" | "thread" => Ok(Self::Tid),
            "cpu" | "core" => Ok(Self::Cpu),
            "module" | "binary" => Ok(Self::Module),
            other => anyhow::bail!(
                "unknown --group-by `{other}` for cpu-sampling \
                 (expected: symbol, tid, cpu, module)"
            ),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Symbol => "symbol",
            Self::Tid => "tid",
            Self::Cpu => "cpu",
            Self::Module => "module",
        }
    }
}

/// One row in the cpu-sampling hotspot histogram. Fields are
/// per-`--group-by` axis: `symbol_name` + `module_name` are
/// populated on the `symbol` axis (and `module_name` on `module`),
/// `cpu` on the `cpu` axis, `global_tid` / `pid` / `tid` on the
/// `tid` axis. The `key` column is always present and stringifies
/// the row's identity in the chosen axis, useful for deterministic
/// sorting and CSV/table rendering without a per-axis schema swap.
///
/// **Unresolved leaves**: on `symbol` axis, unresolved frames roll up
/// per-module — each unique `<unresolved>@<module>` becomes its own
/// row so agents can distinguish "47% unresolved in `[kernel.kallsyms]`"
/// from "21% unresolved in `libToolsInjection64.so`".
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct HotspotRow {
    pub key: String,
    pub samples: i64,
    pub percentage: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub module_name: Option<String>,
    /// Present on `--group-by symbol` and `module` axes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_mode: Option<bool>,
    /// Present on `--group-by symbol` axis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unresolved: Option<bool>,
    /// Present on `--group-by cpu` axis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu: Option<i64>,
    /// Present on `--group-by tid` axis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub global_tid: Option<i64>,
    /// Decoded process id — `(global_tid >> 24) & 0xFFFFFF`.
    /// See [`crate::decode_global_tid`] for the full bit layout.
    /// Present on `--group-by tid` axis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pid: Option<i64>,
    /// Decoded thread id — `global_tid & 0xFFFF`. NSys's TID slot is
    /// 16 bits; bits 16..23 carry the source-domain id. Present on
    /// `--group-by tid` axis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tid: Option<i64>,
}

pub(super) fn run_cpu_sampling(
    trace: &Trace,
    req: &CpuSamplingRequest,
    abs_window: Option<(i64, i64)>,
    trace_origin_ns: i64,
    trace_span_ns: (i64, i64),
) -> Result<CpuSamplingBody> {
    if !trace.table_exists("COMPOSITE_EVENTS") {
        anyhow::bail!(
            "metrics --type cpu-sampling requires `COMPOSITE_EVENTS`, \
             which is absent from this trace; re-capture with \
             `nsys profile --sample=process-tree`"
        );
    }
    // SAMPLING_CALLCHAINS missing is recoverable — `--group-by tid` /
    // `cpu` don't need stacks. Surface a hint only when the requested
    // axis actually needs them.
    let group_by = match req.group_by.as_deref() {
        Some(s) => CpuGroupBy::parse(s)?,
        None => CpuGroupBy::Symbol,
    };
    let needs_callchains = matches!(group_by, CpuGroupBy::Symbol | CpuGroupBy::Module);
    if needs_callchains && !trace.table_exists("SAMPLING_CALLCHAINS") {
        anyhow::bail!(
            "--group-by {} needs `SAMPLING_CALLCHAINS` (per-sample stacks), \
             which is absent from this trace; either re-capture with stack \
             sampling enabled or switch to `--group-by tid` / `cpu`",
            group_by.as_str()
        );
    }
    // `--name` is only meaningful on the symbol / module axes where it
    // matches against a string key. On `tid` / `cpu` axes the key is
    // numeric and the glob has nothing to bind against — silently
    // ignoring would let an agent burn time wondering why the filter
    // had no effect.
    if req.name_glob.is_some() && !needs_callchains {
        anyhow::bail!(
            "--name doesn't apply on --group-by {} (keys are numeric); \
             drop it or switch to `--group-by symbol` / `module`",
            group_by.as_str()
        );
    }

    let stats = query_cpu_sample_stats(trace, req, abs_window)?;

    // Apply name glob filter on the relevant axis. We do it in Rust on
    // the resolved rows so the SQL stays uniform and the LIKE pattern
    // doesn't have to compose with the per-axis column expression.
    let like_pattern = req.name_glob.as_deref().map(crate::search_glob_to_like);

    let hotspot = if req.common.bucket_ns.is_none() {
        let rows = match group_by {
            CpuGroupBy::Symbol => query_cpu_hotspot_symbol(trace, req, abs_window)?,
            CpuGroupBy::Module => query_cpu_hotspot_module(trace, req, abs_window)?,
            CpuGroupBy::Tid => query_cpu_hotspot_tid(trace, req, abs_window)?,
            CpuGroupBy::Cpu => query_cpu_hotspot_cpu(trace, req, abs_window)?,
        };
        finalize_hotspot(rows, stats.samples_total, like_pattern.as_deref(), group_by)
    } else {
        Vec::new()
    };

    let cpu_buckets = match req.common.bucket_ns {
        None => Vec::new(),
        Some(bucket_ns) => query_cpu_buckets(
            trace,
            req,
            abs_window,
            bucket_ns,
            group_by,
            like_pattern.as_deref(),
            trace_origin_ns,
        )?,
    };

    // --sort applies to the hotspot list only. Bucket mode is
    // always time-ordered (cf. `run_gpu`).
    let mut hotspot = hotspot;
    if req.common.bucket_ns.is_none() {
        let default_spec = SortSpec::single("samples");
        let sort_spec = req.common.sort.as_ref().unwrap_or(&default_spec);
        sort_hotspot(&mut hotspot, sort_spec)?;
    }
    let hotspot_pre_limit = hotspot.len() as i64;
    if req.common.bucket_ns.is_none() {
        hotspot.truncate(req.common.limit);
    }

    let buckets_pre_limit = cpu_buckets.len() as i64;
    let mut cpu_buckets = cpu_buckets;
    if req.common.bucket_ns.is_some() {
        cpu_buckets.truncate(req.common.limit);
    }

    let metrics_span_ns = stats.span;
    let coverage = Coverage::compute(metrics_span_ns, trace_span_ns, stats.samples_total, None);

    let safe_share = |n: i64, d: i64| -> Option<f64> {
        if d <= 0 {
            None
        } else {
            Some((n as f64 / d as f64).clamp(0.0, 1.0))
        }
    };
    let unresolved_leaf_share = safe_share(stats.n_unresolved_leaf, stats.samples_total);
    let kernel_leaf_share = safe_share(stats.n_kernel_leaf, stats.samples_total);
    let truncated_stack_share = safe_share(stats.n_truncated_stack, stats.samples_total);

    let (count, total_matched) = match req.common.bucket_ns {
        None => (hotspot.len(), hotspot_pre_limit),
        Some(_) => (cpu_buckets.len(), buckets_pre_limit),
    };

    Ok(CpuSamplingBody {
        count,
        total_matched,
        rows: hotspot,
        auxiliary: super::CpuSamplingAuxiliary {
            common: MetricsCommon {
                trace_origin_ns,
                trace_span_ns,
                metrics_span_ns,
                coverage,
                time_window_ns: abs_window,
                bucket_ns: req.common.bucket_ns,
            },
            group_by: group_by.as_str(),
            name_glob: req.name_glob.clone(),
            cpu_filter: req.cpu,
            tid_filter: req.tid,
            unresolved_leaf_share,
            kernel_leaf_share,
            truncated_stack_share,
            cpu_buckets,
        },
    })
}

/// Trust-signal aggregates over the filtered sample set:
/// total count, time span, and three "what's the data like" tallies.
/// One query so we hit `COMPOSITE_EVENTS` + `SAMPLING_CALLCHAINS`
/// once, not three times.
struct CpuSampleStats {
    samples_total: i64,
    span: Option<(i64, i64)>,
    n_unresolved_leaf: i64,
    n_kernel_leaf: i64,
    n_truncated_stack: i64,
}

fn query_cpu_sample_stats(
    trace: &Trace,
    req: &CpuSamplingRequest,
    abs_window: Option<(i64, i64)>,
) -> Result<CpuSampleStats> {
    let (filtered_sql, params) = build_cpu_filtered_samples_cte(req, abs_window);
    let has_callchains = trace.table_exists("SAMPLING_CALLCHAINS");
    // Trust signals all come from SAMPLING_CALLCHAINS. When that table
    // is absent (rare for cpu-sampling captures, but defensible), zero
    // the share fields rather than fail — caller can still get span +
    // count for `--group-by tid` / `cpu`.
    let sql = if has_callchains {
        format!(
            r#"
            WITH {filtered_sql},
            leaf AS (
                SELECT c.id, c.kernelMode, c.unresolved
                FROM nsight.SAMPLING_CALLCHAINS c
                WHERE c.stackDepth = 0
            ),
            deepest AS (
                SELECT c.id, s.value AS deepest_symbol
                FROM nsight.SAMPLING_CALLCHAINS c
                JOIN (
                    SELECT id, MAX(stackDepth) AS md
                    FROM nsight.SAMPLING_CALLCHAINS
                    GROUP BY id
                ) m ON m.id = c.id AND m.md = c.stackDepth
                LEFT JOIN nsight.StringIds s ON s.id = c.symbol
            )
            SELECT
                CAST(COUNT(fs.id) AS BIGINT) AS samples_total,
                CAST(COALESCE(MIN(fs.start), 0) AS BIGINT) AS span_lo,
                CAST(COALESCE(MAX(fs.start), 0) AS BIGINT) AS span_hi,
                CAST(SUM(CASE WHEN COALESCE(leaf.unresolved, 0) = 1 THEN 1 ELSE 0 END)
                    AS BIGINT) AS n_unresolved,
                CAST(SUM(CASE WHEN COALESCE(leaf.kernelMode, 0) = 1 THEN 1 ELSE 0 END)
                    AS BIGINT) AS n_kernel,
                CAST(SUM(CASE WHEN deepest.deepest_symbol = '[Max depth]' THEN 1 ELSE 0 END)
                    AS BIGINT) AS n_truncated
            FROM filtered_samples fs
            LEFT JOIN leaf ON leaf.id = fs.id
            LEFT JOIN deepest ON deepest.id = fs.id
            "#
        )
    } else {
        format!(
            r#"
            WITH {filtered_sql}
            SELECT
                CAST(COUNT(fs.id) AS BIGINT) AS samples_total,
                CAST(COALESCE(MIN(fs.start), 0) AS BIGINT) AS span_lo,
                CAST(COALESCE(MAX(fs.start), 0) AS BIGINT) AS span_hi,
                CAST(0 AS BIGINT) AS n_unresolved,
                CAST(0 AS BIGINT) AS n_kernel,
                CAST(0 AS BIGINT) AS n_truncated
            FROM filtered_samples fs
            "#
        )
    };
    let conn = trace.conn();
    let mut stmt = conn
        .prepare(&sql)
        .context("preparing cpu-sampling stats SQL")?;
    let params_ref = crate::bind(&params);
    let mut rows = stmt.query(params_ref.as_slice())?;
    let r = rows
        .next()?
        .ok_or_else(|| anyhow::anyhow!("internal: cpu-sampling stats returned no row"))?;
    let samples_total: i64 = r.get("samples_total")?;
    let span = if samples_total > 0 {
        Some((r.get("span_lo")?, r.get("span_hi")?))
    } else {
        None
    };
    Ok(CpuSampleStats {
        samples_total,
        span,
        n_unresolved_leaf: r.get("n_unresolved")?,
        n_kernel_leaf: r.get("n_kernel")?,
        n_truncated_stack: r.get("n_truncated")?,
    })
}

/// Filtered-samples CTE — applied uniformly across summary / bucket /
/// trust-signal queries so they all aggregate over the same row set.
/// Returns the CTE body (without `WITH` prefix) plus the parameter
/// list to bind in order.
fn build_cpu_filtered_samples_cte(
    req: &CpuSamplingRequest,
    abs_window: Option<(i64, i64)>,
) -> (String, Vec<Value>) {
    let global_tid = veloq_nsys_data::sql_expr::u64_bits_to_i64("globalTid");
    super::build_filtered_cte(
        "filtered_samples",
        "COMPOSITE_EVENTS",
        &format!("id, start, cpu, {global_tid} AS globalTid"),
        req.cpu,
        req.tid,
        abs_window,
    )
}

/// Internal row carrying raw SQL output before percentage / key
/// derivation. Each grouping axis populates only the fields it has
/// data for; the rest stay `None`.
struct RawHotspot {
    symbol_name: Option<String>,
    module_path: Option<String>,
    kernel_mode: Option<bool>,
    unresolved: Option<bool>,
    cpu: Option<i64>,
    global_tid: Option<i64>,
    samples: i64,
}

fn query_cpu_hotspot_symbol(
    trace: &Trace,
    req: &CpuSamplingRequest,
    abs_window: Option<(i64, i64)>,
) -> Result<Vec<RawHotspot>> {
    let (filtered_sql, params) = build_cpu_filtered_samples_cte(req, abs_window);
    let sql = format!(
        r#"
        WITH {filtered_sql},
        leaf AS (
            SELECT c.id, c.symbol, c.module, c.kernelMode, c.unresolved
            FROM nsight.SAMPLING_CALLCHAINS c
            WHERE c.stackDepth = 0
        )
        SELECT
            -- Erase per-frame raw-address strings for unresolved
            -- leaves so all addresses in the same (module, kernel_mode)
            -- collapse to a single bucket. Without this, each unique
            -- kernel IP (one StringIds row per `0x<addr>`) becomes its
            -- own group and the unresolved tail balloons.
            CASE WHEN COALESCE(leaf.unresolved, 0) = 1 THEN NULL ELSE s.value END AS symbol_name,
            m.value AS module_path,
            CAST(COALESCE(leaf.kernelMode, 0) AS BIGINT) AS kernel_mode_int,
            CAST(COALESCE(leaf.unresolved, 0) AS BIGINT) AS unresolved_int,
            CAST(COUNT(*) AS BIGINT) AS samples
        FROM filtered_samples fs
        LEFT JOIN leaf ON leaf.id = fs.id
        LEFT JOIN nsight.StringIds s ON s.id = leaf.symbol
        LEFT JOIN nsight.StringIds m ON m.id = leaf.module
        GROUP BY symbol_name, module_path, kernel_mode_int, unresolved_int
        "#
    );
    super::query_rows(trace, &sql, &params, "cpu-sampling hotspot-symbol", |r| {
        Ok(RawHotspot {
            symbol_name: r.get("symbol_name")?,
            module_path: r.get("module_path")?,
            kernel_mode: Some(r.get::<_, i64>("kernel_mode_int")? != 0),
            unresolved: Some(r.get::<_, i64>("unresolved_int")? != 0),
            cpu: None,
            global_tid: None,
            samples: r.get("samples")?,
        })
    })
}

fn query_cpu_hotspot_module(
    trace: &Trace,
    req: &CpuSamplingRequest,
    abs_window: Option<(i64, i64)>,
) -> Result<Vec<RawHotspot>> {
    let (filtered_sql, params) = build_cpu_filtered_samples_cte(req, abs_window);
    let sql = format!(
        r#"
        WITH {filtered_sql},
        leaf AS (
            SELECT c.id, c.module, c.kernelMode
            FROM nsight.SAMPLING_CALLCHAINS c
            WHERE c.stackDepth = 0
        )
        SELECT
            m.value AS module_path,
            CAST(COALESCE(leaf.kernelMode, 0) AS BIGINT) AS kernel_mode_int,
            CAST(COUNT(*) AS BIGINT) AS samples
        FROM filtered_samples fs
        LEFT JOIN leaf ON leaf.id = fs.id
        LEFT JOIN nsight.StringIds m ON m.id = leaf.module
        GROUP BY module_path, kernel_mode_int
        "#
    );
    super::query_rows(trace, &sql, &params, "cpu-sampling hotspot-module", |r| {
        Ok(RawHotspot {
            symbol_name: None,
            module_path: r.get("module_path")?,
            kernel_mode: Some(r.get::<_, i64>("kernel_mode_int")? != 0),
            unresolved: None,
            cpu: None,
            global_tid: None,
            samples: r.get("samples")?,
        })
    })
}

fn query_cpu_hotspot_tid(
    trace: &Trace,
    req: &CpuSamplingRequest,
    abs_window: Option<(i64, i64)>,
) -> Result<Vec<RawHotspot>> {
    let (filtered_sql, params) = build_cpu_filtered_samples_cte(req, abs_window);
    let sql = format!(
        r#"
        WITH {filtered_sql}
        SELECT
            globalTid AS global_tid,
            CAST(COUNT(*) AS BIGINT) AS samples
        FROM filtered_samples
        GROUP BY global_tid
        "#
    );
    super::query_rows(trace, &sql, &params, "cpu-sampling hotspot-tid", |r| {
        Ok(RawHotspot {
            symbol_name: None,
            module_path: None,
            kernel_mode: None,
            unresolved: None,
            cpu: None,
            global_tid: Some(r.get("global_tid")?),
            samples: r.get("samples")?,
        })
    })
}

fn query_cpu_hotspot_cpu(
    trace: &Trace,
    req: &CpuSamplingRequest,
    abs_window: Option<(i64, i64)>,
) -> Result<Vec<RawHotspot>> {
    let (filtered_sql, params) = build_cpu_filtered_samples_cte(req, abs_window);
    let sql = format!(
        r#"
        WITH {filtered_sql}
        SELECT
            CAST(cpu AS BIGINT) AS cpu,
            CAST(COUNT(*) AS BIGINT) AS samples
        FROM filtered_samples
        GROUP BY cpu
        "#
    );
    super::query_rows(trace, &sql, &params, "cpu-sampling hotspot-cpu", |r| {
        Ok(RawHotspot {
            symbol_name: None,
            module_path: None,
            kernel_mode: None,
            unresolved: None,
            cpu: Some(r.get("cpu")?),
            global_tid: None,
            samples: r.get("samples")?,
        })
    })
}

/// Convert raw rows to public `HotspotRow`s — basename modules, derive
/// `key`, apply `--name` glob, and compute percentages against the
/// total filtered sample count.
fn finalize_hotspot(
    raw: Vec<RawHotspot>,
    samples_total: i64,
    name_like: Option<&str>,
    group_by: CpuGroupBy,
) -> Vec<HotspotRow> {
    let total = samples_total.max(1);
    let mut out: Vec<HotspotRow> = Vec::with_capacity(raw.len());
    for r in raw {
        let module_name = r.module_path.as_deref().map(crate::module_basename);
        let (key, symbol_name) = match group_by {
            CpuGroupBy::Symbol => {
                if r.unresolved == Some(true) {
                    let m = module_name.as_deref().unwrap_or("?");
                    (format!("<unresolved>@{m}"), None)
                } else {
                    match r.symbol_name.clone() {
                        Some(s) => (s.clone(), Some(s)),
                        None => ("<no symbol>".to_string(), None),
                    }
                }
            }
            CpuGroupBy::Module => {
                let m = module_name
                    .clone()
                    .unwrap_or_else(|| "<unknown>".to_string());
                (m, None)
            }
            CpuGroupBy::Tid => match r.global_tid {
                Some(g) => (g.to_string(), None),
                None => continue, // shouldn't happen; defensive
            },
            CpuGroupBy::Cpu => match r.cpu {
                Some(c) => (c.to_string(), None),
                None => continue,
            },
        };
        // --name applies to the row's `key` for symbol/module axes;
        // tid/cpu axes ignore it (numeric keys aren't glob-meaningful).
        if let Some(pat) = name_like
            && matches!(group_by, CpuGroupBy::Symbol | CpuGroupBy::Module)
            && !crate::sql_like_match(&key, pat)
        {
            continue;
        }
        let (pid, tid) = match r.global_tid {
            Some(g) => {
                let (p, t) = crate::decode_global_tid(g);
                (Some(p), Some(t))
            }
            None => (None, None),
        };
        let percentage = (r.samples as f64 * 100.0 / total as f64).clamp(0.0, 100.0);
        out.push(HotspotRow {
            key,
            samples: r.samples,
            percentage,
            symbol_name,
            module_name,
            kernel_mode: r.kernel_mode,
            unresolved: r.unresolved,
            cpu: r.cpu,
            global_tid: r.global_tid,
            pid,
            tid,
        });
    }
    out
}

fn query_cpu_buckets(
    trace: &Trace,
    req: &CpuSamplingRequest,
    abs_window: Option<(i64, i64)>,
    bucket_ns: i64,
    group_by: CpuGroupBy,
    name_like: Option<&str>,
    primary_origin_ns: i64,
) -> Result<Vec<CpuBucketSample>> {
    let anchor = abs_window.map(|(s, _)| s).unwrap_or(primary_origin_ns);
    let (filtered_sql, params) = build_cpu_filtered_samples_cte(req, abs_window);
    let needs_callchains = matches!(group_by, CpuGroupBy::Symbol | CpuGroupBy::Module);
    if needs_callchains && !trace.table_exists("SAMPLING_CALLCHAINS") {
        anyhow::bail!(
            "--group-by {} bucketed mode needs `SAMPLING_CALLCHAINS`, \
             which is absent from this trace",
            group_by.as_str()
        );
    }

    // Group-by expression — same axis a hotspot query uses, but inside
    // a per-bucket aggregation. `key_expr` is the SQL projection that
    // becomes the `key` column. The Rust side basenames modules and
    // applies the `--name` glob after the SELECT (so the SQL stays
    // axis-uniform).
    let (key_expr, joins) = match group_by {
        CpuGroupBy::Symbol => (
            "COALESCE(s.value, '<unresolved>@' || COALESCE(m.value, '?'))",
            "LEFT JOIN nsight.SAMPLING_CALLCHAINS c \
                ON c.id = fs.id AND c.stackDepth = 0 \
             LEFT JOIN nsight.StringIds s ON s.id = c.symbol \
             LEFT JOIN nsight.StringIds m ON m.id = c.module",
        ),
        CpuGroupBy::Module => (
            "COALESCE(m.value, '<unknown>')",
            "LEFT JOIN nsight.SAMPLING_CALLCHAINS c \
                ON c.id = fs.id AND c.stackDepth = 0 \
             LEFT JOIN nsight.StringIds m ON m.id = c.module",
        ),
        CpuGroupBy::Tid => ("CAST(fs.globalTid AS VARCHAR)", ""),
        CpuGroupBy::Cpu => ("CAST(fs.cpu AS VARCHAR)", ""),
    };

    let sql = format!(
        r#"
        WITH {filtered_sql},
        bucketed AS (
            SELECT
                CAST(FLOOR(CAST(fs.start - {anchor} AS DOUBLE) / {bucket}) AS BIGINT)
                    * {bucket} + {anchor} AS t_start,
                {key_expr} AS key
            FROM filtered_samples fs
            {joins}
        ),
        agg AS (
            SELECT
                t_start,
                t_start + {bucket} AS t_end,
                key,
                CAST(COUNT(*) AS BIGINT) AS samples
            FROM bucketed
            GROUP BY t_start, key
        )
        SELECT *
        FROM agg
        ORDER BY t_start ASC, samples DESC, key ASC
        "#,
        bucket = bucket_ns,
    );
    let conn = trace.conn();
    let mut stmt = conn
        .prepare(&sql)
        .context("preparing cpu-sampling bucket SQL")?;
    let params_ref = crate::bind(&params);
    let mut rows = stmt.query(params_ref.as_slice())?;
    let mut out: Vec<CpuBucketSample> = Vec::new();
    while let Some(r) = rows.next()? {
        let key: String = r.get("key")?;
        // Apply --name glob in Rust on symbol / module axes; the SQL
        // already projects basenames-as-part-of-key for symbol axis,
        // so glob matches against the projected `key` correctly.
        let render_key = match group_by {
            CpuGroupBy::Module => crate::module_basename(&key),
            _ => key.clone(),
        };
        if let Some(pat) = name_like
            && matches!(group_by, CpuGroupBy::Symbol | CpuGroupBy::Module)
            && !crate::sql_like_match(&render_key, pat)
        {
            continue;
        }
        let samples: i64 = r.get("samples")?;
        out.push(CpuBucketSample {
            t_start_ns: r.get("t_start")?,
            t_end_ns: r.get("t_end")?,
            key: render_key,
            agg: "sum",
            value: samples as f64,
            samples,
        });
    }
    Ok(out)
}

/// Sort axes the cpu-sampling hotspot list supports.
///
/// Enum name is `HotspotSortKey` (not just `Key`) so that the
/// `RowKey` variant doesn't trip clippy's `enum_variant_names` lint,
/// which fires when a variant name ends with the enum's own name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotspotSortKey {
    Samples,
    Percentage,
    RowKey,
    Symbol,
    Module,
    Cpu,
    Tid,
}

impl SortKeyDef for HotspotSortKey {
    fn specs() -> &'static [SortKeySpec<Self>] {
        // `samples` leads — agents almost always want "what's hottest"
        // first. `key` is the canonical name for the row-grouping
        // column, with `name` as an ergonomic alias.
        &[
            SortKeySpec {
                variant: HotspotSortKey::Samples,
                canonical: "samples",
                aliases: &[],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: HotspotSortKey::Percentage,
                canonical: "percentage",
                aliases: &["percent", "pct"],
                default_dir: Direction::Desc,
            },
            SortKeySpec {
                variant: HotspotSortKey::RowKey,
                canonical: "key",
                aliases: &["name"],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: HotspotSortKey::Symbol,
                canonical: "symbol",
                aliases: &[],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: HotspotSortKey::Module,
                canonical: "module",
                aliases: &[],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: HotspotSortKey::Cpu,
                canonical: "cpu",
                aliases: &[],
                default_dir: Direction::Asc,
            },
            SortKeySpec {
                variant: HotspotSortKey::Tid,
                canonical: "tid",
                aliases: &[],
                default_dir: Direction::Asc,
            },
        ]
    }
}

/// Sort the cpu-sampling hotspot list per the user's `--sort` spec.
/// Default direction by key matches what an agent would expect:
/// `samples` and `percentage` DESC (biggest first), names ASC.
fn sort_hotspot(out: &mut [HotspotRow], spec: &SortSpec) -> Result<()> {
    let resolved: Vec<(HotspotSortKey, Direction)> = spec
        .fields()
        .iter()
        .map(|f| HotspotSortKey::from_field(f).map_err(Into::into))
        .collect::<Result<_>>()?;
    // Stable: tie-break on key ASC.
    veloq_core::sort_in_memory(
        out,
        &resolved,
        |k, a, b| match k {
            HotspotSortKey::Samples => a.samples.cmp(&b.samples),
            HotspotSortKey::Percentage => a.percentage.total_cmp(&b.percentage),
            HotspotSortKey::RowKey => a.key.cmp(&b.key),
            HotspotSortKey::Symbol => a.symbol_name.cmp(&b.symbol_name),
            HotspotSortKey::Module => a.module_name.cmp(&b.module_name),
            HotspotSortKey::Cpu => a.cpu.cmp(&b.cpu),
            HotspotSortKey::Tid => a.global_tid.cmp(&b.global_tid),
        },
        |a, b| a.key.cmp(&b.key),
    );
    Ok(())
}
