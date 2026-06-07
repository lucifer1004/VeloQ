//! Table/CSV projections for the native NCU summary and the detail-verb
//! payloads.
//!
//! The JSON shape remains the full per-verb response. These projections
//! mirror the JSON keys (one nesting level + BTreeMap expansion) so
//! csv / table / JSON stay isomorphic on column set, the cross-format
//! consistency contract.

use std::collections::BTreeSet;
use veloq_core::tabular::{TabularView, cell_opt, push_count_meta};

use crate::disasm::DisasmResponse;
use crate::inspect::{InspectResponse, LaunchDetailsRow};
use crate::launches::{LaunchRow, LaunchesResponse};
use crate::lists::{GraphsResponse, RangesResponse, SourceRow, SourcesResponse, WorkloadRow};
use crate::metrics::{MetricRow, MetricsResponse};
use crate::native::NativeSummaryResponse;
use crate::source_metrics::{SourceMetricsResponse, SourceMetricsRow};
use crate::warp_stalls::{WarpStallsResponse, WarpStallsRow};

fn format_float(value: f64) -> String {
    format!("{value:.3}")
}

fn fmt_xyz(v: [u64; 3]) -> String {
    let [x, y, z] = v;
    format!("({x}, {y}, {z})")
}

// ---- detail-verb projectors -------------------------------------------
//
// Column rule: header column names equal the keys present in a serde-
// serialized `data.rows[0]`, with nested Objects expanded one level via
// dotted keys (`source.file`, `source.line`, `sass_address_range.start`)
// and BTreeMap fields expanded to one column per resolved key (BTreeMap
// iteration order preserved). Arrays of scalars stay as a single
// semicolon-joined column. See the cross-format
// consistency AC and the smoke test in `tests/ncu_tabular_smoke.rs`.

fn join_u64s(xs: &[u64]) -> String {
    xs.iter().map(u64::to_string).collect::<Vec<_>>().join(";")
}

fn render_json_scalar(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => String::new(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn cell_f64_opt(value: Option<f64>) -> String {
    value.map(format_float).unwrap_or_default()
}

/// CSV / table projection of the native `ncu summary`.
///
/// Renders the launch-derived totals row plus the NCU-version-only
/// session as a long `section / key / value` table. `count` /
/// `total_matched` / `ncu_version` ride the command-metadata header.
pub fn native_summary_view(resp: &NativeSummaryResponse) -> TabularView {
    let mut view = TabularView::new(["section", "key", "value"]);
    push_count_meta(&mut view, resp.count, resp.total_matched);
    view.push_meta("ncu_version", resp.auxiliary.ncu_version.clone());

    for row in &resp.rows {
        let t = &row.totals;
        let totals: [(&str, usize); 6] = [
            ("launch_count", t.launch_count),
            ("range_count", t.range_count),
            ("graph_count", t.graph_count),
            ("metric_count", t.metric_count),
            ("rule_count", t.rule_count),
            ("kernel_disasm_count", t.kernel_disasm_count),
        ];
        for (key, value) in totals {
            view.push_row(vec![row.key.clone(), key.to_string(), value.to_string()]);
        }
    }

    for version in &resp.auxiliary.session.versions {
        view.push_row(vec![
            "version".to_string(),
            version.provider.clone(),
            version.version.clone(),
        ]);
    }

    view
}

// ranges / graphs — native non-KERNEL workloads. Both share the
// `WorkloadRow` shape.

fn workload_columns() -> [&'static str; 8] {
    [
        "key",
        "row_id",
        "name",
        "context_id",
        "device_id",
        "stream_id",
        "metric_count",
        "rule_count",
    ]
}

fn workload_row_cells(r: &WorkloadRow) -> Vec<String> {
    vec![
        r.key.clone(),
        r.row_id.clone(),
        r.name.clone(),
        cell_opt(r.context_id),
        cell_opt(r.device_id),
        cell_opt(r.stream_id),
        r.metric_count.to_string(),
        r.rule_count.to_string(),
    ]
}

pub fn ranges_view(resp: &RangesResponse) -> TabularView {
    let mut view = TabularView::new(workload_columns());
    push_count_meta(&mut view, resp.count, resp.total_matched);
    for r in &resp.rows {
        view.push_row(workload_row_cells(r));
    }
    view
}

pub fn graphs_view(resp: &GraphsResponse) -> TabularView {
    let mut view = TabularView::new(workload_columns());
    push_count_meta(&mut view, resp.count, resp.total_matched);
    for r in &resp.rows {
        view.push_row(workload_row_cells(r));
    }
    view
}

pub fn sources_view(resp: &SourcesResponse) -> TabularView {
    let mut view = TabularView::new([
        "key",
        "row_id",
        "cuda_sm_name",
        "embedded_source_file_count",
        "has_disasm",
    ]);
    push_count_meta(&mut view, resp.count, resp.total_matched);
    for r in &resp.rows {
        view.push_row(source_row_cells(r));
    }
    view
}

fn source_row_cells(r: &SourceRow) -> Vec<String> {
    vec![
        r.key.clone(),
        r.row_id.clone(),
        r.cuda_sm_name.clone().unwrap_or_default(),
        r.embedded_source_file_count.to_string(),
        r.has_disasm.to_string(),
    ]
}

// launches / metrics / disasm — Phase 2 (nested rows; dotted-key
// flattening for one level).

pub fn launches_view(resp: &LaunchesResponse) -> TabularView {
    // `grid_size` / `block_size` serialize as JSON arrays of three
    // u64s. The cross-format consistency contract treats arrays as
    // single keys, so we keep them as one column each (rendered as
    // `(x, y, z)` for readability) rather than dotted keys.
    let mut view = TabularView::new([
        "key",
        "row_id",
        "kernel_demangled",
        "kernel_mangled",
        "grid_size",
        "block_size",
        "context_id",
        "device_id",
        "stream_id",
        "nvtx_range_path",
    ]);
    push_count_meta(&mut view, resp.count, resp.total_matched);
    for r in &resp.rows {
        view.push_row(launch_row_cells(r));
    }
    view
}

fn launch_row_cells(r: &LaunchRow) -> Vec<String> {
    vec![
        r.key.clone(),
        r.row_id.clone(),
        r.kernel_demangled.clone(),
        r.kernel_mangled.clone(),
        fmt_xyz(r.grid_size),
        fmt_xyz(r.block_size),
        cell_opt(r.context_id),
        cell_opt(r.device_id),
        cell_opt(r.stream_id),
        r.nvtx_range_path.clone().unwrap_or_default(),
    ]
}

pub fn metrics_view(resp: &MetricsResponse) -> TabularView {
    match resp.rows.first() {
        Some(MetricRow::Long(_)) | None => metrics_long_view(resp),
        Some(MetricRow::PerLaunch(_)) => metrics_wide_view(resp),
    }
}

fn metrics_long_view(resp: &MetricsResponse) -> TabularView {
    let mut view = TabularView::new([
        "key",
        "launch_row_id",
        "counter_name",
        "value",
        "unit",
        "value_type",
    ]);
    push_count_meta(&mut view, resp.count, resp.total_matched);
    for row in &resp.rows {
        let MetricRow::Long(r) = row else {
            // Mixed formats can't happen — `format` is set once per
            // request and every row matches it. Skip defensively
            // rather than panic so a future bug doesn't take out the
            // tabular path.
            continue;
        };
        view.push_row(vec![
            r.key.clone(),
            r.launch_row_id.clone(),
            r.counter_name.clone(),
            render_json_scalar(&r.value),
            r.unit.clone().unwrap_or_default(),
            r.value_type.clone().unwrap_or_default(),
        ]);
    }
    view
}

fn metrics_wide_view(resp: &MetricsResponse) -> TabularView {
    // Counter set is the union of every row's BTreeMap keys; in
    // practice every row's set is identical (same --counter glob
    // filters all launches) but we union defensively so a stray
    // missing-from-one-launch counter still appears as an empty
    // cell rather than silently dropping the column.
    let mut counter_names: BTreeSet<String> = BTreeSet::new();
    for row in &resp.rows {
        if let MetricRow::PerLaunch(r) = row {
            for name in r.counters.keys() {
                counter_names.insert(name.clone());
            }
        }
    }

    let mut columns: Vec<String> = vec![
        "key".to_string(),
        "row_id".to_string(),
        "kernel_demangled".to_string(),
    ];
    columns.extend(counter_names.iter().map(|name| format!("counters.{name}")));
    let mut view = TabularView::new(columns);
    push_count_meta(&mut view, resp.count, resp.total_matched);

    for row in &resp.rows {
        let MetricRow::PerLaunch(r) = row else {
            continue;
        };
        let mut cells: Vec<String> =
            vec![r.key.clone(), r.row_id.clone(), r.kernel_demangled.clone()];
        for name in &counter_names {
            cells.push(
                r.counters
                    .get(name)
                    .map(render_json_scalar)
                    .unwrap_or_default(),
            );
        }
        view.push_row(cells);
    }
    view
}

pub fn disasm_view(resp: &DisasmResponse) -> TabularView {
    // disasm's `data.rows[]` carries one `KernelDisasm` per row, and
    // each kernel owns a `Vec<SassInstruction>` — the natural
    // spreadsheet view is one CSV row per SASS instruction with
    // parent-kernel identity columns denormalized. Header columns
    // are the union of the per-kernel scalar fields + the per-
    // instruction fields (one nesting level of `source.*`); the
    // `instructions` Vec itself isn't a column because the rows
    // are the expansion.
    let mut view = TabularView::new([
        "kernel_key",
        "kernel_function_name",
        "kernel_start",
        "kernel_length",
        "address",
        "opcode",
        "operands",
        "predicate",
        "control_flow",
        "source.file",
        "source.line",
        "source.column",
    ]);
    push_count_meta(&mut view, resp.count, resp.total_matched);
    view.push_meta("kernel_count", resp.rows.len().to_string());
    for kernel in &resp.rows {
        for ins in &kernel.instructions {
            let (file, line, column) = match ins.source.as_ref() {
                Some(s) => (s.file.clone(), s.line.to_string(), cell_opt(s.column)),
                None => (String::new(), String::new(), String::new()),
            };
            view.push_row(vec![
                kernel.key.clone(),
                kernel.function_name.clone(),
                kernel.start.to_string(),
                kernel.length.to_string(),
                ins.address.to_string(),
                ins.opcode.clone(),
                ins.operands.clone(),
                ins.predicate.clone().unwrap_or_default(),
                ins.control_flow.to_string(),
                file,
                line,
                column,
            ]);
        }
    }
    view
}

// inspect / source-metrics — Phase 3 (heterogeneous rows).

pub fn inspect_view(resp: &InspectResponse) -> TabularView {
    // Heterogeneous-row strategy: emit one wide CSV with every column
    // either variant could carry. NULL (empty string) cells mark
    // fields the variant doesn't have. The `type` tag is the
    // serde-emitted discriminator (`launch` | `not_found`) so an
    // agent can branch on it without re-parsing.
    let mut view = TabularView::new([
        "type",
        "key",
        "row_id",
        "reason",
        "kernel_demangled",
        "kernel_mangled",
        "kernel_function",
        "grid_size",
        "block_size",
        "context_id",
        "device_id",
        "stream_id",
        "cubin_load_base",
        "metric_count",
        "rule_count",
        "has_disasm",
    ]);
    push_count_meta(&mut view, resp.count, resp.total_matched);
    for row in &resp.rows {
        match row {
            LaunchDetailsRow::Launch(launch) => {
                view.push_row(vec![
                    "launch".to_string(),
                    launch.key.clone(),
                    launch.row_id.clone(),
                    String::new(),
                    launch.kernel_demangled.clone(),
                    launch.kernel_mangled.clone(),
                    launch.kernel_function.clone(),
                    fmt_xyz(launch.grid_size),
                    fmt_xyz(launch.block_size),
                    cell_opt(launch.context_id),
                    cell_opt(launch.device_id),
                    cell_opt(launch.stream_id),
                    cell_opt(launch.cubin_load_base),
                    launch.metric_count.to_string(),
                    launch.rule_count.to_string(),
                    launch.has_disasm.to_string(),
                ]);
            }
            LaunchDetailsRow::NotFound {
                key,
                row_id,
                reason,
            } => {
                let mut cells = vec![
                    "not_found".to_string(),
                    key.clone(),
                    row_id.clone(),
                    reason.clone(),
                ];
                cells.resize(16, String::new());
                view.push_row(cells);
            }
        }
    }
    view
}

pub fn source_metrics_view(resp: &SourceMetricsResponse) -> TabularView {
    match resp.axis {
        "line" => source_metrics_line_view(resp),
        "sass" => source_metrics_sass_view(resp),
        "file" => source_metrics_file_view(resp),
        // Shouldn't happen — `axis` is set from the verb's `--by`
        // parser, which only ever emits one of the three labels. Fall
        // back to a trivially empty view so a future axis addition
        // surfaces as a clearly-missing projector instead of a panic.
        _ => TabularView::new(["axis", "note"]),
    }
}

fn source_metrics_counter_keys(resp: &SourceMetricsResponse) -> Vec<String> {
    let mut names: BTreeSet<String> = BTreeSet::new();
    for row in &resp.rows {
        match row {
            SourceMetricsRow::Line(r) => {
                for name in r.counters.keys() {
                    names.insert(name.clone());
                }
            }
            SourceMetricsRow::Sass(r) => {
                for name in r.counters.keys() {
                    names.insert(name.clone());
                }
            }
            SourceMetricsRow::File(r) => {
                for name in r.counters.keys() {
                    names.insert(name.clone());
                }
            }
        }
    }
    names.into_iter().collect()
}

fn source_metrics_line_view(resp: &SourceMetricsResponse) -> TabularView {
    let counter_names = source_metrics_counter_keys(resp);
    let mut columns: Vec<String> = vec![
        "key".to_string(),
        "launch_row_id".to_string(),
        "file".to_string(),
        "line".to_string(),
        "sass_addresses".to_string(),
        "sass_address_range.start".to_string(),
        "sass_address_range.end".to_string(),
        "sass_count".to_string(),
    ];
    for name in &counter_names {
        columns.push(format!("counters.{name}"));
    }
    for name in &counter_names {
        columns.push(format!("counter_coverage.{name}"));
    }
    let mut view = TabularView::new(columns);
    push_count_meta(&mut view, resp.count, resp.total_matched);
    view.push_meta("axis", "line");
    for row in &resp.rows {
        let SourceMetricsRow::Line(r) = row else {
            continue;
        };
        let mut cells: Vec<String> = vec![
            r.key.clone(),
            r.launch_row_id.clone(),
            r.file.clone(),
            r.line.to_string(),
            join_u64s(&r.sass_addresses),
            r.sass_address_range.start.to_string(),
            r.sass_address_range.end.to_string(),
            r.sass_count.to_string(),
        ];
        for name in &counter_names {
            cells.push(cell_f64_opt(r.counters.get(name).copied().flatten()));
        }
        for name in &counter_names {
            cells.push(
                r.counter_coverage
                    .get(name)
                    .map(|v| v.to_string())
                    .unwrap_or_default(),
            );
        }
        view.push_row(cells);
    }
    view
}

fn source_metrics_sass_view(resp: &SourceMetricsResponse) -> TabularView {
    let counter_names = source_metrics_counter_keys(resp);
    let mut columns: Vec<String> = vec![
        "key".to_string(),
        "launch_row_id".to_string(),
        "address".to_string(),
        "opcode".to_string(),
        "operands".to_string(),
        "source.file".to_string(),
        "source.line".to_string(),
        "source.column".to_string(),
    ];
    for name in &counter_names {
        columns.push(format!("counters.{name}"));
    }
    for name in &counter_names {
        columns.push(format!("counter_coverage.{name}"));
    }
    let mut view = TabularView::new(columns);
    push_count_meta(&mut view, resp.count, resp.total_matched);
    view.push_meta("axis", "sass");
    for row in &resp.rows {
        let SourceMetricsRow::Sass(r) = row else {
            continue;
        };
        let (file, line, column) = match r.source.as_ref() {
            Some(s) => (s.file.clone(), s.line.to_string(), cell_opt(s.column)),
            None => (String::new(), String::new(), String::new()),
        };
        let mut cells: Vec<String> = vec![
            r.key.clone(),
            r.launch_row_id.clone(),
            r.address.to_string(),
            r.opcode.clone(),
            r.operands.clone(),
            file,
            line,
            column,
        ];
        for name in &counter_names {
            cells.push(cell_f64_opt(r.counters.get(name).copied().flatten()));
        }
        for name in &counter_names {
            cells.push(
                r.counter_coverage
                    .get(name)
                    .map(|v| v.to_string())
                    .unwrap_or_default(),
            );
        }
        view.push_row(cells);
    }
    view
}

fn source_metrics_file_view(resp: &SourceMetricsResponse) -> TabularView {
    let counter_names = source_metrics_counter_keys(resp);
    let mut columns: Vec<String> = vec![
        "key".to_string(),
        "launch_row_id".to_string(),
        "file".to_string(),
        "line_count".to_string(),
        "sass_count".to_string(),
    ];
    for name in &counter_names {
        columns.push(format!("counters.{name}"));
    }
    for name in &counter_names {
        columns.push(format!("counter_coverage.{name}"));
    }
    let mut view = TabularView::new(columns);
    push_count_meta(&mut view, resp.count, resp.total_matched);
    view.push_meta("axis", "file");
    for row in &resp.rows {
        let SourceMetricsRow::File(r) = row else {
            continue;
        };
        let mut cells: Vec<String> = vec![
            r.key.clone(),
            r.launch_row_id.clone(),
            r.file.clone(),
            r.line_count.to_string(),
            r.sass_count.to_string(),
        ];
        for name in &counter_names {
            cells.push(cell_f64_opt(r.counters.get(name).copied().flatten()));
        }
        for name in &counter_names {
            cells.push(
                r.counter_coverage
                    .get(name)
                    .map(|v| v.to_string())
                    .unwrap_or_default(),
            );
        }
        view.push_row(cells);
    }
    view
}

/// `ncu warp-stalls` console projection. Long format —
/// one `(key, reason, samples)` row per stall reason at each location —
/// so the table stays jq-symmetric across the line / sass / reason
/// axes. JSON remains the full payload (the per-row `stalls` map).
pub fn warp_stalls_view(resp: &WarpStallsResponse) -> TabularView {
    let mut view = TabularView::new(["key", "reason", "samples"]);
    push_count_meta(&mut view, resp.count, resp.total_matched);
    view.push_meta("axis", resp.axis.clone());
    view.push_meta("total_samples", resp.auxiliary.total_samples.to_string());
    view.push_meta(
        "not_issued_samples",
        resp.auxiliary.not_issued_samples.to_string(),
    );
    view.push_meta(
        "unattributed_samples",
        resp.auxiliary.unattributed_samples.to_string(),
    );
    view.push_meta(
        "out_of_cubin_samples",
        resp.auxiliary.out_of_cubin_samples.to_string(),
    );
    for row in &resp.rows {
        match row {
            WarpStallsRow::Line(r) => {
                for (reason, count) in &r.stalls {
                    view.push_row(vec![r.key.clone(), reason.clone(), count.to_string()]);
                }
            }
            WarpStallsRow::Sass(r) => {
                for (reason, count) in &r.stalls {
                    view.push_row(vec![r.key.clone(), reason.clone(), count.to_string()]);
                }
            }
            WarpStallsRow::Reason(r) => {
                view.push_row(vec![
                    r.key.clone(),
                    r.reason.clone(),
                    r.total_samples.to_string(),
                ]);
            }
        }
    }
    view
}
