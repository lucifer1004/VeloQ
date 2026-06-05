//! Per-axis aggregation for `ncu source-metrics`.
//!
//! Pure module — takes hand-built `Vec<MetricInstance>` +
//! `KernelDisasm` literals so it's unit-testable without an
//! `.ncu-rep` fixture.
//!
//! ## Semantic distinctions on `counters: BTreeMap<String, Option<f64>>`
//!
//! Invariants:
//!
//! - `Some(v)` — at least one instance for this counter overlapped
//!   the row's SASS addresses; `v` is the SUM (additive case) or
//!   the per-PC value (sass axis identity). `v == 0.0` means
//!   "summed to exactly zero" and is distinct from `None`.
//! - `None` — no instance for this counter overlapped any of the
//!   row's SASS addresses, but the row exists because *another*
//!   counter matched.
//!
//! `counter_coverage: BTreeMap<String, u32>` carries the per-row,
//! per-counter count of distinct SASS addresses that had ≥1 instance.
//! Same key set as `counters`.
//!
//! ## Unattributed-instance accounting
//!
//! In-cubin instances landing on `source: None`
//! SASS addresses (DWARF holes / compiler-inserted code) feed
//! `unattributed_sass_counter_totals`; instances whose
//! `correlation_id` falls outside the cubin range feed
//! `out_of_cubin_counter_totals`. Both are per-counter, both are
//! dropped from `data.rows[]` — they exist only in the auxiliary
//! block so an agent reconciling against the kernel-level scalar
//! from `ncu metrics` can find the missing budget.

use crate::disasm_pipeline::SourceLineRef;
use crate::native::{NativeInsn, NativeInstance, NativeMetric, Placement};
use std::collections::BTreeMap;

/// Which axis the rollup should produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    Line,
    Sass,
    File,
}

/// One source-line-aggregated row per (file, line) covered by the
/// matched counters. `counters` / `counter_coverage` always share a
/// key set.
#[derive(Debug, Clone)]
pub struct LineRow {
    pub file: String,
    pub line: u32,
    pub sass_addresses: Vec<u64>,
    pub counters: BTreeMap<String, Option<f64>>,
    pub counter_coverage: BTreeMap<String, u32>,
}

/// One row per (launch, cubin-relative SASS address). `source` is
/// `None` for DWARF holes / compiler-inserted code; `counters` /
/// `counter_coverage` always share a key set.
#[derive(Debug, Clone)]
pub struct SassRow {
    pub address: u64,
    pub opcode: String,
    pub operands: String,
    pub source: Option<SourceLineRef>,
    pub counters: BTreeMap<String, Option<f64>>,
    pub counter_coverage: BTreeMap<String, u32>,
}

/// Per-file aggregate of every (file, line) row belonging to the
/// file. `line_count` and `sass_count` are denominator hints for
/// agents that want to weight the file-level total.
#[derive(Debug, Clone)]
pub struct FileRow {
    pub file: String,
    pub line_count: u32,
    pub sass_count: u32,
    pub counters: BTreeMap<String, Option<f64>>,
    pub counter_coverage: BTreeMap<String, u32>,
}

/// Per-counter unattributed totals — instances dropped from
/// `data.rows[]` by the unattributed accounting.
#[derive(Debug, Clone, Default)]
pub struct UnattributedTotals {
    /// Per-counter sum of values for instances landing in-cubin on a
    /// SASS instruction with `source: None`.
    pub unattributed_sass: BTreeMap<String, f64>,
    /// Per-counter sum of values for instances whose correlation_id
    /// fell outside `[cubin_load_base, cubin_load_base + length)`.
    pub out_of_cubin: BTreeMap<String, f64>,
    /// Count of distinct in-cubin-but-source-None instances summed
    /// into `unattributed_sass` across every counter.
    pub unattributed_sass_count: u32,
    /// Count of distinct out-of-cubin instances summed into
    /// `out_of_cubin` across every counter.
    pub out_of_cubin_instance_count: u32,
}

/// A counter the caller requested via `--counter` paired with the
/// additivity classification. Caller computes additivity from the
/// metric name once (see `super::additivity::is_additive`) and
/// passes the result here so this module stays purely about
/// aggregation.
#[derive(Debug, Clone, Copy)]
pub struct CounterSpec<'a> {
    pub name: &'a str,
    pub additive: bool,
}

/// Input to a rollup call: every matched counter's native
/// [`NativeMetric`] alongside its name + additivity classification, and
/// the launch's SASS listing (for source-line resolution via
/// `ncu_report`'s `source_info`). The per-PC placement
/// (`attributed` / `in_cubin_no_source` / `out_of_cubin`) is carried on
/// each instance.
pub struct RollupInput<'a> {
    pub counters: &'a [(CounterSpec<'a>, &'a NativeMetric)],
    pub disasm: &'a [NativeInsn],
}

/// Output of a rollup call. Caller projects this into the wire shape
/// per axis.
pub struct RollupOutput {
    pub line_rows: Vec<LineRow>,
    pub sass_rows: Vec<SassRow>,
    pub file_rows: Vec<FileRow>,
    pub unattributed: UnattributedTotals,
}

/// Run the rollup. The result carries values for all three axes;
/// the caller picks the appropriate one based on `--by`. The cost
/// of producing the unused axes is bounded by `Σ(instances)` — small
/// in practice and lets us keep `rollup()` axis-agnostic for tests.
///
/// Note: file rows are derived from line rows post-hoc so the
/// additivity gate is enforced once.
pub fn rollup(input: RollupInput<'_>) -> RollupOutput {
    let counter_names: Vec<String> = input
        .counters
        .iter()
        .map(|(spec, _)| spec.name.to_string())
        .collect();

    // Build a SASS-address → source ref map once; index every disasm
    // instruction by its cubin-relative address. `source` is
    // `ncu_report`'s `source_info`, which is authoritative per-PC;
    // nvdisasm's `--print-line-info` map mis-attributes across
    // multi-kernel cubins (both `.text` sections are 0-based).
    let mut addr_to_source: BTreeMap<u64, Option<SourceLineRef>> = BTreeMap::new();
    let mut addr_to_instruction: BTreeMap<u64, (&str, &str)> = BTreeMap::new();
    for inst in input.disasm {
        addr_to_source.insert(
            inst.address,
            inst.source.as_ref().map(|s| SourceLineRef {
                file: s.file.clone(),
                line: s.line,
                column: None,
            }),
        );
        addr_to_instruction.insert(inst.address, (inst.opcode.as_str(), inst.operands.as_str()));
    }

    // For each counter, walk its instances once and bucket them by:
    //   - source-attributed (in cubin, source present) → contributes
    //     to line/sass/file rows.
    //   - in-cubin-no-source → unattributed_sass + sass rows only.
    //   - out-of-cubin → out_of_cubin totals only.
    //
    // Aggregations are computed into intermediate maps so we can
    // emit the BTreeMap-ordered final rows in one pass.

    // Per (file, line) → per-counter (sum, set of SASS addrs hit).
    let mut line_acc: BTreeMap<(String, u32), LineAccumulator> = BTreeMap::new();
    // Per SASS address → (source, per-counter optional value).
    let mut sass_acc: BTreeMap<u64, SassAccumulator> = BTreeMap::new();
    let mut unattributed = UnattributedTotals::default();

    for (spec, entry) in input.counters {
        let Some(instances) = entry.instances.as_ref() else {
            continue;
        };
        for inst in instances {
            if inst.placement == Placement::OutOfCubin {
                // Out of cubin: feed out_of_cubin_counter_totals only.
                if let Some(v) = instance_value_f64(inst) {
                    *unattributed
                        .out_of_cubin
                        .entry(spec.name.to_string())
                        .or_insert(0.0) += v;
                    unattributed.out_of_cubin_instance_count =
                        unattributed.out_of_cubin_instance_count.saturating_add(1);
                }
                continue;
            }
            // In-cubin (attributed | in_cubin_no_source): the helper
            // pre-tagged it, so `rel_address` is the cubin-relative PC.
            let Some(rel) = inst.rel_address else {
                continue;
            };
            let Some(value) = instance_value_f64(inst) else {
                continue;
            };

            // Sass-axis row always gets the per-PC value verbatim,
            // even for unattributed (source: None) PCs. Identity
            // rollup — additivity gate does NOT apply to the sass
            // axis.
            let source = addr_to_source.get(&rel).cloned().unwrap_or(None);
            let (opcode, operands) = addr_to_instruction.get(&rel).copied().unwrap_or(("", ""));
            let sass_entry = sass_acc.entry(rel).or_insert_with(|| SassAccumulator {
                source: source.clone(),
                opcode: opcode.to_string(),
                operands: operands.to_string(),
                values: BTreeMap::new(),
            });
            sass_entry
                .values
                .entry(spec.name.to_string())
                .or_insert(Some(value));

            // Line-axis row only materialises when (a) the counter
            // is additive AND (b) the SASS address has a source ref.
            // Otherwise the instance lands in unattributed.
            if !spec.additive {
                continue;
            }
            match source {
                Some(src) => {
                    let key = (src.file.clone(), src.line);
                    let acc = line_acc.entry(key).or_default();
                    let counter_acc = acc.counters.entry(spec.name.to_string()).or_default();
                    counter_acc.sum += value;
                    counter_acc.addrs.insert(rel);
                    acc.all_addrs.insert(rel);
                }
                None => {
                    *unattributed
                        .unattributed_sass
                        .entry(spec.name.to_string())
                        .or_insert(0.0) += value;
                    unattributed.unattributed_sass_count =
                        unattributed.unattributed_sass_count.saturating_add(1);
                }
            }
        }
    }

    // Project line accumulator → LineRow with every counter name
    // represented (null when the counter had no instances on the
    // row). Maintain the row-existence invariant: a row exists iff
    // at least one counter had at least one instance.
    let line_rows: Vec<LineRow> = line_acc
        .into_iter()
        .map(|((file, line), acc)| {
            let mut counters = BTreeMap::new();
            let mut counter_coverage = BTreeMap::new();
            for name in &counter_names {
                match acc.counters.get(name) {
                    Some(c) => {
                        counters.insert(name.clone(), Some(c.sum));
                        counter_coverage.insert(name.clone(), c.addrs.len() as u32);
                    }
                    None => {
                        counters.insert(name.clone(), None);
                        counter_coverage.insert(name.clone(), 0);
                    }
                }
            }
            let mut sass_addresses: Vec<u64> = acc.all_addrs.into_iter().collect();
            sass_addresses.sort_unstable();
            sass_addresses.dedup();
            LineRow {
                file,
                line,
                sass_addresses,
                counters,
                counter_coverage,
            }
        })
        .collect();

    // Project sass accumulator → SassRow. Same null-vs-value rule
    // per (row, counter). Sass rows include in-cubin PCs even when
    // source is None.
    let sass_rows: Vec<SassRow> = sass_acc
        .into_iter()
        .map(|(addr, acc)| {
            let mut counters = BTreeMap::new();
            let mut counter_coverage = BTreeMap::new();
            for name in &counter_names {
                match acc.values.get(name) {
                    Some(v) => {
                        counters.insert(name.clone(), *v);
                        // Identity axis: 1 if the counter had an
                        // instance at this address, else 0.
                        counter_coverage.insert(name.clone(), 1);
                    }
                    None => {
                        counters.insert(name.clone(), None);
                        counter_coverage.insert(name.clone(), 0);
                    }
                }
            }
            SassRow {
                address: addr,
                opcode: acc.opcode,
                operands: acc.operands,
                source: acc.source,
                counters,
                counter_coverage,
            }
        })
        .collect();

    // File rows aggregate the line rows additively. counter_coverage
    // on a file row counts the UNION of SASS addresses contributing
    // to that counter across the file's lines.
    let mut file_acc: BTreeMap<String, FileAccumulator> = BTreeMap::new();
    for line_row in &line_rows {
        let acc = file_acc.entry(line_row.file.clone()).or_default();
        acc.line_count = acc.line_count.saturating_add(1);
        acc.sass_addrs
            .extend(line_row.sass_addresses.iter().copied());
        for (name, value) in &line_row.counters {
            // Coverage union per counter.
            if let Some(cov) = line_row.counter_coverage.get(name)
                && *cov > 0
            {
                acc.counters.entry(name.clone()).or_insert_with(|| (0.0, 0));
            }
            // Sum the additive contributions. Lines where the
            // counter is null (no overlap) don't contribute.
            if let Some(v) = value {
                let entry = acc.counters.entry(name.clone()).or_insert((0.0, 0));
                entry.0 += v;
                // Approximate coverage union by summing per-line
                // coverages; lines never share a SASS address so
                // the union equals the sum.
                let line_cov = line_row.counter_coverage.get(name).copied().unwrap_or(0);
                entry.1 = entry.1.saturating_add(line_cov);
            }
        }
    }
    let file_rows: Vec<FileRow> = file_acc
        .into_iter()
        .map(|(file, acc)| {
            let mut counters = BTreeMap::new();
            let mut counter_coverage = BTreeMap::new();
            for name in &counter_names {
                match acc.counters.get(name) {
                    Some((sum, cov)) => {
                        counters.insert(name.clone(), Some(*sum));
                        counter_coverage.insert(name.clone(), *cov);
                    }
                    None => {
                        counters.insert(name.clone(), None);
                        counter_coverage.insert(name.clone(), 0);
                    }
                }
            }
            FileRow {
                file,
                line_count: acc.line_count,
                sass_count: acc.sass_addrs.len() as u32,
                counters,
                counter_coverage,
            }
        })
        .collect();

    RollupOutput {
        line_rows,
        sass_rows,
        file_rows,
        unattributed,
    }
}

/// Drop a row from `rows` when every counter cell is `None`. The
/// row-existence invariant is "exists iff ≥1
/// counter matched on this row" — the accumulator never inserts an
/// all-null row, but a future caller might filter rows post-hoc
/// (e.g. via `--file`) and accidentally produce one. Cheap defensive
/// pass.
pub fn drop_all_null_rows<R>(rows: Vec<R>, all_null: impl Fn(&R) -> bool) -> Vec<R> {
    rows.into_iter().filter(|r| !all_null(r)).collect()
}

// ---- internal accumulators ----------------------------------------------

#[derive(Default)]
struct LineAccumulator {
    counters: BTreeMap<String, CounterLineAcc>,
    all_addrs: std::collections::BTreeSet<u64>,
}

#[derive(Default)]
struct CounterLineAcc {
    sum: f64,
    addrs: std::collections::BTreeSet<u64>,
}

struct SassAccumulator {
    source: Option<SourceLineRef>,
    opcode: String,
    operands: String,
    values: BTreeMap<String, Option<f64>>,
}

#[derive(Default)]
struct FileAccumulator {
    line_count: u32,
    sass_addrs: std::collections::BTreeSet<u64>,
    /// `name → (sum, summed_coverage)`. Coverage is summed across
    /// lines (lines never share SASS addresses so this equals the
    /// union).
    counters: BTreeMap<String, (f64, u32)>,
}

fn instance_value_f64(inst: &NativeInstance) -> Option<f64> {
    // Accept the numeric union {uint64, uint32, double,
    // float}. serde_json::Value::as_f64 covers all four.
    inst.value.as_f64()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::{NativeInsn, NativeInstance, NativeSourceRef};
    use serde_json::json;

    /// In-cubin `attributed` instance at cubin-relative `addr`. Whether
    /// it lands on a line row vs `unattributed_sass` is decided by the
    /// disasm's `source` at `addr`.
    fn inst_at(load_base: u64, addr: u64, value: f64) -> NativeInstance {
        NativeInstance {
            correlation_id: load_base + addr,
            rel_address: Some(addr),
            value: json!(value),
            placement: Placement::Attributed,
        }
    }

    /// Out-of-cubin instance (placement `out_of_cubin`, no
    /// `rel_address`) — what the helper tags for non-VA correlations.
    fn inst_at_runtime_va(va: u64, value: f64) -> NativeInstance {
        NativeInstance {
            correlation_id: va,
            rel_address: None,
            value: json!(value),
            placement: Placement::OutOfCubin,
        }
    }

    fn metric_entry(name: &str, instances: Vec<NativeInstance>) -> NativeMetric {
        NativeMetric {
            name: name.to_string(),
            label: None,
            unit: None,
            value: json!(0.0),
            value_type: "double".to_string(),
            // `Other` => the additivity name-suffix fallback.
            metric_type: crate::native::MetricType::Other,
            metric_type_code: None,
            metric_subtype: None,
            metric_subtype_code: None,
            rollup: None,
            rollup_code: None,
            instances: Some(instances),
        }
    }

    fn sass_inst(addr: u64, source: Option<(&str, u32)>) -> NativeInsn {
        NativeInsn {
            address: addr,
            opcode: "LDS".to_string(),
            operands: "R0, [R1]".to_string(),
            source: source.map(|(f, l)| NativeSourceRef {
                file: f.to_string(),
                line: l,
            }),
        }
    }

    /// Identity passthrough (the rollup takes `&[NativeInsn]` directly).
    fn disasm_with(instructions: Vec<NativeInsn>) -> Vec<NativeInsn> {
        instructions
    }

    /// Motivating case for the additivity rule: a
    /// double-valued source counter named `derived__memory_l1_*`
    /// must roll up additively to f64. The earlier uint64-only
    /// heuristic would have dropped it.
    #[test]
    fn motivating_double_valued_metric_sums_per_line() -> anyhow::Result<()> {
        let base = 0x1000;
        // SASS instructions carry cubin-relative addresses; metric
        // instances carry runtime VAs (load_base + relative).
        let disasm = disasm_with(vec![
            sass_inst(0x40, Some(("synthetic.cu", 10))),
            sass_inst(0x80, Some(("synthetic.cu", 10))),
        ]);
        let metric = metric_entry(
            "derived__memory_l1_conflicts_shared_nway",
            vec![inst_at(base, 0x40, 32.0), inst_at(base, 0x80, 16.0)],
        );
        let spec = CounterSpec {
            name: "derived__memory_l1_conflicts_shared_nway",
            additive: true,
        };
        let out = rollup(RollupInput {
            counters: &[(spec, &metric)],
            disasm: &disasm,
        });
        assert_eq!(out.line_rows.len(), 1);
        let row = out
            .line_rows
            .first()
            .ok_or_else(|| anyhow::anyhow!("expected one line row"))?;
        assert_eq!(row.file, "synthetic.cu");
        assert_eq!(row.line, 10);
        let val = row
            .counters
            .get("derived__memory_l1_conflicts_shared_nway")
            .copied()
            .flatten()
            .ok_or_else(|| anyhow::anyhow!("expected counter to be non-null"))?;
        assert!((val - 48.0).abs() < 1e-9, "expected 48.0, got {val}");
        let cov = row
            .counter_coverage
            .get("derived__memory_l1_conflicts_shared_nway")
            .copied()
            .ok_or_else(|| anyhow::anyhow!("expected counter_coverage entry"))?;
        assert_eq!(cov, 2);
        assert_eq!(row.sass_addresses, vec![0x40, 0x80]);
        Ok(())
    }

    /// `0.0` (counter applied and summed to zero) is distinct from
    /// `null` (counter had no instances on the row). Both serialize
    /// to distinguishable JSON.
    #[test]
    fn null_vs_zero_serialize_distinctly() -> anyhow::Result<()> {
        let base = 0x1000;
        let disasm = disasm_with(vec![
            sass_inst(0x40, Some(("synthetic.cu", 10))),
            sass_inst(0x80, Some(("synthetic.cu", 10))),
        ]);
        // Counter A: instances on both PCs, both valued 0.0 (sums to 0.0).
        let a = metric_entry(
            "counter_a",
            vec![inst_at(base, 0x40, 0.0), inst_at(base, 0x80, 0.0)],
        );
        // Counter B: no instances on any PC of line 10. Row exists
        // only because counter A matched.
        let b = metric_entry("counter_b", vec![]);
        let out = rollup(RollupInput {
            counters: &[
                (
                    CounterSpec {
                        name: "counter_a",
                        additive: true,
                    },
                    &a,
                ),
                (
                    CounterSpec {
                        name: "counter_b",
                        additive: true,
                    },
                    &b,
                ),
            ],
            disasm: &disasm,
        });
        assert_eq!(out.line_rows.len(), 1);
        let row = out
            .line_rows
            .first()
            .ok_or_else(|| anyhow::anyhow!("expected one row"))?;
        // counter_a is Some(0.0); counter_b is None.
        let a = row
            .counters
            .get("counter_a")
            .ok_or_else(|| anyhow::anyhow!("missing counter_a"))?;
        let b = row
            .counters
            .get("counter_b")
            .ok_or_else(|| anyhow::anyhow!("missing counter_b"))?;
        assert!(
            matches!(a, Some(v) if *v == 0.0),
            "counter_a expected Some(0.0), got {a:?}"
        );
        assert!(b.is_none(), "counter_b expected None, got {b:?}");
        // Serialize and check JSON bytes are distinguishable.
        let json = serde_json::to_value(row.counters.clone())?;
        let s = serde_json::to_string(&json)?;
        assert!(
            s.contains("\"counter_a\":0.0"),
            "expected 0.0 literal; got {s}"
        );
        assert!(
            s.contains("\"counter_b\":null"),
            "expected null literal; got {s}"
        );
        Ok(())
    }

    /// Two counters on the same line with different SASS coverage:
    /// counter A on PCs P0+P1, counter B on P0 only. Both
    /// `counters[name]` are non-null; `counter_coverage` differs.
    #[test]
    fn multi_counter_coverage_divergence() -> anyhow::Result<()> {
        let base = 0x1000;
        let p0 = 0x40_u64;
        let p1 = 0x80_u64;
        let disasm = disasm_with(vec![
            sass_inst(p0, Some(("synthetic.cu", 10))),
            sass_inst(p1, Some(("synthetic.cu", 10))),
        ]);
        let a = metric_entry(
            "counter_a",
            vec![inst_at(base, p0, 3.0), inst_at(base, p1, 5.0)],
        );
        let b = metric_entry("counter_b", vec![inst_at(base, p0, 7.0)]);
        let out = rollup(RollupInput {
            counters: &[
                (
                    CounterSpec {
                        name: "counter_a",
                        additive: true,
                    },
                    &a,
                ),
                (
                    CounterSpec {
                        name: "counter_b",
                        additive: true,
                    },
                    &b,
                ),
            ],
            disasm: &disasm,
        });
        let row = out
            .line_rows
            .first()
            .ok_or_else(|| anyhow::anyhow!("expected one row"))?;
        let va = row.counters.get("counter_a").copied().flatten();
        let vb = row.counters.get("counter_b").copied().flatten();
        assert_eq!(va, Some(8.0));
        assert_eq!(vb, Some(7.0));
        assert_eq!(row.counter_coverage.get("counter_a").copied(), Some(2));
        assert_eq!(row.counter_coverage.get("counter_b").copied(), Some(1));
        Ok(())
    }

    /// `counters` and `counter_coverage` have identical ordered key
    /// sequences (BTreeMap serializes in `Ord` order).
    #[test]
    fn counters_and_coverage_share_ordered_key_set() -> anyhow::Result<()> {
        let base = 0x1000;
        let disasm = disasm_with(vec![sass_inst(0x40, Some(("synthetic.cu", 10)))]);
        let a = metric_entry("zzz_last", vec![inst_at(base, 0x40, 1.0)]);
        let b = metric_entry("aaa_first", vec![inst_at(base, 0x40, 2.0)]);
        // Pass in arbitrary order; BTreeMap will sort.
        let out = rollup(RollupInput {
            counters: &[
                (
                    CounterSpec {
                        name: "zzz_last",
                        additive: true,
                    },
                    &a,
                ),
                (
                    CounterSpec {
                        name: "aaa_first",
                        additive: true,
                    },
                    &b,
                ),
            ],
            disasm: &disasm,
        });
        let row = out
            .line_rows
            .first()
            .ok_or_else(|| anyhow::anyhow!("expected one row"))?;
        let counter_keys: Vec<&String> = row.counters.keys().collect();
        let coverage_keys: Vec<&String> = row.counter_coverage.keys().collect();
        assert_eq!(counter_keys, coverage_keys);
        // BTreeMap iterates alphabetical.
        assert_eq!(
            counter_keys,
            vec![&"aaa_first".to_string(), &"zzz_last".to_string()]
        );
        Ok(())
    }

    /// Identity axis: per-PC values verbatim, including for PCs
    /// whose `source` is None.
    #[test]
    fn by_sass_emits_unattributed_pc_with_null_source() -> anyhow::Result<()> {
        let base = 0x1000;
        let attributed = 0x40_u64;
        let unattributed = 0x80_u64;
        let disasm = disasm_with(vec![
            sass_inst(attributed, Some(("synthetic.cu", 10))),
            sass_inst(unattributed, None), // DWARF hole
        ]);
        let metric = metric_entry(
            "counter_a",
            vec![
                inst_at(base, attributed, 11.0),
                inst_at(base, unattributed, 13.0),
            ],
        );
        let out = rollup(RollupInput {
            counters: &[(
                CounterSpec {
                    name: "counter_a",
                    additive: true,
                },
                &metric,
            )],
            disasm: &disasm,
        });
        assert_eq!(out.sass_rows.len(), 2);
        let attr_row = out
            .sass_rows
            .iter()
            .find(|r| r.address == attributed)
            .ok_or_else(|| anyhow::anyhow!("missing attributed row"))?;
        let unattr_row = out
            .sass_rows
            .iter()
            .find(|r| r.address == unattributed)
            .ok_or_else(|| anyhow::anyhow!("missing unattributed row"))?;
        assert!(attr_row.source.is_some());
        assert!(unattr_row.source.is_none());
        assert_eq!(attr_row.opcode, "LDS");
        assert_eq!(attr_row.operands, "R0, [R1]");
        assert_eq!(
            attr_row.counters.get("counter_a").copied().flatten(),
            Some(11.0)
        );
        assert_eq!(
            unattr_row.counters.get("counter_a").copied().flatten(),
            Some(13.0)
        );
        Ok(())
    }

    /// Unattributed (in-cubin source:None) lands in
    /// `unattributed_sass_counter_totals`, not in any line row.
    /// Out-of-cubin lands in `out_of_cubin_counter_totals`, never in
    /// any row at all. The two never double-count.
    #[test]
    fn split_unattributed_accounting_no_double_count() -> anyhow::Result<()> {
        let base = 0x1000;
        let attributed: u64 = 0x40;
        let unattributed: u64 = 0x80;
        let out_of_cubin_va: u64 = base + 0x200; // outside cubin (runtime VA)
        let disasm = disasm_with(vec![
            sass_inst(attributed, Some(("synthetic.cu", 10))),
            sass_inst(unattributed, None),
        ]);
        let metric = metric_entry(
            "counter_a",
            vec![
                inst_at(base, attributed, 5.0),
                inst_at(base, unattributed, 7.0),
                inst_at_runtime_va(out_of_cubin_va, 11.0),
            ],
        );
        let out = rollup(RollupInput {
            counters: &[(
                CounterSpec {
                    name: "counter_a",
                    additive: true,
                },
                &metric,
            )],
            disasm: &disasm,
        });
        // Line row covers only the attributed PC.
        let line = out
            .line_rows
            .first()
            .ok_or_else(|| anyhow::anyhow!("expected line row"))?;
        assert_eq!(line.counters.get("counter_a").copied().flatten(), Some(5.0));
        // Unattributed sums = 7.0.
        assert_eq!(
            out.unattributed.unattributed_sass.get("counter_a").copied(),
            Some(7.0)
        );
        assert_eq!(out.unattributed.unattributed_sass_count, 1);
        // Out-of-cubin sums = 11.0.
        assert_eq!(
            out.unattributed.out_of_cubin.get("counter_a").copied(),
            Some(11.0)
        );
        assert_eq!(out.unattributed.out_of_cubin_instance_count, 1);
        // Reconciliation: 5 + 7 + 11 = 23 (kernel-level total
        // observable via `ncu metrics`).
        let line_total = line
            .counters
            .get("counter_a")
            .copied()
            .flatten()
            .unwrap_or(0.0);
        let unattr_total = out
            .unattributed
            .unattributed_sass
            .get("counter_a")
            .copied()
            .unwrap_or(0.0);
        let oob_total = out
            .unattributed
            .out_of_cubin
            .get("counter_a")
            .copied()
            .unwrap_or(0.0);
        assert!((line_total + unattr_total + oob_total - 23.0).abs() < 1e-9);
        Ok(())
    }

    /// Non-additive counters are filtered out of `--by line` /
    /// `--by file` (caller passes `additive: false`) but pass
    /// through verbatim on `--by sass`.
    #[test]
    fn non_additive_passes_through_sass_axis_only() -> anyhow::Result<()> {
        let base = 0x1000;
        let p0: u64 = 0x40;
        let p1: u64 = 0x80;
        let disasm = disasm_with(vec![
            sass_inst(p0, Some(("synthetic.cu", 10))),
            sass_inst(p1, Some(("synthetic.cu", 10))),
        ]);
        let metric = metric_entry(
            "smsp__some_ratio.ratio",
            vec![inst_at(base, p0, 0.4), inst_at(base, p1, 0.6)],
        );
        let spec = CounterSpec {
            name: "smsp__some_ratio.ratio",
            additive: false,
        };
        let out = rollup(RollupInput {
            counters: &[(spec, &metric)],
            disasm: &disasm,
        });
        // No line rows — additive: false suppresses line/file rollup.
        assert_eq!(out.line_rows.len(), 0);
        assert_eq!(out.file_rows.len(), 0);
        // Sass rows still carry per-PC values verbatim.
        assert_eq!(out.sass_rows.len(), 2);
        let r0 = out
            .sass_rows
            .iter()
            .find(|r| r.address == p0)
            .ok_or_else(|| anyhow::anyhow!("missing p0 row"))?;
        let r1 = out
            .sass_rows
            .iter()
            .find(|r| r.address == p1)
            .ok_or_else(|| anyhow::anyhow!("missing p1 row"))?;
        assert_eq!(
            r0.counters.get("smsp__some_ratio.ratio").copied().flatten(),
            Some(0.4)
        );
        assert_eq!(
            r1.counters.get("smsp__some_ratio.ratio").copied().flatten(),
            Some(0.6)
        );
        Ok(())
    }

    /// Per-file aggregate sums every (file, line) row's counter values.
    #[test]
    fn file_rollup_sums_line_rows() -> anyhow::Result<()> {
        let base = 0x1000;
        let disasm = disasm_with(vec![
            sass_inst(0x40, Some(("synthetic.cu", 10))),
            sass_inst(0x80, Some(("synthetic.cu", 11))),
        ]);
        let metric = metric_entry(
            "counter_a",
            vec![inst_at(base, 0x40, 3.0), inst_at(base, 0x80, 5.0)],
        );
        let out = rollup(RollupInput {
            counters: &[(
                CounterSpec {
                    name: "counter_a",
                    additive: true,
                },
                &metric,
            )],
            disasm: &disasm,
        });
        assert_eq!(out.file_rows.len(), 1);
        let row = out
            .file_rows
            .first()
            .ok_or_else(|| anyhow::anyhow!("expected one file row"))?;
        assert_eq!(row.file, "synthetic.cu");
        assert_eq!(row.line_count, 2);
        assert_eq!(row.sass_count, 2);
        assert_eq!(row.counters.get("counter_a").copied().flatten(), Some(8.0));
        assert_eq!(row.counter_coverage.get("counter_a").copied(), Some(2));
        Ok(())
    }
}
