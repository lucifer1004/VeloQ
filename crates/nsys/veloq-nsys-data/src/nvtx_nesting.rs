//! Nesting depth + per-(tid, domain, name) iteration index for NVTX events.
//!
//! NSys's `NVTX_EVENTS` is a flat table — every NVTX range is one row,
//! with no hint that range B starts while range A is still open, and no
//! hint that two same-named ranges on one thread are repeats of an
//! outer loop's iterations. For callers asking "which iteration step
//! does this kernel belong to" (the typical agent question), both
//! shapes matter:
//!
//! * `depth` is the *innermost* containing range's nesting depth: 0 for
//!   outermost ranges, 1 for ranges fully inside a single depth-0
//!   range, etc.
//! * `iter_index` is the 0-based ordinal of this range among
//!   same-`(global_tid, domain_id, name)` repeats, in start-time order
//!   — answers "step_0 / step_1 / step_2" directly even when the user
//!   gave every iteration the same `step` label.
//!
//! Both come from one rayon-parallel scan keyed by the table `rowid`.
//!
//! ## Algorithm
//!
//! Group rows by `(global_tid, domain_id)` for depth (distinct host
//! threads / NVTX domains produce independent stacks), and by
//! `(global_tid, domain_id, name)` for iter_index. Within each depth
//! group:
//!   1. sort by `start_ns`
//!   2. maintain a stack of currently-open ranges (their `end_ns`)
//!   3. at each event, pop ranges that ended before the event starts;
//!      the event's depth is the resulting stack size; push the event's
//!      `end_ns` if it has duration > 0 (instant markers don't nest).
//!
//! iter_index reuses the per-(tid,domain) sorted order — within each
//! such group we further bucket by name and assign 0-based ordinals.
//!
//! Groups are processed in parallel via `rayon`; events within a group
//! are stack-ordered and must run sequentially. The result is keyed
//! by the table `rowid` so downstream callers (`slices`, `search`,
//! `inspect`, the `nvtx_reverse` query) join back without
//! re-materialising the events.
//!
//! ## Disk caching
//!
//! The metadata sidecar stores the computed `i64 → NvtxEntry` map.
//! Callers should normally go through `Trace::nvtx_nesting()`, which
//! reuses an already-loaded or valid on-disk sidecar and falls back to
//! this direct computation only when the sidecar is absent.

use crate::Trace;
use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// One NVTX row's derived attributes — nesting depth + iteration index
/// among same-named repeats. Both fields are independent answers to
/// "what's this range's position in its enclosing structure" / "which
/// repeat of itself is this".
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct NvtxEntry {
    /// Stack-nesting depth: 0 = outermost. Instant markers
    /// (`end IS NULL`) get the active stack size at their `start_ns`
    /// but never push.
    pub depth: u8,
    /// 0-based ordinal among `(global_tid, domain_id, name)` repeats,
    /// in start-time order. `0` for the first occurrence; the same
    /// 0 when this name appears only once.
    pub iter_index: u32,
}

/// Per-NVTX-row derived attributes, keyed by table `rowid`. Empty when
/// the trace has no `NVTX_EVENTS` table.
pub type NvtxNesting = HashMap<i64, NvtxEntry>;

/// One NVTX row pulled from the trace, in the minimum shape needed to
/// compute depth and iter_index. Kept private so the public API stays
/// a `HashMap<i64, NvtxEntry>`.
#[derive(Debug, Clone)]
struct NvtxRow {
    rowid: i64,
    start: i64,
    /// `None` for instant markers (`NVTX_EVENTS."end" IS NULL`).
    end: Option<i64>,
    /// Thread + domain together key the nesting stack — different
    /// host threads (or NVTX domains) have independent nesting.
    global_tid: i64,
    domain_id: i64,
    /// Resolved range name (`COALESCE(text, StringIds.value, '')`).
    /// Used only to bucket iter_index repeats; identical names share
    /// a bucket. Empty string is a valid name (matches NSys's own
    /// fallback for un-textIded ranges).
    name: String,
}

impl crate::nvtx_stack::NvtxScanRow for NvtxRow {
    fn start(&self) -> i64 {
        self.start
    }
    fn end(&self) -> Option<i64> {
        self.end
    }
    fn rowid(&self) -> i64 {
        self.rowid
    }
}

/// Compute the per-rowid nesting + iter_index map for every NVTX
/// range in the trace.
///
/// Returns an empty map when the trace has no `NVTX_EVENTS` table —
/// agents reading the response should treat "depth absent" as "depth
/// 0" and "iter_index absent" as "0", same as if the trace had a
/// single non-overlapping unnamed range per thread.
pub fn compute(trace: &Trace) -> Result<NvtxNesting> {
    if !trace_has_nvtx(trace)? {
        log::debug!("nvtx_nesting: NVTX_EVENTS table absent — empty result");
        return Ok(HashMap::new());
    }

    let rows = collect_rows(trace).context("collecting NVTX rows for nesting computation")?;
    if rows.is_empty() {
        return Ok(HashMap::new());
    }

    Ok(compute_from_rows(rows))
}

fn trace_has_nvtx(trace: &Trace) -> Result<bool> {
    // Cheap probe — same pattern correlation.rs uses for optional tables.
    let probe_sql = "SELECT 1 FROM nsight.NVTX_EVENTS LIMIT 0";
    Ok(trace.conn().execute(probe_sql, []).is_ok())
}

fn collect_rows(trace: &Trace) -> Result<Vec<NvtxRow>> {
    // `domainId` is nullable on older schemas — coerce NULL to 0 so the
    // default-domain ranges still share a stack the way NSys's GUI
    // groups them.
    //
    // `name` resolves to inline `text` first (rangePushEx with a
    // literal), falling back to `StringIds.value` (registered string),
    // empty string when both are NULL.
    //
    // Intentional divergence from `nvtx_tree::collect_rows`:
    // that path filters to range eventTypes so NvtxDomainCreate / NvtxMark
    // rows can't pose as ranges. This nesting map is consumed only for
    // depth lookups keyed by a real range `rowid`, so the extra non-range
    // rows are inert (instant rows with no `end` don't nest) and a filter
    // would only add cost. Add the same filter here if a future caller
    // ever enumerates or counts this map's rows directly.
    let global_tid = crate::sql_expr::u64_bits_to_i64("n.globalTid");
    let sql = format!(
        r#"SELECT n.rowid,
                  n.start,
                  n."end",
                  {global_tid},
                  COALESCE(n.domainId, 0) AS domain_id,
                  COALESCE(n.text, s.value, '') AS name
           FROM nsight.NVTX_EVENTS n
           LEFT JOIN nsight.StringIds s ON n.textId = s.id
           WHERE n.start IS NOT NULL"#
    );
    let mut stmt = trace.conn().prepare(&sql)?;
    let mut rows = stmt.query([])?;
    let mut out: Vec<NvtxRow> = Vec::new();
    while let Some(r) = rows.next()? {
        out.push(NvtxRow {
            rowid: r.get(0)?,
            start: r.get(1)?,
            end: r.get(2)?,
            global_tid: r.get(3)?,
            domain_id: r.get(4)?,
            name: r.get(5)?,
        });
    }
    Ok(out)
}

/// Pure function over already-collected rows. Split out so tests can
/// exercise the algorithm without a DuckDB round-trip and so a future
/// caller (e.g. a meta cache builder) can reuse the same logic against
/// rows produced by some other path.
fn compute_from_rows(rows: Vec<NvtxRow>) -> NvtxNesting {
    // Step 1: bucket rows by (global_tid, domain_id). Owning the rows
    // here (rather than carrying indices into the source `Vec`) sidesteps
    // index lookups in the hot loop — the workspace denies
    // `clippy::indexing_slicing`, and indices-into-shared-Vec also
    // wouldn't satisfy rayon's borrow checker without an `Arc`.
    // BTreeMap gives deterministic group order for debugging.
    let mut groups: BTreeMap<(i64, i64), Vec<NvtxRow>> = BTreeMap::new();
    for row in rows {
        groups
            .entry((row.global_tid, row.domain_id))
            .or_default()
            .push(row);
    }

    // Step 2: per-group stack scan + name-bucketed iter_index,
    // parallel across groups. Each group emits its (rowid → entry)
    // assignments; we flatten in the merge step below.
    let assignments: Vec<Vec<(i64, NvtxEntry)>> = groups
        .into_par_iter()
        .map(|(_key, mut group)| {
            // The nesting walk (sort-by-start + descending-end stack +
            // touching-boundary rule + depth) is shared with `nvtx_tree`
            // via `nvtx_stack`; only the per-name iter_index counter is
            // nesting-independent and stays here. Keyed by owned `String`
            // (not `&str`) because the scan closure's row borrow can't
            // escape into the counter; cloned only on a name's first
            // occurrence. Depth clamps to u8::MAX on pathological nesting.
            let mut iter_counter: HashMap<String, u32> = HashMap::new();
            crate::nvtx_stack::scan_group(&mut group, |row, _parent, depth| {
                let depth = u8::try_from(depth).unwrap_or(u8::MAX);
                let iter_index = match iter_counter.get_mut(row.name.as_str()) {
                    Some(counter) => {
                        let index = *counter;
                        *counter = counter.saturating_add(1);
                        index
                    }
                    None => {
                        iter_counter.insert(row.name.clone(), 1);
                        0
                    }
                };
                (row.rowid, NvtxEntry { depth, iter_index })
            })
        })
        .collect();

    // Step 3: flatten per-group assignments into the final map. Run
    // sequentially — HashMap inserts don't parallelise and the merge
    // is O(N) over already-sized vectors.
    let total: usize = assignments.iter().map(|v| v.len()).sum();
    let mut result: NvtxNesting = HashMap::with_capacity(total);
    for group in assignments {
        for (rowid, entry) in group {
            result.insert(rowid, entry);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(rowid: i64, start: i64, end: Option<i64>, tid: i64, domain: i64, name: &str) -> NvtxRow {
        NvtxRow {
            rowid,
            start,
            end,
            global_tid: tid,
            domain_id: domain,
            name: name.to_string(),
        }
    }

    /// Two independent groups, ranges, and one instant marker exercise
    /// every branch: pop on close, push on open, depth at marker, no
    /// cross-group contamination.
    #[test]
    fn stack_depths_match_expected_layout() {
        // Group (tid=100, domain=0):
        //   rowid=1: [0, 10)
        //   rowid=2: [2, 5)
        //   rowid=3: instant @ 4 (inside both)
        //   rowid=4: [6, 8)  (after #2 closed; still inside #1)
        // Group (tid=200, domain=1):
        //   rowid=5: [1, 3)
        let rows = vec![
            r(1, 0, Some(10), 100, 0, "outer"),
            r(2, 2, Some(5), 100, 0, "inner"),
            r(3, 4, None, 100, 0, "marker"),
            r(4, 6, Some(8), 100, 0, "inner"),
            r(5, 1, Some(3), 200, 1, "elsewhere"),
        ];

        let entries = compute_from_rows(rows);
        assert_eq!(entries.get(&1).map(|e| e.depth), Some(0));
        assert_eq!(entries.get(&2).map(|e| e.depth), Some(1));
        assert_eq!(entries.get(&3).map(|e| e.depth), Some(2));
        assert_eq!(entries.get(&4).map(|e| e.depth), Some(1));
        assert_eq!(entries.get(&5).map(|e| e.depth), Some(0));
    }

    /// Touching-boundary ranges (`end == next.start`) must NOT nest —
    /// they're sibling ranges, not parent/child. Pin the `<=`
    /// semantics so a future "fix" to `<` doesn't silently change
    /// nesting topology.
    #[test]
    fn touching_boundary_does_not_nest() {
        let rows = vec![r(1, 0, Some(10), 1, 0, "a"), r(2, 10, Some(20), 1, 0, "b")];
        let entries = compute_from_rows(rows);
        assert_eq!(entries.get(&1).map(|e| e.depth), Some(0));
        assert_eq!(entries.get(&2).map(|e| e.depth), Some(0));
    }

    /// Zero-duration ranges (end == start) behave like markers — they
    /// take a depth but never push onto the stack. Without the
    /// `duration > 0` guard, a degenerate range would push a
    /// same-instant `end` that the very next event would immediately
    /// pop. Fine functionally, wasteful in the hot loop — explicit
    /// guard prevents it.
    #[test]
    fn zero_duration_range_is_treated_as_marker() {
        let rows = vec![
            r(1, 0, Some(10), 1, 0, "a"),
            r(2, 5, Some(5), 1, 0, "b"), // degenerate inside #1
            r(3, 6, Some(8), 1, 0, "c"), // still nested under #1 only
        ];
        let entries = compute_from_rows(rows);
        assert_eq!(entries.get(&1).map(|e| e.depth), Some(0));
        assert_eq!(entries.get(&2).map(|e| e.depth), Some(1));
        assert_eq!(entries.get(&3).map(|e| e.depth), Some(1));
    }

    /// Same-name repeats on one thread get sequential iter_index
    /// values in start order — that's the load-bearing invariant for
    /// reverse attribution surfacing "which step" to agents.
    /// Different names share the same (tid, domain) bucket but use
    /// independent counters.
    #[test]
    fn iter_index_counts_same_name_repeats_per_thread() {
        let rows = vec![
            r(1, 0, Some(10), 1, 0, "step"),    // step_0
            r(2, 20, Some(30), 1, 0, "step"),   // step_1
            r(3, 40, Some(50), 1, 0, "decode"), // decode_0 (independent counter)
            r(4, 60, Some(70), 1, 0, "step"),   // step_2
            r(5, 0, Some(10), 2, 0, "step"),    // separate thread → step_0
        ];
        let entries = compute_from_rows(rows);
        assert_eq!(entries.get(&1).map(|e| e.iter_index), Some(0));
        assert_eq!(entries.get(&2).map(|e| e.iter_index), Some(1));
        assert_eq!(entries.get(&3).map(|e| e.iter_index), Some(0));
        assert_eq!(entries.get(&4).map(|e| e.iter_index), Some(2));
        assert_eq!(entries.get(&5).map(|e| e.iter_index), Some(0));
    }
}
