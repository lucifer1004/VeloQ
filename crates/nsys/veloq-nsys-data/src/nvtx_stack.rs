//! Shared NVTX nesting stack-scan.
//!
//! Both the nesting sidecar ([`crate::nvtx_nesting`], depth + per-name
//! iter_index) and the tree sidecar ([`crate::nvtx_tree`], parent_id +
//! depth + path) walk one `(global_tid, domain_id)` bucket of NVTX rows
//! the same way: sort by start, maintain a descending-end stack, pop
//! ranges that closed at-or-before the current start, then push the
//! current range. This module is that walk, factored out so the
//! touching-boundary rule (below) is defined exactly once.
//!
//! (The runtime→enclosing-NVTX sidecar [`crate::runtime_nvtx_parent`] is
//! a *different* algorithm — an interval-containment join between two row
//! sets, not this single-set nesting scan — and is intentionally not
//! built on this.)

/// A row the nesting scan needs: its `start` / `end` span and a stable
/// `rowid` to report as a parent. `end == None` is an instant marker.
pub trait NvtxScanRow {
    fn start(&self) -> i64;
    fn end(&self) -> Option<i64>;
    fn rowid(&self) -> i64;
}

/// Run the nesting stack over one already-grouped bucket (one
/// `(global_tid, domain_id)`), invoking `emit(row, parent_rowid, depth)`
/// per row in start order: `parent_rowid` is the immediately enclosing
/// open range's rowid (`None` at depth 0), `depth` the count of open
/// enclosing ranges. Returns the emitted values in scan order.
///
/// Semantics (shared by both sidecars):
/// - **Touching boundaries do not nest**: `top_end <= start` pops, so a
///   range ending exactly when the next starts is a sibling, not a parent.
/// - **Zero-duration / instant rows take a depth but never become a
///   parent** (only `end > start` ranges are pushed onto the stack).
/// - Stable sort by `start` keeps equal-start rows in input order.
pub fn scan_group<R: NvtxScanRow, T>(
    group: &mut [R],
    mut emit: impl FnMut(&R, Option<i64>, usize) -> T,
) -> Vec<T> {
    group.sort_by_key(|r| r.start());
    // (end, rowid) sorted by end descending so `last()` is the
    // soonest-closing open range.
    let mut stack: Vec<(i64, i64)> = Vec::with_capacity(32);
    let mut out: Vec<T> = Vec::with_capacity(group.len());
    for row in group.iter() {
        while let Some(&(top_end, _)) = stack.last() {
            if top_end <= row.start() {
                stack.pop();
            } else {
                break;
            }
        }
        let parent = stack.last().map(|&(_, id)| id);
        out.push(emit(row, parent, stack.len()));
        if let Some(end) = row.end()
            && end > row.start()
        {
            // Sorted-insert by descending end: the first slot whose end
            // is <= `end` is exactly where this range belongs.
            let pos = stack.partition_point(|&(e, _)| e > end);
            stack.insert(pos, (end, row.rowid()));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Row {
        rowid: i64,
        start: i64,
        end: Option<i64>,
    }
    impl NvtxScanRow for Row {
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

    fn scan(mut rows: Vec<Row>) -> Vec<(i64, Option<i64>, usize)> {
        scan_group(&mut rows, |r, parent, depth| (r.rowid, parent, depth))
    }

    #[test]
    fn touching_boundary_is_sibling_not_child() {
        // A: [0,100], B: [100,200] — B starts exactly when A ends.
        // top_end (100) <= start (100) pops A, so B is a sibling: both
        // depth 0, parent None.
        let out = scan(vec![
            Row {
                rowid: 1,
                start: 0,
                end: Some(100),
            },
            Row {
                rowid: 2,
                start: 100,
                end: Some(200),
            },
        ]);
        assert_eq!(out, vec![(1, None, 0), (2, None, 0)]);
    }

    #[test]
    fn strict_nesting_reports_parent_and_depth() {
        // outer [0,100] ⊃ inner [25,75] ⊃ leaf [40,60]; insertion order
        // shuffled to exercise the start-sort.
        let out = scan(vec![
            Row {
                rowid: 3,
                start: 40,
                end: Some(60),
            },
            Row {
                rowid: 1,
                start: 0,
                end: Some(100),
            },
            Row {
                rowid: 2,
                start: 25,
                end: Some(75),
            },
        ]);
        assert_eq!(out, vec![(1, None, 0), (2, Some(1), 1), (3, Some(2), 2)]);
    }

    #[test]
    fn instant_marker_takes_depth_but_is_not_a_parent() {
        // outer [0,100] ⊃ marker @50 (end None). A later range [60,70]
        // inside outer must see outer as parent, never the marker.
        let out = scan(vec![
            Row {
                rowid: 1,
                start: 0,
                end: Some(100),
            },
            Row {
                rowid: 2,
                start: 50,
                end: None,
            },
            Row {
                rowid: 3,
                start: 60,
                end: Some(70),
            },
        ]);
        assert_eq!(out, vec![(1, None, 0), (2, Some(1), 1), (3, Some(1), 1)]);
    }
}
