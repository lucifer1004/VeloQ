use super::compute::{NvtxRow, collect_rows, compute_from_rows};
use super::parquet::{KV_VERSION, parquet_schema, read_parquet, sidecar_is_fresh, write_parquet};
use super::{NVTX_TREE_VERSION, NvtxTree, NvtxTreeRecord, source_fingerprint};
use crate::Trace;
use crate::test_support::{parquet_fixture_with_rows, write_test_parquet};
use ::parquet::arrow::ArrowWriter;
use ::parquet::basic::Compression;
use ::parquet::file::properties::WriterProperties;
use anyhow::{Context, Result};
use arrow::array::{ArrayRef, Int32Array, Int64Array, StringArray, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::fs::File;
use std::sync::Arc;
use veloq_core::{SourceFingerprint, VeloqDiagnostic};

fn valid_empty_kernel() -> (&'static str, &'static str, Vec<&'static str>) {
    (
        "CUPTI_ACTIVITY_KIND_KERNEL",
        r#"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start BIGINT, "end" BIGINT)"#,
        Vec::new(),
    )
}

fn valid_string_ids() -> (&'static str, &'static str, Vec<&'static str>) {
    (
        "StringIds",
        "CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT)",
        Vec::new(),
    )
}

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

fn tree_record(
    range_id: i64,
    parent_range_id: Option<i64>,
    depth: i32,
    domain_id: i64,
    name: &str,
    start: i64,
    end: i64,
) -> NvtxTreeRecord {
    NvtxTreeRecord {
        range_id,
        parent_range_id,
        depth,
        domain_id,
        name: name.to_string(),
        path: name.to_string(),
        start,
        end: Some(end),
        duration_ns: Some(end - start),
        global_tid: 7,
    }
}

fn find(records: &[NvtxTreeRecord], range_id: i64) -> Result<&NvtxTreeRecord> {
    records
        .iter()
        .find(|r| r.range_id == range_id)
        .with_context(|| format!("range_id {range_id} not present in records"))
}

#[test]
fn collect_rows_missing_stringids_error_is_typed() -> Result<()> {
    let (_dir, pqtdir) = parquet_fixture_with_rows(&[
        valid_empty_kernel(),
        (
            "NVTX_EVENTS",
            r#"CREATE TABLE NVTX_EVENTS (
                rowid BIGINT,
                start BIGINT,
                "end" BIGINT,
                eventType BIGINT,
                globalTid BIGINT,
                domainId BIGINT,
                text TEXT,
                textId BIGINT
            )"#,
            Vec::new(),
        ),
    ])?;
    let trace = Trace::open(&pqtdir)?;

    let err = match collect_rows(&trace) {
        Ok(rows) => anyhow::bail!("missing StringIds should not collect rows: {rows:?}"),
        Err(err) => err,
    };

    assert_eq!(err.code().as_str(), "nsys.data.duckdb-prepare");
    assert_eq!(
        err.duckdb_parts(),
        Some(("nvtx tree", crate::DuckdbPhase::Prepare, "rows"))
    );
    Ok(())
}

#[test]
fn collect_rows_bad_start_type_error_is_typed() -> Result<()> {
    let (_dir, pqtdir) = parquet_fixture_with_rows(&[
        valid_empty_kernel(),
        valid_string_ids(),
        (
            "NVTX_EVENTS",
            r#"CREATE TABLE NVTX_EVENTS (
                rowid BIGINT,
                start TEXT,
                "end" BIGINT,
                eventType BIGINT,
                globalTid BIGINT,
                domainId BIGINT,
                text TEXT,
                textId BIGINT
            )"#,
            vec![
                r#"INSERT INTO NVTX_EVENTS
                   (rowid, start, "end", eventType, globalTid, domainId, text, textId)
                   VALUES (1, 'bad', 10, 59, 7, 0, 'outer', NULL)"#,
            ],
        ),
    ])?;
    let trace = Trace::open(&pqtdir)?;

    let err = match collect_rows(&trace) {
        Ok(rows) => anyhow::bail!("text start should not collect rows: {rows:?}"),
        Err(err) => err,
    };

    assert_eq!(err.code().as_str(), "nsys.data.duckdb-read");
    assert_eq!(
        err.duckdb_parts(),
        Some(("nvtx tree", crate::DuckdbPhase::Read, "rows"))
    );
    Ok(())
}

#[test]
fn source_fingerprint_missing_trace_error_is_typed() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("missing_pqtdir");

    let err = match source_fingerprint(&path) {
        Ok(fp) => anyhow::bail!("missing trace should not fingerprint: {fp:?}"),
        Err(err) => err,
    };

    assert_eq!(err.code().as_str(), "nsys.data.nvtx-tree-trace-fingerprint");
    match err {
        crate::NsysDataError::NvtxTreeTraceFingerprint { path, .. } => {
            assert!(path.contains("missing_pqtdir"));
        }
        other => anyhow::bail!("expected NvtxTreeTraceFingerprint, got {other:?}"),
    }
    Ok(())
}

#[test]
fn read_parquet_missing_file_error_is_typed() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("missing.parquet");

    let err = match read_parquet(&path) {
        Ok(rows) => anyhow::bail!("missing nvtx-tree sidecar should not load: {rows:?}"),
        Err(err) => err,
    };

    assert_eq!(err.code().as_str(), "nsys.data.nvtx-tree-sidecar-open");
    match err {
        crate::NsysDataError::NvtxTreeSidecarOpen { path, .. } => {
            assert!(path.contains("missing.parquet"));
        }
        other => anyhow::bail!("expected NvtxTreeSidecarOpen, got {other:?}"),
    }
    Ok(())
}

#[test]
fn read_parquet_invalid_file_error_is_typed() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("bad.parquet");
    std::fs::write(&path, b"not a parquet file")?;

    let err = match read_parquet(&path) {
        Ok(rows) => anyhow::bail!("invalid nvtx-tree sidecar should not load: {rows:?}"),
        Err(err) => err,
    };

    assert_eq!(err.code().as_str(), "nsys.data.nvtx-tree-reader-open");
    match err {
        crate::NsysDataError::NvtxTreeReaderOpen { path, .. } => {
            assert!(path.contains("bad.parquet"));
        }
        other => anyhow::bail!("expected NvtxTreeReaderOpen, got {other:?}"),
    }
    Ok(())
}

/// Nested push/pop on a single thread -> outer first by depth,
/// inner gets outer as parent, path is "outer/inner".
#[test]
fn nested_stack_emits_parent_and_path() -> Result<()> {
    let rows = vec![
        r(1, 0, Some(100), 7, 0, "outer"),
        r(2, 40, Some(60), 7, 0, "inner"),
    ];
    let out = compute_from_rows(rows);
    let outer = find(&out, 1)?;
    let inner = find(&out, 2)?;
    assert_eq!(outer.parent_range_id, None);
    assert_eq!(outer.depth, 0);
    assert_eq!(outer.path, "outer");
    assert_eq!(inner.parent_range_id, Some(1));
    assert_eq!(inner.depth, 1);
    assert_eq!(inner.path, "outer/inner");
    Ok(())
}

/// Unterminated range (end is None - an instant marker, in the
/// codebase's terminology) gets the post-pop depth + parent but
/// never pushes onto the stack. Subsequent ranges therefore see
/// the same stack the marker did.
#[test]
fn unterminated_range_does_not_push() -> Result<()> {
    let rows = vec![
        r(1, 0, Some(100), 7, 0, "outer"),
        r(2, 10, None, 7, 0, "marker"),
        r(3, 20, Some(30), 7, 0, "inner"),
    ];
    let out = compute_from_rows(rows);
    assert_eq!(find(&out, 2)?.end, None);
    assert_eq!(find(&out, 2)?.duration_ns, None);
    assert_eq!(find(&out, 2)?.parent_range_id, Some(1));
    // `inner` must still see only `outer` on the stack (marker
    // didn't push), so depth=1 and parent=outer.
    assert_eq!(find(&out, 3)?.depth, 1);
    assert_eq!(find(&out, 3)?.parent_range_id, Some(1));
    assert_eq!(find(&out, 3)?.path, "outer/inner");
    Ok(())
}

/// Two threads each open their own stack - no cross-tid
/// contamination even if their intervals overlap.
#[test]
fn per_tid_stacks_are_isolated() -> Result<()> {
    let rows = vec![
        r(1, 0, Some(100), 7, 0, "tid7_outer"),
        r(2, 50, Some(80), 7, 0, "tid7_inner"),
        r(3, 10, Some(90), 8, 0, "tid8_outer"),
        r(4, 20, Some(40), 8, 0, "tid8_inner"),
    ];
    let out = compute_from_rows(rows);
    assert_eq!(find(&out, 2)?.parent_range_id, Some(1));
    assert_eq!(find(&out, 4)?.parent_range_id, Some(3));
    assert_eq!(find(&out, 2)?.path, "tid7_outer/tid7_inner");
    assert_eq!(find(&out, 4)?.path, "tid8_outer/tid8_inner");
    Ok(())
}

/// Two domains on the same tid form independent stacks too,
/// mirroring `nvtx_nesting`'s grouping.
#[test]
fn per_domain_stacks_are_isolated() -> Result<()> {
    let rows = vec![
        r(1, 0, Some(100), 7, 1, "domain1_outer"),
        r(2, 10, Some(20), 7, 2, "domain2_outer"),
        r(3, 30, Some(40), 7, 2, "domain2_inner"),
    ];
    let out = compute_from_rows(rows);
    // `domain2_inner` is on domain 2 - its parent is the outer in
    // domain 2, not the wider range in domain 1.
    assert_eq!(find(&out, 3)?.parent_range_id, None);
    // Same start ordering: domain 2 has the inner at 30, which
    // arrives after domain2_outer (10..20) has already closed; so
    // it's a root at depth 0 on its own stack.
    assert_eq!(find(&out, 3)?.depth, 0);
    Ok(())
}

/// Touching boundaries (`end == next.start`) do NOT nest - pin
/// the same `<=` semantics that `nvtx_nesting` uses.
#[test]
fn touching_boundary_does_not_nest() -> Result<()> {
    let rows = vec![r(1, 0, Some(10), 1, 0, "a"), r(2, 10, Some(20), 1, 0, "b")];
    let out = compute_from_rows(rows);
    assert_eq!(find(&out, 2)?.parent_range_id, None);
    assert_eq!(find(&out, 2)?.depth, 0);
    Ok(())
}

/// Names containing `/` get escaped in `path` so the join character
/// stays unambiguous.
#[test]
fn slashes_in_names_are_escaped() -> Result<()> {
    let rows = vec![
        r(1, 0, Some(100), 1, 0, "a/b"),
        r(2, 10, Some(20), 1, 0, "c\\d"),
    ];
    let out = compute_from_rows(rows);
    assert_eq!(find(&out, 1)?.path, r"a\/b");
    // Both the literal backslash from the name and the parent's
    // escaped slash round-trip cleanly.
    assert_eq!(find(&out, 2)?.path, r"a\/b/c\\d");
    Ok(())
}

/// `domain_id` is taken straight from the row; `compute_rows`
/// collects NULL as 0 via `COALESCE` so the sidecar can stay
/// non-nullable.
#[test]
fn domain_id_defaults_to_zero_when_unknown() -> Result<()> {
    // `collect_rows` does the COALESCE at SQL time; the algorithm
    // here just reads whatever it's given. So this test simulates
    // the post-COALESCE shape.
    let rows = vec![r(1, 0, Some(10), 1, 0, "default")];
    let out = compute_from_rows(rows);
    assert_eq!(find(&out, 1)?.domain_id, 0);
    Ok(())
}

/// Roundtrip preserves every field - including nullable end /
/// duration / parent_range_id - and the fingerprint validates.
#[test]
fn parquet_roundtrip_preserves_records() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("nvtx_tree.parquet");
    let records = vec![
        NvtxTreeRecord {
            range_id: 1,
            parent_range_id: None,
            depth: 0,
            domain_id: 0,
            name: "outer".into(),
            path: "outer".into(),
            start: 0,
            end: Some(100),
            duration_ns: Some(100),
            global_tid: 7,
        },
        NvtxTreeRecord {
            range_id: 2,
            parent_range_id: Some(1),
            depth: 1,
            domain_id: 0,
            name: "inner".into(),
            path: "outer/inner".into(),
            start: 10,
            end: Some(20),
            duration_ns: Some(10),
            global_tid: 7,
        },
        // Instant marker: NULL end / duration; must round-trip.
        NvtxTreeRecord {
            range_id: 3,
            parent_range_id: Some(1),
            depth: 1,
            domain_id: 0,
            name: "marker".into(),
            path: "outer/marker".into(),
            start: 50,
            end: None,
            duration_ns: None,
            global_tid: 7,
        },
    ];
    let fp = SourceFingerprint {
        mtime_secs: 1_234_567_890,
        size: 4096,
    };
    write_parquet(&path, fp, &records)?;
    assert!(sidecar_is_fresh(&path, fp)?);
    let loaded = read_parquet(&path)?;
    assert_eq!(loaded, records);
    Ok(())
}

#[test]
fn read_parquet_rejects_wrong_column_type_with_typed_error() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("nvtx_tree.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("range_id", DataType::Int64, false),
        Field::new("parent_range_id", DataType::Int64, true),
        Field::new("depth", DataType::Int32, false),
        Field::new("domain_id", DataType::Int64, false),
        Field::new("name", DataType::Int64, false),
        Field::new("path", DataType::Utf8, false),
        Field::new("start", DataType::Int64, false),
        Field::new("end", DataType::Int64, true),
        Field::new("duration_ns", DataType::Int64, true),
        Field::new("global_tid", DataType::Int64, false),
    ]));
    write_test_parquet(
        &path,
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1])),
            Arc::new(Int64Array::from(vec![None::<i64>])),
            Arc::new(Int32Array::from(vec![0])),
            Arc::new(Int64Array::from(vec![0])),
            Arc::new(Int64Array::from(vec![99])),
            Arc::new(StringArray::from(vec!["outer"])),
            Arc::new(Int64Array::from(vec![0])),
            Arc::new(Int64Array::from(vec![Some(100)])),
            Arc::new(Int64Array::from(vec![Some(100)])),
            Arc::new(Int64Array::from(vec![7])),
        ],
    )?;

    let err = match read_parquet(&path) {
        Ok(rows) => anyhow::bail!("wrong-typed nvtx-tree sidecar should not load: {rows:?}"),
        Err(err) => err,
    };

    assert_eq!(
        err.code().as_str(),
        "nsys.data.nvtx-tree-column-type-mismatch"
    );
    match err {
        crate::NsysDataError::NvtxTreeColumnTypeMismatch {
            column,
            expected,
            actual,
            ..
        } => {
            assert_eq!(column, "name");
            assert_eq!(expected, "Utf8");
            assert!(actual.contains("Int64"));
        }
        other => anyhow::bail!("expected NvtxTreeColumnTypeMismatch, got {other:?}"),
    }
    Ok(())
}

#[test]
fn read_parquet_rejects_missing_column_with_typed_error() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("nvtx_tree.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("range_id", DataType::Int64, false),
        Field::new("parent_range_id", DataType::Int64, true),
        Field::new("depth", DataType::Int32, false),
        Field::new("domain_id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("path", DataType::Utf8, false),
        Field::new("start", DataType::Int64, false),
        Field::new("end", DataType::Int64, true),
        Field::new("duration_ns", DataType::Int64, true),
    ]));
    write_test_parquet(
        &path,
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1])),
            Arc::new(Int64Array::from(vec![None::<i64>])),
            Arc::new(Int32Array::from(vec![0])),
            Arc::new(Int64Array::from(vec![0])),
            Arc::new(StringArray::from(vec!["outer"])),
            Arc::new(StringArray::from(vec!["outer"])),
            Arc::new(Int64Array::from(vec![0])),
            Arc::new(Int64Array::from(vec![Some(100)])),
            Arc::new(Int64Array::from(vec![Some(100)])),
        ],
    )?;

    let err = match read_parquet(&path) {
        Ok(rows) => anyhow::bail!("truncated nvtx-tree sidecar should not load: {rows:?}"),
        Err(err) => err,
    };

    assert_eq!(err.code().as_str(), "nsys.data.nvtx-tree-column-missing");
    match err {
        crate::NsysDataError::NvtxTreeColumnMissing { column, .. } => {
            assert_eq!(column, "global_tid");
        }
        other => anyhow::bail!("expected NvtxTreeColumnMissing, got {other:?}"),
    }
    Ok(())
}

/// Fingerprint mismatch (different mtime or size) invalidates.
#[test]
fn mtime_or_size_change_invalidates_sidecar() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("nvtx_tree.parquet");
    let fp = SourceFingerprint {
        mtime_secs: 100,
        size: 200,
    };
    write_parquet(&path, fp, &[])?;
    assert!(sidecar_is_fresh(&path, fp)?);
    assert!(!sidecar_is_fresh(
        &path,
        SourceFingerprint {
            mtime_secs: 101,
            size: 200,
        },
    )?);
    assert!(!sidecar_is_fresh(
        &path,
        SourceFingerprint {
            mtime_secs: 100,
            size: 201,
        },
    )?);
    Ok(())
}

/// A sidecar written under a different `NVTX_TREE_VERSION` is
/// stale even if the fingerprint matches - readers rebuild
/// silently.
#[test]
fn version_mismatch_invalidates_sidecar() -> Result<()> {
    // Easiest way to simulate a version mismatch is to write a
    // parquet whose `KV_VERSION` key holds a different number,
    // then assert `sidecar_is_fresh` returns false.
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("nvtx_tree.parquet");
    let schema = parquet_schema();
    // Empty batch - we only care about the KV metadata for this
    // freshness check.
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(Vec::<i64>::new())),
        Arc::new(Int64Array::from(Vec::<Option<i64>>::new())),
        Arc::new(Int32Array::from(Vec::<i32>::new())),
        Arc::new(Int64Array::from(Vec::<i64>::new())),
        Arc::new(StringBuilder::new().finish()),
        Arc::new(StringBuilder::new().finish()),
        Arc::new(Int64Array::from(Vec::<i64>::new())),
        Arc::new(Int64Array::from(Vec::<Option<i64>>::new())),
        Arc::new(Int64Array::from(Vec::<Option<i64>>::new())),
        Arc::new(Int64Array::from(Vec::<i64>::new())),
    ];
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns)?;
    use parquet::file::metadata::KeyValue;
    let kv = vec![
        KeyValue::new(
            KV_VERSION.to_string(),
            Some((NVTX_TREE_VERSION + 1).to_string()),
        ),
        KeyValue::new(
            crate::sidecar::KV_MTIME.to_string(),
            Some("100".to_string()),
        ),
        KeyValue::new(crate::sidecar::KV_SIZE.to_string(), Some("200".to_string())),
    ];
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_key_value_metadata(Some(kv))
        .build();
    let file = File::create(&path)?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    assert!(!sidecar_is_fresh(
        &path,
        SourceFingerprint {
            mtime_secs: 100,
            size: 200,
        },
    )?);
    Ok(())
}

/// `stack_at` returns ancestors outer->inner for a tid+timestamp
/// inside several nested ranges, and empty when nothing covers
/// the point.
#[test]
fn stack_at_returns_outer_to_inner_chain() {
    let rows = vec![
        r(1, 0, Some(100), 7, 0, "outer"),
        r(2, 40, Some(80), 7, 0, "mid"),
        r(3, 50, Some(60), 7, 0, "inner"),
    ];
    let records = compute_from_rows(rows);
    let tree = NvtxTree::from_records(records);

    let stack = tree.stack_at(7, 55);
    let names: Vec<&str> = stack.iter().map(|r| r.name.as_str()).collect();
    assert_eq!(names, vec!["outer", "mid", "inner"]);

    // Outside any range.
    assert!(tree.stack_at(7, 200).is_empty());
    // Wrong tid.
    assert!(tree.stack_at(99, 55).is_empty());
}

/// When unrelated ranges cover the same timestamp on one tid, the
/// public stack API still returns one deterministic parent chain
/// rather than sibling rows that cannot form a stack.
#[test]
fn stack_at_returns_one_chain_for_unrelated_covering_ranges() {
    let records = vec![
        tree_record(1, None, 0, 1, "domain1_root", 0, 100),
        tree_record(2, None, 0, 2, "domain2_root", 10, 90),
        tree_record(3, Some(2), 1, 2, "domain2_child", 20, 80),
    ];
    let tree = NvtxTree::from_records(records);

    let stack = tree.stack_at(7, 50);
    let names: Vec<&str> = stack.iter().map(|r| r.name.as_str()).collect();
    assert_eq!(names, vec!["domain2_root", "domain2_child"]);
}

/// `ancestors` walks up through parent links - emit self first,
/// then parents toward the root.
#[test]
fn ancestors_walks_parent_chain() {
    let rows = vec![
        r(1, 0, Some(100), 7, 0, "outer"),
        r(2, 40, Some(80), 7, 0, "mid"),
        r(3, 50, Some(60), 7, 0, "inner"),
    ];
    let records = compute_from_rows(rows);
    let tree = NvtxTree::from_records(records);

    let names: Vec<&str> = tree.ancestors(3).iter().map(|r| r.name.as_str()).collect();
    assert_eq!(names, vec!["inner", "mid", "outer"]);
}
