use anyhow::Result;
use arrow::array::ArrayRef;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tempfile::TempDir;

pub(crate) fn write_test_parquet(
    path: &Path,
    schema: SchemaRef,
    columns: Vec<ArrayRef>,
) -> Result<()> {
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns)?;
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

pub(crate) fn parquet_fixture_with_rows(
    tables: &[(&str, &str, Vec<&str>)],
) -> Result<(TempDir, PathBuf)> {
    let dir = tempfile::tempdir()?;
    let pqtdir = dir.path().join("test_pqtdir");
    std::fs::create_dir_all(&pqtdir)?;
    let conn = duckdb::Connection::open_in_memory()?;
    for (_, ddl, inserts) in tables {
        conn.execute_batch(ddl)?;
        for insert in inserts {
            conn.execute_batch(insert)?;
        }
    }
    for (table, _, _) in tables {
        let out = pqtdir.join(format!("{table}.parquet"));
        let out_lit = out.to_string_lossy().replace('\'', "''");
        conn.execute(
            &format!(r#"COPY (SELECT * FROM "{table}") TO '{out_lit}' (FORMAT PARQUET)"#),
            [],
        )?;
    }
    Ok((dir, pqtdir))
}
