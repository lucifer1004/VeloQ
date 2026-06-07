//! Shared CSV / table rendering for source-specific row projections.
//!
//! JSON is the agent contract, but several commands have a natural
//! row-shaped view for humans and spreadsheets. Sources flatten their
//! own payload into [`TabularView`] and this module handles the common
//! rendering details.

use crate::{ErrorCode, VeloqDiagnostic};
use comfy_table::{ContentArrangement, Table, presets::UTF8_BORDERS_ONLY};
use std::fmt;
use std::io::Write;
use thiserror::Error;

pub type TabularResult<T> = Result<T, TabularError>;

#[derive(Debug, Error)]
pub enum TabularError {
    #[error("writing csv header")]
    WriteCsvHeader {
        #[source]
        source: csv::Error,
    },
    #[error("writing csv row")]
    WriteCsvRow {
        #[source]
        source: csv::Error,
    },
    #[error("flushing csv")]
    FlushCsv {
        #[source]
        source: std::io::Error,
    },
}

impl VeloqDiagnostic for TabularError {
    fn code(&self) -> ErrorCode {
        match self {
            Self::WriteCsvHeader { .. } => ErrorCode::OUTPUT_CSV_HEADER,
            Self::WriteCsvRow { .. } => ErrorCode::OUTPUT_CSV_ROW,
            Self::FlushCsv { .. } => ErrorCode::OUTPUT_CSV_FLUSH,
        }
    }
}

/// Decimal precision for floating-point columns in CSV/table output.
pub const DISPLAY_PRECISION: usize = 3;

/// Flattened response shape for non-JSON outputs.
pub struct TabularView {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<String>>,
    /// Envelope-level facts that don't fit in the column grid. Rendered
    /// as `# key=value` comment lines before the CSV header, or as a
    /// trailing key/value table beneath comfy-table output.
    pub meta: Vec<(String, String)>,
}

impl TabularView {
    pub fn new<I, S>(columns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            columns: columns.into_iter().map(Into::into).collect(),
            rows: Vec::new(),
            meta: Vec::new(),
        }
    }

    pub fn push_row(&mut self, cells: Vec<String>) {
        self.rows.push(cells);
    }

    pub fn push_meta(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.meta.push((key.into(), value.into()));
    }
}

pub fn push_count_meta(
    view: &mut TabularView,
    count: impl fmt::Display,
    total_matched: impl fmt::Display,
) {
    view.push_meta("count", count.to_string());
    view.push_meta("total_matched", total_matched.to_string());
}

pub fn push_optional_meta<T>(view: &mut TabularView, key: impl Into<String>, value: Option<T>)
where
    T: fmt::Display,
{
    if let Some(value) = value {
        view.push_meta(key, value.to_string());
    }
}

pub fn emit_csv(view: &TabularView, command: &str, trace_path: &str) -> TabularResult<()> {
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    writeln!(handle, "# command={command}").ok();
    writeln!(handle, "# trace={trace_path}").ok();
    for (k, v) in &view.meta {
        writeln!(handle, "# {k}={v}").ok();
    }

    let mut w = csv::Writer::from_writer(handle);
    w.write_record(view.columns.iter().map(String::as_str))
        .map_err(|source| TabularError::WriteCsvHeader { source })?;
    for row in &view.rows {
        w.write_record(row)
            .map_err(|source| TabularError::WriteCsvRow { source })?;
    }
    w.flush()
        .map_err(|source| TabularError::FlushCsv { source })?;
    Ok(())
}

pub fn emit_table(view: &TabularView, command: &str, trace_path: &str) -> TabularResult<()> {
    let width = terminal_width().unwrap_or(200);
    println!(
        "{}",
        build_table(view.columns.iter().map(String::as_str), &view.rows, width)
    );

    let mut foot_rows: Vec<Vec<String>> = Vec::with_capacity(view.meta.len() + 2);
    foot_rows.push(vec!["command".to_string(), command.to_string()]);
    foot_rows.push(vec!["trace".to_string(), trace_path.to_string()]);
    for (k, v) in &view.meta {
        foot_rows.push(vec![(*k).to_string(), v.clone()]);
    }
    println!("{}", build_table(["field", "value"], &foot_rows, width));
    Ok(())
}

fn build_table<'a, I>(headers: I, rows: &[Vec<String>], width: u16) -> Table
where
    I: IntoIterator<Item = &'a str>,
{
    let mut t = Table::new();
    t.load_preset(UTF8_BORDERS_ONLY);
    t.set_content_arrangement(ContentArrangement::Dynamic);
    t.set_width(width);
    t.set_header(headers);
    for row in rows {
        t.add_row(row.iter().map(|s| s.as_str()));
    }
    t
}

fn terminal_width() -> Option<u16> {
    terminal_size::terminal_size().map(|(w, _)| w.0)
}

/// Format an `Option<T>` as `"<value>"` or empty string.
pub fn cell_opt<T: fmt::Display>(v: Option<T>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => String::new(),
    }
}
