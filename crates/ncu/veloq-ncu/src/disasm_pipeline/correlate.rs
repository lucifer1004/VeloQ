//! Top-level disasm acquisition: run nvdisasm + cuobjdump on a
//! cubin, parse both outputs, build the inverted source-index, and
//! return one [`CorrelatedEntry`] for the caller (typically the
//! walker, which then caches it under `<sha>.correlated.json`).

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::path::Path;

use super::cuobjdump::parse_ptx_lines;
use super::nvdisasm::{parse_emit_json, parse_line_annotations};
use super::tools::run_tool;
use super::types::{CorrelatedEntry, KernelDisasm, PtxLine, SourceIndexRow};

/// Run the full acquisition pipeline against a cubin file:
/// nvdisasm (twice — `--emit-json` for SASS opcodes, `--print-line-info`
/// for source attribution), then `cuobjdump --dump-ptx` to recover
/// PTX text. Builds the source-index by inverting both attributions.
///
/// Always produces the full correlated entry; callers project out
/// subsets in the response based on `ReportOptions.correlate`.
/// `cubin_sha` and `sm` are left as the caller's defaults (empty /
/// `None`) — the caller fills those in from the cubin it extracted.
pub fn acquire_correlated(cubin_path: &Path, instruction_stride: u64) -> Result<CorrelatedEntry> {
    let json_out = run_tool("nvdisasm", cubin_path, &["--emit-json"])?;
    let text_out = run_tool("nvdisasm", cubin_path, &["--print-line-info"])?;
    let line_map = parse_line_annotations(&text_out.stdout);
    let kernels = parse_emit_json(&json_out.stdout, &line_map, instruction_stride)
        .context("parsing nvdisasm --emit-json output")?;
    let source_lineinfo_present = kernels
        .iter()
        .flat_map(|k| k.instructions.iter())
        .any(|i| i.source.is_some());

    let ptx_out = run_tool("cuobjdump", cubin_path, &["--dump-ptx"])?;
    let ptx_lines = parse_ptx_lines(&ptx_out.stdout);

    let source_index = build_source_index(&kernels, &ptx_lines);

    let mut warnings = json_out.warnings;
    warnings.extend(text_out.warnings);
    warnings.extend(ptx_out.warnings);

    Ok(CorrelatedEntry {
        cubin_sha: String::new(),
        sm: None,
        instruction_stride,
        source_lineinfo_present,
        kernels,
        ptx_lines,
        source_index,
        warnings,
    })
}

/// Invert SASS-instruction + PTX-line source attributions into the
/// `source_index` table: one row per `(file, line)` pair, sorted by
/// file ascending then line ascending. SASS addresses and PTX line
/// numbers within each row are sorted + deduplicated.
pub fn build_source_index(kernels: &[KernelDisasm], ptx_lines: &[PtxLine]) -> Vec<SourceIndexRow> {
    let mut rows: BTreeMap<(String, u32), SourceIndexRow> = BTreeMap::new();
    for insn in kernels.iter().flat_map(|k| k.instructions.iter()) {
        if let Some(s) = insn.source.as_ref() {
            let row = rows
                .entry((s.file.clone(), s.line))
                .or_insert_with(|| SourceIndexRow {
                    file: s.file.clone(),
                    line: s.line,
                    sass_addresses: Vec::new(),
                    ptx_line_numbers: Vec::new(),
                });
            row.sass_addresses.push(insn.address);
        }
    }
    for ptx in ptx_lines {
        if let Some(s) = ptx.source.as_ref() {
            let row = rows
                .entry((s.file.clone(), s.line))
                .or_insert_with(|| SourceIndexRow {
                    file: s.file.clone(),
                    line: s.line,
                    sass_addresses: Vec::new(),
                    ptx_line_numbers: Vec::new(),
                });
            row.ptx_line_numbers.push(ptx.line_number);
        }
    }
    let mut out: Vec<SourceIndexRow> = rows.into_values().collect();
    for row in &mut out {
        row.sass_addresses.sort_unstable();
        row.sass_addresses.dedup();
        row.ptx_line_numbers.sort_unstable();
        row.ptx_line_numbers.dedup();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disasm_pipeline::types::{SassInstruction, SourceLineRef};
    use anyhow::Result;

    #[test]
    fn build_source_index_sorts_and_dedupes() -> Result<()> {
        let kernels = vec![KernelDisasm {
            key: "kernel-test".to_string(),
            function_name: "k".into(),
            start: 0,
            length: 64,
            instructions: vec![
                SassInstruction {
                    address: 32,
                    opcode: "X".into(),
                    operands: "".into(),
                    predicate: None,
                    control_flow: false,
                    source: Some(SourceLineRef {
                        file: "b.cu".into(),
                        line: 80,
                        column: None,
                    }),
                },
                SassInstruction {
                    address: 0,
                    opcode: "Y".into(),
                    operands: "".into(),
                    predicate: None,
                    control_flow: false,
                    source: Some(SourceLineRef {
                        file: "a.cu".into(),
                        line: 78,
                        column: None,
                    }),
                },
                SassInstruction {
                    address: 16,
                    opcode: "Z".into(),
                    operands: "".into(),
                    predicate: None,
                    control_flow: false,
                    source: Some(SourceLineRef {
                        file: "b.cu".into(),
                        line: 80,
                        column: None,
                    }),
                },
            ],
        }];
        let ptx = vec![
            PtxLine {
                line_number: 5,
                text: ".loc 1 80 0".into(),
                source: None,
            },
            PtxLine {
                line_number: 7,
                text: "ld.u32 ...".into(),
                source: Some(SourceLineRef {
                    file: "b.cu".into(),
                    line: 80,
                    column: None,
                }),
            },
            PtxLine {
                line_number: 12,
                text: "st.u32 ...".into(),
                source: Some(SourceLineRef {
                    file: "a.cu".into(),
                    line: 78,
                    column: None,
                }),
            },
        ];
        let idx = build_source_index(&kernels, &ptx);
        assert_eq!(idx.len(), 2);
        let a = idx
            .first()
            .ok_or_else(|| anyhow::anyhow!("expected a.cu row"))?;
        let b = idx
            .get(1)
            .ok_or_else(|| anyhow::anyhow!("expected b.cu row"))?;
        // Sort: a.cu first, then b.cu
        assert_eq!(a.file, "a.cu");
        assert_eq!(a.line, 78);
        assert_eq!(a.sass_addresses, vec![0]);
        assert_eq!(a.ptx_line_numbers, vec![12]);
        assert_eq!(b.file, "b.cu");
        assert_eq!(b.line, 80);
        // SASS addresses sorted + deduped (32 then 16 → 16, 32)
        assert_eq!(b.sass_addresses, vec![16, 32]);
        assert_eq!(b.ptx_line_numbers, vec![7]);
        Ok(())
    }

    #[test]
    fn source_lineinfo_present_reflects_at_least_one_attribution() -> Result<()> {
        let mut kernels = [KernelDisasm {
            key: "kernel-test".to_string(),
            function_name: "k".into(),
            start: 0,
            length: 0,
            instructions: vec![],
        }];
        let derived_empty = kernels
            .iter()
            .flat_map(|k| k.instructions.iter())
            .any(|i| i.source.is_some());
        assert!(!derived_empty);
        kernels
            .first_mut()
            .ok_or_else(|| anyhow::anyhow!("kernel"))?
            .instructions
            .push(SassInstruction {
                address: 0,
                opcode: "X".into(),
                operands: String::new(),
                predicate: None,
                control_flow: false,
                source: Some(SourceLineRef {
                    file: "f".into(),
                    line: 1,
                    column: None,
                }),
            });
        let derived = kernels
            .iter()
            .flat_map(|k| k.instructions.iter())
            .any(|i| i.source.is_some());
        assert!(derived);
        Ok(())
    }
}
