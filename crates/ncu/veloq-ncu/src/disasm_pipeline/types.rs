//! Wire-format types for the SASS / PTX / source correlation surface.
//!
//! All line numbers are **1-based**. PTX line numbers index into the
//! literal `cuobjdump --dump-ptx` output (so blanks/comments/directive
//! lines are present; `text` may be empty).
//!
//! On-disk JSON cache: see [`super::cache`]. The disasm cache uses the
//! plain JSON file format (not the bincode SidecarCache the NSys
//! crate uses) because each cubin's correlated payload is small,
//! and human-readable JSON makes nvdisasm-output debugging
//! immediate.

use serde::{Deserialize, Serialize};

/// One cubin's worth of correlated disassembly. `cubin_sha` keys
/// both the extracted-cubin sidecar file and this entry's cache
/// file.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CorrelatedEntry {
    pub cubin_sha: String,
    /// `"sm_90"` etc., the compute capability the cubin was compiled
    /// for. `None` when the version isn't available (rare).
    pub sm: Option<String>,
    /// Instruction width used to reconstruct SASS addresses from
    /// nvdisasm's positional `--emit-json` output. 16 for Volta+,
    /// 8 for pre-Volta arches.
    pub instruction_stride: u64,
    /// `true` when at least one SASS instruction has a populated
    /// `source` (i.e. the cubin carries DWARF line info). Lets
    /// agents distinguish "no DWARF in cubin" from "DWARF present
    /// but no line attribution at the queried address".
    pub source_lineinfo_present: bool,
    pub kernels: Vec<KernelDisasm>,
    /// One entry per line of `cuobjdump --dump-ptx` output,
    /// including blanks/comments/directives. `line_number` is
    /// 1-based and indexes into that literal text.
    pub ptx_lines: Vec<PtxLine>,
    /// Inverted source map: rows sorted by `(file, line)` ascending,
    /// each row pointing back at the SASS addresses and PTX line
    /// numbers attributed to that source line.
    pub source_index: Vec<SourceIndexRow>,
    /// nvdisasm + cuobjdump stderr / warning lines — surfaces
    /// "Disassembling Std Elf to Old format" and similar trust
    /// signals an agent should see at the same time as the data.
    pub warnings: Vec<String>,
}

/// SASS for one kernel. `start` is the cubin offset of the first
/// instruction; subsequent addresses are derived positionally by
/// stride (16 on Volta+, 8 on older arches).
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct KernelDisasm {
    /// Cross-trace key. Format: `kernel|<function_name>`.
    /// `#[serde(default)]` so older per-cubin JSON caches that lack the
    /// field deserialize cleanly; the `ncu disasm` response builder fills
    /// the key at projection time.
    #[serde(default)]
    pub key: String,
    pub function_name: String,
    pub start: u64,
    pub length: u64,
    pub instructions: Vec<SassInstruction>,
}

/// One SASS instruction. `address` is the cubin offset (callers can
/// look up the same instruction across decode runs); `source` is the
/// optional DWARF line-info attribution.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SassInstruction {
    pub address: u64,
    pub opcode: String,
    pub operands: String,
    /// Predicate guarding execution, if any (`"@P0"`, `"@!P3"`,
    /// `"@PT"`). `None` for unconditional instructions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicate: Option<String>,
    /// Branch / jump / call. Lets agents segment the kernel into
    /// basic blocks without re-parsing operands.
    pub control_flow: bool,
    /// Source line this instruction was attributed to. `None` for
    /// unattributed instructions (cubin without DWARF, or holes in
    /// the line table).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceLineRef>,
}

/// `(file, line)` plus optional column. 1-based.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SourceLineRef {
    pub file: String,
    pub line: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub column: Option<u32>,
}

/// One line of PTX text plus its `.loc`-derived source attribution.
/// PTX `.loc` directives are scoped — every instruction line until
/// the next `.loc` carries the same source ref. Empty / comment /
/// directive lines have `source: None`.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct PtxLine {
    pub line_number: u32,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceLineRef>,
}

/// One row of the `(file, line) → {sass_addresses, ptx_line_numbers}`
/// inverted index. Rows are sorted by `(file, line)` ascending; SASS
/// addresses and PTX line numbers are sorted + deduped.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SourceIndexRow {
    pub file: String,
    pub line: u32,
    pub sass_addresses: Vec<u64>,
    pub ptx_line_numbers: Vec<u32>,
}

/// Schema version for the on-disk `<sha>.correlated.json` cache.
/// Bump when [`CorrelatedEntry`]'s shape changes — [`super::cache::load_cached`]
/// rejects files at other versions and forces a fresh acquire.
pub const CACHE_SCHEMA: u32 = 1;

// Hand-written clone helpers. The wire-format structs intentionally
// don't derive `Clone` so we can spot per-field clones in review.

pub fn clone_source_ref(s: &SourceLineRef) -> SourceLineRef {
    SourceLineRef {
        file: s.file.clone(),
        line: s.line,
        column: s.column,
    }
}

pub fn clone_instruction(i: &SassInstruction) -> SassInstruction {
    SassInstruction {
        address: i.address,
        opcode: i.opcode.clone(),
        operands: i.operands.clone(),
        predicate: i.predicate.clone(),
        control_flow: i.control_flow,
        source: i.source.as_ref().map(clone_source_ref),
    }
}

pub fn clone_kernel(k: &KernelDisasm) -> KernelDisasm {
    KernelDisasm {
        key: k.key.clone(),
        function_name: k.function_name.clone(),
        start: k.start,
        length: k.length,
        instructions: k.instructions.iter().map(clone_instruction).collect(),
    }
}

/// Deep-clone a `CorrelatedEntry`. Used by [`super::cache::write_cache`]
/// so the on-disk envelope can borrow a freshly-cloned copy without
/// forcing the wire-format struct to derive `Clone`.
pub fn clone_entry(e: &CorrelatedEntry) -> CorrelatedEntry {
    CorrelatedEntry {
        cubin_sha: e.cubin_sha.clone(),
        sm: e.sm.clone(),
        instruction_stride: e.instruction_stride,
        source_lineinfo_present: e.source_lineinfo_present,
        kernels: e.kernels.iter().map(clone_kernel).collect(),
        ptx_lines: e
            .ptx_lines
            .iter()
            .map(|p| PtxLine {
                line_number: p.line_number,
                text: p.text.clone(),
                source: p.source.as_ref().map(clone_source_ref),
            })
            .collect(),
        source_index: e
            .source_index
            .iter()
            .map(|r| SourceIndexRow {
                file: r.file.clone(),
                line: r.line,
                sass_addresses: r.sass_addresses.clone(),
                ptx_line_numbers: r.ptx_line_numbers.clone(),
            })
            .collect(),
        warnings: e.warnings.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    /// JSON Pointer lookup that returns `Value::Null` for missing keys
    /// rather than panicking — keeps test asserts terse while honouring
    /// the workspace `clippy::indexing_slicing` deny lint.
    fn at(v: &serde_json::Value, ptr: &str) -> serde_json::Value {
        v.pointer(ptr).cloned().unwrap_or(serde_json::Value::Null)
    }

    #[test]
    fn sass_instruction_serializes_with_predicate_and_source() -> Result<()> {
        let insn = SassInstruction {
            address: 0x40,
            opcode: "BRA".into(),
            operands: "0x150".into(),
            predicate: Some("@P1".into()),
            control_flow: true,
            source: Some(SourceLineRef {
                file: "k.cu".into(),
                line: 42,
                column: None,
            }),
        };
        let json = serde_json::to_value(&insn)?;
        assert_eq!(at(&json, "/address"), 0x40);
        assert_eq!(at(&json, "/opcode"), "BRA");
        assert_eq!(at(&json, "/predicate"), "@P1");
        assert_eq!(at(&json, "/control_flow"), true);
        assert_eq!(at(&json, "/source/file"), "k.cu");
        assert_eq!(at(&json, "/source/line"), 42);
        Ok(())
    }

    #[test]
    fn sass_instruction_omits_optional_fields_when_absent() -> Result<()> {
        let insn = SassInstruction {
            address: 0,
            opcode: "NOP".into(),
            operands: String::new(),
            predicate: None,
            control_flow: false,
            source: None,
        };
        let json = serde_json::to_value(&insn)?;
        assert!(at(&json, "/predicate").is_null());
        assert!(at(&json, "/source").is_null());
        Ok(())
    }
}
