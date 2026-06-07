//! Parse `nvdisasm`'s two output modes:
//!
//! - `--emit-json` → positional SASS-instruction list (no addresses
//!   carried; we reconstruct them from `kernel_start` + index ×
//!   `instruction_stride`).
//! - `--print-line-info` → text dump with `// File "...", line N`
//!   annotations followed by `/*HEXADDR*/`-prefixed instruction
//!   lines. We walk it once to build an `address → SourceLineRef` map
//!   that the JSON-mode pass joins against.
//!
//! Both passes run once per cubin during disasm acquisition; the
//! join lands in [`super::types::SassInstruction::source`].

use std::collections::HashMap;

use crate::error::{NcuSourceError, NcuSourceResult};

use super::types::{KernelDisasm, SassInstruction, SourceLineRef, clone_source_ref};

/// Decode nvdisasm's `--emit-json` payload into per-kernel SASS lists,
/// joining each instruction against `line_map` (built from
/// [`parse_line_annotations`]) for source attribution.
pub fn parse_emit_json(
    stdout: &str,
    line_map: &HashMap<u64, SourceLineRef>,
    instruction_stride: u64,
) -> NcuSourceResult<Vec<KernelDisasm>> {
    let json_start = match stdout.find('[') {
        Some(i) => i,
        None => return Ok(Vec::new()),
    };
    let payload = match stdout.get(json_start..) {
        Some(s) => s,
        None => return Ok(Vec::new()),
    };
    let v: serde_json::Value =
        serde_json::from_str(payload).map_err(NcuSourceError::nvdisasm_json_decode)?;
    let arr = v
        .as_array()
        .ok_or(NcuSourceError::NvdisasmTopLevelNotArray)?;
    let kernel_list = arr
        .get(1)
        .and_then(|x| x.as_array())
        .ok_or(NcuSourceError::NvdisasmKernelArrayMissing)?;
    let mut kernels = Vec::with_capacity(kernel_list.len());
    for (i, k) in kernel_list.iter().enumerate() {
        kernels.push(parse_kernel(k, i, line_map, instruction_stride)?);
    }
    Ok(kernels)
}

fn parse_kernel(
    v: &serde_json::Value,
    index: usize,
    line_map: &HashMap<u64, SourceLineRef>,
    instruction_stride: u64,
) -> NcuSourceResult<KernelDisasm> {
    let function_name = v
        .get("function-name")
        .and_then(|x| x.as_str())
        .ok_or_else(|| NcuSourceError::nvdisasm_kernel_function_name_missing(index))?
        .to_string();
    let start = v
        .get("start")
        .and_then(|x| x.as_u64())
        .ok_or_else(|| NcuSourceError::nvdisasm_kernel_start_missing(index))?;
    let length = v.get("length").and_then(|x| x.as_u64()).unwrap_or(0);
    let empty: Vec<serde_json::Value> = Vec::new();
    let raw = v
        .get("sass-instructions")
        .and_then(|x| x.as_array())
        .unwrap_or(&empty);
    let instructions = raw
        .iter()
        .enumerate()
        .map(|(i, insn)| parse_instruction(insn, start, i, line_map, instruction_stride))
        .collect();
    Ok(KernelDisasm {
        key: format!("kernel|{function_name}"),
        function_name,
        start,
        length,
        instructions,
    })
}

fn parse_instruction(
    v: &serde_json::Value,
    kernel_start: u64,
    index: usize,
    line_map: &HashMap<u64, SourceLineRef>,
    instruction_stride: u64,
) -> SassInstruction {
    let address = kernel_start.saturating_add((index as u64).saturating_mul(instruction_stride));
    let opcode = v
        .get("opcode")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let operands = v
        .get("operands")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let predicate = v
        .get("predicate")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string());
    let control_flow = v
        .get("other-attributes")
        .and_then(|x| x.get("control-flow"))
        .and_then(|x| x.as_str())
        .map(|s| s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let source = line_map.get(&address).map(clone_source_ref);
    SassInstruction {
        address,
        opcode,
        operands,
        predicate,
        control_flow,
        source,
    }
}

/// Walk nvdisasm's `--print-line-info` text output and build an
/// address → SourceLineRef map. Annotations look like
/// `//## File "/path/foo.cu", line 42` (modern) or
/// `// File "/path/foo.cu", line 42` (older variant), and each one
/// applies to the immediately-following `/*HEXADDR*/` instruction
/// lines until a new annotation appears.
pub fn parse_line_annotations(text: &str) -> HashMap<u64, SourceLineRef> {
    let mut map: HashMap<u64, SourceLineRef> = HashMap::new();
    let mut current: Option<SourceLineRef> = None;
    for raw in text.lines() {
        let line = raw.trim_start();
        if let Some(loc) = parse_file_line_annotation(line) {
            current = Some(loc);
            continue;
        }
        if let Some(addr) = parse_address_marker(line)
            && let Some(loc) = current.as_ref()
        {
            map.insert(addr, clone_source_ref(loc));
        }
    }
    map
}

/// Parse a `//## File "...", line N` / `// File "...", line N`
/// annotation. Returns `None` if the line doesn't match. Hand-
/// rolled splitting; avoids pulling in `regex` for one pattern.
fn parse_file_line_annotation(line: &str) -> Option<SourceLineRef> {
    let stripped = line
        .strip_prefix("//## ")
        .or_else(|| line.strip_prefix("// "))?;
    let stripped = stripped.strip_prefix("File ")?;
    let quote_start = stripped.find('"')?;
    let after_open = stripped.get(quote_start.checked_add(1)?..)?;
    let quote_end = after_open.find('"')?;
    let file = after_open.get(..quote_end)?.to_string();
    let after_close = after_open.get(quote_end.checked_add(1)?..)?;
    let comma_pos = after_close.find(",")?;
    let after_comma = after_close.get(comma_pos.checked_add(1)?..)?;
    let after_line_kw = after_comma.trim_start().strip_prefix("line ")?;
    let (line_str, rest) = match after_line_kw.find(|c: char| !c.is_ascii_digit()) {
        Some(i) => (after_line_kw.get(..i)?, after_line_kw.get(i..)?),
        None => (after_line_kw, ""),
    };
    let line_num: u32 = line_str.parse().ok()?;
    let column = parse_optional_column(rest);
    Some(SourceLineRef {
        file,
        line: line_num,
        column,
    })
}

fn parse_optional_column(rest: &str) -> Option<u32> {
    let r = rest.trim_start_matches(',').trim_start();
    let r = r
        .strip_prefix("col ")
        .or_else(|| r.strip_prefix("column "))?;
    let end = r.find(|c: char| !c.is_ascii_digit()).unwrap_or(r.len());
    r.get(..end)?.parse().ok()
}

/// Parse `/*HEXADDR*/` instruction-address markers — e.g.
/// `/*0040*/`. Returns the address as a `u64`, or `None` if the
/// line doesn't start with the marker.
fn parse_address_marker(line: &str) -> Option<u64> {
    let after_open = line.strip_prefix("/*")?;
    let close = after_open.find("*/")?;
    let hex = after_open.get(..close)?;
    u64::from_str_radix(hex, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    // A non-default stride (production uses 16); 8 proves `parse_emit_json`
    // applies the caller-supplied stride when reconstructing addresses.
    const TEST_STRIDE: u64 = 8;

    #[test]
    fn parse_emit_json_uses_requested_instruction_stride() -> Result<()> {
        let stdout = r#"[
            {},
            [{
                "function-name": "_Z6kernelv",
                "start": 64, "length": 16,
                "sass-instructions": [
                    { "opcode": "NOP", "operands": "" },
                    { "opcode": "RET", "operands": "" }
                ]
            }]
        ]"#;
        let kernels = parse_emit_json(stdout, &HashMap::new(), TEST_STRIDE)?;
        let kernel = kernels
            .first()
            .ok_or_else(|| anyhow::anyhow!("expected kernel"))?;
        assert_eq!(
            kernel
                .instructions
                .first()
                .ok_or_else(|| anyhow::anyhow!("i0"))?
                .address,
            64
        );
        assert_eq!(
            kernel
                .instructions
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("i1"))?
                .address,
            72
        );
        Ok(())
    }
}
