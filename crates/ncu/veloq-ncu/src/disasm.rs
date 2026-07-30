//! `veloq ncu disasm --row-id launch:<idx>` — SASS / PTX / source-index
//! correlation for the cubin a single launch ran out of.
//!
//! Hybrid disasm: the launch
//! list comes from the `ncu_report` native sidecar
//! ([`crate::native::cache`]), but the SASS itself comes from the
//! **embedded ELF cubin** ([`crate::native::cubin`]), fed to veloq's
//! existing `nvdisasm`/`cuobjdump` pipeline ([`crate::disasm_pipeline`]). This
//! keeps `predicate` + `control_flow` (authoritative, from
//! `nvdisasm --emit-json`) and full PTX — `ncu_report`'s `sass_by_pc`
//! gives neither.
//!
//! Launch→cubin join is by **kernel symbol** (`kernel_mangled` → the
//! cubin defining that function); the per-cubin
//! `<sha>.correlated.json` cache is reused, so a committed cache makes
//! this verb NCU-free *and* nvdisasm-free at query time.

use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;

use crate::disasm_pipeline::{
    CorrelatedEntry, KernelDisasm, PtxLine, SourceIndexRow, SourceLineRef,
};
use crate::disasm_pipeline::{
    acquire_correlated, build_source_index, correlated_cache_path, cubin_sha,
    extract_and_cache_cubin, load_cached, write_cache,
};
use crate::error::{NcuSourceError, NcuSourceResult};
use crate::native::{NativeSidecar, NativeSourceRef, cache, cubin};

/// SASS instruction stride: 16 bytes on Volta and later.
const INSTRUCTION_STRIDE: u64 = 16;

/// `veloq ncu disasm` response payload.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct DisasmResponse {
    /// Number of disassembled kernels returned — every function in the
    /// launch's cubin. `0` when no embedded cubin defines the launch's
    /// kernel symbol.
    pub count: usize,
    /// Same as `count`; disasm doesn't paginate.
    pub total_matched: usize,
    /// Canonical primary table. Each row is one kernel's SASS
    /// listing, in cubin order.
    pub rows: Vec<KernelDisasm>,
    pub auxiliary: DisasmAuxiliary,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct DisasmAuxiliary {
    /// The row_id this response is for (echoed for diagnostics).
    pub row_id: String,
    /// SHA-256 of the ELF-scanned cubin bytes. Joins to the per-cubin
    /// cache file under `<file>.veloq/disasm/`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cubin_sha: Option<String>,
    /// Compute capability label (`sm_120` etc.) the cubin was
    /// compiled for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sm: Option<String>,
    /// Bytes between consecutive SASS instructions in the cubin
    /// (16 on Volta+). Lets agents compute per-address offsets
    /// without a per-instruction lookup.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instruction_stride: Option<u64>,
    /// `true` when at least one SASS instruction carried DWARF
    /// line-info attribution.
    pub source_lineinfo_present: bool,
    /// `cuobjdump --dump-ptx` lines.
    pub ptx_lines: Vec<PtxLine>,
    /// Inverted `(file, line) → {sass_addresses, ptx_line_numbers}`
    /// index for the "show me everything that ran for line N" query.
    pub source_index: Vec<SourceIndexRow>,
    /// Warnings from nvdisasm / cuobjdump / cache decode, surfaced
    /// verbatim so an agent can flag partial disasm.
    pub warnings: Vec<String>,
}

/// Resolve `row_id` → launch → kernel symbol, find the embedded cubin
/// that defines it, and project that cubin's correlated disasm.
pub fn run<P: AsRef<Path>>(path: P, row_id: &str) -> NcuSourceResult<DisasmResponse> {
    let idx = crate::row_id::parse_launch_idx(row_id)?;
    let path = path.as_ref();
    let sidecar = cache::build_or_load(path)?;
    let n_launches = sidecar.launches.len();
    if idx >= n_launches {
        return Err(NcuSourceError::launch_row_id_out_of_range(
            row_id, idx, n_launches,
        ));
    }
    let Some(launch) = sidecar.launches.get(idx) else {
        return Err(NcuSourceError::launch_vanished_after_bounds_check(idx));
    };
    let mangled = &launch.kernel_mangled;

    // Locate the embedded cubin that defines this launch's kernel
    // symbol. Cubins are position-independent, so the symbol — not
    // `cubin_load_base` — is the join key.
    let cubins = cubin::extract_cuda_cubins(path)?;
    let Some(cub) = cubins.iter().find(|c| c.defines(mangled)) else {
        return Ok(empty_response(
            row_id,
            format!(
                "no embedded cubin defines kernel symbol {mangled} (cubins found: {})",
                cubins.len()
            ),
        ));
    };

    // Reuse the existing per-cubin acquisition + cache pipeline.
    let sha = cubin_sha(&cub.bytes);
    let cache_path = correlated_cache_path(path, &sha);
    let sm = launch.sm_label();
    let mut entry = if let Some(loaded) = load_cached(&cache_path, &sha, INSTRUCTION_STRIDE)? {
        loaded
    } else {
        let cubin_path = extract_and_cache_cubin(path, &sha, &cub.bytes)?;
        let mut produced = acquire_correlated(&cubin_path, INSTRUCTION_STRIDE)?;
        produced.cubin_sha = sha.clone();
        produced.sm = sm.clone();
        write_cache(&cache_path, &produced)?;
        produced
    };
    entry.cubin_sha = sha;
    entry.sm = sm;

    // Overlay authoritative source attribution from `ncu_report`
    // `source_info` (the sidecar's `disasm[].source`). nvdisasm's
    // `--print-line-info` map collides across multi-kernel cubins (both
    // `.text` sections are 0-based), mis-attributing one kernel's SASS
    // to another's source; `source_info` is per-PC and correct. We keep
    // nvdisasm's opcode/operands/predicate/control_flow and replace only
    // the source line. See the multi-kernel-source-attribution note.
    overlay_source_info(&sidecar, &mut entry);

    Ok(project(row_id, entry))
}

/// Replace each instruction's `source` with `ncu_report`'s `source_info`
/// (from the sidecar), keyed by `(kernel mangled name, cubin-relative
/// address)`, then rebuild the inverted index + lineinfo flag from the
/// corrected sources. A cubin kernel with no matching launch (not
/// profiled) gets `source: None` rather than nvdisasm's unreliable line.
fn overlay_source_info(sidecar: &NativeSidecar, entry: &mut CorrelatedEntry) {
    let mut by_kernel: HashMap<&str, HashMap<u64, &NativeSourceRef>> = HashMap::new();
    for launch in &sidecar.launches {
        if let Some(disasm) = launch.disasm.as_ref() {
            let map = by_kernel.entry(launch.kernel_mangled.as_str()).or_default();
            for insn in disasm {
                if let Some(src) = insn.source.as_ref() {
                    map.insert(insn.address, src);
                }
            }
        }
    }
    for kernel in &mut entry.kernels {
        let map = by_kernel.get(kernel.function_name.as_str());
        for insn in &mut kernel.instructions {
            insn.source = map
                .and_then(|m| m.get(&insn.address))
                .map(|s| SourceLineRef {
                    file: s.file.clone(),
                    line: s.line,
                    column: None,
                });
        }
    }
    entry.source_index = build_source_index(&entry.kernels, &entry.ptx_lines);
    entry.source_lineinfo_present = entry
        .kernels
        .iter()
        .any(|k| k.instructions.iter().any(|i| i.source.is_some()));
}

/// Build a response for the no-cubin / no-symbol case. `rows` is empty;
/// `auxiliary.warnings` carries the reason.
fn empty_response(row_id: &str, reason: String) -> DisasmResponse {
    DisasmResponse {
        count: 0,
        total_matched: 0,
        rows: Vec::new(),
        auxiliary: DisasmAuxiliary {
            row_id: row_id.to_string(),
            cubin_sha: None,
            sm: None,
            instruction_stride: None,
            source_lineinfo_present: false,
            ptx_lines: Vec::new(),
            source_index: Vec::new(),
            warnings: vec![reason],
        },
    }
}

/// Project a [`CorrelatedEntry`] (all kernels in the launch's cubin)
/// into the response shape, filling each kernel's `key`.
fn project(row_id: &str, mut entry: CorrelatedEntry) -> DisasmResponse {
    for k in &mut entry.kernels {
        if k.key.is_empty() {
            k.key = format!("kernel|{}", k.function_name);
        }
    }
    let count = entry.kernels.len();
    DisasmResponse {
        count,
        total_matched: count,
        rows: entry.kernels,
        auxiliary: DisasmAuxiliary {
            row_id: row_id.to_string(),
            cubin_sha: Some(entry.cubin_sha),
            sm: entry.sm,
            instruction_stride: Some(entry.instruction_stride),
            source_lineinfo_present: entry.source_lineinfo_present,
            ptx_lines: entry.ptx_lines,
            source_index: entry.source_index,
            warnings: entry.warnings,
        },
    }
}
