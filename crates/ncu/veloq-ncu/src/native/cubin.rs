//! Proto-free extraction of the ELF cubins embedded in a `.ncu-rep`.
//!
//! `ncu` imports the device ELF cubins into the report by default
//! (`--import-sass`). They are ordinary CUDA ELF objects
//! (`e_machine == EM_CUDA`), so we locate them by scanning the raw
//! report bytes for the `\x7fELF` magic and parsing each candidate
//! with the `object` crate — **no NVIDIA proto schema involved**.
//! Reading a file for the public ELF format redistributes none of
//! NVIDIA's IP (unlike vendoring `.proto`), which is what keeps the
//! open-source footing while feeding veloq's existing
//! `nvdisasm`/`cuobjdump` disasm pipeline (`crate::disasm_pipeline`).
//!
//! The join from a launch to its cubin is by **kernel symbol**
//! (`kernel_mangled` → a function symbol in exactly one cubin); cubins
//! are position-independent, so the runtime load address can't be the
//! key (that's why `cubin_load_base` comes from `ncu_report`).

use object::read::elf::{ElfFile64, FileHeader};
use object::{Endianness, Object, ObjectSection, ObjectSegment, ObjectSymbol, SymbolKind};
use std::fs;
use std::path::Path;

use crate::error::{NcuSourceError, NcuSourceResult};

/// `e_machine` value for NVIDIA CUDA ELF objects.
const EM_CUDA: u16 = 190;

/// One embedded cubin: its exact ELF bytes (deterministic, so the
/// `disasm_pipeline` per-cubin cache key is stable across runs) plus
/// the global function symbol names it defines (the launch→cubin join
/// keys).
#[derive(Debug, Clone)]
pub struct ExtractedCubin {
    pub bytes: Vec<u8>,
    pub symbols: Vec<String>,
}

impl ExtractedCubin {
    /// `true` when this cubin defines a function named `mangled`.
    pub fn defines(&self, mangled: &str) -> bool {
        self.symbols.iter().any(|s| s == mangled)
    }
}

/// Scan `report` for every embedded CUDA ELF cubin and return each
/// with its exact bytes + function symbols. Non-CUDA ELFs and
/// unparseable candidates are skipped.
///
/// Committed-sidecar mode (see [`crate::native::cache`] module docs):
/// when the source `.ncu-rep` is absent, the cubins are loaded from the
/// committed `<report>.veloq/disasm/<sha>.cubin` files instead. The
/// extracted cubins are clean compiled device code (no host/network
/// strings), so they are committable where the report itself is not.
pub fn extract_cuda_cubins(report: &Path) -> NcuSourceResult<Vec<ExtractedCubin>> {
    if !report.exists() {
        return load_committed_cubins(report);
    }
    let data = fs::read(report)
        .map_err(|source| NcuSourceError::cubin_report_read(report.display(), source))?;
    let mut out = Vec::new();
    for off in elf_magic_offsets(&data) {
        let Some(rest) = data.get(off..) else {
            continue;
        };
        match parse_one(rest) {
            Ok(Some(cubin)) => out.push(cubin),
            Ok(None) => {} // valid ELF but not a CUDA cubin — skip
            Err(_) => {}   // not a parseable ELF at this offset — skip
        }
    }
    Ok(out)
}

/// Load committed `<report>.veloq/disasm/<sha>.cubin` files (the
/// report-absent path). The bytes are exactly what `extract_and_cache_cubin`
/// wrote, so `parse_one` recovers identical bytes + symbols and the sha
/// re-derives to the filename — the launch→cubin join and the per-cubin
/// correlated cache key both hold. A missing disasm dir yields an empty
/// list (the caller then reports no cubin defines the kernel symbol).
fn load_committed_cubins(report: &Path) -> NcuSourceResult<Vec<ExtractedCubin>> {
    let dir = veloq_core::artifact_dir_for(report).join("disasm");
    let entries = match fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return Ok(Vec::new()),
    };
    let mut out = Vec::new();
    for entry in entries {
        let path = entry
            .map_err(|source| {
                NcuSourceError::cubin_committed_dir_entry_read(dir.display(), source)
            })?
            .path();
        if path.extension().and_then(|e| e.to_str()) != Some("cubin") {
            continue;
        }
        let bytes = fs::read(&path)
            .map_err(|source| NcuSourceError::cubin_committed_read(path.display(), source))?;
        if let Ok(Some(cubin)) = parse_one(&bytes) {
            out.push(cubin);
        }
    }
    Ok(out)
}

/// Offsets of every `\x7fELF` magic in `data`.
fn elf_magic_offsets(data: &[u8]) -> Vec<usize> {
    const MAGIC: &[u8; 4] = b"\x7fELF";
    let mut offs = Vec::new();
    if data.len() < MAGIC.len() {
        return offs;
    }
    for i in 0..=data.len() - MAGIC.len() {
        if data.get(i..i + MAGIC.len()) == Some(MAGIC.as_slice()) {
            offs.push(i);
        }
    }
    offs
}

/// Parse an ELF starting at the front of `rest` (which runs to EOF).
/// Returns `Ok(Some)` for a CUDA cubin, `Ok(None)` for a non-CUDA ELF.
fn parse_one(rest: &[u8]) -> NcuSourceResult<Option<ExtractedCubin>> {
    let elf = ElfFile64::<Endianness>::parse(rest).map_err(NcuSourceError::cubin_elf_parse)?;
    let endian = elf.endian();
    let header = elf.elf_header();
    if header.e_machine(endian) != EM_CUDA {
        return Ok(None);
    }

    // Exact byte length = the furthest extent of any section file range,
    // any segment file range, or the section-header table. ELF magic
    // can appear inside other blocks, so trust the structure, not a
    // fixed stride.
    let mut end: u64 = header.e_shoff(endian).saturating_add(
        u64::from(header.e_shnum(endian)).saturating_mul(u64::from(header.e_shentsize(endian))),
    );
    for seg in elf.segments() {
        let (o, sz) = seg.file_range();
        end = end.max(o.saturating_add(sz));
    }
    for sec in elf.sections() {
        if let Some((o, sz)) = sec.file_range() {
            end = end.max(o.saturating_add(sz));
        }
    }
    let end = usize::try_from(end)
        .map_err(|source| NcuSourceError::cubin_length_overflow(end, source))?;
    let Some(bytes) = rest.get(..end) else {
        return Err(NcuSourceError::cubin_length_exceeds_available_bytes(
            end,
            rest.len(),
        ));
    };

    let mut symbols: Vec<String> = elf
        .symbols()
        .filter(|s| s.kind() == SymbolKind::Text)
        .filter_map(|s| s.name().ok())
        .filter(|n| !n.is_empty())
        .map(|n| n.to_string())
        .collect();
    symbols.sort();
    symbols.dedup();

    Ok(Some(ExtractedCubin {
        bytes: bytes.to_vec(),
        symbols,
    }))
}
