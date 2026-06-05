//! Source ↔ SASS ↔ PTX correlation for the cubin bytes embedded in
//! each `SourceData`.
//!
//! The data model is agent-facing JSON: per-kernel `SassInstruction`
//! arrays with cubin offsets, opcode, operands, predicate,
//! control-flow flag, and an optional `SourceLineRef`; per-PTX-line
//! `PtxLine` entries with the same source attribution; and a
//! `source_index` that inverts both into a
//! `(file, line) → {sass_addresses, ptx_line_numbers}` table.
//!
//! **Source is the pivot.** PTX ↔ SASS direct mapping isn't exposed
//! by the toolchain — both ends carry source attribution through
//! DWARF (SASS, via `nvdisasm --print-line-info`) and `.loc`
//! directives (PTX, via `cuobjdump --dump-ptx`). The wire format
//! gives agents the two attributions plus the inverted index so the
//! typical "show me the hardware that ran for line N of foo.cu"
//! query is one hop.
//!
//! Acquisition is opt-in (`ReportOptions::correlate`) and shells
//! out to `nvdisasm` (always, when correlation is on) and
//! `cuobjdump --dump-ptx` for the PTX listing. Results are cached in a sidecar
//! directory next to the report:
//!
//! ```text
//! <trace>.ncu-rep
//! <trace>.ncu-rep.veloq/disasm/
//! ├── <cubin_sha_hex>.cubin               raw bytes from SourceData.Binary
//! └── <cubin_sha_hex>.correlated.json     merged SASS + PTX + source_index
//! ```
//!
//! Cache key = `(cubin_sha, instruction_stride)`. The cache always
//! holds the FULL correlated payload; `ReportOptions.correlate.{disasm,
//! ptx, source_index}` only choose which slices the response surfaces.
//! Schema bumps invalidate cleanly (obsolete files get re-written).
//!
//! All line numbers in the wire format are **1-based**. PTX line
//! numbers index into the literal `cuobjdump --dump-ptx` output and
//! include blanks/comments/directive lines (so `text` may be empty).
//!
//! ## Module layout
//!
//! - [`types`] — wire-format structs ([`CorrelatedEntry`],
//!   [`KernelDisasm`], [`SassInstruction`], …) and the per-arch
//!   instruction-stride constants.
//! - [`cache`] — sidecar directory, `<sha>.cubin` extraction,
//!   `<sha>.correlated.json` load/store. JSON-on-disk for human
//!   debuggability of nvdisasm output.
//! - [`tools`] — subprocess runner for `nvdisasm` / `cuobjdump`
//!   plus warning collection.
//! - [`nvdisasm`] — parser for `--emit-json` and
//!   `--print-line-info` outputs.
//! - [`cuobjdump`] — parser for `--dump-ptx`.
//! - [`correlate`] — top-level acquisition pipeline +
//!   `source_index` inversion.

mod cache;
mod correlate;
mod cuobjdump;
mod nvdisasm;
mod tools;
mod types;

// Wire-format structs surfaced for the `disasm` verb module
// (`crate::disasm`) and the `source_metrics` rollup, which reuse
// `SourceLineRef` and the correlated payload shape.
pub use types::{
    CorrelatedEntry, KernelDisasm, PtxLine, SassInstruction, SourceIndexRow, SourceLineRef,
};

// `pub(crate)` so the top-level `disasm.rs` verb module can drive the
// same per-cubin cache + acquisition pipeline.
pub(crate) use cache::{
    correlated_cache_path, cubin_sha, extract_and_cache_cubin, load_cached, write_cache,
};
pub(crate) use correlate::{acquire_correlated, build_source_index};
