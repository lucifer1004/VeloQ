//! Subprocess plumbing for `nvdisasm` + `cuobjdump`. Each tool is
//! invoked once per cubin during disasm-acquisition; this module
//! captures stdout, surfaces warnings from stderr (and the leading
//! `<tool> warning :` lines that some captures put on stdout before
//! the real payload), and rejects non-zero exits with a typed
//! [`crate::error::NcuSourceError`] so the caller can decide whether
//! to bail or degrade.
//!
//! Tool-missing errors land here too: `Command::new` itself doesn't
//! check `PATH`, but `.output()` does — failure to spawn surfaces as
//! the `io::Error` from `output()`, wrapped via `with_context` with
//! the binary name and cubin path so an agent's envelope chain
//! pinpoints what's missing.

use std::path::Path;
use std::process::Command;

use crate::error::{NcuSourceError, NcuSourceResult};

/// Captured tool output. `warnings` is the union of stderr lines and
/// leading non-data lines on stdout (the JSON / PTX payload starts
/// after a few `<tool> warning :` lines in some captures).
pub struct ToolOutput {
    pub stdout: String,
    pub warnings: Vec<String>,
}

/// Run `bin args... cubin_path` and capture its output. Non-zero exit
/// or non-UTF-8 output is an error; stderr always lands in `warnings`
/// regardless of the exit code so partial captures still expose what
/// the tool complained about.
pub fn run_tool(
    bin: &'static str,
    cubin_path: &Path,
    args: &[&str],
) -> NcuSourceResult<ToolOutput> {
    let mut cmd = Command::new(bin);
    cmd.args(args).arg(cubin_path);
    let output = cmd
        .output()
        .map_err(|source| NcuSourceError::disasm_tool_spawn(bin, cubin_path, args, source))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(NcuSourceError::disasm_tool_failed(
            bin,
            cubin_path,
            args,
            output.status.to_string(),
            stderr.trim().to_string(),
        ));
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|source| NcuSourceError::disasm_tool_output_utf8(bin, "stdout", source))?;
    let stderr = String::from_utf8(output.stderr)
        .map_err(|source| NcuSourceError::disasm_tool_output_utf8(bin, "stderr", source))?;
    let warnings = collect_warnings(bin, &stdout, &stderr);
    Ok(ToolOutput { stdout, warnings })
}

/// Collect tool warning lines from stderr plus any leading
/// `<tool> warning :` / `<tool> info :` lines on stdout. For
/// nvdisasm in JSON mode, those land before the `[`; for cuobjdump,
/// before the actual PTX payload.
pub fn collect_warnings(bin: &str, stdout: &str, stderr: &str) -> Vec<String> {
    let mut warnings = Vec::new();
    for line in stderr.lines() {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            warnings.push(trimmed.to_string());
        }
    }
    let warning_prefix = format!("{bin} warning");
    let info_prefix = format!("{bin} info");
    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(&warning_prefix) || trimmed.starts_with(&info_prefix) {
            warnings.push(trimmed.to_string());
        } else if !trimmed.is_empty() {
            break;
        }
    }
    warnings
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use veloq_core::VeloqDiagnostic;

    #[test]
    fn missing_tool_error_is_typed() -> Result<()> {
        let cubin = std::env::temp_dir().join(format!(
            "veloq-ncu-missing-tool-{}.cubin",
            std::process::id()
        ));
        let err = run_tool(
            "veloq-ncu-definitely-missing-tool",
            &cubin,
            &["--emit-json"],
        )
        .err()
        .ok_or_else(|| anyhow::anyhow!("missing tool should error"))?;
        assert_eq!(err.code().as_str(), "ncu.input.disasm-tool-spawn");
        Ok(())
    }
}
