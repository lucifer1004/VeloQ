//! Subprocess plumbing for `nvdisasm` + `cuobjdump`. Each tool is
//! invoked once per cubin during disasm-acquisition; this module
//! captures stdout, surfaces warnings from stderr (and the leading
//! `<tool> warning :` lines that some captures put on stdout before
//! the real payload), and rejects non-zero exits with a contextful
//! `anyhow::Error` so the caller can decide whether to bail or
//! degrade.
//!
//! Tool-missing errors land here too: `Command::new` itself doesn't
//! check `PATH`, but `.output()` does — failure to spawn surfaces as
//! the `io::Error` from `output()`, wrapped via `with_context` with
//! the binary name and cubin path so an agent's envelope chain
//! pinpoints what's missing.

use anyhow::{Context, Result, bail};
use std::path::Path;
use std::process::Command;

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
pub fn run_tool(bin: &'static str, cubin_path: &Path, args: &[&str]) -> Result<ToolOutput> {
    let mut cmd = Command::new(bin);
    cmd.args(args).arg(cubin_path);
    let output = cmd
        .output()
        .with_context(|| format!("invoking {bin} {args:?} on {}", cubin_path.display()))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "{bin} {args:?} on {} exited with status {}: {stderr}",
            cubin_path.display(),
            output.status,
        );
    }
    let stdout =
        String::from_utf8(output.stdout).with_context(|| format!("{bin} stdout was not UTF-8"))?;
    let stderr =
        String::from_utf8(output.stderr).with_context(|| format!("{bin} stderr was not UTF-8"))?;
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
