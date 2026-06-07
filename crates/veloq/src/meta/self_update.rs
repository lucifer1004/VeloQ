//! `veloq self-update` — update the running binary *and* the bundled
//! Claude Code skills to the latest GitHub release.
//!
//! A release ships both the binary (per-target archives) and a
//! `veloq-skills.tar.gz` of `.claude/skills/{nsys,ncu}-profile-analysis`.
//! `scripts/install.sh` installs both; this verb keeps them in lockstep so
//! a self-updated binary doesn't leave stale skills behind.
//!
//! - default: update binary + skills.
//! - `--no-binary` / `--no-skills`: update just one half (mirrors
//!   `install.sh`'s flags).
//! - `--skills-dir <path>`: install skills under a different root — a
//!   project-local `.claude`, an agent-agnostic `.agents`, etc. `skills/` is
//!   appended unless already present, so the agent root or the full skills
//!   dir both work (overrides `VELOQ_SKILLS_DIR`).
//! - `--check`: report whether a newer release exists, with no side
//!   effects.
//!
//! Unlike a query verb this touches the network and the filesystem, but it
//! keeps the meta-verb JSON contract: the `self_update` crate's own status
//! output and progress bar are suppressed and confirmation is disabled, so
//! stdout carries only the envelope and the run never blocks on a prompt.

use clap::{Arg, ArgAction, ArgMatches, Command};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use veloq_core::OutputFormat;

use super::{MetaError, MetaResult, emit_meta_error, emit_or_error};

const VERB: &str = "self-update";
const REPO_OWNER: &str = "lucifer1004";
const REPO_NAME: &str = "veloq";
const BIN_NAME: &str = "veloq";
/// Path of the binary inside each release archive. Mirrors the directory
/// layout `.github/workflows/release.yml` packs:
/// `veloq-v<version>-<target>/veloq[.exe]`. `self_update` expands the
/// `{{ version }}` / `{{ target }}` / `{{ bin }}` placeholders at runtime.
const BIN_PATH_IN_ARCHIVE: &str = "veloq-v{{ version }}-{{ target }}/{{ bin }}";
/// Skills archive attached to every release. Its entries are prefixed
/// `.claude/skills/<name>/...` (see the `skills` job in release.yml).
const SKILLS_ASSET: &str = "veloq-skills.tar.gz";

#[derive(Debug, Serialize)]
struct SelfUpdatePayload {
    current_version: String,
    latest_version: String,
    update_available: bool,
    /// True only when the binary was actually replaced (not when it was
    /// already current or `--no-binary` was passed).
    binary_updated: bool,
    /// True when the bundled skills were (re)installed from the release.
    skills_updated: bool,
    /// Where the skills were installed, when `skills_updated`.
    skills_dir: Option<String>,
    /// True for `--check` (no side effects performed).
    checked_only: bool,
}

pub fn cli() -> Command {
    Command::new(VERB)
        .about("Update the veloq binary and bundled skills to the latest GitHub release")
        .arg(
            Arg::new("check")
                .long("check")
                .action(ArgAction::SetTrue)
                .help("Report whether a newer release exists, without installing it"),
        )
        .arg(
            Arg::new("no-binary")
                .long("no-binary")
                .action(ArgAction::SetTrue)
                .help("Update only the bundled skills, not the binary"),
        )
        .arg(
            Arg::new("no-skills")
                .long("no-skills")
                .action(ArgAction::SetTrue)
                .help("Update only the binary, not the bundled skills"),
        )
        .arg(
            Arg::new("skills-dir")
                .long("skills-dir")
                .value_name("PATH")
                .help(
                    "Install skills under this directory instead of the default \
                     (~/.claude). `skills/` is appended unless already present, so \
                     pass an agent root (.agents, ~/.claude) or a full skills dir. \
                     Overrides VELOQ_SKILLS_DIR.",
                ),
        )
}

pub fn run(matches: &ArgMatches, fmt: OutputFormat) -> MetaResult<i32> {
    let current = env!("CARGO_PKG_VERSION");
    let skills_dir = matches.get_one::<String>("skills-dir").map(PathBuf::from);
    let outcome = self_update(
        current,
        matches.get_flag("check"),
        !matches.get_flag("no-binary"),
        !matches.get_flag("no-skills"),
        skills_dir,
    );
    match outcome {
        Ok(payload) => Ok(emit_or_error(fmt, VERB, None, None, payload)),
        Err(err) => {
            emit_meta_error(fmt, VERB, None, &err);
            Ok(1)
        }
    }
}

fn self_update(
    current: &str,
    check_only: bool,
    want_binary: bool,
    want_skills: bool,
    skills_dir_override: Option<PathBuf>,
) -> MetaResult<SelfUpdatePayload> {
    // One release lookup powers the check, the skills download URL, and the
    // reported `latest_version`.
    let latest = latest_release()?;
    let latest_version = strip_v(&latest.version);
    let update_available = is_newer(current, &latest_version)?;

    if check_only {
        return Ok(SelfUpdatePayload {
            current_version: strip_v(current),
            latest_version,
            update_available,
            binary_updated: false,
            skills_updated: false,
            skills_dir: None,
            checked_only: true,
        });
    }

    let binary_updated = if want_binary {
        perform_binary_update(current)?
    } else {
        false
    };

    let skills_dir = if want_skills {
        Some(update_skills(
            &latest_version,
            skills_dir_override.as_deref(),
        )?)
    } else {
        None
    };

    Ok(SelfUpdatePayload {
        current_version: strip_v(current),
        latest_version,
        update_available,
        binary_updated,
        skills_updated: skills_dir.is_some(),
        skills_dir: skills_dir.map(|d| d.display().to_string()),
        checked_only: false,
    })
}

/// Latest release (the first entry GitHub returns, newest-first).
fn latest_release() -> MetaResult<self_update::update::Release> {
    self_update::backends::github::ReleaseList::configure()
        .repo_owner(REPO_OWNER)
        .repo_name(REPO_NAME)
        .build()
        .map_err(MetaError::self_update_release_lookup_config)?
        .fetch()
        .map_err(MetaError::self_update_release_fetch)?
        .into_iter()
        .next()
        .ok_or(MetaError::SelfUpdateReleaseMissing)
}

/// Download the matching release archive and atomically replace the running
/// binary. Output is suppressed so the JSON envelope is the only thing on
/// stdout; the run never prompts (`no_confirm`). Returns whether the binary
/// actually changed.
fn perform_binary_update(current: &str) -> MetaResult<bool> {
    let status = self_update::backends::github::Update::configure()
        .repo_owner(REPO_OWNER)
        .repo_name(REPO_NAME)
        .bin_name(BIN_NAME)
        .bin_path_in_archive(BIN_PATH_IN_ARCHIVE)
        .current_version(current)
        .show_output(false)
        .show_download_progress(false)
        .no_confirm(true)
        .build()
        .map_err(MetaError::self_update_binary_config)?
        .update()
        .map_err(MetaError::self_update_binary_install)?;
    Ok(!matches!(status, self_update::Status::UpToDate(_)))
}

/// Download `veloq-skills.tar.gz` for `version` and install the two skills
/// under the Claude Code skills directory, overwriting any prior copy.
/// Returns the skills directory.
fn update_skills(version: &str, skills_dir_override: Option<&Path>) -> MetaResult<PathBuf> {
    let skills_dir = resolve_skills_dir(skills_dir_override)?;
    // Public release-download URL — a direct link (302 -> CDN) that needs no
    // auth header, unlike the API asset URL self_update stores.
    let url = format!(
        "https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/v{version}/{SKILLS_ASSET}"
    );

    let tmp = self_update::TempDir::new().map_err(MetaError::self_update_skills_temp_dir)?;
    let archive = tmp.path().join(SKILLS_ASSET);
    {
        let mut file = fs::File::create(&archive).map_err(|source| {
            MetaError::self_update_skills_archive_create(archive.display(), source)
        })?;
        self_update::Download::from_url(&url)
            .download_to(&mut file)
            .map_err(|source| {
                MetaError::self_update_skills_download(SKILLS_ASSET, url.clone(), source)
            })?;
    }

    let extract_dir = tmp.path().join("extract");
    self_update::Extract::from_source(&archive)
        .extract_into(&extract_dir)
        .map_err(|source| MetaError::self_update_skills_extract(archive.display(), source))?;

    // Tarball entries are `.claude/skills/<name>/...`.
    let staged = extract_dir.join(".claude").join("skills");
    install_staged_skills(&staged, &skills_dir)?;
    Ok(skills_dir)
}

/// Resolve the skills install directory. Precedence, highest first:
/// `--skills-dir` (`override_dir`), `$VELOQ_SKILLS_DIR`, then the default
/// agent home `$HOME/.claude` (`%USERPROFILE%` on Windows) — matching
/// `scripts/install.sh`. The chosen base is normalized by
/// [`with_skills_leaf`], so a caller may pass either the agent root
/// (`.agents`, `~/.claude`) or the full skills dir (`.agents/skills`).
fn resolve_skills_dir(override_dir: Option<&Path>) -> MetaResult<PathBuf> {
    let base = if let Some(dir) = override_dir {
        dir.to_path_buf()
    } else if let Some(dir) = std::env::var_os("VELOQ_SKILLS_DIR") {
        PathBuf::from(dir)
    } else {
        let home = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .ok_or(MetaError::SelfUpdateHomeMissing)?;
        PathBuf::from(home).join(".claude")
    };
    Ok(with_skills_leaf(base))
}

/// Skills live under `<dir>/skills/` by convention (`.claude/skills`,
/// `.agents/skills`, …) and consumers only read from a `skills/` subdir, so
/// append `skills` unless `dir` already ends in it. This lets callers pass
/// the agent root or the full skills dir interchangeably.
fn with_skills_leaf(dir: PathBuf) -> PathBuf {
    if dir.file_name().and_then(|n| n.to_str()) == Some("skills") {
        dir
    } else {
        dir.join("skills")
    }
}

/// Copy each `<name>/` skill from the extracted staging tree into
/// `skills_dir`, replacing any existing copy so removed files don't linger.
fn install_staged_skills(staged: &Path, skills_dir: &Path) -> MetaResult<()> {
    if !staged.is_dir() {
        return Err(MetaError::SelfUpdateSkillsLayoutMissing {
            path: staged.display().to_string(),
        });
    }
    fs::create_dir_all(skills_dir)
        .map_err(|source| MetaError::self_update_skills_dir_create(skills_dir.display(), source))?;
    for entry in fs::read_dir(staged)
        .map_err(|source| MetaError::self_update_skills_staging_read(staged.display(), source))?
    {
        let entry = entry
            .map_err(|source| MetaError::self_update_skills_entry_read(staged.display(), source))?;
        if !entry
            .file_type()
            .map_err(|source| {
                MetaError::self_update_skills_entry_file_type(entry.path().display(), source)
            })?
            .is_dir()
        {
            continue;
        }
        let dest = skills_dir.join(entry.file_name());
        if dest.exists() {
            fs::remove_dir_all(&dest).map_err(|source| {
                MetaError::self_update_skill_remove_stale(dest.display(), source)
            })?;
        }
        copy_dir_all(&entry.path(), &dest)?;
    }
    Ok(())
}

/// Recursively copy `src` into `dst` (std has no built-in for this).
fn copy_dir_all(src: &Path, dst: &Path) -> MetaResult<()> {
    fs::create_dir_all(dst)
        .map_err(|source| MetaError::self_update_copy_dir_create(dst.display(), source))?;
    for entry in fs::read_dir(src)
        .map_err(|source| MetaError::self_update_copy_dir_read(src.display(), source))?
    {
        let entry = entry
            .map_err(|source| MetaError::self_update_copy_dir_entry_read(src.display(), source))?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if entry
            .file_type()
            .map_err(|source| {
                MetaError::self_update_copy_dir_entry_file_type(from.display(), source)
            })?
            .is_dir()
        {
            copy_dir_all(&from, &to)?;
        } else {
            fs::copy(&from, &to).map_err(|source| {
                MetaError::self_update_copy_file(from.display(), to.display(), source)
            })?;
        }
    }
    Ok(())
}

/// Drop a leading `v` from a release tag so versions render and compare
/// uniformly (`v0.1.0` -> `0.1.0`).
fn strip_v(version: &str) -> String {
    version.trim_start_matches('v').to_string()
}

/// True when `latest` is a strictly newer semver than `current`. A
/// `current` ahead of the newest release (local dev build) reports `false`
/// rather than offering a downgrade.
fn is_newer(current: &str, latest: &str) -> MetaResult<bool> {
    let cur = semver::Version::parse(current.trim_start_matches('v'))
        .map_err(|source| MetaError::self_update_current_version_parse(current, source))?;
    let lat = semver::Version::parse(latest.trim_start_matches('v'))
        .map_err(|source| MetaError::self_update_latest_version_parse(latest, source))?;
    Ok(lat > cur)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context, Result};
    use std::fs;
    use veloq_core::VeloqDiagnostic;

    #[test]
    fn newer_release_is_an_update() -> Result<()> {
        assert!(is_newer("0.1.0", "0.2.0")?);
        assert!(is_newer("0.1.0", "v0.1.1")?);
        assert!(is_newer("0.1.0", "1.0.0")?);
        Ok(())
    }

    #[test]
    fn same_or_older_is_not_an_update() -> Result<()> {
        assert!(!is_newer("0.1.0", "0.1.0")?);
        assert!(!is_newer("0.2.0", "0.1.9")?);
        // Local build ahead of the newest release must not downgrade.
        assert!(!is_newer("1.0.0", "0.9.9")?);
        Ok(())
    }

    #[test]
    fn prerelease_is_older_than_its_release() -> Result<()> {
        // Per semver, 1.0.0-alpha < 1.0.0.
        assert!(is_newer("1.0.0-alpha", "1.0.0")?);
        assert!(!is_newer("1.0.0", "1.0.0-alpha")?);
        Ok(())
    }

    #[test]
    fn invalid_version_is_an_error() {
        assert!(is_newer("not-a-version", "0.1.0").is_err());
    }

    #[test]
    fn strip_v_handles_both_forms() {
        assert_eq!(strip_v("v0.1.0"), "0.1.0");
        assert_eq!(strip_v("0.1.0"), "0.1.0");
    }

    #[test]
    fn install_staged_skills_replaces_and_strips_prefix() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        // Staged tree mirrors the extracted tarball: .claude/skills/<name>/...
        let staged = tmp.path().join("extract").join(".claude").join("skills");
        fs::create_dir_all(staged.join("nsys-profile-analysis"))?;
        fs::write(staged.join("nsys-profile-analysis/SKILL.md"), "new nsys")?;
        fs::create_dir_all(staged.join("ncu-profile-analysis/references"))?;
        fs::write(staged.join("ncu-profile-analysis/SKILL.md"), "new ncu")?;
        fs::write(staged.join("ncu-profile-analysis/references/x.md"), "ref")?;

        let skills_dir = tmp.path().join("skills");
        // A stale file from a prior install that must be removed on update.
        fs::create_dir_all(skills_dir.join("nsys-profile-analysis"))?;
        fs::write(skills_dir.join("nsys-profile-analysis/OLD.md"), "stale")?;

        install_staged_skills(&staged, &skills_dir)?;

        // Skills land at <skills_dir>/<name>/... (prefix stripped).
        assert_eq!(
            fs::read_to_string(skills_dir.join("nsys-profile-analysis/SKILL.md"))?,
            "new nsys"
        );
        assert_eq!(
            fs::read_to_string(skills_dir.join("ncu-profile-analysis/references/x.md"))?,
            "ref"
        );
        // The stale file is gone (the skill dir was replaced wholesale).
        assert!(!skills_dir.join("nsys-profile-analysis/OLD.md").exists());
        Ok(())
    }

    #[test]
    fn skills_dir_override_wins() -> Result<()> {
        // An explicit --skills-dir takes precedence over env/default without
        // consulting the environment.
        let dir = resolve_skills_dir(Some(Path::new("/tmp/proj/.agents/skills")))?;
        assert_eq!(dir, PathBuf::from("/tmp/proj/.agents/skills"));
        Ok(())
    }

    #[test]
    fn skills_leaf_is_appended_to_a_root() -> Result<()> {
        // Passing an agent root gets `skills/` appended...
        assert_eq!(
            resolve_skills_dir(Some(Path::new("/tmp/proj/.agents")))?,
            PathBuf::from("/tmp/proj/.agents/skills")
        );
        // ...but a path already ending in `skills` is left as-is (idempotent).
        assert_eq!(
            with_skills_leaf(PathBuf::from("/x/.claude/skills")),
            PathBuf::from("/x/.claude/skills")
        );
        assert_eq!(
            with_skills_leaf(PathBuf::from("/x/.claude")),
            PathBuf::from("/x/.claude/skills")
        );
        Ok(())
    }

    #[test]
    fn missing_skills_layout_is_an_error() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let staged = tmp.path().join("does-not-exist");
        let err = install_staged_skills(&staged, &tmp.path().join("skills"))
            .err()
            .context("missing staged skills should error")?;
        assert_eq!(
            err.code().as_str(),
            "meta.self-update.skills-layout-missing"
        );
        assert!(matches!(
            err,
            MetaError::SelfUpdateSkillsLayoutMissing { .. }
        ));
        Ok(())
    }
}
