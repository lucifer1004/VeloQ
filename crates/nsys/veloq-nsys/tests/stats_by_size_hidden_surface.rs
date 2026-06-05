//! Feature-specific hidden-surface checks for `stats --by size`.
//! The cross-target contract (registry membership, env-gated listing,
//! schema resolves/errors, drift guard) lives in
//! `hidden_target_contract.rs` and runs once for every
//! `HIDDEN_TARGETS` entry.

use anyhow::Result;
use veloq_nsys::cli::Cmd;

#[test]
fn by_flag_is_clap_hidden_on_stats() -> Result<()> {
    // `--by size` is the only valid non-default value, and it gates
    // on VELOQ_UNSTABLE — so the flag itself stays out of `--help`.
    let parent = clap::Command::new("veloq-test")
        .subcommand_required(false)
        .arg_required_else_help(false);
    let parent = <Cmd as clap::Subcommand>::augment_subcommands(parent);
    let stats = parent
        .find_subcommand("stats")
        .ok_or_else(|| anyhow::anyhow!("stats subcommand must exist"))?;
    let by_arg = stats
        .get_arguments()
        .find(|a| a.get_id() == "by")
        .ok_or_else(|| anyhow::anyhow!("--by arg must exist on stats"))?;
    assert!(
        by_arg.is_hide_set(),
        "--by must be hidden from stats --help"
    );
    Ok(())
}

#[test]
fn cmd_name_reflects_by_size_mode() -> Result<()> {
    // Cmd::name() must return "stats-by-size" when by=Size so the
    // error envelope agrees with the success envelope for the same
    // invocation. Otherwise success says nsys.stats-by-size and
    // failure says nsys.stats.
    let parent = clap::Command::new("veloq-test")
        .subcommand_required(true)
        .arg_required_else_help(true);
    let parent = <Cmd as clap::Subcommand>::augment_subcommands(parent);
    let matches = parent.clone().try_get_matches_from([
        "veloq-test",
        "stats",
        "/tmp/x.sqlite",
        "--by",
        "size",
    ])?;
    let cmd = <Cmd as clap::FromArgMatches>::from_arg_matches(&matches)?;
    assert_eq!(cmd.name(), "stats-by-size");
    Ok(())
}
