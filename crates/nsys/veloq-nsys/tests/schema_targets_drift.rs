//! Drift regression.
//!
//! Three sites must agree on the set of schema targets:
//!   * `schema::schema_value_for`  — resolves every target
//!   * `cli::Cmd::Schema` doc-comment — lists targets in `--help`
//!   * `help::long_about_schema` — lists targets in `--help` blurb
//!
//! All three derive from `schema_targets::TARGETS`. This test asserts
//! every entry in `TARGETS`:
//!   * resolves through `schema::schema_value_for`,
//!   * appears in the long_about returned by `help::long_about_schema`, and
//!   * appears in the per-arg help that `help::inject_long_about`
//!     patches onto the schema subcommand's `target` arg.
//!
//! Adding a new public target only requires one new row in
//! `schema_targets::TARGETS`; the three call sites pick it up
//! automatically and this test holds them honest.

use anyhow::Result;
use clap::{Command, Subcommand};
use veloq_nsys::cli::Cmd;
use veloq_nsys::help::{inject_long_about, long_about_schema};
use veloq_nsys::schema::schema_value_for;
use veloq_nsys::schema_targets::TARGETS;

fn built_schema_subcommand() -> Result<Command> {
    let parent = Command::new("veloq-test")
        .subcommand_required(false)
        .arg_required_else_help(false);
    let parent = Cmd::augment_subcommands(parent);
    let parent = inject_long_about(parent);
    parent
        .find_subcommand("schema")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("schema subcommand not found after inject_long_about"))
}

#[test]
fn every_registry_target_resolves_to_schema_json() -> Result<()> {
    for entry in TARGETS {
        let value = schema_value_for(entry.name)?;
        assert!(
            value.is_object(),
            "schema_value_for({}) returned non-object",
            entry.name
        );
    }
    Ok(())
}

#[test]
fn long_about_schema_lists_every_registry_target() {
    let blurb = long_about_schema();
    for entry in TARGETS {
        assert!(
            blurb.contains(entry.name),
            "long_about_schema missing target `{}` — drift between TARGETS and help.rs",
            entry.name
        );
    }
}

#[test]
fn schema_target_arg_help_lists_every_registry_target() -> Result<()> {
    let sub = built_schema_subcommand()?;
    let target_arg = sub
        .get_arguments()
        .find(|a| a.get_id() == "target")
        .ok_or_else(|| anyhow::anyhow!("target arg not found on schema subcommand"))?;
    let help = target_arg
        .get_help()
        .map(|s| s.to_string())
        .unwrap_or_default();
    for entry in TARGETS {
        assert!(
            help.contains(entry.name),
            "schema target arg help missing `{}` — drift between TARGETS and cli.rs help",
            entry.name
        );
    }
    Ok(())
}
