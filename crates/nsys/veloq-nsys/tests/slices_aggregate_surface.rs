//! Public-surface checks for `slices --aggregate`, and that there is
//! no separate `slices-summary` command.

use anyhow::{Result, anyhow, bail};
use clap::Subcommand;
use veloq_nsys::cli::Cmd;
use veloq_nsys::help::inject_long_about;
use veloq_nsys::schema::schema_value_for;
use veloq_nsys::schema_targets::{HIDDEN_TARGETS, TARGETS};

#[test]
fn slices_has_aggregate_flags_and_slices_summary_is_absent() -> Result<()> {
    let parent = clap::Command::new("veloq-test")
        .subcommand_required(false)
        .arg_required_else_help(false);
    let parent = Cmd::augment_subcommands(parent);
    let parent = inject_long_about(parent);

    let slices = parent
        .find_subcommand("slices")
        .ok_or_else(|| anyhow!("slices subcommand must exist"))?;
    assert!(
        slices.get_arguments().any(|a| a.get_id() == "aggregate"),
        "slices must expose --aggregate"
    );
    assert!(
        slices.get_arguments().all(|a| a.get_id() != "rollup"),
        "slices must not expose the abandoned --rollup spelling"
    );
    assert!(
        slices.get_arguments().any(|a| a.get_id() == "group_by"),
        "slices must expose --group-by"
    );
    assert!(
        parent.find_subcommand("slices-summary").is_none(),
        "slices-summary must not be a CLI subcommand"
    );

    let mut named = parent.clone();
    let parse =
        named.try_get_matches_from_mut(["veloq-test", "slices-summary", "/tmp/trace.nsys-rep"]);
    let err = match parse {
        Ok(_) => bail!("slices-summary should not parse"),
        Err(e) => e,
    };
    assert_eq!(err.kind(), clap::error::ErrorKind::InvalidSubcommand);
    Ok(())
}

#[test]
fn slices_schema_absorbs_aggregate_and_slices_summary_target_is_absent() -> Result<()> {
    assert!(
        TARGETS.iter().any(|entry| entry.name == "slices"),
        "slices must live in public TARGETS"
    );
    assert!(
        TARGETS.iter().all(|entry| entry.name != "slices-summary"),
        "slices-summary must not live in public TARGETS"
    );
    assert!(
        HIDDEN_TARGETS
            .iter()
            .all(|entry| entry.name != "slices-summary"),
        "slices-summary must not live in HIDDEN_TARGETS"
    );

    let slices_schema = schema_value_for("slices")?;
    assert!(slices_schema.is_object());
    let Err(err) = schema_value_for("slices-summary") else {
        bail!("slices-summary schema target should be removed");
    };
    assert!(err.to_string().contains("unknown schema target"));
    Ok(())
}
