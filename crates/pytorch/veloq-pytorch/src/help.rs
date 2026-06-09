//! Help-text projector for PyTorch verbs.
//!
//! Today only the `schema` subcommand needs runtime projection: its
//! valid-target list comes from [`crate::schema_targets::TARGETS`] so
//! CLI help cannot drift from the schema resolver.

pub fn inject_long_about(cmd: clap::Command) -> clap::Command {
    cmd.mut_subcommand("schema", |sub| {
        sub.long_about(long_about_schema())
            .mut_arg("target", |arg| arg.help(schema_target_arg_help()))
    })
}

pub(crate) fn long_about_schema() -> String {
    format!(
        "Emit the strict JSON Schema for one PyTorch response payload. \
         Meta endpoint -- reads no trace.\n\nValid targets: {}.\n\n\
         Response envelope:\n  {{ schema: \"v1\", source: {{ kind: \"pytorch\", version: \"v0\" }}, \
         command: \"pytorch.schema\", data: {{ target: <string>, schema: <JSON Schema document> }} }}",
        crate::schema_targets::render_target_list()
    )
}

pub(crate) fn schema_target_arg_help() -> String {
    format!(
        "Subcommand whose response schema to print. One of: {}",
        crate::schema_targets::render_target_list()
    )
}
