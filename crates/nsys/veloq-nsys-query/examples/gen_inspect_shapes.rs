//! Regenerate `inspect-shapes.md` from the EventDetails projection.
//!
//!   cargo run --release -p veloq-nsys-query --example gen_inspect_shapes -- write
//!   cargo run --release -p veloq-nsys-query --example gen_inspect_shapes -- check   (default)
//!
//! `check` is what the freshness test runs under the hood (via
//! `veloq_nsys_query::docgen::inspect_shapes_body`); the example exists so
//! humans don't have to remember the path.

use std::path::PathBuf;
use veloq_nsys_query::docgen::inspect_shapes_body;

fn target_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join(".claude/skills/nsys-profile-analysis/references/inspect-shapes.md")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let body = inspect_shapes_body();
    let path = target_path();
    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "check".to_string());
    match mode.as_str() {
        "write" => {
            std::fs::write(&path, &body)?;
            println!("wrote {}", path.display());
        }
        "check" => {
            let on_disk = std::fs::read_to_string(&path).unwrap_or_default();
            if on_disk == body {
                println!("inspect-shapes.md is current ({} bytes)", body.len());
            } else {
                eprintln!(
                    "inspect-shapes.md is stale\n\
                     run: cargo run --release -p veloq-nsys-query --example gen_inspect_shapes -- write\n\
                     path: {}",
                    path.display()
                );
                std::process::exit(1);
            }
        }
        other => {
            eprintln!("unknown mode `{other}`; expected `write` or `check`");
            std::process::exit(2);
        }
    }
    Ok(())
}
