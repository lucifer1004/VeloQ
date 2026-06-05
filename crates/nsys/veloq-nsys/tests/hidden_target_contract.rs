//! Table-driven contract for every `HIDDEN_TARGETS` entry. Each
//! hidden schema target must satisfy:
//!
//! 1. lives only in `HIDDEN_TARGETS` (never in the public `TARGETS`),
//! 2. is absent from the rendered target list in the default env,
//! 3. is present in the rendered target list under `VELOQ_UNSTABLE=1`,
//! 4. `schema_value_for(name)` errors in the default env,
//! 5. `schema_value_for(name)` resolves under `VELOQ_UNSTABLE=1`.
//!
//! Feature-specific checks (e.g. the `--by` flag is `clap`-hidden and
//! `Cmd::name()` reflects the by-mode) stay in their own per-feature
//! test files.

use anyhow::{Result, bail};
use std::sync::{Mutex, MutexGuard};
use veloq_nsys::schema::schema_value_for;
use veloq_nsys::schema_targets::{HIDDEN_TARGETS, TARGETS, render_target_list};

// Env-mutating tests serialise via a shared mutex so the parallel
// runner never sees a half-set state from a sibling test.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn lock_env() -> MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn unset_unstable() {
    // SAFETY: env mutation is serialised via ENV_LOCK; other test
    // binaries run in their own process and don't share env.
    unsafe {
        std::env::remove_var("VELOQ_UNSTABLE");
    }
}

fn set_unstable() {
    unsafe {
        std::env::set_var("VELOQ_UNSTABLE", "1");
    }
}

#[test]
fn every_hidden_target_satisfies_contract() -> Result<()> {
    for entry in HIDDEN_TARGETS {
        let name = entry.name;
        // (1) Disjoint from public TARGETS.
        assert!(
            TARGETS.iter().all(|p| p.name != name),
            "hidden target `{name}` must not also live in public TARGETS"
        );

        // (2) + (3): rendered listing tracks env. Both checks share
        // one lock guard so a parallel sibling can't flip the env
        // between them. Split on the literal delimiter and compare
        // exact names — substring matching would false-pass on a
        // future target whose name is a prefix/suffix of this one.
        let _g = lock_env();
        unset_unstable();
        let default_listing = render_target_list();
        let default_names: Vec<&str> = default_listing.split(", ").collect();
        assert!(
            !default_names.contains(&name),
            "default env must not list `{name}`; got `{default_names:?}`"
        );

        set_unstable();
        let unstable_listing = render_target_list();
        let unstable_names: Vec<&str> = unstable_listing.split(", ").collect();
        assert!(
            unstable_names.contains(&name),
            "VELOQ_UNSTABLE=1 must list `{name}`; got `{unstable_names:?}`"
        );

        // (4): resolver errors without env.
        unset_unstable();
        let Err(err) = schema_value_for(name) else {
            bail!("schema target `{name}` must error in default env");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("unknown schema target") && msg.contains(name),
            "default-env error for `{name}` should follow the `unknown schema target` \
             wording; got `{msg}`"
        );

        // (5): resolver returns a JSON object under env.
        set_unstable();
        let v = schema_value_for(name)?;
        assert!(v.is_object(), "schema for `{name}` should be a JSON object");

        unset_unstable();
    }
    Ok(())
}

// Public-target resolution is covered by `schema_targets_drift.rs`;
// public/hidden disjointness is checked inside the per-target loop
// above. No standalone drift guard here.
