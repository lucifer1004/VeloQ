# Local build / test recipes for veloq.
#
# Mirrors the GitHub Actions CI stages so a contributor can validate
# locally before pushing. `just ci-checks` runs the same gate the
# CI's test job runs.

set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]
# Recipes should not source interactive shell startup files under nounset.
set shell := ["env", "BASH_ENV=", "bash", "-euo", "pipefail", "-c"]

# Default: list recipes.
default:
    @just --list

# ---------------------------------------------------------------------------- #
#                                    BUILD                                     #
# ---------------------------------------------------------------------------- #

# Release build for the host platform.
[unix]
[group("build")]
release:
    #!/usr/bin/env bash
    set -euo pipefail
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64)  cargo build --release -p veloq --target x86_64-unknown-linux-musl ;;
        Linux-aarch64) cargo build --release -p veloq --target aarch64-unknown-linux-musl ;;
        Darwin-x86_64) cargo build --release -p veloq --target x86_64-apple-darwin ;;
        Darwin-arm64)  cargo build --release -p veloq --target aarch64-apple-darwin ;;
        *) echo "Unsupported platform: $(uname -s)-$(uname -m)"; exit 1 ;;
    esac

# Cross-compile both Linux musl-static binaries (needs cargo-zigbuild + zig).
[group("build")]
release-linux:
    cargo zigbuild --release -p veloq --target x86_64-unknown-linux-musl
    cargo zigbuild --release -p veloq --target aarch64-unknown-linux-musl

# Cross-compile all macOS release binaries via cargo-zigbuild.
[group("build")]
release-macos:
    cargo zigbuild --release -p veloq --target x86_64-apple-darwin
    cargo zigbuild --release -p veloq --target aarch64-apple-darwin

# Verify a Linux release binary is statically linked.
[unix]
[group("build")]
check-static binary:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! [ -f "{{ binary }}" ]; then
        echo -e '{{ RED }}File not found: {{ binary }}{{ NORMAL }}'; exit 1
    fi
    if file "{{ binary }}" | grep -q "statically linked"; then
        echo -e '{{ GREEN }}OK: {{ binary }} is statically linked{{ NORMAL }}'
    elif ldd "{{ binary }}" 2>&1 | grep -q "not a dynamic executable"; then
        echo -e '{{ GREEN }}OK: {{ binary }} is statically linked{{ NORMAL }}'
    else
        echo -e '{{ RED }}FAIL: {{ binary }} is dynamically linked — DO NOT ship{{ NORMAL }}'
        ldd "{{ binary }}" 2>/dev/null || true
        exit 1
    fi

# Symbolized-but-optimized build for self-profiling (samply / perf /
# Instruments): release-level opt + full debuginfo via [profile.prof].
# Lands at target/prof/veloq. Example: samply record target/prof/veloq stats t.nsys-rep
[group("build")]
prof:
    cargo build --profile prof -p veloq

# ---------------------------------------------------------------------------- #
#                                    CHECKS                                    #
# ---------------------------------------------------------------------------- #

# Run the full pre-commit gate (fmt + clippy + test).
[unix]
[group("checks")]
pre-commit:
    @if command -v prek > /dev/null 2>&1; then prek run --all-files; else pre-commit run --all-files; fi

[windows]
[group("checks")]
pre-commit:
    if (Get-Command prek -ErrorAction SilentlyContinue) { prek run --all-files } else { pre-commit run --all-files }

# Same gate as `pre-commit` but without the pre-commit framework.
[group("checks")]
ci-checks:
    @echo -e '{{ CYAN }}→ cargo fmt --check{{ NORMAL }}'
    cargo fmt --all -- --check
    @echo -e '{{ CYAN }}→ cargo clippy{{ NORMAL }}'
    cargo clippy --workspace --all-targets --profile ci -- -D warnings
    @echo -e '{{ CYAN }}→ cargo test{{ NORMAL }}'
    cargo test --workspace --profile ci
    @echo -e '{{ GREEN }}✓ all CI checks passed{{ NORMAL }}'
alias cc := ci-checks

# Validate the GitHub Actions workflows with actionlint. Falls back to
# a Python YAML parse check when actionlint isn't installed.
[group("checks")]
[doc("Lint .github/workflows via actionlint (or YAML-parse fallback).")]
ci-lint:
    @if command -v actionlint > /dev/null 2>&1; then \
        actionlint; \
    elif command -v python3 > /dev/null 2>&1; then \
        python3 -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yml')]; print('OK (YAML parse only — install actionlint for full lint)')"; \
    else \
        echo "actionlint and python3 both unavailable; cannot lint workflows locally" >&2; \
        exit 1; \
    fi

# Format the workspace; squash diff into the in-flight commit.
[group("checks")]
fmt:
    cargo fmt --all

# ---------------------------------------------------------------------------- #
#                                   RELEASE                                    #
# ---------------------------------------------------------------------------- #

# Register the gov release and bump Cargo.toml + the two Claude
# plugin manifests + Cargo.lock. Pre-flight for tagging a release.
# Usage:
#   just bump-version 0.1.0
#   just bump-version --dry-run 0.1.0
[group("release")]
[doc("Register the gov release and sync Cargo.toml plus plugin manifests.")]
bump-version +args:
    scripts/bump-version.sh {{ args }}

# Publish every crate to crates.io in dependency order. cargo (>= 1.90)
# topologically orders the workspace and waits for the index between
# crates, so the whole family goes up in one command. Run AFTER
# `just bump-version`, committing, and tagging — and rehearse with
# `publish-dry` first (it packages + verify-builds each crate without
# uploading, which is what catches a too-narrow `include` whitelist).
[group("release")]
[doc("Dry-run the crates.io publish: package + verify-build every crate, no upload.")]
publish-dry:
    cargo publish --workspace --dry-run

[group("release")]
[doc("Publish every crate to crates.io in dependency order (real upload).")]
publish:
    cargo publish --workspace
