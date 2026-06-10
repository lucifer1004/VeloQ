#!/usr/bin/env bash
# Sync the version surfaces before tagging:
#
#   - Cargo.toml  [workspace.package].version        (flows into every
#                                                     member crate via
#                                                     version.workspace = true)
#   - Cargo.toml  [workspace.dependencies] veloq-*   the path+version reqs
#                                                     `cargo publish` needs;
#                                                     kept equal to the
#                                                     workspace version
#   - .claude-plugin/plugin.json       .version
#   - .claude-plugin/marketplace.json  .plugins[0].version
#   - .codex-plugin/plugin.json        .version
#
# Also refreshes Cargo.lock so the bump commit is self-contained
# (otherwise the next `cargo build` would dirty the working copy).
#
# Usage:
#   scripts/bump-version.sh 0.1.0
#   scripts/bump-version.sh --dry-run 0.1.0
#
# The install.sh URL in README.md does NOT need updating — the CI
# build:installer job rewrites `VELOQ_VERSION=latest` to the actual
# tag at release time, so README stays version-less.

set -euo pipefail

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
  shift
fi

NEW_VERSION="${1:-}"
if [ -z "$NEW_VERSION" ]; then
  echo "usage: $0 [--dry-run] <new-version>" >&2
  echo "example: $0 0.1.0" >&2
  exit 2
fi

if ! printf '%s' "$NEW_VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.-]+)?$'; then
  echo "error: '$NEW_VERSION' doesn't look like a semver (X.Y.Z or X.Y.Z-pre)" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

for tool in jq cargo awk; do
  command -v "$tool" >/dev/null 2>&1 || { echo "error: '$tool' not on PATH" >&2; exit 1; }
done

CURRENT="$(awk '
  /^\[workspace\.package\]/ { in_block = 1; next }
  /^\[/ { in_block = 0 }
  in_block && /^version *= *"[^"]+"/ {
    gsub(/^version *= *"/, ""); gsub(/".*$/, "");
    print; exit
  }
' Cargo.toml)"
PLUGIN_CURRENT="$(jq -r '.version' .claude-plugin/plugin.json)"
MARKETPLACE_CURRENT="$(jq -r '.plugins[0].version' .claude-plugin/marketplace.json)"
CODEX_PLUGIN_CURRENT="$(jq -r '.version' .codex-plugin/plugin.json)"

if [ -z "$CURRENT" ]; then
  echo "error: could not locate [workspace.package].version in Cargo.toml" >&2
  exit 1
fi

if [ "$CURRENT" = "$NEW_VERSION" ] &&
   [ "$PLUGIN_CURRENT" = "$NEW_VERSION" ] &&
   [ "$MARKETPLACE_CURRENT" = "$NEW_VERSION" ] &&
   [ "$CODEX_PLUGIN_CURRENT" = "$NEW_VERSION" ]; then
  echo "Already at $NEW_VERSION — nothing to bump."
  exit 0
fi

echo "Bumping $CURRENT → $NEW_VERSION"

if [ "$DRY_RUN" = "1" ]; then
  echo "[dry-run] would edit:"
  echo "  Cargo.toml: [workspace.package].version → $NEW_VERSION"
  echo "  Cargo.toml: [workspace.dependencies] veloq-* version → $NEW_VERSION"
  echo "  .claude-plugin/plugin.json: .version → $NEW_VERSION"
  echo "  .claude-plugin/marketplace.json: .plugins[0].version → $NEW_VERSION"
  echo "  .codex-plugin/plugin.json: .version → $NEW_VERSION"
  echo "  Cargo.lock: cargo update --workspace"
  exit 0
fi

# 1. Cargo.toml [workspace.package].version — scoped to that block so
#    third-party dependency `version = "..."` lines stay untouched.
tmp="$(mktemp)"
awk -v new="$NEW_VERSION" '
  /^\[workspace\.package\]/ { in_block = 1; print; next }
  /^\[/                     { in_block = 0; print; next }
  in_block && /^version *= *"[^"]+"/ {
    sub(/"[^"]+"/, "\"" new "\""); print; next
  }
  { print }
' Cargo.toml > "$tmp"
mv "$tmp" Cargo.toml

# 2. Cargo.toml [workspace.dependencies] — bump the version req on the
#    in-workspace veloq-* crates so it tracks the package version.
#    Targets only path+version entries (third-party deps have no
#    `path =`, so they're left alone).
tmp="$(mktemp)"
awk -v new="$NEW_VERSION" '
  /^\[workspace\.dependencies\]/ { in_block = 1; print; next }
  /^\[/                          { in_block = 0; print; next }
  in_block && /path *=/ && /version *= *"[^"]+"/ {
    sub(/version *= *"[^"]+"/, "version = \"" new "\""); print; next
  }
  { print }
' Cargo.toml > "$tmp"
mv "$tmp" Cargo.toml

# 3 & 4. JSON manifests — jq preserves structure (matches existing
# 2-space indentation in both files).
update_json() {
  local file="$1" expr="$2" tmp
  tmp="$(mktemp)"
  jq --indent 2 --arg v "$NEW_VERSION" "$expr" "$file" > "$tmp"
  mv "$tmp" "$file"
}
update_json .claude-plugin/plugin.json      '.version = $v'
update_json .claude-plugin/marketplace.json '.plugins[0].version = $v'
update_json .codex-plugin/plugin.json       '.version = $v'

# 5. Cargo.lock — bump workspace-internal crate entries so the change
#    is committable in one go. `--workspace` scopes to in-workspace
#    members; it won't drift third-party deps.
cargo update --workspace --quiet

# Verify all surfaces now agree, to catch any awk/jq edge cases.
verify() {
  local label="$1" got="$2"
  if [ "$got" != "$NEW_VERSION" ]; then
    echo "error: $label is '$got', expected '$NEW_VERSION'" >&2
    exit 1
  fi
}
verify "Cargo.toml [workspace.package]"      "$(awk '/^\[workspace\.package\]/{b=1;next} /^\[/{b=0} b && /^version/ {gsub(/^version *= *"|".*$/, ""); print; exit}' Cargo.toml)"
verify "Cargo.toml [workspace.dependencies]" "$(awk '/^\[workspace\.dependencies\]/{b=1;next} /^\[/{b=0} b && /^veloq-core *=/ && match($0,/version *= *"[^"]+"/){s=substr($0,RSTART,RLENGTH);sub(/^version *= *"/,"",s);sub(/"$/,"",s);print s;exit}' Cargo.toml)"
verify ".claude-plugin/plugin.json"   "$(jq -r '.version' .claude-plugin/plugin.json)"
verify "marketplace.json"             "$(jq -r '.plugins[0].version' .claude-plugin/marketplace.json)"
verify ".codex-plugin/plugin.json"    "$(jq -r '.version' .codex-plugin/plugin.json)"

echo "Done."
echo
echo "Next steps:"
echo "  1. Review the diff:    git diff Cargo.toml Cargo.lock .claude-plugin/ .codex-plugin/"
echo "  2. Preview release:    govctl release ${NEW_VERSION} --date $(date +%F) --dry-run"
echo "  3. Cut gov release:    govctl release ${NEW_VERSION} --date $(date +%F)"
echo "  4. Commit:             jj describe -m 'release: bump to v${NEW_VERSION}'"
echo "  5. Push to main:       jj git push --bookmark main"
echo "  6. Tag once main has the bump commit:"
echo "                         git tag v${NEW_VERSION} && git push origin v${NEW_VERSION}"
echo "  7. Publish the crates to crates.io (dependency-ordered):"
echo "                         just publish-dry   # rehearse, then:"
echo "                         just publish"
