#!/usr/bin/env bash
# Validate the repo-local VeloQ agent plugin package layout.
#
# Canonical package source:
#   - plugins/veloq/
#
# Compatibility entrypoints:
#   - .agents/skills
#   - .claude/skills
#   - .codex-plugin/plugin.json
#   - .claude-plugin/plugin.json
#
# Usage:
#   scripts/check-agent-plugin-package.sh

set -euo pipefail

if [ "${1:-}" != "" ]; then
  echo "usage: $0" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PACKAGE_ROOT="plugins/veloq"
PACKAGE_SKILLS="$PACKAGE_ROOT/skills"
PACKAGE_CODEX_PLUGIN="$PACKAGE_ROOT/.codex-plugin/plugin.json"
PACKAGE_CLAUDE_PLUGIN="$PACKAGE_ROOT/.claude-plugin/plugin.json"

require_file() {
  if [ ! -f "$1" ]; then
    echo "error: missing file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [ ! -d "$1" ]; then
    echo "error: missing directory: $1" >&2
    exit 1
  fi
}

require_symlink_target() {
  local link="$1"
  local target="$2"
  if [ ! -L "$link" ]; then
    echo "error: expected symlink: $link" >&2
    exit 1
  fi
  local actual
  actual="$(readlink "$link")"
  if [ "$actual" != "$target" ]; then
    echo "error: $link points to $actual; expected $target" >&2
    exit 1
  fi
}

require_file_pattern() {
  local file="$1"
  local pattern="$2"
  local label="$3"
  if ! grep -Eq "$pattern" "$file"; then
    echo "error: $file does not contain expected $label" >&2
    exit 1
  fi
}

require_dir "$PACKAGE_ROOT"
require_dir "$PACKAGE_SKILLS"
require_file "$PACKAGE_CODEX_PLUGIN"
require_file "$PACKAGE_CLAUDE_PLUGIN"

require_file "$PACKAGE_SKILLS/nsys-profile-analysis/SKILL.md"
require_file "$PACKAGE_SKILLS/ncu-profile-analysis/SKILL.md"
require_file "$PACKAGE_SKILLS/pytorch-profile-analysis/SKILL.md"

if [ -n "$(find "$PACKAGE_ROOT" -type l -print -quit)" ]; then
  echo "error: $PACKAGE_ROOT must not contain symlinks" >&2
  exit 1
fi

require_symlink_target ".agents/skills" "../plugins/veloq/skills"
require_symlink_target ".claude/skills" "../.agents/skills"
require_symlink_target ".codex-plugin/plugin.json" "../plugins/veloq/.codex-plugin/plugin.json"
require_symlink_target ".claude-plugin/plugin.json" "../plugins/veloq/.claude-plugin/plugin.json"

require_file_pattern \
  ".agents/plugins/marketplace.json" \
  '"path"[[:space:]]*:[[:space:]]*"./plugins/veloq"' \
  "Codex marketplace path ./plugins/veloq"
require_file_pattern \
  ".claude-plugin/marketplace.json" \
  '"source"[[:space:]]*:[[:space:]]*"./plugins/veloq"' \
  "Claude marketplace source ./plugins/veloq"

echo "Agent plugin package layout is valid."
