#!/usr/bin/env bash
# veloq installer — run with: curl -fsSL <url>/install.sh | bash
#
# Installs:
#   1. The VeloQ binary into ~/.local/bin (or --bin-dir).
#   2. Agent Skills for profile analysis (nsys-profile-analysis,
#      ncu-profile-analysis, pytorch-profile-analysis) into ~/.agents/skills/.
#
# Re-run safe: existing files are overwritten. Use --no-binary or
# --no-skills to update one half without touching the other. The
# skills require a VeloQ CLI on PATH for evidence extraction.
#
# Options:
#   --bin-dir <path>    Install binary to this directory (default: ~/.local/bin)
#   --skills-dir <path> Install skills under this dir (default: ~/.agents); `skills/`
#                       is appended if absent, so pass an agent root (.agents, ~/.agents)
#                       or a full skills dir
#   --no-binary         Skip the binary download (Agent Skills only; manage VeloQ separately)
#   --no-skills         Skip the Agent Skills install (binary only)
#   --help              Show this help and exit
#
# Environment:
#   VELOQ_VERSION      Pin a specific release tag (default: latest)
#   VELOQ_INSTALL_DIR  Same as --bin-dir
#   VELOQ_SKILLS_DIR   Same as --skills-dir
#   VELOQ_BASE_URL     Override the release-asset download base URL

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults / config
# ---------------------------------------------------------------------------

VERSION="${VELOQ_VERSION:-latest}"  # Replaced by CI at tag-build time to pin
REPO="${VELOQ_REPO:-lucifer1004/veloq}"
GITHUB_HOST="${VELOQ_GITHUB_HOST:-https://github.com}"
API_URL="${VELOQ_GITHUB_API:-https://api.github.com}"
# GitHub release assets live at <host>/<repo>/releases/download/<tag>/<asset>.
RELEASE_BASE="${VELOQ_BASE_URL:-${GITHUB_HOST}/${REPO}/releases/download}"
BIN_DIR="${VELOQ_INSTALL_DIR:-$HOME/.local/bin}"
SKILLS_DIR="${VELOQ_SKILLS_DIR:-$HOME/.agents}"
INSTALL_BINARY=true
INSTALL_SKILLS=true

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

if [ -t 1 ]; then
    RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m' NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' NC=''
fi

info()  { echo -e "${GREEN}[veloq]${NC} $*"; }
warn()  { echo -e "${YELLOW}[veloq] WARNING:${NC} $*" >&2; }
die()   { echo -e "${RED}[veloq] ERROR:${NC} $*" >&2; exit 1; }

show_help() {
    # Strip the leading `# ` (or bare `#`) from the top comment
    # block; stop at the first non-comment line. Simpler than a sed
    # range because the block has no terminating blank line — every
    # line either starts with `#` or is the first `set …` directive.
    awk '
        NR == 1 { next }                       # shebang
        /^#/ { sub(/^# ?/, ""); print; next }  # comment line
        { exit }                                # first non-comment ends help
    ' "$0"
    exit 0
}

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

while [ $# -gt 0 ]; do
    case "$1" in
        --bin-dir)    BIN_DIR="$2"; shift 2 ;;
        --skills-dir) SKILLS_DIR="$2"; shift 2 ;;
        --no-binary)  INSTALL_BINARY=false; shift ;;
        --no-skills)  INSTALL_SKILLS=false; shift ;;
        --help|-h)    show_help ;;
        *) die "unknown option: $1" ;;
    esac
done

# Skills live under <dir>/skills/ by convention; append it unless the path
# already ends in `skills`, so --skills-dir / VELOQ_SKILLS_DIR may be an
# agent root (~/.agents, .agents, ~/.claude) or a full skills dir. Mirrors
# `veloq self-update`'s resolution.
if [ "$(basename "$SKILLS_DIR")" != "skills" ]; then
    SKILLS_DIR="$SKILLS_DIR/skills"
fi

# ---------------------------------------------------------------------------
# Resolve VERSION (only when something needs downloading)
# ---------------------------------------------------------------------------

resolve_version() {
    if [ "$VERSION" != "latest" ]; then
        return
    fi
    info "fetching latest release version..."
    VERSION=$(curl -fsSL "${API_URL}/repos/${REPO}/releases/latest" \
        | grep -o '"tag_name": *"[^"]*"' \
        | head -1 \
        | cut -d'"' -f4)
    if [ -z "$VERSION" ]; then
        die "could not resolve latest release. Either no releases exist yet or the project is unreachable."
    fi
    info "resolved latest = $VERSION"
}

# ---------------------------------------------------------------------------
# OS / arch detection (binary only)
# ---------------------------------------------------------------------------

detect_asset() {
    local os arch
    os=$(uname -s)
    arch=$(uname -m)
    case "$os" in
        Linux)
            case "$arch" in
                x86_64)  echo "veloq-x86_64-linux"  ;;
                aarch64) echo "veloq-aarch64-linux" ;;
                *) die "unsupported Linux architecture: $arch" ;;
            esac
            ;;
        Darwin)
            case "$arch" in
                x86_64) echo "veloq-x86_64-macos"  ;;
                arm64)  echo "veloq-aarch64-macos" ;;
                *) die "unsupported macOS architecture: $arch" ;;
            esac
            ;;
        *)
            die "unsupported OS: $os (Windows users: grab veloq-x86_64-windows.exe from the Release page)"
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Download helper (curl / wget either)
# ---------------------------------------------------------------------------

fetch() {
    local url="$1" dest="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$url" -o "$dest"
    elif command -v wget >/dev/null 2>&1; then
        wget -q "$url" -O "$dest"
    else
        die "neither curl nor wget found"
    fi
}

# Verify <dir>/<asset> against the release's published <asset>.sha256.
# The .sha256 lists the ORIGINAL asset name, so we verify inside <dir>
# where the file still has that name (before it's renamed to `veloq`).
# Aborts on mismatch; warns and skips when no checksum tool exists or the
# release predates checksums.
verify_sha256() {
    local dir="$1" asset="$2" base_url="$3" checker
    if command -v sha256sum >/dev/null 2>&1; then
        checker="sha256sum -c"
    elif command -v shasum >/dev/null 2>&1; then
        checker="shasum -a 256 -c"
    else
        warn "no sha256 tool (sha256sum/shasum) found; skipping checksum verification"
        return 0
    fi
    if ! fetch "${base_url}.sha256" "${dir}/${asset}.sha256" 2>/dev/null; then
        warn "checksum ${asset}.sha256 not published; skipping verification"
        return 0
    fi
    info "verifying checksum"
    ( cd "$dir" && $checker "${asset}.sha256" >/dev/null 2>&1 ) \
        || die "checksum verification FAILED for $asset — refusing to install a tampered/corrupt binary"
}

# ---------------------------------------------------------------------------
# Binary install
# ---------------------------------------------------------------------------

install_binary() {
    local asset url dest tmp
    asset=$(detect_asset)
    url="${RELEASE_BASE}/${VERSION}/${asset}"
    dest="${BIN_DIR}/veloq"

    mkdir -p "$BIN_DIR"
    # Download to a temp dir under the original asset name so the published
    # checksum (which lists that name) verifies before we install + rename.
    tmp="$(mktemp -d "${TMPDIR:-/tmp}/veloq-install.XXXXXX")"
    trap 'rm -rf "$tmp"' EXIT
    info "downloading $asset → $dest"
    fetch "$url" "${tmp}/${asset}"
    verify_sha256 "$tmp" "$asset" "$url"
    chmod +x "${tmp}/${asset}"
    mv -f "${tmp}/${asset}" "$dest"
    rm -rf "$tmp"
    trap - EXIT
    info "installed: $dest"
}

# ---------------------------------------------------------------------------
# Skills install
# ---------------------------------------------------------------------------
# At release time the CI's skills-tarball job packs .agents/skills/ and
# a .claude/skills/ compatibility alias, then attaches the archive to the
# GitHub Release as veloq-skills.tar.gz. We fetch + extract into a staging
# directory, then replace each installed skill directory wholesale so removed
# files do not linger across upgrades.

install_skills() {
    local url archive extract_dir staged skill_dir skill_name dest installed
    url="${RELEASE_BASE}/${VERSION}/veloq-skills.tar.gz"
    extract_dir=
    archive=$(mktemp -t veloq-skills.XXXXXX.tar.gz)
    trap 'rm -f "$archive"; [ -z "$extract_dir" ] || rm -rf "$extract_dir"' EXIT
    extract_dir="$(mktemp -d "${TMPDIR:-/tmp}/veloq-skills.XXXXXX")"

    mkdir -p "$SKILLS_DIR"
    info "downloading skills → $SKILLS_DIR"
    fetch "$url" "$archive"
    tar -xz -f "$archive" -C "$extract_dir"

    staged="${extract_dir}/.agents/skills"
    if [ ! -d "$staged" ]; then
        staged="${extract_dir}/.claude/skills"
    fi
    [ -d "$staged" ] || die "skills archive is missing .agents/skills/ or .claude/skills/"

    installed=false
    for skill_dir in "$staged"/*; do
        [ -d "$skill_dir" ] || continue
        skill_name="$(basename "$skill_dir")"
        dest="${SKILLS_DIR}/${skill_name}"
        rm -rf "$dest"
        cp -R "$skill_dir" "$dest"
        installed=true
    done
    $installed || die "skills archive did not contain any skill directories"

    rm -f "$archive"
    rm -rf "$extract_dir"
    trap - EXIT
    info "installed skills under $SKILLS_DIR/{nsys,ncu,pytorch}-profile-analysis"
}

# ---------------------------------------------------------------------------
# PATH sanity
# ---------------------------------------------------------------------------

check_path() {
    case ":$PATH:" in
        *":${BIN_DIR}:"*) return ;;
    esac
    warn "${BIN_DIR} is not in PATH"
    info "add this to your shell profile:"
    info "  export PATH=\"${BIN_DIR}:\$PATH\""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    info "installing veloq..."

    if $INSTALL_BINARY || $INSTALL_SKILLS; then
        resolve_version
    fi

    if $INSTALL_BINARY; then
        install_binary
    fi

    if $INSTALL_SKILLS; then
        install_skills
    fi

    if $INSTALL_BINARY; then
        check_path
    fi

    info "done."
    if $INSTALL_BINARY; then
        info "try: veloq --help"
    fi
    if $INSTALL_SKILLS; then
        info "Agent Skills: installed under $SKILLS_DIR/{nsys,ncu,pytorch}-profile-analysis"
    fi
}

main
