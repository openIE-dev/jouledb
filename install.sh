#!/bin/bash
# JouleDB — The World's First Energy-Aware Database
# One-command installer: curl -fsSL https://jouledb.org/install.sh | sh
#
# Environment variables:
#   JOULEDB_VERSION=<version>      Install a specific version (default: latest)
#   JOULEDB_INSTALL_DIR=<path>     Binary install directory (default: /usr/local/bin)
#   JOULEDB_NO_DAEMON=1            Don't start the daemon after install
#   JOULEDB_UNINSTALL=1            Uninstall JouleDB
#
# Usage:
#   Install:      curl -fsSL https://jouledb.org/install.sh | sh
#   Uninstall:    curl -fsSL https://jouledb.org/install.sh | JOULEDB_UNINSTALL=1 sh

set -euo pipefail

# --- Configuration ---
JOULEDB_INSTALL_DIR="${JOULEDB_INSTALL_DIR:-/usr/local/bin}"
JOULEDB_DATA_DIR="$HOME/.jouledb"
JOULEDB_REPO="openIE-dev/jouledb"
JOULEDB_BINARIES="jouledb jouledb-cli"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

# --- Helpers ---
info()  { printf "${BLUE}  info${RESET}  %s\n" "$1"; }
ok()    { printf "${GREEN}    ok${RESET}  %s\n" "$1"; }
warn()  { printf "${YELLOW}  warn${RESET}  %s\n" "$1"; }
err()   { printf "${RED} error${RESET}  %s\n" "$1" >&2; }
fatal() { err "$1"; exit 1; }

banner() {
    printf "\n"
    printf "${CYAN}${BOLD}"
    printf "  ┌─────────────────────────────────────┐\n"
    printf "  │            JouleDB                   │\n"
    printf "  │   Energy-Aware Database Engine       │\n"
    printf "  └─────────────────────────────────────┘${RESET}\n"
    printf "\n"
}

# --- Uninstall ---
uninstall() {
    banner
    info "Uninstalling JouleDB..."

    # Stop daemon if running
    if [ -f "$JOULEDB_DATA_DIR/daemon.pid" ]; then
        local pid
        pid=$(cat "$JOULEDB_DATA_DIR/daemon.pid" 2>/dev/null || true)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            info "Stopping daemon (PID $pid)..."
            kill "$pid" 2>/dev/null || true
            sleep 1
        fi
    fi

    # Uninstall OS service
    if [ "$(uname -s)" = "Darwin" ]; then
        local plist="$HOME/Library/LaunchAgents/com.jouledb.daemon.plist"
        if [ -f "$plist" ]; then
            launchctl unload -w "$plist" 2>/dev/null || true
            rm -f "$plist"
            ok "Removed LaunchAgent"
        fi
    elif command -v systemctl &>/dev/null; then
        systemctl --user stop jouled.service 2>/dev/null || true
        systemctl --user disable jouled.service 2>/dev/null || true
        rm -f "$HOME/.config/systemd/user/jouled.service"
        systemctl --user daemon-reload 2>/dev/null || true
        ok "Removed systemd service"
    fi

    # Remove binaries
    for bin in $JOULEDB_BINARIES; do
        if [ -f "$JOULEDB_INSTALL_DIR/$bin" ]; then
            sudo rm -f "$JOULEDB_INSTALL_DIR/$bin"
            ok "Removed $JOULEDB_INSTALL_DIR/$bin"
        fi
    done

    # Ask about data
    if [ -d "$JOULEDB_DATA_DIR" ]; then
        printf "\n"
        printf "  Remove ${BOLD}$JOULEDB_DATA_DIR${RESET} (daemon state, instance data)? [y/N] "
        read -r answer
        if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
            rm -rf "$JOULEDB_DATA_DIR"
            ok "Removed $JOULEDB_DATA_DIR"
        else
            info "Kept $JOULEDB_DATA_DIR"
        fi
    fi

    printf "\n${GREEN}${BOLD}  JouleDB uninstalled.${RESET}\n\n"
    exit 0
}

# --- Platform detection ---
detect_platform() {
    local os arch

    os="$(uname -s)"
    arch="$(uname -m)"

    case "$os" in
        Darwin)  OS="darwin" ;;
        Linux)   OS="linux" ;;
        MINGW*|MSYS*|CYGWIN*) OS="windows" ;;
        *)       fatal "Unsupported OS: $os" ;;
    esac

    case "$arch" in
        x86_64|amd64)  ARCH="x64" ;;
        arm64|aarch64) ARCH="arm64" ;;
        *)             fatal "Unsupported architecture: $arch" ;;
    esac

    PLATFORM="${OS}-${ARCH}"
    info "Detected platform: $PLATFORM"
}

# --- Download and install binaries ---
install_binaries() {
    info "Installing JouleDB binaries..."

    local version="${JOULEDB_VERSION:-latest}"
    local base_url="https://github.com/$JOULEDB_REPO/releases"

    if [ "$version" = "latest" ]; then
        base_url="$base_url/latest/download"
    else
        base_url="$base_url/download/v$version"
    fi

    local ext="tar.gz"
    [ "$OS" = "windows" ] && ext="zip"

    local archive="jouledb-${PLATFORM}.${ext}"
    local url="$base_url/$archive"
    local tmp_dir
    tmp_dir="$(mktemp -d)"

    info "Downloading $url"
    if ! curl -fsSL "$url" -o "$tmp_dir/$archive" 2>/dev/null; then
        fatal "Download failed. Check https://github.com/$JOULEDB_REPO/releases for available versions.\n\n  To install from source:\n    git clone https://github.com/$JOULEDB_REPO\n    cd jouledb && cargo build --release\n    sudo cp target/release/jouledb target/release/jouledb-cli $JOULEDB_INSTALL_DIR/"
    fi

    info "Extracting..."
    if [ "$ext" = "tar.gz" ]; then
        tar -xzf "$tmp_dir/$archive" -C "$tmp_dir"
    else
        unzip -q "$tmp_dir/$archive" -d "$tmp_dir"
    fi

    for bin in $JOULEDB_BINARIES; do
        local src="$tmp_dir/$bin"
        [ "$OS" = "windows" ] && src="${src}.exe"
        if [ -f "$src" ]; then
            sudo cp "$src" "$JOULEDB_INSTALL_DIR/$bin"
            sudo chmod 755 "$JOULEDB_INSTALL_DIR/$bin"
            ok "$bin -> $JOULEDB_INSTALL_DIR/$bin"
        else
            warn "$bin not found in archive (skipped)"
        fi
    done

    rm -rf "$tmp_dir"
}

# --- Set up data directory ---
setup_data_dir() {
    mkdir -p "$JOULEDB_DATA_DIR"
    chmod 700 "$JOULEDB_DATA_DIR"
    ok "Data directory: $JOULEDB_DATA_DIR"
}

# --- Print summary ---
print_summary() {
    printf "\n"
    printf "${GREEN}${BOLD}  JouleDB installed successfully${RESET}\n"
    printf "\n"
    printf "  ${DIM}Binaries${RESET}      ${JOULEDB_INSTALL_DIR}/jouledb\n"
    printf "                ${JOULEDB_INSTALL_DIR}/jouledb-cli\n"
    printf "  ${DIM}Data${RESET}          ${JOULEDB_DATA_DIR}/\n"
    printf "\n"
    printf "  ${BOLD}Get started:${RESET}\n"
    printf "\n"
    printf "    ${CYAN}# Start the daemon${RESET}\n"
    printf "    jouledb-cli daemon start\n"
    printf "\n"
    printf "    ${CYAN}# Run any database with energy telemetry${RESET}\n"
    printf "    jouledb-cli run postgres\n"
    printf "    jouledb-cli run redis\n"
    printf "    jouledb-cli run mysql\n"
    printf "\n"
    printf "    ${CYAN}# Or run JouleDB natively${RESET}\n"
    printf "    jouledb-cli server start --foreground\n"
    printf "\n"
    printf "    ${CYAN}# Connect with psql${RESET}\n"
    printf "    psql -h localhost -p 5433 -U joule\n"
    printf "\n"
    printf "  ${DIM}Dashboard${RESET}     http://127.0.0.1:7000\n"
    printf "  ${DIM}Docs${RESET}          https://jouledb.org\n"
    printf "  ${DIM}Uninstall${RESET}     curl -fsSL https://jouledb.org/install.sh | JOULEDB_UNINSTALL=1 sh\n"
    printf "\n"
}

# === Main ===
main() {
    banner

    # Handle uninstall
    if [ "${JOULEDB_UNINSTALL:-}" = "1" ]; then
        uninstall
    fi

    # Detect platform
    detect_platform

    # Install binaries
    install_binaries

    # Set up data directory
    setup_data_dir

    # Print summary
    print_summary
}

main "$@"
