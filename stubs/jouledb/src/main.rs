//! Installer hint stub for `cargo binstall jouledb`.
//!
//! When invoked via `cargo binstall jouledb`, cargo-binstall reads the
//! `[package.metadata.binstall]` table in Cargo.toml and installs the
//! prebuilt `jouledb` server binary directly from GitHub Releases.

fn main() {
    let name = env!("CARGO_PKG_NAME");
    let version = env!("CARGO_PKG_VERSION");
    let repo = env!("CARGO_PKG_REPOSITORY");

    eprintln!(
        "\
{name} v{version} (crates.io installer stub)

To install the prebuilt `{name}` server binary without compiling from source:

    cargo binstall {name}

Or download a prebuilt archive (macOS ARM64/x86_64, Linux GLIBC <= 2.35) from:

    {repo}/releases/tag/v{version}

To compile the full server from source:

    git clone {repo}
    cd jouledb && cargo build --release --locked -p joule-db-server --bin jouledb
"
    );
    std::process::exit(2);
}
