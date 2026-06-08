//! Stub package for `cargo install` / `cargo binstall`.
//!
//! When invoked via `cargo binstall <name>`, cargo-binstall reads the
//! `[package.metadata.binstall]` table in Cargo.toml, fetches the
//! prebuilt binary from GitHub Releases, and replaces this stub with
//! the real tool — no compilation required.
//!
//! When invoked via `cargo install <name>`, cargo compiles this stub
//! and the resulting binary just prints a hint pointing the user at
//! the faster paths. The real tool is the binary published to GitHub
//! Releases, not this stub.

fn main() {
    let name = env!("CARGO_PKG_NAME");
    let version = env!("CARGO_PKG_VERSION");
    let repo = env!("CARGO_PKG_REPOSITORY");

    eprintln!("\
{name} v{version}

This is the stub installer. To install the real binary fast, use:

    cargo binstall {name}

Or download a prebuilt binary directly from:

    {repo}/releases/latest

If you really want to compile from source, the source mirror is linked
from {repo}.
");
    std::process::exit(2);
}
