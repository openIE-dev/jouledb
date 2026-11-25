# Contributing

This repository hosts the **open-source (`Apache-2.0 OR MIT`) 40-crate Rust workspace, Python SDK, documentation, and release workflows** for JouleDB (`crates/`, `python/`, `docs/`, `examples/`).

*(Historical note: During `v0.1` prior to `v0.2.0`, this repository served only as a binary release mirror; starting in `v0.2.0`, the full Rust workspace source lives directly in `crates/` under `MIT OR Apache-2.0`.)*

## What We Welcome

- **Bug reports & benchmark replications:** Open an issue with exact reproduction steps, OS/CPU details, and logs.
- **Documentation & benchmark clarity fixes:** Typos, protocol clarifications, broken links, and prior-art additions—open a PR directly against `docs/` or `examples/`.
- **Targeted engine/SDK fixes:** Focused, well-tested PRs against `crates/` or `python/` that pass `cargo test --workspace --locked` and maintain `#![deny(unsafe_code)]` in `joule-db-core` and `deny(clippy::unwrap_used)` across core crates.

## Reporting Issues

When filing a bug:

- Use the issue templates in `.github/ISSUE_TEMPLATE/`
- Include the exact version (`jouledb --version` or commit SHA)
- Include OS + architecture (`uname -m -r -s` or equivalent)
- Include a minimal reproduction
- Attach logs with `RUST_LOG=debug` if relevant

## Discussions

For questions and design discussions, use [GitHub Discussions](../../discussions) rather than issues.

## Code of Conduct

By participating in this project, you agree to abide by the [Code of Conduct](./CODE_OF_CONDUCT.md).

## License

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this repository is dual-licensed under `MIT OR Apache-2.0` ([`LICENSE-MIT`](./LICENSE-MIT) and [`LICENSE-APACHE`](./LICENSE-APACHE)), without any additional terms or conditions.
