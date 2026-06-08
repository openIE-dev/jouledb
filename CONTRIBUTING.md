# Contributing

This repository hosts the **public release surface** for an Open Interface Engineering (openIE) project. The source lives in a private mirror; contributions to this repository follow these rules:

## What we accept

- **Bug reports** — open an issue describing the problem, environment, and reproduction steps
- **Documentation fixes** — typos, clarifications, broken links: open a PR directly
- **Example improvements** — bug fixes or new examples in `examples/`: open a PR

## What we don't accept here

- **Source code changes** — the source is private; the binaries and examples in this repository are mirrors of internal work. We can't merge source PRs.
- **Feature requests outside our roadmap** — feature ideas are welcome as issues, but we may close them if they don't fit.

## Reporting issues

When filing a bug:

- Use the issue templates in `.github/ISSUE_TEMPLATE/`
- Include the exact version (`<tool> --version`)
- Include OS + architecture (`uname -a` or equivalent)
- Include a minimal reproduction
- Attach logs with `RUST_LOG=debug` if relevant

## Discussions

For questions and design discussions, use [GitHub Discussions](../../discussions) rather than issues.

## Code of Conduct

By participating in this project, you agree to abide by the [Code of Conduct](./CODE_OF_CONDUCT.md).

## License

Contributions to documentation and examples are accepted under the same terms as the rest of this repository: documentation under CC-BY-4.0, examples under Apache-2.0, binaries under BSL-1.1.
