# Install & Deployment Templates (`v0.13.0` Preview)

All installation channels deliver the `jouledb` `v0.13.0` server and CLI binary (`crates/joule-db-server`), exposing HTTP REST & Streamable HTTP MCP (`/mcp`, including the built-in `ui://jouledb/studio.html` MCP App) on `--port` (default `8080`), local `stdio` MCP via `jouledb mcp`, native TCP on `+1` (`8081`), PostgreSQL `pgwire` on `+2` (`8082`), and binary `JWP` on `+3` (`8083`).

## 1. Prebuilt Binary via `cargo-binstall`, Claude `.mcpb` Bundle, or GitHub Releases

> **Packaging & Linux `glibc` Portability Note:**
> - The [`jouledb` crate on `crates.io`](https://crates.io/crates/jouledb) (`stubs/jouledb`, `v0.13.0`; `1.0.0` is yanked) is a small **`cargo-binstall` binary-fetcher stub** (`15 KB`). Running `cargo binstall jouledb` downloads the prebuilt release archive from GitHub Releases; plain `cargo install jouledb` prints source-build instructions and exits with status `2`.
> - `v0.13.0` `x86_64-unknown-linux-gnu` release binaries are built against **`GLIBC <= 2.35`** (`Debian 12` / `Ubuntu 22.04` baseline) and verified on **Ubuntu 24.04 LTS (`glibc 2.39`)**.
> - One-click **Claude Desktop `.mcpb` (`manifest_version: "0.3"`)** bundles (`jouledb-0.13.0-{aarch64-apple-darwin,x86_64-apple-darwin,x86_64-unknown-linux-gnu}.mcpb`), prebuilt Python `cp39-abi3` wheels (`.whl`), and the TypeScript `openie-jouledb-0.13.0.tgz` tarball are attached to [GitHub Release `v0.13.0`](https://github.com/openIE-dev/jouledb/releases/tag/v0.13.0). See [Claude Apps & MCP Studio](./claude-apps.md).

```bash
cargo binstall jouledb
jouledb --version   # jouledb 0.13.0
```

Or download the `v0.13.0` release tarball directly for macOS (`aarch64` / `x86_64`) or Linux (`x86_64`):

```bash
curl -fsSL https://github.com/openIE-dev/jouledb/releases/download/v0.13.0/jouledb-0.13.0-x86_64-unknown-linux-gnu.tar.gz | tar xz
./jouledb-0.13.0-x86_64-unknown-linux-gnu/jouledb --version
```

## 2. Build from Source (`Apache-2.0 OR MIT`)

All `41` Rust workspace crates (**`41 / 41` [`100%`] enforced with `#![cfg_attr(not(test), deny(clippy::unwrap_used))]`**, zero bypasses) ship in this repository:

```bash
git clone https://github.com/openIE-dev/jouledb.git
cd jouledb
cargo build --release --locked -p joule-db-server --bin jouledb
./target/release/jouledb --port 8080 --data ./joule-data
```

## 3. Connect Standard PostgreSQL Clients, Claude/MCP Hosts & Run Self-Checks

```bash
# Connect psql, psycopg3, SQLAlchemy, node-pg, or JDBC over pgwire (:8082)
psql "host=127.0.0.1 port=8082 user=postgres dbname=postgres sslmode=disable"

# Register durable local MCP server + Studio UI in Claude Code
claude mcp add jouledb -- jouledb mcp --data ~/.jouledb/claude

# Run the 6 fast in-process subsystem smoke checks (not an external benchmark)
jouledb self-check
```

## 4. Starter `Helm` & `Terraform` Templates (`deploy/`)

`deploy/helm/jouledb` (`5` files, `~198` lines) and `deploy/terraform/jouledb` (`main.tf`, `52` lines) provide minimal starter templates for Kubernetes `StatefulSet` deployment and configuration flags. They are **starter infrastructure templates**, not externally audited compliance modules.

## 5. Package Registry Status Disclosure (`v0.13.0`)

- **`crates.io` (`jouledb` `0.13.0`):** Published (`cargo-binstall` binary fetcher stub).
- **GitHub Releases (`v0.13.0`):** Published — server tarballs (macOS `aarch64`/`x86_64`, Linux `x86_64`), **Claude Desktop `.mcpb` bundles**, **Python `cp39-abi3` `.whl` wheels** (macOS ARM64/x86_64 & `manylinux_2_34_x86_64`), **TypeScript `openie-jouledb-0.13.0.tgz`**, and `SHA256SUMS.txt`.
- **PyPI (`jouledb`), npm (`@openie/jouledb`), NuGet (`OpenIE.JouleDB`), Maven Central:** **Not yet published** on public package registries as of `v0.13.0` (OIDC/token publish jobs are wired in `.github/workflows/release.yml` pending registry setup); install the prebuilt `.whl` / `.tgz` directly from GitHub Release `v0.13.0` or build from `python/` and `sdks/*` in-repo.
