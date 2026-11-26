# Install & Deployment Templates (`v0.8.0` Preview)

All installation channels deliver the `jouledb` `v0.8.0` server and CLI binary (`crates/joule-db-server`), exposing HTTP REST & MCP on `--port` (default `8080`), native TCP on `+1` (`8081`), PostgreSQL `pgwire` on `+2` (`8082`), and binary `JWP` on `+3` (`8083`).

## 1. Prebuilt Binary via `cargo-binstall` or GitHub Releases

> **Packaging & Linux `glibc` Portability Note:**
> - The [`jouledb` crate on `crates.io`](https://crates.io/crates/jouledb) (`stubs/jouledb`, `v0.8.0`; `1.0.0` is yanked) is a small **`cargo-binstall` binary-fetcher stub** (`15 KB`). Running `cargo binstall jouledb` downloads the prebuilt release archive from GitHub Releases; plain `cargo install jouledb` prints source-build instructions and exits with status `2`.
> - `v0.8.0` `x86_64-unknown-linux-gnu` release binaries are built against **`GLIBC <= 2.35`** (`Debian 12` / `Ubuntu 22.04` baseline) and verified on **Ubuntu 24.04 LTS (`glibc 2.39`)**.

```bash
cargo binstall jouledb
jouledb --version   # jouledb 0.8.0
```

Or download the `v0.8.0` release tarball directly for macOS (`aarch64` / `x86_64`) or Linux (`x86_64`):

```bash
curl -fsSL https://github.com/openIE-dev/jouledb/releases/download/v0.8.0/jouledb-0.8.0-x86_64-unknown-linux-gnu.tar.gz | tar xz
./jouledb-0.8.0-x86_64-unknown-linux-gnu/jouledb --version
```

## 2. Build from Source (`Apache-2.0 OR MIT`)

All `41` Rust workspace crates ship in this repository:

```bash
git clone https://github.com/openIE-dev/jouledb.git
cd jouledb
cargo build --release --locked -p joule-db-server --bin jouledb
./target/release/jouledb --port 8080 --data ./joule-data
```

## 3. Connect Standard PostgreSQL Clients & Run Self-Checks

```bash
# Connect psql, psycopg3, SQLAlchemy, node-pg, or JDBC over pgwire (:8082)
psql "host=127.0.0.1 port=8082 user=postgres dbname=postgres sslmode=disable"

# Run the 6 fast in-process subsystem smoke checks (not an external benchmark)
jouledb self-check
```

## 4. Starter `Helm` & `Terraform` Templates (`deploy/`)

`deploy/helm/jouledb` (`5` files, `~198` lines) and `deploy/terraform/jouledb` (`main.tf`, `52` lines) provide minimal starter templates for Kubernetes `StatefulSet` deployment and configuration flags. They are **starter infrastructure templates**, not externally audited compliance modules.

## 5. Package Registry Status Disclosure

- **`crates.io` (`jouledb` `0.8.0`):** Published (`cargo-binstall` binary fetcher stub).
- **GitHub Releases (`v0.8.0`):** Published (macOS `aarch64`/`x86_64` & Linux `x86_64` server tarballs + `SHA256SUMS.txt`).
- **PyPI (`jouledb`), npm (`@openie/jouledb`), NuGet (`OpenIE.JouleDB`):** **Not published** on public package registries as of `v0.8.0`. Build from `python/` and `sdks/*` in-repo.
