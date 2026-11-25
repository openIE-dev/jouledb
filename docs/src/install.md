# Install

Every channel installs the `jouledb` server & CLI binary (`crates/joule-db-server`), exposing HTTP REST / Arrow Flight / MCP / OpenAI `/v1/*` on `--port` (default `8080`), native TCP on `+1` (`8081`), PostgreSQL `pgwire` on `+2` (`8082`), and binary `JWP` on `+3` (`8083`).

## 1. Build from Source (`Apache-2.0 OR MIT` Workspace)

All 40 Rust workspace crates live in this repository with zero proprietary dependencies (`rust-version = "1.94.1"`):

```bash
git clone https://github.com/openIE-dev/jouledb.git
cd jouledb
cargo build --release --locked -p joule-db-server --bin jouledb
./target/release/jouledb --port 8080 --data ./joule-data
```

## 2. Prebuilt Binary via `cargo-binstall` or GitHub Releases

> **Packaging note:** The [`jouledb` crate on `crates.io`](https://crates.io/crates/jouledb) (`stubs/jouledb`) is a `cargo-binstall` binary-fetcher stub. Use **`cargo binstall jouledb`** (not plain `cargo install jouledb`, which prints this hint and exits with code `2`) or build from the workspace above.

```bash
cargo binstall jouledb
jouledb --version
```

Or download a release archive directly for macOS (`arm64`/`x86_64`), Linux (`x86_64`/`aarch64`), or Windows (`x86_64`):

```bash
curl -fsSL https://github.com/openIE-dev/jouledb/releases/latest/download/jouledb-0.4.0-aarch64-apple-darwin.tar.gz | tar xz
```

## 3. Docker / OCI (`GHCR`)

```bash
docker run -p 8080-8083:8080-8083 -v jdb:/data ghcr.io/openie-dev/jouledb:latest
```

## 4. Connect Any PostgreSQL Client

```bash
psql "host=127.0.0.1 port=8082 user=postgres dbname=postgres sslmode=disable"
```
