# Install

Every channel installs the `jouledb` server & CLI binary (`joule-db-server`), exposing HTTP REST / Arrow Flight / MCP / OpenAI `/v1/*` on `--port` (default `8080`), native TCP on `+1` (`8081`), PostgreSQL `pgwire` on `+2` (`8082`), and binary `JWP` on `+3` (`8083`).

## 1. Prebuilt Binary via `cargo-binstall` (Fastest)

```bash
cargo binstall jouledb
jouledb --version
```

Or download a signed release archive (`v0.3.0`) directly for macOS (`arm64`/`x86_64`), Linux (`x86_64`/`aarch64`), or Windows (`x86_64`):

```bash
curl -fsSL https://github.com/openIE-dev/jouledb/releases/download/v0.3.0/jouledb-0.3.0-aarch64-apple-darwin.tar.gz | tar xz
```

## 2. Docker / OCI (`GHCR`)

```bash
docker run -p 8080-8083:8080-8083 -v jdb:/data ghcr.io/openie-dev/jouledb:0.3.0
```

## 3. Build from Source

```bash
git clone https://github.com/openIE-dev/jouledb.git
cd jouledb
cargo build --release -p joule-db-server --bin jouledb
./target/release/jouledb --port 8080 --data ./joule-data
```

## 4. Connect Any PostgreSQL Client

```bash
psql "host=127.0.0.1 port=8082 user=postgres dbname=postgres sslmode=disable"
```
