# Install

Every channel installs the same thing: the `jouledb` server binary (crate `joule-db-server`). It serves HTTP on `-p` (default 8080), TCP on +1, the Postgres wire protocol on +2, and JWP on +3.

## From source (works today)

```bash
cargo install --locked --git https://github.com/openIE-dev/jouledb joule-db-server
jouledb --version
```

## Prebuilt binaries

The Release workflow builds `jouledb-<version>-<target>.tar.gz` (plus `.sha256`) for `aarch64-apple-darwin`, `x86_64-apple-darwin`, `x86_64-unknown-linux-gnu`, and `aarch64-unknown-linux-gnu`. Windows is experimental. Once a version is tagged:

```bash
cargo binstall jouledb            # the crates.io `jouledb` stub points binstall at these archives
# or
curl -fsSL https://github.com/openIE-dev/jouledb/releases/download/v0.1.0/jouledb-0.1.0-aarch64-apple-darwin.tar.gz | tar xz
```

Plain `cargo install jouledb` from crates.io only installs the stub, which prints these instructions. Use the `--git` form above to compile.

## Homebrew

`packaging/homebrew/jouledb.rb` is the tap formula. Before a release exists: `brew install --HEAD ./packaging/homebrew/jouledb.rb`.

## Docker

```bash
docker build -t jouledb .
docker run -e JOULEDB_JWT_SECRET=change-me -p 8080-8083:8080-8083 -v jdb:/data jouledb
```

## Run and connect

```bash
jouledb -d ./data                 # loopback only, trust auth, like a local Postgres
psql "host=127.0.0.1 port=8082 user=postgres dbname=postgres sslmode=disable"
```

Binding to a non-loopback address (`JOULEDB_BIND_HOST=0.0.0.0`, as the container does) always turns auth on. Set `JOULEDB_JWT_SECRET`, which is also the pgwire password. Drivers tested in CI: psql, psycopg 3, SQLAlchemy 2, node-postgres, and pgJDBC (see `tests/clients`).
