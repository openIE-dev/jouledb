# Install, Helm & Terraform (`v1.0.0`)

Every installation channel delivers the unified `jouledb` `v1.0.0` server & CLI binary (`crates/joule-db-server`), exposing HTTP REST / Universal Bridges / MCP / OpenAI `/v1/*` on `--port` (default `8080`), native TCP on `+1` (`8081`), PostgreSQL `pgwire` on `+2` (`8082`), binary `JWP` on `+3` (`8083`), and Arrow Flight SQL on `+4` (`8084`).

## 1. Prebuilt Binary via `cargo-binstall` or GitHub Releases

> **Packaging note:** The [`jouledb` crate on `crates.io`](https://crates.io/crates/jouledb) (`stubs/jouledb`, `v1.0.0`) is a `cargo-binstall` binary-fetcher stub. Run **`cargo binstall jouledb`** (plain `cargo install jouledb` prints this hint and exits with code `2`) or compile directly from the workspace below.

```bash
cargo binstall jouledb
jouledb --version   # jouledb 1.0.0
```

Or download a signed `v1.0.0` release archive directly for macOS (`arm64` / `x86_64`), Linux (`x86_64` / `aarch64`), or Windows (`x86_64`):

```bash
curl -fsSL https://github.com/openIE-dev/jouledb/releases/download/v1.0.0/jouledb-1.0.0-aarch64-apple-darwin.tar.gz | tar xz
```

## 2. Build from Source (`Apache-2.0 OR MIT` Workspace)

All 40 Rust workspace crates, first-party polyglot SDKs (`sdks/*`, `python/`), and deployment modules (`deploy/*`) ship in this repository (`rust-version = "1.94.1"`):

```bash
git clone https://github.com/openIE-dev/jouledb.git
cd jouledb
cargo build --release --locked -p joule-db-server --bin jouledb
./target/release/jouledb --port 8080 --data ./joule-data
```

## 3. Docker / OCI (`GHCR`)

```bash
docker run -p 8080-8084:8080-8084 -v jdb:/data ghcr.io/openie-dev/jouledb:1.0.0
```

## 4. Production Kubernetes `Helm` Chart (`deploy/helm/jouledb`)

Deploy a 3-node (`soc2` / `regulated`) or 5-node (`sovereign-mls`) StatefulSet with `PodDisruptionBudget`, zero-trust `NetworkPolicy`, and optional `SEC 17a-4(f)` WORM retention:

```bash
helm upgrade --install jouledb ./deploy/helm/jouledb \
  --namespace jouledb --create-namespace \
  --set trustTier=soc2 \
  --set replicaCount=3 \
  --set persistence.size=200Gi
```

## 5. `Terraform` Compliance Module (`deploy/terraform/jouledb`)

```bash
cd deploy/terraform/jouledb
terraform init
terraform apply -var="cluster_name=jouledb-prod-v1" -var="trust_tier=sovereign-mls"
```

## 6. Connect Standard Clients Immediately (`0` Code Changes)

```bash
# PostgreSQL CLI / SQLAlchemy / psycopg / JDBC
psql "host=127.0.0.1 port=8082 user=postgres dbname=postgres sslmode=disable"

# Master 36-Pillar v1.0.0 Self-Certification
jouledb powerhouse
```
