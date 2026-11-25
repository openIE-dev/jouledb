# Introduction

**JouleDB (`v1.0.0` GA)** is an open-source (`Apache-2.0 OR MIT`) global multi-model database, un-siloed amorphous semantic continuum, and living-weights (`.jsyn`) AI execution powerhouse written in panic-hardened safe Rust (`40` workspace crates in [`crates/`](https://github.com/openIE-dev/jouledb/tree/main/crates), `#![deny(unsafe_code)]` in `joule-db-core`, `deny(clippy::unwrap_used)` across all core crates).

Every operation—across ACID/SSI SQL, `SQL:2023` analytical cubes & runtime Adaptive Query Execution (`AQE`), 5-node Geo-Partitioned Raft, zero-DDL Amorphous streaming, `JouleCapsule` containers, `JouleKube` manifests, and sub-1-bit `.jsyn` neural forward passes—emits explicit 6-rail silicon energy telemetry (`reported_j` hardware PMU counters on bare metal or `est_j` analytical byte/instruction models; see [Why JouleDB & Honest Boundaries](./design/why.md)).

- **Website & Interactive Simulators:** [https://jouledb.org](https://jouledb.org) ([Amorphic Studio](https://jouledb.org/amorphic) · [Trust & Bridges](https://jouledb.org/trust) · [Benchmarks](https://jouledb.org/benchmarks))
- **Full Rust Workspace, First-Party SDKs & Helm/Terraform (`Apache-2.0 OR MIT`):** [https://github.com/openIE-dev/jouledb](https://github.com/openIE-dev/jouledb)
- **82 Public `.jsyn` Models, Spaces & Benchmark Dataset:** [https://huggingface.co/dcharlot65-openie](https://huggingface.co/dcharlot65-openie)
- **`cargo-binstall` Installer (`crates.io` `v1.0.0`):** [https://crates.io/crates/jouledb](https://crates.io/crates/jouledb) *(fetches signed GitHub Release binaries via `cargo binstall jouledb`; build from source via `cargo build --release --locked -p joule-db-server --bin jouledb`)*

---

## Why `v1.0.0` Is a Global Database Powerhouse

JouleDB `v1.0.0` closes the architectural divide between **enterprise incumbents** (`Oracle`, `PostgreSQL`, `SQL Server`, `Snowflake`, `Databricks`, `MongoDB`, `Elasticsearch`, `kdb+`), **specialized modern engines** (`DuckDB`, `ClickHouse`, `CockroachDB`, `TigerBeetle`, `Qdrant`, `Feldera`), and **2025–2026 academic frontiers** (`Umbra`, `WCOJ`, `DBSP`, `RaBitQ`, `RelBench v2`, `Titans`, `Kanerva VSA`, `FIPS 203/204 PQC`):

1. **[v1.0 Wire/Storage ABI Stability & Microsecond PITR](./pitr-and-abi.md):** Locked `v1.0` TLV forward/backward-compatible framing (`JWAL-v1`, `JSYN-v1`, `JWP-v1`, `ASNP-v1`, `TCAP-v1`), IEEE `CRC-32` bit-rot detection, hash-chained continuous WAL archiving, exact microsecond Point-in-Time Recovery (`RPO = 0`), and Expand-Contract zero-downtime rolling upgrades.
2. **[Un-Siloed Amorphous Continuum & Multi-GB Streaming](./amorphic.md):** Dissolves JSON, CSV, XML, `k=v`, prose logs, and RDF triples (`0` shared field names) into a `10,240`-bit Kanerva VSA hologram in bounded $O(E)$ RAM (`1.35 GB` / `47.5M` atoms in `57.6 MB` Peak RSS), paired with programmable `MorphRule` SI algebra (`25°C == 77°F == 298.15 K`) and 8 fluid visual lenses.
3. **[Zero-Downtime Migration & Universal Bridges](./migration.md):** One-command `jouledb migrate --from postgres|sqlite|mongo|elastic|csv` with order-independent 256-bit Multiset `BLAKE3` reconciliation, CoW branch shadow-traffic parity, live CDC catch-up (`0 ms` write-freeze), `ML-DSA-65` signed cutover certificates, **10 DB wire/REST bridges**, and **7 AI/agentic protocol hubs** (`54` MCP 2026 tools).
4. **[Enterprise `SQL:2023` + Runtime `AQE` + Multi-Region Geo-Raft](./sql-and-consensus.md):** Partitioned window frames, 3D `CUBE`/`ROLLUP`, cycle-safe `WITH RECURSIVE` BOM CTEs, mid-flight Adaptive Query Execution (`>1,000×` join probe reduction under `100×` skew), `1.03M TPS` `TPC-C` Cahill `SSI`, and 5-node 3-region Geo-Raft with `0`-WAN-hop bounded-staleness follower reads certified under `Jepsen` split-brain & clock-skew nemeses.
5. **[Progressive Maker-to-Defense Trust & Polyglot SDKs](./trust.md):** Scales from zero-ceremony maker boards (`Tier 0`) to `GDPR Art. 17` $O(1)$ crypto-shredding (`Tier 1`), `SOC 2 Type II` + `PCI-DSS` (`Tier 2`), `HIPAA` + `SEC 17a-4(f)` WORM (`Tier 3`), `IEC 62443` + `CycloneDX 1.6 SBOM/AI-BOM` (`Tier 4`), and `FedRAMP High` / `CNSA 2.0 FIPS 203 & 204` + Mandatory `Bell-LaPadula MLS` (`Tier 5`), with first-party SDKs in `Rust`, `Python`, `TypeScript/JS`, `Go`, `C#/.NET`, `Swift`, `Java`, `C99/C++20`, `ODBC`, and `WASM/WASI`.
