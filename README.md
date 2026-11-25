# JouleDB (`v0.7.0` — Pre-1.0 Research Preview)

**An experimental energy-metered multi-model database in Rust (`Apache-2.0 OR MIT`) exploring Copy-on-Write `SSI` storage, vectorized columnar analytics, PostgreSQL `pgwire` SQL, 5-node TCP Raft `ReadIndex` consensus, `10,240`-bit `Amorphic` VSA storage, and `.jsyn` sub-1-bit sentence embeddings.**

[![Release](https://github.com/openIE-dev/jouledb/actions/workflows/release.yml/badge.svg)](https://github.com/openIE-dev/jouledb/releases)
[![Quality](https://github.com/openIE-dev/jouledb/actions/workflows/quality.yml/badge.svg)](https://github.com/openIE-dev/jouledb/actions/workflows/quality.yml)
[![Docs](https://github.com/openIE-dev/jouledb/actions/workflows/docs.yml/badge.svg)](https://openie-dev.github.io/jouledb)
[![crates.io](https://img.shields.io/crates/v/jouledb.svg)](https://crates.io/crates/jouledb)
[![Hugging Face Embedder](https://img.shields.io/badge/Hugging%20Face-MiniLM%20.jsyn%20(cos%200.944)-0D9488)](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit)
[![Website](https://img.shields.io/badge/Web-jouledb.org-2563EB)](https://jouledb.org)

> **Pre-1.0 Status & `v0.7.0` Release Summary (`2026-11-25`):** Following our `v0.6.0` external-audit remediation (which retracted the premature `v1.0.0` GA tag, removed the unrelated `1.11M`-LOC `joule-web` crate, deleted synthetic frontier AI exam routes, and pruned 81 unverified low-BPW causal LLM exports from Hugging Face), **`v0.7.0`** lands major engine, consensus, OLTP, and safety upgrades:
> - **Official ClickBench (`43 / 43` Queries Succeeded up to `10,000,000` Rows):** Upgraded `columnar_exec`, width-adaptive integer columns (`I8`/`I16`/`I32`/`I64`), and streaming Parquet `COPY` — completing **`43 / 43` queries (`0` timeouts, `100%` SQL-valid match)** across `100k` (`1.30×` geomean vs DuckDB `2.0.1`, beating DuckDB on `25 / 43` queries), `1M` (`9.34×`), and **`10,000,000` official rows** (`9.74×` geomean latency, `7.50×` geomean `reported_j`).
> - **Concurrent Multi-Client OLTP over Real TCP `pgwire` (`invariant_verified: true`):** Fixed `MvccTableStorage` lazy secondary-index seeding and atomic group-committed `COMMIT` batching (`605 TPS` at `c=1` to **`1,319 TPS` at `c=16`** with durable `fsync` across `100,000` accounts and `6,431` committed balance+ledger transactions, benchmarked candidly against embedded SQLite WAL).
> - **True Raft §6.4 `ReadIndex` Quorum Linearizable Reads:** Added `RaftNode::read_index` and `RaftNode::read_linearizable` (zero log/WAL writes on reads), verified **`{:valid? true}`** by official Clojure `elle/elle 0.2.8`.
> - **Expanded Panic-Lint Gate (`6` Core Crates) & Workspace Alignment:** Enforced `#![cfg_attr(not(test), deny(clippy::unwrap_used))]` in **6 crates** (`joule-db-core`, `joule-db-branch`, `joule-db-ledger`, `joule-db-local`, `joule-db-energy`, `joule-db-weights`) with a blocking CI gate and pruned 173 unused workspace dependencies. See [`CHANGELOG.md`](./CHANGELOG.md).

---

## What Is Real & Verified Today (`41` Workspace Crates · `728k` LOC · `12,268` Tests)

1. **Core Storage, MVCC `SSI` & Recovery (`joule-db-core`, `joule-db-local`):**
   - Copy-on-Write B-tree pages, lock-free positional `pread`/`pwrite`, Cahill Serializable Snapshot Isolation (`SSI` `rw`-antidependency detection), index-assisted `MvccTableStorage` with group-committed WAL durability, `CRC-32` torn-WAL truncation, `ALP + FSST` columnar codecs, and `RaBitQ` 1-bit `POPCNT` vector quantization.
2. **Vectorized SQL, `pgwire`, Streaming Parquet `COPY` & Incremental Views (`joule-db-query`, `joule-db-server`):**
   - PostgreSQL v3 wire listener (`psql`, `psycopg3`), bounded-chunk streaming Parquet `COPY` ingress, morsel-parallel `columnar_exec` engine (`VCol` virtual expressions, dictionary `REGEXP_REPLACE`/`LIKE`, algebraic `SUM(x+k)` sharing, Top-K scans, narrow `I8..I64` columns), `DBSP` `ZSet<T>` incremental view algebra, and experimental multi-language query frontends (Cypher, Datalog, SPARQL, Gremlin, `SigQL`).
3. **5-Node Real TCP Raft Consensus with `ReadIndex` (`raft.rs`, `raft_transport.rs`):**
   - Quorum write commit via `RaftNode::propose_and_commit` (Raft §5.4.2 current-term commit safety), quorum linearizable reads via `RaftNode::read_linearizable` (Raft §6.4 `ReadIndex` heartbeat quorum without WAL log growth), durable `CRC-32` `raft.wal` restart recovery, and a 5-node TCP socket-severing fault-injection test verified **`{:valid? true}`** (`:strict-serializable` `rw-register`) by Kyle Kingsbury's official Clojure `elle/elle 0.2.8`.
4. **Energy Telemetry (`joule-db-energy`, `joule-db-ledger`):**
   - Hardware counter readers on bare metal (**`reported_j`** via Linux `/sys/class/powercap/intel-rapl` and macOS `IOKit SMC` / `powermetrics`) segregated explicitly from analytical cost-model estimates (**`est_j`**) for sub-millisecond operations, anchored in Merkle receipts.
5. **`.jsyn` (`JSYN-v1`) Sub-1-Bit Sentence Embeddings & `Amorphic` Storage (`joule-db-weights`, `joule-db-amorphic`):**
   - Pure-Rust `LittleBit-2 ⊕ Sakana SVF` weight factorization (`0` PyTorch / ONNX Runtime): verified **`0.9441` sentence-embedding cosine similarity** vs upstream `FP32` on [`dcharlot65-openie/all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit) (`1.95 MB`, `0.4378 BPW`, `22.23×` smaller than `FP16`), plus experimental `10,240`-bit Kanerva VSA `Amorphic` storage (`1.35 GB` streamed in `57.64 MB` Peak RSS).

---

## Reproducible External Evaluations (`bench/`)

### 1. Official ClickBench (`hits.parquet`, 105 Columns, 43 Queries) vs. DuckDB `2.0.1`

Executed via [`bench/clickbench_official.py`](./bench/clickbench_official.py) over real `pgwire` (`psycopg 3.3.9`) on `framework-128gb-0000` (`AMD Ryzen AI Max+ 395`, `128 GB` RAM, Linux `rapl-package` **`reported_j`**, `10 s` query timeout):

| Official `hits.parquet` Slice | Parquet Load (`DuckDB` vs `JouleDB`) | JouleDB RSS | Queries `< 10s` | SQL-Valid Match (`Exact`) | Faster (`JouleDB` / `DuckDB`) | Geomean Latency (`JouleDB / DuckDB`) | Geomean Energy (`reported_j`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`100,000` rows** ([JSON](./bench/results/clickbench_official_100k.json)) | `0.197 s` vs `0.355 s` | `934 MB` | **`43 / 43`** (`0` timeouts) | **`43 / 43`** (`35` exact) | **`25`** / `18` | **`1.30×` slower** (**`median 0.80×`**) | **`1.53×` more J** |
| **`1,000,000` rows** ([JSON](./bench/results/clickbench_official_1m.json)) | `0.402 s` vs `2.889 s` | `4,902 MB` | **`43 / 43`** (`0` timeouts) | **`43 / 43`** (`30` exact) | `3` / `40` | **`9.34×` slower** (`median 10.08×`) | **`9.39×` more J** |
| **`10,000,000` rows** ([JSON](./bench/results/clickbench_official_10m.json)) | `0.799 s` vs `16.66 s` | `41,996 MB` | **`43 / 43`** (`0` timeouts) | **`43 / 43`** (`30` exact) | `2` / `41` | **`9.74×` slower** (`median 9.36×`) | **`7.50×` more J** |

- **Candid Scale & Performance Summary:** Upgrading `columnar_exec` in `v0.7.0` eliminated all `20` query timeouts seen in `v0.6.0` and improved `100k`-row geomean latency **20×** (`26.37×` -> `1.30×`, beating DuckDB on `25 / 43` queries; at `10M` rows JouleDB also edges DuckDB on `Q1` `1.17 ms` vs `2.30 ms` and `Q29` `11.05 ms` vs `11.79 ms`). DuckDB `2.0.1` remains **`9.3×–9.7×` faster** on geometric mean at `1M–10M` rows, and JouleDB's queued columnar/row cache (`42.0 GB` RSS at `10M` 105-column rows, `~4.2 KB/row`) still requires out-of-core column eviction to load the full `99,997,497`-row (`~410 GB` projected RAM) file on a `128 GB` machine. *(Note: the 8–13 non-byte-identical queries among the `43 / 43` SQL-valid matches are verified `ORDER BY` ties on equal aggregate counts and integer epoch-day/second formatting.)*

### 2. Concurrent Multi-Client OLTP over Real TCP `pgwire` (`100` Warehouses = `100,000` Accounts)

Executed via [`bench/oltp_pgwire.py`](./bench/oltp_pgwire.py) ([`oltp_pgwire_100wh.json`](./bench/results/oltp_pgwire_100wh.json)) on `framework-128gb-0000` (`rapl-package` **`reported_j`**): `50%` indexed point read, `30%` multi-statement `BEGIN; UPDATE accounts ...; INSERT INTO history ...; COMMIT` (`JOULEDB_SYNC_COMMIT=on` durable `fsync`), and `20%` warehouse `COUNT`/`SUM` scan across `1, 4, 8, 16` concurrent workers:

| Concurrency | JouleDB `0.7.0` (`pgwire` TCP + `fsync`) | JouleDB `p50 / p95 / p99` | JouleDB `mJ/txn` (`reported_j`) | Embedded `SQLite 3.53.1` (`WAL`, `NORMAL`, in-proc) | Post-Run Conservation Invariant |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **`c = 1`** | **`604.97 TPS`** (`0` errors) | `1.15 / 5.83 / 11.07 ms` | `208.5 mJ` | `18,754.8 TPS` (`p50 0.007 ms`, `6.8 mJ`) | **`invariant_verified = true`** |
| **`c = 4`** | **`720.57 TPS`** (`0` errors) | `2.17 / 24.34 / 42.55 ms` | `176.9 mJ` | `35,532.9 TPS` (`p50 0.010 ms`, `3.6 mJ`) | `100,000` accts · `6,431` writes |
| **`c = 8`** | **`888.93 TPS`** (`0` errors) | `2.99 / 32.00 / 44.72 ms` | `145.7 mJ` | `36,273.3 TPS` (`p50 0.010 ms`, `3.4 mJ`) | `SUM(balance) == 100,096,250` |
| **`c = 16`** | **`1,318.62 TPS`** (`1` SI conflict) | `3.52 / 44.44 / 60.82 ms` | `97.5 mJ` | `45,142.2 TPS` (`p50 0.009 ms`, `2.5 mJ`) | Exact match (`expected == actual`) |

- **Candid OLTP Summary:** Building `oltp_pgwire.py` uncovered and fixed three real MVCC bugs in `v0.7.0` (post-startup `COPY` rows missing from `MvccTableStorage`, global `begin_batch` contention across concurrent `COMMIT`s, and missing `amorphic_id` writeback on `INSERT` -> `UPDATE`), raising `c=16` throughput from `47.5 TPS` to **`1,318.6 TPS`** with bit-exact conservation. Embedded in-process SQLite WAL (`synchronous=NORMAL`, zero TCP wire hops, non-`fsync` commits) remains `31×–34×` faster than JouleDB over TCP `pgwire` (4 socket round-trips + `fsync` per write transaction).

### 3. 5-Node Real TCP Raft (`ReadIndex`) Fault-Injection + Official Clojure `elle 0.2.8`

Executed via [`crates/joule-db-server/tests/raft_tcp_elle_fault_injection.rs`](./crates/joule-db-server/tests/raft_tcp_elle_fault_injection.rs) and [`bench/elle_check.clj`](./bench/elle_check.clj):
- **Setup:** 5 real `RaftNode<KvStateMachine, TcpRaftTransport>` servers with durable `raft.wal` across a 20-link socket-severing `TcpPartitionMatrix`, using `propose_and_commit` for writes and Raft §6.4 **`read_linearizable` (`ReadIndex` heartbeat quorum, zero log growth)** for reads.
- **Nemeses:** Asymmetric `2/5` vs `3/5` TCP partition isolating initial leader -> minority write & `ReadIndex` rejection -> `3/5` higher-term election -> node stop + disk `raft.wal` restart -> link heal & 5-node convergence (`lost_committed_writes = 0`).
- **Official `elle/elle 0.2.8` Verdict ([`raft_tcp_elle_official_verdict.txt`](./bench/results/raft_tcp_elle_official_verdict.txt)):** **`{:valid? true}`** (`rw-register`, `:strict-serializable`, `76` events).

### 4. `.jsyn` (`JSYN-v1`) Fidelity vs. Compression Curve

| Model / Layer | `BPW` | Shrink vs `FP16` | Measured Fidelity vs Upstream | Guidance |
| :--- | :---: | :---: | :---: | :--- |
| **[`all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit)** (`6L` Encoder) | `0.438` | `22.2×` (`1.95 MB`) | **`0.9441` embedding cosine** vs `FP32` | **Recommended** for semantic search & `TriModal RRF` |
| `Gemma-4-E2B` `q_proj` (`r = 8..128`) | `0.028–0.166` | `564×–96.6×` | `0.271–0.455` output cos (`KL 3.87–2.72`) | **Experimental only** — not a drop-in causal LLM (retracted from HF) |
| `Inkling-2-Mini` `w13` (`128`-Exp MoE, `r = 8..128`) | `0.023–0.134` | `701×–120×` | `0.113–0.350` output cos (`KL 2.91–2.59`) | **Experimental only** — not a drop-in causal LLM (retracted from HF) |

---

## Quickstart (`v0.7.0`)

```bash
# 1. Install prebuilt v0.7.0 binary (macOS ARM64/x86_64; Linux x86_64 GLIBC <= 2.35, Ubuntu 22.04/24.04 & Debian 12 ready)
cargo binstall jouledb

# 2. Launch JouleDB server (HTTP :8080, TCP :8081, PgWire :8082, JWP :8083, FlightSQL :8084)
jouledb --port 8080 --data ./joule-data

# 3. Connect with psql or run one-shot SQL
psql -h 127.0.0.1 -p 8082 -U postgres -c "SELECT version();"
jouledb query "CREATE TABLE items (id INT PRIMARY KEY, name TEXT)"

# 4. Run fast in-process subsystem smoke checks & technical security-control self-tests
jouledb self-check
jouledb trust
```

### Build & Test from Source

```bash
cargo build --release --locked -p joule-db-server --bin jouledb
cargo clippy -p joule-db-core -p joule-db-branch -p joule-db-ledger -p joule-db-local -p joule-db-energy -p joule-db-weights --lib
cargo test  -p joule-db-server --lib mvcc_adapter::tests
cargo test  -p joule-db-server --test raft_tcp_elle_fault_injection
```

---

## Honest Engineering Boundaries (Read Before Adopting)

- **Pre-1.0 research preview (`v0.7.0`), not a battle-tested production RDBMS:** For mission-critical high-throughput OLTP or full `100M+`-row out-of-core OLAP today, use PostgreSQL, SQLite, or DuckDB; evaluate JouleDB for experimental multi-model, CoW branching, `Amorphic`, and joule-metered workloads.
- **No external compliance certifications:** `jouledb trust` exercises technical security primitives in code (`SHA-256` audit hash-chains, WORM retention timestamps, per-subject DEK zeroization, Luhn redaction, `CycloneDX v1.6` SBOM, Bell-LaPadula lattice checks). **No third-party SOC 2, ISO 27001, HIPAA, PCI-DSS, NIST CMVP, or FedRAMP audit has been conducted.**
- **Exact compiler safety lint scope:** `#![deny(unsafe_code)]` is set in `joule-db-core` with **1** `#![allow(unsafe_code)]` module (`crates/joule-db-core/src/storage/mmap_extent.rs`). `#![cfg_attr(not(test), deny(clippy::unwrap_used))]` is enforced in **6** core crates (`joule-db-core`, `joule-db-branch`, `joule-db-ledger`, `joule-db-local`, `joule-db-energy`, `joule-db-weights`); the workspace-wide lint level is `warn`.
- **Client SDKs are in-tree (`sdks/*`, `python/`):** Client libraries for Python (`PyO3`/`maturin`), TypeScript, Go, .NET, Swift, Java, and C99 ship in source form inside this repository and are **not yet published** to PyPI, npm, NuGet, or Maven Central.

---

## Documentation & Links

- **Website:** [https://jouledb.org](https://jouledb.org) ([Benchmarks](https://jouledb.org/benchmarks) · [Honest Boundaries](https://jouledb.org/honest) · [Security Controls & Bridges](https://jouledb.org/trust) · [`.jsyn` Fidelity](https://jouledb.org/jsyn))
- **Online Book (`mdBook`):** [https://openie-dev.github.io/jouledb](https://openie-dev.github.io/jouledb)
- **Verified `.jsyn` Sentence Embedder & Benchmark Dataset:** [https://huggingface.co/dcharlot65-openie](https://huggingface.co/dcharlot65-openie)
- **Architecture Whitepaper (`v0.7.0`):** [`docs/WHITEPAPER-JOULEDB-2026-11.md`](./docs/WHITEPAPER-JOULEDB-2026-11.md)
- **`.jsyn` Binary Specification:** [`docs/JSYN-LIVING-WEIGHTS-SPEC.md`](./docs/JSYN-LIVING-WEIGHTS-SPEC.md)
- **Benchmark Scripts & Raw JSON/EDN Outputs:** [`bench/`](./bench/)

## License

Copyright (c) 2026 Open Interface Engineering, Inc. Dual-licensed under either [Apache License, Version 2.0](./LICENSE-APACHE) or [MIT License](./LICENSE-MIT), at your option.
