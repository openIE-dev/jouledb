# JouleDB (`v0.6.0` — Pre-1.0 Research Preview)

**An experimental energy-metered multi-model database in Rust (`Apache-2.0 OR MIT`) exploring Copy-on-Write `SSI` storage, PostgreSQL `pgwire` SQL, 5-node TCP Raft consensus, `10,240`-bit `Amorphic` VSA storage, and `.jsyn` sub-1-bit sentence embeddings.**

[![Release](https://github.com/openIE-dev/jouledb/actions/workflows/release.yml/badge.svg)](https://github.com/openIE-dev/jouledb/releases)
[![Quality](https://github.com/openIE-dev/jouledb/actions/workflows/quality.yml/badge.svg)](https://github.com/openIE-dev/jouledb/actions/workflows/quality.yml)
[![Docs](https://github.com/openIE-dev/jouledb/actions/workflows/docs.yml/badge.svg)](https://openie-dev.github.io/jouledb)
[![crates.io](https://img.shields.io/crates/v/jouledb.svg)](https://crates.io/crates/jouledb)
[![Hugging Face Embedder](https://img.shields.io/badge/Hugging%20Face-MiniLM%20.jsyn%20(cos%200.944)-0D9488)](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit)
[![Website](https://img.shields.io/badge/Web-jouledb.org-2563EB)](https://jouledb.org)

> **Pre-1.0 Status & Audit Remediation (`v0.6.0`, 2026-11-25):** Following an external source-level audit, we retracted the premature `v1.0.0` GA tag (`crates.io` `1.0.0` yanked), removed the unrelated `1.11M`-LOC `joule-web` crate, deleted synthetic frontier AI benchmark routes (`HLE`, `ARC-AGI-2`, `SimpleQA`, `SWE-bench`, `TAU-bench`, `SafeVLA`), pruned 81 unverified low-BPW causal LLM checkpoints from Hugging Face, fixed the `GLIBC_2.43` Linux binary portability bug (`v0.6.0` targets `GLIBC <= 2.35`, verified on Ubuntu 24.04 LTS), fixed the `cargo binstall` URL template, and published reproducible **Official ClickBench vs. DuckDB** and **5-Node TCP Raft + Official Clojure `elle 0.2.8`** evaluations below (including all losses, timeouts, and bugs uncovered). See [`CHANGELOG.md`](./CHANGELOG.md) for the complete errata.

---

## What Is Real & Verified Today (`41` Workspace Crates · `725k` LOC · `12,265` Tests)

1. **Core Storage, MVCC `SSI` & Recovery (`joule-db-core`, `joule-db-local`):**
   - Copy-on-Write B-tree pages, lock-free positional `pread`/`pwrite`, Cahill Serializable Snapshot Isolation (`SSI` `rw`-antidependency detection), `CRC-32` torn-WAL truncation, LSN-ordered recovery, `ALP + FSST` columnar codecs, and `RaBitQ` 1-bit `POPCNT` vector quantization.
2. **SQL, `pgwire`, Columnar Scans & Incremental Views (`joule-db-query`, `joule-db-server`):**
   - PostgreSQL v3 wire listener (`psql`, `psycopg3`), Parquet `COPY` ingress, vectorized `columnar_exec` aggregate path, `DBSP` `ZSet<T>` incremental view algebra, and experimental multi-language query frontends (Cypher, Datalog, SPARQL, Gremlin, `SigQL`).
3. **5-Node Real TCP Raft Consensus (`raft.rs`, `raft_transport.rs`):**
   - Quorum replication via `RaftNode::propose_and_commit` (with Raft §5.4.2 current-term commit safety), durable `CRC-32` `raft.wal` restart recovery, and a 5-node TCP socket-severing fault-injection test verified **`{:valid? true}`** (`:strict-serializable` `rw-register`) by Kyle Kingsbury's official Clojure `elle/elle 0.2.8`.
4. **Energy Telemetry (`joule-db-energy`, `joule-db-ledger`):**
   - Hardware counter readers on bare metal (**`reported_j`** via Linux `/sys/class/powercap/intel-rapl` and macOS `IOKit SMC` / `powermetrics`) segregated explicitly from analytical cost-model estimates (**`est_j`**) for sub-millisecond operations, anchored in Merkle receipts.
5. **`.jsyn` (`JSYN-v1`) Sub-1-Bit Sentence Embeddings & `Amorphic` Storage (`joule-db-weights`, `joule-db-amorphic`):**
   - Pure-Rust `LittleBit-2 ⊕ Sakana SVF` weight factorization (`0` PyTorch / ONNX Runtime): verified **`0.9441` sentence-embedding cosine similarity** vs upstream `FP32` on [`dcharlot65-openie/all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit) (`1.95 MB`, `0.4378 BPW`, `22.23x` smaller than `FP16`), plus experimental `10,240`-bit Kanerva VSA `Amorphic` storage (`1.35 GB` streamed in `57.64 MB` Peak RSS).

---

## Reproducible External Evaluations (`bench/`)

### 1. Official ClickBench (`hits.parquet`, 105 Columns, 43 Queries) vs. DuckDB `2.0.1`

Executed via [`bench/clickbench_official.py`](./bench/clickbench_official.py) over real `pgwire` (`psycopg 3.3.9`) on `framework-128gb-0000` (`AMD Ryzen AI Max+ 395`, `128 GB` RAM, Linux `rapl-package` **`reported_j`**, `10 s` query timeout):

| Official `hits.parquet` Slice | Parquet Load (`DuckDB` vs `JouleDB`) | JouleDB RSS | Queries `< 10s` | Exact Row Match | Faster (`JouleDB` / `DuckDB`) | Geomean Latency (`JouleDB / DuckDB`) | Geomean Energy (`reported_j`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`100,000` rows** ([JSON](./bench/results/clickbench_official_100k.json)) | `0.192 s` vs `0.203 s` | `722 MB` | **`42 / 43`** (`0` panics) | `33 / 43` | `10` / `32` | **`26.37×` slower** (`median 17.03×`) | **`44.96×` more J** |
| **`1,000,000` rows** ([JSON](./bench/results/clickbench_official_1m.json)) | `0.153 s` vs `1.125 s` | `4,177 MB` | **`23 / 43`** (`20` timeouts) | `18 / 23` | `2` / `21` | **`10.19×` slower** *(23 completed)* | **`12.76×` more J** *(23 completed)* |

- **Candid Scale & Performance Summary:** DuckDB `2.0.1` is **`10×–26×` faster** on geometric mean and completes all `43 / 43` queries in `< 90 ms` at `1M` rows. JouleDB's `columnar_exec` fast path executes `1M`-row `COUNT` / `SUM` / `COUNT(DISTINCT)` group-bys (`Q0..Q16`, `Q20`, `Q33`) in `1.1–140 ms`, whereas complex expression/regex/sort queries (`Q18`, `Q21–Q29`, `Q34–Q42`) fall back to the row-AST executor (`> 10 s` at `1M` rows), `Q32` exhibits an open `AVG` accumulator bug, and in-memory column caching (`~4.18 KB/row` across 105 columns) currently caps single-table Parquet loads well below the full `100M`-row (`~418 GB` projected RAM) dataset on a `128 GB` host.

### 2. 5-Node Real TCP Raft Fault-Injection + Official Clojure `elle 0.2.8`

Executed via [`crates/joule-db-server/tests/raft_tcp_elle_fault_injection.rs`](./crates/joule-db-server/tests/raft_tcp_elle_fault_injection.rs) and [`bench/elle_check.clj`](./bench/elle_check.clj):
- **Setup:** 5 real `RaftNode<KvStateMachine, TcpRaftTransport>` servers with durable `raft.wal` across a 20-link socket-severing `TcpPartitionMatrix`.
- **Nemeses:** Asymmetric `2/5` vs `3/5` TCP partition isolating initial leader -> minority write/quorum-read rejection -> `3/5` higher-term election -> node stop + disk `raft.wal` restart -> link heal & 5-node convergence (`lost_committed_writes = 0`).
- **Official `elle/elle 0.2.8` Verdict ([`raft_tcp_elle_official_verdict.txt`](./bench/results/raft_tcp_elle_official_verdict.txt)):** **`{:valid? true}`** (`rw-register`, `:strict-serializable`, `76` events).

### 3. `.jsyn` (`JSYN-v1`) Fidelity vs. Compression Curve

| Model / Layer | `BPW` | Shrink vs `FP16` | Measured Fidelity vs Upstream | Guidance |
| :--- | :---: | :---: | :---: | :--- |
| **[`all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit)** (`6L` Encoder) | `0.438` | `22.2×` (`1.95 MB`) | **`0.9441` embedding cosine** vs `FP32` | **Recommended** for semantic search & `TriModal RRF` |
| `Gemma-4-E2B` `q_proj` (`r = 8..128`) | `0.028–0.166` | `564×–96.6×` | `0.271–0.455` output cos (`KL 3.87–2.72`) | **Experimental only** — not a drop-in causal LLM (retracted from HF) |
| `Inkling-2-Mini` `w13` (`128`-Exp MoE, `r = 8..128`) | `0.023–0.134` | `701×–120×` | `0.113–0.350` output cos (`KL 2.91–2.59`) | **Experimental only** — not a drop-in causal LLM (retracted from HF) |

---

## Quickstart (`v0.6.0`)

```bash
# 1. Install prebuilt v0.6.0 binary (macOS ARM64/x86_64; Linux x86_64 GLIBC <= 2.35, Ubuntu 22.04/24.04 & Debian 12 ready)
cargo binstall jouledb

# 2. Launch JouleDB server (HTTP :8080, TCP :8081, PgWire :8082, JWP :8083, FlightSQL :8084)
jouledb --port 8080 --data ./joule-data

# 3. Connect with psql or run one-shot SQL
psql -h 127.0.0.1 -p 8082 -U postgres -c "SELECT 1 + 1;"
jouledb query "CREATE TABLE items (id INT PRIMARY KEY, name TEXT)"

# 4. Run fast in-process subsystem smoke checks & technical security-control self-tests
jouledb self-check
jouledb trust
```

### Build & Test from Source

```bash
cargo build --release --locked -p joule-db-server --bin jouledb
cargo test  -p joule-db-core --lib
cargo test  -p joule-db-server --test raft_chaos
cargo test  -p joule-db-server --test raft_tcp_elle_fault_injection
```

---

## Honest Engineering Boundaries (Read Before Adopting)

- **Pre-1.0 research preview (`v0.6.0`), not a battle-tested production RDBMS:** For mission-critical OLTP or full-scale 100M+ row OLAP today, use PostgreSQL or DuckDB; evaluate JouleDB for experimental multi-model, CoW branching, `Amorphic`, and joule-metered workloads.
- **No external compliance certifications:** `jouledb trust` exercises technical security primitives in code (`SHA-256` audit hash-chains, WORM retention timestamps, per-subject DEK zeroization, Luhn redaction, `CycloneDX v1.6` SBOM, Bell-LaPadula lattice checks). **No third-party SOC 2, ISO 27001, HIPAA, PCI-DSS, NIST CMVP, or FedRAMP audit has been conducted.**
- **Exact compiler safety lint scope:** `#![deny(unsafe_code)]` is set in `joule-db-core` with **1** `#![allow(unsafe_code)]` module (`crates/joule-db-core/src/storage/mmap_extent.rs`). `deny(clippy::unwrap_used)` is enforced in **3** crates (`joule-db-core`, `joule-db-branch`, `joule-db-ledger`); the workspace-wide lint level is `warn`.
- **Client SDKs are in-tree (`sdks/*`, `python/`):** Client libraries for Python (`PyO3`/`maturin`), TypeScript, Go, .NET, Swift, Java, and C99 ship in source form inside this repository and are **not yet published** to PyPI, npm, NuGet, or Maven Central.

---

## Documentation & Links

- **Website:** [https://jouledb.org](https://jouledb.org) ([Benchmarks](https://jouledb.org/benchmarks) · [Honest Boundaries](https://jouledb.org/honest) · [Security Controls & Bridges](https://jouledb.org/trust) · [`.jsyn` Fidelity](https://jouledb.org/jsyn))
- **Online Book (`mdBook`):** [https://openie-dev.github.io/jouledb](https://openie-dev.github.io/jouledb)
- **Verified `.jsyn` Sentence Embedder & Benchmark Dataset:** [https://huggingface.co/dcharlot65-openie](https://huggingface.co/dcharlot65-openie)
- **Architecture Whitepaper (`v0.6.0`):** [`docs/WHITEPAPER-JOULEDB-2026-11.md`](./docs/WHITEPAPER-JOULEDB-2026-11.md)
- **`.jsyn` Binary Specification:** [`docs/JSYN-LIVING-WEIGHTS-SPEC.md`](./docs/JSYN-LIVING-WEIGHTS-SPEC.md)
- **Benchmark Scripts & Raw JSON/EDN Outputs:** [`bench/`](./bench/)

## License

Copyright (c) 2026 Open Interface Engineering, Inc. Dual-licensed under either [Apache License, Version 2.0](./LICENSE-APACHE) or [MIT License](./LICENSE-MIT), at your option.
