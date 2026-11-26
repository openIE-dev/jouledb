# JouleDB (`v0.8.0` — Pre-1.0 Research Preview)

**An experimental energy-metered multi-model database in Rust (`Apache-2.0 OR MIT`) exploring Copy-on-Write `SSI` storage, out-of-core `.jcol` vectorized columnar analytics (`43 / 43` Official ClickBench at `100M` rows), PostgreSQL `pgwire` SQL, 5-node TCP Raft `ReadIndex` consensus, `10,240`-bit `Amorphic` VSA storage, and `.jsyn` sub-1-bit sentence embeddings.**

[![Release](https://github.com/openIE-dev/jouledb/actions/workflows/release.yml/badge.svg)](https://github.com/openIE-dev/jouledb/releases)
[![Quality](https://github.com/openIE-dev/jouledb/actions/workflows/quality.yml/badge.svg)](https://github.com/openIE-dev/jouledb/actions/workflows/quality.yml)
[![Docs](https://github.com/openIE-dev/jouledb/actions/workflows/docs.yml/badge.svg)](https://openie-dev.github.io/jouledb)
[![crates.io](https://img.shields.io/crates/v/jouledb.svg)](https://crates.io/crates/jouledb)
[![Hugging Face Embedder](https://img.shields.io/badge/Hugging%20Face-MiniLM%20.jsyn%20(cos%200.944)-0D9488)](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit)
[![Website](https://img.shields.io/badge/Web-jouledb.org-2563EB)](https://jouledb.org)

> **Pre-1.0 Status & `v0.8.0` Release Summary (`2026-11-25`):** Building on our `v0.6.0` external-audit remediation (`v1.0.0` GA tag retracted/yanked, unrelated `1.11M`-LOC `joule-web` crate and synthetic `frontier_exam_routes.rs` deleted, 81 unverified low-BPW causal LLM exports pruned from Hugging Face) and `v0.7.0` consensus/MVCC fixes, **`v0.8.0`** delivers full-scale out-of-core `100M`-row Official ClickBench execution, head-to-head `PostgreSQL 16.16` `pgwire` OLTP benchmarks, and expanded compiler safety gates:
> - **Full Official `99,997,497`-Row (`14.78 GB`) ClickBench Completed in Bounded RAM (`6.20 GB` RSS, `20.74 GB` Peak `VmHWM`):** Added out-of-core `.jcol` columnar segment files (`disk_col_segments.rs`, `<= 262,144` rows/segment, CRC-32 slice + `JCL1` footer verification, `pread` column-subset I/O) and two-phase `WHERE`-first + 4-pass Radix-64 sharded `execute_segmented` (`columnar_exec.rs`) — cutting `10M`-row RSS **17.5×** (`42.0 GB -> 2.41 GB`), speeding up `10M` geomean latency **3.0×** (`9.74× -> 3.21×`), and completing **`43 / 43` official queries (`0` errors, `0` restarts, `43 / 43` SQL-valid match)** on the **full `99,997,497`-row dataset** (`4.85×` geomean latency, `4.25×` geomean `reported_j` vs DuckDB `2.0.1`, beating DuckDB on `5 / 43` queries at `100M` rows).
> - **Head-to-Head `100`-Warehouse TCP `pgwire` OLTP vs. `PostgreSQL 16.16` (`fsync=on`) & `SQLite 3.53.1`:** Added `O(1)` secondary-index PK/UNIQUE constraint verification, multi-row PK `INSERT` batching, parameterized `SELECT` fast-path binding, and PostgreSQL v3 §55.2.3 `Sync` pipeline error recovery — raising JouleDB `pgwire` (`fsync=on`) throughput **2.54×** (`1,319 -> 3,353.5 TPS` at `c=16`; **`1,659.5 TPS` at `c=1`, `1.44×` faster than `PostgreSQL 16.16` at `c=1`**), with `invariant_verified: true` across all `5` engine/protocol modes.
> - **True Raft §6.4 `ReadIndex` Quorum Linearizable Reads:** `RaftNode::propose_and_commit` (§5.4.2) and `RaftNode::read_linearizable` (§6.4 `ReadIndex`, zero WAL log writes on reads), verified **`{:valid? true}`** by official Clojure `elle/elle 0.2.8`.
> - **Expanded Panic-Lint Gate (`7` Core Crates):** Enforced `#![cfg_attr(not(test), deny(clippy::unwrap_used))]` across **7 crates** (`joule-db-core`, `joule-db-branch`, `joule-db-ledger`, `joule-db-local`, `joule-db-energy`, `joule-db-weights`, `joule-db-query`, plus `#![deny(clippy::unwrap_used)]` on `disk_col_segments.rs`). See [`CHANGELOG.md`](./CHANGELOG.md).

---

## What Is Real & Verified Today (`41` Workspace Crates · `732.5k` LOC · `12,269` Tests)

1. **Core Storage, MVCC `SSI` & Recovery (`joule-db-core`, `joule-db-local`):**
   - Copy-on-Write B-tree pages, lock-free positional `pread`/`pwrite`, Cahill Serializable Snapshot Isolation (`SSI` `rw`-antidependency detection), index-assisted `MvccTableStorage` with `O(1)` PK/UNIQUE index checks and group-committed WAL durability, `CRC-32` torn-WAL truncation, `ALP + FSST` columnar codecs, and `RaBitQ` 1-bit `POPCNT` vector quantization.
2. **Out-of-Core `.jcol` Columnar Analytics, `pgwire` & Incremental Views (`joule-db-query`, `joule-db-server`):**
   - PostgreSQL v3 wire listener (`psql`, `psycopg3` Simple & Extended Pipeline modes), bounded-chunk streaming Parquet `COPY` spilling `262,144`-row `.jcol` disk segments (`I8/I16/I32/I64` narrowing, bitpacked null masks, compact `u8/u16/u32` local dictionaries, CRC-32 integrity), two-phase `WHERE`-first `pread` segment execution + 64-shard Radix `GROUP BY` (`execute_segmented`), `DBSP` `ZSet<T>` incremental view algebra, and experimental multi-language query frontends (Cypher, Datalog, SPARQL, Gremlin, `SigQL`).
3. **5-Node Real TCP Raft Consensus with `ReadIndex` (`raft.rs`, `raft_transport.rs`):**
   - Quorum write commit via `RaftNode::propose_and_commit` (Raft §5.4.2 current-term commit safety), quorum linearizable reads via `RaftNode::read_linearizable` (Raft §6.4 `ReadIndex` heartbeat quorum without WAL log growth), durable `CRC-32` `raft.wal` restart recovery, and a 5-node TCP socket-severing fault-injection test verified **`{:valid? true}`** (`:strict-serializable` `rw-register`) by Kyle Kingsbury's official Clojure `elle/elle 0.2.8`.
4. **Energy Telemetry (`joule-db-energy`, `joule-db-ledger`):**
   - Hardware counter readers on bare metal (**`reported_j`** via Linux `/sys/class/powercap/intel-rapl` and macOS `IOKit SMC` / `powermetrics`) segregated explicitly from analytical cost-model estimates (**`est_j`**) for sub-millisecond operations, anchored in Merkle receipts.
5. **`.jsyn` (`JSYN-v1`) Sub-1-Bit Sentence Embeddings & `Amorphic` Storage (`joule-db-weights`, `joule-db-amorphic`):**
   - Pure-Rust `LittleBit-2 ⊕ Sakana SVF` weight factorization (`0` PyTorch / ONNX Runtime): verified **`0.9441` sentence-embedding cosine similarity** vs upstream `FP32` on [`dcharlot65-openie/all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit) (`1.95 MB`, `0.4378 BPW`, `22.23×` smaller than `FP16`), plus experimental `10,240`-bit Kanerva VSA `Amorphic` storage (`1.35 GB` streamed in `57.64 MB` Peak RSS).

---

## Reproducible External Evaluations (`bench/`)

### 1. Full Official ClickBench (`hits.parquet`, `99,997,497` Rows, `105` Columns, `43` Queries) vs. DuckDB `2.0.1`

Executed via [`bench/clickbench_official.py`](./bench/clickbench_official.py) over real `pgwire` (`psycopg 3.3.9`) on `framework-128gb-0000` (`AMD Ryzen AI Max+ 395`, `128 GB` RAM, Linux `rapl-package` **`reported_j`**):

| Official `hits.parquet` Scale | Parquet Load (`DuckDB` vs `JouleDB`) | JouleDB Load RSS (`VmHWM` Peak) | Queries Succeeded | SQL-Valid Match (`Exact`) | Faster (`JouleDB` / `DuckDB`) | Geomean Latency (`JouleDB / DuckDB`) | Geomean Energy (`reported_j`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`100,000` rows** ([JSON](./bench/results/clickbench_official_100k.json)) | `0.197 s` vs `0.355 s` | `934 MB` | **`43 / 43`** (`0` err) | **`43 / 43`** (`35` exact) | **`25`** / `18` | **`1.30×` slower** (**`median 0.80×`**) | **`1.53×` more J** |
| **`1,000,000` rows** ([JSON](./bench/results/clickbench_official_1m.json)) | `0.147 s` vs `0.965 s` | `1,237 MB` (`2,197 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`28` exact) | **`8`** / `35` | **`3.01×` slower** (`median 3.14×`) | **`3.07×` more J** |
| **`10,000,000` rows** ([JSON](./bench/results/clickbench_official_10m.json)) | `1.402 s` vs `10.755 s` | `2,406 MB` (`9,006 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`29` exact) | **`10`** / `33` | **`3.21×` slower** (`median 4.73×`) | **`3.21×` more J** |
| **`99,997,497` rows — FULL `14.78 GB`** ([JSON](./bench/results/clickbench_official_100m.json)) | `15.546 s` vs `91.621 s` | **`6,199 MB` (`20,740 MB`)** | **`43 / 43`** (`0` err) | **`43 / 43`** (`31` exact) | **`5`** / `38` | **`4.85×` slower** (`median 4.57×`) | **`4.25×` more J** |

- **Candid Full-`100M` Scale & Performance Summary:** Out-of-core `.jcol` segment spilling (`<= 262,144` rows/segment) + `pread` column-subset loading reduced post-load memory from `~4.2 KB/row` (`~410 GB` projected in `v0.7.0`) to **`62 B/row` (`6.20 GB` RSS after loading all `99,997,497` 105-column rows; `20.74 GB` kernel `VmHWM` peak across all 43 queries)**. At full `100M` scale, JouleDB `0.8.0` completes **`43 / 43` queries (`0` errors, `0` restarts)** and beats DuckDB `2.0.1` on **`5 / 43` queries** (`Q0` `0.88 ms` vs `2.07 ms`, `Q6` `1.59 ms` vs `6.65 ms`, `Q7` `8.79 ms` vs `27.79 ms`, `Q28` `REGEXP_REPLACE` `2,524 ms` vs `2,769 ms`, `Q42` `11.61 ms` vs `14.90 ms`). DuckDB `2.0.1` wins **`38 / 43` queries (`4.85×` geomean latency, `5.89×` faster Parquet load)**, with JouleDB's widest gap on `100M`-group high-cardinality `Q32` (`17.42 s` vs `0.98 s`, `17.75×`). *(All 12 non-byte-identical results among the `43 / 43` SQL-valid matches at `100M` rows are verified `ORDER BY` ties on equal aggregate counts and integer epoch-day/second formatting.)*

### 2. Head-to-Head `100`-Warehouse TCP `pgwire` OLTP vs. `PostgreSQL 16.16` & `SQLite 3.53.1`

Executed via [`bench/oltp_pgwire.py`](./bench/oltp_pgwire.py) ([`oltp_pgwire_100wh.json`](./bench/results/oltp_pgwire_100wh.json)) on `framework-128gb-0000` (`rapl-package` **`reported_j`**): `100,000` accounts (`100` warehouses), `50%` indexed point read, `30%` durable `BEGIN; UPDATE accounts ...; INSERT INTO history ...; COMMIT` (`fsync=on`, `synchronous_commit=on`), and `20%` warehouse `COUNT`/`SUM` scan across `1, 4, 8, 16` concurrent workers:

| Engine & Wire Protocol (`invariant_verified = true` on all 5) | `c = 1` TPS (`p50`) | `c = 4` TPS (`p50`) | `c = 8` TPS (`p50`) | `c = 16` TPS (`p50`, `mJ/txn`) | Committed Writes |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **JouleDB `0.8.0` (`pgwire` Simple Query, 4-RTT + `fsync`)** | **`1,659.5`** (`0.30 ms`) | `1,086.4` (`0.97 ms`) | `2,397.5` (`0.96 ms`) | **`3,353.5`** (`1.08 ms`, `17.2 mJ`) | `20,366` (`2` SI conflicts) |
| **PostgreSQL `16.16` (`pgwire` Simple Query, 4-RTT + `fsync`)** | `1,149.1` (`0.08 ms`) | **`3,020.2`** (`0.09 ms`) | **`5,826.9`** (`0.15 ms`) | **`11,452.5`** (`0.25 ms`, `3.79 mJ`) | `51,319` (`0` conflicts) |
| **PostgreSQL `16.16` (`pgwire` Extended Pipeline, 1-RTT + `fsync`)** | `1,255.4` (`0.07 ms`) | `2,984.2` (`0.09 ms`) | `6,045.0` (`0.14 ms`) | **`10,441.8`** (`0.25 ms`, `3.71 mJ`) | `49,555` (`0` conflicts) |
| **JouleDB `0.8.0` (`pgwire` Extended Pipeline, 1-RTT + `fsync`)** | `256.9` (`0.40 ms`) | `490.2` (`1.06 ms`) | `514.4` (`1.07 ms`) | `593.7` (`1.11 ms`, `71.4 mJ`) | `4,394` (`1` SI conflict) |
| **SQLite `3.53.1` (Embedded in-proc C API, `WAL`, `NORMAL`)** | `43,264.1` (`0.005 ms`) | `87,546.8` (`0.006 ms`) | `99,632.0` (`0.007 ms`) | **`111,119.3`** (`0.007 ms`, `0.36 mJ`) | `845,266` (no TCP / `fsync`) |

- **Candid OLTP Summary:** `O(1)` secondary-index PK/UNIQUE checks raised JouleDB Simple Query throughput **2.54×** over `v0.7.0` (`1,318.6 -> 3,353.5 TPS` at `c=16`, and **`1,659.5 TPS` vs `PostgreSQL 16.16`'s `1,149.1 TPS` at `c=1`**). At `c=16`, `PostgreSQL 16.16`'s mature C buffer/lock manager is **`3.42×` faster** (`11,452.5 TPS`, `4.5×` lower `mJ/txn`), and in `psycopg` Extended Query `pipeline()` mode JouleDB's `Describe` (`'D'`) statement probe currently executes a dry-run schema query before `Execute`, limiting pipelined mode to `593.7 TPS` (`invariant_verified = true`).

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

## Quickstart (`v0.8.0`)

```bash
# 1. Install prebuilt v0.8.0 binary (macOS ARM64/x86_64; Linux x86_64 GLIBC <= 2.35, Ubuntu 22.04/24.04 & Debian 12 ready)
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
cargo clippy -p joule-db-core -p joule-db-branch -p joule-db-ledger -p joule-db-local -p joule-db-energy -p joule-db-weights -p joule-db-query --lib
cargo test  -p joule-db-server --lib columnar_exec::tests
cargo test  -p joule-db-server --lib mvcc_adapter::tests
cargo test  -p joule-db-server --test raft_tcp_elle_fault_injection
```

---

## Honest Engineering Boundaries (Read Before Adopting)

- **Pre-1.0 research preview (`v0.8.0`), not a battle-tested production RDBMS:** While JouleDB `0.8.0` now executes all `43 / 43` Official ClickBench queries on the full `99,997,497`-row dataset in bounded RAM (`6.2 GB` load RSS) and `3,353.5 TPS` durable TCP `pgwire` OLTP, mature single-purpose engines (`PostgreSQL 16` at `11.5k TPS` `c=16`, `DuckDB 2.0` at `4.85×` faster `100M` geomean, `SQLite` embedded WAL) remain faster for conventional single-model production workloads.
- **No external compliance certifications:** `jouledb trust` exercises technical security primitives in code (`SHA-256` audit hash-chains, WORM retention timestamps, per-subject DEK zeroization, Luhn redaction, `CycloneDX v1.6` SBOM, Bell-LaPadula lattice checks). **No third-party SOC 2, ISO 27001, HIPAA, PCI-DSS, NIST CMVP, or FedRAMP audit has been conducted.**
- **Exact compiler safety lint scope:** `#![deny(unsafe_code)]` is set in `joule-db-core` with **1** `#![allow(unsafe_code)]` module (`crates/joule-db-core/src/storage/mmap_extent.rs`). `#![cfg_attr(not(test), deny(clippy::unwrap_used))]` is enforced in **7** core crates (`joule-db-core`, `joule-db-branch`, `joule-db-ledger`, `joule-db-local`, `joule-db-energy`, `joule-db-weights`, `joule-db-query`, plus `#![deny(clippy::unwrap_used)]` on `crates/joule-db-server/src/disk_col_segments.rs`); the workspace-wide lint level is `warn`.
- **Client SDKs are in-tree (`sdks/*`, `python/`):** Client libraries for Python (`PyO3`/`maturin`), TypeScript, Go, .NET, Swift, Java, and C99 ship in source form inside this repository and are **not yet published** to PyPI, npm, NuGet, or Maven Central.

---

## Documentation & Links

- **Website:** [https://jouledb.org](https://jouledb.org) ([Benchmarks](https://jouledb.org/benchmarks) · [Honest Boundaries](https://jouledb.org/honest) · [Security Controls & Bridges](https://jouledb.org/trust) · [`.jsyn` Fidelity](https://jouledb.org/jsyn))
- **Online Book (`mdBook`):** [https://openie-dev.github.io/jouledb](https://openie-dev.github.io/jouledb)
- **Verified `.jsyn` Sentence Embedder & Benchmark Dataset:** [https://huggingface.co/dcharlot65-openie](https://huggingface.co/dcharlot65-openie)
- **Architecture Whitepaper (`v0.8.0`):** [`docs/WHITEPAPER-JOULEDB-2026-11.md`](./docs/WHITEPAPER-JOULEDB-2026-11.md)
- **`.jsyn` Binary Specification:** [`docs/JSYN-LIVING-WEIGHTS-SPEC.md`](./docs/JSYN-LIVING-WEIGHTS-SPEC.md)
- **Benchmark Scripts & Raw JSON/EDN Outputs:** [`bench/`](./bench/)

## License

Copyright (c) 2026 Open Interface Engineering, Inc. Dual-licensed under either [Apache License, Version 2.0](./LICENSE-APACHE) or [MIT License](./LICENSE-MIT), at your option.
