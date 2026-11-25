# JouleDB: An Energy-Metered Multi-Model Database & Sub-1-Bit Weights Research Substrate

## Architecture & Empirical Status Paper — Version `0.7.0` (Pre-1.0 Research Preview, November 2026)
**Open Interface Engineering, Inc. (`OpenIE`)**  
**Website:** [https://jouledb.org](https://jouledb.org) | **Docs Book:** [https://openie-dev.github.io/jouledb](https://openie-dev.github.io/jouledb) | **GitHub:** [openIE-dev/jouledb](https://github.com/openIE-dev/jouledb) | **Hugging Face (`.jsyn` Embedder & Raw Benchmarks):** [huggingface.co/dcharlot65-openie](https://huggingface.co/dcharlot65-openie)  
**License:** `Apache-2.0 OR MIT`

> **Revision & Audit Errata Note (`v0.7.0`, 2026-11-25):** Following our `v0.6.0` external-audit remediation (which retracted the premature `v1.0.0` GA tag, removed the unrelated `1.11M`-line `joule-web` crate, deleted synthetic frontier AI exam endpoints, and retracted 81 unverified low-BPW causal LLM exports from Hugging Face), **`v0.7.0`** describes the **`41`-crate (`727,951` LOC, `12,268` tests)** codebase with **6** `deny(clippy::unwrap_used)` crates, vectorized `43 / 43` Official ClickBench execution up to **`10,000,000` rows**, concurrent multi-client `pgwire` OLTP with verified balance conservation, and 5-node TCP Raft §6.4 **`ReadIndex`** linearizable reads verified by official Clojure `elle 0.2.8`.

---

## Abstract

Modern data and AI stacks partition state across separate OLTP databases, analytical engines, vector indexes, container sandboxes, and neural inference runtimes. **JouleDB (`v0.7.0`)** explores unifying these workloads inside a single open-source Rust workspace where Copy-on-Write (`CoW`) Serializable Snapshot Isolation (`SSI`) tables, vectorized columnar scans, PostgreSQL `pgwire` SQL, 5-node TCP Raft `ReadIndex` consensus, `10,240`-bit Hyperdimensional Computing (`Amorphic`) holograms, and sub-1-bit factorized sentence-embedding bitplanes (`.jsyn`) share one storage and energy-accounting substrate. Every energy figure is tagged by measurement provenance: **`reported_j`** from bare-metal OS hardware counters (Linux RAPL, macOS IOKit SMC) over multi-millisecond windows, or **`est_j`** from analytical cost models for sub-millisecond operations.

---

## 1. Core Subsystems in `v0.7.0` (`41` Workspace Crates)

1. **Storage, `SSI` Concurrency & Group-Committed Recovery (`joule-db-core`, `joule-db-local`):**
   - Lock-free positional `pread`/`pwrite`, `IoRingEngine` extent coalescing, Cahill `SSI` (`rw`-antidependency cycle prevention), index-assisted `MvccTableStorage` with `commit_with_hook` + `GroupSync` shared `fsync`, `CRC-32` torn-WAL truncation, LSN-ordered replay, `ALP + FSST` columnar codecs, `RaBitQ` 1-bit `POPCNT` vector quantization, and double-entry `i128` ledger primitives (`tiger_ledger`).
   - **Compiler Safety Scope:** `#![deny(unsafe_code)]` in `joule-db-core` with **1** `#![allow(unsafe_code)]` module (`crates/joule-db-core/src/storage/mmap_extent.rs`); `#![cfg_attr(not(test), deny(clippy::unwrap_used))]` enforced in **6** crates (`joule-db-core`, `joule-db-branch`, `joule-db-ledger`, `joule-db-local`, `joule-db-energy`, `joule-db-weights`), with `warn` across the remaining workspace crates.
2. **Vectorized Query Execution, `pgwire` & Incremental Views (`joule-db-query`, `joule-db-server`):**
   - PostgreSQL v3 wire server (`pgwire` on `port + 2`), bounded-chunk streaming Parquet `COPY` loader, width-adaptive `I8/I16/I32/I64` columnar vectors, parallel morsel `columnar_exec` engine (`PrepPred` zone-map/dictionary filters, `VCol` virtual expressions, algebraic `SUM(x+k)` sharing, Top-K scans), `DBSP` `ZSet<T>` incremental view maintenance, `Arrow Flight SQL` (`JDB_IPC1`), and experimental Cypher, Datalog, SPARQL, Gremlin, and `SigQL` biosignal frontends.
3. **5-Node Real TCP Raft Consensus with `ReadIndex` (`raft.rs`, `raft_server.rs`, `raft_transport.rs`):**
   - Quorum write replication via `RaftNode::propose_and_commit` (Raft §5.4.2 current-term safety), quorum linearizable reads via `RaftNode::read_linearizable` (Raft §6.4 `ReadIndex` heartbeat quorum with zero WAL log growth), `CRC-32` durable `raft.wal` persistence, and 2PC participant/coordinator recovery logs.
4. **`.jsyn` (`JSYN-v1`) Sub-1-Bit Weights & `Amorphic` Continuum (`joule-db-weights`, `joule-db-amorphic`, `joule-db-gpu`):**
   - Compiles `.safetensors` and `.gguf` checkpoints into packed `u64` sign bitplanes (`LittleBit-2 ⊕ Sakana SVF`) executed via CPU `T-MAC` byte lookup tables or `wgpu` `countOneBits` shaders (`0` PyTorch / `llama.cpp` / ONNX Runtime).
   - Dissolves heterogeneous JSON, CSV, XML, `k=v`, prose logs, and N-Triples into `10,240`-bit Kanerva VSA holograms (`StreamingAmorphicContinuum` + `MorphRule` SI unit normalization).
5. **Confinement, Branching & Energy Ledger (`joule-db-branch`, `joule-db-energy`, `joule-db-ledger`):**
   - $O(1)$ CoW workspace/table branching (`JouleAgentFS`, `JouleCapsule` daemonless OCI v2 layer CAS), `HMAC-SHA256` `ThermoCap` capability tokens, technical security-control primitives (`trust_compliance.rs`), and Merkle-anchored energy receipts (`reported_j` / `est_j`).

---

## 2. Reproducible External Evaluations (`bench/`)

### 2.1 Official ClickBench (`hits.parquet`, 105 Columns, 43 Queries) vs. DuckDB `2.0.1`

Evaluated via [`bench/clickbench_official.py`](../bench/clickbench_official.py) on `framework-128gb-0000` (`AMD Ryzen AI Max+ 395`, 32 threads, `128 GB` RAM, Linux `rapl-package` **`reported_j`**, `10 s` query timeout) over real `pgwire` (`psycopg 3.3.9`):

| Slice of Official `14.78 GB` `hits.parquet` | Load (`DuckDB` vs `JouleDB`) | JouleDB RSS | Queries `< 10s` | SQL-Valid Match (`Exact`) | Faster (`JouleDB` / `DuckDB`) | Geomean Latency (`JouleDB / DuckDB`) | Geomean Energy (`reported_j`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`100,000` rows** | `0.197 s` vs `0.355 s` | `933.8 MB` | **`43 / 43`** (`0` timeouts) | **`43 / 43`** (`35` exact) | **`25`** / `18` | **`1.30×` slower** (**`median 0.80×`**) | **`1.53×` more J** |
| **`1,000,000` rows** | `0.402 s` vs `2.889 s` | `4,902.0 MB` | **`43 / 43`** (`0` timeouts) | **`43 / 43`** (`30` exact) | `3` / `40` | **`9.34×` slower** (`median 10.08×`) | **`9.39×` more J** |
| **`10,000,000` rows** | `0.799 s` vs `16.660 s` | `41,995.9 MB` | **`43 / 43`** (`0` timeouts) | **`43 / 43`** (`30` exact) | `2` / `41` | **`9.74×` slower** (`median 9.36×`) | **`7.50×` more J** |

### 2.2 Concurrent Multi-Client OLTP over Real TCP `pgwire` (`100` Warehouses = `100,000` Accounts)

Evaluated via [`bench/oltp_pgwire.py`](../bench/oltp_pgwire.py) ([`oltp_pgwire_100wh.json`](../bench/results/oltp_pgwire_100wh.json)) across `1, 4, 8, 16` concurrent `psycopg` TCP clients (`50%` indexed point read, `30%` durable `fsync` `BEGIN..UPDATE..INSERT..COMMIT`, `20%` warehouse aggregate):
- **JouleDB `0.7.0` (`pgwire` + `fsync`):** **`604.97 TPS`** (`c=1`, `p50 1.15 ms`, `0` errors) -> **`720.57 TPS`** (`c=4`) -> **`888.93 TPS`** (`c=8`) -> **`1,318.62 TPS`** (`c=16`, `p50 3.52 ms`, `97.5 mJ/txn`, `1` genuine row-level SI conflict cleanly rolled back; **`invariant_verified = true`** across all `6,431` committed writes, `SUM(balance) == 100,096,250`).
- **Candid Baseline Comparison:** Embedded in-process `SQLite 3.53.1` (`WAL`, `synchronous=NORMAL`, no TCP wire hops, non-`fsync` commits) achieves `18,755–45,142 TPS` (`31×–34×` higher TPS).

### 2.3 5-Node Real TCP Raft (`ReadIndex`) Fault-Injection + Official Clojure `elle 0.2.8`

Evaluated via [`crates/joule-db-server/tests/raft_tcp_elle_fault_injection.rs`](../crates/joule-db-server/tests/raft_tcp_elle_fault_injection.rs) and [`bench/elle_check.clj`](../bench/elle_check.clj):
- **Faults Injected:** 5 `RaftNode<KvStateMachine, TcpRaftTransport>` servers across a 20-link socket-severing `TcpPartitionMatrix`; asymmetric `2/5` vs `3/5` split isolating initial leader -> minority write & `ReadIndex` (`read_linearizable`) rejection -> `3/5` higher-term leader election -> follower stop + disk `raft.wal` restart -> partition heal & 5-node convergence (`x = [17; 5]`, `y = [18; 5]`, `lost_committed_writes = 0`).
- **Official `elle 0.2.8` Verdict:** **`{:valid? true}`** (`rw-register`, `:strict-serializable`, `76` events).

### 2.4 `.jsyn` Compression vs. Fidelity Boundaries

- **Verified High-Fidelity Encoder (`0.4378 BPW`):** [`dcharlot65-openie/all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit) (`Apache-2.0`, `1.95 MB`, `22.23×` smaller than `FP16`) achieves **`0.9441` mean sentence-embedding cosine similarity** against upstream `FP32`.
- **Degraded Ultra-Low-BPW Causal LLMs (`0.02–0.17 BPW`, Retracted):** Out-of-sample rank sweeps (`r = 8..128`) on `Gemma-4-E2B` (`q_proj`) and `Inkling-2-Mini` (`128`-Exp MoE) yield **`0.113–0.455` output cosine** (`KL = 2.59–3.87`, `65.6%` Top-5 agreement at `r = 128`). All 81 unverified causal LLM checkpoints remain deleted from Hugging Face.

---

## 3. Explicit Engineering & Governance Boundaries

1. **Pre-1.0 Research Preview (`v0.7.0`):** While ClickBench now executes `43 / 43` queries cleanly up to `10,000,000` rows, retaining queued `ColumnBatch` + `TableColumns` vectors in RAM (`42.0 GB` RSS at `10M` 105-column rows) means loading the full `100M`-row (`~410 GB`) dataset on `128 GB` hardware requires out-of-core column eviction, and DuckDB / SQLite remain faster on single-node OLAP / embedded OLTP.
2. **No Third-Party Compliance Certifications:** `trust_compliance.rs` implements technical controls (SHA-256 hash-chained audit logs, WORM retention locks, per-subject DEK zeroization, Luhn redaction, `CycloneDX v1.6` SBOM, Bell-LaPadula lattice labels, and unvalidated Rust `ML-KEM-768`/`ML-DSA-65` signatures). **No SOC 2, ISO 27001, HIPAA, PCI-DSS, NIST CMVP, or FedRAMP audit has been performed.**
3. **In-Tree SDKs & Portable Binaries:** Prebuilt `v0.7.0` server binaries target macOS (`aarch64`/`x86_64`) and Linux (`GLIBC <= 2.35`, verified on Ubuntu 24.04 LTS & Debian 12). Client SDKs (`sdks/*`, `python/`) live in-tree in this repository and are not yet published to PyPI, npm, or NuGet.
