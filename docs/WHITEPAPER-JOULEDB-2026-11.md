# JouleDB: An Energy-Metered Multi-Model Database & Sub-1-Bit Weights Research Substrate

## Architecture & Empirical Status Paper — Version `0.6.0` (Pre-1.0 Research Preview, November 2026)
**Open Interface Engineering, Inc. (`OpenIE`)**  
**Website:** [https://jouledb.org](https://jouledb.org) | **Docs Book:** [https://openie-dev.github.io/jouledb](https://openie-dev.github.io/jouledb) | **GitHub:** [openIE-dev/jouledb](https://github.com/openIE-dev/jouledb) | **Hugging Face (`.jsyn` Embedder & Raw Benchmarks):** [huggingface.co/dcharlot65-openie](https://huggingface.co/dcharlot65-openie)  
**License:** `Apache-2.0 OR MIT`

> **Revision & Audit Errata Note (`v0.6.0`, 2026-11-25):** Following an external source-level audit, OpenIE retracted the premature `v1.0.0` GA tag (`crates.io` `1.0.0` yanked), removed the unrelated `1.11M`-line `joule-web` crate, deleted synthetic frontier AI exam endpoints (`frontier_exam_routes.rs`), retracted 81 unverified low-BPW causal LLM exports from Hugging Face, and aligned all crate counts, lint scopes, compliance disclaimers, and energy provenance tags (`reported_j` vs `est_j`). This document describes the **`41`-crate (`724,986` LOC, `12,265` tests) `v0.6.0` codebase** and reports reproducible external evaluations on **Official ClickBench vs. DuckDB** and **5-Node Real TCP Raft + Official Clojure `elle 0.2.8`**.

---

## Abstract

Modern data and AI stacks partition state across separate OLTP databases, analytical engines, vector indexes, container sandboxes, and neural inference runtimes. **JouleDB (`v0.6.0`)** explores unifying these workloads inside a single open-source Rust workspace where Copy-on-Write (`CoW`) Serializable Snapshot Isolation (`SSI`) tables, PostgreSQL `pgwire` SQL, 5-node TCP Raft consensus, `10,240`-bit Hyperdimensional Computing (`Amorphic`) holograms, and sub-1-bit factorized sentence-embedding bitplanes (`.jsyn`) share one storage and energy-accounting substrate. Every energy figure is tagged by measurement provenance: **`reported_j`** from bare-metal OS hardware counters (Linux RAPL, macOS IOKit SMC) over multi-millisecond windows, or **`est_j`** from analytical cost models for sub-millisecond operations.

---

## 1. Core Subsystems in `v0.6.0` (`41` Workspace Crates)

1. **Storage, `SSI` Concurrency & Recovery (`joule-db-core`, `joule-db-local`):**
   - Lock-free positional `pread`/`pwrite`, `IoRingEngine` extent coalescing, Cahill `SSI` (`rw`-antidependency cycle prevention), `CRC-32` torn-WAL truncation, LSN-ordered replay, `ALP + FSST` columnar codecs, `RaBitQ` 1-bit `POPCNT` vector quantization, and double-entry `i128` ledger primitives (`tiger_ledger`).
   - **Compiler Safety Scope:** `#![deny(unsafe_code)]` in `joule-db-core` with **1** `#![allow(unsafe_code)]` module (`crates/joule-db-core/src/storage/mmap_extent.rs`); `deny(clippy::unwrap_used)` active in **3** crates (`joule-db-core`, `joule-db-branch`, `joule-db-ledger`), with `warn` across the remaining workspace crates.
2. **Query Execution, `pgwire` & Incremental Views (`joule-db-query`, `joule-db-server`):**
   - PostgreSQL v3 wire server (`pgwire` on `port + 2`), Parquet `COPY` loader, parallel morsel `columnar_exec` aggregate engine, `DBSP` `ZSet<T>` incremental view maintenance, `Arrow Flight SQL` (`JDB_IPC1`), `DuckLake` / `Iceberg v3` metadata structures, and experimental Cypher, Datalog, SPARQL, Gremlin, and `SigQL` biosignal frontends.
3. **5-Node Real TCP Raft Consensus (`raft.rs`, `raft_server.rs`, `raft_transport.rs`):**
   - Quorum replication via `RaftNode::propose_and_commit` (verifying `commit_index >= index` and Raft §5.4.2 current-term safety), `CRC-32` durable `raft.wal` persistence, and 2PC participant/coordinator recovery logs.
4. **`.jsyn` (`JSYN-v1`) Sub-1-Bit Weights & `Amorphic` Continuum (`joule-db-weights`, `joule-db-amorphic`, `joule-db-gpu`):**
   - Compiles `.safetensors` and `.gguf` checkpoints into packed `u64` sign bitplanes (`LittleBit-2 ⊕ Sakana SVF`) executed via CPU `T-MAC` byte lookup tables or `wgpu` `countOneBits` shaders (`0` PyTorch / `llama.cpp` / ONNX Runtime).
   - Dissolves heterogeneous JSON, CSV, XML, `k=v`, prose logs, and N-Triples into `10,240`-bit Kanerva VSA holograms (`StreamingAmorphicContinuum` + `MorphRule` SI unit normalization).
5. **Confinement, Branching & Energy Ledger (`joule-db-branch`, `joule-db-energy`, `joule-db-ledger`):**
   - $O(1)$ CoW workspace/table branching (`JouleAgentFS`, `JouleCapsule` daemonless OCI v2 layer CAS), `HMAC-SHA256` `ThermoCap` capability tokens, technical security-control primitives (`trust_compliance.rs`), and Merkle-anchored energy receipts (`reported_j` / `est_j`).

---

## 2. Reproducible External Evaluations (`bench/`)

### 2.1 Official ClickBench (`hits.parquet`, 105 Columns, 43 Queries) vs. DuckDB `2.0.1`

Evaluated via [`bench/clickbench_official.py`](../bench/clickbench_official.py) on `framework-128gb-0000` (`AMD Ryzen AI Max+ 395`, 32 threads, `128 GB` RAM, Linux `rapl-package` **`reported_j`**, `10 s` query timeout) over real `pgwire` (`psycopg 3.3.9`):

| Slice of Official `14.78 GB` `hits.parquet` | Load (`DuckDB` vs `JouleDB`) | JouleDB RSS | Queries `< 10s` | Exact Row Match | Faster (`JouleDB` / `DuckDB`) | Geomean Latency (`JouleDB / DuckDB`) | Geomean Energy (`reported_j`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`100,000` rows** | `0.192 s` vs `0.203 s` | `722.2 MB` | **`42 / 43`** (`0` panics) | `33 / 43` | `10` / `32` | **`26.37×` slower** (`median 17.03×`) | **`44.96×` more J** |
| **`1,000,000` rows** | `0.153 s` vs `1.125 s` | `4,177.2 MB` | **`23 / 43`** (`20` timeouts) | `18 / 23` | `2` / `21` | **`10.19×` slower** *(23 completed)* | **`12.76×` more J** *(23 completed)* |

### 2.2 5-Node Real TCP Raft Fault-Injection + Official Clojure `elle 0.2.8`

Evaluated via [`crates/joule-db-server/tests/raft_tcp_elle_fault_injection.rs`](../crates/joule-db-server/tests/raft_tcp_elle_fault_injection.rs) and [`bench/elle_check.clj`](../bench/elle_check.clj):
- **Faults Injected:** 5 `RaftNode<KvStateMachine, TcpRaftTransport>` servers across a 20-link socket-severing `TcpPartitionMatrix`; asymmetric `2/5` vs `3/5` split isolating initial leader -> minority write/quorum-read rejection -> `3/5` higher-term leader election -> follower stop + disk `raft.wal` restart -> partition heal & 5-node convergence (`x = [17; 5]`, `y = [18; 5]`, `lost_committed_writes = 0`).
- **Official `elle 0.2.8` Verdict:** **`{:valid? true}`** (`rw-register`, `:strict-serializable`, `76` events).

### 2.3 `.jsyn` Compression vs. Fidelity Boundaries

- **Verified High-Fidelity Encoder (`0.4378 BPW`):** [`dcharlot65-openie/all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit) (`Apache-2.0`, `1.95 MB`, `22.23×` smaller than `FP16`) achieves **`0.9441` mean sentence-embedding cosine similarity** against upstream `FP32`.
- **Degraded Ultra-Low-BPW Causal LLMs (`0.02–0.17 BPW`, Retracted):** Out-of-sample rank sweeps (`r = 8..128`) on `Gemma-4-E2B` (`q_proj`) and `Inkling-2-Mini` (`128`-Exp MoE) yield **`0.113–0.455` output cosine** (`KL = 2.59–3.87`, `65.6%` Top-5 agreement at `r = 128`). Because models at that fidelity do not behave like their upstream counterparts, all 81 unverified causal LLM checkpoints were deleted from Hugging Face in `v0.6.0`.

---

## 3. Explicit Engineering & Governance Boundaries

1. **Pre-1.0 Research Preview (`v0.6.0`):** In-memory Parquet column caching (`~4.18 KB/row` across 105 columns), row-AST fallback timeouts on complex analytical SQL at `1M+` rows, and an open `AVG` accumulator bug on `Q32` remain active engineering work items prior to any future `1.0` candidacy.
2. **No Third-Party Compliance Certifications:** `trust_compliance.rs` implements technical controls (SHA-256 hash-chained audit logs, WORM retention locks, per-subject DEK zeroization, Luhn redaction, `CycloneDX v1.6` SBOM, Bell-LaPadula lattice labels, and unvalidated Rust `ML-KEM-768`/`ML-DSA-65` signatures). **No SOC 2, ISO 27001, HIPAA, PCI-DSS, NIST CMVP, or FedRAMP audit has been performed.**
3. **In-Tree SDKs & Portable Binaries:** Prebuilt `v0.6.0` server binaries target macOS (`aarch64`/`x86_64`) and Linux (`GLIBC <= 2.35`, verified on Ubuntu 24.04 LTS & Debian 12). Client SDKs (`sdks/*`, `python/`) live in-tree in this repository and are not yet published to PyPI, npm, or NuGet.
