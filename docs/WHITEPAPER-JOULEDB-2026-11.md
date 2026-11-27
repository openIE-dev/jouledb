# JouleDB: An Energy-Metered Multi-Model Database & Sub-1-Bit Weights Research Substrate

## Architecture & Empirical Status Paper — Version `0.17.0` (Pre-1.0 Research Preview, November 2026)
**Open Interface Engineering, Inc. (`OpenIE`)**  
**Website:** [https://jouledb.org](https://jouledb.org) | **Docs Book:** [https://openie-dev.github.io/jouledb](https://openie-dev.github.io/jouledb) | **GitHub:** [openIE-dev/jouledb](https://github.com/openIE-dev/jouledb) | **Hugging Face (`.jsyn` Embedder & Raw Benchmarks):** [huggingface.co/dcharlot65-openie](https://huggingface.co/dcharlot65-openie)  
**License:** `Apache-2.0 OR MIT`

> **Revision & Audit Errata Note (`v0.17.0`, 2026-11-27):** Following our `v0.6.0` external-audit remediation (which retracted the premature `v1.0.0` GA tag, removed the unrelated `1.11M`-line `joule-web` crate, deleted synthetic frontier AI exam endpoints, and retracted 81 unverified low-BPW causal LLM exports from Hugging Face) and `v0.7.0`–`v0.15.0` upgrades, **`v0.17.0`** describes the **`41`-crate (`759,870` LOC, `12,297` tests)** codebase with **`41 / 41` (`100%`, `0` bypasses) `deny(clippy::unwrap_used)` enforcement**, delivering: **(1) a `4 / 4` regime sweep over `PostgreSQL 16.16` on durable NVMe Core Banking `LEDGER`** via `32 MiB` pre-allocated `JWL1` extents, adaptive `target_wave` `Condvar` group-commit coalescing, and `pgwire` `block_in_place` (`3,424.5 TPS` [`9.21×`] at `c=1`, `1,412.7 TPS` [`1.82×`] at `c=4`, `3,919.0 TPS` [`1.37×`] at `c=16`, and `91,316.1 TPS` [`1.69×`; `104.7k` peak] at `batch=500`), **(2) `2.18×` faster out-of-core `TPC-H` `SF=10.0` (`7.44 s`, `6.41×` geomean, `Q10: 342.8 ms` [`13.57×` faster], `Q21: 573.2 ms` [`3.75×` faster]) & sub-second `SF=1.0` (`927.5 ms`, `3.61×` geomean)** via `PK-FD` group-by key compression & 3-hop semi-join pushdown, **(3) `42.4×` lower post-load RSS (`139.2 MB`, `9.45 GB` `VmHWM`), `8 / 43` wins, & `3.05×–4.73×` `Q39`/`Q32`/`Q30`/`Q7` tail cuts on full `100M`-row ClickBench**, **(4) decoupled lock-free `WalManager::log_atomic_tx_fast` NVMe `pgwire` OLTP** (`3,698.0 TPS` [`3.25×` vs PG16] at `c=1`, `5,069.3 TPS` [`1.94×` vs PG16] at `c=4`, `2,486.1 TPS` [`1.04×` vs PG16] at `c=4` Pipelined, `6,364.8 TPS` at `c=16` Pipelined), **(5) `0.553 ms` lazy CoW `CREATE BRANCH` + native `pgvector` HNSW SQL**, and **(6) the `LTCB` (`enwik8`/`enwik9`) `Joules-per-Bit` & Seekable Database Pareto Suite** (`2.83 nJ/bit` & `5.41 ms` SQL lookup vs `tufazip v2`'s `~157.5M nJ/bit` & `7.29-day` sequential decode).

---

## Abstract

Modern data stacks fracture across three incompatible silos — **Agent-native sandboxes**, **Core Banking vaults**, and **Human developer/analyst RDBMSs** — while archival AI compression benchmarks optimize static bits-per-byte (`bpb`) at the cost of megajoules/gigajoules per gigabyte and zero random-access queryability. **JouleDB (`v0.17.0`)** explores unifying transactional, analytical, and agentic workflows inside a single open-source Rust workspace where exact `128-bit` `NUMERIC(38, s)` & adaptive group-committed `JWL1` double-entry accounting, `<1 ms` (`0.553 ms`) lazy Copy-on-Write SQL branches & durable `WORKFLOW` checkpoints, native `pgvector` HNSW SQL, out-of-core `.jcol` vectorized columnar scans & `PK-FD` Snowflake multi-table hash joins, PostgreSQL `pgwire` SQL, multi-host WAN TCP Raft `ReadIndex` consensus, `SEP-1865` MCP Apps interactive UI, `10,240`-bit `Amorphic` VSA holograms, and `.jsyn` sentence-embedding bitplanes share one storage and hardware energy-accounting substrate (`reported_j` vs `est_j`).

---

## 1. Core Subsystems in `v0.17.0` (`41` Workspace Crates)

1. **Storage, Decoupled `WalManager::log_atomic_tx_fast`, Adaptive `JWL1` Group Commit & 64-Shard `SSI` (`joule-db-core`, `joule-db-local`, `joule-db-amorphic`, `joule-db-server`):**
   - Lock-free positional `pread`/`pwrite`, `32 MiB` pre-allocated WAL extents (`sync_data()` without `ext4` `jbd2` `i_size` commits), `Arc<AtomicU64>` lock-free ID reservation, zero-clone `encode_single_row` (`JWB1`), WAL encoding & `fdatasync` decoupled completely outside `AmorphicStore`'s `RwLockWriteGuard`, `FinancialLedger` (`(Mutex<GroupCommitQueue>, Condvar)` adaptive `target_wave` coalescer + `pgwire` `block_in_place`), `ALP + FSST` codecs, and `RaBitQ` 1-bit `POPCNT` vectors.
   - **Compiler Safety Scope:** `#![deny(unsafe_code)]` in `joule-db-core` with **1** `#![allow(unsafe_code)]` module (`crates/joule-db-core/src/storage/mmap_extent.rs`); `#![cfg_attr(not(test), deny(clippy::unwrap_used))]` enforced in CI across **all `41 / 41` workspace crates (`100%`, `0` bypasses)**.
2. **Out-of-Core `.jcol` Vectorized Execution, `PK-FD` Multi-Table `TPC-H` Joins, `pgvector` HNSW & `pgwire` (`joule-db-query`, `joule-db-server`):**
   - PostgreSQL v3 wire server (`block_in_place`, `TCP_NODELAY`, Simple & Extended Pipeline modes, `VECTOR(dim)` + `<->`/`<=>`/`<#>` + `USING hnsw`), streaming Parquet `COPY` spilling `262,144`-row `.jcol` disk segments (`139.2 MB` post-load RSS on `100M`-row ClickBench; `754.9 MB` post-load RSS & `4.26 GB` peak `VmHWM` on `86.6M`-row `TPC-H` `SF=10.0`), `execute_segmented` (zero-`Arc<str>` `eval_raw_text_filter`, stack-inline `HKey::K4/K5` + `GKey::IIISS`, `128`-shard radix aggregation), and `columnar_join` (`ColRel::col_is_unique_nonnull_int` `PK-FD` group-by key compression + 20-row late materialization, 3-hop `KeyIn` semi-join pushdown, 64-shard `NeqExists`, `CompactNumAgg`).
3. **Sub-Millisecond (`0.553 ms`) Lazy CoW SQL Branching & Claude/MCP Apps (`SEP-1865`) Host (`sql_branch_sandbox.rs`, `mcp_app_ui.rs`, `mcp_transport.rs`):**
   - `0.553 ms` lazy `CREATE BRANCH`, 3-way `40001` conflict `MERGE`, `42501`/`53200` guardrails, `WORKFLOW CHECKPOINT`/`RESUME`, `59` MCP tools, and interactive zero-CDN **JouleDB Studio** (`ui://jouledb/studio.html`, `.mcpb` bundles).
4. **5-Node Multi-Host WAN & Local TCP Raft Consensus (`raft.rs`, `raft_transport.rs`, `examples/raft_wan_node.rs`):**
   - Quorum write replication (`propose_and_commit` §5.4.2), quorum linearizable reads (`read_linearizable` §6.4 `ReadIndex`), and `CRC-32` `raft.wal` recovery.
5. **`.jsyn` (`JSYN-v1`) Sub-1-Bit Weights, `Amorphic` Continuum & Energy Ledger (`joule-db-weights`, `joule-db-amorphic`, `joule-db-energy`, `joule-db-ledger`):**
   - Pure-Rust `LittleBit-2 ⊕ Sakana SVF` bitplanes, `10,240`-bit Kanerva VSA `Amorphic` holograms, and Merkle-anchored energy receipts (`reported_j` / `est_j`).

---

## 2. Reproducible External Evaluations (`bench/`)

### 2.1 Full Official ClickBench (`hits.parquet`, `99,997,497` Rows, `105` Columns, `43` Queries) vs. DuckDB `2.0.1`

Evaluated via [`bench/clickbench_official.py`](../bench/clickbench_official.py) (`--data-parent-dir /var/tmp` physical NVMe on `100M`) on `framework-128gb-0000` (`AMD Ryzen AI Max+ 395`, 32 threads, `128 GB` RAM, Linux `rapl-package` **`reported_j`**) over real `pgwire` (`psycopg 3.3.9`):

| Official `14.78 GB` `hits.parquet` Scale | Load (`DuckDB` vs `JouleDB`) | JouleDB Post-Load RSS (`VmHWM` Peak) | Queries OK | SQL-Valid (`Exact`) | Faster (`JDB` / `DuckDB`) | Geomean Latency (`JouleDB / DuckDB`) | Geomean Energy (`reported_j`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`100,000` rows** | `0.179 s` vs `0.265 s` | `917.8 MB` (`2,651.5 MB`) | **`43 / 43`** | **`43 / 43`** (`34`) | **`20`** / `23` | **`1.47×` slower** (`median 1.19×`) | **`1.97×` more J** |
| **`1,000,000` rows** | `0.158 s` vs `0.871 s` | `1,282.9 MB` (`1,922.8 MB`) | **`43 / 43`** | **`43 / 43`** (`29`) | **`11`** / `32` | **`2.34×` slower** (`median 2.95×`) | **`2.50×` more J** |
| **`10,000,000` rows** | `2.422 s` vs `13.340 s` | `2,384.7 MB` (**`3,448.4 MB`**) | **`43 / 43`** | **`43 / 43`** (`31`) | **`9`** / `34` | **`2.28×` slower** (`median 2.63×`) | **`2.22×` more J** |
| **`99,997,497` rows (FULL `100M` NVMe)** | `18.208 s` vs `161.403 s` | **`139.2 MB` (`9,454.5 MB`)** | **`43 / 43`** | **`43 / 43`** (`30`) | **`8`** / `35` | **`2.35×` slower NVMe** (`2.11×` `tmpfs`) | **`2.09×` more J** |

### 2.2 Official 22-Query `TPC-H` Relational Suite (`SF = 0.1`, `1.0` & `10.0`) vs. DuckDB `2.0.1`

Evaluated via [`bench/tpch_official.py`](../bench/tpch_official.py) ([`tpch_official_sf01.json`](../bench/results/tpch_official_sf01.json), [`tpch_official_sf1.json`](../bench/results/tpch_official_sf1.json), [`tpch_official_sf10.json`](../bench/results/tpch_official_sf10.json)) over TCP `pgwire` (`--work-dir /var/tmp` NVMe, `rapl-package` **`reported_j`**):
- **`SF = 0.1` (`866,602` rows):** **`22 / 22` exact in `253.0 ms`** vs DuckDB `88.2 ms` (**`1.99×` geomean, `5 / 22` wins**).
- **`SF = 1.0` (`8,661,245` rows):** Load `4.707 s` (`143.8 J`, **`201.6 MB` post-load RSS, `578.5 MB` peak `VmHWM`**) vs DuckDB `1.200 s` (`65.0 J`); **`22 / 22` exact in `927.50 ms` (`0.93 s` sub-second suite, `1.71×` faster than `v0.15.0`)** vs DuckDB `238.61 ms` (**`3.61×` geomean, `median 3.59×`**; **`Q10` slashed `8.02×` [`399.22 -> 49.78 ms`, `2.20×`]**, **`Q21` slashed `3.15×` [`447.5 -> 142.00 ms`]**, `Q16: 42.15 ms`, `Q18: 60.98 ms`).
- **`SF = 10.0` (`86,586,082` rows, `59,986,052` `lineitem` out-of-core `.jcol`):** Load `46.135 s` (`1,373.0 J`, **`754.9 MB` post-load RSS, `4,259.3 MB` peak `VmHWM`**) vs DuckDB `5.839 s` (`508.5 J`); **`22 / 22` exact in `7.44 s` (`7,440.97 ms` — `2.18×` faster than `v0.15.0`'s `16.20 s` & `4.58×` faster than `v0.11.0`)** vs DuckDB `945.61 ms` (**`6.41×` geomean, `median 6.51×`**; **`Q10` slashed `13.57×` [`4,652.99 -> 342.78 ms`, `4.44×` vs DuckDB]**, **`Q21` slashed `3.75×` [`2,147.9 -> 573.18 ms`]**, **`Q18` `621.63 ms`**, **`Q16` `153.09 ms` [`3.64×`]**).

### 2.3 Head-to-Head `100`-Warehouse Physical-NVMe TCP `pgwire` OLTP (`100,000` Accounts) vs. `PostgreSQL 16.16` & `SQLite 3.53.1`

Evaluated via [`bench/oltp_pgwire.py`](../bench/oltp_pgwire.py) ([`oltp_pgwire_100wh.json`](../bench/results/oltp_pgwire_100wh.json), physical NVMe `--work-dir /var/tmp`, `idle_watts = 10.03 W`, `invariant_verified = true` across all 5 runs):
- **JouleDB `0.17.0` (`pgwire` Simple Query, 4-RTT + Decoupled `log_atomic_tx_fast`):** **`3,697.98 TPS` (`c=1`, `p50 0.137 ms`, `9.42 mJ/txn` — `3.25×` faster than `PostgreSQL 16.16`)** -> **`5,069.30 TPS` (`c=4`, `p95 1.904 ms` — `1.94×` faster than `PostgreSQL 16.16`)** -> `4,094.67 TPS` (`c=8`) -> **`5,474.86 TPS` (`c=16`, `43,783` writes)**.
- **JouleDB `0.17.0` (`pgwire` Extended Pipeline, 1-RTT + Decoupled `log_atomic_tx_fast`):** `1,083.79 TPS` (`c=1`) -> **`2,486.05 TPS` (`c=4`, `p95 5.064 ms` — `1.04×` faster than `PostgreSQL 16.16`)** -> `4,342.65 TPS` (`c=8`) -> **`6,364.75 TPS` (`c=16`, `9.57 mJ/txn`, `34,071` writes — `+28%` over `v0.15.0`)**.
- **PostgreSQL `16.16` (`fsync=on`, `synchronous_commit=on`):** `1,139.42 -> 2,614.86 -> 4,901.20 -> 10,081.67 TPS` Simple; `1,161.69 -> 2,386.67 -> 5,275.86 -> 10,205.75 TPS` Pipelined (`1.20×–1.84×` ahead at `c=8..16`).
- **Embedded `SQLite 3.53.1` (`WAL`, `NORMAL`):** `44,204.91 -> 111,079.64 TPS` (`0.36 mJ/txn`, in-process).

### 2.3b Head-to-Head Core Banking Double-Entry Ledger (`4 / 4` Regime Sweep vs. `PostgreSQL 16.16`)

Evaluated via [`bench/ledger_banking_head_to_head.py`](../bench/ledger_banking_head_to_head.py) ([`ledger_banking_nvme_rapl_v016.json`](../bench/results/ledger_banking_nvme_rapl_v016.json), `--work-dir /var/tmp` NVMe, `rapl-package` **`reported_j`**, `493,941` HTTP & `436,801` `pgwire` SHA-256 chain entries verified `valid = true`):
- **`c = 1`:** **JouleDB `0.17.0` HTTP `3,424.5 TPS`** (`p50 0.242 ms`, **`26.96 mJ/tx`**) & **`pgwire` `400.6 TPS`** vs **`PostgreSQL 16.16` `371.9 TPS`** (`129.73 mJ/tx`) — **`9.21×` faster HTTP (`1.08×` `pgwire`) & `4.81×` less J/tx**.
- **`c = 4`:** **JouleDB `0.17.0` `pgwire` `1,412.7 TPS`** (`p50 2.761 ms`, **`56.08 mJ/tx`**) & **HTTP `1,250.0 TPS`** vs **`PostgreSQL 16.16` `777.6 TPS`** (`p50 5.271 ms`, `61.57 mJ/tx`) — **`1.82×` faster `pgwire` (`1.61×` HTTP) & `1.10×` less J/tx**.
- **`c = 16`:** **JouleDB `0.17.0` `pgwire` `3,919.0 TPS`** (`p50 3.595 ms`, `22.56 mJ/tx`; HTTP `2,636.9 TPS`) vs **`PostgreSQL 16.16` `2,861.0 TPS`** (`p50 5.658 ms`, `20.51 mJ/tx`) — **`1.37×` faster over identical `psycopg3` TCP `pgwire`**.
- **`batch = 500`:** **JouleDB `0.17.0` HTTP `91,316.1 TPS`** & **`pgwire` `81,453.5 TPS`** (`104,721.1 TPS` peak) vs **`PostgreSQL 16.16` `53,956.4 TPS`** — **`1.69×` faster**.

### 2.4 5-Node Multi-Host Cross-Machine WAN/Tailscale TCP Raft (`ReadIndex`) + Official Clojure `elle 0.2.8`

Evaluated via [`bench/raft_multihost_wan.py`](../bench/raft_multihost_wan.py) and [`bench/elle_check.clj`](../bench/elle_check.clj): 5 OS `raft_wan_node` processes across 2 Linux hosts (`27.58 ms` Tailscale WAN RTT + `2/5` split + `SIGKILL` + `raft.wal` recovery) -> **`{:valid? true}`** (`270` WAN events).

### 2.5 `.jsyn` Compression vs. Fidelity & Large Text Compression Benchmark (`enwik8`/`enwik9` `Joules-per-Bit` Pareto)

- **Verified High-Fidelity Encoder (`0.4378 BPW`):** [`dcharlot65-openie/all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit) (`1.95 MB`, `22.23×` smaller than `FP16`) achieves **`0.9441` mean sentence-embedding cosine similarity** against upstream `FP32` (81 low-BPW causal LLM exports remain retracted).
- **Large Text Compression Benchmark (`enwik8` Live RAPL + `enwik9` Published Neural Anchors, [`ltcb_energy_pareto_v016.json`](../bench/results/ltcb_energy_pareto_v016.json)):** While online-trained Transformer arithmetic coders (`tufazip v2`: `105.39 MB` = `0.8431 bpb` on `1 GB` `enwik9`; `cmix-lex-transformer`: `95.84 MB` = `0.7667 bpb`; `zmix 1.0`: `96.10 MB` = `0.7688 bpb`) achieve `9.49×–10.43×` static archival compression, `tufazip v2` requires `630,136 s` (`7.29 days` on dual B200 GPUs, `~1.26 GJ` = `~157.5M nJ/bit` `analytical_est`) with **zero random-access seekability**. Applying Ontology+SimHash semantic locality ordering (`1.46%–1.68%` smaller block windows) with seekable `Parquet Zstd` (`2.870 bpb`) and JouleDB `.jcol` SQL yields **`28.36 ms` (`3.374 GB/s`, `2.16 J` = `2.83 nJ/bit` `reported_j` — `55.7M×` lower energy/bit) full-corpus `LIKE` scans** and **`5.41 ms` (`116.4M×` faster) single-article SQL lookups**.

---

## 3. Explicit Engineering & Governance Boundaries

1. **Pre-1.0 Research Preview (`v0.17.0`):** `41 / 41` (`100%`) `deny(clippy::unwrap_used)` crates; specialized single-axis engines (`DuckDB 2.0.1` `2.11×–2.35×` faster on `100M` ClickBench & `3.61×–6.41×` faster on `SF=1..10` `TPC-H`; `PostgreSQL 16.16` `1.20×–1.84×` faster on `c=8..16` general NVMe `pgwire` OLTP; `SQLite` embedded WAL; non-seekable `cmix`/`zmix`/`tufazip` archival codecs) remain ahead on their narrow single axis.
2. **No Third-Party Compliance Certifications:** `external_audit_performed = false`. **No SOC 2, ISO 27001, HIPAA, PCI-DSS, NIST CMVP, or FedRAMP audit has been performed.**
3. **SDK Artifacts & Portable Binaries:** Prebuilt `v0.17.0` server archives, Claude `.mcpb` bundles, Python `cp39-abi3` `.whl` wheels, and TypeScript `openie-jouledb-0.17.0.tgz` (+ GitHub Packages npm) ship on GitHub Releases (`GLIBC <= 2.35` on Linux); client SDKs are not yet published on public PyPI, npmjs, or NuGet pending upstream OIDC registration.
