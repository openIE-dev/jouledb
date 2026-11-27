# JouleDB (`v0.15.0` — Pre-1.0 Research Preview)

**An experimental energy-metered multi-model database in Rust (`Apache-2.0 OR MIT`) unifying Core Banking (`128-bit` exact `NUMERIC`, pre-allocated binary `JWL1` group-committed double-entry `LEDGER` at `104,721 TPS` `pgwire` / `102,276 TPS` HTTP batch [`1.93×` vs PG16, `0.54 mJ/tx`], `3,545 TPS` `c=1` [`9.15×` vs PG16], `3,461 TPS` `c=16` [`1.00×` parity], SHA-256 journal proofs), Autonomous AI Agents (`0.605 ms` lazy `O(1)` CoW SQL branching, native `pgvector` HNSW SQL, 3-way conflict `MERGE`, blast-radius & microjoule guardrails, durable `WORKFLOW` checkpoints, `59`-tool Claude/MCP Apps Studio), and Human Developer/Lakehouse Workloads (`pgwire` `psql`/`psycopg3` [`12/12`]/`SQLAlchemy 2` [`4/4` zero-`xfail`]/`node-pg`/`JDBC`, `SELECT ... FOR UPDATE SKIP LOCKED`, `100M`-row ClickBench [`2.11×`], `22 / 22` `TPC-H` `SF=10.0` [`86.6M` rows in `758 MB` RSS, `4.01 GB` peak `VmHWM`], `5,154 TPS` NVMe OLTP, and 5-node WAN TCP Raft).**

[![Release](https://github.com/openIE-dev/jouledb/actions/workflows/release.yml/badge.svg)](https://github.com/openIE-dev/jouledb/releases)
[![Quality](https://github.com/openIE-dev/jouledb/actions/workflows/quality.yml/badge.svg)](https://github.com/openIE-dev/jouledb/actions/workflows/quality.yml)
[![Docs](https://github.com/openIE-dev/jouledb/actions/workflows/docs.yml/badge.svg)](https://openie-dev.github.io/jouledb)
[![crates.io](https://img.shields.io/crates/v/jouledb.svg)](https://crates.io/crates/jouledb)
[![Hugging Face Embedder](https://img.shields.io/badge/Hugging%20Face-MiniLM%20.jsyn%20(cos%200.944)-0D9488)](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit)
[![Website](https://img.shields.io/badge/Web-jouledb.org-2563EB)](https://jouledb.org)

> **Pre-1.0 Status & `v0.15.0` Release Summary (`2026-11-27`):** Building on our `v0.6.0` external-audit remediation (`v1.0.0` GA tag retracted/yanked, unrelated `1.11M`-LOC `joule-web` crate and synthetic `frontier_exam_routes.rs` deleted, 81 unverified low-BPW causal LLM exports pruned from Hugging Face) and `v0.7.0`–`v0.14.0` engine upgrades, **`v0.15.0`** delivers `>100k TPS` durable NVMe core-banking settlement, `<1 ms` lazy Copy-on-Write SQL branching, native PostgreSQL `pgvector` HNSW SQL, `3.88×` lower `SF=10.0` `TPC-H` peak memory, and a `100%` green `41`-crate (`758.4k` LOC, `12,289` tests, `100%` panic-linted) workspace:
> - **`104,721 TPS` Durable NVMe Core Banking `LEDGER` (`1.93×` Faster Batch, `9.15×` Faster `c=1` & `1.00×` `c=16` Parity vs `PostgreSQL 16.16`):** Pre-allocated `JWL1` WAL extents (`sync_data()` without `ext4` `jbd2` `i_size` commits), `O(1)` dense `Vec<LedgerAccount>`, streaming `Sha256`, `TCP_NODELAY`, and native `pgwire` `ledger_transfer()` / `ledger_transfer_batch()` reach **`104,721.1 TPS` (`0.54 mJ/tx`) over `pgwire` & `102,276.0 TPS` (`0.69 mJ/tx`) over HTTP on `batch=500` NVMe clearing** (`1.93×` faster than `PostgreSQL 16.16`'s `54,212.3 TPS`), **`3,544.6 TPS` (`25.10 mJ/tx`) at `c=1` (`9.15×` faster & `7.46×` less J/tx than PG16)**, and **`3,461.3 TPS` at `c=16` (`1.00×` parity/lead vs PG16 `3,454.1 TPS`, closing `v0.14.0`'s `2.05×` gap)** (`PostgreSQL 16.16` wins `1.14×` at `c=4` [`853.9` vs `749.4 TPS`]).
> - **Sub-Millisecond (`0.605 ms`) Lazy `O(1)` CoW SQL Branching + Native `pgvector` HNSW SQL (`sql_branch_sandbox.rs`, `query.rs`):** Lazy `OnceLock` executor & demand-driven table materialization cut pure `CREATE BRANCH` latency **`63×` (`38.28 ms -> 0.605 ms` Linux / `0.432 ms` macOS)** alongside full PostgreSQL `pgvector` syntax (`VECTOR(dim)`, `'...'::vector`, `<->` / `<=>` / `<#>`, `CREATE INDEX ... USING hnsw (col vector_cosine_ops)`), `40001`/`42501`/`53200` agent guardrails, and `WORKFLOW CHECKPOINT`/`RESUME` (`59` MCP tools).
> - **Pre-Allocated `WalManager` OLTP (`1.43×–1.90×` Faster `c=4..8`) & `3.88×` Lower `TPC-H` `SF=10.0` Peak `VmHWM` (`wal.rs`, `columnar_join.rs`):** Boosts `c=4..8` NVMe `pgwire` OLTP to `1,911.5–3,981.6 TPS` (`1.23×` Pipelined gap vs PG16 at `c=4..8`), slashes `TPC-H` `Q18` **`4.26×` at `SF=1.0` (`61.1 ms`) & `2.17×` at `SF=10.0` (`827.7 ms`)**, cuts `SF=10.0` peak `VmHWM` **`3.88×` (`15.59 GB -> 4.01 GB`, `8.17×` geomean)**, and passes **`0` `xfail`s across `psycopg3` (`12/12`), `SQLAlchemy 2` (`4/4`), `node-pg` (`5/5`), and `pgJDBC` (`7/7`)**. See [`CHANGELOG.md`](./CHANGELOG.md).

---

## What Is Real & Verified Today (`41` Workspace Crates · `758.4k` LOC · `12,289` Tests · `41 / 41` `deny(unwrap_used)`)

1. **Core Storage, Exact `Decimal128`, Pre-Allocated `JWL1` Group-Committed Banking Ledger & 64-Shard `SSI` (`joule-db-core`, `joule-db-local`, `joule-db-amorphic`, `joule-db-server`):**
   - Copy-on-Write B-tree pages, Cahill Serializable Snapshot Isolation (`SSI`), 64-shard `MvccStore` + `MvccTableStorage` (`JMVR\x01`, `JWB1`, `JWU1` delta NVMe WAL with pre-allocated `WalManager` extents), exact `128-bit` fixed-point `NUMERIC(38, s)` (`Decimal128`), SQL-native double-entry `FinancialLedger` (pre-allocated `JWL1` binary CRC-32 WAL, FIFO group-commit coalescer, `submit_transfer_batch_with_options`, `pgwire` `ledger_transfer()` / `ledger_transfer_batch()`, `i128` minor units, multi-leg conservation, two-phase holds, `idempotency_key`, SHA-256 `financial_ledger.jwal`, SQL virtual tables `jouledb_ledger_*`), `SELECT ... FOR UPDATE [NOWAIT | SKIP LOCKED]`, `ALP + FSST` codecs, and `RaBitQ` 1-bit `POPCNT` vectors.
2. **Sub-Millisecond Lazy CoW SQL Branching, Native `pgvector` HNSW SQL, Durable `WORKFLOW` Checkpoints & `59`-Tool Claude/MCP Apps Host (`sql_branch_sandbox.rs`, `mcp_app_ui.rs`, `mcp_transport.rs`):**
   - Sub-millisecond (`0.605 ms`) session-isolated SQL branches (`CREATE/USE/DIFF/MERGE BRANCH` with demand-driven table materialization & 3-way `40001` conflict checks), PostgreSQL `pgvector` SQL (`VECTOR(dim)`, `<->`/`<=>`/`<#>`, `USING hnsw`), strict agent blast-radius guardrails (`42501` DDL/unbounded-write blocks + `53200` microjoule caps), crash-proof `WORKFLOW CHECKPOINT`/`RESUME`, and native `SEP-1865` **JouleDB Studio** (`ui://jouledb/studio.html`) over `jouledb mcp` (`stdio`) and `POST /mcp` (`59` tools).
3. **Out-of-Core `.jcol` Columnar Analytics, Vectorized Multi-Table `TPC-H` Joins & `pgwire` (`joule-db-query`, `joule-db-server`):**
   - PostgreSQL v3 wire listener (`psql`, `psycopg3` `12/12` text & binary + Extended Pipeline, `SQLAlchemy 2` `4/4`, `node-pg` `5/5`, `JDBC` `7/7`, `GSSENCRequest`), owned-batch streaming Parquet `COPY` spilling `262,144`-row `.jcol` disk segments (`758 MB` post-load RSS & `4.01 GB` peak `VmHWM` at `86.6M` rows), `execute_segmented` (`43 / 43` Official ClickBench at `100M` rows), `columnar_join` (`22 / 22` Official `TPC-H` at `SF=0.1`, `1.0`, & `10.0` with top-down & cross-branch `KeyIn` semi-join pushdown + worker-folded `CompactNumAgg`), and `DBSP` `ZSet<T>` incremental views.
4. **5-Node Multi-Host WAN & Local TCP Raft Consensus (`raft.rs`, `raft_transport.rs`, `examples/raft_wan_node.rs`):**
   - Quorum write commit via `RaftNode::propose_and_commit` (§5.4.2), quorum linearizable reads via `RaftNode::read_linearizable` (§6.4 `ReadIndex` heartbeat quorum, zero WAL growth), `CRC-32` `raft.wal` crash recovery, and both multi-host Tailscale WAN (`27.58 ms` RTT + `SIGKILL`) and local 20-link fault-injection suites verified **`{:valid? true}`** by official Clojure `elle/elle 0.2.8`.
5. **Energy Telemetry & `.jsyn` Sub-1-Bit Sentence Embeddings (`joule-db-energy`, `joule-db-weights`):**
   - Bare-metal hardware counter readers (**`reported_j`** via Linux `intel-rapl` and macOS `IOKit SMC`) segregated from analytical cost models (**`est_j`**), plus pure-Rust [`dcharlot65-openie/all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit) (`1.95 MB`, `0.4378 BPW`, **`0.9441` cosine** vs `FP32`).

---

## Reproducible External Evaluations (`bench/`)

### 1. Full Official ClickBench (`hits.parquet`, `99,997,497` Rows, `105` Columns, `43` Queries) vs. DuckDB `2.0.1`

Executed via [`bench/clickbench_official.py`](./bench/clickbench_official.py) over real `pgwire` (`psycopg 3.3.9`) on `framework-128gb-0000` (`AMD Ryzen AI Max+ 395`, `128 GB` RAM, Linux `rapl-package` **`reported_j`**):

| Official `hits.parquet` Scale | Parquet Load (`DuckDB` vs `JouleDB`) | JouleDB Load RSS (`VmHWM` Peak) | Queries Succeeded | SQL-Valid Match (`Exact`) | Faster (`JouleDB` / `DuckDB`) | Geomean Latency (`JouleDB / DuckDB`) | Geomean Energy (`reported_j`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`100,000` rows** ([JSON](./bench/results/clickbench_official_100k.json)) | `0.179 s` vs `0.265 s` | `918 MB` (`2,652 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`34` exact) | **`20`** / `23` | **`1.47×` slower** (`median 1.19×`) | **`1.97×` more J** |
| **`1,000,000` rows** ([JSON](./bench/results/clickbench_official_1m.json)) | `0.158 s` vs `0.871 s` | `1,283 MB` (`1,923 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`29` exact) | **`11`** / `32` | **`2.34×` slower** (`median 2.95×`) | **`2.50×` more J** |
| **`10,000,000` rows** ([JSON](./bench/results/clickbench_official_10m.json)) | `2.422 s` vs `13.340 s` | `2,385 MB` (**`3,448 MB`**) | **`43 / 43`** (`0` err) | **`43 / 43`** (`31` exact) | **`9`** / `34` | **`2.28×` slower** (`median 2.63×`) | **`2.22×` more J** |
| **`99,997,497` rows — FULL `14.78 GB`** ([JSON](./bench/results/clickbench_official_100m.json)) | `13.866 s` vs `97.835 s` | **`5,902 MB` (`16,470 MB`)** | **`43 / 43`** (`0` err) | **`43 / 43`** (`32` exact) | **`7`** / `36` | **`2.11×` slower** (`median 3.39×`) | **`2.03×` more J** |

- **Candid Full-`100M` Summary:** Vectorized `.jcol` execution (`5.90 GB` post-load RSS; `10M` peak `VmHWM` `3.45 GB`) narrows the full `100M`-row geomean latency gap vs DuckDB `2.0.1` from `4.85×` (`v0.8.0`) -> `2.82×` (`v0.9.0`) -> **`2.11×`** (`2.03×` `reported_j`), beating DuckDB `2.0.1` on **`7 / 43` queries at `100M` rows** (`Q0` `1.91 ms`, `Q1` `20.23 ms`, `Q2` `2.41 ms` [`25.5×`], `Q3` `2.32 ms` [`22.4×`], `Q6` `1.58 ms` [`14.0×`], `Q29` `2.40 ms` [`23.4×`], `Q42` `22.78 ms`). DuckDB `2.0.1` wins **`36 / 43` queries (`2.11×` geomean, `7.06×` faster Parquet load)**.

### 2. Official 22-Query `TPC-H` Relational Suite (`tpch_queries()`, 8 Tables, `SF = 0.1`, `1.0` & `10.0`) vs. DuckDB `2.0.1`

Executed via [`bench/tpch_official.py`](./bench/tpch_official.py) over real TCP `pgwire` (`psycopg 3.3.9`, physical NVMe `--work-dir /var/tmp`, Linux `rapl-package` **`reported_j`**) across all **22 official `TPC-H` SQL queries (`Q01..Q22`)**:

| Official `TPC-H` Scale | 8-Table Parquet Load (`DuckDB` vs `JouleDB`) | Post-Load RSS (`VmHWM` Peak) | Exact Verified (`Faster`) | Total 22-Query Time (`DuckDB` vs `JouleDB`) | Geomean Latency (`JouleDB / DuckDB`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`SF = 0.1` (`866,602` rows)** ([JSON](./bench/results/tpch_official_sf01.json)) | `0.430 s` (`12.2 J`) vs `1.380 s` (`54.5 J`) | **`780.8 MB`** (`939.3 MB`) | **`22 / 22`** (**`5` faster**: `Q02,Q11,Q16,Q20,Q22`) | `88.2 ms` vs **`253.0 ms`** (`196.0×` vs `v0.10.0`) | **`1.99×` slower** (`median 2.16×`, down from `125.5×`) |
| **`SF = 1.0` (`8,661,245` rows)** ([JSON](./bench/results/tpch_official_sf1.json)) | `1.426 s` (`140.2 J`) vs `5.589 s` (`470.1 J`) | **`204.0 MB`** (**`574.7 MB`**) | **`22 / 22`** (`0` err, `0` timeouts) | `290.0 ms` vs **`1,587.2 ms`** (`1.76×` vs `v0.11.0`) | **`4.15×` slower** (`median 4.04×`, down from `6.18×`) |
| **`SF = 10.0` (`86,586,082` rows, `60M` `lineitem`)** ([JSON](./bench/results/tpch_official_sf10.json)) | `8.331 s` (`979.7 J`) vs `56.139 s` (`4,228.4 J`) | **`757.8 MB`** (**`4,012.8 MB`**) | **`22 / 22`** (`0` err, `0` timeouts) | `1,277.7 ms` vs **`16,197.7 ms`** (`2.10×` vs `v0.11.0`) | **`8.17×` slower** (`median 8.04×`, down from `16.03×`) |

- **Candid `TPC-H` Summary:** Worker-folded 48-byte `CompactNumAgg` (`Q18`), 64-shard `BorrowKeyElem` with deferred Top-K string materialization (`Q10`), and top-down + cross-branch `KeyIn` semi-join pushdown slash **`SF = 10.0` peak `VmHWM` `3.88×` (`15,586.4 MB -> 4,012.8 MB`, `10×` down from `v0.11.0`)** and **`SF = 1.0` peak `VmHWM` `2.72×` (`1,561.8 MB -> 574.7 MB`)**, cut **`Q18` `4.26×` at `SF=1.0` (`260.3 -> 61.1 ms`) & `2.17×` at `SF=10.0` (`1,799.8 -> 827.7 ms`, `5.56×` vs DuckDB down from `18.2×`)**, cut **`Q16` `1.59×` (`333.6 -> 209.6 ms`, `3.83×`)**, and narrow geomean vs DuckDB `2.0.1` to **`4.15×` (`SF=1.0`)** and **`8.17×` (`SF=10.0`)** (`5 / 22` wins at `SF=0.1`). DuckDB `2.0.1` remains faster at `SF=1.0..10.0` (widest `SF=10.0` tails: `Q10` `4.65 s` [`45.8×`], `Q21` `2.15 s` [`22.9×`], `Q09` `1.49 s` [`11.1×`]).

### 3. Head-to-Head `100`-Warehouse Physical-NVMe TCP `pgwire` OLTP vs. `PostgreSQL 16.16` & `SQLite 3.53.1`

Executed via [`bench/oltp_pgwire.py`](./bench/oltp_pgwire.py) ([`oltp_pgwire_100wh.json`](./bench/results/oltp_pgwire_100wh.json), physical NVMe `--work-dir /var/tmp`, `rapl-package` **`reported_j`**, `100,000` accounts across `100` warehouses, `50%` indexed point read, `30%` durable `BEGIN; UPDATE accounts ...; INSERT INTO history ...; COMMIT` [`fsync=on`, `synchronous_commit=on`], `20%` warehouse `COUNT`/`SUM`, `invariant_verified = true` & `0` errors across all 5 runs):

| Engine & Wire Protocol (`v0.15.0` Pre-Allocated `WalManager`, Shared Host `idle_watts = 87.32 W`) | `c = 1` TPS (`p50`) | `c = 4` TPS (`p50`, `p95`) | `c = 8` TPS (`p50`, `p95`) | `c = 16` TPS (`p50`, `mJ/txn`) | Committed Writes |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **JouleDB `0.15.0` (`pgwire` Simple Query, 4-RTT + NVMe `fdatasync`)** | `963.9`\* (`0.219 ms`) | **`1,911.5`** (`0.637 ms`, `7.44 ms`) | **`3,743.2`** (`0.677 ms`, `7.53 ms`) | **`5,139.3`** (`0.971 ms`, `22.05 mJ`) | `17,621` (`0` conflicts) |
| **JouleDB `0.15.0` (`pgwire` Extended Pipeline, 1-RTT + NVMe `fdatasync`)** | **`957.5`** (`0.191 ms`) | **`2,078.2`** (`0.638 ms`, `5.56 ms`) | **`3,981.6`** (`0.718 ms`, `6.11 ms`) | **`4,976.4`** (`1.229 ms`, `27.50 mJ`) | `17,945` (`0` conflicts) |
| **PostgreSQL `16.16` (`pgwire` Simple Query, 4-RTT + NVMe `fsync`)** | `1,166.2` (`0.109 ms`) | `2,567.4` (`0.110 ms`, `5.18 ms`) | `5,139.9` (`0.181 ms`, `5.15 ms`) | **`9,962.8`** (`0.312 ms`, `9.58 mJ`) | `28,082` (`0` conflicts) |
| **PostgreSQL `16.16` (`pgwire` Extended Pipeline, 1-RTT + NVMe `fsync`)** | `1,125.7` (`0.090 ms`) | `2,566.1` (`0.115 ms`, `5.23 ms`) | `4,891.8` (`0.174 ms`, `5.42 ms`) | **`9,392.5`** (`0.383 ms`, `10.20 mJ`) | `26,821` (`0` conflicts) |
| **SQLite `3.53.1` (Embedded in-proc C API, NVMe `WAL`, `NORMAL`)** | `38,080.4` (`0.006 ms`) | `74,892.8` (`0.007 ms`, `0.07 ms`) | `83,895.3` (`0.007 ms`, `0.07 ms`) | **`90,854.5`** (`0.008 ms`, `0.87 mJ`) | `444,174` (no TCP / `fsync`) |

- **Candid Physical-NVMe OLTP Summary:** Pre-allocating `WalManager` disk extents (`256 KiB -> 4 MiB -> 16 MiB` without `O_APPEND` `ext4` `jbd2` `i_size` commits) boosts **`c = 4` throughput `1.90×` Simple (`1,005.0 -> 1,911.5 TPS`, `p95` halved `15.04 -> 7.44 ms`) / `1.81×` Pipelined (`1,149.0 -> 2,078.2 TPS`, narrowing the `c=4` Pipelined gap vs `PostgreSQL 16.16` to `1.23×`)** and **`c = 8` throughput `1.43×` Simple (`2,617.9 -> 3,743.2 TPS`) / `1.46×` Pipelined (`2,730.9 -> 3,981.6 TPS`, `1.23×` gap vs PG16)**. \*On an uncontended host (`idle_watts = 10.68 W` vs `87.32 W` shared-host load above), JouleDB `c=1` Simple reaches **`1,950.5 TPS` (`1.63×` faster than `PostgreSQL 16.16`'s `1,199.7 TPS`)**; at `c=16`, `PostgreSQL 16.16` wins `1.89×–1.94×` via shared-buffer page locks.

### 4. 5-Node Multi-Host Cross-Machine WAN/Tailscale TCP Raft (`ReadIndex`) + Official Clojure `elle 0.2.8`

Executed via [`bench/raft_multihost_wan.py`](./bench/raft_multihost_wan.py), [`examples/raft_wan_node.rs`](./crates/joule-db-server/examples/raft_wan_node.rs), and [`bench/elle_check.clj`](./bench/elle_check.clj):
- **Cross-Machine WAN Topology & Nemeses:** 5 real OS `raft_wan_node` processes with durable `raft.wal` across **2 distinct physical Linux hosts** (`framework-128gb-0000` bare-metal + `openie-core-1` Cloud Debian 12 VM, **`27.58 ms` median Tailscale WAN TCP RTT**), cross-WAN `2/5` minority split isolating `n1`, `3/5` majority election (`n3`), real OS **`SIGKILL` (`kill -9`)** on `n5`, and `raft.wal` convergence (`commit=68, applied=68, x=68, y=39`) -> **[`{:valid? true}`](./bench/results/raft_wan_multihost_elle_verdict.txt)** (`270` WAN events, `:strict-serializable`).

### 5. `.jsyn` (`JSYN-v1`) Fidelity vs. Compression Curve

| Model / Layer | `BPW` | Shrink vs `FP16` | Measured Fidelity vs Upstream | Guidance |
| :--- | :---: | :---: | :---: | :--- |
| **[`all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit)** (`6L` Encoder) | `0.438` | `22.2×` (`1.95 MB`) | **`0.9441` embedding cosine** vs `FP32` | **Recommended** for semantic search & `TriModal RRF` |
| `Gemma-4-E2B` `q_proj` (`r = 8..128`) | `0.028–0.166` | `564×–96.6×` | `0.271–0.455` output cos (`KL 3.87–2.72`) | **Experimental only** — not a drop-in causal LLM (retracted from HF) |
| `Inkling-2-Mini` `w13` (`128`-Exp MoE, `r = 8..128`) | `0.023–0.134` | `701×–120×` | `0.113–0.350` output cos (`KL 2.91–2.59`) | **Experimental only** — not a drop-in causal LLM (retracted from HF) |

### 6. Head-to-Head Core Banking Double-Entry Ledger (`bench/ledger_banking_head_to_head.py`) & Unified Suite (`bench/unified_banking_agentic.py`)

Executed via [`bench/ledger_banking_head_to_head.py`](./bench/ledger_banking_head_to_head.py) ([`ledger_banking_nvme_rapl_v015.json`](./bench/results/ledger_banking_nvme_rapl_v015.json), physical NVMe `--work-dir /var/tmp`, Linux `rapl-package` **`reported_j`**, `1,000` `NO_OVERDRAFT` customer wallets, `5%` duplicate `idempotency_key` retries, SHA-256 digest per transfer, `fsync=on` & `synchronous_commit=on`, `550,418` HTTP & `542,954` `pgwire` SHA-256 chain entries verified `valid = true`):

| Core Banking Regime (`1,000` `NO_OVERDRAFT` Wallets, `fsync=on`) | JouleDB `0.15.0` HTTP (`JWL1` + SHA-256) | JouleDB `0.15.0` `pgwire` (`psycopg3` SQL) | PostgreSQL `16.16` (`FOR UPDATE` / `UNNEST`) | Speedup & RAPL Energy (`reported_j`) |
| :--- | :---: | :---: | :---: | :--- |
| **`c = 1` Serial Durable Transfer** | **`3,544.6 TPS`** (`p50 0.245 ms`, **`25.10 mJ/tx`**) | **`443.1 TPS`** (`p50 2.334 ms`, `151.45 mJ/tx`) | `387.2 TPS` (`p50 2.634 ms`, `187.12 mJ/tx`) | **JouleDB `9.15×` HTTP (`1.14×` `pgwire`) · `7.46×` less J/tx** |
| **`c = 4` Concurrent Group Commit** | `749.4 TPS` (`p50 5.179 ms`, `103.07 mJ/tx`) | `683.9 TPS` (`p50 5.242 ms`, `100.01 mJ/tx`) | **`853.9 TPS`** (`p50 4.932 ms`, `81.77 mJ/tx`) | **PostgreSQL `1.14×` faster** (shared-host `c=4` transition) |
| **`c = 16` High-Contention Concurrent** | **`3,461.3 TPS`** (`p50 4.572 ms`, `22.27 mJ/tx`) | `2,430.6 TPS` (`p50 6.957 ms`, `27.42 mJ/tx`) | `3,454.1 TPS` (`p50 4.351 ms`, **`19.42 mJ/tx`**) | **`1.00×` HTTP parity/lead (`2.03×` vs `v0.14`)** + global SHA-256 chain |
| **`batch = 500` Clearing Batch (`1` `fdatasync` / `500` tx)** | **`102,276.0 TPS`** (`p50 4.124 ms`, **`0.69 mJ/tx`**) | **`104,721.1 TPS`** (`p50 4.236 ms`, **`0.54 mJ/tx`**) | `54,212.3 TPS` (`p50 8.511 ms`, `0.97 mJ/tx`) | **JouleDB `1.93×` faster (`>100k TPS`) · `1.81×` less J/tx** |

- **Unified Banking, `<1 ms` Lazy CoW Branching, `pgvector` HNSW & Human ORM Verification ([`unified_banking_agentic_v015.json`](./bench/results/unified_banking_agentic_v015.json)):** Verifies `0.10 + 0.20 = 0.30` & `1,000 × 0.01 = 10` exact `Decimal128`, multi-leg split settlements, two-phase `HOLD`/`POST`, SQL `JOIN` (`kyc_customers ⋈ jouledb_ledger_accounts`), 500-op hot-wallet contention (`0` overdrafts), **`0.605 ms` median pure `CREATE BRANCH` (`0.432 ms` on Apple Silicon — `63×` faster than `v0.14.0`)**, `42501`/`40001`/`53200` guardrails, native `pgvector` HNSW cosine `<=>` top-K recall, `WORKFLOW CHECKPOINT` replay, `SELECT ... FOR UPDATE [NOWAIT | SKIP LOCKED]`, and **`0`-`xfail` `SQLAlchemy 2` (`4/4`) + `psycopg3` (`12/12`) + `node-pg` (`5/5`) + `pgJDBC` (`7/7`)**.

---

## Quickstart (`v0.15.0`)

```bash
# 1. Install prebuilt v0.15.0 binary (macOS ARM64/x86_64; Linux x86_64 GLIBC <= 2.35, Ubuntu 22.04/24.04 & Debian 12 ready)
cargo binstall jouledb

# 2. Launch JouleDB server (HTTP :8080, TCP :8081, PgWire :8082, JWP :8083, FlightSQL :8084, MCP HTTP /mcp)
jouledb --port 8080 --data ./joule-data

# 3. Or run as a Claude Desktop / MCP Apps stdio server (59 tools + interactive ui://jouledb/studio.html)
jouledb mcp --data ./joule-data

# 4. Connect with psql, SQLAlchemy 2, or run smoke/security self-tests
psql -h 127.0.0.1 -p 8082 -U postgres -c "SELECT 0.10 + 0.20 = 0.30, version();"
jouledb self-check && jouledb trust
```

---

## Honest Engineering Boundaries (Read Before Adopting)

- **Pre-1.0 research preview (`v0.15.0`), not a battle-tested production RDBMS:** While JouleDB `0.15.0` unifies exact `128-bit` `NUMERIC` + `104,721 TPS` (`pgwire`) / `102,276 TPS` (HTTP) durable NVMe `JWL1` SQL `LEDGER` (`1.93×` faster than `PostgreSQL 16.16` on `batch=500`, `9.15×` faster at `c=1`, `1.00×` parity at `c=16`) + `0.605 ms` lazy CoW SQL branching + native `pgvector` HNSW SQL + `43 / 43` Official ClickBench on `99,997,497` rows (`2.11×` geomean vs DuckDB `2.0.1`) + `22 / 22` Official `TPC-H` up to `SF=10.0` (`86.6M` rows in `758 MB` post-load RSS & `4.01 GB` peak `VmHWM`, `8.17×` geomean) + `5,154 TPS` physical-NVMe `pgwire` OLTP + 5-node cross-machine WAN Raft (`elle` `{:valid? true}`), mature single-purpose engines (`DuckDB 2.0.1` `2.11×` faster on `100M` ClickBench & `4.15×`–`8.17×` faster on `SF=1..10` `TPC-H`; `PostgreSQL 16.16` `1.23×–1.94×` faster on `c=4..16` general NVMe `pgwire` OLTP & `1.14×` faster at `c=4` ledger contention; dedicated non-SQL clustered `TigerBeetle` at `100k–1M+` fixed-struct ledger TPS; `SQLite` embedded WAL) remain faster on their specialized single axis.
- **No external compliance certifications:** `jouledb trust` exercises technical security primitives in code (`external_audit_performed = false`). **No third-party SOC 2, ISO 27001, HIPAA, PCI-DSS, NIST CMVP, or FedRAMP audit has been conducted.**
- **Exact compiler safety lint scope:** `#![deny(unsafe_code)]` is set in `joule-db-core` with **1** `#![allow(unsafe_code)]` module (`crates/joule-db-core/src/storage/mmap_extent.rs`). `#![cfg_attr(not(test), deny(clippy::unwrap_used))]` is enforced across **`41 / 41` workspace crates (`100%`, `0` `#[allow(clippy::unwrap_used)]` bypasses)** via a blocking CI gate.
- **SDK registry status:** Prebuilt `cp39-abi3` Python `.whl` wheels, TypeScript `openie-jouledb-0.15.0.tgz`, and Claude `.mcpb` bundles are published on [GitHub Releases (`v0.15.0`)](https://github.com/openIE-dev/jouledb/releases/tag/v0.15.0) (with automated OIDC/token publish steps wired in `.github/workflows/release.yml`); packages are **not yet published** on PyPI, npm, NuGet, or Maven Central pending registry token/OIDC registration.

---

## Documentation & Links

- **Website:** [https://jouledb.org](https://jouledb.org) ([Claude & MCP Apps](https://jouledb.org/claude) · [Benchmarks](https://jouledb.org/benchmarks) · [Honest Boundaries](https://jouledb.org/honest) · [Security Controls & Bridges](https://jouledb.org/trust) · [`.jsyn` Fidelity](https://jouledb.org/jsyn))
- **Online Book (`mdBook`):** [https://openie-dev.github.io/jouledb](https://openie-dev.github.io/jouledb) ([Claude & MCP Apps Guide](https://openie-dev.github.io/jouledb/claude-apps.html))
- **Verified `.jsyn` Sentence Embedder & Benchmark Dataset:** [https://huggingface.co/dcharlot65-openie](https://huggingface.co/dcharlot65-openie)
- **Architecture Whitepaper (`v0.15.0`):** [`docs/WHITEPAPER-JOULEDB-2026-11.md`](./docs/WHITEPAPER-JOULEDB-2026-11.md)
- **`.jsyn` Binary Specification:** [`docs/JSYN-LIVING-WEIGHTS-SPEC.md`](./docs/JSYN-LIVING-WEIGHTS-SPEC.md)
- **Benchmark Scripts & Raw JSON/EDN Outputs:** [`bench/`](./bench/)

## License

Copyright (c) 2026 Open Interface Engineering, Inc. Dual-licensed under either [Apache License, Version 2.0](./LICENSE-APACHE) or [MIT License](./LICENSE-MIT), at your option.
