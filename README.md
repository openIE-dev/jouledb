# JouleDB (`v0.16.0` — Pre-1.0 Research Preview)

**An experimental energy-metered multi-model database in Rust (`Apache-2.0 OR MIT`) unifying Core Banking (`128-bit` exact `NUMERIC`, pre-allocated binary `JWL1` adaptive group-committed double-entry `LEDGER` beating `PostgreSQL 16.16` across all `4 / 4` NVMe regimes: `3,425 TPS` `c=1` [`9.21×`], `1,413 TPS` `c=4` [`1.82×`], `3,919 TPS` `c=16` [`1.37×`], & `91.3k–104.7k TPS` `batch=500` [`1.69×–1.93×`] + SHA-256 journal proofs), Autonomous AI Agents (`0.553 ms` lazy `O(1)` CoW SQL branching, native `pgvector` HNSW SQL, 3-way conflict `MERGE`, blast-radius & microjoule guardrails, durable `WORKFLOW` checkpoints, `59`-tool Claude/MCP Apps Studio), and Human Developer/Lakehouse Workloads (`pgwire` `psql`/`psycopg3` [`12/12`]/`SQLAlchemy 2` [`4/4` zero-`xfail`]/`node-pg`/`JDBC`, `SELECT ... FOR UPDATE SKIP LOCKED`, `100M`-row ClickBench [`139.2 MB` post-load RSS, `8 / 43` wins], `22 / 22` `TPC-H` `SF=1.0` in `0.93 s` [`3.61×`] & `SF=10.0` in `7.44 s` [`86.6M` rows in `755 MB` RSS, `6.41×`], `3,698 TPS` `c=1` [`3.25×` vs PG16] / `6,365 TPS` `c=16` NVMe OLTP, and 5-node WAN TCP Raft).**

[![Release](https://github.com/openIE-dev/jouledb/actions/workflows/release.yml/badge.svg)](https://github.com/openIE-dev/jouledb/releases)
[![Quality](https://github.com/openIE-dev/jouledb/actions/workflows/quality.yml/badge.svg)](https://github.com/openIE-dev/jouledb/actions/workflows/quality.yml)
[![Docs](https://github.com/openIE-dev/jouledb/actions/workflows/docs.yml/badge.svg)](https://openie-dev.github.io/jouledb)
[![crates.io](https://img.shields.io/crates/v/jouledb.svg)](https://crates.io/crates/jouledb)
[![Hugging Face Embedder](https://img.shields.io/badge/Hugging%20Face-MiniLM%20.jsyn%20(cos%200.944)-0D9488)](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit)
[![Website](https://img.shields.io/badge/Web-jouledb.org-2563EB)](https://jouledb.org)

> **Pre-1.0 Status & `v0.16.0` Release Summary (`2026-11-27`):** Building on our `v0.6.0` external-audit remediation (`v1.0.0` GA tag retracted/yanked, unrelated `1.11M`-LOC `joule-web` crate and synthetic `frontier_exam_routes.rs` deleted, 81 unverified low-BPW causal LLM exports pruned from Hugging Face) and `v0.7.0`–`v0.15.0` upgrades, **`v0.16.0`** achieves a `4 / 4` clean sweep over `PostgreSQL 16.16` on durable NVMe core banking, cuts `SF=10.0` `TPC-H` `2.18×` (`Q10` `13.57×` faster), slashes `100M` ClickBench post-load RSS `42.4×` (`139.2 MB`) & `Q39`/`Q32` `3.05×–4.04×`, boosts `c=1..4` NVMe `pgwire` OLTP up to `3.25×` past `PostgreSQL 16.16`, and adds our `LTCB` (`enwik8`/`enwik9`) `Joules-per-Bit` & seekable compression benchmark across a `100%` panic-linted `41`-crate (`759.9k` LOC, `12,297` tests) workspace:
> - **`4 / 4` Regime Clean Sweep vs `PostgreSQL 16.16` on Durable NVMe Core Banking `LEDGER` (`financial_ledger.rs`, `pgwire.rs`):** Adaptive `target_wave` `Condvar` coalescing (`0 ns` wait at `c=1`) + `pgwire` `block_in_place` + `32 MiB` pre-allocated `JWL1` WAL extents beat `PostgreSQL 16.16` (`fsync=on`) across **every** regime: **`c=1` `3,424.5 TPS` (`9.21×` faster, `4.81×` less J/tx)**, **`c=4` `1,412.7 TPS` `pgwire` / `1,250.0 TPS` HTTP (`1.82×` / `1.61×` faster — flipping `v0.15.0`'s `c=4` loss)**, **`c=16` `3,919.0 TPS` `pgwire` (`1.37×` faster over identical `psycopg3` TCP)**, and **`batch=500` `91,316.1 TPS` (`1.69×` faster; `104,721 TPS` peak)** with `930k+` SHA-256 chain entries verified `valid = true`.
> - **`2.18×` Faster `TPC-H` `SF=10.0` (`Q10` `13.57×` Faster!) & `42.4×` Lower `100M` ClickBench Post-Load RSS (`columnar_join.rs`, `columnar_exec.rs`):** Primary-Key Functional-Dependency (`PK-FD`) group-by compression + 3-hop semi-join pushdown slash **`TPC-H` `SF=1.0` to `927.5 ms` (`3.61×` geomean, `Q10: 49.8 ms` [`8.02×` faster])** and **`SF=10.0` to `7.44 s` (`6.41×` geomean, `Q10: 342.8 ms` [`13.57×` faster], `Q21: 573.2 ms` [`3.75×` faster])**, while zero-`Arc<str>` `.jcol` filters & inline `HKey::K4/K5` cut **`100M` ClickBench post-load RSS to `139.2 MB` (`9.45 GB` `VmHWM`), win `8 / 43` queries vs DuckDB `2.0.1`**, and slash **`Q39` `3.05×` (`297.0 ms`) & `Q32` `4.04×` (`1,467.5 ms`)**.
> - **Decoupled Lock-Free `WalManager` OLTP (`3.25×` Faster `c=1`, `1.94×` Faster `c=4`, `6,365 TPS` `c=16`) + `LTCB` `Joules-per-Bit` Pareto Suite (`wal.rs`, `ltcb_energy_pareto.py`):** Moving WAL serialization & `fdatasync` outside `AmorphicStore`'s write lock lifts `100`-warehouse NVMe `pgwire` OLTP to **`3,698.0 TPS` (`c=1` Simple, `3.25×` vs PG16)**, **`5,069.3 TPS` (`c=4` Simple, `1.94×` vs PG16)**, **`2,486.1 TPS` (`c=4` Pipelined, `1.04×` vs PG16)**, and **`6,364.8 TPS` (`c=16` Pipelined)**, alongside **`0.553 ms` `CREATE BRANCH`** and live RAPL `enwik8`/`enwik9` `Joules-per-Bit` measurements (`2.83 nJ/bit` & `5.41 ms` SQL seekability vs `tufazip v2`'s `~157.5M nJ/bit` & `7.29-day` sequential decode). See [`CHANGELOG.md`](./CHANGELOG.md).

---

## What Is Real & Verified Today (`41` Workspace Crates · `759.9k` LOC · `12,297` Tests · `41 / 41` `deny(unwrap_used)`)

1. **Core Storage, Exact `Decimal128`, Adaptive Group-Committed `JWL1` Banking Ledger & 64-Shard `SSI` (`joule-db-core`, `joule-db-local`, `joule-db-amorphic`, `joule-db-server`):**
   - Copy-on-Write B-tree pages, Cahill Serializable Snapshot Isolation (`SSI`), 64-shard `MvccStore` + `MvccTableStorage` (`JMVR\x01`, zero-clone single-row `JWB1`, `JWU1` delta NVMe WAL with `WalManager::log_atomic_tx_fast` decoupled outside `RwLockWriteGuard`), exact `128-bit` fixed-point `NUMERIC(38, s)` (`Decimal128`), SQL-native double-entry `FinancialLedger` (`32 MiB` pre-allocated `JWL1` binary CRC-32 WAL, adaptive `target_wave` `Condvar` coalescer, `pgwire` `ledger_transfer()` / `ledger_transfer_batch()`, `i128` minor units, multi-leg conservation, two-phase holds, `idempotency_key`, SHA-256 `financial_ledger.jwal`, virtual tables `jouledb_ledger_*`), `SELECT ... FOR UPDATE [NOWAIT | SKIP LOCKED]`, `ALP + FSST` codecs, and `RaBitQ` 1-bit `POPCNT` vectors.
2. **Sub-Millisecond (`0.553 ms`) Lazy CoW SQL Branching, Native `pgvector` HNSW SQL, Durable `WORKFLOW` Checkpoints & `59`-Tool Claude/MCP Apps Host (`sql_branch_sandbox.rs`, `mcp_app_ui.rs`, `mcp_transport.rs`):**
   - Sub-millisecond (`0.553 ms` Linux / `0.432 ms` macOS) session-isolated SQL branches (`CREATE/USE/DIFF/MERGE BRANCH` with demand-driven table materialization & 3-way `40001` conflict checks), PostgreSQL `pgvector` SQL (`VECTOR(dim)`, `<->`/`<=>`/`<#>`, `USING hnsw`), strict agent blast-radius guardrails (`42501` DDL/unbounded-write blocks + `53200` microjoule caps), crash-proof `WORKFLOW CHECKPOINT`/`RESUME`, and native `SEP-1865` **JouleDB Studio** (`ui://jouledb/studio.html`) over `jouledb mcp` (`stdio`) and `POST /mcp` (`59` tools).
3. **Out-of-Core `.jcol` Columnar Analytics, `PK-FD` Multi-Table `TPC-H` Joins & `pgwire` (`joule-db-query`, `joule-db-server`):**
   - PostgreSQL v3 wire listener (`block_in_place`, `psql`, `psycopg3` `12/12`, `SQLAlchemy 2` `4/4`, `node-pg` `5/5`, `JDBC` `7/7`), owned-batch streaming Parquet `COPY` spilling `262,144`-row `.jcol` disk segments (`139.2 MB` post-load RSS at `100M` ClickBench rows; `754.9 MB` post-load RSS & `4.26 GB` peak `VmHWM` at `86.6M` `TPC-H` rows), `execute_segmented` (`43 / 43` Official ClickBench at `100M` rows with zero-`Arc<str>` `eval_raw_text_filter` & inline `HKey::K4/K5`), `columnar_join` (`22 / 22` Official `TPC-H` at `SF=0.1`, `1.0` [`0.93 s`], & `10.0` [`7.44 s`] with `PK-FD` group-by key compression, 3-hop `KeyIn` semi-join pushdown, 64-shard `NeqExists`, and `CompactNumAgg`), and `DBSP` `ZSet<T>` incremental views.
4. **5-Node Multi-Host WAN & Local TCP Raft Consensus (`raft.rs`, `raft_transport.rs`, `examples/raft_wan_node.rs`):**
   - Quorum write commit via `RaftNode::propose_and_commit` (§5.4.2), quorum linearizable reads via `RaftNode::read_linearizable` (§6.4 `ReadIndex` heartbeat quorum, zero WAL growth), `CRC-32` `raft.wal` crash recovery, and both multi-host Tailscale WAN (`27.58 ms` RTT + `SIGKILL`) and local 20-link fault-injection suites verified **`{:valid? true}`** by official Clojure `elle/elle 0.2.8`.
5. **Energy Telemetry & `.jsyn` Sub-1-Bit Sentence Embeddings (`joule-db-energy`, `joule-db-weights`):**
   - Bare-metal hardware counter readers (**`reported_j`** via Linux `intel-rapl` and macOS `IOKit SMC`) segregated from analytical cost models (**`est_j`**), plus pure-Rust [`dcharlot65-openie/all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit) (`1.95 MB`, `0.4378 BPW`, **`0.9441` cosine** vs `FP32`).

---

## Reproducible External Evaluations (`bench/`)

### 1. Full Official ClickBench (`hits.parquet`, `99,997,497` Rows, `105` Columns, `43` Queries) vs. DuckDB `2.0.1`

Executed via [`bench/clickbench_official.py`](./bench/clickbench_official.py) over real `pgwire` (`psycopg 3.3.9`, `--data-parent-dir /var/tmp` physical NVMe on `100M`) on `framework-128gb-0000` (`AMD Ryzen AI Max+ 395`, `128 GB` RAM, Linux `rapl-package` **`reported_j`**):

| Official `hits.parquet` Scale | Parquet Load (`DuckDB` vs `JouleDB`) | JouleDB Post-Load RSS (`VmHWM` Peak) | Queries Succeeded | SQL-Valid Match (`Exact`) | Faster (`JouleDB` / `DuckDB`) | Geomean Latency (`JouleDB / DuckDB`) | Geomean Energy (`reported_j`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`100,000` rows** ([JSON](./bench/results/clickbench_official_100k.json)) | `0.179 s` vs `0.265 s` | `918 MB` (`2,652 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`34` exact) | **`20`** / `23` | **`1.47×` slower** (`median 1.19×`) | **`1.97×` more J** |
| **`1,000,000` rows** ([JSON](./bench/results/clickbench_official_1m.json)) | `0.158 s` vs `0.871 s` | `1,283 MB` (`1,923 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`29` exact) | **`11`** / `32` | **`2.34×` slower** (`median 2.95×`) | **`2.50×` more J** |
| **`10,000,000` rows** ([JSON](./bench/results/clickbench_official_10m.json)) | `2.422 s` vs `13.340 s` | `2,385 MB` (**`3,448 MB`**) | **`43 / 43`** (`0` err) | **`43 / 43`** (`31` exact) | **`9`** / `34` | **`2.28×` slower** (`median 2.63×`) | **`2.22×` more J** |
| **`99,997,497` rows — FULL `14.78 GB` NVMe** ([JSON](./bench/results/clickbench_official_100m.json)) | `18.208 s` vs `161.403 s` | **`139.2 MB` (`9,454 MB`)** | **`43 / 43`** (`0` err) | **`43 / 43`** (`30` exact) | **`8`** / `35` | **`2.35×` slower NVMe** (`2.11×` `tmpfs`, `median 3.36×`) | **`2.09×` more J** |

- **Candid Full-`100M` Summary:** Out-of-core `.jcol` segments on physical NVMe slash **post-load RSS `42.4×` (`5,901.8 MB -> 139.2 MB`) & peak `VmHWM` `1.74×` (`16,469.7 MB -> 9,454.5 MB`)**, beat DuckDB `2.0.1` on **`8 / 43` queries at `100M` rows** (`Q0` `1.40 ms` [`1.84×`], `Q2` `1.16 ms` [`7.48×`], `Q3` `1.56 ms` [`25.7×`], `Q6` `1.35 ms` [`5.05×`], `Q29` `1.81 ms` [`7.82×`], `Q37` `30.48 ms` [`1.07×`], `Q41` `11.86 ms` [`2.06×`], `Q42` `11.31 ms` [`3.21×`]), and slash tail latencies **`3.05×` on `Q39` (`906.85 -> 297.01 ms`), `4.04×` on `Q32` (`5,933.5 -> 1,467.5 ms`), `2.47×` on `Q30` (`668.5 -> 270.9 ms`), and `4.73×` on `Q7` (`34.95 -> 7.40 ms`)**. DuckDB `2.0.1` wins **`35 / 43` queries (`2.35×` geomean on NVMe / `2.11×` on `tmpfs`)**.

### 2. Official 22-Query `TPC-H` Relational Suite (`tpch_queries()`, 8 Tables, `SF = 0.1`, `1.0` & `10.0`) vs. DuckDB `2.0.1`

Executed via [`bench/tpch_official.py`](./bench/tpch_official.py) over real TCP `pgwire` (`psycopg 3.3.9`, physical NVMe `--work-dir /var/tmp`, Linux `rapl-package` **`reported_j`**) across all **22 official `TPC-H` SQL queries (`Q01..Q22`)**:

| Official `TPC-H` Scale | 8-Table Parquet Load (`DuckDB` vs `JouleDB`) | Post-Load RSS (`VmHWM` Peak) | Exact Verified (`Faster`) | Total 22-Query Time (`DuckDB` vs `JouleDB`) | Geomean Latency (`JouleDB / DuckDB`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`SF = 0.1` (`866,602` rows)** ([JSON](./bench/results/tpch_official_sf01.json)) | `0.430 s` (`12.2 J`) vs `1.380 s` (`54.5 J`) | **`780.8 MB`** (`939.3 MB`) | **`22 / 22`** (**`5` faster**: `Q02,Q11,Q16,Q20,Q22`) | `88.2 ms` vs **`253.0 ms`** (`196.0×` vs `v0.10.0`) | **`1.99×` slower** (`median 2.16×`, down from `125.5×`) |
| **`SF = 1.0` (`8,661,245` rows)** ([JSON](./bench/results/tpch_official_sf1.json)) | `1.200 s` (`65.0 J`) vs `4.707 s` (`143.8 J`) | **`201.6 MB`** (**`578.5 MB`**) | **`22 / 22`** (`0` err, `0` timeouts) | `238.6 ms` vs **`927.5 ms`** (**sub-second**, `1.71×` vs `v0.15.0`) | **`3.61×` slower** (`median 3.59×`, down from `6.18×`) |
| **`SF = 10.0` (`86,586,082` rows, `60M` `lineitem`)** ([JSON](./bench/results/tpch_official_sf10.json)) | `5.839 s` (`508.5 J`) vs `46.135 s` (`1,373.0 J`) | **`754.9 MB`** (**`4,259.3 MB`**) | **`22 / 22`** (`0` err, `0` timeouts) | `945.6 ms` vs **`7,441.0 ms`** (**`2.18×` faster vs `v0.15.0`**) | **`6.41×` slower** (`median 6.51×`, down from `16.03×`) |

- **Candid `TPC-H` Summary:** `PK-FD` group-by key compression (`customer -> nation` payload stripping + 20-row late materialization), 3-hop transitive semi-join pushdown, and 64-shard `NeqExists` slash **`Q10` `13.57×` at `SF=10.0` (`4,653.0 -> 342.78 ms`, narrowing `45.8× -> 4.44×` vs DuckDB) & `8.02×` at `SF=1.0` (`399.2 -> 49.78 ms`, `2.20×`)**, slash **`Q21` `3.75×` at `SF=10.0` (`2,147.9 -> 573.18 ms`) & `3.15×` at `SF=1.0` (`447.5 -> 142.00 ms`)**, cut **`Q18` to `621.63 ms` & `Q16` to `153.09 ms` (`3.64×`)**, and cut total `SF=10.0` runtime **`2.18×` (`16.20 s -> 7.44 s`, `6.41×` geomean)** while holding post-load RSS at **`754.9 MB`**. DuckDB `2.0.1` remains faster at `SF=1.0..10.0` (widest `SF=10.0` ratios: `Q12` `959.9 ms` [`33.0×`], `Q11` `179.0 ms` [`22.7×`], `Q09` `1,251.3 ms` [`12.9×`]).

### 3. Head-to-Head `100`-Warehouse Physical-NVMe TCP `pgwire` OLTP vs. `PostgreSQL 16.16` & `SQLite 3.53.1`

Executed via [`bench/oltp_pgwire.py`](./bench/oltp_pgwire.py) ([`oltp_pgwire_100wh.json`](./bench/results/oltp_pgwire_100wh.json), physical NVMe `--work-dir /var/tmp`, `rapl-package` **`reported_j`**, `idle_watts = 10.03 W`, `100,000` accounts across `100` warehouses, `50%` indexed point read, `30%` durable `BEGIN; UPDATE accounts ...; INSERT INTO history ...; COMMIT` [`fsync=on`, `synchronous_commit=on`], `20%` warehouse `COUNT`/`SUM`, `invariant_verified = true` across all 5 runs):

| Engine & Wire Protocol (`v0.16.0` Decoupled `log_atomic_tx_fast` WAL) | `c = 1` TPS (`p50`) | `c = 4` TPS (`p50`, `p95`) | `c = 8` TPS (`p50`, `p95`) | `c = 16` TPS (`p50`, `mJ/txn`) | Committed Writes |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **JouleDB `0.16.0` (`pgwire` Simple Query, 4-RTT + NVMe `fdatasync`)** | **`3,698.0`** (`0.137 ms`) | **`5,069.3`** (`0.537 ms`, **`1.90 ms`**) | `4,094.7` (`0.676 ms`, `6.71 ms`) | **`5,474.9`** (`0.926 ms`, `11.94 mJ`) | `43,783` |
| **JouleDB `0.16.0` (`pgwire` Extended Pipeline, 1-RTT + NVMe `fdatasync`)** | `1,083.8` (`0.137 ms`) | **`2,486.1`** (`0.579 ms`, **`5.06 ms`**) | `4,342.7` (`0.633 ms`, `5.59 ms`) | **`6,364.8`** (`0.851 ms`, **`9.57 mJ`**) | `34,071` |
| **PostgreSQL `16.16` (`pgwire` Simple Query, 4-RTT + NVMe `fsync`)** | `1,139.4` (`0.073 ms`) | `2,614.9` (`0.089 ms`, `5.11 ms`) | **`4,901.2`** (`0.148 ms`, `5.47 ms`) | **`10,081.7`** (`0.250 ms`, `3.98 mJ`) | `44,780` |
| **PostgreSQL `16.16` (`pgwire` Extended Pipeline, 1-RTT + NVMe `fsync`)** | **`1,161.7`** (`0.071 ms`) | `2,386.7` (`0.103 ms`, `5.74 ms`) | **`5,275.9`** (`0.126 ms`, `5.10 ms`) | **`10,205.8`** (`0.238 ms`, `3.68 mJ`) | `45,496` |
| **SQLite `3.53.1` (Embedded in-proc C API, NVMe `WAL`, `NORMAL`)** | `44,204.9` (`0.005 ms`) | `87,633.3` (`0.006 ms`, `0.04 ms`) | `98,374.2` (`0.007 ms`, `0.06 ms`) | **`111,079.6`** (`0.007 ms`, `0.36 mJ`) | `845,139` (in-proc) |

- **Candid Physical-NVMe OLTP Summary:** Decoupling `JMVR`/`JWB1`/`JWU1` WAL encoding & OS `write_all` + `fdatasync` completely outside `AmorphicStore`'s `RwLockWriteGuard` (`WalManager::log_atomic_tx_fast` + `encode_single_row`) propels JouleDB **`3.25×` past `PostgreSQL 16.16` at `c = 1` Simple (`3,698.0` vs `1,139.4 TPS`, `1.54×` less J/txn)**, **`1.94×` past `PostgreSQL 16.16` at `c = 4` Simple (`5,069.3` vs `2,614.9 TPS`, `p95 1.90 ms` vs `5.11 ms`)**, **`1.04×` past `PostgreSQL 16.16` at `c = 4` Extended Pipeline (`2,486.1` vs `2,386.7 TPS`)**, and lifts `c = 16` Extended Pipeline **`+28%` to `6,364.8 TPS`** (`PostgreSQL 16.16` wins `1.20×–1.21×` at `c=8` and `1.60×–1.84×` at `c=16`).

### 4. 5-Node Multi-Host Cross-Machine WAN/Tailscale TCP Raft (`ReadIndex`) + Official Clojure `elle 0.2.8`

Executed via [`bench/raft_multihost_wan.py`](./bench/raft_multihost_wan.py), [`examples/raft_wan_node.rs`](./crates/joule-db-server/examples/raft_wan_node.rs), and [`bench/elle_check.clj`](./bench/elle_check.clj):
- **Cross-Machine WAN Topology & Nemeses:** 5 real OS `raft_wan_node` processes with durable `raft.wal` across **2 distinct physical Linux hosts** (`framework-128gb-0000` bare-metal + `openie-core-1` Cloud Debian 12 VM, **`27.58 ms` median Tailscale WAN TCP RTT**), cross-WAN `2/5` minority split isolating `n1`, `3/5` majority election (`n3`), real OS **`SIGKILL` (`kill -9`)** on `n5`, and `raft.wal` convergence (`commit=68, applied=68, x=68, y=39`) -> **[`{:valid? true}`](./bench/results/raft_wan_multihost_elle_verdict.txt)** (`270` WAN events, `:strict-serializable`).

### 5. `.jsyn` (`JSYN-v1`) Fidelity vs. Compression Curve

| Model / Layer | `BPW` | Shrink vs `FP16` | Measured Fidelity vs Upstream | Guidance |
| :--- | :---: | :---: | :---: | :--- |
| **[`all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit)** (`6L` Encoder) | `0.438` | `22.2×` (`1.95 MB`) | **`0.9441` embedding cosine** vs `FP32` | **Recommended** for semantic search & `TriModal RRF` |
| `Gemma-4-E2B` `q_proj` (`r = 8..128`) | `0.028–0.166` | `564×–96.6×` | `0.271–0.455` output cos (`KL 3.87–2.72`) | **Experimental only** — not a drop-in causal LLM (retracted from HF) |
| `Inkling-2-Mini` `w13` (`128`-Exp MoE, `r = 8..128`) | `0.023–0.134` | `701×–120×` | `0.113–0.350` output cos (`KL 2.91–2.59`) | **Experimental only** — not a drop-in causal LLM (retracted from HF) |

### 6. Head-to-Head Core Banking Double-Entry Ledger (`4 / 4` Regime Sweep vs `PostgreSQL 16.16`) & Unified Suite

Executed via [`bench/ledger_banking_head_to_head.py`](./bench/ledger_banking_head_to_head.py) ([`ledger_banking_nvme_rapl_v016.json`](./bench/results/ledger_banking_nvme_rapl_v016.json), physical NVMe `--work-dir /var/tmp`, Linux `rapl-package` **`reported_j`**, `1,000` `NO_OVERDRAFT` customer wallets, `5%` duplicate `idempotency_key` retries, SHA-256 digest per transfer, `fsync=on` & `synchronous_commit=on`, `493,941` HTTP & `436,801` `pgwire` SHA-256 chain entries verified `valid = true`):

| Core Banking Regime (`1,000` `NO_OVERDRAFT` Wallets, `fsync=on`) | JouleDB `0.16.0` HTTP (`JWL1` + SHA-256) | JouleDB `0.16.0` `pgwire` (`psycopg3` SQL) | PostgreSQL `16.16` (`FOR UPDATE` / `UNNEST`) | Speedup & RAPL Energy (`reported_j`) |
| :--- | :---: | :---: | :---: | :--- |
| **`c = 1` Serial Durable Transfer** | **`3,424.5 TPS`** (`p50 0.242 ms`, **`26.96 mJ/tx`**) | **`400.6 TPS`** (`p50 2.601 ms`, `201.89 mJ/tx`) | `371.9 TPS` (`p50 2.781 ms`, `129.73 mJ/tx`) | **JouleDB `9.21×` HTTP (`1.08×` `pgwire`) · `4.81×` less J/tx** |
| **`c = 4` Concurrent Group Commit** | **`1,250.0 TPS`** (`p50 2.993 ms`, `62.58 mJ/tx`) | **`1,412.7 TPS`** (`p50 2.761 ms`, **`56.08 mJ/tx`**) | `777.6 TPS` (`p50 5.271 ms`, `61.57 mJ/tx`) | **JouleDB `1.82×` `pgwire` (`1.61×` HTTP) · `1.10×` less J/tx** |
| **`c = 16` High-Contention Concurrent** | `2,636.9 TPS` (`p50 5.129 ms`, `44.07 mJ/tx`) | **`3,919.0 TPS`** (`p50 3.595 ms`, `22.56 mJ/tx`) | `2,861.0 TPS` (`p50 5.658 ms`, **`20.51 mJ/tx`**) | **JouleDB `1.37×` `pgwire` win** + global SHA-256 chain |
| **`batch = 500` Clearing Batch (`1` `fdatasync` / `500` tx)** | **`91,316.1 TPS`** (`p50 4.290 ms`, `1.06 mJ/tx`) | **`81,453.5 TPS`** (`104,721 TPS` peak) | `53,956.4 TPS` (`p50 8.556 ms`, `1.03 mJ/tx`) | **JouleDB `1.69×` faster (`>100k TPS` peak)** |

- **Unified Banking, `0.553 ms` Lazy CoW Branching, `pgvector` HNSW & Human ORM Verification ([`unified_banking_agentic_v016.json`](./bench/results/unified_banking_agentic_v016.json)):** Verifies `0.10 + 0.20 = 0.30` & `1,000 × 0.01 = 10` exact `Decimal128`, multi-leg split settlements, two-phase `HOLD`/`POST`, SQL `JOIN` (`kyc_customers ⋈ jouledb_ledger_accounts`), 500-op hot-wallet contention (`0` overdrafts, `879.0 ops/s`), **`0.553 ms` median pure `CREATE BRANCH` (`0.432 ms` on Apple Silicon — `69×` faster than `v0.14.0`)**, `42501`/`40001`/`53200` guardrails, native `pgvector` HNSW cosine `<=>` top-K recall, `WORKFLOW CHECKPOINT` replay, `SELECT ... FOR UPDATE [NOWAIT | SKIP LOCKED]`, and **`0`-`xfail` `SQLAlchemy 2` (`4/4`) + `psycopg3` (`12/12`) + `node-pg` (`5/5`) + `pgJDBC` (`7/7`)**.

### 7. Large Text Compression Benchmark (`LTCB` `enwik8`/`enwik9`) — `Joules-per-Bit` & Seekable Database Pareto (`bench/ltcb_energy_pareto.py`)

Executed via [`bench/ltcb_energy_pareto.py`](./bench/ltcb_energy_pareto.py) ([`ltcb_energy_pareto_v016.json`](./bench/results/ltcb_energy_pareto_v016.json)) on Matt Mahoney's official `enwik8` (`100,000,000` B = `12,346` English Wikipedia XML articles, Linux RAPL **`reported_j`**) alongside published primary-source `enwik9` (`1,000,000,000` B) neural compressors from [`mattmahoney.net/dc/text.html`](https://mattmahoney.net/dc/text.html) (`analytical_est` clearly segregated):

| System / Storage Format | Corpus | Ratio (`bpb`) | Decode / Scan Speed | `O(1)` Article Lookup | Decode / Scan Energy (`nJ/bit`) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`cmix-lex-transformer`** *(published LTCB #1)* | `enwik9` (`1 GB`) | **`10.43×` (`0.767 bpb`)** | `0.0047 MB/s` (`2.47 d`) | None (full replay) | `~668,250 nJ/bit` *(`analytical_est` 25W)* |
| **`zmix 1.0` (`fx2-cmix` Zig)** *(published LTCB #2)* | `enwik9` (`1 GB`) | **`10.41×` (`0.769 bpb`)** | `0.0076 MB/s` (`1.52 d`) | None (full replay) | `~245,970 nJ/bit` *(`analytical_est` 15W)* |
| **`tufazip v2` (Tufa Labs `163M`)** *(published LTCB #3)* | `enwik9` (`1 GB`) | **`9.49×` (`0.843 bpb`)** | `0.0016 MB/s` (`7.29 d`) | None (`630,136 s`) | `~157,534,000 nJ/bit` *(`analytical_est` 2kW B200)* |
| **`zstd -19` (stream)** *(live RAPL)* | `enwik8` (`100 MB`) | `3.71×` (`2.156 bpb`) | `177.7 MB/s` (`0.56 s`) | None (stream) | **`90.54 nJ/bit`** (`reported_j` `72.43 J`) |
| **`zstd -3` (stream)** *(live RAPL)* | `enwik8` (`100 MB`) | `2.82×` (`2.836 bpb`) | `577.0 MB/s` (`0.17 s`) | None (stream) | **`27.97 nJ/bit`** (`reported_j` `22.37 J`) |
| **Seekable `Parquet Zstd` (Ontology+SimHash sorted)** | `enwik8` (`12.3k` docs) | **`2.79×` (`2.870 bpb`)** | Row-group seekable | `O(row_group)` | **`1.46%` smaller** than stream order |
| **`JouleDB v0.16.0` `.jcol` SQL (`pgwire`)** *(live RAPL)* | `enwik8` (`12.3k` docs) | Raw mmap `.jcol` | **`3,374 MB/s`** (`28.36 ms` `LIKE`) | **`5.41 ms` (`116.4M×` vs `tufazip`)** | **`2.83 nJ/bit`** (`reported_j` `2.16 J` — **`55.7M×` less J/bit**) |

---

## Quickstart (`v0.16.0`)

```bash
# 1. Install prebuilt v0.16.0 binary (macOS ARM64/x86_64; Linux x86_64 GLIBC <= 2.35, Ubuntu 22.04/24.04 & Debian 12 ready)
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

- **Pre-1.0 research preview (`v0.16.0`), not a battle-tested production RDBMS:** While JouleDB `0.16.0` unifies exact `128-bit` `NUMERIC` + adaptive group-committed `JWL1` SQL `LEDGER` beating `PostgreSQL 16.16` across all `4 / 4` NVMe regimes (`9.21×` `c=1`, `1.82×` `c=4`, `1.37×` `c=16`, `1.69×–1.93×` `batch=500`) + `0.553 ms` lazy CoW SQL branching + native `pgvector` HNSW SQL + `43 / 43` Official ClickBench on `99,997,497` rows (`139.2 MB` post-load RSS, `8 / 43` wins) + `22 / 22` Official `TPC-H` up to `SF=10.0` (`7.44 s` total, `754.9 MB` post-load RSS, `6.41×` geomean) + `3.25×` `c=1` / `1.94×` `c=4` / `6,365 TPS` `c=16` NVMe `pgwire` OLTP + 5-node WAN Raft (`elle` `{:valid? true}`), mature single-purpose systems (`DuckDB 2.0.1` `2.11×–2.35×` faster on `100M` ClickBench & `3.61×–6.41×` faster on `SF=1..10` `TPC-H`; `PostgreSQL 16.16` `1.20×–1.84×` faster on `c=8..16` general NVMe `pgwire` OLTP; non-SQL `TigerBeetle` on `io_uring` fixed-struct clusters; offline neural compressors `cmix`/`zmix`/`tufazip` on non-seekable `0.77–0.84 bpb` archival ratio) remain ahead on their specialized single axis.
- **No external compliance certifications:** `jouledb trust` exercises technical security primitives in code (`external_audit_performed = false`). **No third-party SOC 2, ISO 27001, HIPAA, PCI-DSS, NIST CMVP, or FedRAMP audit has been conducted.**
- **Exact compiler safety lint scope:** `#![deny(unsafe_code)]` is set in `joule-db-core` with **1** `#![allow(unsafe_code)]` module (`crates/joule-db-core/src/storage/mmap_extent.rs`). `#![cfg_attr(not(test), deny(clippy::unwrap_used))]` is enforced across **`41 / 41` workspace crates (`100%`, `0` `#[allow(clippy::unwrap_used)]` bypasses)** via a blocking CI gate.
- **SDK registry status:** Prebuilt `cp39-abi3` Python `.whl` wheels, TypeScript `openie-jouledb-0.16.0.tgz` (+ GitHub Packages npm), and Claude `.mcpb` bundles are published on [GitHub Releases (`v0.16.0`)](https://github.com/openIE-dev/jouledb/releases/tag/v0.16.0) (with OIDC Trusted Publisher & token jobs wired in `.github/workflows/release.yml`); packages are **not yet published** on public PyPI, npmjs, NuGet, or Maven Central pending upstream registry OIDC registration.

---

## Documentation & Links

- **Website:** [https://jouledb.org](https://jouledb.org) ([Claude & MCP Apps](https://jouledb.org/claude) · [Benchmarks](https://jouledb.org/benchmarks) · [Honest Boundaries](https://jouledb.org/honest) · [Security Controls & Bridges](https://jouledb.org/trust) · [`.jsyn` Fidelity](https://jouledb.org/jsyn))
- **Online Book (`mdBook`):** [https://openie-dev.github.io/jouledb](https://openie-dev.github.io/jouledb) ([Claude & MCP Apps Guide](https://openie-dev.github.io/jouledb/claude-apps.html))
- **Verified `.jsyn` Sentence Embedder & Benchmark Dataset:** [https://huggingface.co/dcharlot65-openie](https://huggingface.co/dcharlot65-openie)
- **Architecture Whitepaper (`v0.16.0`):** [`docs/WHITEPAPER-JOULEDB-2026-11.md`](./docs/WHITEPAPER-JOULEDB-2026-11.md)
- **`.jsyn` Binary Specification:** [`docs/JSYN-LIVING-WEIGHTS-SPEC.md`](./docs/JSYN-LIVING-WEIGHTS-SPEC.md)
- **Benchmark Scripts & Raw JSON/EDN Outputs:** [`bench/`](./bench/)

## License

Copyright (c) 2026 Open Interface Engineering, Inc. Dual-licensed under either [Apache License, Version 2.0](./LICENSE-APACHE) or [MIT License](./LICENSE-MIT), at your option.
