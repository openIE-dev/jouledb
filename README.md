# JouleDB (`v0.13.0` — Pre-1.0 Research Preview)

**An experimental energy-metered multi-model database in Rust (`Apache-2.0 OR MIT`) unifying Core Banking (`128-bit` exact `NUMERIC`, SQL-native double-entry `LEDGER`, SHA-256 journal proofs), Autonomous AI Agents (`O(1)` CoW SQL branching, 3-way conflict `MERGE`, blast-radius & microjoule guardrails, durable `WORKFLOW` checkpoints, `58`-tool Claude/MCP Apps Studio), and Human Developer/Lakehouse Workloads (`pgwire` `psql`/`psycopg3`/`SQLAlchemy 2`/`node-pg`/`JDBC`, `SELECT ... FOR UPDATE SKIP LOCKED`, `100M`-row ClickBench [`2.11×`], `22 / 22` `TPC-H` `SF=10.0` [`86.6M` rows in `755 MB` RSS], `5,160 TPS` NVMe OLTP, and 5-node WAN TCP Raft).**

[![Release](https://github.com/openIE-dev/jouledb/actions/workflows/release.yml/badge.svg)](https://github.com/openIE-dev/jouledb/releases)
[![Quality](https://github.com/openIE-dev/jouledb/actions/workflows/quality.yml/badge.svg)](https://github.com/openIE-dev/jouledb/actions/workflows/quality.yml)
[![Docs](https://github.com/openIE-dev/jouledb/actions/workflows/docs.yml/badge.svg)](https://openie-dev.github.io/jouledb)
[![crates.io](https://img.shields.io/crates/v/jouledb.svg)](https://crates.io/crates/jouledb)
[![Hugging Face Embedder](https://img.shields.io/badge/Hugging%20Face-MiniLM%20.jsyn%20(cos%200.944)-0D9488)](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit)
[![Website](https://img.shields.io/badge/Web-jouledb.org-2563EB)](https://jouledb.org)

> **Pre-1.0 Status & `v0.13.0` Release Summary (`2026-11-27`):** Building on our `v0.6.0` external-audit remediation (`v1.0.0` GA tag retracted/yanked, unrelated `1.11M`-LOC `joule-web` crate and synthetic `frontier_exam_routes.rs` deleted, 81 unverified low-BPW causal LLM exports pruned from Hugging Face) and `v0.7.0`–`v0.12.0` engine upgrades, **`v0.13.0`** closes the demarcation chasms between **Agentic AI sandboxes**, **Core Banking ledgers**, and **Human developer/analyst RDBMSs** inside one `41`-crate (`754.5k` LOC, `12,282` tests, `100%` panic-linted) binary:
> - **Exact `128`-Bit Fixed-Point `DECIMAL(38, s)` + SQL-Native Double-Entry Core Banking `LEDGER` (`decimal128.rs`, `financial_ledger.rs`):** Eliminates IEEE-754 float penny drift (`0.10 + 0.20 = 0.30` exact; `1,000 × 0.01` `SUM = 10` exact; 38-digit scaled `i128` math) and embeds a `TigerBeetle`-class double-entry state machine (`LEDGER CREATE ACCOUNT`, `LEDGER TRANSFER`, `LEDGER MULTILEG`, two-phase `HOLD`/`POST HOLD`/`VOID HOLD`, `NO_OVERDRAFT` balance ceilings, mandatory `idempotency_key` replay/conflict detection, SHA-256 hash-chained `financial_ledger.jwal` `LEDGER VERIFY`, and live SQL `JOIN`s against `jouledb_ledger_accounts`).
> - **Session-Isolated CoW SQL Branching, 3-Way Conflict `MERGE`, Agent Guardrails & Durable `WORKFLOW` Checkpoints (`sql_branch_sandbox.rs`):** `CREATE/USE/SHOW/DIFF/MERGE/DROP BRANCH` isolates agent mutations from `main`, detects 3-way write-write conflicts (`SQLSTATE 40001`), blocks prompt-injected `DROP TABLE` / `TRUNCATE` and `WHERE`-less `DELETE`/`UPDATE` (`SQLSTATE 42501`), enforces per-branch microjoule caps (`SQLSTATE 53200`), and adds DBOS-style crash-proof `WORKFLOW CHECKPOINT` / `WORKFLOW RESUME` (`58` MCP tools).
> - **Human ORM & Queue Worker Parity (`pgwire.rs`, `query.rs`):** Adds `SELECT ... FOR UPDATE [NOWAIT | SKIP LOCKED]`, fixes `SQLAlchemy 2 + psycopg3` `pg_type JOIN pg_range` connect introspection and `psycopg3` `cursor(binary=True)` `RowDescription`/`DataRow` OID width alignment (`psql`, `psycopg3`, `SQLAlchemy 2`, `node-pg`, `pgJDBC` green), alongside `v0.12.0`'s **`46.1×` lower `TPC-H` `SF = 10.0` (`86.6M` rows) post-load RSS (`755.3 MB`, `15.46 s`, `8.49×` geomean)** and **`5,160 TPS` physical-NVMe `pgwire` OLTP**. See [`CHANGELOG.md`](./CHANGELOG.md).

---

## What Is Real & Verified Today (`41` Workspace Crates · `754.5k` LOC · `12,282` Tests · `41 / 41` `deny(unwrap_used)`)

1. **Core Storage, Exact `Decimal128`, SQL-Native Double-Entry Banking Ledger & 64-Shard `SSI` (`joule-db-core`, `joule-db-local`, `joule-db-amorphic`, `joule-db-server`):**
   - Copy-on-Write B-tree pages, Cahill Serializable Snapshot Isolation (`SSI`), 64-shard `MvccStore` + `MvccTableStorage` (`JMVR\x01`, `JWB1`, `JWU1` delta NVMe WAL), exact `128-bit` fixed-point `NUMERIC(38, s)` (`Decimal128`), SQL-native double-entry `FinancialLedger` (`i128` minor units, multi-leg conservation, two-phase holds, `idempotency_key`, SHA-256 `financial_ledger.jwal`, SQL virtual tables `jouledb_ledger_*`), `SELECT ... FOR UPDATE [NOWAIT | SKIP LOCKED]`, `ALP + FSST` codecs, and `RaBitQ` 1-bit `POPCNT` vectors.
2. **Agentic CoW SQL Branching, Durable `WORKFLOW` Checkpoints & `58`-Tool Claude/MCP Apps Host (`sql_branch_sandbox.rs`, `mcp_app_ui.rs`, `mcp_transport.rs`):**
   - Session-isolated SQL branches (`CREATE/USE/DIFF/MERGE BRANCH` with 3-way `40001` conflict checks), strict agent blast-radius guardrails (`42501` DDL/unbounded-write blocks + `53200` microjoule caps), crash-proof `WORKFLOW CHECKPOINT`/`RESUME`, and native `SEP-1865` **JouleDB Studio** (`ui://jouledb/studio.html`) over `jouledb mcp` (`stdio`) and `POST /mcp` (`58` tools incl. `ledger.*` & `db.workflow_*`).
3. **Out-of-Core `.jcol` Columnar Analytics, Vectorized Multi-Table `TPC-H` Joins & `pgwire` (`joule-db-query`, `joule-db-server`):**
   - PostgreSQL v3 wire listener (`psql`, `psycopg3` text & binary + Extended Pipeline, `SQLAlchemy 2`, `node-pg`, `JDBC`, `GSSENCRequest`), owned-batch streaming Parquet `COPY` spilling `262,144`-row `.jcol` disk segments (`755 MB` post-load RSS at `86.6M` rows), `execute_segmented` (`43 / 43` Official ClickBench at `100M` rows), `columnar_join` (`22 / 22` Official `TPC-H` at `SF=0.1`, `1.0`, & `10.0`), and `DBSP` `ZSet<T>` incremental views.
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
| **`SF = 1.0` (`8,661,245` rows)** ([JSON](./bench/results/tpch_official_sf1.json)) | `1.142 s` (`64.6 J`) vs `5.504 s` (`288.5 J`) | **`201.4 MB`** (`1,561.8 MB`) | **`22 / 22`** (`0` err, `0` timeouts) | `253.0 ms` vs **`1,779.0 ms`** (`1.57×` vs `v0.11.0`) | **`4.33×` slower** (`median 3.60×`, down from `6.18×`) |
| **`SF = 10.0` (`86,586,082` rows, `60M` `lineitem`)** ([JSON](./bench/results/tpch_official_sf10.json)) | `5.434 s` (`506.5 J`) vs `38.059 s` (`1,242.5 J`) | **`755.3 MB`** (`15,586.4 MB`) | **`22 / 22`** (`20/22 < 1.2 s`, `0` err) | `1,001.4 ms` vs **`15,464.6 ms`** (**`2.20×` vs `v0.11.0`**) | **`8.49×` slower** (`median 8.38×`, down from `16.03×`) |

- **Candid `TPC-H` Summary:** Separating `.jcol` segment IDs (`JCOL_ID_TAG`) from `AmorphicStore.next_id` collapses `SF = 10.0` post-load RSS **`46.1×` (`34.81 GB -> 755.3 MB`, `15.59 GB` peak `VmHWM`)** and `SF = 1.0` post-load RSS **`28.6×` (`5.77 GB -> 201.4 MB`)**, while ZoneMap `DenseKeyBits` pruning, two-stage sparse dictionary loading, 64-shard radix `GROUP BY` (`Q18`: `11.33 s -> 1.80 s`), and `20M`-row semi-join propagation (`Q17`: `1,529 ms -> 88.6 ms`, `Q20`: `657 ms -> 99.7 ms`, `Q10`: `9.11 s -> 5.23 s`) cut `SF = 10.0` geomean from **`16.03× -> 8.49×`** and `SF = 0.1` to **`1.99×` (`5 / 22` wins)**. DuckDB `2.0.1` remains **`4.33×` faster at `SF=1.0`** and **`8.49×` faster at `SF=10.0`** (widest `SF=10.0` tails: `Q10` `5.23 s` [`64.8×`], `Q18` `1.80 s` [`18.2×`], `Q09` `1.68 s` [`16.2×`]).

### 3. Head-to-Head `100`-Warehouse Physical-NVMe TCP `pgwire` OLTP vs. `PostgreSQL 16.16` & `SQLite 3.53.1`

Executed via [`bench/oltp_pgwire.py`](./bench/oltp_pgwire.py) ([`oltp_pgwire_100wh.json`](./bench/results/oltp_pgwire_100wh.json), physical NVMe `--work-dir /var/tmp` rather than RAM `tmpfs`, `rapl-package` **`reported_j`**, quiet host `idle_watts = 10.68 W`): `100,000` accounts (`100` warehouses), `50%` indexed point read, `30%` durable `BEGIN; UPDATE accounts ...; INSERT INTO history ...; COMMIT` (`fsync=on`, `synchronous_commit=on`), `20%` warehouse `COUNT`/`SUM` across `1, 4, 8, 16` concurrent TCP clients:

| Engine & Wire Protocol (`invariant_verified = true`, `0` errors on all 5) | `c = 1` TPS (`p50`) | `c = 4` TPS (`p50`) | `c = 8` TPS (`p50`) | `c = 16` TPS (`p50`, `mJ/txn`) | Committed Writes |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **JouleDB `0.12.0` (`pgwire` Simple Query, 4-RTT + NVMe `fdatasync`)** | **`1,950.5`** (`0.148 ms`) | `1,005.0` (`0.666 ms`) | `2,617.9` (`0.640 ms`) | **`5,154.1`** (`0.718 ms`, `10.34 mJ`) | `25,677` (`0` conflicts) |
| **JouleDB `0.12.0` (`pgwire` Extended Pipeline, 1-RTT + NVMe `fdatasync`)** | `620.9` (`0.169 ms`) | `1,149.0` (`0.616 ms`) | `2,730.9` (`0.633 ms`) | **`5,159.7`** (`0.773 ms`, `11.71 mJ`) | `23,158` (`0` conflicts) |
| **PostgreSQL `16.16` (`pgwire` Simple Query, 4-RTT + NVMe `fsync`)** | `1,199.7` (`0.100 ms`) | `2,813.0` (`0.142 ms`) | `5,571.7` (`0.111 ms`) | **`10,419.4`** (`0.273 ms`, `4.14 mJ`) | `47,846` (`0` conflicts) |
| **PostgreSQL `16.16` (`pgwire` Extended Pipeline, 1-RTT + NVMe `fsync`)** | `1,171.7` (`0.071 ms`) | `2,555.3` (`0.100 ms`) | `5,047.0` (`0.167 ms`) | **`9,635.6`** (`0.231 ms`, `3.67 mJ`) | `44,020` (`0` conflicts) |
| **SQLite `3.53.1` (Embedded in-proc C API, NVMe `WAL`, `NORMAL`)** | `42,194.7` (`0.005 ms`) | `86,523.3` (`0.006 ms`) | `97,425.3` (`0.007 ms`) | **`109,483.5`** (`0.007 ms`, `0.36 mJ`) | `830,955` (no TCP / `fsync`) |

- **Candid Physical-NVMe OLTP Summary:** *(Methodology note: `v0.11.0` ran on default `/tmp` [`tmpfs` RAM disk], whereas `v0.12.0` runs on `--work-dir /var/tmp` so every `COMMIT` forces a real NVMe `fdatasync`.)* Binary `JWU1` delta `UPDATE`s, deferred-hologram updates, and `GroupSync` **beat `PostgreSQL 16.16` `1.63×` at `c=1` Simple (`1,950.5` vs `1,199.7 TPS`, `3.42×` less energy/txn)** and scale to **`5,154.1 TPS` Simple / `5,159.7 TPS` Pipelined at `c=16`**, narrowing the `c=16` gap vs `PostgreSQL 16.16` from `2.91× / 2.61×` (`v0.11.0`) to **`2.02×` Simple / `1.87×` Pipelined** (`PostgreSQL 16.16` still wins `1.87×–2.80×` at `c=4..16` via shared-buffer page locks, and JouleDB's adaptive `GroupSync` batch window adds `~1.5 ms` wait at `c=1` Pipelined / `c=4` transition).

### 4. 5-Node Multi-Host Cross-Machine WAN/Tailscale TCP Raft (`ReadIndex`) + Official Clojure `elle 0.2.8`

Executed via [`bench/raft_multihost_wan.py`](./bench/raft_multihost_wan.py), [`examples/raft_wan_node.rs`](./crates/joule-db-server/examples/raft_wan_node.rs), and [`bench/elle_check.clj`](./bench/elle_check.clj):
- **Cross-Machine WAN Topology & Nemeses:** 5 real OS `raft_wan_node` processes with durable `raft.wal` across **2 distinct physical Linux hosts** (`framework-128gb-0000` bare-metal + `openie-core-1` Cloud Debian 12 VM, **`27.58 ms` median Tailscale WAN TCP RTT**), cross-WAN `2/5` minority split isolating `n1`, `3/5` majority election (`n3`), real OS **`SIGKILL` (`kill -9`)** on `n5`, and `raft.wal` convergence (`commit=68, applied=68, x=68, y=39`) -> **[`{:valid? true}`](./bench/results/raft_wan_multihost_elle_verdict.txt)** (`270` WAN events, `:strict-serializable`).

### 5. `.jsyn` (`JSYN-v1`) Fidelity vs. Compression Curve

| Model / Layer | `BPW` | Shrink vs `FP16` | Measured Fidelity vs Upstream | Guidance |
| :--- | :---: | :---: | :---: | :--- |
| **[`all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit)** (`6L` Encoder) | `0.438` | `22.2×` (`1.95 MB`) | **`0.9441` embedding cosine** vs `FP32` | **Recommended** for semantic search & `TriModal RRF` |
| `Gemma-4-E2B` `q_proj` (`r = 8..128`) | `0.028–0.166` | `564×–96.6×` | `0.271–0.455` output cos (`KL 3.87–2.72`) | **Experimental only** — not a drop-in causal LLM (retracted from HF) |
| `Inkling-2-Mini` `w13` (`128`-Exp MoE, `r = 8..128`) | `0.023–0.134` | `701×–120×` | `0.113–0.350` output cos (`KL 2.91–2.59`) | **Experimental only** — not a drop-in causal LLM (retracted from HF) |

### 6. Unified Core Banking, Agentic CoW Sandbox & Human Row-Locking Suite (`bench/unified_banking_agentic.py`)

Executed via [`bench/unified_banking_agentic.py`](./bench/unified_banking_agentic.py) ([`unified_banking_agentic_v013.json`](./bench/results/unified_banking_agentic_v013.json)) against a single live `jouledb` process:
- **Core Banking (`Decimal128` + `FinancialLedger`):** `0.10 + 0.20 = 0.30` exact, `30`-digit `DECIMAL_ADD` exact, `1,000 × 0.01` `SUM = 10` exact, multi-leg split settlement, two-phase `HOLD` -> partial `POST HOLD`, live SQL `JOIN` (`kyc_customers JOIN jouledb_ledger_accounts`), 500 concurrent hot-wallet debit attempts against a `$350.00` balance (**`350` committed, `150` rejected, `0` overdrafts**), and `LEDGER VERIFY` SHA-256 chain proof (`valid = true`).
- **Autonomous Agent Sandbox (`SqlBranchSandboxManager`):** 10 isolated `CREATE BRANCH ... GUARDRAILS STRICT` sandboxes (`DROP TABLE` & unbounded `DELETE` blocked with `SQLSTATE 42501`), `DIFF BRANCH`, clean `MERGE BRANCH`, true 3-way write-write conflict rejection (`SQLSTATE 40001`), and idempotent `WORKFLOW CHECKPOINT` replay.
- **Human Queue Worker Locking (`RowLockManager`):** Concurrent `SELECT ... FOR UPDATE NOWAIT` (`SQLSTATE 55P03`) and `SELECT ... FOR UPDATE SKIP LOCKED` row-claim verification.

---

## Quickstart (`v0.13.0`)

```bash
# 1. Install prebuilt v0.13.0 binary (macOS ARM64/x86_64; Linux x86_64 GLIBC <= 2.35, Ubuntu 22.04/24.04 & Debian 12 ready)
cargo binstall jouledb

# 2. Launch JouleDB server (HTTP :8080, TCP :8081, PgWire :8082, JWP :8083, FlightSQL :8084, MCP HTTP /mcp)
jouledb --port 8080 --data ./joule-data

# 3. Or run as a Claude Desktop / MCP Apps stdio server (58 tools + interactive ui://jouledb/studio.html)
jouledb mcp --data ./joule-data

# 4. Connect with psql, SQLAlchemy 2, or run smoke/security self-tests
psql -h 127.0.0.1 -p 8082 -U postgres -c "SELECT 0.10 + 0.20 = 0.30, version();"
jouledb self-check && jouledb trust
```

---

## Honest Engineering Boundaries (Read Before Adopting)

- **Pre-1.0 research preview (`v0.13.0`), not a battle-tested production RDBMS:** While JouleDB `0.13.0` unifies exact `128-bit` `NUMERIC` + double-entry SQL `LEDGER` + session-isolated CoW SQL branching + `43 / 43` Official ClickBench on `99,997,497` rows (`2.11×` geomean vs DuckDB `2.0.1`) + `22 / 22` Official `TPC-H` up to `SF=10.0` (`86.6M` rows in `755 MB` post-load RSS, `8.49×` geomean) + `5,160 TPS` physical-NVMe `pgwire` OLTP + 5-node cross-machine WAN Raft (`elle` `{:valid? true}`), mature single-purpose engines (`DuckDB 2.0.1` `2.11×` faster on `100M` ClickBench & `4.33×`–`8.49×` faster on `SF=1..10` `TPC-H`; `PostgreSQL 16.16` `1.87×–2.80×` faster on `c=4..16` NVMe `pgwire` OLTP; dedicated clustered `TigerBeetle` at `100k–1M+` batched ledger TPS; `SQLite` embedded WAL) remain faster on their specialized single axis.
- **No external compliance certifications:** `jouledb trust` exercises technical security primitives in code (`external_audit_performed = false`). **No third-party SOC 2, ISO 27001, HIPAA, PCI-DSS, NIST CMVP, or FedRAMP audit has been conducted.**
- **Exact compiler safety lint scope:** `#![deny(unsafe_code)]` is set in `joule-db-core` with **1** `#![allow(unsafe_code)]` module (`crates/joule-db-core/src/storage/mmap_extent.rs`). `#![cfg_attr(not(test), deny(clippy::unwrap_used))]` is enforced across **`41 / 41` workspace crates (`100%`, `0` `#[allow(clippy::unwrap_used)]` bypasses)** via a blocking CI gate.
- **SDK registry status:** Prebuilt `cp39-abi3` Python `.whl` wheels, TypeScript `openie-jouledb-0.13.0.tgz`, and Claude `.mcpb` bundles are published on [GitHub Releases (`v0.13.0`)](https://github.com/openIE-dev/jouledb/releases/tag/v0.13.0) (with automated OIDC/token publish steps wired in `.github/workflows/release.yml`); packages are **not yet published** on PyPI, npm, NuGet, or Maven Central pending registry token/OIDC registration.

---

## Documentation & Links

- **Website:** [https://jouledb.org](https://jouledb.org) ([Claude & MCP Apps](https://jouledb.org/claude) · [Benchmarks](https://jouledb.org/benchmarks) · [Honest Boundaries](https://jouledb.org/honest) · [Security Controls & Bridges](https://jouledb.org/trust) · [`.jsyn` Fidelity](https://jouledb.org/jsyn))
- **Online Book (`mdBook`):** [https://openie-dev.github.io/jouledb](https://openie-dev.github.io/jouledb) ([Claude & MCP Apps Guide](https://openie-dev.github.io/jouledb/claude-apps.html))
- **Verified `.jsyn` Sentence Embedder & Benchmark Dataset:** [https://huggingface.co/dcharlot65-openie](https://huggingface.co/dcharlot65-openie)
- **Architecture Whitepaper (`v0.13.0`):** [`docs/WHITEPAPER-JOULEDB-2026-11.md`](./docs/WHITEPAPER-JOULEDB-2026-11.md)
- **`.jsyn` Binary Specification:** [`docs/JSYN-LIVING-WEIGHTS-SPEC.md`](./docs/JSYN-LIVING-WEIGHTS-SPEC.md)
- **Benchmark Scripts & Raw JSON/EDN Outputs:** [`bench/`](./bench/)

## License

Copyright (c) 2026 Open Interface Engineering, Inc. Dual-licensed under either [Apache License, Version 2.0](./LICENSE-APACHE) or [MIT License](./LICENSE-MIT), at your option.
