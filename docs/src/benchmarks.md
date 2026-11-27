# Benchmarks, Consensus Verification & Subsystem Smoke Tests (`v0.15.0`)

> **Audit Remediation & Evidence Standard (`v0.15.0`):** Earlier `v0.2.8`–`v1.0.0` documentation presented 36 in-process micro-harnesses and synthetic simulations as *"36 / 36 Certified Global Benchmark Pillars"*, including synthetic frontier AI exam routes (`frontier_exam_routes.rs`). **All synthetic frontier AI benchmark claims remain retracted and `frontier_exam_routes.rs` has been deleted.**
>
> Below we strictly separate **(1) reproducible external dataset, multi-table `TPC-H` (`SF=0.1..10.0`), physical-NVMe core-banking & multi-client `pgwire` OLTP, sub-millisecond lazy CoW branching, & multi-host WAN consensus evaluations** (with every win, loss, memory profile, and architectural bottleneck reported openly) from **(2) fast in-process functional smoke checks**, and tag every energy figure as **`reported_j`** (hardware RAPL / SMC counters) or **`est_j`** (analytical cost model).

---

## 1. Full Official ClickBench (`hits.parquet`, `99,997,497` Rows, `105` Columns, `43` Queries) vs. DuckDB `2.0.1`

- **Harness & Raw Artifacts:** [`bench/clickbench_official.py`](https://github.com/openIE-dev/jouledb/blob/main/bench/clickbench_official.py), [`clickbench_official_100k.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/clickbench_official_100k.json), [`clickbench_official_1m.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/clickbench_official_1m.json), [`clickbench_official_10m.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/clickbench_official_10m.json), **[`clickbench_official_100m.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/clickbench_official_100m.json)**.
- **Methodology:** Streams the official `14.78 GB` (`99,997,497`-row, `105`-column) ClickBench `hits.parquet` file into DuckDB `2.0.1` (in-process) and JouleDB (over real TCP `pgwire` via `psycopg 3.3.9`, spilling immutable `262,144`-row `.jcol` segments to disk), then runs all **43 official ClickBench SQL queries (`Q0..Q42`)** unchanged on `framework-128gb-0000` (`AMD Ryzen AI Max+ 395`, 32 logical cores, `128 GB` UMA RAM, Linux `rapl-package` **`reported_j`**).

| Dataset Scale (`hits.parquet`) | Parquet Load (`DuckDB` vs `JouleDB`) | JouleDB Load RSS (`VmHWM` Peak) | Queries Succeeded | SQL-Valid Match (`Exact`) | Faster (`JouleDB` / `DuckDB`) | Geomean Latency (`JouleDB / DuckDB`) | Geomean Energy (`reported_j`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`100,000` rows** (`105` cols) | `0.179 s` (`11.2 J`) vs `0.265 s` (`18.3 J`) | `917.8 MB` (`2,651.5 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`34` exact) | **`20`** / `23` | **`1.47×` slower** (`median 1.19×`) | **`1.97×` more J** |
| **`1,000,000` rows** (`105` cols) | `0.158 s` (`15.8 J`) vs `0.871 s` (`77.7 J`) | `1,282.9 MB` (`1,922.8 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`29` exact) | **`11`** / `32` | **`2.34×` slower** (`median 2.95×`) | **`2.50×` more J** |
| **`10,000,000` rows** (`105` cols) | `2.422 s` (`336.4 J`) vs `13.340 s` (`1,850.8 J`) | `2,384.7 MB` (**`3,448.4 MB`**) | **`43 / 43`** (`0` err) | **`43 / 43`** (`31` exact) | **`9`** / `34` | **`2.28×` slower** (`median 2.63×`) | **`2.22×` more J** |
| **`99,997,497` rows — FULL `14.78 GB`** (`is_full_100m_dataset: true`) | `13.866 s` (`1,805.6 J`) vs `97.835 s` (`10,047.0 J`) | **`5,901.8 MB` (`16,469.7 MB`)** | **`43 / 43`** (`0` err) | **`43 / 43`** (`32` exact) | **`7`** / `36` | **`2.11×` slower** (`median 3.39×`) | **`2.03×` more J** |

### Selected Per-Query Results on the Full `99,997,497`-Row (`100M`) Dataset (`reported_j`)

| Query | SQL Summary | DuckDB `2.0.1` | JouleDB (`.jcol` + `pgwire`) | Outcome |
|---|---|---:|---:|---|
| `Q0` | `SELECT COUNT(*) FROM hits` | `5.67 ms` | **`1.91 ms`** | Exact match (**JouleDB `2.98×` faster** — segment metadata) |
| `Q1` | `SELECT COUNT(*) FROM hits WHERE AdvEngineID <> 0` | `41.65 ms` | **`20.23 ms`** | Exact match (**JouleDB `2.06×` faster** — `WordMask` bitpacked selection) |
| `Q2` | `SELECT SUM(AdvEngineID), COUNT(*), AVG(ResolutionWidth) FROM hits` | `61.55 ms` | **`2.41 ms`** | Exact match (**JouleDB `25.5×` faster** — zero-I/O `global_int_sum`) |
| `Q3` | `SELECT AVG(UserID) FROM hits` | `51.82 ms` | **`2.32 ms`** | Exact match (**JouleDB `22.4×` faster** — zero-I/O `global_int_sum`) |
| `Q6` | `SELECT MIN(EventDate), MAX(EventDate) FROM hits` | `22.07 ms` | **`1.58 ms`** | Exact match (**JouleDB `14.0×` faster** — segment ZoneMap) |
| `Q29` | `SELECT SUM(ResolutionWidth), SUM(ResolutionWidth + 1), ...` (`90` sums) | `56.20 ms` | **`2.40 ms`** | Exact match (**JouleDB `23.4×` faster** — algebraic `global_int_sum`) |
| `Q42` | `SELECT DATE_TRUNC('minute', EventTime), COUNT(*) WHERE CounterID = 62 ...` | `34.83 ms` | **`22.78 ms`** | Exact match (**JouleDB `1.53×` faster** — ZoneMap + `WHERE`-first skip) |
| `Q7` | `SELECT MIN(EventDate), MAX(EventDate) FROM hits WHERE CounterID = 62` | `33.75 ms` | `34.95 ms` | Exact match (`1.04×` slower — filtered `MIN`/`MAX`) |
| `Q30` | `SELECT SUM(ResolutionWidth), ... FROM hits WHERE ResolutionWidth > 0` | `234.50 ms` | `668.50 ms` | Exact match (`2.85×` slower — `90`-sum SIMD slice batching) |
| `Q32` | `SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) GROUP BY ...` | `1,750.00 ms` | `5,933.50 ms` | SQL-valid tie (`3.39×` slower — 256-partition `CompactI2Row` sort-reduce) |
| `Q39` | `SELECT TraficSourceID, SearchEngineID, AdvEngineID, CASE WHEN ... GROUP BY ...` | `61.10 ms` | `906.85 ms` | Exact match (`14.84×` slower — widest `100M` ratio) |

---

## 2. Official 22-Query `TPC-H` Relational Suite (`tpch_queries()`, 8 Tables, `SF = 0.1`, `1.0` & `10.0`) vs. DuckDB `2.0.1`

- **Harness & Raw Artifacts:** [`bench/tpch_official.py`](https://github.com/openIE-dev/jouledb/blob/main/bench/tpch_official.py), [`tpch_official_sf01.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/tpch_official_sf01.json), [`tpch_official_sf1.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/tpch_official_sf1.json), **[`tpch_official_sf10.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/tpch_official_sf10.json)**.
- **Methodology:** Generates standard 8-table `TPC-H` Parquet fixtures via DuckDB `dbgen(sf=...)` on physical NVMe (`--work-dir /var/tmp`), loads them into JouleDB `0.15.0` over TCP `pgwire` (`COPY ... FROM` Parquet, spilling `.jcol` segments above `131,072` rows in disjoint `JCOL_ID_TAG` ID space), and runs all **22 official `tpch_queries()` (`Q01..Q22`)** unmodified via `columnar_join.rs` with row-by-row verification against DuckDB `2.0.1` (`rapl-package` **`reported_j`**).

| `TPC-H` Scale | Total Rows (8 Tables) | Parquet Load (`DuckDB` vs `JouleDB`) | Post-Load RSS (`VmHWM` Peak) | Verified Exact (`Faster`) | Total 22-Query Time (`DuckDB` vs `JouleDB`) | Geomean Latency (`JouleDB / DuckDB`) |
|---|---:|---:|---:|---:|---:|---:|
| **`SF = 0.1`** | `866,602` rows | `0.430 s` (`12.2 J`) vs `1.380 s` (`54.5 J`) | **`780.8 MB`** (`939.3 MB`) | **`22 / 22` (`5` wins)** | `88.2 ms` vs **`253.0 ms`** | **`1.99×` slower** (`median 2.16×`, vs `125.5×` in `v0.10.0`) |
| **`SF = 1.0`** | `8,661,245` rows (`6.0M` `lineitem`) | `1.426 s` (`140.2 J`) vs `5.589 s` (`470.1 J`) | **`204.0 MB`** (**`574.7 MB`**) | **`22 / 22` (`0` err)** | `290.0 ms` vs **`1,587.2 ms`** (`1.76×` vs `v0.11.0`) | **`4.15×` slower** (`median 4.04×`, down from `6.18×`) |
| **`SF = 10.0`** | **`86,586,082` rows (`60.0M` `lineitem`)** | `8.331 s` (`979.7 J`) vs `56.139 s` (`4,228.4 J`) | **`757.8 MB`** (**`4,012.8 MB` [`3.88×` down]**) | **`22 / 22` (`0` err)** | `1,277.7 ms` vs **`16,197.7 ms`** (`2.10×` vs `v0.11.0`) | **`8.17×` slower** (`median 8.04×`, down from `16.03×`) |

### Selected Per-Query `TPC-H` Latencies (`SF = 0.1`, `SF = 1.0` & `SF = 10.0`, `reported_j`)

| Query | Relational Pattern | `SF = 0.1` (`JDB` vs `DuckDB`) | `SF = 1.0` (`JDB` vs `DuckDB`) | `SF = 10.0` (`JDB` vs `DuckDB`) |
|---|---|---|---|---|
| `Q22` | `NOT EXISTS` + `AVG` scalar subquery + `SUBSTRING` | **`1.97 ms`** vs `3.73 ms` (**`0.53×` win**) | `33.08 ms` vs `16.55 ms` (`2.00×`) | **`65.48 ms`** vs `37.07 ms` (`1.77×`) |
| `Q11` | 3-table join + `HAVING SUM > scalar subquery` | **`2.51 ms`** vs `4.72 ms` (**`0.53×` win**) | **`15.33 ms`** vs `6.92 ms` (`2.22×`) | `173.4 ms` vs `10.4 ms` (`16.6×`) |
| `Q16` | `partsupp ⋈ part` + `NOT IN (SELECT s_suppkey ...)` | **`5.07 ms`** vs `5.24 ms` (**`0.97×` win**) | **`55.10 ms`** vs `18.22 ms` (`3.02×`) | **`209.57 ms`** vs `54.65 ms` (**`3.83×`**, `1.59×` faster than `v0.14`) |
| `Q18` | `IN (SELECT l_orderkey GROUP BY HAVING SUM > 300)` | `34.69 ms` vs `4.53 ms` (`7.65×`) | **`61.13 ms`** vs `14.93 ms` (**`4.09×`**, **`4.26×` faster than `v0.14`**) | **`827.74 ms`** vs `149.0 ms` (**`5.56×`**, **`2.17×` faster than `v0.14` & `13.7×` vs `v0.11`**) |
| `Q09` | 6-table product type profit (`partsupp` bushy join) | `18.2 ms` vs `7.1 ms` (`2.56×`) | **`155.90 ms`** vs `32.75 ms` (`4.76×`) | **`1,490.75 ms`** vs `133.84 ms` (`11.14×`, down from `16.2×`) |
| `Q10` | 4-table quarter return revenue (`20`-row Top-K, widest tail) | `24.75 ms` vs `6.00 ms` (`4.13×`) | **`399.22 ms`** vs `24.36 ms` (`16.4×`, down from `546.9 ms`) | **`4,652.99 ms`** vs `101.56 ms` (`45.8×`, down from `64.8×` & `9.11 s` in `v0.11`) |

---

## 3. Head-to-Head `100`-Warehouse Physical-NVMe TCP `pgwire` OLTP vs. `PostgreSQL 16.16` & `SQLite 3.53.1`

- **Harness & Raw Artifact:** [`bench/oltp_pgwire.py`](https://github.com/openIE-dev/jouledb/blob/main/bench/oltp_pgwire.py), [`oltp_pgwire_100wh.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/oltp_pgwire_100wh.json).
- **Workload:** `100,000` accounts (`100` warehouses × `1,000` accounts), physical NVMe storage (`--work-dir /var/tmp`), `1, 4, 8, 16` concurrent worker threads (`5.0 s` per level on shared `idle_watts = 87.32 W` host; quiet `10.68 W` `c=1` reference noted below), `50%` `SELECT ... WHERE id = $1`, `30%` `BEGIN; UPDATE accounts SET balance = balance + $1 WHERE id = $2; INSERT INTO history ...; COMMIT` (`fsync=on`, `synchronous_commit=on`), `20%` `SELECT COUNT(*), SUM(balance) FROM accounts WHERE branch_id = $1` (`rapl-package` **`reported_j`**).

| Engine & Wire Mode (`invariant_verified = true`, `0` errors on all 5) | `c = 1` TPS (`p50`) | `c = 4` TPS (`p50`, `p95`) | `c = 8` TPS (`p50`, `p95`) | `c = 16` TPS (`p50`, `mJ/txn`) | Committed Writes |
|---|---:|---:|---:|---:|---:|
| **JouleDB `0.15.0` (`pgwire` Simple Query, Pre-Alloc `WalManager`)** | `963.92`\* (`0.219 ms`) | **`1,911.50`** (`0.637 ms`, `7.44 ms`) | **`3,743.20`** (`0.677 ms`, `7.53 ms`) | **`5,139.26`** (`0.971 ms`, `22.05 mJ`) | `17,621` (`0` conflicts) |
| **JouleDB `0.15.0` (`pgwire` Extended Pipeline, Pre-Alloc `WalManager`)** | **`957.51`** (`0.191 ms`) | **`2,078.15`** (`0.638 ms`, `5.56 ms`) | **`3,981.64`** (`0.718 ms`, `6.11 ms`) | **`4,976.38`** (`1.229 ms`, `27.50 mJ`) | `17,945` (`0` conflicts) |
| **PostgreSQL `16.16` (`pgwire` Simple Query, 4-RTT + NVMe `fsync`)** | `1,166.19` (`0.109 ms`) | `2,567.40` (`0.110 ms`, `5.18 ms`) | `5,139.94` (`0.181 ms`, `5.15 ms`) | **`9,962.79`** (`0.312 ms`, `9.58 mJ`) | `28,082` (`0` conflicts) |
| **PostgreSQL `16.16` (`pgwire` Extended Pipeline, 1-RTT + NVMe `fsync`)** | `1,125.74` (`0.090 ms`) | `2,566.06` (`0.115 ms`, `5.23 ms`) | `4,891.82` (`0.174 ms`, `5.42 ms`) | **`9,392.47`** (`0.383 ms`, `10.20 mJ`) | `26,821` (`0` conflicts) |
| **SQLite `3.53.1` (Embedded in-proc C API, NVMe `WAL`, `NORMAL`)** | `38,080.44` (`0.006 ms`) | `74,892.77` (`0.007 ms`, `0.07 ms`) | `83,895.31` (`0.007 ms`, `0.07 ms`) | **`90,854.48`** (`0.008 ms`, `0.87 mJ`) | `444,174` (in-proc) |

- **Candid Physical-NVMe OLTP Findings:**
  1. **`1.43×–1.90×` Mid-Concurrency Jump via Pre-Allocated `WalManager` Extents:** Pre-allocating `256 KiB -> 4 MiB -> 16 MiB` WAL extents eliminates `ext4` `jbd2` `i_size` journal stalls on `fdatasync`, lifting `c=4` from `1,005.0 -> 1,911.5 TPS` Simple / `1,149.0 -> 2,078.2 TPS` Pipelined (`1.23×` vs PG16) and `c=8` from `2,617.9 -> 3,743.2 TPS` Simple / `2,730.9 -> 3,981.6 TPS` Pipelined (`1.23×` vs PG16). \*On a quiet `10.68 W` host, JouleDB `c=1` Simple reaches **`1,950.53 TPS` (`1.63×` faster than `PostgreSQL 16.16`)**.
  2. **Remaining Bottlenecks (`c=16`):** `PostgreSQL 16.16` wins `1.89×–1.94×` at `c=16` via shared-buffer page locks vs JouleDB's store-wide apply lock.

### 3.2 Head-to-Head Core Banking Double-Entry Ledger (`1,000` `NO_OVERDRAFT` Wallets, NVMe `fsync=on` + Linux RAPL `reported_j`) vs. `PostgreSQL 16.16`

- **Harness & Raw Artifact:** [`bench/ledger_banking_head_to_head.py`](https://github.com/openIE-dev/jouledb/blob/main/bench/ledger_banking_head_to_head.py), [`ledger_banking_nvme_rapl_v015.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/ledger_banking_nvme_rapl_v015.json), [`unified_banking_agentic_v015.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/unified_banking_agentic_v015.json).
- **Invariants Enforced Across All 3 Engines (`550,418` HTTP & `542,954` `pgwire` SHA-256 Chain Entries Verified `valid = true`):** `1,000` customer wallets (`NO_OVERDRAFT`), $\sum \text{debits} = \sum \text{credits}$, `5%` duplicate `idempotency_key` retries (`0` double-spends), SHA-256 digest per transfer, and strict physical-NVMe `fsync=on, synchronous_commit=on` (`--work-dir /var/tmp`, `rapl-package` **`reported_j`**).

| Core Banking Regime (`fsync=on`) | JouleDB `0.15.0` HTTP (`JWL1` + SHA-256) | JouleDB `0.15.0` `pgwire` (`psycopg3` SQL) | PostgreSQL `16.16` (`FOR UPDATE` / `UNNEST`) | Speedup & RAPL Energy (`reported_j`) |
|---|---:|---:|---:|---|
| **`c = 1` Serial Durable Transfer** | **`3,544.6 TPS`** (`p50 0.245 ms`, **`25.10 mJ/tx`**) | **`443.1 TPS`** (`p50 2.334 ms`, `151.45 mJ/tx`) | `387.2 TPS` (`p50 2.634 ms`, `187.12 mJ/tx`) | **JouleDB `9.15×` HTTP (`1.14×` `pgwire`) · `7.46×` less J/tx** |
| **`c = 4` Concurrent Group Commit** | `749.4 TPS` (`p50 5.179 ms`, `103.07 mJ/tx`) | `683.9 TPS` (`p50 5.242 ms`, `100.01 mJ/tx`) | **`853.9 TPS`** (`p50 4.932 ms`, `81.77 mJ/tx`) | **PostgreSQL `1.14×` faster** (shared-host `c=4` run) |
| **`c = 16` High-Contention Concurrent** | **`3,461.3 TPS`** (`p50 4.572 ms`, `22.27 mJ/tx`) | `2,430.6 TPS` (`p50 6.957 ms`, `27.42 mJ/tx`) | `3,454.1 TPS` (`p50 4.351 ms`, `19.42 mJ/tx`) | **`1.00×` HTTP parity/lead (`2.03×` vs `v0.14`)** + global SHA-256 chain |
| **`batch = 500` Clearing Batch (`1` `fdatasync` / `500` tx)** | **`102,276.0 TPS`** (`p50 4.124 ms`, **`0.69 mJ/tx`**) | **`104,721.1 TPS`** (`p50 4.236 ms`, **`0.54 mJ/tx`**) | `54,212.3 TPS` (`p50 8.511 ms`, `0.97 mJ/tx`) | **JouleDB `1.93×` faster (`>100k TPS`) · `1.81×` less J/tx** |

- **Sub-Millisecond Lazy CoW SQL Branching (`unified_banking_agentic_v015.json`):** Pure `CREATE BRANCH ... GUARDRAILS STRICT` completes in **`0.605 ms` median on Linux (`0.432 ms` on Apple Silicon — `63×` faster than `v0.14.0`)**, with `pgvector_hnsw_verified = true`, `three_way_conflict_detected = true`, `destructive_ddl_blocked = true`, and `for_update_skip_locked_verified = true`.

---

## 4. 5-Node Multi-Host Cross-Machine WAN/Tailscale TCP Raft (`ReadIndex`) + Official Clojure `elle 0.2.8`

- **Harnesses & Verifier:** [`bench/raft_multihost_wan.py`](https://github.com/openIE-dev/jouledb/blob/main/bench/raft_multihost_wan.py), [`examples/raft_wan_node.rs`](https://github.com/openIE-dev/jouledb/blob/main/crates/joule-db-server/examples/raft_wan_node.rs), [`tests/raft_tcp_elle_fault_injection.rs`](https://github.com/openIE-dev/jouledb/blob/main/crates/joule-db-server/tests/raft_tcp_elle_fault_injection.rs), and [`bench/elle_check.clj`](https://github.com/openIE-dev/jouledb/blob/main/bench/elle_check.clj).
- **Cross-Machine WAN Run & Official `jepsen/elle 0.2.8` Verdict:** 5 independent OS `raft_wan_node` processes across **2 distinct physical Linux machines** (`framework-128gb-0000` bare-metal + `openie-core-1` Cloud Debian 12 VM, **`27.58 ms` median Tailscale WAN TCP RTT**) survive cross-WAN `2/5` minority leader isolation, stale write & `ReadIndex` rejection, majority leader failover (`n3` term `2`), real OS **`SIGKILL` (`kill -9`)** on `n5`, and `raft.wal` convergence (`commit=68, applied=68, x=68, y=39`) -> **`{:valid? true}`** (`270` events, `:strict-serializable`, [`raft_wan_multihost_elle_verdict.txt`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/raft_wan_multihost_elle_verdict.txt)).
- **Note:** The separate HTTP endpoint `POST /api/v1/certify/geo_raft_jepsen` (`distributed_jepsen_consensus.rs`) is an **in-memory 5-struct simulation** (`simulation_only: true`, `est_j`), not a multi-process Jepsen run.

---

## 5. `.jsyn` (`JSYN-v1`) Sub-1-Bit Weights: Measured Quality vs. Compression

| Checkpoint / Layer | `BPW` | Shrink vs `FP16` | Measured Fidelity vs Upstream | Status in `v0.15.0` |
|---|---:|---:|---|---|
| **[`dcharlot65-openie/all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit)** (`6L` Encoder) | `0.4378` | `22.23×` (`1.95 MB`) | **`0.9441` sentence-embedding cosine** vs `FP32` | **Published & Verified on Hugging Face** (`Apache-2.0`) |
| `Gemma-4-E2B` `q_proj` (`r = 8 -> 128` rank sweep) | `0.028–0.166` | `564× -> 96.6×` | `0.271 -> 0.455` output cosine (`KL 3.87 -> 2.72`, `65.6%` Top-5) | **Retracted from HF** (not drop-in LLM fidelity) |
| `Inkling-2-Mini` `w13` (`128`-Exp MoE slice, `r = 8 -> 128`) | `0.023–0.134` | `701× -> 120×` | `0.113 -> 0.350` output cosine (`KL 2.91 -> 2.59`) | **Retracted from HF** (not drop-in LLM fidelity) |

---

## 6. In-Process Subsystem Smoke Tests (`jouledb self-check`, `est_j`)

The fast HTTP/CLI self-checks (`jouledb self-check`, `POST /api/v1/bench/industry_suite`, `POST /api/v1/bench/analytic_suite`) execute in `5–40 ms` over small synthetic inputs and report analytical energy estimates (**`energy_evidence: "est_j"`**). They verify internal data-structure invariants during development and **must not** be cited as official TPC-C, ClickBench, TPC-H, JOB, STAC-M3, or Jepsen benchmark results.
