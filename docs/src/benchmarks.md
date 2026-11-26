# Benchmarks, Consensus Verification & Subsystem Smoke Tests (`v0.11.0`)

> **Audit Remediation & Evidence Standard (`v0.11.0`):** Earlier `v0.2.8`–`v1.0.0` documentation presented 36 in-process micro-harnesses and synthetic simulations as *"36 / 36 Certified Global Benchmark Pillars"*, including synthetic frontier AI exam routes (`frontier_exam_routes.rs`). **All synthetic frontier AI benchmark claims remain retracted and `frontier_exam_routes.rs` has been deleted.**
>
> Below we strictly separate **(1) reproducible external dataset, multi-table `TPC-H` (`SF=0.1..10.0`), multi-client `pgwire` OLTP, & multi-host WAN consensus evaluations** (with every win, loss, memory profile, and architectural bottleneck reported openly) from **(2) fast in-process functional smoke checks**, and tag every energy figure as **`reported_j`** (hardware RAPL / SMC counters) or **`est_j`** (analytical cost model).

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
- **Methodology:** Generates standard 8-table `TPC-H` Parquet fixtures via DuckDB `dbgen(sf=...)`, loads them into JouleDB `0.11.0` over TCP `pgwire` (`COPY ... FROM` Parquet, spilling `.jcol` segments above `262,144` rows), and runs all **22 official `tpch_queries()` (`Q01..Q22`)** unmodified via `columnar_join.rs` with row-by-row numeric/text verification against DuckDB `2.0.1` (`rapl-package` **`reported_j`**).

| `TPC-H` Scale | Total Rows (8 Tables) | Parquet Load (`DuckDB` vs `JouleDB`) | Peak `VmHWM` | Verified Exact (`Faster`) | Total 22-Query Time (`DuckDB` vs `JouleDB`) | Geomean Latency (`JouleDB / DuckDB`) |
|---|---:|---:|---:|---:|---:|---:|
| **`SF = 0.1`** | `866,602` rows | `0.328 s` (`13.8 J`) vs `1.979 s` (`135.3 J`) | `2,204.7 MB` | **`22 / 22` (`3` wins)** | `99.5 ms` vs **`318.0 ms`** | **`2.22×` slower** (`median 1.92×`, vs `125.5×` in `v0.10.0`) |
| **`SF = 1.0`** | `8,661,245` rows (`6.0M` `lineitem`) | `1.099 s` (`70.9 J`) vs **`4.884 s` (`256.0 J`)** | **`6,303.2 MB`** | **`22 / 22` (`1` win)** | `239.4 ms` vs **`2,787.1 ms`** | **`6.18×` slower** (`median 6.39×`, vs `13/22` in `v0.10.0`) |
| **`SF = 10.0`** | **`86,586,082` rows (`60.0M` `lineitem`)** | `4.964 s` (`530.4 J`) vs **`27.813 s` (`2,026.3 J`)** | `40,031.2 MB` | **`22 / 22` (`0` err)** | `979.8 ms` vs **`34,060.0 ms`** | **`16.03×` slower** (`median 12.44×`, `18/22 < 1.0 s`) |

### Selected Per-Query `TPC-H` Latencies (`SF = 0.1`, `SF = 1.0` & `SF = 10.0`, `reported_j`)

| Query | Relational Pattern | `SF = 0.1` (`JDB` vs `DuckDB`) | `SF = 1.0` (`JDB` vs `DuckDB`) | `SF = 10.0` (`JDB` vs `DuckDB`) |
|---|---|---|---|---|
| `Q22` | `NOT EXISTS` + `AVG` scalar subquery + `SUBSTRING` | **`1.31 ms`** vs `4.27 ms` (**`0.31×` win**) | **`12.59 ms`** vs `17.12 ms` (**`0.74×` win**) | `138.0 ms` vs `32.9 ms` (`4.19×`) |
| `Q11` | 3-table join + `HAVING SUM > scalar subquery` | **`2.46 ms`** vs `4.40 ms` (**`0.56×` win**) | `14.99 ms` vs `4.44 ms` (`3.38×`) | `189.0 ms` vs `8.32 ms` (`22.7×`) |
| `Q20` | Nested `IN` + correlated `0.5 * SUM(l_quantity)` | **`3.54 ms`** vs `5.66 ms` (**`0.63×` win**) | `75.06 ms` vs `10.61 ms` (`7.07×`) | `657.2 ms` vs `31.8 ms` (`20.6×`) |
| `Q06` | Single-table `lineitem` range filter + `SUM` | `3.86 ms` vs `1.05 ms` (`3.67×`) | `9.80 ms` vs `1.51 ms` (`6.49×`) | `74.38 ms` vs `9.85 ms` (`7.55×`) |
| `Q14` | `lineitem ⋈ part` promo `CASE` ratio | `5.25 ms` vs `2.19 ms` (`2.40×`) | `16.81 ms` vs `7.67 ms` (`2.19×`) | `176.0 ms` vs `31.4 ms` (`5.61×`) |
| `Q17` | `lineitem ⋈ part` correlated `0.2 * AVG(l_quantity)` | `5.13 ms` vs `3.59 ms` (`1.43×`) | `15.00 ms` vs `6.83 ms` (`2.20×`) | `1,528.8 ms` vs `32.6 ms` (`46.9×`) |
| `Q05` | 6-table cyclic `region..lineitem` join (`36 GB` OOM in `v0.10.0`) | `10.66 ms` vs `5.57 ms` (`1.92×`) | **`61.83 ms`** vs `8.80 ms` (`7.03×`) | `790.2 ms` vs `40.8 ms` (`19.4×`) |
| `Q08` | 8-table Snowflake join + `CASE` market share (`>25 s` in `v0.10.0`) | `8.64 ms` vs `4.79 ms` (`1.80×`) | **`52.48 ms`** vs `9.83 ms` (`5.34×`) | `476.4 ms` vs `40.1 ms` (`11.9×`) |
| `Q13` | `customer LEFT OUTER JOIN orders` derived `GROUP BY` | `19.50 ms` vs `10.19 ms` (`1.91×`) | `98.46 ms` vs `22.29 ms` (`4.42×`) | `745.3 ms` vs `104.0 ms` (`7.16×`) |
| `Q21` | 4-table `EXISTS` + `NOT EXISTS` (`NeqSummary`, `>25 s` in `v0.10.0`) | `28.71 ms` vs `6.08 ms` (`4.72×`) | `195.74 ms` vs `14.93 ms` (`13.1×`) | `2,759.6 ms` vs `67.2 ms` (`41.1×`) |
| `Q18` | `IN (SELECT l_orderkey GROUP BY HAVING SUM > 300)` (widest tail) | `74.08 ms` vs `4.95 ms` (`15.0×`) | `874.07 ms` vs `15.30 ms` (`57.1×`) | `11,328.8 ms` vs `119.5 ms` (`94.8×`) |

---

## 3. Head-to-Head `100`-Warehouse TCP `pgwire` OLTP vs. `PostgreSQL 16.16` & `SQLite 3.53.1`

- **Harness & Raw Artifact:** [`bench/oltp_pgwire.py`](https://github.com/openIE-dev/jouledb/blob/main/bench/oltp_pgwire.py), [`oltp_pgwire_100wh.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/oltp_pgwire_100wh.json).
- **Workload:** `100,000` accounts (`100` warehouses × `1,000` accounts, initial `SUM(balance) = 100,000,000`), `1, 4, 8, 16` concurrent worker threads (`8.0 s` per level), `50%` `SELECT ... WHERE id = $1`, `30%` `BEGIN; UPDATE accounts SET balance = balance + $1 WHERE id = $2; INSERT INTO history ...; COMMIT` (`fsync=on`, `synchronous_commit=on`), `20%` `SELECT COUNT(*), SUM(balance) FROM accounts WHERE branch_id = $1` (`rapl-package` **`reported_j`**, quiet host `idle_watts = 10.96 W`).

| Engine & Wire Mode (`invariant_verified = true` on all 5) | `c = 1` TPS (`p50`) | `c = 4` TPS (`p50`) | `c = 8` TPS (`p50`) | `c = 16` TPS (`p50`, `mJ/txn`) | Committed Writes |
|---|---:|---:|---:|---:|---:|
| **JouleDB `0.11.0` (`pgwire` Simple Query, 4-RTT + `fsync`)** | `4,094.66` (`0.134 ms`) | **`5,797.01`** (`0.658 ms`) | `5,376.65` (`0.997 ms`) | **`5,038.38`** (`1.370 ms`, `15.93 mJ`) | `48,473` (`1` SI conflict) |
| **JouleDB `0.11.0` (`pgwire` Extended Pipeline, 1-RTT + `fsync`)** | `4,020.27` (`0.150 ms`) | **`5,205.58`** (`0.780 ms`) | `5,186.25` (`1.056 ms`) | **`4,990.75`** (`1.384 ms`, `24.55 mJ`) | `46,337` (`1` SI conflict) |
| **PostgreSQL `16.16` (`pgwire` Simple Query, 4-RTT + `fsync`)** | `17,466.14` (`0.064 ms`) | `35,913.17` (`0.097 ms`) | `22,680.27` (`0.247 ms`) | **`14,676.40`** (`0.723 ms`, `5.18 mJ`) | `217,684` (`0` conflicts) |
| **PostgreSQL `16.16` (`pgwire` Extended Pipeline, 1-RTT + `fsync`)** | `23,516.32` (`0.041 ms`) | `26,620.99` (`0.132 ms`) | `18,251.52` (`0.347 ms`) | **`13,029.46`** (`0.975 ms`, `5.66 mJ`) | `195,399` (`0` conflicts) |
| **SQLite `3.53.1` (Embedded in-proc C API, `WAL`, `NORMAL`)** | `90,161.56` (`0.005 ms`) | `104,040.70` (`0.005 ms`) | `109,016.55` (`0.006 ms`) | **`127,046.29`** (`0.006 ms`, `0.44 mJ`) | `1,035,821` (in-proc) |

- **Candid OLTP Findings:**
  1. **`4.03×`–`10.53×` Higher JouleDB Throughput (`5,797.0 TPS` at `c=4`, `0.134 ms` `c=1` `p50`):** 64-shard `MvccStore`/`MvccTableStorage`, `SchemaMetaCache`, `O(1)`/`O(k)` direct indexed `SELECT`/`COUNT`/`SUM`, adaptive `BTreeSet` `ValueIds`, and deferred-hologram binary `JWB1` `batch_ingest_fields` lift JouleDB to **`5,797.01 TPS` (`c=4`)** and **`5,038.38 TPS` Simple / `4,990.75 TPS` Pipelined (`c=16`)**.
  2. **`PostgreSQL 16.16` Remains `2.61×`–`6.19×` Faster:** Narrowed at `c=16` from `4.18×` (`v0.10.0`) to **`2.91×` Simple (`2.61×` Pipelined)**, while PostgreSQL's shared-buffer page locks still outscale JouleDB's store-wide write lock at `c=4` (`35,913` vs `5,797 TPS`).

---

## 4. 5-Node Multi-Host Cross-Machine WAN/Tailscale TCP Raft (`ReadIndex`) + Official Clojure `elle 0.2.8`

- **Harnesses & Verifier:** [`bench/raft_multihost_wan.py`](https://github.com/openIE-dev/jouledb/blob/main/bench/raft_multihost_wan.py), [`examples/raft_wan_node.rs`](https://github.com/openIE-dev/jouledb/blob/main/crates/joule-db-server/examples/raft_wan_node.rs), [`tests/raft_tcp_elle_fault_injection.rs`](https://github.com/openIE-dev/jouledb/blob/main/crates/joule-db-server/tests/raft_tcp_elle_fault_injection.rs), and [`bench/elle_check.clj`](https://github.com/openIE-dev/jouledb/blob/main/bench/elle_check.clj).
- **Cross-Machine WAN Run & Official `jepsen/elle 0.2.8` Verdict:** 5 independent OS `raft_wan_node` processes across **2 distinct physical Linux machines** (`framework-128gb-0000` bare-metal + `openie-core-1` Cloud Debian 12 VM, **`27.58 ms` median Tailscale WAN TCP RTT**) survive cross-WAN `2/5` minority leader isolation, stale write & `ReadIndex` rejection, majority leader failover (`n3` term `2`), real OS **`SIGKILL` (`kill -9`)** on `n5`, and `raft.wal` convergence (`commit=68, applied=68, x=68, y=39`) -> **`{:valid? true}`** (`270` events, `:strict-serializable`, [`raft_wan_multihost_elle_verdict.txt`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/raft_wan_multihost_elle_verdict.txt)).
- **Note:** The separate HTTP endpoint `POST /api/v1/certify/geo_raft_jepsen` (`distributed_jepsen_consensus.rs`) is an **in-memory 5-struct simulation** (`simulation_only: true`, `est_j`), not a multi-process Jepsen run.

---

## 5. `.jsyn` (`JSYN-v1`) Sub-1-Bit Weights: Measured Quality vs. Compression

| Checkpoint / Layer | `BPW` | Shrink vs `FP16` | Measured Fidelity vs Upstream | Status in `v0.11.0` |
|---|---:|---:|---|---|
| **[`dcharlot65-openie/all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit)** (`6L` Encoder) | `0.4378` | `22.23×` (`1.95 MB`) | **`0.9441` sentence-embedding cosine** vs `FP32` | **Published & Verified on Hugging Face** (`Apache-2.0`) |
| `Gemma-4-E2B` `q_proj` (`r = 8 -> 128` rank sweep) | `0.028–0.166` | `564× -> 96.6×` | `0.271 -> 0.455` output cosine (`KL 3.87 -> 2.72`, `65.6%` Top-5) | **Retracted from HF** (not drop-in LLM fidelity) |
| `Inkling-2-Mini` `w13` (`128`-Exp MoE slice, `r = 8 -> 128`) | `0.023–0.134` | `701× -> 120×` | `0.113 -> 0.350` output cosine (`KL 2.91 -> 2.59`) | **Retracted from HF** (not drop-in LLM fidelity) |

---

## 6. In-Process Subsystem Smoke Tests (`jouledb self-check`, `est_j`)

The fast HTTP/CLI self-checks (`jouledb self-check`, `POST /api/v1/bench/industry_suite`, `POST /api/v1/bench/analytic_suite`) execute in `5–40 ms` over small synthetic inputs and report analytical energy estimates (**`energy_evidence: "est_j"`**). They verify internal data-structure invariants during development and **must not** be cited as official TPC-C, ClickBench, TPC-H, JOB, STAC-M3, or Jepsen benchmark results.
