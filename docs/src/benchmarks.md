# Benchmarks, Consensus Verification & Subsystem Smoke Tests (`v0.12.0`)

> **Audit Remediation & Evidence Standard (`v0.12.0`):** Earlier `v0.2.8`–`v1.0.0` documentation presented 36 in-process micro-harnesses and synthetic simulations as *"36 / 36 Certified Global Benchmark Pillars"*, including synthetic frontier AI exam routes (`frontier_exam_routes.rs`). **All synthetic frontier AI benchmark claims remain retracted and `frontier_exam_routes.rs` has been deleted.**
>
> Below we strictly separate **(1) reproducible external dataset, multi-table `TPC-H` (`SF=0.1..10.0`), physical-NVMe multi-client `pgwire` OLTP, & multi-host WAN consensus evaluations** (with every win, loss, memory profile, and architectural bottleneck reported openly) from **(2) fast in-process functional smoke checks**, and tag every energy figure as **`reported_j`** (hardware RAPL / SMC counters) or **`est_j`** (analytical cost model).

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
- **Methodology:** Generates standard 8-table `TPC-H` Parquet fixtures via DuckDB `dbgen(sf=...)` on physical NVMe (`--work-dir /var/tmp`), loads them into JouleDB `0.12.0` over TCP `pgwire` (`COPY ... FROM` Parquet, spilling `.jcol` segments above `131,072` rows in disjoint `JCOL_ID_TAG` ID space), and runs all **22 official `tpch_queries()` (`Q01..Q22`)** unmodified via `columnar_join.rs` with row-by-row verification against DuckDB `2.0.1` (`rapl-package` **`reported_j`**).

| `TPC-H` Scale | Total Rows (8 Tables) | Parquet Load (`DuckDB` vs `JouleDB`) | Post-Load RSS (`VmHWM` Peak) | Verified Exact (`Faster`) | Total 22-Query Time (`DuckDB` vs `JouleDB`) | Geomean Latency (`JouleDB / DuckDB`) |
|---|---:|---:|---:|---:|---:|---:|
| **`SF = 0.1`** | `866,602` rows | `0.430 s` (`12.2 J`) vs `1.380 s` (`54.5 J`) | **`780.8 MB`** (`939.3 MB`) | **`22 / 22` (`5` wins)** | `88.2 ms` vs **`253.0 ms`** | **`1.99×` slower** (`median 2.16×`, vs `125.5×` in `v0.10.0`) |
| **`SF = 1.0`** | `8,661,245` rows (`6.0M` `lineitem`) | `1.142 s` (`64.6 J`) vs `5.504 s` (`288.5 J`) | **`201.4 MB`** (`1,561.8 MB`) | **`22 / 22` (`0` err)** | `253.0 ms` vs **`1,779.0 ms`** (`1.57×` vs `v0.11.0`) | **`4.33×` slower** (`median 3.60×`, down from `6.18×`) |
| **`SF = 10.0`** | **`86,586,082` rows (`60.0M` `lineitem`)** | `5.434 s` (`506.5 J`) vs `38.059 s` (`1,242.5 J`) | **`755.3 MB`** (`15,586.4 MB`) | **`22 / 22` (`0` err)** | `1,001.4 ms` vs **`15,464.6 ms`** (**`2.20×` vs `v0.11.0`**) | **`8.49×` slower** (`median 8.38×`, `20/22 < 1.2 s`) |

### Selected Per-Query `TPC-H` Latencies (`SF = 0.1`, `SF = 1.0` & `SF = 10.0`, `reported_j`)

| Query | Relational Pattern | `SF = 0.1` (`JDB` vs `DuckDB`) | `SF = 1.0` (`JDB` vs `DuckDB`) | `SF = 10.0` (`JDB` vs `DuckDB`) |
|---|---|---|---|---|
| `Q22` | `NOT EXISTS` + `AVG` scalar subquery + `SUBSTRING` | **`1.97 ms`** vs `3.73 ms` (**`0.53×` win**) | `33.08 ms` vs `16.55 ms` (`2.00×`) | **`65.48 ms`** vs `37.07 ms` (`1.77×`) |
| `Q11` | 3-table join + `HAVING SUM > scalar subquery` | **`2.51 ms`** vs `4.72 ms` (**`0.53×` win**) | `20.41 ms` vs `6.92 ms` (`2.95×`) | `173.4 ms` vs `10.4 ms` (`16.6×`) |
| `Q02` | 5-table join + correlated `MIN(ps_supplycost)` | **`2.39 ms`** vs `2.99 ms` (**`0.80×` win**) | `15.53 ms` vs `5.57 ms` (`2.79×`) | `572.1 ms` vs `21.8 ms` (`26.3×`) |
| `Q20` | Nested `IN` + correlated `0.5 * SUM(l_quantity)` | **`3.03 ms`** vs `3.66 ms` (**`0.83×` win**) | **`16.99 ms`** vs `10.41 ms` (`1.63×`) | **`99.73 ms`** vs `34.4 ms` (`2.90×`, `6.59×` vs `v0.11`) |
| `Q16` | `partsupp ⋈ part` + `NOT IN (SELECT s_suppkey ...)` | **`5.07 ms`** vs `5.24 ms` (**`0.97×` win**) | `61.71 ms` vs `17.97 ms` (`3.43×`) | `333.6 ms` vs `43.8 ms` (`7.61×`) |
| `Q04` | `orders` quarter filter + `EXISTS (lineitem)` | `3.80 ms` vs `2.23 ms` (`1.70×`) | **`11.09 ms`** vs `6.43 ms` (`1.73×`) | **`65.07 ms`** vs `41.7 ms` (`1.56×`, `3.47×` vs `v0.11`) |
| `Q17` | `lineitem ⋈ part` correlated `0.2 * AVG(l_quantity)` | `3.86 ms` vs `3.20 ms` (`1.21×`) | `21.63 ms` vs `8.36 ms` (`2.59×`) | **`88.65 ms`** vs `32.0 ms` (`2.77×`, **`17.2×` vs `v0.11`**) |
| `Q05` | 6-table cyclic `region..lineitem` join | `8.86 ms` vs `4.46 ms` (`1.99×`) | `37.43 ms` vs `8.76 ms` (`4.27×`) | `470.2 ms` vs `41.6 ms` (`11.3×`) |
| `Q08` | 8-table Snowflake join + `CASE` market share | `8.14 ms` vs `3.87 ms` (`2.10×`) | `35.11 ms` vs `9.30 ms` (`3.77×`) | `363.2 ms` vs `45.2 ms` (`8.03×`) |
| `Q21` | 4-table `EXISTS` + `NOT EXISTS` (`NeqSummary`) | `44.45 ms` vs `5.14 ms` (`8.65×`) | `196.55 ms` vs `16.52 ms` (`11.9×`) | `1,187.4 ms` vs `72.4 ms` (`16.4×`, `2.32×` vs `v0.11`) |
| `Q18` | `IN (SELECT l_orderkey GROUP BY HAVING SUM > 300)` | `34.69 ms` vs `4.53 ms` (`7.65×`) | `260.34 ms` vs `14.93 ms` (`17.4×`) | `1,799.8 ms` vs `99.1 ms` (`18.2×`, **`6.29×` vs `v0.11`**) |
| `Q10` | 4-table quarter return revenue (`20`-row Top-K, widest tail) | `24.75 ms` vs `6.00 ms` (`4.13×`) | `546.87 ms` vs `25.49 ms` (`21.5×`) | `5,230.7 ms` vs `80.7 ms` (`64.8×`, `1.74×` vs `v0.11`) |

---

## 3. Head-to-Head `100`-Warehouse Physical-NVMe TCP `pgwire` OLTP vs. `PostgreSQL 16.16` & `SQLite 3.53.1`

- **Harness & Raw Artifact:** [`bench/oltp_pgwire.py`](https://github.com/openIE-dev/jouledb/blob/main/bench/oltp_pgwire.py), [`oltp_pgwire_100wh.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/oltp_pgwire_100wh.json).
- **Workload:** `100,000` accounts (`100` warehouses × `1,000` accounts, initial `SUM(balance) = 100,000,000`), physical NVMe storage (`--work-dir /var/tmp`), `1, 4, 8, 16` concurrent worker threads (`8.0 s` per level), `50%` `SELECT ... WHERE id = $1`, `30%` `BEGIN; UPDATE accounts SET balance = balance + $1 WHERE id = $2; INSERT INTO history ...; COMMIT` (`fsync=on`, `synchronous_commit=on`), `20%` `SELECT COUNT(*), SUM(balance) FROM accounts WHERE branch_id = $1` (`rapl-package` **`reported_j`**, quiet host `idle_watts = 10.68 W`).

| Engine & Wire Mode (`invariant_verified = true`, `0` errors on all 5) | `c = 1` TPS (`p50`) | `c = 4` TPS (`p50`) | `c = 8` TPS (`p50`) | `c = 16` TPS (`p50`, `mJ/txn`) | Committed Writes |
|---|---:|---:|---:|---:|---:|
| **JouleDB `0.12.0` (`pgwire` Simple Query, 4-RTT + NVMe `fdatasync`)** | **`1,950.53`** (`0.148 ms`) | `1,005.03` (`0.666 ms`) | `2,617.89` (`0.640 ms`) | **`5,154.08`** (`0.718 ms`, `10.34 mJ`) | `25,677` (`0` conflicts) |
| **JouleDB `0.12.0` (`pgwire` Extended Pipeline, 1-RTT + NVMe `fdatasync`)** | `620.92` (`0.169 ms`) | `1,149.02` (`0.616 ms`) | `2,730.90` (`0.633 ms`) | **`5,159.73`** (`0.773 ms`, `11.71 mJ`) | `23,158` (`0` conflicts) |
| **PostgreSQL `16.16` (`pgwire` Simple Query, 4-RTT + NVMe `fsync`)** | `1,199.69` (`0.100 ms`) | `2,813.01` (`0.142 ms`) | `5,571.65` (`0.111 ms`) | **`10,419.39`** (`0.273 ms`, `4.14 mJ`) | `47,846` (`0` conflicts) |
| **PostgreSQL `16.16` (`pgwire` Extended Pipeline, 1-RTT + NVMe `fsync`)** | `1,171.69` (`0.071 ms`) | `2,555.25` (`0.100 ms`) | `5,046.95` (`0.167 ms`) | **`9,635.60`** (`0.231 ms`, `3.67 mJ`) | `44,020` (`0` conflicts) |
| **SQLite `3.53.1` (Embedded in-proc C API, NVMe `WAL`, `NORMAL`)** | `42,194.70` (`0.005 ms`) | `86,523.31` (`0.006 ms`) | `97,425.32` (`0.007 ms`) | **`109,483.53`** (`0.007 ms`, `0.36 mJ`) | `830,955` (in-proc) |

- **Candid Physical-NVMe OLTP Findings:**
  1. **JouleDB Beats `PostgreSQL 16.16` `1.63×` at `c=1` Simple (`1,950.5` vs `1,199.7 TPS`) & Scales to `5,159.7 TPS` at `c=16`:** On real NVMe `fdatasync` (`--work-dir /var/tmp`, vs `/tmp` `tmpfs` in `v0.11.0`), binary `JWU1` delta `UPDATE`s, deferred-hologram updates, and `GroupSync` narrow the `c=16` gap vs `PostgreSQL 16.16` from `2.91× / 2.61×` (`v0.11.0`) to **`2.02×` Simple / `1.87×` Pipelined**.
  2. **Remaining Bottlenecks (`c=4..16` & `c=1` Pipelined):** `PostgreSQL 16.16` wins `1.87×–2.80×` at `c=4..16` via shared-buffer page locks vs JouleDB's store-wide apply lock, and `GroupSync`'s batch coalescing window adds `~1.5 ms` wait when only 1 pipelined client is active (`620.9 TPS`).

---

## 4. 5-Node Multi-Host Cross-Machine WAN/Tailscale TCP Raft (`ReadIndex`) + Official Clojure `elle 0.2.8`

- **Harnesses & Verifier:** [`bench/raft_multihost_wan.py`](https://github.com/openIE-dev/jouledb/blob/main/bench/raft_multihost_wan.py), [`examples/raft_wan_node.rs`](https://github.com/openIE-dev/jouledb/blob/main/crates/joule-db-server/examples/raft_wan_node.rs), [`tests/raft_tcp_elle_fault_injection.rs`](https://github.com/openIE-dev/jouledb/blob/main/crates/joule-db-server/tests/raft_tcp_elle_fault_injection.rs), and [`bench/elle_check.clj`](https://github.com/openIE-dev/jouledb/blob/main/bench/elle_check.clj).
- **Cross-Machine WAN Run & Official `jepsen/elle 0.2.8` Verdict:** 5 independent OS `raft_wan_node` processes across **2 distinct physical Linux machines** (`framework-128gb-0000` bare-metal + `openie-core-1` Cloud Debian 12 VM, **`27.58 ms` median Tailscale WAN TCP RTT**) survive cross-WAN `2/5` minority leader isolation, stale write & `ReadIndex` rejection, majority leader failover (`n3` term `2`), real OS **`SIGKILL` (`kill -9`)** on `n5`, and `raft.wal` convergence (`commit=68, applied=68, x=68, y=39`) -> **`{:valid? true}`** (`270` events, `:strict-serializable`, [`raft_wan_multihost_elle_verdict.txt`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/raft_wan_multihost_elle_verdict.txt)).
- **Note:** The separate HTTP endpoint `POST /api/v1/certify/geo_raft_jepsen` (`distributed_jepsen_consensus.rs`) is an **in-memory 5-struct simulation** (`simulation_only: true`, `est_j`), not a multi-process Jepsen run.

---

## 5. `.jsyn` (`JSYN-v1`) Sub-1-Bit Weights: Measured Quality vs. Compression

| Checkpoint / Layer | `BPW` | Shrink vs `FP16` | Measured Fidelity vs Upstream | Status in `v0.12.0` |
|---|---:|---:|---|---|
| **[`dcharlot65-openie/all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit)** (`6L` Encoder) | `0.4378` | `22.23×` (`1.95 MB`) | **`0.9441` sentence-embedding cosine** vs `FP32` | **Published & Verified on Hugging Face** (`Apache-2.0`) |
| `Gemma-4-E2B` `q_proj` (`r = 8 -> 128` rank sweep) | `0.028–0.166` | `564× -> 96.6×` | `0.271 -> 0.455` output cosine (`KL 3.87 -> 2.72`, `65.6%` Top-5) | **Retracted from HF** (not drop-in LLM fidelity) |
| `Inkling-2-Mini` `w13` (`128`-Exp MoE slice, `r = 8 -> 128`) | `0.023–0.134` | `701× -> 120×` | `0.113 -> 0.350` output cosine (`KL 2.91 -> 2.59`) | **Retracted from HF** (not drop-in LLM fidelity) |

---

## 6. In-Process Subsystem Smoke Tests (`jouledb self-check`, `est_j`)

The fast HTTP/CLI self-checks (`jouledb self-check`, `POST /api/v1/bench/industry_suite`, `POST /api/v1/bench/analytic_suite`) execute in `5–40 ms` over small synthetic inputs and report analytical energy estimates (**`energy_evidence: "est_j"`**). They verify internal data-structure invariants during development and **must not** be cited as official TPC-C, ClickBench, TPC-H, JOB, STAC-M3, or Jepsen benchmark results.
