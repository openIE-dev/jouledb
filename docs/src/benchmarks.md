# Benchmarks, Consensus Verification & Subsystem Smoke Tests (`v0.10.0`)

> **Audit Remediation & Evidence Standard (`v0.10.0`):** Earlier `v0.2.8`–`v1.0.0` documentation presented 36 in-process micro-harnesses and synthetic simulations as *"36 / 36 Certified Global Benchmark Pillars"*, including synthetic frontier AI exam routes (`frontier_exam_routes.rs`). **All synthetic frontier AI benchmark claims remain retracted and `frontier_exam_routes.rs` has been deleted.**
>
> Below we strictly separate **(1) reproducible external dataset, multi-table `TPC-H`, multi-client `pgwire` OLTP, & multi-host WAN consensus evaluations** (with every win, loss, timeout, memory profile, and architectural bottleneck reported openly) from **(2) fast in-process functional smoke checks**, and tag every energy figure as **`reported_j`** (hardware RAPL / SMC counters) or **`est_j`** (analytical cost model).

---

## 1. Full Official ClickBench (`hits.parquet`, `99,997,497` Rows, `105` Columns, `43` Queries) vs. DuckDB `2.0.1`

- **Harness & Raw Artifacts:** [`bench/clickbench_official.py`](https://github.com/openIE-dev/jouledb/blob/main/bench/clickbench_official.py), [`clickbench_official_100k.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/clickbench_official_100k.json), [`clickbench_official_1m.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/clickbench_official_1m.json), [`clickbench_official_10m.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/clickbench_official_10m.json), **[`clickbench_official_100m.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/clickbench_official_100m.json)**.
- **Methodology:** Streams the official `14.78 GB` (`99,997,497`-row, `105`-column) ClickBench `hits.parquet` file into DuckDB `2.0.1` (in-process) and JouleDB `0.10.0` (over real TCP `pgwire` via `psycopg 3.3.9`, spilling immutable `262,144`-row `.jcol` segments to disk), then runs all **43 official ClickBench SQL queries (`Q0..Q42`)** unchanged on `framework-128gb-0000` (`AMD Ryzen AI Max+ 395`, 32 logical cores, `128 GB` UMA RAM, Linux `rapl-package` **`reported_j`**).

| Dataset Scale (`hits.parquet`) | Parquet Load (`DuckDB` vs `JouleDB`) | JouleDB Load RSS (`VmHWM` Peak) | Queries Succeeded | SQL-Valid Match (`Exact`) | Faster (`JouleDB` / `DuckDB`) | Geomean Latency (`JouleDB / DuckDB`) | Geomean Energy (`reported_j`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`100,000` rows** (`105` cols) | `0.179 s` (`11.2 J`) vs `0.265 s` (`18.3 J`) | `917.8 MB` (`2,651.5 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`34` exact) | **`20`** / `23` | **`1.47×` slower** (`median 1.19×`) | **`1.97×` more J** |
| **`1,000,000` rows** (`105` cols) | `0.158 s` (`15.8 J`) vs `0.871 s` (`77.7 J`) | `1,282.9 MB` (`1,922.8 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`29` exact) | **`11`** / `32` | **`2.34×` slower** (`median 2.95×`) | **`2.50×` more J** |
| **`10,000,000` rows** (`105` cols) | `2.422 s` (`336.4 J`) vs `13.340 s` (`1,850.8 J`) | `2,384.7 MB` (**`3,448.4 MB`**) | **`43 / 43`** (`0` err) | **`43 / 43`** (`31` exact) | **`9`** / `34` | **`2.28×` slower** (`median 2.63×`) | **`2.22×` more J** |
| **`99,997,497` rows — FULL `14.78 GB`** (`is_full_100m_dataset: true`) | `13.866 s` (`1,805.6 J`) vs `97.835 s` (`10,047.0 J`) | **`5,901.8 MB` (`16,469.7 MB`)** | **`43 / 43`** (`0` err) | **`43 / 43`** (`32` exact) | **`7`** / `36` | **`2.11×` slower** (`median 3.39×`) | **`2.03×` more J** |

### Selected Per-Query Results on the Full `99,997,497`-Row (`100M`) Dataset (`reported_j`)

| Query | SQL Summary | DuckDB `2.0.1` | JouleDB `0.10.0` (`pgwire` + `.jcol`) | Outcome |
|---|---|---:|---:|---|
| `Q0` | `SELECT COUNT(*) FROM hits` | `5.67 ms` | **`1.91 ms`** | Exact match (**JouleDB `2.98×` faster** — segment metadata) |
| `Q1` | `SELECT COUNT(*) FROM hits WHERE AdvEngineID <> 0` | `41.65 ms` | **`20.23 ms`** | Exact match (**JouleDB `2.06×` faster** — `WordMask` bitpacked selection) |
| `Q2` | `SELECT SUM(AdvEngineID), COUNT(*), AVG(ResolutionWidth) FROM hits` | `61.55 ms` | **`2.41 ms`** | Exact match (**JouleDB `25.5×` faster** — zero-I/O `global_int_sum`) |
| `Q3` | `SELECT AVG(UserID) FROM hits` | `51.82 ms` | **`2.32 ms`** | Exact match (**JouleDB `22.4×` faster** — zero-I/O `global_int_sum`) |
| `Q6` | `SELECT MIN(EventDate), MAX(EventDate) FROM hits` | `22.07 ms` | **`1.58 ms`** | Exact match (**JouleDB `14.0×` faster** — segment ZoneMap) |
| `Q29` | `SELECT SUM(ResolutionWidth), SUM(ResolutionWidth + 1), ...` (`90` sums) | `56.20 ms` | **`2.40 ms`** | Exact match (**JouleDB `23.4×` faster** — algebraic `global_int_sum`) |
| `Q42` | `SELECT DATE_TRUNC('minute', EventTime), COUNT(*) WHERE CounterID = 62 ...` | `34.83 ms` | **`22.78 ms`** | Exact match (**JouleDB `1.53×` faster** — ZoneMap + `WHERE`-first skip) |
| `Q7` | `SELECT MIN(EventDate), MAX(EventDate) FROM hits WHERE CounterID = 62` | `33.75 ms` | `34.95 ms` | Exact match (`1.04×` slower — **cut from `10.71×` in `v0.9.0`** via filtered `MIN`/`MAX`) |
| `Q30` | `SELECT SUM(ResolutionWidth), ... FROM hits WHERE ResolutionWidth > 0` | `234.50 ms` | `668.50 ms` | Exact match (`2.85×` slower — **cut from `14.26×` in `v0.9.0`** via `90`-sum SIMD slice batching) |
| `Q20` | `SELECT UserID FROM hits WHERE URL LIKE '%google%' LIMIT 1` | `435.60 ms` | `1,543.30 ms` | SQL-valid (`3.54×` slower — **cut from `6.75×` in `v0.9.0`** via dictionary `LIKE` bitset) |
| `Q32` | `SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) GROUP BY ...` | `1,750.00 ms` | `5,933.50 ms` | SQL-valid tie (`3.39×` slower — 256-partition `CompactI2Row` sort-reduce) |
| `Q39` | `SELECT TraficSourceID, SearchEngineID, AdvEngineID, CASE WHEN ... GROUP BY ...` | `61.10 ms` | `906.85 ms` | Exact match (`14.84×` slower — widest `100M` ratio, narrowed from `22.36×` in `v0.9.0`) |

---

## 2. Official 22-Query `TPC-H` Relational Suite (`tpch_queries()`, 8 Tables) vs. DuckDB `2.0.1`

- **Harness & Raw Artifacts:** [`bench/tpch_official.py`](https://github.com/openIE-dev/jouledb/blob/main/bench/tpch_official.py), [`tpch_official_sf01.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/tpch_official_sf01.json), [`tpch_official_sf1.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/tpch_official_sf1.json).
- **Methodology:** Generates standard 8-table `TPC-H` Parquet fixtures via DuckDB `dbgen(sf=...)`, loads them into JouleDB `0.10.0` over TCP `pgwire` (`COPY ... FROM` Parquet), and runs all **22 official `tpch_queries()` (`Q01..Q22`)** unmodified with row-by-row numeric/text verification against DuckDB `2.0.1` (`rapl-package` **`reported_j`**).

| `TPC-H` Scale | Total Rows (8 Tables) | Parquet Load (`DuckDB` vs `JouleDB`) | Verified Exact Matches | Selected `JouleDB 0.10.0` Query Latencies | Candid Multi-Join Gap vs. DuckDB `2.0.1` |
|---|---:|---:|---:|---|---|
| **`SF = 0.1`** | `866,602` rows | `0.423 s` (`49.2 J`) vs `4.138 s` (`533.4 J`) | **`22 / 22` (`100%`, `0` err)** | `Q06: 5.00 ms`, `Q15: 5.65 ms`, `Q01: 62.57 ms`, `Q10: 144.56 ms`, `Q14: 219.56 ms`, `Q13: 245.66 ms`, `Q22: 258.24 ms`, `Q19: 301.55 ms` | All 22 queries exact (`3 / 22` faster at `SF=0.01`: `Q01, Q06, Q15`); DuckDB `125.50×` faster geomean (`132.8 ms` vs `49.59 s` total) |
| **`SF = 1.0`** | `8,661,245` rows (`6.0M` `lineitem`) | `2.076 s` (`277.1 J`) vs **`9.816 s` (`1,202.3 J`)** | **`13 / 22` exact** (`8` `25s` timeout, `1` `36 GB` RSS) | `Q15: 70.83 ms`, `Q01: 84.18 ms`, `Q06: 336.84 ms`, `Q14: 2,405.5 ms`, `Q20: 2,797.2 ms`, `Q03: 2,871.6 ms`, `Q19: 3,393.5 ms`, `Q10: 3,496.5 ms` | `112.75×` geomean across `13` completed queries; `Q04,Q07-09,Q12,Q17-18,Q21` exceed `25 s` & `Q05` trips `36 GB` RSS cap on `Vec<Value>` joins |

---

## 3. Head-to-Head `100`-Warehouse TCP `pgwire` OLTP vs. `PostgreSQL 16.16` & `SQLite 3.53.1`

- **Harness & Raw Artifact:** [`bench/oltp_pgwire.py`](https://github.com/openIE-dev/jouledb/blob/main/bench/oltp_pgwire.py), [`oltp_pgwire_100wh.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/oltp_pgwire_100wh.json).
- **Workload:** `100,000` accounts (`100` warehouses × `1,000` accounts, initial `SUM(balance) = 100,000,000`), `1, 4, 8, 16` concurrent worker threads (`8.0 s` per level), `50%` `SELECT ... WHERE id = $1`, `30%` `BEGIN; UPDATE accounts SET balance = balance + $1 WHERE id = $2; INSERT INTO history ...; COMMIT` (`fsync=on`, `synchronous_commit=on`), `20%` `SELECT COUNT(*), SUM(balance) FROM accounts WHERE branch_id = $1` (`rapl-package` **`reported_j`**, measured back-to-back under shared host CI load `idle_watts = 145.12 W`).

| Engine & Wire Mode (`invariant_verified = true` on all 5) | `c = 1` TPS (`p50`) | `c = 4` TPS (`p50`) | `c = 8` TPS (`p50`) | `c = 16` TPS (`p50`, `mJ/txn`) | Committed Writes |
|---|---:|---:|---:|---:|---:|
| **JouleDB `0.10.0` (`pgwire` Simple Query, 4-RTT + `fsync`)** | `721.59` (`1.208 ms`) | **`1,439.95`** (`1.682 ms`) | `1,358.36` (`2.119 ms`) | `1,351.57` (`2.748 ms`, `101.6 mJ`) | `11,537` (`0` errors) |
| **JouleDB `0.10.0` (`pgwire` Extended Pipeline, 1-RTT + `fsync`)** | `381.88` (`2.264 ms`) | `1,105.94` (`2.423 ms`) | **`1,370.54`** (`2.664 ms`) | **`1,375.67`** (`2.909 ms`, `99.5 mJ`) | `10,044` (`1` SI conflict) |
| **PostgreSQL `16.16` (`pgwire` Simple Query, 4-RTT + `fsync`)** | `836.71` (`1.073 ms`) | `2,132.16` (`1.383 ms`) | `4,073.87` (`1.069 ms`) | **`5,744.85`** (`1.050 ms`, `22.8 mJ`) | `30,605` (`0` conflicts) |
| **PostgreSQL `16.16` (`pgwire` Extended Pipeline, 1-RTT + `fsync`)** | `896.62` (`0.614 ms`) | `2,029.98` (`1.121 ms`) | `3,404.61` (`1.442 ms`) | **`4,720.52`** (`1.908 ms`, `27.4 mJ`) | `26,462` (`0` conflicts) |
| **SQLite `3.53.1` (Embedded in-proc C API, `WAL`, `NORMAL`)** | `19,826.03` (`0.008 ms`) | `33,279.64` (`0.011 ms`) | `34,769.44` (`0.015 ms`) | **`36,085.06`** (`0.018 ms`, `3.49 mJ`) | `303,731` (in-proc) |

- **Candid OLTP Findings:**
  1. **Extended Query Pipeline Parity (`1,375.67 TPS` Pipelined vs `1,351.57 TPS` Simple at `c=16`, `1.02×` vs `0.51×` in `v0.9.0`):** Caching `Describe` rows at `Parse`, binding `portal.params` directly without SQL string re-formatting, thread-local SQL AST caching, and binary `JMVR\x01` `MvccRecord` serialization eliminated the `1.97×` Extended Pipeline overhead (note: on an uncontended host in `v0.9.0`, Simple reached `3,351.04 TPS`).
  2. **`PostgreSQL 16.16` Wins Multi-Client Scaling (`4.18×` Higher TPS at `c=16`):** While `c=1` Simple is within **`1.16×`** (`721.6` vs `836.7 TPS`), PostgreSQL's shared-buffer page locks outscale JouleDB's table-wide `commit_lock` at `c=16` (`5,744.9` vs `1,375.7 TPS`).

---

## 4. 5-Node Multi-Host Cross-Machine WAN/Tailscale TCP Raft (`ReadIndex`) + Official Clojure `elle 0.2.8`

- **Harnesses & Verifier:** [`bench/raft_multihost_wan.py`](https://github.com/openIE-dev/jouledb/blob/main/bench/raft_multihost_wan.py), [`examples/raft_wan_node.rs`](https://github.com/openIE-dev/jouledb/blob/main/crates/joule-db-server/examples/raft_wan_node.rs), [`tests/raft_tcp_elle_fault_injection.rs`](https://github.com/openIE-dev/jouledb/blob/main/crates/joule-db-server/tests/raft_tcp_elle_fault_injection.rs), and [`bench/elle_check.clj`](https://github.com/openIE-dev/jouledb/blob/main/bench/elle_check.clj).
- **Cross-Machine WAN Run & Official `jepsen/elle 0.2.8` Verdict:** 5 independent OS `raft_wan_node` processes across **2 distinct physical Linux machines** (`framework-128gb-0000` bare-metal + `openie-core-1` Cloud Debian 12 VM, **`27.58 ms` median Tailscale WAN TCP RTT**) survive cross-WAN `2/5` minority leader isolation, stale write & `ReadIndex` rejection, majority leader failover (`n3` term `2`), real OS **`SIGKILL` (`kill -9`)** on `n5`, and `raft.wal` convergence (`commit=68, applied=68, x=68, y=39`) -> **`{:valid? true}`** (`270` events, `:strict-serializable`, [`raft_wan_multihost_elle_verdict.txt`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/raft_wan_multihost_elle_verdict.txt)).
- **Note:** The separate HTTP endpoint `POST /api/v1/certify/geo_raft_jepsen` (`distributed_jepsen_consensus.rs`) is an **in-memory 5-struct simulation** (`simulation_only: true`, `est_j`), not a multi-process Jepsen run.

---

## 5. `.jsyn` (`JSYN-v1`) Sub-1-Bit Weights: Measured Quality vs. Compression

| Checkpoint / Layer | `BPW` | Shrink vs `FP16` | Measured Fidelity vs Upstream | Status in `v0.10.0` |
|---|---:|---:|---|---|
| **[`dcharlot65-openie/all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit)** (`6L` Encoder) | `0.4378` | `22.23×` (`1.95 MB`) | **`0.9441` sentence-embedding cosine** vs `FP32` | **Published & Verified on Hugging Face** (`Apache-2.0`) |
| `Gemma-4-E2B` `q_proj` (`r = 8 -> 128` rank sweep) | `0.028–0.166` | `564× -> 96.6×` | `0.271 -> 0.455` output cosine (`KL 3.87 -> 2.72`, `65.6%` Top-5) | **Retracted from HF** (not drop-in LLM fidelity) |
| `Inkling-2-Mini` `w13` (`128`-Exp MoE slice, `r = 8 -> 128`) | `0.023–0.134` | `701× -> 120×` | `0.113 -> 0.350` output cosine (`KL 2.91 -> 2.59`) | **Retracted from HF** (not drop-in LLM fidelity) |

---

## 6. In-Process Subsystem Smoke Tests (`jouledb self-check`, `est_j`)

The fast HTTP/CLI self-checks (`jouledb self-check`, `POST /api/v1/bench/industry_suite`, `POST /api/v1/bench/analytic_suite`) execute in `5–40 ms` over small synthetic inputs and report analytical energy estimates (**`energy_evidence: "est_j"`**). They verify internal data-structure invariants during development and **must not** be cited as official TPC-C, ClickBench, TPC-H, JOB, STAC-M3, or Jepsen benchmark results.
