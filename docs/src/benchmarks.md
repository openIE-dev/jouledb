# Benchmarks, Consensus Verification & Subsystem Smoke Tests (`v0.8.0`)

> **Audit Remediation & Evidence Standard (`v0.8.0`):** Earlier `v0.2.8`–`v1.0.0` documentation presented 36 in-process micro-harnesses and synthetic simulations as *"36 / 36 Certified Global Benchmark Pillars"*, including synthetic frontier AI exam routes (`frontier_exam_routes.rs`). **All synthetic frontier AI benchmark claims remain retracted and `frontier_exam_routes.rs` has been deleted.**
>
> Below we strictly separate **(1) reproducible external dataset, multi-client `pgwire`, & multi-socket evaluations** (with every win, loss, memory profile, and architectural bottleneck reported openly) from **(2) fast in-process functional smoke checks**, and tag every energy figure as **`reported_j`** (hardware RAPL / SMC counters) or **`est_j`** (analytical cost model).

---

## 1. Full Official ClickBench (`hits.parquet`, `99,997,497` Rows, `105` Columns, `43` Queries) vs. DuckDB `2.0.1`

- **Harness & Raw Artifacts:** [`bench/clickbench_official.py`](https://github.com/openIE-dev/jouledb/blob/main/bench/clickbench_official.py), [`clickbench_official_100k.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/clickbench_official_100k.json), [`clickbench_official_1m.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/clickbench_official_1m.json), [`clickbench_official_10m.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/clickbench_official_10m.json), **[`clickbench_official_100m.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/clickbench_official_100m.json)**.
- **Methodology:** Streams the official `14.78 GB` (`99,997,497`-row, `105`-column) ClickBench `hits.parquet` file into DuckDB `2.0.1` (in-process) and JouleDB `0.8.0` (over real TCP `pgwire` via `psycopg 3.3.9`, spilling immutable `262,144`-row `.jcol` segments to disk), then runs all **43 official ClickBench SQL queries (`Q0..Q42`)** unchanged on `framework-128gb-0000` (`AMD Ryzen AI Max+ 395`, 32 logical cores, `128 GB` UMA RAM, Linux `rapl-package` **`reported_j`**).

| Dataset Scale (`hits.parquet`) | Parquet Load (`DuckDB` vs `JouleDB`) | JouleDB Load RSS (`VmHWM` Peak) | Queries Succeeded | SQL-Valid Match (`Exact`) | Faster (`JouleDB` / `DuckDB`) | Geomean Latency (`JouleDB / DuckDB`) | Geomean Energy (`reported_j`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`100,000` rows** (`105` cols) | `0.197 s` (`23.6 J`) vs `0.355 s` (`41.7 J`) | `933.8 MB` | **`43 / 43`** (`0` err) | **`43 / 43`** (`35` exact) | **`25`** / `18` | **`1.30×` slower** (**`median 0.80×`**) | **`1.53×` more J** |
| **`1,000,000` rows** (`105` cols) | `0.147 s` (`14.8 J`) vs `0.965 s` (`62.0 J`) | `1,236.9 MB` (`2,196.6 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`28` exact) | **`8`** / `35` | **`3.01×` slower** (`median 3.14×`) | **`3.07×` more J** |
| **`10,000,000` rows** (`105` cols) | `1.402 s` (`163.0 J`) vs `10.755 s` (`838.0 J`) | `2,405.9 MB` (`9,006.2 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`29` exact) | **`10`** / `33` | **`3.21×` slower** (`median 4.73×`) | **`3.21×` more J** |
| **`99,997,497` rows — FULL `14.78 GB`** (`is_full_100m_dataset: true`) | `15.546 s` (`1,745.3 J`) vs `91.621 s` (`5,291.9 J`) | **`6,199.0 MB` (`20,740.5 MB`)** | **`43 / 43`** (`0` err) | **`43 / 43`** (`31` exact) | **`5`** / `38` | **`4.85×` slower** (`median 4.57×`) | **`4.25×` more J** |

### Selected Per-Query Results on the Full `99,997,497`-Row (`100M`) Dataset (`reported_j`)

| Query | SQL Summary | DuckDB `2.0.1` | JouleDB `0.8.0` (`pgwire` + `.jcol`) | Outcome |
|---|---|---:|---:|---|
| `Q0` | `SELECT COUNT(*) FROM hits` | `2.07 ms` | **`0.88 ms`** | Exact match (**JouleDB `2.35×` faster** — segment metadata) |
| `Q6` | `SELECT MIN(EventDate), MAX(EventDate) FROM hits` | `6.65 ms` | **`1.59 ms`** | Exact match (**JouleDB `4.18×` faster** — segment ZoneMap) |
| `Q7` | `SELECT AdvEngineID, COUNT(*) WHERE AdvEngineID <> 0 GROUP BY ...` | `27.79 ms` | **`8.79 ms`** | Exact match (**JouleDB `3.16×` faster** — two-phase `WHERE`-first) |
| `Q28` | `SELECT REGEXP_REPLACE(Referer, ...) AS k, AVG(STRLEN(Referer)) ... HAVING COUNT(*) > 100000` | `2,768.95 ms` | **`2,524.39 ms`** | Exact match (**JouleDB `1.10×` faster** — per-segment dictionary regex) |
| `Q42` | `SELECT MINUTE, COUNT(*) AS PageViews WHERE CounterID = 62 AND EventDate >= ...` | `14.90 ms` | **`11.61 ms`** | Exact match (**JouleDB `1.28×` faster** — ZoneMap + `WHERE`-first skip) |
| `Q32` | `SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) GROUP BY ...` | `981.36 ms` | `17,419.48 ms` | SQL-valid tie (`17.75×` slower — `100M` distinct `(WatchID, ClientIP)` groups) |

### Candid Takeaways from Full `100M`-Row ClickBench
1. **Bounded Out-of-Core Memory (`6.20 GB` Load RSS, `20.74 GB` Peak `VmHWM`):** Spilling `262,144`-row `.jcol` segments (`disk_col_segments.rs`) and loading only referenced columns per segment via `pread` cut `10M`-row RSS **17.5×** (`42.0 GB -> 2.41 GB`) and executed the entire `99,997,497`-row (`14.78 GB` Parquet) dataset well inside `< 21 GB` peak RSS.
2. **3× Narrower Latency Gap vs. DuckDB (`9.74× -> 3.21×` at `10M`; `4.85×` at `100M`):** Two-phase `WHERE`-first segment scanning, `Inline4` small-distinct accumulators, and 64-shard Radix `GROUP BY` cut geomean latency ~3× at `1M–10M` rows and beat DuckDB on `5 / 43` queries at `100M` rows.
3. **DuckDB Wins Overall (`38 / 43` at `100M`, `4.85×` Geomean Latency & `5.89×` Faster Load):** DuckDB's SIMD-vectorized hash tables and multi-threaded Parquet reader remain substantially faster on unselective `100M`-group aggregations (`Q32` `0.98 s` vs `17.42 s`) and initial Parquet ingestion (`15.55 s` vs `91.62 s`).

---

## 2. Head-to-Head `100`-Warehouse TCP `pgwire` OLTP vs. `PostgreSQL 16.16` & `SQLite 3.53.1`

- **Harness & Raw Artifact:** [`bench/oltp_pgwire.py`](https://github.com/openIE-dev/jouledb/blob/main/bench/oltp_pgwire.py), [`oltp_pgwire_100wh.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/oltp_pgwire_100wh.json).
- **Workload:** `100,000` accounts (`100` warehouses × `1,000` accounts, initial `SUM(balance) = 100,000,000`), `1, 4, 8, 16` concurrent worker threads (`8.0 s` per level), `50%` `SELECT ... WHERE id = ?`, `30%` `BEGIN; UPDATE accounts SET balance = balance + delta WHERE id = ?; INSERT INTO history ...; COMMIT` (`fsync=on`, `synchronous_commit=on`), `20%` `SELECT COUNT(*), SUM(balance) FROM accounts WHERE branch_id = ?` (`rapl-package` **`reported_j`**).

| Engine & Wire Mode (`invariant_verified = true` on all 5) | `c = 1` TPS (`p50`) | `c = 4` TPS (`p50`) | `c = 8` TPS (`p50`) | `c = 16` TPS (`p50 / p99`, `mJ/txn`) | Committed Writes |
|---|---:|---:|---:|---:|---:|
| **JouleDB `0.8.0` (`pgwire` Simple Query, 4-RTT + `fsync`)** | **`1,659.49`** (`0.304 ms`) | `1,086.43` (`0.974 ms`) | `2,397.52` (`0.958 ms`) | **`3,353.47`** (`1.076 / 21.99 ms`, `17.23 mJ`) | `20,366` (`2` SI conflicts) |
| **PostgreSQL `16.16` (`pgwire` Simple Query, 4-RTT + `fsync`)** | `1,149.05` (`0.077 ms`) | **`3,020.15`** (`0.088 ms`) | **`5,826.89`** (`0.149 ms`) | **`11,452.46`** (`0.253 / 4.85 ms`, `3.79 mJ`) | `51,319` (`0` conflicts) |
| **PostgreSQL `16.16` (`pgwire` Extended Pipeline, 1-RTT + `fsync`)** | `1,255.37` (`0.070 ms`) | `2,984.18` (`0.085 ms`) | `6,044.99` (`0.140 ms`) | **`10,441.83`** (`0.252 / 5.87 ms`, `3.71 mJ`) | `49,555` (`0` conflicts) |
| **JouleDB `0.8.0` (`pgwire` Extended Pipeline, 1-RTT + `fsync`)** | `256.92` (`0.400 ms`) | `490.15` (`1.057 ms`) | `514.37` (`1.067 ms`) | `593.70` (`1.111 / 120.19 ms`, `71.44 mJ`) | `4,394` (`1` SI conflict) |
| **SQLite `3.53.1` (Embedded in-proc C API, `WAL`, `NORMAL`)** | `43,264.13` (`0.005 ms`) | `87,546.78` (`0.006 ms`) | `99,632.02` (`0.007 ms`) | **`111,119.33`** (`0.007 / 3.12 ms`, `0.36 mJ`) | `845,266` (in-proc) |

- **Candid OLTP Findings:**
  1. **`2.54×` Faster JouleDB Simple Query (`1,318.6 -> 3,353.5 TPS` at `c=16`; `1.44×` Ahead of `PostgreSQL 16.16` at `c=1`):** `O(1)` secondary-index PK/UNIQUE constraint checks and `bind_select_query_params` cut `c=1` median latency from `1.15 ms` to `0.304 ms` (`1,659.5 TPS` vs `PostgreSQL 16.16`'s `1,149.1 TPS` on `c=1` single-client `fsync` commits).
  2. **`PostgreSQL 16.16` Wins Multi-Client Scaling (`3.42×` Higher TPS & `4.5×` Lower `mJ/txn` at `c=16`):** PostgreSQL's shared-buffer page locks and WAL writer scale smoothly from `1,149 -> 11,452 TPS` (`p99 4.85 ms`), whereas JouleDB's `commit_lock` + group-sync batching reaches `3,353.5 TPS` (`p99 21.99 ms`).
  3. **Extended Query Pipeline Bottleneck (`593.7 TPS`):** `psycopg` `conn.pipeline()` emits `Parse + Bind + Describe + Execute` for every statement; JouleDB's current `Describe` (`'D'`) handler executes a dry-run query probe to infer result column types before `Execute`, whereas PostgreSQL resolves `Describe` from cached portal metadata without executing the plan. Static AST/catalog type inference for `Describe` is a clear next optimization.

---

## 3. 5-Node Real TCP Raft (`ReadIndex`) Fault-Injection + Official Clojure `elle 0.2.8`

- **Test & Verifier:** [`crates/joule-db-server/tests/raft_tcp_elle_fault_injection.rs`](https://github.com/openIE-dev/jouledb/blob/main/crates/joule-db-server/tests/raft_tcp_elle_fault_injection.rs) + [`bench/elle_check.clj`](https://github.com/openIE-dev/jouledb/blob/main/bench/elle_check.clj).
- **What Runs:** 5 independent `RaftNode<KvStateMachine, TcpRaftTransport>` instances communicating over real loopback TCP sockets via a 20-link socket-severing `TcpPartitionMatrix` with per-node `CRC-32` `raft.wal` disk persistence, `propose_and_commit` for writes, and Raft §6.4 **`read_linearizable` (`ReadIndex` heartbeat quorum, zero log writes)** for reads.
- **Official `jepsen/elle 0.2.8` Verdict:** Checking the exported `76`-event EDN history ([`raft_tcp_elle_history.edn`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/raft_tcp_elle_history.edn)) with Kyle Kingsbury's Clojure `elle/elle 0.2.8` (`rw-register`, `:strict-serializable`) returns **`{:valid? true}`** ([`raft_tcp_elle_official_verdict.txt`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/raft_tcp_elle_official_verdict.txt)).
- **Note:** The separate HTTP endpoint `POST /api/v1/certify/geo_raft_jepsen` (`distributed_jepsen_consensus.rs`) is an **in-memory 5-struct simulation** (`simulation_only: true`, `est_j`), not a multi-process Jepsen run.

---

## 4. `.jsyn` (`JSYN-v1`) Sub-1-Bit Weights: Measured Quality vs. Compression

| Checkpoint / Layer | `BPW` | Shrink vs `FP16` | Measured Fidelity vs Upstream | Status in `v0.8.0` |
|---|---:|---:|---|---|
| **[`dcharlot65-openie/all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit)** (`6L` Encoder) | `0.4378` | `22.23×` (`1.95 MB`) | **`0.9441` sentence-embedding cosine** vs `FP32` | **Published & Verified on Hugging Face** (`Apache-2.0`) |
| `Gemma-4-E2B` `q_proj` (`r = 8 -> 128` rank sweep) | `0.028–0.166` | `564× -> 96.6×` | `0.271 -> 0.455` output cosine (`KL 3.87 -> 2.72`, `65.6%` Top-5) | **Retracted from HF** (not drop-in LLM fidelity) |
| `Inkling-2-Mini` `w13` (`128`-Exp MoE slice, `r = 8 -> 128`) | `0.023–0.134` | `701× -> 120×` | `0.113 -> 0.350` output cosine (`KL 2.91 -> 2.59`) | **Retracted from HF** (not drop-in LLM fidelity) |

---

## 5. In-Process Subsystem Smoke Tests (`jouledb self-check`, `est_j`)

The fast HTTP/CLI self-checks (`jouledb self-check`, `POST /api/v1/bench/industry_suite`, `POST /api/v1/bench/analytic_suite`) execute in `5–40 ms` over small synthetic inputs and report analytical energy estimates (**`energy_evidence: "est_j"`**). They verify internal data-structure invariants during development and **must not** be cited as official TPC-C, ClickBench, JOB, STAC-M3, or Jepsen benchmark results.
