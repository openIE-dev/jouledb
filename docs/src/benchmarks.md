# Benchmarks, Consensus Verification & Subsystem Smoke Tests (`v0.7.0`)

> **Audit Remediation & Evidence Standard (`v0.7.0`):** Earlier `v0.2.8`–`v1.0.0` documentation presented 36 in-process micro-harnesses and synthetic simulations as *"36 / 36 Certified Global Benchmark Pillars"*, including synthetic frontier AI exam routes (`frontier_exam_routes.rs`). **All synthetic frontier AI benchmark claims remain retracted and `frontier_exam_routes.rs` has been deleted.**
>
> Below we strictly separate **(1) reproducible external dataset, multi-client `pgwire`, & multi-socket evaluations** (with every win, loss, memory limit, and bug reported openly) from **(2) fast in-process functional smoke checks**, and tag every energy figure as **`reported_j`** (hardware RAPL / SMC counters) or **`est_j`** (analytical cost model).

---

## 1. Official ClickBench (`hits.parquet`, 105 Columns, 43 Queries) vs. DuckDB `2.0.1`

- **Harness & Raw Artifacts:** [`bench/clickbench_official.py`](https://github.com/openIE-dev/jouledb/blob/main/bench/clickbench_official.py), [`clickbench_official_100k.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/clickbench_official_100k.json), [`clickbench_official_1m.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/clickbench_official_1m.json), [`clickbench_official_10m.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/clickbench_official_10m.json).
- **Methodology:** Streams rows from the official `14.78 GB` (`99,997,497`-row, `105`-column) ClickBench `hits.parquet` file into DuckDB `2.0.1` (in-process) and JouleDB `0.7.0` (over real TCP `pgwire` via `psycopg 3.3.9`), then runs all **43 official ClickBench SQL queries (`Q0..Q42`)** unchanged with a `10 s` per-query timeout on `framework-128gb-0000` (`AMD Ryzen AI Max+ 395`, 32 logical cores, `128 GB` UMA RAM, Linux `rapl-package` **`reported_j`**).

| Dataset Slice (`hits.parquet`) | Parquet Load (`DuckDB` vs `JouleDB`) | JouleDB RSS After Load | Queries Succeeded (`<10s`) | SQL-Valid Match (`Exact`) | Faster (`JouleDB` / `DuckDB`) | Geomean Latency (`JouleDB / DuckDB`) | Geomean Energy (`reported_j`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`100,000` rows** (`105` cols) | `0.197 s` (`23.6 J`) vs `0.355 s` (`41.7 J`) | `933.8 MB` | **`43 / 43`** (`0` timeouts) | **`43 / 43`** (`35` exact) | **`25`** / `18` | **`1.30×` slower** (**`median 0.80×`**) | **`1.53×` more J** |
| **`1,000,000` rows** (`105` cols) | `0.402 s` (`52.7 J`) vs `2.889 s` (`374.4 J`) | `4,902.0 MB` | **`43 / 43`** (`0` timeouts) | **`43 / 43`** (`30` exact) | `3` / `40` | **`9.34×` slower** (`median 10.08×`) | **`9.39×` more J** |
| **`10,000,000` rows** (`105` cols) | `0.799 s` (`109.9 J`) vs `16.660 s` (`1,385.8 J`) | `41,995.9 MB` | **`43 / 43`** (`0` timeouts) | **`43 / 43`** (`30` exact) | `2` / `41` | **`9.74×` slower** (`median 9.36×`) | **`7.50×` more J** |

### Representative Per-Query Comparison at `10,000,000` Official Rows (`reported_j`)

| Query | SQL Summary | DuckDB `2.0.1` | JouleDB `0.7.0` (`pgwire`) | Match / Outcome |
|---|---|---:|---:|---|
| `Q0` | `SELECT COUNT(*) FROM hits` | `1.50 ms` (`0.157 J`) | `81.02 ms` (`7.987 J`) | Exact match (`0.98 ms` at `1M` rows) |
| `Q1` | `SELECT COUNT(*) FROM hits WHERE AdvEngineID <> 0` | `2.30 ms` (`0.178 J`) | **`1.17 ms`** (`0.206 J`) | Exact match (**JouleDB `2.0×` faster**) |
| `Q4` | `SELECT COUNT(DISTINCT UserID) FROM hits` | `27.24 ms` (`3.557 J`) | `160.96 ms` (`16.061 J`) | Exact match (`5.9×` slower) |
| `Q8` | `SELECT RegionID, COUNT(DISTINCT UserID) AS u ... GROUP BY RegionID ORDER BY u DESC LIMIT 10` | `34.72 ms` (`4.435 J`) | `230.29 ms` (`21.055 J`) | Exact match (`6.6×` slower) |
| `Q12` | `SELECT SearchPhrase, COUNT(*) AS c WHERE SearchPhrase <> '' GROUP BY ... LIMIT 10` | `26.17 ms` (`3.615 J`) | `135.06 ms` (`9.927 J`) | Exact match (`5.2×` slower) |
| `Q28` | `SELECT REGEXP_REPLACE(Referer, ...) AS k, AVG(STRLEN(Referer)) AS l ... HAVING COUNT(*) > 100000` | `476.29 ms` (`72.183 J`) | `955.50 ms` (`87.630 J`) | Exact match (`2.0×` slower; timed out in `v0.6.0`) |
| `Q29` | `SELECT SUM(ResolutionWidth), SUM(ResolutionWidth + 1), ..., SUM(ResolutionWidth + 89)` | `11.79 ms` (`1.119 J`) | **`11.05 ms` (`0.879 J`)** | Exact match (**JouleDB `1.07×` faster**, single scan) |
| `Q33` | `SELECT URL, COUNT(*) AS c FROM hits GROUP BY URL ORDER BY c DESC LIMIT 10` | `77.49 ms` (`9.934 J`) | `254.29 ms` (`18.441 J`) | Exact match (`3.3×` slower) |
| `Q36` | `SELECT URL, COUNT(*) AS PageViews WHERE CounterID = 62 AND EventDate >= '2013-07-01' ...` | `38.30 ms` (`5.176 J`) | `259.51 ms` (`33.193 J`) | Exact match (`6.8×` slower; timed out in `v0.6.0`) |
| `Q32` | `SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) GROUP BY ...` | `81.42 ms` (`11.204 J`) | `6,770.13 ms` (`509.08 J`) | SQL-valid tie verified in DuckDB (`83.1×` slower — slowest query) |

### Candid Takeaways from Official ClickBench
1. **From `23/43` (`v0.6.0` @ `1M`) to `43/43` (`v0.7.0` @ `100k`, `1M`, & `10M`):** Vectorizing `EXTRACT`/`DATE_TRUNC`, dictionary `REGEXP_REPLACE`, byte-exact `STRLEN`, algebraic `SUM(x+k)` scan sharing, `HAVING`, and parallel Top-K scans eliminated all 20 timeouts and cut `100k` geomean latency from `26.37×` to `1.30×` (`25 / 43` faster than DuckDB).
2. **DuckDB Wins Overall at `1M–10M` Scale (`9.3×–9.7×` Geomean Latency):** DuckDB's mature morsel-driven C++ engine remains ~`9.5×` faster on geometric mean at `10M` rows (especially on `10M`-group high-cardinality `Q32` at `81 ms` vs `6.77 s`).
3. **Explicit `100M`-Row Memory Ceiling (`42.0 GB` RSS at `10M` rows):** `DurableAmorphicStore` retains queued `ColumnBatch` chunks alongside `TableColumns` in RAM (`~4.2 KB/row` across 105 columns), so full `100M`-row ClickBench (`~410 GB` projected RAM) requires out-of-core column spilling not yet wired into `DurableAmorphicStore`.

---

## 2. Concurrent Multi-Client OLTP over Real TCP `pgwire` (`100` Warehouses = `100,000` Accounts)

- **Harness & Raw Artifact:** [`bench/oltp_pgwire.py`](https://github.com/openIE-dev/jouledb/blob/main/bench/oltp_pgwire.py), [`oltp_pgwire_100wh.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/oltp_pgwire_100wh.json).
- **Workload:** Populates `100,000` accounts (`100` warehouses × `1,000` accounts, initial `SUM(balance) = 100,000,000`) via PostgreSQL `COPY ... FROM STDIN` (`0.607 s`, `70.5 J`), builds secondary indexes on `id` and `branch_id`, and runs `1, 4, 8, 16` concurrent worker threads (`50%` `SELECT ... WHERE id = ?`, `30%` `BEGIN; UPDATE accounts SET balance = balance + delta WHERE id = ?; INSERT INTO history ...; COMMIT` with default `JOULEDB_SYNC_COMMIT=on` durable `fsync`, `20%` `SELECT COUNT(*), SUM(balance) FROM accounts WHERE branch_id = ?`).

| Concurrency | JouleDB `0.7.0` (`pgwire` TCP + `fsync`) | JouleDB `p50 / p95 / p99` (ms) | JouleDB `mJ / txn` (`reported_j`) | Embedded `SQLite 3.53.1` (`WAL`, `synchronous=NORMAL`) |
|---:|---:|---:|---:|---:|
| **`1` client** | **`604.97 TPS`** (`3,630` ok, `0` err) | `1.149 / 5.829 / 11.069` | `208.52 mJ` | `18,754.83 TPS` (`p50 0.007 ms`, `6.78 mJ`) |
| **`4` clients** | **`720.57 TPS`** (`4,330` ok, `0` err) | `2.168 / 24.336 / 42.552` | `176.90 mJ` | `35,532.94 TPS` (`p50 0.010 ms`, `3.58 mJ`) |
| **`8` clients** | **`888.93 TPS`** (`5,348` ok, `0` err) | `2.989 / 31.996 / 44.715` | `145.70 mJ` | `36,273.26 TPS` (`p50 0.010 ms`, `3.44 mJ`) |
| **`16` clients** | **`1,318.62 TPS`** (`7,945` ok, `1` SI conflict) | `3.522 / 44.437 / 60.821` | `97.50 mJ` | `45,142.20 TPS` (`p50 0.009 ms`, `2.51 mJ`) |

- **Post-Run Conservation Verification:** **`invariant_verified: true`** (`accounts_count = 100,000`, `committed_write_txns = 6,431`, `history_rows = 6,431`, `expected_balance_sum = 100,096,250 == actual_balance_sum = 100,096,250`).
- **Candid OLTP Findings:** Embedded in-process `SQLite 3` (`synchronous=NORMAL`, C function calls without TCP socket serialization or per-commit `fsync`) is `31×–34×` faster than JouleDB over TCP `pgwire` (where each write transaction executes 4 sequential TCP query round-trips + `GroupSync` `fsync`). Multi-row `INSERT INTO ... VALUES` on `PRIMARY KEY` tables also still performs per-row uniqueness checks, so bulk loading should use `COPY ... FROM`.

---

## 3. 5-Node Real TCP Raft (`ReadIndex`) Fault-Injection + Official Clojure `elle 0.2.8`

- **Test & Verifier:** [`crates/joule-db-server/tests/raft_tcp_elle_fault_injection.rs`](https://github.com/openIE-dev/jouledb/blob/main/crates/joule-db-server/tests/raft_tcp_elle_fault_injection.rs) + [`bench/elle_check.clj`](https://github.com/openIE-dev/jouledb/blob/main/bench/elle_check.clj).
- **What Runs:** 5 independent `RaftNode<KvStateMachine, TcpRaftTransport>` instances communicating over real loopback TCP sockets via a 20-link socket-severing `TcpPartitionMatrix` with per-node `CRC-32` `raft.wal` disk persistence, `propose_and_commit` for writes, and Raft §6.4 **`read_linearizable` (`ReadIndex` heartbeat quorum, zero log writes)** for reads.
- **Official `jepsen/elle 0.2.8` Verdict:** Checking the exported `76`-event EDN history ([`raft_tcp_elle_history.edn`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/raft_tcp_elle_history.edn)) with Kyle Kingsbury's Clojure `elle/elle 0.2.8` (`rw-register`, `:strict-serializable`) returns **`{:valid? true}`** ([`raft_tcp_elle_official_verdict.txt`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/raft_tcp_elle_official_verdict.txt)).
- **Note:** The separate HTTP endpoint `POST /api/v1/certify/geo_raft_jepsen` (`distributed_jepsen_consensus.rs`) is an **in-memory 5-struct simulation** (`simulation_only: true`, `est_j`), not a multi-process Jepsen run.

---

## 4. `.jsyn` (`JSYN-v1`) Sub-1-Bit Weights: Measured Quality vs. Compression

| Checkpoint / Layer | `BPW` | Shrink vs `FP16` | Measured Fidelity vs Upstream | Status in `v0.7.0` |
|---|---:|---:|---|---|
| **[`dcharlot65-openie/all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit)** (`6L` Encoder) | `0.4378` | `22.23×` (`1.95 MB`) | **`0.9441` sentence-embedding cosine** vs `FP32` | **Published & Verified on Hugging Face** (`Apache-2.0`) |
| `Gemma-4-E2B` `q_proj` (`r = 8 -> 128` rank sweep) | `0.028–0.166` | `564× -> 96.6×` | `0.271 -> 0.455` output cosine (`KL 3.87 -> 2.72`, `65.6%` Top-5) | **Retracted from HF** (not drop-in LLM fidelity) |
| `Inkling-2-Mini` `w13` (`128`-Exp MoE slice, `r = 8 -> 128`) | `0.023–0.134` | `701× -> 120×` | `0.113 -> 0.350` output cosine (`KL 2.91 -> 2.59`) | **Retracted from HF** (not drop-in LLM fidelity) |

---

## 5. In-Process Subsystem Smoke Tests (`jouledb self-check`, `est_j`)

The fast HTTP/CLI self-checks (`jouledb self-check`, `POST /api/v1/bench/industry_suite`, `POST /api/v1/bench/analytic_suite`) execute in `5–40 ms` over small synthetic inputs and report analytical energy estimates (**`energy_evidence: "est_j"`**). They verify internal data-structure invariants during development and **must not** be cited as official TPC-C, ClickBench, JOB, STAC-M3, or Jepsen benchmark results.
