# Benchmarks, Consensus Verification & Subsystem Smoke Tests (`v0.9.0`)

> **Audit Remediation & Evidence Standard (`v0.9.0`):** Earlier `v0.2.8`–`v1.0.0` documentation presented 36 in-process micro-harnesses and synthetic simulations as *"36 / 36 Certified Global Benchmark Pillars"*, including synthetic frontier AI exam routes (`frontier_exam_routes.rs`). **All synthetic frontier AI benchmark claims remain retracted and `frontier_exam_routes.rs` has been deleted.**
>
> Below we strictly separate **(1) reproducible external dataset, multi-client `pgwire`, & multi-host WAN consensus evaluations** (with every win, loss, memory profile, and architectural bottleneck reported openly) from **(2) fast in-process functional smoke checks**, and tag every energy figure as **`reported_j`** (hardware RAPL / SMC counters) or **`est_j`** (analytical cost model).

---

## 1. Full Official ClickBench (`hits.parquet`, `99,997,497` Rows, `105` Columns, `43` Queries) vs. DuckDB `2.0.1`

- **Harness & Raw Artifacts:** [`bench/clickbench_official.py`](https://github.com/openIE-dev/jouledb/blob/main/bench/clickbench_official.py), [`clickbench_official_100k.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/clickbench_official_100k.json), [`clickbench_official_1m.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/clickbench_official_1m.json), [`clickbench_official_10m.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/clickbench_official_10m.json), **[`clickbench_official_100m.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/clickbench_official_100m.json)**.
- **Methodology:** Streams the official `14.78 GB` (`99,997,497`-row, `105`-column) ClickBench `hits.parquet` file into DuckDB `2.0.1` (in-process) and JouleDB `0.9.0` (over real TCP `pgwire` via `psycopg 3.3.9`, spilling immutable `262,144`-row `.jcol` segments to disk), then runs all **43 official ClickBench SQL queries (`Q0..Q42`)** unchanged on `framework-128gb-0000` (`AMD Ryzen AI Max+ 395`, 32 logical cores, `128 GB` UMA RAM, Linux `rapl-package` **`reported_j`**).

| Dataset Scale (`hits.parquet`) | Parquet Load (`DuckDB` vs `JouleDB`) | JouleDB Load RSS (`VmHWM` Peak) | Queries Succeeded | SQL-Valid Match (`Exact`) | Faster (`JouleDB` / `DuckDB`) | Geomean Latency (`JouleDB / DuckDB`) | Geomean Energy (`reported_j`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`100,000` rows** (`105` cols) | `0.326 s` (`42.6 J`) vs `0.409 s` (`51.3 J`) | `927.5 MB` (`2,719.5 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`35` exact) | **`21`** / `22` | **`1.31×` slower** (`median 1.04×`) | **`1.65×` more J** |
| **`1,000,000` rows** (`105` cols) | `0.257 s` (`35.5 J`) vs `1.363 s` (`175.3 J`) | `1,246.4 MB` (`1,958.6 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`29` exact) | **`13`** / `30` | **`2.42×` slower** (`median 3.22×`) | **`2.34×` more J** |
| **`10,000,000` rows** (`105` cols) | `1.459 s` (`200.8 J`) vs `12.568 s` (`1,415.8 J`) | `2,163.3 MB` (`9,060.4 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`29` exact) | **`12`** / `31` | **`2.52×` slower** (`median 3.88×`) | **`2.47×` more J** |
| **`99,997,497` rows — FULL `14.78 GB`** (`is_full_100m_dataset: true`) | `16.497 s` (`1,939.2 J`) vs `86.068 s` (`6,604.6 J`) | **`6,204.0 MB` (`15,483.2 MB`)** | **`43 / 43`** (`0` err) | **`43 / 43`** (`31` exact) | **`8`** / `35` | **`2.82×` slower** (`median 3.42×`) | **`2.43×` more J** |

### Selected Per-Query Results on the Full `99,997,497`-Row (`100M`) Dataset (`reported_j`)

| Query | SQL Summary | DuckDB `2.0.1` | JouleDB `0.9.0` (`pgwire` + `.jcol`) | Outcome |
|---|---|---:|---:|---|
| `Q0` | `SELECT COUNT(*) FROM hits` | `9.36 ms` | **`1.51 ms`** | Exact match (**JouleDB `6.19×` faster** — segment metadata) |
| `Q2` | `SELECT SUM(AdvEngineID), COUNT(*), AVG(ResolutionWidth) FROM hits` | `8.93 ms` | **`1.60 ms`** | Exact match (**JouleDB `5.60×` faster** — zero-I/O `global_int_sum`) |
| `Q3` | `SELECT AVG(UserID) FROM hits` | `12.77 ms` | **`1.84 ms`** | Exact match (**JouleDB `6.93×` faster** — zero-I/O `global_int_sum`) |
| `Q6` | `SELECT MIN(EventDate), MAX(EventDate) FROM hits` | `7.33 ms` | **`1.24 ms`** | Exact match (**JouleDB `5.90×` faster** — segment ZoneMap) |
| `Q19` | `SELECT UserID FROM hits WHERE UserID = 435090932899640449` | `42.46 ms` | **`31.32 ms`** | Exact match (**JouleDB `1.36×` faster** — ZoneMap pruning) |
| `Q28` | `SELECT REGEXP_REPLACE(Referer, ...) AS k, AVG(STRLEN(Referer)) ... HAVING COUNT(*) > 100000` | `3,442.50 ms` | **`2,912.02 ms`** | Exact match (**JouleDB `1.18×` faster** — per-segment dictionary regex) |
| `Q29` | `SELECT SUM(ResolutionWidth), SUM(ResolutionWidth + 1), ...` (`90` sums) | `13.68 ms` | **`2.27 ms`** | Exact match (**JouleDB `6.02×` faster** — algebraic `global_int_sum`) |
| `Q42` | `SELECT DATE_TRUNC('minute', EventTime), COUNT(*) WHERE CounterID = 62 ...` | `23.70 ms` | **`12.00 ms`** | Exact match (**JouleDB `1.98×` faster** — ZoneMap + `WHERE`-first skip) |
| `Q32` | `SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) GROUP BY ...` | `1,063.86 ms` | `2,405.47 ms` | SQL-valid tie (`2.26×` slower — **cut `7.24×` from `17.42 s` in `v0.8.0`** via `CompactI2Row`) |
| `Q39` | `SELECT TraficSourceID, SearchEngineID, AdvEngineID, CASE WHEN ... GROUP BY ...` | `23.23 ms` | `519.27 ms` | Exact match (`22.36×` slower — widest `100M` ratio on filtered multi-expression `CASE` scan) |

### Candid Takeaways from Full `100M`-Row ClickBench
1. **Bounded Out-of-Core Memory (`6.20 GB` Load RSS, `15.48 GB` Peak `VmHWM`):** Compact `CompactI2Row` 256-partition sort-reduce cut `100M` kernel `VmHWM` peak by **`5.26 GB`** (`20.74 GB -> 15.48 GB`) and sped up `100M`-group `Q32` **`7.24×`** (`17.42 s -> 2.41 s`).
2. **`1.72×` Narrower Full-`100M` Latency Gap (`4.85× -> 2.82×` Geomean, `8 / 43` Wins):** Zero-I/O `global_int_sum` metadata, dynamic `AtomicI64` Top-K ZoneMap pruning, and zero-`Arc<str>` `RawTextSlice` text hashing lifted JouleDB wins to **`8 / 43` at `100M` rows** (`13 / 43` at `1M`, `12 / 43` at `10M`, `21 / 43` at `100k`) and cut `100M` geomean `reported_j` to **`2.43×`**.
3. **DuckDB Wins Overall (`35 / 43` at `100M`, `2.82×` Geomean Latency & `5.22×` Faster Load):** DuckDB's SIMD-vectorized pipelines and multi-threaded Parquet reader remain faster on `35 / 43` queries and initial Parquet ingestion (`16.50 s` vs `86.07 s`).

---

## 2. Head-to-Head `100`-Warehouse TCP `pgwire` OLTP vs. `PostgreSQL 16.16` & `SQLite 3.53.1`

- **Harness & Raw Artifact:** [`bench/oltp_pgwire.py`](https://github.com/openIE-dev/jouledb/blob/main/bench/oltp_pgwire.py), [`oltp_pgwire_100wh.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/oltp_pgwire_100wh.json).
- **Workload:** `100,000` accounts (`100` warehouses × `1,000` accounts, initial `SUM(balance) = 100,000,000`), `1, 4, 8, 16` concurrent worker threads (`8.0 s` per level), `50%` `SELECT ... WHERE id = $1`, `30%` `BEGIN; UPDATE accounts SET balance = balance + $1 WHERE id = $2; INSERT INTO history ...; COMMIT` (`fsync=on`, `synchronous_commit=on`), `20%` `SELECT COUNT(*), SUM(balance) FROM accounts WHERE branch_id = $1` (`rapl-package` **`reported_j`**).

| Engine & Wire Mode (`invariant_verified = true` on all 5) | `c = 1` TPS (`p50`) | `c = 4` TPS (`p50`) | `c = 8` TPS (`p50`) | `c = 16` TPS (`p50 / p99`, `mJ/txn`) | Committed Writes |
|---|---:|---:|---:|---:|---:|
| **JouleDB `0.9.0` (`pgwire` Simple Query, 4-RTT + `fsync`)** | `522.23` (`0.340 ms`) | `954.49` (`0.972 ms`) | `2,125.09` (`1.040 ms`) | **`3,351.04`** (`1.069 / 31.40 ms`, `24.29 mJ`) | `16,647` (`0` errors) |
| **JouleDB `0.9.0` (`pgwire` Extended Pipeline, 1-RTT + `fsync`)** | `260.01` (`0.567 ms`) | `871.90` (`1.584 ms`) | **`1,789.98`** (`2.074 ms`) | **`1,703.25`** (`2.093 / 50.29 ms`, `75.70 mJ`) | `10,973` (`1` SI conflict) |
| **PostgreSQL `16.16` (`pgwire` Simple Query, 4-RTT + `fsync`)** | `3,504.95` (`0.171 ms`) | `2,957.16` (`0.208 ms`) | `6,100.72` (`0.122 ms`) | **`10,815.93`** (`0.375 / 5.63 ms`, `9.13 mJ`) | `55,902` (`0` conflicts) |
| **PostgreSQL `16.16` (`pgwire` Extended Pipeline, 1-RTT + `fsync`)** | `1,291.43` (`0.084 ms`) | `2,666.46` (`0.198 ms`) | `5,607.16` (`0.304 ms`) | **`8,634.44`** (`0.683 / 6.96 ms`, `14.66 mJ`) | `43,602` (`0` conflicts) |
| **SQLite `3.53.1` (Embedded in-proc C API, `WAL`, `NORMAL`)** | `33,431.06` (`0.006 ms`) | `56,406.57` (`0.008 ms`) | `55,520.22` (`0.011 ms`) | **`60,263.42`** (`0.012 / 5.65 ms`, `2.18 mJ`) | `506,957` (in-proc) |

- **Candid OLTP Findings:**
  1. **`3.01×` Faster JouleDB Extended Query Pipeline (`593.7 -> 1,790.0 TPS` at `c=8`, `1,703.3 TPS` at `c=16`, `2.50×` More Writes):** Static AST/catalog `describe_sql_static` OID inference eliminated the `v0.8.0` dry-run `Describe` execution probe, while `DROP TABLE` index metadata cleanup prevented stale `__indexes__` collisions across table reloads.
  2. **`PostgreSQL 16.16` Wins Multi-Client Scaling (`3.23×` Higher TPS at `c=16`):** PostgreSQL's shared-buffer page locks and dedicated WAL writer scale to `10,815.9 TPS` (`p99 5.63 ms`) vs JouleDB's `3,351.0 TPS` (`p99 31.40 ms`) under table-wide `commit_lock`.

---

## 3. 5-Node Multi-Host Cross-Machine WAN/Tailscale TCP Raft (`ReadIndex`) + Official Clojure `elle 0.2.8`

- **Harnesses & Verifier:** [`bench/raft_multihost_wan.py`](https://github.com/openIE-dev/jouledb/blob/main/bench/raft_multihost_wan.py), [`examples/raft_wan_node.rs`](https://github.com/openIE-dev/jouledb/blob/main/crates/joule-db-server/examples/raft_wan_node.rs), [`tests/raft_tcp_elle_fault_injection.rs`](https://github.com/openIE-dev/jouledb/blob/main/crates/joule-db-server/tests/raft_tcp_elle_fault_injection.rs), and [`bench/elle_check.clj`](https://github.com/openIE-dev/jouledb/blob/main/bench/elle_check.clj).
- **Cross-Machine WAN Run (`v0.9.0`):** 5 independent OS `raft_wan_node` processes communicating across **2 distinct physical Linux machines** (`framework-128gb-0000` bare-metal `n1..n3` + `openie-core-1` Cloud Debian 12 VM `n4..n5`) over encrypted Tailscale WAN (**`27.58 ms` median TCP RTT**, `p95 28.65 ms`) with durable `CRC-32` `raft.wal`, `propose_and_commit` (§5.4.2), and `read_linearizable` (§6.4 `ReadIndex` heartbeat quorum, zero log writes).
- **Nemeses & Convergence:** Cross-WAN `2/5` minority split (`["n1", "n4"]` vs `["n2", "n3", "n5"]`) isolating initial leader `n1` -> stale write & `ReadIndex` read rejected -> `3/5` majority elects `n3` (term `2`, `24` committed ops) -> real OS **`SIGKILL` (`kill -9`)** on cloud node `n5` (`16` ops committed during crash) -> `raft.wal` restart & 5-node convergence (`commit=68, applied=68, x=68, y=39` across all 5 nodes; WAN client `p50 111.57 ms`, `p95 160.02 ms`).
- **Official `jepsen/elle 0.2.8` Verdict:** Checking the `270`-event WAN EDN history ([`raft_wan_multihost_elle_history.edn`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/raft_wan_multihost_elle_history.edn)) with Kyle Kingsbury's Clojure `elle/elle 0.2.8` (`rw-register`, `:strict-serializable`) returns **`{:valid? true}`** ([`raft_wan_multihost_elle_verdict.txt`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/raft_wan_multihost_elle_verdict.txt)).
- **Note:** The separate HTTP endpoint `POST /api/v1/certify/geo_raft_jepsen` (`distributed_jepsen_consensus.rs`) is an **in-memory 5-struct simulation** (`simulation_only: true`, `est_j`), not a multi-process Jepsen run.

---

## 4. `.jsyn` (`JSYN-v1`) Sub-1-Bit Weights: Measured Quality vs. Compression

| Checkpoint / Layer | `BPW` | Shrink vs `FP16` | Measured Fidelity vs Upstream | Status in `v0.9.0` |
|---|---:|---:|---|---|
| **[`dcharlot65-openie/all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit)** (`6L` Encoder) | `0.4378` | `22.23×` (`1.95 MB`) | **`0.9441` sentence-embedding cosine** vs `FP32` | **Published & Verified on Hugging Face** (`Apache-2.0`) |
| `Gemma-4-E2B` `q_proj` (`r = 8 -> 128` rank sweep) | `0.028–0.166` | `564× -> 96.6×` | `0.271 -> 0.455` output cosine (`KL 3.87 -> 2.72`, `65.6%` Top-5) | **Retracted from HF** (not drop-in LLM fidelity) |
| `Inkling-2-Mini` `w13` (`128`-Exp MoE slice, `r = 8 -> 128`) | `0.023–0.134` | `701× -> 120×` | `0.113 -> 0.350` output cosine (`KL 2.91 -> 2.59`) | **Retracted from HF** (not drop-in LLM fidelity) |

---

## 5. In-Process Subsystem Smoke Tests (`jouledb self-check`, `est_j`)

The fast HTTP/CLI self-checks (`jouledb self-check`, `POST /api/v1/bench/industry_suite`, `POST /api/v1/bench/analytic_suite`) execute in `5–40 ms` over small synthetic inputs and report analytical energy estimates (**`energy_evidence: "est_j"`**). They verify internal data-structure invariants during development and **must not** be cited as official TPC-C, ClickBench, JOB, STAC-M3, or Jepsen benchmark results.
