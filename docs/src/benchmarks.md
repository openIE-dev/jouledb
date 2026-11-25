# Benchmarks, Consensus Verification & Subsystem Smoke Tests (`v0.6.0`)

> **Audit Remediation & Retraction Notice (`v0.6.0`):** Earlier `v0.2.8`–`v1.0.0` documentation presented 36 in-process micro-harnesses and synthetic simulations as *"36 / 36 Certified Global Benchmark Pillars"*, including hard-coded or tiny-sample frontier AI exam routes (`HLE-Diamond`, `SuperGPQA`, `SimpleQA`, `FaithEval`, `ARC-AGI-2`, `FrontierMath`, `SWE-bench`, `TAU-bench`, `MRCR`, `SafeVLA`). **All synthetic frontier AI benchmark claims have been retracted and `frontier_exam_routes.rs` has been deleted.**
>
> Below we strictly separate **(1) real external dataset & multi-socket evaluations** (with every loss, timeout, memory limit, and bug reported openly) from **(2) fast in-process functional smoke checks**, and tag every energy figure as **`reported_j`** (hardware RAPL / SMC counters) or **`est_j`** (analytical cost model).

---

## 1. Official ClickBench (`hits.parquet`, 105 Columns, 43 Queries) vs. DuckDB `2.0.1`

- **Harness & Raw Artifacts:** [`bench/clickbench_official.py`](https://github.com/openIE-dev/jouledb/blob/main/bench/clickbench_official.py), [`clickbench_official_100k.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/clickbench_official_100k.json), [`clickbench_official_1m.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/clickbench_official_1m.json).
- **Methodology:** Loads rows from the official `14.78 GB` (`99,997,497`-row, `105`-column) ClickBench `hits.parquet` file into DuckDB `2.0.1` (in-process) and JouleDB `0.6.0` (over real TCP `pgwire` via `psycopg 3.3.9`), then runs all **43 official ClickBench SQL queries (`Q0..Q42`)** unchanged with a `10 s` per-query timeout on `framework-128gb-0000` (`AMD Ryzen AI Max+ 395`, 32 logical cores, `128 GB` UMA RAM, Linux `rapl-package` **`reported_j`**).

| Dataset Slice (`hits.parquet`) | Parquet Load (`DuckDB` vs `JouleDB`) | JouleDB RSS After Load | Queries Succeeded (`<10s`) | Exact Match vs DuckDB | Faster (`JouleDB` / `DuckDB`) | Geomean Latency (`JouleDB / DuckDB`) | Geomean Energy (`reported_j`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`100,000` rows** (`105` cols) | `0.192 s` (`13.5 J`) vs `0.203 s` (`14.8 J`) | `722.2 MB` | **`42 / 43`** (`1` timeout: `Q29`) | **`33 / 43`** | `10` / `32` | **`26.37×` slower** (`median 17.03×`) | **`44.96×` more J** |
| **`1,000,000` rows** (`105` cols) | `0.153 s` (`13.2 J`) vs `1.125 s` (`54.7 J`) | `4,177.2 MB` | **`23 / 43`** (`20` timeouts `>10s`) | **`18 / 23`** completed | `2` / `21` | **`10.19×` slower** *(23 completed)* | **`12.76×` more J** *(23 completed)* |

### Representative Per-Query Comparison at `1,000,000` Official Rows (`reported_j`)

| Query | SQL Summary | DuckDB `2.0.1` | JouleDB `0.6.0` (`pgwire`) | Match / Outcome |
|---|---|---:|---:|---|
| `Q0` | `SELECT COUNT(*) FROM hits` | `1.23 ms` (`0.026 J`) | `2.71 ms` (`0.083 J`) | Exact match (`2.2×` slower) |
| `Q2` | `SELECT SUM(AdvEngineID), COUNT(*), AVG(ResolutionWidth) FROM hits` | `1.80 ms` (`0.109 J`) | **`1.10 ms` (`0.034 J`)** | Exact match (**JouleDB `1.6×` faster**) |
| `Q4` | `SELECT COUNT(DISTINCT UserID) FROM hits` | `3.03 ms` (`0.136 J`) | `12.02 ms` (`1.419 J`) | Exact match (`4.0×` slower) |
| `Q8` | `SELECT RegionID, COUNT(DISTINCT UserID) AS u ... GROUP BY RegionID ORDER BY u DESC LIMIT 10` | `6.19 ms` (`0.502 J`) | `117.74 ms` (`7.779 J`) | Exact match (`19.0×` slower) |
| `Q12` | `SELECT SearchPhrase, COUNT(*) AS c WHERE SearchPhrase <> '' GROUP BY ... LIMIT 10` | `4.79 ms` (`0.265 J`) | `72.65 ms` (`4.440 J`) | Exact match (`15.2×` slower) |
| `Q20` | `SELECT COUNT(*) FROM hits WHERE URL LIKE '%google%'` | `6.34 ms` (`0.331 J`) | `77.51 ms` (`10.471 J`) | Exact match (`12.2×` slower) |
| `Q32` | `SELECT WatchID, ClientIP, COUNT(*), SUM(IsRefresh), AVG(ResolutionWidth) ...` | `22.38 ms` (`2.227 J`) | `1,428.1 ms` (`92.22 J`) | **Bug:** `AVG(ResolutionWidth)` returned `0.0` |
| `Q33` | `SELECT URL, COUNT(*) AS c FROM hits GROUP BY URL ORDER BY c DESC LIMIT 10` | `11.75 ms` (`0.779 J`) | `76.17 ms` (`5.250 J`) | Exact match (`6.5×` slower) |
| `Q18, Q21–Q29, Q34–Q42` | `EXTRACT`, `REGEXP_REPLACE`, `90× SUM(x+k)`, `ORDER BY` sort-top-N, string-date filters | `1.47–87.9 ms` | **`> 10,000 ms` (timeout)** | Falls off `columnar_exec` onto row-AST path |

### Candid Takeaways from Official ClickBench
1. **Scale Limit (`~4.18 GB` RSS per `1M` 105-column rows):** `COPY FROM parquet` currently caches decoded columns in RAM without out-of-core column eviction, so full `100M`-row ClickBench (`~418 GB` projected RAM) exceeds a `128 GB` workstation.
2. **DuckDB Wins Decisively Overall (`10×–26×` Geomean Latency):** DuckDB's mature morsel-driven vectorized engine is an order of magnitude faster and lower-energy across grouped OLAP scans and supports all `43 / 43` queries in `< 90 ms` at `1M` rows.
3. **Real Bugs Found & Fixed:** Running the official 43 queries exposed an empty-group panic on `Q41` in `columnar_exec.rs` (fixed in `v0.6.0`) and a remaining `AVG` accumulator bug on `Q32` (documented openly).

---

## 2. 5-Node Real TCP Raft Fault-Injection + Official Clojure `elle 0.2.8`

- **Test & Verifier:** [`crates/joule-db-server/tests/raft_tcp_elle_fault_injection.rs`](https://github.com/openIE-dev/jouledb/blob/main/crates/joule-db-server/tests/raft_tcp_elle_fault_injection.rs) + [`bench/elle_check.clj`](https://github.com/openIE-dev/jouledb/blob/main/bench/elle_check.clj).
- **What Runs:** 5 independent `RaftNode<KvStateMachine, TcpRaftTransport>` instances communicating over real loopback TCP sockets via a 20-link socket-severing `TcpPartitionMatrix` with per-node `CRC-32` `raft.wal` disk persistence.
- **Faults Injected:**
  1. Asymmetric `2 / 5` vs `3 / 5` TCP partition isolating the initial leader (`raft-tcp-4`, term `2`).
  2. Client writes and quorum-barriered reads against the isolated minority leader are rejected/timed out (`0` split-brain commits).
  3. Majority `3 / 5` partition elects a new leader (`raft-tcp-1`, term `3`) and commits concurrent `rw-register` transactions.
  4. A follower is stopped, restarted from its on-disk `raft.wal` on a new port, and re-joined after healing all 20 TCP links (`x = [17, 17, 17, 17, 17]`, `y = [18, 18, 18, 18, 18]`, `lost_committed_writes = 0`).
- **Official `jepsen/elle 0.2.8` Verdict:** Checking the exported `76`-event EDN history ([`raft_tcp_elle_history.edn`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/raft_tcp_elle_history.edn)) with Kyle Kingsbury's Clojure `elle/elle 0.2.8` (`rw-register`, `:strict-serializable`) returns **`{:valid? true}`** ([`raft_tcp_elle_official_verdict.txt`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/raft_tcp_elle_official_verdict.txt)).
- **Candid Consensus Findings:**
  - Building this real TCP test uncovered that `RaftNode::propose()` previously returned `Ok(index)` upon local leader WAL append before majority replication; `v0.6.0` added `RaftNode::propose_and_commit(&self, command, timeout)` (enforcing `commit_index >= index` + Raft §5.4.2 current-term commit check) and removed vacuous `|| true` assertions in `raft_chaos.rs`.
  - `RaftNode::with_state_machine` reads local state-machine memory without an automatic `ReadIndex` round-trip, so callers requiring linearizable reads during partitions must precede reads with a quorum barrier.
  - Note: the separate HTTP endpoint `POST /api/v1/certify/geo_raft_jepsen` (`distributed_jepsen_consensus.rs`) is an **in-memory 5-struct simulation** (`simulation_only: true`, `est_j`), not a multi-process Jepsen run.

---

## 3. `.jsyn` (`JSYN-v1`) Sub-1-Bit Weights: Measured Quality vs. Compression

| Checkpoint / Layer | `BPW` | Shrink vs `FP16` | Measured Fidelity vs Upstream | Status in `v0.6.0` |
|---|---:|---:|---|---|
| **[`dcharlot65-openie/all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit)** (`6L` Encoder) | `0.4378` | `22.23×` (`1.95 MB`) | **`0.9441` sentence-embedding cosine** vs `FP32` | **Published & Verified on Hugging Face** (`Apache-2.0`) |
| `Gemma-4-E2B` `q_proj` (`r = 8 -> 128` rank sweep) | `0.028–0.166` | `564× -> 96.6×` | `0.271 -> 0.455` output cosine (`KL 3.87 -> 2.72`, `65.6%` Top-5) | **Retracted from HF** (not drop-in LLM fidelity) |
| `Inkling-2-Mini` `w13` (`128`-Exp MoE slice, `r = 8 -> 128`) | `0.023–0.134` | `701× -> 120×` | `0.113 -> 0.350` output cosine (`KL 2.91 -> 2.59`) | **Retracted from HF** (not drop-in LLM fidelity) |

---

## 4. In-Process Subsystem Smoke Tests (`jouledb self-check`, `est_j`)

The fast HTTP/CLI self-checks (`jouledb self-check`, `POST /api/v1/bench/industry_suite`, `POST /api/v1/bench/analytic_suite`) execute in `5–40 ms` over small synthetic inputs and report analytical energy estimates (**`energy_evidence: "est_j"`**). They verify internal data-structure invariants during development and **must not** be cited as official TPC-C, ClickBench, JOB, STAC-M3, or Jepsen benchmark results:

- **SSI Counter Conservation Smoke (`tpc_c_ssi`):** Single-threaded in-memory loop of `500–1,000` synthetic warehouse balance transfers checking `ytd_conserved = true`.
- **Columnar Codec & ZoneMap Smoke (`alp_fsst`, `zonemap_scan`):** Lossless `ALP + FSST` round-trip and `1,024`-row granule min/max pruning over `100k` synthetic integers.
- **Morsel Spill Smoke (`morsel_spill`):** `15,000 ⋈ 3,000` syntheticGrace hash join spilling `CRC-32` partitions under a `16 KiB` memory cap.
- **Adya / Hermitage Cycle Detector Smoke (`elle_hermitage`):** Checks `6` small scripted isolation schedules (`G0`, `G1a/b`, `P4`, `G2-item`) via Tarjan `SCC`.
- **Double-Entry Ledger Fault Smoke (`tiger_ledger`):** `i128` linked two-phase transfers under simulated torn-WAL / bit-flip buffers.
- **Amorphic Streaming Run (`streaming_continuum`):** Single-node streaming pass over `1.35 GB` (`47.5M` atoms across 6 file formats) in `57.64 MB` Peak RSS.
