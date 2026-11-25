# Benchmarks & Consensus Verification (`bench/`)

This directory contains JouleDB's external comparison and fault-injection harnesses, along with raw machine-readable outputs under [`bench/results/`](./results/).

## 1. Official ClickBench (`hits.parquet` + 43 Queries) vs. DuckDB (`clickbench_official.py`)

Unlike `harness.py` (which generates a small synthetic 6-column table), [`clickbench_official.py`](./clickbench_official.py) loads rows directly from the official ClickBench [`hits.parquet`](https://datasets.clickhouse.com/hits_compatible/hits.parquet) dataset (`14.78 GB`, `99,997,497` rows, `105` columns) into both **DuckDB `2.0.1`** and **JouleDB `0.6.0`** (over PostgreSQL v3 wire protocol via `psycopg3`) and runs all **43 official ClickBench SQL queries (`Q0..Q42`)** unchanged, metering whole-package energy via Linux `/sys/class/powercap/intel-rapl` (`reported_j`).

```bash
python3 bench/clickbench_official.py \
  --jouledb-bin ./target/release/jouledb \
  --parquet /path/to/hits.parquet \
  --queries /path/to/ClickBench/duckdb/queries.sql \
  --copy-dir /path/to/staging \
  --rows 1000000 \
  --query-timeout-s 10 \
  --out bench/results/clickbench_official_1m.json
```

### Summary on `framework-128gb-0000` (AMD Ryzen AI Max+ 395, 32 Threads, 128 GB RAM, Linux RAPL `reported_j`)

| Slice (`hits.parquet`, 105 cols) | Load Time (`DuckDB` vs `JouleDB`) | JouleDB RSS After Load | Queries Completed (`<10s`) | Exact Row Match vs DuckDB | Faster Engine (`JouleDB` / `DuckDB`) | Geomean Latency (`JouleDB / DuckDB`) | Geomean RAPL Energy (`reported_j`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`100,000` rows** ([`clickbench_official_100k.json`](./results/clickbench_official_100k.json)) | `0.192 s` vs `0.203 s` | `722.2 MB` | **`42 / 43`** (`0` crashes, `1` timeout `Q29`) | `33 / 43` (`8` tie/epoch-format diffs, `1` `AVG` bug `Q32`) | `10` / `32` | **`26.37x` slower** (`median 17.03x`) | **`44.96x` more J** |
| **`1,000,000` rows** ([`clickbench_official_1m.json`](./results/clickbench_official_1m.json)) | `0.153 s` vs `1.125 s` | `4,177.2 MB` | **`23 / 43`** (`20` timeouts on AST fallback) | `18 / 23` completed | `2` / `21` | **`10.19x` slower** *(on 23 completed)* | **`12.76x` more J** *(on 23 completed)* |

**Candid Findings & Current Engine Limits:**
- **Memory Scaling Ceiling (~`4.18 KB/row` across 105 columns):** JouleDB's current `COPY FROM parquet` caches decoded 105-column vectors in memory (`722 MB` at `100k` rows, `4.18 GB` at `1M` rows). Loading the full `99,997,497`-row `hits.parquet` without out-of-core column eviction would require `~418 GB` RAM and is not yet supported on a `128 GB` host.
- **Vectorized Fast Path vs. Row-AST Fallback:** Simple aggregates and grouped `COUNT`/`SUM`/`COUNT(DISTINCT)` queries handled by `columnar_exec.rs` (`Q0..Q16`, `Q20`, `Q33`) complete in `1.1–140 ms` on `1M` rows (still `4x–42x` behind DuckDB on most grouped scans, while faster on `Q2` and `Q6`). Queries using `EXTRACT`/`DATE_TRUNC` (`Q18`, `Q42`), `REGEXP_REPLACE` (`Q28`), `90`-projection `SUM(x + k)` (`Q29`), `SELECT * ... ORDER BY` (`Q23`), or string-literal predicates against integer `EventDate` (`Q36..Q41`) fall back to the row-at-a-time AST executor and exceed `10 s` at `1M` rows.
- **Correctness Bug Uncovered (`Q32`):** Unfiltered `GROUP BY WatchID, ClientIP` (`Q32`) returned `0.0` for `AVG(ResolutionWidth)` despite matching group counts; tracked as an open pre-1.0 query-execution bug. (An empty-result `GROUP BY` index-out-of-bounds panic on `Q41` was also uncovered by this harness and fixed in `v0.6.0`.)

---

## 2. 5-Node Real TCP Raft Fault-Injection + Official Clojure `elle 0.2.8` (`elle_check.clj`)

- **Rust Fault-Injection Test:** [`crates/joule-db-server/tests/raft_tcp_elle_fault_injection.rs`](../crates/joule-db-server/tests/raft_tcp_elle_fault_injection.rs) boots 5 real `RaftNode<KvStateMachine, TcpRaftTransport>` servers over loopback TCP sockets routed through a 20-link socket-severing `TcpPartitionMatrix` with on-disk `raft.wal` persistence.
- **Injected Nemeses:** Isolates the initial leader in a `2/5` minority partition (verifying minority write & quorum-read rejection), elects a new higher-term leader in the `3/5` majority partition, stops a follower and restarts it from on-disk `raft.wal` on a new ephemeral port, heals all links, and verifies 5-node state convergence (`x = [17; 5]`, `y = [18; 5]`, `lost_committed_writes = 0`).
- **Official `elle` Verdict ([`raft_tcp_elle_official_verdict.txt`](./results/raft_tcp_elle_official_verdict.txt)):** Checked with Kyle Kingsbury's Clojure `elle/elle 0.2.8` (`rw-register`, `:strict-serializable`) over [`raft_tcp_elle_history.edn`](./results/raft_tcp_elle_history.edn) (`76` events): **`{:valid? true}`**.

```bash
cargo test -p joule-db-server --test raft_tcp_elle_fault_injection -- --nocapture
clojure -Sdeps '{:deps {elle/elle {:mvn/version "0.2.8"} spootnik/unilog {:mvn/version "0.7.31"}}}' \
  -M bench/elle_check.clj bench/results/raft_tcp_elle_history.edn
```

---

## 3. Synthetic 6-Column Micro-Harness (`harness.py` / `parquet_load.py`)

`harness.py` and `parquet_load.py` generate small synthetic seeded 6-column tables (`ROWS=10000` by default) for quick regression checks across developer laptops (`powermetrics` on macOS, `RAPL` on Linux). They are **not** the official ClickBench benchmark.
