# Benchmarks, Consensus Verification & Subsystem Smoke Tests (`v0.16.0`)

> **Audit Remediation & Evidence Standard (`v0.16.0`):** Earlier `v0.2.8`–`v1.0.0` documentation presented 36 in-process micro-harnesses and synthetic simulations as *"36 / 36 Certified Global Benchmark Pillars"*, including synthetic frontier AI exam routes (`frontier_exam_routes.rs`). **All synthetic frontier AI benchmark claims remain retracted and `frontier_exam_routes.rs` has been deleted.**
>
> Below we strictly separate **(1) reproducible external dataset, multi-table `TPC-H` (`SF=0.1..10.0`), physical-NVMe core-banking & multi-client `pgwire` OLTP, sub-millisecond lazy CoW branching, `LTCB` (`enwik8`/`enwik9`) `Joules-per-Bit` compression, & multi-host WAN consensus evaluations** (with every win, loss, memory profile, and architectural bottleneck reported openly) from **(2) fast in-process functional smoke checks**, and tag every energy figure as **`reported_j`** (hardware RAPL / SMC counters) or **`est_j`** / **`analytical_est`** (modeled).

---

## 1. Full Official ClickBench (`hits.parquet`, `99,997,497` Rows, `105` Columns, `43` Queries) vs. DuckDB `2.0.1`

- **Harness & Raw Artifacts:** [`bench/clickbench_official.py`](https://github.com/openIE-dev/jouledb/blob/main/bench/clickbench_official.py), [`clickbench_official_100k.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/clickbench_official_100k.json), [`clickbench_official_1m.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/clickbench_official_1m.json), [`clickbench_official_10m.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/clickbench_official_10m.json), **[`clickbench_official_100m.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/clickbench_official_100m.json)**.
- **Methodology:** Streams the official `14.78 GB` (`99,997,497`-row, `105`-column) ClickBench `hits.parquet` file into DuckDB `2.0.1` (in-process) and JouleDB `0.16.0` (over TCP `pgwire` via `psycopg 3.3.9`, `--data-parent-dir /var/tmp` physical NVMe on `100M`, spilling immutable `262,144`-row `.jcol` segments), then runs all **43 official ClickBench SQL queries (`Q0..Q42`)** unchanged on `framework-128gb-0000` (`AMD Ryzen AI Max+ 395`, 32 logical cores, `128 GB` UMA RAM, Linux `rapl-package` **`reported_j`**).

| Dataset Scale (`hits.parquet`) | Parquet Load (`DuckDB` vs `JouleDB`) | JouleDB Post-Load RSS (`VmHWM` Peak) | Queries Succeeded | SQL-Valid Match (`Exact`) | Faster (`JouleDB` / `DuckDB`) | Geomean Latency (`JouleDB / DuckDB`) | Geomean Energy (`reported_j`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`100,000` rows** (`105` cols) | `0.179 s` (`11.2 J`) vs `0.265 s` (`18.3 J`) | `917.8 MB` (`2,651.5 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`34` exact) | **`20`** / `23` | **`1.47×` slower** (`median 1.19×`) | **`1.97×` more J** |
| **`1,000,000` rows** (`105` cols) | `0.158 s` (`15.8 J`) vs `0.871 s` (`77.7 J`) | `1,282.9 MB` (`1,922.8 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`29` exact) | **`11`** / `32` | **`2.34×` slower** (`median 2.95×`) | **`2.50×` more J** |
| **`10,000,000` rows** (`105` cols) | `2.422 s` (`336.4 J`) vs `13.340 s` (`1,850.8 J`) | `2,384.7 MB` (**`3,448.4 MB`**) | **`43 / 43`** (`0` err) | **`43 / 43`** (`31` exact) | **`9`** / `34` | **`2.28×` slower** (`median 2.63×`) | **`2.22×` more J** |
| **`99,997,497` rows — FULL `14.78 GB` NVMe** | `18.208 s` (`2,050.1 J`) vs `161.403 s` (`13,236.0 J`) | **`139.2 MB` (`9,454.5 MB`)** | **`43 / 43`** (`0` err) | **`43 / 43`** (`30` exact) | **`8`** / `35` | **`2.35×` slower NVMe** (`2.11×` `tmpfs`, `median 3.36×`) | **`2.09×` more J** |

### Selected Per-Query Results on the Full `99,997,497`-Row (`100M`) Dataset (`reported_j`)

| Query | SQL Summary | DuckDB `2.0.1` | JouleDB `0.16.0` (`.jcol` + `pgwire`) | Outcome |
|---|---|---:|---:|---|
| `Q0` | `SELECT COUNT(*) FROM hits` | `2.58 ms` | **`1.40 ms`** | Exact match (**JouleDB `1.84×` faster** — segment metadata) |
| `Q2` | `SELECT SUM(AdvEngineID), COUNT(*), AVG(ResolutionWidth) FROM hits` | `8.69 ms` | **`1.16 ms`** | Exact match (**JouleDB `7.48×` faster** — zero-I/O `global_int_sum`) |
| `Q3` | `SELECT AVG(UserID) FROM hits` | `40.13 ms` | **`1.56 ms`** | Exact match (**JouleDB `25.7×` faster** — zero-I/O `global_int_sum`) |
| `Q6` | `SELECT MIN(EventDate), MAX(EventDate) FROM hits` | `6.82 ms` | **`1.35 ms`** | Exact match (**JouleDB `5.05×` faster** — segment ZoneMap) |
| `Q29` | `SELECT SUM(ResolutionWidth), SUM(ResolutionWidth + 1), ...` (`90` sums) | `14.18 ms` | **`1.81 ms`** | Exact match (**JouleDB `7.82×` faster** — algebraic `global_int_sum`) |
| `Q37` | `SELECT URL, COUNT(*) ... WHERE CounterID = 62 ... LIMIT 10` | `32.70 ms` | **`30.48 ms`** | SQL-valid tie (**JouleDB `1.07×` faster** — ZoneMap + raw `.jcol` filter) |
| `Q41` | `SELECT URLHash, EventDate, COUNT(*) ... WHERE CounterID = 62 ...` | `24.40 ms` | **`11.86 ms`** | Exact match (**JouleDB `2.06×` faster** — inline `HKey` + ZoneMap) |
| `Q42` | `SELECT DATE_TRUNC('minute', EventTime), COUNT(*) WHERE CounterID = 62 ...` | `36.31 ms` | **`11.31 ms`** | Exact match (**JouleDB `3.21×` faster** — ZoneMap + `WHERE`-first skip) |
| `Q7` | `SELECT MIN(EventDate), MAX(EventDate) FROM hits WHERE CounterID = 62` | `6.06 ms` | **`7.40 ms`** | Exact match (**`4.73×` faster than `v0.15.0`'s `34.95 ms`**) |
| `Q24` | `SELECT * FROM hits WHERE URL LIKE '%google%' ORDER BY EventTime LIMIT 10` | `7.82 ms` | **`16.65 ms`** | Exact match (zero-`Arc<str>` `eval_raw_text_filter`) |
| `Q30` | `SELECT SUM(ResolutionWidth), ... FROM hits WHERE ResolutionWidth > 0` | `103.25 ms` | **`270.87 ms`** | Exact match (**`2.47×` faster than `v0.15.0`'s `668.50 ms`**) |
| `Q32` | `SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) GROUP BY ...` | `667.69 ms` | **`1,467.50 ms`** | SQL-valid tie (**`4.04×` faster than `v0.15.0`'s `5,933.50 ms`**, `2.20×` vs DuckDB) |
| `Q39` | `SELECT TraficSourceID, SearchEngineID, AdvEngineID, CASE WHEN ... GROUP BY ...` | `26.88 ms` | **`297.01 ms`** | Exact match (**`3.05×` faster than `v0.15.0`'s `906.85 ms`**, inline `GKey::IIISS`) |

---

## 2. Official 22-Query `TPC-H` Relational Suite (`tpch_queries()`, 8 Tables, `SF = 0.1`, `1.0` & `10.0`) vs. DuckDB `2.0.1`

- **Harness & Raw Artifacts:** [`bench/tpch_official.py`](https://github.com/openIE-dev/jouledb/blob/main/bench/tpch_official.py), [`tpch_official_sf01.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/tpch_official_sf01.json), [`tpch_official_sf1.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/tpch_official_sf1.json), **[`tpch_official_sf10.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/tpch_official_sf10.json)**.
- **Methodology:** Generates standard 8-table `TPC-H` Parquet fixtures via DuckDB `dbgen(sf=...)` on physical NVMe (`--work-dir /var/tmp`), loads them into JouleDB `0.16.0` over TCP `pgwire`, and runs all **22 official `tpch_queries()` (`Q01..Q22`)** unmodified via `columnar_join.rs` (`PK-FD` group-by compression + 3-hop semi-join pushdown + 64-shard `NeqExists`) against DuckDB `2.0.1` (`rapl-package` **`reported_j`**).

| `TPC-H` Scale | Total Rows (8 Tables) | Parquet Load (`DuckDB` vs `JouleDB`) | Post-Load RSS (`VmHWM` Peak) | Verified Exact (`Faster`) | Total 22-Query Time (`DuckDB` vs `JouleDB`) | Geomean Latency (`JouleDB / DuckDB`) |
|---|---:|---:|---:|---:|---:|---:|
| **`SF = 0.1`** | `866,602` rows | `0.430 s` (`12.2 J`) vs `1.380 s` (`54.5 J`) | **`780.8 MB`** (`939.3 MB`) | **`22 / 22` (`5` wins)** | `88.2 ms` vs **`253.0 ms`** | **`1.99×` slower** (`median 2.16×`) |
| **`SF = 1.0`** | `8,661,245` rows (`6.0M` `lineitem`) | `1.200 s` (`65.0 J`) vs `4.707 s` (`143.8 J`) | **`201.6 MB`** (**`578.5 MB`**) | **`22 / 22` (`0` err)** | `238.6 ms` vs **`927.5 ms`** (**sub-second**, `1.71×` vs `v0.15`) | **`3.61×` slower** (`median 3.59×`, down from `6.18×`) |
| **`SF = 10.0`** | **`86,586,082` rows (`60.0M` `lineitem`)** | `5.839 s` (`508.5 J`) vs `46.135 s` (`1,373.0 J`) | **`754.9 MB`** (**`4,259.3 MB`**) | **`22 / 22` (`0` err)** | `945.6 ms` vs **`7,441.0 ms`** (**`2.18×` faster vs `v0.15`**) | **`6.41×` slower** (`median 6.51×`, down from `16.03×`) |

### Selected Per-Query `TPC-H` Latencies (`SF = 0.1`, `SF = 1.0` & `SF = 10.0`, `reported_j`)

| Query | Relational Pattern | `SF = 0.1` (`JDB` vs `DuckDB`) | `SF = 1.0` (`JDB` vs `DuckDB`) | `SF = 10.0` (`JDB` vs `DuckDB`) |
|---|---|---|---|---|
| `Q10` | 4-table quarter return revenue (`PK-FD` 20-row Top-K) | `24.75 ms` vs `6.00 ms` (`4.13×`) | **`49.78 ms`** vs `22.63 ms` (**`2.20×`**, **`8.02×` faster than `v0.15`**) | **`342.78 ms`** vs `77.21 ms` (**`4.44×`**, **`13.57×` faster than `v0.15`'s `4.65 s`!**) |
| `Q21` | `EXISTS` + `NOT EXISTS` (`l2.l_suppkey <> l1.l_suppkey`) | `14.1 ms` vs `4.9 ms` (`2.88×`) | **`142.00 ms`** vs `15.28 ms` (**`3.15×` faster than `v0.15`**) | **`573.18 ms`** vs `68.29 ms` (**`8.39×`**, **`3.75×` faster than `v0.15`'s `2.15 s`!**) |
| `Q16` | `partsupp ⋈ part` + `NOT IN (SELECT s_suppkey ...)` | **`5.07 ms`** vs `5.24 ms` (**`0.97×` win**) | **`42.15 ms`** vs `18.41 ms` (`2.29×`) | **`153.09 ms`** vs `42.04 ms` (**`3.64×`**, `1.37×` faster than `v0.15`) |
| `Q18` | `IN (SELECT l_orderkey GROUP BY HAVING SUM > 300)` | `34.69 ms` vs `4.53 ms` (`7.65×`) | **`60.98 ms`** vs `14.96 ms` (`4.08×`) | **`621.63 ms`** vs `103.80 ms` (**`5.99×`**, `1.33×` faster than `v0.15` & `18.2×` vs `v0.11`) |
| `Q04` | `orders` quarter + `EXISTS (lineitem l_commitdate < l_receiptdate)` | `3.1 ms` vs `2.1 ms` (`1.48×`) | **`10.73 ms`** vs `7.07 ms` (**`1.52×`**) | **`67.12 ms`** vs `39.68 ms` (**`1.69×`**) |
| `Q22` | `NOT EXISTS` + `AVG` scalar subquery + `SUBSTRING` | **`1.97 ms`** vs `3.73 ms` (**`0.53×` win**) | `35.28 ms` vs `17.37 ms` (`2.03×`) | **`78.62 ms`** vs `34.75 ms` (`2.26×`) |

---

## 3. Head-to-Head `100`-Warehouse Physical-NVMe TCP `pgwire` OLTP vs. `PostgreSQL 16.16` & `SQLite 3.53.1`

- **Harness & Raw Artifact:** [`bench/oltp_pgwire.py`](https://github.com/openIE-dev/jouledb/blob/main/bench/oltp_pgwire.py), [`oltp_pgwire_100wh.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/oltp_pgwire_100wh.json) (`--work-dir /var/tmp` NVMe, `idle_watts = 10.03 W`, `invariant_verified = true` across all 5 runs).

| Engine & Wire Mode (`invariant_verified = true`) | `c = 1` TPS (`p50`) | `c = 4` TPS (`p50`, `p95`) | `c = 8` TPS (`p50`, `p95`) | `c = 16` TPS (`p50`, `mJ/txn`) | Committed Writes |
|---|---:|---:|---:|---:|---:|
| **JouleDB `0.16.0` (`pgwire` Simple, Decoupled `log_atomic_tx_fast`)** | **`3,697.98`** (`0.137 ms`) | **`5,069.30`** (`0.537 ms`, **`1.90 ms`**) | `4,094.67` (`0.676 ms`, `6.71 ms`) | **`5,474.86`** (`0.926 ms`, `11.94 mJ`) | `43,783` |
| **JouleDB `0.16.0` (`pgwire` Pipeline, Decoupled `log_atomic_tx_fast`)** | `1,083.79` (`0.137 ms`) | **`2,486.05`** (`0.579 ms`, **`5.06 ms`**) | `4,342.65` (`0.633 ms`, `5.59 ms`) | **`6,364.75`** (`0.851 ms`, **`9.57 mJ`**) | `34,071` |
| **PostgreSQL `16.16` (`pgwire` Simple, 4-RTT + NVMe `fsync`)** | `1,139.42` (`0.073 ms`) | `2,614.86` (`0.089 ms`, `5.11 ms`) | **`4,901.20`** (`0.148 ms`, `5.47 ms`) | **`10,081.67`** (`0.250 ms`, `3.98 mJ`) | `44,780` |
| **PostgreSQL `16.16` (`pgwire` Pipeline, 1-RTT + NVMe `fsync`)** | **`1,161.69`** (`0.071 ms`) | `2,386.67` (`0.103 ms`, `5.74 ms`) | **`5,275.86`** (`0.126 ms`, `5.10 ms`) | **`10,205.75`** (`0.238 ms`, `3.68 mJ`) | `45,496` |
| **SQLite `3.53.1` (Embedded in-proc C API, NVMe `WAL`, `NORMAL`)** | `44,204.91` (`0.005 ms`) | `87,633.29` (`0.006 ms`, `0.04 ms`) | `98,374.22` (`0.007 ms`, `0.06 ms`) | **`111,079.64`** (`0.007 ms`, `0.36 mJ`) | `845,139` (in-proc) |

### 3.2 Head-to-Head Core Banking Double-Entry Ledger (`4 / 4` Regime Sweep vs. `PostgreSQL 16.16`)

- **Harness & Raw Artifact:** [`bench/ledger_banking_head_to_head.py`](https://github.com/openIE-dev/jouledb/blob/main/bench/ledger_banking_head_to_head.py), [`ledger_banking_nvme_rapl_v016.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/ledger_banking_nvme_rapl_v016.json), [`unified_banking_agentic_v016.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/unified_banking_agentic_v016.json) (`493,941` HTTP & `436,801` `pgwire` SHA-256 chain entries verified `valid = true`).

| Core Banking Regime (`fsync=on`) | JouleDB `0.16.0` HTTP (`JWL1` + SHA-256) | JouleDB `0.16.0` `pgwire` (`psycopg3` SQL) | PostgreSQL `16.16` (`FOR UPDATE` / `UNNEST`) | Speedup & RAPL Energy (`reported_j`) |
|---|---:|---:|---:|---|
| **`c = 1` Serial Durable Transfer** | **`3,424.5 TPS`** (`p50 0.242 ms`, **`26.96 mJ/tx`**) | **`400.6 TPS`** (`p50 2.601 ms`, `201.89 mJ/tx`) | `371.9 TPS` (`p50 2.781 ms`, `129.73 mJ/tx`) | **JouleDB `9.21×` HTTP (`1.08×` `pgwire`) · `4.81×` less J/tx** |
| **`c = 4` Concurrent Group Commit** | **`1,250.0 TPS`** (`p50 2.993 ms`, `62.58 mJ/tx`) | **`1,412.7 TPS`** (`p50 2.761 ms`, **`56.08 mJ/tx`**) | `777.6 TPS` (`p50 5.271 ms`, `61.57 mJ/tx`) | **JouleDB `1.82×` `pgwire` (`1.61×` HTTP) · `1.10×` less J/tx** |
| **`c = 16` High-Contention Concurrent** | `2,636.9 TPS` (`p50 5.129 ms`, `44.07 mJ/tx`) | **`3,919.0 TPS`** (`p50 3.595 ms`, `22.56 mJ/tx`) | `2,861.0 TPS` (`p50 5.658 ms`, `20.51 mJ/tx`) | **JouleDB `1.37×` `pgwire` win** + global SHA-256 chain |
| **`batch = 500` Clearing Batch (`1` `fdatasync` / `500` tx)** | **`91,316.1 TPS`** (`p50 4.290 ms`, `1.06 mJ/tx`) | **`81,453.5 TPS`** (`104,721 TPS` peak) | `53,956.4 TPS` (`p50 8.556 ms`, `1.03 mJ/tx`) | **JouleDB `1.69×` faster (`>100k TPS` peak)** |

- **Sub-Millisecond Lazy CoW SQL Branching (`unified_banking_agentic_v016.json`):** Pure `CREATE BRANCH ... GUARDRAILS STRICT` completes in **`0.553 ms` median on Linux (`0.432 ms` on Apple Silicon — `69×` faster than `v0.14.0`)**.

---

## 4. 5-Node Multi-Host Cross-Machine WAN/Tailscale TCP Raft (`ReadIndex`) + Official Clojure `elle 0.2.8`

- **Harnesses & Verifier:** [`bench/raft_multihost_wan.py`](https://github.com/openIE-dev/jouledb/blob/main/bench/raft_multihost_wan.py), [`examples/raft_wan_node.rs`](https://github.com/openIE-dev/jouledb/blob/main/crates/joule-db-server/examples/raft_wan_node.rs), and [`bench/elle_check.clj`](https://github.com/openIE-dev/jouledb/blob/main/bench/elle_check.clj) -> **`{:valid? true}`** (`270` WAN events, `:strict-serializable`, [`raft_wan_multihost_elle_verdict.txt`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/raft_wan_multihost_elle_verdict.txt)).

---

## 5. `.jsyn` Sub-1-Bit Weights & Large Text Compression Benchmark (`LTCB` `enwik8`/`enwik9` `Joules-per-Bit` Pareto)

- **`.jsyn` Encoder Fidelity (`0.4378 BPW`):** [`dcharlot65-openie/all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit) (`1.95 MB`, `22.23×` smaller than `FP16`) achieves **`0.9441` sentence-embedding cosine** vs `FP32` (81 low-BPW causal LLM exports remain retracted).
- **`LTCB` `enwik8`/`enwik9` `Joules-per-Bit` & Seekable SQL Suite ([`bench/ltcb_energy_pareto.py`](https://github.com/openIE-dev/jouledb/blob/main/bench/ltcb_energy_pareto.py), [`ltcb_energy_pareto_v016.json`](https://github.com/openIE-dev/jouledb/blob/main/bench/results/ltcb_energy_pareto_v016.json)):**
  - **Published `enwik9` (`1 GB`) Neural Anchors (`analytical_est`):** `cmix-lex-transformer` `95.84 MB` (`0.7667 bpb`, `213,840 s` decode, `~668.3k nJ/bit`), `zmix 1.0` `96.10 MB` (`0.7688 bpb`, `131,184 s` decode, `~246.0k nJ/bit`), **`tufazip v2` (Tufa Labs)** `105.39 MB` (`0.8431 bpb`, **`630,136 s` [`7.29 days`] dual-B200 decode, `~157.5M nJ/bit`**, non-seekable).
  - **Live `enwik8` (`100 MB` = `12,346` articles) Hardware RAPL (`reported_j`):** `zstd -3` decodes at `577.0 MB/s` (`27.97 nJ/bit`), `zstd -19` at `177.7 MB/s` (`90.54 nJ/bit`), Ontology+SimHash semantic locality ordering shrinks bounded-window `gzip -6` by `1.68%`, `zstd -3` by `1.53%`, and seekable `Parquet Zstd` (`2.870 bpb`) by `1.46%`, while **JouleDB `0.16.0` `.jcol` SQL** executes full-corpus `LIKE` scans in **`28.36 ms` (`3.374 GB/s`, `2.16 J` = `2.83 nJ/bit` `reported_j` — `55.7M×` lower energy/bit)** and single-article SQL lookups in **`5.41 ms` (`116.4M×` faster than `tufazip v2` sequential decode)**.

---

## 6. In-Process Subsystem Smoke Tests (`jouledb self-check`, `est_j`)

The fast HTTP/CLI self-checks (`jouledb self-check`, `POST /api/v1/bench/industry_suite`, `POST /api/v1/bench/analytic_suite`) execute in `5–40 ms` over small synthetic inputs and report analytical energy estimates (**`energy_evidence: "est_j"`**). They verify internal data-structure invariants during development and **must not** be cited as official TPC-C, ClickBench, TPC-H, JOB, STAC-M3, or Jepsen benchmark results.
