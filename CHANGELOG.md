# Changelog

All notable changes to this release surface are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/).

> **Version & Audit Errata Notice (`2026-11-25`):** Following an independent source-level audit of `openIE-dev/jouledb`, the premature `v1.0.0` GA tag has been **retracted** (`crates.io` `jouledb 1.0.0` yanked; GitHub Release `v1.0.0` marked retracted), and the project has returned to **`0.x` (`v0.6.0` Pre-1.0 Research Preview)**. Earlier changelog entries (`[0.2.8]` through `[1.0.0]`) are preserved below for historical transparency, subject to the following explicit corrections:
> - **"36 / 36 Certified Pillars" & Frontier AI Exam Claims Retracted:** The synthetic benchmark routes (`frontier_exam_routes.rs` — `HLE-Diamond`, `SuperGPQA`, `SimpleQA`, `FaithEval`, `ARC-AGI-2`, `FrontierMath`, `SWE-bench`, `TAU-bench`, `MRCR`, `SafeVLA`) were synthetic/seeded in-process micro-harnesses or hard-coded loop simulations, not official benchmark runs. Those routes have been deleted, and remaining subsystem checks (`TPC-C` micro-loop, `ClickBench` codec smoke, `Hermitage`, `TigerBeetle` ledger conservation, `Geo-Raft` state-machine simulation) are explicitly labeled as **in-process functional smoke tests** with `energy_evidence: "est_j"`.
> - **Compliance Disclaimers:** No third-party SOC 2, ISO 27001, HIPAA, PCI-DSS, NIST CMVP, or FedRAMP audit has been conducted; `trust_compliance.rs` runs technical control self-tests only (`external_audit_performed: false`).
> - **`.jsyn` Hugging Face Fleet Pruned (`82 -> 1`):** 81 unverified ultra-low-BPW (`0.02–0.17 BPW`, `cos = 0.27–0.45`) or custom-vendor-licensed `.jsyn` causal LLM checkpoints have been deleted from `huggingface.co/dcharlot65-openie`; only the verified `Apache-2.0` sentence embedder `dcharlot65-openie/all-MiniLM-L6-v2-LittleBit` (`0.4378 BPW`, `cos = 0.9441` vs upstream `FP32`) remains published.

## [Unreleased]

## [0.15.0] - 2026-11-27

### Added
- **`104,721 TPS` Durable NVMe Core Banking `LEDGER`, Pre-Allocated `JWL1` Extents & Native `pgwire` Ledger SQL Functions (`crates/joule-db-server/src/{financial_ledger.rs, lib.rs}`, `bench/ledger_banking_head_to_head.py`)**:
  - Replaced `O_APPEND` per-sync `ext4` `jbd2` `i_size` metadata stalls in `FinancialLedger` with progressive zero-initialized WAL pre-allocation (`256 KiB -> 4 MiB -> 16 MiB` extents + `sync_data()`), upgraded account lookup to `O(1)` dense `Vec<LedgerAccount>` + `FxHashMap<String, usize>` with 72-byte `Copy` `AccountUndo`, replaced `format!` SHA-256 hashing with zero-allocation streaming `Sha256::update`, enabled `TCP_NODELAY` on Axum HTTP connections (`lib.rs`), added `include_receipts: bool` (`submit_transfer_batch_with_options`), and wired native `pgwire` SQL functions **`SELECT ledger_transfer($1,$2,$3,$4)`** and **`SELECT ledger_transfer_batch($1,$2,$3,$4 [, $5])`** (accepting PostgreSQL `{a,b}` text arrays, JSON arrays, or `ARRAY[...]`).
  - Measured head-to-head against **`PostgreSQL 16.16`** (`fsync=on`, `synchronous_commit=on`, `FOR UPDATE` + `sha256()` + `UNNEST` batch CTE) over `1,000` `NO_OVERDRAFT` customer wallets on physical NVMe (`--work-dir /var/tmp`, `bench/results/ledger_banking_nvme_rapl_v015.json`, Linux `rapl-package` **`reported_j`**, **`550,418` HTTP & `542,954` `pgwire` SHA-256 chain entries verified `valid = true`**):
    - **`c = 1` Serial Durable Transfer:** JouleDB HTTP **`3,544.6 TPS`** (`p50 0.245 ms`, **`25.10 mJ/tx`**, `1.89×` faster than `v0.14.0`) & JouleDB `pgwire` **`443.1 TPS`** (`p50 2.334 ms`) vs PostgreSQL `16.16` `387.2 TPS` (`p50 2.634 ms`, `187.12 mJ/tx`) — **`9.15×` faster (`1.14×` over `pgwire`) & `7.46×` lower energy/tx**.
    - **`c = 4` Concurrent Group Commit:** JouleDB HTTP `749.4 TPS` / `pgwire` `683.9 TPS` vs PostgreSQL `16.16` **`853.9 TPS`** (`1.14×` PostgreSQL edge on shared-host run).
    - **`c = 16` High-Contention Concurrent Transfers:** JouleDB HTTP **`3,461.3 TPS`** (`p50 4.572 ms`, `22.27 mJ/tx`, **`2.03×` jump from `v0.14.0`'s `1,706.6 TPS`**) & `pgwire` `2,430.6 TPS` vs PostgreSQL `16.16` **`3,454.1 TPS`** (`p50 4.351 ms`, `19.42 mJ/tx`) — **eliminating `v0.14.0`'s `2.05×` gap to achieve `1.00×` throughput parity/edge while maintaining a global genesis-to-head SHA-256 hash chain**.
    - **`batch = 500` Clearing-House Batch (`1` `fdatasync` / `500` tx):** JouleDB **`pgwire` `104,721.1 TPS`** (`p50 4.236 ms/batch`, **`0.54 mJ/tx`**) & JouleDB **HTTP `102,276.0 TPS`** (`p50 4.124 ms/batch`, **`0.69 mJ/tx`**) vs PostgreSQL `16.16` `UNNEST` CTE **`54,212.3 TPS`** (`p50 8.511 ms/batch`, `0.97 mJ/tx`) — **surpassing `100,000 TPS` on durable NVMe (`1.93×` faster & `1.81×` lower energy/tx than `PostgreSQL 16.16`, `3.33×` faster than `v0.14.0`, and `293×` faster than `v0.13.0`)**.
- **Sub-Millisecond (`0.605 ms`) Lazy `O(1)` CoW SQL Branching & Native PostgreSQL `pgvector` HNSW SQL Parity (`crates/joule-db-server/src/{sql_branch_sandbox.rs, query.rs}`, `crates/joule-db-query/src/{sql.rs, functions.rs}`, `tests/clients/psycopg3.py`)**:
  - Upgraded `SqlBranchEntry` to lazy `OnceLock<Arc<SimpleQueryExecutor>>` + `Arc<Vec<RowData>>` snapshots + demand-driven `materialize_referenced_branch_tables` (`amorphic().insert_rows()`), slashing pure `CREATE BRANCH` median latency **`63×` from `38.28 ms` -> `0.605 ms` on Linux (`0.432 ms` on Apple Silicon)** (`bench/results/unified_banking_agentic_v015.json`).
  - Added full PostgreSQL `pgvector` SQL compatibility: `VECTOR(dim)` columns, `'[...]'::vector` string/cast literals, `<->` (L2), `<=>` (cosine), and `<#>` (negative inner product) distance operators with `$1` bind parameter support, and `CREATE INDEX [name] ON tbl USING hnsw | ivfflat (col vector_cosine_ops | vector_l2_ops | vector_ip_ops)` wired directly into HNSW/IVF/LSH KNN index pushdown (`psycopg3.py` expanded to **`12 / 12` `ok`** alongside `SQLAlchemy 2` **`4 / 4` `ok`**).

### Fixed
- **Pre-Allocated `WalManager` Extents (`1.43×–1.90×` Faster `c=4..8` NVMe `pgwire` OLTP) & `3.88×` Lower Peak `VmHWM` on `TPC-H` `SF=10.0` (`crates/joule-db-local/src/storage/wal.rs`, `crates/joule-db-server/src/columnar_join.rs`)**:
  - Added `allocated_len: AtomicU64`, single-pass `scan_wal_tail` recovery, and `ensure_wal_capacity` (`256 KiB -> 4 MiB -> 16 MiB` zero-filled extents) in `WalManager`, boosting `100`-warehouse NVMe `pgwire` OLTP (`bench/results/oltp_pgwire_100wh.json`, `invariant_verified = true`) at **`c = 4` by `1.90×` Simple (`1,005.0 -> 1,911.5 TPS`, `p95` halved `15.04 -> 7.44 ms`) / `1.81×` Pipelined (`1,149.0 -> 2,078.2 TPS`, `1.23×` vs PG16)** and at **`c = 8` by `1.43×` Simple (`2,617.9 -> 3,743.2 TPS`) / `1.46×` Pipelined (`2,730.9 -> 3,981.6 TPS`, `1.23×` vs PG16)**.
  - Added top-down semi-join `KeyIn` pushdown (`collect_surviving_fk_set`), cross-branch 1-key-to-2-key `KeyIn` pushdown in `exec_bushy_plan`, worker-folded 48-byte `CompactNumAgg` (`Q18`), and 64-shard `BorrowKeyElem` with deferred Top-K string materialization (`Q10`) in `columnar_join.rs` (`tpch_official_{sf1,sf10}.json`):
    - **`SF = 1.0` (`8.66M` rows):** `22 / 22` exact in **`1,587.2 ms`** (`4.15×` geomean vs DuckDB `2.0.1`, down from `4.33×`), `204.0 MB` post-load RSS, **`574.7 MB` peak `VmHWM` (`2.72×` down from `1,561.8 MB`)**, **`Q18` slashed `4.26×` (`260.34 -> 61.13 ms`)**, `Q10` cut `546.87 -> 399.22 ms`.
    - **`SF = 10.0` (`86.6M` rows, `60M` `lineitem`):** `22 / 22` exact, `757.8 MB` post-load RSS, **peak `VmHWM` slashed `3.88×` (`15,586.4 MB -> 4,012.8 MB`)**, **`8.17×` geomean (`8.04×` median, down from `8.49×` in `v0.14.0` & `16.03×` in `v0.11.0`)**, **`Q18` slashed `2.17×` (`1,799.8 -> 827.7 ms`, `5.56×` vs DuckDB)**, **`Q16` cut `1.59×` (`333.6 -> 209.6 ms`, `3.83×`)**, `Q09` cut `1,680 -> 1,490.7 ms`, `Q10` cut `5,230.7 -> 4,653.0 ms`.

## [0.14.0] - 2026-11-27

### Added
- **Binary `JWL1` CRC-32 Core Banking WAL, Single-`fdatasync` Settlement Batching & FIFO Leader-Follower Group Commit (`crates/joule-db-server/src/{financial_ledger.rs, mcp_bridge.rs, mcp_transport.rs}`, `bench/ledger_banking_head_to_head.py`)**:
  - Upgraded `FinancialLedger` (`#![deny(clippy::unwrap_used)]`) to compact binary **`JWL1`** frames (`0x89 'J' 'W' 'L' 0x01` + `u32` length + IEEE `CRC-32` + raw 32-byte SHA-256 digests, seamlessly backward-compatible with JSONL `.jwal` logs), added `submit_transfer_batch(reqs, atomic)` (`POST /api/v1/ledger/transfers/batch`, SQL `LEDGER BATCH [NON_ATOMIC] TRANSFER ...`, and MCP `ledger.transfer_batch` — expanding the MCP server to **`59` tools**), and added zero-starvation FIFO single-wave Leader-Follower group-commit coalescing on concurrent persistent `submit_transfer` callers.
  - Measured head-to-head against **`PostgreSQL 16.16`** (`fsync=on`, `synchronous_commit=on`, deadlock-free `FOR UPDATE` + `sha256()` + set-based `UNNEST` batch CTE) over `1,000` `NO_OVERDRAFT` customer wallets on physical NVMe (`--work-dir /var/tmp`, `bench/results/ledger_banking_nvme_rapl_v014.json`, Linux `rapl-package` **`reported_j`**, `183,610` SHA-256 chain entries verified `valid = true`):
    - **`c = 1` Serial Durable Transfer:** JouleDB **`1,873.2 TPS`** (`p50 0.559 ms`, `37.41 mJ/tx`) vs PostgreSQL `16.16` `420.7 TPS` (`p50 2.379 ms`, `115.42 mJ/tx`) — **`4.45×` higher throughput & `3.09×` lower energy/tx**.
    - **`c = 4` Concurrent Group Commit:** JouleDB **`1,485.9 TPS`** (`p50 1.068 ms`, `41.58 mJ/tx`) vs PostgreSQL `16.16` `914.1 TPS` (`p50 4.432 ms`, `56.57 mJ/tx`) — **`1.63×` higher throughput & `1.36×` lower energy/tx**.
    - **`c = 16` High-Contention Concurrent Transfers:** JouleDB `1,706.6 TPS` (`p50 9.291 ms`, `41.25 mJ/tx`) vs PostgreSQL `16.16` **`3,504.0 TPS`** (`p50 4.624 ms`, `18.50 mJ/tx`) — PostgreSQL wins `2.05×` via tuple-level `FOR UPDATE` locking vs JouleDB's global rolling SHA-256 chain mutex.
    - **`batch = 500` Clearing-House Batch (`1` `fdatasync` / `500` tx):** JouleDB **`31,463.2 transfers/sec`** (`p50 15.313 ms` per 500-tx batch, **`2.08 mJ/tx`**, `88.1×` faster than `v0.13.0`'s `357 TPS`) vs PostgreSQL `16.16` `UNNEST` CTE **`51,062.1 transfers/sec`** (`p50 8.745 ms`, `1.62 mJ/tx` without genesis-to-head rolling hash chain).
- **Zero-`xfail` `SQLAlchemy 2` + `pandas.DataFrame.to_sql` + `Inspector` Parity (`crates/joule-db-query/src/{sql.rs, functions.rs}`, `crates/joule-db-server/src/query.rs`, `tests/clients/sqlalchemy_test.py`)**:
  - Added PostgreSQL `Token::DoubleColon` (`expr::type` / `::VARCHAR[]` / `::regclass`), quantified array comparisons (`op ANY(expr)` / `op SOME(expr)` / `op ALL(expr)`), `pg_table_is_visible` / `pg_type_is_visible` / `pg_function_is_visible` / `current_schemas` / `format_type`, `pg_class.relpersistence`, and MVCC `session_id.is_none()` read-your-own-writes isolation on `try_columnar_select` — eliminating the final 2 `xfail`s in `tests/clients/sqlalchemy_test.py` (**`4 / 4` `ok`: `core_text`, `pandas_read_sql`, `pandas_to_sql`, `inspector_tables`**).

### Fixed
- **100% Green Workspace `Engine` Test Suite (`crates/joule-db-weights/src/waios_block64.rs`, `crates/joule-db-server/src/{columnar_exec.rs, columnar_join.rs, query.rs, trust_compliance.rs}`)**:
  - Preserved exact integer SQL semantics on `INT +-*/ INT` and `CAST(int_col AS TEXT)` in `columnar_exec::build_vcol`, gated `columnar_join` to `.jcol` / `>= 256`-row analytical relations while rejecting CTE `set_op`/`distinct`, excluded nested subqueries from flat semi-join pushdown (`w359_nested_exists`), fixed debug `i8` wrapping in `waios_block64` test vectors, and aligned session-ID, Arrow PK ingest, `trust_compliance`, ANSI `NULL` join, and `59`-tool MCP assertions.

## [0.13.0] - 2026-11-27

### Added
- **Exact `128`-Bit Fixed-Point `DECIMAL(38, s)` / `NUMERIC(38, s)` Arithmetic (`crates/joule-db-server/src/{decimal128.rs, query.rs, columnar_join.rs}`, `amorphic_adapter/expressions.rs`)**:
  - Added `Decimal128` (`i128` mantissa, `scale 0..=18`, up to 38 decimal digits, `#![deny(clippy::unwrap_used)]`) and `Decimal128Accumulator`, eliminating IEEE-754 binary floating-point penny drift across SQL `+`, `-`, `*`, `/` (banker's half-even rounding), `=`, `<`, `<=`, `>`, `>=`, `CAST(... AS NUMERIC(p, s))` (half-away-from-zero), `SUM()`, `DECIMAL_ADD/SUB/MUL/DIV/ROUND/CMP`, and multi-table/bulk-loaded `NUMERIC`/`DECIMAL` tables (`0.10 + 0.20 = 0.30` exact; `1,000 × 0.01` `SUM` = `10` exact; `30+`-digit institutional wires exact), plus Neumaier compensated `f64` summation in `columnar_join` & `columnar_aggregate`.
- **SQL-Native Double-Entry Core Banking Ledger (`crates/joule-db-server/src/financial_ledger.rs`)**:
  - Unified `TigerBeetle`-style accounting invariants directly inside SQL without split-brain dual databases: `128-bit` minor-unit `LedgerAccount` & `LedgerTransferRecord`, structural conservation ($\sum \text{debits} \equiv \sum \text{credits}$) across 2-account `LEDGER TRANSFER` and $N$-leg `LEDGER MULTILEG` split settlements, hard `NO_OVERDRAFT` (`debits_must_not_exceed_credits` / `credits_must_not_exceed_debits`) balance floors, two-phase `LEDGER HOLD` -> full/partial `LEDGER POST HOLD` / `LEDGER VOID HOLD` card authorization flows, mandatory `idempotency_key` exact replay vs payload-mismatch conflict (`SQLSTATE 23505` / HTTP `409`), SHA-256 hash-chained `financial_ledger.jwal` crash recovery (`LEDGER VERIFY`), live relational SQL `JOIN`s via virtual tables (`jouledb_ledger_accounts`, `jouledb_ledger_transfers`, `jouledb_ledger_legs`), Axum `/api/v1/ledger/{accounts,transfers,holds,holds/post,holds/void,verify}`, and 6 MCP `ledger.*` tools.
- **Session-Isolated Copy-on-Write SQL Branching, 3-Way Conflict-Checked `MERGE`, Agent Blast-Radius Guardrails & Durable Workflows (`crates/joule-db-server/src/sql_branch_sandbox.rs`)**:
  - Added real row/table-isolated `CREATE BRANCH [name] [FROM parent] [ENERGY BUDGET <uj>] [GUARDRAILS STRICT]`, `USE BRANCH` / `SET jouledb.branch`, `SHOW BRANCHES`, `DIFF BRANCH`, 3-way write-write conflict detection (`SQLSTATE 40001`) on `MERGE BRANCH`, and `DROP BRANCH`.
  - Enforced per-branch and per-session autonomous agent blast-radius guardrails (`SET jouledb.agent_guardrails = 'strict'`): blocks destructive DDL (`DROP TABLE`, `TRUNCATE`, `ALTER TABLE ... DROP` -> `SQLSTATE 42501`), blocks unbounded `DELETE`/`UPDATE` without `WHERE` (`SQLSTATE 42501`), and enforces microjoule (`µJ`) session/branch budgets (`SQLSTATE 53200`).
  - Added DBOS-style crash-proof durable workflow step checkpointing (`WORKFLOW CHECKPOINT`, `WORKFLOW RESUME`, virtual table `jouledb_workflow_steps`, `workflow_steps.jwal`, and MCP `db.workflow_checkpoint` / `db.workflow_resume`, expanding the MCP server to **`58` tools**).
- **Pessimistic Row Locking (`SELECT ... FOR UPDATE [NOWAIT | SKIP LOCKED]`), `JouleOS` Receipt Bridge & `SIMD-LUT4` `.jsyn` Kernel**:
  - Added `RowLockManager` (`SELECT ... FOR UPDATE` / `FOR SHARE` with `NOWAIT` `SQLSTATE 55P03` and `SKIP LOCKED` concurrent worker queue claiming), `jouleos_ingest.rs` (`4,352`-bit Kulisch `ExactF64Superaccumulator` + SHA-256 Merkle bridge in `joule-db-ledger`), and `forward_simd_lut4_q24` nibble kernel in `joule-db-weights`.

### Fixed
- **100% Green Real-Driver `pgwire` Compatibility (`SQLAlchemy 2 + psycopg3` + Binary `DataRow` OID Width Parity) & `Engine` CI (`pgwire.rs`, `query.rs`, `amorphic/src/lib.rs`)**:
  - Enabled parameterized `type_info_lookup` (`psycopg3` `TypeInfo.fetch(conn, "hstore")` with `$1`) and `pg_catalog` / `information_schema` resolution inside `optimized_joins` (`SQLAlchemy 2` `pg_type JOIN pg_range` connect introspection), aligned `Portal.described_oids` between `RowDescription` and binary `DataRow` encoding (`psycopg3` `cursor(binary=True)` `10 / 10` green), and added `catch_up_holograms` before LSH comparison in `bulk_ingest_tests::delete_batch_equals_sequential`.

## [0.12.0] - 2026-11-26

### Added
- **Binary `JWU1` Delta `UPDATE` WAL Codec, Deferred-Hologram `UPDATE`s & 64-Shard `table_id_shards` (`crates/joule-db-amorphic/src/{wal_codec.rs, lib.rs, durable.rs}`, `crates/joule-db-server/src/mvcc_adapter.rs`)**:
  - Added binary `JWU1` `UpdateFields` WAL encoding (`roundtrip_jwu1_update` test), `update_fields_deferred_holo` + unchanged-column index/columnar skip, `SegmentedHologram::remove_one`, zero-allocation `for_each_record_equal`, and 64-shard `table_id_shards` in `MvccTableStorage` with changed-column-only `AtomicBatchOp::Update` deltas.
  - Measured head-to-head on `100`-warehouse (`100,000`-account) TCP `pgwire` OLTP over **physical NVMe `fdatasync`** (`--work-dir /var/tmp` rather than RAM `tmpfs`, `bench/results/oltp_pgwire_100wh.json`, `idle_watts = 10.68 W`, `invariant_verified = true` & `0` errors across all 5 runs): JouleDB `0.12.0` **beats `PostgreSQL 16.16` `1.63×` at `c=1` Simple (`1,950.53` vs `1,199.69 TPS`, `13.39` vs `45.79 mJ/txn`)** and scales to **`5,154.08 TPS` Simple (`0.718 ms` `p50`, `10.34 mJ/txn`) / `5,159.73 TPS` Extended Pipeline (`0.773 ms` `p50`, `11.71 mJ/txn`) at `c=16`**, narrowing the `c=16` gap vs `PostgreSQL 16.16` (`10,419.39` Simple / `9,635.60` Pipelined) to **`2.02×` / `1.87×`**.

### Fixed
- **`46.1×` Reduction in Out-of-Core `.jcol` Post-Load RSS (`34.81 GB -> 755.3 MB` at `SF=10.0`) & `2.20×` Faster `SF=10.0` `TPC-H` (`amorphic_adapter/{crud.rs, mod.rs}`, `disk_col_segments.rs`, `file_formats.rs`, `columnar_join.rs`)**:
  - Separated `.jcol` segment record IDs (`JCOL_ID_TAG = 1 << 62`) from `AmorphicStore.next_id` so multi-million-row `.jcol` loads never inflate `next_id` or zero-fill dense `1,280`-byte VSA hologram slots on subsequent DDL writes, paired with owned-batch Parquet streaming (`for_each_parquet_batch_group_owned`, `append_column_batch_owned`) and isolated per-scale `--work-dir` paths in `bench/tpch_official.py`.
  - Added ZoneMap `DenseKeyBits::overlaps_range` zero-I/O segment pruning, two-stage sparse dictionary loading (`load_filtered_segment`), parallel batched `materialize_node`, `<= 20,000,000`-row `propagate_initial_key_sets` (`Q10` `orders` quarter semi-join pushdown), outer-domain pushdown for `NeqExists`/`CorrScalar` (`Q17`, `Q20`, `Q21`), 64-shard radix-partitioned single-`i64` `GROUP BY` (`Q18`), and branch-row-indexed `u32` pre-aggregation.
  - Verified **`22 / 22` (`100%`, `0` errors) Official `TPC-H` exact across all three scales** (`bench/results/tpch_official_{sf01,sf1,sf10}.json`, NVMe + Linux `rapl-package` `reported_j`):
    - **`SF = 0.1` (`866,602` rows):** `253.0 ms` total vs DuckDB `88.2 ms` (**`1.992×` geomean**, **`5 / 22` faster than DuckDB `2.0.1`**: `Q22: 0.53×`, `Q11: 0.53×`, `Q02: 0.80×`, `Q20: 0.83×`, `Q16: 0.97×`), `780.8 MB` post-load RSS (`939.3 MB` `VmHWM`).
    - **`SF = 1.0` (`8,661,245` rows):** **`201.4 MB` post-load RSS (`28.6×` down from `5,769.3 MB`; `1,561.8 MB` `VmHWM`)**, **`1,779.0 ms` (`1.78 s`) total (`1.57×` faster)**, **`4.331×` geomean (`median 3.604×`, down from `6.180×`)**.
    - **`SF = 10.0` (`86,586,082` rows, `60.0M` `lineitem`):** **`755.3 MB` post-load RSS (`46.1×` down from `34,809.5 MB`; `15,586.4 MB` `VmHWM`)**, **`15,464.6 ms` (`15.46 s`) total (`2.20×` faster than `34.06 s`)**, **`8.494×` geomean (`median 8.377×`, cut nearly `2×` from `16.026×`; `Q17: 88.6 ms` [`17.2×` faster], `Q20: 99.7 ms` [`6.59×` faster], `Q18: 1,799.8 ms` [`6.29×` faster], `Q21: 1,187.4 ms` [`2.32×` faster], `Q10: 5,230.7 ms` [`1.74×` faster])**.
- **`pgwire` Client Protocol Compatibility (`pgwire.rs`, `energy_executor.rs`, `columnar_exec.rs`, `columnar_join.rs`)**:
  - Coerced untyped text-formatted numeric bind parameters (`"1" -> 1`) across `try_direct_indexed_eq_query`, `columnar_exec::prepare_preds`, and `columnar_join::cmp_bvals` (fixing `node-postgres` `tests/clients/node_pg.js` `select_params` — 5/5 passing), added PostgreSQL v12+ `GSSENCRequest` (`80877104`) `'N'` negotiation for default `libpq`/`psycopg` connections, forwarded `EnergyAwareExecutor::describe_query` & `bulk_insert_arrow_owned_with_hint`, and replaced hardcoded `"0.6.0"` strings in `jouledb self-check` with `env!("CARGO_PKG_VERSION")`.

## [0.11.0] - 2026-11-26

### Added
- **Vectorized Out-of-Core Columnar Multi-Table Join & Subquery Engine (`crates/joule-db-server/src/columnar_join.rs`) — `22 / 22` Official `TPC-H` Exact at `SF = 0.1`, `SF = 1.0`, AND `SF = 10.0` (`86.6M` Rows)**:
  - Added `columnar_join.rs` (`#![deny(clippy::unwrap_used)]`, `5,493` LOC) streaming `262,144`-row `.jcol` `DiskColumnTable` segments directly against compact post-order Snowflake dimension hash branches (`CompactCol`, `DenseUnique1`, `HashUnique1`, `HashMulti1`, `Hash2`, `O(1)` `DictBitset` text/date filters, `FROM (SELECT ...)` derived-table flattening, `WITH` CTEs, parallel `LEFT OUTER JOIN` grouped count for `Q13`, `DenseKeyBits` `IN`/`EXISTS`/`NOT EXISTS` + `NeqSummary` for `Q21`, outer-domain-pushed correlated scalar `MIN`/`MAX`/`SUM`/`AVG` for `Q02, Q17, Q20`, `OR`-branch per-table bounding filters for `Q07, Q19`, and `PostAggExpr` ratio/arithmetic projections).
  - Verified **`22 / 22` (`100%`, `0` errors) Official `TPC-H` (`tpch_queries()`) exact across three scale factors** (`bench/results/tpch_official_{sf01,sf1,sf10}.json`, Linux `rapl-package` `reported_j`):
    - **`SF = 0.1` (`866,602` rows):** `318.0 ms` total (`155.9×` faster than `v0.10.0`'s `49.59 s`), narrowing geomean vs DuckDB `2.0.1` from **`125.50× -> 2.224×`** (`median 1.920×`) and **beating DuckDB `2.0.1` on `3 / 22` queries** (`Q22: 1.31 ms` vs `4.27 ms` [`0.31×`], `Q11: 2.46 ms` vs `4.40 ms` [`0.56×`], `Q20: 3.54 ms` vs `5.66 ms` [`0.63×`]).
    - **`SF = 1.0` (`8,661,245` rows, `6.0M` `lineitem`):** Upgraded from `13 / 22` (`8` `25 s` timeouts + `1` `36 GB` RSS abort, `112.75×` geomean) to **`22 / 22` exact in `2,787.1 ms` total (`6.180×` geomean, `median 6.386×`, `6,303.2 MB` peak `VmHWM`)**, **beating DuckDB `2.0.1` on `Q22` (`12.59 ms` vs `17.12 ms`, `0.74×`)** (`Q06: 9.80 ms`, `Q11: 14.99 ms`, `Q17: 15.00 ms`, `Q14: 16.81 ms`, `Q05: 61.83 ms`).
    - **`SF = 10.0` (`86,586,082` rows, `59,986,052` `lineitem` out-of-core `.jcol`):** **`22 / 22` exact in `34,060.0 ms` total (`16.026×` geomean, `median 12.445×`, `18 / 22` queries `< 1.0 s`)**, candidly reporting DuckDB `2.0.1`'s `16.03×` advantage and widest tail queries (`Q10: 9.11 s`, `Q18: 11.33 s`).

### Fixed
- **64-Shard `MvccStore` + Deferred-Hologram Binary `JWB1` `pgwire` OLTP (`5,797.0 TPS` at `c=4`, `5,038.4 TPS` at `c=16`, `invariant_verified = true`)**:
  - Replaced single global `RwLock` in `MvccStore` (`crates/joule-db-core/src/concurrency/mvcc.rs`) with 64 FNV-1a shards and `VersionChain::find_visible_at(read_ts)` (eliminating `get_active_txns()` Mutex lock + `HashSet` clone on every read/scan), sharded `MvccTableStorage` sessions across 64 buckets with `AtomicU64` session IDs (`mvcc_adapter.rs`), added `SchemaMetaCache` + `try_direct_indexed_eq_query` (`O(1)` point reads and `O(k)` register-accumulated indexed `COUNT`/`SUM` without `TableColumns` allocation), added `O(1)` `deferred_set`, lazy `BTreeSet` promotion on `ValueIds::remove_one`, `O(log n)` monotonic `Column::find_pos`, and binary `JWB1` `batch_ingest_fields` / `ingest_fields_deferred_sync` (eliminating `catch_up_holograms(usize::MAX)` + JSON round-trips on `INSERT` commits).
  - Measured head-to-head on `100`-warehouse (`100,000`-account) TCP `pgwire` OLTP (`bench/results/oltp_pgwire_100wh.json`, `idle_watts = 10.96 W`): Simple Query reaches **`4,094.66 TPS` (`c=1`, `p50 0.134 ms`) -> `5,797.01 TPS` (`c=4`) -> `5,376.65 TPS` (`c=8`) -> `5,038.38 TPS` (`c=16`, `15.93 mJ/txn`, `48,473` writes)** and Extended Pipeline reaches **`4,020.27 -> 5,205.58 -> 5,186.25 -> 4,990.75 TPS`**, narrowing the `c=16` gap vs `PostgreSQL 16.16` (`14,676.40` Simple / `13,029.46` Pipelined) to **`2.91×` / `2.61×`**.
- **Streaming `.jcol` Pending-Batch Spill Accounting & GNU/Linux Arena Trim (`disk_col_segments.rs`, `file_formats.rs`, `crud.rs`)**:
  - Counted queued `pending_batches` rows per table in `insert_column_batch` so multi-chunk Parquet loads always transition cleanly to `.jcol` disk segments above `JOULEDB_COL_SPILL_ROWS`, and added `trim_unused_heap()` (`malloc_trim(0)` on GNU/Linux).

### Changed
- **Achieved `41 / 41` (`100%` of Workspace) `deny(clippy::unwrap_used)` Enforcement (`0` `#[allow(clippy::unwrap_used)]` Bypasses)**:
  - Eliminated over 1,000 non-test `.unwrap()` sites across the final 12 workspace crates (`joule-db-amorphic`, `joule-db-features`, `joule-db-hdc`, `joule-db-browser` incl. `wasm32`, `joule-db-domains`, `joule-db-gpu`, `joule-runtime`, `sigql`, `inv-auth`, `inv-crypto`, `inv-energy`, `inv-telemetry`), promoted `.github/workflows/engine.yml` to a workspace-wide `cargo clippy --workspace --lib --bins --locked` blocking gate, wired `@openie/jouledb` npm tarball packaging/publishing in `.github/workflows/release.yml`, and bumped all manifests & release artifacts to `0.11.0`.

## [0.10.0] - 2026-11-26

### Added
- **Official 22-Query `TPC-H` Multi-Table Relational Benchmark Suite (`bench/tpch_official.py`, `tpch_official_{sf01,sf1}.json`)**:
  - Added reproducible multi-table relational evaluation across all **22 official `TPC-H` SQL queries (`tpch_queries()`, `Q01..Q22`)** over 8 Parquet-loaded relational tables (`lineitem`, `orders`, `customer`, `part`, `partsupp`, `supplier`, `nation`, `region`) over real TCP `pgwire` (`psycopg 3.3.9`, Linux `rapl-package` `reported_j`) against `DuckDB 2.0.1` with client `SIGALRM` timeout and `/proc/<pid>/status` `VmRSS` safety guards.
  - Verified **`22 / 22` (`100%`, `0` errors) exact query matches at `SF = 0.1` (`866,602` rows)** (`Q06: 5.00 ms`, `Q15: 5.65 ms`, `Q01: 62.57 ms`, `Q10: 144.56 ms`, `Q14: 219.56 ms`, `Q13: 245.66 ms`, `Q22: 258.24 ms`; `3 / 22` faster than DuckDB at `SF = 0.01`), and **`13 / 22` exact matches at `SF = 1.0` (`8,661,245` rows, `9.816 s` Parquet load)** (`Q15: 70.83 ms`, `Q01: 84.18 ms`, `Q06: 336.84 ms`, `Q03: 2,871.6 ms`, `Q10: 3,496.5 ms`), candidly documenting `8 / 22` `25 s` wall timeouts (`Q04, Q07–Q09, Q12, Q17, Q18, Q21`) and `1 / 22` `36 GB` RSS guard trip (`Q05`) on unvectorized `Vec<serde_json::Value>` multi-table hash joins.
- **Decorrelated Subqueries, Semi/Anti-Joins & SQL Dialect Support (`query.rs`, `sql.rs`, `functions.rs`, `columnar_exec.rs`)**:
  - Added hash semi/anti-joins for correlated `EXISTS` / `NOT EXISTS` (`Q04, Q21, Q22`), correlated scalar `GROUP BY` subquery decorrelation (`Q02, Q17, Q20`), uncorrelated `IN (SELECT ...)` `HashSet` filters (`Q16, Q18, Q20`), `OR`-conjunct join predicate factoring (`Q19`), `col_cache` vectorized projection/filter pushdown for multi-table joins, derived-table column alias lists (`AS c_orders (c_custkey, c_count)`, `Q13`), `DOUBLE PRECISION` & `CHARACTER VARYING` DDL/`CAST`, constant-expression folding in `columnar_exec::literal()` (`Q06`), and ISO date string <-> epoch `Timestamp` coercion in comparisons (`Q14`) and `EXTRACT` / `DATE_TRUNC` (`Q07–Q09`).
- **Binary `JMVR\x01` `MvccRecord` Codec & Cached `pgwire` Extended Pipeline (`mvcc_adapter.rs`, `pgwire.rs`, `query.rs`)**:
  - Added compact tag-encoded binary `JMVR\x01` `MvccRecord` serialization (with backward-compatible JSON fallback), cached `PreparedStatement.describe_cols` (`RowDescription` built once at `Parse`), direct `portal.params` parameter binding (bypassing SQL string reconstruction), and thread-local 256-entry LRU SQL AST caching — eliminating the `1.97×` Extended Pipeline penalty (`1,375.67 TPS` Pipelined vs `1,351.57 TPS` Simple at `c=16`, `1,439.95 TPS` Simple at `c=4`, `invariant_verified = true` across all 5 runs).

### Fixed
- **Full `99,997,497`-Row (`100M`) Official ClickBench Narrowed from `2.82×` to `2.11×` Geomean (`2.03×` `reported_j`, `10M` `VmHWM` Cut `2.63×`)**:
  - Upgraded `columnar_exec::execute_segmented` (`columnar_exec.rs`, `disk_col_segments.rs`) with `WordMask` (`Vec<u64>`) bitpacked selection vectors, filtered `MIN`/`MAX` fast paths (`Q7`: `10.71× -> 1.04×` vs DuckDB `2.0.1`), `90`-sum SIMD slice batching (`Q30`: `14.26× -> 2.85×`), dictionary `LIKE` bitsets (`Q20`: `6.75× -> 3.54×`), 256-partition parallel hash-reduce (`Q15`: `11.40× -> 4.82×`), and narrower filtered `CASE` grouping (`Q39`: `22.36× -> 14.84×`) — achieving **`2.114×` geomean latency & `2.034×` `reported_j` on all `99,997,497` rows** (`5,901.8 MB` post-load RSS) and cutting `10M`-row peak `VmHWM` from `9,060.4 MB -> 3,448.4 MB`.
- **Unqualified Equi-Join Right-Column Key Offset & `NULL` Hash-Join Semantics (`query.rs`)**:
  - Fixed unqualified right-side column resolution in `execute_join` (`idx % join_col_count` over deduplicated `join_rows`) and changed `compute_hash_key` to return `None` on `NULL` join keys per SQL three-valued logic.

### Changed
- **Expanded `deny(clippy::unwrap_used)` from `7 / 41` Crates to `29 / 41` Crates (`70.7%` of Workspace) — Including `joule-db-server` & `jouledb-cli`**:
  - Eliminated non-test `.unwrap()` calls across `joule-db-server` (`lib` + `bin`), `jouledb-cli`, `joule-db-amorphic`, `joule-db-hdc`, `joule-db-viz`, and 17 additional engine/transport crates, enforcing `#![cfg_attr(not(test), deny(clippy::unwrap_used))]` across **29 workspace crates** in `.github/workflows/engine.yml`, and bumped all workspace, stub (`crates.io`), `.mcpb`, wheel, SDK, and Helm/Terraform manifests to `0.10.0`.

## [0.9.0] - 2026-11-26

### Added
- **Claude Apps & MCP Apps (`SEP-1865` Interactive Studio UI, `jouledb mcp`, `.mcpb` Bundles & Antigravity Plugin)**:
  - Added `crates/joule-db-server/src/mcp_app_ui.rs` (`#![deny(clippy::unwrap_used)]`, `ui://jouledb/studio.html`, MIME `text/html;profile=mcp-app`, `30,179` bytes, strict zero-CDN `Content-Security-Policy`) implementing the MCP Apps `SEP-1865` (`2026-01-26`) bidirectional `postMessage` JSON-RPC bridge (`ui/initialize`, `ui/notifications/tool-result`, `tools/call`, `ui/update-model-context`, `ui/message`) with 5 interactive tabs (SQL Workbench, `reported_j`/`est_j` Energy Telemetry, Schema & `.jcol` Segments, `10,240`-Bit Amorphic Lens, CoW Branches & `.jsyn` Weights).
  - Added `jouledb mcp --data <dir>` `stdio` JSON-RPC `2025-06-18` subcommand and upgraded `POST /mcp` (`crates/joule-db-server/src/mcp_transport.rs`) to `50` MCP tools (`25` core `db.*` incl. `db.studio` + `25` extended) attaching `_meta.ui.resourceUri` + legacy `_meta["ui/resourceUri"]` and `structuredContent`.
  - Added Claude Desktop Desktop Extension manifest & packaging script (`deploy/claude-app/{manifest.json, build_mcpb.sh}`, `manifest_version: "0.3"`), Antigravity plugin (`deploy/antigravity-plugin/`), `docs/src/claude-apps.md`, and `https://jouledb.org/claude`.
- **5-Node Multi-Host Cross-Machine WAN/Tailscale TCP Raft Fault-Injection + Official Clojure `elle 0.2.8` (`bench/raft_multihost_wan.py`, `examples/raft_wan_node.rs`)**:
  - Executed 5 independent OS `raft_wan_node` processes across **2 distinct physical Linux machines** (`framework-128gb-0000` bare-metal + `openie-core-1` Cloud Debian 12 VM) over encrypted Tailscale WAN (`27.576 ms` median TCP RTT), injecting cross-WAN `2/5` minority leader isolation (`["n1", "n4"]` vs `["n2", "n3", "n5"]`), stale write & `ReadIndex` read rejection, majority leader failover (`n3` term `2`, `24` ops), real OS `SIGKILL` (`kill -9`) on `n5` (`16` ops during crash), and `raft.wal` crash recovery (`commit=68, applied=68, x=68, y=39` across all 5 nodes) -> verified **`{:valid? true}`** (`270` events, `:strict-serializable` `rw-register`) by official Clojure `elle/elle 0.2.8`.
- **Distributable SDK Release Artifacts (`v0.9.0` GitHub Release)**:
  - Built prebuilt `cp39-abi3` Python `.whl` wheels (`macosx_11_0_arm64` & `manylinux_2_34_x86_64`), TypeScript `openie-jouledb-0.9.0.tgz` (`node --test` verified), C99 `smoke_test.c`, and platform `.mcpb` bundles alongside server `.tar.gz` archives and `SHA256SUMS.txt`.

### Fixed
- **`1.72×` Faster Full `99,997,497`-Row (`100M`) Official ClickBench (`4.85× -> 2.816×` Geomean vs DuckDB `2.0.1`, `Q32` `7.24×` Faster, `VmHWM` `-5.26 GB`)**:
  - Added per-column `global_int_sum: i128` in `.jcol` segment metadata (`disk_col_segments.rs`) for zero-I/O unfiltered `SUM`/`AVG` (`Q2` `1.60 ms`, `Q3` `1.84 ms`, `Q29` `2.27 ms` on `100M` rows — `5.6×–6.9×` faster than DuckDB `2.0.1`), dynamic `AtomicI64` Top-K ZoneMap bound pruning (`Q19` `31.32 ms` vs `42.46 ms`), zero-`Arc<str>` `RawTextSlice` text distinct/grouping (`Q5, Q12, Q33, Q34`), 256-partition `CompactI2Row` in-place sort-reduce cutting `100M`-group `Q32` from **`17,419.5 ms -> 2,405.5 ms` (`7.24×` faster)** and lowering `100M` peak `VmHWM` from `20,740.5 MB -> 15,483.2 MB`, and single-metadata row-budgeted (`MAX_CHUNK_ROWS = 524,288`) Parquet `COPY` (`file_formats.rs`).
- **Static `pgwire` Extended Query `Describe` Type Inference & `DROP TABLE` Index Cleanup (`3.01×` Faster Pipelined OLTP)**:
  - Added `describe_sql_static` (`query.rs`) to infer parameter & result column PostgreSQL OIDs (`INT8`, `FLOAT8`, `BOOL`, `TEXT`) without executing a dry-run query during `Parse`/`Describe`, eliminated per-query thread spawning for normal query timeouts (`pgwire.rs`), and cleaned up `__indexes__` metadata records in `AmorphicTableStorage::drop_table` (`schema.rs`) — lifting `psycopg` Extended Pipeline throughput from `593.7 TPS` (`v0.8.0`) to **`1,790.0 TPS` (`c=8`) / `1,703.3 TPS` (`c=16`)** (`10,973` committed writes, `invariant_verified = true`).

## [0.8.0] - 2026-11-25

### Added
- **Out-of-Core `.jcol` Columnar Segment Engine (`disk_col_segments.rs` + `execute_segmented`) & Full `99,997,497`-Row (`14.78 GB`) Official ClickBench Completion**:
  - Added `crates/joule-db-server/src/disk_col_segments.rs` (`#![deny(clippy::unwrap_used)]`, `MAX_SEGMENT_ROWS = 262_144`, configurable via `JOULEDB_COL_SPILL_ROWS`): drains large columnar tables to immutable `.jcol` segment files with parallel Rayon column encoding, automatic `I8/I16/I32/I64` bit-width narrowing, bitpacked null masks, compact `u8/u16/u32` local string dictionaries, CRC-32 per-slice and `JCL1` footer checksums, Unix `pread` (`read_exact_at`) column-subset loading, startup manifest recovery, and `DROP TABLE` cleanup.
  - Added `columnar_exec::execute_segmented` (`crates/joule-db-server/src/columnar_exec.rs`): zero-I/O metadata aggregates (`COUNT(*)`, `MIN`, `MAX`), segment-level ZoneMap pruning, two-phase `WHERE`-first column loading (skipping projection/grouping column reads on 0-match segments), two-phase Top-K `SELECT * ... ORDER BY ... LIMIT k` (`Q23`), 64-shard exact `COUNT(DISTINCT int/text)` (`Q4`, `Q5`), constant & functionally-determined `col ± k` group-key elimination (`Q34`, `Q35`), inline small-set `DistinctSet` (`Inline4`), and 4-pass Radix-64 parallel sharded `GROUP BY` with per-shard `HAVING`/Top-K reduction.
  - Completed the **entire official `99,997,497`-row (`14.78 GB`, `105`-column) ClickBench `hits.parquet` dataset** (`bench/results/clickbench_official_100m.json`, `is_full_100m_dataset: true`, Linux `rapl-package` `reported_j`): **`6,199.0 MB` (`6.05 GiB`) post-load RSS** (`20,740.5 MB` kernel `VmHWM` peak), **`43 / 43` queries succeeded (`0` errors, `0` restarts), `43 / 43` SQL-valid matches (`31` byte-exact)**, **`4.854×` geomean latency** (`median 4.570×`) and **`4.246×` geomean `reported_j`** vs DuckDB `2.0.1` (`5 / 43` JouleDB wins at `100M` rows; `10 / 43` wins & `3.213×` geomean at `10M` rows vs `9.74×` in `v0.7.0`).
- **Head-to-Head `100`-Warehouse TCP `pgwire` OLTP Benchmark vs. `PostgreSQL 16.16` & `SQLite 3.53.1` (`bench/oltp_pgwire.py`)**:
  - Extended `bench/oltp_pgwire.py` (`bench/results/oltp_pgwire_100wh.json`) to benchmark `JouleDB 0.8.0` alongside real `PostgreSQL 16.16` (`fsync=on`, `synchronous_commit=on`) across both PostgreSQL v3 Simple Query (4-RTT) and `psycopg` Extended Query Pipeline (1-RTT) modes plus embedded `SQLite 3.53.1` WAL (`invariant_verified: true` across all 5 runs).

### Fixed
- **`O(1)` PK/UNIQUE Index Lookups, Multi-Row `INSERT` Batching & `pgwire` Extended Query Pipeline Recovery**:
  - Upgraded `check_primary_key_unique` / `check_unique_constraints` (`crates/joule-db-server/src/amorphic_adapter/{indexes.rs, constraints.rs, crud.rs}`) to `O(1)` secondary-index `records_equal` verification plus `check_batch_unique_constraints` for multi-row PK/UNIQUE `INSERT`s, lifting JouleDB `100`-warehouse `pgwire` Simple Query throughput **2.54×** (`1,318.6 -> 3,353.5 TPS` at `c=16`; `1,659.5 TPS` at `c=1`).
  - Added `bind_select_query_params` (`crates/joule-db-server/src/query.rs`) for parameterized `$1..$N` `SELECT` fast-path execution and PostgreSQL v3 §55.2.3 `discard_until_sync` Extended Query pipeline error recovery (`crates/joule-db-server/src/pgwire.rs`).

### Changed
- **Promoted `joule-db-query` to 7th `deny(clippy::unwrap_used)` Crate & Bumped Workspace to `0.8.0`**:
  - Eliminated all 15 non-test `.unwrap()` calls across `crates/joule-db-query/src/{like.rs, sql.rs, functions.rs, geo.rs, graph.rs, wcoj.rs, storage_executor.rs}`, enforced `#![cfg_attr(not(test), deny(clippy::unwrap_used))]` across **7 crates** in `.github/workflows/engine.yml`, and bumped all workspace, stub, Python, SDK, and Helm/Terraform manifests to `0.8.0`.

## [0.7.0] - 2026-11-25

### Added
- **Vectorized Columnar Engine (`columnar_exec.rs`), Width-Adaptive Integer Columns (`I8..I64`) & Streaming Parquet `COPY` (`43 / 43` Official ClickBench up to `10,000,000` Rows)**:
  - Added width-adaptive integer columns (`Ints::{I8, I16, I32, I64}`) and narrow `ColData::{Int8, Int16, Int32, Int}` batches (`crates/joule-db-amorphic/src/column_batch.rs`), plus bounded 4-row-group streaming Parquet `COPY ... FROM` (`crates/joule-db-server/src/{file_formats.rs, pgwire.rs}`).
  - Extended `crates/joule-db-server/src/columnar_exec.rs` with precompiled `PrepPred` filters (`O(1)` dictionary equality, parallel `LIKE`/`IN` dictionary bitsets, Howard Hinnant ISO date/timestamp coercion with zone-map pruning), zero-materialization `VCol` virtual columns (`EXTRACT`, `DATE_TRUNC`, `STRLEN`, `REGEXP_REPLACE` dictionary remapping, lazy `CASE WHEN`), algebraic `SUM(col ± k)` scan sharing (single pass for `Q29`'s 90 sums), `MIN`/`MAX` over dictionary text, `HAVING`, inline `HKey::{K1, K2, K3}` group keys, and morsel-parallel Top-K `ORDER BY ... LIMIT` scans.
  - Published reproducible Official ClickBench runs (`bench/results/clickbench_official_{100k,1m,10m}.json`) against DuckDB `2.0.1` over real TCP `pgwire` (`rapl-package` `reported_j`): **`43 / 43` queries succeeded (`0` timeouts, `100%` SQL-valid match)** at `100k` rows (`1.30×` geomean latency, median `0.80×`, beating DuckDB on `25 / 43` queries), `1M` rows (`9.34×` geomean), and **`10,000,000` official rows** (`42.0 GB` RSS, `9.74×` geomean latency, `7.50×` geomean `reported_j`).
- **True Raft §6.4 `ReadIndex` Quorum Linearizable Reads (`RaftNode::read_index`, `RaftNode::read_linearizable`)**:
  - Implemented quorum empty-`AppendEntries` leadership confirmation + `last_applied >= read_index` wait in `crates/joule-db-server/src/raft.rs`, delivering linearizable reads across partitions with **zero Raft log / WAL writes**, verified `{:valid? true}` (`rw-register`, `:strict-serializable`) by official Clojure `elle/elle 0.2.8`.
- **Concurrent Multi-Client `pgwire` OLTP Benchmark (`bench/oltp_pgwire.py`)**:
  - Added reproducible `100`-warehouse (`100,000`-account) multi-client OLTP benchmark across `1, 4, 8, 16` concurrent `psycopg` TCP workers (`50%` point read, `30%` durable `BEGIN..UPDATE..INSERT..COMMIT`, `20%` warehouse aggregate) with post-run balance & ledger conservation checks (`invariant_verified = true`, `605 -> 1,319 TPS` over TCP `pgwire` with `fsync` vs embedded SQLite WAL).

### Fixed
- **MVCC / `AmorphicTableStorage` Transactional Consistency, Contention & OLTP Indexing (`mvcc_adapter.rs`, `amorphic_adapter/*`)**:
  - Fixed `MvccTableStorage` so tables populated after startup via `COPY ... FROM` or auto-commit `INSERT` are lazily seeded on demand via `O(1)` secondary-index equality lookup (`resolve_session_candidates`) or table scan, fixing silent `UPDATE 0` misses inside `BEGIN ... COMMIT` sessions.
  - Replaced interleaved multi-lock `begin_batch` / `batch_*` / `commit_batch` calls with `MvccTransaction::commit_with_hook` + `AmorphicTableStorage::apply_atomic_batch_deferred`, validating SI/SSI conflicts before WAL writes, eliminating concurrent `"Batch already active"` commit errors, writing newly assigned `amorphic_id`s back onto inserted MVCC records (preventing duplicate rows on subsequent `UPDATE`), and coalescing `GroupSync` `fsync` outside `commit_lock`.
  - Removed blocking `catch_up_holograms(usize::MAX)` from `AmorphicStore::update_fields` for deferred-hologram rows and added selective `O(k)` equality column slices (`try_selective_table_columns`) when `col_cache` is invalidated by concurrent OLTP writes.
- **SQL Function & Grouping Compatibility (`functions.rs`, `query.rs`)**:
  - Separated byte-level `STRLEN` / `OCTET_LENGTH` from codepoint `LENGTH` / `CHAR_LENGTH`, added POSIX `\1..\9` regex backreference normalization in `REGEXP_REPLACE`, and supported `GROUP BY <select_alias>`.

### Changed
- **Expanded `deny(clippy::unwrap_used)` to 6 Core Crates & Workspace Hygiene**:
  - Enforced `#![cfg_attr(not(test), deny(clippy::unwrap_used))]` across **6 crates** (`joule-db-core`, `joule-db-branch`, `joule-db-ledger`, `joule-db-local`, `joule-db-energy`, `joule-db-weights`) with a blocking `cargo clippy` CI step in `.github/workflows/engine.yml`.
  - Aligned all 41 workspace crates, `stubs/jouledb`, Python bindings, SDKs (`sdks/*`), and Helm/Terraform manifests to `0.7.0` (`SELECT version()` now reports `JouleDB 0.7.0`), and pruned 173 unused `[workspace.dependencies]`.

## [0.6.0] - 2026-11-25

### Changed
- **Reverted Release Track to `0.6.0` (Pre-1.0 Research Preview)**:
  - Yanked `jouledb 1.0.0` on `crates.io` and marked GitHub Release `v1.0.0` as retracted so `cargo binstall jouledb` resolves `0.6.0` directly without requiring `--version`.
  - Updated workspace, CLI, Python bindings (`python/`), all 6 client SDK manifests (`sdks/*`), and Helm/Terraform charts (`deploy/*`) to `0.6.0`.
- **Removed Unrelated `joule-web` Crate (`1.11M` LOC / `34,928` Tests) & Deleted Synthetic Exam Routes**:
  - Removed `crates/joule-web` and dead CLI command `crates/bin/jouledb-cli/src/commands/web.rs` from `openIE-dev/jouledb`, reducing the repository to **`41` workspace crates (`724,986` lines of Rust, `12,265` tests)** focused on the database, storage, query, consensus, energy, and weights runtime.
  - Deleted `crates/joule-db-server/src/frontier_exam_routes.rs` and `wave26_frontier_exam_hle_arc_math_sql_swe_calibration_e2e.rs`; renamed CLI `powerhouse` to `jouledb self-check` and pruned MCP tools from `54` to `49` non-benchmark-branded tools.
- **Truth-in-Labeling Across Security Controls, Bridges, Smoke Checks & Energy Evidence**:
  - Updated `trust_compliance.rs`, `universal_bridge.rs`, `distributed_jepsen_consensus.rs`, `certification_routes.rs`, `analytic_benchmarks.rs`, and `frontier_eval.rs` to distinguish in-process smoke/control self-tests from external benchmarks or compliance certifications, and tagged analytical energy estimates explicitly with `energy_evidence: "est_j"` vs hardware `reported_j`.
  - Aligned documentation on exact workspace lints: `#![deny(unsafe_code)]` in `joule-db-core` (with 1 `#![allow(unsafe_code)]` module `storage/mmap_extent.rs`) and `deny(clippy::unwrap_used)` active in 3 crates (`joule-db-core`, `joule-db-branch`, `joule-db-ledger`).

### Fixed
- **Consensus Quorum Commit Fix (`RaftNode::propose_and_commit`) & 5-Node Real TCP Fault-Injection Verified by Official Clojure `elle 0.2.8`**:
  - Fixed `crates/joule-db-server/src/raft.rs` where `RaftNode::propose()` previously returned `Ok(index)` upon local leader WAL append before quorum replication; added `propose_and_commit(&self, command, timeout)` awaiting majority commit (`commit_index >= index`) and Raft §5.4.2 current-term verification, and removed vacuous `|| true` assertions in `tests/raft_chaos.rs`.
  - Added `tests/raft_tcp_elle_fault_injection.rs` + `bench/elle_check.clj` exercising 5 real `RaftNode<KvStateMachine, TcpRaftTransport>` servers with durable `raft.wal` across a 20-link socket-severing `TcpPartitionMatrix` (`2/5` minority leader isolation & write rejection, `3/5` majority new-term election, node stop + disk WAL restart, 5-node state convergence), verified `{:valid? true}` (`:strict-serializable` `rw-register`) by Kyle Kingsbury's official Clojure `elle/elle 0.2.8`.
- **Portable Linux `glibc <= 2.36` Release Binaries (Fixes Ubuntu 24.04 LTS `GLIBC_2.43 not found`)**:
  - Switched `x86_64-unknown-linux-gnu` release builds from `ubuntu-latest` (`glibc 2.43`) to a `Debian 12` / `ubuntu-22.04` (`glibc 2.35–2.36`) baseline with an automated `objdump -T` CI gate rejecting `GLIBC >= 2.40`, verified end-to-end inside `ubuntu:24.04`.
- **Official ClickBench Evaluation Harness (`bench/clickbench_official.py`)**:
  - Added reproducible end-to-end evaluation against `DuckDB` over real `pgwire` (`psycopg`) using the official `14.78 GB` (`99,997,497`-row) ClickBench `hits.parquet` dataset and all 43 official `ClickBench` SQL queries (`Q0..Q42`), recording hardware RAPL `reported_j`, wins, losses, and SQL compatibility errors without filtering.

---

## [1.0.0] - 2026-11-25 — **[RETRACTED — Superseded by `v0.6.0` Pre-1.0 Research Preview]**

> *Retracted on 2026-11-25 (`crates.io` `1.0.0` yanked). See Errata Notice at top of file.*

### Added (Historical Record)
- Initial `live_migration` (`jouledb migrate`), `v1_stability_and_pitr` (`JWAL-v1` / `JSYN-v1` / `JWP-v1` / `ASNP-v1` / `TCAP-v1` TLV + CRC-32 framing & LSN PITR), `sql2023_powerhouse` (window frames, `CUBE`, recursive CTEs, runtime `AQE`), `distributed_jepsen_consensus` (in-process Raft state-machine simulator, relabeled in `0.6.0`), client SDK source trees (`sdks/*`), and `deploy/helm` + `deploy/terraform`.

## [0.5.0] - 2026-11-25 *(Historical — see Errata Notice above regarding smoke-test vs certification framing)*

### Added
- **`MorphRule` DSL, Bounded-Memory Multi-GB Streaming, Protocol Bridges & Security Control Self-Tests (`morph_rules`, `custom_lens`, `streaming_continuum`, `universal_bridge`, `trust_compliance`)**:
  - SI/financial unit algebra (`25°C == 77°F == 298.15 K`, `532 nm == 0.532 µm`, `\$14,250.00 == 1,425,000 cents`), Kanerva VSA `H ⊗ ROLE⁻¹` role unbinding, and `ParetoScatter` + `SankeyLineage` lenses (`8` fluid lenses total).
  - Bounded-memory $O(E)$ `StreamingAmorphicContinuum` (`1.35 GB` / `47.5M` semantic atoms streamed in `57.64 MB` OS Peak RSS).
  - Protocol adapter routes (`universal_bridge.rs`) and technical security control self-tests (`trust_compliance.rs` — SHA-256 hash-chained audit logs, WORM retention locks, per-subject crypto-shredding, Luhn PAN redaction, $\varepsilon$-DP noise, CycloneDX v1.6 SBOM, ML-KEM-768/ML-DSA-65 receipts, Bell-LaPadula MLS lattice).

## [0.4.0] - 2026-11-25 *(Historical — see Errata Notice above)*

### Added
- `joule-db-amorphic::unsilo`, `joule-db-viz::amorphic_display`, `OrthogonalTitansSynapse`, `TemporalRelationalGraph`, `ML-KEM-768 + ML-DSA-65` Merkle execution envelopes, `LiveCxlTieredPool`, and 12-script multilingual text normalization checks.

## [0.3.1] - 2026-11-24 *(Historical — see Errata Notice above)*

### Added
- In-process operator micro-benchmarks (`analytic_benchmarks.rs`: `ALP+FSST` codec, Leapfrog `WCOJ` synthetic skew join, `ASOF` tick stream, small-corpus `TriModal RRF` smoke).

## [0.3.0] - 2026-11-24 *(Historical — Synthetic `frontier_exam_routes` retracted in `0.6.0`)*

### Added
- Calibration math (`RMS-CE`, `ECE`, `Brier`, Platt scaling), 2D grid DSL synthesizer (`arc_synthesize`), SQL dialect normalizer (`sql_critic`), modular CRT / DPLL SAT utilities, idempotency token gate, and 1D STL projection utilities (`frontier_eval.rs`). Synthetic benchmark-named HTTP routes (`frontier_exam_routes.rs`) were subsequently removed in `0.6.0`.

## [0.2.9] - 2026-11-24

### Added
- Out-of-sample `.jsyn` rank-sweep quality-vs-joules evaluator (`rank_sweep_bench`, `POST /api/v1/synapse/bench/rank_sweep`), Empirical-Bayes `SVF` shrinkage (`calibrate_activation_aware_svf`), and `docs/JSYN-LIVING-WEIGHTS-SPEC.md`. *(Note: 81 low-BPW `.jsyn` causal LLM checkpoints mentioned in `0.2.4`–`0.2.9` were retracted from Hugging Face in `0.6.0` due to `0.27–0.45` output cosine at `0.02–0.17 BPW`.)*

## [0.2.8] - 2026-11-24

### Added
- Double-entry `i128` accounting ledger & torn-WAL fault simulator (`tiger_ledger`), topological dirty-cone DAG + freshness-SLA controller (`secdb_napa`), `1,024`-row `ZoneMap` + `SelectionVector` scanner (`zonemap_scan`), and in-process smoke runner (`certification_routes`).

## [0.2.7] - 2026-11-24

### Added
- Morsel-driven out-of-core spill-to-disk Grace Hash Join & GroupBy (`morsel_spill`), in-process `Hermitage` + Tarjan `SCC` cycle detector (`elle_hermitage`), `.slt` MD5 + `TLP`/`NoREC` SQL conformance checks (`conformance_oracle`), and activation-aware `SVF` post-compression calibration.

## [0.2.6] - 2026-11-24

### Added
- SplitMix64 Rademacher seeding + Joint Binary Gram Gauss-Seidel singular refinement (`synapse`), memory-mapped `.safetensors` & GGUF v2/v3 unpackers (`checkpoint_ingest`), and multi-architecture causal/MoE operator support (`causal_decoder`).

## [0.2.5] - 2026-11-23

### Added
- `NVFP4`, scaled `FP8`, 3D stacked MoE, and `BitNet.cpp` `I2_S` unpackers, plus `Mamba-2` selective SSM and `GQA` block operators.

## [0.2.4] - 2026-11-23

### Added
- MLX packed affine `U32` `.safetensors` and memory-mapped GGUF v2/v3 unpackers, plus `ShortConv`, `QK-Norm`, and `ModernBERT` operators.

## [0.2.3] - 2026-11-23

### Added
- `JouleKube` MVCC SQL-backed Kubernetes object store (`kube_control_plane`), `Whisper` 400-FFT cross-attention Seq2Seq decoder & `SigLIP` encoder (`multimodal_encoder`), in-flight request batching, and `.jsyn` Hugging Face export/pull routes.

## [0.2.2] - 2026-11-23

### Added
- Pure-Rust `.ppm`/`.bmp`/RGB `SigLIP`/`ViT` patchifier, `RIFF/WAVE` Log-Mel `Whisper` audio encoder, self-speculative `LayerSkip` draft-and-verify, and GitHub Release binary/wheel workflow.

## [0.2.1] - 2026-11-23

### Added
- GPU-resident batched `LittleBit-2` + causal `GQA` WGSL shaders (`joule-db-gpu`), `T-MAC` 8-bit LUT CPU matvec, database-native `MCP` host & agent loop (`mcp_agent_loop`), coordinate-ridge `SVF` self-distillation, and SSE per-token energy streaming.

## [0.2.0] - 2026-11-23

### Added
- Pure-Rust causal autoregressive decoder & `WordPiece` encoder (`JouleSynapse`), OpenAI/Anthropic/Ollama HTTP routes, `JouleWard` RFC-2104 `HMAC-SHA256` `ThermoCap` macaroons & `.jcap` bundles, 6-rail `FabricEnergyLedger`, `JoulePulse`, `JouleFlow`, `JouleSynapseMesh`, `JouleCapsule`, and `JouleSpace`.
- Dual-licensed workspace under `MIT OR Apache-2.0`.

## [0.1.0] - 2026-10-15

### Added
- Initial public release surface, runnable `examples/`, and `mdBook` documentation site at `openie-dev.github.io/jouledb`.

[Unreleased]: https://github.com/openIE-dev/jouledb/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/openIE-dev/jouledb/releases/tag/v0.8.0
[0.7.0]: https://github.com/openIE-dev/jouledb/releases/tag/v0.7.0
[0.6.0]: https://github.com/openIE-dev/jouledb/releases/tag/v0.6.0
[0.1.0]: https://github.com/openIE-dev/jouledb/releases/tag/v0.1.0
