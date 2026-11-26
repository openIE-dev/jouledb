# Introduction

**JouleDB (`v0.8.0` Pre-1.0 Research Preview)** is an open-source (`Apache-2.0 OR MIT`) multi-model database, hyperdimensional semantic store, and experimental low-bitwidth tensor runtime written in Rust.

> **Audit Remediation (`v0.6.0`) & Engine Upgrades (`v0.7.0`–`v0.8.0`, 2026-11-25):** Following an external source-level audit in November 2026, **`v1.0.0` was retracted/yanked**, synthetic frontier AI exam routes (`frontier_exam_routes.rs`) and the unrelated `1.11M`-line `joule-web` crate were deleted, 81 unverified low-BPW `.jsyn` causal LLM checkpoints were removed from Hugging Face, and Linux release binaries were rebuilt against `GLIBC <= 2.35` (verified on Ubuntu 24.04 LTS). **`v0.8.0`** completes the **entire official `99,997,497`-row (`14.78 GB`) ClickBench `hits.parquet` suite (`43 / 43` queries, `0` errors, `6.20 GB` post-load RSS / `20.74 GB` kernel `VmHWM` peak, `4.85×` geomean latency vs DuckDB `2.0.1`)** via out-of-core `.jcol` segments, benchmarks **100-warehouse TCP `pgwire` OLTP (`1,659.5 -> 3,353.5 TPS`, `invariant_verified = true`) head-to-head against real `PostgreSQL 16.16` (`fsync=on`) and `SQLite 3.53.1`**, preserves **Raft §6.4 `ReadIndex` quorum linearizable reads** verified by official Clojure `elle 0.2.8`, and expands **`deny(clippy::unwrap_used)` to `7` core crates**. All benchmark scripts, raw JSON/EDN outputs, wins, and losses are published in [`bench/`](https://github.com/openIE-dev/jouledb/tree/main/bench).

---

## Verified Repository Facts (`v0.8.0`)

- **License:** Dual-licensed under **`Apache-2.0 OR MIT`** ([`LICENSE-APACHE`](https://github.com/openIE-dev/jouledb/blob/main/LICENSE-APACHE), [`LICENSE-MIT`](https://github.com/openIE-dev/jouledb/blob/main/LICENSE-MIT)).
- **Workspace Scope:** **`41` Cargo workspace member crates** (`732,508` lines of Rust across `984` `.rs` files; `12,269` `#[test]` / `#[tokio::test]` attributes).
- **Static Safety Lints:**
  - `joule-db-core` enforces `#![deny(unsafe_code)]`, with `1` targeted `#![allow(unsafe_code)]` module (`storage/mmap_extent.rs` for memory-mapped I/O).
  - `#![cfg_attr(not(test), deny(clippy::unwrap_used))]` is enforced in **`7` crates** (`joule-db-core`, `joule-db-branch`, `joule-db-ledger`, `joule-db-local`, `joule-db-energy`, `joule-db-weights`, `joule-db-query`, plus `#![deny(clippy::unwrap_used)]` on `crates/joule-db-server/src/disk_col_segments.rs`) with a blocking CI gate; workspace-wide `clippy::unwrap_used` is `warn`.
- **Energy Evidence Discipline (`measured_j ⊥ reported_j ⊥ est_j ⊥ unmetered`):** Matching [research.openie.dev](https://research.openie.dev), JouleDB separates inline hardware shunt meters (`measured_j` — never emitted by software alone), OS/PMU counters (`reported_j` via Linux RAPL / macOS `powermetrics` or `IOReport` on multi-millisecond/second workloads), analytical byte/instruction cost models (`est_j` for sub-millisecond in-process paths), and `unmetered` (`null` joules).

---

## Core Subsystems

1. **[Storage, MVCC SSI & PITR](./pitr-and-abi.md):** B-Tree storage, Write-Ahead Logging (`WAL`), Serializable Snapshot Isolation (`SSI`), index-assisted `MvccTableStorage` with `O(1)` PK/UNIQUE index checks and group-committed WAL batches, Copy-on-Write (`CoW`) branching (`joule-db-branch`), binary `TLV` + `CRC-32` envelope framing, and WAL `LSN`/timestamp recovery (`v1_stability_and_pitr.rs`).
2. **[SQL, Out-of-Core `.jcol` Columnar Engine & Networked TCP Raft](./sql-and-consensus.md):** SQL execution over HTTP and PostgreSQL v3 wire (`pgwire`), streaming Parquet `COPY` spilling `262,144`-row `.jcol` disk segments, vectorized `execute_segmented` (`43 / 43` Official ClickBench on all `99,997,497` rows), window functions, `CUBE`, recursive CTEs, `ALP`/`FSST` codecs, and networked TCP Raft (`raft.rs`, `raft_server.rs`, `raft_transport.rs`) with `propose_and_commit`, Raft §6.4 `read_linearizable` (`ReadIndex`), TLA+ specifications, and external Clojure `elle` (`0.2.8`) verification.
3. **[Amorphous Hyperdimensional Store](./amorphic.md):** Extracts typed semantic atoms (`JSON`, `CSV`, `XML`, `k=v`, prose, triples) into `10,240`-bit Vector Symbolic Architecture (`VSA`) bitvectors with SI unit normalization and bounded-memory streaming.
4. **[Experimental `.jsyn` Low-Rank Bitplanes](./jsyn.md):** Compresses 2D projection tensors into packed binary sign bitplanes (`LittleBit-2 + SVF`). Preserves `0.9441` cosine similarity on compact sentence encoders (`all-MiniLM-L6-v2-LittleBit` at `0.44 BPW`), while ultra-low `0.02–0.17 BPW` settings degrade autoregressive LLMs (`cos = 0.27–0.45`) and are **not** drop-in LLM replacements.
5. **[Wire Adapters & Unaudited Security Controls](./trust.md):** PostgreSQL `pgwire` (tested against `psql`, `psycopg3`, `node-pg`, `JDBC`, `SQLAlchemy`), REST compatibility adapters, and technical security primitives (hash-chained logs, retention-lock flags, per-key DEK zeroization, Luhn redaction, Bell-LaPadula lattice checks) — **explicitly unaudited by any external SOC 2, HIPAA, PCI-DSS, CMVP, or FedRAMP assessor**.
