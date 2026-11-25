# Introduction

**JouleDB (`v0.6.0` Pre-1.0 Research Preview)** is an open-source (`Apache-2.0 OR MIT`) multi-model database, hyperdimensional semantic store, and experimental low-bitwidth tensor runtime written in Rust.

> **November 25, 2026 Audit & Version Retraction Notice:** Between November 23 and 25, 2026, rapid releases (`v0.2.0` through `v1.0.0`) introduced synthetic in-memory micro-checks named after external industry/AI benchmarks (`HLE`, `ARC-AGI-2`, `SimpleQA`, `SWE-bench`, `TAU-bench`, `TPC-C`, `Jepsen`, `"36 / 36 certified pillars"`), overstated compliance self-attestations, included an unrelated `1.11M`-line `joule-web` crate in workspace line/test counts, shipped Linux binaries requiring `GLIBC_2.43`, and uploaded 81 unverified low-BPW `.jsyn` checkpoints to Hugging Face. Following a source-level audit, **`v1.0.0` has been retracted/yanked**, the project is reverted to **`v0.6.0` (pre-1.0 preview)**, synthetic exam routes and `joule-web` have been removed, 81 unverified/vendor-licensed Hugging Face repos have been deleted, Linux release binaries are built against `glibc <= 2.36` (verified on Ubuntu 24.04 LTS), and external benchmarks (`ClickBench` against `DuckDB` and 5-node TCP `raft.rs` fault injection verified by `elle/elle 0.2.8`) are published with raw scripts, outputs, and losses in [`bench/`](https://github.com/openIE-dev/jouledb/tree/main/bench).

---

## Verified Repository Facts (`v0.6.0`)

- **License:** Dual-licensed under **`Apache-2.0 OR MIT`** ([`LICENSE-APACHE`](https://github.com/openIE-dev/jouledb/blob/main/LICENSE-APACHE), [`LICENSE-MIT`](https://github.com/openIE-dev/jouledb/blob/main/LICENSE-MIT)).
- **Workspace Scope:** **`41` Cargo workspace member crates** (`724,986` lines of Rust across `983` `.rs` files; `12,265` `#[test]` / `#[tokio::test]` attributes, including `280,320` LOC and `6,766` tests in `joule-db-server`) after removing the unrelated `joule-web` crate.
- **Static Safety Lints:**
  - `joule-db-core` enforces `#![deny(unsafe_code)]`, with `1` targeted `#![allow(unsafe_code)]` module (`storage/mmap_extent.rs` for memory-mapped I/O).
  - `#![cfg_attr(not(test), deny(clippy::unwrap_used))]` is enforced in **`3` crates** (`joule-db-core`, `joule-db-branch`, `joule-db-ledger`); workspace-wide `clippy::unwrap_used` is currently `warn`.
- **Energy Evidence Discipline (`measured_j ⊥ reported_j ⊥ est_j ⊥ unmetered`):** Matching [research.openie.dev](https://research.openie.dev), JouleDB separates inline hardware shunt meters (`measured_j` — never emitted by software alone), OS/PMU counters (`reported_j` via Linux RAPL / macOS `powermetrics` or `IOReport` on multi-second workloads), analytical byte/instruction cost models (`est_j` for sub-millisecond in-process paths), and `unmetered` (`null` joules).

---

## Core Subsystems

1. **[Storage, MVCC SSI & PITR](./pitr-and-abi.md):** B-Tree storage, Write-Ahead Logging (`WAL`), Serializable Snapshot Isolation (`SSI`), Copy-on-Write (`CoW`) branching (`joule-db-branch`), binary `TLV` + `CRC-32` envelope framing, and WAL `LSN`/timestamp recovery (`v1_stability_and_pitr.rs`).
2. **[SQL, Columnar Primitives & Networked TCP Raft](./sql-and-consensus.md):** SQL execution over HTTP and PostgreSQL v3 wire (`pgwire`), window functions, `CUBE`, recursive CTEs, `ALP`/`FSST` codecs, and a `3,635`-line Raft state machine (`raft.rs`) with TCP RPC transport (`raft_server.rs`, `raft_transport.rs`), `RaftNode::propose_and_commit` quorum durability, TLA+ specifications, and external `elle` (`0.2.8`) linearizability verification.
3. **[Amorphous Hyperdimensional Store](./amorphic.md):** Extracts typed semantic atoms (`JSON`, `CSV`, `XML`, `k=v`, prose, triples) into `10,240`-bit Vector Symbolic Architecture (`VSA`) bitvectors with SI unit normalization and bounded-memory streaming.
4. **[Experimental `.jsyn` Low-Rank Bitplanes](./jsyn.md):** Compresses 2D projection tensors into packed binary sign bitplanes (`LittleBit-2 + SVF`). Preserves `0.9441` cosine similarity on compact sentence encoders (`all-MiniLM-L6-v2-LittleBit` at `0.44 BPW`), while ultra-low `0.02–0.17 BPW` settings degrade autoregressive LLMs (`cos = 0.27–0.45`) and are **not** drop-in LLM replacements.
5. **[Wire Adapters & Unaudited Security Controls](./trust.md):** PostgreSQL `pgwire` (tested against `psql`, `psycopg3`, `node-pg`, `JDBC`, `SQLAlchemy` in `tests/clients/`), REST compatibility adapters, and technical security primitives (hash-chained logs, retention-lock flags, per-key DEK zeroization, Luhn redaction, Bell-LaPadula lattice checks) — **explicitly unaudited by any external SOC 2, HIPAA, PCI-DSS, CMVP, or FedRAMP assessor**.
