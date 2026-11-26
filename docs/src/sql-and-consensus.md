# SQL Vectorized Execution, Out-of-Core `.jcol` Segments & Networked TCP Raft (`v0.8.0`)

This chapter separates JouleDB's **query execution engines** from its **networked TCP Raft consensus engine** and states what each test actually exercises.

---

## 1. Out-of-Core `.jcol` Segments & Vectorized SQL Operators (`disk_col_segments.rs`, `columnar_exec.rs`, `sql2023_powerhouse.rs`)

- **Out-of-Core `.jcol` Segment Engine (`disk_col_segments.rs` + `execute_segmented`, `v0.8.0`):** Drains tables exceeding `JOULEDB_COL_SPILL_ROWS` (`262,144` rows) into immutable `.jcol` files (`I8/I16/I32/I64` narrowing, bitpacked null masks, compact `u8/u16/u32` local string dictionaries, CRC-32 slice + `JCL1` footer checksums). `columnar_exec::execute_segmented` loads only query-referenced columns via `pread`, applies segment ZoneMap + two-phase `WHERE`-first pruning, and reduces high-cardinality aggregations via 64-shard Radix `GROUP BY` — executing all **`43 / 43` Official ClickBench queries on the full `99,997,497`-row (`14.78 GB`) dataset in `6.20 GB` post-load RSS (`20.74 GB` kernel `VmHWM` peak)**.
- **Morsel-Parallel In-Memory Engine (`columnar_exec.rs`):** Executes single-table scans, `PrepPred` zone-map/dictionary filters, `VCol` virtual projections (`EXTRACT`, `DATE_TRUNC`, `STRLEN`, `REGEXP_REPLACE`, `CASE WHEN`), algebraic `SUM(col ± k)` scan sharing, inline `HKey`/`GKey` grouped aggregations, `HAVING`, and parallel Top-K `ORDER BY ... LIMIT`.
- **Selective Equality Slice & `O(1)` PK Checks for OLTP (`try_selective_table_columns`, `pgwire.rs`):** Parameterized `$1..$N` `SELECT` fast-path binding, `O(1)` secondary-index PK/UNIQUE constraint verification, multi-row PK `INSERT` batching, and PostgreSQL v3 §55.2.3 `discard_until_sync` Extended Query pipeline error recovery.
- **Window Frames, `CUBE` & `WITH RECURSIVE` (`sql2023_powerhouse.rs`):** Partitioned `ROW_NUMBER`/`RANK`/`DENSE_RANK`/`LAG`/`LEAD`/`CUME_DIST` + `ROWS BETWEEN` sliding frames, 3D `CUBE` with `GROUPING_ID` bitmasks, cycle-pruned recursive CTEs, and runtime Adaptive Query Execution (`AQE`) join switching.
- **Out-of-Core Morsel Spill (`morsel_spill.rs`):** `2,048`-row columnar morsels spilling `CRC-32`-checked `JMSB` partitions to disk when hash state exceeds `memory_budget_bytes`.

---

## 2. Networked TCP Raft Consensus (`raft.rs`, `raft_server.rs`, `raft_transport.rs`)

JouleDB contains **two distinct consensus code paths** that must not be conflated:

1. **Production Networked TCP Raft Engine (`crates/joule-db-server/src/{raft.rs, raft_server.rs, raft_transport.rs}` + TLA+ spec):**
   - Full Ongaro & Ousterhout Raft state machine (`Follower`, `Candidate`, `Leader`), `RequestVote`, `AppendEntries`, `InstallSnapshot` (with optional Reed-Solomon erasure shards), joint-consensus membership changes, and `CRC-32` on-disk `RaftWal` recovery.
   - **Quorum Writes (`RaftNode::propose_and_commit`):** Appends to the leader WAL and awaits majority replication (`commit_index >= index`) plus Raft §5.4.2 current-term commit verification before returning.
   - **Quorum Linearizable Reads (`RaftNode::read_index` & `RaftNode::read_linearizable`, `v0.7.0`):** Implements Ongaro & Ousterhout Raft §6.4 `ReadIndex`: records `read_index = commit_index`, confirms current-term leadership via a majority round of empty-`AppendEntries` heartbeats (**zero Raft log / WAL writes**), waits until `last_applied >= read_index`, and reads the local state machine linearizably while rejecting stale-leader reads on isolated minority nodes. *(Direct `RaftNode::with_state_machine` remains available for explicitly local/bounded-staleness reads.)*
   - **Real Fault-Injection + Official `elle` (`0.2.8`) Verification (`tests/raft_tcp_elle_fault_injection.rs` & `bench/elle_check.clj`):** Boots 5 real `RaftNode<KvStateMachine, TcpRaftTransport>` servers with separate on-disk `raft.wal` directories across a 20-link TCP socket-severing proxy (`TcpPartitionMatrix`), injects an asymmetric `2/5` vs `3/5` partition isolating the initial leader, verifies minority write & `read_linearizable` rejection, stops and restarts a node from disk WAL, heals the network, verifies 5-way state convergence with `0` lost committed writes, and checks the exported 76-event EDN history (`bench/results/raft_tcp_elle_history.edn`) with Kyle Kingsbury's official Clojure `elle/elle 0.2.8` (`rw-register`, `:strict-serializable` -> `{:valid? true}`).

2. **Fast Scripted In-Memory Unit Simulation (`distributed_jepsen_consensus.rs`):**
   - Simulates 5 `GeoRaftReplica` structs in a `Vec` to unit-test `HlcStamp` monotonicity and staleness-bound arithmetic (`est_wire_energy_nj = 12` analytical constant, `energy_evidence = "est_j"`). **It is a deterministic unit check (`simulation_only: true`), not a networked Raft or Jepsen run.**
