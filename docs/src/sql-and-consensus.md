# SQL Vectorized Execution, Out-of-Core `.jcol` Segments & Multi-Host WAN TCP Raft (`v0.9.0`)

This chapter separates JouleDB's **query execution engines** from its **networked TCP Raft consensus engine** and states what each test actually exercises.

---

## 1. Out-of-Core `.jcol` Segments & Vectorized SQL Operators (`disk_col_segments.rs`, `columnar_exec.rs`, `sql2023_powerhouse.rs`)

- **Out-of-Core `.jcol` Segment Engine (`disk_col_segments.rs` + `execute_segmented`, `v0.8.0`–`v0.9.0`):** Drains tables exceeding `JOULEDB_COL_SPILL_ROWS` (`262,144` rows) into immutable `.jcol` files (`I8/I16/I32/I64` narrowing, bitpacked null masks, compact `u8/u16/u32` local string dictionaries, per-column `global_int_sum` + `min_i64`/`max_i64` ZoneMaps, CRC-32 slice + `JCL1` footer checksums). `columnar_exec::execute_segmented` resolves unfiltered `COUNT(*)`, `MIN`, `MAX`, and `SUM`/`AVG` (`Q0, Q2, Q3, Q6, Q29`) from segment metadata in **zero disk I/O**, loads only query-referenced columns via `pread`, applies dynamic `AtomicI64` Top-K ZoneMap bound pruning (`Q19, Q23, Q24, Q26`), zero-`Arc<str>` `RawTextSlice` text distinct/grouping (`Q05, Q12, Q33, Q34`), and 256-partition `CompactI2Row` in-place sort-reduce for `100M`-group integer aggregations (`Q15, Q31, Q32, Q35`) — executing all **`43 / 43` Official ClickBench queries on the full `99,997,497`-row (`14.78 GB`) dataset in bounded RAM**.
- **Morsel-Parallel In-Memory Engine (`columnar_exec.rs`):** Executes single-table scans, `PrepPred` zone-map/dictionary filters, `VCol` virtual projections (`EXTRACT`, `DATE_TRUNC`, `STRLEN`, `REGEXP_REPLACE`, `CASE WHEN`), algebraic `SUM(col ± k)` scan sharing, inline `HKey`/`GKey` grouped aggregations, `HAVING`, and parallel Top-K `ORDER BY ... LIMIT`.
- **Static `pgwire` `Describe` Inference, Selective Equality Slice & `O(1)` PK Checks (`pgwire.rs`, `query.rs`, `schema.rs`):** Static AST/catalog parameter & result column OID inference (`describe_sql_static` for `Parse`/`Bind`/`Describe` Extended Query pipelines without dry-run execution), `DROP TABLE` `__indexes__` metadata cleanup, parameterized `$1..$N` `SELECT` fast-path binding, `O(1)` secondary-index PK/UNIQUE constraint verification, multi-row PK `INSERT` batching, and PostgreSQL v3 §55.2.3 `discard_until_sync` pipeline error recovery.
- **Window Frames, `CUBE` & `WITH RECURSIVE` (`sql2023_powerhouse.rs`):** Partitioned `ROW_NUMBER`/`RANK`/`DENSE_RANK`/`LAG`/`LEAD`/`CUME_DIST` + `ROWS BETWEEN` sliding frames, 3D `CUBE` with `GROUPING_ID` bitmasks, cycle-pruned recursive CTEs, and runtime Adaptive Query Execution (`AQE`) join switching.
- **Out-of-Core Morsel Spill (`morsel_spill.rs`):** `2,048`-row columnar morsels spilling `CRC-32`-checked `JMSB` partitions to disk when hash state exceeds `memory_budget_bytes`.

---

## 2. Networked TCP Raft Consensus (`raft.rs`, `raft_server.rs`, `raft_transport.rs`)

JouleDB contains **two distinct consensus code paths** that must not be conflated:

1. **Production Networked TCP Raft Engine (`crates/joule-db-server/src/{raft.rs, raft_server.rs, raft_transport.rs}` + `examples/raft_wan_node.rs` + TLA+ spec):**
   - Full Ongaro & Ousterhout Raft state machine (`Follower`, `Candidate`, `Leader`), `RequestVote`, `AppendEntries`, `InstallSnapshot` (with optional Reed-Solomon erasure shards), joint-consensus membership changes, and `CRC-32` on-disk `RaftWal` recovery.
   - **Quorum Writes (`RaftNode::propose_and_commit`):** Appends to the leader WAL and awaits majority replication (`commit_index >= index`) plus Raft §5.4.2 current-term commit verification before returning.
   - **Quorum Linearizable Reads (`RaftNode::read_index` & `RaftNode::read_linearizable`):** Implements Ongaro & Ousterhout Raft §6.4 `ReadIndex`: records `read_index = commit_index`, confirms current-term leadership via a majority round of empty-`AppendEntries` heartbeats (**zero Raft log / WAL writes**), waits until `last_applied >= read_index`, and reads the local state machine linearizably while rejecting stale-leader reads on isolated minority nodes. *(Direct `RaftNode::with_state_machine` remains available for explicitly local/bounded-staleness reads.)*
   - **5-Node Multi-Host Cross-Machine WAN/Tailscale Fault-Injection + Official `elle` (`0.2.8`) Verification (`bench/raft_multihost_wan.py` & `tests/raft_tcp_elle_fault_injection.rs`):**
     - **Cross-Machine WAN Run (`v0.9.0`):** Spawns 5 separate OS `raft_wan_node` processes across **2 distinct physical Linux hosts over encrypted Tailscale WAN** (`framework-128gb-0000` bare-metal + `openie-core-1` Cloud Debian 12 VM, `27.58 ms` median WAN TCP RTT), injects a cross-WAN `2/5` minority split (`["n1", "n4"]` vs `["n2", "n3", "n5"]`) isolating initial leader `n1`, verifies stale write & `ReadIndex` read rejection on `n1` while majority quorum elects `n3` (term `2`) and commits `24` ops, sends real OS **`SIGKILL` (`kill -9`)** to cloud node `n5` (`16` ops committed during crash), restarts `n5` from `raft.wal`, verifies 5-node state convergence (`commit=68, applied=68, x=68, y=39` across both hosts), and checks the `270`-event EDN history (`bench/results/raft_wan_multihost_elle_history.edn`) with Kyle Kingsbury's official Clojure `elle/elle 0.2.8` (`rw-register`, `:strict-serializable` -> **`{:valid? true}`**).
     - **Single-Host Integration Test:** `cargo test -p joule-db-server --test raft_tcp_elle_fault_injection` runs the deterministic 20-link socket-severing `TcpPartitionMatrix` check (`76` events -> `{:valid? true}`).

2. **Fast Scripted In-Memory Unit Simulation (`distributed_jepsen_consensus.rs`):**
   - Simulates 5 `GeoRaftReplica` structs in a `Vec` to unit-test `HlcStamp` monotonicity and staleness-bound arithmetic (`est_wire_energy_nj = 12` analytical constant, `energy_evidence = "est_j"`). **It is a deterministic unit check (`simulation_only: true`), not a networked Raft or Jepsen run.**
