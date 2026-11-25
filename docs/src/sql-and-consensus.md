# SQL Window/CUBE/AQE & Networked TCP Raft Consensus

This chapter separates JouleDB's **in-process query operators** from its **networked TCP Raft consensus engine** and states what each test actually exercises.

---

## 1. Analytical SQL Operators (`sql2023_powerhouse.rs`, `morsel_spill.rs`, `zonemap_scan.rs`)

- **Partitioned Window Functions & Frames:** `ROW_NUMBER`, `RANK`, `DENSE_RANK`, `LAG`, `LEAD`, `CUME_DIST`, and bounded `ROWS BETWEEN` sliding `SUM` / `AVG` frames.
- **3D `CUBE` / `ROLLUP`:** Single-pass multi-dimensional grouping with `GROUPING_ID` bitmasks.
- **Cycle-Pruned Recursive CTE:** Semi-naive fixpoint traversal with cycle detection (`cycles_detected_and_pruned`).
- **Runtime Adaptive Join Switch (`AQE`):** Switches from nested-loop broadcast to radix-partitioned hash join with heavy-hitter key splitting when observed build cardinality exceeds static estimates on skewed synthetic inputs.
- **Out-of-Core Morsel Spill (`morsel_spill.rs`):** `2,048`-row columnar morsels spilling `CRC-32`-checked `JMSB` partitions to disk when hash state exceeds `memory_budget_bytes`.

---

## 2. Networked TCP Raft Consensus (`raft.rs`, `raft_server.rs`, `raft_transport.rs`)

JouleDB contains **two distinct consensus code paths** that must not be conflated:

1. **Production Networked TCP Raft Engine (`crates/joule-db-server/src/{raft.rs, raft_server.rs, raft_transport.rs}` — `5,269` combined lines + TLA+ spec):**
   - Full Ongaro & Ousterhout Raft state machine (`Follower`, `Candidate`, `Leader`), `RequestVote`, `AppendEntries`, `InstallSnapshot` (with optional Reed-Solomon erasure shards), joint-consensus membership changes, and `CRC-32` on-disk `RaftWal` recovery.
   - **`RaftNode::propose` vs `RaftNode::propose_and_commit` (`v0.6.0` fix):** `propose()` appends to the local leader WAL and notifies background replication asynchronously without waiting for a majority; callers requiring linearizable durability across network partitions **must** call `propose_and_commit(cmd, timeout)`, which awaits `commit_index >= index` and verifies Raft §5.4.2 term preservation before returning.
   - **Real Fault-Injection + Official `elle` (`0.2.8`) Verification (`tests/raft_tcp_elle_fault_injection.rs` & `bench/elle_check.clj`):** Boots 5 real `RaftNode<KvStateMachine, TcpRaftTransport>` servers with separate on-disk `raft.wal` directories across a 20-link TCP socket-severing proxy (`TcpPartitionMatrix`), injects an asymmetric `2/5` vs `3/5` partition isolating the initial leader, stops and restarts a node from disk WAL, heals the network, verifies 5-way state convergence with `0` lost committed writes, and checks the exported 76-event EDN history (`bench/results/raft_tcp_elle_history.edn`) with Kyle Kingsbury's official Clojure `elle/elle 0.2.8` (`rw-register`, `:strict-serializable` -> `{:valid? true}`).
   - **Candid Limitation:** `RaftNode::with_state_machine` reads local memory without an automatic `ReadIndex` round-trip, so linearizable reads must execute a quorum barrier (`propose_and_commit(Command::Noop, ...)`). In addition, `raft_tcp_elle_fault_injection.rs` runs 5 TCP servers over loopback in one multi-threaded test process rather than across 5 separate physical hosts.

2. **Fast Scripted In-Memory Unit Simulation (`distributed_jepsen_consensus.rs`):**
   - Simulates 5 `GeoRaftReplica` structs in a `Vec` to unit-test `HlcStamp` monotonicity and staleness-bound arithmetic (`est_wire_energy_nj = 12` analytical constant, `energy_evidence = "est_j"`). **It is a deterministic unit check (`simulation_only: true`), not a networked Raft or Jepsen run.**
