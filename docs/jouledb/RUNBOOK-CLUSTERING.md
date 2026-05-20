# JouleDB Clustering Runbook

**Version 1.0 — 2026-05-18**
**Source:** [`crates/joule-db-server/src/raft.rs`](../../crates/joule-db-server/src/raft.rs), [`two_phase_commit.rs`](../../crates/joule-db-server/src/two_phase_commit.rs), [`sharding.rs`](../../crates/joule-db-server/src/sharding.rs), [`read_replica.rs`](../../crates/joule-db-server/src/read_replica.rs), [`replication.rs`](../../crates/joule-db-server/src/replication.rs)

Multi-node JouleDB — Raft consensus for HA, 2PC for distributed transactions, sharding (hash + range), read replicas for scale-out.

**Caveat:** the code ships and tests; production deployments remain single-instance as of May 2026. This runbook describes the supported paths; treat as alpha-grade until you've burned it in.

---

## 1. Topologies

| Topology | When |
|---|---|
| **Single instance** | Default. Up to a few TB on one host. |
| **Raft cluster (3 / 5 / 7 nodes)** | HA needed; can tolerate 1 / 2 / 3 node failures. |
| **Sharded (hash or range)** | Database doesn't fit on one host. |
| **Sharded + Raft per shard** | Both HA and scale-out. |
| **Read replicas** | Read-heavy workloads, eventual consistency acceptable. |

For managed cloud, the `JouleDBCluster` CRD selects topology via `replicas`:

```yaml
spec:
  replicas: 3   # one leader + two followers, Raft
```

---

## 2. Raft consensus

Server-side: [`crates/joule-db-server/src/raft_server.rs`](../../crates/joule-db-server/src/raft_server.rs) and [`raft_transport.rs`](../../crates/joule-db-server/src/raft_transport.rs).

### 2.1 Cluster bootstrap

```toml
# Node 1 config.toml
[raft]
enabled = true
node_id = "node-1"
peers = ["node-2:9091", "node-3:9091"]
data_dir = "/var/lib/jouledb/raft"
heartbeat_ms = 100
election_timeout_ms = 1000
```

Start node 1, then node 2, then node 3. Within ~election-timeout the cluster elects a leader. Verify:

```bash
curl http://node-1:8080/admin/raft/status
# { "term": 4, "leader": "node-2", "state": "Follower", ... }
```

### 2.2 Adding a node

```bash
curl -X POST http://leader:8080/admin/raft/add-member \
     -d '{"node_id": "node-4", "address": "node-4:9091"}'
```

The new node catches up via WAL streaming + a base snapshot from the leader.

### 2.3 Removing a node

```bash
curl -X POST http://leader:8080/admin/raft/remove-member \
     -d '{"node_id": "node-4"}'
```

### 2.4 Reads on followers

By default, reads go to the leader for linearizability. Use a follower for eventually-consistent reads:

```rust
let client = Client::connect("jouledb://leader:9000").await?;
client.set_read_consistency(ReadConsistency::Eventual).await?;
let rows = client.query("SELECT count(*) FROM users").await?;  // may hit follower
```

---

## 3. Sharding

Server-side: [`crates/joule-db-server/src/sharding.rs`](../../crates/joule-db-server/src/sharding.rs).

### 3.1 Hash sharding

```toml
[sharding]
strategy = "hash"
shards = 4
shard_key = "user_id"   # column used to compute hash → shard
```

Each shard runs a separate `joule-db-server` instance. The client library auto-routes by hash; for direct SQL connections, the server proxies the query to the right shard.

### 3.2 Range sharding

```toml
[sharding]
strategy = "range"
shards = [
  { id = 1, range = ["", "m"] },
  { id = 2, range = ["m", ""] },
]
shard_key = "username"
```

Better when access patterns are range-ordered (alphabetical, time-series); allows efficient range scans within a shard.

### 3.3 Cross-shard queries — 2PC

Queries that span shards use 2-phase commit via [`two_phase_commit.rs`](../../crates/joule-db-server/src/two_phase_commit.rs):

```sql
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 'shard-1-user';
UPDATE accounts SET balance = balance + 100 WHERE id = 'shard-2-user';
COMMIT;  -- coordinated 2PC across shard-1 and shard-2
```

Network cost is real (one extra round trip per shard); design schemas so common transactions stay within a single shard.

---

## 4. Read replicas

Lightweight followers that ingest WAL frames from a primary and serve eventually-consistent reads. Lower-overhead than full Raft.

```toml
# Replica config
[read_replica]
enabled = true
primary = "primary-host:9090"
lag_alert_sec = 30
```

Use:

```rust
let primary = Client::connect("jouledb://primary:9090").await?;
let replica = Client::connect("jouledb://replica:9090").await?;

// Writes always go to primary
primary.execute("INSERT INTO orders ...").await?;

// Reads can go to replica (may be slightly stale)
let rows = replica.query("SELECT * FROM orders ORDER BY created_at DESC LIMIT 100").await?;
```

Monitor replication lag via `/admin/replication/status`.

---

## 5. Cross-region

For multi-region deployments:

1. **Synchronous within region** — Raft cluster of 3-5 nodes per region.
2. **Asynchronous across regions** — read replicas in each region tail the primary's WAL.
3. **Active-active** is **not** supported (requires CRDTs at every layer; see [`joule-db-crdt`](../../crates/joule-db-crdt/) for edge-sync semantics that come close).

---

## 6. Failover

### 6.1 With Raft

Automatic. On leader failure, followers detect missed heartbeats within `election_timeout_ms`, hold an election, and one promotes to leader. Total unavailability window: ~1 × election_timeout (default 1 sec).

### 6.2 With read replicas + manual failover

```bash
# 1. Verify replica is sufficiently caught up
curl http://replica:8080/admin/replication/status
# { "lag_sec": 0.2, "last_lsn": "0x..." }

# 2. Promote replica to primary
curl -X POST http://replica:8080/admin/replica/promote

# 3. Update client connection strings (DNS, env vars, service discovery)

# 4. (optional) Reattach the old primary as a new replica once it's healthy
```

---

## 7. Health and observability

Per [`MGAI-ACP-REFERENCE.md`](../MGAI-ACP-REFERENCE.md):

- `/health` — liveness probe (always returns 200 if the process is running)
- `/ready` — readiness probe (returns 503 if catching up Raft, restoring snapshot, etc.)
- `/admin/raft/status` — Raft state, term, leader, peer health
- `/admin/replication/status` — lag, last LSN, replication topology
- Prometheus metrics on `/metrics`

---

## 8. Test coverage

The clustering surface has 6,202 tests in [`joule-db-server`](../../crates/joule-db-server/) — most relevant suites:

| Suite | Path |
|---|---|
| Raft consensus | [`tests/raft_*.rs`](../../crates/joule-db-server/tests/) |
| 2PC | [`tests/two_phase_commit_*.rs`](../../crates/joule-db-server/tests/) |
| Sharding | [`tests/sharding_*.rs`](../../crates/joule-db-server/tests/) |
| Replication | [`tests/replication_*.rs`](../../crates/joule-db-server/tests/) |

---

## 9. Production status

| Surface | Code | Production-tested |
|---|---|---|
| Single instance | ✓ | ✓ (this is the default path) |
| Raft consensus | ✓ | partial — needs a soak before claiming GA |
| 2PC | ✓ | partial |
| Hash sharding | ✓ | partial |
| Range sharding | ✓ | partial |
| Read replicas | ✓ | partial |
| Cross-region replicas | ✓ | not validated |

The [`WHITEPAPER-JOULEDB-2026-05.md`](../WHITEPAPER-JOULEDB-2026-05.md) §6 (The Bad #6) is honest about this: code ships, prod deployments are still single-instance.

---

## 10. See also

- [`RUNBOOK-RECOVERY.md`](RUNBOOK-RECOVERY.md) — what to do when something's wrong
- [`RUNBOOK-BACKUP.md`](RUNBOOK-BACKUP.md) — backup the cluster
- [`CLOUD-OPERATOR.md`](CLOUD-OPERATOR.md) — Kubernetes-managed clusters
- [`crates/joule-db-server/README.md`](../../crates/joule-db-server/README.md) — full module map
