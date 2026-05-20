# JouleDB Backup Runbook

**Version 1.0 — 2026-05-18**
**Source:** [`crates/joule-db-server/src/backup.rs`](../../crates/joule-db-server/src/backup.rs)

JouleDB backups are page-level — full or incremental snapshots of `data.wdb` + `meta.wdb` + `wal.wdb`, optionally compressed and encrypted.

---

## 1. Backup types

| Type | What it captures | Frequency | Restore time |
|---|---|---|---|
| **Full** | All pages | Daily / weekly | O(database size) |
| **Incremental** | Pages changed since last backup | Every few hours | O(full + N incrementals) |
| **WAL archive** | Continuous WAL stream | Real-time | O(WAL replay window) |

Most deployments combine: full + incremental + WAL archive for point-in-time recovery.

---

## 2. Take a backup

### Full

```bash
joule db backup full --dest /backups/jouledb-$(date -I).wdb
# OR via the daemon admin endpoint
curl -X POST http://localhost:8080/admin/backup \
     -H "Authorization: Bearer $TOKEN" \
     -d '{"type": "full", "dest": "/backups/jouledb-2026-05-18.wdb"}'
```

The backup is taken **online** — the engine takes a CoW snapshot (Phase 5/6 atomic publish, see [`docs/joule-db/cow-mvcc-design.md`](../joule-db/cow-mvcc-design.md)) and streams it to the destination. Writes during the backup are versioned against the snapshot id; the backup itself is a consistent point-in-time view.

### Incremental

```bash
joule db backup incremental \
    --since /backups/jouledb-2026-05-17.wdb \
    --dest   /backups/jouledb-2026-05-18-incr.wdb
```

The engine tracks page-version watermarks; the incremental contains only pages whose version exceeds the parent backup's watermark.

### WAL archive

Configure the server to push WAL segments to remote storage as they're finalized:

```toml
# config.toml
[wal]
archive_destination = "s3://my-bucket/jouledb/wal/"
archive_interval_sec = 60
```

---

## 3. Compression & encryption

```bash
joule db backup full \
    --dest /backups/jouledb-2026-05-18.wdb.zst \
    --compress zstd \
    --encrypt-key /etc/jouledb/backup-key.pem
```

Compression: `none` (default), `zstd`, `lz4`. Encryption: AES-256-GCM with a PEM-supplied key. Encryption is at the page level — the backup file is `<plaintext header><encrypted body>`.

---

## 4. Restore

### From a full backup

```bash
# Stop the server
systemctl stop jouledb

# Restore
joule db restore --src /backups/jouledb-2026-05-17.wdb --dest /var/lib/jouledb/

# Restart
systemctl start jouledb
```

### From full + incrementals + WAL

```bash
# 1. Restore the full
joule db restore --src /backups/jouledb-full-2026-05-15.wdb --dest /var/lib/jouledb/

# 2. Apply incrementals in order
for f in /backups/jouledb-incr-2026-05-1{6,7,8}.wdb; do
    joule db restore --apply-incremental "$f" --dest /var/lib/jouledb/
done

# 3. Replay WAL up to your target point
joule db wal replay /backups/wal/ \
    --to "2026-05-18T13:24:55"

# 4. Restart
systemctl start jouledb
```

The result is a database at the WAL replay end-time, with everything from the full snapshot + each incremental + every WAL frame up to the cutoff.

---

## 5. Schedule

```cron
# /etc/cron.d/jouledb-backup
# Full at 02:00 daily
0 2 * * *   jouledb  joule db backup full        --dest /backups/full-$(date -I).wdb --compress zstd
# Incremental every 4 hours
0 */4 * * * jouledb  joule db backup incremental --dest /backups/incr-$(date +\%Y-\%m-\%dT\%H).wdb \
                                                 --since /backups/$(ls /backups/full-*.wdb | sort | tail -1)
# Prune older than 30 days
0 3 * * *   jouledb  find /backups -name "incr-*" -mtime +30 -delete
0 3 * * *   jouledb  find /backups -name "full-*" -mtime +90 -delete
```

For K8s deployments, use a `CronJob` with the same commands.

---

## 6. Validate backups

```bash
# Verify backup integrity (checksum + page magic)
joule db backup verify --src /backups/jouledb-2026-05-18.wdb

# Restore to a scratch location and run smoke queries
joule db restore --src /backups/jouledb-2026-05-18.wdb --dest /tmp/restore-test/
JOULE_DB_URL=jouledb-local:///tmp/restore-test/ joule db query run "SELECT count(*) FROM users"
rm -rf /tmp/restore-test
```

Untested backups aren't backups. Schedule a weekly restore-validation job.

---

## 7. Backup storage location

In-tree backup snapshots used for tests live in [`crates/joule-db-server/backups/`](../../crates/joule-db-server/backups/) — these are NOT a substitute for real backup. Production backups should live on:

- A different host than the database
- A different storage class (S3 / GCS / on-prem object store)
- With versioning + cross-region replication

For managed-cloud deployments, the operator can provision a backup `PVC` per cluster — but customer-driven backup-to-S3 is the recommended path.

---

## 8. WAL truncate behavior

As of May 2026 (commit `57400f00c`), WAL truncate happens per-sync rather than per-commit. This means:

- WAL retention is longer (good for archive workflows)
- Truncation is amortized across many commits (good for ingest throughput)
- Restore-from-WAL has a slightly larger replay window in normal operation

Account for this when sizing WAL archive storage.

---

## 9. See also

- [`RUNBOOK-RECOVERY.md`](RUNBOOK-RECOVERY.md) — when backups aren't enough
- [`RUNBOOK-CLUSTERING.md`](RUNBOOK-CLUSTERING.md) — multi-replica + cross-region
- [`crates/joule-db-server/src/backup.rs`](../../crates/joule-db-server/src/backup.rs) — source of truth
