# Zero-Downtime Migration (`jouledb migrate`)

Eliminating the **curse of adoption** requires migrating existing relational, document, search, and lakehouse estates into JouleDB with **mathematical proof of zero data loss**, **shadow-traffic behavioral parity**, and **zero write-freeze downtime**.

`JouleDB v1.0.0` ships a native migration & parity certification engine (`crates/joule-db-server/src/live_migration.rs`, Pillar `31`).

---

## Supported Source Ecosystems

| `--from` Flag | Accepted Input Formats | Automatic Normalization |
| :--- | :--- | :--- |
| **`postgres`** / **`mysql`** | `.sql` (`CREATE TABLE` + `INSERT INTO` dumps) | Schema inference, SQL literal unescaping, `NULL` & numeric preservation |
| **`sqlite`** | Live binary `.db` / `.sqlite` (`SQLite format 3\0`) or `.sql` | Direct `sqlite3 .dump` extraction + canonical primary-key mapping |
| **`mongo`** | MongoDB Extended JSONL (`mongoexport`) | Unwraps `$oid`, `$numberDecimal`, `$numberLong`, `$numberInt` |
| **`elastic`** | Elasticsearch / OpenSearch `_bulk` NDJSON | Extracts `_index`, `_id`, and `_source` documents |
| **`csv`** / **`jsonl`** | RFC-4180 CSV with headers or raw NDJSON | Automatic integer, IEEE float, boolean, and string inference |

---

## 4-Stage Zero-Downtime Cutover Pipeline

1. **Snapshot Bulk Ingest into an `O(1)` CoW Branch:** Source exports load into an isolated Copy-on-Write branch in microseconds without disturbing live readers.
2. **Live CDC High-Watermark Catch-Up (`0 ms` Write Freeze):** In-flight `INSERT` / `UPDATE` / `DELETE` mutations (`Debezium` / WAL `lsn > snapshot_lsn`) replay idempotently until lag reaches `0`.
3. **Order-Independent 256-Bit Multiset `BLAKE3` Reconciliation:** Every row on both source and JouleDB is canonicalized (`-0.0 -> 0.0`, NFC-clean sorted keys), hashed with `BLAKE3`, and folded via a commutative `(sum_lo, sum_hi, xor_lo, xor_hi)` lattice—guaranteeing identical digests regardless of B-Tree/heap/shard scan order while catching any 1-bit tamper, missing row, or duplicate row.
4. **Dual-Run Shadow-Traffic Validator & `FIPS 204 ML-DSA-65` Certificate:** Replays analytical and point queries across source and target (`shadow_parity_rate = 1.0`) and signs an auditor-ready `SOC 2 CC8.1` Post-Quantum Cutover Certificate (`3,309`-byte `ML-DSA-65` signature).

---

## CLI & HTTP Usage

```bash
# Migrate and certify a live SQLite3 binary database file
jouledb migrate --from sqlite --input ./legacy_app.db

# Migrate a PostgreSQL SQL dump
jouledb migrate --from postgres --input ./pg_dump.sql

# Run the built-in 4-ecosystem (Postgres + CDC, Mongo, Elastic, CSV) certification suite
jouledb migrate
```

Over HTTP REST (`POST /api/v1/migrate/run`):

```bash
curl -s -X POST http://127.0.0.1:8080/api/v1/migrate/run \
  -H "Content-Type: application/json" \
  -d '{
    "migration_id": "prod-pg-cutover-01",
    "source_format": "postgres",
    "collection_or_table": "ledger_entries",
    "payload": "CREATE TABLE ledger_entries (id INT PRIMARY KEY, acct TEXT, cents INT);\nINSERT INTO ledger_entries VALUES (1, '\''operating'\'', 1425000);",
    "cdc_events": []
  }'
```
