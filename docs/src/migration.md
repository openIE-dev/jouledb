# Multi-Format Import & Reconciliation (`jouledb migrate`)

`crates/joule-db-server/src/live_migration.rs` provides a parser, canonicalizer, and order-independent `BLAKE3` multiset checksum tool for importing structured exports into JouleDB (`v0.14.0` preview).

## Supported Input Formats

| `--from` Flag | Input File / Payload | Parser Scope |
| :--- | :--- | :--- |
| **`sqlite`** | Binary `.db` / `.sqlite` file (via local `sqlite3 .dump`) or `.sql` dump | Simple `CREATE TABLE` + `INSERT INTO` value tuples |
| **`postgres`** / **`mysql`** | `.sql` dump (`CREATE TABLE` + `INSERT INTO`) | Standard literal tuples (`NULL`, integers, floats, single-quoted strings); does not parse arbitrary `COPY ... FROM stdin` or PL/pgSQL procedures |
| **`mongo`** | MongoDB Extended JSONL (`mongoexport` one JSON object per line) | Unwraps `$oid`, `$numberDecimal`, `$numberLong`, `$numberInt` |
| **`elastic`** | Elasticsearch `_bulk` NDJSON | Pairs action metadata lines (`index`/`create`) with document lines |
| **`csv`** / **`jsonl`** | Headered RFC-4180 CSV or newline-delimited JSON | Infers integer, float, boolean, and string scalars |

## Multiset `BLAKE3` Checksum & Replay

1. **Order-Independent Multiset Digest:** Each normalized record is hashed with `BLAKE3` and combined via wrapping `128`-bit sum and `XOR` accumulators, verifying whether source and target row sets match regardless of scan order.
2. **Optional Delta Event Replay:** Applies ordered `INSERT` / `UPDATE` / `DELETE` JSON change events after snapshot load during testing or staged cutover, then signs the JSON receipt with `ML-DSA-65` (`inv-pqcrypto`).

```bash
# Import a local SQLite database file and verify multiset BLAKE3 parity
jouledb migrate --from sqlite --input ./legacy_app.db

# Run the built-in 4-format parser & reconciliation smoke check
jouledb migrate
```
