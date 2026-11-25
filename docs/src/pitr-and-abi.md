# `v1.0` ABI Stability, Continuous WAL Archiving & Microsecond `PITR`

Graduating to **`JouleDB v1.0.0`** establishes hard production durability and wire/storage compatibility guarantees (`crates/joule-db-core/src/v1_stability_and_pitr.rs`, Pillar `32`).

---

## 1. The `JouleDB-ABI-1.0.0` Binary & Wire Envelope Contract

All five persistent and wire surfaces use a locked 16-byte little-endian header protected by IEEE `CRC-32` (`[magic: 4B][major: u16][minor: u16][flags: u16][tlv_count: u16][crc32: u32]`) followed by Tag-Length-Value (`TLV`) metadata extensions:

| Magic | Format Family | Scope |
| :---: | :--- | :--- |
| `JWAL` | `WalSegment` (`JWAL-v1`) | Continuous hash-chained Write-Ahead Log segments |
| `JSYN` | `JsynWeights` (`JSYN-v1`) | Sub-1-bit `LittleBit-2 ⊕ Sakana SVF` living model snapshots |
| `JWPV` | `JouleWireProtocol` (`JWP-v1`) | Energy-metered binary client/cluster RPC frames |
| `ASNP` | `AmorphicSnapshot` (`ASNP-v1`) | Bounded-memory Amorphic hologram checkpoints |
| `TCAP` | `ThermoCapEnvelope` (`TCAP-v1`) | `HMAC-SHA256` / PQC attenuated capability macaroons |

### Forward & Backward Compatibility Guarantees (`1.x`)

- **Backward Compatibility:** Any `v1.y` (`y >= 0`) engine reads all `v1.0` artifacts natively without offline conversion.
- **Forward Compatibility:** When a `v1.0` reader encounters a `v1.y` artifact carrying unknown optional TLV tags (`critical == false`), it preserves those extension bytes losslessly. Unknown *critical* security/encryption tags (`critical == true`) fail closed with `AbiContractError::UnsupportedCriticalExtension`.
- **Single-Bit NAND Bit-Rot Detection:** Every envelope verifies its IEEE `CRC-32` before parsing TLVs or payload slices.

---

## 2. Continuous WAL Archiving & Microsecond `PITR` (`RPO = 0`)

`ContinuousPitrArchive` pairs periodic Copy-on-Write base checkpoints (`PitrBaseSnapshot`) with immutable, cryptographically hash-chained `JWAL-v1` archive segments (`prev_segment_hash -> segment_hash`):

- **Exact `LSN` or Microsecond Recovery:** Call `recover_to_lsn(target_lsn)` or `recover_to_timestamp_us(target_ts_us)` (e.g., `1_999_999 µs` right before an accidental `DROP TABLE` or bad batch job at `2_000_000 µs`).
- **Transactional Atomicity:** Only transactions with a durable `Commit` record `<= target` materialize; in-flight uncommitted and `Abort`ed transactions are deterministically excluded.
- **Tamper Evidence:** `verify_archive_integrity()` validates every segment's `CRC-32` and `FNV-128` chain link from `"GENESIS"` before replay.

---

## 3. Zero-Downtime Expand-Contract Rolling Upgrades (`v0.5.x -> v1.0.0`)

Online schema and binary upgrades execute across four lock-free waves (`Expand -> Dual-Write -> Online Backfill -> Contract`), serving 100% of read/write traffic across mixed-version nodes with `0` failed requests.

```bash
curl -s -X POST http://127.0.0.1:8080/api/v1/certify/v1_abi_pitr
```
