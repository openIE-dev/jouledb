# Binary TLV Envelope Framing, WAL Archiving & PITR (`v1_stability_and_pitr.rs`)

`crates/joule-db-core/src/v1_stability_and_pitr.rs` implements versioned binary framing (`JWAL`, `JSYN`, `JWPV`, `ASNP`, `TCAP`) and an in-engine hash-chained WAL archive with `LSN` and microsecond timestamp Point-in-Time Recovery (`ContinuousPitrArchive`).

> **Pre-1.0 Status (`v0.18.0`):** Although the framing header supports forward/backward Tag-Length-Value (`TLV`) extensions (`JWAL`, `JSYN`, `JWPV`, `ASNP`, `TCAP`, plus binary `JMVR\x01` & `JWB1` MVCC/WAL encoding), JouleDB itself is at `v0.18.0` (pre-1.0) and wire/storage schemas may still evolve prior to a future `1.0` release.

## Capabilities Implemented

- **16-Byte Header + IEEE `CRC-32`:** `[magic: 4B][major: u16][minor: u16][flags: u16][tlv_count: u16][crc32: u32]`. Detects single-bit payload or header corruption before decoding.
- **Critical vs. Non-Critical `TLV` Extensions:** Preserves unknown optional tags (`critical == false`) while failing closed (`UnsupportedCriticalExtension`) on unknown security-critical tags (`critical == true`).
- **Hash-Chained WAL Segments & `PITR`:** `ContinuousPitrArchive` chains segments via `FNV-128` + `CRC-32` and replays committed transactions up to `recover_to_lsn(lsn)` or `recover_to_timestamp_us(ts_us)`, excluding uncommitted or aborted transactions.
