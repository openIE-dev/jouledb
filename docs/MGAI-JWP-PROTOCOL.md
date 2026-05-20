# MGAI JWP Protocol Reference

**Version 1.0 — 2026-05-18**
**Protocol:** Joule Wire Protocol (JWP)
**Reference implementation:** [`jwp/`](../crates/jwp/) (frame layer), [`joule-db-server/src/jwp_server.rs`](../crates/joule-db-server/src/jwp_server.rs) (server)
**Sister docs:** [`MGAI-ACP-REFERENCE.md`](MGAI-ACP-REFERENCE.md), [`MGAI-MCP-REFERENCE.md`](MGAI-MCP-REFERENCE.md), [`MGAI-CLI-REFERENCE.md`](MGAI-CLI-REFERENCE.md)

---

## 1. What is JWP

JWP — the **Joule Wire Protocol** — is OpenIE's energy-aware binary transport. It's the JouleDB-native wire protocol, sitting alongside pgwire (PostgreSQL v3 compatibility), HTTP/REST, WebSocket, and MCP in the [`joule-db-server`](../crates/joule-db-server/) surface.

What makes JWP different from pgwire / gRPC / Postgres-replication-protocol:

- **Every frame header carries cumulative energy cost** in µWh (microwatt-hours). The protocol itself is the metering surface — clients see the running energy total ahead of receiving the result payload, and can cancel mid-stream once a budget is exceeded.
- **Two header sizes.** Standard 21-byte for data-carrying frames; compact 8-byte for control frames (heartbeat, cancel) where there's nothing to measure.
- **Compression-negotiated at the frame level.** Flags bits 3–4 carry the per-frame compression algorithm; the server can adapt compression strategy mid-stream as the energy / bandwidth tradeoff shifts.
- **Version-negotiable.** v1 (static 21-byte headers) and v2 (adaptive amorphous headers, compression negotiation, push-style profile updates) coexist on the same port.
- **Auth lives in the frame header**, not in a separate handshake protocol. Bit 5 marks an authenticated connection; bits 6–7 identify which auth path (device-code, passkey, etc.) succeeded.
- **45 frame types** cover the database surface (query/result/done), control (heartbeat/cancel/error), auth (challenge/response/passkey/device-code), session (revoke/extend), CDK commands (deploy/secrets), streaming (chunk-by-chunk LLM output), prepaid billing (balance/topup/usage), and agent-contract lifecycle (propose/sign/extend/recall).

JWP is **not** a pgwire alternative. They coexist — pgwire for "your BI tool already speaks PostgreSQL," JWP for "the client wants energy metering, agent-contract semantics, or streaming LLM responses."

Default port: **`9000`** in managed-cloud deployments (`*.jouledb.cloud:9000`), **`9090`** for local `joule-db-server` instances, **`9200`** when the server is run with `JwpServerConfig::default()` for testing.

---

## 2. Wire format

### 2.1 Standard frame (21-byte header + payload)

```text
Offset  Size  Field            Type / Notes
─────   ────  ──────────────   ──────────────────────────────────────
  0       1   version          0x01 (v1) | 0x02 (v2 standard) | 0xC2 (v2 extended)
  1       1   frame_type       discriminant (see §3)
  2       4   payload_length   u32 big-endian, max 16 MiB
  6       8   energy_uwh       u64 big-endian, cumulative µWh
 14       4   sequence         u32 big-endian, monotonic per connection
 18       3   flags            24-bit bitfield (see §2.3)
─────   ────  ──────────────
 21      …    payload          length = `payload_length` bytes
```

Reference: [`crates/jwp/src/frame.rs:263-294`](../crates/jwp/src/frame.rs#L263). Encode/decode are mechanical big-endian.

### 2.2 Compact frame (v2 only — 8-byte header, no payload)

For control-only frames (heartbeat, cancel) v2 introduces a compact form. No `payload_length` (implicit 0), no `energy_uwh` (inherited from the most recent standard-header frame on this connection).

```text
Offset  Size  Field            Type / Notes
─────   ────  ──────────────   ──────────────────────────────────────
  0       1   version          0x82 (PROTOCOL_VERSION_V2_COMPACT — bit 7 set)
  1       1   frame_type       discriminant
  2       4   sequence         u32 big-endian
  6       2   flags            lower 16 bits of the standard flag set
```

Reference: [`crates/jwp/src/frame.rs:321-339`](../crates/jwp/src/frame.rs#L321).

### 2.3 Flags (24-bit bitfield)

```text
Bit   Name              Meaning
───   ───────────────   ────────────────────────────────────────────
  0   COMPRESSED        Payload is compressed
  1   HAS_CHECKSUM      Payload trailed by a checksum
  2   FINAL_FRAME       Last frame of a logical response
3-4   COMPRESSION_MASK  0=None, 1=Zstd, 2=Lz4
  5   AUTHENTICATED     Connection has been authenticated
6-7   AUTH_PATH_MASK    2-bit auth path identifier (device-code / passkey / …)
  8   ENERGY_SIGNED     Energy values in this frame are cryptographically signed
```

Bits 9–23 are reserved. The `HEADER_LEN`, version constants, and flag mask are public in the [`jwp`](../crates/jwp/) crate so adapters can encode/decode without re-implementing them.

### 2.4 Payload encoding

All structured payloads use **CBOR** (RFC 8949) for compactness and schema evolution. Helpers `cbor_encode` / `cbor_decode` are re-exported from `jwp::frame`. Raw-byte payloads (e.g., for streaming LLM tokens) are passed through unmodified.

---

## 3. Frame types (45 total)

### 3.1 Connection lifecycle

| Type | ID | Direction | Purpose |
|---|---|---|---|
| `Handshake` | `0x01` | both | Initial protocol-version + capability handshake |
| `Heartbeat` | `0x08` | both | Liveness probe; server echoes back |
| `Cancel` | `0x06` | C→S | Abort streaming response / unsubscribe |
| `Error` | `0x07` | S→C | Structured error response (`ErrorPayload`) |
| `Negotiate` | `0x0A` | both | v2: mid-connection capability renegotiation |
| `ProfileUpdate` | `0x0B` | S→C | v2: live connection profile push |
| `EnergyGradient` | `0x0C` | S→C | v2: trending energy-cost direction (getting cheaper / more expensive) |
| `Batch` | `0x0D` | both | v2: multiple sub-frames packed into one wire frame |
| `RateLimit` | `0x11` | S→C | Advisory remaining-budget signal |

### 3.2 Query / result

| Type | ID | Direction | Purpose |
|---|---|---|---|
| `Query` | `0x02` | C→S | Carries `DbQueryPayload` (SQL/Cypher/etc., args, named params, session, limit, explain) |
| `Meta` | `0x03` | S→C | Column metadata for the upcoming result stream |
| `Result` | `0x04` | S→C | Batch of rows (may be multiple in a single response) |
| `Done` | `0x05` | S→C | Query completion summary (row count, affected rows, total µWh, elapsed ms). Marked `FINAL_FRAME`. |
| `Receipt` | `0x09` | S→C | Per-query energy receipt (anchorable via [`joule-db-ledger`](../crates/joule-db-ledger/)) |

### 3.3 Streaming LLM / cascade output

| Type | ID | Direction | Purpose |
|---|---|---|---|
| `StreamChunk` | `0x16` | S→C | Token-by-token cascade or LLM output |

### 3.4 Authentication

| Type | ID | Direction | Purpose |
|---|---|---|---|
| `AuthChallenge` | `0x0E` | S→C | Server challenges client to prove identity |
| `AuthResponse` | `0x0F` | C→S | Client responds with signed credential |
| `AuthSuccess` | `0x10` | S→C | Server confirms authentication |

### 3.5 Device-code flow

| Type | ID | Direction | Purpose |
|---|---|---|---|
| `DeviceCodeRequest` | `0x17` | C→S | Client requests a device code for out-of-band auth |
| `DeviceCodeResponse` | `0x18` | S→C | Server returns `device_code` + `user_code` + verification URL |
| `DeviceCodePoll` | `0x19` | C→S | Client polls for approval |
| `DeviceCodeResult` | `0x1A` | S→C | `pending` / `approved` / `denied` |

### 3.6 Passkey (WebAuthn over JWP)

| Type | ID | Direction | Purpose |
|---|---|---|---|
| `PasskeyRegisterBegin` | `0x1B` | C→S | Begin registration |
| `PasskeyRegisterChallenge` | `0x1C` | S→C | WebAuthn creation challenge |
| `PasskeyRegisterComplete` | `0x1D` | C→S | Complete with attestation |
| `PasskeyLoginBegin` | `0x1E` | C→S | Begin login |
| `PasskeyLoginChallenge` | `0x1F` | S→C | WebAuthn request challenge |
| `PasskeyLoginComplete` | `0x20` | C→S | Complete with assertion |

### 3.7 Session

| Type | ID | Direction | Purpose |
|---|---|---|---|
| `SessionRevoke` | `0x12` | C→S | Client requests session revocation |
| `SessionExtend` | `0x13` | both | Client requests / server confirms session extension |

### 3.8 CDK commands

| Type | ID | Direction | Purpose |
|---|---|---|---|
| `Command` | `0x14` | C→S | Generic CDK command (`deploy`, `create_secret`, …) |
| `CommandResponse` | `0x15` | S→C | Command result |

### 3.9 Billing — prepaid energy balance

| Type | ID | Direction | Purpose |
|---|---|---|---|
| `BalanceQuery` | `0x21` | C→S | Query current energy balance |
| `BalanceResponse` | `0x22` | S→C | Balance info |
| `TopupBegin` | `0x23` | C→S | Request prepaid energy purchase |
| `TopupResponse` | `0x24` | S→C | Checkout URL or confirmation |
| `UsageQuery` | `0x25` | C→S | Request usage history |
| `UsageResponse` | `0x26` | S→C | Usage entries |

### 3.10 Agent-contract lifecycle

The JWP agent-contract surface — used when a host runtime (Claude Code, Cursor, etc.) invokes an OpenIE agent binary as a sub-process under an explicit work contract:

| Type | ID | Direction | Purpose |
|---|---|---|---|
| `ContractPropose` | `0x27` | H→A | Host proposes scope + energy budget + return terms |
| `ContractRespond` | `0x28` | A→H | Agent accepts / rejects / counter-proposes |
| `ContractSigned` | `0x29` | H→A | Host signs the mutually-agreed contract |
| `ExtensionRequest` | `0x2A` | A→H | Agent requests more energy / time with rationale |
| `ExtensionResponse` | `0x2B` | H→A | Host grants / denies / partially grants |
| `AgentReturn` | `0x2C` | A→H | Agent voluntarily returns with findings |
| `AgentRecall` | `0x2D` | H→A | Host force-recalls (budget exceeded / timeout / anomaly) |

H = Host. A = Agent.

---

## 4. CBOR payloads (database-specific)

These payloads ride inside `Query` / `Meta` / `Result` / `Done` frames. Defined in [`jwp_server.rs:38-79`](../crates/joule-db-server/src/jwp_server.rs#L38).

### `DbQueryPayload` (Query frame)

```rust
struct DbQueryPayload {
    sql: String,                              // any of the 7 query languages
    args: Vec<serde_json::Value>,             // positional parameters
    named: BTreeMap<String, serde_json::Value>, // named parameters
    session_id: Option<String>,               // session continuation
    limit: Option<usize>,                     // result-row cap
    explain: bool,                            // return EXPLAIN tree, not rows
}
```

### `DbMetaPayload` (Meta frame)

```rust
struct DbMetaPayload {
    columns: Vec<String>,
    session_id: Option<String>,
}
```

### `DbResultPayload` (Result frame)

```rust
struct DbResultPayload {
    rows: Vec<Vec<serde_json::Value>>,
}
```

### `DbDonePayload` (Done frame)

```rust
struct DbDonePayload {
    row_count: u64,
    affected_rows: Option<u64>,
    total_cost_uwh: u64,    // total µWh consumed by this query
    elapsed_ms: u64,
}
```

Authoritative shapes for the non-database frames (auth, passkey, contract, etc.) live alongside their respective handlers in [`joule-db-server/src/jwp_server.rs`](../crates/joule-db-server/src/jwp_server.rs).

---

## 5. Connection lifecycle

The standard query flow:

```text
Client                              Server
──────                              ──────
Handshake (0x01) ─────────────────► Handshake (0x01)
                                    ◄── HandshakeAck (Handshake reply)

Query (0x02) ─────────────────────► Meta (0x03)
                                    ◄── Result (0x04)   [one or more]
                                    ◄── Result (0x04)
                                    ◄── Done (0x05) FINAL_FRAME

Heartbeat (0x08) ─────────────────► Heartbeat (0x08)    [echo]

Cancel (0x06) ────────────────────► (abort current streaming / unsubscribe)
```

Reference: [`jwp_server.rs:230-330`](../crates/joule-db-server/src/jwp_server.rs#L230) — `handle_connection` dispatch loop.

Every frame on a logical response shares an `energy_uwh` field that monotonically increases; the `Done` frame's value is the authoritative total. Clients reading the value mid-stream can `Cancel` once a budget is exceeded — the server stops mid-result, and the next `Done` reports the partial cost.

### State machine

JWP enforces a state machine on every connection — see [`crates/jwp/src/state_machine.rs`](../crates/jwp/src/state_machine.rs). Transitions are validated server-side; invalid transitions (e.g., a `Result` frame before a `Meta`) close the connection with `Error` (`0x07`).

---

## 6. Server configuration

```rust
pub struct JwpServerConfig {
    pub bind_addr: String,                  // "0.0.0.0:9200" (test default)
    pub max_connections: usize,             // 1000
    pub connection_timeout_secs: u64,       // 300
}
```

In managed cloud (via [`joule-cloud-provisioner`](../crates/joule-cloud-provisioner/)), the bind address is the pod's service IP and the public endpoint is `{name}.jouledb.cloud:9000`. TLS is on by default and terminated at the cloud API gateway.

For local development, the default is `9090` to avoid colliding with the cloud production port.

---

## 7. Transport bindings

JWP can be carried over multiple transports:

| Transport | Crate module | Use case |
|---|---|---|
| Plain TCP | [`crates/jwp/src/transport_tcp.rs`](../crates/jwp/src/transport_tcp.rs) | Local / server-to-server |
| TLS-over-TCP | (TLS feature in `joule-db-server`) | Public internet, cert-pinned |
| WebSocket | (`websocket.rs` in `joule-db-server`) | Browser clients |
| QUIC | [`crates/jwp/src/transport_quic.rs`](../crates/jwp/src/transport_quic.rs) | Low-latency mobile / edge |

Frame format is identical across all transports; the transport layer is responsible for framing (TCP needs length prefixing per frame; WebSocket and QUIC have native message boundaries).

---

## 8. Compression negotiation

Flags bits 3–4 carry the per-frame `CompressionId` (0 = None, 1 = Zstd, 2 = Lz4). The server decides per-frame which compressor to use based on:

1. Client-advertised capabilities in the `Handshake` payload
2. Connection-level energy gradient (compression has its own joule cost; the server skips it when energy is cheap)
3. Frame size — small payloads aren't worth compressing

Either side can issue a `Negotiate` (`0x0A`) mid-stream to renegotiate compression.

Reference: [`crates/jwp/src/compression.rs`](../crates/jwp/src/compression.rs).

---

## 9. Adaptive header (v2)

The v2 extended header (`PROTOCOL_VERSION_V2_EXTENDED = 0xC2`) appends a variable-length **energy breakdown suffix** to the standard 21-byte header. This is used when a `Receipt` frame needs to enumerate per-stage cost (`parse: 2 µWh`, `plan: 14 µWh`, `exec: 8910 µWh`, `serialize: 3 µWh`) without packing it into the CBOR payload.

Reference: [`HeaderFormat::Extended` in frame.rs:38`](../crates/jwp/src/frame.rs#L38). Decode logic is in `frame.rs`; encoded by the receipt-emission path in `joule-db-server::ledger_bridge`.

---

## 10. Error semantics

Errors are reported via the `Error` (`0x07`) frame with an `ErrorPayload` (CBOR). The connection stays open after a recoverable error so the client can issue another `Query` on the same session; fatal errors (`InvalidVersion`, `UnknownFrameType`, state-machine violations) close the connection.

`ErrorPayload` carries:
- `code: u16` — structured error code
- `message: String` — human-readable
- `details: BTreeMap<String, serde_json::Value>` — context-specific fields

The server only reports projected-energy-exceeds-budget errors *before* execution starts; mid-execution overruns trigger an early `Done` frame with the partial cost in `total_cost_uwh`, not an `Error`.

---

## 11. Reference implementation files

| File | Role |
|---|---|
| [`crates/jwp/src/frame.rs`](../crates/jwp/src/frame.rs) | Wire format — headers, frame types, flags |
| [`crates/jwp/src/codec.rs`](../crates/jwp/src/codec.rs) | Tokio framed-codec wrapper |
| [`crates/jwp/src/adaptive_codec.rs`](../crates/jwp/src/adaptive_codec.rs) | v2 adaptive-header codec |
| [`crates/jwp/src/compression.rs`](../crates/jwp/src/compression.rs) | Per-frame Zstd / Lz4 |
| [`crates/jwp/src/encoding.rs`](../crates/jwp/src/encoding.rs) | CBOR helpers |
| [`crates/jwp/src/state_machine.rs`](../crates/jwp/src/state_machine.rs) | Connection state validation |
| [`crates/jwp/src/transport.rs`](../crates/jwp/src/transport.rs) | Transport trait |
| [`crates/jwp/src/transport_tcp.rs`](../crates/jwp/src/transport_tcp.rs) | Plain-TCP transport |
| [`crates/jwp/src/transport_quic.rs`](../crates/jwp/src/transport_quic.rs) | QUIC transport |
| [`crates/jwp/src/profile.rs`](../crates/jwp/src/profile.rs) | Connection profile (capabilities) |
| [`crates/jwp/src/negotiation.rs`](../crates/jwp/src/negotiation.rs) | Capability negotiation |
| [`crates/jwp/src/error.rs`](../crates/jwp/src/error.rs) | `JwpError` |
| [`crates/joule-db-server/src/jwp_server.rs`](../crates/joule-db-server/src/jwp_server.rs) | JouleDB-specific JWP server (Query/Meta/Result/Done dispatch, ledger receipts) |

---

## 12. See also

- [`MGAI-ACP-REFERENCE.md`](MGAI-ACP-REFERENCE.md) — process-level invocation contract (when JWP runs as a daemon)
- [`MGAI-MCP-REFERENCE.md`](MGAI-MCP-REFERENCE.md) — the MCP (Model Context Protocol) alternative for LLM-client integration
- [`MGAI-CLI-REFERENCE.md`](MGAI-CLI-REFERENCE.md) — the CLI surface for `ask-server`, `askdavidc`, etc.
- [`MGAI-SPEC-DOMAIN-JOULEDB.md`](MGAI-SPEC-DOMAIN-JOULEDB.md) — JouleDB domain spec
- [`WHITEPAPER-JOULEDB-2026-05.md`](WHITEPAPER-JOULEDB-2026-05.md) — v0.2 whitepaper

---

*Drafted 2026-05-18 as wave 2 of the JouleDB documentation parity pass. Closes the "not started" item in `MGAI-ACP-REFERENCE.md` §13.*
