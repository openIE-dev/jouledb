use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::JwpError;

// ── Constants ─────────────────────────────────────────────────────

/// Standard JWP header size in bytes (v1 and v2 standard).
pub const HEADER_LEN: usize = 21;

/// Compact header size in bytes (v2 only — control frames).
pub const COMPACT_HEADER_LEN: usize = 8;

/// v1 protocol version byte.
pub const PROTOCOL_VERSION: u8 = 0x01;

/// v2 protocol version byte (standard header).
pub const PROTOCOL_VERSION_V2: u8 = 0x02;

/// v2 compact header version byte (bit 7 set).
pub const PROTOCOL_VERSION_V2_COMPACT: u8 = 0x82;

/// v2 extended header version byte (bits 7+6 set).
pub const PROTOCOL_VERSION_V2_EXTENDED: u8 = 0xC2;

// ── Header format ────────────────────────────────────────────────

/// Describes which header format a frame uses on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HeaderFormat {
    /// 21-byte standard header (v1 or v2).
    Standard,
    /// 8-byte compact header for control frames (heartbeat, cancel).
    /// No payload_length (implicit 0), no energy_uwh (inherited).
    Compact,
    /// Standard 21-byte header + variable-length energy breakdown suffix.
    Extended,
}

// ── Frame types ───────────────────────────────────────────────────

/// Wire frame type discriminant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FrameType {
    Handshake = 0x01,
    Query = 0x02,
    Meta = 0x03,
    Result = 0x04,
    Done = 0x05,
    Cancel = 0x06,
    Error = 0x07,
    Heartbeat = 0x08,
    Receipt = 0x09,
    /// v2: Mid-connection capability renegotiation.
    Negotiate = 0x0A,
    /// v2: Server pushes live connection profile to client.
    ProfileUpdate = 0x0B,
    /// v2: Energy cost trend (getting cheaper/more expensive).
    EnergyGradient = 0x0C,
    /// v2: Multiple sub-frames packed into one wire frame.
    Batch = 0x0D,
    /// Auth: Server challenges client to prove identity.
    AuthChallenge = 0x0E,
    /// Auth: Client responds with signed credential.
    AuthResponse = 0x0F,
    /// Auth: Server confirms authentication succeeded.
    AuthSuccess = 0x10,
    /// Advisory rate limit: server tells client its remaining budget.
    RateLimit = 0x11,
    /// Session: client requests session revocation.
    SessionRevoke = 0x12,
    /// Session: client requests session extension / server confirms.
    SessionExtend = 0x13,
    /// Client → Server: generic CDK command (deploy, create_secret, etc.).
    Command = 0x14,
    /// Server → Client: result of a CDK command.
    CommandResponse = 0x15,
    /// Server → Client: streaming text chunk (token-by-token cascade/LLM output).
    StreamChunk = 0x16,
    /// Device-code: client requests a device code for out-of-band auth.
    DeviceCodeRequest = 0x17,
    /// Device-code: server returns device_code + user_code + verification info.
    DeviceCodeResponse = 0x18,
    /// Device-code: client polls for approval status.
    DeviceCodePoll = 0x19,
    /// Device-code: server returns poll result (pending/approved/denied).
    DeviceCodeResult = 0x1A,

    // ── Passkey ceremony frames (WebAuthn over JWP) ──

    /// Passkey: client begins registration (email + optional display name).
    PasskeyRegisterBegin = 0x1B,
    /// Passkey: server returns WebAuthn creation challenge.
    PasskeyRegisterChallenge = 0x1C,
    /// Passkey: client completes registration with attestation.
    PasskeyRegisterComplete = 0x1D,
    /// Passkey: client begins login (optional email for discoverable creds).
    PasskeyLoginBegin = 0x1E,
    /// Passkey: server returns WebAuthn request challenge.
    PasskeyLoginChallenge = 0x1F,
    /// Passkey: client completes login with assertion.
    PasskeyLoginComplete = 0x20,

    // ── Billing frames (prepaid energy balance) ──

    /// Client queries their current energy balance.
    BalanceQuery = 0x21,
    /// Server responds with balance info.
    BalanceResponse = 0x22,
    /// Client requests a top-up (prepaid energy purchase).
    TopupBegin = 0x23,
    /// Server responds with checkout URL or confirmation.
    TopupResponse = 0x24,
    /// Client queries their usage history.
    UsageQuery = 0x25,
    /// Server responds with usage entries.
    UsageResponse = 0x26,

    // ── Agent contract lifecycle frames ──

    /// Host → Agent: propose a work contract (scope, energy budget, return terms).
    ContractPropose = 0x27,
    /// Agent → Host: accept, reject, or counter-propose the contract.
    ContractRespond = 0x28,
    /// Host → Agent: signed contract confirmation (mutual agreement).
    ContractSigned = 0x29,
    /// Agent → Host: request more resources (energy, time) with rationale.
    ExtensionRequest = 0x2A,
    /// Host → Agent: grant, deny, or partially grant extension.
    ExtensionResponse = 0x2B,
    /// Agent → Host: voluntary return with findings.
    AgentReturn = 0x2C,
    /// Host → Agent: force recall (budget exceeded, timeout, anomaly).
    AgentRecall = 0x2D,
}

impl FrameType {
    pub fn from_u8(b: u8) -> Result<Self, JwpError> {
        match b {
            0x01 => Ok(Self::Handshake),
            0x02 => Ok(Self::Query),
            0x03 => Ok(Self::Meta),
            0x04 => Ok(Self::Result),
            0x05 => Ok(Self::Done),
            0x06 => Ok(Self::Cancel),
            0x07 => Ok(Self::Error),
            0x08 => Ok(Self::Heartbeat),
            0x09 => Ok(Self::Receipt),
            0x0A => Ok(Self::Negotiate),
            0x0B => Ok(Self::ProfileUpdate),
            0x0C => Ok(Self::EnergyGradient),
            0x0D => Ok(Self::Batch),
            0x0E => Ok(Self::AuthChallenge),
            0x0F => Ok(Self::AuthResponse),
            0x10 => Ok(Self::AuthSuccess),
            0x11 => Ok(Self::RateLimit),
            0x12 => Ok(Self::SessionRevoke),
            0x13 => Ok(Self::SessionExtend),
            0x14 => Ok(Self::Command),
            0x15 => Ok(Self::CommandResponse),
            0x16 => Ok(Self::StreamChunk),
            0x17 => Ok(Self::DeviceCodeRequest),
            0x18 => Ok(Self::DeviceCodeResponse),
            0x19 => Ok(Self::DeviceCodePoll),
            0x1A => Ok(Self::DeviceCodeResult),
            0x1B => Ok(Self::PasskeyRegisterBegin),
            0x1C => Ok(Self::PasskeyRegisterChallenge),
            0x1D => Ok(Self::PasskeyRegisterComplete),
            0x1E => Ok(Self::PasskeyLoginBegin),
            0x1F => Ok(Self::PasskeyLoginChallenge),
            0x20 => Ok(Self::PasskeyLoginComplete),
            0x21 => Ok(Self::BalanceQuery),
            0x22 => Ok(Self::BalanceResponse),
            0x23 => Ok(Self::TopupBegin),
            0x24 => Ok(Self::TopupResponse),
            0x25 => Ok(Self::UsageQuery),
            0x26 => Ok(Self::UsageResponse),
            0x27 => Ok(Self::ContractPropose),
            0x28 => Ok(Self::ContractRespond),
            0x29 => Ok(Self::ContractSigned),
            0x2A => Ok(Self::ExtensionRequest),
            0x2B => Ok(Self::ExtensionResponse),
            0x2C => Ok(Self::AgentReturn),
            0x2D => Ok(Self::AgentRecall),
            other => Err(JwpError::UnknownFrameType(other)),
        }
    }
}

// ── Flags ─────────────────────────────────────────────────────────

/// 24-bit flags packed into 3 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameFlags(pub u32); // only lower 24 bits used

impl FrameFlags {
    pub const COMPRESSED: u32 = 1 << 0;
    pub const HAS_CHECKSUM: u32 = 1 << 1;
    pub const FINAL_FRAME: u32 = 1 << 2;
    /// Bits 3-4 encode the [`CompressionId`] (0=None, 1=Zstd, 2=Lz4).
    pub const COMPRESSION_MASK: u32 = 0b11 << 3;
    /// Bit 5: connection has been authenticated.
    pub const AUTHENTICATED: u32 = 1 << 5;
    /// Bits 6-7: auth path identifier (2-bit, see `AuthPath::to_wire_bits()`).
    pub const AUTH_PATH_MASK: u32 = 0b11 << 6;
    /// Bit 8: energy values in this frame are cryptographically signed.
    pub const ENERGY_SIGNED: u32 = 1 << 8;

    pub fn new() -> Self {
        Self(0)
    }

    pub fn set(&mut self, flag: u32) {
        self.0 |= flag & 0x00FF_FFFF;
    }

    pub fn is_set(self, flag: u32) -> bool {
        (self.0 & flag) != 0
    }

    /// Extract the compression algorithm ID from bits 3-4.
    pub fn compression_id(self) -> u8 {
        ((self.0 & Self::COMPRESSION_MASK) >> 3) as u8
    }

    /// Set the compression algorithm ID in bits 3-4.
    pub fn with_compression(mut self, id: u8) -> Self {
        self.0 = (self.0 & !Self::COMPRESSION_MASK) | (((id & 0b11) as u32) << 3);
        if id != 0 {
            self.set(Self::COMPRESSED);
        }
        self
    }

    /// Extract the auth path identifier from bits 6-7.
    pub fn auth_path_bits(self) -> u8 {
        ((self.0 & Self::AUTH_PATH_MASK) >> 6) as u8
    }

    /// Set the auth path identifier in bits 6-7.
    pub fn with_auth_path(mut self, bits: u8) -> Self {
        self.0 = (self.0 & !Self::AUTH_PATH_MASK) | (((bits & 0b11) as u32) << 6);
        self.set(Self::AUTHENTICATED);
        self
    }

    /// Encode as 3 bytes (big-endian of lower 24 bits).
    pub fn to_bytes(self) -> [u8; 3] {
        let v = self.0 & 0x00FF_FFFF;
        [(v >> 16) as u8, (v >> 8) as u8, v as u8]
    }

    /// Decode from 3 bytes.
    pub fn from_bytes(b: [u8; 3]) -> Self {
        Self(((b[0] as u32) << 16) | ((b[1] as u32) << 8) | (b[2] as u32))
    }
}

// ── Frame header ──────────────────────────────────────────────────

/// Parsed 21-byte JWP frame header.
///
/// ```text
/// Offset  Size  Field
///   0       1   version          (0x01)
///   1       1   frame_type
///   2       4   payload_length   (u32 big-endian, max 16 MiB)
///   6       8   energy_uwh       (u64 big-endian, cumulative µWh)
///  14       4   sequence         (u32 big-endian, monotonic)
///  18       3   flags            (24-bit bitfield)
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameHeader {
    pub version: u8,
    pub frame_type: FrameType,
    pub payload_length: u32,
    pub energy_uwh: u64,
    pub sequence: u32,
    pub flags: FrameFlags,
}

impl FrameHeader {
    /// Encode to 21 bytes (standard format — v1 or v2).
    pub fn encode(&self, dst: &mut [u8; HEADER_LEN]) {
        dst[0] = self.version;
        dst[1] = self.frame_type as u8;
        dst[2..6].copy_from_slice(&self.payload_length.to_be_bytes());
        dst[6..14].copy_from_slice(&self.energy_uwh.to_be_bytes());
        dst[14..18].copy_from_slice(&self.sequence.to_be_bytes());
        let flag_bytes = self.flags.to_bytes();
        dst[18..21].copy_from_slice(&flag_bytes);
    }

    /// Decode from 21 bytes (v1 only — rejects non-v1 version bytes).
    pub fn decode(src: &[u8; HEADER_LEN]) -> Result<Self, JwpError> {
        let version = src[0];
        if version != PROTOCOL_VERSION {
            return Err(JwpError::InvalidVersion(version));
        }

        let frame_type = FrameType::from_u8(src[1])?;
        let payload_length = u32::from_be_bytes([src[2], src[3], src[4], src[5]]);
        let energy_uwh = u64::from_be_bytes([
            src[6], src[7], src[8], src[9], src[10], src[11], src[12], src[13],
        ]);
        let sequence = u32::from_be_bytes([src[14], src[15], src[16], src[17]]);
        let flags = FrameFlags::from_bytes([src[18], src[19], src[20]]);

        Ok(Self {
            version,
            frame_type,
            payload_length,
            energy_uwh,
            sequence,
            flags,
        })
    }

    /// Encode as an 8-byte compact header (v2 only).
    ///
    /// Layout: `version(1) + frame_type(1) + sequence(4) + flags(2)`
    ///
    /// No `payload_length` (implicit 0) and no `energy_uwh` (inherited
    /// from the last standard-header frame on this connection).
    pub fn encode_compact(&self, dst: &mut [u8; COMPACT_HEADER_LEN]) {
        dst[0] = PROTOCOL_VERSION_V2_COMPACT;
        dst[1] = self.frame_type as u8;
        dst[2..6].copy_from_slice(&self.sequence.to_be_bytes());
        // Only lower 16 bits of flags in compact form
        let v = self.flags.0 & 0xFFFF;
        dst[6] = (v >> 8) as u8;
        dst[7] = v as u8;
    }

    /// Decode a compact 8-byte header (v2 only).
    pub fn decode_compact(src: &[u8; COMPACT_HEADER_LEN]) -> Result<Self, JwpError> {
        let version = src[0];
        if version != PROTOCOL_VERSION_V2_COMPACT {
            return Err(JwpError::InvalidHeaderFormat(version));
        }

        let frame_type = FrameType::from_u8(src[1])?;
        let sequence = u32::from_be_bytes([src[2], src[3], src[4], src[5]]);
        let flags = FrameFlags(((src[6] as u32) << 8) | (src[7] as u32));

        Ok(Self {
            version: PROTOCOL_VERSION_V2,
            frame_type,
            payload_length: 0,
            energy_uwh: 0, // inherited from last standard frame
            sequence,
            flags,
        })
    }

    /// Decode from a byte buffer, auto-detecting header format from the
    /// version byte. Returns `(header, format, bytes_consumed)`.
    ///
    /// - `0x01` → v1 standard 21-byte
    /// - `0x02` → v2 standard 21-byte
    /// - `0x82` → v2 compact 8-byte
    /// - `0xC2` → v2 extended (21-byte + energy breakdown suffix)
    pub fn decode_any(src: &[u8]) -> Result<(Self, HeaderFormat, usize), JwpError> {
        if src.is_empty() {
            return Err(JwpError::IncompleteFrame {
                needed: 1,
                available: 0,
            });
        }

        match src[0] {
            // v1 standard
            PROTOCOL_VERSION => {
                if src.len() < HEADER_LEN {
                    return Err(JwpError::IncompleteFrame {
                        needed: HEADER_LEN,
                        available: src.len(),
                    });
                }
                let buf: [u8; HEADER_LEN] = src[..HEADER_LEN]
                    .try_into()
                    .map_err(|_| JwpError::IncompleteFrame {
                        needed: HEADER_LEN,
                        available: src.len(),
                    })?;
                let header = Self::decode(&buf)?;
                Ok((header, HeaderFormat::Standard, HEADER_LEN))
            }

            // v2 standard
            PROTOCOL_VERSION_V2 => {
                if src.len() < HEADER_LEN {
                    return Err(JwpError::IncompleteFrame {
                        needed: HEADER_LEN,
                        available: src.len(),
                    });
                }
                let header = Self::decode_v2_standard(src)?;
                Ok((header, HeaderFormat::Standard, HEADER_LEN))
            }

            // v2 compact
            PROTOCOL_VERSION_V2_COMPACT => {
                if src.len() < COMPACT_HEADER_LEN {
                    return Err(JwpError::IncompleteFrame {
                        needed: COMPACT_HEADER_LEN,
                        available: src.len(),
                    });
                }
                let buf: [u8; COMPACT_HEADER_LEN] = src[..COMPACT_HEADER_LEN]
                    .try_into()
                    .map_err(|_| JwpError::IncompleteFrame {
                        needed: COMPACT_HEADER_LEN,
                        available: src.len(),
                    })?;
                let header = Self::decode_compact(&buf)?;
                Ok((header, HeaderFormat::Compact, COMPACT_HEADER_LEN))
            }

            // v2 extended
            PROTOCOL_VERSION_V2_EXTENDED => {
                if src.len() < HEADER_LEN {
                    return Err(JwpError::IncompleteFrame {
                        needed: HEADER_LEN,
                        available: src.len(),
                    });
                }
                // Decode the standard 21 bytes first (with v2 version)
                let header = Self::decode_v2_standard_raw(src)?;
                // The energy breakdown suffix is part of the payload —
                // the caller reads it from the payload bytes.
                Ok((header, HeaderFormat::Extended, HEADER_LEN))
            }

            other => Err(JwpError::InvalidHeaderFormat(other)),
        }
    }

    /// Decode v2 standard 21-byte header (accepts version 0x02).
    fn decode_v2_standard(src: &[u8]) -> Result<Self, JwpError> {
        let version = src[0];
        if version != PROTOCOL_VERSION_V2 {
            return Err(JwpError::InvalidVersion(version));
        }
        Self::decode_v2_standard_raw(src)
    }

    /// Decode 21-byte header fields without version check (for v2 standard and extended).
    fn decode_v2_standard_raw(src: &[u8]) -> Result<Self, JwpError> {
        let version = src[0];
        let frame_type = FrameType::from_u8(src[1])?;
        let payload_length = u32::from_be_bytes([src[2], src[3], src[4], src[5]]);
        let energy_uwh = u64::from_be_bytes([
            src[6], src[7], src[8], src[9], src[10], src[11], src[12], src[13],
        ]);
        let sequence = u32::from_be_bytes([src[14], src[15], src[16], src[17]]);
        let flags = FrameFlags::from_bytes([src[18], src[19], src[20]]);

        // Normalize version to base v2
        Ok(Self {
            version: if version == PROTOCOL_VERSION_V2_EXTENDED {
                PROTOCOL_VERSION_V2
            } else {
                version
            },
            frame_type,
            payload_length,
            energy_uwh,
            sequence,
            flags,
        })
    }
}

// ── Energy breakdown (v2 extended header suffix) ─────────────────

/// Per-stage energy breakdown carried in v2 extended header frames.
///
/// Encoded as: `stage_count(1) + [stage_id(1) + energy_uwh(8)] × N`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnergyBreakdown {
    /// Individual stage costs: `(stage_id, energy_uwh)`.
    pub stages: Vec<(u8, u64)>,
}

impl EnergyBreakdown {
    /// Byte size of this breakdown: 1 + 9 × stage_count.
    pub fn wire_len(&self) -> usize {
        1 + self.stages.len() * 9
    }

    /// Encode to bytes.
    pub fn encode(&self, dst: &mut Vec<u8>) {
        dst.push(self.stages.len() as u8);
        for &(stage_id, energy) in &self.stages {
            dst.push(stage_id);
            dst.extend_from_slice(&energy.to_be_bytes());
        }
    }

    /// Decode from bytes. Returns `(breakdown, bytes_consumed)`.
    pub fn decode(src: &[u8]) -> Result<(Self, usize), JwpError> {
        if src.is_empty() {
            return Err(JwpError::IncompleteFrame {
                needed: 1,
                available: 0,
            });
        }
        let stage_count = src[0] as usize;
        let needed = 1 + stage_count * 9;
        if src.len() < needed {
            return Err(JwpError::IncompleteFrame {
                needed,
                available: src.len(),
            });
        }
        let mut stages = Vec::with_capacity(stage_count);
        let mut offset = 1;
        for _ in 0..stage_count {
            let stage_id = src[offset];
            let energy = u64::from_be_bytes([
                src[offset + 1],
                src[offset + 2],
                src[offset + 3],
                src[offset + 4],
                src[offset + 5],
                src[offset + 6],
                src[offset + 7],
                src[offset + 8],
            ]);
            stages.push((stage_id, energy));
            offset += 9;
        }
        Ok((Self { stages }, needed))
    }
}

// ── Complete frame ────────────────────────────────────────────────

/// A complete JWP frame: header + payload bytes.
#[derive(Debug, Clone)]
pub struct JwpFrame {
    pub header: FrameHeader,
    /// Raw payload (CBOR-encoded or empty for Cancel/Heartbeat).
    pub payload: Vec<u8>,
}

impl JwpFrame {
    /// Build a new frame.
    pub fn new(frame_type: FrameType, sequence: u32, energy_uwh: u64, payload: Vec<u8>) -> Self {
        Self {
            header: FrameHeader {
                version: PROTOCOL_VERSION,
                frame_type,
                payload_length: payload.len() as u32,
                energy_uwh,
                sequence,
                flags: FrameFlags::new(),
            },
            payload,
        }
    }

    /// Build a frame with the FINAL_FRAME flag set.
    pub fn new_final(
        frame_type: FrameType,
        sequence: u32,
        energy_uwh: u64,
        payload: Vec<u8>,
    ) -> Self {
        let mut frame = Self::new(frame_type, sequence, energy_uwh, payload);
        frame.header.flags.set(FrameFlags::FINAL_FRAME);
        frame
    }
}

// ── CBOR payload types ────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakePayload {
    pub version: u8,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPayload {
    pub query: String,
    pub limit: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaPayload {
    pub qid: String,
    pub session_id: String,
    pub intent: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultPayload {
    pub rank: u32,
    pub url: String,
    pub title: String,
    pub domain: String,
    pub score: f32,
    pub content_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DonePayload {
    pub total: u32,
    pub total_cost_uwh: u64,
    pub carbon_ug_co2e: u64,
    pub measurement_type: String,
    pub stage_count: u32,
    pub elapsed_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPayload {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiptPayload {
    pub total_cost_uwh: u64,
    pub carbon_ug_co2e: u64,
    pub measurement_type: String,
    pub stage_count: u32,
}

// ── v2 payload types ─────────────────────────────────────────────

/// v2 handshake payload — superset of v1 [`HandshakePayload`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeV2Payload {
    pub version: u8,
    pub capabilities: Vec<String>,
    /// Supported encoding IDs (e.g., `[1]` for CBOR-only).
    #[serde(default)]
    pub supported_encodings: Vec<u8>,
    /// Supported compression IDs (e.g., `[0, 1, 2]` for None+Zstd+Lz4).
    #[serde(default)]
    pub supported_compressions: Vec<u8>,
    /// Maximum batch size the peer can handle.
    #[serde(default)]
    pub max_batch_size: u16,
    /// Preferred energy reporting granularity (0=PerFrame, 1=PerQuery, 2=PerSession).
    #[serde(default)]
    pub energy_reporting: u8,
    /// Supported header formats (0=Standard, 1=Compact, 2=Extended).
    #[serde(default)]
    pub supported_headers: Vec<u8>,
    /// Optional credential for in-band authentication.
    /// Absent for unauthenticated connections (v1 compat).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credential: Option<String>,
}

/// Mid-connection capability renegotiation payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegotiatePayload {
    pub supported_encodings: Vec<u8>,
    pub supported_compressions: Vec<u8>,
    pub max_batch_size: u16,
    pub energy_reporting: u8,
    pub supported_headers: Vec<u8>,
}

/// Server pushes live connection profile to client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileUpdatePayload {
    pub avg_rtt_us: u64,
    pub cache_hit_rate: u16,
    pub queries_served: u64,
    pub cumulative_energy_uwh: u64,
    pub optimal_encoding: u8,
    pub optimal_compression: u8,
    pub optimal_batch_size: u16,
}

/// Energy cost trend — tells client if queries are getting cheaper or
/// more expensive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyGradientPayload {
    /// µWh/query slope. Negative = getting cheaper (corpus learned).
    pub gradient_uwh_per_query: i64,
    /// Number of queries in the gradient window.
    pub window_size: u32,
    /// Per-stage gradient (deterministic BTreeMap ordering).
    pub per_stage: BTreeMap<String, i64>,
    /// Primary savings source if gradient is negative.
    pub savings_source: String,
}

/// Batch frame: packs multiple [`ResultPayload`]s into a single frame.
/// Amortizes 21-byte header overhead across N results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchPayload {
    /// Packed result payloads.
    pub results: Vec<ResultPayload>,
    /// Total energy consumed producing this batch (µWh).
    pub total_energy_uwh: u64,
}

/// Rate limit advisory: server tells client its remaining budget.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitPayload {
    /// Queries remaining in the current window.
    pub queries_remaining: u64,
    /// Window duration in seconds.
    pub window_seconds: u32,
    /// If set, client should wait this many ms before next query.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_after_ms: Option<u64>,
    /// If set, remaining energy budget for this window (µWh).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub energy_budget_uwh: Option<u64>,
}

// ── Command/Response payload types (CDK operations) ──────────────

/// Client → Server: generic CDK operation.
///
/// The `operation` field uses dot-separated namespacing to route to the
/// correct handler: `"admin.deploy"`, `"consumer.list_projects"`,
/// `"shared.health"`, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandPayload {
    /// Unique request ID (UUID) — correlates request with response.
    pub command_id: String,
    /// Operation name (e.g. `"admin.deploy"`, `"shared.health"`).
    pub operation: String,
    /// Operation-specific parameters as a CBOR map.
    #[serde(default)]
    pub parameters: BTreeMap<String, ciborium::Value>,
}

/// Server → Client: result of a CDK command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandResponsePayload {
    /// Echo of the request's `command_id`.
    pub command_id: String,
    /// Status code: 0 = success, non-zero = error category.
    pub status: u16,
    /// CBOR-encoded result body (present on success).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Vec<u8>>,
    /// Human-readable error message (present on failure).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Server → Client: streaming text chunk for cascade/LLM output.
///
/// Sent as tokens are generated. The `Done` frame signals end of stream.
/// The `layer` field indicates which cascade layer is producing output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunkPayload {
    /// Text delta (one or more tokens).
    pub delta: String,
    /// Which cascade layer is producing this chunk (e.g. "cache", "llm", "federation").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer: Option<String>,
    /// Chunk index within this stream (0-based).
    pub index: u32,
}

// ── Billing payload types ─────────────────────────────────────────

/// Client → Server: query current energy balance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceQueryPayload {
    /// Optional session token for auth (if not in handshake).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_token: Option<String>,
}

/// Server → Client: current energy balance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceResponsePayload {
    /// Available energy in microjoules (µJ).
    pub balance_uj: u64,
    /// Stripe customer ID (if linked).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stripe_customer_id: Option<String>,
    /// Lifetime energy consumed in µJ.
    pub lifetime_consumed_uj: u64,
}

/// Client → Server: request energy top-up.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopupBeginPayload {
    /// Amount in USD cents (minimum 500 = $5.00).
    pub amount_cents: u64,
    /// Optional currency override (default: "usd").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub currency: Option<String>,
}

/// Server → Client: top-up response with checkout info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopupResponsePayload {
    /// Stripe checkout session URL (client opens in browser).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkout_url: Option<String>,
    /// Status: "created", "completed", "error".
    pub status: String,
    /// Energy credited in µJ (if completed).
    #[serde(default)]
    pub energy_credited_uj: u64,
    /// Error message if status is "error".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Client → Server: query usage history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageQueryPayload {
    /// Return entries from this Unix timestamp (inclusive).
    #[serde(default)]
    pub from_ts: u64,
    /// Return entries up to this Unix timestamp (inclusive). 0 = now.
    #[serde(default)]
    pub to_ts: u64,
    /// Maximum entries to return (default 50).
    #[serde(default)]
    pub limit: u32,
}

/// Server → Client: usage history response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageResponsePayload {
    /// Usage entries in reverse chronological order.
    pub entries: Vec<UsageEntryPayload>,
    /// Total energy consumed in the queried window (µJ).
    pub total_uj: u64,
    /// Total queries in the queried window.
    pub total_queries: u64,
}

/// A single usage entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageEntryPayload {
    /// Query text (truncated to 120 chars).
    pub query: String,
    /// Energy consumed in µJ.
    pub energy_uj: u64,
    /// Cascade levels used (e.g. "L0-L2" or "L0-L3").
    pub levels: String,
    /// Unix timestamp.
    pub timestamp: u64,
}

// ── Agent contract lifecycle payloads ─────────────────────────────

/// Host → Agent: propose a work contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractProposePayload {
    /// Unique contract identifier.
    pub contract_id: String,
    /// Instance the contract is for.
    pub instance_id: String,
    /// What the agent is being asked to do.
    pub scope_type: String,
    /// Human-readable scope description.
    pub scope_description: String,
    /// Energy budget in microjoules.
    pub energy_budget_uj: u64,
    /// Maximum wall-clock time in milliseconds.
    pub time_limit_ms: u64,
    /// What the agent should return ("result", "report", "data", "findings_and_anomalies", "best_effort").
    pub return_terms: String,
    /// Extension policy ("none", "single", "renewable", "host_approval").
    pub extension_policy: String,
    /// Max additional µJ per extension (if applicable).
    #[serde(default)]
    pub max_extension_uj: u64,
    /// Max number of extensions (if renewable).
    #[serde(default)]
    pub max_extensions: u32,
    /// SHA-256 hash of the proposal for integrity verification.
    pub proposal_hash: String,
}

/// Agent → Host: respond to a contract proposal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractRespondPayload {
    /// Contract being responded to.
    pub contract_id: String,
    /// "accept", "reject", or "counter_propose".
    pub decision: String,
    /// For accept: echo of proposal_hash. For reject: reason. For counter: rationale.
    #[serde(default)]
    pub detail: String,
    /// Counter-propose: requested energy budget (µJ).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_energy_uj: Option<u64>,
    /// Counter-propose: requested time limit (ms).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_time_limit_ms: Option<u64>,
    /// Counter-propose: requested extension policy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_extension_policy: Option<String>,
    /// Agent's timestamp (nanos since epoch).
    #[serde(default)]
    pub timestamp_ns: u64,
}

/// Host → Agent: signed contract confirmation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractSignedPayload {
    /// Contract that was signed.
    pub contract_id: String,
    /// HMAC-SHA256 signature by the host.
    pub host_signature: String,
    /// Proposal hash that was agreed upon.
    pub proposal_hash: String,
    /// Final energy budget (µJ) after negotiation.
    pub final_energy_budget_uj: u64,
    /// Final time limit (ms) after negotiation.
    pub final_time_limit_ms: u64,
}

/// Agent → Host: request more resources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionRequestPayload {
    /// Contract being extended.
    pub contract_id: String,
    /// Additional energy requested (µJ).
    pub additional_energy_uj: u64,
    /// Additional time requested (ms). 0 = no additional time.
    #[serde(default)]
    pub additional_time_ms: u64,
    /// Why the agent needs more resources.
    pub rationale: String,
    /// What the agent has found so far.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interim_findings: Option<String>,
    /// Current energy consumed (µJ).
    pub consumed_uj: u64,
}

/// Host → Agent: extension grant/deny.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionResponsePayload {
    /// Contract being extended.
    pub contract_id: String,
    /// "granted", "denied", or "partial".
    pub decision: String,
    /// Energy granted (µJ). 0 if denied.
    #[serde(default)]
    pub granted_energy_uj: u64,
    /// Time granted (ms). 0 if denied or not requested.
    #[serde(default)]
    pub granted_time_ms: u64,
    /// Reason (for denial or partial grant).
    #[serde(default)]
    pub note: String,
}

/// Agent → Host: voluntary return with findings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentReturnPayload {
    /// Contract the agent is returning from.
    pub contract_id: String,
    /// Summary of work completed.
    pub summary: String,
    /// Whether the agent completed its full scope.
    pub scope_completed: bool,
    /// If not completed, why.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub incomplete_reason: Option<String>,
    /// Energy consumed during engagement (µJ).
    pub energy_consumed_uj: u64,
    /// Wall-clock time spent (ms).
    pub wall_time_ms: u64,
    /// Structured results (CBOR-in-CBOR, opaque to JWP).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub results_cbor: Option<Vec<u8>>,
    /// Anomalies detected.
    #[serde(default)]
    pub anomalies: Vec<AgentAnomalyPayload>,
    /// Energy trace pattern detected.
    #[serde(default)]
    pub energy_pattern: String,
    /// Energy trace confidence (0.0-1.0).
    #[serde(default)]
    pub energy_pattern_confidence: f64,
}

/// An anomaly reported by the agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentAnomalyPayload {
    /// What the agent observed.
    pub description: String,
    /// Severity (0.0 = informational, 1.0 = critical).
    pub severity: f64,
    /// Category: "data_quality", "unexpected_pattern", "performance", "security", "resource", "other".
    pub category: String,
}

/// Host → Agent: force recall.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRecallPayload {
    /// Contract being terminated.
    pub contract_id: String,
    /// Reason for recall.
    pub reason: String,
    /// Energy consumed at recall time (µJ).
    pub energy_consumed_uj: u64,
    /// Whether the energy budget was exceeded.
    pub budget_exceeded: bool,
    /// Energy trace pattern at recall.
    #[serde(default)]
    pub energy_pattern: String,
}

// ── CBOR helpers ──────────────────────────────────────────────────

/// Encode a serializable value to canonical CBOR bytes.
pub fn cbor_encode<T: Serialize>(value: &T) -> Result<Vec<u8>, JwpError> {
    let mut buf = Vec::new();
    ciborium::into_writer(value, &mut buf).map_err(|e| JwpError::CborEncode(e.to_string()))?;
    Ok(buf)
}

/// Decode CBOR bytes into a deserializable value.
pub fn cbor_decode<T: for<'a> Deserialize<'a>>(data: &[u8]) -> Result<T, JwpError> {
    ciborium::from_reader(data).map_err(|e| JwpError::CborDecode(e.to_string()))
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip() {
        let header = FrameHeader {
            version: PROTOCOL_VERSION,
            frame_type: FrameType::Result,
            payload_length: 1234,
            energy_uwh: 42_000,
            sequence: 7,
            flags: FrameFlags(FrameFlags::COMPRESSED | FrameFlags::FINAL_FRAME),
        };

        let mut buf = [0u8; HEADER_LEN];
        header.encode(&mut buf);
        let decoded = FrameHeader::decode(&buf).unwrap();

        assert_eq!(header, decoded);
    }

    #[test]
    fn flags_roundtrip() {
        let mut f = FrameFlags::new();
        f.set(FrameFlags::COMPRESSED);
        f.set(FrameFlags::HAS_CHECKSUM);

        let bytes = f.to_bytes();
        let decoded = FrameFlags::from_bytes(bytes);
        assert!(decoded.is_set(FrameFlags::COMPRESSED));
        assert!(decoded.is_set(FrameFlags::HAS_CHECKSUM));
        assert!(!decoded.is_set(FrameFlags::FINAL_FRAME));
    }

    #[test]
    fn frame_type_roundtrip() {
        for ft in [
            FrameType::Handshake,
            FrameType::Query,
            FrameType::Meta,
            FrameType::Result,
            FrameType::Done,
            FrameType::Cancel,
            FrameType::Error,
            FrameType::Heartbeat,
            FrameType::Receipt,
        ] {
            assert_eq!(FrameType::from_u8(ft as u8).unwrap(), ft);
        }
    }

    #[test]
    fn unknown_frame_type() {
        assert!(FrameType::from_u8(0xFF).is_err());
    }

    #[test]
    fn invalid_version() {
        let mut buf = [0u8; HEADER_LEN];
        buf[0] = 0xFF; // invalid version
        buf[1] = FrameType::Heartbeat as u8;
        assert!(FrameHeader::decode(&buf).is_err());
    }

    #[test]
    fn cbor_roundtrip_query() {
        let payload = QueryPayload {
            query: "rust async".into(),
            limit: 10,
            session_id: Some("sess-1".into()),
        };
        let bytes = cbor_encode(&payload).unwrap();
        let decoded: QueryPayload = cbor_decode(&bytes).unwrap();
        assert_eq!(decoded.query, "rust async");
        assert_eq!(decoded.limit, 10);
        assert_eq!(decoded.session_id.as_deref(), Some("sess-1"));
    }

    #[test]
    fn cbor_roundtrip_result() {
        let payload = ResultPayload {
            rank: 1,
            url: "https://example.com".into(),
            title: "Example".into(),
            domain: "example.com".into(),
            score: 2.5,
            content_hash: "abc123".into(),
        };
        let bytes = cbor_encode(&payload).unwrap();
        let decoded: ResultPayload = cbor_decode(&bytes).unwrap();
        assert_eq!(decoded.rank, 1);
        assert_eq!(decoded.url, "https://example.com");
    }

    // ── v2 frame type tests ──────────────────────────────────────

    #[test]
    fn frame_type_v2_roundtrip() {
        for ft in [
            FrameType::Negotiate,
            FrameType::ProfileUpdate,
            FrameType::EnergyGradient,
            FrameType::Batch,
        ] {
            assert_eq!(FrameType::from_u8(ft as u8).unwrap(), ft);
        }
    }

    // ── Compact header tests ─────────────────────────────────────

    #[test]
    fn compact_header_roundtrip() {
        let header = FrameHeader {
            version: PROTOCOL_VERSION_V2,
            frame_type: FrameType::Heartbeat,
            payload_length: 0,
            energy_uwh: 0,
            sequence: 42,
            flags: FrameFlags::new(),
        };

        let mut buf = [0u8; COMPACT_HEADER_LEN];
        header.encode_compact(&mut buf);
        let decoded = FrameHeader::decode_compact(&buf).unwrap();

        assert_eq!(decoded.frame_type, FrameType::Heartbeat);
        assert_eq!(decoded.sequence, 42);
        assert_eq!(decoded.payload_length, 0);
    }

    #[test]
    fn compact_header_saves_13_bytes() {
        assert_eq!(HEADER_LEN - COMPACT_HEADER_LEN, 13);
    }

    // ── decode_any tests ─────────────────────────────────────────

    #[test]
    fn decode_any_v1_identical_to_decode() {
        let header = FrameHeader {
            version: PROTOCOL_VERSION,
            frame_type: FrameType::Query,
            payload_length: 100,
            energy_uwh: 5000,
            sequence: 3,
            flags: FrameFlags::new(),
        };
        let mut buf = [0u8; HEADER_LEN];
        header.encode(&mut buf);

        // decode_any should produce the same result as decode
        let (decoded_any, format, consumed) = FrameHeader::decode_any(&buf).unwrap();
        let decoded_v1 = FrameHeader::decode(&buf).unwrap();

        assert_eq!(decoded_any, decoded_v1);
        assert_eq!(format, HeaderFormat::Standard);
        assert_eq!(consumed, HEADER_LEN);
    }

    #[test]
    fn decode_any_v2_compact() {
        let header = FrameHeader {
            version: PROTOCOL_VERSION_V2,
            frame_type: FrameType::Cancel,
            payload_length: 0,
            energy_uwh: 0,
            sequence: 7,
            flags: FrameFlags::new(),
        };
        let mut buf = [0u8; COMPACT_HEADER_LEN];
        header.encode_compact(&mut buf);

        let (decoded, format, consumed) = FrameHeader::decode_any(&buf).unwrap();
        assert_eq!(format, HeaderFormat::Compact);
        assert_eq!(consumed, COMPACT_HEADER_LEN);
        assert_eq!(decoded.frame_type, FrameType::Cancel);
        assert_eq!(decoded.sequence, 7);
    }

    #[test]
    fn decode_any_v2_extended() {
        let header = FrameHeader {
            version: PROTOCOL_VERSION_V2,
            frame_type: FrameType::EnergyGradient,
            payload_length: 50,
            energy_uwh: 12000,
            sequence: 10,
            flags: FrameFlags::new(),
        };
        // Write as extended header (version byte 0xC2)
        let mut buf = [0u8; HEADER_LEN];
        header.encode(&mut buf);
        buf[0] = PROTOCOL_VERSION_V2_EXTENDED;

        let (decoded, format, consumed) = FrameHeader::decode_any(&buf).unwrap();
        assert_eq!(format, HeaderFormat::Extended);
        assert_eq!(consumed, HEADER_LEN);
        assert_eq!(decoded.version, PROTOCOL_VERSION_V2); // normalized
        assert_eq!(decoded.energy_uwh, 12000);
    }

    // ── Energy breakdown tests ───────────────────────────────────

    #[test]
    fn energy_breakdown_roundtrip() {
        let breakdown = EnergyBreakdown {
            stages: vec![(1, 100), (2, 250), (3, 50)],
        };

        let mut buf = Vec::new();
        breakdown.encode(&mut buf);
        assert_eq!(buf.len(), breakdown.wire_len());

        let (decoded, consumed) = EnergyBreakdown::decode(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(decoded, breakdown);
    }

    #[test]
    fn energy_breakdown_empty() {
        let breakdown = EnergyBreakdown { stages: vec![] };
        let mut buf = Vec::new();
        breakdown.encode(&mut buf);
        assert_eq!(buf.len(), 1); // just the count byte

        let (decoded, consumed) = EnergyBreakdown::decode(&buf).unwrap();
        assert_eq!(consumed, 1);
        assert_eq!(decoded.stages.len(), 0);
    }

    // ── Flags compression bits ───────────────────────────────────

    #[test]
    fn flags_compression_id_roundtrip() {
        let f = FrameFlags::new().with_compression(1); // Zstd
        assert_eq!(f.compression_id(), 1);
        assert!(f.is_set(FrameFlags::COMPRESSED));

        let f2 = FrameFlags::new().with_compression(2); // Lz4
        assert_eq!(f2.compression_id(), 2);

        let f3 = FrameFlags::new().with_compression(0); // None
        assert_eq!(f3.compression_id(), 0);
        assert!(!f3.is_set(FrameFlags::COMPRESSED));
    }

    // ── v2 payload types ─────────────────────────────────────────

    #[test]
    fn cbor_roundtrip_energy_gradient() {
        let mut per_stage = BTreeMap::new();
        per_stage.insert("normalization".into(), -15_i64);
        per_stage.insert("search".into(), 5_i64);

        let payload = EnergyGradientPayload {
            gradient_uwh_per_query: -10,
            window_size: 16,
            per_stage,
            savings_source: "local_cache".into(),
        };

        let bytes = cbor_encode(&payload).unwrap();
        let decoded: EnergyGradientPayload = cbor_decode(&bytes).unwrap();
        assert_eq!(decoded.gradient_uwh_per_query, -10);
        assert_eq!(decoded.per_stage.len(), 2);
        assert_eq!(decoded.per_stage["normalization"], -15);
    }

    #[test]
    fn cbor_roundtrip_handshake_v2() {
        let payload = HandshakeV2Payload {
            version: 2,
            capabilities: vec!["search".into(), "cancel".into()],
            supported_encodings: vec![1],
            supported_compressions: vec![0, 1, 2],
            max_batch_size: 10,
            energy_reporting: 0,
            supported_headers: vec![0, 1, 2],
            credential: None,
        };
        let bytes = cbor_encode(&payload).unwrap();
        let decoded: HandshakeV2Payload = cbor_decode(&bytes).unwrap();
        assert_eq!(decoded.version, 2);
        assert_eq!(decoded.supported_compressions, vec![0, 1, 2]);
        assert_eq!(decoded.max_batch_size, 10);
        assert!(decoded.credential.is_none());
    }

    #[test]
    fn cbor_roundtrip_handshake_v2_with_credential() {
        let payload = HandshakeV2Payload {
            version: 2,
            capabilities: vec!["search".into()],
            supported_encodings: vec![1],
            supported_compressions: vec![0],
            max_batch_size: 8,
            energy_reporting: 0,
            supported_headers: vec![0],
            credential: Some("user@example.com".into()),
        };
        let bytes = cbor_encode(&payload).unwrap();
        let decoded: HandshakeV2Payload = cbor_decode(&bytes).unwrap();
        assert_eq!(decoded.credential.as_deref(), Some("user@example.com"));
    }

    #[test]
    fn handshake_v2_credential_absent_by_default() {
        // Without credential field in CBOR, serde(default) fills None
        let payload = HandshakeV2Payload {
            version: 2,
            capabilities: vec![],
            supported_encodings: vec![1],
            supported_compressions: vec![0],
            max_batch_size: 1,
            energy_reporting: 0,
            supported_headers: vec![],
            credential: None,
        };
        let bytes = cbor_encode(&payload).unwrap();
        // Verify the CBOR doesn't contain "credential" key (skip_serializing_if)
        let decoded: HandshakeV2Payload = cbor_decode(&bytes).unwrap();
        assert!(decoded.credential.is_none());
    }

    #[test]
    fn auth_flags_roundtrip() {
        let mut flags = FrameFlags::new();
        flags = flags.with_auth_path(0b10); // DeviceCode
        assert!(flags.is_set(FrameFlags::AUTHENTICATED));
        assert_eq!(flags.auth_path_bits(), 0b10);

        // Roundtrip through bytes
        let bytes = flags.to_bytes();
        let restored = FrameFlags::from_bytes(bytes);
        assert!(restored.is_set(FrameFlags::AUTHENTICATED));
        assert_eq!(restored.auth_path_bits(), 0b10);
    }

    // ── Batch + RateLimit payload tests ────────────────────────────

    #[test]
    fn cbor_roundtrip_batch_payload() {
        let payload = BatchPayload {
            results: vec![
                ResultPayload {
                    rank: 1,
                    url: "https://a.com".into(),
                    title: "A".into(),
                    domain: "a.com".into(),
                    score: 1.0,
                    content_hash: "aaa".into(),
                },
                ResultPayload {
                    rank: 2,
                    url: "https://b.com".into(),
                    title: "B".into(),
                    domain: "b.com".into(),
                    score: 0.8,
                    content_hash: "bbb".into(),
                },
            ],
            total_energy_uwh: 500,
        };
        let bytes = cbor_encode(&payload).unwrap();
        let decoded: BatchPayload = cbor_decode(&bytes).unwrap();
        assert_eq!(decoded.results.len(), 2);
        assert_eq!(decoded.results[0].rank, 1);
        assert_eq!(decoded.results[1].url, "https://b.com");
        assert_eq!(decoded.total_energy_uwh, 500);
    }

    #[test]
    fn cbor_roundtrip_batch_empty() {
        let payload = BatchPayload {
            results: vec![],
            total_energy_uwh: 0,
        };
        let bytes = cbor_encode(&payload).unwrap();
        let decoded: BatchPayload = cbor_decode(&bytes).unwrap();
        assert_eq!(decoded.results.len(), 0);
        assert_eq!(decoded.total_energy_uwh, 0);
    }

    #[test]
    fn cbor_roundtrip_rate_limit_full() {
        let payload = RateLimitPayload {
            queries_remaining: 42,
            window_seconds: 60,
            retry_after_ms: Some(500),
            energy_budget_uwh: Some(10_000),
        };
        let bytes = cbor_encode(&payload).unwrap();
        let decoded: RateLimitPayload = cbor_decode(&bytes).unwrap();
        assert_eq!(decoded.queries_remaining, 42);
        assert_eq!(decoded.window_seconds, 60);
        assert_eq!(decoded.retry_after_ms, Some(500));
        assert_eq!(decoded.energy_budget_uwh, Some(10_000));
    }

    #[test]
    fn cbor_roundtrip_rate_limit_minimal() {
        let payload = RateLimitPayload {
            queries_remaining: 100,
            window_seconds: 3600,
            retry_after_ms: None,
            energy_budget_uwh: None,
        };
        let bytes = cbor_encode(&payload).unwrap();
        let decoded: RateLimitPayload = cbor_decode(&bytes).unwrap();
        assert_eq!(decoded.queries_remaining, 100);
        assert_eq!(decoded.window_seconds, 3600);
        assert!(decoded.retry_after_ms.is_none());
        assert!(decoded.energy_budget_uwh.is_none());
    }

    #[test]
    fn frame_type_rate_limit_roundtrip() {
        assert_eq!(FrameType::from_u8(0x11).unwrap(), FrameType::RateLimit);
    }

    #[test]
    fn energy_signed_flag() {
        let mut flags = FrameFlags::new();
        assert!(!flags.is_set(FrameFlags::ENERGY_SIGNED));
        flags.set(FrameFlags::ENERGY_SIGNED);
        assert!(flags.is_set(FrameFlags::ENERGY_SIGNED));

        let bytes = flags.to_bytes();
        let restored = FrameFlags::from_bytes(bytes);
        assert!(restored.is_set(FrameFlags::ENERGY_SIGNED));
    }

    // ── Command/CommandResponse tests ─────────────────────────────

    #[test]
    fn frame_type_command_roundtrip() {
        assert_eq!(FrameType::from_u8(0x14).unwrap(), FrameType::Command);
        assert_eq!(
            FrameType::from_u8(0x15).unwrap(),
            FrameType::CommandResponse
        );
    }

    #[test]
    fn cbor_roundtrip_command_payload() {
        let mut params = BTreeMap::new();
        params.insert("name".into(), ciborium::Value::Text("test-wl".into()));
        params.insert("memory_mb".into(), ciborium::Value::Integer(128.into()));

        let payload = CommandPayload {
            command_id: "cmd-001".into(),
            operation: "admin.deploy".into(),
            parameters: params,
        };
        let bytes = cbor_encode(&payload).unwrap();
        let decoded: CommandPayload = cbor_decode(&bytes).unwrap();
        assert_eq!(decoded.command_id, "cmd-001");
        assert_eq!(decoded.operation, "admin.deploy");
        assert_eq!(decoded.parameters.len(), 2);
    }

    #[test]
    fn cbor_roundtrip_command_response_success() {
        let payload = CommandResponsePayload {
            command_id: "cmd-001".into(),
            status: 0,
            result: Some(vec![0xA1, 0x62, 0x6F, 0x6B, 0xF5]), // {"ok": true} in CBOR
            error: None,
        };
        let bytes = cbor_encode(&payload).unwrap();
        let decoded: CommandResponsePayload = cbor_decode(&bytes).unwrap();
        assert_eq!(decoded.command_id, "cmd-001");
        assert_eq!(decoded.status, 0);
        assert!(decoded.result.is_some());
        assert!(decoded.error.is_none());
    }

    #[test]
    fn cbor_roundtrip_command_response_error() {
        let payload = CommandResponsePayload {
            command_id: "cmd-002".into(),
            status: 404,
            result: None,
            error: Some("workload not found".into()),
        };
        let bytes = cbor_encode(&payload).unwrap();
        let decoded: CommandResponsePayload = cbor_decode(&bytes).unwrap();
        assert_eq!(decoded.command_id, "cmd-002");
        assert_eq!(decoded.status, 404);
        assert!(decoded.result.is_none());
        assert_eq!(decoded.error.as_deref(), Some("workload not found"));
    }

    #[test]
    fn cbor_roundtrip_command_empty_params() {
        let payload = CommandPayload {
            command_id: "cmd-003".into(),
            operation: "shared.health".into(),
            parameters: BTreeMap::new(),
        };
        let bytes = cbor_encode(&payload).unwrap();
        let decoded: CommandPayload = cbor_decode(&bytes).unwrap();
        assert_eq!(decoded.operation, "shared.health");
        assert!(decoded.parameters.is_empty());
    }

    // ── Passkey + Billing frame type tests ────────────────────────

    #[test]
    fn frame_type_passkey_roundtrip() {
        for ft in [
            FrameType::PasskeyRegisterBegin,
            FrameType::PasskeyRegisterChallenge,
            FrameType::PasskeyRegisterComplete,
            FrameType::PasskeyLoginBegin,
            FrameType::PasskeyLoginChallenge,
            FrameType::PasskeyLoginComplete,
        ] {
            assert_eq!(FrameType::from_u8(ft as u8).unwrap(), ft);
        }
    }

    #[test]
    fn frame_type_billing_roundtrip() {
        for ft in [
            FrameType::BalanceQuery,
            FrameType::BalanceResponse,
            FrameType::TopupBegin,
            FrameType::TopupResponse,
            FrameType::UsageQuery,
            FrameType::UsageResponse,
        ] {
            assert_eq!(FrameType::from_u8(ft as u8).unwrap(), ft);
        }
    }

    #[test]
    fn frame_type_passkey_discriminants() {
        assert_eq!(FrameType::PasskeyRegisterBegin as u8, 0x1B);
        assert_eq!(FrameType::PasskeyLoginComplete as u8, 0x20);
        assert_eq!(FrameType::BalanceQuery as u8, 0x21);
        assert_eq!(FrameType::UsageResponse as u8, 0x26);
    }

    #[test]
    fn cbor_roundtrip_balance_query() {
        let payload = BalanceQueryPayload {
            session_token: Some("st-abc".into()),
        };
        let bytes = cbor_encode(&payload).unwrap();
        let decoded: BalanceQueryPayload = cbor_decode(&bytes).unwrap();
        assert_eq!(decoded.session_token.as_deref(), Some("st-abc"));
    }

    #[test]
    fn cbor_roundtrip_balance_response() {
        let payload = BalanceResponsePayload {
            balance_uj: 50_000_000,
            stripe_customer_id: Some("cus_abc".into()),
            lifetime_consumed_uj: 1_200_000,
        };
        let bytes = cbor_encode(&payload).unwrap();
        let decoded: BalanceResponsePayload = cbor_decode(&bytes).unwrap();
        assert_eq!(decoded.balance_uj, 50_000_000);
        assert_eq!(decoded.stripe_customer_id.as_deref(), Some("cus_abc"));
        assert_eq!(decoded.lifetime_consumed_uj, 1_200_000);
    }

    #[test]
    fn cbor_roundtrip_topup() {
        let begin = TopupBeginPayload {
            amount_cents: 500,
            currency: None,
        };
        let bytes = cbor_encode(&begin).unwrap();
        let decoded: TopupBeginPayload = cbor_decode(&bytes).unwrap();
        assert_eq!(decoded.amount_cents, 500);
        assert!(decoded.currency.is_none());

        let response = TopupResponsePayload {
            checkout_url: Some("https://checkout.stripe.com/cs_test_abc".into()),
            status: "created".into(),
            energy_credited_uj: 0,
            error: None,
        };
        let bytes = cbor_encode(&response).unwrap();
        let decoded: TopupResponsePayload = cbor_decode(&bytes).unwrap();
        assert_eq!(decoded.status, "created");
        assert!(decoded.checkout_url.is_some());
    }

    #[test]
    fn cbor_roundtrip_usage() {
        let query = UsageQueryPayload {
            from_ts: 1700000000,
            to_ts: 1700100000,
            limit: 10,
        };
        let bytes = cbor_encode(&query).unwrap();
        let decoded: UsageQueryPayload = cbor_decode(&bytes).unwrap();
        assert_eq!(decoded.from_ts, 1700000000);
        assert_eq!(decoded.limit, 10);

        let response = UsageResponsePayload {
            entries: vec![UsageEntryPayload {
                query: "what is rust".into(),
                energy_uj: 42_000,
                levels: "L0-L2".into(),
                timestamp: 1700050000,
            }],
            total_uj: 42_000,
            total_queries: 1,
        };
        let bytes = cbor_encode(&response).unwrap();
        let decoded: UsageResponsePayload = cbor_decode(&bytes).unwrap();
        assert_eq!(decoded.entries.len(), 1);
        assert_eq!(decoded.entries[0].energy_uj, 42_000);
        assert_eq!(decoded.total_queries, 1);
    }

    // ── Agent contract frame type + payload tests ────────────────

    #[test]
    fn frame_type_contract_roundtrip() {
        for ft in [
            FrameType::ContractPropose,
            FrameType::ContractRespond,
            FrameType::ContractSigned,
            FrameType::ExtensionRequest,
            FrameType::ExtensionResponse,
            FrameType::AgentReturn,
            FrameType::AgentRecall,
        ] {
            assert_eq!(FrameType::from_u8(ft as u8).unwrap(), ft);
        }
    }

    #[test]
    fn frame_type_contract_discriminants() {
        assert_eq!(FrameType::ContractPropose as u8, 0x27);
        assert_eq!(FrameType::ContractRespond as u8, 0x28);
        assert_eq!(FrameType::ContractSigned as u8, 0x29);
        assert_eq!(FrameType::ExtensionRequest as u8, 0x2A);
        assert_eq!(FrameType::ExtensionResponse as u8, 0x2B);
        assert_eq!(FrameType::AgentReturn as u8, 0x2C);
        assert_eq!(FrameType::AgentRecall as u8, 0x2D);
    }

    #[test]
    fn cbor_roundtrip_contract_propose() {
        let payload = ContractProposePayload {
            contract_id: "c-001".into(),
            instance_id: "inst-001".into(),
            scope_type: "query".into(),
            scope_description: "find nearest neighbors in graph".into(),
            energy_budget_uj: 50_000_000,
            time_limit_ms: 30_000,
            return_terms: "result".into(),
            extension_policy: "single".into(),
            max_extension_uj: 25_000_000,
            max_extensions: 1,
            proposal_hash: "sha256:abc123".into(),
        };
        let bytes = cbor_encode(&payload).unwrap();
        let decoded: ContractProposePayload = cbor_decode(&bytes).unwrap();
        assert_eq!(decoded.contract_id, "c-001");
        assert_eq!(decoded.energy_budget_uj, 50_000_000);
        assert_eq!(decoded.max_extension_uj, 25_000_000);
    }

    #[test]
    fn cbor_roundtrip_contract_respond_accept() {
        let payload = ContractRespondPayload {
            contract_id: "c-001".into(),
            decision: "accept".into(),
            detail: "sha256:abc123".into(),
            requested_energy_uj: None,
            requested_time_limit_ms: None,
            requested_extension_policy: None,
            timestamp_ns: 1700000000_000_000_000,
        };
        let bytes = cbor_encode(&payload).unwrap();
        let decoded: ContractRespondPayload = cbor_decode(&bytes).unwrap();
        assert_eq!(decoded.decision, "accept");
        assert!(decoded.requested_energy_uj.is_none());
    }

    #[test]
    fn cbor_roundtrip_contract_respond_counter() {
        let payload = ContractRespondPayload {
            contract_id: "c-001".into(),
            decision: "counter_propose".into(),
            detail: "need more energy for exploration".into(),
            requested_energy_uj: Some(100_000_000),
            requested_time_limit_ms: Some(60_000),
            requested_extension_policy: Some("renewable".into()),
            timestamp_ns: 0,
        };
        let bytes = cbor_encode(&payload).unwrap();
        let decoded: ContractRespondPayload = cbor_decode(&bytes).unwrap();
        assert_eq!(decoded.decision, "counter_propose");
        assert_eq!(decoded.requested_energy_uj, Some(100_000_000));
    }

    #[test]
    fn cbor_roundtrip_contract_signed() {
        let payload = ContractSignedPayload {
            contract_id: "c-001".into(),
            host_signature: "contract:deadbeef".into(),
            proposal_hash: "sha256:abc123".into(),
            final_energy_budget_uj: 75_000_000,
            final_time_limit_ms: 45_000,
        };
        let bytes = cbor_encode(&payload).unwrap();
        let decoded: ContractSignedPayload = cbor_decode(&bytes).unwrap();
        assert_eq!(decoded.contract_id, "c-001");
        assert_eq!(decoded.final_energy_budget_uj, 75_000_000);
    }

    #[test]
    fn cbor_roundtrip_extension_request() {
        let payload = ExtensionRequestPayload {
            contract_id: "c-001".into(),
            additional_energy_uj: 20_000_000,
            additional_time_ms: 15_000,
            rationale: "found interesting subgraph cluster".into(),
            interim_findings: Some("3 anomalous nodes identified".into()),
            consumed_uj: 45_000_000,
        };
        let bytes = cbor_encode(&payload).unwrap();
        let decoded: ExtensionRequestPayload = cbor_decode(&bytes).unwrap();
        assert_eq!(decoded.additional_energy_uj, 20_000_000);
        assert_eq!(decoded.interim_findings.as_deref(), Some("3 anomalous nodes identified"));
    }

    #[test]
    fn cbor_roundtrip_extension_response() {
        let payload = ExtensionResponsePayload {
            contract_id: "c-001".into(),
            decision: "granted".into(),
            granted_energy_uj: 20_000_000,
            granted_time_ms: 15_000,
            note: String::new(),
        };
        let bytes = cbor_encode(&payload).unwrap();
        let decoded: ExtensionResponsePayload = cbor_decode(&bytes).unwrap();
        assert_eq!(decoded.decision, "granted");
        assert_eq!(decoded.granted_energy_uj, 20_000_000);
    }

    #[test]
    fn cbor_roundtrip_agent_return() {
        let payload = AgentReturnPayload {
            contract_id: "c-001".into(),
            summary: "analyzed 1000 graph nodes, found 3 clusters".into(),
            scope_completed: true,
            incomplete_reason: None,
            energy_consumed_uj: 42_000_000,
            wall_time_ms: 25_000,
            results_cbor: Some(vec![0xA1, 0x61, 0x6E, 0x03]), // {"n": 3}
            anomalies: vec![AgentAnomalyPayload {
                description: "node 42 has unusually high connectivity".into(),
                severity: 0.6,
                category: "unexpected_pattern".into(),
            }],
            energy_pattern: "declining".into(),
            energy_pattern_confidence: 0.85,
        };
        let bytes = cbor_encode(&payload).unwrap();
        let decoded: AgentReturnPayload = cbor_decode(&bytes).unwrap();
        assert_eq!(decoded.contract_id, "c-001");
        assert!(decoded.scope_completed);
        assert_eq!(decoded.anomalies.len(), 1);
        assert_eq!(decoded.energy_pattern, "declining");
    }

    #[test]
    fn cbor_roundtrip_agent_recall() {
        let payload = AgentRecallPayload {
            contract_id: "c-001".into(),
            reason: "energy budget exceeded".into(),
            energy_consumed_uj: 55_000_000,
            budget_exceeded: true,
            energy_pattern: "escalating".into(),
        };
        let bytes = cbor_encode(&payload).unwrap();
        let decoded: AgentRecallPayload = cbor_decode(&bytes).unwrap();
        assert_eq!(decoded.reason, "energy budget exceeded");
        assert!(decoded.budget_exceeded);
        assert_eq!(decoded.energy_pattern, "escalating");
    }
}
