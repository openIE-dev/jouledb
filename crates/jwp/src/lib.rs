//! # Joule Wire Protocol (JWP)
//!
//! A custom binary framing protocol where every frame header carries
//! cumulative energy cost. Transport-agnostic — works over TCP, QUIC
//! streams, or Unix sockets.
//!
//! ## Frame format (21-byte header + CBOR payload)
//!
//! ```text
//! Offset  Size  Field
//!   0       1   version          (0x01)
//!   1       1   frame_type
//!   2       4   payload_length   (u32 big-endian, max 16 MiB)
//!   6       8   energy_uwh       (u64 big-endian, cumulative µWh)
//!  14       4   sequence         (u32 big-endian, monotonic)
//!  18       3   flags            (24-bit bitfield)
//!  21       N   payload          (CBOR-encoded, canonical RFC 8949)
//! ```

pub mod adaptive_codec;
pub mod codec;
pub mod compression;
pub mod encoding;
pub mod error;
pub mod frame;
pub mod negotiation;
pub mod profile;
pub mod state_machine;
pub mod transport;
#[cfg(feature = "quic")]
pub mod transport_quic;
pub mod transport_tcp;

pub use adaptive_codec::AdaptiveCodec;
pub use codec::JwpCodec;
pub use compression::{
    CompressionId, CompressionStrategy, Lz4Compression, NoCompression, ZstdCompression,
};
pub use encoding::{CborPayloadCodec, EncodingId, PayloadCodec};
pub use error::JwpError;
pub use frame::{
    BalanceQueryPayload, BalanceResponsePayload, BatchPayload, COMPACT_HEADER_LEN, CommandPayload,
    CommandResponsePayload, DonePayload, EnergyBreakdown, EnergyGradientPayload, ErrorPayload,
    FrameFlags, FrameHeader, FrameType, HEADER_LEN, HandshakePayload, HandshakeV2Payload,
    HeaderFormat, JwpFrame, MetaPayload, NegotiatePayload, PROTOCOL_VERSION, PROTOCOL_VERSION_V2,
    PROTOCOL_VERSION_V2_COMPACT, PROTOCOL_VERSION_V2_EXTENDED, ProfileUpdatePayload, QueryPayload,
    RateLimitPayload, ReceiptPayload, ResultPayload, StreamChunkPayload, TopupBeginPayload,
    TopupResponsePayload, UsageEntryPayload, UsageQueryPayload, UsageResponsePayload, cbor_decode,
    cbor_encode,
};
pub use negotiation::NegotiatedCapabilities;
pub use profile::{ConnectionProfile, EnergyReporting, RateLimitInfo};
pub use state_machine::{ProtocolState, ProtocolStateMachine};
pub use transport::{Transport, TransportStats};
#[cfg(feature = "quic")]
pub use transport_quic::QuicTransport;
pub use transport_tcp::TcpTransport;
