//! Pluggable payload encoding for JWP frames.
//!
//! The [`PayloadCodec`] trait abstracts payload serialization so the
//! protocol can swap encodings per-frame based on adaptive decisions.

use serde::{Serialize, de::DeserializeOwned};

use crate::error::JwpError;

// ── Encoding identifier ──────────────────────────────────────────

/// Identifies the payload encoding used in a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EncodingId {
    /// Canonical CBOR (RFC 8949) — the default and only v1 encoding.
    Cbor = 0x01,
    // Future: FlatBuf = 0x02, Raw = 0x03, MsgPack = 0x04
}

impl EncodingId {
    /// Parse from wire byte.
    pub fn from_u8(b: u8) -> Option<Self> {
        match b {
            0x01 => Some(Self::Cbor),
            _ => None,
        }
    }
}

// ── Trait ─────────────────────────────────────────────────────────

/// Trait for pluggable payload serialization.
///
/// Implementations must be `Send + Sync` so they can be stored in
/// codecs shared across async tasks.
pub trait PayloadCodec: Send + Sync {
    /// Serialize a value to wire bytes.
    fn encode_payload<T: Serialize>(&self, value: &T) -> Result<Vec<u8>, JwpError>;

    /// Deserialize wire bytes back to a value.
    fn decode_payload<T: DeserializeOwned>(&self, data: &[u8]) -> Result<T, JwpError>;

    /// The encoding identifier for this codec.
    fn encoding_id(&self) -> EncodingId;

    /// Estimated per-payload overhead in bytes (for adaptive decisions).
    ///
    /// This is the typical number of bytes that the encoding format
    /// adds beyond the raw field values. Used by the [`ConnectionProfile`]
    /// to decide when a lighter encoding would save energy.
    fn estimated_overhead_bytes(&self) -> usize;
}

// ── CBOR implementation ──────────────────────────────────────────

/// The default CBOR codec — wraps the existing `cbor_encode`/`cbor_decode`.
///
/// Uses canonical deterministic CBOR (RFC 8949) via `ciborium`.
#[derive(Debug, Clone, Copy)]
pub struct CborPayloadCodec;

impl PayloadCodec for CborPayloadCodec {
    fn encode_payload<T: Serialize>(&self, value: &T) -> Result<Vec<u8>, JwpError> {
        crate::frame::cbor_encode(value)
    }

    fn decode_payload<T: DeserializeOwned>(&self, data: &[u8]) -> Result<T, JwpError> {
        crate::frame::cbor_decode(data)
    }

    fn encoding_id(&self) -> EncodingId {
        EncodingId::Cbor
    }

    fn estimated_overhead_bytes(&self) -> usize {
        // CBOR adds ~15-30 bytes of type/length metadata for typical
        // search payloads (map keys, string length prefixes, etc.)
        20
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{HandshakePayload, QueryPayload, ResultPayload};

    #[test]
    fn cbor_roundtrip_query_via_trait() {
        let codec = CborPayloadCodec;
        let payload = QueryPayload {
            query: "rust async runtime".into(),
            limit: 25,
            session_id: Some("sess-42".into()),
        };

        let bytes = codec.encode_payload(&payload).unwrap();
        let decoded: QueryPayload = codec.decode_payload(&bytes).unwrap();

        assert_eq!(decoded.query, "rust async runtime");
        assert_eq!(decoded.limit, 25);
        assert_eq!(decoded.session_id.as_deref(), Some("sess-42"));
    }

    #[test]
    fn cbor_roundtrip_result_via_trait() {
        let codec = CborPayloadCodec;
        let payload = ResultPayload {
            rank: 3,
            url: "https://docs.rs/tokio".into(),
            title: "Tokio Docs".into(),
            domain: "docs.rs".into(),
            score: 8.5,
            content_hash: "blake3_abc".into(),
        };

        let bytes = codec.encode_payload(&payload).unwrap();
        let decoded: ResultPayload = codec.decode_payload(&bytes).unwrap();

        assert_eq!(decoded.rank, 3);
        assert_eq!(decoded.url, "https://docs.rs/tokio");
        assert_eq!(decoded.domain, "docs.rs");
    }

    #[test]
    fn cbor_roundtrip_handshake_via_trait() {
        let codec = CborPayloadCodec;
        let payload = HandshakePayload {
            version: 1,
            capabilities: vec!["search".into(), "cancel".into()],
        };

        let bytes = codec.encode_payload(&payload).unwrap();
        let decoded: HandshakePayload = codec.decode_payload(&bytes).unwrap();

        assert_eq!(decoded.version, 1);
        assert_eq!(decoded.capabilities, vec!["search", "cancel"]);
    }

    #[test]
    fn encoding_id_and_overhead() {
        let codec = CborPayloadCodec;
        assert_eq!(codec.encoding_id(), EncodingId::Cbor);
        assert!(codec.estimated_overhead_bytes() > 0);
        assert!(codec.estimated_overhead_bytes() < 100);
    }

    #[test]
    fn encoding_id_from_u8() {
        assert_eq!(EncodingId::from_u8(0x01), Some(EncodingId::Cbor));
        assert_eq!(EncodingId::from_u8(0x00), None);
        assert_eq!(EncodingId::from_u8(0xFF), None);
    }
}
