use crate::compression::CompressionId;
use crate::encoding::EncodingId;
use crate::frame::HandshakeV2Payload;
use crate::profile::EnergyReporting;

/// The result of handshake negotiation — capabilities agreed upon by
/// both client and server for the duration of the connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NegotiatedCapabilities {
    /// Protocol version (1 or 2).
    pub protocol_version: u8,
    /// Agreed-upon payload encoding.
    pub encoding: EncodingId,
    /// Agreed-upon compression strategy.
    pub compression: CompressionId,
    /// Maximum results per Batch frame.
    pub max_batch_size: u16,
    /// Energy reporting granularity.
    pub energy_reporting: EnergyReporting,
    /// Whether compact headers are allowed.
    pub compact_headers: bool,
    /// Whether extended headers are allowed.
    pub extended_headers: bool,
}

impl NegotiatedCapabilities {
    /// Default v1 capabilities — no adaptive features.
    ///
    /// Used when the client sends a v1 handshake (version byte `0x01`).
    /// All behavior matches the original static protocol exactly.
    pub fn v1_default() -> Self {
        Self {
            protocol_version: 1,
            encoding: EncodingId::Cbor,
            compression: CompressionId::None,
            max_batch_size: 1,
            energy_reporting: EnergyReporting::PerFrame,
            compact_headers: false,
            extended_headers: false,
        }
    }

    /// Negotiate v2 capabilities from client and server handshakes.
    ///
    /// For each capability, selects the intersection of what both sides
    /// support. When multiple options exist, prefers the most
    /// energy-efficient choice.
    ///
    /// # Negotiation rules
    ///
    /// - **Encoding**: Intersection of supported encodings. Prefers CBOR
    ///   (most tested). Falls back to CBOR if no overlap.
    /// - **Compression**: Intersection of supported compressions. Prefers
    ///   Zstd (best ratio), then Lz4 (lowest latency), then None.
    /// - **Batch size**: Minimum of both sides' max_batch_size.
    /// - **Energy reporting**: Most detailed level both support (PerFrame > PerQuery > PerSession).
    /// - **Headers**: Compact enabled if both support it. Extended likewise.
    pub fn negotiate(client: &HandshakeV2Payload, server: &HandshakeV2Payload) -> Self {
        // ── Encoding ─────────────────────────────────────────────
        let encoding = negotiate_encoding(&client.supported_encodings, &server.supported_encodings);

        // ── Compression ──────────────────────────────────────────
        let compression = negotiate_compression(
            &client.supported_compressions,
            &server.supported_compressions,
        );

        // ── Batch size: minimum of both ──────────────────────────
        let max_batch_size = client.max_batch_size.min(server.max_batch_size);

        // ── Energy reporting: most detailed common level ─────────
        let energy_reporting =
            negotiate_energy_reporting(client.energy_reporting, server.energy_reporting);

        // ── Header formats: both must support ────────────────────
        let client_headers: Vec<u8> = client.supported_headers.clone();
        let server_headers: Vec<u8> = server.supported_headers.clone();

        let compact_headers = client_headers.contains(&1) && server_headers.contains(&1);
        let extended_headers = client_headers.contains(&2) && server_headers.contains(&2);

        Self {
            protocol_version: 2,
            encoding,
            compression,
            max_batch_size,
            energy_reporting,
            compact_headers,
            extended_headers,
        }
    }

    /// Whether this connection has v2 adaptive features.
    pub fn is_v2(&self) -> bool {
        self.protocol_version >= 2
    }
}

/// Pick the best common encoding. Prefers CBOR (0x01) as most tested.
fn negotiate_encoding(client: &[u8], server: &[u8]) -> EncodingId {
    // Preference order: CBOR first
    let preference = [EncodingId::Cbor];

    for enc in &preference {
        let id = *enc as u8;
        if client.contains(&id) && server.contains(&id) {
            return *enc;
        }
    }

    // Fallback: always CBOR
    EncodingId::Cbor
}

/// Pick the best common compression. Prefers Zstd (best ratio),
/// then Lz4 (lowest latency), then None.
fn negotiate_compression(client: &[u8], server: &[u8]) -> CompressionId {
    let preference = [CompressionId::Zstd, CompressionId::Lz4, CompressionId::None];

    for comp in &preference {
        let id = *comp as u8;
        if client.contains(&id) && server.contains(&id) {
            return *comp;
        }
    }

    CompressionId::None
}

/// Pick the most detailed common energy reporting level.
/// PerFrame (0) > PerQuery (1) > PerSession (2).
fn negotiate_energy_reporting(client: u8, server: u8) -> EnergyReporting {
    // Lower value = more detailed. Pick the max (least detailed that both support).
    let level = client.max(server);
    EnergyReporting::from_u8(level).unwrap_or(EnergyReporting::PerFrame)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_v2_handshake() -> HandshakeV2Payload {
        HandshakeV2Payload {
            version: 2,
            capabilities: vec!["search".into()],
            supported_encodings: vec![0x01],                // CBOR
            supported_compressions: vec![0x00, 0x01, 0x02], // None, Zstd, Lz4
            max_batch_size: 16,
            energy_reporting: 0,              // PerFrame
            supported_headers: vec![0, 1, 2], // Standard, Compact, Extended
            credential: None,
        }
    }

    #[test]
    fn v1_default_no_adaptive_features() {
        let cap = NegotiatedCapabilities::v1_default();
        assert_eq!(cap.protocol_version, 1);
        assert_eq!(cap.encoding, EncodingId::Cbor);
        assert_eq!(cap.compression, CompressionId::None);
        assert_eq!(cap.max_batch_size, 1);
        assert!(!cap.compact_headers);
        assert!(!cap.extended_headers);
        assert!(!cap.is_v2());
    }

    #[test]
    fn v2_full_capabilities_negotiation() {
        let client = full_v2_handshake();
        let server = full_v2_handshake();

        let cap = NegotiatedCapabilities::negotiate(&client, &server);
        assert_eq!(cap.protocol_version, 2);
        assert_eq!(cap.encoding, EncodingId::Cbor);
        // Prefers Zstd (best ratio)
        assert_eq!(cap.compression, CompressionId::Zstd);
        assert_eq!(cap.max_batch_size, 16);
        assert_eq!(cap.energy_reporting, EnergyReporting::PerFrame);
        assert!(cap.compact_headers);
        assert!(cap.extended_headers);
        assert!(cap.is_v2());
    }

    #[test]
    fn mismatched_compression_falls_back() {
        let mut client = full_v2_handshake();
        let mut server = full_v2_handshake();

        // Client only supports Zstd, server only supports Lz4
        client.supported_compressions = vec![0x01]; // Zstd only
        server.supported_compressions = vec![0x02]; // Lz4 only

        let cap = NegotiatedCapabilities::negotiate(&client, &server);
        // No overlap → falls back to None
        assert_eq!(cap.compression, CompressionId::None);
    }

    #[test]
    fn batch_size_uses_minimum() {
        let mut client = full_v2_handshake();
        let mut server = full_v2_handshake();

        client.max_batch_size = 8;
        server.max_batch_size = 16;

        let cap = NegotiatedCapabilities::negotiate(&client, &server);
        assert_eq!(cap.max_batch_size, 8);
    }

    #[test]
    fn energy_reporting_most_detailed_common() {
        let client = full_v2_handshake(); // energy_reporting = 0 (PerFrame)
        let mut server = full_v2_handshake();
        server.energy_reporting = 1; // PerQuery

        let cap = NegotiatedCapabilities::negotiate(&client, &server);
        // Takes max (least detailed both support): PerQuery
        assert_eq!(cap.energy_reporting, EnergyReporting::PerQuery);
    }

    #[test]
    fn compact_headers_require_both_sides() {
        let client = full_v2_handshake();
        let mut server = full_v2_handshake();

        // Server doesn't support compact
        server.supported_headers = vec![0, 2]; // Standard + Extended, no Compact

        let cap = NegotiatedCapabilities::negotiate(&client, &server);
        assert!(!cap.compact_headers);
        assert!(cap.extended_headers);
    }
}
