use bytes::{Buf, BufMut, BytesMut};
use tokio_util::codec::{Decoder, Encoder};

use crate::compression::{CompressionId, CompressionStrategy, strategy_for};
use crate::error::{JwpError, MAX_PAYLOAD_LEN};
use crate::frame::{
    COMPACT_HEADER_LEN, FrameFlags, FrameHeader, HEADER_LEN, HeaderFormat, JwpFrame,
    PROTOCOL_VERSION_V2, PROTOCOL_VERSION_V2_COMPACT, PROTOCOL_VERSION_V2_EXTENDED,
};
use crate::negotiation::NegotiatedCapabilities;
use crate::profile::ConnectionProfile;

/// Stateful v2 codec for JWP — makes per-frame decisions about header
/// format, compression, and encoding based on measured connection profile.
///
/// `AdaptiveCodec::v1_compat()` produces **byte-identical** output to
/// [`JwpCodec`](crate::JwpCodec). Existing v1 tests pass unchanged.
///
/// For v2 connections, the codec:
/// - Chooses compact headers for heartbeats/cancel (saves 62%)
/// - Compresses payloads when profile indicates benefit
/// - Uses extended headers for energy-rich frames
/// - Updates the connection profile after each frame
pub struct AdaptiveCodec {
    capabilities: NegotiatedCapabilities,
    profile: ConnectionProfile,
    compression: Box<dyn CompressionStrategy>,
    /// Last energy value from a standard/extended header, used for
    /// compact header inheritance.
    last_energy_uwh: u64,
}

impl AdaptiveCodec {
    /// Create a v2 codec from negotiated capabilities.
    pub fn new(capabilities: NegotiatedCapabilities) -> Self {
        let compression = strategy_for(capabilities.compression);
        Self {
            capabilities,
            profile: ConnectionProfile::new(),
            compression,
            last_energy_uwh: 0,
        }
    }

    /// Create a v1-compatible codec that produces byte-identical output
    /// to [`JwpCodec`](crate::JwpCodec).
    pub fn v1_compat() -> Self {
        Self::new(NegotiatedCapabilities::v1_default())
    }

    /// Read-only access to the connection profile.
    pub fn profile(&self) -> &ConnectionProfile {
        &self.profile
    }

    /// Mutable access to the connection profile.
    pub fn profile_mut(&mut self) -> &mut ConnectionProfile {
        &mut self.profile
    }

    /// Current negotiated capabilities.
    pub fn capabilities(&self) -> &NegotiatedCapabilities {
        &self.capabilities
    }

    /// Whether this codec operates in v2 adaptive mode.
    fn is_v2(&self) -> bool {
        self.capabilities.is_v2()
    }

    /// Choose header format for a frame type, respecting negotiated caps.
    fn choose_header_format(&self, frame_type: crate::frame::FrameType) -> HeaderFormat {
        if !self.is_v2() {
            return HeaderFormat::Standard;
        }
        match self.profile.header_for(frame_type) {
            HeaderFormat::Compact if !self.capabilities.compact_headers => HeaderFormat::Standard,
            HeaderFormat::Extended if !self.capabilities.extended_headers => HeaderFormat::Standard,
            other => other,
        }
    }
}

impl Decoder for AdaptiveCodec {
    type Item = JwpFrame;
    type Error = JwpError;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        if src.is_empty() {
            return Ok(None);
        }

        // Peek version byte to determine minimum header size
        let min_header = match src[0] {
            PROTOCOL_VERSION_V2_COMPACT => COMPACT_HEADER_LEN,
            _ => HEADER_LEN, // v1, v2 standard, v2 extended
        };

        if src.len() < min_header {
            src.reserve(min_header - src.len());
            return Ok(None);
        }

        // Decode header (auto-detect format)
        let (header, format, header_consumed) = match FrameHeader::decode_any(&src[..]) {
            Ok(result) => result,
            Err(JwpError::IncompleteFrame { needed, .. }) => {
                src.reserve(needed.saturating_sub(src.len()));
                return Ok(None);
            }
            Err(e) => return Err(e),
        };

        let payload_len = header.payload_length as usize;
        if payload_len > MAX_PAYLOAD_LEN {
            return Err(JwpError::PayloadTooLarge(payload_len));
        }

        let total_len = header_consumed + payload_len;
        if src.len() < total_len {
            src.reserve(total_len - src.len());
            return Ok(None);
        }

        // Consume header bytes
        src.advance(header_consumed);

        // Consume payload bytes
        let raw_payload = if payload_len > 0 {
            let p = src[..payload_len].to_vec();
            src.advance(payload_len);
            p
        } else {
            vec![]
        };

        // Decompress if COMPRESSED flag is set
        let payload = if header.flags.is_set(FrameFlags::COMPRESSED) {
            let comp_id = CompressionId::from_u8(header.flags.compression_id())
                .unwrap_or(CompressionId::None);
            let decompressor = strategy_for(comp_id);
            decompressor.decompress(&raw_payload, MAX_PAYLOAD_LEN)?
        } else {
            raw_payload
        };

        // Inherit energy from last standard/extended frame for compact headers
        let energy_uwh = if format == HeaderFormat::Compact {
            self.last_energy_uwh
        } else {
            self.last_energy_uwh = header.energy_uwh;
            header.energy_uwh
        };

        let frame_header = FrameHeader {
            energy_uwh,
            // Normalize payload_length to decompressed size
            payload_length: payload.len() as u32,
            ..header
        };

        self.profile
            .observe_frame(payload.len() as u64, energy_uwh, None);

        Ok(Some(JwpFrame {
            header: frame_header,
            payload,
        }))
    }
}

impl Encoder<JwpFrame> for AdaptiveCodec {
    type Error = JwpError;

    fn encode(&mut self, item: JwpFrame, dst: &mut BytesMut) -> Result<(), Self::Error> {
        if item.payload.len() > MAX_PAYLOAD_LEN {
            return Err(JwpError::PayloadTooLarge(item.payload.len()));
        }

        // ── v1 path: byte-identical to JwpCodec ─────────────────
        if !self.is_v2() {
            dst.reserve(HEADER_LEN + item.payload.len());
            let mut header_buf = [0u8; HEADER_LEN];
            item.header.encode(&mut header_buf);
            dst.put_slice(&header_buf);
            dst.put_slice(&item.payload);
            self.profile
                .observe_frame(item.payload.len() as u64, item.header.energy_uwh, None);
            return Ok(());
        }

        // ── v2 adaptive path ────────────────────────────────────
        let format = self.choose_header_format(item.header.frame_type);
        let original_payload_len = item.payload.len();
        let energy_uwh = item.header.energy_uwh;

        // Optionally compress the payload
        let should_compress =
            !item.payload.is_empty() && self.profile.should_compress(item.payload.len());

        let (wire_payload, flags) = if should_compress {
            let compressed = self.compression.compress(&item.payload)?;
            if compressed.len() < original_payload_len {
                let comp_id = self.compression.compression_id() as u8;
                let flags = item.header.flags.with_compression(comp_id);
                (compressed, flags)
            } else {
                // Compression didn't help — send uncompressed
                (item.payload, item.header.flags)
            }
        } else {
            (item.payload, item.header.flags)
        };

        match format {
            HeaderFormat::Compact => {
                // 8-byte compact header, no payload
                dst.reserve(COMPACT_HEADER_LEN);
                let compact_header = FrameHeader {
                    flags,
                    ..item.header
                };
                let mut buf = [0u8; COMPACT_HEADER_LEN];
                compact_header.encode_compact(&mut buf);
                dst.put_slice(&buf);
            }
            HeaderFormat::Standard => {
                let header = FrameHeader {
                    version: PROTOCOL_VERSION_V2,
                    payload_length: wire_payload.len() as u32,
                    flags,
                    ..item.header
                };
                dst.reserve(HEADER_LEN + wire_payload.len());
                let mut buf = [0u8; HEADER_LEN];
                header.encode(&mut buf);
                dst.put_slice(&buf);
                dst.put_slice(&wire_payload);
            }
            HeaderFormat::Extended => {
                // Extended: standard 21-byte header with version 0xC2.
                // Energy breakdown is prepended to the payload by the handler.
                let header = FrameHeader {
                    version: PROTOCOL_VERSION_V2_EXTENDED,
                    payload_length: wire_payload.len() as u32,
                    flags,
                    ..item.header
                };
                dst.reserve(HEADER_LEN + wire_payload.len());
                let mut buf = [0u8; HEADER_LEN];
                header.encode(&mut buf);
                dst.put_slice(&buf);
                dst.put_slice(&wire_payload);
            }
        }

        self.last_energy_uwh = energy_uwh;
        self.profile
            .observe_frame(original_payload_len as u64, energy_uwh, None);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::JwpCodec;
    use crate::frame::{FrameType, PROTOCOL_VERSION};

    fn make_v1_frame(frame_type: FrameType, seq: u32, energy: u64, payload: Vec<u8>) -> JwpFrame {
        JwpFrame {
            header: FrameHeader {
                version: PROTOCOL_VERSION,
                frame_type,
                payload_length: payload.len() as u32,
                energy_uwh: energy,
                sequence: seq,
                flags: FrameFlags::new(),
            },
            payload,
        }
    }

    #[test]
    fn v1_compat_identical_to_jwp_codec() {
        let mut jwp = JwpCodec::new();
        let mut adaptive = AdaptiveCodec::v1_compat();

        let frames = vec![
            make_v1_frame(FrameType::Heartbeat, 1, 0, vec![]),
            make_v1_frame(FrameType::Query, 2, 100, vec![0xDE, 0xAD]),
            make_v1_frame(FrameType::Result, 3, 200, vec![0xBE, 0xEF, 0xCA, 0xFE]),
        ];

        for frame in frames {
            let mut jwp_buf = BytesMut::new();
            let mut adaptive_buf = BytesMut::new();

            jwp.encode(frame.clone(), &mut jwp_buf).unwrap();
            adaptive.encode(frame, &mut adaptive_buf).unwrap();

            assert_eq!(
                jwp_buf, adaptive_buf,
                "v1_compat output must be byte-identical to JwpCodec"
            );
        }
    }

    #[test]
    fn v1_compat_decode_identical() {
        let mut jwp = JwpCodec::new();
        let mut adaptive = AdaptiveCodec::v1_compat();

        let frame = make_v1_frame(FrameType::Query, 5, 500, vec![0x01, 0x02, 0x03]);

        // Encode with JwpCodec
        let mut buf = BytesMut::new();
        jwp.encode(frame, &mut buf).unwrap();

        // Decode with both
        let mut jwp_buf = buf.clone();
        let mut adaptive_buf = buf;

        let jwp_frame = jwp.decode(&mut jwp_buf).unwrap().unwrap();
        let adaptive_frame = adaptive.decode(&mut adaptive_buf).unwrap().unwrap();

        assert_eq!(jwp_frame.header, adaptive_frame.header);
        assert_eq!(jwp_frame.payload, adaptive_frame.payload);
    }

    #[test]
    fn v2_compact_header_for_heartbeat() {
        let mut caps = NegotiatedCapabilities::v1_default();
        caps.protocol_version = 2;
        caps.compact_headers = true;

        let mut codec = AdaptiveCodec::new(caps);

        let frame = JwpFrame {
            header: FrameHeader {
                version: PROTOCOL_VERSION_V2,
                frame_type: FrameType::Heartbeat,
                payload_length: 0,
                energy_uwh: 0,
                sequence: 1,
                flags: FrameFlags::new(),
            },
            payload: vec![],
        };

        let mut buf = BytesMut::new();
        codec.encode(frame, &mut buf).unwrap();

        // Should use compact header (8 bytes)
        assert_eq!(buf.len(), COMPACT_HEADER_LEN);
        assert_eq!(buf[0], PROTOCOL_VERSION_V2_COMPACT);
    }

    #[test]
    fn v2_standard_header_when_compact_not_negotiated() {
        let mut caps = NegotiatedCapabilities::v1_default();
        caps.protocol_version = 2;
        caps.compact_headers = false; // not negotiated

        let mut codec = AdaptiveCodec::new(caps);

        let frame = JwpFrame {
            header: FrameHeader {
                version: PROTOCOL_VERSION_V2,
                frame_type: FrameType::Heartbeat,
                payload_length: 0,
                energy_uwh: 0,
                sequence: 1,
                flags: FrameFlags::new(),
            },
            payload: vec![],
        };

        let mut buf = BytesMut::new();
        codec.encode(frame, &mut buf).unwrap();

        // Should fall back to standard header (21 bytes)
        assert_eq!(buf.len(), HEADER_LEN);
        assert_eq!(buf[0], PROTOCOL_VERSION_V2);
    }

    #[test]
    fn v2_compact_header_roundtrip() {
        let mut caps = NegotiatedCapabilities::v1_default();
        caps.protocol_version = 2;
        caps.compact_headers = true;

        let mut encoder = AdaptiveCodec::new(caps.clone());
        let mut decoder = AdaptiveCodec::new(caps);

        // Send a standard frame first to establish energy baseline
        let setup = JwpFrame {
            header: FrameHeader {
                version: PROTOCOL_VERSION_V2,
                frame_type: FrameType::Query,
                payload_length: 4,
                energy_uwh: 500,
                sequence: 1,
                flags: FrameFlags::new(),
            },
            payload: vec![0x01, 0x02, 0x03, 0x04],
        };

        let mut buf = BytesMut::new();
        encoder.encode(setup, &mut buf).unwrap();
        decoder.decode(&mut buf).unwrap().unwrap();

        // Now send a compact heartbeat
        let heartbeat = JwpFrame {
            header: FrameHeader {
                version: PROTOCOL_VERSION_V2,
                frame_type: FrameType::Heartbeat,
                payload_length: 0,
                energy_uwh: 500,
                sequence: 2,
                flags: FrameFlags::new(),
            },
            payload: vec![],
        };

        encoder.encode(heartbeat, &mut buf).unwrap();
        let decoded = decoder.decode(&mut buf).unwrap().unwrap();

        assert_eq!(decoded.header.frame_type, FrameType::Heartbeat);
        assert_eq!(decoded.header.sequence, 2);
        // Energy inherited from last standard frame
        assert_eq!(decoded.header.energy_uwh, 500);
    }

    #[test]
    fn v2_extended_header_for_energy_gradient() {
        let mut caps = NegotiatedCapabilities::v1_default();
        caps.protocol_version = 2;
        caps.extended_headers = true;

        let mut codec = AdaptiveCodec::new(caps);

        let frame = JwpFrame {
            header: FrameHeader {
                version: PROTOCOL_VERSION_V2,
                frame_type: FrameType::EnergyGradient,
                payload_length: 10,
                energy_uwh: 1000,
                sequence: 5,
                flags: FrameFlags::new(),
            },
            payload: vec![0x01; 10],
        };

        let mut buf = BytesMut::new();
        codec.encode(frame, &mut buf).unwrap();

        // Should use extended header (version byte 0xC2)
        assert_eq!(buf[0], PROTOCOL_VERSION_V2_EXTENDED);
    }

    #[test]
    fn profile_updates_after_decode() {
        let mut codec = AdaptiveCodec::v1_compat();
        assert_eq!(codec.profile().frames_exchanged, 0);

        let frame = make_v1_frame(FrameType::Query, 1, 100, vec![0x01, 0x02]);

        let mut buf = BytesMut::new();
        let mut enc = JwpCodec::new();
        enc.encode(frame, &mut buf).unwrap();

        codec.decode(&mut buf).unwrap().unwrap();
        assert_eq!(codec.profile().frames_exchanged, 1);
        assert_eq!(codec.profile().cumulative_energy_uwh, 100);
    }

    #[test]
    fn profile_updates_after_encode() {
        let mut codec = AdaptiveCodec::v1_compat();
        assert_eq!(codec.profile().frames_exchanged, 0);

        let frame = make_v1_frame(FrameType::Result, 1, 200, vec![0xFF; 128]);

        let mut buf = BytesMut::new();
        codec.encode(frame, &mut buf).unwrap();
        assert_eq!(codec.profile().frames_exchanged, 1);
        assert_eq!(codec.profile().cumulative_energy_uwh, 200);
    }
}
