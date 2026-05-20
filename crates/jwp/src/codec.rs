use bytes::{Buf, BufMut, BytesMut};
use tokio_util::codec::{Decoder, Encoder};

use crate::error::{JwpError, MAX_PAYLOAD_LEN};
use crate::frame::{FrameHeader, HEADER_LEN, JwpFrame};

/// tokio-util codec for JWP frames.
///
/// Reads/writes the 21-byte header followed by the variable-length payload.
pub struct JwpCodec;

impl JwpCodec {
    pub fn new() -> Self {
        Self
    }
}

impl Default for JwpCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder for JwpCodec {
    type Item = JwpFrame;
    type Error = JwpError;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        // Need at least the header
        if src.len() < HEADER_LEN {
            return Ok(None);
        }

        // Peek at header to get payload length (don't consume yet)
        let header_bytes: [u8; HEADER_LEN] = src[..HEADER_LEN]
            .try_into()
            .expect("slice is exactly HEADER_LEN");
        let header = FrameHeader::decode(&header_bytes)?;

        let payload_len = header.payload_length as usize;
        if payload_len > MAX_PAYLOAD_LEN {
            return Err(JwpError::PayloadTooLarge(payload_len));
        }

        let total_len = HEADER_LEN + payload_len;
        if src.len() < total_len {
            // Reserve space for the full frame so tokio reads enough
            src.reserve(total_len - src.len());
            return Ok(None);
        }

        // Consume header
        src.advance(HEADER_LEN);

        // Consume payload
        let payload = src[..payload_len].to_vec();
        src.advance(payload_len);

        Ok(Some(JwpFrame { header, payload }))
    }
}

impl Encoder<JwpFrame> for JwpCodec {
    type Error = JwpError;

    fn encode(&mut self, item: JwpFrame, dst: &mut BytesMut) -> Result<(), Self::Error> {
        if item.payload.len() > MAX_PAYLOAD_LEN {
            return Err(JwpError::PayloadTooLarge(item.payload.len()));
        }

        dst.reserve(HEADER_LEN + item.payload.len());

        let mut header_buf = [0u8; HEADER_LEN];
        item.header.encode(&mut header_buf);
        dst.put_slice(&header_buf);
        dst.put_slice(&item.payload);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{FrameFlags, FrameType, PROTOCOL_VERSION};

    fn make_frame(frame_type: FrameType, seq: u32, energy: u64, payload: Vec<u8>) -> JwpFrame {
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
    fn codec_roundtrip_empty_payload() {
        let mut codec = JwpCodec::new();
        let frame = make_frame(FrameType::Heartbeat, 1, 0, vec![]);

        let mut buf = BytesMut::new();
        codec.encode(frame.clone(), &mut buf).unwrap();

        let decoded = codec.decode(&mut buf).unwrap().unwrap();
        assert_eq!(decoded.header.frame_type, FrameType::Heartbeat);
        assert_eq!(decoded.header.sequence, 1);
        assert!(decoded.payload.is_empty());
    }

    #[test]
    fn codec_roundtrip_with_payload() {
        let mut codec = JwpCodec::new();
        let payload = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let frame = make_frame(FrameType::Result, 42, 1500, payload.clone());

        let mut buf = BytesMut::new();
        codec.encode(frame, &mut buf).unwrap();

        let decoded = codec.decode(&mut buf).unwrap().unwrap();
        assert_eq!(decoded.header.frame_type, FrameType::Result);
        assert_eq!(decoded.header.sequence, 42);
        assert_eq!(decoded.header.energy_uwh, 1500);
        assert_eq!(decoded.payload, payload);
    }

    #[test]
    fn codec_handles_partial_reads() {
        let mut codec = JwpCodec::new();
        let payload = b"hello world".to_vec();
        let frame = make_frame(FrameType::Query, 1, 100, payload);

        let mut buf = BytesMut::new();
        codec.encode(frame, &mut buf).unwrap();

        // Split into two halves
        let full = buf.split();
        let half1 = &full[..10];
        let half2 = &full[10..];

        let mut recv_buf = BytesMut::new();
        recv_buf.extend_from_slice(half1);

        // First decode: not enough data
        assert!(codec.decode(&mut recv_buf).unwrap().is_none());

        // Feed the rest
        recv_buf.extend_from_slice(half2);
        let decoded = codec.decode(&mut recv_buf).unwrap().unwrap();
        assert_eq!(decoded.header.frame_type, FrameType::Query);
        assert_eq!(decoded.payload, b"hello world");
    }

    #[test]
    fn codec_multiple_frames() {
        let mut codec = JwpCodec::new();
        let mut buf = BytesMut::new();

        // Encode two frames back-to-back
        let f1 = make_frame(FrameType::Meta, 1, 100, vec![0x01]);
        let f2 = make_frame(FrameType::Result, 2, 200, vec![0x02, 0x03]);
        codec.encode(f1, &mut buf).unwrap();
        codec.encode(f2, &mut buf).unwrap();

        // Decode both
        let d1 = codec.decode(&mut buf).unwrap().unwrap();
        assert_eq!(d1.header.sequence, 1);
        assert_eq!(d1.payload, vec![0x01]);

        let d2 = codec.decode(&mut buf).unwrap().unwrap();
        assert_eq!(d2.header.sequence, 2);
        assert_eq!(d2.payload, vec![0x02, 0x03]);

        // No more frames
        assert!(codec.decode(&mut buf).unwrap().is_none());
    }

    #[test]
    fn codec_rejects_oversized_payload() {
        let mut codec = JwpCodec::new();
        let payload = vec![0u8; MAX_PAYLOAD_LEN + 1];
        let frame = make_frame(FrameType::Result, 1, 0, payload);

        let mut buf = BytesMut::new();
        assert!(codec.encode(frame, &mut buf).is_err());
    }
}
