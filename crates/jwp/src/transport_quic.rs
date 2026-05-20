use crate::error::JwpError;
use crate::frame::{FrameHeader, HEADER_LEN, JwpFrame};
use crate::transport::{Transport, TransportStats};

/// JWP transport over QUIC bidirectional streams (quinn).
///
/// Wraps a `quinn::SendStream` + `quinn::RecvStream` pair with JWP
/// frame I/O. Each QUIC bidirectional stream maps to one JWP session.
pub struct QuicTransport {
    send: quinn::SendStream,
    recv: quinn::RecvStream,
    stats: TransportStats,
}

impl QuicTransport {
    /// Create a transport from a QUIC bidirectional stream pair.
    pub fn new(send: quinn::SendStream, recv: quinn::RecvStream) -> Self {
        Self {
            send,
            recv,
            stats: TransportStats::default(),
        }
    }

    /// Access the underlying send stream.
    pub fn send_stream(&self) -> &quinn::SendStream {
        &self.send
    }

    /// Access the underlying recv stream.
    pub fn recv_stream(&self) -> &quinn::RecvStream {
        &self.recv
    }
}

impl Transport for QuicTransport {
    async fn send_frame(&mut self, frame: JwpFrame) -> Result<(), JwpError> {
        let mut header_buf = [0u8; HEADER_LEN];
        frame.header.encode(&mut header_buf);

        self.send
            .write_all(&header_buf)
            .await
            .map_err(|e| JwpError::Io(std::io::Error::other(e.to_string())))?;

        if !frame.payload.is_empty() {
            self.send
                .write_all(&frame.payload)
                .await
                .map_err(|e| JwpError::Io(std::io::Error::other(e.to_string())))?;
        }

        self.stats.bytes_sent += HEADER_LEN as u64 + frame.payload.len() as u64;
        self.stats.frames_sent += 1;
        Ok(())
    }

    async fn recv_frame(&mut self) -> Result<Option<JwpFrame>, JwpError> {
        let mut header_buf = [0u8; HEADER_LEN];
        match self.recv.read_exact(&mut header_buf).await {
            Ok(()) => {}
            Err(quinn::ReadExactError::FinishedEarly(_)) => return Ok(None),
            Err(quinn::ReadExactError::ReadError(e)) => {
                return Err(JwpError::Io(std::io::Error::other(e.to_string())));
            }
        }

        let header = FrameHeader::decode(&header_buf)?;
        let payload_len = header.payload_length as usize;

        let mut payload = vec![0u8; payload_len];
        if payload_len > 0 {
            match self.recv.read_exact(&mut payload).await {
                Ok(()) => {}
                Err(quinn::ReadExactError::FinishedEarly(n)) => {
                    return Err(JwpError::IncompleteFrame {
                        needed: payload_len,
                        available: n,
                    });
                }
                Err(quinn::ReadExactError::ReadError(e)) => {
                    return Err(JwpError::Io(std::io::Error::other(e.to_string())));
                }
            }
        }

        self.stats.bytes_received += HEADER_LEN as u64 + payload_len as u64;
        self.stats.frames_received += 1;

        Ok(Some(JwpFrame { header, payload }))
    }

    fn transport_id(&self) -> &str {
        "quic"
    }

    fn stats(&self) -> &TransportStats {
        &self.stats
    }

    fn stats_mut(&mut self) -> &mut TransportStats {
        &mut self.stats
    }
}
