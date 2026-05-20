use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio_util::codec::Framed;

use crate::codec::JwpCodec;
use crate::error::JwpError;
use crate::frame::{HEADER_LEN, JwpFrame};
use crate::transport::{Transport, TransportStats};

/// JWP transport over TCP using `tokio_util::codec::Framed`.
pub struct TcpTransport {
    framed: Framed<TcpStream, JwpCodec>,
    stats: TransportStats,
}

impl TcpTransport {
    /// Wrap a raw TCP stream with JWP framing.
    pub fn new(stream: TcpStream) -> Self {
        Self {
            framed: Framed::new(stream, JwpCodec::new()),
            stats: TransportStats::default(),
        }
    }

    /// Wrap an already-framed stream (for migration from existing code).
    pub fn from_framed(framed: Framed<TcpStream, JwpCodec>) -> Self {
        Self {
            framed,
            stats: TransportStats::default(),
        }
    }

    /// Access the inner `Framed` stream.
    pub fn inner(&self) -> &Framed<TcpStream, JwpCodec> {
        &self.framed
    }

    /// Mutable access to the inner `Framed` stream.
    pub fn inner_mut(&mut self) -> &mut Framed<TcpStream, JwpCodec> {
        &mut self.framed
    }
}

impl Transport for TcpTransport {
    async fn send_frame(&mut self, frame: JwpFrame) -> Result<(), JwpError> {
        let payload_len = frame.payload.len() as u64;
        self.framed.send(frame).await?;
        self.stats.bytes_sent += HEADER_LEN as u64 + payload_len;
        self.stats.frames_sent += 1;
        Ok(())
    }

    async fn recv_frame(&mut self) -> Result<Option<JwpFrame>, JwpError> {
        match self.framed.next().await {
            Some(Ok(frame)) => {
                self.stats.bytes_received += HEADER_LEN as u64 + frame.payload.len() as u64;
                self.stats.frames_received += 1;
                Ok(Some(frame))
            }
            Some(Err(e)) => Err(e),
            None => Ok(None),
        }
    }

    fn transport_id(&self) -> &str {
        "tcp"
    }

    fn stats(&self) -> &TransportStats {
        &self.stats
    }

    fn stats_mut(&mut self) -> &mut TransportStats {
        &mut self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{
        FrameFlags, FrameHeader, FrameType, HandshakePayload, PROTOCOL_VERSION, QueryPayload,
        cbor_encode,
    };
    use tokio::net::TcpListener;

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

    /// Create a connected (client, server) TcpTransport pair.
    async fn transport_pair() -> (TcpTransport, TcpTransport) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let (client_stream, server_stream) =
            tokio::join!(TcpStream::connect(addr), async { listener.accept().await });

        let client = TcpTransport::new(client_stream.unwrap());
        let server = TcpTransport::new(server_stream.unwrap().0);
        (client, server)
    }

    #[tokio::test]
    async fn tcp_transport_roundtrip() {
        let (mut client, mut server) = transport_pair().await;

        let handshake_payload = cbor_encode(&HandshakePayload {
            version: 1,
            capabilities: vec!["search".into()],
        })
        .unwrap();
        let frame = make_frame(FrameType::Handshake, 1, 0, handshake_payload.clone());

        client.send_frame(frame).await.unwrap();

        let received = server.recv_frame().await.unwrap().unwrap();
        assert_eq!(received.header.frame_type, FrameType::Handshake);
        assert_eq!(received.header.sequence, 1);
        assert_eq!(received.payload, handshake_payload);
    }

    #[tokio::test]
    async fn tcp_transport_stats_incremented() {
        let (mut client, mut server) = transport_pair().await;

        assert_eq!(client.stats().frames_sent, 0);
        assert_eq!(client.stats().bytes_sent, 0);
        assert_eq!(server.stats().frames_received, 0);

        // Send two frames
        let f1 = make_frame(FrameType::Heartbeat, 1, 0, vec![]);
        let query_payload = cbor_encode(&QueryPayload {
            query: "test".into(),
            limit: 10,
            session_id: None,
        })
        .unwrap();
        let f2 = make_frame(FrameType::Query, 2, 100, query_payload.clone());

        client.send_frame(f1).await.unwrap();
        client.send_frame(f2).await.unwrap();

        assert_eq!(client.stats().frames_sent, 2);
        assert_eq!(
            client.stats().bytes_sent,
            HEADER_LEN as u64 * 2 + query_payload.len() as u64
        );

        // Receive both
        server.recv_frame().await.unwrap().unwrap();
        server.recv_frame().await.unwrap().unwrap();

        assert_eq!(server.stats().frames_received, 2);
        assert_eq!(
            server.stats().bytes_received,
            HEADER_LEN as u64 * 2 + query_payload.len() as u64
        );
    }

    #[tokio::test]
    async fn tcp_transport_id() {
        let (client, _server) = transport_pair().await;
        assert_eq!(client.transport_id(), "tcp");
    }

    #[tokio::test]
    async fn tcp_transport_clean_shutdown() {
        let (client, mut server) = transport_pair().await;

        // Drop the client side to close the connection
        drop(client);

        // Server should see None (clean shutdown)
        let result = server.recv_frame().await.unwrap();
        assert!(result.is_none());
    }
}
