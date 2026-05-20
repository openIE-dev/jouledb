//! TCP-based Raft transport for networked consensus.
//!
//! HRP Phase 1.3: Dual wire format with automatic detection.
//!
//! **Binary (HRP)**: `[4-byte magic 0x48525001][4-byte payload_len][bincode payload][4-byte CRC32]`
//! **JSON (legacy)**: `[4-byte big-endian length][JSON payload]`
//!
//! The reader checks the first 4 bytes: if they match the HRP magic number,
//! the binary path is used; otherwise the JSON path handles them as a length prefix.
//! This enables rolling upgrades where old and new nodes can communicate.

use std::collections::HashMap;
use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, ReadBuf};
use tokio::net::TcpStream;
use tokio::sync::{Mutex, RwLock};

#[cfg(feature = "tls")]
use tokio_rustls::{TlsConnector, client as tls_client, server as tls_server};

use crate::raft::{
    AppendEntriesRequest, AppendEntriesResponse, InstallSnapshotRequest, InstallSnapshotResponse,
    NodeId, RaftError, RaftMessage, RaftTransport, RequestVoteRequest, RequestVoteResponse,
};

// ============================================================================
// Stream Abstraction
// ============================================================================

/// A Raft transport stream that may be plain TCP or TLS-wrapped.
///
/// TLS variants are only available when compiled with `--features tls`.
/// All inner types implement `Unpin`, so pin-projection is trivially safe.
pub enum RaftStream {
    /// Plain TCP (no encryption).
    Plain(TcpStream),
    /// Client-side TLS (outgoing connections to peers).
    #[cfg(feature = "tls")]
    TlsClient(tls_client::TlsStream<TcpStream>),
    /// Server-side TLS (incoming connections from peers).
    #[cfg(feature = "tls")]
    TlsServer(tls_server::TlsStream<TcpStream>),
}

impl AsyncRead for RaftStream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        match self.get_mut() {
            RaftStream::Plain(s) => Pin::new(s).poll_read(cx, buf),
            #[cfg(feature = "tls")]
            RaftStream::TlsClient(s) => Pin::new(s).poll_read(cx, buf),
            #[cfg(feature = "tls")]
            RaftStream::TlsServer(s) => Pin::new(s).poll_read(cx, buf),
        }
    }
}

impl AsyncWrite for RaftStream {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        match self.get_mut() {
            RaftStream::Plain(s) => Pin::new(s).poll_write(cx, buf),
            #[cfg(feature = "tls")]
            RaftStream::TlsClient(s) => Pin::new(s).poll_write(cx, buf),
            #[cfg(feature = "tls")]
            RaftStream::TlsServer(s) => Pin::new(s).poll_write(cx, buf),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.get_mut() {
            RaftStream::Plain(s) => Pin::new(s).poll_flush(cx),
            #[cfg(feature = "tls")]
            RaftStream::TlsClient(s) => Pin::new(s).poll_flush(cx),
            #[cfg(feature = "tls")]
            RaftStream::TlsServer(s) => Pin::new(s).poll_flush(cx),
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.get_mut() {
            RaftStream::Plain(s) => Pin::new(s).poll_shutdown(cx),
            #[cfg(feature = "tls")]
            RaftStream::TlsClient(s) => Pin::new(s).poll_shutdown(cx),
            #[cfg(feature = "tls")]
            RaftStream::TlsServer(s) => Pin::new(s).poll_shutdown(cx),
        }
    }
}

// ============================================================================
// Wire Format
// ============================================================================

/// Envelope wrapping every Raft RPC message on the wire.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftEnvelope {
    /// Sender's node ID
    pub from: NodeId,
    /// The Raft message (request or response)
    pub msg: RaftMessage,
}

/// Maximum message size (16 MB) — protects against malformed length prefixes.
const MAX_MESSAGE_SIZE: u32 = 16 * 1024 * 1024;

/// Write a length-prefixed JSON envelope to any async stream.
pub async fn write_envelope<S: AsyncWrite + Unpin>(
    stream: &mut S,
    envelope: &RaftEnvelope,
) -> Result<(), RaftError> {
    let json = serde_json::to_vec(envelope)
        .map_err(|e| RaftError::Internal(format!("serialize: {}", e)))?;
    let len = json.len() as u32;
    stream
        .write_all(&len.to_be_bytes())
        .await
        .map_err(|e| RaftError::Internal(format!("write len: {}", e)))?;
    stream
        .write_all(&json)
        .await
        .map_err(|e| RaftError::Internal(format!("write payload: {}", e)))?;
    stream
        .flush()
        .await
        .map_err(|e| RaftError::Internal(format!("flush: {}", e)))?;
    Ok(())
}

/// Read a length-prefixed JSON envelope from any async stream.
pub async fn read_envelope<S: AsyncRead + Unpin>(
    stream: &mut S,
) -> Result<RaftEnvelope, RaftError> {
    // Read first 4 bytes — could be HRP magic or JSON length prefix
    let mut header = [0u8; 4];
    stream
        .read_exact(&mut header)
        .await
        .map_err(|e| RaftError::Internal(format!("read header: {}", e)))?;

    if header == HRP_MAGIC {
        // HRP binary format: read payload_len, bincode payload, CRC32
        return read_hrp_payload(stream).await;
    }

    // Legacy JSON format: header is the length prefix
    let len = u32::from_be_bytes(header);
    if len > MAX_MESSAGE_SIZE {
        return Err(RaftError::Internal(format!(
            "message too large: {} bytes",
            len
        )));
    }
    let mut buf = vec![0u8; len as usize];
    stream
        .read_exact(&mut buf)
        .await
        .map_err(|e| RaftError::Internal(format!("read payload: {}", e)))?;
    serde_json::from_slice(&buf).map_err(|e| RaftError::Internal(format!("deserialize: {}", e)))
}

// ============================================================================
// HRP Binary Wire Format (Phase 1.3)
// ============================================================================

/// HRP magic number: "HRP\x01" — identifies a binary-encoded envelope.
const HRP_MAGIC: [u8; 4] = [0x48, 0x52, 0x50, 0x01];

/// Write an HRP binary envelope: `[magic][payload_len][bincode payload][CRC32]`
///
/// 2-4x smaller than JSON and ~10x faster to serialize/deserialize.
pub async fn write_hrp_envelope<S: AsyncWrite + Unpin>(
    stream: &mut S,
    envelope: &RaftEnvelope,
) -> Result<(), RaftError> {
    let bincode_data = bincode::serde::encode_to_vec(envelope, bincode::config::standard())
        .map_err(|e| RaftError::Internal(format!("hrp bincode serialize: {}", e)))?;

    let payload_len = bincode_data.len() as u32;
    if payload_len > MAX_MESSAGE_SIZE {
        return Err(RaftError::Internal(format!(
            "hrp message too large: {} bytes",
            payload_len
        )));
    }

    // Build the frame: magic + len + payload + CRC32
    let frame_size = 4 + 4 + bincode_data.len() + 4;
    let mut frame = Vec::with_capacity(frame_size);
    frame.extend_from_slice(&HRP_MAGIC);
    frame.extend_from_slice(&payload_len.to_be_bytes());
    frame.extend_from_slice(&bincode_data);

    // CRC32 over magic + len + payload
    let crc = crc32fast::hash(&frame);
    frame.extend_from_slice(&crc.to_be_bytes());

    stream
        .write_all(&frame)
        .await
        .map_err(|e| RaftError::Internal(format!("hrp write: {}", e)))?;
    stream
        .flush()
        .await
        .map_err(|e| RaftError::Internal(format!("hrp flush: {}", e)))?;
    Ok(())
}

/// Read the HRP binary payload after the magic has already been consumed.
async fn read_hrp_payload<S: AsyncRead + Unpin>(stream: &mut S) -> Result<RaftEnvelope, RaftError> {
    // Read payload length (4 bytes, big-endian)
    let mut len_buf = [0u8; 4];
    stream
        .read_exact(&mut len_buf)
        .await
        .map_err(|e| RaftError::Internal(format!("hrp read len: {}", e)))?;
    let payload_len = u32::from_be_bytes(len_buf);

    if payload_len > MAX_MESSAGE_SIZE {
        return Err(RaftError::Internal(format!(
            "hrp message too large: {} bytes",
            payload_len
        )));
    }

    // Read bincode payload
    let mut payload = vec![0u8; payload_len as usize];
    stream
        .read_exact(&mut payload)
        .await
        .map_err(|e| RaftError::Internal(format!("hrp read payload: {}", e)))?;

    // Read CRC32 trailer
    let mut crc_buf = [0u8; 4];
    stream
        .read_exact(&mut crc_buf)
        .await
        .map_err(|e| RaftError::Internal(format!("hrp read crc: {}", e)))?;
    let expected_crc = u32::from_be_bytes(crc_buf);

    // Verify CRC32 over magic + len + payload
    let mut check_data = Vec::with_capacity(4 + 4 + payload.len());
    check_data.extend_from_slice(&HRP_MAGIC);
    check_data.extend_from_slice(&len_buf);
    check_data.extend_from_slice(&payload);
    let actual_crc = crc32fast::hash(&check_data);

    if expected_crc != actual_crc {
        return Err(RaftError::Internal(format!(
            "hrp CRC mismatch: expected {:08x}, got {:08x}",
            expected_crc, actual_crc
        )));
    }

    bincode::serde::decode_from_slice(&payload, bincode::config::standard())
        .map(|(v, _)| v)
        .map_err(|e| RaftError::Internal(format!("hrp bincode deserialize: {}", e)))
}

// ============================================================================
// HRP v2 Binary Wire Format (Phase 3: Write Tokens + HMAC)
// ============================================================================

/// HRP v2 magic number: "HRP\x02" — binary envelope with write token + HMAC.
const HRP_V2_MAGIC: [u8; 4] = [0x48, 0x52, 0x50, 0x02];

/// Result of reading an envelope — includes optional write token for v2.
pub struct ReadEnvelopeResult {
    pub envelope: RaftEnvelope,
    pub write_token: Option<crate::hrp_security::WriteToken>,
}

/// Write an HRP v2 binary envelope with optional write token and HMAC trailer.
///
/// v2 format:
/// ```text
/// [4B] magic 0x48525002
/// [4B] payload_length
/// [N bytes] bincode payload
/// [1B] has_write_token (0 or 1)
/// [56B] write token (if has_write_token == 1)
/// [32B] HMAC-SHA256 (over magic + len + payload + token_flag + token)
/// [4B] CRC32 (over everything including HMAC)
/// ```
pub async fn write_hrp_v2_envelope<S: AsyncWrite + Unpin>(
    stream: &mut S,
    envelope: &RaftEnvelope,
    write_token: Option<&crate::hrp_security::WriteToken>,
    hmac_key: Option<&[u8; 32]>,
) -> Result<(), RaftError> {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    let bincode_data = bincode::serde::encode_to_vec(envelope, bincode::config::standard())
        .map_err(|e| RaftError::Internal(format!("hrp v2 bincode serialize: {}", e)))?;

    let payload_len = bincode_data.len() as u32;
    if payload_len > MAX_MESSAGE_SIZE {
        return Err(RaftError::Internal(format!(
            "hrp v2 message too large: {} bytes",
            payload_len
        )));
    }

    // Build frame: magic + len + payload + token_flag + [token]
    let token_size = if write_token.is_some() {
        1 + crate::hrp_security::WriteToken::WIRE_SIZE
    } else {
        1
    };
    let frame_size = 4 + 4 + bincode_data.len() + token_size + 32 + 4;
    let mut frame = Vec::with_capacity(frame_size);

    frame.extend_from_slice(&HRP_V2_MAGIC);
    frame.extend_from_slice(&payload_len.to_be_bytes());
    frame.extend_from_slice(&bincode_data);

    // Write token flag + optional token
    if let Some(token) = write_token {
        frame.push(1u8);
        frame.extend_from_slice(&token.to_bytes());
    } else {
        frame.push(0u8);
    }

    // Compute HMAC-SHA256 over everything so far (if key available)
    let hmac_bytes = if let Some(key) = hmac_key {
        let mut mac = <Hmac<Sha256>>::new_from_slice(key).expect("HMAC can take key of any size");
        mac.update(&frame);
        let result = mac.finalize().into_bytes();
        let mut h = [0u8; 32];
        h.copy_from_slice(&result);
        h
    } else {
        [0u8; 32] // No HMAC when security is not configured
    };
    frame.extend_from_slice(&hmac_bytes);

    // CRC32 over everything including HMAC
    let crc = crc32fast::hash(&frame);
    frame.extend_from_slice(&crc.to_be_bytes());

    stream
        .write_all(&frame)
        .await
        .map_err(|e| RaftError::Internal(format!("hrp v2 write: {}", e)))?;
    stream
        .flush()
        .await
        .map_err(|e| RaftError::Internal(format!("hrp v2 flush: {}", e)))?;
    Ok(())
}

/// Read the HRP v2 payload after the magic has already been consumed.
async fn read_hrp_v2_payload<S: AsyncRead + Unpin>(
    stream: &mut S,
) -> Result<ReadEnvelopeResult, RaftError> {
    // Read payload length
    let mut len_buf = [0u8; 4];
    stream
        .read_exact(&mut len_buf)
        .await
        .map_err(|e| RaftError::Internal(format!("hrp v2 read len: {}", e)))?;
    let payload_len = u32::from_be_bytes(len_buf);

    if payload_len > MAX_MESSAGE_SIZE {
        return Err(RaftError::Internal(format!(
            "hrp v2 message too large: {} bytes",
            payload_len
        )));
    }

    // Read bincode payload
    let mut payload = vec![0u8; payload_len as usize];
    stream
        .read_exact(&mut payload)
        .await
        .map_err(|e| RaftError::Internal(format!("hrp v2 read payload: {}", e)))?;

    // Read write token flag
    let mut flag_buf = [0u8; 1];
    stream
        .read_exact(&mut flag_buf)
        .await
        .map_err(|e| RaftError::Internal(format!("hrp v2 read token flag: {}", e)))?;

    let write_token = if flag_buf[0] == 1 {
        let mut token_buf = [0u8; crate::hrp_security::WriteToken::WIRE_SIZE];
        stream
            .read_exact(&mut token_buf)
            .await
            .map_err(|e| RaftError::Internal(format!("hrp v2 read token: {}", e)))?;
        Some(crate::hrp_security::WriteToken::from_bytes(&token_buf))
    } else {
        None
    };

    // Read HMAC-SHA256
    let mut hmac_buf = [0u8; 32];
    stream
        .read_exact(&mut hmac_buf)
        .await
        .map_err(|e| RaftError::Internal(format!("hrp v2 read hmac: {}", e)))?;

    // Read CRC32 trailer
    let mut crc_buf = [0u8; 4];
    stream
        .read_exact(&mut crc_buf)
        .await
        .map_err(|e| RaftError::Internal(format!("hrp v2 read crc: {}", e)))?;
    let expected_crc = u32::from_be_bytes(crc_buf);

    // Reconstruct the frame for CRC verification
    let mut check_data = Vec::with_capacity(4 + 4 + payload.len() + 1 + 56 + 32);
    check_data.extend_from_slice(&HRP_V2_MAGIC);
    check_data.extend_from_slice(&len_buf);
    check_data.extend_from_slice(&payload);
    check_data.push(flag_buf[0]);
    if let Some(ref token) = write_token {
        check_data.extend_from_slice(&token.to_bytes());
    }
    check_data.extend_from_slice(&hmac_buf);

    let actual_crc = crc32fast::hash(&check_data);
    if expected_crc != actual_crc {
        return Err(RaftError::Internal(format!(
            "hrp v2 CRC mismatch: expected {:08x}, got {:08x}",
            expected_crc, actual_crc
        )));
    }

    let (envelope, _): (RaftEnvelope, _) = bincode::serde::decode_from_slice(&payload, bincode::config::standard())
        .map_err(|e| RaftError::Internal(format!("hrp v2 bincode deserialize: {}", e)))?;

    Ok(ReadEnvelopeResult {
        envelope,
        write_token,
    })
}

/// Read an envelope with full format auto-detection (JSON, HRP v1, HRP v2).
///
/// Returns the envelope and optional write token (v2 only).
pub async fn read_envelope_v2<S: AsyncRead + Unpin>(
    stream: &mut S,
) -> Result<ReadEnvelopeResult, RaftError> {
    let mut header = [0u8; 4];
    stream
        .read_exact(&mut header)
        .await
        .map_err(|e| RaftError::Internal(format!("read header: {}", e)))?;

    if header == HRP_V2_MAGIC {
        return read_hrp_v2_payload(stream).await;
    }

    if header == HRP_MAGIC {
        let envelope = read_hrp_payload(stream).await?;
        return Ok(ReadEnvelopeResult {
            envelope,
            write_token: None,
        });
    }

    // Legacy JSON format
    let len = u32::from_be_bytes(header);
    if len > MAX_MESSAGE_SIZE {
        return Err(RaftError::Internal(format!(
            "message too large: {} bytes",
            len
        )));
    }
    let mut buf = vec![0u8; len as usize];
    stream
        .read_exact(&mut buf)
        .await
        .map_err(|e| RaftError::Internal(format!("read payload: {}", e)))?;
    let envelope: RaftEnvelope = serde_json::from_slice(&buf)
        .map_err(|e| RaftError::Internal(format!("deserialize: {}", e)))?;
    Ok(ReadEnvelopeResult {
        envelope,
        write_token: None,
    })
}

// ============================================================================
// TCP Raft Transport
// ============================================================================

/// TCP-based transport implementing the `RaftTransport` trait.
///
/// Maintains a lazy connection pool to peers. Connections are established
/// on first use and cached. Failed connections are evicted so the next
/// RPC will reconnect.
///
/// When compiled with `--features tls` and configured with a `TlsConnector`,
/// outgoing connections are automatically upgraded to TLS.
pub struct TcpRaftTransport {
    /// This node's ID (included in every outgoing envelope).
    node_id: NodeId,
    /// Peer address map: node_id → "host:port".
    peers: RwLock<HashMap<NodeId, String>>,
    /// Cached connections per peer (plain TCP or TLS-wrapped).
    connections: RwLock<HashMap<NodeId, Arc<Mutex<RaftStream>>>>,
    /// Timeout for establishing a new TCP connection.
    connect_timeout: Duration,
    /// Timeout for a complete RPC round-trip (send + receive).
    rpc_timeout: Duration,
    /// Optional TLS connector for outgoing connections.
    #[cfg(feature = "tls")]
    tls_connector: Option<TlsConnector>,
    /// HRP Phase 3: Optional security manager for write token generation/verification.
    security: Option<Arc<crate::hrp_security::EpochKeyManager>>,
    /// HRP Phase 3: Current Raft term (for write token generation).
    current_term: Arc<std::sync::atomic::AtomicU64>,
}

impl TcpRaftTransport {
    /// Create a new TCP transport.
    ///
    /// `peers` maps each peer's node ID to its "host:port" address.
    pub fn new(node_id: NodeId, peers: HashMap<NodeId, String>) -> Self {
        Self {
            node_id,
            peers: RwLock::new(peers),
            connections: RwLock::new(HashMap::new()),
            connect_timeout: Duration::from_secs(2),
            rpc_timeout: Duration::from_secs(5),
            #[cfg(feature = "tls")]
            tls_connector: None,
            security: None,
            current_term: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Enable TLS for outgoing connections.
    #[cfg(feature = "tls")]
    pub fn with_tls(mut self, connector: TlsConnector) -> Self {
        self.tls_connector = Some(connector);
        self
    }

    /// Enable HRP v2 security (write tokens + HMAC).
    pub fn with_security(
        mut self,
        key_mgr: Arc<crate::hrp_security::EpochKeyManager>,
        term: Arc<std::sync::atomic::AtomicU64>,
    ) -> Self {
        self.security = Some(key_mgr);
        self.current_term = term;
        self
    }

    /// Get a reference to the security manager (if configured).
    pub fn security(&self) -> Option<&Arc<crate::hrp_security::EpochKeyManager>> {
        self.security.as_ref()
    }

    /// Add or update a peer's address (for dynamic membership changes).
    pub async fn add_peer(&self, node_id: NodeId, addr: String) {
        self.peers.write().await.insert(node_id, addr);
    }

    /// Remove a peer (for dynamic membership changes).
    pub async fn remove_peer(&self, node_id: &NodeId) {
        self.peers.write().await.remove(node_id);
        self.connections.write().await.remove(node_id);
    }

    /// Get or establish a connection to a peer.
    ///
    /// If a TLS connector is configured, the TCP connection is upgraded
    /// to TLS after the initial handshake.
    async fn get_connection(&self, target: &NodeId) -> Result<Arc<Mutex<RaftStream>>, RaftError> {
        // Check for an existing cached connection
        {
            let conns = self.connections.read().await;
            if let Some(conn) = conns.get(target) {
                return Ok(conn.clone());
            }
        }

        // Look up the peer's address
        let addr = {
            let peers = self.peers.read().await;
            peers
                .get(target)
                .cloned()
                .ok_or_else(|| RaftError::NodeNotFound(target.clone()))?
        };

        // Establish a new TCP connection with timeout
        let tcp_stream = tokio::time::timeout(self.connect_timeout, TcpStream::connect(&addr))
            .await
            .map_err(|_| RaftError::Internal(format!("connect timeout to {} ({})", target, addr)))?
            .map_err(|e| RaftError::Internal(format!("connect to {} ({}): {}", target, addr, e)))?;

        // Disable Nagle for low-latency RPCs
        let _ = tcp_stream.set_nodelay(true);

        // Wrap in TLS if a connector is configured
        let stream = self.maybe_wrap_tls(tcp_stream, &addr).await?;

        let conn = Arc::new(Mutex::new(stream));
        self.connections
            .write()
            .await
            .insert(target.clone(), conn.clone());
        Ok(conn)
    }

    /// Wrap a TCP stream in TLS if a connector is configured, otherwise
    /// return it as a plain stream.
    #[cfg(feature = "tls")]
    async fn maybe_wrap_tls(
        &self,
        tcp_stream: TcpStream,
        addr: &str,
    ) -> Result<RaftStream, RaftError> {
        match &self.tls_connector {
            Some(connector) => {
                // Extract the host part for SNI (strip port)
                let host = addr.split(':').next().unwrap_or(addr);

                let server_name = rustls::pki_types::ServerName::try_from(host.to_string())
                    .map_err(|e| {
                        RaftError::Internal(format!("invalid server name '{}': {}", host, e))
                    })?;

                let tls_stream = connector
                    .connect(server_name, tcp_stream)
                    .await
                    .map_err(|e| {
                        RaftError::Internal(format!("TLS handshake with {}: {}", addr, e))
                    })?;

                Ok(RaftStream::TlsClient(tls_stream))
            }
            None => Ok(RaftStream::Plain(tcp_stream)),
        }
    }

    /// Without TLS feature, always returns a plain stream.
    #[cfg(not(feature = "tls"))]
    async fn maybe_wrap_tls(
        &self,
        tcp_stream: TcpStream,
        _addr: &str,
    ) -> Result<RaftStream, RaftError> {
        Ok(RaftStream::Plain(tcp_stream))
    }

    /// Evict a cached connection (called on I/O errors).
    async fn evict_connection(&self, target: &NodeId) {
        self.connections.write().await.remove(target);
    }

    /// Send a Raft RPC and wait for the response.
    ///
    /// HRP Phase 3: Uses v2 binary wire format with write token + HMAC when
    /// security is configured. Falls back to v1 binary format otherwise.
    /// The reader auto-detects all three formats (JSON, v1, v2) via magic number.
    ///
    /// On connection failure, evicts the cached connection so the next
    /// call will reconnect.
    async fn send_rpc(&self, target: &NodeId, msg: RaftMessage) -> Result<RaftMessage, RaftError> {
        let conn = self.get_connection(target).await?;

        let envelope = RaftEnvelope {
            from: self.node_id.clone(),
            msg,
        };

        // Capture security context outside the async block
        let security = self.security.clone();
        // Extract the term from the Raft message itself (not the stale counter)
        // so the token's term matches what the receiver sees in the message.
        let term = match &envelope.msg {
            RaftMessage::RequestVote(req) => req.term,
            RaftMessage::AppendEntries(req) => req.term,
            RaftMessage::InstallSnapshot(req) => req.term,
            RaftMessage::RequestVoteResponse(resp) => resp.term,
            RaftMessage::AppendEntriesResponse(resp) => resp.term,
            RaftMessage::InstallSnapshotResponse(resp) => resp.term,
        };

        let result = tokio::time::timeout(self.rpc_timeout, async {
            let mut stream = conn.lock().await;

            if let Some(ref key_mgr) = security {
                // HRP v2: write with write token + HMAC
                let bincode_data = bincode::serde::encode_to_vec(&envelope, bincode::config::standard())
                    .map_err(|e| RaftError::Internal(format!("serialize for token: {}", e)))?;
                let token = key_mgr.generate_token(term, &bincode_data);
                let hmac_key = key_mgr.current_key();
                write_hrp_v2_envelope(&mut *stream, &envelope, Some(&token), Some(&hmac_key))
                    .await?;
            } else {
                // HRP v1: write binary format (no security)
                write_hrp_envelope(&mut *stream, &envelope).await?;
            }

            // Read response — auto-detects v1/v2/JSON
            let result = read_envelope_v2(&mut *stream).await?;
            Ok::<_, RaftError>(result.envelope)
        })
        .await;

        match result {
            Ok(Ok(response)) => Ok(response.msg),
            Ok(Err(e)) => {
                self.evict_connection(target).await;
                Err(e)
            }
            Err(_) => {
                self.evict_connection(target).await;
                Err(RaftError::Timeout)
            }
        }
    }
}

#[async_trait::async_trait]
impl RaftTransport for TcpRaftTransport {
    async fn send_request_vote(
        &self,
        target: &NodeId,
        request: RequestVoteRequest,
    ) -> Result<RequestVoteResponse, RaftError> {
        match self
            .send_rpc(target, RaftMessage::RequestVote(request))
            .await?
        {
            RaftMessage::RequestVoteResponse(resp) => Ok(resp),
            other => Err(RaftError::Internal(format!(
                "unexpected response: expected RequestVoteResponse, got {:?}",
                std::mem::discriminant(&other)
            ))),
        }
    }

    async fn send_append_entries(
        &self,
        target: &NodeId,
        request: AppendEntriesRequest,
    ) -> Result<AppendEntriesResponse, RaftError> {
        match self
            .send_rpc(target, RaftMessage::AppendEntries(request))
            .await?
        {
            RaftMessage::AppendEntriesResponse(resp) => Ok(resp),
            other => Err(RaftError::Internal(format!(
                "unexpected response: expected AppendEntriesResponse, got {:?}",
                std::mem::discriminant(&other)
            ))),
        }
    }

    async fn send_install_snapshot(
        &self,
        target: &NodeId,
        request: InstallSnapshotRequest,
    ) -> Result<InstallSnapshotResponse, RaftError> {
        match self
            .send_rpc(target, RaftMessage::InstallSnapshot(request))
            .await?
        {
            RaftMessage::InstallSnapshotResponse(resp) => Ok(resp),
            other => Err(RaftError::Internal(format!(
                "unexpected response: expected InstallSnapshotResponse, got {:?}",
                std::mem::discriminant(&other)
            ))),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_envelope_serde_roundtrip() {
        let envelope = RaftEnvelope {
            from: "node1".to_string(),
            msg: RaftMessage::RequestVote(RequestVoteRequest {
                term: 1,
                candidate_id: "node1".to_string(),
                last_log_index: 0,
                last_log_term: 0,
            }),
        };
        let json = serde_json::to_vec(&envelope).unwrap();
        let decoded: RaftEnvelope = serde_json::from_slice(&json).unwrap();
        assert_eq!(decoded.from, "node1");
        match decoded.msg {
            RaftMessage::RequestVote(req) => {
                assert_eq!(req.term, 1);
                assert_eq!(req.candidate_id, "node1");
            }
            _ => panic!("wrong message type"),
        }
    }

    #[test]
    fn test_envelope_append_entries_serde() {
        use crate::raft::Command;
        use crate::raft::LogEntry;

        let envelope = RaftEnvelope {
            from: "leader1".to_string(),
            msg: RaftMessage::AppendEntries(AppendEntriesRequest {
                term: 3,
                leader_id: "leader1".to_string(),
                prev_log_index: 5,
                prev_log_term: 2,
                entries: vec![LogEntry::new(
                    3,
                    6,
                    Command::Set {
                        key: b"hello".to_vec(),
                        value: b"world".to_vec(),
                    },
                )],
                leader_commit: 5,
                energy_state: None,
            }),
        };
        let json = serde_json::to_vec(&envelope).unwrap();
        let decoded: RaftEnvelope = serde_json::from_slice(&json).unwrap();
        match decoded.msg {
            RaftMessage::AppendEntries(req) => {
                assert_eq!(req.term, 3);
                assert_eq!(req.entries.len(), 1);
            }
            _ => panic!("wrong message type"),
        }
    }

    #[test]
    fn test_envelope_install_snapshot_serde() {
        use crate::raft::ClusterConfig;

        let envelope = RaftEnvelope {
            from: "leader1".to_string(),
            msg: RaftMessage::InstallSnapshot(InstallSnapshotRequest {
                term: 5,
                leader_id: "leader1".to_string(),
                last_included_index: 100,
                last_included_term: 4,
                offset: 0,
                data: vec![1, 2, 3, 4],
                done: true,
                config: ClusterConfig::single("leader1".to_string()),
            }),
        };
        let json = serde_json::to_vec(&envelope).unwrap();
        let decoded: RaftEnvelope = serde_json::from_slice(&json).unwrap();
        match decoded.msg {
            RaftMessage::InstallSnapshot(req) => {
                assert_eq!(req.term, 5);
                assert!(req.done);
                assert_eq!(req.data, vec![1, 2, 3, 4]);
            }
            _ => panic!("wrong message type"),
        }
    }

    #[tokio::test]
    async fn test_tcp_transport_peer_management() {
        let transport = TcpRaftTransport::new("node1".to_string(), HashMap::new());

        // Add peers
        transport
            .add_peer("node2".to_string(), "127.0.0.1:17002".to_string())
            .await;
        transport
            .add_peer("node3".to_string(), "127.0.0.1:17003".to_string())
            .await;

        {
            let peers = transport.peers.read().await;
            assert_eq!(peers.len(), 2);
            assert_eq!(peers.get("node2").unwrap(), "127.0.0.1:17002");
        }

        // Remove a peer
        transport.remove_peer(&"node2".to_string()).await;
        {
            let peers = transport.peers.read().await;
            assert_eq!(peers.len(), 1);
            assert!(!peers.contains_key("node2"));
        }
    }

    #[tokio::test]
    async fn test_tcp_transport_unknown_peer_error() {
        let transport = TcpRaftTransport::new("node1".to_string(), HashMap::new());
        let result = transport
            .send_request_vote(
                &"unknown".to_string(),
                RequestVoteRequest {
                    term: 1,
                    candidate_id: "node1".to_string(),
                    last_log_index: 0,
                    last_log_term: 0,
                },
            )
            .await;
        assert!(result.is_err());
        match result.unwrap_err() {
            RaftError::NodeNotFound(id) => assert_eq!(id, "unknown"),
            other => panic!("expected NodeNotFound, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_tcp_transport_connection_refused() {
        let mut peers = HashMap::new();
        // Port that nothing is listening on
        peers.insert("node2".to_string(), "127.0.0.1:19999".to_string());
        let transport = TcpRaftTransport::new("node1".to_string(), peers);
        let result = transport
            .send_request_vote(
                &"node2".to_string(),
                RequestVoteRequest {
                    term: 1,
                    candidate_id: "node1".to_string(),
                    last_log_index: 0,
                    last_log_term: 0,
                },
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_write_read_envelope_roundtrip() {
        // Start a TCP listener
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let envelope = RaftEnvelope {
            from: "test_node".to_string(),
            msg: RaftMessage::RequestVote(RequestVoteRequest {
                term: 42,
                candidate_id: "test_node".to_string(),
                last_log_index: 10,
                last_log_term: 5,
            }),
        };
        let envelope_clone = envelope.clone();

        // Spawn a task to accept and read the envelope
        let handle = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            read_envelope(&mut stream).await.unwrap()
        });

        // Connect and write the envelope
        let mut stream = TcpStream::connect(addr).await.unwrap();
        write_envelope(&mut stream, &envelope_clone).await.unwrap();

        let received = handle.await.unwrap();
        assert_eq!(received.from, "test_node");
        match received.msg {
            RaftMessage::RequestVote(req) => {
                assert_eq!(req.term, 42);
                assert_eq!(req.last_log_index, 10);
            }
            _ => panic!("wrong message type"),
        }
    }

    #[tokio::test]
    async fn test_raft_stream_plain_roundtrip() {
        // Verify RaftStream::Plain delegates correctly
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let envelope = RaftEnvelope {
            from: "stream_test".to_string(),
            msg: RaftMessage::RequestVote(RequestVoteRequest {
                term: 99,
                candidate_id: "stream_test".to_string(),
                last_log_index: 0,
                last_log_term: 0,
            }),
        };
        let envelope_clone = envelope.clone();

        let handle = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut raft_stream = RaftStream::Plain(stream);
            read_envelope(&mut raft_stream).await.unwrap()
        });

        let stream = TcpStream::connect(addr).await.unwrap();
        let mut raft_stream = RaftStream::Plain(stream);
        write_envelope(&mut raft_stream, &envelope_clone)
            .await
            .unwrap();

        let received = handle.await.unwrap();
        assert_eq!(received.from, "stream_test");
        match received.msg {
            RaftMessage::RequestVote(req) => assert_eq!(req.term, 99),
            _ => panic!("wrong message type"),
        }
    }

    #[tokio::test]
    async fn test_hrp_v2_envelope_roundtrip() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let secret = [0x42u8; 32];
        let key_mgr = crate::hrp_security::EpochKeyManager::new(secret);

        let envelope = RaftEnvelope {
            from: "v2_test".to_string(),
            msg: RaftMessage::RequestVote(RequestVoteRequest {
                term: 77,
                candidate_id: "v2_test".to_string(),
                last_log_index: 5,
                last_log_term: 3,
            }),
        };
        let envelope_clone = envelope.clone();

        // Generate write token
        let bincode_data = bincode::serde::encode_to_vec(&envelope_clone, bincode::config::standard()).unwrap();
        let token = key_mgr.generate_token(77, &bincode_data);
        let hmac_key = key_mgr.current_key();

        let handle = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            read_envelope_v2(&mut stream).await.unwrap()
        });

        let mut stream = TcpStream::connect(addr).await.unwrap();
        write_hrp_v2_envelope(&mut stream, &envelope_clone, Some(&token), Some(&hmac_key))
            .await
            .unwrap();

        let result = handle.await.unwrap();
        assert_eq!(result.envelope.from, "v2_test");
        assert!(result.write_token.is_some());
        let recv_token = result.write_token.unwrap();
        assert_eq!(recv_token.term, 77);
        assert_eq!(recv_token.epoch, 0);
        match result.envelope.msg {
            RaftMessage::RequestVote(req) => assert_eq!(req.term, 77),
            _ => panic!("wrong message type"),
        }
    }

    #[tokio::test]
    async fn test_v1_v2_interop() {
        // v1 writer → v2 reader should auto-detect v1 format
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let envelope = RaftEnvelope {
            from: "v1_node".to_string(),
            msg: RaftMessage::RequestVote(RequestVoteRequest {
                term: 33,
                candidate_id: "v1_node".to_string(),
                last_log_index: 0,
                last_log_term: 0,
            }),
        };
        let envelope_clone = envelope.clone();

        let handle = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            // Read with v2 reader (should auto-detect v1)
            read_envelope_v2(&mut stream).await.unwrap()
        });

        let mut stream = TcpStream::connect(addr).await.unwrap();
        // Write with v1 writer
        write_hrp_envelope(&mut stream, &envelope_clone)
            .await
            .unwrap();

        let result = handle.await.unwrap();
        assert_eq!(result.envelope.from, "v1_node");
        assert!(result.write_token.is_none()); // v1 has no write token
        match result.envelope.msg {
            RaftMessage::RequestVote(req) => assert_eq!(req.term, 33),
            _ => panic!("wrong message type"),
        }
    }
}
