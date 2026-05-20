//! Async TCP connection to an JouleDB server.
//!
//! [`Connection`] manages a single TCP socket and serialises wire access
//! through a `tokio::sync::Mutex` so that the connection can be safely shared
//! across tasks.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::Mutex;

use crate::client::QueryResult;
use crate::error::{ClientError, Result};
use crate::protocol::{self, Flags, HEADER_SIZE, Message, MessageType};

// ============================================================================
// ConnectionConfig
// ============================================================================

/// Configuration for a single [`Connection`].
#[derive(Debug, Clone)]
pub struct ConnectionConfig {
    /// Hostname or IP address of the JouleDB server.
    pub host: String,
    /// TCP port.
    pub port: u16,
    /// Timeout for the initial TCP connect.
    pub connect_timeout: Duration,
    /// Timeout for reading a response from the server.
    pub read_timeout: Duration,
    /// Timeout for writing a request to the server.
    pub write_timeout: Duration,
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 9000,
            connect_timeout: Duration::from_secs(5),
            read_timeout: Duration::from_secs(30),
            write_timeout: Duration::from_secs(10),
        }
    }
}

// ============================================================================
// Connection
// ============================================================================

/// A single async TCP connection to an JouleDB server.
///
/// All methods acquire an internal mutex before touching the socket, so it is
/// safe to share a `Connection` between tasks via `Arc`. For higher
/// throughput, consider using the [`ConnectionPool`](crate::pool::ConnectionPool).
pub struct Connection {
    stream: Mutex<TcpStream>,
    next_request_id: AtomicU32,
    config: ConnectionConfig,
}

impl std::fmt::Debug for Connection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Connection")
            .field(
                "next_request_id",
                &self.next_request_id.load(Ordering::Relaxed),
            )
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl Connection {
    /// Open a connection to the server described by `config`.
    pub async fn connect(config: ConnectionConfig) -> Result<Self> {
        let addr = format!("{}:{}", config.host, config.port);
        let stream = tokio::time::timeout(config.connect_timeout, TcpStream::connect(&addr))
            .await
            .map_err(|_| ClientError::Timeout(config.connect_timeout))?
            .map_err(|e| ClientError::connection_failed(format!("{}: {}", addr, e)))?;

        // Disable Nagle for lower latency on small messages.
        stream
            .set_nodelay(true)
            .map_err(|e| ClientError::connection_failed(format!("set_nodelay: {}", e)))?;

        Ok(Self {
            stream: Mutex::new(stream),
            next_request_id: AtomicU32::new(1),
            config,
        })
    }

    /// Send a `Ping` and return the round-trip duration.
    pub async fn ping(&self) -> Result<Duration> {
        let start = tokio::time::Instant::now();
        let request_id = self.next_id();
        let msg = Message::ping(request_id);
        let resp = self.send_recv(msg).await?;

        if resp.msg_type == MessageType::Error {
            let (code, message) = resp.parse_error()?;
            return Err(ClientError::server(code, message));
        }
        if resp.msg_type != MessageType::Pong {
            return Err(ClientError::invalid_response(format!(
                "expected Pong, got {:?}",
                resp.msg_type
            )));
        }

        Ok(start.elapsed())
    }

    /// Retrieve the value for `key`. Returns `None` if the key does not exist.
    pub async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let request_id = self.next_id();
        let msg = Message::get(request_id, key.as_bytes());
        let resp = self.send_recv(msg).await?;
        self.check_error(&resp)?;

        if resp.msg_type != MessageType::GetResponse {
            return Err(ClientError::invalid_response(format!(
                "expected GetResponse, got {:?}",
                resp.msg_type
            )));
        }

        resp.parse_get_response()
    }

    /// Store `value` under `key`, optionally with a TTL in seconds.
    /// Returns `true` if the server reports success.
    pub async fn put(&self, key: &str, value: &[u8], ttl: Option<u64>) -> Result<bool> {
        let request_id = self.next_id();
        let msg = Message::set(request_id, key.as_bytes(), value, ttl);
        let resp = self.send_recv(msg).await?;
        self.check_error(&resp)?;

        if resp.msg_type != MessageType::SetResponse {
            return Err(ClientError::invalid_response(format!(
                "expected SetResponse, got {:?}",
                resp.msg_type
            )));
        }

        resp.parse_set_response()
    }

    /// Delete the value for `key`. Returns `true` if the key existed.
    pub async fn delete(&self, key: &str) -> Result<bool> {
        let request_id = self.next_id();
        let msg = Message::delete(request_id, key.as_bytes());
        let resp = self.send_recv(msg).await?;
        self.check_error(&resp)?;

        if resp.msg_type != MessageType::DeleteResponse {
            return Err(ClientError::invalid_response(format!(
                "expected DeleteResponse, got {:?}",
                resp.msg_type
            )));
        }

        resp.parse_delete_response()
    }

    /// Execute a SQL query and return the result set.
    pub async fn query(&self, sql: &str, params: &[serde_json::Value]) -> Result<QueryResult> {
        let request_id = self.next_id();
        let params_json =
            if params.is_empty() {
                None
            } else {
                Some(serde_json::to_vec(params).map_err(|e| {
                    ClientError::protocol(format!("failed to serialize params: {}", e))
                })?)
            };
        let msg = Message::query(request_id, sql, params_json.as_deref());
        let resp = self.send_recv(msg).await?;
        self.check_error(&resp)?;

        if resp.msg_type != MessageType::QueryResponse {
            return Err(ClientError::invalid_response(format!(
                "expected QueryResponse, got {:?}",
                resp.msg_type
            )));
        }

        let raw = resp.parse_query_response()?;
        QueryResult::from_json(&raw)
    }

    /// Execute a SQL statement (INSERT/UPDATE/DELETE) and return the number
    /// of affected rows.
    pub async fn execute(&self, sql: &str, params: &[serde_json::Value]) -> Result<u64> {
        let result = self.query(sql, params).await?;
        Ok(result.row_count as u64)
    }

    /// Begin a transaction. Returns a [`Transaction`] handle that must be
    /// committed or rolled back.
    pub async fn begin(&self) -> Result<Transaction<'_>> {
        let request_id = self.next_id();
        let msg = Message::begin_tx(request_id);
        let resp = self.send_recv(msg).await?;
        self.check_error(&resp)?;

        if resp.msg_type != MessageType::BeginTxResponse {
            return Err(ClientError::invalid_response(format!(
                "expected BeginTxResponse, got {:?}",
                resp.msg_type
            )));
        }

        Ok(Transaction { conn: self })
    }

    // -- Internal -----------------------------------------------------------

    /// Allocate the next request ID.
    fn next_id(&self) -> u32 {
        self.next_request_id.fetch_add(1, Ordering::Relaxed)
    }

    /// If the response is an `Error` message, convert it into a
    /// `ClientError::ServerError`.
    fn check_error(&self, resp: &Message) -> Result<()> {
        if resp.msg_type == MessageType::Error {
            let (code, message) = resp.parse_error()?;
            return Err(ClientError::server(code, message));
        }
        Ok(())
    }

    /// Send a request and read the corresponding response.
    ///
    /// This acquires the stream mutex for the entire duration of the
    /// round-trip, ensuring no interleaving on the wire.
    async fn send_recv(&self, mut msg: Message) -> Result<Message> {
        // Mark that we expect a response.
        msg.flags.set(Flags::EXPECT_RESPONSE);

        let encoded = protocol::encode(&msg)?;

        let mut stream = self.stream.lock().await;

        // Write
        tokio::time::timeout(self.config.write_timeout, stream.write_all(&encoded))
            .await
            .map_err(|_| ClientError::Timeout(self.config.write_timeout))?
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::BrokenPipe
                    || e.kind() == std::io::ErrorKind::ConnectionReset
                {
                    ClientError::ConnectionClosed
                } else {
                    ClientError::IoError(e)
                }
            })?;

        tokio::time::timeout(self.config.write_timeout, stream.flush())
            .await
            .map_err(|_| ClientError::Timeout(self.config.write_timeout))?
            .map_err(|e| ClientError::IoError(e))?;

        // Read header (16 bytes)
        let mut header_buf = [0u8; HEADER_SIZE];
        tokio::time::timeout(self.config.read_timeout, stream.read_exact(&mut header_buf))
            .await
            .map_err(|_| ClientError::Timeout(self.config.read_timeout))?
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    ClientError::ConnectionClosed
                } else {
                    ClientError::IoError(e)
                }
            })?;

        let (_msg_type, _request_id, _flags, payload_len) = protocol::decode_header(&header_buf)?;

        // Read payload
        let mut payload_buf = vec![0u8; payload_len as usize];
        if payload_len > 0 {
            tokio::time::timeout(
                self.config.read_timeout,
                stream.read_exact(&mut payload_buf),
            )
            .await
            .map_err(|_| ClientError::Timeout(self.config.read_timeout))?
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    ClientError::ConnectionClosed
                } else {
                    ClientError::IoError(e)
                }
            })?;
        }

        // Re-assemble the full buffer and decode the complete message.
        let mut full = Vec::with_capacity(HEADER_SIZE + payload_len as usize);
        full.extend_from_slice(&header_buf);
        full.extend_from_slice(&payload_buf);
        protocol::decode(&full)
    }

    /// Expose the config for pool usage.
    pub(crate) fn config(&self) -> &ConnectionConfig {
        &self.config
    }
}

// ============================================================================
// Transaction
// ============================================================================

/// A handle to an active server-side transaction.
///
/// **You must call [`commit`](Transaction::commit) or
/// [`rollback`](Transaction::rollback)**. Dropping a `Transaction` without
/// doing so will **not** automatically roll back on the server (the server
/// may time it out eventually).
pub struct Transaction<'a> {
    conn: &'a Connection,
}

impl std::fmt::Debug for Transaction<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transaction").finish_non_exhaustive()
    }
}

impl<'a> Transaction<'a> {
    /// Execute a SQL query within this transaction.
    pub async fn query(&self, sql: &str, params: &[serde_json::Value]) -> Result<QueryResult> {
        self.conn.query(sql, params).await
    }

    /// Execute a SQL statement within this transaction and return the number
    /// of affected rows.
    pub async fn execute(&self, sql: &str, params: &[serde_json::Value]) -> Result<u64> {
        self.conn.execute(sql, params).await
    }

    /// Commit the transaction. Consumes `self`.
    pub async fn commit(self) -> Result<()> {
        let request_id = self.conn.next_id();
        let msg = Message::commit(request_id);
        let resp = self.conn.send_recv(msg).await?;
        self.conn.check_error(&resp)?;

        if resp.msg_type != MessageType::CommitResponse {
            return Err(ClientError::invalid_response(format!(
                "expected CommitResponse, got {:?}",
                resp.msg_type
            )));
        }
        Ok(())
    }

    /// Roll back the transaction. Consumes `self`.
    pub async fn rollback(self) -> Result<()> {
        let request_id = self.conn.next_id();
        let msg = Message::rollback(request_id);
        let resp = self.conn.send_recv(msg).await?;
        self.conn.check_error(&resp)?;

        if resp.msg_type != MessageType::RollbackResponse {
            return Err(ClientError::invalid_response(format!(
                "expected RollbackResponse, got {:?}",
                resp.msg_type
            )));
        }
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = ConnectionConfig::default();
        assert_eq!(cfg.host, "127.0.0.1");
        assert_eq!(cfg.port, 9000);
        assert_eq!(cfg.connect_timeout, Duration::from_secs(5));
        assert_eq!(cfg.read_timeout, Duration::from_secs(30));
        assert_eq!(cfg.write_timeout, Duration::from_secs(10));
    }

    /// Helper: create a mock server that sends a canned response for every
    /// request it receives (reads header + payload, then writes `response`).
    async fn mock_server(response: Vec<u8>) -> (tokio::net::TcpListener, std::net::SocketAddr) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                let resp = response.clone();
                tokio::spawn(async move {
                    loop {
                        // Read the 16-byte header.
                        let mut header = [0u8; HEADER_SIZE];
                        if sock.read_exact(&mut header).await.is_err() {
                            break;
                        }
                        let payload_len =
                            u32::from_le_bytes([header[12], header[13], header[14], header[15]]);
                        // Drain the payload.
                        let mut discard = vec![0u8; payload_len as usize];
                        if payload_len > 0 {
                            if sock.read_exact(&mut discard).await.is_err() {
                                break;
                            }
                        }
                        // Write our canned response.
                        if sock.write_all(&resp).await.is_err() {
                            break;
                        }
                        if sock.flush().await.is_err() {
                            break;
                        }
                    }
                });
            }
        });
        // Return the listener so the task does not get dropped.
        // Actually, we already spawned the loop; just return addr.
        // We need to keep the listener alive, but we moved it into the spawn.
        // Re-bind to get a new listener just for addr; let the spawned task
        // own the real one.
        // Trick: we already have the addr. The spawned task owns the listener.
        // We cannot return the listener because it was moved. Return a dummy.
        let dummy = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        (dummy, addr)
    }

    #[tokio::test]
    async fn test_ping() {
        // Build a Pong response for request_id=1.
        let pong = Message::new(MessageType::Pong, 1, Vec::new());
        let pong_bytes = protocol::encode(&pong).unwrap();

        let (_listener, addr) = mock_server(pong_bytes).await;

        let config = ConnectionConfig {
            host: addr.ip().to_string(),
            port: addr.port(),
            ..Default::default()
        };
        let conn = Connection::connect(config).await.unwrap();
        let rtt = conn.ping().await.unwrap();
        // RTT should be very short for localhost.
        assert!(rtt < Duration::from_secs(1));
    }

    #[tokio::test]
    async fn test_get_found() {
        // Build a GetResponse with value "hello".
        let mut payload = vec![1u8]; // found
        protocol::write_bytes(&mut payload, b"hello");
        let resp = Message::new(MessageType::GetResponse, 1, payload);
        let resp_bytes = protocol::encode(&resp).unwrap();

        let (_listener, addr) = mock_server(resp_bytes).await;

        let config = ConnectionConfig {
            host: addr.ip().to_string(),
            port: addr.port(),
            ..Default::default()
        };
        let conn = Connection::connect(config).await.unwrap();
        let result = conn.get("mykey").await.unwrap();
        assert_eq!(result, Some(b"hello".to_vec()));
    }

    #[tokio::test]
    async fn test_get_not_found() {
        let resp = Message::new(MessageType::GetResponse, 1, vec![0u8]);
        let resp_bytes = protocol::encode(&resp).unwrap();

        let (_listener, addr) = mock_server(resp_bytes).await;

        let config = ConnectionConfig {
            host: addr.ip().to_string(),
            port: addr.port(),
            ..Default::default()
        };
        let conn = Connection::connect(config).await.unwrap();
        let result = conn.get("nope").await.unwrap();
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn test_put() {
        let resp = Message::new(MessageType::SetResponse, 1, vec![1u8]);
        let resp_bytes = protocol::encode(&resp).unwrap();

        let (_listener, addr) = mock_server(resp_bytes).await;

        let config = ConnectionConfig {
            host: addr.ip().to_string(),
            port: addr.port(),
            ..Default::default()
        };
        let conn = Connection::connect(config).await.unwrap();
        let ok = conn.put("key", b"value", None).await.unwrap();
        assert!(ok);
    }

    #[tokio::test]
    async fn test_delete() {
        let resp = Message::new(MessageType::DeleteResponse, 1, vec![1u8]);
        let resp_bytes = protocol::encode(&resp).unwrap();

        let (_listener, addr) = mock_server(resp_bytes).await;

        let config = ConnectionConfig {
            host: addr.ip().to_string(),
            port: addr.port(),
            ..Default::default()
        };
        let conn = Connection::connect(config).await.unwrap();
        let existed = conn.delete("key").await.unwrap();
        assert!(existed);
    }

    #[tokio::test]
    async fn test_server_error_response() {
        let mut err_payload = Vec::new();
        protocol::write_string(&mut err_payload, "INTERNAL");
        protocol::write_string(&mut err_payload, "something went wrong");
        let resp = Message::new(MessageType::Error, 1, err_payload);
        let resp_bytes = protocol::encode(&resp).unwrap();

        let (_listener, addr) = mock_server(resp_bytes).await;

        let config = ConnectionConfig {
            host: addr.ip().to_string(),
            port: addr.port(),
            ..Default::default()
        };
        let conn = Connection::connect(config).await.unwrap();
        let err = conn.get("key").await.unwrap_err();
        match err {
            ClientError::ServerError { code, message } => {
                assert_eq!(code, "INTERNAL");
                assert_eq!(message, "something went wrong");
            }
            other => panic!("expected ServerError, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_connect_refused() {
        let config = ConnectionConfig {
            host: "127.0.0.1".to_string(),
            port: 1, // unlikely to be open
            connect_timeout: Duration::from_millis(200),
            ..Default::default()
        };
        let err = Connection::connect(config).await.unwrap_err();
        assert!(
            matches!(
                err,
                ClientError::ConnectionFailed { .. } | ClientError::Timeout(_)
            ),
            "expected ConnectionFailed or Timeout, got {:?}",
            err
        );
    }
}
