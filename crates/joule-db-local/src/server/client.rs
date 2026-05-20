//! TCP client for JouleDB binary protocol
//!
//! This module provides a client that can connect to a JouleDB TCP server
//! and perform database operations using the binary protocol.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use joule_db_local::server::TcpClient;
//!
//! #[tokio::main]
//! async fn main() {
//!     let mut client = TcpClient::connect("127.0.0.1:6380").await.unwrap();
//!     
//!     // Simple operations
//!     client.put(b"key", b"value").await.unwrap();
//!     let value = client.get(b"key").await.unwrap();
//!     
//!     // Batch operations
//!     let values = client.mget(&[b"key1", b"key2"]).await.unwrap();
//!     
//!     client.close().await.unwrap();
//! }
//! ```

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::timeout;

use joule_db_core::StorageError;
use joule_db_core::persistence::network::{
    ErrorCode, HEADER_SIZE, Message, OpCode, PROTOCOL_MAGIC, encode_key_value, encode_key_values,
    encode_keys,
};

/// TCP client configuration
#[derive(Debug, Clone)]
pub struct TcpClientConfig {
    /// Connection timeout in milliseconds
    pub connect_timeout_ms: u64,
    /// Read timeout in milliseconds
    pub read_timeout_ms: u64,
    /// Write timeout in milliseconds
    pub write_timeout_ms: u64,
    /// Read buffer size
    pub read_buffer_size: usize,
    /// Enable TCP nodelay
    pub tcp_nodelay: bool,
}

impl Default for TcpClientConfig {
    fn default() -> Self {
        Self {
            connect_timeout_ms: 5000,
            read_timeout_ms: 30000,
            write_timeout_ms: 30000,
            read_buffer_size: 64 * 1024,
            tcp_nodelay: true,
        }
    }
}

/// TCP client for JouleDB
pub struct TcpClient {
    /// TCP stream
    stream: TcpStream,
    /// Peer address
    peer_addr: SocketAddr,
    /// Configuration
    config: TcpClientConfig,
    /// Request ID counter
    next_request_id: AtomicU32,
    /// Read buffer
    read_buf: Vec<u8>,
}

impl TcpClient {
    /// Connect to a JouleDB server
    pub async fn connect(addr: &str) -> Result<Self, StorageError> {
        Self::connect_with_config(addr, TcpClientConfig::default()).await
    }

    /// Connect with custom configuration
    pub async fn connect_with_config(
        addr: &str,
        config: TcpClientConfig,
    ) -> Result<Self, StorageError> {
        let connect_timeout = Duration::from_millis(config.connect_timeout_ms);

        let stream = timeout(connect_timeout, TcpStream::connect(addr))
            .await
            .map_err(|_| StorageError::Backend("Connection timeout".to_string()))?
            .map_err(|e| StorageError::Backend(format!("Connect failed: {}", e)))?;

        if config.tcp_nodelay {
            let _ = stream.set_nodelay(true);
        }

        let peer_addr = stream
            .peer_addr()
            .map_err(|e| StorageError::Backend(format!("Failed to get peer addr: {}", e)))?;

        Ok(Self {
            stream,
            peer_addr,
            config: config.clone(),
            next_request_id: AtomicU32::new(1),
            read_buf: vec![0u8; config.read_buffer_size],
        })
    }

    /// Get the server address
    pub fn peer_addr(&self) -> SocketAddr {
        self.peer_addr
    }

    /// Send a ping and wait for pong
    pub async fn ping(&mut self) -> Result<Duration, StorageError> {
        let start = std::time::Instant::now();
        let response = self.send_request(OpCode::Ping, vec![]).await?;

        if response.opcode != OpCode::Pong {
            return Err(StorageError::Backend("Expected PONG response".to_string()));
        }

        Ok(start.elapsed())
    }

    /// Get a value by key
    pub async fn get(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>, StorageError> {
        let response = self.send_request(OpCode::Get, key.to_vec()).await?;

        if response.flags.is_error() {
            // Check if it's a NotFound error
            if response.payload.len() >= 2 {
                let error_code = u16::from_le_bytes([response.payload[0], response.payload[1]]);
                if error_code == ErrorCode::NotFound as u16 {
                    return Ok(None);
                }
            }
            return Err(StorageError::Backend("Get failed".to_string()));
        }

        Ok(Some(response.payload))
    }

    /// Set a value by key
    pub async fn put(&mut self, key: &[u8], value: &[u8]) -> Result<(), StorageError> {
        let payload = encode_key_value(key, value);
        let response = self.send_request(OpCode::Put, payload).await?;

        if response.flags.is_error() {
            return Err(StorageError::Backend("Put failed".to_string()));
        }

        Ok(())
    }

    /// Delete a key
    pub async fn delete(&mut self, key: &[u8]) -> Result<bool, StorageError> {
        let response = self.send_request(OpCode::Delete, key.to_vec()).await?;

        if response.flags.is_error() {
            return Err(StorageError::Backend("Delete failed".to_string()));
        }

        Ok(response.payload.first().copied() == Some(1))
    }

    /// Check if a key exists
    pub async fn exists(&mut self, key: &[u8]) -> Result<bool, StorageError> {
        let response = self.send_request(OpCode::Exists, key.to_vec()).await?;

        if response.flags.is_error() {
            return Err(StorageError::Backend("Exists failed".to_string()));
        }

        Ok(response.payload.first().copied() == Some(1))
    }

    /// Get multiple values at once
    pub async fn mget(&mut self, keys: &[&[u8]]) -> Result<Vec<Option<Vec<u8>>>, StorageError> {
        let payload = encode_keys(keys);
        let response = self.send_request(OpCode::MGet, payload).await?;

        if response.flags.is_error() {
            return Err(StorageError::Backend("MGet failed".to_string()));
        }

        // Parse results
        let mut results = Vec::with_capacity(keys.len());
        let mut offset = 0;

        for _ in 0..keys.len() {
            if offset >= response.payload.len() {
                break;
            }

            let status = response.payload[offset];
            offset += 1;

            match status {
                1 => {
                    // Found
                    if offset + 4 > response.payload.len() {
                        break;
                    }
                    let value_len = u32::from_le_bytes([
                        response.payload[offset],
                        response.payload[offset + 1],
                        response.payload[offset + 2],
                        response.payload[offset + 3],
                    ]) as usize;
                    offset += 4;

                    if offset + value_len > response.payload.len() {
                        break;
                    }
                    let value = response.payload[offset..offset + value_len].to_vec();
                    offset += value_len;
                    results.push(Some(value));
                }
                0 => {
                    // Not found
                    results.push(None);
                }
                _ => {
                    // Error
                    results.push(None);
                }
            }
        }

        Ok(results)
    }

    /// Set multiple values at once
    pub async fn mput(&mut self, pairs: &[(&[u8], &[u8])]) -> Result<Vec<bool>, StorageError> {
        let payload = encode_key_values(pairs);
        let response = self.send_request(OpCode::MPut, payload).await?;

        if response.flags.is_error() {
            return Err(StorageError::Backend("MPut failed".to_string()));
        }

        Ok(response.payload.iter().map(|&b| b == 1).collect())
    }

    /// Flush WAL to disk
    pub async fn flush(&mut self) -> Result<(), StorageError> {
        let response = self.send_request(OpCode::Flush, vec![]).await?;

        if response.flags.is_error() {
            return Err(StorageError::Backend("Flush failed".to_string()));
        }

        Ok(())
    }

    /// Get server info
    pub async fn info(&mut self) -> Result<String, StorageError> {
        let response = self.send_request(OpCode::Info, vec![]).await?;

        if response.flags.is_error() {
            return Err(StorageError::Backend("Info failed".to_string()));
        }

        String::from_utf8(response.payload)
            .map_err(|_| StorageError::Backend("Invalid UTF-8 in info".to_string()))
    }

    /// Close the connection
    pub async fn close(&mut self) -> Result<(), StorageError> {
        let _ = self.send_request(OpCode::Close, vec![]).await;
        self.stream
            .shutdown()
            .await
            .map_err(|e| StorageError::Backend(format!("Shutdown failed: {}", e)))
    }

    /// Send a request and wait for response
    async fn send_request(
        &mut self,
        opcode: OpCode,
        payload: Vec<u8>,
    ) -> Result<Message, StorageError> {
        let request_id = self.next_request_id.fetch_add(1, Ordering::SeqCst);

        let mut message = Message::request(opcode, payload);
        message.request_id = request_id;

        let encoded = message.encode();

        // Send with timeout
        let write_timeout = Duration::from_millis(self.config.write_timeout_ms);
        timeout(write_timeout, self.stream.write_all(&encoded))
            .await
            .map_err(|_| StorageError::Backend("Write timeout".to_string()))?
            .map_err(|e| StorageError::Backend(format!("Write failed: {}", e)))?;

        // Read response with timeout
        let read_timeout = Duration::from_millis(self.config.read_timeout_ms);
        let n = timeout(read_timeout, self.stream.read(&mut self.read_buf))
            .await
            .map_err(|_| StorageError::Backend("Read timeout".to_string()))?
            .map_err(|e| StorageError::Backend(format!("Read failed: {}", e)))?;

        if n == 0 {
            return Err(StorageError::Backend("Connection closed".to_string()));
        }

        // Parse response
        if n < HEADER_SIZE {
            return Err(StorageError::Backend("Response too short".to_string()));
        }

        if &self.read_buf[0..2] != &PROTOCOL_MAGIC {
            return Err(StorageError::Backend("Invalid protocol magic".to_string()));
        }

        Message::decode(&self.read_buf[..n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_config_default() {
        let config = TcpClientConfig::default();
        assert_eq!(config.connect_timeout_ms, 5000);
        assert_eq!(config.read_timeout_ms, 30000);
        assert!(config.tcp_nodelay);
    }

    // Integration tests would require a running server
    // They are better placed in the tests/ directory
}
