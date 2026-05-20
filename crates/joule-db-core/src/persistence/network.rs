//! Platform-agnostic network backend traits
//!
//! Provides abstractions for network communication across platforms:
//! - Native: TCP, Unix sockets
//! - Browser: WebSocket, WebRTC
//! - Embedded: UART, SPI, I2C
//!
//! ## Protocol
//!
//! JouleDB uses a simple binary protocol inspired by Redis RESP but optimized
//! for database operations and GPU batch processing.
//!
//! ```text
//! +--------+--------+--------+--------+--------+--------+
//! | Magic  | Flags  | Length |  OpCode  |    Payload    |
//! | 2 bytes| 1 byte | 4 bytes| 2 bytes  |    N bytes    |
//! +--------+--------+--------+--------+--------+--------+
//! ```

use crate::error::StorageError;
use std::fmt;

/// Protocol magic number: "WD"
pub const PROTOCOL_MAGIC: [u8; 2] = [0x57, 0x44];

/// Protocol version
pub const PROTOCOL_VERSION: u8 = 1;

/// Maximum message size (16MB)
pub const MAX_MESSAGE_SIZE: usize = 16 * 1024 * 1024;

/// Header size: magic(2) + flags(1) + length(4) + opcode(2)
pub const HEADER_SIZE: usize = 9;

/// Message flags
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MessageFlags(u8);

impl MessageFlags {
    /// Request message
    pub const REQUEST: Self = Self(0x00);
    /// Response message
    pub const RESPONSE: Self = Self(0x01);
    /// Error response
    pub const ERROR: Self = Self(0x02);
    /// Compressed payload
    pub const COMPRESSED: Self = Self(0x04);
    /// Encrypted payload
    pub const ENCRYPTED: Self = Self(0x08);
    /// Batch operation
    pub const BATCH: Self = Self(0x10);
    /// Requires acknowledgment
    pub const REQUIRES_ACK: Self = Self(0x20);

    /// Create from raw value
    pub fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// Get raw value
    pub fn bits(self) -> u8 {
        self.0
    }

    /// Check if flag is set
    pub fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Set a flag
    pub fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }

    /// Check if this is a response
    pub fn is_response(self) -> bool {
        self.contains(Self::RESPONSE)
    }

    /// Check if this is an error
    pub fn is_error(self) -> bool {
        self.contains(Self::ERROR)
    }
}

/// Operation codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum OpCode {
    // Connection management (0x00xx)
    /// Ping/keepalive
    Ping = 0x0001,
    /// Pong response
    Pong = 0x0002,
    /// Authentication
    Auth = 0x0003,
    /// Close connection
    Close = 0x0004,

    // Key-value operations (0x01xx)
    /// Get value
    Get = 0x0101,
    /// Set value
    Put = 0x0102,
    /// Delete key
    Delete = 0x0103,
    /// Check if key exists
    Exists = 0x0104,
    /// Multi-get (batch)
    MGet = 0x0105,
    /// Multi-set (batch)
    MPut = 0x0106,
    /// Multi-delete (batch)
    MDelete = 0x0107,
    /// Scan keys by prefix
    Scan = 0x0108,

    // Transaction operations (0x02xx)
    /// Begin transaction
    TxBegin = 0x0201,
    /// Commit transaction
    TxCommit = 0x0202,
    /// Rollback transaction
    TxRollback = 0x0203,
    /// Create savepoint
    TxSavepoint = 0x0204,
    /// Rollback to savepoint
    TxRollbackTo = 0x0205,

    // Index operations (0x03xx)
    /// Create index
    CreateIndex = 0x0301,
    /// Drop index
    DropIndex = 0x0302,
    /// Index lookup
    IndexGet = 0x0303,
    /// Index range query
    IndexRange = 0x0304,

    // Admin operations (0x04xx)
    /// Get database info
    Info = 0x0401,
    /// Flush WAL
    Flush = 0x0402,
    /// Checkpoint
    Checkpoint = 0x0403,
    /// Create snapshot
    Snapshot = 0x0404,
    /// Compact database
    Compact = 0x0405,

    // GPU operations (0x05xx)
    /// GPU batch query
    GpuBatchQuery = 0x0501,
    /// GPU vector search
    GpuVectorSearch = 0x0502,
    /// GPU aggregation
    GpuAggregate = 0x0503,

    // Replication (0x06xx)
    /// Request replication stream
    ReplStream = 0x0601,
    /// Replication data
    ReplData = 0x0602,
    /// Replication ACK
    ReplAck = 0x0603,
}

impl TryFrom<u16> for OpCode {
    type Error = StorageError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            0x0001 => Ok(OpCode::Ping),
            0x0002 => Ok(OpCode::Pong),
            0x0003 => Ok(OpCode::Auth),
            0x0004 => Ok(OpCode::Close),
            0x0101 => Ok(OpCode::Get),
            0x0102 => Ok(OpCode::Put),
            0x0103 => Ok(OpCode::Delete),
            0x0104 => Ok(OpCode::Exists),
            0x0105 => Ok(OpCode::MGet),
            0x0106 => Ok(OpCode::MPut),
            0x0107 => Ok(OpCode::MDelete),
            0x0108 => Ok(OpCode::Scan),
            0x0201 => Ok(OpCode::TxBegin),
            0x0202 => Ok(OpCode::TxCommit),
            0x0203 => Ok(OpCode::TxRollback),
            0x0204 => Ok(OpCode::TxSavepoint),
            0x0205 => Ok(OpCode::TxRollbackTo),
            0x0301 => Ok(OpCode::CreateIndex),
            0x0302 => Ok(OpCode::DropIndex),
            0x0303 => Ok(OpCode::IndexGet),
            0x0304 => Ok(OpCode::IndexRange),
            0x0401 => Ok(OpCode::Info),
            0x0402 => Ok(OpCode::Flush),
            0x0403 => Ok(OpCode::Checkpoint),
            0x0404 => Ok(OpCode::Snapshot),
            0x0405 => Ok(OpCode::Compact),
            0x0501 => Ok(OpCode::GpuBatchQuery),
            0x0502 => Ok(OpCode::GpuVectorSearch),
            0x0503 => Ok(OpCode::GpuAggregate),
            0x0601 => Ok(OpCode::ReplStream),
            0x0602 => Ok(OpCode::ReplData),
            0x0603 => Ok(OpCode::ReplAck),
            _ => Err(StorageError::Backend(format!(
                "Unknown opcode: {:#06x}",
                value
            ))),
        }
    }
}

/// Protocol message
#[derive(Debug, Clone)]
pub struct Message {
    /// Message flags
    pub flags: MessageFlags,
    /// Operation code
    pub opcode: OpCode,
    /// Request ID (for matching responses)
    pub request_id: u32,
    /// Payload data
    pub payload: Vec<u8>,
}

impl Message {
    /// Create a new request message
    pub fn request(opcode: OpCode, payload: Vec<u8>) -> Self {
        Self {
            flags: MessageFlags::REQUEST,
            opcode,
            request_id: 0, // Will be set by the connection
            payload,
        }
    }

    /// Create a response message
    pub fn response(request_id: u32, opcode: OpCode, payload: Vec<u8>) -> Self {
        Self {
            flags: MessageFlags::RESPONSE,
            opcode,
            request_id,
            payload,
        }
    }

    /// Create an error response
    pub fn error(request_id: u32, error_code: u16, message: &str) -> Self {
        let mut payload = Vec::new();
        payload.extend_from_slice(&error_code.to_le_bytes());
        payload.extend_from_slice(&(message.len() as u16).to_le_bytes());
        payload.extend_from_slice(message.as_bytes());

        Self {
            flags: MessageFlags::from_bits(
                MessageFlags::RESPONSE.bits() | MessageFlags::ERROR.bits(),
            ),
            opcode: OpCode::Pong, // Error uses Pong as placeholder
            request_id,
            payload,
        }
    }

    /// Encode message to bytes
    pub fn encode(&self) -> Vec<u8> {
        let total_len = HEADER_SIZE + 4 + self.payload.len(); // +4 for request_id
        let mut buf = Vec::with_capacity(total_len);

        // Magic
        buf.extend_from_slice(&PROTOCOL_MAGIC);

        // Flags
        buf.push(self.flags.bits());

        // Length (payload + request_id)
        let payload_len = (self.payload.len() + 4) as u32;
        buf.extend_from_slice(&payload_len.to_le_bytes());

        // OpCode
        buf.extend_from_slice(&(self.opcode as u16).to_le_bytes());

        // Request ID
        buf.extend_from_slice(&self.request_id.to_le_bytes());

        // Payload
        buf.extend_from_slice(&self.payload);

        buf
    }

    /// Decode message from bytes
    pub fn decode(buf: &[u8]) -> Result<Self, StorageError> {
        if buf.len() < HEADER_SIZE {
            return Err(StorageError::Backend("Message too short".to_string()));
        }

        // Verify magic
        if &buf[0..2] != &PROTOCOL_MAGIC {
            return Err(StorageError::Backend("Invalid protocol magic".to_string()));
        }

        // Flags
        let flags = MessageFlags::from_bits(buf[2]);

        // Length
        let length = u32::from_le_bytes([buf[3], buf[4], buf[5], buf[6]]) as usize;

        if length > MAX_MESSAGE_SIZE {
            return Err(StorageError::Backend(format!(
                "Message too large: {} > {}",
                length, MAX_MESSAGE_SIZE
            )));
        }

        // OpCode
        let opcode_raw = u16::from_le_bytes([buf[7], buf[8]]);
        let opcode = OpCode::try_from(opcode_raw)?;

        if buf.len() < HEADER_SIZE + length {
            return Err(StorageError::Backend("Message truncated".to_string()));
        }

        // Request ID
        let request_id = u32::from_le_bytes([buf[9], buf[10], buf[11], buf[12]]);

        // Payload
        let payload = buf[HEADER_SIZE + 4..HEADER_SIZE + length].to_vec();

        Ok(Self {
            flags,
            opcode,
            request_id,
            payload,
        })
    }

    /// Get the total encoded size
    pub fn encoded_size(&self) -> usize {
        HEADER_SIZE + 4 + self.payload.len()
    }
}

/// Error codes for protocol errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum ErrorCode {
    /// No error
    Ok = 0x0000,
    /// Unknown error
    Unknown = 0x0001,
    /// Key not found
    NotFound = 0x0002,
    /// Key already exists
    AlreadyExists = 0x0003,
    /// Invalid request
    InvalidRequest = 0x0004,
    /// Authentication failed
    AuthFailed = 0x0005,
    /// Permission denied
    PermissionDenied = 0x0006,
    /// Transaction conflict
    TxConflict = 0x0007,
    /// Transaction timeout
    TxTimeout = 0x0008,
    /// Server overloaded
    Overloaded = 0x0009,
    /// Internal error
    Internal = 0x000A,
    /// Not implemented
    NotImplemented = 0x000B,
}

/// Connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConnectionState {
    /// Not connected
    #[default]
    Disconnected,
    /// Connection in progress
    Connecting,
    /// Connected, not authenticated
    Connected,
    /// Connected and authenticated
    Authenticated,
    /// Connection closing
    Closing,
}

/// Connection statistics
#[derive(Debug, Clone, Default)]
pub struct ConnectionStats {
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Total bytes received
    pub bytes_received: u64,
    /// Total messages sent
    pub messages_sent: u64,
    /// Total messages received
    pub messages_received: u64,
    /// Connection uptime in milliseconds
    pub uptime_ms: u64,
    /// Average latency in microseconds
    pub avg_latency_us: u64,
}

/// Network backend trait for synchronous I/O
///
/// Implemented by platform-specific code:
/// - Native: TCP, Unix sockets
/// - Embedded: UART, SPI
pub trait NetworkBackend: Send + Sync {
    /// Connect to a server
    fn connect(&mut self, address: &str) -> Result<(), StorageError>;

    /// Disconnect from server
    fn disconnect(&mut self) -> Result<(), StorageError>;

    /// Send a message
    fn send(&mut self, message: &Message) -> Result<(), StorageError>;

    /// Receive a message (blocking)
    fn receive(&mut self) -> Result<Message, StorageError>;

    /// Receive with timeout
    fn receive_timeout(&mut self, timeout_ms: u32) -> Result<Option<Message>, StorageError>;

    /// Get connection state
    fn state(&self) -> ConnectionState;

    /// Get connection statistics
    fn stats(&self) -> ConnectionStats;

    /// Check if connected
    fn is_connected(&self) -> bool {
        matches!(
            self.state(),
            ConnectionState::Connected | ConnectionState::Authenticated
        )
    }
}

/// Async network backend trait
///
/// For platforms requiring async I/O (browser WebSocket, async TCP).
#[cfg(feature = "async")]
pub trait AsyncNetworkBackend: Send + Sync {
    /// Connect to a server asynchronously
    fn connect(
        &mut self,
        address: &str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), StorageError>> + Send + '_>>;

    /// Disconnect asynchronously
    fn disconnect(
        &mut self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), StorageError>> + Send + '_>>;

    /// Send a message asynchronously
    fn send(
        &mut self,
        message: &Message,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), StorageError>> + Send + '_>>;

    /// Receive a message asynchronously
    fn receive(
        &mut self,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Message, StorageError>> + Send + '_>,
    >;

    /// Get connection state
    fn state(&self) -> ConnectionState;

    /// Get connection statistics
    fn stats(&self) -> ConnectionStats;
}

/// Server listener trait
///
/// For accepting incoming connections.
pub trait NetworkListener: Send + Sync {
    /// The connection type returned by accept
    type Connection: NetworkBackend;

    /// Bind to an address
    fn bind(&mut self, address: &str) -> Result<(), StorageError>;

    /// Accept a connection (blocking)
    fn accept(&mut self) -> Result<Self::Connection, StorageError>;

    /// Close the listener
    fn close(&mut self) -> Result<(), StorageError>;

    /// Get the bound address
    fn local_address(&self) -> Result<String, StorageError>;
}

/// Helper to encode a key-value pair for PUT operations
pub fn encode_key_value(key: &[u8], value: &[u8]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(8 + key.len() + value.len());
    payload.extend_from_slice(&(key.len() as u32).to_le_bytes());
    payload.extend_from_slice(key);
    payload.extend_from_slice(&(value.len() as u32).to_le_bytes());
    payload.extend_from_slice(value);
    payload
}

/// Helper to decode a key-value pair
pub fn decode_key_value(payload: &[u8]) -> Result<(&[u8], &[u8]), StorageError> {
    if payload.len() < 8 {
        return Err(StorageError::Backend("Payload too short".to_string()));
    }

    let key_len = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;

    if payload.len() < 4 + key_len + 4 {
        return Err(StorageError::Backend("Payload truncated".to_string()));
    }

    let key = &payload[4..4 + key_len];
    let value_len = u32::from_le_bytes([
        payload[4 + key_len],
        payload[5 + key_len],
        payload[6 + key_len],
        payload[7 + key_len],
    ]) as usize;

    if payload.len() < 8 + key_len + value_len {
        return Err(StorageError::Backend("Value truncated".to_string()));
    }

    let value = &payload[8 + key_len..8 + key_len + value_len];
    Ok((key, value))
}

/// Helper to encode multiple keys for MGET operations
pub fn encode_keys(keys: &[&[u8]]) -> Vec<u8> {
    let total_size: usize = 4 + keys.iter().map(|k| 4 + k.len()).sum::<usize>();
    let mut payload = Vec::with_capacity(total_size);

    payload.extend_from_slice(&(keys.len() as u32).to_le_bytes());
    for key in keys {
        payload.extend_from_slice(&(key.len() as u32).to_le_bytes());
        payload.extend_from_slice(key);
    }

    payload
}

/// Helper to encode multiple key-value pairs for MPUT operations
pub fn encode_key_values(pairs: &[(&[u8], &[u8])]) -> Vec<u8> {
    let total_size: usize = 4 + pairs
        .iter()
        .map(|(k, v)| 8 + k.len() + v.len())
        .sum::<usize>();
    let mut payload = Vec::with_capacity(total_size);

    payload.extend_from_slice(&(pairs.len() as u32).to_le_bytes());
    for (key, value) in pairs {
        payload.extend_from_slice(&(key.len() as u32).to_le_bytes());
        payload.extend_from_slice(key);
        payload.extend_from_slice(&(value.len() as u32).to_le_bytes());
        payload.extend_from_slice(value);
    }

    payload
}

impl fmt::Display for OpCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpCode::Ping => write!(f, "PING"),
            OpCode::Pong => write!(f, "PONG"),
            OpCode::Auth => write!(f, "AUTH"),
            OpCode::Close => write!(f, "CLOSE"),
            OpCode::Get => write!(f, "GET"),
            OpCode::Put => write!(f, "PUT"),
            OpCode::Delete => write!(f, "DELETE"),
            OpCode::Exists => write!(f, "EXISTS"),
            OpCode::MGet => write!(f, "MGET"),
            OpCode::MPut => write!(f, "MPUT"),
            OpCode::MDelete => write!(f, "MDELETE"),
            OpCode::Scan => write!(f, "SCAN"),
            OpCode::TxBegin => write!(f, "TX_BEGIN"),
            OpCode::TxCommit => write!(f, "TX_COMMIT"),
            OpCode::TxRollback => write!(f, "TX_ROLLBACK"),
            OpCode::TxSavepoint => write!(f, "TX_SAVEPOINT"),
            OpCode::TxRollbackTo => write!(f, "TX_ROLLBACK_TO"),
            OpCode::CreateIndex => write!(f, "CREATE_INDEX"),
            OpCode::DropIndex => write!(f, "DROP_INDEX"),
            OpCode::IndexGet => write!(f, "INDEX_GET"),
            OpCode::IndexRange => write!(f, "INDEX_RANGE"),
            OpCode::Info => write!(f, "INFO"),
            OpCode::Flush => write!(f, "FLUSH"),
            OpCode::Checkpoint => write!(f, "CHECKPOINT"),
            OpCode::Snapshot => write!(f, "SNAPSHOT"),
            OpCode::Compact => write!(f, "COMPACT"),
            OpCode::GpuBatchQuery => write!(f, "GPU_BATCH_QUERY"),
            OpCode::GpuVectorSearch => write!(f, "GPU_VECTOR_SEARCH"),
            OpCode::GpuAggregate => write!(f, "GPU_AGGREGATE"),
            OpCode::ReplStream => write!(f, "REPL_STREAM"),
            OpCode::ReplData => write!(f, "REPL_DATA"),
            OpCode::ReplAck => write!(f, "REPL_ACK"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_encode_decode() {
        let msg = Message::request(OpCode::Get, b"test_key".to_vec());
        let encoded = msg.encode();
        let decoded = Message::decode(&encoded).unwrap();

        assert_eq!(decoded.opcode, OpCode::Get);
        assert_eq!(decoded.payload, b"test_key");
        assert!(!decoded.flags.is_response());
    }

    #[test]
    fn test_message_response() {
        let msg = Message::response(42, OpCode::Get, b"test_value".to_vec());
        let encoded = msg.encode();
        let decoded = Message::decode(&encoded).unwrap();

        assert_eq!(decoded.request_id, 42);
        assert_eq!(decoded.opcode, OpCode::Get);
        assert!(decoded.flags.is_response());
    }

    #[test]
    fn test_message_error() {
        let msg = Message::error(42, ErrorCode::NotFound as u16, "Key not found");
        let encoded = msg.encode();
        let decoded = Message::decode(&encoded).unwrap();

        assert!(decoded.flags.is_error());
    }

    #[test]
    fn test_encode_decode_key_value() {
        let key = b"my_key";
        let value = b"my_value";
        let encoded = encode_key_value(key, value);
        let (dec_key, dec_value) = decode_key_value(&encoded).unwrap();

        assert_eq!(dec_key, key);
        assert_eq!(dec_value, value);
    }

    #[test]
    fn test_encode_keys() {
        let keys: Vec<&[u8]> = vec![b"key1", b"key2", b"key3"];
        let encoded = encode_keys(&keys);

        // Verify count
        let count = u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]) as usize;
        assert_eq!(count, 3);
    }

    #[test]
    fn test_message_flags() {
        let mut flags = MessageFlags::REQUEST;
        assert!(!flags.is_response());

        flags.insert(MessageFlags::RESPONSE);
        assert!(flags.is_response());

        flags.insert(MessageFlags::COMPRESSED);
        assert!(flags.contains(MessageFlags::COMPRESSED));
    }

    #[test]
    fn test_opcode_display() {
        assert_eq!(format!("{}", OpCode::Get), "GET");
        assert_eq!(format!("{}", OpCode::GpuBatchQuery), "GPU_BATCH_QUERY");
    }
}
