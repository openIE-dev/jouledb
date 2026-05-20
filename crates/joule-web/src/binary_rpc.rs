//! Binary RPC protocol — length-prefixed framing, method dispatch, correlation.
//!
//! Pure-Rust binary RPC protocol with compact wire format. Messages carry a
//! `method_id` (u32), `correlation_id` for request/response matching, and an
//! opaque payload. The [`RpcCodec`] handles length-prefixed framing for
//! encoding/decoding from byte buffers. [`RpcServer`] dispatches incoming
//! messages to registered handlers.

use std::collections::HashMap;
use std::fmt;

// ── Message Types ──────────────────────────────────────────────

/// Classifies the kind of RPC message on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MessageKind {
    Request,
    Response,
    Notification,
    Error,
}

impl MessageKind {
    fn tag(self) -> u8 {
        match self {
            Self::Request => 0x01,
            Self::Response => 0x02,
            Self::Notification => 0x03,
            Self::Error => 0x04,
        }
    }

    fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0x01 => Some(Self::Request),
            0x02 => Some(Self::Response),
            0x03 => Some(Self::Notification),
            0x04 => Some(Self::Error),
            _ => None,
        }
    }
}

impl fmt::Display for MessageKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Request => write!(f, "Request"),
            Self::Response => write!(f, "Response"),
            Self::Notification => write!(f, "Notification"),
            Self::Error => write!(f, "Error"),
        }
    }
}

// ── Status Codes ───────────────────────────────────────────────

/// Standard RPC status codes for error responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StatusCode {
    Ok = 0,
    InvalidMethod = 1,
    InvalidPayload = 2,
    InternalError = 3,
    Timeout = 4,
    Unavailable = 5,
    PayloadTooLarge = 6,
    Unauthenticated = 7,
}

impl StatusCode {
    pub fn as_u16(self) -> u16 {
        self as u16
    }

    pub fn from_u16(v: u16) -> Self {
        match v {
            0 => Self::Ok,
            1 => Self::InvalidMethod,
            2 => Self::InvalidPayload,
            3 => Self::InternalError,
            4 => Self::Timeout,
            5 => Self::Unavailable,
            6 => Self::PayloadTooLarge,
            7 => Self::Unauthenticated,
            _ => Self::InternalError,
        }
    }
}

impl fmt::Display for StatusCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}({})", self, self.as_u16())
    }
}

// ── RPC Message ────────────────────────────────────────────────

/// A single RPC message on the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RpcMessage {
    pub kind: MessageKind,
    pub method_id: u32,
    pub correlation_id: u64,
    pub status: StatusCode,
    pub payload: Vec<u8>,
}

impl RpcMessage {
    /// Build a request message.
    pub fn request(method_id: u32, correlation_id: u64, payload: Vec<u8>) -> Self {
        Self { kind: MessageKind::Request, method_id, correlation_id, status: StatusCode::Ok, payload }
    }

    /// Build a response message.
    pub fn response(correlation_id: u64, payload: Vec<u8>) -> Self {
        Self { kind: MessageKind::Response, method_id: 0, correlation_id, status: StatusCode::Ok, payload }
    }

    /// Build a notification (one-way, no response expected).
    pub fn notification(method_id: u32, payload: Vec<u8>) -> Self {
        Self { kind: MessageKind::Notification, method_id, correlation_id: 0, status: StatusCode::Ok, payload }
    }

    /// Build an error response.
    pub fn error(correlation_id: u64, status: StatusCode, detail: Vec<u8>) -> Self {
        Self { kind: MessageKind::Error, method_id: 0, correlation_id, status, payload: detail }
    }

    /// Total wire size (header + payload).
    pub fn wire_size(&self) -> usize {
        // 4 (length) + 1 (kind) + 4 (method_id) + 8 (correlation_id) + 2 (status) + payload
        19 + self.payload.len()
    }
}

impl fmt::Display for RpcMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[method={}, corr={}, status={}, {}B payload]",
            self.kind, self.method_id, self.correlation_id, self.status, self.payload.len())
    }
}

// ── Codec Error ────────────────────────────────────────────────

/// Errors that occur during encode/decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodecError {
    BufferTooSmall { needed: usize, available: usize },
    InvalidKindTag(u8),
    IncompleteFrame { needed: usize, available: usize },
    PayloadTooLarge { size: usize, limit: usize },
    CorruptedFrame,
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferTooSmall { needed, available } =>
                write!(f, "buffer too small: need {needed}, have {available}"),
            Self::InvalidKindTag(t) => write!(f, "invalid message kind tag: 0x{t:02x}"),
            Self::IncompleteFrame { needed, available } =>
                write!(f, "incomplete frame: need {needed}, have {available}"),
            Self::PayloadTooLarge { size, limit } =>
                write!(f, "payload too large: {size} > {limit}"),
            Self::CorruptedFrame => write!(f, "corrupted frame data"),
        }
    }
}

// ── RPC Codec ──────────────────────────────────────────────────

/// Length-prefixed codec for encoding/decoding [`RpcMessage`] to/from bytes.
///
/// Wire format:
/// ```text
/// [4B length][1B kind][4B method_id][8B correlation_id][2B status][payload...]
/// ```
#[derive(Debug, Clone)]
pub struct RpcCodec {
    max_payload_size: usize,
    bytes_encoded: u64,
    bytes_decoded: u64,
    messages_encoded: u64,
    messages_decoded: u64,
}

impl RpcCodec {
    const HEADER_SIZE: usize = 15; // 1 + 4 + 8 + 2

    pub fn new(max_payload_size: usize) -> Self {
        Self {
            max_payload_size,
            bytes_encoded: 0,
            bytes_decoded: 0,
            messages_encoded: 0,
            messages_decoded: 0,
        }
    }

    /// Encode a message into bytes (length-prefixed frame).
    pub fn encode(&mut self, msg: &RpcMessage) -> Result<Vec<u8>, CodecError> {
        if msg.payload.len() > self.max_payload_size {
            return Err(CodecError::PayloadTooLarge {
                size: msg.payload.len(),
                limit: self.max_payload_size,
            });
        }
        let frame_len = Self::HEADER_SIZE + msg.payload.len();
        let mut buf = Vec::with_capacity(4 + frame_len);
        // Length prefix (4 bytes, big-endian)
        buf.extend_from_slice(&(frame_len as u32).to_be_bytes());
        // Kind tag
        buf.push(msg.kind.tag());
        // Method ID
        buf.extend_from_slice(&msg.method_id.to_be_bytes());
        // Correlation ID
        buf.extend_from_slice(&msg.correlation_id.to_be_bytes());
        // Status
        buf.extend_from_slice(&msg.status.as_u16().to_be_bytes());
        // Payload
        buf.extend_from_slice(&msg.payload);

        self.bytes_encoded += buf.len() as u64;
        self.messages_encoded += 1;
        Ok(buf)
    }

    /// Decode one message from a byte buffer. Returns the message and bytes consumed.
    pub fn decode(&mut self, data: &[u8]) -> Result<(RpcMessage, usize), CodecError> {
        if data.len() < 4 {
            return Err(CodecError::IncompleteFrame { needed: 4, available: data.len() });
        }
        let frame_len = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let total = 4 + frame_len;
        if data.len() < total {
            return Err(CodecError::IncompleteFrame { needed: total, available: data.len() });
        }
        if frame_len < Self::HEADER_SIZE {
            return Err(CodecError::CorruptedFrame);
        }
        let body = &data[4..total];
        let kind = MessageKind::from_tag(body[0])
            .ok_or(CodecError::InvalidKindTag(body[0]))?;
        let method_id = u32::from_be_bytes([body[1], body[2], body[3], body[4]]);
        let correlation_id = u64::from_be_bytes([
            body[5], body[6], body[7], body[8], body[9], body[10], body[11], body[12],
        ]);
        let status = StatusCode::from_u16(u16::from_be_bytes([body[13], body[14]]));
        let payload_len = frame_len - Self::HEADER_SIZE;
        if payload_len > self.max_payload_size {
            return Err(CodecError::PayloadTooLarge {
                size: payload_len,
                limit: self.max_payload_size,
            });
        }
        let payload = body[Self::HEADER_SIZE..].to_vec();

        self.bytes_decoded += total as u64;
        self.messages_decoded += 1;
        Ok((RpcMessage { kind, method_id, correlation_id, status, payload }, total))
    }

    /// Try to decode all complete messages from a buffer.
    pub fn decode_all(&mut self, data: &[u8]) -> (Vec<RpcMessage>, usize) {
        let mut messages = Vec::new();
        let mut offset = 0;
        loop {
            match self.decode(&data[offset..]) {
                Ok((msg, consumed)) => {
                    messages.push(msg);
                    offset += consumed;
                }
                Err(_) => break,
            }
        }
        (messages, offset)
    }

    pub fn bytes_encoded(&self) -> u64 { self.bytes_encoded }
    pub fn bytes_decoded(&self) -> u64 { self.bytes_decoded }
    pub fn messages_encoded(&self) -> u64 { self.messages_encoded }
    pub fn messages_decoded(&self) -> u64 { self.messages_decoded }
}

// ── Handler Result ─────────────────────────────────────────────

/// Result from an RPC handler invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandlerResult {
    pub status: StatusCode,
    pub payload: Vec<u8>,
}

/// A handler function: takes payload bytes, returns a result.
pub type HandlerFn = fn(&[u8]) -> HandlerResult;

// ── Method Registry ────────────────────────────────────────────

/// Maps method IDs to handler functions.
#[derive(Debug, Clone)]
pub struct MethodRegistry {
    handlers: HashMap<u32, HandlerFn>,
    names: HashMap<u32, String>,
}

impl MethodRegistry {
    pub fn new() -> Self {
        Self { handlers: HashMap::new(), names: HashMap::new() }
    }

    /// Register a handler for the given method ID.
    pub fn register(&mut self, method_id: u32, name: &str, handler: HandlerFn) {
        self.handlers.insert(method_id, handler);
        self.names.insert(method_id, name.to_string());
    }

    /// Look up a handler by method ID.
    pub fn get(&self, method_id: u32) -> Option<&HandlerFn> {
        self.handlers.get(&method_id)
    }

    /// Check if a method is registered.
    pub fn contains(&self, method_id: u32) -> bool {
        self.handlers.contains_key(&method_id)
    }

    /// Get the name of a registered method.
    pub fn name(&self, method_id: u32) -> Option<&str> {
        self.names.get(&method_id).map(|s| s.as_str())
    }

    /// Number of registered methods.
    pub fn len(&self) -> usize { self.handlers.len() }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool { self.handlers.is_empty() }

    /// List all registered method IDs.
    pub fn method_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self.handlers.keys().copied().collect();
        ids.sort();
        ids
    }
}

impl Default for MethodRegistry {
    fn default() -> Self { Self::new() }
}

// ── RPC Server ─────────────────────────────────────────────────

/// Server-side RPC dispatcher. Routes incoming messages to registered
/// handlers and produces response messages.
#[derive(Debug)]
pub struct RpcServer {
    registry: MethodRegistry,
    codec: RpcCodec,
    requests_handled: u64,
    errors_returned: u64,
    notifications_received: u64,
    pending_correlation_ids: Vec<u64>,
}

impl RpcServer {
    pub fn new(max_payload_size: usize) -> Self {
        Self {
            registry: MethodRegistry::new(),
            codec: RpcCodec::new(max_payload_size),
            requests_handled: 0,
            errors_returned: 0,
            notifications_received: 0,
            pending_correlation_ids: Vec::new(),
        }
    }

    /// Register a handler for a method ID.
    pub fn register(&mut self, method_id: u32, name: &str, handler: HandlerFn) {
        self.registry.register(method_id, name, handler);
    }

    /// Process raw incoming bytes. Returns encoded response bytes.
    pub fn process_bytes(&mut self, data: &[u8]) -> Vec<Vec<u8>> {
        let (messages, _consumed) = self.codec.decode_all(data);
        let mut responses = Vec::new();
        for msg in messages {
            if let Some(resp_bytes) = self.dispatch(msg) {
                responses.push(resp_bytes);
            }
        }
        responses
    }

    /// Dispatch a single decoded message.
    fn dispatch(&mut self, msg: RpcMessage) -> Option<Vec<u8>> {
        match msg.kind {
            MessageKind::Request => {
                self.requests_handled += 1;
                let resp = if let Some(handler) = self.registry.get(msg.method_id) {
                    let result = handler(&msg.payload);
                    if result.status == StatusCode::Ok {
                        RpcMessage::response(msg.correlation_id, result.payload)
                    } else {
                        self.errors_returned += 1;
                        RpcMessage::error(msg.correlation_id, result.status, result.payload)
                    }
                } else {
                    self.errors_returned += 1;
                    RpcMessage::error(
                        msg.correlation_id,
                        StatusCode::InvalidMethod,
                        format!("unknown method {}", msg.method_id).into_bytes(),
                    )
                };
                self.codec.encode(&resp).ok()
            }
            MessageKind::Notification => {
                self.notifications_received += 1;
                // Fire-and-forget: run handler if registered, no response
                if let Some(handler) = self.registry.get(msg.method_id) {
                    let _ = handler(&msg.payload);
                }
                None
            }
            _ => None,
        }
    }

    pub fn requests_handled(&self) -> u64 { self.requests_handled }
    pub fn errors_returned(&self) -> u64 { self.errors_returned }
    pub fn notifications_received(&self) -> u64 { self.notifications_received }
    pub fn registry(&self) -> &MethodRegistry { &self.registry }
}

// ── Request/Response Correlator ────────────────────────────────

/// Client-side correlator that tracks outstanding requests.
#[derive(Debug)]
pub struct Correlator {
    next_id: u64,
    pending: HashMap<u64, PendingRequest>,
}

/// A pending request awaiting a response.
#[derive(Debug, Clone)]
pub struct PendingRequest {
    pub method_id: u32,
    pub sent_at_ms: u64,
    pub timeout_ms: u64,
}

impl Correlator {
    pub fn new() -> Self {
        Self { next_id: 1, pending: HashMap::new() }
    }

    /// Allocate a new correlation ID and track the request.
    pub fn track(&mut self, method_id: u32, sent_at_ms: u64, timeout_ms: u64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.pending.insert(id, PendingRequest { method_id, sent_at_ms, timeout_ms });
        id
    }

    /// Complete a pending request, returning its metadata.
    pub fn complete(&mut self, correlation_id: u64) -> Option<PendingRequest> {
        self.pending.remove(&correlation_id)
    }

    /// Expire requests that have exceeded their timeout.
    pub fn expire(&mut self, now_ms: u64) -> Vec<(u64, PendingRequest)> {
        let mut expired = Vec::new();
        self.pending.retain(|&id, req| {
            if now_ms >= req.sent_at_ms + req.timeout_ms {
                expired.push((id, req.clone()));
                false
            } else {
                true
            }
        });
        expired
    }

    /// Number of pending requests.
    pub fn pending_count(&self) -> usize { self.pending.len() }

    /// Check if a correlation ID is pending.
    pub fn is_pending(&self, id: u64) -> bool { self.pending.contains_key(&id) }
}

impl Default for Correlator {
    fn default() -> Self { Self::new() }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn echo_handler(payload: &[u8]) -> HandlerResult {
        HandlerResult { status: StatusCode::Ok, payload: payload.to_vec() }
    }

    fn error_handler(_payload: &[u8]) -> HandlerResult {
        HandlerResult { status: StatusCode::InternalError, payload: b"boom".to_vec() }
    }

    #[test]
    fn encode_decode_request() {
        let mut codec = RpcCodec::new(1024);
        let msg = RpcMessage::request(42, 100, vec![1, 2, 3]);
        let bytes = codec.encode(&msg).unwrap();
        let (decoded, consumed) = codec.decode(&bytes).unwrap();
        assert_eq!(decoded, msg);
        assert_eq!(consumed, bytes.len());
    }

    #[test]
    fn encode_decode_response() {
        let mut codec = RpcCodec::new(1024);
        let msg = RpcMessage::response(200, vec![10, 20]);
        let bytes = codec.encode(&msg).unwrap();
        let (decoded, _) = codec.decode(&bytes).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn encode_decode_notification() {
        let mut codec = RpcCodec::new(1024);
        let msg = RpcMessage::notification(5, vec![0xFF]);
        let bytes = codec.encode(&msg).unwrap();
        let (decoded, _) = codec.decode(&bytes).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn encode_decode_error() {
        let mut codec = RpcCodec::new(1024);
        let msg = RpcMessage::error(300, StatusCode::Timeout, b"timed out".to_vec());
        let bytes = codec.encode(&msg).unwrap();
        let (decoded, _) = codec.decode(&bytes).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn payload_too_large_on_encode() {
        let mut codec = RpcCodec::new(4);
        let msg = RpcMessage::request(1, 1, vec![0; 10]);
        assert!(matches!(codec.encode(&msg), Err(CodecError::PayloadTooLarge { .. })));
    }

    #[test]
    fn incomplete_frame_on_short_data() {
        let mut codec = RpcCodec::new(1024);
        assert!(matches!(codec.decode(&[0, 0]), Err(CodecError::IncompleteFrame { .. })));
    }

    #[test]
    fn incomplete_frame_on_partial_body() {
        let mut codec = RpcCodec::new(1024);
        // length says 100 bytes but only provide header
        let mut data = vec![0, 0, 0, 100];
        data.extend_from_slice(&[0; 10]);
        assert!(matches!(codec.decode(&data), Err(CodecError::IncompleteFrame { .. })));
    }

    #[test]
    fn corrupted_frame_too_small_body() {
        let mut codec = RpcCodec::new(1024);
        // frame_len = 2 < HEADER_SIZE
        let data = [0, 0, 0, 2, 0, 0];
        assert!(matches!(codec.decode(&data), Err(CodecError::CorruptedFrame)));
    }

    #[test]
    fn decode_all_multiple_messages() {
        let mut codec = RpcCodec::new(1024);
        let m1 = RpcMessage::request(1, 1, vec![10]);
        let m2 = RpcMessage::request(2, 2, vec![20]);
        let b1 = codec.encode(&m1).unwrap();
        let b2 = codec.encode(&m2).unwrap();
        let mut combined = b1;
        combined.extend(b2);
        let (msgs, consumed) = codec.decode_all(&combined);
        assert_eq!(msgs.len(), 2);
        assert_eq!(consumed, combined.len());
        assert_eq!(msgs[0], m1);
        assert_eq!(msgs[1], m2);
    }

    #[test]
    fn codec_statistics() {
        let mut codec = RpcCodec::new(1024);
        let msg = RpcMessage::request(1, 1, vec![0; 10]);
        let bytes = codec.encode(&msg).unwrap();
        assert_eq!(codec.messages_encoded(), 1);
        assert!(codec.bytes_encoded() > 0);
        let _ = codec.decode(&bytes).unwrap();
        assert_eq!(codec.messages_decoded(), 1);
        assert!(codec.bytes_decoded() > 0);
    }

    #[test]
    fn method_registry_basics() {
        let mut reg = MethodRegistry::new();
        assert!(reg.is_empty());
        reg.register(1, "echo", echo_handler);
        assert_eq!(reg.len(), 1);
        assert!(reg.contains(1));
        assert!(!reg.contains(2));
        assert_eq!(reg.name(1), Some("echo"));
        assert_eq!(reg.method_ids(), vec![1]);
    }

    #[test]
    fn server_dispatches_request() {
        let mut server = RpcServer::new(1024);
        server.register(1, "echo", echo_handler);
        let mut codec = RpcCodec::new(1024);
        let req = RpcMessage::request(1, 42, vec![1, 2, 3]);
        let req_bytes = codec.encode(&req).unwrap();
        let responses = server.process_bytes(&req_bytes);
        assert_eq!(responses.len(), 1);
        let (resp, _) = codec.decode(&responses[0]).unwrap();
        assert_eq!(resp.kind, MessageKind::Response);
        assert_eq!(resp.correlation_id, 42);
        assert_eq!(resp.payload, vec![1, 2, 3]);
    }

    #[test]
    fn server_returns_error_for_unknown_method() {
        let mut server = RpcServer::new(1024);
        let mut codec = RpcCodec::new(1024);
        let req = RpcMessage::request(99, 1, vec![]);
        let req_bytes = codec.encode(&req).unwrap();
        let responses = server.process_bytes(&req_bytes);
        assert_eq!(responses.len(), 1);
        let (resp, _) = codec.decode(&responses[0]).unwrap();
        assert_eq!(resp.kind, MessageKind::Error);
        assert_eq!(resp.status, StatusCode::InvalidMethod);
    }

    #[test]
    fn server_handles_error_handler() {
        let mut server = RpcServer::new(1024);
        server.register(2, "fail", error_handler);
        let mut codec = RpcCodec::new(1024);
        let req = RpcMessage::request(2, 10, vec![]);
        let req_bytes = codec.encode(&req).unwrap();
        let responses = server.process_bytes(&req_bytes);
        let (resp, _) = codec.decode(&responses[0]).unwrap();
        assert_eq!(resp.kind, MessageKind::Error);
        assert_eq!(resp.status, StatusCode::InternalError);
        assert_eq!(server.errors_returned(), 1);
    }

    #[test]
    fn server_notification_no_response() {
        let mut server = RpcServer::new(1024);
        server.register(1, "notify", echo_handler);
        let mut codec = RpcCodec::new(1024);
        let notif = RpcMessage::notification(1, vec![5]);
        let notif_bytes = codec.encode(&notif).unwrap();
        let responses = server.process_bytes(&notif_bytes);
        assert!(responses.is_empty());
        assert_eq!(server.notifications_received(), 1);
    }

    #[test]
    fn server_statistics() {
        let mut server = RpcServer::new(1024);
        server.register(1, "echo", echo_handler);
        let mut codec = RpcCodec::new(1024);
        let req = RpcMessage::request(1, 1, vec![]);
        let bytes = codec.encode(&req).unwrap();
        server.process_bytes(&bytes);
        server.process_bytes(&bytes);
        assert_eq!(server.requests_handled(), 2);
    }

    #[test]
    fn correlator_track_and_complete() {
        let mut corr = Correlator::new();
        let id = corr.track(1, 1000, 5000);
        assert!(corr.is_pending(id));
        assert_eq!(corr.pending_count(), 1);
        let req = corr.complete(id).unwrap();
        assert_eq!(req.method_id, 1);
        assert_eq!(corr.pending_count(), 0);
    }

    #[test]
    fn correlator_expire_timed_out() {
        let mut corr = Correlator::new();
        corr.track(1, 1000, 100);
        corr.track(2, 1000, 5000);
        let expired = corr.expire(1200);
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].1.method_id, 1);
        assert_eq!(corr.pending_count(), 1);
    }

    #[test]
    fn message_wire_size() {
        let msg = RpcMessage::request(1, 1, vec![0; 100]);
        assert_eq!(msg.wire_size(), 19 + 100);
    }

    #[test]
    fn message_display() {
        let msg = RpcMessage::request(42, 7, vec![0; 5]);
        let s = format!("{}", msg);
        assert!(s.contains("Request"));
        assert!(s.contains("42"));
        assert!(s.contains("5B payload"));
    }

    #[test]
    fn status_code_roundtrip() {
        for code in [StatusCode::Ok, StatusCode::Timeout, StatusCode::PayloadTooLarge] {
            assert_eq!(StatusCode::from_u16(code.as_u16()), code);
        }
    }
}
