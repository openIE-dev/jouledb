//! Channel multiplexing — logical channels over a single connection, channel
//! create/close, backpressure, ordered delivery, and channel metadata.
//!
//! Pure-Rust multiplexer that models multiple logical streams over one
//! transport without actual I/O. Callers feed frames in and read frames out.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;

// ── Channel ID ─────────────────────────────────────────────────────

/// A logical channel identifier (u32 for compact wire representation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChannelId(pub u32);

impl fmt::Display for ChannelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ch:{}", self.0)
    }
}

// ── Frame types ────────────────────────────────────────────────────

/// Frame types in the multiplexing protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    /// Open a new channel.
    Open,
    /// Close an existing channel.
    Close,
    /// Data payload.
    Data,
    /// Flow control: window update (credit-based).
    WindowUpdate,
    /// Ping (keepalive).
    Ping,
    /// Pong (keepalive response).
    Pong,
    /// Reset a channel (error).
    Reset,
}

impl FrameType {
    pub fn to_u8(self) -> u8 {
        match self {
            Self::Open => 0,
            Self::Close => 1,
            Self::Data => 2,
            Self::WindowUpdate => 3,
            Self::Ping => 4,
            Self::Pong => 5,
            Self::Reset => 6,
        }
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Open),
            1 => Some(Self::Close),
            2 => Some(Self::Data),
            3 => Some(Self::WindowUpdate),
            4 => Some(Self::Ping),
            5 => Some(Self::Pong),
            6 => Some(Self::Reset),
            _ => None,
        }
    }
}

/// A multiplexed frame on the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MuxFrame {
    pub channel_id: ChannelId,
    pub frame_type: FrameType,
    pub payload: Vec<u8>,
    pub seq: u64,
}

impl MuxFrame {
    pub fn new(channel_id: ChannelId, frame_type: FrameType, payload: Vec<u8>, seq: u64) -> Self {
        Self { channel_id, frame_type, payload, seq }
    }

    /// Create a data frame.
    pub fn data(channel_id: ChannelId, payload: Vec<u8>, seq: u64) -> Self {
        Self::new(channel_id, FrameType::Data, payload, seq)
    }

    /// Create an open frame with optional metadata.
    pub fn open(channel_id: ChannelId, metadata: Vec<u8>) -> Self {
        Self::new(channel_id, FrameType::Open, metadata, 0)
    }

    /// Create a close frame.
    pub fn close(channel_id: ChannelId) -> Self {
        Self::new(channel_id, FrameType::Close, Vec::new(), 0)
    }

    /// Create a window update frame with the credit amount.
    pub fn window_update(channel_id: ChannelId, credit: u32) -> Self {
        Self::new(channel_id, FrameType::WindowUpdate, credit.to_be_bytes().to_vec(), 0)
    }

    /// Create a ping frame.
    pub fn ping(data: Vec<u8>) -> Self {
        Self::new(ChannelId(0), FrameType::Ping, data, 0)
    }

    /// Create a pong frame.
    pub fn pong(data: Vec<u8>) -> Self {
        Self::new(ChannelId(0), FrameType::Pong, data, 0)
    }

    /// Create a reset frame with a reason code.
    pub fn reset(channel_id: ChannelId, reason: u32) -> Self {
        Self::new(channel_id, FrameType::Reset, reason.to_be_bytes().to_vec(), 0)
    }

    /// Serialize frame to bytes: [type:1][channel_id:4][seq:8][len:4][payload:N].
    pub fn to_bytes(&self) -> Vec<u8> {
        let len = self.payload.len() as u32;
        let mut buf = Vec::with_capacity(17 + self.payload.len());
        buf.push(self.frame_type.to_u8());
        buf.extend_from_slice(&self.channel_id.0.to_be_bytes());
        buf.extend_from_slice(&self.seq.to_be_bytes());
        buf.extend_from_slice(&len.to_be_bytes());
        buf.extend_from_slice(&self.payload);
        buf
    }

    /// Parse a frame from bytes. Returns (frame, bytes_consumed) or None.
    pub fn from_bytes(data: &[u8]) -> Option<(Self, usize)> {
        if data.len() < 17 {
            return None;
        }
        let frame_type = FrameType::from_u8(data[0])?;
        let channel_id = ChannelId(u32::from_be_bytes([data[1], data[2], data[3], data[4]]));
        let seq = u64::from_be_bytes([
            data[5], data[6], data[7], data[8],
            data[9], data[10], data[11], data[12],
        ]);
        let len = u32::from_be_bytes([data[13], data[14], data[15], data[16]]) as usize;
        if data.len() < 17 + len {
            return None;
        }
        let payload = data[17..17 + len].to_vec();
        Some((Self { channel_id, frame_type, payload, seq }, 17 + len))
    }
}

// ── Channel state ──────────────────────────────────────────────────

/// State of a logical channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelState {
    Opening,
    Open,
    Closing,
    Closed,
}

/// Metadata associated with a channel.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChannelMetadata {
    pub labels: HashMap<String, String>,
}

impl ChannelMetadata {
    pub fn new() -> Self {
        Self { labels: HashMap::new() }
    }

    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    /// Serialize labels to a simple key=value\n format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut sorted: Vec<(&String, &String)> = self.labels.iter().collect();
        sorted.sort_by_key(|(k, _)| *k);
        let mut buf = String::new();
        for (k, v) in sorted {
            buf.push_str(k);
            buf.push('=');
            buf.push_str(v);
            buf.push('\n');
        }
        buf.into_bytes()
    }

    /// Parse labels from bytes.
    pub fn from_bytes(data: &[u8]) -> Self {
        let s = String::from_utf8_lossy(data);
        let mut labels = HashMap::new();
        for line in s.lines() {
            if let Some((k, v)) = line.split_once('=') {
                labels.insert(k.to_string(), v.to_string());
            }
        }
        Self { labels }
    }
}

/// Per-channel state tracked by the multiplexer.
#[derive(Debug)]
struct Channel {
    id: ChannelId,
    state: ChannelState,
    metadata: ChannelMetadata,
    /// Send window: how many bytes we are allowed to send.
    send_window: u32,
    /// Receive window: how many bytes the remote is allowed to send.
    recv_window: u32,
    /// Outbound data queue.
    send_queue: VecDeque<Vec<u8>>,
    /// Inbound data queue.
    recv_queue: VecDeque<Vec<u8>>,
    /// Next outbound sequence number.
    next_send_seq: u64,
    /// Next expected inbound sequence number.
    next_recv_seq: u64,
    /// Total bytes sent.
    bytes_sent: u64,
    /// Total bytes received.
    bytes_received: u64,
}

// ── Multiplexer config ─────────────────────────────────────────────

/// Configuration for the channel multiplexer.
#[derive(Debug, Clone)]
pub struct MuxConfig {
    /// Initial send/receive window size in bytes.
    pub initial_window_size: u32,
    /// Maximum number of concurrent channels.
    pub max_channels: usize,
    /// Maximum payload size per frame.
    pub max_frame_payload: usize,
}

impl Default for MuxConfig {
    fn default() -> Self {
        Self {
            initial_window_size: 65535,
            max_channels: 256,
            max_frame_payload: 16384,
        }
    }
}

// ── Multiplexer events ─────────────────────────────────────────────

/// Events produced by the multiplexer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MuxEvent {
    /// A new channel was opened by the remote.
    ChannelOpened { channel_id: ChannelId, metadata: ChannelMetadata },
    /// A channel was closed.
    ChannelClosed { channel_id: ChannelId },
    /// A channel was reset with a reason.
    ChannelReset { channel_id: ChannelId, reason: u32 },
    /// Data received on a channel.
    DataReceived { channel_id: ChannelId, data: Vec<u8> },
    /// Backpressure: send window exhausted on a channel.
    Backpressure { channel_id: ChannelId },
    /// Ping received.
    PingReceived { data: Vec<u8> },
}

// ── Multiplexer errors ─────────────────────────────────────────────

/// Errors from the multiplexer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MuxError {
    ChannelNotFound(ChannelId),
    ChannelNotOpen(ChannelId),
    TooManyChannels,
    ChannelAlreadyExists(ChannelId),
    PayloadTooLarge { size: usize, max: usize },
    WindowExhausted(ChannelId),
    InvalidFrame,
}

impl fmt::Display for MuxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChannelNotFound(id) => write!(f, "channel not found: {}", id),
            Self::ChannelNotOpen(id) => write!(f, "channel not open: {}", id),
            Self::TooManyChannels => write!(f, "too many channels"),
            Self::ChannelAlreadyExists(id) => write!(f, "channel already exists: {}", id),
            Self::PayloadTooLarge { size, max } => write!(f, "payload too large: {} > {}", size, max),
            Self::WindowExhausted(id) => write!(f, "send window exhausted for {}", id),
            Self::InvalidFrame => write!(f, "invalid frame"),
        }
    }
}

// ── Channel multiplexer ────────────────────────────────────────────

/// A channel multiplexer managing logical channels over a single connection.
#[derive(Debug)]
pub struct ChannelMux {
    config: MuxConfig,
    channels: BTreeMap<ChannelId, Channel>,
    /// Outbound frame queue (ready to send on the wire).
    outbound: VecDeque<MuxFrame>,
    /// Events for the application layer.
    events: Vec<MuxEvent>,
    next_channel_id: u32,
}

impl ChannelMux {
    pub fn new(config: MuxConfig) -> Self {
        Self {
            config,
            channels: BTreeMap::new(),
            outbound: VecDeque::new(),
            events: Vec::new(),
            next_channel_id: 1,
        }
    }

    /// Open a new channel. Returns the channel ID and queues an Open frame.
    pub fn open_channel(&mut self, metadata: ChannelMetadata) -> Result<ChannelId, MuxError> {
        if self.channels.len() >= self.config.max_channels {
            return Err(MuxError::TooManyChannels);
        }
        let id = ChannelId(self.next_channel_id);
        self.next_channel_id += 1;

        let meta_bytes = metadata.to_bytes();
        self.channels.insert(id, Channel {
            id,
            state: ChannelState::Open,
            metadata,
            send_window: self.config.initial_window_size,
            recv_window: self.config.initial_window_size,
            send_queue: VecDeque::new(),
            recv_queue: VecDeque::new(),
            next_send_seq: 1,
            next_recv_seq: 1,
            bytes_sent: 0,
            bytes_received: 0,
        });

        self.outbound.push_back(MuxFrame::open(id, meta_bytes));
        Ok(id)
    }

    /// Close a channel. Queues a Close frame.
    pub fn close_channel(&mut self, channel_id: ChannelId) -> Result<(), MuxError> {
        let ch = self.channels.get_mut(&channel_id)
            .ok_or(MuxError::ChannelNotFound(channel_id))?;
        ch.state = ChannelState::Closing;
        self.outbound.push_back(MuxFrame::close(channel_id));
        Ok(())
    }

    /// Send data on a channel. Respects backpressure (send window).
    pub fn send(&mut self, channel_id: ChannelId, data: Vec<u8>) -> Result<(), MuxError> {
        if data.len() > self.config.max_frame_payload {
            return Err(MuxError::PayloadTooLarge {
                size: data.len(),
                max: self.config.max_frame_payload,
            });
        }
        let ch = self.channels.get_mut(&channel_id)
            .ok_or(MuxError::ChannelNotFound(channel_id))?;
        if ch.state != ChannelState::Open {
            return Err(MuxError::ChannelNotOpen(channel_id));
        }
        if ch.send_window < data.len() as u32 {
            // Queue it — backpressure
            ch.send_queue.push_back(data);
            self.events.push(MuxEvent::Backpressure { channel_id });
            return Ok(());
        }
        let seq = ch.next_send_seq;
        ch.next_send_seq += 1;
        ch.send_window -= data.len() as u32;
        ch.bytes_sent += data.len() as u64;
        self.outbound.push_back(MuxFrame::data(channel_id, data, seq));
        Ok(())
    }

    /// Read received data from a channel.
    pub fn recv(&mut self, channel_id: ChannelId) -> Result<Option<Vec<u8>>, MuxError> {
        let ch = self.channels.get_mut(&channel_id)
            .ok_or(MuxError::ChannelNotFound(channel_id))?;
        Ok(ch.recv_queue.pop_front())
    }

    /// Process an inbound frame from the wire.
    pub fn process_frame(&mut self, frame: MuxFrame) -> Result<(), MuxError> {
        match frame.frame_type {
            FrameType::Open => {
                let id = frame.channel_id;
                if self.channels.contains_key(&id) {
                    return Err(MuxError::ChannelAlreadyExists(id));
                }
                if self.channels.len() >= self.config.max_channels {
                    return Err(MuxError::TooManyChannels);
                }
                let metadata = ChannelMetadata::from_bytes(&frame.payload);
                self.channels.insert(id, Channel {
                    id,
                    state: ChannelState::Open,
                    metadata: metadata.clone(),
                    send_window: self.config.initial_window_size,
                    recv_window: self.config.initial_window_size,
                    send_queue: VecDeque::new(),
                    recv_queue: VecDeque::new(),
                    next_send_seq: 1,
                    next_recv_seq: 1,
                    bytes_sent: 0,
                    bytes_received: 0,
                });
                self.events.push(MuxEvent::ChannelOpened { channel_id: id, metadata });
            }
            FrameType::Close => {
                let id = frame.channel_id;
                if let Some(ch) = self.channels.get_mut(&id) {
                    ch.state = ChannelState::Closed;
                    self.events.push(MuxEvent::ChannelClosed { channel_id: id });
                }
            }
            FrameType::Data => {
                let id = frame.channel_id;
                let ch = self.channels.get_mut(&id)
                    .ok_or(MuxError::ChannelNotFound(id))?;
                ch.recv_window = ch.recv_window.saturating_sub(frame.payload.len() as u32);
                ch.bytes_received += frame.payload.len() as u64;
                ch.next_recv_seq = frame.seq + 1;
                let data = frame.payload.clone();
                ch.recv_queue.push_back(data.clone());
                self.events.push(MuxEvent::DataReceived { channel_id: id, data });
            }
            FrameType::WindowUpdate => {
                let id = frame.channel_id;
                let ch = self.channels.get_mut(&id)
                    .ok_or(MuxError::ChannelNotFound(id))?;
                if frame.payload.len() >= 4 {
                    let credit = u32::from_be_bytes([
                        frame.payload[0], frame.payload[1],
                        frame.payload[2], frame.payload[3],
                    ]);
                    ch.send_window = ch.send_window.saturating_add(credit);
                    // Drain send queue if window allows
                    self.drain_send_queue(id);
                }
            }
            FrameType::Ping => {
                self.events.push(MuxEvent::PingReceived { data: frame.payload.clone() });
                self.outbound.push_back(MuxFrame::pong(frame.payload));
            }
            FrameType::Pong => {
                // No-op: pong received
            }
            FrameType::Reset => {
                let id = frame.channel_id;
                let reason = if frame.payload.len() >= 4 {
                    u32::from_be_bytes([
                        frame.payload[0], frame.payload[1],
                        frame.payload[2], frame.payload[3],
                    ])
                } else {
                    0
                };
                if let Some(ch) = self.channels.get_mut(&id) {
                    ch.state = ChannelState::Closed;
                    ch.send_queue.clear();
                    ch.recv_queue.clear();
                }
                self.events.push(MuxEvent::ChannelReset { channel_id: id, reason });
            }
        }
        Ok(())
    }

    fn drain_send_queue(&mut self, channel_id: ChannelId) {
        let ch = match self.channels.get_mut(&channel_id) {
            Some(ch) => ch,
            None => return,
        };
        while let Some(data) = ch.send_queue.front().cloned() {
            if ch.send_window < data.len() as u32 {
                break;
            }
            ch.send_queue.pop_front();
            let seq = ch.next_send_seq;
            ch.next_send_seq += 1;
            ch.send_window -= data.len() as u32;
            ch.bytes_sent += data.len() as u64;
            self.outbound.push_back(MuxFrame::data(channel_id, data, seq));
        }
    }

    /// Grant receive window credit to a channel (sends WindowUpdate to remote).
    pub fn grant_window(&mut self, channel_id: ChannelId, credit: u32) -> Result<(), MuxError> {
        let ch = self.channels.get_mut(&channel_id)
            .ok_or(MuxError::ChannelNotFound(channel_id))?;
        ch.recv_window = ch.recv_window.saturating_add(credit);
        self.outbound.push_back(MuxFrame::window_update(channel_id, credit));
        Ok(())
    }

    /// Take outbound frames to send on the wire.
    pub fn take_outbound(&mut self) -> Vec<MuxFrame> {
        self.outbound.drain(..).collect()
    }

    /// Take application-level events.
    pub fn take_events(&mut self) -> Vec<MuxEvent> {
        std::mem::take(&mut self.events)
    }

    /// Number of channels (all states).
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Number of open channels.
    pub fn open_channel_count(&self) -> usize {
        self.channels.values().filter(|ch| ch.state == ChannelState::Open).count()
    }

    /// Get the state of a channel.
    pub fn channel_state(&self, channel_id: ChannelId) -> Option<ChannelState> {
        self.channels.get(&channel_id).map(|ch| ch.state)
    }

    /// Get channel metadata.
    pub fn channel_metadata(&self, channel_id: ChannelId) -> Option<&ChannelMetadata> {
        self.channels.get(&channel_id).map(|ch| &ch.metadata)
    }

    /// Get send window remaining for a channel.
    pub fn send_window(&self, channel_id: ChannelId) -> Option<u32> {
        self.channels.get(&channel_id).map(|ch| ch.send_window)
    }

    /// Get bytes sent on a channel.
    pub fn bytes_sent(&self, channel_id: ChannelId) -> Option<u64> {
        self.channels.get(&channel_id).map(|ch| ch.bytes_sent)
    }

    /// Get bytes received on a channel.
    pub fn bytes_received(&self, channel_id: ChannelId) -> Option<u64> {
        self.channels.get(&channel_id).map(|ch| ch.bytes_received)
    }

    /// List all channel IDs.
    pub fn channel_ids(&self) -> Vec<ChannelId> {
        self.channels.keys().copied().collect()
    }

    /// Remove closed channels from tracking.
    pub fn prune_closed(&mut self) -> usize {
        let before = self.channels.len();
        self.channels.retain(|_, ch| ch.state != ChannelState::Closed);
        before - self.channels.len()
    }

    /// Send a ping.
    pub fn ping(&mut self, data: Vec<u8>) {
        self.outbound.push_back(MuxFrame::ping(data));
    }

    /// Reset a channel with a reason code.
    pub fn reset_channel(&mut self, channel_id: ChannelId, reason: u32) -> Result<(), MuxError> {
        let ch = self.channels.get_mut(&channel_id)
            .ok_or(MuxError::ChannelNotFound(channel_id))?;
        ch.state = ChannelState::Closed;
        ch.send_queue.clear();
        ch.recv_queue.clear();
        self.outbound.push_back(MuxFrame::reset(channel_id, reason));
        Ok(())
    }
}

impl Default for ChannelMux {
    fn default() -> Self {
        Self::new(MuxConfig::default())
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ChannelId ──────────────────────────────────────────────────

    #[test]
    fn channel_id_display() {
        assert_eq!(ChannelId(42).to_string(), "ch:42");
    }

    // ── FrameType ──────────────────────────────────────────────────

    #[test]
    fn frame_type_roundtrip() {
        for ft in [FrameType::Open, FrameType::Close, FrameType::Data,
                    FrameType::WindowUpdate, FrameType::Ping, FrameType::Pong, FrameType::Reset] {
            assert_eq!(FrameType::from_u8(ft.to_u8()), Some(ft));
        }
    }

    #[test]
    fn frame_type_invalid() {
        assert_eq!(FrameType::from_u8(255), None);
    }

    // ── MuxFrame serialization ─────────────────────────────────────

    #[test]
    fn frame_roundtrip() {
        let frame = MuxFrame::data(ChannelId(7), b"hello".to_vec(), 42);
        let bytes = frame.to_bytes();
        let (parsed, consumed) = MuxFrame::from_bytes(&bytes).unwrap();
        assert_eq!(consumed, bytes.len());
        assert_eq!(parsed.channel_id, ChannelId(7));
        assert_eq!(parsed.frame_type, FrameType::Data);
        assert_eq!(parsed.payload, b"hello");
        assert_eq!(parsed.seq, 42);
    }

    #[test]
    fn frame_too_short() {
        assert!(MuxFrame::from_bytes(&[0; 10]).is_none());
    }

    #[test]
    fn frame_payload_truncated() {
        let frame = MuxFrame::data(ChannelId(1), b"hello".to_vec(), 1);
        let mut bytes = frame.to_bytes();
        bytes.truncate(19); // cut payload short
        assert!(MuxFrame::from_bytes(&bytes).is_none());
    }

    #[test]
    fn frame_empty_payload() {
        let frame = MuxFrame::close(ChannelId(5));
        let bytes = frame.to_bytes();
        let (parsed, _) = MuxFrame::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.frame_type, FrameType::Close);
        assert!(parsed.payload.is_empty());
    }

    // ── ChannelMetadata ────────────────────────────────────────────

    #[test]
    fn metadata_roundtrip() {
        let meta = ChannelMetadata::new()
            .with_label("type", "chat")
            .with_label("room", "lobby");
        let bytes = meta.to_bytes();
        let parsed = ChannelMetadata::from_bytes(&bytes);
        assert_eq!(parsed.labels.get("type").unwrap(), "chat");
        assert_eq!(parsed.labels.get("room").unwrap(), "lobby");
    }

    #[test]
    fn metadata_empty() {
        let meta = ChannelMetadata::new();
        let bytes = meta.to_bytes();
        assert!(bytes.is_empty());
        let parsed = ChannelMetadata::from_bytes(&bytes);
        assert!(parsed.labels.is_empty());
    }

    // ── Channel lifecycle ──────────────────────────────────────────

    #[test]
    fn open_and_close_channel() {
        let mut mux = ChannelMux::new(MuxConfig::default());
        let id = mux.open_channel(ChannelMetadata::new()).unwrap();
        assert_eq!(mux.channel_count(), 1);
        assert_eq!(mux.open_channel_count(), 1);
        assert_eq!(mux.channel_state(id), Some(ChannelState::Open));

        mux.close_channel(id).unwrap();
        assert_eq!(mux.channel_state(id), Some(ChannelState::Closing));

        let outbound = mux.take_outbound();
        assert_eq!(outbound.len(), 2); // Open + Close frames
        assert_eq!(outbound[0].frame_type, FrameType::Open);
        assert_eq!(outbound[1].frame_type, FrameType::Close);
    }

    #[test]
    fn max_channels_exceeded() {
        let mut mux = ChannelMux::new(MuxConfig { max_channels: 2, ..Default::default() });
        mux.open_channel(ChannelMetadata::new()).unwrap();
        mux.open_channel(ChannelMetadata::new()).unwrap();
        assert!(matches!(
            mux.open_channel(ChannelMetadata::new()),
            Err(MuxError::TooManyChannels)
        ));
    }

    #[test]
    fn close_unknown_channel() {
        let mut mux = ChannelMux::default();
        assert!(matches!(
            mux.close_channel(ChannelId(99)),
            Err(MuxError::ChannelNotFound(_))
        ));
    }

    // ── Send and receive ───────────────────────────────────────────

    #[test]
    fn send_data() {
        let mut mux = ChannelMux::new(MuxConfig::default());
        let id = mux.open_channel(ChannelMetadata::new()).unwrap();
        mux.take_outbound(); // clear open frame

        mux.send(id, b"hello".to_vec()).unwrap();
        let out = mux.take_outbound();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].frame_type, FrameType::Data);
        assert_eq!(out[0].payload, b"hello");
        assert_eq!(out[0].seq, 1);

        assert_eq!(mux.bytes_sent(id), Some(5));
    }

    #[test]
    fn send_on_closed_channel() {
        let mut mux = ChannelMux::new(MuxConfig::default());
        let id = mux.open_channel(ChannelMetadata::new()).unwrap();
        mux.close_channel(id).unwrap();
        assert!(matches!(
            mux.send(id, b"x".to_vec()),
            Err(MuxError::ChannelNotOpen(_))
        ));
    }

    #[test]
    fn send_payload_too_large() {
        let mut mux = ChannelMux::new(MuxConfig { max_frame_payload: 5, ..Default::default() });
        let id = mux.open_channel(ChannelMetadata::new()).unwrap();
        assert!(matches!(
            mux.send(id, vec![0; 10]),
            Err(MuxError::PayloadTooLarge { size: 10, max: 5 })
        ));
    }

    #[test]
    fn receive_data() {
        let mut mux = ChannelMux::new(MuxConfig::default());
        let id = mux.open_channel(ChannelMetadata::new()).unwrap();

        // Simulate remote sending data
        let frame = MuxFrame::data(id, b"world".to_vec(), 1);
        mux.process_frame(frame).unwrap();

        let data = mux.recv(id).unwrap();
        assert_eq!(data, Some(b"world".to_vec()));
        assert_eq!(mux.bytes_received(id), Some(5));
    }

    #[test]
    fn recv_empty_channel() {
        let mut mux = ChannelMux::new(MuxConfig::default());
        let id = mux.open_channel(ChannelMetadata::new()).unwrap();
        assert_eq!(mux.recv(id).unwrap(), None);
    }

    // ── Backpressure ───────────────────────────────────────────────

    #[test]
    fn backpressure_when_window_exhausted() {
        let mut mux = ChannelMux::new(MuxConfig {
            initial_window_size: 10,
            max_frame_payload: 100,
            ..Default::default()
        });
        let id = mux.open_channel(ChannelMetadata::new()).unwrap();
        mux.take_outbound();

        // Send 10 bytes — fills window
        mux.send(id, vec![0; 10]).unwrap();
        assert_eq!(mux.send_window(id), Some(0));

        // Next send gets queued (backpressure)
        mux.send(id, vec![1; 5]).unwrap();
        let events = mux.take_events();
        assert!(events.iter().any(|e| matches!(e, MuxEvent::Backpressure { .. })));

        // Only the first send produced a data frame
        let out = mux.take_outbound();
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn window_update_drains_queue() {
        let mut mux = ChannelMux::new(MuxConfig {
            initial_window_size: 5,
            max_frame_payload: 100,
            ..Default::default()
        });
        let id = mux.open_channel(ChannelMetadata::new()).unwrap();
        mux.take_outbound();

        mux.send(id, vec![0; 5]).unwrap(); // fills window
        mux.send(id, vec![1; 3]).unwrap(); // queued

        let out = mux.take_outbound();
        assert_eq!(out.len(), 1); // only first send

        // Receive window update
        mux.process_frame(MuxFrame::window_update(id, 10)).unwrap();
        let out = mux.take_outbound();
        assert_eq!(out.len(), 1); // queued data now sent
        assert_eq!(out[0].payload, vec![1; 3]);
    }

    // ── Remote channel open ────────────────────────────────────────

    #[test]
    fn process_remote_open() {
        let mut mux = ChannelMux::new(MuxConfig::default());
        let meta = ChannelMetadata::new().with_label("type", "rpc");
        let frame = MuxFrame::open(ChannelId(100), meta.to_bytes());
        mux.process_frame(frame).unwrap();

        assert_eq!(mux.channel_count(), 1);
        assert_eq!(mux.channel_state(ChannelId(100)), Some(ChannelState::Open));
        let events = mux.take_events();
        assert!(events.iter().any(|e| matches!(e, MuxEvent::ChannelOpened { channel_id, .. } if *channel_id == ChannelId(100))));
    }

    #[test]
    fn process_remote_close() {
        let mut mux = ChannelMux::new(MuxConfig::default());
        let id = mux.open_channel(ChannelMetadata::new()).unwrap();
        mux.take_events();

        mux.process_frame(MuxFrame::close(id)).unwrap();
        assert_eq!(mux.channel_state(id), Some(ChannelState::Closed));
        let events = mux.take_events();
        assert!(events.iter().any(|e| matches!(e, MuxEvent::ChannelClosed { .. })));
    }

    // ── Ping/Pong ──────────────────────────────────────────────────

    #[test]
    fn ping_generates_pong() {
        let mut mux = ChannelMux::new(MuxConfig::default());
        mux.process_frame(MuxFrame::ping(b"hello".to_vec())).unwrap();

        let events = mux.take_events();
        assert!(events.iter().any(|e| matches!(e, MuxEvent::PingReceived { .. })));

        let out = mux.take_outbound();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].frame_type, FrameType::Pong);
        assert_eq!(out[0].payload, b"hello");
    }

    #[test]
    fn send_ping() {
        let mut mux = ChannelMux::default();
        mux.ping(b"test".to_vec());
        let out = mux.take_outbound();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].frame_type, FrameType::Ping);
    }

    // ── Reset ──────────────────────────────────────────────────────

    #[test]
    fn reset_channel() {
        let mut mux = ChannelMux::new(MuxConfig::default());
        let id = mux.open_channel(ChannelMetadata::new()).unwrap();
        mux.reset_channel(id, 42).unwrap();

        assert_eq!(mux.channel_state(id), Some(ChannelState::Closed));
        let out = mux.take_outbound();
        assert!(out.iter().any(|f| f.frame_type == FrameType::Reset));
    }

    #[test]
    fn process_remote_reset() {
        let mut mux = ChannelMux::new(MuxConfig::default());
        let id = mux.open_channel(ChannelMetadata::new()).unwrap();
        mux.take_events();

        mux.process_frame(MuxFrame::reset(id, 99)).unwrap();
        let events = mux.take_events();
        assert!(events.iter().any(|e| matches!(e, MuxEvent::ChannelReset { reason: 99, .. })));
    }

    // ── Grant window ───────────────────────────────────────────────

    #[test]
    fn grant_window_sends_update() {
        let mut mux = ChannelMux::new(MuxConfig::default());
        let id = mux.open_channel(ChannelMetadata::new()).unwrap();
        mux.take_outbound();

        mux.grant_window(id, 1000).unwrap();
        let out = mux.take_outbound();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].frame_type, FrameType::WindowUpdate);
    }

    // ── Prune closed ───────────────────────────────────────────────

    #[test]
    fn prune_closed_channels() {
        let mut mux = ChannelMux::new(MuxConfig::default());
        let id1 = mux.open_channel(ChannelMetadata::new()).unwrap();
        let _id2 = mux.open_channel(ChannelMetadata::new()).unwrap();
        mux.close_channel(id1).unwrap();
        // Simulate remote closing it
        mux.process_frame(MuxFrame::close(id1)).unwrap();

        let pruned = mux.prune_closed();
        assert_eq!(pruned, 1);
        assert_eq!(mux.channel_count(), 1);
    }

    // ── Channel listing ────────────────────────────────────────────

    #[test]
    fn channel_ids_sorted() {
        let mut mux = ChannelMux::new(MuxConfig::default());
        mux.open_channel(ChannelMetadata::new()).unwrap();
        mux.open_channel(ChannelMetadata::new()).unwrap();
        mux.open_channel(ChannelMetadata::new()).unwrap();
        let ids = mux.channel_ids();
        assert_eq!(ids.len(), 3);
        assert!(ids[0].0 < ids[1].0);
        assert!(ids[1].0 < ids[2].0);
    }

    // ── Error display ──────────────────────────────────────────────

    #[test]
    fn error_display() {
        assert!(MuxError::TooManyChannels.to_string().contains("too many"));
        assert!(MuxError::InvalidFrame.to_string().contains("invalid"));
        assert!(MuxError::ChannelNotFound(ChannelId(1)).to_string().contains("ch:1"));
    }

    // ── Data on unknown channel ────────────────────────────────────

    #[test]
    fn data_on_unknown_channel() {
        let mut mux = ChannelMux::default();
        let frame = MuxFrame::data(ChannelId(99), b"x".to_vec(), 1);
        assert!(matches!(
            mux.process_frame(frame),
            Err(MuxError::ChannelNotFound(_))
        ));
    }

    // ── Ordered delivery ───────────────────────────────────────────

    #[test]
    fn send_sequence_numbers_increment() {
        let mut mux = ChannelMux::new(MuxConfig::default());
        let id = mux.open_channel(ChannelMetadata::new()).unwrap();
        mux.take_outbound();

        mux.send(id, b"a".to_vec()).unwrap();
        mux.send(id, b"b".to_vec()).unwrap();
        mux.send(id, b"c".to_vec()).unwrap();
        let out = mux.take_outbound();
        assert_eq!(out[0].seq, 1);
        assert_eq!(out[1].seq, 2);
        assert_eq!(out[2].seq, 3);
    }

    #[test]
    fn channel_metadata_labels() {
        let mut mux = ChannelMux::new(MuxConfig::default());
        let meta = ChannelMetadata::new().with_label("service", "rpc");
        let id = mux.open_channel(meta).unwrap();
        let stored = mux.channel_metadata(id).unwrap();
        assert_eq!(stored.labels.get("service").unwrap(), "rpc");
    }
}
