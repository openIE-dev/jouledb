//! Transport multiplexing layer.
//!
//! Manages multiple logical [`Channel`]s over a single transport.
//! Supports round-robin and priority-based scheduling, per-channel
//! send/receive buffers, flow control credits, channel metadata,
//! and per-channel throughput statistics.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Transport mux domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MuxError {
    /// Channel not found.
    ChannelNotFound(u32),
    /// Duplicate channel ID.
    DuplicateChannel(u32),
    /// Maximum channels limit reached.
    MaxChannelsReached { max: usize },
    /// Channel is closed.
    ChannelClosed(u32),
    /// No flow control credits remaining.
    NoCredits { channel_id: u32, available: u64 },
    /// Send buffer is full.
    SendBufferFull { channel_id: u32, capacity: usize },
}

impl fmt::Display for MuxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChannelNotFound(id) => write!(f, "channel not found: {id}"),
            Self::DuplicateChannel(id) => write!(f, "duplicate channel: {id}"),
            Self::MaxChannelsReached { max } => write!(f, "max channels reached: {max}"),
            Self::ChannelClosed(id) => write!(f, "channel {id} is closed"),
            Self::NoCredits { channel_id, available } => {
                write!(f, "no credits on channel {channel_id} (available={available})")
            }
            Self::SendBufferFull { channel_id, capacity } => {
                write!(f, "send buffer full on channel {channel_id} (cap={capacity})")
            }
        }
    }
}

impl std::error::Error for MuxError {}

// ── Priority ────────────────────────────────────────────────────

/// Channel priority level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Urgent = 3,
}

impl Default for Priority {
    fn default() -> Self {
        Self::Normal
    }
}

impl fmt::Display for Priority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "low"),
            Self::Normal => write!(f, "normal"),
            Self::High => write!(f, "high"),
            Self::Urgent => write!(f, "urgent"),
        }
    }
}

// ── Scheduling Mode ─────────────────────────────────────────────

/// Scheduling discipline for draining channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleMode {
    RoundRobin,
    PriorityBased,
}

impl Default for ScheduleMode {
    fn default() -> Self {
        Self::RoundRobin
    }
}

// ── Channel Metadata ────────────────────────────────────────────

/// User-defined metadata attached to a channel.
#[derive(Debug, Clone, Default)]
pub struct ChannelMeta {
    pub label: String,
    pub tags: Vec<String>,
}

impl ChannelMeta {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            tags: Vec::new(),
        }
    }

    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }
}

// ── Channel Stats ───────────────────────────────────────────────

/// Per-channel throughput statistics.
#[derive(Debug, Clone, Default)]
pub struct ChannelStats {
    pub messages_sent: u64,
    pub messages_received: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
}

impl ChannelStats {
    pub fn throughput_bytes(&self) -> u64 {
        self.bytes_sent + self.bytes_received
    }
}

impl fmt::Display for ChannelStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "sent={}msg/{}B recv={}msg/{}B",
            self.messages_sent,
            self.bytes_sent,
            self.messages_received,
            self.bytes_received,
        )
    }
}

// ── Channel ─────────────────────────────────────────────────────

/// A logical multiplexed channel.
pub struct Channel {
    pub id: u32,
    pub priority: Priority,
    pub meta: ChannelMeta,
    send_buffer: VecDeque<Vec<u8>>,
    recv_buffer: VecDeque<Vec<u8>>,
    credits: u64,
    max_send_buffer: usize,
    open: bool,
    stats: ChannelStats,
}

impl Channel {
    fn new(id: u32, credits: u64, max_send_buffer: usize) -> Self {
        Self {
            id,
            priority: Priority::default(),
            meta: ChannelMeta::default(),
            send_buffer: VecDeque::new(),
            recv_buffer: VecDeque::new(),
            credits,
            max_send_buffer,
            open: true,
            stats: ChannelStats::default(),
        }
    }

    /// Remaining flow control credits.
    pub fn available_credits(&self) -> u64 {
        self.credits
    }

    /// Whether the channel is open.
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Number of messages in the send buffer.
    pub fn send_pending(&self) -> usize {
        self.send_buffer.len()
    }

    /// Number of messages in the receive buffer.
    pub fn recv_pending(&self) -> usize {
        self.recv_buffer.len()
    }

    /// Channel statistics.
    pub fn stats(&self) -> &ChannelStats {
        &self.stats
    }
}

impl fmt::Display for Channel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Channel(id={}, priority={}, credits={}, send={}, recv={})",
            self.id,
            self.priority,
            self.credits,
            self.send_buffer.len(),
            self.recv_buffer.len(),
        )
    }
}

// ── Mux Config ──────────────────────────────────────────────────

/// Configuration for the transport mux.
#[derive(Debug, Clone)]
pub struct MuxConfig {
    pub max_channels: usize,
    pub initial_credits: u64,
    pub max_send_buffer: usize,
    pub schedule_mode: ScheduleMode,
}

impl Default for MuxConfig {
    fn default() -> Self {
        Self {
            max_channels: 256,
            initial_credits: 65536,
            max_send_buffer: 512,
            schedule_mode: ScheduleMode::RoundRobin,
        }
    }
}

impl MuxConfig {
    pub fn with_max_channels(mut self, max: usize) -> Self {
        self.max_channels = max;
        self
    }

    pub fn with_schedule_mode(mut self, mode: ScheduleMode) -> Self {
        self.schedule_mode = mode;
        self
    }

    pub fn with_initial_credits(mut self, credits: u64) -> Self {
        self.initial_credits = credits;
        self
    }
}

// ── Transport Mux ───────────────────────────────────────────────

/// Multiplexes multiple logical channels over a single transport.
pub struct TransportMux {
    config: MuxConfig,
    channels: BTreeMap<u32, Channel>,
    next_channel_id: u32,
    round_robin_index: usize,
}

impl TransportMux {
    pub fn new(config: MuxConfig) -> Self {
        Self {
            config,
            channels: BTreeMap::new(),
            next_channel_id: 0,
            round_robin_index: 0,
        }
    }

    /// Open a new channel, returning its ID.
    pub fn open_channel(&mut self) -> Result<u32, MuxError> {
        if self.channels.len() >= self.config.max_channels {
            return Err(MuxError::MaxChannelsReached { max: self.config.max_channels });
        }
        let id = self.next_channel_id;
        self.next_channel_id += 1;
        let ch = Channel::new(id, self.config.initial_credits, self.config.max_send_buffer);
        self.channels.insert(id, ch);
        Ok(id)
    }

    /// Open a channel with specific priority and metadata.
    pub fn open_channel_with(
        &mut self,
        priority: Priority,
        meta: ChannelMeta,
    ) -> Result<u32, MuxError> {
        let id = self.open_channel()?;
        if let Some(ch) = self.channels.get_mut(&id) {
            ch.priority = priority;
            ch.meta = meta;
        }
        Ok(id)
    }

    /// Close a channel.
    pub fn close_channel(&mut self, channel_id: u32) -> Result<(), MuxError> {
        let ch = self.channels.get_mut(&channel_id)
            .ok_or(MuxError::ChannelNotFound(channel_id))?;
        if !ch.open {
            return Err(MuxError::ChannelClosed(channel_id));
        }
        ch.open = false;
        Ok(())
    }

    /// Remove a closed channel entirely.
    pub fn remove_channel(&mut self, channel_id: u32) -> Result<(), MuxError> {
        if !self.channels.contains_key(&channel_id) {
            return Err(MuxError::ChannelNotFound(channel_id));
        }
        self.channels.remove(&channel_id);
        Ok(())
    }

    /// Send data on a channel.
    pub fn send(&mut self, channel_id: u32, data: Vec<u8>) -> Result<(), MuxError> {
        let ch = self.channels.get_mut(&channel_id)
            .ok_or(MuxError::ChannelNotFound(channel_id))?;
        if !ch.open {
            return Err(MuxError::ChannelClosed(channel_id));
        }
        if ch.send_buffer.len() >= ch.max_send_buffer {
            return Err(MuxError::SendBufferFull {
                channel_id,
                capacity: ch.max_send_buffer,
            });
        }
        let data_len = data.len() as u64;
        if ch.credits < data_len {
            return Err(MuxError::NoCredits {
                channel_id,
                available: ch.credits,
            });
        }
        ch.credits -= data_len;
        ch.stats.bytes_sent += data_len;
        ch.stats.messages_sent += 1;
        ch.send_buffer.push_back(data);
        Ok(())
    }

    /// Deliver data to a channel's receive buffer (simulated incoming).
    pub fn deliver(&mut self, channel_id: u32, data: Vec<u8>) -> Result<(), MuxError> {
        let ch = self.channels.get_mut(&channel_id)
            .ok_or(MuxError::ChannelNotFound(channel_id))?;
        let data_len = data.len() as u64;
        ch.stats.bytes_received += data_len;
        ch.stats.messages_received += 1;
        ch.recv_buffer.push_back(data);
        Ok(())
    }

    /// Drain received data from a channel.
    pub fn drain_recv(&mut self, channel_id: u32) -> Result<Vec<Vec<u8>>, MuxError> {
        let ch = self.channels.get_mut(&channel_id)
            .ok_or(MuxError::ChannelNotFound(channel_id))?;
        Ok(ch.recv_buffer.drain(..).collect())
    }

    /// Grant additional credits to a channel.
    pub fn grant_credits(&mut self, channel_id: u32, credits: u64) -> Result<(), MuxError> {
        let ch = self.channels.get_mut(&channel_id)
            .ok_or(MuxError::ChannelNotFound(channel_id))?;
        ch.credits = ch.credits.saturating_add(credits);
        Ok(())
    }

    /// Schedule one message from the next channel according to the scheduling mode.
    /// Returns (channel_id, data) if a message was available.
    pub fn schedule_one(&mut self) -> Option<(u32, Vec<u8>)> {
        match self.config.schedule_mode {
            ScheduleMode::RoundRobin => self.schedule_round_robin(),
            ScheduleMode::PriorityBased => self.schedule_priority(),
        }
    }

    /// Number of open channels.
    pub fn open_channel_count(&self) -> usize {
        self.channels.values().filter(|c| c.open).count()
    }

    /// Total channels (open + closed but not removed).
    pub fn total_channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Get a reference to a channel.
    pub fn channel(&self, channel_id: u32) -> Option<&Channel> {
        self.channels.get(&channel_id)
    }

    /// Get aggregate statistics across all channels.
    pub fn aggregate_stats(&self) -> ChannelStats {
        let mut agg = ChannelStats::default();
        for ch in self.channels.values() {
            agg.messages_sent += ch.stats.messages_sent;
            agg.messages_received += ch.stats.messages_received;
            agg.bytes_sent += ch.stats.bytes_sent;
            agg.bytes_received += ch.stats.bytes_received;
        }
        agg
    }

    // ── Internal ────────────────────────────────────────────────

    fn schedule_round_robin(&mut self) -> Option<(u32, Vec<u8>)> {
        let open_ids: Vec<u32> = self.channels.iter()
            .filter(|(_, c)| c.open && !c.send_buffer.is_empty())
            .map(|(&id, _)| id)
            .collect();
        if open_ids.is_empty() {
            return None;
        }
        let idx = self.round_robin_index % open_ids.len();
        self.round_robin_index = self.round_robin_index.wrapping_add(1);
        let channel_id = open_ids[idx];
        let ch = self.channels.get_mut(&channel_id)?;
        ch.send_buffer.pop_front().map(|data| (channel_id, data))
    }

    fn schedule_priority(&mut self) -> Option<(u32, Vec<u8>)> {
        // Find the highest-priority channel with data.
        let mut best: Option<(Priority, u32)> = None;
        for (&id, ch) in &self.channels {
            if ch.open && !ch.send_buffer.is_empty() {
                match best {
                    None => best = Some((ch.priority, id)),
                    Some((bp, _)) if ch.priority > bp => best = Some((ch.priority, id)),
                    _ => {}
                }
            }
        }
        let (_, channel_id) = best?;
        let ch = self.channels.get_mut(&channel_id)?;
        ch.send_buffer.pop_front().map(|data| (channel_id, data))
    }
}

impl fmt::Display for TransportMux {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TransportMux(channels={}/{}, mode={:?})",
            self.open_channel_count(),
            self.config.max_channels,
            self.config.schedule_mode,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_mux() -> TransportMux {
        TransportMux::new(MuxConfig::default())
    }

    #[test]
    fn open_channel_returns_id() {
        let mut mux = default_mux();
        let id = mux.open_channel().unwrap();
        assert_eq!(id, 0);
        assert_eq!(mux.open_channel_count(), 1);
    }

    #[test]
    fn max_channels_enforced() {
        let config = MuxConfig::default().with_max_channels(2);
        let mut mux = TransportMux::new(config);
        mux.open_channel().unwrap();
        mux.open_channel().unwrap();
        let err = mux.open_channel().unwrap_err();
        assert!(matches!(err, MuxError::MaxChannelsReached { max: 2 }));
    }

    #[test]
    fn close_channel() {
        let mut mux = default_mux();
        let id = mux.open_channel().unwrap();
        mux.close_channel(id).unwrap();
        assert_eq!(mux.open_channel_count(), 0);
    }

    #[test]
    fn close_already_closed_errors() {
        let mut mux = default_mux();
        let id = mux.open_channel().unwrap();
        mux.close_channel(id).unwrap();
        let err = mux.close_channel(id).unwrap_err();
        assert!(matches!(err, MuxError::ChannelClosed(_)));
    }

    #[test]
    fn send_and_receive() {
        let mut mux = default_mux();
        let id = mux.open_channel().unwrap();
        mux.send(id, vec![1, 2, 3]).unwrap();
        mux.deliver(id, vec![4, 5]).unwrap();
        let received = mux.drain_recv(id).unwrap();
        assert_eq!(received, vec![vec![4, 5]]);
    }

    #[test]
    fn send_on_closed_channel_errors() {
        let mut mux = default_mux();
        let id = mux.open_channel().unwrap();
        mux.close_channel(id).unwrap();
        let err = mux.send(id, vec![1]).unwrap_err();
        assert!(matches!(err, MuxError::ChannelClosed(_)));
    }

    #[test]
    fn flow_control_credits() {
        let config = MuxConfig::default().with_initial_credits(10);
        let mut mux = TransportMux::new(config);
        let id = mux.open_channel().unwrap();
        mux.send(id, vec![0; 8]).unwrap();
        let err = mux.send(id, vec![0; 5]).unwrap_err();
        assert!(matches!(err, MuxError::NoCredits { .. }));
    }

    #[test]
    fn grant_credits_allows_more_sends() {
        let config = MuxConfig::default().with_initial_credits(5);
        let mut mux = TransportMux::new(config);
        let id = mux.open_channel().unwrap();
        mux.send(id, vec![0; 5]).unwrap();
        assert!(mux.send(id, vec![0; 1]).is_err());
        mux.grant_credits(id, 10).unwrap();
        mux.send(id, vec![0; 1]).unwrap();
    }

    #[test]
    fn round_robin_scheduling() {
        let config = MuxConfig::default().with_schedule_mode(ScheduleMode::RoundRobin);
        let mut mux = TransportMux::new(config);
        let a = mux.open_channel().unwrap();
        let b = mux.open_channel().unwrap();
        mux.send(a, vec![1]).unwrap();
        mux.send(b, vec![2]).unwrap();
        let (id1, d1) = mux.schedule_one().unwrap();
        let (id2, d2) = mux.schedule_one().unwrap();
        // Both channels should get scheduled.
        let mut ids = vec![id1, id2];
        ids.sort();
        assert_eq!(ids, vec![a, b]);
        let mut payloads = vec![d1, d2];
        payloads.sort();
        assert_eq!(payloads, vec![vec![1], vec![2]]);
    }

    #[test]
    fn priority_scheduling() {
        let config = MuxConfig::default().with_schedule_mode(ScheduleMode::PriorityBased);
        let mut mux = TransportMux::new(config);
        let low = mux.open_channel_with(Priority::Low, ChannelMeta::new("low")).unwrap();
        let high = mux.open_channel_with(Priority::High, ChannelMeta::new("high")).unwrap();
        mux.send(low, vec![10]).unwrap();
        mux.send(high, vec![20]).unwrap();
        let (id, data) = mux.schedule_one().unwrap();
        assert_eq!(id, high);
        assert_eq!(data, vec![20]);
    }

    #[test]
    fn schedule_empty_returns_none() {
        let mut mux = default_mux();
        mux.open_channel().unwrap();
        assert!(mux.schedule_one().is_none());
    }

    #[test]
    fn channel_stats_tracking() {
        let mut mux = default_mux();
        let id = mux.open_channel().unwrap();
        mux.send(id, vec![0; 100]).unwrap();
        mux.deliver(id, vec![0; 50]).unwrap();
        let ch = mux.channel(id).unwrap();
        assert_eq!(ch.stats().bytes_sent, 100);
        assert_eq!(ch.stats().bytes_received, 50);
        assert_eq!(ch.stats().messages_sent, 1);
    }

    #[test]
    fn aggregate_stats() {
        let mut mux = default_mux();
        let a = mux.open_channel().unwrap();
        let b = mux.open_channel().unwrap();
        mux.send(a, vec![0; 10]).unwrap();
        mux.send(b, vec![0; 20]).unwrap();
        let agg = mux.aggregate_stats();
        assert_eq!(agg.bytes_sent, 30);
        assert_eq!(agg.messages_sent, 2);
    }

    #[test]
    fn remove_channel() {
        let mut mux = default_mux();
        let id = mux.open_channel().unwrap();
        mux.close_channel(id).unwrap();
        mux.remove_channel(id).unwrap();
        assert_eq!(mux.total_channel_count(), 0);
    }

    #[test]
    fn channel_metadata() {
        let mut mux = default_mux();
        let meta = ChannelMeta::new("control").with_tag("critical");
        let id = mux.open_channel_with(Priority::Urgent, meta).unwrap();
        let ch = mux.channel(id).unwrap();
        assert_eq!(ch.meta.label, "control");
        assert_eq!(ch.meta.tags, vec!["critical"]);
        assert_eq!(ch.priority, Priority::Urgent);
    }

    #[test]
    fn send_buffer_full_error() {
        let mut config = MuxConfig::default();
        config.max_send_buffer = 2;
        config.initial_credits = 100_000;
        let mut mux = TransportMux::new(config);
        let id = mux.open_channel().unwrap();
        mux.send(id, vec![1]).unwrap();
        mux.send(id, vec![2]).unwrap();
        let err = mux.send(id, vec![3]).unwrap_err();
        assert!(matches!(err, MuxError::SendBufferFull { .. }));
    }

    #[test]
    fn channel_display() {
        let ch = Channel::new(5, 1000, 64);
        let s = format!("{ch}");
        assert!(s.contains("id=5"));
        assert!(s.contains("credits=1000"));
    }

    #[test]
    fn mux_display() {
        let mux = default_mux();
        let s = format!("{mux}");
        assert!(s.contains("TransportMux"));
    }

    #[test]
    fn config_builder() {
        let config = MuxConfig::default()
            .with_max_channels(10)
            .with_schedule_mode(ScheduleMode::PriorityBased)
            .with_initial_credits(1000);
        assert_eq!(config.max_channels, 10);
        assert_eq!(config.schedule_mode, ScheduleMode::PriorityBased);
        assert_eq!(config.initial_credits, 1000);
    }
}
