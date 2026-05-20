//! Reliable UDP transport layer.
//!
//! Provides ordered, acknowledged delivery over an unreliable transport.
//! Each [`Packet`] carries a sequence number, acknowledgment, and a
//! 32-bit ack bitfield for selective acknowledgment of previous packets.
//! [`ReliableChannel`] tracks in-flight packets, performs retransmission
//! after a configurable timeout, estimates RTT via EWMA, and implements
//! a basic congestion window.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Reliable-UDP domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReliableUdpError {
    /// Payload exceeds maximum packet size.
    PayloadTooLarge { size: usize, max: usize },
    /// Sequence number wrapped unexpectedly.
    SequenceWrap,
    /// Congestion window full — cannot send.
    WindowFull { cwnd: u32 },
    /// Duplicate packet received.
    DuplicatePacket(u32),
    /// Channel is closed.
    ChannelClosed,
}

impl fmt::Display for ReliableUdpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PayloadTooLarge { size, max } => {
                write!(f, "payload too large: {size} bytes (max {max})")
            }
            Self::SequenceWrap => write!(f, "sequence number wrapped"),
            Self::WindowFull { cwnd } => write!(f, "congestion window full (cwnd={cwnd})"),
            Self::DuplicatePacket(seq) => write!(f, "duplicate packet: seq={seq}"),
            Self::ChannelClosed => write!(f, "channel is closed"),
        }
    }
}

impl std::error::Error for ReliableUdpError {}

// ── Packet ──────────────────────────────────────────────────────

/// A reliable-UDP packet with sequencing and selective ack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Packet {
    pub sequence_number: u32,
    pub ack: u32,
    pub ack_bitfield: u32,
    pub payload: Vec<u8>,
}

impl Packet {
    pub fn new(sequence_number: u32, payload: Vec<u8>) -> Self {
        Self {
            sequence_number,
            ack: 0,
            ack_bitfield: 0,
            payload,
        }
    }

    /// Wire size: 4 (seq) + 4 (ack) + 4 (bitfield) + payload.
    pub fn wire_size(&self) -> usize {
        12 + self.payload.len()
    }
}

impl fmt::Display for Packet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Packet(seq={}, ack={}, bits={:#010x}, {}B)",
            self.sequence_number,
            self.ack,
            self.ack_bitfield,
            self.payload.len()
        )
    }
}

// ── Sent-Packet Record ──────────────────────────────────────────

/// Metadata for an in-flight packet awaiting acknowledgment.
#[derive(Debug, Clone)]
struct SentRecord {
    packet: Packet,
    send_tick: u64,
    retransmit_count: u32,
}

// ── Channel Statistics ──────────────────────────────────────────

/// Cumulative statistics for a [`ReliableChannel`].
#[derive(Debug, Clone, Default)]
pub struct ChannelStats {
    pub packets_sent: u64,
    pub packets_received: u64,
    pub packets_acked: u64,
    pub packets_lost: u64,
    pub retransmissions: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub rtt_ms: f64,
    pub cwnd: u32,
}

impl ChannelStats {
    /// Packet loss rate as a fraction in [0, 1].
    pub fn loss_rate(&self) -> f64 {
        if self.packets_sent == 0 {
            return 0.0;
        }
        self.packets_lost as f64 / self.packets_sent as f64
    }

    /// Estimated bandwidth in bytes per tick.
    pub fn bandwidth_estimate(&self) -> f64 {
        if self.rtt_ms <= 0.0 {
            return 0.0;
        }
        (self.cwnd as f64 * 1200.0) / self.rtt_ms
    }
}

impl fmt::Display for ChannelStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "sent={} recv={} acked={} lost={} rtt={:.1}ms cwnd={}",
            self.packets_sent,
            self.packets_received,
            self.packets_acked,
            self.packets_lost,
            self.rtt_ms,
            self.cwnd,
        )
    }
}

// ── Channel Config ──────────────────────────────────────────────

/// Configuration for a reliable channel.
#[derive(Debug, Clone)]
pub struct ChannelConfig {
    pub max_payload_size: usize,
    pub initial_cwnd: u32,
    pub max_cwnd: u32,
    pub retransmit_timeout_ticks: u64,
    pub max_retransmissions: u32,
    pub rtt_alpha: f64,
    pub receive_buffer_size: usize,
}

impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            max_payload_size: 1200,
            initial_cwnd: 4,
            max_cwnd: 64,
            retransmit_timeout_ticks: 100,
            max_retransmissions: 5,
            rtt_alpha: 0.125,
            receive_buffer_size: 256,
        }
    }
}

impl ChannelConfig {
    pub fn with_max_payload_size(mut self, size: usize) -> Self {
        self.max_payload_size = size;
        self
    }

    pub fn with_initial_cwnd(mut self, cwnd: u32) -> Self {
        self.initial_cwnd = cwnd;
        self
    }

    pub fn with_retransmit_timeout(mut self, ticks: u64) -> Self {
        self.retransmit_timeout_ticks = ticks;
        self
    }
}

// ── Reliable Channel ────────────────────────────────────────────

/// Reliable channel managing send/receive over unreliable transport.
pub struct ReliableChannel {
    config: ChannelConfig,
    // Send state
    next_sequence: u32,
    sent_packets: BTreeMap<u32, SentRecord>,
    cwnd: u32,
    ssthresh: u32,
    // Receive state
    expected_sequence: u32,
    receive_buffer: BTreeMap<u32, Packet>,
    delivered: VecDeque<Packet>,
    // Ack tracking
    last_ack: u32,
    ack_bitfield: u32,
    // RTT
    rtt_estimate_ms: f64,
    rtt_initialized: bool,
    // Stats
    stats: ChannelStats,
    // Tick
    current_tick: u64,
    closed: bool,
}

impl ReliableChannel {
    pub fn new(config: ChannelConfig) -> Self {
        let cwnd = config.initial_cwnd;
        Self {
            config,
            next_sequence: 0,
            sent_packets: BTreeMap::new(),
            cwnd,
            ssthresh: 32,
            expected_sequence: 0,
            receive_buffer: BTreeMap::new(),
            delivered: VecDeque::new(),
            last_ack: 0,
            ack_bitfield: 0,
            rtt_estimate_ms: 100.0,
            rtt_initialized: false,
            stats: ChannelStats::default(),
            current_tick: 0,
            closed: false,
        }
    }

    /// Advance the channel clock.
    pub fn tick(&mut self, tick: u64) {
        self.current_tick = tick;
    }

    /// Prepare a packet for sending. Returns the packet ready for the wire.
    pub fn send(&mut self, payload: Vec<u8>) -> Result<Packet, ReliableUdpError> {
        if self.closed {
            return Err(ReliableUdpError::ChannelClosed);
        }
        if payload.len() > self.config.max_payload_size {
            return Err(ReliableUdpError::PayloadTooLarge {
                size: payload.len(),
                max: self.config.max_payload_size,
            });
        }
        if self.sent_packets.len() as u32 >= self.cwnd {
            return Err(ReliableUdpError::WindowFull { cwnd: self.cwnd });
        }

        let seq = self.next_sequence;
        self.next_sequence = seq.wrapping_add(1);

        let mut packet = Packet::new(seq, payload);
        packet.ack = self.last_ack;
        packet.ack_bitfield = self.ack_bitfield;

        self.sent_packets.insert(
            seq,
            SentRecord {
                packet: packet.clone(),
                send_tick: self.current_tick,
                retransmit_count: 0,
            },
        );

        self.stats.packets_sent += 1;
        self.stats.bytes_sent += packet.wire_size() as u64;
        self.stats.cwnd = self.cwnd;

        Ok(packet)
    }

    /// Process a received packet: buffer, reorder, produce ack state.
    pub fn receive(&mut self, packet: Packet) -> Result<(), ReliableUdpError> {
        if self.closed {
            return Err(ReliableUdpError::ChannelClosed);
        }

        self.stats.packets_received += 1;
        self.stats.bytes_received += packet.wire_size() as u64;

        // Process ack information piggybacked on this packet.
        self.process_ack(packet.ack, packet.ack_bitfield);

        let seq = packet.sequence_number;

        // Update our ack state.
        if seq >= self.last_ack {
            // Shift bitfield for the gap.
            let diff = seq - self.last_ack;
            if diff > 0 {
                self.ack_bitfield = if diff <= 32 {
                    (self.ack_bitfield << diff) | (1 << (diff - 1))
                } else {
                    0
                };
            }
            self.last_ack = seq;
        } else {
            let diff = self.last_ack - seq;
            if diff > 0 && diff <= 32 {
                self.ack_bitfield |= 1 << (diff - 1);
            }
        }

        // Insert into receive buffer.
        if self.receive_buffer.contains_key(&seq) {
            return Err(ReliableUdpError::DuplicatePacket(seq));
        }
        self.receive_buffer.insert(seq, packet);

        // Deliver in-order packets.
        while let Some(pkt) = self.receive_buffer.remove(&self.expected_sequence) {
            self.delivered.push_back(pkt);
            self.expected_sequence = self.expected_sequence.wrapping_add(1);
        }

        // Trim receive buffer if too large.
        while self.receive_buffer.len() > self.config.receive_buffer_size {
            if let Some((&first, _)) = self.receive_buffer.iter().next() {
                self.receive_buffer.remove(&first);
            }
        }

        Ok(())
    }

    /// Drain delivered (in-order) packets.
    pub fn drain_delivered(&mut self) -> Vec<Packet> {
        self.delivered.drain(..).collect()
    }

    /// Get packets that need retransmission.
    pub fn retransmit(&mut self) -> Vec<Packet> {
        let timeout = self.config.retransmit_timeout_ticks;
        let max_retries = self.config.max_retransmissions;
        let tick = self.current_tick;

        let mut to_retransmit = Vec::new();
        let mut lost = Vec::new();

        for (&seq, record) in &self.sent_packets {
            if tick.saturating_sub(record.send_tick) >= timeout {
                if record.retransmit_count >= max_retries {
                    lost.push(seq);
                } else {
                    to_retransmit.push(seq);
                }
            }
        }

        for seq in &lost {
            self.sent_packets.remove(seq);
            self.stats.packets_lost += 1;
            // Multiplicative decrease on loss.
            self.ssthresh = (self.cwnd / 2).max(2);
            self.cwnd = self.ssthresh;
        }

        let mut packets = Vec::new();
        for seq in to_retransmit {
            if let Some(record) = self.sent_packets.get_mut(&seq) {
                record.retransmit_count += 1;
                record.send_tick = tick;
                packets.push(record.packet.clone());
                self.stats.retransmissions += 1;
            }
        }

        self.stats.cwnd = self.cwnd;
        packets
    }

    /// Number of packets delivered and available for reading.
    pub fn delivered_count(&self) -> usize {
        self.delivered.len()
    }

    /// Number of packets in flight (sent but not yet acked).
    pub fn in_flight(&self) -> usize {
        self.sent_packets.len()
    }

    /// Current congestion window size.
    pub fn congestion_window(&self) -> u32 {
        self.cwnd
    }

    /// Current RTT estimate in ms.
    pub fn rtt_estimate(&self) -> f64 {
        self.rtt_estimate_ms
    }

    /// Get channel statistics.
    pub fn stats(&self) -> &ChannelStats {
        &self.stats
    }

    /// Close the channel.
    pub fn close(&mut self) {
        self.closed = true;
    }

    /// Whether the channel is closed.
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    // ── Internal ────────────────────────────────────────────────

    fn process_ack(&mut self, ack: u32, ack_bitfield: u32) {
        // Ack the primary sequence number.
        self.ack_sequence(ack);

        // Ack previous packets indicated by bitfield.
        for i in 0..32u32 {
            if ack_bitfield & (1 << i) != 0 {
                if let Some(seq) = ack.checked_sub(i + 1) {
                    self.ack_sequence(seq);
                }
            }
        }
    }

    fn ack_sequence(&mut self, seq: u32) {
        if let Some(record) = self.sent_packets.remove(&seq) {
            self.stats.packets_acked += 1;

            // RTT estimation via EWMA.
            let sample = self.current_tick.saturating_sub(record.send_tick) as f64;
            if !self.rtt_initialized {
                self.rtt_estimate_ms = sample;
                self.rtt_initialized = true;
            } else {
                let alpha = self.config.rtt_alpha;
                self.rtt_estimate_ms = (1.0 - alpha) * self.rtt_estimate_ms + alpha * sample;
            }
            self.stats.rtt_ms = self.rtt_estimate_ms;

            // Congestion window growth.
            if self.cwnd < self.ssthresh {
                // Slow start: exponential growth.
                self.cwnd = (self.cwnd + 1).min(self.config.max_cwnd);
            } else {
                // Congestion avoidance: linear growth.
                // Increase by ~1/cwnd per ack (additive increase).
                self.cwnd = (self.cwnd + 1).min(self.config.max_cwnd);
            }
            self.stats.cwnd = self.cwnd;
        }
    }
}

impl fmt::Display for ReliableChannel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ReliableChannel(seq={}, cwnd={}, in_flight={}, rtt={:.1}ms)",
            self.next_sequence,
            self.cwnd,
            self.sent_packets.len(),
            self.rtt_estimate_ms,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_channel() -> ReliableChannel {
        ReliableChannel::new(ChannelConfig::default())
    }

    #[test]
    fn send_creates_packet_with_sequence() {
        let mut ch = default_channel();
        let pkt = ch.send(vec![1, 2, 3]).unwrap();
        assert_eq!(pkt.sequence_number, 0);
        assert_eq!(pkt.payload, vec![1, 2, 3]);
    }

    #[test]
    fn sequential_send_increments_sequence() {
        let mut ch = default_channel();
        let p0 = ch.send(vec![0]).unwrap();
        let p1 = ch.send(vec![1]).unwrap();
        let p2 = ch.send(vec![2]).unwrap();
        assert_eq!(p0.sequence_number, 0);
        assert_eq!(p1.sequence_number, 1);
        assert_eq!(p2.sequence_number, 2);
    }

    #[test]
    fn payload_too_large_rejected() {
        let mut ch = default_channel();
        let big = vec![0u8; 2000];
        let err = ch.send(big).unwrap_err();
        assert!(matches!(err, ReliableUdpError::PayloadTooLarge { .. }));
    }

    #[test]
    fn window_full_rejected() {
        let config = ChannelConfig::default().with_initial_cwnd(2);
        let mut ch = ReliableChannel::new(config);
        ch.send(vec![1]).unwrap();
        ch.send(vec![2]).unwrap();
        let err = ch.send(vec![3]).unwrap_err();
        assert!(matches!(err, ReliableUdpError::WindowFull { cwnd: 2 }));
    }

    #[test]
    fn receive_delivers_in_order() {
        let mut ch = default_channel();
        ch.receive(Packet::new(0, vec![10])).unwrap();
        ch.receive(Packet::new(1, vec![20])).unwrap();
        let delivered = ch.drain_delivered();
        assert_eq!(delivered.len(), 2);
        assert_eq!(delivered[0].payload, vec![10]);
        assert_eq!(delivered[1].payload, vec![20]);
    }

    #[test]
    fn out_of_order_receive_reorders() {
        let mut ch = default_channel();
        ch.receive(Packet::new(1, vec![20])).unwrap();
        assert_eq!(ch.drain_delivered().len(), 0);
        ch.receive(Packet::new(0, vec![10])).unwrap();
        let delivered = ch.drain_delivered();
        assert_eq!(delivered.len(), 2);
        assert_eq!(delivered[0].sequence_number, 0);
        assert_eq!(delivered[1].sequence_number, 1);
    }

    #[test]
    fn duplicate_packet_rejected() {
        let mut ch = default_channel();
        // Out-of-order: seq 1 arrives first, sits in receive buffer.
        ch.receive(Packet::new(1, vec![1])).unwrap();
        // Sending seq 1 again should be rejected as duplicate.
        let err = ch.receive(Packet::new(1, vec![1])).unwrap_err();
        assert!(matches!(err, ReliableUdpError::DuplicatePacket(1)));
    }

    #[test]
    fn ack_removes_in_flight() {
        let mut ch = default_channel();
        ch.send(vec![1]).unwrap();
        assert_eq!(ch.in_flight(), 1);
        // Simulate receiving a packet that acks seq 0.
        let mut ack_pkt = Packet::new(0, vec![]);
        ack_pkt.ack = 0;
        ch.receive(ack_pkt).unwrap();
        assert_eq!(ch.in_flight(), 0);
    }

    #[test]
    fn selective_ack_via_bitfield() {
        let mut ch = default_channel();
        ch.send(vec![0]).unwrap(); // seq 0
        ch.send(vec![1]).unwrap(); // seq 1
        ch.send(vec![2]).unwrap(); // seq 2
        assert_eq!(ch.in_flight(), 3);
        // Ack seq 2 with bitfield indicating seq 0 (bit 1 = seq 2-2=0).
        let mut ack_pkt = Packet::new(0, vec![]);
        ack_pkt.ack = 2;
        ack_pkt.ack_bitfield = 0b10; // bit 1 set => ack seq 2-2=0
        ch.receive(ack_pkt).unwrap();
        assert_eq!(ch.in_flight(), 1); // only seq 1 remains
    }

    #[test]
    fn retransmit_after_timeout() {
        let config = ChannelConfig::default().with_retransmit_timeout(10);
        let mut ch = ReliableChannel::new(config);
        ch.tick(0);
        ch.send(vec![42]).unwrap();
        ch.tick(5);
        assert!(ch.retransmit().is_empty());
        ch.tick(11);
        let retx = ch.retransmit();
        assert_eq!(retx.len(), 1);
        assert_eq!(retx[0].payload, vec![42]);
    }

    #[test]
    fn packet_lost_after_max_retransmissions() {
        let mut config = ChannelConfig::default();
        config.retransmit_timeout_ticks = 5;
        config.max_retransmissions = 2;
        let mut ch = ReliableChannel::new(config);
        ch.tick(0);
        ch.send(vec![1]).unwrap();
        // First retransmit.
        ch.tick(6);
        ch.retransmit();
        // Second retransmit.
        ch.tick(12);
        ch.retransmit();
        // Now it should be declared lost.
        ch.tick(18);
        ch.retransmit();
        assert_eq!(ch.stats().packets_lost, 1);
        assert_eq!(ch.in_flight(), 0);
    }

    #[test]
    fn rtt_estimate_updates() {
        let mut ch = default_channel();
        ch.tick(0);
        ch.send(vec![1]).unwrap();
        ch.tick(50);
        let mut ack = Packet::new(0, vec![]);
        ack.ack = 0;
        ch.receive(ack).unwrap();
        assert!((ch.rtt_estimate() - 50.0).abs() < 0.001);
    }

    #[test]
    fn rtt_ewma_smoothing() {
        let mut ch = default_channel();
        ch.tick(0);
        ch.send(vec![1]).unwrap();
        ch.tick(100);
        let mut ack1 = Packet::new(0, vec![]);
        ack1.ack = 0;
        ch.receive(ack1).unwrap();
        // RTT = 100 initially.
        ch.tick(100);
        ch.send(vec![2]).unwrap();
        ch.tick(120);
        let mut ack2 = Packet::new(1, vec![]);
        ack2.ack = 1;
        ch.receive(ack2).unwrap();
        // RTT should be smoothed between 100 and 20.
        assert!(ch.rtt_estimate() < 100.0);
        assert!(ch.rtt_estimate() > 20.0);
    }

    #[test]
    fn congestion_window_grows_on_ack() {
        let config = ChannelConfig::default().with_initial_cwnd(2);
        let mut ch = ReliableChannel::new(config);
        let initial = ch.congestion_window();
        ch.send(vec![1]).unwrap();
        let mut ack = Packet::new(0, vec![]);
        ack.ack = 0;
        ch.receive(ack).unwrap();
        assert!(ch.congestion_window() > initial);
    }

    #[test]
    fn congestion_window_shrinks_on_loss() {
        let mut config = ChannelConfig::default();
        config.retransmit_timeout_ticks = 5;
        config.max_retransmissions = 0;
        config.initial_cwnd = 10;
        let mut ch = ReliableChannel::new(config);
        ch.tick(0);
        ch.send(vec![1]).unwrap();
        ch.tick(10);
        ch.retransmit();
        assert!(ch.congestion_window() < 10);
    }

    #[test]
    fn stats_track_bytes() {
        let mut ch = default_channel();
        ch.send(vec![1, 2, 3, 4, 5]).unwrap();
        assert_eq!(ch.stats().bytes_sent, 17); // 12 header + 5 payload
        ch.receive(Packet::new(0, vec![10, 20])).unwrap();
        assert_eq!(ch.stats().bytes_received, 14); // 12 header + 2 payload
    }

    #[test]
    fn loss_rate_calculation() {
        let mut stats = ChannelStats::default();
        assert_eq!(stats.loss_rate(), 0.0);
        stats.packets_sent = 100;
        stats.packets_lost = 5;
        assert!((stats.loss_rate() - 0.05).abs() < 0.001);
    }

    #[test]
    fn bandwidth_estimate_positive() {
        let mut stats = ChannelStats::default();
        stats.rtt_ms = 50.0;
        stats.cwnd = 10;
        assert!(stats.bandwidth_estimate() > 0.0);
    }

    #[test]
    fn channel_close_prevents_send() {
        let mut ch = default_channel();
        ch.close();
        assert!(ch.is_closed());
        let err = ch.send(vec![1]).unwrap_err();
        assert!(matches!(err, ReliableUdpError::ChannelClosed));
    }

    #[test]
    fn channel_close_prevents_receive() {
        let mut ch = default_channel();
        ch.close();
        let err = ch.receive(Packet::new(0, vec![])).unwrap_err();
        assert!(matches!(err, ReliableUdpError::ChannelClosed));
    }

    #[test]
    fn packet_wire_size() {
        let pkt = Packet::new(0, vec![1, 2, 3]);
        assert_eq!(pkt.wire_size(), 15); // 12 + 3
    }

    #[test]
    fn packet_display() {
        let pkt = Packet::new(42, vec![0; 100]);
        let s = format!("{pkt}");
        assert!(s.contains("seq=42"));
        assert!(s.contains("100B"));
    }

    #[test]
    fn channel_display() {
        let ch = default_channel();
        let s = format!("{ch}");
        assert!(s.contains("ReliableChannel"));
        assert!(s.contains("cwnd="));
    }

    #[test]
    fn config_builder() {
        let config = ChannelConfig::default()
            .with_max_payload_size(500)
            .with_initial_cwnd(8)
            .with_retransmit_timeout(200);
        assert_eq!(config.max_payload_size, 500);
        assert_eq!(config.initial_cwnd, 8);
        assert_eq!(config.retransmit_timeout_ticks, 200);
    }
}
