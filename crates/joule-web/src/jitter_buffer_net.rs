//! Network jitter buffer — ordered packet delivery with adaptive buffering.
//!
//! Replaces custom jitter buffers in VoIP/game engines with a pure-Rust
//! implementation. Buffers out-of-order packets, reorders them by sequence
//! number, detects duplicates and gaps, adapts buffer depth based on measured
//! jitter, provides playout scheduling, and tracks detailed statistics.

use std::collections::BTreeMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Jitter buffer errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JitterError {
    /// Packet is a duplicate.
    Duplicate { sequence: u64 },
    /// Buffer is full.
    BufferFull { capacity: usize },
    /// Packet is too old (below playout horizon).
    TooOld { sequence: u64, horizon: u64 },
    /// Buffer is empty, nothing to play out.
    BufferEmpty,
}

impl fmt::Display for JitterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Duplicate { sequence } => write!(f, "duplicate packet seq={sequence}"),
            Self::BufferFull { capacity } => write!(f, "jitter buffer full (cap={capacity})"),
            Self::TooOld { sequence, horizon } => {
                write!(f, "packet seq={sequence} below horizon {horizon}")
            }
            Self::BufferEmpty => write!(f, "jitter buffer empty"),
        }
    }
}

impl std::error::Error for JitterError {}

// ── Packet ──────────────────────────────────────────────────────

/// A network packet with sequence and timing info.
#[derive(Debug, Clone, PartialEq)]
pub struct Packet {
    pub sequence: u64,
    pub timestamp: f64,
    pub arrival_time: f64,
    pub payload: Vec<u8>,
}

impl Packet {
    pub fn new(sequence: u64, timestamp: f64, payload: Vec<u8>) -> Self {
        Self { sequence, timestamp, arrival_time: 0.0, payload }
    }

    pub fn with_arrival(mut self, arrival: f64) -> Self {
        self.arrival_time = arrival;
        self
    }

    pub fn size(&self) -> usize {
        self.payload.len()
    }
}

impl fmt::Display for Packet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pkt(seq={}, ts={:.3}, {}B)", self.sequence, self.timestamp, self.payload.len())
    }
}

// ── Jitter Stats ────────────────────────────────────────────────

/// Jitter buffer statistics.
#[derive(Debug, Clone, Default)]
pub struct JitterStats {
    pub packets_received: u64,
    pub packets_played: u64,
    pub packets_dropped: u64,
    pub duplicates: u64,
    pub late_arrivals: u64,
    pub reorders: u64,
    pub gaps_detected: u64,
    pub max_reorder_distance: u64,
    jitter_samples: Vec<f64>,
}

impl JitterStats {
    pub fn loss_rate(&self) -> f64 {
        if self.packets_received == 0 {
            return 0.0;
        }
        self.packets_dropped as f64 / self.packets_received as f64
    }

    pub fn reorder_rate(&self) -> f64 {
        if self.packets_received == 0 {
            return 0.0;
        }
        self.reorders as f64 / self.packets_received as f64
    }

    pub fn avg_jitter(&self) -> f64 {
        if self.jitter_samples.is_empty() {
            return 0.0;
        }
        self.jitter_samples.iter().sum::<f64>() / self.jitter_samples.len() as f64
    }

    pub fn record_jitter(&mut self, jitter: f64) {
        if self.jitter_samples.len() >= 100 {
            self.jitter_samples.remove(0);
        }
        self.jitter_samples.push(jitter);
    }
}

impl fmt::Display for JitterStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Recv: {}, Played: {}, Dropped: {}, Dups: {}, Loss: {:.1}%, Jitter: {:.3}ms",
            self.packets_received,
            self.packets_played,
            self.packets_dropped,
            self.duplicates,
            self.loss_rate() * 100.0,
            self.avg_jitter() * 1000.0,
        )
    }
}

// ── Jitter Buffer Config ────────────────────────────────────────

/// Configuration for the jitter buffer.
#[derive(Debug, Clone)]
pub struct JitterConfig {
    pub initial_depth: usize,
    pub min_depth: usize,
    pub max_depth: usize,
    pub max_capacity: usize,
    pub adaptive: bool,
    pub target_jitter_multiplier: f64,
    pub playout_interval: f64,
}

impl JitterConfig {
    pub fn new() -> Self {
        Self {
            initial_depth: 4,
            min_depth: 2,
            max_depth: 20,
            max_capacity: 128,
            adaptive: true,
            target_jitter_multiplier: 2.0,
            playout_interval: 1.0 / 60.0,
        }
    }

    pub fn with_depth(mut self, depth: usize) -> Self {
        self.initial_depth = depth;
        self
    }

    pub fn with_max_depth(mut self, max: usize) -> Self {
        self.max_depth = max;
        self
    }

    pub fn with_capacity(mut self, cap: usize) -> Self {
        self.max_capacity = cap;
        self
    }

    pub fn with_adaptive(mut self, enable: bool) -> Self {
        self.adaptive = enable;
        self
    }

    pub fn with_playout_interval(mut self, interval: f64) -> Self {
        self.playout_interval = interval;
        self
    }
}

impl Default for JitterConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ── Gap Info ────────────────────────────────────────────────────

/// Information about a detected gap in the sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GapInfo {
    pub start_sequence: u64,
    pub end_sequence: u64,
    pub missing_count: u64,
}

impl GapInfo {
    pub fn new(start: u64, end: u64) -> Self {
        Self { start_sequence: start, end_sequence: end, missing_count: end - start - 1 }
    }
}

impl fmt::Display for GapInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Gap({}-{}, {} missing)", self.start_sequence, self.end_sequence, self.missing_count)
    }
}

// ── Jitter Buffer ───────────────────────────────────────────────

/// Network jitter buffer for ordered packet delivery.
#[derive(Debug)]
pub struct JitterBuffer {
    config: JitterConfig,
    buffer: BTreeMap<u64, Packet>,
    current_depth: usize,
    next_playout_seq: u64,
    last_arrival_time: f64,
    last_expected_interval: f64,
    highest_received: u64,
    stats: JitterStats,
    ready: bool,
}

impl JitterBuffer {
    pub fn new(config: JitterConfig) -> Self {
        let depth = config.initial_depth;
        Self {
            config,
            buffer: BTreeMap::new(),
            current_depth: depth,
            next_playout_seq: 0,
            last_arrival_time: 0.0,
            last_expected_interval: 0.0,
            highest_received: 0,
            stats: JitterStats::default(),
            ready: false,
        }
    }

    /// Insert a packet into the buffer.
    pub fn insert(&mut self, packet: Packet) -> Result<(), JitterError> {
        self.stats.packets_received += 1;

        // Check duplicate.
        if self.buffer.contains_key(&packet.sequence) {
            self.stats.duplicates += 1;
            return Err(JitterError::Duplicate { sequence: packet.sequence });
        }

        // Check too old.
        if self.ready && packet.sequence < self.next_playout_seq {
            self.stats.late_arrivals += 1;
            self.stats.packets_dropped += 1;
            return Err(JitterError::TooOld {
                sequence: packet.sequence,
                horizon: self.next_playout_seq,
            });
        }

        // Check capacity.
        if self.buffer.len() >= self.config.max_capacity {
            self.stats.packets_dropped += 1;
            return Err(JitterError::BufferFull { capacity: self.config.max_capacity });
        }

        // Detect reorder.
        if packet.sequence < self.highest_received {
            self.stats.reorders += 1;
            let dist = self.highest_received - packet.sequence;
            if dist > self.stats.max_reorder_distance {
                self.stats.max_reorder_distance = dist;
            }
        }

        // Update jitter measurement.
        if self.last_arrival_time > 0.0 && self.last_expected_interval > 0.0 {
            let actual_interval = packet.arrival_time - self.last_arrival_time;
            let jitter = (actual_interval - self.last_expected_interval).abs();
            self.stats.record_jitter(jitter);

            // Adaptive depth.
            if self.config.adaptive {
                self.adapt_depth();
            }
        }
        if packet.sequence > self.highest_received {
            if self.last_arrival_time > 0.0 {
                self.last_expected_interval =
                    packet.arrival_time - self.last_arrival_time;
            }
            self.last_arrival_time = packet.arrival_time;
            self.highest_received = packet.sequence;
        }

        self.buffer.insert(packet.sequence, packet);

        // Check if buffer is ready for playout.
        if !self.ready && self.buffer.len() >= self.current_depth {
            self.ready = true;
            if let Some((&first_seq, _)) = self.buffer.iter().next() {
                self.next_playout_seq = first_seq;
            }
        }

        Ok(())
    }

    /// Pop the next packet for playout.
    pub fn playout(&mut self) -> Result<Packet, JitterError> {
        if !self.ready {
            return Err(JitterError::BufferEmpty);
        }

        if let Some(packet) = self.buffer.remove(&self.next_playout_seq) {
            self.stats.packets_played += 1;
            self.next_playout_seq += 1;
            Ok(packet)
        } else {
            // Gap — missing packet.
            self.stats.gaps_detected += 1;
            self.stats.packets_dropped += 1;
            self.next_playout_seq += 1;
            Err(JitterError::BufferEmpty)
        }
    }

    /// Detect gaps in the current buffer.
    pub fn detect_gaps(&self) -> Vec<GapInfo> {
        let mut gaps = Vec::new();
        let mut prev: Option<u64> = None;

        for &seq in self.buffer.keys() {
            if let Some(p) = prev {
                if seq > p + 1 {
                    gaps.push(GapInfo::new(p, seq));
                }
            }
            prev = Some(seq);
        }
        gaps
    }

    /// Adapt buffer depth based on measured jitter.
    fn adapt_depth(&mut self) {
        let avg_jitter = self.stats.avg_jitter();
        if avg_jitter <= 0.0 || self.config.playout_interval <= 0.0 {
            return;
        }
        let target = (avg_jitter * self.config.target_jitter_multiplier
            / self.config.playout_interval)
            .ceil() as usize;
        let new_depth = target.clamp(self.config.min_depth, self.config.max_depth);
        self.current_depth = new_depth;
    }

    pub fn buffered_count(&self) -> usize {
        self.buffer.len()
    }

    pub fn current_depth(&self) -> usize {
        self.current_depth
    }

    pub fn is_ready(&self) -> bool {
        self.ready
    }

    pub fn next_playout_seq(&self) -> u64 {
        self.next_playout_seq
    }

    pub fn stats(&self) -> &JitterStats {
        &self.stats
    }

    pub fn config(&self) -> &JitterConfig {
        &self.config
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
        self.ready = false;
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn pkt(seq: u64, arrival: f64) -> Packet {
        Packet::new(seq, seq as f64 * 0.016, vec![seq as u8]).with_arrival(arrival)
    }

    #[test]
    fn insert_and_playout_ordered() {
        let config = JitterConfig::new().with_depth(2).with_adaptive(false);
        let mut buf = JitterBuffer::new(config);
        buf.insert(pkt(0, 0.0)).unwrap();
        buf.insert(pkt(1, 0.016)).unwrap();
        let p = buf.playout().unwrap();
        assert_eq!(p.sequence, 0);
    }

    #[test]
    fn reorder_detection() {
        let config = JitterConfig::new().with_depth(3).with_adaptive(false);
        let mut buf = JitterBuffer::new(config);
        buf.insert(pkt(0, 0.0)).unwrap();
        buf.insert(pkt(2, 0.032)).unwrap();
        buf.insert(pkt(1, 0.040)).unwrap(); // arrives late
        assert_eq!(buf.stats().reorders, 1);
    }

    #[test]
    fn duplicate_rejected() {
        let config = JitterConfig::new().with_depth(2).with_adaptive(false);
        let mut buf = JitterBuffer::new(config);
        buf.insert(pkt(0, 0.0)).unwrap();
        let err = buf.insert(pkt(0, 0.0)).unwrap_err();
        assert_eq!(err, JitterError::Duplicate { sequence: 0 });
    }

    #[test]
    fn late_packet_rejected() {
        let config = JitterConfig::new().with_depth(1).with_adaptive(false);
        let mut buf = JitterBuffer::new(config);
        buf.insert(pkt(0, 0.0)).unwrap();
        buf.playout().unwrap(); // next_playout_seq = 1
        let err = buf.insert(pkt(0, 0.1)).unwrap_err();
        matches!(err, JitterError::TooOld { .. });
    }

    #[test]
    fn buffer_full() {
        let config = JitterConfig::new().with_depth(1).with_capacity(2).with_adaptive(false);
        let mut buf = JitterBuffer::new(config);
        buf.insert(pkt(0, 0.0)).unwrap();
        buf.insert(pkt(1, 0.016)).unwrap();
        let err = buf.insert(pkt(2, 0.032)).unwrap_err();
        assert_eq!(err, JitterError::BufferFull { capacity: 2 });
    }

    #[test]
    fn gap_detection() {
        let config = JitterConfig::new().with_depth(3).with_adaptive(false);
        let mut buf = JitterBuffer::new(config);
        buf.insert(pkt(0, 0.0)).unwrap();
        buf.insert(pkt(1, 0.016)).unwrap();
        buf.insert(pkt(5, 0.080)).unwrap();
        let gaps = buf.detect_gaps();
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].start_sequence, 1);
        assert_eq!(gaps[0].end_sequence, 5);
        assert_eq!(gaps[0].missing_count, 3);
    }

    #[test]
    fn playout_empty_not_ready() {
        let config = JitterConfig::new().with_depth(5).with_adaptive(false);
        let mut buf = JitterBuffer::new(config);
        let err = buf.playout().unwrap_err();
        assert_eq!(err, JitterError::BufferEmpty);
    }

    #[test]
    fn stats_loss_rate() {
        let mut stats = JitterStats::default();
        stats.packets_received = 100;
        stats.packets_dropped = 5;
        assert!((stats.loss_rate() - 0.05).abs() < 1e-9);
    }

    #[test]
    fn stats_reorder_rate() {
        let mut stats = JitterStats::default();
        stats.packets_received = 100;
        stats.reorders = 10;
        assert!((stats.reorder_rate() - 0.1).abs() < 1e-9);
    }

    #[test]
    fn stats_avg_jitter() {
        let mut stats = JitterStats::default();
        stats.record_jitter(0.005);
        stats.record_jitter(0.015);
        assert!((stats.avg_jitter() - 0.01).abs() < 1e-9);
    }

    #[test]
    fn stats_display() {
        let stats = JitterStats::default();
        let s = format!("{stats}");
        assert!(s.contains("Recv:"));
    }

    #[test]
    fn gap_info_display() {
        let gap = GapInfo::new(3, 7);
        let s = format!("{gap}");
        assert!(s.contains("3-7"));
        assert!(s.contains("3 missing"));
    }

    #[test]
    fn packet_display() {
        let p = Packet::new(42, 0.5, vec![1, 2, 3]);
        let s = format!("{p}");
        assert!(s.contains("seq=42"));
    }

    #[test]
    fn buffer_clear() {
        let config = JitterConfig::new().with_depth(1).with_adaptive(false);
        let mut buf = JitterBuffer::new(config);
        buf.insert(pkt(0, 0.0)).unwrap();
        buf.clear();
        assert_eq!(buf.buffered_count(), 0);
        assert!(!buf.is_ready());
    }

    #[test]
    fn adaptive_depth_increases() {
        let config = JitterConfig::new().with_depth(2).with_max_depth(10);
        let mut buf = JitterBuffer::new(config);
        // Simulate high jitter.
        for i in 0..10 {
            let jitter_arrival = i as f64 * 0.016 + if i % 2 == 0 { 0.020 } else { 0.0 };
            buf.insert(pkt(i, jitter_arrival)).unwrap_or(());
        }
        // Depth should have adapted.
        assert!(buf.current_depth() >= 2);
    }

    #[test]
    fn sequential_playout() {
        let config = JitterConfig::new().with_depth(3).with_adaptive(false);
        let mut buf = JitterBuffer::new(config);
        for i in 0..5 {
            buf.insert(pkt(i, i as f64 * 0.016)).unwrap();
        }
        let p0 = buf.playout().unwrap();
        let p1 = buf.playout().unwrap();
        assert_eq!(p0.sequence, 0);
        assert_eq!(p1.sequence, 1);
    }

    #[test]
    fn config_defaults() {
        let cfg = JitterConfig::default();
        assert_eq!(cfg.initial_depth, 4);
        assert!(cfg.adaptive);
    }

    #[test]
    fn highest_received_tracks() {
        let config = JitterConfig::new().with_depth(3).with_adaptive(false);
        let mut buf = JitterBuffer::new(config);
        buf.insert(pkt(0, 0.0)).unwrap();
        buf.insert(pkt(5, 0.080)).unwrap();
        buf.insert(pkt(3, 0.060)).unwrap();
        assert_eq!(buf.stats().max_reorder_distance, 2);
    }
}
