//! Packet loss and network condition simulation for testing.
//!
//! Replaces Clumsy/tc/netem with a pure-Rust network simulator. Configures
//! loss%, latency, jitter, duplication, reordering, and bandwidth throttling.
//! Supports burst loss via Gilbert-Elliott model, correlated loss, deterministic
//! mode with seed, and tracks detailed statistics.

use std::collections::VecDeque;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Simulation errors.
#[derive(Debug, Clone, PartialEq)]
pub enum SimError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Simulator has been reset and has no packets.
    Empty,
}

impl fmt::Display for SimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::Empty => write!(f, "simulator queue empty"),
        }
    }
}

impl std::error::Error for SimError {}

// ── Network Condition ───────────────────────────────────────────

/// Network condition profile.
#[derive(Debug, Clone)]
pub struct NetworkCondition {
    /// Packet loss percentage [0.0, 1.0].
    pub loss_rate: f64,
    /// Base latency in seconds.
    pub latency: f64,
    /// Jitter (random latency variation) in seconds.
    pub jitter: f64,
    /// Duplicate packet rate [0.0, 1.0].
    pub duplicate_rate: f64,
    /// Reorder rate [0.0, 1.0].
    pub reorder_rate: f64,
    /// Bandwidth limit in bytes per second (0 = unlimited).
    pub bandwidth_bps: u64,
}

impl NetworkCondition {
    pub fn perfect() -> Self {
        Self {
            loss_rate: 0.0,
            latency: 0.0,
            jitter: 0.0,
            duplicate_rate: 0.0,
            reorder_rate: 0.0,
            bandwidth_bps: 0,
        }
    }

    pub fn lossy(loss: f64) -> Self {
        Self { loss_rate: loss.clamp(0.0, 1.0), ..Self::perfect() }
    }

    pub fn with_latency(mut self, latency: f64, jitter: f64) -> Self {
        self.latency = latency;
        self.jitter = jitter;
        self
    }

    pub fn with_duplicate(mut self, rate: f64) -> Self {
        self.duplicate_rate = rate.clamp(0.0, 1.0);
        self
    }

    pub fn with_reorder(mut self, rate: f64) -> Self {
        self.reorder_rate = rate.clamp(0.0, 1.0);
        self
    }

    pub fn with_bandwidth(mut self, bps: u64) -> Self {
        self.bandwidth_bps = bps;
        self
    }

    /// Validate the condition.
    pub fn validate(&self) -> Result<(), SimError> {
        if self.loss_rate < 0.0 || self.loss_rate > 1.0 {
            return Err(SimError::InvalidConfig(format!("loss_rate {} out of [0,1]", self.loss_rate)));
        }
        if self.latency < 0.0 {
            return Err(SimError::InvalidConfig("negative latency".into()));
        }
        Ok(())
    }
}

impl fmt::Display for NetworkCondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Loss: {:.1}%, Latency: {:.0}ms +/- {:.0}ms, Dup: {:.1}%, Reorder: {:.1}%",
            self.loss_rate * 100.0,
            self.latency * 1000.0,
            self.jitter * 1000.0,
            self.duplicate_rate * 100.0,
            self.reorder_rate * 100.0,
        )
    }
}

impl Default for NetworkCondition {
    fn default() -> Self {
        Self::perfect()
    }
}

// ── Gilbert-Elliott Model ───────────────────────────────────────

/// Gilbert-Elliott two-state burst loss model.
#[derive(Debug, Clone)]
pub struct GilbertElliott {
    /// Probability of transitioning from good to bad state.
    pub p_good_to_bad: f64,
    /// Probability of transitioning from bad to good state.
    pub p_bad_to_good: f64,
    /// Loss probability in good state.
    pub loss_in_good: f64,
    /// Loss probability in bad state.
    pub loss_in_bad: f64,
    /// Current state: false = good, true = bad.
    in_bad_state: bool,
}

impl GilbertElliott {
    pub fn new(p_gb: f64, p_bg: f64, loss_good: f64, loss_bad: f64) -> Self {
        Self {
            p_good_to_bad: p_gb.clamp(0.0, 1.0),
            p_bad_to_good: p_bg.clamp(0.0, 1.0),
            loss_in_good: loss_good.clamp(0.0, 1.0),
            loss_in_bad: loss_bad.clamp(0.0, 1.0),
            in_bad_state: false,
        }
    }

    /// Step the model and return whether this packet should be lost.
    pub fn step(&mut self, rng_val: f64, transition_val: f64) -> bool {
        // Transition.
        if self.in_bad_state {
            if transition_val < self.p_bad_to_good {
                self.in_bad_state = false;
            }
        } else if transition_val < self.p_good_to_bad {
            self.in_bad_state = true;
        }

        // Loss decision.
        let loss_prob = if self.in_bad_state { self.loss_in_bad } else { self.loss_in_good };
        rng_val < loss_prob
    }

    pub fn is_bad_state(&self) -> bool {
        self.in_bad_state
    }

    pub fn reset(&mut self) {
        self.in_bad_state = false;
    }
}

impl Default for GilbertElliott {
    fn default() -> Self {
        Self::new(0.05, 0.3, 0.0, 0.6)
    }
}

// ── Deterministic RNG ───────────────────────────────────────────

/// Simple deterministic PRNG (xorshift64) for reproducible simulations.
#[derive(Debug, Clone)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    pub fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    /// Returns a value in [0.0, 1.0).
    pub fn next_f64(&mut self) -> f64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state as f64) / (u64::MAX as f64)
    }

    /// Returns a value in [min, max).
    pub fn range(&mut self, min: f64, max: f64) -> f64 {
        min + self.next_f64() * (max - min)
    }
}

// ── Simulated Packet ────────────────────────────────────────────

/// A packet with simulation metadata.
#[derive(Debug, Clone)]
pub struct SimPacket {
    pub sequence: u64,
    pub payload: Vec<u8>,
    pub send_time: f64,
    pub delivery_time: f64,
    pub is_duplicate: bool,
}

impl SimPacket {
    pub fn new(sequence: u64, payload: Vec<u8>, send_time: f64) -> Self {
        Self { sequence, payload, send_time, delivery_time: send_time, is_duplicate: false }
    }

    pub fn size(&self) -> usize {
        self.payload.len()
    }
}

impl fmt::Display for SimPacket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SimPkt(seq={}, {}B, deliver={:.3}{})",
            self.sequence,
            self.payload.len(),
            self.delivery_time,
            if self.is_duplicate { " DUP" } else { "" }
        )
    }
}

// ── Simulation Stats ────────────────────────────────────────────

/// Statistics tracked by the simulator.
#[derive(Debug, Clone, Default)]
pub struct SimStats {
    pub packets_sent: u64,
    pub packets_delivered: u64,
    pub packets_dropped: u64,
    pub packets_duplicated: u64,
    pub packets_reordered: u64,
    pub total_bytes_sent: u64,
    pub total_latency: f64,
    pub burst_loss_events: u64,
}

impl SimStats {
    pub fn delivery_rate(&self) -> f64 {
        if self.packets_sent == 0 { 1.0 } else { self.packets_delivered as f64 / self.packets_sent as f64 }
    }

    pub fn avg_latency(&self) -> f64 {
        if self.packets_delivered == 0 { 0.0 } else { self.total_latency / self.packets_delivered as f64 }
    }

    pub fn loss_rate(&self) -> f64 {
        if self.packets_sent == 0 { 0.0 } else { self.packets_dropped as f64 / self.packets_sent as f64 }
    }
}

impl fmt::Display for SimStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Sent: {}, Delivered: {}, Dropped: {}, Dup: {}, AvgLat: {:.1}ms, Loss: {:.1}%",
            self.packets_sent,
            self.packets_delivered,
            self.packets_dropped,
            self.packets_duplicated,
            self.avg_latency() * 1000.0,
            self.loss_rate() * 100.0,
        )
    }
}

// ── Packet Simulator ────────────────────────────────────────────

/// Simulates network conditions for testing.
#[derive(Debug)]
pub struct PacketSimulator {
    condition: NetworkCondition,
    gilbert: Option<GilbertElliott>,
    rng: DeterministicRng,
    delivery_queue: VecDeque<SimPacket>,
    stats: SimStats,
    bandwidth_tokens: f64,
    last_send_time: f64,
}

impl PacketSimulator {
    pub fn new(condition: NetworkCondition, seed: u64) -> Self {
        Self {
            condition,
            gilbert: None,
            rng: DeterministicRng::new(seed),
            delivery_queue: VecDeque::new(),
            stats: SimStats::default(),
            bandwidth_tokens: 0.0,
            last_send_time: 0.0,
        }
    }

    pub fn with_gilbert_elliott(mut self, model: GilbertElliott) -> Self {
        self.gilbert = Some(model);
        self
    }

    /// Send a packet through the simulator.
    pub fn send(&mut self, sequence: u64, payload: Vec<u8>, send_time: f64) {
        self.stats.packets_sent += 1;
        self.stats.total_bytes_sent += payload.len() as u64;

        // Bandwidth throttling.
        if self.condition.bandwidth_bps > 0 {
            let elapsed = send_time - self.last_send_time;
            self.bandwidth_tokens += elapsed * self.condition.bandwidth_bps as f64;
            self.bandwidth_tokens = self.bandwidth_tokens.min(self.condition.bandwidth_bps as f64);

            if (payload.len() as f64) > self.bandwidth_tokens {
                self.stats.packets_dropped += 1;
                self.last_send_time = send_time;
                return;
            }
            self.bandwidth_tokens -= payload.len() as f64;
        }
        self.last_send_time = send_time;

        // Loss check (Gilbert-Elliott or simple).
        let should_drop = if let Some(ref mut ge) = self.gilbert {
            let rng_val = self.rng.next_f64();
            let trans_val = self.rng.next_f64();
            let lost = ge.step(rng_val, trans_val);
            if ge.is_bad_state() && lost {
                self.stats.burst_loss_events += 1;
            }
            lost
        } else {
            self.rng.next_f64() < self.condition.loss_rate
        };

        if should_drop {
            self.stats.packets_dropped += 1;
            return;
        }

        // Compute delivery time.
        let jitter_offset = if self.condition.jitter > 0.0 {
            self.rng.range(-self.condition.jitter, self.condition.jitter)
        } else {
            0.0
        };
        let delivery = send_time + self.condition.latency + jitter_offset;
        let delivery = delivery.max(send_time);

        let mut packet = SimPacket::new(sequence, payload.clone(), send_time);
        packet.delivery_time = delivery;
        let latency = delivery - send_time;
        self.stats.total_latency += latency;
        self.stats.packets_delivered += 1;

        self.delivery_queue.push_back(packet);

        // Reorder: swap with previous packet randomly.
        if self.condition.reorder_rate > 0.0
            && self.rng.next_f64() < self.condition.reorder_rate
            && self.delivery_queue.len() >= 2
        {
            let n = self.delivery_queue.len();
            self.delivery_queue.swap(n - 1, n - 2);
            self.stats.packets_reordered += 1;
        }

        // Duplication.
        if self.condition.duplicate_rate > 0.0 && self.rng.next_f64() < self.condition.duplicate_rate {
            let mut dup = SimPacket::new(sequence, payload, send_time);
            dup.delivery_time = delivery + self.rng.range(0.0, self.condition.latency * 0.5 + 0.001);
            dup.is_duplicate = true;
            self.delivery_queue.push_back(dup);
            self.stats.packets_duplicated += 1;
        }
    }

    /// Collect all packets ready for delivery at or before the given time.
    pub fn collect(&mut self, current_time: f64) -> Vec<SimPacket> {
        let mut ready = Vec::new();
        let mut remaining = VecDeque::new();

        while let Some(pkt) = self.delivery_queue.pop_front() {
            if pkt.delivery_time <= current_time {
                ready.push(pkt);
            } else {
                remaining.push_back(pkt);
            }
        }
        self.delivery_queue = remaining;
        ready.sort_by(|a, b| a.delivery_time.partial_cmp(&b.delivery_time).unwrap());
        ready
    }

    /// Peek at queued packet count.
    pub fn queued_count(&self) -> usize {
        self.delivery_queue.len()
    }

    pub fn stats(&self) -> &SimStats {
        &self.stats
    }

    pub fn condition(&self) -> &NetworkCondition {
        &self.condition
    }

    pub fn set_condition(&mut self, condition: NetworkCondition) {
        self.condition = condition;
    }

    pub fn reset(&mut self) {
        self.delivery_queue.clear();
        self.stats = SimStats::default();
        self.bandwidth_tokens = 0.0;
        if let Some(ref mut ge) = self.gilbert {
            ge.reset();
        }
    }
}

// ── Preset Conditions ───────────────────────────────────────────

/// Common network condition presets for testing.
pub struct Presets;

impl Presets {
    /// LAN conditions: ~1ms latency, no loss.
    pub fn lan() -> NetworkCondition {
        NetworkCondition::perfect().with_latency(0.001, 0.0005)
    }

    /// Good broadband: ~30ms latency, 0.1% loss.
    pub fn broadband() -> NetworkCondition {
        NetworkCondition::lossy(0.001).with_latency(0.030, 0.005)
    }

    /// Mobile 4G: ~50ms latency, 1% loss, some jitter.
    pub fn mobile_4g() -> NetworkCondition {
        NetworkCondition::lossy(0.01).with_latency(0.050, 0.020)
    }

    /// Satellite: ~600ms latency, 2% loss.
    pub fn satellite() -> NetworkCondition {
        NetworkCondition::lossy(0.02).with_latency(0.600, 0.050)
    }

    /// Terrible: 20% loss, 200ms latency, high jitter.
    pub fn terrible() -> NetworkCondition {
        NetworkCondition::lossy(0.20)
            .with_latency(0.200, 0.100)
            .with_duplicate(0.05)
            .with_reorder(0.10)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_conditions_no_loss() {
        let mut sim = PacketSimulator::new(NetworkCondition::perfect(), 42);
        for i in 0..100 {
            sim.send(i, vec![0; 64], i as f64 * 0.016);
        }
        assert_eq!(sim.stats().packets_dropped, 0);
        assert_eq!(sim.stats().packets_delivered, 100);
    }

    #[test]
    fn lossy_drops_packets() {
        let mut sim = PacketSimulator::new(NetworkCondition::lossy(0.5), 42);
        for i in 0..1000 {
            sim.send(i, vec![0; 64], i as f64 * 0.016);
        }
        // With 50% loss, should drop roughly half.
        let rate = sim.stats().loss_rate();
        assert!(rate > 0.3 && rate < 0.7, "loss rate was {rate}");
    }

    #[test]
    fn latency_applied() {
        let cond = NetworkCondition::perfect().with_latency(0.100, 0.0);
        let mut sim = PacketSimulator::new(cond, 42);
        sim.send(0, vec![0; 64], 0.0);
        let early = sim.collect(0.050);
        assert!(early.is_empty());
        let late = sim.collect(0.150);
        assert_eq!(late.len(), 1);
    }

    #[test]
    fn duplication_creates_copies() {
        let cond = NetworkCondition::perfect().with_duplicate(1.0);
        let mut sim = PacketSimulator::new(cond, 42);
        sim.send(0, vec![0; 64], 0.0);
        let pkts = sim.collect(100.0);
        assert_eq!(pkts.len(), 2);
        assert!(pkts.iter().any(|p| p.is_duplicate));
    }

    #[test]
    fn reorder_swaps_packets() {
        let cond = NetworkCondition::perfect().with_reorder(1.0);
        let mut sim = PacketSimulator::new(cond, 42);
        sim.send(0, vec![0; 64], 0.0);
        sim.send(1, vec![0; 64], 0.016);
        let pkts = sim.collect(100.0);
        assert_eq!(pkts.len(), 2);
        // At least one reorder should have happened.
        assert!(sim.stats().packets_reordered >= 1);
    }

    #[test]
    fn gilbert_elliott_burst() {
        let ge = GilbertElliott::new(0.3, 0.5, 0.0, 1.0);
        let cond = NetworkCondition::perfect();
        let mut sim = PacketSimulator::new(cond, 42).with_gilbert_elliott(ge);
        for i in 0..500 {
            sim.send(i, vec![0; 64], i as f64 * 0.016);
        }
        assert!(sim.stats().packets_dropped > 0);
        assert!(sim.stats().burst_loss_events > 0);
    }

    #[test]
    fn deterministic_rng_reproducible() {
        let mut rng1 = DeterministicRng::new(12345);
        let mut rng2 = DeterministicRng::new(12345);
        for _ in 0..100 {
            assert_eq!(rng1.next_f64(), rng2.next_f64());
        }
    }

    #[test]
    fn deterministic_rng_range() {
        let mut rng = DeterministicRng::new(42);
        for _ in 0..100 {
            let v = rng.range(10.0, 20.0);
            assert!(v >= 10.0 && v < 20.0);
        }
    }

    #[test]
    fn bandwidth_throttle() {
        let cond = NetworkCondition::perfect().with_bandwidth(1000); // 1000 B/s
        let mut sim = PacketSimulator::new(cond, 42);
        // Send 10 packets of 200B at time 0 — should throttle.
        for i in 0..10 {
            sim.send(i, vec![0; 200], 0.0);
        }
        assert!(sim.stats().packets_dropped > 0);
    }

    #[test]
    fn network_condition_validate() {
        let ok = NetworkCondition::lossy(0.5);
        assert!(ok.validate().is_ok());

        let mut bad = NetworkCondition::perfect();
        bad.loss_rate = 2.0;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn network_condition_display() {
        let cond = NetworkCondition::lossy(0.05).with_latency(0.050, 0.010);
        let s = format!("{cond}");
        assert!(s.contains("5.0%"));
        assert!(s.contains("50ms"));
    }

    #[test]
    fn preset_lan() {
        let cond = Presets::lan();
        assert!((cond.loss_rate).abs() < 1e-9);
        assert!(cond.latency < 0.01);
    }

    #[test]
    fn preset_terrible() {
        let cond = Presets::terrible();
        assert!(cond.loss_rate > 0.1);
        assert!(cond.duplicate_rate > 0.0);
    }

    #[test]
    fn sim_stats_display() {
        let stats = SimStats {
            packets_sent: 100,
            packets_delivered: 90,
            packets_dropped: 10,
            packets_duplicated: 2,
            packets_reordered: 5,
            total_bytes_sent: 6400,
            total_latency: 4.5,
            burst_loss_events: 1,
        };
        let s = format!("{stats}");
        assert!(s.contains("Sent: 100"));
    }

    #[test]
    fn sim_packet_display() {
        let p = SimPacket::new(42, vec![0; 100], 0.0);
        let s = format!("{p}");
        assert!(s.contains("seq=42"));
    }

    #[test]
    fn simulator_reset() {
        let mut sim = PacketSimulator::new(NetworkCondition::perfect(), 42);
        sim.send(0, vec![0; 64], 0.0);
        sim.reset();
        assert_eq!(sim.queued_count(), 0);
        assert_eq!(sim.stats().packets_sent, 0);
    }

    #[test]
    fn gilbert_elliott_reset() {
        let mut ge = GilbertElliott::new(1.0, 0.0, 0.0, 1.0);
        // Force into bad state.
        ge.step(0.5, 0.5);
        ge.reset();
        assert!(!ge.is_bad_state());
    }

    #[test]
    fn collect_preserves_unready_packets() {
        let cond = NetworkCondition::perfect().with_latency(1.0, 0.0);
        let mut sim = PacketSimulator::new(cond, 42);
        sim.send(0, vec![0; 64], 0.0);
        sim.send(1, vec![0; 64], 0.5);
        let early = sim.collect(0.8);
        assert!(early.is_empty()); // both need latency 1.0s
        let later = sim.collect(1.5);
        assert_eq!(later.len(), 2);
    }
}
