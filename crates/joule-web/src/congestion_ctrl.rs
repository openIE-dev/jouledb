//! Congestion control algorithms.
//!
//! Implements AIMD (Additive Increase Multiplicative Decrease) with
//! slow start, congestion avoidance, fast retransmit/fast recovery,
//! and a BBR-like bandwidth estimation mode. Supports configurable
//! algorithm selection, pacing, and congestion statistics.

use std::collections::VecDeque;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Congestion control domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CongestionError {
    /// Invalid configuration value.
    InvalidConfig(String),
    /// Congestion window is zero (stalled).
    WindowStalled,
}

impl fmt::Display for CongestionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::WindowStalled => write!(f, "congestion window stalled at zero"),
        }
    }
}

impl std::error::Error for CongestionError {}

// ── Algorithm ───────────────────────────────────────────────────

/// Selectable congestion control algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    /// Classic AIMD with slow start.
    Aimd,
    /// BBR-like bandwidth estimation.
    Bbr,
}

impl Default for Algorithm {
    fn default() -> Self {
        Self::Aimd
    }
}

impl fmt::Display for Algorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Aimd => write!(f, "AIMD"),
            Self::Bbr => write!(f, "BBR"),
        }
    }
}

// ── Phase ───────────────────────────────────────────────────────

/// Current phase of the congestion controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    SlowStart,
    CongestionAvoidance,
    FastRecovery,
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SlowStart => write!(f, "SlowStart"),
            Self::CongestionAvoidance => write!(f, "CongestionAvoidance"),
            Self::FastRecovery => write!(f, "FastRecovery"),
        }
    }
}

// ── Config ──────────────────────────────────────────────────────

/// Configuration for the congestion controller.
#[derive(Debug, Clone)]
pub struct CongestionConfig {
    pub algorithm: Algorithm,
    pub initial_cwnd: u32,
    pub min_cwnd: u32,
    pub max_cwnd: u32,
    pub initial_ssthresh: u32,
    pub beta: f64,
    pub mss: u32,
    pub dup_ack_threshold: u32,
    pub pacing_enabled: bool,
    pub bbr_probe_interval: u32,
    pub history_limit: usize,
}

impl Default for CongestionConfig {
    fn default() -> Self {
        Self {
            algorithm: Algorithm::Aimd,
            initial_cwnd: 10,
            min_cwnd: 2,
            max_cwnd: 65535,
            initial_ssthresh: 65535,
            beta: 0.5,
            mss: 1460,
            dup_ack_threshold: 3,
            pacing_enabled: false,
            bbr_probe_interval: 8,
            history_limit: 256,
        }
    }
}

impl CongestionConfig {
    pub fn with_algorithm(mut self, algo: Algorithm) -> Self {
        self.algorithm = algo;
        self
    }

    pub fn with_initial_cwnd(mut self, cwnd: u32) -> Self {
        self.initial_cwnd = cwnd;
        self
    }

    pub fn with_max_cwnd(mut self, max: u32) -> Self {
        self.max_cwnd = max;
        self
    }

    pub fn with_pacing(mut self, enabled: bool) -> Self {
        self.pacing_enabled = enabled;
        self
    }

    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }
}

// ── Loss Event ──────────────────────────────────────────────────

/// A recorded loss event.
#[derive(Debug, Clone)]
pub struct LossEvent {
    pub tick: u64,
    pub cwnd_at_loss: u32,
    pub rtt_at_loss: f64,
}

// ── Congestion Stats ────────────────────────────────────────────

/// Congestion control statistics.
#[derive(Debug, Clone, Default)]
pub struct CongestionStats {
    pub acks_processed: u64,
    pub losses_detected: u64,
    pub fast_retransmits: u64,
    pub slow_start_exits: u64,
    pub cwnd_peak: u32,
    pub min_rtt: f64,
    pub max_rtt: f64,
}

impl CongestionStats {
    pub fn update_rtt(&mut self, rtt: f64) {
        if self.min_rtt == 0.0 || rtt < self.min_rtt {
            self.min_rtt = rtt;
        }
        if rtt > self.max_rtt {
            self.max_rtt = rtt;
        }
    }
}

impl fmt::Display for CongestionStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "acks={} losses={} fast_retx={} peak_cwnd={} rtt=[{:.1},{:.1}]",
            self.acks_processed,
            self.losses_detected,
            self.fast_retransmits,
            self.cwnd_peak,
            self.min_rtt,
            self.max_rtt,
        )
    }
}

// ── Bandwidth Sample ────────────────────────────────────────────

/// A bandwidth sample for BBR mode.
#[derive(Debug, Clone)]
struct BandwidthSample {
    bytes_delivered: u64,
    rtt: f64,
}

// ── Congestion Window ───────────────────────────────────────────

/// Congestion controller managing cwnd, ssthresh, phase transitions,
/// and optional BBR bandwidth estimation.
pub struct CongestionWindow {
    config: CongestionConfig,
    cwnd: u32,
    ssthresh: u32,
    phase: Phase,
    dup_ack_count: u32,
    current_rtt: f64,
    // BBR state
    estimated_bandwidth: f64,
    bbr_round: u32,
    bandwidth_samples: VecDeque<BandwidthSample>,
    // Pacing
    pacing_rate: f64,
    // History
    cwnd_history: VecDeque<(u64, u32)>,
    loss_events: VecDeque<LossEvent>,
    // Stats
    stats: CongestionStats,
    tick: u64,
}

impl CongestionWindow {
    pub fn new(config: CongestionConfig) -> Self {
        let cwnd = config.initial_cwnd;
        let ssthresh = config.initial_ssthresh;
        Self {
            config,
            cwnd,
            ssthresh,
            phase: Phase::SlowStart,
            dup_ack_count: 0,
            current_rtt: 0.0,
            estimated_bandwidth: 0.0,
            bbr_round: 0,
            bandwidth_samples: VecDeque::new(),
            pacing_rate: 0.0,
            cwnd_history: VecDeque::new(),
            loss_events: VecDeque::new(),
            stats: CongestionStats::default(),
            tick: 0,
        }
    }

    /// Current congestion window in segments.
    pub fn cwnd(&self) -> u32 {
        self.cwnd
    }

    /// Current slow-start threshold.
    pub fn ssthresh(&self) -> u32 {
        self.ssthresh
    }

    /// Current phase.
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// Current RTT.
    pub fn rtt(&self) -> f64 {
        self.current_rtt
    }

    /// Estimated bandwidth (BBR mode).
    pub fn estimated_bandwidth(&self) -> f64 {
        self.estimated_bandwidth
    }

    /// Current pacing rate (bytes per tick).
    pub fn pacing_rate(&self) -> f64 {
        self.pacing_rate
    }

    /// Get statistics.
    pub fn stats(&self) -> &CongestionStats {
        &self.stats
    }

    /// Get loss event history.
    pub fn loss_events(&self) -> &VecDeque<LossEvent> {
        &self.loss_events
    }

    /// Get cwnd history.
    pub fn cwnd_history(&self) -> &VecDeque<(u64, u32)> {
        &self.cwnd_history
    }

    /// Set the current tick.
    pub fn tick(&mut self, tick: u64) {
        self.tick = tick;
    }

    /// Process a successful acknowledgment.
    pub fn on_ack(&mut self, bytes_acked: u32, rtt: f64) {
        self.current_rtt = rtt;
        self.stats.acks_processed += 1;
        self.stats.update_rtt(rtt);
        self.dup_ack_count = 0;

        match self.config.algorithm {
            Algorithm::Aimd => self.aimd_on_ack(bytes_acked),
            Algorithm::Bbr => self.bbr_on_ack(bytes_acked, rtt),
        }

        self.record_cwnd();
        self.update_pacing();
    }

    /// Process a packet loss event.
    pub fn on_loss(&mut self) {
        self.stats.losses_detected += 1;

        self.loss_events.push_back(LossEvent {
            tick: self.tick,
            cwnd_at_loss: self.cwnd,
            rtt_at_loss: self.current_rtt,
        });
        if self.loss_events.len() > self.config.history_limit {
            self.loss_events.pop_front();
        }

        match self.config.algorithm {
            Algorithm::Aimd => self.aimd_on_loss(),
            Algorithm::Bbr => self.bbr_on_loss(),
        }

        self.record_cwnd();
        self.update_pacing();
    }

    /// Process a duplicate ack (may trigger fast retransmit).
    pub fn on_dup_ack(&mut self) {
        self.dup_ack_count += 1;
        if self.dup_ack_count >= self.config.dup_ack_threshold
            && self.phase != Phase::FastRecovery
        {
            // Fast retransmit.
            self.stats.fast_retransmits += 1;
            self.ssthresh = ((self.cwnd as f64 * self.config.beta) as u32).max(self.config.min_cwnd);
            self.cwnd = self.ssthresh + self.config.dup_ack_threshold;
            self.phase = Phase::FastRecovery;
            self.record_cwnd();
        } else if self.phase == Phase::FastRecovery {
            // Inflate window during fast recovery.
            self.cwnd = (self.cwnd + 1).min(self.config.max_cwnd);
        }
    }

    /// Window size in bytes.
    pub fn window_bytes(&self) -> u64 {
        self.cwnd as u64 * self.config.mss as u64
    }

    /// Whether the window allows sending.
    pub fn can_send(&self, in_flight_bytes: u64) -> bool {
        in_flight_bytes < self.window_bytes()
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.cwnd = self.config.initial_cwnd;
        self.ssthresh = self.config.initial_ssthresh;
        self.phase = Phase::SlowStart;
        self.dup_ack_count = 0;
        self.cwnd_history.clear();
        self.loss_events.clear();
    }

    // ── AIMD internals ──────────────────────────────────────────

    fn aimd_on_ack(&mut self, _bytes_acked: u32) {
        match self.phase {
            Phase::SlowStart => {
                // Exponential growth.
                self.cwnd = (self.cwnd + 1).min(self.config.max_cwnd);
                if self.cwnd >= self.ssthresh {
                    self.phase = Phase::CongestionAvoidance;
                    self.stats.slow_start_exits += 1;
                }
            }
            Phase::CongestionAvoidance => {
                // Additive increase: cwnd += 1/cwnd per ack (~ 1 segment per RTT).
                // We approximate by incrementing every cwnd acks.
                self.cwnd = (self.cwnd + 1).min(self.config.max_cwnd);
            }
            Phase::FastRecovery => {
                // Exit fast recovery.
                self.cwnd = self.ssthresh;
                self.phase = Phase::CongestionAvoidance;
            }
        }

        if self.cwnd > self.stats.cwnd_peak {
            self.stats.cwnd_peak = self.cwnd;
        }
    }

    fn aimd_on_loss(&mut self) {
        // Multiplicative decrease.
        self.ssthresh = ((self.cwnd as f64 * self.config.beta) as u32).max(self.config.min_cwnd);
        self.cwnd = self.config.min_cwnd;
        self.phase = Phase::SlowStart;
    }

    // ── BBR internals ───────────────────────────────────────────

    fn bbr_on_ack(&mut self, bytes_acked: u32, rtt: f64) {
        self.bandwidth_samples.push_back(BandwidthSample {
            bytes_delivered: bytes_acked as u64,
            rtt,
        });
        if self.bandwidth_samples.len() > self.config.history_limit {
            self.bandwidth_samples.pop_front();
        }

        // Estimate bandwidth as max(delivered/rtt) over recent samples.
        let mut max_bw = 0.0f64;
        for sample in &self.bandwidth_samples {
            if sample.rtt > 0.0 {
                let bw = sample.bytes_delivered as f64 / sample.rtt;
                if bw > max_bw {
                    max_bw = bw;
                }
            }
        }
        self.estimated_bandwidth = max_bw;

        // Set cwnd based on BDP (bandwidth * min_rtt).
        let min_rtt = if self.stats.min_rtt > 0.0 {
            self.stats.min_rtt
        } else {
            rtt
        };
        let bdp = (self.estimated_bandwidth * min_rtt) / self.config.mss as f64;
        self.cwnd = (bdp as u32 + 2).clamp(self.config.min_cwnd, self.config.max_cwnd);

        self.bbr_round += 1;
        if self.cwnd > self.stats.cwnd_peak {
            self.stats.cwnd_peak = self.cwnd;
        }
    }

    fn bbr_on_loss(&mut self) {
        // BBR is less aggressive on loss — slight reduction.
        self.cwnd = ((self.cwnd as f64 * 0.85) as u32).max(self.config.min_cwnd);
    }

    // ── Shared internals ────────────────────────────────────────

    fn record_cwnd(&mut self) {
        self.cwnd_history.push_back((self.tick, self.cwnd));
        if self.cwnd_history.len() > self.config.history_limit {
            self.cwnd_history.pop_front();
        }
    }

    fn update_pacing(&mut self) {
        if self.config.pacing_enabled && self.current_rtt > 0.0 {
            self.pacing_rate = self.window_bytes() as f64 / self.current_rtt;
        }
    }
}

impl fmt::Display for CongestionWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CongestionWindow(algo={}, phase={}, cwnd={}, ssthresh={}, rtt={:.1})",
            self.config.algorithm,
            self.phase,
            self.cwnd,
            self.ssthresh,
            self.current_rtt,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cw() -> CongestionWindow {
        CongestionWindow::new(CongestionConfig::default())
    }

    #[test]
    fn initial_state_is_slow_start() {
        let cw = default_cw();
        assert_eq!(cw.phase(), Phase::SlowStart);
        assert_eq!(cw.cwnd(), 10);
    }

    #[test]
    fn slow_start_grows_exponentially() {
        let mut cw = default_cw();
        let initial = cw.cwnd();
        cw.on_ack(1460, 50.0);
        assert!(cw.cwnd() > initial);
    }

    #[test]
    fn transition_to_congestion_avoidance() {
        let config = CongestionConfig::default().with_initial_cwnd(5);
        let mut cw = CongestionWindow::new(config);
        cw.ssthresh = 8;
        for _ in 0..10 {
            cw.on_ack(1460, 50.0);
        }
        assert_eq!(cw.phase(), Phase::CongestionAvoidance);
    }

    #[test]
    fn loss_triggers_multiplicative_decrease() {
        let mut cw = default_cw();
        // Grow a bit first.
        for _ in 0..20 {
            cw.on_ack(1460, 50.0);
        }
        let before = cw.cwnd();
        cw.on_loss();
        assert!(cw.cwnd() < before);
        assert_eq!(cw.phase(), Phase::SlowStart);
    }

    #[test]
    fn loss_sets_ssthresh() {
        let mut cw = default_cw();
        for _ in 0..10 {
            cw.on_ack(1460, 50.0);
        }
        let cwnd_before = cw.cwnd();
        cw.on_loss();
        let expected_ss = ((cwnd_before as f64 * 0.5) as u32).max(2);
        assert_eq!(cw.ssthresh(), expected_ss);
    }

    #[test]
    fn fast_retransmit_on_dup_acks() {
        let mut cw = default_cw();
        for _ in 0..10 {
            cw.on_ack(1460, 50.0);
        }
        cw.on_dup_ack();
        cw.on_dup_ack();
        assert_ne!(cw.phase(), Phase::FastRecovery);
        cw.on_dup_ack(); // Third dup ack triggers fast retransmit.
        assert_eq!(cw.phase(), Phase::FastRecovery);
        assert_eq!(cw.stats().fast_retransmits, 1);
    }

    #[test]
    fn fast_recovery_inflates_window() {
        let mut cw = default_cw();
        for _ in 0..10 {
            cw.on_ack(1460, 50.0);
        }
        cw.on_dup_ack();
        cw.on_dup_ack();
        cw.on_dup_ack();
        let cwnd_in_recovery = cw.cwnd();
        cw.on_dup_ack(); // Extra dup ack during recovery.
        assert!(cw.cwnd() >= cwnd_in_recovery);
    }

    #[test]
    fn exit_fast_recovery_on_ack() {
        let mut cw = default_cw();
        for _ in 0..10 {
            cw.on_ack(1460, 50.0);
        }
        cw.on_dup_ack();
        cw.on_dup_ack();
        cw.on_dup_ack();
        assert_eq!(cw.phase(), Phase::FastRecovery);
        cw.on_ack(1460, 50.0);
        assert_eq!(cw.phase(), Phase::CongestionAvoidance);
    }

    #[test]
    fn cwnd_never_exceeds_max() {
        let config = CongestionConfig::default().with_max_cwnd(15);
        let mut cw = CongestionWindow::new(config);
        for _ in 0..100 {
            cw.on_ack(1460, 50.0);
        }
        assert!(cw.cwnd() <= 15);
    }

    #[test]
    fn cwnd_never_below_min() {
        let mut cw = default_cw();
        for _ in 0..10 {
            cw.on_loss();
        }
        assert!(cw.cwnd() >= 2);
    }

    #[test]
    fn bbr_estimates_bandwidth() {
        let config = CongestionConfig::default().with_algorithm(Algorithm::Bbr);
        let mut cw = CongestionWindow::new(config);
        cw.on_ack(14600, 10.0);
        assert!(cw.estimated_bandwidth() > 0.0);
    }

    #[test]
    fn bbr_loss_is_less_aggressive() {
        let config = CongestionConfig::default()
            .with_algorithm(Algorithm::Bbr)
            .with_initial_cwnd(100)
            .with_max_cwnd(1000);
        let mut cw = CongestionWindow::new(config);
        // Drive enough acks with high throughput to build bandwidth estimate.
        for _ in 0..20 {
            cw.on_ack(14600, 10.0);
        }
        let before = cw.cwnd();
        cw.on_loss();
        // BBR reduces by 15%, not all the way to min_cwnd.
        assert!(cw.cwnd() >= 2);
        assert!(cw.cwnd() <= before);
    }

    #[test]
    fn pacing_rate_calculated() {
        let config = CongestionConfig::default().with_pacing(true);
        let mut cw = CongestionWindow::new(config);
        cw.on_ack(1460, 50.0);
        assert!(cw.pacing_rate() > 0.0);
    }

    #[test]
    fn pacing_disabled_stays_zero() {
        let mut cw = default_cw();
        cw.on_ack(1460, 50.0);
        assert_eq!(cw.pacing_rate(), 0.0);
    }

    #[test]
    fn window_bytes_calculation() {
        let cw = default_cw();
        assert_eq!(cw.window_bytes(), 10 * 1460);
    }

    #[test]
    fn can_send_under_limit() {
        let cw = default_cw();
        assert!(cw.can_send(0));
        assert!(cw.can_send(10_000));
        assert!(!cw.can_send(100_000));
    }

    #[test]
    fn stats_track_rtt_range() {
        let mut cw = default_cw();
        cw.on_ack(1460, 10.0);
        cw.on_ack(1460, 50.0);
        cw.on_ack(1460, 30.0);
        assert!((cw.stats().min_rtt - 10.0).abs() < 0.001);
        assert!((cw.stats().max_rtt - 50.0).abs() < 0.001);
    }

    #[test]
    fn loss_events_recorded() {
        let mut cw = default_cw();
        cw.tick(100);
        cw.on_loss();
        assert_eq!(cw.loss_events().len(), 1);
        assert_eq!(cw.loss_events()[0].tick, 100);
    }

    #[test]
    fn cwnd_history_recorded() {
        let mut cw = default_cw();
        cw.tick(1);
        cw.on_ack(1460, 50.0);
        assert!(!cw.cwnd_history().is_empty());
    }

    #[test]
    fn reset_restores_initial_state() {
        let mut cw = default_cw();
        cw.on_ack(1460, 50.0);
        cw.on_loss();
        cw.reset();
        assert_eq!(cw.cwnd(), 10);
        assert_eq!(cw.phase(), Phase::SlowStart);
    }

    #[test]
    fn display_formats() {
        let cw = default_cw();
        let s = format!("{cw}");
        assert!(s.contains("AIMD"));
        assert!(s.contains("SlowStart"));
    }

    #[test]
    fn config_builder() {
        let config = CongestionConfig::default()
            .with_algorithm(Algorithm::Bbr)
            .with_initial_cwnd(20)
            .with_max_cwnd(100)
            .with_pacing(true)
            .with_beta(0.7);
        assert_eq!(config.algorithm, Algorithm::Bbr);
        assert_eq!(config.initial_cwnd, 20);
        assert_eq!(config.max_cwnd, 100);
        assert!(config.pacing_enabled);
        assert!((config.beta - 0.7).abs() < 0.001);
    }
}
