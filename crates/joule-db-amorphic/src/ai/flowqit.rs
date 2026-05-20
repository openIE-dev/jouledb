//! flowQIT — Quantum Information Theory LUT (Axiom 6 Formalized)
//!
//! The missing domain LUT. Formalizes axiom 6:
//! "The rates of change for contrast recognition translate to operational value."
//!
//! Using quantum information theory as the language:
//! - **Von Neumann entropy** measures how much a system doesn't know
//! - **Decoherence rate** measures how fast recognized contrasts fade
//! - **Landauer floor** gives the minimum energy per bit of contrast
//! - **Information gain rate** tracks bits/joule efficiency over time
//!
//! Maps to flowG's 22 primitives with QIT-specific semantics:
//! - Bind → Prepare state (initialize density matrix)
//! - Apply → Measure (wavefunction collapse, Landauer cost)
//! - Branch → Decoherence check (has contrast survived?)
//! - Iterate → Entropy reduction loop (repeated measurement)
//! - Observe → Von Neumann entropy computation
//! - FeedbackDelay → Decoherence time tracking
//! - Match → Regime classification (coherent/decoherent/thermal)

use std::f64::consts::{E, LN_2};

/// Boltzmann constant in J/K
pub const K_BOLTZMANN: f64 = 1.380649e-23;

/// Room temperature in Kelvin (standard conditions)
pub const ROOM_TEMP_K: f64 = 293.15;

/// Landauer limit at room temperature: kT ln(2) ≈ 2.805 × 10⁻²¹ J/bit
pub const LANDAUER_FLOOR: f64 = K_BOLTZMANN * ROOM_TEMP_K * LN_2;

/// QIT compute regime: same concept as flowR's ComputeRegime but
/// defined through entropy thresholds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QitRegime {
    /// Pure state: entropy = 0. Fully known, no measurement needed.
    /// Maps to ComputeRegime::Trivial.
    Coherent,
    /// Mixed state: entropy reducible by measurement at Landauer cost.
    /// Maps to ComputeRegime::Tractable.
    Decoherent,
    /// Maximally mixed: no measurement helps. Entropy = log₂(d).
    /// Maps to ComputeRegime::Intractable.
    Thermal,
}

/// A density matrix state for tracking what the system knows/doesn't know.
/// Simplified to diagonal form (classical mixture) for computational tractability.
#[derive(Clone, Debug)]
pub struct DensityState {
    /// Diagonal of the density matrix: probability of each basis state.
    /// Must sum to 1.0 and all entries ≥ 0.
    pub probabilities: Vec<f64>,
    /// Dimension of the Hilbert space (number of basis states).
    pub dimension: usize,
    /// Timestamp of last observation (for decoherence tracking).
    pub last_observed_ms: u64,
    /// Label for what this state represents.
    pub label: String,
}

impl DensityState {
    /// Create a pure state (fully known): all probability on one basis state.
    pub fn pure(dimension: usize, known_state: usize, label: &str) -> Self {
        let mut probs = vec![0.0; dimension];
        if known_state < dimension {
            probs[known_state] = 1.0;
        }
        Self {
            probabilities: probs,
            dimension,
            last_observed_ms: 0,
            label: label.to_string(),
        }
    }

    /// Create a maximally mixed state (nothing known): uniform distribution.
    pub fn maximally_mixed(dimension: usize, label: &str) -> Self {
        let p = 1.0 / dimension as f64;
        Self {
            probabilities: vec![p; dimension],
            dimension,
            last_observed_ms: 0,
            label: label.to_string(),
        }
    }

    /// Create from an explicit probability distribution.
    pub fn from_probabilities(probs: Vec<f64>, label: &str) -> Self {
        let dimension = probs.len();
        Self {
            probabilities: probs,
            dimension,
            last_observed_ms: 0,
            label: label.to_string(),
        }
    }

    /// Von Neumann entropy: S = -Σ p_i log₂(p_i)
    /// Range: [0, log₂(d)] where d = dimension.
    /// 0 = pure state (fully known), log₂(d) = maximally mixed (nothing known).
    pub fn entropy(&self) -> f64 {
        let mut s = 0.0;
        for &p in &self.probabilities {
            if p > 1e-15 {
                s -= p * p.log2();
            }
        }
        s
    }

    /// Maximum possible entropy for this dimension.
    pub fn max_entropy(&self) -> f64 {
        (self.dimension as f64).log2()
    }

    /// Purity: Tr(ρ²) = Σ p_i². Range: [1/d, 1].
    /// 1 = pure state, 1/d = maximally mixed.
    pub fn purity(&self) -> f64 {
        self.probabilities.iter().map(|p| p * p).sum()
    }

    /// Classify the regime based on entropy.
    pub fn regime(&self) -> QitRegime {
        let entropy = self.entropy();
        let max = self.max_entropy();
        if max == 0.0 {
            return QitRegime::Coherent;
        }
        let ratio = entropy / max;
        if ratio < 0.01 {
            QitRegime::Coherent
        } else if ratio < 0.90 {
            QitRegime::Decoherent
        } else {
            QitRegime::Thermal
        }
    }

    /// Simulate measurement: collapse probability toward the most likely state.
    /// `strength` ∈ (0, 1] controls how much information the measurement extracts.
    /// Energy cost is at least Landauer floor per bit of information gained.
    pub fn measure(&mut self, strength: f64) -> MeasurementResult {
        let entropy_before = self.entropy();

        // Find the most probable state
        let (max_idx, _max_p) = self
            .probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        // Sharpen distribution toward the most probable state
        let strength = strength.clamp(0.01, 1.0);
        for (i, p) in self.probabilities.iter_mut().enumerate() {
            if i == max_idx {
                *p = *p + (1.0 - *p) * strength;
            } else {
                *p *= 1.0 - strength;
            }
        }

        // Renormalize
        let sum: f64 = self.probabilities.iter().sum();
        if sum > 0.0 {
            for p in &mut self.probabilities {
                *p /= sum;
            }
        }

        let entropy_after = self.entropy();
        let info_gained = (entropy_before - entropy_after).max(0.0);
        let energy_cost = info_gained * LANDAUER_FLOOR;

        MeasurementResult {
            state_collapsed_to: max_idx,
            info_gained_bits: info_gained,
            energy_cost_joules: energy_cost,
            entropy_before,
            entropy_after,
            regime_after: self.regime(),
        }
    }

    /// Simulate decoherence: state drifts toward maximally mixed over time.
    /// `elapsed_ms` is time since last observation.
    /// `t_decoherence_ms` is the characteristic decoherence time.
    pub fn decohere(&mut self, elapsed_ms: u64, t_decoherence_ms: f64) {
        if t_decoherence_ms <= 0.0 || elapsed_ms == 0 {
            return;
        }
        let decay = (-((elapsed_ms as f64) / t_decoherence_ms)).exp();
        let uniform = 1.0 / self.dimension as f64;
        for p in &mut self.probabilities {
            // Interpolate toward uniform: p(t) = p(0) * exp(-t/τ) + uniform * (1 - exp(-t/τ))
            *p = *p * decay + uniform * (1.0 - decay);
        }
    }
}

/// Result of a measurement operation.
#[derive(Clone, Debug)]
pub struct MeasurementResult {
    /// Which basis state the measurement collapsed toward.
    pub state_collapsed_to: usize,
    /// Bits of information gained by this measurement.
    pub info_gained_bits: f64,
    /// Energy cost at Landauer floor (theoretical minimum).
    pub energy_cost_joules: f64,
    /// Entropy before measurement.
    pub entropy_before: f64,
    /// Entropy after measurement.
    pub entropy_after: f64,
    /// Regime after measurement.
    pub regime_after: QitRegime,
}

/// Tracks contrast recognition rate over time — the heart of axiom 6.
///
/// "The rates of change for contrast recognition translate to operational value."
/// This means: how fast are we reducing entropy? That rate IS the value.
#[derive(Clone, Debug)]
pub struct ContrastDynamics {
    /// History of entropy values at measurement timestamps.
    pub entropy_history: Vec<(u64, f64)>, // (timestamp_ms, entropy)
    /// History of information gain per measurement.
    pub gain_history: Vec<(u64, f64)>, // (timestamp_ms, bits_gained)
    /// Cumulative energy spent on measurements.
    pub total_energy_joules: f64,
    /// Cumulative bits of information gained.
    pub total_bits_gained: f64,
}

impl ContrastDynamics {
    pub fn new() -> Self {
        Self {
            entropy_history: Vec::new(),
            gain_history: Vec::new(),
            total_energy_joules: 0.0,
            total_bits_gained: 0.0,
        }
    }

    /// Record a measurement result.
    pub fn record(&mut self, timestamp_ms: u64, result: &MeasurementResult) {
        self.entropy_history
            .push((timestamp_ms, result.entropy_after));
        self.gain_history
            .push((timestamp_ms, result.info_gained_bits));
        self.total_energy_joules += result.energy_cost_joules;
        self.total_bits_gained += result.info_gained_bits;
    }

    /// Contrast recognition rate: d(entropy)/dt over recent window.
    /// Negative means we're learning (entropy decreasing). More negative = faster learning.
    /// Units: bits/ms
    pub fn contrast_recognition_rate(&self) -> f64 {
        if self.entropy_history.len() < 2 {
            return 0.0;
        }
        let n = self.entropy_history.len();
        let (t1, e1) = self.entropy_history[n - 2];
        let (t2, e2) = self.entropy_history[n - 1];
        let dt = (t2 as f64 - t1 as f64).max(1.0);
        (e2 - e1) / dt
    }

    /// Decoherence rate: how fast does recognized contrast fade?
    /// Estimated from entropy increases between measurements.
    /// Positive means information is being lost. Units: bits/ms
    pub fn decoherence_rate(&self) -> f64 {
        if self.entropy_history.len() < 3 {
            return 0.0;
        }
        // Look at entropy increases (decoherence events)
        let mut increases = Vec::new();
        for w in self.entropy_history.windows(2) {
            let (t1, e1) = w[0];
            let (t2, e2) = w[1];
            if e2 > e1 {
                let dt = (t2 as f64 - t1 as f64).max(1.0);
                increases.push((e2 - e1) / dt);
            }
        }
        if increases.is_empty() {
            return 0.0;
        }
        increases.iter().sum::<f64>() / increases.len() as f64
    }

    /// Information gain rate: bits gained per joule of energy spent.
    /// This is the operational efficiency — axiom 6's "operational value."
    pub fn information_gain_rate(&self) -> f64 {
        if self.total_energy_joules <= 0.0 {
            return 0.0;
        }
        self.total_bits_gained / self.total_energy_joules
    }

    /// Landauer efficiency: how close are we to the theoretical minimum?
    /// 1.0 = perfect (operating at Landauer floor). < 1.0 = wasting energy.
    /// Real systems operate at ~10⁻⁶ to 10⁻³ efficiency.
    pub fn landauer_efficiency(&self) -> f64 {
        if self.total_bits_gained <= 0.0 || self.total_energy_joules <= 0.0 {
            return 0.0;
        }
        let theoretical_minimum = self.total_bits_gained * LANDAUER_FLOOR;
        (theoretical_minimum / self.total_energy_joules).min(1.0)
    }

    /// Net information flow: recognition rate minus decoherence rate.
    /// Positive = net learning. Negative = net forgetting.
    pub fn net_information_flow(&self) -> f64 {
        // Recognition rate is negative (entropy decreasing), decoherence is positive
        // Net flow = -recognition_rate - decoherence_rate
        // Positive net flow means we're learning faster than forgetting
        let recognition = -self.contrast_recognition_rate();
        let decoherence = self.decoherence_rate();
        recognition - decoherence
    }

    /// Predict time until entropy reaches zero (full knowledge) at current rate.
    /// Returns None if we're not converging.
    pub fn estimated_time_to_resolution_ms(&self) -> Option<f64> {
        let rate = self.contrast_recognition_rate();
        if rate >= 0.0 {
            // Not converging
            return None;
        }
        if let Some(&(_, current_entropy)) = self.entropy_history.last() {
            if current_entropy <= 0.0 {
                return Some(0.0);
            }
            // time = current_entropy / |rate|
            Some(current_entropy / (-rate))
        } else {
            None
        }
    }
}

impl Default for ContrastDynamics {
    fn default() -> Self {
        Self::new()
    }
}

/// QIT prior table: maps flowG node kinds to QIT regime priors.
/// These priors guide how much compute to allocate before execution.
pub struct QitPriors {
    /// Prior regime for each node label pattern.
    priors: Vec<(String, QitRegime, f64)>, // (pattern, regime, confidence)
}

impl QitPriors {
    pub fn new() -> Self {
        Self {
            priors: vec![
                // Trivial: deterministic operations
                ("cache_lookup".into(), QitRegime::Coherent, 0.99),
                ("exact_match".into(), QitRegime::Coherent, 0.99),
                ("hash_check".into(), QitRegime::Coherent, 0.99),
                ("pattern_resolve".into(), QitRegime::Coherent, 0.95),
                // Tractable: reducible with effort
                ("similarity_search".into(), QitRegime::Decoherent, 0.8),
                ("classify".into(), QitRegime::Decoherent, 0.7),
                ("recommend".into(), QitRegime::Decoherent, 0.6),
                ("summarize".into(), QitRegime::Decoherent, 0.5),
                // Intractable: maximally uncertain
                ("generate_novel".into(), QitRegime::Thermal, 0.3),
                ("predict_future".into(), QitRegime::Thermal, 0.2),
                ("creative_synthesis".into(), QitRegime::Thermal, 0.3),
            ],
        }
    }

    /// Look up the prior regime for a node label.
    pub fn prior_for(&self, label: &str) -> (QitRegime, f64) {
        let lower = label.to_lowercase();
        for (pattern, regime, confidence) in &self.priors {
            if lower.contains(pattern) {
                return (*regime, *confidence);
            }
        }
        // Default: decoherent with moderate confidence
        (QitRegime::Decoherent, 0.5)
    }

    /// Minimum energy for an operation: Landauer floor × estimated bits.
    pub fn landauer_floor_for(&self, label: &str) -> f64 {
        let (regime, _) = self.prior_for(label);
        match regime {
            QitRegime::Coherent => 0.0,                    // Pure state, nothing to erase
            QitRegime::Decoherent => LANDAUER_FLOOR * 10.0, // ~10 bits typical
            QitRegime::Thermal => LANDAUER_FLOOR * 1000.0,  // ~1000 bits for open-ended
        }
    }
}

impl Default for QitPriors {
    fn default() -> Self {
        Self::new()
    }
}

/// The flowQIT engine: combines state tracking, measurement, and dynamics.
pub struct FlowQitEngine {
    /// Active density states being tracked.
    pub states: Vec<DensityState>,
    /// Contrast dynamics tracker.
    pub dynamics: ContrastDynamics,
    /// Prior table for regime classification.
    pub priors: QitPriors,
    /// Decoherence timescale for this system (ms).
    pub t_decoherence_ms: f64,
}

impl FlowQitEngine {
    pub fn new(t_decoherence_ms: f64) -> Self {
        Self {
            states: Vec::new(),
            dynamics: ContrastDynamics::new(),
            priors: QitPriors::new(),
            t_decoherence_ms,
        }
    }

    /// Initialize a new state to track (e.g., "what genre is this content?").
    pub fn prepare_state(&mut self, dimension: usize, label: &str) -> usize {
        let state = DensityState::maximally_mixed(dimension, label);
        let idx = self.states.len();
        self.states.push(state);
        idx
    }

    /// Perform a measurement on a tracked state.
    pub fn measure(&mut self, state_idx: usize, strength: f64, timestamp_ms: u64) -> Option<MeasurementResult> {
        if state_idx >= self.states.len() {
            return None;
        }
        let result = self.states[state_idx].measure(strength);
        self.states[state_idx].last_observed_ms = timestamp_ms;
        self.dynamics.record(timestamp_ms, &result);
        Some(result)
    }

    /// Apply decoherence to all tracked states.
    pub fn decohere_all(&mut self, current_ms: u64) {
        for state in &mut self.states {
            let elapsed = current_ms.saturating_sub(state.last_observed_ms);
            state.decohere(elapsed, self.t_decoherence_ms);
        }
    }

    /// Get the overall system entropy (sum of all tracked states).
    pub fn total_entropy(&self) -> f64 {
        self.states.iter().map(|s| s.entropy()).sum()
    }

    /// Summary of the QIT engine's current state.
    pub fn summary(&self) -> QitSummary {
        QitSummary {
            num_states: self.states.len(),
            total_entropy: self.total_entropy(),
            recognition_rate: self.dynamics.contrast_recognition_rate(),
            decoherence_rate: self.dynamics.decoherence_rate(),
            net_flow: self.dynamics.net_information_flow(),
            efficiency: self.dynamics.information_gain_rate(),
            landauer_efficiency: self.dynamics.landauer_efficiency(),
            time_to_resolution_ms: self.dynamics.estimated_time_to_resolution_ms(),
        }
    }
}

/// Summary statistics for the QIT engine.
#[derive(Clone, Debug)]
pub struct QitSummary {
    pub num_states: usize,
    pub total_entropy: f64,
    pub recognition_rate: f64,
    pub decoherence_rate: f64,
    pub net_flow: f64,
    pub efficiency: f64,
    pub landauer_efficiency: f64,
    pub time_to_resolution_ms: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_landauer_floor() {
        // kT ln(2) at room temp ≈ 2.8 × 10⁻²¹ J
        assert!(LANDAUER_FLOOR > 2.0e-21);
        assert!(LANDAUER_FLOOR < 3.0e-21);
    }

    #[test]
    fn test_pure_state_entropy() {
        let state = DensityState::pure(4, 0, "test");
        assert!((state.entropy() - 0.0).abs() < 1e-10);
        assert_eq!(state.regime(), QitRegime::Coherent);
        assert!((state.purity() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_maximally_mixed_entropy() {
        let state = DensityState::maximally_mixed(8, "test");
        let expected = 3.0; // log₂(8)
        assert!((state.entropy() - expected).abs() < 1e-10);
        assert_eq!(state.regime(), QitRegime::Thermal);
    }

    #[test]
    fn test_measurement_reduces_entropy() {
        let mut state = DensityState::maximally_mixed(4, "test");
        let before = state.entropy();
        let result = state.measure(0.5);
        assert!(result.entropy_after < before);
        assert!(result.info_gained_bits > 0.0);
        assert!(result.energy_cost_joules > 0.0);
    }

    #[test]
    fn test_decoherence_increases_entropy() {
        let mut state = DensityState::pure(4, 0, "test");
        assert!((state.entropy() - 0.0).abs() < 1e-10);
        state.decohere(10000, 100.0); // Long time, short decoherence
        assert!(state.entropy() > 0.1);
    }

    #[test]
    fn test_regime_classification() {
        let pure = DensityState::pure(4, 0, "test");
        assert_eq!(pure.regime(), QitRegime::Coherent);

        let mixed = DensityState::maximally_mixed(4, "test");
        assert_eq!(mixed.regime(), QitRegime::Thermal);

        let partial = DensityState::from_probabilities(vec![0.7, 0.1, 0.1, 0.1], "test");
        assert_eq!(partial.regime(), QitRegime::Decoherent);
    }

    #[test]
    fn test_contrast_dynamics() {
        let mut dynamics = ContrastDynamics::new();

        // Simulate measurements that reduce entropy over time
        let results = vec![
            (0, 2.0, 1.5, 0.5),   // t=0, before=2.0, after=1.5, gained=0.5
            (100, 1.5, 1.0, 0.5), // t=100ms
            (200, 1.0, 0.5, 0.5), // t=200ms
            (300, 0.5, 0.1, 0.4), // t=300ms
        ];
        for (t, _before, after, gained) in &results {
            dynamics.entropy_history.push((*t, *after));
            dynamics.gain_history.push((*t, *gained));
            dynamics.total_bits_gained += gained;
            dynamics.total_energy_joules += gained * LANDAUER_FLOOR * 1e6; // realistic overhead
        }

        // Recognition rate should be negative (entropy decreasing)
        let rate = dynamics.contrast_recognition_rate();
        assert!(rate < 0.0, "recognition rate should be negative (learning), got {rate}");

        // Information gain rate should be positive
        let igr = dynamics.information_gain_rate();
        assert!(igr > 0.0, "info gain rate should be positive, got {igr}");

        // Should have an estimated time to resolution
        let ttr = dynamics.estimated_time_to_resolution_ms();
        assert!(ttr.is_some());
    }

    #[test]
    fn test_engine_workflow() {
        let mut engine = FlowQitEngine::new(10000.0);

        // Prepare a 4-state system (e.g., classifying content into 4 genres)
        let idx = engine.prepare_state(4, "genre_classification");
        assert_eq!(engine.states[idx].regime(), QitRegime::Thermal);

        // Perform measurements
        engine.measure(idx, 0.3, 0);
        engine.measure(idx, 0.3, 100);
        engine.measure(idx, 0.3, 200);

        // Entropy should have decreased
        assert!(engine.states[idx].entropy() < 2.0);

        // Summary should show progress
        let summary = engine.summary();
        assert_eq!(summary.num_states, 1);
        assert!(summary.total_entropy < 2.0);
    }

    #[test]
    fn test_priors() {
        let priors = QitPriors::new();
        let (regime, conf) = priors.prior_for("cache_lookup");
        assert_eq!(regime, QitRegime::Coherent);
        assert!(conf > 0.9);

        let (regime, _) = priors.prior_for("creative_synthesis");
        assert_eq!(regime, QitRegime::Thermal);
    }

    #[test]
    fn test_net_information_flow() {
        let mut dynamics = ContrastDynamics::new();
        // Monotonically decreasing entropy = positive net flow (learning)
        dynamics.entropy_history.push((0, 3.0));
        dynamics.entropy_history.push((100, 2.0));
        dynamics.entropy_history.push((200, 1.0));
        let net = dynamics.net_information_flow();
        assert!(net > 0.0, "should be net learning, got {net}");
    }
}
