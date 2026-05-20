//! Energy-Proportional Compute — metabolic rate tracks surprise magnitude.
//!
//! Axiom 6: The rates of change for contrast recognition translate to operational value.
//!
//! A system that burns the same energy for trivial and novel queries is wasteful.
//! This module makes compute proportional to contrast magnitude:
//! - Trivial query (high cache similarity) → near-zero energy
//! - Moderately novel query → moderate energy
//! - Highly novel query → full compute
//!
//! Biological analogy: resting metabolic rate vs fight-or-flight.
//! The brain uses ~20W, but most neurons are silent most of the time.
//! Energy surges happen only when surprise exceeds threshold.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use super::contrast::ContrastEngine;
use super::tier::InferenceTier;

/// Metabolic state of the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetabolicState {
    /// Resting: low contrast, minimal compute. Most queries served from cache/LUT.
    Resting,
    /// Alert: moderate contrast detected. Engage Tier 1-2 compute.
    Alert,
    /// Active: high contrast. Full compute path engaged.
    Active,
    /// Surge: extreme novelty or threat. Escalate to highest available tier.
    Surge,
}

/// Compute budget allocated based on contrast magnitude.
#[derive(Debug, Clone)]
pub struct ComputeBudget {
    /// Current metabolic state
    pub state: MetabolicState,
    /// Maximum energy to spend on this operation (joules)
    pub energy_limit: f64,
    /// Maximum latency allowed (microseconds)
    pub latency_limit_us: u64,
    /// Which inference tiers are allowed at this metabolic level
    pub allowed_tiers: Vec<InferenceTier>,
    /// Number of similarity comparisons to perform
    pub max_comparisons: usize,
}

/// The metabolic controller: maps contrast magnitude to compute allocation.
pub struct MetabolicController {
    /// Thresholds for state transitions
    pub resting_threshold: f64,    // Below this: resting
    pub alert_threshold: f64,      // Above this: alert
    pub active_threshold: f64,     // Above this: active
    pub surge_threshold: f64,      // Above this: surge

    /// Current state
    state: MetabolicState,
    /// Exponentially weighted moving average of contrast magnitude
    ewma_contrast: f64,
    /// Decay factor for EWMA (0.0 = no memory, 1.0 = no decay)
    ewma_alpha: f64,
    /// Total energy consumed
    total_energy: f64,
    /// Total operations
    total_ops: u64,
}

impl MetabolicController {
    pub fn new() -> Self {
        Self {
            resting_threshold: 0.1,
            alert_threshold: 0.3,
            active_threshold: 0.6,
            surge_threshold: 0.9,
            state: MetabolicState::Resting,
            ewma_contrast: 0.0,
            ewma_alpha: 0.1,
            total_energy: 0.0,
            total_ops: 0,
        }
    }

    /// Compute the budget for an operation given its contrast magnitude.
    pub fn allocate(&mut self, contrast_magnitude: f64) -> ComputeBudget {
        // Update EWMA
        self.ewma_contrast =
            self.ewma_alpha * contrast_magnitude + (1.0 - self.ewma_alpha) * self.ewma_contrast;

        // Determine state from magnitude
        self.state = if contrast_magnitude >= self.surge_threshold {
            MetabolicState::Surge
        } else if contrast_magnitude >= self.active_threshold {
            MetabolicState::Active
        } else if contrast_magnitude >= self.alert_threshold {
            MetabolicState::Alert
        } else {
            MetabolicState::Resting
        };

        self.total_ops += 1;

        // Allocate compute proportional to state
        match self.state {
            MetabolicState::Resting => ComputeBudget {
                state: MetabolicState::Resting,
                energy_limit: 0.000_001,     // 1 µJ
                latency_limit_us: 10,         // 10 µs
                allowed_tiers: vec![InferenceTier::Holographic],
                max_comparisons: 10,
            },
            MetabolicState::Alert => ComputeBudget {
                state: MetabolicState::Alert,
                energy_limit: 0.001,          // 1 mJ
                latency_limit_us: 1_000,      // 1 ms
                allowed_tiers: vec![InferenceTier::Holographic, InferenceTier::Embedded],
                max_comparisons: 100,
            },
            MetabolicState::Active => ComputeBudget {
                state: MetabolicState::Active,
                energy_limit: 0.1,            // 100 mJ
                latency_limit_us: 100_000,    // 100 ms
                allowed_tiers: vec![
                    InferenceTier::Holographic,
                    InferenceTier::Embedded,
                    InferenceTier::Local,
                ],
                max_comparisons: 1000,
            },
            MetabolicState::Surge => ComputeBudget {
                state: MetabolicState::Surge,
                energy_limit: 5.0,            // 5 J
                latency_limit_us: 30_000_000, // 30 s
                allowed_tiers: vec![
                    InferenceTier::Holographic,
                    InferenceTier::Embedded,
                    InferenceTier::Local,
                    InferenceTier::Frontier,
                ],
                max_comparisons: usize::MAX,
            },
        }
    }

    /// Record actual energy consumed by an operation.
    pub fn record_energy(&mut self, joules: f64) {
        self.total_energy += joules;
    }

    /// Current metabolic state.
    pub fn state(&self) -> MetabolicState {
        self.state
    }

    /// Average contrast level (EWMA).
    pub fn average_contrast(&self) -> f64 {
        self.ewma_contrast
    }

    /// Total energy consumed.
    pub fn total_energy(&self) -> f64 {
        self.total_energy
    }

    /// Average energy per operation.
    pub fn energy_per_op(&self) -> f64 {
        if self.total_ops == 0 {
            return 0.0;
        }
        self.total_energy / self.total_ops as f64
    }

    /// Metabolic efficiency: what fraction of operations were handled at resting state?
    /// Higher = more efficient (most work is trivial, correctly identified as trivial).
    pub fn efficiency(&self) -> f64 {
        // Approximated from EWMA: low average contrast = high efficiency
        1.0 - self.ewma_contrast
    }
}

impl Default for MetabolicController {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resting_state() {
        let mut mc = MetabolicController::new();
        let budget = mc.allocate(0.05); // Low contrast

        assert_eq!(budget.state, MetabolicState::Resting);
        assert_eq!(budget.allowed_tiers, vec![InferenceTier::Holographic]);
        assert!(budget.energy_limit < 0.00001);
        assert!(budget.max_comparisons <= 10);
    }

    #[test]
    fn test_surge_state() {
        let mut mc = MetabolicController::new();
        let budget = mc.allocate(0.95); // High contrast

        assert_eq!(budget.state, MetabolicState::Surge);
        assert_eq!(budget.allowed_tiers.len(), 4); // All tiers
        assert!(budget.energy_limit > 1.0);
    }

    #[test]
    fn test_proportional_energy() {
        let mut mc = MetabolicController::new();

        let resting = mc.allocate(0.05);
        let active = mc.allocate(0.7);
        let surge = mc.allocate(0.95);

        assert!(resting.energy_limit < active.energy_limit);
        assert!(active.energy_limit < surge.energy_limit);

        // Energy spans ~6 orders of magnitude (1µJ to 5J)
        assert!(surge.energy_limit / resting.energy_limit > 1_000_000.0);
    }

    #[test]
    fn test_ewma_tracking() {
        let mut mc = MetabolicController::new();

        // Series of low-contrast operations
        for _ in 0..10 {
            mc.allocate(0.05);
        }
        assert!(mc.average_contrast() < 0.1);

        // Sudden high contrast
        mc.allocate(0.9);
        // EWMA should jump but not to 0.9 (smoothed)
        assert!(mc.average_contrast() > 0.05);
        assert!(mc.average_contrast() < 0.9);
    }

    #[test]
    fn test_efficiency() {
        let mut mc = MetabolicController::new();

        // All low contrast = high efficiency
        for _ in 0..100 {
            mc.allocate(0.02);
        }
        assert!(mc.efficiency() > 0.9);
    }

    #[test]
    fn test_state_transitions() {
        let mut mc = MetabolicController::new();

        assert_eq!(mc.allocate(0.05).state, MetabolicState::Resting);
        assert_eq!(mc.allocate(0.35).state, MetabolicState::Alert);
        assert_eq!(mc.allocate(0.65).state, MetabolicState::Active);
        assert_eq!(mc.allocate(0.95).state, MetabolicState::Surge);

        // Back to resting
        assert_eq!(mc.allocate(0.01).state, MetabolicState::Resting);
    }
}
