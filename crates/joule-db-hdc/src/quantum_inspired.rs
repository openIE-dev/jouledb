//! Quantum-Inspired Optimization
//!
//! Implements simulated quantum annealing with tunneling for query optimization.
//! This provides faster escape from local minima compared to classical simulated annealing.
//!
//! ## Key Concepts
//!
//! - **Quantum Tunneling**: Probabilistic jumps through energy barriers
//! - **Transverse Field**: Controls tunneling probability (decreases over time)
//! - **Path Integral Monte Carlo**: Samples multiple "replicas" of the system
//!
//! ## Example
//!
//! ```rust,ignore
//! use joule_db_hdc::quantum_inspired::{QuantumAnnealer, AnnealingConfig};
//!
//! let config = AnnealingConfig::default();
//! let mut annealer = QuantumAnnealer::new(config);
//!
//! // Define cost function for query plan
//! let cost_fn = |state: &[bool]| -> f64 {
//!     // Lower is better
//!     state.iter().filter(|&&b| b).count() as f64
//! };
//!
//! let solution = annealer.optimize(100, cost_fn);
//! ```

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

/// Configuration for quantum-inspired annealing
#[derive(Debug, Clone)]
pub struct AnnealingConfig {
    /// Initial temperature (thermal fluctuations)
    pub initial_temp: f64,
    /// Final temperature
    pub final_temp: f64,
    /// Initial transverse field strength (quantum fluctuations)
    pub initial_gamma: f64,
    /// Final transverse field strength
    pub final_gamma: f64,
    /// Number of Trotter slices (path integral replicas)
    pub trotter_slices: usize,
    /// Number of Monte Carlo sweeps per temperature step
    pub sweeps_per_step: usize,
    /// Total annealing steps
    pub total_steps: usize,
    /// Random seed
    pub seed: u64,
}

impl Default for AnnealingConfig {
    fn default() -> Self {
        Self {
            initial_temp: 10.0,
            final_temp: 0.01,
            initial_gamma: 5.0,
            final_gamma: 0.01,
            trotter_slices: 16,
            sweeps_per_step: 10,
            total_steps: 1000,
            seed: 42,
        }
    }
}

/// Quantum-inspired annealer using path integral Monte Carlo
pub struct QuantumAnnealer {
    config: AnnealingConfig,
    rng: StdRng,
    /// Best solution found
    best_state: Vec<bool>,
    /// Best cost found
    best_cost: f64,
    /// Current step
    current_step: usize,
}

impl QuantumAnnealer {
    /// Create a new quantum annealer
    pub fn new(config: AnnealingConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        Self {
            config,
            rng,
            best_state: Vec::new(),
            best_cost: f64::INFINITY,
            current_step: 0,
        }
    }

    /// Get current temperature based on annealing schedule
    fn temperature(&self) -> f64 {
        let progress = self.current_step as f64 / self.config.total_steps as f64;
        self.config.initial_temp
            * (self.config.final_temp / self.config.initial_temp).powf(progress)
    }

    /// Get current transverse field strength
    fn gamma(&self) -> f64 {
        let progress = self.current_step as f64 / self.config.total_steps as f64;
        self.config.initial_gamma
            * (self.config.final_gamma / self.config.initial_gamma).powf(progress)
    }

    /// Coupling strength between Trotter slices
    fn trotter_coupling(&self) -> f64 {
        let temp = self.temperature();
        let gamma = self.gamma();
        let m = self.config.trotter_slices as f64;

        // J_perp = -T/2 * ln(tanh(gamma / (m * T)))
        let x = gamma / (m * temp);
        if x > 20.0 {
            // For large x, tanh(x) ≈ 1, ln(1) = 0
            0.0
        } else {
            -0.5 * temp * x.tanh().ln()
        }
    }

    /// Optimize a binary state using quantum-inspired annealing
    ///
    /// # Arguments
    /// * `dimension` - Number of binary variables
    /// * `cost_fn` - Cost function to minimize (lower is better)
    ///
    /// # Returns
    /// The best binary state found
    pub fn optimize<F>(&mut self, dimension: usize, cost_fn: F) -> Vec<bool>
    where
        F: Fn(&[bool]) -> f64,
    {
        let m = self.config.trotter_slices;

        // Initialize Trotter replicas (imaginary time slices)
        let mut replicas: Vec<Vec<bool>> = (0..m)
            .map(|_| (0..dimension).map(|_| self.rng.random_bool(0.5)).collect())
            .collect();

        self.best_state = replicas[0].clone();
        self.best_cost = cost_fn(&self.best_state);
        self.current_step = 0;

        // Annealing loop
        for step in 0..self.config.total_steps {
            self.current_step = step;
            let temp = self.temperature();
            let j_perp = self.trotter_coupling();

            // Monte Carlo sweeps
            for _ in 0..self.config.sweeps_per_step {
                // Sweep over all spins in all replicas
                for replica_idx in 0..m {
                    for spin_idx in 0..dimension {
                        // Calculate energy change from flipping this spin
                        let delta_e = self.calculate_flip_energy(
                            &replicas,
                            replica_idx,
                            spin_idx,
                            j_perp,
                            &cost_fn,
                        );

                        // Metropolis acceptance
                        if delta_e < 0.0 || self.rng.random::<f64>() < (-delta_e / temp).exp() {
                            replicas[replica_idx][spin_idx] = !replicas[replica_idx][spin_idx];
                        }
                    }
                }
            }

            // Track best solution across all replicas
            for replica in &replicas {
                let cost = cost_fn(replica);
                if cost < self.best_cost {
                    self.best_cost = cost;
                    self.best_state = replica.clone();
                }
            }
        }

        self.best_state.clone()
    }

    /// Calculate energy change from flipping a spin
    fn calculate_flip_energy<F>(
        &self,
        replicas: &[Vec<bool>],
        replica_idx: usize,
        spin_idx: usize,
        j_perp: f64,
        cost_fn: &F,
    ) -> f64
    where
        F: Fn(&[bool]) -> f64,
    {
        let m = replicas.len();
        let current_spin = replicas[replica_idx][spin_idx];

        // Classical cost change (within this replica)
        let mut test_state = replicas[replica_idx].clone();
        let old_cost = cost_fn(&test_state);
        test_state[spin_idx] = !current_spin;
        let new_cost = cost_fn(&test_state);
        let classical_delta = (new_cost - old_cost) / m as f64;

        // Quantum coupling to neighboring Trotter slices
        let prev_replica = (replica_idx + m - 1) % m;
        let next_replica = (replica_idx + 1) % m;

        let prev_spin = replicas[prev_replica][spin_idx];
        let next_spin = replicas[next_replica][spin_idx];

        // Coupling energy: J_perp * sigma_i^k * (sigma_i^{k-1} + sigma_i^{k+1})
        // Flipping changes sign, so delta = -2 * J_perp * sigma * (neighbors)
        let spin_val = if current_spin { 1.0 } else { -1.0 };
        let neighbor_sum =
            (if prev_spin { 1.0 } else { -1.0 }) + (if next_spin { 1.0 } else { -1.0 });

        let quantum_delta = 2.0 * j_perp * spin_val * neighbor_sum;

        classical_delta + quantum_delta
    }

    /// Get the best cost found
    pub fn best_cost(&self) -> f64 {
        self.best_cost
    }

    /// Get the best state found
    pub fn best_state(&self) -> &[bool] {
        &self.best_state
    }
}

/// Query plan optimizer using quantum-inspired annealing
pub struct QueryPlanOptimizer {
    annealer: QuantumAnnealer,
    /// Number of available indexes
    num_indexes: usize,
    /// Number of possible join orders (factorial, but we encode as binary choices)
    num_join_choices: usize,
}

impl QueryPlanOptimizer {
    /// Create a new query plan optimizer
    pub fn new(num_indexes: usize, num_join_choices: usize) -> Self {
        let config = AnnealingConfig {
            total_steps: 500,
            trotter_slices: 8,
            ..Default::default()
        };
        Self {
            annealer: QuantumAnnealer::new(config),
            num_indexes,
            num_join_choices,
        }
    }

    /// Optimize query plan given cost estimates
    ///
    /// # Arguments
    /// * `index_costs` - Cost of using each index (benefit if negative)
    /// * `join_costs` - Cost matrix for join order choices
    /// * `base_cost` - Base query cost without optimizations
    ///
    /// # Returns
    /// (selected_indexes, join_order_bits, estimated_cost)
    pub fn optimize(
        &mut self,
        index_costs: &[f64],
        join_costs: &[Vec<f64>],
        base_cost: f64,
    ) -> (Vec<usize>, Vec<bool>, f64) {
        let dimension = self.num_indexes + self.num_join_choices;

        let index_costs = index_costs.to_vec();
        let join_costs: Vec<Vec<f64>> = join_costs.to_vec();
        let base = base_cost;
        let n_idx = self.num_indexes;

        let cost_fn = move |state: &[bool]| -> f64 {
            let mut cost = base;

            // Index selection costs
            for (i, &selected) in state.iter().take(n_idx).enumerate() {
                if selected && i < index_costs.len() {
                    cost += index_costs[i];
                }
            }

            // Join order costs (simple model)
            for (i, &choice) in state.iter().skip(n_idx).enumerate() {
                if i < join_costs.len() && !join_costs[i].is_empty() {
                    let idx = if choice {
                        1.min(join_costs[i].len() - 1)
                    } else {
                        0
                    };
                    cost += join_costs[i][idx];
                }
            }

            cost
        };

        let solution = self.annealer.optimize(dimension, cost_fn);

        let selected_indexes: Vec<usize> = solution
            .iter()
            .take(self.num_indexes)
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i) } else { None })
            .collect();

        let join_order: Vec<bool> = solution.iter().skip(self.num_indexes).copied().collect();

        (selected_indexes, join_order, self.annealer.best_cost())
    }
}

/// Simulated Quantum Tunneling for escaping local minima
pub struct QuantumTunneler {
    /// Tunneling probability
    tunnel_prob: f64,
    /// Maximum tunnel distance (Hamming)
    max_distance: usize,
    rng: StdRng,
}

impl QuantumTunneler {
    /// Create a new quantum tunneler
    pub fn new(tunnel_prob: f64, max_distance: usize, seed: u64) -> Self {
        Self {
            tunnel_prob,
            max_distance,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Attempt quantum tunnel to escape local minimum
    ///
    /// Returns a new state if tunneling succeeds, None otherwise
    pub fn tunnel(&mut self, state: &[bool]) -> Option<Vec<bool>> {
        if self.rng.random::<f64>() > self.tunnel_prob {
            return None;
        }

        let mut new_state = state.to_vec();
        let distance = self
            .rng
            .random_range(1..=self.max_distance.min(state.len()));

        // Flip `distance` random bits
        let mut flipped = std::collections::HashSet::new();
        while flipped.len() < distance {
            let idx = self.rng.random_range(0..state.len());
            if flipped.insert(idx) {
                new_state[idx] = !new_state[idx];
            }
        }

        Some(new_state)
    }

    /// Calculate tunneling amplitude (for diagnostics)
    pub fn tunneling_amplitude(&self, distance: usize, barrier_height: f64) -> f64 {
        // WKB approximation: T ≈ exp(-2 * ∫ sqrt(2m(V-E)) dx)
        // Simplified to: T ≈ exp(-α * d * sqrt(V))
        let alpha = 0.5;
        (-alpha * distance as f64 * barrier_height.sqrt()).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_annealer_basic() {
        let config = AnnealingConfig {
            total_steps: 100,
            trotter_slices: 4,
            ..Default::default()
        };
        let mut annealer = QuantumAnnealer::new(config);

        // Simple problem: minimize number of 1s
        let cost_fn = |state: &[bool]| state.iter().filter(|&&b| b).count() as f64;

        let solution = annealer.optimize(10, cost_fn);
        let cost = cost_fn(&solution);

        // Should find near-optimal (all zeros)
        assert!(cost <= 3.0, "Expected low cost, got {}", cost);
    }

    #[test]
    fn test_quantum_annealer_max_cut() {
        // Small MAX-CUT instance
        let config = AnnealingConfig {
            total_steps: 200,
            trotter_slices: 8,
            ..Default::default()
        };
        let mut annealer = QuantumAnnealer::new(config);

        // 4-node graph: edges (0,1), (0,2), (1,3), (2,3)
        let edges = vec![(0, 1), (0, 2), (1, 3), (2, 3)];

        // Cost = negative cut size (we minimize, so maximize cut)
        let cost_fn = |state: &[bool]| {
            let mut cut = 0.0;
            for &(i, j) in &edges {
                if state[i] != state[j] {
                    cut += 1.0;
                }
            }
            -cut // Negate to minimize
        };

        let solution = annealer.optimize(4, &cost_fn);
        let cut = -cost_fn(&solution);

        // Optimal cut is 4 (bipartite: {0,3} vs {1,2})
        assert!(cut >= 3.0, "Expected good cut, got {}", cut);
    }

    #[test]
    fn test_query_plan_optimizer() {
        let mut optimizer = QueryPlanOptimizer::new(3, 2);

        // Index costs: using index 0 saves -10, index 1 saves -5, index 2 costs +2
        let index_costs = vec![-10.0, -5.0, 2.0];

        // Join costs: each join has two choices
        let join_costs = vec![
            vec![5.0, 3.0], // Second choice is better
            vec![2.0, 4.0], // First choice is better
        ];

        let (indexes, joins, cost) = optimizer.optimize(&index_costs, &join_costs, 100.0);

        // Should select indexes 0 and 1 (both save cost)
        assert!(indexes.contains(&0), "Should select index 0");
        assert!(indexes.contains(&1), "Should select index 1");
        assert!(!indexes.contains(&2), "Should not select index 2");

        // Cost should be around 100 - 10 - 5 + 3 + 2 = 90
        assert!(cost < 95.0, "Expected optimized cost < 95, got {}", cost);
    }

    #[test]
    fn test_quantum_tunneler() {
        let mut tunneler = QuantumTunneler::new(1.0, 3, 42);

        let state = vec![true; 10];
        let tunneled = tunneler.tunnel(&state).unwrap();

        // Should have 1-3 bits flipped
        let hamming: usize = state
            .iter()
            .zip(tunneled.iter())
            .filter(|(a, b)| a != b)
            .count();

        assert!(hamming >= 1 && hamming <= 3);
    }

    #[test]
    fn test_tunneling_amplitude() {
        let tunneler = QuantumTunneler::new(0.1, 5, 42);

        let low_barrier = tunneler.tunneling_amplitude(2, 1.0);
        let high_barrier = tunneler.tunneling_amplitude(2, 10.0);

        // Higher barrier = lower amplitude
        assert!(low_barrier > high_barrier);
    }

    #[test]
    fn test_annealing_schedule() {
        let config = AnnealingConfig {
            initial_temp: 10.0,
            final_temp: 0.1,
            initial_gamma: 5.0,
            final_gamma: 0.05,
            total_steps: 100,
            ..Default::default()
        };

        let mut annealer = QuantumAnnealer::new(config);
        annealer.current_step = 0;
        let initial_temp = annealer.temperature();
        let initial_gamma = annealer.gamma();

        annealer.current_step = 50;
        let mid_temp = annealer.temperature();
        let mid_gamma = annealer.gamma();

        annealer.current_step = 99;
        let final_temp = annealer.temperature();
        let final_gamma = annealer.gamma();

        // Temperature and gamma should decrease
        assert!(initial_temp > mid_temp);
        assert!(mid_temp > final_temp);
        assert!(initial_gamma > mid_gamma);
        assert!(mid_gamma > final_gamma);
    }
}
