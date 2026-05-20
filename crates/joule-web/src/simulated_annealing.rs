//! Simulated annealing optimizer with configurable temperature schedules,
//! adaptive step size, reheating, Boltzmann/fast annealing, and parallel
//! tempering (multiple simultaneous temperatures).

// ── Simple deterministic PRNG ────────────────────────────────────

#[derive(Debug, Clone)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

// ── Temperature schedules ────────────────────────────────────────

/// Temperature cooling schedule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoolingSchedule {
    /// T(k) = T0 - k * (T0 - T_min) / max_iter
    Linear,
    /// T(k) = T0 * alpha^k
    Exponential { alpha: f64 },
    /// T(k) = T0 / (1 + alpha * ln(1 + k))
    Logarithmic { alpha: f64 },
}

// ── Annealing variant ────────────────────────────────────────────

/// Annealing flavor affecting neighbor generation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnnealingVariant {
    /// Standard Boltzmann: Gaussian perturbation proportional to sqrt(T).
    Boltzmann,
    /// Fast (Cauchy-like): perturbation proportional to T.
    Fast,
}

// ── Configuration ────────────────────────────────────────────────

/// Simulated annealing configuration.
#[derive(Debug, Clone)]
pub struct SaConfig {
    pub dimensions: usize,
    pub initial_temperature: f64,
    pub min_temperature: f64,
    pub cooling: CoolingSchedule,
    pub variant: AnnealingVariant,
    pub max_iterations: usize,
    pub step_size: f64,
    pub adaptive_step: bool,
    pub step_increase_factor: f64,
    pub step_decrease_factor: f64,
    pub bounds: (f64, f64),
    pub reheat_threshold: usize,
    pub reheat_factor: f64,
    pub seed: u64,
}

impl Default for SaConfig {
    fn default() -> Self {
        Self {
            dimensions: 5,
            initial_temperature: 100.0,
            min_temperature: 0.001,
            cooling: CoolingSchedule::Exponential { alpha: 0.995 },
            variant: AnnealingVariant::Boltzmann,
            max_iterations: 1000,
            step_size: 1.0,
            adaptive_step: true,
            step_increase_factor: 1.05,
            step_decrease_factor: 0.95,
            bounds: (-10.0, 10.0),
            reheat_threshold: 0,
            reheat_factor: 0.5,
            seed: 42,
        }
    }
}

// ── Result ───────────────────────────────────────────────────────

/// Result of an SA run.
#[derive(Debug, Clone, PartialEq)]
pub struct SaResult {
    pub best_solution: Vec<f64>,
    pub best_energy: f64,
    pub iterations_run: usize,
    pub energy_history: Vec<f64>,
    pub temperature_history: Vec<f64>,
    pub acceptance_rate: f64,
}

// ── Single-chain SA ──────────────────────────────────────────────

/// Simulated annealing optimizer.
pub struct SimulatedAnnealing {
    config: SaConfig,
    current: Vec<f64>,
    current_energy: f64,
    best: Vec<f64>,
    best_energy: f64,
    temperature: f64,
    iteration: usize,
    step_size: f64,
    energy_history: Vec<f64>,
    temperature_history: Vec<f64>,
    acceptances: usize,
    rejections: usize,
    stagnation_count: usize,
    rng: Rng,
}

impl SimulatedAnnealing {
    pub fn new(config: SaConfig) -> Self {
        let dims = config.dimensions;
        let temp = config.initial_temperature;
        let ss = config.step_size;
        Self {
            rng: Rng::new(config.seed),
            current: vec![0.0; dims],
            current_energy: f64::INFINITY,
            best: vec![0.0; dims],
            best_energy: f64::INFINITY,
            temperature: temp,
            iteration: 0,
            step_size: ss,
            energy_history: Vec::new(),
            temperature_history: Vec::new(),
            acceptances: 0,
            rejections: 0,
            stagnation_count: 0,
            config,
        }
    }

    /// Initialize with a random starting point.
    pub fn initialize(&mut self) {
        let (lo, hi) = self.config.bounds;
        self.current = (0..self.config.dimensions)
            .map(|_| lo + self.rng.next_f64() * (hi - lo))
            .collect();
        self.best = self.current.clone();
        self.temperature = self.config.initial_temperature;
        self.step_size = self.config.step_size;
        self.iteration = 0;
        self.acceptances = 0;
        self.rejections = 0;
        self.stagnation_count = 0;
    }

    /// Initialize with a specific starting point.
    pub fn initialize_at(&mut self, start: Vec<f64>) {
        self.current = start.clone();
        self.best = start;
        self.temperature = self.config.initial_temperature;
        self.step_size = self.config.step_size;
        self.iteration = 0;
        self.acceptances = 0;
        self.rejections = 0;
        self.stagnation_count = 0;
    }

    /// Evaluate the current solution.
    pub fn evaluate<F: Fn(&[f64]) -> f64>(&mut self, energy_fn: &F) {
        self.current_energy = energy_fn(&self.current);
        self.best_energy = self.current_energy;
    }

    fn update_temperature(&mut self) {
        self.temperature = match self.config.cooling {
            CoolingSchedule::Linear => {
                let t0 = self.config.initial_temperature;
                let t_min = self.config.min_temperature;
                let frac = self.iteration as f64 / self.config.max_iterations as f64;
                (t0 - frac * (t0 - t_min)).max(t_min)
            }
            CoolingSchedule::Exponential { alpha } => {
                (self.config.initial_temperature * alpha.powi(self.iteration as i32))
                    .max(self.config.min_temperature)
            }
            CoolingSchedule::Logarithmic { alpha } => {
                let t0 = self.config.initial_temperature;
                (t0 / (1.0 + alpha * (1.0 + self.iteration as f64).ln()))
                    .max(self.config.min_temperature)
            }
        };
    }

    fn generate_neighbor(&mut self) -> Vec<f64> {
        let (lo, hi) = self.config.bounds;
        // Use temperature ratio for a smooth anneal from full-range to fine-grained search.
        let t_ratio = (self.temperature / self.config.initial_temperature.max(1e-12)).max(0.001);
        let range = hi - lo;
        let scale = match self.config.variant {
            AnnealingVariant::Boltzmann => self.step_size * t_ratio.sqrt() * range * 0.1,
            AnnealingVariant::Fast => self.step_size * t_ratio * range * 0.1,
        };

        self.current.iter().map(|x| {
            let delta = self.rng.next_gaussian() * scale;
            (x + delta).clamp(lo, hi)
        }).collect()
    }

    /// Acceptance probability for a given energy delta.
    pub fn acceptance_probability(delta_e: f64, temperature: f64) -> f64 {
        if delta_e <= 0.0 {
            1.0
        } else {
            (-delta_e / temperature.max(1e-15)).exp()
        }
    }

    /// Run one iteration of SA.
    pub fn step<F: Fn(&[f64]) -> f64>(&mut self, energy_fn: &F) {
        let neighbor = self.generate_neighbor();
        let neighbor_energy = energy_fn(&neighbor);
        let delta = neighbor_energy - self.current_energy;
        let accept_prob = Self::acceptance_probability(delta, self.temperature);

        if self.rng.next_f64() < accept_prob {
            self.current = neighbor;
            self.current_energy = neighbor_energy;
            self.acceptances += 1;
            self.stagnation_count = 0;

            if self.config.adaptive_step {
                self.step_size *= self.config.step_increase_factor;
            }

            if self.current_energy < self.best_energy {
                self.best = self.current.clone();
                self.best_energy = self.current_energy;
            }
        } else {
            self.rejections += 1;
            self.stagnation_count += 1;

            if self.config.adaptive_step {
                self.step_size *= self.config.step_decrease_factor;
                // Prevent step collapse: keep at least 1% of original step.
                let min_step = self.config.step_size * 0.01;
                if self.step_size < min_step {
                    self.step_size = min_step;
                }
            }
        }

        // Reheating check
        if self.config.reheat_threshold > 0 && self.stagnation_count >= self.config.reheat_threshold {
            self.temperature = self.config.initial_temperature * self.config.reheat_factor;
            self.stagnation_count = 0;
        }

        self.iteration += 1;
        self.update_temperature();
        self.energy_history.push(self.best_energy);
        self.temperature_history.push(self.temperature);
    }

    /// Run the full SA optimization.
    pub fn run<F: Fn(&[f64]) -> f64>(&mut self, energy_fn: &F) -> SaResult {
        self.evaluate(energy_fn);

        for _ in 0..self.config.max_iterations {
            self.step(energy_fn);
            if self.temperature <= self.config.min_temperature {
                break;
            }
        }

        let total = (self.acceptances + self.rejections) as f64;
        SaResult {
            best_solution: self.best.clone(),
            best_energy: self.best_energy,
            iterations_run: self.iteration,
            energy_history: self.energy_history.clone(),
            temperature_history: self.temperature_history.clone(),
            acceptance_rate: if total > 0.0 { self.acceptances as f64 / total } else { 0.0 },
        }
    }

    /// Get current temperature.
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Get current best energy.
    pub fn best_energy(&self) -> f64 {
        self.best_energy
    }

    /// Get current best solution.
    pub fn best_solution(&self) -> &[f64] {
        &self.best
    }

    /// Get current iteration.
    pub fn iteration(&self) -> usize {
        self.iteration
    }
}

// ── Parallel tempering ───────────────────────────────────────────

/// Parallel tempering: multiple SA chains at different temperatures that
/// periodically swap states.
pub struct ParallelTempering {
    chains: Vec<SimulatedAnnealing>,
    swap_interval: usize,
    rng: Rng,
}

impl ParallelTempering {
    /// Create with a list of temperatures (one chain per temperature).
    pub fn new(base_config: SaConfig, temperatures: &[f64], swap_interval: usize) -> Self {
        let mut rng = Rng::new(base_config.seed);
        let chains: Vec<SimulatedAnnealing> = temperatures.iter().enumerate().map(|(i, &temp)| {
            let mut cfg = base_config.clone();
            cfg.initial_temperature = temp;
            cfg.seed = base_config.seed.wrapping_add(i as u64 * 7919);
            SimulatedAnnealing::new(cfg)
        }).collect();
        Self { chains, swap_interval, rng }
    }

    /// Initialize all chains.
    pub fn initialize(&mut self) {
        for c in &mut self.chains {
            c.initialize();
        }
    }

    /// Run parallel tempering.
    pub fn run<F: Fn(&[f64]) -> f64>(&mut self, energy_fn: &F, max_iterations: usize) -> SaResult {
        for c in &mut self.chains {
            c.evaluate(energy_fn);
        }

        for iter in 0..max_iterations {
            // Step each chain
            for c in &mut self.chains {
                c.step(energy_fn);
            }

            // Attempt swaps between adjacent chains
            if self.swap_interval > 0 && iter % self.swap_interval == 0 && self.chains.len() >= 2 {
                for i in 0..self.chains.len() - 1 {
                    let e_i = self.chains[i].current_energy;
                    let e_j = self.chains[i + 1].current_energy;
                    let t_i = self.chains[i].temperature;
                    let t_j = self.chains[i + 1].temperature;
                    let beta_i = 1.0 / t_i.max(1e-15);
                    let beta_j = 1.0 / t_j.max(1e-15);
                    let delta = (beta_i - beta_j) * (e_j - e_i);
                    let swap_prob = if delta <= 0.0 { 1.0 } else { (-delta).exp() };

                    if self.rng.next_f64() < swap_prob {
                        // Swap current solutions
                        let tmp_current = self.chains[i].current.clone();
                        let tmp_energy = self.chains[i].current_energy;
                        self.chains[i].current = self.chains[i + 1].current.clone();
                        self.chains[i].current_energy = self.chains[i + 1].current_energy;
                        self.chains[i + 1].current = tmp_current;
                        self.chains[i + 1].current_energy = tmp_energy;
                    }
                }
            }
        }

        // Find overall best
        let mut best_chain_idx = 0;
        let mut best_e = f64::INFINITY;
        for (i, c) in self.chains.iter().enumerate() {
            if c.best_energy < best_e {
                best_e = c.best_energy;
                best_chain_idx = i;
            }
        }

        let best_chain = &self.chains[best_chain_idx];
        SaResult {
            best_solution: best_chain.best.clone(),
            best_energy: best_chain.best_energy,
            iterations_run: max_iterations,
            energy_history: best_chain.energy_history.clone(),
            temperature_history: best_chain.temperature_history.clone(),
            acceptance_rate: {
                let total = (best_chain.acceptances + best_chain.rejections) as f64;
                if total > 0.0 { best_chain.acceptances as f64 / total } else { 0.0 }
            },
        }
    }

    /// Number of chains.
    pub fn chain_count(&self) -> usize {
        self.chains.len()
    }

    /// Best energy across all chains.
    pub fn best_energy(&self) -> f64 {
        self.chains.iter().map(|c| c.best_energy).fold(f64::INFINITY, f64::min)
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple sphere: min at origin = 0.
    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum()
    }

    fn rosenbrock(x: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..x.len() - 1 {
            sum += 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
        }
        sum
    }

    #[test]
    fn test_acceptance_probability_downhill() {
        assert!((SimulatedAnnealing::acceptance_probability(-1.0, 10.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_acceptance_probability_uphill() {
        let p = SimulatedAnnealing::acceptance_probability(1.0, 1.0);
        assert!(p > 0.0 && p < 1.0);
        assert!((p - (-1.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_acceptance_high_temp() {
        let p = SimulatedAnnealing::acceptance_probability(1.0, 1000.0);
        assert!(p > 0.99);
    }

    #[test]
    fn test_acceptance_low_temp() {
        let p = SimulatedAnnealing::acceptance_probability(10.0, 0.001);
        assert!(p < 0.01);
    }

    #[test]
    fn test_default_config() {
        let c = SaConfig::default();
        assert_eq!(c.dimensions, 5);
        assert!((c.initial_temperature - 100.0).abs() < 1e-10);
        assert!((c.min_temperature - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_initialize_random() {
        let config = SaConfig { dimensions: 3, ..Default::default() };
        let mut sa = SimulatedAnnealing::new(config);
        sa.initialize();
        assert_eq!(sa.best_solution().len(), 3);
    }

    #[test]
    fn test_initialize_at() {
        let config = SaConfig { dimensions: 2, ..Default::default() };
        let mut sa = SimulatedAnnealing::new(config);
        sa.initialize_at(vec![1.0, 2.0]);
        assert!((sa.best_solution()[0] - 1.0).abs() < 1e-10);
        assert!((sa.best_solution()[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_step_increases_iteration() {
        let config = SaConfig { dimensions: 2, ..Default::default() };
        let mut sa = SimulatedAnnealing::new(config);
        sa.initialize();
        sa.evaluate(&sphere);
        assert_eq!(sa.iteration(), 0);
        sa.step(&sphere);
        assert_eq!(sa.iteration(), 1);
    }

    #[test]
    fn test_temperature_decreases_exponential() {
        let config = SaConfig {
            dimensions: 2,
            cooling: CoolingSchedule::Exponential { alpha: 0.99 },
            ..Default::default()
        };
        let mut sa = SimulatedAnnealing::new(config);
        sa.initialize();
        sa.evaluate(&sphere);
        let t0 = sa.temperature();
        sa.step(&sphere);
        assert!(sa.temperature() < t0);
    }

    #[test]
    fn test_temperature_decreases_linear() {
        let config = SaConfig {
            dimensions: 2,
            cooling: CoolingSchedule::Linear,
            ..Default::default()
        };
        let mut sa = SimulatedAnnealing::new(config);
        sa.initialize();
        sa.evaluate(&sphere);
        let t0 = sa.temperature();
        sa.step(&sphere);
        assert!(sa.temperature() < t0 + 1e-10);
    }

    #[test]
    fn test_temperature_decreases_logarithmic() {
        let config = SaConfig {
            dimensions: 2,
            cooling: CoolingSchedule::Logarithmic { alpha: 1.0 },
            ..Default::default()
        };
        let mut sa = SimulatedAnnealing::new(config);
        sa.initialize();
        sa.evaluate(&sphere);
        let t0 = sa.temperature();
        sa.step(&sphere);
        assert!(sa.temperature() < t0 + 1e-10);
    }

    #[test]
    fn test_sphere_optimization() {
        let config = SaConfig {
            dimensions: 3,
            max_iterations: 500,
            initial_temperature: 50.0,
            bounds: (-5.0, 5.0),
            ..Default::default()
        };
        let mut sa = SimulatedAnnealing::new(config);
        sa.initialize();
        let result = sa.run(&sphere);
        assert!(result.best_energy < 10.0, "Expected < 10.0, got {}", result.best_energy);
    }

    #[test]
    fn test_best_energy_monotonic() {
        let config = SaConfig { dimensions: 2, max_iterations: 100, ..Default::default() };
        let mut sa = SimulatedAnnealing::new(config);
        sa.initialize();
        let result = sa.run(&sphere);
        // Best energy should be non-increasing in the history
        for w in result.energy_history.windows(2) {
            assert!(w[1] <= w[0] + 1e-10);
        }
    }

    #[test]
    fn test_adaptive_step_size() {
        let config = SaConfig {
            dimensions: 2,
            adaptive_step: true,
            step_increase_factor: 1.1,
            step_decrease_factor: 0.9,
            max_iterations: 50,
            ..Default::default()
        };
        let mut sa = SimulatedAnnealing::new(config);
        sa.initialize();
        let result = sa.run(&sphere);
        assert!(result.iterations_run > 0);
    }

    #[test]
    fn test_acceptance_rate() {
        let config = SaConfig {
            dimensions: 2,
            max_iterations: 200,
            initial_temperature: 100.0,
            ..Default::default()
        };
        let mut sa = SimulatedAnnealing::new(config);
        sa.initialize();
        let result = sa.run(&sphere);
        assert!(result.acceptance_rate >= 0.0 && result.acceptance_rate <= 1.0);
    }

    #[test]
    fn test_reheating() {
        let config = SaConfig {
            dimensions: 2,
            max_iterations: 200,
            reheat_threshold: 20,
            reheat_factor: 0.5,
            ..Default::default()
        };
        let mut sa = SimulatedAnnealing::new(config);
        sa.initialize();
        let result = sa.run(&sphere);
        assert!(result.iterations_run > 0);
    }

    #[test]
    fn test_fast_annealing_variant() {
        let config = SaConfig {
            dimensions: 2,
            variant: AnnealingVariant::Fast,
            max_iterations: 100,
            ..Default::default()
        };
        let mut sa = SimulatedAnnealing::new(config);
        sa.initialize();
        let result = sa.run(&sphere);
        assert!(result.best_energy < 200.0);
    }

    #[test]
    fn test_energy_history_length() {
        let config = SaConfig { dimensions: 2, max_iterations: 50, ..Default::default() };
        let mut sa = SimulatedAnnealing::new(config);
        sa.initialize();
        let result = sa.run(&sphere);
        assert_eq!(result.energy_history.len(), result.iterations_run);
    }

    #[test]
    fn test_parallel_tempering() {
        let config = SaConfig {
            dimensions: 2,
            bounds: (-5.0, 5.0),
            ..Default::default()
        };
        let temps = vec![100.0, 50.0, 10.0, 1.0];
        let mut pt = ParallelTempering::new(config, &temps, 10);
        pt.initialize();
        let result = pt.run(&sphere, 200);
        assert!(result.best_energy < 50.0, "PT best energy too high: {}", result.best_energy);
    }

    #[test]
    fn test_parallel_tempering_chain_count() {
        let config = SaConfig::default();
        let pt = ParallelTempering::new(config, &[100.0, 10.0, 1.0], 5);
        assert_eq!(pt.chain_count(), 3);
    }

    #[test]
    fn test_parallel_tempering_best_energy() {
        let config = SaConfig { dimensions: 2, bounds: (-3.0, 3.0), ..Default::default() };
        let mut pt = ParallelTempering::new(config, &[50.0, 5.0], 10);
        pt.initialize();
        pt.run(&sphere, 50);
        assert!(pt.best_energy() < f64::INFINITY);
    }

    #[test]
    fn test_rosenbrock_improvement() {
        let config = SaConfig {
            dimensions: 2,
            max_iterations: 500,
            bounds: (-5.0, 5.0),
            initial_temperature: 100.0,
            ..Default::default()
        };
        let mut sa = SimulatedAnnealing::new(config);
        sa.initialize();
        sa.evaluate(&rosenbrock);
        let start_energy = sa.best_energy();
        let result = sa.run(&rosenbrock);
        assert!(result.best_energy <= start_energy + 1e-10);
    }

    #[test]
    fn test_bounds_enforced() {
        let config = SaConfig {
            dimensions: 2,
            bounds: (-2.0, 2.0),
            max_iterations: 100,
            step_size: 5.0,
            ..Default::default()
        };
        let mut sa = SimulatedAnnealing::new(config);
        sa.initialize();
        sa.run(&sphere);
        let (lo, hi) = (-2.0, 2.0);
        for &v in sa.best_solution() {
            assert!(v >= lo - 1e-10 && v <= hi + 1e-10);
        }
    }
}
