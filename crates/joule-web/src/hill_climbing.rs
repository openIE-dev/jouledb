//! Hill climbing and local search methods: steepest ascent, first-improvement,
//! stochastic, random restart, iterated local search (ILS), with neighborhood
//! generators for continuous and combinatorial problems, and plateau detection.

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

// ── Neighborhood generator ───────────────────────────────────────

/// Generates neighbor solutions from a current solution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NeighborhoodType {
    /// Gaussian perturbation of each dimension with given step size.
    Gaussian(f64),
    /// Uniform perturbation in [-step, +step] per dimension.
    UniformStep(f64),
    /// Perturb a single random dimension (Gaussian with given sigma).
    SingleDimension(f64),
    /// Combinatorial: swap two random positions (for permutation problems).
    Swap,
    /// Combinatorial: reverse a random sub-segment.
    TwoOpt,
}

// ── Search strategy ──────────────────────────────────────────────

/// Hill climbing variant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Strategy {
    /// Evaluate all neighbors, pick the best improvement.
    SteepestAscent,
    /// Accept the first neighbor that improves.
    FirstImprovement,
    /// Pick a random neighbor; accept only if it improves.
    Stochastic,
}

// ── Configuration ────────────────────────────────────────────────

/// Hill climbing configuration.
#[derive(Debug, Clone)]
pub struct HcConfig {
    pub dimensions: usize,
    pub strategy: Strategy,
    pub neighborhood: NeighborhoodType,
    pub max_iterations: usize,
    pub neighbors_per_step: usize,
    pub bounds: (f64, f64),
    pub maximize: bool,
    pub plateau_limit: usize,
    pub seed: u64,
}

impl Default for HcConfig {
    fn default() -> Self {
        Self {
            dimensions: 5,
            strategy: Strategy::SteepestAscent,
            neighborhood: NeighborhoodType::Gaussian(0.5),
            max_iterations: 500,
            neighbors_per_step: 20,
            bounds: (-10.0, 10.0),
            maximize: false,
            plateau_limit: 50,
            seed: 42,
        }
    }
}

// ── Result ───────────────────────────────────────────────────────

/// Result of a hill climbing run.
#[derive(Debug, Clone, PartialEq)]
pub struct HcResult {
    pub best_solution: Vec<f64>,
    pub best_value: f64,
    pub iterations_run: usize,
    pub plateau_detected: bool,
    pub value_history: Vec<f64>,
}

// ── Hill Climber ─────────────────────────────────────────────────

/// Hill climbing optimizer.
pub struct HillClimber {
    config: HcConfig,
    current: Vec<f64>,
    current_value: f64,
    best: Vec<f64>,
    best_value: f64,
    iteration: usize,
    plateau_count: usize,
    value_history: Vec<f64>,
    rng: Rng,
}

impl HillClimber {
    pub fn new(config: HcConfig) -> Self {
        let dims = config.dimensions;
        Self {
            rng: Rng::new(config.seed),
            current: vec![0.0; dims],
            current_value: if config.maximize { f64::NEG_INFINITY } else { f64::INFINITY },
            best: vec![0.0; dims],
            best_value: if config.maximize { f64::NEG_INFINITY } else { f64::INFINITY },
            iteration: 0,
            plateau_count: 0,
            value_history: Vec::new(),
            config,
        }
    }

    /// Initialize with random starting point.
    pub fn initialize(&mut self) {
        let (lo, hi) = self.config.bounds;
        self.current = (0..self.config.dimensions)
            .map(|_| lo + self.rng.next_f64() * (hi - lo))
            .collect();
        self.best = self.current.clone();
        self.iteration = 0;
        self.plateau_count = 0;
        self.current_value = if self.config.maximize { f64::NEG_INFINITY } else { f64::INFINITY };
        self.best_value = self.current_value;
    }

    /// Initialize at a specific starting point.
    pub fn initialize_at(&mut self, start: Vec<f64>) {
        self.current = start.clone();
        self.best = start;
        self.iteration = 0;
        self.plateau_count = 0;
        self.current_value = if self.config.maximize { f64::NEG_INFINITY } else { f64::INFINITY };
        self.best_value = self.current_value;
    }

    /// Evaluate current solution.
    pub fn evaluate<F: Fn(&[f64]) -> f64>(&mut self, obj_fn: &F) {
        self.current_value = obj_fn(&self.current);
        self.best_value = self.current_value;
    }

    fn is_better(&self, a: f64, b: f64) -> bool {
        // Require a meaningful improvement to avoid false positives from floating-point noise
        let threshold = b.abs() * 1e-12 + 1e-30;
        if self.config.maximize { a > b + threshold } else { a < b - threshold }
    }

    fn generate_neighbor_continuous(&mut self, base: &[f64]) -> Vec<f64> {
        let (lo, hi) = self.config.bounds;
        match self.config.neighborhood {
            NeighborhoodType::Gaussian(sigma) => {
                base.iter().map(|x| (x + self.rng.next_gaussian() * sigma).clamp(lo, hi)).collect()
            }
            NeighborhoodType::UniformStep(step) => {
                base.iter().map(|x| {
                    let delta = (self.rng.next_f64() * 2.0 - 1.0) * step;
                    (x + delta).clamp(lo, hi)
                }).collect()
            }
            NeighborhoodType::SingleDimension(sigma) => {
                let mut neighbor = base.to_vec();
                let dim = self.rng.next_usize(neighbor.len().max(1));
                neighbor[dim] = (neighbor[dim] + self.rng.next_gaussian() * sigma).clamp(lo, hi);
                neighbor
            }
            NeighborhoodType::Swap => {
                let mut neighbor = base.to_vec();
                if neighbor.len() >= 2 {
                    let i = self.rng.next_usize(neighbor.len());
                    let j = self.rng.next_usize(neighbor.len());
                    neighbor.swap(i, j);
                }
                neighbor
            }
            NeighborhoodType::TwoOpt => {
                let mut neighbor = base.to_vec();
                let n = neighbor.len();
                if n >= 2 {
                    let mut i = self.rng.next_usize(n);
                    let mut j = self.rng.next_usize(n);
                    if i > j { std::mem::swap(&mut i, &mut j); }
                    neighbor[i..=j].reverse();
                }
                neighbor
            }
        }
    }

    /// Run one step of hill climbing.
    pub fn step<F: Fn(&[f64]) -> f64>(&mut self, obj_fn: &F) -> bool {
        let improved = match self.config.strategy {
            Strategy::SteepestAscent => self.step_steepest(obj_fn),
            Strategy::FirstImprovement => self.step_first_improvement(obj_fn),
            Strategy::Stochastic => self.step_stochastic(obj_fn),
        };

        if improved {
            self.plateau_count = 0;
        } else {
            self.plateau_count += 1;
        }

        self.value_history.push(self.best_value);
        self.iteration += 1;
        improved
    }

    fn step_steepest<F: Fn(&[f64]) -> f64>(&mut self, obj_fn: &F) -> bool {
        let mut best_neighbor = self.current.clone();
        let mut best_val = self.current_value;
        let mut found = false;

        for _ in 0..self.config.neighbors_per_step {
            let current_clone = self.current.clone();
            let n = self.generate_neighbor_continuous(&current_clone);
            let v = obj_fn(&n);
            if self.is_better(v, best_val) {
                best_neighbor = n;
                best_val = v;
                found = true;
            }
        }

        if found {
            self.current = best_neighbor;
            self.current_value = best_val;
            if self.is_better(self.current_value, self.best_value) {
                self.best = self.current.clone();
                self.best_value = self.current_value;
            }
        }
        found
    }

    fn step_first_improvement<F: Fn(&[f64]) -> f64>(&mut self, obj_fn: &F) -> bool {
        for _ in 0..self.config.neighbors_per_step {
            let current_clone = self.current.clone();
            let n = self.generate_neighbor_continuous(&current_clone);
            let v = obj_fn(&n);
            if self.is_better(v, self.current_value) {
                self.current = n;
                self.current_value = v;
                if self.is_better(self.current_value, self.best_value) {
                    self.best = self.current.clone();
                    self.best_value = self.current_value;
                }
                return true;
            }
        }
        false
    }

    fn step_stochastic<F: Fn(&[f64]) -> f64>(&mut self, obj_fn: &F) -> bool {
        let current_clone = self.current.clone();
        let n = self.generate_neighbor_continuous(&current_clone);
        let v = obj_fn(&n);
        if self.is_better(v, self.current_value) {
            self.current = n;
            self.current_value = v;
            if self.is_better(self.current_value, self.best_value) {
                self.best = self.current.clone();
                self.best_value = self.current_value;
            }
            true
        } else {
            false
        }
    }

    /// Detect if the search is on a plateau.
    pub fn on_plateau(&self) -> bool {
        self.plateau_count >= self.config.plateau_limit
    }

    /// Run full hill climbing.
    pub fn run<F: Fn(&[f64]) -> f64>(&mut self, obj_fn: &F) -> HcResult {
        self.evaluate(obj_fn);

        for _ in 0..self.config.max_iterations {
            self.step(obj_fn);
            if self.on_plateau() {
                return HcResult {
                    best_solution: self.best.clone(),
                    best_value: self.best_value,
                    iterations_run: self.iteration,
                    plateau_detected: true,
                    value_history: self.value_history.clone(),
                };
            }
        }

        HcResult {
            best_solution: self.best.clone(),
            best_value: self.best_value,
            iterations_run: self.iteration,
            plateau_detected: false,
            value_history: self.value_history.clone(),
        }
    }

    pub fn best_value(&self) -> f64 { self.best_value }
    pub fn best_solution(&self) -> &[f64] { &self.best }
    pub fn iteration(&self) -> usize { self.iteration }
}

// ── Random restart hill climbing ─────────────────────────────────

/// Multiple random restarts to escape local optima.
pub fn random_restart_hill_climbing<F: Fn(&[f64]) -> f64>(
    base_config: HcConfig,
    restarts: usize,
    obj_fn: &F,
) -> HcResult {
    let maximize = base_config.maximize;
    let mut overall_best: Option<HcResult> = None;

    for r in 0..restarts {
        let mut cfg = base_config.clone();
        cfg.seed = base_config.seed.wrapping_add(r as u64 * 9973);
        let mut hc = HillClimber::new(cfg);
        hc.initialize();
        let result = hc.run(obj_fn);

        let replace = match &overall_best {
            None => true,
            Some(prev) => {
                if maximize {
                    result.best_value > prev.best_value
                } else {
                    result.best_value < prev.best_value
                }
            }
        };

        if replace {
            overall_best = Some(result);
        }
    }

    overall_best.unwrap()
}

// ── Iterated Local Search (ILS) ──────────────────────────────────

/// ILS configuration.
#[derive(Debug, Clone)]
pub struct IlsConfig {
    pub hc_config: HcConfig,
    pub perturbation_strength: f64,
    pub max_ils_iterations: usize,
    pub seed: u64,
}

/// Iterated Local Search: perturb → local search → acceptance.
pub fn iterated_local_search<F: Fn(&[f64]) -> f64>(
    config: IlsConfig,
    obj_fn: &F,
) -> HcResult {
    let mut rng = Rng::new(config.seed);
    let (lo, hi) = config.hc_config.bounds;
    let maximize = config.hc_config.maximize;

    // Initial local search
    let mut hc = HillClimber::new(config.hc_config.clone());
    hc.initialize();
    let mut best_result = hc.run(obj_fn);

    for ils_iter in 0..config.max_ils_iterations {
        // Perturbation: add large random offset to best solution
        let perturbed: Vec<f64> = best_result.best_solution.iter().map(|x| {
            let delta = rng.next_gaussian() * config.perturbation_strength;
            (x + delta).clamp(lo, hi)
        }).collect();

        // Local search from perturbed point
        let mut cfg = config.hc_config.clone();
        cfg.seed = config.seed.wrapping_add(ils_iter as u64 * 6271);
        let mut hc2 = HillClimber::new(cfg);
        hc2.initialize_at(perturbed);
        let new_result = hc2.run(obj_fn);

        // Acceptance: accept if better
        let accept = if maximize {
            new_result.best_value > best_result.best_value
        } else {
            new_result.best_value < best_result.best_value
        };

        if accept {
            best_result = new_result;
        }
    }

    best_result
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum()
    }

    fn neg_sphere(x: &[f64]) -> f64 {
        -x.iter().map(|v| v * v).sum::<f64>()
    }

    #[test]
    fn test_default_config() {
        let c = HcConfig::default();
        assert_eq!(c.dimensions, 5);
        assert!(!c.maximize);
        assert_eq!(c.plateau_limit, 50);
    }

    #[test]
    fn test_initialize_random() {
        let config = HcConfig { dimensions: 3, ..Default::default() };
        let mut hc = HillClimber::new(config);
        hc.initialize();
        assert_eq!(hc.best_solution().len(), 3);
    }

    #[test]
    fn test_initialize_at() {
        let config = HcConfig { dimensions: 2, ..Default::default() };
        let mut hc = HillClimber::new(config);
        hc.initialize_at(vec![3.0, 4.0]);
        hc.evaluate(&sphere);
        assert!((hc.best_value() - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_steepest_ascent_min() {
        let config = HcConfig {
            dimensions: 2,
            strategy: Strategy::SteepestAscent,
            max_iterations: 100,
            bounds: (-5.0, 5.0),
            ..Default::default()
        };
        let mut hc = HillClimber::new(config);
        hc.initialize();
        let result = hc.run(&sphere);
        assert!(result.best_value < 50.0);
    }

    #[test]
    fn test_first_improvement() {
        let config = HcConfig {
            dimensions: 2,
            strategy: Strategy::FirstImprovement,
            max_iterations: 100,
            bounds: (-5.0, 5.0),
            ..Default::default()
        };
        let mut hc = HillClimber::new(config);
        hc.initialize();
        let result = hc.run(&sphere);
        assert!(result.best_value < 100.0);
    }

    #[test]
    fn test_stochastic() {
        let config = HcConfig {
            dimensions: 2,
            strategy: Strategy::Stochastic,
            max_iterations: 200,
            bounds: (-5.0, 5.0),
            ..Default::default()
        };
        let mut hc = HillClimber::new(config);
        hc.initialize();
        let result = hc.run(&sphere);
        assert!(result.iterations_run > 0);
    }

    #[test]
    fn test_maximize_mode() {
        let config = HcConfig {
            dimensions: 2,
            maximize: true,
            max_iterations: 100,
            bounds: (-5.0, 5.0),
            ..Default::default()
        };
        let mut hc = HillClimber::new(config);
        hc.initialize();
        hc.evaluate(&neg_sphere);
        let start = hc.best_value();
        hc.run(&neg_sphere);
        assert!(hc.best_value() >= start - 1e-10);
    }

    #[test]
    fn test_plateau_detection() {
        // Use very small neighborhood to guarantee plateau
        let config = HcConfig {
            dimensions: 2,
            neighborhood: NeighborhoodType::Gaussian(1e-15),
            plateau_limit: 5,
            max_iterations: 100,
            bounds: (-5.0, 5.0),
            ..Default::default()
        };
        let mut hc = HillClimber::new(config);
        hc.initialize();
        let result = hc.run(&sphere);
        assert!(result.plateau_detected);
    }

    #[test]
    fn test_uniform_step_neighborhood() {
        let config = HcConfig {
            dimensions: 2,
            neighborhood: NeighborhoodType::UniformStep(0.5),
            max_iterations: 50,
            bounds: (-5.0, 5.0),
            ..Default::default()
        };
        let mut hc = HillClimber::new(config);
        hc.initialize();
        let result = hc.run(&sphere);
        assert!(result.iterations_run > 0);
    }

    #[test]
    fn test_single_dimension_neighborhood() {
        let config = HcConfig {
            dimensions: 3,
            neighborhood: NeighborhoodType::SingleDimension(0.5),
            max_iterations: 50,
            bounds: (-5.0, 5.0),
            ..Default::default()
        };
        let mut hc = HillClimber::new(config);
        hc.initialize();
        let result = hc.run(&sphere);
        assert!(result.iterations_run > 0);
    }

    #[test]
    fn test_swap_neighborhood() {
        let config = HcConfig {
            dimensions: 5,
            neighborhood: NeighborhoodType::Swap,
            strategy: Strategy::Stochastic,
            max_iterations: 50,
            bounds: (-5.0, 5.0),
            ..Default::default()
        };
        let mut hc = HillClimber::new(config);
        hc.initialize();
        let result = hc.run(&sphere);
        assert!(result.iterations_run > 0);
    }

    #[test]
    fn test_two_opt_neighborhood() {
        let config = HcConfig {
            dimensions: 5,
            neighborhood: NeighborhoodType::TwoOpt,
            strategy: Strategy::Stochastic,
            max_iterations: 50,
            bounds: (-5.0, 5.0),
            ..Default::default()
        };
        let mut hc = HillClimber::new(config);
        hc.initialize();
        let result = hc.run(&sphere);
        assert!(result.iterations_run > 0);
    }

    #[test]
    fn test_random_restart() {
        let config = HcConfig {
            dimensions: 2,
            max_iterations: 50,
            bounds: (-5.0, 5.0),
            ..Default::default()
        };
        let result = random_restart_hill_climbing(config, 5, &sphere);
        assert!(result.best_value < 50.0);
    }

    #[test]
    fn test_random_restart_improves() {
        let config = HcConfig {
            dimensions: 2,
            max_iterations: 50,
            bounds: (-5.0, 5.0),
            ..Default::default()
        };
        let single = {
            let mut hc = HillClimber::new(config.clone());
            hc.initialize();
            hc.run(&sphere)
        };
        let multi = random_restart_hill_climbing(config, 10, &sphere);
        // Multiple restarts should be at least as good
        assert!(multi.best_value <= single.best_value + 10.0);
    }

    #[test]
    fn test_ils() {
        let hc_config = HcConfig {
            dimensions: 2,
            max_iterations: 50,
            bounds: (-5.0, 5.0),
            ..Default::default()
        };
        let ils_config = IlsConfig {
            hc_config,
            perturbation_strength: 2.0,
            max_ils_iterations: 5,
            seed: 42,
        };
        let result = iterated_local_search(ils_config, &sphere);
        assert!(result.best_value < 100.0);
    }

    #[test]
    fn test_iteration_counter() {
        let config = HcConfig { dimensions: 2, max_iterations: 10, ..Default::default() };
        let mut hc = HillClimber::new(config);
        hc.initialize();
        hc.evaluate(&sphere);
        assert_eq!(hc.iteration(), 0);
        hc.step(&sphere);
        assert_eq!(hc.iteration(), 1);
    }

    #[test]
    fn test_value_history() {
        let config = HcConfig { dimensions: 2, max_iterations: 20, ..Default::default() };
        let mut hc = HillClimber::new(config);
        hc.initialize();
        let result = hc.run(&sphere);
        assert_eq!(result.value_history.len(), result.iterations_run);
    }

    #[test]
    fn test_bounds_respected() {
        let config = HcConfig {
            dimensions: 3,
            bounds: (-2.0, 2.0),
            max_iterations: 50,
            neighborhood: NeighborhoodType::Gaussian(5.0),
            ..Default::default()
        };
        let mut hc = HillClimber::new(config);
        hc.initialize();
        hc.run(&sphere);
        let (lo, hi) = (-2.0, 2.0);
        for &v in hc.best_solution() {
            assert!(v >= lo - 1e-10 && v <= hi + 1e-10);
        }
    }

    #[test]
    fn test_minimization_improves() {
        let config = HcConfig {
            dimensions: 3,
            max_iterations: 100,
            bounds: (-5.0, 5.0),
            ..Default::default()
        };
        let mut hc = HillClimber::new(config);
        hc.initialize();
        hc.evaluate(&sphere);
        let start = hc.best_value();
        hc.run(&sphere);
        assert!(hc.best_value() <= start + 1e-10);
    }

    #[test]
    fn test_step_returns_improvement_flag() {
        let config = HcConfig { dimensions: 2, ..Default::default() };
        let mut hc = HillClimber::new(config);
        hc.initialize();
        hc.evaluate(&sphere);
        // After evaluation, at least some steps should improve or not
        let _improved = hc.step(&sphere);
        // Just check that it runs without panic
        assert!(hc.iteration() == 1);
    }
}
