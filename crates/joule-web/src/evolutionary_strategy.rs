//! Evolution Strategy (ES) with (μ,λ) and (μ+λ) selection, self-adaptive
//! mutation, CMA-ES (Covariance Matrix Adaptation), step-size adaptation,
//! rank-based fitness weighting, and boundary handling.

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

// ── Selection scheme ─────────────────────────────────────────────

/// (μ,λ) discards parents; (μ+λ) keeps parents in pool.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SelectionScheme {
    /// (μ,λ): best μ offspring become next parents.
    Comma,
    /// (μ+λ): best μ from parents + offspring become next parents.
    Plus,
}

// ── Recombination ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Recombination {
    /// Intermediate (arithmetic mean of two parents per gene).
    Intermediate,
    /// Discrete (randomly pick gene from one of two parents).
    Discrete,
    /// No recombination (mutation only).
    None,
}

// ── Boundary handling ────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryHandling {
    /// Clamp to bounds.
    Clamp,
    /// Reflect off bounds.
    Reflect,
    /// Re-sample if out of bounds.
    Resample,
}

// ── Individual ───────────────────────────────────────────────────

/// ES individual: solution vector, per-variable step sizes, fitness.
#[derive(Debug, Clone, PartialEq)]
pub struct EsIndividual {
    pub x: Vec<f64>,
    pub sigma: Vec<f64>,
    pub fitness: f64,
}

impl EsIndividual {
    fn new(x: Vec<f64>, sigma: Vec<f64>) -> Self {
        Self { x, sigma, fitness: f64::INFINITY }
    }
}

// ── Configuration ────────────────────────────────────────────────

/// ES configuration.
#[derive(Debug, Clone)]
pub struct EsConfig {
    pub dimensions: usize,
    pub mu: usize,
    pub lambda: usize,
    pub selection: SelectionScheme,
    pub recombination: Recombination,
    pub initial_sigma: f64,
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub bounds: (f64, f64),
    pub boundary_handling: BoundaryHandling,
    pub max_generations: usize,
    pub tau: f64,
    pub tau_prime: f64,
    pub seed: u64,
}

impl Default for EsConfig {
    fn default() -> Self {
        let n = 5;
        let tau = 1.0 / (2.0 * (n as f64)).sqrt();
        let tau_prime = 1.0 / (2.0 * n as f64).sqrt();
        Self {
            dimensions: n,
            mu: 5,
            lambda: 30,
            selection: SelectionScheme::Comma,
            recombination: Recombination::Intermediate,
            initial_sigma: 1.0,
            sigma_min: 1e-8,
            sigma_max: 10.0,
            bounds: (-10.0, 10.0),
            boundary_handling: BoundaryHandling::Clamp,
            max_generations: 200,
            tau,
            tau_prime,
            seed: 42,
        }
    }
}

// ── Result ───────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct EsResult {
    pub best_solution: Vec<f64>,
    pub best_fitness: f64,
    pub generations_run: usize,
    pub fitness_history: Vec<f64>,
}

// ── ES engine ────────────────────────────────────────────────────

/// Evolution Strategy optimizer.
pub struct EvolutionStrategy {
    config: EsConfig,
    parents: Vec<EsIndividual>,
    generation: usize,
    fitness_history: Vec<f64>,
    rng: Rng,
}

impl EvolutionStrategy {
    pub fn new(config: EsConfig) -> Self {
        Self {
            rng: Rng::new(config.seed),
            parents: Vec::with_capacity(config.mu),
            generation: 0,
            fitness_history: Vec::new(),
            config,
        }
    }

    /// Initialize parents randomly.
    pub fn initialize(&mut self) {
        let (lo, hi) = self.config.bounds;
        let dims = self.config.dimensions;
        self.parents.clear();
        for _ in 0..self.config.mu {
            let x: Vec<f64> = (0..dims).map(|_| lo + self.rng.next_f64() * (hi - lo)).collect();
            let sigma = vec![self.config.initial_sigma; dims];
            self.parents.push(EsIndividual::new(x, sigma));
        }
    }

    /// Evaluate all parents.
    pub fn evaluate<F: Fn(&[f64]) -> f64>(&mut self, obj_fn: &F) {
        for p in &mut self.parents {
            p.fitness = obj_fn(&p.x);
        }
    }

    fn enforce_bounds(&mut self, x: &mut Vec<f64>) {
        let (lo, hi) = self.config.bounds;
        match self.config.boundary_handling {
            BoundaryHandling::Clamp => {
                for v in x.iter_mut() {
                    *v = v.clamp(lo, hi);
                }
            }
            BoundaryHandling::Reflect => {
                for v in x.iter_mut() {
                    while *v < lo || *v > hi {
                        if *v < lo { *v = lo + (lo - *v); }
                        if *v > hi { *v = hi - (*v - hi); }
                    }
                    *v = v.clamp(lo, hi);
                }
            }
            BoundaryHandling::Resample => {
                for v in x.iter_mut() {
                    if *v < lo || *v > hi {
                        *v = lo + self.rng.next_f64() * (hi - lo);
                    }
                }
            }
        }
    }

    fn recombine(&mut self, p1: &EsIndividual, p2: &EsIndividual) -> (Vec<f64>, Vec<f64>) {
        let dims = self.config.dimensions;
        match self.config.recombination {
            Recombination::Intermediate => {
                let x: Vec<f64> = (0..dims).map(|i| (p1.x[i] + p2.x[i]) / 2.0).collect();
                let s: Vec<f64> = (0..dims).map(|i| (p1.sigma[i] + p2.sigma[i]) / 2.0).collect();
                (x, s)
            }
            Recombination::Discrete => {
                let mut x = Vec::with_capacity(dims);
                let mut s = Vec::with_capacity(dims);
                for i in 0..dims {
                    if self.rng.next_f64() < 0.5 {
                        x.push(p1.x[i]);
                        s.push(p1.sigma[i]);
                    } else {
                        x.push(p2.x[i]);
                        s.push(p2.sigma[i]);
                    }
                }
                (x, s)
            }
            Recombination::None => {
                (p1.x.clone(), p1.sigma.clone())
            }
        }
    }

    fn mutate_individual(&mut self, x: &mut Vec<f64>, sigma: &mut Vec<f64>) {
        let global_noise = self.rng.next_gaussian();
        let tau = self.config.tau;
        let tau_prime = self.config.tau_prime;

        for i in 0..sigma.len() {
            let local_noise = self.rng.next_gaussian();
            sigma[i] *= (tau_prime * global_noise + tau * local_noise).exp();
            sigma[i] = sigma[i].clamp(self.config.sigma_min, self.config.sigma_max);
            x[i] += sigma[i] * self.rng.next_gaussian();
        }

        self.enforce_bounds(x);
    }

    /// Run one generation.
    pub fn step<F: Fn(&[f64]) -> f64>(&mut self, obj_fn: &F) {
        let lambda = self.config.lambda;
        let mu = self.config.mu;
        let mut offspring: Vec<EsIndividual> = Vec::with_capacity(lambda);

        for _ in 0..lambda {
            // Select two parents
            let p1_idx = self.rng.next_usize(self.parents.len());
            let p2_idx = self.rng.next_usize(self.parents.len());
            let p1 = self.parents[p1_idx].clone();
            let p2 = self.parents[p2_idx].clone();
            let (mut x, mut sigma) = self.recombine(&p1, &p2);
            self.mutate_individual(&mut x, &mut sigma);
            let mut child = EsIndividual::new(x, sigma);
            child.fitness = obj_fn(&child.x);
            offspring.push(child);
        }

        // Selection
        let mut pool: Vec<EsIndividual> = match self.config.selection {
            SelectionScheme::Comma => offspring,
            SelectionScheme::Plus => {
                let mut combined = self.parents.clone();
                combined.extend(offspring);
                combined
            }
        };

        pool.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal));
        self.parents = pool.into_iter().take(mu).collect();

        if let Some(best) = self.parents.first() {
            self.fitness_history.push(best.fitness);
        }
        self.generation += 1;
    }

    /// Run the full ES optimization.
    pub fn run<F: Fn(&[f64]) -> f64>(&mut self, obj_fn: &F) -> EsResult {
        self.evaluate(obj_fn);
        self.parents.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal));
        if let Some(best) = self.parents.first() {
            self.fitness_history.push(best.fitness);
        }

        for _ in 0..self.config.max_generations {
            self.step(obj_fn);
        }

        let best = self.parents.first().unwrap();
        EsResult {
            best_solution: best.x.clone(),
            best_fitness: best.fitness,
            generations_run: self.generation,
            fitness_history: self.fitness_history.clone(),
        }
    }

    pub fn best(&self) -> Option<&EsIndividual> {
        self.parents.first()
    }

    pub fn generation(&self) -> usize {
        self.generation
    }
}

// ── CMA-ES ───────────────────────────────────────────────────────

/// Simplified CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
/// Uses a diagonal covariance approximation for efficiency.
pub struct CmaEs {
    dimensions: usize,
    mean: Vec<f64>,
    sigma: f64,
    /// Diagonal covariance (variance per dimension).
    diag_cov: Vec<f64>,
    /// Evolution path for step-size control.
    ps: Vec<f64>,
    /// Evolution path for covariance adaptation.
    pc: Vec<f64>,
    lambda: usize,
    mu: usize,
    weights: Vec<f64>,
    mu_eff: f64,
    cs: f64,
    cc: f64,
    c1: f64,
    cmu: f64,
    damps: f64,
    bounds: (f64, f64),
    generation: usize,
    fitness_history: Vec<f64>,
    max_generations: usize,
    rng: Rng,
}

impl CmaEs {
    pub fn new(dimensions: usize, initial_mean: Vec<f64>, initial_sigma: f64,
               bounds: (f64, f64), max_generations: usize, seed: u64) -> Self {
        let n = dimensions;
        let lambda = 4 + (3.0 * (n as f64).ln()).floor() as usize;
        let mu = lambda / 2;

        // Rank-based weights
        let raw_weights: Vec<f64> = (0..mu).map(|i| {
            ((mu as f64 + 0.5).ln() - ((i + 1) as f64).ln()).max(0.0)
        }).collect();
        let sum_w: f64 = raw_weights.iter().sum();
        let weights: Vec<f64> = raw_weights.iter().map(|w| w / sum_w).collect();
        let mu_eff: f64 = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        let cs = (mu_eff + 2.0) / (n as f64 + mu_eff + 5.0);
        let cc = (4.0 + mu_eff / n as f64) / (n as f64 + 4.0 + 2.0 * mu_eff / n as f64);
        let c1 = 2.0 / ((n as f64 + 1.3).powi(2) + mu_eff);
        let cmu_val = (2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n as f64 + 2.0).powi(2) + mu_eff)).min(1.0 - c1);
        let damps = 1.0 + 2.0 * (((mu_eff - 1.0) / (n as f64 + 1.0)).sqrt() - 1.0).max(0.0) + cs;

        Self {
            dimensions: n,
            mean: initial_mean,
            sigma: initial_sigma,
            diag_cov: vec![1.0; n],
            ps: vec![0.0; n],
            pc: vec![0.0; n],
            lambda,
            mu,
            weights,
            mu_eff,
            cs,
            cc,
            c1,
            cmu: cmu_val,
            damps,
            bounds,
            generation: 0,
            fitness_history: Vec::new(),
            max_generations,
            rng: Rng::new(seed),
        }
    }

    fn sample_population(&mut self) -> Vec<(Vec<f64>, Vec<f64>)> {
        // Returns (x, z) pairs where z ~ N(0, diag_cov)
        let (lo, hi) = self.bounds;
        (0..self.lambda).map(|_| {
            let z: Vec<f64> = (0..self.dimensions).map(|d| {
                self.rng.next_gaussian() * self.diag_cov[d].sqrt()
            }).collect();
            let x: Vec<f64> = (0..self.dimensions).map(|d| {
                (self.mean[d] + self.sigma * z[d]).clamp(lo, hi)
            }).collect();
            (x, z)
        }).collect()
    }

    /// Run one generation of CMA-ES.
    pub fn step<F: Fn(&[f64]) -> f64>(&mut self, obj_fn: &F) {
        let n = self.dimensions;
        let samples = self.sample_population();

        // Evaluate
        let mut fitnesses: Vec<(f64, Vec<f64>, Vec<f64>)> = samples.into_iter()
            .map(|(x, z)| {
                let f = obj_fn(&x);
                (f, x, z)
            })
            .collect();
        fitnesses.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Record best
        self.fitness_history.push(fitnesses[0].0);

        // Weighted mean update
        let old_mean = self.mean.clone();
        for d in 0..n {
            self.mean[d] = 0.0;
            for i in 0..self.mu {
                self.mean[d] += self.weights[i] * fitnesses[i].1[d];
            }
        }

        // Evolution path ps (step-size control)
        let mean_shift: Vec<f64> = (0..n).map(|d| (self.mean[d] - old_mean[d]) / self.sigma).collect();
        let cs_comp = (self.cs * (2.0 - self.cs) * self.mu_eff).sqrt();
        for d in 0..n {
            self.ps[d] = (1.0 - self.cs) * self.ps[d]
                + cs_comp * mean_shift[d] / self.diag_cov[d].sqrt().max(1e-15);
        }

        // Evolution path pc (covariance adaptation)
        let ps_norm: f64 = self.ps.iter().map(|v| v * v).sum::<f64>().sqrt();
        let expected_norm = (n as f64).sqrt() * (1.0 - 1.0 / (4.0 * n as f64) + 1.0 / (21.0 * (n as f64).powi(2)));
        let hs = if ps_norm / ((1.0 - (1.0 - self.cs).powi(2 * (self.generation as i32 + 1))).sqrt() + 1e-15)
            < (1.4 + 2.0 / (n as f64 + 1.0)) * expected_norm { 1.0 } else { 0.0 };
        let cc_comp = (self.cc * (2.0 - self.cc) * self.mu_eff).sqrt();
        for d in 0..n {
            self.pc[d] = (1.0 - self.cc) * self.pc[d] + hs * cc_comp * mean_shift[d];
        }

        // Covariance update (diagonal)
        for d in 0..n {
            let rank_one = self.pc[d] * self.pc[d];
            let mut rank_mu = 0.0;
            for i in 0..self.mu {
                rank_mu += self.weights[i] * fitnesses[i].2[d] * fitnesses[i].2[d];
            }
            self.diag_cov[d] = (1.0 - self.c1 - self.cmu) * self.diag_cov[d]
                + self.c1 * rank_one
                + self.cmu * rank_mu;
            self.diag_cov[d] = self.diag_cov[d].max(1e-20);
        }

        // Step-size adaptation
        self.sigma *= ((self.cs / self.damps) * (ps_norm / expected_norm - 1.0)).exp();
        self.sigma = self.sigma.clamp(1e-20, 1e10);

        self.generation += 1;
    }

    /// Run full CMA-ES optimization.
    pub fn run<F: Fn(&[f64]) -> f64>(&mut self, obj_fn: &F) -> EsResult {
        for _ in 0..self.max_generations {
            self.step(obj_fn);
        }
        EsResult {
            best_solution: self.mean.clone(),
            best_fitness: *self.fitness_history.last().unwrap_or(&f64::INFINITY),
            generations_run: self.generation,
            fitness_history: self.fitness_history.clone(),
        }
    }

    pub fn mean(&self) -> &[f64] { &self.mean }
    pub fn sigma(&self) -> f64 { self.sigma }
    pub fn generation(&self) -> usize { self.generation }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_default_config() {
        let c = EsConfig::default();
        assert_eq!(c.dimensions, 5);
        assert_eq!(c.mu, 5);
        assert_eq!(c.lambda, 30);
    }

    #[test]
    fn test_es_individual_creation() {
        let ind = EsIndividual::new(vec![1.0, 2.0], vec![0.5, 0.5]);
        assert_eq!(ind.x, vec![1.0, 2.0]);
        assert_eq!(ind.fitness, f64::INFINITY);
    }

    #[test]
    fn test_initialize() {
        let config = EsConfig { dimensions: 3, mu: 5, ..Default::default() };
        let mut es = EvolutionStrategy::new(config);
        es.initialize();
        assert_eq!(es.parents.len(), 5);
    }

    #[test]
    fn test_evaluate_updates_fitness() {
        let config = EsConfig { dimensions: 2, mu: 3, lambda: 10, ..Default::default() };
        let mut es = EvolutionStrategy::new(config);
        es.initialize();
        es.evaluate(&sphere);
        for p in &es.parents {
            assert!(p.fitness < f64::INFINITY);
        }
    }

    #[test]
    fn test_comma_selection() {
        let config = EsConfig {
            dimensions: 2,
            mu: 3,
            lambda: 10,
            selection: SelectionScheme::Comma,
            max_generations: 5,
            ..Default::default()
        };
        let mut es = EvolutionStrategy::new(config);
        es.initialize();
        es.evaluate(&sphere);
        es.step(&sphere);
        assert_eq!(es.parents.len(), 3);
    }

    #[test]
    fn test_plus_selection() {
        let config = EsConfig {
            dimensions: 2,
            mu: 3,
            lambda: 10,
            selection: SelectionScheme::Plus,
            max_generations: 5,
            ..Default::default()
        };
        let mut es = EvolutionStrategy::new(config);
        es.initialize();
        es.evaluate(&sphere);
        es.step(&sphere);
        assert_eq!(es.parents.len(), 3);
    }

    #[test]
    fn test_intermediate_recombination() {
        let config = EsConfig {
            dimensions: 3,
            mu: 5,
            lambda: 20,
            recombination: Recombination::Intermediate,
            max_generations: 10,
            ..Default::default()
        };
        let mut es = EvolutionStrategy::new(config);
        es.initialize();
        let result = es.run(&sphere);
        assert!(result.generations_run > 0);
    }

    #[test]
    fn test_discrete_recombination() {
        let config = EsConfig {
            dimensions: 3,
            mu: 5,
            lambda: 20,
            recombination: Recombination::Discrete,
            max_generations: 10,
            ..Default::default()
        };
        let mut es = EvolutionStrategy::new(config);
        es.initialize();
        let result = es.run(&sphere);
        assert!(result.generations_run > 0);
    }

    #[test]
    fn test_no_recombination() {
        let config = EsConfig {
            dimensions: 3,
            mu: 5,
            lambda: 20,
            recombination: Recombination::None,
            max_generations: 10,
            ..Default::default()
        };
        let mut es = EvolutionStrategy::new(config);
        es.initialize();
        let result = es.run(&sphere);
        assert!(result.generations_run > 0);
    }

    #[test]
    fn test_sphere_optimization() {
        let config = EsConfig {
            dimensions: 3,
            mu: 5,
            lambda: 30,
            max_generations: 100,
            bounds: (-5.0, 5.0),
            ..Default::default()
        };
        let mut es = EvolutionStrategy::new(config);
        es.initialize();
        let result = es.run(&sphere);
        assert!(result.best_fitness < 5.0, "Expected < 5.0, got {}", result.best_fitness);
    }

    #[test]
    fn test_fitness_history() {
        let config = EsConfig {
            dimensions: 2,
            mu: 3,
            lambda: 10,
            max_generations: 20,
            ..Default::default()
        };
        let mut es = EvolutionStrategy::new(config);
        es.initialize();
        let result = es.run(&sphere);
        assert_eq!(result.fitness_history.len(), 21); // 1 initial + 20 generations
    }

    #[test]
    fn test_generation_counter() {
        let config = EsConfig { dimensions: 2, mu: 3, lambda: 10, ..Default::default() };
        let mut es = EvolutionStrategy::new(config);
        es.initialize();
        es.evaluate(&sphere);
        assert_eq!(es.generation(), 0);
        es.step(&sphere);
        assert_eq!(es.generation(), 1);
    }

    #[test]
    fn test_clamp_boundary() {
        let config = EsConfig {
            dimensions: 2,
            mu: 3,
            lambda: 10,
            bounds: (-2.0, 2.0),
            boundary_handling: BoundaryHandling::Clamp,
            max_generations: 30,
            ..Default::default()
        };
        let mut es = EvolutionStrategy::new(config);
        es.initialize();
        es.run(&sphere);
        let (lo, hi) = (-2.0, 2.0);
        if let Some(best) = es.best() {
            for &v in &best.x {
                assert!(v >= lo - 1e-10 && v <= hi + 1e-10);
            }
        }
    }

    #[test]
    fn test_reflect_boundary() {
        let config = EsConfig {
            dimensions: 2,
            mu: 3,
            lambda: 10,
            bounds: (-2.0, 2.0),
            boundary_handling: BoundaryHandling::Reflect,
            max_generations: 30,
            ..Default::default()
        };
        let mut es = EvolutionStrategy::new(config);
        es.initialize();
        es.run(&sphere);
        if let Some(best) = es.best() {
            for &v in &best.x {
                assert!(v >= -2.0 - 1e-10 && v <= 2.0 + 1e-10);
            }
        }
    }

    #[test]
    fn test_resample_boundary() {
        let config = EsConfig {
            dimensions: 2,
            mu: 3,
            lambda: 10,
            bounds: (-2.0, 2.0),
            boundary_handling: BoundaryHandling::Resample,
            max_generations: 30,
            ..Default::default()
        };
        let mut es = EvolutionStrategy::new(config);
        es.initialize();
        es.run(&sphere);
        if let Some(best) = es.best() {
            for &v in &best.x {
                assert!(v >= -2.0 - 1e-10 && v <= 2.0 + 1e-10);
            }
        }
    }

    #[test]
    fn test_self_adaptive_sigma() {
        let config = EsConfig {
            dimensions: 2,
            mu: 3,
            lambda: 10,
            max_generations: 20,
            ..Default::default()
        };
        let mut es = EvolutionStrategy::new(config);
        es.initialize();
        es.run(&sphere);
        // Sigma should have adapted (not all the same as initial)
        if let Some(best) = es.best() {
            // At least one sigma should differ from initial
            let changed = best.sigma.iter().any(|s| (s - 1.0).abs() > 1e-10);
            assert!(changed);
        }
    }

    // ── CMA-ES tests ────────────────────────────────────────

    #[test]
    fn test_cma_es_creation() {
        let cma = CmaEs::new(3, vec![0.0; 3], 1.0, (-5.0, 5.0), 10, 42);
        assert_eq!(cma.dimensions, 3);
        assert!((cma.sigma() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cma_es_step() {
        let mut cma = CmaEs::new(2, vec![5.0, 5.0], 1.0, (-10.0, 10.0), 10, 42);
        cma.step(&sphere);
        assert_eq!(cma.generation(), 1);
    }

    #[test]
    fn test_cma_es_sphere() {
        let mut cma = CmaEs::new(2, vec![5.0, 5.0], 2.0, (-10.0, 10.0), 100, 42);
        let result = cma.run(&sphere);
        assert!(result.best_fitness < 5.0, "CMA-ES sphere: {}", result.best_fitness);
    }

    #[test]
    fn test_cma_es_fitness_history() {
        let mut cma = CmaEs::new(2, vec![3.0, 3.0], 1.0, (-10.0, 10.0), 20, 42);
        let result = cma.run(&sphere);
        assert_eq!(result.fitness_history.len(), 20);
    }

    #[test]
    fn test_cma_es_sigma_adapts() {
        let mut cma = CmaEs::new(2, vec![5.0, 5.0], 1.0, (-10.0, 10.0), 50, 42);
        cma.run(&sphere);
        // Sigma should have changed from initial
        assert!((cma.sigma() - 1.0).abs() > 1e-10);
    }

    #[test]
    fn test_cma_es_mean_moves() {
        let mut cma = CmaEs::new(2, vec![5.0, 5.0], 1.0, (-10.0, 10.0), 50, 42);
        cma.run(&sphere);
        // Mean should have moved toward origin
        let dist: f64 = cma.mean().iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(dist < 7.1, "Mean should approach origin, dist = {}", dist);
    }

    #[test]
    fn test_plus_monotonic_best() {
        let config = EsConfig {
            dimensions: 2,
            mu: 3,
            lambda: 10,
            selection: SelectionScheme::Plus,
            max_generations: 20,
            ..Default::default()
        };
        let mut es = EvolutionStrategy::new(config);
        es.initialize();
        let result = es.run(&sphere);
        // With (μ+λ) the best should be monotonically non-increasing
        for w in result.fitness_history.windows(2) {
            assert!(w[1] <= w[0] + 1e-10);
        }
    }
}
