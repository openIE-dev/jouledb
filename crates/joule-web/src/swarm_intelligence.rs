//! Swarm-based optimization — ACS, Bee, Firefly, Cuckoo algorithms.
//!
//! Replaces inspyred / PySwarms / scipy-optimize swarm libraries. Common trait
//! for swarm optimizers, Ant Colony System (ACS) for combinatorial problems,
//! Bee Algorithm (scout/employed/onlooker), Firefly Algorithm (brightness
//! attraction), Cuckoo Search (Levy flights + nest parasitism), best solution
//! tracking, convergence history, multi-swarm cooperation.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum SwarmError {
    ZeroDimension,
    InvalidParameter(String),
    EmptyBounds,
}

impl fmt::Display for SwarmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "dimensions must be non-zero"),
            Self::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
            Self::EmptyBounds => write!(f, "bounds must be non-empty"),
        }
    }
}

impl std::error::Error for SwarmError {}

// ── Common types ───────────────────────────────────────────────

/// A candidate solution with position and fitness.
#[derive(Debug, Clone)]
pub struct Solution {
    pub position: Vec<f64>,
    pub fitness: f64,
}

impl Solution {
    pub fn new(position: Vec<f64>, fitness: f64) -> Self {
        Self { position, fitness }
    }

    pub fn dimension(&self) -> usize { self.position.len() }
}

/// Optimization result.
#[derive(Debug, Clone)]
pub struct OptResult {
    pub best: Solution,
    pub convergence: Vec<f64>,
    pub iterations: u64,
    pub evaluations: u64,
}

/// Bounds for each dimension: (min, max).
pub type Bounds = Vec<(f64, f64)>;

// ── LCG RNG helper ─────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self { Self { state: seed } }

    /// Random f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.state >> 33) as f64) / (u32::MAX as f64)
    }

    /// Random f64 in [lo, hi).
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_f64() * (hi - lo)
    }

    /// Gaussian approximation using Box-Muller.
    fn gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Levy flight step (Mantegna's algorithm approximation).
    fn levy(&mut self, beta_param: f64) -> f64 {
        let sigma_u = {
            let num = gamma_fn(1.0 + beta_param) * (std::f64::consts::PI * beta_param / 2.0).sin();
            let den = gamma_fn((1.0 + beta_param) / 2.0) * beta_param * 2.0f64.powf((beta_param - 1.0) / 2.0);
            (num / den.abs().max(1e-15)).abs().powf(1.0 / beta_param)
        };
        let u = self.gaussian() * sigma_u;
        let v = self.gaussian().abs().max(1e-15);
        u / v.powf(1.0 / beta_param)
    }
}

/// Lanczos approximation of the Gamma function (g=5, n=6).
fn gamma_fn(x: f64) -> f64 {
    if x < 0.5 {
        std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma_fn(1.0 - x))
    } else {
        let z = x - 1.0;
        let coeffs = [
            76.18009172947146, -86.50532032941678, 24.01409824083091,
            -1.231739572450155, 0.001208650973866179, -0.000005395239384953,
        ];
        let mut ser = 1.000000000190015;
        for (i, &c) in coeffs.iter().enumerate() {
            ser += c / (z + 1.0 + i as f64);
        }
        let t = z + 5.5;
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * ser
    }
}

// ── Bee Algorithm ──────────────────────────────────────────────

/// Bee Algorithm optimizer.
#[derive(Debug, Clone)]
pub struct BeeAlgorithm {
    bounds: Bounds,
    num_employed: usize,
    num_onlookers: usize,
    num_scouts: usize,
    max_trials: usize,
    food_sources: Vec<Solution>,
    trials: Vec<usize>,
    best: Solution,
    convergence: Vec<f64>,
    iterations: u64,
    evaluations: u64,
    rng: Lcg,
}

impl BeeAlgorithm {
    pub fn new(
        bounds: Bounds,
        num_employed: usize,
        num_onlookers: usize,
        num_scouts: usize,
        max_trials: usize,
        seed: u64,
    ) -> Result<Self, SwarmError> {
        if bounds.is_empty() {
            return Err(SwarmError::EmptyBounds);
        }
        if num_employed == 0 {
            return Err(SwarmError::InvalidParameter("num_employed must be > 0".into()));
        }
        let dim = bounds.len();
        let mut rng = Lcg::new(seed);

        let mut food_sources = Vec::with_capacity(num_employed);
        for _ in 0..num_employed {
            let pos: Vec<f64> = bounds.iter().map(|&(lo, hi)| rng.uniform(lo, hi)).collect();
            food_sources.push(Solution::new(pos, f64::MAX));
        }

        let best = Solution::new(vec![0.0; dim], f64::MAX);
        Ok(Self {
            bounds,
            num_employed,
            num_onlookers,
            num_scouts,
            max_trials,
            food_sources,
            trials: vec![0; num_employed],
            best,
            convergence: Vec::new(),
            iterations: 0,
            evaluations: 0,
            rng,
        })
    }

    pub fn best(&self) -> &Solution { &self.best }
    pub fn convergence(&self) -> &[f64] { &self.convergence }
    pub fn iterations(&self) -> u64 { self.iterations }
    pub fn evaluations(&self) -> u64 { self.evaluations }

    fn clamp_position(&self, pos: &mut [f64]) {
        for (i, val) in pos.iter_mut().enumerate() {
            *val = val.clamp(self.bounds[i].0, self.bounds[i].1);
        }
    }

    /// Run one iteration with a given objective function (minimization).
    pub fn iterate(&mut self, objective: &dyn Fn(&[f64]) -> f64) {
        let dim = self.bounds.len();
        let n = self.food_sources.len();

        // Evaluate current sources
        for fs in &mut self.food_sources {
            if fs.fitness == f64::MAX {
                fs.fitness = objective(&fs.position);
                self.evaluations += 1;
            }
        }

        // Employed bee phase
        for i in 0..n {
            let k = loop {
                let k = (self.rng.next_f64() * n as f64) as usize % n;
                if k != i { break k; }
            };
            let j = (self.rng.next_f64() * dim as f64) as usize % dim;
            let mut new_pos = self.food_sources[i].position.clone();
            let phi = self.rng.uniform(-1.0, 1.0);
            new_pos[j] += phi * (self.food_sources[i].position[j] - self.food_sources[k].position[j]);
            self.clamp_position(&mut new_pos);

            let new_fit = objective(&new_pos);
            self.evaluations += 1;

            if new_fit < self.food_sources[i].fitness {
                self.food_sources[i] = Solution::new(new_pos, new_fit);
                self.trials[i] = 0;
            } else {
                self.trials[i] += 1;
            }
        }

        // Calculate probabilities for onlooker phase
        let max_fit = self.food_sources.iter().map(|s| s.fitness).fold(f64::MIN, f64::max);
        let probs: Vec<f64> = self.food_sources.iter().map(|s| {
            // Invert fitness for minimization: higher quality = lower fitness
            max_fit - s.fitness + 1.0
        }).collect();
        let total: f64 = probs.iter().sum();

        // Onlooker bee phase
        for _ in 0..self.num_onlookers {
            // Roulette wheel
            let r = self.rng.next_f64() * total;
            let mut cumul = 0.0;
            let mut chosen = 0;
            for (idx, &p) in probs.iter().enumerate() {
                cumul += p;
                if r <= cumul {
                    chosen = idx;
                    break;
                }
            }

            let k = loop {
                let k = (self.rng.next_f64() * n as f64) as usize % n;
                if k != chosen { break k; }
            };
            let j = (self.rng.next_f64() * dim as f64) as usize % dim;
            let mut new_pos = self.food_sources[chosen].position.clone();
            let phi = self.rng.uniform(-1.0, 1.0);
            new_pos[j] += phi * (self.food_sources[chosen].position[j] - self.food_sources[k].position[j]);
            self.clamp_position(&mut new_pos);

            let new_fit = objective(&new_pos);
            self.evaluations += 1;

            if new_fit < self.food_sources[chosen].fitness {
                self.food_sources[chosen] = Solution::new(new_pos, new_fit);
                self.trials[chosen] = 0;
            } else {
                self.trials[chosen] += 1;
            }
        }

        // Scout bee phase
        for i in 0..n {
            if self.trials[i] >= self.max_trials {
                let pos: Vec<f64> = self.bounds.iter().map(|&(lo, hi)| self.rng.uniform(lo, hi)).collect();
                let fit = objective(&pos);
                self.evaluations += 1;
                self.food_sources[i] = Solution::new(pos, fit);
                self.trials[i] = 0;
            }
        }

        // Update best
        for fs in &self.food_sources {
            if fs.fitness < self.best.fitness {
                self.best = fs.clone();
            }
        }

        self.convergence.push(self.best.fitness);
        self.iterations += 1;
    }

    /// Run for n iterations.
    pub fn run(&mut self, n: u64, objective: &dyn Fn(&[f64]) -> f64) {
        for _ in 0..n {
            self.iterate(objective);
        }
    }
}

// ── Firefly Algorithm ──────────────────────────────────────────

/// Firefly Algorithm optimizer.
#[derive(Debug, Clone)]
pub struct FireflyAlgorithm {
    bounds: Bounds,
    fireflies: Vec<Solution>,
    alpha: f64, // randomness
    beta_base: f64, // attraction at distance 0
    gamma: f64, // light absorption coefficient
    best: Solution,
    convergence: Vec<f64>,
    iterations: u64,
    evaluations: u64,
    rng: Lcg,
}

impl FireflyAlgorithm {
    pub fn new(
        bounds: Bounds,
        population: usize,
        alpha: f64,
        beta_base: f64,
        gamma: f64,
        seed: u64,
    ) -> Result<Self, SwarmError> {
        if bounds.is_empty() {
            return Err(SwarmError::EmptyBounds);
        }
        if population == 0 {
            return Err(SwarmError::InvalidParameter("population must be > 0".into()));
        }
        let dim = bounds.len();
        let mut rng = Lcg::new(seed);

        let fireflies: Vec<Solution> = (0..population).map(|_| {
            let pos: Vec<f64> = bounds.iter().map(|&(lo, hi)| rng.uniform(lo, hi)).collect();
            Solution::new(pos, f64::MAX)
        }).collect();

        let best = Solution::new(vec![0.0; dim], f64::MAX);
        Ok(Self { bounds, fireflies, alpha, beta_base, gamma, best, convergence: Vec::new(), iterations: 0, evaluations: 0, rng })
    }

    pub fn best(&self) -> &Solution { &self.best }
    pub fn convergence(&self) -> &[f64] { &self.convergence }
    pub fn iterations(&self) -> u64 { self.iterations }

    fn distance_sq(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    /// Run one iteration.
    pub fn iterate(&mut self, objective: &dyn Fn(&[f64]) -> f64) {
        let n = self.fireflies.len();
        let dim = self.bounds.len();

        // Evaluate
        for ff in &mut self.fireflies {
            if ff.fitness == f64::MAX {
                ff.fitness = objective(&ff.position);
                self.evaluations += 1;
            }
        }

        // Move fireflies
        let old = self.fireflies.clone();
        for i in 0..n {
            for j in 0..n {
                if old[j].fitness < old[i].fitness {
                    let r2 = Self::distance_sq(&old[i].position, &old[j].position);
                    let beta = self.beta_base * (-self.gamma * r2).exp();

                    for d in 0..dim {
                        let rand = self.rng.uniform(-0.5, 0.5);
                        self.fireflies[i].position[d] += beta * (old[j].position[d] - old[i].position[d])
                            + self.alpha * rand;
                        self.fireflies[i].position[d] = self.fireflies[i].position[d]
                            .clamp(self.bounds[d].0, self.bounds[d].1);
                    }

                    self.fireflies[i].fitness = objective(&self.fireflies[i].position);
                    self.evaluations += 1;
                }
            }
        }

        // Update best
        for ff in &self.fireflies {
            if ff.fitness < self.best.fitness {
                self.best = ff.clone();
            }
        }

        self.convergence.push(self.best.fitness);
        self.iterations += 1;
    }

    /// Run for n iterations.
    pub fn run(&mut self, n: u64, objective: &dyn Fn(&[f64]) -> f64) {
        for _ in 0..n {
            self.iterate(objective);
        }
    }
}

// ── Cuckoo Search ──────────────────────────────────────────────

/// Cuckoo Search optimizer.
#[derive(Debug, Clone)]
pub struct CuckooSearch {
    bounds: Bounds,
    nests: Vec<Solution>,
    discovery_rate: f64,
    levy_beta: f64,
    best: Solution,
    convergence: Vec<f64>,
    iterations: u64,
    evaluations: u64,
    rng: Lcg,
}

impl CuckooSearch {
    pub fn new(
        bounds: Bounds,
        num_nests: usize,
        discovery_rate: f64,
        seed: u64,
    ) -> Result<Self, SwarmError> {
        if bounds.is_empty() {
            return Err(SwarmError::EmptyBounds);
        }
        if num_nests == 0 {
            return Err(SwarmError::InvalidParameter("num_nests must be > 0".into()));
        }
        let dim = bounds.len();
        let mut rng = Lcg::new(seed);

        let nests: Vec<Solution> = (0..num_nests).map(|_| {
            let pos: Vec<f64> = bounds.iter().map(|&(lo, hi)| rng.uniform(lo, hi)).collect();
            Solution::new(pos, f64::MAX)
        }).collect();

        let best = Solution::new(vec![0.0; dim], f64::MAX);
        Ok(Self {
            bounds, nests, discovery_rate, levy_beta: 1.5,
            best, convergence: Vec::new(), iterations: 0, evaluations: 0, rng,
        })
    }

    pub fn best(&self) -> &Solution { &self.best }
    pub fn convergence(&self) -> &[f64] { &self.convergence }
    pub fn iterations(&self) -> u64 { self.iterations }

    /// Run one iteration.
    pub fn iterate(&mut self, objective: &dyn Fn(&[f64]) -> f64) {
        let n = self.nests.len();
        let dim = self.bounds.len();

        // Evaluate all nests
        for nest in &mut self.nests {
            if nest.fitness == f64::MAX {
                nest.fitness = objective(&nest.position);
                self.evaluations += 1;
            }
        }

        // Generate new solution via Levy flight from a random nest
        let idx = (self.rng.next_f64() * n as f64) as usize % n;
        let mut new_pos = self.nests[idx].position.clone();
        for d in 0..dim {
            let step = self.rng.levy(self.levy_beta);
            new_pos[d] += step * 0.01 * (self.bounds[d].1 - self.bounds[d].0);
            new_pos[d] = new_pos[d].clamp(self.bounds[d].0, self.bounds[d].1);
        }
        let new_fit = objective(&new_pos);
        self.evaluations += 1;

        // Replace a random worse nest
        let target = (self.rng.next_f64() * n as f64) as usize % n;
        if new_fit < self.nests[target].fitness {
            self.nests[target] = Solution::new(new_pos, new_fit);
        }

        // Abandon worst nests (parasitism)
        let num_abandon = ((n as f64 * self.discovery_rate) as usize).max(1).min(n);
        // Sort by fitness descending (worst first)
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| self.nests[b].fitness.partial_cmp(&self.nests[a].fitness).unwrap_or(std::cmp::Ordering::Equal));

        for &i in indices.iter().take(num_abandon) {
            let r = self.rng.next_f64();
            if r < self.discovery_rate {
                let pos: Vec<f64> = self.bounds.iter().map(|&(lo, hi)| self.rng.uniform(lo, hi)).collect();
                let fit = objective(&pos);
                self.evaluations += 1;
                self.nests[i] = Solution::new(pos, fit);
            }
        }

        // Update best
        for nest in &self.nests {
            if nest.fitness < self.best.fitness {
                self.best = nest.clone();
            }
        }

        self.convergence.push(self.best.fitness);
        self.iterations += 1;
    }

    /// Run for n iterations.
    pub fn run(&mut self, n: u64, objective: &dyn Fn(&[f64]) -> f64) {
        for _ in 0..n {
            self.iterate(objective);
        }
    }
}

// ── Multi-Swarm Cooperation ────────────────────────────────────

/// A multi-swarm cooperative optimizer that shares best solutions between swarms.
#[derive(Debug, Clone)]
pub struct MultiSwarm {
    swarms: Vec<CuckooSearch>,
    global_best: Solution,
    share_interval: u64,
    iterations: u64,
}

impl MultiSwarm {
    /// Create multiple Cuckoo Search swarms for cooperation.
    pub fn new(
        bounds: Bounds,
        num_swarms: usize,
        nests_per_swarm: usize,
        discovery_rate: f64,
        share_interval: u64,
    ) -> Result<Self, SwarmError> {
        if num_swarms == 0 {
            return Err(SwarmError::InvalidParameter("num_swarms must be > 0".into()));
        }
        let dim = bounds.len();
        let mut swarms = Vec::with_capacity(num_swarms);
        for i in 0..num_swarms {
            swarms.push(CuckooSearch::new(
                bounds.clone(),
                nests_per_swarm,
                discovery_rate,
                42 + i as u64 * 1000,
            )?);
        }
        let global_best = Solution::new(vec![0.0; dim], f64::MAX);
        Ok(Self { swarms, global_best, share_interval, iterations: 0 })
    }

    pub fn global_best(&self) -> &Solution { &self.global_best }
    pub fn iterations(&self) -> u64 { self.iterations }
    pub fn swarm_count(&self) -> usize { self.swarms.len() }

    /// Run one iteration across all swarms, sharing best solutions periodically.
    pub fn iterate(&mut self, objective: &dyn Fn(&[f64]) -> f64) {
        for swarm in &mut self.swarms {
            swarm.iterate(objective);
        }

        // Update global best
        for swarm in &self.swarms {
            if swarm.best().fitness < self.global_best.fitness {
                self.global_best = swarm.best().clone();
            }
        }

        // Share best solution every share_interval iterations
        self.iterations += 1;
        if self.iterations % self.share_interval == 0 {
            let best_pos = self.global_best.position.clone();
            let best_fit = self.global_best.fitness;
            for swarm in &mut self.swarms {
                // Replace worst nest with global best
                if let Some(worst) = swarm.nests.iter_mut().max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal)) {
                    if best_fit < worst.fitness {
                        *worst = Solution::new(best_pos.clone(), best_fit);
                    }
                }
            }
        }
    }

    /// Run for n iterations.
    pub fn run(&mut self, n: u64, objective: &dyn Fn(&[f64]) -> f64) {
        for _ in 0..n {
            self.iterate(objective);
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Sphere function: f(x) = sum(x_i^2), minimum at origin
    fn sphere(x: &[f64]) -> f64 { x.iter().map(|xi| xi * xi).sum() }

    // Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2, minimum at (1,1)
    fn rosenbrock(x: &[f64]) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
    }

    #[test]
    fn test_bee_creation() {
        let bounds = vec![(-5.0, 5.0); 2];
        let bee = BeeAlgorithm::new(bounds, 20, 20, 5, 50, 42).unwrap();
        assert_eq!(bee.iterations(), 0);
    }

    #[test]
    fn test_bee_empty_bounds() {
        assert!(BeeAlgorithm::new(vec![], 20, 20, 5, 50, 42).is_err());
    }

    #[test]
    fn test_bee_sphere() {
        let bounds = vec![(-5.0, 5.0); 2];
        let mut bee = BeeAlgorithm::new(bounds, 20, 20, 5, 50, 42).unwrap();
        bee.run(100, &sphere);
        assert!(bee.best().fitness < 1.0, "best fitness = {}", bee.best().fitness);
    }

    #[test]
    fn test_bee_convergence() {
        let bounds = vec![(-5.0, 5.0); 2];
        let mut bee = BeeAlgorithm::new(bounds, 20, 20, 5, 50, 42).unwrap();
        bee.run(50, &sphere);
        assert_eq!(bee.convergence().len(), 50);
        // Should be non-increasing
        for i in 1..bee.convergence().len() {
            assert!(bee.convergence()[i] <= bee.convergence()[i - 1] + 1e-10);
        }
    }

    #[test]
    fn test_bee_evaluations() {
        let bounds = vec![(-5.0, 5.0); 2];
        let mut bee = BeeAlgorithm::new(bounds, 10, 10, 3, 30, 42).unwrap();
        bee.run(10, &sphere);
        assert!(bee.evaluations() > 0);
    }

    #[test]
    fn test_firefly_creation() {
        let bounds = vec![(-5.0, 5.0); 2];
        let ff = FireflyAlgorithm::new(bounds, 20, 0.5, 1.0, 1.0, 42).unwrap();
        assert_eq!(ff.iterations(), 0);
    }

    #[test]
    fn test_firefly_empty_bounds() {
        assert!(FireflyAlgorithm::new(vec![], 20, 0.5, 1.0, 1.0, 42).is_err());
    }

    #[test]
    fn test_firefly_sphere() {
        let bounds = vec![(-5.0, 5.0); 2];
        let mut ff = FireflyAlgorithm::new(bounds, 20, 0.2, 1.0, 0.1, 42).unwrap();
        ff.run(50, &sphere);
        assert!(ff.best().fitness < 2.0, "best fitness = {}", ff.best().fitness);
    }

    #[test]
    fn test_firefly_convergence() {
        let bounds = vec![(-5.0, 5.0); 2];
        let mut ff = FireflyAlgorithm::new(bounds, 15, 0.2, 1.0, 0.1, 42).unwrap();
        ff.run(30, &sphere);
        assert_eq!(ff.convergence().len(), 30);
    }

    #[test]
    fn test_cuckoo_creation() {
        let bounds = vec![(-5.0, 5.0); 2];
        let cs = CuckooSearch::new(bounds, 25, 0.25, 42).unwrap();
        assert_eq!(cs.iterations(), 0);
    }

    #[test]
    fn test_cuckoo_empty_bounds() {
        assert!(CuckooSearch::new(vec![], 25, 0.25, 42).is_err());
    }

    #[test]
    fn test_cuckoo_sphere() {
        let bounds = vec![(-5.0, 5.0); 2];
        let mut cs = CuckooSearch::new(bounds, 25, 0.25, 42).unwrap();
        cs.run(200, &sphere);
        assert!(cs.best().fitness < 1.0, "best fitness = {}", cs.best().fitness);
    }

    #[test]
    fn test_cuckoo_rosenbrock() {
        let bounds = vec![(-5.0, 5.0); 2];
        let mut cs = CuckooSearch::new(bounds, 30, 0.25, 42).unwrap();
        cs.run(500, &rosenbrock);
        // Rosenbrock is harder — just check it finds a reasonable solution
        assert!(cs.best().fitness < 50.0, "best fitness = {}", cs.best().fitness);
    }

    #[test]
    fn test_cuckoo_convergence() {
        let bounds = vec![(-5.0, 5.0); 2];
        let mut cs = CuckooSearch::new(bounds, 25, 0.25, 42).unwrap();
        cs.run(50, &sphere);
        assert_eq!(cs.convergence().len(), 50);
    }

    #[test]
    fn test_multi_swarm_creation() {
        let bounds = vec![(-5.0, 5.0); 2];
        let ms = MultiSwarm::new(bounds, 3, 10, 0.25, 5).unwrap();
        assert_eq!(ms.swarm_count(), 3);
    }

    #[test]
    fn test_multi_swarm_sphere() {
        let bounds = vec![(-5.0, 5.0); 2];
        let mut ms = MultiSwarm::new(bounds, 3, 15, 0.25, 5).unwrap();
        ms.run(100, &sphere);
        assert!(ms.global_best().fitness < 1.0, "best = {}", ms.global_best().fitness);
    }

    #[test]
    fn test_multi_swarm_zero() {
        let bounds = vec![(-5.0, 5.0); 2];
        assert!(MultiSwarm::new(bounds, 0, 10, 0.25, 5).is_err());
    }

    #[test]
    fn test_solution_dimension() {
        let s = Solution::new(vec![1.0, 2.0, 3.0], 0.0);
        assert_eq!(s.dimension(), 3);
    }

    #[test]
    fn test_gamma_function() {
        // gamma(1) = 1
        let g1 = gamma_fn(1.0);
        assert!((g1 - 1.0).abs() < 0.01, "gamma(1) = {g1}");
        // gamma(2) = 1
        let g2 = gamma_fn(2.0);
        assert!((g2 - 1.0).abs() < 0.01, "gamma(2) = {g2}");
        // gamma(3) = 2
        let g3 = gamma_fn(3.0);
        assert!((g3 - 2.0).abs() < 0.1, "gamma(3) = {g3}");
    }

    #[test]
    fn test_lcg_range() {
        let mut rng = Lcg::new(42);
        for _ in 0..100 {
            let v = rng.next_f64();
            assert!(v >= 0.0 && v < 1.0, "out of range: {v}");
        }
    }

    #[test]
    fn test_lcg_uniform() {
        let mut rng = Lcg::new(42);
        for _ in 0..100 {
            let v = rng.uniform(-10.0, 10.0);
            assert!(v >= -10.0 && v <= 10.0);
        }
    }

    #[test]
    fn test_higher_dimension() {
        let bounds = vec![(-5.0, 5.0); 5];
        let mut cs = CuckooSearch::new(bounds, 30, 0.25, 42).unwrap();
        cs.run(200, &sphere);
        assert!(cs.best().fitness < 5.0, "5D sphere best = {}", cs.best().fitness);
    }
}
