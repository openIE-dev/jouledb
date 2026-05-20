//! Particle Swarm Optimization with configurable topology, inertia weight
//! strategies, velocity clamping, position bounds, and convergence tracking.

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
}

// ── Topology ─────────────────────────────────────────────────────

/// Neighborhood topology for information sharing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Topology {
    /// Global best — all particles share the same global best.
    Global,
    /// Ring topology — each particle communicates with k nearest neighbors.
    Ring(usize),
    /// Von Neumann grid topology.
    VonNeumann,
}

// ── Inertia weight strategy ──────────────────────────────────────

/// Inertia weight schedule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InertiaWeight {
    /// Constant inertia weight.
    Constant(f64),
    /// Linearly decreasing from w_max to w_min over max_iter iterations.
    LinearDecrease { w_max: f64, w_min: f64 },
    /// Adaptive based on improvement ratio.
    Adaptive { w_max: f64, w_min: f64 },
}

// ── Particle ─────────────────────────────────────────────────────

/// A single particle in the swarm.
#[derive(Debug, Clone, PartialEq)]
pub struct Particle {
    pub position: Vec<f64>,
    pub velocity: Vec<f64>,
    pub personal_best_position: Vec<f64>,
    pub personal_best_fitness: f64,
    pub fitness: f64,
}

impl Particle {
    fn new(position: Vec<f64>, dimensions: usize) -> Self {
        Self {
            personal_best_position: position.clone(),
            personal_best_fitness: f64::NEG_INFINITY,
            fitness: f64::NEG_INFINITY,
            velocity: vec![0.0; dimensions],
            position,
        }
    }
}

// ── Configuration ────────────────────────────────────────────────

/// PSO configuration.
#[derive(Debug, Clone)]
pub struct PsoConfig {
    pub swarm_size: usize,
    pub dimensions: usize,
    pub c1: f64,
    pub c2: f64,
    pub inertia: InertiaWeight,
    pub topology: Topology,
    pub max_iterations: usize,
    pub position_bounds: (f64, f64),
    pub velocity_clamp: f64,
    pub convergence_threshold: f64,
    pub convergence_window: usize,
    pub seed: u64,
}

impl Default for PsoConfig {
    fn default() -> Self {
        Self {
            swarm_size: 30,
            dimensions: 5,
            c1: 2.0,
            c2: 2.0,
            inertia: InertiaWeight::LinearDecrease { w_max: 0.9, w_min: 0.4 },
            topology: Topology::Global,
            max_iterations: 200,
            position_bounds: (-10.0, 10.0),
            velocity_clamp: 5.0,
            convergence_threshold: 1e-6,
            convergence_window: 15,
            seed: 42,
        }
    }
}

// ── Result ───────────────────────────────────────────────────────

/// Result of a PSO run.
#[derive(Debug, Clone, PartialEq)]
pub struct PsoResult {
    pub best_position: Vec<f64>,
    pub best_fitness: f64,
    pub iterations_run: usize,
    pub converged: bool,
    pub fitness_history: Vec<f64>,
}

// ── Engine ───────────────────────────────────────────────────────

/// Particle Swarm Optimization engine.
pub struct ParticleSwarm {
    config: PsoConfig,
    particles: Vec<Particle>,
    global_best_position: Vec<f64>,
    global_best_fitness: f64,
    iteration: usize,
    fitness_history: Vec<f64>,
    rng: Rng,
}

impl ParticleSwarm {
    pub fn new(config: PsoConfig) -> Self {
        let dims = config.dimensions;
        Self {
            rng: Rng::new(config.seed),
            particles: Vec::with_capacity(config.swarm_size),
            global_best_position: vec![0.0; dims],
            global_best_fitness: f64::NEG_INFINITY,
            iteration: 0,
            fitness_history: Vec::new(),
            config,
        }
    }

    /// Initialize swarm with random positions within bounds.
    pub fn initialize(&mut self) {
        let (lo, hi) = self.config.position_bounds;
        let dims = self.config.dimensions;
        self.particles.clear();
        self.global_best_fitness = f64::NEG_INFINITY;

        for _ in 0..self.config.swarm_size {
            let pos: Vec<f64> = (0..dims).map(|_| lo + self.rng.next_f64() * (hi - lo)).collect();
            let vel: Vec<f64> = (0..dims).map(|_| {
                let range = hi - lo;
                (self.rng.next_f64() - 0.5) * range * 0.1
            }).collect();
            let mut p = Particle::new(pos, dims);
            p.velocity = vel;
            self.particles.push(p);
        }
    }

    /// Evaluate all particles using the given objective function.
    pub fn evaluate<F: Fn(&[f64]) -> f64>(&mut self, obj_fn: &F) {
        for p in &mut self.particles {
            p.fitness = obj_fn(&p.position);
            if p.fitness > p.personal_best_fitness {
                p.personal_best_fitness = p.fitness;
                p.personal_best_position = p.position.clone();
            }
        }

        // Update global best
        for p in &self.particles {
            if p.personal_best_fitness > self.global_best_fitness {
                self.global_best_fitness = p.personal_best_fitness;
                self.global_best_position = p.personal_best_position.clone();
            }
        }
    }

    fn current_inertia(&self) -> f64 {
        match self.config.inertia {
            InertiaWeight::Constant(w) => w,
            InertiaWeight::LinearDecrease { w_max, w_min } => {
                let frac = self.iteration as f64 / self.config.max_iterations as f64;
                w_max - (w_max - w_min) * frac
            }
            InertiaWeight::Adaptive { w_max, w_min } => {
                // Use improvement ratio
                if self.fitness_history.len() < 2 {
                    return w_max;
                }
                let prev = self.fitness_history[self.fitness_history.len() - 2];
                let curr = *self.fitness_history.last().unwrap();
                let improvement = (curr - prev).abs() / (prev.abs() + 1e-15);
                if improvement < 0.01 {
                    w_min // Low improvement → explore less
                } else {
                    w_max
                }
            }
        }
    }

    /// Get the neighborhood best for a given particle index.
    fn neighborhood_best(&self, idx: usize) -> &[f64] {
        match self.config.topology {
            Topology::Global => &self.global_best_position,
            Topology::Ring(k) => {
                let n = self.particles.len();
                let half_k = k / 2;
                let mut best_idx = idx;
                let mut best_fit = self.particles[idx].personal_best_fitness;
                for offset in 1..=half_k {
                    let left = (idx + n - offset) % n;
                    let right = (idx + offset) % n;
                    if self.particles[left].personal_best_fitness > best_fit {
                        best_fit = self.particles[left].personal_best_fitness;
                        best_idx = left;
                    }
                    if self.particles[right].personal_best_fitness > best_fit {
                        best_fit = self.particles[right].personal_best_fitness;
                        best_idx = right;
                    }
                }
                &self.particles[best_idx].personal_best_position
            }
            Topology::VonNeumann => {
                let n = self.particles.len();
                let side = (n as f64).sqrt().ceil() as usize;
                let row = idx / side;
                let col = idx % side;
                let neighbors = [
                    ((row + side - 1) % side) * side + col,
                    ((row + 1) % side) * side + col,
                    row * side + (col + side - 1) % side,
                    row * side + (col + 1) % side,
                ];
                let mut best_idx = idx;
                let mut best_fit = self.particles[idx].personal_best_fitness;
                for &ni in &neighbors {
                    if ni < n && self.particles[ni].personal_best_fitness > best_fit {
                        best_fit = self.particles[ni].personal_best_fitness;
                        best_idx = ni;
                    }
                }
                &self.particles[best_idx].personal_best_position
            }
        }
    }

    /// Run one iteration of PSO.
    pub fn step<F: Fn(&[f64]) -> f64>(&mut self, obj_fn: &F) {
        let w = self.current_inertia();
        let (lo, hi) = self.config.position_bounds;
        let v_max = self.config.velocity_clamp;
        let c1 = self.config.c1;
        let c2 = self.config.c2;
        let n = self.particles.len();
        let dims = self.config.dimensions;

        // Collect neighborhood bests first (to avoid borrow issues)
        let nbest: Vec<Vec<f64>> = (0..n)
            .map(|i| self.neighborhood_best(i).to_vec())
            .collect();

        for i in 0..n {
            let r1 = self.rng.next_f64();
            let r2 = self.rng.next_f64();

            for d in 0..dims {
                let cognitive = c1 * r1 * (self.particles[i].personal_best_position[d] - self.particles[i].position[d]);
                let social = c2 * r2 * (nbest[i][d] - self.particles[i].position[d]);
                self.particles[i].velocity[d] = w * self.particles[i].velocity[d] + cognitive + social;

                // Velocity clamping
                self.particles[i].velocity[d] = self.particles[i].velocity[d].clamp(-v_max, v_max);

                // Position update
                self.particles[i].position[d] += self.particles[i].velocity[d];

                // Bounds enforcement (reflect)
                if self.particles[i].position[d] < lo {
                    self.particles[i].position[d] = lo;
                    self.particles[i].velocity[d] = -self.particles[i].velocity[d] * 0.5;
                } else if self.particles[i].position[d] > hi {
                    self.particles[i].position[d] = hi;
                    self.particles[i].velocity[d] = -self.particles[i].velocity[d] * 0.5;
                }
            }
        }

        self.evaluate(obj_fn);
        self.fitness_history.push(self.global_best_fitness);
        self.iteration += 1;
    }

    /// Check convergence based on recent fitness improvement.
    pub fn has_converged(&self) -> bool {
        let w = self.config.convergence_window;
        if self.fitness_history.len() < w {
            return false;
        }
        let recent = &self.fitness_history[self.fitness_history.len() - w..];
        let min_r = recent.iter().copied().fold(f64::INFINITY, f64::min);
        let max_r = recent.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (max_r - min_r).abs() < self.config.convergence_threshold
    }

    /// Run the full PSO optimization.
    pub fn run<F: Fn(&[f64]) -> f64>(&mut self, obj_fn: &F) -> PsoResult {
        self.evaluate(obj_fn);
        self.fitness_history.push(self.global_best_fitness);

        for _ in 0..self.config.max_iterations {
            self.step(obj_fn);
            if self.has_converged() {
                return PsoResult {
                    best_position: self.global_best_position.clone(),
                    best_fitness: self.global_best_fitness,
                    iterations_run: self.iteration,
                    converged: true,
                    fitness_history: self.fitness_history.clone(),
                };
            }
        }

        PsoResult {
            best_position: self.global_best_position.clone(),
            best_fitness: self.global_best_fitness,
            iterations_run: self.iteration,
            converged: false,
            fitness_history: self.fitness_history.clone(),
        }
    }

    /// Get current global best fitness.
    pub fn global_best_fitness(&self) -> f64 {
        self.global_best_fitness
    }

    /// Get current global best position.
    pub fn global_best_position(&self) -> &[f64] {
        &self.global_best_position
    }

    /// Get current iteration.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Get the particles.
    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    /// Compute swarm diversity (average distance to centroid).
    pub fn diversity(&self) -> f64 {
        let n = self.particles.len() as f64;
        let dims = self.config.dimensions;
        let mut centroid = vec![0.0; dims];
        for p in &self.particles {
            for d in 0..dims {
                centroid[d] += p.position[d];
            }
        }
        for c in &mut centroid {
            *c /= n;
        }

        let avg_dist: f64 = self.particles.iter().map(|p| {
            p.position.iter().zip(centroid.iter())
                .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt()
        }).sum::<f64>() / n;

        avg_dist
    }

    /// Average velocity magnitude.
    pub fn average_velocity(&self) -> f64 {
        let n = self.particles.len() as f64;
        self.particles.iter().map(|p| {
            p.velocity.iter().map(|v| v * v).sum::<f64>().sqrt()
        }).sum::<f64>() / n
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Negative sphere: maximum at origin = 0.
    fn neg_sphere(x: &[f64]) -> f64 {
        -x.iter().map(|v| v * v).sum::<f64>()
    }

    /// Rastrigin-like (inverted, max at origin).
    fn neg_rastrigin(x: &[f64]) -> f64 {
        let n = x.len() as f64;
        let sum: f64 = x.iter().map(|xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos()).sum();
        -(10.0 * n + sum)
    }

    #[test]
    fn test_particle_creation() {
        let p = Particle::new(vec![1.0, 2.0, 3.0], 3);
        assert_eq!(p.position, vec![1.0, 2.0, 3.0]);
        assert_eq!(p.velocity, vec![0.0, 0.0, 0.0]);
        assert_eq!(p.personal_best_fitness, f64::NEG_INFINITY);
    }

    #[test]
    fn test_default_config() {
        let c = PsoConfig::default();
        assert_eq!(c.swarm_size, 30);
        assert_eq!(c.dimensions, 5);
        assert!((c.c1 - 2.0).abs() < 1e-10);
        assert!((c.c2 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_initialize_swarm() {
        let config = PsoConfig { swarm_size: 20, dimensions: 3, ..Default::default() };
        let mut pso = ParticleSwarm::new(config);
        pso.initialize();
        assert_eq!(pso.particles().len(), 20);
        let (lo, hi) = pso.config.position_bounds;
        for p in pso.particles() {
            for &v in &p.position {
                assert!(v >= lo && v <= hi);
            }
        }
    }

    #[test]
    fn test_evaluate_updates_pbest() {
        let config = PsoConfig { swarm_size: 10, dimensions: 2, ..Default::default() };
        let mut pso = ParticleSwarm::new(config);
        pso.initialize();
        pso.evaluate(&neg_sphere);
        for p in pso.particles() {
            assert!(p.personal_best_fitness > f64::NEG_INFINITY);
        }
    }

    #[test]
    fn test_evaluate_updates_gbest() {
        let config = PsoConfig { swarm_size: 10, dimensions: 2, ..Default::default() };
        let mut pso = ParticleSwarm::new(config);
        pso.initialize();
        pso.evaluate(&neg_sphere);
        assert!(pso.global_best_fitness() > f64::NEG_INFINITY);
    }

    #[test]
    fn test_step_increases_iteration() {
        let config = PsoConfig { swarm_size: 10, dimensions: 2, ..Default::default() };
        let mut pso = ParticleSwarm::new(config);
        pso.initialize();
        pso.evaluate(&neg_sphere);
        assert_eq!(pso.iteration(), 0);
        pso.step(&neg_sphere);
        assert_eq!(pso.iteration(), 1);
    }

    #[test]
    fn test_monotonic_gbest() {
        let config = PsoConfig { swarm_size: 20, dimensions: 3, max_iterations: 30, ..Default::default() };
        let mut pso = ParticleSwarm::new(config);
        pso.initialize();
        pso.evaluate(&neg_sphere);
        let mut prev = pso.global_best_fitness();
        for _ in 0..30 {
            pso.step(&neg_sphere);
            assert!(pso.global_best_fitness() >= prev - 1e-15);
            prev = pso.global_best_fitness();
        }
    }

    #[test]
    fn test_sphere_optimization() {
        let config = PsoConfig {
            swarm_size: 30,
            dimensions: 3,
            max_iterations: 100,
            position_bounds: (-5.0, 5.0),
            ..Default::default()
        };
        let mut pso = ParticleSwarm::new(config);
        pso.initialize();
        let result = pso.run(&neg_sphere);
        assert!(result.best_fitness > -1.0, "Expected near-zero, got {}", result.best_fitness);
    }

    #[test]
    fn test_velocity_clamping() {
        let config = PsoConfig {
            swarm_size: 10,
            dimensions: 2,
            velocity_clamp: 2.0,
            ..Default::default()
        };
        let mut pso = ParticleSwarm::new(config);
        pso.initialize();
        pso.evaluate(&neg_sphere);
        pso.step(&neg_sphere);
        for p in pso.particles() {
            for &v in &p.velocity {
                assert!(v.abs() <= 2.0 + 1e-10);
            }
        }
    }

    #[test]
    fn test_position_bounds() {
        let config = PsoConfig {
            swarm_size: 10,
            dimensions: 2,
            position_bounds: (-3.0, 3.0),
            max_iterations: 20,
            ..Default::default()
        };
        let mut pso = ParticleSwarm::new(config);
        pso.initialize();
        for _ in 0..20 {
            pso.step(&neg_sphere);
        }
        for p in pso.particles() {
            for &x in &p.position {
                assert!(x >= -3.0 - 1e-10 && x <= 3.0 + 1e-10);
            }
        }
    }

    #[test]
    fn test_ring_topology() {
        let config = PsoConfig {
            swarm_size: 20,
            dimensions: 3,
            topology: Topology::Ring(4),
            max_iterations: 50,
            ..Default::default()
        };
        let mut pso = ParticleSwarm::new(config);
        pso.initialize();
        let result = pso.run(&neg_sphere);
        assert!(result.best_fitness > -50.0);
    }

    #[test]
    fn test_von_neumann_topology() {
        let config = PsoConfig {
            swarm_size: 25,
            dimensions: 3,
            topology: Topology::VonNeumann,
            max_iterations: 50,
            ..Default::default()
        };
        let mut pso = ParticleSwarm::new(config);
        pso.initialize();
        let result = pso.run(&neg_sphere);
        assert!(result.best_fitness > -50.0);
    }

    #[test]
    fn test_constant_inertia() {
        let config = PsoConfig {
            swarm_size: 15,
            dimensions: 2,
            inertia: InertiaWeight::Constant(0.7),
            max_iterations: 30,
            ..Default::default()
        };
        let mut pso = ParticleSwarm::new(config);
        pso.initialize();
        let result = pso.run(&neg_sphere);
        assert!(result.iterations_run > 0);
    }

    #[test]
    fn test_adaptive_inertia() {
        let config = PsoConfig {
            swarm_size: 15,
            dimensions: 2,
            inertia: InertiaWeight::Adaptive { w_max: 0.9, w_min: 0.4 },
            max_iterations: 30,
            ..Default::default()
        };
        let mut pso = ParticleSwarm::new(config);
        pso.initialize();
        let result = pso.run(&neg_sphere);
        assert!(result.iterations_run > 0);
    }

    #[test]
    fn test_convergence_detection() {
        let config = PsoConfig {
            convergence_window: 3,
            convergence_threshold: 0.1,
            ..Default::default()
        };
        let mut pso = ParticleSwarm::new(config);
        pso.fitness_history = vec![5.0, 5.0, 5.0];
        assert!(pso.has_converged());
    }

    #[test]
    fn test_no_convergence_early() {
        let config = PsoConfig {
            convergence_window: 5,
            ..Default::default()
        };
        let mut pso = ParticleSwarm::new(config);
        pso.fitness_history = vec![1.0, 2.0];
        assert!(!pso.has_converged());
    }

    #[test]
    fn test_diversity_positive() {
        let config = PsoConfig { swarm_size: 10, dimensions: 3, ..Default::default() };
        let mut pso = ParticleSwarm::new(config);
        pso.initialize();
        assert!(pso.diversity() > 0.0);
    }

    #[test]
    fn test_average_velocity() {
        let config = PsoConfig { swarm_size: 10, dimensions: 3, ..Default::default() };
        let mut pso = ParticleSwarm::new(config);
        pso.initialize();
        pso.evaluate(&neg_sphere);
        pso.step(&neg_sphere);
        assert!(pso.average_velocity() >= 0.0);
    }

    #[test]
    fn test_fitness_history_grows() {
        let config = PsoConfig { swarm_size: 10, dimensions: 2, ..Default::default() };
        let mut pso = ParticleSwarm::new(config);
        pso.initialize();
        pso.evaluate(&neg_sphere);
        pso.step(&neg_sphere);
        pso.step(&neg_sphere);
        pso.step(&neg_sphere);
        assert!(pso.fitness_history.len() >= 3);
    }

    #[test]
    fn test_result_has_correct_dimensions() {
        let config = PsoConfig { swarm_size: 10, dimensions: 4, max_iterations: 10, ..Default::default() };
        let mut pso = ParticleSwarm::new(config);
        pso.initialize();
        let result = pso.run(&neg_sphere);
        assert_eq!(result.best_position.len(), 4);
    }

    #[test]
    fn test_rastrigin_improvement() {
        let config = PsoConfig {
            swarm_size: 30,
            dimensions: 2,
            max_iterations: 50,
            position_bounds: (-5.0, 5.0),
            ..Default::default()
        };
        let mut pso = ParticleSwarm::new(config);
        pso.initialize();
        pso.evaluate(&neg_rastrigin);
        let start = pso.global_best_fitness();
        for _ in 0..50 {
            pso.step(&neg_rastrigin);
        }
        assert!(pso.global_best_fitness() >= start - 1e-10);
    }

    #[test]
    fn test_run_returns_result() {
        let config = PsoConfig {
            swarm_size: 15,
            dimensions: 2,
            max_iterations: 20,
            ..Default::default()
        };
        let mut pso = ParticleSwarm::new(config);
        pso.initialize();
        let result = pso.run(&neg_sphere);
        assert!(result.iterations_run <= 20);
        assert!(!result.fitness_history.is_empty());
    }
}
