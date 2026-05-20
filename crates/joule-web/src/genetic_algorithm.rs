//! Genetic algorithm framework with configurable selection, crossover, mutation,
//! elitism, generational and steady-state models, and convergence detection.

// ── Simple deterministic PRNG ────────────────────────────────────

/// Xorshift64 PRNG for reproducible runs without external deps.
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

    /// Uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform usize in [0, bound).
    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    /// Gaussian via Box-Muller.
    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ── Encoding ─────────────────────────────────────────────────────

/// Chromosome representation.
#[derive(Debug, Clone, PartialEq)]
pub enum Chromosome {
    /// Real-valued genes.
    RealValued(Vec<f64>),
    /// Binary (bit-vector) genes.
    Binary(Vec<bool>),
}

impl Chromosome {
    pub fn len(&self) -> usize {
        match self {
            Self::RealValued(v) => v.len(),
            Self::Binary(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ── Individual ───────────────────────────────────────────────────

/// A single individual in the population.
#[derive(Debug, Clone, PartialEq)]
pub struct Individual {
    pub chromosome: Chromosome,
    pub fitness: f64,
}

impl Individual {
    pub fn new(chromosome: Chromosome) -> Self {
        Self { chromosome, fitness: f64::NEG_INFINITY }
    }
}

// ── Selection methods ────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Selection {
    /// Tournament selection with given tournament size.
    Tournament(usize),
    /// Roulette wheel (fitness proportionate).
    RouletteWheel,
    /// Rank-based selection.
    RankBased,
}

// ── Crossover methods ────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Crossover {
    SinglePoint,
    TwoPoint,
    /// Uniform crossover with mixing ratio.
    Uniform(f64),
    /// BLX-α blend crossover for real-valued chromosomes.
    BlendAlpha(f64),
}

// ── Mutation methods ─────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mutation {
    /// Bit-flip for binary chromosomes.
    BitFlip,
    /// Gaussian perturbation with given std deviation.
    Gaussian(f64),
    /// Uniform random within bounds.
    UniformRandom,
}

// ── Generational model ──────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GenerationalModel {
    /// Replace entire population each generation.
    Generational,
    /// Replace only a few individuals each iteration.
    SteadyState(usize),
}

// ── Configuration ────────────────────────────────────────────────

/// Genetic algorithm parameters.
#[derive(Debug, Clone)]
pub struct GaConfig {
    pub population_size: usize,
    pub chromosome_length: usize,
    pub selection: Selection,
    pub crossover: Crossover,
    pub mutation: Mutation,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub elitism_count: usize,
    pub model: GenerationalModel,
    pub max_generations: usize,
    pub convergence_threshold: f64,
    pub convergence_generations: usize,
    /// Bounds for real-valued genes (min, max).
    pub bounds: (f64, f64),
    pub seed: u64,
}

impl Default for GaConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            chromosome_length: 10,
            selection: Selection::Tournament(3),
            crossover: Crossover::SinglePoint,
            mutation: Mutation::Gaussian(0.1),
            mutation_rate: 0.05,
            crossover_rate: 0.8,
            elitism_count: 2,
            model: GenerationalModel::Generational,
            max_generations: 200,
            convergence_threshold: 1e-6,
            convergence_generations: 20,
            bounds: (-10.0, 10.0),
            seed: 42,
        }
    }
}

// ── Result ───────────────────────────────────────────────────────

/// Result of a GA run.
#[derive(Debug, Clone, PartialEq)]
pub struct GaResult {
    pub best_individual: Individual,
    pub generations_run: usize,
    pub converged: bool,
    pub fitness_history: Vec<f64>,
}

// ── Engine ───────────────────────────────────────────────────────

/// Genetic algorithm engine.
pub struct GeneticAlgorithm {
    config: GaConfig,
    population: Vec<Individual>,
    rng: Rng,
    generation: usize,
    fitness_history: Vec<f64>,
}

impl GeneticAlgorithm {
    pub fn new(config: GaConfig) -> Self {
        let rng = Rng::new(config.seed);
        Self {
            population: Vec::with_capacity(config.population_size),
            rng,
            generation: 0,
            fitness_history: Vec::new(),
            config,
        }
    }

    /// Initialize population with random real-valued chromosomes.
    pub fn initialize_real(&mut self) {
        self.population.clear();
        let (lo, hi) = self.config.bounds;
        for _ in 0..self.config.population_size {
            let genes: Vec<f64> = (0..self.config.chromosome_length)
                .map(|_| lo + self.rng.next_f64() * (hi - lo))
                .collect();
            self.population.push(Individual::new(Chromosome::RealValued(genes)));
        }
    }

    /// Initialize population with random binary chromosomes.
    pub fn initialize_binary(&mut self) {
        self.population.clear();
        for _ in 0..self.config.population_size {
            let bits: Vec<bool> = (0..self.config.chromosome_length)
                .map(|_| self.rng.next_f64() < 0.5)
                .collect();
            self.population.push(Individual::new(Chromosome::Binary(bits)));
        }
    }

    /// Evaluate all individuals using the given fitness function.
    pub fn evaluate<F: Fn(&Chromosome) -> f64>(&mut self, fitness_fn: &F) {
        for ind in &mut self.population {
            ind.fitness = fitness_fn(&ind.chromosome);
        }
    }

    /// Get the best individual in the current population.
    pub fn best(&self) -> Option<&Individual> {
        self.population.iter().max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get current generation number.
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Get population snapshot.
    pub fn population(&self) -> &[Individual] {
        &self.population
    }

    // ── Selection ────────────────────────────────────────────

    fn select_one(&mut self) -> Individual {
        match self.config.selection {
            Selection::Tournament(k) => self.tournament_select(k),
            Selection::RouletteWheel => self.roulette_select(),
            Selection::RankBased => self.rank_select(),
        }
    }

    fn tournament_select(&mut self, k: usize) -> Individual {
        let mut best: Option<&Individual> = None;
        for _ in 0..k {
            let idx = self.rng.next_usize(self.population.len());
            let candidate = &self.population[idx];
            if best.is_none() || candidate.fitness > best.unwrap().fitness {
                best = Some(candidate);
            }
        }
        best.unwrap().clone()
    }

    fn roulette_select(&mut self) -> Individual {
        let min_fit = self.population.iter().map(|i| i.fitness).fold(f64::INFINITY, f64::min);
        let shifted: Vec<f64> = self.population.iter().map(|i| i.fitness - min_fit + 1e-10).collect();
        let total: f64 = shifted.iter().sum();
        let mut spin = self.rng.next_f64() * total;
        for (i, s) in shifted.iter().enumerate() {
            spin -= s;
            if spin <= 0.0 {
                return self.population[i].clone();
            }
        }
        self.population.last().unwrap().clone()
    }

    fn rank_select(&mut self) -> Individual {
        let n = self.population.len();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            self.population[a].fitness.partial_cmp(&self.population[b].fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let total_rank: usize = n * (n + 1) / 2;
        let mut spin = self.rng.next_usize(total_rank);
        for (rank, &idx) in indices.iter().enumerate() {
            let r = rank + 1;
            if spin < r {
                return self.population[idx].clone();
            }
            spin -= r;
        }
        self.population.last().unwrap().clone()
    }

    // ── Crossover ────────────────────────────────────────────

    fn crossover(&mut self, p1: &Individual, p2: &Individual) -> (Individual, Individual) {
        if self.rng.next_f64() > self.config.crossover_rate {
            return (p1.clone(), p2.clone());
        }
        match (&p1.chromosome, &p2.chromosome) {
            (Chromosome::RealValued(g1), Chromosome::RealValued(g2)) => {
                let (c1, c2) = self.crossover_real(g1, g2);
                (Individual::new(Chromosome::RealValued(c1)), Individual::new(Chromosome::RealValued(c2)))
            }
            (Chromosome::Binary(g1), Chromosome::Binary(g2)) => {
                let (c1, c2) = self.crossover_binary(g1, g2);
                (Individual::new(Chromosome::Binary(c1)), Individual::new(Chromosome::Binary(c2)))
            }
            _ => (p1.clone(), p2.clone()),
        }
    }

    fn crossover_real(&mut self, g1: &[f64], g2: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = g1.len();
        match self.config.crossover {
            Crossover::SinglePoint => {
                let pt = self.rng.next_usize(n.max(1));
                let c1: Vec<f64> = g1[..pt].iter().chain(g2[pt..].iter()).copied().collect();
                let c2: Vec<f64> = g2[..pt].iter().chain(g1[pt..].iter()).copied().collect();
                (c1, c2)
            }
            Crossover::TwoPoint => {
                let mut a = self.rng.next_usize(n.max(1));
                let mut b = self.rng.next_usize(n.max(1));
                if a > b { std::mem::swap(&mut a, &mut b); }
                let mut c1 = g1.to_vec();
                let mut c2 = g2.to_vec();
                for i in a..b {
                    c1[i] = g2[i];
                    c2[i] = g1[i];
                }
                (c1, c2)
            }
            Crossover::Uniform(ratio) => {
                let mut c1 = Vec::with_capacity(n);
                let mut c2 = Vec::with_capacity(n);
                for i in 0..n {
                    if self.rng.next_f64() < ratio {
                        c1.push(g2[i]);
                        c2.push(g1[i]);
                    } else {
                        c1.push(g1[i]);
                        c2.push(g2[i]);
                    }
                }
                (c1, c2)
            }
            Crossover::BlendAlpha(alpha) => {
                let mut c1 = Vec::with_capacity(n);
                let mut c2 = Vec::with_capacity(n);
                for i in 0..n {
                    let lo = g1[i].min(g2[i]);
                    let hi = g1[i].max(g2[i]);
                    let d = hi - lo;
                    let low = lo - alpha * d;
                    let high = hi + alpha * d;
                    c1.push(low + self.rng.next_f64() * (high - low));
                    c2.push(low + self.rng.next_f64() * (high - low));
                }
                (c1, c2)
            }
        }
    }

    fn crossover_binary(&mut self, g1: &[bool], g2: &[bool]) -> (Vec<bool>, Vec<bool>) {
        let n = g1.len();
        match self.config.crossover {
            Crossover::SinglePoint | Crossover::BlendAlpha(_) => {
                let pt = self.rng.next_usize(n.max(1));
                let c1: Vec<bool> = g1[..pt].iter().chain(g2[pt..].iter()).copied().collect();
                let c2: Vec<bool> = g2[..pt].iter().chain(g1[pt..].iter()).copied().collect();
                (c1, c2)
            }
            Crossover::TwoPoint => {
                let mut a = self.rng.next_usize(n.max(1));
                let mut b = self.rng.next_usize(n.max(1));
                if a > b { std::mem::swap(&mut a, &mut b); }
                let mut c1 = g1.to_vec();
                let mut c2 = g2.to_vec();
                for i in a..b {
                    c1[i] = g2[i];
                    c2[i] = g1[i];
                }
                (c1, c2)
            }
            Crossover::Uniform(ratio) => {
                let mut c1 = Vec::with_capacity(n);
                let mut c2 = Vec::with_capacity(n);
                for i in 0..n {
                    if self.rng.next_f64() < ratio {
                        c1.push(g2[i]);
                        c2.push(g1[i]);
                    } else {
                        c1.push(g1[i]);
                        c2.push(g2[i]);
                    }
                }
                (c1, c2)
            }
        }
    }

    // ── Mutation ─────────────────────────────────────────────

    fn mutate(&mut self, ind: &mut Individual) {
        match &mut ind.chromosome {
            Chromosome::RealValued(genes) => self.mutate_real(genes),
            Chromosome::Binary(bits) => self.mutate_binary(bits),
        }
    }

    fn mutate_real(&mut self, genes: &mut Vec<f64>) {
        let (lo, hi) = self.config.bounds;
        for g in genes.iter_mut() {
            if self.rng.next_f64() < self.config.mutation_rate {
                match self.config.mutation {
                    Mutation::Gaussian(sigma) => {
                        *g += self.rng.next_gaussian() * sigma;
                        *g = g.clamp(lo, hi);
                    }
                    Mutation::UniformRandom => {
                        *g = lo + self.rng.next_f64() * (hi - lo);
                    }
                    Mutation::BitFlip => {
                        // For real-valued, use sign flip as analogy
                        *g = -*g;
                        *g = g.clamp(lo, hi);
                    }
                }
            }
        }
    }

    fn mutate_binary(&mut self, bits: &mut Vec<bool>) {
        for b in bits.iter_mut() {
            if self.rng.next_f64() < self.config.mutation_rate {
                *b = !*b;
            }
        }
    }

    // ── Evolution step ───────────────────────────────────────

    /// Run one generation of the GA.
    pub fn step<F: Fn(&Chromosome) -> f64>(&mut self, fitness_fn: &F) {
        // Sort for elitism
        self.population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));

        match self.config.model {
            GenerationalModel::Generational => self.step_generational(fitness_fn),
            GenerationalModel::SteadyState(replacements) => self.step_steady_state(fitness_fn, replacements),
        }

        // Track best fitness
        if let Some(best) = self.best() {
            self.fitness_history.push(best.fitness);
        }
        self.generation += 1;
    }

    fn step_generational<F: Fn(&Chromosome) -> f64>(&mut self, fitness_fn: &F) {
        let mut new_pop: Vec<Individual> = Vec::with_capacity(self.config.population_size);

        // Elitism
        let elite_count = self.config.elitism_count.min(self.population.len());
        for i in 0..elite_count {
            new_pop.push(self.population[i].clone());
        }

        // Fill rest via selection + crossover + mutation
        while new_pop.len() < self.config.population_size {
            let p1 = self.select_one();
            let p2 = self.select_one();
            let (mut c1, mut c2) = self.crossover(&p1, &p2);
            self.mutate(&mut c1);
            self.mutate(&mut c2);
            c1.fitness = fitness_fn(&c1.chromosome);
            new_pop.push(c1);
            if new_pop.len() < self.config.population_size {
                c2.fitness = fitness_fn(&c2.chromosome);
                new_pop.push(c2);
            }
        }

        self.population = new_pop;
    }

    fn step_steady_state<F: Fn(&Chromosome) -> f64>(&mut self, fitness_fn: &F, replacements: usize) {
        for _ in 0..replacements {
            let p1 = self.select_one();
            let p2 = self.select_one();
            let (mut c1, _c2) = self.crossover(&p1, &p2);
            self.mutate(&mut c1);
            c1.fitness = fitness_fn(&c1.chromosome);

            // Replace worst
            if let Some(worst_idx) = self.population.iter().enumerate()
                .min_by(|(_, a), (_, b)| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
            {
                if c1.fitness > self.population[worst_idx].fitness {
                    self.population[worst_idx] = c1;
                }
            }
        }
    }

    /// Check if the GA has converged.
    pub fn has_converged(&self) -> bool {
        let cg = self.config.convergence_generations;
        if self.fitness_history.len() < cg {
            return false;
        }
        let recent = &self.fitness_history[self.fitness_history.len() - cg..];
        let min_r = recent.iter().copied().fold(f64::INFINITY, f64::min);
        let max_r = recent.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (max_r - min_r).abs() < self.config.convergence_threshold
    }

    /// Run the full GA optimization.
    pub fn run<F: Fn(&Chromosome) -> f64>(&mut self, fitness_fn: &F) -> GaResult {
        self.evaluate(fitness_fn);
        if let Some(best) = self.best() {
            self.fitness_history.push(best.fitness);
        }

        for _ in 0..self.config.max_generations {
            self.step(fitness_fn);
            if self.has_converged() {
                let best = self.best().unwrap().clone();
                return GaResult {
                    best_individual: best,
                    generations_run: self.generation,
                    converged: true,
                    fitness_history: self.fitness_history.clone(),
                };
            }
        }

        let best = self.best().unwrap().clone();
        GaResult {
            best_individual: best,
            generations_run: self.generation,
            converged: false,
            fitness_history: self.fitness_history.clone(),
        }
    }

    /// Get diversity (average pairwise distance for real-valued populations).
    pub fn diversity(&self) -> f64 {
        let reals: Vec<&Vec<f64>> = self.population.iter().filter_map(|ind| {
            if let Chromosome::RealValued(g) = &ind.chromosome { Some(g) } else { None }
        }).collect();

        if reals.len() < 2 { return 0.0; }

        let mut total = 0.0;
        let mut count = 0usize;
        for i in 0..reals.len() {
            for j in (i + 1)..reals.len() {
                let dist: f64 = reals[i].iter().zip(reals[j].iter())
                    .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
                total += dist;
                count += 1;
            }
        }
        if count == 0 { 0.0 } else { total / count as f64 }
    }

    /// Population statistics.
    pub fn stats(&self) -> PopulationStats {
        let fits: Vec<f64> = self.population.iter().map(|i| i.fitness).collect();
        let n = fits.len() as f64;
        let mean = fits.iter().sum::<f64>() / n;
        let var = fits.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / n;
        let best = fits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let worst = fits.iter().copied().fold(f64::INFINITY, f64::min);
        PopulationStats { mean, variance: var, best, worst, size: fits.len() }
    }
}

/// Summary statistics for a population.
#[derive(Debug, Clone, PartialEq)]
pub struct PopulationStats {
    pub mean: f64,
    pub variance: f64,
    pub best: f64,
    pub worst: f64,
    pub size: usize,
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere_fitness(c: &Chromosome) -> f64 {
        match c {
            Chromosome::RealValued(genes) => {
                -genes.iter().map(|x| x * x).sum::<f64>()
            }
            _ => 0.0,
        }
    }

    fn onemax_fitness(c: &Chromosome) -> f64 {
        match c {
            Chromosome::Binary(bits) => bits.iter().filter(|&&b| b).count() as f64,
            _ => 0.0,
        }
    }

    #[test]
    fn test_individual_creation() {
        let ind = Individual::new(Chromosome::RealValued(vec![1.0, 2.0, 3.0]));
        assert_eq!(ind.fitness, f64::NEG_INFINITY);
        assert_eq!(ind.chromosome.len(), 3);
    }

    #[test]
    fn test_chromosome_len() {
        assert_eq!(Chromosome::RealValued(vec![1.0, 2.0]).len(), 2);
        assert_eq!(Chromosome::Binary(vec![true, false, true]).len(), 3);
        assert!(Chromosome::RealValued(vec![]).is_empty());
    }

    #[test]
    fn test_initialize_real() {
        let config = GaConfig { population_size: 20, chromosome_length: 5, ..Default::default() };
        let mut ga = GeneticAlgorithm::new(config);
        ga.initialize_real();
        assert_eq!(ga.population().len(), 20);
        for ind in ga.population() {
            assert_eq!(ind.chromosome.len(), 5);
        }
    }

    #[test]
    fn test_initialize_binary() {
        let config = GaConfig { population_size: 15, chromosome_length: 8, ..Default::default() };
        let mut ga = GeneticAlgorithm::new(config);
        ga.initialize_binary();
        assert_eq!(ga.population().len(), 15);
        for ind in ga.population() {
            if let Chromosome::Binary(bits) = &ind.chromosome {
                assert_eq!(bits.len(), 8);
            } else {
                panic!("Expected binary chromosome");
            }
        }
    }

    #[test]
    fn test_evaluate() {
        let config = GaConfig { population_size: 10, chromosome_length: 3, ..Default::default() };
        let mut ga = GeneticAlgorithm::new(config);
        ga.initialize_real();
        ga.evaluate(&sphere_fitness);
        for ind in ga.population() {
            assert!(ind.fitness <= 0.0);
        }
    }

    #[test]
    fn test_best_individual() {
        let config = GaConfig { population_size: 10, chromosome_length: 3, ..Default::default() };
        let mut ga = GeneticAlgorithm::new(config);
        ga.initialize_real();
        ga.evaluate(&sphere_fitness);
        let best = ga.best().unwrap();
        for ind in ga.population() {
            assert!(best.fitness >= ind.fitness - 1e-10);
        }
    }

    #[test]
    fn test_generational_step() {
        let config = GaConfig {
            population_size: 20,
            chromosome_length: 5,
            max_generations: 10,
            ..Default::default()
        };
        let mut ga = GeneticAlgorithm::new(config);
        ga.initialize_real();
        ga.evaluate(&sphere_fitness);
        let initial_best = ga.best().unwrap().fitness;
        for _ in 0..10 {
            ga.step(&sphere_fitness);
        }
        assert!(ga.best().unwrap().fitness >= initial_best - 1e-10);
    }

    #[test]
    fn test_steady_state() {
        let config = GaConfig {
            population_size: 20,
            chromosome_length: 5,
            model: GenerationalModel::SteadyState(5),
            max_generations: 10,
            ..Default::default()
        };
        let mut ga = GeneticAlgorithm::new(config);
        ga.initialize_real();
        ga.evaluate(&sphere_fitness);
        for _ in 0..10 {
            ga.step(&sphere_fitness);
        }
        assert_eq!(ga.population().len(), 20);
    }

    #[test]
    fn test_tournament_selection() {
        let config = GaConfig {
            population_size: 30,
            chromosome_length: 5,
            selection: Selection::Tournament(5),
            ..Default::default()
        };
        let mut ga = GeneticAlgorithm::new(config);
        ga.initialize_real();
        ga.evaluate(&sphere_fitness);
        ga.step(&sphere_fitness);
        assert_eq!(ga.population().len(), 30);
    }

    #[test]
    fn test_roulette_selection() {
        let config = GaConfig {
            population_size: 20,
            chromosome_length: 5,
            selection: Selection::RouletteWheel,
            ..Default::default()
        };
        let mut ga = GeneticAlgorithm::new(config);
        ga.initialize_real();
        ga.evaluate(&sphere_fitness);
        ga.step(&sphere_fitness);
        assert_eq!(ga.population().len(), 20);
    }

    #[test]
    fn test_rank_selection() {
        let config = GaConfig {
            population_size: 20,
            chromosome_length: 5,
            selection: Selection::RankBased,
            ..Default::default()
        };
        let mut ga = GeneticAlgorithm::new(config);
        ga.initialize_real();
        ga.evaluate(&sphere_fitness);
        ga.step(&sphere_fitness);
        assert_eq!(ga.population().len(), 20);
    }

    #[test]
    fn test_two_point_crossover() {
        let config = GaConfig {
            crossover: Crossover::TwoPoint,
            crossover_rate: 1.0,
            population_size: 20,
            chromosome_length: 10,
            ..Default::default()
        };
        let mut ga = GeneticAlgorithm::new(config);
        ga.initialize_real();
        ga.evaluate(&sphere_fitness);
        ga.step(&sphere_fitness);
        assert_eq!(ga.population().len(), 20);
    }

    #[test]
    fn test_uniform_crossover() {
        let config = GaConfig {
            crossover: Crossover::Uniform(0.5),
            crossover_rate: 1.0,
            population_size: 20,
            chromosome_length: 10,
            ..Default::default()
        };
        let mut ga = GeneticAlgorithm::new(config);
        ga.initialize_real();
        ga.evaluate(&sphere_fitness);
        ga.step(&sphere_fitness);
        assert_eq!(ga.population().len(), 20);
    }

    #[test]
    fn test_blend_crossover() {
        let config = GaConfig {
            crossover: Crossover::BlendAlpha(0.5),
            crossover_rate: 1.0,
            population_size: 20,
            chromosome_length: 10,
            ..Default::default()
        };
        let mut ga = GeneticAlgorithm::new(config);
        ga.initialize_real();
        ga.evaluate(&sphere_fitness);
        ga.step(&sphere_fitness);
        assert_eq!(ga.population().len(), 20);
    }

    #[test]
    fn test_binary_onemax() {
        let config = GaConfig {
            population_size: 30,
            chromosome_length: 20,
            mutation: Mutation::BitFlip,
            mutation_rate: 0.05,
            max_generations: 100,
            ..Default::default()
        };
        let mut ga = GeneticAlgorithm::new(config);
        ga.initialize_binary();
        ga.evaluate(&onemax_fitness);
        for _ in 0..100 {
            ga.step(&onemax_fitness);
        }
        assert!(ga.best().unwrap().fitness >= 10.0);
    }

    #[test]
    fn test_convergence_detection() {
        let config = GaConfig {
            convergence_generations: 3,
            convergence_threshold: 0.1,
            ..Default::default()
        };
        let mut ga = GeneticAlgorithm::new(config);
        ga.fitness_history = vec![5.0, 5.0, 5.0];
        assert!(ga.has_converged());
    }

    #[test]
    fn test_no_convergence_when_improving() {
        let config = GaConfig {
            convergence_generations: 3,
            convergence_threshold: 0.001,
            ..Default::default()
        };
        let mut ga = GeneticAlgorithm::new(config);
        ga.fitness_history = vec![1.0, 2.0, 3.0];
        assert!(!ga.has_converged());
    }

    #[test]
    fn test_run_returns_result() {
        let config = GaConfig {
            population_size: 20,
            chromosome_length: 3,
            max_generations: 50,
            ..Default::default()
        };
        let mut ga = GeneticAlgorithm::new(config);
        ga.initialize_real();
        let result = ga.run(&sphere_fitness);
        assert!(result.generations_run > 0);
        assert!(!result.fitness_history.is_empty());
        assert!(result.best_individual.fitness <= 0.0);
    }

    #[test]
    fn test_elitism_preserves_best() {
        let config = GaConfig {
            population_size: 10,
            chromosome_length: 3,
            elitism_count: 2,
            ..Default::default()
        };
        let mut ga = GeneticAlgorithm::new(config);
        ga.initialize_real();
        ga.evaluate(&sphere_fitness);
        let best_before = ga.best().unwrap().fitness;
        ga.step(&sphere_fitness);
        assert!(ga.best().unwrap().fitness >= best_before - 1e-10);
    }

    #[test]
    fn test_population_stats() {
        let config = GaConfig { population_size: 20, chromosome_length: 3, ..Default::default() };
        let mut ga = GeneticAlgorithm::new(config);
        ga.initialize_real();
        ga.evaluate(&sphere_fitness);
        let stats = ga.stats();
        assert_eq!(stats.size, 20);
        assert!(stats.best >= stats.worst);
        assert!(stats.mean >= stats.worst);
        assert!(stats.mean <= stats.best);
        assert!(stats.variance >= 0.0);
    }

    #[test]
    fn test_diversity() {
        let config = GaConfig { population_size: 10, chromosome_length: 3, ..Default::default() };
        let mut ga = GeneticAlgorithm::new(config);
        ga.initialize_real();
        let div = ga.diversity();
        assert!(div > 0.0);
    }

    #[test]
    fn test_uniform_random_mutation() {
        let config = GaConfig {
            mutation: Mutation::UniformRandom,
            mutation_rate: 1.0,
            population_size: 10,
            chromosome_length: 5,
            ..Default::default()
        };
        let mut ga = GeneticAlgorithm::new(config);
        ga.initialize_real();
        ga.evaluate(&sphere_fitness);
        ga.step(&sphere_fitness);
        let (lo, hi) = ga.config.bounds;
        for ind in ga.population() {
            if let Chromosome::RealValued(genes) = &ind.chromosome {
                for g in genes {
                    assert!(*g >= lo && *g <= hi);
                }
            }
        }
    }

    #[test]
    fn test_sphere_optimization_improves() {
        let config = GaConfig {
            population_size: 40,
            chromosome_length: 5,
            mutation: Mutation::Gaussian(0.5),
            mutation_rate: 0.1,
            max_generations: 50,
            bounds: (-5.0, 5.0),
            ..Default::default()
        };
        let mut ga = GeneticAlgorithm::new(config);
        ga.initialize_real();
        ga.evaluate(&sphere_fitness);
        let start_best = ga.best().unwrap().fitness;
        for _ in 0..50 {
            ga.step(&sphere_fitness);
        }
        let end_best = ga.best().unwrap().fitness;
        assert!(end_best >= start_best - 1e-10);
    }

    #[test]
    fn test_generation_counter() {
        let config = GaConfig { population_size: 10, chromosome_length: 3, ..Default::default() };
        let mut ga = GeneticAlgorithm::new(config);
        ga.initialize_real();
        ga.evaluate(&sphere_fitness);
        assert_eq!(ga.generation(), 0);
        ga.step(&sphere_fitness);
        assert_eq!(ga.generation(), 1);
        ga.step(&sphere_fitness);
        assert_eq!(ga.generation(), 2);
    }

    #[test]
    fn test_default_config() {
        let config = GaConfig::default();
        assert_eq!(config.population_size, 50);
        assert_eq!(config.chromosome_length, 10);
        assert_eq!(config.elitism_count, 2);
        assert!((config.mutation_rate - 0.05).abs() < 1e-10);
        assert!((config.crossover_rate - 0.8).abs() < 1e-10);
    }
}
