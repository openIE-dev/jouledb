//! Multi-objective optimization: NSGA-II (non-dominated sorting, crowding
//! distance), MOEA/D (decomposition with Tchebycheff scalarization), Pareto
//! dominance, hypervolume indicator, and non-dominated archive.

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
}

// ── Solution ─────────────────────────────────────────────────────

/// A multi-objective solution with decision variables and objective values.
#[derive(Debug, Clone, PartialEq)]
pub struct MoSolution {
    pub x: Vec<f64>,
    pub objectives: Vec<f64>,
    pub rank: usize,
    pub crowding_distance: f64,
}

impl MoSolution {
    pub fn new(x: Vec<f64>, num_objectives: usize) -> Self {
        Self { x, objectives: vec![0.0; num_objectives], rank: 0, crowding_distance: 0.0 }
    }

    pub fn with_objectives(x: Vec<f64>, objectives: Vec<f64>) -> Self {
        Self { x, objectives, rank: 0, crowding_distance: 0.0 }
    }
}

// ── Pareto dominance ─────────────────────────────────────────────

/// Check if solution `a` dominates `b` (all objectives minimized).
pub fn dominates(a: &[f64], b: &[f64]) -> bool {
    let mut at_least_one_better = false;
    for (ai, bi) in a.iter().zip(b.iter()) {
        if ai > bi {
            return false;
        }
        if ai < bi {
            at_least_one_better = true;
        }
    }
    at_least_one_better
}

/// Extract the Pareto front from a set of objective vectors.
pub fn pareto_front(solutions: &[MoSolution]) -> Vec<usize> {
    let n = solutions.len();
    let mut is_dominated = vec![false; n];

    for i in 0..n {
        if is_dominated[i] { continue; }
        for j in 0..n {
            if i == j || is_dominated[j] { continue; }
            if dominates(&solutions[j].objectives, &solutions[i].objectives) {
                is_dominated[i] = true;
                break;
            }
        }
    }

    (0..n).filter(|i| !is_dominated[*i]).collect()
}

// ── Non-dominated sorting ────────────────────────────────────────

/// NSGA-II non-dominated sorting. Returns fronts as Vec of index lists.
pub fn non_dominated_sort(solutions: &[MoSolution]) -> Vec<Vec<usize>> {
    let n = solutions.len();
    let mut domination_count = vec![0usize; n];
    let mut dominated_set: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut front_one: Vec<usize> = Vec::new();

    for i in 0..n {
        for j in 0..n {
            if i == j { continue; }
            if dominates(&solutions[i].objectives, &solutions[j].objectives) {
                dominated_set[i].push(j);
            } else if dominates(&solutions[j].objectives, &solutions[i].objectives) {
                domination_count[i] += 1;
            }
        }
        if domination_count[i] == 0 {
            front_one.push(i);
        }
    }

    let mut current_front = front_one;
    while !current_front.is_empty() {
        let mut next_front = Vec::new();
        for &i in &current_front {
            for &j in &dominated_set[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        fronts.push(current_front);
        current_front = next_front;
    }

    fronts
}

// ── Crowding distance ────────────────────────────────────────────

/// Compute crowding distance for a front.
pub fn crowding_distance(solutions: &[MoSolution], front: &[usize]) -> Vec<f64> {
    let n = front.len();
    if n == 0 { return Vec::new(); }
    let m = solutions[front[0]].objectives.len();
    let mut distances = vec![0.0f64; n];

    for obj in 0..m {
        // Sort front by objective value
        let mut sorted: Vec<usize> = (0..n).collect();
        sorted.sort_by(|&a, &b| {
            solutions[front[a]].objectives[obj]
                .partial_cmp(&solutions[front[b]].objectives[obj])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let f_max = solutions[front[sorted[n - 1]]].objectives[obj];
        let f_min = solutions[front[sorted[0]]].objectives[obj];
        let range = f_max - f_min;

        distances[sorted[0]] = f64::INFINITY;
        distances[sorted[n - 1]] = f64::INFINITY;

        if range > 1e-15 {
            for i in 1..n - 1 {
                let diff = solutions[front[sorted[i + 1]]].objectives[obj]
                    - solutions[front[sorted[i - 1]]].objectives[obj];
                distances[sorted[i]] += diff / range;
            }
        }
    }

    distances
}

// ── Hypervolume indicator (2D) ───────────────────────────────────

/// Compute 2D hypervolume indicator relative to a reference point.
/// All objectives are minimized.
pub fn hypervolume_2d(objectives: &[(f64, f64)], reference: (f64, f64)) -> f64 {
    if objectives.is_empty() { return 0.0; }

    // Filter out points dominated by reference
    let mut points: Vec<(f64, f64)> = objectives.iter()
        .filter(|p| p.0 <= reference.0 && p.1 <= reference.1)
        .copied()
        .collect();

    if points.is_empty() { return 0.0; }

    // Sort by first objective
    points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Extract non-dominated front: sorted by f1, keep those with strictly decreasing f2.
    let mut front: Vec<(f64, f64)> = Vec::new();
    let mut min_f2 = f64::INFINITY;
    for &p in &points {
        if p.1 < min_f2 {
            front.push(p);
            min_f2 = p.1;
        }
    }

    // Compute hypervolume
    let mut hv = 0.0;
    for i in 0..front.len() {
        let x_right = if i + 1 < front.len() { front[i + 1].0 } else { reference.0 };
        let height = reference.1 - front[i].1;
        hv += (x_right - front[i].0) * height;
    }

    hv
}

// ── NSGA-II ──────────────────────────────────────────────────────

/// NSGA-II configuration.
#[derive(Debug, Clone)]
pub struct Nsga2Config {
    pub population_size: usize,
    pub dimensions: usize,
    pub num_objectives: usize,
    pub max_generations: usize,
    pub crossover_rate: f64,
    pub mutation_rate: f64,
    pub mutation_sigma: f64,
    pub bounds: (f64, f64),
    pub seed: u64,
}

impl Default for Nsga2Config {
    fn default() -> Self {
        Self {
            population_size: 50,
            dimensions: 5,
            num_objectives: 2,
            max_generations: 100,
            crossover_rate: 0.9,
            mutation_rate: 0.1,
            mutation_sigma: 0.2,
            bounds: (-5.0, 5.0),
            seed: 42,
        }
    }
}

/// NSGA-II result.
#[derive(Debug, Clone, PartialEq)]
pub struct Nsga2Result {
    pub pareto_front: Vec<MoSolution>,
    pub generations_run: usize,
    pub hypervolume_history: Vec<f64>,
}

/// NSGA-II multi-objective optimizer.
pub struct Nsga2 {
    config: Nsga2Config,
    population: Vec<MoSolution>,
    generation: usize,
    rng: Rng,
}

impl Nsga2 {
    pub fn new(config: Nsga2Config) -> Self {
        Self {
            rng: Rng::new(config.seed),
            population: Vec::new(),
            generation: 0,
            config,
        }
    }

    pub fn initialize(&mut self) {
        let (lo, hi) = self.config.bounds;
        self.population.clear();
        for _ in 0..self.config.population_size {
            let x: Vec<f64> = (0..self.config.dimensions)
                .map(|_| lo + self.rng.next_f64() * (hi - lo))
                .collect();
            self.population.push(MoSolution::new(x, self.config.num_objectives));
        }
    }

    pub fn evaluate<F: Fn(&[f64]) -> Vec<f64>>(&mut self, obj_fn: &F) {
        for sol in &mut self.population {
            sol.objectives = obj_fn(&sol.x);
        }
    }

    fn sbx_crossover(&mut self, p1: &[f64], p2: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (lo, hi) = self.config.bounds;
        if self.rng.next_f64() > self.config.crossover_rate {
            return (p1.to_vec(), p2.to_vec());
        }
        let eta = 20.0;
        let mut c1 = Vec::with_capacity(p1.len());
        let mut c2 = Vec::with_capacity(p1.len());
        for i in 0..p1.len() {
            if self.rng.next_f64() < 0.5 {
                let u = self.rng.next_f64();
                let beta = if u <= 0.5 {
                    (2.0 * u).powf(1.0 / (eta + 1.0))
                } else {
                    (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (eta + 1.0))
                };
                c1.push((0.5 * ((1.0 + beta) * p1[i] + (1.0 - beta) * p2[i])).clamp(lo, hi));
                c2.push((0.5 * ((1.0 - beta) * p1[i] + (1.0 + beta) * p2[i])).clamp(lo, hi));
            } else {
                c1.push(p1[i]);
                c2.push(p2[i]);
            }
        }
        (c1, c2)
    }

    fn polynomial_mutation(&mut self, x: &mut Vec<f64>) {
        let (lo, hi) = self.config.bounds;
        let eta_m = 20.0;
        for v in x.iter_mut() {
            if self.rng.next_f64() < self.config.mutation_rate {
                let u = self.rng.next_f64();
                let delta = if u < 0.5 {
                    (2.0 * u).powf(1.0 / (eta_m + 1.0)) - 1.0
                } else {
                    1.0 - (2.0 * (1.0 - u)).powf(1.0 / (eta_m + 1.0))
                };
                *v += delta * (hi - lo);
                *v = v.clamp(lo, hi);
            }
        }
    }

    fn tournament_select(&mut self) -> usize {
        let i = self.rng.next_f64() as usize % self.population.len();
        let j = self.rng.next_f64() as usize % self.population.len();
        let i = i.min(self.population.len() - 1);
        let j = j.min(self.population.len() - 1);
        if self.population[i].rank < self.population[j].rank {
            i
        } else if self.population[j].rank < self.population[i].rank {
            j
        } else if self.population[i].crowding_distance > self.population[j].crowding_distance {
            i
        } else {
            j
        }
    }

    pub fn step<F: Fn(&[f64]) -> Vec<f64>>(&mut self, obj_fn: &F) {
        let pop_size = self.config.population_size;
        let num_obj = self.config.num_objectives;

        // Create offspring
        let mut offspring: Vec<MoSolution> = Vec::with_capacity(pop_size);
        while offspring.len() < pop_size {
            let p1 = self.tournament_select();
            let p2 = self.tournament_select();
            let p1x = self.population[p1].x.clone();
            let p2x = self.population[p2].x.clone();
            let (mut c1x, mut c2x) = self.sbx_crossover(&p1x, &p2x);
            self.polynomial_mutation(&mut c1x);
            self.polynomial_mutation(&mut c2x);
            let mut c1 = MoSolution::new(c1x, num_obj);
            c1.objectives = obj_fn(&c1.x);
            offspring.push(c1);
            if offspring.len() < pop_size {
                let mut c2 = MoSolution::new(c2x, num_obj);
                c2.objectives = obj_fn(&c2.x);
                offspring.push(c2);
            }
        }

        // Merge parent + offspring
        let mut combined = self.population.clone();
        combined.extend(offspring);

        // Non-dominated sorting
        let fronts = non_dominated_sort(&combined);

        // Assign ranks and crowding distances
        let mut new_pop: Vec<MoSolution> = Vec::with_capacity(pop_size);
        for (rank, front) in fronts.iter().enumerate() {
            let dists = crowding_distance(&combined, front);
            for (fi, &idx) in front.iter().enumerate() {
                combined[idx].rank = rank;
                combined[idx].crowding_distance = dists[fi];
            }

            if new_pop.len() + front.len() <= pop_size {
                for &idx in front {
                    new_pop.push(combined[idx].clone());
                }
            } else {
                // Partial front: sort by crowding distance
                let mut sorted_front: Vec<(usize, usize)> = front.iter().enumerate().map(|(fi, &idx)| (fi, idx)).collect();
                sorted_front.sort_by(|a, b| {
                    dists[b.0].partial_cmp(&dists[a.0]).unwrap_or(std::cmp::Ordering::Equal)
                });
                let remaining = pop_size - new_pop.len();
                for (_, idx) in sorted_front.into_iter().take(remaining) {
                    new_pop.push(combined[idx].clone());
                }
                break;
            }
        }

        self.population = new_pop;
        self.generation += 1;
    }

    pub fn run<F: Fn(&[f64]) -> Vec<f64>>(&mut self, obj_fn: &F) -> Nsga2Result {
        self.evaluate(obj_fn);
        let mut hv_history = Vec::new();

        for _ in 0..self.config.max_generations {
            self.step(obj_fn);
            // Compute hypervolume for 2-objective case
            if self.config.num_objectives == 2 {
                let pts: Vec<(f64, f64)> = self.population.iter()
                    .map(|s| (s.objectives[0], s.objectives[1])).collect();
                let hv = hypervolume_2d(&pts, (10.0, 10.0));
                hv_history.push(hv);
            }
        }

        let front_indices = pareto_front(&self.population);
        let pf: Vec<MoSolution> = front_indices.iter().map(|i| self.population[*i].clone()).collect();

        Nsga2Result {
            pareto_front: pf,
            generations_run: self.generation,
            hypervolume_history: hv_history,
        }
    }

    pub fn population(&self) -> &[MoSolution] { &self.population }
    pub fn generation(&self) -> usize { self.generation }
}

// ── MOEA/D ───────────────────────────────────────────────────────

/// MOEA/D with Tchebycheff scalarization.
pub struct Moead {
    dimensions: usize,
    num_objectives: usize,
    weight_vectors: Vec<Vec<f64>>,
    neighborhood_size: usize,
    neighbors: Vec<Vec<usize>>,
    solutions: Vec<MoSolution>,
    ideal_point: Vec<f64>,
    bounds: (f64, f64),
    mutation_rate: f64,
    mutation_sigma: f64,
    generation: usize,
    max_generations: usize,
    rng: Rng,
}

impl Moead {
    pub fn new(
        dimensions: usize,
        num_objectives: usize,
        num_subproblems: usize,
        neighborhood_size: usize,
        bounds: (f64, f64),
        max_generations: usize,
        seed: u64,
    ) -> Self {
        // Generate evenly spaced weight vectors for 2 objectives
        let weights: Vec<Vec<f64>> = (0..num_subproblems).map(|i| {
            let w1 = i as f64 / (num_subproblems - 1).max(1) as f64;
            vec![w1, 1.0 - w1]
        }).collect();

        // Compute neighborhoods based on Euclidean distance between weight vectors
        let k = neighborhood_size.min(num_subproblems);
        let mut neighbors: Vec<Vec<usize>> = Vec::new();
        for i in 0..num_subproblems {
            let mut dists: Vec<(f64, usize)> = (0..num_subproblems).map(|j| {
                let d: f64 = weights[i].iter().zip(weights[j].iter())
                    .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
                (d, j)
            }).collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            neighbors.push(dists.iter().take(k).map(|d| d.1).collect());
        }

        Self {
            dimensions,
            num_objectives,
            weight_vectors: weights,
            neighborhood_size: k,
            neighbors,
            solutions: Vec::new(),
            ideal_point: vec![f64::INFINITY; num_objectives],
            bounds,
            mutation_rate: 0.1,
            mutation_sigma: 0.2,
            generation: 0,
            max_generations,
            rng: Rng::new(seed),
        }
    }

    pub fn initialize<F: Fn(&[f64]) -> Vec<f64>>(&mut self, obj_fn: &F) {
        let (lo, hi) = self.bounds;
        let n = self.weight_vectors.len();
        self.solutions.clear();
        for _ in 0..n {
            let x: Vec<f64> = (0..self.dimensions)
                .map(|_| lo + self.rng.next_f64() * (hi - lo))
                .collect();
            let mut sol = MoSolution::new(x, self.num_objectives);
            sol.objectives = obj_fn(&sol.x);
            self.update_ideal(&sol.objectives);
            self.solutions.push(sol);
        }
    }

    fn update_ideal(&mut self, objectives: &[f64]) {
        for (i, &v) in objectives.iter().enumerate() {
            if v < self.ideal_point[i] {
                self.ideal_point[i] = v;
            }
        }
    }

    /// Tchebycheff scalarization.
    fn tchebycheff(&self, objectives: &[f64], weight: &[f64]) -> f64 {
        objectives.iter().zip(weight.iter()).zip(self.ideal_point.iter())
            .map(|((&f, &w), &z)| w.max(1e-10) * (f - z).abs())
            .fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn step<F: Fn(&[f64]) -> Vec<f64>>(&mut self, obj_fn: &F) {
        let n = self.solutions.len();
        let (lo, hi) = self.bounds;

        for i in 0..n {
            // Pick two parents from neighborhood
            let ni = &self.neighbors[i];
            let p1_idx = ni[self.rng.next_f64() as usize % ni.len()];
            let p2_idx = ni[self.rng.next_f64() as usize % ni.len()];

            // Crossover (intermediate)
            let mut child_x: Vec<f64> = (0..self.dimensions).map(|d| {
                let alpha = self.rng.next_f64();
                alpha * self.solutions[p1_idx].x[d] + (1.0 - alpha) * self.solutions[p2_idx].x[d]
            }).collect();

            // Mutation
            for v in child_x.iter_mut() {
                if self.rng.next_f64() < self.mutation_rate {
                    *v += self.rng.next_gaussian() * self.mutation_sigma;
                    *v = v.clamp(lo, hi);
                }
            }

            let child_obj = obj_fn(&child_x);
            self.update_ideal(&child_obj);

            // Update neighbors if child is better
            for &j in &self.neighbors[i].clone() {
                let old_scalar = self.tchebycheff(&self.solutions[j].objectives, &self.weight_vectors[j]);
                let new_scalar = self.tchebycheff(&child_obj, &self.weight_vectors[j]);
                if new_scalar < old_scalar {
                    self.solutions[j].x = child_x.clone();
                    self.solutions[j].objectives = child_obj.clone();
                }
            }
        }

        self.generation += 1;
    }

    pub fn run<F: Fn(&[f64]) -> Vec<f64>>(&mut self, obj_fn: &F) -> Vec<MoSolution> {
        for _ in 0..self.max_generations {
            self.step(obj_fn);
        }
        self.solutions.clone()
    }

    pub fn solutions(&self) -> &[MoSolution] { &self.solutions }
    pub fn generation(&self) -> usize { self.generation }
}

// ── Non-dominated archive ────────────────────────────────────────

/// Archive maintaining non-dominated solutions.
#[derive(Debug, Clone)]
pub struct NondominatedArchive {
    pub solutions: Vec<MoSolution>,
    pub max_size: usize,
}

impl NondominatedArchive {
    pub fn new(max_size: usize) -> Self {
        Self { solutions: Vec::new(), max_size }
    }

    /// Add a solution; remove any that it dominates.
    pub fn add(&mut self, sol: MoSolution) -> bool {
        // Check if new solution is dominated
        for existing in &self.solutions {
            if dominates(&existing.objectives, &sol.objectives) {
                return false;
            }
        }

        // Remove solutions dominated by new one
        self.solutions.retain(|s| !dominates(&sol.objectives, &s.objectives));
        self.solutions.push(sol);

        // Truncate if over capacity (remove lowest crowding distance)
        if self.solutions.len() > self.max_size {
            let indices: Vec<usize> = (0..self.solutions.len()).collect();
            let dists = crowding_distance(&self.solutions, &indices);
            let mut indexed: Vec<(usize, f64)> = dists.into_iter().enumerate().collect();
            indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            if let Some((worst, _)) = indexed.first() {
                self.solutions.remove(*worst);
            }
        }

        true
    }

    pub fn len(&self) -> usize { self.solutions.len() }
    pub fn is_empty(&self) -> bool { self.solutions.is_empty() }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn zdt1(x: &[f64]) -> Vec<f64> {
        let f1 = x[0];
        let n = x.len() as f64;
        let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (n - 1.0);
        let f2 = g * (1.0 - (f1 / g).sqrt());
        vec![f1, f2]
    }

    fn simple_2obj(x: &[f64]) -> Vec<f64> {
        vec![x[0] * x[0], (x[0] - 2.0).powi(2)]
    }

    #[test]
    fn test_dominates_yes() {
        assert!(dominates(&[1.0, 1.0], &[2.0, 2.0]));
        assert!(dominates(&[1.0, 2.0], &[2.0, 2.0]));
    }

    #[test]
    fn test_dominates_no() {
        assert!(!dominates(&[1.0, 3.0], &[2.0, 2.0]));
        assert!(!dominates(&[2.0, 2.0], &[2.0, 2.0]));
    }

    #[test]
    fn test_pareto_front_simple() {
        let solutions = vec![
            MoSolution::with_objectives(vec![], vec![1.0, 3.0]),
            MoSolution::with_objectives(vec![], vec![2.0, 2.0]),
            MoSolution::with_objectives(vec![], vec![3.0, 1.0]),
            MoSolution::with_objectives(vec![], vec![2.0, 3.0]),
        ];
        let front = pareto_front(&solutions);
        assert_eq!(front.len(), 3);
        assert!(front.contains(&0));
        assert!(front.contains(&1));
        assert!(front.contains(&2));
        assert!(!front.contains(&3));
    }

    #[test]
    fn test_non_dominated_sort() {
        let solutions = vec![
            MoSolution::with_objectives(vec![], vec![1.0, 4.0]),
            MoSolution::with_objectives(vec![], vec![2.0, 3.0]),
            MoSolution::with_objectives(vec![], vec![3.0, 2.0]),
            MoSolution::with_objectives(vec![], vec![4.0, 1.0]),
            MoSolution::with_objectives(vec![], vec![3.0, 3.0]),
        ];
        let fronts = non_dominated_sort(&solutions);
        assert_eq!(fronts.len(), 2);
        assert_eq!(fronts[0].len(), 4);
        assert_eq!(fronts[1].len(), 1);
        assert!(fronts[1].contains(&4));
    }

    #[test]
    fn test_crowding_distance_boundaries() {
        let solutions = vec![
            MoSolution::with_objectives(vec![], vec![1.0, 5.0]),
            MoSolution::with_objectives(vec![], vec![3.0, 3.0]),
            MoSolution::with_objectives(vec![], vec![5.0, 1.0]),
        ];
        let front: Vec<usize> = vec![0, 1, 2];
        let dists = crowding_distance(&solutions, &front);
        assert!(dists[0].is_infinite());
        assert!(dists[2].is_infinite());
        assert!(dists[1].is_finite());
    }

    #[test]
    fn test_hypervolume_2d_simple() {
        let pts = vec![(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)];
        let hv = hypervolume_2d(&pts, (5.0, 5.0));
        // Expected: (2-1)*(5-3) + (3-2)*(5-2) + (5-3)*(5-1) = 2 + 3 + 8 = 13
        assert!((hv - 13.0).abs() < 1e-4, "HV = {}", hv);
    }

    #[test]
    fn test_hypervolume_2d_empty() {
        let hv = hypervolume_2d(&[], (5.0, 5.0));
        assert!((hv - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_hypervolume_2d_single() {
        let pts = vec![(2.0, 3.0)];
        let hv = hypervolume_2d(&pts, (5.0, 5.0));
        // (5-2) * (5-3) = 6
        assert!((hv - 6.0).abs() < 1e-4, "HV = {}", hv);
    }

    #[test]
    fn test_nsga2_default_config() {
        let c = Nsga2Config::default();
        assert_eq!(c.population_size, 50);
        assert_eq!(c.num_objectives, 2);
    }

    #[test]
    fn test_nsga2_initialize() {
        let config = Nsga2Config { population_size: 20, dimensions: 3, ..Default::default() };
        let mut nsga2 = Nsga2::new(config);
        nsga2.initialize();
        assert_eq!(nsga2.population().len(), 20);
    }

    #[test]
    fn test_nsga2_step() {
        let config = Nsga2Config {
            population_size: 20,
            dimensions: 2,
            bounds: (0.0, 1.0),
            ..Default::default()
        };
        let mut nsga2 = Nsga2::new(config);
        nsga2.initialize();
        nsga2.evaluate(&simple_2obj);
        nsga2.step(&simple_2obj);
        assert_eq!(nsga2.generation(), 1);
        assert_eq!(nsga2.population().len(), 20);
    }

    #[test]
    fn test_nsga2_run() {
        let config = Nsga2Config {
            population_size: 20,
            dimensions: 2,
            max_generations: 20,
            bounds: (0.0, 2.0),
            ..Default::default()
        };
        let mut nsga2 = Nsga2::new(config);
        nsga2.initialize();
        let result = nsga2.run(&simple_2obj);
        assert!(!result.pareto_front.is_empty());
        assert_eq!(result.generations_run, 20);
    }

    #[test]
    fn test_nsga2_hypervolume_history() {
        let config = Nsga2Config {
            population_size: 20,
            dimensions: 2,
            max_generations: 10,
            num_objectives: 2,
            bounds: (0.0, 2.0),
            ..Default::default()
        };
        let mut nsga2 = Nsga2::new(config);
        nsga2.initialize();
        let result = nsga2.run(&simple_2obj);
        assert_eq!(result.hypervolume_history.len(), 10);
    }

    #[test]
    fn test_moead_creation() {
        let moead = Moead::new(3, 2, 10, 3, (0.0, 1.0), 20, 42);
        assert_eq!(moead.generation(), 0);
    }

    #[test]
    fn test_moead_step() {
        let mut moead = Moead::new(2, 2, 10, 3, (0.0, 2.0), 20, 42);
        moead.initialize(&simple_2obj);
        moead.step(&simple_2obj);
        assert_eq!(moead.generation(), 1);
    }

    #[test]
    fn test_moead_run() {
        let mut moead = Moead::new(2, 2, 15, 5, (0.0, 2.0), 20, 42);
        moead.initialize(&simple_2obj);
        let solutions = moead.run(&simple_2obj);
        assert!(!solutions.is_empty());
    }

    #[test]
    fn test_archive_add() {
        let mut archive = NondominatedArchive::new(100);
        assert!(archive.add(MoSolution::with_objectives(vec![], vec![1.0, 3.0])));
        assert!(archive.add(MoSolution::with_objectives(vec![], vec![3.0, 1.0])));
        assert_eq!(archive.len(), 2);
    }

    #[test]
    fn test_archive_removes_dominated() {
        let mut archive = NondominatedArchive::new(100);
        archive.add(MoSolution::with_objectives(vec![], vec![3.0, 3.0]));
        archive.add(MoSolution::with_objectives(vec![], vec![1.0, 1.0]));
        assert_eq!(archive.len(), 1);
        assert!((archive.solutions[0].objectives[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_archive_rejects_dominated() {
        let mut archive = NondominatedArchive::new(100);
        archive.add(MoSolution::with_objectives(vec![], vec![1.0, 1.0]));
        let added = archive.add(MoSolution::with_objectives(vec![], vec![3.0, 3.0]));
        assert!(!added);
        assert_eq!(archive.len(), 1);
    }

    #[test]
    fn test_archive_capacity() {
        let mut archive = NondominatedArchive::new(3);
        for i in 0..5 {
            archive.add(MoSolution::with_objectives(vec![], vec![i as f64, 5.0 - i as f64]));
        }
        assert!(archive.len() <= 3);
    }

    #[test]
    fn test_solution_creation() {
        let sol = MoSolution::new(vec![1.0, 2.0], 2);
        assert_eq!(sol.x, vec![1.0, 2.0]);
        assert_eq!(sol.objectives, vec![0.0, 0.0]);
        assert_eq!(sol.rank, 0);
    }
}
