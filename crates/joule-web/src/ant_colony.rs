//! Ant Colony Optimization — pheromone grids, foraging, TSP solver.
//!
//! Replaces AntColony.js / ACOpy / scipy ACO libraries. 2D pheromone grid,
//! probabilistic ant movement, pheromone deposit and evaporation, multiple
//! pheromone types (food/danger), food/nest foraging, path optimization,
//! ACO for TSP (construct tours, pheromone update), convergence tracking.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum AcoError {
    ZeroDimension,
    InvalidParameter(String),
    NoFood,
    NoCities,
}

impl fmt::Display for AcoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "dimensions must be non-zero"),
            Self::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
            Self::NoFood => write!(f, "no food sources defined"),
            Self::NoCities => write!(f, "no cities defined for TSP"),
        }
    }
}

impl std::error::Error for AcoError {}

// ── Pheromone Grid ─────────────────────────────────────────────

/// Type of pheromone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PheromoneType {
    Food,
    Danger,
}

/// A 2D pheromone grid for ant foraging.
#[derive(Debug, Clone)]
pub struct PheromoneGrid {
    width: usize,
    height: usize,
    food_pheromone: Vec<f64>,
    danger_pheromone: Vec<f64>,
    evaporation_rate: f64,
    min_pheromone: f64,
    max_pheromone: f64,
}

impl PheromoneGrid {
    pub fn new(width: usize, height: usize, evaporation_rate: f64) -> Result<Self, AcoError> {
        if width == 0 || height == 0 {
            return Err(AcoError::ZeroDimension);
        }
        if evaporation_rate < 0.0 || evaporation_rate > 1.0 {
            return Err(AcoError::InvalidParameter("evaporation_rate must be in [0,1]".into()));
        }
        let size = width * height;
        Ok(Self {
            width,
            height,
            food_pheromone: vec![0.0; size],
            danger_pheromone: vec![0.0; size],
            evaporation_rate,
            min_pheromone: 0.001,
            max_pheromone: 100.0,
        })
    }

    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }

    fn idx(&self, x: usize, y: usize) -> usize { y * self.width + x }

    /// Get pheromone level.
    pub fn get(&self, ptype: PheromoneType, x: usize, y: usize) -> f64 {
        if x >= self.width || y >= self.height { return 0.0; }
        let grid = match ptype {
            PheromoneType::Food => &self.food_pheromone,
            PheromoneType::Danger => &self.danger_pheromone,
        };
        grid[self.idx(x, y)]
    }

    /// Deposit pheromone.
    pub fn deposit(&mut self, ptype: PheromoneType, x: usize, y: usize, amount: f64) {
        if x >= self.width || y >= self.height { return; }
        let idx = self.idx(x, y);
        let grid = match ptype {
            PheromoneType::Food => &mut self.food_pheromone,
            PheromoneType::Danger => &mut self.danger_pheromone,
        };
        grid[idx] = (grid[idx] + amount).min(self.max_pheromone);
    }

    /// Evaporate all pheromones by the evaporation rate.
    pub fn evaporate(&mut self) {
        let factor = 1.0 - self.evaporation_rate;
        let min = self.min_pheromone;
        for val in &mut self.food_pheromone {
            *val = (*val * factor).max(min);
            if *val <= min { *val = 0.0; }
        }
        for val in &mut self.danger_pheromone {
            *val = (*val * factor).max(min);
            if *val <= min { *val = 0.0; }
        }
    }

    /// Total food pheromone.
    pub fn total_food_pheromone(&self) -> f64 {
        self.food_pheromone.iter().sum()
    }

    /// Max food pheromone.
    pub fn max_food_pheromone(&self) -> f64 {
        self.food_pheromone.iter().cloned().fold(0.0f64, f64::max)
    }
}

// ── Foraging Ant ───────────────────────────────────────────────

/// State of a foraging ant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AntState {
    Searching,
    Returning,
}

/// A foraging ant on the pheromone grid.
#[derive(Debug, Clone)]
pub struct ForagingAnt {
    pub x: usize,
    pub y: usize,
    pub state: AntState,
    pub carrying_food: bool,
    pub path: Vec<(usize, usize)>,
    pub steps: u64,
}

impl ForagingAnt {
    pub fn new(x: usize, y: usize) -> Self {
        Self {
            x, y,
            state: AntState::Searching,
            carrying_food: false,
            path: vec![(x, y)],
            steps: 0,
        }
    }
}

// ── Foraging Simulation ────────────────────────────────────────

/// Foraging simulation on a 2D grid.
#[derive(Debug, Clone)]
pub struct ForagingSim {
    grid: PheromoneGrid,
    ants: Vec<ForagingAnt>,
    nest: (usize, usize),
    food_sources: Vec<(usize, usize, f64)>, // (x, y, remaining)
    deposit_amount: f64,
    iteration: u64,
    food_collected: u64,
    rng_state: u64,
}

impl ForagingSim {
    pub fn new(
        width: usize,
        height: usize,
        nest: (usize, usize),
        num_ants: usize,
        evaporation_rate: f64,
    ) -> Result<Self, AcoError> {
        let grid = PheromoneGrid::new(width, height, evaporation_rate)?;
        let ants = (0..num_ants).map(|_| ForagingAnt::new(nest.0, nest.1)).collect();
        Ok(Self {
            grid,
            ants,
            nest,
            food_sources: Vec::new(),
            deposit_amount: 1.0,
            iteration: 0,
            food_collected: 0,
            rng_state: 42,
        })
    }

    /// Add a food source.
    pub fn add_food(&mut self, x: usize, y: usize, amount: f64) {
        self.food_sources.push((x, y, amount));
    }

    pub fn iteration(&self) -> u64 { self.iteration }
    pub fn food_collected(&self) -> u64 { self.food_collected }
    pub fn ant_count(&self) -> usize { self.ants.len() }

    /// Simple LCG random.
    fn next_rand(&mut self) -> f64 {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.rng_state >> 33) as f64) / ((1u64 << 31) as f64)
    }

    /// Get valid neighboring cells.
    fn neighbors(&self, x: usize, y: usize) -> Vec<(usize, usize)> {
        let w = self.grid.width();
        let h = self.grid.height();
        let mut nbrs = Vec::with_capacity(8);
        for dy in [-1i32, 0, 1] {
            for dx in [-1i32, 0, 1] {
                if dx == 0 && dy == 0 { continue; }
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                    nbrs.push((nx as usize, ny as usize));
                }
            }
        }
        nbrs
    }

    /// Advance one timestep.
    pub fn step(&mut self) {
        let nest = self.nest;
        let deposit = self.deposit_amount;
        let n_ants = self.ants.len();

        for ai in 0..n_ants {
            let x = self.ants[ai].x;
            let y = self.ants[ai].y;
            let nbrs = self.neighbors(x, y);

            if nbrs.is_empty() { continue; }

            match self.ants[ai].state {
                AntState::Searching => {
                    // Check if at food source
                    let mut found_food = false;
                    for fs in &mut self.food_sources {
                        if fs.0 == x && fs.1 == y && fs.2 > 0.0 {
                            fs.2 -= 1.0;
                            self.ants[ai].carrying_food = true;
                            self.ants[ai].state = AntState::Returning;
                            found_food = true;
                            break;
                        }
                    }

                    if !found_food {
                        // Move probabilistically based on food pheromone
                        let weights: Vec<f64> = nbrs.iter().map(|&(nx, ny)| {
                            let p = self.grid.get(PheromoneType::Food, nx, ny);
                            let d = self.grid.get(PheromoneType::Danger, nx, ny);
                            (p + 0.01) / (d + 1.0)
                        }).collect();

                        let total: f64 = weights.iter().sum();
                        let r = self.next_rand() * total;
                        let mut cumulative = 0.0;
                        let mut chosen = 0;
                        for (i, &w) in weights.iter().enumerate() {
                            cumulative += w;
                            if r <= cumulative {
                                chosen = i;
                                break;
                            }
                        }
                        if chosen >= nbrs.len() { chosen = nbrs.len() - 1; }

                        let (nx, ny) = nbrs[chosen];
                        self.ants[ai].x = nx;
                        self.ants[ai].y = ny;
                        self.ants[ai].path.push((nx, ny));
                    }
                }
                AntState::Returning => {
                    // Deposit food pheromone on the path back
                    self.grid.deposit(PheromoneType::Food, x, y, deposit);

                    if x == nest.0 && y == nest.1 {
                        // Arrived at nest
                        self.ants[ai].carrying_food = false;
                        self.ants[ai].state = AntState::Searching;
                        self.ants[ai].path.clear();
                        self.ants[ai].path.push((nest.0, nest.1));
                        self.food_collected += 1;
                    } else {
                        // Move toward nest (greedy: minimize distance)
                        let mut best = nbrs[0];
                        let mut best_dist = f64::MAX;
                        for &(nx, ny) in &nbrs {
                            let dx = (nx as f64 - nest.0 as f64).abs();
                            let dy = (ny as f64 - nest.1 as f64).abs();
                            let d = dx + dy;
                            if d < best_dist {
                                best_dist = d;
                                best = (nx, ny);
                            }
                        }
                        self.ants[ai].x = best.0;
                        self.ants[ai].y = best.1;
                        self.ants[ai].path.push(best);
                    }
                }
            }
            self.ants[ai].steps += 1;
        }

        self.grid.evaporate();
        self.iteration += 1;
    }

    /// Advance by n steps.
    pub fn step_n(&mut self, n: u64) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Access the pheromone grid.
    pub fn pheromone_grid(&self) -> &PheromoneGrid {
        &self.grid
    }
}

// ── ACO for TSP ────────────────────────────────────────────────

/// ACO solver for the Traveling Salesman Problem.
#[derive(Debug, Clone)]
pub struct AcoTsp {
    /// City positions (x, y).
    cities: Vec<(f64, f64)>,
    /// Distance matrix.
    distances: Vec<Vec<f64>>,
    /// Pheromone matrix (edge pheromones).
    pheromones: Vec<Vec<f64>>,
    /// Number of ants.
    num_ants: usize,
    /// Pheromone influence.
    alpha: f64,
    /// Distance influence (heuristic).
    beta: f64,
    /// Evaporation rate.
    evaporation: f64,
    /// Pheromone deposit factor.
    q_factor: f64,
    /// Best tour found.
    best_tour: Vec<usize>,
    /// Best tour length.
    best_length: f64,
    /// Convergence history: best length per iteration.
    convergence: Vec<f64>,
    iteration: u64,
    rng_state: u64,
}

impl AcoTsp {
    /// Create a new ACO-TSP solver.
    pub fn new(
        cities: Vec<(f64, f64)>,
        num_ants: usize,
        alpha: f64,
        beta: f64,
        evaporation: f64,
    ) -> Result<Self, AcoError> {
        let n = cities.len();
        if n == 0 {
            return Err(AcoError::NoCities);
        }

        // Compute distance matrix
        let mut distances = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let dx = cities[i].0 - cities[j].0;
                let dy = cities[i].1 - cities[j].1;
                distances[i][j] = (dx * dx + dy * dy).sqrt();
            }
        }

        let pheromones = vec![vec![1.0; n]; n];

        Ok(Self {
            cities,
            distances,
            pheromones,
            num_ants,
            alpha,
            beta,
            evaporation,
            q_factor: 100.0,
            best_tour: Vec::new(),
            best_length: f64::MAX,
            convergence: Vec::new(),
            iteration: 0,
            rng_state: 12345,
        })
    }

    pub fn city_count(&self) -> usize { self.cities.len() }
    pub fn best_tour(&self) -> &[usize] { &self.best_tour }
    pub fn best_length(&self) -> f64 { self.best_length }
    pub fn convergence(&self) -> &[f64] { &self.convergence }
    pub fn iteration(&self) -> u64 { self.iteration }

    fn next_rand(&mut self) -> f64 {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.rng_state >> 33) as f64) / ((1u64 << 31) as f64)
    }

    /// Tour length.
    fn tour_length(&self, tour: &[usize]) -> f64 {
        let mut total = 0.0;
        for i in 0..tour.len() {
            let j = (i + 1) % tour.len();
            total += self.distances[tour[i]][tour[j]];
        }
        total
    }

    /// Construct a tour for one ant starting from a given city.
    fn construct_tour(&mut self, start: usize) -> Vec<usize> {
        let n = self.cities.len();
        let mut tour = vec![start];
        let mut visited = vec![false; n];
        visited[start] = true;

        for _ in 1..n {
            let current = *tour.last().unwrap();
            let mut probs = Vec::with_capacity(n);
            let mut total = 0.0;

            for j in 0..n {
                if visited[j] {
                    probs.push(0.0);
                    continue;
                }
                let tau = self.pheromones[current][j].powf(self.alpha);
                let eta = if self.distances[current][j] > 1e-12 {
                    (1.0 / self.distances[current][j]).powf(self.beta)
                } else {
                    1e12
                };
                let p = tau * eta;
                probs.push(p);
                total += p;
            }

            // Roulette wheel selection
            let r = self.next_rand() * total;
            let mut cumulative = 0.0;
            let mut chosen = 0;
            for (j, &p) in probs.iter().enumerate() {
                cumulative += p;
                if r <= cumulative {
                    chosen = j;
                    break;
                }
            }
            if visited[chosen] {
                // Fallback: pick first unvisited
                for j in 0..n {
                    if !visited[j] {
                        chosen = j;
                        break;
                    }
                }
            }

            visited[chosen] = true;
            tour.push(chosen);
        }

        tour
    }

    /// Run one iteration: construct tours for all ants, update pheromones.
    pub fn iterate(&mut self) {
        let n = self.cities.len();
        let mut tours = Vec::with_capacity(self.num_ants);
        let mut lengths = Vec::with_capacity(self.num_ants);

        for a in 0..self.num_ants {
            let start = a % n;
            let tour = self.construct_tour(start);
            let length = self.tour_length(&tour);
            tours.push(tour);
            lengths.push(length);
        }

        // Update best
        for (i, &len) in lengths.iter().enumerate() {
            if len < self.best_length {
                self.best_length = len;
                self.best_tour = tours[i].clone();
            }
        }

        // Evaporate
        for row in &mut self.pheromones {
            for val in row.iter_mut() {
                *val *= 1.0 - self.evaporation;
                if *val < 0.001 { *val = 0.001; }
            }
        }

        // Deposit pheromone
        for (ti, tour) in tours.iter().enumerate() {
            let deposit = self.q_factor / lengths[ti];
            for i in 0..tour.len() {
                let j = (i + 1) % tour.len();
                self.pheromones[tour[i]][tour[j]] += deposit;
                self.pheromones[tour[j]][tour[i]] += deposit;
            }
        }

        self.convergence.push(self.best_length);
        self.iteration += 1;
    }

    /// Run multiple iterations.
    pub fn iterate_n(&mut self, n: u64) {
        for _ in 0..n {
            self.iterate();
        }
    }

    /// Check if the solution has converged (improvement below threshold over last k iterations).
    pub fn has_converged(&self, window: usize, threshold: f64) -> bool {
        if self.convergence.len() < window { return false; }
        let recent = &self.convergence[self.convergence.len() - window..];
        let first = recent[0];
        let last = recent[recent.len() - 1];
        (first - last).abs() < threshold
    }
}

impl fmt::Display for AcoTsp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AcoTsp(cities={}, best={:.2}, iter={})",
            self.cities.len(), self.best_length, self.iteration)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < EPS }

    #[test]
    fn test_pheromone_grid_new() {
        let g = PheromoneGrid::new(10, 10, 0.1).unwrap();
        assert_eq!(g.width(), 10);
        assert_eq!(g.height(), 10);
    }

    #[test]
    fn test_pheromone_grid_zero_dim() {
        assert!(PheromoneGrid::new(0, 10, 0.1).is_err());
    }

    #[test]
    fn test_pheromone_deposit_and_get() {
        let mut g = PheromoneGrid::new(5, 5, 0.1).unwrap();
        g.deposit(PheromoneType::Food, 2, 3, 5.0);
        assert!(approx(g.get(PheromoneType::Food, 2, 3), 5.0));
        assert!(approx(g.get(PheromoneType::Danger, 2, 3), 0.0));
    }

    #[test]
    fn test_pheromone_evaporation() {
        let mut g = PheromoneGrid::new(5, 5, 0.5).unwrap();
        g.deposit(PheromoneType::Food, 2, 2, 10.0);
        g.evaporate();
        let val = g.get(PheromoneType::Food, 2, 2);
        assert!(approx(val, 5.0));
    }

    #[test]
    fn test_pheromone_max_clamp() {
        let mut g = PheromoneGrid::new(5, 5, 0.0).unwrap();
        g.deposit(PheromoneType::Food, 0, 0, 200.0);
        assert!(g.get(PheromoneType::Food, 0, 0) <= 100.0);
    }

    #[test]
    fn test_foraging_sim_creation() {
        let sim = ForagingSim::new(20, 20, (10, 10), 5, 0.1).unwrap();
        assert_eq!(sim.ant_count(), 5);
        assert_eq!(sim.iteration(), 0);
    }

    #[test]
    fn test_foraging_sim_step() {
        let mut sim = ForagingSim::new(20, 20, (10, 10), 5, 0.1).unwrap();
        sim.add_food(15, 15, 10.0);
        sim.step();
        assert_eq!(sim.iteration(), 1);
    }

    #[test]
    fn test_foraging_ants_move() {
        let mut sim = ForagingSim::new(20, 20, (10, 10), 1, 0.1).unwrap();
        sim.add_food(15, 15, 100.0);
        sim.step();
        // Ant should have moved from nest
        let ant = &sim.ants[0];
        let moved = ant.x != 10 || ant.y != 10;
        assert!(moved, "ant should have moved from nest");
    }

    #[test]
    fn test_foraging_food_collection() {
        let mut sim = ForagingSim::new(10, 10, (5, 5), 10, 0.05).unwrap();
        sim.add_food(6, 5, 100.0); // Very close to nest
        sim.step_n(200);
        // With food right next to nest, some food should have been collected
        assert!(sim.food_collected() > 0, "should have collected food");
    }

    #[test]
    fn test_foraging_pheromone_trail() {
        let mut sim = ForagingSim::new(10, 10, (5, 5), 5, 0.01).unwrap();
        sim.add_food(5, 6, 100.0); // One step below nest
        sim.step_n(50);
        let total = sim.pheromone_grid().total_food_pheromone();
        assert!(total > 0.0, "pheromone should have been deposited");
    }

    #[test]
    fn test_aco_tsp_creation() {
        let cities = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let tsp = AcoTsp::new(cities, 10, 1.0, 2.0, 0.1).unwrap();
        assert_eq!(tsp.city_count(), 4);
    }

    #[test]
    fn test_aco_tsp_no_cities() {
        assert!(AcoTsp::new(vec![], 10, 1.0, 2.0, 0.1).is_err());
    }

    #[test]
    fn test_aco_tsp_single_iteration() {
        let cities = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let mut tsp = AcoTsp::new(cities, 10, 1.0, 2.0, 0.1).unwrap();
        tsp.iterate();
        assert_eq!(tsp.iteration(), 1);
        assert!(!tsp.best_tour().is_empty());
        assert!(tsp.best_length() < f64::MAX);
    }

    #[test]
    fn test_aco_tsp_improves() {
        let cities = vec![
            (0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 1.0),
            (1.0, 1.0), (0.0, 1.0),
        ];
        let mut tsp = AcoTsp::new(cities, 20, 1.0, 3.0, 0.2).unwrap();
        tsp.iterate();
        let first_best = tsp.best_length();
        tsp.iterate_n(50);
        assert!(tsp.best_length() <= first_best);
    }

    #[test]
    fn test_aco_tsp_square_optimal() {
        // Square: optimal tour = 4.0 (perimeter)
        let cities = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let mut tsp = AcoTsp::new(cities, 20, 1.0, 5.0, 0.1).unwrap();
        tsp.iterate_n(100);
        // Best tour should be close to 4.0 (the perimeter)
        assert!(tsp.best_length() < 4.5, "best length = {}", tsp.best_length());
    }

    #[test]
    fn test_aco_tsp_convergence_history() {
        let cities = vec![(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)];
        let mut tsp = AcoTsp::new(cities, 5, 1.0, 2.0, 0.1).unwrap();
        tsp.iterate_n(10);
        assert_eq!(tsp.convergence().len(), 10);
    }

    #[test]
    fn test_aco_tsp_has_converged() {
        let cities = vec![(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)];
        let mut tsp = AcoTsp::new(cities, 10, 1.0, 2.0, 0.1).unwrap();
        tsp.iterate_n(100);
        // After 100 iterations on 3 cities, should be converged
        assert!(tsp.has_converged(10, 0.1));
    }

    #[test]
    fn test_aco_tsp_tour_visits_all_cities() {
        let cities = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)];
        let mut tsp = AcoTsp::new(cities, 10, 1.0, 2.0, 0.1).unwrap();
        tsp.iterate();
        let tour = tsp.best_tour();
        assert_eq!(tour.len(), 5);
        let mut sorted = tour.to_vec();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_pheromone_danger() {
        let mut g = PheromoneGrid::new(5, 5, 0.1).unwrap();
        g.deposit(PheromoneType::Danger, 1, 1, 3.0);
        assert!(approx(g.get(PheromoneType::Danger, 1, 1), 3.0));
    }

    #[test]
    fn test_display() {
        let cities = vec![(0.0, 0.0), (1.0, 0.0)];
        let tsp = AcoTsp::new(cities, 5, 1.0, 2.0, 0.1).unwrap();
        let s = format!("{tsp}");
        assert!(s.contains("AcoTsp"));
    }

    #[test]
    fn test_invalid_evaporation_rate() {
        assert!(PheromoneGrid::new(5, 5, 1.5).is_err());
        assert!(PheromoneGrid::new(5, 5, -0.1).is_err());
    }
}
