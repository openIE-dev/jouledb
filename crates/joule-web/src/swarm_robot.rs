//! Swarm robotics — stigmergy, local interaction rules, emergent behavior,
//! scalability, and self-organization for large robot collectives.
//!
//! Pure-Rust swarm simulation with configurable interaction rules,
//! pheromone-based stigmergy, and statistical analysis of emergent
//! properties like clustering, dispersion, and collective transport.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Swarm simulation errors.
#[derive(Debug, Clone, PartialEq)]
pub enum SwarmError {
    /// Invalid swarm configuration.
    InvalidConfig(String),
    /// Agent not found.
    AgentNotFound(u64),
    /// Environment bounds exceeded.
    OutOfBounds { x: f64, y: f64 },
}

impl fmt::Display for SwarmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::AgentNotFound(id) => write!(f, "agent not found: {id}"),
            Self::OutOfBounds { x, y } => write!(f, "out of bounds: ({x:.2}, {y:.2})"),
        }
    }
}

impl std::error::Error for SwarmError {}

// ── PRNG ────────────────────────────────────────────────────────

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
        (self.next_u64() & 0x000F_FFFF_FFFF_FFFF) as f64 / (1u64 << 52) as f64
    }

    /// Uniform in [-1, 1].
    fn next_symmetric(&mut self) -> f64 {
        self.next_f64() * 2.0 - 1.0
    }
}

// ── Swarm Agent ─────────────────────────────────────────────────

/// State of an individual swarm agent.
#[derive(Debug, Clone)]
pub struct SwarmAgent {
    pub id: u64,
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub heading: f64,
    pub max_speed: f64,
    pub sensor_range: f64,
    pub carrying: bool,
}

impl SwarmAgent {
    pub fn new(id: u64, x: f64, y: f64) -> Self {
        Self {
            id,
            x,
            y,
            vx: 0.0,
            vy: 0.0,
            heading: 0.0,
            max_speed: 1.0,
            sensor_range: 5.0,
            carrying: false,
        }
    }

    pub fn with_max_speed(mut self, speed: f64) -> Self {
        self.max_speed = speed;
        self
    }

    pub fn with_sensor_range(mut self, range: f64) -> Self {
        self.sensor_range = range;
        self
    }

    pub fn distance_to(&self, other: &SwarmAgent) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    pub fn distance_to_point(&self, px: f64, py: f64) -> f64 {
        let dx = self.x - px;
        let dy = self.y - py;
        (dx * dx + dy * dy).sqrt()
    }

    fn clamp_speed(&mut self) {
        let speed = (self.vx * self.vx + self.vy * self.vy).sqrt();
        if speed > self.max_speed && speed > 1e-12 {
            self.vx = self.vx / speed * self.max_speed;
            self.vy = self.vy / speed * self.max_speed;
        }
    }

    fn step(&mut self, dt: f64) {
        self.clamp_speed();
        self.x += self.vx * dt;
        self.y += self.vy * dt;
        let speed = (self.vx * self.vx + self.vy * self.vy).sqrt();
        if speed > 1e-12 {
            self.heading = self.vy.atan2(self.vx);
        }
    }
}

impl fmt::Display for SwarmAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Agent({}, pos=({:.2},{:.2}))", self.id, self.x, self.y)
    }
}

// ── Pheromone Grid (Stigmergy) ──────────────────────────────────

/// A 2D pheromone field for indirect communication (stigmergy).
#[derive(Debug, Clone)]
pub struct PheromoneGrid {
    pub width: usize,
    pub height: usize,
    pub cell_size: f64,
    pub evaporation_rate: f64,
    pub diffusion_rate: f64,
    cells: Vec<f64>,
}

impl PheromoneGrid {
    pub fn new(width: usize, height: usize, cell_size: f64) -> Self {
        Self {
            width,
            height,
            cell_size,
            evaporation_rate: 0.01,
            diffusion_rate: 0.05,
            cells: vec![0.0; width * height],
        }
    }

    pub fn with_evaporation(mut self, rate: f64) -> Self {
        self.evaporation_rate = rate.clamp(0.0, 1.0);
        self
    }

    pub fn with_diffusion(mut self, rate: f64) -> Self {
        self.diffusion_rate = rate.clamp(0.0, 1.0);
        self
    }

    fn to_cell(&self, world_x: f64, world_y: f64) -> Option<(usize, usize)> {
        if world_x < 0.0 || world_y < 0.0 {
            return None;
        }
        let cx = (world_x / self.cell_size) as usize;
        let cy = (world_y / self.cell_size) as usize;
        if cx < self.width && cy < self.height {
            Some((cx, cy))
        } else {
            None
        }
    }

    fn idx(&self, cx: usize, cy: usize) -> usize {
        cy * self.width + cx
    }

    /// Deposit pheromone at a world position.
    pub fn deposit(&mut self, world_x: f64, world_y: f64, amount: f64) {
        if let Some((cx, cy)) = self.to_cell(world_x, world_y) {
            let i = self.idx(cx, cy);
            self.cells[i] += amount;
        }
    }

    /// Read pheromone at a world position.
    pub fn read(&self, world_x: f64, world_y: f64) -> f64 {
        match self.to_cell(world_x, world_y) {
            Some((cx, cy)) => self.cells[self.idx(cx, cy)],
            None => 0.0,
        }
    }

    /// Evaporate and diffuse pheromones one step.
    pub fn update(&mut self) {
        let old = self.cells.clone();
        for cy in 0..self.height {
            for cx in 0..self.width {
                let i = self.idx(cx, cy);
                // Evaporation.
                let mut val = old[i] * (1.0 - self.evaporation_rate);
                // Diffusion from neighbors.
                let neighbors = [
                    (cx.wrapping_sub(1), cy),
                    (cx + 1, cy),
                    (cx, cy.wrapping_sub(1)),
                    (cx, cy + 1),
                ];
                let mut nbr_sum = 0.0;
                let mut nbr_count = 0;
                for (nx, ny) in &neighbors {
                    if *nx < self.width && *ny < self.height {
                        nbr_sum += old[self.idx(*nx, *ny)];
                        nbr_count += 1;
                    }
                }
                if nbr_count > 0 {
                    let avg = nbr_sum / nbr_count as f64;
                    val += self.diffusion_rate * (avg - old[i]);
                }
                self.cells[i] = val.max(0.0);
            }
        }
    }

    /// Total pheromone in the grid.
    pub fn total(&self) -> f64 {
        self.cells.iter().sum()
    }

    /// Maximum pheromone concentration.
    pub fn max_concentration(&self) -> f64 {
        self.cells.iter().copied().fold(0.0f64, f64::max)
    }

    /// Find the gradient direction from a world position (toward higher concentration).
    pub fn gradient(&self, world_x: f64, world_y: f64) -> (f64, f64) {
        let here = self.read(world_x, world_y);
        let dx_pos = self.read(world_x + self.cell_size, world_y);
        let dx_neg = self.read(world_x - self.cell_size, world_y);
        let dy_pos = self.read(world_x, world_y + self.cell_size);
        let dy_neg = self.read(world_x, world_y - self.cell_size);
        let gx = (dx_pos - dx_neg) / (2.0 * self.cell_size);
        let gy = (dy_pos - dy_neg) / (2.0 * self.cell_size);
        let _ = here;
        (gx, gy)
    }
}

impl fmt::Display for PheromoneGrid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PheromoneGrid({}x{}, total={:.2}, max={:.4})",
            self.width,
            self.height,
            self.total(),
            self.max_concentration(),
        )
    }
}

// ── Interaction Rules ───────────────────────────────────────────

/// Configurable local interaction rule weights.
#[derive(Debug, Clone)]
pub struct InteractionRules {
    /// Separation: steer away from too-close neighbors.
    pub separation_weight: f64,
    pub separation_dist: f64,
    /// Cohesion: steer toward local centroid.
    pub cohesion_weight: f64,
    /// Alignment: match neighbors' heading.
    pub alignment_weight: f64,
    /// Attraction to pheromone gradient.
    pub pheromone_weight: f64,
    /// Random walk noise magnitude.
    pub noise_weight: f64,
}

impl InteractionRules {
    pub fn new() -> Self {
        Self {
            separation_weight: 2.0,
            separation_dist: 2.0,
            cohesion_weight: 1.0,
            alignment_weight: 1.0,
            pheromone_weight: 0.5,
            noise_weight: 0.2,
        }
    }

    pub fn with_separation(mut self, weight: f64, dist: f64) -> Self {
        self.separation_weight = weight;
        self.separation_dist = dist;
        self
    }

    pub fn with_cohesion(mut self, weight: f64) -> Self {
        self.cohesion_weight = weight;
        self
    }

    pub fn with_alignment(mut self, weight: f64) -> Self {
        self.alignment_weight = weight;
        self
    }

    pub fn with_pheromone(mut self, weight: f64) -> Self {
        self.pheromone_weight = weight;
        self
    }

    pub fn with_noise(mut self, weight: f64) -> Self {
        self.noise_weight = weight;
        self
    }
}

impl fmt::Display for InteractionRules {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Rules(sep={:.1}, coh={:.1}, align={:.1})",
            self.separation_weight, self.cohesion_weight, self.alignment_weight
        )
    }
}

// ── Swarm Simulation ────────────────────────────────────────────

/// The main swarm simulation engine.
#[derive(Debug, Clone)]
pub struct SwarmSimulation {
    pub agents: Vec<SwarmAgent>,
    pub rules: InteractionRules,
    pub pheromones: Option<PheromoneGrid>,
    pub world_width: f64,
    pub world_height: f64,
    pub wrap_around: bool,
    pub time: f64,
    pub step_count: u64,
    rng: Rng,
}

impl SwarmSimulation {
    pub fn new(world_width: f64, world_height: f64) -> Self {
        Self {
            agents: Vec::new(),
            rules: InteractionRules::new(),
            pheromones: None,
            world_width,
            world_height,
            wrap_around: true,
            time: 0.0,
            step_count: 0,
            rng: Rng::new(42),
        }
    }

    pub fn with_rules(mut self, rules: InteractionRules) -> Self {
        self.rules = rules;
        self
    }

    pub fn with_pheromones(mut self, grid: PheromoneGrid) -> Self {
        self.pheromones = Some(grid);
        self
    }

    pub fn with_wrap(mut self, wrap: bool) -> Self {
        self.wrap_around = wrap;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Rng::new(seed);
        self
    }

    pub fn add_agent(&mut self, agent: SwarmAgent) {
        self.agents.push(agent);
    }

    /// Spawn n agents at random positions.
    pub fn spawn_random(&mut self, count: usize, max_speed: f64, sensor_range: f64) {
        let base_id = self.agents.len() as u64;
        for i in 0..count {
            let x = self.rng.next_f64() * self.world_width;
            let y = self.rng.next_f64() * self.world_height;
            let agent = SwarmAgent::new(base_id + i as u64, x, y)
                .with_max_speed(max_speed)
                .with_sensor_range(sensor_range);
            self.agents.push(agent);
        }
    }

    /// Step the simulation forward by dt.
    pub fn step(&mut self, dt: f64) {
        let n = self.agents.len();
        let positions: Vec<(f64, f64)> = self.agents.iter().map(|a| (a.x, a.y)).collect();
        let velocities: Vec<(f64, f64)> = self.agents.iter().map(|a| (a.vx, a.vy)).collect();
        let sensor_ranges: Vec<f64> = self.agents.iter().map(|a| a.sensor_range).collect();

        let mut new_vx = vec![0.0f64; n];
        let mut new_vy = vec![0.0f64; n];

        for i in 0..n {
            let (px, py) = positions[i];
            let (cvx, cvy) = velocities[i];
            let sr = sensor_ranges[i];

            // Find neighbors within sensor range.
            let mut sep_x = 0.0f64;
            let mut sep_y = 0.0f64;
            let mut coh_x = 0.0f64;
            let mut coh_y = 0.0f64;
            let mut align_x = 0.0f64;
            let mut align_y = 0.0f64;
            let mut nbr_count = 0usize;

            for j in 0..n {
                if j == i {
                    continue;
                }
                let (ox, oy) = positions[j];
                let dx = ox - px;
                let dy = oy - py;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist > sr || dist < 1e-12 {
                    continue;
                }
                nbr_count += 1;

                // Separation.
                if dist < self.rules.separation_dist {
                    let inv = 1.0 / dist;
                    sep_x -= dx * inv;
                    sep_y -= dy * inv;
                }

                // Cohesion (accumulate neighbor positions).
                coh_x += ox;
                coh_y += oy;

                // Alignment.
                let (nvx, nvy) = velocities[j];
                align_x += nvx;
                align_y += nvy;
            }

            let mut fx = 0.0f64;
            let mut fy = 0.0f64;

            // Separation force.
            fx += sep_x * self.rules.separation_weight;
            fy += sep_y * self.rules.separation_weight;

            if nbr_count > 0 {
                // Cohesion: steer toward centroid.
                let avg_x = coh_x / nbr_count as f64;
                let avg_y = coh_y / nbr_count as f64;
                fx += (avg_x - px) * self.rules.cohesion_weight;
                fy += (avg_y - py) * self.rules.cohesion_weight;

                // Alignment: match average velocity.
                let avg_vx = align_x / nbr_count as f64;
                let avg_vy = align_y / nbr_count as f64;
                fx += (avg_vx - cvx) * self.rules.alignment_weight;
                fy += (avg_vy - cvy) * self.rules.alignment_weight;
            }

            // Pheromone gradient following.
            if let Some(ref grid) = self.pheromones {
                let (gx, gy) = grid.gradient(px, py);
                fx += gx * self.rules.pheromone_weight;
                fy += gy * self.rules.pheromone_weight;
            }

            // Random walk noise.
            fx += self.rng.next_symmetric() * self.rules.noise_weight;
            fy += self.rng.next_symmetric() * self.rules.noise_weight;

            new_vx[i] = cvx + fx * dt;
            new_vy[i] = cvy + fy * dt;
        }

        // Apply velocities and step agents.
        for i in 0..n {
            self.agents[i].vx = new_vx[i];
            self.agents[i].vy = new_vy[i];
            self.agents[i].step(dt);

            // Wrap or clamp to world bounds.
            if self.wrap_around {
                if self.agents[i].x < 0.0 {
                    self.agents[i].x += self.world_width;
                }
                if self.agents[i].x >= self.world_width {
                    self.agents[i].x -= self.world_width;
                }
                if self.agents[i].y < 0.0 {
                    self.agents[i].y += self.world_height;
                }
                if self.agents[i].y >= self.world_height {
                    self.agents[i].y -= self.world_height;
                }
            } else {
                self.agents[i].x = self.agents[i].x.clamp(0.0, self.world_width);
                self.agents[i].y = self.agents[i].y.clamp(0.0, self.world_height);
            }
        }

        // Update pheromone grid.
        if let Some(ref mut grid) = self.pheromones {
            grid.update();
        }

        self.time += dt;
        self.step_count += 1;
    }

    // ── Emergent behavior metrics ───────────────────────────────

    /// Average distance to the swarm centroid (dispersion metric).
    pub fn dispersion(&self) -> f64 {
        if self.agents.is_empty() {
            return 0.0;
        }
        let (cx, cy) = self.centroid();
        let sum: f64 = self.agents.iter().map(|a| a.distance_to_point(cx, cy)).sum();
        sum / self.agents.len() as f64
    }

    /// Centroid of the swarm.
    pub fn centroid(&self) -> (f64, f64) {
        if self.agents.is_empty() {
            return (0.0, 0.0);
        }
        let n = self.agents.len() as f64;
        let sx: f64 = self.agents.iter().map(|a| a.x).sum();
        let sy: f64 = self.agents.iter().map(|a| a.y).sum();
        (sx / n, sy / n)
    }

    /// Average nearest-neighbor distance (clustering metric).
    pub fn avg_nearest_neighbor(&self) -> f64 {
        if self.agents.len() < 2 {
            return 0.0;
        }
        let mut total = 0.0f64;
        for i in 0..self.agents.len() {
            let mut min_d = f64::INFINITY;
            for j in 0..self.agents.len() {
                if i == j {
                    continue;
                }
                let d = self.agents[i].distance_to(&self.agents[j]);
                if d < min_d {
                    min_d = d;
                }
            }
            total += min_d;
        }
        total / self.agents.len() as f64
    }

    /// Order parameter: alignment of velocities (0=random, 1=aligned).
    pub fn alignment_order(&self) -> f64 {
        if self.agents.is_empty() {
            return 0.0;
        }
        let mut svx = 0.0f64;
        let mut svy = 0.0f64;
        let mut total_speed = 0.0f64;
        for a in &self.agents {
            svx += a.vx;
            svy += a.vy;
            total_speed += (a.vx * a.vx + a.vy * a.vy).sqrt();
        }
        if total_speed < 1e-12 {
            return 0.0;
        }
        let avg_vel_mag = (svx * svx + svy * svy).sqrt();
        avg_vel_mag / total_speed
    }

    /// Count distinct clusters using a distance threshold.
    pub fn cluster_count(&self, threshold: f64) -> usize {
        let n = self.agents.len();
        if n == 0 {
            return 0;
        }
        let mut labels = vec![0usize; n];
        let mut current_label = 0usize;
        let mut visited = vec![false; n];

        for start in 0..n {
            if visited[start] {
                continue;
            }
            current_label += 1;
            // BFS.
            let mut queue = vec![start];
            visited[start] = true;
            labels[start] = current_label;
            let mut qi = 0;
            while qi < queue.len() {
                let idx = queue[qi];
                qi += 1;
                for j in 0..n {
                    if !visited[j] && self.agents[idx].distance_to(&self.agents[j]) < threshold {
                        visited[j] = true;
                        labels[j] = current_label;
                        queue.push(j);
                    }
                }
            }
        }
        current_label
    }
}

impl fmt::Display for SwarmSimulation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Swarm({} agents, {:.0}x{:.0}, t={:.2}, steps={})",
            self.agents.len(),
            self.world_width,
            self.world_height,
            self.time,
            self.step_count,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_distance() {
        let a = SwarmAgent::new(1, 0.0, 0.0);
        let b = SwarmAgent::new(2, 3.0, 4.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_agent_speed_clamp() {
        let mut a = SwarmAgent::new(1, 0.0, 0.0).with_max_speed(1.0);
        a.vx = 10.0;
        a.vy = 0.0;
        a.step(1.0);
        assert!((a.x - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_pheromone_deposit_read() {
        let mut grid = PheromoneGrid::new(10, 10, 1.0);
        grid.deposit(5.5, 3.5, 10.0);
        let val = grid.read(5.5, 3.5);
        assert!((val - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_pheromone_evaporation() {
        let mut grid = PheromoneGrid::new(10, 10, 1.0).with_evaporation(0.5).with_diffusion(0.0);
        grid.deposit(5.5, 5.5, 100.0);
        let before = grid.total();
        grid.update();
        let after = grid.total();
        assert!(after < before);
    }

    #[test]
    fn test_pheromone_out_of_bounds() {
        let grid = PheromoneGrid::new(5, 5, 1.0);
        assert!((grid.read(-1.0, -1.0)).abs() < 1e-9);
        assert!((grid.read(100.0, 100.0)).abs() < 1e-9);
    }

    #[test]
    fn test_pheromone_gradient() {
        let mut grid = PheromoneGrid::new(20, 20, 1.0).with_evaporation(0.0).with_diffusion(0.0);
        // Deposit at cell (15,10); gradient reads cells at +/- cell_size
        // Query from (14.5,10.5): dx_pos reads cell(15,10)=100, dx_neg reads cell(13,10)=0
        grid.deposit(15.5, 10.5, 100.0);
        let (gx, _gy) = grid.gradient(14.5, 10.5);
        assert!(gx > 0.0); // Gradient points toward higher concentration.
    }

    #[test]
    fn test_swarm_spawn() {
        let mut sim = SwarmSimulation::new(100.0, 100.0);
        sim.spawn_random(20, 1.0, 5.0);
        assert_eq!(sim.agents.len(), 20);
    }

    #[test]
    fn test_swarm_centroid() {
        let mut sim = SwarmSimulation::new(100.0, 100.0);
        sim.add_agent(SwarmAgent::new(1, 0.0, 0.0));
        sim.add_agent(SwarmAgent::new(2, 10.0, 0.0));
        let (cx, cy) = sim.centroid();
        assert!((cx - 5.0).abs() < 1e-9);
        assert!((cy).abs() < 1e-9);
    }

    #[test]
    fn test_swarm_dispersion() {
        let mut sim = SwarmSimulation::new(100.0, 100.0);
        sim.add_agent(SwarmAgent::new(1, 5.0, 5.0));
        sim.add_agent(SwarmAgent::new(2, 5.0, 5.0));
        assert!((sim.dispersion()).abs() < 1e-9);
    }

    #[test]
    fn test_swarm_step_moves_agents() {
        let mut sim = SwarmSimulation::new(100.0, 100.0).with_wrap(false);
        let mut a = SwarmAgent::new(1, 50.0, 50.0);
        a.vx = 5.0;
        a.vy = 0.0;
        sim.add_agent(a);
        let x0 = sim.agents[0].x;
        sim.step(1.0);
        assert!(sim.agents[0].x > x0);
    }

    #[test]
    fn test_swarm_wrap_around() {
        let mut sim = SwarmSimulation::new(100.0, 100.0).with_wrap(true);
        let mut a = SwarmAgent::new(1, 99.0, 50.0).with_max_speed(10.0);
        a.vx = 5.0;
        sim.add_agent(a);
        sim.step(1.0);
        // Should wrap.
        assert!(sim.agents[0].x < 50.0);
    }

    #[test]
    fn test_swarm_no_wrap_clamp() {
        let mut sim = SwarmSimulation::new(100.0, 100.0).with_wrap(false);
        let mut a = SwarmAgent::new(1, 99.0, 50.0).with_max_speed(200.0);
        a.vx = 200.0;
        sim.add_agent(a);
        sim.step(1.0);
        assert!((sim.agents[0].x - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_alignment_order_single() {
        let mut sim = SwarmSimulation::new(100.0, 100.0);
        let mut a = SwarmAgent::new(1, 50.0, 50.0);
        a.vx = 1.0;
        sim.add_agent(a);
        assert!((sim.alignment_order() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cluster_count() {
        let mut sim = SwarmSimulation::new(100.0, 100.0);
        sim.add_agent(SwarmAgent::new(1, 0.0, 0.0));
        sim.add_agent(SwarmAgent::new(2, 1.0, 0.0));
        sim.add_agent(SwarmAgent::new(3, 50.0, 50.0));
        sim.add_agent(SwarmAgent::new(4, 51.0, 50.0));
        assert_eq!(sim.cluster_count(5.0), 2);
    }

    #[test]
    fn test_cluster_count_all_one() {
        let mut sim = SwarmSimulation::new(100.0, 100.0);
        sim.add_agent(SwarmAgent::new(1, 0.0, 0.0));
        sim.add_agent(SwarmAgent::new(2, 1.0, 0.0));
        sim.add_agent(SwarmAgent::new(3, 2.0, 0.0));
        assert_eq!(sim.cluster_count(5.0), 1);
    }

    #[test]
    fn test_avg_nearest_neighbor() {
        let mut sim = SwarmSimulation::new(100.0, 100.0);
        sim.add_agent(SwarmAgent::new(1, 0.0, 0.0));
        sim.add_agent(SwarmAgent::new(2, 3.0, 4.0));
        let ann = sim.avg_nearest_neighbor();
        assert!((ann - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_interaction_rules_builder() {
        let rules = InteractionRules::new()
            .with_separation(3.0, 1.5)
            .with_cohesion(2.0)
            .with_alignment(0.5);
        assert!((rules.separation_weight - 3.0).abs() < 1e-9);
        assert!((rules.cohesion_weight - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_display_impls() {
        let a = SwarmAgent::new(42, 1.0, 2.0);
        assert!(format!("{a}").contains("42"));
        let grid = PheromoneGrid::new(5, 5, 1.0);
        assert!(format!("{grid}").contains("5x5"));
        let rules = InteractionRules::new();
        assert!(format!("{rules}").contains("sep="));
        let sim = SwarmSimulation::new(100.0, 100.0);
        assert!(format!("{sim}").contains("0 agents"));
    }

    #[test]
    fn test_separation_behavior() {
        let mut sim = SwarmSimulation::new(100.0, 100.0)
            .with_rules(InteractionRules::new()
                .with_separation(10.0, 5.0)
                .with_cohesion(0.0)
                .with_alignment(0.0)
                .with_noise(0.0))
            .with_wrap(false);
        sim.add_agent(SwarmAgent::new(1, 50.0, 50.0).with_max_speed(10.0).with_sensor_range(20.0));
        sim.add_agent(SwarmAgent::new(2, 51.0, 50.0).with_max_speed(10.0).with_sensor_range(20.0));
        let d0 = sim.agents[0].distance_to(&sim.agents[1]);
        for _ in 0..20 {
            sim.step(0.1);
        }
        let d1 = sim.agents[0].distance_to(&sim.agents[1]);
        assert!(d1 > d0); // Agents should have moved apart.
    }
}
