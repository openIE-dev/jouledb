//! Agent-based simulation — agents with position, state, behavior, interactions.
//!
//! Replaces Mesa.js / AgentScript / NetLogo-web with pure Rust.
//! Supports agent with position/state/behavior, environment grid,
//! step function, agent interactions, population dynamics,
//! a simple predator-prey model, and simulation statistics.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Domain errors for agent-based simulation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentSimError {
    /// Grid dimensions are zero.
    ZeroDimension,
    /// Agent not found.
    AgentNotFound(u64),
    /// Position out of bounds.
    OutOfBounds { x: usize, y: usize, width: usize, height: usize },
    /// Duplicate agent ID.
    DuplicateAgent(u64),
}

impl fmt::Display for AgentSimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "grid dimensions must be non-zero"),
            Self::AgentNotFound(id) => write!(f, "agent not found: {id}"),
            Self::OutOfBounds { x, y, width, height } => {
                write!(f, "({x}, {y}) out of bounds for {width}x{height} grid")
            }
            Self::DuplicateAgent(id) => write!(f, "duplicate agent: {id}"),
        }
    }
}

impl std::error::Error for AgentSimError {}

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
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    fn next_usize(&mut self, max: usize) -> usize {
        if max == 0 { return 0; }
        (self.next_u64() % max as u64) as usize
    }
}

// ── Agent Types ─────────────────────────────────────────────────

/// Species of an agent (used for predator-prey model).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Species {
    Prey,
    Predator,
}

/// State of an agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentState {
    Alive,
    Dead,
    Reproducing,
}

/// Direction for movement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    North,
    South,
    East,
    West,
    NorthEast,
    NorthWest,
    SouthEast,
    SouthWest,
    Stay,
}

impl Direction {
    /// All 8 movement directions (no Stay).
    pub fn all_moves() -> &'static [Direction] {
        &[
            Direction::North, Direction::South, Direction::East, Direction::West,
            Direction::NorthEast, Direction::NorthWest,
            Direction::SouthEast, Direction::SouthWest,
        ]
    }

    /// Offset (dx, dy) for the direction.
    pub fn offset(&self) -> (i32, i32) {
        match self {
            Direction::North => (0, -1),
            Direction::South => (0, 1),
            Direction::East => (1, 0),
            Direction::West => (-1, 0),
            Direction::NorthEast => (1, -1),
            Direction::NorthWest => (-1, -1),
            Direction::SouthEast => (1, 1),
            Direction::SouthWest => (-1, 1),
            Direction::Stay => (0, 0),
        }
    }
}

// ── Agent ───────────────────────────────────────────────────────

/// An agent in the simulation.
#[derive(Debug, Clone)]
pub struct Agent {
    pub id: u64,
    pub x: usize,
    pub y: usize,
    pub species: Species,
    pub state: AgentState,
    pub energy: f64,
    pub age: u64,
    pub max_energy: f64,
    pub reproduction_threshold: f64,
    pub energy_gain_per_step: f64,
    pub energy_cost_per_step: f64,
}

impl Agent {
    /// Create a new prey agent.
    pub fn prey(id: u64, x: usize, y: usize) -> Self {
        Self {
            id, x, y,
            species: Species::Prey,
            state: AgentState::Alive,
            energy: 10.0,
            age: 0,
            max_energy: 20.0,
            reproduction_threshold: 15.0,
            energy_gain_per_step: 1.0,
            energy_cost_per_step: 0.5,
        }
    }

    /// Create a new predator agent.
    pub fn predator(id: u64, x: usize, y: usize) -> Self {
        Self {
            id, x, y,
            species: Species::Predator,
            state: AgentState::Alive,
            energy: 20.0,
            age: 0,
            max_energy: 40.0,
            reproduction_threshold: 30.0,
            energy_gain_per_step: 0.0,
            energy_cost_per_step: 1.0,
        }
    }

    /// Whether the agent is alive.
    pub fn is_alive(&self) -> bool {
        self.state == AgentState::Alive || self.state == AgentState::Reproducing
    }
}

// ── Environment ─────────────────────────────────────────────────

/// The environment grid.
#[derive(Debug, Clone)]
pub struct Environment {
    pub width: usize,
    pub height: usize,
    /// Resource level at each cell (e.g. grass for prey).
    pub resources: Vec<f64>,
    /// Max resource per cell.
    pub max_resource: f64,
    /// Resource regrowth rate per step.
    pub regrowth_rate: f64,
}

impl Environment {
    /// Create a new environment with uniform resources.
    pub fn new(width: usize, height: usize) -> Result<Self, AgentSimError> {
        if width == 0 || height == 0 {
            return Err(AgentSimError::ZeroDimension);
        }
        let size = width * height;
        Ok(Self {
            width,
            height,
            resources: vec![1.0; size],
            max_resource: 1.0,
            regrowth_rate: 0.1,
        })
    }

    /// Get resource at position.
    pub fn resource_at(&self, x: usize, y: usize) -> f64 {
        if x < self.width && y < self.height {
            self.resources[y * self.width + x]
        } else {
            0.0
        }
    }

    /// Consume resource at position. Returns amount consumed.
    pub fn consume(&mut self, x: usize, y: usize, amount: f64) -> f64 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        let idx = y * self.width + x;
        let available = self.resources[idx];
        let consumed = available.min(amount);
        self.resources[idx] -= consumed;
        consumed
    }

    /// Regrow resources across the grid.
    pub fn regrow(&mut self) {
        for r in &mut self.resources {
            *r = (*r + self.regrowth_rate).min(self.max_resource);
        }
    }

    /// Wrap coordinates.
    pub fn wrap(&self, x: i32, y: i32) -> (usize, usize) {
        let w = self.width as i32;
        let h = self.height as i32;
        (((x % w + w) % w) as usize, ((y % h + h) % h) as usize)
    }
}

// ── Simulation Statistics ───────────────────────────────────────

/// Snapshot of simulation stats at one step.
#[derive(Debug, Clone)]
pub struct SimStats {
    pub step: u64,
    pub prey_count: usize,
    pub predator_count: usize,
    pub total_prey_energy: f64,
    pub total_predator_energy: f64,
    pub average_resource: f64,
}

// ── Simulation ──────────────────────────────────────────────────

/// The agent-based simulation engine.
#[derive(Debug, Clone)]
pub struct Simulation {
    pub env: Environment,
    agents: Vec<Agent>,
    next_id: u64,
    rng: Rng,
    step_count: u64,
    history: Vec<SimStats>,
    predator_hunt_energy: f64,
}

impl Simulation {
    /// Create a new simulation.
    pub fn new(env: Environment, seed: u64) -> Self {
        Self {
            env,
            agents: Vec::new(),
            next_id: 0,
            rng: Rng::new(seed),
            step_count: 0,
            history: Vec::new(),
            predator_hunt_energy: 10.0,
        }
    }

    /// Set the energy a predator gains from eating prey.
    pub fn with_hunt_energy(mut self, energy: f64) -> Self {
        self.predator_hunt_energy = energy;
        self
    }

    /// Add an agent to the simulation.
    pub fn add_agent(&mut self, mut agent: Agent) -> u64 {
        agent.id = self.next_id;
        let id = self.next_id;
        self.next_id += 1;
        self.agents.push(agent);
        id
    }

    /// Seed initial populations randomly.
    pub fn seed_population(&mut self, prey_count: usize, predator_count: usize) {
        let w = self.env.width;
        let h = self.env.height;
        for _ in 0..prey_count {
            let x = self.rng.next_usize(w);
            let y = self.rng.next_usize(h);
            let agent = Agent::prey(0, x, y);
            self.add_agent(agent);
        }
        for _ in 0..predator_count {
            let x = self.rng.next_usize(w);
            let y = self.rng.next_usize(h);
            let agent = Agent::predator(0, x, y);
            self.add_agent(agent);
        }
    }

    /// Current step.
    pub fn step_count(&self) -> u64 { self.step_count }

    /// All agents (including dead).
    pub fn agents(&self) -> &[Agent] { &self.agents }

    /// Count of living agents of a given species.
    pub fn count_species(&self, species: Species) -> usize {
        self.agents.iter()
            .filter(|a| a.species == species && a.is_alive())
            .count()
    }

    /// Total living agents.
    pub fn living_count(&self) -> usize {
        self.agents.iter().filter(|a| a.is_alive()).count()
    }

    /// History of stats snapshots.
    pub fn history(&self) -> &[SimStats] { &self.history }

    /// Get a specific agent by ID.
    pub fn get_agent(&self, id: u64) -> Option<&Agent> {
        self.agents.iter().find(|a| a.id == id)
    }

    /// Record current stats.
    fn record_stats(&mut self) {
        let prey_count = self.count_species(Species::Prey);
        let predator_count = self.count_species(Species::Predator);
        let total_prey_energy: f64 = self.agents.iter()
            .filter(|a| a.species == Species::Prey && a.is_alive())
            .map(|a| a.energy)
            .sum();
        let total_predator_energy: f64 = self.agents.iter()
            .filter(|a| a.species == Species::Predator && a.is_alive())
            .map(|a| a.energy)
            .sum();
        let average_resource = self.env.resources.iter().sum::<f64>()
            / self.env.resources.len() as f64;

        self.history.push(SimStats {
            step: self.step_count,
            prey_count,
            predator_count,
            total_prey_energy,
            total_predator_energy,
            average_resource,
        });
    }

    /// Advance the simulation by one step.
    pub fn step(&mut self) {
        let w = self.env.width;
        let h = self.env.height;

        // Move all living agents randomly.
        let n = self.agents.len();
        for i in 0..n {
            if !self.agents[i].is_alive() {
                continue;
            }

            // Random movement.
            let dirs = Direction::all_moves();
            let dir_idx = self.rng.next_usize(dirs.len());
            let dir = dirs[dir_idx];
            let (dx, dy) = dir.offset();
            let (nx, ny) = self.env.wrap(
                self.agents[i].x as i32 + dx,
                self.agents[i].y as i32 + dy,
            );
            self.agents[i].x = nx;
            self.agents[i].y = ny;

            // Energy cost.
            self.agents[i].energy -= self.agents[i].energy_cost_per_step;

            // Prey eats resources.
            if self.agents[i].species == Species::Prey {
                let gain = self.agents[i].energy_gain_per_step;
                let consumed = self.env.consume(nx, ny, gain);
                self.agents[i].energy += consumed;
                self.agents[i].energy = self.agents[i].energy.min(self.agents[i].max_energy);
            }

            self.agents[i].age += 1;

            // Die if no energy.
            if self.agents[i].energy <= 0.0 {
                self.agents[i].state = AgentState::Dead;
            }
        }

        // Predator-prey interactions.
        // Build a spatial map of living prey.
        let mut prey_at: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (i, a) in self.agents.iter().enumerate() {
            if a.species == Species::Prey && a.is_alive() {
                prey_at.entry((a.x, a.y)).or_default().push(i);
            }
        }

        // Predators eat prey at their location.
        let hunt_energy = self.predator_hunt_energy;
        for i in 0..n {
            if self.agents[i].species != Species::Predator || !self.agents[i].is_alive() {
                continue;
            }
            let pos = (self.agents[i].x, self.agents[i].y);
            if let Some(prey_list) = prey_at.get_mut(&pos) {
                if let Some(prey_idx) = prey_list.pop() {
                    self.agents[prey_idx].state = AgentState::Dead;
                    self.agents[i].energy = (self.agents[i].energy + hunt_energy)
                        .min(self.agents[i].max_energy);
                }
            }
        }

        // Reproduction.
        let mut new_agents = Vec::new();
        for i in 0..self.agents.len() {
            if !self.agents[i].is_alive() {
                continue;
            }
            if self.agents[i].energy >= self.agents[i].reproduction_threshold {
                self.agents[i].energy /= 2.0;
                let (nx, ny) = self.env.wrap(
                    self.agents[i].x as i32 + 1,
                    self.agents[i].y as i32,
                );
                let mut child = match self.agents[i].species {
                    Species::Prey => Agent::prey(0, nx, ny),
                    Species::Predator => Agent::predator(0, nx, ny),
                };
                child.energy = self.agents[i].energy;
                new_agents.push(child);
            }
        }
        for child in new_agents {
            self.add_agent(child);
        }

        // Environment regrow.
        self.env.regrow();

        self.step_count += 1;
        self.record_stats();
    }

    /// Run for multiple steps.
    pub fn run(&mut self, steps: u64) {
        for _ in 0..steps {
            self.step();
        }
    }

    /// Remove all dead agents from the list (garbage collection).
    pub fn gc_dead(&mut self) {
        self.agents.retain(|a| a.is_alive());
    }

    /// Agents at a specific position.
    pub fn agents_at(&self, x: usize, y: usize) -> Vec<&Agent> {
        self.agents.iter()
            .filter(|a| a.x == x && a.y == y && a.is_alive())
            .collect()
    }

    /// Average energy of living agents by species.
    pub fn average_energy(&self, species: Species) -> Option<f64> {
        let living: Vec<f64> = self.agents.iter()
            .filter(|a| a.species == species && a.is_alive())
            .map(|a| a.energy)
            .collect();
        if living.is_empty() {
            None
        } else {
            Some(living.iter().sum::<f64>() / living.len() as f64)
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sim(w: usize, h: usize) -> Simulation {
        let env = Environment::new(w, h).unwrap();
        Simulation::new(env, 42)
    }

    #[test]
    fn test_environment_creation() {
        let env = Environment::new(10, 10).unwrap();
        assert_eq!(env.width, 10);
        assert_eq!(env.height, 10);
    }

    #[test]
    fn test_environment_zero() {
        assert!(Environment::new(0, 5).is_err());
    }

    #[test]
    fn test_environment_resource() {
        let env = Environment::new(5, 5).unwrap();
        assert!((env.resource_at(2, 2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_environment_consume() {
        let mut env = Environment::new(5, 5).unwrap();
        let consumed = env.consume(2, 2, 0.5);
        assert!((consumed - 0.5).abs() < 1e-10);
        assert!((env.resource_at(2, 2) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_environment_regrow() {
        let mut env = Environment::new(5, 5).unwrap();
        env.consume(2, 2, 1.0);
        env.regrow();
        assert!(env.resource_at(2, 2) > 0.0);
    }

    #[test]
    fn test_environment_wrap() {
        let env = Environment::new(10, 10).unwrap();
        assert_eq!(env.wrap(-1, -1), (9, 9));
        assert_eq!(env.wrap(10, 10), (0, 0));
    }

    #[test]
    fn test_add_agent() {
        let mut sim = make_sim(10, 10);
        let id = sim.add_agent(Agent::prey(0, 3, 3));
        assert_eq!(id, 0);
        assert_eq!(sim.living_count(), 1);
    }

    #[test]
    fn test_seed_population() {
        let mut sim = make_sim(20, 20);
        sim.seed_population(30, 10);
        assert_eq!(sim.count_species(Species::Prey), 30);
        assert_eq!(sim.count_species(Species::Predator), 10);
        assert_eq!(sim.living_count(), 40);
    }

    #[test]
    fn test_step_advances() {
        let mut sim = make_sim(10, 10);
        sim.seed_population(5, 2);
        sim.step();
        assert_eq!(sim.step_count(), 1);
    }

    #[test]
    fn test_agent_moves() {
        let mut sim = make_sim(10, 10);
        sim.add_agent(Agent::prey(0, 5, 5));
        sim.step();
        let a = sim.get_agent(0).unwrap();
        // Agent should have moved or stayed (unlikely to be exact same spot).
        assert!(a.age == 1);
    }

    #[test]
    fn test_agent_death_no_energy() {
        let mut sim = make_sim(10, 10);
        let mut agent = Agent::prey(0, 5, 5);
        agent.energy = 0.1;
        agent.energy_cost_per_step = 1.0;
        agent.energy_gain_per_step = 0.0;
        sim.add_agent(agent);
        sim.step();
        assert_eq!(sim.count_species(Species::Prey), 0);
    }

    #[test]
    fn test_predator_eats_prey() {
        let mut sim = make_sim(10, 10);
        let mut prey = Agent::prey(0, 5, 5);
        prey.energy_cost_per_step = 0.0;
        prey.energy_gain_per_step = 0.0;
        sim.add_agent(prey);

        let mut predator = Agent::predator(0, 5, 5);
        predator.energy = 5.0;
        predator.energy_cost_per_step = 0.0;
        sim.add_agent(predator);

        // We need them to be at the same position after step.
        // Place them at same spot and hope at least once they co-locate.
        // Run a few steps.
        for _ in 0..50 {
            // Reset positions to same cell.
            let n = sim.agents.len();
            for i in 0..n {
                if sim.agents[i].is_alive() {
                    sim.agents[i].x = 5;
                    sim.agents[i].y = 5;
                }
            }
            sim.step();
        }
        // After 50 steps with constant co-location, prey should be eaten.
        let prey_left = sim.count_species(Species::Prey);
        assert_eq!(prey_left, 0, "Prey should have been eaten");
    }

    #[test]
    fn test_reproduction() {
        let mut sim = make_sim(10, 10);
        let mut prey = Agent::prey(0, 5, 5);
        prey.energy = 20.0;
        prey.reproduction_threshold = 15.0;
        prey.energy_cost_per_step = 0.0;
        sim.add_agent(prey);
        sim.step();
        // Should have reproduced.
        assert!(sim.count_species(Species::Prey) >= 2);
    }

    #[test]
    fn test_gc_dead() {
        let mut sim = make_sim(10, 10);
        let mut a = Agent::prey(0, 5, 5);
        a.energy = 0.01;
        a.energy_cost_per_step = 1.0;
        a.energy_gain_per_step = 0.0;
        sim.add_agent(a);
        sim.step();
        assert_eq!(sim.agents().len(), 1);
        sim.gc_dead();
        assert_eq!(sim.agents().len(), 0);
    }

    #[test]
    fn test_history_recorded() {
        let mut sim = make_sim(10, 10);
        sim.seed_population(5, 2);
        sim.run(3);
        assert_eq!(sim.history().len(), 3);
    }

    #[test]
    fn test_agents_at() {
        let mut sim = make_sim(10, 10);
        sim.add_agent(Agent::prey(0, 3, 4));
        sim.add_agent(Agent::prey(0, 3, 4));
        sim.add_agent(Agent::prey(0, 7, 7));
        let at_34 = sim.agents_at(3, 4);
        assert_eq!(at_34.len(), 2);
    }

    #[test]
    fn test_average_energy() {
        let mut sim = make_sim(10, 10);
        let mut a1 = Agent::prey(0, 3, 3);
        a1.energy = 10.0;
        let mut a2 = Agent::prey(0, 4, 4);
        a2.energy = 20.0;
        sim.add_agent(a1);
        sim.add_agent(a2);
        let avg = sim.average_energy(Species::Prey).unwrap();
        assert!((avg - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_average_energy_none() {
        let sim = make_sim(10, 10);
        assert!(sim.average_energy(Species::Prey).is_none());
    }

    #[test]
    fn test_direction_offsets() {
        assert_eq!(Direction::North.offset(), (0, -1));
        assert_eq!(Direction::SouthEast.offset(), (1, 1));
        assert_eq!(Direction::Stay.offset(), (0, 0));
    }

    #[test]
    fn test_direction_all_moves() {
        assert_eq!(Direction::all_moves().len(), 8);
    }

    #[test]
    fn test_prey_species() {
        let a = Agent::prey(1, 0, 0);
        assert_eq!(a.species, Species::Prey);
        assert!(a.is_alive());
    }

    #[test]
    fn test_predator_species() {
        let a = Agent::predator(1, 0, 0);
        assert_eq!(a.species, Species::Predator);
    }

    #[test]
    fn test_population_dynamics_run() {
        let mut sim = make_sim(30, 30);
        sim.seed_population(50, 10);
        sim.run(20);
        // After 20 steps, some agents should still be alive.
        assert!(sim.living_count() > 0);
        assert!(sim.history().len() == 20);
    }
}
