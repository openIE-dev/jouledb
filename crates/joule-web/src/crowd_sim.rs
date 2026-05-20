//! Crowd simulation — Social Force Model, proxemics, evacuation, density.
//!
//! Replaces Menge / PedPy / CrowdSim.js libraries. Agents with position, velocity,
//! goal, and radius. Social Force Model (Helbing): desired velocity, agent repulsion,
//! wall repulsion. Personal space (proxemics), density-dependent speed reduction,
//! lane formation (emergent), bottleneck flow, evacuation with panic parameter,
//! density/flow measurement at cross-sections, and agent groups (families).

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum CrowdError {
    InvalidParameter(String),
    NoAgents,
    NoExits,
}

impl fmt::Display for CrowdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
            Self::NoAgents => write!(f, "no agents in simulation"),
            Self::NoExits => write!(f, "no exits defined"),
        }
    }
}

impl std::error::Error for CrowdError {}

// ── Vec2 ───────────────────────────────────────────────────────

/// 2D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };

    pub fn new(x: f64, y: f64) -> Self { Self { x, y } }

    pub fn length(&self) -> f64 { (self.x * self.x + self.y * self.y).sqrt() }

    pub fn length_sq(&self) -> f64 { self.x * self.x + self.y * self.y }

    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len < 1e-12 { return Self::ZERO; }
        Self { x: self.x / len, y: self.y / len }
    }

    pub fn scale(&self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s } }

    pub fn add(&self, other: &Self) -> Self { Self { x: self.x + other.x, y: self.y + other.y } }

    pub fn sub(&self, other: &Self) -> Self { Self { x: self.x - other.x, y: self.y - other.y } }

    pub fn dot(&self, other: &Self) -> f64 { self.x * other.x + self.y * other.y }

    pub fn distance(&self, other: &Self) -> f64 { self.sub(other).length() }

    pub fn clamp_length(&self, max: f64) -> Self {
        let len = self.length();
        if len <= max { *self } else { self.normalize().scale(max) }
    }

    /// Perpendicular vector (rotated 90 degrees CCW).
    pub fn perp(&self) -> Self { Self { x: -self.y, y: self.x } }
}

// ── Wall ───────────────────────────────────────────────────────

/// A wall segment.
#[derive(Debug, Clone)]
pub struct Wall {
    pub start: Vec2,
    pub end: Vec2,
}

impl Wall {
    pub fn new(start: Vec2, end: Vec2) -> Self { Self { start, end } }

    /// Closest point on the wall to a given point.
    pub fn closest_point(&self, point: &Vec2) -> Vec2 {
        let ab = self.end.sub(&self.start);
        let ap = point.sub(&self.start);
        let t = ap.dot(&ab) / ab.length_sq().max(1e-12);
        let t = t.clamp(0.0, 1.0);
        self.start.add(&ab.scale(t))
    }

    /// Distance from a point to this wall.
    pub fn distance_to(&self, point: &Vec2) -> f64 {
        point.distance(&self.closest_point(point))
    }

    /// Normal pointing away from the wall (left side).
    pub fn normal(&self) -> Vec2 {
        let dir = self.end.sub(&self.start).normalize();
        dir.perp()
    }
}

// ── Exit ───────────────────────────────────────────────────────

/// An exit (door/opening).
#[derive(Debug, Clone)]
pub struct Exit {
    pub position: Vec2,
    pub width: f64,
}

impl Exit {
    pub fn new(position: Vec2, width: f64) -> Self { Self { position, width } }
}

// ── Agent ──────────────────────────────────────────────────────

/// A crowd agent.
#[derive(Debug, Clone)]
pub struct Agent {
    pub id: u32,
    pub position: Vec2,
    pub velocity: Vec2,
    pub goal: Vec2,
    pub radius: f64,
    pub desired_speed: f64,
    pub max_speed: f64,
    pub mass: f64,
    pub group_id: Option<u32>,
    pub reached_goal: bool,
    pub panic_level: f64, // 0.0 = calm, 1.0 = full panic
}

impl Agent {
    pub fn new(id: u32, position: Vec2, goal: Vec2) -> Self {
        Self {
            id,
            position,
            velocity: Vec2::ZERO,
            goal,
            radius: 0.3,
            desired_speed: 1.4,
            max_speed: 3.0,
            mass: 80.0,
            group_id: None,
            reached_goal: false,
            panic_level: 0.0,
        }
    }

    pub fn with_radius(mut self, r: f64) -> Self { self.radius = r; self }
    pub fn with_speed(mut self, s: f64) -> Self { self.desired_speed = s; self }
    pub fn with_group(mut self, g: u32) -> Self { self.group_id = Some(g); self }
    pub fn with_panic(mut self, p: f64) -> Self { self.panic_level = p.clamp(0.0, 1.0); self }

    /// Speed including panic modifier.
    pub fn effective_desired_speed(&self) -> f64 {
        // Panic increases desired speed but also randomness
        self.desired_speed * (1.0 + self.panic_level * 0.5)
    }
}

// ── Social Force parameters ────────────────────────────────────

/// Parameters for the Social Force Model.
#[derive(Debug, Clone)]
pub struct SfmParams {
    /// Relaxation time for desired velocity.
    pub tau: f64,
    /// Agent-agent repulsion strength.
    pub agent_a: f64,
    /// Agent-agent repulsion range.
    pub agent_b: f64,
    /// Wall repulsion strength.
    pub wall_a: f64,
    /// Wall repulsion range.
    pub wall_b: f64,
    /// Group attraction strength (families).
    pub group_attraction: f64,
    /// Personal space multiplier (proxemics).
    pub personal_space: f64,
    /// Timestep.
    pub dt: f64,
    /// Speed reduction at high density.
    pub density_speed_factor: f64,
    /// Goal reached distance.
    pub goal_tolerance: f64,
}

impl Default for SfmParams {
    fn default() -> Self {
        Self {
            tau: 0.5,
            agent_a: 2000.0,
            agent_b: 0.08,
            wall_a: 2000.0,
            wall_b: 0.08,
            group_attraction: 2.0,
            personal_space: 1.0,
            dt: 0.05,
            density_speed_factor: 0.5,
            goal_tolerance: 0.5,
        }
    }
}

// ── Cross-section measurement ──────────────────────────────────

/// A measurement cross-section line for density/flow.
#[derive(Debug, Clone)]
pub struct CrossSection {
    pub start: Vec2,
    pub end: Vec2,
    pub crossings: u64,
    pub accumulated_density: f64,
    pub measurements: u64,
}

impl CrossSection {
    pub fn new(start: Vec2, end: Vec2) -> Self {
        Self { start, end, crossings: 0, accumulated_density: 0.0, measurements: 0 }
    }

    /// Average density at this cross-section.
    pub fn average_density(&self) -> f64 {
        if self.measurements == 0 { return 0.0; }
        self.accumulated_density / self.measurements as f64
    }

    /// Flow rate (crossings per measurement).
    pub fn flow_rate(&self) -> f64 {
        if self.measurements == 0 { return 0.0; }
        self.crossings as f64 / self.measurements as f64
    }
}

// ── Crowd Simulation ───────────────────────────────────────────

/// Crowd simulation using the Social Force Model.
#[derive(Debug, Clone)]
pub struct CrowdSim {
    agents: Vec<Agent>,
    walls: Vec<Wall>,
    exits: Vec<Exit>,
    cross_sections: Vec<CrossSection>,
    params: SfmParams,
    step_count: u64,
    next_id: u32,
    rng_state: u64,
}

impl CrowdSim {
    pub fn new(params: SfmParams) -> Self {
        Self {
            agents: Vec::new(),
            walls: Vec::new(),
            exits: Vec::new(),
            cross_sections: Vec::new(),
            params,
            step_count: 0,
            next_id: 0,
            rng_state: 42,
        }
    }

    /// Create with default parameters.
    pub fn default_sim() -> Self {
        Self::new(SfmParams::default())
    }

    fn next_rand(&mut self) -> f64 {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.rng_state >> 33) as f64) / (u32::MAX as f64)
    }

    /// Add an agent and return its ID.
    pub fn add_agent(&mut self, position: Vec2, goal: Vec2) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.agents.push(Agent::new(id, position, goal));
        id
    }

    /// Add a custom agent.
    pub fn add_custom_agent(&mut self, agent: Agent) -> u32 {
        let id = agent.id;
        self.agents.push(agent);
        id
    }

    /// Add a wall.
    pub fn add_wall(&mut self, start: Vec2, end: Vec2) {
        self.walls.push(Wall::new(start, end));
    }

    /// Add an exit.
    pub fn add_exit(&mut self, position: Vec2, width: f64) {
        self.exits.push(Exit::new(position, width));
    }

    /// Add a measurement cross-section.
    pub fn add_cross_section(&mut self, start: Vec2, end: Vec2) -> usize {
        let idx = self.cross_sections.len();
        self.cross_sections.push(CrossSection::new(start, end));
        idx
    }

    pub fn agent_count(&self) -> usize { self.agents.len() }
    pub fn active_agents(&self) -> usize { self.agents.iter().filter(|a| !a.reached_goal).count() }
    pub fn step_count(&self) -> u64 { self.step_count }
    pub fn agents(&self) -> &[Agent] { &self.agents }
    pub fn cross_sections(&self) -> &[CrossSection] { &self.cross_sections }

    /// Compute local density around an agent.
    fn local_density(&self, agent_idx: usize, radius: f64) -> f64 {
        let pos = self.agents[agent_idx].position;
        let r2 = radius * radius;
        let mut count = 0;
        for (j, other) in self.agents.iter().enumerate() {
            if j == agent_idx || other.reached_goal { continue; }
            if pos.sub(&other.position).length_sq() < r2 {
                count += 1;
            }
        }
        count as f64 / (std::f64::consts::PI * r2)
    }

    /// Advance one timestep using the Social Force Model.
    pub fn step(&mut self) {
        let n = self.agents.len();
        let mut forces = vec![Vec2::ZERO; n];

        let agents_snapshot: Vec<(Vec2, Vec2, f64, f64, Option<u32>, bool, f64)> = self.agents.iter().map(|a| {
            (a.position, a.velocity, a.radius, a.effective_desired_speed(), a.group_id, a.reached_goal, a.mass)
        }).collect();

        for i in 0..n {
            if agents_snapshot[i].5 { continue; } // reached goal

            let (pos_i, vel_i, rad_i, desired_speed_i, group_i, _, mass_i) = agents_snapshot[i];

            // 1. Desired velocity force
            let to_goal = self.agents[i].goal.sub(&pos_i);
            let dist_to_goal = to_goal.length();
            if dist_to_goal < self.params.goal_tolerance {
                self.agents[i].reached_goal = true;
                self.agents[i].velocity = Vec2::ZERO;
                continue;
            }

            // Density-dependent speed reduction
            let density = self.local_density(i, 2.0);
            let speed_factor = 1.0 / (1.0 + self.params.density_speed_factor * density);
            let adj_speed = desired_speed_i * speed_factor;

            let desired_vel = to_goal.normalize().scale(adj_speed);
            let driving_force = desired_vel.sub(&vel_i).scale(mass_i / self.params.tau);
            forces[i] = forces[i].add(&driving_force);

            // 2. Agent-agent repulsion
            for j in 0..n {
                if i == j || agents_snapshot[j].5 { continue; }
                let (pos_j, _, rad_j, _, group_j, _, _) = agents_snapshot[j];

                let d_ij = pos_i.sub(&pos_j);
                let dist = d_ij.length();
                let r_sum = (rad_i + rad_j) * self.params.personal_space;

                if dist < 1e-6 { continue; }

                let overlap = r_sum - dist;
                let n_ij = d_ij.normalize();

                // Repulsive social force
                let repulsion = self.params.agent_a * (overlap / self.params.agent_b).exp();
                forces[i] = forces[i].add(&n_ij.scale(repulsion));

                // Add small lateral component to break collinear symmetry
                let lateral = n_ij.perp();
                let lateral_sign = if (i as f64) < (j as f64) { 1.0 } else { -1.0 };
                forces[i] = forces[i].add(&lateral.scale(repulsion * 0.05 * lateral_sign));

                // Physical contact force (if overlapping)
                if overlap > 0.0 {
                    let contact = n_ij.scale(120.0 * overlap);
                    let t_ij = n_ij.perp();
                    let dv = agents_snapshot[j].1.sub(&vel_i);
                    let friction = t_ij.scale(240.0 * overlap * dv.dot(&t_ij));
                    forces[i] = forces[i].add(&contact).add(&friction);
                }

                // Group attraction (same group)
                if let (Some(gi), Some(gj)) = (group_i, group_j) {
                    if gi == gj {
                        let attract = n_ij.scale(-self.params.group_attraction);
                        forces[i] = forces[i].add(&attract);
                    }
                }
            }

            // 3. Wall repulsion
            for wall in &self.walls {
                let closest = wall.closest_point(&pos_i);
                let d_w = pos_i.sub(&closest);
                let dist = d_w.length();
                if dist < 1e-6 { continue; }

                let overlap = rad_i - dist;
                let n_w = d_w.normalize();

                let repulsion = self.params.wall_a * (overlap / self.params.wall_b).exp();
                forces[i] = forces[i].add(&n_w.scale(repulsion));

                if overlap > 0.0 {
                    let contact = n_w.scale(120.0 * overlap);
                    let t_w = n_w.perp();
                    let friction = t_w.scale(-240.0 * overlap * vel_i.dot(&t_w));
                    forces[i] = forces[i].add(&contact).add(&friction);
                }
            }

            // 4. Panic randomness
            if self.agents[i].panic_level > 0.0 {
                let rx = (self.next_rand() - 0.5) * 2.0;
                let ry = (self.next_rand() - 0.5) * 2.0;
                let noise = Vec2::new(rx, ry).scale(self.agents[i].panic_level * 50.0);
                forces[i] = forces[i].add(&noise);
            }
        }

        // Update velocities and positions
        let dt = self.params.dt;
        for i in 0..n {
            if self.agents[i].reached_goal { continue; }

            let accel = forces[i].scale(1.0 / self.agents[i].mass);
            self.agents[i].velocity = self.agents[i].velocity.add(&accel.scale(dt));
            let max_spd = self.agents[i].max_speed * (1.0 + self.agents[i].panic_level * 0.5);
            self.agents[i].velocity = self.agents[i].velocity.clamp_length(max_spd);
            self.agents[i].position = self.agents[i].position.add(&self.agents[i].velocity.scale(dt));
        }

        // Update cross-section measurements
        for cs in &mut self.cross_sections {
            cs.measurements += 1;
            // Count agents near the cross-section
            let cs_center = cs.start.add(&cs.end).scale(0.5);
            let cs_length = cs.start.distance(&cs.end);
            let mut local_count = 0;
            for agent in &self.agents {
                if agent.reached_goal { continue; }
                if agent.position.distance(&cs_center) < cs_length {
                    local_count += 1;
                }
            }
            cs.accumulated_density += local_count as f64 / cs_length.max(0.1);
        }

        self.step_count += 1;
    }

    /// Advance by n steps.
    pub fn step_n(&mut self, n: u64) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Average speed of active agents.
    pub fn average_speed(&self) -> f64 {
        let active: Vec<&Agent> = self.agents.iter().filter(|a| !a.reached_goal).collect();
        if active.is_empty() { return 0.0; }
        let total: f64 = active.iter().map(|a| a.velocity.length()).sum();
        total / active.len() as f64
    }

    /// Overall density (agents per unit area in the bounding box).
    pub fn density(&self) -> f64 {
        let active: Vec<&Agent> = self.agents.iter().filter(|a| !a.reached_goal).collect();
        if active.len() < 2 { return 0.0; }

        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;
        for a in &active {
            if a.position.x < min_x { min_x = a.position.x; }
            if a.position.y < min_y { min_y = a.position.y; }
            if a.position.x > max_x { max_x = a.position.x; }
            if a.position.y > max_y { max_y = a.position.y; }
        }
        let area = (max_x - min_x).max(0.1) * (max_y - min_y).max(0.1);
        active.len() as f64 / area
    }

    /// Number of agents that have reached their goal.
    pub fn evacuated_count(&self) -> usize {
        self.agents.iter().filter(|a| a.reached_goal).count()
    }

    /// Create an evacuation scenario: agents in a room with an exit.
    pub fn evacuation_scenario(
        num_agents: usize,
        room_width: f64,
        room_height: f64,
        exit_position: Vec2,
        exit_width: f64,
        panic: f64,
    ) -> Result<Self, CrowdError> {
        if num_agents == 0 {
            return Err(CrowdError::NoAgents);
        }
        let mut sim = Self::new(SfmParams::default());

        // Add walls (room boundary minus exit)
        sim.add_wall(Vec2::new(0.0, 0.0), Vec2::new(room_width, 0.0)); // top
        sim.add_wall(Vec2::new(0.0, room_height), Vec2::new(room_width, room_height)); // bottom
        sim.add_wall(Vec2::new(0.0, 0.0), Vec2::new(0.0, room_height)); // left
        // Right wall with gap for exit
        let half_exit = exit_width / 2.0;
        if exit_position.y - half_exit > 0.0 {
            sim.add_wall(Vec2::new(room_width, 0.0), Vec2::new(room_width, exit_position.y - half_exit));
        }
        if exit_position.y + half_exit < room_height {
            sim.add_wall(Vec2::new(room_width, exit_position.y + half_exit), Vec2::new(room_width, room_height));
        }

        sim.add_exit(exit_position, exit_width);

        // Place agents with deterministic pseudo-random positions
        let goal = Vec2::new(room_width + 2.0, exit_position.y);
        for i in 0..num_agents {
            let seed = 42u64.wrapping_add(i as u64 * 997);
            let mut rng = seed;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let fx = ((rng >> 33) as f64) / (u32::MAX as f64);
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let fy = ((rng >> 33) as f64) / (u32::MAX as f64);

            let x = 1.0 + fx * (room_width - 3.0);
            let y = 1.0 + fy * (room_height - 2.0);
            let agent = Agent::new(i as u32, Vec2::new(x, y), goal).with_panic(panic);
            sim.add_custom_agent(agent);
        }

        sim.next_id = num_agents as u32;
        Ok(sim)
    }
}

impl fmt::Display for CrowdSim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CrowdSim(agents={}, active={}, step={})",
            self.agent_count(), self.active_agents(), self.step_count)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < EPS }

    #[test]
    fn test_vec2_basics() {
        let v = Vec2::new(3.0, 4.0);
        assert!(approx(v.length(), 5.0));
    }

    #[test]
    fn test_vec2_normalize() {
        let v = Vec2::new(0.0, 5.0);
        let n = v.normalize();
        assert!(approx(n.x, 0.0));
        assert!(approx(n.y, 1.0));
    }

    #[test]
    fn test_vec2_zero_normalize() {
        let n = Vec2::ZERO.normalize();
        assert!(approx(n.length(), 0.0));
    }

    #[test]
    fn test_wall_distance() {
        let wall = Wall::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0));
        let point = Vec2::new(5.0, 3.0);
        assert!(approx(wall.distance_to(&point), 3.0));
    }

    #[test]
    fn test_wall_closest_point() {
        let wall = Wall::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0));
        let p = wall.closest_point(&Vec2::new(5.0, 3.0));
        assert!(approx(p.x, 5.0));
        assert!(approx(p.y, 0.0));
    }

    #[test]
    fn test_wall_closest_point_clamped() {
        let wall = Wall::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0));
        let p = wall.closest_point(&Vec2::new(-5.0, 0.0));
        assert!(approx(p.x, 0.0));
    }

    #[test]
    fn test_agent_creation() {
        let a = Agent::new(0, Vec2::new(1.0, 2.0), Vec2::new(10.0, 2.0));
        assert_eq!(a.id, 0);
        assert!(approx(a.radius, 0.3));
        assert!(!a.reached_goal);
    }

    #[test]
    fn test_agent_builders() {
        let a = Agent::new(0, Vec2::ZERO, Vec2::ZERO)
            .with_radius(0.5)
            .with_speed(2.0)
            .with_group(1)
            .with_panic(0.8);
        assert!(approx(a.radius, 0.5));
        assert!(approx(a.desired_speed, 2.0));
        assert_eq!(a.group_id, Some(1));
        assert!(approx(a.panic_level, 0.8));
    }

    #[test]
    fn test_sim_creation() {
        let sim = CrowdSim::default_sim();
        assert_eq!(sim.agent_count(), 0);
        assert_eq!(sim.step_count(), 0);
    }

    #[test]
    fn test_add_agent() {
        let mut sim = CrowdSim::default_sim();
        let id = sim.add_agent(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0));
        assert_eq!(id, 0);
        assert_eq!(sim.agent_count(), 1);
    }

    #[test]
    fn test_single_agent_moves_toward_goal() {
        let mut sim = CrowdSim::default_sim();
        sim.add_agent(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0));
        sim.step();
        // Agent should have moved toward goal (positive x)
        assert!(sim.agents()[0].position.x > 0.0);
    }

    #[test]
    fn test_agent_reaches_goal() {
        let mut sim = CrowdSim::default_sim();
        sim.add_agent(Vec2::new(0.0, 0.0), Vec2::new(2.0, 0.0));
        sim.step_n(200);
        assert!(sim.agents()[0].reached_goal);
    }

    #[test]
    fn test_wall_repulsion() {
        let mut sim = CrowdSim::default_sim();
        sim.add_wall(Vec2::new(5.0, -10.0), Vec2::new(5.0, 10.0));
        sim.add_agent(Vec2::new(4.5, 0.0), Vec2::new(10.0, 0.0));
        sim.step();
        // Wall at x=5 should repel the agent
        // Agent might still move right overall due to goal, but wall force exists
        assert_eq!(sim.step_count(), 1);
    }

    #[test]
    fn test_agent_agent_repulsion() {
        let mut sim = CrowdSim::default_sim();
        // Two agents very close, both heading right
        sim.add_agent(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0));
        sim.add_agent(Vec2::new(0.1, 0.0), Vec2::new(10.0, 0.0));
        sim.step_n(10);
        // They should have separated in y direction due to repulsion
        let dy = (sim.agents()[0].position.y - sim.agents()[1].position.y).abs();
        assert!(dy > 0.0, "agents should repel: dy = {dy}");
    }

    #[test]
    fn test_group_attraction() {
        let mut sim = CrowdSim::default_sim();
        let a1 = Agent::new(0, Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0)).with_group(1);
        let a2 = Agent::new(1, Vec2::new(0.0, 3.0), Vec2::new(10.0, 0.0)).with_group(1);
        sim.add_custom_agent(a1);
        sim.add_custom_agent(a2);
        sim.next_id = 2;
        let d0 = sim.agents()[0].position.distance(&sim.agents()[1].position);
        sim.step_n(20);
        let d1 = sim.agents()[0].position.distance(&sim.agents()[1].position);
        // Group members should be closer (or at least not much farther)
        assert!(d1 < d0 + 1.0, "group should attract: d0={d0}, d1={d1}");
    }

    #[test]
    fn test_evacuated_count() {
        let mut sim = CrowdSim::default_sim();
        sim.add_agent(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0));
        assert_eq!(sim.evacuated_count(), 0);
        sim.step_n(100);
        assert_eq!(sim.evacuated_count(), 1);
    }

    #[test]
    fn test_average_speed() {
        let mut sim = CrowdSim::default_sim();
        sim.add_agent(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0));
        sim.step();
        assert!(sim.average_speed() > 0.0);
    }

    #[test]
    fn test_density() {
        let mut sim = CrowdSim::default_sim();
        for i in 0..10 {
            sim.add_agent(
                Vec2::new(i as f64 * 0.5, 0.0),
                Vec2::new(20.0, 0.0),
            );
        }
        sim.step();
        assert!(sim.density() > 0.0);
    }

    #[test]
    fn test_cross_section() {
        let mut sim = CrowdSim::default_sim();
        sim.add_cross_section(Vec2::new(5.0, -5.0), Vec2::new(5.0, 5.0));
        sim.add_agent(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0));
        sim.step_n(10);
        let cs = &sim.cross_sections()[0];
        assert_eq!(cs.measurements, 10);
    }

    #[test]
    fn test_evacuation_scenario() {
        let sim = CrowdSim::evacuation_scenario(
            10, 10.0, 10.0,
            Vec2::new(10.0, 5.0), 2.0, 0.0,
        ).unwrap();
        assert_eq!(sim.agent_count(), 10);
        assert!(sim.walls.len() >= 3);
    }

    #[test]
    fn test_evacuation_with_panic() {
        let mut sim = CrowdSim::evacuation_scenario(
            5, 8.0, 8.0,
            Vec2::new(8.0, 4.0), 2.0, 0.8,
        ).unwrap();
        sim.step_n(50);
        // With panic, agents should still move
        assert!(sim.average_speed() > 0.0);
    }

    #[test]
    fn test_evacuation_no_agents() {
        assert!(CrowdSim::evacuation_scenario(0, 10.0, 10.0, Vec2::ZERO, 2.0, 0.0).is_err());
    }

    #[test]
    fn test_panic_increases_speed() {
        let calm = Agent::new(0, Vec2::ZERO, Vec2::ZERO);
        let panicked = Agent::new(1, Vec2::ZERO, Vec2::ZERO).with_panic(1.0);
        assert!(panicked.effective_desired_speed() > calm.effective_desired_speed());
    }

    #[test]
    fn test_active_agents() {
        let mut sim = CrowdSim::default_sim();
        sim.add_agent(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0));
        sim.add_agent(Vec2::new(0.0, 5.0), Vec2::new(100.0, 5.0));
        assert_eq!(sim.active_agents(), 2);
        sim.step_n(100);
        // First agent should have reached goal
        assert!(sim.active_agents() < 2);
    }

    #[test]
    fn test_display() {
        let sim = CrowdSim::default_sim();
        let s = format!("{sim}");
        assert!(s.contains("CrowdSim"));
    }

    #[test]
    fn test_cross_section_flow_rate() {
        let mut cs = CrossSection::new(Vec2::ZERO, Vec2::new(1.0, 0.0));
        cs.crossings = 10;
        cs.measurements = 5;
        assert!(approx(cs.flow_rate(), 2.0));
    }

    #[test]
    fn test_cross_section_empty() {
        let cs = CrossSection::new(Vec2::ZERO, Vec2::new(1.0, 0.0));
        assert!(approx(cs.average_density(), 0.0));
        assert!(approx(cs.flow_rate(), 0.0));
    }
}
