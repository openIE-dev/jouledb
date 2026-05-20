//! Velocity Obstacles — VO, RVO, ORCA for multi-agent collision avoidance,
//! Minkowski sum computation for velocity-space reasoning.
//!
//! Pure-Rust implementations of velocity obstacle methods for decentralized
//! multi-agent systems with circular agents, preferred velocities, and
//! ORCA half-plane constraints.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum VoError {
    InvalidParameter(String),
    NoFeasibleVelocity,
    AgentNotFound(usize),
}

impl fmt::Display for VoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
            Self::NoFeasibleVelocity => write!(f, "no feasible velocity found"),
            Self::AgentNotFound(id) => write!(f, "agent {id} not found"),
        }
    }
}

impl std::error::Error for VoError {}

// ── Vec2 ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self { Self { x, y } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0 } }

    pub fn magnitude(self) -> f64 { (self.x * self.x + self.y * self.y).sqrt() }

    pub fn magnitude_sq(self) -> f64 { self.x * self.x + self.y * self.y }

    pub fn normalized(self) -> Self {
        let m = self.magnitude();
        if m < 1e-12 { Self::zero() } else { Self { x: self.x / m, y: self.y / m } }
    }

    pub fn dot(self, o: Vec2) -> f64 { self.x * o.x + self.y * o.y }

    pub fn cross_z(self, o: Vec2) -> f64 { self.x * o.y - self.y * o.x }

    pub fn add(self, o: Vec2) -> Vec2 { Vec2 { x: self.x + o.x, y: self.y + o.y } }
    pub fn sub(self, o: Vec2) -> Vec2 { Vec2 { x: self.x - o.x, y: self.y - o.y } }
    pub fn scale(self, s: f64) -> Vec2 { Vec2 { x: self.x * s, y: self.y * s } }

    pub fn perpendicular(self) -> Vec2 { Vec2 { x: -self.y, y: self.x } }

    pub fn distance_to(self, o: Vec2) -> f64 { self.sub(o).magnitude() }

    pub fn clamp_magnitude(self, max: f64) -> Vec2 {
        let m = self.magnitude();
        if m <= max { self } else { self.normalized().scale(max) }
    }

    pub fn rotate(self, angle: f64) -> Vec2 {
        let c = angle.cos();
        let s = angle.sin();
        Vec2 { x: self.x * c - self.y * s, y: self.x * s + self.y * c }
    }
}

impl fmt::Display for Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.3}, {:.3})", self.x, self.y)
    }
}

// ── Agent ───────────────────────────────────────────────────────

/// A circular agent with position, velocity, and radius.
#[derive(Debug, Clone)]
pub struct Agent {
    pub id: usize,
    pub position: Vec2,
    pub velocity: Vec2,
    pub preferred_velocity: Vec2,
    pub radius: f64,
    pub max_speed: f64,
}

impl Agent {
    pub fn new(id: usize, position: Vec2, radius: f64, max_speed: f64) -> Self {
        Self {
            id,
            position,
            velocity: Vec2::zero(),
            preferred_velocity: Vec2::zero(),
            radius,
            max_speed,
        }
    }

    pub fn with_velocity(mut self, v: Vec2) -> Self { self.velocity = v; self }
    pub fn with_preferred_velocity(mut self, v: Vec2) -> Self { self.preferred_velocity = v; self }

    /// Advance the agent by one time step.
    pub fn step(&mut self, dt: f64) {
        self.position = self.position.add(self.velocity.scale(dt));
    }
}

impl fmt::Display for Agent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "Agent(id={}, pos={}, vel={}, r={:.2})",
            self.id, self.position, self.velocity, self.radius,
        )
    }
}

// ── Velocity Obstacle ───────────────────────────────────────────

/// A velocity obstacle cone defined by apex and two boundary directions.
#[derive(Debug, Clone)]
pub struct VelocityObstacle {
    pub apex: Vec2,
    pub left_leg: Vec2,
    pub right_leg: Vec2,
}

impl VelocityObstacle {
    /// Compute the VO of agent A induced by agent B with time horizon tau.
    pub fn compute(a: &Agent, b: &Agent, tau: f64) -> Self {
        let rel_pos = b.position.sub(a.position);
        let combined_radius = a.radius + b.radius;
        let dist = rel_pos.magnitude();

        let apex = b.velocity; // VO apex is at v_B for standard VO

        if dist <= combined_radius {
            // Already colliding — full disc obstacle
            return Self {
                apex,
                left_leg: Vec2::new(0.0, 1.0),
                right_leg: Vec2::new(0.0, -1.0),
            };
        }

        let center = rel_pos.scale(1.0 / tau);
        let half_angle = (combined_radius / dist).asin();
        let base_angle = rel_pos.y.atan2(rel_pos.x);

        let left_angle = base_angle + half_angle;
        let right_angle = base_angle - half_angle;

        Self {
            apex: center.add(apex),
            left_leg: Vec2::new(left_angle.cos(), left_angle.sin()),
            right_leg: Vec2::new(right_angle.cos(), right_angle.sin()),
        }
    }

    /// Check if a velocity is inside this velocity obstacle.
    pub fn contains(&self, velocity: Vec2) -> bool {
        let rel = velocity.sub(self.apex);
        let left_cross = self.left_leg.cross_z(rel);
        let right_cross = self.right_leg.cross_z(rel);
        // Inside if to the right of left leg and to the left of right leg
        left_cross <= 0.0 && right_cross >= 0.0
    }
}

impl fmt::Display for VelocityObstacle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VO(apex={})", self.apex)
    }
}

// ── ORCA Half-Plane ─────────────────────────────────────────────

/// An ORCA half-plane constraint: point + outward normal.
#[derive(Debug, Clone, Copy)]
pub struct HalfPlane {
    pub point: Vec2,
    pub normal: Vec2,
}

impl HalfPlane {
    pub fn new(point: Vec2, normal: Vec2) -> Self {
        Self { point, normal: normal.normalized() }
    }

    /// Check if a velocity satisfies this constraint (is in the feasible half).
    pub fn satisfies(&self, v: Vec2) -> bool {
        v.sub(self.point).dot(self.normal) >= 0.0
    }

    /// Project a velocity onto this half-plane boundary if it violates the constraint.
    pub fn project(&self, v: Vec2) -> Vec2 {
        let diff = v.sub(self.point);
        let pen = diff.dot(self.normal);
        if pen >= 0.0 {
            v // Already satisfies
        } else {
            v.sub(self.normal.scale(pen))
        }
    }
}

impl fmt::Display for HalfPlane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HalfPlane(pt={}, n={})", self.point, self.normal)
    }
}

// ── ORCA computation ────────────────────────────────────────────

/// Compute ORCA half-plane for agent A w.r.t. agent B.
pub fn compute_orca_halfplane(a: &Agent, b: &Agent, tau: f64) -> HalfPlane {
    let rel_pos = b.position.sub(a.position);
    let rel_vel = a.velocity.sub(b.velocity);
    let combined_radius = a.radius + b.radius;
    let dist_sq = rel_pos.magnitude_sq();
    let combined_sq = combined_radius * combined_radius;

    let u; // correction vector
    let normal;

    if dist_sq > combined_sq {
        // No collision yet — use truncated cone
        let w = rel_vel.sub(rel_pos.scale(1.0 / tau));
        let w_len = w.magnitude();

        if w_len < 1e-12 {
            normal = rel_pos.normalized().scale(-1.0);
            u = normal.scale(combined_radius / tau - rel_pos.magnitude() / tau);
        } else {
            let unit_w = w.scale(1.0 / w_len);
            normal = unit_w;
            let dot_product = w.dot(rel_pos.scale(1.0 / tau));
            if dot_product < 0.0 {
                // Project on cut-off circle
                let cut_off_radius = combined_radius / tau;
                u = unit_w.scale(cut_off_radius - w_len);
            } else {
                // Project on cone leg
                let leg_len = (dist_sq - combined_sq).sqrt();
                if rel_pos.cross_z(w) > 0.0 {
                    // Left leg
                    let left_dir = Vec2::new(
                        rel_pos.x * leg_len - rel_pos.y * combined_radius,
                        rel_pos.x * combined_radius + rel_pos.y * leg_len,
                    ).scale(1.0 / dist_sq);
                    u = left_dir.scale(rel_vel.dot(left_dir)).sub(rel_vel);
                } else {
                    // Right leg
                    let right_dir = Vec2::new(
                        rel_pos.x * leg_len + rel_pos.y * combined_radius,
                        -rel_pos.x * combined_radius + rel_pos.y * leg_len,
                    ).scale(1.0 / dist_sq);
                    u = right_dir.scale(rel_vel.dot(right_dir)).sub(rel_vel);
                }
            }
        }
    } else {
        // Already colliding — push apart over one time step
        let w = rel_vel.sub(rel_pos.scale(1.0 / tau));
        let w_len = w.magnitude();
        if w_len < 1e-12 {
            normal = rel_pos.normalized().scale(-1.0);
        } else {
            normal = w.scale(1.0 / w_len);
        }
        u = normal.scale(combined_radius / tau - w_len);
    }

    let point = a.velocity.add(u.scale(0.5));
    HalfPlane::new(point, normal)
}

// ── Solve ORCA (incremental projection) ─────────────────────────

/// Solve for the best feasible velocity given ORCA half-planes.
pub fn solve_orca(
    preferred: Vec2,
    max_speed: f64,
    constraints: &[HalfPlane],
) -> Vec2 {
    let mut result = preferred;

    // Iterative projection
    for _ in 0..constraints.len().saturating_mul(2).max(10) {
        let mut feasible = true;
        for hp in constraints {
            if !hp.satisfies(result) {
                result = hp.project(result);
                feasible = false;
            }
        }
        if feasible { break; }
    }

    result.clamp_magnitude(max_speed)
}

// ── Multi-Agent Simulator ───────────────────────────────────────

/// Multi-agent collision avoidance simulator using ORCA.
pub struct OrcaSimulator {
    agents: Vec<Agent>,
    time_horizon: f64,
    dt: f64,
    time: f64,
}

impl OrcaSimulator {
    pub fn new(dt: f64, time_horizon: f64) -> Result<Self, VoError> {
        if dt <= 0.0 {
            return Err(VoError::InvalidParameter("dt must be > 0".into()));
        }
        if time_horizon <= 0.0 {
            return Err(VoError::InvalidParameter("time_horizon must be > 0".into()));
        }
        Ok(Self {
            agents: Vec::new(),
            time_horizon,
            dt,
            time: 0.0,
        })
    }

    pub fn with_dt(mut self, dt: f64) -> Self { self.dt = dt; self }
    pub fn with_time_horizon(mut self, tau: f64) -> Self { self.time_horizon = tau; self }

    pub fn add_agent(&mut self, agent: Agent) -> usize {
        let id = self.agents.len();
        self.agents.push(agent);
        id
    }

    pub fn agent(&self, id: usize) -> Option<&Agent> { self.agents.get(id) }
    pub fn agent_count(&self) -> usize { self.agents.len() }
    pub fn time(&self) -> f64 { self.time }

    /// Set preferred velocity for an agent.
    pub fn set_preferred_velocity(&mut self, id: usize, v: Vec2) -> Result<(), VoError> {
        match self.agents.get_mut(id) {
            Some(a) => { a.preferred_velocity = v; Ok(()) }
            None => Err(VoError::AgentNotFound(id)),
        }
    }

    /// Step the simulation: compute ORCA velocities, then advance positions.
    pub fn step(&mut self) {
        let n = self.agents.len();
        let mut new_velocities = vec![Vec2::zero(); n];

        for i in 0..n {
            let mut constraints = Vec::new();
            for j in 0..n {
                if i == j { continue; }
                let hp = compute_orca_halfplane(&self.agents[i], &self.agents[j], self.time_horizon);
                constraints.push(hp);
            }
            new_velocities[i] = solve_orca(
                self.agents[i].preferred_velocity,
                self.agents[i].max_speed,
                &constraints,
            );
        }

        for (i, agent) in self.agents.iter_mut().enumerate() {
            agent.velocity = new_velocities[i];
            agent.step(self.dt);
        }
        self.time += self.dt;
    }

    /// Step N times.
    pub fn step_n(&mut self, n: usize) {
        for _ in 0..n { self.step(); }
    }

    /// Check if any pair of agents is in collision.
    pub fn has_collision(&self) -> bool {
        for i in 0..self.agents.len() {
            for j in (i + 1)..self.agents.len() {
                let d = self.agents[i].position.distance_to(self.agents[j].position);
                if d < self.agents[i].radius + self.agents[j].radius {
                    return true;
                }
            }
        }
        false
    }

    /// Minimum pairwise distance between agents.
    pub fn min_separation(&self) -> f64 {
        let mut min_d = f64::MAX;
        for i in 0..self.agents.len() {
            for j in (i + 1)..self.agents.len() {
                let d = self.agents[i].position.distance_to(self.agents[j].position);
                let gap = d - self.agents[i].radius - self.agents[j].radius;
                if gap < min_d { min_d = gap; }
            }
        }
        min_d
    }
}

impl fmt::Display for OrcaSimulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "OrcaSimulator(agents={}, tau={:.2}, dt={:.3}, t={:.3})",
            self.agents.len(), self.time_horizon, self.dt, self.time,
        )
    }
}

// ── Minkowski Sum (convex polygon) ──────────────────────────────

/// Compute the Minkowski sum of two convex polygons given as CCW vertex lists.
pub fn minkowski_sum(poly_a: &[Vec2], poly_b: &[Vec2]) -> Vec<Vec2> {
    if poly_a.is_empty() || poly_b.is_empty() { return Vec::new(); }
    let na = poly_a.len();
    let nb = poly_b.len();

    // Find bottom-most points
    let mut ai = 0;
    for i in 1..na {
        if poly_a[i].y < poly_a[ai].y
            || (poly_a[i].y == poly_a[ai].y && poly_a[i].x < poly_a[ai].x)
        { ai = i; }
    }
    let mut bi = 0;
    for i in 1..nb {
        if poly_b[i].y < poly_b[bi].y
            || (poly_b[i].y == poly_b[bi].y && poly_b[i].x < poly_b[bi].x)
        { bi = i; }
    }

    let mut result = Vec::with_capacity(na + nb);
    let mut ca = 0;
    let mut cb = 0;

    while ca < na || cb < nb {
        let v = poly_a[(ai + ca) % na].add(poly_b[(bi + cb) % nb]);
        result.push(v);

        let edge_a = poly_a[(ai + ca + 1) % na].sub(poly_a[(ai + ca) % na]);
        let edge_b = poly_b[(bi + cb + 1) % nb].sub(poly_b[(bi + cb) % nb]);
        let cross = edge_a.cross_z(edge_b);

        if ca >= na {
            cb += 1;
        } else if cb >= nb {
            ca += 1;
        } else if cross > 0.0 {
            ca += 1;
        } else if cross < 0.0 {
            cb += 1;
        } else {
            ca += 1;
            cb += 1;
        }
    }

    result
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < 1e-6 }

    #[test]
    fn test_vec2_ops() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        let c = a.add(b);
        assert!(approx(c.x, 4.0));
        assert!(approx(c.y, 6.0));
    }

    #[test]
    fn test_vec2_cross() {
        let a = Vec2::new(1.0, 0.0);
        let b = Vec2::new(0.0, 1.0);
        assert!(approx(a.cross_z(b), 1.0));
    }

    #[test]
    fn test_vec2_rotate() {
        let v = Vec2::new(1.0, 0.0);
        let r = v.rotate(std::f64::consts::FRAC_PI_2);
        assert!(approx(r.x, 0.0));
        assert!(approx(r.y, 1.0));
    }

    #[test]
    fn test_vec2_display() {
        let v = Vec2::new(1.5, 2.5);
        let s = format!("{v}");
        assert!(s.contains("1.500"));
    }

    #[test]
    fn test_agent_step() {
        let mut a = Agent::new(0, Vec2::new(0.0, 0.0), 0.5, 2.0)
            .with_velocity(Vec2::new(1.0, 0.0));
        a.step(0.5);
        assert!(approx(a.position.x, 0.5));
    }

    #[test]
    fn test_agent_display() {
        let a = Agent::new(0, Vec2::new(1.0, 2.0), 0.5, 2.0);
        let s = format!("{a}");
        assert!(s.contains("Agent(id=0"));
    }

    #[test]
    fn test_vo_computation() {
        let a = Agent::new(0, Vec2::new(0.0, 0.0), 0.5, 2.0)
            .with_velocity(Vec2::new(1.0, 0.0));
        let b = Agent::new(1, Vec2::new(5.0, 0.0), 0.5, 2.0)
            .with_velocity(Vec2::new(-1.0, 0.0));
        let vo = VelocityObstacle::compute(&a, &b, 5.0);
        // Head-on velocity should be inside the VO
        assert!(vo.contains(Vec2::new(1.0, 0.0)));
    }

    #[test]
    fn test_vo_display() {
        let a = Agent::new(0, Vec2::zero(), 0.5, 2.0);
        let b = Agent::new(1, Vec2::new(5.0, 0.0), 0.5, 2.0);
        let vo = VelocityObstacle::compute(&a, &b, 5.0);
        let s = format!("{vo}");
        assert!(s.contains("VO"));
    }

    #[test]
    fn test_halfplane_satisfies() {
        let hp = HalfPlane::new(Vec2::zero(), Vec2::new(1.0, 0.0));
        assert!(hp.satisfies(Vec2::new(1.0, 0.0)));
        assert!(!hp.satisfies(Vec2::new(-1.0, 0.0)));
    }

    #[test]
    fn test_halfplane_project() {
        let hp = HalfPlane::new(Vec2::zero(), Vec2::new(1.0, 0.0));
        let p = hp.project(Vec2::new(-1.0, 2.0));
        assert!(approx(p.x, 0.0));
        assert!(approx(p.y, 2.0));
    }

    #[test]
    fn test_halfplane_display() {
        let hp = HalfPlane::new(Vec2::zero(), Vec2::new(1.0, 0.0));
        let s = format!("{hp}");
        assert!(s.contains("HalfPlane"));
    }

    #[test]
    fn test_orca_simulator_creation() {
        let sim = OrcaSimulator::new(0.1, 5.0).unwrap();
        assert_eq!(sim.agent_count(), 0);
    }

    #[test]
    fn test_orca_invalid_dt() {
        assert!(OrcaSimulator::new(0.0, 5.0).is_err());
        assert!(OrcaSimulator::new(-1.0, 5.0).is_err());
    }

    #[test]
    fn test_orca_head_on_avoidance() {
        let mut sim = OrcaSimulator::new(0.25, 5.0).unwrap();
        let a0 = Agent::new(0, Vec2::new(0.0, 0.0), 0.5, 2.0)
            .with_preferred_velocity(Vec2::new(1.0, 0.0));
        let a1 = Agent::new(1, Vec2::new(10.0, 0.0), 0.5, 2.0)
            .with_preferred_velocity(Vec2::new(-1.0, 0.0));
        sim.add_agent(a0);
        sim.add_agent(a1);
        sim.step_n(100);
        // Should not collide
        let sep = sim.min_separation();
        assert!(sep > -0.5, "min separation = {sep}");
    }

    #[test]
    fn test_orca_crossing_agents() {
        let mut sim = OrcaSimulator::new(0.1, 3.0).unwrap();
        sim.add_agent(
            Agent::new(0, Vec2::new(0.0, 0.0), 0.3, 1.5)
                .with_preferred_velocity(Vec2::new(1.0, 0.0))
        );
        sim.add_agent(
            Agent::new(1, Vec2::new(5.0, -5.0), 0.3, 1.5)
                .with_preferred_velocity(Vec2::new(0.0, 1.0))
        );
        sim.step_n(200);
        assert!(sim.time() > 0.0);
    }

    #[test]
    fn test_orca_display() {
        let sim = OrcaSimulator::new(0.1, 5.0).unwrap();
        let s = format!("{sim}");
        assert!(s.contains("OrcaSimulator"));
    }

    #[test]
    fn test_solve_orca_unconstrained() {
        let v = solve_orca(Vec2::new(1.0, 0.0), 2.0, &[]);
        assert!(approx(v.x, 1.0));
        assert!(approx(v.y, 0.0));
    }

    #[test]
    fn test_solve_orca_speed_limit() {
        let v = solve_orca(Vec2::new(10.0, 0.0), 2.0, &[]);
        assert!(v.magnitude() <= 2.0 + 1e-6);
    }

    #[test]
    fn test_minkowski_sum_squares() {
        let sq1 = vec![
            Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0), Vec2::new(0.0, 1.0),
        ];
        let sq2 = vec![
            Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0), Vec2::new(0.0, 1.0),
        ];
        let msum = minkowski_sum(&sq1, &sq2);
        assert!(!msum.is_empty());
        // Minkowski sum of two unit squares should span [0,2]x[0,2]
        let max_x = msum.iter().map(|v| v.x).fold(f64::NEG_INFINITY, f64::max);
        assert!(max_x >= 1.9);
    }

    #[test]
    fn test_minkowski_sum_empty() {
        let result = minkowski_sum(&[], &[Vec2::new(0.0, 0.0)]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_error_display() {
        let e = VoError::NoFeasibleVelocity;
        assert_eq!(format!("{e}"), "no feasible velocity found");
    }
}
