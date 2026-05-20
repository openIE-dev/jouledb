//! Formation control — leader-follower, virtual structure, behavior-based,
//! consensus formation, and shape maintenance for multi-robot systems.
//!
//! Pure-Rust formation controllers with realistic kinematics and
//! convergence analysis. Supports 2D planar formations with arbitrary
//! topologies defined via adjacency graphs.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Formation control errors.
#[derive(Debug, Clone, PartialEq)]
pub enum FormationError {
    /// Robot ID not found in the formation.
    RobotNotFound(u64),
    /// Duplicate robot ID.
    DuplicateRobot(u64),
    /// Invalid formation specification.
    InvalidFormation(String),
    /// Convergence failure.
    ConvergenceFailed(String),
    /// No leader assigned.
    NoLeader,
}

impl fmt::Display for FormationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RobotNotFound(id) => write!(f, "robot not found: {id}"),
            Self::DuplicateRobot(id) => write!(f, "duplicate robot: {id}"),
            Self::InvalidFormation(msg) => write!(f, "invalid formation: {msg}"),
            Self::ConvergenceFailed(msg) => write!(f, "convergence failed: {msg}"),
            Self::NoLeader => write!(f, "no leader assigned"),
        }
    }
}

impl std::error::Error for FormationError {}

// ── Vec2 ────────────────────────────────────────────────────────

/// 2D vector for positions and velocities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn normalized(&self) -> Self {
        let n = self.norm();
        if n < 1e-12 {
            Self::zero()
        } else {
            Self { x: self.x / n, y: self.y / n }
        }
    }

    pub fn dist(&self, other: &Vec2) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    pub fn add(&self, other: &Vec2) -> Vec2 {
        Vec2 { x: self.x + other.x, y: self.y + other.y }
    }

    pub fn sub(&self, other: &Vec2) -> Vec2 {
        Vec2 { x: self.x - other.x, y: self.y - other.y }
    }

    pub fn scale(&self, s: f64) -> Vec2 {
        Vec2 { x: self.x * s, y: self.y * s }
    }

    pub fn dot(&self, other: &Vec2) -> f64 {
        self.x * other.x + self.y * other.y
    }
}

impl fmt::Display for Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.3}, {:.3})", self.x, self.y)
    }
}

// ── Formation Strategy ──────────────────────────────────────────

/// The formation control strategy to employ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormationStrategy {
    /// Leader-follower: one leader, followers maintain offsets.
    LeaderFollower,
    /// Virtual structure: entire formation treated as rigid body.
    VirtualStructure,
    /// Behavior-based: weighted sum of separation, cohesion, alignment.
    BehaviorBased,
    /// Consensus-based: distributed agreement on desired offsets.
    Consensus,
}

impl fmt::Display for FormationStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LeaderFollower => write!(f, "LeaderFollower"),
            Self::VirtualStructure => write!(f, "VirtualStructure"),
            Self::BehaviorBased => write!(f, "BehaviorBased"),
            Self::Consensus => write!(f, "Consensus"),
        }
    }
}

// ── Robot State ─────────────────────────────────────────────────

/// State of a single robot in the formation.
#[derive(Debug, Clone)]
pub struct RobotState {
    pub id: u64,
    pub position: Vec2,
    pub velocity: Vec2,
    pub heading: f64,
    pub max_speed: f64,
    pub is_leader: bool,
}

impl RobotState {
    pub fn new(id: u64, position: Vec2) -> Self {
        Self {
            id,
            position,
            velocity: Vec2::zero(),
            heading: 0.0,
            max_speed: 1.0,
            is_leader: false,
        }
    }

    pub fn with_max_speed(mut self, speed: f64) -> Self {
        self.max_speed = speed.abs();
        self
    }

    pub fn with_heading(mut self, heading: f64) -> Self {
        self.heading = heading;
        self
    }

    pub fn with_leader(mut self, is_leader: bool) -> Self {
        self.is_leader = is_leader;
        self
    }

    /// Apply velocity for a time step, clamping to max_speed.
    pub fn step(&mut self, dt: f64) {
        let speed = self.velocity.norm();
        if speed > self.max_speed {
            self.velocity = self.velocity.normalized().scale(self.max_speed);
        }
        self.position = self.position.add(&self.velocity.scale(dt));
        if speed > 1e-12 {
            self.heading = self.velocity.y.atan2(self.velocity.x);
        }
    }
}

impl fmt::Display for RobotState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let role = if self.is_leader { "leader" } else { "follower" };
        write!(f, "Robot({}, pos={}, {})", self.id, self.position, role)
    }
}

// ── Desired Shape ───────────────────────────────────────────────

/// Desired offsets relative to the formation center (or leader).
#[derive(Debug, Clone)]
pub struct DesiredShape {
    /// Map from robot ID to desired offset from reference point.
    pub offsets: HashMap<u64, Vec2>,
}

impl DesiredShape {
    pub fn new() -> Self {
        Self { offsets: HashMap::new() }
    }

    pub fn with_offset(mut self, robot_id: u64, offset: Vec2) -> Self {
        self.offsets.insert(robot_id, offset);
        self
    }

    /// Create a regular polygon formation around the origin.
    pub fn regular_polygon(robot_ids: &[u64], radius: f64) -> Self {
        let n = robot_ids.len() as f64;
        let mut offsets = HashMap::new();
        for (i, &id) in robot_ids.iter().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / n;
            offsets.insert(id, Vec2::new(radius * angle.cos(), radius * angle.sin()));
        }
        Self { offsets }
    }

    /// Create a line formation along the x-axis.
    pub fn line(robot_ids: &[u64], spacing: f64) -> Self {
        let mut offsets = HashMap::new();
        let half = (robot_ids.len() as f64 - 1.0) * spacing / 2.0;
        for (i, &id) in robot_ids.iter().enumerate() {
            offsets.insert(id, Vec2::new(i as f64 * spacing - half, 0.0));
        }
        Self { offsets }
    }

    /// Create a V-formation (wedge).
    pub fn wedge(robot_ids: &[u64], arm_spacing: f64, arm_angle_deg: f64) -> Self {
        let mut offsets = HashMap::new();
        let angle = arm_angle_deg.to_radians();
        if let Some(&leader_id) = robot_ids.first() {
            offsets.insert(leader_id, Vec2::zero());
            let mut left_idx = 0usize;
            let mut right_idx = 0usize;
            for (i, &id) in robot_ids.iter().enumerate().skip(1) {
                if i % 2 == 1 {
                    left_idx += 1;
                    let d = left_idx as f64 * arm_spacing;
                    offsets.insert(id, Vec2::new(-d * angle.cos(), d * angle.sin()));
                } else {
                    right_idx += 1;
                    let d = right_idx as f64 * arm_spacing;
                    offsets.insert(id, Vec2::new(-d * angle.cos(), -d * angle.sin()));
                }
            }
        }
        Self { offsets }
    }

    /// Compute the centroid of the desired shape.
    pub fn centroid(&self) -> Vec2 {
        if self.offsets.is_empty() {
            return Vec2::zero();
        }
        let mut sum = Vec2::zero();
        for off in self.offsets.values() {
            sum = sum.add(off);
        }
        sum.scale(1.0 / self.offsets.len() as f64)
    }
}

impl fmt::Display for DesiredShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DesiredShape({} robots)", self.offsets.len())
    }
}

// ── Formation Controller ────────────────────────────────────────

/// Top-level formation controller managing a fleet of robots.
#[derive(Debug, Clone)]
pub struct FormationController {
    pub robots: HashMap<u64, RobotState>,
    pub desired_shape: DesiredShape,
    pub strategy: FormationStrategy,
    /// Proportional gain for position control.
    pub kp: f64,
    /// Damping gain for velocity feedback.
    pub kd: f64,
    /// Separation weight (behavior-based).
    pub separation_weight: f64,
    /// Cohesion weight (behavior-based).
    pub cohesion_weight: f64,
    /// Alignment weight (behavior-based).
    pub alignment_weight: f64,
    /// Separation distance threshold.
    pub separation_dist: f64,
    /// Consensus gain (for consensus strategy).
    pub consensus_gain: f64,
    /// Adjacency list for communication topology.
    pub neighbors: HashMap<u64, Vec<u64>>,
    /// Leader waypoints (x, y) for the leader to follow.
    pub leader_waypoints: Vec<Vec2>,
    /// Current waypoint index.
    waypoint_idx: usize,
    /// Waypoint arrival threshold.
    pub waypoint_threshold: f64,
    /// Simulation time.
    pub time: f64,
}

impl FormationController {
    pub fn new(strategy: FormationStrategy) -> Self {
        Self {
            robots: HashMap::new(),
            desired_shape: DesiredShape::new(),
            strategy,
            kp: 1.0,
            kd: 0.5,
            separation_weight: 1.5,
            cohesion_weight: 1.0,
            alignment_weight: 1.0,
            separation_dist: 2.0,
            consensus_gain: 0.5,
            neighbors: HashMap::new(),
            leader_waypoints: Vec::new(),
            waypoint_idx: 0,
            waypoint_threshold: 0.5,
            time: 0.0,
        }
    }

    pub fn with_gains(mut self, kp: f64, kd: f64) -> Self {
        self.kp = kp;
        self.kd = kd;
        self
    }

    pub fn with_behavior_weights(mut self, sep: f64, coh: f64, align: f64) -> Self {
        self.separation_weight = sep;
        self.cohesion_weight = coh;
        self.alignment_weight = align;
        self
    }

    pub fn with_consensus_gain(mut self, gain: f64) -> Self {
        self.consensus_gain = gain;
        self
    }

    pub fn with_desired_shape(mut self, shape: DesiredShape) -> Self {
        self.desired_shape = shape;
        self
    }

    pub fn with_separation_dist(mut self, dist: f64) -> Self {
        self.separation_dist = dist;
        self
    }

    pub fn with_waypoints(mut self, waypoints: Vec<Vec2>) -> Self {
        self.leader_waypoints = waypoints;
        self.waypoint_idx = 0;
        self
    }

    /// Add a robot to the formation.
    pub fn add_robot(&mut self, robot: RobotState) -> Result<(), FormationError> {
        if self.robots.contains_key(&robot.id) {
            return Err(FormationError::DuplicateRobot(robot.id));
        }
        self.robots.insert(robot.id, robot);
        Ok(())
    }

    /// Set the communication topology (adjacency list).
    pub fn set_neighbors(&mut self, adj: HashMap<u64, Vec<u64>>) {
        self.neighbors = adj;
    }

    /// Build a fully-connected communication graph from current robots.
    pub fn fully_connect(&mut self) {
        let ids: Vec<u64> = self.robots.keys().copied().collect();
        for &id in &ids {
            let nbrs: Vec<u64> = ids.iter().copied().filter(|x| *x != id).collect();
            self.neighbors.insert(id, nbrs);
        }
    }

    /// Find the leader robot.
    pub fn leader(&self) -> Option<&RobotState> {
        self.robots.values().find(|r| r.is_leader)
    }

    /// Compute formation centroid.
    pub fn centroid(&self) -> Vec2 {
        if self.robots.is_empty() {
            return Vec2::zero();
        }
        let mut sum = Vec2::zero();
        for r in self.robots.values() {
            sum = sum.add(&r.position);
        }
        sum.scale(1.0 / self.robots.len() as f64)
    }

    /// Compute formation error: RMS distance from desired positions.
    pub fn formation_error(&self, reference: &Vec2) -> f64 {
        if self.robots.is_empty() {
            return 0.0;
        }
        let mut sse = 0.0;
        let mut count = 0usize;
        for (id, robot) in &self.robots {
            if let Some(offset) = self.desired_shape.offsets.get(id) {
                let desired = reference.add(offset);
                let err = robot.position.dist(&desired);
                sse += err * err;
                count += 1;
            }
        }
        if count == 0 { 0.0 } else { (sse / count as f64).sqrt() }
    }

    /// Step the formation by one time increment.
    pub fn step(&mut self, dt: f64) -> Result<(), FormationError> {
        let robot_ids: Vec<u64> = self.robots.keys().copied().collect();
        let old_positions: HashMap<u64, Vec2> =
            self.robots.iter().map(|(&id, r)| (id, r.position)).collect();
        let old_velocities: HashMap<u64, Vec2> =
            self.robots.iter().map(|(&id, r)| (id, r.velocity)).collect();

        match self.strategy {
            FormationStrategy::LeaderFollower => {
                self.step_leader_follower(&robot_ids, &old_positions, dt)?;
            }
            FormationStrategy::VirtualStructure => {
                self.step_virtual_structure(&robot_ids, &old_positions, dt);
            }
            FormationStrategy::BehaviorBased => {
                self.step_behavior_based(&robot_ids, &old_positions, &old_velocities, dt);
            }
            FormationStrategy::Consensus => {
                self.step_consensus(&robot_ids, &old_positions, dt);
            }
        }

        for id in &robot_ids {
            if let Some(robot) = self.robots.get_mut(id) {
                robot.step(dt);
            }
        }
        self.time += dt;
        Ok(())
    }

    fn step_leader_follower(
        &mut self,
        ids: &[u64],
        positions: &HashMap<u64, Vec2>,
        dt: f64,
    ) -> Result<(), FormationError> {
        // Find leader position (advance toward waypoint).
        let leader_id = ids.iter().find(|&&id| {
            self.robots.get(&id).map(|r| r.is_leader).unwrap_or(false)
        }).copied().ok_or(FormationError::NoLeader)?;

        let leader_pos = positions[&leader_id];

        // Move leader toward next waypoint.
        if let Some(wp) = self.leader_waypoints.get(self.waypoint_idx) {
            let to_wp = wp.sub(&leader_pos);
            if to_wp.norm() < self.waypoint_threshold {
                if self.waypoint_idx + 1 < self.leader_waypoints.len() {
                    self.waypoint_idx += 1;
                }
            }
            if let Some(wp_cur) = self.leader_waypoints.get(self.waypoint_idx) {
                let dir = wp_cur.sub(&leader_pos).normalized();
                if let Some(leader) = self.robots.get_mut(&leader_id) {
                    leader.velocity = dir.scale(leader.max_speed);
                }
            }
        }

        let leader_pos_current =
            self.robots.get(&leader_id).map(|r| r.position).unwrap_or(leader_pos);

        // Followers track desired offset from leader.
        for &id in ids {
            if id == leader_id {
                continue;
            }
            if let Some(offset) = self.desired_shape.offsets.get(&id) {
                let desired = leader_pos_current.add(offset);
                let pos = positions[&id];
                let error = desired.sub(&pos);
                let vel = self.robots.get(&id).map(|r| r.velocity).unwrap_or(Vec2::zero());
                let cmd = error.scale(self.kp).sub(&vel.scale(self.kd));
                if let Some(robot) = self.robots.get_mut(&id) {
                    robot.velocity = cmd;
                }
            }
        }
        let _ = dt;
        Ok(())
    }

    fn step_virtual_structure(
        &mut self,
        ids: &[u64],
        positions: &HashMap<u64, Vec2>,
        _dt: f64,
    ) {
        // Virtual structure center = current centroid.
        let center = self.centroid();
        // Move center toward first waypoint (if any).
        let target = self.leader_waypoints.get(self.waypoint_idx).copied().unwrap_or(center);
        let center_vel = target.sub(&center).scale(self.kp * 0.5);

        // Advance waypoint.
        if center.dist(&target) < self.waypoint_threshold
            && self.waypoint_idx + 1 < self.leader_waypoints.len()
        {
            self.waypoint_idx += 1;
        }

        let new_center = center.add(&center_vel);

        for &id in ids {
            if let Some(offset) = self.desired_shape.offsets.get(&id) {
                let desired = new_center.add(offset);
                let pos = positions[&id];
                let error = desired.sub(&pos);
                let vel =
                    self.robots.get(&id).map(|r| r.velocity).unwrap_or(Vec2::zero());
                let cmd = error.scale(self.kp).sub(&vel.scale(self.kd));
                if let Some(robot) = self.robots.get_mut(&id) {
                    robot.velocity = cmd;
                }
            }
        }
    }

    fn step_behavior_based(
        &mut self,
        ids: &[u64],
        positions: &HashMap<u64, Vec2>,
        velocities: &HashMap<u64, Vec2>,
        _dt: f64,
    ) {
        let center = self.centroid();

        for &id in ids {
            let pos = positions[&id];
            let vel = velocities.get(&id).copied().unwrap_or(Vec2::zero());

            // Separation: steer away from nearby robots.
            let mut sep = Vec2::zero();
            for &oid in ids {
                if oid == id {
                    continue;
                }
                let opos = positions[&oid];
                let d = pos.dist(&opos);
                if d < self.separation_dist && d > 1e-12 {
                    let away = pos.sub(&opos).normalized().scale(1.0 / d);
                    sep = sep.add(&away);
                }
            }

            // Cohesion: steer toward formation-desired position.
            let desired_offset =
                self.desired_shape.offsets.get(&id).copied().unwrap_or(Vec2::zero());
            let desired_pos = center.add(&desired_offset);
            let coh = desired_pos.sub(&pos);

            // Alignment: match average neighbor velocity.
            let nbr_ids = self.neighbors.get(&id).cloned().unwrap_or_default();
            let mut align = Vec2::zero();
            if !nbr_ids.is_empty() {
                for &nid in &nbr_ids {
                    let nvel = velocities.get(&nid).copied().unwrap_or(Vec2::zero());
                    align = align.add(&nvel);
                }
                align = align.scale(1.0 / nbr_ids.len() as f64);
                align = align.sub(&vel);
            }

            let cmd = sep.scale(self.separation_weight)
                .add(&coh.scale(self.cohesion_weight))
                .add(&align.scale(self.alignment_weight));

            if let Some(robot) = self.robots.get_mut(&id) {
                robot.velocity = cmd;
            }
        }
    }

    fn step_consensus(
        &mut self,
        ids: &[u64],
        positions: &HashMap<u64, Vec2>,
        _dt: f64,
    ) {
        // Laplacian-based consensus: u_i = -gain * sum_{j in N_i} (x_i - x_j - (d_i - d_j))
        for &id in ids {
            let pos = positions[&id];
            let desired_i =
                self.desired_shape.offsets.get(&id).copied().unwrap_or(Vec2::zero());
            let nbr_ids = self.neighbors.get(&id).cloned().unwrap_or_default();

            let mut control = Vec2::zero();
            for &nid in &nbr_ids {
                let npos = positions.get(&nid).copied().unwrap_or(Vec2::zero());
                let desired_j =
                    self.desired_shape.offsets.get(&nid).copied().unwrap_or(Vec2::zero());
                // Consensus error.
                let rel_err = pos.sub(&npos).sub(&desired_i.sub(&desired_j));
                control = control.sub(&rel_err);
            }
            control = control.scale(self.consensus_gain);

            if let Some(robot) = self.robots.get_mut(&id) {
                robot.velocity = control;
            }
        }
    }

    /// Check if the formation has converged to within a tolerance.
    pub fn is_converged(&self, tolerance: f64) -> bool {
        let reference = match self.strategy {
            FormationStrategy::LeaderFollower => {
                self.leader().map(|l| l.position).unwrap_or_else(|| self.centroid())
            }
            _ => self.centroid(),
        };
        self.formation_error(&reference) < tolerance
    }
}

impl fmt::Display for FormationController {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FormationController({}, {} robots, t={:.2})",
            self.strategy,
            self.robots.len(),
            self.time,
        )
    }
}

// ── Shape Maintenance ───────────────────────────────────────────

/// Rigid shape maintenance using spring-damper model between edges.
#[derive(Debug, Clone)]
pub struct ShapeMaintainer {
    /// Desired inter-robot distances: (id_a, id_b) -> distance.
    pub desired_edges: Vec<(u64, u64, f64)>,
    /// Spring stiffness.
    pub stiffness: f64,
    /// Damping coefficient.
    pub damping: f64,
}

impl ShapeMaintainer {
    pub fn new(stiffness: f64, damping: f64) -> Self {
        Self { desired_edges: Vec::new(), stiffness, damping }
    }

    pub fn with_edge(mut self, a: u64, b: u64, distance: f64) -> Self {
        self.desired_edges.push((a, b, distance));
        self
    }

    /// Compute spring-damper corrections for each robot.
    pub fn compute_corrections(
        &self,
        positions: &HashMap<u64, Vec2>,
        velocities: &HashMap<u64, Vec2>,
    ) -> HashMap<u64, Vec2> {
        let mut corrections: HashMap<u64, Vec2> = HashMap::new();
        for &(a, b, desired_d) in &self.desired_edges {
            let pa = match positions.get(&a) {
                Some(p) => *p,
                None => continue,
            };
            let pb = match positions.get(&b) {
                Some(p) => *p,
                None => continue,
            };
            let va = velocities.get(&a).copied().unwrap_or(Vec2::zero());
            let vb = velocities.get(&b).copied().unwrap_or(Vec2::zero());

            let diff = pb.sub(&pa);
            let dist = diff.norm();
            if dist < 1e-12 {
                continue;
            }
            let dir = diff.normalized();
            let stretch = dist - desired_d;
            let rel_vel = vb.sub(&va);
            let vel_along = rel_vel.dot(&dir);

            let force_mag = self.stiffness * stretch + self.damping * vel_along;
            let force = dir.scale(force_mag);

            let ca = corrections.entry(a).or_insert(Vec2::zero());
            *ca = ca.add(&force);
            let cb = corrections.entry(b).or_insert(Vec2::zero());
            *cb = cb.sub(&force);
        }
        corrections
    }

    /// Maximum edge error across all edges.
    pub fn max_edge_error(&self, positions: &HashMap<u64, Vec2>) -> f64 {
        let mut max_err = 0.0f64;
        for &(a, b, desired_d) in &self.desired_edges {
            if let (Some(pa), Some(pb)) = (positions.get(&a), positions.get(&b)) {
                let err = (pa.dist(pb) - desired_d).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        max_err
    }
}

impl fmt::Display for ShapeMaintainer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ShapeMaintainer(edges={}, k={:.2}, b={:.2})",
            self.desired_edges.len(),
            self.stiffness,
            self.damping,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec2_norm() {
        let v = Vec2::new(3.0, 4.0);
        assert!((v.norm() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_vec2_normalized() {
        let v = Vec2::new(0.0, 5.0).normalized();
        assert!((v.x).abs() < 1e-9);
        assert!((v.y - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_vec2_zero_normalized() {
        let v = Vec2::zero().normalized();
        assert!((v.norm()).abs() < 1e-9);
    }

    #[test]
    fn test_vec2_dist() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(4.0, 6.0);
        assert!((a.dist(&b) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_robot_state_step() {
        let mut r = RobotState::new(1, Vec2::zero()).with_max_speed(10.0);
        r.velocity = Vec2::new(3.0, 4.0);
        r.step(1.0);
        assert!((r.position.x - 3.0).abs() < 1e-9);
        assert!((r.position.y - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_robot_speed_clamp() {
        let mut r = RobotState::new(1, Vec2::zero()).with_max_speed(1.0);
        r.velocity = Vec2::new(10.0, 0.0);
        r.step(1.0);
        assert!((r.position.x - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_regular_polygon_shape() {
        let ids = [1, 2, 3, 4];
        let shape = DesiredShape::regular_polygon(&ids, 5.0);
        assert_eq!(shape.offsets.len(), 4);
        // First robot at angle 0 => (5, 0).
        let o = shape.offsets[&1];
        assert!((o.x - 5.0).abs() < 1e-9);
        assert!((o.y).abs() < 1e-9);
    }

    #[test]
    fn test_line_shape_symmetric() {
        let ids = [1, 2, 3];
        let shape = DesiredShape::line(&ids, 2.0);
        let centroid = shape.centroid();
        assert!((centroid.x).abs() < 1e-9);
    }

    #[test]
    fn test_wedge_shape() {
        let ids = [1, 2, 3, 4, 5];
        let shape = DesiredShape::wedge(&ids, 3.0, 30.0);
        assert_eq!(shape.offsets.len(), 5);
        // Leader is at origin.
        let l = shape.offsets[&1];
        assert!((l.x).abs() < 1e-9);
        assert!((l.y).abs() < 1e-9);
    }

    #[test]
    fn test_formation_add_robot() {
        let mut ctrl = FormationController::new(FormationStrategy::LeaderFollower);
        ctrl.add_robot(RobotState::new(1, Vec2::new(0.0, 0.0))).unwrap();
        assert_eq!(ctrl.robots.len(), 1);
    }

    #[test]
    fn test_formation_duplicate_robot() {
        let mut ctrl = FormationController::new(FormationStrategy::LeaderFollower);
        ctrl.add_robot(RobotState::new(1, Vec2::zero())).unwrap();
        let err = ctrl.add_robot(RobotState::new(1, Vec2::zero()));
        assert!(err.is_err());
    }

    #[test]
    fn test_formation_centroid() {
        let mut ctrl = FormationController::new(FormationStrategy::Consensus);
        ctrl.add_robot(RobotState::new(1, Vec2::new(0.0, 0.0))).unwrap();
        ctrl.add_robot(RobotState::new(2, Vec2::new(4.0, 0.0))).unwrap();
        let c = ctrl.centroid();
        assert!((c.x - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_leader_follower_convergence() {
        let mut ctrl = FormationController::new(FormationStrategy::LeaderFollower)
            .with_gains(2.0, 0.8);
        let shape = DesiredShape::new()
            .with_offset(1, Vec2::zero())
            .with_offset(2, Vec2::new(2.0, 0.0))
            .with_offset(3, Vec2::new(-2.0, 0.0));
        ctrl.desired_shape = shape;
        ctrl.add_robot(RobotState::new(1, Vec2::zero()).with_leader(true).with_max_speed(5.0)).unwrap();
        ctrl.add_robot(RobotState::new(2, Vec2::new(10.0, 5.0)).with_max_speed(5.0)).unwrap();
        ctrl.add_robot(RobotState::new(3, Vec2::new(-10.0, -5.0)).with_max_speed(5.0)).unwrap();
        ctrl.fully_connect();

        for _ in 0..200 {
            ctrl.step(0.05).unwrap();
        }
        let leader_pos = ctrl.robots[&1].position;
        assert!(ctrl.formation_error(&leader_pos) < 1.0);
    }

    #[test]
    fn test_consensus_convergence() {
        let mut ctrl = FormationController::new(FormationStrategy::Consensus)
            .with_consensus_gain(0.8);
        let shape = DesiredShape::regular_polygon(&[1, 2, 3], 3.0);
        ctrl.desired_shape = shape;
        ctrl.add_robot(RobotState::new(1, Vec2::new(10.0, 0.0)).with_max_speed(10.0)).unwrap();
        ctrl.add_robot(RobotState::new(2, Vec2::new(0.0, 10.0)).with_max_speed(10.0)).unwrap();
        ctrl.add_robot(RobotState::new(3, Vec2::new(-5.0, -5.0)).with_max_speed(10.0)).unwrap();
        ctrl.fully_connect();

        for _ in 0..300 {
            ctrl.step(0.05).unwrap();
        }
        let c = ctrl.centroid();
        assert!(ctrl.formation_error(&c) < 1.0);
    }

    #[test]
    fn test_virtual_structure_step() {
        let mut ctrl = FormationController::new(FormationStrategy::VirtualStructure)
            .with_gains(1.5, 0.5)
            .with_waypoints(vec![Vec2::new(20.0, 0.0)]);
        let shape = DesiredShape::line(&[1, 2], 4.0);
        ctrl.desired_shape = shape;
        ctrl.add_robot(RobotState::new(1, Vec2::new(0.0, 0.0)).with_max_speed(5.0)).unwrap();
        ctrl.add_robot(RobotState::new(2, Vec2::new(1.0, 0.0)).with_max_speed(5.0)).unwrap();

        for _ in 0..100 {
            ctrl.step(0.05).unwrap();
        }
        // Both robots should have moved toward the waypoint.
        assert!(ctrl.centroid().x > 1.0);
    }

    #[test]
    fn test_behavior_based_separation() {
        let mut ctrl = FormationController::new(FormationStrategy::BehaviorBased)
            .with_behavior_weights(5.0, 0.1, 0.1)
            .with_separation_dist(10.0);
        let shape = DesiredShape::new()
            .with_offset(1, Vec2::zero())
            .with_offset(2, Vec2::zero());
        ctrl.desired_shape = shape;
        ctrl.add_robot(RobotState::new(1, Vec2::new(0.0, 0.0)).with_max_speed(5.0)).unwrap();
        ctrl.add_robot(RobotState::new(2, Vec2::new(0.5, 0.0)).with_max_speed(5.0)).unwrap();
        ctrl.fully_connect();

        ctrl.step(0.1).unwrap();
        // Robot 1 should move left (away from robot 2), robot 2 should move right.
        let v1 = ctrl.robots[&1].velocity.x;
        let v2 = ctrl.robots[&2].velocity.x;
        assert!(v1 < v2);
    }

    #[test]
    fn test_shape_maintainer_spring() {
        let sm = ShapeMaintainer::new(10.0, 1.0)
            .with_edge(1, 2, 5.0);
        let mut positions = HashMap::new();
        positions.insert(1, Vec2::new(0.0, 0.0));
        positions.insert(2, Vec2::new(3.0, 0.0)); // Too close by 2.
        let velocities = HashMap::new();

        let corrections = sm.compute_corrections(&positions, &velocities);
        // Robot 1 should be pushed left (negative x), robot 2 pushed right.
        assert!(corrections[&1].x < 0.0);
        assert!(corrections[&2].x > 0.0);
    }

    #[test]
    fn test_shape_maintainer_max_error() {
        let sm = ShapeMaintainer::new(10.0, 1.0)
            .with_edge(1, 2, 5.0)
            .with_edge(2, 3, 3.0);
        let mut positions = HashMap::new();
        positions.insert(1, Vec2::new(0.0, 0.0));
        positions.insert(2, Vec2::new(5.0, 0.0)); // Perfect.
        positions.insert(3, Vec2::new(10.0, 0.0)); // Error = 2.
        let err = sm.max_edge_error(&positions);
        assert!((err - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_no_leader_error() {
        let mut ctrl = FormationController::new(FormationStrategy::LeaderFollower);
        ctrl.add_robot(RobotState::new(1, Vec2::zero())).unwrap();
        let result = ctrl.step(0.1);
        assert!(matches!(result, Err(FormationError::NoLeader)));
    }

    #[test]
    fn test_display_impls() {
        let v = Vec2::new(1.5, 2.5);
        assert!(format!("{v}").contains("1.500"));
        let r = RobotState::new(42, Vec2::zero()).with_leader(true);
        assert!(format!("{r}").contains("leader"));
        let ctrl = FormationController::new(FormationStrategy::Consensus);
        assert!(format!("{ctrl}").contains("Consensus"));
        let sm = ShapeMaintainer::new(1.0, 0.5);
        assert!(format!("{sm}").contains("ShapeMaintainer"));
        let shape = DesiredShape::new();
        assert!(format!("{shape}").contains("0 robots"));
    }
}
