//! Artificial Potential Field — attractive/repulsive forces, gradient descent
//! navigation, local minima escape for reactive motion planning.
//!
//! Pure-Rust potential field planner with configurable force gains, multiple
//! obstacle types, random perturbation for local minima escape, and full
//! trajectory recording.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum PotentialFieldError {
    InvalidParameter(String),
    StuckInLocalMinimum,
    MaxIterationsReached,
    GoalInObstacle,
}

impl fmt::Display for PotentialFieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
            Self::StuckInLocalMinimum => write!(f, "stuck in local minimum"),
            Self::MaxIterationsReached => write!(f, "max iterations reached"),
            Self::GoalInObstacle => write!(f, "goal position is inside an obstacle"),
        }
    }
}

impl std::error::Error for PotentialFieldError {}

// ── Vec2 ────────────────────────────────────────────────────────

/// 2D vector for force/position calculations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self { Self { x, y } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0 } }

    pub fn magnitude(self) -> f64 { (self.x * self.x + self.y * self.y).sqrt() }

    pub fn normalized(self) -> Self {
        let m = self.magnitude();
        if m < 1e-12 { Self::zero() } else { Self { x: self.x / m, y: self.y / m } }
    }

    pub fn distance_to(self, other: Vec2) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    pub fn dot(self, other: Vec2) -> f64 { self.x * other.x + self.y * other.y }

    pub fn add(self, other: Vec2) -> Vec2 {
        Vec2 { x: self.x + other.x, y: self.y + other.y }
    }

    pub fn scale(self, s: f64) -> Vec2 {
        Vec2 { x: self.x * s, y: self.y * s }
    }

    pub fn clamp_magnitude(self, max: f64) -> Vec2 {
        let m = self.magnitude();
        if m <= max { self } else { self.normalized().scale(max) }
    }
}

impl fmt::Display for Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.3}, {:.3})", self.x, self.y)
    }
}

// ── Obstacle Types ──────────────────────────────────────────────

/// An obstacle that generates repulsive force.
#[derive(Debug, Clone)]
pub enum Obstacle {
    /// Circular obstacle with center and radius.
    Circle { center: Vec2, radius: f64 },
    /// Axis-aligned rectangle.
    Rect { min: Vec2, max: Vec2 },
    /// Line segment wall.
    Segment { a: Vec2, b: Vec2 },
}

impl Obstacle {
    /// Closest point on obstacle surface to query point.
    pub fn closest_point(&self, p: Vec2) -> Vec2 {
        match self {
            Obstacle::Circle { center, radius } => {
                let d = Vec2::new(p.x - center.x, p.y - center.y);
                let m = d.magnitude();
                if m < 1e-12 {
                    Vec2::new(center.x + radius, center.y)
                } else {
                    Vec2::new(
                        center.x + d.x / m * radius,
                        center.y + d.y / m * radius,
                    )
                }
            }
            Obstacle::Rect { min, max } => {
                Vec2::new(
                    p.x.clamp(min.x, max.x),
                    p.y.clamp(min.y, max.y),
                )
            }
            Obstacle::Segment { a, b } => {
                let ab = Vec2::new(b.x - a.x, b.y - a.y);
                let ap = Vec2::new(p.x - a.x, p.y - a.y);
                let t = (ap.dot(ab) / ab.dot(ab)).clamp(0.0, 1.0);
                Vec2::new(a.x + ab.x * t, a.y + ab.y * t)
            }
        }
    }

    /// Distance from point to nearest surface of obstacle.
    pub fn distance_to(&self, p: Vec2) -> f64 {
        let closest = self.closest_point(p);
        p.distance_to(closest)
    }

    /// Whether the point is inside the obstacle.
    pub fn contains(&self, p: Vec2) -> bool {
        match self {
            Obstacle::Circle { center, radius } => p.distance_to(*center) <= *radius,
            Obstacle::Rect { min, max } => {
                p.x >= min.x && p.x <= max.x && p.y >= min.y && p.y <= max.y
            }
            Obstacle::Segment { .. } => false, // A line has no interior
        }
    }
}

impl fmt::Display for Obstacle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Obstacle::Circle { center, radius } => {
                write!(f, "Circle(center={center}, r={radius:.2})")
            }
            Obstacle::Rect { min, max } => write!(f, "Rect({min} to {max})"),
            Obstacle::Segment { a, b } => write!(f, "Seg({a} to {b})"),
        }
    }
}

// ── Simple LCG RNG ──────────────────────────────────────────────

struct SimpleRng { state: u64 }

impl SimpleRng {
    fn new(seed: u64) -> Self { Self { state: seed.wrapping_add(1) } }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_symmetric(&mut self) -> f64 { self.next_f64() * 2.0 - 1.0 }
}

// ── Potential Field Config ──────────────────────────────────────

/// Configuration for the potential field planner.
#[derive(Debug, Clone)]
pub struct PotentialFieldConfig {
    pub attractive_gain: f64,
    pub repulsive_gain: f64,
    pub influence_distance: f64,
    pub step_size: f64,
    pub goal_threshold: f64,
    pub max_iterations: usize,
    pub max_force: f64,
    pub escape_perturbation: f64,
    pub escape_patience: usize,
    pub seed: u64,
}

impl PotentialFieldConfig {
    pub fn new() -> Self {
        Self {
            attractive_gain: 1.0,
            repulsive_gain: 100.0,
            influence_distance: 3.0,
            step_size: 0.1,
            goal_threshold: 0.2,
            max_iterations: 5000,
            max_force: 5.0,
            escape_perturbation: 0.5,
            escape_patience: 50,
            seed: 42,
        }
    }

    pub fn with_attractive_gain(mut self, g: f64) -> Self { self.attractive_gain = g; self }
    pub fn with_repulsive_gain(mut self, g: f64) -> Self { self.repulsive_gain = g; self }
    pub fn with_influence_distance(mut self, d: f64) -> Self { self.influence_distance = d; self }
    pub fn with_step_size(mut self, s: f64) -> Self { self.step_size = s; self }
    pub fn with_goal_threshold(mut self, t: f64) -> Self { self.goal_threshold = t; self }
    pub fn with_max_iterations(mut self, n: usize) -> Self { self.max_iterations = n; self }
    pub fn with_max_force(mut self, f: f64) -> Self { self.max_force = f; self }
    pub fn with_escape_perturbation(mut self, p: f64) -> Self { self.escape_perturbation = p; self }
    pub fn with_escape_patience(mut self, p: usize) -> Self { self.escape_patience = p; self }
    pub fn with_seed(mut self, s: u64) -> Self { self.seed = s; self }
}

impl fmt::Display for PotentialFieldConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PotentialFieldConfig(k_att={:.1}, k_rep={:.1}, d0={:.1}, step={:.2})",
            self.attractive_gain, self.repulsive_gain,
            self.influence_distance, self.step_size,
        )
    }
}

// ── Potential Field Result ──────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PotentialFieldResult {
    pub trajectory: Vec<Vec2>,
    pub total_distance: f64,
    pub iterations: usize,
    pub escapes: usize,
    pub reached_goal: bool,
}

impl fmt::Display for PotentialFieldResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PotentialFieldResult(points={}, dist={:.3}, iters={}, escapes={}, reached={})",
            self.trajectory.len(), self.total_distance,
            self.iterations, self.escapes, self.reached_goal,
        )
    }
}

// ── Potential Field Planner ─────────────────────────────────────

/// Artificial potential field planner.
pub struct PotentialFieldPlanner {
    config: PotentialFieldConfig,
    obstacles: Vec<Obstacle>,
}

impl PotentialFieldPlanner {
    pub fn new(config: PotentialFieldConfig) -> Self {
        Self { config, obstacles: Vec::new() }
    }

    pub fn with_obstacles(mut self, obs: Vec<Obstacle>) -> Self {
        self.obstacles = obs;
        self
    }

    pub fn add_obstacle(&mut self, obs: Obstacle) {
        self.obstacles.push(obs);
    }

    /// Compute attractive force toward goal (quadratic potential).
    fn attractive_force(&self, pos: Vec2, goal: Vec2) -> Vec2 {
        let diff = Vec2::new(goal.x - pos.x, goal.y - pos.y);
        diff.scale(self.config.attractive_gain)
    }

    /// Compute repulsive force from a single obstacle.
    fn repulsive_force_single(&self, pos: Vec2, obs: &Obstacle) -> Vec2 {
        let dist = obs.distance_to(pos);
        if dist > self.config.influence_distance || dist < 1e-12 {
            return Vec2::zero();
        }
        let closest = obs.closest_point(pos);
        let away = Vec2::new(pos.x - closest.x, pos.y - closest.y).normalized();
        let magnitude = self.config.repulsive_gain
            * (1.0 / dist - 1.0 / self.config.influence_distance)
            / (dist * dist);
        away.scale(magnitude)
    }

    /// Compute total repulsive force from all obstacles.
    fn repulsive_force(&self, pos: Vec2) -> Vec2 {
        let mut total = Vec2::zero();
        for obs in &self.obstacles {
            let f = self.repulsive_force_single(pos, obs);
            total = total.add(f);
        }
        total
    }

    /// Compute the total potential at a position.
    pub fn potential(&self, pos: Vec2, goal: Vec2) -> f64 {
        let att = 0.5 * self.config.attractive_gain * pos.distance_to(goal).powi(2);
        let mut rep = 0.0;
        for obs in &self.obstacles {
            let d = obs.distance_to(pos);
            if d < self.config.influence_distance && d > 1e-12 {
                let term = 1.0 / d - 1.0 / self.config.influence_distance;
                rep += 0.5 * self.config.repulsive_gain * term * term;
            }
        }
        att + rep
    }

    /// Plan a path using gradient descent on the potential field.
    pub fn plan(&self, start: Vec2, goal: Vec2) -> Result<PotentialFieldResult, PotentialFieldError> {
        let mut rng = SimpleRng::new(self.config.seed);
        let mut pos = start;
        let mut trajectory = vec![start];
        let mut total_distance = 0.0;
        let mut escapes = 0usize;
        let mut stall_count = 0usize;
        let mut prev_potential = self.potential(pos, goal);

        for iter in 0..self.config.max_iterations {
            if pos.distance_to(goal) <= self.config.goal_threshold {
                return Ok(PotentialFieldResult {
                    trajectory,
                    total_distance,
                    iterations: iter,
                    escapes,
                    reached_goal: true,
                });
            }

            let f_att = self.attractive_force(pos, goal);
            let f_rep = self.repulsive_force(pos);
            let mut total_force = f_att.add(f_rep).clamp_magnitude(self.config.max_force);

            // Local minima detection and escape
            let current_potential = self.potential(pos, goal);
            if (current_potential - prev_potential).abs() < 1e-6 {
                stall_count += 1;
            } else {
                stall_count = 0;
            }

            if stall_count >= self.config.escape_patience {
                let perturb = Vec2::new(
                    rng.next_symmetric() * self.config.escape_perturbation,
                    rng.next_symmetric() * self.config.escape_perturbation,
                );
                total_force = total_force.add(perturb);
                stall_count = 0;
                escapes += 1;
            }

            prev_potential = current_potential;

            let step = total_force.normalized().scale(self.config.step_size);
            let new_pos = pos.add(step);

            // Reject steps into obstacles
            let in_obstacle = self.obstacles.iter().any(|o| o.contains(new_pos));
            if !in_obstacle {
                total_distance += pos.distance_to(new_pos);
                pos = new_pos;
                trajectory.push(pos);
            }
        }

        Ok(PotentialFieldResult {
            trajectory,
            total_distance,
            iterations: self.config.max_iterations,
            escapes,
            reached_goal: pos.distance_to(goal) <= self.config.goal_threshold,
        })
    }

    pub fn obstacle_count(&self) -> usize { self.obstacles.len() }
}

impl fmt::Display for PotentialFieldPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PotentialFieldPlanner(obstacles={}, {})",
            self.obstacles.len(), self.config,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < 1e-6 }

    #[test]
    fn test_vec2_magnitude() {
        let v = Vec2::new(3.0, 4.0);
        assert!(approx(v.magnitude(), 5.0));
    }

    #[test]
    fn test_vec2_normalized() {
        let v = Vec2::new(3.0, 4.0).normalized();
        assert!(approx(v.magnitude(), 1.0));
    }

    #[test]
    fn test_vec2_zero_normalized() {
        let v = Vec2::zero().normalized();
        assert!(approx(v.magnitude(), 0.0));
    }

    #[test]
    fn test_vec2_clamp() {
        let v = Vec2::new(10.0, 0.0).clamp_magnitude(3.0);
        assert!(approx(v.magnitude(), 3.0));
    }

    #[test]
    fn test_vec2_display() {
        let v = Vec2::new(1.5, 2.5);
        let s = format!("{v}");
        assert!(s.contains("1.500"));
    }

    #[test]
    fn test_obstacle_circle_distance() {
        let obs = Obstacle::Circle { center: Vec2::new(5.0, 5.0), radius: 1.0 };
        let d = obs.distance_to(Vec2::new(8.0, 5.0));
        assert!(approx(d, 2.0));
    }

    #[test]
    fn test_obstacle_circle_contains() {
        let obs = Obstacle::Circle { center: Vec2::new(5.0, 5.0), radius: 1.0 };
        assert!(obs.contains(Vec2::new(5.5, 5.0)));
        assert!(!obs.contains(Vec2::new(8.0, 5.0)));
    }

    #[test]
    fn test_obstacle_rect_closest() {
        let obs = Obstacle::Rect {
            min: Vec2::new(2.0, 2.0),
            max: Vec2::new(4.0, 4.0),
        };
        let cp = obs.closest_point(Vec2::new(5.0, 3.0));
        assert!(approx(cp.x, 4.0));
        assert!(approx(cp.y, 3.0));
    }

    #[test]
    fn test_obstacle_display() {
        let obs = Obstacle::Circle { center: Vec2::new(1.0, 2.0), radius: 0.5 };
        let s = format!("{obs}");
        assert!(s.contains("Circle"));
    }

    #[test]
    fn test_attractive_force_direction() {
        let cfg = PotentialFieldConfig::new();
        let planner = PotentialFieldPlanner::new(cfg);
        let f = planner.attractive_force(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0));
        assert!(f.x > 0.0);
        assert!(approx(f.y, 0.0));
    }

    #[test]
    fn test_repulsive_force_pushes_away() {
        let cfg = PotentialFieldConfig::new().with_influence_distance(5.0);
        let obs = vec![Obstacle::Circle { center: Vec2::new(5.0, 0.0), radius: 1.0 }];
        let planner = PotentialFieldPlanner::new(cfg).with_obstacles(obs);
        let f = planner.repulsive_force(Vec2::new(3.0, 0.0));
        assert!(f.x < 0.0, "repulsive force should push away from obstacle");
    }

    #[test]
    fn test_plan_straight_line() {
        let cfg = PotentialFieldConfig::new()
            .with_step_size(0.1)
            .with_goal_threshold(0.3)
            .with_max_iterations(2000);
        let planner = PotentialFieldPlanner::new(cfg);
        let result = planner.plan(Vec2::new(0.0, 0.0), Vec2::new(5.0, 0.0)).unwrap();
        assert!(result.reached_goal);
        assert!(result.total_distance < 6.0);
    }

    #[test]
    fn test_plan_around_obstacle() {
        // Place obstacle off the direct line so the planner can route around it
        let cfg = PotentialFieldConfig::new()
            .with_step_size(0.1)
            .with_goal_threshold(0.5)
            .with_max_iterations(5000)
            .with_repulsive_gain(50.0)
            .with_influence_distance(2.0);
        let obs = vec![Obstacle::Circle { center: Vec2::new(5.0, 0.5), radius: 1.0 }];
        let planner = PotentialFieldPlanner::new(cfg).with_obstacles(obs);
        let result = planner.plan(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0)).unwrap();
        assert!(result.reached_goal);
        assert!(result.total_distance > 10.0); // must detour
    }

    #[test]
    fn test_potential_decreases_toward_goal() {
        let cfg = PotentialFieldConfig::new();
        let planner = PotentialFieldPlanner::new(cfg);
        let goal = Vec2::new(10.0, 0.0);
        let p1 = planner.potential(Vec2::new(0.0, 0.0), goal);
        let p2 = planner.potential(Vec2::new(5.0, 0.0), goal);
        assert!(p2 < p1);
    }

    #[test]
    fn test_config_display() {
        let cfg = PotentialFieldConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("k_att="));
    }

    #[test]
    fn test_result_display() {
        let r = PotentialFieldResult {
            trajectory: vec![Vec2::zero(), Vec2::new(1.0, 0.0)],
            total_distance: 1.0,
            iterations: 10,
            escapes: 0,
            reached_goal: true,
        };
        let s = format!("{r}");
        assert!(s.contains("reached=true"));
    }

    #[test]
    fn test_planner_display() {
        let cfg = PotentialFieldConfig::new();
        let planner = PotentialFieldPlanner::new(cfg);
        let s = format!("{planner}");
        assert!(s.contains("PotentialFieldPlanner"));
    }

    #[test]
    fn test_segment_obstacle() {
        let obs = Obstacle::Segment {
            a: Vec2::new(0.0, 5.0),
            b: Vec2::new(10.0, 5.0),
        };
        let d = obs.distance_to(Vec2::new(5.0, 7.0));
        assert!(approx(d, 2.0));
    }

    #[test]
    fn test_error_display() {
        let e = PotentialFieldError::StuckInLocalMinimum;
        assert_eq!(format!("{e}"), "stuck in local minimum");
    }
}
