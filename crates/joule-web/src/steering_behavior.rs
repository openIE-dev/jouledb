//! Reynolds-style steering behaviors — seek, flee, arrive, pursue, evade,
//! wander, obstacle avoidance, wall following, path following, behavior
//! composition (weighted blending, priority-based), max force/speed clamping,
//! smooth steering via low-pass filter.
//!
//! Replaces JavaScript steering libraries (yuka, steering-behaviors) with
//! pure-Rust agent movement for games and simulations.

// ── Vec2 ────────────────────────────────────────────────────────

/// 2D vector for steering math.
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

    pub fn normalized(&self) -> Self {
        let len = self.length();
        if len < 1e-12 { Self::ZERO } else { Self { x: self.x / len, y: self.y / len } }
    }

    pub fn truncate(&self, max_len: f64) -> Self {
        let len_sq = self.length_sq();
        if len_sq <= max_len * max_len { *self }
        else {
            let len = len_sq.sqrt();
            let s = max_len / len;
            Self { x: self.x * s, y: self.y * s }
        }
    }

    pub fn dot(&self, other: Vec2) -> f64 { self.x * other.x + self.y * other.y }

    pub fn dist(&self, other: Vec2) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    pub fn add(&self, other: Vec2) -> Vec2 {
        Vec2 { x: self.x + other.x, y: self.y + other.y }
    }

    pub fn sub(&self, other: Vec2) -> Vec2 {
        Vec2 { x: self.x - other.x, y: self.y - other.y }
    }

    pub fn scale(&self, s: f64) -> Vec2 {
        Vec2 { x: self.x * s, y: self.y * s }
    }

    pub fn perp(&self) -> Vec2 {
        Vec2 { x: -self.y, y: self.x }
    }

    pub fn rotate(&self, angle: f64) -> Vec2 {
        let c = angle.cos();
        let s = angle.sin();
        Vec2 { x: self.x * c - self.y * s, y: self.x * s + self.y * c }
    }

    pub fn lerp(&self, other: Vec2, t: f64) -> Vec2 {
        Vec2 {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }
}

// ── Agent ───────────────────────────────────────────────────────

/// A steerable agent with position, velocity, and limits.
#[derive(Debug, Clone, PartialEq)]
pub struct Agent {
    pub position: Vec2,
    pub velocity: Vec2,
    pub max_speed: f64,
    pub max_force: f64,
    pub mass: f64,
}

impl Agent {
    pub fn new(position: Vec2, max_speed: f64, max_force: f64) -> Self {
        Self {
            position,
            velocity: Vec2::ZERO,
            max_speed,
            max_force,
            mass: 1.0,
        }
    }

    /// Heading (normalized velocity or default forward).
    pub fn heading(&self) -> Vec2 {
        if self.velocity.length_sq() > 1e-12 {
            self.velocity.normalized()
        } else {
            Vec2::new(1.0, 0.0)
        }
    }

    /// Speed (magnitude of velocity).
    pub fn speed(&self) -> f64 { self.velocity.length() }

    /// Apply a steering force for one timestep.
    pub fn apply_force(&mut self, force: Vec2, dt: f64) {
        let clamped_force = force.truncate(self.max_force);
        let accel = clamped_force.scale(1.0 / self.mass);
        self.velocity = self.velocity.add(accel.scale(dt)).truncate(self.max_speed);
        self.position = self.position.add(self.velocity.scale(dt));
    }
}

// ── Obstacle & Wall ─────────────────────────────────────────────

/// Circular obstacle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Obstacle {
    pub center: Vec2,
    pub radius: f64,
}

/// Wall segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wall {
    pub start: Vec2,
    pub end: Vec2,
}

// ── Steering behaviors ──────────────────────────────────────────

/// Seek: steer toward a target position.
pub fn seek(agent: &Agent, target: Vec2) -> Vec2 {
    let desired = target.sub(agent.position).normalized().scale(agent.max_speed);
    desired.sub(agent.velocity)
}

/// Flee: steer away from a target position.
pub fn flee(agent: &Agent, target: Vec2) -> Vec2 {
    let desired = agent.position.sub(target).normalized().scale(agent.max_speed);
    desired.sub(agent.velocity)
}

/// Arrive: seek with slow-down near target.
pub fn arrive(agent: &Agent, target: Vec2, slow_radius: f64) -> Vec2 {
    let offset = target.sub(agent.position);
    let dist = offset.length();
    if dist < 1e-10 {
        return agent.velocity.scale(-1.0); // brake
    }
    let speed = if dist < slow_radius {
        agent.max_speed * (dist / slow_radius)
    } else {
        agent.max_speed
    };
    let desired = offset.normalized().scale(speed);
    desired.sub(agent.velocity)
}

/// Pursue: predict target's future position and seek it.
pub fn pursue(agent: &Agent, target_pos: Vec2, target_vel: Vec2) -> Vec2 {
    let to_target = target_pos.sub(agent.position);
    let dist = to_target.length();
    let speed = agent.speed().max(1.0);
    let prediction_time = dist / (speed + target_vel.length().max(0.001));
    let future_pos = target_pos.add(target_vel.scale(prediction_time));
    seek(agent, future_pos)
}

/// Evade: predict target and flee from future position.
pub fn evade(agent: &Agent, target_pos: Vec2, target_vel: Vec2) -> Vec2 {
    let to_target = target_pos.sub(agent.position);
    let dist = to_target.length();
    let speed = agent.speed().max(1.0);
    let prediction_time = dist / (speed + target_vel.length().max(0.001));
    let future_pos = target_pos.add(target_vel.scale(prediction_time));
    flee(agent, future_pos)
}

/// Wander: constrained random steering.
/// `wander_radius`: radius of wander circle
/// `wander_dist`: distance of wander circle ahead
/// `wander_angle`: current angle on wander circle (mutated)
/// `jitter`: max random change to angle per call
pub fn wander(
    agent: &Agent,
    wander_radius: f64,
    wander_dist: f64,
    wander_angle: &mut f64,
    jitter: f64,
    rng_value: f64, // -1.0 to 1.0, caller-provided
) -> Vec2 {
    *wander_angle += rng_value * jitter;
    let heading = agent.heading();
    let circle_center = agent.position.add(heading.scale(wander_dist));
    let offset = Vec2::new(wander_angle.cos(), wander_angle.sin()).scale(wander_radius);
    let target = circle_center.add(offset);
    seek(agent, target)
}

/// Obstacle avoidance: raycast ahead and steer away from obstacles.
pub fn obstacle_avoidance(
    agent: &Agent,
    obstacles: &[Obstacle],
    detection_length: f64,
) -> Vec2 {
    let heading = agent.heading();
    let mut nearest_dist = f64::MAX;
    let mut steer = Vec2::ZERO;

    for obs in obstacles {
        let to_obs = obs.center.sub(agent.position);
        let proj = to_obs.dot(heading);

        if proj < 0.0 || proj > detection_length + obs.radius {
            continue; // behind or too far
        }

        let lateral = to_obs.sub(heading.scale(proj));
        let lateral_dist = lateral.length();

        if lateral_dist > obs.radius {
            continue; // not on collision course
        }

        if proj < nearest_dist {
            nearest_dist = proj;
            // Steer perpendicular to heading, away from obstacle
            let side = heading.perp();
            let sign = if side.dot(to_obs) > 0.0 { -1.0 } else { 1.0 };
            let urgency = 1.0 + (detection_length - proj) / detection_length;
            steer = side.scale(sign * agent.max_force * urgency);
        }
    }

    steer
}

/// Wall following: steer to maintain distance from a wall.
pub fn wall_following(agent: &Agent, wall: &Wall, desired_dist: f64) -> Vec2 {
    let wall_vec = wall.end.sub(wall.start);
    let wall_len = wall_vec.length();
    if wall_len < 1e-10 {
        return Vec2::ZERO;
    }
    let wall_dir = wall_vec.normalized();
    let to_agent = agent.position.sub(wall.start);
    let proj = to_agent.dot(wall_dir).clamp(0.0, wall_len);
    let closest = wall.start.add(wall_dir.scale(proj));
    let offset = agent.position.sub(closest);
    let dist = offset.length();

    if dist < 1e-10 {
        return wall_dir.perp().scale(agent.max_force);
    }

    let normal = offset.normalized();
    if dist < desired_dist {
        normal.scale(agent.max_force * (1.0 - dist / desired_dist))
    } else {
        Vec2::ZERO
    }
}

/// Path following: follow waypoints with look-ahead.
pub fn path_following(
    agent: &Agent,
    waypoints: &[Vec2],
    look_ahead: f64,
    current_waypoint: &mut usize,
) -> Vec2 {
    if waypoints.is_empty() {
        return Vec2::ZERO;
    }

    // Advance waypoint if close enough
    while *current_waypoint < waypoints.len() - 1 {
        let dist = agent.position.dist(waypoints[*current_waypoint]);
        if dist < look_ahead {
            *current_waypoint += 1;
        } else {
            break;
        }
    }

    let target = waypoints[*current_waypoint];
    if *current_waypoint == waypoints.len() - 1 {
        arrive(agent, target, look_ahead * 2.0)
    } else {
        seek(agent, target)
    }
}

// ── Behavior composition ────────────────────────────────────────

/// A weighted steering behavior.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightedBehavior {
    pub force: Vec2,
    pub weight: f64,
}

/// Weighted blending of multiple steering forces.
pub fn weighted_blend(behaviors: &[WeightedBehavior]) -> Vec2 {
    let mut total = Vec2::ZERO;
    for b in behaviors {
        total = total.add(b.force.scale(b.weight));
    }
    total
}

/// Priority-based behavior selection: pick highest-priority non-zero force.
pub fn priority_select(behaviors: &[Vec2]) -> Vec2 {
    for force in behaviors {
        if force.length_sq() > 1e-10 {
            return *force;
        }
    }
    Vec2::ZERO
}

// ── Smooth steering ─────────────────────────────────────────────

/// Low-pass filter for smooth steering transitions.
#[derive(Debug, Clone, PartialEq)]
pub struct SteeringFilter {
    prev: Vec2,
    smoothing: f64, // 0.0 = no smoothing, 0.99 = very smooth
}

impl SteeringFilter {
    pub fn new(smoothing: f64) -> Self {
        Self {
            prev: Vec2::ZERO,
            smoothing: smoothing.clamp(0.0, 0.999),
        }
    }

    /// Filter a steering force.
    pub fn filter(&mut self, force: Vec2) -> Vec2 {
        let result = self.prev.scale(self.smoothing).add(force.scale(1.0 - self.smoothing));
        self.prev = result;
        result
    }

    /// Reset the filter state.
    pub fn reset(&mut self) {
        self.prev = Vec2::ZERO;
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_agent(x: f64, y: f64) -> Agent {
        Agent::new(Vec2::new(x, y), 10.0, 5.0)
    }

    fn make_moving_agent(x: f64, y: f64, vx: f64, vy: f64) -> Agent {
        let mut a = make_agent(x, y);
        a.velocity = Vec2::new(vx, vy);
        a
    }

    #[test]
    fn test_vec2_length() {
        let v = Vec2::new(3.0, 4.0);
        assert!((v.length() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec2_normalized() {
        let v = Vec2::new(3.0, 4.0).normalized();
        assert!((v.length() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec2_truncate() {
        let v = Vec2::new(10.0, 0.0).truncate(3.0);
        assert!((v.length() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec2_truncate_no_change() {
        let v = Vec2::new(2.0, 0.0).truncate(5.0);
        assert!((v.length() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_seek_toward_target() {
        let agent = make_agent(0.0, 0.0);
        let force = seek(&agent, Vec2::new(10.0, 0.0));
        assert!(force.x > 0.0);
    }

    #[test]
    fn test_flee_away_from_target() {
        let agent = make_agent(0.0, 0.0);
        let force = flee(&agent, Vec2::new(10.0, 0.0));
        assert!(force.x < 0.0);
    }

    #[test]
    fn test_arrive_slows_near_target() {
        let agent = make_moving_agent(9.0, 0.0, 5.0, 0.0);
        let force_far = arrive(&agent, Vec2::new(100.0, 0.0), 5.0);
        let force_near = arrive(&make_moving_agent(9.5, 0.0, 5.0, 0.0), Vec2::new(10.0, 0.0), 5.0);
        // Near target: force should brake (negative contribution to velocity)
        let net_far = force_far.x + agent.velocity.x;
        let near_agent = make_moving_agent(9.5, 0.0, 5.0, 0.0);
        let net_near = force_near.x + near_agent.velocity.x;
        // The near agent should have less desired speed
        assert!(net_near < net_far);
    }

    #[test]
    fn test_pursue_ahead_of_target() {
        let agent = make_agent(0.0, 0.0);
        let force = pursue(&agent, Vec2::new(10.0, 0.0), Vec2::new(5.0, 0.0));
        // Should seek ahead of target
        assert!(force.x > 0.0);
    }

    #[test]
    fn test_evade_away_from_predicted() {
        let agent = make_agent(0.0, 0.0);
        let force = evade(&agent, Vec2::new(5.0, 0.0), Vec2::new(-5.0, 0.0));
        // Target moving toward us, should flee harder
        assert!(force.x < 0.0);
    }

    #[test]
    fn test_wander_produces_force() {
        let agent = make_moving_agent(0.0, 0.0, 5.0, 0.0);
        let mut angle = 0.0;
        let force = wander(&agent, 2.0, 4.0, &mut angle, 0.5, 0.3);
        assert!(force.length() > 0.0);
    }

    #[test]
    fn test_obstacle_avoidance_no_obstacles() {
        let agent = make_moving_agent(0.0, 0.0, 5.0, 0.0);
        let force = obstacle_avoidance(&agent, &[], 10.0);
        assert!((force.length()).abs() < 1e-10);
    }

    #[test]
    fn test_obstacle_avoidance_steers_away() {
        let agent = make_moving_agent(0.0, 0.0, 5.0, 0.0);
        let obstacles = vec![Obstacle { center: Vec2::new(5.0, 0.5), radius: 1.0 }];
        let force = obstacle_avoidance(&agent, &obstacles, 10.0);
        // Should steer perpendicular to heading
        assert!(force.length() > 0.0);
        assert!(force.y.abs() > 0.1); // lateral steer
    }

    #[test]
    fn test_obstacle_behind_ignored() {
        let agent = make_moving_agent(0.0, 0.0, 5.0, 0.0);
        let obstacles = vec![Obstacle { center: Vec2::new(-5.0, 0.0), radius: 1.0 }];
        let force = obstacle_avoidance(&agent, &obstacles, 10.0);
        assert!((force.length()).abs() < 1e-10);
    }

    #[test]
    fn test_wall_following() {
        let agent = make_moving_agent(0.0, 0.5, 5.0, 0.0);
        let wall = Wall {
            start: Vec2::new(-10.0, 0.0),
            end: Vec2::new(10.0, 0.0),
        };
        let force = wall_following(&agent, &wall, 2.0);
        // Agent is too close (0.5 < 2.0), should push away from wall (positive y)
        assert!(force.y > 0.0);
    }

    #[test]
    fn test_wall_following_far_enough() {
        let agent = make_moving_agent(0.0, 5.0, 5.0, 0.0);
        let wall = Wall {
            start: Vec2::new(-10.0, 0.0),
            end: Vec2::new(10.0, 0.0),
        };
        let force = wall_following(&agent, &wall, 2.0);
        assert!((force.length()).abs() < 1e-10);
    }

    #[test]
    fn test_path_following_seeks_waypoint() {
        let agent = make_agent(0.0, 0.0);
        let waypoints = vec![Vec2::new(5.0, 0.0), Vec2::new(10.0, 0.0)];
        let mut wp = 0;
        let force = path_following(&agent, &waypoints, 1.0, &mut wp);
        assert!(force.x > 0.0);
    }

    #[test]
    fn test_path_following_advances() {
        let agent = make_agent(4.5, 0.0);
        let waypoints = vec![Vec2::new(5.0, 0.0), Vec2::new(10.0, 0.0)];
        let mut wp = 0;
        let _force = path_following(&agent, &waypoints, 1.0, &mut wp);
        assert_eq!(wp, 1); // advanced to next waypoint
    }

    #[test]
    fn test_weighted_blend() {
        let behaviors = vec![
            WeightedBehavior { force: Vec2::new(1.0, 0.0), weight: 2.0 },
            WeightedBehavior { force: Vec2::new(0.0, 1.0), weight: 1.0 },
        ];
        let result = weighted_blend(&behaviors);
        assert!((result.x - 2.0).abs() < 1e-10);
        assert!((result.y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_priority_select() {
        let behaviors = vec![
            Vec2::ZERO,
            Vec2::new(3.0, 0.0),
            Vec2::new(0.0, 5.0),
        ];
        let result = priority_select(&behaviors);
        assert!((result.x - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_priority_select_all_zero() {
        let behaviors = vec![Vec2::ZERO, Vec2::ZERO];
        let result = priority_select(&behaviors);
        assert!((result.length()).abs() < 1e-10);
    }

    #[test]
    fn test_steering_filter() {
        let mut filter = SteeringFilter::new(0.5);
        let f1 = filter.filter(Vec2::new(10.0, 0.0));
        assert!((f1.x - 5.0).abs() < 1e-6); // 0.5*0 + 0.5*10
        let f2 = filter.filter(Vec2::new(10.0, 0.0));
        assert!((f2.x - 7.5).abs() < 1e-6); // 0.5*5 + 0.5*10
    }

    #[test]
    fn test_steering_filter_reset() {
        let mut filter = SteeringFilter::new(0.5);
        filter.filter(Vec2::new(10.0, 0.0));
        filter.reset();
        let f = filter.filter(Vec2::new(4.0, 0.0));
        assert!((f.x - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_agent_apply_force() {
        let mut agent = make_agent(0.0, 0.0);
        agent.apply_force(Vec2::new(5.0, 0.0), 1.0);
        assert!(agent.velocity.x > 0.0);
        assert!(agent.position.x > 0.0);
    }

    #[test]
    fn test_agent_max_speed_clamp() {
        let mut agent = make_agent(0.0, 0.0);
        agent.apply_force(Vec2::new(100.0, 0.0), 10.0);
        assert!(agent.velocity.length() <= agent.max_speed + 1e-10);
    }

    #[test]
    fn test_agent_heading_stationary() {
        let agent = make_agent(0.0, 0.0);
        let h = agent.heading();
        assert!((h.x - 1.0).abs() < 1e-10); // default forward
    }
}
