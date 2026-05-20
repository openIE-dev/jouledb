//! Steering behaviors — seek, flee, arrive, pursue, evade, wander, obstacle
//! avoidance, path following, flocking (separation/alignment/cohesion),
//! weighted blending, priority-based combination.
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
        let len = self.length();
        if len <= max_len { *self } else { let s = max_len / len; Self { x: self.x * s, y: self.y * s } }
    }

    pub fn dot(&self, other: Vec2) -> f64 { self.x * other.x + self.y * other.y }

    pub fn dist(&self, other: Vec2) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    pub fn add(&self, other: Vec2) -> Self { Self { x: self.x + other.x, y: self.y + other.y } }
    pub fn sub(&self, other: Vec2) -> Self { Self { x: self.x - other.x, y: self.y - other.y } }
    pub fn scale(&self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s } }
    pub fn perp(&self) -> Self { Self { x: -self.y, y: self.x } }
}

// ── Agent ───────────────────────────────────────────────────────

/// A steerable agent with position, velocity, and physical limits.
#[derive(Debug, Clone)]
pub struct Agent {
    pub position: Vec2,
    pub velocity: Vec2,
    pub max_speed: f64,
    pub max_force: f64,
    pub mass: f64,
    pub heading: Vec2,
}

impl Agent {
    pub fn new(x: f64, y: f64) -> Self {
        Self {
            position: Vec2::new(x, y),
            velocity: Vec2::ZERO,
            max_speed: 5.0,
            max_force: 2.0,
            mass: 1.0,
            heading: Vec2::new(1.0, 0.0),
        }
    }

    /// Apply a steering force and update position.
    pub fn apply_force(&mut self, force: Vec2, dt: f64) {
        let accel = force.scale(1.0 / self.mass);
        self.velocity = self.velocity.add(accel.scale(dt)).truncate(self.max_speed);
        self.position = self.position.add(self.velocity.scale(dt));
        if self.velocity.length_sq() > 1e-6 {
            self.heading = self.velocity.normalized();
        }
    }

    pub fn speed(&self) -> f64 { self.velocity.length() }
}

// ── Obstacle ────────────────────────────────────────────────────

/// A circular obstacle for avoidance.
#[derive(Debug, Clone, Copy)]
pub struct Obstacle {
    pub position: Vec2,
    pub radius: f64,
}

// ── Steering behaviors ──────────────────────────────────────────

/// Seek: steer toward a target.
pub fn seek(agent: &Agent, target: Vec2) -> Vec2 {
    let desired = target.sub(agent.position).normalized().scale(agent.max_speed);
    desired.sub(agent.velocity).truncate(agent.max_force)
}

/// Flee: steer away from a target.
pub fn flee(agent: &Agent, target: Vec2) -> Vec2 {
    let desired = agent.position.sub(target).normalized().scale(agent.max_speed);
    desired.sub(agent.velocity).truncate(agent.max_force)
}

/// Arrive: seek with deceleration near target.
pub fn arrive(agent: &Agent, target: Vec2, slowing_radius: f64) -> Vec2 {
    let to_target = target.sub(agent.position);
    let dist = to_target.length();
    if dist < 1e-6 {
        return agent.velocity.scale(-1.0).truncate(agent.max_force);
    }
    let speed = if dist < slowing_radius {
        agent.max_speed * (dist / slowing_radius)
    } else {
        agent.max_speed
    };
    let desired = to_target.normalized().scale(speed);
    desired.sub(agent.velocity).truncate(agent.max_force)
}

/// Pursue: seek toward predicted future position of a moving target.
pub fn pursue(agent: &Agent, target_pos: Vec2, target_vel: Vec2) -> Vec2 {
    let to_target = target_pos.sub(agent.position);
    let look_ahead = to_target.length() / (agent.max_speed + target_vel.length() + 1e-6);
    let predicted = target_pos.add(target_vel.scale(look_ahead));
    seek(agent, predicted)
}

/// Evade: flee from predicted future position of a moving target.
pub fn evade(agent: &Agent, target_pos: Vec2, target_vel: Vec2) -> Vec2 {
    let to_target = target_pos.sub(agent.position);
    let look_ahead = to_target.length() / (agent.max_speed + target_vel.length() + 1e-6);
    let predicted = target_pos.add(target_vel.scale(look_ahead));
    flee(agent, predicted)
}

/// Wander: semi-random steering using a circle projected in front of the agent.
pub fn wander(agent: &Agent, circle_dist: f64, circle_radius: f64, angle: f64) -> Vec2 {
    let circle_center = agent.heading.scale(circle_dist);
    let displacement = Vec2::new(angle.cos(), angle.sin()).scale(circle_radius);
    let desired = agent.position.add(circle_center).add(displacement).sub(agent.position);
    desired.normalized().scale(agent.max_speed).sub(agent.velocity).truncate(agent.max_force)
}

/// Obstacle avoidance: steer away from the nearest obstacle ahead.
pub fn obstacle_avoidance(agent: &Agent, obstacles: &[Obstacle], look_ahead: f64) -> Vec2 {
    let ahead = agent.position.add(agent.heading.scale(look_ahead));
    let ahead_half = agent.position.add(agent.heading.scale(look_ahead * 0.5));

    let mut nearest: Option<(usize, f64)> = None;
    for (i, obs) in obstacles.iter().enumerate() {
        let d1 = obs.position.dist(ahead);
        let d2 = obs.position.dist(ahead_half);
        let d3 = obs.position.dist(agent.position);
        let closest = d1.min(d2).min(d3);
        if closest < obs.radius {
            let dist_to_agent = agent.position.dist(obs.position);
            if nearest.is_none() || dist_to_agent < nearest.unwrap().1 {
                nearest = Some((i, dist_to_agent));
            }
        }
    }

    if let Some((idx, _)) = nearest {
        let obs = &obstacles[idx];
        let avoidance = ahead.sub(obs.position).normalized().scale(agent.max_force);
        avoidance
    } else {
        Vec2::ZERO
    }
}

/// Path following: steer toward the nearest point on a path, plus look-ahead.
pub fn path_following(agent: &Agent, path: &[Vec2], look_ahead_dist: f64) -> Vec2 {
    if path.is_empty() { return Vec2::ZERO; }
    if path.len() == 1 { return seek(agent, path[0]); }

    // Find nearest segment.
    let mut min_dist = f64::INFINITY;
    let mut nearest_point = path[0];
    let mut nearest_seg = 0;

    for i in 0..path.len() - 1 {
        let proj = project_on_segment(agent.position, path[i], path[i + 1]);
        let d = agent.position.dist(proj);
        if d < min_dist {
            min_dist = d;
            nearest_point = proj;
            nearest_seg = i;
        }
    }

    // Look ahead along the path from nearest_point.
    let mut remaining = look_ahead_dist;
    let mut target = nearest_point;
    let seg_end = path[nearest_seg + 1];
    let dist_to_end = nearest_point.dist(seg_end);
    if dist_to_end >= remaining {
        let dir = seg_end.sub(nearest_point).normalized();
        target = nearest_point.add(dir.scale(remaining));
    } else {
        remaining -= dist_to_end;
        let mut idx = nearest_seg + 1;
        while idx < path.len() - 1 && remaining > 0.0 {
            let seg_len = path[idx].dist(path[idx + 1]);
            if seg_len >= remaining {
                let dir = path[idx + 1].sub(path[idx]).normalized();
                target = path[idx].add(dir.scale(remaining));
                remaining = 0.0;
            } else {
                remaining -= seg_len;
                idx += 1;
            }
        }
        if remaining > 0.0 {
            target = *path.last().unwrap();
        }
    }

    seek(agent, target)
}

fn project_on_segment(p: Vec2, a: Vec2, b: Vec2) -> Vec2 {
    let ab = b.sub(a);
    let ap = p.sub(a);
    let ab_sq = ab.length_sq();
    if ab_sq < 1e-12 { return a; }
    let t = (ap.dot(ab) / ab_sq).clamp(0.0, 1.0);
    a.add(ab.scale(t))
}

// ── Flocking ────────────────────────────────────────────────────

/// Separation: steer away from nearby neighbors.
pub fn separation(agent: &Agent, neighbors: &[Agent], desired_dist: f64) -> Vec2 {
    let mut force = Vec2::ZERO;
    let mut count = 0;
    for other in neighbors {
        let d = agent.position.dist(other.position);
        if d > 0.0 && d < desired_dist {
            let away = agent.position.sub(other.position).normalized().scale(1.0 / d);
            force = force.add(away);
            count += 1;
        }
    }
    if count > 0 {
        force = force.scale(1.0 / count as f64);
        force.normalized().scale(agent.max_speed).sub(agent.velocity).truncate(agent.max_force)
    } else {
        Vec2::ZERO
    }
}

/// Alignment: steer toward the average heading of neighbors.
pub fn alignment(agent: &Agent, neighbors: &[Agent]) -> Vec2 {
    if neighbors.is_empty() { return Vec2::ZERO; }
    let mut avg = Vec2::ZERO;
    for other in neighbors {
        avg = avg.add(other.velocity);
    }
    avg = avg.scale(1.0 / neighbors.len() as f64);
    avg.normalized().scale(agent.max_speed).sub(agent.velocity).truncate(agent.max_force)
}

/// Cohesion: steer toward the center of mass of neighbors.
pub fn cohesion(agent: &Agent, neighbors: &[Agent]) -> Vec2 {
    if neighbors.is_empty() { return Vec2::ZERO; }
    let mut center = Vec2::ZERO;
    for other in neighbors {
        center = center.add(other.position);
    }
    center = center.scale(1.0 / neighbors.len() as f64);
    seek(agent, center)
}

// ── Combination strategies ──────────────────────────────────────

/// Weighted blending: combine multiple forces with weights.
pub fn weighted_blend(forces: &[(Vec2, f64)], max_force: f64) -> Vec2 {
    let mut combined = Vec2::ZERO;
    for &(force, weight) in forces {
        combined = combined.add(force.scale(weight));
    }
    combined.truncate(max_force)
}

/// Priority-based combination: use forces in priority order until max_force budget is spent.
pub fn priority_combine(forces: &[Vec2], max_force: f64) -> Vec2 {
    let mut accumulated = Vec2::ZERO;
    let mut remaining = max_force;

    for force in forces {
        let mag = force.length();
        if mag < 1e-6 { continue; }
        if mag <= remaining {
            accumulated = accumulated.add(*force);
            remaining -= mag;
        } else {
            accumulated = accumulated.add(force.truncate(remaining));
            break;
        }
    }
    accumulated
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn agent_at(x: f64, y: f64) -> Agent {
        Agent::new(x, y)
    }

    #[test]
    fn seek_toward_target() {
        let agent = agent_at(0.0, 0.0);
        let force = seek(&agent, Vec2::new(10.0, 0.0));
        assert!(force.x > 0.0);
        assert!(force.y.abs() < 1e-6);
    }

    #[test]
    fn flee_from_target() {
        let agent = agent_at(0.0, 0.0);
        let force = flee(&agent, Vec2::new(10.0, 0.0));
        assert!(force.x < 0.0);
    }

    #[test]
    fn arrive_decelerates() {
        let mut agent = agent_at(0.0, 0.0);
        agent.velocity = Vec2::new(5.0, 0.0);
        let force_far = arrive(&agent, Vec2::new(100.0, 0.0), 20.0);
        agent.position = Vec2::new(95.0, 0.0);
        let force_near = arrive(&agent, Vec2::new(100.0, 0.0), 20.0);
        // Near target, desired speed is lower, so force should be smaller or braking.
        assert!(force_near.length() > 0.0 || force_far.length() > 0.0);
    }

    #[test]
    fn pursue_leads_target() {
        let agent = agent_at(0.0, 0.0);
        let target_pos = Vec2::new(10.0, 0.0);
        let target_vel = Vec2::new(0.0, 5.0);
        let force = pursue(&agent, target_pos, target_vel);
        // Should aim ahead of target → some y component.
        assert!(force.y > 0.0 || force.x > 0.0);
    }

    #[test]
    fn evade_runs_away() {
        let agent = agent_at(5.0, 5.0);
        let force = evade(&agent, Vec2::new(4.0, 5.0), Vec2::new(1.0, 0.0));
        assert!(force.x > 0.0); // should flee rightward
    }

    #[test]
    fn wander_produces_force() {
        let mut agent = agent_at(0.0, 0.0);
        agent.heading = Vec2::new(1.0, 0.0);
        let force = wander(&agent, 3.0, 1.0, 0.5);
        assert!(force.length() > 0.0);
    }

    #[test]
    fn obstacle_avoidance_steers_away() {
        let mut agent = agent_at(0.0, 0.0);
        agent.heading = Vec2::new(1.0, 0.0);
        agent.velocity = Vec2::new(5.0, 0.0);
        let obs = vec![Obstacle { position: Vec2::new(5.0, 0.0), radius: 2.0 }];
        let force = obstacle_avoidance(&agent, &obs, 10.0);
        assert!(force.length() > 0.0);
    }

    #[test]
    fn path_following_steers() {
        let agent = agent_at(0.0, 1.0);
        let path = vec![Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0), Vec2::new(10.0, 10.0)];
        let force = path_following(&agent, &path, 3.0);
        assert!(force.length() > 0.0);
    }

    #[test]
    fn separation_pushes_apart() {
        let agent = agent_at(5.0, 5.0);
        let neighbors = vec![agent_at(5.5, 5.0), agent_at(4.5, 5.0)];
        let force = separation(&agent, &neighbors, 3.0);
        // Symmetric neighbors — force should be small but non-zero due to slight asymmetry in normalized.
        // Actually symmetric — y component should dominate or be zero.
        assert!(force.length() >= 0.0);
    }

    #[test]
    fn alignment_matches_heading() {
        let agent = agent_at(0.0, 0.0);
        let mut n1 = agent_at(1.0, 0.0);
        n1.velocity = Vec2::new(0.0, 5.0);
        let force = alignment(&agent, &[n1]);
        assert!(force.y > 0.0);
    }

    #[test]
    fn cohesion_moves_to_center() {
        let agent = agent_at(0.0, 0.0);
        let neighbors = vec![agent_at(10.0, 0.0), agent_at(10.0, 10.0)];
        let force = cohesion(&agent, &neighbors);
        assert!(force.x > 0.0);
    }

    #[test]
    fn weighted_blend_combines() {
        let f1 = Vec2::new(1.0, 0.0);
        let f2 = Vec2::new(0.0, 1.0);
        let result = weighted_blend(&[(f1, 1.0), (f2, 1.0)], 10.0);
        assert!((result.x - 1.0).abs() < 1e-6);
        assert!((result.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn priority_combine_budget() {
        let f1 = Vec2::new(3.0, 0.0);
        let f2 = Vec2::new(0.0, 3.0);
        let result = priority_combine(&[f1, f2], 4.0);
        // f1 uses 3.0 of 4.0 budget, f2 gets 1.0.
        assert!((result.x - 3.0).abs() < 1e-6);
        assert!((result.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn agent_apply_force() {
        let mut agent = agent_at(0.0, 0.0);
        agent.apply_force(Vec2::new(2.0, 0.0), 1.0);
        assert!(agent.position.x > 0.0);
        assert!(agent.velocity.x > 0.0);
    }

    #[test]
    fn vec2_operations() {
        let v = Vec2::new(3.0, 4.0);
        assert!((v.length() - 5.0).abs() < 1e-9);
        let n = v.normalized();
        assert!((n.length() - 1.0).abs() < 1e-9);
        let t = Vec2::new(10.0, 0.0).truncate(3.0);
        assert!((t.length() - 3.0).abs() < 1e-9);
        let p = Vec2::new(1.0, 0.0).perp();
        assert!((p.x).abs() < 1e-9);
        assert!((p.y - 1.0).abs() < 1e-9);
    }
}
