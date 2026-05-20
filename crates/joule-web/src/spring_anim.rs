//! Spring physics animation engine.
//!
//! Replaces react-spring / Framer Motion spring animations. Models a
//! damped harmonic oscillator using semi-implicit Euler integration.
//! Supports overdamped, underdamped, and critically-damped springs,
//! preset configurations, and spring chains.

use std::fmt;

// ── Configuration ──────────────────────────────────────────────

/// Damping classification of a spring system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DampingType {
    Underdamped,
    CriticallyDamped,
    Overdamped,
}

/// Configuration parameters for a spring.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpringConfig {
    /// Spring stiffness (N/m). Higher = snappier.
    pub stiffness: f64,
    /// Damping coefficient (Ns/m). Higher = less oscillation.
    pub damping: f64,
    /// Mass of the object (kg).
    pub mass: f64,
    /// Initial velocity (units/s).
    pub initial_velocity: f64,
}

impl SpringConfig {
    pub fn new(stiffness: f64, damping: f64, mass: f64) -> Self {
        Self { stiffness, damping, mass, initial_velocity: 0.0 }
    }

    pub fn with_initial_velocity(mut self, vel: f64) -> Self {
        self.initial_velocity = vel;
        self
    }

    /// Gentle spring — slow settle, low stiffness.
    pub fn gentle() -> Self {
        Self { stiffness: 120.0, damping: 14.0, mass: 1.0, initial_velocity: 0.0 }
    }

    /// Wobbly spring — low damping, lots of oscillation.
    pub fn wobbly() -> Self {
        Self { stiffness: 180.0, damping: 12.0, mass: 1.0, initial_velocity: 0.0 }
    }

    /// Stiff spring — high stiffness, moderate damping.
    pub fn stiff() -> Self {
        Self { stiffness: 400.0, damping: 28.0, mass: 1.0, initial_velocity: 0.0 }
    }

    /// Slow spring — very low stiffness, heavy mass feel.
    pub fn slow() -> Self {
        Self { stiffness: 60.0, damping: 20.0, mass: 3.0, initial_velocity: 0.0 }
    }

    /// Determine damping type from the system parameters.
    pub fn damping_type(&self) -> DampingType {
        let critical = 2.0 * (self.stiffness * self.mass).sqrt();
        let ratio = self.damping / critical;
        if (ratio - 1.0).abs() < 1e-6 {
            DampingType::CriticallyDamped
        } else if ratio < 1.0 {
            DampingType::Underdamped
        } else {
            DampingType::Overdamped
        }
    }
}

impl Default for SpringConfig {
    fn default() -> Self {
        Self::stiff()
    }
}

// ── State ──────────────────────────────────────────────────────

/// Current state of a spring simulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpringState {
    /// Current position.
    pub position: f64,
    /// Current velocity (units/s).
    pub velocity: f64,
}

// ── Spring ─────────────────────────────────────────────────────

/// A single spring animating from a start value to a target value.
#[derive(Debug, Clone)]
pub struct Spring {
    config: SpringConfig,
    from: f64,
    target: f64,
    state: SpringState,
    rest_threshold: f64,
    velocity_threshold: f64,
}

impl Spring {
    /// Create a new spring from `from` to `target`.
    pub fn new(from: f64, target: f64, config: SpringConfig) -> Self {
        Self {
            config,
            from,
            target,
            state: SpringState {
                position: from,
                velocity: config.initial_velocity,
            },
            rest_threshold: 0.001,
            velocity_threshold: 0.001,
        }
    }

    /// Override rest detection thresholds.
    pub fn with_thresholds(mut self, position: f64, velocity: f64) -> Self {
        self.rest_threshold = position;
        self.velocity_threshold = velocity;
        self
    }

    /// Get the current state.
    pub fn state(&self) -> SpringState {
        self.state
    }

    /// Get the target value.
    pub fn target(&self) -> f64 {
        self.target
    }

    /// Get the start value.
    pub fn from_value(&self) -> f64 {
        self.from
    }

    /// Set a new target, keeping current position/velocity.
    pub fn set_target(&mut self, target: f64) {
        self.target = target;
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.state.position = self.from;
        self.state.velocity = self.config.initial_velocity;
    }

    /// Is the spring at rest (position ≈ target and velocity ≈ 0)?
    pub fn is_at_rest(&self) -> bool {
        let pos_diff = (self.state.position - self.target).abs();
        let vel = self.state.velocity.abs();
        pos_diff < self.rest_threshold && vel < self.velocity_threshold
    }

    /// Advance the spring simulation by `dt` seconds using semi-implicit Euler.
    pub fn step(&mut self, dt: f64) -> SpringState {
        if self.is_at_rest() {
            self.state.position = self.target;
            self.state.velocity = 0.0;
            return self.state;
        }

        let displacement = self.state.position - self.target;

        // F = -kx - cv
        let spring_force = -self.config.stiffness * displacement;
        let damping_force = -self.config.damping * self.state.velocity;
        let acceleration = (spring_force + damping_force) / self.config.mass;

        // Semi-implicit Euler: update velocity first, then position.
        self.state.velocity += acceleration * dt;
        self.state.position += self.state.velocity * dt;

        // Snap to rest if close enough.
        if self.is_at_rest() {
            self.state.position = self.target;
            self.state.velocity = 0.0;
        }

        self.state
    }

    /// Run the simulation until at rest or `max_steps` reached.
    /// Returns the number of steps taken.
    pub fn run_to_rest(&mut self, dt: f64, max_steps: usize) -> usize {
        for i in 0..max_steps {
            self.step(dt);
            if self.is_at_rest() {
                return i + 1;
            }
        }
        max_steps
    }
}

impl fmt::Display for Spring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Spring({:.2} -> {:.2}, pos={:.4}, vel={:.4})",
            self.from, self.target, self.state.position, self.state.velocity
        )
    }
}

// ── Spring Chain ───────────────────────────────────────────────

/// A chain of springs that execute in sequence. Each spring starts when the
/// previous one reaches rest.
#[derive(Debug, Clone)]
pub struct SpringChain {
    springs: Vec<Spring>,
    current_index: usize,
}

impl SpringChain {
    pub fn new(springs: Vec<Spring>) -> Self {
        Self { springs, current_index: 0 }
    }

    /// Convenience: build a chain from (from, to) pairs using a shared config.
    pub fn from_values(values: &[(f64, f64)], config: SpringConfig) -> Self {
        let springs = values
            .iter()
            .map(|(from, to)| Spring::new(*from, *to, config))
            .collect();
        Self { springs, current_index: 0 }
    }

    /// Get the current spring (if any remain).
    pub fn current_spring(&self) -> Option<&Spring> {
        self.springs.get(self.current_index)
    }

    /// Is the entire chain complete?
    pub fn is_complete(&self) -> bool {
        self.current_index >= self.springs.len()
    }

    /// Current position (from the active spring, or the last spring's target).
    pub fn position(&self) -> f64 {
        if let Some(spring) = self.springs.get(self.current_index) {
            spring.state().position
        } else if let Some(last) = self.springs.last() {
            last.target()
        } else {
            0.0
        }
    }

    /// Advance the chain by dt seconds.
    pub fn step(&mut self, dt: f64) -> f64 {
        if self.is_complete() {
            return self.position();
        }

        let spring = &mut self.springs[self.current_index];
        spring.step(dt);

        if spring.is_at_rest() {
            self.current_index += 1;
        }

        self.position()
    }

    /// Reset the entire chain.
    pub fn reset(&mut self) {
        self.current_index = 0;
        for spring in &mut self.springs {
            spring.reset();
        }
    }

    /// Number of springs in the chain.
    pub fn len(&self) -> usize {
        self.springs.len()
    }

    /// Whether the chain has no springs.
    pub fn is_empty(&self) -> bool {
        self.springs.is_empty()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spring_reaches_target() {
        let mut spring = Spring::new(0.0, 100.0, SpringConfig::stiff());
        spring.run_to_rest(1.0 / 60.0, 600);
        assert!(spring.is_at_rest());
        assert!((spring.state().position - 100.0).abs() < 0.01);
    }

    #[test]
    fn spring_with_initial_velocity() {
        let config = SpringConfig::stiff().with_initial_velocity(500.0);
        let mut spring = Spring::new(0.0, 100.0, config);
        // After a few steps the velocity should push it past the target initially.
        for _ in 0..5 {
            spring.step(1.0 / 60.0);
        }
        assert!(spring.state().position > 0.0);
    }

    #[test]
    fn damping_types() {
        // Underdamped: damping < 2*sqrt(k*m)
        let under = SpringConfig::new(100.0, 5.0, 1.0);
        assert_eq!(under.damping_type(), DampingType::Underdamped);

        // Critically damped: damping = 2*sqrt(k*m)
        let critical_d = 2.0 * (100.0_f64).sqrt();
        let crit = SpringConfig::new(100.0, critical_d, 1.0);
        assert_eq!(crit.damping_type(), DampingType::CriticallyDamped);

        // Overdamped: damping > 2*sqrt(k*m)
        let over = SpringConfig::new(100.0, 30.0, 1.0);
        assert_eq!(over.damping_type(), DampingType::Overdamped);
    }

    #[test]
    fn gentle_preset_is_underdamped() {
        let g = SpringConfig::gentle();
        assert_eq!(g.damping_type(), DampingType::Underdamped);
    }

    #[test]
    fn wobbly_preset_oscillates() {
        let mut spring = Spring::new(0.0, 100.0, SpringConfig::wobbly());
        let mut crossed_target = false;
        for _ in 0..300 {
            spring.step(1.0 / 60.0);
            if spring.state().position > 100.0 {
                crossed_target = true;
                break;
            }
        }
        assert!(crossed_target, "Wobbly spring should overshoot");
    }

    #[test]
    fn overdamped_no_overshoot() {
        let config = SpringConfig::new(100.0, 40.0, 1.0);
        assert_eq!(config.damping_type(), DampingType::Overdamped);
        let mut spring = Spring::new(0.0, 100.0, config);
        for _ in 0..1000 {
            spring.step(1.0 / 60.0);
            assert!(spring.state().position <= 100.01, "Overdamped should not overshoot");
        }
    }

    #[test]
    fn set_target_midway() {
        let mut spring = Spring::new(0.0, 100.0, SpringConfig::stiff());
        for _ in 0..30 {
            spring.step(1.0 / 60.0);
        }
        let mid_pos = spring.state().position;
        assert!(mid_pos > 0.0 && mid_pos < 100.0);

        spring.set_target(200.0);
        spring.run_to_rest(1.0 / 60.0, 1000);
        assert!(spring.is_at_rest());
        assert!((spring.state().position - 200.0).abs() < 0.01);
    }

    #[test]
    fn reset_returns_to_start() {
        let mut spring = Spring::new(10.0, 50.0, SpringConfig::stiff());
        spring.run_to_rest(1.0 / 60.0, 600);
        spring.reset();
        assert!((spring.state().position - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn chain_executes_in_order() {
        let config = SpringConfig::stiff();
        let mut chain = SpringChain::from_values(&[(0.0, 50.0), (50.0, 100.0)], config);
        assert!(!chain.is_complete());
        assert_eq!(chain.len(), 2);

        // Run until the first spring settles.
        for _ in 0..600 {
            chain.step(1.0 / 60.0);
            if chain.current_index > 0 {
                break;
            }
        }
        // First spring done, second should be active or chain done.
        assert!(chain.current_index >= 1);

        // Run the rest.
        for _ in 0..600 {
            chain.step(1.0 / 60.0);
        }
        assert!(chain.is_complete());
        assert!((chain.position() - 100.0).abs() < 0.01);
    }

    #[test]
    fn chain_reset() {
        let config = SpringConfig::stiff();
        let mut chain = SpringChain::from_values(&[(0.0, 50.0)], config);
        for _ in 0..600 {
            chain.step(1.0 / 60.0);
        }
        assert!(chain.is_complete());
        chain.reset();
        assert!(!chain.is_complete());
        assert!((chain.position() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn slow_preset_converges() {
        let mut spring = Spring::new(0.0, 1.0, SpringConfig::slow());
        spring.run_to_rest(1.0 / 60.0, 2000);
        assert!(spring.is_at_rest());
    }

    #[test]
    fn display_format() {
        let spring = Spring::new(0.0, 100.0, SpringConfig::stiff());
        let s = format!("{spring}");
        assert!(s.contains("Spring("));
        assert!(s.contains("0.00"));
        assert!(s.contains("100.00"));
    }
}
