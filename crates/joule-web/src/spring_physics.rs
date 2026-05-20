//! Spring/constraint physics — Hooke's law springs, damped springs, spring chains,
//! cloth simulation (grid of springs), Verlet integration, distance constraints, pin constraints.

// ── Vec2 ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self { Self { x, y } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0 } }
    pub fn add(self, o: Self) -> Self { Self { x: self.x + o.x, y: self.y + o.y } }
    pub fn sub(self, o: Self) -> Self { Self { x: self.x - o.x, y: self.y - o.y } }
    pub fn scale(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s } }
    pub fn dot(self, o: Self) -> f64 { self.x * o.x + self.y * o.y }
    pub fn cross(self, o: Self) -> f64 { self.x * o.y - self.y * o.x }
    pub fn length(self) -> f64 { (self.x * self.x + self.y * self.y).sqrt() }
    pub fn length_sq(self) -> f64 { self.x * self.x + self.y * self.y }
    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-12 { Self::zero() } else { Self { x: self.x / len, y: self.y / len } }
    }
    pub fn distance(self, o: Self) -> f64 { self.sub(o).length() }
}

impl Default for Vec2 {
    fn default() -> Self { Self::zero() }
}

// ── VerletParticle ───────────────────────────────────────────

/// A particle using Verlet integration (position-based).
#[derive(Debug, Clone, PartialEq)]
pub struct VerletParticle {
    pub position: Vec2,
    /// Previous position (for Verlet integration).
    pub prev_position: Vec2,
    pub acceleration: Vec2,
    /// Inverse mass (0 = pinned / infinite mass).
    pub inv_mass: f64,
    /// Whether this particle is pinned (immovable).
    pub pinned: bool,
    /// Pin target (where the particle should stay if pinned).
    pub pin_target: Vec2,
    /// Damping factor [0, 1]. 0 = no damping, 1 = full stop.
    pub damping: f64,
}

impl VerletParticle {
    pub fn new(x: f64, y: f64, mass: f64) -> Self {
        let inv_mass = if mass <= 0.0 { 0.0 } else { 1.0 / mass };
        let pos = Vec2::new(x, y);
        Self {
            position: pos,
            prev_position: pos,
            acceleration: Vec2::zero(),
            inv_mass,
            pinned: false,
            pin_target: pos,
            damping: 0.01,
        }
    }

    /// Pin this particle at its current position.
    pub fn pin(&mut self) {
        self.pinned = true;
        self.pin_target = self.position;
    }

    /// Pin this particle at a specific position.
    pub fn pin_at(&mut self, x: f64, y: f64) {
        self.pinned = true;
        self.pin_target = Vec2::new(x, y);
        self.position = self.pin_target;
        self.prev_position = self.pin_target;
    }

    /// Unpin this particle.
    pub fn unpin(&mut self) {
        self.pinned = false;
    }

    /// Apply a force (accumulated as acceleration).
    pub fn apply_force(&mut self, force: Vec2) {
        if self.pinned { return; }
        self.acceleration = self.acceleration.add(force.scale(self.inv_mass));
    }

    /// Verlet integration step.
    pub fn integrate(&mut self, dt: f64) {
        if self.pinned {
            self.position = self.pin_target;
            self.prev_position = self.pin_target;
            self.acceleration = Vec2::zero();
            return;
        }

        let vel = self.position.sub(self.prev_position).scale(1.0 - self.damping);
        let new_pos = self.position.add(vel).add(self.acceleration.scale(dt * dt));
        self.prev_position = self.position;
        self.position = new_pos;
        self.acceleration = Vec2::zero();
    }

    /// Current velocity estimate.
    pub fn velocity(&self, dt: f64) -> Vec2 {
        if dt < 1e-12 { return Vec2::zero(); }
        self.position.sub(self.prev_position).scale(1.0 / dt)
    }
}

// ── Spring ───────────────────────────────────────────────────

/// A spring connecting two particles (by index).
#[derive(Debug, Clone, PartialEq)]
pub struct Spring {
    /// Index of particle A.
    pub a: usize,
    /// Index of particle B.
    pub b: usize,
    /// Rest length.
    pub rest_length: f64,
    /// Stiffness (spring constant k in Hooke's law).
    pub stiffness: f64,
    /// Damping coefficient.
    pub damping: f64,
}

impl Spring {
    pub fn new(a: usize, b: usize, rest_length: f64, stiffness: f64) -> Self {
        Self { a, b, rest_length, stiffness, damping: 0.0 }
    }

    pub fn with_damping(mut self, damping: f64) -> Self {
        self.damping = damping;
        self
    }

    /// Compute Hooke's law force for this spring.
    /// Returns (force_on_a, force_on_b).
    pub fn compute_force(&self, pos_a: Vec2, pos_b: Vec2, vel_a: Vec2, vel_b: Vec2) -> (Vec2, Vec2) {
        let delta = pos_b.sub(pos_a);
        let dist = delta.length();
        if dist < 1e-12 {
            return (Vec2::zero(), Vec2::zero());
        }
        let direction = delta.normalized();

        // Hooke's law: F = -k * (x - rest_length)
        let displacement = dist - self.rest_length;
        let spring_force = self.stiffness * displacement;

        // Damping: F_d = -d * v_relative_along_spring
        let relative_vel = vel_b.sub(vel_a);
        let damping_force = self.damping * relative_vel.dot(direction);

        let total = spring_force + damping_force;
        let force = direction.scale(total);

        (force, force.scale(-1.0))
    }
}

// ── DistanceConstraint ───────────────────────────────────────

/// Distance constraint solved via position projection.
#[derive(Debug, Clone, PartialEq)]
pub struct DistanceConstraint {
    pub a: usize,
    pub b: usize,
    pub rest_length: f64,
    /// Stiffness in [0, 1]. 1 = rigid.
    pub stiffness: f64,
}

impl DistanceConstraint {
    pub fn new(a: usize, b: usize, rest_length: f64) -> Self {
        Self { a, b, rest_length, stiffness: 1.0 }
    }

    pub fn with_stiffness(mut self, stiffness: f64) -> Self {
        self.stiffness = stiffness.clamp(0.0, 1.0);
        self
    }

    /// Solve this constraint by adjusting particle positions.
    pub fn solve(&self, particles: &mut [VerletParticle]) {
        let pos_a = particles[self.a].position;
        let pos_b = particles[self.b].position;
        let inv_ma = if particles[self.a].pinned { 0.0 } else { particles[self.a].inv_mass };
        let inv_mb = if particles[self.b].pinned { 0.0 } else { particles[self.b].inv_mass };

        let total_inv = inv_ma + inv_mb;
        if total_inv < 1e-12 { return; }

        let delta = pos_b.sub(pos_a);
        let dist = delta.length();
        if dist < 1e-12 { return; }

        let diff = (dist - self.rest_length) / dist;
        let correction = delta.scale(diff * self.stiffness / total_inv);

        if !particles[self.a].pinned {
            particles[self.a].position = particles[self.a].position.add(correction.scale(inv_ma));
        }
        if !particles[self.b].pinned {
            particles[self.b].position = particles[self.b].position.sub(correction.scale(inv_mb));
        }
    }
}

// ── SpringSystem ─────────────────────────────────────────────

/// A system of particles connected by springs and constraints.
#[derive(Debug, Clone)]
pub struct SpringSystem {
    pub particles: Vec<VerletParticle>,
    pub springs: Vec<Spring>,
    pub constraints: Vec<DistanceConstraint>,
    /// Global gravity.
    pub gravity: Vec2,
    /// Number of constraint solving iterations.
    pub constraint_iterations: u32,
}

impl SpringSystem {
    pub fn new() -> Self {
        Self {
            particles: Vec::new(),
            springs: Vec::new(),
            constraints: Vec::new(),
            gravity: Vec2::new(0.0, -9.81),
            constraint_iterations: 4,
        }
    }

    /// Add a particle and return its index.
    pub fn add_particle(&mut self, x: f64, y: f64, mass: f64) -> usize {
        let idx = self.particles.len();
        self.particles.push(VerletParticle::new(x, y, mass));
        idx
    }

    /// Add a spring between two particles.
    pub fn add_spring(&mut self, a: usize, b: usize, stiffness: f64, damping: f64) -> usize {
        let rest = self.particles[a].position.distance(self.particles[b].position);
        let idx = self.springs.len();
        self.springs.push(Spring::new(a, b, rest, stiffness).with_damping(damping));
        idx
    }

    /// Add a distance constraint between two particles.
    pub fn add_constraint(&mut self, a: usize, b: usize) -> usize {
        let rest = self.particles[a].position.distance(self.particles[b].position);
        let idx = self.constraints.len();
        self.constraints.push(DistanceConstraint::new(a, b, rest));
        idx
    }

    /// Pin a particle at its current position.
    pub fn pin_particle(&mut self, idx: usize) {
        self.particles[idx].pin();
    }

    /// Step the simulation.
    pub fn step(&mut self, dt: f64) {
        // Apply gravity
        let g = self.gravity;
        for p in &mut self.particles {
            if !p.pinned {
                p.apply_force(g.scale(1.0 / p.inv_mass.max(1e-12)));
            }
        }

        // Apply spring forces (Hooke's law)
        let springs: Vec<Spring> = self.springs.clone();
        for spring in &springs {
            let pos_a = self.particles[spring.a].position;
            let pos_b = self.particles[spring.b].position;
            let vel_a = self.particles[spring.a].velocity(dt);
            let vel_b = self.particles[spring.b].velocity(dt);
            let (fa, fb) = spring.compute_force(pos_a, pos_b, vel_a, vel_b);
            self.particles[spring.a].apply_force(fa);
            self.particles[spring.b].apply_force(fb);
        }

        // Integrate
        for p in &mut self.particles {
            p.integrate(dt);
        }

        // Solve constraints iteratively
        let constraints: Vec<DistanceConstraint> = self.constraints.clone();
        for _ in 0..self.constraint_iterations {
            for c in &constraints {
                c.solve(&mut self.particles);
            }
        }
    }

    /// Total potential energy stored in springs.
    pub fn spring_energy(&self) -> f64 {
        let mut energy = 0.0;
        for s in &self.springs {
            let dist = self.particles[s.a].position.distance(self.particles[s.b].position);
            let dx = dist - s.rest_length;
            energy += 0.5 * s.stiffness * dx * dx;
        }
        energy
    }
}

impl Default for SpringSystem {
    fn default() -> Self { Self::new() }
}

// ── Spring Chain ─────────────────────────────────────────────

/// Create a chain of particles connected by distance constraints.
pub fn create_chain(
    system: &mut SpringSystem,
    start: Vec2,
    end: Vec2,
    segments: usize,
    mass: f64,
    pin_start: bool,
    pin_end: bool,
) -> Vec<usize> {
    if segments == 0 { return Vec::new(); }

    let mut indices = Vec::with_capacity(segments + 1);
    for i in 0..=segments {
        let t = i as f64 / segments as f64;
        let pos = start.add(end.sub(start).scale(t));
        let idx = system.add_particle(pos.x, pos.y, mass);
        indices.push(idx);
    }

    // Connect consecutive particles with constraints
    for i in 0..segments {
        system.add_constraint(indices[i], indices[i + 1]);
    }

    if pin_start {
        system.pin_particle(indices[0]);
    }
    if pin_end {
        system.pin_particle(*indices.last().unwrap());
    }

    indices
}

// ── Cloth Simulation ─────────────────────────────────────────

/// Create a cloth grid of particles connected by springs.
/// Returns a 2D grid of particle indices (row-major).
pub fn create_cloth(
    system: &mut SpringSystem,
    origin: Vec2,
    width: f64,
    height: f64,
    cols: usize,
    rows: usize,
    mass: f64,
    pin_top: bool,
) -> Vec<Vec<usize>> {
    if cols < 2 || rows < 2 { return Vec::new(); }

    let dx = width / (cols - 1) as f64;
    let dy = height / (rows - 1) as f64;

    let mut grid = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for c in 0..cols {
            let x = origin.x + c as f64 * dx;
            let y = origin.y - r as f64 * dy; // y goes down
            let idx = system.add_particle(x, y, mass);
            row.push(idx);
        }
        grid.push(row);
    }

    // Structural constraints: horizontal and vertical
    for r in 0..rows {
        for c in 0..cols {
            if c + 1 < cols {
                system.add_constraint(grid[r][c], grid[r][c + 1]);
            }
            if r + 1 < rows {
                system.add_constraint(grid[r][c], grid[r + 1][c]);
            }
        }
    }

    // Shear constraints: diagonals
    for r in 0..(rows - 1) {
        for c in 0..(cols - 1) {
            system.add_constraint(grid[r][c], grid[r + 1][c + 1]);
            system.add_constraint(grid[r][c + 1], grid[r + 1][c]);
        }
    }

    // Pin top row if requested
    if pin_top {
        for c in 0..cols {
            system.pin_particle(grid[0][c]);
        }
    }

    grid
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < 0.1 }

    #[test]
    fn verlet_particle_creation() {
        let p = VerletParticle::new(5.0, 10.0, 2.0);
        assert!(approx(p.position.x, 5.0));
        assert!(approx(p.inv_mass, 0.5));
        assert!(!p.pinned);
    }

    #[test]
    fn verlet_integration_no_force() {
        let mut p = VerletParticle::new(0.0, 0.0, 1.0);
        p.damping = 0.0;
        // Give initial velocity by setting prev_position
        p.prev_position = Vec2::new(-1.0, 0.0);
        p.integrate(1.0);
        // Should continue moving in +x
        assert!(p.position.x > 0.0);
    }

    #[test]
    fn verlet_pinned_stays() {
        let mut p = VerletParticle::new(5.0, 5.0, 1.0);
        p.pin();
        p.apply_force(Vec2::new(100.0, 0.0));
        p.integrate(1.0);
        assert!(approx(p.position.x, 5.0));
        assert!(approx(p.position.y, 5.0));
    }

    #[test]
    fn spring_hooke_law() {
        let s = Spring::new(0, 1, 10.0, 100.0);
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(15.0, 0.0);
        let (fa, fb) = s.compute_force(a, b, Vec2::zero(), Vec2::zero());
        // Spring stretched by 5, force = 100 * 5 = 500 toward b on a
        assert!(fa.x > 0.0); // pulls a toward b
        assert!(fb.x < 0.0); // pulls b toward a
        assert!(approx(fa.x, 500.0));
    }

    #[test]
    fn spring_at_rest() {
        let s = Spring::new(0, 1, 10.0, 100.0);
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(10.0, 0.0);
        let (fa, fb) = s.compute_force(a, b, Vec2::zero(), Vec2::zero());
        assert!(fa.length() < 0.01);
        assert!(fb.length() < 0.01);
    }

    #[test]
    fn damped_spring() {
        let s = Spring::new(0, 1, 10.0, 100.0).with_damping(10.0);
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(10.0, 0.0); // at rest length
        // But velocity of b is moving away
        let vb = Vec2::new(5.0, 0.0);
        let (fa, _fb) = s.compute_force(a, b, Vec2::zero(), vb);
        // Damping should produce a force (along the spring direction)
        assert!(fa.x.abs() > 0.0);
    }

    #[test]
    fn distance_constraint_solve() {
        let mut particles = vec![
            VerletParticle::new(0.0, 0.0, 1.0),
            VerletParticle::new(20.0, 0.0, 1.0),
        ];
        let c = DistanceConstraint::new(0, 1, 10.0);
        c.solve(&mut particles);
        let dist = particles[0].position.distance(particles[1].position);
        assert!(approx(dist, 10.0));
    }

    #[test]
    fn distance_constraint_with_pin() {
        let mut particles = vec![
            VerletParticle::new(0.0, 0.0, 1.0),
            VerletParticle::new(20.0, 0.0, 1.0),
        ];
        particles[0].pin();
        let c = DistanceConstraint::new(0, 1, 10.0);
        c.solve(&mut particles);
        // A is pinned at 0,0 so only B should move
        assert!(approx(particles[0].position.x, 0.0));
        assert!(approx(particles[1].position.x, 10.0));
    }

    #[test]
    fn spring_system_step() {
        let mut sys = SpringSystem::new();
        sys.gravity = Vec2::new(0.0, -10.0);
        let a = sys.add_particle(0.0, 10.0, 1.0);
        let b = sys.add_particle(0.0, 0.0, 1.0);
        sys.add_spring(a, b, 100.0, 5.0);
        sys.pin_particle(a);
        sys.particles[b].damping = 0.1;

        // Step several times
        for _ in 0..100 {
            sys.step(0.01);
        }
        // Particle b should be below a, pulled down by gravity, held by spring
        assert!(sys.particles[b].position.y < sys.particles[a].position.y);
    }

    #[test]
    fn spring_system_energy() {
        let mut sys = SpringSystem::new();
        sys.gravity = Vec2::zero();
        let a = sys.add_particle(0.0, 0.0, 1.0);
        let b = sys.add_particle(15.0, 0.0, 1.0);
        sys.add_spring(a, b, 100.0, 0.0);
        // springs[0].rest_length should be 15.0
        // Energy at rest = 0
        assert!(sys.spring_energy() < 0.01);
        // Move b to 20.0
        sys.particles[b].position = Vec2::new(20.0, 0.0);
        // Energy = 0.5 * 100 * 5^2 = 1250
        assert!(approx(sys.spring_energy(), 1250.0));
    }

    #[test]
    fn create_chain_basic() {
        let mut sys = SpringSystem::new();
        let indices = create_chain(
            &mut sys,
            Vec2::new(0.0, 10.0),
            Vec2::new(10.0, 10.0),
            5,
            1.0,
            true,
            false,
        );
        assert_eq!(indices.len(), 6); // 5 segments = 6 particles
        assert!(sys.particles[indices[0]].pinned);
        assert!(!sys.particles[*indices.last().unwrap()].pinned);
        assert_eq!(sys.constraints.len(), 5);
    }

    #[test]
    fn create_cloth_basic() {
        let mut sys = SpringSystem::new();
        let grid = create_cloth(
            &mut sys,
            Vec2::new(0.0, 10.0),
            10.0,
            10.0,
            4,
            3,
            1.0,
            true,
        );
        assert_eq!(grid.len(), 3); // 3 rows
        assert_eq!(grid[0].len(), 4); // 4 cols
        // Top row should be pinned
        for &idx in &grid[0] {
            assert!(sys.particles[idx].pinned);
        }
        // Other rows should not be pinned
        for &idx in &grid[1] {
            assert!(!sys.particles[idx].pinned);
        }
    }

    #[test]
    fn cloth_simulation_step() {
        let mut sys = SpringSystem::new();
        sys.gravity = Vec2::new(0.0, -10.0);
        let grid = create_cloth(
            &mut sys,
            Vec2::new(0.0, 10.0),
            4.0,
            4.0,
            3,
            3,
            1.0,
            true,
        );
        let bottom_y_before = sys.particles[grid[2][1]].position.y;
        for _ in 0..50 {
            sys.step(0.01);
        }
        let bottom_y_after = sys.particles[grid[2][1]].position.y;
        // Bottom row should have moved down due to gravity
        assert!(bottom_y_after < bottom_y_before);
    }

    #[test]
    fn constraint_stiffness() {
        let mut particles = vec![
            VerletParticle::new(0.0, 0.0, 1.0),
            VerletParticle::new(20.0, 0.0, 1.0),
        ];
        // Soft constraint (stiffness = 0.5)
        let c = DistanceConstraint::new(0, 1, 10.0).with_stiffness(0.5);
        c.solve(&mut particles);
        let dist = particles[0].position.distance(particles[1].position);
        // Should be closer to rest but not fully there (soft)
        assert!(dist < 20.0);
        assert!(dist > 10.0);
    }

    #[test]
    fn pin_at_specific_position() {
        let mut p = VerletParticle::new(0.0, 0.0, 1.0);
        p.pin_at(5.0, 5.0);
        assert!(p.pinned);
        assert!(approx(p.position.x, 5.0));
        assert!(approx(p.position.y, 5.0));
    }

    #[test]
    fn unpin_particle() {
        let mut p = VerletParticle::new(0.0, 0.0, 1.0);
        p.pin();
        assert!(p.pinned);
        p.unpin();
        assert!(!p.pinned);
    }

    #[test]
    fn chain_with_gravity() {
        let mut sys = SpringSystem::new();
        sys.gravity = Vec2::new(0.0, -10.0);
        let indices = create_chain(
            &mut sys,
            Vec2::new(0.0, 10.0),
            Vec2::new(5.0, 10.0),
            3,
            1.0,
            true,
            false,
        );
        let end_y_before = sys.particles[*indices.last().unwrap()].position.y;
        for _ in 0..100 {
            sys.step(0.01);
        }
        let end_y_after = sys.particles[*indices.last().unwrap()].position.y;
        assert!(end_y_after < end_y_before);
    }
}
