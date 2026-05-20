//! Particle simulation — emitter, particle lifecycle, forces (gravity, wind, drag),
//! color/size over lifetime, burst mode, particle pool recycling.

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
    pub fn length(self) -> f64 { (self.x * self.x + self.y * self.y).sqrt() }
    pub fn length_sq(self) -> f64 { self.x * self.x + self.y * self.y }
    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-12 { Self::zero() } else { Self { x: self.x / len, y: self.y / len } }
    }
}

impl Default for Vec2 {
    fn default() -> Self { Self::zero() }
}

// ── Color ────────────────────────────────────────────────────

/// RGBA color with components in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

impl Color {
    pub fn new(r: f64, g: f64, b: f64, a: f64) -> Self {
        Self { r, g, b, a }
    }

    pub fn white() -> Self { Self::new(1.0, 1.0, 1.0, 1.0) }
    pub fn red() -> Self { Self::new(1.0, 0.0, 0.0, 1.0) }
    pub fn transparent() -> Self { Self::new(0.0, 0.0, 0.0, 0.0) }

    /// Linearly interpolate between two colors.
    pub fn lerp(self, other: Self, t: f64) -> Self {
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }
}

impl Default for Color {
    fn default() -> Self { Self::white() }
}

// ── Particle ─────────────────────────────────────────────────

/// A single particle in the simulation.
#[derive(Debug, Clone, PartialEq)]
pub struct Particle {
    pub position: Vec2,
    pub velocity: Vec2,
    pub acceleration: Vec2,
    /// Current age in seconds.
    pub age: f64,
    /// Maximum lifetime in seconds.
    pub lifetime: f64,
    /// Current size.
    pub size: f64,
    /// Current color.
    pub color: Color,
    /// Whether this particle slot is alive.
    pub alive: bool,
    /// Initial size for interpolation.
    start_size: f64,
    /// End size for interpolation.
    end_size: f64,
    /// Start color for interpolation.
    start_color: Color,
    /// End color for interpolation.
    end_color: Color,
    /// Mass for drag computations.
    pub mass: f64,
}

impl Particle {
    fn new() -> Self {
        Self {
            position: Vec2::zero(),
            velocity: Vec2::zero(),
            acceleration: Vec2::zero(),
            age: 0.0,
            lifetime: 1.0,
            size: 1.0,
            color: Color::white(),
            alive: false,
            start_size: 1.0,
            end_size: 0.0,
            start_color: Color::white(),
            end_color: Color::transparent(),
            mass: 1.0,
        }
    }

    /// Normalized age in [0, 1].
    pub fn age_ratio(&self) -> f64 {
        if self.lifetime <= 0.0 { 1.0 } else { (self.age / self.lifetime).min(1.0) }
    }

    /// Update color and size based on age.
    fn update_visuals(&mut self) {
        let t = self.age_ratio();
        self.size = self.start_size + (self.end_size - self.start_size) * t;
        self.color = self.start_color.lerp(self.end_color, t);
    }
}

// ── Force ────────────────────────────────────────────────────

/// A force that can be applied to particles.
#[derive(Debug, Clone, PartialEq)]
pub enum Force {
    /// Constant gravity (e.g., (0, -9.81)).
    Gravity(Vec2),
    /// Constant wind force.
    Wind(Vec2),
    /// Drag coefficient. F_drag = -coeff * v * |v|
    Drag(f64),
    /// Attraction / repulsion toward a point. Positive = attract.
    Radial { center: Vec2, strength: f64 },
    /// Turbulence: pseudo-random varying force based on position.
    Turbulence { strength: f64, scale: f64 },
}

impl Force {
    /// Compute the force vector for a given particle.
    pub fn evaluate(&self, particle: &Particle) -> Vec2 {
        match self {
            Force::Gravity(g) => g.scale(particle.mass),
            Force::Wind(w) => *w,
            Force::Drag(coeff) => {
                let speed = particle.velocity.length();
                if speed < 1e-12 { Vec2::zero() }
                else { particle.velocity.normalized().scale(-coeff * speed * speed) }
            }
            Force::Radial { center, strength } => {
                let diff = center.sub(particle.position);
                let dist_sq = diff.length_sq();
                if dist_sq < 1e-12 { Vec2::zero() }
                else { diff.normalized().scale(*strength / dist_sq.sqrt().max(1.0)) }
            }
            Force::Turbulence { strength, scale } => {
                // Simple deterministic pseudo-noise based on position
                let px = particle.position.x * scale;
                let py = particle.position.y * scale;
                let fx = (px * 12.9898 + py * 78.233).sin() * 43758.5453;
                let fy = (px * 78.233 + py * 12.9898).sin() * 43758.5453;
                Vec2::new(
                    (fx - fx.floor() - 0.5) * 2.0 * strength,
                    (fy - fy.floor() - 0.5) * 2.0 * strength,
                )
            }
        }
    }
}

// ── EmitterConfig ────────────────────────────────────────────

/// Configuration for a particle emitter.
#[derive(Debug, Clone, PartialEq)]
pub struct EmitterConfig {
    /// Emitter position.
    pub position: Vec2,
    /// Emission rate (particles per second) for continuous mode.
    pub rate: f64,
    /// Emission spread angle in radians (full cone).
    pub spread: f64,
    /// Base emission direction.
    pub direction: Vec2,
    /// Initial speed range [min, max].
    pub speed_range: (f64, f64),
    /// Particle lifetime range [min, max] in seconds.
    pub lifetime_range: (f64, f64),
    /// Start size range.
    pub start_size_range: (f64, f64),
    /// End size range.
    pub end_size_range: (f64, f64),
    /// Start color.
    pub start_color: Color,
    /// End color.
    pub end_color: Color,
    /// Maximum particles (pool size).
    pub max_particles: usize,
    /// Particle mass.
    pub mass: f64,
}

impl Default for EmitterConfig {
    fn default() -> Self {
        Self {
            position: Vec2::zero(),
            rate: 10.0,
            spread: 0.5,
            direction: Vec2::new(0.0, 1.0),
            speed_range: (2.0, 5.0),
            lifetime_range: (1.0, 3.0),
            start_size_range: (1.0, 2.0),
            end_size_range: (0.0, 0.5),
            start_color: Color::white(),
            end_color: Color::transparent(),
            max_particles: 1000,
            mass: 1.0,
        }
    }
}

// ── ParticleSystem ───────────────────────────────────────────

/// Particle system with pool recycling, forces, and emitter.
#[derive(Debug, Clone)]
pub struct ParticleSystem {
    pub config: EmitterConfig,
    /// Particle pool (pre-allocated, recycled).
    particles: Vec<Particle>,
    /// Forces applied to all particles.
    forces: Vec<Force>,
    /// Accumulator for emission timing.
    emit_accum: f64,
    /// Next index to check for dead particles when recycling.
    recycle_cursor: usize,
    /// Whether the emitter is active.
    pub emitting: bool,
    /// Simple deterministic counter for pseudo-random values.
    rng_counter: u64,
    /// Total particles ever spawned.
    pub total_spawned: u64,
}

impl ParticleSystem {
    pub fn new(config: EmitterConfig) -> Self {
        let max = config.max_particles;
        let mut particles = Vec::with_capacity(max);
        for _ in 0..max {
            particles.push(Particle::new());
        }
        Self {
            config,
            particles,
            forces: Vec::new(),
            emit_accum: 0.0,
            recycle_cursor: 0,
            emitting: true,
            rng_counter: 0,
            total_spawned: 0,
        }
    }

    /// Add a force to the system.
    pub fn add_force(&mut self, force: Force) {
        self.forces.push(force);
    }

    /// Remove all forces.
    pub fn clear_forces(&mut self) {
        self.forces.clear();
    }

    /// Number of currently alive particles.
    pub fn alive_count(&self) -> usize {
        self.particles.iter().filter(|p| p.alive).count()
    }

    /// Reference to all particles (alive and dead).
    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    /// Iterate only alive particles.
    pub fn alive_particles(&self) -> impl Iterator<Item = &Particle> {
        self.particles.iter().filter(|p| p.alive)
    }

    /// Emit a single particle burst of `count` particles.
    pub fn burst(&mut self, count: usize) {
        for _ in 0..count {
            self.spawn_particle();
        }
    }

    /// Update the system by `dt` seconds.
    pub fn update(&mut self, dt: f64) {
        // Emit new particles
        if self.emitting {
            self.emit_accum += dt * self.config.rate;
            while self.emit_accum >= 1.0 {
                self.spawn_particle();
                self.emit_accum -= 1.0;
            }
        }

        // Build force list snapshot to avoid borrow issues
        let forces: Vec<Force> = self.forces.clone();

        // Update existing particles
        for p in &mut self.particles {
            if !p.alive { continue; }

            // Accumulate forces
            let mut total_force = Vec2::zero();
            for f in &forces {
                total_force = total_force.add(f.evaluate(p));
            }

            // Apply forces: a = F / m
            let accel = if p.mass > 1e-12 {
                total_force.scale(1.0 / p.mass)
            } else {
                Vec2::zero()
            };

            p.velocity = p.velocity.add(accel.add(p.acceleration).scale(dt));
            p.position = p.position.add(p.velocity.scale(dt));
            p.age += dt;

            // Update visual properties
            p.update_visuals();

            // Kill if expired
            if p.age > p.lifetime {
                p.alive = false;
            }
        }
    }

    /// Deterministic pseudo-random in [0, 1).
    fn next_random(&mut self) -> f64 {
        self.rng_counter = self.rng_counter.wrapping_add(1);
        let state = self.rng_counter.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = (state >> 33) as u32;
        (bits as f64) / (u32::MAX as f64)
    }

    /// Lerp between two values using a random factor.
    fn rand_range(&mut self, lo: f64, hi: f64) -> f64 {
        let t = self.next_random();
        lo + (hi - lo) * t
    }

    fn spawn_particle(&mut self) {
        // Find a dead slot (pool recycling)
        let start = self.recycle_cursor;
        let len = self.particles.len();
        let mut found = None;
        for i in 0..len {
            let idx = (start + i) % len;
            if !self.particles[idx].alive {
                found = Some(idx);
                self.recycle_cursor = (idx + 1) % len;
                break;
            }
        }
        let idx = match found {
            Some(i) => i,
            None => return, // Pool full
        };

        let speed = self.rand_range(self.config.speed_range.0, self.config.speed_range.1);
        let lifetime = self.rand_range(self.config.lifetime_range.0, self.config.lifetime_range.1);
        let start_size = self.rand_range(self.config.start_size_range.0, self.config.start_size_range.1);
        let end_size = self.rand_range(self.config.end_size_range.0, self.config.end_size_range.1);

        // Compute direction with spread
        let base_angle = self.config.direction.y.atan2(self.config.direction.x);
        let offset = self.rand_range(-self.config.spread * 0.5, self.config.spread * 0.5);
        let angle = base_angle + offset;
        let vx = angle.cos() * speed;
        let vy = angle.sin() * speed;

        let p = &mut self.particles[idx];
        p.position = self.config.position;
        p.velocity = Vec2::new(vx, vy);
        p.acceleration = Vec2::zero();
        p.age = 0.0;
        p.lifetime = lifetime;
        p.size = start_size;
        p.start_size = start_size;
        p.end_size = end_size;
        p.start_color = self.config.start_color;
        p.end_color = self.config.end_color;
        p.color = self.config.start_color;
        p.mass = self.config.mass;
        p.alive = true;

        self.total_spawned += 1;
    }

    /// Kill all particles.
    pub fn clear(&mut self) {
        for p in &mut self.particles {
            p.alive = false;
        }
    }

    /// Set the emitter position.
    pub fn set_position(&mut self, pos: Vec2) {
        self.config.position = pos;
    }

    /// Set the emission rate.
    pub fn set_rate(&mut self, rate: f64) {
        self.config.rate = rate;
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_system() -> ParticleSystem {
        let cfg = EmitterConfig {
            rate: 100.0,
            max_particles: 200,
            lifetime_range: (1.0, 1.0),
            speed_range: (1.0, 1.0),
            start_size_range: (1.0, 1.0),
            end_size_range: (0.0, 0.0),
            ..Default::default()
        };
        ParticleSystem::new(cfg)
    }

    #[test]
    fn initial_state() {
        let sys = default_system();
        assert_eq!(sys.alive_count(), 0);
        assert_eq!(sys.total_spawned, 0);
    }

    #[test]
    fn emit_particles() {
        let mut sys = default_system();
        sys.update(1.0); // 100 particles/sec * 1s = 100
        assert!(sys.alive_count() > 0);
        assert!(sys.alive_count() <= 100);
    }

    #[test]
    fn burst_mode() {
        let mut sys = default_system();
        sys.emitting = false;
        sys.burst(50);
        assert_eq!(sys.alive_count(), 50);
    }

    #[test]
    fn particles_die_after_lifetime() {
        let mut sys = default_system();
        sys.burst(10);
        assert_eq!(sys.alive_count(), 10);
        // All have lifetime 1.0, so after 1.5s they should be dead
        sys.update(1.5);
        assert_eq!(sys.alive_count(), 0);
    }

    #[test]
    fn particle_pool_recycling() {
        let cfg = EmitterConfig {
            rate: 100.0,
            max_particles: 10,
            lifetime_range: (0.5, 0.5),
            speed_range: (1.0, 1.0),
            ..Default::default()
        };
        let mut sys = ParticleSystem::new(cfg);
        sys.burst(10); // Fill pool
        assert_eq!(sys.alive_count(), 10);
        // Pool is full, no more can spawn
        sys.burst(5);
        assert_eq!(sys.alive_count(), 10);
        // Let them die
        sys.update(1.0);
        assert_eq!(sys.alive_count(), 0);
        // Now we can spawn again (recycled)
        sys.burst(5);
        assert_eq!(sys.alive_count(), 5);
    }

    #[test]
    fn gravity_force() {
        let mut sys = default_system();
        sys.add_force(Force::Gravity(Vec2::new(0.0, -10.0)));
        sys.burst(1);
        let initial_y = sys.particles().iter().find(|p| p.alive).unwrap().position.y;
        sys.update(1.0);
        let alive: Vec<_> = sys.alive_particles().collect();
        if !alive.is_empty() {
            assert!(alive[0].position.y < initial_y);
        }
    }

    #[test]
    fn wind_force() {
        let mut sys = default_system();
        sys.config.direction = Vec2::new(0.0, 1.0);
        sys.add_force(Force::Wind(Vec2::new(10.0, 0.0)));
        sys.burst(1);
        sys.update(0.5);
        let alive: Vec<_> = sys.alive_particles().collect();
        assert!(!alive.is_empty());
        // Wind should push particle in +x
        assert!(alive[0].position.x > 0.0);
    }

    #[test]
    fn drag_force() {
        let mut sys = default_system();
        sys.add_force(Force::Drag(1.0));
        sys.burst(1);
        // Get initial speed
        let initial_speed = sys.particles().iter()
            .find(|p| p.alive).unwrap().velocity.length();
        sys.update(0.1);
        let alive: Vec<_> = sys.alive_particles().collect();
        if !alive.is_empty() {
            // Drag should slow it down
            assert!(alive[0].velocity.length() < initial_speed);
        }
    }

    #[test]
    fn size_interpolation() {
        let cfg = EmitterConfig {
            rate: 0.0,
            max_particles: 10,
            lifetime_range: (2.0, 2.0),
            start_size_range: (10.0, 10.0),
            end_size_range: (0.0, 0.0),
            speed_range: (0.0, 0.0),
            ..Default::default()
        };
        let mut sys = ParticleSystem::new(cfg);
        sys.burst(1);
        let initial_size = sys.alive_particles().next().unwrap().size;
        assert!((initial_size - 10.0).abs() < 0.01);
        sys.update(1.0); // Half lifetime
        let mid_size = sys.alive_particles().next().unwrap().size;
        assert!(mid_size < initial_size);
        assert!(mid_size > 0.0);
    }

    #[test]
    fn color_interpolation() {
        let cfg = EmitterConfig {
            rate: 0.0,
            max_particles: 10,
            lifetime_range: (2.0, 2.0),
            start_color: Color::red(),
            end_color: Color::transparent(),
            speed_range: (0.0, 0.0),
            ..Default::default()
        };
        let mut sys = ParticleSystem::new(cfg);
        sys.burst(1);
        let c0 = sys.alive_particles().next().unwrap().color;
        assert!((c0.r - 1.0).abs() < 0.01);
        sys.update(1.0);
        let c1 = sys.alive_particles().next().unwrap().color;
        assert!(c1.r < c0.r);
        assert!(c1.a < c0.a);
    }

    #[test]
    fn clear_kills_all() {
        let mut sys = default_system();
        sys.burst(50);
        assert_eq!(sys.alive_count(), 50);
        sys.clear();
        assert_eq!(sys.alive_count(), 0);
    }

    #[test]
    fn set_position() {
        let mut sys = default_system();
        sys.set_position(Vec2::new(100.0, 200.0));
        sys.burst(1);
        let p = sys.alive_particles().next().unwrap();
        assert!((p.position.x - 100.0).abs() < 0.01);
        assert!((p.position.y - 200.0).abs() < 0.01);
    }

    #[test]
    fn radial_force() {
        let mut sys = default_system();
        sys.config.position = Vec2::new(10.0, 0.0);
        sys.config.speed_range = (0.0, 0.0);
        sys.add_force(Force::Radial { center: Vec2::zero(), strength: 100.0 });
        sys.burst(1);
        sys.update(0.1);
        let p = sys.alive_particles().next().unwrap();
        // Should be pulled toward (0,0)
        assert!(p.position.x < 10.0);
    }

    #[test]
    fn turbulence_force() {
        let mut sys = default_system();
        sys.config.speed_range = (0.0, 0.0);
        sys.add_force(Force::Turbulence { strength: 5.0, scale: 1.0 });
        sys.burst(1);
        sys.update(0.5);
        let p = sys.alive_particles().next().unwrap();
        // Should have moved from origin due to turbulence
        let dist = p.position.length();
        assert!(dist > 0.0);
    }

    #[test]
    fn emitting_toggle() {
        let mut sys = default_system();
        sys.emitting = false;
        sys.update(1.0);
        assert_eq!(sys.alive_count(), 0);
        sys.emitting = true;
        sys.update(1.0);
        assert!(sys.alive_count() > 0);
    }

    #[test]
    fn particle_age_ratio() {
        let mut p = Particle::new();
        p.lifetime = 4.0;
        p.age = 2.0;
        assert!((p.age_ratio() - 0.5).abs() < 0.01);
    }

    #[test]
    fn total_spawned_tracking() {
        let mut sys = default_system();
        sys.burst(10);
        assert_eq!(sys.total_spawned, 10);
        sys.burst(5);
        assert_eq!(sys.total_spawned, 15);
    }

    #[test]
    fn multiple_forces() {
        let mut sys = default_system();
        sys.add_force(Force::Gravity(Vec2::new(0.0, -10.0)));
        sys.add_force(Force::Wind(Vec2::new(5.0, 0.0)));
        sys.burst(1);
        sys.update(0.5);
        let p = sys.alive_particles().next().unwrap();
        // Should have moved right (wind) and possibly up/down (gravity + initial velocity)
        assert!(p.position.x > 0.0 || p.velocity.x > 0.0);
    }
}
