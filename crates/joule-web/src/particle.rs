//! Particle System — particles, emitters, burst emission, affectors
//! (gravity, drag, color-over-life, size-over-life), and update loop.

use crate::webgl::Vec3;

// ── Particle ──────────────────────────────────────────────────

/// A single particle with position, velocity, life, color, and size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub color: [f64; 4],
    pub size: f64,
    pub lifetime: f64,
    pub age: f64,
}

impl Particle {
    pub fn new(position: Vec3, velocity: Vec3, lifetime: f64) -> Self {
        Self {
            position,
            velocity,
            color: [1.0, 1.0, 1.0, 1.0],
            size: 1.0,
            lifetime,
            age: 0.0,
        }
    }

    /// Normalized age in [0, 1].
    pub fn normalized_age(&self) -> f64 {
        if self.lifetime <= 0.0 { 1.0 } else { (self.age / self.lifetime).min(1.0) }
    }

    /// Is this particle dead?
    pub fn is_dead(&self) -> bool {
        self.age >= self.lifetime
    }
}

// ── Simple RNG ────────────────────────────────────────────────

/// Lightweight xorshift64 PRNG for deterministic particle emission.
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

    /// Random f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() % 1_000_000) as f64 / 1_000_000.0
    }

    /// Random f64 in [min, max).
    fn range(&mut self, min: f64, max: f64) -> f64 {
        min + self.next_f64() * (max - min)
    }
}

// ── Affector ──────────────────────────────────────────────────

/// Affectors modify particles during the update step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Affector {
    /// Constant gravity force.
    Gravity(Vec3),
    /// Drag coefficient — velocity *= (1 - drag * dt).
    Drag(f64),
    /// Lerp color from `start` to `end` over the particle's life.
    ColorOverLife {
        start: [f64; 4],
        end: [f64; 4],
    },
    /// Lerp size from `start` to `end` over the particle's life.
    SizeOverLife {
        start: f64,
        end: f64,
    },
}

impl Affector {
    pub fn apply(&self, particle: &mut Particle, dt: f64) {
        match self {
            Affector::Gravity(g) => {
                particle.velocity = particle.velocity + *g * dt;
            }
            Affector::Drag(drag) => {
                let factor = (1.0 - drag * dt).max(0.0);
                particle.velocity = particle.velocity * factor;
            }
            Affector::ColorOverLife { start, end } => {
                let t = particle.normalized_age();
                for i in 0..4 {
                    particle.color[i] = start[i] + (end[i] - start[i]) * t;
                }
            }
            Affector::SizeOverLife { start, end } => {
                let t = particle.normalized_age();
                particle.size = start + (end - start) * t;
            }
        }
    }
}

// ── EmitterShape ──────────────────────────────────────────────

/// Shape of the emission volume.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmitterShape {
    /// Emit from a single point.
    Point,
    /// Emit from within a sphere of given radius.
    Sphere { radius: f64 },
    /// Emit from within an AABB.
    Box { half_extents: Vec3 },
}

// ── ParticleEmitter ───────────────────────────────────────────

/// Emits and manages particles.
pub struct ParticleEmitter {
    pub position: Vec3,
    pub shape: EmitterShape,
    /// Particles emitted per second.
    pub emission_rate: f64,
    /// Min/max initial speed.
    pub speed_range: (f64, f64),
    /// Min/max particle lifetime.
    pub lifetime_range: (f64, f64),
    /// Initial particle size.
    pub initial_size: f64,
    /// Initial particle color.
    pub initial_color: [f64; 4],
    /// Direction bias — particles are emitted toward this direction (if non-zero).
    pub direction: Vec3,
    /// Spread angle (radians) around the direction. 0 = focused, PI = hemisphere.
    pub spread: f64,
    /// Maximum number of live particles.
    pub max_particles: usize,
    /// Affectors applied each frame.
    pub affectors: Vec<Affector>,

    // Internal state.
    particles: Vec<Particle>,
    emission_accumulator: f64,
    rng: Rng,
}

impl ParticleEmitter {
    pub fn new(position: Vec3, emission_rate: f64) -> Self {
        Self {
            position,
            shape: EmitterShape::Point,
            emission_rate,
            speed_range: (1.0, 3.0),
            lifetime_range: (1.0, 3.0),
            initial_size: 1.0,
            initial_color: [1.0, 1.0, 1.0, 1.0],
            direction: Vec3::up(),
            spread: std::f64::consts::PI,
            max_particles: 1000,
            affectors: Vec::new(),
            particles: Vec::new(),
            emission_accumulator: 0.0,
            rng: Rng::new(42),
        }
    }

    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    pub fn alive_count(&self) -> usize {
        self.particles.len()
    }

    /// Emit `count` particles in a single burst.
    pub fn burst(&mut self, count: usize) {
        for _ in 0..count {
            if self.particles.len() >= self.max_particles {
                break;
            }
            let p = self.spawn_particle();
            self.particles.push(p);
        }
    }

    /// Update all particles by `dt` seconds. Ages, removes dead, emits new.
    pub fn update(&mut self, dt: f64) {
        // Age and integrate existing particles.
        for p in &mut self.particles {
            p.age += dt;
            p.position = p.position + p.velocity * dt;
        }

        // Apply affectors.
        let affectors = self.affectors.clone();
        for p in &mut self.particles {
            for aff in &affectors {
                aff.apply(p, dt);
            }
        }

        // Remove dead particles.
        self.particles.retain(|p| !p.is_dead());

        // Emit new particles.
        self.emission_accumulator += self.emission_rate * dt;
        let to_emit = self.emission_accumulator.floor() as usize;
        self.emission_accumulator -= to_emit as f64;
        for _ in 0..to_emit {
            if self.particles.len() >= self.max_particles {
                break;
            }
            let p = self.spawn_particle();
            self.particles.push(p);
        }
    }

    /// Clear all particles.
    pub fn clear(&mut self) {
        self.particles.clear();
        self.emission_accumulator = 0.0;
    }

    fn spawn_particle(&mut self) -> Particle {
        let spawn_pos = match self.shape {
            EmitterShape::Point => self.position,
            EmitterShape::Sphere { radius } => {
                let theta = self.rng.range(0.0, std::f64::consts::TAU);
                let phi = self.rng.range(0.0, std::f64::consts::PI);
                let r = self.rng.range(0.0, radius);
                self.position
                    + Vec3::new(
                        r * phi.sin() * theta.cos(),
                        r * phi.cos(),
                        r * phi.sin() * theta.sin(),
                    )
            }
            EmitterShape::Box { half_extents } => {
                self.position
                    + Vec3::new(
                        self.rng.range(-half_extents.x, half_extents.x),
                        self.rng.range(-half_extents.y, half_extents.y),
                        self.rng.range(-half_extents.z, half_extents.z),
                    )
            }
        };

        // Generate random direction within spread of the emitter direction.
        let dir = if self.direction.length_squared() < 1e-12 {
            // Random direction.
            let theta = self.rng.range(0.0, std::f64::consts::TAU);
            let z = self.rng.range(-1.0, 1.0);
            let r = (1.0 - z * z).sqrt();
            Vec3::new(r * theta.cos(), r * theta.sin(), z)
        } else {
            // Biased direction with spread.
            let base = self.direction.normalize();
            let theta = self.rng.range(0.0, std::f64::consts::TAU);
            let cos_spread = (1.0 - self.rng.next_f64() * (1.0 - self.spread.cos())).max(-1.0);
            let sin_spread = (1.0 - cos_spread * cos_spread).sqrt();

            // Find a perpendicular basis.
            let up = if base.y.abs() < 0.99 { Vec3::up() } else { Vec3::right() };
            let t = base.cross(&up).normalize();
            let b = base.cross(&t);

            (base * cos_spread + t * (sin_spread * theta.cos()) + b * (sin_spread * theta.sin()))
                .normalize()
        };

        let speed = self.rng.range(self.speed_range.0, self.speed_range.1);
        let lifetime = self.rng.range(self.lifetime_range.0, self.lifetime_range.1);

        let mut p = Particle::new(spawn_pos, dir * speed, lifetime);
        p.size = self.initial_size;
        p.color = self.initial_color;
        p
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    #[test]
    fn particle_starts_alive() {
        let p = Particle::new(Vec3::zero(), Vec3::up(), 2.0);
        assert!(!p.is_dead());
        assert!((p.normalized_age()).abs() < EPS);
    }

    #[test]
    fn particle_dies_after_lifetime() {
        let mut p = Particle::new(Vec3::zero(), Vec3::up(), 1.0);
        p.age = 1.0;
        assert!(p.is_dead());
        assert!((p.normalized_age() - 1.0).abs() < EPS);
    }

    #[test]
    fn emitter_emits_particles() {
        let mut emitter = ParticleEmitter::new(Vec3::zero(), 100.0);
        emitter.update(1.0);
        assert!(emitter.alive_count() > 0);
    }

    #[test]
    fn burst_emission() {
        let mut emitter = ParticleEmitter::new(Vec3::zero(), 0.0);
        emitter.burst(50);
        assert_eq!(emitter.alive_count(), 50);
    }

    #[test]
    fn max_particles_respected() {
        let mut emitter = ParticleEmitter::new(Vec3::zero(), 0.0);
        emitter.max_particles = 10;
        emitter.burst(100);
        assert_eq!(emitter.alive_count(), 10);
    }

    #[test]
    fn dead_particles_removed() {
        let mut emitter = ParticleEmitter::new(Vec3::zero(), 0.0);
        emitter.lifetime_range = (0.1, 0.1);
        emitter.burst(10);
        assert_eq!(emitter.alive_count(), 10);
        emitter.update(0.2); // all should die
        assert_eq!(emitter.alive_count(), 0);
    }

    #[test]
    fn gravity_affector() {
        let mut p = Particle::new(Vec3::zero(), Vec3::zero(), 5.0);
        let grav = Affector::Gravity(Vec3::new(0.0, -9.81, 0.0));
        grav.apply(&mut p, 1.0);
        assert!((p.velocity.y - (-9.81)).abs() < EPS);
    }

    #[test]
    fn drag_affector() {
        let mut p = Particle::new(Vec3::zero(), Vec3::new(10.0, 0.0, 0.0), 5.0);
        let drag = Affector::Drag(0.5);
        drag.apply(&mut p, 1.0);
        assert!((p.velocity.x - 5.0).abs() < EPS);
    }

    #[test]
    fn color_over_life() {
        let mut p = Particle::new(Vec3::zero(), Vec3::zero(), 2.0);
        p.age = 1.0; // 50%
        let aff = Affector::ColorOverLife {
            start: [1.0, 0.0, 0.0, 1.0],
            end: [0.0, 0.0, 1.0, 0.0],
        };
        aff.apply(&mut p, 0.0);
        assert!((p.color[0] - 0.5).abs() < EPS);
        assert!((p.color[2] - 0.5).abs() < EPS);
        assert!((p.color[3] - 0.5).abs() < EPS);
    }

    #[test]
    fn size_over_life() {
        let mut p = Particle::new(Vec3::zero(), Vec3::zero(), 2.0);
        p.age = 2.0; // 100%
        let aff = Affector::SizeOverLife { start: 2.0, end: 0.0 };
        aff.apply(&mut p, 0.0);
        assert!((p.size).abs() < EPS);
    }

    #[test]
    fn emitter_clear() {
        let mut emitter = ParticleEmitter::new(Vec3::zero(), 0.0);
        emitter.burst(20);
        assert_eq!(emitter.alive_count(), 20);
        emitter.clear();
        assert_eq!(emitter.alive_count(), 0);
    }

    #[test]
    fn particles_move_over_time() {
        let mut emitter = ParticleEmitter::new(Vec3::zero(), 0.0);
        emitter.speed_range = (5.0, 5.0);
        emitter.lifetime_range = (10.0, 10.0);
        emitter.burst(1);
        let pos_before = emitter.particles()[0].position;
        emitter.update(1.0);
        let pos_after = emitter.particles()[0].position;
        assert!(pos_before.distance(&pos_after) > 0.1);
    }

    #[test]
    fn sphere_shape_offsets_spawn() {
        let mut emitter = ParticleEmitter::new(Vec3::zero(), 0.0);
        emitter.shape = EmitterShape::Sphere { radius: 5.0 };
        emitter.lifetime_range = (10.0, 10.0);
        emitter.burst(100);
        // At least some particles should be offset from origin.
        let any_offset = emitter.particles().iter().any(|p| p.position.length() > 0.01);
        assert!(any_offset);
    }
}
