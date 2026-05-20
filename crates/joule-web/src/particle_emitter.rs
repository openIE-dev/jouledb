// particle_emitter.rs — Particle emitter shapes and configurations
// Part of joule-web: Particles & VFX cluster

/// 3D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const UP: Self = Self { x: 0.0, y: 1.0, z: 0.0 };

    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-10 {
            Self::ZERO
        } else {
            Self { x: self.x / len, y: self.y / len, z: self.z / len }
        }
    }

    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            z: self.z + (other.z - self.z) * t,
        }
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z }
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, s: f32) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z }
    }
}

/// RGBA color.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const WHITE: Self = Self { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const RED: Self = Self { r: 1.0, g: 0.0, b: 0.0, a: 1.0 };

    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }
}

/// Range with min/max for randomizable properties.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangeF32 {
    pub min: f32,
    pub max: f32,
}

impl RangeF32 {
    pub fn constant(v: f32) -> Self {
        Self { min: v, max: v }
    }

    pub fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }

    /// Sample at a normalized t in [0,1].
    pub fn sample(&self, t: f32) -> f32 {
        self.min + (self.max - self.min) * t.clamp(0.0, 1.0)
    }

    pub fn midpoint(&self) -> f32 {
        (self.min + self.max) * 0.5
    }
}

/// Emitter shape controlling where particles spawn and their initial direction.
#[derive(Debug, Clone, PartialEq)]
pub enum EmitterShape {
    /// Emit from a single point.
    Point,
    /// Emit from a sphere surface or volume.
    Sphere {
        radius: f32,
        /// If true, emit from volume; if false, from surface only.
        volume: bool,
    },
    /// Emit from a cone.
    Cone {
        /// Half-angle in radians.
        angle: f32,
        /// Radius at the base.
        radius: f32,
        /// Emit from volume vs surface.
        volume: bool,
    },
    /// Emit from an axis-aligned box.
    Box {
        half_extents: Vec3,
    },
    /// Emit along a line segment (edge).
    Edge {
        start: Vec3,
        end: Vec3,
    },
    /// Emit from triangles of a mesh surface.
    MeshSurface {
        /// Triangle vertices (each group of 3 is a triangle).
        vertices: Vec<Vec3>,
    },
}

/// Space for emission direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmissionSpace {
    Local,
    World,
}

/// Descriptor for a sub-emitter that triggers on particle events.
#[derive(Debug, Clone, PartialEq)]
pub struct SubEmitterDesc {
    /// The config index to use when sub-emitting.
    pub config_id: u32,
    /// How many particles to burst on trigger.
    pub burst_count: u32,
    /// Trigger event.
    pub trigger: SubEmitterTrigger,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SubEmitterTrigger {
    OnDeath,
    OnBirth,
    OnCollision,
}

/// Emitted particle properties for one spawned particle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmittedParticle {
    pub position: Vec3,
    pub direction: Vec3,
    pub speed: f32,
    pub size: f32,
    pub lifetime: f32,
    pub color: Color,
    pub rotation: f32,
}

/// Full emitter configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct EmitterConfig {
    pub shape: EmitterShape,
    pub emit_rate: f32,
    pub space: EmissionSpace,
    pub speed: RangeF32,
    pub size: RangeF32,
    pub lifetime: RangeF32,
    pub rotation: RangeF32,
    pub color_start: Color,
    pub color_end: Color,
    pub inherit_velocity_factor: f32,
    pub sub_emitters: Vec<SubEmitterDesc>,
    pub max_particles: u32,
}

impl Default for EmitterConfig {
    fn default() -> Self {
        Self {
            shape: EmitterShape::Point,
            emit_rate: 10.0,
            space: EmissionSpace::Local,
            speed: RangeF32::new(1.0, 3.0),
            size: RangeF32::new(0.05, 0.15),
            lifetime: RangeF32::new(1.0, 3.0),
            rotation: RangeF32::new(0.0, std::f32::consts::TAU),
            color_start: Color::WHITE,
            color_end: Color::new(1.0, 1.0, 1.0, 0.0),
            inherit_velocity_factor: 0.0,
            sub_emitters: Vec::new(),
            max_particles: 1000,
        }
    }
}

/// Runtime state of a particle emitter.
pub struct ParticleEmitter {
    config: EmitterConfig,
    position: Vec3,
    velocity: Vec3,
    emission_accumulator: f32,
    total_emitted: u64,
    active: bool,
    burst_queue: Vec<u32>,
}

impl ParticleEmitter {
    pub fn new(config: EmitterConfig) -> Self {
        Self {
            config,
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            emission_accumulator: 0.0,
            total_emitted: 0,
            active: true,
            burst_queue: Vec::new(),
        }
    }

    pub fn config(&self) -> &EmitterConfig {
        &self.config
    }

    pub fn set_position(&mut self, pos: Vec3) {
        self.position = pos;
    }

    pub fn position(&self) -> Vec3 {
        self.position
    }

    pub fn set_velocity(&mut self, vel: Vec3) {
        self.velocity = vel;
    }

    pub fn velocity(&self) -> Vec3 {
        self.velocity
    }

    pub fn total_emitted(&self) -> u64 {
        self.total_emitted
    }

    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Queue a burst for the next update.
    pub fn queue_burst(&mut self, count: u32) {
        self.burst_queue.push(count);
    }

    /// Compute a spawn position and direction from the emitter shape using a
    /// deterministic parameter `t` in [0,1] (standing in for randomness).
    pub fn sample_shape(&self, t: f32) -> (Vec3, Vec3) {
        let t = t.clamp(0.0, 1.0);
        match &self.config.shape {
            EmitterShape::Point => (self.position, Vec3::UP),
            EmitterShape::Sphere { radius, volume } => {
                // Deterministic spherical coordinates from t
                let theta = t * std::f32::consts::TAU;
                let phi = t * std::f32::consts::PI;
                let st = phi.sin();
                let dir = Vec3::new(st * theta.cos(), phi.cos(), st * theta.sin());
                let r = if *volume { *radius * t } else { *radius };
                (self.position + dir * r, dir)
            }
            EmitterShape::Cone { angle, radius, volume } => {
                let theta = t * std::f32::consts::TAU;
                let cone_r = if *volume { *radius * t } else { *radius };
                let spread = angle.sin() * cone_r;
                let local_pos = Vec3::new(spread * theta.cos(), 0.0, spread * theta.sin());
                let dir = Vec3::new(
                    angle.sin() * theta.cos(),
                    angle.cos(),
                    angle.sin() * theta.sin(),
                ).normalized();
                (self.position + local_pos, dir)
            }
            EmitterShape::Box { half_extents } => {
                // Map t to [-1, 1] range for each axis using simple phase offsets
                let tx = (t * 6.283).sin();
                let ty = (t * 9.871).sin();
                let tz = (t * 14.17).sin();
                let pos = Vec3::new(
                    half_extents.x * tx,
                    half_extents.y * ty,
                    half_extents.z * tz,
                );
                (self.position + pos, Vec3::UP)
            }
            EmitterShape::Edge { start, end } => {
                let pos = start.lerp(*end, t);
                let edge_dir = (*end - *start).normalized();
                let dir = edge_dir.cross(Vec3::UP).normalized();
                let dir = if dir.length() < 0.5 { Vec3::UP } else { dir };
                (self.position + pos, dir)
            }
            EmitterShape::MeshSurface { vertices } => {
                if vertices.len() < 3 {
                    return (self.position, Vec3::UP);
                }
                let tri_count = vertices.len() / 3;
                let tri_idx = ((t * tri_count as f32).floor() as usize).min(tri_count - 1);
                let base = tri_idx * 3;
                let v0 = vertices[base];
                let v1 = vertices[base + 1];
                let v2 = vertices[base + 2];
                // Barycentric from t
                let u = t.fract();
                let v = (t * 3.7).fract();
                let (u, v) = if u + v > 1.0 { (1.0 - u, 1.0 - v) } else { (u, v) };
                let w = 1.0 - u - v;
                let pos = v0 * w + v1 * u + v2 * v;
                let normal = (v1 - v0).cross(v2 - v0).normalized();
                let normal = if normal.length() < 0.5 { Vec3::UP } else { normal };
                (self.position + pos, normal)
            }
        }
    }

    /// Emit a single particle with the given parameter t in [0,1].
    pub fn emit_one(&mut self, t: f32) -> EmittedParticle {
        let (pos, dir) = self.sample_shape(t);
        let speed = self.config.speed.sample(t);
        let size = self.config.size.sample(t);
        let lifetime = self.config.lifetime.sample(t);
        let rotation = self.config.rotation.sample(t);
        let color = self.config.color_start.lerp(self.config.color_end, t);

        let mut final_dir = dir * speed;
        if self.config.inherit_velocity_factor.abs() > 1e-9 {
            final_dir = final_dir + self.velocity * self.config.inherit_velocity_factor;
        }

        self.total_emitted += 1;
        EmittedParticle {
            position: pos,
            direction: final_dir.normalized(),
            speed: final_dir.length(),
            size,
            lifetime,
            color,
            rotation,
        }
    }

    /// Advance the emitter by dt seconds, returning particles to spawn.
    pub fn update(&mut self, dt: f32) -> Vec<EmittedParticle> {
        if !self.active {
            return Vec::new();
        }

        let mut result = Vec::new();

        // Process burst queue
        let bursts: Vec<u32> = self.burst_queue.drain(..).collect();
        for count in bursts {
            for i in 0..count {
                let t = if count <= 1 { 0.5 } else { i as f32 / (count - 1) as f32 };
                result.push(self.emit_one(t));
            }
        }

        // Continuous emission
        self.emission_accumulator += self.config.emit_rate * dt;
        let to_emit = self.emission_accumulator.floor() as u32;
        self.emission_accumulator -= to_emit as f32;

        for i in 0..to_emit {
            let t = if to_emit <= 1 { 0.5 } else { i as f32 / (to_emit - 1) as f32 };
            result.push(self.emit_one(t));
        }

        result
    }

    /// Get sub-emitter descriptors for a given trigger.
    pub fn sub_emitters_for(&self, trigger: SubEmitterTrigger) -> Vec<&SubEmitterDesc> {
        self.config
            .sub_emitters
            .iter()
            .filter(|se| se.trigger == trigger)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_f32_constant() {
        let r = RangeF32::constant(5.0);
        assert!((r.sample(0.0) - 5.0).abs() < 1e-6);
        assert!((r.sample(1.0) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_range_f32_sample() {
        let r = RangeF32::new(10.0, 20.0);
        assert!((r.sample(0.0) - 10.0).abs() < 1e-6);
        assert!((r.sample(0.5) - 15.0).abs() < 1e-6);
        assert!((r.sample(1.0) - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_range_f32_midpoint() {
        let r = RangeF32::new(2.0, 8.0);
        assert!((r.midpoint() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_default_config() {
        let cfg = EmitterConfig::default();
        assert!((cfg.emit_rate - 10.0).abs() < 1e-6);
        assert_eq!(cfg.space, EmissionSpace::Local);
        assert!(cfg.sub_emitters.is_empty());
    }

    #[test]
    fn test_emitter_new() {
        let emitter = ParticleEmitter::new(EmitterConfig::default());
        assert!(emitter.is_active());
        assert_eq!(emitter.total_emitted(), 0);
    }

    #[test]
    fn test_point_shape() {
        let emitter = ParticleEmitter::new(EmitterConfig::default());
        let (pos, dir) = emitter.sample_shape(0.5);
        assert!((pos.x).abs() < 1e-6);
        assert!((dir.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sphere_surface_shape() {
        let cfg = EmitterConfig {
            shape: EmitterShape::Sphere { radius: 2.0, volume: false },
            ..Default::default()
        };
        let emitter = ParticleEmitter::new(cfg);
        let (pos, _dir) = emitter.sample_shape(0.5);
        let dist = pos.length();
        assert!((dist - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_sphere_volume_shape() {
        let cfg = EmitterConfig {
            shape: EmitterShape::Sphere { radius: 3.0, volume: true },
            ..Default::default()
        };
        let emitter = ParticleEmitter::new(cfg);
        let (pos, _) = emitter.sample_shape(0.25);
        let dist = pos.length();
        assert!(dist <= 3.0 + 0.1);
    }

    #[test]
    fn test_box_shape() {
        let cfg = EmitterConfig {
            shape: EmitterShape::Box {
                half_extents: Vec3::new(1.0, 1.0, 1.0),
            },
            ..Default::default()
        };
        let emitter = ParticleEmitter::new(cfg);
        let (pos, _) = emitter.sample_shape(0.3);
        assert!(pos.x.abs() <= 1.0 + 1e-6);
        assert!(pos.y.abs() <= 1.0 + 1e-6);
        assert!(pos.z.abs() <= 1.0 + 1e-6);
    }

    #[test]
    fn test_edge_shape() {
        let cfg = EmitterConfig {
            shape: EmitterShape::Edge {
                start: Vec3::new(-5.0, 0.0, 0.0),
                end: Vec3::new(5.0, 0.0, 0.0),
            },
            ..Default::default()
        };
        let emitter = ParticleEmitter::new(cfg);
        let (pos, _) = emitter.sample_shape(0.5);
        // At t=0.5, position should be midpoint of edge = (0,0,0) + emitter_pos
        assert!((pos.y).abs() < 1e-5);
    }

    #[test]
    fn test_cone_shape() {
        let cfg = EmitterConfig {
            shape: EmitterShape::Cone {
                angle: std::f32::consts::FRAC_PI_4,
                radius: 1.0,
                volume: false,
            },
            ..Default::default()
        };
        let emitter = ParticleEmitter::new(cfg);
        let (_pos, dir) = emitter.sample_shape(0.5);
        assert!(dir.length() > 0.9);
    }

    #[test]
    fn test_mesh_surface_shape() {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let cfg = EmitterConfig {
            shape: EmitterShape::MeshSurface { vertices: verts },
            ..Default::default()
        };
        let emitter = ParticleEmitter::new(cfg);
        let (pos, dir) = emitter.sample_shape(0.3);
        // Should be somewhere on or near the triangle
        assert!(pos.x >= -0.1 && pos.y >= -0.1);
        assert!(dir.length() > 0.5);
    }

    #[test]
    fn test_mesh_surface_empty() {
        let cfg = EmitterConfig {
            shape: EmitterShape::MeshSurface { vertices: vec![] },
            ..Default::default()
        };
        let emitter = ParticleEmitter::new(cfg);
        let (pos, dir) = emitter.sample_shape(0.5);
        assert!((pos.x).abs() < 1e-6);
        assert!((dir.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_emit_one() {
        let mut emitter = ParticleEmitter::new(EmitterConfig::default());
        let p = emitter.emit_one(0.5);
        assert!(p.speed > 0.0);
        assert!(p.size > 0.0);
        assert!(p.lifetime > 0.0);
        assert_eq!(emitter.total_emitted(), 1);
    }

    #[test]
    fn test_continuous_emission() {
        let cfg = EmitterConfig {
            emit_rate: 20.0,
            ..Default::default()
        };
        let mut emitter = ParticleEmitter::new(cfg);
        // 20/sec * 0.5s = 10 particles
        let particles = emitter.update(0.5);
        assert_eq!(particles.len(), 10);
    }

    #[test]
    fn test_burst_emission() {
        let mut emitter = ParticleEmitter::new(EmitterConfig {
            emit_rate: 0.0,
            ..Default::default()
        });
        emitter.queue_burst(15);
        let particles = emitter.update(0.1);
        assert_eq!(particles.len(), 15);
    }

    #[test]
    fn test_inactive_emitter() {
        let mut emitter = ParticleEmitter::new(EmitterConfig::default());
        emitter.set_active(false);
        let particles = emitter.update(1.0);
        assert!(particles.is_empty());
    }

    #[test]
    fn test_inherit_velocity() {
        let cfg = EmitterConfig {
            inherit_velocity_factor: 1.0,
            ..Default::default()
        };
        let mut emitter = ParticleEmitter::new(cfg);
        emitter.set_velocity(Vec3::new(10.0, 0.0, 0.0));
        let p = emitter.emit_one(0.5);
        // Speed should be influenced by inherited velocity
        assert!(p.speed > 1.0);
    }

    #[test]
    fn test_sub_emitter_query() {
        let cfg = EmitterConfig {
            sub_emitters: vec![
                SubEmitterDesc {
                    config_id: 1,
                    burst_count: 5,
                    trigger: SubEmitterTrigger::OnDeath,
                },
                SubEmitterDesc {
                    config_id: 2,
                    burst_count: 3,
                    trigger: SubEmitterTrigger::OnBirth,
                },
            ],
            ..Default::default()
        };
        let emitter = ParticleEmitter::new(cfg);
        let death_subs = emitter.sub_emitters_for(SubEmitterTrigger::OnDeath);
        assert_eq!(death_subs.len(), 1);
        assert_eq!(death_subs[0].config_id, 1);
        let birth_subs = emitter.sub_emitters_for(SubEmitterTrigger::OnBirth);
        assert_eq!(birth_subs.len(), 1);
        let collision_subs = emitter.sub_emitters_for(SubEmitterTrigger::OnCollision);
        assert!(collision_subs.is_empty());
    }

    #[test]
    fn test_color_lerp() {
        let c = Color::WHITE.lerp(Color::RED, 0.5);
        assert!((c.r - 1.0).abs() < 1e-6);
        assert!((c.g - 0.5).abs() < 1e-6);
        assert!((c.b - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_set_position() {
        let mut emitter = ParticleEmitter::new(EmitterConfig::default());
        emitter.set_position(Vec3::new(3.0, 4.0, 5.0));
        let pos = emitter.position();
        assert!((pos.x - 3.0).abs() < 1e-6);
        assert!((pos.y - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_total_emitted_tracks() {
        let mut emitter = ParticleEmitter::new(EmitterConfig {
            emit_rate: 100.0,
            ..Default::default()
        });
        emitter.update(0.1);
        assert_eq!(emitter.total_emitted(), 10);
        emitter.update(0.1);
        assert_eq!(emitter.total_emitted(), 20);
    }

    #[test]
    fn test_vec3_cross() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        let z = x.cross(y);
        assert!((z.x).abs() < 1e-6);
        assert!((z.y).abs() < 1e-6);
        assert!((z.z - 1.0).abs() < 1e-6);
    }
}
