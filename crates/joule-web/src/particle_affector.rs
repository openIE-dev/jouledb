// particle_affector.rs — Particle affectors/modifiers over lifetime
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

    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-10 { Self::ZERO } else { Self { x: self.x / len, y: self.y / len, z: self.z / len } }
    }

    pub fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(self, o: Self) -> Self {
        Self {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, r: Self) -> Self { Self { x: self.x + r.x, y: self.y + r.y, z: self.z + r.z } }
}
impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, r: Self) -> Self { Self { x: self.x - r.x, y: self.y - r.y, z: self.z - r.z } }
}
impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, s: f32) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s } }
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

/// Color gradient: sorted stops with interpolation.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorGradient {
    /// Sorted by position [0..1].
    pub stops: Vec<(f32, Color)>,
}

impl ColorGradient {
    pub fn new(stops: Vec<(f32, Color)>) -> Self {
        let mut s = stops;
        s.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Self { stops: s }
    }

    pub fn sample(&self, t: f32) -> Color {
        if self.stops.is_empty() {
            return Color::WHITE;
        }
        if self.stops.len() == 1 || t <= self.stops[0].0 {
            return self.stops[0].1;
        }
        let last = self.stops.len() - 1;
        if t >= self.stops[last].0 {
            return self.stops[last].1;
        }
        for i in 0..last {
            let (t0, c0) = self.stops[i];
            let (t1, c1) = self.stops[i + 1];
            if t >= t0 && t <= t1 {
                let frac = if (t1 - t0).abs() < 1e-9 { 0.0 } else { (t - t0) / (t1 - t0) };
                return c0.lerp(c1, frac);
            }
        }
        self.stops[last].1
    }
}

/// Float curve: piecewise linear from sorted keyframes.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatCurve {
    pub keys: Vec<(f32, f32)>,
}

impl FloatCurve {
    pub fn new(keys: Vec<(f32, f32)>) -> Self {
        let mut k = keys;
        k.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Self { keys: k }
    }

    pub fn constant(v: f32) -> Self {
        Self { keys: vec![(0.0, v), (1.0, v)] }
    }

    pub fn sample(&self, t: f32) -> f32 {
        if self.keys.is_empty() {
            return 0.0;
        }
        if self.keys.len() == 1 || t <= self.keys[0].0 {
            return self.keys[0].1;
        }
        let last = self.keys.len() - 1;
        if t >= self.keys[last].0 {
            return self.keys[last].1;
        }
        for i in 0..last {
            let (t0, v0) = self.keys[i];
            let (t1, v1) = self.keys[i + 1];
            if t >= t0 && t <= t1 {
                let frac = if (t1 - t0).abs() < 1e-9 { 0.0 } else { (t - t0) / (t1 - t0) };
                return v0 + (v1 - v0) * frac;
            }
        }
        self.keys[last].1
    }
}

/// Mutable particle data an affector can modify.
#[derive(Debug, Clone, PartialEq)]
pub struct ParticleData {
    pub position: Vec3,
    pub velocity: Vec3,
    pub size: f32,
    pub color: Color,
    pub rotation: f32,
    pub age: f32,
    pub lifetime: f32,
}

impl ParticleData {
    /// Normalized age in [0, 1].
    pub fn normalized_age(&self) -> f32 {
        if self.lifetime <= 0.0 { 1.0 } else { (self.age / self.lifetime).clamp(0.0, 1.0) }
    }
}

/// Value noise using hash-based interpolation.
fn noise_1d(x: f32) -> f32 {
    let xi = x.floor() as i32;
    let xf = x - x.floor();
    let u = xf * xf * (3.0 - 2.0 * xf);
    let v0 = hash_to_float(xi);
    let v1 = hash_to_float(xi.wrapping_add(1));
    v0 * (1.0 - u) + v1 * u
}

fn hash_i32(n: i32) -> i32 {
    let n = n.wrapping_mul(1597334677);
    let n = n ^ (n >> 16);
    n.wrapping_mul(1103515245).wrapping_add(12345)
}

/// Map integer to [-1, 1] deterministic float.
fn hash_to_float(n: i32) -> f32 {
    let h = hash_i32(n);
    (h as f32) / (i32::MAX as f32)
}

fn noise_3d(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(
        noise_1d(x + 0.0),
        noise_1d(y + 100.0),
        noise_1d(z + 200.0),
    )
}

/// Affector type.
#[derive(Debug, Clone, PartialEq)]
pub enum AffectorType {
    /// Constant gravitational force.
    Gravity { force: Vec3 },
    /// Directional wind with turbulence.
    Wind { direction: Vec3, strength: f32, turbulence: f32 },
    /// Velocity damping.
    Drag { coefficient: f32 },
    /// Color over normalized lifetime.
    ColorOverLife { gradient: ColorGradient },
    /// Size over normalized lifetime.
    SizeOverLife { curve: FloatCurve },
    /// Rotation over normalized lifetime (radians/sec).
    RotationOverLife { curve: FloatCurve },
    /// Point attractor.
    Attractor { position: Vec3, strength: f32, radius: f32 },
    /// Vortex around an axis.
    Vortex { axis: Vec3, center: Vec3, strength: f32 },
    /// Noise-based displacement.
    Noise { frequency: f32, amplitude: f32 },
    /// Clamp velocity magnitude.
    VelocityLimit { max_speed: f32 },
}

/// A particle affector with a priority for ordering.
#[derive(Debug, Clone, PartialEq)]
pub struct ParticleAffector {
    pub affector_type: AffectorType,
    pub priority: i32,
    pub enabled: bool,
}

impl ParticleAffector {
    pub fn new(affector_type: AffectorType, priority: i32) -> Self {
        Self { affector_type, priority, enabled: true }
    }

    /// Apply this affector to a particle for a timestep dt.
    pub fn apply(&self, particle: &mut ParticleData, dt: f32) {
        if !self.enabled {
            return;
        }
        match &self.affector_type {
            AffectorType::Gravity { force } => {
                particle.velocity = particle.velocity + *force * dt;
            }
            AffectorType::Wind { direction, strength, turbulence } => {
                let wind_force = *direction * *strength;
                let turb = if *turbulence > 0.0 {
                    let n = noise_3d(
                        particle.position.x * 0.1,
                        particle.position.y * 0.1,
                        particle.position.z * 0.1,
                    );
                    n * *turbulence
                } else {
                    Vec3::ZERO
                };
                particle.velocity = particle.velocity + (wind_force + turb) * dt;
            }
            AffectorType::Drag { coefficient } => {
                let factor = (1.0 - *coefficient * dt).max(0.0);
                particle.velocity = particle.velocity * factor;
            }
            AffectorType::ColorOverLife { gradient } => {
                particle.color = gradient.sample(particle.normalized_age());
            }
            AffectorType::SizeOverLife { curve } => {
                particle.size = curve.sample(particle.normalized_age());
            }
            AffectorType::RotationOverLife { curve } => {
                particle.rotation += curve.sample(particle.normalized_age()) * dt;
            }
            AffectorType::Attractor { position, strength, radius } => {
                let to_attractor = *position - particle.position;
                let dist = to_attractor.length();
                if dist > 1e-6 && dist < *radius {
                    let dir = to_attractor.normalized();
                    let falloff = 1.0 - (dist / *radius);
                    particle.velocity = particle.velocity + dir * (*strength * falloff * dt);
                }
            }
            AffectorType::Vortex { axis, center, strength } => {
                let to_particle = particle.position - *center;
                let axis_n = axis.normalized();
                // Project onto plane perpendicular to axis
                let along_axis = axis_n * to_particle.dot(axis_n);
                let radial = to_particle - along_axis;
                let dist = radial.length();
                if dist > 1e-6 {
                    let tangent = axis_n.cross(radial.normalized());
                    particle.velocity = particle.velocity + tangent * (*strength * dt / dist.max(0.1));
                }
            }
            AffectorType::Noise { frequency, amplitude } => {
                let n = noise_3d(
                    particle.position.x * *frequency + particle.age,
                    particle.position.y * *frequency + particle.age,
                    particle.position.z * *frequency + particle.age,
                );
                particle.position = particle.position + n * (*amplitude * dt);
            }
            AffectorType::VelocityLimit { max_speed } => {
                let speed = particle.velocity.length();
                if speed > *max_speed && speed > 1e-9 {
                    particle.velocity = particle.velocity.normalized() * *max_speed;
                }
            }
        }
    }
}

/// System to compose and apply multiple affectors in priority order.
pub struct AffectorPipeline {
    affectors: Vec<ParticleAffector>,
    sorted: bool,
}

impl AffectorPipeline {
    pub fn new() -> Self {
        Self { affectors: Vec::new(), sorted: true }
    }

    pub fn add(&mut self, affector: ParticleAffector) {
        self.affectors.push(affector);
        self.sorted = false;
    }

    pub fn remove_at(&mut self, index: usize) -> Option<ParticleAffector> {
        if index < self.affectors.len() {
            Some(self.affectors.remove(index))
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.affectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.affectors.is_empty()
    }

    fn ensure_sorted(&mut self) {
        if !self.sorted {
            self.affectors.sort_by_key(|a| a.priority);
            self.sorted = true;
        }
    }

    /// Apply all affectors to a single particle.
    pub fn apply_all(&mut self, particle: &mut ParticleData, dt: f32) {
        self.ensure_sorted();
        // Clone to avoid borrow conflict
        let affectors: Vec<ParticleAffector> = self.affectors.clone();
        for aff in &affectors {
            aff.apply(particle, dt);
        }
    }

    /// Apply all affectors to a batch of particles.
    pub fn apply_batch(&mut self, particles: &mut [ParticleData], dt: f32) {
        self.ensure_sorted();
        let affectors: Vec<ParticleAffector> = self.affectors.clone();
        for p in particles.iter_mut() {
            for aff in &affectors {
                aff.apply(p, dt);
            }
        }
    }

    pub fn affectors(&self) -> &[ParticleAffector] {
        &self.affectors
    }

    pub fn clear(&mut self) {
        self.affectors.clear();
        self.sorted = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_particle() -> ParticleData {
        ParticleData {
            position: Vec3::ZERO,
            velocity: Vec3::new(1.0, 0.0, 0.0),
            size: 1.0,
            color: Color::WHITE,
            rotation: 0.0,
            age: 0.0,
            lifetime: 2.0,
        }
    }

    #[test]
    fn test_gravity_affector() {
        let aff = ParticleAffector::new(
            AffectorType::Gravity { force: Vec3::new(0.0, -10.0, 0.0) },
            0,
        );
        let mut p = make_particle();
        aff.apply(&mut p, 1.0);
        assert!((p.velocity.y - (-10.0)).abs() < 1e-5);
    }

    #[test]
    fn test_drag_affector() {
        let aff = ParticleAffector::new(
            AffectorType::Drag { coefficient: 0.5 },
            0,
        );
        let mut p = make_particle();
        p.velocity = Vec3::new(10.0, 0.0, 0.0);
        aff.apply(&mut p, 1.0);
        assert!((p.velocity.x - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_drag_clamps_to_zero() {
        let aff = ParticleAffector::new(
            AffectorType::Drag { coefficient: 2.0 },
            0,
        );
        let mut p = make_particle();
        aff.apply(&mut p, 1.0);
        assert!((p.velocity.x).abs() < 1e-5);
    }

    #[test]
    fn test_wind_affector() {
        let aff = ParticleAffector::new(
            AffectorType::Wind {
                direction: Vec3::new(1.0, 0.0, 0.0),
                strength: 5.0,
                turbulence: 0.0,
            },
            0,
        );
        let mut p = make_particle();
        p.velocity = Vec3::ZERO;
        aff.apply(&mut p, 1.0);
        assert!((p.velocity.x - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_wind_with_turbulence() {
        let aff = ParticleAffector::new(
            AffectorType::Wind {
                direction: Vec3::new(1.0, 0.0, 0.0),
                strength: 5.0,
                turbulence: 2.0,
            },
            0,
        );
        let mut p = make_particle();
        p.position = Vec3::new(3.14, 2.71, 1.41);
        aff.apply(&mut p, 1.0);
        // Should have velocity with wind + some turbulence component
        assert!(p.velocity.length() > 0.1);
    }

    #[test]
    fn test_color_over_life() {
        let gradient = ColorGradient::new(vec![
            (0.0, Color::WHITE),
            (1.0, Color::new(1.0, 0.0, 0.0, 0.0)),
        ]);
        let aff = ParticleAffector::new(
            AffectorType::ColorOverLife { gradient },
            0,
        );
        let mut p = make_particle();
        p.age = 1.0;
        p.lifetime = 2.0;
        aff.apply(&mut p, 0.1);
        assert!((p.color.g - 0.5).abs() < 1e-5);
        assert!((p.color.a - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_size_over_life() {
        let curve = FloatCurve::new(vec![(0.0, 1.0), (1.0, 0.0)]);
        let aff = ParticleAffector::new(
            AffectorType::SizeOverLife { curve },
            0,
        );
        let mut p = make_particle();
        p.age = 1.0;
        p.lifetime = 2.0;
        aff.apply(&mut p, 0.1);
        assert!((p.size - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_rotation_over_life() {
        let curve = FloatCurve::constant(std::f32::consts::PI);
        let aff = ParticleAffector::new(
            AffectorType::RotationOverLife { curve },
            0,
        );
        let mut p = make_particle();
        aff.apply(&mut p, 1.0);
        assert!((p.rotation - std::f32::consts::PI).abs() < 1e-5);
    }

    #[test]
    fn test_attractor() {
        let aff = ParticleAffector::new(
            AffectorType::Attractor {
                position: Vec3::new(10.0, 0.0, 0.0),
                strength: 5.0,
                radius: 20.0,
            },
            0,
        );
        let mut p = make_particle();
        p.velocity = Vec3::ZERO;
        aff.apply(&mut p, 1.0);
        assert!(p.velocity.x > 0.0);
    }

    #[test]
    fn test_attractor_outside_radius() {
        let aff = ParticleAffector::new(
            AffectorType::Attractor {
                position: Vec3::new(100.0, 0.0, 0.0),
                strength: 5.0,
                radius: 2.0,
            },
            0,
        );
        let mut p = make_particle();
        let vx_before = p.velocity.x;
        aff.apply(&mut p, 1.0);
        assert!((p.velocity.x - vx_before).abs() < 1e-6);
    }

    #[test]
    fn test_vortex() {
        let aff = ParticleAffector::new(
            AffectorType::Vortex {
                axis: Vec3::new(0.0, 1.0, 0.0),
                center: Vec3::ZERO,
                strength: 5.0,
            },
            0,
        );
        let mut p = make_particle();
        p.position = Vec3::new(1.0, 0.0, 0.0);
        p.velocity = Vec3::ZERO;
        aff.apply(&mut p, 1.0);
        // Should gain velocity perpendicular to radial direction
        assert!(p.velocity.length() > 0.1);
    }

    #[test]
    fn test_noise_displacement() {
        let aff = ParticleAffector::new(
            AffectorType::Noise { frequency: 1.0, amplitude: 2.0 },
            0,
        );
        let mut p = make_particle();
        p.position = Vec3::new(1.37, 2.71, 3.14);
        p.age = 0.5;
        let pos_before = p.position;
        aff.apply(&mut p, 1.0);
        let delta = (p.position.x - pos_before.x).abs()
            + (p.position.y - pos_before.y).abs()
            + (p.position.z - pos_before.z).abs();
        // Value noise at non-integer coords should produce displacement
        assert!(delta > 1e-6);
    }

    #[test]
    fn test_velocity_limit() {
        let aff = ParticleAffector::new(
            AffectorType::VelocityLimit { max_speed: 2.0 },
            0,
        );
        let mut p = make_particle();
        p.velocity = Vec3::new(10.0, 10.0, 10.0);
        aff.apply(&mut p, 1.0);
        let speed = p.velocity.length();
        assert!((speed - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_velocity_limit_under() {
        let aff = ParticleAffector::new(
            AffectorType::VelocityLimit { max_speed: 100.0 },
            0,
        );
        let mut p = make_particle();
        let speed_before = p.velocity.length();
        aff.apply(&mut p, 1.0);
        assert!((p.velocity.length() - speed_before).abs() < 1e-5);
    }

    #[test]
    fn test_disabled_affector() {
        let mut aff = ParticleAffector::new(
            AffectorType::Gravity { force: Vec3::new(0.0, -100.0, 0.0) },
            0,
        );
        aff.enabled = false;
        let mut p = make_particle();
        let vy = p.velocity.y;
        aff.apply(&mut p, 1.0);
        assert!((p.velocity.y - vy).abs() < 1e-6);
    }

    #[test]
    fn test_pipeline_priority_order() {
        let mut pipeline = AffectorPipeline::new();
        pipeline.add(ParticleAffector::new(
            AffectorType::Gravity { force: Vec3::new(0.0, -10.0, 0.0) },
            10,
        ));
        pipeline.add(ParticleAffector::new(
            AffectorType::VelocityLimit { max_speed: 5.0 },
            20,
        ));
        let mut p = make_particle();
        p.velocity = Vec3::ZERO;
        pipeline.apply_all(&mut p, 1.0);
        // Gravity adds -10, then velocity limit clamps to 5
        let speed = p.velocity.length();
        assert!((speed - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_pipeline_batch() {
        let mut pipeline = AffectorPipeline::new();
        pipeline.add(ParticleAffector::new(
            AffectorType::Gravity { force: Vec3::new(0.0, -10.0, 0.0) },
            0,
        ));
        let mut particles = vec![make_particle(), make_particle(), make_particle()];
        pipeline.apply_batch(&mut particles, 1.0);
        for p in &particles {
            assert!((p.velocity.y - (-10.0)).abs() < 1e-5);
        }
    }

    #[test]
    fn test_pipeline_clear() {
        let mut pipeline = AffectorPipeline::new();
        pipeline.add(ParticleAffector::new(AffectorType::Drag { coefficient: 1.0 }, 0));
        assert_eq!(pipeline.len(), 1);
        pipeline.clear();
        assert!(pipeline.is_empty());
    }

    #[test]
    fn test_pipeline_remove() {
        let mut pipeline = AffectorPipeline::new();
        pipeline.add(ParticleAffector::new(AffectorType::Drag { coefficient: 1.0 }, 0));
        pipeline.add(ParticleAffector::new(AffectorType::Drag { coefficient: 2.0 }, 1));
        let removed = pipeline.remove_at(0);
        assert!(removed.is_some());
        assert_eq!(pipeline.len(), 1);
    }

    #[test]
    fn test_color_gradient_single_stop() {
        let g = ColorGradient::new(vec![(0.5, Color::new(1.0, 0.0, 0.0, 1.0))]);
        let c = g.sample(0.0);
        assert!((c.r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_float_curve_sample_midpoint() {
        let curve = FloatCurve::new(vec![(0.0, 0.0), (1.0, 10.0)]);
        assert!((curve.sample(0.5) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_float_curve_constant() {
        let curve = FloatCurve::constant(7.0);
        assert!((curve.sample(0.0) - 7.0).abs() < 1e-6);
        assert!((curve.sample(0.5) - 7.0).abs() < 1e-6);
        assert!((curve.sample(1.0) - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalized_age() {
        let p = ParticleData {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            size: 1.0,
            color: Color::WHITE,
            rotation: 0.0,
            age: 1.5,
            lifetime: 3.0,
        };
        assert!((p.normalized_age() - 0.5).abs() < 1e-6);
    }
}
