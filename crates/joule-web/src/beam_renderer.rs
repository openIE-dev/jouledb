// beam_renderer.rs — Beam/laser/lightning renderer
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
        let l = self.length();
        if l < 1e-10 { Self::ZERO } else { Self { x: self.x / l, y: self.y / l, z: self.z / l } }
    }

    pub fn cross(self, o: Self) -> Self {
        Self {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }

    pub fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            z: self.z + (other.z - self.z) * t,
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
impl std::ops::Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self { Self { x: -self.x, y: -self.y, z: -self.z } }
}

/// 2D UV coordinate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self { Self { x, y } }
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

    pub fn lerp(self, o: Self, t: f32) -> Self {
        Self {
            r: self.r + (o.r - self.r) * t,
            g: self.g + (o.g - self.g) * t,
            b: self.b + (o.b - self.b) * t,
            a: self.a + (o.a - self.a) * t,
        }
    }
}

/// Width profile along beam length (piecewise linear).
#[derive(Debug, Clone, PartialEq)]
pub struct WidthProfile {
    pub keys: Vec<(f32, f32)>,
}

impl WidthProfile {
    pub fn constant(w: f32) -> Self {
        Self { keys: vec![(0.0, w), (1.0, w)] }
    }

    pub fn tapered(center: f32) -> Self {
        Self { keys: vec![(0.0, 0.0), (0.5, center), (1.0, 0.0)] }
    }

    pub fn sample(&self, t: f32) -> f32 {
        if self.keys.is_empty() { return 1.0; }
        if self.keys.len() == 1 || t <= self.keys[0].0 { return self.keys[0].1; }
        let last = self.keys.len() - 1;
        if t >= self.keys[last].0 { return self.keys[last].1; }
        for i in 0..last {
            let (t0, v0) = self.keys[i];
            let (t1, v1) = self.keys[i + 1];
            if t >= t0 && t <= t1 {
                let f = if (t1 - t0).abs() < 1e-9 { 0.0 } else { (t - t0) / (t1 - t0) };
                return v0 + (v1 - v0) * f;
            }
        }
        self.keys[last].1
    }
}

/// End-cap shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CapShape {
    None,
    Flat,
    Round { segments: u32 },
    Arrow { length: f32 },
}

/// Beam type.
#[derive(Debug, Clone, PartialEq)]
pub enum BeamType {
    /// Straight line.
    Straight,
    /// Parabolic arc with a peak height.
    Arc { peak_height: f32 },
    /// Lightning with recursive midpoint displacement.
    Lightning {
        /// Number of recursive subdivision levels.
        subdivisions: u32,
        /// Displacement magnitude.
        displacement: f32,
        /// Seed for deterministic jitter.
        seed: u32,
    },
}

/// A single beam segment point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeamPoint {
    pub position: Vec3,
    pub t: f32,
}

/// Beam vertex for rendering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeamVertex {
    pub position: Vec3,
    pub color: Color,
    pub uv: Vec2,
}

/// Configuration for a beam.
#[derive(Debug, Clone, PartialEq)]
pub struct BeamConfig {
    pub beam_type: BeamType,
    pub width_profile: WidthProfile,
    pub color: Color,
    pub uv_tile_count: f32,
    pub glow_intensity: f32,
    pub start_cap: CapShape,
    pub end_cap: CapShape,
}

impl Default for BeamConfig {
    fn default() -> Self {
        Self {
            beam_type: BeamType::Straight,
            width_profile: WidthProfile::constant(0.2),
            color: Color::WHITE,
            uv_tile_count: 1.0,
            glow_intensity: 1.0,
            start_cap: CapShape::None,
            end_cap: CapShape::None,
        }
    }
}

/// Simple hash for lightning noise.
fn hash_u32(mut n: u32) -> u32 {
    n = n.wrapping_mul(2654435761);
    n ^= n >> 16;
    n = n.wrapping_mul(2246822519);
    n ^= n >> 13;
    n
}

/// Returns a float in [-1, 1] from a seed.
fn hash_float(seed: u32) -> f32 {
    let h = hash_u32(seed);
    (h as f32 / u32::MAX as f32) * 2.0 - 1.0
}

/// Generate lightning points via recursive midpoint displacement.
fn generate_lightning(start: Vec3, end: Vec3, subdivisions: u32, displacement: f32, seed: u32) -> Vec<BeamPoint> {
    let mut points = vec![
        BeamPoint { position: start, t: 0.0 },
        BeamPoint { position: end, t: 1.0 },
    ];

    let beam_dir = (end - start).normalized();
    let perp1 = if beam_dir.dot(Vec3::UP).abs() < 0.99 {
        beam_dir.cross(Vec3::UP).normalized()
    } else {
        beam_dir.cross(Vec3::new(1.0, 0.0, 0.0)).normalized()
    };
    let perp2 = beam_dir.cross(perp1).normalized();

    let mut disp = displacement;
    for level in 0..subdivisions {
        let mut new_points = Vec::with_capacity(points.len() * 2);
        for i in 0..points.len() - 1 {
            new_points.push(points[i]);
            let mid_pos = points[i].position.lerp(points[i + 1].position, 0.5);
            let mid_t = (points[i].t + points[i + 1].t) * 0.5;
            let s = seed.wrapping_add(level * 1000 + i as u32);
            let offset = perp1 * (hash_float(s) * disp)
                + perp2 * (hash_float(s.wrapping_add(7919)) * disp);
            new_points.push(BeamPoint {
                position: mid_pos + offset,
                t: mid_t,
            });
        }
        new_points.push(*points.last().unwrap());
        points = new_points;
        disp *= 0.5;
    }

    points
}

/// A single beam instance.
pub struct Beam {
    pub start: Vec3,
    pub end: Vec3,
    pub config: BeamConfig,
    /// If set, the beam is shortened to this collision point.
    pub collision_point: Option<Vec3>,
    cached_points: Vec<BeamPoint>,
    time: f32,
}

impl Beam {
    pub fn new(start: Vec3, end: Vec3, config: BeamConfig) -> Self {
        let mut beam = Self {
            start,
            end,
            config,
            collision_point: None,
            cached_points: Vec::new(),
            time: 0.0,
        };
        beam.rebuild_points();
        beam
    }

    pub fn effective_end(&self) -> Vec3 {
        self.collision_point.unwrap_or(self.end)
    }

    pub fn set_endpoints(&mut self, start: Vec3, end: Vec3) {
        self.start = start;
        self.end = end;
        self.rebuild_points();
    }

    pub fn set_collision(&mut self, point: Option<Vec3>) {
        self.collision_point = point;
        self.rebuild_points();
    }

    /// Rebuild the beam segment points from the beam type.
    pub fn rebuild_points(&mut self) {
        let start = self.start;
        let end = self.effective_end();

        self.cached_points = match &self.config.beam_type {
            BeamType::Straight => {
                let segments = 8;
                (0..=segments)
                    .map(|i| {
                        let t = i as f32 / segments as f32;
                        BeamPoint { position: start.lerp(end, t), t }
                    })
                    .collect()
            }
            BeamType::Arc { peak_height } => {
                let segments = 16;
                (0..=segments)
                    .map(|i| {
                        let t = i as f32 / segments as f32;
                        let mut pos = start.lerp(end, t);
                        // Parabolic height: 4h*t*(1-t)
                        pos.y += 4.0 * peak_height * t * (1.0 - t);
                        BeamPoint { position: pos, t }
                    })
                    .collect()
            }
            BeamType::Lightning { subdivisions, displacement, seed } => {
                generate_lightning(start, end, *subdivisions, *displacement, *seed)
            }
        };
    }

    /// Update the beam for animation.
    pub fn update(&mut self, dt: f32) {
        self.time += dt;
        // For lightning, regenerate with time-based seed variation
        if let BeamType::Lightning { subdivisions, displacement, seed } = &self.config.beam_type {
            let animated_seed = seed.wrapping_add((self.time * 10.0) as u32);
            let start = self.start;
            let end = self.effective_end();
            self.cached_points = generate_lightning(start, end, *subdivisions, *displacement, animated_seed);
        }
    }

    pub fn points(&self) -> &[BeamPoint] {
        &self.cached_points
    }

    pub fn point_count(&self) -> usize {
        self.cached_points.len()
    }

    /// Generate beam geometry (quad strip).
    pub fn build_geometry(&self, camera_pos: Vec3) -> Vec<BeamVertex> {
        let pts = &self.cached_points;
        if pts.len() < 2 {
            return Vec::new();
        }

        let mut verts = Vec::with_capacity(pts.len() * 2);

        for (i, bp) in pts.iter().enumerate() {
            let dir = if i == 0 {
                (pts[1].position - pts[0].position).normalized()
            } else if i == pts.len() - 1 {
                (pts[i].position - pts[i - 1].position).normalized()
            } else {
                (pts[i + 1].position - pts[i - 1].position).normalized()
            };

            let to_cam = (camera_pos - bp.position).normalized();
            let mut right = dir.cross(to_cam).normalized();
            if right.length() < 0.5 {
                right = dir.cross(Vec3::UP).normalized();
            }

            let width = self.config.width_profile.sample(bp.t);
            let u = bp.t * self.config.uv_tile_count;
            let color = self.config.color;

            verts.push(BeamVertex {
                position: bp.position + right * (width * 0.5),
                color,
                uv: Vec2::new(u, 0.0),
            });
            verts.push(BeamVertex {
                position: bp.position - right * (width * 0.5),
                color,
                uv: Vec2::new(u, 1.0),
            });
        }

        verts
    }

    /// Compute glow radius at a given point along the beam.
    pub fn glow_radius_at(&self, t: f32) -> f32 {
        let base_width = self.config.width_profile.sample(t);
        base_width * self.config.glow_intensity
    }

    pub fn beam_length(&self) -> f32 {
        (self.effective_end() - self.start).length()
    }
}

/// A chain of connected beams (multi-segment).
pub struct BeamChain {
    beams: Vec<Beam>,
}

impl BeamChain {
    pub fn new() -> Self {
        Self { beams: Vec::new() }
    }

    pub fn add_beam(&mut self, beam: Beam) {
        self.beams.push(beam);
    }

    pub fn len(&self) -> usize {
        self.beams.len()
    }

    pub fn is_empty(&self) -> bool {
        self.beams.is_empty()
    }

    pub fn from_waypoints(waypoints: &[Vec3], config: BeamConfig) -> Self {
        let mut chain = Self::new();
        for i in 0..waypoints.len().saturating_sub(1) {
            chain.add_beam(Beam::new(waypoints[i], waypoints[i + 1], config.clone()));
        }
        chain
    }

    pub fn update(&mut self, dt: f32) {
        for b in &mut self.beams {
            b.update(dt);
        }
    }

    pub fn total_length(&self) -> f32 {
        self.beams.iter().map(|b| b.beam_length()).sum()
    }

    pub fn build_all_geometry(&self, camera_pos: Vec3) -> Vec<BeamVertex> {
        let mut all = Vec::new();
        for b in &self.beams {
            all.extend(b.build_geometry(camera_pos));
        }
        all
    }

    pub fn beams(&self) -> &[Beam] {
        &self.beams
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam() -> Vec3 {
        Vec3::new(0.0, 0.0, 10.0)
    }

    #[test]
    fn test_straight_beam_points() {
        let beam = Beam::new(
            Vec3::ZERO,
            Vec3::new(10.0, 0.0, 0.0),
            BeamConfig::default(),
        );
        assert!(beam.point_count() >= 2);
        let first = beam.points()[0].position;
        assert!((first.x).abs() < 1e-5);
        let last = beam.points().last().unwrap().position;
        assert!((last.x - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_arc_beam_peak() {
        let config = BeamConfig {
            beam_type: BeamType::Arc { peak_height: 5.0 },
            ..Default::default()
        };
        let beam = Beam::new(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), config);
        // Midpoint should have elevated Y
        let mid = beam.points().iter().find(|p| (p.t - 0.5).abs() < 0.05);
        assert!(mid.is_some());
        assert!(mid.unwrap().position.y > 4.0);
    }

    #[test]
    fn test_lightning_beam_subdivisions() {
        let config = BeamConfig {
            beam_type: BeamType::Lightning {
                subdivisions: 3,
                displacement: 1.0,
                seed: 42,
            },
            ..Default::default()
        };
        let beam = Beam::new(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), config);
        // 3 subdivisions: 2 -> 3 -> 5 -> 9 points
        assert_eq!(beam.point_count(), 9);
    }

    #[test]
    fn test_lightning_different_seeds() {
        let make = |seed: u32| {
            let config = BeamConfig {
                beam_type: BeamType::Lightning {
                    subdivisions: 2,
                    displacement: 2.0,
                    seed,
                },
                ..Default::default()
            };
            Beam::new(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), config)
        };
        let b1 = make(1);
        let b2 = make(999);
        // At least one intermediate point should differ
        let mut any_differ = false;
        for i in 1..b1.point_count() - 1 {
            let d = (b1.points()[i].position.y - b2.points()[i].position.y).abs();
            if d > 1e-3 {
                any_differ = true;
                break;
            }
        }
        assert!(any_differ);
    }

    #[test]
    fn test_beam_collision_shortens() {
        let mut beam = Beam::new(
            Vec3::ZERO,
            Vec3::new(10.0, 0.0, 0.0),
            BeamConfig::default(),
        );
        beam.set_collision(Some(Vec3::new(5.0, 0.0, 0.0)));
        let eff_end = beam.effective_end();
        assert!((eff_end.x - 5.0).abs() < 1e-5);
        assert!((beam.beam_length() - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_beam_no_collision() {
        let beam = Beam::new(
            Vec3::ZERO,
            Vec3::new(10.0, 0.0, 0.0),
            BeamConfig::default(),
        );
        assert!(beam.collision_point.is_none());
        assert!((beam.beam_length() - 10.0).abs() < 1e-4);
    }

    #[test]
    fn test_width_profile_constant() {
        let wp = WidthProfile::constant(0.5);
        assert!((wp.sample(0.0) - 0.5).abs() < 1e-6);
        assert!((wp.sample(1.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_width_profile_tapered() {
        let wp = WidthProfile::tapered(2.0);
        assert!((wp.sample(0.0) - 0.0).abs() < 1e-6);
        assert!((wp.sample(0.5) - 2.0).abs() < 1e-6);
        assert!((wp.sample(1.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_build_geometry_straight() {
        let beam = Beam::new(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), BeamConfig::default());
        let geo = beam.build_geometry(cam());
        assert!(!geo.is_empty());
        // Should be pairs: 2 vertices per point
        assert_eq!(geo.len() % 2, 0);
    }

    #[test]
    fn test_build_geometry_uv_tiling() {
        let config = BeamConfig {
            uv_tile_count: 3.0,
            ..Default::default()
        };
        let beam = Beam::new(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), config);
        let geo = beam.build_geometry(cam());
        let last_u = geo.last().unwrap().uv.x;
        assert!((last_u - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_glow_radius() {
        let config = BeamConfig {
            width_profile: WidthProfile::constant(1.0),
            glow_intensity: 2.0,
            ..Default::default()
        };
        let beam = Beam::new(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), config);
        assert!((beam.glow_radius_at(0.5) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_set_endpoints() {
        let mut beam = Beam::new(Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0), BeamConfig::default());
        beam.set_endpoints(Vec3::new(1.0, 0.0, 0.0), Vec3::new(11.0, 0.0, 0.0));
        assert!((beam.start.x - 1.0).abs() < 1e-5);
        assert!((beam.end.x - 11.0).abs() < 1e-5);
        assert!((beam.beam_length() - 10.0).abs() < 1e-4);
    }

    #[test]
    fn test_beam_update_lightning_changes() {
        let config = BeamConfig {
            beam_type: BeamType::Lightning {
                subdivisions: 2,
                displacement: 1.0,
                seed: 42,
            },
            ..Default::default()
        };
        let mut beam = Beam::new(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), config);
        let pts_before: Vec<Vec3> = beam.points().iter().map(|p| p.position).collect();
        beam.update(1.0);
        let pts_after: Vec<Vec3> = beam.points().iter().map(|p| p.position).collect();
        // Some intermediate points should change after animation
        let mut changed = false;
        for i in 1..pts_before.len() - 1 {
            if (pts_before[i].y - pts_after[i].y).abs() > 1e-3 {
                changed = true;
                break;
            }
        }
        assert!(changed);
    }

    #[test]
    fn test_beam_chain_from_waypoints() {
        let waypoints = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(10.0, 5.0, 0.0),
        ];
        let chain = BeamChain::from_waypoints(&waypoints, BeamConfig::default());
        assert_eq!(chain.len(), 2);
    }

    #[test]
    fn test_beam_chain_total_length() {
        let waypoints = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(3.0, 4.0, 0.0),
        ];
        let chain = BeamChain::from_waypoints(&waypoints, BeamConfig::default());
        assert!((chain.total_length() - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_beam_chain_empty() {
        let chain = BeamChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
        assert!((chain.total_length()).abs() < 1e-6);
    }

    #[test]
    fn test_beam_chain_geometry() {
        let waypoints = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
        ];
        let chain = BeamChain::from_waypoints(&waypoints, BeamConfig::default());
        let geo = chain.build_all_geometry(cam());
        assert!(!geo.is_empty());
    }

    #[test]
    fn test_cap_shapes() {
        let config = BeamConfig {
            start_cap: CapShape::Round { segments: 8 },
            end_cap: CapShape::Arrow { length: 0.5 },
            ..Default::default()
        };
        let beam = Beam::new(Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0), config);
        assert_eq!(beam.config.start_cap, CapShape::Round { segments: 8 });
        assert_eq!(beam.config.end_cap, CapShape::Arrow { length: 0.5 });
    }

    #[test]
    fn test_hash_float_range() {
        for seed in 0..100 {
            let v = hash_float(seed);
            assert!(v >= -1.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_width_profile_empty() {
        let wp = WidthProfile { keys: vec![] };
        assert!((wp.sample(0.5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_color_lerp() {
        let c = Color::WHITE.lerp(Color::new(0.0, 0.0, 0.0, 0.0), 0.5);
        assert!((c.r - 0.5).abs() < 1e-6);
        assert!((c.a - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_beam_chain_update() {
        let config = BeamConfig {
            beam_type: BeamType::Lightning {
                subdivisions: 2,
                displacement: 1.0,
                seed: 0,
            },
            ..Default::default()
        };
        let waypoints = vec![Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0)];
        let mut chain = BeamChain::from_waypoints(&waypoints, config);
        chain.update(0.5);
        assert_eq!(chain.len(), 1);
    }
}
