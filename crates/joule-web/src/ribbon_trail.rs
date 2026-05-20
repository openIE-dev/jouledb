// ribbon_trail.rs — Ribbon/trail renderer with quad strip geometry
// Part of joule-web: Particles & VFX cluster

use std::collections::VecDeque;

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
    pub const TRANSPARENT: Self = Self { r: 1.0, g: 1.0, b: 1.0, a: 0.0 };

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

/// A sampled point along the trail.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrailSample {
    pub position: Vec3,
    pub time: f32,
}

/// 2D UV coordinate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// Vertex of a trail quad strip.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrailVertex {
    pub position: Vec3,
    pub color: Color,
    pub uv: Vec2,
}

/// Width curve over trail length (piecewise linear).
#[derive(Debug, Clone, PartialEq)]
pub struct WidthCurve {
    pub keys: Vec<(f32, f32)>,
}

impl WidthCurve {
    pub fn constant(w: f32) -> Self {
        Self { keys: vec![(0.0, w), (1.0, w)] }
    }

    pub fn tapered(base_width: f32) -> Self {
        Self { keys: vec![(0.0, base_width), (1.0, 0.0)] }
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
                let frac = if (t1 - t0).abs() < 1e-9 { 0.0 } else { (t - t0) / (t1 - t0) };
                return v0 + (v1 - v0) * frac;
            }
        }
        self.keys[last].1
    }
}

/// Color gradient over trail length.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorGradient {
    pub stops: Vec<(f32, Color)>,
}

impl ColorGradient {
    pub fn new(stops: Vec<(f32, Color)>) -> Self {
        let mut s = stops;
        s.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Self { stops: s }
    }

    pub fn fade_out() -> Self {
        Self::new(vec![(0.0, Color::WHITE), (1.0, Color::TRANSPARENT)])
    }

    pub fn sample(&self, t: f32) -> Color {
        if self.stops.is_empty() { return Color::WHITE; }
        if self.stops.len() == 1 || t <= self.stops[0].0 { return self.stops[0].1; }
        let last = self.stops.len() - 1;
        if t >= self.stops[last].0 { return self.stops[last].1; }
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

/// Configuration for a ribbon trail.
#[derive(Debug, Clone, PartialEq)]
pub struct RibbonTrailConfig {
    /// Minimum distance between samples.
    pub min_sample_distance: f32,
    /// Maximum number of samples (trail length in points).
    pub max_samples: usize,
    /// Maximum trail length in world units (0 = unlimited).
    pub max_distance: f32,
    pub width_curve: WidthCurve,
    pub color_gradient: ColorGradient,
    /// UV scroll speed along the trail (units per second).
    pub uv_scroll_speed: f32,
    /// Up vector for billboard orientation.
    pub up_vector: Vec3,
}

impl Default for RibbonTrailConfig {
    fn default() -> Self {
        Self {
            min_sample_distance: 0.1,
            max_samples: 128,
            max_distance: 0.0,
            width_curve: WidthCurve::tapered(0.5),
            color_gradient: ColorGradient::fade_out(),
            uv_scroll_speed: 0.0,
            up_vector: Vec3::new(0.0, 1.0, 0.0),
        }
    }
}

/// Catmull-Rom interpolation between four points.
fn catmull_rom(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;
    Vec3 {
        x: 0.5 * ((2.0 * p1.x) + (-p0.x + p2.x) * t + (2.0 * p0.x - 5.0 * p1.x + 4.0 * p2.x - p3.x) * t2 + (-p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x) * t3),
        y: 0.5 * ((2.0 * p1.y) + (-p0.y + p2.y) * t + (2.0 * p0.y - 5.0 * p1.y + 4.0 * p2.y - p3.y) * t2 + (-p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y) * t3),
        z: 0.5 * ((2.0 * p1.z) + (-p0.z + p2.z) * t + (2.0 * p0.z - 5.0 * p1.z + 4.0 * p2.z - p3.z) * t2 + (-p0.z + 3.0 * p1.z - 3.0 * p2.z + p3.z) * t3),
    }
}

/// Runtime ribbon trail state.
pub struct RibbonTrail {
    config: RibbonTrailConfig,
    samples: VecDeque<TrailSample>,
    current_time: f32,
    uv_offset: f32,
    attached: bool,
    total_distance: f32,
}

impl RibbonTrail {
    pub fn new(config: RibbonTrailConfig) -> Self {
        Self {
            config,
            samples: VecDeque::new(),
            current_time: 0.0,
            uv_offset: 0.0,
            attached: true,
            total_distance: 0.0,
        }
    }

    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    pub fn is_attached(&self) -> bool {
        self.attached
    }

    /// Detach the trail so it stops following and fades.
    pub fn detach(&mut self) {
        self.attached = false;
    }

    /// Re-attach the trail.
    pub fn attach(&mut self) {
        self.attached = true;
    }

    pub fn clear(&mut self) {
        self.samples.clear();
        self.total_distance = 0.0;
    }

    pub fn total_distance(&self) -> f32 {
        self.total_distance
    }

    pub fn config(&self) -> &RibbonTrailConfig {
        &self.config
    }

    /// Add a position sample if enough distance from the last.
    pub fn add_sample(&mut self, position: Vec3, time: f32) {
        self.current_time = time;

        if let Some(last) = self.samples.back() {
            let dist = (position - last.position).length();
            if dist < self.config.min_sample_distance {
                return;
            }
            self.total_distance += dist;
        }

        self.samples.push_back(TrailSample { position, time });

        // Enforce max samples
        while self.samples.len() > self.config.max_samples {
            self.samples.pop_front();
        }

        // Enforce max distance
        if self.config.max_distance > 0.0 {
            self.trim_by_distance();
        }
    }

    fn trim_by_distance(&mut self) {
        loop {
            if self.samples.len() < 2 {
                break;
            }
            // Compute total distance of remaining trail
            let mut total = 0.0f32;
            for i in 1..self.samples.len() {
                total += (self.samples[i].position - self.samples[i - 1].position).length();
            }
            if total <= self.config.max_distance {
                break;
            }
            self.samples.pop_front();
        }
    }

    /// Update UV scrolling.
    pub fn update(&mut self, dt: f32) {
        self.uv_offset += self.config.uv_scroll_speed * dt;
        self.current_time += dt;
    }

    /// Get Catmull-Rom interpolated position at a fractional sample index.
    pub fn interpolated_position(&self, index_f: f32) -> Vec3 {
        let n = self.samples.len();
        if n == 0 { return Vec3::ZERO; }
        if n == 1 { return self.samples[0].position; }

        let idx = index_f.floor() as i64;
        let t = index_f - index_f.floor();

        let get = |i: i64| -> Vec3 {
            let clamped = i.max(0).min(n as i64 - 1) as usize;
            self.samples[clamped].position
        };

        catmull_rom(get(idx - 1), get(idx), get(idx + 1), get(idx + 2), t)
    }

    /// Build quad strip geometry for rendering.
    pub fn build_geometry(&self, camera_pos: Vec3) -> Vec<TrailVertex> {
        let n = self.samples.len();
        if n < 2 {
            return Vec::new();
        }

        let mut vertices = Vec::with_capacity(n * 2);
        let mut cumulative_dist = vec![0.0f32; n];
        for i in 1..n {
            cumulative_dist[i] = cumulative_dist[i - 1]
                + (self.samples[i].position - self.samples[i - 1].position).length();
        }
        let total_len = cumulative_dist[n - 1].max(1e-6);

        for i in 0..n {
            let pos = self.samples[i].position;
            let t_along = cumulative_dist[i] / total_len;

            // Direction along trail
            let dir = if i == 0 {
                (self.samples[1].position - pos).normalized()
            } else if i == n - 1 {
                (pos - self.samples[n - 2].position).normalized()
            } else {
                (self.samples[i + 1].position - self.samples[i - 1].position).normalized()
            };

            // Billboard right vector
            let to_camera = (camera_pos - pos).normalized();
            let right = dir.cross(to_camera).normalized();
            let right = if right.length() < 0.5 {
                dir.cross(self.config.up_vector).normalized()
            } else {
                right
            };

            let width = self.config.width_curve.sample(t_along);
            let color = self.config.color_gradient.sample(t_along);
            let u = t_along + self.uv_offset;

            // Left vertex
            vertices.push(TrailVertex {
                position: pos + right * (width * 0.5),
                color,
                uv: Vec2::new(u, 0.0),
            });
            // Right vertex
            vertices.push(TrailVertex {
                position: pos - right * (width * 0.5),
                color,
                uv: Vec2::new(u, 1.0),
            });
        }

        vertices
    }

    /// Get raw samples for inspection.
    pub fn samples(&self) -> &VecDeque<TrailSample> {
        &self.samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_trail_empty() {
        let trail = RibbonTrail::new(RibbonTrailConfig::default());
        assert_eq!(trail.sample_count(), 0);
        assert!(trail.is_attached());
    }

    #[test]
    fn test_add_sample() {
        let mut trail = RibbonTrail::new(RibbonTrailConfig::default());
        trail.add_sample(Vec3::new(0.0, 0.0, 0.0), 0.0);
        trail.add_sample(Vec3::new(1.0, 0.0, 0.0), 0.1);
        assert_eq!(trail.sample_count(), 2);
    }

    #[test]
    fn test_add_sample_too_close() {
        let cfg = RibbonTrailConfig {
            min_sample_distance: 1.0,
            ..Default::default()
        };
        let mut trail = RibbonTrail::new(cfg);
        trail.add_sample(Vec3::ZERO, 0.0);
        trail.add_sample(Vec3::new(0.1, 0.0, 0.0), 0.1);
        assert_eq!(trail.sample_count(), 1); // second too close
    }

    #[test]
    fn test_max_samples_enforced() {
        let cfg = RibbonTrailConfig {
            max_samples: 5,
            min_sample_distance: 0.01,
            ..Default::default()
        };
        let mut trail = RibbonTrail::new(cfg);
        for i in 0..10 {
            trail.add_sample(Vec3::new(i as f32, 0.0, 0.0), i as f32 * 0.1);
        }
        assert!(trail.sample_count() <= 5);
    }

    #[test]
    fn test_max_distance_trim() {
        let cfg = RibbonTrailConfig {
            max_distance: 3.0,
            min_sample_distance: 0.01,
            max_samples: 100,
            ..Default::default()
        };
        let mut trail = RibbonTrail::new(cfg);
        for i in 0..20 {
            trail.add_sample(Vec3::new(i as f32 * 0.5, 0.0, 0.0), i as f32 * 0.1);
        }
        // Total trail distance should be <= 3.0
        let samples = trail.samples();
        let mut total = 0.0f32;
        for i in 1..samples.len() {
            total += (samples[i].position - samples[i - 1].position).length();
        }
        assert!(total <= 3.0 + 0.6); // allow for one segment overshoot
    }

    #[test]
    fn test_detach_and_attach() {
        let mut trail = RibbonTrail::new(RibbonTrailConfig::default());
        assert!(trail.is_attached());
        trail.detach();
        assert!(!trail.is_attached());
        trail.attach();
        assert!(trail.is_attached());
    }

    #[test]
    fn test_clear() {
        let mut trail = RibbonTrail::new(RibbonTrailConfig::default());
        trail.add_sample(Vec3::ZERO, 0.0);
        trail.add_sample(Vec3::new(1.0, 0.0, 0.0), 0.1);
        trail.clear();
        assert_eq!(trail.sample_count(), 0);
    }

    #[test]
    fn test_width_curve_constant() {
        let wc = WidthCurve::constant(2.0);
        assert!((wc.sample(0.0) - 2.0).abs() < 1e-6);
        assert!((wc.sample(0.5) - 2.0).abs() < 1e-6);
        assert!((wc.sample(1.0) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_width_curve_tapered() {
        let wc = WidthCurve::tapered(1.0);
        assert!((wc.sample(0.0) - 1.0).abs() < 1e-6);
        assert!((wc.sample(0.5) - 0.5).abs() < 1e-6);
        assert!((wc.sample(1.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_color_gradient_fade_out() {
        let g = ColorGradient::fade_out();
        let c0 = g.sample(0.0);
        assert!((c0.a - 1.0).abs() < 1e-6);
        let c1 = g.sample(1.0);
        assert!((c1.a - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_build_geometry_empty() {
        let trail = RibbonTrail::new(RibbonTrailConfig::default());
        let geo = trail.build_geometry(Vec3::new(0.0, 0.0, 5.0));
        assert!(geo.is_empty());
    }

    #[test]
    fn test_build_geometry_single_sample() {
        let mut trail = RibbonTrail::new(RibbonTrailConfig::default());
        trail.add_sample(Vec3::ZERO, 0.0);
        let geo = trail.build_geometry(Vec3::new(0.0, 0.0, 5.0));
        assert!(geo.is_empty()); // Need at least 2 samples
    }

    #[test]
    fn test_build_geometry_two_samples() {
        let cfg = RibbonTrailConfig {
            min_sample_distance: 0.01,
            width_curve: WidthCurve::constant(1.0),
            ..Default::default()
        };
        let mut trail = RibbonTrail::new(cfg);
        trail.add_sample(Vec3::new(0.0, 0.0, 0.0), 0.0);
        trail.add_sample(Vec3::new(1.0, 0.0, 0.0), 0.1);
        let geo = trail.build_geometry(Vec3::new(0.0, 0.0, 5.0));
        // 2 samples * 2 vertices each = 4
        assert_eq!(geo.len(), 4);
    }

    #[test]
    fn test_build_geometry_vertex_pairs() {
        let cfg = RibbonTrailConfig {
            min_sample_distance: 0.01,
            width_curve: WidthCurve::constant(2.0),
            color_gradient: ColorGradient::new(vec![(0.0, Color::WHITE), (1.0, Color::WHITE)]),
            ..Default::default()
        };
        let mut trail = RibbonTrail::new(cfg);
        trail.add_sample(Vec3::new(0.0, 0.0, 0.0), 0.0);
        trail.add_sample(Vec3::new(2.0, 0.0, 0.0), 0.1);
        let geo = trail.build_geometry(Vec3::new(0.0, 0.0, 5.0));
        // Each pair should have uv.y = 0 and uv.y = 1
        assert!((geo[0].uv.y - 0.0).abs() < 1e-6);
        assert!((geo[1].uv.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_uv_scrolling() {
        let cfg = RibbonTrailConfig {
            uv_scroll_speed: 2.0,
            min_sample_distance: 0.01,
            ..Default::default()
        };
        let mut trail = RibbonTrail::new(cfg);
        trail.add_sample(Vec3::ZERO, 0.0);
        trail.add_sample(Vec3::new(1.0, 0.0, 0.0), 0.1);
        trail.update(0.5);
        // uv_offset should be 2.0 * 0.5 = 1.0
        let geo = trail.build_geometry(Vec3::new(0.0, 0.0, 5.0));
        assert!(!geo.is_empty());
        // First vertex u should include offset
        assert!(geo[0].uv.x >= 0.9);
    }

    #[test]
    fn test_catmull_rom_interpolation() {
        let p0 = Vec3::new(-1.0, 0.0, 0.0);
        let p1 = Vec3::new(0.0, 0.0, 0.0);
        let p2 = Vec3::new(1.0, 0.0, 0.0);
        let p3 = Vec3::new(2.0, 0.0, 0.0);
        // At t=0 should be p1
        let r0 = catmull_rom(p0, p1, p2, p3, 0.0);
        assert!((r0.x - 0.0).abs() < 1e-5);
        // At t=1 should be p2
        let r1 = catmull_rom(p0, p1, p2, p3, 1.0);
        assert!((r1.x - 1.0).abs() < 1e-5);
        // At t=0.5 should be midpoint for linear arrangement
        let rm = catmull_rom(p0, p1, p2, p3, 0.5);
        assert!((rm.x - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_interpolated_position() {
        let cfg = RibbonTrailConfig {
            min_sample_distance: 0.01,
            ..Default::default()
        };
        let mut trail = RibbonTrail::new(cfg);
        trail.add_sample(Vec3::new(0.0, 0.0, 0.0), 0.0);
        trail.add_sample(Vec3::new(1.0, 0.0, 0.0), 0.1);
        trail.add_sample(Vec3::new(2.0, 0.0, 0.0), 0.2);
        let mid = trail.interpolated_position(1.0);
        assert!((mid.x - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_interpolated_position_empty() {
        let trail = RibbonTrail::new(RibbonTrailConfig::default());
        let p = trail.interpolated_position(0.0);
        assert!((p.x).abs() < 1e-6);
    }

    #[test]
    fn test_total_distance() {
        let cfg = RibbonTrailConfig {
            min_sample_distance: 0.01,
            ..Default::default()
        };
        let mut trail = RibbonTrail::new(cfg);
        trail.add_sample(Vec3::new(0.0, 0.0, 0.0), 0.0);
        trail.add_sample(Vec3::new(3.0, 4.0, 0.0), 0.1);
        assert!((trail.total_distance() - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_color_gradient_multi_stops() {
        let g = ColorGradient::new(vec![
            (0.0, Color::new(1.0, 0.0, 0.0, 1.0)),
            (0.5, Color::new(0.0, 1.0, 0.0, 1.0)),
            (1.0, Color::new(0.0, 0.0, 1.0, 1.0)),
        ]);
        let c = g.sample(0.25);
        assert!((c.r - 0.5).abs() < 1e-5);
        assert!((c.g - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_many_samples_geometry() {
        let cfg = RibbonTrailConfig {
            min_sample_distance: 0.01,
            max_samples: 50,
            ..Default::default()
        };
        let mut trail = RibbonTrail::new(cfg);
        for i in 0..20 {
            trail.add_sample(Vec3::new(i as f32 * 0.5, (i as f32 * 0.3).sin(), 0.0), i as f32 * 0.1);
        }
        let geo = trail.build_geometry(Vec3::new(0.0, 0.0, 10.0));
        assert_eq!(geo.len(), trail.sample_count() * 2);
    }

    #[test]
    fn test_vec3_cross() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let c = a.cross(b);
        assert!((c.z - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_width_curve_empty_keys() {
        let wc = WidthCurve { keys: vec![] };
        assert!((wc.sample(0.5) - 1.0).abs() < 1e-6);
    }
}
