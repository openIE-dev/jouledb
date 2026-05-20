//! Spot light (cone) for the lighting engine.
//!
//! A focused light source with position, direction, inner/outer cone
//! angles, range, smooth angular falloff, shadow-casting perspective
//! projection, cookie/gobo texture support, and volumetric cone
//! approximation for frustum culling.

use std::f64::consts::PI;

// ── Vector / Matrix types ──────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const UP: Self = Self { x: 0.0, y: 1.0, z: 0.0 };

    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    pub fn dot(self, o: Self) -> f64 { self.x * o.x + self.y * o.y + self.z * o.z }
    pub fn cross(self, o: Self) -> Self {
        Self {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }
    pub fn length(self) -> f64 { self.dot(self).sqrt() }
    pub fn length_sq(self) -> f64 { self.dot(self) }
    pub fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-12 { return Self::ZERO; }
        Self { x: self.x / l, y: self.y / l, z: self.z / l }
    }
    pub fn sub(self, o: Self) -> Self { Self { x: self.x - o.x, y: self.y - o.y, z: self.z - o.z } }
    pub fn add(self, o: Self) -> Self { Self { x: self.x + o.x, y: self.y + o.y, z: self.z + o.z } }
    pub fn scale(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s } }
    pub fn neg(self) -> Self { Self { x: -self.x, y: -self.y, z: -self.z } }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Vec4 {
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Self { Self { x, y, z, w } }
}

/// Row-major 4×4 matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    pub m: [f64; 16],
}

impl Mat4 {
    pub const IDENTITY: Self = Self {
        m: [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
    };

    pub fn look_at(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        let f = center.sub(eye).normalized();
        let s = f.cross(up).normalized();
        let u = s.cross(f);
        Self {
            m: [
                s.x,  s.y,  s.z, -s.dot(eye),
                u.x,  u.y,  u.z, -u.dot(eye),
               -f.x, -f.y, -f.z,  f.dot(eye),
                0.0,  0.0,  0.0,  1.0,
            ],
        }
    }

    /// Perspective projection matrix. `fov_y` in radians.
    pub fn perspective(fov_y: f64, aspect: f64, near: f64, far: f64) -> Self {
        let f = 1.0 / (fov_y * 0.5).tan();
        let nf = near - far;
        Self {
            m: [
                f / aspect, 0.0, 0.0,                    0.0,
                0.0,        f,   0.0,                    0.0,
                0.0,        0.0, (far + near) / nf,     2.0 * far * near / nf,
                0.0,        0.0, -1.0,                   0.0,
            ],
        }
    }

    pub fn mul(&self, o: &Self) -> Self {
        let mut r = [0.0f64; 16];
        for row in 0..4 {
            for col in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 { sum += self.m[row * 4 + k] * o.m[k * 4 + col]; }
                r[row * 4 + col] = sum;
            }
        }
        Self { m: r }
    }

    pub fn transform_vec4(&self, v: Vec4) -> Vec4 {
        Vec4 {
            x: self.m[0] * v.x + self.m[1] * v.y + self.m[2] * v.z + self.m[3] * v.w,
            y: self.m[4] * v.x + self.m[5] * v.y + self.m[6] * v.z + self.m[7] * v.w,
            z: self.m[8] * v.x + self.m[9] * v.y + self.m[10] * v.z + self.m[11] * v.w,
            w: self.m[12] * v.x + self.m[13] * v.y + self.m[14] * v.z + self.m[15] * v.w,
        }
    }
}

// ── Color ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl Color {
    pub const WHITE: Self = Self { r: 1.0, g: 1.0, b: 1.0 };
    pub const BLACK: Self = Self { r: 0.0, g: 0.0, b: 0.0 };

    pub fn new(r: f64, g: f64, b: f64) -> Self { Self { r, g, b } }
    pub fn scale(self, s: f64) -> Self { Self { r: self.r * s, g: self.g * s, b: self.b * s } }
}

// ── Window / Attenuation helpers ───────────────────────────────

/// Smooth distance windowing: saturate(1 - (d/R)^4)^2.
fn distance_window(dist: f64, range: f64) -> f64 {
    if range <= 0.0 { return 0.0; }
    let ratio = dist / range;
    if ratio >= 1.0 { return 0.0; }
    let r4 = ratio * ratio * ratio * ratio;
    let v = (1.0 - r4).max(0.0);
    v * v
}

/// Inverse-square attenuation.
fn inverse_square(dist: f64) -> f64 { 1.0 / (dist * dist).max(1e-6) }

/// Smooth angular falloff between inner and outer cone angles.
/// `cos_theta` is the cosine of the angle from the light axis.
/// Returns 0..1.
fn angular_falloff(cos_theta: f64, cos_inner: f64, cos_outer: f64) -> f64 {
    if cos_inner <= cos_outer { return if cos_theta >= cos_inner { 1.0 } else { 0.0 }; }
    let t = ((cos_theta - cos_outer) / (cos_inner - cos_outer)).clamp(0.0, 1.0);
    // Smooth Hermite interpolation.
    t * t * (3.0 - 2.0 * t)
}

// ── Cookie / Gobo ──────────────────────────────────────────────

/// A projected texture (cookie/gobo) mask for the spot light.
/// Represented as a 2D grid of intensity multipliers.
#[derive(Debug, Clone, PartialEq)]
pub struct CookieTexture {
    pub width: u32,
    pub height: u32,
    /// Row-major intensity values in [0, 1].
    pub pixels: Vec<f64>,
}

impl CookieTexture {
    /// Create a solid (fully transparent / no-mask) cookie.
    pub fn solid(width: u32, height: u32) -> Self {
        Self { width, height, pixels: vec![1.0; (width * height) as usize] }
    }

    /// Create a circular falloff cookie.
    pub fn circular(size: u32) -> Self {
        let mut pixels = Vec::with_capacity((size * size) as usize);
        let center = size as f64 * 0.5;
        let max_r = center;
        for y in 0..size {
            for x in 0..size {
                let dx = x as f64 + 0.5 - center;
                let dy = y as f64 + 0.5 - center;
                let r = (dx * dx + dy * dy).sqrt() / max_r;
                let v = (1.0 - r).clamp(0.0, 1.0);
                pixels.push(v * v);
            }
        }
        Self { width: size, height: size, pixels }
    }

    /// Sample the cookie at normalized UV coordinates (0..1, 0..1).
    /// Returns intensity multiplier [0, 1].
    pub fn sample(&self, u: f64, v: f64) -> f64 {
        if self.width == 0 || self.height == 0 { return 1.0; }
        let u = u.clamp(0.0, 1.0);
        let v = v.clamp(0.0, 1.0);
        let px = ((u * self.width as f64) as u32).min(self.width - 1);
        let py = ((v * self.height as f64) as u32).min(self.height - 1);
        self.pixels[(py * self.width + px) as usize]
    }
}

// ── SpotLight ──────────────────────────────────────────────────

/// A spot light source with cone geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct SpotLight {
    /// World position.
    pub position: Vec3,
    /// Cone axis direction (normalized, points toward lit area).
    pub direction: Vec3,
    /// Light color.
    pub color: Color,
    /// Intensity (luminous power).
    pub intensity: f64,
    /// Maximum range.
    pub range: f64,
    /// Inner cone half-angle in radians (full intensity).
    pub inner_angle: f64,
    /// Outer cone half-angle in radians (falloff to zero).
    pub outer_angle: f64,
    /// Shadow casting flag.
    pub casts_shadows: bool,
    /// Shadow map resolution.
    pub shadow_resolution: u32,
    /// Optional cookie/gobo texture.
    pub cookie: Option<CookieTexture>,
}

impl SpotLight {
    pub fn new(
        position: Vec3,
        direction: Vec3,
        color: Color,
        intensity: f64,
        range: f64,
        inner_angle: f64,
        outer_angle: f64,
    ) -> Self {
        Self {
            position,
            direction: direction.normalized(),
            color,
            intensity,
            range,
            inner_angle: inner_angle.clamp(0.0, PI),
            outer_angle: outer_angle.clamp(0.0, PI),
            casts_shadows: false,
            shadow_resolution: 1024,
            cookie: None,
        }
    }

    pub fn with_shadows(mut self, resolution: u32) -> Self {
        self.casts_shadows = true;
        self.shadow_resolution = resolution;
        self
    }

    pub fn with_cookie(mut self, cookie: CookieTexture) -> Self {
        self.cookie = Some(cookie);
        self
    }

    /// Compute intensity at a world position (distance + angular falloff).
    pub fn intensity_at(&self, world_pos: Vec3) -> f64 {
        let to_point = world_pos.sub(self.position);
        let dist = to_point.length();
        if dist >= self.range || dist < 1e-12 { return 0.0; }

        let dir_to_point = to_point.scale(1.0 / dist);
        let cos_theta = dir_to_point.dot(self.direction);

        let cos_inner = self.inner_angle.cos();
        let cos_outer = self.outer_angle.cos();

        if cos_theta < cos_outer { return 0.0; }

        let angular = angular_falloff(cos_theta, cos_inner, cos_outer);
        let dist_atten = inverse_square(dist) * distance_window(dist, self.range);

        self.intensity * angular * dist_atten
    }

    /// Compute color contribution at a world position.
    pub fn color_at(&self, world_pos: Vec3) -> Color {
        self.color.scale(self.intensity_at(world_pos))
    }

    /// Lambertian diffuse at a surface point.
    pub fn diffuse_at(&self, surface_pos: Vec3, surface_normal: Vec3) -> Color {
        let to_light = self.position.sub(surface_pos);
        let dist = to_light.length();
        if dist >= self.range || dist < 1e-12 { return Color::BLACK; }

        let dir = to_light.scale(1.0 / dist);
        let n_dot_l = surface_normal.normalized().dot(dir).max(0.0);

        let i = self.intensity_at(surface_pos);
        self.color.scale(i * n_dot_l)
    }

    /// Build the shadow projection matrix (perspective × view).
    pub fn shadow_matrix(&self) -> Mat4 {
        let up = if self.direction.cross(Vec3::UP).length() < 1e-6 {
            Vec3::new(0.0, 0.0, 1.0)
        } else {
            Vec3::UP
        };
        let target = self.position.add(self.direction);
        let view = Mat4::look_at(self.position, target, up);
        let fov = self.outer_angle * 2.0;
        let proj = Mat4::perspective(fov, 1.0, 0.1, self.range);
        proj.mul(&view)
    }

    /// Bounding sphere for the inscribed cone (for culling).
    /// Returns (center, radius).
    pub fn bounding_sphere(&self) -> (Vec3, f64) {
        let sin_outer = self.outer_angle.sin();
        let cos_outer = self.outer_angle.cos();

        if self.outer_angle > PI / 4.0 {
            // Wide cone: sphere centered at light position.
            (self.position, self.range)
        } else {
            // Narrow cone: sphere can be tighter.
            let radius = self.range * sin_outer;
            let center_dist = self.range * cos_outer;
            // Place sphere center along cone axis.
            let half_dist = center_dist * 0.5;
            let sphere_radius = (radius * radius + half_dist * half_dist).sqrt();
            let center = self.position.add(self.direction.scale(half_dist));
            (center, sphere_radius.max(radius))
        }
    }

    /// Project a world point onto the cookie UV space.
    /// Returns Some((u, v)) if the point is within the cone, None otherwise.
    pub fn project_to_cookie_uv(&self, world_pos: Vec3) -> Option<(f64, f64)> {
        let to_point = world_pos.sub(self.position);
        let dist_along = to_point.dot(self.direction);
        if dist_along <= 0.0 { return None; }

        let cos_theta = dist_along / to_point.length();
        if cos_theta < self.outer_angle.cos() { return None; }

        // Build a local coordinate frame.
        let up = if self.direction.cross(Vec3::UP).length() < 1e-6 {
            Vec3::new(0.0, 0.0, 1.0)
        } else {
            Vec3::UP
        };
        let right = self.direction.cross(up).normalized();
        let local_up = right.cross(self.direction).normalized();

        let lateral = to_point.sub(self.direction.scale(dist_along));
        let half_spread = dist_along * self.outer_angle.tan();
        if half_spread < 1e-12 { return Some((0.5, 0.5)); }

        let u = 0.5 + lateral.dot(right) / (2.0 * half_spread);
        let v = 0.5 + lateral.dot(local_up) / (2.0 * half_spread);
        Some((u.clamp(0.0, 1.0), v.clamp(0.0, 1.0)))
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-5;

    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < EPS }

    fn make_spot() -> SpotLight {
        SpotLight::new(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, -1.0),
            Color::WHITE,
            100.0,
            50.0,
            PI / 8.0,   // 22.5° inner
            PI / 4.0,   // 45° outer
        )
    }

    #[test]
    fn angular_falloff_inside_inner() {
        let f = angular_falloff(1.0, 0.9, 0.7);
        assert!(approx(f, 1.0));
    }

    #[test]
    fn angular_falloff_outside_outer() {
        let f = angular_falloff(0.5, 0.9, 0.7);
        assert!(approx(f, 0.0));
    }

    #[test]
    fn angular_falloff_at_boundary() {
        let f = angular_falloff(0.7, 0.9, 0.7);
        assert!(approx(f, 0.0));
    }

    #[test]
    fn angular_falloff_midpoint() {
        let f = angular_falloff(0.8, 0.9, 0.7);
        assert!(f > 0.0 && f < 1.0);
    }

    #[test]
    fn spot_on_axis_has_intensity() {
        let s = make_spot();
        let i = s.intensity_at(Vec3::new(0.0, 0.0, -5.0));
        assert!(i > 0.0);
    }

    #[test]
    fn spot_outside_cone_no_intensity() {
        let s = make_spot();
        // Point far off-axis.
        let i = s.intensity_at(Vec3::new(100.0, 0.0, -1.0));
        assert!(approx(i, 0.0));
    }

    #[test]
    fn spot_beyond_range_zero() {
        let s = make_spot();
        let i = s.intensity_at(Vec3::new(0.0, 0.0, -60.0));
        assert!(approx(i, 0.0));
    }

    #[test]
    fn spot_behind_light_zero() {
        let s = make_spot();
        let i = s.intensity_at(Vec3::new(0.0, 0.0, 5.0));
        assert!(approx(i, 0.0));
    }

    #[test]
    fn spot_inner_brighter_than_outer() {
        let s = make_spot();
        let on_axis = s.intensity_at(Vec3::new(0.0, 0.0, -10.0));
        // At outer edge: point at 44° from axis at distance 10.
        let off = 10.0 * (PI / 4.0 - 0.02).tan();
        let at_outer = s.intensity_at(Vec3::new(off, 0.0, -10.0));
        assert!(on_axis > at_outer);
    }

    #[test]
    fn diffuse_facing() {
        let s = make_spot();
        let c = s.diffuse_at(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(c.r > 0.0);
    }

    #[test]
    fn diffuse_away() {
        let s = make_spot();
        let c = s.diffuse_at(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(approx(c.r, 0.0));
    }

    #[test]
    fn shadow_matrix_finite() {
        let s = make_spot();
        let m = s.shadow_matrix();
        for val in &m.m {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn bounding_sphere_narrow() {
        let s = SpotLight::new(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, -1.0),
            Color::WHITE,
            1.0,
            10.0,
            PI / 16.0,
            PI / 8.0,
        );
        let (_center, radius) = s.bounding_sphere();
        // Should be tighter than range.
        assert!(radius <= s.range + EPS);
    }

    #[test]
    fn bounding_sphere_wide() {
        let s = SpotLight::new(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, -1.0),
            Color::WHITE,
            1.0,
            10.0,
            PI / 4.0,
            PI / 2.0,
        );
        let (center, radius) = s.bounding_sphere();
        assert!(approx(center.x, 0.0));
        assert!(approx(radius, s.range));
    }

    #[test]
    fn cookie_solid_all_ones() {
        let c = CookieTexture::solid(4, 4);
        assert!(approx(c.sample(0.0, 0.0), 1.0));
        assert!(approx(c.sample(0.5, 0.5), 1.0));
    }

    #[test]
    fn cookie_circular_center_bright() {
        let c = CookieTexture::circular(16);
        let center = c.sample(0.5, 0.5);
        let edge = c.sample(0.0, 0.0);
        assert!(center > edge);
    }

    #[test]
    fn cookie_uv_on_axis() {
        let s = make_spot();
        let uv = s.project_to_cookie_uv(Vec3::new(0.0, 0.0, -5.0));
        assert!(uv.is_some());
        let (u, v) = uv.unwrap();
        assert!(approx(u, 0.5));
        assert!(approx(v, 0.5));
    }

    #[test]
    fn cookie_uv_behind_none() {
        let s = make_spot();
        assert!(s.project_to_cookie_uv(Vec3::new(0.0, 0.0, 5.0)).is_none());
    }

    #[test]
    fn with_shadows_sets_flag() {
        let s = make_spot().with_shadows(2048);
        assert!(s.casts_shadows);
        assert_eq!(s.shadow_resolution, 2048);
    }

    #[test]
    fn direction_normalized() {
        let s = SpotLight::new(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, -10.0),
            Color::WHITE,
            1.0,
            10.0,
            0.1,
            0.2,
        );
        assert!(approx(s.direction.length(), 1.0));
    }

    #[test]
    fn distance_window_monotone() {
        let r = 10.0;
        let mut prev = distance_window(0.0, r);
        for i in 1..=10 {
            let d = i as f64;
            let w = distance_window(d, r);
            assert!(w <= prev + EPS);
            prev = w;
        }
    }
}
