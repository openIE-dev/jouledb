//! Directional (sun) light for the lighting engine.
//!
//! Models an infinitely distant light source (no position). Provides
//! direction, color, intensity, ambient contribution, cascaded shadow
//! map parameters, and a day/night cycle that shifts color temperature
//! based on sun angle.

use std::f64::consts::PI;

// ── Vector / Matrix types ──────────────────────────────────────

/// 3-component vector (f64).
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

    pub fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-12 { return Self::ZERO; }
        Self { x: self.x / l, y: self.y / l, z: self.z / l }
    }

    pub fn neg(self) -> Self { Self { x: -self.x, y: -self.y, z: -self.z } }

    pub fn scale(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s } }

    pub fn add(self, o: Self) -> Self { Self { x: self.x + o.x, y: self.y + o.y, z: self.z + o.z } }

    pub fn sub(self, o: Self) -> Self { Self { x: self.x - o.x, y: self.y - o.y, z: self.z - o.z } }
}

/// 4-component vector.
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

    /// Column-access helper: element at (row, col).
    pub fn at(&self, row: usize, col: usize) -> f64 { self.m[row * 4 + col] }

    /// Build a look-at (view) matrix: eye looking at center, with up hint.
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

    /// Orthographic projection matrix.
    pub fn ortho(left: f64, right: f64, bottom: f64, top: f64, near: f64, far: f64) -> Self {
        let w = right - left;
        let h = top - bottom;
        let d = far - near;
        Self {
            m: [
                2.0 / w, 0.0,     0.0,      -(right + left) / w,
                0.0,     2.0 / h, 0.0,      -(top + bottom) / h,
                0.0,     0.0,    -2.0 / d,  -(far + near) / d,
                0.0,     0.0,     0.0,       1.0,
            ],
        }
    }

    /// Multiply two 4×4 matrices.
    pub fn mul(&self, o: &Self) -> Self {
        let mut r = [0.0f64; 16];
        for row in 0..4 {
            for col in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += self.m[row * 4 + k] * o.m[k * 4 + col];
                }
                r[row * 4 + col] = sum;
            }
        }
        Self { m: r }
    }

    /// Transform a Vec4.
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

/// Linear RGB color (f64, range 0..1+).
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

    pub fn lerp(self, other: Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
        }
    }
}

// ── Cascade parameters ─────────────────────────────────────────

/// Parameters for cascaded shadow maps associated with this directional light.
#[derive(Debug, Clone, PartialEq)]
pub struct CascadeParams {
    /// Number of shadow cascades (typically 2–4).
    pub cascade_count: u32,
    /// Split distances (length = cascade_count + 1, including near and far).
    pub split_distances: Vec<f64>,
    /// Shadow map resolution per cascade.
    pub resolution: u32,
}

impl CascadeParams {
    /// Create with uniform split distances.
    pub fn uniform(cascade_count: u32, near: f64, far: f64, resolution: u32) -> Self {
        let mut splits = Vec::with_capacity(cascade_count as usize + 1);
        for i in 0..=cascade_count {
            splits.push(near + (far - near) * (i as f64 / cascade_count as f64));
        }
        Self { cascade_count, split_distances: splits, resolution }
    }

    /// Create with logarithmic split distances.
    pub fn logarithmic(cascade_count: u32, near: f64, far: f64, resolution: u32) -> Self {
        let mut splits = Vec::with_capacity(cascade_count as usize + 1);
        let n = near.max(1e-6);
        for i in 0..=cascade_count {
            let t = i as f64 / cascade_count as f64;
            splits.push(n * (far / n).powf(t));
        }
        Self { cascade_count, split_distances: splits, resolution }
    }

    /// Practical (PSSM) split: blend between uniform and logarithmic.
    pub fn practical(cascade_count: u32, near: f64, far: f64, lambda: f64, resolution: u32) -> Self {
        let uniform = Self::uniform(cascade_count, near, far, resolution);
        let log = Self::logarithmic(cascade_count, near, far, resolution);
        let lambda = lambda.clamp(0.0, 1.0);
        let splits: Vec<f64> = uniform
            .split_distances
            .iter()
            .zip(log.split_distances.iter())
            .map(|(u, l)| lambda * l + (1.0 - lambda) * u)
            .collect();
        Self { cascade_count, split_distances: splits, resolution }
    }
}

// ── Day/Night color temperature ────────────────────────────────

/// Map a color temperature in Kelvin to approximate linear RGB.
/// Tanner Helland's algorithm (simplified).
fn color_temperature_to_rgb(kelvin: f64) -> Color {
    let k = (kelvin / 100.0).clamp(1.0, 400.0);
    let r = if k <= 66.0 {
        1.0
    } else {
        let x = k - 60.0;
        (329.698727446 * x.powf(-0.1332047592) / 255.0).clamp(0.0, 1.0)
    };
    let g = if k <= 66.0 {
        let x = k;
        (99.4708025861 * x.ln() - 161.1195681661).clamp(0.0, 255.0) / 255.0
    } else {
        let x = k - 60.0;
        (288.1221695283 * x.powf(-0.0755148492) / 255.0).clamp(0.0, 1.0)
    };
    let b = if k >= 66.0 {
        1.0
    } else if k <= 19.0 {
        0.0
    } else {
        let x = k - 10.0;
        (138.5177312231 * x.ln() - 305.0447927307).clamp(0.0, 255.0) / 255.0
    };
    Color::new(r, g, b)
}

/// Map sun elevation angle (radians, 0 = horizon, PI/2 = zenith) to
/// an approximate color temperature.
pub fn sun_angle_to_temperature(elevation_rad: f64) -> f64 {
    let t = (elevation_rad / (PI / 2.0)).clamp(0.0, 1.0);
    // Horizon ≈ 2000K (warm), zenith ≈ 6500K (daylight)
    2000.0 + t * 4500.0
}

// ── DirectionalLight ───────────────────────────────────────────

/// A directional (sun/moon) light with infinite distance.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectionalLight {
    /// Direction the light travels (normalized, points *toward* surfaces).
    pub direction: Vec3,
    /// Base color (before temperature shift).
    pub color: Color,
    /// Intensity multiplier.
    pub intensity: f64,
    /// Whether this light casts shadows.
    pub casts_shadows: bool,
    /// Cascaded shadow map config (optional).
    pub cascade_params: Option<CascadeParams>,
    /// Ambient contribution (fraction of color added everywhere).
    pub ambient: f64,
}

impl DirectionalLight {
    /// Create a default directional light pointing straight down.
    pub fn new(direction: Vec3, color: Color, intensity: f64) -> Self {
        Self {
            direction: direction.normalized(),
            color,
            intensity,
            casts_shadows: false,
            cascade_params: None,
            ambient: 0.05,
        }
    }

    /// Enable shadow casting with the given cascade parameters.
    pub fn with_shadows(mut self, params: CascadeParams) -> Self {
        self.casts_shadows = true;
        self.cascade_params = Some(params);
        self
    }

    /// Set ambient contribution.
    pub fn with_ambient(mut self, ambient: f64) -> Self {
        self.ambient = ambient.clamp(0.0, 1.0);
        self
    }

    /// Effective color at the surface (color × intensity).
    pub fn effective_color(&self) -> Color {
        self.color.scale(self.intensity)
    }

    /// Ambient color contribution.
    pub fn ambient_color(&self) -> Color {
        self.color.scale(self.intensity * self.ambient)
    }

    /// Lambertian diffuse contribution at a surface with given normal.
    pub fn diffuse_at(&self, surface_normal: Vec3) -> Color {
        let n_dot_l = surface_normal.normalized().dot(self.direction.neg());
        let factor = n_dot_l.max(0.0) * self.intensity;
        self.color.scale(factor)
    }

    /// Compute the light-space matrix for orthographic shadow projection.
    ///
    /// `scene_center` and `scene_radius` describe a bounding sphere that
    /// the ortho frustum must enclose. The view looks from far away along
    /// `self.direction` toward `scene_center`.
    pub fn light_space_matrix(&self, scene_center: Vec3, scene_radius: f64) -> Mat4 {
        let light_pos = scene_center.sub(self.direction.scale(scene_radius));
        let up = if self.direction.cross(Vec3::UP).length() < 1e-6 {
            Vec3::new(0.0, 0.0, 1.0)
        } else {
            Vec3::UP
        };
        let view = Mat4::look_at(light_pos, scene_center, up);
        let proj = Mat4::ortho(
            -scene_radius, scene_radius,
            -scene_radius, scene_radius,
            0.0, 2.0 * scene_radius,
        );
        proj.mul(&view)
    }

    /// Apply day/night cycle: compute color from sun elevation angle.
    /// Returns a new light with color temperature–shifted color.
    pub fn with_day_night_cycle(&self, elevation_rad: f64) -> Self {
        let temp = sun_angle_to_temperature(elevation_rad);
        let temp_color = color_temperature_to_rgb(temp);
        // Blend base color with temperature color (50/50).
        let blended = self.color.lerp(temp_color, 0.5);
        // Intensity drops near horizon.
        let elevation_factor = (elevation_rad / (PI / 2.0)).clamp(0.0, 1.0);
        let intensity = self.intensity * (0.1 + 0.9 * elevation_factor);
        Self {
            direction: self.direction,
            color: blended,
            intensity,
            casts_shadows: self.casts_shadows,
            cascade_params: self.cascade_params.clone(),
            ambient: self.ambient,
        }
    }

    /// Compute cascade light-space matrices for each cascade slice.
    /// Returns one Mat4 per cascade.
    pub fn cascade_matrices(&self, scene_center: Vec3, scene_radius: f64) -> Vec<Mat4> {
        let params = match &self.cascade_params {
            Some(p) => p,
            None => return Vec::new(),
        };
        let mut matrices = Vec::with_capacity(params.cascade_count as usize);
        for i in 0..params.cascade_count as usize {
            let near = params.split_distances[i];
            let far = params.split_distances[i + 1];
            let mid = (near + far) * 0.5;
            let radius = (far - near) * 0.5;
            // Offset the cascade center along the view direction (forward = -direction of light).
            let offset = self.direction.neg().scale(mid);
            let cascade_center = scene_center.add(offset);
            let effective_radius = radius.max(scene_radius * 0.1);
            matrices.push(self.light_space_matrix(cascade_center, effective_radius));
        }
        matrices
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < EPS }
    fn color_approx(a: Color, b: Color) -> bool {
        approx(a.r, b.r) && approx(a.g, b.g) && approx(a.b, b.b)
    }

    #[test]
    fn vec3_normalize() {
        let v = Vec3::new(3.0, 0.0, 4.0).normalized();
        assert!(approx(v.length(), 1.0));
        assert!(approx(v.x, 0.6));
        assert!(approx(v.z, 0.8));
    }

    #[test]
    fn vec3_zero_normalize() {
        let v = Vec3::ZERO.normalized();
        assert!(approx(v.length(), 0.0));
    }

    #[test]
    fn vec3_cross() {
        let c = Vec3::new(1.0, 0.0, 0.0).cross(Vec3::new(0.0, 1.0, 0.0));
        assert!(approx(c.z, 1.0));
    }

    #[test]
    fn mat4_identity_mul() {
        let m = Mat4::IDENTITY.mul(&Mat4::IDENTITY);
        assert_eq!(m, Mat4::IDENTITY);
    }

    #[test]
    fn mat4_ortho_center_maps_to_origin() {
        let p = Mat4::ortho(-10.0, 10.0, -10.0, 10.0, 0.0, 20.0);
        let v = p.transform_vec4(Vec4::new(0.0, 0.0, -10.0, 1.0));
        assert!(approx(v.x, 0.0));
        assert!(approx(v.y, 0.0));
    }

    #[test]
    fn color_lerp() {
        let c = Color::BLACK.lerp(Color::WHITE, 0.5);
        assert!(approx(c.r, 0.5));
        assert!(approx(c.g, 0.5));
        assert!(approx(c.b, 0.5));
    }

    #[test]
    fn color_lerp_clamp() {
        let c = Color::BLACK.lerp(Color::WHITE, 2.0);
        assert!(color_approx(c, Color::WHITE));
    }

    #[test]
    fn cascade_uniform_splits() {
        let cp = CascadeParams::uniform(4, 0.1, 100.0, 1024);
        assert_eq!(cp.split_distances.len(), 5);
        assert!(approx(cp.split_distances[0], 0.1));
        assert!(approx(cp.split_distances[4], 100.0));
    }

    #[test]
    fn cascade_logarithmic_splits() {
        let cp = CascadeParams::logarithmic(3, 1.0, 1000.0, 1024);
        assert_eq!(cp.split_distances.len(), 4);
        assert!(approx(cp.split_distances[0], 1.0));
        assert!(approx(cp.split_distances[3], 1000.0));
        // Middle split should be geometric mean–ish.
        assert!(cp.split_distances[1] > 1.0);
        assert!(cp.split_distances[1] < cp.split_distances[2]);
    }

    #[test]
    fn cascade_practical_lambda0_is_uniform() {
        let uni = CascadeParams::uniform(3, 1.0, 100.0, 512);
        let prac = CascadeParams::practical(3, 1.0, 100.0, 0.0, 512);
        for (a, b) in uni.split_distances.iter().zip(prac.split_distances.iter()) {
            assert!(approx(*a, *b));
        }
    }

    #[test]
    fn cascade_practical_lambda1_is_log() {
        let log = CascadeParams::logarithmic(3, 1.0, 100.0, 512);
        let prac = CascadeParams::practical(3, 1.0, 100.0, 1.0, 512);
        for (a, b) in log.split_distances.iter().zip(prac.split_distances.iter()) {
            assert!(approx(*a, *b));
        }
    }

    #[test]
    fn directional_light_effective_color() {
        let l = DirectionalLight::new(
            Vec3::new(0.0, -1.0, 0.0),
            Color::new(1.0, 0.8, 0.6),
            2.0,
        );
        let c = l.effective_color();
        assert!(approx(c.r, 2.0));
        assert!(approx(c.g, 1.6));
    }

    #[test]
    fn directional_light_ambient_color() {
        let l = DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Color::WHITE, 1.0)
            .with_ambient(0.1);
        let a = l.ambient_color();
        assert!(approx(a.r, 0.1));
    }

    #[test]
    fn diffuse_facing_light() {
        let l = DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Color::WHITE, 1.0);
        let c = l.diffuse_at(Vec3::new(0.0, 1.0, 0.0));
        assert!(approx(c.r, 1.0));
    }

    #[test]
    fn diffuse_perpendicular() {
        let l = DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Color::WHITE, 1.0);
        let c = l.diffuse_at(Vec3::new(1.0, 0.0, 0.0));
        assert!(approx(c.r, 0.0));
    }

    #[test]
    fn diffuse_facing_away() {
        let l = DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Color::WHITE, 1.0);
        let c = l.diffuse_at(Vec3::new(0.0, -1.0, 0.0));
        assert!(approx(c.r, 0.0));
    }

    #[test]
    fn light_space_matrix_not_identity() {
        let l = DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Color::WHITE, 1.0);
        let m = l.light_space_matrix(Vec3::ZERO, 10.0);
        assert_ne!(m, Mat4::IDENTITY);
    }

    #[test]
    fn shadow_flag_default_false() {
        let l = DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Color::WHITE, 1.0);
        assert!(!l.casts_shadows);
    }

    #[test]
    fn with_shadows_enables_flag() {
        let l = DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Color::WHITE, 1.0)
            .with_shadows(CascadeParams::uniform(3, 0.1, 100.0, 1024));
        assert!(l.casts_shadows);
        assert!(l.cascade_params.is_some());
    }

    #[test]
    fn cascade_matrices_count() {
        let l = DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Color::WHITE, 1.0)
            .with_shadows(CascadeParams::uniform(4, 0.1, 100.0, 1024));
        let matrices = l.cascade_matrices(Vec3::ZERO, 50.0);
        assert_eq!(matrices.len(), 4);
    }

    #[test]
    fn cascade_matrices_empty_without_params() {
        let l = DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Color::WHITE, 1.0);
        let matrices = l.cascade_matrices(Vec3::ZERO, 50.0);
        assert!(matrices.is_empty());
    }

    #[test]
    fn day_night_zenith() {
        let l = DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Color::WHITE, 1.0);
        let noon = l.with_day_night_cycle(PI / 2.0);
        // At zenith, intensity should be close to original.
        assert!(approx(noon.intensity, 1.0));
    }

    #[test]
    fn day_night_horizon_low_intensity() {
        let l = DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Color::WHITE, 1.0);
        let dawn = l.with_day_night_cycle(0.0);
        // At horizon, intensity drops significantly.
        assert!(dawn.intensity < 0.2);
    }

    #[test]
    fn sun_temperature_at_horizon() {
        let t = sun_angle_to_temperature(0.0);
        assert!(approx(t, 2000.0));
    }

    #[test]
    fn sun_temperature_at_zenith() {
        let t = sun_angle_to_temperature(PI / 2.0);
        assert!(approx(t, 6500.0));
    }

    #[test]
    fn color_temperature_warm() {
        let c = color_temperature_to_rgb(2000.0);
        // Warm light: red > green > blue.
        assert!(c.r > c.g);
        assert!(c.g > c.b);
    }

    #[test]
    fn color_temperature_daylight() {
        let c = color_temperature_to_rgb(6500.0);
        // Daylight is roughly white.
        assert!(c.r > 0.8);
        assert!(c.g > 0.8);
        assert!(c.b > 0.8);
    }

    #[test]
    fn direction_normalized_on_create() {
        let l = DirectionalLight::new(Vec3::new(0.0, -10.0, 0.0), Color::WHITE, 1.0);
        assert!(approx(l.direction.length(), 1.0));
    }

    #[test]
    fn light_space_matrix_with_straight_down_direction() {
        // Straight down is collinear with UP — should still produce valid matrix.
        let l = DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Color::WHITE, 1.0);
        let m = l.light_space_matrix(Vec3::ZERO, 5.0);
        // Matrix should be finite.
        for val in &m.m {
            assert!(val.is_finite(), "matrix element must be finite");
        }
    }
}
