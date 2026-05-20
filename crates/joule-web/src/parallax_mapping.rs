//! Parallax and displacement mapping.
//!
//! Basic parallax mapping (single UV offset), steep parallax mapping
//! (ray-march through height layers, 8–32 steps), parallax occlusion mapping
//! (POM) with binary search refinement, self-shadowing via light-direction
//! height tracing, height scale/bias parameters. Pure Rust — no external deps.

use std::fmt;

// ── Inline vector types ─────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn add(self, other: Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y }
    }

    pub fn scale(self, s: f32) -> Self {
        Self { x: self.x * s, y: self.y * s }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn normalize(self) -> Self {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if len < 1e-10 { return Self::new(0.0, 0.0, 0.0); }
        Self { x: self.x / len, y: self.y / len, z: self.z / len }
    }

    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

// ── Height map ──────────────────────────────────────────────────

/// A 2D height field with f32 values in 0..1.
#[derive(Debug, Clone)]
pub struct HeightField {
    pub width: u32,
    pub height: u32,
    data: Vec<f32>,
}

impl HeightField {
    pub fn new(width: u32, height: u32, data: Vec<f32>) -> Result<Self, String> {
        let expected = (width as usize) * (height as usize);
        if data.len() != expected {
            return Err(format!("expected {} samples, got {}", expected, data.len()));
        }
        Ok(Self { width, height, data })
    }

    /// Flat height field (all zeros).
    pub fn flat(width: u32, height: u32) -> Self {
        let count = (width as usize) * (height as usize);
        Self { width, height, data: vec![0.0; count] }
    }

    /// Constant height.
    pub fn constant(width: u32, height: u32, h: f32) -> Self {
        let count = (width as usize) * (height as usize);
        Self { width, height, data: vec![h; count] }
    }

    /// Sample height at UV (bilinear interpolation, clamped).
    pub fn sample(&self, u: f32, v: f32) -> f32 {
        let u = u.clamp(0.0, 1.0);
        let v = v.clamp(0.0, 1.0);

        let fx = u * (self.width as f32 - 1.0);
        let fy = v * (self.height as f32 - 1.0);
        let x0 = fx.floor() as u32;
        let y0 = fy.floor() as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let frac_x = fx - fx.floor();
        let frac_y = fy - fy.floor();

        let get = |x: u32, y: u32| self.data[(y * self.width + x) as usize];
        let top = get(x0, y0) + (get(x1, y0) - get(x0, y0)) * frac_x;
        let bot = get(x0, y1) + (get(x1, y1) - get(x0, y1)) * frac_x;
        top + (bot - top) * frac_y
    }

    /// Set height at integer coordinate.
    pub fn set(&mut self, x: u32, y: u32, h: f32) {
        if x < self.width && y < self.height {
            self.data[(y * self.width + x) as usize] = h;
        }
    }
}

impl fmt::Display for HeightField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HeightField({}x{})", self.width, self.height)
    }
}

// ── Parallax parameters ─────────────────────────────────────────

/// Configuration for parallax mapping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParallaxParams {
    /// World-space displacement scale.
    pub height_scale: f32,
    /// Bias offset (shifts the zero plane).
    pub height_bias: f32,
    /// Minimum number of ray-march steps (steep/POM).
    pub min_steps: u32,
    /// Maximum number of ray-march steps.
    pub max_steps: u32,
    /// Binary search refinement iterations (POM only).
    pub binary_refinements: u32,
}

impl ParallaxParams {
    pub fn new(height_scale: f32) -> Self {
        Self {
            height_scale,
            height_bias: 0.0,
            min_steps: 8,
            max_steps: 32,
            binary_refinements: 5,
        }
    }

    pub fn with_bias(mut self, bias: f32) -> Self {
        self.height_bias = bias;
        self
    }

    pub fn with_steps(mut self, min_val: u32, max_val: u32) -> Self {
        self.min_steps = min_val.max(1);
        self.max_steps = max_val.max(min_val);
        self
    }

    pub fn with_refinements(mut self, r: u32) -> Self {
        self.binary_refinements = r;
        self
    }

    /// Compute effective step count based on view angle (steeper = more steps).
    /// `cos_angle` is the cosine of the angle between view dir and surface normal (0..1).
    pub fn step_count(&self, cos_angle: f32) -> u32 {
        let t = 1.0 - cos_angle.abs().clamp(0.0, 1.0);
        let steps = self.min_steps as f32 + (self.max_steps as f32 - self.min_steps as f32) * t;
        steps.round() as u32
    }
}

impl Default for ParallaxParams {
    fn default() -> Self {
        Self::new(0.05)
    }
}

// ── Basic parallax mapping ──────────────────────────────────────

/// Simple parallax mapping: single UV offset based on height and view direction.
///
/// `view_dir` should be in tangent space, normalised.
/// Returns the offset UV coordinate.
pub fn parallax_basic(
    uv: Vec2,
    view_dir: Vec3,
    height_field: &HeightField,
    params: &ParallaxParams,
) -> Vec2 {
    let h = height_field.sample(uv.x, uv.y) + params.height_bias;
    let vz = view_dir.z.abs().max(0.001);
    let offset = Vec2::new(
        view_dir.x / vz * h * params.height_scale,
        view_dir.y / vz * h * params.height_scale,
    );
    uv.add(offset.scale(-1.0))
}

// ── Steep parallax mapping ──────────────────────────────────────

/// Steep parallax mapping: ray-march through height layers.
///
/// Returns the offset UV and the depth at which the ray intersected the surface.
pub fn parallax_steep(
    uv: Vec2,
    view_dir: Vec3,
    height_field: &HeightField,
    params: &ParallaxParams,
) -> (Vec2, f32) {
    let cos_angle = view_dir.z.abs();
    let num_steps = params.step_count(cos_angle);
    let layer_depth = 1.0 / num_steps as f32;

    let vz = view_dir.z.abs().max(0.001);
    let delta_uv = Vec2::new(
        -view_dir.x / vz * params.height_scale / num_steps as f32,
        -view_dir.y / vz * params.height_scale / num_steps as f32,
    );

    let mut current_uv = uv;
    let mut current_depth = 0.0f32;
    let mut current_map_h = height_field.sample(current_uv.x, current_uv.y) + params.height_bias;

    for _ in 0..num_steps {
        if current_depth >= current_map_h {
            break;
        }
        current_uv = current_uv.add(delta_uv);
        current_depth += layer_depth;
        current_map_h = height_field.sample(
            current_uv.x.clamp(0.0, 1.0),
            current_uv.y.clamp(0.0, 1.0),
        ) + params.height_bias;
    }

    (current_uv, current_depth)
}

// ── Parallax occlusion mapping (POM) ────────────────────────────

/// POM: steep parallax + binary search refinement between last two layers.
///
/// Returns the offset UV and final depth.
pub fn parallax_occlusion(
    uv: Vec2,
    view_dir: Vec3,
    height_field: &HeightField,
    params: &ParallaxParams,
) -> (Vec2, f32) {
    let cos_angle = view_dir.z.abs();
    let num_steps = params.step_count(cos_angle);
    let layer_depth = 1.0 / num_steps as f32;

    let vz = view_dir.z.abs().max(0.001);
    let delta_uv = Vec2::new(
        -view_dir.x / vz * params.height_scale / num_steps as f32,
        -view_dir.y / vz * params.height_scale / num_steps as f32,
    );

    let mut prev_uv = uv;
    let mut current_uv = uv;
    let mut current_depth = 0.0f32;
    let mut prev_depth = 0.0f32;
    let mut current_map_h = height_field.sample(current_uv.x, current_uv.y) + params.height_bias;

    // March until we go below the surface.
    for _ in 0..num_steps {
        if current_depth >= current_map_h {
            break;
        }
        prev_uv = current_uv;
        prev_depth = current_depth;
        current_uv = current_uv.add(delta_uv);
        current_depth += layer_depth;
        current_map_h = height_field.sample(
            current_uv.x.clamp(0.0, 1.0),
            current_uv.y.clamp(0.0, 1.0),
        ) + params.height_bias;
    }

    // Binary search refinement.
    let mut lo_uv = prev_uv;
    let mut hi_uv = current_uv;
    let mut lo_depth = prev_depth;
    let mut hi_depth = current_depth;

    for _ in 0..params.binary_refinements {
        let mid_uv = Vec2::new(
            (lo_uv.x + hi_uv.x) * 0.5,
            (lo_uv.y + hi_uv.y) * 0.5,
        );
        let mid_depth = (lo_depth + hi_depth) * 0.5;
        let mid_h = height_field.sample(
            mid_uv.x.clamp(0.0, 1.0),
            mid_uv.y.clamp(0.0, 1.0),
        ) + params.height_bias;

        if mid_depth < mid_h {
            // Still above surface.
            lo_uv = mid_uv;
            lo_depth = mid_depth;
        } else {
            // Below surface.
            hi_uv = mid_uv;
            hi_depth = mid_depth;
        }
    }

    let final_uv = Vec2::new(
        (lo_uv.x + hi_uv.x) * 0.5,
        (lo_uv.y + hi_uv.y) * 0.5,
    );
    let final_depth = (lo_depth + hi_depth) * 0.5;
    (final_uv, final_depth)
}

// ── Self-shadowing ──────────────────────────────────────────────

/// Compute self-shadow factor by tracing from the surface point toward the light.
///
/// Returns 0.0 (fully shadowed) to 1.0 (fully lit).
/// `surface_uv` is the parallax-offset UV, `surface_depth` is where the ray hit,
/// `light_dir` is in tangent space (normalised, pointing toward light).
pub fn self_shadow(
    surface_uv: Vec2,
    surface_depth: f32,
    light_dir: Vec3,
    height_field: &HeightField,
    params: &ParallaxParams,
) -> f32 {
    let lz = light_dir.z.abs().max(0.001);
    let num_steps = params.step_count(light_dir.z.abs()).max(4);
    let layer_depth = surface_depth / num_steps as f32;

    let delta_uv = Vec2::new(
        light_dir.x / lz * params.height_scale / num_steps as f32,
        light_dir.y / lz * params.height_scale / num_steps as f32,
    );

    let mut current_uv = surface_uv;
    let mut current_depth = surface_depth;
    let mut shadow = 1.0f32;

    for _ in 0..num_steps {
        current_uv = current_uv.add(delta_uv);
        current_depth -= layer_depth;

        if current_depth <= 0.0 {
            break;
        }

        let map_h = height_field.sample(
            current_uv.x.clamp(0.0, 1.0),
            current_uv.y.clamp(0.0, 1.0),
        ) + params.height_bias;

        if map_h > current_depth {
            // Occluded — soft shadow based on how much we're blocked.
            let occlude = (map_h - current_depth).min(1.0);
            shadow = shadow.min(1.0 - occlude);
        }
    }

    shadow.clamp(0.0, 1.0)
}

// ── Displacement mapping helper ─────────────────────────────────

/// Displace a vertex position along its normal based on height.
pub fn displace_vertex(
    position: Vec3,
    normal: Vec3,
    height: f32,
    params: &ParallaxParams,
) -> Vec3 {
    let n = normal.normalize();
    let displacement = (height + params.height_bias) * params.height_scale;
    Vec3::new(
        position.x + n.x * displacement,
        position.y + n.y * displacement,
        position.z + n.z * displacement,
    )
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn test_height_field_creation() {
        let hf = HeightField::new(4, 4, vec![0.5; 16]).unwrap();
        assert!(approx(hf.sample(0.5, 0.5), 0.5));
    }

    #[test]
    fn test_height_field_wrong_size() {
        assert!(HeightField::new(4, 4, vec![0.0; 15]).is_err());
    }

    #[test]
    fn test_height_field_flat() {
        let hf = HeightField::flat(8, 8);
        assert!(approx(hf.sample(0.3, 0.7), 0.0));
    }

    #[test]
    fn test_height_field_constant() {
        let hf = HeightField::constant(4, 4, 0.8);
        assert!(approx(hf.sample(0.5, 0.5), 0.8));
    }

    #[test]
    fn test_height_field_set() {
        let mut hf = HeightField::flat(4, 4);
        hf.set(0, 0, 1.0);
        assert!(approx(hf.sample(0.0, 0.0), 1.0));
    }

    #[test]
    fn test_height_field_bilinear() {
        let mut hf = HeightField::flat(2, 1);
        hf.set(0, 0, 0.0);
        hf.set(1, 0, 1.0);
        let mid = hf.sample(0.5, 0.0);
        assert!(approx(mid, 0.5));
    }

    #[test]
    fn test_params_default() {
        let p = ParallaxParams::default();
        assert!(approx(p.height_scale, 0.05));
        assert_eq!(p.min_steps, 8);
        assert_eq!(p.max_steps, 32);
    }

    #[test]
    fn test_params_step_count_head_on() {
        let p = ParallaxParams::new(0.05).with_steps(8, 32);
        // Head-on: cos_angle close to 1 → min steps.
        let steps = p.step_count(1.0);
        assert_eq!(steps, 8);
    }

    #[test]
    fn test_params_step_count_grazing() {
        let p = ParallaxParams::new(0.05).with_steps(8, 32);
        // Grazing angle: cos_angle close to 0 → max steps.
        let steps = p.step_count(0.0);
        assert_eq!(steps, 32);
    }

    #[test]
    fn test_basic_parallax_flat() {
        let hf = HeightField::flat(4, 4);
        let params = ParallaxParams::new(0.05);
        let view = Vec3::new(0.0, 0.0, 1.0).normalize();
        let result = parallax_basic(Vec2::new(0.5, 0.5), view, &hf, &params);
        // Flat surface → no offset.
        assert!(approx(result.x, 0.5));
        assert!(approx(result.y, 0.5));
    }

    #[test]
    fn test_basic_parallax_offset() {
        let hf = HeightField::constant(4, 4, 0.5);
        let params = ParallaxParams::new(0.1);
        let view = Vec3::new(0.3, 0.0, 0.95).normalize();
        let result = parallax_basic(Vec2::new(0.5, 0.5), view, &hf, &params);
        // Should shift U coordinate.
        assert!((result.x - 0.5).abs() > 1e-5);
    }

    #[test]
    fn test_steep_parallax_flat() {
        let hf = HeightField::flat(4, 4);
        let params = ParallaxParams::new(0.05).with_steps(8, 16);
        let view = Vec3::new(0.0, 0.0, 1.0).normalize();
        let (uv, depth) = parallax_steep(Vec2::new(0.5, 0.5), view, &hf, &params);
        // Flat surface → minimal offset and depth near 0.
        assert!((uv.x - 0.5).abs() < 0.1);
        assert!(depth < 0.2);
    }

    #[test]
    fn test_steep_parallax_bump() {
        let hf = HeightField::constant(8, 8, 0.5);
        let params = ParallaxParams::new(0.1).with_steps(16, 32);
        let view = Vec3::new(0.2, 0.0, 0.98).normalize();
        let (uv, depth) = parallax_steep(Vec2::new(0.5, 0.5), view, &hf, &params);
        // Should converge to some depth.
        assert!(depth > 0.0);
        let _ = uv; // UV is valid.
    }

    #[test]
    fn test_pom_flat() {
        let hf = HeightField::flat(4, 4);
        let params = ParallaxParams::new(0.05).with_refinements(5);
        let view = Vec3::new(0.0, 0.0, 1.0).normalize();
        let (uv, depth) = parallax_occlusion(Vec2::new(0.5, 0.5), view, &hf, &params);
        assert!((uv.x - 0.5).abs() < 0.1);
        assert!(depth < 0.2);
    }

    #[test]
    fn test_pom_refinement_converges() {
        let hf = HeightField::constant(8, 8, 0.5);
        let params_coarse = ParallaxParams::new(0.1).with_steps(4, 8).with_refinements(0);
        let params_fine = ParallaxParams::new(0.1).with_steps(4, 8).with_refinements(8);
        let view = Vec3::new(0.2, 0.0, 0.98).normalize();
        let (_, d_coarse) = parallax_occlusion(Vec2::new(0.5, 0.5), view, &hf, &params_coarse);
        let (_, d_fine) = parallax_occlusion(Vec2::new(0.5, 0.5), view, &hf, &params_fine);
        // Both should be near 0.5 (constant height), fine should be closer.
        assert!((d_fine - 0.5).abs() <= (d_coarse - 0.5).abs() + 1e-3);
    }

    #[test]
    fn test_self_shadow_flat_fully_lit() {
        let hf = HeightField::flat(8, 8);
        let params = ParallaxParams::new(0.05);
        let light = Vec3::new(0.0, 0.0, 1.0).normalize();
        let shadow = self_shadow(Vec2::new(0.5, 0.5), 0.1, light, &hf, &params);
        assert!(approx(shadow, 1.0));
    }

    #[test]
    fn test_self_shadow_occluded() {
        let mut hf = HeightField::flat(8, 8);
        // Create a tall ridge.
        for x in 0..8 {
            hf.set(x, 3, 1.0);
            hf.set(x, 4, 1.0);
        }
        let params = ParallaxParams::new(0.5).with_steps(16, 32);
        let light = Vec3::new(0.0, 0.5, 0.5).normalize();
        let shadow = self_shadow(Vec2::new(0.5, 0.2), 0.3, light, &hf, &params);
        // Should be partially or fully shadowed.
        assert!(shadow < 1.0);
    }

    #[test]
    fn test_displace_vertex() {
        let pos = Vec3::new(0.0, 0.0, 0.0);
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let params = ParallaxParams::new(2.0);
        let displaced = displace_vertex(pos, normal, 0.5, &params);
        assert!(approx(displaced.y, 1.0));
        assert!(approx(displaced.x, 0.0));
    }

    #[test]
    fn test_displace_with_bias() {
        let pos = Vec3::new(1.0, 0.0, 0.0);
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let params = ParallaxParams::new(1.0).with_bias(-0.5);
        let displaced = displace_vertex(pos, normal, 1.0, &params);
        assert!(approx(displaced.z, 0.5));
    }

    #[test]
    fn test_params_with_steps() {
        let p = ParallaxParams::new(0.1).with_steps(4, 64);
        assert_eq!(p.min_steps, 4);
        assert_eq!(p.max_steps, 64);
    }

    #[test]
    fn test_height_field_display() {
        let hf = HeightField::flat(16, 16);
        let s = format!("{hf}");
        assert!(s.contains("16x16"));
    }
}
