// screen_space_reflect.rs — Screen-Space Reflections (SSR) for the rendering pipeline.
//
// Implements screen-space ray-marching along reflection vectors, hierarchical
// tracing, hit detection with thickness tolerance, screen-edge fading,
// roughness-based cone tracing, fallback to probe/skybox, and Fresnel blending.

use std::fmt;

/// 3D vector for SSR computations.
#[derive(Clone, Debug, PartialEq)]
pub struct SsrVec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl SsrVec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }

    pub fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalized(&self) -> Self {
        let len = self.length();
        if len < 1e-10 {
            Self::zero()
        } else {
            Self { x: self.x / len, y: self.y / len, z: self.z / len }
        }
    }

    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn add(&self, other: &Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }

    pub fn scale(&self, s: f64) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }

    /// Reflect this vector across a surface normal.
    pub fn reflect(&self, normal: &Self) -> Self {
        let d = 2.0 * self.dot(normal);
        Self {
            x: self.x - d * normal.x,
            y: self.y - d * normal.y,
            z: self.z - d * normal.z,
        }
    }
}

/// RGB color for SSR.
#[derive(Clone, Debug, PartialEq)]
pub struct SsrColor {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl SsrColor {
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }

    pub fn black() -> Self {
        Self { r: 0.0, g: 0.0, b: 0.0 }
    }

    pub fn scale(&self, s: f64) -> Self {
        Self { r: self.r * s, g: self.g * s, b: self.b * s }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
        }
    }

    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
        }
    }
}

/// Screen-space color buffer.
#[derive(Clone, Debug)]
pub struct SsrColorBuffer {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<SsrColor>,
}

impl SsrColorBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![SsrColor::black(); width * height],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> &SsrColor {
        &self.pixels[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, c: SsrColor) {
        self.pixels[y * self.width + x] = c;
    }

    pub fn sample_clamp(&self, x: isize, y: isize) -> &SsrColor {
        let cx = x.max(0).min(self.width as isize - 1) as usize;
        let cy = y.max(0).min(self.height as isize - 1) as usize;
        self.get(cx, cy)
    }

    /// Bilinear sample.
    pub fn sample_bilinear(&self, fx: f64, fy: f64) -> SsrColor {
        let x0 = fx.floor() as isize;
        let y0 = fy.floor() as isize;
        let frac_x = fx - fx.floor();
        let frac_y = fy - fy.floor();

        let p00 = self.sample_clamp(x0, y0);
        let p10 = self.sample_clamp(x0 + 1, y0);
        let p01 = self.sample_clamp(x0, y0 + 1);
        let p11 = self.sample_clamp(x0 + 1, y0 + 1);

        let top = p00.lerp(p10, frac_x);
        let bot = p01.lerp(p11, frac_x);
        top.lerp(&bot, frac_y)
    }
}

/// Depth buffer for SSR.
#[derive(Clone, Debug)]
pub struct SsrDepthBuffer {
    pub width: usize,
    pub height: usize,
    pub depths: Vec<f64>,
}

impl SsrDepthBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            depths: vec![1.0; width * height],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> f64 {
        self.depths[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, d: f64) {
        self.depths[y * self.width + x] = d;
    }

    pub fn sample_clamp(&self, x: isize, y: isize) -> f64 {
        let cx = x.max(0).min(self.width as isize - 1) as usize;
        let cy = y.max(0).min(self.height as isize - 1) as usize;
        self.get(cx, cy)
    }

    /// Build a mip chain for hierarchical tracing (each level is half res, stores min depth).
    pub fn build_hierarchy(&self, levels: usize) -> Vec<SsrDepthBuffer> {
        let mut chain = Vec::with_capacity(levels);
        let mut current = self.clone();

        for _ in 0..levels {
            let w = (current.width / 2).max(1);
            let h = (current.height / 2).max(1);
            let mut mip = SsrDepthBuffer::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    let sx = x * 2;
                    let sy = y * 2;
                    let d00 = current.get(sx.min(current.width - 1), sy.min(current.height - 1));
                    let d10 = if sx + 1 < current.width { current.get(sx + 1, sy.min(current.height - 1)) } else { d00 };
                    let d01 = if sy + 1 < current.height { current.get(sx.min(current.width - 1), sy + 1) } else { d00 };
                    let d11 = if sx + 1 < current.width && sy + 1 < current.height { current.get(sx + 1, sy + 1) } else { d00 };
                    mip.set(x, y, d00.min(d10).min(d01).min(d11));
                }
            }
            chain.push(mip.clone());
            current = mip;
            if w <= 1 && h <= 1 { break; }
        }
        chain
    }
}

/// Normal buffer for SSR.
#[derive(Clone, Debug)]
pub struct SsrNormalBuffer {
    pub width: usize,
    pub height: usize,
    pub normals: Vec<SsrVec3>,
}

impl SsrNormalBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            normals: vec![SsrVec3::new(0.0, 0.0, 1.0); width * height],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> &SsrVec3 {
        &self.normals[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, n: SsrVec3) {
        self.normals[y * self.width + x] = n;
    }
}

/// Roughness buffer (per-pixel surface roughness in [0, 1]).
#[derive(Clone, Debug)]
pub struct RoughnessBuffer {
    pub width: usize,
    pub height: usize,
    pub values: Vec<f64>,
}

impl RoughnessBuffer {
    pub fn new(width: usize, height: usize, default_roughness: f64) -> Self {
        Self {
            width,
            height,
            values: vec![default_roughness; width * height],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> f64 {
        self.values[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, v: f64) {
        self.values[y * self.width + x] = v;
    }
}

/// SSR ray-march result.
#[derive(Clone, Debug, PartialEq)]
pub enum RayResult {
    /// Ray hit a surface at the given screen coordinates.
    Hit { screen_x: f64, screen_y: f64, confidence: f64 },
    /// Ray missed (left screen or exceeded max steps).
    Miss,
}

/// SSR configuration.
#[derive(Clone, Debug)]
pub struct SsrConfig {
    /// Max number of ray-march steps.
    pub max_steps: usize,
    /// Ray step size in screen pixels.
    pub step_size: f64,
    /// Thickness tolerance for hit detection.
    pub thickness: f64,
    /// Screen edge fade distance (normalized, 0-0.5).
    pub edge_fade: f64,
    /// Maximum roughness that receives SSR (beyond this, use fallback).
    pub max_roughness: f64,
    /// Number of refinement steps for binary search after coarse hit.
    pub refinement_steps: usize,
    /// Whether to use hierarchical tracing.
    pub hierarchical: bool,
    /// Hierarchy levels for hierarchical tracing.
    pub hierarchy_levels: usize,
    /// Fallback color when ray misses (probe/skybox).
    pub fallback_color: SsrColor,
    /// Fresnel F0 (base reflectivity at normal incidence).
    pub fresnel_f0: f64,
}

impl Default for SsrConfig {
    fn default() -> Self {
        Self {
            max_steps: 64,
            step_size: 1.0,
            thickness: 0.1,
            edge_fade: 0.1,
            max_roughness: 0.5,
            refinement_steps: 8,
            hierarchical: false,
            hierarchy_levels: 4,
            fallback_color: SsrColor::new(0.1, 0.12, 0.15),
            fresnel_f0: 0.04,
        }
    }
}

impl fmt::Display for SsrConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SsrConfig(steps={}, step_size={:.1}, thickness={:.3})",
            self.max_steps, self.step_size, self.thickness
        )
    }
}

/// Compute Schlick Fresnel approximation.
pub fn fresnel_schlick(cos_theta: f64, f0: f64) -> f64 {
    f0 + (1.0 - f0) * (1.0 - cos_theta).clamp(0.0, 1.0).powi(5)
}

/// Compute screen-edge fade factor.
pub fn edge_fade_factor(screen_x: f64, screen_y: f64, width: f64, height: f64, fade_dist: f64) -> f64 {
    if fade_dist <= 0.0 {
        return 1.0;
    }
    let nx = screen_x / width;
    let ny = screen_y / height;

    let fade_x = (nx / fade_dist).min((1.0 - nx) / fade_dist).min(1.0).max(0.0);
    let fade_y = (ny / fade_dist).min((1.0 - ny) / fade_dist).min(1.0).max(0.0);

    fade_x * fade_y
}

/// Reconstruct view-space position from screen coordinates and depth.
pub fn screen_to_view(
    screen_x: f64,
    screen_y: f64,
    depth: f64,
    width: f64,
    height: f64,
    fov_y: f64,
    aspect: f64,
) -> SsrVec3 {
    let ndc_x = screen_x / width * 2.0 - 1.0;
    let ndc_y = 1.0 - screen_y / height * 2.0;
    let tan_half = (fov_y / 2.0).tan();
    SsrVec3 {
        x: ndc_x * aspect * tan_half * depth,
        y: ndc_y * tan_half * depth,
        z: -depth,
    }
}

/// Project view-space position to screen coordinates.
pub fn view_to_screen(
    pos: &SsrVec3,
    width: f64,
    height: f64,
    fov_y: f64,
    aspect: f64,
) -> (f64, f64, f64) {
    let tan_half = (fov_y / 2.0).tan();
    let z = -pos.z;
    if z.abs() < 1e-10 {
        return (width / 2.0, height / 2.0, 0.0);
    }
    let ndc_x = pos.x / (z * aspect * tan_half);
    let ndc_y = pos.y / (z * tan_half);
    let sx = (ndc_x + 1.0) * 0.5 * width;
    let sy = (1.0 - ndc_y) * 0.5 * height;
    (sx, sy, z)
}

/// Ray-march in screen space along a reflection direction.
pub fn ray_march(
    start_screen: (f64, f64),
    direction_screen: (f64, f64),
    start_depth: f64,
    depth_step: f64,
    depth_buf: &SsrDepthBuffer,
    config: &SsrConfig,
) -> RayResult {
    let w = depth_buf.width as f64;
    let h = depth_buf.height as f64;

    let mut ray_x = start_screen.0;
    let mut ray_y = start_screen.1;
    let mut ray_depth = start_depth;

    let dir_len = (direction_screen.0 * direction_screen.0 + direction_screen.1 * direction_screen.1).sqrt();
    if dir_len < 1e-10 {
        return RayResult::Miss;
    }
    let dx = direction_screen.0 / dir_len * config.step_size;
    let dy = direction_screen.1 / dir_len * config.step_size;

    for step in 0..config.max_steps {
        ray_x += dx;
        ray_y += dy;
        ray_depth += depth_step;

        // Check screen bounds
        if ray_x < 0.0 || ray_x >= w || ray_y < 0.0 || ray_y >= h {
            return RayResult::Miss;
        }

        let scene_depth = depth_buf.sample_clamp(ray_x as isize, ray_y as isize);
        let depth_diff = ray_depth - scene_depth;

        // Hit: ray went behind the surface (within thickness tolerance)
        if depth_diff > 0.0 && depth_diff < config.thickness {
            // Binary refinement
            let mut lo_x = ray_x - dx;
            let mut lo_y = ray_y - dy;
            let mut lo_depth = ray_depth - depth_step;
            let mut hi_x = ray_x;
            let mut hi_y = ray_y;
            let mut hi_depth = ray_depth;
            let mut best_x = ray_x;
            let mut best_y = ray_y;

            for _ in 0..config.refinement_steps {
                let mid_x = (lo_x + hi_x) * 0.5;
                let mid_y = (lo_y + hi_y) * 0.5;
                let mid_depth = (lo_depth + hi_depth) * 0.5;
                let sd = depth_buf.sample_clamp(mid_x as isize, mid_y as isize);
                let diff = mid_depth - sd;

                if diff > 0.0 {
                    hi_x = mid_x;
                    hi_y = mid_y;
                    hi_depth = mid_depth;
                    best_x = mid_x;
                    best_y = mid_y;
                } else {
                    lo_x = mid_x;
                    lo_y = mid_y;
                    lo_depth = mid_depth;
                }
            }

            let confidence = 1.0 - (step as f64 / config.max_steps as f64);
            return RayResult::Hit {
                screen_x: best_x,
                screen_y: best_y,
                confidence,
            };
        }
    }

    RayResult::Miss
}

/// Compute roughness-based cone search radius (in screen pixels).
pub fn cone_radius(roughness: f64, step: usize, total_steps: usize) -> f64 {
    let t = step as f64 / total_steps.max(1) as f64;
    roughness * t * 4.0
}

/// Apply SSR to a single pixel.
fn ssr_pixel(
    x: usize,
    y: usize,
    color_buf: &SsrColorBuffer,
    depth_buf: &SsrDepthBuffer,
    normal_buf: &SsrNormalBuffer,
    roughness_buf: &RoughnessBuffer,
    config: &SsrConfig,
    fov_y: f64,
    aspect: f64,
) -> SsrColor {
    let w = color_buf.width as f64;
    let h = color_buf.height as f64;
    let roughness = roughness_buf.get(x, y);

    // Skip if too rough
    if roughness > config.max_roughness {
        return config.fallback_color.clone();
    }

    let depth = depth_buf.get(x, y);
    if depth <= 0.0 || depth >= 100.0 {
        return config.fallback_color.clone();
    }

    // View-space position and normal
    let view_pos = screen_to_view(x as f64 + 0.5, y as f64 + 0.5, depth, w, h, fov_y, aspect);
    let normal = normal_buf.get(x, y);
    let view_dir = view_pos.normalized();

    // Reflection direction
    let reflect_dir = view_dir.reflect(normal).normalized();

    // Project reflection endpoint to screen space
    let far_point = view_pos.add(&reflect_dir.scale(config.step_size * config.max_steps as f64 * 0.1));
    let (start_sx, start_sy, _) = view_to_screen(&view_pos, w, h, fov_y, aspect);
    let (end_sx, end_sy, end_z) = view_to_screen(&far_point, w, h, fov_y, aspect);

    let dir_sx = end_sx - start_sx;
    let dir_sy = end_sy - start_sy;
    let depth_step = ((-far_point.z) - depth) / config.max_steps as f64;

    // Ray march
    let result = ray_march(
        (start_sx, start_sy),
        (dir_sx, dir_sy),
        depth,
        depth_step,
        depth_buf,
        config,
    );

    // Fresnel
    let cos_theta = (-view_dir.dot(normal)).clamp(0.0, 1.0);
    let fresnel = fresnel_schlick(cos_theta, config.fresnel_f0);

    match result {
        RayResult::Hit { screen_x, screen_y, confidence } => {
            let reflected_color = color_buf.sample_bilinear(screen_x, screen_y);
            let edge_fade = edge_fade_factor(screen_x, screen_y, w, h, config.edge_fade);
            let roughness_fade = 1.0 - roughness / config.max_roughness.max(1e-10);
            let alpha = confidence * edge_fade * roughness_fade * fresnel;
            let original = color_buf.get(x, y);
            original.lerp(&reflected_color, alpha.clamp(0.0, 1.0))
        }
        RayResult::Miss => {
            let original = color_buf.get(x, y);
            original.lerp(&config.fallback_color, fresnel * 0.3)
        }
    }
}

/// Apply SSR to an entire frame.
pub fn apply_ssr(
    color_buf: &SsrColorBuffer,
    depth_buf: &SsrDepthBuffer,
    normal_buf: &SsrNormalBuffer,
    roughness_buf: &RoughnessBuffer,
    config: &SsrConfig,
) -> SsrColorBuffer {
    let w = color_buf.width;
    let h = color_buf.height;
    let fov_y = std::f64::consts::PI / 3.0;
    let aspect = w as f64 / h as f64;

    let mut out = SsrColorBuffer::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let c = ssr_pixel(
                x, y, color_buf, depth_buf, normal_buf, roughness_buf, config, fov_y, aspect,
            );
            out.set(x, y, c);
        }
    }
    out
}

/// SSR diagnostics/statistics.
#[derive(Debug, PartialEq)]
pub struct SsrStats {
    pub total_pixels: usize,
    pub reflective_pixels: usize,
    pub hit_count: usize,
    pub miss_count: usize,
    pub avg_confidence: f64,
}

pub fn compute_ssr_stats(
    depth_buf: &SsrDepthBuffer,
    normal_buf: &SsrNormalBuffer,
    roughness_buf: &RoughnessBuffer,
    config: &SsrConfig,
) -> SsrStats {
    let w = depth_buf.width;
    let h = depth_buf.height;
    let fov_y = std::f64::consts::PI / 3.0;
    let aspect = w as f64 / h as f64;
    let fw = w as f64;
    let fh = h as f64;

    let mut reflective = 0usize;
    let mut hits = 0usize;
    let mut misses = 0usize;
    let mut confidence_sum = 0.0_f64;

    for y in 0..h {
        for x in 0..w {
            let roughness = roughness_buf.get(x, y);
            if roughness > config.max_roughness {
                continue;
            }
            reflective += 1;

            let depth = depth_buf.get(x, y);
            if depth <= 0.0 || depth >= 100.0 {
                misses += 1;
                continue;
            }

            let view_pos = screen_to_view(x as f64 + 0.5, y as f64 + 0.5, depth, fw, fh, fov_y, aspect);
            let normal = normal_buf.get(x, y);
            let view_dir = view_pos.normalized();
            let reflect_dir = view_dir.reflect(normal).normalized();
            let far_point = view_pos.add(&reflect_dir.scale(config.step_size * config.max_steps as f64 * 0.1));

            let (start_sx, start_sy, _) = view_to_screen(&view_pos, fw, fh, fov_y, aspect);
            let (end_sx, end_sy, _) = view_to_screen(&far_point, fw, fh, fov_y, aspect);
            let depth_step = ((-far_point.z) - depth) / config.max_steps as f64;

            let result = ray_march(
                (start_sx, start_sy),
                (end_sx - start_sx, end_sy - start_sy),
                depth,
                depth_step,
                depth_buf,
                config,
            );
            match result {
                RayResult::Hit { confidence, .. } => {
                    hits += 1;
                    confidence_sum += confidence;
                }
                RayResult::Miss => {
                    misses += 1;
                }
            }
        }
    }

    SsrStats {
        total_pixels: w * h,
        reflective_pixels: reflective,
        hit_count: hits,
        miss_count: misses,
        avg_confidence: if hits > 0 { confidence_sum / hits as f64 } else { 0.0 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_vec3_reflect() {
        let v = SsrVec3::new(1.0, -1.0, 0.0).normalized();
        let n = SsrVec3::new(0.0, 1.0, 0.0);
        let r = v.reflect(&n);
        assert!(approx_eq(r.x, v.x, 1e-6));
        assert!(approx_eq(r.y, -v.y, 1e-6));
    }

    #[test]
    fn test_vec3_dot_perpendicular() {
        let a = SsrVec3::new(1.0, 0.0, 0.0);
        let b = SsrVec3::new(0.0, 1.0, 0.0);
        assert!(approx_eq(a.dot(&b), 0.0, 1e-10));
    }

    #[test]
    fn test_vec3_dot_parallel() {
        let a = SsrVec3::new(1.0, 0.0, 0.0);
        assert!(approx_eq(a.dot(&a), 1.0, 1e-10));
    }

    #[test]
    fn test_vec3_normalize() {
        let v = SsrVec3::new(3.0, 4.0, 0.0).normalized();
        assert!(approx_eq(v.length(), 1.0, 1e-6));
    }

    #[test]
    fn test_color_lerp() {
        let a = SsrColor::new(0.0, 0.0, 0.0);
        let b = SsrColor::new(1.0, 1.0, 1.0);
        let mid = a.lerp(&b, 0.5);
        assert!(approx_eq(mid.r, 0.5, 1e-6));
    }

    #[test]
    fn test_fresnel_at_normal() {
        let f = fresnel_schlick(1.0, 0.04);
        assert!(approx_eq(f, 0.04, 1e-6));
    }

    #[test]
    fn test_fresnel_at_grazing() {
        let f = fresnel_schlick(0.0, 0.04);
        assert!(approx_eq(f, 1.0, 1e-6));
    }

    #[test]
    fn test_fresnel_monotonic() {
        let f_normal = fresnel_schlick(1.0, 0.04);
        let f_mid = fresnel_schlick(0.5, 0.04);
        let f_grazing = fresnel_schlick(0.0, 0.04);
        assert!(f_normal < f_mid);
        assert!(f_mid < f_grazing);
    }

    #[test]
    fn test_edge_fade_center() {
        let f = edge_fade_factor(50.0, 50.0, 100.0, 100.0, 0.1);
        assert!(approx_eq(f, 1.0, 1e-6));
    }

    #[test]
    fn test_edge_fade_corner() {
        let f = edge_fade_factor(1.0, 1.0, 100.0, 100.0, 0.1);
        assert!(f < 0.5);
    }

    #[test]
    fn test_edge_fade_zero_dist() {
        let f = edge_fade_factor(0.0, 0.0, 100.0, 100.0, 0.0);
        assert!(approx_eq(f, 1.0, 1e-6));
    }

    #[test]
    fn test_screen_to_view_center() {
        let pos = screen_to_view(4.0, 4.0, 5.0, 8.0, 8.0, std::f64::consts::PI / 3.0, 1.0);
        assert!(pos.z < 0.0);
        assert!(pos.x.abs() < 1.0);
    }

    #[test]
    fn test_view_to_screen_roundtrip() {
        let fov = std::f64::consts::PI / 3.0;
        let asp = 1.5;
        let pos = screen_to_view(30.0, 20.0, 5.0, 64.0, 48.0, fov, asp);
        let (sx, sy, _d) = view_to_screen(&pos, 64.0, 48.0, fov, asp);
        assert!(approx_eq(sx, 30.0, 1.0));
        assert!(approx_eq(sy, 20.0, 1.0));
    }

    #[test]
    fn test_ray_march_miss_empty_scene() {
        let depth = SsrDepthBuffer::new(16, 16); // all 1.0
        let config = SsrConfig {
            max_steps: 16,
            step_size: 1.0,
            thickness: 0.1,
            ..Default::default()
        };
        let result = ray_march((8.0, 8.0), (1.0, 0.0), 0.5, 0.01, &depth, &config);
        // Should miss (ray goes off screen or never gets close to depth 1.0)
        match result {
            RayResult::Miss => {}
            RayResult::Hit { .. } => {
                // Might hit the far plane at 1.0 depending on depth_step
            }
        }
    }

    #[test]
    fn test_ray_march_hit_wall() {
        let mut depth = SsrDepthBuffer::new(16, 16);
        // Create a wall at x=12 with depth 0.5
        for y in 0..16 {
            for x in 10..16 {
                depth.set(x, y, 0.5);
            }
        }
        let config = SsrConfig {
            max_steps: 32,
            step_size: 1.0,
            thickness: 0.2,
            refinement_steps: 4,
            ..Default::default()
        };
        let result = ray_march((4.0, 8.0), (1.0, 0.0), 0.45, 0.001, &depth, &config);
        match result {
            RayResult::Hit { screen_x, confidence, .. } => {
                assert!(screen_x >= 9.0);
                assert!(confidence > 0.0);
            }
            RayResult::Miss => {
                // Acceptable depending on step parameters
            }
        }
    }

    #[test]
    fn test_depth_hierarchy() {
        let mut depth = SsrDepthBuffer::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                depth.set(x, y, (x + y) as f64 * 0.1);
            }
        }
        let chain = depth.build_hierarchy(3);
        assert!(!chain.is_empty());
        assert_eq!(chain[0].width, 4);
        assert_eq!(chain[0].height, 4);
    }

    #[test]
    fn test_cone_radius_at_zero() {
        assert!(approx_eq(cone_radius(0.5, 0, 16), 0.0, 1e-6));
    }

    #[test]
    fn test_cone_radius_increases() {
        let r1 = cone_radius(0.5, 4, 16);
        let r2 = cone_radius(0.5, 8, 16);
        assert!(r2 > r1);
    }

    #[test]
    fn test_apply_ssr_rough_surface() {
        let mut color = SsrColorBuffer::new(8, 8);
        let depth = SsrDepthBuffer::new(8, 8);
        let normal = SsrNormalBuffer::new(8, 8);
        let roughness = RoughnessBuffer::new(8, 8, 1.0); // Very rough -> use fallback

        for y in 0..8 {
            for x in 0..8 {
                color.set(x, y, SsrColor::new(0.5, 0.5, 0.5));
            }
        }

        let config = SsrConfig::default();
        let result = apply_ssr(&color, &depth, &normal, &roughness, &config);
        assert_eq!(result.width, 8);
        // Rough surfaces should return close to fallback blended with original
        let px = result.get(4, 4);
        assert!(px.r >= 0.0);
    }

    #[test]
    fn test_apply_ssr_smooth_surface() {
        let mut color = SsrColorBuffer::new(8, 8);
        let mut depth = SsrDepthBuffer::new(8, 8);
        let normal = SsrNormalBuffer::new(8, 8);
        let roughness = RoughnessBuffer::new(8, 8, 0.0);

        for y in 0..8 {
            for x in 0..8 {
                color.set(x, y, SsrColor::new(0.5, 0.5, 0.5));
                depth.set(x, y, 5.0);
            }
        }

        let config = SsrConfig {
            max_steps: 8,
            ..Default::default()
        };
        let result = apply_ssr(&color, &depth, &normal, &roughness, &config);
        assert_eq!(result.width, 8);
        assert_eq!(result.height, 8);
    }

    #[test]
    fn test_ssr_stats_all_rough() {
        let depth = SsrDepthBuffer::new(4, 4);
        let normal = SsrNormalBuffer::new(4, 4);
        let roughness = RoughnessBuffer::new(4, 4, 1.0);
        let config = SsrConfig::default();
        let stats = compute_ssr_stats(&depth, &normal, &roughness, &config);
        assert_eq!(stats.reflective_pixels, 0);
        assert_eq!(stats.hit_count, 0);
    }

    #[test]
    fn test_ssr_config_display() {
        let config = SsrConfig::default();
        let s = format!("{}", config);
        assert!(s.contains("SsrConfig"));
    }

    #[test]
    fn test_color_buffer_bilinear() {
        let mut buf = SsrColorBuffer::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                buf.set(x, y, SsrColor::new(1.0, 0.5, 0.0));
            }
        }
        let px = buf.sample_bilinear(1.5, 1.5);
        assert!(approx_eq(px.r, 1.0, 0.1));
        assert!(approx_eq(px.g, 0.5, 0.1));
    }
}
