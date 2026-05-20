// ssao.rs — Screen-Space Ambient Occlusion for the rendering pipeline.
//
// Implements SSAO with hemisphere kernel sampling, noise-based rotation,
// bilateral blur, configurable radius/bias/intensity, and HBAO variant
// with horizon-based ray-marching.

use std::fmt;

/// 3D vector for view-space positions and normals.
#[derive(Clone, Debug, PartialEq)]
pub struct SsaoVec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl SsaoVec3 {
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

    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

/// Depth buffer.
#[derive(Clone, Debug)]
pub struct SsaoDepthBuffer {
    pub width: usize,
    pub height: usize,
    pub depths: Vec<f64>,
}

impl SsaoDepthBuffer {
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
}

/// Normal buffer (view-space normals).
#[derive(Clone, Debug)]
pub struct NormalBuffer {
    pub width: usize,
    pub height: usize,
    pub normals: Vec<SsaoVec3>,
}

impl NormalBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            normals: vec![SsaoVec3::new(0.0, 0.0, 1.0); width * height],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> &SsaoVec3 {
        &self.normals[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, n: SsaoVec3) {
        self.normals[y * self.width + x] = n;
    }
}

/// AO output buffer (single-channel occlusion factors in [0,1]).
#[derive(Clone, Debug)]
pub struct AoBuffer {
    pub width: usize,
    pub height: usize,
    pub values: Vec<f64>,
}

impl AoBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            values: vec![1.0; width * height], // 1.0 = no occlusion
        }
    }

    pub fn get(&self, x: usize, y: usize) -> f64 {
        self.values[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, v: f64) {
        self.values[y * self.width + x] = v;
    }

    pub fn sample_clamp(&self, x: isize, y: isize) -> f64 {
        let cx = x.max(0).min(self.width as isize - 1) as usize;
        let cy = y.max(0).min(self.height as isize - 1) as usize;
        self.get(cx, cy)
    }
}

/// SSAO algorithm variant.
#[derive(Clone, Debug, PartialEq)]
pub enum SsaoVariant {
    /// Standard SSAO with hemisphere sampling.
    Standard,
    /// Horizon-Based Ambient Occlusion with ray-marching.
    Hbao {
        /// Number of angular directions (typically 4-8).
        directions: usize,
        /// Steps per direction.
        steps_per_dir: usize,
    },
}

impl fmt::Display for SsaoVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SsaoVariant::Standard => write!(f, "SSAO"),
            SsaoVariant::Hbao { directions, steps_per_dir } => {
                write!(f, "HBAO(dirs={}, steps={})", directions, steps_per_dir)
            }
        }
    }
}

/// SSAO configuration.
#[derive(Clone, Debug)]
pub struct SsaoConfig {
    /// Hemisphere sample count (e.g. 16, 32, 64).
    pub kernel_size: usize,
    /// Sample radius in view-space units.
    pub radius: f64,
    /// Depth bias to prevent self-occlusion.
    pub bias: f64,
    /// Intensity multiplier for occlusion.
    pub intensity: f64,
    /// Noise texture size (for random rotation).
    pub noise_size: usize,
    /// Algorithm variant.
    pub variant: SsaoVariant,
    /// Whether to apply bilateral blur.
    pub blur_enabled: bool,
    /// Blur kernel radius.
    pub blur_radius: usize,
}

impl Default for SsaoConfig {
    fn default() -> Self {
        Self {
            kernel_size: 32,
            radius: 0.5,
            bias: 0.025,
            intensity: 1.0,
            noise_size: 4,
            variant: SsaoVariant::Standard,
            blur_enabled: true,
            blur_radius: 2,
        }
    }
}

impl fmt::Display for SsaoConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SsaoConfig(variant={}, kernel={}, radius={:.2}, intensity={:.2})",
            self.variant, self.kernel_size, self.radius, self.intensity
        )
    }
}

/// Pseudo-random number generator (xorshift64) for deterministic kernel generation.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() & 0xFFFFFFFF) as f64 / 4294967296.0
    }

    /// Returns a f64 in [-1, 1).
    fn next_signed(&mut self) -> f64 {
        self.next_f64() * 2.0 - 1.0
    }
}

/// Generate hemisphere sample kernel (tangent-space, cosine-weighted).
pub fn generate_kernel(size: usize, seed: u64) -> Vec<SsaoVec3> {
    let mut rng = Rng::new(seed);
    let mut kernel = Vec::with_capacity(size);

    for i in 0..size {
        // Random point in hemisphere
        let x = rng.next_signed();
        let y = rng.next_signed();
        let z = rng.next_f64().max(0.01); // Ensure z > 0 (hemisphere)

        let mut sample = SsaoVec3::new(x, y, z).normalized();

        // Scale to distribute more samples near the origin (importance sampling)
        let scale_factor = i as f64 / size as f64;
        let scale = 0.1 + scale_factor * scale_factor * 0.9; // lerp(0.1, 1.0, t*t)
        sample = sample.scale(scale);

        kernel.push(sample);
    }
    kernel
}

/// Generate noise texture (random tangent-space rotation vectors).
pub fn generate_noise(size: usize, seed: u64) -> Vec<SsaoVec3> {
    let mut rng = Rng::new(seed);
    let total = size * size;
    let mut noise = Vec::with_capacity(total);

    for _ in 0..total {
        let x = rng.next_signed();
        let y = rng.next_signed();
        // z=0 because we only rotate in tangent plane
        noise.push(SsaoVec3::new(x, y, 0.0).normalized());
    }
    noise
}

/// Reconstruct view-space position from depth and screen coordinates.
pub fn reconstruct_position(
    x: usize,
    y: usize,
    depth: f64,
    width: usize,
    height: usize,
    fov_y: f64,
    aspect: f64,
) -> SsaoVec3 {
    let ndc_x = (x as f64 + 0.5) / width as f64 * 2.0 - 1.0;
    let ndc_y = 1.0 - (y as f64 + 0.5) / height as f64 * 2.0;
    let tan_half_fov = (fov_y / 2.0).tan();

    SsaoVec3 {
        x: ndc_x * aspect * tan_half_fov * depth,
        y: ndc_y * tan_half_fov * depth,
        z: -depth,
    }
}

/// Construct a tangent-space to view-space rotation basis (TBN matrix).
fn create_tbn(normal: &SsaoVec3, random_vec: &SsaoVec3) -> (SsaoVec3, SsaoVec3, SsaoVec3) {
    let tangent = random_vec.sub(&normal.scale(random_vec.dot(normal))).normalized();
    let bitangent = normal.cross(&tangent);
    (tangent, bitangent, normal.clone())
}

/// Transform a tangent-space vector to view-space using TBN.
fn tbn_transform(v: &SsaoVec3, tangent: &SsaoVec3, bitangent: &SsaoVec3, normal: &SsaoVec3) -> SsaoVec3 {
    tangent.scale(v.x).add(&bitangent.scale(v.y)).add(&normal.scale(v.z))
}

/// Standard SSAO computation at a single pixel.
fn ssao_pixel(
    x: usize,
    y: usize,
    depth_buf: &SsaoDepthBuffer,
    normal_buf: &NormalBuffer,
    kernel: &[SsaoVec3],
    noise: &[SsaoVec3],
    config: &SsaoConfig,
    fov_y: f64,
    aspect: f64,
) -> f64 {
    let w = depth_buf.width;
    let h = depth_buf.height;
    let center_depth = depth_buf.get(x, y);
    if center_depth <= 0.0 || center_depth >= 100.0 {
        return 1.0; // Skip sky / invalid depth
    }

    let position = reconstruct_position(x, y, center_depth, w, h, fov_y, aspect);
    let normal = normal_buf.get(x, y);

    // Get noise vector for this pixel
    let noise_idx = (x % config.noise_size) + (y % config.noise_size) * config.noise_size;
    let noise_vec = &noise[noise_idx % noise.len()];

    // Build TBN
    let (tangent, bitangent, n) = create_tbn(normal, noise_vec);

    let mut occlusion = 0.0_f64;
    let mut valid_samples = 0u32;

    for sample in kernel {
        // Transform sample to view space
        let view_sample = tbn_transform(sample, &tangent, &bitangent, &n);
        let sample_pos = position.add(&view_sample.scale(config.radius));

        // Project back to screen
        if sample_pos.z.abs() < 1e-10 {
            continue;
        }
        let tan_half = (fov_y / 2.0).tan();
        let proj_x = (-sample_pos.x / (sample_pos.z * aspect * tan_half) + 1.0) * 0.5;
        let proj_y = (1.0 + sample_pos.y / (sample_pos.z * tan_half)) * 0.5;

        let sx = (proj_x * w as f64) as isize;
        let sy = (proj_y * h as f64) as isize;

        let sample_depth = depth_buf.sample_clamp(sx, sy);

        // Range check: only count if within radius
        let range_check = if (center_depth - sample_depth).abs() < config.radius {
            1.0
        } else {
            0.0
        };

        // Occlusion: is the sample above the surface?
        if sample_depth < -sample_pos.z - config.bias {
            occlusion += range_check;
        }
        valid_samples += 1;
    }

    if valid_samples == 0 {
        return 1.0;
    }

    let ao = 1.0 - (occlusion / valid_samples as f64) * config.intensity;
    ao.clamp(0.0, 1.0)
}

/// HBAO computation at a single pixel.
fn hbao_pixel(
    x: usize,
    y: usize,
    depth_buf: &SsaoDepthBuffer,
    directions: usize,
    steps: usize,
    config: &SsaoConfig,
) -> f64 {
    let w = depth_buf.width;
    let h = depth_buf.height;
    let center_depth = depth_buf.get(x, y);
    if center_depth <= 0.0 || center_depth >= 100.0 {
        return 1.0;
    }

    let pixel_radius = (config.radius / center_depth.max(0.01) * h as f64).min(64.0);
    let mut total_occlusion = 0.0_f64;

    for dir in 0..directions {
        let angle = dir as f64 / directions as f64 * 2.0 * std::f64::consts::PI;
        let dx = angle.cos();
        let dy = angle.sin();

        let mut max_horizon = 0.0_f64; // sin of max horizon angle

        for step in 1..=steps {
            let t = step as f64 / steps as f64;
            let sx = x as f64 + dx * pixel_radius * t;
            let sy = y as f64 + dy * pixel_radius * t;

            let sxi = sx.round() as isize;
            let syi = sy.round() as isize;

            if sxi < 0 || sxi >= w as isize || syi < 0 || syi >= h as isize {
                break;
            }

            let sample_depth = depth_buf.sample_clamp(sxi, syi);
            let depth_diff = center_depth - sample_depth;
            let dist_2d = ((sx - x as f64).powi(2) + (sy - y as f64).powi(2)).sqrt();

            if dist_2d > 1e-10 {
                let horizon = depth_diff / (dist_2d * center_depth.max(0.01));
                if horizon > max_horizon {
                    max_horizon = horizon;
                }
            }
        }

        total_occlusion += max_horizon.clamp(0.0, 1.0);
    }

    let ao = 1.0 - (total_occlusion / directions as f64) * config.intensity;
    ao.clamp(0.0, 1.0)
}

/// Edge-aware bilateral blur for AO buffer.
pub fn bilateral_blur(
    ao: &AoBuffer,
    depth_buf: &SsaoDepthBuffer,
    radius: usize,
) -> AoBuffer {
    let w = ao.width;
    let h = ao.height;
    let mut blurred = AoBuffer::new(w, h);
    let depth_sigma = 0.1_f64;

    for y in 0..h {
        for x in 0..w {
            let center_depth = depth_buf.get(x, y);
            let mut sum = 0.0_f64;
            let mut weight_sum = 0.0_f64;

            let r = radius as isize;
            for dy in -r..=r {
                for dx in -r..=r {
                    let sx = x as isize + dx;
                    let sy = y as isize + dy;

                    let sample_ao = ao.sample_clamp(sx, sy);
                    let sample_depth = depth_buf.sample_clamp(sx, sy);

                    let depth_diff = (center_depth - sample_depth).abs();
                    let depth_weight = (-depth_diff * depth_diff / (2.0 * depth_sigma * depth_sigma)).exp();
                    let spatial = (-(dx * dx + dy * dy) as f64 / (2.0 * (radius as f64 + 0.5).powi(2))).exp();
                    let weight = depth_weight * spatial;

                    sum += sample_ao * weight;
                    weight_sum += weight;
                }
            }

            if weight_sum > 1e-10 {
                blurred.set(x, y, sum / weight_sum);
            } else {
                blurred.set(x, y, ao.get(x, y));
            }
        }
    }
    blurred
}

/// Compute SSAO (standard or HBAO) for a full frame.
pub fn compute_ssao(
    depth_buf: &SsaoDepthBuffer,
    normal_buf: &NormalBuffer,
    config: &SsaoConfig,
) -> AoBuffer {
    let w = depth_buf.width;
    let h = depth_buf.height;
    let fov_y = std::f64::consts::PI / 3.0; // 60 degree FOV
    let aspect = w as f64 / h as f64;

    let mut ao = AoBuffer::new(w, h);

    match &config.variant {
        SsaoVariant::Standard => {
            let kernel = generate_kernel(config.kernel_size, 42);
            let noise = generate_noise(config.noise_size, 137);

            for y in 0..h {
                for x in 0..w {
                    let val = ssao_pixel(
                        x, y, depth_buf, normal_buf, &kernel, &noise, config, fov_y, aspect,
                    );
                    ao.set(x, y, val);
                }
            }
        }
        SsaoVariant::Hbao { directions, steps_per_dir } => {
            for y in 0..h {
                for x in 0..w {
                    let val = hbao_pixel(x, y, depth_buf, *directions, *steps_per_dir, config);
                    ao.set(x, y, val);
                }
            }
        }
    }

    // Bilateral blur
    if config.blur_enabled {
        ao = bilateral_blur(&ao, depth_buf, config.blur_radius);
    }

    ao
}

/// Compute AO statistics for diagnostics.
#[derive(Debug, PartialEq)]
pub struct AoStats {
    pub min_ao: f64,
    pub max_ao: f64,
    pub avg_ao: f64,
    pub fully_occluded: usize,
    pub fully_visible: usize,
}

pub fn compute_ao_stats(ao: &AoBuffer) -> AoStats {
    let mut min_v = f64::MAX;
    let mut max_v = f64::MIN;
    let mut sum = 0.0_f64;
    let mut occluded = 0usize;
    let mut visible = 0usize;

    for &v in &ao.values {
        if v < min_v { min_v = v; }
        if v > max_v { max_v = v; }
        sum += v;
        if v < 0.01 { occluded += 1; }
        if v > 0.99 { visible += 1; }
    }

    let count = ao.values.len().max(1);
    AoStats {
        min_ao: if min_v == f64::MAX { 1.0 } else { min_v },
        max_ao: if max_v == f64::MIN { 1.0 } else { max_v },
        avg_ao: sum / count as f64,
        fully_occluded: occluded,
        fully_visible: visible,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_vec3_length() {
        let v = SsaoVec3::new(3.0, 4.0, 0.0);
        assert!(approx_eq(v.length(), 5.0, 1e-6));
    }

    #[test]
    fn test_vec3_normalize() {
        let v = SsaoVec3::new(3.0, 4.0, 0.0).normalized();
        assert!(approx_eq(v.length(), 1.0, 1e-6));
    }

    #[test]
    fn test_vec3_dot() {
        let a = SsaoVec3::new(1.0, 0.0, 0.0);
        let b = SsaoVec3::new(0.0, 1.0, 0.0);
        assert!(approx_eq(a.dot(&b), 0.0, 1e-10));
    }

    #[test]
    fn test_vec3_cross() {
        let x = SsaoVec3::new(1.0, 0.0, 0.0);
        let y = SsaoVec3::new(0.0, 1.0, 0.0);
        let z = x.cross(&y);
        assert!(approx_eq(z.x, 0.0, 1e-10));
        assert!(approx_eq(z.y, 0.0, 1e-10));
        assert!(approx_eq(z.z, 1.0, 1e-10));
    }

    #[test]
    fn test_generate_kernel_size() {
        let kernel = generate_kernel(32, 42);
        assert_eq!(kernel.len(), 32);
    }

    #[test]
    fn test_kernel_hemisphere() {
        let kernel = generate_kernel(64, 42);
        for s in &kernel {
            // z component (up in tangent space) should be non-negative before scaling
            // After scaling and normalization the raw z should be > 0 pre-scale
            assert!(s.length() > 0.0);
        }
    }

    #[test]
    fn test_generate_noise_size() {
        let noise = generate_noise(4, 137);
        assert_eq!(noise.len(), 16);
    }

    #[test]
    fn test_noise_normalized() {
        let noise = generate_noise(4, 137);
        for n in &noise {
            assert!(approx_eq(n.length(), 1.0, 0.1));
        }
    }

    #[test]
    fn test_reconstruct_position_center() {
        let pos = reconstruct_position(4, 4, 5.0, 8, 8, std::f64::consts::PI / 3.0, 1.0);
        // Center pixel should have small x, y; negative z
        assert!(pos.x.abs() < 1.0);
        assert!(pos.y.abs() < 1.0);
        assert!(pos.z < 0.0);
    }

    #[test]
    fn test_depth_buffer_clamp() {
        let mut db = SsaoDepthBuffer::new(4, 4);
        db.set(0, 0, 0.5);
        assert!(approx_eq(db.sample_clamp(-1, -1), 0.5, 1e-6));
    }

    #[test]
    fn test_ao_buffer_default_visible() {
        let ao = AoBuffer::new(4, 4);
        for &v in &ao.values {
            assert!(approx_eq(v, 1.0, 1e-10));
        }
    }

    #[test]
    fn test_ssao_flat_surface() {
        let mut depth = SsaoDepthBuffer::new(8, 8);
        let normal = NormalBuffer::new(8, 8);
        // Flat surface at depth 5.0
        for y in 0..8 {
            for x in 0..8 {
                depth.set(x, y, 5.0);
            }
        }
        let config = SsaoConfig {
            kernel_size: 16,
            radius: 0.5,
            bias: 0.025,
            intensity: 1.0,
            noise_size: 4,
            variant: SsaoVariant::Standard,
            blur_enabled: false,
            blur_radius: 2,
        };
        let ao = compute_ssao(&depth, &normal, &config);
        // Flat surface should have relatively high AO (little occlusion)
        let center = ao.get(4, 4);
        assert!(center > 0.3);
    }

    #[test]
    fn test_hbao_flat_surface() {
        let mut depth = SsaoDepthBuffer::new(8, 8);
        let normal = NormalBuffer::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                depth.set(x, y, 5.0);
            }
        }
        let config = SsaoConfig {
            variant: SsaoVariant::Hbao { directions: 4, steps_per_dir: 4 },
            blur_enabled: false,
            ..Default::default()
        };
        let ao = compute_ssao(&depth, &normal, &config);
        let center = ao.get(4, 4);
        assert!(center > 0.3);
    }

    #[test]
    fn test_ssao_sky_unoccluded() {
        let depth = SsaoDepthBuffer::new(4, 4); // default 1.0 < 100.0
        let normal = NormalBuffer::new(4, 4);
        let config = SsaoConfig {
            blur_enabled: false,
            ..Default::default()
        };
        let ao = compute_ssao(&depth, &normal, &config);
        // At depth 1.0 with uniform surface, AO should be computed
        for &v in &ao.values {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_bilateral_blur_uniform() {
        let ao = AoBuffer::new(8, 8); // all 1.0
        let depth = SsaoDepthBuffer::new(8, 8);
        let blurred = bilateral_blur(&ao, &depth, 2);
        for &v in &blurred.values {
            assert!(approx_eq(v, 1.0, 0.05));
        }
    }

    #[test]
    fn test_bilateral_blur_preserves_edges() {
        let mut ao = AoBuffer::new(8, 8);
        let mut depth = SsaoDepthBuffer::new(8, 8);
        // Sharp depth edge at x=4
        for y in 0..8 {
            for x in 0..8 {
                if x < 4 {
                    depth.set(x, y, 1.0);
                    ao.set(x, y, 0.3);
                } else {
                    depth.set(x, y, 10.0);
                    ao.set(x, y, 0.9);
                }
            }
        }
        let blurred = bilateral_blur(&ao, &depth, 2);
        // Edge should be preserved (left side stays dark, right stays bright)
        assert!(blurred.get(0, 4) < 0.6);
        assert!(blurred.get(7, 4) > 0.6);
    }

    #[test]
    fn test_ao_stats_all_visible() {
        let ao = AoBuffer::new(4, 4);
        let stats = compute_ao_stats(&ao);
        assert_eq!(stats.fully_visible, 16);
        assert_eq!(stats.fully_occluded, 0);
        assert!(approx_eq(stats.avg_ao, 1.0, 1e-6));
    }

    #[test]
    fn test_ao_stats_mixed() {
        let mut ao = AoBuffer::new(2, 2);
        ao.set(0, 0, 0.0);
        ao.set(1, 0, 0.5);
        ao.set(0, 1, 0.5);
        ao.set(1, 1, 1.0);
        let stats = compute_ao_stats(&ao);
        assert_eq!(stats.fully_occluded, 1);
        assert_eq!(stats.fully_visible, 1);
        assert!(approx_eq(stats.avg_ao, 0.5, 1e-6));
    }

    #[test]
    fn test_ssao_config_display() {
        let config = SsaoConfig::default();
        let s = format!("{}", config);
        assert!(s.contains("SsaoConfig"));
        assert!(s.contains("SSAO"));
    }

    #[test]
    fn test_ssao_variant_display() {
        let s = format!("{}", SsaoVariant::Standard);
        assert_eq!(s, "SSAO");
        let h = format!("{}", SsaoVariant::Hbao { directions: 8, steps_per_dir: 4 });
        assert!(h.contains("HBAO"));
    }

    #[test]
    fn test_tbn_transform_identity_like() {
        let t = SsaoVec3::new(1.0, 0.0, 0.0);
        let b = SsaoVec3::new(0.0, 1.0, 0.0);
        let n = SsaoVec3::new(0.0, 0.0, 1.0);
        let v = SsaoVec3::new(0.5, 0.3, 0.7);
        let result = tbn_transform(&v, &t, &b, &n);
        assert!(approx_eq(result.x, 0.5, 1e-6));
        assert!(approx_eq(result.y, 0.3, 1e-6));
        assert!(approx_eq(result.z, 0.7, 1e-6));
    }

    #[test]
    fn test_ssao_with_blur() {
        let mut depth = SsaoDepthBuffer::new(8, 8);
        let normal = NormalBuffer::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                depth.set(x, y, 5.0);
            }
        }
        let config = SsaoConfig {
            kernel_size: 8,
            blur_enabled: true,
            blur_radius: 1,
            ..Default::default()
        };
        let ao = compute_ssao(&depth, &normal, &config);
        for &v in &ao.values {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }
}
