// Denoiser for noisy path-traced images.
// Edge-aware bilateral filter, A-trous wavelet filter, temporal accumulation.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0, z: 0.0 } }
    pub fn dot(self, o: Self) -> f64 { self.x * o.x + self.y * o.y + self.z * o.z }
    pub fn length_squared(self) -> f64 { self.dot(self) }
    pub fn length(self) -> f64 { self.length_squared().sqrt() }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, r: Self) -> Self { Self { x: self.x + r.x, y: self.y + r.y, z: self.z + r.z } }
}
impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, r: Self) -> Self { Self { x: self.x - r.x, y: self.y - r.y, z: self.z - r.z } }
}
impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s } }
}

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4}, {:.4})", self.x, self.y, self.z)
    }
}

/// RGB color for image processing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl Color {
    pub fn new(r: f64, g: f64, b: f64) -> Self { Self { r, g, b } }
    pub fn black() -> Self { Self { r: 0.0, g: 0.0, b: 0.0 } }
    pub fn dist_sq(self, other: Self) -> f64 {
        let dr = self.r - other.r;
        let dg = self.g - other.g;
        let db = self.b - other.b;
        dr * dr + dg * dg + db * db
    }
}

impl std::ops::Add for Color {
    type Output = Self;
    fn add(self, r: Self) -> Self { Self { r: self.r + r.r, g: self.g + r.g, b: self.b + r.b } }
}
impl std::ops::Mul<f64> for Color {
    type Output = Self;
    fn mul(self, s: f64) -> Self { Self { r: self.r * s, g: self.g * s, b: self.b * s } }
}

/// Feature buffers for edge-stopping denoising.
#[derive(Debug, Clone)]
pub struct FeatureBuffers {
    pub width: usize,
    pub height: usize,
    pub normals: Vec<Vec3>,
    pub depth: Vec<f64>,
}

impl FeatureBuffers {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            normals: vec![Vec3::new(0.0, 0.0, 1.0); width * height],
            depth: vec![0.0; width * height],
        }
    }

    pub fn set_normal(&mut self, x: usize, y: usize, n: Vec3) {
        let idx = y * self.width + x;
        if idx < self.normals.len() { self.normals[idx] = n; }
    }

    pub fn set_depth(&mut self, x: usize, y: usize, d: f64) {
        let idx = y * self.width + x;
        if idx < self.depth.len() { self.depth[idx] = d; }
    }

    pub fn get_normal(&self, x: usize, y: usize) -> Vec3 {
        let idx = y * self.width + x;
        if idx < self.normals.len() { self.normals[idx] } else { Vec3::zero() }
    }

    pub fn get_depth(&self, x: usize, y: usize) -> f64 {
        let idx = y * self.width + x;
        if idx < self.depth.len() { self.depth[idx] } else { 0.0 }
    }
}

/// Denoiser configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct DenoiserConfig {
    pub sigma_color: f64,
    pub sigma_normal: f64,
    pub sigma_depth: f64,
    pub sigma_spatial: f64,
    pub kernel_radius: usize,
}

impl DenoiserConfig {
    pub fn default_config() -> Self {
        Self {
            sigma_color: 0.1,
            sigma_normal: 0.1,
            sigma_depth: 0.5,
            sigma_spatial: 2.0,
            kernel_radius: 3,
        }
    }
}

/// 2D image buffer.
#[derive(Debug, Clone)]
pub struct ImageBuffer {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<Color>,
}

impl ImageBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height, pixels: vec![Color::black(); width * height] }
    }

    pub fn from_pixels(width: usize, height: usize, pixels: Vec<Color>) -> Self {
        assert_eq!(pixels.len(), width * height);
        Self { width, height, pixels }
    }

    pub fn get(&self, x: usize, y: usize) -> Color {
        let idx = y * self.width + x;
        if idx < self.pixels.len() { self.pixels[idx] } else { Color::black() }
    }

    pub fn set(&mut self, x: usize, y: usize, c: Color) {
        let idx = y * self.width + x;
        if idx < self.pixels.len() { self.pixels[idx] = c; }
    }

    pub fn pixel_count(&self) -> usize { self.width * self.height }
}

/// Gaussian weight function.
fn gaussian_weight(dist_sq: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 { return 0.0; }
    (-dist_sq / (2.0 * sigma * sigma)).exp()
}

/// Edge-aware bilateral filter.
pub fn bilateral_filter(
    input: &ImageBuffer,
    features: &FeatureBuffers,
    config: &DenoiserConfig,
) -> ImageBuffer {
    let w = input.width;
    let h = input.height;
    let mut output = ImageBuffer::new(w, h);
    let r = config.kernel_radius as i64;

    for y in 0..h {
        for x in 0..w {
            let center_color = input.get(x, y);
            let center_normal = features.get_normal(x, y);
            let center_depth = features.get_depth(x, y);

            let mut sum = Color::black();
            let mut weight_sum = 0.0;

            for dy in -r..=r {
                for dx in -r..=r {
                    let nx = x as i64 + dx;
                    let ny = y as i64 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let nx = nx as usize;
                    let ny = ny as usize;

                    let neighbor_color = input.get(nx, ny);
                    let neighbor_normal = features.get_normal(nx, ny);
                    let neighbor_depth = features.get_depth(nx, ny);

                    // Spatial weight
                    let spatial_dist_sq = (dx * dx + dy * dy) as f64;
                    let w_spatial = gaussian_weight(spatial_dist_sq, config.sigma_spatial);

                    // Color range weight
                    let color_dist_sq = center_color.dist_sq(neighbor_color);
                    let w_color = gaussian_weight(color_dist_sq, config.sigma_color);

                    // Normal weight
                    let normal_diff = center_normal - neighbor_normal;
                    let normal_dist_sq = normal_diff.length_squared();
                    let w_normal = gaussian_weight(normal_dist_sq, config.sigma_normal);

                    // Depth weight
                    let depth_diff = center_depth - neighbor_depth;
                    let w_depth = gaussian_weight(depth_diff * depth_diff, config.sigma_depth);

                    let w = w_spatial * w_color * w_normal * w_depth;
                    sum = sum + neighbor_color * w;
                    weight_sum += w;
                }
            }

            if weight_sum > 1e-15 {
                output.set(x, y, sum * (1.0 / weight_sum));
            } else {
                output.set(x, y, center_color);
            }
        }
    }
    output
}

/// A-trous wavelet filter weights (5-tap B3 spline).
const ATROUS_KERNEL: [f64; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

/// A-trous wavelet filter (single iteration with given step size).
pub fn atrous_filter_pass(
    input: &ImageBuffer,
    features: &FeatureBuffers,
    config: &DenoiserConfig,
    step_size: usize,
) -> ImageBuffer {
    let w = input.width;
    let h = input.height;
    let mut output = ImageBuffer::new(w, h);
    let step = step_size as i64;

    for y in 0..h {
        for x in 0..w {
            let center_color = input.get(x, y);
            let center_normal = features.get_normal(x, y);
            let center_depth = features.get_depth(x, y);

            let mut sum = Color::black();
            let mut weight_sum = 0.0;

            for ky in 0..5i64 {
                for kx in 0..5i64 {
                    let ox = (kx - 2) * step;
                    let oy = (ky - 2) * step;
                    let nx = x as i64 + ox;
                    let ny = y as i64 + oy;

                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let nx = nx as usize;
                    let ny = ny as usize;

                    let neighbor_color = input.get(nx, ny);
                    let neighbor_normal = features.get_normal(nx, ny);
                    let neighbor_depth = features.get_depth(nx, ny);

                    let kernel_w = ATROUS_KERNEL[kx as usize] * ATROUS_KERNEL[ky as usize];

                    // Edge-stopping weights
                    let color_dist_sq = center_color.dist_sq(neighbor_color);
                    let w_color = gaussian_weight(color_dist_sq, config.sigma_color);

                    let normal_diff = center_normal - neighbor_normal;
                    let w_normal = gaussian_weight(normal_diff.length_squared(), config.sigma_normal);

                    let depth_diff = center_depth - neighbor_depth;
                    let w_depth = gaussian_weight(depth_diff * depth_diff, config.sigma_depth);

                    let total_w = kernel_w * w_color * w_normal * w_depth;
                    sum = sum + neighbor_color * total_w;
                    weight_sum += total_w;
                }
            }

            if weight_sum > 1e-15 {
                output.set(x, y, sum * (1.0 / weight_sum));
            } else {
                output.set(x, y, center_color);
            }
        }
    }
    output
}

/// Full A-trous wavelet denoising (5 iterations with doubling step size).
pub fn atrous_denoise(
    input: &ImageBuffer,
    features: &FeatureBuffers,
    config: &DenoiserConfig,
    iterations: usize,
) -> ImageBuffer {
    let mut current = input.clone();
    for i in 0..iterations {
        let step = 1 << i; // 1, 2, 4, 8, 16
        current = atrous_filter_pass(&current, features, config, step);
    }
    current
}

/// 2D motion vector for temporal accumulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionVector {
    pub dx: f64,
    pub dy: f64,
}

impl MotionVector {
    pub fn zero() -> Self { Self { dx: 0.0, dy: 0.0 } }
    pub fn new(dx: f64, dy: f64) -> Self { Self { dx, dy } }
}

/// Temporal accumulation: blend current denoised frame with previous using motion vectors.
pub fn temporal_accumulate(
    current: &ImageBuffer,
    previous: &ImageBuffer,
    motion_vectors: &[MotionVector],
    blend_factor: f64,
) -> ImageBuffer {
    let w = current.width;
    let h = current.height;
    let mut output = ImageBuffer::new(w, h);
    let alpha = blend_factor.clamp(0.0, 1.0);

    for y in 0..h {
        for x in 0..w {
            let mv_idx = y * w + x;
            let current_color = current.get(x, y);

            if mv_idx >= motion_vectors.len() {
                // No motion vector available for this pixel — use current frame only
                output.set(x, y, current_color);
                continue;
            }

            let mv = motion_vectors[mv_idx];

            let prev_x = (x as f64 + mv.dx).round() as i64;
            let prev_y = (y as f64 + mv.dy).round() as i64;

            if prev_x >= 0 && prev_y >= 0 && prev_x < w as i64 && prev_y < h as i64 {
                let prev_color = previous.get(prev_x as usize, prev_y as usize);
                // Reject if too different (ghosting prevention)
                let diff = current_color.dist_sq(prev_color);
                let threshold = 2.0;
                if diff <= threshold {
                    let blended = current_color * (1.0 - alpha) + prev_color * alpha;
                    output.set(x, y, blended);
                } else {
                    output.set(x, y, current_color);
                }
            } else {
                output.set(x, y, current_color);
            }
        }
    }
    output
}

/// Compute simple MSE (mean squared error) between two images.
pub fn compute_mse(a: &ImageBuffer, b: &ImageBuffer) -> f64 {
    assert_eq!(a.pixel_count(), b.pixel_count());
    if a.pixel_count() == 0 { return 0.0; }
    let mut sum = 0.0;
    for i in 0..a.pixels.len() {
        sum += a.pixels[i].dist_sq(b.pixels[i]);
    }
    sum / a.pixel_count() as f64
}

/// Compute PSNR (peak signal-to-noise ratio) between two images.
pub fn compute_psnr(a: &ImageBuffer, b: &ImageBuffer) -> f64 {
    let mse = compute_mse(a, b);
    if mse < 1e-15 { return f64::INFINITY; }
    10.0 * (1.0 / mse).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool { (a - b).abs() < eps }

    fn color_approx(a: Color, b: Color, eps: f64) -> bool {
        approx_eq(a.r, b.r, eps) && approx_eq(a.g, b.g, eps) && approx_eq(a.b, b.b, eps)
    }

    fn uniform_image(w: usize, h: usize, c: Color) -> ImageBuffer {
        ImageBuffer::from_pixels(w, h, vec![c; w * h])
    }

    fn uniform_features(w: usize, h: usize) -> FeatureBuffers {
        FeatureBuffers::new(w, h)
    }

    #[test]
    fn test_image_buffer_create() {
        let img = ImageBuffer::new(4, 4);
        assert_eq!(img.pixel_count(), 16);
        assert!(color_approx(img.get(0, 0), Color::black(), 1e-9));
    }

    #[test]
    fn test_image_buffer_set_get() {
        let mut img = ImageBuffer::new(4, 4);
        img.set(2, 3, Color::new(0.5, 0.6, 0.7));
        let c = img.get(2, 3);
        assert!(approx_eq(c.r, 0.5, 1e-9));
        assert!(approx_eq(c.g, 0.6, 1e-9));
    }

    #[test]
    fn test_color_dist_sq() {
        let a = Color::new(1.0, 0.0, 0.0);
        let b = Color::new(0.0, 1.0, 0.0);
        assert!(approx_eq(a.dist_sq(b), 2.0, 1e-9));
    }

    #[test]
    fn test_gaussian_weight_zero_dist() {
        let w = gaussian_weight(0.0, 1.0);
        assert!(approx_eq(w, 1.0, 1e-9));
    }

    #[test]
    fn test_gaussian_weight_large_dist() {
        let w = gaussian_weight(100.0, 0.1);
        assert!(w < 1e-10);
    }

    #[test]
    fn test_bilateral_uniform_image() {
        let img = uniform_image(8, 8, Color::new(0.5, 0.5, 0.5));
        let features = uniform_features(8, 8);
        let config = DenoiserConfig::default_config();
        let out = bilateral_filter(&img, &features, &config);
        // Uniform image should remain unchanged
        for y in 0..8 {
            for x in 0..8 {
                assert!(color_approx(out.get(x, y), Color::new(0.5, 0.5, 0.5), 1e-6));
            }
        }
    }

    #[test]
    fn test_bilateral_preserves_edge() {
        let mut img = ImageBuffer::new(8, 8);
        let mut features = FeatureBuffers::new(8, 8);
        // Left half white, right half black
        for y in 0..8 {
            for x in 0..4 {
                img.set(x, y, Color::new(1.0, 1.0, 1.0));
                features.set_normal(x, y, Vec3::new(0.0, 0.0, 1.0));
            }
            for x in 4..8 {
                img.set(x, y, Color::black());
                features.set_normal(x, y, Vec3::new(1.0, 0.0, 0.0)); // different normal
            }
        }
        let config = DenoiserConfig { sigma_color: 0.01, sigma_normal: 0.01, sigma_depth: 0.5, sigma_spatial: 2.0, kernel_radius: 2 };
        let out = bilateral_filter(&img, &features, &config);
        // Interior pixel should stay close to original
        let left = out.get(1, 4);
        let right = out.get(6, 4);
        assert!(left.r > 0.8); // still mostly white
        assert!(right.r < 0.2); // still mostly black
    }

    #[test]
    fn test_atrous_uniform() {
        let img = uniform_image(8, 8, Color::new(0.3, 0.4, 0.5));
        let features = uniform_features(8, 8);
        let config = DenoiserConfig::default_config();
        let out = atrous_filter_pass(&img, &features, &config, 1);
        // Center pixels should be unchanged for uniform image
        assert!(color_approx(out.get(4, 4), Color::new(0.3, 0.4, 0.5), 1e-6));
    }

    #[test]
    fn test_atrous_denoise_reduces_noise() {
        // Create noisy image: base 0.5 with some noise
        let mut img = ImageBuffer::new(16, 16);
        let ground_truth = uniform_image(16, 16, Color::new(0.5, 0.5, 0.5));
        let mut seed = 42u64;
        for y in 0..16 {
            for x in 0..16 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise = ((seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.4;
                img.set(x, y, Color::new(0.5 + noise, 0.5 + noise, 0.5 + noise));
            }
        }
        let features = uniform_features(16, 16);
        let config = DenoiserConfig { sigma_color: 0.3, sigma_normal: 1.0, sigma_depth: 1.0, sigma_spatial: 2.0, kernel_radius: 3 };

        let mse_before = compute_mse(&img, &ground_truth);
        let denoised = atrous_denoise(&img, &features, &config, 3);
        let mse_after = compute_mse(&denoised, &ground_truth);
        assert!(mse_after < mse_before, "denoised MSE {} should be less than noisy MSE {}", mse_after, mse_before);
    }

    #[test]
    fn test_temporal_accumulate_static() {
        let current = uniform_image(4, 4, Color::new(1.0, 0.0, 0.0));
        let previous = uniform_image(4, 4, Color::new(0.0, 0.0, 1.0));
        let mvs = vec![MotionVector::zero(); 16];
        let out = temporal_accumulate(&current, &previous, &mvs, 0.5);
        let c = out.get(2, 2);
        // Should be blend of red and blue
        assert!(approx_eq(c.r, 0.5, 1e-6));
        assert!(approx_eq(c.b, 0.5, 1e-6));
    }

    #[test]
    fn test_temporal_rejects_ghosting() {
        let current = uniform_image(4, 4, Color::new(1.0, 1.0, 1.0));
        let previous = uniform_image(4, 4, Color::black()); // very different
        let mvs = vec![MotionVector::zero(); 16];
        let out = temporal_accumulate(&current, &previous, &mvs, 0.8);
        let c = out.get(2, 2);
        // Difference is large -> should reject and use current
        assert!(approx_eq(c.r, 1.0, 1e-6));
    }

    #[test]
    fn test_temporal_with_motion() {
        let mut current = ImageBuffer::new(4, 4);
        let mut previous = ImageBuffer::new(4, 4);
        current.set(2, 2, Color::new(0.5, 0.5, 0.5));
        previous.set(1, 1, Color::new(0.6, 0.6, 0.6)); // shifted by (-1,-1)
        let mut mvs = vec![MotionVector::zero(); 16];
        mvs[2 * 4 + 2] = MotionVector::new(-1.0, -1.0);
        let out = temporal_accumulate(&current, &previous, &mvs, 0.5);
        let c = out.get(2, 2);
        // Should blend current (0.5) with previous at (1,1) which is (0.6)
        assert!(approx_eq(c.r, 0.55, 1e-6));
    }

    #[test]
    fn test_compute_mse_identical() {
        let a = uniform_image(4, 4, Color::new(0.5, 0.5, 0.5));
        let b = uniform_image(4, 4, Color::new(0.5, 0.5, 0.5));
        assert!(approx_eq(compute_mse(&a, &b), 0.0, 1e-12));
    }

    #[test]
    fn test_compute_mse_different() {
        let a = uniform_image(2, 2, Color::new(1.0, 0.0, 0.0));
        let b = uniform_image(2, 2, Color::black());
        let mse = compute_mse(&a, &b);
        // each pixel: dist_sq = 1.0, 4 pixels, mse = 1.0
        assert!(approx_eq(mse, 1.0, 1e-9));
    }

    #[test]
    fn test_compute_psnr_identical() {
        let a = uniform_image(4, 4, Color::new(0.5, 0.5, 0.5));
        assert_eq!(compute_psnr(&a, &a), f64::INFINITY);
    }

    #[test]
    fn test_compute_psnr_different() {
        let a = uniform_image(4, 4, Color::new(1.0, 1.0, 1.0));
        let b = uniform_image(4, 4, Color::new(0.9, 0.9, 0.9));
        let psnr = compute_psnr(&a, &b);
        assert!(psnr > 0.0 && psnr.is_finite());
    }

    #[test]
    fn test_feature_buffers() {
        let mut fb = FeatureBuffers::new(4, 4);
        fb.set_normal(1, 2, Vec3::new(1.0, 0.0, 0.0));
        fb.set_depth(1, 2, 5.5);
        let n = fb.get_normal(1, 2);
        assert!(approx_eq(n.x, 1.0, 1e-9));
        assert!(approx_eq(fb.get_depth(1, 2), 5.5, 1e-9));
    }

    #[test]
    fn test_denoiser_config_default() {
        let config = DenoiserConfig::default_config();
        assert!(config.sigma_color > 0.0);
        assert!(config.sigma_spatial > 0.0);
        assert!(config.kernel_radius > 0);
    }

    #[test]
    fn test_atrous_step_sizes() {
        let img = uniform_image(16, 16, Color::new(0.5, 0.5, 0.5));
        let features = uniform_features(16, 16);
        let config = DenoiserConfig::default_config();
        // Different step sizes should all work
        for step in [1, 2, 4, 8] {
            let out = atrous_filter_pass(&img, &features, &config, step);
            assert_eq!(out.pixel_count(), 256);
        }
    }

    #[test]
    fn test_motion_vector_creation() {
        let mv = MotionVector::new(1.5, -2.3);
        assert!(approx_eq(mv.dx, 1.5, 1e-9));
        assert!(approx_eq(mv.dy, -2.3, 1e-9));
    }

    #[test]
    fn test_bilateral_1x1() {
        let img = ImageBuffer::from_pixels(1, 1, vec![Color::new(0.3, 0.4, 0.5)]);
        let features = FeatureBuffers::new(1, 1);
        let config = DenoiserConfig::default_config();
        let out = bilateral_filter(&img, &features, &config);
        assert!(color_approx(out.get(0, 0), Color::new(0.3, 0.4, 0.5), 1e-6));
    }

    #[test]
    fn test_temporal_no_motion_vectors() {
        let current = uniform_image(4, 4, Color::new(0.5, 0.5, 0.5));
        let previous = uniform_image(4, 4, Color::new(0.4, 0.4, 0.4));
        let out = temporal_accumulate(&current, &previous, &[], 0.5);
        // No MVs -> use current
        let c = out.get(0, 0);
        assert!(approx_eq(c.r, 0.5, 1e-6));
    }

    #[test]
    fn test_gaussian_zero_sigma() {
        let w = gaussian_weight(1.0, 0.0);
        assert!(approx_eq(w, 0.0, 1e-9));
    }
}
