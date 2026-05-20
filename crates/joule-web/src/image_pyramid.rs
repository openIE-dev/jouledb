//! # Image Pyramid
//!
//! Multi-resolution image representation using Gaussian and Laplacian pyramids.
//! Supports scale-space construction, octave-based processing, and pyramid
//! blending for seamless image compositing.

use std::fmt;

// ── Core Types ──

/// A single level in the pyramid containing pixel data and metadata.
#[derive(Clone, Debug)]
pub struct PyramidLevel {
    pub data: Vec<f64>,
    pub width: usize,
    pub height: usize,
    pub scale: f64,
    pub octave: usize,
}

impl PyramidLevel {
    pub fn new(width: usize, height: usize, scale: f64, octave: usize) -> Self {
        Self {
            data: vec![0.0; width * height],
            width,
            height,
            scale,
            octave,
        }
    }

    pub fn from_data(data: Vec<f64>, width: usize, height: usize, scale: f64, octave: usize) -> Self {
        assert_eq!(data.len(), width * height);
        Self { data, width, height, scale, octave }
    }

    pub fn get(&self, x: usize, y: usize) -> f64 {
        if x < self.width && y < self.height {
            self.data[y * self.width + x]
        } else {
            0.0
        }
    }

    pub fn set(&mut self, x: usize, y: usize, val: f64) {
        if x < self.width && y < self.height {
            self.data[y * self.width + x] = val;
        }
    }

    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }
}

impl fmt::Display for PyramidLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Level({}x{}, scale={:.3}, oct={})", self.width, self.height, self.scale, self.octave)
    }
}

// ── Gaussian Kernel ──

/// Generate a 1D Gaussian kernel.
fn gaussian_kernel_1d(sigma: f64, radius: usize) -> Vec<f64> {
    let size = 2 * radius + 1;
    let mut kernel = Vec::with_capacity(size);
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut sum = 0.0;
    for i in 0..size {
        let x = i as f64 - radius as f64;
        let val = (-x * x / two_sigma_sq).exp();
        kernel.push(val);
        sum += val;
    }
    for v in &mut kernel {
        *v /= sum;
    }
    kernel
}

/// Apply separable Gaussian blur to a level.
fn gaussian_blur(level: &PyramidLevel, sigma: f64) -> PyramidLevel {
    let radius = (sigma * 3.0).ceil() as usize;
    if radius == 0 {
        return level.clone();
    }
    let kernel = gaussian_kernel_1d(sigma, radius);
    let w = level.width;
    let h = level.height;

    // Horizontal pass
    let mut temp = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for (ki, kv) in kernel.iter().enumerate() {
                let sx = x as isize + ki as isize - radius as isize;
                let sx = sx.max(0).min(w as isize - 1) as usize;
                sum += level.data[y * w + sx] * kv;
            }
            temp[y * w + x] = sum;
        }
    }

    // Vertical pass
    let mut result = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for (ki, kv) in kernel.iter().enumerate() {
                let sy = y as isize + ki as isize - radius as isize;
                let sy = sy.max(0).min(h as isize - 1) as usize;
                sum += temp[sy * w + x] * kv;
            }
            result[y * w + x] = sum;
        }
    }

    PyramidLevel::from_data(result, w, h, level.scale, level.octave)
}

// ── Downsample / Upsample ──

/// Downsample a level by factor of 2 using nearest-neighbor averaging.
fn downsample(level: &PyramidLevel) -> PyramidLevel {
    let nw = (level.width + 1) / 2;
    let nh = (level.height + 1) / 2;
    let mut data = vec![0.0; nw * nh];
    for y in 0..nh {
        for x in 0..nw {
            let sx = (x * 2).min(level.width - 1);
            let sy = (y * 2).min(level.height - 1);
            data[y * nw + x] = level.data[sy * level.width + sx];
        }
    }
    PyramidLevel::from_data(data, nw, nh, level.scale * 2.0, level.octave + 1)
}

/// Upsample a level by factor of 2 using bilinear interpolation.
fn upsample(level: &PyramidLevel, target_w: usize, target_h: usize) -> PyramidLevel {
    let mut data = vec![0.0; target_w * target_h];
    for y in 0..target_h {
        for x in 0..target_w {
            let sx = x as f64 * level.width as f64 / target_w as f64;
            let sy = y as f64 * level.height as f64 / target_h as f64;
            let x0 = (sx.floor() as usize).min(level.width.saturating_sub(1));
            let y0 = (sy.floor() as usize).min(level.height.saturating_sub(1));
            let x1 = (x0 + 1).min(level.width - 1);
            let y1 = (y0 + 1).min(level.height - 1);
            let fx = sx - x0 as f64;
            let fy = sy - y0 as f64;
            let v00 = level.data[y0 * level.width + x0];
            let v10 = level.data[y0 * level.width + x1];
            let v01 = level.data[y1 * level.width + x0];
            let v11 = level.data[y1 * level.width + x1];
            data[y * target_w + x] = v00 * (1.0 - fx) * (1.0 - fy)
                + v10 * fx * (1.0 - fy)
                + v01 * (1.0 - fx) * fy
                + v11 * fx * fy;
        }
    }
    PyramidLevel::from_data(data, target_w, target_h, level.scale / 2.0, level.octave.saturating_sub(1))
}

// ── Gaussian Pyramid ──

/// Multi-resolution Gaussian pyramid.
#[derive(Clone, Debug)]
pub struct GaussianPyramid {
    pub levels: Vec<PyramidLevel>,
    sigma: f64,
    num_levels: usize,
}

impl GaussianPyramid {
    pub fn build(image: &[f64], width: usize, height: usize, num_levels: usize, sigma: f64) -> Self {
        let mut levels = Vec::with_capacity(num_levels);
        let base = PyramidLevel::from_data(image.to_vec(), width, height, 1.0, 0);
        levels.push(base);

        for i in 1..num_levels {
            let prev = &levels[i - 1];
            if prev.width <= 1 || prev.height <= 1 {
                break;
            }
            let blurred = gaussian_blur(prev, sigma);
            let down = downsample(&blurred);
            levels.push(down);
        }

        Self { levels, sigma, num_levels }
    }

    pub fn level(&self, idx: usize) -> Option<&PyramidLevel> {
        self.levels.get(idx)
    }

    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    pub fn total_pixels(&self) -> usize {
        self.levels.iter().map(|l| l.pixel_count()).sum()
    }
}

impl fmt::Display for GaussianPyramid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GaussianPyramid({} levels, sigma={:.2})", self.levels.len(), self.sigma)
    }
}

// ── Laplacian Pyramid ──

/// Laplacian pyramid storing band-pass filtered images for reconstruction.
#[derive(Clone, Debug)]
pub struct LaplacianPyramid {
    pub levels: Vec<PyramidLevel>,
    pub residual: PyramidLevel,
}

impl LaplacianPyramid {
    /// Build from a Gaussian pyramid by computing differences between levels.
    pub fn from_gaussian(gauss: &GaussianPyramid) -> Self {
        let n = gauss.levels.len();
        let mut levels = Vec::with_capacity(n.saturating_sub(1));

        for i in 0..n.saturating_sub(1) {
            let current = &gauss.levels[i];
            let next = &gauss.levels[i + 1];
            let upsampled = upsample(next, current.width, current.height);
            let mut diff_data = vec![0.0; current.pixel_count()];
            for (j, d) in diff_data.iter_mut().enumerate() {
                *d = current.data[j] - upsampled.data[j];
            }
            levels.push(PyramidLevel::from_data(diff_data, current.width, current.height, current.scale, current.octave));
        }

        let residual = gauss.levels.last().cloned().unwrap_or_else(|| PyramidLevel::new(1, 1, 1.0, 0));
        Self { levels, residual }
    }

    /// Reconstruct original image from Laplacian pyramid.
    pub fn reconstruct(&self) -> PyramidLevel {
        let mut current = self.residual.clone();

        for level in self.levels.iter().rev() {
            let upsampled = upsample(&current, level.width, level.height);
            let mut data = vec![0.0; level.pixel_count()];
            for (j, d) in data.iter_mut().enumerate() {
                *d = upsampled.data[j] + level.data[j];
            }
            current = PyramidLevel::from_data(data, level.width, level.height, level.scale, level.octave);
        }

        current
    }

    pub fn depth(&self) -> usize {
        self.levels.len() + 1
    }
}

impl fmt::Display for LaplacianPyramid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LaplacianPyramid({} bands + residual)", self.levels.len())
    }
}

// ── Scale Space ──

/// Scale-space representation with multiple scales per octave.
#[derive(Clone, Debug)]
pub struct ScaleSpace {
    pub octaves: Vec<Vec<PyramidLevel>>,
    pub scales_per_octave: usize,
    pub base_sigma: f64,
}

impl ScaleSpace {
    pub fn build(
        image: &[f64],
        width: usize,
        height: usize,
        num_octaves: usize,
        scales_per_octave: usize,
        base_sigma: f64,
    ) -> Self {
        let mut octaves = Vec::with_capacity(num_octaves);
        let k = 2.0_f64.powf(1.0 / scales_per_octave as f64);

        let mut base_level = PyramidLevel::from_data(image.to_vec(), width, height, 1.0, 0);

        for oct in 0..num_octaves {
            if base_level.width <= 1 || base_level.height <= 1 {
                break;
            }
            let mut scale_levels = Vec::with_capacity(scales_per_octave + 3);
            for s in 0..(scales_per_octave + 3) {
                let sigma = base_sigma * k.powi(s as i32);
                let blurred = gaussian_blur(&base_level, sigma);
                let mut level = blurred;
                level.octave = oct;
                level.scale = sigma * (1 << oct) as f64;
                scale_levels.push(level);
            }
            octaves.push(scale_levels);

            // Downsample for next octave (take the scale_per_octave-th level)
            let ref_level = &octaves[oct][scales_per_octave];
            base_level = downsample(ref_level);
        }

        Self { octaves, scales_per_octave, base_sigma }
    }

    pub fn num_octaves(&self) -> usize {
        self.octaves.len()
    }

    /// Compute Difference of Gaussians for an octave.
    pub fn dog_octave(&self, octave: usize) -> Vec<PyramidLevel> {
        let Some(levels) = self.octaves.get(octave) else {
            return Vec::new();
        };
        let mut dogs = Vec::with_capacity(levels.len().saturating_sub(1));
        for i in 0..levels.len().saturating_sub(1) {
            let a = &levels[i];
            let b = &levels[i + 1];
            let mut data = vec![0.0; a.pixel_count()];
            for (j, d) in data.iter_mut().enumerate() {
                *d = b.data[j] - a.data[j];
            }
            dogs.push(PyramidLevel::from_data(data, a.width, a.height, a.scale, octave));
        }
        dogs
    }
}

impl fmt::Display for ScaleSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ScaleSpace({} octaves, {} scales/oct, sigma={:.2})",
            self.octaves.len(), self.scales_per_octave, self.base_sigma)
    }
}

// ── Pyramid Blending ──

/// Blend two images using Laplacian pyramid blending.
pub fn pyramid_blend(
    img_a: &[f64], img_b: &[f64], mask: &[f64],
    width: usize, height: usize, num_levels: usize, sigma: f64,
) -> Vec<f64> {
    let ga = GaussianPyramid::build(img_a, width, height, num_levels, sigma);
    let gb = GaussianPyramid::build(img_b, width, height, num_levels, sigma);
    let gm = GaussianPyramid::build(mask, width, height, num_levels, sigma);

    let la = LaplacianPyramid::from_gaussian(&ga);
    let lb = LaplacianPyramid::from_gaussian(&gb);

    // Blend each Laplacian level
    let mut blended_levels = Vec::with_capacity(la.levels.len());
    for i in 0..la.levels.len() {
        let la_l = &la.levels[i];
        let lb_l = &lb.levels[i];
        let m_l = &gm.levels[i];
        let mut data = vec![0.0; la_l.pixel_count()];
        for (j, d) in data.iter_mut().enumerate() {
            let m = m_l.data.get(j).copied().unwrap_or(0.5);
            *d = la_l.data[j] * m + lb_l.data[j] * (1.0 - m);
        }
        blended_levels.push(PyramidLevel::from_data(data, la_l.width, la_l.height, la_l.scale, la_l.octave));
    }

    // Blend residual
    let ra = &la.residual;
    let rb = &lb.residual;
    let rm = gm.levels.last().unwrap_or(&gm.levels[0]);
    let mut res_data = vec![0.0; ra.pixel_count()];
    for (j, d) in res_data.iter_mut().enumerate() {
        let m = rm.data.get(j).copied().unwrap_or(0.5);
        *d = ra.data[j] * m + rb.data[j] * (1.0 - m);
    }
    let blended_residual = PyramidLevel::from_data(res_data, ra.width, ra.height, ra.scale, ra.octave);

    let blended_lap = LaplacianPyramid {
        levels: blended_levels,
        residual: blended_residual,
    };

    blended_lap.reconstruct().data
}

// ── Pyramid Configuration ──

/// Builder for configuring pyramid construction.
#[derive(Clone, Debug)]
pub struct PyramidConfig {
    pub num_levels: usize,
    pub sigma: f64,
    pub scales_per_octave: usize,
    pub min_size: usize,
}

impl PyramidConfig {
    pub fn new() -> Self {
        Self {
            num_levels: 4,
            sigma: 1.6,
            scales_per_octave: 3,
            min_size: 8,
        }
    }

    pub fn with_num_levels(mut self, n: usize) -> Self {
        self.num_levels = n.max(1);
        self
    }

    pub fn with_sigma(mut self, s: f64) -> Self {
        self.sigma = s.max(0.1);
        self
    }

    pub fn with_scales_per_octave(mut self, s: usize) -> Self {
        self.scales_per_octave = s.max(1);
        self
    }

    pub fn with_min_size(mut self, s: usize) -> Self {
        self.min_size = s.max(1);
        self
    }

    /// Compute maximum useful levels for given image dimensions.
    pub fn max_levels_for(&self, width: usize, height: usize) -> usize {
        let min_dim = width.min(height);
        if min_dim <= self.min_size {
            return 1;
        }
        let mut levels = 1;
        let mut dim = min_dim;
        while dim / 2 >= self.min_size {
            dim /= 2;
            levels += 1;
        }
        levels
    }
}

impl Default for PyramidConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for PyramidConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PyramidConfig(levels={}, sigma={:.2}, scales/oct={}, min={})",
            self.num_levels, self.sigma, self.scales_per_octave, self.min_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_image(w: usize, h: usize) -> Vec<f64> {
        (0..w * h).map(|i| (i as f64) / (w * h) as f64).collect()
    }

    fn constant_image(w: usize, h: usize, val: f64) -> Vec<f64> {
        vec![val; w * h]
    }

    #[test]
    fn test_pyramid_level_creation() {
        let level = PyramidLevel::new(10, 10, 1.0, 0);
        assert_eq!(level.pixel_count(), 100);
        assert_eq!(level.get(5, 5), 0.0);
    }

    #[test]
    fn test_pyramid_level_set_get() {
        let mut level = PyramidLevel::new(4, 4, 1.0, 0);
        level.set(2, 3, 0.75);
        assert!((level.get(2, 3) - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_pyramid_level_display() {
        let level = PyramidLevel::new(32, 16, 2.0, 1);
        let s = format!("{level}");
        assert!(s.contains("32x16"));
    }

    #[test]
    fn test_gaussian_kernel_sums_to_one() {
        let k = gaussian_kernel_1d(1.5, 4);
        let sum: f64 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_kernel_symmetric() {
        let k = gaussian_kernel_1d(2.0, 5);
        for i in 0..k.len() / 2 {
            assert!((k[i] - k[k.len() - 1 - i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_gaussian_blur_preserves_constant() {
        let img = constant_image(16, 16, 0.5);
        let level = PyramidLevel::from_data(img, 16, 16, 1.0, 0);
        let blurred = gaussian_blur(&level, 1.5);
        for v in &blurred.data {
            assert!((*v - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_downsample_halves_dimensions() {
        let level = PyramidLevel::new(32, 24, 1.0, 0);
        let down = downsample(&level);
        assert_eq!(down.width, 16);
        assert_eq!(down.height, 12);
        assert_eq!(down.octave, 1);
    }

    #[test]
    fn test_upsample_doubles_dimensions() {
        let level = PyramidLevel::new(8, 8, 2.0, 1);
        let up = upsample(&level, 16, 16);
        assert_eq!(up.width, 16);
        assert_eq!(up.height, 16);
    }

    #[test]
    fn test_gaussian_pyramid_levels() {
        let img = make_image(64, 64);
        let pyr = GaussianPyramid::build(&img, 64, 64, 4, 1.6);
        assert_eq!(pyr.depth(), 4);
        assert_eq!(pyr.levels[0].width, 64);
        assert_eq!(pyr.levels[1].width, 32);
        assert_eq!(pyr.levels[2].width, 16);
        assert_eq!(pyr.levels[3].width, 8);
    }

    #[test]
    fn test_gaussian_pyramid_total_pixels() {
        let img = make_image(32, 32);
        let pyr = GaussianPyramid::build(&img, 32, 32, 3, 1.0);
        assert!(pyr.total_pixels() > 32 * 32);
    }

    #[test]
    fn test_gaussian_pyramid_display() {
        let img = make_image(16, 16);
        let pyr = GaussianPyramid::build(&img, 16, 16, 3, 1.6);
        let s = format!("{pyr}");
        assert!(s.contains("3 levels"));
    }

    #[test]
    fn test_laplacian_reconstruction() {
        let img = make_image(32, 32);
        let gauss = GaussianPyramid::build(&img, 32, 32, 4, 1.0);
        let lap = LaplacianPyramid::from_gaussian(&gauss);
        let recon = lap.reconstruct();
        assert_eq!(recon.width, 32);
        assert_eq!(recon.height, 32);
        // Reconstruction should be close to original
        let max_err: f64 = img.iter().zip(recon.data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(max_err < 0.1, "max reconstruction error: {max_err}");
    }

    #[test]
    fn test_laplacian_depth() {
        let img = make_image(16, 16);
        let gauss = GaussianPyramid::build(&img, 16, 16, 3, 1.0);
        let lap = LaplacianPyramid::from_gaussian(&gauss);
        assert_eq!(lap.depth(), 3);
    }

    #[test]
    fn test_laplacian_display() {
        let img = make_image(16, 16);
        let gauss = GaussianPyramid::build(&img, 16, 16, 3, 1.0);
        let lap = LaplacianPyramid::from_gaussian(&gauss);
        let s = format!("{lap}");
        assert!(s.contains("bands"));
    }

    #[test]
    fn test_scale_space_construction() {
        let img = make_image(32, 32);
        let ss = ScaleSpace::build(&img, 32, 32, 3, 3, 1.6);
        assert_eq!(ss.num_octaves(), 3);
        assert_eq!(ss.octaves[0].len(), 6); // scales_per_octave + 3
    }

    #[test]
    fn test_scale_space_dog() {
        let img = make_image(32, 32);
        let ss = ScaleSpace::build(&img, 32, 32, 2, 3, 1.6);
        let dogs = ss.dog_octave(0);
        assert_eq!(dogs.len(), 5); // 6 levels - 1
    }

    #[test]
    fn test_pyramid_blend_constant() {
        let a = constant_image(16, 16, 1.0);
        let b = constant_image(16, 16, 0.0);
        let mask = constant_image(16, 16, 0.5);
        let result = pyramid_blend(&a, &b, &mask, 16, 16, 3, 1.0);
        assert_eq!(result.len(), 256);
        for v in &result {
            assert!((*v - 0.5).abs() < 0.15, "blended value {v} not near 0.5");
        }
    }

    #[test]
    fn test_pyramid_config_defaults() {
        let cfg = PyramidConfig::new();
        assert_eq!(cfg.num_levels, 4);
        assert!((cfg.sigma - 1.6).abs() < 1e-6);
    }

    #[test]
    fn test_pyramid_config_builder() {
        let cfg = PyramidConfig::new()
            .with_num_levels(6)
            .with_sigma(2.0)
            .with_scales_per_octave(4)
            .with_min_size(16);
        assert_eq!(cfg.num_levels, 6);
        assert_eq!(cfg.scales_per_octave, 4);
    }

    #[test]
    fn test_pyramid_config_max_levels() {
        let cfg = PyramidConfig::new().with_min_size(8);
        assert_eq!(cfg.max_levels_for(256, 256), 6); // 256 -> 128 -> 64 -> 32 -> 16 -> 8
    }
}
