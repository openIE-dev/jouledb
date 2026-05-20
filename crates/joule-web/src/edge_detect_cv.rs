//! Edge detection: Sobel, Canny, Laplacian of Gaussian, gradient
//! magnitude/direction, and hysteresis thresholding.
//!
//! Operates on grayscale image buffers (`&[f64]`, row-major, values
//! in `[0.0, 255.0]`) with explicit `(width, height)` dimensions.

use std::fmt;

// ── GrayImage helper ───────────────────────────────────────────

/// A simple grayscale image buffer with width and height.
#[derive(Debug, Clone, PartialEq)]
pub struct GrayImage {
    pub data: Vec<f64>,
    pub width: usize,
    pub height: usize,
}

impl GrayImage {
    pub fn new(width: usize, height: usize) -> Self {
        Self { data: vec![0.0; width * height], width, height }
    }

    pub fn from_data(data: Vec<f64>, width: usize, height: usize) -> Self {
        assert_eq!(data.len(), width * height, "data length must match width*height");
        Self { data, width, height }
    }

    pub fn get(&self, x: usize, y: usize) -> f64 {
        self.data[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, val: f64) {
        self.data[y * self.width + x] = val;
    }
}

impl fmt::Display for GrayImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GrayImage({}x{})", self.width, self.height)
    }
}

// ── Convolution ────────────────────────────────────────────────

/// 2D convolution with a square kernel (odd dimension), zero-padded.
pub fn convolve2d(image: &GrayImage, kernel: &[f64], ksize: usize) -> GrayImage {
    assert!(ksize % 2 == 1, "kernel size must be odd");
    let half = ksize / 2;
    let mut out = GrayImage::new(image.width, image.height);
    let w = image.width as i64;
    let h = image.height as i64;

    for row in 0..image.height {
        for col in 0..image.width {
            let mut sum = 0.0_f64;
            for kr in 0..ksize {
                for kc in 0..ksize {
                    let sr = row as i64 + kr as i64 - half as i64;
                    let sc = col as i64 + kc as i64 - half as i64;
                    if sr >= 0 && sr < h && sc >= 0 && sc < w {
                        sum += image.data[sr as usize * image.width + sc as usize]
                             * kernel[kr * ksize + kc];
                    }
                }
            }
            out.data[row * image.width + col] = sum;
        }
    }
    out
}

// ── Gaussian Blur ──────────────────────────────────────────────

/// Generate a 1D Gaussian kernel of given size and sigma.
pub fn gaussian_kernel_1d(size: usize, sigma: f64) -> Vec<f64> {
    assert!(size % 2 == 1, "kernel size must be odd");
    let half = size as f64 / 2.0;
    let mut kernel = Vec::with_capacity(size);
    let mut sum = 0.0_f64;
    for i in 0..size {
        let x = i as f64 - half + 0.5;
        let val = (-x * x / (2.0 * sigma * sigma)).exp();
        kernel.push(val);
        sum += val;
    }
    for v in &mut kernel {
        *v /= sum;
    }
    kernel
}

/// Generate a 2D Gaussian kernel.
pub fn gaussian_kernel_2d(size: usize, sigma: f64) -> Vec<f64> {
    let k1d = gaussian_kernel_1d(size, sigma);
    let mut k2d = vec![0.0_f64; size * size];
    for r in 0..size {
        for c in 0..size {
            k2d[r * size + c] = k1d[r] * k1d[c];
        }
    }
    k2d
}

/// Apply Gaussian blur.
pub fn gaussian_blur(image: &GrayImage, size: usize, sigma: f64) -> GrayImage {
    let kernel = gaussian_kernel_2d(size, sigma);
    convolve2d(image, &kernel, size)
}

// ── Sobel ──────────────────────────────────────────────────────

/// Sobel gradient in X direction.
pub fn sobel_x(image: &GrayImage) -> GrayImage {
    #[rustfmt::skip]
    let kernel = [
        -1.0, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0,
    ];
    convolve2d(image, &kernel, 3)
}

/// Sobel gradient in Y direction.
pub fn sobel_y(image: &GrayImage) -> GrayImage {
    #[rustfmt::skip]
    let kernel = [
        -1.0, -2.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  2.0,  1.0,
    ];
    convolve2d(image, &kernel, 3)
}

/// Gradient magnitude from X and Y gradient images.
pub fn gradient_magnitude(gx: &GrayImage, gy: &GrayImage) -> GrayImage {
    assert_eq!(gx.width, gy.width);
    assert_eq!(gx.height, gy.height);
    let mut mag = GrayImage::new(gx.width, gx.height);
    for i in 0..gx.data.len() {
        mag.data[i] = (gx.data[i].powi(2) + gy.data[i].powi(2)).sqrt();
    }
    mag
}

/// Gradient direction (radians, range [0, 2*pi)) from X and Y.
pub fn gradient_direction(gx: &GrayImage, gy: &GrayImage) -> GrayImage {
    assert_eq!(gx.width, gy.width);
    assert_eq!(gx.height, gy.height);
    let mut dir = GrayImage::new(gx.width, gx.height);
    for i in 0..gx.data.len() {
        let mut angle = gy.data[i].atan2(gx.data[i]);
        if angle < 0.0 {
            angle += 2.0 * std::f64::consts::PI;
        }
        dir.data[i] = angle;
    }
    dir
}

// ── Canny Edge Detector ────────────────────────────────────────

/// Configuration for Canny edge detection.
#[derive(Debug, Clone)]
pub struct CannyConfig {
    pub low_threshold: f64,
    pub high_threshold: f64,
    pub blur_size: usize,
    pub blur_sigma: f64,
}

impl CannyConfig {
    pub fn new() -> Self {
        Self {
            low_threshold: 50.0,
            high_threshold: 150.0,
            blur_size: 5,
            blur_sigma: 1.4,
        }
    }

    pub fn with_thresholds(mut self, low: f64, high: f64) -> Self {
        self.low_threshold = low;
        self.high_threshold = high;
        self
    }

    pub fn with_blur(mut self, size: usize, sigma: f64) -> Self {
        self.blur_size = size;
        self.blur_sigma = sigma;
        self
    }
}

impl fmt::Display for CannyConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CannyConfig(lo={}, hi={}, blur={}x{:.1})",
               self.low_threshold, self.high_threshold, self.blur_size, self.blur_sigma)
    }
}

/// Quantize gradient direction to one of four angle bins: 0, 45, 90, 135.
fn quantize_angle(angle: f64) -> u8 {
    let deg = angle.to_degrees() % 180.0;
    if deg < 22.5 || deg >= 157.5 {
        0 // horizontal
    } else if deg < 67.5 {
        45
    } else if deg < 112.5 {
        90 // vertical
    } else {
        135
    }
}

/// Non-maximum suppression along gradient direction.
fn non_max_suppress(mag: &GrayImage, dir: &GrayImage) -> GrayImage {
    let mut out = GrayImage::new(mag.width, mag.height);
    let w = mag.width;
    let h = mag.height;

    for r in 1..h - 1 {
        for c in 1..w - 1 {
            let angle = quantize_angle(dir.data[r * w + c]);
            let m = mag.data[r * w + c];

            let (n1, n2) = match angle {
                0   => (mag.data[r * w + c - 1], mag.data[r * w + c + 1]),
                45  => (mag.data[(r - 1) * w + c + 1], mag.data[(r + 1) * w + c - 1]),
                90  => (mag.data[(r - 1) * w + c], mag.data[(r + 1) * w + c]),
                _   => (mag.data[(r - 1) * w + c - 1], mag.data[(r + 1) * w + c + 1]),
            };

            out.data[r * w + c] = if m >= n1 && m >= n2 { m } else { 0.0 };
        }
    }
    out
}

/// Hysteresis thresholding: pixels above `high` are strong edges;
/// pixels between `low` and `high` are weak edges kept only if
/// connected to a strong edge.
pub fn hysteresis(
    thin: &GrayImage, low: f64, high: f64,
) -> GrayImage {
    let w = thin.width;
    let h = thin.height;
    let mut out = GrayImage::new(w, h);

    const STRONG: f64 = 255.0;
    const WEAK: f64 = 128.0;

    // Classify
    for i in 0..thin.data.len() {
        if thin.data[i] >= high {
            out.data[i] = STRONG;
        } else if thin.data[i] >= low {
            out.data[i] = WEAK;
        }
    }

    // Connect weak to strong via flood-fill
    let mut changed = true;
    while changed {
        changed = false;
        for r in 1..h - 1 {
            for c in 1..w - 1 {
                if out.data[r * w + c] != WEAK {
                    continue;
                }
                let has_strong_neighbor =
                    out.data[(r - 1) * w + c - 1] == STRONG
                    || out.data[(r - 1) * w + c] == STRONG
                    || out.data[(r - 1) * w + c + 1] == STRONG
                    || out.data[r * w + c - 1] == STRONG
                    || out.data[r * w + c + 1] == STRONG
                    || out.data[(r + 1) * w + c - 1] == STRONG
                    || out.data[(r + 1) * w + c] == STRONG
                    || out.data[(r + 1) * w + c + 1] == STRONG;
                if has_strong_neighbor {
                    out.data[r * w + c] = STRONG;
                    changed = true;
                }
            }
        }
    }

    // Remove remaining weak edges
    for v in &mut out.data {
        if *v != STRONG {
            *v = 0.0;
        }
    }
    out
}

/// Full Canny edge detection pipeline.
pub fn canny(image: &GrayImage, config: &CannyConfig) -> GrayImage {
    let blurred = gaussian_blur(image, config.blur_size, config.blur_sigma);
    let gx = sobel_x(&blurred);
    let gy = sobel_y(&blurred);
    let mag = gradient_magnitude(&gx, &gy);
    let dir = gradient_direction(&gx, &gy);
    let thin = non_max_suppress(&mag, &dir);
    hysteresis(&thin, config.low_threshold, config.high_threshold)
}

// ── Laplacian of Gaussian ──────────────────────────────────────

/// Compute Laplacian of Gaussian (LoG) kernel.
pub fn log_kernel(size: usize, sigma: f64) -> Vec<f64> {
    assert!(size % 2 == 1, "kernel size must be odd");
    let half = size as f64 / 2.0;
    let s2 = sigma * sigma;
    let s4 = s2 * s2;
    let mut kernel = vec![0.0_f64; size * size];
    let mut sum = 0.0_f64;

    for r in 0..size {
        for c in 0..size {
            let x = c as f64 - half + 0.5;
            let y = r as f64 - half + 0.5;
            let r2 = x * x + y * y;
            let val = -1.0 / (std::f64::consts::PI * s4)
                * (1.0 - r2 / (2.0 * s2))
                * (-r2 / (2.0 * s2)).exp();
            kernel[r * size + c] = val;
            sum += val;
        }
    }

    // Normalize to zero-sum
    let mean = sum / (size * size) as f64;
    for v in &mut kernel {
        *v -= mean;
    }
    kernel
}

/// Apply Laplacian of Gaussian.
pub fn laplacian_of_gaussian(image: &GrayImage, size: usize, sigma: f64) -> GrayImage {
    let kernel = log_kernel(size, sigma);
    convolve2d(image, &kernel, size)
}

/// Detect zero-crossings in a LoG response image.
pub fn zero_crossings(log_image: &GrayImage, threshold: f64) -> GrayImage {
    let w = log_image.width;
    let h = log_image.height;
    let mut edges = GrayImage::new(w, h);

    for r in 1..h - 1 {
        for c in 1..w - 1 {
            let val = log_image.data[r * w + c];
            let right = log_image.data[r * w + c + 1];
            let below = log_image.data[(r + 1) * w + c];

            if (val > 0.0 && right < 0.0) || (val < 0.0 && right > 0.0) {
                if (val - right).abs() > threshold {
                    edges.data[r * w + c] = 255.0;
                }
            }
            if (val > 0.0 && below < 0.0) || (val < 0.0 && below > 0.0) {
                if (val - below).abs() > threshold {
                    edges.data[r * w + c] = 255.0;
                }
            }
        }
    }
    edges
}

// ── Edge Strength Statistics ───────────────────────────────────

/// Count non-zero edge pixels in an edge map.
pub fn edge_pixel_count(edge_map: &GrayImage) -> usize {
    edge_map.data.iter().filter(|&&v| v > 0.0).count()
}

/// Fraction of edge pixels relative to total.
pub fn edge_density(edge_map: &GrayImage) -> f64 {
    let total = edge_map.data.len();
    if total == 0 { return 0.0; }
    edge_pixel_count(edge_map) as f64 / total as f64
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_image(w: usize, h: usize, val: f64) -> GrayImage {
        GrayImage::from_data(vec![val; w * h], w, h)
    }

    fn step_edge_image(w: usize, h: usize) -> GrayImage {
        let mut data = vec![0.0_f64; w * h];
        for r in 0..h {
            for c in w / 2..w {
                data[r * w + c] = 255.0;
            }
        }
        GrayImage::from_data(data, w, h)
    }

    #[test]
    fn test_gray_image_basic() {
        let mut img = GrayImage::new(10, 10);
        img.set(5, 3, 42.0);
        assert_eq!(img.get(5, 3), 42.0);
    }

    #[test]
    fn test_gray_image_display() {
        let img = GrayImage::new(320, 240);
        assert_eq!(format!("{}", img), "GrayImage(320x240)");
    }

    #[test]
    fn test_gaussian_kernel_1d_sums_to_one() {
        let k = gaussian_kernel_1d(5, 1.0);
        let sum: f64 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_gaussian_kernel_1d_symmetric() {
        let k = gaussian_kernel_1d(7, 1.5);
        assert!((k[0] - k[6]).abs() < 1e-9);
        assert!((k[1] - k[5]).abs() < 1e-9);
        assert!((k[2] - k[4]).abs() < 1e-9);
    }

    #[test]
    fn test_gaussian_blur_preserves_uniform() {
        let img = uniform_image(20, 20, 100.0);
        let blurred = gaussian_blur(&img, 3, 1.0);
        // Interior pixels should stay ~100
        let center = blurred.get(10, 10);
        assert!((center - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_sobel_x_uniform_zero() {
        let img = uniform_image(20, 20, 128.0);
        let gx = sobel_x(&img);
        for r in 1..19 {
            for c in 1..19 {
                assert!(gx.get(c, r).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_sobel_detects_vertical_edge() {
        let img = step_edge_image(20, 20);
        let gx = sobel_x(&img);
        // Strong response near the vertical edge at column 10
        let edge_response = gx.get(10, 10).abs();
        let flat_response = gx.get(3, 10).abs();
        assert!(edge_response > flat_response);
    }

    #[test]
    fn test_gradient_magnitude_pythagorean() {
        let mut gx = GrayImage::new(5, 5);
        let mut gy = GrayImage::new(5, 5);
        gx.set(2, 2, 3.0);
        gy.set(2, 2, 4.0);
        let mag = gradient_magnitude(&gx, &gy);
        assert!((mag.get(2, 2) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_gradient_direction_quadrants() {
        let mut gx = GrayImage::new(5, 5);
        let mut gy = GrayImage::new(5, 5);
        gx.set(2, 2, 1.0);
        gy.set(2, 2, 0.0);
        let dir = gradient_direction(&gx, &gy);
        assert!(dir.get(2, 2).abs() < 1e-9); // 0 radians
    }

    #[test]
    fn test_canny_config_builder() {
        let cfg = CannyConfig::new().with_thresholds(30.0, 100.0).with_blur(3, 1.0);
        assert_eq!(cfg.low_threshold, 30.0);
        assert_eq!(cfg.high_threshold, 100.0);
        assert_eq!(cfg.blur_size, 3);
    }

    #[test]
    fn test_canny_config_display() {
        let cfg = CannyConfig::new();
        let s = format!("{}", cfg);
        assert!(s.contains("CannyConfig"));
    }

    #[test]
    fn test_canny_uniform_no_edges() {
        let img = uniform_image(30, 30, 128.0);
        let cfg = CannyConfig::new().with_blur(3, 1.0);
        let edges = canny(&img, &cfg);
        assert_eq!(edge_pixel_count(&edges), 0);
    }

    #[test]
    fn test_canny_step_edge() {
        let img = step_edge_image(40, 40);
        let cfg = CannyConfig::new().with_thresholds(10.0, 30.0).with_blur(3, 0.8);
        let edges = canny(&img, &cfg);
        assert!(edge_pixel_count(&edges) > 0, "should detect the vertical edge");
    }

    #[test]
    fn test_hysteresis_strong_only() {
        let mut img = GrayImage::new(10, 10);
        img.set(5, 5, 200.0);
        let out = hysteresis(&img, 50.0, 100.0);
        assert_eq!(out.get(5, 5), 255.0);
    }

    #[test]
    fn test_log_kernel_size() {
        let k = log_kernel(7, 1.0);
        assert_eq!(k.len(), 49);
    }

    #[test]
    fn test_laplacian_of_gaussian_uniform() {
        let img = uniform_image(20, 20, 128.0);
        let result = laplacian_of_gaussian(&img, 5, 1.0);
        // Interior should be near zero for uniform input
        let center = result.get(10, 10);
        assert!(center.abs() < 1.0);
    }

    #[test]
    fn test_zero_crossings() {
        let mut img = GrayImage::new(10, 10);
        // Create a sign change at column 5
        for r in 0..10 {
            for c in 0..5 { img.set(c, r, 10.0); }
            for c in 5..10 { img.set(c, r, -10.0); }
        }
        let zc = zero_crossings(&img, 5.0);
        // Should have edges near column 4-5
        let mut found = false;
        for r in 1..9 {
            if zc.get(4, r) > 0.0 { found = true; }
        }
        assert!(found, "should find zero crossings at sign boundary");
    }

    #[test]
    fn test_edge_density() {
        let mut img = GrayImage::new(10, 10);
        for i in 0..25 {
            img.data[i] = 255.0;
        }
        let density = edge_density(&img);
        assert!((density - 0.25).abs() < 1e-9);
    }
}
