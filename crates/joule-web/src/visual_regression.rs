//! Visual regression testing.
//!
//! Replaces Percy, BackstopJS, and reg-suit with pure-Rust pixel buffer
//! comparison. Supports exact match, perceptual diff, SSIM, region masking,
//! and diff image generation.

use serde::{Deserialize, Serialize};

// ── Pixel buffer ────────────────────────────────────────────────

/// RGBA pixel (red, green, blue, alpha).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pixel {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Pixel {
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    /// Luminance using BT.709 coefficients.
    pub fn luminance(&self) -> f64 {
        0.2126 * (self.r as f64 / 255.0)
            + 0.7152 * (self.g as f64 / 255.0)
            + 0.0722 * (self.b as f64 / 255.0)
    }

    /// Per-channel difference magnitude.
    fn diff_magnitude(&self, other: &Pixel) -> u32 {
        let dr = (self.r as i32 - other.r as i32).unsigned_abs();
        let dg = (self.g as i32 - other.g as i32).unsigned_abs();
        let db = (self.b as i32 - other.b as i32).unsigned_abs();
        let da = (self.a as i32 - other.a as i32).unsigned_abs();
        dr + dg + db + da
    }
}

/// A buffer of pixels with known dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PixelBuffer {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<Pixel>,
}

impl PixelBuffer {
    /// Create a buffer filled with a single color.
    pub fn filled(width: u32, height: u32, pixel: Pixel) -> Self {
        let count = (width as usize) * (height as usize);
        Self {
            width,
            height,
            pixels: vec![pixel; count],
        }
    }

    /// Get pixel at (x, y).
    pub fn get(&self, x: u32, y: u32) -> Option<&Pixel> {
        if x < self.width && y < self.height {
            Some(&self.pixels[(y as usize) * (self.width as usize) + (x as usize)])
        } else {
            None
        }
    }

    /// Set pixel at (x, y).
    pub fn set(&mut self, x: u32, y: u32, pixel: Pixel) {
        if x < self.width && y < self.height {
            self.pixels[(y as usize) * (self.width as usize) + (x as usize)] = pixel;
        }
    }

    /// Total number of pixels.
    pub fn pixel_count(&self) -> usize {
        self.pixels.len()
    }
}

// ── Masking ─────────────────────────────────────────────────────

/// A rectangular region to mask (ignore) during comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl MaskRegion {
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self { x, y, width, height }
    }

    /// Check if a pixel coordinate falls within this mask.
    pub fn contains(&self, px: u32, py: u32) -> bool {
        px >= self.x && px < self.x + self.width && py >= self.y && py < self.y + self.height
    }
}

// ── Diff metrics ────────────────────────────────────────────────

/// Results of comparing two pixel buffers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffMetrics {
    pub pixel_mismatch_count: usize,
    pub total_pixels: usize,
    pub mismatch_percentage: f64,
    pub ssim: f64,
}

impl DiffMetrics {
    /// Check if the diff passes a given mismatch threshold (percentage 0..100).
    pub fn passes_threshold(&self, max_mismatch_pct: f64) -> bool {
        self.mismatch_percentage <= max_mismatch_pct
    }

    /// Check if SSIM exceeds a minimum value (0.0..1.0).
    pub fn passes_ssim_threshold(&self, min_ssim: f64) -> bool {
        self.ssim >= min_ssim
    }
}

/// A diff image highlighting changed pixels.
#[derive(Debug, Clone)]
pub struct DiffImage {
    pub buffer: PixelBuffer,
    pub changed_count: usize,
}

// ── Comparison ──────────────────────────────────────────────────

/// Compare two pixel buffers exactly (pixel-by-pixel), optionally ignoring masked regions.
pub fn compare(
    baseline: &PixelBuffer,
    actual: &PixelBuffer,
    masks: &[MaskRegion],
) -> DiffMetrics {
    assert_eq!(baseline.width, actual.width, "Width mismatch");
    assert_eq!(baseline.height, actual.height, "Height mismatch");

    let total = baseline.pixel_count();
    let mut mismatches = 0usize;

    for y in 0..baseline.height {
        for x in 0..baseline.width {
            if masks.iter().any(|m| m.contains(x, y)) {
                continue;
            }
            let bp = &baseline.pixels[(y as usize) * (baseline.width as usize) + (x as usize)];
            let ap = &actual.pixels[(y as usize) * (actual.width as usize) + (x as usize)];
            if bp != ap {
                mismatches += 1;
            }
        }
    }

    let pct = if total > 0 {
        (mismatches as f64 / total as f64) * 100.0
    } else {
        0.0
    };

    let ssim = compute_ssim(baseline, actual);

    DiffMetrics {
        pixel_mismatch_count: mismatches,
        total_pixels: total,
        mismatch_percentage: pct,
        ssim,
    }
}

/// Compare two pixel buffers with a per-channel tolerance.
/// Pixels whose per-channel difference is within `tolerance` are considered matching.
pub fn compare_perceptual(
    baseline: &PixelBuffer,
    actual: &PixelBuffer,
    tolerance: u32,
    masks: &[MaskRegion],
) -> DiffMetrics {
    assert_eq!(baseline.width, actual.width, "Width mismatch");
    assert_eq!(baseline.height, actual.height, "Height mismatch");

    let total = baseline.pixel_count();
    let mut mismatches = 0usize;

    for y in 0..baseline.height {
        for x in 0..baseline.width {
            if masks.iter().any(|m| m.contains(x, y)) {
                continue;
            }
            let idx = (y as usize) * (baseline.width as usize) + (x as usize);
            let bp = &baseline.pixels[idx];
            let ap = &actual.pixels[idx];
            if bp.diff_magnitude(ap) > tolerance {
                mismatches += 1;
            }
        }
    }

    let pct = if total > 0 {
        (mismatches as f64 / total as f64) * 100.0
    } else {
        0.0
    };

    let ssim = compute_ssim(baseline, actual);

    DiffMetrics {
        pixel_mismatch_count: mismatches,
        total_pixels: total,
        mismatch_percentage: pct,
        ssim,
    }
}

/// Generate a diff image highlighting changed pixels in red.
pub fn generate_diff_image(
    baseline: &PixelBuffer,
    actual: &PixelBuffer,
    masks: &[MaskRegion],
) -> DiffImage {
    assert_eq!(baseline.width, actual.width);
    assert_eq!(baseline.height, actual.height);

    let mut output = baseline.clone();
    let mut changed = 0usize;
    let red = Pixel::new(255, 0, 0, 255);
    let dim = Pixel::new(50, 50, 50, 128);

    for y in 0..baseline.height {
        for x in 0..baseline.width {
            let idx = (y as usize) * (baseline.width as usize) + (x as usize);
            if masks.iter().any(|m| m.contains(x, y)) {
                output.pixels[idx] = dim;
            } else if baseline.pixels[idx] != actual.pixels[idx] {
                output.pixels[idx] = red;
                changed += 1;
            }
        }
    }

    DiffImage {
        buffer: output,
        changed_count: changed,
    }
}

/// Compute a simplified SSIM (structural similarity index) between two buffers.
/// Returns a value in [0.0, 1.0] where 1.0 means identical.
fn compute_ssim(a: &PixelBuffer, b: &PixelBuffer) -> f64 {
    if a.pixel_count() == 0 {
        return 1.0;
    }

    let n = a.pixel_count() as f64;

    // Compute mean luminance
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    for i in 0..a.pixels.len() {
        sum_a += a.pixels[i].luminance();
        sum_b += b.pixels[i].luminance();
    }
    let mu_a = sum_a / n;
    let mu_b = sum_b / n;

    // Compute variance and covariance
    let mut var_a = 0.0f64;
    let mut var_b = 0.0f64;
    let mut cov = 0.0f64;
    for i in 0..a.pixels.len() {
        let la = a.pixels[i].luminance() - mu_a;
        let lb = b.pixels[i].luminance() - mu_b;
        var_a += la * la;
        var_b += lb * lb;
        cov += la * lb;
    }
    var_a /= n;
    var_b /= n;
    cov /= n;

    // SSIM constants (for luminance range 0..1)
    let c1 = 0.0001; // (k1*L)^2 where k1=0.01, L=1
    let c2 = 0.0009; // (k2*L)^2 where k2=0.03, L=1

    let numerator = (2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2);
    let denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2);

    if denominator == 0.0 {
        1.0
    } else {
        numerator / denominator
    }
}

// ── Snapshot save/load ──────────────────────────────────────────

/// Serialize a pixel buffer to JSON bytes.
pub fn save_snapshot(buffer: &PixelBuffer) -> String {
    serde_json::to_string(buffer).unwrap_or_default()
}

/// Deserialize a pixel buffer from JSON bytes.
pub fn load_snapshot(data: &str) -> Option<PixelBuffer> {
    serde_json::from_str(data).ok()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn white() -> Pixel {
        Pixel::rgb(255, 255, 255)
    }

    fn black() -> Pixel {
        Pixel::rgb(0, 0, 0)
    }

    fn red() -> Pixel {
        Pixel::rgb(255, 0, 0)
    }

    #[test]
    fn identical_buffers_zero_diff() {
        let a = PixelBuffer::filled(10, 10, white());
        let b = PixelBuffer::filled(10, 10, white());
        let metrics = compare(&a, &b, &[]);
        assert_eq!(metrics.pixel_mismatch_count, 0);
        assert!((metrics.mismatch_percentage - 0.0).abs() < f64::EPSILON);
        assert!(metrics.ssim > 0.99);
    }

    #[test]
    fn fully_different_buffers() {
        let a = PixelBuffer::filled(4, 4, white());
        let b = PixelBuffer::filled(4, 4, black());
        let metrics = compare(&a, &b, &[]);
        assert_eq!(metrics.pixel_mismatch_count, 16);
        assert!((metrics.mismatch_percentage - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn single_pixel_change() {
        let a = PixelBuffer::filled(10, 10, white());
        let mut b = a.clone();
        b.set(5, 5, black());
        let metrics = compare(&a, &b, &[]);
        assert_eq!(metrics.pixel_mismatch_count, 1);
        assert!((metrics.mismatch_percentage - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mask_region_ignores_changes() {
        let a = PixelBuffer::filled(10, 10, white());
        let mut b = a.clone();
        // Change pixel within mask region
        b.set(2, 2, black());
        let mask = MaskRegion::new(0, 0, 5, 5);
        let metrics = compare(&a, &b, &[mask]);
        assert_eq!(metrics.pixel_mismatch_count, 0);
    }

    #[test]
    fn mask_region_does_not_ignore_outside() {
        let a = PixelBuffer::filled(10, 10, white());
        let mut b = a.clone();
        b.set(8, 8, black());
        let mask = MaskRegion::new(0, 0, 5, 5);
        let metrics = compare(&a, &b, &[mask]);
        assert_eq!(metrics.pixel_mismatch_count, 1);
    }

    #[test]
    fn perceptual_diff_within_tolerance() {
        let a = PixelBuffer::filled(4, 4, Pixel::rgb(100, 100, 100));
        let b = PixelBuffer::filled(4, 4, Pixel::rgb(101, 100, 100));
        // diff_magnitude = 1, tolerance = 5 → match
        let metrics = compare_perceptual(&a, &b, 5, &[]);
        assert_eq!(metrics.pixel_mismatch_count, 0);
    }

    #[test]
    fn perceptual_diff_exceeds_tolerance() {
        let a = PixelBuffer::filled(4, 4, Pixel::rgb(100, 100, 100));
        let b = PixelBuffer::filled(4, 4, Pixel::rgb(200, 100, 100));
        let metrics = compare_perceptual(&a, &b, 5, &[]);
        assert_eq!(metrics.pixel_mismatch_count, 16);
    }

    #[test]
    fn diff_image_highlights_changes() {
        let a = PixelBuffer::filled(4, 4, white());
        let mut b = a.clone();
        b.set(1, 1, black());
        b.set(2, 2, black());
        let diff = generate_diff_image(&a, &b, &[]);
        assert_eq!(diff.changed_count, 2);
        // Changed pixels should be red
        assert_eq!(diff.buffer.get(1, 1), Some(&Pixel::new(255, 0, 0, 255)));
        // Unchanged pixels stay white
        assert_eq!(diff.buffer.get(0, 0), Some(&white()));
    }

    #[test]
    fn threshold_pass_fail() {
        let metrics = DiffMetrics {
            pixel_mismatch_count: 5,
            total_pixels: 100,
            mismatch_percentage: 5.0,
            ssim: 0.95,
        };
        assert!(metrics.passes_threshold(10.0));
        assert!(!metrics.passes_threshold(2.0));
        assert!(metrics.passes_ssim_threshold(0.9));
        assert!(!metrics.passes_ssim_threshold(0.99));
    }

    #[test]
    fn ssim_identical_is_one() {
        let a = PixelBuffer::filled(8, 8, Pixel::rgb(128, 128, 128));
        let ssim = compute_ssim(&a, &a);
        assert!((ssim - 1.0).abs() < 0.001);
    }

    #[test]
    fn snapshot_round_trip() {
        let buf = PixelBuffer::filled(2, 2, red());
        let json = save_snapshot(&buf);
        let loaded = load_snapshot(&json).expect("deserialize");
        assert_eq!(loaded.width, 2);
        assert_eq!(loaded.height, 2);
        assert_eq!(loaded.pixels.len(), 4);
        assert_eq!(loaded.pixels[0], red());
    }

    #[test]
    fn pixel_luminance() {
        let white = Pixel::rgb(255, 255, 255);
        assert!((white.luminance() - 1.0).abs() < 0.01);
        let black = Pixel::rgb(0, 0, 0);
        assert!((black.luminance() - 0.0).abs() < 0.01);
    }

    #[test]
    fn mask_region_contains() {
        let mask = MaskRegion::new(10, 20, 5, 5);
        assert!(mask.contains(10, 20));
        assert!(mask.contains(14, 24));
        assert!(!mask.contains(15, 20));
        assert!(!mask.contains(9, 20));
    }
}
