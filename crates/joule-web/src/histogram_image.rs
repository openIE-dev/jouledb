//! Image histogram — per-channel histogram computation, cumulative histogram,
//! histogram equalization, contrast stretching, brightness analysis,
//! dominant color detection, histogram matching.
//!
//! Replaces JS histogram libraries with energy-efficient Rust implementations.

use serde::{Deserialize, Serialize};

// ── Histogram ────────────────────────────────────────────────────

/// A 256-bin histogram for a single channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Histogram {
    pub bins: [u32; 256],
    pub total: u32,
}

impl Histogram {
    pub fn new() -> Self {
        Self { bins: [0u32; 256], total: 0 }
    }

    /// Build histogram from raw channel data.
    pub fn from_channel(data: &[u8]) -> Self {
        let mut h = Self::new();
        for &v in data {
            h.bins[v as usize] += 1;
            h.total += 1;
        }
        h
    }

    /// Cumulative histogram.
    pub fn cumulative(&self) -> [u32; 256] {
        let mut cum = [0u32; 256];
        cum[0] = self.bins[0];
        for i in 1..256 {
            cum[i] = cum[i - 1] + self.bins[i];
        }
        cum
    }

    /// Normalized cumulative distribution (values in [0,1]).
    pub fn cdf_normalized(&self) -> [f64; 256] {
        let cum = self.cumulative();
        let mut cdf = [0.0f64; 256];
        if self.total == 0 {
            return cdf;
        }
        for i in 0..256 {
            cdf[i] = cum[i] as f64 / self.total as f64;
        }
        cdf
    }

    /// Minimum non-zero bin value.
    pub fn min_value(&self) -> u8 {
        for i in 0..256 {
            if self.bins[i] > 0 {
                return i as u8;
            }
        }
        0
    }

    /// Maximum non-zero bin value.
    pub fn max_value(&self) -> u8 {
        for i in (0..256).rev() {
            if self.bins[i] > 0 {
                return i as u8;
            }
        }
        255
    }

    /// Mean value.
    pub fn mean(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let sum: u64 = self.bins.iter().enumerate().map(|(i, &c)| i as u64 * c as u64).sum();
        sum as f64 / self.total as f64
    }

    /// Median value.
    pub fn median(&self) -> u8 {
        let half = self.total / 2;
        let mut acc = 0u32;
        for i in 0..256 {
            acc += self.bins[i];
            if acc >= half {
                return i as u8;
            }
        }
        255
    }

    /// Mode (most frequent value).
    pub fn mode(&self) -> u8 {
        let mut max_count = 0u32;
        let mut mode_val = 0u8;
        for i in 0..256 {
            if self.bins[i] > max_count {
                max_count = self.bins[i];
                mode_val = i as u8;
            }
        }
        mode_val
    }
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

// ── Multi-channel histograms ─────────────────────────────────────

/// Compute histograms for R, G, B channels from RGBA data.
pub fn compute_rgb_histograms(rgba_data: &[u8]) -> (Histogram, Histogram, Histogram) {
    let mut r = Histogram::new();
    let mut g = Histogram::new();
    let mut b = Histogram::new();
    let pixels = rgba_data.len() / 4;
    for i in 0..pixels {
        let off = i * 4;
        r.bins[rgba_data[off] as usize] += 1;
        r.total += 1;
        g.bins[rgba_data[off + 1] as usize] += 1;
        g.total += 1;
        b.bins[rgba_data[off + 2] as usize] += 1;
        b.total += 1;
    }
    (r, g, b)
}

/// Compute luminance histogram from RGBA data.
pub fn compute_luminance_histogram(rgba_data: &[u8]) -> Histogram {
    let mut h = Histogram::new();
    let pixels = rgba_data.len() / 4;
    for i in 0..pixels {
        let off = i * 4;
        let lum = (rgba_data[off] as u32 * 299
            + rgba_data[off + 1] as u32 * 587
            + rgba_data[off + 2] as u32 * 114) / 1000;
        h.bins[lum as usize] += 1;
        h.total += 1;
    }
    h
}

// ── Histogram equalization ───────────────────────────────────────

/// Build a lookup table for histogram equalization.
pub fn equalization_lut(hist: &Histogram) -> [u8; 256] {
    let cdf = hist.cumulative();
    let cdf_min = cdf.iter().copied().find(|v| *v > 0).unwrap_or(0);
    let mut lut = [0u8; 256];
    if hist.total <= cdf_min {
        return lut;
    }
    let denom = (hist.total - cdf_min) as f64;
    for i in 0..256 {
        let v = ((cdf[i] as f64 - cdf_min as f64) / denom * 255.0).round().clamp(0.0, 255.0);
        lut[i] = v as u8;
    }
    lut
}

/// Apply histogram equalization to grayscale data in-place.
pub fn equalize_grayscale(data: &mut [u8]) {
    let hist = Histogram::from_channel(data);
    let lut = equalization_lut(&hist);
    for v in data.iter_mut() {
        *v = lut[*v as usize];
    }
}

/// Apply histogram equalization per-channel to RGBA data.
pub fn equalize_rgba(data: &mut [u8]) {
    let (rh, gh, bh) = compute_rgb_histograms(data);
    let rl = equalization_lut(&rh);
    let gl = equalization_lut(&gh);
    let bl = equalization_lut(&bh);
    let pixels = data.len() / 4;
    for i in 0..pixels {
        let off = i * 4;
        data[off] = rl[data[off] as usize];
        data[off + 1] = gl[data[off + 1] as usize];
        data[off + 2] = bl[data[off + 2] as usize];
    }
}

// ── Contrast stretching ─────────────────────────────────────────

/// Linear contrast stretch to map [min, max] → [0, 255].
pub fn contrast_stretch(data: &mut [u8]) {
    if data.is_empty() {
        return;
    }
    let min_v = *data.iter().min().unwrap();
    let max_v = *data.iter().max().unwrap();
    if min_v == max_v {
        return;
    }
    let range = (max_v - min_v) as f64;
    for v in data.iter_mut() {
        *v = ((*v as f64 - min_v as f64) / range * 255.0).round() as u8;
    }
}

// ── Brightness analysis ─────────────────────────────────────────

/// Brightness classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrightnessLevel {
    VeryDark,
    Dark,
    Medium,
    Bright,
    VeryBright,
}

/// Classify image brightness from luminance histogram.
pub fn classify_brightness(hist: &Histogram) -> BrightnessLevel {
    let mean = hist.mean();
    if mean < 50.0 {
        BrightnessLevel::VeryDark
    } else if mean < 100.0 {
        BrightnessLevel::Dark
    } else if mean < 155.0 {
        BrightnessLevel::Medium
    } else if mean < 205.0 {
        BrightnessLevel::Bright
    } else {
        BrightnessLevel::VeryBright
    }
}

// ── Dominant color detection ─────────────────────────────────────

/// A quantized color bucket.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ColorBucket {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub count: u32,
}

/// Find top-N dominant colors by quantizing to `levels` per channel.
pub fn dominant_colors(rgba_data: &[u8], levels: u8, top_n: usize) -> Vec<ColorBucket> {
    use std::collections::HashMap;
    let levels = levels.max(2);
    let step = 256u16 / levels as u16;
    let mut buckets: HashMap<(u8, u8, u8), u32> = HashMap::new();
    let pixels = rgba_data.len() / 4;
    for i in 0..pixels {
        let off = i * 4;
        let qr = ((rgba_data[off] as u16 / step) * step + step / 2).min(255) as u8;
        let qg = ((rgba_data[off + 1] as u16 / step) * step + step / 2).min(255) as u8;
        let qb = ((rgba_data[off + 2] as u16 / step) * step + step / 2).min(255) as u8;
        *buckets.entry((qr, qg, qb)).or_insert(0) += 1;
    }
    let mut sorted: Vec<_> = buckets
        .into_iter()
        .map(|((r, g, b), count)| ColorBucket { r, g, b, count })
        .collect();
    sorted.sort_by(|a, b| b.count.cmp(&a.count));
    sorted.truncate(top_n);
    sorted
}

// ── Histogram matching ───────────────────────────────────────────

/// Build a LUT that maps `src` histogram to match `target` histogram.
pub fn matching_lut(src: &Histogram, target: &Histogram) -> [u8; 256] {
    let src_cdf = src.cdf_normalized();
    let tgt_cdf = target.cdf_normalized();
    let mut lut = [0u8; 256];
    for i in 0..256 {
        let mut best = 0usize;
        let mut best_diff = f64::MAX;
        for j in 0..256 {
            let diff = (src_cdf[i] - tgt_cdf[j]).abs();
            if diff < best_diff {
                best_diff = diff;
                best = j;
            }
        }
        lut[i] = best as u8;
    }
    lut
}

/// Apply histogram matching to grayscale data.
pub fn match_histogram(data: &mut [u8], target: &Histogram) {
    let src = Histogram::from_channel(data);
    let lut = matching_lut(&src, target);
    for v in data.iter_mut() {
        *v = lut[*v as usize];
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_from_channel() {
        let data = vec![0, 0, 0, 128, 128, 255];
        let h = Histogram::from_channel(&data);
        assert_eq!(h.bins[0], 3);
        assert_eq!(h.bins[128], 2);
        assert_eq!(h.bins[255], 1);
        assert_eq!(h.total, 6);
    }

    #[test]
    fn test_cumulative() {
        let data = vec![0, 1, 1, 2, 2, 2];
        let h = Histogram::from_channel(&data);
        let cum = h.cumulative();
        assert_eq!(cum[0], 1);
        assert_eq!(cum[1], 3);
        assert_eq!(cum[2], 6);
    }

    #[test]
    fn test_min_max() {
        let data = vec![50, 100, 200];
        let h = Histogram::from_channel(&data);
        assert_eq!(h.min_value(), 50);
        assert_eq!(h.max_value(), 200);
    }

    #[test]
    fn test_mean_median_mode() {
        let data = vec![10, 10, 10, 20, 20, 30];
        let h = Histogram::from_channel(&data);
        let mean = h.mean();
        assert!((mean - 16.666).abs() < 1.0);
        assert_eq!(h.median(), 10);
        assert_eq!(h.mode(), 10);
    }

    #[test]
    fn test_equalize_grayscale() {
        let mut data = vec![52, 55, 61, 59, 70, 61, 76, 61];
        equalize_grayscale(&mut data);
        // After equalization, values should be spread across [0, 255]
        let min = *data.iter().min().unwrap();
        let max = *data.iter().max().unwrap();
        assert!(max > min);
    }

    #[test]
    fn test_contrast_stretch() {
        let mut data = vec![100, 120, 140, 160];
        contrast_stretch(&mut data);
        assert_eq!(*data.iter().min().unwrap(), 0);
        assert_eq!(*data.iter().max().unwrap(), 255);
    }

    #[test]
    fn test_contrast_stretch_uniform() {
        let mut data = vec![50, 50, 50];
        contrast_stretch(&mut data);
        assert!(data.iter().all(|v| *v == 50)); // No change for uniform
    }

    #[test]
    fn test_rgb_histograms() {
        let rgba = vec![
            255, 0, 0, 255,
            0, 255, 0, 255,
            0, 0, 255, 255,
        ];
        let (r, g, b) = compute_rgb_histograms(&rgba);
        assert_eq!(r.bins[255], 1);
        assert_eq!(g.bins[255], 1);
        assert_eq!(b.bins[255], 1);
    }

    #[test]
    fn test_luminance_histogram() {
        let rgba = vec![255, 255, 255, 255, 0, 0, 0, 255];
        let h = compute_luminance_histogram(&rgba);
        assert_eq!(h.bins[255], 1);
        assert_eq!(h.bins[0], 1);
    }

    #[test]
    fn test_classify_brightness() {
        let dark_data: Vec<u8> = vec![10; 100];
        let dark_h = Histogram::from_channel(&dark_data);
        assert_eq!(classify_brightness(&dark_h), BrightnessLevel::VeryDark);

        let bright_data: Vec<u8> = vec![220; 100];
        let bright_h = Histogram::from_channel(&bright_data);
        assert_eq!(classify_brightness(&bright_h), BrightnessLevel::VeryBright);
    }

    #[test]
    fn test_dominant_colors() {
        let mut rgba = Vec::new();
        for _ in 0..100 {
            rgba.extend_from_slice(&[255, 0, 0, 255]); // red
        }
        for _ in 0..50 {
            rgba.extend_from_slice(&[0, 0, 255, 255]); // blue
        }
        let dom = dominant_colors(&rgba, 4, 5);
        assert!(!dom.is_empty());
        assert!(dom[0].count >= dom.last().unwrap().count);
    }

    #[test]
    fn test_histogram_matching() {
        let mut data = vec![10, 20, 30, 40, 50];
        let target = Histogram::from_channel(&[200, 210, 220, 230, 240]);
        match_histogram(&mut data, &target);
        // After matching, values should be shifted toward target range
        let mean: f64 = data.iter().map(|v| *v as f64).sum::<f64>() / data.len() as f64;
        assert!(mean > 100.0);
    }

    #[test]
    fn test_equalize_rgba() {
        let mut data = vec![
            50, 50, 50, 255,
            100, 100, 100, 255,
            150, 150, 150, 255,
            200, 200, 200, 255,
        ];
        equalize_rgba(&mut data);
        // Alpha should be preserved
        assert_eq!(data[3], 255);
        assert_eq!(data[7], 255);
    }

    #[test]
    fn test_cdf_normalized() {
        let data = vec![0, 1, 2, 3];
        let h = Histogram::from_channel(&data);
        let cdf = h.cdf_normalized();
        assert!((cdf[0] - 0.25).abs() < 0.01);
        assert!((cdf[3] - 1.0).abs() < 0.01);
    }
}
