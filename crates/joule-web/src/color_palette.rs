//! Color palette generator: harmonies, tints/shades, dominant color extraction,
//! and accessibility-checked palettes.
//!
//! Supports complementary, analogous, triadic, split-complementary, and tetradic
//! color harmonies. Includes k-means clustering on RGB pixel data for dominant
//! color extraction from image buffers.

use std::fmt;

// ── Core Color Types ────────────────────────────────────────────

/// An RGBA color with 8-bit channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rgba {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Rgba {
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    pub fn to_hsl(self) -> Hsl {
        let r = self.r as f64 / 255.0;
        let g = self.g as f64 / 255.0;
        let b = self.b as f64 / 255.0;

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let delta = max - min;

        let l = (max + min) / 2.0;

        if delta < 1e-10 {
            return Hsl { h: 0.0, s: 0.0, l };
        }

        let s = if l < 0.5 {
            delta / (max + min)
        } else {
            delta / (2.0 - max - min)
        };

        let h = if (max - r).abs() < 1e-10 {
            ((g - b) / delta) % 6.0
        } else if (max - g).abs() < 1e-10 {
            (b - r) / delta + 2.0
        } else {
            (r - g) / delta + 4.0
        };

        let h = (h * 60.0 + 360.0) % 360.0;

        Hsl { h, s, l }
    }

    /// Hex string like `#ff8800`.
    pub fn to_hex(self) -> String {
        if self.a == 255 {
            format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
        } else {
            format!("#{:02x}{:02x}{:02x}{:02x}", self.r, self.g, self.b, self.a)
        }
    }

    /// Parse from hex string (3, 6, or 8 hex digits, optional `#`).
    pub fn from_hex(hex: &str) -> Option<Self> {
        let hex = hex.trim_start_matches('#');
        match hex.len() {
            3 => {
                let r = u8::from_str_radix(&hex[0..1], 16).ok()? * 17;
                let g = u8::from_str_radix(&hex[1..2], 16).ok()? * 17;
                let b = u8::from_str_radix(&hex[2..3], 16).ok()? * 17;
                Some(Rgba::rgb(r, g, b))
            }
            6 => {
                let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
                let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
                let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
                Some(Rgba::rgb(r, g, b))
            }
            8 => {
                let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
                let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
                let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
                let a = u8::from_str_radix(&hex[6..8], 16).ok()?;
                Some(Rgba::new(r, g, b, a))
            }
            _ => None,
        }
    }

    /// Relative luminance per WCAG 2.x.
    pub fn relative_luminance(self) -> f64 {
        fn linearize(c: u8) -> f64 {
            let v = c as f64 / 255.0;
            if v <= 0.04045 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        }
        0.2126 * linearize(self.r) + 0.7152 * linearize(self.g) + 0.0722 * linearize(self.b)
    }

    /// WCAG contrast ratio between two colors (1.0..21.0).
    pub fn contrast_ratio(self, other: Rgba) -> f64 {
        let l1 = self.relative_luminance();
        let l2 = other.relative_luminance();
        let (lighter, darker) = if l1 > l2 { (l1, l2) } else { (l2, l1) };
        (lighter + 0.05) / (darker + 0.05)
    }
}

impl fmt::Display for Rgba {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

/// HSL color (hue 0..360, saturation 0..1, lightness 0..1).
#[derive(Debug, Clone, Copy)]
pub struct Hsl {
    pub h: f64,
    pub s: f64,
    pub l: f64,
}

impl Hsl {
    pub fn new(h: f64, s: f64, l: f64) -> Self {
        Self {
            h: h % 360.0,
            s: s.clamp(0.0, 1.0),
            l: l.clamp(0.0, 1.0),
        }
    }

    pub fn to_rgba(self) -> Rgba {
        if self.s < 1e-10 {
            let v = (self.l * 255.0).round() as u8;
            return Rgba::rgb(v, v, v);
        }

        let c = (1.0 - (2.0 * self.l - 1.0).abs()) * self.s;
        let x = c * (1.0 - ((self.h / 60.0) % 2.0 - 1.0).abs());
        let m = self.l - c / 2.0;

        let (r1, g1, b1) = match self.h as u32 {
            0..60 => (c, x, 0.0),
            60..120 => (x, c, 0.0),
            120..180 => (0.0, c, x),
            180..240 => (0.0, x, c),
            240..300 => (x, 0.0, c),
            _ => (c, 0.0, x),
        };

        Rgba::rgb(
            ((r1 + m) * 255.0).round() as u8,
            ((g1 + m) * 255.0).round() as u8,
            ((b1 + m) * 255.0).round() as u8,
        )
    }

    /// Rotate hue by `degrees`.
    pub fn rotate(self, degrees: f64) -> Self {
        Self::new((self.h + degrees + 360.0) % 360.0, self.s, self.l)
    }
}

// ── Harmony Types ───────────────────────────────────────────────

/// Which color harmony to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Harmony {
    Complementary,
    Analogous,
    Triadic,
    SplitComplementary,
    Tetradic,
}

/// A named palette of colors.
#[derive(Debug, Clone)]
pub struct Palette {
    pub name: String,
    pub colors: Vec<Rgba>,
}

impl Palette {
    pub fn new(name: impl Into<String>, colors: Vec<Rgba>) -> Self {
        Self {
            name: name.into(),
            colors,
        }
    }

    pub fn len(&self) -> usize {
        self.colors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.colors.is_empty()
    }
}

// ── Harmony Generation ──────────────────────────────────────────

/// Generate a harmony palette from a base color.
pub fn harmony(base: Rgba, kind: Harmony) -> Palette {
    let hsl = base.to_hsl();
    let colors = match kind {
        Harmony::Complementary => {
            vec![base, hsl.rotate(180.0).to_rgba()]
        }
        Harmony::Analogous => {
            vec![
                hsl.rotate(-30.0).to_rgba(),
                base,
                hsl.rotate(30.0).to_rgba(),
            ]
        }
        Harmony::Triadic => {
            vec![
                base,
                hsl.rotate(120.0).to_rgba(),
                hsl.rotate(240.0).to_rgba(),
            ]
        }
        Harmony::SplitComplementary => {
            vec![
                base,
                hsl.rotate(150.0).to_rgba(),
                hsl.rotate(210.0).to_rgba(),
            ]
        }
        Harmony::Tetradic => {
            vec![
                base,
                hsl.rotate(90.0).to_rgba(),
                hsl.rotate(180.0).to_rgba(),
                hsl.rotate(270.0).to_rgba(),
            ]
        }
    };

    let name = format!("{:?}", kind);
    Palette::new(name, colors)
}

// ── Tints & Shades ──────────────────────────────────────────────

/// Generate `count` tints (lighter versions) of a color.
pub fn tints(base: Rgba, count: usize) -> Vec<Rgba> {
    let hsl = base.to_hsl();
    (1..=count)
        .map(|i| {
            let factor = i as f64 / (count as f64 + 1.0);
            let l = hsl.l + (1.0 - hsl.l) * factor;
            Hsl::new(hsl.h, hsl.s, l).to_rgba()
        })
        .collect()
}

/// Generate `count` shades (darker versions) of a color.
pub fn shades(base: Rgba, count: usize) -> Vec<Rgba> {
    let hsl = base.to_hsl();
    (1..=count)
        .map(|i| {
            let factor = i as f64 / (count as f64 + 1.0);
            let l = hsl.l * (1.0 - factor);
            Hsl::new(hsl.h, hsl.s, l).to_rgba()
        })
        .collect()
}

/// Full scale: shades + base + tints.
pub fn color_scale(base: Rgba, steps: usize) -> Vec<Rgba> {
    let mut result = shades(base, steps);
    result.reverse();
    result.push(base);
    result.extend(tints(base, steps));
    result
}

// ── Dominant Color Extraction (k-means) ─────────────────────────

/// A pixel in an image buffer (R, G, B).
type Pixel = [u8; 3];

/// Extract `k` dominant colors from raw RGB pixel data using k-means.
pub fn dominant_colors(pixels: &[u8], k: usize, max_iterations: usize) -> Vec<Rgba> {
    if pixels.len() < 3 || k == 0 {
        return Vec::new();
    }

    let pixel_count = pixels.len() / 3;
    let actual_k = k.min(pixel_count);

    // Initialize centroids by sampling evenly spaced pixels.
    let mut centroids: Vec<[f64; 3]> = (0..actual_k)
        .map(|i| {
            let idx = (i * pixel_count / actual_k) * 3;
            [
                pixels[idx] as f64,
                pixels[idx + 1] as f64,
                pixels[idx + 2] as f64,
            ]
        })
        .collect();

    let mut assignments = vec![0usize; pixel_count];

    for _ in 0..max_iterations {
        let mut changed = false;

        // Assign each pixel to nearest centroid.
        for p in 0..pixel_count {
            let px: Pixel = [pixels[p * 3], pixels[p * 3 + 1], pixels[p * 3 + 2]];
            let mut best = 0;
            let mut best_dist = f64::MAX;
            for (ci, c) in centroids.iter().enumerate() {
                let dr = px[0] as f64 - c[0];
                let dg = px[1] as f64 - c[1];
                let db = px[2] as f64 - c[2];
                let dist = dr * dr + dg * dg + db * db;
                if dist < best_dist {
                    best_dist = dist;
                    best = ci;
                }
            }
            if assignments[p] != best {
                assignments[p] = best;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Recompute centroids.
        let mut sums = vec![[0.0f64; 3]; actual_k];
        let mut counts = vec![0usize; actual_k];
        for p in 0..pixel_count {
            let c = assignments[p];
            sums[c][0] += pixels[p * 3] as f64;
            sums[c][1] += pixels[p * 3 + 1] as f64;
            sums[c][2] += pixels[p * 3 + 2] as f64;
            counts[c] += 1;
        }
        for ci in 0..actual_k {
            if counts[ci] > 0 {
                centroids[ci][0] = sums[ci][0] / counts[ci] as f64;
                centroids[ci][1] = sums[ci][1] / counts[ci] as f64;
                centroids[ci][2] = sums[ci][2] / counts[ci] as f64;
            }
        }
    }

    // Sort by cluster size (largest first) and return.
    let mut cluster_sizes: Vec<(usize, usize)> = (0..actual_k).map(|i| (i, 0)).collect();
    for &a in &assignments {
        cluster_sizes[a].1 += 1;
    }
    cluster_sizes.sort_by(|a, b| b.1.cmp(&a.1));

    cluster_sizes
        .iter()
        .map(|&(ci, _)| {
            Rgba::rgb(
                centroids[ci][0].round() as u8,
                centroids[ci][1].round() as u8,
                centroids[ci][2].round() as u8,
            )
        })
        .collect()
}

// ── Accessibility-Checked Palette ───────────────────────────────

/// WCAG conformance level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WcagLevel {
    /// Contrast ratio >= 4.5 (normal text).
    Aa,
    /// Contrast ratio >= 7.0 (enhanced).
    Aaa,
    /// Contrast ratio >= 3.0 (large text / UI components).
    AaLarge,
}

impl WcagLevel {
    pub fn min_ratio(self) -> f64 {
        match self {
            WcagLevel::Aa => 4.5,
            WcagLevel::Aaa => 7.0,
            WcagLevel::AaLarge => 3.0,
        }
    }
}

/// A color pair that has been verified for WCAG contrast.
#[derive(Debug, Clone)]
pub struct AccessiblePair {
    pub foreground: Rgba,
    pub background: Rgba,
    pub contrast_ratio: f64,
    pub level: WcagLevel,
}

/// Check if two colors meet a given WCAG level.
pub fn meets_wcag(fg: Rgba, bg: Rgba, level: WcagLevel) -> bool {
    fg.contrast_ratio(bg) >= level.min_ratio()
}

/// Filter a palette to only pairs that meet the given WCAG level against `background`.
pub fn accessible_palette(
    colors: &[Rgba],
    background: Rgba,
    level: WcagLevel,
) -> Vec<AccessiblePair> {
    colors
        .iter()
        .filter_map(|fg| {
            let ratio = fg.contrast_ratio(background);
            if ratio >= level.min_ratio() {
                Some(AccessiblePair {
                    foreground: *fg,
                    background,
                    contrast_ratio: ratio,
                    level,
                })
            } else {
                None
            }
        })
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgba_to_hex() {
        assert_eq!(Rgba::rgb(255, 136, 0).to_hex(), "#ff8800");
        assert_eq!(Rgba::new(0, 0, 0, 128).to_hex(), "#00000080");
    }

    #[test]
    fn test_rgba_from_hex() {
        assert_eq!(Rgba::from_hex("#ff8800"), Some(Rgba::rgb(255, 136, 0)));
        assert_eq!(Rgba::from_hex("f80"), Some(Rgba::rgb(255, 136, 0)));
        assert_eq!(
            Rgba::from_hex("#00000080"),
            Some(Rgba::new(0, 0, 0, 128))
        );
        assert_eq!(Rgba::from_hex("xyz"), None);
    }

    #[test]
    fn test_hsl_roundtrip() {
        let colors = [
            Rgba::rgb(255, 0, 0),
            Rgba::rgb(0, 255, 0),
            Rgba::rgb(0, 0, 255),
            Rgba::rgb(128, 128, 128),
            Rgba::rgb(255, 255, 255),
            Rgba::rgb(0, 0, 0),
        ];
        for c in &colors {
            let hsl = c.to_hsl();
            let back = hsl.to_rgba();
            assert_eq!(back.r, c.r, "R mismatch for {c}");
            assert_eq!(back.g, c.g, "G mismatch for {c}");
            assert_eq!(back.b, c.b, "B mismatch for {c}");
        }
    }

    #[test]
    fn test_complementary() {
        let red = Rgba::rgb(255, 0, 0);
        let pal = harmony(red, Harmony::Complementary);
        assert_eq!(pal.colors.len(), 2);
        // Complement of red is cyan.
        let comp = pal.colors[1];
        assert_eq!(comp.r, 0);
        assert!(comp.g > 250);
        assert!(comp.b > 250);
    }

    #[test]
    fn test_triadic_count() {
        let base = Rgba::rgb(200, 100, 50);
        let pal = harmony(base, Harmony::Triadic);
        assert_eq!(pal.colors.len(), 3);
        assert_eq!(pal.colors[0], base);
    }

    #[test]
    fn test_analogous_count() {
        let pal = harmony(Rgba::rgb(0, 128, 255), Harmony::Analogous);
        assert_eq!(pal.colors.len(), 3);
    }

    #[test]
    fn test_split_complementary_count() {
        let pal = harmony(Rgba::rgb(100, 200, 50), Harmony::SplitComplementary);
        assert_eq!(pal.colors.len(), 3);
    }

    #[test]
    fn test_tetradic_count() {
        let pal = harmony(Rgba::rgb(80, 120, 200), Harmony::Tetradic);
        assert_eq!(pal.colors.len(), 4);
    }

    #[test]
    fn test_tints_get_lighter() {
        let base = Rgba::rgb(100, 50, 25);
        let t = tints(base, 3);
        assert_eq!(t.len(), 3);
        let base_l = base.to_hsl().l;
        for tint in &t {
            assert!(tint.to_hsl().l > base_l);
        }
    }

    #[test]
    fn test_shades_get_darker() {
        let base = Rgba::rgb(200, 150, 100);
        let s = shades(base, 3);
        assert_eq!(s.len(), 3);
        let base_l = base.to_hsl().l;
        for shade in &s {
            assert!(shade.to_hsl().l < base_l);
        }
    }

    #[test]
    fn test_color_scale_length() {
        let scale = color_scale(Rgba::rgb(128, 64, 32), 4);
        // 4 shades + base + 4 tints = 9.
        assert_eq!(scale.len(), 9);
    }

    #[test]
    fn test_dominant_colors_simple() {
        // 6 red pixels + 3 blue pixels.
        let mut pixels = Vec::new();
        for _ in 0..6 {
            pixels.extend_from_slice(&[255, 0, 0]);
        }
        for _ in 0..3 {
            pixels.extend_from_slice(&[0, 0, 255]);
        }
        let dom = dominant_colors(&pixels, 2, 20);
        assert_eq!(dom.len(), 2);
        // First should be red (majority).
        assert_eq!(dom[0], Rgba::rgb(255, 0, 0));
    }

    #[test]
    fn test_contrast_ratio_bw() {
        let black = Rgba::rgb(0, 0, 0);
        let white = Rgba::rgb(255, 255, 255);
        let ratio = black.contrast_ratio(white);
        assert!((ratio - 21.0).abs() < 0.1);
    }

    #[test]
    fn test_meets_wcag_aa() {
        let dark = Rgba::rgb(0, 0, 0);
        let white = Rgba::rgb(255, 255, 255);
        assert!(meets_wcag(dark, white, WcagLevel::Aa));
        assert!(meets_wcag(dark, white, WcagLevel::Aaa));
    }

    #[test]
    fn test_accessible_palette_filters() {
        let bg = Rgba::rgb(255, 255, 255);
        let colors = vec![
            Rgba::rgb(0, 0, 0),       // high contrast
            Rgba::rgb(200, 200, 200), // low contrast
            Rgba::rgb(50, 50, 50),    // high contrast
        ];
        let accessible = accessible_palette(&colors, bg, WcagLevel::Aa);
        assert!(accessible.len() >= 2);
        for pair in &accessible {
            assert!(pair.contrast_ratio >= 4.5);
        }
    }

    #[test]
    fn test_palette_display() {
        let pal = Palette::new("test", vec![Rgba::rgb(255, 0, 0)]);
        assert_eq!(pal.name, "test");
        assert_eq!(pal.len(), 1);
        assert!(!pal.is_empty());
    }
}
