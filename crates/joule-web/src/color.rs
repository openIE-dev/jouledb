//! Color manipulation and palette generation.
//!
//! Replaces chroma.js, polished, and tinycolor with a pure-Rust color
//! library supporting RGB, HSL, hex, WCAG analysis, and palette generation.

use std::fmt;

// ── Color ───────────────────────────────────────────────────────

/// An RGBA color.
#[derive(Debug, Clone)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: f64,
}

impl PartialEq for Color {
    fn eq(&self, other: &Self) -> bool {
        self.r == other.r
            && self.g == other.g
            && self.b == other.b
            && (self.a - other.a).abs() < 1e-6
    }
}

impl Eq for Color {}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

// ── Constructors ────────────────────────────────────────────────

impl Color {
    /// Create an opaque RGB color.
    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    /// Create an RGBA color.
    pub fn rgba(r: u8, g: u8, b: u8, a: f64) -> Self {
        Self {
            r,
            g,
            b,
            a: a.clamp(0.0, 1.0),
        }
    }

    /// Parse a hex string: `#RGB`, `#RRGGBB`, or `#RRGGBBAA`.
    pub fn from_hex(hex: &str) -> Option<Self> {
        let h = hex.strip_prefix('#').unwrap_or(hex);
        match h.len() {
            3 => {
                let r = u8::from_str_radix(&h[0..1], 16).ok()? * 17;
                let g = u8::from_str_radix(&h[1..2], 16).ok()? * 17;
                let b = u8::from_str_radix(&h[2..3], 16).ok()? * 17;
                Some(Self::rgb(r, g, b))
            }
            6 => {
                let r = u8::from_str_radix(&h[0..2], 16).ok()?;
                let g = u8::from_str_radix(&h[2..4], 16).ok()?;
                let b = u8::from_str_radix(&h[4..6], 16).ok()?;
                Some(Self::rgb(r, g, b))
            }
            8 => {
                let r = u8::from_str_radix(&h[0..2], 16).ok()?;
                let g = u8::from_str_radix(&h[2..4], 16).ok()?;
                let b = u8::from_str_radix(&h[4..6], 16).ok()?;
                let a = u8::from_str_radix(&h[6..8], 16).ok()?;
                Some(Self::rgba(r, g, b, a as f64 / 255.0))
            }
            _ => None,
        }
    }

    /// Create a color from HSL (h: 0-360, s: 0-1, l: 0-1).
    pub fn from_hsl(h: f64, s: f64, l: f64) -> Self {
        Self::from_hsla(h, s, l, 1.0)
    }

    /// Create a color from HSLA.
    pub fn from_hsla(h: f64, s: f64, l: f64, a: f64) -> Self {
        let (r, g, b) = hsl_to_rgb(h, s.clamp(0.0, 1.0), l.clamp(0.0, 1.0));
        Self::rgba(r, g, b, a)
    }

    /// Lookup a CSS named color.
    pub fn from_name(name: &str) -> Option<Self> {
        let lower = name.to_lowercase();
        let (r, g, b) = match lower.as_str() {
            "black" => (0, 0, 0),
            "white" => (255, 255, 255),
            "red" => (255, 0, 0),
            "green" => (0, 128, 0),
            "blue" => (0, 0, 255),
            "yellow" => (255, 255, 0),
            "cyan" | "aqua" => (0, 255, 255),
            "magenta" | "fuchsia" => (255, 0, 255),
            "orange" => (255, 165, 0),
            "purple" => (128, 0, 128),
            "pink" => (255, 192, 203),
            "gray" | "grey" => (128, 128, 128),
            "brown" => (165, 42, 42),
            "navy" => (0, 0, 128),
            "teal" => (0, 128, 128),
            "silver" => (192, 192, 192),
            "gold" => (255, 215, 0),
            "lime" => (0, 255, 0),
            "coral" => (255, 127, 80),
            "salmon" => (250, 128, 114),
            _ => return None,
        };
        Some(Self::rgb(r, g, b))
    }
}

// ── Conversions ─────────────────────────────────────────────────

impl Color {
    /// Output as hex string `#rrggbb` or `#rrggbbaa` if alpha < 1.
    pub fn to_hex(&self) -> String {
        if (self.a - 1.0).abs() < 1e-6 {
            format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
        } else {
            let a_byte = (self.a * 255.0).round() as u8;
            format!("#{:02x}{:02x}{:02x}{:02x}", self.r, self.g, self.b, a_byte)
        }
    }

    /// Output as `rgb(r, g, b)` or `rgba(r, g, b, a)`.
    pub fn to_rgb_string(&self) -> String {
        if (self.a - 1.0).abs() < 1e-6 {
            format!("rgb({}, {}, {})", self.r, self.g, self.b)
        } else {
            format!("rgba({}, {}, {}, {:.2})", self.r, self.g, self.b, self.a)
        }
    }

    /// Convert to HSL as (h: 0-360, s: 0-1, l: 0-1).
    pub fn to_hsl(&self) -> (f64, f64, f64) {
        rgb_to_hsl(self.r, self.g, self.b)
    }

    /// Output as `hsl(h, s%, l%)`.
    pub fn to_hsl_string(&self) -> String {
        let (h, s, l) = self.to_hsl();
        format!("hsl({:.0}, {:.0}%, {:.0}%)", h, s * 100.0, l * 100.0)
    }
}

// ── Manipulation ────────────────────────────────────────────────

impl Color {
    /// Increase lightness by `amount` (0-1).
    pub fn lighten(&self, amount: f64) -> Color {
        let (h, s, l) = self.to_hsl();
        Color::from_hsla(h, s, (l + amount).clamp(0.0, 1.0), self.a)
    }

    /// Decrease lightness by `amount` (0-1).
    pub fn darken(&self, amount: f64) -> Color {
        let (h, s, l) = self.to_hsl();
        Color::from_hsla(h, s, (l - amount).clamp(0.0, 1.0), self.a)
    }

    /// Increase saturation by `amount` (0-1).
    pub fn saturate(&self, amount: f64) -> Color {
        let (h, s, l) = self.to_hsl();
        Color::from_hsla(h, (s + amount).clamp(0.0, 1.0), l, self.a)
    }

    /// Decrease saturation by `amount` (0-1).
    pub fn desaturate(&self, amount: f64) -> Color {
        let (h, s, l) = self.to_hsl();
        Color::from_hsla(h, (s - amount).clamp(0.0, 1.0), l, self.a)
    }

    /// Rotate the hue by `degrees`.
    pub fn rotate_hue(&self, degrees: f64) -> Color {
        let (h, s, l) = self.to_hsl();
        let new_h = (h + degrees).rem_euclid(360.0);
        Color::from_hsla(new_h, s, l, self.a)
    }

    /// Invert the color (255 − channel).
    pub fn invert(&self) -> Color {
        Color::rgba(255 - self.r, 255 - self.g, 255 - self.b, self.a)
    }

    /// Convert to grayscale (fully desaturate).
    pub fn grayscale(&self) -> Color {
        self.desaturate(1.0)
    }

    /// Return a copy with a different alpha.
    pub fn with_alpha(&self, a: f64) -> Color {
        Color::rgba(self.r, self.g, self.b, a)
    }

    /// Linear interpolation between `self` and `other`.
    /// `ratio` of 0.0 = self, 1.0 = other.
    pub fn mix(&self, other: &Color, ratio: f64) -> Color {
        let t = ratio.clamp(0.0, 1.0);
        let r = lerp_u8(self.r, other.r, t);
        let g = lerp_u8(self.g, other.g, t);
        let b = lerp_u8(self.b, other.b, t);
        let a = self.a + (other.a - self.a) * t;
        Color::rgba(r, g, b, a)
    }
}

// ── Analysis ────────────────────────────────────────────────────

impl Color {
    /// WCAG relative luminance.
    pub fn luminance(&self) -> f64 {
        let r = srgb_channel(self.r);
        let g = srgb_channel(self.g);
        let b = srgb_channel(self.b);
        0.2126 * r + 0.7152 * g + 0.0722 * b
    }

    /// WCAG contrast ratio between two colors (1 to 21).
    pub fn contrast_ratio(&self, other: &Color) -> f64 {
        let l1 = self.luminance();
        let l2 = other.luminance();
        let lighter = l1.max(l2);
        let darker = l1.min(l2);
        (lighter + 0.05) / (darker + 0.05)
    }

    /// True if the color is perceived as light.
    pub fn is_light(&self) -> bool {
        self.luminance() > 0.5
    }

    /// True if the color is perceived as dark.
    pub fn is_dark(&self) -> bool {
        !self.is_light()
    }
}

// ── Palette ─────────────────────────────────────────────────────

impl Color {
    /// Complementary color (180° rotation).
    pub fn complementary(&self) -> Color {
        self.rotate_hue(180.0)
    }

    /// Triadic colors (±120° rotation).
    pub fn triadic(&self) -> (Color, Color) {
        (self.rotate_hue(120.0), self.rotate_hue(240.0))
    }

    /// Analogous colors spread evenly around `spread` degrees.
    pub fn analogous(&self, count: usize, spread: f64) -> Vec<Color> {
        if count == 0 {
            return Vec::new();
        }
        let step = spread / count as f64;
        let start = -(spread / 2.0);
        (0..count)
            .map(|i| self.rotate_hue(start + step * i as f64))
            .collect()
    }

    /// Generate a gradient of `steps` colors from `from` to `to`.
    pub fn gradient(from: &Color, to: &Color, steps: usize) -> Vec<Color> {
        if steps == 0 {
            return Vec::new();
        }
        if steps == 1 {
            return vec![from.clone()];
        }
        (0..steps)
            .map(|i| {
                let t = i as f64 / (steps - 1) as f64;
                from.mix(to, t)
            })
            .collect()
    }
}

// ── Internal helpers ────────────────────────────────────────────

fn srgb_channel(c: u8) -> f64 {
    let v = c as f64 / 255.0;
    if v <= 0.03928 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn lerp_u8(a: u8, b: u8, t: f64) -> u8 {
    let result = a as f64 + (b as f64 - a as f64) * t;
    result.round().clamp(0.0, 255.0) as u8
}

fn rgb_to_hsl(r: u8, g: u8, b: u8) -> (f64, f64, f64) {
    let rf = r as f64 / 255.0;
    let gf = g as f64 / 255.0;
    let bf = b as f64 / 255.0;
    let max = rf.max(gf).max(bf);
    let min = rf.min(gf).min(bf);
    let l = (max + min) / 2.0;

    if (max - min).abs() < 1e-10 {
        return (0.0, 0.0, l);
    }

    let d = max - min;
    let s = if l > 0.5 {
        d / (2.0 - max - min)
    } else {
        d / (max + min)
    };

    let h = if (max - rf).abs() < 1e-10 {
        let mut h = (gf - bf) / d;
        if gf < bf {
            h += 6.0;
        }
        h
    } else if (max - gf).abs() < 1e-10 {
        (bf - rf) / d + 2.0
    } else {
        (rf - gf) / d + 4.0
    };

    (h * 60.0, s, l)
}

fn hsl_to_rgb(h: f64, s: f64, l: f64) -> (u8, u8, u8) {
    if s.abs() < 1e-10 {
        let v = (l * 255.0).round() as u8;
        return (v, v, v);
    }

    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let h_norm = h / 360.0;

    let r = hue_to_rgb(p, q, h_norm + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, h_norm);
    let b = hue_to_rgb(p, q, h_norm - 1.0 / 3.0);

    (
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
    )
}

fn hue_to_rgb(p: f64, q: f64, mut t: f64) -> f64 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 1.0 / 2.0 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    p
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_hex_rgb_shorthand() {
        let c = Color::from_hex("#fff").unwrap();
        assert_eq!(c.r, 255);
        assert_eq!(c.g, 255);
        assert_eq!(c.b, 255);
    }

    #[test]
    fn from_hex_rrggbb() {
        let c = Color::from_hex("#ff8000").unwrap();
        assert_eq!(c.r, 255);
        assert_eq!(c.g, 128);
        assert_eq!(c.b, 0);
    }

    #[test]
    fn to_hex_roundtrip() {
        let c = Color::rgb(10, 20, 30);
        let hex = c.to_hex();
        let c2 = Color::from_hex(&hex).unwrap();
        assert_eq!(c, c2);
    }

    #[test]
    fn hsl_roundtrip() {
        let c = Color::rgb(200, 100, 50);
        let (h, s, l) = c.to_hsl();
        let c2 = Color::from_hsl(h, s, l);
        // Allow ±1 for rounding
        assert!((c.r as i16 - c2.r as i16).unsigned_abs() <= 1);
        assert!((c.g as i16 - c2.g as i16).unsigned_abs() <= 1);
        assert!((c.b as i16 - c2.b as i16).unsigned_abs() <= 1);
    }

    #[test]
    fn lighten_darken() {
        let c = Color::from_hsl(0.0, 1.0, 0.5);
        let lighter = c.lighten(0.2);
        let (_, _, l) = lighter.to_hsl();
        assert!((l - 0.7).abs() < 0.05);

        let darker = c.darken(0.2);
        let (_, _, l2) = darker.to_hsl();
        assert!((l2 - 0.3).abs() < 0.05);
    }

    #[test]
    fn mix_midpoint() {
        let black = Color::rgb(0, 0, 0);
        let white = Color::rgb(255, 255, 255);
        let mid = black.mix(&white, 0.5);
        // Should be approximately 128
        assert!((mid.r as i16 - 128).unsigned_abs() <= 1);
        assert!((mid.g as i16 - 128).unsigned_abs() <= 1);
        assert!((mid.b as i16 - 128).unsigned_abs() <= 1);
    }

    #[test]
    fn contrast_ratio_black_white() {
        let black = Color::rgb(0, 0, 0);
        let white = Color::rgb(255, 255, 255);
        let ratio = white.contrast_ratio(&black);
        assert!((ratio - 21.0).abs() < 0.1);
    }

    #[test]
    fn is_light_dark() {
        assert!(Color::rgb(255, 255, 255).is_light());
        assert!(Color::rgb(0, 0, 0).is_dark());
    }

    #[test]
    fn invert_color() {
        let c = Color::rgb(255, 0, 128);
        let inv = c.invert();
        assert_eq!(inv.r, 0);
        assert_eq!(inv.g, 255);
        assert_eq!(inv.b, 127);
    }

    #[test]
    fn named_colors() {
        assert!(Color::from_name("red").is_some());
        assert!(Color::from_name("blue").is_some());
        assert!(Color::from_name("coral").is_some());
        assert!(Color::from_name("nonexistent").is_none());
    }

    #[test]
    fn complementary_rotation() {
        let c = Color::from_hsl(60.0, 1.0, 0.5);
        let comp = c.complementary();
        let (h, _, _) = comp.to_hsl();
        assert!((h - 240.0).abs() < 2.0);
    }

    #[test]
    fn gradient_correct_count() {
        let a = Color::rgb(0, 0, 0);
        let b = Color::rgb(255, 255, 255);
        let grad = Color::gradient(&a, &b, 5);
        assert_eq!(grad.len(), 5);
        assert_eq!(grad[0], a);
        assert_eq!(grad[4], b);
    }

    #[test]
    fn from_hex_invalid() {
        assert!(Color::from_hex("#gg").is_none());
        assert!(Color::from_hex("#12345").is_none());
        assert!(Color::from_hex("").is_none());
    }

    #[test]
    fn alpha_handling() {
        let c = Color::from_hex("#ff000080").unwrap();
        assert!((c.a - 0.502).abs() < 0.01);
        let hex = c.to_hex();
        assert!(hex.len() == 9); // #rrggbbaa
    }
}
