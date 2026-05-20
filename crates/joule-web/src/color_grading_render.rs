// color_grading_render.rs — Color grading for the rendering pipeline.
//
// Implements 3D LUT with trilinear interpolation, LUT generation from color
// adjustments, temperature/tint, saturation, contrast, brightness, hue shift,
// lift/gamma/gain, channel mixer, split-toning, and LUT serialization.

use std::fmt;

/// RGB color in [0,1] range for grading.
#[derive(Clone, Debug, PartialEq)]
pub struct GradeColor {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl GradeColor {
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }

    pub fn black() -> Self {
        Self { r: 0.0, g: 0.0, b: 0.0 }
    }

    pub fn white() -> Self {
        Self { r: 1.0, g: 1.0, b: 1.0 }
    }

    pub fn gray(v: f64) -> Self {
        Self { r: v, g: v, b: v }
    }

    pub fn luminance(&self) -> f64 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
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

    pub fn clamp01(&self) -> Self {
        Self {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
        }
    }

    /// Convert to HSL (returns h in [0,360), s and l in [0,1]).
    pub fn to_hsl(&self) -> (f64, f64, f64) {
        let max = self.r.max(self.g).max(self.b);
        let min = self.r.min(self.g).min(self.b);
        let l = (max + min) / 2.0;
        if (max - min).abs() < 1e-10 {
            return (0.0, 0.0, l);
        }
        let d = max - min;
        let s = if l > 0.5 { d / (2.0 - max - min) } else { d / (max + min) };
        let h = if (max - self.r).abs() < 1e-10 {
            let mut h = (self.g - self.b) / d;
            if self.g < self.b { h += 6.0; }
            h
        } else if (max - self.g).abs() < 1e-10 {
            (self.b - self.r) / d + 2.0
        } else {
            (self.r - self.g) / d + 4.0
        };
        (h * 60.0, s, l)
    }

    /// Construct from HSL (h in [0,360), s and l in [0,1]).
    pub fn from_hsl(h: f64, s: f64, l: f64) -> Self {
        if s < 1e-10 {
            return Self::gray(l);
        }
        let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
        let p = 2.0 * l - q;
        let h_norm = h / 360.0;

        fn hue_to_rgb(p: f64, q: f64, mut t: f64) -> f64 {
            if t < 0.0 { t += 1.0; }
            if t > 1.0 { t -= 1.0; }
            if t < 1.0 / 6.0 { return p + (q - p) * 6.0 * t; }
            if t < 0.5 { return q; }
            if t < 2.0 / 3.0 { return p + (q - p) * (2.0 / 3.0 - t) * 6.0; }
            p
        }

        Self {
            r: hue_to_rgb(p, q, h_norm + 1.0 / 3.0),
            g: hue_to_rgb(p, q, h_norm),
            b: hue_to_rgb(p, q, h_norm - 1.0 / 3.0),
        }
    }
}

/// 3D LUT for color grading.
#[derive(Clone, Debug)]
pub struct Lut3D {
    /// Resolution of the cube (e.g. 32 or 64).
    pub size: usize,
    /// Color data in row-major order: [r + g*size + b*size*size].
    pub data: Vec<GradeColor>,
}

impl Lut3D {
    /// Create an identity LUT (no color transform).
    pub fn identity(size: usize) -> Self {
        let total = size * size * size;
        let mut data = Vec::with_capacity(total);
        let inv = 1.0 / (size - 1).max(1) as f64;
        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    data.push(GradeColor::new(
                        r as f64 * inv,
                        g as f64 * inv,
                        b as f64 * inv,
                    ));
                }
            }
        }
        Self { size, data }
    }

    fn index(&self, r: usize, g: usize, b: usize) -> usize {
        r + g * self.size + b * self.size * self.size
    }

    pub fn get(&self, r: usize, g: usize, b: usize) -> &GradeColor {
        &self.data[self.index(r, g, b)]
    }

    pub fn set(&mut self, r: usize, g: usize, b: usize, c: GradeColor) {
        let idx = self.index(r, g, b);
        self.data[idx] = c;
    }

    /// Trilinear interpolation lookup.
    pub fn sample(&self, color: &GradeColor) -> GradeColor {
        let s = (self.size - 1) as f64;
        let r = (color.r * s).clamp(0.0, s);
        let g = (color.g * s).clamp(0.0, s);
        let b = (color.b * s).clamp(0.0, s);

        let r0 = r.floor() as usize;
        let g0 = g.floor() as usize;
        let b0 = b.floor() as usize;
        let r1 = (r0 + 1).min(self.size - 1);
        let g1 = (g0 + 1).min(self.size - 1);
        let b1 = (b0 + 1).min(self.size - 1);

        let fr = r - r.floor();
        let fg = g - g.floor();
        let fb = b - b.floor();

        // Trilinear: 8 corners
        let c000 = self.get(r0, g0, b0);
        let c100 = self.get(r1, g0, b0);
        let c010 = self.get(r0, g1, b0);
        let c110 = self.get(r1, g1, b0);
        let c001 = self.get(r0, g0, b1);
        let c101 = self.get(r1, g0, b1);
        let c011 = self.get(r0, g1, b1);
        let c111 = self.get(r1, g1, b1);

        let c00 = c000.lerp(c100, fr);
        let c10 = c010.lerp(c110, fr);
        let c01 = c001.lerp(c101, fr);
        let c11 = c011.lerp(c111, fr);

        let c0 = c00.lerp(&c10, fg);
        let c1 = c01.lerp(&c11, fg);

        c0.lerp(&c1, fb)
    }

    /// Serialize LUT data as a flat list of f64 triples (for export).
    pub fn serialize_flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.data.len() * 3);
        for c in &self.data {
            out.push(c.r);
            out.push(c.g);
            out.push(c.b);
        }
        out
    }

    /// Deserialize from a flat list of f64 triples.
    pub fn from_flat(size: usize, flat: &[f64]) -> Option<Self> {
        let expected = size * size * size * 3;
        if flat.len() != expected {
            return None;
        }
        let mut data = Vec::with_capacity(size * size * size);
        for chunk in flat.chunks(3) {
            data.push(GradeColor::new(chunk[0], chunk[1], chunk[2]));
        }
        Some(Self { size, data })
    }
}

impl fmt::Display for Lut3D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Lut3D(size={}, entries={})", self.size, self.data.len())
    }
}

/// Color grading adjustments.
#[derive(Clone, Debug)]
pub struct GradingParams {
    /// White balance temperature shift (negative = cool, positive = warm).
    pub temperature: f64,
    /// White balance tint shift (negative = green, positive = magenta).
    pub tint: f64,
    /// Saturation multiplier (1.0 = no change).
    pub saturation: f64,
    /// Contrast (1.0 = no change, >1 = more contrast).
    pub contrast: f64,
    /// Brightness offset.
    pub brightness: f64,
    /// Hue shift in degrees.
    pub hue_shift: f64,
    /// Lift (shadows) color offset.
    pub lift: GradeColor,
    /// Gamma (midtones) color multiplier.
    pub gamma: GradeColor,
    /// Gain (highlights) color multiplier.
    pub gain: GradeColor,
    /// Channel mixer: each row is (r, g, b) contribution to output channel.
    pub channel_mixer: [[f64; 3]; 3],
    /// Split-toning shadow color.
    pub split_shadow: GradeColor,
    /// Split-toning highlight color.
    pub split_highlight: GradeColor,
    /// Split-toning balance (-1 = all shadow, 0 = balanced, 1 = all highlight).
    pub split_balance: f64,
}

impl Default for GradingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            tint: 0.0,
            saturation: 1.0,
            contrast: 1.0,
            brightness: 0.0,
            hue_shift: 0.0,
            lift: GradeColor::black(),
            gamma: GradeColor::white(),
            gain: GradeColor::white(),
            channel_mixer: [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            split_shadow: GradeColor::black(),
            split_highlight: GradeColor::white(),
            split_balance: 0.0,
        }
    }
}

impl fmt::Display for GradingParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GradingParams(temp={:.1}, sat={:.2}, contrast={:.2})",
            self.temperature, self.saturation, self.contrast
        )
    }
}

/// Apply white balance (temperature/tint) to a color.
pub fn apply_white_balance(color: &GradeColor, temperature: f64, tint: f64) -> GradeColor {
    // Temperature: shift R/B balance
    // Tint: shift G/M balance
    let temp_shift = temperature * 0.01;
    let tint_shift = tint * 0.01;
    GradeColor {
        r: color.r + temp_shift,
        g: color.g + tint_shift,
        b: color.b - temp_shift,
    }
}

/// Apply saturation adjustment.
pub fn apply_saturation(color: &GradeColor, saturation: f64) -> GradeColor {
    let lum = color.luminance();
    let gray = GradeColor::gray(lum);
    gray.lerp(color, saturation)
}

/// Apply contrast (around midpoint 0.5).
pub fn apply_contrast(color: &GradeColor, contrast: f64) -> GradeColor {
    GradeColor {
        r: (color.r - 0.5) * contrast + 0.5,
        g: (color.g - 0.5) * contrast + 0.5,
        b: (color.b - 0.5) * contrast + 0.5,
    }
}

/// Apply brightness offset.
pub fn apply_brightness(color: &GradeColor, brightness: f64) -> GradeColor {
    color.add(&GradeColor::gray(brightness))
}

/// Apply hue shift (in degrees).
pub fn apply_hue_shift(color: &GradeColor, degrees: f64) -> GradeColor {
    let (h, s, l) = color.clamp01().to_hsl();
    let new_h = (h + degrees) % 360.0;
    let new_h = if new_h < 0.0 { new_h + 360.0 } else { new_h };
    GradeColor::from_hsl(new_h, s, l)
}

/// Apply lift/gamma/gain (ASC CDL style).
pub fn apply_lift_gamma_gain(color: &GradeColor, lift: &GradeColor, gamma: &GradeColor, gain: &GradeColor) -> GradeColor {
    // out = (color * gain + lift) ^ (1/gamma)
    let apply_ch = |c: f64, g: f64, l: f64, gam: f64| -> f64 {
        let gained = c * g + l;
        let safe_gam = if gam.abs() < 1e-10 { 1.0 } else { gam };
        if gained <= 0.0 {
            0.0
        } else {
            gained.powf(1.0 / safe_gam)
        }
    };

    GradeColor {
        r: apply_ch(color.r, gain.r, lift.r, gamma.r),
        g: apply_ch(color.g, gain.g, lift.g, gamma.g),
        b: apply_ch(color.b, gain.b, lift.b, gamma.b),
    }
}

/// Apply channel mixer.
pub fn apply_channel_mixer(color: &GradeColor, mixer: &[[f64; 3]; 3]) -> GradeColor {
    GradeColor {
        r: color.r * mixer[0][0] + color.g * mixer[0][1] + color.b * mixer[0][2],
        g: color.r * mixer[1][0] + color.g * mixer[1][1] + color.b * mixer[1][2],
        b: color.r * mixer[2][0] + color.g * mixer[2][1] + color.b * mixer[2][2],
    }
}

/// Apply split-toning.
pub fn apply_split_toning(
    color: &GradeColor,
    shadow_color: &GradeColor,
    highlight_color: &GradeColor,
    balance: f64,
) -> GradeColor {
    let lum = color.luminance();
    // Balance shifts the midpoint
    let midpoint = 0.5 + balance * 0.5;
    let shadow_factor = (1.0 - lum / midpoint.max(1e-10)).clamp(0.0, 1.0);
    let highlight_factor = ((lum - midpoint) / (1.0 - midpoint).max(1e-10)).clamp(0.0, 1.0);

    let shadow_tint = shadow_color.scale(shadow_factor * 0.2);
    let highlight_tint = highlight_color.scale(highlight_factor * 0.2);

    color.add(&shadow_tint).add(&highlight_tint)
}

/// Apply all grading adjustments to a single color.
pub fn grade_color(color: &GradeColor, params: &GradingParams) -> GradeColor {
    let mut c = apply_white_balance(color, params.temperature, params.tint);
    c = apply_saturation(&c, params.saturation);
    c = apply_contrast(&c, params.contrast);
    c = apply_brightness(&c, params.brightness);
    if params.hue_shift.abs() > 1e-6 {
        c = apply_hue_shift(&c.clamp01(), params.hue_shift);
    }
    c = apply_lift_gamma_gain(&c, &params.lift, &params.gamma, &params.gain);
    c = apply_channel_mixer(&c, &params.channel_mixer);
    c = apply_split_toning(&c, &params.split_shadow, &params.split_highlight, params.split_balance);
    c.clamp01()
}

/// Generate a 3D LUT from grading parameters.
pub fn generate_lut(size: usize, params: &GradingParams) -> Lut3D {
    let mut lut = Lut3D::identity(size);
    let inv = 1.0 / (size - 1).max(1) as f64;
    for bi in 0..size {
        for gi in 0..size {
            for ri in 0..size {
                let input = GradeColor::new(ri as f64 * inv, gi as f64 * inv, bi as f64 * inv);
                let output = grade_color(&input, params);
                lut.set(ri, gi, bi, output);
            }
        }
    }
    lut
}

/// Apply a 3D LUT to a color buffer.
pub fn apply_lut(
    pixels: &[GradeColor],
    width: usize,
    height: usize,
    lut: &Lut3D,
) -> Vec<GradeColor> {
    let _total = width * height;
    pixels.iter().map(|px| lut.sample(px)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn colors_approx_eq(a: &GradeColor, b: &GradeColor, eps: f64) -> bool {
        approx_eq(a.r, b.r, eps) && approx_eq(a.g, b.g, eps) && approx_eq(a.b, b.b, eps)
    }

    #[test]
    fn test_color_luminance() {
        let c = GradeColor::white();
        assert!(approx_eq(c.luminance(), 1.0, 1e-6));
    }

    #[test]
    fn test_color_lerp() {
        let a = GradeColor::black();
        let b = GradeColor::white();
        let mid = a.lerp(&b, 0.5);
        assert!(approx_eq(mid.r, 0.5, 1e-6));
    }

    #[test]
    fn test_hsl_roundtrip() {
        let colors = [
            GradeColor::new(1.0, 0.0, 0.0),
            GradeColor::new(0.0, 1.0, 0.0),
            GradeColor::new(0.0, 0.0, 1.0),
            GradeColor::new(0.5, 0.3, 0.7),
        ];
        for c in &colors {
            let (h, s, l) = c.to_hsl();
            let back = GradeColor::from_hsl(h, s, l);
            assert!(colors_approx_eq(c, &back, 1e-4),
                "HSL roundtrip failed: {:?} -> ({:.2},{:.2},{:.2}) -> {:?}", c, h, s, l, back);
        }
    }

    #[test]
    fn test_hsl_gray() {
        let c = GradeColor::gray(0.5);
        let (h, s, _l) = c.to_hsl();
        assert!(approx_eq(s, 0.0, 1e-6));
        // h is undefined for gray, just check s=0
        let _ = h;
    }

    #[test]
    fn test_identity_lut() {
        let lut = Lut3D::identity(8);
        assert_eq!(lut.data.len(), 512);
        // Identity should pass through
        let test = GradeColor::new(0.5, 0.3, 0.7);
        let out = lut.sample(&test);
        assert!(colors_approx_eq(&test, &out, 0.05));
    }

    #[test]
    fn test_lut_corners() {
        let lut = Lut3D::identity(16);
        let black = lut.sample(&GradeColor::black());
        assert!(colors_approx_eq(&black, &GradeColor::black(), 1e-6));
        let white = lut.sample(&GradeColor::white());
        assert!(colors_approx_eq(&white, &GradeColor::white(), 1e-6));
    }

    #[test]
    fn test_lut_serialize_roundtrip() {
        let lut = Lut3D::identity(4);
        let flat = lut.serialize_flat();
        assert_eq!(flat.len(), 4 * 4 * 4 * 3);
        let restored = Lut3D::from_flat(4, &flat).unwrap();
        assert_eq!(restored.data.len(), lut.data.len());
        for (a, b) in lut.data.iter().zip(restored.data.iter()) {
            assert!(colors_approx_eq(a, b, 1e-10));
        }
    }

    #[test]
    fn test_lut_from_flat_wrong_size() {
        let result = Lut3D::from_flat(4, &[0.0; 10]);
        assert!(result.is_none());
    }

    #[test]
    fn test_white_balance_neutral() {
        let c = GradeColor::new(0.5, 0.5, 0.5);
        let out = apply_white_balance(&c, 0.0, 0.0);
        assert!(colors_approx_eq(&c, &out, 1e-10));
    }

    #[test]
    fn test_white_balance_warm() {
        let c = GradeColor::gray(0.5);
        let out = apply_white_balance(&c, 50.0, 0.0);
        assert!(out.r > c.r);
        assert!(out.b < c.b);
    }

    #[test]
    fn test_saturation_zero() {
        let c = GradeColor::new(1.0, 0.0, 0.0);
        let out = apply_saturation(&c, 0.0);
        // Should be grayscale
        assert!(approx_eq(out.r, out.g, 1e-6));
        assert!(approx_eq(out.g, out.b, 1e-6));
    }

    #[test]
    fn test_saturation_identity() {
        let c = GradeColor::new(0.8, 0.2, 0.5);
        let out = apply_saturation(&c, 1.0);
        assert!(colors_approx_eq(&c, &out, 1e-6));
    }

    #[test]
    fn test_contrast_identity() {
        let c = GradeColor::new(0.3, 0.5, 0.7);
        let out = apply_contrast(&c, 1.0);
        assert!(colors_approx_eq(&c, &out, 1e-6));
    }

    #[test]
    fn test_contrast_increased() {
        let dark = GradeColor::new(0.3, 0.3, 0.3);
        let out = apply_contrast(&dark, 2.0);
        assert!(out.r < 0.3); // Darks get darker
    }

    #[test]
    fn test_brightness() {
        let c = GradeColor::gray(0.5);
        let brighter = apply_brightness(&c, 0.1);
        assert!(approx_eq(brighter.r, 0.6, 1e-6));
    }

    #[test]
    fn test_hue_shift() {
        let red = GradeColor::new(1.0, 0.0, 0.0);
        let shifted = apply_hue_shift(&red, 120.0);
        // Red shifted 120 degrees should be roughly green
        assert!(shifted.g > shifted.r);
    }

    #[test]
    fn test_lift_gamma_gain_identity() {
        let c = GradeColor::new(0.5, 0.5, 0.5);
        let out = apply_lift_gamma_gain(
            &c,
            &GradeColor::black(),
            &GradeColor::white(),
            &GradeColor::white(),
        );
        assert!(colors_approx_eq(&c, &out, 1e-4));
    }

    #[test]
    fn test_channel_mixer_identity() {
        let c = GradeColor::new(0.3, 0.5, 0.7);
        let mixer = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let out = apply_channel_mixer(&c, &mixer);
        assert!(colors_approx_eq(&c, &out, 1e-10));
    }

    #[test]
    fn test_channel_mixer_swap_rb() {
        let c = GradeColor::new(1.0, 0.0, 0.0);
        let mixer = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
        let out = apply_channel_mixer(&c, &mixer);
        assert!(approx_eq(out.r, 0.0, 1e-10));
        assert!(approx_eq(out.b, 1.0, 1e-10));
    }

    #[test]
    fn test_split_toning() {
        let dark = GradeColor::new(0.1, 0.1, 0.1);
        let shadow_color = GradeColor::new(0.0, 0.0, 1.0);
        let highlight_color = GradeColor::new(1.0, 0.0, 0.0);
        let out = apply_split_toning(&dark, &shadow_color, &highlight_color, 0.0);
        // Dark pixel should pick up some blue from shadow tint
        assert!(out.b > dark.b);
    }

    #[test]
    fn test_grade_color_default_identity() {
        let c = GradeColor::new(0.5, 0.3, 0.7);
        let params = GradingParams::default();
        let out = grade_color(&c, &params);
        // Default params should be close to identity
        assert!(colors_approx_eq(&c, &out, 0.15));
    }

    #[test]
    fn test_generate_lut() {
        let params = GradingParams {
            saturation: 0.5,
            ..Default::default()
        };
        let lut = generate_lut(8, &params);
        assert_eq!(lut.size, 8);
        assert_eq!(lut.data.len(), 512);
    }

    #[test]
    fn test_apply_lut_to_pixels() {
        let lut = Lut3D::identity(16);
        let pixels = vec![
            GradeColor::new(0.2, 0.4, 0.6),
            GradeColor::new(0.8, 0.1, 0.3),
        ];
        let result = apply_lut(&pixels, 2, 1, &lut);
        assert_eq!(result.len(), 2);
        assert!(colors_approx_eq(&result[0], &pixels[0], 0.05));
    }

    #[test]
    fn test_lut_display() {
        let lut = Lut3D::identity(4);
        let s = format!("{}", lut);
        assert!(s.contains("Lut3D"));
        assert!(s.contains("size=4"));
    }

    #[test]
    fn test_grading_params_display() {
        let p = GradingParams::default();
        let s = format!("{}", p);
        assert!(s.contains("GradingParams"));
    }
}
