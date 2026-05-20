//! Color space conversions — RGB, HSL, HSV, CMYK, CIE L*a*b*, XYZ,
//! sRGB gamma, color temperature, blending modes.
//!
//! Replaces JS color libraries (chroma.js, color, tinycolor2) with
//! precise, energy-efficient Rust implementations.

use serde::{Deserialize, Serialize};

// ── Color types ──────────────────────────────────────────────────

/// Linear RGB color in [0,1].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rgb {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl Rgb {
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }

    /// From 8-bit sRGB values.
    pub fn from_u8(r: u8, g: u8, b: u8) -> Self {
        Self { r: r as f64 / 255.0, g: g as f64 / 255.0, b: b as f64 / 255.0 }
    }

    /// To 8-bit sRGB values.
    pub fn to_u8(self) -> (u8, u8, u8) {
        (
            (self.r.clamp(0.0, 1.0) * 255.0).round() as u8,
            (self.g.clamp(0.0, 1.0) * 255.0).round() as u8,
            (self.b.clamp(0.0, 1.0) * 255.0).round() as u8,
        )
    }
}

/// HSL color (hue in degrees, saturation and lightness in [0,1]).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Hsl {
    pub h: f64,
    pub s: f64,
    pub l: f64,
}

/// HSV color (hue in degrees, saturation and value in [0,1]).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Hsv {
    pub h: f64,
    pub s: f64,
    pub v: f64,
}

/// CMYK color (all values in [0,1]).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Cmyk {
    pub c: f64,
    pub m: f64,
    pub y: f64,
    pub k: f64,
}

/// CIE XYZ color.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Xyz {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// CIE L*a*b* color.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Lab {
    pub l: f64,
    pub a: f64,
    pub b: f64,
}

// ── RGB ↔ HSL ────────────────────────────────────────────────────

pub fn rgb_to_hsl(c: Rgb) -> Hsl {
    let max = c.r.max(c.g).max(c.b);
    let min = c.r.min(c.g).min(c.b);
    let l = (max + min) / 2.0;

    if (max - min).abs() < 1e-10 {
        return Hsl { h: 0.0, s: 0.0, l };
    }

    let d = max - min;
    let s = if l > 0.5 { d / (2.0 - max - min) } else { d / (max + min) };

    let h = if (max - c.r).abs() < 1e-10 {
        let mut h = (c.g - c.b) / d;
        if c.g < c.b { h += 6.0; }
        h
    } else if (max - c.g).abs() < 1e-10 {
        (c.b - c.r) / d + 2.0
    } else {
        (c.r - c.g) / d + 4.0
    };

    Hsl { h: h * 60.0, s, l }
}

pub fn hsl_to_rgb(c: Hsl) -> Rgb {
    if c.s.abs() < 1e-10 {
        return Rgb::new(c.l, c.l, c.l);
    }

    let q = if c.l < 0.5 { c.l * (1.0 + c.s) } else { c.l + c.s - c.l * c.s };
    let p = 2.0 * c.l - q;
    let h = c.h / 360.0;

    let hue_to_rgb = |t: f64| -> f64 {
        let t = ((t % 1.0) + 1.0) % 1.0;
        if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 0.5 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        }
    };

    Rgb::new(
        hue_to_rgb(h + 1.0 / 3.0),
        hue_to_rgb(h),
        hue_to_rgb(h - 1.0 / 3.0),
    )
}

// ── RGB ↔ HSV ────────────────────────────────────────────────────

pub fn rgb_to_hsv(c: Rgb) -> Hsv {
    let max = c.r.max(c.g).max(c.b);
    let min = c.r.min(c.g).min(c.b);
    let d = max - min;

    let v = max;
    let s = if max.abs() < 1e-10 { 0.0 } else { d / max };

    if d.abs() < 1e-10 {
        return Hsv { h: 0.0, s: 0.0, v };
    }

    let h = if (max - c.r).abs() < 1e-10 {
        let mut h = (c.g - c.b) / d;
        if c.g < c.b { h += 6.0; }
        h
    } else if (max - c.g).abs() < 1e-10 {
        (c.b - c.r) / d + 2.0
    } else {
        (c.r - c.g) / d + 4.0
    };

    Hsv { h: h * 60.0, s, v }
}

pub fn hsv_to_rgb(c: Hsv) -> Rgb {
    if c.s.abs() < 1e-10 {
        return Rgb::new(c.v, c.v, c.v);
    }

    let h = (c.h / 60.0) % 6.0;
    let i = h.floor() as i32;
    let f = h - i as f64;
    let p = c.v * (1.0 - c.s);
    let q = c.v * (1.0 - c.s * f);
    let t = c.v * (1.0 - c.s * (1.0 - f));

    match i {
        0 => Rgb::new(c.v, t, p),
        1 => Rgb::new(q, c.v, p),
        2 => Rgb::new(p, c.v, t),
        3 => Rgb::new(p, q, c.v),
        4 => Rgb::new(t, p, c.v),
        _ => Rgb::new(c.v, p, q),
    }
}

// ── RGB ↔ CMYK ───────────────────────────────────────────────────

pub fn rgb_to_cmyk(c: Rgb) -> Cmyk {
    let k = 1.0 - c.r.max(c.g).max(c.b);
    if k >= 1.0 - 1e-10 {
        return Cmyk { c: 0.0, m: 0.0, y: 0.0, k: 1.0 };
    }
    Cmyk {
        c: (1.0 - c.r - k) / (1.0 - k),
        m: (1.0 - c.g - k) / (1.0 - k),
        y: (1.0 - c.b - k) / (1.0 - k),
        k,
    }
}

pub fn cmyk_to_rgb(c: Cmyk) -> Rgb {
    Rgb::new(
        (1.0 - c.c) * (1.0 - c.k),
        (1.0 - c.m) * (1.0 - c.k),
        (1.0 - c.y) * (1.0 - c.k),
    )
}

// ── sRGB gamma ───────────────────────────────────────────────────

/// sRGB gamma to linear.
pub fn srgb_to_linear(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear to sRGB gamma.
pub fn linear_to_srgb(v: f64) -> f64 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// Convert sRGB Rgb to linear Rgb.
pub fn rgb_srgb_to_linear(c: Rgb) -> Rgb {
    Rgb::new(srgb_to_linear(c.r), srgb_to_linear(c.g), srgb_to_linear(c.b))
}

/// Convert linear Rgb to sRGB Rgb.
pub fn rgb_linear_to_srgb(c: Rgb) -> Rgb {
    Rgb::new(linear_to_srgb(c.r), linear_to_srgb(c.g), linear_to_srgb(c.b))
}

// ── RGB ↔ XYZ (D65 illuminant) ──────────────────────────────────

pub fn rgb_to_xyz(c: Rgb) -> Xyz {
    let lin = rgb_srgb_to_linear(c);
    Xyz {
        x: lin.r * 0.4124564 + lin.g * 0.3575761 + lin.b * 0.1804375,
        y: lin.r * 0.2126729 + lin.g * 0.7151522 + lin.b * 0.0721750,
        z: lin.r * 0.0193339 + lin.g * 0.1191920 + lin.b * 0.9503041,
    }
}

pub fn xyz_to_rgb(c: Xyz) -> Rgb {
    let r = c.x * 3.2404542 + c.y * -1.5371385 + c.z * -0.4985314;
    let g = c.x * -0.9692660 + c.y * 1.8760108 + c.z * 0.0415560;
    let b = c.x * 0.0556434 + c.y * -0.2040259 + c.z * 1.0572252;
    rgb_linear_to_srgb(Rgb::new(r, g, b))
}

// ── XYZ ↔ Lab ────────────────────────────────────────────────────

const XN: f64 = 0.950456;
const YN: f64 = 1.0;
const ZN: f64 = 1.088754;

fn lab_f(t: f64) -> f64 {
    if t > 0.008856 {
        t.cbrt()
    } else {
        7.787 * t + 16.0 / 116.0
    }
}

fn lab_f_inv(t: f64) -> f64 {
    if t > 0.206897 {
        t * t * t
    } else {
        (t - 16.0 / 116.0) / 7.787
    }
}

pub fn xyz_to_lab(c: Xyz) -> Lab {
    let fx = lab_f(c.x / XN);
    let fy = lab_f(c.y / YN);
    let fz = lab_f(c.z / ZN);
    Lab {
        l: 116.0 * fy - 16.0,
        a: 500.0 * (fx - fy),
        b: 200.0 * (fy - fz),
    }
}

pub fn lab_to_xyz(c: Lab) -> Xyz {
    let fy = (c.l + 16.0) / 116.0;
    let fx = c.a / 500.0 + fy;
    let fz = fy - c.b / 200.0;
    Xyz {
        x: XN * lab_f_inv(fx),
        y: YN * lab_f_inv(fy),
        z: ZN * lab_f_inv(fz),
    }
}

/// Convenience: RGB → Lab.
pub fn rgb_to_lab(c: Rgb) -> Lab {
    xyz_to_lab(rgb_to_xyz(c))
}

/// Convenience: Lab → RGB.
pub fn lab_to_rgb(c: Lab) -> Rgb {
    xyz_to_rgb(lab_to_xyz(c))
}

// ── Color temperature ────────────────────────────────────────────

/// Convert color temperature (Kelvin) to sRGB (Tanner Helland algorithm).
pub fn color_temp_to_rgb(kelvin: f64) -> Rgb {
    let temp = kelvin / 100.0;
    let r;
    let g;
    let b;

    if temp <= 66.0 {
        r = 255.0;
        g = 99.4708025861 * temp.ln() - 161.1195681661;
        b = if temp <= 19.0 {
            0.0
        } else {
            138.5177312231 * (temp - 10.0).ln() - 305.0447927307
        };
    } else {
        r = 329.698727446 * (temp - 60.0).powf(-0.1332047592);
        g = 288.1221695283 * (temp - 60.0).powf(-0.0755148492);
        b = 255.0;
    }

    Rgb::new(
        (r / 255.0).clamp(0.0, 1.0),
        (g / 255.0).clamp(0.0, 1.0),
        (b / 255.0).clamp(0.0, 1.0),
    )
}

// ── Blending modes ───────────────────────────────────────────────

/// Color blending mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlendMode {
    Normal,
    Multiply,
    Screen,
    Overlay,
    SoftLight,
    HardLight,
    Difference,
    Exclusion,
}

fn blend_channel(mode: BlendMode, base: f64, blend: f64) -> f64 {
    match mode {
        BlendMode::Normal => blend,
        BlendMode::Multiply => base * blend,
        BlendMode::Screen => 1.0 - (1.0 - base) * (1.0 - blend),
        BlendMode::Overlay => {
            if base < 0.5 {
                2.0 * base * blend
            } else {
                1.0 - 2.0 * (1.0 - base) * (1.0 - blend)
            }
        }
        BlendMode::SoftLight => {
            if blend < 0.5 {
                base - (1.0 - 2.0 * blend) * base * (1.0 - base)
            } else {
                let d = if base <= 0.25 {
                    ((16.0 * base - 12.0) * base + 4.0) * base
                } else {
                    base.sqrt()
                };
                base + (2.0 * blend - 1.0) * (d - base)
            }
        }
        BlendMode::HardLight => {
            if blend < 0.5 {
                2.0 * base * blend
            } else {
                1.0 - 2.0 * (1.0 - base) * (1.0 - blend)
            }
        }
        BlendMode::Difference => (base - blend).abs(),
        BlendMode::Exclusion => base + blend - 2.0 * base * blend,
    }
}

/// Blend two RGB colors.
pub fn blend_rgb(base: Rgb, over: Rgb, mode: BlendMode) -> Rgb {
    Rgb::new(
        blend_channel(mode, base.r, over.r).clamp(0.0, 1.0),
        blend_channel(mode, base.g, over.g).clamp(0.0, 1.0),
        blend_channel(mode, base.b, over.b).clamp(0.0, 1.0),
    )
}

// ── CIE76 color difference ──────────────────────────────────────

/// Delta-E (CIE76) between two Lab colors.
pub fn delta_e_76(a: Lab, b: Lab) -> f64 {
    let dl = a.l - b.l;
    let da = a.a - b.a;
    let db = a.b - b.b;
    (dl * dl + da * da + db * db).sqrt()
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_rgb_hsl_roundtrip() {
        let c = Rgb::new(0.5, 0.2, 0.8);
        let hsl = rgb_to_hsl(c);
        let back = hsl_to_rgb(hsl);
        assert!(approx(c.r, back.r, 0.01));
        assert!(approx(c.g, back.g, 0.01));
        assert!(approx(c.b, back.b, 0.01));
    }

    #[test]
    fn test_rgb_hsv_roundtrip() {
        let c = Rgb::new(0.3, 0.7, 0.1);
        let hsv = rgb_to_hsv(c);
        let back = hsv_to_rgb(hsv);
        assert!(approx(c.r, back.r, 0.01));
        assert!(approx(c.g, back.g, 0.01));
        assert!(approx(c.b, back.b, 0.01));
    }

    #[test]
    fn test_rgb_cmyk_roundtrip() {
        let c = Rgb::new(0.4, 0.6, 0.8);
        let cmyk = rgb_to_cmyk(c);
        let back = cmyk_to_rgb(cmyk);
        assert!(approx(c.r, back.r, 0.01));
        assert!(approx(c.g, back.g, 0.01));
        assert!(approx(c.b, back.b, 0.01));
    }

    #[test]
    fn test_cmyk_black() {
        let c = Rgb::new(0.0, 0.0, 0.0);
        let cmyk = rgb_to_cmyk(c);
        assert!(approx(cmyk.k, 1.0, 0.01));
    }

    #[test]
    fn test_srgb_gamma_roundtrip() {
        for v in [0.0, 0.04, 0.1, 0.5, 1.0] {
            let lin = srgb_to_linear(v);
            let back = linear_to_srgb(lin);
            assert!(approx(v, back, 0.001), "failed for {v}");
        }
    }

    #[test]
    fn test_rgb_xyz_roundtrip() {
        let c = Rgb::new(0.5, 0.3, 0.7);
        let xyz = rgb_to_xyz(c);
        let back = xyz_to_rgb(xyz);
        assert!(approx(c.r, back.r, 0.02));
        assert!(approx(c.g, back.g, 0.02));
        assert!(approx(c.b, back.b, 0.02));
    }

    #[test]
    fn test_rgb_lab_roundtrip() {
        let c = Rgb::new(0.6, 0.2, 0.8);
        let lab = rgb_to_lab(c);
        let back = lab_to_rgb(lab);
        assert!(approx(c.r, back.r, 0.02));
        assert!(approx(c.g, back.g, 0.02));
        assert!(approx(c.b, back.b, 0.02));
    }

    #[test]
    fn test_lab_white() {
        let white = Rgb::new(1.0, 1.0, 1.0);
        let lab = rgb_to_lab(white);
        assert!(approx(lab.l, 100.0, 1.0));
        assert!(approx(lab.a, 0.0, 1.0));
        assert!(approx(lab.b, 0.0, 1.0));
    }

    #[test]
    fn test_color_temp_daylight() {
        let c = color_temp_to_rgb(6500.0);
        // Daylight should be close to white
        assert!(c.r > 0.9);
        assert!(c.g > 0.9);
        assert!(c.b > 0.9);
    }

    #[test]
    fn test_color_temp_warm() {
        let c = color_temp_to_rgb(2700.0);
        // Warm light: more red, less blue
        assert!(c.r > c.b);
    }

    #[test]
    fn test_blend_multiply() {
        let a = Rgb::new(0.5, 0.5, 0.5);
        let b = Rgb::new(1.0, 0.0, 0.5);
        let r = blend_rgb(a, b, BlendMode::Multiply);
        assert!(approx(r.r, 0.5, 0.01));
        assert!(approx(r.g, 0.0, 0.01));
        assert!(approx(r.b, 0.25, 0.01));
    }

    #[test]
    fn test_blend_screen() {
        let a = Rgb::new(0.5, 0.5, 0.5);
        let b = Rgb::new(0.5, 0.5, 0.5);
        let r = blend_rgb(a, b, BlendMode::Screen);
        assert!(approx(r.r, 0.75, 0.01));
    }

    #[test]
    fn test_blend_difference() {
        let a = Rgb::new(0.8, 0.2, 0.5);
        let b = Rgb::new(0.3, 0.7, 0.5);
        let r = blend_rgb(a, b, BlendMode::Difference);
        assert!(approx(r.r, 0.5, 0.01));
        assert!(approx(r.g, 0.5, 0.01));
        assert!(approx(r.b, 0.0, 0.01));
    }

    #[test]
    fn test_delta_e() {
        let a = rgb_to_lab(Rgb::new(1.0, 0.0, 0.0));
        let b = rgb_to_lab(Rgb::new(0.0, 1.0, 0.0));
        let de = delta_e_76(a, b);
        assert!(de > 50.0); // Red and green are far apart
    }

    #[test]
    fn test_hsl_gray() {
        let gray = Rgb::new(0.5, 0.5, 0.5);
        let hsl = rgb_to_hsl(gray);
        assert!(approx(hsl.s, 0.0, 0.01));
        assert!(approx(hsl.l, 0.5, 0.01));
    }

    #[test]
    fn test_from_to_u8() {
        let c = Rgb::from_u8(128, 64, 255);
        let (r, g, b) = c.to_u8();
        assert_eq!(r, 128);
        assert_eq!(g, 64);
        assert_eq!(b, 255);
    }
}
