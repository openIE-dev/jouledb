//! Image blending — alpha compositing, layer blending modes,
//! opacity, mask-based blending.
//!
//! Replaces HTML5 Canvas globalCompositeOperation and CSS mix-blend-mode
//! with a pure-Rust compositing engine.

use serde::{Deserialize, Serialize};

// ── Color ────────────────────────────────────────────────────────

/// RGBA pixel with components in [0, 255].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rgba {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Rgba {
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    pub const TRANSPARENT: Self = Self::new(0, 0, 0, 0);

    fn to_f64(self) -> [f64; 4] {
        [
            self.r as f64 / 255.0,
            self.g as f64 / 255.0,
            self.b as f64 / 255.0,
            self.a as f64 / 255.0,
        ]
    }

    fn from_f64(c: [f64; 4]) -> Self {
        Self {
            r: (c[0].clamp(0.0, 1.0) * 255.0).round() as u8,
            g: (c[1].clamp(0.0, 1.0) * 255.0).round() as u8,
            b: (c[2].clamp(0.0, 1.0) * 255.0).round() as u8,
            a: (c[3].clamp(0.0, 1.0) * 255.0).round() as u8,
        }
    }
}

// ── Image ────────────────────────────────────────────────────────

/// RGBA image buffer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl Image {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height, data: vec![0u8; width as usize * height as usize * 4] }
    }

    pub fn filled(width: u32, height: u32, c: Rgba) -> Self {
        let mut img = Self::new(width, height);
        for i in 0..(width as usize * height as usize) {
            let off = i * 4;
            img.data[off] = c.r;
            img.data[off + 1] = c.g;
            img.data[off + 2] = c.b;
            img.data[off + 3] = c.a;
        }
        img
    }

    pub fn get(&self, x: u32, y: u32) -> Rgba {
        let off = (y as usize * self.width as usize + x as usize) * 4;
        Rgba::new(self.data[off], self.data[off + 1], self.data[off + 2], self.data[off + 3])
    }

    pub fn set(&mut self, x: u32, y: u32, c: Rgba) {
        let off = (y as usize * self.width as usize + x as usize) * 4;
        self.data[off] = c.r;
        self.data[off + 1] = c.g;
        self.data[off + 2] = c.b;
        self.data[off + 3] = c.a;
    }
}

// ── Blend Mode ───────────────────────────────────────────────────

/// Layer blending modes (matching CSS/Photoshop names).
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
            if base < 0.5 { 2.0 * base * blend } else { 1.0 - 2.0 * (1.0 - base) * (1.0 - blend) }
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
            if blend < 0.5 { 2.0 * base * blend } else { 1.0 - 2.0 * (1.0 - base) * (1.0 - blend) }
        }
        BlendMode::Difference => (base - blend).abs(),
        BlendMode::Exclusion => base + blend - 2.0 * base * blend,
    }
}

// ── Alpha compositing (Porter-Duff "over") ───────────────────────

/// Alpha composite `over` onto `base` (Porter-Duff "source over").
pub fn alpha_composite(base: Rgba, over: Rgba) -> Rgba {
    let b = base.to_f64();
    let o = over.to_f64();
    let aa = o[3] + b[3] * (1.0 - o[3]);
    if aa < 1e-10 {
        return Rgba::TRANSPARENT;
    }
    let blend = |i: usize| (o[i] * o[3] + b[i] * b[3] * (1.0 - o[3])) / aa;
    Rgba::from_f64([blend(0), blend(1), blend(2), aa])
}

/// Blend two pixels with a blending mode and alpha compositing.
pub fn blend_pixel(base: Rgba, over: Rgba, mode: BlendMode) -> Rgba {
    let b = base.to_f64();
    let o = over.to_f64();
    let aa = o[3] + b[3] * (1.0 - o[3]);
    if aa < 1e-10 {
        return Rgba::TRANSPARENT;
    }
    let mut result = [0.0f64; 4];
    for i in 0..3 {
        let blended = blend_channel(mode, b[i], o[i]);
        result[i] = (blended * o[3] + b[i] * b[3] * (1.0 - o[3])) / aa;
    }
    result[3] = aa;
    Rgba::from_f64(result)
}

/// Blend two images with a blending mode. Images must have the same dimensions.
pub fn blend_images(base: &Image, over: &Image, mode: BlendMode) -> Image {
    assert_eq!(base.width, over.width);
    assert_eq!(base.height, over.height);
    let mut out = Image::new(base.width, base.height);
    let pixels = base.width as usize * base.height as usize;
    for i in 0..pixels {
        let off = i * 4;
        let b = Rgba::new(base.data[off], base.data[off + 1], base.data[off + 2], base.data[off + 3]);
        let o = Rgba::new(over.data[off], over.data[off + 1], over.data[off + 2], over.data[off + 3]);
        let r = blend_pixel(b, o, mode);
        out.data[off] = r.r;
        out.data[off + 1] = r.g;
        out.data[off + 2] = r.b;
        out.data[off + 3] = r.a;
    }
    out
}

// ── Opacity ──────────────────────────────────────────────────────

/// Apply a global opacity to an image (multiplies alpha channel).
pub fn apply_opacity(img: &Image, opacity: f64) -> Image {
    let opacity = opacity.clamp(0.0, 1.0);
    let mut out = img.clone();
    let pixels = img.width as usize * img.height as usize;
    for i in 0..pixels {
        let off = i * 4 + 3;
        out.data[off] = (img.data[off] as f64 * opacity).round() as u8;
    }
    out
}

// ── Mask-based blending ──────────────────────────────────────────

/// A grayscale mask (0 = transparent, 255 = opaque).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mask {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl Mask {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height, data: vec![0u8; width as usize * height as usize] }
    }

    pub fn filled(width: u32, height: u32, v: u8) -> Self {
        Self { width, height, data: vec![v; width as usize * height as usize] }
    }

    pub fn get(&self, x: u32, y: u32) -> u8 {
        self.data[y as usize * self.width as usize + x as usize]
    }

    pub fn set(&mut self, x: u32, y: u32, v: u8) {
        self.data[y as usize * self.width as usize + x as usize] = v;
    }
}

/// Blend two images using a mask. Where mask = 255, use `over`; where 0, use `base`.
pub fn blend_with_mask(base: &Image, over: &Image, mask: &Mask) -> Image {
    assert_eq!(base.width, over.width);
    assert_eq!(base.width, mask.width);
    assert_eq!(base.height, over.height);
    assert_eq!(base.height, mask.height);

    let mut out = Image::new(base.width, base.height);
    let pixels = base.width as usize * base.height as usize;
    for i in 0..pixels {
        let off = i * 4;
        let t = mask.data[i] as f64 / 255.0;
        for c in 0..4 {
            let bv = base.data[off + c] as f64;
            let ov = over.data[off + c] as f64;
            out.data[off + c] = (bv * (1.0 - t) + ov * t).round() as u8;
        }
    }
    out
}

// ── Layer stack ──────────────────────────────────────────────────

/// A compositing layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub image: Image,
    pub mode: BlendMode,
    pub opacity: f64,
}

/// Flatten a stack of layers (bottom to top).
pub fn flatten_layers(layers: &[Layer]) -> Image {
    if layers.is_empty() {
        return Image::new(0, 0);
    }
    let w = layers[0].image.width;
    let h = layers[0].image.height;
    let mut result = Image::new(w, h);

    for layer in layers {
        let with_opacity = apply_opacity(&layer.image, layer.opacity);
        result = blend_images(&result, &with_opacity, layer.mode);
    }
    result
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_composite_opaque_over() {
        let base = Rgba::rgb(255, 0, 0);
        let over = Rgba::rgb(0, 255, 0);
        let result = alpha_composite(base, over);
        assert_eq!(result.r, 0);
        assert_eq!(result.g, 255);
        assert_eq!(result.a, 255);
    }

    #[test]
    fn test_alpha_composite_transparent_over() {
        let base = Rgba::rgb(255, 0, 0);
        let over = Rgba::TRANSPARENT;
        let result = alpha_composite(base, over);
        assert_eq!(result.r, 255);
        assert_eq!(result.a, 255);
    }

    #[test]
    fn test_alpha_composite_half_alpha() {
        let base = Rgba::rgb(0, 0, 0);
        let over = Rgba::new(255, 255, 255, 128);
        let result = alpha_composite(base, over);
        // Should be a blend
        assert!(result.r > 50 && result.r < 200);
        assert!(result.a > 128);
    }

    #[test]
    fn test_blend_multiply() {
        let base = Rgba::rgb(200, 200, 200);
        let over = Rgba::rgb(128, 128, 128);
        let result = blend_pixel(base, over, BlendMode::Multiply);
        // 200/255 * 128/255 ≈ 0.392 → 100
        assert!(result.r > 90 && result.r < 110);
    }

    #[test]
    fn test_blend_screen() {
        let base = Rgba::rgb(100, 100, 100);
        let over = Rgba::rgb(100, 100, 100);
        let result = blend_pixel(base, over, BlendMode::Screen);
        // Screen makes things brighter
        assert!(result.r > 100);
    }

    #[test]
    fn test_blend_overlay() {
        let base = Rgba::rgb(128, 128, 128);
        let over = Rgba::rgb(200, 50, 100);
        let result = blend_pixel(base, over, BlendMode::Overlay);
        assert!(result.r > 0);
        assert_eq!(result.a, 255);
    }

    #[test]
    fn test_blend_difference() {
        let base = Rgba::rgb(200, 100, 50);
        let over = Rgba::rgb(100, 100, 50);
        let result = blend_pixel(base, over, BlendMode::Difference);
        // |200/255 - 100/255| ≈ 0.392 → 100
        assert!(result.r > 90 && result.r < 110);
        assert!(result.g < 10); // |100-100| = 0
    }

    #[test]
    fn test_blend_exclusion() {
        let base = Rgba::rgb(128, 128, 128);
        let over = Rgba::rgb(128, 128, 128);
        let result = blend_pixel(base, over, BlendMode::Exclusion);
        // exclusion(0.5, 0.5) = 0.5 + 0.5 - 2*0.5*0.5 = 0.5 → 128
        assert!((result.r as i32 - 128).abs() <= 1);
    }

    #[test]
    fn test_apply_opacity() {
        let img = Image::filled(2, 2, Rgba::rgb(255, 0, 0));
        let faded = apply_opacity(&img, 0.5);
        let px = faded.get(0, 0);
        assert_eq!(px.r, 255);
        assert!((px.a as i32 - 128).abs() <= 1);
    }

    #[test]
    fn test_blend_images() {
        let base = Image::filled(2, 2, Rgba::rgb(255, 0, 0));
        let over = Image::filled(2, 2, Rgba::rgb(0, 0, 255));
        let result = blend_images(&base, &over, BlendMode::Normal);
        let px = result.get(0, 0);
        assert_eq!(px.r, 0);
        assert_eq!(px.b, 255);
    }

    #[test]
    fn test_mask_blend() {
        let base = Image::filled(2, 2, Rgba::rgb(255, 0, 0));
        let over = Image::filled(2, 2, Rgba::rgb(0, 255, 0));
        let mut mask = Mask::new(2, 2);
        mask.set(0, 0, 255); // full over
        mask.set(1, 0, 0);   // full base
        mask.set(0, 1, 128); // half blend

        let result = blend_with_mask(&base, &over, &mask);
        assert_eq!(result.get(0, 0).g, 255); // full green
        assert_eq!(result.get(1, 0).r, 255); // full red
        let mid = result.get(0, 1);
        assert!(mid.r > 100 && mid.r < 140);
        assert!(mid.g > 110 && mid.g < 150);
    }

    #[test]
    fn test_flatten_layers() {
        let base = Layer {
            image: Image::filled(2, 2, Rgba::rgb(255, 0, 0)),
            mode: BlendMode::Normal,
            opacity: 1.0,
        };
        let over = Layer {
            image: Image::filled(2, 2, Rgba::rgb(0, 255, 0)),
            mode: BlendMode::Normal,
            opacity: 0.5,
        };
        let result = flatten_layers(&[base, over]);
        let px = result.get(0, 0);
        // Green at 50% opacity over red
        assert!(px.r > 80 && px.r < 180);
        assert!(px.g > 50 && px.g < 180);
    }

    #[test]
    fn test_soft_light() {
        let base = Rgba::rgb(128, 128, 128);
        let over = Rgba::rgb(200, 200, 200);
        let result = blend_pixel(base, over, BlendMode::SoftLight);
        assert!(result.r > 0);
    }

    #[test]
    fn test_hard_light() {
        let base = Rgba::rgb(128, 128, 128);
        let over = Rgba::rgb(200, 200, 200);
        let result = blend_pixel(base, over, BlendMode::HardLight);
        assert!(result.r > 128); // Hard light with bright over → brighter
    }

    #[test]
    fn test_empty_layers() {
        let result = flatten_layers(&[]);
        assert_eq!(result.width, 0);
        assert_eq!(result.height, 0);
    }
}
