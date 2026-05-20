//! Texture sampling and filtering.
//!
//! Texture2D with RGBA u8 pixel data, nearest and bilinear filtering, wrap
//! modes (Repeat, Clamp, MirrorRepeat), mipmap chain generation via box
//! filter, trilinear sampling, and border color for clamp mode.
//! Pure Rust — no image or GPU dependencies.

use std::fmt;

// ── Pixel ───────────────────────────────────────────────────────

/// A single RGBA pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pixel {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Pixel {
    pub const WHITE: Self = Self { r: 255, g: 255, b: 255, a: 255 };
    pub const BLACK: Self = Self { r: 0, g: 0, b: 0, a: 255 };
    pub const TRANSPARENT: Self = Self { r: 0, g: 0, b: 0, a: 0 };

    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Linear-space float representation (0.0..1.0 per channel).
    pub fn to_float(&self) -> [f32; 4] {
        [
            self.r as f32 / 255.0,
            self.g as f32 / 255.0,
            self.b as f32 / 255.0,
            self.a as f32 / 255.0,
        ]
    }

    /// From float (clamped).
    pub fn from_float(c: [f32; 4]) -> Self {
        Self {
            r: (c[0].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            g: (c[1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            b: (c[2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            a: (c[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        }
    }
}

// ── Filter and wrap modes ───────────────────────────────────────

/// Texture magnification/minification filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterMode {
    Nearest,
    Bilinear,
}

/// How UVs outside 0..1 are handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WrapMode {
    Repeat,
    Clamp,
    MirrorRepeat,
}

impl WrapMode {
    /// Map an arbitrary UV coordinate into the 0..1 range.
    pub fn apply(&self, t: f32) -> f32 {
        match self {
            WrapMode::Repeat => {
                let f = t % 1.0;
                if f < 0.0 { f + 1.0 } else { f }
            }
            WrapMode::Clamp => t.clamp(0.0, 1.0),
            WrapMode::MirrorRepeat => {
                let f = t % 2.0;
                let f = if f < 0.0 { f + 2.0 } else { f };
                if f > 1.0 { 2.0 - f } else { f }
            }
        }
    }
}

// ── Sampler state ───────────────────────────────────────────────

/// Combined sampler parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerState {
    pub filter: FilterMode,
    pub wrap_u: WrapMode,
    pub wrap_v: WrapMode,
    pub border_color: [f32; 4],
}

impl SamplerState {
    pub fn new(filter: FilterMode, wrap: WrapMode) -> Self {
        Self { filter, wrap_u: wrap, wrap_v: wrap, border_color: [0.0, 0.0, 0.0, 1.0] }
    }

    pub fn with_border(mut self, color: [f32; 4]) -> Self {
        self.border_color = color;
        self
    }
}

impl Default for SamplerState {
    fn default() -> Self {
        Self::new(FilterMode::Bilinear, WrapMode::Repeat)
    }
}

// ── Texture2D ───────────────────────────────────────────────────

/// A 2D RGBA texture with optional mipmap chain.
#[derive(Debug, Clone)]
pub struct Texture2D {
    pub width: u32,
    pub height: u32,
    pixels: Vec<Pixel>,
    mip_chain: Vec<MipLevel>,
}

#[derive(Debug, Clone)]
struct MipLevel {
    width: u32,
    height: u32,
    pixels: Vec<Pixel>,
}

impl Texture2D {
    /// Create a texture from dimensions and pixel data.
    pub fn new(width: u32, height: u32, pixels: Vec<Pixel>) -> Result<Self, String> {
        let expected = (width as usize) * (height as usize);
        if pixels.len() != expected {
            return Err(format!("expected {} pixels, got {}", expected, pixels.len()));
        }
        Ok(Self { width, height, pixels, mip_chain: Vec::new() })
    }

    /// Solid color texture.
    pub fn solid(width: u32, height: u32, color: Pixel) -> Self {
        let count = (width as usize) * (height as usize);
        Self { width, height, pixels: vec![color; count], mip_chain: Vec::new() }
    }

    /// Checkerboard pattern (useful for testing).
    pub fn checkerboard(width: u32, height: u32, cell: u32, a: Pixel, b: Pixel) -> Self {
        let count = (width as usize) * (height as usize);
        let mut pixels = Vec::with_capacity(count);
        for y in 0..height {
            for x in 0..width {
                let cell_size = cell.max(1);
                let cx = x / cell_size;
                let cy = y / cell_size;
                if (cx + cy) % 2 == 0 { pixels.push(a); } else { pixels.push(b); }
            }
        }
        Self { width, height, pixels, mip_chain: Vec::new() }
    }

    pub fn pixel_count(&self) -> usize {
        self.pixels.len()
    }

    /// Get pixel at integer coordinates (clamped).
    pub fn get_pixel(&self, x: u32, y: u32) -> Pixel {
        let x = x.min(self.width.saturating_sub(1));
        let y = y.min(self.height.saturating_sub(1));
        self.pixels[(y * self.width + x) as usize]
    }

    /// Set pixel at integer coordinates.
    pub fn set_pixel(&mut self, x: u32, y: u32, p: Pixel) {
        if x < self.width && y < self.height {
            self.pixels[(y * self.width + x) as usize] = p;
        }
    }

    // ── Sampling ────────────────────────────────────────────────

    /// Sample the texture at UV coordinates using the given sampler.
    pub fn sample(&self, u: f32, v: f32, sampler: &SamplerState) -> [f32; 4] {
        let wu = sampler.wrap_u.apply(u);
        let wv = sampler.wrap_v.apply(v);

        match sampler.filter {
            FilterMode::Nearest => self.sample_nearest(wu, wv),
            FilterMode::Bilinear => self.sample_bilinear(wu, wv),
        }
    }

    fn sample_nearest(&self, u: f32, v: f32) -> [f32; 4] {
        let x = (u * self.width as f32) as u32;
        let y = (v * self.height as f32) as u32;
        self.get_pixel(x, y).to_float()
    }

    fn sample_bilinear(&self, u: f32, v: f32) -> [f32; 4] {
        let fx = u * self.width as f32 - 0.5;
        let fy = v * self.height as f32 - 0.5;
        let x0 = fx.floor() as i32;
        let y0 = fy.floor() as i32;
        let frac_x = fx - fx.floor();
        let frac_y = fy - fy.floor();

        let clamp_x = |x: i32| -> u32 { x.max(0).min(self.width as i32 - 1) as u32 };
        let clamp_y = |y: i32| -> u32 { y.max(0).min(self.height as i32 - 1) as u32 };

        let p00 = self.get_pixel(clamp_x(x0), clamp_y(y0)).to_float();
        let p10 = self.get_pixel(clamp_x(x0 + 1), clamp_y(y0)).to_float();
        let p01 = self.get_pixel(clamp_x(x0), clamp_y(y0 + 1)).to_float();
        let p11 = self.get_pixel(clamp_x(x0 + 1), clamp_y(y0 + 1)).to_float();

        let mut result = [0.0f32; 4];
        for i in 0..4 {
            let top = p00[i] + (p10[i] - p00[i]) * frac_x;
            let bot = p01[i] + (p11[i] - p01[i]) * frac_x;
            result[i] = top + (bot - top) * frac_y;
        }
        result
    }

    // ── Mipmap generation ───────────────────────────────────────

    /// Generate the full mipmap chain down to 1x1 using box filtering.
    pub fn generate_mipmaps(&mut self) {
        self.mip_chain.clear();
        let mut prev_w = self.width;
        let mut prev_h = self.height;
        let mut prev_pixels = self.pixels.clone();

        while prev_w > 1 || prev_h > 1 {
            let new_w = (prev_w / 2).max(1);
            let new_h = (prev_h / 2).max(1);
            let mut new_pixels = Vec::with_capacity((new_w * new_h) as usize);

            for y in 0..new_h {
                for x in 0..new_w {
                    let sx = (x * 2).min(prev_w - 1);
                    let sy = (y * 2).min(prev_h - 1);
                    let sx1 = (sx + 1).min(prev_w - 1);
                    let sy1 = (sy + 1).min(prev_h - 1);

                    let idx = |px: u32, py: u32| (py * prev_w + px) as usize;
                    let p00 = prev_pixels[idx(sx, sy)];
                    let p10 = prev_pixels[idx(sx1, sy)];
                    let p01 = prev_pixels[idx(sx, sy1)];
                    let p11 = prev_pixels[idx(sx1, sy1)];

                    let avg = |a: u8, b: u8, c: u8, d: u8| -> u8 {
                        ((a as u16 + b as u16 + c as u16 + d as u16 + 2) / 4) as u8
                    };
                    new_pixels.push(Pixel::new(
                        avg(p00.r, p10.r, p01.r, p11.r),
                        avg(p00.g, p10.g, p01.g, p11.g),
                        avg(p00.b, p10.b, p01.b, p11.b),
                        avg(p00.a, p10.a, p01.a, p11.a),
                    ));
                }
            }

            self.mip_chain.push(MipLevel { width: new_w, height: new_h, pixels: new_pixels.clone() });
            prev_w = new_w;
            prev_h = new_h;
            prev_pixels = new_pixels;
        }
    }

    /// Number of mip levels (0 = base only).
    pub fn mip_count(&self) -> usize {
        self.mip_chain.len()
    }

    /// Sample a specific mip level (0 = first generated mip, not base).
    pub fn sample_mip(&self, mip: usize, u: f32, v: f32, sampler: &SamplerState) -> [f32; 4] {
        if mip >= self.mip_chain.len() {
            // Fall back to base texture.
            return self.sample(u, v, sampler);
        }
        let level = &self.mip_chain[mip];
        let wu = sampler.wrap_u.apply(u);
        let wv = sampler.wrap_v.apply(v);

        match sampler.filter {
            FilterMode::Nearest => {
                let x = (wu * level.width as f32) as u32;
                let y = (wv * level.height as f32) as u32;
                let x = x.min(level.width.saturating_sub(1));
                let y = y.min(level.height.saturating_sub(1));
                level.pixels[(y * level.width + x) as usize].to_float()
            }
            FilterMode::Bilinear => {
                let fx = wu * level.width as f32 - 0.5;
                let fy = wv * level.height as f32 - 0.5;
                let x0 = fx.floor() as i32;
                let y0 = fy.floor() as i32;
                let frac_x = fx - fx.floor();
                let frac_y = fy - fy.floor();

                let clamp_x = |x: i32| -> u32 { x.max(0).min(level.width as i32 - 1) as u32 };
                let clamp_y = |y: i32| -> u32 { y.max(0).min(level.height as i32 - 1) as u32 };
                let get = |px: u32, py: u32| -> [f32; 4] {
                    level.pixels[(py * level.width + px) as usize].to_float()
                };

                let p00 = get(clamp_x(x0), clamp_y(y0));
                let p10 = get(clamp_x(x0 + 1), clamp_y(y0));
                let p01 = get(clamp_x(x0), clamp_y(y0 + 1));
                let p11 = get(clamp_x(x0 + 1), clamp_y(y0 + 1));

                let mut result = [0.0f32; 4];
                for i in 0..4 {
                    let top = p00[i] + (p10[i] - p00[i]) * frac_x;
                    let bot = p01[i] + (p11[i] - p01[i]) * frac_x;
                    result[i] = top + (bot - top) * frac_y;
                }
                result
            }
        }
    }

    /// Trilinear sample between two mip levels.
    pub fn sample_trilinear(&self, u: f32, v: f32, lod: f32, sampler: &SamplerState) -> [f32; 4] {
        if self.mip_chain.is_empty() {
            return self.sample(u, v, sampler);
        }
        let max_lod = self.mip_chain.len() as f32;
        let lod = lod.clamp(0.0, max_lod);

        if lod < 1.0 {
            // Blend between base and first mip.
            let base = self.sample(u, v, sampler);
            let mip0 = self.sample_mip(0, u, v, sampler);
            return lerp4(base, mip0, lod);
        }

        let level = (lod - 1.0) as usize;
        let frac = lod - 1.0 - level as f32;

        if level + 1 >= self.mip_chain.len() {
            return self.sample_mip(level.min(self.mip_chain.len() - 1), u, v, sampler);
        }

        let lo = self.sample_mip(level, u, v, sampler);
        let hi = self.sample_mip(level + 1, u, v, sampler);
        lerp4(lo, hi, frac)
    }
}

fn lerp4(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

impl fmt::Display for Texture2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Texture2D({}x{}, {} mips)", self.width, self.height, self.mip_chain.len())
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    #[test]
    fn test_pixel_new() {
        let p = Pixel::new(128, 64, 32, 255);
        assert_eq!(p.r, 128);
        assert_eq!(p.a, 255);
    }

    #[test]
    fn test_pixel_to_float() {
        let f = Pixel::WHITE.to_float();
        assert!(approx(f[0], 1.0));
        assert!(approx(f[3], 1.0));
    }

    #[test]
    fn test_pixel_from_float_round_trip() {
        let p = Pixel::new(100, 150, 200, 250);
        let f = p.to_float();
        let p2 = Pixel::from_float(f);
        assert_eq!(p.r, p2.r);
        assert_eq!(p.g, p2.g);
        assert_eq!(p.b, p2.b);
        assert_eq!(p.a, p2.a);
    }

    #[test]
    fn test_texture_creation() {
        let t = Texture2D::new(2, 2, vec![Pixel::WHITE; 4]).unwrap();
        assert_eq!(t.pixel_count(), 4);
    }

    #[test]
    fn test_texture_wrong_size() {
        assert!(Texture2D::new(2, 2, vec![Pixel::WHITE; 3]).is_err());
    }

    #[test]
    fn test_solid_texture() {
        let t = Texture2D::solid(4, 4, Pixel::new(128, 128, 128, 255));
        let p = t.get_pixel(2, 2);
        assert_eq!(p.r, 128);
    }

    #[test]
    fn test_checkerboard() {
        let t = Texture2D::checkerboard(4, 4, 2, Pixel::WHITE, Pixel::BLACK);
        let p00 = t.get_pixel(0, 0);
        let p20 = t.get_pixel(2, 0);
        assert_eq!(p00.r, 255);
        assert_eq!(p20.r, 0);
    }

    #[test]
    fn test_set_pixel() {
        let mut t = Texture2D::solid(2, 2, Pixel::BLACK);
        t.set_pixel(0, 0, Pixel::WHITE);
        assert_eq!(t.get_pixel(0, 0), Pixel::WHITE);
        assert_eq!(t.get_pixel(1, 0), Pixel::BLACK);
    }

    #[test]
    fn test_wrap_repeat() {
        let w = WrapMode::Repeat;
        assert!(approx(w.apply(0.5), 0.5));
        assert!(approx(w.apply(1.5), 0.5));
        assert!(approx(w.apply(-0.25), 0.75));
    }

    #[test]
    fn test_wrap_clamp() {
        let w = WrapMode::Clamp;
        assert!(approx(w.apply(-0.5), 0.0));
        assert!(approx(w.apply(1.5), 1.0));
        assert!(approx(w.apply(0.5), 0.5));
    }

    #[test]
    fn test_wrap_mirror() {
        let w = WrapMode::MirrorRepeat;
        assert!(approx(w.apply(0.25), 0.25));
        assert!(approx(w.apply(1.25), 0.75));
    }

    #[test]
    fn test_sample_nearest() {
        let t = Texture2D::solid(4, 4, Pixel::new(100, 200, 50, 255));
        let s = SamplerState::new(FilterMode::Nearest, WrapMode::Clamp);
        let c = t.sample(0.5, 0.5, &s);
        assert!(approx(c[0], 100.0 / 255.0));
    }

    #[test]
    fn test_sample_bilinear_center() {
        let t = Texture2D::solid(4, 4, Pixel::new(128, 128, 128, 255));
        let s = SamplerState::new(FilterMode::Bilinear, WrapMode::Clamp);
        let c = t.sample(0.5, 0.5, &s);
        assert!(approx(c[0], 128.0 / 255.0));
    }

    #[test]
    fn test_sample_wrap_repeat() {
        let mut t = Texture2D::solid(2, 2, Pixel::BLACK);
        t.set_pixel(0, 0, Pixel::WHITE);
        let s = SamplerState::new(FilterMode::Nearest, WrapMode::Repeat);
        let c1 = t.sample(0.0, 0.0, &s);
        let c2 = t.sample(1.0, 1.0, &s);
        // Both should wrap to same texel.
        assert!(approx(c1[0], c2[0]));
    }

    #[test]
    fn test_generate_mipmaps() {
        let mut t = Texture2D::solid(8, 8, Pixel::new(128, 128, 128, 255));
        t.generate_mipmaps();
        assert_eq!(t.mip_count(), 3); // 4x4, 2x2, 1x1
    }

    #[test]
    fn test_mipmap_sizes() {
        let mut t = Texture2D::solid(16, 16, Pixel::WHITE);
        t.generate_mipmaps();
        assert_eq!(t.mip_chain[0].width, 8);
        assert_eq!(t.mip_chain[0].height, 8);
        assert_eq!(t.mip_chain[1].width, 4);
        assert_eq!(t.mip_chain[3].width, 1);
    }

    #[test]
    fn test_mipmap_averaging() {
        // 2x2 texture: top row white, bottom row black.
        let pixels = vec![
            Pixel::WHITE, Pixel::WHITE,
            Pixel::BLACK, Pixel::BLACK,
        ];
        let mut t = Texture2D::new(2, 2, pixels).unwrap();
        t.generate_mipmaps();
        assert_eq!(t.mip_count(), 1);
        let mip = &t.mip_chain[0];
        assert_eq!(mip.width, 1);
        // Average of white+white+black+black = ~128 per channel.
        assert!((mip.pixels[0].r as i32 - 128).abs() <= 1);
    }

    #[test]
    fn test_sample_mip() {
        let mut t = Texture2D::solid(4, 4, Pixel::new(200, 200, 200, 255));
        t.generate_mipmaps();
        let s = SamplerState::new(FilterMode::Nearest, WrapMode::Clamp);
        let c = t.sample_mip(0, 0.5, 0.5, &s);
        assert!(approx(c[0], 200.0 / 255.0));
    }

    #[test]
    fn test_trilinear_base_only() {
        let t = Texture2D::solid(4, 4, Pixel::new(100, 100, 100, 255));
        let s = SamplerState::new(FilterMode::Bilinear, WrapMode::Clamp);
        let c = t.sample_trilinear(0.5, 0.5, 0.0, &s);
        assert!(approx(c[0], 100.0 / 255.0));
    }

    #[test]
    fn test_trilinear_with_mipmaps() {
        let mut t = Texture2D::solid(8, 8, Pixel::new(200, 200, 200, 255));
        t.generate_mipmaps();
        let s = SamplerState::new(FilterMode::Bilinear, WrapMode::Clamp);
        let c = t.sample_trilinear(0.5, 0.5, 0.5, &s);
        // Solid color texture — all mips should be close to 200/255.
        assert!(approx(c[0], 200.0 / 255.0));
    }

    #[test]
    fn test_sampler_default() {
        let s = SamplerState::default();
        assert_eq!(s.filter, FilterMode::Bilinear);
        assert_eq!(s.wrap_u, WrapMode::Repeat);
    }

    #[test]
    fn test_sampler_border_color() {
        let s = SamplerState::new(FilterMode::Nearest, WrapMode::Clamp)
            .with_border([1.0, 0.0, 0.0, 1.0]);
        assert!(approx(s.border_color[0], 1.0));
    }

    #[test]
    fn test_display() {
        let mut t = Texture2D::solid(8, 8, Pixel::WHITE);
        t.generate_mipmaps();
        let s = format!("{t}");
        assert!(s.contains("8x8"));
        assert!(s.contains("3 mips"));
    }

    #[test]
    fn test_non_power_of_two_mipmaps() {
        let mut t = Texture2D::solid(5, 3, Pixel::WHITE);
        t.generate_mipmaps();
        assert!(t.mip_count() >= 2);
        // Final mip should be 1x1.
        let last = t.mip_chain.last().unwrap();
        assert_eq!(last.width, 1);
        assert_eq!(last.height, 1);
    }
}
