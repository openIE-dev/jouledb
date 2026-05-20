//! Procedural texture generation.
//!
//! Patterns: checkerboard, bricks, gradients, stripes, dots, Voronoi.
//! Noise-based: wood grain, marble, clouds (fBm). Operations: blend,
//! warp, threshold, invert. Output as RGBA pixel buffer. Tile-seamless
//! generation. Resolution-independent UV sampling.

use std::collections::HashMap;

// ── Color ──

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self { Self { r, g, b, a } }
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self { Self { r, g, b, a: 255 } }
    pub const WHITE: Self = Self::rgb(255, 255, 255);
    pub const BLACK: Self = Self::rgb(0, 0, 0);
    pub const TRANSPARENT: Self = Self::rgba(0, 0, 0, 0);

    pub fn lerp(self, other: Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            r: (self.r as f64 + (other.r as f64 - self.r as f64) * t) as u8,
            g: (self.g as f64 + (other.g as f64 - self.g as f64) * t) as u8,
            b: (self.b as f64 + (other.b as f64 - self.b as f64) * t) as u8,
            a: (self.a as f64 + (other.a as f64 - self.a as f64) * t) as u8,
        }
    }

    pub fn invert(self) -> Self {
        Self { r: 255 - self.r, g: 255 - self.g, b: 255 - self.b, a: self.a }
    }

    fn to_f64(self) -> (f64, f64, f64, f64) {
        (self.r as f64 / 255.0, self.g as f64 / 255.0, self.b as f64 / 255.0, self.a as f64 / 255.0)
    }

    fn from_f64(r: f64, g: f64, b: f64, a: f64) -> Self {
        Self {
            r: (r.clamp(0.0, 1.0) * 255.0) as u8,
            g: (g.clamp(0.0, 1.0) * 255.0) as u8,
            b: (b.clamp(0.0, 1.0) * 255.0) as u8,
            a: (a.clamp(0.0, 1.0) * 255.0) as u8,
        }
    }
}

// ── Texture ──

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Texture {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<Color>,
}

impl Texture {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height, pixels: vec![Color::BLACK; width * height] }
    }

    pub fn get(&self, x: usize, y: usize) -> Color {
        if x < self.width && y < self.height {
            self.pixels[y * self.width + x]
        } else {
            Color::TRANSPARENT
        }
    }

    pub fn set(&mut self, x: usize, y: usize, color: Color) {
        if x < self.width && y < self.height {
            self.pixels[y * self.width + x] = color;
        }
    }

    /// Sample at UV coordinates (0..1, 0..1) with wrapping.
    pub fn sample_uv(&self, u: f64, v: f64) -> Color {
        let u = u.rem_euclid(1.0);
        let v = v.rem_euclid(1.0);
        let x = (u * self.width as f64) as usize % self.width;
        let y = (v * self.height as f64) as usize % self.height;
        self.get(x, y)
    }

    /// Convert to RGBA byte buffer.
    pub fn to_rgba_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.pixels.len() * 4);
        for c in &self.pixels {
            buf.push(c.r);
            buf.push(c.g);
            buf.push(c.b);
            buf.push(c.a);
        }
        buf
    }

    /// Fill from a sampling function over UV space.
    pub fn fill_from<F: Fn(f64, f64) -> Color>(&mut self, sampler: F) {
        for y in 0..self.height {
            for x in 0..self.width {
                let u = x as f64 / self.width as f64;
                let v = y as f64 / self.height as f64;
                self.set(x, y, sampler(u, v));
            }
        }
    }
}

// ── Noise (value noise with hash) ──

fn hash_noise(x: i64, y: i64, seed: u64) -> f64 {
    let mut h = seed.wrapping_add(x as u64).wrapping_mul(0x9e3779b97f4a7c15);
    h = h.wrapping_add(y as u64).wrapping_mul(0xbf58476d1ce4e5b9);
    h = (h ^ (h >> 30)).wrapping_mul(0x94d049bb133111eb);
    h = h ^ (h >> 31);
    (h & 0xFFFF) as f64 / 65535.0
}

fn smooth_noise(x: f64, y: f64, seed: u64) -> f64 {
    let ix = x.floor() as i64;
    let iy = y.floor() as i64;
    let fx = x - x.floor();
    let fy = y - y.floor();

    // Smoothstep
    let sx = fx * fx * (3.0 - 2.0 * fx);
    let sy = fy * fy * (3.0 - 2.0 * fy);

    let n00 = hash_noise(ix, iy, seed);
    let n10 = hash_noise(ix + 1, iy, seed);
    let n01 = hash_noise(ix, iy + 1, seed);
    let n11 = hash_noise(ix + 1, iy + 1, seed);

    let nx0 = n00 + (n10 - n00) * sx;
    let nx1 = n01 + (n11 - n01) * sx;
    nx0 + (nx1 - nx0) * sy
}

/// Fractional Brownian motion (fBm) noise.
pub fn fbm(x: f64, y: f64, octaves: u32, seed: u64) -> f64 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_val = 0.0;

    for i in 0..octaves {
        value += smooth_noise(x * frequency, y * frequency, seed.wrapping_add(i as u64)) * amplitude;
        max_val += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    if max_val > 0.0 { value / max_val } else { 0.0 }
}

fn turbulence(x: f64, y: f64, octaves: u32, seed: u64) -> f64 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_val = 0.0;

    for i in 0..octaves {
        let n = smooth_noise(x * frequency, y * frequency, seed.wrapping_add(i as u64));
        value += (n * 2.0 - 1.0).abs() * amplitude;
        max_val += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    if max_val > 0.0 { value / max_val } else { 0.0 }
}

// ── Pattern generators ──

/// Generate a checkerboard pattern.
pub fn checkerboard(width: usize, height: usize, cell_size: usize, c1: Color, c2: Color) -> Texture {
    let mut tex = Texture::new(width, height);
    let cell_size = cell_size.max(1);
    tex.fill_from(|u, v| {
        let cx = (u * width as f64) as usize / cell_size;
        let cy = (v * height as f64) as usize / cell_size;
        if (cx + cy) % 2 == 0 { c1 } else { c2 }
    });
    tex
}

/// Generate a brick pattern.
pub fn bricks(width: usize, height: usize, brick_w: usize, brick_h: usize, mortar: f64, wall_color: Color, mortar_color: Color) -> Texture {
    let mut tex = Texture::new(width, height);
    let bw = brick_w.max(1) as f64;
    let bh = brick_h.max(1) as f64;
    let mortar = mortar.clamp(0.0, 0.5);
    tex.fill_from(|u, v| {
        let px = u * width as f64;
        let py = v * height as f64;
        let row = (py / bh) as usize;
        let offset = if row % 2 == 1 { bw / 2.0 } else { 0.0 };
        let bx = ((px + offset) % bw) / bw;
        let by = (py % bh) / bh;
        if bx < mortar || bx > 1.0 - mortar || by < mortar || by > 1.0 - mortar {
            mortar_color
        } else {
            wall_color
        }
    });
    tex
}

/// Generate a linear gradient.
pub fn gradient_linear(width: usize, height: usize, c1: Color, c2: Color, horizontal: bool) -> Texture {
    let mut tex = Texture::new(width, height);
    tex.fill_from(|u, v| {
        let t = if horizontal { u } else { v };
        c1.lerp(c2, t)
    });
    tex
}

/// Generate a radial gradient.
pub fn gradient_radial(width: usize, height: usize, c_center: Color, c_edge: Color) -> Texture {
    let mut tex = Texture::new(width, height);
    tex.fill_from(|u, v| {
        let dx = u - 0.5;
        let dy = v - 0.5;
        let dist = (dx * dx + dy * dy).sqrt() * 2.0;
        c_center.lerp(c_edge, dist.min(1.0))
    });
    tex
}

/// Generate stripes.
pub fn stripes(width: usize, height: usize, stripe_width: usize, c1: Color, c2: Color, horizontal: bool) -> Texture {
    let mut tex = Texture::new(width, height);
    let sw = stripe_width.max(1);
    tex.fill_from(|u, v| {
        let pos = if horizontal { (v * height as f64) as usize } else { (u * width as f64) as usize };
        if (pos / sw) % 2 == 0 { c1 } else { c2 }
    });
    tex
}

/// Generate a dot pattern.
pub fn dots(width: usize, height: usize, spacing: usize, radius: f64, dot_color: Color, bg_color: Color) -> Texture {
    let mut tex = Texture::new(width, height);
    let sp = spacing.max(1) as f64;
    tex.fill_from(|u, v| {
        let px = u * width as f64;
        let py = v * height as f64;
        let cx = ((px / sp).round()) * sp;
        let cy = ((py / sp).round()) * sp;
        let dist = ((px - cx).powi(2) + (py - cy).powi(2)).sqrt();
        if dist <= radius { dot_color } else { bg_color }
    });
    tex
}

/// Generate Voronoi cell pattern.
pub fn voronoi(width: usize, height: usize, num_points: usize, seed: u64) -> Texture {
    let mut tex = Texture::new(width, height);

    // Generate random cell centers
    let mut rng_state = seed;
    let mut points: Vec<(f64, f64)> = Vec::new();
    let mut colors: Vec<Color> = Vec::new();

    for _ in 0..num_points {
        rng_state = rng_state.wrapping_add(0x9e3779b97f4a7c15);
        let x = (rng_state & 0xFFFF) as f64 / 65535.0;
        rng_state = rng_state.wrapping_mul(0xbf58476d1ce4e5b9);
        let y = (rng_state & 0xFFFF) as f64 / 65535.0;
        rng_state = rng_state.wrapping_mul(0x94d049bb133111eb);
        let r = ((rng_state >> 0) & 0xFF) as u8;
        let g = ((rng_state >> 8) & 0xFF) as u8;
        let b = ((rng_state >> 16) & 0xFF) as u8;
        points.push((x, y));
        colors.push(Color::rgb(r, g, b));
    }

    tex.fill_from(|u, v| {
        let mut min_dist = f64::MAX;
        let mut closest = 0usize;
        for (i, &(px, py)) in points.iter().enumerate() {
            let d = (u - px).powi(2) + (v - py).powi(2);
            if d < min_dist {
                min_dist = d;
                closest = i;
            }
        }
        colors.get(closest).copied().unwrap_or(Color::BLACK)
    });
    tex
}

// ── Noise-based patterns ──

/// Wood grain texture.
pub fn wood_grain(width: usize, height: usize, rings: f64, seed: u64, c1: Color, c2: Color) -> Texture {
    let mut tex = Texture::new(width, height);
    tex.fill_from(|u, v| {
        let dx = u - 0.5;
        let dy = v - 0.5;
        let dist = (dx * dx + dy * dy).sqrt();
        let noise_val = smooth_noise(u * 8.0, v * 8.0, seed) * 0.1;
        let ring = ((dist + noise_val) * rings).sin() * 0.5 + 0.5;
        c1.lerp(c2, ring)
    });
    tex
}

/// Marble texture.
pub fn marble(width: usize, height: usize, scale: f64, seed: u64, c1: Color, c2: Color) -> Texture {
    let mut tex = Texture::new(width, height);
    tex.fill_from(|u, v| {
        let t = turbulence(u * scale, v * scale, 4, seed);
        let vein = (u * scale + t * 5.0).sin() * 0.5 + 0.5;
        c1.lerp(c2, vein)
    });
    tex
}

/// Cloud texture (fBm).
pub fn clouds(width: usize, height: usize, scale: f64, octaves: u32, seed: u64, c1: Color, c2: Color) -> Texture {
    let mut tex = Texture::new(width, height);
    tex.fill_from(|u, v| {
        let n = fbm(u * scale, v * scale, octaves, seed);
        c1.lerp(c2, n)
    });
    tex
}

// ── Texture operations ──

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    Add,
    Multiply,
    Overlay,
}

/// Blend two textures. They must have the same dimensions.
pub fn blend(a: &Texture, b: &Texture, mode: BlendMode) -> Texture {
    assert_eq!(a.width, b.width);
    assert_eq!(a.height, b.height);
    let mut result = Texture::new(a.width, a.height);
    for i in 0..a.pixels.len() {
        let (ar, ag, ab, aa) = a.pixels[i].to_f64();
        let (br, bg, bb, ba) = b.pixels[i].to_f64();
        let (rr, rg, rb) = match mode {
            BlendMode::Add => ((ar + br).min(1.0), (ag + bg).min(1.0), (ab + bb).min(1.0)),
            BlendMode::Multiply => (ar * br, ag * bg, ab * bb),
            BlendMode::Overlay => {
                let ov = |base: f64, top: f64| -> f64 {
                    if base < 0.5 { 2.0 * base * top } else { 1.0 - 2.0 * (1.0 - base) * (1.0 - top) }
                };
                (ov(ar, br), ov(ag, bg), ov(ab, bb))
            }
        };
        result.pixels[i] = Color::from_f64(rr, rg, rb, (aa + ba).min(1.0));
    }
    result
}

/// Warp a texture by displacing UV coordinates using noise.
pub fn warp(tex: &Texture, strength: f64, seed: u64) -> Texture {
    let mut result = Texture::new(tex.width, tex.height);
    for y in 0..tex.height {
        for x in 0..tex.width {
            let u = x as f64 / tex.width as f64;
            let v = y as f64 / tex.height as f64;
            let dx = smooth_noise(u * 10.0, v * 10.0, seed) * 2.0 - 1.0;
            let dy = smooth_noise(u * 10.0 + 100.0, v * 10.0 + 100.0, seed) * 2.0 - 1.0;
            let su = u + dx * strength;
            let sv = v + dy * strength;
            result.set(x, y, tex.sample_uv(su, sv));
        }
    }
    result
}

/// Apply a threshold to convert to two colors.
pub fn threshold(tex: &Texture, thresh: f64, c_below: Color, c_above: Color) -> Texture {
    let mut result = Texture::new(tex.width, tex.height);
    for i in 0..tex.pixels.len() {
        let (r, g, b, _) = tex.pixels[i].to_f64();
        let luminance = r * 0.299 + g * 0.587 + b * 0.114;
        result.pixels[i] = if luminance < thresh { c_below } else { c_above };
    }
    result
}

/// Invert all pixel colors.
pub fn invert(tex: &Texture) -> Texture {
    let mut result = Texture::new(tex.width, tex.height);
    for i in 0..tex.pixels.len() {
        result.pixels[i] = tex.pixels[i].invert();
    }
    result
}

/// Generate a tile-seamless version of a pattern by blending borders.
pub fn make_seamless(tex: &Texture, blend_width: usize) -> Texture {
    let mut result = tex.clone();
    let bw = blend_width.min(tex.width / 2).min(tex.height / 2);

    for y in 0..tex.height {
        for x in 0..bw {
            let t = x as f64 / bw as f64;
            let orig = tex.get(x, y);
            let mirror = tex.get(tex.width - 1 - x, y);
            result.set(x, y, mirror.lerp(orig, t));
        }
    }
    for x in 0..tex.width {
        for y in 0..bw {
            let t = y as f64 / bw as f64;
            let orig = result.get(x, y);
            let mirror = result.get(x, tex.height - 1 - y);
            result.set(x, y, mirror.lerp(orig, t));
        }
    }
    result
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_lerp() {
        let c = Color::BLACK.lerp(Color::WHITE, 0.5);
        assert!((c.r as i16 - 127).abs() <= 1);
        assert!((c.g as i16 - 127).abs() <= 1);
    }

    #[test]
    fn test_color_invert() {
        let c = Color::rgb(100, 150, 200).invert();
        assert_eq!(c.r, 155);
        assert_eq!(c.g, 105);
        assert_eq!(c.b, 55);
    }

    #[test]
    fn test_texture_new() {
        let tex = Texture::new(16, 16);
        assert_eq!(tex.pixels.len(), 256);
        assert_eq!(tex.get(0, 0), Color::BLACK);
    }

    #[test]
    fn test_texture_set_get() {
        let mut tex = Texture::new(8, 8);
        tex.set(3, 4, Color::WHITE);
        assert_eq!(tex.get(3, 4), Color::WHITE);
        assert_eq!(tex.get(0, 0), Color::BLACK);
    }

    #[test]
    fn test_sample_uv_wrapping() {
        let mut tex = Texture::new(4, 4);
        tex.set(0, 0, Color::rgb(255, 0, 0));
        let c = tex.sample_uv(1.0, 1.0); // wraps to (0,0)
        assert_eq!(c.r, 255);
    }

    #[test]
    fn test_to_rgba_bytes() {
        let mut tex = Texture::new(2, 2);
        tex.set(0, 0, Color::rgba(1, 2, 3, 4));
        let bytes = tex.to_rgba_bytes();
        assert_eq!(bytes.len(), 16);
        assert_eq!(&bytes[0..4], &[1, 2, 3, 4]);
    }

    #[test]
    fn test_checkerboard() {
        let tex = checkerboard(16, 16, 4, Color::WHITE, Color::BLACK);
        assert_eq!(tex.get(0, 0), Color::WHITE);
        assert_eq!(tex.get(4, 0), Color::BLACK);
    }

    #[test]
    fn test_bricks() {
        let tex = bricks(32, 32, 8, 4, 0.1, Color::rgb(180, 80, 60), Color::rgb(120, 120, 120));
        assert_eq!(tex.width, 32);
        // Check mortar exists (grey pixels)
        let has_mortar = tex.pixels.iter().any(|c| c.r == 120);
        assert!(has_mortar);
    }

    #[test]
    fn test_gradient_linear() {
        let tex = gradient_linear(64, 1, Color::BLACK, Color::WHITE, true);
        let left = tex.get(0, 0);
        let right = tex.get(63, 0);
        assert!(right.r > left.r);
    }

    #[test]
    fn test_gradient_radial() {
        let tex = gradient_radial(32, 32, Color::WHITE, Color::BLACK);
        let center = tex.get(16, 16);
        let corner = tex.get(0, 0);
        assert!(center.r > corner.r);
    }

    #[test]
    fn test_stripes() {
        let tex = stripes(16, 16, 4, Color::WHITE, Color::BLACK, false);
        assert_eq!(tex.get(0, 0), Color::WHITE);
        assert_eq!(tex.get(4, 0), Color::BLACK);
    }

    #[test]
    fn test_dots() {
        let tex = dots(32, 32, 8, 2.0, Color::WHITE, Color::BLACK);
        // Should have some white pixels (dots)
        let white_count = tex.pixels.iter().filter(|c| c.r == 255).count();
        assert!(white_count > 0);
    }

    #[test]
    fn test_voronoi() {
        let tex = voronoi(32, 32, 5, 42);
        // Should have multiple colors
        let unique: std::collections::HashSet<(u8, u8, u8)> =
            tex.pixels.iter().map(|c| (c.r, c.g, c.b)).collect();
        assert!(unique.len() > 1);
    }

    #[test]
    fn test_wood_grain() {
        let tex = wood_grain(32, 32, 10.0, 42, Color::rgb(139, 90, 43), Color::rgb(210, 160, 80));
        assert_eq!(tex.width, 32);
        let unique: std::collections::HashSet<(u8, u8, u8)> =
            tex.pixels.iter().map(|c| (c.r, c.g, c.b)).collect();
        assert!(unique.len() > 2);
    }

    #[test]
    fn test_marble() {
        let tex = marble(32, 32, 4.0, 42, Color::WHITE, Color::rgb(50, 50, 50));
        assert_eq!(tex.pixels.len(), 32 * 32);
    }

    #[test]
    fn test_clouds() {
        let tex = clouds(32, 32, 4.0, 4, 42, Color::rgb(135, 206, 235), Color::WHITE);
        assert_eq!(tex.width, 32);
    }

    #[test]
    fn test_blend_add() {
        let a = checkerboard(8, 8, 4, Color::rgb(100, 0, 0), Color::BLACK);
        let b = checkerboard(8, 8, 4, Color::rgb(0, 100, 0), Color::BLACK);
        let c = blend(&a, &b, BlendMode::Add);
        let p = c.get(0, 0);
        assert!(p.r > 0 || p.g > 0);
    }

    #[test]
    fn test_blend_multiply() {
        let a = gradient_linear(8, 8, Color::WHITE, Color::BLACK, true);
        let b = gradient_linear(8, 8, Color::rgb(255, 0, 0), Color::rgb(255, 0, 0), true);
        let c = blend(&a, &b, BlendMode::Multiply);
        assert_eq!(c.width, 8);
    }

    #[test]
    fn test_threshold_op() {
        let tex = gradient_linear(16, 1, Color::BLACK, Color::WHITE, true);
        let th = threshold(&tex, 0.5, Color::BLACK, Color::WHITE);
        assert_eq!(th.get(0, 0), Color::BLACK);
        assert_eq!(th.get(15, 0), Color::WHITE);
    }

    #[test]
    fn test_invert_op() {
        let tex = checkerboard(4, 4, 2, Color::WHITE, Color::BLACK);
        let inv = invert(&tex);
        assert_eq!(inv.get(0, 0), Color::BLACK);
        assert_eq!(inv.get(2, 0), Color::WHITE);
    }

    #[test]
    fn test_warp() {
        let tex = checkerboard(16, 16, 4, Color::WHITE, Color::BLACK);
        let warped = warp(&tex, 0.1, 42);
        assert_eq!(warped.width, 16);
    }

    #[test]
    fn test_make_seamless() {
        let tex = clouds(32, 32, 4.0, 3, 42, Color::BLACK, Color::WHITE);
        let seamless = make_seamless(&tex, 4);
        assert_eq!(seamless.width, 32);
    }

    #[test]
    fn test_fbm_range() {
        for i in 0..100 {
            let val = fbm(i as f64 * 0.1, i as f64 * 0.07, 4, 42);
            assert!(val >= -0.1 && val <= 1.1, "fbm out of expected range: {}", val);
        }
    }

    #[test]
    fn test_color_from_f64_clamping() {
        let c = Color::from_f64(2.0, -1.0, 0.5, 1.0);
        assert_eq!(c.r, 255);
        assert_eq!(c.g, 0);
        assert_eq!(c.b, 127);
    }
}
