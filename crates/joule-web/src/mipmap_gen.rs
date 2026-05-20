// mipmap_gen.rs — Mipmap chain generation.
// Box filter, Kaiser filter, sRGB-aware, alpha-weighted, normal map mipmaps.

use std::f32::consts::PI;

/// RGBA pixel as f32.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pixel {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Pixel {
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub fn black() -> Self {
        Self::new(0.0, 0.0, 0.0, 1.0)
    }

    pub fn add(&self, other: &Pixel) -> Pixel {
        Pixel {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
            a: self.a + other.a,
        }
    }

    pub fn scale(&self, f: f32) -> Pixel {
        Pixel {
            r: self.r * f,
            g: self.g * f,
            b: self.b * f,
            a: self.a * f,
        }
    }

    pub fn clamped(&self) -> Pixel {
        Pixel {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
            a: self.a.clamp(0.0, 1.0),
        }
    }
}

/// Image buffer (a single mip level).
#[derive(Debug, Clone, PartialEq)]
pub struct MipLevel {
    pub pixels: Vec<Pixel>,
    pub width: usize,
    pub height: usize,
}

impl MipLevel {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            pixels: vec![Pixel::black(); width * height],
            width,
            height,
        }
    }

    pub fn from_pixels(pixels: Vec<Pixel>, width: usize, height: usize) -> Option<Self> {
        if pixels.len() != width * height {
            return None;
        }
        Some(Self { pixels, width, height })
    }

    pub fn get(&self, x: usize, y: usize) -> Pixel {
        self.pixels[y * self.width + x]
    }

    pub fn get_clamped(&self, x: isize, y: isize) -> Pixel {
        let cx = x.clamp(0, self.width as isize - 1) as usize;
        let cy = y.clamp(0, self.height as isize - 1) as usize;
        self.pixels[cy * self.width + cx]
    }

    pub fn set(&mut self, x: usize, y: usize, p: Pixel) {
        if x < self.width && y < self.height {
            self.pixels[y * self.width + x] = p;
        }
    }
}

/// Mipmap chain — a sequence of MipLevels from full resolution down to 1x1.
#[derive(Debug, Clone)]
pub struct MipChain {
    pub levels: Vec<MipLevel>,
}

impl MipChain {
    pub fn new() -> Self {
        Self { levels: Vec::new() }
    }

    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    pub fn base_level(&self) -> Option<&MipLevel> {
        self.levels.first()
    }
}

/// Compute the number of mip levels for a given resolution.
pub fn mip_level_count(width: usize, height: usize) -> usize {
    if width == 0 || height == 0 {
        return 0;
    }
    let max_dim = width.max(height);
    // floor(log2(max_dim)) + 1
    let mut levels = 1usize;
    let mut dim = max_dim;
    while dim > 1 {
        dim /= 2;
        levels += 1;
    }
    levels
}

/// Compute the dimensions of a specific mip level.
pub fn mip_dimensions(base_width: usize, base_height: usize, level: usize) -> (usize, usize) {
    let w = (base_width >> level).max(1);
    let h = (base_height >> level).max(1);
    (w, h)
}

/// Mip filter mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MipFilter {
    /// Simple 2x2 box filter (average).
    Box,
    /// Higher-quality Kaiser windowed sinc filter.
    Kaiser,
}

/// Generate a single downsampled level using box filter (2x2 average).
pub fn downsample_box(src: &MipLevel) -> MipLevel {
    let new_w = (src.width / 2).max(1);
    let new_h = (src.height / 2).max(1);
    let mut dst = MipLevel::new(new_w, new_h);

    for y in 0..new_h {
        for x in 0..new_w {
            let sx = x * 2;
            let sy = y * 2;

            let p00 = src.get_clamped(sx as isize, sy as isize);
            let p10 = src.get_clamped(sx as isize + 1, sy as isize);
            let p01 = src.get_clamped(sx as isize, sy as isize + 1);
            let p11 = src.get_clamped(sx as isize + 1, sy as isize + 1);

            let avg = p00.add(&p10).add(&p01).add(&p11).scale(0.25);
            dst.set(x, y, avg);
        }
    }

    dst
}

/// Kaiser windowed sinc function.
fn kaiser_window(x: f32, width: f32, beta: f32) -> f32 {
    if x.abs() > width {
        return 0.0;
    }
    let t = x / width;
    let arg = (1.0 - t * t).max(0.0).sqrt() * beta;
    bessel_i0(arg) / bessel_i0(beta)
}

/// Approximate modified Bessel function of the first kind, order 0.
fn bessel_i0(x: f32) -> f32 {
    let mut sum = 1.0f32;
    let mut term = 1.0f32;
    let x2 = x * x;
    for k in 1..20 {
        term *= x2 / (4.0 * (k as f32) * (k as f32));
        sum += term;
        if term < 1e-10 {
            break;
        }
    }
    sum
}

/// Sinc function.
fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-6 {
        1.0
    } else {
        let px = PI * x;
        px.sin() / px
    }
}

/// Generate a single downsampled level using Kaiser filter (wider, higher quality).
pub fn downsample_kaiser(src: &MipLevel) -> MipLevel {
    let new_w = (src.width / 2).max(1);
    let new_h = (src.height / 2).max(1);
    let mut dst = MipLevel::new(new_w, new_h);

    let filter_radius = 3.0f32;
    let beta = 4.0f32;
    let radius = filter_radius.ceil() as isize;

    for y in 0..new_h {
        for x in 0..new_w {
            let center_x = (x as f32 + 0.5) * 2.0 - 0.5;
            let center_y = (y as f32 + 0.5) * 2.0 - 0.5;

            let mut accum = Pixel::new(0.0, 0.0, 0.0, 0.0);
            let mut total_weight = 0.0f32;

            let ix = center_x.floor() as isize;
            let iy = center_y.floor() as isize;

            for jj in (-radius)..=radius {
                let sy = iy + jj;
                let dy = center_y - sy as f32;
                let wy = sinc(dy / 2.0) * kaiser_window(dy / 2.0, filter_radius, beta);

                for ii in (-radius)..=radius {
                    let sx = ix + ii;
                    let dx = center_x - sx as f32;
                    let wx = sinc(dx / 2.0) * kaiser_window(dx / 2.0, filter_radius, beta);

                    let w = wx * wy;
                    if w.abs() < 1e-10 {
                        continue;
                    }

                    let p = src.get_clamped(sx, sy);
                    accum = accum.add(&p.scale(w));
                    total_weight += w;
                }
            }

            if total_weight.abs() > 1e-10 {
                accum = accum.scale(1.0 / total_weight);
            }
            dst.set(x, y, accum.clamped());
        }
    }

    dst
}

/// Generate a complete mip chain using the specified filter.
pub fn generate_mip_chain(base: &MipLevel, filter: MipFilter) -> MipChain {
    let count = mip_level_count(base.width, base.height);
    let mut chain = MipChain::new();
    chain.levels.push(base.clone());

    let downsample_fn = match filter {
        MipFilter::Box => downsample_box,
        MipFilter::Kaiser => downsample_kaiser,
    };

    for _ in 1..count {
        let prev = chain.levels.last().unwrap();
        if prev.width <= 1 && prev.height <= 1 {
            break;
        }
        let next = downsample_fn(prev);
        chain.levels.push(next);
    }

    chain
}

/// sRGB gamma to linear conversion.
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear to sRGB gamma conversion.
pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Convert a pixel from sRGB to linear space.
pub fn pixel_to_linear(p: &Pixel) -> Pixel {
    Pixel {
        r: srgb_to_linear(p.r),
        g: srgb_to_linear(p.g),
        b: srgb_to_linear(p.b),
        a: p.a, // Alpha is already linear.
    }
}

/// Convert a pixel from linear to sRGB space.
pub fn pixel_to_srgb(p: &Pixel) -> Pixel {
    Pixel {
        r: linear_to_srgb(p.r),
        g: linear_to_srgb(p.g),
        b: linear_to_srgb(p.b),
        a: p.a,
    }
}

/// sRGB-aware mipmap generation: linearize, filter, re-encode.
pub fn generate_mip_chain_srgb(base: &MipLevel, filter: MipFilter) -> MipChain {
    // Linearize the base level.
    let linear_base = {
        let mut level = base.clone();
        for p in &mut level.pixels {
            *p = pixel_to_linear(p);
        }
        level
    };

    // Generate chain in linear space.
    let linear_chain = generate_mip_chain(&linear_base, filter);

    // Convert all levels back to sRGB.
    let mut srgb_chain = MipChain::new();
    for level in &linear_chain.levels {
        let mut srgb_level = level.clone();
        for p in &mut srgb_level.pixels {
            *p = pixel_to_srgb(p);
        }
        srgb_chain.levels.push(srgb_level);
    }

    srgb_chain
}

/// Normalize a 3D vector stored in RGB (for normal maps).
fn normalize_normal(p: &Pixel) -> Pixel {
    let nx = p.r * 2.0 - 1.0;
    let ny = p.g * 2.0 - 1.0;
    let nz = p.b * 2.0 - 1.0;
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len < 1e-6 {
        return Pixel::new(0.5, 0.5, 1.0, p.a); // Default up normal.
    }
    Pixel {
        r: (nx / len) * 0.5 + 0.5,
        g: (ny / len) * 0.5 + 0.5,
        b: (nz / len) * 0.5 + 0.5,
        a: p.a,
    }
}

/// Generate mip chain for a normal map — renormalize after filtering.
pub fn generate_normal_map_mip_chain(base: &MipLevel) -> MipChain {
    let chain = generate_mip_chain(base, MipFilter::Box);
    let mut result = MipChain::new();

    for (i, level) in chain.levels.iter().enumerate() {
        if i == 0 {
            result.levels.push(level.clone());
            continue;
        }
        let mut normalized = level.clone();
        for p in &mut normalized.pixels {
            *p = normalize_normal(p);
        }
        result.levels.push(normalized);
    }

    result
}

/// Alpha-weighted mipmapping for cutout textures.
/// Premultiply alpha before filtering, then recover coverage.
pub fn downsample_alpha_weighted(src: &MipLevel) -> MipLevel {
    let new_w = (src.width / 2).max(1);
    let new_h = (src.height / 2).max(1);
    let mut dst = MipLevel::new(new_w, new_h);

    for y in 0..new_h {
        for x in 0..new_w {
            let sx = x * 2;
            let sy = y * 2;

            let mut r_sum = 0.0f32;
            let mut g_sum = 0.0f32;
            let mut b_sum = 0.0f32;
            let mut a_sum = 0.0f32;

            let coords = [
                (sx as isize, sy as isize),
                (sx as isize + 1, sy as isize),
                (sx as isize, sy as isize + 1),
                (sx as isize + 1, sy as isize + 1),
            ];

            for &(cx, cy) in &coords {
                let p = src.get_clamped(cx, cy);
                // Premultiply by alpha.
                r_sum += p.r * p.a;
                g_sum += p.g * p.a;
                b_sum += p.b * p.a;
                a_sum += p.a;
            }

            let avg_alpha = a_sum / 4.0;

            if avg_alpha > 1e-6 {
                // Recover straight color from premultiplied sum.
                dst.set(x, y, Pixel::new(
                    (r_sum / a_sum).clamp(0.0, 1.0),
                    (g_sum / a_sum).clamp(0.0, 1.0),
                    (b_sum / a_sum).clamp(0.0, 1.0),
                    avg_alpha,
                ));
            } else {
                dst.set(x, y, Pixel::new(0.0, 0.0, 0.0, 0.0));
            }
        }
    }

    dst
}

/// Generate alpha-weighted mip chain for cutout textures.
pub fn generate_alpha_weighted_mip_chain(base: &MipLevel) -> MipChain {
    let count = mip_level_count(base.width, base.height);
    let mut chain = MipChain::new();
    chain.levels.push(base.clone());

    for _ in 1..count {
        let prev = chain.levels.last().unwrap();
        if prev.width <= 1 && prev.height <= 1 {
            break;
        }
        let next = downsample_alpha_weighted(prev);
        chain.levels.push(next);
    }

    chain
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_level(w: usize, h: usize, c: Pixel) -> MipLevel {
        MipLevel {
            pixels: vec![c; w * h],
            width: w,
            height: h,
        }
    }

    #[test]
    fn test_mip_level_count_powers_of_two() {
        assert_eq!(mip_level_count(1, 1), 1);
        assert_eq!(mip_level_count(2, 2), 2);
        assert_eq!(mip_level_count(4, 4), 3);
        assert_eq!(mip_level_count(256, 256), 9);
    }

    #[test]
    fn test_mip_level_count_non_power() {
        assert_eq!(mip_level_count(3, 3), 2);
        assert_eq!(mip_level_count(5, 5), 3);
    }

    #[test]
    fn test_mip_level_count_rectangular() {
        assert_eq!(mip_level_count(16, 4), 5);
        assert_eq!(mip_level_count(1, 8), 4);
    }

    #[test]
    fn test_mip_level_count_zero() {
        assert_eq!(mip_level_count(0, 0), 0);
        assert_eq!(mip_level_count(0, 5), 0);
    }

    #[test]
    fn test_mip_dimensions() {
        assert_eq!(mip_dimensions(256, 256, 0), (256, 256));
        assert_eq!(mip_dimensions(256, 256, 1), (128, 128));
        assert_eq!(mip_dimensions(256, 256, 8), (1, 1));
        assert_eq!(mip_dimensions(256, 256, 20), (1, 1));
    }

    #[test]
    fn test_downsample_box_dimensions() {
        let src = solid_level(8, 8, Pixel::new(0.5, 0.5, 0.5, 1.0));
        let dst = downsample_box(&src);
        assert_eq!(dst.width, 4);
        assert_eq!(dst.height, 4);
    }

    #[test]
    fn test_downsample_box_solid() {
        let src = solid_level(8, 8, Pixel::new(0.5, 0.5, 0.5, 1.0));
        let dst = downsample_box(&src);
        for p in &dst.pixels {
            assert!((p.r - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_downsample_box_averages() {
        let mut src = MipLevel::new(2, 2);
        src.set(0, 0, Pixel::new(1.0, 0.0, 0.0, 1.0));
        src.set(1, 0, Pixel::new(0.0, 1.0, 0.0, 1.0));
        src.set(0, 1, Pixel::new(0.0, 0.0, 1.0, 1.0));
        src.set(1, 1, Pixel::new(1.0, 1.0, 1.0, 1.0));

        let dst = downsample_box(&src);
        assert_eq!(dst.width, 1);
        assert_eq!(dst.height, 1);
        let p = dst.get(0, 0);
        assert!((p.r - 0.5).abs() < 1e-6);
        assert!((p.g - 0.5).abs() < 1e-6);
        assert!((p.b - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_downsample_kaiser_dimensions() {
        let src = solid_level(8, 8, Pixel::new(0.5, 0.5, 0.5, 1.0));
        let dst = downsample_kaiser(&src);
        assert_eq!(dst.width, 4);
        assert_eq!(dst.height, 4);
    }

    #[test]
    fn test_downsample_kaiser_solid() {
        let src = solid_level(8, 8, Pixel::new(0.4, 0.4, 0.4, 1.0));
        let dst = downsample_kaiser(&src);
        for p in &dst.pixels {
            assert!((p.r - 0.4).abs() < 0.01, "kaiser solid deviation: {}", p.r);
        }
    }

    #[test]
    fn test_generate_mip_chain_box() {
        let base = solid_level(16, 16, Pixel::new(0.5, 0.5, 0.5, 1.0));
        let chain = generate_mip_chain(&base, MipFilter::Box);
        assert_eq!(chain.level_count(), 5); // 16 -> 8 -> 4 -> 2 -> 1
        assert_eq!(chain.levels[0].width, 16);
        assert_eq!(chain.levels[1].width, 8);
        assert_eq!(chain.levels[4].width, 1);
    }

    #[test]
    fn test_generate_mip_chain_kaiser() {
        let base = solid_level(8, 8, Pixel::new(0.5, 0.5, 0.5, 1.0));
        let chain = generate_mip_chain(&base, MipFilter::Kaiser);
        assert!(chain.level_count() >= 3);
    }

    #[test]
    fn test_mip_chain_values_preserved() {
        let base = solid_level(8, 8, Pixel::new(0.7, 0.7, 0.7, 1.0));
        let chain = generate_mip_chain(&base, MipFilter::Box);
        for level in &chain.levels {
            for p in &level.pixels {
                assert!((p.r - 0.7).abs() < 1e-4, "solid mip should stay: {}", p.r);
            }
        }
    }

    #[test]
    fn test_srgb_roundtrip() {
        for i in 0..=10 {
            let v = i as f32 / 10.0;
            let linear = srgb_to_linear(v);
            let back = linear_to_srgb(linear);
            assert!((back - v).abs() < 1e-4, "sRGB roundtrip failed for {}: got {}", v, back);
        }
    }

    #[test]
    fn test_srgb_to_linear_bounds() {
        assert!(srgb_to_linear(0.0).abs() < 1e-6);
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-4);
        // Mid-gray should be darker in linear.
        let mid = srgb_to_linear(0.5);
        assert!(mid < 0.5 && mid > 0.1);
    }

    #[test]
    fn test_srgb_mip_chain() {
        let base = solid_level(8, 8, Pixel::new(0.5, 0.5, 0.5, 1.0));
        let chain = generate_mip_chain_srgb(&base, MipFilter::Box);
        assert!(chain.level_count() >= 3);
        // Solid color should survive sRGB round-trip.
        for level in &chain.levels {
            for p in &level.pixels {
                assert!((p.r - 0.5).abs() < 0.02, "sRGB mip solid deviation: {}", p.r);
            }
        }
    }

    #[test]
    fn test_normalize_normal() {
        // Pixel encoding up normal: (0.5, 0.5, 1.0) = (0, 0, 1) in normal space.
        let up = Pixel::new(0.5, 0.5, 1.0, 1.0);
        let n = normalize_normal(&up);
        assert!((n.b - 1.0).abs() < 1e-3, "up normal z should be ~1.0: {}", n.b);
        assert!((n.r - 0.5).abs() < 1e-3);
    }

    #[test]
    fn test_normal_map_mip_chain() {
        let base = solid_level(8, 8, Pixel::new(0.5, 0.5, 1.0, 1.0));
        let chain = generate_normal_map_mip_chain(&base);
        assert!(chain.level_count() >= 3);
        // Each level should have unit-length normals (encoded in [0,1]).
        for level in chain.levels.iter().skip(1) {
            for p in &level.pixels {
                let nx = p.r * 2.0 - 1.0;
                let ny = p.g * 2.0 - 1.0;
                let nz = p.b * 2.0 - 1.0;
                let len = (nx * nx + ny * ny + nz * nz).sqrt();
                assert!((len - 1.0).abs() < 0.05, "normal should be unit length: {}", len);
            }
        }
    }

    #[test]
    fn test_alpha_weighted_downsample() {
        let mut src = MipLevel::new(2, 2);
        src.set(0, 0, Pixel::new(1.0, 0.0, 0.0, 1.0));
        src.set(1, 0, Pixel::new(1.0, 0.0, 0.0, 1.0));
        src.set(0, 1, Pixel::new(0.0, 0.0, 0.0, 0.0)); // Transparent.
        src.set(1, 1, Pixel::new(0.0, 0.0, 0.0, 0.0)); // Transparent.

        let dst = downsample_alpha_weighted(&src);
        let p = dst.get(0, 0);
        // Alpha should be 0.5 (average of 1.0, 1.0, 0.0, 0.0).
        assert!((p.a - 0.5).abs() < 1e-6);
        // Color should be red (only opaque pixels contribute).
        assert!((p.r - 1.0).abs() < 1e-6, "red should dominate: {}", p.r);
    }

    #[test]
    fn test_alpha_weighted_all_transparent() {
        let src = solid_level(2, 2, Pixel::new(1.0, 0.0, 0.0, 0.0));
        let dst = downsample_alpha_weighted(&src);
        let p = dst.get(0, 0);
        assert!(p.a.abs() < 1e-6);
    }

    #[test]
    fn test_alpha_weighted_mip_chain() {
        let base = solid_level(8, 8, Pixel::new(0.5, 0.5, 0.5, 0.8));
        let chain = generate_alpha_weighted_mip_chain(&base);
        assert!(chain.level_count() >= 3);
    }

    #[test]
    fn test_mip_chain_base_level() {
        let base = solid_level(4, 4, Pixel::black());
        let chain = generate_mip_chain(&base, MipFilter::Box);
        let bl = chain.base_level().unwrap();
        assert_eq!(bl.width, 4);
        assert_eq!(bl.height, 4);
    }

    #[test]
    fn test_bessel_i0_at_zero() {
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sinc_at_zero() {
        assert!((sinc(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sinc_at_integer() {
        assert!(sinc(1.0).abs() < 1e-5);
        assert!(sinc(2.0).abs() < 1e-5);
    }

    #[test]
    fn test_kaiser_window_center() {
        let w = kaiser_window(0.0, 3.0, 4.0);
        assert!((w - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_kaiser_window_outside() {
        let w = kaiser_window(5.0, 3.0, 4.0);
        assert!(w.abs() < 1e-6);
    }
}
