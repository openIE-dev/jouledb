// upscaler.rs — Image upscaling algorithms.
// Nearest neighbor, bilinear, bicubic (Catmull-Rom), Lanczos (2/3 lobe), edge-directed.

use std::f32::consts::PI;

/// Pixel as f32 RGBA.
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

    pub fn luminance(&self) -> f32 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
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

/// Image buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageBuf {
    pub pixels: Vec<Pixel>,
    pub width: usize,
    pub height: usize,
}

impl ImageBuf {
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

/// Upscaling algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpscaleMethod {
    NearestNeighbor,
    Bilinear,
    Bicubic,
    Lanczos2,
    Lanczos3,
    EdgeDirected,
}

/// Quality vs performance tradeoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpscaleQuality {
    Fast,
    Balanced,
    HighQuality,
}

impl UpscaleQuality {
    /// Suggest an upscale method for this quality level.
    pub fn suggested_method(&self) -> UpscaleMethod {
        match self {
            UpscaleQuality::Fast => UpscaleMethod::NearestNeighbor,
            UpscaleQuality::Balanced => UpscaleMethod::Bilinear,
            UpscaleQuality::HighQuality => UpscaleMethod::Lanczos3,
        }
    }
}

/// Scale factor for upscaling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleFactor(pub f32);

impl ScaleFactor {
    pub fn new(factor: f32) -> Self {
        Self(factor.max(1.0))
    }

    pub fn x1_5() -> Self { Self(1.5) }
    pub fn x2() -> Self { Self(2.0) }
    pub fn x4() -> Self { Self(4.0) }

    pub fn output_size(&self, width: usize, height: usize) -> (usize, usize) {
        let w = (width as f32 * self.0).round() as usize;
        let h = (height as f32 * self.0).round() as usize;
        (w.max(1), h.max(1))
    }
}

/// Nearest neighbor upscale.
pub fn upscale_nearest(src: &ImageBuf, scale: ScaleFactor) -> ImageBuf {
    let (ow, oh) = scale.output_size(src.width, src.height);
    let mut dst = ImageBuf::new(ow, oh);

    for y in 0..oh {
        for x in 0..ow {
            let sx = (x as f32 / scale.0) as usize;
            let sy = (y as f32 / scale.0) as usize;
            let sx = sx.min(src.width - 1);
            let sy = sy.min(src.height - 1);
            dst.set(x, y, src.get(sx, sy));
        }
    }

    dst
}

/// Bilinear interpolation upscale.
pub fn upscale_bilinear(src: &ImageBuf, scale: ScaleFactor) -> ImageBuf {
    let (ow, oh) = scale.output_size(src.width, src.height);
    let mut dst = ImageBuf::new(ow, oh);

    for y in 0..oh {
        for x in 0..ow {
            let fx = x as f32 / scale.0 - 0.5;
            let fy = y as f32 / scale.0 - 0.5;

            let x0 = fx.floor() as isize;
            let y0 = fy.floor() as isize;
            let frac_x = fx - fx.floor();
            let frac_y = fy - fy.floor();

            let c00 = src.get_clamped(x0, y0);
            let c10 = src.get_clamped(x0 + 1, y0);
            let c01 = src.get_clamped(x0, y0 + 1);
            let c11 = src.get_clamped(x0 + 1, y0 + 1);

            let top = pixel_lerp(&c00, &c10, frac_x);
            let bot = pixel_lerp(&c01, &c11, frac_x);
            dst.set(x, y, pixel_lerp(&top, &bot, frac_y));
        }
    }

    dst
}

fn pixel_lerp(a: &Pixel, b: &Pixel, t: f32) -> Pixel {
    Pixel {
        r: a.r + (b.r - a.r) * t,
        g: a.g + (b.g - a.g) * t,
        b: a.b + (b.b - a.b) * t,
        a: a.a + (b.a - a.a) * t,
    }
}

/// Catmull-Rom cubic interpolation weight.
fn catmull_rom(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 1.0 {
        (3.0 * ax * ax * ax - 5.0 * ax * ax + 2.0) / 2.0
    } else if ax < 2.0 {
        (-ax * ax * ax + 5.0 * ax * ax - 8.0 * ax + 4.0) / 2.0
    } else {
        0.0
    }
}

/// Bicubic (Catmull-Rom) upscale.
pub fn upscale_bicubic(src: &ImageBuf, scale: ScaleFactor) -> ImageBuf {
    let (ow, oh) = scale.output_size(src.width, src.height);
    let mut dst = ImageBuf::new(ow, oh);

    for y in 0..oh {
        for x in 0..ow {
            let fx = x as f32 / scale.0 - 0.5;
            let fy = y as f32 / scale.0 - 0.5;

            let ix = fx.floor() as isize;
            let iy = fy.floor() as isize;
            let frac_x = fx - fx.floor();
            let frac_y = fy - fy.floor();

            let mut accum = Pixel::new(0.0, 0.0, 0.0, 0.0);
            let mut total_weight = 0.0f32;

            for jj in -1isize..=2 {
                let wy = catmull_rom(frac_y - jj as f32);
                for ii in -1isize..=2 {
                    let wx = catmull_rom(frac_x - ii as f32);
                    let w = wx * wy;
                    let p = src.get_clamped(ix + ii, iy + jj);
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

/// Lanczos kernel.
fn lanczos_kernel(x: f32, lobes: f32) -> f32 {
    if x.abs() < 1e-6 {
        return 1.0;
    }
    if x.abs() >= lobes {
        return 0.0;
    }
    let px = PI * x;
    (px.sin() / px) * ((px / lobes).sin() / (px / lobes))
}

/// Generic Lanczos upscale with configurable lobe count.
fn upscale_lanczos_generic(src: &ImageBuf, scale: ScaleFactor, lobes: usize) -> ImageBuf {
    let (ow, oh) = scale.output_size(src.width, src.height);
    let mut dst = ImageBuf::new(ow, oh);
    let lobes_f = lobes as f32;

    for y in 0..oh {
        for x in 0..ow {
            let fx = x as f32 / scale.0 - 0.5;
            let fy = y as f32 / scale.0 - 0.5;

            let ix = fx.floor() as isize;
            let iy = fy.floor() as isize;
            let frac_x = fx - fx.floor();
            let frac_y = fy - fy.floor();

            let mut accum = Pixel::new(0.0, 0.0, 0.0, 0.0);
            let mut total_weight = 0.0f32;
            let radius = lobes as isize;

            for jj in (1 - radius)..=(radius) {
                let wy = lanczos_kernel(frac_y - jj as f32, lobes_f);
                for ii in (1 - radius)..=(radius) {
                    let wx = lanczos_kernel(frac_x - ii as f32, lobes_f);
                    let w = wx * wy;
                    if w.abs() < 1e-10 {
                        continue;
                    }
                    let p = src.get_clamped(ix + ii, iy + jj);
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

/// Lanczos-2 upscale (2 lobes).
pub fn upscale_lanczos2(src: &ImageBuf, scale: ScaleFactor) -> ImageBuf {
    upscale_lanczos_generic(src, scale, 2)
}

/// Lanczos-3 upscale (3 lobes).
pub fn upscale_lanczos3(src: &ImageBuf, scale: ScaleFactor) -> ImageBuf {
    upscale_lanczos_generic(src, scale, 3)
}

/// Edge-directed upscale — detects edges and preserves them during upscaling.
/// Uses gradient analysis to bias interpolation along edges rather than across them.
pub fn upscale_edge_directed(src: &ImageBuf, scale: ScaleFactor) -> ImageBuf {
    let (ow, oh) = scale.output_size(src.width, src.height);
    let mut dst = ImageBuf::new(ow, oh);

    for y in 0..oh {
        for x in 0..ow {
            let fx = x as f32 / scale.0 - 0.5;
            let fy = y as f32 / scale.0 - 0.5;

            let ix = fx.floor() as isize;
            let iy = fy.floor() as isize;
            let frac_x = fx - fx.floor();
            let frac_y = fy - fy.floor();

            // Get the 2x2 neighborhood.
            let c00 = src.get_clamped(ix, iy);
            let c10 = src.get_clamped(ix + 1, iy);
            let c01 = src.get_clamped(ix, iy + 1);
            let c11 = src.get_clamped(ix + 1, iy + 1);

            // Compute luminance gradients.
            let l00 = c00.luminance();
            let l10 = c10.luminance();
            let l01 = c01.luminance();
            let l11 = c11.luminance();

            // Diagonal gradients.
            let d1 = (l00 - l11).abs(); // top-left to bottom-right
            let d2 = (l10 - l01).abs(); // top-right to bottom-left

            let epsilon = 1e-6;
            let w1 = 1.0 / (d1 + epsilon);
            let w2 = 1.0 / (d2 + epsilon);
            let total = w1 + w2;

            // Blend diagonals: if d1 is large, favor d2 direction (and vice versa).
            let diag1 = pixel_lerp(&c00, &c11, frac_x * 0.5 + frac_y * 0.5);
            let diag2 = pixel_lerp(&c10, &c01, 1.0 - (frac_x * 0.5 + frac_y * 0.5));

            let edge_result = pixel_lerp(&diag1, &diag2, w2 / total);

            // Also compute standard bilinear.
            let top = pixel_lerp(&c00, &c10, frac_x);
            let bot = pixel_lerp(&c01, &c11, frac_x);
            let bilinear = pixel_lerp(&top, &bot, frac_y);

            // Blend between edge-directed and bilinear based on edge strength.
            let edge_strength = (d1 - d2).abs() / (d1 + d2 + epsilon);
            let result = pixel_lerp(&bilinear, &edge_result, edge_strength);

            dst.set(x, y, result.clamped());
        }
    }

    dst
}

/// Upscale an image using the specified method and scale factor.
pub fn upscale(src: &ImageBuf, method: UpscaleMethod, scale: ScaleFactor) -> ImageBuf {
    match method {
        UpscaleMethod::NearestNeighbor => upscale_nearest(src, scale),
        UpscaleMethod::Bilinear => upscale_bilinear(src, scale),
        UpscaleMethod::Bicubic => upscale_bicubic(src, scale),
        UpscaleMethod::Lanczos2 => upscale_lanczos2(src, scale),
        UpscaleMethod::Lanczos3 => upscale_lanczos3(src, scale),
        UpscaleMethod::EdgeDirected => upscale_edge_directed(src, scale),
    }
}

/// Upscale using the quality-based method suggestion.
pub fn upscale_auto(src: &ImageBuf, quality: UpscaleQuality, scale: ScaleFactor) -> ImageBuf {
    upscale(src, quality.suggested_method(), scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_image(w: usize, h: usize, c: Pixel) -> ImageBuf {
        ImageBuf {
            pixels: vec![c; w * h],
            width: w,
            height: h,
        }
    }

    fn gradient_image(w: usize, h: usize) -> ImageBuf {
        let mut img = ImageBuf::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let val = x as f32 / (w - 1).max(1) as f32;
                img.set(x, y, Pixel::new(val, val, val, 1.0));
            }
        }
        img
    }

    #[test]
    fn test_scale_factor_min() {
        let sf = ScaleFactor::new(0.5);
        assert!((sf.0 - 1.0).abs() < 1e-6, "scale clamped to 1.0 minimum");
    }

    #[test]
    fn test_scale_factor_output_size() {
        let sf = ScaleFactor::x2();
        let (w, h) = sf.output_size(10, 8);
        assert_eq!(w, 20);
        assert_eq!(h, 16);
    }

    #[test]
    fn test_scale_factor_1_5x() {
        let sf = ScaleFactor::x1_5();
        let (w, h) = sf.output_size(10, 10);
        assert_eq!(w, 15);
        assert_eq!(h, 15);
    }

    #[test]
    fn test_nearest_neighbor_dimensions() {
        let src = solid_image(4, 4, Pixel::new(0.5, 0.5, 0.5, 1.0));
        let dst = upscale_nearest(&src, ScaleFactor::x2());
        assert_eq!(dst.width, 8);
        assert_eq!(dst.height, 8);
    }

    #[test]
    fn test_nearest_neighbor_solid() {
        let src = solid_image(4, 4, Pixel::new(0.3, 0.7, 0.5, 1.0));
        let dst = upscale_nearest(&src, ScaleFactor::x2());
        for p in &dst.pixels {
            assert!((p.r - 0.3).abs() < 1e-6);
            assert!((p.g - 0.7).abs() < 1e-6);
        }
    }

    #[test]
    fn test_bilinear_dimensions() {
        let src = solid_image(4, 4, Pixel::black());
        let dst = upscale_bilinear(&src, ScaleFactor::x4());
        assert_eq!(dst.width, 16);
        assert_eq!(dst.height, 16);
    }

    #[test]
    fn test_bilinear_solid() {
        let src = solid_image(4, 4, Pixel::new(0.5, 0.5, 0.5, 1.0));
        let dst = upscale_bilinear(&src, ScaleFactor::x2());
        for p in &dst.pixels {
            assert!((p.r - 0.5).abs() < 1e-3);
        }
    }

    #[test]
    fn test_bilinear_interpolates() {
        let mut src = ImageBuf::new(2, 1);
        src.set(0, 0, Pixel::new(0.0, 0.0, 0.0, 1.0));
        src.set(1, 0, Pixel::new(1.0, 1.0, 1.0, 1.0));
        let dst = upscale_bilinear(&src, ScaleFactor::x4());
        // Should have smooth gradient.
        let first = dst.get(0, 0).luminance();
        let last = dst.get(dst.width - 1, 0).luminance();
        assert!(first < last, "should be a gradient");
    }

    #[test]
    fn test_catmull_rom_kernel() {
        assert!((catmull_rom(0.0) - 1.0).abs() < 1e-6);
        assert!(catmull_rom(2.0).abs() < 1e-6);
        assert!(catmull_rom(3.0).abs() < 1e-6);
    }

    #[test]
    fn test_bicubic_solid() {
        let src = solid_image(4, 4, Pixel::new(0.5, 0.5, 0.5, 1.0));
        let dst = upscale_bicubic(&src, ScaleFactor::x2());
        for p in &dst.pixels {
            assert!((p.r - 0.5).abs() < 1e-2);
        }
    }

    #[test]
    fn test_bicubic_dimensions() {
        let src = solid_image(6, 4, Pixel::black());
        let dst = upscale_bicubic(&src, ScaleFactor::x2());
        assert_eq!(dst.width, 12);
        assert_eq!(dst.height, 8);
    }

    #[test]
    fn test_lanczos_kernel_center() {
        assert!((lanczos_kernel(0.0, 2.0) - 1.0).abs() < 1e-6);
        assert!((lanczos_kernel(0.0, 3.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_lanczos_kernel_outside() {
        assert!(lanczos_kernel(3.0, 2.0).abs() < 1e-6);
        assert!(lanczos_kernel(4.0, 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_lanczos2_solid() {
        let src = solid_image(4, 4, Pixel::new(0.4, 0.4, 0.4, 1.0));
        let dst = upscale_lanczos2(&src, ScaleFactor::x2());
        for p in &dst.pixels {
            assert!((p.r - 0.4).abs() < 0.02, "lanczos2 solid deviation: {}", p.r);
        }
    }

    #[test]
    fn test_lanczos3_solid() {
        let src = solid_image(4, 4, Pixel::new(0.6, 0.6, 0.6, 1.0));
        let dst = upscale_lanczos3(&src, ScaleFactor::x2());
        for p in &dst.pixels {
            assert!((p.r - 0.6).abs() < 0.02, "lanczos3 solid deviation: {}", p.r);
        }
    }

    #[test]
    fn test_lanczos3_smoother_than_nearest() {
        let src = gradient_image(8, 8);
        let nn = upscale_nearest(&src, ScaleFactor::x2());
        let l3 = upscale_lanczos3(&src, ScaleFactor::x2());

        // Compute sum of absolute differences between adjacent pixels (smoothness).
        fn row_smoothness(img: &ImageBuf) -> f32 {
            let mut sum = 0.0f32;
            for y in 0..img.height {
                for x in 1..img.width {
                    sum += (img.get(x, y).luminance() - img.get(x - 1, y).luminance()).abs();
                }
            }
            sum
        }

        // Nearest neighbor should have blockier transitions.
        let nn_smooth = row_smoothness(&nn);
        let l3_smooth = row_smoothness(&l3);
        // Both are gradients; Lanczos should not have more discontinuities.
        assert!(l3_smooth <= nn_smooth + 0.1, "lanczos should be at least as smooth");
    }

    #[test]
    fn test_edge_directed_dimensions() {
        let src = solid_image(4, 4, Pixel::black());
        let dst = upscale_edge_directed(&src, ScaleFactor::x2());
        assert_eq!(dst.width, 8);
        assert_eq!(dst.height, 8);
    }

    #[test]
    fn test_edge_directed_solid() {
        let src = solid_image(4, 4, Pixel::new(0.5, 0.5, 0.5, 1.0));
        let dst = upscale_edge_directed(&src, ScaleFactor::x2());
        for p in &dst.pixels {
            assert!((p.r - 0.5).abs() < 0.01);
        }
    }

    #[test]
    fn test_upscale_dispatch() {
        let src = solid_image(4, 4, Pixel::new(0.5, 0.5, 0.5, 1.0));
        let methods = [
            UpscaleMethod::NearestNeighbor,
            UpscaleMethod::Bilinear,
            UpscaleMethod::Bicubic,
            UpscaleMethod::Lanczos2,
            UpscaleMethod::Lanczos3,
            UpscaleMethod::EdgeDirected,
        ];
        for m in &methods {
            let dst = upscale(&src, *m, ScaleFactor::x2());
            assert_eq!(dst.width, 8);
            assert_eq!(dst.height, 8);
        }
    }

    #[test]
    fn test_upscale_auto() {
        let src = solid_image(4, 4, Pixel::new(0.5, 0.5, 0.5, 1.0));
        let dst = upscale_auto(&src, UpscaleQuality::Balanced, ScaleFactor::x2());
        assert_eq!(dst.width, 8);
        assert_eq!(dst.height, 8);
    }

    #[test]
    fn test_quality_suggestions() {
        assert_eq!(UpscaleQuality::Fast.suggested_method(), UpscaleMethod::NearestNeighbor);
        assert_eq!(UpscaleQuality::Balanced.suggested_method(), UpscaleMethod::Bilinear);
        assert_eq!(UpscaleQuality::HighQuality.suggested_method(), UpscaleMethod::Lanczos3);
    }

    #[test]
    fn test_output_clamped() {
        // Edge-directed and Lanczos can produce values outside [0,1] — verify clamping.
        let mut src = ImageBuf::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                let v = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
                src.set(x, y, Pixel::new(v, v, v, 1.0));
            }
        }
        let dst = upscale_lanczos3(&src, ScaleFactor::x2());
        for p in &dst.pixels {
            assert!(p.r >= 0.0 && p.r <= 1.0);
            assert!(p.g >= 0.0 && p.g <= 1.0);
            assert!(p.b >= 0.0 && p.b <= 1.0);
        }
    }

    #[test]
    fn test_pixel_operations() {
        let a = Pixel::new(0.5, 0.3, 0.2, 1.0);
        let b = Pixel::new(0.1, 0.2, 0.3, 0.5);
        let sum = a.add(&b);
        assert!((sum.r - 0.6).abs() < 1e-6);

        let scaled = a.scale(2.0);
        assert!((scaled.r - 1.0).abs() < 1e-6);

        let clamped = Pixel::new(1.5, -0.1, 0.5, 2.0).clamped();
        assert!((clamped.r - 1.0).abs() < 1e-6);
        assert!(clamped.g.abs() < 1e-6);
    }

    #[test]
    fn test_scale_4x() {
        let src = solid_image(2, 2, Pixel::new(0.8, 0.8, 0.8, 1.0));
        let dst = upscale_bilinear(&src, ScaleFactor::x4());
        assert_eq!(dst.width, 8);
        assert_eq!(dst.height, 8);
        for p in &dst.pixels {
            assert!((p.r - 0.8).abs() < 1e-2);
        }
    }
}
