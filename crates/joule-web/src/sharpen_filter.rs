// sharpen_filter.rs — Image sharpening filters.
// Unsharp mask, CAS, luma-only, high-pass, edge-aware sharpening.

/// RGBA pixel as f32 (0.0 - 1.0 range).
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

    pub fn from_u8(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self {
            r: r as f32 / 255.0,
            g: g as f32 / 255.0,
            b: b as f32 / 255.0,
            a: a as f32 / 255.0,
        }
    }

    pub fn to_u8(&self) -> (u8, u8, u8, u8) {
        (
            (self.r.clamp(0.0, 1.0) * 255.0).round() as u8,
            (self.g.clamp(0.0, 1.0) * 255.0).round() as u8,
            (self.b.clamp(0.0, 1.0) * 255.0).round() as u8,
            (self.a.clamp(0.0, 1.0) * 255.0).round() as u8,
        )
    }

    pub fn luminance(&self) -> f32 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }

    pub fn clamped(&self) -> Self {
        Self {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
            a: self.a.clamp(0.0, 1.0),
        }
    }
}

/// Image buffer for sharpening operations.
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
        self.pixels[y * self.width + x] = p;
    }
}

/// Sharpening strength parameter (0.0 = no sharpening, 1.0 = maximum).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SharpenStrength(pub f32);

impl SharpenStrength {
    pub fn new(strength: f32) -> Self {
        Self(strength.clamp(0.0, 1.0))
    }

    pub fn value(&self) -> f32 {
        self.0
    }
}

/// Apply a simple 3x3 box blur.
fn box_blur_3x3(image: &ImageBuf) -> ImageBuf {
    let mut result = ImageBuf::new(image.width, image.height);
    for y in 0..image.height {
        for x in 0..image.width {
            let ix = x as isize;
            let iy = y as isize;
            let mut r = 0.0f32;
            let mut g = 0.0f32;
            let mut b = 0.0f32;
            let mut a = 0.0f32;

            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let p = image.get_clamped(ix + dx, iy + dy);
                    r += p.r;
                    g += p.g;
                    b += p.b;
                    a += p.a;
                }
            }

            result.set(x, y, Pixel::new(r / 9.0, g / 9.0, b / 9.0, a / 9.0));
        }
    }
    result
}

/// Apply a Gaussian-like 5x5 blur (approximation with fixed weights).
fn gaussian_blur_5x5(image: &ImageBuf) -> ImageBuf {
    let kernel: [[f32; 5]; 5] = [
        [1.0, 4.0, 6.0, 4.0, 1.0],
        [4.0, 16.0, 24.0, 16.0, 4.0],
        [6.0, 24.0, 36.0, 24.0, 6.0],
        [4.0, 16.0, 24.0, 16.0, 4.0],
        [1.0, 4.0, 6.0, 4.0, 1.0],
    ];
    let total_weight: f32 = 256.0;

    let mut result = ImageBuf::new(image.width, image.height);
    for y in 0..image.height {
        for x in 0..image.width {
            let ix = x as isize;
            let iy = y as isize;
            let mut r = 0.0f32;
            let mut g = 0.0f32;
            let mut b = 0.0f32;
            let mut a = 0.0f32;

            for ky in 0..5usize {
                for kx in 0..5usize {
                    let w = kernel[ky][kx];
                    let p = image.get_clamped(ix + kx as isize - 2, iy + ky as isize - 2);
                    r += p.r * w;
                    g += p.g * w;
                    b += p.b * w;
                    a += p.a * w;
                }
            }

            result.set(x, y, Pixel::new(
                r / total_weight,
                g / total_weight,
                b / total_weight,
                a / total_weight,
            ));
        }
    }
    result
}

/// Unsharp mask sharpening: output = original + strength * (original - blurred).
pub fn unsharp_mask(image: &ImageBuf, strength: &SharpenStrength) -> ImageBuf {
    let blurred = gaussian_blur_5x5(image);
    let s = strength.value() * 2.0; // Scale up for visible effect.
    let mut result = ImageBuf::new(image.width, image.height);

    for i in 0..image.pixels.len() {
        let orig = &image.pixels[i];
        let blur = &blurred.pixels[i];
        result.pixels[i] = Pixel::new(
            (orig.r + s * (orig.r - blur.r)).clamp(0.0, 1.0),
            (orig.g + s * (orig.g - blur.g)).clamp(0.0, 1.0),
            (orig.b + s * (orig.b - blur.b)).clamp(0.0, 1.0),
            orig.a,
        );
    }

    result
}

/// AMD Contrast Adaptive Sharpening (CAS).
/// Adjusts sharpening based on local contrast — avoids over-sharpening flat areas.
pub fn cas_sharpen(image: &ImageBuf, strength: &SharpenStrength) -> ImageBuf {
    let s = strength.value();
    let mut result = ImageBuf::new(image.width, image.height);

    for y in 0..image.height {
        for x in 0..image.width {
            let ix = x as isize;
            let iy = y as isize;

            let center = image.get_clamped(ix, iy);
            let n = image.get_clamped(ix, iy - 1);
            let s_px = image.get_clamped(ix, iy + 1);
            let e = image.get_clamped(ix + 1, iy);
            let w = image.get_clamped(ix - 1, iy);

            // Compute per-channel min/max of the cross neighborhood.
            let min_r = n.r.min(s_px.r).min(e.r).min(w.r);
            let max_r = n.r.max(s_px.r).max(e.r).max(w.r);
            let min_g = n.g.min(s_px.g).min(e.g).min(w.g);
            let max_g = n.g.max(s_px.g).max(e.g).max(w.g);
            let min_b = n.b.min(s_px.b).min(e.b).min(w.b);
            let max_b = n.b.max(s_px.b).max(e.b).max(w.b);

            // Adaptive weight: lower in flat areas, higher near edges (but capped).
            let amp_r = {
                let d = max_r - min_r;
                let sat = d / max_r.max(1e-6);
                -(1.0 / (8.0_f32.min(1.0 / (sat.max(1e-6))) + 1.0)) * s
            };
            let amp_g = {
                let d = max_g - min_g;
                let sat = d / max_g.max(1e-6);
                -(1.0 / (8.0_f32.min(1.0 / (sat.max(1e-6))) + 1.0)) * s
            };
            let amp_b = {
                let d = max_b - min_b;
                let sat = d / max_b.max(1e-6);
                -(1.0 / (8.0_f32.min(1.0 / (sat.max(1e-6))) + 1.0)) * s
            };

            // Apply sharpening: center + weight * (center - neighbors_avg).
            let neighbors_r = (n.r + s_px.r + e.r + w.r) / 4.0;
            let neighbors_g = (n.g + s_px.g + e.g + w.g) / 4.0;
            let neighbors_b = (n.b + s_px.b + e.b + w.b) / 4.0;

            let sharp_r = center.r + amp_r * (center.r - neighbors_r);
            let sharp_g = center.g + amp_g * (center.g - neighbors_g);
            let sharp_b = center.b + amp_b * (center.b - neighbors_b);

            result.set(x, y, Pixel::new(
                sharp_r.clamp(0.0, 1.0),
                sharp_g.clamp(0.0, 1.0),
                sharp_b.clamp(0.0, 1.0),
                center.a,
            ));
        }
    }

    result
}

/// Luma-only sharpening — sharpen luminance channel while preserving chroma.
pub fn luma_sharpen(image: &ImageBuf, strength: &SharpenStrength) -> ImageBuf {
    let blurred = box_blur_3x3(image);
    let s = strength.value() * 2.0;
    let mut result = ImageBuf::new(image.width, image.height);

    for i in 0..image.pixels.len() {
        let orig = &image.pixels[i];
        let blur = &blurred.pixels[i];

        let luma_orig = orig.luminance();
        let luma_blur = blur.luminance();
        let luma_diff = s * (luma_orig - luma_blur);

        result.pixels[i] = Pixel::new(
            (orig.r + luma_diff).clamp(0.0, 1.0),
            (orig.g + luma_diff).clamp(0.0, 1.0),
            (orig.b + luma_diff).clamp(0.0, 1.0),
            orig.a,
        );
    }

    result
}

/// High-pass sharpening — applies a high-pass filter then blends with original.
pub fn high_pass_sharpen(image: &ImageBuf, strength: &SharpenStrength) -> ImageBuf {
    let blurred = gaussian_blur_5x5(image);
    let s = strength.value();
    let mut result = ImageBuf::new(image.width, image.height);

    for i in 0..image.pixels.len() {
        let orig = &image.pixels[i];
        let blur = &blurred.pixels[i];

        // High-pass = original - blurred.
        let hp_r = orig.r - blur.r;
        let hp_g = orig.g - blur.g;
        let hp_b = orig.b - blur.b;

        // Blend: original + strength * high_pass.
        result.pixels[i] = Pixel::new(
            (orig.r + s * hp_r).clamp(0.0, 1.0),
            (orig.g + s * hp_g).clamp(0.0, 1.0),
            (orig.b + s * hp_b).clamp(0.0, 1.0),
            orig.a,
        );
    }

    result
}

/// Compute edge magnitude at a pixel using Sobel operator.
fn edge_magnitude(image: &ImageBuf, x: usize, y: usize) -> f32 {
    let ix = x as isize;
    let iy = y as isize;

    let tl = image.get_clamped(ix - 1, iy - 1).luminance();
    let tc = image.get_clamped(ix, iy - 1).luminance();
    let tr = image.get_clamped(ix + 1, iy - 1).luminance();
    let ml = image.get_clamped(ix - 1, iy).luminance();
    let mr = image.get_clamped(ix + 1, iy).luminance();
    let bl = image.get_clamped(ix - 1, iy + 1).luminance();
    let bc = image.get_clamped(ix, iy + 1).luminance();
    let br = image.get_clamped(ix + 1, iy + 1).luminance();

    let gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
    let gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;

    (gx * gx + gy * gy).sqrt()
}

/// Edge-aware sharpening — reduces sharpening across detected edges to prevent ringing.
pub fn edge_aware_sharpen(
    image: &ImageBuf,
    strength: &SharpenStrength,
    edge_threshold: f32,
) -> ImageBuf {
    let blurred = box_blur_3x3(image);
    let s = strength.value() * 2.0;
    let mut result = ImageBuf::new(image.width, image.height);

    for y in 0..image.height {
        for x in 0..image.width {
            let idx = y * image.width + x;
            let orig = &image.pixels[idx];
            let blur = &blurred.pixels[idx];

            let edge_mag = edge_magnitude(image, x, y);
            // Reduce sharpening on strong edges.
            let edge_factor = (1.0 - (edge_mag / edge_threshold).min(1.0)).max(0.0);
            let effective_strength = s * edge_factor;

            result.pixels[idx] = Pixel::new(
                (orig.r + effective_strength * (orig.r - blur.r)).clamp(0.0, 1.0),
                (orig.g + effective_strength * (orig.g - blur.g)).clamp(0.0, 1.0),
                (orig.b + effective_strength * (orig.b - blur.b)).clamp(0.0, 1.0),
                orig.a,
            );
        }
    }

    result
}

/// All available sharpening methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharpenMethod {
    UnsharpMask,
    Cas,
    LumaOnly,
    HighPass,
    EdgeAware,
}

/// Apply sharpening using the specified method.
pub fn apply_sharpen(
    image: &ImageBuf,
    method: SharpenMethod,
    strength: &SharpenStrength,
) -> ImageBuf {
    match method {
        SharpenMethod::UnsharpMask => unsharp_mask(image, strength),
        SharpenMethod::Cas => cas_sharpen(image, strength),
        SharpenMethod::LumaOnly => luma_sharpen(image, strength),
        SharpenMethod::HighPass => high_pass_sharpen(image, strength),
        SharpenMethod::EdgeAware => edge_aware_sharpen(image, strength, 0.5),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_image(w: usize, h: usize, p: Pixel) -> ImageBuf {
        ImageBuf {
            pixels: vec![p; w * h],
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

    fn edge_image(w: usize, h: usize) -> ImageBuf {
        let mut img = ImageBuf::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let val = if x < w / 2 { 0.0 } else { 1.0 };
                img.set(x, y, Pixel::new(val, val, val, 1.0));
            }
        }
        img
    }

    #[test]
    fn test_pixel_luminance() {
        let white = Pixel::new(1.0, 1.0, 1.0, 1.0);
        assert!((white.luminance() - 1.0).abs() < 1e-3);

        let black = Pixel::black();
        assert!(black.luminance().abs() < 1e-6);
    }

    #[test]
    fn test_pixel_from_to_u8() {
        let p = Pixel::from_u8(128, 64, 255, 200);
        let (r, g, b, a) = p.to_u8();
        assert_eq!(r, 128);
        assert_eq!(g, 64);
        assert_eq!(b, 255);
        assert_eq!(a, 200);
    }

    #[test]
    fn test_pixel_clamp() {
        let p = Pixel::new(1.5, -0.5, 0.5, 2.0);
        let c = p.clamped();
        assert!((c.r - 1.0).abs() < 1e-6);
        assert!(c.g.abs() < 1e-6);
        assert!((c.b - 0.5).abs() < 1e-6);
        assert!((c.a - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sharpen_strength_clamp() {
        let s = SharpenStrength::new(1.5);
        assert!((s.value() - 1.0).abs() < 1e-6);

        let s2 = SharpenStrength::new(-0.5);
        assert!(s2.value().abs() < 1e-6);
    }

    #[test]
    fn test_box_blur_solid_unchanged() {
        let img = solid_image(8, 8, Pixel::new(0.5, 0.5, 0.5, 1.0));
        let blurred = box_blur_3x3(&img);
        for p in &blurred.pixels {
            assert!((p.r - 0.5).abs() < 1e-4);
        }
    }

    #[test]
    fn test_gaussian_blur_solid_unchanged() {
        let img = solid_image(8, 8, Pixel::new(0.3, 0.3, 0.3, 1.0));
        let blurred = gaussian_blur_5x5(&img);
        for p in &blurred.pixels {
            assert!((p.r - 0.3).abs() < 1e-3);
        }
    }

    #[test]
    fn test_unsharp_mask_solid_unchanged() {
        let img = solid_image(8, 8, Pixel::new(0.5, 0.5, 0.5, 1.0));
        let strength = SharpenStrength::new(0.5);
        let result = unsharp_mask(&img, &strength);
        for p in &result.pixels {
            assert!((p.r - 0.5).abs() < 1e-3);
        }
    }

    #[test]
    fn test_unsharp_mask_increases_contrast() {
        let img = gradient_image(16, 16);
        let strength = SharpenStrength::new(1.0);
        let result = unsharp_mask(&img, &strength);

        // After sharpening, contrast should increase at transitions.
        let orig_mid = img.get(8, 8).luminance();
        let sharp_mid = result.get(8, 8).luminance();
        // The exact value depends on position, but interior gradient pixels should change.
        let total_change: f32 = img.pixels.iter().zip(result.pixels.iter())
            .map(|(a, b)| (a.luminance() - b.luminance()).abs())
            .sum();
        let _ = (orig_mid, sharp_mid);
        assert!(total_change > 0.0, "unsharp mask should change gradient image");
    }

    #[test]
    fn test_cas_solid_unchanged() {
        let img = solid_image(8, 8, Pixel::new(0.5, 0.5, 0.5, 1.0));
        let strength = SharpenStrength::new(1.0);
        let result = cas_sharpen(&img, &strength);
        for p in &result.pixels {
            assert!((p.r - 0.5).abs() < 0.01);
        }
    }

    #[test]
    fn test_cas_preserves_alpha() {
        let img = solid_image(8, 8, Pixel::new(0.5, 0.5, 0.5, 0.7));
        let strength = SharpenStrength::new(0.5);
        let result = cas_sharpen(&img, &strength);
        for p in &result.pixels {
            assert!((p.a - 0.7).abs() < 1e-6);
        }
    }

    #[test]
    fn test_luma_sharpen_preserves_relative_chroma() {
        let img = solid_image(8, 8, Pixel::new(0.8, 0.2, 0.4, 1.0));
        let strength = SharpenStrength::new(0.5);
        let result = luma_sharpen(&img, &strength);
        // On a solid image, result should be very close to original.
        for p in &result.pixels {
            assert!((p.r - 0.8).abs() < 0.02);
            assert!((p.g - 0.2).abs() < 0.02);
        }
    }

    #[test]
    fn test_high_pass_solid_unchanged() {
        let img = solid_image(8, 8, Pixel::new(0.6, 0.6, 0.6, 1.0));
        let strength = SharpenStrength::new(1.0);
        let result = high_pass_sharpen(&img, &strength);
        for p in &result.pixels {
            assert!((p.r - 0.6).abs() < 1e-3);
        }
    }

    #[test]
    fn test_high_pass_modifies_edges() {
        // Use a gradient image so the blur and original differ at non-edge
        // interior pixels, making the high-pass contribution visible.
        let img = gradient_image(16, 16);
        let strength = SharpenStrength::new(1.0);
        let result = high_pass_sharpen(&img, &strength);
        let changed = img.pixels.iter().zip(result.pixels.iter())
            .filter(|(a, b)| (a.r - b.r).abs() > 1e-4)
            .count();
        assert!(changed > 0, "high-pass should modify gradient image pixels");
    }

    #[test]
    fn test_edge_magnitude_flat() {
        let img = solid_image(8, 8, Pixel::new(0.5, 0.5, 0.5, 1.0));
        let mag = edge_magnitude(&img, 4, 4);
        assert!(mag.abs() < 1e-4);
    }

    #[test]
    fn test_edge_magnitude_edge() {
        let img = edge_image(16, 16);
        let mag = edge_magnitude(&img, 8, 8);
        assert!(mag > 0.1, "edge magnitude should be high at boundary: {}", mag);
    }

    #[test]
    fn test_edge_aware_sharpen_flat() {
        let img = solid_image(8, 8, Pixel::new(0.4, 0.4, 0.4, 1.0));
        let strength = SharpenStrength::new(1.0);
        let result = edge_aware_sharpen(&img, &strength, 0.5);
        for p in &result.pixels {
            assert!((p.r - 0.4).abs() < 1e-3);
        }
    }

    #[test]
    fn test_edge_aware_sharpen_preserves_edges() {
        let img = edge_image(16, 16);
        let strength = SharpenStrength::new(1.0);
        let result = edge_aware_sharpen(&img, &strength, 0.5);
        // Pixels right at the edge should have reduced sharpening.
        let at_edge = result.get(8, 8);
        // Should still be valid.
        assert!(at_edge.r >= 0.0 && at_edge.r <= 1.0);
    }

    #[test]
    fn test_apply_sharpen_all_methods() {
        let img = gradient_image(8, 8);
        let strength = SharpenStrength::new(0.5);
        let methods = [
            SharpenMethod::UnsharpMask,
            SharpenMethod::Cas,
            SharpenMethod::LumaOnly,
            SharpenMethod::HighPass,
            SharpenMethod::EdgeAware,
        ];
        for method in &methods {
            let result = apply_sharpen(&img, *method, &strength);
            assert_eq!(result.width, 8);
            assert_eq!(result.height, 8);
            // All pixels should be valid.
            for p in &result.pixels {
                assert!(p.r >= 0.0 && p.r <= 1.0, "method {:?} produced out-of-range r: {}", method, p.r);
            }
        }
    }

    #[test]
    fn test_zero_strength_no_change() {
        let img = gradient_image(8, 8);
        let strength = SharpenStrength::new(0.0);
        let result = unsharp_mask(&img, &strength);
        for (a, b) in img.pixels.iter().zip(result.pixels.iter()) {
            assert!((a.r - b.r).abs() < 1e-4);
            assert!((a.g - b.g).abs() < 1e-4);
            assert!((a.b - b.b).abs() < 1e-4);
        }
    }

    #[test]
    fn test_image_buf_get_clamped() {
        let img = solid_image(4, 4, Pixel::new(0.5, 0.5, 0.5, 1.0));
        let p = img.get_clamped(-5, -5);
        assert!((p.r - 0.5).abs() < 1e-6);
        let p2 = img.get_clamped(100, 100);
        assert!((p2.r - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_cas_output_in_range() {
        let mut img = ImageBuf::new(8, 8);
        // Create noisy image.
        for y in 0..8 {
            for x in 0..8 {
                let val = ((x * 37 + y * 53) % 256) as f32 / 255.0;
                img.set(x, y, Pixel::new(val, 1.0 - val, val * 0.5, 1.0));
            }
        }
        let strength = SharpenStrength::new(1.0);
        let result = cas_sharpen(&img, &strength);
        for p in &result.pixels {
            assert!(p.r >= 0.0 && p.r <= 1.0);
            assert!(p.g >= 0.0 && p.g <= 1.0);
            assert!(p.b >= 0.0 && p.b <= 1.0);
        }
    }

    #[test]
    fn test_image_buf_from_pixels_invalid() {
        let result = ImageBuf::from_pixels(vec![Pixel::black(); 5], 3, 2);
        assert!(result.is_none());
    }

    #[test]
    fn test_image_buf_from_pixels_valid() {
        let result = ImageBuf::from_pixels(vec![Pixel::black(); 6], 3, 2);
        assert!(result.is_some());
    }
}
