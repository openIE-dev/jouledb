// dithering.rs — Dithering algorithms for banding reduction.
// Ordered (Bayer), Floyd-Steinberg, blue noise, temporal, triangle-distribution noise.

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

    pub fn clamped(&self) -> Self {
        Self {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
            a: self.a.clamp(0.0, 1.0),
        }
    }
}

/// Image buffer for dithering.
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

    pub fn set(&mut self, x: usize, y: usize, p: Pixel) {
        if x < self.width && y < self.height {
            self.pixels[y * self.width + x] = p;
        }
    }
}

/// Bayer matrix size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BayerSize {
    X2,
    X4,
    X8,
}

impl BayerSize {
    pub fn dimension(&self) -> usize {
        match self {
            BayerSize::X2 => 2,
            BayerSize::X4 => 4,
            BayerSize::X8 => 8,
        }
    }
}

/// Generate a Bayer threshold matrix of given size, normalized to [0, 1).
pub fn bayer_matrix(size: BayerSize) -> Vec<Vec<f32>> {
    match size {
        BayerSize::X2 => {
            let n = 4.0;
            vec![
                vec![0.0 / n, 2.0 / n],
                vec![3.0 / n, 1.0 / n],
            ]
        }
        BayerSize::X4 => {
            generate_bayer_recursive(4)
        }
        BayerSize::X8 => {
            generate_bayer_recursive(8)
        }
    }
}

/// Recursively generate a Bayer matrix of size n (must be power of 2).
fn generate_bayer_recursive(n: usize) -> Vec<Vec<f32>> {
    if n == 2 {
        let scale = 1.0 / (n * n) as f32;
        return vec![
            vec![0.0 * scale, 2.0 * scale],
            vec![3.0 * scale, 1.0 * scale],
        ];
    }

    let half = n / 2;
    let sub = generate_bayer_recursive(half);
    let nsq = (n * n) as f32;
    let mut result = vec![vec![0.0f32; n]; n];

    for y in 0..n {
        for x in 0..n {
            let sy = y % half;
            let sx = x % half;
            let base = sub[sy][sx] * (half * half) as f32;

            let quadrant = if y < half {
                if x < half { 0.0 } else { 2.0 }
            } else if x < half {
                3.0
            } else {
                1.0
            };

            result[y][x] = (4.0 * base + quadrant) / nsq;
        }
    }

    result
}

/// Quantize a value to N bits per channel.
pub fn quantize(value: f32, bits: u32) -> f32 {
    if bits == 0 || bits >= 32 {
        return value;
    }
    let levels = ((1u32 << bits) - 1) as f32;
    (value * levels).round() / levels
}

/// Quantize a pixel to N bits per channel.
pub fn quantize_pixel(p: &Pixel, bits: u32) -> Pixel {
    Pixel {
        r: quantize(p.r, bits),
        g: quantize(p.g, bits),
        b: quantize(p.b, bits),
        a: p.a, // Keep alpha untouched.
    }
}

/// Apply ordered (Bayer) dithering to an image.
pub fn ordered_dither(image: &ImageBuf, bits: u32, size: BayerSize) -> ImageBuf {
    let matrix = bayer_matrix(size);
    let dim = size.dimension();
    let levels = ((1u32 << bits) - 1) as f32;
    let spread = 1.0 / levels;

    let mut result = ImageBuf::new(image.width, image.height);

    for y in 0..image.height {
        for x in 0..image.width {
            let p = image.get(x, y);
            let threshold = matrix[y % dim][x % dim] - 0.5;

            let dithered = Pixel::new(
                quantize(p.r + threshold * spread, bits),
                quantize(p.g + threshold * spread, bits),
                quantize(p.b + threshold * spread, bits),
                p.a,
            ).clamped();

            result.set(x, y, dithered);
        }
    }

    result
}

/// Apply Floyd-Steinberg error diffusion dithering.
pub fn floyd_steinberg_dither(image: &ImageBuf, bits: u32) -> ImageBuf {
    let mut buffer: Vec<(f32, f32, f32)> = image.pixels.iter()
        .map(|p| (p.r, p.g, p.b))
        .collect();

    let w = image.width;
    let h = image.height;

    let mut result = ImageBuf::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let (old_r, old_g, old_b) = buffer[idx];

            let new_r = quantize(old_r, bits);
            let new_g = quantize(old_g, bits);
            let new_b = quantize(old_b, bits);

            result.set(x, y, Pixel::new(
                new_r.clamp(0.0, 1.0),
                new_g.clamp(0.0, 1.0),
                new_b.clamp(0.0, 1.0),
                image.pixels[idx].a,
            ));

            let err_r = old_r - new_r;
            let err_g = old_g - new_g;
            let err_b = old_b - new_b;

            // Distribute error to neighbors.
            let distribute = |buf: &mut [(f32, f32, f32)], idx: usize, factor: f32, er: f32, eg: f32, eb: f32| {
                buf[idx].0 += er * factor;
                buf[idx].1 += eg * factor;
                buf[idx].2 += eb * factor;
            };

            if x + 1 < w {
                distribute(&mut buffer, idx + 1, 7.0 / 16.0, err_r, err_g, err_b);
            }
            if y + 1 < h {
                if x > 0 {
                    distribute(&mut buffer, idx + w - 1, 3.0 / 16.0, err_r, err_g, err_b);
                }
                distribute(&mut buffer, idx + w, 5.0 / 16.0, err_r, err_g, err_b);
                if x + 1 < w {
                    distribute(&mut buffer, idx + w + 1, 1.0 / 16.0, err_r, err_g, err_b);
                }
            }
        }
    }

    result
}

/// Simple pseudo-random hash for blue noise generation (deterministic).
fn hash_u32(mut x: u32) -> u32 {
    x = x.wrapping_mul(0x45d9f3b);
    x = (x >> 16) ^ x;
    x = x.wrapping_mul(0x45d9f3b);
    x = (x >> 16) ^ x;
    x
}

/// Generate a blue-noise-like dither texture of given size.
/// Uses a simple interleaved gradient noise approximation.
pub fn generate_blue_noise(width: usize, height: usize, seed: u32) -> Vec<Vec<f32>> {
    let mut noise = vec![vec![0.0f32; width]; height];

    for y in 0..height {
        for x in 0..width {
            // Interleaved gradient noise (Jimenez 2014).
            let fx = x as f32 + 0.5;
            let fy = y as f32 + 0.5;
            let magic = fx * 52.9829189 + fy * 0.06711056;
            let val = (magic * 78.233 + seed as f32).sin().abs().fract();
            noise[y][x] = val;
        }
    }

    noise
}

/// Apply blue noise dithering.
pub fn blue_noise_dither(image: &ImageBuf, bits: u32, noise_size: usize, seed: u32) -> ImageBuf {
    let noise = generate_blue_noise(noise_size, noise_size, seed);
    let levels = ((1u32 << bits) - 1) as f32;
    let spread = 1.0 / levels;

    let mut result = ImageBuf::new(image.width, image.height);

    for y in 0..image.height {
        for x in 0..image.width {
            let p = image.get(x, y);
            let threshold = noise[y % noise_size][x % noise_size] - 0.5;

            let dithered = Pixel::new(
                quantize(p.r + threshold * spread, bits),
                quantize(p.g + threshold * spread, bits),
                quantize(p.b + threshold * spread, bits),
                p.a,
            ).clamped();

            result.set(x, y, dithered);
        }
    }

    result
}

/// Apply temporal dithering — varies the dither pattern per frame.
pub fn temporal_dither(
    image: &ImageBuf,
    bits: u32,
    size: BayerSize,
    frame_index: u32,
) -> ImageBuf {
    let matrix = bayer_matrix(size);
    let dim = size.dimension();
    let levels = ((1u32 << bits) - 1) as f32;
    let spread = 1.0 / levels;

    // Frame-dependent offset to cycle through different thresholds.
    let frame_offset = (frame_index % (dim * dim) as u32) as f32 / (dim * dim) as f32;

    let mut result = ImageBuf::new(image.width, image.height);

    for y in 0..image.height {
        for x in 0..image.width {
            let p = image.get(x, y);
            let base_threshold = matrix[y % dim][x % dim];
            let threshold = (base_threshold + frame_offset).fract() - 0.5;

            let dithered = Pixel::new(
                quantize(p.r + threshold * spread, bits),
                quantize(p.g + threshold * spread, bits),
                quantize(p.b + threshold * spread, bits),
                p.a,
            ).clamped();

            result.set(x, y, dithered);
        }
    }

    result
}

/// Generate triangle-distribution noise from two uniform random values.
/// Returns a value in [-1, 1] with triangular PDF centered at 0.
pub fn triangle_noise(u1: f32, u2: f32) -> f32 {
    // Sum of two uniform [-0.5, 0.5] gives triangular [-1, 1].
    (u1 - 0.5) + (u2 - 0.5)
}

/// Apply triangle-distribution noise dithering (good for HDR to LDR transitions).
pub fn triangle_dither(image: &ImageBuf, bits: u32, seed: u32) -> ImageBuf {
    let levels = ((1u32 << bits) - 1) as f32;
    let spread = 1.0 / levels;

    let mut result = ImageBuf::new(image.width, image.height);

    for y in 0..image.height {
        for x in 0..image.width {
            let p = image.get(x, y);

            // Generate two pseudo-random values for triangular distribution.
            let idx = (y * image.width + x) as u32;
            let h1 = hash_u32(idx.wrapping_add(seed));
            let h2 = hash_u32(idx.wrapping_add(seed).wrapping_add(0x9E3779B9));
            let u1 = h1 as f32 / u32::MAX as f32;
            let u2 = h2 as f32 / u32::MAX as f32;

            let noise = triangle_noise(u1, u2);

            let dithered = Pixel::new(
                quantize(p.r + noise * spread, bits),
                quantize(p.g + noise * spread, bits),
                quantize(p.b + noise * spread, bits),
                p.a,
            ).clamped();

            result.set(x, y, dithered);
        }
    }

    result
}

/// Dithering method selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DitherMethod {
    Ordered,
    FloydSteinberg,
    BlueNoise,
    Temporal,
    Triangle,
}

/// Apply dithering using the specified method.
pub fn apply_dither(
    image: &ImageBuf,
    method: DitherMethod,
    bits: u32,
) -> ImageBuf {
    match method {
        DitherMethod::Ordered => ordered_dither(image, bits, BayerSize::X4),
        DitherMethod::FloydSteinberg => floyd_steinberg_dither(image, bits),
        DitherMethod::BlueNoise => blue_noise_dither(image, bits, 64, 0),
        DitherMethod::Temporal => temporal_dither(image, bits, BayerSize::X4, 0),
        DitherMethod::Triangle => triangle_dither(image, bits, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn solid_image(w: usize, h: usize, val: f32) -> ImageBuf {
        let mut img = ImageBuf::new(w, h);
        for y in 0..h {
            for x in 0..w {
                img.set(x, y, Pixel::new(val, val, val, 1.0));
            }
        }
        img
    }

    #[test]
    fn test_bayer_2x2() {
        let m = bayer_matrix(BayerSize::X2);
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].len(), 2);
        for row in &m {
            for &val in row {
                assert!(val >= 0.0 && val < 1.0, "bayer value out of range: {}", val);
            }
        }
    }

    #[test]
    fn test_bayer_4x4() {
        let m = bayer_matrix(BayerSize::X4);
        assert_eq!(m.len(), 4);
        assert_eq!(m[0].len(), 4);
        for row in &m {
            for &val in row {
                assert!(val >= 0.0 && val < 1.0, "bayer value out of range: {}", val);
            }
        }
    }

    #[test]
    fn test_bayer_8x8() {
        let m = bayer_matrix(BayerSize::X8);
        assert_eq!(m.len(), 8);
        assert_eq!(m[0].len(), 8);
    }

    #[test]
    fn test_bayer_values_unique_4x4() {
        let m = bayer_matrix(BayerSize::X4);
        let mut vals: Vec<f32> = m.iter().flat_map(|r| r.iter().copied()).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for i in 1..vals.len() {
            assert!((vals[i] - vals[i - 1]).abs() > 1e-6, "duplicate bayer values");
        }
    }

    #[test]
    fn test_quantize_8bit() {
        assert!((quantize(0.0, 8) - 0.0).abs() < 1e-6);
        assert!((quantize(1.0, 8) - 1.0).abs() < 1e-6);
        let mid = quantize(0.5, 8);
        assert!((mid - 128.0 / 255.0).abs() < 1.0 / 255.0);
    }

    #[test]
    fn test_quantize_1bit() {
        assert!((quantize(0.0, 1) - 0.0).abs() < 1e-6);
        assert!((quantize(1.0, 1) - 1.0).abs() < 1e-6);
        assert!((quantize(0.3, 1) - 0.0).abs() < 1e-6);
        assert!((quantize(0.7, 1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_quantize_4bit() {
        let levels = 15.0;
        let q = quantize(0.5, 4);
        // Should be one of 0/15, 1/15, ..., 15/15.
        let nearest = (q * levels).round();
        assert!((q - nearest / levels).abs() < 1e-6);
    }

    #[test]
    fn test_ordered_dither_dimensions() {
        let img = gradient_image(16, 16);
        let result = ordered_dither(&img, 4, BayerSize::X4);
        assert_eq!(result.width, 16);
        assert_eq!(result.height, 16);
    }

    #[test]
    fn test_ordered_dither_in_range() {
        let img = gradient_image(32, 32);
        let result = ordered_dither(&img, 4, BayerSize::X8);
        for p in &result.pixels {
            assert!(p.r >= 0.0 && p.r <= 1.0);
            assert!(p.g >= 0.0 && p.g <= 1.0);
        }
    }

    #[test]
    fn test_ordered_dither_modifies_gradient() {
        let img = gradient_image(16, 16);
        let result = ordered_dither(&img, 2, BayerSize::X4);
        let changed = img.pixels.iter().zip(result.pixels.iter())
            .filter(|(a, b)| (a.r - b.r).abs() > 1e-6)
            .count();
        assert!(changed > 0, "dithering should modify gradient pixels");
    }

    #[test]
    fn test_floyd_steinberg_dimensions() {
        let img = gradient_image(16, 16);
        let result = floyd_steinberg_dither(&img, 4);
        assert_eq!(result.width, 16);
        assert_eq!(result.height, 16);
    }

    #[test]
    fn test_floyd_steinberg_in_range() {
        let img = gradient_image(16, 16);
        let result = floyd_steinberg_dither(&img, 4);
        for p in &result.pixels {
            assert!(p.r >= 0.0 && p.r <= 1.0, "FS produced out of range: {}", p.r);
        }
    }

    #[test]
    fn test_floyd_steinberg_preserves_average() {
        let img = solid_image(16, 16, 0.5);
        let result = floyd_steinberg_dither(&img, 4);
        let avg: f32 = result.pixels.iter().map(|p| p.r).sum::<f32>() / result.pixels.len() as f32;
        assert!((avg - 0.5).abs() < 0.05, "FS should preserve average: {}", avg);
    }

    #[test]
    fn test_blue_noise_generation() {
        let noise = generate_blue_noise(16, 16, 42);
        assert_eq!(noise.len(), 16);
        assert_eq!(noise[0].len(), 16);
        for row in &noise {
            for &val in row {
                assert!(val >= 0.0 && val <= 1.0, "noise out of range: {}", val);
            }
        }
    }

    #[test]
    fn test_blue_noise_dither_in_range() {
        let img = gradient_image(16, 16);
        let result = blue_noise_dither(&img, 4, 32, 0);
        for p in &result.pixels {
            assert!(p.r >= 0.0 && p.r <= 1.0);
        }
    }

    #[test]
    fn test_temporal_dither_varies_with_frame() {
        let img = gradient_image(8, 8);
        let frame0 = temporal_dither(&img, 4, BayerSize::X4, 0);
        let frame1 = temporal_dither(&img, 4, BayerSize::X4, 1);

        let different = frame0.pixels.iter().zip(frame1.pixels.iter())
            .filter(|(a, b)| (a.r - b.r).abs() > 1e-6)
            .count();

        assert!(different > 0, "temporal dithering should vary between frames");
    }

    #[test]
    fn test_temporal_dither_in_range() {
        let img = gradient_image(16, 16);
        for frame in 0..8 {
            let result = temporal_dither(&img, 4, BayerSize::X4, frame);
            for p in &result.pixels {
                assert!(p.r >= 0.0 && p.r <= 1.0);
            }
        }
    }

    #[test]
    fn test_triangle_noise_range() {
        for i in 0..100 {
            let u1 = i as f32 / 100.0;
            let u2 = (100 - i) as f32 / 100.0;
            let n = triangle_noise(u1, u2);
            assert!(n >= -1.0 && n <= 1.0, "triangle noise out of range: {}", n);
        }
    }

    #[test]
    fn test_triangle_dither_in_range() {
        let img = gradient_image(16, 16);
        let result = triangle_dither(&img, 4, 42);
        for p in &result.pixels {
            assert!(p.r >= 0.0 && p.r <= 1.0);
        }
    }

    #[test]
    fn test_apply_dither_all_methods() {
        let img = gradient_image(16, 16);
        let methods = [
            DitherMethod::Ordered,
            DitherMethod::FloydSteinberg,
            DitherMethod::BlueNoise,
            DitherMethod::Temporal,
            DitherMethod::Triangle,
        ];
        for method in &methods {
            let result = apply_dither(&img, *method, 4);
            assert_eq!(result.width, 16);
            assert_eq!(result.height, 16);
            for p in &result.pixels {
                assert!(p.r >= 0.0 && p.r <= 1.0, "method {:?} out of range", method);
            }
        }
    }

    #[test]
    fn test_pixel_roundtrip_u8() {
        let p = Pixel::from_u8(128, 64, 200, 255);
        let (r, g, b, a) = p.to_u8();
        assert_eq!(r, 128);
        assert_eq!(g, 64);
        assert_eq!(b, 200);
        assert_eq!(a, 255);
    }

    #[test]
    fn test_hash_deterministic() {
        let a = hash_u32(42);
        let b = hash_u32(42);
        assert_eq!(a, b);
    }

    #[test]
    fn test_hash_varies() {
        let a = hash_u32(0);
        let b = hash_u32(1);
        assert_ne!(a, b);
    }

    #[test]
    fn test_bayer_dimension() {
        assert_eq!(BayerSize::X2.dimension(), 2);
        assert_eq!(BayerSize::X4.dimension(), 4);
        assert_eq!(BayerSize::X8.dimension(), 8);
    }

    #[test]
    fn test_image_buf_from_pixels() {
        let ok = ImageBuf::from_pixels(vec![Pixel::black(); 6], 3, 2);
        assert!(ok.is_some());

        let bad = ImageBuf::from_pixels(vec![Pixel::black(); 5], 3, 2);
        assert!(bad.is_none());
    }
}
