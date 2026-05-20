//! Dithering algorithms — Floyd-Steinberg, ordered (Bayer), Atkinson,
//! random dithering, palette quantization (median cut), color reduction.
//!
//! Replaces JS/Canvas dithering libraries with energy-efficient Rust
//! implementations for image processing on native and WASM.

use serde::{Deserialize, Serialize};

// ── Grayscale Image ──────────────────────────────────────────────

/// Grayscale image for dithering operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrayImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl GrayImage {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height, data: vec![0u8; width as usize * height as usize] }
    }

    pub fn from_data(width: u32, height: u32, data: Vec<u8>) -> Self {
        assert_eq!(data.len(), width as usize * height as usize);
        Self { width, height, data }
    }

    pub fn get(&self, x: u32, y: u32) -> u8 {
        self.data[y as usize * self.width as usize + x as usize]
    }

    pub fn set(&mut self, x: u32, y: u32, v: u8) {
        self.data[y as usize * self.width as usize + x as usize] = v;
    }
}

// ── Color Image ──────────────────────────────────────────────────

/// RGBA color.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

    fn distance_sq(self, other: Self) -> u32 {
        let dr = self.r as i32 - other.r as i32;
        let dg = self.g as i32 - other.g as i32;
        let db = self.b as i32 - other.b as i32;
        (dr * dr + dg * dg + db * db) as u32
    }
}

/// RGBA image for color dithering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColorImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl ColorImage {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height, data: vec![0u8; width as usize * height as usize * 4] }
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

// ── Floyd-Steinberg dithering ────────────────────────────────────

/// Floyd-Steinberg error diffusion dithering (grayscale → 1-bit).
pub fn floyd_steinberg(img: &GrayImage) -> GrayImage {
    let w = img.width as usize;
    let h = img.height as usize;
    let mut errors = vec![0.0f64; w * h];

    // Init with pixel values
    for i in 0..errors.len() {
        errors[i] = img.data[i] as f64;
    }

    let mut out = GrayImage::new(img.width, img.height);

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let old = errors[idx];
            let new_val: f64 = if old >= 128.0 { 255.0 } else { 0.0 };
            out.data[idx] = new_val as u8;
            let err = old - new_val;

            // Distribute error: right, bottom-left, bottom, bottom-right
            if x + 1 < w {
                errors[idx + 1] += err * 7.0 / 16.0;
            }
            if y + 1 < h {
                if x > 0 {
                    errors[(y + 1) * w + x - 1] += err * 3.0 / 16.0;
                }
                errors[(y + 1) * w + x] += err * 5.0 / 16.0;
                if x + 1 < w {
                    errors[(y + 1) * w + x + 1] += err * 1.0 / 16.0;
                }
            }
        }
    }
    out
}

// ── Atkinson dithering ───────────────────────────────────────────

/// Atkinson error diffusion dithering (grayscale → 1-bit).
/// Distributes only 3/4 of the error for a lighter result.
pub fn atkinson(img: &GrayImage) -> GrayImage {
    let w = img.width as usize;
    let h = img.height as usize;
    let mut errors = vec![0.0f64; w * h];
    for i in 0..errors.len() {
        errors[i] = img.data[i] as f64;
    }

    let mut out = GrayImage::new(img.width, img.height);

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let old = errors[idx];
            let new_val: f64 = if old >= 128.0 { 255.0 } else { 0.0 };
            out.data[idx] = new_val as u8;
            let err = (old - new_val) / 8.0;

            // Atkinson distributes error/8 to 6 neighbors (= 6/8 = 3/4 total)
            let neighbors: [(i32, i32); 6] = [
                (1, 0), (2, 0),
                (-1, 1), (0, 1), (1, 1),
                (0, 2),
            ];
            for (dx, dy) in neighbors {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && (nx as usize) < w && ny >= 0 && (ny as usize) < h {
                    errors[ny as usize * w + nx as usize] += err;
                }
            }
        }
    }
    out
}

// ── Ordered dithering (Bayer matrix) ─────────────────────────────

/// Generate a Bayer dithering matrix of the given order (2, 4, or 8).
pub fn bayer_matrix(order: u32) -> Vec<Vec<f64>> {
    match order {
        2 => vec![
            vec![0.0, 2.0],
            vec![3.0, 1.0],
        ].into_iter().map(|row| row.into_iter().map(|v| (v + 0.5) / 4.0).collect()).collect(),
        4 => {
            let m4 = [
                [0, 8, 2, 10],
                [12, 4, 14, 6],
                [3, 11, 1, 9],
                [15, 7, 13, 5],
            ];
            m4.iter().map(|row| row.iter().map(|v| (*v as f64 + 0.5) / 16.0).collect()).collect()
        }
        _ => {
            // 8x8
            let m8 = [
                [ 0, 32,  8, 40,  2, 34, 10, 42],
                [48, 16, 56, 24, 50, 18, 58, 26],
                [12, 44,  4, 36, 14, 46,  6, 38],
                [60, 28, 52, 20, 62, 30, 54, 22],
                [ 3, 35, 11, 43,  1, 33,  9, 41],
                [51, 19, 59, 27, 49, 17, 57, 25],
                [15, 47,  7, 39, 13, 45,  5, 37],
                [63, 31, 55, 23, 61, 29, 53, 21],
            ];
            m8.iter().map(|row| row.iter().map(|v| (*v as f64 + 0.5) / 64.0).collect()).collect()
        }
    }
}

/// Apply ordered dithering (Bayer matrix) to a grayscale image.
pub fn ordered_dither(img: &GrayImage, order: u32) -> GrayImage {
    let matrix = bayer_matrix(order);
    let n = matrix.len();
    let mut out = GrayImage::new(img.width, img.height);

    for y in 0..img.height {
        for x in 0..img.width {
            let v = img.get(x, y) as f64 / 255.0;
            let threshold = matrix[y as usize % n][x as usize % n];
            out.set(x, y, if v > threshold { 255 } else { 0 });
        }
    }
    out
}

// ── Random dithering ─────────────────────────────────────────────

/// Simple random (white noise) dithering using a deterministic PRNG.
pub fn random_dither(img: &GrayImage, seed: u64) -> GrayImage {
    let mut out = GrayImage::new(img.width, img.height);
    let mut state = seed;
    for y in 0..img.height {
        for x in 0..img.width {
            // Simple xorshift64
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let threshold = (state % 256) as u8;
            out.set(x, y, if img.get(x, y) > threshold { 255 } else { 0 });
        }
    }
    out
}

// ── Palette / color quantization ─────────────────────────────────

/// A color palette.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Palette {
    pub colors: Vec<Rgba>,
}

impl Palette {
    pub fn new(colors: Vec<Rgba>) -> Self {
        Self { colors }
    }

    /// Find the nearest color in the palette.
    pub fn nearest(&self, c: Rgba) -> Rgba {
        self.colors
            .iter()
            .min_by_key(|&&pc| c.distance_sq(pc))
            .copied()
            .unwrap_or(Rgba::rgb(0, 0, 0))
    }
}

/// Median cut color quantization.
pub fn median_cut(pixels: &[Rgba], target_colors: usize) -> Palette {
    if pixels.is_empty() || target_colors == 0 {
        return Palette::new(vec![]);
    }

    let mut buckets: Vec<Vec<Rgba>> = vec![pixels.to_vec()];

    while buckets.len() < target_colors {
        // Find the bucket with the widest range
        let mut best_idx = 0;
        let mut best_range = 0u32;
        let mut best_channel = 0usize; // 0=r, 1=g, 2=b

        for (i, bucket) in buckets.iter().enumerate() {
            if bucket.len() < 2 { continue; }
            for ch in 0..3 {
                let vals: Vec<u8> = bucket.iter().map(|c| match ch {
                    0 => c.r,
                    1 => c.g,
                    _ => c.b,
                }).collect();
                let range = *vals.iter().max().unwrap() as u32 - *vals.iter().min().unwrap() as u32;
                if range > best_range {
                    best_range = range;
                    best_idx = i;
                    best_channel = ch;
                }
            }
        }

        if best_range == 0 { break; }

        let mut bucket = buckets.remove(best_idx);
        bucket.sort_by_key(|c| match best_channel {
            0 => c.r,
            1 => c.g,
            _ => c.b,
        });
        let mid = bucket.len() / 2;
        let right = bucket.split_off(mid);
        buckets.push(bucket);
        buckets.push(right);
    }

    // Average each bucket to get palette color
    let colors: Vec<Rgba> = buckets.iter().filter(|b| !b.is_empty()).map(|bucket| {
        let n = bucket.len() as u32;
        let r = bucket.iter().map(|c| c.r as u32).sum::<u32>() / n;
        let g = bucket.iter().map(|c| c.g as u32).sum::<u32>() / n;
        let b = bucket.iter().map(|c| c.b as u32).sum::<u32>() / n;
        Rgba::rgb(r as u8, g as u8, b as u8)
    }).collect();

    Palette::new(colors)
}

/// Reduce colors in an image to a palette (no dithering).
pub fn reduce_colors(img: &ColorImage, palette: &Palette) -> ColorImage {
    let mut out = ColorImage::new(img.width, img.height);
    for y in 0..img.height {
        for x in 0..img.width {
            let c = img.get(x, y);
            let nearest = palette.nearest(c);
            out.set(x, y, Rgba::new(nearest.r, nearest.g, nearest.b, c.a));
        }
    }
    out
}

/// Floyd-Steinberg dithering with a color palette.
pub fn floyd_steinberg_color(img: &ColorImage, palette: &Palette) -> ColorImage {
    let w = img.width as usize;
    let h = img.height as usize;
    let mut errors_r = vec![0.0f64; w * h];
    let mut errors_g = vec![0.0f64; w * h];
    let mut errors_b = vec![0.0f64; w * h];

    for i in 0..(w * h) {
        let off = i * 4;
        errors_r[i] = img.data[off] as f64;
        errors_g[i] = img.data[off + 1] as f64;
        errors_b[i] = img.data[off + 2] as f64;
    }

    let mut out = ColorImage::new(img.width, img.height);

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let old_r = errors_r[idx].clamp(0.0, 255.0);
            let old_g = errors_g[idx].clamp(0.0, 255.0);
            let old_b = errors_b[idx].clamp(0.0, 255.0);

            let old_color = Rgba::rgb(old_r as u8, old_g as u8, old_b as u8);
            let new_color = palette.nearest(old_color);
            out.set(x as u32, y as u32, Rgba::new(new_color.r, new_color.g, new_color.b, img.data[idx * 4 + 3]));

            let err_r = old_r - new_color.r as f64;
            let err_g = old_g - new_color.g as f64;
            let err_b = old_b - new_color.b as f64;

            let diffuse = |errors: &mut [f64], err: f64| {
                if x + 1 < w {
                    errors[idx + 1] += err * 7.0 / 16.0;
                }
                if y + 1 < h {
                    if x > 0 {
                        errors[(y + 1) * w + x - 1] += err * 3.0 / 16.0;
                    }
                    errors[(y + 1) * w + x] += err * 5.0 / 16.0;
                    if x + 1 < w {
                        errors[(y + 1) * w + x + 1] += err * 1.0 / 16.0;
                    }
                }
            };
            diffuse(&mut errors_r, err_r);
            diffuse(&mut errors_g, err_g);
            diffuse(&mut errors_b, err_b);
        }
    }
    out
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gradient() -> GrayImage {
        let mut img = GrayImage::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                img.set(x, y, (x * 32 + y * 4) as u8);
            }
        }
        img
    }

    #[test]
    fn test_floyd_steinberg_binary() {
        let img = make_gradient();
        let dithered = floyd_steinberg(&img);
        assert_eq!(dithered.width, 8);
        assert!(dithered.data.iter().all(|v| *v == 0 || *v == 255));
    }

    #[test]
    fn test_atkinson_binary() {
        let img = make_gradient();
        let dithered = atkinson(&img);
        assert_eq!(dithered.width, 8);
        assert!(dithered.data.iter().all(|v| *v == 0 || *v == 255));
    }

    #[test]
    fn test_ordered_dither_2x2() {
        let img = make_gradient();
        let dithered = ordered_dither(&img, 2);
        assert!(dithered.data.iter().all(|v| *v == 0 || *v == 255));
    }

    #[test]
    fn test_ordered_dither_4x4() {
        let img = make_gradient();
        let dithered = ordered_dither(&img, 4);
        assert!(dithered.data.iter().all(|v| *v == 0 || *v == 255));
    }

    #[test]
    fn test_ordered_dither_8x8() {
        let img = make_gradient();
        let dithered = ordered_dither(&img, 8);
        assert!(dithered.data.iter().all(|v| *v == 0 || *v == 255));
    }

    #[test]
    fn test_random_dither() {
        let img = make_gradient();
        let dithered = random_dither(&img, 42);
        assert!(dithered.data.iter().all(|v| *v == 0 || *v == 255));
    }

    #[test]
    fn test_random_dither_deterministic() {
        let img = make_gradient();
        let d1 = random_dither(&img, 42);
        let d2 = random_dither(&img, 42);
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_bayer_matrix_sizes() {
        assert_eq!(bayer_matrix(2).len(), 2);
        assert_eq!(bayer_matrix(4).len(), 4);
        assert_eq!(bayer_matrix(8).len(), 8);
    }

    #[test]
    fn test_median_cut() {
        let pixels = vec![
            Rgba::rgb(255, 0, 0),
            Rgba::rgb(250, 5, 5),
            Rgba::rgb(0, 255, 0),
            Rgba::rgb(5, 250, 5),
        ];
        let palette = median_cut(&pixels, 2);
        assert_eq!(palette.colors.len(), 2);
    }

    #[test]
    fn test_palette_nearest() {
        let palette = Palette::new(vec![
            Rgba::rgb(0, 0, 0),
            Rgba::rgb(255, 255, 255),
        ]);
        assert_eq!(palette.nearest(Rgba::rgb(200, 200, 200)), Rgba::rgb(255, 255, 255));
        assert_eq!(palette.nearest(Rgba::rgb(50, 50, 50)), Rgba::rgb(0, 0, 0));
    }

    #[test]
    fn test_reduce_colors() {
        let mut img = ColorImage::new(2, 2);
        img.set(0, 0, Rgba::rgb(200, 10, 10));
        img.set(1, 0, Rgba::rgb(10, 200, 10));
        img.set(0, 1, Rgba::rgb(190, 20, 20));
        img.set(1, 1, Rgba::rgb(20, 190, 20));

        let palette = Palette::new(vec![Rgba::rgb(255, 0, 0), Rgba::rgb(0, 255, 0)]);
        let reduced = reduce_colors(&img, &palette);
        assert_eq!(reduced.get(0, 0).r, 255);
        assert_eq!(reduced.get(1, 0).g, 255);
    }

    #[test]
    fn test_floyd_steinberg_color() {
        let mut img = ColorImage::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                img.set(x, y, Rgba::rgb(
                    (x * 60 + 30) as u8,
                    (y * 60 + 30) as u8,
                    128,
                ));
            }
        }
        let palette = Palette::new(vec![
            Rgba::rgb(0, 0, 0),
            Rgba::rgb(255, 0, 0),
            Rgba::rgb(0, 255, 0),
            Rgba::rgb(255, 255, 255),
        ]);
        let dithered = floyd_steinberg_color(&img, &palette);
        // All pixels should be from the palette
        for y in 0..4 {
            for x in 0..4 {
                let c = dithered.get(x, y);
                let in_palette = palette.colors.iter().any(|pc| pc.r == c.r && pc.g == c.g && pc.b == c.b);
                assert!(in_palette, "pixel ({x},{y}) = ({},{},{}) not in palette", c.r, c.g, c.b);
            }
        }
    }

    #[test]
    fn test_white_image_stays_white() {
        let img = GrayImage::from_data(4, 4, vec![255; 16]);
        let dithered = floyd_steinberg(&img);
        assert!(dithered.data.iter().all(|v| *v == 255));
    }

    #[test]
    fn test_black_image_stays_black() {
        let img = GrayImage::from_data(4, 4, vec![0; 16]);
        let dithered = floyd_steinberg(&img);
        assert!(dithered.data.iter().all(|v| *v == 0));
    }

    #[test]
    fn test_median_cut_empty() {
        let palette = median_cut(&[], 4);
        assert!(palette.colors.is_empty());
    }
}
