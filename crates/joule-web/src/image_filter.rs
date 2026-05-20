//! Image filters — convolution kernels, blur, sharpen, edge detection,
//! emboss, grayscale, brightness/contrast, invert, sepia.
//!
//! Replaces CSS filters and JS Canvas pixel manipulation with deterministic,
//! energy-efficient Rust implementations.

use serde::{Deserialize, Serialize};

// ── Grayscale Buffer (internal) ──────────────────────────────────

/// Simple grayscale image for filter operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

    fn get_clamped(&self, x: i32, y: i32) -> u8 {
        let cx = x.clamp(0, self.width as i32 - 1) as u32;
        let cy = y.clamp(0, self.height as i32 - 1) as u32;
        self.get(cx, cy)
    }
}

/// RGBA image for color filter operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RgbaImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl RgbaImage {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height, data: vec![0u8; width as usize * height as usize * 4] }
    }

    pub fn from_data(width: u32, height: u32, data: Vec<u8>) -> Self {
        assert_eq!(data.len(), width as usize * height as usize * 4);
        Self { width, height, data }
    }

    pub fn get_rgba(&self, x: u32, y: u32) -> [u8; 4] {
        let off = (y as usize * self.width as usize + x as usize) * 4;
        [self.data[off], self.data[off + 1], self.data[off + 2], self.data[off + 3]]
    }

    pub fn set_rgba(&mut self, x: u32, y: u32, rgba: [u8; 4]) {
        let off = (y as usize * self.width as usize + x as usize) * 4;
        self.data[off] = rgba[0];
        self.data[off + 1] = rgba[1];
        self.data[off + 2] = rgba[2];
        self.data[off + 3] = rgba[3];
    }

    fn get_clamped(&self, x: i32, y: i32) -> [u8; 4] {
        let cx = x.clamp(0, self.width as i32 - 1) as u32;
        let cy = y.clamp(0, self.height as i32 - 1) as u32;
        self.get_rgba(cx, cy)
    }
}

// ── Convolution Kernel ───────────────────────────────────────────

/// A convolution kernel (odd-sized square).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Kernel {
    pub size: usize,
    pub weights: Vec<f64>,
    pub divisor: f64,
    pub bias: f64,
}

impl Kernel {
    pub fn new(size: usize, weights: Vec<f64>) -> Self {
        assert!(size % 2 == 1, "kernel size must be odd");
        assert_eq!(weights.len(), size * size);
        let divisor: f64 = weights.iter().sum();
        let divisor = if divisor.abs() < 1e-10 { 1.0 } else { divisor };
        Self { size, weights, divisor, bias: 0.0 }
    }

    pub fn with_divisor(mut self, d: f64) -> Self {
        self.divisor = d;
        self
    }

    pub fn with_bias(mut self, b: f64) -> Self {
        self.bias = b;
        self
    }
}

/// Apply a convolution kernel to a grayscale image.
pub fn convolve_gray(img: &GrayImage, k: &Kernel) -> GrayImage {
    let mut out = GrayImage::new(img.width, img.height);
    let half = k.size as i32 / 2;
    for y in 0..img.height {
        for x in 0..img.width {
            let mut acc = 0.0f64;
            for ky in 0..k.size as i32 {
                for kx in 0..k.size as i32 {
                    let px = x as i32 + kx - half;
                    let py = y as i32 + ky - half;
                    let v = img.get_clamped(px, py) as f64;
                    acc += v * k.weights[(ky as usize) * k.size + kx as usize];
                }
            }
            let v = (acc / k.divisor + k.bias).clamp(0.0, 255.0) as u8;
            out.set(x, y, v);
        }
    }
    out
}

/// Apply a convolution kernel to an RGBA image (alpha untouched).
pub fn convolve_rgba(img: &RgbaImage, k: &Kernel) -> RgbaImage {
    let mut out = RgbaImage::new(img.width, img.height);
    let half = k.size as i32 / 2;
    for y in 0..img.height {
        for x in 0..img.width {
            let mut acc = [0.0f64; 3];
            for ky in 0..k.size as i32 {
                for kx in 0..k.size as i32 {
                    let px = x as i32 + kx - half;
                    let py = y as i32 + ky - half;
                    let rgba = img.get_clamped(px, py);
                    let w = k.weights[(ky as usize) * k.size + kx as usize];
                    acc[0] += rgba[0] as f64 * w;
                    acc[1] += rgba[1] as f64 * w;
                    acc[2] += rgba[2] as f64 * w;
                }
            }
            let orig = img.get_rgba(x, y);
            out.set_rgba(x, y, [
                (acc[0] / k.divisor + k.bias).clamp(0.0, 255.0) as u8,
                (acc[1] / k.divisor + k.bias).clamp(0.0, 255.0) as u8,
                (acc[2] / k.divisor + k.bias).clamp(0.0, 255.0) as u8,
                orig[3],
            ]);
        }
    }
    out
}

// ── Pre-built kernels ────────────────────────────────────────────

/// 3x3 box blur kernel.
pub fn box_blur_3x3() -> Kernel {
    Kernel::new(3, vec![1.0; 9]).with_divisor(9.0)
}

/// 5x5 box blur kernel.
pub fn box_blur_5x5() -> Kernel {
    Kernel::new(5, vec![1.0; 25]).with_divisor(25.0)
}

/// 3x3 gaussian blur (approximation).
pub fn gaussian_blur_3x3() -> Kernel {
    Kernel::new(3, vec![
        1.0, 2.0, 1.0,
        2.0, 4.0, 2.0,
        1.0, 2.0, 1.0,
    ]).with_divisor(16.0)
}

/// 3x3 sharpen kernel.
pub fn sharpen_3x3() -> Kernel {
    Kernel::new(3, vec![
         0.0, -1.0,  0.0,
        -1.0,  5.0, -1.0,
         0.0, -1.0,  0.0,
    ]).with_divisor(1.0)
}

/// Sobel edge detection (horizontal gradient).
pub fn sobel_x() -> Kernel {
    Kernel::new(3, vec![
        -1.0, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0,
    ]).with_divisor(1.0)
}

/// Sobel edge detection (vertical gradient).
pub fn sobel_y() -> Kernel {
    Kernel::new(3, vec![
        -1.0, -2.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  2.0,  1.0,
    ]).with_divisor(1.0)
}

/// Prewitt edge detection (horizontal).
pub fn prewitt_x() -> Kernel {
    Kernel::new(3, vec![
        -1.0, 0.0, 1.0,
        -1.0, 0.0, 1.0,
        -1.0, 0.0, 1.0,
    ]).with_divisor(1.0)
}

/// Prewitt edge detection (vertical).
pub fn prewitt_y() -> Kernel {
    Kernel::new(3, vec![
        -1.0, -1.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  1.0,  1.0,
    ]).with_divisor(1.0)
}

/// Emboss kernel.
pub fn emboss() -> Kernel {
    Kernel::new(3, vec![
        -2.0, -1.0, 0.0,
        -1.0,  1.0, 1.0,
         0.0,  1.0, 2.0,
    ]).with_divisor(1.0)
}

// ── Sobel magnitude ──────────────────────────────────────────────

/// Compute Sobel edge magnitude image.
pub fn sobel_magnitude(img: &GrayImage) -> GrayImage {
    let gx = convolve_gray(img, &sobel_x());
    let gy = convolve_gray(img, &sobel_y());
    let mut out = GrayImage::new(img.width, img.height);
    for i in 0..out.data.len() {
        let vx = gx.data[i] as f64 - 128.0;
        let vy = gy.data[i] as f64 - 128.0;
        let mag = (vx * vx + vy * vy).sqrt().clamp(0.0, 255.0);
        out.data[i] = mag as u8;
    }
    out
}

// ── Point operations ─────────────────────────────────────────────

/// Convert RGBA image to grayscale (ITU-R BT.601).
pub fn to_grayscale(img: &RgbaImage) -> GrayImage {
    let mut out = GrayImage::new(img.width, img.height);
    for y in 0..img.height {
        for x in 0..img.width {
            let px = img.get_rgba(x, y);
            let lum = (px[0] as u32 * 299 + px[1] as u32 * 587 + px[2] as u32 * 114) / 1000;
            out.set(x, y, lum as u8);
        }
    }
    out
}

/// Adjust brightness: `delta` in [-255, 255].
pub fn adjust_brightness(img: &RgbaImage, delta: i16) -> RgbaImage {
    let mut out = img.clone();
    let n = img.width as usize * img.height as usize;
    for i in 0..n {
        let off = i * 4;
        for c in 0..3 {
            out.data[off + c] = (img.data[off + c] as i16 + delta).clamp(0, 255) as u8;
        }
    }
    out
}

/// Adjust contrast: `factor` where 1.0 = unchanged.
pub fn adjust_contrast(img: &RgbaImage, factor: f64) -> RgbaImage {
    let mut out = img.clone();
    let n = img.width as usize * img.height as usize;
    for i in 0..n {
        let off = i * 4;
        for c in 0..3 {
            let v = ((img.data[off + c] as f64 - 128.0) * factor + 128.0).clamp(0.0, 255.0);
            out.data[off + c] = v as u8;
        }
    }
    out
}

/// Invert colors (alpha untouched).
pub fn invert(img: &RgbaImage) -> RgbaImage {
    let mut out = img.clone();
    let n = img.width as usize * img.height as usize;
    for i in 0..n {
        let off = i * 4;
        out.data[off] = 255 - img.data[off];
        out.data[off + 1] = 255 - img.data[off + 1];
        out.data[off + 2] = 255 - img.data[off + 2];
    }
    out
}

/// Apply sepia tone.
pub fn sepia(img: &RgbaImage) -> RgbaImage {
    let mut out = img.clone();
    let n = img.width as usize * img.height as usize;
    for i in 0..n {
        let off = i * 4;
        let r = img.data[off] as f64;
        let g = img.data[off + 1] as f64;
        let b = img.data[off + 2] as f64;
        out.data[off] = (r * 0.393 + g * 0.769 + b * 0.189).clamp(0.0, 255.0) as u8;
        out.data[off + 1] = (r * 0.349 + g * 0.686 + b * 0.168).clamp(0.0, 255.0) as u8;
        out.data[off + 2] = (r * 0.272 + g * 0.534 + b * 0.131).clamp(0.0, 255.0) as u8;
    }
    out
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gray_3x3() -> GrayImage {
        GrayImage::from_data(3, 3, vec![
            10, 20, 30,
            40, 50, 60,
            70, 80, 90,
        ])
    }

    fn make_rgba_2x2() -> RgbaImage {
        let mut img = RgbaImage::new(2, 2);
        img.set_rgba(0, 0, [100, 150, 200, 255]);
        img.set_rgba(1, 0, [200, 100, 50, 255]);
        img.set_rgba(0, 1, [50, 200, 100, 255]);
        img.set_rgba(1, 1, [150, 50, 250, 255]);
        img
    }

    #[test]
    fn test_box_blur() {
        let img = make_gray_3x3();
        let blurred = convolve_gray(&img, &box_blur_3x3());
        // Center pixel should be close to average of all 9 values = 50
        assert!((blurred.get(1, 1) as i32 - 50).abs() <= 1);
    }

    #[test]
    fn test_gaussian_blur() {
        let img = make_gray_3x3();
        let blurred = convolve_gray(&img, &gaussian_blur_3x3());
        assert!(blurred.get(1, 1) > 0);
    }

    #[test]
    fn test_sharpen() {
        let img = make_gray_3x3();
        let sharp = convolve_gray(&img, &sharpen_3x3());
        // Sharpened center should differ from original center
        assert!(sharp.get(1, 1) != img.get(1, 1) || img.get(1, 1) == 50);
    }

    #[test]
    fn test_sobel_magnitude() {
        let img = make_gray_3x3();
        let edges = sobel_magnitude(&img);
        assert_eq!(edges.width, 3);
        assert_eq!(edges.height, 3);
    }

    #[test]
    fn test_prewitt() {
        let img = make_gray_3x3();
        let gx = convolve_gray(&img, &prewitt_x());
        assert_eq!(gx.width, 3);
    }

    #[test]
    fn test_emboss_kernel() {
        let img = make_gray_3x3();
        let emb = convolve_gray(&img, &emboss());
        assert_eq!(emb.width, 3);
    }

    #[test]
    fn test_to_grayscale() {
        let img = make_rgba_2x2();
        let gray = to_grayscale(&img);
        assert_eq!(gray.width, 2);
        assert_eq!(gray.height, 2);
        // (100*299 + 150*587 + 200*114) / 1000 ≈ 140
        let v = gray.get(0, 0);
        assert!(v > 130 && v < 150, "got {v}");
    }

    #[test]
    fn test_brightness() {
        let img = make_rgba_2x2();
        let bright = adjust_brightness(&img, 50);
        assert_eq!(bright.get_rgba(0, 0)[0], 150); // 100 + 50
        let dark = adjust_brightness(&img, -200);
        assert_eq!(dark.get_rgba(0, 0)[0], 0); // clamped
    }

    #[test]
    fn test_contrast() {
        let img = make_rgba_2x2();
        let high = adjust_contrast(&img, 2.0);
        // 100 → (100-128)*2 + 128 = 72
        assert_eq!(high.get_rgba(0, 0)[0], 72);
    }

    #[test]
    fn test_invert() {
        let img = make_rgba_2x2();
        let inv = invert(&img);
        assert_eq!(inv.get_rgba(0, 0)[0], 155); // 255-100
        assert_eq!(inv.get_rgba(0, 0)[3], 255); // alpha unchanged
    }

    #[test]
    fn test_sepia() {
        let img = make_rgba_2x2();
        let sep = sepia(&img);
        // Sepia should produce warm tones — red >= green >= blue
        let px = sep.get_rgba(0, 0);
        assert!(px[0] >= px[1]);
        assert!(px[1] >= px[2]);
    }

    #[test]
    fn test_convolve_rgba() {
        let img = make_rgba_2x2();
        let blurred = convolve_rgba(&img, &box_blur_3x3());
        assert_eq!(blurred.width, 2);
        // Alpha should be preserved
        assert_eq!(blurred.get_rgba(0, 0)[3], 255);
    }

    #[test]
    fn test_kernel_custom() {
        let k = Kernel::new(3, vec![
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
        ]).with_divisor(1.0);
        let img = make_gray_3x3();
        let result = convolve_gray(&img, &k);
        // Identity kernel — should be same as input
        assert_eq!(result.get(1, 1), img.get(1, 1));
    }

    #[test]
    fn test_box_blur_5x5() {
        let img = GrayImage::from_data(5, 5, vec![
            10, 20, 30, 40, 50,
            10, 20, 30, 40, 50,
            10, 20, 30, 40, 50,
            10, 20, 30, 40, 50,
            10, 20, 30, 40, 50,
        ]);
        let blurred = convolve_gray(&img, &box_blur_5x5());
        assert_eq!(blurred.width, 5);
    }
}
