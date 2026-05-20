//! Morphological operations — erosion, dilation, opening, closing,
//! structuring elements, top-hat, bottom-hat, morphological gradient,
//! hit-or-miss transform.
//!
//! Replaces OpenCV.js morphology operations with pure-Rust implementations
//! for binary and grayscale images.

use serde::{Deserialize, Serialize};

// ── Binary Image ─────────────────────────────────────────────────

/// A binary image (1 bit per pixel stored as bool).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BinaryImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<bool>,
}

impl BinaryImage {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height, data: vec![false; width as usize * height as usize] }
    }

    pub fn from_data(width: u32, height: u32, data: Vec<bool>) -> Self {
        assert_eq!(data.len(), width as usize * height as usize);
        Self { width, height, data }
    }

    /// Threshold a grayscale buffer into a binary image.
    pub fn from_grayscale(data: &[u8], width: u32, height: u32, threshold: u8) -> Self {
        assert_eq!(data.len(), width as usize * height as usize);
        Self {
            width,
            height,
            data: data.iter().map(|v| *v >= threshold).collect(),
        }
    }

    pub fn get(&self, x: u32, y: u32) -> bool {
        self.data[y as usize * self.width as usize + x as usize]
    }

    pub fn set(&mut self, x: u32, y: u32, v: bool) {
        self.data[y as usize * self.width as usize + x as usize] = v;
    }

    fn get_safe(&self, x: i32, y: i32) -> bool {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            false
        } else {
            self.get(x as u32, y as u32)
        }
    }

    /// Count foreground pixels.
    pub fn count_foreground(&self) -> u32 {
        self.data.iter().filter(|&&v| v).count() as u32
    }
}

// ── Grayscale Image ──────────────────────────────────────────────

/// A grayscale image for grayscale morphology.
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

    fn get_clamped(&self, x: i32, y: i32) -> u8 {
        let cx = x.clamp(0, self.width as i32 - 1) as u32;
        let cy = y.clamp(0, self.height as i32 - 1) as u32;
        self.get(cx, cy)
    }
}

// ── Structuring Element ──────────────────────────────────────────

/// A structuring element for morphological operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuringElement {
    pub width: u32,
    pub height: u32,
    pub data: Vec<bool>,
    pub anchor_x: u32,
    pub anchor_y: u32,
}

impl StructuringElement {
    /// Create a filled square structuring element.
    pub fn square(size: u32) -> Self {
        Self {
            width: size,
            height: size,
            data: vec![true; size as usize * size as usize],
            anchor_x: size / 2,
            anchor_y: size / 2,
        }
    }

    /// Create a cross (plus sign) structuring element.
    pub fn cross(size: u32) -> Self {
        let mut data = vec![false; size as usize * size as usize];
        let mid = size / 2;
        for i in 0..size {
            data[mid as usize * size as usize + i as usize] = true; // horizontal
            data[i as usize * size as usize + mid as usize] = true; // vertical
        }
        Self { width: size, height: size, data, anchor_x: mid, anchor_y: mid }
    }

    /// Create a circular (disk) structuring element.
    pub fn circle(radius: u32) -> Self {
        let size = radius * 2 + 1;
        let center = radius as f64;
        let mut data = vec![false; size as usize * size as usize];
        for y in 0..size {
            for x in 0..size {
                let dx = x as f64 - center;
                let dy = y as f64 - center;
                if dx * dx + dy * dy <= (radius as f64) * (radius as f64) {
                    data[y as usize * size as usize + x as usize] = true;
                }
            }
        }
        Self { width: size, height: size, data, anchor_x: radius, anchor_y: radius }
    }

    fn get(&self, x: u32, y: u32) -> bool {
        self.data[y as usize * self.width as usize + x as usize]
    }

    /// Iterate over active element offsets relative to anchor.
    pub fn offsets(&self) -> Vec<(i32, i32)> {
        let mut v = Vec::new();
        for y in 0..self.height {
            for x in 0..self.width {
                if self.get(x, y) {
                    v.push((x as i32 - self.anchor_x as i32, y as i32 - self.anchor_y as i32));
                }
            }
        }
        v
    }
}

// ── Binary morphology ────────────────────────────────────────────

/// Binary erosion: pixel is 1 only if all SE neighbors are 1.
pub fn erode(img: &BinaryImage, se: &StructuringElement) -> BinaryImage {
    let offsets = se.offsets();
    let mut out = BinaryImage::new(img.width, img.height);
    for y in 0..img.height {
        for x in 0..img.width {
            let all_set = offsets.iter().all(|&(dx, dy)| {
                img.get_safe(x as i32 + dx, y as i32 + dy)
            });
            out.set(x, y, all_set);
        }
    }
    out
}

/// Binary dilation: pixel is 1 if any SE neighbor is 1.
pub fn dilate(img: &BinaryImage, se: &StructuringElement) -> BinaryImage {
    let offsets = se.offsets();
    let mut out = BinaryImage::new(img.width, img.height);
    for y in 0..img.height {
        for x in 0..img.width {
            let any_set = offsets.iter().any(|&(dx, dy)| {
                img.get_safe(x as i32 + dx, y as i32 + dy)
            });
            out.set(x, y, any_set);
        }
    }
    out
}

/// Opening: erosion followed by dilation.
pub fn opening(img: &BinaryImage, se: &StructuringElement) -> BinaryImage {
    dilate(&erode(img, se), se)
}

/// Closing: dilation followed by erosion.
pub fn closing(img: &BinaryImage, se: &StructuringElement) -> BinaryImage {
    erode(&dilate(img, se), se)
}

/// Morphological gradient: dilation - erosion.
pub fn morphological_gradient(img: &BinaryImage, se: &StructuringElement) -> BinaryImage {
    let dilated = dilate(img, se);
    let eroded = erode(img, se);
    let mut out = BinaryImage::new(img.width, img.height);
    for i in 0..out.data.len() {
        out.data[i] = dilated.data[i] && !eroded.data[i];
    }
    out
}

/// Top-hat: original - opening (isolates bright features).
pub fn top_hat(img: &BinaryImage, se: &StructuringElement) -> BinaryImage {
    let opened = opening(img, se);
    let mut out = BinaryImage::new(img.width, img.height);
    for i in 0..out.data.len() {
        out.data[i] = img.data[i] && !opened.data[i];
    }
    out
}

/// Bottom-hat: closing - original (isolates dark features).
pub fn bottom_hat(img: &BinaryImage, se: &StructuringElement) -> BinaryImage {
    let closed = closing(img, se);
    let mut out = BinaryImage::new(img.width, img.height);
    for i in 0..out.data.len() {
        out.data[i] = closed.data[i] && !img.data[i];
    }
    out
}

/// Hit-or-miss transform: detects pattern defined by `hit` (foreground)
/// and `miss` (background must match).
pub fn hit_or_miss(
    img: &BinaryImage,
    hit: &StructuringElement,
    miss: &StructuringElement,
) -> BinaryImage {
    let eroded_fg = erode(img, hit);
    // Complement
    let mut complement = BinaryImage::new(img.width, img.height);
    for i in 0..complement.data.len() {
        complement.data[i] = !img.data[i];
    }
    let eroded_bg = erode(&complement, miss);
    let mut out = BinaryImage::new(img.width, img.height);
    for i in 0..out.data.len() {
        out.data[i] = eroded_fg.data[i] && eroded_bg.data[i];
    }
    out
}

// ── Grayscale morphology ─────────────────────────────────────────

/// Grayscale erosion (minimum filter over SE).
pub fn erode_gray(img: &GrayImage, se: &StructuringElement) -> GrayImage {
    let offsets = se.offsets();
    let mut out = GrayImage::new(img.width, img.height);
    for y in 0..img.height {
        for x in 0..img.width {
            let mut min_v = 255u8;
            for &(dx, dy) in &offsets {
                let v = img.get_clamped(x as i32 + dx, y as i32 + dy);
                if v < min_v { min_v = v; }
            }
            out.set(x, y, min_v);
        }
    }
    out
}

/// Grayscale dilation (maximum filter over SE).
pub fn dilate_gray(img: &GrayImage, se: &StructuringElement) -> GrayImage {
    let offsets = se.offsets();
    let mut out = GrayImage::new(img.width, img.height);
    for y in 0..img.height {
        for x in 0..img.width {
            let mut max_v = 0u8;
            for &(dx, dy) in &offsets {
                let v = img.get_clamped(x as i32 + dx, y as i32 + dy);
                if v > max_v { max_v = v; }
            }
            out.set(x, y, max_v);
        }
    }
    out
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_image() -> BinaryImage {
        // 5x5 with a 3x3 block in the center
        let mut img = BinaryImage::new(5, 5);
        for y in 1..4 {
            for x in 1..4 {
                img.set(x, y, true);
            }
        }
        img
    }

    #[test]
    fn test_square_se() {
        let se = StructuringElement::square(3);
        assert_eq!(se.offsets().len(), 9);
    }

    #[test]
    fn test_cross_se() {
        let se = StructuringElement::cross(3);
        assert_eq!(se.offsets().len(), 5); // center + 4 arms
    }

    #[test]
    fn test_circle_se() {
        let se = StructuringElement::circle(1);
        assert_eq!(se.width, 3);
        // Should have 5 pixels (center + 4 neighbors)
        assert_eq!(se.offsets().len(), 5);
    }

    #[test]
    fn test_erode_shrinks() {
        let img = make_test_image();
        let se = StructuringElement::square(3);
        let eroded = erode(&img, &se);
        assert!(eroded.count_foreground() < img.count_foreground());
        // Only center pixel should survive
        assert!(eroded.get(2, 2));
    }

    #[test]
    fn test_dilate_grows() {
        let img = make_test_image();
        let se = StructuringElement::square(3);
        let dilated = dilate(&img, &se);
        assert!(dilated.count_foreground() > img.count_foreground());
    }

    #[test]
    fn test_opening_removes_small() {
        // Single pixel — should be removed by opening
        let mut img = BinaryImage::new(5, 5);
        img.set(2, 2, true);
        let se = StructuringElement::square(3);
        let opened = opening(&img, &se);
        assert_eq!(opened.count_foreground(), 0);
    }

    #[test]
    fn test_closing_fills_small() {
        // 3x3 block with hole — closing should fill it
        let mut img = BinaryImage::new(5, 5);
        for y in 1..4 {
            for x in 1..4 {
                img.set(x, y, true);
            }
        }
        img.set(2, 2, false); // hole
        let se = StructuringElement::cross(3);
        let closed = closing(&img, &se);
        assert!(closed.get(2, 2)); // hole filled
    }

    #[test]
    fn test_morphological_gradient() {
        let img = make_test_image();
        let se = StructuringElement::square(3);
        let grad = morphological_gradient(&img, &se);
        // Gradient should be non-zero (boundary pixels)
        assert!(grad.count_foreground() > 0);
        // Interior should NOT be in gradient
        // Center pixel (2,2) is in both dilated and eroded, so not in gradient
    }

    #[test]
    fn test_top_hat() {
        let img = make_test_image();
        let se = StructuringElement::square(5);
        let th = top_hat(&img, &se);
        // 3x3 block is smaller than 5x5 SE, so opening removes it, top-hat = original
        assert_eq!(th.count_foreground(), img.count_foreground());
    }

    #[test]
    fn test_bottom_hat() {
        let mut img = BinaryImage::new(7, 7);
        // Fill all, leave a hole
        for y in 0..7 {
            for x in 0..7 {
                img.set(x, y, true);
            }
        }
        img.set(3, 3, false);
        let se = StructuringElement::square(3);
        let bh = bottom_hat(&img, &se);
        // Bottom-hat should detect the hole
        assert!(bh.count_foreground() > 0);
    }

    #[test]
    fn test_hit_or_miss() {
        let img = make_test_image();
        let hit = StructuringElement::square(1);
        let miss = StructuringElement::square(1);
        let result = hit_or_miss(&img, &hit, &miss);
        // With 1x1 SE, hit-or-miss finds nothing (pixel can't be both fg and bg)
        assert_eq!(result.count_foreground(), 0);
    }

    #[test]
    fn test_grayscale_erode() {
        let img = GrayImage::from_data(3, 3, vec![
            100, 200, 100,
            200, 250, 200,
            100, 200, 100,
        ]);
        let se = StructuringElement::square(3);
        let eroded = erode_gray(&img, &se);
        assert_eq!(eroded.get(1, 1), 100); // min of neighborhood
    }

    #[test]
    fn test_grayscale_dilate() {
        let img = GrayImage::from_data(3, 3, vec![
            100, 50, 100,
            50, 10, 50,
            100, 50, 100,
        ]);
        let se = StructuringElement::square(3);
        let dilated = dilate_gray(&img, &se);
        assert_eq!(dilated.get(1, 1), 100); // max of neighborhood
    }

    #[test]
    fn test_from_grayscale_threshold() {
        let data = vec![10, 50, 100, 150, 200, 250, 30, 80, 128];
        let bin = BinaryImage::from_grayscale(&data, 3, 3, 128);
        assert!(!bin.get(0, 0)); // 10 < 128
        assert!(bin.get(1, 1)); // 250 >= 128
        assert!(bin.get(2, 2)); // 128 >= 128
    }

    #[test]
    fn test_idempotent_opening() {
        let img = make_test_image();
        let se = StructuringElement::square(3);
        let o1 = opening(&img, &se);
        let o2 = opening(&o1, &se);
        assert_eq!(o1, o2); // Opening is idempotent
    }
}
