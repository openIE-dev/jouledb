//! Image processing.
//!
//! In-memory RGBA pixel buffer manipulation. No file I/O, no image
//! codec — just pure pixel operations for compositing, transforms,
//! and colour adjustments.

// ── PixelBuffer ────────────────────────────────────────────────────

/// An in-memory RGBA pixel buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PixelBuffer {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Raw RGBA bytes (length = width * height * 4).
    pub data: Vec<u8>,
}

impl PixelBuffer {
    /// Create a new buffer filled with transparent black (0,0,0,0).
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0u8; (width as usize) * (height as usize) * 4],
        }
    }

    /// Create from existing RGBA data. Panics if data length mismatches.
    pub fn from_rgba(width: u32, height: u32, data: Vec<u8>) -> Self {
        assert_eq!(
            data.len(),
            (width as usize) * (height as usize) * 4,
            "data length mismatch"
        );
        Self {
            width,
            height,
            data,
        }
    }

    /// Get the RGBA value of a pixel.
    pub fn get_pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let idx = self.pixel_index(x, y);
        [
            self.data[idx],
            self.data[idx + 1],
            self.data[idx + 2],
            self.data[idx + 3],
        ]
    }

    /// Set the RGBA value of a pixel.
    pub fn set_pixel(&mut self, x: u32, y: u32, rgba: [u8; 4]) {
        let idx = self.pixel_index(x, y);
        self.data[idx] = rgba[0];
        self.data[idx + 1] = rgba[1];
        self.data[idx + 2] = rgba[2];
        self.data[idx + 3] = rgba[3];
    }

    /// Fill the entire buffer with a single colour.
    pub fn fill(&mut self, rgba: [u8; 4]) {
        for chunk in self.data.chunks_exact_mut(4) {
            chunk.copy_from_slice(&rgba);
        }
    }

    /// Clear to transparent black.
    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    /// Crop a rectangular region.
    pub fn crop(&self, x: u32, y: u32, w: u32, h: u32) -> Self {
        let mut out = PixelBuffer::new(w, h);
        for row in 0..h {
            for col in 0..w {
                let sx = x + col;
                let sy = y + row;
                if sx < self.width && sy < self.height {
                    out.set_pixel(col, row, self.get_pixel(sx, sy));
                }
            }
        }
        out
    }

    /// Resize using nearest-neighbour interpolation.
    pub fn resize_nearest(&self, new_w: u32, new_h: u32) -> Self {
        let mut out = PixelBuffer::new(new_w, new_h);
        for y in 0..new_h {
            for x in 0..new_w {
                let sx = (x as u64 * self.width as u64 / new_w as u64) as u32;
                let sy = (y as u64 * self.height as u64 / new_h as u64) as u32;
                out.set_pixel(x, y, self.get_pixel(sx.min(self.width - 1), sy.min(self.height - 1)));
            }
        }
        out
    }

    /// Flip horizontally.
    pub fn flip_horizontal(&self) -> Self {
        let mut out = PixelBuffer::new(self.width, self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                out.set_pixel(self.width - 1 - x, y, self.get_pixel(x, y));
            }
        }
        out
    }

    /// Flip vertically.
    pub fn flip_vertical(&self) -> Self {
        let mut out = PixelBuffer::new(self.width, self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                out.set_pixel(x, self.height - 1 - y, self.get_pixel(x, y));
            }
        }
        out
    }

    /// Rotate 90 degrees clockwise.
    pub fn rotate_90(&self) -> Self {
        let mut out = PixelBuffer::new(self.height, self.width);
        for y in 0..self.height {
            for x in 0..self.width {
                out.set_pixel(self.height - 1 - y, x, self.get_pixel(x, y));
            }
        }
        out
    }

    /// Convert to grayscale using luminance formula (Rec. 709).
    pub fn grayscale(&self) -> Self {
        let mut out = self.clone();
        for chunk in out.data.chunks_exact_mut(4) {
            let r = chunk[0] as f64;
            let g = chunk[1] as f64;
            let b = chunk[2] as f64;
            let lum = (0.2126 * r + 0.7152 * g + 0.0722 * b).round() as u8;
            chunk[0] = lum;
            chunk[1] = lum;
            chunk[2] = lum;
            // alpha unchanged
        }
        out
    }

    /// Adjust brightness by multiplying RGB channels.
    pub fn brightness(&self, factor: f64) -> Self {
        let mut out = self.clone();
        for chunk in out.data.chunks_exact_mut(4) {
            chunk[0] = ((chunk[0] as f64 * factor).round() as u32).min(255) as u8;
            chunk[1] = ((chunk[1] as f64 * factor).round() as u32).min(255) as u8;
            chunk[2] = ((chunk[2] as f64 * factor).round() as u32).min(255) as u8;
        }
        out
    }

    /// Adjust contrast. `factor` = 1.0 is unchanged, >1 increases, <1 decreases.
    pub fn contrast(&self, factor: f64) -> Self {
        let mut out = self.clone();
        for chunk in out.data.chunks_exact_mut(4) {
            for c in &mut chunk[..3] {
                let val = (((*c as f64 - 128.0) * factor + 128.0).round() as i32)
                    .clamp(0, 255);
                *c = val as u8;
            }
        }
        out
    }

    /// Invert colours (255 - channel for R, G, B).
    pub fn invert(&self) -> Self {
        let mut out = self.clone();
        for chunk in out.data.chunks_exact_mut(4) {
            chunk[0] = 255 - chunk[0];
            chunk[1] = 255 - chunk[1];
            chunk[2] = 255 - chunk[2];
            // alpha unchanged
        }
        out
    }

    /// Composite `other` onto `self` at position (x, y) with opacity (0.0-1.0).
    pub fn blend(&self, other: &PixelBuffer, x: i32, y: i32, opacity: f64) -> Self {
        let mut out = self.clone();
        let op = opacity.clamp(0.0, 1.0);

        for oy in 0..other.height as i32 {
            for ox in 0..other.width as i32 {
                let dx = x + ox;
                let dy = y + oy;
                if dx >= 0 && dx < self.width as i32 && dy >= 0 && dy < self.height as i32 {
                    let src = other.get_pixel(ox as u32, oy as u32);
                    let dst = out.get_pixel(dx as u32, dy as u32);
                    let sa = (src[3] as f64 / 255.0) * op;
                    let da = dst[3] as f64 / 255.0;
                    let oa = sa + da * (1.0 - sa);

                    let blend_ch = |s: u8, d: u8| -> u8 {
                        if oa == 0.0 {
                            return 0;
                        }
                        ((s as f64 * sa + d as f64 * da * (1.0 - sa)) / oa).round() as u8
                    };

                    out.set_pixel(dx as u32, dy as u32, [
                        blend_ch(src[0], dst[0]),
                        blend_ch(src[1], dst[1]),
                        blend_ch(src[2], dst[2]),
                        (oa * 255.0).round() as u8,
                    ]);
                }
            }
        }
        out
    }

    /// Compute a luminance histogram (256 bins).
    pub fn histogram(&self) -> [u32; 256] {
        let mut hist = [0u32; 256];
        for chunk in self.data.chunks_exact(4) {
            let lum = (0.2126 * chunk[0] as f64
                + 0.7152 * chunk[1] as f64
                + 0.0722 * chunk[2] as f64)
                .round() as u8;
            hist[lum as usize] += 1;
        }
        hist
    }

    // ── Internal ───────────────────────────────────────────────────

    fn pixel_index(&self, x: u32, y: u32) -> usize {
        debug_assert!(x < self.width && y < self.height, "pixel out of bounds");
        ((y as usize) * (self.width as usize) + (x as usize)) * 4
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_transparent() {
        let buf = PixelBuffer::new(4, 4);
        assert_eq!(buf.get_pixel(0, 0), [0, 0, 0, 0]);
        assert_eq!(buf.get_pixel(3, 3), [0, 0, 0, 0]);
    }

    #[test]
    fn set_get_pixel() {
        let mut buf = PixelBuffer::new(10, 10);
        buf.set_pixel(5, 5, [255, 128, 64, 200]);
        assert_eq!(buf.get_pixel(5, 5), [255, 128, 64, 200]);
        assert_eq!(buf.get_pixel(0, 0), [0, 0, 0, 0]);
    }

    #[test]
    fn fill_and_clear() {
        let mut buf = PixelBuffer::new(3, 3);
        buf.fill([255, 0, 0, 255]);
        assert_eq!(buf.get_pixel(1, 1), [255, 0, 0, 255]);
        buf.clear();
        assert_eq!(buf.get_pixel(1, 1), [0, 0, 0, 0]);
    }

    #[test]
    fn crop_dimensions() {
        let mut buf = PixelBuffer::new(10, 10);
        buf.fill([100, 100, 100, 255]);
        let cropped = buf.crop(2, 2, 5, 5);
        assert_eq!(cropped.width, 5);
        assert_eq!(cropped.height, 5);
        assert_eq!(cropped.get_pixel(0, 0), [100, 100, 100, 255]);
    }

    #[test]
    fn resize_dimensions() {
        let buf = PixelBuffer::new(10, 10);
        let resized = buf.resize_nearest(20, 20);
        assert_eq!(resized.width, 20);
        assert_eq!(resized.height, 20);
    }

    #[test]
    fn flip_horizontal_roundtrip() {
        let mut buf = PixelBuffer::new(4, 4);
        buf.set_pixel(0, 0, [255, 0, 0, 255]);
        let flipped = buf.flip_horizontal().flip_horizontal();
        assert_eq!(flipped.get_pixel(0, 0), [255, 0, 0, 255]);
    }

    #[test]
    fn flip_vertical_roundtrip() {
        let mut buf = PixelBuffer::new(4, 4);
        buf.set_pixel(0, 0, [0, 255, 0, 255]);
        let flipped = buf.flip_vertical().flip_vertical();
        assert_eq!(flipped.get_pixel(0, 0), [0, 255, 0, 255]);
    }

    #[test]
    fn grayscale_preserves_alpha() {
        let mut buf = PixelBuffer::new(2, 2);
        buf.set_pixel(0, 0, [255, 0, 0, 128]);
        let gray = buf.grayscale();
        let px = gray.get_pixel(0, 0);
        assert_eq!(px[3], 128); // alpha preserved
        assert_eq!(px[0], px[1]); // R == G == B
    }

    #[test]
    fn brightness_scales() {
        let mut buf = PixelBuffer::new(1, 1);
        buf.set_pixel(0, 0, [100, 100, 100, 255]);
        let bright = buf.brightness(2.0);
        assert_eq!(bright.get_pixel(0, 0), [200, 200, 200, 255]);
    }

    #[test]
    fn invert_roundtrip() {
        let mut buf = PixelBuffer::new(1, 1);
        buf.set_pixel(0, 0, [100, 150, 200, 255]);
        let inv = buf.invert().invert();
        assert_eq!(inv.get_pixel(0, 0), [100, 150, 200, 255]);
    }

    #[test]
    fn blend_composites() {
        let mut base = PixelBuffer::new(4, 4);
        base.fill([0, 0, 0, 255]);
        let mut overlay = PixelBuffer::new(2, 2);
        overlay.fill([255, 255, 255, 255]);
        let result = base.blend(&overlay, 1, 1, 1.0);
        assert_eq!(result.get_pixel(1, 1), [255, 255, 255, 255]);
        assert_eq!(result.get_pixel(0, 0), [0, 0, 0, 255]);
    }

    #[test]
    fn histogram_counts() {
        let mut buf = PixelBuffer::new(2, 2);
        buf.fill([128, 128, 128, 255]);
        let hist = buf.histogram();
        // All 4 pixels have luminance ~128
        assert_eq!(hist[128], 4);
    }

    #[test]
    fn rotate_90_dimensions() {
        let buf = PixelBuffer::new(10, 5);
        let rotated = buf.rotate_90();
        assert_eq!(rotated.width, 5);
        assert_eq!(rotated.height, 10);
    }

    #[test]
    fn contrast_clamps() {
        let mut buf = PixelBuffer::new(1, 1);
        buf.set_pixel(0, 0, [200, 50, 128, 255]);
        let hi = buf.contrast(10.0);
        let px = hi.get_pixel(0, 0);
        assert_eq!(px[0], 255); // clamped high
        assert_eq!(px[1], 0);   // clamped low
    }
}
