//! Pixel buffer — RGBA/RGB/Grayscale formats, buffer creation, pixel get/set,
//! fill, blit, flip, rotate, crop, resize (nearest neighbor, bilinear).
//!
//! Replaces HTML5 Canvas ImageData and JS image manipulation libraries with a
//! pure-Rust pixel buffer that works identically on native and WASM targets.

use serde::{Deserialize, Serialize};

// ── Pixel Format ─────────────────────────────────────────────────

/// Supported pixel formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PixelFormat {
    /// 4 bytes per pixel: red, green, blue, alpha.
    Rgba,
    /// 3 bytes per pixel: red, green, blue.
    Rgb,
    /// 1 byte per pixel: luminance.
    Grayscale,
}

impl PixelFormat {
    /// Bytes per pixel for this format.
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgba => 4,
            Self::Rgb => 3,
            Self::Grayscale => 1,
        }
    }
}

// ── Errors ───────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PixelBufferError {
    OutOfBounds { x: u32, y: u32, width: u32, height: u32 },
    DimensionMismatch { expected: usize, actual: usize },
    InvalidRegion,
    FormatMismatch,
}

impl std::fmt::Display for PixelBufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfBounds { x, y, width, height } => {
                write!(f, "pixel ({x},{y}) out of bounds ({width}x{height})")
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "expected {expected} bytes, got {actual}")
            }
            Self::InvalidRegion => write!(f, "invalid region"),
            Self::FormatMismatch => write!(f, "pixel format mismatch"),
        }
    }
}

impl std::error::Error for PixelBufferError {}

// ── Color ────────────────────────────────────────────────────────

/// An RGBA color value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    pub const fn gray(v: u8) -> Self {
        Self { r: v, g: v, b: v, a: 255 }
    }

    pub const WHITE: Self = Self::new(255, 255, 255, 255);
    pub const BLACK: Self = Self::new(0, 0, 0, 255);
    pub const TRANSPARENT: Self = Self::new(0, 0, 0, 0);

    /// Luminance using ITU-R BT.601.
    pub fn luminance(self) -> u8 {
        ((self.r as u32 * 299 + self.g as u32 * 587 + self.b as u32 * 114) / 1000) as u8
    }
}

// ── PixelBuffer ──────────────────────────────────────────────────

/// A 2-D pixel buffer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PixelBuffer {
    width: u32,
    height: u32,
    format: PixelFormat,
    data: Vec<u8>,
}

impl PixelBuffer {
    /// Create a zeroed buffer.
    pub fn new(width: u32, height: u32, format: PixelFormat) -> Self {
        let len = width as usize * height as usize * format.bytes_per_pixel();
        Self { width, height, format, data: vec![0u8; len] }
    }

    /// Create from existing data.
    pub fn from_data(
        width: u32,
        height: u32,
        format: PixelFormat,
        data: Vec<u8>,
    ) -> Result<Self, PixelBufferError> {
        let expected = width as usize * height as usize * format.bytes_per_pixel();
        if data.len() != expected {
            return Err(PixelBufferError::DimensionMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { width, height, format, data })
    }

    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn format(&self) -> PixelFormat {
        self.format
    }
    pub fn data(&self) -> &[u8] {
        &self.data
    }
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    fn offset(&self, x: u32, y: u32) -> Result<usize, PixelBufferError> {
        if x >= self.width || y >= self.height {
            return Err(PixelBufferError::OutOfBounds {
                x,
                y,
                width: self.width,
                height: self.height,
            });
        }
        Ok((y as usize * self.width as usize + x as usize) * self.format.bytes_per_pixel())
    }

    /// Get a pixel as RGBA (always converts to Color).
    pub fn get_pixel(&self, x: u32, y: u32) -> Result<Color, PixelBufferError> {
        let off = self.offset(x, y)?;
        Ok(match self.format {
            PixelFormat::Rgba => {
                Color::new(self.data[off], self.data[off + 1], self.data[off + 2], self.data[off + 3])
            }
            PixelFormat::Rgb => Color::rgb(self.data[off], self.data[off + 1], self.data[off + 2]),
            PixelFormat::Grayscale => Color::gray(self.data[off]),
        })
    }

    /// Set a pixel from a Color (converts to buffer format).
    pub fn set_pixel(&mut self, x: u32, y: u32, c: Color) -> Result<(), PixelBufferError> {
        let off = self.offset(x, y)?;
        match self.format {
            PixelFormat::Rgba => {
                self.data[off] = c.r;
                self.data[off + 1] = c.g;
                self.data[off + 2] = c.b;
                self.data[off + 3] = c.a;
            }
            PixelFormat::Rgb => {
                self.data[off] = c.r;
                self.data[off + 1] = c.g;
                self.data[off + 2] = c.b;
            }
            PixelFormat::Grayscale => {
                self.data[off] = c.luminance();
            }
        }
        Ok(())
    }

    /// Fill the entire buffer with a color.
    pub fn fill(&mut self, c: Color) {
        for y in 0..self.height {
            for x in 0..self.width {
                let _ = self.set_pixel(x, y, c);
            }
        }
    }

    /// Copy a rectangular region from `src` into `self` at `(dst_x, dst_y)`.
    pub fn blit(
        &mut self,
        src: &PixelBuffer,
        src_x: u32,
        src_y: u32,
        width: u32,
        height: u32,
        dst_x: u32,
        dst_y: u32,
    ) -> Result<(), PixelBufferError> {
        if src_x + width > src.width || src_y + height > src.height {
            return Err(PixelBufferError::InvalidRegion);
        }
        if dst_x + width > self.width || dst_y + height > self.height {
            return Err(PixelBufferError::InvalidRegion);
        }
        for row in 0..height {
            for col in 0..width {
                let c = src.get_pixel(src_x + col, src_y + row)?;
                self.set_pixel(dst_x + col, dst_y + row, c)?;
            }
        }
        Ok(())
    }

    /// Flip horizontally (mirror left-right).
    pub fn flip_horizontal(&mut self) {
        for y in 0..self.height {
            for x in 0..self.width / 2 {
                let rx = self.width - 1 - x;
                let left = self.get_pixel(x, y).unwrap();
                let right = self.get_pixel(rx, y).unwrap();
                let _ = self.set_pixel(x, y, right);
                let _ = self.set_pixel(rx, y, left);
            }
        }
    }

    /// Flip vertically (mirror top-bottom).
    pub fn flip_vertical(&mut self) {
        let bpp = self.format.bytes_per_pixel();
        let row_bytes = self.width as usize * bpp;
        let mut tmp = vec![0u8; row_bytes];
        for y in 0..self.height as usize / 2 {
            let by = self.height as usize - 1 - y;
            let top_off = y * row_bytes;
            let bot_off = by * row_bytes;
            tmp.copy_from_slice(&self.data[top_off..top_off + row_bytes]);
            self.data.copy_within(bot_off..bot_off + row_bytes, top_off);
            self.data[bot_off..bot_off + row_bytes].copy_from_slice(&tmp);
        }
    }

    /// Rotate 90 degrees clockwise.
    pub fn rotate_90(&self) -> Self {
        let mut out = PixelBuffer::new(self.height, self.width, self.format);
        for y in 0..self.height {
            for x in 0..self.width {
                let c = self.get_pixel(x, y).unwrap();
                let _ = out.set_pixel(self.height - 1 - y, x, c);
            }
        }
        out
    }

    /// Rotate 180 degrees.
    pub fn rotate_180(&self) -> Self {
        let mut out = PixelBuffer::new(self.width, self.height, self.format);
        for y in 0..self.height {
            for x in 0..self.width {
                let c = self.get_pixel(x, y).unwrap();
                let _ = out.set_pixel(self.width - 1 - x, self.height - 1 - y, c);
            }
        }
        out
    }

    /// Rotate 270 degrees clockwise (= 90 CCW).
    pub fn rotate_270(&self) -> Self {
        let mut out = PixelBuffer::new(self.height, self.width, self.format);
        for y in 0..self.height {
            for x in 0..self.width {
                let c = self.get_pixel(x, y).unwrap();
                let _ = out.set_pixel(y, self.width - 1 - x, c);
            }
        }
        out
    }

    /// Crop a rectangular region.
    pub fn crop(&self, x: u32, y: u32, w: u32, h: u32) -> Result<Self, PixelBufferError> {
        if x + w > self.width || y + h > self.height {
            return Err(PixelBufferError::InvalidRegion);
        }
        let mut out = PixelBuffer::new(w, h, self.format);
        for row in 0..h {
            for col in 0..w {
                let c = self.get_pixel(x + col, y + row)?;
                let _ = out.set_pixel(col, row, c);
            }
        }
        Ok(out)
    }

    /// Resize using nearest-neighbor interpolation.
    pub fn resize_nearest(&self, new_w: u32, new_h: u32) -> Self {
        let mut out = PixelBuffer::new(new_w, new_h, self.format);
        for y in 0..new_h {
            for x in 0..new_w {
                let sx = (x as u64 * self.width as u64 / new_w as u64) as u32;
                let sy = (y as u64 * self.height as u64 / new_h as u64) as u32;
                let sx = sx.min(self.width - 1);
                let sy = sy.min(self.height - 1);
                let c = self.get_pixel(sx, sy).unwrap();
                let _ = out.set_pixel(x, y, c);
            }
        }
        out
    }

    /// Resize using bilinear interpolation.
    pub fn resize_bilinear(&self, new_w: u32, new_h: u32) -> Self {
        let mut out = PixelBuffer::new(new_w, new_h, self.format);
        if self.width == 0 || self.height == 0 || new_w == 0 || new_h == 0 {
            return out;
        }
        for y in 0..new_h {
            for x in 0..new_w {
                let gx = x as f64 * (self.width - 1) as f64 / new_w.max(1) as f64;
                let gy = y as f64 * (self.height - 1) as f64 / new_h.max(1) as f64;
                let x0 = gx.floor() as u32;
                let y0 = gy.floor() as u32;
                let x1 = (x0 + 1).min(self.width - 1);
                let y1 = (y0 + 1).min(self.height - 1);
                let fx = gx - x0 as f64;
                let fy = gy - y0 as f64;

                let c00 = self.get_pixel(x0, y0).unwrap();
                let c10 = self.get_pixel(x1, y0).unwrap();
                let c01 = self.get_pixel(x0, y1).unwrap();
                let c11 = self.get_pixel(x1, y1).unwrap();

                let lerp = |a: u8, b: u8, c: u8, d: u8| -> u8 {
                    let v = a as f64 * (1.0 - fx) * (1.0 - fy)
                        + b as f64 * fx * (1.0 - fy)
                        + c as f64 * (1.0 - fx) * fy
                        + d as f64 * fx * fy;
                    v.round().clamp(0.0, 255.0) as u8
                };

                let c = Color::new(
                    lerp(c00.r, c10.r, c01.r, c11.r),
                    lerp(c00.g, c10.g, c01.g, c11.g),
                    lerp(c00.b, c10.b, c01.b, c11.b),
                    lerp(c00.a, c10.a, c01.a, c11.a),
                );
                let _ = out.set_pixel(x, y, c);
            }
        }
        out
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_buffer_zeroed() {
        let buf = PixelBuffer::new(4, 4, PixelFormat::Rgba);
        assert_eq!(buf.data().len(), 64);
        assert!(buf.data().iter().all(|b| *b == 0));
    }

    #[test]
    fn test_from_data_dimension_check() {
        let r = PixelBuffer::from_data(2, 2, PixelFormat::Rgb, vec![0u8; 11]);
        assert!(r.is_err());
        let r = PixelBuffer::from_data(2, 2, PixelFormat::Rgb, vec![0u8; 12]);
        assert!(r.is_ok());
    }

    #[test]
    fn test_get_set_pixel_rgba() {
        let mut buf = PixelBuffer::new(3, 3, PixelFormat::Rgba);
        buf.set_pixel(1, 2, Color::new(10, 20, 30, 40)).unwrap();
        let c = buf.get_pixel(1, 2).unwrap();
        assert_eq!(c, Color::new(10, 20, 30, 40));
    }

    #[test]
    fn test_get_set_pixel_grayscale() {
        let mut buf = PixelBuffer::new(2, 2, PixelFormat::Grayscale);
        buf.set_pixel(0, 0, Color::rgb(100, 150, 200)).unwrap();
        let c = buf.get_pixel(0, 0).unwrap();
        // Luminance of (100,150,200) ≈ 142
        assert!(c.r > 130 && c.r < 160);
    }

    #[test]
    fn test_out_of_bounds() {
        let buf = PixelBuffer::new(2, 2, PixelFormat::Rgba);
        assert!(buf.get_pixel(2, 0).is_err());
        assert!(buf.get_pixel(0, 2).is_err());
    }

    #[test]
    fn test_fill() {
        let mut buf = PixelBuffer::new(3, 3, PixelFormat::Rgba);
        buf.fill(Color::WHITE);
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(buf.get_pixel(x, y).unwrap(), Color::WHITE);
            }
        }
    }

    #[test]
    fn test_blit() {
        let mut src = PixelBuffer::new(4, 4, PixelFormat::Rgba);
        src.fill(Color::rgb(255, 0, 0));
        let mut dst = PixelBuffer::new(6, 6, PixelFormat::Rgba);
        dst.blit(&src, 0, 0, 2, 2, 1, 1).unwrap();
        assert_eq!(dst.get_pixel(1, 1).unwrap(), Color::rgb(255, 0, 0));
        assert_eq!(dst.get_pixel(0, 0).unwrap().r, 0);
    }

    #[test]
    fn test_flip_horizontal() {
        let mut buf = PixelBuffer::new(3, 1, PixelFormat::Rgba);
        buf.set_pixel(0, 0, Color::rgb(10, 0, 0)).unwrap();
        buf.set_pixel(2, 0, Color::rgb(30, 0, 0)).unwrap();
        buf.flip_horizontal();
        assert_eq!(buf.get_pixel(0, 0).unwrap().r, 30);
        assert_eq!(buf.get_pixel(2, 0).unwrap().r, 10);
    }

    #[test]
    fn test_flip_vertical() {
        let mut buf = PixelBuffer::new(1, 3, PixelFormat::Rgba);
        buf.set_pixel(0, 0, Color::rgb(10, 0, 0)).unwrap();
        buf.set_pixel(0, 2, Color::rgb(30, 0, 0)).unwrap();
        buf.flip_vertical();
        assert_eq!(buf.get_pixel(0, 0).unwrap().r, 30);
        assert_eq!(buf.get_pixel(0, 2).unwrap().r, 10);
    }

    #[test]
    fn test_rotate_90() {
        let mut buf = PixelBuffer::new(3, 2, PixelFormat::Rgba);
        buf.set_pixel(0, 0, Color::rgb(1, 0, 0)).unwrap();
        buf.set_pixel(2, 1, Color::rgb(2, 0, 0)).unwrap();
        let rot = buf.rotate_90();
        assert_eq!(rot.width(), 2);
        assert_eq!(rot.height(), 3);
        // (0,0) → (h-1-0, 0) = (1, 0)
        assert_eq!(rot.get_pixel(1, 0).unwrap().r, 1);
    }

    #[test]
    fn test_rotate_180() {
        let mut buf = PixelBuffer::new(2, 2, PixelFormat::Rgba);
        buf.set_pixel(0, 0, Color::rgb(1, 0, 0)).unwrap();
        let rot = buf.rotate_180();
        assert_eq!(rot.get_pixel(1, 1).unwrap().r, 1);
    }

    #[test]
    fn test_crop() {
        let mut buf = PixelBuffer::new(4, 4, PixelFormat::Rgba);
        buf.fill(Color::WHITE);
        buf.set_pixel(1, 1, Color::rgb(42, 0, 0)).unwrap();
        let cropped = buf.crop(1, 1, 2, 2).unwrap();
        assert_eq!(cropped.width(), 2);
        assert_eq!(cropped.height(), 2);
        assert_eq!(cropped.get_pixel(0, 0).unwrap().r, 42);
    }

    #[test]
    fn test_crop_invalid() {
        let buf = PixelBuffer::new(4, 4, PixelFormat::Rgba);
        assert!(buf.crop(3, 3, 2, 2).is_err());
    }

    #[test]
    fn test_resize_nearest() {
        let mut buf = PixelBuffer::new(2, 2, PixelFormat::Rgba);
        buf.fill(Color::rgb(100, 100, 100));
        let resized = buf.resize_nearest(4, 4);
        assert_eq!(resized.width(), 4);
        assert_eq!(resized.height(), 4);
        assert_eq!(resized.get_pixel(0, 0).unwrap().r, 100);
    }

    #[test]
    fn test_resize_bilinear() {
        let mut buf = PixelBuffer::new(2, 2, PixelFormat::Rgba);
        buf.set_pixel(0, 0, Color::rgb(0, 0, 0)).unwrap();
        buf.set_pixel(1, 0, Color::rgb(255, 0, 0)).unwrap();
        buf.set_pixel(0, 1, Color::rgb(0, 255, 0)).unwrap();
        buf.set_pixel(1, 1, Color::rgb(255, 255, 0)).unwrap();
        let resized = buf.resize_bilinear(4, 4);
        assert_eq!(resized.width(), 4);
        // Center pixel should be a blend
        let c = resized.get_pixel(2, 2).unwrap();
        assert!(c.r > 50 && c.r < 200);
    }

    #[test]
    fn test_rotate_270() {
        let mut buf = PixelBuffer::new(3, 2, PixelFormat::Rgba);
        buf.set_pixel(0, 0, Color::rgb(5, 0, 0)).unwrap();
        let rot = buf.rotate_270();
        assert_eq!(rot.width(), 2);
        assert_eq!(rot.height(), 3);
        // (0,0) → (0, w-1-0) = (0, 2)
        assert_eq!(rot.get_pixel(0, 2).unwrap().r, 5);
    }

    #[test]
    fn test_luminance() {
        let c = Color::rgb(255, 255, 255);
        assert_eq!(c.luminance(), 255);
        let c = Color::rgb(0, 0, 0);
        assert_eq!(c.luminance(), 0);
    }

    #[test]
    fn test_format_bytes_per_pixel() {
        assert_eq!(PixelFormat::Rgba.bytes_per_pixel(), 4);
        assert_eq!(PixelFormat::Rgb.bytes_per_pixel(), 3);
        assert_eq!(PixelFormat::Grayscale.bytes_per_pixel(), 1);
    }
}
