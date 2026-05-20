//! Image operations — resize nearest/bilinear, crop, rotate 90/180/270,
//! flip, grayscale, brightness/contrast, pixel buffer, format-agnostic.
//!
//! Pure-Rust replacement for the `image` crate's core operations.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ImageError {
    InvalidDimensions { width: u32, height: u32, data_len: usize },
    CropOutOfBounds,
    EmptyImage,
}

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImageError::InvalidDimensions { width, height, data_len } =>
                write!(f, "data length {data_len} does not match {width}x{height} image"),
            ImageError::CropOutOfBounds => write!(f, "crop region out of bounds"),
            ImageError::EmptyImage => write!(f, "empty image"),
        }
    }
}

// ── Pixel ───────────────────────────────────────────────────────

/// An RGBA pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgba {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Rgba {
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self { Self { r, g, b, a } }
    pub fn rgb(r: u8, g: u8, b: u8) -> Self { Self { r, g, b, a: 255 } }
    pub fn gray(v: u8) -> Self { Self { r: v, g: v, b: v, a: 255 } }

    /// Convert to grayscale using luminance formula.
    pub fn to_gray(&self) -> u8 {
        let lum = 0.299 * self.r as f64 + 0.587 * self.g as f64 + 0.114 * self.b as f64;
        lum.round().min(255.0).max(0.0) as u8
    }

    /// Linearly interpolate between two pixels.
    pub fn lerp(&self, other: &Rgba, t: f64) -> Rgba {
        let inv = 1.0 - t;
        Rgba {
            r: (self.r as f64 * inv + other.r as f64 * t).round().min(255.0).max(0.0) as u8,
            g: (self.g as f64 * inv + other.g as f64 * t).round().min(255.0).max(0.0) as u8,
            b: (self.b as f64 * inv + other.b as f64 * t).round().min(255.0).max(0.0) as u8,
            a: (self.a as f64 * inv + other.a as f64 * t).round().min(255.0).max(0.0) as u8,
        }
    }
}

// ── Pixel Buffer ────────────────────────────────────────────────

/// A 2D buffer of RGBA pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct PixelBuffer {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<Rgba>,
}

impl PixelBuffer {
    /// Create a new pixel buffer. `pixels.len()` must equal `width * height`.
    pub fn new(width: u32, height: u32, pixels: Vec<Rgba>) -> Result<Self, ImageError> {
        let expected = (width as usize) * (height as usize);
        if pixels.len() != expected {
            return Err(ImageError::InvalidDimensions { width, height, data_len: pixels.len() });
        }
        Ok(Self { width, height, pixels })
    }

    /// Create a buffer filled with a single color.
    pub fn filled(width: u32, height: u32, color: Rgba) -> Self {
        let pixels = vec![color; (width as usize) * (height as usize)];
        Self { width, height, pixels }
    }

    /// Get a pixel at (x, y).
    pub fn get_pixel(&self, x: u32, y: u32) -> Rgba {
        self.pixels[(y as usize) * (self.width as usize) + (x as usize)]
    }

    /// Set a pixel at (x, y).
    pub fn set_pixel(&mut self, x: u32, y: u32, color: Rgba) {
        self.pixels[(y as usize) * (self.width as usize) + (x as usize)] = color;
    }

    /// Get pixel with bounds clamping.
    pub fn get_pixel_clamped(&self, x: i32, y: i32) -> Rgba {
        let cx = x.max(0).min(self.width as i32 - 1) as u32;
        let cy = y.max(0).min(self.height as i32 - 1) as u32;
        self.get_pixel(cx, cy)
    }

    /// Create from raw RGBA bytes (4 bytes per pixel).
    pub fn from_rgba_bytes(width: u32, height: u32, data: &[u8]) -> Result<Self, ImageError> {
        let expected = (width as usize) * (height as usize) * 4;
        if data.len() != expected {
            return Err(ImageError::InvalidDimensions { width, height, data_len: data.len() });
        }
        let pixels: Vec<Rgba> = data.chunks_exact(4)
            .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
            .collect();
        Ok(Self { width, height, pixels })
    }

    /// Convert to raw RGBA bytes.
    pub fn to_rgba_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.pixels.len() * 4);
        for p in &self.pixels {
            bytes.push(p.r);
            bytes.push(p.g);
            bytes.push(p.b);
            bytes.push(p.a);
        }
        bytes
    }

    /// Convert to raw RGB bytes (no alpha).
    pub fn to_rgb_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.pixels.len() * 3);
        for p in &self.pixels {
            bytes.push(p.r);
            bytes.push(p.g);
            bytes.push(p.b);
        }
        bytes
    }
}

// ── Resize ──────────────────────────────────────────────────────

/// Interpolation method for resizing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeFilter { Nearest, Bilinear }

/// Resize an image to new dimensions.
pub fn resize(buf: &PixelBuffer, new_width: u32, new_height: u32, filter: ResizeFilter) -> PixelBuffer {
    if new_width == 0 || new_height == 0 {
        return PixelBuffer { width: new_width, height: new_height, pixels: Vec::new() };
    }
    let mut out = Vec::with_capacity((new_width as usize) * (new_height as usize));
    for y in 0..new_height {
        for x in 0..new_width {
            let src_x = x as f64 * (buf.width as f64) / (new_width as f64);
            let src_y = y as f64 * (buf.height as f64) / (new_height as f64);
            let pixel = match filter {
                ResizeFilter::Nearest => {
                    let sx = (src_x.floor() as u32).min(buf.width - 1);
                    let sy = (src_y.floor() as u32).min(buf.height - 1);
                    buf.get_pixel(sx, sy)
                }
                ResizeFilter::Bilinear => {
                    bilinear_sample(buf, src_x, src_y)
                }
            };
            out.push(pixel);
        }
    }
    PixelBuffer { width: new_width, height: new_height, pixels: out }
}

fn bilinear_sample(buf: &PixelBuffer, x: f64, y: f64) -> Rgba {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let p00 = buf.get_pixel_clamped(x0, y0);
    let p10 = buf.get_pixel_clamped(x0 + 1, y0);
    let p01 = buf.get_pixel_clamped(x0, y0 + 1);
    let p11 = buf.get_pixel_clamped(x0 + 1, y0 + 1);

    let top = p00.lerp(&p10, fx);
    let bottom = p01.lerp(&p11, fx);
    top.lerp(&bottom, fy)
}

// ── Crop ────────────────────────────────────────────────────────

/// Crop a rectangular region from the image.
pub fn crop(buf: &PixelBuffer, x: u32, y: u32, w: u32, h: u32) -> Result<PixelBuffer, ImageError> {
    if x + w > buf.width || y + h > buf.height {
        return Err(ImageError::CropOutOfBounds);
    }
    let mut pixels = Vec::with_capacity((w as usize) * (h as usize));
    for row in y..y + h {
        for col in x..x + w {
            pixels.push(buf.get_pixel(col, row));
        }
    }
    Ok(PixelBuffer { width: w, height: h, pixels })
}

// ── Rotate ──────────────────────────────────────────────────────

/// Rotate 90 degrees clockwise.
pub fn rotate_90(buf: &PixelBuffer) -> PixelBuffer {
    let new_w = buf.height;
    let new_h = buf.width;
    let mut pixels = vec![Rgba::gray(0); (new_w as usize) * (new_h as usize)];
    for y in 0..buf.height {
        for x in 0..buf.width {
            let nx = buf.height - 1 - y;
            let ny = x;
            pixels[(ny as usize) * (new_w as usize) + (nx as usize)] = buf.get_pixel(x, y);
        }
    }
    PixelBuffer { width: new_w, height: new_h, pixels }
}

/// Rotate 180 degrees.
pub fn rotate_180(buf: &PixelBuffer) -> PixelBuffer {
    let mut pixels = vec![Rgba::gray(0); buf.pixels.len()];
    for y in 0..buf.height {
        for x in 0..buf.width {
            let nx = buf.width - 1 - x;
            let ny = buf.height - 1 - y;
            pixels[(ny as usize) * (buf.width as usize) + (nx as usize)] = buf.get_pixel(x, y);
        }
    }
    PixelBuffer { width: buf.width, height: buf.height, pixels }
}

/// Rotate 270 degrees clockwise (= 90 counter-clockwise).
pub fn rotate_270(buf: &PixelBuffer) -> PixelBuffer {
    let new_w = buf.height;
    let new_h = buf.width;
    let mut pixels = vec![Rgba::gray(0); (new_w as usize) * (new_h as usize)];
    for y in 0..buf.height {
        for x in 0..buf.width {
            let nx = y;
            let ny = buf.width - 1 - x;
            pixels[(ny as usize) * (new_w as usize) + (nx as usize)] = buf.get_pixel(x, y);
        }
    }
    PixelBuffer { width: new_w, height: new_h, pixels }
}

// ── Flip ────────────────────────────────────────────────────────

/// Flip horizontally (mirror around vertical axis).
pub fn flip_horizontal(buf: &PixelBuffer) -> PixelBuffer {
    let mut pixels = vec![Rgba::gray(0); buf.pixels.len()];
    for y in 0..buf.height {
        for x in 0..buf.width {
            pixels[(y as usize) * (buf.width as usize) + (buf.width - 1 - x) as usize] =
                buf.get_pixel(x, y);
        }
    }
    PixelBuffer { width: buf.width, height: buf.height, pixels }
}

/// Flip vertically (mirror around horizontal axis).
pub fn flip_vertical(buf: &PixelBuffer) -> PixelBuffer {
    let mut pixels = vec![Rgba::gray(0); buf.pixels.len()];
    for y in 0..buf.height {
        for x in 0..buf.width {
            pixels[((buf.height - 1 - y) as usize) * (buf.width as usize) + (x as usize)] =
                buf.get_pixel(x, y);
        }
    }
    PixelBuffer { width: buf.width, height: buf.height, pixels }
}

// ── Color Adjustments ───────────────────────────────────────────

/// Convert the image to grayscale.
pub fn grayscale(buf: &PixelBuffer) -> PixelBuffer {
    let pixels: Vec<Rgba> = buf.pixels.iter().map(|p| {
        let g = p.to_gray();
        Rgba::new(g, g, g, p.a)
    }).collect();
    PixelBuffer { width: buf.width, height: buf.height, pixels }
}

/// Adjust brightness. `factor` is added to each channel (-255..255).
pub fn adjust_brightness(buf: &PixelBuffer, factor: i16) -> PixelBuffer {
    let pixels: Vec<Rgba> = buf.pixels.iter().map(|p| {
        Rgba::new(
            clamp_u8(p.r as i16 + factor),
            clamp_u8(p.g as i16 + factor),
            clamp_u8(p.b as i16 + factor),
            p.a,
        )
    }).collect();
    PixelBuffer { width: buf.width, height: buf.height, pixels }
}

/// Adjust contrast. `factor` > 1.0 increases contrast, < 1.0 decreases.
pub fn adjust_contrast(buf: &PixelBuffer, factor: f64) -> PixelBuffer {
    let pixels: Vec<Rgba> = buf.pixels.iter().map(|p| {
        Rgba::new(
            contrast_channel(p.r, factor),
            contrast_channel(p.g, factor),
            contrast_channel(p.b, factor),
            p.a,
        )
    }).collect();
    PixelBuffer { width: buf.width, height: buf.height, pixels }
}

fn contrast_channel(v: u8, factor: f64) -> u8 {
    let centered = v as f64 - 128.0;
    let adjusted = centered * factor + 128.0;
    adjusted.round().min(255.0).max(0.0) as u8
}

/// Invert all colors.
pub fn invert(buf: &PixelBuffer) -> PixelBuffer {
    let pixels: Vec<Rgba> = buf.pixels.iter().map(|p| {
        Rgba::new(255 - p.r, 255 - p.g, 255 - p.b, p.a)
    }).collect();
    PixelBuffer { width: buf.width, height: buf.height, pixels }
}

/// Apply a sepia tone.
pub fn sepia(buf: &PixelBuffer) -> PixelBuffer {
    let pixels: Vec<Rgba> = buf.pixels.iter().map(|p| {
        let r = p.r as f64;
        let g = p.g as f64;
        let b = p.b as f64;
        Rgba::new(
            (r * 0.393 + g * 0.769 + b * 0.189).min(255.0).round() as u8,
            (r * 0.349 + g * 0.686 + b * 0.168).min(255.0).round() as u8,
            (r * 0.272 + g * 0.534 + b * 0.131).min(255.0).round() as u8,
            p.a,
        )
    }).collect();
    PixelBuffer { width: buf.width, height: buf.height, pixels }
}

fn clamp_u8(v: i16) -> u8 {
    v.max(0).min(255) as u8
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_2x2() -> PixelBuffer {
        PixelBuffer::new(2, 2, vec![
            Rgba::rgb(255, 0, 0),   Rgba::rgb(0, 255, 0),
            Rgba::rgb(0, 0, 255),   Rgba::rgb(255, 255, 0),
        ]).unwrap()
    }

    fn test_3x2() -> PixelBuffer {
        PixelBuffer::new(3, 2, vec![
            Rgba::rgb(1, 0, 0), Rgba::rgb(2, 0, 0), Rgba::rgb(3, 0, 0),
            Rgba::rgb(4, 0, 0), Rgba::rgb(5, 0, 0), Rgba::rgb(6, 0, 0),
        ]).unwrap()
    }

    #[test]
    fn pixel_creation() {
        let p = Rgba::rgb(100, 150, 200);
        assert_eq!(p.a, 255);
        let g = Rgba::gray(128);
        assert_eq!(g.r, 128);
        assert_eq!(g.g, 128);
    }

    #[test]
    fn pixel_to_gray() {
        let p = Rgba::rgb(255, 0, 0);
        let g = p.to_gray();
        assert_eq!(g, 76); // 0.299*255 = 76.245
    }

    #[test]
    fn pixel_lerp() {
        let a = Rgba::rgb(0, 0, 0);
        let b = Rgba::rgb(200, 100, 50);
        let mid = a.lerp(&b, 0.5);
        assert_eq!(mid.r, 100);
        assert_eq!(mid.g, 50);
        assert_eq!(mid.b, 25);
    }

    #[test]
    fn buffer_creation() {
        let buf = test_2x2();
        assert_eq!(buf.width, 2);
        assert_eq!(buf.height, 2);
    }

    #[test]
    fn buffer_invalid_dims() {
        let result = PixelBuffer::new(3, 3, vec![Rgba::gray(0); 4]);
        assert!(result.is_err());
    }

    #[test]
    fn buffer_get_set_pixel() {
        let mut buf = PixelBuffer::filled(4, 4, Rgba::gray(0));
        buf.set_pixel(2, 3, Rgba::rgb(255, 0, 0));
        assert_eq!(buf.get_pixel(2, 3), Rgba::rgb(255, 0, 0));
        assert_eq!(buf.get_pixel(0, 0), Rgba::gray(0));
    }

    #[test]
    fn buffer_clamped_access() {
        let buf = test_2x2();
        assert_eq!(buf.get_pixel_clamped(-1, -1), buf.get_pixel(0, 0));
        assert_eq!(buf.get_pixel_clamped(100, 100), buf.get_pixel(1, 1));
    }

    #[test]
    fn from_rgba_bytes() {
        let data = vec![255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 128, 128, 128, 255];
        let buf = PixelBuffer::from_rgba_bytes(2, 2, &data).unwrap();
        assert_eq!(buf.get_pixel(0, 0), Rgba::rgb(255, 0, 0));
        assert_eq!(buf.get_pixel(1, 1), Rgba::rgb(128, 128, 128));
    }

    #[test]
    fn to_rgba_bytes_roundtrip() {
        let buf = test_2x2();
        let bytes = buf.to_rgba_bytes();
        let buf2 = PixelBuffer::from_rgba_bytes(2, 2, &bytes).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn to_rgb_bytes() {
        let buf = PixelBuffer::filled(1, 1, Rgba::rgb(100, 200, 50));
        let bytes = buf.to_rgb_bytes();
        assert_eq!(bytes, vec![100, 200, 50]);
    }

    #[test]
    fn resize_nearest_upscale() {
        let buf = test_2x2();
        let resized = resize(&buf, 4, 4, ResizeFilter::Nearest);
        assert_eq!(resized.width, 4);
        assert_eq!(resized.height, 4);
        assert_eq!(resized.get_pixel(0, 0), Rgba::rgb(255, 0, 0));
    }

    #[test]
    fn resize_nearest_downscale() {
        let buf = PixelBuffer::filled(10, 10, Rgba::rgb(42, 42, 42));
        let resized = resize(&buf, 2, 2, ResizeFilter::Nearest);
        assert_eq!(resized.width, 2);
        assert_eq!(resized.get_pixel(0, 0), Rgba::rgb(42, 42, 42));
    }

    #[test]
    fn resize_bilinear() {
        let buf = test_2x2();
        let resized = resize(&buf, 4, 4, ResizeFilter::Bilinear);
        assert_eq!(resized.width, 4);
        assert_eq!(resized.height, 4);
    }

    #[test]
    fn crop_basic() {
        let buf = PixelBuffer::filled(4, 4, Rgba::gray(0));
        let cropped = crop(&buf, 1, 1, 2, 2).unwrap();
        assert_eq!(cropped.width, 2);
        assert_eq!(cropped.height, 2);
    }

    #[test]
    fn crop_out_of_bounds() {
        let buf = test_2x2();
        assert_eq!(crop(&buf, 1, 1, 2, 2), Err(ImageError::CropOutOfBounds));
    }

    #[test]
    fn rotate_90_dimensions() {
        let buf = test_3x2();
        let rotated = rotate_90(&buf);
        assert_eq!(rotated.width, 2); // height becomes width
        assert_eq!(rotated.height, 3); // width becomes height
    }

    #[test]
    fn rotate_180_identity_twice() {
        let buf = test_2x2();
        let r1 = rotate_180(&buf);
        let r2 = rotate_180(&r1);
        assert_eq!(buf, r2);
    }

    #[test]
    fn rotate_270_dimensions() {
        let buf = test_3x2();
        let rotated = rotate_270(&buf);
        assert_eq!(rotated.width, 2);
        assert_eq!(rotated.height, 3);
    }

    #[test]
    fn rotate_360_identity() {
        let buf = test_2x2();
        let r = rotate_90(&rotate_90(&rotate_90(&rotate_90(&buf))));
        assert_eq!(buf, r);
    }

    #[test]
    fn flip_horizontal_twice() {
        let buf = test_2x2();
        let flipped = flip_horizontal(&flip_horizontal(&buf));
        assert_eq!(buf, flipped);
    }

    #[test]
    fn flip_vertical_twice() {
        let buf = test_2x2();
        let flipped = flip_vertical(&flip_vertical(&buf));
        assert_eq!(buf, flipped);
    }

    #[test]
    fn flip_horizontal_pixels() {
        let buf = test_2x2();
        let flipped = flip_horizontal(&buf);
        assert_eq!(flipped.get_pixel(0, 0), Rgba::rgb(0, 255, 0));
        assert_eq!(flipped.get_pixel(1, 0), Rgba::rgb(255, 0, 0));
    }

    #[test]
    fn grayscale_conversion() {
        let buf = PixelBuffer::filled(2, 2, Rgba::rgb(100, 150, 200));
        let gray = grayscale(&buf);
        let p = gray.get_pixel(0, 0);
        assert_eq!(p.r, p.g);
        assert_eq!(p.g, p.b);
    }

    #[test]
    fn grayscale_preserves_alpha() {
        let buf = PixelBuffer::new(1, 1, vec![Rgba::new(100, 100, 100, 128)]).unwrap();
        let gray = grayscale(&buf);
        assert_eq!(gray.get_pixel(0, 0).a, 128);
    }

    #[test]
    fn brightness_increase() {
        let buf = PixelBuffer::filled(1, 1, Rgba::rgb(100, 100, 100));
        let bright = adjust_brightness(&buf, 50);
        assert_eq!(bright.get_pixel(0, 0), Rgba::rgb(150, 150, 150));
    }

    #[test]
    fn brightness_clamp() {
        let buf = PixelBuffer::filled(1, 1, Rgba::rgb(200, 200, 200));
        let bright = adjust_brightness(&buf, 100);
        assert_eq!(bright.get_pixel(0, 0).r, 255);
    }

    #[test]
    fn contrast_increase() {
        let buf = PixelBuffer::filled(1, 1, Rgba::rgb(200, 50, 128));
        let high = adjust_contrast(&buf, 2.0);
        // 200: (200-128)*2+128 = 272 -> clamped to 255
        assert_eq!(high.get_pixel(0, 0).r, 255);
        // 50: (50-128)*2+128 = -28 -> clamped to 0
        assert_eq!(high.get_pixel(0, 0).g, 0);
        // 128: (128-128)*2+128 = 128
        assert_eq!(high.get_pixel(0, 0).b, 128);
    }

    #[test]
    fn invert_colors() {
        let buf = PixelBuffer::filled(1, 1, Rgba::rgb(100, 200, 0));
        let inv = invert(&buf);
        assert_eq!(inv.get_pixel(0, 0), Rgba::rgb(155, 55, 255));
    }

    #[test]
    fn invert_preserves_alpha() {
        let buf = PixelBuffer::new(1, 1, vec![Rgba::new(100, 100, 100, 50)]).unwrap();
        let inv = invert(&buf);
        assert_eq!(inv.get_pixel(0, 0).a, 50);
    }

    #[test]
    fn sepia_filter() {
        let buf = PixelBuffer::filled(1, 1, Rgba::rgb(100, 100, 100));
        let sep = sepia(&buf);
        let p = sep.get_pixel(0, 0);
        // Sepia should have warm tones: r > g > b
        assert!(p.r >= p.g);
        assert!(p.g >= p.b);
    }

    #[test]
    fn resize_zero_dimension() {
        let buf = test_2x2();
        let r = resize(&buf, 0, 0, ResizeFilter::Nearest);
        assert_eq!(r.width, 0);
        assert_eq!(r.pixels.len(), 0);
    }

    #[test]
    fn error_display() {
        let e = ImageError::CropOutOfBounds;
        assert_eq!(format!("{e}"), "crop region out of bounds");
    }

    #[test]
    fn filled_buffer() {
        let buf = PixelBuffer::filled(3, 3, Rgba::rgb(50, 50, 50));
        assert_eq!(buf.pixels.len(), 9);
        assert_eq!(buf.get_pixel(2, 2), Rgba::rgb(50, 50, 50));
    }
}
