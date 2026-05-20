//! Image preprocessing pipeline for vision models.
//!
//! Nearest-neighbor resize, center crop, normalize, HWC↔CHW transpose,
//! RGB→grayscale, letterbox, and colour jitter. All operations work on
//! `ImageBuffer` — a flat f32 pixel buffer with HWC or CHW layout.

// ── ImageBuffer ─────────────────────────────────────────────────

/// A floating-point image buffer.
///
/// Layout can be HWC (height × width × channels) or CHW (channels × height × width).
#[derive(Debug, Clone, PartialEq)]
pub struct ImageBuffer {
    pub height: usize,
    pub width: usize,
    pub channels: usize,
    /// Pixel data as f32. Length = height * width * channels.
    pub data: Vec<f32>,
    /// True if CHW layout, false if HWC.
    pub is_chw: bool,
}

impl ImageBuffer {
    /// Create a new HWC buffer filled with zeros.
    pub fn new(height: usize, width: usize, channels: usize) -> Self {
        Self {
            height,
            width,
            channels,
            data: vec![0.0; height * width * channels],
            is_chw: false,
        }
    }

    /// Create from raw f32 data in HWC layout.
    pub fn from_hwc(height: usize, width: usize, channels: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), height * width * channels, "data length mismatch");
        Self { height, width, channels, data, is_chw: false }
    }

    /// Get pixel value at (row, col, channel) regardless of layout.
    pub fn get(&self, row: usize, col: usize, ch: usize) -> f32 {
        if self.is_chw {
            self.data[ch * self.height * self.width + row * self.width + col]
        } else {
            self.data[row * self.width * self.channels + col * self.channels + ch]
        }
    }

    /// Set pixel value at (row, col, channel) regardless of layout.
    pub fn set(&mut self, row: usize, col: usize, ch: usize, value: f32) {
        if self.is_chw {
            let idx = ch * self.height * self.width + row * self.width + col;
            self.data[idx] = value;
        } else {
            let idx = row * self.width * self.channels + col * self.channels + ch;
            self.data[idx] = value;
        }
    }
}

// ── Transform enum ──────────────────────────────────────────────

/// A single image preprocessing transform.
#[derive(Debug, Clone)]
pub enum Transform {
    /// Resize to (height, width) using nearest-neighbor interpolation.
    Resize { height: usize, width: usize },
    /// Center crop to (height, width).
    CenterCrop { height: usize, width: usize },
    /// Per-channel normalization: (pixel - mean) / std.
    Normalize { mean: Vec<f32>, std: Vec<f32> },
    /// Convert HWC → CHW layout.
    HwcToChw,
    /// Convert RGB → grayscale (channel count 3 → 1).
    RgbToGrayscale,
    /// Letterbox resize: fit into (height, width) preserving aspect ratio, pad with `fill`.
    Letterbox { height: usize, width: usize, fill: f32 },
    /// Brightness adjustment: pixel * factor.
    Brightness { factor: f32 },
    /// Contrast adjustment: (pixel - 0.5) * factor + 0.5.
    Contrast { factor: f32 },
}

// ── Individual transform implementations ────────────────────────

fn resize_nearest(img: &ImageBuffer, new_h: usize, new_w: usize) -> ImageBuffer {
    let mut out = ImageBuffer::new(new_h, new_w, img.channels);
    for r in 0..new_h {
        let src_r = (r * img.height) / new_h;
        for c in 0..new_w {
            let src_c = (c * img.width) / new_w;
            for ch in 0..img.channels {
                out.set(r, c, ch, img.get(src_r, src_c, ch));
            }
        }
    }
    out
}

fn center_crop(img: &ImageBuffer, crop_h: usize, crop_w: usize) -> ImageBuffer {
    assert!(crop_h <= img.height && crop_w <= img.width, "crop larger than image");
    let y_off = (img.height - crop_h) / 2;
    let x_off = (img.width - crop_w) / 2;
    let mut out = ImageBuffer::new(crop_h, crop_w, img.channels);
    for r in 0..crop_h {
        for c in 0..crop_w {
            for ch in 0..img.channels {
                out.set(r, c, ch, img.get(r + y_off, c + x_off, ch));
            }
        }
    }
    out
}

fn normalize(img: &ImageBuffer, mean: &[f32], std: &[f32]) -> ImageBuffer {
    assert_eq!(mean.len(), img.channels);
    assert_eq!(std.len(), img.channels);
    let mut out = img.clone();
    for r in 0..img.height {
        for c in 0..img.width {
            for ch in 0..img.channels {
                let v = img.get(r, c, ch);
                out.set(r, c, ch, (v - mean[ch]) / std[ch]);
            }
        }
    }
    out
}

fn hwc_to_chw(img: &ImageBuffer) -> ImageBuffer {
    assert!(!img.is_chw, "already CHW");
    let mut data = vec![0.0f32; img.height * img.width * img.channels];
    let hw = img.height * img.width;
    for r in 0..img.height {
        for c in 0..img.width {
            for ch in 0..img.channels {
                data[ch * hw + r * img.width + c] = img.get(r, c, ch);
            }
        }
    }
    ImageBuffer {
        height: img.height,
        width: img.width,
        channels: img.channels,
        data,
        is_chw: true,
    }
}

fn rgb_to_grayscale(img: &ImageBuffer) -> ImageBuffer {
    assert_eq!(img.channels, 3, "expected 3 channels for RGB");
    let mut out = ImageBuffer::new(img.height, img.width, 1);
    for r in 0..img.height {
        for c in 0..img.width {
            let gray = img.get(r, c, 0) * 0.299
                + img.get(r, c, 1) * 0.587
                + img.get(r, c, 2) * 0.114;
            out.set(r, c, 0, gray);
        }
    }
    out
}

fn letterbox(img: &ImageBuffer, target_h: usize, target_w: usize, fill: f32) -> ImageBuffer {
    let scale_h = target_h as f32 / img.height as f32;
    let scale_w = target_w as f32 / img.width as f32;
    let scale = scale_h.min(scale_w);

    let new_h = (img.height as f32 * scale).round() as usize;
    let new_w = (img.width as f32 * scale).round() as usize;

    let resized = resize_nearest(img, new_h, new_w);

    let mut out = ImageBuffer::new(target_h, target_w, img.channels);
    // Fill with pad value.
    for v in &mut out.data {
        *v = fill;
    }

    let y_off = (target_h - new_h) / 2;
    let x_off = (target_w - new_w) / 2;
    for r in 0..new_h {
        for c in 0..new_w {
            for ch in 0..img.channels {
                out.set(r + y_off, c + x_off, ch, resized.get(r, c, ch));
            }
        }
    }
    out
}

fn brightness(img: &ImageBuffer, factor: f32) -> ImageBuffer {
    let mut out = img.clone();
    for v in &mut out.data {
        *v *= factor;
    }
    out
}

fn contrast(img: &ImageBuffer, factor: f32) -> ImageBuffer {
    let mut out = img.clone();
    for v in &mut out.data {
        *v = (*v - 0.5) * factor + 0.5;
    }
    out
}

fn apply_transform(img: &ImageBuffer, transform: &Transform) -> ImageBuffer {
    match transform {
        Transform::Resize { height, width } => resize_nearest(img, *height, *width),
        Transform::CenterCrop { height, width } => center_crop(img, *height, *width),
        Transform::Normalize { mean, std } => normalize(img, mean, std),
        Transform::HwcToChw => hwc_to_chw(img),
        Transform::RgbToGrayscale => rgb_to_grayscale(img),
        Transform::Letterbox { height, width, fill } => letterbox(img, *height, *width, *fill),
        Transform::Brightness { factor } => brightness(img, *factor),
        Transform::Contrast { factor } => contrast(img, *factor),
    }
}

// ── Pipeline builder ────────────────────────────────────────────

/// A pipeline of chained image transforms.
#[derive(Debug, Clone)]
pub struct VisionPipeline {
    transforms: Vec<Transform>,
}

impl VisionPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self { transforms: Vec::new() }
    }

    /// Append a transform.
    pub fn add(mut self, t: Transform) -> Self {
        self.transforms.push(t);
        self
    }

    /// Run all transforms in order.
    pub fn run(&self, img: &ImageBuffer) -> ImageBuffer {
        let mut current = img.clone();
        for t in &self.transforms {
            current = apply_transform(&current, t);
        }
        current
    }

    /// Number of transforms.
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Whether the pipeline is empty.
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }
}

impl Default for VisionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rgb(h: usize, w: usize, val: f32) -> ImageBuffer {
        ImageBuffer::from_hwc(h, w, 3, vec![val; h * w * 3])
    }

    #[test]
    fn test_resize_nearest() {
        let img = make_rgb(4, 4, 0.5);
        let resized = resize_nearest(&img, 2, 2);
        assert_eq!(resized.height, 2);
        assert_eq!(resized.width, 2);
        assert_eq!(resized.channels, 3);
        assert!((resized.get(0, 0, 0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_center_crop() {
        let mut img = ImageBuffer::new(6, 6, 1);
        // Mark center pixels
        img.set(2, 2, 0, 1.0);
        img.set(3, 3, 0, 1.0);
        let cropped = center_crop(&img, 2, 2);
        assert_eq!(cropped.height, 2);
        assert_eq!(cropped.width, 2);
        assert_eq!(cropped.get(0, 0, 0), 1.0);
        assert_eq!(cropped.get(1, 1, 0), 1.0);
    }

    #[test]
    fn test_normalize() {
        let img = ImageBuffer::from_hwc(1, 1, 3, vec![0.5, 0.5, 0.5]);
        let normed = normalize(&img, &[0.5, 0.5, 0.5], &[0.5, 0.5, 0.5]);
        for ch in 0..3 {
            assert!((normed.get(0, 0, ch)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_hwc_to_chw() {
        let img = ImageBuffer::from_hwc(2, 2, 3, vec![
            1.0, 2.0, 3.0,  4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,  10.0, 11.0, 12.0,
        ]);
        let chw = hwc_to_chw(&img);
        assert!(chw.is_chw);
        // Channel 0: [1, 4, 7, 10]
        assert_eq!(chw.get(0, 0, 0), 1.0);
        assert_eq!(chw.get(0, 1, 0), 4.0);
        assert_eq!(chw.get(1, 0, 0), 7.0);
        // Channel 1: [2, 5, 8, 11]
        assert_eq!(chw.get(0, 0, 1), 2.0);
    }

    #[test]
    fn test_rgb_to_grayscale() {
        let img = ImageBuffer::from_hwc(1, 1, 3, vec![1.0, 0.0, 0.0]);
        let gray = rgb_to_grayscale(&img);
        assert_eq!(gray.channels, 1);
        assert!((gray.get(0, 0, 0) - 0.299).abs() < 1e-3);
    }

    #[test]
    fn test_letterbox_preserves_aspect() {
        let img = make_rgb(100, 200, 0.5);
        let lb = letterbox(&img, 300, 300, 0.0);
        assert_eq!(lb.height, 300);
        assert_eq!(lb.width, 300);
        // Padded regions should be 0.0 (fill)
        assert_eq!(lb.get(0, 0, 0), 0.0);
    }

    #[test]
    fn test_brightness() {
        let img = make_rgb(2, 2, 0.5);
        let bright = brightness(&img, 2.0);
        assert!((bright.get(0, 0, 0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_contrast() {
        let img = make_rgb(1, 1, 0.5);
        let c = contrast(&img, 2.0);
        // (0.5 - 0.5)*2.0 + 0.5 = 0.5
        assert!((c.get(0, 0, 0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_pipeline_chain() {
        let pipe = VisionPipeline::new()
            .add(Transform::Resize { height: 256, width: 256 })
            .add(Transform::CenterCrop { height: 224, width: 224 })
            .add(Transform::Normalize {
                mean: vec![0.485, 0.456, 0.406],
                std: vec![0.229, 0.224, 0.225],
            })
            .add(Transform::HwcToChw);

        assert_eq!(pipe.len(), 4);

        let img = make_rgb(480, 640, 0.5);
        let out = pipe.run(&img);
        assert_eq!(out.height, 224);
        assert_eq!(out.width, 224);
        assert!(out.is_chw);
    }

    #[test]
    fn test_pipeline_empty() {
        let pipe = VisionPipeline::new();
        assert!(pipe.is_empty());
        let img = make_rgb(2, 2, 1.0);
        let out = pipe.run(&img);
        assert_eq!(out, img);
    }
}
