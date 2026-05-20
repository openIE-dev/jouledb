//! Data Augmentation — random crop, flip, rotate, color jitter, mixup,
//! cutout, and elastic deformation for ML training pipelines.
//!
//! Pure Rust, std-only. All transforms operate on flat 2D grids of f64.

use std::fmt;

// ── PRNG Helper ─────────────────────────────────────────────────

/// Simple deterministic LCG-based PRNG for augmentation reproducibility.
#[derive(Debug, Clone)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    /// Uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform f64 in [lo, hi).
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_f64() * (hi - lo)
    }

    /// Approximate Gaussian via Box-Muller.
    fn gaussian(&mut self, mean: f64, std: f64) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + z * std
    }
}

// ── Image2D ─────────────────────────────────────────────────────

/// A simple 2D grid of f64 values (single channel).
#[derive(Debug, Clone, PartialEq)]
pub struct Image2D {
    pub data: Vec<f64>,
    pub width: usize,
    pub height: usize,
}

impl Image2D {
    pub fn new(width: usize, height: usize) -> Self {
        Self { data: vec![0.0; width * height], width, height }
    }

    pub fn from_data(data: Vec<f64>, width: usize, height: usize) -> Self {
        assert_eq!(data.len(), width * height, "data length mismatch");
        Self { data, width, height }
    }

    pub fn get(&self, x: usize, y: usize) -> f64 {
        self.data[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, val: f64) {
        self.data[y * self.width + x] = val;
    }

    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }

    /// Fill with a gradient for testing.
    pub fn gradient(width: usize, height: usize) -> Self {
        let mut img = Self::new(width, height);
        for y in 0..height {
            for x in 0..width {
                img.set(x, y, (x + y) as f64 / (width + height) as f64);
            }
        }
        img
    }
}

impl fmt::Display for Image2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Image2D({}x{})", self.width, self.height)
    }
}

// ── Transform Enum ──────────────────────────────────────────────

/// Available augmentation transforms.
#[derive(Debug, Clone)]
pub enum Transform {
    RandomCrop { crop_w: usize, crop_h: usize },
    HorizontalFlip,
    VerticalFlip,
    Rotate90,
    Rotate180,
    Rotate270,
    ColorJitter { brightness: f64, contrast: f64 },
    Cutout { size: usize },
    ElasticDeform { alpha: f64, sigma: f64 },
}

impl fmt::Display for Transform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RandomCrop { crop_w, crop_h } => write!(f, "RandomCrop({}x{})", crop_w, crop_h),
            Self::HorizontalFlip => write!(f, "HFlip"),
            Self::VerticalFlip => write!(f, "VFlip"),
            Self::Rotate90 => write!(f, "Rot90"),
            Self::Rotate180 => write!(f, "Rot180"),
            Self::Rotate270 => write!(f, "Rot270"),
            Self::ColorJitter { brightness, contrast } =>
                write!(f, "Jitter(b={:.2},c={:.2})", brightness, contrast),
            Self::Cutout { size } => write!(f, "Cutout({})", size),
            Self::ElasticDeform { alpha, sigma } =>
                write!(f, "Elastic(a={:.1},s={:.1})", alpha, sigma),
        }
    }
}

// ── Augmentor ───────────────────────────────────────────────────

/// Applies a pipeline of transforms to images.
#[derive(Debug, Clone)]
pub struct Augmentor {
    transforms: Vec<Transform>,
    probability: f64,
    seed: u64,
}

impl Augmentor {
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
            probability: 1.0,
            seed: 42,
        }
    }

    pub fn with_transform(mut self, t: Transform) -> Self {
        self.transforms.push(t);
        self
    }

    pub fn with_probability(mut self, p: f64) -> Self {
        self.probability = p.clamp(0.0, 1.0);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Apply all transforms in sequence.
    pub fn apply(&self, img: &Image2D) -> Image2D {
        let mut rng = Rng::new(self.seed);
        let mut result = img.clone();
        for t in &self.transforms {
            if rng.next_f64() > self.probability {
                continue;
            }
            result = apply_transform(&result, t, &mut rng);
        }
        result
    }

    pub fn num_transforms(&self) -> usize {
        self.transforms.len()
    }
}

impl fmt::Display for Augmentor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Augmentor(n={}, p={:.2})", self.transforms.len(), self.probability)
    }
}

// ── Transform Implementations ───────────────────────────────────

fn apply_transform(img: &Image2D, t: &Transform, rng: &mut Rng) -> Image2D {
    match t {
        Transform::RandomCrop { crop_w, crop_h } => random_crop(img, *crop_w, *crop_h, rng),
        Transform::HorizontalFlip => horizontal_flip(img),
        Transform::VerticalFlip => vertical_flip(img),
        Transform::Rotate90 => rotate_90(img),
        Transform::Rotate180 => rotate_180(img),
        Transform::Rotate270 => rotate_270(img),
        Transform::ColorJitter { brightness, contrast } =>
            color_jitter(img, *brightness, *contrast, rng),
        Transform::Cutout { size } => cutout(img, *size, rng),
        Transform::ElasticDeform { alpha, sigma } => elastic_deform(img, *alpha, *sigma, rng),
    }
}

fn random_crop(img: &Image2D, crop_w: usize, crop_h: usize, rng: &mut Rng) -> Image2D {
    let cw = crop_w.min(img.width);
    let ch = crop_h.min(img.height);
    let x0 = if img.width > cw {
        (rng.next_u64() as usize) % (img.width - cw)
    } else {
        0
    };
    let y0 = if img.height > ch {
        (rng.next_u64() as usize) % (img.height - ch)
    } else {
        0
    };
    let mut out = Image2D::new(cw, ch);
    for y in 0..ch {
        for x in 0..cw {
            out.set(x, y, img.get(x0 + x, y0 + y));
        }
    }
    out
}

fn horizontal_flip(img: &Image2D) -> Image2D {
    let mut out = Image2D::new(img.width, img.height);
    for y in 0..img.height {
        for x in 0..img.width {
            out.set(img.width - 1 - x, y, img.get(x, y));
        }
    }
    out
}

fn vertical_flip(img: &Image2D) -> Image2D {
    let mut out = Image2D::new(img.width, img.height);
    for y in 0..img.height {
        for x in 0..img.width {
            out.set(x, img.height - 1 - y, img.get(x, y));
        }
    }
    out
}

fn rotate_90(img: &Image2D) -> Image2D {
    let mut out = Image2D::new(img.height, img.width);
    for y in 0..img.height {
        for x in 0..img.width {
            out.set(img.height - 1 - y, x, img.get(x, y));
        }
    }
    out
}

fn rotate_180(img: &Image2D) -> Image2D {
    let mut out = Image2D::new(img.width, img.height);
    for y in 0..img.height {
        for x in 0..img.width {
            out.set(img.width - 1 - x, img.height - 1 - y, img.get(x, y));
        }
    }
    out
}

fn rotate_270(img: &Image2D) -> Image2D {
    let mut out = Image2D::new(img.height, img.width);
    for y in 0..img.height {
        for x in 0..img.width {
            out.set(y, img.width - 1 - x, img.get(x, y));
        }
    }
    out
}

fn color_jitter(img: &Image2D, brightness: f64, contrast: f64, rng: &mut Rng) -> Image2D {
    let b_offset = rng.uniform(-brightness, brightness);
    let c_factor = 1.0 + rng.uniform(-contrast, contrast);
    let mean: f64 = img.data.iter().sum::<f64>() / img.data.len() as f64;
    let mut out = img.clone();
    for v in out.data.iter_mut() {
        *v = (*v - mean) * c_factor + mean + b_offset;
    }
    out
}

fn cutout(img: &Image2D, size: usize, rng: &mut Rng) -> Image2D {
    let mut out = img.clone();
    let cx = if img.width > 0 { (rng.next_u64() as usize) % img.width } else { 0 };
    let cy = if img.height > 0 { (rng.next_u64() as usize) % img.height } else { 0 };
    let half = size / 2;
    let x0 = cx.saturating_sub(half);
    let y0 = cy.saturating_sub(half);
    let x1 = (cx + half).min(img.width);
    let y1 = (cy + half).min(img.height);
    for y in y0..y1 {
        for x in x0..x1 {
            out.set(x, y, 0.0);
        }
    }
    out
}

fn elastic_deform(img: &Image2D, alpha: f64, sigma: f64, rng: &mut Rng) -> Image2D {
    let w = img.width;
    let h = img.height;
    let n = w * h;
    // Generate random displacement fields
    let mut dx = vec![0.0f64; n];
    let mut dy = vec![0.0f64; n];
    for i in 0..n {
        dx[i] = rng.uniform(-1.0, 1.0);
        dy[i] = rng.uniform(-1.0, 1.0);
    }
    // Approximate Gaussian blur via repeated box-blur (3 passes)
    let kernel = (sigma * 3.0).ceil() as usize | 1;
    for _ in 0..3 {
        dx = box_blur_1d(&dx, w, h, kernel);
        dy = box_blur_1d(&dy, w, h, kernel);
    }
    // Scale by alpha
    for v in dx.iter_mut() {
        *v *= alpha;
    }
    for v in dy.iter_mut() {
        *v *= alpha;
    }
    // Apply displacement with bilinear interpolation
    let mut out = Image2D::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let sx = x as f64 + dx[idx];
            let sy = y as f64 + dy[idx];
            out.set(x, y, bilinear_sample(img, sx, sy));
        }
    }
    out
}

fn box_blur_1d(data: &[f64], w: usize, h: usize, kernel: usize) -> Vec<f64> {
    let half = kernel / 2;
    let mut tmp = vec![0.0f64; data.len()];
    // Horizontal pass
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            let mut count = 0;
            let lo = x.saturating_sub(half);
            let hi = (x + half + 1).min(w);
            for kx in lo..hi {
                sum += data[y * w + kx];
                count += 1;
            }
            tmp[y * w + x] = sum / count as f64;
        }
    }
    // Vertical pass
    let mut out = vec![0.0f64; data.len()];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            let mut count = 0;
            let lo = y.saturating_sub(half);
            let hi = (y + half + 1).min(h);
            for ky in lo..hi {
                sum += tmp[ky * w + x];
                count += 1;
            }
            out[y * w + x] = sum / count as f64;
        }
    }
    out
}

fn bilinear_sample(img: &Image2D, sx: f64, sy: f64) -> f64 {
    let x0 = sx.floor().max(0.0) as usize;
    let y0 = sy.floor().max(0.0) as usize;
    let x1 = (x0 + 1).min(img.width - 1);
    let y1 = (y0 + 1).min(img.height - 1);
    let fx = sx - sx.floor();
    let fy = sy - sy.floor();
    let v00 = img.get(x0.min(img.width - 1), y0.min(img.height - 1));
    let v10 = img.get(x1, y0.min(img.height - 1));
    let v01 = img.get(x0.min(img.width - 1), y1);
    let v11 = img.get(x1, y1);
    v00 * (1.0 - fx) * (1.0 - fy) + v10 * fx * (1.0 - fy) + v01 * (1.0 - fx) * fy + v11 * fx * fy
}

// ── Mixup ───────────────────────────────────────────────────────

/// Mixup: linearly interpolate two samples and their labels.
pub fn mixup(a: &[f64], b: &[f64], lambda: f64) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| lambda * x + (1.0 - lambda) * y).collect()
}

/// Generate a mixup lambda from a Beta(alpha, alpha) distribution approximation.
pub fn mixup_lambda(alpha: f64, seed: u64) -> f64 {
    if alpha <= 0.0 {
        return 1.0;
    }
    // Approximate Beta via Joehnk's method
    let mut rng = Rng::new(seed);
    for _ in 0..100 {
        let u1 = rng.next_f64().max(1e-15);
        let u2 = rng.next_f64().max(1e-15);
        let x = u1.powf(1.0 / alpha);
        let y = u2.powf(1.0 / alpha);
        if x + y <= 1.0 {
            return x / (x + y);
        }
    }
    0.5
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_img() -> Image2D {
        Image2D::gradient(8, 8)
    }

    #[test]
    fn image_creation() {
        let img = Image2D::new(10, 10);
        assert_eq!(img.pixel_count(), 100);
        assert_eq!(img.get(0, 0), 0.0);
    }

    #[test]
    fn image_gradient() {
        let img = Image2D::gradient(4, 4);
        assert!(img.get(3, 3) > img.get(0, 0));
    }

    #[test]
    fn image_display() {
        let img = Image2D::new(5, 3);
        assert_eq!(format!("{}", img), "Image2D(5x3)");
    }

    #[test]
    fn horizontal_flip_involution() {
        let img = test_img();
        let flipped = horizontal_flip(&horizontal_flip(&img));
        assert_eq!(img.data, flipped.data);
    }

    #[test]
    fn vertical_flip_involution() {
        let img = test_img();
        let flipped = vertical_flip(&vertical_flip(&img));
        assert_eq!(img.data, flipped.data);
    }

    #[test]
    fn rotate_360() {
        let img = test_img();
        let r = rotate_90(&rotate_90(&rotate_90(&rotate_90(&img))));
        assert_eq!(img.data, r.data);
    }

    #[test]
    fn rotate_180_matches() {
        let img = test_img();
        let r1 = rotate_180(&img);
        let r2 = rotate_90(&rotate_90(&img));
        assert_eq!(r1.data, r2.data);
    }

    #[test]
    fn random_crop_size() {
        let img = test_img();
        let mut rng = Rng::new(42);
        let cropped = random_crop(&img, 4, 4, &mut rng);
        assert_eq!(cropped.width, 4);
        assert_eq!(cropped.height, 4);
    }

    #[test]
    fn cutout_zeroes_region() {
        let img = Image2D::from_data(vec![1.0; 64], 8, 8);
        let mut rng = Rng::new(42);
        let result = cutout(&img, 4, &mut rng);
        let zero_count = result.data.iter().filter(|&&v| v == 0.0).count();
        assert!(zero_count > 0);
    }

    #[test]
    fn color_jitter_changes_values() {
        let img = Image2D::gradient(8, 8);
        let mut rng = Rng::new(99);
        let jittered = color_jitter(&img, 0.2, 0.2, &mut rng);
        assert_ne!(img.data, jittered.data);
    }

    #[test]
    fn elastic_deform_preserves_size() {
        let img = test_img();
        let mut rng = Rng::new(7);
        let deformed = elastic_deform(&img, 5.0, 2.0, &mut rng);
        assert_eq!(deformed.width, img.width);
        assert_eq!(deformed.height, img.height);
    }

    #[test]
    fn mixup_interpolation() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let mixed = mixup(&a, &b, 0.5);
        assert!((mixed[0] - 0.5).abs() < 1e-9);
        assert!((mixed[1] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn mixup_lambda_range() {
        let lam = mixup_lambda(0.2, 42);
        assert!((0.0..=1.0).contains(&lam));
    }

    #[test]
    fn augmentor_pipeline() {
        let img = test_img();
        let aug = Augmentor::new()
            .with_transform(Transform::HorizontalFlip)
            .with_transform(Transform::ColorJitter { brightness: 0.1, contrast: 0.1 })
            .with_seed(42);
        let result = aug.apply(&img);
        assert_eq!(result.width, img.width);
    }

    #[test]
    fn augmentor_probability_zero() {
        let img = test_img();
        let aug = Augmentor::new()
            .with_transform(Transform::HorizontalFlip)
            .with_probability(0.0)
            .with_seed(42);
        let result = aug.apply(&img);
        assert_eq!(result.data, img.data);
    }

    #[test]
    fn augmentor_display() {
        let aug = Augmentor::new()
            .with_transform(Transform::Rotate90)
            .with_transform(Transform::Cutout { size: 4 });
        assert!(format!("{}", aug).contains("n=2"));
    }

    #[test]
    fn transform_display() {
        let t = Transform::ElasticDeform { alpha: 10.0, sigma: 3.0 };
        assert!(format!("{}", t).contains("Elastic"));
    }

    #[test]
    fn crop_larger_than_image() {
        let img = Image2D::gradient(4, 4);
        let mut rng = Rng::new(1);
        let cropped = random_crop(&img, 100, 100, &mut rng);
        assert_eq!(cropped.width, 4);
        assert_eq!(cropped.height, 4);
    }

    #[test]
    fn augmentor_deterministic() {
        let img = test_img();
        let aug = Augmentor::new()
            .with_transform(Transform::HorizontalFlip)
            .with_transform(Transform::ColorJitter { brightness: 0.3, contrast: 0.3 })
            .with_seed(77);
        let r1 = aug.apply(&img);
        let r2 = aug.apply(&img);
        assert_eq!(r1.data, r2.data);
    }
}
