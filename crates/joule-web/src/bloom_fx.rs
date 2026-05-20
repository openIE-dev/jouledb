// bloom_fx.rs — HDR bloom post-processing for the rendering pipeline.
//
// Implements threshold extraction, downsample/upsample chains, separable
// Gaussian blur, soft knee curve, and energy-preserving bloom compositing.

use std::fmt;

/// Pixel with HDR color channels.
#[derive(Clone, Debug, PartialEq)]
pub struct HdrPixel {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl HdrPixel {
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }

    pub fn black() -> Self {
        Self { r: 0.0, g: 0.0, b: 0.0 }
    }

    pub fn luminance(&self) -> f64 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }

    pub fn scale(&self, s: f64) -> Self {
        Self { r: self.r * s, g: self.g * s, b: self.b * s }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
        }
    }

    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
        }
    }
}

/// 2D buffer of HDR pixels.
#[derive(Clone, Debug)]
pub struct HdrBuffer {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<HdrPixel>,
}

impl HdrBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![HdrPixel::black(); width * height],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> &HdrPixel {
        &self.pixels[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, px: HdrPixel) {
        self.pixels[y * self.width + x] = px;
    }

    /// Sample with clamp-to-edge for out-of-bounds coordinates.
    pub fn sample_clamp(&self, x: isize, y: isize) -> &HdrPixel {
        let cx = x.max(0).min(self.width as isize - 1) as usize;
        let cy = y.max(0).min(self.height as isize - 1) as usize;
        self.get(cx, cy)
    }

    /// Bilinear sample at fractional coordinates.
    pub fn sample_bilinear(&self, u: f64, v: f64) -> HdrPixel {
        let fx = u * (self.width as f64) - 0.5;
        let fy = v * (self.height as f64) - 0.5;
        let x0 = fx.floor() as isize;
        let y0 = fy.floor() as isize;
        let frac_x = fx - fx.floor();
        let frac_y = fy - fy.floor();

        let p00 = self.sample_clamp(x0, y0);
        let p10 = self.sample_clamp(x0 + 1, y0);
        let p01 = self.sample_clamp(x0, y0 + 1);
        let p11 = self.sample_clamp(x0 + 1, y0 + 1);

        let top = p00.lerp(p10, frac_x);
        let bot = p01.lerp(p11, frac_x);
        top.lerp(&bot, frac_y)
    }
}

/// Bloom configuration.
#[derive(Clone, Debug)]
pub struct BloomConfig {
    /// Brightness threshold for bloom extraction (default 1.0).
    pub threshold: f64,
    /// Soft knee width for smooth threshold curve (0 = hard, 1 = very soft).
    pub knee: f64,
    /// Number of downsample levels (typically 5-6).
    pub levels: usize,
    /// Overall bloom intensity multiplier.
    pub intensity: f64,
    /// Bloom radius multiplier (scales the blur kernel).
    pub radius: f64,
    /// Whether to apply energy preservation normalization.
    pub energy_preserving: bool,
}

impl Default for BloomConfig {
    fn default() -> Self {
        Self {
            threshold: 1.0,
            knee: 0.5,
            levels: 5,
            intensity: 1.0,
            radius: 1.0,
            energy_preserving: true,
        }
    }
}

impl fmt::Display for BloomConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BloomConfig(threshold={:.2}, knee={:.2}, levels={}, intensity={:.2}, radius={:.2})",
            self.threshold, self.knee, self.levels, self.intensity, self.radius
        )
    }
}

/// Compute soft threshold (knee curve) contribution for a given luminance.
/// Returns the brightness factor after soft thresholding.
pub fn soft_threshold(luminance: f64, threshold: f64, knee: f64) -> f64 {
    if knee <= 0.0 {
        // Hard threshold
        if luminance > threshold { luminance - threshold } else { 0.0 }
    } else {
        let half_knee = threshold * knee;
        let low = threshold - half_knee;
        let high = threshold + half_knee;
        if luminance <= low {
            0.0
        } else if luminance >= high {
            luminance - threshold
        } else {
            // Quadratic ease in the knee region
            let t = (luminance - low) / (high - low);
            t * t * (high - low)
        }
    }
}

/// Extract bright pixels above the configured threshold using soft knee.
pub fn threshold_pass(src: &HdrBuffer, config: &BloomConfig) -> HdrBuffer {
    let mut out = HdrBuffer::new(src.width, src.height);
    for y in 0..src.height {
        for x in 0..src.width {
            let px = src.get(x, y);
            let lum = px.luminance();
            let contribution = soft_threshold(lum, config.threshold, config.knee);
            if contribution > 0.0 && lum > 1e-10 {
                let factor = contribution / lum;
                out.set(x, y, px.scale(factor));
            }
        }
    }
    out
}

/// Downsample a buffer to half resolution using a 2x2 box filter.
pub fn downsample(src: &HdrBuffer) -> HdrBuffer {
    let w = (src.width / 2).max(1);
    let h = (src.height / 2).max(1);
    let mut out = HdrBuffer::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let sx = x * 2;
            let sy = y * 2;
            let mut acc = src.get(sx.min(src.width - 1), sy.min(src.height - 1)).clone();
            let mut count = 1.0_f64;
            if sx + 1 < src.width {
                acc = acc.add(src.get(sx + 1, sy.min(src.height - 1)));
                count += 1.0;
            }
            if sy + 1 < src.height {
                acc = acc.add(src.get(sx.min(src.width - 1), sy + 1));
                count += 1.0;
            }
            if sx + 1 < src.width && sy + 1 < src.height {
                acc = acc.add(src.get(sx + 1, sy + 1));
                count += 1.0;
            }
            out.set(x, y, acc.scale(1.0 / count));
        }
    }
    out
}

/// Build the full downsample chain from a source buffer.
pub fn build_downsample_chain(src: &HdrBuffer, levels: usize) -> Vec<HdrBuffer> {
    let mut chain = Vec::with_capacity(levels);
    let mut current = src.clone();
    for _ in 0..levels {
        let down = downsample(&current);
        chain.push(down.clone());
        current = down;
        if current.width <= 1 && current.height <= 1 {
            break;
        }
    }
    chain
}

/// Compute 1D Gaussian kernel weights for a given radius.
pub fn gaussian_kernel(radius: f64) -> Vec<f64> {
    let size = (radius * 2.0).ceil() as usize;
    let half = size.max(1);
    let sigma = radius / 2.0;
    let sigma2 = 2.0 * sigma * sigma;
    let mut weights = Vec::with_capacity(2 * half + 1);
    let mut sum = 0.0_f64;
    for i in 0..=(2 * half) {
        let d = i as f64 - half as f64;
        let w = (-d * d / sigma2.max(1e-10)).exp();
        weights.push(w);
        sum += w;
    }
    // Normalize
    if sum > 1e-10 {
        for w in &mut weights {
            *w /= sum;
        }
    }
    weights
}

/// Horizontal Gaussian blur pass.
pub fn blur_horizontal(src: &HdrBuffer, kernel: &[f64]) -> HdrBuffer {
    let half = kernel.len() / 2;
    let mut out = HdrBuffer::new(src.width, src.height);
    for y in 0..src.height {
        for x in 0..src.width {
            let mut acc = HdrPixel::black();
            for (i, &w) in kernel.iter().enumerate() {
                let sx = x as isize + i as isize - half as isize;
                let px = src.sample_clamp(sx, y as isize);
                acc = acc.add(&px.scale(w));
            }
            out.set(x, y, acc);
        }
    }
    out
}

/// Vertical Gaussian blur pass.
pub fn blur_vertical(src: &HdrBuffer, kernel: &[f64]) -> HdrBuffer {
    let half = kernel.len() / 2;
    let mut out = HdrBuffer::new(src.width, src.height);
    for y in 0..src.height {
        for x in 0..src.width {
            let mut acc = HdrPixel::black();
            for (i, &w) in kernel.iter().enumerate() {
                let sy = y as isize + i as isize - half as isize;
                let px = src.sample_clamp(x as isize, sy);
                acc = acc.add(&px.scale(w));
            }
            out.set(x, y, acc);
        }
    }
    out
}

/// Full separable Gaussian blur (horizontal + vertical).
pub fn gaussian_blur(src: &HdrBuffer, radius: f64) -> HdrBuffer {
    let kernel = gaussian_kernel(radius);
    let h_pass = blur_horizontal(src, &kernel);
    blur_vertical(&h_pass, &kernel)
}

/// Upsample and additively blend with target-sized buffer.
pub fn upsample_and_blend(low_res: &HdrBuffer, target: &HdrBuffer, blend: f64) -> HdrBuffer {
    let mut out = HdrBuffer::new(target.width, target.height);
    for y in 0..target.height {
        for x in 0..target.width {
            let u = (x as f64 + 0.5) / target.width as f64;
            let v = (y as f64 + 0.5) / target.height as f64;
            let upsampled = low_res.sample_bilinear(u, v);
            let original = target.get(x, y);
            out.set(x, y, original.add(&upsampled.scale(blend)));
        }
    }
    out
}

/// Compute total energy of a buffer (sum of luminances).
pub fn buffer_energy(buf: &HdrBuffer) -> f64 {
    buf.pixels.iter().map(|p| p.luminance()).sum()
}

/// Full bloom pipeline: threshold -> downsample chain -> blur each level ->
/// upsample chain -> composite with original.
pub fn apply_bloom(src: &HdrBuffer, config: &BloomConfig) -> HdrBuffer {
    if config.levels == 0 || config.intensity <= 0.0 {
        return src.clone();
    }

    // 1. Threshold pass
    let bright = threshold_pass(src, config);
    let bright_energy = if config.energy_preserving {
        buffer_energy(&bright)
    } else {
        0.0
    };

    // 2. Build downsample chain
    let mut chain = build_downsample_chain(&bright, config.levels);

    // 3. Blur each level
    let base_radius = 3.0 * config.radius;
    for (i, buf) in chain.iter_mut().enumerate() {
        let r = base_radius * (1.0 + i as f64 * 0.5);
        *buf = gaussian_blur(buf, r);
    }

    // 4. Upsample chain (bottom-up blending)
    let mut bloom_result = chain.last().unwrap().clone();
    for i in (0..chain.len().saturating_sub(1)).rev() {
        bloom_result = upsample_and_blend(&bloom_result, &chain[i], 1.0);
    }

    // Upsample to original resolution
    let mut final_bloom = HdrBuffer::new(src.width, src.height);
    for y in 0..src.height {
        for x in 0..src.width {
            let u = (x as f64 + 0.5) / src.width as f64;
            let v = (y as f64 + 0.5) / src.height as f64;
            final_bloom.set(x, y, bloom_result.sample_bilinear(u, v));
        }
    }

    // 5. Energy preservation normalization
    let mut bloom_intensity = config.intensity;
    if config.energy_preserving && bright_energy > 1e-10 {
        let bloom_energy = buffer_energy(&final_bloom);
        if bloom_energy > 1e-10 {
            bloom_intensity *= bright_energy / bloom_energy;
        }
    }

    // 6. Composite
    let mut out = HdrBuffer::new(src.width, src.height);
    for y in 0..src.height {
        for x in 0..src.width {
            let orig = src.get(x, y);
            let bloom = final_bloom.get(x, y);
            out.set(x, y, orig.add(&bloom.scale(bloom_intensity)));
        }
    }
    out
}

/// Create a test pattern with a single bright spot in the center.
pub fn create_bright_spot_buffer(width: usize, height: usize, brightness: f64) -> HdrBuffer {
    let mut buf = HdrBuffer::new(width, height);
    let cx = width / 2;
    let cy = height / 2;
    buf.set(cx, cy, HdrPixel::new(brightness, brightness, brightness));
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_hdr_pixel_luminance() {
        let px = HdrPixel::new(1.0, 1.0, 1.0);
        assert!(approx_eq(px.luminance(), 1.0, 1e-6));
    }

    #[test]
    fn test_hdr_pixel_black_luminance() {
        let px = HdrPixel::black();
        assert!(approx_eq(px.luminance(), 0.0, 1e-10));
    }

    #[test]
    fn test_hdr_pixel_scale() {
        let px = HdrPixel::new(2.0, 3.0, 4.0).scale(0.5);
        assert!(approx_eq(px.r, 1.0, 1e-6));
        assert!(approx_eq(px.g, 1.5, 1e-6));
        assert!(approx_eq(px.b, 2.0, 1e-6));
    }

    #[test]
    fn test_hdr_pixel_add() {
        let a = HdrPixel::new(1.0, 2.0, 3.0);
        let b = HdrPixel::new(0.5, 0.5, 0.5);
        let c = a.add(&b);
        assert!(approx_eq(c.r, 1.5, 1e-6));
        assert!(approx_eq(c.g, 2.5, 1e-6));
        assert!(approx_eq(c.b, 3.5, 1e-6));
    }

    #[test]
    fn test_hdr_pixel_lerp() {
        let a = HdrPixel::new(0.0, 0.0, 0.0);
        let b = HdrPixel::new(1.0, 1.0, 1.0);
        let mid = a.lerp(&b, 0.5);
        assert!(approx_eq(mid.r, 0.5, 1e-6));
    }

    #[test]
    fn test_buffer_creation_and_access() {
        let mut buf = HdrBuffer::new(4, 4);
        buf.set(2, 3, HdrPixel::new(1.0, 2.0, 3.0));
        let px = buf.get(2, 3);
        assert!(approx_eq(px.r, 1.0, 1e-6));
        assert!(approx_eq(px.g, 2.0, 1e-6));
    }

    #[test]
    fn test_sample_clamp_in_bounds() {
        let mut buf = HdrBuffer::new(4, 4);
        buf.set(1, 1, HdrPixel::new(5.0, 5.0, 5.0));
        let px = buf.sample_clamp(1, 1);
        assert!(approx_eq(px.r, 5.0, 1e-6));
    }

    #[test]
    fn test_sample_clamp_negative() {
        let mut buf = HdrBuffer::new(4, 4);
        buf.set(0, 0, HdrPixel::new(1.0, 2.0, 3.0));
        let px = buf.sample_clamp(-5, -3);
        assert!(approx_eq(px.r, 1.0, 1e-6));
    }

    #[test]
    fn test_soft_threshold_hard() {
        assert!(approx_eq(soft_threshold(0.5, 1.0, 0.0), 0.0, 1e-6));
        assert!(approx_eq(soft_threshold(1.5, 1.0, 0.0), 0.5, 1e-6));
    }

    #[test]
    fn test_soft_threshold_knee() {
        let val = soft_threshold(1.0, 1.0, 0.5);
        // At threshold exactly, with knee, should be in the quadratic region
        assert!(val >= 0.0);
        assert!(val < 0.5);
    }

    #[test]
    fn test_soft_threshold_below_knee() {
        assert!(approx_eq(soft_threshold(0.3, 1.0, 0.5), 0.0, 1e-6));
    }

    #[test]
    fn test_threshold_pass_black_image() {
        let buf = HdrBuffer::new(8, 8);
        let config = BloomConfig::default();
        let result = threshold_pass(&buf, &config);
        for px in &result.pixels {
            assert!(approx_eq(px.luminance(), 0.0, 1e-10));
        }
    }

    #[test]
    fn test_threshold_pass_bright_pixel() {
        let mut buf = HdrBuffer::new(4, 4);
        buf.set(2, 2, HdrPixel::new(5.0, 5.0, 5.0));
        let config = BloomConfig { threshold: 1.0, knee: 0.0, ..Default::default() };
        let result = threshold_pass(&buf, &config);
        let px = result.get(2, 2);
        assert!(px.luminance() > 0.0);
    }

    #[test]
    fn test_downsample_halves_resolution() {
        let buf = HdrBuffer::new(16, 16);
        let down = downsample(&buf);
        assert_eq!(down.width, 8);
        assert_eq!(down.height, 8);
    }

    #[test]
    fn test_downsample_averages() {
        let mut buf = HdrBuffer::new(4, 4);
        // Set a 2x2 block to known values
        buf.set(0, 0, HdrPixel::new(1.0, 0.0, 0.0));
        buf.set(1, 0, HdrPixel::new(3.0, 0.0, 0.0));
        buf.set(0, 1, HdrPixel::new(5.0, 0.0, 0.0));
        buf.set(1, 1, HdrPixel::new(7.0, 0.0, 0.0));
        let down = downsample(&buf);
        let px = down.get(0, 0);
        assert!(approx_eq(px.r, 4.0, 1e-6)); // (1+3+5+7)/4
    }

    #[test]
    fn test_downsample_chain_length() {
        let buf = HdrBuffer::new(64, 64);
        let chain = build_downsample_chain(&buf, 5);
        assert_eq!(chain.len(), 5);
        assert_eq!(chain[0].width, 32);
        assert_eq!(chain[1].width, 16);
    }

    #[test]
    fn test_gaussian_kernel_sums_to_one() {
        let k = gaussian_kernel(3.0);
        let sum: f64 = k.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-6));
    }

    #[test]
    fn test_gaussian_kernel_symmetric() {
        let k = gaussian_kernel(4.0);
        let n = k.len();
        for i in 0..n / 2 {
            assert!(approx_eq(k[i], k[n - 1 - i], 1e-10));
        }
    }

    #[test]
    fn test_gaussian_blur_preserves_uniform() {
        let mut buf = HdrBuffer::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                buf.set(x, y, HdrPixel::new(2.0, 2.0, 2.0));
            }
        }
        let blurred = gaussian_blur(&buf, 2.0);
        // Interior pixels should remain close to 2.0
        let px = blurred.get(4, 4);
        assert!(approx_eq(px.r, 2.0, 0.1));
    }

    #[test]
    fn test_upsample_and_blend() {
        let low = HdrBuffer::new(2, 2);
        let mut target = HdrBuffer::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                target.set(x, y, HdrPixel::new(1.0, 1.0, 1.0));
            }
        }
        let result = upsample_and_blend(&low, &target, 1.0);
        assert_eq!(result.width, 4);
        assert_eq!(result.height, 4);
        // Low res is black so result should be close to target
        let px = result.get(2, 2);
        assert!(approx_eq(px.r, 1.0, 0.1));
    }

    #[test]
    fn test_buffer_energy() {
        let mut buf = HdrBuffer::new(2, 2);
        buf.set(0, 0, HdrPixel::new(1.0, 1.0, 1.0));
        buf.set(1, 1, HdrPixel::new(1.0, 1.0, 1.0));
        let e = buffer_energy(&buf);
        assert!(approx_eq(e, 2.0, 1e-6));
    }

    #[test]
    fn test_apply_bloom_zero_intensity() {
        let buf = create_bright_spot_buffer(8, 8, 5.0);
        let config = BloomConfig { intensity: 0.0, ..Default::default() };
        let result = apply_bloom(&buf, &config);
        // With zero intensity, output equals input
        let orig = buf.get(4, 4);
        let out = result.get(4, 4);
        assert!(approx_eq(orig.r, out.r, 1e-6));
    }

    #[test]
    fn test_apply_bloom_adds_glow() {
        let buf = create_bright_spot_buffer(16, 16, 10.0);
        let config = BloomConfig {
            threshold: 1.0,
            knee: 0.0,
            levels: 3,
            intensity: 1.0,
            radius: 1.0,
            energy_preserving: false,
        };
        let result = apply_bloom(&buf, &config);
        // Neighbors of the bright spot should now have some glow
        let neighbor = result.get(7, 8);
        assert!(neighbor.luminance() > 0.0);
    }

    #[test]
    fn test_bloom_config_display() {
        let config = BloomConfig::default();
        let s = format!("{}", config);
        assert!(s.contains("BloomConfig"));
        assert!(s.contains("threshold"));
    }

    #[test]
    fn test_bilinear_sample_center() {
        let mut buf = HdrBuffer::new(2, 2);
        buf.set(0, 0, HdrPixel::new(1.0, 0.0, 0.0));
        buf.set(1, 0, HdrPixel::new(1.0, 0.0, 0.0));
        buf.set(0, 1, HdrPixel::new(1.0, 0.0, 0.0));
        buf.set(1, 1, HdrPixel::new(1.0, 0.0, 0.0));
        let px = buf.sample_bilinear(0.5, 0.5);
        assert!(approx_eq(px.r, 1.0, 0.1));
    }

    #[test]
    fn test_energy_preserving_bloom() {
        let buf = create_bright_spot_buffer(16, 16, 10.0);
        let config = BloomConfig {
            threshold: 1.0,
            knee: 0.0,
            levels: 3,
            intensity: 1.0,
            radius: 1.0,
            energy_preserving: true,
        };
        let result = apply_bloom(&buf, &config);
        // The result should have roughly the original pixel energy plus bloom contribution.
        let center = result.get(8, 8);
        assert!(center.luminance() >= 10.0 - 1.0);
    }
}
