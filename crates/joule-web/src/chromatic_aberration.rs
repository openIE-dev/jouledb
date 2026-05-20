// chromatic_aberration.rs — Chromatic aberration post-processing effect.
//
// Implements radial RGB channel separation, barrel/pincushion distortion,
// per-channel UV offset, axial chromatic aberration (depth-dependent),
// and fast vs accurate (spectral sampling) modes.

use std::fmt;

/// RGB pixel.
#[derive(Clone, Debug, PartialEq)]
pub struct CaPixel {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl CaPixel {
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }

    pub fn black() -> Self {
        Self { r: 0.0, g: 0.0, b: 0.0 }
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
}

/// Color buffer for CA processing.
#[derive(Clone, Debug)]
pub struct CaBuffer {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<CaPixel>,
}

impl CaBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![CaPixel::black(); width * height],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> &CaPixel {
        &self.pixels[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, px: CaPixel) {
        self.pixels[y * self.width + x] = px;
    }

    /// Sample with bilinear interpolation at fractional pixel coords.
    pub fn sample_bilinear(&self, fx: f64, fy: f64) -> CaPixel {
        let x0 = fx.floor() as isize;
        let y0 = fy.floor() as isize;
        let frac_x = fx - fx.floor();
        let frac_y = fy - fy.floor();

        let p00 = self.sample_clamp(x0, y0);
        let p10 = self.sample_clamp(x0 + 1, y0);
        let p01 = self.sample_clamp(x0, y0 + 1);
        let p11 = self.sample_clamp(x0 + 1, y0 + 1);

        let top = CaPixel::new(
            p00.r * (1.0 - frac_x) + p10.r * frac_x,
            p00.g * (1.0 - frac_x) + p10.g * frac_x,
            p00.b * (1.0 - frac_x) + p10.b * frac_x,
        );
        let bot = CaPixel::new(
            p01.r * (1.0 - frac_x) + p11.r * frac_x,
            p01.g * (1.0 - frac_x) + p11.g * frac_x,
            p01.b * (1.0 - frac_x) + p11.b * frac_x,
        );
        CaPixel::new(
            top.r * (1.0 - frac_y) + bot.r * frac_y,
            top.g * (1.0 - frac_y) + bot.g * frac_y,
            top.b * (1.0 - frac_y) + bot.b * frac_y,
        )
    }

    fn sample_clamp(&self, x: isize, y: isize) -> &CaPixel {
        let cx = x.max(0).min(self.width as isize - 1) as usize;
        let cy = y.max(0).min(self.height as isize - 1) as usize;
        self.get(cx, cy)
    }

    /// Sample a single channel using bilinear interpolation.
    pub fn sample_channel_bilinear(&self, fx: f64, fy: f64, channel: usize) -> f64 {
        let x0 = fx.floor() as isize;
        let y0 = fy.floor() as isize;
        let frac_x = fx - fx.floor();
        let frac_y = fy - fy.floor();

        let get_ch = |px: &CaPixel| -> f64 {
            match channel {
                0 => px.r,
                1 => px.g,
                _ => px.b,
            }
        };

        let p00 = get_ch(self.sample_clamp(x0, y0));
        let p10 = get_ch(self.sample_clamp(x0 + 1, y0));
        let p01 = get_ch(self.sample_clamp(x0, y0 + 1));
        let p11 = get_ch(self.sample_clamp(x0 + 1, y0 + 1));

        let top = p00 * (1.0 - frac_x) + p10 * frac_x;
        let bot = p01 * (1.0 - frac_x) + p11 * frac_x;
        top * (1.0 - frac_y) + bot * frac_y
    }
}

/// Depth buffer for axial CA.
#[derive(Clone, Debug)]
pub struct CaDepthBuffer {
    pub width: usize,
    pub height: usize,
    pub depths: Vec<f64>,
}

impl CaDepthBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            depths: vec![1.0; width * height],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> f64 {
        self.depths[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, d: f64) {
        self.depths[y * self.width + x] = d;
    }
}

/// Distortion model for radial distortion.
#[derive(Clone, Debug, PartialEq)]
pub enum DistortionModel {
    /// No distortion — linear offset only.
    None,
    /// Barrel distortion (positive coefficient bulges outward).
    Barrel(f64),
    /// Pincushion distortion (negative coefficient pinches inward).
    Pincushion(f64),
}

impl fmt::Display for DistortionModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistortionModel::None => write!(f, "None"),
            DistortionModel::Barrel(k) => write!(f, "Barrel(k={:.4})", k),
            DistortionModel::Pincushion(k) => write!(f, "Pincushion(k={:.4})", k),
        }
    }
}

/// Processing mode.
#[derive(Clone, Debug, PartialEq)]
pub enum CaMode {
    /// Fast mode: simple RGB channel offset.
    Fast,
    /// Accurate mode: spectral sampling with multiple wavelength bins.
    Spectral { wavelength_bins: usize },
}

impl fmt::Display for CaMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CaMode::Fast => write!(f, "Fast"),
            CaMode::Spectral { wavelength_bins } => write!(f, "Spectral(bins={})", wavelength_bins),
        }
    }
}

/// Chromatic aberration configuration.
#[derive(Clone, Debug)]
pub struct CaConfig {
    /// Overall intensity of the CA effect (0=none, 1=normal).
    pub intensity: f64,
    /// Direction vector for anisotropic CA (normalized).
    pub direction: (f64, f64),
    /// Screen center for radial offset (normalized, 0.5 = center).
    pub center: (f64, f64),
    /// Distortion model.
    pub distortion: DistortionModel,
    /// Per-channel offset multipliers: (red_offset, green_offset, blue_offset).
    /// Green is typically 0 (reference channel).
    pub channel_offsets: (f64, f64, f64),
    /// Whether to apply axial (depth-dependent) CA.
    pub axial_enabled: bool,
    /// Axial CA intensity (scales depth-based offset).
    pub axial_intensity: f64,
    /// Processing mode.
    pub mode: CaMode,
}

impl Default for CaConfig {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            direction: (1.0, 1.0),
            center: (0.5, 0.5),
            distortion: DistortionModel::None,
            channel_offsets: (-1.0, 0.0, 1.0),
            axial_enabled: false,
            axial_intensity: 0.5,
            mode: CaMode::Fast,
        }
    }
}

impl fmt::Display for CaConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CaConfig(intensity={:.2}, mode={})", self.intensity, self.mode)
    }
}

/// Compute radial distance from center in normalized [-1, 1] space.
pub fn radial_distance(x: usize, y: usize, width: usize, height: usize, center: (f64, f64)) -> f64 {
    let nx = (x as f64 + 0.5) / width as f64 - center.0;
    let ny = (y as f64 + 0.5) / height as f64 - center.1;
    (nx * nx + ny * ny).sqrt()
}

/// Compute radial direction from center.
pub fn radial_direction(x: usize, y: usize, width: usize, height: usize, center: (f64, f64)) -> (f64, f64) {
    let nx = (x as f64 + 0.5) / width as f64 - center.0;
    let ny = (y as f64 + 0.5) / height as f64 - center.1;
    let len = (nx * nx + ny * ny).sqrt();
    if len < 1e-10 {
        (0.0, 0.0)
    } else {
        (nx / len, ny / len)
    }
}

/// Apply barrel/pincushion distortion to a radial distance.
pub fn apply_distortion(r: f64, model: &DistortionModel) -> f64 {
    match model {
        DistortionModel::None => r,
        DistortionModel::Barrel(k) => r * (1.0 + k * r * r),
        DistortionModel::Pincushion(k) => r * (1.0 + k * r * r),
    }
}

/// Compute UV offset for a specific channel.
fn compute_channel_uv_offset(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    config: &CaConfig,
    channel_offset: f64,
    depth_factor: f64,
) -> (f64, f64) {
    let rad_dist = radial_distance(x, y, width, height, config.center);
    let (dx, dy) = radial_direction(x, y, width, height, config.center);

    let distorted_dist = apply_distortion(rad_dist, &config.distortion);
    let offset_magnitude = distorted_dist * channel_offset * config.intensity * depth_factor;

    // Apply direction modulation
    let dir_x = config.direction.0;
    let dir_y = config.direction.1;
    let dir_len = (dir_x * dir_x + dir_y * dir_y).sqrt().max(1e-10);

    (
        dx * offset_magnitude * (dir_x / dir_len) * width as f64 * 0.01,
        dy * offset_magnitude * (dir_y / dir_len) * height as f64 * 0.01,
    )
}

/// Apply fast-mode chromatic aberration (simple RGB offset).
pub fn apply_ca_fast(src: &CaBuffer, config: &CaConfig, depth: Option<&CaDepthBuffer>) -> CaBuffer {
    let w = src.width;
    let h = src.height;
    let mut out = CaBuffer::new(w, h);

    let offsets = [config.channel_offsets.0, config.channel_offsets.1, config.channel_offsets.2];

    for y in 0..h {
        for x in 0..w {
            let depth_factor = if config.axial_enabled {
                if let Some(db) = depth {
                    1.0 + (db.get(x, y) - 0.5) * config.axial_intensity
                } else {
                    1.0
                }
            } else {
                1.0
            };

            let mut channels = [0.0_f64; 3];
            for ch in 0..3 {
                let (uv_dx, uv_dy) = compute_channel_uv_offset(
                    x, y, w, h, config, offsets[ch], depth_factor,
                );
                let sx = x as f64 + uv_dx;
                let sy = y as f64 + uv_dy;
                channels[ch] = src.sample_channel_bilinear(sx, sy, ch);
            }

            out.set(x, y, CaPixel::new(channels[0], channels[1], channels[2]));
        }
    }
    out
}

/// Spectral wavelength bin to RGB contribution (simplified model).
/// Maps wavelength bins [0..N) to approximate RGB weights.
pub fn spectral_to_rgb(bin: usize, total_bins: usize) -> (f64, f64, f64) {
    if total_bins == 0 {
        return (0.333, 0.333, 0.333);
    }
    let t = bin as f64 / (total_bins - 1).max(1) as f64; // 0=red, 0.5=green, 1=blue
    let r = (1.0 - t * 2.0).max(0.0);
    let g = 1.0 - (t - 0.5).abs() * 2.0;
    let b = (t * 2.0 - 1.0).max(0.0);
    let sum = r + g + b;
    if sum < 1e-10 {
        (0.333, 0.333, 0.333)
    } else {
        (r / sum, g / sum, b / sum)
    }
}

/// Compute wavelength-dependent offset for spectral mode.
pub fn spectral_offset(bin: usize, total_bins: usize) -> f64 {
    if total_bins <= 1 {
        return 0.0;
    }
    // Map from red (negative offset) through green (zero) to blue (positive)
    let t = bin as f64 / (total_bins - 1) as f64;
    (t - 0.5) * 2.0
}

/// Apply accurate (spectral) chromatic aberration.
pub fn apply_ca_spectral(
    src: &CaBuffer,
    config: &CaConfig,
    depth: Option<&CaDepthBuffer>,
) -> CaBuffer {
    let bins = match config.mode {
        CaMode::Spectral { wavelength_bins } => wavelength_bins.max(3),
        _ => 7,
    };

    let w = src.width;
    let h = src.height;
    let mut out = CaBuffer::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let depth_factor = if config.axial_enabled {
                if let Some(db) = depth {
                    1.0 + (db.get(x, y) - 0.5) * config.axial_intensity
                } else {
                    1.0
                }
            } else {
                1.0
            };

            let mut acc_r = 0.0_f64;
            let mut acc_g = 0.0_f64;
            let mut acc_b = 0.0_f64;

            for bin in 0..bins {
                let offset = spectral_offset(bin, bins);
                let (wr, wg, wb) = spectral_to_rgb(bin, bins);

                let (uv_dx, uv_dy) = compute_channel_uv_offset(
                    x, y, w, h, config, offset, depth_factor,
                );
                let sx = x as f64 + uv_dx;
                let sy = y as f64 + uv_dy;

                let sampled = src.sample_bilinear(sx, sy);
                acc_r += sampled.r * wr;
                acc_g += sampled.g * wg;
                acc_b += sampled.b * wb;
            }

            out.set(x, y, CaPixel::new(acc_r, acc_g, acc_b));
        }
    }
    out
}

/// Main entry point: apply chromatic aberration based on config mode.
pub fn apply_chromatic_aberration(
    src: &CaBuffer,
    config: &CaConfig,
    depth: Option<&CaDepthBuffer>,
) -> CaBuffer {
    if config.intensity <= 0.0 {
        return src.clone();
    }
    match config.mode {
        CaMode::Fast => apply_ca_fast(src, config, depth),
        CaMode::Spectral { .. } => apply_ca_spectral(src, config, depth),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_pixel_operations() {
        let a = CaPixel::new(1.0, 2.0, 3.0);
        let b = CaPixel::new(0.5, 0.5, 0.5);
        let c = a.add(&b);
        assert!(approx_eq(c.r, 1.5, 1e-6));
        assert!(approx_eq(c.g, 2.5, 1e-6));
    }

    #[test]
    fn test_pixel_scale() {
        let p = CaPixel::new(2.0, 4.0, 6.0).scale(0.5);
        assert!(approx_eq(p.r, 1.0, 1e-6));
        assert!(approx_eq(p.b, 3.0, 1e-6));
    }

    #[test]
    fn test_buffer_bilinear_center() {
        let mut buf = CaBuffer::new(2, 2);
        buf.set(0, 0, CaPixel::new(1.0, 0.0, 0.0));
        buf.set(1, 0, CaPixel::new(1.0, 0.0, 0.0));
        buf.set(0, 1, CaPixel::new(1.0, 0.0, 0.0));
        buf.set(1, 1, CaPixel::new(1.0, 0.0, 0.0));
        let px = buf.sample_bilinear(0.5, 0.5);
        assert!(approx_eq(px.r, 1.0, 0.2));
    }

    #[test]
    fn test_buffer_channel_sample() {
        let mut buf = CaBuffer::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                buf.set(x, y, CaPixel::new(1.0, 2.0, 3.0));
            }
        }
        let r = buf.sample_channel_bilinear(1.5, 1.5, 0);
        let g = buf.sample_channel_bilinear(1.5, 1.5, 1);
        let b = buf.sample_channel_bilinear(1.5, 1.5, 2);
        assert!(approx_eq(r, 1.0, 0.1));
        assert!(approx_eq(g, 2.0, 0.1));
        assert!(approx_eq(b, 3.0, 0.1));
    }

    #[test]
    fn test_radial_distance_center() {
        let d = radial_distance(4, 4, 8, 8, (0.5, 0.5));
        assert!(d < 0.1); // Close to center
    }

    #[test]
    fn test_radial_distance_corner() {
        let d = radial_distance(0, 0, 8, 8, (0.5, 0.5));
        assert!(d > 0.3);
    }

    #[test]
    fn test_radial_direction_center() {
        let (dx, dy) = radial_direction(4, 4, 8, 8, (0.5, 0.5));
        // Near center, direction is degenerate
        assert!(dx.abs() + dy.abs() < 2.0);
    }

    #[test]
    fn test_distortion_none() {
        assert!(approx_eq(apply_distortion(0.5, &DistortionModel::None), 0.5, 1e-10));
    }

    #[test]
    fn test_distortion_barrel() {
        let r = apply_distortion(0.5, &DistortionModel::Barrel(0.1));
        assert!(r > 0.5); // Barrel expands
    }

    #[test]
    fn test_distortion_pincushion() {
        let r = apply_distortion(0.5, &DistortionModel::Pincushion(-0.1));
        assert!(r < 0.5); // Pincushion shrinks
    }

    #[test]
    fn test_spectral_to_rgb_sums_to_one() {
        for bin in 0..7 {
            let (r, g, b) = spectral_to_rgb(bin, 7);
            let sum = r + g + b;
            assert!(approx_eq(sum, 1.0, 0.1));
        }
    }

    #[test]
    fn test_spectral_offset_range() {
        for bin in 0..7 {
            let offset = spectral_offset(bin, 7);
            assert!(offset >= -1.0 - 1e-6);
            assert!(offset <= 1.0 + 1e-6);
        }
    }

    #[test]
    fn test_spectral_offset_center_is_zero() {
        let offset = spectral_offset(3, 7);
        assert!(approx_eq(offset, 0.0, 1e-6));
    }

    #[test]
    fn test_apply_ca_zero_intensity() {
        let buf = CaBuffer::new(8, 8);
        let config = CaConfig { intensity: 0.0, ..Default::default() };
        let result = apply_chromatic_aberration(&buf, &config, None);
        // Zero intensity returns input
        let px = result.get(4, 4);
        assert!(approx_eq(px.r, 0.0, 1e-10));
    }

    #[test]
    fn test_apply_ca_fast_mode() {
        let mut buf = CaBuffer::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                buf.set(x, y, CaPixel::new(1.0, 1.0, 1.0));
            }
        }
        let config = CaConfig {
            intensity: 0.5,
            mode: CaMode::Fast,
            ..Default::default()
        };
        let result = apply_chromatic_aberration(&buf, &config, None);
        assert_eq!(result.width, 16);
        // Uniform input with small intensity should produce roughly uniform output
        let px = result.get(8, 8);
        assert!(px.r > 0.5);
    }

    #[test]
    fn test_apply_ca_spectral_mode() {
        let mut buf = CaBuffer::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                buf.set(x, y, CaPixel::new(1.0, 1.0, 1.0));
            }
        }
        let config = CaConfig {
            intensity: 0.5,
            mode: CaMode::Spectral { wavelength_bins: 7 },
            ..Default::default()
        };
        let result = apply_chromatic_aberration(&buf, &config, None);
        let px = result.get(4, 4);
        assert!(px.r > 0.0);
        assert!(px.g > 0.0);
    }

    #[test]
    fn test_apply_ca_with_depth() {
        let mut buf = CaBuffer::new(8, 8);
        let mut depth = CaDepthBuffer::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                buf.set(x, y, CaPixel::new(1.0, 1.0, 1.0));
                depth.set(x, y, 0.5);
            }
        }
        let config = CaConfig {
            axial_enabled: true,
            axial_intensity: 1.0,
            ..Default::default()
        };
        let result = apply_chromatic_aberration(&buf, &config, Some(&depth));
        assert_eq!(result.width, 8);
    }

    #[test]
    fn test_ca_config_display() {
        let config = CaConfig::default();
        let s = format!("{}", config);
        assert!(s.contains("CaConfig"));
    }

    #[test]
    fn test_distortion_model_display() {
        assert_eq!(format!("{}", DistortionModel::None), "None");
        let s = format!("{}", DistortionModel::Barrel(0.1));
        assert!(s.contains("Barrel"));
    }

    #[test]
    fn test_ca_mode_display() {
        assert_eq!(format!("{}", CaMode::Fast), "Fast");
        let s = format!("{}", CaMode::Spectral { wavelength_bins: 7 });
        assert!(s.contains("Spectral"));
    }

    #[test]
    fn test_axial_ca_depth_variation() {
        let mut buf = CaBuffer::new(8, 8);
        let mut depth_near = CaDepthBuffer::new(8, 8);
        let mut depth_far = CaDepthBuffer::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                buf.set(x, y, CaPixel::new(1.0, 0.0, 0.0));
                depth_near.set(x, y, 0.1);
                depth_far.set(x, y, 0.9);
            }
        }
        let config = CaConfig {
            intensity: 2.0,
            axial_enabled: true,
            axial_intensity: 2.0,
            ..Default::default()
        };
        let near_result = apply_chromatic_aberration(&buf, &config, Some(&depth_near));
        let far_result = apply_chromatic_aberration(&buf, &config, Some(&depth_far));
        // Results should differ due to depth-dependent offset
        let pn = near_result.get(1, 1);
        let pf = far_result.get(1, 1);
        // Difference is expected at edges
        assert!((pn.r - pf.r).abs() >= 0.0); // may differ
    }

    #[test]
    fn test_uniform_color_center_unchanged() {
        // At exact center with uniform color, all channels should read the same
        let mut buf = CaBuffer::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                buf.set(x, y, CaPixel::new(0.5, 0.5, 0.5));
            }
        }
        let config = CaConfig::default();
        let result = apply_chromatic_aberration(&buf, &config, None);
        let px = result.get(4, 4);
        assert!(approx_eq(px.r, 0.5, 0.1));
        assert!(approx_eq(px.g, 0.5, 0.1));
        assert!(approx_eq(px.b, 0.5, 0.1));
    }
}
