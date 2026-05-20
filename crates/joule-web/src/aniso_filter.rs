// aniso_filter.rs — Anisotropic texture filtering.
// EWA (Elliptical Weighted Average), mip level selection, trilinear-aniso combination.

use std::f32::consts::PI;

/// RGBA pixel as f32.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pixel {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Pixel {
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub fn black() -> Self {
        Self::new(0.0, 0.0, 0.0, 1.0)
    }

    pub fn add(&self, other: &Pixel) -> Pixel {
        Pixel {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
            a: self.a + other.a,
        }
    }

    pub fn scale(&self, f: f32) -> Pixel {
        Pixel {
            r: self.r * f,
            g: self.g * f,
            b: self.b * f,
            a: self.a * f,
        }
    }

    pub fn clamped(&self) -> Pixel {
        Pixel {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
            a: self.a.clamp(0.0, 1.0),
        }
    }

    pub fn lerp(a: &Pixel, b: &Pixel, t: f32) -> Pixel {
        Pixel {
            r: a.r + (b.r - a.r) * t,
            g: a.g + (b.g - a.g) * t,
            b: a.b + (b.b - a.b) * t,
            a: a.a + (b.a - a.a) * t,
        }
    }
}

/// Screen-space texture derivatives (partial derivatives of UV with respect to screen XY).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextureDerivatives {
    pub du_dx: f32,
    pub du_dy: f32,
    pub dv_dx: f32,
    pub dv_dy: f32,
}

impl TextureDerivatives {
    pub fn new(du_dx: f32, du_dy: f32, dv_dx: f32, dv_dy: f32) -> Self {
        Self { du_dx, du_dy, dv_dx, dv_dy }
    }

    /// Compute the length of the major and minor axes of the screen-space ellipse.
    pub fn axis_lengths(&self) -> (f32, f32) {
        let a = self.du_dx * self.du_dx + self.dv_dx * self.dv_dx;
        let b = self.du_dx * self.du_dy + self.dv_dx * self.dv_dy;
        let c = self.du_dy * self.du_dy + self.dv_dy * self.dv_dy;

        let det = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
        let major_sq = 0.5 * ((a + c) + det);
        let minor_sq = 0.5 * ((a + c) - det);

        (major_sq.max(0.0).sqrt(), minor_sq.max(0.0).sqrt())
    }

    /// Compute the anisotropy ratio (major / minor axis), clamped.
    pub fn anisotropy_ratio(&self, max_aniso: f32) -> f32 {
        let (major, minor) = self.axis_lengths();
        if minor < 1e-10 {
            return max_aniso;
        }
        (major / minor).min(max_aniso).max(1.0)
    }

    /// Compute the ideal mip level based on the minor axis length.
    /// Assumes texture dimensions in texels.
    pub fn mip_level(&self, tex_width: f32, tex_height: f32) -> f32 {
        let (_, minor) = self.axis_lengths();
        let texel_size = minor * tex_width.max(tex_height);
        if texel_size < 1e-10 {
            return 0.0;
        }
        texel_size.log2().max(0.0)
    }

    /// Compute the angle of the major axis (in radians).
    pub fn major_axis_angle(&self) -> f32 {
        let a = self.du_dx * self.du_dx + self.dv_dx * self.dv_dx;
        let b = self.du_dx * self.du_dy + self.dv_dx * self.dv_dy;
        let c = self.du_dy * self.du_dy + self.dv_dy * self.dv_dy;

        0.5 * (2.0 * b).atan2(a - c)
    }
}

/// Texture map (single mip level).
#[derive(Debug, Clone, PartialEq)]
pub struct TextureMap {
    pub pixels: Vec<Pixel>,
    pub width: usize,
    pub height: usize,
}

impl TextureMap {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            pixels: vec![Pixel::black(); width * height],
            width,
            height,
        }
    }

    pub fn from_pixels(pixels: Vec<Pixel>, width: usize, height: usize) -> Option<Self> {
        if pixels.len() != width * height {
            return None;
        }
        Some(Self { pixels, width, height })
    }

    pub fn get(&self, x: usize, y: usize) -> Pixel {
        self.pixels[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, p: Pixel) {
        if x < self.width && y < self.height {
            self.pixels[y * self.width + x] = p;
        }
    }

    /// Sample with wrapping (repeat mode).
    pub fn sample_wrap(&self, u: f32, v: f32) -> Pixel {
        let fu = u.fract();
        let fv = v.fract();
        let fu = if fu < 0.0 { fu + 1.0 } else { fu };
        let fv = if fv < 0.0 { fv + 1.0 } else { fv };
        let x = ((fu * self.width as f32) as usize).min(self.width - 1);
        let y = ((fv * self.height as f32) as usize).min(self.height - 1);
        self.get(x, y)
    }

    /// Bilinear sample with wrapping.
    pub fn sample_bilinear(&self, u: f32, v: f32) -> Pixel {
        let fu = u * self.width as f32 - 0.5;
        let fv = v * self.height as f32 - 0.5;

        let x0 = fu.floor() as isize;
        let y0 = fv.floor() as isize;
        let frac_x = fu - fu.floor();
        let frac_y = fv - fv.floor();

        let sample = |sx: isize, sy: isize| -> Pixel {
            let wx = sx.rem_euclid(self.width as isize) as usize;
            let wy = sy.rem_euclid(self.height as isize) as usize;
            self.get(wx, wy)
        };

        let c00 = sample(x0, y0);
        let c10 = sample(x0 + 1, y0);
        let c01 = sample(x0, y0 + 1);
        let c11 = sample(x0 + 1, y0 + 1);

        let top = Pixel::lerp(&c00, &c10, frac_x);
        let bot = Pixel::lerp(&c01, &c11, frac_x);
        Pixel::lerp(&top, &bot, frac_y)
    }
}

/// Anisotropic filtering configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnisoConfig {
    /// Maximum anisotropy ratio (typically 2, 4, 8, or 16).
    pub max_aniso: u32,
    /// Use Gaussian weight along major axis (true) or box filter (false).
    pub gaussian_weights: bool,
}

impl Default for AnisoConfig {
    fn default() -> Self {
        Self {
            max_aniso: 16,
            gaussian_weights: true,
        }
    }
}

impl AnisoConfig {
    pub fn new(max_aniso: u32) -> Self {
        Self {
            max_aniso: max_aniso.clamp(1, 16),
            gaussian_weights: true,
        }
    }
}

/// Gaussian weight function.
fn gaussian_weight(distance_sq: f32, sigma_sq: f32) -> f32 {
    (-distance_sq / (2.0 * sigma_sq)).exp()
}

/// EWA (Elliptical Weighted Average) filter for a single texel lookup.
pub fn ewa_filter(
    texture: &TextureMap,
    u: f32,
    v: f32,
    derivs: &TextureDerivatives,
    config: &AnisoConfig,
) -> Pixel {
    let ratio = derivs.anisotropy_ratio(config.max_aniso as f32);
    let (major, minor) = derivs.axis_lengths();

    if ratio < 1.01 {
        // Nearly isotropic — just use bilinear.
        return texture.sample_bilinear(u, v);
    }

    // Number of samples along the major axis.
    let num_samples = (ratio.ceil() as usize).clamp(2, config.max_aniso as usize);

    // Direction of the major axis.
    let angle = derivs.major_axis_angle();
    let step_u = angle.cos() * major / num_samples as f32;
    let step_v = angle.sin() * major / num_samples as f32;

    let sigma_sq = (minor * 0.5).max(1e-6);
    let sigma_sq = sigma_sq * sigma_sq;

    let mut accum = Pixel::new(0.0, 0.0, 0.0, 0.0);
    let mut total_weight = 0.0f32;

    let half = num_samples as f32 / 2.0;

    for i in 0..num_samples {
        let t = i as f32 - half + 0.5;
        let sample_u = u + t * step_u;
        let sample_v = v + t * step_v;

        let w = if config.gaussian_weights {
            let dist_sq = t * t;
            gaussian_weight(dist_sq, half * half * 0.5)
        } else {
            1.0
        };

        let sample = texture.sample_bilinear(sample_u, sample_v);
        accum = accum.add(&sample.scale(w));
        total_weight += w;
    }

    if total_weight > 1e-10 {
        accum = accum.scale(1.0 / total_weight);
    }

    accum.clamped()
}

/// Trilinear filtering between two mip levels.
pub fn trilinear_sample(
    mip_a: &TextureMap,
    mip_b: &TextureMap,
    u: f32,
    v: f32,
    blend: f32,
) -> Pixel {
    let a = mip_a.sample_bilinear(u, v);
    let b = mip_b.sample_bilinear(u, v);
    Pixel::lerp(&a, &b, blend.clamp(0.0, 1.0))
}

/// Trilinear-anisotropic combination: aniso filter at two mip levels, then blend.
pub fn trilinear_aniso(
    mip_a: &TextureMap,
    mip_b: &TextureMap,
    u: f32,
    v: f32,
    derivs: &TextureDerivatives,
    blend: f32,
    config: &AnisoConfig,
) -> Pixel {
    let a = ewa_filter(mip_a, u, v, derivs, config);
    let b = ewa_filter(mip_b, u, v, derivs, config);
    Pixel::lerp(&a, &b, blend.clamp(0.0, 1.0))
}

/// Quality level for anisotropic filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnisoQuality {
    /// Bilinear only (no aniso).
    Off,
    /// 2x anisotropy.
    X2,
    /// 4x anisotropy.
    X4,
    /// 8x anisotropy.
    X8,
    /// 16x anisotropy.
    X16,
}

impl AnisoQuality {
    pub fn max_ratio(&self) -> u32 {
        match self {
            AnisoQuality::Off => 1,
            AnisoQuality::X2 => 2,
            AnisoQuality::X4 => 4,
            AnisoQuality::X8 => 8,
            AnisoQuality::X16 => 16,
        }
    }

    pub fn to_config(&self) -> AnisoConfig {
        AnisoConfig::new(self.max_ratio())
    }
}

/// Compute the texture-space ellipse parameters from screen-space derivatives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EllipseParams {
    pub center_u: f32,
    pub center_v: f32,
    pub semi_major: f32,
    pub semi_minor: f32,
    pub angle: f32,
    pub aniso_ratio: f32,
}

pub fn compute_ellipse(
    u: f32,
    v: f32,
    derivs: &TextureDerivatives,
    max_aniso: f32,
) -> EllipseParams {
    let (major, minor) = derivs.axis_lengths();
    let angle = derivs.major_axis_angle();
    let ratio = derivs.anisotropy_ratio(max_aniso);

    EllipseParams {
        center_u: u,
        center_v: v,
        semi_major: major,
        semi_minor: minor,
        angle,
        aniso_ratio: ratio,
    }
}

/// Estimate the number of texture samples needed for a given aniso ratio.
pub fn estimate_sample_count(derivs: &TextureDerivatives, max_aniso: u32) -> usize {
    let ratio = derivs.anisotropy_ratio(max_aniso as f32);
    (ratio.ceil() as usize).clamp(1, max_aniso as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_texture(w: usize, h: usize, c: Pixel) -> TextureMap {
        TextureMap {
            pixels: vec![c; w * h],
            width: w,
            height: h,
        }
    }

    fn checker_texture(w: usize, h: usize) -> TextureMap {
        let mut tex = TextureMap::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let c = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
                tex.set(x, y, Pixel::new(c, c, c, 1.0));
            }
        }
        tex
    }

    #[test]
    fn test_derivatives_isotropic() {
        let d = TextureDerivatives::new(0.01, 0.0, 0.0, 0.01);
        let (major, minor) = d.axis_lengths();
        assert!((major - minor).abs() < 1e-4, "isotropic should have equal axes");
        assert!((d.anisotropy_ratio(16.0) - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_derivatives_anisotropic() {
        let d = TextureDerivatives::new(0.1, 0.0, 0.0, 0.01);
        let ratio = d.anisotropy_ratio(16.0);
        assert!(ratio > 5.0, "should be highly anisotropic: {}", ratio);
    }

    #[test]
    fn test_derivatives_ratio_clamp() {
        let d = TextureDerivatives::new(1.0, 0.0, 0.0, 0.001);
        let ratio = d.anisotropy_ratio(8.0);
        assert!(ratio <= 8.0, "should clamp to max: {}", ratio);
    }

    #[test]
    fn test_derivatives_mip_level() {
        let d = TextureDerivatives::new(0.01, 0.0, 0.0, 0.01);
        let level = d.mip_level(256.0, 256.0);
        assert!(level >= 0.0);
    }

    #[test]
    fn test_derivatives_angle() {
        let d = TextureDerivatives::new(0.1, 0.0, 0.0, 0.01);
        let angle = d.major_axis_angle();
        // Should be near 0 (horizontal major axis).
        assert!(angle.abs() < PI / 4.0, "angle should be near zero: {}", angle);
    }

    #[test]
    fn test_texture_sample_wrap() {
        let tex = solid_texture(4, 4, Pixel::new(0.5, 0.5, 0.5, 1.0));
        let p = tex.sample_wrap(0.5, 0.5);
        assert!((p.r - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_texture_sample_bilinear_solid() {
        let tex = solid_texture(4, 4, Pixel::new(0.3, 0.7, 0.5, 1.0));
        let p = tex.sample_bilinear(0.5, 0.5);
        assert!((p.r - 0.3).abs() < 1e-3);
        assert!((p.g - 0.7).abs() < 1e-3);
    }

    #[test]
    fn test_ewa_isotropic_solid() {
        let tex = solid_texture(8, 8, Pixel::new(0.5, 0.5, 0.5, 1.0));
        let d = TextureDerivatives::new(0.01, 0.0, 0.0, 0.01);
        let config = AnisoConfig::default();
        let p = ewa_filter(&tex, 0.5, 0.5, &d, &config);
        assert!((p.r - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_ewa_anisotropic_solid() {
        let tex = solid_texture(8, 8, Pixel::new(0.6, 0.6, 0.6, 1.0));
        let d = TextureDerivatives::new(0.1, 0.0, 0.0, 0.01);
        let config = AnisoConfig::default();
        let p = ewa_filter(&tex, 0.5, 0.5, &d, &config);
        assert!((p.r - 0.6).abs() < 0.05, "aniso on solid should be close: {}", p.r);
    }

    #[test]
    fn test_ewa_checker_reduces_aliasing() {
        let tex = checker_texture(16, 16);
        let d = TextureDerivatives::new(0.1, 0.0, 0.0, 0.01);
        let config = AnisoConfig::new(8);
        let p = ewa_filter(&tex, 0.5, 0.5, &d, &config);
        // On a checkerboard, aniso should produce a mid-gray.
        assert!(p.r > 0.1 && p.r < 0.9, "should blend checker: {}", p.r);
    }

    #[test]
    fn test_trilinear_sample() {
        let a = solid_texture(4, 4, Pixel::new(0.0, 0.0, 0.0, 1.0));
        let b = solid_texture(2, 2, Pixel::new(1.0, 1.0, 1.0, 1.0));
        let p = trilinear_sample(&a, &b, 0.5, 0.5, 0.5);
        assert!((p.r - 0.5).abs() < 1e-2);
    }

    #[test]
    fn test_trilinear_aniso() {
        let a = solid_texture(8, 8, Pixel::new(0.3, 0.3, 0.3, 1.0));
        let b = solid_texture(4, 4, Pixel::new(0.7, 0.7, 0.7, 1.0));
        let d = TextureDerivatives::new(0.05, 0.0, 0.0, 0.05);
        let config = AnisoConfig::default();
        let p = trilinear_aniso(&a, &b, 0.5, 0.5, &d, 0.5, &config);
        assert!((p.r - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_aniso_quality_ratios() {
        assert_eq!(AnisoQuality::Off.max_ratio(), 1);
        assert_eq!(AnisoQuality::X2.max_ratio(), 2);
        assert_eq!(AnisoQuality::X4.max_ratio(), 4);
        assert_eq!(AnisoQuality::X8.max_ratio(), 8);
        assert_eq!(AnisoQuality::X16.max_ratio(), 16);
    }

    #[test]
    fn test_aniso_quality_config() {
        let config = AnisoQuality::X8.to_config();
        assert_eq!(config.max_aniso, 8);
    }

    #[test]
    fn test_compute_ellipse() {
        let d = TextureDerivatives::new(0.1, 0.0, 0.0, 0.01);
        let e = compute_ellipse(0.5, 0.5, &d, 16.0);
        assert!((e.center_u - 0.5).abs() < 1e-6);
        assert!(e.semi_major > e.semi_minor);
        assert!(e.aniso_ratio > 1.0);
    }

    #[test]
    fn test_estimate_sample_count() {
        let d_iso = TextureDerivatives::new(0.01, 0.0, 0.0, 0.01);
        assert_eq!(estimate_sample_count(&d_iso, 16), 1);

        let d_aniso = TextureDerivatives::new(0.1, 0.0, 0.0, 0.01);
        let count = estimate_sample_count(&d_aniso, 16);
        assert!(count > 1, "aniso should need multiple samples: {}", count);
    }

    #[test]
    fn test_gaussian_weight_center() {
        let w = gaussian_weight(0.0, 1.0);
        assert!((w - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gaussian_weight_far() {
        let w = gaussian_weight(100.0, 1.0);
        assert!(w < 1e-10);
    }

    #[test]
    fn test_pixel_lerp() {
        let a = Pixel::new(0.0, 0.0, 0.0, 1.0);
        let b = Pixel::new(1.0, 1.0, 1.0, 1.0);
        let mid = Pixel::lerp(&a, &b, 0.5);
        assert!((mid.r - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_config_clamp() {
        let config = AnisoConfig::new(32);
        assert_eq!(config.max_aniso, 16);

        let config2 = AnisoConfig::new(0);
        assert_eq!(config2.max_aniso, 1);
    }

    #[test]
    fn test_ewa_box_weights() {
        let tex = solid_texture(8, 8, Pixel::new(0.5, 0.5, 0.5, 1.0));
        let d = TextureDerivatives::new(0.1, 0.0, 0.0, 0.01);
        let config = AnisoConfig {
            max_aniso: 8,
            gaussian_weights: false,
        };
        let p = ewa_filter(&tex, 0.5, 0.5, &d, &config);
        assert!((p.r - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_texture_from_pixels_invalid() {
        let result = TextureMap::from_pixels(vec![Pixel::black(); 5], 3, 2);
        assert!(result.is_none());
    }

    #[test]
    fn test_texture_from_pixels_valid() {
        let result = TextureMap::from_pixels(vec![Pixel::black(); 6], 3, 2);
        assert!(result.is_some());
    }
}
