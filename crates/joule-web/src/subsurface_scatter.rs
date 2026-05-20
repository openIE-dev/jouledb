//! Subsurface scattering approximation for skin, wax, marble.
//!
//! Gaussian-sum diffusion profiles, screen-space blur kernel generation,
//! curvature-based SSS with pre-integrated lookup tables, transmittance for
//! thin-object back-lighting via thickness maps, and material parameters
//! (scatter color, scatter distance, thickness scale). Pure Rust — no deps.

use std::fmt;

// ── Inline vector / color types ─────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };

    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn normalize(self) -> Self {
        let len = self.dot(self).sqrt();
        if len < 1e-10 { return Self::ZERO; }
        Self { x: self.x / len, y: self.y / len, z: self.z / len }
    }

    pub fn scale(self, s: f32) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }

    pub fn add(self, other: Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }
}

/// RGB color (linear space, 0..1+).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color3 {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Color3 {
    pub const WHITE: Self = Self { r: 1.0, g: 1.0, b: 1.0 };
    pub const BLACK: Self = Self { r: 0.0, g: 0.0, b: 0.0 };

    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    pub fn scale(self, s: f32) -> Self {
        Self { r: self.r * s, g: self.g * s, b: self.b * s }
    }

    pub fn add(self, other: Self) -> Self {
        Self { r: self.r + other.r, g: self.g + other.g, b: self.b + other.b }
    }

    pub fn mul(self, other: Self) -> Self {
        Self { r: self.r * other.r, g: self.g * other.g, b: self.b * other.b }
    }

    pub fn luminance(&self) -> f32 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }

    pub fn clamp01(self) -> Self {
        Self {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
        }
    }
}

impl Default for Color3 {
    fn default() -> Self {
        Self::WHITE
    }
}

impl fmt::Display for Color3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.3}, {:.3}, {:.3})", self.r, self.g, self.b)
    }
}

// ── Gaussian term ───────────────────────────────────────────────

/// A single Gaussian term in the diffusion profile: weight * exp(-r^2 / (2*variance)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianTerm {
    pub weight: Color3,
    pub variance: f32,
}

impl GaussianTerm {
    pub fn new(weight: Color3, variance: f32) -> Self {
        Self { weight, variance: variance.max(1e-6) }
    }

    /// Evaluate this Gaussian at distance `r`.
    pub fn evaluate(&self, r: f32) -> Color3 {
        let exponent = -(r * r) / (2.0 * self.variance);
        let g = exponent.exp();
        self.weight.scale(g)
    }
}

// ── Diffusion profile ───────────────────────────────────────────

/// Sum-of-Gaussians diffusion profile (up to 3 terms, per d'Eon & Luebke).
#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionProfile {
    pub name: String,
    terms: Vec<GaussianTerm>,
}

impl DiffusionProfile {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), terms: Vec::new() }
    }

    /// Add a Gaussian term (up to 3).
    pub fn add_term(mut self, term: GaussianTerm) -> Self {
        if self.terms.len() < 3 {
            self.terms.push(term);
        }
        self
    }

    pub fn term_count(&self) -> usize {
        self.terms.len()
    }

    /// Evaluate the profile at distance `r` (world-space units).
    pub fn evaluate(&self, r: f32) -> Color3 {
        let mut sum = Color3::BLACK;
        for t in &self.terms {
            sum = sum.add(t.evaluate(r));
        }
        sum
    }

    /// Pre-baked skin-like profile.
    pub fn skin() -> Self {
        Self::new("skin")
            .add_term(GaussianTerm::new(Color3::new(0.233, 0.455, 0.649), 0.0064))
            .add_term(GaussianTerm::new(Color3::new(0.100, 0.336, 0.344), 0.0484))
            .add_term(GaussianTerm::new(Color3::new(0.118, 0.198, 0.0), 0.187))
    }

    /// Simple single-term marble/wax profile.
    pub fn marble() -> Self {
        Self::new("marble")
            .add_term(GaussianTerm::new(Color3::new(0.85, 0.85, 0.80), 0.02))
    }
}

impl fmt::Display for DiffusionProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DiffusionProfile({}, {} terms)", self.name, self.terms.len())
    }
}

// ── Screen-space blur kernel ────────────────────────────────────

/// A blur kernel sample for screen-space SSS.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KernelSample {
    /// Offset in texels (signed).
    pub offset: f32,
    /// Weight per channel.
    pub weight: Color3,
}

/// Generate a 1D screen-space blur kernel from a diffusion profile.
///
/// `num_samples` is the total number of taps (should be odd for center tap).
/// `max_radius` is the maximum blur radius in texels.
pub fn generate_blur_kernel(profile: &DiffusionProfile, num_samples: usize, max_radius: f32) -> Vec<KernelSample> {
    let num_samples = num_samples.max(1);
    let half = num_samples / 2;
    let mut samples = Vec::with_capacity(num_samples);

    // Accumulate for normalisation.
    let mut total_weight = Color3::BLACK;

    for i in 0..num_samples {
        let idx = i as f32 - half as f32;
        let offset = if half > 0 { idx / half as f32 * max_radius } else { 0.0 };
        let w = profile.evaluate(offset.abs());
        total_weight = total_weight.add(w);
        samples.push(KernelSample { offset, weight: w });
    }

    // Normalise so weights sum to (1,1,1).
    for s in &mut samples {
        s.weight = Color3::new(
            if total_weight.r > 1e-8 { s.weight.r / total_weight.r } else { 0.0 },
            if total_weight.g > 1e-8 { s.weight.g / total_weight.g } else { 0.0 },
            if total_weight.b > 1e-8 { s.weight.b / total_weight.b } else { 0.0 },
        );
    }

    samples
}

// ── Curvature-based SSS (pre-integrated) ────────────────────────

/// Pre-integrated curvature-based SSS lookup table.
///
/// Indexed by NdotL (row) and curvature (col).
#[derive(Debug, Clone)]
pub struct CurvatureLut {
    pub size: usize,
    data: Vec<Color3>,
}

impl CurvatureLut {
    /// Build the LUT from a diffusion profile.
    ///
    /// `size` = resolution in each dimension. Curvature ranges from 0 (flat) to 1 (tight).
    pub fn build(profile: &DiffusionProfile, size: usize) -> Self {
        let size = size.max(2);
        let mut data = Vec::with_capacity(size * size);

        for row in 0..size {
            let ndl = row as f32 / (size - 1) as f32 * 2.0 - 1.0; // -1..1
            for col in 0..size {
                let curvature = col as f32 / (size - 1) as f32;

                // Integrate the profile over the visible hemisphere, scaled by curvature.
                let scatter_width = (1.0 - curvature).max(0.01);
                let sample_count = 16u32;
                let mut accum = Color3::BLACK;
                let mut total = 0.0f32;

                for s in 0..sample_count {
                    let t = (s as f32 / (sample_count - 1) as f32) * 2.0 - 1.0;
                    let sample_ndl = ndl + t * scatter_width;
                    let diffuse = sample_ndl.clamp(0.0, 1.0);
                    let profile_val = profile.evaluate(t.abs() * scatter_width);
                    accum = accum.add(profile_val.scale(diffuse));
                    total += diffuse.max(0.001);
                }

                if total > 1e-6 {
                    data.push(accum.scale(1.0 / total));
                } else {
                    data.push(Color3::BLACK);
                }
            }
        }

        Self { size, data }
    }

    /// Look up the pre-integrated value.
    pub fn sample(&self, ndl: f32, curvature: f32) -> Color3 {
        let row = ((ndl * 0.5 + 0.5).clamp(0.0, 1.0) * (self.size - 1) as f32) as usize;
        let col = (curvature.clamp(0.0, 1.0) * (self.size - 1) as f32) as usize;
        let row = row.min(self.size - 1);
        let col = col.min(self.size - 1);
        self.data[row * self.size + col]
    }
}

// ── Transmittance ───────────────────────────────────────────────

/// Compute transmittance for thin objects (back-lighting).
///
/// Based on thickness map + Beer-Lambert attenuation.
/// `thickness` is 0..1 (0 = thinnest → most light passes through).
/// `attenuation` is the per-channel absorption color.
/// `light_dot_normal` is the dot product of light with the surface normal.
pub fn transmittance(
    thickness: f32,
    scatter_color: Color3,
    attenuation: Color3,
    thickness_scale: f32,
    light_dot_normal: f32,
) -> Color3 {
    // Wrap lighting: allows light from behind.
    let wrap = 0.5;
    let ndl_wrap = (light_dot_normal + wrap) / (1.0 + wrap);
    let ndl_back = (-light_dot_normal).max(0.0);

    // Transmittance contribution from back face.
    let d = thickness * thickness_scale;
    let falloff = Color3::new(
        (-d * attenuation.r).exp(),
        (-d * attenuation.g).exp(),
        (-d * attenuation.b).exp(),
    );

    // Combine: back-light transmittance scaled by scatter color.
    scatter_color.mul(falloff).scale(ndl_back + ndl_wrap.max(0.0) * 0.2)
}

// ── SSS Material parameters ─────────────────────────────────────

/// Complete SSS material description.
#[derive(Debug, Clone, PartialEq)]
pub struct SssMaterial {
    pub name: String,
    pub scatter_color: Color3,
    pub scatter_distance: f32,
    pub thickness_scale: f32,
    pub profile: DiffusionProfile,
}

impl SssMaterial {
    pub fn new(name: impl Into<String>, profile: DiffusionProfile) -> Self {
        Self {
            name: name.into(),
            scatter_color: Color3::new(0.8, 0.3, 0.2),
            scatter_distance: 1.0,
            thickness_scale: 1.0,
            profile,
        }
    }

    pub fn with_scatter_color(mut self, color: Color3) -> Self {
        self.scatter_color = color;
        self
    }

    pub fn with_scatter_distance(mut self, d: f32) -> Self {
        self.scatter_distance = d.max(0.0);
        self
    }

    pub fn with_thickness_scale(mut self, s: f32) -> Self {
        self.thickness_scale = s.max(0.0);
        self
    }

    /// Evaluate transmittance for this material.
    pub fn eval_transmittance(&self, thickness: f32, light_dot_normal: f32) -> Color3 {
        let attenuation = Color3::new(
            1.0 / self.scatter_distance.max(0.01),
            1.0 / self.scatter_distance.max(0.01),
            1.0 / self.scatter_distance.max(0.01),
        );
        transmittance(thickness, self.scatter_color, attenuation, self.thickness_scale, light_dot_normal)
    }
}

impl fmt::Display for SssMaterial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SssMaterial({}, dist={:.3})", self.name, self.scatter_distance)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn test_gaussian_term_at_zero() {
        let t = GaussianTerm::new(Color3::WHITE, 1.0);
        let v = t.evaluate(0.0);
        assert!(approx(v.r, 1.0));
        assert!(approx(v.g, 1.0));
    }

    #[test]
    fn test_gaussian_term_falloff() {
        let t = GaussianTerm::new(Color3::WHITE, 0.01);
        let v_near = t.evaluate(0.01);
        let v_far = t.evaluate(1.0);
        assert!(v_near.r > v_far.r);
    }

    #[test]
    fn test_diffusion_profile_single() {
        let p = DiffusionProfile::new("test")
            .add_term(GaussianTerm::new(Color3::WHITE, 1.0));
        let v = p.evaluate(0.0);
        assert!(approx(v.r, 1.0));
    }

    #[test]
    fn test_diffusion_profile_max_terms() {
        let p = DiffusionProfile::new("test")
            .add_term(GaussianTerm::new(Color3::WHITE, 1.0))
            .add_term(GaussianTerm::new(Color3::WHITE, 2.0))
            .add_term(GaussianTerm::new(Color3::WHITE, 3.0))
            .add_term(GaussianTerm::new(Color3::WHITE, 4.0)); // ignored
        assert_eq!(p.term_count(), 3);
    }

    #[test]
    fn test_skin_profile() {
        let p = DiffusionProfile::skin();
        assert_eq!(p.term_count(), 3);
        let v = p.evaluate(0.0);
        assert!(v.r > 0.0);
    }

    #[test]
    fn test_marble_profile() {
        let p = DiffusionProfile::marble();
        let v = p.evaluate(0.0);
        assert!(v.r > 0.5);
    }

    #[test]
    fn test_blur_kernel_generation() {
        let p = DiffusionProfile::skin();
        let kernel = generate_blur_kernel(&p, 11, 5.0);
        assert_eq!(kernel.len(), 11);
        // Weights should sum to approximately 1 per channel.
        let sum_r: f32 = kernel.iter().map(|s| s.weight.r).sum();
        assert!((sum_r - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_blur_kernel_center_heaviest() {
        let p = DiffusionProfile::skin();
        let kernel = generate_blur_kernel(&p, 11, 5.0);
        let center = &kernel[5];
        assert!(approx(center.offset, 0.0));
        // Center tap should have the most weight.
        for s in &kernel {
            assert!(center.weight.r >= s.weight.r - 1e-6);
        }
    }

    #[test]
    fn test_blur_kernel_single_sample() {
        let p = DiffusionProfile::marble();
        let kernel = generate_blur_kernel(&p, 1, 1.0);
        assert_eq!(kernel.len(), 1);
        assert!(approx(kernel[0].weight.r, 1.0));
    }

    #[test]
    fn test_curvature_lut_build() {
        let p = DiffusionProfile::skin();
        let lut = CurvatureLut::build(&p, 16);
        assert_eq!(lut.size, 16);
        assert_eq!(lut.data.len(), 256);
    }

    #[test]
    fn test_curvature_lut_lit_face() {
        let p = DiffusionProfile::skin();
        let lut = CurvatureLut::build(&p, 32);
        let lit = lut.sample(1.0, 0.0); // full NdotL, flat
        assert!(lit.r > 0.0);
    }

    #[test]
    fn test_curvature_lut_shadow_face() {
        let p = DiffusionProfile::skin();
        let lut = CurvatureLut::build(&p, 32);
        let shadow = lut.sample(-1.0, 0.0);
        let lit = lut.sample(1.0, 0.0);
        // Shadow should be dimmer than lit.
        assert!(shadow.luminance() <= lit.luminance() + 1e-3);
    }

    #[test]
    fn test_transmittance_thin() {
        let result = transmittance(
            0.1,
            Color3::new(0.8, 0.3, 0.2),
            Color3::new(1.0, 1.0, 1.0),
            1.0,
            -1.0, // light hitting back face
        );
        assert!(result.r > 0.0);
    }

    #[test]
    fn test_transmittance_thick() {
        let thin = transmittance(0.1, Color3::WHITE, Color3::new(5.0, 5.0, 5.0), 1.0, -1.0);
        let thick = transmittance(1.0, Color3::WHITE, Color3::new(5.0, 5.0, 5.0), 1.0, -1.0);
        // Thinner objects transmit more light.
        assert!(thin.r > thick.r);
    }

    #[test]
    fn test_transmittance_front_lit() {
        let result = transmittance(
            0.5,
            Color3::WHITE,
            Color3::new(1.0, 1.0, 1.0),
            1.0,
            1.0, // light hitting front face
        );
        // Front-lit → minimal back transmission.
        assert!(result.r < 0.5);
    }

    #[test]
    fn test_sss_material_builder() {
        let m = SssMaterial::new("skin", DiffusionProfile::skin())
            .with_scatter_color(Color3::new(0.9, 0.4, 0.3))
            .with_scatter_distance(2.0)
            .with_thickness_scale(0.5);
        assert!(approx(m.scatter_distance, 2.0));
        assert!(approx(m.thickness_scale, 0.5));
    }

    #[test]
    fn test_sss_material_eval_transmittance() {
        let m = SssMaterial::new("wax", DiffusionProfile::marble())
            .with_scatter_distance(1.0)
            .with_thickness_scale(1.0);
        let t = m.eval_transmittance(0.2, -0.5);
        assert!(t.r > 0.0);
    }

    #[test]
    fn test_sss_material_display() {
        let m = SssMaterial::new("skin", DiffusionProfile::skin());
        let s = format!("{m}");
        assert!(s.contains("skin"));
    }

    #[test]
    fn test_color3_ops() {
        let a = Color3::new(0.5, 0.5, 0.5);
        let b = Color3::new(0.2, 0.3, 0.4);
        let c = a.add(b);
        assert!(approx(c.r, 0.7));
        let d = a.mul(b);
        assert!(approx(d.r, 0.1));
    }

    #[test]
    fn test_color3_clamp() {
        let c = Color3::new(-0.5, 1.5, 0.5).clamp01();
        assert!(approx(c.r, 0.0));
        assert!(approx(c.g, 1.0));
        assert!(approx(c.b, 0.5));
    }

    #[test]
    fn test_profile_display() {
        let p = DiffusionProfile::skin();
        let s = format!("{p}");
        assert!(s.contains("skin"));
        assert!(s.contains("3 terms"));
    }
}
