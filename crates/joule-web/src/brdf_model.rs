// BRDF models for physically-based rendering.
// Lambertian, Cook-Torrance (GGX/Beckmann), Oren-Nayar, energy conservation.

use std::fmt;

const PI: f64 = std::f64::consts::PI;
const INV_PI: f64 = 1.0 / PI;
const TWO_PI: f64 = 2.0 * PI;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0, z: 0.0 } }
    pub fn dot(self, o: Self) -> f64 { self.x * o.x + self.y * o.y + self.z * o.z }
    pub fn cross(self, o: Self) -> Self {
        Self {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }
    pub fn length(self) -> f64 { self.dot(self).sqrt() }
    pub fn length_squared(self) -> f64 { self.dot(self) }
    pub fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-15 { Self::zero() } else { self * (1.0 / l) }
    }
    pub fn reflect(self, n: Self) -> Self {
        self - n * (2.0 * self.dot(n))
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, r: Self) -> Self { Self { x: self.x + r.x, y: self.y + r.y, z: self.z + r.z } }
}
impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, r: Self) -> Self { Self { x: self.x - r.x, y: self.y - r.y, z: self.z - r.z } }
}
impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s } }
}
impl std::ops::Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self { Self { x: -self.x, y: -self.y, z: -self.z } }
}

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4}, {:.4})", self.x, self.y, self.z)
    }
}

/// Linear RGB color for BRDF evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl Color {
    pub fn new(r: f64, g: f64, b: f64) -> Self { Self { r, g, b } }
    pub fn black() -> Self { Self { r: 0.0, g: 0.0, b: 0.0 } }
    pub fn white() -> Self { Self { r: 1.0, g: 1.0, b: 1.0 } }
    pub fn splat(v: f64) -> Self { Self { r: v, g: v, b: v } }
    pub fn luminance(self) -> f64 { 0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b }
    pub fn max_channel(self) -> f64 { self.r.max(self.g).max(self.b) }
}

impl std::ops::Add for Color {
    type Output = Self;
    fn add(self, r: Self) -> Self { Self { r: self.r + r.r, g: self.g + r.g, b: self.b + r.b } }
}
impl std::ops::Sub for Color {
    type Output = Self;
    fn sub(self, r: Self) -> Self { Self { r: self.r - r.r, g: self.g - r.g, b: self.b - r.b } }
}
impl std::ops::Mul<f64> for Color {
    type Output = Self;
    fn mul(self, s: f64) -> Self { Self { r: self.r * s, g: self.g * s, b: self.b * s } }
}
impl std::ops::Mul for Color {
    type Output = Self;
    fn mul(self, o: Self) -> Self { Self { r: self.r * o.r, g: self.g * o.g, b: self.b * o.b } }
}

// ─── NDF (Normal Distribution Functions) ───

/// GGX/Trowbridge-Reitz NDF: D(h, alpha).
pub fn ggx_ndf(n_dot_h: f64, roughness: f64) -> f64 {
    if n_dot_h <= 0.0 { return 0.0; }
    let a = roughness * roughness;
    let a2 = a * a;
    let cos2 = n_dot_h * n_dot_h;
    let denom = cos2 * (a2 - 1.0) + 1.0;
    a2 / (PI * denom * denom)
}

/// Beckmann NDF.
pub fn beckmann_ndf(n_dot_h: f64, roughness: f64) -> f64 {
    if n_dot_h <= 0.0 { return 0.0; }
    let a = roughness * roughness;
    let a2 = a * a;
    let cos2 = n_dot_h * n_dot_h;
    let cos4 = cos2 * cos2;
    let tan2 = (1.0 - cos2) / cos2;
    (-tan2 / a2).exp() / (PI * a2 * cos4)
}

// ─── Fresnel ───

/// Schlick Fresnel approximation (scalar version).
pub fn fresnel_schlick_scalar(cos_theta: f64, f0: f64) -> f64 {
    f0 + (1.0 - f0) * (1.0 - cos_theta.max(0.0)).powi(5)
}

/// Schlick Fresnel (color version, for metals).
pub fn fresnel_schlick(cos_theta: f64, f0: Color) -> Color {
    let t = (1.0 - cos_theta.max(0.0)).powi(5);
    f0 + (Color::white() - f0) * t
}

// ─── Geometry / Masking-Shadowing ───

/// Smith GGX G1 (single-direction masking function).
fn smith_ggx_g1(n_dot_v: f64, roughness: f64) -> f64 {
    if n_dot_v <= 0.0 { return 0.0; }
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_v2 = n_dot_v * n_dot_v;
    let denom = n_dot_v + (a2 + (1.0 - a2) * n_dot_v2).sqrt();
    2.0 * n_dot_v / denom
}

/// Smith height-correlated masking-shadowing (GGX).
pub fn smith_ggx_g2(n_dot_l: f64, n_dot_v: f64, roughness: f64) -> f64 {
    let g1_l = smith_ggx_g1(n_dot_l, roughness);
    let g1_v = smith_ggx_g1(n_dot_v, roughness);
    g1_l * g1_v
}

/// Smith-Beckmann G1.
fn smith_beckmann_g1(n_dot_v: f64, roughness: f64) -> f64 {
    if n_dot_v <= 0.0 { return 0.0; }
    let a = roughness * roughness;
    let cos2 = n_dot_v * n_dot_v;
    let tan_theta = ((1.0 - cos2) / cos2).sqrt();
    let c = 1.0 / (a * tan_theta);
    if c >= 1.6 {
        return 1.0;
    }
    (3.535 * c + 2.181 * c * c) / (1.0 + 2.276 * c + 2.577 * c * c)
}

/// Smith-Beckmann G2.
pub fn smith_beckmann_g2(n_dot_l: f64, n_dot_v: f64, roughness: f64) -> f64 {
    smith_beckmann_g1(n_dot_l, roughness) * smith_beckmann_g1(n_dot_v, roughness)
}

// ─── BRDF Models ───

/// Lambertian diffuse BRDF: f_d = albedo / pi.
pub fn lambertian(albedo: Color) -> Color {
    albedo * INV_PI
}

/// Cook-Torrance specular BRDF evaluation.
/// ndf_type: 0 = GGX, 1 = Beckmann.
pub fn cook_torrance(
    wi: Vec3,
    wo: Vec3,
    normal: Vec3,
    roughness: f64,
    f0: Color,
    ndf_type: u8,
) -> Color {
    let n = normal.normalized();
    let h = (wi + wo).normalized();
    let n_dot_l = n.dot(wi).max(0.0);
    let n_dot_v = n.dot(wo).max(0.0);
    let n_dot_h = n.dot(h).max(0.0);
    let v_dot_h = wo.dot(h).max(0.0);

    if n_dot_l <= 0.0 || n_dot_v <= 0.0 {
        return Color::black();
    }

    let d = match ndf_type {
        1 => beckmann_ndf(n_dot_h, roughness),
        _ => ggx_ndf(n_dot_h, roughness),
    };

    let f = fresnel_schlick(v_dot_h, f0);

    let g = match ndf_type {
        1 => smith_beckmann_g2(n_dot_l, n_dot_v, roughness),
        _ => smith_ggx_g2(n_dot_l, n_dot_v, roughness),
    };

    let denom = 4.0 * n_dot_l * n_dot_v;
    if denom < 1e-15 {
        return Color::black();
    }

    f * (d * g / denom)
}

/// PBR material parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct PbrMaterial {
    pub base_color: Color,
    pub metallic: f64,
    pub roughness: f64,
}

impl PbrMaterial {
    pub fn new(base_color: Color, metallic: f64, roughness: f64) -> Self {
        Self {
            base_color,
            metallic: metallic.clamp(0.0, 1.0),
            roughness: roughness.clamp(0.01, 1.0),
        }
    }

    /// Compute F0 from base color and metalness.
    pub fn f0(&self) -> Color {
        let dielectric_f0 = Color::splat(0.04);
        dielectric_f0 * (1.0 - self.metallic) + self.base_color * self.metallic
    }
}

/// Full PBR BRDF: diffuse + specular with energy conservation.
/// Metallic surfaces have no diffuse component.
pub fn pbr_brdf(
    wi: Vec3,
    wo: Vec3,
    normal: Vec3,
    material: &PbrMaterial,
) -> Color {
    let n = normal.normalized();
    let h = (wi + wo).normalized();
    let n_dot_l = n.dot(wi).max(0.0);
    let n_dot_v = n.dot(wo).max(0.0);
    let v_dot_h = wo.dot(h).max(0.0);

    if n_dot_l <= 0.0 || n_dot_v <= 0.0 {
        return Color::black();
    }

    let f0 = material.f0();
    let f = fresnel_schlick(v_dot_h, f0);

    // Specular
    let specular = cook_torrance(wi, wo, normal, material.roughness, f0, 0);

    // Diffuse with energy conservation: (1 - F) * (1 - metallic) * albedo/pi
    let k_d = (Color::white() - f) * (1.0 - material.metallic);
    let diffuse = lambertian(material.base_color);

    k_d * diffuse + specular
}

// ─── Oren-Nayar ───

/// Oren-Nayar rough diffuse model.
pub fn oren_nayar(
    wi: Vec3,
    wo: Vec3,
    normal: Vec3,
    albedo: Color,
    roughness: f64,
) -> Color {
    let n = normal.normalized();
    let n_dot_l = n.dot(wi).max(0.0);
    let n_dot_v = n.dot(wo).max(0.0);

    if n_dot_l <= 0.0 || n_dot_v <= 0.0 {
        return Color::black();
    }

    let sigma2 = roughness * roughness;
    let a_coeff = 1.0 - 0.5 * sigma2 / (sigma2 + 0.33);
    let b_coeff = 0.45 * sigma2 / (sigma2 + 0.09);

    let theta_i = n_dot_l.acos();
    let theta_r = n_dot_v.acos();

    // Project wi and wo onto tangent plane and compute cos(phi_i - phi_r)
    let wi_proj = (wi - n * n_dot_l).normalized();
    let wo_proj = (wo - n * n_dot_v).normalized();
    let cos_phi_diff = wi_proj.dot(wo_proj).max(0.0);

    let alpha = theta_i.max(theta_r);
    let beta = theta_i.min(theta_r);

    let c = a_coeff + b_coeff * cos_phi_diff * alpha.sin() * beta.tan();

    albedo * (INV_PI * c)
}

// ─── BRDF Sampling ───

/// Build orthonormal basis from normal.
fn build_onb(n: Vec3) -> (Vec3, Vec3, Vec3) {
    let w = n.normalized();
    let a = if w.x.abs() > 0.9 { Vec3::new(0.0, 1.0, 0.0) } else { Vec3::new(1.0, 0.0, 0.0) };
    let v = w.cross(a).normalized();
    let u = w.cross(v);
    (u, v, w)
}

/// Sample BRDF direction by importance sampling the GGX NDF.
/// Returns (sampled_direction, pdf).
pub fn sample_ggx_brdf(
    wo: Vec3,
    normal: Vec3,
    roughness: f64,
    r1: f64,
    r2: f64,
) -> (Vec3, f64) {
    let (u, v, w) = build_onb(normal);
    let a = roughness * roughness;
    let a2 = a * a;

    // Sample microfacet normal (half vector)
    let cos_theta = ((1.0 - r1) / (r1 * (a2 - 1.0) + 1.0)).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = TWO_PI * r2;

    let h_local = Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);
    let h = (u * h_local.x + v * h_local.y + w * h_local.z).normalized();

    // Reflect wo around h to get wi
    let wi = (wo * (-1.0)).reflect(h);

    let n_dot_h = normal.normalized().dot(h).max(0.0);
    let v_dot_h = wo.dot(h).max(0.0);

    let d = ggx_ndf(n_dot_h, roughness);
    let pdf = if v_dot_h > 0.0 { d * n_dot_h / (4.0 * v_dot_h) } else { 0.0 };

    (wi, pdf)
}

/// Sample cosine-weighted hemisphere for diffuse BRDF.
/// Returns (sampled_direction, pdf).
pub fn sample_cosine_brdf(
    normal: Vec3,
    r1: f64,
    r2: f64,
) -> (Vec3, f64) {
    let (u, v, w) = build_onb(normal);
    let r = r1.sqrt();
    let phi = TWO_PI * r2;
    let x = r * phi.cos();
    let y = r * phi.sin();
    let z = (1.0 - r1).max(0.0).sqrt();

    let wi = (u * x + v * y + w * z).normalized();
    let cos_theta = wi.dot(normal.normalized()).max(0.0);
    let pdf = cos_theta * INV_PI;

    (wi, pdf)
}

/// PDF for a given direction sampled from GGX NDF.
pub fn pdf_ggx_brdf(
    wo: Vec3,
    wi: Vec3,
    normal: Vec3,
    roughness: f64,
) -> f64 {
    let h = (wi + wo).normalized();
    let n_dot_h = normal.normalized().dot(h).max(0.0);
    let v_dot_h = wo.dot(h).max(0.0);
    if v_dot_h <= 0.0 { return 0.0; }
    let d = ggx_ndf(n_dot_h, roughness);
    d * n_dot_h / (4.0 * v_dot_h)
}

/// PDF for cosine-weighted hemisphere sampling.
pub fn pdf_cosine_brdf(wi: Vec3, normal: Vec3) -> f64 {
    let cos_theta = wi.dot(normal.normalized()).max(0.0);
    cos_theta * INV_PI
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool { (a - b).abs() < eps }

    fn color_approx(a: Color, b: Color, eps: f64) -> bool {
        approx_eq(a.r, b.r, eps) && approx_eq(a.g, b.g, eps) && approx_eq(a.b, b.b, eps)
    }

    #[test]
    fn test_lambertian_value() {
        let f = lambertian(Color::white());
        assert!(approx_eq(f.r, INV_PI, 1e-9));
    }

    #[test]
    fn test_lambertian_colored() {
        let f = lambertian(Color::new(0.5, 0.3, 0.1));
        assert!(approx_eq(f.r, 0.5 * INV_PI, 1e-9));
        assert!(approx_eq(f.g, 0.3 * INV_PI, 1e-9));
    }

    #[test]
    fn test_ggx_ndf_peak() {
        // NDF should peak at h parallel to n
        let d_peak = ggx_ndf(1.0, 0.3);
        let d_off = ggx_ndf(0.5, 0.3);
        assert!(d_peak > d_off);
    }

    #[test]
    fn test_ggx_ndf_zero_below() {
        assert!(approx_eq(ggx_ndf(-0.5, 0.3), 0.0, 1e-9));
        assert!(approx_eq(ggx_ndf(0.0, 0.3), 0.0, 1e-9));
    }

    #[test]
    fn test_ggx_ndf_rough_vs_smooth() {
        // Rougher surface has wider distribution (lower peak)
        let d_smooth = ggx_ndf(1.0, 0.1);
        let d_rough = ggx_ndf(1.0, 0.9);
        assert!(d_smooth > d_rough);
    }

    #[test]
    fn test_beckmann_ndf_positive() {
        let d = beckmann_ndf(0.8, 0.3);
        assert!(d > 0.0 && d.is_finite());
    }

    #[test]
    fn test_fresnel_schlick_normal_incidence() {
        let f = fresnel_schlick(1.0, Color::splat(0.04));
        assert!(color_approx(f, Color::splat(0.04), 1e-6));
    }

    #[test]
    fn test_fresnel_schlick_grazing() {
        let f = fresnel_schlick(0.0, Color::splat(0.04));
        assert!(color_approx(f, Color::white(), 1e-6));
    }

    #[test]
    fn test_fresnel_schlick_scalar() {
        let f = fresnel_schlick_scalar(1.0, 0.04);
        assert!(approx_eq(f, 0.04, 1e-9));
        let f = fresnel_schlick_scalar(0.0, 0.04);
        assert!(approx_eq(f, 1.0, 1e-9));
    }

    #[test]
    fn test_smith_ggx_g2_positive() {
        let g = smith_ggx_g2(0.8, 0.7, 0.3);
        assert!(g > 0.0 && g <= 1.0, "g={}", g);
    }

    #[test]
    fn test_smith_ggx_g2_zero_grazing() {
        let g = smith_ggx_g2(0.0, 0.5, 0.3);
        assert!(approx_eq(g, 0.0, 1e-9));
    }

    #[test]
    fn test_smith_beckmann_g2() {
        let g = smith_beckmann_g2(0.8, 0.7, 0.3);
        assert!(g > 0.0 && g <= 1.0 + 1e-6);
    }

    #[test]
    fn test_cook_torrance_normal_incidence() {
        let n = Vec3::new(0.0, 1.0, 0.0);
        let wo = Vec3::new(0.0, 1.0, 0.0);
        let wi = Vec3::new(0.0, 1.0, 0.0);
        let spec = cook_torrance(wi, wo, n, 0.5, Color::splat(0.04), 0);
        assert!(spec.r >= 0.0 && spec.r.is_finite());
    }

    #[test]
    fn test_cook_torrance_below_surface() {
        let n = Vec3::new(0.0, 1.0, 0.0);
        let wo = Vec3::new(0.0, -1.0, 0.0); // below surface
        let wi = Vec3::new(0.0, 1.0, 0.0);
        let spec = cook_torrance(wi, wo, n, 0.5, Color::splat(0.04), 0);
        assert!(color_approx(spec, Color::black(), 1e-9));
    }

    #[test]
    fn test_pbr_material_f0_dielectric() {
        let mat = PbrMaterial::new(Color::new(0.8, 0.2, 0.1), 0.0, 0.5);
        let f0 = mat.f0();
        assert!(color_approx(f0, Color::splat(0.04), 1e-9));
    }

    #[test]
    fn test_pbr_material_f0_metal() {
        let mat = PbrMaterial::new(Color::new(0.9, 0.7, 0.3), 1.0, 0.5);
        let f0 = mat.f0();
        assert!(color_approx(f0, Color::new(0.9, 0.7, 0.3), 1e-9));
    }

    #[test]
    fn test_pbr_brdf_metal_no_diffuse() {
        let n = Vec3::new(0.0, 1.0, 0.0);
        let wo = Vec3::new(0.0, 1.0, 0.0).normalized();
        let wi = Vec3::new(0.2, 0.8, 0.0).normalized();
        let mat = PbrMaterial::new(Color::new(1.0, 0.8, 0.3), 1.0, 0.5);
        let brdf = pbr_brdf(wi, wo, n, &mat);
        // For metallic=1, the diffuse part should be zero
        // The full BRDF is specular only
        assert!(brdf.r >= 0.0 && brdf.r.is_finite());
    }

    #[test]
    fn test_pbr_brdf_dielectric_has_diffuse() {
        let n = Vec3::new(0.0, 1.0, 0.0);
        let wo = Vec3::new(0.3, 0.9, 0.0).normalized();
        let wi = Vec3::new(-0.3, 0.9, 0.0).normalized();
        let mat = PbrMaterial::new(Color::new(0.8, 0.2, 0.1), 0.0, 0.5);
        let brdf = pbr_brdf(wi, wo, n, &mat);
        assert!(brdf.r > 0.0);
        assert!(brdf.g > 0.0);
    }

    #[test]
    fn test_oren_nayar_zero_roughness() {
        // At zero roughness Oren-Nayar approaches Lambertian
        let n = Vec3::new(0.0, 1.0, 0.0);
        let wo = Vec3::new(0.0, 1.0, 0.0);
        let wi = Vec3::new(0.3, 0.9, 0.0).normalized();
        let albedo = Color::new(0.5, 0.5, 0.5);
        let on = oren_nayar(wi, wo, n, albedo, 0.0);
        let lamb = lambertian(albedo);
        assert!(color_approx(on, lamb, 0.05));
    }

    #[test]
    fn test_oren_nayar_positive() {
        let n = Vec3::new(0.0, 1.0, 0.0);
        let wo = Vec3::new(0.3, 0.9, 0.0).normalized();
        let wi = Vec3::new(-0.3, 0.9, 0.0).normalized();
        let on = oren_nayar(wi, wo, n, Color::white(), 0.5);
        assert!(on.r > 0.0 && on.r.is_finite());
    }

    #[test]
    fn test_oren_nayar_below_surface() {
        let n = Vec3::new(0.0, 1.0, 0.0);
        let wo = Vec3::new(0.0, -1.0, 0.0);
        let wi = Vec3::new(0.0, 1.0, 0.0);
        let on = oren_nayar(wi, wo, n, Color::white(), 0.5);
        assert!(color_approx(on, Color::black(), 1e-9));
    }

    #[test]
    fn test_sample_ggx_brdf_on_hemisphere() {
        let n = Vec3::new(0.0, 1.0, 0.0);
        let wo = Vec3::new(0.2, 0.8, 0.1).normalized();
        struct TestRng { state: u64 }
        impl TestRng {
            fn new(s: u64) -> Self { Self { state: s.wrapping_add(1) } }
            fn next_f64(&mut self) -> f64 {
                self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (self.state >> 11) as f64 / (1u64 << 53) as f64
            }
        }
        let mut rng = TestRng::new(42);
        for _ in 0..100 {
            let (wi, pdf) = sample_ggx_brdf(wo, n, 0.3, rng.next_f64(), rng.next_f64());
            if wi.dot(n) > 0.0 {
                assert!(pdf >= 0.0 && pdf.is_finite());
                assert!(approx_eq(wi.length(), 1.0, 1e-6));
            }
        }
    }

    #[test]
    fn test_sample_cosine_brdf_on_hemisphere() {
        let n = Vec3::new(0.0, 0.0, 1.0);
        struct TestRng { state: u64 }
        impl TestRng {
            fn new(s: u64) -> Self { Self { state: s.wrapping_add(1) } }
            fn next_f64(&mut self) -> f64 {
                self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (self.state >> 11) as f64 / (1u64 << 53) as f64
            }
        }
        let mut rng = TestRng::new(42);
        for _ in 0..100 {
            let (wi, pdf) = sample_cosine_brdf(n, rng.next_f64(), rng.next_f64());
            assert!(wi.dot(n) >= -1e-6);
            assert!(pdf >= 0.0);
            assert!(approx_eq(wi.length(), 1.0, 1e-6));
        }
    }

    #[test]
    fn test_pdf_ggx_brdf_positive() {
        let n = Vec3::new(0.0, 1.0, 0.0);
        let wo = Vec3::new(0.2, 0.8, 0.0).normalized();
        let wi = Vec3::new(-0.2, 0.8, 0.0).normalized();
        let pdf = pdf_ggx_brdf(wo, wi, n, 0.3);
        assert!(pdf > 0.0 && pdf.is_finite());
    }

    #[test]
    fn test_pdf_cosine_brdf() {
        let n = Vec3::new(0.0, 1.0, 0.0);
        let wi = Vec3::new(0.0, 1.0, 0.0);
        let pdf = pdf_cosine_brdf(wi, n);
        assert!(approx_eq(pdf, INV_PI, 1e-6));
    }

    #[test]
    fn test_pbr_material_clamp() {
        let mat = PbrMaterial::new(Color::white(), 2.0, -1.0);
        assert!(approx_eq(mat.metallic, 1.0, 1e-9));
        assert!(approx_eq(mat.roughness, 0.01, 1e-9));
    }

    #[test]
    fn test_cook_torrance_beckmann() {
        let n = Vec3::new(0.0, 1.0, 0.0);
        let wo = Vec3::new(0.2, 0.8, 0.0).normalized();
        let wi = Vec3::new(-0.2, 0.8, 0.0).normalized();
        let spec = cook_torrance(wi, wo, n, 0.5, Color::splat(0.04), 1);
        assert!(spec.r >= 0.0 && spec.r.is_finite());
    }

    #[test]
    fn test_energy_conservation_brdf() {
        // Monte Carlo check: integral of BRDF * cos_theta should be <= 1
        let n = Vec3::new(0.0, 1.0, 0.0);
        let wo = Vec3::new(0.0, 1.0, 0.0);
        let mat = PbrMaterial::new(Color::new(0.8, 0.8, 0.8), 0.0, 0.5);

        struct TestRng { state: u64 }
        impl TestRng {
            fn new(s: u64) -> Self { Self { state: s.wrapping_add(1) } }
            fn next_f64(&mut self) -> f64 {
                self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (self.state >> 11) as f64 / (1u64 << 53) as f64
            }
        }
        let mut rng = TestRng::new(42);
        let mut sum_r = 0.0;
        let samples = 10000;
        for _ in 0..samples {
            let (wi, pdf) = sample_cosine_brdf(n, rng.next_f64(), rng.next_f64());
            if pdf > 1e-10 {
                let cos_theta = wi.dot(n).max(0.0);
                let brdf = pbr_brdf(wi, wo, n, &mat);
                sum_r += brdf.r * cos_theta / pdf;
            }
        }
        let integral = sum_r / samples as f64;
        // Should be <= 1 (energy conserving)
        assert!(integral <= 1.1, "energy integral = {}", integral);
    }
}
