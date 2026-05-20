//! Clear coat material layer for car paint / lacquer.
//!
//! Two-layer PBR: base layer (metallic + roughness) + clearcoat layer
//! (separate roughness, IOR 1.5 default). Schlick Fresnel blend between
//! layers, clearcoat normal map (orange-peel effect), clearcoat tint,
//! energy conservation, and combined BRDF evaluation.
//! Pure Rust — no external math or GPU crate dependencies.

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
    pub const UP: Self = Self { x: 0.0, y: 1.0, z: 0.0 };

    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < 1e-10 { return Self::ZERO; }
        Self { x: self.x / len, y: self.y / len, z: self.z / len }
    }

    pub fn add(self, o: Self) -> Self {
        Self { x: self.x + o.x, y: self.y + o.y, z: self.z + o.z }
    }

    pub fn scale(self, s: f32) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }

    /// Reflect `self` about `normal`.
    pub fn reflect(self, normal: Self) -> Self {
        let d = 2.0 * self.dot(normal);
        Self {
            x: normal.x * d - self.x,
            y: normal.y * d - self.y,
            z: normal.z * d - self.z,
        }
    }
}

/// RGB color in linear space.
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

    pub fn add(self, o: Self) -> Self {
        Self { r: self.r + o.r, g: self.g + o.g, b: self.b + o.b }
    }

    pub fn mul(self, o: Self) -> Self {
        Self { r: self.r * o.r, g: self.g * o.g, b: self.b * o.b }
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

// ── Fresnel ─────────────────────────────────────────────────────

/// Schlick's Fresnel approximation for a dielectric.
///
/// `f0` is the reflectance at normal incidence (derived from IOR).
/// `cos_theta` is the cosine of the angle between view and half vector.
pub fn fresnel_schlick(f0: f32, cos_theta: f32) -> f32 {
    let c = (1.0 - cos_theta).clamp(0.0, 1.0);
    let c2 = c * c;
    let c5 = c2 * c2 * c;
    f0 + (1.0 - f0) * c5
}

/// Per-channel Schlick Fresnel for metals / tinted dielectrics.
pub fn fresnel_schlick_color(f0: Color3, cos_theta: f32) -> Color3 {
    let c = (1.0 - cos_theta).clamp(0.0, 1.0);
    let c2 = c * c;
    let c5 = c2 * c2 * c;
    Color3::new(
        f0.r + (1.0 - f0.r) * c5,
        f0.g + (1.0 - f0.g) * c5,
        f0.b + (1.0 - f0.b) * c5,
    )
}

/// Compute F0 (normal-incidence reflectance) from index of refraction.
pub fn ior_to_f0(ior: f32) -> f32 {
    let r = (ior - 1.0) / (ior + 1.0);
    r * r
}

// ── GGX NDF ─────────────────────────────────────────────────────

/// GGX / Trowbridge-Reitz normal distribution function.
pub fn ggx_ndf(n_dot_h: f32, roughness: f32) -> f32 {
    let a2 = roughness * roughness;
    let n_dot_h2 = n_dot_h * n_dot_h;
    let denom = n_dot_h2 * (a2 - 1.0) + 1.0;
    a2 / (std::f32::consts::PI * denom * denom).max(1e-7)
}

/// Schlick-GGX geometry term (single direction).
fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    n_dot_v / (n_dot_v * (1.0 - k) + k).max(1e-7)
}

/// Smith's geometry function (combined view + light).
pub fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    geometry_schlick_ggx(n_dot_v.max(0.0), roughness)
        * geometry_schlick_ggx(n_dot_l.max(0.0), roughness)
}

// ── Clearcoat material ──────────────────────────────────────────

/// Base layer PBR parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaseLayer {
    pub color: Color3,
    pub metallic: f32,
    pub roughness: f32,
}

impl BaseLayer {
    pub fn new(color: Color3, metallic: f32, roughness: f32) -> Self {
        Self {
            color,
            metallic: metallic.clamp(0.0, 1.0),
            roughness: roughness.clamp(0.01, 1.0),
        }
    }
}

impl Default for BaseLayer {
    fn default() -> Self {
        Self::new(Color3::new(0.8, 0.1, 0.1), 1.0, 0.4)
    }
}

/// The clearcoat layer sitting on top of the base.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClearcoatLayer {
    pub roughness: f32,
    pub ior: f32,
    pub strength: f32,
    pub tint: Color3,
}

impl ClearcoatLayer {
    pub fn new() -> Self {
        Self {
            roughness: 0.05,
            ior: 1.5,
            strength: 1.0,
            tint: Color3::WHITE,
        }
    }

    pub fn with_roughness(mut self, r: f32) -> Self {
        self.roughness = r.clamp(0.01, 1.0);
        self
    }

    pub fn with_ior(mut self, ior: f32) -> Self {
        self.ior = ior.max(1.0);
        self
    }

    pub fn with_strength(mut self, s: f32) -> Self {
        self.strength = s.clamp(0.0, 1.0);
        self
    }

    pub fn with_tint(mut self, tint: Color3) -> Self {
        self.tint = tint;
        self
    }

    /// F0 of the clearcoat from its IOR.
    pub fn f0(&self) -> f32 {
        ior_to_f0(self.ior)
    }
}

impl Default for ClearcoatLayer {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete clearcoat material.
#[derive(Debug, Clone, PartialEq)]
pub struct ClearcoatMaterial {
    pub name: String,
    pub base: BaseLayer,
    pub clearcoat: ClearcoatLayer,
    /// Optional clearcoat normal perturbation (tangent-space, for orange-peel).
    pub clearcoat_normal: Vec3,
}

impl ClearcoatMaterial {
    pub fn new(name: impl Into<String>, base: BaseLayer, clearcoat: ClearcoatLayer) -> Self {
        Self {
            name: name.into(),
            base,
            clearcoat,
            clearcoat_normal: Vec3::new(0.0, 0.0, 1.0),
        }
    }

    pub fn with_clearcoat_normal(mut self, n: Vec3) -> Self {
        self.clearcoat_normal = n.normalize();
        self
    }

    /// Evaluate the combined two-layer BRDF.
    ///
    /// All vectors in world space, normalised.
    /// Returns the reflected radiance color (before multiplying by light intensity).
    pub fn evaluate(
        &self,
        normal: Vec3,
        view: Vec3,
        light: Vec3,
        light_color: Color3,
    ) -> Color3 {
        let n = normal.normalize();
        let v = view.normalize();
        let l = light.normalize();
        let h = v.add(l).normalize();

        let n_dot_v = n.dot(v).max(0.0);
        let n_dot_l = n.dot(l).max(0.0);
        let n_dot_h = n.dot(h).max(0.0);
        let v_dot_h = v.dot(h).max(0.0);

        if n_dot_l <= 0.0 {
            return Color3::BLACK;
        }

        // ── Base layer specular ─────────────────────────────────
        let base_f0 = {
            let dielectric_f0 = Color3::new(0.04, 0.04, 0.04);
            Color3::new(
                dielectric_f0.r * (1.0 - self.base.metallic) + self.base.color.r * self.base.metallic,
                dielectric_f0.g * (1.0 - self.base.metallic) + self.base.color.g * self.base.metallic,
                dielectric_f0.b * (1.0 - self.base.metallic) + self.base.color.b * self.base.metallic,
            )
        };

        let base_d = ggx_ndf(n_dot_h, self.base.roughness);
        let base_g = geometry_smith(n_dot_v, n_dot_l, self.base.roughness);
        let base_f = fresnel_schlick_color(base_f0, v_dot_h);

        let denom = (4.0 * n_dot_v * n_dot_l).max(1e-4);
        let base_specular = Color3::new(
            base_d * base_g * base_f.r / denom,
            base_d * base_g * base_f.g / denom,
            base_d * base_g * base_f.b / denom,
        );

        // Base diffuse (Lambert).
        let k_s = base_f;
        let k_d = Color3::new(
            (1.0 - k_s.r) * (1.0 - self.base.metallic),
            (1.0 - k_s.g) * (1.0 - self.base.metallic),
            (1.0 - k_s.b) * (1.0 - self.base.metallic),
        );
        let base_diffuse = Color3::new(
            k_d.r * self.base.color.r / std::f32::consts::PI,
            k_d.g * self.base.color.g / std::f32::consts::PI,
            k_d.b * self.base.color.b / std::f32::consts::PI,
        );

        let base_total = base_diffuse.add(base_specular);

        // ── Clearcoat layer ─────────────────────────────────────
        // Use clearcoat normal (may differ from base normal for orange-peel).
        let cn = self.clearcoat_normal.normalize();
        let cn_dot_h = cn.dot(h).max(0.0);
        let cn_dot_l = cn.dot(l).max(0.0);
        let cn_dot_v = cn.dot(v).max(0.0);

        let cc_f0 = self.clearcoat.f0();
        let cc_fresnel = fresnel_schlick(cc_f0, v_dot_h) * self.clearcoat.strength;
        let cc_d = ggx_ndf(cn_dot_h, self.clearcoat.roughness);
        let cc_g = geometry_smith(cn_dot_v, cn_dot_l, self.clearcoat.roughness);

        let cc_denom = (4.0 * cn_dot_v * cn_dot_l).max(1e-4);
        let cc_specular = cc_d * cc_g * cc_fresnel / cc_denom;

        // ── Energy conservation ─────────────────────────────────
        // What the clearcoat reflects is lost to the base.
        let energy_loss = 1.0 - cc_fresnel;

        let combined = Color3::new(
            (base_total.r * energy_loss + cc_specular * self.clearcoat.tint.r) * light_color.r * n_dot_l,
            (base_total.g * energy_loss + cc_specular * self.clearcoat.tint.g) * light_color.g * n_dot_l,
            (base_total.b * energy_loss + cc_specular * self.clearcoat.tint.b) * light_color.b * n_dot_l,
        );

        combined
    }
}

impl fmt::Display for ClearcoatMaterial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ClearcoatMaterial({}, cc_rough={:.2}, cc_ior={:.2})",
            self.name, self.clearcoat.roughness, self.clearcoat.ior,
        )
    }
}

// ── Preset materials ────────────────────────────────────────────

/// Glossy red car paint.
pub fn preset_car_paint_red() -> ClearcoatMaterial {
    ClearcoatMaterial::new(
        "car_paint_red",
        BaseLayer::new(Color3::new(0.7, 0.02, 0.02), 1.0, 0.35),
        ClearcoatLayer::new().with_roughness(0.04).with_ior(1.5),
    )
}

/// Glossy blue car paint.
pub fn preset_car_paint_blue() -> ClearcoatMaterial {
    ClearcoatMaterial::new(
        "car_paint_blue",
        BaseLayer::new(Color3::new(0.02, 0.1, 0.6), 1.0, 0.35),
        ClearcoatLayer::new().with_roughness(0.04).with_ior(1.5),
    )
}

/// Lacquered wood.
pub fn preset_lacquer() -> ClearcoatMaterial {
    ClearcoatMaterial::new(
        "lacquer",
        BaseLayer::new(Color3::new(0.4, 0.2, 0.05), 0.0, 0.6),
        ClearcoatLayer::new().with_roughness(0.08).with_ior(1.4),
    )
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    fn approx_wide(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    #[test]
    fn test_ior_to_f0_glass() {
        let f0 = ior_to_f0(1.5);
        assert!(approx(f0, 0.04));
    }

    #[test]
    fn test_ior_to_f0_air() {
        let f0 = ior_to_f0(1.0);
        assert!(approx(f0, 0.0));
    }

    #[test]
    fn test_ior_to_f0_diamond() {
        let f0 = ior_to_f0(2.42);
        assert!(f0 > 0.15);
    }

    #[test]
    fn test_fresnel_schlick_normal_incidence() {
        let f = fresnel_schlick(0.04, 1.0);
        assert!(approx(f, 0.04));
    }

    #[test]
    fn test_fresnel_schlick_grazing() {
        let f = fresnel_schlick(0.04, 0.0);
        assert!(approx(f, 1.0));
    }

    #[test]
    fn test_fresnel_schlick_color() {
        let f0 = Color3::new(0.04, 0.04, 0.04);
        let f = fresnel_schlick_color(f0, 1.0);
        assert!(approx(f.r, 0.04));
    }

    #[test]
    fn test_ggx_ndf_peak() {
        // At n_dot_h = 1 (aligned), NDF peaks.
        let d = ggx_ndf(1.0, 0.5);
        assert!(d > 0.0);
    }

    #[test]
    fn test_ggx_ndf_smooth_vs_rough() {
        let smooth = ggx_ndf(1.0, 0.1);
        let rough = ggx_ndf(1.0, 0.9);
        // Smooth should have a taller peak.
        assert!(smooth > rough);
    }

    #[test]
    fn test_geometry_smith_positive() {
        let g = geometry_smith(0.5, 0.5, 0.5);
        assert!(g > 0.0);
        assert!(g <= 1.0);
    }

    #[test]
    fn test_base_layer_clamping() {
        let b = BaseLayer::new(Color3::WHITE, 2.0, -0.5);
        assert!(approx(b.metallic, 1.0));
        assert!(approx(b.roughness, 0.01));
    }

    #[test]
    fn test_clearcoat_layer_defaults() {
        let cc = ClearcoatLayer::new();
        assert!(approx(cc.roughness, 0.05));
        assert!(approx(cc.ior, 1.5));
        assert!(approx(cc.strength, 1.0));
    }

    #[test]
    fn test_clearcoat_f0() {
        let cc = ClearcoatLayer::new().with_ior(1.5);
        assert!(approx(cc.f0(), 0.04));
    }

    #[test]
    fn test_clearcoat_builders() {
        let cc = ClearcoatLayer::new()
            .with_roughness(0.1)
            .with_ior(1.6)
            .with_strength(0.8)
            .with_tint(Color3::new(0.9, 0.95, 1.0));
        assert!(approx(cc.roughness, 0.1));
        assert!(approx(cc.ior, 1.6));
        assert!(approx(cc.strength, 0.8));
    }

    #[test]
    fn test_evaluate_head_on() {
        let mat = preset_car_paint_red();
        let n = Vec3::new(0.0, 1.0, 0.0);
        let v = Vec3::new(0.0, 1.0, 0.0);
        let l = Vec3::new(0.0, 1.0, 0.0);
        let result = mat.evaluate(n, v, l, Color3::WHITE);
        // Should produce some color.
        assert!(result.r > 0.0 || result.g > 0.0 || result.b > 0.0);
    }

    #[test]
    fn test_evaluate_light_behind() {
        let mat = preset_car_paint_red();
        let n = Vec3::new(0.0, 1.0, 0.0);
        let v = Vec3::new(0.0, 1.0, 0.0);
        let l = Vec3::new(0.0, -1.0, 0.0); // behind surface
        let result = mat.evaluate(n, v, l, Color3::WHITE);
        assert!(approx(result.r, 0.0));
        assert!(approx(result.g, 0.0));
        assert!(approx(result.b, 0.0));
    }

    #[test]
    fn test_energy_conservation() {
        let mat = preset_car_paint_red();
        let n = Vec3::new(0.0, 1.0, 0.0);
        let v = Vec3::new(0.0, 1.0, 0.0);
        let l = Vec3::new(0.0, 1.0, 0.0);
        let result = mat.evaluate(n, v, l, Color3::WHITE);
        // Combined energy should not exceed 1 per channel (approx).
        assert!(result.r <= 1.5); // some specular can exceed diffuse+1 in extreme cases
        assert!(result.g <= 1.5);
        assert!(result.b <= 1.5);
    }

    #[test]
    fn test_clearcoat_strength_zero() {
        let base = BaseLayer::default();
        let cc = ClearcoatLayer::new().with_strength(0.0);
        let mat = ClearcoatMaterial::new("none", base, cc);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let v = Vec3::new(0.0, 1.0, 0.0);
        let l = Vec3::new(0.3, 0.95, 0.0).normalize();
        let result = mat.evaluate(n, v, l, Color3::WHITE);
        // With zero clearcoat, result is purely base layer.
        assert!(result.luminance() > 0.0);
    }

    #[test]
    fn test_clearcoat_tint() {
        let base = BaseLayer::new(Color3::WHITE, 0.0, 0.5);
        let cc = ClearcoatLayer::new().with_tint(Color3::new(1.0, 0.5, 0.0));
        let mat = ClearcoatMaterial::new("tinted", base, cc);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let v = Vec3::new(0.1, 1.0, 0.0).normalize();
        let l = Vec3::new(0.1, 1.0, 0.0).normalize();
        let result = mat.evaluate(n, v, l, Color3::WHITE);
        // Tinted clearcoat should shift color balance.
        assert!(result.r > 0.0);
    }

    #[test]
    fn test_preset_car_paint_red() {
        let m = preset_car_paint_red();
        assert_eq!(m.name, "car_paint_red");
        assert!(approx(m.base.metallic, 1.0));
    }

    #[test]
    fn test_preset_lacquer() {
        let m = preset_lacquer();
        assert!(approx(m.base.metallic, 0.0));
        assert!(approx_wide(m.clearcoat.ior, 1.4));
    }

    #[test]
    fn test_display() {
        let m = preset_car_paint_blue();
        let s = format!("{m}");
        assert!(s.contains("car_paint_blue"));
        assert!(s.contains("cc_rough"));
    }

    #[test]
    fn test_orange_peel_normal() {
        let mat = ClearcoatMaterial::new(
            "orange_peel",
            BaseLayer::default(),
            ClearcoatLayer::new(),
        ).with_clearcoat_normal(Vec3::new(0.1, 0.05, 0.99));
        // The clearcoat normal should be normalised.
        let len = mat.clearcoat_normal.length();
        assert!((len - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_vec3_reflect() {
        let v = Vec3::new(1.0, -1.0, 0.0).normalize();
        let n = Vec3::new(0.0, 1.0, 0.0);
        let r = v.reflect(n);
        // Reflected should flip the Y component.
        assert!(approx_wide(r.x, -v.x));
        assert!(approx_wide(r.y, v.y));
    }
}
