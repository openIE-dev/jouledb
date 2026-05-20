//! Physically-based rendering material.
//!
//! Base color (RGBA), metallic/roughness workflow, normal map strength,
//! emissive (RGB + intensity), alpha modes (Opaque/Mask/Blend), double-sided
//! flag, uniform block layout, sort-key generation, and serde round-trip.
//! Pure Rust — no external math or GPU crate dependencies.

use std::fmt;

// ── Inline vector / color types ─────────────────────────────────

/// RGBA color in linear space, each channel 0.0..=1.0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const WHITE: Self = Self { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const BLACK: Self = Self { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const TRANSPARENT: Self = Self { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    /// Clamp all channels to 0..=1.
    pub fn saturate(self) -> Self {
        Self {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
            a: self.a.clamp(0.0, 1.0),
        }
    }

    /// Luminance via Rec. 709 coefficients.
    pub fn luminance(&self) -> f32 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }
}

impl Default for Color {
    fn default() -> Self {
        Self::WHITE
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.3}, {:.3}, {:.3}, {:.3})", self.r, self.g, self.b, self.a)
    }
}

// ── Alpha mode ──────────────────────────────────────────────────

/// How the fragment alpha is interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlphaMode {
    /// Fully opaque — alpha ignored.
    Opaque,
    /// Binary mask — discard if alpha < cutoff.
    Mask,
    /// Standard alpha blending.
    Blend,
}

impl AlphaMode {
    /// Sort priority (lower draws first). Opaque before Mask before Blend.
    pub fn sort_priority(&self) -> u8 {
        match self {
            AlphaMode::Opaque => 0,
            AlphaMode::Mask => 1,
            AlphaMode::Blend => 2,
        }
    }
}

impl Default for AlphaMode {
    fn default() -> Self {
        AlphaMode::Opaque
    }
}

impl fmt::Display for AlphaMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            AlphaMode::Opaque => "Opaque",
            AlphaMode::Mask => "Mask",
            AlphaMode::Blend => "Blend",
        };
        write!(f, "{s}")
    }
}

// ── Emissive ────────────────────────────────────────────────────

/// Emissive channel: RGB color times scalar intensity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Emissive {
    pub color: Color,
    pub intensity: f32,
}

impl Emissive {
    pub const OFF: Self = Self { color: Color::BLACK, intensity: 0.0 };

    pub fn new(r: f32, g: f32, b: f32, intensity: f32) -> Self {
        Self { color: Color::rgb(r, g, b), intensity }
    }

    /// Effective emissive contribution (color * intensity).
    pub fn contribution(&self) -> (f32, f32, f32) {
        (
            self.color.r * self.intensity,
            self.color.g * self.intensity,
            self.color.b * self.intensity,
        )
    }
}

impl Default for Emissive {
    fn default() -> Self {
        Self::OFF
    }
}

// ── PBR Material ────────────────────────────────────────────────

/// A physically-based rendering material (metallic/roughness workflow).
#[derive(Debug, Clone, PartialEq)]
pub struct PbrMaterial {
    pub name: String,
    pub base_color: Color,
    pub metallic: f32,
    pub roughness: f32,
    pub normal_map_strength: f32,
    pub emissive: Emissive,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
}

impl PbrMaterial {
    /// Create a default opaque white material.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            base_color: Color::WHITE,
            metallic: 0.0,
            roughness: 0.5,
            normal_map_strength: 1.0,
            emissive: Emissive::OFF,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
        }
    }

    /// Builder — set base color.
    pub fn with_base_color(mut self, color: Color) -> Self {
        self.base_color = color;
        self
    }

    /// Builder — set metallic factor (clamped 0..=1).
    pub fn with_metallic(mut self, m: f32) -> Self {
        self.metallic = m.clamp(0.0, 1.0);
        self
    }

    /// Builder — set roughness factor (clamped 0..=1).
    pub fn with_roughness(mut self, r: f32) -> Self {
        self.roughness = r.clamp(0.0, 1.0);
        self
    }

    /// Builder — set emissive.
    pub fn with_emissive(mut self, e: Emissive) -> Self {
        self.emissive = e;
        self
    }

    /// Builder — set alpha mode and optional cutoff.
    pub fn with_alpha(mut self, mode: AlphaMode, cutoff: f32) -> Self {
        self.alpha_mode = mode;
        self.alpha_cutoff = cutoff.clamp(0.0, 1.0);
        self
    }

    /// Builder — set double-sided.
    pub fn with_double_sided(mut self, ds: bool) -> Self {
        self.double_sided = ds;
        self
    }

    /// Builder — normal map strength.
    pub fn with_normal_strength(mut self, s: f32) -> Self {
        self.normal_map_strength = s.clamp(0.0, 2.0);
        self
    }

    /// Whether the fragment at given alpha should be discarded.
    pub fn should_discard(&self, alpha: f32) -> bool {
        match self.alpha_mode {
            AlphaMode::Opaque => false,
            AlphaMode::Mask => alpha < self.alpha_cutoff,
            AlphaMode::Blend => false,
        }
    }

    /// Effective alpha for the fragment.
    pub fn effective_alpha(&self, alpha: f32) -> f32 {
        match self.alpha_mode {
            AlphaMode::Opaque => 1.0,
            AlphaMode::Mask => {
                if alpha >= self.alpha_cutoff { 1.0 } else { 0.0 }
            }
            AlphaMode::Blend => alpha.clamp(0.0, 1.0),
        }
    }

    // ── Uniform block ───────────────────────────────────────────

    /// Pack material into a uniform-block-style f32 array (std140 layout).
    ///
    /// Layout (16 floats, 64 bytes):
    ///   [0..4]   base_color RGBA
    ///   [4..8]   metallic, roughness, normal_strength, alpha_cutoff
    ///   [8..12]  emissive_r, emissive_g, emissive_b, emissive_intensity
    ///   [12..16] alpha_mode (as float), double_sided (0/1), pad, pad
    pub fn to_uniform_block(&self) -> [f32; 16] {
        let mut buf = [0.0f32; 16];
        buf[0] = self.base_color.r;
        buf[1] = self.base_color.g;
        buf[2] = self.base_color.b;
        buf[3] = self.base_color.a;
        buf[4] = self.metallic;
        buf[5] = self.roughness;
        buf[6] = self.normal_map_strength;
        buf[7] = self.alpha_cutoff;
        buf[8] = self.emissive.color.r;
        buf[9] = self.emissive.color.g;
        buf[10] = self.emissive.color.b;
        buf[11] = self.emissive.intensity;
        buf[12] = self.alpha_mode.sort_priority() as f32;
        buf[13] = if self.double_sided { 1.0 } else { 0.0 };
        buf[14] = 0.0;
        buf[15] = 0.0;
        buf
    }

    /// Restore material properties from a uniform block (name defaults to empty).
    pub fn from_uniform_block(buf: &[f32; 16]) -> Self {
        let alpha_mode = match buf[12] as u8 {
            0 => AlphaMode::Opaque,
            1 => AlphaMode::Mask,
            _ => AlphaMode::Blend,
        };
        Self {
            name: String::new(),
            base_color: Color::new(buf[0], buf[1], buf[2], buf[3]),
            metallic: buf[4],
            roughness: buf[5],
            normal_map_strength: buf[6],
            emissive: Emissive {
                color: Color::rgb(buf[8], buf[9], buf[10]),
                intensity: buf[11],
            },
            alpha_mode,
            alpha_cutoff: buf[7],
            double_sided: buf[13] > 0.5,
        }
    }

    // ── Sort key ────────────────────────────────────────────────

    /// Generate a 64-bit sort key for draw-call ordering.
    ///
    /// Bits (MSB→LSB):
    ///   [63..62] alpha_mode priority (2 bits)
    ///   [61..45] depth (16 bits, quantised)
    ///   [44..0]  material id hash (45 bits)
    pub fn sort_key(&self, depth: f32, max_depth: f32) -> u64 {
        let alpha_bits = (self.alpha_mode.sort_priority() as u64) << 62;
        let d = if max_depth > 0.0 { (depth / max_depth).clamp(0.0, 1.0) } else { 0.0 };
        let depth_bits = ((d * 65535.0) as u64) << 45;
        let name_hash = {
            let mut h: u64 = 5381;
            for b in self.name.bytes() {
                h = h.wrapping_mul(33).wrapping_add(b as u64);
            }
            h & 0x1FFF_FFFF_FFFF // 45 bits
        };
        alpha_bits | depth_bits | name_hash
    }

    // ── Serialisation (JSON-like, serde-free) ───────────────────

    /// Serialise to a JSON-compatible string.
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"name":"{}","base_color":[{},{},{},{}],"metallic":{},"roughness":{},"normal_map_strength":{},"emissive":[{},{},{},{}],"alpha_mode":"{}","alpha_cutoff":{},"double_sided":{}}}"#,
            self.name,
            self.base_color.r, self.base_color.g, self.base_color.b, self.base_color.a,
            self.metallic,
            self.roughness,
            self.normal_map_strength,
            self.emissive.color.r, self.emissive.color.g, self.emissive.color.b, self.emissive.intensity,
            self.alpha_mode,
            self.alpha_cutoff,
            self.double_sided,
        )
    }

    /// Deserialise from JSON-compatible string (minimal parser).
    pub fn from_json(s: &str) -> Result<Self, String> {
        fn extract_str<'a>(s: &'a str, key: &str) -> Result<&'a str, String> {
            let pattern = format!("\"{}\":\"", key);
            let start = s.find(&pattern)
                .ok_or_else(|| format!("missing key: {key}"))?
                + pattern.len();
            let end = s[start..].find('"').ok_or_else(|| format!("unterminated string for {key}"))? + start;
            Ok(&s[start..end])
        }
        fn extract_array(s: &str, key: &str) -> Result<Vec<f32>, String> {
            let pattern = format!("\"{}\":[", key);
            let start = s.find(&pattern)
                .ok_or_else(|| format!("missing key: {key}"))?
                + pattern.len();
            let end = s[start..].find(']').ok_or_else(|| format!("unterminated array for {key}"))? + start;
            s[start..end]
                .split(',')
                .map(|v| v.trim().parse::<f32>().map_err(|e| format!("bad float in {key}: {e}")))
                .collect()
        }
        fn extract_f32(s: &str, key: &str) -> Result<f32, String> {
            let pattern = format!("\"{}\":", key);
            let start = s.find(&pattern)
                .ok_or_else(|| format!("missing key: {key}"))?
                + pattern.len();
            let rest = &s[start..];
            let end = rest.find(|c: char| c == ',' || c == '}').unwrap_or(rest.len());
            rest[..end].trim().parse::<f32>().map_err(|e| format!("bad float for {key}: {e}"))
        }
        fn extract_bool(s: &str, key: &str) -> Result<bool, String> {
            let pattern = format!("\"{}\":", key);
            let start = s.find(&pattern)
                .ok_or_else(|| format!("missing key: {key}"))?
                + pattern.len();
            let rest = &s[start..];
            let end = rest.find(|c: char| c == ',' || c == '}').unwrap_or(rest.len());
            match rest[..end].trim() {
                "true" => Ok(true),
                "false" => Ok(false),
                other => Err(format!("bad bool for {key}: {other}")),
            }
        }

        let name = extract_str(s, "name")?.to_owned();
        let bc = extract_array(s, "base_color")?;
        if bc.len() != 4 { return Err("base_color needs 4 elements".into()); }
        let em = extract_array(s, "emissive")?;
        if em.len() != 4 { return Err("emissive needs 4 elements".into()); }
        let alpha_str = extract_str(s, "alpha_mode")?;
        let alpha_mode = match alpha_str {
            "Opaque" => AlphaMode::Opaque,
            "Mask" => AlphaMode::Mask,
            "Blend" => AlphaMode::Blend,
            other => return Err(format!("unknown alpha_mode: {other}")),
        };

        Ok(Self {
            name,
            base_color: Color::new(bc[0], bc[1], bc[2], bc[3]),
            metallic: extract_f32(s, "metallic")?,
            roughness: extract_f32(s, "roughness")?,
            normal_map_strength: extract_f32(s, "normal_map_strength")?,
            emissive: Emissive { color: Color::rgb(em[0], em[1], em[2]), intensity: em[3] },
            alpha_mode,
            alpha_cutoff: extract_f32(s, "alpha_cutoff")?,
            double_sided: extract_bool(s, "double_sided")?,
        })
    }
}

impl Default for PbrMaterial {
    fn default() -> Self {
        Self::new("default")
    }
}

impl fmt::Display for PbrMaterial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PbrMaterial({}, metal={:.2}, rough={:.2}, alpha={})",
            self.name, self.metallic, self.roughness, self.alpha_mode,
        )
    }
}

// ── Material comparison helper ──────────────────────────────────

/// Compare two materials for batching equivalence (ignoring name).
pub fn materials_equivalent(a: &PbrMaterial, b: &PbrMaterial) -> bool {
    fn color_eq(c1: &Color, c2: &Color) -> bool {
        (c1.r - c2.r).abs() < 1e-6
            && (c1.g - c2.g).abs() < 1e-6
            && (c1.b - c2.b).abs() < 1e-6
            && (c1.a - c2.a).abs() < 1e-6
    }
    color_eq(&a.base_color, &b.base_color)
        && (a.metallic - b.metallic).abs() < 1e-6
        && (a.roughness - b.roughness).abs() < 1e-6
        && (a.normal_map_strength - b.normal_map_strength).abs() < 1e-6
        && color_eq(&a.emissive.color, &b.emissive.color)
        && (a.emissive.intensity - b.emissive.intensity).abs() < 1e-6
        && a.alpha_mode == b.alpha_mode
        && (a.alpha_cutoff - b.alpha_cutoff).abs() < 1e-6
        && a.double_sided == b.double_sided
}

// ── Preset materials ────────────────────────────────────────────

/// Standard gold PBR preset.
pub fn preset_gold() -> PbrMaterial {
    PbrMaterial::new("gold")
        .with_base_color(Color::rgb(1.0, 0.766, 0.336))
        .with_metallic(1.0)
        .with_roughness(0.3)
}

/// Standard plastic PBR preset.
pub fn preset_plastic_red() -> PbrMaterial {
    PbrMaterial::new("plastic_red")
        .with_base_color(Color::rgb(0.8, 0.05, 0.05))
        .with_metallic(0.0)
        .with_roughness(0.4)
}

/// Standard glass-like PBR preset.
pub fn preset_glass() -> PbrMaterial {
    PbrMaterial::new("glass")
        .with_base_color(Color::new(0.95, 0.95, 0.95, 0.3))
        .with_metallic(0.0)
        .with_roughness(0.05)
        .with_alpha(AlphaMode::Blend, 0.5)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn test_color_new_and_rgb() {
        let c = Color::new(0.1, 0.2, 0.3, 0.4);
        assert!(approx(c.r, 0.1));
        assert!(approx(c.a, 0.4));
        let c2 = Color::rgb(0.5, 0.6, 0.7);
        assert!(approx(c2.a, 1.0));
    }

    #[test]
    fn test_color_saturate() {
        let c = Color::new(-0.5, 1.5, 0.5, 2.0).saturate();
        assert!(approx(c.r, 0.0));
        assert!(approx(c.g, 1.0));
        assert!(approx(c.b, 0.5));
        assert!(approx(c.a, 1.0));
    }

    #[test]
    fn test_color_luminance() {
        let white = Color::WHITE;
        assert!(approx(white.luminance(), 1.0));
        let black = Color::BLACK;
        assert!(approx(black.luminance(), 0.0));
    }

    #[test]
    fn test_default_material() {
        let m = PbrMaterial::default();
        assert_eq!(m.name, "default");
        assert!(approx(m.metallic, 0.0));
        assert!(approx(m.roughness, 0.5));
        assert_eq!(m.alpha_mode, AlphaMode::Opaque);
        assert!(!m.double_sided);
    }

    #[test]
    fn test_builder_chain() {
        let m = PbrMaterial::new("test")
            .with_base_color(Color::rgb(1.0, 0.0, 0.0))
            .with_metallic(0.8)
            .with_roughness(0.2)
            .with_double_sided(true)
            .with_normal_strength(1.5);
        assert!(approx(m.metallic, 0.8));
        assert!(approx(m.roughness, 0.2));
        assert!(m.double_sided);
        assert!(approx(m.normal_map_strength, 1.5));
    }

    #[test]
    fn test_clamping() {
        let m = PbrMaterial::new("c")
            .with_metallic(5.0)
            .with_roughness(-1.0)
            .with_normal_strength(10.0);
        assert!(approx(m.metallic, 1.0));
        assert!(approx(m.roughness, 0.0));
        assert!(approx(m.normal_map_strength, 2.0));
    }

    #[test]
    fn test_alpha_opaque_never_discards() {
        let m = PbrMaterial::new("a");
        assert!(!m.should_discard(0.0));
        assert!(!m.should_discard(1.0));
    }

    #[test]
    fn test_alpha_mask_discard() {
        let m = PbrMaterial::new("a").with_alpha(AlphaMode::Mask, 0.5);
        assert!(m.should_discard(0.3));
        assert!(!m.should_discard(0.7));
        assert!(!m.should_discard(0.5));
    }

    #[test]
    fn test_effective_alpha_opaque() {
        let m = PbrMaterial::new("a");
        assert!(approx(m.effective_alpha(0.3), 1.0));
    }

    #[test]
    fn test_effective_alpha_mask() {
        let m = PbrMaterial::new("a").with_alpha(AlphaMode::Mask, 0.5);
        assert!(approx(m.effective_alpha(0.6), 1.0));
        assert!(approx(m.effective_alpha(0.3), 0.0));
    }

    #[test]
    fn test_effective_alpha_blend() {
        let m = PbrMaterial::new("a").with_alpha(AlphaMode::Blend, 0.5);
        assert!(approx(m.effective_alpha(0.7), 0.7));
    }

    #[test]
    fn test_emissive_contribution() {
        let e = Emissive::new(1.0, 0.5, 0.0, 3.0);
        let (r, g, b) = e.contribution();
        assert!(approx(r, 3.0));
        assert!(approx(g, 1.5));
        assert!(approx(b, 0.0));
    }

    #[test]
    fn test_uniform_block_round_trip() {
        let m = PbrMaterial::new("rt")
            .with_base_color(Color::new(0.1, 0.2, 0.3, 0.9))
            .with_metallic(0.7)
            .with_roughness(0.4)
            .with_emissive(Emissive::new(0.5, 0.6, 0.7, 2.0))
            .with_alpha(AlphaMode::Mask, 0.6)
            .with_double_sided(true);
        let block = m.to_uniform_block();
        let m2 = PbrMaterial::from_uniform_block(&block);
        assert!(approx(m2.base_color.r, 0.1));
        assert!(approx(m2.metallic, 0.7));
        assert_eq!(m2.alpha_mode, AlphaMode::Mask);
        assert!(m2.double_sided);
    }

    #[test]
    fn test_sort_key_alpha_ordering() {
        let opaque = PbrMaterial::new("a");
        let blend = PbrMaterial::new("a").with_alpha(AlphaMode::Blend, 0.5);
        assert!(opaque.sort_key(1.0, 100.0) < blend.sort_key(1.0, 100.0));
    }

    #[test]
    fn test_sort_key_depth_ordering() {
        let m = PbrMaterial::new("a");
        let near = m.sort_key(1.0, 100.0);
        let far = m.sort_key(50.0, 100.0);
        assert!(near < far);
    }

    #[test]
    fn test_json_round_trip() {
        let m = PbrMaterial::new("roundtrip")
            .with_base_color(Color::new(0.1, 0.2, 0.3, 1.0))
            .with_metallic(0.5)
            .with_roughness(0.6)
            .with_emissive(Emissive::new(0.0, 1.0, 0.0, 5.0))
            .with_alpha(AlphaMode::Blend, 0.4)
            .with_double_sided(true);
        let json = m.to_json();
        let m2 = PbrMaterial::from_json(&json).unwrap();
        assert_eq!(m2.name, "roundtrip");
        assert!(approx(m2.metallic, 0.5));
        assert_eq!(m2.alpha_mode, AlphaMode::Blend);
        assert!(m2.double_sided);
    }

    #[test]
    fn test_json_parse_error() {
        let bad = r#"{"name":"x"}"#;
        assert!(PbrMaterial::from_json(bad).is_err());
    }

    #[test]
    fn test_materials_equivalent_same() {
        let a = preset_gold();
        let b = preset_gold();
        assert!(materials_equivalent(&a, &b));
    }

    #[test]
    fn test_materials_equivalent_diff_name() {
        let a = preset_gold();
        let mut b = preset_gold();
        b.name = "other".into();
        assert!(materials_equivalent(&a, &b));
    }

    #[test]
    fn test_materials_not_equivalent() {
        let a = preset_gold();
        let b = preset_plastic_red();
        assert!(!materials_equivalent(&a, &b));
    }

    #[test]
    fn test_preset_gold() {
        let g = preset_gold();
        assert!(approx(g.metallic, 1.0));
        assert!(g.roughness > 0.0);
    }

    #[test]
    fn test_preset_glass_alpha() {
        let g = preset_glass();
        assert_eq!(g.alpha_mode, AlphaMode::Blend);
        assert!(g.base_color.a < 1.0);
    }

    #[test]
    fn test_display() {
        let m = PbrMaterial::new("show");
        let s = format!("{m}");
        assert!(s.contains("show"));
        assert!(s.contains("metal="));
    }

    #[test]
    fn test_alpha_mode_display() {
        assert_eq!(format!("{}", AlphaMode::Opaque), "Opaque");
        assert_eq!(format!("{}", AlphaMode::Mask), "Mask");
        assert_eq!(format!("{}", AlphaMode::Blend), "Blend");
    }

    #[test]
    fn test_sort_key_zero_depth() {
        let m = PbrMaterial::new("z");
        let k = m.sort_key(0.0, 0.0);
        // Should not panic and alpha bits should be 0 (opaque).
        assert!(k < (1u64 << 62));
    }
}
