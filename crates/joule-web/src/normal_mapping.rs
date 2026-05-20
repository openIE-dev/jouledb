//! Normal map processing.
//!
//! Tangent-space to world-space normal conversion via TBN matrix, tangent
//! generation from triangle UVs (Lengyel's method), height-map to normal-map
//! conversion (central differences), normal blending (UDN, RNM, linear),
//! and normal strength scaling. Pure Rust — no external math dependencies.

use std::fmt;

// ── Inline Vec3 ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const UP: Self = Self { x: 0.0, y: 1.0, z: 0.0 };
    pub const FORWARD: Self = Self { x: 0.0, y: 0.0, z: 1.0 };

    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < 1e-10 {
            return Self::ZERO;
        }
        Self { x: self.x / len, y: self.y / len, z: self.z / len }
    }

    pub fn scale(self, s: f32) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }

    pub fn add(self, other: Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }

    pub fn sub(self, other: Self) -> Self {
        Self { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }
}

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4}, {:.4})", self.x, self.y, self.z)
    }
}

// ── Vec2 for UVs ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn sub(self, other: Self) -> Self {
        Self { x: self.x - other.x, y: self.y - other.y }
    }
}

// ── TBN Matrix ──────────────────────────────────────────────────

/// Tangent-Bitangent-Normal matrix for tangent-space to world-space conversion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TbnMatrix {
    pub tangent: Vec3,
    pub bitangent: Vec3,
    pub normal: Vec3,
}

impl TbnMatrix {
    pub fn new(tangent: Vec3, bitangent: Vec3, normal: Vec3) -> Self {
        Self {
            tangent: tangent.normalize(),
            bitangent: bitangent.normalize(),
            normal: normal.normalize(),
        }
    }

    /// Construct from normal only — generates an arbitrary tangent frame.
    pub fn from_normal(normal: Vec3) -> Self {
        let n = normal.normalize();
        // Choose a vector not parallel to n.
        let up = if n.y.abs() < 0.999 { Vec3::UP } else { Vec3::FORWARD };
        let tangent = up.cross(n).normalize();
        let bitangent = n.cross(tangent).normalize();
        Self { tangent, bitangent, normal: n }
    }

    /// Transform tangent-space normal to world space.
    pub fn transform(&self, ts_normal: Vec3) -> Vec3 {
        Vec3::new(
            self.tangent.x * ts_normal.x + self.bitangent.x * ts_normal.y + self.normal.x * ts_normal.z,
            self.tangent.y * ts_normal.x + self.bitangent.y * ts_normal.y + self.normal.y * ts_normal.z,
            self.tangent.z * ts_normal.x + self.bitangent.z * ts_normal.y + self.normal.z * ts_normal.z,
        ).normalize()
    }
}

// ── Tangent generation (Lengyel's method) ───────────────────────

/// Triangle vertex data for tangent computation.
#[derive(Debug, Clone, Copy)]
pub struct TriangleVertex {
    pub position: Vec3,
    pub uv: Vec2,
    pub normal: Vec3,
}

/// Compute tangent and bitangent for a triangle (Lengyel's method).
///
/// Returns (tangent, bitangent) averaged and orthogonalised against `normal`.
pub fn compute_tangent(v0: &TriangleVertex, v1: &TriangleVertex, v2: &TriangleVertex) -> (Vec3, Vec3) {
    let edge1 = v1.position.sub(v0.position);
    let edge2 = v2.position.sub(v0.position);
    let duv1 = v1.uv.sub(v0.uv);
    let duv2 = v2.uv.sub(v0.uv);

    let denom = duv1.x * duv2.y - duv2.x * duv1.y;
    let r = if denom.abs() < 1e-10 { 1.0 } else { 1.0 / denom };

    let tangent = Vec3::new(
        (duv2.y * edge1.x - duv1.y * edge2.x) * r,
        (duv2.y * edge1.y - duv1.y * edge2.y) * r,
        (duv2.y * edge1.z - duv1.y * edge2.z) * r,
    );

    let bitangent = Vec3::new(
        (duv1.x * edge2.x - duv2.x * edge1.x) * r,
        (duv1.x * edge2.y - duv2.x * edge1.y) * r,
        (duv1.x * edge2.z - duv2.x * edge1.z) * r,
    );

    // Gram-Schmidt orthogonalise tangent against the normal.
    let n = v0.normal.normalize();
    let t = tangent.sub(n.scale(n.dot(tangent))).normalize();

    // Calculate handedness and fix bitangent.
    let cross = n.cross(t);
    let handedness = if cross.dot(bitangent) < 0.0 { -1.0 } else { 1.0 };
    let b = cross.scale(handedness);

    (t, b)
}

// ── Height map → Normal map ─────────────────────────────────────

/// Height map: 2D grid of f32 heights in 0..1.
#[derive(Debug, Clone)]
pub struct HeightMap {
    pub width: u32,
    pub height: u32,
    data: Vec<f32>,
}

impl HeightMap {
    pub fn new(width: u32, height: u32, data: Vec<f32>) -> Result<Self, String> {
        let expected = (width as usize) * (height as usize);
        if data.len() != expected {
            return Err(format!("expected {} samples, got {}", expected, data.len()));
        }
        Ok(Self { width, height, data })
    }

    /// Flat (all zero) height map.
    pub fn flat(width: u32, height: u32) -> Self {
        let count = (width as usize) * (height as usize);
        Self { width, height, data: vec![0.0; count] }
    }

    pub fn get(&self, x: u32, y: u32) -> f32 {
        let x = x.min(self.width.saturating_sub(1));
        let y = y.min(self.height.saturating_sub(1));
        self.data[(y * self.width + x) as usize]
    }

    pub fn set(&mut self, x: u32, y: u32, v: f32) {
        if x < self.width && y < self.height {
            self.data[(y * self.width + x) as usize] = v;
        }
    }

    /// Convert to a normal map using central differences.
    ///
    /// Returns normals in tangent space (z-up), each as (x, y, z).
    /// `strength` scales the XY derivatives (higher = more pronounced bumps).
    pub fn to_normal_map(&self, strength: f32) -> Vec<Vec3> {
        let w = self.width;
        let h = self.height;
        let mut normals = Vec::with_capacity((w * h) as usize);

        for y in 0..h {
            for x in 0..w {
                let left = self.get(x.saturating_sub(1), y);
                let right = self.get((x + 1).min(w - 1), y);
                let down = self.get(x, y.saturating_sub(1));
                let up = self.get(x, (y + 1).min(h - 1));

                let dx = (right - left) * strength;
                let dy = (up - down) * strength;

                normals.push(Vec3::new(-dx, -dy, 1.0).normalize());
            }
        }
        normals
    }
}

// ── Normal map blending ─────────────────────────────────────────

/// Blending method for combining two normal maps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMethod {
    /// Linear interpolation (simple but flattens).
    Linear,
    /// Unreal/partial derivative (UDN).
    Udn,
    /// Reoriented Normal Mapping (most accurate).
    Rnm,
}

/// Blend two tangent-space normals.
///
/// `base` and `detail` should be normalised tangent-space normals.
/// `weight` controls detail strength (0 = base only, 1 = full detail).
pub fn blend_normals(base: Vec3, detail: Vec3, method: BlendMethod, weight: f32) -> Vec3 {
    let w = weight.clamp(0.0, 1.0);
    match method {
        BlendMethod::Linear => {
            // Lerp and renormalise.
            let blended = Vec3::new(
                base.x + detail.x * w,
                base.y + detail.y * w,
                base.z + detail.z * w,
            );
            blended.normalize()
        }
        BlendMethod::Udn => {
            // Partial derivative addition — drop detail.z, reconstruct.
            Vec3::new(
                base.x + detail.x * w,
                base.y + detail.y * w,
                base.z,
            ).normalize()
        }
        BlendMethod::Rnm => {
            // Reoriented Normal Mapping.
            // t = base.xyz * vec3(1,1,1) + vec3(0,0,1)
            // u = detail.xyz * vec3(-1,-1,1) + vec3(0,0,0)
            // result = t * dot(t,u) / t.z - u
            let t = Vec3::new(base.x, base.y, base.z + 1.0);
            let u = Vec3::new(-detail.x * w, -detail.y * w, detail.z);
            let d = t.dot(u);
            let tz = if t.z.abs() < 1e-10 { 1e-10 } else { t.z };
            Vec3::new(
                t.x * d / tz - u.x,
                t.y * d / tz - u.y,
                t.z * d / tz - u.z,
            ).normalize()
        }
    }
}

// ── Normal strength scaling ─────────────────────────────────────

/// Scale the strength of a tangent-space normal.
/// `strength` = 1 keeps original, 0 flattens to (0,0,1).
pub fn scale_normal_strength(normal: Vec3, strength: f32) -> Vec3 {
    Vec3::new(
        normal.x * strength,
        normal.y * strength,
        normal.z, // keep Z component unchanged
    ).normalize()
}

/// Decode a packed normal map texel (0..255 → -1..1 range).
pub fn decode_normal(r: u8, g: u8, b: u8) -> Vec3 {
    let from_byte = |v: u8| -> f32 { (v as f32 - 128.0) / 127.0 };
    Vec3::new(
        from_byte(r),
        from_byte(g),
        from_byte(b),
    ).normalize()
}

/// Encode a tangent-space normal to packed 0..255 range.
pub fn encode_normal(n: Vec3) -> (u8, u8, u8) {
    let to_byte = |v: f32| -> u8 { (v * 127.0 + 128.0).round().clamp(0.0, 255.0) as u8 };
    (to_byte(n.x), to_byte(n.y), to_byte(n.z))
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    fn is_unit(v: Vec3) -> bool {
        (v.length() - 1.0).abs() < 1e-4
    }

    #[test]
    fn test_vec3_dot() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        assert!(approx(a.dot(b), 0.0));
    }

    #[test]
    fn test_vec3_cross() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        let z = x.cross(y);
        assert!(approx(z.x, 0.0));
        assert!(approx(z.y, 0.0));
        assert!(approx(z.z, 1.0));
    }

    #[test]
    fn test_vec3_normalize() {
        let v = Vec3::new(3.0, 4.0, 0.0).normalize();
        assert!(approx(v.x, 0.6));
        assert!(approx(v.y, 0.8));
    }

    #[test]
    fn test_vec3_zero_normalize() {
        let v = Vec3::ZERO.normalize();
        assert!(approx(v.length(), 0.0));
    }

    #[test]
    fn test_tbn_identity() {
        let tbn = TbnMatrix::new(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let ts = Vec3::new(0.0, 0.0, 1.0);
        let ws = tbn.transform(ts);
        assert!(approx(ws.z, 1.0));
    }

    #[test]
    fn test_tbn_from_normal() {
        let n = Vec3::new(0.0, 0.0, 1.0);
        let tbn = TbnMatrix::from_normal(n);
        assert!(is_unit(tbn.tangent));
        assert!(is_unit(tbn.bitangent));
        assert!(is_unit(tbn.normal));
        // T, B, N should be orthogonal.
        assert!(tbn.tangent.dot(tbn.normal).abs() < 1e-4);
        assert!(tbn.bitangent.dot(tbn.normal).abs() < 1e-4);
    }

    #[test]
    fn test_tbn_from_normal_up() {
        // Edge case: normal pointing straight up.
        let tbn = TbnMatrix::from_normal(Vec3::UP);
        assert!(is_unit(tbn.tangent));
        assert!(tbn.tangent.dot(tbn.normal).abs() < 1e-4);
    }

    #[test]
    fn test_tbn_transform_tilted() {
        let tbn = TbnMatrix::new(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let ts = Vec3::new(0.5, 0.0, 0.866).normalize();
        let ws = tbn.transform(ts);
        assert!(is_unit(ws));
        assert!(ws.x > 0.0);
        assert!(ws.z > 0.0);
    }

    #[test]
    fn test_compute_tangent_simple_triangle() {
        let v0 = TriangleVertex {
            position: Vec3::new(0.0, 0.0, 0.0),
            uv: Vec2::new(0.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
        };
        let v1 = TriangleVertex {
            position: Vec3::new(1.0, 0.0, 0.0),
            uv: Vec2::new(1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
        };
        let v2 = TriangleVertex {
            position: Vec3::new(0.0, 1.0, 0.0),
            uv: Vec2::new(0.0, 1.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
        };
        let (t, b) = compute_tangent(&v0, &v1, &v2);
        assert!(is_unit(t));
        assert!(is_unit(b));
        // Tangent should be along X.
        assert!(t.x.abs() > 0.9);
    }

    #[test]
    fn test_height_map_creation() {
        let hm = HeightMap::new(4, 4, vec![0.0; 16]).unwrap();
        assert!(approx(hm.get(0, 0), 0.0));
    }

    #[test]
    fn test_height_map_wrong_size() {
        assert!(HeightMap::new(4, 4, vec![0.0; 15]).is_err());
    }

    #[test]
    fn test_height_map_to_normal_flat() {
        let hm = HeightMap::flat(4, 4);
        let normals = hm.to_normal_map(1.0);
        assert_eq!(normals.len(), 16);
        // All normals should point straight up (0, 0, 1).
        for n in &normals {
            assert!(approx(n.z, 1.0));
        }
    }

    #[test]
    fn test_height_map_to_normal_slope() {
        let mut hm = HeightMap::flat(3, 1);
        hm.set(0, 0, 0.0);
        hm.set(1, 0, 0.5);
        hm.set(2, 0, 1.0);
        let normals = hm.to_normal_map(1.0);
        // Middle pixel should have a negative X normal (slope going right).
        assert!(normals[1].x < 0.0);
    }

    #[test]
    fn test_blend_linear() {
        let base = Vec3::new(0.0, 0.0, 1.0);
        let detail = Vec3::new(0.5, 0.0, 0.866).normalize();
        let blended = blend_normals(base, detail, BlendMethod::Linear, 1.0);
        assert!(is_unit(blended));
        assert!(blended.x > 0.0);
    }

    #[test]
    fn test_blend_udn() {
        let base = Vec3::new(0.0, 0.0, 1.0);
        let detail = Vec3::new(0.3, 0.2, 0.93).normalize();
        let blended = blend_normals(base, detail, BlendMethod::Udn, 1.0);
        assert!(is_unit(blended));
    }

    #[test]
    fn test_blend_rnm() {
        let base = Vec3::new(0.0, 0.0, 1.0);
        let detail = Vec3::new(0.0, 0.0, 1.0);
        let blended = blend_normals(base, detail, BlendMethod::Rnm, 1.0);
        assert!(is_unit(blended));
        // Blending flat with flat should stay flat.
        assert!(approx(blended.z, 1.0));
    }

    #[test]
    fn test_blend_zero_weight() {
        let base = Vec3::new(0.0, 0.0, 1.0);
        let detail = Vec3::new(1.0, 0.0, 0.0).normalize();
        let blended = blend_normals(base, detail, BlendMethod::Linear, 0.0);
        assert!(approx(blended.z, 1.0));
    }

    #[test]
    fn test_scale_normal_strength_zero() {
        let n = Vec3::new(0.5, 0.3, 0.81).normalize();
        let scaled = scale_normal_strength(n, 0.0);
        assert!(is_unit(scaled));
        assert!(approx(scaled.z, 1.0));
    }

    #[test]
    fn test_scale_normal_strength_full() {
        let n = Vec3::new(0.3, 0.2, 0.93).normalize();
        let scaled = scale_normal_strength(n, 1.0);
        assert!(is_unit(scaled));
        assert!(approx(scaled.x, n.x));
    }

    #[test]
    fn test_encode_decode_round_trip() {
        let orig = Vec3::new(0.0, 0.0, 1.0);
        let (r, g, b) = encode_normal(orig);
        let decoded = decode_normal(r, g, b);
        assert!(approx(decoded.x, orig.x));
        assert!(approx(decoded.y, orig.y));
        assert!((decoded.z - orig.z).abs() < 0.02); // byte quantisation
    }

    #[test]
    fn test_encode_normal_range() {
        let n = Vec3::new(-1.0, -1.0, 0.0).normalize();
        let (r, g, b) = encode_normal(n);
        // Negative normals should map to < 128.
        assert!(r < 128);
        assert!(g < 128);
        let _ = b; // z component varies
    }
}
