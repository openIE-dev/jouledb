//! Spatial 3D types for JouleDB.
//!
//! Promotes `physical-ai-core`'s geometry types into first-class queryable
//! `DataType` / `Value` variants so spatial 3D becomes a native primitive
//! across the cascade — not just runtime structs in a robotics crate.
//!
//! ## What lives here vs. elsewhere
//!
//! - **Here**: small, copy-cheap geometry that doubles as **index keys**:
//!   `Point3`, `Quat`, `Pose6`, `Bbox3`. R-tree / kd-tree / octree need
//!   these as first-class so the planner can push spatial predicates.
//! - **Not here**: bulk geometry (`PointCloud`, `Mesh`, `SplatCloud`).
//!   Those are large enough that they belong behind a blob reference
//!   (a `Value::Bytes` payload + a small descriptor row), not inlined into
//!   every catalog row.
//!
//! ## Why a wrapper enum
//!
//! Mirrors the `flowG::OpKind::Spatial3d(Spatial3dOp)` pattern from
//! `inv-ai-codegraph`: one outer variant on `DataType` and `Value`,
//! one inner enum that lists the actual geometry types. Keeps the top-level
//! enum scannable and lets us add NeRF / mesh handles later without
//! re-shuffling encoding tags.

// Geometry fields (x/y/z, min/max, qx/qy/qz/qw, tx/ty/tz) and their
// constructors / constants are self-descriptive; suppress per-item doc
// nagging at the file level so the module header stays the canonical source.
#![allow(missing_docs)]

use serde::{Deserialize, Serialize};

use crate::error::CodecError;

// ============================================================================
// Geometric primitives
// ============================================================================

/// 3D point in world coordinates (f64 — robotics needs sub-millimeter precision
/// over kilometers).
#[derive(Debug, Clone, Copy, PartialEq)]
#[derive(Serialize, Deserialize)]
pub struct Point3 {
    /// Cartesian X coordinate.
    pub x: f64,
    /// Cartesian Y coordinate.
    pub y: f64,
    /// Cartesian Z coordinate.
    pub z: f64,
}

impl Point3 {
    /// Construct a 3D point from Cartesian coordinates.
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// The origin `(0, 0, 0)`.
    pub const ORIGIN: Self = Self::new(0.0, 0.0, 0.0);
}

/// Unit quaternion (`w + xi + yj + zk`) for orientation.
///
/// Stored unnormalized — we trust producers to keep it on the unit sphere.
/// Renormalize at boundaries (e.g. when composing many poses) to fight drift.
#[derive(Debug, Clone, Copy, PartialEq)]
#[derive(Serialize, Deserialize)]
pub struct Quat {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quat {
    pub const fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    /// Identity rotation.
    pub const IDENTITY: Self = Self::new(1.0, 0.0, 0.0, 0.0);
}

/// 6-DOF rigid transform (position + orientation).
///
/// Element of SE(3). The `flowG::Spatial3dOp::TransformCompose` /
/// `TransformInverse` ops operate on this type.
#[derive(Debug, Clone, Copy, PartialEq)]
#[derive(Serialize, Deserialize)]
pub struct Pose6 {
    pub position: Point3,
    pub orientation: Quat,
}

impl Pose6 {
    pub const fn new(position: Point3, orientation: Quat) -> Self {
        Self { position, orientation }
    }

    pub const IDENTITY: Self = Self::new(Point3::ORIGIN, Quat::IDENTITY);
}

/// Axis-aligned bounding box in 3D.
///
/// Used as the spatial-index key for R-tree / BVH / octree. `min` is the
/// minimum corner, `max` is the maximum corner. We do not enforce
/// `min <= max` per axis at construction — caller's responsibility.
#[derive(Debug, Clone, Copy, PartialEq)]
#[derive(Serialize, Deserialize)]
pub struct Bbox3 {
    pub min: Point3,
    pub max: Point3,
}

impl Bbox3 {
    pub const fn new(min: Point3, max: Point3) -> Self {
        Self { min, max }
    }

    /// Tightest bbox enclosing a single point (a "degenerate" box).
    pub const fn point(p: Point3) -> Self {
        Self::new(p, p)
    }

    /// Whether this bbox contains the given point (inclusive on all sides).
    pub fn contains(&self, p: Point3) -> bool {
        p.x >= self.min.x && p.x <= self.max.x
            && p.y >= self.min.y && p.y <= self.max.y
            && p.z >= self.min.z && p.z <= self.max.z
    }

    /// Whether this bbox intersects another (inclusive).
    pub fn intersects(&self, other: &Self) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x
            && self.min.y <= other.max.y && self.max.y >= other.min.y
            && self.min.z <= other.max.z && self.max.z >= other.min.z
    }
}

// ============================================================================
// Spatial3dKind — schema-side discriminator (lives in DataType)
// ============================================================================

/// Schema-level tag for a spatial 3D column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub enum Spatial3dKind {
    /// 3D point (3 × f64 = 24 bytes).
    Point3,
    /// Unit quaternion (4 × f64 = 32 bytes).
    Quat,
    /// 6-DOF pose: position + orientation (56 bytes).
    Pose6,
    /// Axis-aligned 3D bounding box (48 bytes). Index key for R-tree / BVH / octree.
    Bbox3,
}

impl Spatial3dKind {
    pub fn sql_name(&self) -> &'static str {
        match self {
            Self::Point3 => "POINT3",
            Self::Quat => "QUAT",
            Self::Pose6 => "POSE6",
            Self::Bbox3 => "BBOX3",
        }
    }

    pub fn from_sql(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "POINT3" => Some(Self::Point3),
            "QUAT" | "QUATERNION" => Some(Self::Quat),
            "POSE6" | "POSE" => Some(Self::Pose6),
            "BBOX3" | "AABB3" => Some(Self::Bbox3),
            _ => None,
        }
    }

    /// Fixed serialized size in bytes (excluding the inner-tag byte).
    pub const fn byte_size(&self) -> usize {
        match self {
            Self::Point3 => 24,
            Self::Quat => 32,
            Self::Pose6 => 56,
            Self::Bbox3 => 48,
        }
    }
}

// ============================================================================
// Spatial3dValue — runtime payload (lives in Value)
// ============================================================================

/// Runtime spatial 3D value.
///
/// One variant per `Spatial3dKind`. The encoding is tag-byte + fixed-width
/// little-endian f64s, so the on-disk size matches `Spatial3dKind::byte_size`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[derive(Serialize, Deserialize)]
pub enum Spatial3dValue {
    Point3(Point3),
    Quat(Quat),
    Pose6(Pose6),
    Bbox3(Bbox3),
}

impl Spatial3dValue {
    pub fn kind(&self) -> Spatial3dKind {
        match self {
            Self::Point3(_) => Spatial3dKind::Point3,
            Self::Quat(_) => Spatial3dKind::Quat,
            Self::Pose6(_) => Spatial3dKind::Pose6,
            Self::Bbox3(_) => Spatial3dKind::Bbox3,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Point3(_) => "point3",
            Self::Quat(_) => "quat",
            Self::Pose6(_) => "pose6",
            Self::Bbox3(_) => "bbox3",
        }
    }
}

// ── Inner-tag bytes for Spatial3dValue encoding ─────────────────────────────
//
// These live alongside the outer `tags::SPATIAL3D` byte in `value.rs`. The
// encoding pattern is: outer tag (`SPATIAL3D`), inner tag (one of the below),
// then the fixed-width payload.

pub(crate) mod inner_tags {
    pub const POINT3: u8 = 1;
    pub const QUAT: u8 = 2;
    pub const POSE6: u8 = 3;
    pub const BBOX3: u8 = 4;
}

// ── Encoding helpers — used from value.rs ───────────────────────────────────

impl Spatial3dValue {
    /// Encode this spatial value into `buf` (without the outer
    /// `tags::SPATIAL3D` byte — the caller writes that first).
    pub(crate) fn encode_into(&self, buf: &mut Vec<u8>) {
        match self {
            Self::Point3(p) => {
                buf.push(inner_tags::POINT3);
                push_f64(buf, p.x);
                push_f64(buf, p.y);
                push_f64(buf, p.z);
            }
            Self::Quat(q) => {
                buf.push(inner_tags::QUAT);
                push_f64(buf, q.w);
                push_f64(buf, q.x);
                push_f64(buf, q.y);
                push_f64(buf, q.z);
            }
            Self::Pose6(pose) => {
                buf.push(inner_tags::POSE6);
                push_f64(buf, pose.position.x);
                push_f64(buf, pose.position.y);
                push_f64(buf, pose.position.z);
                push_f64(buf, pose.orientation.w);
                push_f64(buf, pose.orientation.x);
                push_f64(buf, pose.orientation.y);
                push_f64(buf, pose.orientation.z);
            }
            Self::Bbox3(b) => {
                buf.push(inner_tags::BBOX3);
                push_f64(buf, b.min.x);
                push_f64(buf, b.min.y);
                push_f64(buf, b.min.z);
                push_f64(buf, b.max.x);
                push_f64(buf, b.max.y);
                push_f64(buf, b.max.z);
            }
        }
    }

    /// Decode a spatial value starting at `bytes[*cursor]` (the inner tag).
    /// Caller has already consumed the outer `tags::SPATIAL3D` byte.
    pub(crate) fn decode_at(
        bytes: &[u8],
        cursor: &mut usize,
    ) -> Result<Self, CodecError> {
        if *cursor >= bytes.len() {
            return Err(CodecError::UnexpectedEof { expected: 1, actual: 0 });
        }
        let inner = bytes[*cursor];
        *cursor += 1;

        match inner {
            inner_tags::POINT3 => {
                let x = read_f64(bytes, cursor)?;
                let y = read_f64(bytes, cursor)?;
                let z = read_f64(bytes, cursor)?;
                Ok(Self::Point3(Point3::new(x, y, z)))
            }
            inner_tags::QUAT => {
                let w = read_f64(bytes, cursor)?;
                let x = read_f64(bytes, cursor)?;
                let y = read_f64(bytes, cursor)?;
                let z = read_f64(bytes, cursor)?;
                Ok(Self::Quat(Quat::new(w, x, y, z)))
            }
            inner_tags::POSE6 => {
                let px = read_f64(bytes, cursor)?;
                let py = read_f64(bytes, cursor)?;
                let pz = read_f64(bytes, cursor)?;
                let qw = read_f64(bytes, cursor)?;
                let qx = read_f64(bytes, cursor)?;
                let qy = read_f64(bytes, cursor)?;
                let qz = read_f64(bytes, cursor)?;
                Ok(Self::Pose6(Pose6::new(
                    Point3::new(px, py, pz),
                    Quat::new(qw, qx, qy, qz),
                )))
            }
            inner_tags::BBOX3 => {
                let nx = read_f64(bytes, cursor)?;
                let ny = read_f64(bytes, cursor)?;
                let nz = read_f64(bytes, cursor)?;
                let xx = read_f64(bytes, cursor)?;
                let xy = read_f64(bytes, cursor)?;
                let xz = read_f64(bytes, cursor)?;
                Ok(Self::Bbox3(Bbox3::new(
                    Point3::new(nx, ny, nz),
                    Point3::new(xx, xy, xz),
                )))
            }
            _ => Err(CodecError::UnknownType { tag: inner }),
        }
    }
}

fn push_f64(buf: &mut Vec<u8>, f: f64) {
    buf.extend_from_slice(&f.to_le_bytes());
}

fn read_f64(bytes: &[u8], cursor: &mut usize) -> Result<f64, CodecError> {
    if *cursor + 8 > bytes.len() {
        return Err(CodecError::UnexpectedEof {
            expected: 8,
            actual: bytes.len() - *cursor,
        });
    }
    let arr: [u8; 8] = bytes[*cursor..*cursor + 8]
        .try_into()
        .expect("exact 8-byte slice");
    *cursor += 8;
    Ok(f64::from_le_bytes(arr))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bbox3_contains_point() {
        let b = Bbox3::new(Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 10.0, 10.0));
        assert!(b.contains(Point3::new(5.0, 5.0, 5.0)));
        assert!(b.contains(Point3::new(0.0, 0.0, 0.0))); // boundary inclusive
        assert!(b.contains(Point3::new(10.0, 10.0, 10.0)));
        assert!(!b.contains(Point3::new(11.0, 5.0, 5.0)));
        assert!(!b.contains(Point3::new(5.0, -1.0, 5.0)));
    }

    #[test]
    fn bbox3_intersects() {
        let a = Bbox3::new(Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 10.0, 10.0));
        let b = Bbox3::new(Point3::new(5.0, 5.0, 5.0), Point3::new(15.0, 15.0, 15.0));
        let c = Bbox3::new(Point3::new(20.0, 20.0, 20.0), Point3::new(30.0, 30.0, 30.0));
        assert!(a.intersects(&b));
        assert!(b.intersects(&a));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn spatial3d_kind_sql_roundtrip() {
        for k in [
            Spatial3dKind::Point3,
            Spatial3dKind::Quat,
            Spatial3dKind::Pose6,
            Spatial3dKind::Bbox3,
        ] {
            assert_eq!(Spatial3dKind::from_sql(k.sql_name()), Some(k));
        }
        assert_eq!(Spatial3dKind::from_sql("quaternion"), Some(Spatial3dKind::Quat));
        assert_eq!(Spatial3dKind::from_sql("aabb3"), Some(Spatial3dKind::Bbox3));
        assert_eq!(Spatial3dKind::from_sql("nonsense"), None);
    }

    #[test]
    fn spatial3d_kind_byte_sizes() {
        assert_eq!(Spatial3dKind::Point3.byte_size(), 24);
        assert_eq!(Spatial3dKind::Quat.byte_size(), 32);
        assert_eq!(Spatial3dKind::Pose6.byte_size(), 56);
        assert_eq!(Spatial3dKind::Bbox3.byte_size(), 48);
    }

    fn enc_dec(v: Spatial3dValue) -> Spatial3dValue {
        let mut buf = Vec::new();
        v.encode_into(&mut buf);
        // payload size = 1 inner tag + N f64s
        let expected_payload = 1 + v.kind().byte_size();
        assert_eq!(buf.len(), expected_payload, "payload size for {:?}", v.kind());
        let mut cursor = 0;
        let decoded = Spatial3dValue::decode_at(&buf, &mut cursor).expect("decode");
        assert_eq!(cursor, buf.len(), "decoder consumed entire payload");
        decoded
    }

    #[test]
    fn spatial3d_value_point3_roundtrip() {
        let v = Spatial3dValue::Point3(Point3::new(1.5, -2.5, 3.25));
        assert_eq!(enc_dec(v), v);
    }

    #[test]
    fn spatial3d_value_quat_roundtrip() {
        let v = Spatial3dValue::Quat(Quat::new(0.5, 0.5, 0.5, 0.5));
        assert_eq!(enc_dec(v), v);
    }

    #[test]
    fn spatial3d_value_pose6_roundtrip() {
        let v = Spatial3dValue::Pose6(Pose6::new(
            Point3::new(10.0, 20.0, 30.0),
            Quat::new(1.0, 0.0, 0.0, 0.0),
        ));
        assert_eq!(enc_dec(v), v);
    }

    #[test]
    fn spatial3d_value_bbox3_roundtrip() {
        let v = Spatial3dValue::Bbox3(Bbox3::new(
            Point3::new(-1.0, -2.0, -3.0),
            Point3::new(1.0, 2.0, 3.0),
        ));
        assert_eq!(enc_dec(v), v);
    }

    #[test]
    fn spatial3d_value_kind_matches_variant() {
        assert_eq!(
            Spatial3dValue::Point3(Point3::ORIGIN).kind(),
            Spatial3dKind::Point3
        );
        assert_eq!(
            Spatial3dValue::Pose6(Pose6::IDENTITY).kind(),
            Spatial3dKind::Pose6
        );
    }

    #[test]
    fn spatial3d_decode_unknown_inner_tag_errors() {
        let mut cursor = 0;
        let bad = [99u8, 0, 0, 0];
        assert!(Spatial3dValue::decode_at(&bad, &mut cursor).is_err());
    }
}
