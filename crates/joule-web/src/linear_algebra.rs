//! Linear Algebra — Vec2/3/4, Mat2/3/4, transforms.
//!
//! Pure-Rust vector and matrix types for 2D/3D math, replacing glam, nalgebra,
//! and similar JS/TS math libraries with energy-tracked, headless-testable code.

use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

// ── Vectors ────────────────────────────────────────────────────

/// 2D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

/// 3D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// 4D vector (homogeneous coordinates).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

// ── Vec2 impl ──────────────────────────────────────────────────

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
    pub const ONE: Self = Self { x: 1.0, y: 1.0 };

    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y
    }

    /// 2D cross product (scalar z-component).
    pub fn cross(self, rhs: Self) -> f64 {
        self.x * rhs.y - self.y * rhs.x
    }

    pub fn length(self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn length_sq(self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < 1e-15 {
            return Self::ZERO;
        }
        Self {
            x: self.x / len,
            y: self.y / len,
        }
    }

    pub fn scale(self, s: f64) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
        }
    }

    pub fn lerp(self, other: Self, t: f64) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }

    pub fn distance(self, other: Self) -> f64 {
        (self - other).length()
    }

    pub fn angle(self) -> f64 {
        self.y.atan2(self.x)
    }
}

impl Add for Vec2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Sub for Vec2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl Neg for Vec2 {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl fmt::Display for Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

// ── Vec3 impl ──────────────────────────────────────────────────

impl Vec3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    pub const ONE: Self = Self {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };
    pub const X: Self = Self {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };
    pub const Y: Self = Self {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };
    pub const Z: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    pub fn length(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn length_sq(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < 1e-15 {
            return Self::ZERO;
        }
        Self {
            x: self.x / len,
            y: self.y / len,
            z: self.z / len,
        }
    }

    pub fn scale(self, s: f64) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    pub fn lerp(self, other: Self, t: f64) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            z: self.z + (other.z - self.z) * t,
        }
    }

    pub fn distance(self, other: Self) -> f64 {
        (self - other).length()
    }

    /// Extend to a Vec4 with the given w component.
    pub fn extend(self, w: f64) -> Vec4 {
        Vec4 {
            x: self.x,
            y: self.y,
            z: self.z,
            w,
        }
    }
}

impl Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

// ── Vec4 impl ──────────────────────────────────────────────────

impl Vec4 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 0.0,
    };

    pub const fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self { x, y, z, w }
    }

    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }

    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < 1e-15 {
            return Self::ZERO;
        }
        Self {
            x: self.x / len,
            y: self.y / len,
            z: self.z / len,
            w: self.w / len,
        }
    }

    pub fn scale(self, s: f64) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
            w: self.w * s,
        }
    }

    pub fn truncate(self) -> Vec3 {
        Vec3 {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}

impl Add for Vec4 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            w: self.w + rhs.w,
        }
    }
}

impl Sub for Vec4 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            w: self.w - rhs.w,
        }
    }
}

impl fmt::Display for Vec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

// ── Matrices ───────────────────────────────────────────────────

/// 2x2 matrix stored column-major: `[col0, col1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat2 {
    /// Columns: `m[col][row]`.
    pub m: [[f64; 2]; 2],
}

/// 3x3 matrix stored column-major.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3 {
    pub m: [[f64; 3]; 3],
}

/// 4x4 matrix stored column-major.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    pub m: [[f64; 4]; 4],
}

// ── Mat2 ───────────────────────────────────────────────────────

impl Mat2 {
    pub const IDENTITY: Self = Self {
        m: [[1.0, 0.0], [0.0, 1.0]],
    };

    pub const ZERO: Self = Self {
        m: [[0.0; 2]; 2],
    };

    pub const fn new(m00: f64, m01: f64, m10: f64, m11: f64) -> Self {
        Self {
            m: [[m00, m01], [m10, m11]],
        }
    }

    pub fn determinant(self) -> f64 {
        self.m[0][0] * self.m[1][1] - self.m[1][0] * self.m[0][1]
    }

    pub fn transpose(self) -> Self {
        Self {
            m: [[self.m[0][0], self.m[1][0]], [self.m[0][1], self.m[1][1]]],
        }
    }

    pub fn inverse(self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < 1e-15 {
            return None;
        }
        let inv_det = 1.0 / det;
        Some(Self {
            m: [
                [self.m[1][1] * inv_det, -self.m[0][1] * inv_det],
                [-self.m[1][0] * inv_det, self.m[0][0] * inv_det],
            ],
        })
    }

    pub fn mul_vec2(self, v: Vec2) -> Vec2 {
        Vec2 {
            x: self.m[0][0] * v.x + self.m[1][0] * v.y,
            y: self.m[0][1] * v.x + self.m[1][1] * v.y,
        }
    }

    /// 2D rotation matrix.
    pub fn rotation(angle_rad: f64) -> Self {
        let (s, c) = angle_rad.sin_cos();
        Self {
            m: [[c, s], [-s, c]],
        }
    }

    /// 2D scale matrix.
    pub fn scaling(sx: f64, sy: f64) -> Self {
        Self {
            m: [[sx, 0.0], [0.0, sy]],
        }
    }
}

impl Mul for Mat2 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut out = Self::ZERO;
        for c in 0..2 {
            for r in 0..2 {
                out.m[c][r] = self.m[0][r] * rhs.m[c][0] + self.m[1][r] * rhs.m[c][1];
            }
        }
        out
    }
}

// ── Mat3 ───────────────────────────────────────────────────────

impl Mat3 {
    pub const IDENTITY: Self = Self {
        m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };

    pub const ZERO: Self = Self {
        m: [[0.0; 3]; 3],
    };

    pub fn from_cols(c0: [f64; 3], c1: [f64; 3], c2: [f64; 3]) -> Self {
        Self { m: [c0, c1, c2] }
    }

    pub fn determinant(self) -> f64 {
        let m = &self.m;
        m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2])
            - m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2])
            + m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2])
    }

    pub fn transpose(self) -> Self {
        let mut out = Self::ZERO;
        for c in 0..3 {
            for r in 0..3 {
                out.m[c][r] = self.m[r][c];
            }
        }
        out
    }

    pub fn inverse(self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < 1e-15 {
            return None;
        }
        let inv = 1.0 / det;
        let m = &self.m;
        // cofactor matrix, transposed (adjugate), scaled by 1/det
        Some(Self {
            m: [
                [
                    (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * inv,
                    (m[2][1] * m[0][2] - m[0][1] * m[2][2]) * inv,
                    (m[0][1] * m[1][2] - m[1][1] * m[0][2]) * inv,
                ],
                [
                    (m[2][0] * m[1][2] - m[1][0] * m[2][2]) * inv,
                    (m[0][0] * m[2][2] - m[2][0] * m[0][2]) * inv,
                    (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * inv,
                ],
                [
                    (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * inv,
                    (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * inv,
                    (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * inv,
                ],
            ],
        })
    }

    pub fn mul_vec3(self, v: Vec3) -> Vec3 {
        Vec3 {
            x: self.m[0][0] * v.x + self.m[1][0] * v.y + self.m[2][0] * v.z,
            y: self.m[0][1] * v.x + self.m[1][1] * v.y + self.m[2][1] * v.z,
            z: self.m[0][2] * v.x + self.m[1][2] * v.y + self.m[2][2] * v.z,
        }
    }

    /// 2D translation as a 3x3 homogeneous matrix.
    pub fn translation_2d(tx: f64, ty: f64) -> Self {
        Self {
            m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [tx, ty, 1.0]],
        }
    }

    /// 2D rotation as a 3x3 homogeneous matrix.
    pub fn rotation_2d(angle_rad: f64) -> Self {
        let (s, c) = angle_rad.sin_cos();
        Self {
            m: [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// 2D scale as a 3x3 homogeneous matrix.
    pub fn scaling_2d(sx: f64, sy: f64) -> Self {
        Self {
            m: [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]],
        }
    }
}

impl Mul for Mat3 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut out = Self::ZERO;
        for c in 0..3 {
            for r in 0..3 {
                out.m[c][r] =
                    self.m[0][r] * rhs.m[c][0] + self.m[1][r] * rhs.m[c][1] + self.m[2][r] * rhs.m[c][2];
            }
        }
        out
    }
}

// ── Mat4 ───────────────────────────────────────────────────────

impl Mat4 {
    pub const IDENTITY: Self = Self {
        m: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };

    pub const ZERO: Self = Self {
        m: [[0.0; 4]; 4],
    };

    pub fn from_cols(c0: [f64; 4], c1: [f64; 4], c2: [f64; 4], c3: [f64; 4]) -> Self {
        Self { m: [c0, c1, c2, c3] }
    }

    pub fn transpose(self) -> Self {
        let mut out = Self::ZERO;
        for c in 0..4 {
            for r in 0..4 {
                out.m[c][r] = self.m[r][c];
            }
        }
        out
    }

    pub fn determinant(self) -> f64 {
        let m = &self.m;
        let s0 = m[0][0] * m[1][1] - m[1][0] * m[0][1];
        let s1 = m[0][0] * m[1][2] - m[1][0] * m[0][2];
        let s2 = m[0][0] * m[1][3] - m[1][0] * m[0][3];
        let s3 = m[0][1] * m[1][2] - m[1][1] * m[0][2];
        let s4 = m[0][1] * m[1][3] - m[1][1] * m[0][3];
        let s5 = m[0][2] * m[1][3] - m[1][2] * m[0][3];

        let c5 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
        let c4 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
        let c3 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
        let c2 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
        let c1 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
        let c0 = m[2][0] * m[3][1] - m[3][0] * m[2][1];

        s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0
    }

    pub fn inverse(self) -> Option<Self> {
        let m = &self.m;
        let s0 = m[0][0] * m[1][1] - m[1][0] * m[0][1];
        let s1 = m[0][0] * m[1][2] - m[1][0] * m[0][2];
        let s2 = m[0][0] * m[1][3] - m[1][0] * m[0][3];
        let s3 = m[0][1] * m[1][2] - m[1][1] * m[0][2];
        let s4 = m[0][1] * m[1][3] - m[1][1] * m[0][3];
        let s5 = m[0][2] * m[1][3] - m[1][2] * m[0][3];

        let c5 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
        let c4 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
        let c3 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
        let c2 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
        let c1 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
        let c0 = m[2][0] * m[3][1] - m[3][0] * m[2][1];

        let det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
        if det.abs() < 1e-15 {
            return None;
        }
        let inv = 1.0 / det;

        Some(Self {
            m: [
                [
                    (m[1][1] * c5 - m[1][2] * c4 + m[1][3] * c3) * inv,
                    (-m[0][1] * c5 + m[0][2] * c4 - m[0][3] * c3) * inv,
                    (m[3][1] * s5 - m[3][2] * s4 + m[3][3] * s3) * inv,
                    (-m[2][1] * s5 + m[2][2] * s4 - m[2][3] * s3) * inv,
                ],
                [
                    (-m[1][0] * c5 + m[1][2] * c2 - m[1][3] * c1) * inv,
                    (m[0][0] * c5 - m[0][2] * c2 + m[0][3] * c1) * inv,
                    (-m[3][0] * s5 + m[3][2] * s2 - m[3][3] * s1) * inv,
                    (m[2][0] * s5 - m[2][2] * s2 + m[2][3] * s1) * inv,
                ],
                [
                    (m[1][0] * c4 - m[1][1] * c2 + m[1][3] * c0) * inv,
                    (-m[0][0] * c4 + m[0][1] * c2 - m[0][3] * c0) * inv,
                    (m[3][0] * s4 - m[3][1] * s2 + m[3][3] * s0) * inv,
                    (-m[2][0] * s4 + m[2][1] * s2 - m[2][3] * s0) * inv,
                ],
                [
                    (-m[1][0] * c3 + m[1][1] * c1 - m[1][2] * c0) * inv,
                    (m[0][0] * c3 - m[0][1] * c1 + m[0][2] * c0) * inv,
                    (-m[3][0] * s3 + m[3][1] * s1 - m[3][2] * s0) * inv,
                    (m[2][0] * s3 - m[2][1] * s1 + m[2][2] * s0) * inv,
                ],
            ],
        })
    }

    pub fn mul_vec4(self, v: Vec4) -> Vec4 {
        Vec4 {
            x: self.m[0][0] * v.x + self.m[1][0] * v.y + self.m[2][0] * v.z + self.m[3][0] * v.w,
            y: self.m[0][1] * v.x + self.m[1][1] * v.y + self.m[2][1] * v.z + self.m[3][1] * v.w,
            z: self.m[0][2] * v.x + self.m[1][2] * v.y + self.m[2][2] * v.z + self.m[3][2] * v.w,
            w: self.m[0][3] * v.x + self.m[1][3] * v.y + self.m[2][3] * v.z + self.m[3][3] * v.w,
        }
    }

    /// 3D translation.
    pub fn translation(tx: f64, ty: f64, tz: f64) -> Self {
        Self {
            m: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [tx, ty, tz, 1.0],
            ],
        }
    }

    /// 3D scaling.
    pub fn scaling(sx: f64, sy: f64, sz: f64) -> Self {
        Self {
            m: [
                [sx, 0.0, 0.0, 0.0],
                [0.0, sy, 0.0, 0.0],
                [0.0, 0.0, sz, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Rotation around X axis.
    pub fn rotation_x(angle_rad: f64) -> Self {
        let (s, c) = angle_rad.sin_cos();
        Self {
            m: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, c, s, 0.0],
                [0.0, -s, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Rotation around Y axis.
    pub fn rotation_y(angle_rad: f64) -> Self {
        let (s, c) = angle_rad.sin_cos();
        Self {
            m: [
                [c, 0.0, -s, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [s, 0.0, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Rotation around Z axis.
    pub fn rotation_z(angle_rad: f64) -> Self {
        let (s, c) = angle_rad.sin_cos();
        Self {
            m: [
                [c, s, 0.0, 0.0],
                [-s, c, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Perspective projection (right-handed, depth 0..1).
    pub fn perspective(fov_y_rad: f64, aspect: f64, near: f64, far: f64) -> Self {
        let f = 1.0 / (fov_y_rad / 2.0).tan();
        let nf = 1.0 / (near - far);
        Self {
            m: [
                [f / aspect, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [0.0, 0.0, far * nf, -1.0],
                [0.0, 0.0, near * far * nf, 0.0],
            ],
        }
    }

    /// Orthographic projection.
    pub fn orthographic(left: f64, right: f64, bottom: f64, top: f64, near: f64, far: f64) -> Self {
        let rl = 1.0 / (right - left);
        let tb = 1.0 / (top - bottom);
        let nf = 1.0 / (near - far);
        Self {
            m: [
                [2.0 * rl, 0.0, 0.0, 0.0],
                [0.0, 2.0 * tb, 0.0, 0.0],
                [0.0, 0.0, nf, 0.0],
                [-(right + left) * rl, -(top + bottom) * tb, near * nf, 1.0],
            ],
        }
    }

    /// Look-at matrix (right-handed).
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        let f = (target - eye).normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);
        Self {
            m: [
                [s.x, u.x, -f.x, 0.0],
                [s.y, u.y, -f.y, 0.0],
                [s.z, u.z, -f.z, 0.0],
                [-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0],
            ],
        }
    }
}

impl Mul for Mat4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut out = Self::ZERO;
        for c in 0..4 {
            for r in 0..4 {
                out.m[c][r] = self.m[0][r] * rhs.m[c][0]
                    + self.m[1][r] * rhs.m[c][1]
                    + self.m[2][r] * rhs.m[c][2]
                    + self.m[3][r] * rhs.m[c][3];
            }
        }
        out
    }
}

// ── Approximate equality helper ────────────────────────────────

fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EPS: f64 = 1e-10;

    fn v2_approx(a: Vec2, b: Vec2) -> bool {
        approx_eq(a.x, b.x, EPS) && approx_eq(a.y, b.y, EPS)
    }

    fn v3_approx(a: Vec3, b: Vec3) -> bool {
        approx_eq(a.x, b.x, EPS) && approx_eq(a.y, b.y, EPS) && approx_eq(a.z, b.z, EPS)
    }

    #[test]
    fn vec2_basic_ops() {
        let a = Vec2::new(3.0, 4.0);
        let b = Vec2::new(1.0, 2.0);
        assert_eq!(a + b, Vec2::new(4.0, 6.0));
        assert_eq!(a - b, Vec2::new(2.0, 2.0));
        assert!(approx_eq(a.length(), 5.0, EPS));
        assert!(approx_eq(a.dot(b), 11.0, EPS));
    }

    #[test]
    fn vec2_normalize_and_lerp() {
        let v = Vec2::new(3.0, 4.0).normalize();
        assert!(approx_eq(v.length(), 1.0, EPS));
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(10.0, 20.0);
        assert!(v2_approx(a.lerp(b, 0.5), Vec2::new(5.0, 10.0)));
    }

    #[test]
    fn vec3_cross_product() {
        let x = Vec3::X;
        let y = Vec3::Y;
        let z = x.cross(y);
        assert!(v3_approx(z, Vec3::Z));
    }

    #[test]
    fn vec3_normalize_zero() {
        let v = Vec3::ZERO.normalize();
        assert_eq!(v, Vec3::ZERO);
    }

    #[test]
    fn mat2_inverse() {
        let m = Mat2::new(4.0, 7.0, 2.0, 6.0);
        let inv = m.inverse().unwrap();
        let product = m * inv;
        assert!(approx_eq(product.m[0][0], 1.0, EPS));
        assert!(approx_eq(product.m[1][1], 1.0, EPS));
        assert!(approx_eq(product.m[0][1], 0.0, EPS));
        assert!(approx_eq(product.m[1][0], 0.0, EPS));
    }

    #[test]
    fn mat2_rotation() {
        let rot = Mat2::rotation(PI / 2.0);
        let v = rot.mul_vec2(Vec2::new(1.0, 0.0));
        assert!(v2_approx(v, Vec2::new(0.0, 1.0)));
    }

    #[test]
    fn mat3_determinant_and_inverse() {
        let m = Mat3::from_cols([1.0, 0.0, 5.0], [2.0, 1.0, 6.0], [3.0, 4.0, 0.0]);
        let det = m.determinant();
        assert!(approx_eq(det, 1.0, EPS));
        let inv = m.inverse().unwrap();
        let id = m * inv;
        for c in 0..3 {
            for r in 0..3 {
                let expected = if c == r { 1.0 } else { 0.0 };
                assert!(approx_eq(id.m[c][r], expected, EPS), "m[{c}][{r}] = {} expected {expected}", id.m[c][r]);
            }
        }
    }

    #[test]
    fn mat3_translation_2d() {
        let t = Mat3::translation_2d(5.0, 10.0);
        let p = t.mul_vec3(Vec3::new(1.0, 2.0, 1.0)); // homogeneous
        assert!(v3_approx(p, Vec3::new(6.0, 12.0, 1.0)));
    }

    #[test]
    fn mat4_identity_mul() {
        let id = Mat4::IDENTITY;
        let m = Mat4::translation(1.0, 2.0, 3.0);
        let product = id * m;
        assert_eq!(product.m, m.m);
    }

    #[test]
    fn mat4_inverse() {
        let m = Mat4::translation(3.0, 4.0, 5.0);
        let inv = m.inverse().unwrap();
        let id = m * inv;
        for c in 0..4 {
            for r in 0..4 {
                let expected = if c == r { 1.0 } else { 0.0 };
                assert!(approx_eq(id.m[c][r], expected, EPS), "m[{c}][{r}]");
            }
        }
    }

    #[test]
    fn mat4_rotation_z() {
        let rot = Mat4::rotation_z(PI / 2.0);
        let v = rot.mul_vec4(Vec4::new(1.0, 0.0, 0.0, 1.0));
        assert!(approx_eq(v.x, 0.0, EPS));
        assert!(approx_eq(v.y, 1.0, EPS));
    }

    #[test]
    fn mat4_look_at_basic() {
        let eye = Vec3::new(0.0, 0.0, 5.0);
        let target = Vec3::ZERO;
        let up = Vec3::Y;
        let view = Mat4::look_at(eye, target, up);
        // eye should map to (0,0,-5) in view space (approximately)
        let p = view.mul_vec4(Vec4::new(0.0, 0.0, 5.0, 1.0));
        assert!(approx_eq(p.x, 0.0, EPS));
        assert!(approx_eq(p.y, 0.0, EPS));
        assert!(approx_eq(p.z, 0.0, EPS));
    }

    #[test]
    fn mat4_perspective_basic() {
        let p = Mat4::perspective(PI / 2.0, 1.0, 0.1, 100.0);
        assert!(p.determinant().abs() > 1e-10);
    }

    #[test]
    fn vec4_add_sub() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(4.0, 3.0, 2.0, 1.0);
        assert_eq!(a + b, Vec4::new(5.0, 5.0, 5.0, 5.0));
        assert_eq!(a - b, Vec4::new(-3.0, -1.0, 1.0, 3.0));
    }
}
