//! 3D Math Library — pure Rust replacement for gl-matrix / Three.js math core.
//!
//! Provides `Vec3`, `Vec4`, `Mat4`, `Quaternion`, `Camera`, and `Frustum`
//! for 3D graphics calculations. No actual WebGL calls — just math.

use std::ops;

// ── Vec3 ───────────────────────────────────────────────────────

/// A 3D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }

    pub fn one() -> Self {
        Self { x: 1.0, y: 1.0, z: 1.0 }
    }

    pub fn up() -> Self {
        Self { x: 0.0, y: 1.0, z: 0.0 }
    }

    pub fn right() -> Self {
        Self { x: 1.0, y: 0.0, z: 0.0 }
    }

    pub fn forward() -> Self {
        Self { x: 0.0, y: 0.0, z: -1.0 }
    }

    pub fn dot(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn length(&self) -> f64 {
        self.length_squared().sqrt()
    }

    pub fn length_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn normalize(&self) -> Vec3 {
        let len = self.length();
        if len < 1e-12 {
            return Vec3::zero();
        }
        Vec3 {
            x: self.x / len,
            y: self.y / len,
            z: self.z / len,
        }
    }

    pub fn lerp(&self, other: &Vec3, t: f64) -> Vec3 {
        Vec3 {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            z: self.z + (other.z - self.z) * t,
        }
    }

    pub fn distance(&self, other: &Vec3) -> f64 {
        (*self - *other).length()
    }

    pub fn angle_between(&self, other: &Vec3) -> f64 {
        let d = self.dot(other);
        let denom = self.length() * other.length();
        if denom < 1e-12 {
            return 0.0;
        }
        (d / denom).clamp(-1.0, 1.0).acos()
    }
}

impl ops::Add for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3 { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z }
    }
}

impl ops::Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3 { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z }
    }
}

impl ops::Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: f64) -> Vec3 {
        Vec3 { x: self.x * rhs, y: self.y * rhs, z: self.z * rhs }
    }
}

impl ops::Div<f64> for Vec3 {
    type Output = Vec3;
    fn div(self, rhs: f64) -> Vec3 {
        Vec3 { x: self.x / rhs, y: self.y / rhs, z: self.z / rhs }
    }
}

// ── Vec4 ───────────────────────────────────────────────────────

/// A 4D vector (homogeneous coordinates).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Vec4 {
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self { x, y, z, w }
    }

    pub fn dot(&self, other: &Vec4) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }
}

// ── Mat4 ───────────────────────────────────────────────────────

/// Column-major 4x4 matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    pub data: [f64; 16],
}

impl Mat4 {
    /// Access element at (row, col), 0-indexed.
    #[inline]
    fn at(&self, row: usize, col: usize) -> f64 {
        self.data[col * 4 + row]
    }

    pub fn identity() -> Self {
        let mut data = [0.0; 16];
        data[0] = 1.0;
        data[5] = 1.0;
        data[10] = 1.0;
        data[15] = 1.0;
        Self { data }
    }

    pub fn translation(x: f64, y: f64, z: f64) -> Self {
        let mut m = Self::identity();
        m.data[12] = x;
        m.data[13] = y;
        m.data[14] = z;
        m
    }

    pub fn rotation_x(angle: f64) -> Self {
        let mut m = Self::identity();
        let c = angle.cos();
        let s = angle.sin();
        m.data[5] = c;
        m.data[6] = s;
        m.data[9] = -s;
        m.data[10] = c;
        m
    }

    pub fn rotation_y(angle: f64) -> Self {
        let mut m = Self::identity();
        let c = angle.cos();
        let s = angle.sin();
        m.data[0] = c;
        m.data[2] = -s;
        m.data[8] = s;
        m.data[10] = c;
        m
    }

    pub fn rotation_z(angle: f64) -> Self {
        let mut m = Self::identity();
        let c = angle.cos();
        let s = angle.sin();
        m.data[0] = c;
        m.data[1] = s;
        m.data[4] = -s;
        m.data[5] = c;
        m
    }

    pub fn scaling(x: f64, y: f64, z: f64) -> Self {
        let mut m = Self::identity();
        m.data[0] = x;
        m.data[5] = y;
        m.data[10] = z;
        m
    }

    pub fn perspective(fov_y: f64, aspect: f64, near: f64, far: f64) -> Self {
        let f = 1.0 / (fov_y / 2.0).tan();
        let nf = 1.0 / (near - far);
        let mut m = Self { data: [0.0; 16] };
        m.data[0] = f / aspect;
        m.data[5] = f;
        m.data[10] = (far + near) * nf;
        m.data[11] = -1.0;
        m.data[14] = 2.0 * far * near * nf;
        m
    }

    pub fn orthographic(left: f64, right: f64, bottom: f64, top: f64, near: f64, far: f64) -> Self {
        let mut m = Self { data: [0.0; 16] };
        let lr = 1.0 / (left - right);
        let bt = 1.0 / (bottom - top);
        let nf = 1.0 / (near - far);
        m.data[0] = -2.0 * lr;
        m.data[5] = -2.0 * bt;
        m.data[10] = 2.0 * nf;
        m.data[12] = (left + right) * lr;
        m.data[13] = (top + bottom) * bt;
        m.data[14] = (far + near) * nf;
        m.data[15] = 1.0;
        m
    }

    pub fn look_at(eye: &Vec3, target: &Vec3, up: &Vec3) -> Self {
        let f = (*target - *eye).normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(&f);

        let mut m = Self::identity();
        m.data[0] = s.x;
        m.data[4] = s.y;
        m.data[8] = s.z;
        m.data[1] = u.x;
        m.data[5] = u.y;
        m.data[9] = u.z;
        m.data[2] = -f.x;
        m.data[6] = -f.y;
        m.data[10] = -f.z;
        m.data[12] = -s.dot(eye);
        m.data[13] = -u.dot(eye);
        m.data[14] = f.dot(eye);
        m
    }

    pub fn multiply(&self, other: &Mat4) -> Mat4 {
        let mut result = [0.0; 16];
        for col in 0..4 {
            for row in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += self.data[k * 4 + row] * other.data[col * 4 + k];
                }
                result[col * 4 + row] = sum;
            }
        }
        Mat4 { data: result }
    }

    pub fn transform_vec3(&self, v: &Vec3) -> Vec3 {
        let w = self.data[3] * v.x + self.data[7] * v.y + self.data[11] * v.z + self.data[15];
        let inv_w = if w.abs() > 1e-12 { 1.0 / w } else { 1.0 };
        Vec3 {
            x: (self.data[0] * v.x + self.data[4] * v.y + self.data[8] * v.z + self.data[12]) * inv_w,
            y: (self.data[1] * v.x + self.data[5] * v.y + self.data[9] * v.z + self.data[13]) * inv_w,
            z: (self.data[2] * v.x + self.data[6] * v.y + self.data[10] * v.z + self.data[14]) * inv_w,
        }
    }

    pub fn transform_vec4(&self, v: &Vec4) -> Vec4 {
        Vec4 {
            x: self.data[0] * v.x + self.data[4] * v.y + self.data[8] * v.z + self.data[12] * v.w,
            y: self.data[1] * v.x + self.data[5] * v.y + self.data[9] * v.z + self.data[13] * v.w,
            z: self.data[2] * v.x + self.data[6] * v.y + self.data[10] * v.z + self.data[14] * v.w,
            w: self.data[3] * v.x + self.data[7] * v.y + self.data[11] * v.z + self.data[15] * v.w,
        }
    }

    pub fn transpose(&self) -> Mat4 {
        let mut result = [0.0; 16];
        for col in 0..4 {
            for row in 0..4 {
                result[col * 4 + row] = self.data[row * 4 + col];
            }
        }
        Mat4 { data: result }
    }

    pub fn determinant(&self) -> f64 {
        let a = &self.data;
        let s0 = a[0] * a[5] - a[4] * a[1];
        let s1 = a[0] * a[9] - a[8] * a[1];
        let s2 = a[0] * a[13] - a[12] * a[1];
        let s3 = a[4] * a[9] - a[8] * a[5];
        let s4 = a[4] * a[13] - a[12] * a[5];
        let s5 = a[8] * a[13] - a[12] * a[9];

        let c5 = a[10] * a[15] - a[14] * a[11];
        let c4 = a[6] * a[15] - a[14] * a[7];
        let c3 = a[6] * a[11] - a[10] * a[7];
        let c2 = a[2] * a[15] - a[14] * a[3];
        let c1 = a[2] * a[11] - a[10] * a[3];
        let c0 = a[2] * a[7] - a[6] * a[3];

        s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0
    }

    pub fn inverse(&self) -> Option<Mat4> {
        // Gauss-Jordan elimination on [M | I] to produce [I | M^-1].
        let mut aug = [[0.0f64; 8]; 4];

        // Build augmented matrix [M | I] where M is in row-major form.
        for row in 0..4 {
            for col in 0..4 {
                aug[row][col] = self.data[col * 4 + row]; // column-major to row-major
            }
            for col in 0..4 {
                aug[row][4 + col] = if row == col { 1.0 } else { 0.0 };
            }
        }

        // Forward elimination with partial pivoting.
        for col in 0..4 {
            // Find pivot.
            let mut max_val = aug[col][col].abs();
            let mut max_row = col;
            for row in (col + 1)..4 {
                let v = aug[row][col].abs();
                if v > max_val {
                    max_val = v;
                    max_row = row;
                }
            }
            if max_val < 1e-12 {
                return None;
            }

            // Swap rows.
            if max_row != col {
                aug.swap(col, max_row);
            }

            // Scale pivot row.
            let pivot = aug[col][col];
            for j in 0..8 {
                aug[col][j] /= pivot;
            }

            // Eliminate column in other rows.
            for row in 0..4 {
                if row == col {
                    continue;
                }
                let factor = aug[row][col];
                for j in 0..8 {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        // Extract result back to column-major.
        let mut result = [0.0f64; 16];
        for row in 0..4 {
            for col in 0..4 {
                result[col * 4 + row] = aug[row][4 + col];
            }
        }

        Some(Mat4 { data: result })
    }
}

// ── Quaternion ─────────────────────────────────────────────────

/// A unit quaternion for 3D rotations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Quaternion {
    pub fn identity() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }
    }

    pub fn from_axis_angle(axis: &Vec3, angle: f64) -> Self {
        let half = angle / 2.0;
        let s = half.sin();
        let a = axis.normalize();
        Self {
            x: a.x * s,
            y: a.y * s,
            z: a.z * s,
            w: half.cos(),
        }
    }

    pub fn from_euler(pitch: f64, yaw: f64, roll: f64) -> Self {
        let (sp, cp) = (pitch / 2.0).sin_cos();
        let (sy, cy) = (yaw / 2.0).sin_cos();
        let (sr, cr) = (roll / 2.0).sin_cos();
        Self {
            x: sr * cp * cy - cr * sp * sy,
            y: cr * sp * cy + sr * cp * sy,
            z: cr * cp * sy - sr * sp * cy,
            w: cr * cp * cy + sr * sp * sy,
        }
    }

    pub fn multiply(&self, other: &Quaternion) -> Quaternion {
        Quaternion {
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        }
    }

    pub fn normalize(&self) -> Quaternion {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt();
        if len < 1e-12 {
            return Quaternion::identity();
        }
        Quaternion {
            x: self.x / len,
            y: self.y / len,
            z: self.z / len,
            w: self.w / len,
        }
    }

    pub fn slerp(&self, other: &Quaternion, t: f64) -> Quaternion {
        let mut dot = self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w;
        let mut other = *other;

        // If dot < 0, negate one to take the shorter arc.
        if dot < 0.0 {
            other = Quaternion { x: -other.x, y: -other.y, z: -other.z, w: -other.w };
            dot = -dot;
        }

        if dot > 0.9995 {
            // Linear interpolation for very close quaternions.
            return Quaternion {
                x: self.x + (other.x - self.x) * t,
                y: self.y + (other.y - self.y) * t,
                z: self.z + (other.z - self.z) * t,
                w: self.w + (other.w - self.w) * t,
            }
            .normalize();
        }

        let theta = dot.clamp(-1.0, 1.0).acos();
        let sin_theta = theta.sin();
        let a = ((1.0 - t) * theta).sin() / sin_theta;
        let b = (t * theta).sin() / sin_theta;

        Quaternion {
            x: self.x * a + other.x * b,
            y: self.y * a + other.y * b,
            z: self.z * a + other.z * b,
            w: self.w * a + other.w * b,
        }
    }

    pub fn to_mat4(&self) -> Mat4 {
        let q = self.normalize();
        let xx = q.x * q.x;
        let yy = q.y * q.y;
        let zz = q.z * q.z;
        let xy = q.x * q.y;
        let xz = q.x * q.z;
        let yz = q.y * q.z;
        let wx = q.w * q.x;
        let wy = q.w * q.y;
        let wz = q.w * q.z;

        let mut m = Mat4::identity();
        m.data[0] = 1.0 - 2.0 * (yy + zz);
        m.data[1] = 2.0 * (xy + wz);
        m.data[2] = 2.0 * (xz - wy);
        m.data[4] = 2.0 * (xy - wz);
        m.data[5] = 1.0 - 2.0 * (xx + zz);
        m.data[6] = 2.0 * (yz + wx);
        m.data[8] = 2.0 * (xz + wy);
        m.data[9] = 2.0 * (yz - wx);
        m.data[10] = 1.0 - 2.0 * (xx + yy);
        m
    }

    pub fn conjugate(&self) -> Quaternion {
        Quaternion { x: -self.x, y: -self.y, z: -self.z, w: self.w }
    }

    pub fn rotate_vec3(&self, v: &Vec3) -> Vec3 {
        let qv = Quaternion { x: v.x, y: v.y, z: v.z, w: 0.0 };
        let result = self.multiply(&qv).multiply(&self.conjugate());
        Vec3 { x: result.x, y: result.y, z: result.z }
    }
}

// ── Camera ─────────────────────────────────────────────────────

/// A 3D camera with perspective projection.
pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y: f64,
    pub aspect: f64,
    pub near: f64,
    pub far: f64,
}

impl Camera {
    /// Default camera at origin, looking along -Z.
    pub fn new() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 5.0),
            target: Vec3::zero(),
            up: Vec3::up(),
            fov_y: std::f64::consts::FRAC_PI_4,
            aspect: 16.0 / 9.0,
            near: 0.1,
            far: 1000.0,
        }
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at(&self.position, &self.target, &self.up)
    }

    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective(self.fov_y, self.aspect, self.near, self.far)
    }

    pub fn view_projection(&self) -> Mat4 {
        self.projection_matrix().multiply(&self.view_matrix())
    }

    /// Orbit around the target by horizontal/vertical deltas (in radians).
    pub fn orbit(&mut self, delta_x: f64, delta_y: f64) {
        let offset = self.position - self.target;
        let radius = offset.length();

        // Current spherical angles.
        let theta = offset.x.atan2(offset.z) + delta_x;
        let phi = (offset.y / radius).clamp(-1.0, 1.0).acos() - delta_y;
        let phi = phi.clamp(0.01, std::f64::consts::PI - 0.01);

        self.position = Vec3 {
            x: self.target.x + radius * phi.sin() * theta.sin(),
            y: self.target.y + radius * phi.cos(),
            z: self.target.z + radius * phi.sin() * theta.cos(),
        };
    }

    pub fn zoom(&mut self, factor: f64) {
        let dir = (self.position - self.target).normalize();
        let dist = self.position.distance(&self.target) * factor;
        self.position = self.target + dir * dist;
    }

    pub fn pan(&mut self, dx: f64, dy: f64) {
        let forward = (self.target - self.position).normalize();
        let right = forward.cross(&self.up).normalize();
        let up = right.cross(&forward);
        let offset = right * dx + up * dy;
        self.position = self.position + offset;
        self.target = self.target + offset;
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new()
    }
}

// ── Frustum ────────────────────────────────────────────────────

/// View frustum for culling, extracted from a view-projection matrix.
pub struct Frustum {
    pub planes: Vec<Vec4>,
}

impl Frustum {
    /// Extract 6 frustum planes from a view-projection matrix.
    pub fn from_view_projection(vp: &Mat4) -> Self {
        let m = &vp.data;
        let planes = vec![
            // Left
            Vec4::new(m[3] + m[0], m[7] + m[4], m[11] + m[8], m[15] + m[12]),
            // Right
            Vec4::new(m[3] - m[0], m[7] - m[4], m[11] - m[8], m[15] - m[12]),
            // Bottom
            Vec4::new(m[3] + m[1], m[7] + m[5], m[11] + m[9], m[15] + m[13]),
            // Top
            Vec4::new(m[3] - m[1], m[7] - m[5], m[11] - m[9], m[15] - m[13]),
            // Near
            Vec4::new(m[3] + m[2], m[7] + m[6], m[11] + m[10], m[15] + m[14]),
            // Far
            Vec4::new(m[3] - m[2], m[7] - m[6], m[11] - m[10], m[15] - m[14]),
        ];

        // Normalize planes.
        let planes = planes
            .into_iter()
            .map(|p| {
                let len = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
                if len < 1e-12 {
                    p
                } else {
                    Vec4::new(p.x / len, p.y / len, p.z / len, p.w / len)
                }
            })
            .collect();

        Self { planes }
    }

    /// Test whether a point is inside (or on) all frustum planes.
    pub fn contains_point(&self, point: &Vec3) -> bool {
        for plane in &self.planes {
            let dist = plane.x * point.x + plane.y * point.y + plane.z * point.z + plane.w;
            if dist < 0.0 {
                return false;
            }
        }
        true
    }

    /// Test whether a sphere intersects the frustum.
    pub fn intersects_sphere(&self, center: &Vec3, radius: f64) -> bool {
        for plane in &self.planes {
            let dist = plane.x * center.x + plane.y * center.y + plane.z * center.z + plane.w;
            if dist < -radius {
                return false;
            }
        }
        true
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EPS: f64 = 1e-8;

    #[test]
    fn vec3_dot_cross() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        assert!((a.dot(&b)).abs() < EPS);
        let c = a.cross(&b);
        assert!((c.x).abs() < EPS);
        assert!((c.y).abs() < EPS);
        assert!((c.z - 1.0).abs() < EPS);
    }

    #[test]
    fn normalize_unit_length() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let n = v.normalize();
        assert!((n.length() - 1.0).abs() < EPS);
        assert!((n.x - 0.6).abs() < EPS);
        assert!((n.y - 0.8).abs() < EPS);
    }

    #[test]
    fn mat4_identity_times_vec3() {
        let m = Mat4::identity();
        let v = Vec3::new(1.0, 2.0, 3.0);
        let r = m.transform_vec3(&v);
        assert!((r.x - 1.0).abs() < EPS);
        assert!((r.y - 2.0).abs() < EPS);
        assert!((r.z - 3.0).abs() < EPS);
    }

    #[test]
    fn perspective_projection() {
        let p = Mat4::perspective(PI / 4.0, 1.0, 0.1, 100.0);
        // The origin should project to origin.
        let v = p.transform_vec4(&Vec4::new(0.0, 0.0, -1.0, 1.0));
        // After perspective divide, x and y should be 0.
        assert!((v.x).abs() < EPS);
        assert!((v.y).abs() < EPS);
    }

    #[test]
    fn look_at_produces_correct_view() {
        let eye = Vec3::new(0.0, 0.0, 5.0);
        let target = Vec3::zero();
        let up = Vec3::up();
        let view = Mat4::look_at(&eye, &target, &up);
        // Origin should be at (0, 0, -5) in view space.
        let p = view.transform_vec3(&Vec3::zero());
        assert!((p.x).abs() < EPS);
        assert!((p.y).abs() < EPS);
        assert!((p.z - (-5.0)).abs() < EPS);
    }

    #[test]
    fn quaternion_from_axis_angle() {
        let q = Quaternion::from_axis_angle(&Vec3::up(), PI / 2.0);
        let v = q.rotate_vec3(&Vec3::new(1.0, 0.0, 0.0));
        // Rotating (1,0,0) 90 degrees around Y should give (0,0,-1).
        assert!((v.x).abs() < EPS);
        assert!((v.y).abs() < EPS);
        assert!((v.z - (-1.0)).abs() < EPS);
    }

    #[test]
    fn quaternion_slerp_interpolates() {
        let a = Quaternion::identity();
        let b = Quaternion::from_axis_angle(&Vec3::up(), PI);
        let mid = a.slerp(&b, 0.5);
        // Halfway should be 90 degrees.
        let v = mid.rotate_vec3(&Vec3::new(1.0, 0.0, 0.0));
        assert!((v.x).abs() < EPS);
        assert!((v.z - (-1.0)).abs() < EPS);
    }

    #[test]
    fn quaternion_to_mat4_roundtrip() {
        let q = Quaternion::from_axis_angle(&Vec3::new(1.0, 1.0, 0.0).normalize(), 1.0);
        let m = q.to_mat4();
        let v = Vec3::new(1.0, 2.0, 3.0);
        let via_quat = q.rotate_vec3(&v);
        let via_mat = m.transform_vec3(&v);
        assert!((via_quat.x - via_mat.x).abs() < EPS);
        assert!((via_quat.y - via_mat.y).abs() < EPS);
        assert!((via_quat.z - via_mat.z).abs() < EPS);
    }

    #[test]
    fn camera_orbit_changes_position() {
        let mut cam = Camera::new();
        let orig = cam.position;
        cam.orbit(0.5, 0.0);
        assert!((cam.position.x - orig.x).abs() > 0.01 || (cam.position.z - orig.z).abs() > 0.01);
        // Distance to target should remain the same.
        let orig_dist = orig.distance(&cam.target);
        let new_dist = cam.position.distance(&cam.target);
        assert!((orig_dist - new_dist).abs() < EPS);
    }

    #[test]
    fn frustum_contains_point_inside() {
        let cam = Camera::new();
        let vp = cam.view_projection();
        let frustum = Frustum::from_view_projection(&vp);
        // Target is at origin, should be inside.
        assert!(frustum.contains_point(&Vec3::zero()));
    }

    #[test]
    fn frustum_rejects_outside() {
        let cam = Camera::new();
        let vp = cam.view_projection();
        let frustum = Frustum::from_view_projection(&vp);
        // A point far behind the camera should be outside.
        assert!(!frustum.contains_point(&Vec3::new(0.0, 0.0, 100.0)));
    }

    #[test]
    fn mat4_inverse_times_original_is_identity() {
        let m = Mat4::translation(3.0, -7.0, 2.5)
            .multiply(&Mat4::rotation_y(1.2))
            .multiply(&Mat4::scaling(2.0, 0.5, 1.5));
        let inv = m.inverse().unwrap();
        let product = m.multiply(&inv);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product.at(i, j) - expected).abs() < 1e-6,
                    "at({i},{j}): {} != {expected}",
                    product.at(i, j)
                );
            }
        }
    }

    #[test]
    fn vec3_angle_between() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let angle = a.angle_between(&b);
        assert!((angle - PI / 2.0).abs() < EPS);
    }

    #[test]
    fn frustum_intersects_sphere() {
        let cam = Camera::new();
        let vp = cam.view_projection();
        let frustum = Frustum::from_view_projection(&vp);
        // Large sphere around origin should intersect.
        assert!(frustum.intersects_sphere(&Vec3::zero(), 10.0));
        // Tiny sphere far away should not.
        assert!(!frustum.intersects_sphere(&Vec3::new(0.0, 0.0, 10000.0), 0.01));
    }
}
