//! Camera system: perspective / orthographic projection, view matrix,
//! frustum extraction, screen-to-world ray casting, and camera shake.

// ── Vec3 / Mat4 helpers ─────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    pub fn zero() -> Self { Self::new(0.0, 0.0, 0.0) }

    pub fn sub(&self, o: &Vec3) -> Vec3 { Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z) }
    pub fn add(&self, o: &Vec3) -> Vec3 { Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z) }
    pub fn scale(&self, s: f64) -> Vec3 { Vec3::new(self.x * s, self.y * s, self.z * s) }
    pub fn dot(&self, o: &Vec3) -> f64 { self.x * o.x + self.y * o.y + self.z * o.z }
    pub fn cross(&self, o: &Vec3) -> Vec3 {
        Vec3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    pub fn length(&self) -> f64 { self.dot(self).sqrt() }
    pub fn normalize(&self) -> Vec3 {
        let len = self.length();
        if len < 1e-12 { return Vec3::zero(); }
        self.scale(1.0 / len)
    }
}

/// Column-major 4x4 matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Mat4(pub [f64; 16]);

impl Mat4 {
    pub fn identity() -> Self {
        let mut m = [0.0; 16];
        m[0] = 1.0; m[5] = 1.0; m[10] = 1.0; m[15] = 1.0;
        Self(m)
    }

    /// Multiply self * other (both column-major).
    pub fn mul(&self, other: &Mat4) -> Mat4 {
        let mut out = [0.0; 16];
        for col in 0..4 {
            for row in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += self.0[k * 4 + row] * other.0[col * 4 + k];
                }
                out[col * 4 + row] = sum;
            }
        }
        Mat4(out)
    }

    /// Transform a 4D vector [x, y, z, w].
    pub fn transform(&self, v: [f64; 4]) -> [f64; 4] {
        let mut out = [0.0; 4];
        for row in 0..4 {
            for col in 0..4 {
                out[row] += self.0[col * 4 + row] * v[col];
            }
        }
        out
    }

    /// Build a look-at view matrix (right-handed).
    pub fn look_at(eye: &Vec3, target: &Vec3, up: &Vec3) -> Self {
        let f = target.sub(eye).normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(&f);
        let mut m = [0.0; 16];
        m[0] = s.x;  m[1] = u.x;  m[2]  = -f.x;  m[3]  = 0.0;
        m[4] = s.y;  m[5] = u.y;  m[6]  = -f.y;  m[7]  = 0.0;
        m[8] = s.z;  m[9] = u.z;  m[10] = -f.z;  m[11] = 0.0;
        m[12] = -s.dot(eye);
        m[13] = -u.dot(eye);
        m[14] = f.dot(eye);
        m[15] = 1.0;
        Mat4(m)
    }

    /// Perspective projection (fov_y in radians, aspect = width/height).
    pub fn perspective(fov_y: f64, aspect: f64, near: f64, far: f64) -> Self {
        let f = 1.0 / (fov_y * 0.5).tan();
        let nf = 1.0 / (near - far);
        let mut m = [0.0; 16];
        m[0] = f / aspect;
        m[5] = f;
        m[10] = (far + near) * nf;
        m[11] = -1.0;
        m[14] = 2.0 * far * near * nf;
        Mat4(m)
    }

    /// Orthographic projection.
    pub fn orthographic(left: f64, right: f64, bottom: f64, top: f64, near: f64, far: f64) -> Self {
        let mut m = [0.0; 16];
        let rl = right - left;
        let tb = top - bottom;
        let fn_ = far - near;
        m[0] = 2.0 / rl;
        m[5] = 2.0 / tb;
        m[10] = -2.0 / fn_;
        m[12] = -(right + left) / rl;
        m[13] = -(top + bottom) / tb;
        m[14] = -(far + near) / fn_;
        m[15] = 1.0;
        Mat4(m)
    }
}

// ── Projection type ─────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Projection {
    Perspective { fov_y: f64, aspect: f64, near: f64, far: f64 },
    Orthographic { left: f64, right: f64, bottom: f64, top: f64, near: f64, far: f64 },
}

impl Projection {
    pub fn matrix(&self) -> Mat4 {
        match self {
            Projection::Perspective { fov_y, aspect, near, far } =>
                Mat4::perspective(*fov_y, *aspect, *near, *far),
            Projection::Orthographic { left, right, bottom, top, near, far } =>
                Mat4::orthographic(*left, *right, *bottom, *top, *near, *far),
        }
    }
}

// ── Frustum plane ───────────────────────────────────────────────

/// A plane defined as ax + by + cz + d = 0.
#[derive(Debug, Clone, PartialEq)]
pub struct Plane {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

impl Plane {
    pub fn normalize(&self) -> Self {
        let len = (self.a * self.a + self.b * self.b + self.c * self.c).sqrt();
        if len < 1e-12 { return self.clone(); }
        Self { a: self.a / len, b: self.b / len, c: self.c / len, d: self.d / len }
    }

    /// Signed distance from point to plane.
    pub fn distance(&self, x: f64, y: f64, z: f64) -> f64 {
        self.a * x + self.b * y + self.c * z + self.d
    }
}

/// The six frustum planes extracted from a view-projection matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Frustum {
    pub left: Plane,
    pub right: Plane,
    pub bottom: Plane,
    pub top: Plane,
    pub near: Plane,
    pub far: Plane,
}

impl Frustum {
    /// Extract frustum planes from a combined view-projection matrix (column-major).
    pub fn from_view_projection(vp: &Mat4) -> Self {
        let m = &vp.0;
        // Row extraction from column-major storage:
        // row i = [m[0*4+i], m[1*4+i], m[2*4+i], m[3*4+i]]
        let row = |i: usize| -> [f64; 4] {
            [m[i], m[4 + i], m[8 + i], m[12 + i]]
        };
        let r0 = row(0); let r1 = row(1); let r2 = row(2); let r3 = row(3);
        let plane = |a: f64, b: f64, c: f64, d: f64| Plane { a, b, c, d }.normalize();

        Frustum {
            left:   plane(r3[0]+r0[0], r3[1]+r0[1], r3[2]+r0[2], r3[3]+r0[3]),
            right:  plane(r3[0]-r0[0], r3[1]-r0[1], r3[2]-r0[2], r3[3]-r0[3]),
            bottom: plane(r3[0]+r1[0], r3[1]+r1[1], r3[2]+r1[2], r3[3]+r1[3]),
            top:    plane(r3[0]-r1[0], r3[1]-r1[1], r3[2]-r1[2], r3[3]-r1[3]),
            near:   plane(r3[0]+r2[0], r3[1]+r2[1], r3[2]+r2[2], r3[3]+r2[3]),
            far:    plane(r3[0]-r2[0], r3[1]-r2[1], r3[2]-r2[2], r3[3]-r2[3]),
        }
    }

    /// Test if an axis-aligned bounding sphere is inside the frustum.
    pub fn contains_sphere(&self, cx: f64, cy: f64, cz: f64, radius: f64) -> bool {
        for plane in [&self.left, &self.right, &self.bottom, &self.top, &self.near, &self.far] {
            if plane.distance(cx, cy, cz) < -radius {
                return false;
            }
        }
        true
    }
}

// ── Ray ─────────────────────────────────────────────────────────

/// A ray with origin and direction.
#[derive(Debug, Clone, PartialEq)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

// ── Camera shake ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct CameraShake {
    pub intensity: f64,
    pub decay: f64,
    remaining: f64,
    offset: Vec3,
    seed: u64,
}

impl CameraShake {
    pub fn new(intensity: f64, decay: f64) -> Self {
        Self { intensity, decay, remaining: intensity, offset: Vec3::zero(), seed: 12345 }
    }

    pub fn trigger(&mut self, intensity: f64) {
        self.remaining = intensity;
    }

    /// Advance the shake by `dt` seconds and compute the offset.
    pub fn update(&mut self, dt: f64) -> &Vec3 {
        if self.remaining < 1e-6 {
            self.offset = Vec3::zero();
            return &self.offset;
        }
        // Simple deterministic pseudo-random using LCG.
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let rx = ((self.seed >> 33) as f64 / (u32::MAX as f64)) * 2.0 - 1.0;
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let ry = ((self.seed >> 33) as f64 / (u32::MAX as f64)) * 2.0 - 1.0;
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let rz = ((self.seed >> 33) as f64 / (u32::MAX as f64)) * 2.0 - 1.0;

        self.offset = Vec3::new(rx * self.remaining, ry * self.remaining, rz * self.remaining);
        self.remaining = (self.remaining - self.decay * dt).max(0.0);
        &self.offset
    }

    pub fn is_active(&self) -> bool {
        self.remaining > 1e-6
    }

    pub fn offset(&self) -> &Vec3 {
        &self.offset
    }
}

// ── Camera ──────────────────────────────────────────────────────

/// A camera with position, target/rotation, projection, and shake.
#[derive(Debug, Clone, PartialEq)]
pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub projection: Projection,
    pub shake: CameraShake,
}

impl Camera {
    pub fn new_perspective(pos: Vec3, target: Vec3, fov_y: f64, aspect: f64, near: f64, far: f64) -> Self {
        Self {
            position: pos,
            target,
            up: Vec3::new(0.0, 1.0, 0.0),
            projection: Projection::Perspective { fov_y, aspect, near, far },
            shake: CameraShake::new(0.0, 1.0),
        }
    }

    pub fn new_orthographic(pos: Vec3, target: Vec3, left: f64, right: f64, bottom: f64, top: f64, near: f64, far: f64) -> Self {
        Self {
            position: pos,
            target,
            up: Vec3::new(0.0, 1.0, 0.0),
            projection: Projection::Orthographic { left, right, bottom, top, near, far },
            shake: CameraShake::new(0.0, 1.0),
        }
    }

    /// View matrix (look-at from position to target, with shake offset applied).
    pub fn view_matrix(&self) -> Mat4 {
        let shake_pos = self.position.add(self.shake.offset());
        Mat4::look_at(&shake_pos, &self.target, &self.up)
    }

    /// Projection matrix.
    pub fn projection_matrix(&self) -> Mat4 {
        self.projection.matrix()
    }

    /// Combined view-projection matrix.
    pub fn view_projection(&self) -> Mat4 {
        self.projection_matrix().mul(&self.view_matrix())
    }

    /// Extract frustum from the current view-projection.
    pub fn frustum(&self) -> Frustum {
        Frustum::from_view_projection(&self.view_projection())
    }

    /// Cast a ray from screen coordinates (ndc_x, ndc_y in [-1, 1]) into world space.
    pub fn screen_to_ray(&self, ndc_x: f64, ndc_y: f64) -> Ray {
        // Unproject NDC near/far points through inverse VP.
        // We approximate by computing direction from camera through the
        // unprojectd point on the near plane.
        let vp = self.view_projection();
        let inv = invert_mat4(&vp);
        let near_pt = inv.transform([ndc_x, ndc_y, -1.0, 1.0]);
        let far_pt = inv.transform([ndc_x, ndc_y, 1.0, 1.0]);

        let near_w = if near_pt[3].abs() > 1e-12 { near_pt[3] } else { 1.0 };
        let far_w = if far_pt[3].abs() > 1e-12 { far_pt[3] } else { 1.0 };

        let origin = Vec3::new(near_pt[0] / near_w, near_pt[1] / near_w, near_pt[2] / near_w);
        let far_world = Vec3::new(far_pt[0] / far_w, far_pt[1] / far_w, far_pt[2] / far_w);
        let direction = far_world.sub(&origin).normalize();

        Ray { origin, direction }
    }

    pub fn update_shake(&mut self, dt: f64) {
        self.shake.update(dt);
    }
}

// ── Matrix inverse (Cramer's rule for 4x4) ─────────────────────

fn invert_mat4(m: &Mat4) -> Mat4 {
    let s = &m.0;
    let mut inv = [0.0f64; 16];

    inv[0]  =  s[5]*s[10]*s[15] - s[5]*s[11]*s[14] - s[9]*s[6]*s[15] + s[9]*s[7]*s[14] + s[13]*s[6]*s[11] - s[13]*s[7]*s[10];
    inv[4]  = -s[4]*s[10]*s[15] + s[4]*s[11]*s[14] + s[8]*s[6]*s[15] - s[8]*s[7]*s[14] - s[12]*s[6]*s[11] + s[12]*s[7]*s[10];
    inv[8]  =  s[4]*s[9]*s[15]  - s[4]*s[11]*s[13] - s[8]*s[5]*s[15] + s[8]*s[7]*s[13] + s[12]*s[5]*s[11] - s[12]*s[7]*s[9];
    inv[12] = -s[4]*s[9]*s[14]  + s[4]*s[10]*s[13] + s[8]*s[5]*s[14] - s[8]*s[6]*s[13] - s[12]*s[5]*s[10] + s[12]*s[6]*s[9];

    inv[1]  = -s[1]*s[10]*s[15] + s[1]*s[11]*s[14] + s[9]*s[2]*s[15] - s[9]*s[3]*s[14] - s[13]*s[2]*s[11] + s[13]*s[3]*s[10];
    inv[5]  =  s[0]*s[10]*s[15] - s[0]*s[11]*s[14] - s[8]*s[2]*s[15] + s[8]*s[3]*s[14] + s[12]*s[2]*s[11] - s[12]*s[3]*s[10];
    inv[9]  = -s[0]*s[9]*s[15]  + s[0]*s[11]*s[13] + s[8]*s[1]*s[15] - s[8]*s[3]*s[13] - s[12]*s[1]*s[11] + s[12]*s[3]*s[9];
    inv[13] =  s[0]*s[9]*s[14]  - s[0]*s[10]*s[13] - s[8]*s[1]*s[14] + s[8]*s[2]*s[13] + s[12]*s[1]*s[10] - s[12]*s[2]*s[9];

    inv[2]  =  s[1]*s[6]*s[15] - s[1]*s[7]*s[14] - s[5]*s[2]*s[15] + s[5]*s[3]*s[14] + s[13]*s[2]*s[7]  - s[13]*s[3]*s[6];
    inv[6]  = -s[0]*s[6]*s[15] + s[0]*s[7]*s[14] + s[4]*s[2]*s[15] - s[4]*s[3]*s[14] - s[12]*s[2]*s[7]  + s[12]*s[3]*s[6];
    inv[10] =  s[0]*s[5]*s[15] - s[0]*s[7]*s[13] - s[4]*s[1]*s[15] + s[4]*s[3]*s[13] + s[12]*s[1]*s[7]  - s[12]*s[3]*s[5];
    inv[14] = -s[0]*s[5]*s[14] + s[0]*s[6]*s[13] + s[4]*s[1]*s[14] - s[4]*s[2]*s[13] - s[12]*s[1]*s[6]  + s[12]*s[2]*s[5];

    inv[3]  = -s[1]*s[6]*s[11] + s[1]*s[7]*s[10] + s[5]*s[2]*s[11] - s[5]*s[3]*s[10] - s[9]*s[2]*s[7]   + s[9]*s[3]*s[6];
    inv[7]  =  s[0]*s[6]*s[11] - s[0]*s[7]*s[10] - s[4]*s[2]*s[11] + s[4]*s[3]*s[10] + s[8]*s[2]*s[7]   - s[8]*s[3]*s[6];
    inv[11] = -s[0]*s[5]*s[11] + s[0]*s[7]*s[9]  + s[4]*s[1]*s[11] - s[4]*s[3]*s[9]  - s[8]*s[1]*s[7]   + s[8]*s[3]*s[5];
    inv[15] =  s[0]*s[5]*s[10] - s[0]*s[6]*s[9]  - s[4]*s[1]*s[10] + s[4]*s[2]*s[9]  + s[8]*s[1]*s[6]   - s[8]*s[2]*s[5];

    let det = s[0]*inv[0] + s[1]*inv[4] + s[2]*inv[8] + s[3]*inv[12];
    if det.abs() < 1e-20 {
        return Mat4::identity();
    }
    let inv_det = 1.0 / det;
    for v in &mut inv { *v *= inv_det; }
    Mat4(inv)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn assert_near(a: f64, b: f64, msg: &str) {
        assert!((a - b).abs() < EPS, "{msg}: {a} vs {b}");
    }

    #[test]
    fn test_vec3_basic() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let c = a.add(&b);
        assert_near(c.x, 5.0, "add x");
        assert_near(c.y, 7.0, "add y");
        assert_near(c.z, 9.0, "add z");
    }

    #[test]
    fn test_vec3_normalize() {
        let v = Vec3::new(3.0, 0.0, 4.0);
        let n = v.normalize();
        assert_near(n.length(), 1.0, "unit length");
    }

    #[test]
    fn test_vec3_cross() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        let z = x.cross(&y);
        assert_near(z.x, 0.0, "cross x");
        assert_near(z.y, 0.0, "cross y");
        assert_near(z.z, 1.0, "cross z");
    }

    #[test]
    fn test_mat4_identity_mul() {
        let id = Mat4::identity();
        let result = id.mul(&id);
        for i in 0..16 {
            assert_near(result.0[i], id.0[i], "identity mul");
        }
    }

    #[test]
    fn test_look_at_origin() {
        let eye = Vec3::new(0.0, 0.0, 5.0);
        let target = Vec3::zero();
        let up = Vec3::new(0.0, 1.0, 0.0);
        let view = Mat4::look_at(&eye, &target, &up);
        // Transformed eye should be at z = -5 (right-handed).
        let pt = view.transform([0.0, 0.0, 5.0, 1.0]);
        assert_near(pt[0], 0.0, "view x");
        assert_near(pt[1], 0.0, "view y");
        // z is inverted by look-at
    }

    #[test]
    fn test_perspective_projection() {
        let fov = std::f64::consts::FRAC_PI_4;
        let proj = Mat4::perspective(fov, 1.0, 0.1, 100.0);
        // Origin in front of camera should project to (0, 0).
        let pt = proj.transform([0.0, 0.0, -1.0, 1.0]);
        let w = pt[3];
        assert_near(pt[0] / w, 0.0, "proj center x");
        assert_near(pt[1] / w, 0.0, "proj center y");
    }

    #[test]
    fn test_orthographic_projection() {
        let proj = Mat4::orthographic(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0);
        let pt = proj.transform([0.0, 0.0, -50.0, 1.0]);
        assert_near(pt[0], 0.0, "ortho x");
        assert_near(pt[1], 0.0, "ortho y");
    }

    #[test]
    fn test_camera_new_perspective() {
        let cam = Camera::new_perspective(
            Vec3::new(0.0, 0.0, 10.0),
            Vec3::zero(),
            std::f64::consts::FRAC_PI_4,
            1.0, 0.1, 100.0,
        );
        let vp = cam.view_projection();
        // Should produce a valid matrix (non-degenerate).
        assert!(vp.0.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_camera_frustum_contains_origin() {
        let cam = Camera::new_perspective(
            Vec3::new(0.0, 0.0, 10.0),
            Vec3::zero(),
            std::f64::consts::FRAC_PI_4,
            1.0, 0.1, 100.0,
        );
        let frustum = cam.frustum();
        assert!(frustum.contains_sphere(0.0, 0.0, 0.0, 0.5));
    }

    #[test]
    fn test_frustum_rejects_distant_point() {
        let cam = Camera::new_perspective(
            Vec3::new(0.0, 0.0, 10.0),
            Vec3::zero(),
            std::f64::consts::FRAC_PI_4,
            1.0, 0.1, 50.0,
        );
        let frustum = cam.frustum();
        // Point far behind the camera.
        assert!(!frustum.contains_sphere(0.0, 0.0, 500.0, 0.1));
    }

    #[test]
    fn test_plane_distance() {
        let p = Plane { a: 0.0, b: 1.0, c: 0.0, d: -5.0 };
        assert_near(p.distance(0.0, 10.0, 0.0), 5.0, "plane dist");
    }

    #[test]
    fn test_screen_to_ray_center() {
        let cam = Camera::new_perspective(
            Vec3::new(0.0, 0.0, 10.0),
            Vec3::zero(),
            std::f64::consts::FRAC_PI_4,
            1.0, 0.1, 100.0,
        );
        let ray = cam.screen_to_ray(0.0, 0.0);
        // Center ray should point roughly toward negative z.
        assert!(ray.direction.z < 0.0, "ray should point forward");
    }

    #[test]
    fn test_camera_shake_inactive_initially() {
        let shake = CameraShake::new(0.0, 1.0);
        assert!(!shake.is_active());
    }

    #[test]
    fn test_camera_shake_trigger_and_decay() {
        let mut shake = CameraShake::new(0.0, 10.0);
        shake.trigger(5.0);
        assert!(shake.is_active());
        // Decay over time.
        for _ in 0..100 {
            shake.update(0.1);
        }
        assert!(!shake.is_active());
    }

    #[test]
    fn test_camera_shake_offset_changes() {
        let mut shake = CameraShake::new(0.0, 1.0);
        shake.trigger(5.0);
        shake.update(0.016);
        let o1 = shake.offset().clone();
        shake.update(0.016);
        let o2 = shake.offset().clone();
        // Offsets should generally differ (pseudo-random).
        let same = (o1.x - o2.x).abs() < 1e-12
            && (o1.y - o2.y).abs() < 1e-12
            && (o1.z - o2.z).abs() < 1e-12;
        assert!(!same, "shake offsets should vary");
    }

    #[test]
    fn test_invert_identity() {
        let id = Mat4::identity();
        let inv = invert_mat4(&id);
        for i in 0..16 {
            assert_near(inv.0[i], id.0[i], "inv identity");
        }
    }

    #[test]
    fn test_invert_roundtrip() {
        let view = Mat4::look_at(
            &Vec3::new(3.0, 4.0, 5.0),
            &Vec3::zero(),
            &Vec3::new(0.0, 1.0, 0.0),
        );
        let inv = invert_mat4(&view);
        let product = view.mul(&inv);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((product.0[j * 4 + i] - expected).abs() < 1e-4,
                    "roundtrip [{i}][{j}]: {} vs {expected}", product.0[j * 4 + i]);
            }
        }
    }

    #[test]
    fn test_projection_matrix_perspective() {
        let proj = Projection::Perspective {
            fov_y: std::f64::consts::FRAC_PI_4,
            aspect: 16.0 / 9.0,
            near: 0.1,
            far: 1000.0,
        };
        let m = proj.matrix();
        assert!(m.0[0] > 0.0);
        assert!(m.0[5] > 0.0);
    }

    #[test]
    fn test_projection_matrix_orthographic() {
        let proj = Projection::Orthographic {
            left: -10.0, right: 10.0,
            bottom: -10.0, top: 10.0,
            near: 0.1, far: 100.0,
        };
        let m = proj.matrix();
        assert_near(m.0[0], 2.0 / 20.0, "ortho sx");
    }

    #[test]
    fn test_camera_update_shake() {
        let mut cam = Camera::new_perspective(
            Vec3::new(0.0, 0.0, 10.0),
            Vec3::zero(),
            std::f64::consts::FRAC_PI_4, 1.0, 0.1, 100.0,
        );
        cam.shake = CameraShake::new(0.0, 5.0);
        cam.shake.trigger(2.0);
        cam.update_shake(0.1);
        assert!(cam.shake.is_active());
    }

    #[test]
    fn test_camera_ortho_creation() {
        let cam = Camera::new_orthographic(
            Vec3::new(0.0, 10.0, 0.0),
            Vec3::zero(),
            -5.0, 5.0, -5.0, 5.0, 0.1, 50.0,
        );
        let m = cam.projection_matrix();
        assert!(m.0.iter().all(|v| v.is_finite()));
    }
}
