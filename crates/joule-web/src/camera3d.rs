//! Camera3D — perspective/orthographic projection, view matrix, frustum extraction,
//! ray casting (unproject), and orbit camera controller.

use crate::webgl::{Frustum, Mat4, Vec3, Vec4};

// ── Projection ────────────────────────────────────────────────

/// Projection type for a camera.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Projection {
    Perspective {
        fov_y: f64,
        aspect: f64,
        near: f64,
        far: f64,
    },
    Orthographic {
        left: f64,
        right: f64,
        bottom: f64,
        top: f64,
        near: f64,
        far: f64,
    },
}

impl Projection {
    pub fn perspective(fov_y: f64, aspect: f64, near: f64, far: f64) -> Self {
        Self::Perspective { fov_y, aspect, near, far }
    }

    pub fn orthographic(left: f64, right: f64, bottom: f64, top: f64, near: f64, far: f64) -> Self {
        Self::Orthographic { left, right, bottom, top, near, far }
    }

    /// Build the 4x4 projection matrix.
    pub fn matrix(&self) -> Mat4 {
        match *self {
            Projection::Perspective { fov_y, aspect, near, far } => {
                Mat4::perspective(fov_y, aspect, near, far)
            }
            Projection::Orthographic { left, right, bottom, top, near, far } => {
                Mat4::orthographic(left, right, bottom, top, near, far)
            }
        }
    }
}

// ── Ray ───────────────────────────────────────────────────────

/// A ray with origin and direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
        }
    }

    /// Evaluate the ray at parameter t: origin + t * direction.
    pub fn at(&self, t: f64) -> Vec3 {
        self.origin + self.direction * t
    }
}

// ── Camera3D ──────────────────────────────────────────────────

/// A 3D camera with configurable projection.
pub struct Camera3D {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub projection: Projection,
}

impl Camera3D {
    pub fn new(projection: Projection) -> Self {
        Self {
            eye: Vec3::new(0.0, 0.0, 5.0),
            target: Vec3::zero(),
            up: Vec3::up(),
            projection,
        }
    }

    /// Build the view matrix (lookAt).
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at(&self.eye, &self.target, &self.up)
    }

    /// Build the projection matrix.
    pub fn projection_matrix(&self) -> Mat4 {
        self.projection.matrix()
    }

    /// View * Projection combined.
    pub fn view_projection(&self) -> Mat4 {
        self.projection_matrix().multiply(&self.view_matrix())
    }

    /// Extract the 6-plane frustum from the view-projection matrix.
    pub fn frustum(&self) -> Frustum {
        Frustum::from_view_projection(&self.view_projection())
    }

    /// Cast a ray from screen coordinates (NDC: x,y in [-1,1], z=depth).
    /// Returns a world-space ray through the given screen point.
    pub fn screen_to_ray(&self, ndc_x: f64, ndc_y: f64) -> Option<Ray> {
        let vp = self.view_projection();
        let inv = vp.inverse()?;

        // Unproject near and far points.
        let near_ndc = Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
        let far_ndc = Vec4::new(ndc_x, ndc_y, 1.0, 1.0);

        let near_world = inv.transform_vec4(&near_ndc);
        let far_world = inv.transform_vec4(&far_ndc);

        if near_world.w.abs() < 1e-12 || far_world.w.abs() < 1e-12 {
            return None;
        }

        let near_pt = Vec3::new(
            near_world.x / near_world.w,
            near_world.y / near_world.w,
            near_world.z / near_world.w,
        );
        let far_pt = Vec3::new(
            far_world.x / far_world.w,
            far_world.y / far_world.w,
            far_world.z / far_world.w,
        );

        let dir = (far_pt - near_pt).normalize();
        Some(Ray::new(near_pt, dir))
    }

    /// Unproject a screen point (NDC x, y, depth in [-1,1]) to world space.
    pub fn unproject(&self, ndc_x: f64, ndc_y: f64, ndc_z: f64) -> Option<Vec3> {
        let vp = self.view_projection();
        let inv = vp.inverse()?;
        let clip = Vec4::new(ndc_x, ndc_y, ndc_z, 1.0);
        let world = inv.transform_vec4(&clip);
        if world.w.abs() < 1e-12 {
            return None;
        }
        Some(Vec3::new(world.x / world.w, world.y / world.w, world.z / world.w))
    }
}

// ── OrbitController ───────────────────────────────────────────

/// Orbit camera controller — spherical coordinates around a target.
pub struct OrbitController {
    pub yaw: f64,
    pub pitch: f64,
    pub distance: f64,
    pub target: Vec3,
    pub min_distance: f64,
    pub max_distance: f64,
    pub min_pitch: f64,
    pub max_pitch: f64,
}

impl OrbitController {
    pub fn new(distance: f64) -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            distance,
            target: Vec3::zero(),
            min_distance: 0.1,
            max_distance: 1000.0,
            min_pitch: -std::f64::consts::FRAC_PI_2 + 0.01,
            max_pitch: std::f64::consts::FRAC_PI_2 - 0.01,
        }
    }

    /// Rotate the camera by delta yaw and pitch (radians).
    pub fn rotate(&mut self, delta_yaw: f64, delta_pitch: f64) {
        self.yaw += delta_yaw;
        self.pitch = (self.pitch + delta_pitch).clamp(self.min_pitch, self.max_pitch);
    }

    /// Zoom by changing the distance.
    pub fn zoom(&mut self, delta: f64) {
        self.distance = (self.distance + delta).clamp(self.min_distance, self.max_distance);
    }

    /// Pan the target point in the camera's local XY plane.
    pub fn pan(&mut self, dx: f64, dy: f64) {
        let right = Vec3::new(self.yaw.cos(), 0.0, -self.yaw.sin());
        let up = Vec3::up();
        self.target = self.target + right * dx + up * dy;
    }

    /// Compute the camera eye position from spherical coordinates.
    pub fn eye_position(&self) -> Vec3 {
        let x = self.distance * self.pitch.cos() * self.yaw.sin();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * self.pitch.cos() * self.yaw.cos();
        self.target + Vec3::new(x, y, z)
    }

    /// Apply this controller's state to a Camera3D.
    pub fn apply(&self, camera: &mut Camera3D) {
        camera.eye = self.eye_position();
        camera.target = self.target;
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    const EPS: f64 = 1e-6;

    fn default_perspective() -> Camera3D {
        Camera3D::new(Projection::perspective(FRAC_PI_4, 16.0 / 9.0, 0.1, 1000.0))
    }

    #[test]
    fn perspective_matrix_nonzero() {
        let p = Projection::perspective(FRAC_PI_4, 1.0, 0.1, 100.0);
        let m = p.matrix();
        assert!(m.data[0].abs() > EPS);
        assert!(m.data[5].abs() > EPS);
    }

    #[test]
    fn orthographic_matrix_nonzero() {
        let p = Projection::orthographic(-10.0, 10.0, -10.0, 10.0, 0.1, 100.0);
        let m = p.matrix();
        assert!(m.data[0].abs() > EPS);
        assert_eq!(m.data[15], 1.0);
    }

    #[test]
    fn view_matrix_origin_maps_correctly() {
        let cam = default_perspective();
        let view = cam.view_matrix();
        let p = view.transform_vec3(&Vec3::zero());
        // Camera is at (0,0,5) looking at origin => origin at (0,0,-5) in view space.
        assert!((p.x).abs() < EPS);
        assert!((p.y).abs() < EPS);
        assert!((p.z + 5.0).abs() < EPS);
    }

    #[test]
    fn frustum_contains_target() {
        let cam = default_perspective();
        let frustum = cam.frustum();
        assert!(frustum.contains_point(&Vec3::zero()));
    }

    #[test]
    fn screen_to_ray_center() {
        let cam = default_perspective();
        let ray = cam.screen_to_ray(0.0, 0.0).unwrap();
        // Center ray should point roughly from eye toward target.
        let expected_dir = (cam.target - cam.eye).normalize();
        let dot = ray.direction.dot(&expected_dir);
        assert!(dot > 0.99, "dot = {dot}");
    }

    #[test]
    fn unproject_near_plane_center() {
        let cam = default_perspective();
        let pt = cam.unproject(0.0, 0.0, -1.0).unwrap();
        // Should be on the near plane, close to the eye-target line.
        let dir = (pt - cam.eye).normalize();
        let expected = (cam.target - cam.eye).normalize();
        let dot = dir.dot(&expected);
        assert!(dot > 0.99, "dot = {dot}");
    }

    #[test]
    fn orbit_controller_eye_at_distance() {
        let ctrl = OrbitController::new(10.0);
        let eye = ctrl.eye_position();
        let dist = eye.distance(&ctrl.target);
        assert!((dist - 10.0).abs() < EPS);
    }

    #[test]
    fn orbit_controller_rotate() {
        let mut ctrl = OrbitController::new(5.0);
        ctrl.rotate(PI / 4.0, 0.0);
        assert!((ctrl.yaw - PI / 4.0).abs() < EPS);
        let eye = ctrl.eye_position();
        let dist = eye.distance(&ctrl.target);
        assert!((dist - 5.0).abs() < EPS);
    }

    #[test]
    fn orbit_controller_pitch_clamped() {
        let mut ctrl = OrbitController::new(5.0);
        ctrl.rotate(0.0, 100.0); // huge pitch
        assert!(ctrl.pitch <= ctrl.max_pitch);
    }

    #[test]
    fn orbit_controller_zoom() {
        let mut ctrl = OrbitController::new(5.0);
        ctrl.zoom(3.0);
        assert!((ctrl.distance - 8.0).abs() < EPS);
        ctrl.zoom(-100.0);
        assert!((ctrl.distance - ctrl.min_distance).abs() < EPS);
    }

    #[test]
    fn orbit_controller_apply_to_camera() {
        let mut cam = default_perspective();
        let mut ctrl = OrbitController::new(10.0);
        ctrl.rotate(0.5, 0.3);
        ctrl.apply(&mut cam);
        assert_eq!(cam.target, ctrl.target);
        let dist = cam.eye.distance(&cam.target);
        assert!((dist - 10.0).abs() < EPS);
    }

    #[test]
    fn ray_at_parameter() {
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0));
        let pt = ray.at(5.0);
        assert!((pt.z - (-5.0)).abs() < EPS);
    }
}
