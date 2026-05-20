//! # Hybrid Force-Position Control
//!
//! Implements hybrid force-position control for robotic manipulators.
//! Uses selection matrices to partition task space into force-controlled
//! and position-controlled subspaces, with task-frame formulation.

use std::fmt;

// ── Vector & Matrix Types ──

/// 6-DOF wrench or twist vector (3 linear + 3 angular).
#[derive(Clone, Debug)]
pub struct Vec6 {
    pub data: [f64; 6],
}

impl Vec6 {
    pub fn zeros() -> Self {
        Self { data: [0.0; 6] }
    }

    pub fn new(v0: f64, v1: f64, v2: f64, v3: f64, v4: f64, v5: f64) -> Self {
        Self { data: [v0, v1, v2, v3, v4, v5] }
    }

    pub fn from_slice(s: &[f64]) -> Self {
        let mut data = [0.0; 6];
        for (i, v) in s.iter().take(6).enumerate() {
            data[i] = *v;
        }
        Self { data }
    }

    pub fn linear(&self) -> [f64; 3] {
        [self.data[0], self.data[1], self.data[2]]
    }

    pub fn angular(&self) -> [f64; 3] {
        [self.data[3], self.data[4], self.data[5]]
    }

    pub fn norm(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    pub fn scale(&self, s: f64) -> Self {
        let mut data = [0.0; 6];
        for i in 0..6 {
            data[i] = self.data[i] * s;
        }
        Self { data }
    }

    pub fn add(&self, other: &Self) -> Self {
        let mut data = [0.0; 6];
        for i in 0..6 {
            data[i] = self.data[i] + other.data[i];
        }
        Self { data }
    }

    pub fn sub(&self, other: &Self) -> Self {
        let mut data = [0.0; 6];
        for i in 0..6 {
            data[i] = self.data[i] - other.data[i];
        }
        Self { data }
    }
}

impl fmt::Display for Vec6 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.3}, {:.3}, {:.3} | {:.3}, {:.3}, {:.3}]",
            self.data[0], self.data[1], self.data[2],
            self.data[3], self.data[4], self.data[5])
    }
}

// ── Selection Matrix ──

/// Diagonal selection matrix that partitions task space into force and position subspaces.
/// S[i] = 1 means axis i is force-controlled; S[i] = 0 means position-controlled.
#[derive(Clone, Debug)]
pub struct SelectionMatrix {
    diagonal: [f64; 6],
}

impl SelectionMatrix {
    /// All position-controlled.
    pub fn all_position() -> Self {
        Self { diagonal: [0.0; 6] }
    }

    /// All force-controlled.
    pub fn all_force() -> Self {
        Self { diagonal: [1.0; 6] }
    }

    /// Custom: specify which axes are force-controlled.
    pub fn new(force_axes: &[bool; 6]) -> Self {
        let mut diagonal = [0.0; 6];
        for (i, &is_force) in force_axes.iter().enumerate() {
            diagonal[i] = if is_force { 1.0 } else { 0.0 };
        }
        Self { diagonal }
    }

    /// Set a specific axis to force control.
    pub fn with_force_axis(mut self, axis: usize) -> Self {
        if axis < 6 { self.diagonal[axis] = 1.0; }
        self
    }

    /// Set a specific axis to position control.
    pub fn with_position_axis(mut self, axis: usize) -> Self {
        if axis < 6 { self.diagonal[axis] = 0.0; }
        self
    }

    /// Apply S to a vector: S * v (force-controlled components).
    pub fn apply_force(&self, v: &Vec6) -> Vec6 {
        let mut data = [0.0; 6];
        for i in 0..6 {
            data[i] = self.diagonal[i] * v.data[i];
        }
        Vec6 { data }
    }

    /// Apply (I - S) to a vector: position-controlled components.
    pub fn apply_position(&self, v: &Vec6) -> Vec6 {
        let mut data = [0.0; 6];
        for i in 0..6 {
            data[i] = (1.0 - self.diagonal[i]) * v.data[i];
        }
        Vec6 { data }
    }

    /// Complement of this selection matrix.
    pub fn complement(&self) -> Self {
        let mut diag = [0.0; 6];
        for i in 0..6 {
            diag[i] = 1.0 - self.diagonal[i];
        }
        Self { diagonal: diag }
    }

    pub fn is_force_axis(&self, axis: usize) -> bool {
        axis < 6 && self.diagonal[axis] > 0.5
    }

    pub fn num_force_axes(&self) -> usize {
        self.diagonal.iter().filter(|&&d| d > 0.5).count()
    }

    pub fn num_position_axes(&self) -> usize {
        6 - self.num_force_axes()
    }
}

impl fmt::Display for SelectionMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let labels: Vec<&str> = self.diagonal.iter()
            .map(|d| if *d > 0.5 { "F" } else { "P" })
            .collect();
        write!(f, "S[{} {} {} | {} {} {}]",
            labels[0], labels[1], labels[2], labels[3], labels[4], labels[5])
    }
}

// ── PID Controllers ──

/// Simple PID controller for one axis.
#[derive(Clone, Debug)]
pub struct PidController {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    integral: f64,
    prev_error: f64,
    integral_limit: f64,
}

impl PidController {
    pub fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self { kp, ki, kd, integral: 0.0, prev_error: 0.0, integral_limit: 100.0 }
    }

    pub fn with_integral_limit(mut self, limit: f64) -> Self {
        self.integral_limit = limit;
        self
    }

    pub fn compute(&mut self, error: f64, dt: f64) -> f64 {
        self.integral += error * dt;
        self.integral = self.integral.clamp(-self.integral_limit, self.integral_limit);
        let derivative = if dt > 0.0 { (error - self.prev_error) / dt } else { 0.0 };
        self.prev_error = error;
        self.kp * error + self.ki * self.integral + self.kd * derivative
    }

    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
    }
}

impl fmt::Display for PidController {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PID(kp={:.2}, ki={:.2}, kd={:.2})", self.kp, self.ki, self.kd)
    }
}

// ── Hybrid Controller ──

/// Hybrid force-position controller.
#[derive(Clone, Debug)]
pub struct HybridController {
    selection: SelectionMatrix,
    force_pids: Vec<PidController>,
    position_pids: Vec<PidController>,
    force_desired: Vec6,
    position_desired: Vec6,
    output_limit: f64,
}

impl HybridController {
    pub fn new(selection: SelectionMatrix) -> Self {
        let force_pids = (0..6).map(|_| PidController::new(1.0, 0.0, 0.1)).collect();
        let position_pids = (0..6).map(|_| PidController::new(100.0, 0.0, 20.0)).collect();
        Self {
            selection,
            force_pids,
            position_pids,
            force_desired: Vec6::zeros(),
            position_desired: Vec6::zeros(),
            output_limit: 1000.0,
        }
    }

    pub fn with_force_gains(mut self, kp: f64, ki: f64, kd: f64) -> Self {
        self.force_pids = (0..6).map(|_| PidController::new(kp, ki, kd)).collect();
        self
    }

    pub fn with_position_gains(mut self, kp: f64, ki: f64, kd: f64) -> Self {
        self.position_pids = (0..6).map(|_| PidController::new(kp, ki, kd)).collect();
        self
    }

    pub fn with_output_limit(mut self, limit: f64) -> Self {
        self.output_limit = limit.max(0.0);
        self
    }

    pub fn set_force_desired(&mut self, desired: Vec6) {
        self.force_desired = desired;
    }

    pub fn set_position_desired(&mut self, desired: Vec6) {
        self.position_desired = desired;
    }

    /// Compute control output given current measurements.
    pub fn compute(
        &mut self,
        current_position: &Vec6,
        current_force: &Vec6,
        dt: f64,
    ) -> ControlOutput {
        let mut command = Vec6::zeros();

        for i in 0..6 {
            if self.selection.is_force_axis(i) {
                // Force control: error = desired_force - measured_force
                let error = self.force_desired.data[i] - current_force.data[i];
                let output = self.force_pids[i].compute(error, dt);
                command.data[i] = output.clamp(-self.output_limit, self.output_limit);
            } else {
                // Position control: error = desired_position - current_position
                let error = self.position_desired.data[i] - current_position.data[i];
                let output = self.position_pids[i].compute(error, dt);
                command.data[i] = output.clamp(-self.output_limit, self.output_limit);
            }
        }

        let force_component = self.selection.apply_force(&command);
        let position_component = self.selection.apply_position(&command);

        ControlOutput {
            command,
            force_component,
            position_component,
        }
    }

    pub fn reset(&mut self) {
        for pid in &mut self.force_pids { pid.reset(); }
        for pid in &mut self.position_pids { pid.reset(); }
    }
}

impl fmt::Display for HybridController {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HybridController({}, limit={:.1})", self.selection, self.output_limit)
    }
}

/// Output from the hybrid controller.
#[derive(Clone, Debug)]
pub struct ControlOutput {
    pub command: Vec6,
    pub force_component: Vec6,
    pub position_component: Vec6,
}

impl ControlOutput {
    pub fn total_magnitude(&self) -> f64 {
        self.command.norm()
    }
}

impl fmt::Display for ControlOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Output(mag={:.3})", self.total_magnitude())
    }
}

// ── Task Frame ──

/// Task frame definition for expressing hybrid control in a task-relevant coordinate system.
#[derive(Clone, Debug)]
pub struct TaskFrame {
    pub rotation: [[f64; 3]; 3],
    pub origin: [f64; 3],
}

impl TaskFrame {
    pub fn identity() -> Self {
        Self {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            origin: [0.0, 0.0, 0.0],
        }
    }

    pub fn from_normal(normal: [f64; 3]) -> Self {
        // Build frame where z-axis aligns with normal
        let n = Self::normalize_vec(normal);
        let up = if n[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
        let x = Self::normalize_vec(Self::cross(up, n));
        let y = Self::cross(n, x);
        Self {
            rotation: [x, y, n],
            origin: [0.0, 0.0, 0.0],
        }
    }

    pub fn with_origin(mut self, origin: [f64; 3]) -> Self {
        self.origin = origin;
        self
    }

    fn normalize_vec(v: [f64; 3]) -> [f64; 3] {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if len < 1e-12 { return [0.0, 0.0, 1.0]; }
        [v[0] / len, v[1] / len, v[2] / len]
    }

    fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    /// Transform a wrench/twist from world frame to task frame.
    pub fn to_task_frame(&self, v: &Vec6) -> Vec6 {
        let r = &self.rotation;
        let lin = v.linear();
        let ang = v.angular();
        let tl = [
            r[0][0] * lin[0] + r[0][1] * lin[1] + r[0][2] * lin[2],
            r[1][0] * lin[0] + r[1][1] * lin[1] + r[1][2] * lin[2],
            r[2][0] * lin[0] + r[2][1] * lin[1] + r[2][2] * lin[2],
        ];
        let ta = [
            r[0][0] * ang[0] + r[0][1] * ang[1] + r[0][2] * ang[2],
            r[1][0] * ang[0] + r[1][1] * ang[1] + r[1][2] * ang[2],
            r[2][0] * ang[0] + r[2][1] * ang[1] + r[2][2] * ang[2],
        ];
        Vec6::new(tl[0], tl[1], tl[2], ta[0], ta[1], ta[2])
    }

    /// Transform from task frame back to world frame.
    pub fn to_world_frame(&self, v: &Vec6) -> Vec6 {
        let r = &self.rotation;
        let lin = v.linear();
        let ang = v.angular();
        // R^T * v (transpose for inverse rotation)
        let wl = [
            r[0][0] * lin[0] + r[1][0] * lin[1] + r[2][0] * lin[2],
            r[0][1] * lin[0] + r[1][1] * lin[1] + r[2][1] * lin[2],
            r[0][2] * lin[0] + r[1][2] * lin[1] + r[2][2] * lin[2],
        ];
        let wa = [
            r[0][0] * ang[0] + r[1][0] * ang[1] + r[2][0] * ang[2],
            r[0][1] * ang[0] + r[1][1] * ang[1] + r[2][1] * ang[2],
            r[0][2] * ang[0] + r[1][2] * ang[1] + r[2][2] * ang[2],
        ];
        Vec6::new(wl[0], wl[1], wl[2], wa[0], wa[1], wa[2])
    }
}

impl fmt::Display for TaskFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TaskFrame(origin=[{:.2},{:.2},{:.2}])",
            self.origin[0], self.origin[1], self.origin[2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec6_zeros() {
        let v = Vec6::zeros();
        assert!((v.norm() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec6_operations() {
        let a = Vec6::new(1.0, 2.0, 3.0, 0.0, 0.0, 0.0);
        let b = Vec6::new(0.5, 0.5, 0.5, 0.0, 0.0, 0.0);
        let c = a.add(&b);
        assert!((c.data[0] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_vec6_sub() {
        let a = Vec6::new(3.0, 2.0, 1.0, 0.0, 0.0, 0.0);
        let b = Vec6::new(1.0, 1.0, 1.0, 0.0, 0.0, 0.0);
        let c = a.sub(&b);
        assert!((c.data[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec6_scale() {
        let v = Vec6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let scaled = v.scale(2.0);
        assert!((scaled.data[2] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_selection_all_position() {
        let s = SelectionMatrix::all_position();
        assert_eq!(s.num_position_axes(), 6);
        assert_eq!(s.num_force_axes(), 0);
    }

    #[test]
    fn test_selection_all_force() {
        let s = SelectionMatrix::all_force();
        assert_eq!(s.num_force_axes(), 6);
    }

    #[test]
    fn test_selection_apply() {
        let s = SelectionMatrix::new(&[true, false, true, false, false, false]);
        let v = Vec6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let force = s.apply_force(&v);
        let pos = s.apply_position(&v);
        assert!((force.data[0] - 1.0).abs() < 1e-10);
        assert!((force.data[1] - 0.0).abs() < 1e-10);
        assert!((pos.data[0] - 0.0).abs() < 1e-10);
        assert!((pos.data[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_selection_complement() {
        let s = SelectionMatrix::all_position().with_force_axis(2);
        let c = s.complement();
        assert!(!c.is_force_axis(2));
        assert!(c.is_force_axis(0));
    }

    #[test]
    fn test_pid_proportional() {
        let mut pid = PidController::new(10.0, 0.0, 0.0);
        let out = pid.compute(1.0, 0.01);
        assert!((out - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_pid_integral() {
        let mut pid = PidController::new(0.0, 1.0, 0.0);
        pid.compute(1.0, 1.0);
        let out = pid.compute(1.0, 1.0);
        assert!((out - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_pid_reset() {
        let mut pid = PidController::new(1.0, 1.0, 1.0);
        pid.compute(5.0, 0.1);
        pid.reset();
        assert!((pid.integral - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_hybrid_controller_pure_position() {
        let sel = SelectionMatrix::all_position();
        let mut ctrl = HybridController::new(sel)
            .with_position_gains(100.0, 0.0, 0.0);
        ctrl.set_position_desired(Vec6::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0));
        let out = ctrl.compute(&Vec6::zeros(), &Vec6::zeros(), 0.01);
        assert!(out.command.data[0] > 0.0);
    }

    #[test]
    fn test_hybrid_controller_pure_force() {
        let sel = SelectionMatrix::all_force();
        let mut ctrl = HybridController::new(sel)
            .with_force_gains(1.0, 0.0, 0.0);
        ctrl.set_force_desired(Vec6::new(10.0, 0.0, 0.0, 0.0, 0.0, 0.0));
        let out = ctrl.compute(&Vec6::zeros(), &Vec6::zeros(), 0.01);
        assert!((out.command.data[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_hybrid_controller_mixed() {
        let sel = SelectionMatrix::new(&[false, false, true, false, false, false]);
        let mut ctrl = HybridController::new(sel);
        ctrl.set_position_desired(Vec6::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0));
        ctrl.set_force_desired(Vec6::new(0.0, 0.0, 5.0, 0.0, 0.0, 0.0));
        let out = ctrl.compute(&Vec6::zeros(), &Vec6::zeros(), 0.01);
        assert!(out.position_component.data[0].abs() > 0.0);
        assert!(out.force_component.data[2].abs() > 0.0);
    }

    #[test]
    fn test_task_frame_identity() {
        let tf = TaskFrame::identity();
        let v = Vec6::new(1.0, 2.0, 3.0, 0.0, 0.0, 0.0);
        let transformed = tf.to_task_frame(&v);
        assert!((transformed.data[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_task_frame_roundtrip() {
        let tf = TaskFrame::from_normal([0.0, 0.0, 1.0]);
        let v = Vec6::new(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
        let task = tf.to_task_frame(&v);
        let world = tf.to_world_frame(&task);
        for i in 0..6 {
            assert!((world.data[i] - v.data[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_control_output_magnitude() {
        let out = ControlOutput {
            command: Vec6::new(3.0, 4.0, 0.0, 0.0, 0.0, 0.0),
            force_component: Vec6::zeros(),
            position_component: Vec6::zeros(),
        };
        assert!((out.total_magnitude() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_display_formats() {
        let s = SelectionMatrix::new(&[true, false, true, false, true, false]);
        let display = format!("{s}");
        assert!(display.contains("F"));
        assert!(display.contains("P"));

        let ctrl = HybridController::new(SelectionMatrix::all_position());
        let display = format!("{ctrl}");
        assert!(display.contains("HybridController"));
    }
}
