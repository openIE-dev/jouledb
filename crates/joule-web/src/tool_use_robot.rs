//! Tool Use — Tool-tip calibration, tool frame transforms, tool change
//! sequences, and payload management for robotic tool handling.
//!
//! Implements rigid-body transforms for tool frames, calibration via
//! multiple-pose measurements, tool-change state machines, and payload
//! tracking with centre-of-mass computation.
//! All algorithms are std-only, using `f64` throughout.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Tool-use errors.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolError {
    /// Invalid calibration data.
    InvalidCalibration(String),
    /// Tool not found.
    ToolNotFound(String),
    /// Invalid tool change transition.
    InvalidTransition(String),
    /// Payload exceeds limit.
    PayloadExceeded { current: f64, limit: f64 },
    /// Singular matrix during calibration.
    SingularMatrix,
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCalibration(m) => write!(f, "invalid calibration: {m}"),
            Self::ToolNotFound(id) => write!(f, "tool not found: {id}"),
            Self::InvalidTransition(m) => write!(f, "invalid transition: {m}"),
            Self::PayloadExceeded { current, limit } => {
                write!(f, "payload {current:.2}kg exceeds limit {limit:.2}kg")
            }
            Self::SingularMatrix => write!(f, "singular matrix during calibration"),
        }
    }
}

impl std::error::Error for ToolError {}

// ── 3D Vector ───────────────────────────────────────────────────

/// Minimal 3D vector.
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

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn normalized(self) -> Option<Self> {
        let n = self.norm();
        if n < 1e-12 { None } else { Some(Self { x: self.x / n, y: self.y / n, z: self.z / n }) }
    }

    pub fn scale(self, s: f64) -> Self {
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

// ── 3x3 Rotation Matrix ────────────────────────────────────────

/// A 3x3 rotation matrix stored row-major.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotMat {
    pub rows: [[f64; 3]; 3],
}

impl RotMat {
    pub fn identity() -> Self {
        Self {
            rows: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Rotation about the X axis.
    pub fn rot_x(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            rows: [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        }
    }

    /// Rotation about the Y axis.
    pub fn rot_y(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            rows: [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        }
    }

    /// Rotation about the Z axis.
    pub fn rot_z(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            rows: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Multiply rotation by a vector.
    pub fn apply(&self, v: Vec3) -> Vec3 {
        Vec3 {
            x: self.rows[0][0] * v.x + self.rows[0][1] * v.y + self.rows[0][2] * v.z,
            y: self.rows[1][0] * v.x + self.rows[1][1] * v.y + self.rows[1][2] * v.z,
            z: self.rows[2][0] * v.x + self.rows[2][1] * v.y + self.rows[2][2] * v.z,
        }
    }

    /// Multiply two rotation matrices.
    pub fn mul(&self, other: &RotMat) -> RotMat {
        let mut rows = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                rows[i][j] = self.rows[i][0] * other.rows[0][j]
                    + self.rows[i][1] * other.rows[1][j]
                    + self.rows[i][2] * other.rows[2][j];
            }
        }
        RotMat { rows }
    }

    /// Transpose (inverse for rotation matrices).
    pub fn transpose(&self) -> Self {
        let mut rows = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                rows[i][j] = self.rows[j][i];
            }
        }
        RotMat { rows }
    }

    /// Determinant (should be 1.0 for valid rotation).
    pub fn det(&self) -> f64 {
        let r = &self.rows;
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    }
}

impl fmt::Display for RotMat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "R[{:.3},{:.3},{:.3}|{:.3},{:.3},{:.3}|{:.3},{:.3},{:.3}]",
            self.rows[0][0], self.rows[0][1], self.rows[0][2],
            self.rows[1][0], self.rows[1][1], self.rows[1][2],
            self.rows[2][0], self.rows[2][1], self.rows[2][2],
        )
    }
}

// ── Rigid Transform ─────────────────────────────────────────────

/// A rigid-body transform (rotation + translation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    pub rotation: RotMat,
    pub translation: Vec3,
}

impl Transform {
    pub fn identity() -> Self {
        Self { rotation: RotMat::identity(), translation: Vec3::zero() }
    }

    pub fn new(rotation: RotMat, translation: Vec3) -> Self {
        Self { rotation, translation }
    }

    /// Apply this transform to a point.
    pub fn apply(&self, p: Vec3) -> Vec3 {
        self.rotation.apply(p).add(self.translation)
    }

    /// Compose two transforms: self * other.
    pub fn compose(&self, other: &Transform) -> Transform {
        Transform {
            rotation: self.rotation.mul(&other.rotation),
            translation: self.rotation.apply(other.translation).add(self.translation),
        }
    }

    /// Inverse transform.
    pub fn inverse(&self) -> Transform {
        let r_inv = self.rotation.transpose();
        Transform {
            rotation: r_inv,
            translation: r_inv.apply(self.translation).scale(-1.0),
        }
    }
}

impl fmt::Display for Transform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "T(rot={}, t={})", self.rotation, self.translation)
    }
}

// ── Tool Definition ─────────────────────────────────────────────

/// Definition of a robotic tool with geometry and payload.
#[derive(Debug, Clone)]
pub struct ToolDef {
    /// Unique tool identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Tool mass (kg).
    pub mass: f64,
    /// Centre of mass in tool frame.
    pub com: Vec3,
    /// Tool-tip transform relative to the tool flange.
    pub tip_transform: Transform,
    /// Maximum payload capacity (kg).
    pub max_payload: f64,
}

impl ToolDef {
    pub fn new(id: &str, name: &str, mass: f64) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            mass,
            com: Vec3::zero(),
            tip_transform: Transform::identity(),
            max_payload: 5.0,
        }
    }

    pub fn with_com(mut self, com: Vec3) -> Self {
        self.com = com;
        self
    }

    pub fn with_tip_transform(mut self, tf: Transform) -> Self {
        self.tip_transform = tf;
        self
    }

    pub fn with_max_payload(mut self, p: f64) -> Self {
        self.max_payload = p;
        self
    }

    /// Total inertia at the flange (simplified as point mass).
    pub fn flange_inertia(&self) -> f64 {
        let r = self.com.norm();
        self.mass * r * r
    }
}

impl fmt::Display for ToolDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tool({}, \"{}\", {:.2}kg, max_payload={:.2}kg)",
            self.id, self.name, self.mass, self.max_payload
        )
    }
}

// ── Tool-Tip Calibration ────────────────────────────────────────

/// A calibration measurement: flange pose and known tip position.
#[derive(Debug, Clone, Copy)]
pub struct CalibrationSample {
    pub flange_transform: Transform,
    pub measured_tip: Vec3,
}

/// Calibrate the tool-tip transform from multiple flange poses where the
/// tool tip touches a known fixed point. Uses the least-squares
/// formulation: minimise sum of ||R_i * p_tip + t_i - p_fixed||^2.
pub fn calibrate_tool_tip(
    samples: &[CalibrationSample],
) -> Result<Vec3, ToolError> {
    if samples.len() < 3 {
        return Err(ToolError::InvalidCalibration(
            "need >= 3 calibration samples".into(),
        ));
    }
    // Use the difference method: for pairs (i, i+1):
    //   (R_i - R_{i+1}) * p_tip = t_{i+1} - t_i
    // Accumulate into A*p = b, solve via normal equations.
    let n = samples.len() - 1;
    let mut ata = [[0.0f64; 3]; 3];
    let mut atb = [0.0f64; 3];

    for k in 0..n {
        let r1 = &samples[k].flange_transform.rotation;
        let r2 = &samples[k + 1].flange_transform.rotation;
        let t1 = samples[k].flange_transform.translation;
        let t2 = samples[k + 1].flange_transform.translation;

        // Row of A: (R1 - R2), b: (t2 - t1)
        for row in 0..3 {
            let a_row = [
                r1.rows[row][0] - r2.rows[row][0],
                r1.rows[row][1] - r2.rows[row][1],
                r1.rows[row][2] - r2.rows[row][2],
            ];
            let b_val = [t2.x - t1.x, t2.y - t1.y, t2.z - t1.z][row];

            for i in 0..3 {
                atb[i] += a_row[i] * b_val;
                for j in 0..3 {
                    ata[i][j] += a_row[i] * a_row[j];
                }
            }
        }
    }

    // Solve 3x3 system via Cramer's rule
    let det = det_3x3(&ata);
    if det.abs() < 1e-12 {
        return Err(ToolError::SingularMatrix);
    }

    let px = det_3x3_col(&ata, &atb, 0) / det;
    let py = det_3x3_col(&ata, &atb, 1) / det;
    let pz = det_3x3_col(&ata, &atb, 2) / det;

    Ok(Vec3::new(px, py, pz))
}

/// Determinant of a 3x3 matrix.
fn det_3x3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Determinant with column `col` replaced by vector `b`.
fn det_3x3_col(m: &[[f64; 3]; 3], b: &[f64; 3], col: usize) -> f64 {
    let mut mc = *m;
    for row in 0..3 {
        mc[row][col] = b[row];
    }
    det_3x3(&mc)
}

// ── Tool Change State Machine ───────────────────────────────────

/// Phases of a tool change operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolChangePhase {
    Idle,
    ApproachRack,
    AlignDock,
    Undock,
    Retract,
    MoveToPick,
    AlignPick,
    Dock,
    Verify,
    Ready,
}

impl ToolChangePhase {
    pub fn successors(self) -> &'static [ToolChangePhase] {
        match self {
            Self::Idle => &[Self::ApproachRack],
            Self::ApproachRack => &[Self::AlignDock],
            Self::AlignDock => &[Self::Undock],
            Self::Undock => &[Self::Retract],
            Self::Retract => &[Self::MoveToPick],
            Self::MoveToPick => &[Self::AlignPick],
            Self::AlignPick => &[Self::Dock],
            Self::Dock => &[Self::Verify],
            Self::Verify => &[Self::Ready],
            Self::Ready => &[Self::Idle],
        }
    }
}

impl fmt::Display for ToolChangePhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Idle => "Idle",
            Self::ApproachRack => "ApproachRack",
            Self::AlignDock => "AlignDock",
            Self::Undock => "Undock",
            Self::Retract => "Retract",
            Self::MoveToPick => "MoveToPick",
            Self::AlignPick => "AlignPick",
            Self::Dock => "Dock",
            Self::Verify => "Verify",
            Self::Ready => "Ready",
        };
        write!(f, "{name}")
    }
}

/// Tool change controller.
#[derive(Debug, Clone)]
pub struct ToolChanger {
    phase: ToolChangePhase,
    current_tool: Option<String>,
    target_tool: Option<String>,
    tools: Vec<ToolDef>,
    change_history: Vec<(String, String)>,
}

impl ToolChanger {
    pub fn new(tools: Vec<ToolDef>) -> Self {
        Self {
            phase: ToolChangePhase::Idle,
            current_tool: None,
            target_tool: None,
            tools,
            change_history: Vec::new(),
        }
    }

    pub fn with_current_tool(mut self, tool_id: &str) -> Self {
        self.current_tool = Some(tool_id.to_string());
        self
    }

    pub fn phase(&self) -> ToolChangePhase {
        self.phase
    }

    pub fn current_tool(&self) -> Option<&str> {
        self.current_tool.as_deref()
    }

    /// Initiate a tool change to the specified target tool.
    pub fn initiate_change(&mut self, target_id: &str) -> Result<(), ToolError> {
        if !self.tools.iter().any(|t| t.id == target_id) {
            return Err(ToolError::ToolNotFound(target_id.to_string()));
        }
        self.target_tool = Some(target_id.to_string());
        Ok(())
    }

    /// Advance to the next phase.
    pub fn advance(&mut self) -> Result<ToolChangePhase, ToolError> {
        let succs = self.phase.successors();
        if succs.is_empty() {
            return Err(ToolError::InvalidTransition(
                format!("no successor for {}", self.phase),
            ));
        }
        let next = succs[0];

        match next {
            ToolChangePhase::Undock => {
                // Release current tool
                let old = self.current_tool.take();
                if let (Some(old_id), Some(target_id)) = (old, &self.target_tool) {
                    self.change_history.push((old_id, target_id.clone()));
                }
            }
            ToolChangePhase::Dock => {
                // Attach target tool
                self.current_tool = self.target_tool.clone();
            }
            ToolChangePhase::Ready => {
                self.target_tool = None;
            }
            _ => {}
        }
        self.phase = next;
        Ok(next)
    }

    /// Run the full tool change sequence.
    pub fn run_to_completion(&mut self) -> Result<Vec<ToolChangePhase>, ToolError> {
        let mut history = vec![self.phase];
        while self.phase != ToolChangePhase::Ready {
            let next = self.advance()?;
            history.push(next);
        }
        Ok(history)
    }

    /// Get a tool definition by id.
    pub fn get_tool(&self, id: &str) -> Option<&ToolDef> {
        self.tools.iter().find(|t| t.id == id)
    }

    /// Number of tool changes performed.
    pub fn change_count(&self) -> usize {
        self.change_history.len()
    }
}

impl fmt::Display for ToolChanger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ToolChanger(phase={}, current={:?}, tools={})",
            self.phase,
            self.current_tool,
            self.tools.len()
        )
    }
}

// ── Payload Manager ─────────────────────────────────────────────

/// Manages payload tracking for a robot with a tool.
#[derive(Debug, Clone)]
pub struct PayloadManager {
    /// Maximum robot payload (kg).
    pub robot_max_payload: f64,
    /// Current tool mass (kg).
    pub tool_mass: f64,
    /// Current workpiece mass (kg).
    pub workpiece_mass: f64,
    /// Tool centre of mass.
    pub tool_com: Vec3,
    /// Workpiece centre of mass (in tool frame).
    pub workpiece_com: Vec3,
}

impl PayloadManager {
    pub fn new(robot_max_payload: f64) -> Self {
        Self {
            robot_max_payload,
            tool_mass: 0.0,
            workpiece_mass: 0.0,
            tool_com: Vec3::zero(),
            workpiece_com: Vec3::zero(),
        }
    }

    pub fn with_tool(mut self, mass: f64, com: Vec3) -> Self {
        self.tool_mass = mass;
        self.tool_com = com;
        self
    }

    pub fn with_workpiece(mut self, mass: f64, com: Vec3) -> Self {
        self.workpiece_mass = mass;
        self.workpiece_com = com;
        self
    }

    /// Total payload (tool + workpiece).
    pub fn total_payload(&self) -> f64 {
        self.tool_mass + self.workpiece_mass
    }

    /// Check if payload is within limits.
    pub fn check_payload(&self) -> Result<(), ToolError> {
        let total = self.total_payload();
        if total > self.robot_max_payload {
            Err(ToolError::PayloadExceeded {
                current: total,
                limit: self.robot_max_payload,
            })
        } else {
            Ok(())
        }
    }

    /// Combined centre of mass at the flange.
    pub fn combined_com(&self) -> Vec3 {
        let total = self.total_payload();
        if total < 1e-12 {
            return Vec3::zero();
        }
        let tool_moment = self.tool_com.scale(self.tool_mass);
        let wp_moment = self.workpiece_com.scale(self.workpiece_mass);
        tool_moment.add(wp_moment).scale(1.0 / total)
    }

    /// Payload utilisation ratio (0..1).
    pub fn utilisation(&self) -> f64 {
        if self.robot_max_payload <= 0.0 {
            return 1.0;
        }
        self.total_payload() / self.robot_max_payload
    }
}

impl fmt::Display for PayloadManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Payload(tool={:.2}kg, wp={:.2}kg, total={:.2}/{:.2}kg)",
            self.tool_mass,
            self.workpiece_mass,
            self.total_payload(),
            self.robot_max_payload
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rot_identity() {
        let r = RotMat::identity();
        let v = Vec3::new(1.0, 2.0, 3.0);
        let rv = r.apply(v);
        assert!((rv.x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rot_z_90() {
        let r = RotMat::rot_z(std::f64::consts::FRAC_PI_2);
        let v = Vec3::new(1.0, 0.0, 0.0);
        let rv = r.apply(v);
        assert!(rv.x.abs() < 1e-10);
        assert!((rv.y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rot_det() {
        let r = RotMat::rot_x(0.5).mul(&RotMat::rot_y(0.3));
        assert!((r.det() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rot_transpose_inverse() {
        let r = RotMat::rot_z(0.7);
        let rt = r.transpose();
        let product = r.mul(&rt);
        assert!((product.rows[0][0] - 1.0).abs() < 1e-10);
        assert!(product.rows[0][1].abs() < 1e-10);
    }

    #[test]
    fn test_transform_identity() {
        let t = Transform::identity();
        let p = Vec3::new(1.0, 2.0, 3.0);
        let tp = t.apply(p);
        assert!((tp.x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_compose() {
        let t1 = Transform::new(RotMat::identity(), Vec3::new(1.0, 0.0, 0.0));
        let t2 = Transform::new(RotMat::identity(), Vec3::new(0.0, 2.0, 0.0));
        let t12 = t1.compose(&t2);
        let p = Vec3::zero();
        let tp = t12.apply(p);
        assert!((tp.x - 1.0).abs() < 1e-10);
        assert!((tp.y - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_inverse() {
        let t = Transform::new(
            RotMat::rot_z(0.5),
            Vec3::new(1.0, 2.0, 3.0),
        );
        let t_inv = t.inverse();
        let p = Vec3::new(5.0, 6.0, 7.0);
        let roundtrip = t_inv.apply(t.apply(p));
        assert!((roundtrip.x - p.x).abs() < 1e-10);
        assert!((roundtrip.y - p.y).abs() < 1e-10);
        assert!((roundtrip.z - p.z).abs() < 1e-10);
    }

    #[test]
    fn test_tool_def_creation() {
        let tool = ToolDef::new("gripper", "Parallel Gripper", 1.5);
        assert_eq!(tool.id, "gripper");
    }

    #[test]
    fn test_tool_def_builder() {
        let tool = ToolDef::new("drill", "Drill", 2.0)
            .with_com(Vec3::new(0.0, 0.0, 0.05))
            .with_max_payload(3.0);
        assert!((tool.max_payload - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_tool_flange_inertia() {
        let tool = ToolDef::new("t1", "T1", 1.0)
            .with_com(Vec3::new(0.0, 0.0, 0.1));
        let inertia = tool.flange_inertia();
        assert!((inertia - 0.01).abs() < 1e-10); // 1.0 * 0.1^2
    }

    #[test]
    fn test_calibration_insufficient_samples() {
        let samples = vec![
            CalibrationSample {
                flange_transform: Transform::identity(),
                measured_tip: Vec3::zero(),
            },
        ];
        assert!(calibrate_tool_tip(&samples).is_err());
    }

    #[test]
    fn test_tool_change_sequence() {
        let tools = vec![
            ToolDef::new("grip", "Gripper", 1.0),
            ToolDef::new("drill", "Drill", 2.0),
        ];
        let mut changer = ToolChanger::new(tools).with_current_tool("grip");
        changer.initiate_change("drill").unwrap();
        let history = changer.run_to_completion().unwrap();
        assert_eq!(*history.last().unwrap(), ToolChangePhase::Ready);
        assert_eq!(changer.current_tool(), Some("drill"));
    }

    #[test]
    fn test_tool_change_not_found() {
        let tools = vec![ToolDef::new("grip", "Gripper", 1.0)];
        let mut changer = ToolChanger::new(tools);
        assert!(changer.initiate_change("nonexistent").is_err());
    }

    #[test]
    fn test_tool_change_count() {
        let tools = vec![
            ToolDef::new("a", "A", 1.0),
            ToolDef::new("b", "B", 1.0),
        ];
        let mut changer = ToolChanger::new(tools).with_current_tool("a");
        changer.initiate_change("b").unwrap();
        changer.run_to_completion().unwrap();
        assert_eq!(changer.change_count(), 1);
    }

    #[test]
    fn test_payload_within_limit() {
        let pm = PayloadManager::new(10.0)
            .with_tool(2.0, Vec3::zero())
            .with_workpiece(3.0, Vec3::zero());
        assert!(pm.check_payload().is_ok());
    }

    #[test]
    fn test_payload_exceeded() {
        let pm = PayloadManager::new(5.0)
            .with_tool(3.0, Vec3::zero())
            .with_workpiece(4.0, Vec3::zero());
        assert!(pm.check_payload().is_err());
    }

    #[test]
    fn test_combined_com() {
        let pm = PayloadManager::new(10.0)
            .with_tool(1.0, Vec3::new(0.0, 0.0, 0.05))
            .with_workpiece(1.0, Vec3::new(0.0, 0.0, 0.15));
        let com = pm.combined_com();
        assert!((com.z - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_payload_utilisation() {
        let pm = PayloadManager::new(10.0)
            .with_tool(2.0, Vec3::zero())
            .with_workpiece(3.0, Vec3::zero());
        assert!((pm.utilisation() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_payload_display() {
        let pm = PayloadManager::new(10.0);
        let s = format!("{pm}");
        assert!(s.contains("Payload"));
    }

    #[test]
    fn test_tool_changer_display() {
        let changer = ToolChanger::new(vec![]);
        let s = format!("{changer}");
        assert!(s.contains("ToolChanger"));
    }
}
